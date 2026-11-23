//! The `/v1/realtime` WebSocket transcription session of the speech
//! worker: the client events it parses, the server events it answers
//! with, the model-free [`Session`] state, and the session loop that
//! bounds idle time, outbound writes and a commit in flight. The route
//! handler [`realtime`] is mounted by `server::router`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use super::prompt::PromptError;
use super::server::{
    run_locked, Error, Guard, Request, Shared, CLIP_SECONDS, COMMIT_EMPTY, DEFAULT_LANGUAGE,
    FRAMING_ALLOWANCE, SAMPLE_RATE,
};
use super::wav;

/// Most `/v1/realtime` sessions open at once; a further one gets an
/// error event and is closed.
pub const MAX_SESSIONS: usize = 4;

/// Samples a realtime session buffers at most between commits: one clip,
/// so a commit never splits.
pub const BUFFER_SAMPLES: usize = CLIP_SECONDS * SAMPLE_RATE;

/// Seconds a realtime session may go without a client event before the
/// worker closes it, so idle connections cannot hold the
/// [`MAX_SESSIONS`] slots. Pings and pongs do not count as events. A
/// session with a commit in flight is not closed as idle until the
/// commit finishes, its reply is what the client waits for; the clock
/// restarts when that reply is sent, and the events queued during the
/// commit are handled under that one deadline, not as new activity.
pub const IDLE_SECONDS: u64 = 60;

/// Seconds the worker waits for one outbound frame of a realtime session
/// to be written before it drops the socket, so a client that stops
/// reading cannot hold its session slot once the socket buffers fill.
pub const WRITE_SECONDS: u64 = 10;

/// Seconds a realtime commit may spend queued for the worker lock and
/// decoding before the worker gives up on it. The 30 s buffer decodes in
/// well under 10 s on CPU, so two minutes is a stalled backend, not a
/// slow host: the session is told so and closed, and the compute is
/// aborted at its next check.
pub const COMMIT_SECONDS: u64 = 120;

/// A client event of the `/v1/realtime` transcription session, the subset
/// llama-server's `server-realtime.cpp` handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientEvent {
    /// `session.update`: `session.audio.input.format.rate` and
    /// `session.audio.input.transcription.language`, each when present.
    SessionUpdate {
        rate: Option<u32>,
        language: Option<String>,
    },
    /// `input_audio_buffer.append`: the decoded PCM16 samples of `audio`.
    Append(Vec<i16>),
    /// `input_audio_buffer.commit`.
    Commit,
    /// `input_audio_buffer.clear`.
    Clear,
    /// Any other `type`, carried for the error message.
    Unknown(String),
}

/// The one `audio.input.format.type` a session accepts, llama-server's.
const PCM_FORMAT: &str = "audio/pcm";

/// The `key` member of `v`, an object: a missing one reads as an empty
/// object, so its members are omitted; a value of any other type is
/// refused, not read as omitted. `v["key"]["member"]` alone would turn
/// the string `"audio/pcmu"` under `format` into no format at all.
fn object<'a>(v: &'a Value, key: &str) -> Result<&'a Value, Error> {
    match &v[key] {
        Value::Null | Value::Object(_) => Ok(&v[key]),
        other => Err(Error::Invalid(format!(
            "{key} must be an object, got {other}"
        ))),
    }
}

/// Longest base64 text that can decode to `samples` PCM16 samples: two
/// bytes each, every three bytes four characters, padded.
const fn base64_len(samples: usize) -> usize {
    (samples * 2).div_ceil(3) * 4
}

impl ClientEvent {
    /// Parses one Text frame. `room` is the session's remaining buffer
    /// capacity in samples: an `audio` text longer than its base64 form
    /// is refused before decoding, so a frame costs at most one buffer.
    pub fn parse(text: &str, room: usize) -> Result<ClientEvent, Error> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| Error::Invalid(format!("invalid JSON: {e}")))?;
        let Some(ty) = v["type"].as_str() else {
            return Err(Error::Invalid("the event has no string \"type\"".into()));
        };
        match ty {
            "session.update" => {
                let input = object(object(object(&v, "session")?, "audio")?, "input")?;
                let format = object(input, "format")?;
                match &format["type"] {
                    Value::Null => {}
                    Value::String(t) if t == PCM_FORMAT => {}
                    other => {
                        return Err(Error::Invalid(format!(
                            "session.audio.input.format.type {other}: the worker accepts {PCM_FORMAT:?} only"
                        )))
                    }
                }
                let rate =
                    match &format["rate"] {
                        Value::Null => None,
                        r => Some(r.as_u64().and_then(|r| u32::try_from(r).ok()).ok_or_else(
                            || {
                                Error::Invalid(
                                    "session.audio.input.format.rate must be an integer".into(),
                                )
                            },
                        )?),
                    };
                let language = match &object(input, "transcription")?["language"] {
                    Value::Null => None,
                    Value::String(l) => Some(l.clone()),
                    _ => {
                        return Err(Error::Invalid(
                            "session.audio.input.transcription.language must be a string".into(),
                        ))
                    }
                };
                Ok(ClientEvent::SessionUpdate { rate, language })
            }
            "input_audio_buffer.append" => {
                let Some(audio) = v["audio"].as_str() else {
                    return Err(Error::Invalid("\"audio\" must be a base64 string".into()));
                };
                if audio.len() > base64_len(room) {
                    return Err(Error::TooLarge(format!(
                        "the audio buffer holds at most {BUFFER_SAMPLES} samples"
                    )));
                }
                use base64::Engine as _;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(audio)
                    .map_err(|e| Error::Invalid(format!("\"audio\" is not valid base64: {e}")))?;
                wav::decode_pcm16(&bytes)
                    .map(ClientEvent::Append)
                    .map_err(|_| {
                        Error::Invalid("\"audio\" must be PCM16: an even number of bytes".into())
                    })
            }
            "input_audio_buffer.commit" => Ok(ClientEvent::Commit),
            "input_audio_buffer.clear" => Ok(ClientEvent::Clear),
            other => Ok(ClientEvent::Unknown(other.to_string())),
        }
    }
}

/// The model-free state of one realtime session: the audio buffered since
/// the last commit, the settings a `session.update` may change, and the
/// item counter.
pub struct Session {
    pub buffer: Vec<i16>,
    /// Validated against the model's languages; `en` to begin with, or
    /// the model's first language when it lacks `en` ([`Session::open`]).
    pub language: String,
    /// Always [`SAMPLE_RATE`]; `update` refuses any other.
    pub rate: u32,
    pub next_item: u64,
}

impl Default for Session {
    /// An empty buffer, [`DEFAULT_LANGUAGE`], [`SAMPLE_RATE`], item 0.
    fn default() -> Session {
        Session {
            buffer: Vec::new(),
            language: DEFAULT_LANGUAGE.to_string(),
            rate: SAMPLE_RATE as u32,
            next_item: 0,
        }
    }
}

impl Session {
    /// A new session over a model speaking `languages`: its language is
    /// [`DEFAULT_LANGUAGE`] when the model has it, else the model's first,
    /// so `session.created` always names one the client may keep. A model
    /// with no language gets the error before `session.created`.
    pub fn open(languages: &[String]) -> Result<Session, Error> {
        let mut session = Session::default();
        if !languages.contains(&session.language) {
            let Some(first) = languages.first() else {
                return Err(Error::Invalid(
                    PromptError::UnsupportedLanguage(session.language).to_string(),
                ));
            };
            session.language = first.clone();
        }
        Ok(session)
    }

    /// Samples the buffer still accepts.
    pub fn room(&self) -> usize {
        BUFFER_SAMPLES - self.buffer.len()
    }

    /// Appends `samples`, or refuses them whole when they would cross
    /// [`BUFFER_SAMPLES`].
    pub fn append(&mut self, samples: &[i16]) -> Result<(), Error> {
        if samples.len() > self.room() {
            return Err(Error::TooLarge(format!(
                "the audio buffer holds at most {BUFFER_SAMPLES} samples"
            )));
        }
        self.buffer.extend_from_slice(samples);
        Ok(())
    }

    /// Takes the buffer as a request with the next item id, `item_<n>`.
    pub fn commit(&mut self) -> Result<(Request, String), Error> {
        if self.buffer.is_empty() {
            return Err(Error::CommitEmpty);
        }
        let item = format!("item_{}", self.next_item);
        self.next_item += 1;
        let req = Request {
            samples: std::mem::take(&mut self.buffer),
            language: self.language.clone(),
        };
        Ok((req, item))
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Applies a `session.update`. A rate other than [`SAMPLE_RATE`] or a
    /// language outside `languages` is refused and nothing changes; an
    /// empty language means [`DEFAULT_LANGUAGE`], as on the one-shot route.
    pub fn update(
        &mut self,
        rate: Option<u32>,
        language: Option<String>,
        languages: &[String],
    ) -> Result<(), Error> {
        if let Some(rate) = rate {
            if rate as usize != SAMPLE_RATE {
                return Err(Error::Invalid(format!(
                    "audio.input.format.rate {rate}: the worker accepts {SAMPLE_RATE} Hz PCM16 only"
                )));
            }
        }
        if let Some(language) = language {
            let language = if language.is_empty() {
                DEFAULT_LANGUAGE.to_owned()
            } else {
                language
            };
            if !languages.contains(&language) {
                return Err(Error::Invalid(
                    PromptError::UnsupportedLanguage(language).to_string(),
                ));
            }
            self.language = language;
        }
        Ok(())
    }
}

/// A fresh `event_<n>` for a server event, unique within the process.
fn event_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!("event_{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// The `error` event for `e`, in llama-server's shape, or `None` for a
/// cancellation: the client is gone and nothing is sent.
pub fn error_event(e: &Error) -> Option<Value> {
    let (ty, code, message) = match e {
        Error::Invalid(m) | Error::TooLarge(m) | Error::UnsupportedMedia(m) => {
            ("invalid_request_error", None, m.clone())
        }
        Error::Failed(m) => ("server_error", None, m.clone()),
        Error::CommitEmpty => (
            "input_audio_buffer_commit_empty",
            None,
            COMMIT_EMPTY.to_string(),
        ),
        Error::CommitTimeout(limit) => (
            "server_error",
            Some("commit_timeout"),
            Error::commit_timeout_message(*limit),
        ),
        Error::Cancelled => return None,
    };
    Some(json!({
        "type": "error",
        "event_id": event_id(),
        "error": {"type": ty, "code": code, "message": message, "param": null},
    }))
}

/// The `session` object of `session.created` and `session.updated`. The
/// daemon's relay rewrites `model` to the name the client asked for.
fn session_object(id: &str, model: &str, session: &Session) -> Value {
    json!({
        "id": id,
        "object": "realtime.transcription_session",
        "type": "transcription",
        "model": model,
        "audio": {
            "input": {
                "format": {"type": PCM_FORMAT, "rate": session.rate},
                "transcription": {"model": model, "language": session.language},
                "turn_detection": null,
            }
        },
    })
}

pub fn session_created(id: &str, model: &str, session: &Session) -> Value {
    json!({
        "type": "session.created",
        "event_id": event_id(),
        "session": session_object(id, model, session),
    })
}

pub fn session_updated(id: &str, model: &str, session: &Session) -> Value {
    json!({
        "type": "session.updated",
        "event_id": event_id(),
        "session": session_object(id, model, session),
    })
}

pub fn buffer_committed(previous_item_id: Option<&str>, item_id: &str) -> Value {
    json!({
        "type": "input_audio_buffer.committed",
        "event_id": event_id(),
        "previous_item_id": previous_item_id,
        "item_id": item_id,
    })
}

pub fn transcription_completed(item_id: &str, transcript: &str) -> Value {
    json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    })
}

pub fn buffer_cleared() -> Value {
    json!({"type": "input_audio_buffer.cleared", "event_id": event_id()})
}

/// Text bytes a session queues while a commit runs: the client may keep
/// appending the next item, and the queue is drained in order once the
/// decode ends. One buffer's base64 plus framing; a frame past it is
/// refused with an error event. Every frame is charged its length plus
/// [`QUEUE_EVENT_OVERHEAD`], so the entry count is bounded too.
const QUEUE_BYTES: usize = base64_len(BUFFER_SAMPLES) + FRAMING_ALLOWANCE;

/// Bytes charged per queued frame on top of its text: the `String` header
/// and its `VecDeque` slot, and the error event an unparsable frame costs
/// when drained. Without it an empty frame would be free and a flood of
/// them would grow the queue past [`QUEUE_BYTES`].
const QUEUE_EVENT_OVERHEAD: usize = 64;

/// Admits `text` to the mid-commit queue, or returns the error event that
/// refuses it. `queued_bytes` is the charge of what `queued` holds.
fn queue_event(
    queued: &mut VecDeque<String>,
    queued_bytes: &mut usize,
    text: String,
) -> Result<(), Error> {
    let cost = text.len() + QUEUE_EVENT_OVERHEAD;
    if *queued_bytes + cost > QUEUE_BYTES {
        return Err(Error::TooLarge(format!(
            "at most {QUEUE_BYTES} bytes of events are queued during a commit"
        )));
    }
    *queued_bytes += cost;
    queued.push_back(text);
    Ok(())
}

/// GET /v1/realtime: the WebSocket transcription session. The daemon's
/// relay dials it without the `model` query, so none is read, and a
/// request carrying an `Origin` is refused with 403 before the upgrade
/// ([`direct_hit`]).
pub(crate) async fn realtime(
    State(s): State<Shared>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    upgrade_or_refuse(&headers, ws, move |socket| run_realtime(s, socket))
}

/// The one step between the request and the session: a direct hit
/// ([`direct_hit`]) is refused with 403, anything else is upgraded with
/// the frame cap ([`capped`]) and handed to `session`. Shared with the
/// tests so they exercise the handler's own path.
fn upgrade_or_refuse<F, Fut>(headers: &HeaderMap, ws: WebSocketUpgrade, session: F) -> Response
where
    F: FnOnce(WebSocket) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    if direct_hit(headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    capped(ws).on_upgrade(session)
}

/// Whether a request reached the worker port from a browser: the worker
/// is only reached by the daemon's relay, which sends no `Origin` and
/// applies the origin policy itself, so an `Origin` here means a page
/// hit the worker directly.
fn direct_hit(headers: &HeaderMap) -> bool {
    headers.contains_key(header::ORIGIN)
}

/// Caps a frame and a message at [`QUEUE_BYTES`], so no text larger than
/// the queue admits is read into memory or parsed. A frame past the cap
/// surfaces as `Err` from `recv()` and the session ends.
fn capped(ws: WebSocketUpgrade) -> WebSocketUpgrade {
    ws.max_message_size(QUEUE_BYTES).max_frame_size(QUEUE_BYTES)
}

/// A session's socket, with the deadline every outbound frame gets. A
/// write that misses it fails like a gone client: the caller returns and
/// the socket is dropped, so a client that stops reading cannot hold the
/// session task, and its slot, once the socket buffers fill.
struct Conn {
    socket: WebSocket,
    /// The write deadline; [`WRITE_SECONDS`] outside tests.
    write: std::time::Duration,
}

impl Conn {
    async fn recv(&mut self) -> Option<Result<Message, axum::Error>> {
        self.socket.recv().await
    }

    /// Writes one frame within the deadline; `Err` means the client is
    /// gone or has stopped reading.
    async fn write(&mut self, msg: Message) -> Result<(), ()> {
        match tokio::time::timeout(self.write, self.socket.send(msg)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) | Err(_) => Err(()),
        }
    }

    /// Sends one server event.
    async fn send(&mut self, event: Value) -> Result<(), ()> {
        self.write(Message::Text(event.to_string())).await
    }

    /// Sends the error event for `e`, if it has one.
    async fn send_error(&mut self, e: &Error) -> Result<(), ()> {
        match error_event(e) {
            Some(event) => self.send(event).await,
            None => Ok(()),
        }
    }

    /// Sends the error event for `e`, then the close frame. The caller
    /// returns either way.
    async fn refuse(&mut self, e: &Error) {
        if self.send_error(e).await.is_ok() {
            let _ = self.write(Message::Close(None)).await;
        }
    }
}

/// Takes a session permit, or tells the client the worker is full and
/// closes. Decided after the upgrade so the refusal is an event the client
/// reads, not a handshake failure the relay turns into a 502.
async fn admit(conn: &mut Conn, sessions: &Arc<Semaphore>) -> Option<OwnedSemaphorePermit> {
    match sessions.clone().try_acquire_owned() {
        Ok(permit) => Some(permit),
        Err(_) => {
            let e = Error::Failed(format!(
                "the worker has {MAX_SESSIONS} realtime sessions open, try again later"
            ));
            conn.refuse(&e).await;
            None
        }
    }
}

/// A fresh `sess_<n>`, unique within the process.
fn session_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!("sess_{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// What the session loop got from the socket.
enum Received {
    /// A JSON text frame; `last_event` was reset.
    Text(String),
    /// A binary frame, which the protocol has no use for.
    Binary,
    /// No event arrived within `idle` of the last one.
    Idle,
    /// The client closed or the socket failed.
    Gone,
}

/// The next client event, or why there is none. Pings and pongs are
/// swallowed and do not count as activity, so a client that only keeps
/// the connection alive still runs into the idle limit.
async fn recv_event(
    conn: &mut Conn,
    idle: std::time::Duration,
    last_event: &mut std::time::Instant,
) -> Received {
    loop {
        let left = idle.saturating_sub(last_event.elapsed());
        match tokio::time::timeout(left, conn.recv()).await {
            Err(_) => return Received::Idle,
            Ok(Some(Ok(Message::Text(text)))) => {
                *last_event = std::time::Instant::now();
                return Received::Text(text);
            }
            Ok(Some(Ok(Message::Binary(_)))) => return Received::Binary,
            Ok(Some(Ok(Message::Ping(_) | Message::Pong(_)))) => continue,
            Ok(Some(Ok(Message::Close(_))) | Some(Err(_)) | None) => return Received::Gone,
        }
    }
}

/// The next event to handle: the front of `queued` while it has one, else
/// the socket ([`recv_event`]). A queued frame arrived during a commit,
/// so it is not fresh activity and does not move `last_event`; once
/// `idle` has passed since the commit reply the drain stops as
/// [`Received::Idle`], so a client that stops sending and reads a queued
/// backlog slowly, each reply within the write deadline, is still closed
/// at the idle deadline.
async fn next_event(
    conn: &mut Conn,
    queued: &mut VecDeque<String>,
    idle: std::time::Duration,
    last_event: &mut std::time::Instant,
) -> Received {
    match queued.pop_front() {
        Some(text) if last_event.elapsed() < idle => Received::Text(text),
        Some(_) => Received::Idle,
        None => recv_event(conn, idle, last_event).await,
    }
}

/// One session: `session.created` first, then client events in order. A
/// commit runs on a blocking thread under the worker lock while this task
/// keeps reading the socket, so a client that leaves mid-decode drops the
/// commit's [`Guard`] and stops the compute; Text frames that arrive
/// meanwhile are queued and handled after it. The permit is shared with
/// each commit's blocking task, so it is released when this future ends
/// and no commit of it is still queued for the worker lock. The idle
/// clock does not run during a commit and restarts once its reply is
/// sent ([`IDLE_SECONDS`]); the queued frames are then drained under that
/// one deadline without restarting it ([`next_event`]).
async fn run_realtime(s: Shared, socket: WebSocket) {
    let mut conn = Conn {
        socket,
        write: s.write,
    };
    let Some(permit) = admit(&mut conn, &s.sessions).await else {
        return;
    };
    let permit = Arc::new(permit);
    let id = session_id();
    let mut session = match Session::open(&s.languages) {
        Ok(session) => session,
        Err(e) => {
            conn.refuse(&e).await;
            return;
        }
    };
    if conn
        .send(session_created(&id, &s.model, &session))
        .await
        .is_err()
    {
        return;
    }
    let mut previous_item: Option<String> = None;
    let mut queued: VecDeque<String> = VecDeque::new();
    let mut last_event = std::time::Instant::now();
    let mut restart_clock = false;
    loop {
        let text = match next_event(&mut conn, &mut queued, s.idle, &mut last_event).await {
            Received::Text(text) => text,
            Received::Binary => {
                let e = Error::Invalid("events are JSON text frames, not binary".into());
                if conn.send_error(&e).await.is_err() {
                    return;
                }
                continue;
            }
            Received::Idle => {
                let e = Error::Invalid(format!(
                    "no client event for {} s, closing the session",
                    s.idle.as_secs()
                ));
                conn.refuse(&e).await;
                return;
            }
            Received::Gone => return,
        };
        let outcome = match ClientEvent::parse(&text, session.room()) {
            Err(e) => Err(e),
            Ok(ClientEvent::Unknown(ty)) => {
                Err(Error::Invalid(format!("unknown event type {ty:?}")))
            }
            Ok(ClientEvent::SessionUpdate { rate, language }) => session
                .update(rate, language, &s.languages)
                .map(|()| Some(session_updated(&id, &s.model, &session))),
            Ok(ClientEvent::Append(samples)) => session.append(&samples).map(|()| None),
            Ok(ClientEvent::Clear) => {
                session.clear();
                Ok(Some(buffer_cleared()))
            }
            Ok(ClientEvent::Commit) => match session.commit() {
                Err(e) => Err(e),
                Ok((req, item)) => {
                    if conn
                        .send(buffer_committed(previous_item.as_deref(), &item))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    previous_item = Some(item.clone());
                    let server = s.clone();
                    let work = move |gone: Arc<AtomicBool>| run_locked(&server, &gone, &req);
                    let Some(result) =
                        run_commit(&permit, &mut conn, &mut queued, s.commit, work).await
                    else {
                        return;
                    };
                    // The commit's reply is what the client waited for:
                    // the idle clock starts over once it is written.
                    restart_clock = true;
                    result.map(|text| Some(transcription_completed(&item, &text)))
                }
            },
        };
        let sent = match outcome {
            Ok(Some(event)) => conn.send(event).await,
            Ok(None) => Ok(()),
            Err(e) => conn.send_error(&e).await,
        };
        if sent.is_err() {
            return;
        }
        if std::mem::take(&mut restart_clock) {
            last_event = std::time::Instant::now();
        }
    }
}

/// Runs one commit's `work` (the request under the worker lock,
/// [`run_locked`]) on a blocking thread while reading the socket. `None`
/// means the session is over: the client left, or the commit missed
/// `commit` ([`COMMIT_SECONDS`]) and the client was told so and closed.
/// Either way the [`Guard`] dropped, its flag is the one `work` was
/// given, and the compute stops at its next abort check. The blocking
/// task holds a clone of the session `permit`: a commit whose session
/// ended while it waits for the worker lock keeps its session slot until
/// it exits, so abandoned commits cannot queue more blocking tasks, and
/// their buffered PCM, than [`MAX_SESSIONS`]. The idle clock is the
/// caller's: a session with a commit in flight is exempt from the idle
/// close, and the clock starts over once the reply is written.
async fn run_commit(
    permit: &Arc<OwnedSemaphorePermit>,
    conn: &mut Conn,
    queued: &mut VecDeque<String>,
    commit: std::time::Duration,
    work: impl FnOnce(Arc<AtomicBool>) -> Result<String, Error> + Send + 'static,
) -> Option<Result<String, Error>> {
    let guard = Guard {
        gone: Arc::new(AtomicBool::new(false)),
    };
    let gone = guard.gone.clone();
    let permit = permit.clone();
    let mut task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work(gone)
    });
    let mut queued_bytes: usize = queued.iter().map(|t| t.len() + QUEUE_EVENT_OVERHEAD).sum();
    let deadline = tokio::time::sleep(commit);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            joined = &mut task => {
                let result = joined
                    .map_err(|e| Error::Failed(format!("transcription task failed: {e}")))
                    .and_then(|r| r);
                drop(guard);
                return Some(result);
            }
            () = &mut deadline => {
                // Flag first, so the compute is aborted even if the
                // refusal below waits out the write deadline.
                drop(guard);
                conn.refuse(&Error::CommitTimeout(commit)).await;
                return None;
            }
            frame = conn.recv() => {
                let refused = match frame {
                    Some(Ok(Message::Text(text))) => {
                        queue_event(queued, &mut queued_bytes, text).err()
                    }
                    Some(Ok(Message::Binary(_))) => {
                        Some(Error::Invalid("events are JSON text frames, not binary".into()))
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => None,
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return None,
                };
                if let Some(e) = refused {
                    // Bounded by the commit deadline too, so a client
                    // that stops reading cannot hold the deadline arm.
                    let sent = tokio::time::timeout_at(deadline.deadline(), conn.send_error(&e)).await;
                    if !matches!(sent, Ok(Ok(()))) {
                        return None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::server::tests::serve;
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use futures::{SinkExt as _, StreamExt as _};
    use std::sync::Mutex;

    #[test]
    fn queued_events_are_charged_their_length_plus_overhead() {
        let mut queued = VecDeque::new();
        let mut bytes = 0;
        // Exactly at the cap is accepted.
        let full = "x".repeat(QUEUE_BYTES - QUEUE_EVENT_OVERHEAD);
        assert!(queue_event(&mut queued, &mut bytes, full).is_ok());
        assert_eq!(bytes, QUEUE_BYTES);
        assert_eq!(queued.len(), 1);
        // Anything more, even an empty frame, is refused and not queued.
        let err = queue_event(&mut queued, &mut bytes, String::new()).unwrap_err();
        assert!(matches!(err, Error::TooLarge(_)), "{err:?}");
        assert_eq!(bytes, QUEUE_BYTES);
        assert_eq!(queued.len(), 1);

        // A flood of empty frames is refused once their overhead fills the cap.
        let mut queued = VecDeque::new();
        let mut bytes = 0;
        let admitted = (0..QUEUE_BYTES)
            .take_while(|_| queue_event(&mut queued, &mut bytes, String::new()).is_ok())
            .count();
        assert_eq!(admitted, QUEUE_BYTES / QUEUE_EVENT_OVERHEAD);
        assert_eq!(queued.len(), admitted);
        assert!(bytes <= QUEUE_BYTES);

        // Frames drain in the order they arrived.
        let mut queued = VecDeque::new();
        let mut bytes = 0;
        for text in ["first", "second", "third"] {
            queue_event(&mut queued, &mut bytes, text.to_string()).unwrap();
        }
        assert_eq!(
            bytes,
            3 * QUEUE_EVENT_OVERHEAD + "first".len() + "second".len() + "third".len()
        );
        let drained: Vec<String> = std::iter::from_fn(|| queued.pop_front()).collect();
        assert_eq!(drained, ["first", "second", "third"]);
    }

    /// A [`Conn`] for the handler tests, with a write deadline no test
    /// but the one for it reaches.
    fn conn(socket: WebSocket) -> Conn {
        Conn {
            socket,
            write: std::time::Duration::from_secs(5),
        }
    }

    /// The `/v1/realtime` URI of the served `addr`, for the loopback test
    /// clients: plain `ws`, TLS does not apply to a test on 127.0.0.1.
    fn realtime_uri(addr: std::net::SocketAddr) -> axum::http::Uri {
        axum::http::Uri::builder()
            .scheme("ws")
            .authority(addr.to_string())
            .path_and_query("/v1/realtime")
            .build()
            .unwrap()
    }

    /// A WebSocket client connected to `/v1/realtime` on `addr`.
    async fn connect(
        addr: std::net::SocketAddr,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        let (ws, _) = tokio_tungstenite::connect_async(realtime_uri(addr))
            .await
            .unwrap();
        ws
    }

    /// What [`recv_event`] returned, as the text frame a test reads.
    fn verdict(got: &Received) -> &'static str {
        match got {
            Received::Idle => "idle",
            Received::Text(_) => "text",
            Received::Binary => "binary",
            Received::Gone => "gone",
        }
    }

    /// The next text frame of `ws` within five seconds.
    async fn next_text<S>(ws: &mut S) -> String
    where
        S: futures::Stream<
                Item = Result<
                    tokio_tungstenite::tungstenite::Message,
                    tokio_tungstenite::tungstenite::Error,
                >,
            > + Unpin,
    {
        let read = async {
            loop {
                match ws.next().await {
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(t))) => break t,
                    Some(Ok(_)) => continue,
                    other => panic!("socket ended before a text frame: {other:?}"),
                }
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), read)
            .await
            .expect("no text frame within the deadline")
    }

    /// The worker is only reached by the daemon's relay, which sends no
    /// `Origin`: a request with one is refused before the upgrade with
    /// 403, and one without upgrades.
    #[tokio::test]
    async fn an_origin_header_is_refused_before_the_upgrade() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;

        let app = Router::new().route(
            "/v1/realtime",
            get(|headers: HeaderMap, ws: WebSocketUpgrade| async move {
                upgrade_or_refuse(&headers, ws, |socket| async move {
                    let _ = conn(socket).send(json!({"type": "upgraded"})).await;
                })
            }),
        );
        let addr = serve(app).await;

        let mut request = realtime_uri(addr).into_client_request().unwrap();
        request
            .headers_mut()
            .insert(header::ORIGIN, "http://evil.example".parse().unwrap());
        let err = tokio_tungstenite::connect_async(request)
            .await
            .expect_err("a request with an Origin upgraded");
        match err {
            tokio_tungstenite::tungstenite::Error::Http(response) => {
                assert_eq!(response.status(), StatusCode::FORBIDDEN);
            }
            other => panic!("expected an HTTP refusal, got {other:?}"),
        }

        let mut ws = connect(addr).await;
        let event: Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!(event["type"], "upgraded", "{event}");
    }

    /// A silent client is told so and closed once `idle` passes from the
    /// session start, and a ping in between does not push that deadline:
    /// the verdict arrives before the deadline a ping reset would set.
    #[tokio::test]
    async fn a_silent_session_is_closed_after_the_idle_limit() {
        use std::time::Duration;

        let idle = Duration::from_millis(600);
        let app = Router::new().route(
            "/v1/realtime",
            get(move |ws: WebSocketUpgrade| async move {
                capped(ws).on_upgrade(move |socket| async move {
                    let mut conn = conn(socket);
                    let mut last = std::time::Instant::now();
                    let got = recv_event(&mut conn, idle, &mut last).await;
                    let _ = conn.send(Value::String(verdict(&got).into())).await;
                    let _ = conn.write(Message::Close(None)).await;
                })
            }),
        );
        let addr = serve(app).await;

        let mut ws = connect(addr).await;
        let started = std::time::Instant::now();
        tokio::time::sleep(idle / 2).await;
        ws.send(tokio_tungstenite::tungstenite::Message::Ping(vec![]))
            .await
            .unwrap();
        let verdict: String = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!(verdict, "idle");
        // A reset by the ping would fire at idle / 2 + idle.
        assert!(
            started.elapsed() < idle + idle / 2,
            "the idle limit fired at {:?}, as if the ping had reset it",
            started.elapsed()
        );
    }

    /// A queued backlog is drained under the idle deadline the commit
    /// reply started, not reset by each drained frame: a client that
    /// stops sending and reads the replies slowly, each write well within
    /// its deadline, is told it is idle and closed once `idle` passes,
    /// with most of the backlog still unhandled.
    #[tokio::test]
    async fn a_slow_reader_of_a_queued_backlog_is_closed_at_the_idle_deadline() {
        use std::time::Duration;

        let idle = Duration::from_millis(600);
        const BACKLOG: usize = 1024;
        let app = Router::new().route(
            "/v1/realtime",
            get(move |ws: WebSocketUpgrade| async move {
                capped(ws).on_upgrade(move |socket| async move {
                    let mut conn = conn(socket);
                    // The frames a commit queued, and the clock as the
                    // commit reply just restarted it.
                    let mut queued: VecDeque<String> =
                        (0..BACKLOG).map(|_| "{}".to_string()).collect();
                    let mut last = std::time::Instant::now();
                    // Large enough that the socket buffers fill after a
                    // few, so each further write waits for the reader.
                    let reply = json!({"type": "test.reply", "pad": "x".repeat(256 << 10)});
                    loop {
                        match next_event(&mut conn, &mut queued, idle, &mut last).await {
                            Received::Text(_) => {
                                if conn.send(reply.clone()).await.is_err() {
                                    return;
                                }
                            }
                            got => {
                                let _ = conn.send(Value::String(verdict(&got).into())).await;
                                let _ = conn.write(Message::Close(None)).await;
                                return;
                            }
                        }
                    }
                })
            }),
        );
        let addr = serve(app).await;

        let mut ws = connect(addr).await;
        let started = std::time::Instant::now();
        let mut replies = 0;
        let verdict = loop {
            let event: Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
            match event {
                Value::String(verdict) => break verdict,
                _ => {
                    assert_eq!(event["type"], "test.reply", "{event}");
                    replies += 1;
                    assert!(replies < BACKLOG, "the whole backlog was drained");
                    // Slow, but well within the write deadline.
                    tokio::time::sleep(idle / 4).await;
                }
            }
        };
        assert_eq!(verdict, "idle");
        assert!(
            started.elapsed() >= idle,
            "closed after {:?}, before the idle deadline",
            started.elapsed()
        );
        // A clock that restarted per drained frame would never report
        // idle; the bound turns that into a fast failure. The slack is
        // for the replies already in the socket buffers at the verdict,
        // read at the same slow pace.
        assert!(
            started.elapsed() < idle * 10,
            "closed after {:?}, long past the idle deadline",
            started.elapsed()
        );
    }

    /// A client that stops reading cannot hold its slot: once the socket
    /// buffers fill, a write misses [`Conn::write`]'s deadline, the
    /// handler returns and its permit is free. A cap of one, a handler
    /// that keeps replying to a client that never reads, then a second
    /// client that must be admitted.
    #[tokio::test]
    async fn a_client_that_stops_reading_loses_its_slot_at_the_write_deadline() {
        use std::time::Duration;

        let write = Duration::from_millis(200);
        let sessions = Arc::new(Semaphore::new(1));
        let app = {
            let sessions = sessions.clone();
            Router::new().route(
                "/v1/realtime",
                get(move |ws: WebSocketUpgrade| {
                    let sessions = sessions.clone();
                    async move {
                        capped(ws).on_upgrade(move |socket| async move {
                            let mut conn = Conn { socket, write };
                            let Some(_permit) = admit(&mut conn, &sessions).await else {
                                return;
                            };
                            // Replies the client never reads, until one
                            // misses the deadline and the handler ends.
                            let reply = json!({"type": "test.reply", "pad": "x".repeat(64 << 10)});
                            while conn.send(reply.clone()).await.is_ok() {}
                        })
                    }
                }),
            )
        };
        let addr = serve(app).await;

        // The first client connects and never reads.
        let stalled = connect(addr).await;
        let started = std::time::Instant::now();
        let deadline = started + Duration::from_secs(10);
        // The permit is taken after the upgrade, so wait for the
        // admission before waiting for the slot to come back.
        while sessions.available_permits() == 1 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            sessions.available_permits(),
            0,
            "the first client was not admitted"
        );
        while sessions.available_permits() == 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            sessions.available_permits(),
            1,
            "the slot of a client that stopped reading was not freed"
        );
        // A second client is admitted while the first still holds its
        // end of the stalled socket.
        let mut ws = connect(addr).await;
        let event: Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!(event["type"], "test.reply", "{event}");
        drop(stalled);
    }

    /// The commit deadline for the handler tests, beyond what any of them
    /// but the one for it waits.
    const COMMIT: std::time::Duration = std::time::Duration::from_secs(30);

    /// A served `/v1/realtime` whose commit waits for a worker lock the
    /// test holds: the client leaves, the handler ends, and the session
    /// permit stays taken until the lock is free and the blocking task
    /// exits, so a cap of one still refuses a second session meanwhile.
    #[tokio::test]
    async fn a_disconnected_commit_keeps_its_session_slot_until_the_lock_frees() {
        use std::time::Duration;

        let sessions = Arc::new(Semaphore::new(1));
        let worker = Arc::new(Mutex::new(()));
        let (started, mut was_started) = tokio::sync::mpsc::unbounded_channel::<()>();
        let (ended, mut has_ended) = tokio::sync::mpsc::unbounded_channel::<bool>();
        let app = {
            let (sessions, worker) = (sessions.clone(), worker.clone());
            Router::new().route(
                "/v1/realtime",
                get(move |ws: WebSocketUpgrade| {
                    let (sessions, worker) = (sessions.clone(), worker.clone());
                    let (started, ended) = (started.clone(), ended.clone());
                    async move {
                        capped(ws).on_upgrade(move |socket| async move {
                            let mut conn = conn(socket);
                            let Some(permit) = admit(&mut conn, &sessions).await else {
                                return;
                            };
                            let permit = Arc::new(permit);
                            let mut queued = VecDeque::new();
                            let work = move |gone: Arc<AtomicBool>| {
                                started.send(()).unwrap();
                                let _lock = worker.lock().unwrap_or_else(|e| e.into_inner());
                                if gone.load(Ordering::SeqCst) {
                                    return Err(Error::Cancelled);
                                }
                                Ok("done".to_string())
                            };
                            let outcome =
                                run_commit(&permit, &mut conn, &mut queued, COMMIT, work).await;
                            ended.send(outcome.is_none()).unwrap();
                        })
                    }
                }),
            )
        };
        let addr = serve(app).await;

        // A thread holds the worker lock, as a running decode would,
        // until the test lets it go.
        let (locked, is_locked) = std::sync::mpsc::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let holder = {
            let worker = worker.clone();
            std::thread::spawn(move || {
                let _lock = worker.lock().unwrap_or_else(|e| e.into_inner());
                locked.send(()).unwrap();
                let _ = released.recv();
            })
        };
        is_locked.recv().unwrap();
        let ws = connect(addr).await;
        tokio::time::timeout(Duration::from_secs(5), was_started.recv())
            .await
            .expect("the commit did not reach the worker lock")
            .unwrap();
        // The client leaves while the commit waits for the lock.
        drop(ws);
        let client_left = tokio::time::timeout(Duration::from_secs(5), has_ended.recv())
            .await
            .expect("the handler did not end when its client left")
            .unwrap();
        assert!(
            client_left,
            "run_commit returned a result for a gone client"
        );
        assert!(
            sessions.clone().try_acquire_owned().is_err(),
            "the slot was released while the commit waited for the worker lock"
        );
        drop(release);
        holder.join().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while sessions.available_permits() == 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            sessions.clone().try_acquire_owned().is_ok(),
            "the slot was not released once the blocking task exited"
        );
    }

    /// A commit that misses the deadline closes the session: the client
    /// reads a `commit_timeout` error, then the close frame, the blocking
    /// task is flagged to abort, and the slot stays taken only until the
    /// lock frees and the task exits.
    #[tokio::test]
    async fn a_stuck_commit_is_refused_at_the_deadline_and_the_slot_frees_with_the_lock() {
        use std::time::Duration;

        let commit = Duration::from_millis(200);
        let sessions = Arc::new(Semaphore::new(1));
        let worker = Arc::new(Mutex::new(()));
        let (ended, mut has_ended) = tokio::sync::mpsc::unbounded_channel::<bool>();
        let (aborted, mut was_aborted) = tokio::sync::mpsc::unbounded_channel::<bool>();
        let app = {
            let (sessions, worker) = (sessions.clone(), worker.clone());
            Router::new().route(
                "/v1/realtime",
                get(move |ws: WebSocketUpgrade| {
                    let (sessions, worker) = (sessions.clone(), worker.clone());
                    let (ended, aborted) = (ended.clone(), aborted.clone());
                    async move {
                        capped(ws).on_upgrade(move |socket| async move {
                            let mut conn = conn(socket);
                            let Some(permit) = admit(&mut conn, &sessions).await else {
                                return;
                            };
                            let permit = Arc::new(permit);
                            let mut queued = VecDeque::new();
                            let work = move |gone: Arc<AtomicBool>| {
                                let _lock = worker.lock().unwrap_or_else(|e| e.into_inner());
                                aborted.send(gone.load(Ordering::SeqCst)).unwrap();
                                Ok("done".to_string())
                            };
                            let outcome =
                                run_commit(&permit, &mut conn, &mut queued, commit, work).await;
                            ended.send(outcome.is_none()).unwrap();
                        })
                    }
                }),
            )
        };
        let addr = serve(app).await;

        // A thread holds the worker lock past the deadline, as a stalled
        // backend would, until the test lets it go.
        let (locked, is_locked) = std::sync::mpsc::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let holder = {
            let worker = worker.clone();
            std::thread::spawn(move || {
                let _lock = worker.lock().unwrap_or_else(|e| e.into_inner());
                locked.send(()).unwrap();
                let _ = released.recv();
            })
        };
        is_locked.recv().unwrap();
        let mut ws = connect(addr).await;
        let event: Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!(event["type"], "error", "{event}");
        assert_eq!(event["error"]["code"], "commit_timeout", "{event}");
        assert!(
            event["error"]["message"]
                .as_str()
                .unwrap()
                .contains("within 200ms"),
            "the message does not name the limit: {event}"
        );
        let after = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("the socket stayed open after the error");
        assert!(
            matches!(
                after,
                Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_)))
            ),
            "expected the close, got {after:?}"
        );
        let timed_out = tokio::time::timeout(Duration::from_secs(5), has_ended.recv())
            .await
            .expect("the handler did not end at the deadline")
            .unwrap();
        assert!(
            timed_out,
            "run_commit returned a result for a timed out commit"
        );
        assert!(
            sessions.clone().try_acquire_owned().is_err(),
            "the slot was released while the commit waited for the worker lock"
        );
        drop(release);
        holder.join().unwrap();
        let flagged = tokio::time::timeout(Duration::from_secs(5), was_aborted.recv())
            .await
            .expect("the blocking task never took the lock")
            .unwrap();
        assert!(
            flagged,
            "the compute was not flagged to abort at the deadline"
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while sessions.available_permits() == 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            sessions.clone().try_acquire_owned().is_ok(),
            "the slot was not released once the blocking task exited"
        );
    }

    /// The refusal of a session past the cap is decided after the
    /// upgrade: the client reads an error event, then the close frame.
    /// A served route with a full counter pins that, without a model.
    #[tokio::test]
    async fn a_session_past_the_cap_reads_an_error_event_then_close() {
        use tokio_tungstenite::tungstenite::Message as WsMessage;

        let full = Arc::new(Semaphore::new(0));
        let app = Router::new().route(
            "/v1/realtime",
            get(move |ws: WebSocketUpgrade| {
                let full = full.clone();
                async move {
                    capped(ws).on_upgrade(move |socket| async move {
                        if admit(&mut conn(socket), &full).await.is_some() {
                            panic!("a full counter admitted a session");
                        }
                    })
                }
            }),
        );
        let addr = serve(app).await;

        let mut ws = connect(addr).await;
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("no frame within the deadline")
            .unwrap()
            .unwrap();
        let WsMessage::Text(text) = first else {
            panic!("expected a text frame, got {first:?}");
        };
        let event: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(event["type"], "error");
        assert_eq!(event["error"]["type"], "server_error");
        assert!(event["event_id"].as_str().unwrap().starts_with("event_"));
        let second = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("no close within the deadline")
            .unwrap()
            .unwrap();
        assert!(matches!(second, WsMessage::Close(_)), "got {second:?}");
    }

    /// A text frame past [`QUEUE_BYTES`] is refused by the socket layer
    /// before it is read into memory: the handler sees `Err` from `recv()`
    /// instead of the text, and the stream ends with no error event.
    #[tokio::test]
    async fn a_frame_past_the_queue_cap_ends_the_session_unparsed() {
        use tokio_tungstenite::tungstenite::Message as WsMessage;

        let (seen, mut outcome) = tokio::sync::mpsc::unbounded_channel::<&'static str>();
        let app = Router::new().route(
            "/v1/realtime",
            get(move |ws: WebSocketUpgrade| {
                let seen = seen.clone();
                async move {
                    capped(ws).on_upgrade(move |mut socket| async move {
                        let what = match socket.recv().await {
                            Some(Ok(Message::Text(_))) => "text",
                            Some(Ok(_)) => "other",
                            Some(Err(_)) => "error",
                            None => "end",
                        };
                        seen.send(what).unwrap();
                    })
                }
            }),
        );
        let addr = serve(app).await;

        let mut ws = connect(addr).await;
        // The server may close the connection while this frame is still
        // being written, so the send itself can fail with a reset; that
        // is the refusal, and the handler's verdict below is the check.
        let _ = ws.send(WsMessage::Text("x".repeat(QUEUE_BYTES + 1))).await;
        let what = tokio::time::timeout(std::time::Duration::from_secs(5), outcome.recv())
            .await
            .expect("the handler did not finish within the deadline")
            .unwrap();
        assert_eq!(what, "error");
        let next = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("no close or error within the deadline");
        match next {
            None | Some(Err(_)) | Some(Ok(WsMessage::Close(_))) => {}
            Some(Ok(other)) => panic!("expected the stream to end, got {other:?}"),
        }
    }

    fn append_event(bytes: &[u8]) -> String {
        use base64::Engine as _;
        json!({
            "type": "input_audio_buffer.append",
            "audio": base64::engine::general_purpose::STANDARD.encode(bytes),
        })
        .to_string()
    }

    #[test]
    fn client_events_parse_each_type() {
        let update = json!({"type": "session.update", "session": {"audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "transcription": {"language": "de"},
        }}}});
        assert_eq!(
            ClientEvent::parse(&update.to_string(), BUFFER_SAMPLES).unwrap(),
            ClientEvent::SessionUpdate {
                rate: Some(16000),
                language: Some("de".into()),
            }
        );
        assert_eq!(
            ClientEvent::parse(r#"{"type":"session.update","session":{}}"#, 1).unwrap(),
            ClientEvent::SessionUpdate {
                rate: None,
                language: None,
            }
        );
        assert_eq!(
            ClientEvent::parse(&append_event(&[1, 0, 0xff, 0xff]), BUFFER_SAMPLES).unwrap(),
            ClientEvent::Append(vec![1, -1])
        );
        assert_eq!(
            ClientEvent::parse(r#"{"type":"input_audio_buffer.commit"}"#, 0).unwrap(),
            ClientEvent::Commit
        );
        assert_eq!(
            ClientEvent::parse(r#"{"type":"input_audio_buffer.clear"}"#, 0).unwrap(),
            ClientEvent::Clear
        );
        assert_eq!(
            ClientEvent::parse(r#"{"type":"response.create"}"#, 0).unwrap(),
            ClientEvent::Unknown("response.create".into())
        );
    }

    #[test]
    fn malformed_client_events_are_invalid() {
        let invalid = |text: &str| matches!(ClientEvent::parse(text, 8), Err(Error::Invalid(_)));
        assert!(invalid("{not json"));
        assert!(invalid(r#"{"session":{}}"#));
        assert!(invalid(r#"{"type":"input_audio_buffer.append","audio":7}"#));
        assert!(invalid(
            r#"{"type":"input_audio_buffer.append","audio":"!!!!"}"#
        ));
        assert!(invalid(&append_event(&[1, 2, 3])));
        assert!(invalid(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"rate":"16k"}}}}}"#
        ));
        // A format or transcription that is not an object is refused,
        // not read as omitted.
        assert!(invalid(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":"audio/pcmu"}}}}"#
        ));
        assert!(invalid(
            r#"{"type":"session.update","session":{"audio":{"input":{"transcription":"en"}}}}"#
        ));
        assert!(invalid(r#"{"type":"session.update","session":[]}"#));
        // A format type other than audio/pcm is refused; a missing one is not.
        assert!(invalid(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"type":"audio/pcmu"}}}}}"#
        ));
        assert!(invalid(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"type":7}}}}}"#
        ));
        assert!(ClientEvent::parse(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"rate":16000}}}}}"#,
            0
        )
        .is_ok());
        // A payload longer than the room is refused before decoding.
        assert!(matches!(
            ClientEvent::parse(&append_event(&[0; 8]), 3),
            Err(Error::TooLarge(_))
        ));
        assert!(ClientEvent::parse(&append_event(&[0; 6]), 3).is_ok());
    }

    #[test]
    fn an_append_crossing_the_buffer_cap_leaves_the_buffer_intact() {
        let mut s = Session::default();
        s.append(&vec![1; BUFFER_SAMPLES - 1]).unwrap();
        assert_eq!(s.room(), 1);
        assert!(matches!(s.append(&[2, 2]), Err(Error::TooLarge(_))));
        assert_eq!(s.buffer.len(), BUFFER_SAMPLES - 1);
        s.append(&[2]).unwrap();
        assert_eq!(s.room(), 0);
        assert!(s.append(&[]).is_ok());
    }

    #[test]
    fn commits_number_items_and_refuse_an_empty_buffer() {
        let mut s = Session::default();
        assert_eq!(s.commit(), Err(Error::CommitEmpty));
        s.append(&[1, 2]).unwrap();
        let (req, item) = s.commit().unwrap();
        assert_eq!(item, "item_0");
        assert_eq!(
            req,
            Request {
                samples: vec![1, 2],
                language: "en".into(),
            }
        );
        assert!(s.buffer.is_empty());
        assert_eq!(s.commit(), Err(Error::CommitEmpty));
        s.append(&[3]).unwrap();
        s.clear();
        assert_eq!(s.commit(), Err(Error::CommitEmpty));
        s.append(&[3]).unwrap();
        assert_eq!(s.commit().unwrap().1, "item_1");
    }

    #[test]
    fn session_update_refuses_other_rates_and_unknown_languages() {
        let languages = vec!["en".to_string(), "de".to_string()];
        let mut s = Session::default();
        assert!(matches!(
            s.update(Some(24000), Some("de".into()), &languages),
            Err(Error::Invalid(_))
        ));
        assert_eq!((s.rate, s.language.as_str()), (16000, "en"));
        assert!(matches!(
            s.update(None, Some("xx".into()), &languages),
            Err(Error::Invalid(_))
        ));
        assert_eq!(s.language, "en");
        s.update(Some(16000), Some("de".into()), &languages)
            .unwrap();
        s.update(None, None, &languages).unwrap();
        assert_eq!((s.rate, s.language.as_str()), (16000, "de"));
        // An empty language is the default, as on the one-shot route: it
        // passes the membership check and resets the session to `en`.
        s.update(None, Some(String::new()), &languages).unwrap();
        assert_eq!(s.language, "en");
        let without_en = vec!["de".to_string()];
        assert!(matches!(
            s.update(None, Some(String::new()), &without_en),
            Err(Error::Invalid(_))
        ));
        assert_eq!(s.language, "en");
    }

    #[test]
    fn a_session_opens_in_en_or_the_models_first_language_and_says_which() {
        let s = Session::open(&["fr".to_string(), "en".to_string()]).unwrap();
        assert_eq!(s.language, "en");
        let s = Session::open(&["de".to_string(), "fr".to_string()]).unwrap();
        assert_eq!(s.language, "de");
        let created = session_created("sess_0", "canary", &s);
        assert_eq!(
            created["session"]["audio"]["input"]["transcription"]["language"],
            "de"
        );
        let Err(err) = Session::open(&[]) else {
            panic!("a model with no language opened a session");
        };
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
        assert_eq!(
            error_event(&err).unwrap()["error"]["type"],
            "invalid_request_error"
        );
    }

    #[test]
    fn server_events_carry_ids_and_the_error_mapping() {
        let s = Session::default();
        let created = session_created("sess_0", "canary", &s);
        assert_eq!(created["type"], "session.created");
        assert!(created["event_id"].as_str().unwrap().starts_with("event_"));
        assert_eq!(created["session"]["id"], "sess_0");
        assert_eq!(
            created["session"]["object"],
            "realtime.transcription_session"
        );
        assert!(created["session"]["audio"]["input"]["turn_detection"].is_null());
        assert_eq!(created["session"]["type"], "transcription");
        assert_eq!(created["session"]["model"], "canary");
        assert_eq!(
            created["session"]["audio"]["input"]["format"]["type"],
            "audio/pcm"
        );
        assert_eq!(
            created["session"]["audio"]["input"]["format"]["rate"],
            16000
        );
        let updated = session_updated("sess_0", "canary", &s);
        assert_eq!(updated["type"], "session.updated");
        assert_ne!(updated["event_id"], created["event_id"]);
        let committed = buffer_committed(None, "item_0");
        assert_eq!(committed["previous_item_id"], Value::Null);
        assert_eq!(committed["item_id"], "item_0");
        assert_eq!(
            buffer_committed(Some("item_0"), "item_1")["previous_item_id"],
            "item_0"
        );
        let done = transcription_completed("item_0", "hello");
        assert_eq!(
            done["type"],
            "conversation.item.input_audio_transcription.completed"
        );
        assert_eq!(done["content_index"], 0);
        assert_eq!(done["transcript"], "hello");
        assert_eq!(buffer_cleared()["type"], "input_audio_buffer.cleared");

        let ty =
            |e: &Error| error_event(e).map(|v| v["error"]["type"].as_str().unwrap().to_string());
        assert_eq!(
            ty(&Error::Invalid("x".into())).as_deref(),
            Some("invalid_request_error")
        );
        assert_eq!(
            ty(&Error::TooLarge("x".into())).as_deref(),
            Some("invalid_request_error")
        );
        assert_eq!(
            ty(&Error::Failed("x".into())).as_deref(),
            Some("server_error")
        );
        assert_eq!(
            ty(&Error::CommitEmpty).as_deref(),
            Some("input_audio_buffer_commit_empty")
        );
        assert_eq!(ty(&Error::Cancelled), None);
        let err = error_event(&Error::Invalid("bad".into())).unwrap();
        assert_eq!(err["type"], "error");
        assert_eq!(err["error"]["message"], "bad");
        assert_eq!(err["error"]["code"], Value::Null);
        assert!(err["event_id"].is_string());
    }
}
