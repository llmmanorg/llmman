//! The speech worker's HTTP server: `POST /v1/audio/transcriptions` and
//! the `GET /v1/realtime` WebSocket transcription session over one loaded
//! Canary model, with the discovery routes the daemon polls
//! after a spawn (`/health`, `/v1/models`, `/props`). Requests are
//! parsed and bounded before the model is touched, then run one at a
//! time under the [`Worker`] mutex: mel features, encoder, decoder, per
//! clip of at most [`CLIP_SECONDS`] for longer audio. Each
//! request registers its own abort flag with the backend before it
//! computes; a client that goes away sets that flag, which stops the
//! running graph and the decoder loop, and cannot reach any other request.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use anyhow::{bail, Result};
use axum::body::{Body, Bytes};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::decoder::{self, Decoder};
use super::encoder::Encoder;
use super::mel::{FeatureExtractor, MelConfig, MelError, Normalize};
use super::model::{CanaryModel, Frontend};
use super::prompt::PromptError;
use super::wav::{self, WavError};
use crate::mediagen::backend::Backend;
use crate::mediagen::http::{self, Discovery};

/// Samples per second of the one accepted WAV encoding.
const SAMPLE_RATE: usize = 16_000;

/// Longest clip the model sees at once. Canary 1B v2 was trained on
/// utterances up to 40 s; longer audio is cut into equal clips of at most
/// this length, transcribed one by one and joined with spaces. NeMo's own
/// chunked script does the same with 40 s clips; its dynamic chunking
/// overlaps and merges them, which this does not.
pub const CLIP_SECONDS: usize = 30;

/// Most `/v1/realtime` sessions open at once; a further one gets an
/// error event and is closed.
pub const MAX_SESSIONS: usize = 4;

/// Samples a realtime session buffers at most between commits: one clip,
/// so a commit never splits.
pub const BUFFER_SAMPLES: usize = CLIP_SECONDS * SAMPLE_RATE;

/// Room `max_body` leaves above `max_file` for the multipart framing and
/// the other fields.
const FRAMING_ALLOWANCE: usize = 64 << 10;

/// Caps a request before any audio is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Longest `file` part in bytes, the cap the WAV parser applies.
    pub max_file: usize,
    /// Longest request body in bytes, multipart framing included; at
    /// least `max_file` plus [`FRAMING_ALLOWANCE`], so a file at its cap
    /// still fits. It also bounds the audio: 4 MiB of PCM16 at 16 kHz is
    /// about 131 s, five clips.
    pub max_body: usize,
    /// Most requests admitted at once, each holding up to `max_body` bytes
    /// while it waits for the worker; the rest wait before reading a body.
    pub max_inflight: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_file: 4 << 20,
            max_body: (4 << 20) + FRAMING_ALLOWANCE,
            max_inflight: 8,
        }
    }
}

/// A rejected or failed request, rendered in the OpenAI error shape
/// `{"error": {"code", "message", "type"}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// 400: a missing, unknown or unacceptable field, or an unsupported
    /// language.
    Invalid(String),
    /// 413: the body or the audio exceeds its [`Limits`] cap.
    TooLarge(String),
    /// 415: the `file` part is not a PCM16 mono 16 kHz WAV.
    UnsupportedMedia(String),
    /// 499: the client went away and the compute was aborted. The code
    /// is nginx's "client closed request"; no client receives it over
    /// the wire, it marks the outcome in the worker's own accounting.
    Cancelled,
    /// 500: the pipeline failed.
    Failed(String),
    /// 400: a realtime `input_audio_buffer.commit` with nothing buffered;
    /// its own error type on the wire (`input_audio_buffer_commit_empty`).
    CommitEmpty,
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Error::Invalid(_) | Error::CommitEmpty => StatusCode::BAD_REQUEST,
            Error::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Error::UnsupportedMedia(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            // Outside the registered range, so built from the number.
            Error::Cancelled => StatusCode::from_u16(499).expect("499 is a valid status code"),
            Error::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = self.status();
        let (ty, msg) = match self {
            Error::Invalid(m) => ("invalid_request_error", m),
            Error::TooLarge(m) => ("invalid_request_error", m),
            Error::UnsupportedMedia(m) => ("invalid_request_error", m),
            Error::Cancelled => ("server_error", "the request was cancelled".to_string()),
            Error::Failed(m) => ("server_error", m),
            Error::CommitEmpty => ("input_audio_buffer_commit_empty", COMMIT_EMPTY.to_string()),
        };
        http::error_response(status, ty, &msg)
    }
}

/// The language of a request or session that names none.
const DEFAULT_LANGUAGE: &str = "en";

/// Message of [`Error::CommitEmpty`], llama-server's wording.
const COMMIT_EMPTY: &str = "input audio buffer is empty";

/// A parsed transcription request, ready for the mel front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// PCM16 mono 16 kHz samples from the `file` part.
    pub samples: Vec<i16>,
    /// The `language` field, `en` when absent or empty.
    pub language: String,
}

/// Rejects a body longer than `max_body` before anything reads it.
fn check_body_len(len: usize, limits: &Limits) -> Result<(), Error> {
    if len > limits.max_body {
        return Err(Error::TooLarge(format!(
            "the request body exceeds {} bytes",
            limits.max_body
        )));
    }
    Ok(())
}

/// Stores a field seen once; a repeat is a client error.
fn once<T>(slot: &mut Option<T>, name: &str, value: T) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::Invalid(format!("field \"{name}\" is repeated")));
    }
    *slot = Some(value);
    Ok(())
}

/// Parses a `multipart/form-data` transcription body.
///
/// The fields are those of OpenAI's transcription request this worker
/// honours: `file` (the WAV bytes) and `model` are required, `language`
/// defaults to `en`, `response_format` accepts `json` alone. Any other
/// field name is rejected, so a client learns at once that `prompt` or
/// `temperature` does not reach the model. The body length is checked
/// against `limits.max_body` before parsing and the WAV against
/// `limits.max_file`.
pub async fn parse_transcription(
    body: &[u8],
    content_type: &str,
    limits: &Limits,
) -> Result<Request, Error> {
    check_body_len(body.len(), limits)?;
    let boundary = multer::parse_boundary(content_type)
        .map_err(|_| Error::Invalid("the body must be multipart/form-data".into()))?;
    let chunk = Bytes::copy_from_slice(body);
    let stream = futures::stream::once(async { Ok::<_, std::io::Error>(chunk) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    let malformed = |e: multer::Error| Error::Invalid(format!("malformed multipart body: {e}"));

    let mut file: Option<Bytes> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    while let Some(field) = multipart.next_field().await.map_err(malformed)? {
        let name = field
            .name()
            .ok_or_else(|| Error::Invalid("a form field has no name".into()))?
            .to_owned();
        match name.as_str() {
            "file" => once(&mut file, &name, field.bytes().await.map_err(malformed)?)?,
            "model" => once(&mut model, &name, field.text().await.map_err(malformed)?)?,
            "language" => once(&mut language, &name, field.text().await.map_err(malformed)?)?,
            "response_format" => once(
                &mut response_format,
                &name,
                field.text().await.map_err(malformed)?,
            )?,
            other => {
                return Err(Error::Invalid(format!(
                    "unknown field \"{other}\": the accepted fields are file, model, language and response_format"
                )))
            }
        }
    }
    let file = file.ok_or_else(|| Error::Invalid("field \"file\" is required".into()))?;
    model.ok_or_else(|| Error::Invalid("field \"model\" is required".into()))?;
    if let Some(format) = response_format.filter(|f| !f.is_empty()) {
        if format != "json" {
            return Err(Error::Invalid(format!(
                "response_format {format:?} is not supported, only json is"
            )));
        }
    }
    let language = language
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| DEFAULT_LANGUAGE.to_owned());

    let samples = wav::parse(&file, limits.max_file).map_err(|e| match e {
        WavError::TooLarge => {
            Error::TooLarge(format!("the audio file exceeds {} bytes", limits.max_file))
        }
        e => Error::UnsupportedMedia(format!("the file must be a PCM16 mono 16 kHz WAV: {e}")),
    })?;
    Ok(Request { samples, language })
}

/// Classifies a pipeline failure. `cancelled` says the abort flag was
/// set, so a compute failure is the stop the client asked for by
/// leaving. The handler rejects an unsupported language before the
/// pipeline runs, so a prompt error here is a model-file fault.
fn compute_error(e: anyhow::Error, cancelled: bool) -> Error {
    if cancelled && e.downcast_ref::<PromptError>().is_none() {
        return Error::Cancelled;
    }
    Error::Failed(e.to_string())
}

/// One loaded model with the backend it lives on and the graphs over it.
pub struct Worker {
    // Dropped in this order: the weights before their backend.
    model: CanaryModel,
    enc: Encoder,
    dec: Decoder,
    be: Backend,
    /// The front end the file asks for ([`mel_front_end`]), its tables
    /// built once here and read by every clip.
    mel: FeatureExtractor,
}

/// Shortest hop the worker computes, 5 ms: a clip then yields at most
/// twice the recipe's frames, which bounds the encoder's work per clip
/// whatever a file's header says.
const MIN_HOP: u64 = 80;
/// Longest hop the worker computes: a full clip must yield the two frames
/// per-feature normalization needs, so a longer one would refuse every
/// transcription instead of the load.
const MAX_HOP: u64 = (CLIP_SECONDS * SAMPLE_RATE / 2) as u64;
/// Longest FFT the worker computes, eight times the recipe's.
const MAX_N_FFT: u64 = 4096;

/// The mel front end `frontend` (`stt.frontend.*`) describes, with its
/// tables built, or why this worker cannot compute it: the file must ask
/// for log-mel features over a Hann window at 16 kHz, the one rate the
/// WAV parser admits, with per-feature or no normalization. Every numeric
/// setting is honoured and range-checked, so a file with another hop or
/// FFT size gets the frames it expects instead of the default recipe's;
/// the hop stays within [`MIN_HOP`] and [`MAX_HOP`] and the FFT within
/// [`MAX_N_FFT`], so a header cannot make one clip arbitrarily expensive.
pub fn mel_front_end(frontend: &Frontend) -> Result<FeatureExtractor> {
    if frontend.kind != "mel" {
        bail!(
            "stt.frontend.type {:?}: only mel features are computed",
            frontend.kind
        );
    }
    if frontend.window != "hann" {
        bail!(
            "stt.frontend.window {:?}: only hann is computed",
            frontend.window
        );
    }
    let normalize = match frontend.normalize.as_str() {
        "per_feature" => Normalize::PerFeature,
        "" | "none" => Normalize::None,
        other => bail!("stt.frontend.normalize {other:?}: only per_feature or none is computed"),
    };
    if frontend.sample_rate != SAMPLE_RATE as u64 {
        bail!(
            "stt.frontend.sample_rate {}: the worker accepts {SAMPLE_RATE} Hz WAV only",
            frontend.sample_rate
        );
    }
    if frontend.hop_length < MIN_HOP {
        bail!(
            "stt.frontend.hop_length {}: the worker computes hops of at least {MIN_HOP} samples",
            frontend.hop_length
        );
    }
    if frontend.hop_length > MAX_HOP {
        bail!(
            "stt.frontend.hop_length {}: the worker computes hops of at most {MAX_HOP} samples",
            frontend.hop_length
        );
    }
    if frontend.n_fft > MAX_N_FFT {
        bail!(
            "stt.frontend.n_fft {}: the worker computes FFTs of at most {MAX_N_FFT}",
            frontend.n_fft
        );
    }
    let to_usize = |name: &str, v: u64| -> Result<usize> {
        usize::try_from(v).map_err(|_| anyhow::anyhow!("stt.frontend.{name} {v} is out of range"))
    };
    let cfg = MelConfig {
        n_mels: to_usize("num_mels", frontend.num_mels)?,
        sample_rate: SAMPLE_RATE as u32,
        n_fft: to_usize("n_fft", frontend.n_fft)?,
        win_length: to_usize("win_length", frontend.win_length)?,
        hop: to_usize("hop_length", frontend.hop_length)?,
        preemph: f64::from(frontend.pre_emphasis),
        f_min: f64::from(frontend.f_min),
        f_max: f64::from(frontend.f_max),
        normalize,
        ..MelConfig::default()
    };
    // The field checks, then the filterbank the fields describe: a header
    // the fields admit can still yield weights the features cannot use.
    // The tables built here serve every clip of every request.
    FeatureExtractor::new(cfg).map_err(|e| anyhow::anyhow!("stt.frontend: {e}"))
}

impl Worker {
    /// Checks the model against the encoder and decoder graphs and its
    /// front end against what [`mel_front_end`] computes.
    pub fn new(be: Backend, model: CanaryModel) -> Result<Worker> {
        let mel = mel_front_end(&model.hp().frontend)?;
        let enc = Encoder::new(&model, &be)?;
        let dec = Decoder::new(&model)?;
        Ok(Worker {
            model,
            enc,
            dec,
            be,
            mel,
        })
    }

    /// Runs the whole pipeline for one request with `gone` as the
    /// backend's abort flag, one clip at a time (see [`CLIP_SECONDS`]).
    /// The caller holds the worker lock, so no compute is running when
    /// the previous request's flag is replaced.
    fn transcribe(
        &mut self,
        samples: &[i16],
        language: &str,
        gone: Arc<AtomicBool>,
    ) -> Result<String, Error> {
        self.be.set_abort_flag(gone);
        let mut texts = Vec::new();
        for clip in clip_ranges(samples.len(), CLIP_SECONDS * SAMPLE_RATE) {
            if self.be.cancelled() {
                return Err(Error::Cancelled);
            }
            let text = self.transcribe_clip(&samples[clip], language)?;
            if !text.is_empty() {
                texts.push(text);
            }
        }
        Ok(texts.join(" "))
    }

    /// Mel features, encoder and decoder over one clip.
    fn transcribe_clip(&mut self, samples: &[i16], language: &str) -> Result<String, Error> {
        // The mel stage asks the backend's abort flag between frame
        // blocks, so a client that left stops the work here too, not only
        // in the encoder and decoder computes.
        let be = &self.be;
        let mel = self
            .mel
            .features(samples, &mut || be.cancelled())
            .map_err(|e| match e {
                MelError::TooShort => Error::Invalid(format!("the audio is too short: {e}")),
                MelError::InvalidConfig(_) => Error::Failed(e.to_string()),
                MelError::Cancelled => Error::Cancelled,
            })?;
        let run = self
            .enc
            .run(&self.model, &self.be, &mel, &[])
            .map_err(|e| compute_error(e, self.be.cancelled()))?;
        let out = decoder::transcribe(
            &self.dec,
            &self.model,
            &self.be,
            &run.out,
            run.frames,
            language,
            true,
        )
        .map_err(|e| compute_error(e, self.be.cancelled()))?;
        Ok(out.text.trim().to_owned())
    }
}

/// `len` samples as the fewest equal clips of at most `max` each, so a
/// long recording never ends in a stub: 31 s at a 30 s cap is two clips
/// of 15.5 s. Empty input is one empty clip, which the mel front end
/// refuses as too short.
fn clip_ranges(len: usize, max: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    let n = len.div_ceil(max.max(1)).max(1);
    (0..n).map(move |i| i * len / n..(i + 1) * len / n)
}

pub struct Server {
    worker: Mutex<Worker>,
    limits: Limits,
    /// `limits.max_inflight` permits, one per admitted request, held by
    /// the blocking task so a client that leaves does not free one early.
    inflight: Arc<Semaphore>,
    /// `general.languages`, checked before a request waits for the lock.
    languages: Vec<String>,
    /// The loaded model's name, carried by the session events.
    model: String,
    /// [`MAX_SESSIONS`] permits, one per open `/v1/realtime` session for
    /// its whole life. Separate from `inflight`: a session never queues
    /// behind the one-shot route's admission, it is refused at once.
    sessions: Arc<Semaphore>,
}

type Shared = Arc<Server>;

/// Lives in the handler future for as long as its request. Dropping it,
/// which axum does when the client goes away, sets this request's own
/// flag: the compute it is registered with stops, a compute not yet
/// started sees it and is skipped, and no other request shares it.
struct Guard {
    gone: Arc<AtomicBool>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.gone.store(true, Ordering::SeqCst);
    }
}

/// Takes the worker lock and runs the pipeline with `gone` as its abort
/// flag, unless the client already left.
fn run_locked(s: &Server, gone: &Arc<AtomicBool>, req: &Request) -> Result<String, Error> {
    let mut worker = s.worker.lock().unwrap_or_else(|e| e.into_inner());
    if gone.load(Ordering::SeqCst) {
        return Err(Error::Cancelled);
    }
    worker.transcribe(&req.samples, &req.language, gone.clone())
}

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
                let input = &v["session"]["audio"]["input"];
                match &input["format"]["type"] {
                    Value::Null => {}
                    Value::String(t) if t == PCM_FORMAT => {}
                    other => {
                        return Err(Error::Invalid(format!(
                            "session.audio.input.format.type {other}: the worker accepts {PCM_FORMAT:?} only"
                        )))
                    }
                }
                let rate =
                    match &input["format"]["rate"] {
                        Value::Null => None,
                        r => Some(r.as_u64().and_then(|r| u32::try_from(r).ok()).ok_or_else(
                            || {
                                Error::Invalid(
                                    "session.audio.input.format.rate must be an integer".into(),
                                )
                            },
                        )?),
                    };
                let language = match &input["transcription"]["language"] {
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
    /// Validated against the model's languages, `en` to begin with.
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
    /// A new session over a model speaking `languages`, or the error a
    /// model without [`DEFAULT_LANGUAGE`] gets before `session.created`:
    /// the default is checked as the one-shot route checks its request.
    pub fn open(languages: &[String]) -> Result<Session, Error> {
        let session = Session::default();
        if !languages.contains(&session.language) {
            return Err(Error::Invalid(
                PromptError::UnsupportedLanguage(session.language).to_string(),
            ));
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
    let (ty, message) = match e {
        Error::Invalid(m) | Error::TooLarge(m) | Error::UnsupportedMedia(m) => {
            ("invalid_request_error", m.as_str())
        }
        Error::Failed(m) => ("server_error", m.as_str()),
        Error::CommitEmpty => ("input_audio_buffer_commit_empty", COMMIT_EMPTY),
        Error::Cancelled => return None,
    };
    Some(json!({
        "type": "error",
        "event_id": event_id(),
        "error": {"type": ty, "code": null, "message": message, "param": null},
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

/// What the daemon learns about the model from the discovery routes.
fn discovery(model_name: String, model_path: String) -> Discovery {
    Discovery {
        model_name,
        model_path,
        capabilities: vec!["audio"],
        family: "canary",
        modalities: json!({
            "vision": false, "video": false, "audio": true,
            "image_generation": false, "video_generation": false, "audio_generation": false,
        }),
        extra_props: serde_json::Map::new(),
    }
}

/// The 400 for the text routes.
const NOT_SUPPORTED: &str = "the loaded model transcribes audio, use /v1/audio/transcriptions";

/// Reads at most `limits.max_body` bytes of `body`. A larger declared
/// length is refused before reading; a larger stream stops at the cap.
/// Both answer the JSON 413 rather than axum's plain-text one.
async fn read_body(headers: &HeaderMap, body: Body, limits: &Limits) -> Result<Bytes, Error> {
    if let Some(len) = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
    {
        check_body_len(len, limits)?;
    }
    match axum::body::to_bytes(body, limits.max_body).await {
        Ok(bytes) => Ok(bytes),
        // axum reports the cap through http_body_util's LengthLimitError,
        // whose message names the limit; any other read failure is the
        // client's stream ending early.
        Err(e) if e.to_string().contains("length limit") => Err(Error::TooLarge(format!(
            "the request body exceeds {} bytes",
            limits.max_body
        ))),
        Err(e) => Err(Error::Invalid(format!(
            "could not read the request body: {e}"
        ))),
    }
}

/// POST /v1/audio/transcriptions: admitted by `inflight`, then the parsed
/// request runs on a blocking thread under the worker lock; the reply is
/// `{"text": ...}`.
async fn transcriptions(
    State(s): State<Shared>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Value>, Error> {
    // Owned, so it can move into the blocking task below and outlive this
    // future when the client leaves; the semaphore is never closed.
    let permit = s
        .inflight
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| Error::Failed(format!("admission failed: {e}")))?;
    let body = read_body(&headers, body, &s.limits).await?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Error::Invalid("the body must be multipart/form-data".into()))?;
    let req = parse_transcription(&body, content_type, &s.limits).await?;
    if !s.languages.contains(&req.language) {
        return Err(Error::Invalid(
            PromptError::UnsupportedLanguage(req.language).to_string(),
        ));
    }
    let guard = Guard {
        gone: Arc::new(AtomicBool::new(false)),
    };
    let gone = guard.gone.clone();
    let text = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        run_locked(&s, &gone, &req)
    })
    .await
    .map_err(|e| Error::Failed(format!("transcription task failed: {e}")))??;
    // The guard drops here; its flag stays registered with the backend
    // until the next request replaces it, and nothing computes before then.
    drop(guard);
    Ok(Json(json!({"text": text})))
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
/// relay dials it without the `model` query, so none is read.
async fn realtime(State(s): State<Shared>, ws: WebSocketUpgrade) -> Response {
    capped(ws).on_upgrade(move |socket| run_realtime(s, socket))
}

/// Caps a frame and a message at [`QUEUE_BYTES`], so no text larger than
/// the queue admits is read into memory or parsed. A frame past the cap
/// surfaces as `Err` from `recv()` and the session ends.
fn capped(ws: WebSocketUpgrade) -> WebSocketUpgrade {
    ws.max_message_size(QUEUE_BYTES).max_frame_size(QUEUE_BYTES)
}

/// Sends one server event; `Err` means the client is gone.
async fn send(socket: &mut WebSocket, event: Value) -> Result<(), ()> {
    socket
        .send(Message::Text(event.to_string()))
        .await
        .map_err(|_| ())
}

/// Sends the error event for `e`, if it has one.
async fn send_error(socket: &mut WebSocket, e: &Error) -> Result<(), ()> {
    match error_event(e) {
        Some(event) => send(socket, event).await,
        None => Ok(()),
    }
}

/// Takes a session permit, or tells the client the worker is full and
/// closes. Decided after the upgrade so the refusal is an event the client
/// reads, not a handshake failure the relay turns into a 502.
async fn admit(socket: &mut WebSocket, sessions: &Arc<Semaphore>) -> Option<OwnedSemaphorePermit> {
    match sessions.clone().try_acquire_owned() {
        Ok(permit) => Some(permit),
        Err(_) => {
            let e = Error::Failed(format!(
                "the worker has {MAX_SESSIONS} realtime sessions open, try again later"
            ));
            if send_error(socket, &e).await.is_ok() {
                let _ = socket.send(Message::Close(None)).await;
            }
            None
        }
    }
}

/// A fresh `sess_<n>`, unique within the process.
fn session_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!("sess_{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// One session: `session.created` first, then client events in order. A
/// commit runs on a blocking thread under the worker lock while this task
/// keeps reading the socket, so a client that leaves mid-decode drops the
/// commit's [`Guard`] and stops the compute; Text frames that arrive
/// meanwhile are queued and handled after it. The permit is shared with
/// each commit's blocking task, so it is released when this future ends
/// and no commit of it is still queued for the worker lock.
async fn run_realtime(s: Shared, mut socket: WebSocket) {
    let Some(permit) = admit(&mut socket, &s.sessions).await else {
        return;
    };
    let permit = Arc::new(permit);
    let id = session_id();
    let mut session = match Session::open(&s.languages) {
        Ok(session) => session,
        Err(e) => {
            if send_error(&mut socket, &e).await.is_ok() {
                let _ = socket.send(Message::Close(None)).await;
            }
            return;
        }
    };
    if send(&mut socket, session_created(&id, &s.model, &session))
        .await
        .is_err()
    {
        return;
    }
    let mut previous_item: Option<String> = None;
    let mut queued: VecDeque<String> = VecDeque::new();
    loop {
        let text = match queued.pop_front() {
            Some(text) => text,
            None => match socket.recv().await {
                Some(Ok(Message::Text(text))) => text,
                Some(Ok(Message::Binary(_))) => {
                    let e = Error::Invalid("events are JSON text frames, not binary".into());
                    if send_error(&mut socket, &e).await.is_err() {
                        return;
                    }
                    continue;
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
            },
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
                    if send(
                        &mut socket,
                        buffer_committed(previous_item.as_deref(), &item),
                    )
                    .await
                    .is_err()
                    {
                        return;
                    }
                    previous_item = Some(item.clone());
                    let server = s.clone();
                    let work = move |gone: Arc<AtomicBool>| run_locked(&server, &gone, &req);
                    let Some(result) = run_commit(&permit, &mut socket, &mut queued, work).await
                    else {
                        return;
                    };
                    result.map(|text| Some(transcription_completed(&item, &text)))
                }
            },
        };
        let sent = match outcome {
            Ok(Some(event)) => send(&mut socket, event).await,
            Ok(None) => Ok(()),
            Err(e) => send_error(&mut socket, &e).await,
        };
        if sent.is_err() {
            return;
        }
    }
}

/// Runs one commit's `work` (the request under the worker lock,
/// [`run_locked`]) on a blocking thread while reading the socket. `None`
/// means the client left: the [`Guard`] dropped, its flag is the one
/// `work` was given, and the compute stops at its next abort check. The
/// blocking task holds a clone of the session `permit`: a commit whose
/// client left while it waits for the worker lock keeps its session slot
/// until it exits, so disconnecting clients cannot queue more blocking
/// tasks, and their buffered PCM, than [`MAX_SESSIONS`].
async fn run_commit(
    permit: &Arc<OwnedSemaphorePermit>,
    socket: &mut WebSocket,
    queued: &mut VecDeque<String>,
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
    loop {
        tokio::select! {
            joined = &mut task => {
                let result = joined
                    .map_err(|e| Error::Failed(format!("transcription task failed: {e}")))
                    .and_then(|r| r);
                drop(guard);
                return Some(result);
            }
            frame = socket.recv() => {
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
                    if send_error(socket, &e).await.is_err() {
                        return None;
                    }
                }
            }
        }
    }
}

/// The worker's routes over `worker`, with [`Limits::default`], plus the
/// discovery routes and the text route refusals every worker shares
/// ([`http`]).
pub fn router(worker: Worker, model_name: String, model_path: String) -> Router {
    let limits = Limits::default();
    let s = Arc::new(Server {
        languages: worker.model.hp().languages.clone(),
        worker: Mutex::new(worker),
        limits,
        inflight: Arc::new(Semaphore::new(limits.max_inflight)),
        model: model_name.clone(),
        sessions: Arc::new(Semaphore::new(MAX_SESSIONS)),
    });
    Router::new()
        .route(
            "/v1/audio/transcriptions",
            // `read_body` applies the cap itself, with the JSON reply.
            post(transcriptions).layer(DefaultBodyLimit::disable()),
        )
        .route("/v1/realtime", get(realtime))
        .with_state(s)
        .merge(http::discovery_routes(discovery(model_name, model_path)))
        .merge(http::text_routes_refused(NOT_SUPPORTED))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{SinkExt as _, StreamExt as _};

    const BOUNDARY: &str = "llmman-test-boundary";

    fn content_type() -> String {
        format!("multipart/form-data; boundary={BOUNDARY}")
    }

    /// One `form-data` part; `filename` marks a file part.
    fn part(name: &str, filename: Option<&str>, body: &[u8]) -> Vec<u8> {
        let mut out =
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"").into_bytes();
        if let Some(filename) = filename {
            out.extend_from_slice(
                format!("; filename=\"{filename}\"\r\nContent-Type: audio/wav").as_bytes(),
            );
        }
        out.extend_from_slice(b"\r\n\r\n");
        out.extend_from_slice(body);
        out.extend_from_slice(b"\r\n");
        out
    }

    fn multipart(parts: &[Vec<u8>]) -> Vec<u8> {
        let mut out = parts.concat();
        out.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        out
    }

    /// A PCM16 16 kHz WAV with `channels` channels, laid out as the
    /// fixture `fleurs-en.wav` is (a 16-byte `fmt ` chunk then `data`).
    fn wav(channels: u16, samples: &[i16]) -> Vec<u8> {
        let block_align = channels * 2;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&16_000u32.to_le_bytes());
        fmt.extend_from_slice(&(16_000 * u32::from(block_align)).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((4 + 8 + fmt.len() + 8 + data.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    const SAMPLES: [i16; 4] = [0, -1, 32767, -32768];

    fn small_limits() -> Limits {
        Limits {
            max_file: 512,
            max_body: 1024,
            max_inflight: 1,
        }
    }

    async fn parse(body: &[u8], limits: &Limits) -> Result<Request, Error> {
        parse_transcription(body, &content_type(), limits).await
    }

    fn good_parts() -> Vec<Vec<u8>> {
        vec![
            part("file", Some("a.wav"), &wav(1, &SAMPLES)),
            part("model", None, b"docker.io/ai/canary:latest"),
        ]
    }

    #[tokio::test]
    async fn good_request_parses_samples_with_the_default_language() {
        let req = parse(&multipart(&good_parts()), &small_limits())
            .await
            .unwrap();
        assert_eq!(req.samples, SAMPLES);
        assert_eq!(req.language, "en");
    }

    #[tokio::test]
    async fn language_and_json_response_format_are_accepted() {
        let mut parts = good_parts();
        parts.push(part("language", None, b"de"));
        parts.push(part("response_format", None, b"json"));
        let req = parse(&multipart(&parts), &small_limits()).await.unwrap();
        assert_eq!(req.language, "de");
    }

    #[tokio::test]
    async fn missing_file_is_400() {
        let body = multipart(&[part("model", None, b"m")]);
        let err = parse(&body, &small_limits()).await.unwrap_err();
        assert_eq!(err, Error::Invalid("field \"file\" is required".into()));
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_model_is_400() {
        let body = multipart(&[part("file", Some("a.wav"), &wav(1, &SAMPLES))]);
        let err = parse(&body, &small_limits()).await.unwrap_err();
        assert_eq!(err, Error::Invalid("field \"model\" is required".into()));
    }

    #[tokio::test]
    async fn unknown_field_is_400() {
        let mut parts = good_parts();
        parts.push(part("prompt", None, b"hello"));
        let err = parse(&multipart(&parts), &small_limits())
            .await
            .unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        assert!(matches!(err, Error::Invalid(m) if m.contains("unknown field \"prompt\"")));
    }

    #[tokio::test]
    async fn repeated_field_is_400() {
        let mut parts = good_parts();
        parts.push(part("model", None, b"again"));
        let err = parse(&multipart(&parts), &small_limits())
            .await
            .unwrap_err();
        assert_eq!(err, Error::Invalid("field \"model\" is repeated".into()));
    }

    #[tokio::test]
    async fn bad_response_format_is_400() {
        let mut parts = good_parts();
        parts.push(part("response_format", None, b"srt"));
        let err = parse(&multipart(&parts), &small_limits())
            .await
            .unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        assert!(matches!(err, Error::Invalid(m) if m.contains("\"srt\"")));
    }

    #[tokio::test]
    async fn non_multipart_content_type_is_400() {
        let err = parse_transcription(b"{}", "application/json", &small_limits())
            .await
            .unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    /// The default body cap leaves room above the file cap, and a file at
    /// its cap passes both length checks: it fails later, as not a WAV.
    #[tokio::test]
    async fn a_file_at_its_cap_fits_the_body_cap() {
        let limits = Limits::default();
        assert!(limits.max_body >= limits.max_file + FRAMING_ALLOWANCE);
        let limits = Limits {
            max_file: 512,
            max_body: 512 + 256,
            max_inflight: 1,
        };
        let body = multipart(&[
            part("file", Some("a.wav"), &[b'x'; 512]),
            part("model", None, b"canary"),
        ]);
        assert!(body.len() <= limits.max_body, "{}", body.len());
        let err = parse(&body, &limits).await.unwrap_err();
        assert_eq!(err.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE, "{err:?}");
        let body = multipart(&[
            part("file", Some("a.wav"), &[b'x'; 513]),
            part("model", None, b"canary"),
        ]);
        let err = parse(&body, &limits).await.unwrap_err();
        assert!(
            matches!(err, Error::TooLarge(ref m) if m.contains("audio file")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn body_above_the_limit_is_413_before_parsing() {
        let limits = Limits {
            max_file: 32,
            max_body: 64,
            max_inflight: 1,
        };
        // Not even multipart: the length check runs first.
        let err = parse(&[b'x'; 65], &limits).await.unwrap_err();
        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(matches!(err, Error::TooLarge(_)));
    }

    fn pinned_frontend() -> Frontend {
        Frontend {
            kind: "mel".into(),
            num_mels: 128,
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            window: "hann".into(),
            normalize: "per_feature".into(),
            pre_emphasis: 0.97,
            f_min: 0.0,
            f_max: 8000.0,
        }
    }

    /// The pinned file's front end is the default recipe, the header's
    /// f32 pre-emphasis aside; another hop is honoured; a window or rate
    /// the worker cannot compute is refused by name at load, not
    /// transcribed with the wrong preprocessing.
    #[test]
    fn mel_front_end_follows_the_frontend_and_refuses_what_it_cannot_compute() {
        assert_eq!(
            mel_front_end(&pinned_frontend()).unwrap().config(),
            &MelConfig {
                preemph: f64::from(0.97f32),
                ..MelConfig::default()
            }
        );

        let mut hop = pinned_frontend();
        hop.hop_length = 320;
        assert_eq!(mel_front_end(&hop).unwrap().config().hop, 320);

        let mut window = pinned_frontend();
        window.window = "hamming".into();
        let err = mel_front_end(&window).unwrap_err().to_string();
        assert!(err.contains("stt.frontend.window"), "{err}");

        let mut rate = pinned_frontend();
        rate.sample_rate = 8_000;
        let err = mel_front_end(&rate).unwrap_err().to_string();
        assert!(err.contains("sample_rate"), "{err}");

        let mut fft = pinned_frontend();
        fft.n_fft = 500;
        let err = mel_front_end(&fft).unwrap_err().to_string();
        assert!(err.contains("power of two"), "{err}");

        // A header cannot make a clip arbitrarily expensive.
        let mut tiny_hop = pinned_frontend();
        tiny_hop.hop_length = 1;
        let err = mel_front_end(&tiny_hop).unwrap_err().to_string();
        assert!(err.contains("hop_length"), "{err}");
        let mut huge_hop = pinned_frontend();
        huge_hop.hop_length = MAX_HOP + 1;
        let err = mel_front_end(&huge_hop).unwrap_err().to_string();
        assert!(err.contains("hop_length"), "{err}");
        let mut huge_fft = pinned_frontend();
        huge_fft.n_fft = 65_536;
        let err = mel_front_end(&huge_fft).unwrap_err().to_string();
        assert!(err.contains("n_fft"), "{err}");

        // The fields pass, the filterbank they describe does not: refused
        // at load, not on every request.
        let mut subnormal = pinned_frontend();
        subnormal.f_max = f32::from_bits(1);
        let err = mel_front_end(&subnormal).unwrap_err().to_string();
        assert!(err.contains("filterbank"), "{err}");
    }

    #[test]
    fn clips_are_equal_and_never_a_stub() {
        let ranges = |len, max| clip_ranges(len, max).collect::<Vec<_>>();
        assert_eq!(ranges(10, 30), vec![0..10]);
        assert_eq!(ranges(30, 30), vec![0..30]);
        assert_eq!(ranges(31, 30), vec![0..15, 15..31]);
        assert_eq!(
            ranges(131, 30),
            vec![0..26, 26..52, 52..78, 78..104, 104..131]
        );
        assert_eq!(ranges(0, 30), vec![0..0]);
        // Every clip is within the cap and the clips tile the input.
        for len in [1usize, 29, 30, 31, 59, 60, 61, 1000] {
            let r = ranges(len, 30);
            assert!(r.iter().all(|c| c.len() <= 30), "{len}: {r:?}");
            assert_eq!(r.first().unwrap().start, 0);
            assert_eq!(r.last().unwrap().end, len);
            assert!(r.windows(2).all(|w| w[0].end == w[1].start), "{len}: {r:?}");
        }
    }

    #[tokio::test]
    async fn stereo_wav_is_415() {
        let body = multipart(&[
            part("file", Some("a.wav"), &wav(2, &SAMPLES)),
            part("model", None, b"m"),
        ]);
        let err = parse(&body, &small_limits()).await.unwrap_err();
        assert_eq!(err.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert!(matches!(err, Error::UnsupportedMedia(m) if m.contains("channel count 2")));
    }

    #[tokio::test]
    async fn non_wav_file_is_415() {
        let body = multipart(&[
            part("file", Some("a.mp3"), b"ID3\x04\x00\x00\x00\x00\x00\x00"),
            part("model", None, b"m"),
        ]);
        let err = parse(&body, &small_limits()).await.unwrap_err();
        assert_eq!(err.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[test]
    fn other_prompt_errors_and_failures_map_to_500_or_cancelled() {
        let e = anyhow::Error::from(PromptError::MissingSpecial("pnc".into()));
        assert_eq!(
            compute_error(e, true).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let e = anyhow::anyhow!("graph compute cancelled");
        assert_eq!(compute_error(e, true), Error::Cancelled);
        let e = anyhow::anyhow!("graph compute failed with status 2");
        assert_eq!(
            compute_error(e, false).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

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

    #[test]
    fn dropping_the_guard_sets_its_own_flag() {
        let gone = Arc::new(AtomicBool::new(false));
        let other = Arc::new(AtomicBool::new(false));
        let guard = Guard { gone: gone.clone() };
        assert!(!gone.load(Ordering::SeqCst));
        drop(guard);
        assert!(gone.load(Ordering::SeqCst));
        assert!(!other.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn error_renders_the_openai_shape() {
        let resp = Error::TooLarge("too big".into()).into_response();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], 413);
        assert_eq!(v["error"]["message"], "too big");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(Error::Cancelled.status().as_u16(), 499);
    }

    /// Cancellation rests on axum dropping a handler's future when its
    /// HTTP/1 client closes the socket mid-request: the [`Guard`] inside
    /// drops with it. A served route pins that, without a model.
    #[tokio::test]
    async fn a_client_that_leaves_mid_request_drops_the_handler() {
        use std::time::{Duration, Instant};
        use tokio::io::AsyncWriteExt;

        let gone = Arc::new(AtomicBool::new(false));
        let flag = gone.clone();
        let app = Router::new().route(
            "/slow",
            post(move || {
                let flag = flag.clone();
                async move {
                    let _guard = Guard { gone: flag };
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    "done"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
        sock.write_all(b"POST /slow HTTP/1.1\r\nHost: t\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !gone.load(Ordering::SeqCst),
            "the handler ran before the client left"
        );
        drop(sock);

        let deadline = Instant::now() + Duration::from_secs(5);
        while !gone.load(Ordering::SeqCst) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            gone.load(Ordering::SeqCst),
            "the handler kept running after its client left"
        );
    }

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
                        capped(ws).on_upgrade(move |mut socket| async move {
                            let Some(permit) = admit(&mut socket, &sessions).await else {
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
                            let outcome = run_commit(&permit, &mut socket, &mut queued, work).await;
                            ended.send(outcome.is_none()).unwrap();
                        })
                    }
                }),
            )
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

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
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/realtime"))
            .await
            .unwrap();
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
                    capped(ws).on_upgrade(move |mut socket| async move {
                        if admit(&mut socket, &full).await.is_some() {
                            panic!("a full counter admitted a session");
                        }
                    })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/realtime"))
            .await
            .unwrap();
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
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/realtime"))
            .await
            .unwrap();
        ws.send(WsMessage::Text("x".repeat(QUEUE_BYTES + 1)))
            .await
            .unwrap();
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

    #[tokio::test]
    async fn read_body_caps_the_stream_and_the_declared_length() {
        let limits = Limits {
            max_file: 8,
            max_body: 16,
            max_inflight: 1,
        };
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_LENGTH, "17".parse().unwrap());
        let err = read_body(&headers, Body::from(vec![0u8; 17]), &limits)
            .await
            .unwrap_err();
        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // A stream with no declared length still stops at the cap.
        let stream = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(vec![0u8; 10])),
            Ok(Bytes::from(vec![0u8; 10])),
        ]);
        let err = read_body(&HeaderMap::new(), Body::from_stream(stream), &limits)
            .await
            .unwrap_err();
        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let ok = read_body(&HeaderMap::new(), Body::from(vec![0u8; 16]), &limits)
            .await
            .unwrap();
        assert_eq!(ok.len(), 16);
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
    fn a_session_opens_only_over_a_model_with_the_default_language() {
        let s = Session::open(&["de".to_string(), "en".to_string()]).unwrap();
        assert_eq!(s.language, "en");
        let Err(err) = Session::open(&["de".to_string()]) else {
            panic!("a model without en opened a session");
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
