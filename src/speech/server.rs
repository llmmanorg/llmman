//! The speech worker's HTTP server: `POST /v1/audio/transcriptions` over
//! one loaded Canary model, with the discovery routes the daemon polls
//! after a spawn (`/health`, `/v1/models`, `/props`). Requests are
//! parsed and bounded before the model is touched, then run one at a
//! time under the [`Worker`] mutex: mel features, encoder, decoder, per
//! clip of at most [`CLIP_SECONDS`] for longer audio. Each
//! request registers its own abort flag with the backend before it
//! computes; a client that goes away sets that flag, which stops the
//! running graph and the decoder loop, and cannot reach any other request.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

use anyhow::{bail, Result};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
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
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Error::Invalid(_) => StatusCode::BAD_REQUEST,
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
        };
        http::error_response(status, ty, &msg)
    }
}

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
        .unwrap_or_else(|| "en".to_owned());

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
        let mel = self.mel.features(samples).map_err(|e| match e {
            MelError::TooShort => Error::Invalid(format!("the audio is too short: {e}")),
            MelError::InvalidConfig(_) => Error::Failed(e.to_string()),
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
    });
    Router::new()
        .route(
            "/v1/audio/transcriptions",
            // `read_body` applies the cap itself, with the JSON reply.
            post(transcriptions).layer(DefaultBodyLimit::disable()),
        )
        .with_state(s)
        .merge(http::discovery_routes(discovery(model_name, model_path)))
        .merge(http::text_routes_refused(NOT_SUPPORTED))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
