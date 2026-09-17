//! The last step of a route that reached a backend: the answer on its
//! way to the client, and — for the pass-through helpers — the request
//! on its way out.
//!
//! Two shapes, by how much of the body each has to read. [`relay`]
//! forwards the bytes untouched; the `*_rewriting_model` family reads
//! far enough to put the canonical name back in the `"model"` field a
//! backend addressed by another name echoes (see `backend_wire_model`).
//!
//! Each of them moves the [`ActivityGuard`] into the stream it returns
//! rather than dropping it on the way out — see that guard's own doc
//! comment for what goes wrong otherwise.

use anyhow::Context;
use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::response::Response;
use futures::StreamExt;
use reqwest::Client;

use super::sched::ActivityGuard;
use super::stream::bytes_to_lines;
use super::{AppError, ChatUpstream, Target};

pub(super) async fn proxy(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    // `Bytes` clones are refcounted, not copies — passing `body` straight
    // through (reqwest::Body: From<Bytes>) avoids an extra full-size
    // allocation that `body.to_vec()` would add on top of it, which
    // matters most for large multipart audio uploads.
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    Ok(relay(resp, activity))
}

/// The relay half of [`proxy`], split out so `remote_responses` can
/// inspect the status before deciding to relay.
pub(super) fn relay(resp: reqwest::Response, activity: ActivityGuard) -> Response {
    let status = resp.status();
    let resp_headers = resp.headers().clone();

    // Moved into the stream below (see ActivityGuard's doc comment) so it
    // isn't dropped — resetting this model's idle clock — until the whole
    // response body has actually been relayed.
    let stream = resp.bytes_stream().map(move |item| {
        let _activity = &activity;
        item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

// ---------------------------------------------------------------------------
// Proxy helpers – like `proxy` above, but for a request whose backend
// needed a *different* "model" name than the client itself asked for
// (see `backend_wire_model`'s own doc comment — only ever true for an
// `Engine::Mlx` backend, addressed by its real on-disk directory path
// rather than a human-readable name). `mlx_lm.server` echoes whatever
// "model" value it received straight back into every response it sends
// — the one non-streamed JSON body for `stream: false`, and *every*
// individual `data: {...}` SSE chunk for `stream: true` — so a plain
// byte-for-byte relay like `proxy` would leak that internal directory
// path back to the client instead of the name it actually asked for.
// These two rewrite just that one field back to the canonical name
// before any of it reaches the client; every other field, and (for the
// streaming variant) the SSE framing itself, passes through unchanged.
// ---------------------------------------------------------------------------

/// Sets `value["model"]` to `canonical_model` if that key is present at
/// all — shared by both helpers below so a response shape that happens
/// not to carry one (an error body, a future backend response this
/// doesn't recognize) is left alone rather than gaining a field it
/// never had.
pub(super) fn set_response_model(value: &mut serde_json::Value, canonical_model: &str) {
    if value.get("model").is_some() {
        value["model"] = serde_json::Value::String(canonical_model.to_string());
    }
    // A Responses API event nests it: `response.created`'s `response`.
    // So does a Messages API stream: `message_start`'s `message`.
    for key in ["response", "message"] {
        if let Some(nested) = value.get_mut(key) {
            if nested.get("model").is_some() {
                nested["model"] = serde_json::Value::String(canonical_model.to_string());
            }
        }
    }
}

/// [`proxy_rewriting_model`]'s actual rewrite, split out as a pure
/// `bytes -> bytes` function so it's directly unit-testable without any
/// networking at all. Parses `raw` as JSON, rewrites its `"model"` field
/// (see [`set_response_model`]), and re-serializes — or returns `raw`
/// completely unchanged if it isn't valid JSON at all (an error body's
/// own shape, or a future backend response this doesn't recognize)
/// rather than mangling or dropping it.
pub(super) fn rewrite_json_response_model(raw: &Bytes, canonical_model: &str) -> Bytes {
    match serde_json::from_slice::<serde_json::Value>(raw) {
        Ok(mut value) => {
            set_response_model(&mut value, canonical_model);
            serde_json::to_vec(&value)
                .map(Bytes::from)
                .unwrap_or_else(|_| raw.clone())
        }
        Err(_) => raw.clone(),
    }
}

/// [`stream_rewriting_model`]'s actual per-line rewrite, split out as a
/// pure `&str -> String` function so it's directly unit-testable without
/// any networking at all. `line` is one already-decoded logical line
/// from [`bytes_to_lines`] (its own line ending already stripped, not
/// yet restored here — the caller does that once, uniformly, since
/// every branch below needs it regardless of which one fires): a
/// `data: {...}` line whose payload parses as JSON gets its `"model"`
/// field rewritten (see [`set_response_model`]); `data: [DONE]`, a
/// blank SSE event-separator line, or a `data: ` line whose payload
/// *doesn't* parse as JSON all pass through byte-for-byte unchanged.
pub(super) fn rewrite_sse_line_model(line: &str, canonical_model: &str) -> String {
    match line.strip_prefix("data: ") {
        Some(payload) if payload != "[DONE]" => match serde_json::from_str(payload) {
            Ok(mut value) => {
                set_response_model(&mut value, canonical_model);
                format!(
                    "data: {}",
                    serde_json::to_string(&value).unwrap_or_else(|_| payload.to_string())
                )
            }
            Err(_) => line.to_string(),
        },
        _ => line.to_string(),
    }
}

/// The non-streaming (`stream: false`, or no `stream` concept at all —
/// embeddings, the Responses API's token-counting endpoint) case:
/// buffers the whole response body (unlike `proxy`, which never does)
/// so its `"model"` field can be parsed, rewritten, and re-serialized
/// before forwarding it on. Every route that can reach this returns one
/// complete JSON object either way (never anything token-streamed a
/// client would notice the added latency of buffering first), so this
/// costs nothing a real client could observe.
///
/// `Content-Length`, if the backend sent one, is dropped rather than
/// forwarded: the rewritten body is a different size than the original
/// one that header described, and hyper/axum fill in the correct value
/// for a fixed (`Body::from(Bytes)`, not streamed) body on their own
/// when none is set explicitly.
pub(super) async fn proxy_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: &str,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    relay_rewriting_model(resp, activity, canonical_model).await
}

/// The relay half of [`proxy_rewriting_model`]; see [`relay`].
pub(super) async fn relay_rewriting_model(
    resp: reqwest::Response,
    activity: ActivityGuard,
    canonical_model: &str,
) -> Result<Response, AppError> {
    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let raw = resp
        .bytes()
        .await
        .context("read inference backend response")?;
    // The whole body is already collected by this point, so there's no
    // partial relay left for keeping this alive any longer to protect —
    // see `proxy`'s own comment on why it instead holds this open across
    // its whole (streamed) relay.
    drop(activity);

    let rewritten = rewrite_json_response_model(&raw, canonical_model);

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        if k == reqwest::header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(k, v);
    }
    Ok(builder.body(Body::from(rewritten)).unwrap())
}

/// The streaming (`stream: true`) case: like `stream_ollama`/
/// `anthropic_messages_to`, uses `bytes_to_lines` so a `data: {...}` SSE line
/// split across two TCP reads is never parsed as JSON prematurely — but
/// unlike those two (which convert into a completely different wire
/// format, ndjson/Anthropic SSE, and so don't need to preserve the
/// original SSE framing at all), this must reproduce the exact original
/// OpenAI SSE shape byte-for-byte except for the one field being
/// rewritten: every blank line (an SSE event separator) and the
/// trailing `data: [DONE]` sentinel pass through completely unchanged;
/// only a `data: {...}` line whose payload actually parses as a JSON
/// object carrying a `model` field gets rewritten.
pub(super) async fn stream_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: String,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    Ok(relay_stream_rewriting_model(
        resp,
        activity,
        canonical_model,
    ))
}

/// The relay half of [`stream_rewriting_model`]; see [`relay`].
pub(super) fn relay_stream_rewriting_model(
    resp: reqwest::Response,
    activity: ActivityGuard,
    canonical_model: String,
) -> Response {
    let status = resp.status();
    // Every header but the ones describing a body about to be rewritten
    // line by line: a provider's `request-id`, rate limits and
    // `Retry-After` matter to the client.
    let mut resp_headers = resp.headers().clone();
    resp_headers.remove(reqwest::header::CONTENT_LENGTH);
    resp_headers.remove(reqwest::header::TRANSFER_ENCODING);

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        // See `proxy`'s own comment on this same pattern.
        let _activity = &activity;
        // bytes_to_lines strips the original line ending; restored here,
        // uniformly, regardless of which of rewrite_sse_line_model's own
        // branches actually fired.
        let out = rewrite_sse_line_model(&line, &canonical_model) + "\n";
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

/// [`relay`] for a [`ChatUpstream`]: `activity` lives until the whole
/// body has been relayed (see `ActivityGuard`).
pub(super) fn relay_chat_upstream(upstream: ChatUpstream, activity: ActivityGuard) -> Response {
    let stream = upstream.body.map(move |item| {
        let _activity = &activity;
        item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });
    let mut builder = Response::builder().status(upstream.status.as_u16());
    for (k, v) in &upstream.headers {
        builder = builder.header(k, v);
    }
    builder.body(Body::from_stream(stream)).unwrap()
}
