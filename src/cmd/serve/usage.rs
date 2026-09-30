//! [`record_usage`] reads token usage off each generation reply on its
//! way to the client and appends it to `crate::usage`'s ledger. The reply
//! cannot say where the request went (a provider's names its own model
//! id, a hybrid pair's neither half), so `ensure_model` reports that via
//! [`note_target`] through a task-local scoped around the handler.

use std::sync::{Arc, Mutex};

use axum::body::{Body, HttpBody as _};
use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use futures::StreamExt;
use serde_json::Value;

use crate::providers::Cost;
use crate::usage::{Decoder, Dialect, Entry};

use super::{aggregation, now_rfc3339, AppState, Target};

tokio::task_local! {
    static CAPTURE: Arc<Mutex<Capture>>;
}

#[derive(Default)]
struct Capture {
    routed: Option<Routed>,
    /// [`ask_for_stream_usage`] asked for a chunk the client did not.
    strip_usage_chunk: bool,
}

struct Routed {
    model: String,
    provider: Option<String>,
    cost: Option<Cost>,
}

fn lock(capture: &Mutex<Capture>) -> std::sync::MutexGuard<'_, Capture> {
    capture.lock().unwrap_or_else(|e| e.into_inner())
}

/// Records that the current request went to `target` as `model`; the
/// last call wins. A no-op outside [`record_usage`].
pub(super) fn note_target(model: &str, target: &Target) {
    let _ = CAPTURE.try_with(|capture| {
        let (provider, cost) = match target {
            Target::Remote(remote) => (Some(remote.provider.clone()), remote.cost.clone()),
            Target::Local(_) | Target::Peer(_) => (None, None),
        };
        lock(capture).routed = Some(Routed {
            model: model.to_string(),
            provider,
            cost,
        });
    });
}

/// Sets `stream_options.include_usage` on a streamed OpenAI request so
/// the ledger has usage to read; if the client had not, the chunk is
/// stripped from its stream again. A no-op outside [`record_usage`].
pub(super) fn ask_for_stream_usage(req: &mut Value) {
    let _ = CAPTURE.try_with(|capture| {
        let asked = req
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool);
        if req.get("stream").and_then(Value::as_bool) != Some(true) || asked == Some(true) {
            return;
        }
        let Some(o) = req.as_object_mut() else {
            return;
        };
        let options = o
            .entry("stream_options")
            .or_insert_with(|| serde_json::json!({}));
        if options.is_null() {
            *options = serde_json::json!({});
        }
        if let Some(options) = options.as_object_mut() {
            options.insert("include_usage".to_string(), Value::Bool(true));
            lock(capture).strip_usage_chunk = true;
        }
    });
}

/// Appends each successful, routed generation reply's usage to the
/// ledger. A local reply to a peer's hop ([`aggregation::HOP`]) is the
/// forwarding daemon's to record; the header is the client's to set, so
/// a provider-routed one is recorded regardless.
pub(super) async fn record_usage(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(log) = state.0.usage_log.clone() else {
        return next.run(req).await;
    };
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string());
    let dialect = route.as_deref().and_then(Dialect::of_route);
    let (Some(route), Some(dialect)) = (route, dialect) else {
        return next.run(req).await;
    };
    if req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let hopped = req.headers().contains_key(aggregation::HOP);
    let time = now_rfc3339();
    let client = req
        .headers()
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let id = req
        .extensions()
        .get::<crate::promptlog::PromptId>()
        .map(|id| id.0.clone());

    let capture = Arc::new(Mutex::new(Capture::default()));
    let response = CAPTURE.scope(capture.clone(), next.run(req)).await;
    let Capture {
        routed,
        strip_usage_chunk,
    } = std::mem::take(&mut *lock(&capture));
    let routed =
        routed.filter(|r| response.status().is_success() && !(hopped && r.provider.is_none()));
    let Some(routed) = routed else {
        return response;
    };
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    let mut pending = Pending {
        log,
        id,
        time,
        route,
        client,
        routed,
        decoder: Decoder::new(dialect, content_type, strip_usage_chunk),
    };

    let (mut parts, body) = response.into_parts();
    if strip_usage_chunk {
        // A relayed `Content-Length` counts the chunk.
        parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    }
    let limit = crate::usage::BODY_LIMIT as u64;
    if body.size_hint().exact().is_some_and(|len| len <= limit) {
        // Already in memory (see `track_metrics`): read it whole, keeping
        // the reply's `Content-Length`. Anything larger streams below.
        return match axum::body::to_bytes(body, usize::MAX).await {
            Ok(bytes) => {
                let out = pending.decoder.feed(bytes);
                let rest = pending.decoder.finish();
                let body = if rest.is_empty() {
                    Body::from(out)
                } else {
                    Body::from([&out[..], &rest[..]].concat())
                };
                Response::from_parts(parts, body)
            }
            Err(e) => {
                parts.status = StatusCode::BAD_GATEWAY;
                parts.headers.remove(axum::http::header::CONTENT_LENGTH);
                Response::from_parts(parts, Body::from(format!("reading the reply: {e}")))
            }
        };
    }

    // `pending` records when the stream drops: at its end, or early when
    // the client disconnects.
    let stream = body
        .into_data_stream()
        .map(Some)
        .chain(futures::stream::once(async { None }))
        .filter_map(move |item| {
            let out = match item {
                Some(Ok(chunk)) => Ok(pending.decoder.feed(chunk)),
                Some(Err(e)) => Err(e),
                None => Ok(pending.decoder.finish()),
            };
            futures::future::ready(match out {
                Ok(bytes) if bytes.is_empty() => None,
                out => Some(out),
            })
        });
    Response::from_parts(parts, Body::from_stream(stream))
}

/// One request's ledger entry, written when dropped.
struct Pending {
    log: std::path::PathBuf,
    id: Option<String>,
    time: String,
    route: String,
    client: Option<String>,
    routed: Routed,
    decoder: Decoder,
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.decoder.finish();
        let Some(tokens) = self.decoder.tokens() else {
            return;
        };
        let id = self
            .id
            .take()
            .unwrap_or_else(|| crate::promptlog::new_id(&self.time, &self.route, b""));
        let mut entry = Entry {
            id,
            time: std::mem::take(&mut self.time),
            route: std::mem::take(&mut self.route),
            model: std::mem::take(&mut self.routed.model),
            provider: self.routed.provider.take(),
            client: self.client.take(),
            tokens,
            rate: None,
            cost: None,
        };
        if let Some(cost) = &self.routed.cost {
            entry.price(cost);
        }
        if let Err(e) = crate::usage::append(&self.log, &entry) {
            eprintln!("[llmman] warning: usage ledger {}: {e}", self.log.display());
        }
    }
}
