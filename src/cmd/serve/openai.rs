//! The OpenAI-compatible `/v1/*` surface: `models`, `chat/completions`,
//! `completions` and `embeddings`, the media generation and transcription
//! routes, and the request preparation and relay the Responses and
//! Anthropic routes share with them.

use anyhow::Context;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use tokio::time::{sleep, Duration, Instant};

use super::backend::would_use_mlx;
use super::hybrid::{request_pin, send_with_hybrid_fallback};
use super::messages::content_text;
use super::refusal::{explain_missing_route, unsupported_on_wire};
use super::relay::{
    proxy, proxy_rewriting_model, relay, relay_chat_upstream, stream_rewriting_model,
};
use super::responses::{
    is_responses_route, remote_responses, sanitize_responses_request, RESPONSES_ROUTE,
};
use super::sched::{begin_activity, ActivityGuard};
use super::types::*;
use super::{
    aggregation, backend_wire_model, ensure_model, provider_compat, send_chat_completion,
    strip_llama_fields, AppError, AppState, Engine, Target, CHAT_COMPLETIONS_ROUTE,
};
use crate::storage::OciStore;

pub(super) async fn handle_openai_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let mut data: Vec<serde_json::Value> = {
        let mgr = state.0.manager.lock().await;
        list.into_iter()
            .map(|img| {
                let loaded = mgr.running.contains_key(&img.reference);
                serde_json::json!({
                    "id": img.reference,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llmman",
                    // status field consumed by the web UI to track loaded/unloaded state
                    "status": { "value": if loaded { "loaded" } else { "unloaded" } },
                })
            })
            .collect()
    };
    // As `handle_tags`; loaded anywhere counts as loaded.
    if aggregation::aggregates(&state, &headers) {
        for (_, peer) in aggregation::poll::<serde_json::Value>(&state, "/v1/models").await {
            for m in peer["data"].as_array().into_iter().flatten() {
                match data.iter_mut().find(|have| have["id"] == m["id"]) {
                    Some(have) if m["status"]["value"] == "loaded" => {
                        have["status"] = m["status"].clone();
                    }
                    Some(_) => {}
                    None => data.push(m.clone()),
                }
            }
        }
    }
    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

/// Sets `repeat_penalty` to `DEFAULT_REPEAT_PENALTY` on `req` (an
/// OpenAI-shaped chat/completions request body) unless the caller already
/// supplied its own value. Every other entry point — `/api/chat`,
/// `/api/generate`, and the Anthropic Messages API — already forwards this
/// same default to llama-server via `post_chat` (see
/// `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself); a plain
/// OpenAI-compatible client has no llmman-specific reason to know it
/// should set this itself, so `proxy_openai_generation` applies it here
/// too, keeping every generation-capable API surface's behavior
/// consistent instead of leaving this one raw-passthrough path the sole
/// exception.
pub(super) fn apply_default_repeat_penalty(req: &mut serde_json::Value) {
    if req.get("repeat_penalty").is_none() {
        req["repeat_penalty"] = serde_json::json!(DEFAULT_REPEAT_PENALTY);
    }
}

/// Merges every `system` and `developer` message into one leading
/// `system` message, joined by a blank line.
///
/// Strict chat templates refuse a system message anywhere but index 0,
/// and a client may legitimately send several — an agent runner sends
/// one per toolset. `/v1/messages` and `/v1/responses` already merge
/// (`messages::from_messages_request`,
/// `consolidate_responses_instructions`); this does the same for the
/// third surface, except for how text parts within one message join —
/// `consolidate_responses_instructions` concatenates them with nothing
/// between. Callers pass local targets only, so a provider still gets
/// the array as written.
pub(super) fn consolidate_chat_system_messages(req: &mut serde_json::Value) {
    fn role(message: &serde_json::Value) -> &str {
        message.get("role").and_then(|v| v.as_str()).unwrap_or("")
    }
    fn leads(message: &serde_json::Value) -> bool {
        matches!(role(message), "system" | "developer")
    }

    let Some(messages) = req.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    // Already one leading system message: return it untouched rather
    // than rebuild it, which would flatten block content to text.
    let conforming = messages
        .iter()
        .enumerate()
        .all(|(i, m)| !leads(m) || i == 0)
        && messages.first().map(role) != Some("developer");
    if conforming {
        return;
    }
    let mut merged = MergedInstructions::default();
    messages.retain(|message| {
        if !leads(message) {
            return true;
        }
        merged.push(message.get("content").unwrap_or(&serde_json::Value::Null));
        false
    });
    let Some(content) = merged.finish() else {
        return;
    };
    messages.insert(
        0,
        serde_json::json!({ "role": "system", "content": content }),
    );
}

/// The merged instruction content, built in order. One entry per
/// message, its own text parts joined by a newline; the entries join
/// into one run separated by a blank line. A non-text block closes that
/// run and is kept as itself, making the result a block array.
#[derive(Default)]
struct MergedInstructions {
    blocks: Vec<serde_json::Value>,
    text: Vec<String>,
}

impl MergedInstructions {
    fn push(&mut self, content: &serde_json::Value) {
        match content {
            serde_json::Value::Array(parts) => {
                // One entry for the whole message: `flush_text`'s blank
                // line separates messages, so pushing each part on its
                // own would spell both boundaries the same way. `\n`
                // rather than `""` keeps the end of one part off the
                // start of the next.
                let mut run: Vec<&str> = Vec::new();
                for part in parts {
                    if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                        // Dropped, not joined: an empty part would become
                        // a stray newline, or a blank line between two
                        // real ones.
                        let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        if !text.is_empty() {
                            run.push(text);
                        }
                    } else {
                        // A non-text block ends the run it interrupts.
                        self.push_run(&mut run);
                        self.flush_text();
                        self.blocks.push(part.clone());
                    }
                }
                self.push_run(&mut run);
            }
            other => self.push_text(&content_text(other)),
        }
    }

    /// Takes `run`'s text parts as one instruction.
    fn push_run(&mut self, run: &mut Vec<&str>) {
        if run.is_empty() {
            return;
        }
        self.push_text(&std::mem::take(run).join("\n"));
    }

    fn push_text(&mut self, text: &str) {
        if !text.is_empty() {
            self.text.push(text.to_string());
        }
    }

    fn flush_text(&mut self) {
        if !self.text.is_empty() {
            let text = std::mem::take(&mut self.text).join("\n\n");
            self.blocks
                .push(serde_json::json!({"type": "text", "text": text}));
        }
    }

    /// A plain string when the instructions are text alone.
    fn finish(mut self) -> Option<serde_json::Value> {
        self.flush_text();
        match self.blocks.len() {
            0 => None,
            1 if self.blocks[0].get("type").and_then(|v| v.as_str()) == Some("text") => {
                self.blocks[0].get("text").cloned()
            }
            _ => Some(serde_json::Value::Array(self.blocks)),
        }
    }
}

/// Mirrors a local chat completion's `reasoning_effort` into the
/// `chat_template_kwargs` llama-server's templates read, as
/// `think_to_chat_template_kwargs` does for Ollama's `think`: `none` is
/// thinking off, a level is thinking on at that depth. Recent
/// llama-server reads `reasoning_effort` itself; older builds and vLLM
/// read only the kwargs. The caller's own kwargs win key by key.
/// [`provider_compat`] is the reverse, for a provider.
fn apply_reasoning_effort(req: &mut serde_json::Value) {
    let Some(effort) = req.get("reasoning_effort").and_then(|v| v.as_str()) else {
        return;
    };
    let think = if effort == "none" {
        serde_json::Value::Bool(false)
    } else {
        serde_json::Value::String(effort.to_string())
    };
    let Some(serde_json::Value::Object(kwargs)) = think_to_chat_template_kwargs(&Some(think))
    else {
        return;
    };
    let Some(o) = req.as_object_mut() else {
        return;
    };
    let slot = o
        .entry("chat_template_kwargs")
        .or_insert_with(|| serde_json::json!({}));
    let Some(existing) = slot.as_object_mut() else {
        return;
    };
    for (key, value) in kwargs {
        existing.entry(key).or_insert(value);
    }
}

/// Shared setup for every plain OpenAI-passthrough route: parse just
/// enough of the request to find `model`, make sure it's loaded, rewrite
/// `model` to its canonical name (see `ensure_model`), and open an
/// activity guard for it. `proxy_openai_generation` and
/// `proxy_openai_passthrough` below each finish shaping the parsed body
/// their own way (the former also defaults `repeat_penalty`, the latter
/// doesn't) before actually proxying it through.
///
/// The returned `Option<String>` is `Some(canonical_model)` only when
/// the backend actually needed a different wire name than the one the
/// client asked for (see `backend_wire_model`'s own doc comment — an
/// `Engine::Mlx` backend, or any remote provider, which knows the model
/// by its own unprefixed id) — the signal
/// `proxy_openai_generation`/`proxy_openai_passthrough` need to decide
/// whether the *response* also needs its own `"model"` field rewritten
/// back before reaching the client, via `proxy_rewriting_model`/
/// `stream_rewriting_model` instead of the plain `proxy`.
pub(super) async fn resolve_openai_request(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(serde_json::Value, Target, ActivityGuard, Option<String>), AppError> {
    let req = parse_openai_request(&body)?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    // No `request_threads`: the OpenAI-compatible surface has no Ollama
    // options blob, so there is no num_thread to forward.
    let (model, target, guard) = ensure_model(state, &model, Some(headers), None).await?;
    prepare_openai_request(state, req, model, target, guard).await
}

fn parse_openai_request(body: &Bytes) -> Result<serde_json::Value, AppError> {
    Ok(serde_json::from_slice(body).context("parse OpenAI request body")?)
}

/// [`resolve_openai_request`] after `ensure_model`, for a caller that
/// ran that itself (see [`send_with_hybrid_fallback`]).
async fn prepare_openai_request(
    state: &AppState,
    mut req: serde_json::Value,
    model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<(serde_json::Value, Target, ActivityGuard, Option<String>), AppError> {
    // The OpenAI-compatible surface has no `keep_alive` field of its own
    // (real Ollama's doesn't either) — `None` leaves whatever this model
    // already has untouched (its load-time default, or an explicit value
    // pinned via `/api/chat`) rather than overwriting it, e.g. clobbering
    // a `keep_alive: -1` ("never unload") pin with the daemon default the
    // instant one OpenAI-compatible request comes in.
    let activity = begin_activity(guard, None).await;
    // See backend_wire_model's own doc comment — usually just `model`
    // itself, but a different value for an Engine::Mlx backend or a
    // remote provider.
    let wire_model = backend_wire_model(state, &target, &model).await;
    let response_model_override = (wire_model != model).then_some(model);
    req["model"] = serde_json::Value::String(wire_model);
    Ok((req, target, activity, response_model_override))
}

/// OpenAI-passthrough for the endpoints that actually generate tokens —
/// chat completions, legacy completions, and the Responses API endpoint
/// Codex uses. Always defaults `repeat_penalty` (see
/// `apply_default_repeat_penalty`) rather than taking a bool flag callers
/// could forget to set: whether a route defaults this is now a choice of
/// *which function* it calls (this one, or `proxy_openai_passthrough`
/// below for the two non-generation routes), not an easily-mis-set
/// argument at the call site.
///
/// Picks which of the three proxy helpers actually relays the response
/// based on `resolve_openai_request`'s `response_model_override`: plain
/// `proxy` (untouched byte relay) when it's `None` — every engine
/// except `Engine::Mlx`, unchanged from before that engine existed —
/// otherwise `stream_rewriting_model`/`proxy_rewriting_model` depending
/// on whether this request itself asked for a streamed response, both
/// of which rewrite the response's own `"model"` field back to the
/// canonical name before it reaches the client (see either's own doc
/// comment for why that's needed at all).
pub(super) async fn proxy_openai_generation(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let req = parse_openai_request(&body)?;
    let model_ref = req["model"].as_str().unwrap_or("").to_string();
    send_with_hybrid_fallback(
        state,
        &model_ref,
        Some(headers),
        None,
        |model, target, guard| {
            proxy_openai_generation_to(
                state,
                headers,
                req.clone(),
                llama_path,
                model,
                target,
                guard,
            )
        },
    )
    .await
}

/// [`proxy_openai_generation`] against one resolved target.
async fn proxy_openai_generation_to(
    state: &AppState,
    headers: &HeaderMap,
    req: serde_json::Value,
    llama_path: &str,
    model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<Response, AppError> {
    let (mut req, target, activity, response_model_override) =
        prepare_openai_request(state, req, model, target, guard).await?;
    if let Some(refusal) = unsupported_on_wire(&target, llama_path) {
        drop(activity);
        return Ok(refusal);
    }
    if llama_path == RESPONSES_ROUTE && target.is_remote() {
        strip_llama_fields(&mut req);
        // A remote target always has an override (see backend_wire_model).
        let canonical = response_model_override
            .unwrap_or_else(|| req["model"].as_str().unwrap_or_default().to_string());
        return remote_responses(&state.0.client, &target, headers, req, activity, canonical).await;
    }
    if llama_path == CHAT_COMPLETIONS_ROUTE && target.is_anthropic() {
        let canonical = response_model_override
            .unwrap_or_else(|| req["model"].as_str().unwrap_or_default().to_string());
        // The translated reply already names the canonical model.
        let upstream = send_chat_completion(&state.0.client, &target, &req, &canonical).await?;
        return Ok(relay_chat_upstream(upstream, activity));
    }
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    if (llama_path == CHAT_COMPLETIONS_ROUTE || llama_path == "/v1/completions")
        && asks_for_usage(state, &target).await
    {
        super::usage::ask_for_stream_usage(&mut req);
    }
    match &target {
        Target::Remote(remote) if llama_path == CHAT_COMPLETIONS_ROUTE => {
            provider_compat(remote, &mut req)
        }
        Target::Remote(_) => strip_llama_fields(&mut req),
        _ => {
            apply_default_repeat_penalty(&mut req);
            if llama_path == CHAT_COMPLETIONS_ROUTE {
                apply_reasoning_effort(&mut req);
                consolidate_chat_system_messages(&mut req);
            }
        }
    }
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) if streaming => {
            stream_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                canonical,
            )
            .await
        }
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// Whether a stream's usage has to be asked for. Not of llama-server
/// (a peer's too), whose `timings` report it unasked and which moves them
/// onto the usage chunk the client would then lose; nor of Cohere, which
/// refuses `stream_options`.
async fn asks_for_usage(state: &AppState, target: &Target) -> bool {
    match target {
        Target::Remote(remote) => !remote.refuses_stream_options(),
        Target::Local(_) => local_engine(state, target).await != Some(Engine::LlamaServer),
        Target::Peer(_) => false,
    }
}

/// OpenAI-passthrough for the routes that don't generate anything a
/// repeat penalty could apply to — embeddings, and the Responses
/// token-counting endpoint. Same model-loading/canonicalization as
/// `proxy_openai_generation` (see `resolve_openai_request`), minus the
/// `repeat_penalty` default. Neither route has a `stream` concept of
/// its own, so unlike that function this only ever needs `proxy` or
/// `proxy_rewriting_model` — never the streaming variant.
///
/// `/v1/embeddings` specifically gets two more checks around that:
/// `Engine::Mlx` is never started with `mlx_lm.server`'s own
/// `--embedding-model` flag (`spawn_mlx_server` has no way to know which
/// model a caller would even want for that), so its conditional
/// `/v1/embeddings` route is never registered there at all — forwarding
/// anyway would just surface its own bare, unexplained 404, so this
/// fails fast with [`mlx_embeddings_unsupported_response`] instead,
/// twice over:
///
///   1. Before `resolve_openai_request` (and so `ensure_model`) ever
///      runs, via [`would_use_mlx`]'s own cheap, spawn-free check —
///      covering the overwhelmingly common case of a model that's
///      already resolvable locally, so a repeated embeddings request
///      against it never pays for spawning `mlx_lm.server` and loading
///      however many GB of weights for a request that could never
///      succeed there anyway.
///   2. After, via `response_model_override` — the fallback for the one
///      case (1) can't cheaply rule out ahead of time: a model that
///      wasn't in the local store at all yet, so `ensure_model` above
///      just pulled and loaded it for the first (and, thanks to (1),
///      only ever) time, and it turned out to be `Engine::Mlx` after
///      all.
pub(super) async fn proxy_openai_passthrough(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let embeddings = llama_path == "/v1/embeddings";
    if embeddings {
        if let Ok(peek) = serde_json::from_slice::<serde_json::Value>(&body) {
            if let Some(model_ref) = peek["model"].as_str() {
                // A provider-routed model is never an `Engine::Mlx` one —
                // it isn't local at all — so skip the whole check rather
                // than letting `would_use_mlx` do a pointless store lookup
                // against a reference that was never a store reference.
                if !crate::providers::is_remote_ref(model_ref) {
                    if let Some(canonical) = would_use_mlx(state, model_ref).await {
                        return Ok(mlx_embeddings_unsupported_response(&canonical));
                    }
                }
            }
        }
    }
    let (req, target, activity, response_model_override) =
        resolve_openai_request(state, headers, body).await?;
    forward_openai_request(
        state,
        headers,
        req,
        target,
        activity,
        response_model_override,
        llama_path,
    )
    .await
}

/// [`proxy_openai_passthrough`] after the model is loaded: wire checks,
/// then the proxy. Split out for [`handle_openai_media`].
async fn forward_openai_request(
    state: &AppState,
    headers: &HeaderMap,
    mut req: serde_json::Value,
    target: Target,
    activity: ActivityGuard,
    response_model_override: Option<String>,
    llama_path: &str,
) -> Result<Response, AppError> {
    let embeddings = llama_path == "/v1/embeddings";
    if let Some(refusal) = unsupported_on_wire(&target, llama_path) {
        drop(activity);
        return Ok(refusal);
    }
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    // `!target.is_remote()`, not just `response_model_override.is_some()`:
    // a remote target *always* sets that override (it addresses the model
    // by its own unprefixed id — see `backend_wire_model`), so without
    // this every provider-routed embeddings request would be rejected as
    // an MLX one. Only a local backend can actually be `Engine::Mlx`.
    if embeddings && !target.is_remote() {
        if let Some(canonical) = &response_model_override {
            drop(activity);
            return Ok(mlx_embeddings_unsupported_response(canonical));
        }
    }
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    // `/v1/responses/input_tokens` comes through here, and a provider
    // without the Responses API 404s it just the same.
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// The clear, specific error [`proxy_openai_passthrough`] returns
/// instead of forwarding a `/v1/embeddings` request on to an
/// `Engine::Mlx` backend — see that function's own doc comment on why
/// that request could never succeed there anyway.
fn mlx_embeddings_unsupported_response(canonical_model: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "{canonical_model} is served by mlx_lm.server, which llmman never starts with \
                 --embedding-model — /v1/embeddings isn't supported for it; use a GGUF or \
                 vllm-served model for embeddings instead"
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

pub(super) async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/chat/completions").await
}

pub(super) async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/completions").await
}

pub(super) async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/embeddings").await
}

// -- OpenAI media generation (/v1/images/generations, /v1/videos,
//    /v1/audio/speech) --------------------------------------------------------
//
// Pass-throughs to a diffusion model's backend (crate::mediagen::server),
// like handle_openai_embeddings — except for an `Engine::VllmOmni`
// backend; see omni_images and omni_videos.

pub(super) async fn handle_openai_images(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    handle_openai_media(&state, &headers, body, "/v1/images/generations").await
}

pub(super) async fn handle_openai_videos(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    handle_openai_media(&state, &headers, body, "/v1/videos").await
}

/// [`proxy_openai_passthrough`], but translated for a vLLM-Omni backend.
async fn handle_openai_media(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    route: &str,
) -> Result<Response, AppError> {
    let (req, target, activity, override_) = resolve_openai_request(state, headers, body).await?;
    if local_engine(state, &target).await == Some(Engine::VllmOmni) {
        return if route == "/v1/videos" {
            omni_videos(state, &target, req, activity).await
        } else {
            omni_images(state, &target, req, activity).await
        };
    }
    forward_openai_request(state, headers, req, target, activity, override_, route).await
}

/// The engine behind a [`Target::Local`]; `None` for anything else, or a
/// backend unloaded since `ensure_model` (the request then fails on its own).
async fn local_engine(state: &AppState, target: &Target) -> Option<Engine> {
    let Target::Local(port) = target else {
        return None;
    };
    let mgr = state.0.manager.lock().await;
    mgr.running
        .values()
        .find(|m| m.port == *port)
        .map(|m| m.process.engine())
}

// -- vLLM-Omni media dialect --------------------------------------------------
//
// `llmman run` and crate::mediagen speak llama-server's image API
// (`width`/`height`/`steps`/`cfg_scale`, streamed `image_generation.*`
// events, a synchronous `/v1/videos` job with a `content_url`). vLLM-Omni
// speaks OpenAI's (`size`, `num_inference_steps`, `guidance_scale`, no
// image streaming, a multipart asynchronous `/v1/videos`). Fields a
// client did not send are not invented — the model's defaults apply.

/// llama-server image fields renamed to vLLM-Omni's; `stream` removed and
/// returned (vLLM-Omni's text-to-image does not stream). `Err` names a
/// request that cannot be expressed: one dimension without the other
/// (llama-server fills in a default; vLLM-Omni's `size` needs both).
fn omni_image_request(mut req: serde_json::Value) -> Result<(serde_json::Value, bool), String> {
    let stream = req["stream"].as_bool().unwrap_or(false);
    let Some(obj) = req.as_object_mut() else {
        return Ok((req, stream));
    };
    obj.remove("stream");
    let dim = |v: Option<serde_json::Value>| v.and_then(|v| v.as_u64()).filter(|n| *n > 0);
    let (width, height) = (dim(obj.remove("width")), dim(obj.remove("height")));
    match (width, height) {
        (Some(w), Some(h)) if !obj.contains_key("size") => {
            obj.insert("size".into(), serde_json::json!(format!("{w}x{h}")));
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err("vLLM-Omni needs both width and height, or neither".into());
        }
        _ => {}
    }
    for (llama, omni) in [
        ("steps", "num_inference_steps"),
        ("cfg_scale", "guidance_scale"),
    ] {
        if let Some(v) = obj.remove(llama) {
            if !obj.contains_key(omni) && v.as_f64().is_some_and(|n| n > 0.0) {
                obj.insert(omni.into(), v);
            }
        }
    }
    if !obj.contains_key("response_format") {
        obj.insert("response_format".into(), "b64_json".into());
    }
    Ok((req, stream))
}

/// `POST /v1/images/generations` on a vLLM-Omni backend. A streaming
/// client gets one `image_generation.completed` event per image; a
/// non-streaming one gets vLLM-Omni's response as is.
async fn omni_images(
    state: &AppState,
    target: &Target,
    req: serde_json::Value,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let (req, stream) = match omni_image_request(req) {
        Ok(ok) => ok,
        Err(m) => {
            drop(activity);
            return Err(AppError::status(StatusCode::BAD_REQUEST, m));
        }
    };
    let resp = state
        .0
        .client
        .post(target.url("/v1/images/generations"))
        .json(&req)
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    if !stream || !resp.status().is_success() {
        return Ok(relay(resp, activity));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .context("decode vllm-omni image response")?;
    let created = body["created"]
        .as_i64()
        .unwrap_or_else(|| chrono::Utc::now().timestamp());
    let mut events = String::new();
    for image in body["data"].as_array().into_iter().flatten() {
        let ev = serde_json::json!({
            "type": "image_generation.completed",
            "b64_json": image["b64_json"],
            "revised_prompt": image["revised_prompt"],
            "created_at": created,
        });
        events.push_str(&format!("data: {ev}\n\n"));
    }
    drop(activity);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(events))
        .context("build image event stream")?)
}

/// Form fields for vLLM-Omni's `/v1/videos` from a JSON body: llama-server
/// names renamed (`steps`, `cfg_scale`, `frames`, `audio`), fractional
/// `seconds` rounded to the whole seconds it takes, everything else by
/// name (objects as JSON text, which is how it reads `extra_params`).
fn omni_video_fields(req: &serde_json::Value) -> Vec<(String, String)> {
    let Some(obj) = req.as_object() else {
        return Vec::new();
    };
    let renamed = |name: &str| match name {
        "steps" => Some("num_inference_steps"),
        "cfg_scale" => Some("guidance_scale"),
        "frames" => Some("num_frames"),
        "audio" => Some("generate_sound"),
        _ => None,
    };
    let text = |value: &serde_json::Value| match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    };
    let mut fields: Vec<(String, String)> = Vec::new();
    // Native names first, so an explicit one wins over its alias.
    for (name, value) in obj {
        if matches!(name.as_str(), "stream" | "response_format") || renamed(name).is_some() {
            continue;
        }
        let value = if name == "seconds" {
            match value.as_f64().or_else(|| value.as_str()?.parse().ok()) {
                Some(s) if s > 0.0 => {
                    serde_json::Value::String((s.round().max(1.0) as u64).to_string())
                }
                _ => continue,
            }
        } else {
            value.clone()
        };
        if let Some(text) = text(&value) {
            fields.push((name.clone(), text));
        }
    }
    for (name, value) in obj {
        let Some(native) = renamed(name) else {
            continue;
        };
        if fields.iter().any(|(existing, _)| existing == native) {
            continue;
        }
        if let Some(text) = text(value) {
            fields.push((native.to_string(), text));
        }
    }
    fields
}

/// A `multipart/form-data` body of text fields and its `content-type`.
/// Both come from the caller: a name that is not a plain identifier is
/// dropped (it would be written into a header), and the boundary is
/// re-salted until no value contains it.
fn multipart_form(fields: &[(String, String)]) -> (Vec<u8>, String) {
    let fields: Vec<&(String, String)> = fields
        .iter()
        .filter(|(name, _)| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .collect();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let boundary = (0u32..)
        .map(|salt| format!("----llmman{}{stamp:x}{salt:x}", std::process::id()))
        .find(|b| !fields.iter().any(|(_, v)| v.contains(b.as_str())))
        .unwrap_or_default();
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, format!("multipart/form-data; boundary={boundary}"))
}

/// How often [`omni_videos`] polls the job, and how long before it gives up.
const OMNI_VIDEO_POLL: Duration = Duration::from_secs(1);
const OMNI_VIDEO_MAX_WAIT: Duration = Duration::from_secs(60 * 60);

/// `POST /v1/videos` on a vLLM-Omni backend: submit the multipart job,
/// poll `GET /v1/videos/{id}` until `completed`/`failed`, answer with the
/// job plus the `content_url` [`handle_openai_video_get`] serves.
async fn omni_videos(
    state: &AppState,
    target: &Target,
    req: serde_json::Value,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let (body, content_type) = multipart_form(&omni_video_fields(&req));
    let resp = state
        .0
        .client
        .post(target.url("/v1/videos"))
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    if !resp.status().is_success() {
        return Ok(relay(resp, activity));
    }
    let mut job: serde_json::Value = resp.json().await.context("decode vllm-omni video job")?;
    let id = job["id"]
        .as_str()
        .context("vllm-omni video job has no id")?
        .to_string();
    let poll_url = target.url(&format!("/v1/videos/{id}"));
    let deadline = Instant::now() + OMNI_VIDEO_MAX_WAIT;
    while !matches!(job["status"].as_str(), Some("completed" | "failed")) {
        if Instant::now() >= deadline {
            drop(activity);
            return Err(AppError::status(
                StatusCode::GATEWAY_TIMEOUT,
                format!("video job {id} did not finish within {OMNI_VIDEO_MAX_WAIT:?}"),
            ));
        }
        sleep(OMNI_VIDEO_POLL).await;
        let resp = state
            .0
            .client
            .get(&poll_url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .with_context(|| format!("poll video job {id} on {}", target.describe()))?;
        // A failed job comes back as its own error status and JSON body.
        if !resp.status().is_success() {
            return Ok(relay(resp, activity));
        }
        job = resp.json().await.context("decode vllm-omni video job")?;
    }
    drop(activity);
    if job["status"] == "failed" {
        let message = job["error"]["message"]
            .as_str()
            .unwrap_or("video generation failed")
            .to_string();
        let body = serde_json::json!({
            "error": { "message": message, "type": "server_error" }
        });
        return Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response());
    }
    job["content_url"] = serde_json::json!(format!("/v1/videos/{id}/content"));
    Ok(Json(job).into_response())
}

/// `GET /v1/videos/:id[/content]`: a completed video lives in the
/// backend that generated it, and the job id names no model — so this
/// asks every running local backend in turn and relays the first answer
/// that is not a 404.
pub(super) async fn handle_openai_video_get(
    State(state): State<AppState>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> Result<Response, AppError> {
    let ports: Vec<u16> = {
        let mgr = state.0.manager.lock().await;
        mgr.running.values().map(|m| m.port).collect()
    };
    let path = uri.path();
    // all backends at once, so one stalled backend does not delay the rest
    let probes = ports.into_iter().map(|port| {
        state
            .0
            .client
            .get(format!("http://127.0.0.1:{port}{path}"))
            .timeout(Duration::from_secs(30))
            .send()
    });
    let found = futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .find(|r| r.status() != reqwest::StatusCode::NOT_FOUND);
    if let Some(resp) = found {
        let status = resp.status();
        let mut builder = Response::builder().status(status);
        for name in ["content-type", "content-disposition"] {
            if let Some(v) = resp.headers().get(name) {
                builder = builder.header(name, v);
            }
        }
        let stream = resp
            .bytes_stream()
            .map(|item| item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>));
        return Ok(builder
            .body(Body::from_stream(stream))
            .context("build video response")?);
    }
    let body = serde_json::json!({
        "error": { "message": "video not found", "type": "not_found_error" }
    });
    Ok((StatusCode::NOT_FOUND, Json(body)).into_response())
}

pub(super) async fn handle_openai_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/audio/speech").await
}

// -- OpenAI Audio Transcriptions API (/v1/audio/transcriptions) -------------
//
// llama-server has its own native implementation (requires the model to
// be loaded with mtmd audio support via a companion --mmproj — see
// ModelPath::mmproj), so this is a plain pass-through like
// handle_openai_responses. The request body is multipart/form-data, not
// JSON, so resolve_openai_request's "parse as JSON to find model" doesn't apply —
// multipart_text_field below extracts just the model field instead.

/// Axum's own default `DefaultBodyLimit` (2 MiB) is well under a typical
/// audio file's size — real recordings routinely run tens of MiB — so
/// both transcription routes below opt out of it in favor of this
/// higher cap instead of disabling it outright.
pub(super) const TRANSCRIPTION_BODY_LIMIT_BYTES: usize = 200 * 1024 * 1024;

/// Extracts a top-level form field's text value from a
/// `multipart/form-data` body, or `None` if not multipart / no boundary /
/// field not found.
async fn multipart_text_field(
    body: &Bytes,
    headers: &HeaderMap,
    field_name: &str,
) -> Option<String> {
    let content_type = headers.get("content-type")?.to_str().ok()?;
    let boundary = multer::parse_boundary(content_type).ok()?;
    // Single-chunk stream over a cheap Bytes clone — the body is already
    // fully buffered, so there's nothing to actually stream.
    let stream = futures::stream::once(async { Ok::<_, std::io::Error>(body.clone()) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some(field_name) {
            return field.text().await.ok();
        }
    }
    None
}

pub(super) async fn handle_openai_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let Some(model) = multipart_text_field(&body, &headers, "model")
        .await
        .filter(|m| !m.is_empty())
    else {
        // A malformed request, not a server-side failure — matches
        // handle_pull's own "missing required field" convention instead
        // of AppError's blanket 500.
        let body = serde_json::json!({
            "error": "transcription request is missing a required \"model\" form field"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    };
    // A pair takes its local half whatever its size: audio bodies are
    // past any byte budget, and the check below rules a provider out
    // anyway. An explicit cloud pin still fails with that same message.
    let model = match crate::hybrid::split_ref(&model) {
        Some(pair) => {
            let side = match request_pin(Some(&headers))? {
                Some(crate::hybrid::Side::Cloud) => crate::hybrid::Side::Cloud,
                _ => crate::hybrid::Side::Local,
            };
            eprintln!(
                "[llmman] hybrid {:?} + {:?} -> {} (transcription)",
                pair.local,
                pair.remote_ref(),
                side.as_str()
            );
            pair.side_ref(side)
        }
        None => model,
    };
    // Every other surface rewrites `model` to the provider's own id
    // before forwarding, but this body is multipart, not JSON: the raw
    // relay below would hand the provider a reference it has never heard
    // of. Refuse in llmman's own words rather than let that surface as
    // someone else's "unknown model".
    if crate::providers::is_remote_ref(&model) {
        let body = serde_json::json!({
            "error": "llmman does not route /v1/audio/transcriptions to a provider — \
                      use a locally served model with audio support"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    }
    let (_, target, guard) = ensure_model(&state, &model, Some(&headers), None).await?;
    // No `keep_alive` field on this API surface either — see
    // resolve_openai_request's own comment on the same choice.
    let activity = begin_activity(guard, None).await;
    proxy(
        &state.0.client,
        &target,
        "/v1/audio/transcriptions",
        &headers,
        body,
        activity,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `multipart/form-data` body + matching `content-type`
    /// header out of `fields` (name, value) pairs — a hand-rolled encoder
    /// rather than a dependency, just enough to exercise
    /// `multipart_text_field` against real (if minimal) multipart wire
    /// format.
    fn multipart_body(fields: &[(&str, &str)]) -> (Bytes, HeaderMap) {
        let boundary = "llmman-test-boundary";
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{boundary}--\r\n"));

        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            format!("multipart/form-data; boundary={boundary}")
                .parse()
                .unwrap(),
        );
        (Bytes::from(body), headers)
    }

    #[tokio::test]
    async fn multipart_text_field_finds_a_named_field_among_several() {
        let (body, headers) = multipart_body(&[
            ("language", "en"),
            ("model", "docker.io/ai/whisper:latest"),
            ("response_format", "json"),
        ]);
        assert_eq!(
            multipart_text_field(&body, &headers, "model").await,
            Some("docker.io/ai/whisper:latest".to_string())
        );
        assert_eq!(
            multipart_text_field(&body, &headers, "language").await,
            Some("en".to_string())
        );
    }

    #[tokio::test]
    async fn multipart_text_field_leaves_the_original_body_untouched() {
        // Regression: multipart_text_field parses a *clone* of the body
        // for the field it wants — the original `Bytes` handed to
        // `proxy` afterward must still be the exact, complete multipart
        // payload (file bytes included), not something already partially
        // consumed by this lookup.
        let (body, headers) = multipart_body(&[("model", "m"), ("prompt", "hello")]);
        let before = body.clone();
        let _ = multipart_text_field(&body, &headers, "model").await;
        assert_eq!(body, before);
    }

    #[tokio::test]
    async fn multipart_text_field_is_none_for_a_missing_field_or_non_multipart_body() {
        let (body, headers) = multipart_body(&[("language", "en")]);
        assert_eq!(multipart_text_field(&body, &headers, "model").await, None);

        let plain_body = Bytes::from_static(b"{\"model\":\"m\"}");
        let mut json_headers = HeaderMap::new();
        json_headers.insert("content-type", "application/json".parse().unwrap());
        assert_eq!(
            multipart_text_field(&plain_body, &json_headers, "model").await,
            None
        );

        // No content-type header at all.
        assert_eq!(
            multipart_text_field(&plain_body, &HeaderMap::new(), "model").await,
            None
        );
    }

    /// Regression test for the second CodeRabbit finding in review:
    /// `/v1/embeddings` against an `Engine::Mlx` backend must
    /// fail fast with a clear reason (not a bare, unexplained 404 from
    /// forwarding to `mlx_lm.server`, which never gets a
    /// `--embedding-model` from `spawn_mlx_server` — see
    /// `proxy_openai_passthrough`'s own doc comment).
    #[tokio::test]
    async fn mlx_embeddings_unsupported_response_explains_why_and_names_the_model() {
        let resp = mlx_embeddings_unsupported_response("gemma4:latest");
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("gemma4:latest"));
        assert!(message.contains("mlx_lm.server"));
        assert!(message.contains("/v1/embeddings"));
    }

    #[test]
    fn omni_image_request_translates_llama_server_fields() {
        // what `llmman run` (crate::imagegen) sends
        let req = serde_json::json!({
            "model": "nvidia/Cosmos3-Edge",
            "prompt": "a manatee",
            "stream": true,
            "response_format": "b64_json",
            "width": 640,
            "height": 480,
            "steps": 8,
            "seed": 42,
            "cfg_scale": 5.0,
            "negative_prompt": "blurry"
        });
        let (out, stream) = omni_image_request(req).unwrap();
        assert!(stream);
        assert_eq!(
            out,
            serde_json::json!({
                "model": "nvidia/Cosmos3-Edge",
                "prompt": "a manatee",
                "response_format": "b64_json",
                "size": "640x480",
                "num_inference_steps": 8,
                "seed": 42,
                "guidance_scale": 5.0,
                "negative_prompt": "blurry"
            })
        );
    }

    #[test]
    fn omni_image_request_leaves_a_native_request_alone_and_invents_nothing() {
        // A vLLM-Omni-dialect client: nothing renamed, nothing added but
        // the response format, and no size when none was asked for (the
        // model's own default is the one that works for Cosmos3-Edge).
        let req = serde_json::json!({
            "model": "m", "prompt": "p", "size": "1024x1024",
            "num_inference_steps": 50, "guidance_scale": 7.0
        });
        let (out, stream) = omni_image_request(req.clone()).unwrap();
        assert!(!stream);
        let mut expected = req;
        expected["response_format"] = "b64_json".into();
        assert_eq!(out, expected);
        let (out, _) =
            omni_image_request(serde_json::json!({"model": "m", "prompt": "p"})).unwrap();
        assert!(out.get("size").is_none());
        assert!(out.get("num_inference_steps").is_none());
        // zero means "model default", as it does for llama-server
        let (out, _) = omni_image_request(
            serde_json::json!({"prompt": "p", "width": 0, "height": 0, "steps": 0}),
        )
        .unwrap();
        assert!(out.get("size").is_none());
        assert!(out.get("num_inference_steps").is_none());
        // an explicit native field wins over a translated one
        let (out, _) = omni_image_request(
            serde_json::json!({"prompt": "p", "steps": 8, "num_inference_steps": 30}),
        )
        .unwrap();
        assert_eq!(out["num_inference_steps"], 30);
        // one dimension without the other cannot be expressed as a `size`
        let err = omni_image_request(serde_json::json!({"prompt": "p", "width": 768})).unwrap_err();
        assert!(err.contains("both width and height"), "{err}");
    }

    #[test]
    fn omni_video_fields_translate_and_round_seconds() {
        let req = serde_json::json!({
            "model": "nvidia/Cosmos3-Edge",
            "prompt": "waves",
            "stream": false,
            "seconds": 2.4,
            "fps": 24,
            "steps": 20,
            "cfg_scale": 5.0,
            "audio": false,
            "seed": 7,
            "extra_params": {"guardrails": false}
        });
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&req).into_iter().collect();
        assert_eq!(fields["model"], "nvidia/Cosmos3-Edge");
        assert_eq!(fields["prompt"], "waves");
        assert_eq!(fields["seconds"], "2", "SecondStr is a whole number");
        assert_eq!(fields["fps"], "24");
        assert_eq!(fields["num_inference_steps"], "20");
        assert_eq!(fields["guidance_scale"], "5.0");
        assert_eq!(fields["generate_sound"], "false");
        assert_eq!(fields["seed"], "7");
        assert_eq!(fields["extra_params"], r#"{"guardrails":false}"#);
        assert!(!fields.contains_key("stream"));
        assert!(!fields.contains_key("steps"));
        // sub-second clips round up to one second, not zero
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&serde_json::json!({"prompt": "p", "seconds": 0.3}))
                .into_iter()
                .collect();
        assert_eq!(fields["seconds"], "1");
        // `frames` is llama-server's name for num_frames; a native one wins
        let fields: std::collections::HashMap<String, String> =
            omni_video_fields(&serde_json::json!({"prompt": "p", "frames": 33, "num_frames": 49}))
                .into_iter()
                .collect();
        assert_eq!(fields["num_frames"], "49");
    }

    #[test]
    fn multipart_form_is_well_formed_and_readable_back() {
        let fields = vec![
            ("model".to_string(), "m".to_string()),
            ("prompt".to_string(), "a b\r\nc".to_string()),
        ];
        let (body, content_type) = multipart_form(&fields);
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let text = String::from_utf8(body.clone()).unwrap();
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
        // a name that is not an identifier would land in a header: dropped
        let (filtered, _) = multipart_form(&[
            ("ok_1".to_string(), "v".to_string()),
            ("bad\"\r\nX: y".to_string(), "v".to_string()),
        ]);
        let filtered = String::from_utf8(filtered).unwrap();
        assert!(filtered.contains("name=\"ok_1\""));
        // Look for the field, not the bare word: the boundary is hex from
        // the clock and does spell "bad" now and then.
        assert!(!filtered.contains("name=\"bad\""), "{filtered}");
        assert!(!filtered.contains("X: y"), "{filtered}");
        // a value containing the would-be boundary forces a different one
        let (_, ct2) = multipart_form(&[("prompt".to_string(), format!("x{boundary}y"))]);
        assert_ne!(ct2, content_type);
        // and the same parser the daemon uses for uploads agrees
        let mut headers = HeaderMap::new();
        headers.insert("content-type", content_type.parse().unwrap());
        let body = Bytes::from(body);
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert_eq!(
            rt.block_on(multipart_text_field(&body, &headers, "prompt")),
            Some("a b\r\nc".to_string())
        );
        assert_eq!(
            rt.block_on(multipart_text_field(&body, &headers, "model")),
            Some("m".to_string())
        );
    }

    /// An agent runner sends one `system` message per toolset, which a
    /// strict template refuses past the first.
    #[test]
    fn consolidate_chat_system_messages_merges_them_into_one_leading_message() {
        let mut req = serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "## Shell Tools"},
                {"role": "developer", "content": [{"type": "text", "text": "## Filesystem Tools"}]}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"],
            serde_json::json!([
                {"role": "system", "content":
                    "you are a helpful assistant\n\n## Shell Tools\n\n## Filesystem Tools"},
                {"role": "user", "content": "hi"}
            ])
        );
    }

    /// The common shape, and the one every ordinary request pays for:
    /// already-leading system message, returned byte for byte.
    #[test]
    fn consolidate_chat_system_messages_leaves_a_single_leading_one_alone() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "hi"}
            ]
        });
        let before = req.clone();
        consolidate_chat_system_messages(&mut req);
        assert_eq!(req, before);
    }

    /// Rebuilding a conforming request would flatten its content to text
    /// and drop every block that isn't.
    #[test]
    fn consolidate_chat_system_messages_keeps_block_content_when_conforming() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "be terse"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                ]},
                {"role": "user", "content": "hi"}
            ]
        });
        let before = req.clone();
        consolidate_chat_system_messages(&mut req);
        assert_eq!(req, before);
    }

    /// Folding content to text drops every block that isn't text, which
    /// forwards a truncated prompt with no error.
    #[test]
    fn consolidate_chat_system_messages_keeps_non_text_blocks_while_merging() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "be terse"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                ]},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "and cite sources"}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"],
            serde_json::json!([
                {"role": "system", "content": [
                    {"type": "text", "text": "be terse"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    {"type": "text", "text": "and cite sources"}
                ]},
                {"role": "user", "content": "hi"}
            ])
        );
    }

    /// The blank line separates messages, so one message's own parts must
    /// not use it too — two parts would read as two instructions.
    #[test]
    fn consolidate_chat_system_messages_keep_a_messages_own_parts_off_the_blank_line() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"}
                ]},
                {"role": "user", "content": "hi"},
                // Without this the request conforms and is returned as written.
                {"role": "developer", "content": "developer instruction"}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"][0]["content"],
            "part one\npart two\n\ndeveloper instruction"
        );
    }

    /// A non-text block ends the run it interrupts: parts before it stay one
    /// instruction, parts after it start another. The image is the object
    /// shape an OpenAI client sends, asserted back whole — `push` clones a
    /// non-text block rather than reading it, so its shape does not matter.
    #[test]
    fn consolidate_chat_system_messages_let_a_non_text_block_end_the_run() {
        let image = serde_json::json!({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}
        });
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "before one"},
                    {"type": "text", "text": "before two"},
                    image,
                    {"type": "text", "text": "after"}
                ]},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "and cite sources"}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"][0]["content"],
            serde_json::json!([
                {"type": "text", "text": "before one\nbefore two"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
                {"type": "text", "text": "after\n\nand cite sources"}
            ])
        );
        // Non-instruction turns are left alone.
        assert_eq!(
            req["messages"][1],
            serde_json::json!({"role": "user", "content": "hi"})
        );
        assert_eq!(req["messages"].as_array().map(Vec::len), Some(2));
    }

    /// An empty part is dropped, not joined: it would otherwise leave a
    /// stray newline at either end of the run, or a blank line mid-message.
    #[test]
    fn consolidate_chat_system_messages_drop_empty_parts_from_the_run() {
        let cases = [
            // Leading, trailing and lone empties leave no trace.
            (
                serde_json::json!([{"type": "text", "text": "a"},
                                {"type": "text", "text": ""}]),
                "a\n\ndev",
            ),
            (
                serde_json::json!([{"type": "text", "text": ""},
                                {"type": "text", "text": "b"}]),
                "b\n\ndev",
            ),
            (
                serde_json::json!([{"type": "text", "text": ""},
                                {"type": "text", "text": ""}]),
                "dev",
            ),
            // One between two real parts does not split them.
            (
                serde_json::json!([{"type": "text", "text": "a"},
                                {"type": "text", "text": ""},
                                {"type": "text", "text": "b"}]),
                "a\nb\n\ndev",
            ),
        ];
        for (content, want) in cases {
            let mut req = serde_json::json!({
                "messages": [
                    {"role": "system", "content": content},
                    {"role": "user", "content": "hi"},
                    {"role": "developer", "content": "dev"}
                ]
            });
            consolidate_chat_system_messages(&mut req);
            assert_eq!(req["messages"][0]["content"], want);
        }
    }

    /// A late system turn is the shape templates reject, so it moves — and
    /// reorders relative to the user turn before it, as `/v1/messages` does.
    #[test]
    fn consolidate_chat_system_messages_moves_a_late_lone_system_turn_to_the_front() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "a mid-conversation reminder"}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"],
            serde_json::json!([
                {"role": "system", "content": "a mid-conversation reminder"},
                {"role": "user", "content": "hi"}
            ])
        );
    }

    /// A request with no system message must not gain one.
    #[test]
    fn consolidate_chat_system_messages_is_a_no_op_without_any() {
        let mut req = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        let before = req.clone();
        consolidate_chat_system_messages(&mut req);
        assert_eq!(req, before);

        // An embeddings-shaped body has no `messages` at all.
        let mut input_only = serde_json::json!({"input": "hi"});
        let before = input_only.clone();
        consolidate_chat_system_messages(&mut input_only);
        assert_eq!(input_only, before);
    }

    /// An empty system message must not join a blank line into the merge.
    #[test]
    fn consolidate_chat_system_messages_drops_empty_ones() {
        let mut req = serde_json::json!({
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "the only real instruction"}
            ]
        });

        consolidate_chat_system_messages(&mut req);

        assert_eq!(
            req["messages"],
            serde_json::json!([
                {"role": "system", "content": "the only real instruction"},
                {"role": "user", "content": "hi"}
            ])
        );
    }

    /// A local `reasoning_effort` becomes the `chat_template_kwargs` Ollama's
    /// `think` would; the caller's kwargs win; an unknown level changes nothing.
    #[test]
    fn apply_reasoning_effort_mirrors_it_into_template_kwargs() {
        let with = |req: serde_json::Value| {
            let mut req = req;
            apply_reasoning_effort(&mut req);
            req
        };
        assert_eq!(
            with(serde_json::json!({ "model": "m", "reasoning_effort": "none" })),
            serde_json::json!({
                "model": "m", "reasoning_effort": "none",
                "chat_template_kwargs": { "enable_thinking": false }
            })
        );
        for level in crate::chat_template::EFFORT_LEVELS {
            assert_eq!(
                with(serde_json::json!({ "model": "m", "reasoning_effort": level })),
                serde_json::json!({
                    "model": "m", "reasoning_effort": level,
                    "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": level }
                }),
                "{level}"
            );
        }
        assert_eq!(
            with(serde_json::json!({
                "model": "m", "reasoning_effort": "high",
                "chat_template_kwargs": { "enable_thinking": false }
            })),
            serde_json::json!({
                "model": "m", "reasoning_effort": "high",
                "chat_template_kwargs": { "enable_thinking": false, "reasoning_effort": "high" }
            })
        );
        let unknown = serde_json::json!({ "model": "m", "reasoning_effort": "verbose" });
        assert_eq!(with(unknown.clone()), unknown);
        let absent = serde_json::json!({ "model": "m", "chat_template_kwargs": { "a": 1 } });
        assert_eq!(with(absent.clone()), absent);
        let not_a_string = serde_json::json!({ "model": "m", "reasoning_effort": 3 });
        assert_eq!(with(not_a_string.clone()), not_a_string);
    }
}
