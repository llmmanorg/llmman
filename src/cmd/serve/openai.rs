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
use super::sched::{begin_activity, ActivityGuard};
use super::types::*;
use super::{
    aggregation, backend_wire_model, ensure_model, explain_missing_route, is_responses_route,
    provider_compat, proxy, proxy_rewriting_model, relay, relay_chat_upstream, remote_responses,
    request_pin, sanitize_responses_request, send_chat_completion, send_with_hybrid_fallback,
    stream_rewriting_model, strip_llama_fields, unsupported_on_wire, AppError, AppState, Engine,
    Target, CHAT_COMPLETIONS_ROUTE, RESPONSES_ROUTE,
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

/// Mirrors a local chat completion's `reasoning_effort` into the
/// `chat_template_kwargs` llama-server's templates read, as
/// `think_to_chat_template_kwargs` does for Ollama's `think`: `none` is
/// thinking off, a level is thinking on at that depth. Recent
/// llama-server reads `reasoning_effort` itself; older builds and vLLM
/// read only the kwargs. The caller's own kwargs win key by key.
/// [`provider_compat`] is the reverse, for a provider.
pub(super) fn apply_reasoning_effort(req: &mut serde_json::Value) {
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
    match &target {
        Target::Remote(remote) if llama_path == CHAT_COMPLETIONS_ROUTE => {
            provider_compat(remote, &mut req)
        }
        Target::Remote(_) => strip_llama_fields(&mut req),
        _ => {
            apply_default_repeat_penalty(&mut req);
            if llama_path == CHAT_COMPLETIONS_ROUTE {
                apply_reasoning_effort(&mut req);
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
pub(super) async fn forward_openai_request(
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
pub(super) fn mlx_embeddings_unsupported_response(canonical_model: &str) -> Response {
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
pub(super) fn omni_image_request(
    mut req: serde_json::Value,
) -> Result<(serde_json::Value, bool), String> {
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
pub(super) fn omni_video_fields(req: &serde_json::Value) -> Vec<(String, String)> {
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
pub(super) fn multipart_form(fields: &[(String, String)]) -> (Vec<u8>, String) {
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
pub(super) async fn multipart_text_field(
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
