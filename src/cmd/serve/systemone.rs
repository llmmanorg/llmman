//! `POST /v1/systemone` on the daemon. llama-server cannot give the
//! probability of a chosen token, so a backend per model (`llmman serve
//! MODEL --port`, see `ggml_backend`) reads the GGUF on ggml and this
//! relays to it. Backends live in `running` beside the chat models, so
//! keep_alive, eviction, the model cap and shutdown apply to them.

use anyhow::Context;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use tokio::time::Instant;

use super::relay::proxy;
use super::sched::default_keep_alive;
use super::{
    acquire_load_lock, canonical_ref, check_running, enforce_max_loaded_models, find_free_port,
    load_identity, manifest_digest_and_size, now_rfc3339, pull_if_missing, resolve_model,
    spawn_ggml, try_admit, usage, wait_for_ready, ActivityGuard, AppError, AppState, ModelPath,
    RunningModel, Target,
};
use crate::mediagen::server::Error as Refusal;
use crate::systemone::{request, server, template_kwargs};

/// Added to a model's name in `running`. Its entry has no digest, so the
/// chat model's is never mistaken for it.
const SUFFIX: &str = "#systemone";

impl From<AppError> for Refusal {
    fn from(e: AppError) -> Refusal {
        let msg = format!("{:#}", e.0);
        match e.1 {
            StatusCode::SERVICE_UNAVAILABLE => Refusal::Unavailable(msg),
            s if s.is_client_error() => Refusal::Invalid(msg),
            _ => Refusal::Failed(msg),
        }
    }
}

/// Every refusal is in OpenAI's error envelope, like the backend's.
pub(super) async fn handle_systemone(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, Refusal> {
    // what cannot be answered should not cost a model load
    let req = match request::parse(&body) {
        Ok(r) => r,
        Err(problems) => return Ok(server::invalid(&problems)),
    };
    template_kwargs(&req)?;
    let (key, port, guard) = backend_for(&state, &req.model).await?;
    let target = Target::Local(port);
    usage::note_target(&key, &target);
    Ok(proxy(
        &state.0.client,
        &target,
        "/v1/systemone",
        &headers,
        body,
        guard,
    )
    .await?)
}

/// The claimed backend of `key`, if it is running.
async fn running(state: &AppState, key: &str) -> Option<(String, u16, ActivityGuard)> {
    let (port, guard) = check_running(state, &format!("{key}{SUFFIX}")).await?;
    Some((key.to_string(), port, guard))
}

/// The backend of `model_ref`, started (and the model pulled) if need be.
async fn backend_for(
    state: &AppState,
    model_ref: &str,
) -> Result<(String, u16, ActivityGuard), Refusal> {
    if crate::providers::is_remote_ref(model_ref) || crate::hybrid::split_ref(model_ref).is_some() {
        return Err(Refusal::Invalid(format!(
            "/v1/systemone reads the token probabilities of a local GGUF model, and \
             {model_ref} is served by a provider or as a hybrid pair"
        )));
    }
    let invalid = |e: crate::shortnames::InvalidReference| Refusal::Invalid(e.to_string());
    let store = &state.0.store_path;
    let resolved = crate::shortnames::resolve_ollama_api(model_ref).map_err(invalid)?;
    let key = canonical_ref(store, &crate::storage::default_tag(&resolved));
    if let Some(found) = running(state, &key).await {
        return Ok(found);
    }

    let _queue = try_admit(state.0.max_queue)?;
    // one load of a model at a time, chat's and this one's alike
    let _lock = acquire_load_lock(&load_identity(store, model_ref).map_err(invalid)?).await;
    // a request ahead of us may have loaded it while we waited
    if let Some(found) = running(state, &key).await {
        return Ok(found);
    }
    let started = Instant::now();
    pull_if_missing(state, &key).await?;
    // the pull may settle on a more specific name, which may be running
    let key = canonical_ref(store, &key);
    if let Some(found) = running(state, &key).await {
        return Ok(found);
    }
    let path = resolve_model(store, &state.0.cache_path, &key)
        .with_context(|| format!("resolve model {key}"))?;
    if !matches!(path, ModelPath::Gguf(..)) {
        return Err(Refusal::Invalid(format!(
            "/v1/systemone reads the token probabilities of a GGUF model, which {key} \
             is not: only llama.cpp's libraries can be asked for them"
        )));
    }

    let mut pending = enforce_max_loaded_models(state, state.0.max_loaded_models).await?;
    let port = find_free_port()?;
    eprintln!("[llmman] loading the System One backend of {key} on port {port}");
    let ociman = state.0.runtime.ociman().await?;
    let (mut process, tail) = spawn_ggml(state, ociman, &key, port).await?;
    wait_for_ready(&state.0.client, port, &mut process, Some(&tail))
        .await
        .with_context(|| format!("the System One backend of {key}"))?;
    let entry = format!("{key}{SUFFIX}");
    let mut mgr = state.0.manager.lock().await;
    mgr.running.insert(
        entry.clone(),
        RunningModel {
            process,
            port,
            digest: String::new(),
            size: manifest_digest_and_size(store, &key).1,
            started_at: now_rfc3339(),
            last_active: Instant::now(),
            last_active_wall: chrono::Utc::now(),
            backend_model_path: None,
            context_window: None,
            keep_alive: default_keep_alive(),
            in_flight: 1,
        },
    );
    pending.release_into(&mut mgr);
    drop(mgr);
    crate::metrics::record_model_load(&entry, started.elapsed());
    let guard = ActivityGuard::new(state, &entry);
    Ok((key, port, guard))
}

#[cfg(test)]
mod tests {
    use super::super::check_running_by_digest;
    use super::super::test_support::{running_model_fixture, serve_router, test_state};
    use super::*;
    use std::time::Duration;

    async fn post(body: &str) -> (u16, serde_json::Value) {
        let base = serve_router(test_state()).await;
        let reply = reqwest::Client::new()
            .post(format!("{base}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .unwrap();
        (reply.status().as_u16(), reply.json().await.unwrap())
    }

    #[tokio::test]
    async fn an_invalid_request_is_a_422_with_fastapis_detail_before_any_model_loads() {
        let (status, body) = post(r#"{"model": "m", "questions": {"q": {"type": "noul"}}}"#).await;
        assert_eq!(status, 422);
        let locs: Vec<_> = body["detail"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["loc"].to_string())
            .collect();
        assert_eq!(locs, [r#"["body","state"]"#, r#"["body","questions","q"]"#]);
    }

    #[tokio::test]
    async fn a_model_that_is_not_local_is_refused_in_the_openai_envelope() {
        let (status, body) = post(
            r#"{"state": "s", "model": "llmman.provider/anthropic/claude-sonnet-5-5",
                "questions": {"q": {"type": "noul", "instructions": "i"}}}"#,
        )
        .await;
        assert_eq!(status, 400);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("local GGUF"));
    }

    /// An unknown model would be a failed pull (500) if anything got that far.
    #[tokio::test]
    async fn thinking_is_refused_before_the_model_is_pulled() {
        let (status, body) = post(
            r#"{"state": "s", "model": "docker.io/ai/not-a-model:1",
                "chat_template_kwargs": {"enable_thinking": true},
                "questions": {"q": {"type": "noul", "instructions": "i"}}}"#,
        )
        .await;
        assert_eq!(status, 400, "{body}");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("enable_thinking"));
    }

    #[tokio::test]
    async fn a_running_backend_is_claimed_and_never_taken_for_the_chat_model() {
        let state = test_state();
        let mut backend = running_model_fixture(None, Duration::ZERO, 0);
        backend.port = 4321;
        state
            .0
            .manager
            .lock()
            .await
            .running
            .insert(format!("docker.io/ai/m:1{SUFFIX}"), backend);

        let (key, port, _guard) = running(&state, "docker.io/ai/m:1").await.unwrap();
        assert_eq!((key.as_str(), port), ("docker.io/ai/m:1", 4321));
        assert_eq!(
            state.0.manager.lock().await.running[&format!("docker.io/ai/m:1{SUFFIX}")].in_flight,
            1
        );
        // the chat model of the same name is not it, by name or by content
        assert!(check_running(&state, "docker.io/ai/m:1").await.is_none());
        assert!(
            check_running_by_digest(&state, "docker.io/ai/m:1", "sha256:abc")
                .await
                .is_none()
        );
    }
}
