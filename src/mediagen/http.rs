//! What the in-binary workers (`mediagen::server`, `speech::server`) share
//! on the wire: the discovery routes the daemon polls after a spawn
//! (`/health`, `/v1/models`, `/props`), the refusal of the text routes,
//! the OpenAI error shape and the serve loop.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};

/// Seconds since the epoch: the `created` field of a model listing.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `{"error": {"code", "message", "type"}}` with `status`, the shape
/// llama-server and the OpenAI API answer with.
pub fn error_response(status: StatusCode, ty: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": status.as_u16(), "message": message, "type": ty}})),
    )
        .into_response()
}

/// What a worker tells the daemon about its one model.
pub struct Discovery {
    pub model_name: String,
    pub model_path: String,
    /// `capabilities` of the model listing: `image`, `video`, `audio`.
    pub capabilities: Vec<&'static str>,
    /// `details.family` of the listing.
    pub family: &'static str,
    /// `modalities` of `/props`.
    pub modalities: Value,
    /// Further `/props` members, such as `default_generation_settings`.
    pub extra_props: Map<String, Value>,
}

impl Discovery {
    /// `/v1/models` and `/models`: the ollama and OpenAI listings in one.
    fn models(&self) -> Value {
        json!({
            "models": [{
                "name": self.model_name, "model": self.model_name, "modified_at": "", "size": "", "digest": "",
                "type": "model", "description": "", "tags": [""], "capabilities": self.capabilities, "parameters": "",
                "details": {"parent_model": "", "format": "gguf", "family": self.family, "families": [self.family],
                            "parameter_size": "", "quantization_level": ""}
            }],
            "object": "list",
            "data": [{"id": self.model_name, "object": "model", "created": now(), "owned_by": "llmman", "capabilities": self.capabilities}]
        })
    }

    /// `/props`, llama-server's shape.
    fn props(&self) -> Value {
        let mut props = json!({
            "model_alias": self.model_name,
            "model_path": self.model_path,
            "modalities": self.modalities,
            "build_info": format!("llmman {}", env!("CARGO_PKG_VERSION")),
        });
        if let Some(object) = props.as_object_mut() {
            object.extend(self.extra_props.clone());
        }
        props
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(d): State<Arc<Discovery>>) -> Json<Value> {
    Json(d.models())
}

async fn props(State(d): State<Arc<Discovery>>) -> Json<Value> {
    Json(d.props())
}

/// `/health`, `/v1/models`, `/models` and `/props` over `discovery`, to
/// merge into a worker's own routes.
pub fn discovery_routes(discovery: Discovery) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/props", get(props))
        .with_state(Arc::new(discovery))
}

/// The text routes llama-server would serve, each answering a 400 with
/// `message`, which names the routes this worker does serve.
pub fn text_routes_refused(message: &'static str) -> Router {
    let refuse = move || async move {
        error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    };
    Router::new()
        .route("/v1/chat/completions", post(refuse))
        .route("/completion", post(refuse))
        .route("/v1/completions", post(refuse))
        .route("/v1/embeddings", post(refuse))
}

/// Serves `router` on `addr` until the listener fails or the parent goes
/// away, logging as `worker`.
pub async fn serve(worker: &str, router: Router, addr: std::net::SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "[llmman] {worker}: listening on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovery() -> Discovery {
        let mut extra = Map::new();
        extra.insert("default_generation_settings".into(), json!({"steps": 4}));
        Discovery {
            model_name: "m".into(),
            model_path: "/p/m.gguf".into(),
            capabilities: vec!["audio"],
            family: "canary",
            modalities: json!({"audio": true}),
            extra_props: extra,
        }
    }

    #[test]
    fn the_listing_carries_the_name_capabilities_and_family_in_both_shapes() {
        let v = discovery().models();
        assert_eq!(v["models"][0]["name"], "m");
        assert_eq!(v["models"][0]["capabilities"], json!(["audio"]));
        assert_eq!(v["models"][0]["details"]["family"], "canary");
        assert_eq!(v["data"][0]["id"], "m");
        assert_eq!(v["data"][0]["capabilities"], json!(["audio"]));
    }

    #[test]
    fn props_carry_the_path_modalities_and_extra_members() {
        let v = discovery().props();
        assert_eq!(v["model_alias"], "m");
        assert_eq!(v["model_path"], "/p/m.gguf");
        assert_eq!(v["modalities"]["audio"], true);
        assert_eq!(v["default_generation_settings"]["steps"], 4);
        assert!(v["build_info"].as_str().unwrap().starts_with("llmman "));
    }

    #[tokio::test]
    async fn an_error_renders_the_openai_shape() {
        let resp = error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "too big",
        );
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], 413);
        assert_eq!(v["error"]["message"], "too big");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }
}
