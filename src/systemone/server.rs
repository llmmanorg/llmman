//! The System One backend the daemon spawns for a GGUF: `/v1/systemone` and `/health`.

use std::sync::{Arc, Mutex, PoisonError};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use super::engine::Engine;
use super::request::{self, Json as Ordered};
use super::{decide, Error};
use crate::mediagen::server::Error as Http;

/// Request body limit; a state has to fit the model's context anyway.
pub const BODY_LIMIT: usize = 32 << 20;

struct Server {
    engine: Mutex<Engine>,
    /// The model the answers name, whatever the request called it.
    model: String,
}

pub fn router(engine: Engine, model: String) -> Router {
    let server = Arc::new(Server {
        engine: Mutex::new(engine),
        model,
    });
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(server)
}

/// An invalid request: FastAPI's 422 `detail` list.
pub fn invalid(problems: &[request::Problem]) -> Response {
    let detail: Vec<_> = problems.iter().map(request::Problem::to_json).collect();
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({ "detail": detail })),
    )
        .into_response()
}

async fn systemone(State(server): State<Arc<Server>>, body: Bytes) -> Response {
    let request = match request::parse(&body) {
        Ok(r) => r,
        Err(problems) => return invalid(&problems),
    };
    let served = server.clone();
    let decision = tokio::task::spawn_blocking(move || {
        let mut engine = served.engine.lock().unwrap_or_else(PoisonError::into_inner);
        decide(&mut *engine, &request)
    })
    .await;
    match decision {
        Ok(Ok(d)) => Json(Ordered::Obj(vec![
            ("model".into(), Ordered::Str(server.model.clone())),
            ("answers".into(), Ordered::Obj(d.answers)),
            (
                "usage".into(),
                Ordered::Obj(vec![
                    ("input_tokens".into(), Ordered::Num(d.input_tokens.into())),
                    ("output_tokens".into(), Ordered::Num(0.into())),
                ]),
            ),
        ]))
        .into_response(),
        Ok(Err(e)) => Http::from(e).into_response(),
        Err(e) => Http::Failed(format!("the read panicked: {e}")).into_response(),
    }
}

impl From<Error> for Http {
    fn from(e: Error) -> Http {
        match e {
            Error::Refused(m) => Http::Invalid(m),
            Error::Failed(m) => Http::Failed(m),
        }
    }
}
