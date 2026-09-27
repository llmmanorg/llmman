//! `GET /llmman/search?q=<query>[&limit=<n>]`: `llmman search` over HTTP,
//! for the web UI's Pull dialog. The same Docker Hub and Hugging Face
//! rows, in the same order, each `name` ready for `/api/pull`.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use axum::extract::Query;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use super::AppError;
use crate::cmd::search::{self, Hit, ModelCard, DEFAULT_LIMIT, MAX_LIMIT};

#[derive(Deserialize)]
pub(super) struct SearchParams {
    #[serde(default)]
    q: String,
    /// Rows per registry, as `--limit`; clamped rather than refused.
    limit: Option<u32>,
}

#[derive(Serialize)]
pub(super) struct SearchResponse {
    models: Vec<Hit>,
}

/// A registry that cannot be reached is the daemon's upstream failing,
/// so both registries failing is a 502, not a 500.
pub(super) async fn handle_search(
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, AppError> {
    let query = params.q.trim();
    if query.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "search query `q` must not be empty",
        ));
    }
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let models = search::search(query, limit, None)
        .await
        .map_err(|e| AppError(e, StatusCode::BAD_GATEWAY))?;
    Ok(Json(SearchResponse { models }))
}

#[derive(Deserialize)]
pub(super) struct PopularParams {
    limit: Option<u32>,
}

/// `GET /llmman/search/popular[?limit=<n>]`: what to show before a search,
/// the same row shape as `/llmman/search`.
pub(super) async fn handle_popular(
    Query(params): Query<PopularParams>,
) -> Result<Json<SearchResponse>, AppError> {
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let models = search::popular(limit)
        .await
        .map_err(|e| AppError(e, StatusCode::BAD_GATEWAY))?;
    Ok(Json(SearchResponse { models }))
}

#[derive(Deserialize)]
pub(super) struct NameParam {
    #[serde(default)]
    name: String,
}

/// `GET /llmman/search/model?name=<a search row's name>`: its tags with
/// their sizes, and the repo's facts, for the Models page's card.
pub(super) async fn handle_model(
    Query(params): Query<NameParam>,
) -> Result<Json<ModelCard>, AppError> {
    let name = params.name.trim();
    if name.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "`name` must not be empty",
        ));
    }
    let card = search::model_card(name)
        .await
        .map_err(|e| AppError(e, StatusCode::BAD_GATEWAY))?;
    Ok(Json(card))
}

/// Owner → avatar image URL, so a page of rows asks each registry once.
/// Only answers are kept (an upstream error is asked again next time),
/// and not forever: the names come from clients.
static AVATARS: LazyLock<Mutex<HashMap<String, Option<String>>>> = LazyLock::new(Default::default);
const AVATARS_KEPT: usize = 4096;

/// `GET /llmman/search/avatar?name=<a search row's name>`: a redirect to
/// the owner's avatar image, or 404 for none (the page draws initials).
pub(super) async fn handle_avatar(Query(params): Query<NameParam>) -> Response {
    let owner = params
        .name
        .rsplit_once('/')
        .map(|(o, _)| o.to_owned())
        .unwrap_or_default();
    let cached = AVATARS.lock().unwrap().get(&owner).cloned();
    let url = match cached {
        Some(url) => url,
        None => match search::avatar(&params.name).await {
            Ok(url) => {
                let mut avatars = AVATARS.lock().unwrap();
                if avatars.len() >= AVATARS_KEPT {
                    avatars.clear();
                }
                avatars.insert(owner, url.clone());
                url
            }
            Err(_) => None,
        },
    };
    match url {
        Some(url) => (
            StatusCode::FOUND,
            [
                (header::LOCATION, url),
                (header::CACHE_CONTROL, "max-age=86400".into()),
            ],
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{serve_router, test_state};

    /// Refused before either registry is asked, so this needs no network.
    #[tokio::test]
    async fn an_empty_query_is_a_bad_request() {
        let url = serve_router(test_state()).await;
        for query in ["", "?q=", "?q=%20%20"] {
            let r = reqwest::get(format!("{url}/llmman/search{query}"))
                .await
                .unwrap();
            assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST, "{query}");
            let body: serde_json::Value = r.json().await.unwrap();
            assert!(
                body["error"].as_str().unwrap().contains("`q`"),
                "{query}: {body}"
            );
        }
    }

    /// The row shape the web UI reads: a missing count is `null`, not 0.
    #[test]
    fn a_hit_serializes_with_its_reference_and_nullable_counts() {
        let hit = crate::cmd::search::Hit {
            name: "hf.co/unsloth/Qwen3.5-0.8B-GGUF".into(),
            pulls: None,
            likes: Some(7),
            updated: Some("2026-09-01T00:00:00Z".into()),
        };
        assert_eq!(
            serde_json::to_value(super::SearchResponse { models: vec![hit] }).unwrap(),
            serde_json::json!({"models": [{
                "name": "hf.co/unsloth/Qwen3.5-0.8B-GGUF",
                "pulls": null,
                "likes": 7,
                "updated": "2026-09-01T00:00:00Z",
            }]})
        );
    }
}
