//! `GET /muse-code/models`: the model catalog Muse Code fetches from its
//! provider's origin before its first request, and exits without.
//! `llmman launch muse` points that provider at the daemon.
//!
//! The catalog lists what `/v1/models` lists. Muse Code hides a row unless
//! it says `is_hidden: false`.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use serde_json::{json, Value};

use super::openai::openai_model_list;
use super::{AppError, AppState};

pub(super) const MUSE_MODELS_ROUTE: &str = "/muse-code/models";

pub(super) async fn handle_muse_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AppError> {
    let list = openai_model_list(&state, &headers).await?;
    Ok(Json(muse_catalog(&list)))
}

/// `list`, a `/v1/models` body, as Muse Code's catalog.
fn muse_catalog(list: &Value) -> Value {
    let data: Vec<Value> = list["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(muse_row)
        .collect();
    json!({ "data": data })
}

/// One `/v1/models` entry as a visible row. Its `limit.context` is the
/// window the daemon serves the model at (`meta.n_ctx`), left out when that
/// is not known.
fn muse_row(model: &Value) -> Option<Value> {
    let id = model["id"].as_str()?;
    let mut metadata = json!({ "is_hidden": false });
    if let Some(window) = model["meta"]["n_ctx"].as_u64() {
        metadata["limit"] = json!({ "context": window });
    }
    Some(json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": "llmman",
        "metadata": { "muse-code": metadata },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_listed_model_is_a_visible_row_with_its_served_window() {
        let list = json!({
            "object": "list",
            "data": [
                {
                    "id": "docker.io/ai/qwen3.5:0.8b",
                    "object": "model",
                    "status": { "value": "loaded" },
                    "meta": { "n_ctx": 32768, "n_ctx_train": 262144 }
                },
                { "id": "hf.co/org/model:latest", "object": "model" }
            ]
        });
        let catalog = muse_catalog(&list);
        let rows = catalog["data"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["id"], "docker.io/ai/qwen3.5:0.8b");
        let meta = &rows[0]["metadata"]["muse-code"];
        assert_eq!(meta["is_hidden"], false);
        assert_eq!(meta["limit"]["context"], 32768);
        // No served window: no limit, not the trained one.
        assert_eq!(rows[1]["metadata"]["muse-code"]["is_hidden"], false);
        assert!(rows[1]["metadata"]["muse-code"].get("limit").is_none());
    }

    /// Muse Code exits on a body without `data`.
    #[test]
    fn an_empty_list_is_an_empty_catalog_not_a_missing_one() {
        assert_eq!(
            muse_catalog(&json!({"object": "list", "data": []})),
            json!({"data": []})
        );
        assert_eq!(muse_catalog(&json!({})), json!({"data": []}));
    }

    #[test]
    fn an_entry_without_an_id_is_left_out() {
        let catalog = muse_catalog(&json!({"data": [{"object": "model"}, {"id": "m"}]}));
        let ids: Vec<&str> = catalog["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["m"]);
    }
}
