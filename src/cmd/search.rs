//! `llmman search` — find models on Docker Hub and Hugging Face.
//!
//! Both registries are queried at once; Docker Hub's rows come first,
//! since `docker.io/ai/` is where a bare `llmman run <name>` already
//! looks. Every NAME printed is fully qualified so it pastes straight
//! into `run`/`pull` (a bare `ai/qwen3` would resolve to hf.co).
//!
//! Docker Hub is queried through its website's search API
//! (`hub.docker.com/api/search/v4`, no credentials). Hub reports the
//! manifest media types pushed to each repo; only repos carrying a CNCF
//! ModelPack manifest are listed, since that is the format `pull` reads.
//! Repos published solely in Docker Model Runner's own format
//! (`application/vnd.docker.ai.model.config.v0.1+json`) are left out.
//! Hugging Face goes through `hf::api::search_models`, with the user's
//! token when configured.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::Deserialize;

use crate::fmt::{human_count, relative_time_rfc3339};
use crate::hf;

/// Docker Hub's website search API; not part of the OCI registry
/// protocol, so it doesn't go through the Go shim.
const DOCKER_HUB_SEARCH_URL: &str = "https://hub.docker.com/api/search/v4";

/// Docker Hub's search API rejects a page larger than 100.
const MAX_LIMIT: u32 = 100;

/// The manifest media type of a CNCF ModelPack artifact.
const MODELPACK_MANIFEST: &str = "application/vnd.cncf.model.manifest.v1+json";

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Text to match against model names and descriptions
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Maximum number of results to show from each registry
    #[arg(
        long,
        short = 'n',
        default_value_t = 25,
        value_parser = clap::value_parser!(u32).range(1..=MAX_LIMIT as i64)
    )]
    pub limit: u32,

    /// Only search one registry instead of both
    #[arg(long, value_enum)]
    pub registry: Option<Registry>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Registry {
    /// Docker Hub (docker.io)
    #[value(alias = "docker.io", alias = "hub")]
    Docker,
    /// Hugging Face (hf.co)
    #[value(alias = "huggingface", alias = "hf.co")]
    Hf,
}

/// One table row, from either registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Fully qualified reference, ready for `llmman run`/`pull`.
    pub name: String,
    /// Docker Hub pulls; Hugging Face (last-30-day) downloads.
    pub pulls: Option<u64>,
    /// Docker Hub stars; Hugging Face likes.
    pub likes: Option<u64>,
    /// RFC 3339 timestamp of the last update, if reported.
    pub updated: Option<String>,
}

pub fn run(args: &SearchArgs) -> Result<()> {
    let query = args.query.trim();
    anyhow::ensure!(!query.is_empty(), "search query must not be empty");

    let hits = tokio::runtime::Runtime::new()
        .context("start tokio runtime")?
        .block_on(search(query, args.limit, args.registry))?;

    if hits.is_empty() {
        anyhow::bail!("no models found for {query:?}");
    }
    print!("{}", render(&hits));
    Ok(())
}

/// Docker Hub's rows first. With both registries selected, one failing
/// is a warning and the other's rows still print; both failing is an
/// error.
async fn search(query: &str, limit: u32, registry: Option<Registry>) -> Result<Vec<Hit>> {
    let client = hf::api_client()?;
    match registry {
        Some(Registry::Docker) => search_docker_hub(&client, query, limit).await,
        Some(Registry::Hf) => search_hugging_face(&client, query, limit).await,
        None => {
            let (docker, hugging_face) = tokio::join!(
                search_docker_hub(&client, query, limit),
                search_hugging_face(&client, query, limit),
            );
            match (docker, hugging_face) {
                (Ok(mut hits), Ok(more)) => {
                    hits.extend(more);
                    Ok(hits)
                }
                (Ok(hits), Err(e)) => {
                    eprintln!(
                        "[llmman] Hugging Face search failed, showing Docker Hub only: {e:#}"
                    );
                    Ok(hits)
                }
                (Err(e), Ok(hits)) => {
                    eprintln!(
                        "[llmman] Docker Hub search failed, showing Hugging Face only: {e:#}"
                    );
                    Ok(hits)
                }
                (Err(docker), Err(hugging_face)) => Err(docker
                    .context(format!("Hugging Face: {hugging_face:#}"))
                    .context("search failed on both registries")),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Docker Hub
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HubSearchResponse {
    #[serde(default)]
    results: Vec<HubResult>,
}

#[derive(Debug, Deserialize)]
struct HubResult {
    /// `namespace/repo`, or a bare `repo` for the official `library/`
    /// namespace.
    name: String,
    #[serde(default)]
    star_count: u64,
    /// The exact figure; `pull_count` is a display string like `"500K+"`.
    #[serde(default)]
    raw_pull_count: Option<u64>,
    #[serde(default)]
    updated_at: Option<String>,
    #[serde(default)]
    archived: bool,
    /// Every manifest/config media type pushed to the repo.
    #[serde(default)]
    media_types: Vec<String>,
}

impl HubResult {
    /// A repo can hold several formats (Docker's `ai/` models are
    /// published in both), so this is "has one", not "has only this".
    fn is_modelpack(&self) -> bool {
        self.media_types.iter().any(|t| t == MODELPACK_MANIFEST)
    }
}

/// Fetches Hub's largest page regardless of `limit`: `type=model` also
/// matches Model Runner-only repos, which [`docker_hub_hits`] drops (most
/// of them, today), so the cut to `limit` happens after filtering.
async fn search_docker_hub(client: &reqwest::Client, query: &str, limit: u32) -> Result<Vec<Hit>> {
    let size = MAX_LIMIT.to_string();
    let url = reqwest::Url::parse_with_params(
        DOCKER_HUB_SEARCH_URL,
        [
            ("query", query),
            ("type", "model"),
            ("from", "0"),
            ("size", size.as_str()),
        ],
    )
    .context("build Docker Hub search URL")?;
    let parsed: HubSearchResponse = hf::api::get_json(client, url.as_str(), None)
        .await
        .context("Docker Hub model search")?;
    Ok(docker_hub_hits(parsed, limit))
}

fn docker_hub_hits(response: HubSearchResponse, limit: u32) -> Vec<Hit> {
    response
        .results
        .into_iter()
        .filter(|r| !r.archived && r.is_modelpack())
        .take(limit as usize)
        .map(|r| {
            // Hub omits the official namespace; a pullable reference needs it.
            let name = if r.name.contains('/') {
                format!("docker.io/{}", r.name)
            } else {
                format!("docker.io/library/{}", r.name)
            };
            Hit {
                name,
                pulls: r.raw_pull_count,
                likes: Some(r.star_count),
                updated: r.updated_at,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hugging Face
// ---------------------------------------------------------------------------

async fn search_hugging_face(
    client: &reqwest::Client,
    query: &str,
    limit: u32,
) -> Result<Vec<Hit>> {
    let endpoint = hf::hf_endpoint("hf.co");
    let token = hf::token();
    let models = hf::api::search_models(client, &endpoint, query, limit, token.as_deref()).await?;
    Ok(models.into_iter().map(hugging_face_hit).collect())
}

fn hugging_face_hit(model: hf::api::SearchHit) -> Hit {
    Hit {
        name: format!("hf.co/{}", model.id),
        pulls: Some(model.downloads),
        likes: Some(model.likes),
        updated: model.last_modified,
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Same shape as `list`/`ps`: uppercase headers, 4-space gutters, columns
/// as wide as their widest cell, last column unpadded, nothing after the
/// final row.
fn render(hits: &[Hit]) -> String {
    let dash = || "-".to_string();
    let rows: Vec<[String; 4]> = hits
        .iter()
        .map(|h| {
            [
                h.name.clone(),
                h.pulls.map(human_count).unwrap_or_else(dash),
                h.likes.map(|n| n.to_string()).unwrap_or_else(dash),
                h.updated
                    .as_deref()
                    .map(relative_time_rfc3339)
                    .unwrap_or_else(dash),
            ]
        })
        .collect();

    let headers = ["NAME", "PULLS", "LIKES", "UPDATED"];
    let widths: Vec<usize> = (0..headers.len())
        .map(|c| {
            rows.iter()
                .map(|r| r[c].len())
                .max()
                .unwrap_or(0)
                .max(headers[c].len())
        })
        .collect();

    let line = |cells: [&str; 4]| {
        format!(
            "{:<w0$}    {:<w1$}    {:<w2$}    {}\n",
            cells[0],
            cells[1],
            cells[2],
            cells[3],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
        )
    };
    let mut out = line(headers);
    for r in &rows {
        out.push_str(&line([&r[0], &r[1], &r[2], &r[3]]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hub's `type=model` response shape, plus rows that must not print:
    /// Model Runner-only, archived, a container image, and one past
    /// `limit`.
    #[test]
    fn docker_hub_rows_become_pullable_references() {
        let body = r#"{"total": 6, "results": [
            {"name": "ai/qwen3", "short_description": "Qwen3 LLM", "star_count": 210,
             "pull_count": "500K+", "raw_pull_count": 614283,
             "updated_at": "2026-08-17T14:03:18.618971Z", "archived": false,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json",
                             "application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "ai/qwen2.5", "star_count": 13, "raw_pull_count": 163532,
             "media_types": ["application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "official-thing", "star_count": 1,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"]},
            {"name": "ai/old", "archived": true, "star_count": 0,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"]},
            {"name": "library/nginx", "star_count": 99999,
             "media_types": ["application/vnd.docker.distribution.manifest.v2+json"],
             "content_types": ["image"]},
            {"name": "ai/third", "star_count": 0,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"]}
        ]}"#;
        let parsed: HubSearchResponse = serde_json::from_str(body).unwrap();
        let hits = docker_hub_hits(parsed, 2);
        let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
        // A bare name is the official `library/` namespace.
        assert_eq!(
            names,
            ["docker.io/ai/qwen3", "docker.io/library/official-thing"]
        );
        assert_eq!(
            hits[0],
            Hit {
                name: "docker.io/ai/qwen3".into(),
                pulls: Some(614283),
                likes: Some(210),
                updated: Some("2026-08-17T14:03:18.618971Z".into()),
            }
        );
        assert_eq!(hits[1].pulls, None);
        assert_eq!(hits[1].updated, None);
    }

    #[test]
    fn hugging_face_rows_become_pullable_references() {
        let body = r#"[
            {"id": "unsloth/Qwen3-8B-GGUF", "likes": 989, "downloads": 12853002,
             "lastModified": "2025-07-31T10:27:38.000Z"},
            {"id": "someone/bare"}
        ]"#;
        let models: Vec<hf::api::SearchHit> = serde_json::from_str(body).unwrap();
        let hits: Vec<Hit> = models.into_iter().map(hugging_face_hit).collect();
        assert_eq!(
            hits[0],
            Hit {
                name: "hf.co/unsloth/Qwen3-8B-GGUF".into(),
                pulls: Some(12853002),
                likes: Some(989),
                updated: Some("2025-07-31T10:27:38.000Z".into()),
            }
        );
        assert_eq!(hits[1].pulls, Some(0));
        assert_eq!(hits[1].updated, None);
    }

    #[test]
    fn render_aligns_columns_and_dashes_missing_values() {
        let hits = vec![
            Hit {
                name: "docker.io/ai/qwen3".into(),
                pulls: Some(614_283),
                likes: Some(210),
                updated: None,
            },
            Hit {
                name: "hf.co/unsloth/Qwen3-8B-GGUF".into(),
                pulls: Some(12_853_002),
                likes: Some(989),
                updated: None,
            },
        ];
        let out = render(&hits);
        assert_eq!(
            out,
            "NAME                           PULLS     LIKES    UPDATED\n\
             docker.io/ai/qwen3             614.3K    210      -\n\
             hf.co/unsloth/Qwen3-8B-GGUF    12.9M     989      -\n"
        );
    }
}
