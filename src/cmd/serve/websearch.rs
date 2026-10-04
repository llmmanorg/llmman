//! `GET /llmman/websearch?q=<query>[&limit=<n>]`: the web UI's "Search the
//! web". The daemon asks [Exa](https://exa.ai), so the key stays in
//! `EXA_API_KEY` or `llmman.conf` rather than a browser. The query is the
//! only thing that leaves the machine.

use std::time::Duration;

use anyhow::anyhow;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::{Deserialize, Serialize};

use super::{AppError, AppState};

const EXA_URL: &str = "https://api.exa.ai/search";
const DEFAULT_LIMIT: u32 = 5;
/// Each source is prompt for the model, so more than this is noise.
const MAX_LIMIT: u32 = 10;
const SNIPPET_CHARS: usize = 1500;
const TIMEOUT: Duration = Duration::from_secs(30);
const ERROR_CHARS: usize = 300;

#[derive(Deserialize)]
pub(super) struct Params {
    #[serde(default)]
    q: String,
    /// Clamped rather than refused, as `/llmman/search`'s.
    limit: Option<u32>,
}

#[derive(Serialize, Debug, PartialEq)]
pub(super) struct Source {
    title: String,
    url: String,
    /// `YYYY-MM-DD`, when Exa has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    published: Option<String>,
    /// The page's passages matching the query.
    snippet: String,
}

#[derive(Serialize)]
pub(super) struct Response {
    results: Vec<Source>,
}

pub(super) async fn handle_websearch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<Params>,
) -> Result<Json<Response>, AppError> {
    let query = params.q.trim();
    if query.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "search query `q` must not be empty",
        ));
    }
    // The same gate as a provider's key: a page on another site must not
    // be able to spend the operator's Exa quota.
    if !super::daemon_key_spendable(&state, Some(&headers)) {
        return Err(AppError::status(
            StatusCode::FORBIDDEN,
            "web search is refused for a cross-site or unauthenticated remote request",
        ));
    }
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let key = crate::config::websearch_api_key();
    let results = search(EXA_URL, key.as_deref(), query, limit).await?;
    Ok(Json(Response { results }))
}

fn bad_gateway(e: impl std::fmt::Display) -> AppError {
    AppError(anyhow!("Exa: {e}"), StatusCode::BAD_GATEWAY)
}

/// Asks the Exa-compatible `endpoint`. No `key` is a 503 saying how to
/// set one. Split from the handler so a test can use a local server.
async fn search(
    endpoint: &str,
    key: Option<&str>,
    query: &str,
    limit: u32,
) -> Result<Vec<Source>, AppError> {
    let Some(key) = key else {
        return Err(AppError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "web search is not configured: set EXA_API_KEY, or run \
             `llmman config set websearch.api_key <key>` \
             (keys are at https://dashboard.exa.ai/api-keys)",
        ));
    };
    // No redirects: reqwest would carry `x-api-key` to another host.
    let client = crate::auth::trusted_client()
        .and_then(|b| {
            Ok(b.redirect(reqwest::redirect::Policy::none())
                .timeout(TIMEOUT)
                .build()?)
        })
        .map_err(bad_gateway)?;
    let response = client
        .post(endpoint)
        .header("x-api-key", key)
        .json(&serde_json::json!({
            "query": query,
            "type": "auto",
            "numResults": limit,
            "contents": { "highlights": { "maxCharacters": SNIPPET_CHARS } },
        }))
        .send()
        .await
        .map_err(|e| bad_gateway(e.without_url()))?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .map_err(|e| bad_gateway(e.without_url()))?;
    if !status.is_success() {
        return Err(bad_gateway(upstream_error(status, &body, key)));
    }
    sources(&body, limit as usize).map_err(bad_gateway)
}

/// Exa's `{"error": "..."}`, else the body's start, with `key` scrubbed.
fn upstream_error(status: reqwest::StatusCode, body: &[u8], key: &str) -> String {
    #[derive(Deserialize)]
    struct Envelope {
        error: String,
    }
    let message = serde_json::from_slice::<Envelope>(body)
        .map(|e| e.error)
        .unwrap_or_else(|_| String::from_utf8_lossy(body).into_owned());
    let message: String = message
        .trim()
        .replace(key, "<redacted>")
        .chars()
        .take(ERROR_CHARS)
        .collect();
    if message.is_empty() {
        status.to_string()
    } else {
        format!("{status}: {message}")
    }
}

#[derive(Deserialize)]
struct ExaResponse {
    #[serde(default)]
    results: Vec<ExaResult>,
}

#[derive(Deserialize)]
struct ExaResult {
    title: Option<String>,
    url: String,
    #[serde(default, rename = "publishedDate")]
    published_date: Option<String>,
    #[serde(default)]
    highlights: Vec<String>,
}

/// At most `limit` of Exa's results, in its order. One whose URL is not
/// `http(s)` is dropped: the page links every source.
fn sources(body: &[u8], limit: usize) -> Result<Vec<Source>, serde_json::Error> {
    let reply: ExaResponse = serde_json::from_slice(body)?;
    Ok(reply
        .results
        .into_iter()
        .filter(|r| {
            reqwest::Url::parse(&r.url).is_ok_and(|u| matches!(u.scheme(), "http" | "https"))
        })
        .take(limit)
        .map(|r| Source {
            title: r
                .title
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| r.url.clone()),
            // `2026-09-21T00:00:00.000Z` is `2026-09-21`.
            published: r
                .published_date
                .map(|d| d.split('T').next().unwrap_or(&d).to_string())
                .filter(|d| !d.is_empty()),
            snippet: r
                .highlights
                .iter()
                .map(|h| h.trim())
                .filter(|h| !h.is_empty())
                .collect::<Vec<_>>()
                .join(" … ")
                .chars()
                .take(SNIPPET_CHARS)
                .collect(),
            url: r.url,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::routing::post;
    use axum::Router;
    use serde_json::{json, Value};

    use super::super::test_support::{serve_router, test_state};
    use super::*;

    type Seen = Arc<Mutex<Vec<(String, Value)>>>;

    #[tokio::test]
    async fn an_empty_query_is_a_bad_request() {
        let url = serve_router(test_state()).await;
        for query in ["", "?q=", "?q=%20%20"] {
            let r = reqwest::get(format!("{url}/llmman/websearch{query}"))
                .await
                .unwrap();
            assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST, "{query}");
            let body: Value = r.json().await.unwrap();
            assert!(body["error"].as_str().unwrap().contains("`q`"), "{body}");
        }
    }

    /// Refused before any key is read, so no Exa quota is at stake.
    #[tokio::test]
    async fn a_cross_site_request_is_refused() {
        let url = serve_router(test_state()).await;
        let r = reqwest::Client::new()
            .get(format!("{url}/llmman/websearch?q=rust"))
            .header("sec-fetch-site", "cross-site")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn without_a_key_it_says_how_to_configure_one() {
        let err = search("http://127.0.0.1:1/never-asked", None, "rust", 5)
            .await
            .unwrap_err();
        assert_eq!(err.1, StatusCode::SERVICE_UNAVAILABLE);
        let message = format!("{:#}", err.0);
        assert!(message.contains("EXA_API_KEY"), "{message}");
        assert!(message.contains("websearch.api_key"), "{message}");
    }

    /// A stand-in Exa: records the key header and body, answers `reply`.
    async fn fake_exa(status: u16, reply: Value) -> (String, Seen) {
        let seen = Seen::default();
        let app = Router::new()
            .route(
                "/search",
                post(
                    move |State(seen): State<Seen>,
                          headers: axum::http::HeaderMap,
                          body: axum::Json<Value>| {
                        let reply = reply.clone();
                        async move {
                            let key = headers["x-api-key"].to_str().unwrap().to_string();
                            seen.lock().unwrap().push((key, body.0));
                            (StatusCode::from_u16(status).unwrap(), axum::Json(reply))
                        }
                    },
                ),
            )
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://127.0.0.1:{}/search", addr.port()), seen)
    }

    #[tokio::test]
    async fn it_asks_exa_for_highlights_and_returns_sources_in_order() {
        let (endpoint, seen) = fake_exa(
            200,
            json!({ "results": [
                {
                    "title": " Rust 1.99 released ",
                    "url": "https://blog.rust-lang.org/1.99",
                    "publishedDate": "2026-09-21T00:00:00.000Z",
                    "highlights": ["First passage.", "  ", "Second passage."]
                },
                { "url": "https://example.com/untitled", "highlights": [] },
                { "title": "Script", "url": "javascript:alert(1)", "highlights": ["x"] },
                { "title": "Third", "url": "https://c.test", "highlights": [] }
            ] }),
        )
        .await;

        let results = search(&endpoint, Some("exa-key"), "rust release", 2)
            .await
            .unwrap();
        assert_eq!(
            results,
            vec![
                Source {
                    title: "Rust 1.99 released".into(),
                    url: "https://blog.rust-lang.org/1.99".into(),
                    published: Some("2026-09-21".into()),
                    snippet: "First passage. … Second passage.".into(),
                },
                // No title: the URL. Capped at the limit: no "Third".
                Source {
                    title: "https://example.com/untitled".into(),
                    url: "https://example.com/untitled".into(),
                    published: None,
                    snippet: String::new(),
                },
            ]
        );

        let seen = seen.lock().unwrap();
        let (key, body) = &seen[0];
        assert_eq!(key, "exa-key");
        assert_eq!(body["query"], "rust release");
        assert_eq!(body["numResults"], 2);
        assert_eq!(
            body["contents"]["highlights"]["maxCharacters"],
            SNIPPET_CHARS
        );
    }

    #[tokio::test]
    async fn an_upstream_error_is_a_bad_gateway_that_does_not_echo_the_key() {
        let (endpoint, _) = fake_exa(
            401,
            json!({ "error": "Invalid API key exa-secret-key", "tag": "INVALID_API_KEY" }),
        )
        .await;
        let err = search(&endpoint, Some("exa-secret-key"), "rust", 5)
            .await
            .unwrap_err();
        assert_eq!(err.1, StatusCode::BAD_GATEWAY);
        let message = format!("{:#}", err.0);
        assert!(message.contains("Invalid API key <redacted>"), "{message}");
        assert!(message.contains("401"), "{message}");

        // Unreachable: the same status, with no URL in it.
        let err = search("http://127.0.0.1:1/search", Some("k"), "rust", 5)
            .await
            .unwrap_err();
        assert_eq!(err.1, StatusCode::BAD_GATEWAY);
        assert!(!format!("{:#}", err.0).contains("127.0.0.1"), "{err:?}");
    }

    /// A redirect is not followed, so `x-api-key` stays with Exa.
    #[tokio::test]
    async fn a_redirect_is_not_followed() {
        let (elsewhere, seen) = fake_exa(200, json!({ "results": [] })).await;
        let app = Router::new().route(
            "/search",
            post(move || {
                let to = elsewhere.clone();
                async move { (StatusCode::TEMPORARY_REDIRECT, [("location", to)]) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let err = search(
            &format!("http://127.0.0.1:{}/search", addr.port()),
            Some("k"),
            "rust",
            5,
        )
        .await
        .unwrap_err();
        assert_eq!(err.1, StatusCode::BAD_GATEWAY);
        assert!(seen.lock().unwrap().is_empty(), "the redirect was followed");
    }

    #[test]
    fn a_reply_that_is_not_a_search_result_is_an_error() {
        assert!(sources(b"<html>", 5).is_err());
        assert_eq!(sources(b"{}", 5).unwrap(), vec![]);
        // Snippets are cut on a character boundary, not a byte.
        let long = "é".repeat(SNIPPET_CHARS + 50);
        let body = json!({ "results": [{ "url": "https://a.test", "highlights": [long] }] });
        let out = sources(body.to_string().as_bytes(), 5).unwrap();
        assert_eq!(out[0].snippet.chars().count(), SNIPPET_CHARS);
    }
}
