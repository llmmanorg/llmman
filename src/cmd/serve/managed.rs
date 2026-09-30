//! Native OAuth forwarding on the daemon's TLS listener. These routes have
//! mandatory daemon-key authentication and loopback peer admission, and bypass
//! prompt history, provider credential fallback, and peer routing.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use super::{auth, AppError};

const CONNECTION_KEY: &str = "x-api-key";
const UPSTREAM_IP: &str = "x-llmman-upstream-ip";
const CAPABILITY: &str = "codex-oauth-forwarding-v2";
const CLAUDE_CAPABILITY: &str = "claude-oauth-forwarding-v2";
const CLAUDE_OAUTH_BETA: &str = "oauth-2025-04-20";
const BODY_LIMIT: usize = 32 * 1024 * 1024;
const ERROR_LIMIT: usize = 1024 * 1024;
/// Upstream transports kept warm, one per (profile, pinned IP). The
/// supervisor pins a handful of resolved addresses, so this is a leak
/// guard rather than an eviction policy: past it the map starts over.
const CLIENT_CACHE_LIMIT: usize = 64;
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Profile {
    Codex,
    Claude,
}

impl Profile {
    fn host(self) -> &'static str {
        match self {
            Self::Codex => "chatgpt.com",
            Self::Claude => "api.anthropic.com",
        }
    }

    fn base(self) -> &'static str {
        match self {
            Self::Codex => "https://chatgpt.com/backend-api/codex",
            Self::Claude => "https://api.anthropic.com/v1",
        }
    }

    fn model_prefix(self) -> &'static str {
        match self {
            Self::Codex => "llmman.provider/openai/",
            Self::Claude => "llmman.provider/anthropic/",
        }
    }

    #[cfg(test)]
    fn route_prefix(self) -> &'static str {
        match self {
            Self::Codex => "/api/codex",
            Self::Claude => "/api/anthropic",
        }
    }
}

/// Validates opt-in before any socket binds. Ordinary daemon auth may be
/// optional on loopback, but delegated OAuth forwarding never is.
pub(super) fn routes(
    config: crate::config::Managed,
    policy: auth::Policy,
    tls: bool,
) -> Result<Router> {
    if !config.enabled {
        return Ok(Router::new());
    }
    anyhow::ensure!(
        cfg!(unix),
        "managed forwarding is unavailable on this platform until owner-only key ACLs can be verified"
    );
    anyhow::ensure!(
        tls,
        "managed forwarding requires LLMMAN_TLS_CERT/LLMMAN_TLS_KEY"
    );
    anyhow::ensure!(
        policy.enforced(),
        "managed forwarding requires daemon API keys and LLMMAN_AUTH must not be off"
    );
    Ok(router(ManagedState(Arc::new(ManagedInner {
        connection_auth: policy,
        read_timeout: config.read_timeout,
        clients: Mutex::default(),
        #[cfg(test)]
        test_base: None,
    }))))
}

#[derive(Clone)]
struct ManagedState(Arc<ManagedInner>);

struct ManagedInner {
    connection_auth: auth::Policy,
    read_timeout: Option<std::time::Duration>,
    clients: Mutex<HashMap<(Profile, IpAddr), reqwest::Client>>,
    // No production configuration or request may replace the fixed endpoint.
    #[cfg(test)]
    test_base: Option<String>,
}

impl ManagedInner {
    /// The transport for `profile` pinned to `ip`, reused across requests
    /// so each forwarded call does not pay a fresh TCP + TLS handshake.
    fn upstream_client(&self, profile: Profile, ip: IpAddr) -> Result<reqwest::Client> {
        let mut clients = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(client) = clients.get(&(profile, ip)) {
            return Ok(client.clone());
        }
        let client = upstream_client_builder(profile, ip, self.read_timeout)
            .build()
            .context("build managed provider transport")?;
        if clients.len() >= CLIENT_CACHE_LIMIT {
            clients.clear();
        }
        clients.insert((profile, ip), client.clone());
        Ok(client)
    }
}

/// Pins the supervisor-authorized address while retaining the provider's TLS
/// name. Never inherit proxy/CA overrides or redirect delegated credentials.
fn upstream_client_builder(
    profile: Profile,
    ip: IpAddr,
    read_timeout: Option<std::time::Duration>,
) -> reqwest::ClientBuilder {
    let builder = reqwest::Client::builder()
        .no_proxy()
        .resolve(profile.host(), SocketAddr::new(ip, 443))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(30));
    match read_timeout {
        Some(timeout) => builder.read_timeout(timeout),
        None => builder,
    }
}

fn router(state: ManagedState) -> Router {
    Router::new()
        .route("/api/managed/capabilities", get(capabilities))
        .route("/api/codex/responses", post(responses))
        .route("/api/codex/responses/compact", post(compact))
        .route("/api/anthropic/messages", post(messages))
        .route("/api/anthropic/messages/count_tokens", post(count_tokens))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_connection,
        ))
        .with_state(state)
}

async fn require_connection(
    State(state): State<ManagedState>,
    mut req: Request,
    next: Next,
) -> Response {
    // Only connection metadata is authoritative; forwarded IP headers are not.
    let local = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .is_some_and(|peer| peer.0.ip().to_canonical().is_loopback());
    if !local {
        return error(
            StatusCode::FORBIDDEN,
            "managed forwarding requires a loopback peer",
        )
        .into_response();
    }
    let accepted = state.0.connection_auth.enforced()
        && single_header(req.headers(), CONNECTION_KEY)
            .ok()
            .flatten()
            .is_some_and(|key| state.0.connection_auth.accepts(key));
    req.headers_mut().remove(CONNECTION_KEY);
    if !accepted {
        return error(
            StatusCode::UNAUTHORIZED,
            "managed daemon authentication required",
        )
        .into_response();
    }
    next.run(req).await
}

async fn capabilities() -> Json<Value> {
    Json(serde_json::json!({"capabilities": [CAPABILITY, CLAUDE_CAPABILITY]}))
}

/// Deliberately has no derived Debug/Serialize and never enters shared state.
struct Credentials {
    authorization: HeaderValue,
    account: Option<HeaderValue>,
    upstream_ip: IpAddr,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Credentials(<redacted>)")
    }
}

fn single_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, AppError> {
    let mut values = headers.get_all(name).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "duplicate managed request header",
        ));
    }
    value
        .to_str()
        .map(Some)
        .map_err(|_| error(StatusCode::BAD_REQUEST, "invalid managed request header"))
}

impl Credentials {
    fn parse(headers: &HeaderMap, profile: Profile) -> Result<Self, AppError> {
        let upstream_ip = single_header(headers, UPSTREAM_IP)?
            .and_then(|ip| ip.parse::<IpAddr>().ok())
            .filter(public_upstream_ip)
            .ok_or_else(|| {
                error(
                    StatusCode::BAD_REQUEST,
                    "managed inference requires a public upstream IP",
                )
            })?;
        // One Authorization header, parsed by the same rule as the public
        // listener (`auth::bearer`), then held to this listener's stricter
        // shape: a single printable token that is not the local placeholder.
        single_header(headers, "authorization")?
            .ok_or_else(|| error(StatusCode::UNAUTHORIZED, "provider bearer required"))?;
        let token = auth::bearer(headers)
            .ok_or_else(|| error(StatusCode::UNAUTHORIZED, "provider bearer required"))?;
        if token.is_empty()
            || token.bytes().any(|b| !b.is_ascii_graphic() || b == b',')
            || token == crate::providers::PLACEHOLDER_API_KEY
        {
            return Err(error(
                StatusCode::UNAUTHORIZED,
                "explicit provider bearer required",
            ));
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| error(StatusCode::UNAUTHORIZED, "invalid provider bearer"))?;
        authorization.set_sensitive(true);
        let account = if profile == Profile::Codex {
            single_header(headers, "chatgpt-account-id")?
        } else {
            None
        }
        .map(|account| {
            if account.trim().is_empty() {
                return Err(error(StatusCode::BAD_REQUEST, "invalid account header"));
            }
            let mut value = HeaderValue::from_str(account)
                .map_err(|_| error(StatusCode::BAD_REQUEST, "invalid account header"))?;
            value.set_sensitive(true);
            Ok(value)
        })
        .transpose()?;
        Ok(Self {
            authorization,
            account,
            upstream_ip,
        })
    }

    fn redact(&self, text: String) -> String {
        let token = self
            .authorization
            .to_str()
            .expect("validated bearer")
            .trim_start_matches("Bearer ");
        let text = text.replace(token, "<redacted>");
        self.account
            .as_ref()
            .and_then(|v| v.to_str().ok())
            .map_or_else(
                || text.clone(),
                |account| text.replace(account, "<redacted>"),
            )
    }

    /// Decode JSON strings before redaction so escaped quotes, backslashes,
    /// slashes, and Unicode escapes cannot conceal credentials on the wire.
    fn redact_error_body(&self, bytes: &[u8]) -> String {
        fn redact_value(credentials: &Credentials, value: &mut Value) {
            match value {
                Value::String(text) => *text = credentials.redact(std::mem::take(text)),
                Value::Array(values) => {
                    for value in values {
                        redact_value(credentials, value);
                    }
                }
                Value::Object(values) => {
                    *values = std::mem::take(values)
                        .into_iter()
                        .map(|(key, mut value)| {
                            redact_value(credentials, &mut value);
                            (credentials.redact(key), value)
                        })
                        .collect();
                }
                _ => {}
            }
        }
        match serde_json::from_slice::<Value>(bytes) {
            Ok(mut value) => {
                redact_value(self, &mut value);
                value.to_string()
            }
            Err(_) => self.redact(String::from_utf8_lossy(bytes).into_owned()),
        }
    }
}

// The caller authorizes this address against its network policy.
// Refuse special-use space as defense in depth; never re-resolve the host and
// silently select an address the supervisor did not authorize.
fn public_upstream_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || a >= 224
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && ((b == 0 && (c == 0 || c == 2)) || b == 168))
                || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
                || (a == 203 && b == 0 && c == 113))
        }
        IpAddr::V6(ip) => {
            let words = ip.segments();
            words[0] & 0xe000 == 0x2000
                && words[0] != 0x2002
                && !(words[0] == 0x2001 && (words[1] < 0x0200 || words[1] == 0x0db8))
                && !(words[0] == 0x3fff && words[1] & 0xf000 == 0) // RFC 9637: 3fff::/20
        }
    }
}

fn error(status: StatusCode, message: &'static str) -> AppError {
    AppError::status(status, message)
}

async fn responses(
    State(state): State<ManagedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    forward(state, headers, body, Profile::Codex, "/responses").await
}

async fn compact(
    State(state): State<ManagedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    forward(state, headers, body, Profile::Codex, "/responses/compact").await
}

async fn messages(
    State(state): State<ManagedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    forward(state, headers, body, Profile::Claude, "/messages").await
}

async fn count_tokens(
    State(state): State<ManagedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    forward(
        state,
        headers,
        body,
        Profile::Claude,
        "/messages/count_tokens",
    )
    .await
}

// Reject duplicate top-level keys instead of letting serde's last-value-wins
// turn two conflicting model fields into one admitted routing decision.
#[derive(Deserialize)]
struct NativeRequest(#[serde(deserialize_with = "unique_object")] serde_json::Map<String, Value>);

fn unique_object<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<serde_json::Map<String, Value>, D::Error> {
    struct ObjectVisitor;
    impl<'de> serde::de::Visitor<'de> for ObjectVisitor {
        type Value = serde_json::Map<String, Value>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an object with unique keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut access: A,
        ) -> Result<Self::Value, A::Error> {
            let mut map = serde_json::Map::new();
            while let Some((key, value)) = access.next_entry::<String, Value>()? {
                if map.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate request field"));
                }
            }
            Ok(map)
        }
    }
    deserializer.deserialize_map(ObjectVisitor)
}

/// Forward extensions without maintaining a provider-specific allowlist.
/// Strip connection-nominated fields too, and regenerate framing for JSON
/// rewriting/error redaction. Authentication is set explicitly afterward.
fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let connection_fields: Vec<_> = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .collect();
    let mut forwarded = HeaderMap::new();
    for (name, value) in headers {
        let name_str = name.as_str();
        if matches!(
            name_str,
            "connection"
                | "keep-alive"
                | "transfer-encoding"
                | "te"
                | "trailer"
                | "upgrade"
                | "host"
                | "content-length"
                | "authorization"
                | "x-api-key"
                | "x-goog-api-key"
                | "cookie"
                | "set-cookie"
                | "chatgpt-account-id"
        ) || name_str.starts_with("proxy-")
            || name_str.starts_with("x-llmman-")
            || connection_fields
                .iter()
                .any(|field| name_str.eq_ignore_ascii_case(field))
        {
            continue;
        }
        forwarded.append(name.clone(), value.clone());
    }
    forwarded
}

/// Never put request URLs, bearer credentials, or account IDs in diagnostics.
fn log_upstream_error(credentials: &Credentials, err: reqwest::Error) {
    crate::debug_log!(
        "managed provider transport: {}",
        credentials.redact(err.without_url().to_string())
    );
}

/// Content-coding tokens are case-insensitive, including in combined fields.
fn identity_encoding(headers: &HeaderMap) -> bool {
    headers.get_all("content-encoding").iter().all(|value| {
        value.to_str().is_ok_and(|value| {
            value
                .split(',')
                .all(|coding| coding.trim().eq_ignore_ascii_case("identity"))
        })
    })
}

async fn forward(
    state: ManagedState,
    headers: HeaderMap,
    body: Bytes,
    profile: Profile,
    operation: &'static str,
) -> Result<Response, AppError> {
    if !identity_encoding(&headers) {
        return Err(error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "encoded managed request bodies are not supported",
        ));
    }
    let credentials = Credentials::parse(&headers, profile)?;
    let NativeRequest(mut body) = serde_json::from_slice(&body)
        .map_err(|_| error(StatusCode::BAD_REQUEST, "invalid native inference request"))?;
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| model.strip_prefix(profile.model_prefix()))
        .filter(|model| {
            !model.is_empty()
                && model.bytes().all(|b| b.is_ascii_graphic())
                && !model.contains(['?', '#'])
        })
        .ok_or_else(|| {
            error(
                StatusCode::BAD_REQUEST,
                "managed model must match the route provider and include a model ID",
            )
        })?
        .to_owned();
    body.insert("model".into(), Value::String(model));
    let base = profile.base();
    #[cfg(test)]
    let base = state.0.test_base.as_deref().unwrap_or(base);
    let client = state
        .0
        .upstream_client(profile, credentials.upstream_ip)
        .map_err(|err| {
            crate::debug_log!(
                "managed provider transport setup: {}",
                credentials.redact(err.to_string())
            );
            error(
                StatusCode::BAD_GATEWAY,
                "provider upstream transport unavailable",
            )
        })?;
    let mut forwarded = forward_headers(&headers);
    forwarded.remove("content-type");
    forwarded.remove("content-encoding");
    forwarded.remove("accept-encoding");
    if profile == Profile::Claude {
        // These are singular protocol fields, unlike general extension headers.
        single_header(&headers, "anthropic-version")?;
        forwarded.remove("anthropic-beta");
    }
    forwarded.insert("authorization", credentials.authorization.clone());
    if let Some(account) = &credentials.account {
        forwarded.insert("chatgpt-account-id", account.clone());
    }
    let mut request = client
        .post(format!("{base}{operation}"))
        .headers(forwarded)
        .header("accept-encoding", "identity")
        .json(&body);
    match profile {
        Profile::Codex => {}
        Profile::Claude => {
            if !headers.contains_key("anthropic-version") {
                request = request.header("anthropic-version", "2023-06-01");
            }
            let beta = single_header(&headers, "anthropic-beta")?.unwrap_or("");
            let beta = if beta
                .split(',')
                .any(|value| value.trim() == CLAUDE_OAUTH_BETA)
            {
                beta.to_owned()
            } else if beta.trim().is_empty() {
                CLAUDE_OAUTH_BETA.to_owned()
            } else {
                format!("{beta},{CLAUDE_OAUTH_BETA}")
            };
            request = request.header("anthropic-beta", beta);
        }
    }
    let mut upstream = request.send().await.map_err(|err| {
        log_upstream_error(&credentials, err);
        error(
            StatusCode::BAD_GATEWAY,
            "provider upstream connection failed",
        )
    })?;
    let status = upstream.status();
    if status.is_redirection() {
        return Err(error(
            StatusCode::BAD_GATEWAY,
            "provider upstream redirect refused",
        ));
    }
    let mut response = Response::builder().status(status);
    for (name, value) in &forward_headers(upstream.headers()) {
        if let Ok(value) = value.to_str() {
            response = response.header(name, credentials.redact(value.to_owned()));
        }
    }
    let body = if status.is_client_error() || status.is_server_error() {
        if !identity_encoding(upstream.headers()) {
            return Err(error(
                StatusCode::BAD_GATEWAY,
                "encoded provider error body refused",
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = upstream.chunk().await.map_err(|err| {
            log_upstream_error(&credentials, err);
            error(
                StatusCode::BAD_GATEWAY,
                "provider upstream error body interrupted",
            )
        })? {
            if bytes.len() + chunk.len() > ERROR_LIMIT {
                return Err(error(
                    StatusCode::BAD_GATEWAY,
                    "provider upstream error body too large",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        Body::from(credentials.redact_error_body(&bytes))
    } else {
        // No detached pump: dropping the downstream body drops the reqwest
        // stream and cancels the upstream. Never retry or buffer generation.
        Body::from_stream(upstream.bytes_stream().map(move |chunk| {
            chunk.map_err(|err| {
                log_upstream_error(&credentials, err);
                std::io::Error::other("provider upstream stream interrupted")
            })
        }))
    };
    response.body(body).map_err(|_| {
        error(
            StatusCode::BAD_GATEWAY,
            "invalid provider upstream response",
        )
    })
}

#[cfg(test)]
mod tests;
