//! A supervisor-owned, inference-only TLS listener. Its connection key is
//! independent of the request-local provider bearer; neither can fall back to
//! the public daemon's keys, provider catalog, environment, or peers.

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{auth, AppError};

pub(super) const AUTH_PROFILE: &str = "x-llmman-auth-profile";
const CONNECTION_KEY: &str = "x-llmman-managed-key";
const UPSTREAM_IP: &str = "x-llmman-upstream-ip";
const CAPABILITY: &str = "codex-oauth-forwarding-v1";
const CLAUDE_CAPABILITY: &str = "claude-oauth-forwarding-v1";
const CLAUDE_OAUTH_BETA: &str = "oauth-2025-04-20";
const BODY_LIMIT: usize = 32 * 1024 * 1024;
const ERROR_LIMIT: usize = 1024 * 1024;
/// Upstream transports kept warm, one per (profile, pinned IP). The
/// supervisor pins a handful of resolved addresses, so this is a leak
/// guard rather than an eviction policy: past it the map starts over.
const CLIENT_CACHE_LIMIT: usize = 64;
const CODEX_REQUEST_HEADERS: &[&str] = &[
    "openai-beta",
    "originator",
    "session_id",
    "conversation_id",
    "x-codex-turn-metadata",
    "x-codex-turn-state",
    "x-codex-client-version",
    "x-codex-beta-features",
    "user-agent",
];
const CLAUDE_REQUEST_HEADERS: &[&str] = &["anthropic-version", "x-app", "user-agent"];

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Profile {
    Codex,
    Claude,
}

impl Profile {
    fn name(self) -> &'static str {
        match self {
            Self::Codex => "codex-oauth",
            Self::Claude => "claude-oauth",
        }
    }

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

    fn request_headers(self) -> &'static [&'static str] {
        match self {
            Self::Codex => CODEX_REQUEST_HEADERS,
            Self::Claude => CLAUDE_REQUEST_HEADERS,
        }
    }
}

const RESPONSE_HEADERS: &[&str] = &[
    "content-type",
    "x-request-id",
    "request-id",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-limit",
    "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-input-tokens-reset",
    "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-output-tokens-remaining",
    "anthropic-ratelimit-output-tokens-reset",
    "retry-after",
    "x-codex-turn-state",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
];

#[derive(Deserialize)]
#[cfg_attr(test, derive(Clone))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Config {
    schema_version: u32,
    instance_id: String,
    listen: SocketAddr,
    tls_cert_file: PathBuf,
    tls_key_file: PathBuf,
    connection_key_file: PathBuf,
    ready_file: PathBuf,
}

impl Config {
    pub(super) fn from_env() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("LLMMAN_MANAGED_CONFIG") else {
            return Ok(None);
        };
        let bytes = read_private_file(Path::new(&path))?;
        let config: Self = serde_json::from_slice(&bytes).context("parse LLMMAN_MANAGED_CONFIG")?;
        config.validate()?;
        Ok(Some(config))
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == 1,
            "unsupported managed config schemaVersion"
        );
        anyhow::ensure!(
            !self.instance_id.trim().is_empty(),
            "managed instanceId must not be empty"
        );
        anyhow::ensure!(
            self.listen.ip().is_loopback(),
            "managed listener must bind a loopback address"
        );
        for path in [
            &self.tls_cert_file,
            &self.tls_key_file,
            &self.connection_key_file,
            &self.ready_file,
        ] {
            anyhow::ensure!(path.is_absolute(), "managed file paths must be absolute");
        }
        Ok(())
    }
}

fn read_private_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path).context("read managed file metadata")?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() <= BODY_LIMIT as u64,
        "managed configuration must be a bounded regular file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "managed secret/configuration files must be owner-only"
        );
    }
    std::fs::read(path).context("read managed file")
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Ready {
    schema_version: u32,
    instance_id: String,
    pid: u32,
    endpoint: String,
    capabilities: Vec<&'static str>,
}

/// The fields of a published ready document that identify its writer.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadyIdentity {
    instance_id: String,
    pid: u32,
}

#[derive(Clone)]
struct ManagedState(Arc<ManagedInner>);

struct ManagedInner {
    connection_auth: auth::Policy,
    ready: Ready,
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
        // Do not inherit HTTP(S)_PROXY/ALL_PROXY or LLMMAN_TLS_CA. This profile
        // promises its supervisor exactly this validated HTTPS destination.
        let client = reqwest::Client::builder()
            .no_proxy()
            .resolve(profile.host(), SocketAddr::new(ip, 443))
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_secs(60))
            .build()
            .context("build managed provider transport")?;
        if clients.len() >= CLIENT_CACHE_LIMIT {
            clients.clear();
        }
        clients.insert((profile, ip), client.clone());
        Ok(client)
    }
}

pub(super) struct Listener {
    socket: Option<std::net::TcpListener>,
    tls: axum_server::tls_rustls::RustlsConfig,
    state: ManagedState,
    ready_file: PathBuf,
}

impl Listener {
    /// Requires the process-level rustls provider to be installed already;
    /// `serve_async` does that once before any listener is built.
    pub(super) async fn bind(config: Config) -> Result<Self> {
        let key = read_private_file(&config.connection_key_file)?;
        let key = std::str::from_utf8(&key)
            .context("managed connection key must be ASCII")?
            .trim();
        anyhow::ensure!(
            !key.is_empty() && key.bytes().all(|c| c.is_ascii_graphic()),
            "managed connection key must be nonempty printable ASCII"
        );
        // The private key is checked separately so the TLS library does not
        // quietly load a broadly readable credential.
        let cert = std::fs::read(&config.tls_cert_file).context("read managed TLS certificate")?;
        let tls_key = read_private_file(&config.tls_key_file)?;
        let tls = axum_server::tls_rustls::RustlsConfig::from_pem(cert, tls_key)
            .await
            .context("load managed TLS certificate/key")?;
        let socket = tokio::net::TcpListener::bind(config.listen)
            .await
            .context("bind managed listener")?;
        let ready = Ready {
            schema_version: 1,
            instance_id: config.instance_id,
            pid: std::process::id(),
            endpoint: format!("https://{}", socket.local_addr()?),
            capabilities: vec![CAPABILITY, CLAUDE_CAPABILITY],
        };
        let state = ManagedState(Arc::new(ManagedInner {
            connection_auth: auth::Policy::with_keys([key]),
            ready,
            clients: Mutex::default(),
            #[cfg(test)]
            test_base: None,
        }));
        let listener = Self {
            socket: Some(socket.into_std()?),
            tls,
            state,
            ready_file: config.ready_file,
        };
        write_ready(&listener.ready_file, &listener.state.0.ready)?;
        Ok(listener)
    }

    /// Serves until `shutdown` resolves (the same signal the public
    /// listener drains on, so one Ctrl-C stops both) and the in-flight
    /// requests have drained.
    pub(super) async fn serve(
        mut self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<()> {
        let handle = axum_server::Handle::new();
        let shutdown = tokio::spawn({
            let handle = handle.clone();
            async move {
                shutdown.await;
                handle.graceful_shutdown(Some(std::time::Duration::from_secs(30)));
            }
        });
        let result = axum_server::from_tcp_rustls(
            self.socket.take().expect("listener present"),
            self.tls.clone(),
        )
        .handle(handle)
        .serve(router(self.state.clone()).into_make_service())
        .await;
        shutdown.abort();
        result.context("serve managed inference")
    }
}

impl Drop for Listener {
    /// Retracts this instance's own ready document. A supervisor that
    /// restarts the child reuses the same path, so a successor may have
    /// published there while this instance was still draining; its
    /// document is left alone.
    fn drop(&mut self) {
        if ready_file_is_ours(&self.ready_file, &self.state.0.ready) {
            let _ = std::fs::remove_file(&self.ready_file);
        }
    }
}

fn ready_file_is_ours(path: &Path, ready: &Ready) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ReadyIdentity>(&bytes).ok())
        .is_some_and(|published| {
            published.instance_id == ready.instance_id && published.pid == ready.pid
        })
}

fn write_ready(path: &Path, ready: &Ready) -> Result<()> {
    // Owner-only whatever the umask: the document names a live endpoint.
    crate::fsutil::write_atomic_with_mode(path, &serde_json::to_vec(ready)?, 0o600)
        .context("publish managed ready file")
}

fn router(state: ManagedState) -> Router {
    Router::new()
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/compact", post(compact))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
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
    let accepted = single_header(req.headers(), CONNECTION_KEY)
        .ok()
        .flatten()
        .is_some_and(|key| state.0.connection_auth.accepts(key));
    req.headers_mut().remove(CONNECTION_KEY);
    if !accepted {
        return error(
            StatusCode::UNAUTHORIZED,
            "managed connection authentication required",
        )
        .into_response();
    }
    next.run(req).await
}

async fn capabilities(State(state): State<ManagedState>) -> Json<Ready> {
    Json(state.0.ready.clone())
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
        if single_header(headers, AUTH_PROFILE)? != Some(profile.name()) {
            return Err(error(
                StatusCode::BAD_REQUEST,
                "managed auth profile does not match the requested operation",
            ));
        }
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
                && !(words[0] == 0x2001 && (words[1] < 0x0200 || words[1] == 0x0db8))
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

async fn forward(
    state: ManagedState,
    headers: HeaderMap,
    body: Bytes,
    profile: Profile,
    operation: &'static str,
) -> Result<Response, AppError> {
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
                "managed model must match the auth profile provider and include a model ID",
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
        .map_err(|_| {
            error(
                StatusCode::BAD_GATEWAY,
                "provider upstream transport unavailable",
            )
        })?;
    let mut request = client
        .post(format!("{base}{operation}"))
        .header("authorization", credentials.authorization.clone())
        .json(&body);
    if let Some(account) = &credentials.account {
        request = request.header("chatgpt-account-id", account);
    }
    for name in profile.request_headers() {
        if let Some(value) = single_header(&headers, name)? {
            request = request.header(*name, value);
        }
    }
    match profile {
        Profile::Codex => {
            if !headers.contains_key("originator") {
                request = request.header("originator", "codex_cli_rs");
            }
            if !headers.contains_key("openai-beta") {
                request = request.header("openai-beta", "responses=experimental");
            }
        }
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
    let mut upstream = request.send().await.map_err(|_| {
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
    for name in RESPONSE_HEADERS {
        if let Some(value) = upstream.headers().get(*name) {
            if let Ok(value) = value.to_str() {
                response = response.header(*name, credentials.redact(value.to_owned()));
            }
        }
    }
    let body = if status.is_client_error() || status.is_server_error() {
        let mut bytes = Vec::new();
        while let Some(chunk) = upstream.chunk().await.map_err(|_| {
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
        Body::from(credentials.redact(String::from_utf8_lossy(&bytes).into_owned()))
    } else {
        // No detached pump: dropping the downstream body drops the reqwest
        // stream and cancels the upstream. Never retry or buffer generation.
        Body::from_stream(upstream.bytes_stream().map(|chunk| {
            chunk.map_err(|_| std::io::Error::other("provider upstream stream interrupted"))
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
