//! Tokens minted from a long-lived secret at a token endpoint, and the cache
//! they live in until shortly before they expire.
//!
//! A [`TokenService`] is data: OAuth 2.0 client credentials (RFC 6749 §4.4),
//! a JWT bearer assertion (RFC 7523, [`crate::jwt`]) or IBM IAM. The same
//! value is an `llmman.conf` table, a built-in provider profile and a stored
//! login. Secrets are redacted from `Debug`, the cache key holds a hash, and
//! an error names the endpoint's host, never its reply.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::credentials::{ExternalSource, Resolved, Secret, Status, Style, EXPIRY_SKEW_MS};
use crate::logins::{token_reply, Http};

pub const JWT_BEARER_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// litellm `llms/watsonx/common_utils.py` `generate_iam_token`.
pub const IBM_APIKEY_GRANT: &str = "urn:ibm:params:oauth:grant-type:apikey";
/// litellm `llms/watsonx/common_utils.py` `get_watsonx_iam_url`.
pub const IBM_IAM_URL: &str = "https://iam.cloud.ibm.com/identity/token";

/// How long an assertion this process signs is good for.
const ASSERTION_LIFETIME_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// Minted tokens and their cache
// ---------------------------------------------------------------------------

/// A token and when it stops working.
#[derive(Clone)]
pub struct Minted {
    pub token: Secret,
    pub expires_ms: u64,
}

impl std::fmt::Debug for Minted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Minted")
            .field("expires_ms", &self.expires_ms)
            .finish_non_exhaustive()
    }
}

/// What a [`Cache`] holds: anything with an expiry.
pub trait Expiring: Clone {
    fn expires_ms(&self) -> u64;
}

impl Expiring for Minted {
    fn expires_ms(&self) -> u64 {
        self.expires_ms
    }
}

/// How long a failed mint is shared with the requests queued behind it,
/// instead of each retrying the endpoint in turn.
const FAILURE_BACKOFF: Duration = Duration::from_secs(5);

struct Slot<T> {
    value: Mutex<Option<T>>,
    /// The last failed mint, only touched while `value` is held.
    failed: Mutex<Option<(Instant, String)>>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            value: Mutex::new(None),
            failed: Mutex::new(None),
        }
    }
}

/// Minted tokens by key, reused until [`EXPIRY_SKEW_MS`] before they
/// expire, one mint at a time per key: concurrent requests that find a
/// token lapsed wait on the one that is renewing it and then use its
/// result.
pub struct Cache<T> {
    slots: Mutex<Option<HashMap<String, Arc<Slot<T>>>>>,
}

/// The cache of minted tokens.
pub type TokenCache = Cache<Minted>;

impl<T: Expiring> Cache<T> {
    pub const fn new() -> Self {
        Self {
            slots: Mutex::new(None),
        }
    }

    /// The token for `key`, from `mint` if there is none fresh enough.
    /// `skew_ms` is how long before expiry it counts as lapsed. When `mint`
    /// fails and the old token still has life in it, the old one is used:
    /// a token service that is down for a minute must not take every
    /// request with it.
    pub fn get_or_mint(
        &self,
        key: &str,
        now: u64,
        skew_ms: u64,
        mint: impl FnOnce() -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let slot = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .entry(key.to_string())
            .or_default()
            .clone();
        let mut held = slot.value.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(token) = held
            .as_ref()
            .filter(|t| t.expires_ms() > now.saturating_add(skew_ms))
        {
            return Ok(token.clone());
        }
        let started = Instant::now();
        let mut failed = slot.failed.lock().unwrap_or_else(PoisonError::into_inner);
        let shared = failed
            .as_ref()
            .filter(|(at, _)| at.elapsed() < FAILURE_BACKOFF)
            .map(|(_, why)| anyhow!("{why}"));
        let attempt = match shared {
            Some(e) => Err(e),
            None => mint(),
        };
        match attempt {
            Ok(minted) => {
                *failed = None;
                *held = Some(minted.clone());
                Ok(minted)
            }
            Err(e) => {
                if failed.is_none()
                    || failed
                        .as_ref()
                        .is_some_and(|(at, _)| at.elapsed() >= FAILURE_BACKOFF)
                {
                    *failed = Some((Instant::now(), format!("{e:#}")));
                }
                let now = now.saturating_add(started.elapsed().as_millis() as u64);
                match held.as_ref().filter(|t| t.expires_ms() > now) {
                    Some(old) => {
                        eprintln!(
                            "[llmman] warning: could not renew a token ({e:#}); \
                         using the current one until it expires"
                        );
                        Ok(old.clone())
                    }
                    None => Err(e),
                }
            }
        }
    }

    /// When the token for `key` stops working, if one is held.
    pub fn expires_at(&self, key: &str) -> Option<u64> {
        let slot = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()?
            .get(key)?
            .clone();
        let held = slot.value.lock().unwrap_or_else(PoisonError::into_inner);
        held.as_ref().map(|t| t.expires_ms())
    }
}

impl<T: Expiring> Default for Cache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide cache every minted token shares.
pub fn cache() -> &'static TokenCache {
    static CACHE: TokenCache = TokenCache::new();
    &CACHE
}

/// A cache key that names its inputs without holding a secret.
pub fn cache_key(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    hex::encode(hash.finalize())
}

// ---------------------------------------------------------------------------
// The service
// ---------------------------------------------------------------------------

/// Which grant a [`TokenService`] asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grant {
    #[default]
    ClientCredentials,
    JwtBearer,
    IbmApikey,
}

impl Grant {
    /// The credential kind `llmman providers` reports.
    pub fn kind(self) -> &'static str {
        match self {
            Self::ClientCredentials => "client_credentials",
            Self::JwtBearer => "jwt_bearer",
            Self::IbmApikey => "ibm_iam",
        }
    }
}

/// How the client identifies itself to the token endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientAuth {
    /// `client_id` and `client_secret` in the form (RFC 6749 §2.3.1's
    /// alternative).
    #[default]
    Body,
    /// `Authorization: Basic base64(id:secret)`, as litellm's Databricks
    /// M2M and SAP AI Core requests send it.
    Basic,
}

/// A token endpoint and what it is given. The same shape as the
/// `oauth2 = {...}` and `jwt_bearer = {...}` tables of a provider in
/// `llmman.conf` (see `docs/configuration.md`).
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenService {
    #[serde(default, skip_serializing_if = "is_default")]
    pub grant: Grant,
    /// Where the grant is posted. May hold `${VAR}` (Databricks'
    /// workspace host), filled as a provider's base URL is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id_env: Option<String>,
    /// The secret itself. `client_secret_env` and `client_secret_file`
    /// keep it out of the file: set one of the three.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_auth: Option<ClientAuth>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// The `audience` form parameter of a client-credentials request; for
    /// a JWT bearer, the assertion's `aud` claim (the token URL if unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
    /// The RFC 8707 `resource` parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
    /// Extra form parameters, for an endpoint that wants one more.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, String>,
    /// JWT bearer: the assertion's `iss` (the client id if unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issuer: Option<String>,
    /// JWT bearer: the assertion's `sub` (the issuer if unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// JWT bearer: the PEM private key that signs the assertion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_file: Option<String>,
    /// JWT bearer: the `kid` header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    /// JWT bearer: `RS256` (the default for an RSA key) or `ES256`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// JWT bearer: seconds an assertion is valid for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifetime_secs: Option<u64>,
}

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

impl std::fmt::Debug for TokenService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenService")
            .field("grant", &self.grant)
            .field("token_url", &self.token_url)
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "<redacted>"),
            )
            .field("client_secret_env", &self.client_secret_env)
            .field("client_secret_file", &self.client_secret_file)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

/// What minting reads from outside the service: variables, the clock, the
/// network. A test supplies its own of each.
pub struct Ctx<'a> {
    pub env: &'a dyn Fn(&str) -> Option<String>,
    pub http: &'a Http,
    pub now: u64,
}

fn nonblank(v: &Option<String>) -> Option<&str> {
    v.as_deref().map(str::trim).filter(|v| !v.is_empty())
}

impl TokenService {
    /// `self` with what `over` sets laid on top, field by field. The
    /// secret's three spellings, and the client id's two, replace each other
    /// as a group: a stored login's inline secret displaces the profile's
    /// variable rather than conflicting with it.
    pub fn overlaid(&self, over: &TokenService) -> TokenService {
        let mut out = self.clone();
        if over.grant != Grant::default() {
            out.grant = over.grant;
        }
        macro_rules! take {
            ($($f:ident),*) => {$(
                if over.$f.is_some() { out.$f = over.$f.clone(); }
            )*};
        }
        take!(
            client_auth,
            token_url,
            scope,
            audience,
            resource,
            issuer,
            subject,
            key_file,
            key_id,
            algorithm,
            lifetime_secs
        );
        if over.client_id.is_some() || over.client_id_env.is_some() {
            out.client_id = over.client_id.clone();
            out.client_id_env = over.client_id_env.clone();
        }
        if over.client_secret.is_some()
            || over.client_secret_env.is_some()
            || over.client_secret_file.is_some()
        {
            out.client_secret = over.client_secret.clone();
            out.client_secret_env = over.client_secret_env.clone();
            out.client_secret_file = over.client_secret_file.clone();
        }
        out.params.extend(over.params.clone());
        out
    }

    /// What is wrong with this service as written, without reading any
    /// secret or touching the network.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_with(false)
    }

    /// [`TokenService::validate`], allowing a plain-http token URL for a
    /// caller whose HTTP client is itself the guard (a test's).
    pub fn validate_with(&self, allow_http: bool) -> Result<(), String> {
        let secrets = [
            nonblank(&self.client_secret),
            nonblank(&self.client_secret_env),
            nonblank(&self.client_secret_file),
        ];
        if secrets.iter().flatten().count() > 1 {
            return Err(
                "client_secret, client_secret_env and client_secret_file are alternatives; set one"
                    .into(),
            );
        }
        match self.grant {
            Grant::ClientCredentials => {
                if nonblank(&self.client_id).is_none() && nonblank(&self.client_id_env).is_none() {
                    return Err("oauth2 needs client_id or client_id_env".into());
                }
                if secrets.iter().all(Option::is_none) {
                    return Err(
                        "oauth2 needs client_secret, client_secret_env or client_secret_file"
                            .into(),
                    );
                }
            }
            Grant::JwtBearer => {
                if nonblank(&self.key_file).is_none() {
                    return Err("jwt_bearer needs key_file".into());
                }
                if nonblank(&self.issuer).is_none()
                    && nonblank(&self.client_id).is_none()
                    && nonblank(&self.client_id_env).is_none()
                {
                    return Err("jwt_bearer needs issuer (or client_id)".into());
                }
                if let Some(alg) = nonblank(&self.algorithm) {
                    if crate::jwt::Alg::parse(alg).is_none() {
                        return Err(format!("algorithm: {alg:?} is not RS256 or ES256"));
                    }
                }
            }
            Grant::IbmApikey => {
                if secrets.iter().all(Option::is_none) {
                    return Err(
                        "an IBM IAM service needs client_secret, client_secret_env or \
                         client_secret_file (the IBM Cloud API key)"
                            .into(),
                    );
                }
            }
        }
        if nonblank(&self.token_url).is_none() && self.grant != Grant::IbmApikey {
            return Err("token_url is required".into());
        }
        if let Some(url) = nonblank(&self.token_url) {
            if !url.starts_with("https://")
                && !url.starts_with("${")
                && !(allow_http && url.starts_with("http://"))
            {
                return Err("token_url must be an https:// URL".into());
            }
        }
        Ok(())
    }

    fn url(&self, ctx: &Ctx<'_>) -> anyhow::Result<String> {
        let template = nonblank(&self.token_url).unwrap_or(IBM_IAM_URL);
        crate::authscheme::expand(template, &|v| (ctx.env)(v)).map_err(|e| anyhow!(e))
    }

    fn client_id(&self, ctx: &Ctx<'_>) -> Option<String> {
        nonblank(&self.client_id)
            .map(str::to_string)
            .or_else(|| nonblank(&self.client_id_env).and_then(|v| (ctx.env)(v)))
    }

    /// The secret: written inline, named by a variable, or in an
    /// owner-only file.
    fn secret(&self, ctx: &Ctx<'_>) -> anyhow::Result<Option<Secret>> {
        if let Some(inline) = nonblank(&self.client_secret) {
            return Ok(Some(Secret::new(inline.to_string())));
        }
        if let Some(var) = nonblank(&self.client_secret_env) {
            return Ok((ctx.env)(var).map(Secret::new));
        }
        if let Some(path) = nonblank(&self.client_secret_file) {
            let (secret, _) = crate::credentials::read_external(&ExternalSource::File {
                path: path.to_string(),
            })?;
            return Ok(Some(secret));
        }
        Ok(None)
    }

    /// Whether everything minting needs is in place, without minting: no
    /// network. What `llmman providers` reports as `ready`.
    pub fn configured(&self, ctx: &Ctx<'_>) -> bool {
        if self.url(ctx).is_err() {
            return false;
        }
        match self.grant {
            // An unreadable secret counts as set, so minting reports why.
            Grant::ClientCredentials => {
                self.client_id(ctx).is_some() && !matches!(self.secret(ctx), Ok(None))
            }
            Grant::IbmApikey => !matches!(self.secret(ctx), Ok(None)),
            Grant::JwtBearer => {
                nonblank(&self.key_file)
                    .map(crate::credentials::expand_home)
                    .is_some_and(|p| p.is_file())
                    && nonblank(&self.issuer)
                        .map(str::to_string)
                        .or_else(|| self.client_id(ctx))
                        .is_some()
            }
        }
    }

    /// The credential kind and state `llmman providers` shows.
    pub fn status(&self, ctx: &Ctx<'_>) -> Option<Status> {
        if !self.configured(ctx) {
            return None;
        }
        Some(Status {
            kind: self.grant.kind(),
            state: "ready",
            expires_at_ms: self.cache_key(ctx).and_then(|key| cache().expires_at(&key)),
        })
    }

    fn cache_key(&self, ctx: &Ctx<'_>) -> Option<String> {
        let url = self.url(ctx).ok()?;
        let secret = self.secret(ctx).ok().flatten();
        let id = self.client_id(ctx).unwrap_or_default();
        let key_file = nonblank(&self.key_file).unwrap_or_default();
        let extras = serde_json::to_string(&self.params).unwrap_or_default();
        Some(cache_key(&[
            self.grant.kind(),
            &url,
            &id,
            nonblank(&self.scope).unwrap_or_default(),
            nonblank(&self.audience).unwrap_or_default(),
            nonblank(&self.resource).unwrap_or_default(),
            nonblank(&self.issuer).unwrap_or_default(),
            nonblank(&self.subject).unwrap_or_default(),
            key_file,
            secret.as_ref().map_or("", |s| s.expose()),
            &extras,
        ]))
    }

    /// A bearer credential from this service, from the cache when it holds
    /// a token with life in it. Blocking: call from `spawn_blocking`.
    pub fn resolve(&self, ctx: &Ctx<'_>) -> anyhow::Result<Resolved> {
        let key = self
            .cache_key(ctx)
            .ok_or_else(|| anyhow!("the token service is not set up"))?;
        let minted = cache().get_or_mint(&key, ctx.now, EXPIRY_SKEW_MS, || self.mint(ctx))?;
        Ok(bearer(minted.token, self.grant.kind()))
    }

    /// One request to the token endpoint.
    pub fn mint(&self, ctx: &Ctx<'_>) -> anyhow::Result<Minted> {
        let url = self.url(ctx)?;
        let host = reqwest::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .unwrap_or_default();
        let mut form: Vec<(String, String)> = Vec::new();
        let mut headers: Vec<(String, String)> = Vec::new();
        let mut put = |k: &str, v: &str| form.push((k.to_string(), v.to_string()));
        match self.grant {
            Grant::ClientCredentials => {
                let id = self.client_id(ctx).ok_or_else(|| {
                    anyhow!("no client id (set client_id or the variable client_id_env names)")
                })?;
                let secret = self
                    .secret(ctx)?
                    .ok_or_else(|| anyhow!("no client secret (is the variable or file set?)"))?;
                put("grant_type", "client_credentials");
                match self.client_auth.unwrap_or_default() {
                    ClientAuth::Body => {
                        put("client_id", &id);
                        put("client_secret", secret.expose());
                    }
                    ClientAuth::Basic => {
                        headers.push(("authorization".into(), basic(&id, secret.expose())))
                    }
                }
                if let Some(v) = nonblank(&self.scope) {
                    put("scope", v);
                }
                if let Some(v) = nonblank(&self.audience) {
                    put("audience", v);
                }
                if let Some(v) = nonblank(&self.resource) {
                    put("resource", v);
                }
            }
            Grant::JwtBearer => {
                let assertion = self.assertion(ctx, &url)?;
                put("grant_type", JWT_BEARER_GRANT);
                put("assertion", &assertion);
                if let Some(v) = nonblank(&self.scope) {
                    put("scope", v);
                }
                if let Some(v) = nonblank(&self.resource) {
                    put("resource", v);
                }
            }
            Grant::IbmApikey => {
                let secret = self.secret(ctx)?.ok_or_else(|| {
                    anyhow!("no IBM Cloud API key (is the variable or file set?)")
                })?;
                put("grant_type", IBM_APIKEY_GRANT);
                put("apikey", secret.expose());
            }
        }
        for (k, v) in &self.params {
            put(k, v);
        }
        let form_refs: Vec<(&str, &str)> =
            form.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let reply = ctx.http.post_form(&url, &form_refs, &header_refs);
        let tokens = token_reply(reply).map_err(|e| match e {
            crate::credentials::RefreshError::Rejected(why) => {
                anyhow!("the token endpoint at {host} refused the request ({why})")
            }
            crate::credentials::RefreshError::Transient(why) => {
                anyhow!("the token endpoint at {host} failed: {why}")
            }
        })?;
        Ok(Minted {
            expires_ms: tokens.expires_ms(ctx.now),
            token: Secret::new(tokens.access),
        })
    }

    /// The signed assertion of a [`Grant::JwtBearer`] request.
    fn assertion(&self, ctx: &Ctx<'_>, token_url: &str) -> anyhow::Result<String> {
        let path = nonblank(&self.key_file).ok_or_else(|| anyhow!("no key_file"))?;
        let path = crate::credentials::expand_home(path);
        crate::config::owner_readable_only(&path)
            .map_err(|e| anyhow!("refusing to read the signing key {}: {e}", path.display()))?;
        let pem = std::fs::read_to_string(&path)
            .with_context(|| format!("read the signing key {}", path.display()))?;
        let hint = nonblank(&self.algorithm).and_then(crate::jwt::Alg::parse);
        let key = crate::jwt::SigningKey::from_pem(&pem, hint)
            .with_context(|| format!("the signing key {}", path.display()))?;
        if let Some(want) = hint {
            if want != key.alg() {
                bail!(
                    "algorithm is {} but the signing key is {}",
                    want.as_str(),
                    key.alg().as_str()
                );
            }
        }
        let iss = nonblank(&self.issuer)
            .map(str::to_string)
            .or_else(|| self.client_id(ctx))
            .ok_or_else(|| anyhow!("no issuer"))?;
        let sub = nonblank(&self.subject).map_or_else(|| iss.clone(), str::to_string);
        let now_s = ctx.now / 1000;
        let lifetime = self.lifetime_secs.unwrap_or(ASSERTION_LIFETIME_SECS).max(1);
        let mut jti = [0u8; 16];
        getrandom::getrandom(&mut jti).map_err(|e| anyhow!("no randomness: {e}"))?;
        let claims = json!({
            "iss": iss,
            "sub": sub,
            "aud": nonblank(&self.audience).unwrap_or(token_url),
            "iat": now_s,
            "exp": now_s + lifetime,
            "jti": hex::encode(jti),
        });
        key.sign_jwt(nonblank(&self.key_id), &claims)
    }
}

/// A minted bearer, as the resolver hands it on.
pub fn bearer(token: Secret, kind: &'static str) -> Resolved {
    Resolved {
        secret: token,
        style: Style::Bearer,
        kind,
        flow: None,
        metadata: BTreeMap::new(),
        headers: Vec::new(),
        signer: None,
    }
}

fn basic(id: &str, secret: &str) -> String {
    use base64::Engine as _;
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{id}:{secret}"))
    )
}

/// An OAuth 2.0 `expires_in`/`expires_on` pair to an expiry in
/// milliseconds: `expires_in` seconds from `now`, else an epoch
/// `expires_on` in seconds, else an hour.
pub fn expiry_ms(body: &serde_json::Value, now: u64) -> u64 {
    let number = |k: &str| {
        body.get(k).and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_f64().map(|f| f as u64))
                .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
        })
    };
    if let Some(s) = number("expires_in").filter(|s| *s > 0) {
        return now.saturating_add(s.saturating_mul(1000));
    }
    if let Some(at) = number("expires_on").filter(|s| *s > 0) {
        return at.saturating_mul(1000);
    }
    now + 3_600_000
}

/// The HTTP every token service and cloud chain uses: https only, no
/// redirects, shared so its connection pool and TLS roots are built once.
pub fn http() -> &'static Http {
    static HTTP: std::sync::OnceLock<Http> = std::sync::OnceLock::new();
    HTTP.get_or_init(Http::tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testserver::server;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn http() -> Http {
        Http::tokens()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn service(url: &str) -> TokenService {
        TokenService {
            token_url: Some(url.to_string()),
            client_id: Some("cid".into()),
            client_secret: Some("csecret".into()),
            ..TokenService::default()
        }
    }

    fn ok_token(token: &str, expires_in: u64) -> (u16, Value) {
        (
            200,
            json!({"access_token": token, "expires_in": expires_in, "token_type": "Bearer"}),
        )
    }

    /// Unique per test: the cache is process-wide.
    fn url_of(server: &crate::testserver::Server, name: &str) -> String {
        format!("{}/{name}", server.origin)
    }

    #[test]
    fn client_credentials_post_the_client_in_the_form_and_cache_the_token() {
        let server = server();
        server.fake.on("/cc-body", vec![ok_token("tok-1", 3600)]);
        let svc = TokenService {
            scope: Some("a b".into()),
            audience: Some("https://aud.example".into()),
            resource: Some("https://res.example".into()),
            params: BTreeMap::from([("extra".to_string(), "1".to_string())]),
            ..service(&url_of(&server, "cc-body"))
        };
        let h = http();
        let ctx = Ctx {
            env: &no_env,
            http: &h,
            now: 1_000_000,
        };
        let got = svc.resolve(&ctx).unwrap();
        assert_eq!(got.secret.expose(), "tok-1");
        assert_eq!(got.kind, "client_credentials");
        assert_eq!(got.style, Style::Bearer);
        let hit = &server.fake.hits("/cc-body")[0];
        assert_eq!(
            hit.headers["content-type"],
            "application/x-www-form-urlencoded"
        );
        let form = hit.form();
        for (k, v) in [
            ("grant_type", "client_credentials"),
            ("client_id", "cid"),
            ("client_secret", "csecret"),
            ("scope", "a b"),
            ("audience", "https://aud.example"),
            ("resource", "https://res.example"),
            ("extra", "1"),
        ] {
            assert_eq!(form[k], v, "{k}");
        }
        assert!(!hit.headers.contains_key("authorization"));
        // Reused until a minute before it expires; renewed after.
        svc.resolve(&Ctx {
            now: 1_000_000 + 3_000_000,
            ..Ctx {
                env: &no_env,
                http: &h,
                now: 0,
            }
        })
        .unwrap();
        assert_eq!(server.fake.hits("/cc-body").len(), 1);
        server.fake.on("/cc-body", vec![ok_token("tok-2", 3600)]);
        let later = svc
            .resolve(&Ctx {
                env: &no_env,
                http: &h,
                now: 1_000_000 + 3_541_000,
            })
            .unwrap();
        assert_eq!(later.secret.expose(), "tok-2");
        assert_eq!(server.fake.hits("/cc-body").len(), 2);
    }

    #[test]
    fn basic_client_authentication_keeps_the_secret_out_of_the_form() {
        let server = server();
        server
            .fake
            .on("/cc-basic", vec![ok_token("tok-basic", 3600)]);
        let svc = TokenService {
            client_auth: Some(ClientAuth::Basic),
            scope: Some("all-apis".into()),
            ..service(&url_of(&server, "cc-basic"))
        };
        let h = http();
        svc.resolve(&Ctx {
            env: &no_env,
            http: &h,
            now: 5,
        })
        .unwrap();
        let hit = &server.fake.hits("/cc-basic")[0];
        use base64::Engine as _;
        let expected = format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode("cid:csecret")
        );
        assert_eq!(hit.headers["authorization"], expected);
        let form = hit.form();
        assert!(!form.contains_key("client_secret") && !form.contains_key("client_id"));
        assert_eq!(form["scope"], "all-apis");
    }

    /// Many requests find the token lapsed at once: one request to the
    /// endpoint.
    #[test]
    fn concurrent_requests_share_one_mint() {
        let server = server();
        server
            .fake
            .on("/cc-flight", vec![ok_token("tok-flight", 3600)]);
        let svc = std::sync::Arc::new(service(&url_of(&server, "cc-flight")));
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..12)
            .map(|_| {
                let svc = svc.clone();
                let calls = calls.clone();
                std::thread::spawn(move || {
                    let h = http();
                    let got = svc
                        .resolve(&Ctx {
                            env: &no_env,
                            http: &h,
                            now: 77,
                        })
                        .unwrap();
                    calls.fetch_add(1, Ordering::SeqCst);
                    got.secret.expose().to_string()
                })
            })
            .collect();
        let seen: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(seen.iter().all(|s| s == "tok-flight"));
        assert_eq!(server.fake.hits("/cc-flight").len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 12);
    }

    #[test]
    fn the_secret_may_come_from_a_variable_or_an_owner_only_file() {
        let server = server();
        server
            .fake
            .on("/cc-secret", vec![ok_token("tok-secret", 3600)]);
        let h = http();
        // A variable.
        let by_env = TokenService {
            client_secret: None,
            client_secret_env: Some("MY_SECRET".into()),
            client_id: None,
            client_id_env: Some("MY_ID".into()),
            ..service(&url_of(&server, "cc-secret"))
        };
        let env = |name: &str| match name {
            "MY_SECRET" => Some("from-env".to_string()),
            "MY_ID" => Some("id-from-env".to_string()),
            _ => None,
        };
        let ctx = Ctx {
            env: &env,
            http: &h,
            now: 5,
        };
        assert!(by_env.configured(&ctx));
        by_env.resolve(&ctx).unwrap();
        let form = server.fake.hits("/cc-secret")[0].form();
        assert_eq!(
            (form["client_secret"].as_str(), form["client_id"].as_str()),
            ("from-env", "id-from-env")
        );
        // Unset: not configured, and no request.
        assert!(!by_env.configured(&Ctx {
            env: &no_env,
            http: &h,
            now: 5
        }));
        assert_eq!(
            by_env.status(&Ctx {
                env: &no_env,
                http: &h,
                now: 5
            }),
            None
        );
        let status = by_env.status(&ctx).unwrap();
        assert_eq!((status.kind, status.state), ("client_credentials", "ready"));

        // A file, which must be owner-only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("secret");
            std::fs::write(&path, "from-file\n").unwrap();
            let by_file = TokenService {
                client_secret: None,
                client_secret_file: Some(path.to_string_lossy().into_owned()),
                ..service(&url_of(&server, "cc-secret-file"))
            };
            server
                .fake
                .on("/cc-secret-file", vec![ok_token("tok-file", 3600)]);
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
            let err = by_file
                .resolve(&Ctx {
                    env: &no_env,
                    http: &h,
                    now: 5,
                })
                .unwrap_err();
            assert!(format!("{err:#}").contains("chmod 600"), "{err:#}");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            by_file
                .resolve(&Ctx {
                    env: &no_env,
                    http: &h,
                    now: 5,
                })
                .unwrap();
            assert_eq!(
                server.fake.hits("/cc-secret-file")[0].form()["client_secret"],
                "from-file"
            );
        }
    }

    #[test]
    fn a_service_that_is_written_wrong_is_refused_before_it_is_used() {
        let ok = service("https://idp.example/token");
        assert!(ok.validate().is_ok());
        for (svc, wants) in [
            (
                TokenService {
                    client_secret_env: Some("X".into()),
                    ..ok.clone()
                },
                "alternatives",
            ),
            (
                TokenService {
                    client_secret: None,
                    ..ok.clone()
                },
                "client_secret",
            ),
            (
                TokenService {
                    client_id: None,
                    ..ok.clone()
                },
                "client_id",
            ),
            (
                TokenService {
                    token_url: None,
                    ..ok.clone()
                },
                "token_url",
            ),
            (
                TokenService {
                    token_url: Some("http://idp.example/token".into()),
                    ..ok.clone()
                },
                "https",
            ),
            (
                TokenService {
                    grant: Grant::JwtBearer,
                    client_secret: None,
                    ..ok.clone()
                },
                "key_file",
            ),
            (
                TokenService {
                    grant: Grant::JwtBearer,
                    key_file: Some("k.pem".into()),
                    algorithm: Some("HS256".into()),
                    ..ok.clone()
                },
                "RS256",
            ),
            (
                TokenService {
                    grant: Grant::IbmApikey,
                    client_secret: None,
                    ..ok.clone()
                },
                "IBM",
            ),
        ] {
            let err = svc.validate().unwrap_err();
            assert!(err.contains(wants), "{wants}: {err}");
        }
        // IBM needs no URL: IAM's is the default.
        let ibm = TokenService {
            grant: Grant::IbmApikey,
            client_secret: Some("k".into()),
            ..TokenService::default()
        };
        assert!(ibm.validate().is_ok());
    }

    #[test]
    fn a_plain_http_token_endpoint_is_refused_outside_tests() {
        let server = server();
        server.fake.on("/cc-plain", vec![ok_token("never", 3600)]);
        let svc = service(&url_of(&server, "cc-plain"));
        // The real client: https only.
        let strict = Http::new();
        let err = svc
            .resolve(&Ctx {
                env: &no_env,
                http: &strict,
                now: 5,
            })
            .unwrap_err();
        assert!(format!("{err:#}").contains("plain http"), "{err:#}");
        assert!(server.fake.hits("/cc-plain").is_empty(), "nothing was sent");
    }

    #[test]
    fn a_refusal_names_the_host_and_the_code_but_not_the_body_or_the_secret() {
        let server = server();
        server.fake.on(
            "/cc-refused",
            vec![(
                401,
                json!({"error": "invalid_client", "error_description": "bad csecret sk-leak"}),
            )],
        );
        let svc = service(&url_of(&server, "cc-refused"));
        let h = http();
        let err = svc
            .resolve(&Ctx {
                env: &no_env,
                http: &h,
                now: 5,
            })
            .unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("127.0.0.1") && text.contains("invalid_client"),
            "{text}"
        );
        assert!(
            !text.contains("sk-leak") && !text.contains("csecret"),
            "{text}"
        );
        // A server error is a failure, not a refusal.
        server.fake.on("/cc-down", vec![(503, json!({}))]);
        let down = service(&url_of(&server, "cc-down"));
        let err = down
            .resolve(&Ctx {
                env: &no_env,
                http: &h,
                now: 5,
            })
            .unwrap_err();
        assert!(format!("{err:#}").contains("failed"), "{err:#}");
        // And the service prints no secret.
        let printed = format!("{svc:?}");
        assert!(!printed.contains("csecret"), "{printed}");
    }

    #[test]
    fn a_failing_renewal_keeps_the_token_that_still_has_life() {
        let cache: Cache<Minted> = Cache::new();
        let minted = |t: &str, e: u64| Minted {
            token: Secret::new(t.into()),
            expires_ms: e,
        };
        assert_eq!(
            cache
                .get_or_mint("k", 100, 60, || Ok(minted("a", 1_000)))
                .unwrap()
                .token
                .expose(),
            "a"
        );
        // Inside the skew but not expired, and the service is down: keep it.
        let kept = cache
            .get_or_mint("k", 950, 100, || Err(anyhow!("down")))
            .unwrap();
        assert_eq!(kept.token.expose(), "a");
        // Expired and down: an error.
        assert!(cache
            .get_or_mint("k", 1_001, 100, || Err(anyhow!("down")))
            .is_err());
        assert_eq!(cache.expires_at("k"), Some(1_000));
        assert_eq!(cache.expires_at("other"), None);
    }

    /// A signed assertion in place of a secret: RS256 and ES256.
    #[cfg(unix)]
    #[test]
    fn a_jwt_bearer_assertion_is_signed_with_the_key_and_carries_the_claims() {
        use ring::signature::{
            UnparsedPublicKey, ECDSA_P256_SHA256_FIXED, RSA_PKCS1_2048_8192_SHA256,
        };
        use std::os::unix::fs::PermissionsExt;
        let server = server();
        let dir = tempfile::tempdir().unwrap();
        let h = http();

        // RS256, with a key id, an audience of its own and a scope.
        let rsa = dir.path().join("rsa.pem");
        std::fs::write(&rsa, include_str!("testdata/jwt-test-rsa.pem")).unwrap();
        std::fs::set_permissions(&rsa, std::fs::Permissions::from_mode(0o600)).unwrap();
        server.fake.on("/jwt-rs", vec![ok_token("tok-jwt", 3600)]);
        let svc = TokenService {
            grant: Grant::JwtBearer,
            token_url: Some(url_of(&server, "jwt-rs")),
            issuer: Some("issuer-1".into()),
            subject: Some("subject-1".into()),
            audience: Some("https://aud.example".into()),
            scope: Some("read".into()),
            key_file: Some(rsa.to_string_lossy().into_owned()),
            key_id: Some("kid-1".into()),
            ..TokenService::default()
        };
        let as_written = TokenService {
            token_url: Some("https://idp.example/token".into()),
            ..svc.clone()
        };
        assert!(as_written.validate().is_ok());
        let ctx = Ctx {
            env: &no_env,
            http: &h,
            now: 1_700_000_000_000,
        };
        assert!(svc.configured(&ctx));
        assert_eq!(svc.resolve(&ctx).unwrap().kind, "jwt_bearer");
        let form = server.fake.hits("/jwt-rs")[0].form();
        assert_eq!(form["grant_type"], JWT_BEARER_GRANT);
        assert_eq!(form["scope"], "read");
        use base64::Engine as _;
        let parts: Vec<&str> = form["assertion"].split('.').collect();
        let dec = |p: &str| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(p)
                .unwrap()
        };
        let header: Value = serde_json::from_slice(&dec(parts[0])).unwrap();
        let claims: Value = serde_json::from_slice(&dec(parts[1])).unwrap();
        assert_eq!(
            header,
            json!({"alg": "RS256", "typ": "JWT", "kid": "kid-1"})
        );
        assert_eq!(claims["iss"], "issuer-1");
        assert_eq!(claims["sub"], "subject-1");
        assert_eq!(claims["aud"], "https://aud.example");
        assert_eq!(claims["iat"], 1_700_000_000u64);
        assert_eq!(claims["exp"], 1_700_000_300u64);
        let public = ring::signature::RsaKeyPair::from_pkcs8(
            &base64::engine::general_purpose::STANDARD
                .decode(
                    include_str!("testdata/jwt-test-rsa.pem")
                        .lines()
                        .filter(|l| !l.starts_with("-----"))
                        .collect::<String>(),
                )
                .unwrap(),
        )
        .unwrap();
        UnparsedPublicKey::new(
            &RSA_PKCS1_2048_8192_SHA256,
            public.public().as_ref().to_vec(),
        )
        .verify(
            format!("{}.{}", parts[0], parts[1]).as_bytes(),
            &dec(parts[2]),
        )
        .expect("the assertion verifies");

        // ES256, the audience defaulting to the token URL, issuer to the client id.
        let generated = rcgen::KeyPair::generate().unwrap();
        let ec = dir.path().join("ec.pem");
        std::fs::write(&ec, generated.serialize_pem()).unwrap();
        std::fs::set_permissions(&ec, std::fs::Permissions::from_mode(0o600)).unwrap();
        server.fake.on("/jwt-es", vec![ok_token("tok-es", 3600)]);
        let es = TokenService {
            grant: Grant::JwtBearer,
            token_url: Some(url_of(&server, "jwt-es")),
            client_id: Some("client-as-issuer".into()),
            key_file: Some(ec.to_string_lossy().into_owned()),
            algorithm: Some("ES256".into()),
            ..TokenService::default()
        };
        es.resolve(&Ctx {
            env: &no_env,
            http: &h,
            now: 5_000,
        })
        .unwrap();
        let form = server.fake.hits("/jwt-es")[0].form();
        let parts: Vec<&str> = form["assertion"].split('.').collect();
        let claims: Value = serde_json::from_slice(&dec(parts[1])).unwrap();
        assert_eq!(claims["iss"], "client-as-issuer");
        assert_eq!(claims["sub"], "client-as-issuer");
        assert_eq!(claims["aud"], url_of(&server, "jwt-es"));
        UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, generated.public_key_raw())
            .verify(
                format!("{}.{}", parts[0], parts[1]).as_bytes(),
                &dec(parts[2]),
            )
            .expect("the ES256 assertion verifies");

        // The key must be owner-only, and the algorithm must match it.
        std::fs::set_permissions(&ec, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = es
            .mint(&Ctx {
                env: &no_env,
                http: &h,
                now: 5_000,
            })
            .unwrap_err();
        assert!(format!("{err:#}").contains("refusing"), "{err:#}");
        std::fs::set_permissions(&ec, std::fs::Permissions::from_mode(0o600)).unwrap();
        let wrong = TokenService {
            algorithm: Some("RS256".into()),
            ..es.clone()
        };
        let err = wrong
            .mint(&Ctx {
                env: &no_env,
                http: &h,
                now: 5_000,
            })
            .unwrap_err();
        assert!(err.to_string().contains("algorithm is RS256"), "{err}");
    }

    /// watsonx: the IBM Cloud API key for an IAM bearer.
    #[test]
    fn an_ibm_api_key_is_exchanged_for_an_iam_token() {
        let server = server();
        server.fake.on(
            "/ibm-iam",
            vec![(
                200,
                json!({"access_token": "iam-token", "expires_in": 3600, "token_type": "Bearer"}),
            )],
        );
        let svc = TokenService {
            grant: Grant::IbmApikey,
            token_url: Some(url_of(&server, "ibm-iam")),
            client_secret_env: Some("WATSONX_API_KEY".into()),
            ..TokenService::default()
        };
        let env = |n: &str| (n == "WATSONX_API_KEY").then(|| "ibm-key".to_string());
        let h = http();
        let got = svc
            .resolve(&Ctx {
                env: &env,
                http: &h,
                now: 5,
            })
            .unwrap();
        assert_eq!((got.secret.expose(), got.kind), ("iam-token", "ibm_iam"));
        let hit = &server.fake.hits("/ibm-iam")[0];
        let form = hit.form();
        assert_eq!(form["grant_type"], "urn:ibm:params:oauth:grant-type:apikey");
        assert_eq!(form["apikey"], "ibm-key");
        assert_eq!(hit.headers["accept"], "application/json");
        assert!(TokenService {
            token_url: None,
            ..svc.clone()
        }
        .validate()
        .is_ok());
        assert_eq!(IBM_IAM_URL, "https://iam.cloud.ibm.com/identity/token");
    }

    /// Databricks' `${DATABRICKS_HOST}` is filled as a base URL's is.
    #[test]
    fn a_token_url_template_is_filled_from_the_variables() {
        let svc = TokenService {
            token_url: Some("https://${DATABRICKS_HOST}/oidc/v1/token".into()),
            ..service("unused")
        };
        let h = http();
        let with = |host: &'static str| {
            let env = move |n: &str| (n == "DATABRICKS_HOST").then(|| host.to_string());
            let ctx = Ctx {
                env: &env,
                http: &h,
                now: 5,
            };
            svc.url(&ctx)
        };
        assert_eq!(
            with("https://adb-1.azuredatabricks.net/").unwrap(),
            "https://adb-1.azuredatabricks.net/oidc/v1/token"
        );
        assert_eq!(
            with("adb-1.azuredatabricks.net").unwrap(),
            "https://adb-1.azuredatabricks.net/oidc/v1/token"
        );
        // A host that would move the request is refused, and not echoed.
        let err = with("evil.com/x").unwrap_err().to_string();
        assert!(!err.contains("evil"), "{err}");
        assert!(svc
            .url(&Ctx {
                env: &no_env,
                http: &h,
                now: 5
            })
            .is_err());
        assert!(!svc.configured(&Ctx {
            env: &no_env,
            http: &h,
            now: 5
        }));
    }

    #[test]
    fn overlays_replace_a_secret_as_a_group() {
        let base = TokenService {
            grant: Grant::ClientCredentials,
            token_url: Some("https://x/token".into()),
            client_id_env: Some("ID".into()),
            client_secret_env: Some("SECRET".into()),
            client_auth: Some(ClientAuth::Basic),
            scope: Some("s".into()),
            ..TokenService::default()
        };
        let stored = TokenService {
            client_id: Some("stored-id".into()),
            client_secret: Some("stored-secret".into()),
            ..TokenService::default()
        };
        let merged = base.overlaid(&stored);
        assert_eq!(merged.client_id.as_deref(), Some("stored-id"));
        assert_eq!(merged.client_id_env, None);
        assert_eq!(merged.client_secret.as_deref(), Some("stored-secret"));
        assert_eq!(merged.client_secret_env, None);
        assert_eq!(merged.client_auth, Some(ClientAuth::Basic));
        assert_eq!(merged.token_url.as_deref(), Some("https://x/token"));
        assert!(merged.validate().is_ok());
    }

    #[test]
    fn expiry_reads_the_forms_servers_use() {
        assert_eq!(expiry_ms(&json!({"expires_in": 60}), 1000), 61_000);
        assert_eq!(expiry_ms(&json!({"expires_in": "60"}), 1000), 61_000);
        assert_eq!(
            expiry_ms(&json!({"expires_on": "1506484173"}), 1000),
            1_506_484_173_000
        );
        assert_eq!(expiry_ms(&json!({}), 1000), 3_601_000);
        assert_eq!(cache_key(&["a", "bc"]), cache_key(&["a", "bc"]));
        assert_ne!(cache_key(&["a", "bc"]), cache_key(&["ab", "c"]));
    }
}
