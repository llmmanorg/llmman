//! Amazon Bedrock: AWS credentials and SigV4 request signing.
//!
//! A Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`) is an ordinary bearer key
//! and wins. Otherwise requests are SigV4-signed with credentials from the
//! SDK's own chain (profile, web identity, container, IMDSv2), each given its
//! inputs from [`Runtime`]. Temporary credentials are cached until five
//! minutes before expiry. The region is that of the endpoint's host.
//! Bedrock is reached on its OpenAI-compatible routes, not Converse.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use aws_config::provider_config::ProviderConfig;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{future, ProvideCredentials};
use aws_credential_types::Credentials;
use aws_runtime::env_config::file::{EnvConfigFileKind, EnvConfigFiles};
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use aws_types::region::Region;
use reqwest::header::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::credentials::{RequestSigner, Resolved, Secret, Status, Style};
use crate::tokensvc::{cache_key, Cache, Expiring};

/// How long before a temporary credential expires it is renewed. The SDK's
/// own `DEFAULT_BUFFER_TIME`.
const RENEW_BEFORE_MS: u64 = 5 * 60 * 1000;

/// The region when nothing names one: opencode `bedrock-auth.ts`
/// `resolveRegion`, litellm `BEDROCK_MANTLE_DEFAULT_REGION`.
pub const DEFAULT_REGION: &str = "us-east-1";

/// The SigV4 signing name of `bedrock-runtime`: litellm
/// `llms/bedrock/chat/chat_completions/transformation.py`
/// (`service_name="bedrock"`).
pub const SERVICE_RUNTIME: &str = "bedrock";
/// The signing name of `bedrock-mantle`: opencode
/// `providers/amazon-bedrock-mantle.ts` (`service: "bedrock-mantle"`).
/// litellm signs Mantle as `bedrock` (`bedrock_mantle/common_utils.py`);
/// the two disagree, and `[providers.<id>] aws = { service = ... }` sets
/// either.
pub const SERVICE_MANTLE: &str = "bedrock-mantle";

/// The `aws = {...}` table of a provider in `llmman.conf`, and the profile
/// a built-in Bedrock provider starts from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AwsChain {
    /// A named profile (`AWS_PROFILE` if unset). Naming one uses that
    /// profile alone: environment keys do not override it, as `--profile`
    /// does not in the AWS CLI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// The region, when `AWS_REGION` and the profile do not give it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// The SigV4 signing name, for an endpoint that wants another.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service: Option<String>,
    /// Set when `llmman.conf` has an `aws` table for the provider: try the
    /// whole chain, IMDS included, whether or not anything on this machine
    /// hints that AWS is there.
    #[serde(skip)]
    pub forced: bool,
}

impl AwsChain {
    /// `self` with what `over` sets laid on top.
    pub fn overlaid(&self, over: &AwsChain) -> AwsChain {
        AwsChain {
            profile: over.profile.clone().or_else(|| self.profile.clone()),
            region: over.region.clone().or_else(|| self.region.clone()),
            service: over.service.clone().or_else(|| self.service.clone()),
            forced: self.forced || over.forced,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(region) = &self.region {
            region_ok(region)?;
        }
        if let Some(service) = &self.service {
            if service.is_empty()
                || !service
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            {
                return Err(format!("service: {service:?} is not a signing name"));
            }
        }
        Ok(())
    }
}

pub(crate) fn region_ok(region: &str) -> Result<(), String> {
    if !region.is_empty()
        && region.len() <= 32
        && region
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !region.starts_with('-')
    {
        Ok(())
    } else {
        Err(format!("region: {region:?} is not an AWS region name"))
    }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/// The DNS suffix of `region`'s partition. litellm
/// `litellm_core_utils/aws_partition.py`.
pub fn dns_suffix(region: &str) -> &'static str {
    const PARTITIONS: &[(&str, &str)] = &[
        ("cn-", "amazonaws.com.cn"),
        ("us-gov-", "amazonaws.com"),
        ("us-isob-", "sc2s.sgov.gov"),
        ("us-isof-", "csp.hci.ic.gov"),
        ("us-iso-", "c2s.ic.gov"),
        ("eu-isoe-", "cloud.adc-e.uk"),
    ];
    PARTITIONS
        .iter()
        .find(|(prefix, _)| region.starts_with(prefix))
        .map_or("amazonaws.com", |(_, suffix)| suffix)
}

/// `bedrock-runtime`'s OpenAI-compatible Chat Completions base:
/// `https://bedrock-runtime.{region}.amazonaws.com/openai/v1` — AWS's
/// "Responses API" guide for Bedrock lists it as the recommended
/// endpoint; litellm `llms/bedrock/chat/chat_completions/transformation.py`.
pub fn runtime_base(region: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.{}/openai/v1",
        dns_suffix(region)
    )
}

/// `bedrock-mantle`'s: `https://bedrock-mantle.{region}.api.aws/v1`
/// (opencode `providers/amazon-bedrock-mantle.ts`).
pub fn mantle_base(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/v1")
}

/// The region a Bedrock endpoint's host names, which is the region its
/// requests must be signed for.
pub fn region_of_host(url: &str) -> Option<String> {
    let host = reqwest::Url::parse(url)
        .ok()?
        .host_str()?
        .to_ascii_lowercase();
    let rest = host
        .strip_prefix("bedrock-runtime-fips.")
        .or_else(|| host.strip_prefix("bedrock-runtime."))
        .or_else(|| host.strip_prefix("bedrock-mantle."))?;
    let region = rest.split('.').next()?;
    region_ok(region).ok().map(|_| region.to_string())
}

// ---------------------------------------------------------------------------
// The environment the chain reads
// ---------------------------------------------------------------------------

/// The process environment, files and HTTP the chain runs against. A test
/// supplies its own.
pub struct Runtime {
    /// A fixed environment for a test; the process's when `None`.
    env: Option<BTreeMap<String, String>>,
    conf: ProviderConfig,
    /// Where `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` is relative to: ECS's
    /// fixed address.
    container_host: String,
}

/// ECS's credentials endpoint, which a relative URI is relative to.
const ECS_HOST: &str = "169.254.170.2";

impl Runtime {
    /// The process's own environment. Under test it is an empty one with a
    /// home that does not exist: tests never read the real AWS configuration.
    pub fn live() -> Self {
        Self {
            env: cfg!(test).then(|| {
                BTreeMap::from([(
                    "HOME".to_string(),
                    "/nonexistent-llmman-test-home".to_string(),
                )])
            }),
            conf: ProviderConfig::default(),
            container_host: ECS_HOST.into(),
        }
    }

    /// For a test: a fixed environment, whatever [`ProviderConfig`] (its
    /// HTTP client) the test built, and a stand-in for ECS's address.
    #[cfg(test)]
    pub fn with(env: &[(&str, &str)], conf: ProviderConfig, container_host: &str) -> Self {
        Self {
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            conf,
            container_host: container_host.into(),
        }
    }

    fn get(&self, name: &str) -> Option<String> {
        match &self.env {
            Some(fixed) => fixed
                .get(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty()),
            None => crate::providers::key_from_env(name),
        }
    }

    fn home(&self) -> Option<PathBuf> {
        self.get("HOME")
            .or_else(|| self.get("USERPROFILE"))
            .map(PathBuf::from)
            .or_else(dirs::home_dir)
    }

    fn file(&self, var: &str, default: &str) -> PathBuf {
        match self.get(var) {
            Some(path) => crate::credentials::expand_home(&path),
            None => self
                .home()
                .map_or_else(|| PathBuf::from(default), |h| h.join(".aws").join(default)),
        }
    }

    fn credentials_file(&self) -> PathBuf {
        self.file("AWS_SHARED_CREDENTIALS_FILE", "credentials")
    }

    fn config_file(&self) -> PathBuf {
        self.file("AWS_CONFIG_FILE", "config")
    }
}

/// The value of `key` in `[section]` of an AWS INI file.
fn ini_value(text: &str, section: &str, key: &str) -> Option<String> {
    let mut current = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with(';') || line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            current = name.trim().to_string();
            continue;
        }
        if current == section {
            if let Some((k, v)) = line.split_once('=') {
                if k.trim() == key {
                    return Some(v.trim().to_string()).filter(|v| !v.is_empty());
                }
            }
        }
    }
    None
}

/// The section names of an AWS INI file.
fn ini_sections(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix('[')
                .and_then(|l| l.strip_suffix(']'))
                .map(|n| n.trim().to_string())
        })
        .collect()
}

impl AwsChain {
    fn profile_name(&self, rt: &Runtime) -> String {
        self.profile
            .clone()
            .or_else(|| rt.get("AWS_PROFILE"))
            .unwrap_or_else(|| "default".into())
    }

    /// The region requests are made in: `AWS_REGION`, `AWS_DEFAULT_REGION`,
    /// the profile's `region`, this chain's own, else [`DEFAULT_REGION`].
    /// No network, and so no IMDS region lookup.
    pub fn region(&self, rt: &Runtime) -> String {
        let from_profile = || {
            let name = self.profile_name(rt);
            let section = if name == "default" {
                "default".to_string()
            } else {
                format!("profile {name}")
            };
            let text = std::fs::read_to_string(rt.config_file()).ok()?;
            ini_value(&text, &section, "region")
        };
        [
            self.profile
                .is_none()
                .then(|| rt.get("AWS_REGION"))
                .flatten(),
            self.profile
                .is_none()
                .then(|| rt.get("AWS_DEFAULT_REGION"))
                .flatten(),
            from_profile(),
            self.region.clone(),
        ]
        .into_iter()
        .flatten()
        .find(|r| region_ok(r).is_ok())
        .unwrap_or_else(|| DEFAULT_REGION.to_string())
    }

    /// Whether anything on this machine says AWS credentials are there to
    /// find — the checks opencode makes before it counts Bedrock as usable
    /// (`plugin/provider/amazon-bedrock.ts` `CHAIN_ENV`), plus the shared
    /// files and EC2's DMI vendor, so a bare instance role is seen. The
    /// answer is a guess; resolving is what proves it. `llmman.conf`'s
    /// `aws` table stands for "yes".
    pub fn hinted(&self, rt: &Runtime) -> bool {
        if self.forced || self.profile.is_some() {
            return true;
        }
        const ENV: &[&str] = &[
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        ];
        if ENV.iter().any(|v| rt.get(v).is_some()) {
            return true;
        }
        let has_default = |path: PathBuf| {
            std::fs::read_to_string(path)
                .is_ok_and(|t| ini_sections(&t).iter().any(|s| s == "default"))
        };
        if has_default(rt.credentials_file()) || has_default(rt.config_file()) {
            return true;
        }
        on_ec2()
    }

    /// `ready` when [`AwsChain::hinted`], with the expiry of the
    /// credentials in hand if there are some; `None` otherwise. No network.
    pub fn status(&self, rt: &Runtime) -> Option<Status> {
        if !self.hinted(rt) {
            return None;
        }
        Some(Status {
            kind: "aws_sigv4",
            state: "ready",
            expires_at_ms: CREDENTIALS.expires_at(&self.cache_key(rt)),
        })
    }

    fn cache_key(&self, rt: &Runtime) -> String {
        const ENV: &[&str] = &[
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        ];
        let seen: Vec<String> = ENV
            .iter()
            .map(|v| format!("{v}={}", rt.get(v).unwrap_or_default()))
            .collect();
        let mut parts: Vec<&str> = vec![self.profile.as_deref().unwrap_or(""), "aws"];
        let region = self.region(rt);
        parts.push(&region);
        parts.extend(seen.iter().map(String::as_str));
        cache_key(&parts)
    }
}

/// Whether this is an EC2 instance, by the DMI vendor the AWS CLI and
/// cloud-init also read. Linux only; elsewhere, no.
fn on_ec2() -> bool {
    if cfg!(test) {
        return false;
    }
    ["sys_vendor", "bios_vendor", "product_version"]
        .iter()
        .filter_map(|f| std::fs::read_to_string(format!("/sys/devices/virtual/dmi/id/{f}")).ok())
        .any(|v| v.to_ascii_lowercase().contains("amazon"))
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Held {
    credentials: Credentials,
    expires_ms: u64,
}

impl Expiring for Held {
    fn expires_ms(&self) -> u64 {
        self.expires_ms
    }
}

static CREDENTIALS: Cache<Held> = Cache::new();

/// Runs `future` to completion on a thread of its own with a runtime of its
/// own: this is called from `spawn_blocking` and from plain threads alike,
/// and `Handle::block_on` is only allowed from the first.
fn block_on<F>(future: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a current-thread runtime")
                    .block_on(future)
            })
            .join()
            .unwrap_or_else(|_| panic!("the AWS credential task panicked"))
    })
}

impl AwsChain {
    /// The chain's credentials now: from the cache when they are temporary
    /// and still good, else by running the chain.
    pub fn credentials(&self, rt: &Runtime, now: u64) -> anyhow::Result<Credentials> {
        // Plain environment keys need no chain, and no runtime.
        if self.profile.is_none() {
            if let (Some(id), Some(secret)) =
                (rt.get("AWS_ACCESS_KEY_ID"), rt.get("AWS_SECRET_ACCESS_KEY"))
            {
                return Ok(Credentials::new(
                    id,
                    secret,
                    rt.get("AWS_SESSION_TOKEN"),
                    None,
                    "llmman-env",
                ));
            }
        }
        let held = CREDENTIALS.get_or_mint(&self.cache_key(rt), now, RENEW_BEFORE_MS, || {
            let credentials = self.run_chain(rt)?;
            let expires_ms = credentials
                .expiry()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_millis() as u64);
            Ok(Held {
                credentials,
                expires_ms,
            })
        })?;
        Ok(held.credentials)
    }

    /// The SDK's providers, tried in its own order — the shared files, web
    /// identity, container credentials, IMDSv2 (environment keys are the
    /// caller's) — but each given its inputs from `rt`, so what they read
    /// is what [`Runtime`] says, not whatever the process's environment
    /// holds. A provider with nothing to work from (`CredentialsNotLoaded`)
    /// passes to the next; one that finds its inputs and fails (a denied
    /// `AssumeRole`, a stale SSO login) ends the chain with that error, as
    /// the SDK's chain does. An explicit profile is used alone.
    fn run_chain(&self, rt: &Runtime) -> anyhow::Result<Credentials> {
        let region = Region::new(self.region(rt));
        let conf = rt.conf.clone().with_region(Some(region));
        // A file that is not there is no profile, not a broken one.
        let mut any_file = false;
        let mut files = EnvConfigFiles::builder()
            .include_default_credentials_file(false)
            .include_default_config_file(false);
        for (kind, path) in [
            (EnvConfigFileKind::Credentials, rt.credentials_file()),
            (EnvConfigFileKind::Config, rt.config_file()),
        ] {
            if path.is_file() {
                any_file = true;
                files = files.with_file(kind, path);
            }
        }
        let mut sources: Vec<Box<dyn ProvideCredentials>> = Vec::new();
        if any_file {
            sources.push(Box::new(
                aws_config::profile::ProfileFileCredentialsProvider::builder()
                    .configure(&conf)
                    .profile_files(files.build())
                    .profile_name(self.profile_name(rt))
                    .build(),
            ));
        } else if let Some(profile) = &self.profile {
            anyhow::bail!(
                "AWS profile {profile:?} is named, but there is no AWS config or credentials file"
            );
        }
        if self.profile.is_none() {
            if let (Some(file), Some(role_arn)) = (
                rt.get("AWS_WEB_IDENTITY_TOKEN_FILE"),
                rt.get("AWS_ROLE_ARN"),
            ) {
                sources.push(Box::new(
                    aws_config::web_identity_token::WebIdentityTokenCredentialsProvider::builder()
                        .configure(&conf)
                        .static_configuration(aws_config::web_identity_token::StaticConfiguration {
                            web_identity_token_file: crate::credentials::expand_home(&file),
                            role_arn,
                            session_name: rt.get("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| {
                                format!("llmman-{}", crate::credentials::now_ms())
                            }),
                        })
                        .build(),
                ));
            }
            match Container::from_env(rt) {
                Ok(Some(container)) => sources.push(Box::new(container)),
                Ok(None) => {}
                Err(e) => sources.push(Box::new(Misconfigured(format!("{e:#}")))),
            }
            if rt.get("AWS_EC2_METADATA_DISABLED").as_deref() != Some("true") {
                let client = aws_config::imds::Client::builder().configure(&conf);
                let client = match rt.get("AWS_EC2_METADATA_SERVICE_ENDPOINT") {
                    Some(endpoint) => client
                        .endpoint(endpoint)
                        .map_err(|e| format!("AWS_EC2_METADATA_SERVICE_ENDPOINT: {e}")),
                    None => Ok(client),
                };
                match client {
                    Ok(client) => sources.push(Box::new(
                        aws_config::imds::credentials::ImdsCredentialsProvider::builder()
                            .configure(&conf)
                            .imds_client(client.build())
                            .build(),
                    )),
                    Err(why) => sources.push(Box::new(Misconfigured(why))),
                }
            }
        }
        let provided = block_on(async move {
            let mut last = CredentialsError::not_loaded("no AWS credential source is set up");
            for source in &sources {
                match source.provide_credentials().await {
                    Ok(credentials) => return Ok(credentials),
                    Err(e @ CredentialsError::CredentialsNotLoaded(_)) => last = e,
                    Err(e) => return Err(e),
                }
            }
            Err(last)
        });
        provided.map_err(|e| {
            anyhow!(
                "AWS credentials could not be found: {:#}",
                anyhow::Error::new(e)
            )
        })
    }

    /// A signing credential for a request to `base_url`, or `Ok(None)` when
    /// nothing says AWS is configured here (see [`AwsChain::hinted`]).
    pub fn resolve(
        &self,
        rt: &Runtime,
        base_url: &str,
        default_service: &str,
        now: u64,
    ) -> anyhow::Result<Option<Resolved>> {
        if !self.hinted(rt) {
            return Ok(None);
        }
        let credentials = self.credentials(rt, now)?;
        let region = region_of_host(base_url).unwrap_or_else(|| self.region(rt));
        let service = self
            .service
            .clone()
            .unwrap_or_else(|| default_service.to_string());
        Ok(Some(Resolved {
            secret: Secret::new(String::new()),
            style: Style::Key,
            kind: "aws_sigv4",
            flow: None,
            metadata: BTreeMap::new(),
            headers: Vec::new(),
            signer: Some(Arc::new(SigV4 {
                credentials,
                region,
                service,
                at: None,
            })),
        }))
    }
}

// ---------------------------------------------------------------------------
// Container credentials
// ---------------------------------------------------------------------------

/// ECS task roles and EKS pod identity: a credentials endpoint named by
/// `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` (on ECS's fixed address) or
/// `AWS_CONTAINER_CREDENTIALS_FULL_URI`, with an optional
/// `AWS_CONTAINER_AUTHORIZATION_TOKEN` (or `_FILE`) sent as the
/// `Authorization` header. The SDK's rules for the URI apply: https, or
/// plain http only to loopback or the two container addresses.
struct Container {
    url: String,
    authorization: Option<String>,
}

/// Whether a full container credentials URI may be fetched: https, or http
/// to loopback, `169.254.170.2`, `169.254.170.23` or `fd00:ec2::23` — the
/// SDK's allow-list (the AWS SDKs and CLI reference, "container credential
/// provider").
fn container_uri_allowed(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return false;
    }
    if parsed.scheme() == "https" {
        return true;
    }
    if parsed.scheme() != "http" {
        return false;
    }
    let host = parsed.host_str().unwrap_or("").trim_matches(['[', ']']);
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => {
            ip.is_loopback()
                || ip == std::net::IpAddr::from([169, 254, 170, 2])
                || ip == std::net::IpAddr::from([169, 254, 170, 23])
                || ip == "fd00:ec2::23".parse::<std::net::IpAddr>().unwrap()
        }
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

impl Container {
    fn from_env(rt: &Runtime) -> anyhow::Result<Option<Self>> {
        let url = match (
            rt.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"),
            rt.get("AWS_CONTAINER_CREDENTIALS_FULL_URI"),
        ) {
            (Some(relative), _) => {
                anyhow::ensure!(
                    relative.starts_with('/'),
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI must start with '/'"
                );
                format!("http://{}{relative}", rt.container_host)
            }
            (None, Some(full)) => {
                anyhow::ensure!(
                    container_uri_allowed(&full),
                    "AWS_CONTAINER_CREDENTIALS_FULL_URI must be https, or http to loopback \
                     or a container address"
                );
                full
            }
            (None, None) => return Ok(None),
        };
        let authorization = match rt.get("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
            Some(path) => Some(
                std::fs::read_to_string(crate::credentials::expand_home(&path))
                    .context("read AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")?
                    .trim()
                    .to_string(),
            ),
            None => rt.get("AWS_CONTAINER_AUTHORIZATION_TOKEN"),
        };
        Ok(Some(Self { url, authorization }))
    }

    fn fetch(&self) -> anyhow::Result<Credentials> {
        let http = crate::logins::Http::metadata();
        let headers: Vec<(&str, &str)> = self
            .authorization
            .as_deref()
            .map(|a| ("authorization", a))
            .into_iter()
            .collect();
        let reply = http
            .get(&self.url, &headers)
            .map_err(|e| anyhow!("the container credentials endpoint: {e}"))?;
        anyhow::ensure!(
            reply.ok(),
            "the container credentials endpoint answered HTTP {}",
            reply.status
        );
        let body = reply.json();
        let text = |k: &str| {
            body.get(k)
                .and_then(serde_json::Value::as_str)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        };
        let (Some(id), Some(secret)) = (text("AccessKeyId"), text("SecretAccessKey")) else {
            anyhow::bail!("the container credentials reply had no keys");
        };
        let expiry = text("Expiration")
            .and_then(|e| chrono::DateTime::parse_from_rfc3339(&e).ok())
            .map(SystemTime::from);
        Ok(Credentials::new(
            id,
            secret,
            text("Token"),
            expiry,
            "llmman-container",
        ))
    }
}

/// A source whose settings are wrong. It fails only if the chain gets
/// this far, so an earlier source that works still wins.
#[derive(Debug)]
struct Misconfigured(String);

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Container")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

impl ProvideCredentials for Misconfigured {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::ready(Err(CredentialsError::invalid_configuration(
            self.0.clone(),
        )))
    }
}

impl ProvideCredentials for Container {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(async move {
            // The blocking client must not run on the runtime's own thread.
            let this = Container {
                url: self.url.clone(),
                authorization: self.authorization.clone(),
            };
            tokio::task::spawn_blocking(move || this.fetch())
                .await
                .map_err(CredentialsError::provider_error)?
                .map_err(|e| CredentialsError::provider_error(e.to_string()))
        })
    }
}

// ---------------------------------------------------------------------------
// SigV4
// ---------------------------------------------------------------------------

/// Signs requests for one region and service with one set of credentials.
pub struct SigV4 {
    credentials: Credentials,
    region: String,
    service: String,
    /// The signing time; the clock when `None`. For the test vectors.
    at: Option<SystemTime>,
}

impl SigV4 {
    /// The headers that sign a request with these parts: `Authorization`,
    /// `X-Amz-Date`, and `X-Amz-Security-Token` for a session. Every header
    /// in `headers` is signed. The payload hash is of exactly `body`.
    pub fn headers_for(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: &[u8],
    ) -> anyhow::Result<Vec<(String, String)>> {
        let identity: Identity = self.credentials.clone().into();
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(&self.service)
            .time(self.at.unwrap_or_else(SystemTime::now))
            .settings(SigningSettings::default())
            .build()
            .map_err(|e| anyhow!("SigV4 parameters: {e}"))?
            .into();
        let signable = SignableRequest::new(
            method,
            url,
            headers.iter().map(|(k, v)| (k.as_str(), v.as_str())),
            SignableBody::Bytes(body),
        )
        .map_err(|e| anyhow!("SigV4 cannot sign this request: {e}"))?;
        let (instructions, _) = sign(signable, &params)
            .map_err(|e| anyhow!("SigV4 signing failed: {e}"))?
            .into_parts();
        Ok(instructions
            .headers()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect())
    }
}

impl RequestSigner for SigV4 {
    /// Signs `request` as it will be sent: its method, URL, in-memory body
    /// bytes, and the `content-type` and `x-amz-*` headers it carries. Other
    /// headers (the transport's `Host`, `Content-Length`, `Accept`) are left
    /// out of the signature, as they may still be added or rewritten on the
    /// way out; `Host` is signed from the URL.
    fn sign(&self, request: &mut reqwest::Request) -> anyhow::Result<()> {
        let body: Vec<u8> = match request.body() {
            None => Vec::new(),
            Some(body) => body
                .as_bytes()
                .ok_or_else(|| anyhow!("a streamed body cannot be signed with SigV4"))?
                .to_vec(),
        };
        let signed: Vec<(String, String)> = request
            .headers()
            .iter()
            .filter(|(name, _)| {
                let name = name.as_str();
                name == "content-type" || name.starts_with("x-amz-")
            })
            .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
            .collect();
        let added = self.headers_for(
            request.method().as_str(),
            request.url().as_str(),
            &signed,
            &body,
        )?;
        for (name, value) in added {
            let name =
                HeaderName::from_bytes(name.as_bytes()).context("a signature header name")?;
            let mut value = HeaderValue::from_str(&value).context("a signature header value")?;
            value.set_sensitive(true);
            request.headers_mut().insert(name, value);
        }
        Ok(())
    }
}

/// `SigV4` that signs as of a fixed time, for a test.
#[cfg(test)]
pub fn sigv4_at(credentials: Credentials, region: &str, service: &str, at: SystemTime) -> SigV4 {
    SigV4 {
        credentials,
        region: region.into(),
        service: service.into(),
        at: Some(at),
    }
}

/// Seconds since the epoch of an RFC 3339 time, for tests.
#[cfg(test)]
fn time(rfc3339: &str) -> SystemTime {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .expect("a time")
        .with_timezone(&chrono::Utc)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_smithy_runtime_api::client::http::{
        HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpClient,
        SharedHttpConnector,
    };
    use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
    use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
    use aws_smithy_types::body::SdkBody;
    use std::collections::VecDeque;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    fn example(token: Option<&str>) -> Credentials {
        Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            token.map(str::to_string),
            None,
            "test",
        )
    }

    fn vector(token: Option<&str>) -> SigV4 {
        sigv4_at(
            example(token),
            "us-east-1",
            "service",
            time("2015-08-30T12:36:00Z"),
        )
    }

    fn authorization(headers: &[(String, String)]) -> String {
        headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.clone())
            .expect("an authorization header")
    }

    /// AWS's published Signature Version 4 test suite
    /// (`aws-sig-v4-test-suite`, bundled with `aws-sigv4`):
    /// `get-vanilla`.
    #[test]
    fn the_published_get_vanilla_vector_signs_identically() {
        let headers = vector(None)
            .headers_for("GET", "https://example.amazonaws.com/", &[], b"")
            .unwrap();
        assert_eq!(
            authorization(&headers),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
        assert!(headers.contains(&("x-amz-date".into(), "20150830T123600Z".into())));
    }

    /// `post-vanilla-query`, and `get-vanilla-query-order-key-case`: the
    /// query is part of what is signed.
    #[test]
    fn the_published_query_vectors_sign_identically() {
        let post = vector(None)
            .headers_for(
                "POST",
                "https://example.amazonaws.com/?Param1=value1",
                &[],
                b"",
            )
            .unwrap();
        assert!(authorization(&post).ends_with(
            "Signature=28038455d6de14eafc1f9222cf5aa6f1a96197d7deb8263271d420d138af7f11"
        ));
        let ordered = vector(None)
            .headers_for(
                "GET",
                "https://example.amazonaws.com/?Param2=value2&Param1=value1",
                &[],
                b"",
            )
            .unwrap();
        assert!(authorization(&ordered).ends_with(
            "Signature=b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500"
        ));
    }

    /// `get-header-value-trim`: headers a caller passes are signed, their
    /// values trimmed.
    #[test]
    fn the_published_header_vector_signs_the_headers_it_is_given() {
        let headers = vector(None)
            .headers_for(
                "GET",
                "https://example.amazonaws.com/",
                &[
                    ("My-Header1".into(), " value1".into()),
                    ("My-Header2".into(), "\"a   b   c\"".into()),
                ],
                b"",
            )
            .unwrap();
        assert_eq!(
            authorization(&headers),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;my-header1;my-header2;x-amz-date, \
             Signature=acc3ed3afb60bb290fc8d2dd0098b9911fcaa05412b367055dee359757a9c736"
        );
    }

    /// `get-vanilla-with-session-token`: a session adds
    /// `x-amz-security-token` to what is signed.
    #[test]
    fn the_published_session_token_vector_signs_identically() {
        let headers = vector(Some(
            "6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267",
        ))
        .headers_for("GET", "https://example.amazonaws.com/", &[], b"")
        .unwrap();
        assert_eq!(
            authorization(&headers),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date;x-amz-security-token, \
             Signature=07ec1639c89043aa0e3e2de82b96708f198cceab042d4a97044c66dd9f74e7f8"
        );
        assert!(headers.iter().any(|(k, _)| k == "x-amz-security-token"));
    }

    /// The signature covers the exact bytes of the body: one more byte, a
    /// different signature.
    #[test]
    fn the_signature_covers_the_body_bytes() {
        let sign = |body: &'static [u8]| {
            let client = reqwest::Client::new();
            let mut request = client
                .post("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body)
                .build()
                .unwrap();
            sigv4_at(
                example(None),
                "us-east-1",
                "bedrock",
                time("2024-01-02T03:04:05Z"),
            )
            .sign(&mut request)
            .unwrap();
            request
        };
        let a = sign(br#"{"model":"m"}"#);
        let b = sign(br#"{"model":"m" }"#);
        let again = sign(br#"{"model":"m"}"#);
        let auth =
            |r: &reqwest::Request| r.headers()["authorization"].to_str().unwrap().to_string();
        assert_ne!(auth(&a), auth(&b));
        assert_eq!(auth(&a), auth(&again));
        assert!(auth(&a).contains("Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request"));
        assert!(auth(&a).contains("SignedHeaders=content-type;host;x-amz-date"));
        assert!(a.headers()["authorization"].is_sensitive());
        assert_eq!(a.headers()["x-amz-date"], "20240102T030405Z");
    }

    #[test]
    fn a_streamed_body_is_refused_rather_than_sent_unsigned() {
        let client = reqwest::Client::new();
        let stream =
            futures::stream::once(async { Ok::<_, std::io::Error>(bytes::Bytes::from("x")) });
        let mut request = client
            .post("https://bedrock-runtime.us-east-1.amazonaws.com/x")
            .body(reqwest::Body::wrap_stream(stream))
            .build()
            .unwrap();
        let err = vector(None).sign(&mut request).unwrap_err();
        assert!(err.to_string().contains("streamed"), "{err}");
    }

    #[test]
    fn endpoints_regions_and_partitions() {
        assert_eq!(
            runtime_base("us-west-2"),
            "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
        );
        assert_eq!(
            runtime_base("cn-north-1"),
            "https://bedrock-runtime.cn-north-1.amazonaws.com.cn/openai/v1"
        );
        assert_eq!(
            mantle_base("eu-west-1"),
            "https://bedrock-mantle.eu-west-1.api.aws/v1"
        );
        for (url, region) in [
            (
                "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
                Some("us-west-2"),
            ),
            (
                "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com",
                Some("us-gov-west-1"),
            ),
            (
                "https://bedrock-mantle.us-east-2.api.aws/v1",
                Some("us-east-2"),
            ),
            (
                "https://vpce-0123.bedrock-runtime.us-east-1.vpce.amazonaws.com",
                None,
            ),
            ("https://example.com", None),
        ] {
            assert_eq!(region_of_host(url).as_deref(), region, "{url}");
        }
        assert!(region_ok("us-east-1").is_ok());
        assert!(region_ok("us-east-1.evil.com").is_err());
        assert!(region_ok("").is_err());
        assert!(region_ok("US").is_err());
    }

    fn runtime(env: &[(&str, &str)]) -> Runtime {
        Runtime::with(env, ProviderConfig::empty(), "127.0.0.1:1")
    }

    #[test]
    fn the_region_follows_env_then_profile_then_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        std::fs::write(
            &config,
            "[default]\nregion = eu-west-3\n\n[profile work]\nregion=ap-south-1\n# comment\n",
        )
        .unwrap();
        let cfg = config.to_str().unwrap();
        let chain = AwsChain::default();
        let region = |env: &[(&str, &str)], chain: &AwsChain| chain.region(&runtime(env));
        assert_eq!(
            region(
                &[("AWS_REGION", "us-west-2"), ("AWS_CONFIG_FILE", cfg)],
                &chain
            ),
            "us-west-2"
        );
        assert_eq!(
            region(
                &[
                    ("AWS_DEFAULT_REGION", "us-west-1"),
                    ("AWS_CONFIG_FILE", cfg)
                ],
                &chain
            ),
            "us-west-1"
        );
        assert_eq!(region(&[("AWS_CONFIG_FILE", cfg)], &chain), "eu-west-3");
        assert_eq!(
            region(&[("AWS_CONFIG_FILE", cfg), ("AWS_PROFILE", "work")], &chain),
            "ap-south-1"
        );
        assert_eq!(
            region(&[("AWS_CONFIG_FILE", "/nonexistent")], &chain),
            DEFAULT_REGION
        );
        // A hostile value is not a region.
        assert_eq!(
            region(
                &[
                    ("AWS_REGION", "x.evil.com/"),
                    ("AWS_CONFIG_FILE", "/nonexistent")
                ],
                &chain
            ),
            DEFAULT_REGION
        );
        // A profile this chain names beats the ambient environment.
        let named = AwsChain {
            profile: Some("work".into()),
            ..AwsChain::default()
        };
        assert_eq!(
            region(
                &[("AWS_REGION", "us-west-2"), ("AWS_CONFIG_FILE", cfg)],
                &named
            ),
            "ap-south-1"
        );
        let pinned = AwsChain {
            region: Some("ca-central-1".into()),
            ..AwsChain::default()
        };
        assert_eq!(
            region(&[("AWS_CONFIG_FILE", "/nonexistent")], &pinned),
            "ca-central-1"
        );
    }

    #[test]
    fn hints_say_when_the_chain_is_worth_trying() {
        let dir = tempfile::tempdir().unwrap();
        let none = dir.path().join("none");
        let nothing = [
            ("HOME", dir.path().to_str().unwrap()),
            ("AWS_CONFIG_FILE", none.to_str().unwrap()),
            ("AWS_SHARED_CREDENTIALS_FILE", none.to_str().unwrap()),
        ];
        let chain = AwsChain::default();
        if !on_ec2() {
            assert!(!chain.hinted(&runtime(&nothing)));
            assert_eq!(chain.status(&runtime(&nothing)), None);
        }
        let with = |extra: (&'static str, &'static str)| {
            let mut env = nothing.to_vec();
            env.push(extra);
            runtime(&env)
        };
        for var in [
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        ] {
            assert!(chain.hinted(&with((var, "x"))), "{var}");
        }
        let credentials = dir.path().join("credentials");
        std::fs::write(&credentials, "[default]\naws_access_key_id = a\n").unwrap();
        let rt = runtime(&[
            ("HOME", dir.path().to_str().unwrap()),
            ("AWS_CONFIG_FILE", none.to_str().unwrap()),
            ("AWS_SHARED_CREDENTIALS_FILE", credentials.to_str().unwrap()),
        ]);
        assert!(chain.hinted(&rt));
        let status = chain.status(&rt).unwrap();
        assert_eq!((status.kind, status.state), ("aws_sigv4", "ready"));
        // `aws = {}` in llmman.conf forces it.
        let forced = AwsChain {
            forced: true,
            ..AwsChain::default()
        };
        assert!(forced.hinted(&runtime(&nothing)));
    }

    #[test]
    fn environment_keys_become_a_signing_credential_without_a_chain() {
        let rt = runtime(&[
            ("AWS_ACCESS_KEY_ID", "AKIDENV"),
            ("AWS_SECRET_ACCESS_KEY", "secret-env"),
            ("AWS_SESSION_TOKEN", "session-env"),
            ("AWS_REGION", "us-west-2"),
        ]);
        let resolved = AwsChain::default()
            .resolve(
                &rt,
                "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
                SERVICE_RUNTIME,
                0,
            )
            .unwrap()
            .unwrap();
        assert_eq!(resolved.kind, "aws_sigv4");
        assert!(resolved.secret.expose().is_empty());
        assert!(resolved.signer.is_some());
        let printed = format!("{resolved:?}");
        assert!(
            !printed.contains("secret-env") && !printed.contains("session-env"),
            "{printed}"
        );
        // Nothing hints, nothing is resolved.
        let dir = tempfile::tempdir().unwrap();
        let bare = runtime(&[("HOME", dir.path().to_str().unwrap())]);
        if !on_ec2() {
            assert!(AwsChain::default()
                .resolve(
                    &bare,
                    "https://bedrock-runtime.us-east-1.amazonaws.com",
                    SERVICE_RUNTIME,
                    0
                )
                .unwrap()
                .is_none());
        }
    }

    // -- the chain, against replayed AWS services ------------------------------

    /// An AWS HTTP client that answers from a script, in order, and keeps
    /// what it was asked. The SDK's own replay client would do the same, but
    /// its crate turns on `serde_json/preserve_order` for every test.
    #[derive(Clone, Debug, Default)]
    struct Replay {
        replies: StdArc<StdMutex<VecDeque<(u16, String)>>>,
        seen: StdArc<StdMutex<Vec<HttpRequest>>>,
    }

    impl Replay {
        fn requests(&self) -> std::sync::MutexGuard<'_, Vec<HttpRequest>> {
            self.seen.lock().unwrap()
        }
    }

    impl HttpConnector for Replay {
        fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
            self.seen.lock().unwrap().push(request);
            let (status, body) = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or((500, "unscripted".into()));
            let mut response = HttpResponse::new(status.try_into().unwrap(), SdkBody::from(body));
            // IMDS wants this on its token reply and no other reply minds it.
            response
                .headers_mut()
                .insert("x-aws-ec2-metadata-token-ttl-seconds", "21600");
            HttpConnectorFuture::ready(Ok(response))
        }
    }

    impl HttpClient for Replay {
        fn http_connector(
            &self,
            _: &HttpConnectorSettings,
            _: &RuntimeComponents,
        ) -> SharedHttpConnector {
            SharedHttpConnector::new(self.clone())
        }
    }

    fn replay(events: Vec<(u16, String)>) -> (SharedHttpClient, Replay) {
        let client = Replay {
            replies: StdArc::new(StdMutex::new(events.into())),
            ..Replay::default()
        };
        (SharedHttpClient::new(client.clone()), client)
    }

    fn event(_uri: &str, status: u16, body: &str) -> (u16, String) {
        (status, body.to_string())
    }

    fn conf_with(client: SharedHttpClient) -> ProviderConfig {
        ProviderConfig::empty()
            .with_http_client(client)
            .with_sleep_impl(aws_smithy_async::rt::sleep::TokioSleep::new())
    }

    fn future_time() -> String {
        (chrono::Utc::now() + chrono::Duration::hours(2))
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string()
    }

    /// ECS and EKS: credentials from the container endpoint, with its
    /// authorization token, fetched once and reused until shortly before they
    /// expire.
    #[test]
    fn container_credentials_are_fetched_once_and_reused() {
        let server = crate::testserver::server();
        let body = serde_json::json!({
            "AccessKeyId": "AKIDCONTAINER",
            "SecretAccessKey": "container-secret",
            "Token": "container-token",
            "Expiration": future_time(),
        });
        server.fake.on("/v2/credentials/task", vec![(200, body)]);
        let dir = tempfile::tempdir().unwrap();
        let none = dir.path().join("none");
        let rt = Runtime::with(
            &[
                (
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                    "/v2/credentials/task",
                ),
                ("AWS_CONTAINER_AUTHORIZATION_TOKEN", "Bearer task-token"),
                ("AWS_REGION", "us-east-1"),
                ("AWS_EC2_METADATA_DISABLED", "true"),
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_CONFIG_FILE", none.to_str().unwrap()),
                ("AWS_SHARED_CREDENTIALS_FILE", none.to_str().unwrap()),
            ],
            conf_with(replay(Vec::new()).0),
            server.authority(),
        );
        let chain = AwsChain::default();
        let first = chain.credentials(&rt, now_ms_for_test()).unwrap();
        assert_eq!(first.access_key_id(), "AKIDCONTAINER");
        assert_eq!(first.session_token(), Some("container-token"));
        let second = chain.credentials(&rt, now_ms_for_test()).unwrap();
        assert_eq!(second.access_key_id(), "AKIDCONTAINER");
        let hits = server.fake.hits("/v2/credentials/task");
        assert_eq!(
            hits.len(),
            1,
            "the temporary credential is reused until shortly before it expires"
        );
        assert_eq!(hits[0].headers["authorization"], "Bearer task-token");
    }

    /// `AWS_CONTAINER_CREDENTIALS_FULL_URI`, and the addresses it may name.
    #[test]
    fn a_full_container_uri_must_be_https_or_a_container_address() {
        for (url, allowed) in [
            ("https://creds.example.com/x", true),
            ("http://127.0.0.1:8080/x", true),
            ("http://localhost/x", true),
            ("http://[::1]/x", true),
            ("http://169.254.170.2/v2/credentials/x", true),
            ("http://169.254.170.23/x", true),
            ("http://[fd00:ec2::23]/x", true),
            ("http://169.254.169.254/x", false),
            ("http://creds.example.com/x", false),
            ("http://10.0.0.5/x", false),
            ("ftp://127.0.0.1/x", false),
            ("https://u:p.example.com/x", false),
        ] {
            assert_eq!(container_uri_allowed(url), allowed, "{url}");
        }
        let server = crate::testserver::server();
        server.fake.on(
            "/full",
            vec![(
                200,
                serde_json::json!({"AccessKeyId": "AKIDFULL", "SecretAccessKey": "s"}),
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "file-token\n").unwrap();
        let none = dir.path().join("none");
        let full = format!("{}/full", server.origin);
        let rt = Runtime::with(
            &[
                ("AWS_CONTAINER_CREDENTIALS_FULL_URI", full.as_str()),
                (
                    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
                    token.to_str().unwrap(),
                ),
                ("AWS_REGION", "us-east-1"),
                ("AWS_EC2_METADATA_DISABLED", "true"),
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_CONFIG_FILE", none.to_str().unwrap()),
                ("AWS_SHARED_CREDENTIALS_FILE", none.to_str().unwrap()),
            ],
            conf_with(replay(Vec::new()).0),
            "127.0.0.1:1",
        );
        let got = AwsChain::default()
            .credentials(&rt, now_ms_for_test())
            .unwrap();
        assert_eq!(got.access_key_id(), "AKIDFULL");
        assert_eq!(
            server.fake.hits("/full")[0].headers["authorization"],
            "file-token"
        );
        // A URI the SDK would refuse is refused before any request.
        let bad = Runtime::with(
            &[("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://10.0.0.5/x")],
            ProviderConfig::empty(),
            "127.0.0.1:1",
        );
        let err = AwsChain::default()
            .credentials(&bad, now_ms_for_test())
            .unwrap_err();
        assert!(err.to_string().contains("FULL_URI"), "{err}");
    }

    fn now_ms_for_test() -> u64 {
        crate::credentials::now_ms()
    }

    /// Web identity: the token file is read and exchanged at STS with
    /// `AssumeRoleWithWebIdentity`.
    #[test]
    fn a_web_identity_token_is_exchanged_for_role_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "web-identity-jwt").unwrap();
        let expiry = future_time();
        let xml = format!(
            r#"<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>AKIDWEB</AccessKeyId><SecretAccessKey>web-secret</SecretAccessKey><SessionToken>web-session</SessionToken><Expiration>{expiry}</Expiration></Credentials><AssumedRoleUser><AssumedRoleId>AROA:s</AssumedRoleId><Arn>arn:aws:sts::1:assumed-role/r/s</Arn></AssumedRoleUser></AssumeRoleWithWebIdentityResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></AssumeRoleWithWebIdentityResponse>"#
        );
        let (shared, client) = replay(vec![event(
            "https://sts.us-west-2.amazonaws.com/",
            200,
            &xml,
        )]);
        let rt = Runtime::with(
            &[
                ("AWS_WEB_IDENTITY_TOKEN_FILE", token.to_str().unwrap()),
                ("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/llmman"),
                ("AWS_REGION", "us-west-2"),
                ("AWS_EC2_METADATA_DISABLED", "true"),
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_CONFIG_FILE", dir.path().join("none").to_str().unwrap()),
                (
                    "AWS_SHARED_CREDENTIALS_FILE",
                    dir.path().join("none").to_str().unwrap(),
                ),
            ],
            conf_with(shared),
            "127.0.0.1:1",
        );
        let got = AwsChain::default()
            .credentials(&rt, now_ms_for_test())
            .unwrap();
        assert_eq!(got.access_key_id(), "AKIDWEB");
        assert_eq!(got.session_token(), Some("web-session"));
        let sent = client.requests();
        assert_eq!(sent.len(), 1);
        let body = String::from_utf8_lossy(sent[0].body().bytes().unwrap()).to_string();
        assert!(body.contains("Action=AssumeRoleWithWebIdentity"), "{body}");
        assert!(body.contains("WebIdentityToken=web-identity-jwt"), "{body}");
        assert!(
            body.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fllmman"),
            "{body}"
        );
    }

    /// A shared-config profile with `role_arn` + `source_profile`: the
    /// source's keys assume the role at STS.
    #[test]
    fn a_profile_that_assumes_a_role_through_a_source_profile() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        let credentials = dir.path().join("credentials");
        std::fs::write(
            &config,
            "[profile admin]\nrole_arn = arn:aws:iam::123456789012:role/admin\nsource_profile = base\nregion = us-east-2\n",
        )
        .unwrap();
        std::fs::write(
            &credentials,
            "[base]\naws_access_key_id = AKIDBASE\naws_secret_access_key = base-secret\n",
        )
        .unwrap();
        let xml = format!(
            r#"<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><AssumeRoleResult><Credentials><AccessKeyId>AKIDROLE</AccessKeyId><SecretAccessKey>role-secret</SecretAccessKey><SessionToken>role-session</SessionToken><Expiration>{}</Expiration></Credentials><AssumedRoleUser><AssumedRoleId>AROA:s</AssumedRoleId><Arn>arn:aws:sts::1:assumed-role/admin/s</Arn></AssumedRoleUser></AssumeRoleResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></AssumeRoleResponse>"#,
            future_time()
        );
        let (shared, client) = replay(vec![event(
            "https://sts.us-east-2.amazonaws.com/",
            200,
            &xml,
        )]);
        let rt = Runtime::with(
            &[
                ("AWS_CONFIG_FILE", config.to_str().unwrap()),
                ("AWS_SHARED_CREDENTIALS_FILE", credentials.to_str().unwrap()),
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_EC2_METADATA_DISABLED", "true"),
            ],
            conf_with(shared),
            "127.0.0.1:1",
        );
        let chain = AwsChain {
            profile: Some("admin".into()),
            ..AwsChain::default()
        };
        assert_eq!(chain.region(&rt), "us-east-2");
        let got = chain.credentials(&rt, now_ms_for_test()).unwrap();
        assert_eq!(got.access_key_id(), "AKIDROLE");
        let sent = client.requests();
        assert_eq!(sent.len(), 1);
        let auth = sent[0].headers().get("authorization").unwrap().to_string();
        assert!(
            auth.contains("Credential=AKIDBASE/"),
            "the source profile signs the call: {auth}"
        );
    }

    /// EC2 instance metadata, IMDSv2: a token is fetched first, then the
    /// role name, then its credentials.
    #[test]
    fn ec2_instance_credentials_come_from_imdsv2() {
        let expiry = future_time();
        let creds = format!(
            r#"{{"Code":"Success","Type":"AWS-HMAC","AccessKeyId":"AKIDIMDS","SecretAccessKey":"imds-secret","Token":"imds-token","Expiration":"{expiry}"}}"#
        );
        let (shared, client) = replay(vec![
            event(
                "http://169.254.169.254/latest/api/token",
                200,
                "imds-session-token",
            ),
            event(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
                200,
                "llmman-role",
            ),
            event(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/llmman-role",
                200,
                &creds,
            ),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let rt = Runtime::with(
            &[
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_CONFIG_FILE", dir.path().join("none").to_str().unwrap()),
                (
                    "AWS_SHARED_CREDENTIALS_FILE",
                    dir.path().join("none").to_str().unwrap(),
                ),
                ("AWS_REGION", "us-east-1"),
            ],
            conf_with(shared),
            "127.0.0.1:1",
        );
        // `forced`: nothing on the (test) machine hints at EC2.
        let chain = AwsChain {
            forced: true,
            ..AwsChain::default()
        };
        let got = chain.credentials(&rt, now_ms_for_test()).unwrap();
        assert_eq!(got.access_key_id(), "AKIDIMDS");
        assert_eq!(got.session_token(), Some("imds-token"));
        let sent = client.requests();
        assert_eq!(sent.len(), 3);
        assert_eq!(
            sent[0].method(),
            "PUT",
            "IMDSv2 starts with a session token"
        );
        assert!(sent[1].headers().get("x-aws-ec2-metadata-token").is_some());
    }

    #[test]
    fn a_chain_with_no_credentials_says_so_without_a_secret() {
        let dir = tempfile::tempdir().unwrap();
        let (shared, _) = replay(Vec::new());
        let rt = Runtime::with(
            &[
                ("HOME", dir.path().to_str().unwrap()),
                ("AWS_CONFIG_FILE", dir.path().join("none").to_str().unwrap()),
                (
                    "AWS_SHARED_CREDENTIALS_FILE",
                    dir.path().join("none").to_str().unwrap(),
                ),
                ("AWS_EC2_METADATA_DISABLED", "true"),
                ("AWS_REGION", "us-east-1"),
            ],
            conf_with(shared),
            "127.0.0.1:1",
        );
        let chain = AwsChain {
            forced: true,
            ..AwsChain::default()
        };
        let err = chain.credentials(&rt, now_ms_for_test()).unwrap_err();
        assert!(err.to_string().contains("AWS credentials"), "{err}");
    }

    #[test]
    fn the_aws_table_overlays_field_by_field() {
        let base = AwsChain {
            service: Some("bedrock".into()),
            ..AwsChain::default()
        };
        let over = AwsChain {
            profile: Some("p".into()),
            region: Some("us-west-2".into()),
            forced: true,
            ..AwsChain::default()
        };
        let merged = base.overlaid(&over);
        assert_eq!(merged.service.as_deref(), Some("bedrock"));
        assert_eq!(merged.profile.as_deref(), Some("p"));
        assert!(merged.forced);
        assert!(AwsChain {
            region: Some("a.b".into()),
            ..AwsChain::default()
        }
        .validate()
        .is_err());
        assert!(AwsChain {
            service: Some("a b".into()),
            ..AwsChain::default()
        }
        .validate()
        .is_err());
    }
}
