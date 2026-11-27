//! Azure OpenAI and AI Foundry: Microsoft Entra ID tokens, the alternative to
//! the `api-key` header.
//!
//! A bearer token for `https://cognitiveservices.azure.com/.default` from the
//! first source that has what it needs, in `DefaultAzureCredential` order: a
//! client secret, a federated (workload identity) token file, a managed
//! identity, then the Azure CLI. Tokens are cached until a minute before
//! expiry.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::credentials::{Resolved, Secret, Status, EXPIRY_SKEW_MS};
use crate::logins::{token_reply, Http};
use crate::tokensvc::{bearer, cache, cache_key, expiry_ms, Ctx, Grant, Minted, TokenService};

/// litellm `llms/azure/common_utils.py` `get_azure_ad_token`.
pub const DEFAULT_SCOPE: &str = "https://cognitiveservices.azure.com/.default";
/// litellm `get_azure_ad_token_from_oidc`'s `AZURE_AUTHORITY_HOST` default.
const DEFAULT_AUTHORITY: &str = "https://login.microsoftonline.com";
/// Microsoft Learn, "How to use managed identities for Azure resources on a
/// virtual machine".
const IMDS: &str = "http://169.254.169.254";
const IMDS_API_VERSION: &str = "2018-02-01";
/// Microsoft Learn, "How to use managed identities for App Service".
const APP_SERVICE_API_VERSION: &str = "2019-08-01";
const CLI_TIMEOUT: Duration = Duration::from_secs(15);

/// The `azure = {...}` table of a provider in `llmman.conf`. Every field is
/// optional: the environment (`AZURE_*`) fills what is not here.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AzureChain {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// The app registration, or a user-assigned managed identity's client id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// At most one of the three.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_file: Option<String>,
    /// A federated token file, instead of `AZURE_FEDERATED_TOKEN_FILE`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub federated_token_file: Option<String>,
    /// Try the managed identity endpoints even when no variable hints at one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_identity: Option<bool>,
    /// Try `az account get-access-token`. On by default only when `az` has a
    /// login on this machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cli: Option<bool>,
    /// The token's scope, `AZURE_SCOPE`, else [`DEFAULT_SCOPE`]. Foundry
    /// endpoints may want `https://ai.azure.com/.default`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// The login authority, `AZURE_AUTHORITY_HOST`, else Azure public cloud.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority_host: Option<String>,
    /// Set when `llmman.conf` has an `azure` table for the provider.
    #[serde(skip)]
    pub forced: bool,
}

impl std::fmt::Debug for AzureChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureChain")
            .field("tenant_id", &self.tenant_id)
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "<redacted>"),
            )
            .field("client_secret_env", &self.client_secret_env)
            .field("client_secret_file", &self.client_secret_file)
            .field("federated_token_file", &self.federated_token_file)
            .field("managed_identity", &self.managed_identity)
            .field("cli", &self.cli)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl AzureChain {
    pub fn overlaid(&self, over: &AzureChain) -> AzureChain {
        let pick = |a: &Option<String>, b: &Option<String>| a.clone().or_else(|| b.clone());
        // The secret fields are alternatives: the later table's choice
        // replaces the earlier one's, whichever of the three it is.
        let secret_set = over.client_secret.is_some()
            || over.client_secret_env.is_some()
            || over.client_secret_file.is_some();
        AzureChain {
            tenant_id: pick(&over.tenant_id, &self.tenant_id),
            client_id: pick(&over.client_id, &self.client_id),
            client_secret: if secret_set {
                over.client_secret.clone()
            } else {
                self.client_secret.clone()
            },
            client_secret_env: if secret_set {
                over.client_secret_env.clone()
            } else {
                self.client_secret_env.clone()
            },
            client_secret_file: if secret_set {
                over.client_secret_file.clone()
            } else {
                self.client_secret_file.clone()
            },
            federated_token_file: pick(&over.federated_token_file, &self.federated_token_file),
            managed_identity: over.managed_identity.or(self.managed_identity),
            cli: over.cli.or(self.cli),
            scope: pick(&over.scope, &self.scope),
            authority_host: pick(&over.authority_host, &self.authority_host),
            forced: self.forced || over.forced,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let secrets = [
            &self.client_secret,
            &self.client_secret_env,
            &self.client_secret_file,
        ];
        if secrets.iter().filter(|s| s.is_some()).count() > 1 {
            return Err(
                "client_secret, client_secret_env and client_secret_file are alternatives; set one"
                    .into(),
            );
        }
        if let Some(t) = &self.tenant_id {
            tenant_ok(t)?;
        }
        if let Some(c) = &self.client_id {
            client_ok(c)?;
        }
        if let Some(a) = &self.authority_host {
            authority_ok(a)?;
        }
        if let Some(s) = &self.scope {
            if s.trim().is_empty() || s.contains(['\n', '\r']) {
                return Err("scope: is not a scope".into());
            }
        }
        Ok(())
    }
}

fn tenant_ok(tenant: &str) -> Result<(), String> {
    if !tenant.is_empty()
        && tenant.len() <= 128
        && tenant
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
    {
        Ok(())
    } else {
        Err(format!(
            "tenant_id: {tenant:?} is not a tenant id or domain"
        ))
    }
}

fn client_ok(client: &str) -> Result<(), String> {
    if !client.is_empty()
        && client.len() <= 128
        && client
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    {
        Ok(())
    } else {
        Err(format!("client_id: {client:?} is not a client id"))
    }
}

/// A login authority is an origin: it has the tenant appended.
fn authority_ok(authority: &str) -> Result<(), String> {
    match reqwest::Url::parse(authority) {
        Ok(u)
            if matches!(u.scheme(), "https" | "http")
                && u.host_str().is_some()
                && u.username().is_empty()
                && u.password().is_none()
                && u.query().is_none()
                && u.fragment().is_none()
                && u.path().trim_matches('/').is_empty() =>
        {
            Ok(())
        }
        _ => Err(format!("authority_host: {authority:?} is not an origin")),
    }
}

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

/// The environment, files, processes and network the lookup runs against.
pub struct Runtime {
    env: Option<BTreeMap<String, String>>,
    home: Option<PathBuf>,
    /// The VM metadata service's origin; [`IMDS`] unless a test says.
    imds: String,
    /// The program `az` is run as.
    az: String,
}

impl Runtime {
    /// The process's own environment. Under test it is an empty one with a
    /// home that does not exist: tests never read the real Azure CLI login.
    pub fn live() -> Self {
        Self {
            env: cfg!(test).then(BTreeMap::new),
            home: if cfg!(test) {
                Some(PathBuf::from("/nonexistent-llmman-test-home"))
            } else {
                crate::config::home_dir()
            },
            imds: IMDS.into(),
            az: "az".into(),
        }
    }

    #[cfg(test)]
    pub fn with(env: &[(&str, &str)], home: &std::path::Path, imds: &str, az: &str) -> Self {
        Self {
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            home: Some(home.to_path_buf()),
            imds: imds.into(),
            az: az.into(),
        }
    }

    pub fn get(&self, name: &str) -> Option<String> {
        match &self.env {
            Some(fixed) => fixed
                .get(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty()),
            None => crate::providers::key_from_env(name),
        }
    }

    /// Where the Azure CLI keeps its login: `AZURE_CONFIG_DIR`, else `~/.azure`.
    fn cli_dir(&self) -> Option<PathBuf> {
        self.get("AZURE_CONFIG_DIR")
            .map(|d| crate::credentials::expand_home(&d))
            .or_else(|| self.home.as_ref().map(|h| h.join(".azure")))
    }
}

// ---------------------------------------------------------------------------
// What is configured
// ---------------------------------------------------------------------------

/// The pieces of the chain once the table and the environment are read.
struct Settings {
    tenant: Option<String>,
    client: Option<String>,
    secret: Option<Secret>,
    federated_file: Option<PathBuf>,
    scope: String,
    authority: String,
}

impl AzureChain {
    fn settings(&self, rt: &Runtime) -> anyhow::Result<Settings> {
        let secret = if let Some(s) = &self.client_secret {
            Some(Secret::new(s.trim().to_string()))
        } else if let Some(var) = &self.client_secret_env {
            rt.get(var).map(Secret::new)
        } else if let Some(path) = &self.client_secret_file {
            Some(
                crate::credentials::read_external(&crate::credentials::ExternalSource::File {
                    path: path.clone(),
                })?
                .0,
            )
        } else {
            rt.get("AZURE_CLIENT_SECRET").map(Secret::new)
        };
        let authority = self
            .authority_host
            .clone()
            .or_else(|| rt.get("AZURE_AUTHORITY_HOST"))
            .unwrap_or_else(|| DEFAULT_AUTHORITY.to_string());
        authority_ok(&authority).map_err(|e| anyhow!(e))?;
        let tenant = self.tenant_id.clone().or_else(|| rt.get("AZURE_TENANT_ID"));
        if let Some(t) = &tenant {
            tenant_ok(t).map_err(|e| anyhow!("AZURE_TENANT_ID: {e}"))?;
        }
        let client = self.client_id.clone().or_else(|| rt.get("AZURE_CLIENT_ID"));
        if let Some(c) = &client {
            client_ok(c).map_err(|e| anyhow!("AZURE_CLIENT_ID: {e}"))?;
        }
        Ok(Settings {
            tenant,
            client,
            secret,
            federated_file: self
                .federated_token_file
                .clone()
                .or_else(|| rt.get("AZURE_FEDERATED_TOKEN_FILE"))
                .map(|f| crate::credentials::expand_home(&f)),
            scope: self
                .scope
                .clone()
                .or_else(|| rt.get("AZURE_SCOPE"))
                .unwrap_or_else(|| DEFAULT_SCOPE.to_string()),
            authority: authority.trim_end_matches('/').to_string(),
        })
    }

    fn cli_login(&self, rt: &Runtime) -> bool {
        rt.cli_dir()
            .is_some_and(|d| d.join("azureProfile.json").is_file())
    }

    fn wants_cli(&self, rt: &Runtime) -> bool {
        self.cli.unwrap_or_else(|| self.cli_login(rt))
    }

    fn wants_managed_identity(&self, rt: &Runtime) -> bool {
        self.managed_identity.unwrap_or(false)
            || rt.get("IDENTITY_ENDPOINT").is_some()
            // A client id with nothing to go with it is a user-assigned
            // managed identity (litellm `infer_credential_type_from_environment`).
            || (rt.get("AZURE_CLIENT_ID").is_some()
                && rt.get("AZURE_CLIENT_SECRET").is_none()
                && rt.get("AZURE_FEDERATED_TOKEN_FILE").is_none())
            || (self.client_id.is_some() && self.client_secret().is_none() && self.federated_token_file.is_none())
    }

    fn client_secret(&self) -> Option<()> {
        (self.client_secret.is_some()
            || self.client_secret_env.is_some()
            || self.client_secret_file.is_some())
        .then_some(())
    }

    /// Whether anything says Entra credentials are there to find. The
    /// answer is a guess; resolving proves it.
    pub fn hinted(&self, rt: &Runtime) -> bool {
        if self.forced || self.wants_cli(rt) || self.wants_managed_identity(rt) {
            return true;
        }
        let have_client = self.client_id.is_some() || rt.get("AZURE_CLIENT_ID").is_some();
        let have_tenant = self.tenant_id.is_some() || rt.get("AZURE_TENANT_ID").is_some();
        let have_secret = self.client_secret().is_some() || rt.get("AZURE_CLIENT_SECRET").is_some();
        let have_federated =
            self.federated_token_file.is_some() || rt.get("AZURE_FEDERATED_TOKEN_FILE").is_some();
        have_client && have_tenant && (have_secret || have_federated)
    }

    /// `ready` when [`AzureChain::hinted`]. No network, no process.
    pub fn status(&self, rt: &Runtime) -> Option<Status> {
        if !self.hinted(rt) {
            return None;
        }
        Some(Status {
            kind: "entra",
            state: "ready",
            expires_at_ms: self
                .settings(rt)
                .ok()
                .and_then(|s| cache().expires_at(&cache_key(&["azure", &s.scope, &s.authority]))),
        })
    }

    /// A bearer credential from the first source that has what it needs.
    /// `Ok(None)` when nothing hints at Entra credentials. Blocking.
    pub fn resolve(&self, rt: &Runtime, http: &Http, now: u64) -> anyhow::Result<Option<Resolved>> {
        if !self.hinted(rt) {
            return Ok(None);
        }
        let s = self.settings(rt)?;
        let key = |what: &str, extra: &[&str]| {
            cache_key(&[&["azure", what, &s.scope, &s.authority][..], extra].concat())
        };
        let minted =
            if let (Some(tenant), Some(client), Some(secret)) = (&s.tenant, &s.client, &s.secret) {
                // Entra's v2.0 token endpoint, with the secret in the form.
                let service = TokenService {
                    grant: Grant::ClientCredentials,
                    token_url: Some(format!("{}/{tenant}/oauth2/v2.0/token", s.authority)),
                    client_id: Some(client.clone()),
                    client_secret: Some(secret.expose().to_string()),
                    scope: Some(s.scope.clone()),
                    ..TokenService::default()
                };
                let env = |_: &str| None;
                return service
                    .resolve(&Ctx {
                        env: &env,
                        http,
                        now,
                    })
                    .map(|mut r| {
                        r.kind = "entra";
                        Some(r)
                    });
            } else if let (Some(tenant), Some(client), Some(file)) =
                (&s.tenant, &s.client, &s.federated_file)
            {
                cache().get_or_mint(
                    &key("federated", &[tenant, client, &file.to_string_lossy()]),
                    now,
                    EXPIRY_SKEW_MS,
                    || federated(http, &s, tenant, client, file, now),
                )?
            } else if self.wants_managed_identity(rt) {
                let client = s.client.clone().unwrap_or_default();
                cache().get_or_mint(&key("managed", &[&client]), now, EXPIRY_SKEW_MS, || {
                    managed_identity(rt, &s, now)
                })?
            } else if self.wants_cli(rt) {
                cache().get_or_mint(&key("cli", &[]), now, EXPIRY_SKEW_MS, || cli(rt, &s))?
            } else {
                bail!(
                    "no Azure credential is set up: set AZURE_TENANT_ID, AZURE_CLIENT_ID and \
                 AZURE_CLIENT_SECRET (or AZURE_FEDERATED_TOKEN_FILE), run on a managed identity, \
                 or sign in with `az login`"
                );
            };
        Ok(Some(bearer(minted.token, "entra")))
    }
}

// ---------------------------------------------------------------------------
// The sources
// ---------------------------------------------------------------------------

fn fail(what: &str, e: crate::credentials::RefreshError) -> anyhow::Error {
    match e {
        crate::credentials::RefreshError::Rejected(why) => {
            anyhow!("{what} refused the request ({why})")
        }
        crate::credentials::RefreshError::Transient(why) => anyhow!("{what} failed: {why}"),
    }
}

fn minted(tokens: crate::logins::Tokens, now: u64) -> Minted {
    Minted {
        expires_ms: tokens.expires_ms(now),
        token: Secret::new(tokens.access),
    }
}

/// A federated token file's JWT as a `client_assertion`: litellm
/// `llms/azure/common_utils.py` `_exchange_oidc_token`.
fn federated(
    http: &Http,
    s: &Settings,
    tenant: &str,
    client: &str,
    file: &std::path::Path,
    now: u64,
) -> anyhow::Result<Minted> {
    let assertion = std::fs::read_to_string(file)
        .with_context(|| format!("read the federated token file {}", file.display()))?;
    let assertion = assertion.trim();
    anyhow::ensure!(
        !assertion.is_empty(),
        "the federated token file {} is empty",
        file.display()
    );
    let url = format!("{}/{tenant}/oauth2/v2.0/token", s.authority);
    let reply = http.post_form(
        &url,
        &[
            ("client_id", client),
            ("grant_type", "client_credentials"),
            ("scope", &s.scope),
            (
                "client_assertion_type",
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            ),
            ("client_assertion", assertion),
        ],
        &[],
    );
    token_reply(reply)
        .map(|t| minted(t, now))
        .map_err(|e| fail("Entra's token endpoint", e))
}

/// The managed identity endpoints take the scope's resource, without
/// `/.default`.
fn resource_of(scope: &str) -> &str {
    scope.strip_suffix("/.default").unwrap_or(scope)
}

/// App Service's local token service when `IDENTITY_ENDPOINT` is set, else
/// the VM's IMDS.
fn managed_identity(rt: &Runtime, s: &Settings, now: u64) -> anyhow::Result<Minted> {
    let http = Http::metadata();
    let resource = resource_of(&s.scope);
    let reply = match (rt.get("IDENTITY_ENDPOINT"), rt.get("IDENTITY_HEADER")) {
        (Some(endpoint), Some(header)) => {
            let mut url = reqwest::Url::parse(&endpoint)
                .map_err(|_| anyhow!("IDENTITY_ENDPOINT is not a URL"))?;
            {
                let mut q = url.query_pairs_mut();
                q.append_pair("resource", resource)
                    .append_pair("api-version", APP_SERVICE_API_VERSION);
                if let Some(client) = &s.client {
                    q.append_pair("client_id", client);
                }
            }
            http.get(url.as_str(), &[("x-identity-header", &header)])
        }
        (Some(_), None) => bail!("IDENTITY_ENDPOINT is set without IDENTITY_HEADER"),
        _ => {
            let mut url =
                reqwest::Url::parse(&format!("{}/metadata/identity/oauth2/token", rt.imds))
                    .map_err(|_| anyhow!("the IMDS address is not a URL"))?;
            {
                let mut q = url.query_pairs_mut();
                q.append_pair("api-version", IMDS_API_VERSION)
                    .append_pair("resource", resource);
                if let Some(client) = &s.client {
                    q.append_pair("client_id", client);
                }
            }
            http.get(url.as_str(), &[("metadata", "true")])
        }
    };
    let reply = reply.map_err(|e| anyhow!("the managed identity endpoint failed: {e}"))?;
    if !reply.ok() {
        bail!(
            "the managed identity endpoint answered HTTP {}",
            reply.status
        );
    }
    let body = reply.json();
    let token = body
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| anyhow!("the managed identity reply had no access token"))?;
    Ok(Minted {
        token: Secret::new(token.to_string()),
        expires_ms: expiry_ms(&body, now),
    })
}

/// `az account get-access-token --scope <scope> --output json`: opencode
/// `plugin/provider/azure.ts` `token`, which reads `accessToken` and
/// `expires_on` (epoch seconds) or else `expiresOn` (a local time).
fn cli(rt: &Runtime, s: &Settings) -> anyhow::Result<Minted> {
    let mut command = std::process::Command::new(&rt.az);
    command
        .args([
            "account",
            "get-access-token",
            "--scope",
            &s.scope,
            "--output",
            "json",
        ])
        .env_remove("AZURE_CLIENT_SECRET");
    let out =
        crate::credentials::run_capture(&mut command, "`az account get-access-token`", CLI_TIMEOUT)
            .context("is the Azure CLI installed and signed in (`az login`)?")?;
    let body: Value = serde_json::from_slice(&out)
        .map_err(|_| anyhow!("`az account get-access-token` printed something that is not JSON"))?;
    let token = body
        .get("accessToken")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| anyhow!("`az account get-access-token` returned no accessToken"))?;
    let expires_ms = match body.get("expires_on") {
        Some(Value::Number(n)) => n.as_u64().map(|s| s * 1000),
        Some(Value::String(t)) => t.trim().parse::<u64>().ok().map(|s| s * 1000),
        _ => None,
    }
    .or_else(|| {
        let text = body.get("expiresOn")?.as_str()?;
        local_ms(text)
    })
    .ok_or_else(|| anyhow!("the Azure CLI returned an invalid token expiration"))?;
    Ok(Minted {
        token: Secret::new(token.to_string()),
        expires_ms,
    })
}

/// `expiresOn` as the CLI prints it, `2026-01-02 03:04:05.678901`, in the
/// machine's local time.
fn local_ms(text: &str) -> Option<u64> {
    use chrono::TimeZone as _;
    let naive = chrono::NaiveDateTime::parse_from_str(text.trim(), "%Y-%m-%d %H:%M:%S%.f").ok()?;
    let local = chrono::Local.from_local_datetime(&naive).earliest()?;
    u64::try_from(local.timestamp_millis()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::now_ms;
    use crate::testserver::server;
    use serde_json::json;

    fn rt(env: &[(&str, &str)], dir: &std::path::Path, imds: &str, az: &str) -> Runtime {
        Runtime::with(env, dir, imds, az)
    }

    fn http() -> Http {
        Http::tokens()
    }

    /// A client secret: `client_credentials` at the tenant's v2.0 endpoint,
    /// with the cognitive services scope.
    #[test]
    fn a_client_secret_gets_an_entra_token() {
        let server = server();
        server.fake.on(
            "/tenant-secret/oauth2/v2.0/token",
            vec![(
                200,
                json!({"access_token": "eyJ.entra", "expires_in": 3599, "token_type": "Bearer"}),
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let rt = rt(
            &[
                ("AZURE_TENANT_ID", "tenant-secret"),
                ("AZURE_CLIENT_ID", "client-1"),
                ("AZURE_CLIENT_SECRET", "s3cret"),
                ("AZURE_AUTHORITY_HOST", &server.origin),
            ],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        let chain = AzureChain::default();
        assert!(chain.hinted(&rt));
        let got = chain.resolve(&rt, &http(), now_ms()).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "eyJ.entra");
        assert_eq!(got.kind, "entra");
        assert_eq!(got.style, crate::credentials::Style::Bearer);
        let form = server.fake.hits("/tenant-secret/oauth2/v2.0/token")[0].form();
        assert_eq!(form["grant_type"], "client_credentials");
        assert_eq!(form["client_id"], "client-1");
        assert_eq!(form["client_secret"], "s3cret");
        assert_eq!(form["scope"], DEFAULT_SCOPE);
        // Cached across requests.
        chain.resolve(&rt, &http(), now_ms()).unwrap().unwrap();
        assert_eq!(
            server.fake.hits("/tenant-secret/oauth2/v2.0/token").len(),
            1
        );
        assert!(!format!("{got:?}").contains("eyJ.entra"));
    }

    /// Workload identity: the projected token file is the client assertion.
    #[test]
    fn a_federated_token_file_is_exchanged_as_a_client_assertion() {
        let server = server();
        server.fake.on(
            "/tenant-fed/oauth2/v2.0/token",
            vec![(
                200,
                json!({"access_token": "eyJ.federated", "expires_in": 3599}),
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("azure-identity-token");
        std::fs::write(&token, "projected-jwt\n").unwrap();
        let rt = rt(
            &[
                ("AZURE_TENANT_ID", "tenant-fed"),
                ("AZURE_CLIENT_ID", "client-fed"),
                ("AZURE_FEDERATED_TOKEN_FILE", token.to_str().unwrap()),
                ("AZURE_AUTHORITY_HOST", &server.origin),
            ],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        let got = AzureChain::default()
            .resolve(&rt, &http(), now_ms())
            .unwrap()
            .unwrap();
        assert_eq!(got.secret.expose(), "eyJ.federated");
        let form = server.fake.hits("/tenant-fed/oauth2/v2.0/token")[0].form();
        assert_eq!(form["client_assertion"], "projected-jwt");
        assert_eq!(
            form["client_assertion_type"],
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        );
        assert_eq!(form["grant_type"], "client_credentials");
        assert_eq!(form["scope"], DEFAULT_SCOPE);
        assert!(!form.contains_key("client_secret"));
    }

    /// A VM's managed identity: IMDS, with `Metadata: true`, asking for the
    /// resource (the scope without `/.default`), and a user-assigned
    /// identity by client id.
    #[test]
    fn imds_gives_a_managed_identity_token() {
        let server = server();
        server.fake.on(
            "/metadata/identity/oauth2/token",
            vec![(200, json!({"access_token": "eyJ.imds", "expires_in": "3599", "expires_on": "1506484173"}))],
        );
        let dir = tempfile::tempdir().unwrap();
        let rt = rt(
            &[("AZURE_CLIENT_ID", "user-assigned-1")],
            dir.path(),
            &server.origin,
            "az",
        );
        let chain = AzureChain::default();
        assert!(chain.hinted(&rt));
        let got = chain.resolve(&rt, &http(), now_ms()).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "eyJ.imds");
        let hit = &server.fake.hits("/metadata/identity/oauth2/token")[0];
        assert_eq!(hit.headers["metadata"], "true");
        assert!(
            hit.query
                .contains("resource=https%3A%2F%2Fcognitiveservices.azure.com"),
            "{}",
            hit.query
        );
        assert!(!hit.query.contains(".default"), "{}", hit.query);
        assert!(hit.query.contains("api-version=2018-02-01"));
        assert!(hit.query.contains("client_id=user-assigned-1"));
    }

    /// App Service and Functions: the local token service named by
    /// `IDENTITY_ENDPOINT`, with `X-IDENTITY-HEADER`.
    #[test]
    fn app_service_managed_identity_uses_its_local_endpoint() {
        let server = server();
        server.fake.on(
            "/msi/token",
            vec![(
                200,
                json!({"access_token": "eyJ.appservice", "expires_on": "4102444800"}),
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let endpoint = format!("{}/msi/token", server.origin);
        let rt = rt(
            &[
                ("IDENTITY_ENDPOINT", &endpoint),
                ("IDENTITY_HEADER", "hdr-value"),
            ],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        let got = AzureChain::default()
            .resolve(&rt, &http(), now_ms())
            .unwrap()
            .unwrap();
        assert_eq!(got.secret.expose(), "eyJ.appservice");
        let hit = &server.fake.hits("/msi/token")[0];
        assert_eq!(hit.headers["x-identity-header"], "hdr-value");
        assert!(
            hit.query.contains("api-version=2019-08-01"),
            "{}",
            hit.query
        );
        assert!(hit.query.contains("resource="), "{}", hit.query);
        // Without the header, nothing is sent.
        let rt = Runtime::with(
            &[
                ("IDENTITY_ENDPOINT", &endpoint),
                ("AZURE_SCOPE", "https://other.example/.default"),
            ],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        let err = AzureChain::default()
            .resolve(&rt, &http(), now_ms())
            .unwrap_err();
        assert!(err.to_string().contains("IDENTITY_HEADER"), "{err}");
    }

    /// The Azure CLI as an external source: a stand-in `az` prints the JSON,
    /// and its arguments are the ones opencode passes.
    #[cfg(unix)]
    #[test]
    fn the_azure_cli_is_an_external_token_source() {
        let dir = tempfile::tempdir().unwrap();
        let az = dir.path().join("az");
        let log = dir.path().join("args");
        crate::testscript::write_executable(
            &az,
            format!(
                "#!/bin/sh\necho \"$@\" > {}\nprintf '{{\"accessToken\":\"eyJ.cli\",\"expires_on\":4102444800,\"tokenType\":\"Bearer\"}}'\n",
                log.display()
            ),
        );
        std::fs::create_dir_all(dir.path().join(".azure")).unwrap();
        std::fs::write(dir.path().join(".azure").join("azureProfile.json"), "{}").unwrap();
        let rt = rt(&[], dir.path(), "http://127.0.0.1:1", az.to_str().unwrap());
        let chain = AzureChain::default();
        assert!(chain.hinted(&rt), "an `az login` is a hint");
        let got = chain.resolve(&rt, &http(), now_ms()).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "eyJ.cli");
        assert_eq!(
            std::fs::read_to_string(&log).unwrap().trim(),
            format!("account get-access-token --scope {DEFAULT_SCOPE} --output json")
        );
        // A failing `az` is an error that does not print its output.
        crate::testscript::write_executable(&az, "#!/bin/sh\necho sk-cli-leak\nexit 1\n");
        let cached_elsewhere = AzureChain {
            scope: Some("https://ai.azure.com/.default".into()),
            ..AzureChain::default()
        };
        let err = cached_elsewhere
            .resolve(&rt, &http(), now_ms())
            .unwrap_err();
        assert!(!format!("{err:#}").contains("sk-cli-leak"), "{err:#}");
    }

    #[test]
    fn the_cli_expiry_forms() {
        assert!(local_ms("2026-01-02 03:04:05.678901").is_some());
        assert!(local_ms("2026-01-02 03:04:05").is_some());
        assert!(local_ms("tomorrow").is_none());
        assert_eq!(
            resource_of(DEFAULT_SCOPE),
            "https://cognitiveservices.azure.com"
        );
        assert_eq!(resource_of("https://x"), "https://x");
    }

    #[test]
    fn nothing_hinting_resolves_to_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let bare = rt(&[], dir.path(), "http://127.0.0.1:1", "az");
        let chain = AzureChain::default();
        assert!(!chain.hinted(&bare));
        assert_eq!(chain.status(&bare), None);
        assert!(chain.resolve(&bare, &http(), now_ms()).unwrap().is_none());
        // `azure = {}` in llmman.conf says to try.
        let forced = AzureChain {
            forced: true,
            ..AzureChain::default()
        };
        assert!(forced.hinted(&bare));
        let err = forced.resolve(&bare, &http(), now_ms()).unwrap_err();
        assert!(err.to_string().contains("no Azure credential"), "{err}");
        // A tenant and client with neither secret nor federation is not a hint.
        let partial = rt(
            &[("AZURE_TENANT_ID", "t"), ("AZURE_CLIENT_ID", "c")],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        assert!(
            chain.hinted(&partial),
            "a lone client id is a user-assigned identity"
        );
        let tenant_only = rt(
            &[("AZURE_TENANT_ID", "t")],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        assert!(!chain.hinted(&tenant_only));
    }

    #[test]
    fn tables_validate_and_overlay() {
        assert!(AzureChain {
            tenant_id: Some("a/b".into()),
            ..AzureChain::default()
        }
        .validate()
        .is_err());
        assert!(AzureChain {
            client_id: Some("a b".into()),
            ..AzureChain::default()
        }
        .validate()
        .is_err());
        assert!(AzureChain {
            authority_host: Some("https://login.example.com/x".into()),
            ..AzureChain::default()
        }
        .validate()
        .is_err());
        assert!(AzureChain {
            authority_host: Some("https://login.example.com".into()),
            ..AzureChain::default()
        }
        .validate()
        .is_ok());
        assert!(AzureChain {
            client_secret: Some("a".into()),
            client_secret_env: Some("B".into()),
            ..AzureChain::default()
        }
        .validate()
        .is_err());
        let base = AzureChain {
            client_secret: Some("a".into()),
            tenant_id: Some("t".into()),
            ..AzureChain::default()
        };
        let over = AzureChain {
            client_secret_env: Some("X".into()),
            ..AzureChain::default()
        };
        let merged = base.overlaid(&over);
        assert_eq!(
            merged.client_secret, None,
            "the later secret source replaces the earlier"
        );
        assert_eq!(merged.client_secret_env.as_deref(), Some("X"));
        assert_eq!(merged.tenant_id.as_deref(), Some("t"));
    }

    #[test]
    fn a_refused_token_request_names_the_endpoint_not_the_body() {
        let server = server();
        server.fake.on(
            "/tenant-bad/oauth2/v2.0/token",
            vec![(401, json!({"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret sk-body-leak"}))],
        );
        let dir = tempfile::tempdir().unwrap();
        let rt = rt(
            &[
                ("AZURE_TENANT_ID", "tenant-bad"),
                ("AZURE_CLIENT_ID", "client-bad"),
                ("AZURE_CLIENT_SECRET", "wrong"),
                ("AZURE_AUTHORITY_HOST", &server.origin),
            ],
            dir.path(),
            "http://127.0.0.1:1",
            "az",
        );
        let err = AzureChain::default()
            .resolve(&rt, &http(), now_ms())
            .unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("invalid_client")
                && !text.contains("sk-body-leak")
                && !text.contains("wrong"),
            "{text}"
        );
    }
}
