//! Google Vertex AI: Application Default Credentials and the endpoint they
//! are spent at.
//!
//! The token (`cloud-platform` scope) comes from the first of: the file
//! `GOOGLE_APPLICATION_CREDENTIALS` names (service account or authorized
//! user), gcloud's `application_default_credentials.json`, or the metadata
//! server. `external_account` files are refused. Express-mode API keys are
//! not supported: Vertex's OpenAI-compatible endpoint takes only Google
//! Cloud Auth.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::credentials::{Resolved, Status, EXPIRY_SKEW_MS};
use crate::logins::{token_reply, Http};
use crate::tokensvc::{bearer, cache, cache_key, Minted};

/// opencode `google-vertex-shared.ts` `SCOPE`.
pub const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
/// OpenClaw `vertex-adc.ts` `GOOGLE_OAUTH_TOKEN_URL`.
pub const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
/// litellm `secret_managers/main.py` `_GOOGLE_METADATA_DEFAULT_HOST`.
const METADATA_HOST: &str = "metadata.google.internal";
/// How long a signed assertion is good for: Google's maximum.
const ASSERTION_LIFETIME_SECS: u64 = 3600;

/// The `gcp = {...}` table of a provider in `llmman.conf`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GcpChain {
    /// A service-account or authorized-user file, instead of
    /// `GOOGLE_APPLICATION_CREDENTIALS` and the `gcloud` default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_file: Option<String>,
    /// The project, when `GOOGLE_VERTEX_PROJECT` and its aliases do not
    /// name one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// The location (`global`, `us`, `eu`, or a region).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Set when `llmman.conf` has a `gcp` table for the provider: the
    /// metadata server is tried whether or not anything hints at one.
    #[serde(skip)]
    pub forced: bool,
}

impl GcpChain {
    pub fn overlaid(&self, over: &GcpChain) -> GcpChain {
        GcpChain {
            credentials_file: over
                .credentials_file
                .clone()
                .or_else(|| self.credentials_file.clone()),
            project: over.project.clone().or_else(|| self.project.clone()),
            location: over.location.clone().or_else(|| self.location.clone()),
            forced: self.forced || over.forced,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(project) = &self.project {
            project_ok(project)?;
        }
        if let Some(location) = &self.location {
            location_ok(location)?;
        }
        Ok(())
    }
}

/// A project id or number: letters, digits, `-`, `_`, `.`, `:` (domain-scoped
/// ids). It lands in a URL path, so nothing else.
pub(crate) fn project_ok(project: &str) -> Result<(), String> {
    if !project.is_empty()
        && project.len() <= 128
        && project
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
    {
        Ok(())
    } else {
        Err(format!(
            "project: {project:?} is not a Google Cloud project id"
        ))
    }
}

/// `global`, `us`, `eu` or a region: a host label and a path segment.
pub(crate) fn location_ok(location: &str) -> Result<(), String> {
    if !location.is_empty()
        && location.len() <= 40
        && location
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !location.starts_with('-')
    {
        Ok(())
    } else {
        Err(format!(
            "location: {location:?} is not a Vertex AI location"
        ))
    }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/// The host of `location`'s Vertex endpoint: `aiplatform.googleapis.com`
/// for `global`, `aiplatform.<us|eu>.rep.googleapis.com` for the
/// multi-regions, `<region>-aiplatform.googleapis.com` otherwise. opencode
/// `google-vertex-shared.ts` `host`; litellm `vertex_ai/common_utils.py`
/// `get_vertex_base_url`.
pub fn host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".to_string(),
        "eu" | "us" => format!("aiplatform.{location}.rep.googleapis.com"),
        other => format!("{other}-aiplatform.googleapis.com"),
    }
}

/// Vertex AI's OpenAI-compatible Chat Completions base:
/// `https://<host>/v1/projects/<p>/locations/<l>/endpoints/openapi`
/// (Google's "Using OpenAI libraries" guide; opencode
/// `google-vertex-chat.ts`; litellm `vertex_model_garden/main.py`).
pub fn openapi_base(project: &str, location: &str) -> String {
    format!(
        "https://{}/v1/projects/{project}/locations/{location}/endpoints/openapi",
        host(location)
    )
}

/// The model id Vertex's OpenAI-compatible endpoint wants: Google's own
/// models are `google/<id>` there (the guide's `google/gemini-3.5-flash`),
/// where models.dev lists the bare name; a publisher-qualified id
/// (`meta/llama-...-maas`) is already right.
pub fn wire_model(model: &str) -> String {
    if model.starts_with("gemini") && !model.contains('/') {
        format!("google/{model}")
    } else {
        model.to_string()
    }
}

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

/// The environment, files and network the lookup runs against.
pub struct Runtime {
    env: Option<BTreeMap<String, String>>,
    home: Option<PathBuf>,
    /// Where tokens are asked for, instead of the file's own `token_uri` and
    /// [`TOKEN_URL`]: a test's local server.
    token_url: Option<String>,
    /// The metadata server's origin, instead of `http://<host>`.
    metadata_origin: Option<String>,
}

impl Runtime {
    /// The process's own environment. Under test it is an empty one with a
    /// home that does not exist: tests never read the real gcloud files.
    pub fn live() -> Self {
        Self {
            env: cfg!(test).then(BTreeMap::new),
            home: if cfg!(test) {
                Some(PathBuf::from("/nonexistent-llmman-test-home"))
            } else {
                crate::config::home_dir()
            },
            token_url: None,
            metadata_origin: None,
        }
    }

    #[cfg(test)]
    pub fn with(
        env: &[(&str, &str)],
        home: &std::path::Path,
        token_url: Option<&str>,
        metadata_origin: Option<&str>,
    ) -> Self {
        Self {
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            home: Some(home.to_path_buf()),
            token_url: token_url.map(str::to_string),
            metadata_origin: metadata_origin.map(str::to_string),
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

    /// `gcloud`'s configuration directory, as opencode's `loadADCClient`
    /// and OpenClaw's `resolveGoogleApplicationCredentialsPath` find it.
    fn gcloud_dir(&self) -> Option<PathBuf> {
        if let Some(dir) = self.get("CLOUDSDK_CONFIG") {
            return Some(crate::credentials::expand_home(&dir));
        }
        if cfg!(windows) {
            if let Some(appdata) = self.get("APPDATA") {
                return Some(PathBuf::from(appdata).join("gcloud"));
            }
        }
        self.home.as_ref().map(|h| h.join(".config").join("gcloud"))
    }

    fn well_known_file(&self) -> Option<PathBuf> {
        self.gcloud_dir()
            .map(|d| d.join("application_default_credentials.json"))
    }

    fn metadata_origin(&self) -> String {
        if let Some(origin) = &self.metadata_origin {
            return origin.clone();
        }
        let host = self
            .get("GCE_METADATA_HOST")
            .unwrap_or_else(|| METADATA_HOST.to_string());
        format!("http://{host}")
    }
}

// ---------------------------------------------------------------------------
// Project and location
// ---------------------------------------------------------------------------

impl GcpChain {
    /// The project: this table's, then `GOOGLE_VERTEX_PROJECT`,
    /// `GOOGLE_CLOUD_PROJECT`, `GCP_PROJECT`, `GCLOUD_PROJECT` (opencode
    /// `google-vertex-shared.ts` `project`), then what `gcloud` itself would
    /// use: the active configuration's `project`, then the credentials
    /// file's `project_id` / `quota_project_id` (opencode's
    /// `google-vertex.ts` reads the first and last; litellm reads a service
    /// account's `project_id`). No network.
    pub fn project(&self, rt: &Runtime) -> Option<String> {
        let from_env = || {
            [
                "GOOGLE_VERTEX_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
                "GCP_PROJECT",
                "GCLOUD_PROJECT",
            ]
            .iter()
            .find_map(|v| rt.get(v))
        };
        let from_gcloud = || {
            let dir = rt.gcloud_dir()?;
            let active = std::fs::read_to_string(dir.join("active_config"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "default".into());
            let text = std::fs::read_to_string(
                dir.join("configurations").join(format!("config_{active}")),
            )
            .ok()?;
            text.lines().find_map(|l| {
                let (k, v) = l.split_once('=')?;
                (k.trim() == "project")
                    .then(|| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
        };
        let from_file = || {
            let (_, file) = self.credentials_path(rt)?;
            let json: Value = serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()?;
            ["project_id", "quota_project_id"].iter().find_map(|k| {
                json.get(k)
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .filter(|v| !v.is_empty())
            })
        };
        self.project
            .clone()
            .or_else(from_env)
            .or_else(from_gcloud)
            .or_else(from_file)
            .filter(|p| project_ok(p).is_ok())
    }

    /// The location: this table's, then `GOOGLE_VERTEX_LOCATION`,
    /// `GOOGLE_CLOUD_LOCATION`, `VERTEX_LOCATION`, else `global` (opencode
    /// `google-vertex-shared.ts` `location`).
    pub fn location(&self, rt: &Runtime) -> String {
        self.location
            .clone()
            .or_else(|| {
                [
                    "GOOGLE_VERTEX_LOCATION",
                    "GOOGLE_CLOUD_LOCATION",
                    "VERTEX_LOCATION",
                ]
                .iter()
                .find_map(|v| rt.get(v))
            })
            .filter(|l| location_ok(l).is_ok())
            .unwrap_or_else(|| "global".into())
    }

    /// The credentials file that applies, and where it was named: this
    /// table's `credentials_file`, `GOOGLE_APPLICATION_CREDENTIALS`, or the
    /// `gcloud` well-known one if it exists.
    fn credentials_path(&self, rt: &Runtime) -> Option<(&'static str, PathBuf)> {
        if let Some(file) = &self.credentials_file {
            return Some(("llmman.conf", crate::credentials::expand_home(file)));
        }
        if let Some(file) = rt.get("GOOGLE_APPLICATION_CREDENTIALS") {
            return Some((
                "GOOGLE_APPLICATION_CREDENTIALS",
                crate::credentials::expand_home(&file),
            ));
        }
        rt.well_known_file()
            .filter(|p| p.is_file())
            .map(|p| ("gcloud", p))
    }

    /// Whether anything on this machine says Google credentials are there
    /// to find: an ADC file, `GCE_METADATA_HOST`, GCE's DMI product name,
    /// or `llmman.conf`'s `gcp` table.
    pub fn hinted(&self, rt: &Runtime) -> bool {
        self.forced
            || self.credentials_path(rt).is_some()
            || rt.get("GCE_METADATA_HOST").is_some()
            || on_gce()
    }

    /// `ready` when [`GcpChain::hinted`], with the expiry of the token in
    /// hand. No network.
    pub fn status(&self, rt: &Runtime) -> Option<Status> {
        if !self.hinted(rt) {
            return None;
        }
        Some(Status {
            kind: "adc",
            state: "ready",
            expires_at_ms: self
                .source(rt)
                .ok()
                .and_then(|s| cache().expires_at(&s.cache_key())),
        })
    }
}

/// GCE and GKE report themselves in DMI, which google-auth reads on Linux.
fn on_gce() -> bool {
    if cfg!(test) {
        return false;
    }
    std::fs::read_to_string("/sys/class/dmi/id/product_name")
        .is_ok_and(|v| v.trim().starts_with("Google"))
}

// ---------------------------------------------------------------------------
// Sources and tokens
// ---------------------------------------------------------------------------

/// Where a token comes from.
enum Source {
    ServiceAccount {
        email: String,
        key_id: Option<String>,
        private_key: String,
        token_uri: Option<String>,
    },
    AuthorizedUser {
        client_id: String,
        client_secret: String,
        refresh_token: String,
    },
    Metadata,
}

impl Source {
    fn cache_key(&self) -> String {
        match self {
            Self::ServiceAccount {
                email,
                key_id,
                private_key,
                token_uri,
            } => cache_key(&[
                "gcp-sa",
                email,
                key_id.as_deref().unwrap_or(""),
                private_key,
                token_uri.as_deref().unwrap_or(""),
            ]),
            Self::AuthorizedUser {
                client_id,
                refresh_token,
                ..
            } => cache_key(&["gcp-user", client_id, refresh_token]),
            Self::Metadata => cache_key(&["gcp-metadata"]),
        }
    }
}

/// Whether a service account's `token_uri` may be sent its signed
/// assertion: https, on a `googleapis.com` host. A key file is the
/// operator's, but the assertion is a credential, and a file fetched from
/// somewhere else should not be able to aim it.
pub fn token_uri_allowed(uri: &str) -> bool {
    reqwest::Url::parse(uri).is_ok_and(|u| {
        u.scheme() == "https"
            && u.username().is_empty()
            && u.port().is_none()
            && u.host_str()
                .is_some_and(|h| h == "googleapis.com" || h.ends_with(".googleapis.com"))
    })
}

impl GcpChain {
    fn source(&self, rt: &Runtime) -> anyhow::Result<Source> {
        let Some((named_by, path)) = self.credentials_path(rt) else {
            return Ok(Source::Metadata);
        };
        // A key file is a secret: held to the standard of the credential store.
        crate::config::owner_readable_only(&path).map_err(|e| {
            anyhow!(
                "refusing to read the Google credentials {} ({named_by}): {e}",
                path.display()
            )
        })?;
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "read the Google credentials {} ({named_by})",
                path.display()
            )
        })?;
        let json: Value = serde_json::from_str(&text).map_err(|_| {
            anyhow!(
                "{} ({named_by}) is not a JSON credentials file",
                path.display()
            )
        })?;
        let field = |k: &str| {
            json.get(k)
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        };
        match json.get("type").and_then(Value::as_str) {
            Some("service_account") => Ok(Source::ServiceAccount {
                email: field("client_email")
                    .ok_or_else(|| anyhow!("the service account key has no client_email"))?,
                private_key: field("private_key")
                    .ok_or_else(|| anyhow!("the service account key has no private_key"))?,
                key_id: field("private_key_id"),
                token_uri: field("token_uri"),
            }),
            Some("authorized_user") => Ok(Source::AuthorizedUser {
                client_id: field("client_id")
                    .ok_or_else(|| anyhow!("the authorized_user file has no client_id"))?,
                client_secret: field("client_secret")
                    .ok_or_else(|| anyhow!("the authorized_user file has no client_secret"))?,
                refresh_token: field("refresh_token").ok_or_else(|| {
                    anyhow!("the authorized_user file has no refresh_token; run `gcloud auth application-default login`")
                })?,
            }),
            Some("external_account") => bail!(
                "{} is an external_account (workload identity federation) file, which llmman \
                 does not support; use a service account key, `gcloud auth \
                 application-default login`, or the metadata server",
                path.display()
            ),
            other => bail!(
                "{} has the credential type {:?}, which is not service_account or authorized_user",
                path.display(),
                other.unwrap_or("(none)")
            ),
        }
    }

    /// What this chain would use, in words, or why it cannot: the file is
    /// read and its type checked, nothing is sent. For `llmman login`.
    pub fn describe_source(&self, rt: &Runtime) -> anyhow::Result<String> {
        Ok(match self.source(rt)? {
            Source::ServiceAccount { email, .. } => format!("the service account {email}"),
            Source::AuthorizedUser { .. } => "the signed-in gcloud user".to_string(),
            Source::Metadata => "the metadata server of this VM or pod".to_string(),
        })
    }

    /// A bearer credential, from the cache when it holds a token with life
    /// in it. `Ok(None)` when nothing hints at Google credentials
    /// ([`GcpChain::hinted`]). Blocking.
    pub fn resolve(&self, rt: &Runtime, http: &Http, now: u64) -> anyhow::Result<Option<Resolved>> {
        if !self.hinted(rt) {
            return Ok(None);
        }
        let source = self.source(rt)?;
        let minted = cache().get_or_mint(&source.cache_key(), now, EXPIRY_SKEW_MS, || {
            mint(rt, http, &source, now)
        })?;
        Ok(Some(bearer(minted.token, "adc")))
    }
}

fn mint(rt: &Runtime, http: &Http, source: &Source, now: u64) -> anyhow::Result<Minted> {
    let reply = match source {
        Source::ServiceAccount {
            email,
            key_id,
            private_key,
            token_uri,
        } => {
            let url = rt
                .token_url
                .clone()
                .or_else(|| token_uri.clone().filter(|u| token_uri_allowed(u)))
                .unwrap_or_else(|| TOKEN_URL.to_string());
            let key = crate::jwt::SigningKey::from_pem(private_key, Some(crate::jwt::Alg::Rs256))
                .context("the service account's private_key")?;
            let now_s = now / 1000;
            let assertion = key.sign_jwt(
                key_id.as_deref(),
                &json!({
                    "iss": email,
                    "scope": SCOPE,
                    "aud": url,
                    "iat": now_s,
                    "exp": now_s + ASSERTION_LIFETIME_SECS,
                }),
            )?;
            http.post_form(
                &url,
                &[
                    ("grant_type", crate::tokensvc::JWT_BEARER_GRANT),
                    ("assertion", &assertion),
                ],
                &[],
            )
        }
        Source::AuthorizedUser {
            client_id,
            client_secret,
            refresh_token,
        } => http.post_form(
            rt.token_url.as_deref().unwrap_or(TOKEN_URL),
            &[
                ("grant_type", "refresh_token"),
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("refresh_token", refresh_token),
            ],
            &[],
        ),
        Source::Metadata => {
            let meta = Http::metadata();
            let url = format!(
                "{}/computeMetadata/v1/instance/service-accounts/default/token",
                rt.metadata_origin()
            );
            meta.get(&url, &[("metadata-flavor", "Google")])
        }
    };
    let what = match source {
        Source::Metadata => "the Google metadata server",
        _ => "Google's token endpoint",
    };
    let tokens = token_reply(reply).map_err(|e| match e {
        crate::credentials::RefreshError::Rejected(why) => {
            anyhow!("{what} refused the request ({why})")
        }
        crate::credentials::RefreshError::Transient(why) => anyhow!("{what} failed: {why}"),
    })?;
    Ok(Minted {
        expires_ms: tokens.expires_ms(now),
        token: crate::credentials::Secret::new(tokens.access),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::now_ms;
    use crate::testserver::server;

    const RSA: &str = include_str!("testdata/jwt-test-rsa.pem");

    fn write_private(dir: &std::path::Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .unwrap();
        path
    }

    fn sa_json() -> String {
        json!({
            "type": "service_account",
            "project_id": "sa-project",
            "private_key_id": "kid-9",
            "private_key": RSA,
            "client_email": "llmman@sa-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        })
        .to_string()
    }

    #[test]
    fn endpoints_by_location() {
        assert_eq!(host("global"), "aiplatform.googleapis.com");
        assert_eq!(host("us"), "aiplatform.us.rep.googleapis.com");
        assert_eq!(host("eu"), "aiplatform.eu.rep.googleapis.com");
        assert_eq!(host("us-central1"), "us-central1-aiplatform.googleapis.com");
        assert_eq!(
            openapi_base("p-1", "global"),
            "https://aiplatform.googleapis.com/v1/projects/p-1/locations/global/endpoints/openapi"
        );
        assert_eq!(
            openapi_base("p-1", "europe-west4"),
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/p-1/locations/europe-west4/endpoints/openapi"
        );
        assert_eq!(wire_model("gemini-2.5-flash"), "google/gemini-2.5-flash");
        assert_eq!(
            wire_model("meta/llama-4-scout-17b-16e-instruct-maas"),
            "meta/llama-4-scout-17b-16e-instruct-maas"
        );
        assert!(project_ok("my-proj_1.x:y").is_ok());
        assert!(project_ok("a/b").is_err());
        assert!(project_ok("").is_err());
        assert!(location_ok("us-central1").is_ok());
        assert!(location_ok("a.evil.com").is_err());
    }

    #[test]
    fn project_and_location_follow_the_documented_order() {
        let dir = tempfile::tempdir().unwrap();
        let chain = GcpChain::default();
        let rt = |env: &[(&str, &str)]| Runtime::with(env, dir.path(), None, None);
        // Nothing anywhere.
        assert_eq!(chain.project(&rt(&[])), None);
        assert_eq!(chain.location(&rt(&[])), "global");
        // The aliases, in order.
        assert_eq!(
            chain
                .project(&rt(&[
                    ("GCLOUD_PROJECT", "c"),
                    ("GOOGLE_CLOUD_PROJECT", "b")
                ]))
                .as_deref(),
            Some("b")
        );
        assert_eq!(
            chain
                .project(&rt(&[
                    ("GOOGLE_VERTEX_PROJECT", "a"),
                    ("GOOGLE_CLOUD_PROJECT", "b")
                ]))
                .as_deref(),
            Some("a")
        );
        assert_eq!(
            chain.location(&rt(&[
                ("VERTEX_LOCATION", "eu"),
                ("GOOGLE_CLOUD_LOCATION", "us")
            ])),
            "us"
        );
        // A table in llmman.conf outranks the environment.
        let pinned = GcpChain {
            project: Some("conf".into()),
            location: Some("europe-west1".into()),
            ..GcpChain::default()
        };
        assert_eq!(
            pinned
                .project(&rt(&[("GOOGLE_VERTEX_PROJECT", "a")]))
                .as_deref(),
            Some("conf")
        );
        assert_eq!(
            pinned.location(&rt(&[("GOOGLE_VERTEX_LOCATION", "us")])),
            "europe-west1"
        );
        // A hostile value is no project.
        assert_eq!(
            chain.project(&rt(&[("GOOGLE_VERTEX_PROJECT", "a/../b")])),
            None
        );

        // gcloud's active configuration, in CLOUDSDK_CONFIG.
        let gcloud = dir.path().join("sdk");
        std::fs::create_dir_all(gcloud.join("configurations")).unwrap();
        std::fs::write(gcloud.join("active_config"), "work\n").unwrap();
        std::fs::write(
            gcloud.join("configurations").join("config_work"),
            "[core]\nproject = gcloud-project\n",
        )
        .unwrap();
        let with_sdk = rt(&[("CLOUDSDK_CONFIG", gcloud.to_str().unwrap())]);
        assert_eq!(chain.project(&with_sdk).as_deref(), Some("gcloud-project"));

        // The credentials file's own project_id, when nothing else says.
        let sa = write_private(dir.path(), "sa.json", &sa_json());
        let from_file = rt(&[("GOOGLE_APPLICATION_CREDENTIALS", sa.to_str().unwrap())]);
        assert_eq!(chain.project(&from_file).as_deref(), Some("sa-project"));
    }

    #[test]
    fn adc_files_are_found_where_gcloud_puts_them() {
        let dir = tempfile::tempdir().unwrap();
        let chain = GcpChain::default();
        let rt = |env: &[(&str, &str)]| Runtime::with(env, dir.path(), None, None);
        // The default: ~/.config/gcloud
        assert!(chain.credentials_path(&rt(&[])).is_none());
        let default = dir.path().join(".config").join("gcloud");
        std::fs::create_dir_all(&default).unwrap();
        std::fs::write(default.join("application_default_credentials.json"), "{}").unwrap();
        let (by, path) = chain.credentials_path(&rt(&[])).unwrap();
        assert_eq!(by, "gcloud");
        assert!(path.ends_with(".config/gcloud/application_default_credentials.json"));
        // CLOUDSDK_CONFIG replaces it.
        let sdk = dir.path().join("sdk");
        std::fs::create_dir_all(&sdk).unwrap();
        assert!(chain
            .credentials_path(&rt(&[("CLOUDSDK_CONFIG", sdk.to_str().unwrap())]))
            .is_none());
        std::fs::write(sdk.join("application_default_credentials.json"), "{}").unwrap();
        let (_, path) = chain
            .credentials_path(&rt(&[("CLOUDSDK_CONFIG", sdk.to_str().unwrap())]))
            .unwrap();
        assert!(path.starts_with(&sdk));
        // GOOGLE_APPLICATION_CREDENTIALS outranks both, the table outranks it.
        let (by, _) = chain
            .credentials_path(&rt(&[("GOOGLE_APPLICATION_CREDENTIALS", "/x/y.json")]))
            .unwrap();
        assert_eq!(by, "GOOGLE_APPLICATION_CREDENTIALS");
        let table = GcpChain {
            credentials_file: Some("/conf.json".into()),
            ..GcpChain::default()
        };
        assert_eq!(
            table
                .credentials_path(&rt(&[("GOOGLE_APPLICATION_CREDENTIALS", "/x/y.json")]))
                .unwrap()
                .0,
            "llmman.conf"
        );
    }

    #[test]
    fn hints_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let chain = GcpChain::default();
        let bare = Runtime::with(&[], dir.path(), None, None);
        if !on_gce() {
            assert!(!chain.hinted(&bare));
            assert_eq!(chain.status(&bare), None);
        }
        let host_set = Runtime::with(
            &[("GCE_METADATA_HOST", "169.254.169.254")],
            dir.path(),
            None,
            None,
        );
        assert!(chain.hinted(&host_set));
        let status = chain.status(&host_set).unwrap();
        assert_eq!((status.kind, status.state), ("adc", "ready"));
        let forced = GcpChain {
            forced: true,
            ..GcpChain::default()
        };
        assert!(forced.hinted(&bare));
    }

    /// A service account: a signed assertion goes to the token endpoint and
    /// the access token comes back, once, and is reused.
    #[test]
    fn a_service_account_signs_an_assertion_and_gets_a_token() {
        let server = server();
        server.fake.on(
            "/token",
            vec![(200, json!({"access_token": "ya29.sa-token", "expires_in": 3599, "token_type": "Bearer"}))],
        );
        let dir = tempfile::tempdir().unwrap();
        let sa = write_private(dir.path(), "sa.json", &sa_json());
        let url = format!("{}/token", server.origin);
        let rt = Runtime::with(
            &[("GOOGLE_APPLICATION_CREDENTIALS", sa.to_str().unwrap())],
            dir.path(),
            Some(&url),
            None,
        );
        let http = Http::tokens();
        let chain = GcpChain::default();
        let got = chain.resolve(&rt, &http, now_ms()).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "ya29.sa-token");
        assert_eq!(got.kind, "adc");
        assert_eq!(got.style, crate::credentials::Style::Bearer);
        let hits = server.fake.hits("/token");
        assert_eq!(hits.len(), 1);
        let form = hits[0].form();
        assert_eq!(
            form["grant_type"],
            "urn:ietf:params:oauth:grant-type:jwt-bearer"
        );
        // The assertion: RS256, this key id, this issuer and scope, our endpoint.
        use base64::Engine as _;
        let parts: Vec<&str> = form["assertion"].split('.').collect();
        let dec = |p: &str| -> Value {
            serde_json::from_slice(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(p)
                    .unwrap(),
            )
            .unwrap()
        };
        assert_eq!(
            dec(parts[0]),
            json!({"alg": "RS256", "typ": "JWT", "kid": "kid-9"})
        );
        let claims = dec(parts[1]);
        assert_eq!(claims["iss"], "llmman@sa-project.iam.gserviceaccount.com");
        assert_eq!(claims["scope"], SCOPE);
        assert_eq!(claims["aud"], url);
        assert!(claims["exp"].as_u64().unwrap() > claims["iat"].as_u64().unwrap());
        // Cached.
        let again = chain.resolve(&rt, &http, now_ms()).unwrap().unwrap();
        assert_eq!(again.secret.expose(), "ya29.sa-token");
        assert_eq!(server.fake.hits("/token").len(), 1);
        // And reported, without the token.
        let status = chain.status(&rt).unwrap();
        assert!(status.expires_at_ms.unwrap() > now_ms());
        assert!(!format!("{got:?}").contains("ya29"));
    }

    /// `gcloud auth application-default login`'s file: its refresh token is
    /// exchanged with the client id and secret in it.
    #[test]
    fn an_authorized_user_file_refreshes_through_the_token_endpoint() {
        let server = server();
        server.fake.on(
            "/token",
            vec![(
                200,
                json!({"access_token": "ya29.user-token", "expires_in": 3599}),
            )],
        );
        let dir = tempfile::tempdir().unwrap();
        let sdk = dir.path().join("sdk");
        std::fs::create_dir_all(&sdk).unwrap();
        write_private(
            &sdk,
            "application_default_credentials.json",
            &json!({
                "type": "authorized_user",
                "client_id": "cid.apps.googleusercontent.com",
                "client_secret": "csecret",
                "refresh_token": "1//refresh",
                "quota_project_id": "quota-proj",
            })
            .to_string(),
        );
        let url = format!("{}/token", server.origin);
        let rt = Runtime::with(
            &[("CLOUDSDK_CONFIG", sdk.to_str().unwrap())],
            dir.path(),
            Some(&url),
            None,
        );
        let chain = GcpChain::default();
        assert_eq!(chain.project(&rt).as_deref(), Some("quota-proj"));
        let got = chain
            .resolve(&rt, &Http::tokens(), now_ms())
            .unwrap()
            .unwrap();
        assert_eq!(got.secret.expose(), "ya29.user-token");
        let form = server.fake.hits("/token")[0].form();
        assert_eq!(form["grant_type"], "refresh_token");
        assert_eq!(form["client_id"], "cid.apps.googleusercontent.com");
        assert_eq!(form["client_secret"], "csecret");
        assert_eq!(form["refresh_token"], "1//refresh");
    }

    /// On a GCE VM or GKE pod: the metadata server, with its header.
    #[test]
    fn the_metadata_server_is_the_last_resort() {
        let server = server();
        server.fake.on(
            "/computeMetadata/v1/instance/service-accounts/default/token",
            vec![(200, json!({"access_token": "ya29.meta-token", "expires_in": 1800, "token_type": "Bearer"}))],
        );
        let dir = tempfile::tempdir().unwrap();
        let rt = Runtime::with(
            &[("GCE_METADATA_HOST", server.authority())],
            dir.path(),
            None,
            Some(&server.origin),
        );
        let got = GcpChain::default()
            .resolve(&rt, &Http::tokens(), now_ms())
            .unwrap()
            .unwrap();
        assert_eq!(got.secret.expose(), "ya29.meta-token");
        let hits = server
            .fake
            .hits("/computeMetadata/v1/instance/service-accounts/default/token");
        assert_eq!(hits[0].headers["metadata-flavor"], "Google");
        // Not hinted, not asked.
        let bare = Runtime::with(&[], dir.path(), None, Some(&server.origin));
        if !on_gce() {
            assert!(GcpChain::default()
                .resolve(&bare, &Http::tokens(), now_ms())
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn bad_credential_files_are_refused_without_echoing_their_contents() {
        let dir = tempfile::tempdir().unwrap();
        let http = Http::tokens();
        let try_file = |name: &str, text: &str| {
            let path = write_private(dir.path(), name, text);
            let rt = Runtime::with(
                &[("GOOGLE_APPLICATION_CREDENTIALS", path.to_str().unwrap())],
                dir.path(),
                Some("http://127.0.0.1:1/token"),
                None,
            );
            GcpChain::default()
                .resolve(&rt, &http, now_ms())
                .unwrap_err()
                .to_string()
        };
        let err = try_file(
            "a.json",
            r#"{"type":"external_account","audience":"//iam.googleapis.com/secret-audience"}"#,
        );
        assert!(
            err.contains("external_account") && err.contains("not support"),
            "{err}"
        );
        assert!(!err.contains("secret-audience"), "{err}");
        let err = try_file("b.json", "not json at all sk-leak");
        assert!(!err.contains("sk-leak"), "{err}");
        let err = try_file(
            "c.json",
            r#"{"type":"service_account","client_email":"a@b"}"#,
        );
        assert!(err.contains("private_key"), "{err}");
        let err = try_file(
            "d.json",
            r#"{"type":"service_account","client_email":"a@b","private_key":"not a pem key"}"#,
        );
        assert!(err.contains("private_key"), "{err}");
        assert!(!err.contains("not a pem key"), "{err}");
        let err = try_file(
            "e.json",
            r#"{"type":"authorized_user","client_id":"a","client_secret":"b"}"#,
        );
        assert!(err.contains("refresh_token"), "{err}");

        // A key file others can read is not read at all.
        #[cfg(unix)]
        {
            let path = dir.path().join("loose.json");
            std::fs::write(&path, sa_json()).unwrap();
            std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o644))
                .unwrap();
            let rt = Runtime::with(
                &[("GOOGLE_APPLICATION_CREDENTIALS", path.to_str().unwrap())],
                dir.path(),
                None,
                None,
            );
            let err = GcpChain::default()
                .resolve(&rt, &http, now_ms())
                .unwrap_err();
            assert!(err.to_string().contains("refusing"), "{err}");
        }
    }

    #[test]
    fn a_token_endpoint_that_refuses_is_reported_by_name_not_by_body() {
        let server = server();
        server.fake.on(
            "/token",
            vec![(400, json!({"error": "invalid_grant", "error_description": "Bad Request sk-body-secret"}))],
        );
        let dir = tempfile::tempdir().unwrap();
        let sdk = dir.path().join("sdk");
        std::fs::create_dir_all(&sdk).unwrap();
        write_private(
            &sdk,
            "application_default_credentials.json",
            r#"{"type":"authorized_user","client_id":"unique-refused-client","client_secret":"s","refresh_token":"r"}"#,
        );
        let url = format!("{}/token", server.origin);
        let rt = Runtime::with(
            &[("CLOUDSDK_CONFIG", sdk.to_str().unwrap())],
            dir.path(),
            Some(&url),
            None,
        );
        let err = GcpChain::default()
            .resolve(&rt, &Http::tokens(), now_ms())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("invalid_grant") && !err.contains("sk-body-secret"),
            "{err}"
        );
    }

    #[test]
    fn a_service_account_token_uri_must_be_googles() {
        assert!(token_uri_allowed("https://oauth2.googleapis.com/token"));
        assert!(token_uri_allowed("https://sts.googleapis.com/v1/token"));
        assert!(!token_uri_allowed("http://oauth2.googleapis.com/token"));
        assert!(!token_uri_allowed("https://evil.example.com/token"));
        assert!(!token_uri_allowed("https://googleapis.com.evil.com/token"));
        assert!(!token_uri_allowed(
            "https://u:p@oauth2.googleapis.com/token"
        ));
        assert!(!token_uri_allowed(
            "https://oauth2.googleapis.com:8443/token"
        ));
    }
}
