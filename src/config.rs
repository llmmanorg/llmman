//! `llmman.conf` — one config file, read once, shared by every consumer.
//!
//! Two locations, later overriding earlier:
//!
//!   1. `/etc/llmman/llmman.conf`        system-wide
//!   2. `~/.config/llmman/llmman.conf`   per-user
//!
//! The same paths on every platform, with no llmman-specific variable
//! to move them: one documented answer to "where does this go". (`~` is
//! `$HOME`, or `%USERPROFILE%` on Windows.)
//!
//! ```toml
//! [aliases]                            # crate::shortnames
//! gemma4 = "docker.io/ai/gemma4"
//!
//! [providers.openrouter]               # crate::providers
//! api_key = "sk-or-..."
//!
//! [providers.inferencebox]                   # a provider models.dev does not list
//! base_url = "http://inferencebox:8000/v1"
//! wire     = "openai"                  # default; or "anthropic"
//!
//! [verify]                             # crate::verify
//! default = "off"
//! layers  = "off"                      # hash stored layers on a cold load
//!
//! [[verify.trust]]
//! pattern = "docker.io/myorg/**"
//! keys    = ["keys/myorg.pub"]         # relative to this file's directory
//! mode    = "enforce"
//!
//! [[verify.trust]]
//! pattern = "mirror.corp/**"           # pulled reference selects the rule
//! claims  = "docker.io/myorg/**"       # repository the signature may name
//! keys    = ["keys/myorg.pub"]
//! mode    = "enforce"
//!
//! [aggregation]                        # cmd::serve::aggregation
//! peers = "asahi,spark:17434"
//! api_key = "..."                      # presented to peers; defaults to the first auth key
//!
//! [auth]                               # cmd::serve::auth
//! api_keys = "k1,k2"                   # what a request to this daemon must present
//!
//! [registries."docker.io"]             # crate::ffi, go-shim
//! mirrors = "https://mirror.gcr.io,registry-mirror.corp:5000"
//!
//! [launch.opencode]                    # cmd::launch: flags to use when none are typed
//! provider = "anthropic"
//! model    = "claude-sonnet-5-5"
//! variant  = "xhigh"
//! ```
//!
//! Parsing happens once, in [`files`]; what a *failed* parse means is
//! left to each consumer, because it differs. [`crate::verify`] refuses
//! to run, since a trust policy it cannot read must not quietly become
//! `off`; aliases and keys degrade to none. The error is reported once,
//! centrally, so one typo is not announced three times.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, PoisonError, RwLock};

use serde::Deserialize;

const FILE: &str = "llmman.conf";

/// Everything `llmman.conf` configures.
///
/// `deny_unknown_fields` throughout: a misspelling that parsed happily
/// would be a policy, alias or credential that silently never takes
/// effect, surfacing somewhere else entirely — an unsigned pull that
/// passes, a 401 inside someone else's TUI.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct Conf {
    #[serde(default)]
    pub core: CoreConf,
    #[serde(default)]
    pub pager: PagerConf,
    /// Short-name aliases — see [`crate::shortnames`].
    #[serde(default)]
    pub aliases: HashMap<String, String>,
    /// Private: a key leaves this module only through
    /// [`provider_api_key`], which gates it on the file's mode.
    #[serde(default)]
    providers: HashMap<String, ProviderConf>,
    /// Signature trust policy — see [`crate::verify`].
    #[serde(default)]
    pub verify: VerifyConf,
    /// Peer daemons — see `cmd::serve::aggregation`.
    #[serde(default)]
    pub aggregation: AggregationConf,
    /// Private, like `providers`: reached through [`auth_api_keys`] only.
    #[serde(default)]
    auth: AuthConf,
    /// Native OAuth forwarding on the daemon's TLS listener.
    #[serde(default)]
    pub managed: ManagedConf,
    /// `[registries."<host>"]` tables, keyed by the registry host a
    /// reference names — see [`registry_mirrors`].
    #[serde(default)]
    pub registries: HashMap<String, RegistryConf>,
    /// `[launch.<integration>]` tables, see [`launch_defaults`].
    #[serde(default)]
    pub launch: HashMap<String, LaunchConf>,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CoreConf {
    #[serde(default)]
    pub pager: Option<String>,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct PagerConf {
    #[serde(default)]
    pub log: Option<String>,
    #[serde(default)]
    pub providers: Option<String>,
    #[serde(default)]
    pub list: Option<String>,
    #[serde(default)]
    pub search: Option<String>,
    #[serde(default)]
    pub ps: Option<String>,
}

pub(crate) fn pager(command: &str) -> (Option<String>, Option<String>) {
    pager_from_files(files().unwrap_or_default(), command)
}

fn pager_from_files(files: &[File], command: &str) -> (Option<String>, Option<String>) {
    let mut core = None;
    let mut specific = None;
    for file in files {
        if let Some(value) = &file.conf.core.pager {
            core = Some(value.clone());
        }
        let value = match command {
            "log" => &file.conf.pager.log,
            "providers" => &file.conf.pager.providers,
            "list" => &file.conf.pager.list,
            "search" => &file.conf.pager.search,
            "ps" => &file.conf.pager.ps,
            _ => continue,
        };
        if let Some(value) = value {
            specific = Some(value.clone());
        }
    }
    (core, specific)
}

/// The `[aggregation]` section.
#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct AggregationConf {
    /// Comma-separated `[scheme://]host[:port]`, as `LLMMAN_PEERS` — a
    /// string so `llmman config set aggregation.peers a,b` can write it.
    #[serde(default)]
    pub peers: Option<String>,
    /// Presented to peers, as `LLMMAN_PEER_API_KEY`; see [`peer_api_key`].
    #[serde(default)]
    api_key: Option<String>,
}

impl std::fmt::Debug for AggregationConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AggregationConf")
            .field("peers", &self.peers)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// The `[auth]` section — see `cmd::serve::auth`.
#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
struct AuthConf {
    /// Comma-separated keys a request must present, as `LLMMAN_API_KEYS`
    /// — a string so `llmman config set auth.api_keys a,b` can write it.
    #[serde(default)]
    api_keys: Option<String>,
}

/// The `[managed]` section. Strings work with `llmman config set`.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ManagedConf {
    /// Explicit opt-in; absent means disabled.
    #[serde(default)]
    pub enabled: Option<String>,
    /// Optional upstream idle-read limit; absent or zero means no limit.
    #[serde(default)]
    pub read_timeout_seconds: Option<String>,
}

/// Validated forwarding settings. No provider credentials are stored here.
#[derive(Default, Debug, Clone)]
pub struct Managed {
    /// Whether the daemon mounts the forwarding routes.
    pub enabled: bool,
    /// Maximum wait for each upstream read; no total request deadline.
    pub read_timeout: Option<std::time::Duration>,
}

/// Loads managed settings using the normal configuration precedence.
pub fn managed() -> Result<Managed, String> {
    managed_from_files(files()?)
}

fn managed_from_files(files: &[File]) -> Result<Managed, String> {
    let mut managed = Managed::default();
    for file in files {
        managed.apply(&file.conf.managed)?;
    }
    Ok(managed)
}

impl Managed {
    /// Shared by parse-time validation and field-by-field configuration merging.
    fn apply(&mut self, conf: &ManagedConf) -> Result<(), String> {
        if let Some(enabled) = &conf.enabled {
            self.enabled = enabled
                .parse()
                .map_err(|_| "managed.enabled must be \"true\" or \"false\"".to_string())?;
        }
        if let Some(seconds) = &conf.read_timeout_seconds {
            let seconds: u32 = seconds.parse().map_err(|_| {
                "managed.read_timeout_seconds must be an unsigned 32-bit integer".to_string()
            })?;
            self.read_timeout =
                (seconds != 0).then(|| std::time::Duration::from_secs(seconds.into()));
        }
        Ok(())
    }
}

impl std::fmt::Debug for AuthConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConf")
            .field("api_keys", &self.api_keys.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// One `[registries."<host>"]` table.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RegistryConf {
    /// Comma-separated `[scheme://]host[:port][/path]` mirrors, tried in
    /// order before the registry for pulls. A string, like `peers`, so
    /// `llmman config set` can write it.
    #[serde(default)]
    pub mirrors: Option<String>,
}

/// One `[launch.<integration>]` table: the flags `llmman launch` uses when
/// none are typed. A blank value clears one that an earlier file set.
#[derive(Deserialize, Default, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaunchConf {
    /// As `--provider`.
    #[serde(default)]
    pub provider: Option<String>,
    /// As `--model`.
    #[serde(default)]
    pub model: Option<String>,
    /// As `--variant`.
    #[serde(default)]
    pub variant: Option<String>,
}

/// One `[providers.<id>]` table. Without `base_url` it keys a catalog
/// provider; with it, it *defines* one — an inference server at a host
/// models.dev does not list — and shadows any catalog entry with its id.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderConf {
    #[serde(default)]
    api_key: Option<String>,
    /// Base URL the wire's route is appended to. Plain `http://` is
    /// allowed: the user typed it into their own owner-only file.
    #[serde(default)]
    base_url: Option<String>,
    /// What is spoken at `base_url`; `openai` by default.
    #[serde(default)]
    wire: Option<crate::providers::Wire>,
    /// Variable holding the key. With neither this nor `api_key`, no
    /// credential is sent — most local servers take none.
    #[serde(default)]
    api_key_env: Option<String>,
    /// Display name for listings; the id when absent.
    #[serde(default)]
    name: Option<String>,
    /// How the credential is attached: `bearer`, `x-api-key`, `header` or
    /// `query`; the wire's own when absent. Applies to a catalog provider
    /// too. See [`crate::authscheme`].
    #[serde(default)]
    auth: Option<String>,
    /// The header's name, for `auth = "header"`.
    #[serde(default)]
    auth_header: Option<String>,
    /// The query parameter's name, for `auth = "query"`.
    #[serde(default)]
    auth_query: Option<String>,
    /// Extra headers sent on every request to this provider.
    #[serde(default)]
    headers: Option<BTreeMap<String, String>>,
    /// Extra query parameters sent on every request.
    #[serde(default)]
    query: Option<BTreeMap<String, String>>,
    /// Values for `${VAR}` in a catalog provider's base URL.
    #[serde(default)]
    vars: Option<BTreeMap<String, String>>,
    /// A shell command whose output is the key. Run when a request needs
    /// it; the key is never copied here.
    #[serde(default)]
    api_key_command: Option<String>,
    /// A file holding the key, which must be owner-only like this one.
    #[serde(default)]
    api_key_file: Option<String>,
    /// What stands behind this provider when it has no key: AWS SigV4
    /// (`aws`), Google ADC (`gcp`), Entra ID (`azure`), an OAuth 2.0 token
    /// service (`oauth2`, `jwt_bearer`, `ibm_iam`). One at most. See
    /// [`crate::chains`].
    #[serde(default)]
    aws: Option<crate::aws::AwsChain>,
    #[serde(default)]
    gcp: Option<crate::gcp::GcpChain>,
    #[serde(default)]
    azure: Option<crate::azure::AzureChain>,
    #[serde(default)]
    oauth2: Option<crate::tokensvc::TokenService>,
    #[serde(default)]
    jwt_bearer: Option<crate::tokensvc::TokenService>,
    #[serde(default)]
    ibm_iam: Option<crate::tokensvc::TokenService>,
}

/// Hand-written so the key cannot reach a log through the derived
/// `Debug` of the public [`Conf`] and [`File`]. Same reasoning as
/// `RemoteTarget` in `cmd::serve`.
impl std::fmt::Debug for ProviderConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConf")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("wire", &self.wire)
            .field("api_key_env", &self.api_key_env)
            .field("name", &self.name)
            .field("auth", &self.auth)
            .field("auth_header", &self.auth_header)
            .field("auth_query", &self.auth_query)
            .field(
                "headers",
                &self.headers.as_ref().map(|h| h.keys().collect::<Vec<_>>()),
            )
            .field(
                "query",
                &self.query.as_ref().map(|q| q.keys().collect::<Vec<_>>()),
            )
            .field("vars", &self.vars)
            .field("api_key_command", &self.api_key_command)
            .field("api_key_file", &self.api_key_file)
            .field("chain", &self.chain().ok().flatten().map(|c| c.kind()))
            .finish()
    }
}

impl ProviderConf {
    /// The credential chain this table configures, or an error naming the
    /// table. At most one of `aws`, `gcp`, `azure`, `oauth2`, `jwt_bearer`,
    /// `ibm_iam`: they are alternatives. A table written in `llmman.conf`
    /// marks its chain `forced`, which says "try it" without a hint.
    fn chain(&self) -> Result<Option<crate::chains::Chain>, String> {
        use crate::chains::Chain;
        use crate::tokensvc::Grant;
        let token = |t: &crate::tokensvc::TokenService, table: &str, grant: Grant| {
            if t.grant != Grant::default() && t.grant != grant {
                return Err(format!("{table}: grant cannot be set here"));
            }
            let mut t = t.clone();
            t.grant = grant;
            t.validate().map_err(|e| format!("{table}: {e}"))?;
            Ok(Chain::Token(t))
        };
        let mut found: Vec<(&str, Chain)> = Vec::new();
        if let Some(a) = &self.aws {
            a.validate().map_err(|e| format!("aws: {e}"))?;
            found.push((
                "aws",
                Chain::Aws(crate::aws::AwsChain {
                    forced: true,
                    ..a.clone()
                }),
            ));
        }
        if let Some(g) = &self.gcp {
            g.validate().map_err(|e| format!("gcp: {e}"))?;
            found.push((
                "gcp",
                Chain::Gcp(crate::gcp::GcpChain {
                    forced: true,
                    ..g.clone()
                }),
            ));
        }
        if let Some(a) = &self.azure {
            a.validate().map_err(|e| format!("azure: {e}"))?;
            found.push((
                "azure",
                Chain::Azure(crate::azure::AzureChain {
                    forced: true,
                    ..a.clone()
                }),
            ));
        }
        if let Some(t) = &self.oauth2 {
            found.push(("oauth2", token(t, "oauth2", Grant::ClientCredentials)?));
        }
        if let Some(t) = &self.jwt_bearer {
            found.push(("jwt_bearer", token(t, "jwt_bearer", Grant::JwtBearer)?));
        }
        if let Some(t) = &self.ibm_iam {
            found.push(("ibm_iam", token(t, "ibm_iam", Grant::IbmApikey)?));
        }
        if found.len() > 1 {
            return Err(format!(
                "{} are alternatives; set one",
                found.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
            ));
        }
        Ok(found.pop().map(|(_, c)| c))
    }

    /// The auth scheme, extra headers and query this table configures, or
    /// an error naming the field. Used by [`validate`] and
    /// [`provider_auth`].
    fn auth(&self) -> Result<crate::authscheme::Auth, String> {
        let scheme = crate::authscheme::Scheme::from_fields(
            self.auth.as_deref(),
            self.auth_header.as_deref(),
            self.auth_query.as_deref(),
        )?;
        let mut headers = Vec::new();
        for (name, value) in self.headers.iter().flatten() {
            crate::authscheme::header_name(name).map_err(|e| format!("headers: {e}"))?;
            crate::authscheme::header_value(name, value).map_err(|e| format!("headers: {e}"))?;
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
        let mut query = Vec::new();
        for (name, value) in self.query.iter().flatten() {
            crate::authscheme::query_param(name)?;
            query.push((name.clone(), value.clone()));
        }
        let mut vars = BTreeMap::new();
        for (name, value) in self.vars.iter().flatten() {
            if name.is_empty()
                || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                || name.starts_with(|c: char| c.is_ascii_digit())
            {
                return Err(format!(
                    "vars: {name:?} is not a variable name (letters, digits, `_`)"
                ));
            }
            vars.insert(name.clone(), value.trim().to_string());
        }
        Ok(crate::authscheme::Auth {
            scheme,
            headers,
            query,
            vars,
        })
    }

    /// `api_key_command` / `api_key_file`, trimmed; blank is `Some("")`
    /// so a later file can clear one an earlier file set.
    fn external(&self) -> Result<Option<crate::credentials::ExternalSource>, String> {
        let command = self.api_key_command.as_deref().map(str::trim);
        let file = self.api_key_file.as_deref().map(str::trim);
        match (command, file) {
            (Some(c), Some(f)) if !c.is_empty() && !f.is_empty() => {
                Err("api_key_command and api_key_file are alternatives; set one".to_string())
            }
            (Some(c), _) if !c.is_empty() => {
                Ok(Some(crate::credentials::ExternalSource::Command {
                    command: c.to_string(),
                }))
            }
            (_, Some(f)) if !f.is_empty() => Ok(Some(crate::credentials::ExternalSource::File {
                path: f.to_string(),
            })),
            _ => Ok(None),
        }
    }

    /// Whether this table defines a provider rather than only keying one.
    fn defines(&self) -> bool {
        self.base_url.is_some()
    }

    /// Fields that only mean something on a definition (see [`validate`]).
    fn definition_only_fields(&self) -> Vec<&'static str> {
        let mut set = Vec::new();
        if self.wire.is_some() {
            set.push("wire");
        }
        if self.api_key_env.is_some() {
            set.push("api_key_env");
        }
        if self.name.is_some() {
            set.push("name");
        }
        set
    }
}

/// A provider `llmman.conf` defines, merged across every file that
/// mentions it. What `crate::providers` turns into a `Provider`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfiguredProvider {
    pub id: String,
    /// `name` from the file, else the id.
    pub name: String,
    /// Without its trailing slash, as `Provider::base_url` is kept.
    pub base_url: String,
    pub wire: crate::providers::Wire,
    /// `api_key_env`, when set. The `api_key` itself is not here: it is
    /// reached through [`provider_api_key`], behind the file-mode gate.
    pub key_env: Option<String>,
}

/// The `[verify]` section. [`crate::verify`] owns what these strings
/// mean; this module only spells out their shape.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct VerifyConf {
    /// Mode for references no rule matches.
    #[serde(default)]
    pub default: Option<String>,
    /// Mode for hashing the stored manifest and layer blobs on a cold
    /// load. Independent of `default` and the trust rules: those speak
    /// about signatures, this about the bytes already on disk.
    #[serde(default)]
    pub layers: Option<String>,
    /// `[[verify.trust]]` entries, in the order they appear.
    #[serde(default)]
    pub trust: Vec<TrustConf>,
    /// Sigstore TrustedRoot JSON (Fulcio roots and Rekor log keys) that
    /// keyless rules, those with `issuer` or `subject`, are checked
    /// against. Relative to this file's directory, like `keys`.
    #[serde(default)]
    pub trusted_root: Option<String>,
}

/// One `[[verify.trust]]` entry.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct TrustConf {
    pub pattern: String,
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub mode: Option<String>,
    /// Glob the signature's claimed repository must match, in place of
    /// equality with the pulled one. Same grammar as `pattern`.
    #[serde(default)]
    pub claims: Option<String>,
    /// Keyless rule: the OIDC issuer the signing certificate must name.
    /// Exclusive with `keys`.
    #[serde(default)]
    pub issuer: Option<String>,
    /// Keyless rule: the identity (a certificate SAN) that must have
    /// signed. Exclusive with `keys`.
    #[serde(default)]
    pub subject: Option<String>,
}

impl Conf {
    /// Every entry that sets `api_key` at all, trimmed, by provider id.
    ///
    /// A blank value is kept here rather than dropped, so a user file can
    /// blank out a key `/etc` set — otherwise there is no way to opt out
    /// of a system-wide credential. [`provider_keys_in`] drops blanks
    /// after merging, since sending `Authorization: Bearer ` upstream
    /// would turn a clear "no API key" into someone else's 401.
    pub(crate) fn provider_keys(&self) -> HashMap<String, String> {
        self.providers
            .iter()
            .filter_map(|(id, p)| Some((id.clone(), p.api_key.as_deref()?.trim().to_string())))
            .collect()
    }
}

/// One `llmman.conf` that exists on disk.
#[derive(Debug)]
pub struct File {
    /// Where it was read from.
    pub path: PathBuf,
    /// Its own directory — what relative paths inside it resolve
    /// against, so a trust policy can ship alongside its keys.
    pub dir: PathBuf,
    pub conf: Conf,
}

// ---------------------------------------------------------------------------
// Search paths
// ---------------------------------------------------------------------------

/// Every place [`FILE`] may live, in ascending priority order.
///
/// Not the `/usr/share/llmman/`, `<binary>/../share/llmman/` and
/// `<binary-dir>/` tiers earlier versions searched: nothing ever shipped
/// a file to them, and they are package-managed and world-readable,
/// which `llmman.conf` cannot be.
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = vec![system_dir().join(FILE)];
    paths.extend(user_dir().map(|d| d.join(FILE)));
    paths
}

/// `/etc/llmman`.
#[cfg(not(windows))]
fn system_dir() -> PathBuf {
    PathBuf::from("/etc/llmman")
}

/// `/etc/llmman` anchored to the system drive.
///
/// A leading `/` is drive-*relative* on Windows, so a bare `/etc/llmman`
/// would mean `D:\etc\llmman` for a process launched from `D:` — the
/// system-wide location would move with the working directory.
#[cfg(windows)]
fn system_dir() -> PathBuf {
    let drive = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".to_string());
    PathBuf::from(format!("{drive}\\etc\\llmman"))
}

/// `~/.config/llmman`, on every platform. No llmman-specific override;
/// `~` is the home directory below.
fn user_dir() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".config").join("llmman"))
}

/// The home directory `~` names, here and for the integrations
/// `cmd::launch` configures. The environment comes first because
/// `dirs::home_dir` reads the Windows known-folder API, which ignores
/// the home a session set for every other program in it — node's
/// `os.homedir()` included, which is what Cline and pi resolve through.
pub(crate) fn home_dir() -> Option<PathBuf> {
    env_home().or_else(dirs::home_dir)
}

/// The variables that name the home directory. `%USERPROFILE%` is what
/// a Windows session sets and what node's `os.homedir()` reads; `HOME`
/// is not consulted there, because a POSIX shell exports it in a form
/// Windows reads as drive-relative.
const HOME_KEYS: &[&str] = if cfg!(windows) {
    &["USERPROFILE"]
} else {
    &["HOME"]
};

/// The first of [`HOME_KEYS`] that names an absolute path. A relative
/// or empty one would anchor the config on the current drive rather
/// than at a home directory.
fn env_home() -> Option<PathBuf> {
    env_home_from(HOME_KEYS, |key| std::env::var_os(key))
}

/// Split from [`env_home`] so the rule can be asserted without changing
/// the process environment.
fn env_home_from(
    keys: &[&str],
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    keys.iter()
        .filter_map(|key| lookup(key))
        .map(PathBuf::from)
        .find(|value| value.is_absolute())
}

/// Where a user's own config belongs, for a message that has to name
/// it or a [`crate::cmd::config`] that has to write it. `None` only when
/// there is no home directory.
pub fn user_path() -> Option<PathBuf> {
    user_dir().map(|d| d.join(FILE))
}

/// Where the system-wide config belongs. Always known, unlike
/// [`user_path`]: it does not depend on there being a home directory.
pub fn system_path() -> PathBuf {
    system_dir().join(FILE)
}

/// `key` as one part of a dotted TOML key, quoted unless it is bare: an
/// id like `wafer.ai` would otherwise name two nested tables.
pub fn toml_key(key: &str) -> String {
    toml_edit::Key::new(key).display_repr().into_owned()
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Every `llmman.conf` that exists, lowest priority first, parsed once
/// for the process.
///
/// `Err` when one is there but unreadable or malformed — see the module
/// docs for why what to do about that is the caller's call.
pub fn files() -> Result<&'static [File], &'static str> {
    cache().as_deref().map_err(String::as_str)
}

fn cache() -> &'static Result<Vec<File>, String> {
    static CACHE: OnceLock<Result<Vec<File>, String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let loaded = load();
        // Once, here, rather than by each of the three consumers.
        if let Err(e) = &loaded {
            eprintln!("[llmman] warning: {e}");
        }
        loaded
    })
}

fn load() -> Result<Vec<File>, String> {
    // No unit test may depend on whoever runs it having an llmman.conf:
    // their aliases would redirect a fixture reference, and a provider
    // called `openai` would resolve their real key, flipping the
    // assertions in `cmd::serve` and `cmd::providers` that prove no key
    // leaks. `parse`, `provider_keys` and `Policy::parse` are covered
    // directly instead. The binary the e2e tests spawn is built without
    // `cfg(test)`, so it reads the real files.
    if cfg!(test) {
        return Ok(Vec::new());
    }
    load_from(search_paths())
}

/// The files among `paths` that exist, parsed. Split out from [`load`]
/// so a test can point it at a temporary directory.
fn load_from(paths: Vec<PathBuf>) -> Result<Vec<File>, String> {
    let mut files = Vec::new();
    for path in paths {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            // Absence is the common case: most machines never have one.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("ignoring {}: {e}", path.display())),
        };
        let conf = parse(&text).map_err(|e| format!("ignoring {}: {e}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        files.push(File { path, dir, conf });
    }
    Ok(files)
}

/// Parse one `llmman.conf`. Split out from [`load`] so the format is
/// testable without a file, a home directory, or a particular umask.
pub(crate) fn parse(text: &str) -> Result<Conf, String> {
    let conf: Conf = toml::from_str(text).map_err(|e| e.to_string())?;
    validate(&conf)?;
    Ok(conf)
}

/// What the TOML shape alone cannot say, checked at parse time so
/// `llmman config set` refuses it on the spot.
fn validate(conf: &Conf) -> Result<(), String> {
    Managed::default().apply(&conf.managed)?;
    let mut seen: HashMap<String, &str> = HashMap::new();
    for (host, r) in &conf.registries {
        let canonical = canonical_registry_host(host).ok_or_else(|| {
            format!(
                "[registries.{host:?}]: expected a host[:port], as in [registries.\"docker.io\"]"
            )
        })?;
        if let Some(other) = seen.insert(canonical, host) {
            return Err(format!(
                "[registries.{host:?}] and [registries.{other:?}] name the same registry"
            ));
        }
        for mirror in split_list(r.mirrors.as_deref()) {
            mirror_url_of(&mirror).map_err(|e| format!("[registries.{host:?}] mirrors: {e}"))?;
        }
    }
    validate_launch(&conf.launch)?;
    for (id, p) in &conf.providers {
        // `<provider>/<model>` is how a reference travels (see
        // `crate::providers::split_remote_ref`), so a slash in the id
        // could never be routed.
        if id.is_empty() || id.contains('/') {
            return Err(format!(
                "[providers.{id:?}]: a provider id cannot be empty or contain '/'"
            ));
        }
        p.auth().map_err(|e| format!("[providers.{id}] {e}"))?;
        p.external().map_err(|e| format!("[providers.{id}] {e}"))?;
        p.chain().map_err(|e| format!("[providers.{id}] {e}"))?;
        match &p.base_url {
            Some(url) => {
                base_url_of(url).map_err(|e| format!("[providers.{id}] base_url: {e}"))?;
            }
            None => {
                let only_on_definition = p.definition_only_fields();
                if !only_on_definition.is_empty() {
                    return Err(format!(
                        "[providers.{id}] sets {} but no base_url, so it only keys the catalog \
                         provider {id:?}",
                        only_on_definition.join(", ")
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Each table must name an integration (once) and hold a variant that
/// `--variant` takes; a misspelling would silently never apply.
fn validate_launch(launch: &HashMap<String, LaunchConf>) -> Result<(), String> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    for (name, l) in launch {
        let canonical = crate::cmd::launch::canonical_integration(name).ok_or_else(|| {
            format!(
                "[launch.{name:?}]: not an integration `llmman launch` knows; \
                 run `llmman launch` to list them"
            )
        })?;
        // `copilot` and its alias `copilot-cli` are one integration.
        if let Some(other) = seen.insert(canonical, name) {
            return Err(format!(
                "[launch.{name:?}] and [launch.{other:?}] name the same integration"
            ));
        }
        if let Some(variant) = set_value(l.variant.as_deref()) {
            if !crate::cmd::run::variants().any(|v| v == variant) {
                return Err(format!(
                    "[launch.{name}] variant: {variant:?} is not one of {}",
                    crate::cmd::run::variants().collect::<Vec<_>>().join(", ")
                ));
            }
        }
    }
    Ok(())
}

/// `value` trimmed, `None` when absent or blank.
fn set_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

/// Parses an `http`/`https` URL with a host and no userinfo, query or
/// fragment. Userinfo is refused because these URLs are reported and
/// logged; `credential` names where a credential goes instead.
fn plain_http_url(url: &str, credential: &str) -> Result<reqwest::Url, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("{url:?}: {e}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "{url:?}: scheme must be http or https, not {other}"
            ))
        }
    }
    if parsed.host_str().is_none() {
        return Err(format!("{url:?}: no host"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!(
            "{url:?}: cannot carry a username or password; use {credential}"
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(format!("{url:?}: no query or fragment allowed"));
    }
    Ok(parsed)
}

/// Normalizes a configured `base_url`: lowercased scheme and host, no
/// trailing slash.
fn base_url_of(url: &str) -> Result<String, String> {
    let parsed = plain_http_url(url.trim(), "api_key")?;
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// Normalizes a mirror, `[scheme://]host[:port][/path]`, `https` when no
/// scheme is given. A path is kept (a mirror served under a prefix);
/// `llmman login <mirror-host>` is where a mirror's credential goes.
fn mirror_url_of(mirror: &str) -> Result<String, String> {
    let mirror = mirror.trim();
    if mirror.is_empty() {
        return Err("an empty entry".to_string());
    }
    let with_scheme = if mirror.contains("://") {
        mirror.to_string()
    } else {
        format!("https://{mirror}")
    };
    let parsed = plain_http_url(&with_scheme, "`llmman login`")?;
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// The `host[:port]` a `[registries."<host>"]` key names, as the Go shim
/// looks it up: lowercased, Docker Hub's several names folded onto
/// `docker.io`. `None` for anything that is not a bare authority.
fn canonical_registry_host(host: &str) -> Option<String> {
    let host = host.trim();
    // `*` is a wildcard in registries.conf, which the url crate lets through.
    if host.is_empty() || host.contains("://") || host.contains(['/', '*']) {
        return None;
    }
    let parsed = plain_http_url(&format!("https://{host}"), "").ok()?;
    if parsed.path() != "/" {
        return None;
    }
    let mut canonical = parsed.host_str()?.to_string();
    if let Some(port) = parsed.port() {
        canonical = format!("{canonical}:{port}");
    }
    Some(match canonical.as_str() {
        "index.docker.io" | "registry-1.docker.io" => "docker.io".to_string(),
        _ => canonical,
    })
}

/// Splits a comma-separated setting into its trimmed, non-empty entries.
fn split_list(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Registry mirrors
// ---------------------------------------------------------------------------

/// Mirrors per registry host, normalized and in order: for Docker Hub
/// `LLMMAN_REGISTRY_MIRRORS` when set (even empty; Hub-only, like
/// dockerd's `--registry-mirror`), else the last file that sets the
/// host's `mirrors`. Replaced per host, not merged, so `mirrors = ""`
/// opts a host out. What a mirror does is the Go shim's business (see
/// go-shim/registry_mirrors.go).
pub fn registry_mirrors() -> BTreeMap<String, Vec<String>> {
    registry_mirrors_from(
        std::env::var("LLMMAN_REGISTRY_MIRRORS").ok().as_deref(),
        files().unwrap_or_default(),
    )
}

/// Whether `host`, as a reference spells it, has mirrors configured.
/// Cached for the process; `crate::hf` asks per reference classified.
pub fn has_registry_mirrors(host: &str) -> bool {
    static CACHE: OnceLock<BTreeMap<String, Vec<String>>> = OnceLock::new();
    canonical_registry_host(host)
        .is_some_and(|host| CACHE.get_or_init(registry_mirrors).contains_key(&host))
}

fn registry_mirrors_from(env: Option<&str>, files: &[File]) -> BTreeMap<String, Vec<String>> {
    let mut by_host: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        for (host, r) in &file.conf.registries {
            let Some(mirrors) = r.mirrors.as_deref() else {
                continue;
            };
            let host = canonical_registry_host(host).expect("validated at parse");
            let mirrors = split_list(Some(mirrors))
                .iter()
                .map(|m| mirror_url_of(m).expect("validated at parse"))
                .collect();
            by_host.insert(host, mirrors);
        }
    }
    if let Some(env) = env {
        let mut mirrors = Vec::new();
        for m in split_list(Some(env)) {
            match mirror_url_of(&m) {
                Ok(m) => mirrors.push(m),
                Err(e) => eprintln!("[llmman] warning: ignoring LLMMAN_REGISTRY_MIRRORS entry {e}"),
            }
        }
        by_host.insert("docker.io".to_string(), mirrors);
    }
    by_host.retain(|_, mirrors| !mirrors.is_empty());
    by_host
}

// ---------------------------------------------------------------------------
// Launch defaults
// ---------------------------------------------------------------------------

/// The `[launch.<integration>]` defaults for `integration` (any case, or
/// an alias), merged one key at a time with later files overriding.
///
/// `Err` when a file does not parse: the defaults cannot be known, and
/// running without them would silently launch another model.
pub fn launch_defaults(integration: &str) -> Result<LaunchConf, &'static str> {
    Ok(launch_defaults_from(files()?, integration))
}

fn launch_defaults_from(files: &[File], integration: &str) -> LaunchConf {
    let wanted = crate::cmd::launch::canonical_integration(integration);
    let mut merged = LaunchConf::default();
    for conf in files.iter().filter_map(|file| {
        // At most one table per integration: see `validate_launch`.
        file.conf
            .launch
            .iter()
            .find(|(name, _)| crate::cmd::launch::canonical_integration(name) == wanted)
            .map(|(_, conf)| conf)
    }) {
        for (field, value) in [
            (&mut merged.provider, &conf.provider),
            (&mut merged.model, &conf.model),
            (&mut merged.variant, &conf.variant),
        ] {
            if value.is_some() {
                *field = set_value(value.as_deref()).map(str::to_string);
            }
        }
    }
    merged
}

#[cfg(test)]
pub(crate) fn launch_conf(
    provider: Option<&str>,
    model: Option<&str>,
    variant: Option<&str>,
) -> LaunchConf {
    let own = |v: Option<&str>| v.map(str::to_string);
    LaunchConf {
        provider: own(provider),
        model: own(model),
        variant: own(variant),
    }
}

// ---------------------------------------------------------------------------
// Configured providers
// ---------------------------------------------------------------------------

/// Every provider `llmman.conf` defines, sorted by id. Read once for the
/// process, like the keys.
pub fn configured_providers() -> &'static [ConfiguredProvider] {
    static CACHE: OnceLock<Vec<ConfiguredProvider>> = OnceLock::new();
    CACHE.get_or_init(|| configured_from(files().unwrap_or_default()))
}

/// Merged field by field, later files overriding earlier, as keys do.
/// Split out to be testable without a filesystem.
fn configured_from(files: &[File]) -> Vec<ConfiguredProvider> {
    let mut merged: HashMap<&str, ConfiguredProvider> = HashMap::new();
    for file in files {
        for (id, p) in file.conf.providers.iter().filter(|(_, p)| p.defines()) {
            let base_url = p
                .base_url
                .as_deref()
                .and_then(|u| base_url_of(u).ok())
                .expect("validated at parse");
            let entry = merged.entry(id).or_insert_with(|| ConfiguredProvider {
                id: id.clone(),
                name: id.clone(),
                base_url: String::new(),
                wire: crate::providers::Wire::OpenAi,
                key_env: None,
            });
            entry.base_url = base_url;
            if let Some(wire) = p.wire {
                entry.wire = wire;
            }
            // Blank clears a variable an earlier file named, as `api_key = ""`
            // clears a key.
            if let Some(var) = p.api_key_env.as_deref().map(str::trim) {
                entry.key_env = (!var.is_empty()).then(|| var.to_string());
            }
            if let Some(name) = p.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
                entry.name = name.to_string();
            }
        }
    }
    let mut out: Vec<ConfiguredProvider> = merged.into_values().collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

// ---------------------------------------------------------------------------
// Provider auth: scheme, extra headers, variables, key commands
// ---------------------------------------------------------------------------

/// What `[providers.<id>]` says about how to authenticate, merged across
/// files field by field, later overriding earlier: the scheme if the later
/// file names one, headers, query parameters and variables key by key.
/// The default [`crate::authscheme::Auth`] for a provider no file
/// mentions. Read once for the process, like [`configured_providers`].
pub fn provider_auth(provider_id: &str) -> crate::authscheme::Auth {
    static CACHE: OnceLock<HashMap<String, crate::authscheme::Auth>> = OnceLock::new();
    CACHE
        .get_or_init(|| auth_from(files().unwrap_or_default()))
        .get(provider_id)
        .cloned()
        .unwrap_or_default()
}

fn auth_from(files: &[File]) -> HashMap<String, crate::authscheme::Auth> {
    let mut merged: HashMap<String, crate::authscheme::Auth> = HashMap::new();
    for file in files {
        let mut loose = None;
        for (id, p) in &file.conf.providers {
            let Ok(mut auth) = p.auth() else { continue };
            // A static header or query value may be a gateway token: held
            // to the standard of an `api_key`.
            if !(auth.headers.is_empty() && auth.query.is_empty()) {
                let gate = loose.get_or_insert_with(|| owner_readable_only(&file.path));
                if let Err(e) = gate {
                    eprintln!(
                        "[llmman] warning: ignoring the headers and query of provider {id} in {}: {e}",
                        file.path.display()
                    );
                    auth.headers.clear();
                    auth.query.clear();
                }
            }
            let entry = merged.entry(id.clone()).or_default();
            if p.auth.is_some() || p.auth_header.is_some() || p.auth_query.is_some() {
                entry.scheme = auth.scheme;
            }
            for (k, v) in auth.headers {
                entry
                    .headers
                    .retain(|(name, _)| !name.eq_ignore_ascii_case(&k));
                entry.headers.push((k, v));
            }
            for (k, v) in auth.query {
                entry.query.retain(|(name, _)| *name != k);
                entry.query.push((k, v));
            }
            entry.vars.extend(auth.vars);
        }
    }
    merged
}

/// The value `[providers.<id>] vars` gives `name`. Blank counts as unset.
pub fn provider_var(provider_id: &str, name: &str) -> Option<String> {
    provider_auth(provider_id)
        .vars
        .get(name)
        .filter(|v| !v.is_empty())
        .cloned()
}

/// The `api_key_command` / `api_key_file` that applies to `provider_id`,
/// from the last file that sets one, behind the same mode gate as an
/// `api_key`: a command in a file others can write would be theirs to
/// choose, and a key file's path is held to the same standard as the key.
/// A loose file is skipped with a warning.
pub fn provider_external(provider_id: &str) -> Option<crate::credentials::ExternalSource> {
    static CACHE: OnceLock<HashMap<String, crate::credentials::ExternalSource>> = OnceLock::new();
    CACHE
        .get_or_init(|| externals_in(files().unwrap_or_default()))
        .get(provider_id)
        .cloned()
}

fn externals_in(files: &[File]) -> HashMap<String, crate::credentials::ExternalSource> {
    let mut merged: HashMap<String, crate::credentials::ExternalSource> = HashMap::new();
    for file in files {
        let set: Vec<_> = file
            .conf
            .providers
            .iter()
            .filter(|(_, p)| p.api_key_command.is_some() || p.api_key_file.is_some())
            .collect();
        let holds_one = set.iter().any(|(_, p)| matches!(p.external(), Ok(Some(_))));
        if holds_one {
            if let Err(e) = owner_readable_only(&file.path) {
                eprintln!(
                    "[llmman] warning: ignoring api_key_command/api_key_file in {}: {e}",
                    file.path.display()
                );
                continue;
            }
        }
        for (id, p) in set {
            match p.external() {
                Ok(Some(source)) => {
                    merged.insert(id.clone(), source);
                }
                // Blank: clears one an earlier file set.
                Ok(None) => {
                    merged.remove(id);
                }
                Err(_) => {}
            }
        }
    }
    merged
}

/// The credential chain `[providers.<id>]` configures, merged across files
/// as [`provider_auth`] is: a later table of the same kind overlays the
/// earlier field by field, one of another kind replaces it. See
/// [`crate::chains`].
pub fn provider_chain(provider_id: &str) -> Option<crate::chains::Chain> {
    static CACHE: OnceLock<HashMap<String, crate::chains::Chain>> = OnceLock::new();
    CACHE
        .get_or_init(|| chains_from(files().unwrap_or_default()))
        .get(provider_id)
        .cloned()
}

fn chains_from(files: &[File]) -> HashMap<String, crate::chains::Chain> {
    let mut merged: HashMap<String, crate::chains::Chain> = HashMap::new();
    for file in files {
        for (id, p) in &file.conf.providers {
            let Ok(Some(chain)) = p.chain() else { continue };
            // A secret written in the table is held to the standard of an
            // `api_key`: a file others can read is not trusted with one.
            let inline_secret = p.azure.as_ref().is_some_and(|a| a.client_secret.is_some())
                || [&p.oauth2, &p.jwt_bearer, &p.ibm_iam]
                    .iter()
                    .any(|t| t.as_ref().is_some_and(|t| t.client_secret.is_some()));
            if inline_secret {
                if let Err(e) = owner_readable_only(&file.path) {
                    eprintln!(
                        "[llmman] warning: ignoring the credential chain of provider {id} in {}: {e}",
                        file.path.display()
                    );
                    continue;
                }
            }
            let next = match merged.remove(id) {
                Some(earlier) => earlier.overlaid(chain),
                None => chain,
            };
            merged.insert(id.clone(), next);
        }
    }
    merged
}

/// Where the per-user credential store lives: beside `llmman.conf`. See
/// [`crate::credentials`]. Tests have no store unless one names a path
/// ([`TEST_CREDENTIALS_FILE`]), for the reason [`load`] reads no real
/// config.
#[cfg(not(test))]
pub fn credentials_path() -> Option<PathBuf> {
    user_dir().map(|d| d.join("credentials.json"))
}

#[cfg(test)]
pub fn credentials_path() -> Option<PathBuf> {
    TEST_CREDENTIALS_FILE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// The credentials file a test reads and writes, if any.
#[cfg(test)]
pub(crate) static TEST_CREDENTIALS_FILE: std::sync::Mutex<Option<PathBuf>> =
    std::sync::Mutex::new(None);

/// Held by every test that points [`TEST_CREDENTIALS_FILE`] somewhere: it
/// is one process-wide value.
#[cfg(test)]
pub(crate) static CREDENTIALS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A temporary credentials store the process's providers read, for as long
/// as the fixture lives. One at a time, process-wide.
#[cfg(test)]
pub(crate) struct CredentialsFixture {
    pub store: crate::credentials::Store,
    _dir: tempfile::TempDir,
    _one: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl CredentialsFixture {
    pub fn new() -> Self {
        let one = CREDENTIALS_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("credentials.json");
        *TEST_CREDENTIALS_FILE
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(path.clone());
        Self {
            store: crate::credentials::Store::at(path),
            _dir: dir,
            _one: one,
        }
    }
}

#[cfg(test)]
impl Drop for CredentialsFixture {
    fn drop(&mut self) {
        *TEST_CREDENTIALS_FILE
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }
}

// ---------------------------------------------------------------------------
// Provider API keys
// ---------------------------------------------------------------------------

/// The key `llmman.conf` configures for `provider_id`, or `None`. Keyed
/// by the id `llmman providers` prints, not by the environment variable,
/// which is models.dev's naming rather than llmman's.
///
/// From a process-lifetime cache, so a daemon serving requests is not
/// stat'ing `/etc` per token; [`reload_provider_keys`] refreshes it after
/// `PUT /llmman/providers/:id/key` writes one.
pub fn provider_api_key(provider_id: &str) -> Option<String> {
    if let Some(keys) = PROVIDER_KEYS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
    {
        return keys.get(provider_id).cloned();
    }
    PROVIDER_KEYS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(|| provider_keys_in(files().unwrap_or_default()))
        .get(provider_id)
        .cloned()
}

static PROVIDER_KEYS: RwLock<Option<HashMap<String, String>>> = RwLock::new(None);

/// Re-reads the provider keys of every `llmman.conf`, so a key the daemon
/// just wrote is spent from the next request on, without a restart. The
/// other settings (aliases, providers, policy) keep their startup values.
///
/// Only `PUT|DELETE /llmman/providers/:id/key` call this: a key written
/// by `llmman config set` or a hand edit reaches a running daemon at its
/// next start, as every other setting does.
///
/// `Err` leaves the keys in force: a file broken by a hand edit must not
/// switch every provider off. The message can quote the file, keys and
/// all, so it is for the daemon's log, not a client.
pub fn reload_provider_keys() -> Result<(), String> {
    reload_provider_keys_from(provider_key_paths())
}

#[cfg(not(test))]
fn provider_key_paths() -> Vec<PathBuf> {
    search_paths()
}

/// Tests read no real config: only the files a test names here.
#[cfg(test)]
fn provider_key_paths() -> Vec<PathBuf> {
    TEST_PROVIDER_FILES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// The file `PUT|DELETE /llmman/providers/:id/key` writes: the per-user
/// llmman.conf, as `llmman config set` does.
#[cfg(not(test))]
pub fn provider_key_file() -> Option<PathBuf> {
    user_path()
}

#[cfg(test)]
pub fn provider_key_file() -> Option<PathBuf> {
    provider_key_paths().last().cloned()
}

/// The config files a test reads provider keys from, and writes them to.
#[cfg(test)]
pub(crate) static TEST_PROVIDER_FILES: std::sync::Mutex<Vec<PathBuf>> =
    std::sync::Mutex::new(Vec::new());

/// Held by every test that reloads or reads the provider-key cache:
/// it is one process-wide value, and a reload in one test would
/// otherwise change what another reads mid-assertion.
#[cfg(test)]
pub(crate) static PROVIDER_KEYS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn reload_provider_keys_from(paths: Vec<PathBuf>) -> Result<(), String> {
    let files = load_from(paths)?;
    *PROVIDER_KEYS
        .write()
        .unwrap_or_else(PoisonError::into_inner) = Some(provider_keys_in(&files));
    Ok(())
}

fn provider_keys_in(files: &[File]) -> HashMap<String, String> {
    let mut accepted = Vec::new();
    for file in files {
        let keys = file.conf.provider_keys();
        // The mode gate is for a file that holds a secret. One that only
        // blanks a key out does not.
        if !keys.values().all(String::is_empty) {
            if let Err(e) = owner_readable_only(&file.path) {
                eprintln!(
                    "[llmman] warning: ignoring the API keys in {}: {e}",
                    file.path.display()
                );
                continue;
            }
        }
        accepted.push(keys);
    }
    merge_keys(accepted)
}

/// Merge per-file key tables, lowest priority first, then drop blanks.
///
/// Blanks survive the merge so a user file can shadow a key `/etc` set,
/// and are dropped only at the end so none is ever spent. Split out to
/// be testable without a filesystem.
fn merge_keys(per_file: Vec<HashMap<String, String>>) -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    for keys in per_file {
        merged.extend(keys);
    }
    merged.retain(|_, key| !key.is_empty());
    merged
}

// ---------------------------------------------------------------------------
// Aggregation peers
// ---------------------------------------------------------------------------

/// Peer daemons, as written: `LLMMAN_PEERS` when set (even empty), else
/// the last file that sets `aggregation.peers`. Replaced, not merged —
/// half of two aggregations is no one's intent — so `peers = ""` opts out.
pub fn peers() -> Vec<String> {
    peers_from(
        std::env::var("LLMMAN_PEERS").ok().as_deref(),
        files().unwrap_or_default(),
    )
}

fn peers_from(env: Option<&str>, files: &[File]) -> Vec<String> {
    split_list(env.or_else(|| {
        files
            .iter()
            .rev()
            .find_map(|f| f.conf.aggregation.peers.as_deref())
    }))
}

// ---------------------------------------------------------------------------
// Daemon API keys
// ---------------------------------------------------------------------------

/// The keys `llmman serve` requires: `LLMMAN_API_KEYS` when set (even
/// empty), else the last file setting `auth.api_keys` that passes the
/// mode gate. Replaced, not merged, like [`peers`].
pub fn auth_api_keys() -> Vec<String> {
    split_list(
        std::env::var("LLMMAN_API_KEYS")
            .ok()
            .or_else(|| {
                secret_from_files(files().unwrap_or_default(), "auth.api_keys", |f| {
                    f.conf.auth.api_keys.as_deref()
                })
            })
            .as_deref(),
    )
}

/// The key presented to peers: `LLMMAN_PEER_API_KEY`, else the last file
/// setting `aggregation.api_key` that passes the mode gate. `None` for blank.
pub fn peer_api_key() -> Option<String> {
    std::env::var("LLMMAN_PEER_API_KEY")
        .ok()
        .or_else(|| {
            secret_from_files(files().unwrap_or_default(), "aggregation.api_key", |f| {
                f.conf.aggregation.api_key.as_deref()
            })
        })
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

/// The last file that sets a secret field, behind [`provider_keys_in`]'s
/// mode gate: a loose file is skipped with a warning. A blank is returned
/// as such — it is how a user file clears a system-wide value.
fn secret_from_files<'a>(
    files: &'a [File],
    field: &str,
    pick: impl Fn(&'a File) -> Option<&'a str>,
) -> Option<String> {
    for file in files.iter().rev() {
        let Some(value) = pick(file) else { continue };
        if !value.trim().is_empty() {
            if let Err(e) = owner_readable_only(&file.path) {
                eprintln!(
                    "[llmman] warning: ignoring {field} in {}: {e}",
                    file.path.display()
                );
                continue;
            }
        }
        return Some(value.to_string());
    }
    None
}

/// Refuses a file that group or other can read, the way `ssh` refuses a
/// loose private key.
///
/// Gates the keys, not the read: `/etc/llmman/llmman.conf` is
/// legitimately world-readable for the aliases and trust policy it also
/// carries, so a loose file still supplies everything but the key.
#[cfg(unix)]
pub(crate) fn owner_readable_only(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)
        .map_err(|e| e.to_string())?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err(format!(
            "mode {:04o} lets other users read it. Run: chmod 600 {}",
            mode & 0o7777,
            path.display()
        ));
    }
    Ok(())
}

/// Unchecked on Windows: there are no mode bits, and reading an ACL
/// needs a `windows` dependency this crate does not have. Defaults are
/// owner-only under the user's profile, but a file with a widened ACL,
/// or one under the system directory, is accepted as-is.
#[cfg(not(unix))]
pub(crate) fn owner_readable_only(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key written while the daemon runs is spent from the next
    /// request on: the reload re-reads the files, behind the same mode
    /// gate as at startup.
    #[cfg(unix)]
    #[test]
    fn reloaded_provider_keys_take_effect_without_a_restart() {
        use std::os::unix::fs::PermissionsExt;

        let _one = PROVIDER_KEYS_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let id = "llmman-test-reload";
        let dir = std::env::temp_dir().join(format!("llmman-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(FILE);
        std::fs::write(&path, format!("[providers.{id}]\napi_key = \"sk-new\"\n")).expect("write");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        reload_provider_keys_from(vec![path.clone()]).expect("reload");
        assert_eq!(
            provider_api_key(id),
            None,
            "a world-readable file's key is ignored"
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        reload_provider_keys_from(vec![path.clone()]).expect("reload");
        assert_eq!(provider_api_key(id).as_deref(), Some("sk-new"));

        // A file that no longer parses: the error, and the keys in force stay.
        std::fs::write(&path, format!("[providers.{id}]\napi_key = \"sk-broken\n")).expect("write");
        assert!(reload_provider_keys_from(vec![path.clone()]).is_err());
        assert_eq!(provider_api_key(id).as_deref(), Some("sk-new"));

        reload_provider_keys_from(Vec::new()).expect("reload");
        assert_eq!(provider_api_key(id), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn conf(text: &str) -> Conf {
        parse(text).expect("valid conf")
    }

    #[test]
    fn pager_configuration_merges_each_key_independently() {
        let system = file(
            r#"[core]
pager = "less -S"
[pager]
providers = "false"
log = "more""#,
        );
        let user = file(
            r#"[pager]
log = """#,
        );
        assert_eq!(
            pager_from_files(&[system, user], "log"),
            (Some("less -S".into()), Some("".into()))
        );
        let system = file(
            r#"[core]
pager = "less"
[pager]
providers = "false""#,
        );
        let user = file(
            r#"[core]
pager = "cat""#,
        );
        assert_eq!(
            pager_from_files(&[system, user], "providers"),
            (Some("cat".into()), Some("false".into()))
        );
        assert!(parse(
            r#"[core]
pagr = "less""#
        )
        .is_err());
        assert!(parse(
            r#"[pager]
provider = "false""#
        )
        .is_err());
    }

    #[test]
    fn pager_settings_support_model_listing_commands() {
        let settings = file(
            r#"[pager]
list = "yes"
search = "less -S"
ps = "off""#,
        );
        for (command, value) in [("list", "yes"), ("search", "less -S"), ("ps", "off")] {
            assert_eq!(
                pager_from_files(std::slice::from_ref(&settings), command),
                (None, Some(value.into()))
            );
        }
    }

    /// `$HOME` is what `dirs::home_dir` reads on Unix anyway, so this
    /// pins the Windows half: `%USERPROFILE%`, which it ignores.
    #[test]
    fn a_home_comes_from_the_environment_when_it_is_absolute() {
        let expected: &[&str] = if cfg!(windows) {
            &["USERPROFILE"]
        } else {
            &["HOME"]
        };
        assert_eq!(HOME_KEYS, expected);

        let take = |value: &str| {
            let value = value.to_string();
            env_home_from(HOME_KEYS, move |_| {
                Some(std::ffi::OsString::from(value.clone()))
            })
        };
        let home = if cfg!(windows) {
            "C:\\Users\\u"
        } else {
            "/home/u"
        };
        assert_eq!(take(home), Some(PathBuf::from(home)));
        assert_eq!(take(""), None);
        assert_eq!(take("relative/home"), None);
        // What a POSIX shell exports to a Windows binary: rooted, but
        // not absolute, so it would follow the current drive.
        #[cfg(windows)]
        assert_eq!(take("/c/Users/u"), None);
        assert_eq!(env_home_from(HOME_KEYS, |_| None), None);
    }

    /// A key must not reach a log through the derived `Debug` of the
    /// public `Conf`/`File`.
    #[test]
    fn debug_output_never_carries_a_key() {
        let c = conf("[providers.openai]\napi_key = \"sk-secret-value\"");
        let rendered = format!("{c:?}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
        assert!(rendered.contains("redacted"), "{rendered}");
    }

    /// The documented shape, all three sections in one file.
    #[test]
    fn parses_every_section_of_one_file() {
        let c = conf(
            r#"
            [aliases]
            gemma4 = "docker.io/ai/gemma4"

            [providers.openrouter]
            api_key = "sk-or-abc"

            [verify]
            default = "warn"
            trusted_root = "sigstore/trusted_root.json"

            [[verify.trust]]
            pattern = "docker.io/myorg/**"
            keys    = ["keys/myorg.pub"]
            mode    = "enforce"
            claims  = "docker.io/myorg/**"

            [[verify.trust]]
            pattern = "ghcr.io/myorg/**"
            issuer  = "https://token.actions.githubusercontent.com"
            subject = "https://github.com/myorg/models/.github/workflows/sign.yml@refs/heads/main"
            mode    = "enforce"

            [aggregation]
            peers = "asahi,spark:17434"

            [registries."docker.io"]
            mirrors = "https://mirror.gcr.io,registry-mirror.corp:5000"
            "#,
        );
        assert_eq!(
            c.aliases.get("gemma4").map(String::as_str),
            Some("docker.io/ai/gemma4")
        );
        assert_eq!(c.aggregation.peers.as_deref(), Some("asahi,spark:17434"));
        assert_eq!(
            c.registries["docker.io"].mirrors.as_deref(),
            Some("https://mirror.gcr.io,registry-mirror.corp:5000")
        );
        assert_eq!(
            c.provider_keys().get("openrouter").map(String::as_str),
            Some("sk-or-abc")
        );
        assert_eq!(c.verify.default.as_deref(), Some("warn"));
        assert_eq!(
            c.verify.trusted_root.as_deref(),
            Some("sigstore/trusted_root.json")
        );
        assert_eq!(c.verify.trust.len(), 2);
        assert_eq!(c.verify.trust[0].mode.as_deref(), Some("enforce"));
        assert_eq!(
            c.verify.trust[0].claims.as_deref(),
            Some("docker.io/myorg/**")
        );
        assert!(c.verify.trust[0].issuer.is_none());
        assert!(c.verify.trust[0].subject.is_none());
        assert!(c.verify.trust[1].keys.is_empty());
        assert_eq!(
            c.verify.trust[1].issuer.as_deref(),
            Some("https://token.actions.githubusercontent.com")
        );
        assert_eq!(
            c.verify.trust[1].subject.as_deref(),
            Some("https://github.com/myorg/models/.github/workflows/sign.yml@refs/heads/main")
        );
    }

    /// A file that sets one section must not need the other two.
    #[test]
    fn every_section_is_optional() {
        let c = conf("");
        assert!(c.aliases.is_empty());
        assert!(c.provider_keys().is_empty());
        assert!(c.verify.default.is_none());
        assert!(c.verify.trust.is_empty());

        assert!(!conf("[aliases]\nx = \"y\"").aliases.is_empty());
        assert!(conf("[verify]\ndefault = \"off\"").aliases.is_empty());
    }

    /// A key is trimmed; one that sets no `api_key` at all is absent.
    /// A blank *is* carried this far — see [`merge_keys`].
    #[test]
    fn a_key_is_trimmed_and_an_unset_one_is_absent() {
        let keys = conf(
            r#"
            [providers.openai]
            api_key = "  sk-padded  "

            [providers.groq]
            api_key = "   "

            [providers.mistral]
            "#,
        )
        .provider_keys();
        assert_eq!(keys.get("openai").map(String::as_str), Some("sk-padded"));
        assert_eq!(keys.get("groq").map(String::as_str), Some(""));
        assert_eq!(keys.get("mistral"), None);
    }

    /// A user file must be able to blank out a key `/etc` set, or there
    /// is no way to opt out of a system-wide credential. A blank never
    /// survives as an empty bearer token either.
    #[test]
    fn a_later_blank_shadows_an_earlier_key_and_is_never_spent() {
        let system = HashMap::from([
            ("openai".to_string(), "sk-system".to_string()),
            ("groq".to_string(), "sk-groq".to_string()),
        ]);
        let user = HashMap::from([("openai".to_string(), String::new())]);

        let merged = merge_keys(vec![system, user]);
        assert_eq!(merged.get("openai"), None, "blanked out by the user file");
        assert_eq!(merged.get("groq").map(String::as_str), Some("sk-groq"));

        // And the ordinary case: a later real key replaces an earlier one.
        let merged = merge_keys(vec![
            HashMap::from([("openai".to_string(), "sk-system".to_string())]),
            HashMap::from([("openai".to_string(), "sk-user".to_string())]),
        ]);
        assert_eq!(merged.get("openai").map(String::as_str), Some("sk-user"));
    }

    /// A misspelling that parsed happily would silently never take
    /// effect. Every section rejects one.
    #[test]
    fn a_misspelled_section_or_field_is_rejected_rather_than_ignored() {
        assert!(parse("[provider.openai]\napi_key = \"x\"").is_err());
        assert!(parse("[providers.openai]\napi_kye = \"x\"").is_err());
        assert!(parse("[alias]\nx = \"y\"").is_err());
        assert!(parse("[verify]\ndefualt = \"off\"").is_err());
        assert!(parse("[verify]\nlayres = \"off\"").is_err());
        assert!(parse("[[verify.trust]]\npattern = \"a/b\"\nkyes = []").is_err());
        assert!(parse("[[verify.trust]]\npattern = \"a/b\"\nclaim = \"c/d\"").is_err());
        assert!(parse("[registry.\"docker.io\"]\nmirrors = \"m\"").is_err());
        assert!(parse("[registries.\"docker.io\"]\nmirror = \"m\"").is_err());
        assert!(parse("api_key = \"x\"").is_err());
    }

    /// A mangled file is reported, not treated as empty.
    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse("[providers.openai").is_err());
    }

    // -- configured providers ------------------------------------------------

    /// A `[providers.<id>]` with a `base_url` defines a provider; every
    /// other field has a default. One without is only a key, as before.
    #[test]
    fn a_base_url_turns_a_provider_table_into_a_definition() {
        let files = [file(
            r#"
            [providers.inferencebox]
            base_url = "http://inferencebox:8000/v1/"

            [providers.relay]
            base_url    = "https://relay.example/v1"
            wire        = "anthropic"
            api_key_env = "RELAY_KEY"
            name        = "Claude relay"
            api_key     = "sk-relay"

            [providers.openrouter]
            api_key = "sk-or"
            "#,
        )];
        let configured = configured_from(&files);
        assert_eq!(
            configured,
            vec![
                ConfiguredProvider {
                    id: "inferencebox".into(),
                    name: "inferencebox".into(),
                    // Trailing slash gone: a route appends its own.
                    base_url: "http://inferencebox:8000/v1".into(),
                    wire: crate::providers::Wire::OpenAi,
                    key_env: None,
                },
                ConfiguredProvider {
                    id: "relay".into(),
                    name: "Claude relay".into(),
                    base_url: "https://relay.example/v1".into(),
                    wire: crate::providers::Wire::Anthropic,
                    key_env: Some("RELAY_KEY".into()),
                },
            ]
        );
        // The key of a defined provider is reached the same way as any.
        assert_eq!(
            files[0]
                .conf
                .provider_keys()
                .get("relay")
                .map(String::as_str),
            Some("sk-relay")
        );
        assert!(configured_from(&[]).is_empty());
    }

    /// A later file re-defining an id replaces the fields it sets and
    /// keeps the rest, as keys merge.
    #[test]
    fn a_later_definition_overrides_field_by_field() {
        let system = file(
            "[providers.inferencebox]\nbase_url = \"http://inferencebox:8000/v1\"\nname = \"Shared box\"\n\
             api_key_env = \"SHARED_KEY\"",
        );
        let user = file(
            "[providers.inferencebox]\nbase_url = \"http://10.0.0.5:8000/v1\"\nwire = \"anthropic\"\n\
             api_key_env = \"\"",
        );
        let merged = configured_from(&[system, user]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].base_url, "http://10.0.0.5:8000/v1");
        assert_eq!(merged[0].name, "Shared box");
        assert_eq!(merged[0].wire, crate::providers::Wire::Anthropic);
        // A blank clears the inherited variable, so /etc's key does not
        // follow the user to another endpoint.
        assert_eq!(merged[0].key_env, None);
    }

    /// What `config set` has to refuse on the spot: a `base_url` that is
    /// not an absolute http(s) URL, a definition field on a table that
    /// defines nothing, a `wire` llmman does not speak.
    #[test]
    fn a_bad_provider_definition_is_a_parse_error() {
        let bad = |body: &str| parse(&format!("[providers.inferencebox]\n{body}")).expect_err(body);

        assert!(bad("base_url = \"inferencebox:8000\"").contains("base_url"));
        assert!(bad("base_url = \"inferencebox:8000/v1\"").contains("base_url"));
        assert!(bad("base_url = \"ftp://inferencebox/v1\"").contains("http or https"));
        assert!(bad("base_url = \"http://\"").contains("base_url"));
        assert!(bad("base_url = \"http://inferencebox/v1?x=1\"").contains("query"));
        assert!(bad("base_url = \"\"").contains("base_url"));
        // The URL is reported by the daemon's API; a secret does not go in it.
        assert!(bad("base_url = \"https://user:pw@inferencebox/v1\"").contains("api_key"));
        assert!(bad("base_url = \"https://user@inferencebox/v1\"").contains("api_key"));
        // A slash in the id would never survive `split_remote_ref`.
        assert!(
            parse("[providers.\"team/gpu\"]\nbase_url = \"http://g/v1\"")
                .expect_err("slash id")
                .contains("'/'")
        );
        assert!(parse("[providers.\"\"]\napi_key = \"x\"").is_err());

        let err = bad("wire = \"anthropic\"");
        assert!(err.contains("no base_url"), "{err}");
        assert!(err.contains("wire"), "{err}");
        let err = bad("api_key_env = \"X\"\nname = \"n\"");
        assert!(err.contains("api_key_env, name"), "{err}");

        assert!(bad("base_url = \"http://g/v1\"\nwire = \"ollama\"").contains("wire"));
        assert!(bad("base_url = \"http://g/v1\"\nbase_ulr = \"x\"").contains("base_ulr"));

        // The good shapes parse, and come out normalized: lowercase
        // scheme and host, no trailing slash.
        for (url, want) in [
            ("http://inferencebox:8000/v1", "http://inferencebox:8000/v1"),
            ("http://inferencebox", "http://inferencebox"),
            (
                "HTTP://InferenceBox:8000/v1/",
                "http://inferencebox:8000/v1",
            ),
            ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
            ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
            (
                "https://relay.example/api/v1",
                "https://relay.example/api/v1",
            ),
        ] {
            assert_eq!(base_url_of(url).expect(url), want);
            parse(&format!("[providers.inferencebox]\nbase_url = {url:?}")).expect(url);
        }
    }

    /// The definition fields show in `Debug`; the key still does not.
    #[test]
    fn debug_output_shows_a_definition_but_never_a_key() {
        let c = conf(
            "[providers.inferencebox]\nbase_url = \"http://inferencebox:8000/v1\"\napi_key = \"sk-secret-value\"",
        );
        let rendered = format!("{c:?}");
        assert!(rendered.contains("inferencebox:8000"), "{rendered}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }

    #[test]
    fn managed_configuration_is_opt_in_strict_and_uses_field_precedence() {
        assert!(!managed_from_files(&[]).unwrap().enabled);
        let base = file("[managed]\nenabled = \"true\"\nread_timeout_seconds = \"600\"");
        let override_file = file("[managed]\nread_timeout_seconds = \"0\"");
        let settings = managed_from_files(&[base, override_file]).unwrap();
        assert!(settings.enabled);
        assert!(settings.read_timeout.is_none());
        let settings =
            managed_from_files(&[file("[managed]\nread_timeout_seconds = \"300\"")]).unwrap();
        assert_eq!(
            settings.read_timeout,
            Some(std::time::Duration::from_secs(300))
        );
        assert!(!settings.enabled);
        for text in [
            "[managed]\nenabled = \"yes\"",
            "[managed]\nread_timeout_seconds = \"-1\"",
            "[managed]\nread_timeout_seconds = \"4294967296\"",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
        assert!(parse("[managed]\nenabled = true").is_err());
        assert!(parse("[managed]\nenabeld = \"true\"").is_err());
    }

    // -- aggregation peers ---------------------------------------------------

    fn file(text: &str) -> File {
        File {
            path: PathBuf::from("llmman.conf"),
            dir: PathBuf::from("."),
            conf: conf(text),
        }
    }

    #[test]
    fn peers_come_from_the_environment_then_the_last_file_that_sets_them() {
        let system = || file("[aggregation]\npeers = \"a, b\"");
        let user = file("[aggregation]\npeers = \"c\"");
        let silent = file("");
        let opted_out = file("[aggregation]\npeers = \"\"");

        assert_eq!(
            peers_from(Some(" x , ,y:1 "), &[]),
            vec!["x".to_string(), "y:1".to_string()]
        );
        assert!(peers_from(Some(""), &[system()]).is_empty());
        assert_eq!(
            peers_from(None, &[system(), silent]),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(peers_from(None, &[system(), user]), vec!["c".to_string()]);
        assert!(peers_from(None, &[system(), opted_out]).is_empty());
        assert!(peers_from(None, &[]).is_empty());
        assert!(parse("[aggregation]\npeer = \"a\"").is_err());
    }

    /// The daemon's own keys and the peer key are secrets: redacted in
    /// `Debug`, and read from the last file that sets them — skipping,
    /// with a warning, one that other users can read.
    #[cfg(unix)]
    #[test]
    fn daemon_and_peer_keys_are_redacted_and_gated_on_the_file_mode() {
        use std::os::unix::fs::PermissionsExt;

        let c = conf("[auth]\napi_keys = \"k-secret,k2\"\n[aggregation]\napi_key = \"p-secret\"");
        let rendered = format!("{c:?}");
        assert!(!rendered.contains("secret"), "{rendered}");
        assert!(
            parse("[auth]\napi_key = \"k\"").is_err(),
            "the field is plural"
        );

        let dir = std::env::temp_dir().join(format!("llmman-auth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let write = |name: &str, text: &str, mode: u32| {
            let path = dir.join(name);
            std::fs::write(&path, text).expect("write");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).expect("chmod");
        };
        write(
            "system.conf",
            "[auth]\napi_keys = \"a, b\"\n[aggregation]\napi_key = \"pa\"",
            0o600,
        );
        write(
            "loose.conf",
            "[auth]\napi_keys = \"c\"\n[aggregation]\napi_key = \"pc\"",
            0o644,
        );
        write(
            "out.conf",
            "[auth]\napi_keys = \"\"\n[aggregation]\napi_key = \"\"",
            0o644,
        );
        let files = |names: &[&str]| -> Vec<File> {
            names
                .iter()
                .map(|n| {
                    let path = dir.join(n);
                    File {
                        dir: dir.clone(),
                        conf: parse(&std::fs::read_to_string(&path).unwrap()).unwrap(),
                        path,
                    }
                })
                .collect()
        };
        let keys = |files: &[File]| {
            split_list(
                secret_from_files(files, "auth.api_keys", |f| f.conf.auth.api_keys.as_deref())
                    .as_deref(),
            )
        };
        let peer = |files: &[File]| {
            secret_from_files(files, "aggregation.api_key", |f| {
                f.conf.aggregation.api_key.as_deref()
            })
        };
        let ab = vec!["a".to_string(), "b".to_string()];
        assert_eq!(keys(&files(&["system.conf"])), ab);
        assert_eq!(peer(&files(&["system.conf"])).as_deref(), Some("pa"));
        // The loose file's secret is skipped, and the system one used.
        assert_eq!(keys(&files(&["system.conf", "loose.conf"])), ab);
        assert_eq!(
            peer(&files(&["system.conf", "loose.conf"])).as_deref(),
            Some("pa")
        );
        // A blank needs no mode: it holds nothing, and clears the system one.
        assert!(keys(&files(&["system.conf", "out.conf"])).is_empty());
        assert_eq!(
            peer(&files(&["system.conf", "out.conf"])).as_deref(),
            Some("")
        );
        assert!(keys(&[]).is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- registry mirrors ----------------------------------------------------

    /// A mirror is spelled like a URL with the scheme optional, and
    /// comes out normalized the way the Go shim wants it.
    #[test]
    fn a_mirror_is_normalized_and_defaults_to_https() {
        assert_eq!(
            mirror_url_of("mirror.gcr.io").as_deref(),
            Ok("https://mirror.gcr.io")
        );
        assert_eq!(
            mirror_url_of(" HTTP://Mirror.Corp:5000/ ").as_deref(),
            Ok("http://mirror.corp:5000")
        );
        assert_eq!(
            mirror_url_of("https://proxy.corp/registry/").as_deref(),
            Ok("https://proxy.corp/registry")
        );
        for bad in [
            "",
            "ftp://mirror",
            "https://",
            "https://user:pw@mirror",
            "https://mirror?x=1",
            "https://mirror#frag",
        ] {
            assert!(mirror_url_of(bad).is_err(), "{bad:?}");
        }
    }

    /// The table key is a host. Hub's aliases fold onto `docker.io`,
    /// which is what a reference's `docker.io/ai/x` names and what the
    /// Go shim is asked for.
    #[test]
    fn a_registry_key_is_a_host_and_hub_has_one_name() {
        assert_eq!(
            canonical_registry_host("Index.Docker.io").as_deref(),
            Some("docker.io")
        );
        assert_eq!(
            canonical_registry_host("registry-1.docker.io").as_deref(),
            Some("docker.io")
        );
        assert_eq!(
            canonical_registry_host("localhost:5000").as_deref(),
            Some("localhost:5000")
        );
        assert_eq!(
            canonical_registry_host("ghcr.io").as_deref(),
            Some("ghcr.io")
        );
        assert_eq!(
            canonical_registry_host("[::1]:5000").as_deref(),
            Some("[::1]:5000")
        );
        assert_eq!(
            canonical_registry_host("ghcr.io:443").as_deref(),
            Some("ghcr.io")
        );
        for bad in [
            "",
            "https://ghcr.io",
            "ghcr.io/org",
            "*.example.com",
            "host:port",
            ":5000",
            "user@registry",
            "registry?q=1",
            "registry#f",
            "[not-an-ip]:5000",
            "a b",
        ] {
            assert_eq!(canonical_registry_host(bad), None, "{bad:?}");
        }
        assert!(parse("[registries.\"https://ghcr.io\"]\nmirrors = \"m\"").is_err());
        assert!(parse("[registries.\"ghcr.io\"]\nmirrors = \"ftp://m\"").is_err());
        assert!(parse("[registries.\"ghcr.io\"]\nmirrors = \"m\"").is_ok());
        // Two keys for one registry would leave which wins to HashMap order.
        assert!(parse(
            "[registries.\"docker.io\"]\nmirrors = \"a\"\n[registries.\"index.docker.io\"]\nmirrors = \"b\""
        )
        .is_err());
        assert!(parse(
            "[registries.\"ghcr.io\"]\nmirrors = \"a\"\n[registries.\"GHCR.io\"]\nmirrors = \"b\""
        )
        .is_err());
    }

    /// Per host, the environment (Hub only), then the last file that
    /// sets it; a blank opts a host out; hosts sort, mirrors keep order.
    #[test]
    fn mirrors_come_from_the_environment_then_the_last_file_that_sets_them() {
        let system = || {
            file(
                r#"
                [registries."docker.io"]
                mirrors = "https://a, b:5000"

                [registries."ghcr.io"]
                mirrors = "http://g"
                "#,
            )
        };
        let user = file("[registries.\"index.docker.io\"]\nmirrors = \"c\"");
        let opted_out = file("[registries.\"docker.io\"]\nmirrors = \"\"");
        let silent = file("[registries.\"docker.io\"]");

        let got = registry_mirrors_from(None, &[system(), silent]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://a".to_string(), "https://b:5000".to_string()][..])
        );
        assert_eq!(
            got.get("ghcr.io").map(Vec::as_slice),
            Some(&["http://g".to_string()][..])
        );

        let got = registry_mirrors_from(None, &[system(), user]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://c".to_string()][..]),
            "a Hub alias replaces docker.io's own entry"
        );

        let got = registry_mirrors_from(None, &[system(), opted_out]);
        assert!(!got.contains_key("docker.io"));
        assert!(
            got.contains_key("ghcr.io"),
            "opting Hub out leaves ghcr.io alone"
        );

        let got = registry_mirrors_from(Some(" x , ,http://y:1/ "), &[system()]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://x".to_string(), "http://y:1".to_string()][..])
        );
        assert!(!registry_mirrors_from(Some(""), &[system()]).contains_key("docker.io"));
        assert!(registry_mirrors_from(Some("ftp://nope"), &[]).is_empty());
        assert!(registry_mirrors_from(None, &[]).is_empty());
    }

    // -- launch defaults -----------------------------------------------------

    #[test]
    fn launch_defaults_come_from_the_integrations_own_table() {
        let f = [file(
            "[launch.opencode]\nprovider = \"anthropic\"\nmodel = \"claude-sonnet-5-5\"\n\
             variant = \"xhigh\"\n[launch.claude]\nmodel = \"qwen3.8\"",
        )];
        assert_eq!(
            launch_defaults_from(&f, "opencode"),
            launch_conf(Some("anthropic"), Some("claude-sonnet-5-5"), Some("xhigh"))
        );
        assert_eq!(
            launch_defaults_from(&f, "claude"),
            launch_conf(None, Some("qwen3.8"), None)
        );
        for other in ["codex", "nonesuch"] {
            assert_eq!(
                launch_defaults_from(&f, other),
                LaunchConf::default(),
                "{other}"
            );
        }
        assert_eq!(launch_defaults_from(&[], "opencode"), LaunchConf::default());
    }

    /// `launch` takes any case, and `copilot-cli` for `copilot`.
    #[test]
    fn launch_defaults_are_found_through_case_and_alias() {
        let f = [file(
            "[launch.OpenCode]\nmodel = \"m\"\n[launch.copilot-cli]\nmodel = \"c\"",
        )];
        assert_eq!(
            launch_defaults_from(&f, "OPENCODE").model.as_deref(),
            Some("m")
        );
        for name in ["copilot", "copilot-cli", "Copilot"] {
            assert_eq!(
                launch_defaults_from(&f, name).model.as_deref(),
                Some("c"),
                "{name}"
            );
        }
    }

    /// One key at a time; a blank (or whitespace) value clears one.
    #[test]
    fn a_later_file_overrides_launch_defaults_one_key_at_a_time() {
        let system = file(
            "[launch.opencode]\nprovider = \"anthropic\"\nmodel = \"sys\"\nvariant = \"high\"",
        );
        let user = file("[launch.opencode]\nmodel = \"mine\"\nvariant = \"\"");
        assert_eq!(
            launch_defaults_from(&[system, user], "opencode"),
            launch_conf(Some("anthropic"), Some("mine"), None)
        );
        let blank = file("[launch.opencode]\nprovider = \"  \"\nmodel = \" m \"");
        assert_eq!(
            launch_defaults_from(&[blank], "opencode"),
            launch_conf(None, Some("m"), None)
        );
    }

    #[test]
    fn a_launch_table_must_name_an_integration_a_key_and_a_variant() {
        let err = |text: &str| parse(text).expect_err("must not parse");
        assert!(err("[launch.opencod]\nmodel = \"m\"").contains("opencod"));
        assert!(err("[launch.opencode]\nmodle = \"m\"").contains("modle"));
        assert!(err("[launch.opencode]\nvariant = \"turbo\"").contains("turbo"));
        assert!(err("[launch.opencode]\nvariant = 3").contains("string"));
        assert!(err("[launch]\nmodel = \"m\"").contains("model"));
        let both = "[launch.copilot]\nmodel = \"a\"\n[launch.copilot-cli]\nmodel = \"b\"";
        assert!(err(both).contains("same integration"));
        for variant in crate::cmd::run::variants().chain(["", "  "]) {
            parse(&format!("[launch.opencode]\nvariant = {variant:?}")).expect(variant);
        }
    }

    // -- search paths --------------------------------------------------------

    /// Two locations, system first so the user's file wins — and none
    /// of the package-managed ones earlier versions searched.
    #[test]
    fn the_search_path_is_etc_then_the_user_directory() {
        let paths = search_paths();
        assert_eq!(
            paths.first().expect("a system path"),
            &system_dir().join(FILE)
        );
        assert!(system_dir().ends_with("etc/llmman"), "{:?}", system_dir());
        // Rooted, so it cannot resolve against the working directory.
        assert!(system_dir().is_absolute(), "{:?}", system_dir());
        assert!(
            !paths.iter().any(|p| p.starts_with("/usr/share")),
            "{paths:?}"
        );
        if let Some(user) = user_path() {
            assert_eq!(paths.last(), Some(&user));
            assert!(user.ends_with("llmman/llmman.conf"), "{}", user.display());
        }
    }

    /// One per-user location on every platform, so there is exactly one
    /// place `user_path` can name in an error.
    #[test]
    fn there_is_one_user_directory_on_every_platform() {
        assert_eq!(search_paths().len(), user_dir().iter().count() + 1);
        if let Some(dir) = user_dir() {
            assert!(dir.ends_with(".config/llmman"), "{}", dir.display());
        }
    }

    /// A key any account on the box can read is not one worth spending.
    ///
    /// Everything here is inlined rather than sharing a helper: a helper
    /// used only by a `cfg(unix)` test is dead code on Windows, which
    /// `clippy --all-targets -D warnings` rejects.
    #[cfg(unix)]
    #[test]
    fn a_group_or_world_readable_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("llmman-conf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(FILE);
        std::fs::write(&path, "[providers.openai]\napi_key = \"sk-x\"\n").expect("write");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let err = owner_readable_only(&path).expect_err("0644 is refused");
        assert!(err.contains("chmod 600"), "{err}");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        assert!(owner_readable_only(&path).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- provider auth ---------------------------------------------------------

    /// A file on disk, owner-only, for the checks that read its mode.
    fn private_file(dir: &Path, name: &str, text: &str) -> File {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        File {
            path,
            dir: dir.to_path_buf(),
            conf: conf(text),
        }
    }

    #[cfg(unix)]
    #[test]
    fn static_headers_are_only_trusted_from_an_owner_only_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let text = "[providers.p]\nheaders = { X-Gateway = \"t\" }\n";
        let loaded = private_file(dir.path(), FILE, text);
        assert_eq!(
            auth_from(std::slice::from_ref(&loaded))["p"].headers.len(),
            1
        );
        std::fs::set_permissions(&loaded.path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(auth_from(&[loaded])["p"].headers.is_empty());
    }

    #[test]
    fn provider_auth_fields_parse_merge_by_name_and_later_files_win() {
        use crate::authscheme::Scheme;
        let dir = tempfile::tempdir().unwrap();
        let system = private_file(
            dir.path(),
            "sys.conf",
            "[providers.azure]\nauth = \"header\"\nauth_header = \"api-key\"\n\
             headers = { X-A = \"1\", X-B = \"2\" }\nquery = { api-version = \"v1\" }\n\
             vars = { AZURE_RESOURCE_NAME = \"res\" }\n",
        );
        let user = private_file(
            dir.path(),
            "user.conf",
            "[providers.azure]\nheaders = { x-a = \"9\" }\nquery = { api-version = \"2024\" }\n\
             vars = { OTHER = \"o\" }\n[providers.google]\nauth_header = \"x-goog-api-key\"\n",
        );
        let merged = auth_from(&[system, user]);
        let azure = &merged["azure"];
        // The user file named no scheme, so the system's stands.
        assert_eq!(azure.scheme, Scheme::Header("api-key".into()));
        assert_eq!(
            azure.headers,
            vec![
                ("X-B".to_string(), "2".to_string()),
                ("x-a".to_string(), "9".to_string())
            ]
        );
        assert_eq!(
            azure.query,
            vec![("api-version".to_string(), "2024".to_string())]
        );
        assert_eq!(azure.vars.len(), 2);
        assert_eq!(
            merged["google"].scheme,
            Scheme::Header("x-goog-api-key".into())
        );
        // Nothing configured: the default, whatever the id.
        assert_eq!(
            provider_auth("nobody-configured-this"),
            crate::authscheme::Auth::default()
        );
    }

    #[test]
    fn provider_auth_mistakes_are_refused_at_load() {
        for (text, wants) in [
            ("[providers.p]\nauth = \"basic\"", "not one of"),
            ("[providers.p]\nauth = \"header\"", "auth_header"),
            (
                "[providers.p]\nauth = \"bearer\"\nauth_header = \"x\"",
                "only go with",
            ),
            ("[providers.p]\nauth_header = \"a b\"", "header name"),
            (
                "[providers.p]\nauth_header = \"a\"\nauth_query = \"b\"",
                "alternatives",
            ),
            (
                "[providers.p]\nheaders = { \"bad name\" = \"v\" }",
                "header name",
            ),
            (
                "[providers.p]\nheaders = { Host = \"evil\" }",
                "frames the request",
            ),
            ("[providers.p]\nquery = { \"a&b\" = \"1\" }", "query"),
            (
                "[providers.p]\nvars = { \"1bad\" = \"x\" }",
                "variable name",
            ),
            (
                "[providers.p]\napi_key_command = \"a\"\napi_key_file = \"b\"",
                "alternatives",
            ),
            ("[providers.p]\ntypo = 1", "unknown field"),
        ] {
            let err = parse(text).expect_err(text);
            assert!(err.contains(wants), "{text}: {err}");
        }
        // And a header's value is not echoed.
        let err = parse("[providers.p]\nheaders = { X = \"line\\nbreak-secret\" }").unwrap_err();
        assert!(!err.contains("break-secret"), "{err}");
        // A catalog provider may be given auth without a base_url.
        parse("[providers.azure]\nauth_header = \"api-key\"\nheaders = { X = \"1\" }").unwrap();
    }

    /// A command in a file anyone can write is theirs to choose: it is held
    /// to the standard of a key.
    #[cfg(unix)]
    #[test]
    fn a_key_command_is_only_read_from_an_owner_only_file() {
        use crate::credentials::ExternalSource;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let text = "[providers.p]\napi_key_command = \"pass show p\"\n[providers.q]\napi_key_file = \"~/q.key\"\n";
        std::fs::write(&path, text).unwrap();
        let loaded = |path: &PathBuf| {
            vec![File {
                path: path.clone(),
                dir: dir.path().to_path_buf(),
                conf: conf(text),
            }]
        };

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            externals_in(&loaded(&path)).is_empty(),
            "a loose file's command is ignored"
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let found = externals_in(&loaded(&path));
        assert_eq!(
            found["p"],
            ExternalSource::Command {
                command: "pass show p".into()
            }
        );
        assert_eq!(
            found["q"],
            ExternalSource::File {
                path: "~/q.key".into()
            }
        );

        // A later file blanks one out, as `api_key = ""` does.
        let blank = File {
            path: path.clone(),
            dir: dir.path().to_path_buf(),
            conf: conf("[providers.p]\napi_key_command = \"\"\n"),
        };
        let mut files = loaded(&path);
        files.push(blank);
        let found = externals_in(&files);
        assert!(!found.contains_key("p") && found.contains_key("q"));
    }

    // -- provider credential chains ---------------------------------------------

    #[test]
    fn credential_chain_tables_parse_and_mark_the_chain_as_asked_for() {
        use crate::chains::Chain;
        use crate::tokensvc::{ClientAuth, Grant};
        let merged = chains_from(&[file(
            "[providers.amazon-bedrock]\naws = { profile = \"work\", region = \"us-west-2\" }\n\
             [providers.google-vertex]\ngcp = { project = \"p\", location = \"eu\" }\n\
             [providers.azure]\nazure = { tenant_id = \"t\", client_id = \"c\", client_secret_env = \"AZ\", cli = false }\n\
             [providers.idp]\nbase_url = \"https://gw.example/v1\"\n\
             oauth2 = { token_url = \"https://idp.example/token\", client_id = \"c\", client_secret_file = \"~/s\", scope = \"a b\", client_auth = \"basic\" }\n\
             [providers.jwt]\nbase_url = \"https://gw.example/v1\"\n\
             jwt_bearer = { token_url = \"https://idp.example/token\", issuer = \"me\", key_file = \"/k.pem\", algorithm = \"ES256\" }\n\
             [providers.watsonx]\nibm_iam = { client_secret_env = \"IBM\" }\n",
        )]);
        let Chain::Aws(aws) = &merged["amazon-bedrock"] else {
            panic!()
        };
        assert_eq!(
            (aws.profile.as_deref(), aws.region.as_deref(), aws.forced),
            (Some("work"), Some("us-west-2"), true)
        );
        let Chain::Gcp(gcp) = &merged["google-vertex"] else {
            panic!()
        };
        assert_eq!(
            (gcp.project.as_deref(), gcp.location.as_deref(), gcp.forced),
            (Some("p"), Some("eu"), true)
        );
        let Chain::Azure(az) = &merged["azure"] else {
            panic!()
        };
        assert_eq!(
            (az.client_secret_env.as_deref(), az.cli),
            (Some("AZ"), Some(false))
        );
        let Chain::Token(oauth) = &merged["idp"] else {
            panic!()
        };
        assert_eq!(
            (oauth.grant, oauth.client_auth),
            (Grant::ClientCredentials, Some(ClientAuth::Basic))
        );
        let Chain::Token(jwt) = &merged["jwt"] else {
            panic!()
        };
        assert_eq!(jwt.grant, Grant::JwtBearer);
        let Chain::Token(ibm) = &merged["watsonx"] else {
            panic!()
        };
        assert_eq!(ibm.grant, Grant::IbmApikey);
        assert_eq!(merged["idp"].kind(), "client_credentials");
        // No table, no chain.
        assert!(!merged.contains_key("openai"));
        assert_eq!(provider_chain("nobody-configured-this"), None);
    }

    #[test]
    fn credential_chain_mistakes_are_refused_at_load_without_echoing_secrets() {
        for (text, wants) in [
            ("[providers.p]\naws = { profile = \"a\" }\ngcp = {}", "alternatives"),
            ("[providers.p]\noauth2 = { token_url = \"https://x/t\", client_id = \"c\" }", "client_secret"),
            ("[providers.p]\noauth2 = { token_url = \"http://x/t\", client_id = \"c\", client_secret = \"s\" }", "https"),
            (
                "[providers.p]\noauth2 = { token_url = \"https://x/t\", client_id = \"c\", client_secret = \"s\", client_secret_env = \"E\" }",
                "alternatives",
            ),
            (
                "[providers.p]\noauth2 = { token_url = \"https://x/t\", client_id = \"c\", client_secret = \"s\", grant = \"jwt_bearer\" }",
                "grant",
            ),
            ("[providers.p]\njwt_bearer = { token_url = \"https://x/t\", issuer = \"i\" }", "key_file"),
            ("[providers.p]\naws = { region = \"a.b\" }", "region"),
            ("[providers.p]\ngcp = { project = \"a/b\" }", "project"),
            ("[providers.p]\nazure = { tenant_id = \"a/b\" }", "tenant_id"),
            ("[providers.p]\naws = { typo = 1 }", "unknown field"),
        ] {
            let err = parse(text).expect_err(text);
            assert!(err.contains(wants), "{text}: {err}");
            assert!(!err.contains("\"s\""), "{err}");
        }
        // A secret written in a table is not printed by the provider's Debug.
        let conf = conf("[providers.p]\nbase_url = \"https://x/v1\"\noauth2 = { token_url = \"https://x/t\", client_id = \"c\", client_secret = \"table-secret\" }\n");
        let printed = format!("{:?}", conf.providers["p"]);
        assert!(!printed.contains("table-secret"), "{printed}");
        assert!(printed.contains("client_credentials"), "{printed}");
    }

    #[test]
    fn a_later_chain_table_overlays_an_earlier_one_of_its_kind_and_replaces_another() {
        use crate::chains::Chain;
        let merged = chains_from(&[
            file("[providers.a]\naws = { profile = \"one\", service = \"bedrock\" }\n[providers.b]\naws = { profile = \"x\" }\n"),
            file("[providers.a]\naws = { region = \"us-west-2\" }\n[providers.b]\ngcp = { project = \"p\" }\n"),
        ]);
        let Chain::Aws(a) = &merged["a"] else {
            panic!()
        };
        assert_eq!(
            (
                a.profile.as_deref(),
                a.region.as_deref(),
                a.service.as_deref()
            ),
            (Some("one"), Some("us-west-2"), Some("bedrock"))
        );
        assert_eq!(merged["b"].kind(), "adc");
    }

    #[cfg(unix)]
    #[test]
    fn a_secret_in_a_chain_table_is_only_trusted_from_an_owner_only_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let text = "[providers.p]\nbase_url = \"https://x/v1\"\noauth2 = { token_url = \"https://x/t\", client_id = \"c\", client_secret = \"inline\" }\n[providers.q]\nbase_url = \"https://x/v1\"\noauth2 = { token_url = \"https://x/t\", client_id = \"c\", client_secret_env = \"ENV\" }\n";
        std::fs::write(&path, text).unwrap();
        let loaded = |path: &PathBuf| {
            vec![File {
                path: path.clone(),
                dir: dir.path().to_path_buf(),
                conf: conf(text),
            }]
        };
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let found = chains_from(&loaded(&path));
        assert!(
            !found.contains_key("p"),
            "an inline secret in a loose file is ignored"
        );
        assert!(found.contains_key("q"), "a variable's name is not a secret");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(chains_from(&loaded(&path)).contains_key("p"));
    }
}
