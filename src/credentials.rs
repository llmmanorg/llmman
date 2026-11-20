//! What `llmman serve` authenticates to a hosted provider with: an API key,
//! an OAuth login, a pasted token, an external source (a command, a file or
//! another CLI's login, read but never changed) or a cloud chain selection
//! ([`crate::chains`]).
//!
//! They live in one owner-only `credentials.json` beside `llmman.conf`; a
//! readable-by-others file is ignored with a warning, and writes go through
//! a temp file and rename. Nothing here prints a secret: [`Secret`] and every
//! type holding one implement `Debug` by hand.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};

/// An access token is refreshed this long before it expires, as litellm
/// (`TOKEN_EXPIRY_SKEW_SECONDS`) and opencode (`credential.expires <=
/// Date.now() + 60_000`) do, so one does not expire in flight.
pub const EXPIRY_SKEW_MS: u64 = 60_000;

/// How long a key command's output is reused. A `pass` or `op` lookup can
/// take a second and prompt; once per request would be unusable.
const COMMAND_CACHE: Duration = Duration::from_secs(300);

/// How long a key command may run.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(20);

/// Milliseconds since the epoch.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ---------------------------------------------------------------------------
// Secret
// ---------------------------------------------------------------------------

/// A string that must not reach a log: no `Display`, and `Debug` prints
/// nothing of it. [`Secret::expose`] is the one way out, and every call to
/// it is a place to check.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// One stored credential.
// `Chain` carries a whole token-service description; boxing it would put a
// `Box` in every pattern that names the variant, for a few of these ever alive.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey {
        key: String,
        /// The login that produced it, for the headers and host rules that
        /// go with it ([`crate::logins`]).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        flow: Option<String>,
    },
    OAuth(OAuth),
    Token {
        token: String,
        /// When it stops working, if known. Absent for a token that does
        /// not say.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        flow: Option<String>,
    },
    External {
        source: ExternalSource,
    },
    /// A selection of a cloud credential chain rather than a secret: an AWS
    /// profile, a Google credentials file, an Entra client, a token service
    /// with its client secret ([`crate::chains`]). A provider's chain
    /// resolves it ([`crate::providers::Provider::credential`]); the store
    /// only keeps it.
    Chain(crate::chains::Chain),
}

/// An OAuth login: the pair of tokens and what the flow keeps with them.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuth {
    /// Which login ([`crate::logins`]) made it, and so how to refresh it.
    pub flow: String,
    pub access: String,
    pub refresh: String,
    /// When `access` stops working, in milliseconds since the epoch; 0
    /// for one that does not expire (GitHub's device-flow token).
    pub expires_ms: u64,
    /// Flow-specific facts that are not secrets: `account_id`,
    /// `api_endpoint`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl OAuth {
    /// Whether `access` is good for a request made at `now`.
    pub fn fresh(&self, now: u64) -> bool {
        self.expires_ms == 0 || self.expires_ms > now.saturating_add(EXPIRY_SKEW_MS)
    }
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey { flow, .. } => f
                .debug_struct("ApiKey")
                .field("flow", flow)
                .finish_non_exhaustive(),
            Self::OAuth(o) => std::fmt::Debug::fmt(o, f),
            Self::Token {
                expires_ms, flow, ..
            } => f
                .debug_struct("Token")
                .field("expires_ms", expires_ms)
                .field("flow", flow)
                .finish_non_exhaustive(),
            Self::External { source } => {
                f.debug_struct("External").field("source", source).finish()
            }
            Self::Chain(chain) => f.debug_tuple("Chain").field(chain).finish(),
        }
    }
}

impl std::fmt::Debug for OAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuth")
            .field("flow", &self.flow)
            .field("expires_ms", &self.expires_ms)
            .field("metadata", &self.metadata.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

/// Where an external credential is read from, each time it is needed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExternalSource {
    /// A shell command whose first line of output is the key.
    Command { command: String },
    /// A file whose first line is the key; owner-only, like this store.
    File { path: String },
    /// Another CLI's own login file, which llmman reads and never writes
    /// ([`crate::logins::import`]).
    CliLogin {
        cli: CliLogin,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
}

/// A CLI whose login llmman can reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliLogin {
    Claude,
    Codex,
}

/// How a resolved credential is attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Wherever the provider's own scheme says ([`crate::authscheme`]).
    Key,
    /// `Authorization: Bearer`, whatever the provider's scheme: a login's
    /// token is not the provider's API key.
    Bearer,
}

/// A credential ready to spend: the secret, how to attach it, and the
/// extra headers that go with it.
#[derive(Clone)]
pub struct Resolved {
    pub secret: Secret,
    pub style: Style,
    /// Where it came from, as [`Status::kind`] spells it.
    pub kind: &'static str,
    /// The login behind it, if one is ([`crate::logins`]).
    pub flow: Option<String>,
    /// [`OAuth::metadata`], for a stored login.
    pub metadata: BTreeMap<String, String>,
    /// Sent with it: ChatGPT's account id, Anthropic's OAuth beta.
    pub headers: Vec<(String, String)>,
    /// Set for a credential that signs each request instead of travelling
    /// in a header (AWS SigV4): the request is signed once its body is
    /// final, by [`crate::cmd::serve`]'s send. [`Resolved::secret`] is
    /// empty then.
    pub signer: Option<std::sync::Arc<dyn RequestSigner>>,
}

/// Signs a finished request: its URL, headers and exact body bytes.
pub trait RequestSigner: Send + Sync {
    fn sign(&self, request: &mut reqwest::Request) -> anyhow::Result<()>;
}

impl std::fmt::Debug for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resolved")
            .field("kind", &self.kind)
            .field("flow", &self.flow)
            .field("style", &self.style)
            .field(
                "headers",
                &self.headers.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field("signed", &self.signer.is_some())
            .finish_non_exhaustive()
    }
}

impl Resolved {
    /// A plain key from `kind` (`request` for one a caller sent).
    pub fn from_key(secret: String, kind: &'static str) -> Self {
        Self::new(secret, kind, None)
    }

    fn new(secret: String, kind: &'static str, flow: Option<String>) -> Self {
        Self {
            secret: Secret::new(secret),
            style: Style::Key,
            kind,
            flow,
            metadata: BTreeMap::new(),
            headers: Vec::new(),
            signer: None,
        }
    }
}

/// What a credential is and whether it works, for `llmman providers` and
/// `GET /llmman/providers`: no secret, and nothing that took the network
/// to learn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Status {
    /// `env`, `config`, `api_key`, `oauth`, `token`, `command`, `file` or
    /// `cli_login`.
    pub kind: &'static str,
    /// `ready`; `refresh` (an OAuth access token that has lapsed and will
    /// be renewed by the next request); `expired` (cannot be renewed:
    /// log in again).
    pub state: &'static str,
    /// When the access token stops working, if it does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
}

impl Credential {
    /// See [`Status`].
    pub fn status(&self, now: u64) -> Status {
        let ready = |kind| Status {
            kind,
            state: "ready",
            expires_at_ms: None,
        };
        match self {
            Self::ApiKey { .. } => ready("api_key"),
            Self::OAuth(o) => Status {
                kind: "oauth",
                state: if o.fresh(now) { "ready" } else { "refresh" },
                expires_at_ms: (o.expires_ms != 0).then_some(o.expires_ms),
            },
            Self::Token { expires_ms, .. } => Status {
                kind: "token",
                state: if expires_ms.is_some_and(|e| e <= now) {
                    "expired"
                } else {
                    "ready"
                },
                expires_at_ms: *expires_ms,
            },
            Self::Chain(chain) => Status {
                kind: chain.kind(),
                state: "ready",
                expires_at_ms: None,
            },
            Self::External { source } => match source {
                ExternalSource::Command { .. } => ready("command"),
                ExternalSource::File { .. } => ready("file"),
                ExternalSource::CliLogin { cli, path } => {
                    match crate::logins::import::read(*cli, path.as_deref()) {
                        Ok(login) => Status {
                            kind: "cli_login",
                            state: if login.expires_ms.is_some_and(|e| e <= now) {
                                "expired"
                            } else {
                                "ready"
                            },
                            expires_at_ms: login.expires_ms,
                        },
                        Err(_) => Status {
                            kind: "cli_login",
                            state: "expired",
                            expires_at_ms: None,
                        },
                    }
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Errors and refreshing
// ---------------------------------------------------------------------------

/// A stored login that can no longer be used and has to be redone: the
/// refresh token was rejected, or a token ran out with nothing to renew
/// it. The daemon answers 401 rather than quietly spending some lesser
/// credential on the caller's behalf.
#[derive(Debug)]
pub struct ReloginRequired {
    pub provider: String,
    pub reason: String,
}

impl std::fmt::Display for ReloginRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the stored login for provider {:?} can no longer be used ({}); \
             run `llmman login --provider {}` again",
            self.provider, self.reason, self.provider
        )
    }
}

impl std::error::Error for ReloginRequired {}

/// Why a refresh failed.
#[derive(Debug)]
pub enum RefreshError {
    /// The server said no: the refresh token is spent, revoked or
    /// expired. Retrying cannot help.
    Rejected(String),
    /// The server could not be asked, or failed: try again later, and use
    /// the access token meanwhile if it still has life in it.
    Transient(String),
}

/// Exchanges a refresh token for a new pair. Implemented by
/// [`crate::logins::Live`]; a test supplies its own.
pub trait Refresher {
    /// The renewed credential. A server that does not rotate the refresh
    /// token keeps the old one in the result.
    fn refresh(&self, oauth: &OAuth) -> Result<OAuth, RefreshError>;
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// Version of the `credentials.json` layout.
const FILE_VERSION: u32 = 1;

#[derive(Default, Serialize, Deserialize)]
struct FileData {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    providers: BTreeMap<String, Credential>,
}

/// The credentials file.
#[derive(Clone, Debug)]
pub struct Store {
    path: PathBuf,
}

impl Store {
    pub fn at(path: PathBuf) -> Self {
        Self { path }
    }

    /// The per-user store, or `None` when there is no home directory.
    pub fn user() -> Option<Self> {
        crate::config::credentials_path().map(Self::at)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file's contents. Absent is empty; one others can read, or that
    /// does not parse, is empty too — with a warning, once — so that a
    /// daemon never spends a credential it was not meant to have. A write
    /// refuses both instead ([`Store::read_strict`]).
    fn load(&self) -> FileData {
        match self.read_strict() {
            Ok(data) => data,
            Err(e) => {
                warn_once(&self.path, &format!("{e:#}"));
                FileData::default()
            }
        }
    }

    fn read_strict(&self) -> anyhow::Result<FileData> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(FileData::default()),
            Err(e) => return Err(e).with_context(|| format!("read {}", self.path.display())),
        };
        crate::config::owner_readable_only(&self.path)
            .map_err(|e| anyhow!("ignoring the credentials in {}: {e}", self.path.display()))?;
        // The serde message can quote the input; say only where.
        serde_json::from_str(&text).map_err(|e| {
            anyhow!(
                "ignoring the credentials in {}: not valid JSON at line {}, column {}",
                self.path.display(),
                e.line(),
                e.column()
            )
        })
    }

    /// The credential stored for provider `id`.
    pub fn get(&self, id: &str) -> Option<Credential> {
        self.load().providers.remove(id)
    }

    /// Every stored credential, by provider id.
    pub fn list(&self) -> BTreeMap<String, Credential> {
        self.load().providers
    }

    /// Stores `credential` for `id`, replacing any.
    pub fn put(&self, id: &str, credential: Credential) -> anyhow::Result<()> {
        let _lock = self.lock()?;
        self.put_locked(id, credential)
    }

    /// Removes `id`'s credential; whether there was one.
    pub fn remove(&self, id: &str) -> anyhow::Result<bool> {
        let _lock = self.lock()?;
        let mut data = self.read_strict()?;
        if data.providers.remove(id).is_none() {
            return Ok(false);
        }
        self.save(&mut data)?;
        Ok(true)
    }

    fn put_locked(&self, id: &str, credential: Credential) -> anyhow::Result<()> {
        let mut data = self.read_strict()?;
        data.providers.insert(id.to_string(), credential);
        self.save(&mut data)
    }

    fn save(&self, data: &mut FileData) -> anyhow::Result<()> {
        data.version = FILE_VERSION;
        if let Some(dir) = self.path.parent() {
            create_private_dir(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(data).context("serialize credentials")?;
        crate::fsutil::write_private(&self.path, &bytes)
            .with_context(|| format!("write {}", self.path.display()))
    }

    /// Takes the store's lock: this process's, and — on Unix — every other
    /// llmman's, so two refreshes of one rotating token cannot interleave
    /// and a read-modify-write cannot lose the other's write.
    fn lock(&self) -> anyhow::Result<StoreLock> {
        // One leaked mutex per store path: a guard has to borrow from
        // something that outlives it, and there are a handful of paths.
        static LOCKS: Mutex<Option<HashMap<PathBuf, &'static Mutex<()>>>> = Mutex::new(None);
        let process: &'static Mutex<()> = LOCKS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .entry(self.path.clone())
            .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))));
        let mut lock_path = self.path.clone().into_os_string();
        lock_path.push(".lock");
        StoreLock::acquire(process, Path::new(&lock_path))
    }

    /// The credential for `id` ready to spend, renewing it first when it
    /// needs it; `Ok(None)` when none is stored.
    ///
    /// One refresh at a time: concurrent requests all find the access
    /// token lapsed, one takes the lock and renews it, and the rest —
    /// which read the file again once they hold the lock — find it fresh.
    /// The same holds across processes, which is what stops a CLI and the
    /// daemon from each spending the one refresh token that rotates.
    pub fn resolve(
        &self,
        id: &str,
        refresher: &dyn Refresher,
        now: u64,
    ) -> anyhow::Result<Option<Resolved>> {
        let Some(credential) = self.get(id) else {
            return Ok(None);
        };
        let relogin = |reason: &str| ReloginRequired {
            provider: id.to_string(),
            reason: reason.to_string(),
        };
        match credential {
            Credential::ApiKey { key, flow } => Ok(Some(Resolved::new(key, "api_key", flow))),
            Credential::Token {
                token,
                expires_ms,
                flow,
            } => {
                if expires_ms.is_some_and(|e| e <= now) {
                    return Err(relogin("the token has expired").into());
                }
                Ok(Some(Resolved::new(token, "token", flow)))
            }
            // Resolved by the provider's chain, which knows the endpoint and the
            // environment it runs in; the store has no secret to hand over.
            Credential::Chain(_) => Ok(None),
            Credential::External { source } => {
                let (secret, kind) = read_external(&source)?;
                let mut resolved = Resolved::new(secret.expose().to_string(), kind, None);
                if let ExternalSource::CliLogin { cli, path } = &source {
                    let login = crate::logins::import::read(*cli, path.as_deref())?;
                    resolved.flow = Some(login.flow.to_string());
                    resolved.metadata = login.metadata;
                }
                Ok(Some(resolved))
            }
            Credential::OAuth(oauth) if oauth.fresh(now) => Ok(Some(oauth_resolved(oauth))),
            Credential::OAuth(_) => {
                let _lock = self.lock()?;
                // Another request or process may have renewed it while this
                // one waited.
                let Some(Credential::OAuth(oauth)) = self.read_strict()?.providers.remove(id)
                else {
                    return Err(relogin("the login was removed").into());
                };
                if oauth.fresh(now) {
                    return Ok(Some(oauth_resolved(oauth)));
                }
                match refresher.refresh(&oauth) {
                    Ok(renewed) => {
                        if let Err(e) = self.put_locked(id, Credential::OAuth(renewed.clone())) {
                            // The new tokens work for this request; what is
                            // lost is the rotation, which the next refresh
                            // will find out.
                            eprintln!(
                                "[llmman] warning: could not save the renewed login for {id}: {e:#}"
                            );
                        }
                        Ok(Some(oauth_resolved(renewed)))
                    }
                    Err(RefreshError::Rejected(why)) => Err(relogin(&why).into()),
                    Err(RefreshError::Transient(why)) if oauth.expires_ms > now => {
                        eprintln!(
                            "[llmman] warning: could not renew the login for {id} ({why}); \
                             using the current token until it expires"
                        );
                        Ok(Some(oauth_resolved(oauth)))
                    }
                    Err(RefreshError::Transient(why)) => Err(anyhow!(
                        "could not renew the login for provider {id:?}: {why}"
                    )),
                }
            }
        }
    }
}

fn oauth_resolved(oauth: OAuth) -> Resolved {
    let mut resolved = Resolved::new(oauth.access, "oauth", Some(oauth.flow));
    resolved.metadata = oauth.metadata;
    resolved
}

/// Warns that a store was ignored, once per path.
fn warn_once(path: &Path, message: &str) {
    static WARNED: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);
    let first = WARNED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashSet::new)
        .insert(path.to_path_buf());
    if first {
        eprintln!("[llmman] warning: {message}");
    }
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

/// Holds [`Store::lock`].
struct StoreLock {
    _process: std::sync::MutexGuard<'static, ()>,
    #[cfg(unix)]
    file: std::fs::File,
}

impl StoreLock {
    fn acquire(process: &'static Mutex<()>, lock_path: &Path) -> anyhow::Result<Self> {
        let guard = process.lock().unwrap_or_else(PoisonError::into_inner);
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::OpenOptionsExt;
            if let Some(dir) = lock_path.parent() {
                create_private_dir(dir)?;
            }
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .mode(0o600)
                .open(lock_path)
                .with_context(|| format!("open {}", lock_path.display()))?;
            // SAFETY: `flock` on a descriptor this function owns.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("lock {}", lock_path.display()));
            }
            Ok(Self {
                _process: guard,
                file,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = lock_path;
            Ok(Self { _process: guard })
        }
    }
}

#[cfg(unix)]
impl Drop for StoreLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        // SAFETY: as in `acquire`; closing the file would release it too.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

// ---------------------------------------------------------------------------
// External sources
// ---------------------------------------------------------------------------

/// The secret an [`ExternalSource`] yields now, and the kind it is
/// reported as ([`Status::kind`]).
///
/// A file must be owner-only, as the store is. A command's output is
/// reused for [`COMMAND_CACHE`]; its stderr is discarded, since a password
/// manager is as likely to print a prompt as a hint.
pub fn read_external(source: &ExternalSource) -> anyhow::Result<(Secret, &'static str)> {
    match source {
        ExternalSource::Command { command } => Ok((run_command(command)?, "command")),
        ExternalSource::File { path } => {
            let path = expand_home(path);
            crate::config::owner_readable_only(&path)
                .map_err(|e| anyhow!("refusing to read the key file {}: {e}", path.display()))?;
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read the key file {}", path.display()))?;
            Ok((first_line(&text, &path.display().to_string())?, "file"))
        }
        ExternalSource::CliLogin { cli, path } => {
            let login = crate::logins::import::read(*cli, path.as_deref())?;
            if login.expires_ms.is_some_and(|e| e <= now_ms()) {
                return Err(ReloginRequired {
                    provider: login.provider.to_string(),
                    reason: format!(
                        "{}'s own login has expired; run `{}` to refresh it, or log in again",
                        login.cli_name, login.cli_name
                    ),
                }
                .into());
            }
            Ok((Secret::new(login.access), "cli_login"))
        }
    }
}

/// `~/` in a path from `llmman.conf`.
pub fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => crate::config::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| PathBuf::from(path)),
        None => PathBuf::from(path),
    }
}

fn first_line(text: &str, from: &str) -> anyhow::Result<Secret> {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| Secret::new(l.to_string()))
        .ok_or_else(|| anyhow!("{from} is empty"))
}

fn run_command(command: &str) -> anyhow::Result<Secret> {
    static CACHE: Mutex<Option<HashMap<String, (Instant, Secret)>>> = Mutex::new(None);
    if let Some((at, secret)) = CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .get(command)
    {
        if at.elapsed() < COMMAND_CACHE {
            return Ok(secret.clone());
        }
    }
    let secret = run_command_uncached(command, COMMAND_TIMEOUT)?;
    CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .insert(command.to_string(), (Instant::now(), secret.clone()));
    Ok(secret)
}

fn run_command_uncached(command: &str, timeout: Duration) -> anyhow::Result<Secret> {
    use std::process::Command;

    #[cfg(unix)]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    #[cfg(not(unix))]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };
    let out = run_capture(&mut cmd, "the key command", timeout)?;
    first_line(&String::from_utf8_lossy(&out), "the key command's output")
}

/// Runs `cmd` with no stdin and its stderr discarded, and returns up to 64
/// KiB of its stdout once it exits successfully. `what` names it in the
/// errors, which never carry its output.
pub(crate) fn run_capture(
    cmd: &mut std::process::Command,
    what: &str,
    timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    use std::process::Stdio;

    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("run {what}"))?;
    let mut stdout = child.stdout.take().expect("stdout is piped");
    // Read on a thread: a full pipe would otherwise stall the child past
    // the timeout, and a helper holding the pipe open must not either.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = stdout.by_ref().take(64 * 1024).read_to_end(&mut out);
        let _ = tx.send(out);
    });
    let start = Instant::now();
    let late = || anyhow::anyhow!("{what} did not finish within {}s", timeout.as_secs());
    let status = crate::container::wait_with_timeout(&mut child, timeout)
        .with_context(|| format!("wait for {what}"))?
        .ok_or_else(late)?;
    let out = rx
        .recv_timeout(timeout.saturating_sub(start.elapsed()))
        .map_err(|_| late())?;
    anyhow::ensure!(status.success(), "{what} failed ({status})");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = Store::at(dir.path().join("cfg").join("credentials.json"));
        (dir, store)
    }

    fn oauth(access: &str, refresh: &str, expires_ms: u64) -> OAuth {
        OAuth {
            flow: "test".into(),
            access: access.into(),
            refresh: refresh.into(),
            expires_ms,
            metadata: BTreeMap::from([("account_id".into(), "acct".into())]),
        }
    }

    /// Hands out `access-N` / `refresh-N`, counting calls, optionally slowly.
    struct Counting {
        calls: AtomicUsize,
        delay: Duration,
    }
    impl Counting {
        fn new(delay: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                delay,
            }
        }
    }
    impl Refresher for Counting {
        fn refresh(&self, old: &OAuth) -> Result<OAuth, RefreshError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            std::thread::sleep(self.delay);
            Ok(OAuth {
                access: format!("access-{n}"),
                refresh: format!("refresh-{n}"),
                expires_ms: now_ms() + 3_600_000,
                ..old.clone()
            })
        }
    }

    struct Failing(fn() -> RefreshError);
    impl Refresher for Failing {
        fn refresh(&self, _: &OAuth) -> Result<OAuth, RefreshError> {
            Err((self.0)())
        }
    }

    #[test]
    fn credentials_round_trip_and_never_print_a_secret() {
        let (_dir, store) = store();
        store
            .put(
                "a",
                Credential::ApiKey {
                    key: "sk-very-secret".into(),
                    flow: None,
                },
            )
            .unwrap();
        store
            .put("b", Credential::OAuth(oauth("acc-secret", "ref-secret", 5)))
            .unwrap();
        store
            .put(
                "c",
                Credential::Token {
                    token: "tok-secret".into(),
                    expires_ms: Some(9),
                    flow: Some("f".into()),
                },
            )
            .unwrap();
        store
            .put(
                "d",
                Credential::External {
                    source: ExternalSource::Command {
                        command: "pass show x".into(),
                    },
                },
            )
            .unwrap();

        let all = store.list();
        assert_eq!(all.len(), 4);
        assert_eq!(
            all["b"],
            Credential::OAuth(oauth("acc-secret", "ref-secret", 5))
        );
        for credential in all.values() {
            let printed = format!("{credential:?}");
            for secret in ["sk-very-secret", "acc-secret", "ref-secret", "tok-secret"] {
                assert!(!printed.contains(secret), "{printed}");
            }
        }
        assert!(store.remove("a").unwrap());
        assert!(!store.remove("a").unwrap());
        assert_eq!(store.get("a"), None);
        assert_eq!(store.list().len(), 3);
    }

    #[cfg(unix)]
    #[test]
    fn the_store_is_private_and_a_loose_one_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, store) = store();
        let key = || Credential::ApiKey {
            key: "k".into(),
            flow: None,
        };
        store.put("a", key()).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(store.path()), 0o600);
        assert_eq!(mode(store.path().parent().unwrap()), 0o700);

        // A file others can read is not read, and not written over.
        std::fs::set_permissions(store.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(store.get("a"), None);
        assert!(store.list().is_empty());
        let err = store.put("b", key()).unwrap_err();
        assert!(format!("{err:#}").contains("chmod 600"), "{err:#}");
        assert!(store
            .resolve("a", &Counting::new(Duration::ZERO), 0)
            .unwrap()
            .is_none());

        std::fs::set_permissions(store.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(store.get("a").is_some());
    }

    #[test]
    fn a_corrupt_store_is_ignored_for_reads_and_kept_by_writes() {
        let (_dir, store) = store();
        store
            .put(
                "a",
                Credential::ApiKey {
                    key: "k".into(),
                    flow: None,
                },
            )
            .unwrap();
        std::fs::write(store.path(), "{ not json sk-leaky").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(
            store.path(),
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .unwrap();
        assert_eq!(store.get("a"), None);
        let err = store
            .put(
                "b",
                Credential::ApiKey {
                    key: "k".into(),
                    flow: None,
                },
            )
            .unwrap_err();
        assert!(!format!("{err:#}").contains("sk-leaky"), "{err:#}");
        assert_eq!(
            std::fs::read_to_string(store.path()).unwrap(),
            "{ not json sk-leaky"
        );
    }

    #[test]
    fn writes_leave_no_temp_files_behind() {
        let (_dir, store) = store();
        for i in 0..5 {
            store
                .put(
                    &format!("p{i}"),
                    Credential::ApiKey {
                        key: "k".into(),
                        flow: None,
                    },
                )
                .unwrap();
        }
        let names: Vec<String> = std::fs::read_dir(store.path().parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.ends_with(".tmp")), "{names:?}");
    }

    #[test]
    fn a_fresh_oauth_token_is_used_as_is() {
        let (_dir, store) = store();
        let now = 1_000_000;
        store
            .put(
                "p",
                Credential::OAuth(oauth("acc", "ref", now + EXPIRY_SKEW_MS + 1)),
            )
            .unwrap();
        let r = Counting::new(Duration::ZERO);
        let got = store.resolve("p", &r, now).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "acc");
        assert_eq!(got.kind, "oauth");
        assert_eq!(got.flow.as_deref(), Some("test"));
        assert_eq!(got.metadata["account_id"], "acct");
        assert_eq!(r.calls.load(Ordering::SeqCst), 0);
        assert!(store.resolve("absent", &r, now).unwrap().is_none());
    }

    #[test]
    fn a_token_inside_the_skew_is_refreshed_and_the_rotation_persisted() {
        let (_dir, store) = store();
        let now = 1_000_000;
        // Not yet expired, but within the skew.
        store
            .put(
                "p",
                Credential::OAuth(oauth("old", "old-refresh", now + EXPIRY_SKEW_MS)),
            )
            .unwrap();
        let r = Counting::new(Duration::ZERO);
        let got = store.resolve("p", &r, now).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "access-1");
        let Some(Credential::OAuth(saved)) = store.get("p") else {
            panic!("an OAuth credential is stored");
        };
        assert_eq!(
            saved.refresh, "refresh-1",
            "the rotated refresh token is kept"
        );
        assert_eq!(saved.metadata["account_id"], "acct");
        // And the next request does not refresh again.
        store.resolve("p", &r, now + 1).unwrap().unwrap();
        assert_eq!(r.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_never_expiring_token_is_never_refreshed() {
        let (_dir, store) = store();
        store
            .put("p", Credential::OAuth(oauth("gh", "gh", 0)))
            .unwrap();
        let r = Counting::new(Duration::ZERO);
        let got = store.resolve("p", &r, u64::MAX / 2).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "gh");
        assert_eq!(r.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            store.get("p").unwrap().status(5),
            Status {
                kind: "oauth",
                state: "ready",
                expires_at_ms: None
            }
        );
    }

    /// Many requests find the token lapsed at once: one refresh, because
    /// the rest read the file again once they hold the lock — the refresh
    /// token rotates, so a second refresh with the old one would fail.
    #[test]
    fn concurrent_requests_share_one_refresh() {
        let (_dir, store) = store();
        let now = now_ms();
        store
            .put(
                "p",
                Credential::OAuth(oauth("old", "r0", now.saturating_sub(1))),
            )
            .unwrap();
        let refresher = Arc::new(Counting::new(Duration::from_millis(100)));
        let handles: Vec<_> = (0..12)
            .map(|_| {
                let store = store.clone();
                let refresher = refresher.clone();
                std::thread::spawn(move || {
                    store
                        .resolve("p", &*refresher, now_ms())
                        .unwrap()
                        .unwrap()
                        .secret
                        .expose()
                        .to_string()
                })
            })
            .collect();
        let seen: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
        assert!(seen.iter().all(|s| s == "access-1"), "{seen:?}");
    }

    #[test]
    fn a_rejected_refresh_asks_for_a_new_login_and_keeps_the_old_one() {
        let (_dir, store) = store();
        store
            .put("p", Credential::OAuth(oauth("old", "r0", 1)))
            .unwrap();
        let rejected = Failing(|| RefreshError::Rejected("refresh_token_reused".into()));
        let err = store.resolve("p", &rejected, now_ms()).unwrap_err();
        let relogin = err.downcast_ref::<ReloginRequired>().expect("relogin");
        assert_eq!(relogin.provider, "p");
        assert!(
            err.to_string().contains("llmman login --provider p"),
            "{err}"
        );
        // Nothing was thrown away: a later, better refresher can still try.
        assert!(store.get("p").is_some());
    }

    #[test]
    fn a_transient_failure_uses_a_still_valid_token_and_otherwise_fails() {
        let (_dir, store) = store();
        let now = 1_000_000;
        let down = Failing(|| RefreshError::Transient("connection refused".into()));
        // Inside the skew but not expired: keep going.
        store
            .put("p", Credential::OAuth(oauth("still-ok", "r", now + 10)))
            .unwrap();
        let got = store.resolve("p", &down, now).unwrap().unwrap();
        assert_eq!(got.secret.expose(), "still-ok");
        // Expired: an error, but not a relogin — the login is fine.
        store
            .put("q", Credential::OAuth(oauth("dead", "r", now - 1)))
            .unwrap();
        let err = store.resolve("q", &down, now).unwrap_err();
        assert!(err.downcast_ref::<ReloginRequired>().is_none(), "{err}");
        assert!(err.to_string().contains("connection refused"), "{err}");
    }

    #[test]
    fn a_token_that_expired_needs_a_new_login() {
        let (_dir, store) = store();
        store
            .put(
                "p",
                Credential::Token {
                    token: "t".into(),
                    expires_ms: Some(10),
                    flow: None,
                },
            )
            .unwrap();
        let r = Counting::new(Duration::ZERO);
        assert!(store
            .resolve("p", &r, 11)
            .unwrap_err()
            .downcast_ref::<ReloginRequired>()
            .is_some());
        assert_eq!(store.resolve("p", &r, 9).unwrap().unwrap().kind, "token");
        assert_eq!(store.get("p").unwrap().status(11).state, "expired");
    }

    #[cfg(unix)]
    #[test]
    fn key_commands_and_files_resolve_and_report_their_kind() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, store) = store();
        store
            .put(
                "cmd",
                Credential::External {
                    source: ExternalSource::Command {
                        command: "printf '\\n  sk-from-command  \\nsecond line\\n'".into(),
                    },
                },
            )
            .unwrap();
        let got = store
            .resolve("cmd", &Counting::new(Duration::ZERO), 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            (got.secret.expose(), got.kind),
            ("sk-from-command", "command")
        );

        let key_file = dir.path().join("key");
        std::fs::write(&key_file, "sk-from-file\n").unwrap();
        std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o644)).unwrap();
        let source = ExternalSource::File {
            path: key_file.to_string_lossy().into_owned(),
        };
        let err = read_external(&source).unwrap_err();
        assert!(format!("{err:#}").contains("chmod 600"), "{err:#}");
        std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        let (secret, kind) = read_external(&source).unwrap();
        assert_eq!((secret.expose(), kind), ("sk-from-file", "file"));
    }

    #[cfg(unix)]
    #[test]
    fn a_failing_or_slow_key_command_is_an_error_without_its_output() {
        let err = run_command_uncached("echo sk-leak; exit 3", Duration::from_secs(5)).unwrap_err();
        assert!(!format!("{err:#}").contains("sk-leak"), "{err:#}");
        let err = run_command_uncached("sleep 5", Duration::from_millis(100)).unwrap_err();
        assert!(format!("{err:#}").contains("did not finish"), "{err:#}");
        assert!(run_command_uncached("true", Duration::from_secs(5)).is_err());
    }
}
