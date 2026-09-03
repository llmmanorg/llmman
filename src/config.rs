//! `llmman.conf` — one config file, read once, shared by every consumer.
//!
//! Two locations, later overriding earlier:
//!
//!   1. `/etc/llmman/llmman.conf`        system-wide
//!   2. `~/.config/llmman/llmman.conf`   per-user
//!
//! The same paths on every platform, with no llmman-specific variable
//! to move them: one documented answer to "where does this go". (`~` is
//! `$HOME` as usual.)
//!
//! ```toml
//! [aliases]                            # crate::shortnames
//! gemma4 = "docker.io/ai/gemma4"
//!
//! [providers.openrouter]               # crate::providers
//! api_key = "sk-or-..."
//!
//! [verify]                             # crate::verify
//! default = "off"
//!
//! [[verify.trust]]
//! pattern = "docker.io/myorg/**"
//! keys    = ["keys/myorg.pub"]         # relative to this file's directory
//! mode    = "enforce"
//! ```
//!
//! Parsing happens once, in [`files`]; what a *failed* parse means is
//! left to each consumer, because it differs. [`crate::verify`] refuses
//! to run, since a trust policy it cannot read must not quietly become
//! `off`; aliases and keys degrade to none. The error is reported once,
//! centrally, so one typo is not announced three times.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderConf {
    #[serde(default)]
    api_key: Option<String>,
}

/// Hand-written so the key cannot reach a log through the derived
/// `Debug` of the public [`Conf`] and [`File`]. Same reasoning as
/// `RemoteTarget` in `cmd::serve`.
impl std::fmt::Debug for ProviderConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConf")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// The `[verify]` section. [`crate::verify`] owns what these strings
/// mean; this module only spells out their shape.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct VerifyConf {
    /// Mode for references no rule matches.
    #[serde(default)]
    pub default: Option<String>,
    /// `[[verify.trust]]` entries, in the order they appear.
    #[serde(default)]
    pub trust: Vec<TrustConf>,
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
}

impl Conf {
    /// Every entry that sets `api_key` at all, trimmed, by provider id.
    ///
    /// A blank value is kept here rather than dropped, so a user file can
    /// blank out a key `/etc` set — otherwise there is no way to opt out
    /// of a system-wide credential. [`load_provider_keys`] drops blanks
    /// after merging, since sending `Authorization: Bearer ` upstream
    /// would turn a clear "no API key" into someone else's 401.
    fn provider_keys(&self) -> HashMap<String, String> {
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
fn search_paths() -> Vec<PathBuf> {
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
/// `~` is `$HOME` as usual.
fn user_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("llmman"))
}

/// Where a user's own config belongs, for a message that has to name
/// it. `None` only when there is no home directory.
fn user_path() -> Option<PathBuf> {
    user_dir().map(|d| d.join(FILE))
}

/// [`user_path`] for printing, falling back to the bare file name.
pub fn user_path_display() -> String {
    user_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| FILE.to_string())
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

    let mut files = Vec::new();
    for path in search_paths() {
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
    toml::from_str(text).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Provider API keys
// ---------------------------------------------------------------------------

/// The key `llmman.conf` configures for `provider_id`, or `None`. Keyed
/// by the id `llmman providers` prints, not by the environment variable,
/// which is models.dev's naming rather than llmman's.
///
/// Borrowed from a process-lifetime cache, so a daemon serving requests
/// is not stat'ing `/etc` per token.
pub fn provider_api_key(provider_id: &str) -> Option<&'static str> {
    static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
    CACHE
        .get_or_init(load_provider_keys)
        .get(provider_id)
        .map(String::as_str)
}

fn load_provider_keys() -> HashMap<String, String> {
    let mut accepted = Vec::new();
    for file in files().unwrap_or_default() {
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

/// Refuses a file that group or other can read, the way `ssh` refuses a
/// loose private key.
///
/// Gates the keys, not the read: `/etc/llmman/llmman.conf` is
/// legitimately world-readable for the aliases and trust policy it also
/// carries, so a loose file still supplies everything but the key.
#[cfg(unix)]
fn owner_readable_only(path: &Path) -> Result<(), String> {
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
fn owner_readable_only(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conf(text: &str) -> Conf {
        parse(text).expect("valid conf")
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

            [[verify.trust]]
            pattern = "docker.io/myorg/**"
            keys    = ["keys/myorg.pub"]
            mode    = "enforce"
            "#,
        );
        assert_eq!(
            c.aliases.get("gemma4").map(String::as_str),
            Some("docker.io/ai/gemma4")
        );
        assert_eq!(
            c.provider_keys().get("openrouter").map(String::as_str),
            Some("sk-or-abc")
        );
        assert_eq!(c.verify.default.as_deref(), Some("warn"));
        assert_eq!(c.verify.trust.len(), 1);
        assert_eq!(c.verify.trust[0].mode.as_deref(), Some("enforce"));
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
        assert!(parse("[[verify.trust]]\npattern = \"a/b\"\nkyes = []").is_err());
        assert!(parse("api_key = \"x\"").is_err());
    }

    /// A mangled file is reported, not treated as empty.
    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse("[providers.openai").is_err());
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
}
