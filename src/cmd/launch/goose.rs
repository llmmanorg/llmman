//! `llmman launch goose`.
//!
//! The CLI. The desktop app is [`super::goose_desktop`].

use std::path::{Path, PathBuf};

use super::goose_desktop::is_electron_bundle;
use super::{exec_with_env, find_on_path_unless, server};

/// goose: configured entirely through the environment, which goose reads
/// in preference to its own `config.yaml` — so unlike hermes and qwen
/// nothing is written to disk and no key is persisted. `OPENAI_HOST` is
/// the bare origin, not a `/v1` base URL: goose joins it with
/// `OPENAI_BASE_PATH` itself. Verified against goose 1.50.0 with no
/// config file and no `goose configure`.
pub(super) fn launch_goose(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_goose().ok_or_else(|| anyhow::anyhow!("goose is not installed"))?;
    let host = server();
    exec_with_env(&bin, extra_args, &goose_env(model, api_key, &host))
}

/// Split out so a test can assert what goose is handed: [`super::exec_with_env`]
/// never returns, so calling [`launch_goose`] would take the test runner
/// with it.
pub(super) fn goose_env<'a>(
    model: &'a str,
    api_key: &'a str,
    host: &'a str,
) -> Vec<(&'a str, &'a str)> {
    let mut env = vec![
        ("GOOSE_PROVIDER", "openai"),
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_HOST", host),
        ("OPENAI_BASE_PATH", "v1/chat/completions"),
    ];
    // Absent, not empty: goose reads "" as a model actually named "".
    if !model.is_empty() {
        env.push(("GOOSE_MODEL", model));
    }
    env
}

/// The CLI, never the desktop app standing in for it: the unpacked zip's
/// executable answers to this name too — exactly on Linux, by case
/// elsewhere — and handing the GUI `goose run`'s arguments is
/// block/goose#4079 the other way round.
pub(super) fn find_goose() -> Option<PathBuf> {
    find_on_path_unless("goose", is_electron_bundle).or_else(|| goose_fallback(&dirs::home_dir()?))
}

/// goose's own installer target: `download_cli.sh` writes to
/// `$GOOSE_BIN_DIR` without putting it on `PATH`. Its default is
/// `$USERPROFILE/goose` on Windows (what `dirs::home_dir` returns there)
/// and `~/.local/bin` elsewhere; Windows probes both, since goose's
/// install instructions and this repo's CI pass the latter (v1.50.0).
pub(super) fn goose_fallback(home: &Path) -> Option<PathBuf> {
    let bin = if cfg!(windows) { "goose.exe" } else { "goose" };
    let mut candidates = Vec::new();
    if cfg!(windows) {
        candidates.push(home.join("goose").join(bin));
    }
    candidates.push(home.join(".local").join("bin").join(bin));
    // is_file, not exists: a directory of that name would be reported as
    // installed and then fail to spawn. Not the desktop app either: a
    // zip can be unpacked into the installer's own directory, and these
    // names match its executable — exactly on Linux, by case elsewhere.
    candidates
        .into_iter()
        .find(|p| p.is_file() && !is_electron_bundle(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The only configuration goose gets: a wrong or missing one sends
    /// the session to api.openai.com instead of the daemon. `OPENAI_HOST`
    /// is the bare origin — goose appends `OPENAI_BASE_PATH` itself, so a
    /// `/v1` here would request `/v1/v1/chat/completions`.
    #[test]
    fn goose_env_points_at_the_daemon_and_carries_the_key() {
        let env = goose_env("m", "k", "http://127.0.0.1:17434");
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
        assert_eq!(get("GOOSE_PROVIDER"), Some("openai"));
        assert_eq!(get("GOOSE_MODEL"), Some("m"));
        assert_eq!(get("OPENAI_API_KEY"), Some("k"));
        assert_eq!(get("OPENAI_HOST"), Some("http://127.0.0.1:17434"));
        assert_eq!(get("OPENAI_BASE_PATH"), Some("v1/chat/completions"));

        let env = goose_env("", "k", "http://127.0.0.1:17434");
        assert!(!env.iter().any(|(n, _)| *n == "GOOSE_MODEL"));
    }

    /// `download_cli.sh`'s target is off `PATH` on a fresh shell, so this
    /// fallback is the one that fires for most installs — at every
    /// directory that installer writes to, Windows included.
    #[test]
    fn goose_fallback_finds_the_installers_target() {
        let name = if cfg!(windows) { "goose.exe" } else { "goose" };
        let dirs: &[&[&str]] = if cfg!(windows) {
            &[&["goose"], &[".local", "bin"]]
        } else {
            &[&[".local", "bin"]]
        };
        for (i, parts) in dirs.iter().enumerate() {
            let home = std::env::temp_dir().join(format!(
                "llmman-goose-{}-{}-{i}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let bin = parts.iter().fold(home.clone(), |p, part| p.join(part));
            std::fs::create_dir_all(&bin).unwrap();
            assert_eq!(goose_fallback(&home), None);
            let goose = bin.join(name);
            // A directory of that name is not the binary: returning it
            // would report goose as installed and then fail to spawn.
            std::fs::create_dir(&goose).unwrap();
            assert_eq!(goose_fallback(&home), None);
            std::fs::remove_dir(&goose).unwrap();
            std::fs::write(&goose, "").unwrap();
            assert_eq!(goose_fallback(&home), Some(goose));
            assert_eq!(goose_fallback(&home.join("nowhere")), None);
            let _ = std::fs::remove_dir_all(&home);
        }
    }
}
