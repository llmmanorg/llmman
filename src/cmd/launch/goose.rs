//! `llmman launch goose`.
//!
//! The CLI. The desktop app is [`super::goose_desktop`].

use super::models::{write_transient, ModelEntry, ModelList};
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::goose_desktop::is_electron_bundle;
use super::{env_dir, exec_with_env, find_on_path_unless, server, xdg_dir};

/// The variable holding the provider's key: a name in the provider file,
/// never the key.
const GOOSE_KEY_ENV: &str = "LLMMAN_API_KEY";

/// goose: configured through the environment (which outranks `config.yaml`) plus one provider file
/// for the model list: goose takes a list only from a provider file declaring `models` with
/// `dynamic_models: false`, since `/v1/models` holds stored models alone. The file holds no key and
/// is removed when the launch ends. `OPENAI_HOST` is the bare origin, not a `/v1` URL: goose joins
/// it with `OPENAI_BASE_PATH`. Verified against goose 1.60.2.
pub(super) fn launch_goose(
    model: &str,
    api_key: &str,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_goose().ok_or_else(|| anyhow::anyhow!("goose is not installed"))?;
    let host = server();
    let env = goose_session_env(model, api_key, models, &host);
    exec_with_env(&bin, extra_args, &env)
}

/// The provider goose lists the models under. goose registers every file in
/// its directory by this name, so each launch takes one of its own: two
/// launches at different daemons would otherwise replace each other.
fn goose_provider_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(|| format!("llmman-{}", std::process::id()))
}

/// What goose is handed: the plain `openai` provider for a launch with nothing to switch to, else a
/// provider file holding every model (removed when the launch ends). A file that cannot be written
/// does not stop the launch: it says so and starts on its one model.
pub(super) fn goose_session_env<'a>(
    model: &'a str,
    api_key: &'a str,
    models: &ModelList,
    host: &'a str,
) -> Vec<(&'a str, &'a str)> {
    if !models.offers_choice() {
        return goose_env(model, api_key, host);
    }
    let written = goose_config_dir().and_then(|dir| write_provider_file(&dir, models, host));
    env_after_writing(written, model, api_key, host)
}

fn write_provider_file(config_dir: &Path, models: &ModelList, host: &str) -> anyhow::Result<()> {
    let dir = config_dir.join("custom_providers");
    write_transient(
        &dir,
        "goose",
        "json",
        &goose_provider_document(goose_provider_id(), models, host).to_string(),
    )?;
    Ok(())
}

fn env_after_writing<'a>(
    written: anyhow::Result<()>,
    model: &'a str,
    api_key: &'a str,
    host: &'a str,
) -> Vec<(&'a str, &'a str)> {
    match written {
        Ok(()) => goose_listed_env(goose_provider_id(), model, api_key),
        Err(error) => {
            eprintln!(
                "llmman: goose: cannot write its provider file ({error:#}); \
                 starting on {model} alone, without the model list"
            );
            goose_env(model, api_key, host)
        }
    }
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

/// What goose is handed when the provider file carries the list: the
/// provider and its key, and the model it starts on.
fn goose_listed_env<'a>(
    provider: &'a str,
    model: &'a str,
    api_key: &'a str,
) -> Vec<(&'a str, &'a str)> {
    vec![
        ("GOOSE_PROVIDER", provider),
        (GOOSE_KEY_ENV, api_key),
        ("GOOSE_MODEL", model),
    ]
}

/// goose's own `custom_providers` parent: `$GOOSE_PATH_ROOT/config` when
/// that is absolute, else the XDG config home's `goose` (also on macOS),
/// or `%APPDATA%\Block\goose\config`.
pub(super) fn goose_config_dir() -> anyhow::Result<PathBuf> {
    if let Some(root) = env_dir("GOOSE_PATH_ROOT").filter(|p| p.is_absolute()) {
        return Ok(root.join("config"));
    }
    if cfg!(windows) {
        let appdata = roaming_app_data(
            env_dir("APPDATA"),
            dirs::config_dir,
            crate::config::home_dir,
        )
        .context("no application data directory")?;
        return Ok(appdata.join("Block").join("goose").join("config"));
    }
    let home = dirs::home_dir().context("no home directory")?;
    Ok(xdg_dir(&home, "XDG_CONFIG_HOME", ".config").join("goose"))
}

/// The Roaming application data directory the way goose finds it (via `etcetera`): `%APPDATA%`,
/// else the OS's answer, else `AppData\Roaming` under home. `dirs::config_dir` alone answers
/// nothing when `%USERPROFILE%` was moved to a directory with no `AppData`, where goose still
/// starts.
fn roaming_app_data(
    env: Option<PathBuf>,
    os: impl FnOnce() -> Option<PathBuf>,
    home: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    env.or_else(os)
        .or_else(|| home().map(|home| home.join("AppData").join("Roaming")))
}

/// The OpenAI-compatible provider goose reads the models from, one record per row. goose has no
/// display name apart from the name it sends (the `reference`), and takes no per-model variants,
/// output limit or tool flag.
fn goose_provider_document(provider: &str, models: &ModelList, host: &str) -> serde_json::Value {
    serde_json::json!({
        "name": provider,
        "engine": "openai",
        "display_name": "llmman",
        "description": "Every model llmman serves",
        "api_key_env": GOOSE_KEY_ENV,
        "requires_auth": true,
        "base_url": host,
        "base_path": "v1/chat/completions",
        "dynamic_models": false,
        "skip_canonical_filtering": true,
        "models": models.entries().iter().map(goose_model).collect::<Vec<_>>(),
    })
}

fn goose_model(entry: &ModelEntry) -> serde_json::Value {
    serde_json::json!({
        "name": entry.reference,
        "context_limit": entry.context_window,
        "supports_vision": entry.vision,
    })
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

    fn hosted(provider: &str, id: &str) -> ModelEntry {
        ModelEntry::unlisted(&format!("llmman.provider/{provider}/{id}"))
    }

    fn list() -> ModelList {
        let mut agent = hosted("alpha", "agent");
        agent.context_window = Some(200_000);
        agent.vision = true;
        ModelList::of(vec![
            ModelEntry::unlisted("qwen3.5:0.8b"),
            agent,
            hosted("beta", "agent"),
        ])
    }

    /// goose reads the list from the provider file, one record per row, and
    /// tells the models apart by the name it sends: the daemon's reference.
    #[test]
    fn the_provider_file_holds_every_row_and_no_key() {
        let doc = goose_provider_document("llmman-1", &list(), "http://127.0.0.1:17434");
        assert_eq!(doc["name"], "llmman-1");
        assert_eq!(doc["engine"], "openai");
        assert_eq!(doc["base_url"], "http://127.0.0.1:17434");
        assert_eq!(doc["base_path"], "v1/chat/completions");
        assert_eq!(doc["dynamic_models"], false);
        assert_eq!(doc["api_key_env"], "LLMMAN_API_KEY");
        let models = doc["models"].as_array().unwrap();
        let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "qwen3.5:0.8b",
                "llmman.provider/alpha/agent",
                "llmman.provider/beta/agent"
            ]
        );
        assert_eq!(models[1]["context_limit"], 200_000);
        assert_eq!(models[1]["supports_vision"], true);
        assert_eq!(models[0]["context_limit"], serde_json::Value::Null);
        assert_eq!(models[2]["supports_vision"], false);
        let text = doc.to_string();
        assert!(!text.contains("tok-secret") && doc.get("api_key").is_none());
    }

    /// goose is started on the first row, with the key in the variable the
    /// provider file names and no `OPENAI_*` left to point it elsewhere.
    #[test]
    fn goose_listed_env_selects_the_provider_and_the_starting_model() {
        let list = list();
        let start = list.start().unwrap().reference.as_str();
        let env = goose_listed_env("llmman-1", start, "tok-secret");
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
        assert_eq!(get("GOOSE_PROVIDER"), Some("llmman-1"));
        assert_eq!(get("GOOSE_MODEL"), Some("qwen3.5:0.8b"));
        assert_eq!(get("LLMMAN_API_KEY"), Some("tok-secret"));
        assert!(!env.iter().any(|(n, _)| n.starts_with("OPENAI_")));
    }

    #[test]
    fn the_provider_file_is_private_and_found_in_goose_config_dir() {
        let root = super::super::test_temp_dir("goose-providers");
        let dir = root.join("config").join("custom_providers");
        let doc = goose_provider_document("llmman-1", &list(), "http://h");
        let path = write_transient(&dir, "goose", "json", &doc.to_string()).unwrap();
        assert_eq!(path.parent(), Some(dir.as_path()));
        let read: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(read, doc);
        super::super::models::finish_file(&path);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    /// goose on Windows takes `%APPDATA%` first, then the OS, then the home
    /// directory's `AppData\Roaming`; the provider file must go where it
    /// looks, and a missing answer from the OS must not stop the launch.
    #[test]
    fn the_roaming_app_data_is_found_the_way_goose_finds_it() {
        let p = |s: &str| Some(PathBuf::from(s));
        assert_eq!(
            roaming_app_data(p("env"), || p("os"), || p("home")),
            p("env")
        );
        assert_eq!(roaming_app_data(None, || p("os"), || p("home")), p("os"));
        assert_eq!(
            roaming_app_data(None, || None, || p("home")),
            Some(PathBuf::from("home").join("AppData").join("Roaming"))
        );
        assert_eq!(roaming_app_data(None, || None, || None), None);
    }

    /// A provider file goose's directory cannot take leaves goose on the
    /// plain `openai` provider and the one model, not a failed launch.
    #[test]
    fn a_provider_file_that_cannot_be_written_falls_back_to_the_plain_env() {
        let root = super::super::test_temp_dir("goose-no-dir");
        let blocker = root.join("file");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&blocker, b"x").unwrap();
        let written = write_provider_file(&blocker.join("config"), &list(), "http://h");
        assert!(written.is_err());
        let env = env_after_writing(written, "m", "k", "http://h");
        assert_eq!(env, goose_env("m", "k", "http://h"));
        let written = write_provider_file(&root.join("config"), &list(), "http://h");
        assert!(written.is_ok());
        let env = env_after_writing(written, "m", "k", "http://h");
        assert_eq!(env, goose_listed_env(goose_provider_id(), "m", "k"));
        super::super::models::finish_file(
            &root
                .join("config")
                .join("custom_providers")
                .join(format!("llmman-goose-{}.json", std::process::id())),
        );
        let _ = std::fs::remove_dir_all(root);
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
