//! `llmman launch qwen` (Qwen Code).

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::models::{ModelEntry, ModelList};
use super::pi::default_effort;
use super::{
    before_terminator, exec_with_env, find_on_path, has_flag, has_forwarded_model_flag, server,
    Effort, WINDOWS_PATH_EXTS,
};

/// qwen: Qwen Code's OpenAI-compatible mode, pointed at our /v1 by the
/// command line, the environment and its settings file together, since
/// Qwen Code reads the three in a different order for each value:
/// `--auth-type` and `--model` win on the command line, the base URL is
/// won by a `modelProviders` entry for the model (`resolveModelConfig` in
/// its `packages/core/src/models/modelConfigResolver.ts`), which is what
/// `write_qwen_settings` is for, and the key stays in the environment,
/// named in that entry as `LLMMAN_API_KEY` so llmman's entry is told from
/// any other. Ollama's `cmd/launch/qwen.go` does the same three. A
/// `--model` after `--` is the one Qwen Code uses, so the settings and
/// `OPENAI_MODEL` follow it.
///
/// `modelProviders.openai` holds every row of `models`, which `/model`
/// switches among; Qwen Code sends the entry's `id`, the `reference` the
/// daemon routes. The one key is still the environment's, so a switch to a
/// provider other than the start's is for the daemon to key (see [`super::keys`]).
pub(super) fn launch_qwen(
    model: &str,
    api_key: &str,
    _vision: bool,
    effort: Option<&Effort>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_qwen().ok_or_else(|| anyhow::anyhow!("qwen is not installed"))?;
    let model = qwen_model(model, models, extra_args);
    // After the lookup, so nothing is written for an integration that is
    // not there; the plan has made sure there is a model.
    write_qwen_settings(model, models, effort)?;

    let base_url = format!("{}/v1", server());
    let mut env = vec![
        ("OPENAI_BASE_URL", base_url.as_str()),
        ("OPENAI_API_KEY", api_key),
        (QWEN_ENV_KEY, api_key),
        ("OPENAI_MODEL", model),
    ];
    // A shim the fallback found is a `#!/usr/bin/env node` script whose
    // `node` sits beside it, so its directory goes on the child's `PATH`.
    let path = path_with_dir_prepended(bin.parent(), &std::env::var_os("PATH").unwrap_or_default())
        .map(|p| p.to_string_lossy().into_owned());
    if let Some(path) = &path {
        env.push(("PATH", path.as_str()));
    }
    exec_with_env(&bin, &qwen_args(model, extra_args), &env)
}

/// The model Qwen Code will use: a `--model` after `--` wins. It arrives as
/// typed and `model` resolved, so a row it names is found resolved:
/// `gemma4:12b` is the `docker.io/ai/gemma4:12b` it names.
fn qwen_model<'a>(model: &'a str, models: &'a ModelList, extra_args: &'a [String]) -> &'a str {
    let Some(forwarded) = common::forwarded_model(extra_args) else {
        return model;
    };
    let resolved = crate::shortnames::resolve_ollama_api(forwarded).ok();
    models
        .entries()
        .iter()
        .map(|entry| entry.reference.as_str())
        .find(|reference| *reference == forwarded || Some(*reference) == resolved.as_deref())
        .unwrap_or(forwarded)
}

/// `path_var` with `dir` in front, or `None` when it is there already or
/// there is no `dir`. Empty components go: one is how an unset `PATH`
/// arrives, and on POSIX it means the working directory.
fn path_with_dir_prepended(
    dir: Option<&Path>,
    path_var: &std::ffi::OsStr,
) -> Option<std::ffi::OsString> {
    let dir = dir?;
    let mut components: Vec<PathBuf> = std::env::split_paths(path_var)
        .filter(|d| !d.as_os_str().is_empty())
        .collect();
    if components.iter().any(|d| d == dir) {
        return None;
    }
    components.insert(0, dir.to_path_buf());
    std::env::join_paths(components).ok()
}

/// `--auth-type openai --model <model>` ahead of the caller's own
/// arguments, each dropped when the caller already passed it after `--`:
/// Qwen Code 0.22.3 crashes on either flag repeated (a `toLowerCase`
/// TypeError) rather than taking the last one, and a caller who spelled
/// out an auth type meant it. `--authType` is checked too — yargs accepts
/// a flag's camelCase spelling as well. Neither is looked for after a
/// second `--`: Qwen Code does not read anything there as a flag.
fn qwen_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 4);
    if !has_flag(
        before_terminator(extra_args),
        "--auth-type",
        Some("--authType"),
    ) {
        args.extend(["--auth-type".to_string(), "openai".to_string()]);
    }
    if !has_forwarded_model_flag("qwen", extra_args) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// `PATH`, then the installers' own targets; see `qwen_fallback_paths`.
pub(super) fn find_qwen() -> Option<PathBuf> {
    find_on_path("qwen").or_else(|| qwen_fallback_paths().into_iter().find(|p| p.is_file()))
}

/// Where a Qwen Code install lands that a process without the user's
/// login shell does not see: the standalone installer's `~/.local/bin`
/// and, on Windows, its `%LOCALAPPDATA%\qwen-code\bin`
/// (`Get-QwenInstallBinDir` in Qwen Code's
/// `scripts/installation/install-qwen-standalone.ps1`); the
/// `~/.npm-global` prefix its older npm installer set; any node under
/// `~/.nvm` that has it; Homebrew's prefixes and `/usr/local/bin`; and the rest of
/// what ollama's `cmd/launch/qwen.go` probes, `~/.cargo/bin`, macOS's
/// `~/Library/Application Support/qwen/bin`, and on Windows npm's global
/// directory under both `%APPDATA%` and `%LOCALAPPDATA%`,
/// `%LOCALAPPDATA%\Programs\qwen` and `%APPDATA%\qwen\bin`.
fn qwen_fallback_paths() -> Vec<PathBuf> {
    let home = dirs::home_dir();
    let mut paths = Vec::new();
    if cfg!(windows) {
        // Blank counts as unset, or the candidate would be relative.
        let roaming = std::env::var_os("APPDATA")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join("AppData").join("Roaming")));
        let local = std::env::var_os("LOCALAPPDATA")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join("AppData").join("Local")));
        let dirs = [
            roaming.as_ref().map(|d| d.join("npm")),
            local.as_ref().map(|d| d.join("npm")),
            local.as_ref().map(|d| d.join("qwen-code").join("bin")),
            local.as_ref().map(|d| d.join("Programs").join("qwen")),
            roaming.as_ref().map(|d| d.join("qwen").join("bin")),
        ];
        for dir in dirs.into_iter().flatten() {
            paths.extend(
                WINDOWS_PATH_EXTS
                    .iter()
                    .map(|ext| dir.join(format!("qwen.{ext}"))),
            );
        }
        return paths;
    }
    if let Some(h) = &home {
        paths.push(h.join(".local").join("bin").join("qwen"));
        paths.push(h.join(".npm-global").join("bin").join("qwen"));
        paths.push(h.join(".cargo").join("bin").join("qwen"));
        if cfg!(target_os = "macos") {
            paths.push(
                h.join("Library")
                    .join("Application Support")
                    .join("qwen")
                    .join("bin")
                    .join("qwen"),
            );
        }
        paths.extend(nvm_qwen(h));
    }
    if cfg!(target_os = "macos") {
        paths.push(PathBuf::from("/opt/homebrew/bin/qwen"));
    } else {
        paths.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin/qwen"));
    }
    paths.push(PathBuf::from("/usr/local/bin/qwen"));
    paths
}

/// The `qwen` under any node version in `~/.nvm`, the way ollama's
/// `cmd/launch/qwen.go` globs for it.
fn nvm_qwen(home: &Path) -> Option<PathBuf> {
    std::fs::read_dir(home.join(".nvm").join("versions").join("node"))
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("bin").join("qwen"))
        .find(|p| p.is_file())
}

/// `$QWEN_HOME` if set, else `~/.qwen`: `Storage.getGlobalQwenDir` in
/// Qwen Code's `packages/core/src/config/storage.ts`. Qwen Code can also
/// take it from `~/.qwen/.env`; that is left to the user's shell.
pub(super) fn qwen_home() -> anyhow::Result<PathBuf> {
    let home = || dirs::home_dir().context("no home directory");
    match std::env::var("QWEN_HOME").ok().filter(|d| !d.is_empty()) {
        Some(dir) if !dir.starts_with('~') => Ok(PathBuf::from(dir)),
        Some(dir) => Ok(common::expand_tilde(&dir, &home()?)),
        None => Ok(home()?.join(".qwen")),
    }
}

/// Records llmman as the `openai` provider for `models` in Qwen Code's
/// `settings.json`, as `codex::write_codex_config` and `hermes::write_hermes_config` do
/// for theirs, and `model` as the one it starts on. See `qwen_settings_merged`
/// for what goes in.
fn write_qwen_settings(
    model: &str,
    models: &ModelList,
    effort: Option<&Effort>,
) -> anyhow::Result<()> {
    write_qwen_settings_at(
        &qwen_home()?,
        model,
        models,
        &format!("{}/v1", server()),
        effort,
    )
}

/// Read as Qwen Code reads it, comments stripped and an empty file as
/// `{}`. A file that is not a JSON object is left alone with a line
/// printed, since Qwen Code resets such a file to `{}` itself; one that
/// parses but cannot be written is an error, since an entry in it may be
/// the one this write was to outrank. The user's own file, and any later
/// one carrying comments, is kept as `settings.json.bak`.
fn write_qwen_settings_at(
    dir: &Path,
    model: &str,
    models: &ModelList,
    base_url: &str,
    effort: Option<&Effort>,
) -> anyhow::Result<()> {
    let rows = qwen_models(models, model, base_url, effort);
    common::write_json_merged(&dir.join("settings.json"), "qwen", |existing| {
        qwen_settings_merged(existing, model, &rows, base_url, effort.is_some())
    })
}

/// The variable llmman's entry names as its key, and what marks the entry
/// as llmman's: a user can rename it in `/model`, but not re-key it.
pub(super) const QWEN_ENV_KEY: &str = "LLMMAN_API_KEY";

/// Every row of `models` as a Qwen Code model. A `--variant` starts the row
/// of `selected` at its level.
fn qwen_models(
    models: &ModelList,
    selected: &str,
    base_url: &str,
    effort: Option<&Effort>,
) -> Vec<serde_json::Value> {
    models.map_named(|entry, name| {
        let chosen = effort
            .map(|e| e.default)
            .filter(|_| entry.reference == selected);
        qwen_model_entry(entry, name, base_url, chosen)
    })
}

/// One row as an entry of `modelProviders.openai`. The window and image input are declared in the
/// entry's `generationConfig`, the only place Qwen Code reads them; `max_output` is its
/// `samplingParams.max_tokens`, which Qwen Code otherwise defaults to more than a smaller model
/// accepts. Declared levels make `/effort` offer them (Qwen Code needs the default named for an
/// unknown id). `chosen` is a `--variant` (`none`: thinking off), which Qwen Code would merge over
/// the level `/effort` saved.
fn qwen_model_entry(
    entry: &ModelEntry,
    name: &str,
    base_url: &str,
    chosen: Option<&str>,
) -> serde_json::Value {
    let mut ours = serde_json::json!({
        "id": entry.reference,
        "name": name,
        "baseUrl": base_url,
        "envKey": QWEN_ENV_KEY,
    });
    let mut generation = serde_json::Map::new();
    if let Some(window) = entry.context_window {
        generation.insert("contextWindowSize".into(), window.into());
    }
    if entry.vision {
        generation.insert("modalities".into(), serde_json::json!({ "image": true }));
    }
    if let Some(max) = entry.max_output {
        generation.insert(
            "samplingParams".into(),
            serde_json::json!({ "max_tokens": max }),
        );
    }
    let spelled = entry.spelled_variants("qwen");
    let efforts: Vec<&str> = spelled
        .iter()
        .map(String::as_str)
        .filter(|level| *level != "none")
        .collect();
    if !efforts.is_empty() {
        let start = chosen
            .filter(|c| *c != "none")
            .or_else(|| default_effort(&efforts));
        ours["capabilities"] = serde_json::json!({ "reasoning": {
            "thinking": true,
            "disableField": "reasoning_effort",
            "profile": "openai-effort",
            "efforts": efforts,
            "defaultEffort": start,
        } });
        match chosen {
            Some("none") => generation.insert("reasoning".into(), serde_json::json!(false)),
            Some(level) => {
                generation.insert("reasoning".into(), serde_json::json!({ "effort": level }))
            }
            None => None,
        };
    }
    if !generation.is_empty() {
        ours["generationConfig"] = generation.into();
    }
    ours
}

/// `existing` with llmman's entries merged in, pure so a test can hand it a literal. Keys follow
/// Qwen Code's own `/auth` and `/model` and ollama's `applyQwenOllamaConfig`: entries first in
/// `modelProviders.openai` (earlier ones of llmman's replaced, a `{ protocol, models }` wrapper
/// unwrapped with `$version` 4), `security.auth`, `model.name` (the start, unless a `--model` after
/// `--` says otherwise) and `model.baseUrl`. A `--variant` (`reset_effort`) drops the
/// `model.reasoningEffort` `/effort` saved, which Qwen Code would merge over the entry's level.
fn qwen_settings_merged(
    existing: &serde_json::Value,
    selected: &str,
    ours: &[serde_json::Value],
    base_url: &str,
    reset_effort: bool,
) -> serde_json::Value {
    let mut doc = existing.as_object().cloned().unwrap_or_default();
    let openai = common::object_under(&mut doc, "modelProviders")
        .entry("openai")
        .or_insert_with(|| serde_json::json!([]));
    let unwrapped = openai.get("models").is_some_and(|m| m.is_array());
    let entries = openai
        .as_array()
        .or_else(|| openai.get("models").and_then(serde_json::Value::as_array));
    let kept = entries.map_or_else(Vec::new, |entries| {
        entries
            .iter()
            .filter(|e| !qwen_entry_is_ours(e, base_url))
            .cloned()
            .collect()
    });
    *openai = serde_json::Value::Array(ours.iter().cloned().chain(kept).collect());
    if unwrapped {
        doc.insert("$version".into(), 4.into());
    }
    let auth = common::object_under(common::object_under(&mut doc, "security"), "auth");
    auth.insert("selectedType".into(), "openai".into());
    auth.insert("baseUrl".into(), base_url.into());
    let model_cfg = common::object_under(&mut doc, "model");
    model_cfg.insert("name".into(), selected.into());
    model_cfg.insert("baseUrl".into(), base_url.into());
    if reset_effort {
        model_cfg.remove("reasoningEffort");
    }
    serde_json::Value::Object(doc)
}

/// An entry llmman wrote: `QWEN_ENV_KEY` as its key, at this daemon's
/// address, the test `qwenIsOllamaProvider` makes in ollama's
/// `cmd/launch/qwen.go`. The id is the model name and the display name
/// is the user's to change, so neither marks an owner.
fn qwen_entry_is_ours(entry: &serde_json::Value, base_url: &str) -> bool {
    let field = |k: &str| entry.get(k).and_then(serde_json::Value::as_str);
    field("envKey") == Some(QWEN_ENV_KEY)
        && field("baseUrl")
            .is_some_and(|u| u.trim_end_matches('/') == base_url.trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::super::PROVIDER_NEEDS_DAEMON_KEY;
    use super::*;

    fn entry(provider: Option<&str>, id: &str) -> ModelEntry {
        let reference = match provider {
            Some(p) => crate::providers::format_remote_ref(p, id),
            None => id.to_string(),
        };
        ModelEntry {
            reference,
            id: id.to_string(),
            provider: provider.map(|p| (p.to_string(), p.to_uppercase())),
            context_window: None,
            max_output: None,
            thinking: Some(Vec::new()),
            vision: false,
            tools: None,
            cost: None,
            released: None,
        }
    }

    fn one_row(id: &str, vision: bool) -> ModelList {
        ModelList::of(vec![ModelEntry {
            vision,
            ..entry(None, id)
        }])
    }

    /// `list`'s entries merged into `existing`, starting on `selected`.
    fn merged(
        existing: &serde_json::Value,
        selected: &str,
        list: &ModelList,
        url: &str,
        effort: Option<&Effort>,
    ) -> serde_json::Value {
        let rows = qwen_models(list, selected, url, effort);
        qwen_settings_merged(existing, selected, &rows, url, effort.is_some())
    }

    /// One row, `model`, merged into `existing`.
    fn settings(
        existing: &serde_json::Value,
        model: &str,
        url: &str,
        vision: bool,
        effort: Option<&Effort>,
    ) -> serde_json::Value {
        merged(existing, model, &one_row(model, vision), url, effort)
    }

    /// The found directory goes in front of `PATH` only when it is not
    /// there, with no empty component either way.
    #[test]
    fn path_with_dir_prepended_only_when_it_is_missing() {
        let path_var =
            std::env::join_paths([PathBuf::from("/usr/bin"), PathBuf::from("/bin")]).unwrap();
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/usr/bin")), &path_var),
            None
        );
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), &path_var),
            Some(
                std::env::join_paths([
                    PathBuf::from("/opt/nvm/bin"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ])
                .unwrap()
            )
        );
        assert_eq!(path_with_dir_prepended(None, &path_var), None);
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), std::ffi::OsStr::new("")),
            Some(std::ffi::OsString::from("/opt/nvm/bin"))
        );
        let gappy = std::env::join_paths([
            PathBuf::from("/usr/bin"),
            PathBuf::from(""),
            PathBuf::from("/bin"),
        ])
        .unwrap();
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), &gappy),
            Some(
                std::env::join_paths([
                    PathBuf::from("/opt/nvm/bin"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ])
                .unwrap()
            )
        );
    }

    /// The two flags `launch_qwen` relies on to beat a persisted
    /// `~/.qwen/settings.json` (see its doc comment) go first, and each
    /// yields to the caller's own spelling of it — a repeated `--model`
    /// crashes Qwen Code.
    #[test]
    fn qwen_args_prefix_auth_type_and_model_unless_the_caller_passed_them() {
        let none: Vec<String> = vec![];
        assert_eq!(
            qwen_args("m:latest", &none),
            ["--auth-type", "openai", "--model", "m:latest"]
        );

        let user_model = vec![
            "--model".to_string(),
            "theirs".to_string(),
            "-p".to_string(),
        ];
        assert_eq!(
            qwen_args("m:latest", &user_model),
            ["--auth-type", "openai", "--model", "theirs", "-p"]
        );
        let user_short = vec!["-m=theirs".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_short),
            ["--auth-type", "openai", "-m=theirs"]
        );

        let user_auth = vec!["--auth-type=qwen-oauth".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_auth),
            ["--model", "m:latest", "--auth-type=qwen-oauth"]
        );
        let user_camel = vec!["--authType".to_string(), "openai".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_camel),
            ["--model", "m:latest", "--authType", "openai"]
        );

        // After a second `--` neither is a flag, so llmman's are kept.
        let after_terminator: Vec<String> = ["--", "--model", "x", "--auth-type=qwen-oauth"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            qwen_args("m:latest", &after_terminator),
            [
                "--auth-type",
                "openai",
                "--model",
                "m:latest",
                "--",
                "--model",
                "x",
                "--auth-type=qwen-oauth"
            ]
        );
    }

    /// Any node version under `~/.nvm` that has qwen.
    #[test]
    fn nvm_qwen_finds_it_under_a_node_version() {
        let home = std::env::temp_dir().join(format!(
            "llmman-nvm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bin = home.join(".nvm/versions/node/v22.9.1/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(home.join(".nvm/versions/node/v20.19.0/bin")).unwrap();
        std::fs::write(bin.join("qwen"), "").unwrap();
        assert_eq!(nvm_qwen(&home), Some(bin.join("qwen")));
        assert_eq!(nvm_qwen(&home.join("nowhere")), None);
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The documented targets are on the list (see `find_qwen`).
    #[cfg(unix)]
    #[test]
    fn qwen_fallback_paths_name_the_documented_targets() {
        let home = dirs::home_dir().unwrap();
        let paths = qwen_fallback_paths();
        assert!(paths.contains(&home.join(".local/bin/qwen")));
        assert!(paths.contains(&home.join(".npm-global/bin/qwen")));
        assert!(paths.contains(&home.join(".cargo/bin/qwen")));
        assert!(paths.contains(&PathBuf::from("/usr/local/bin/qwen")));
    }

    /// A file a Qwen Code user already has: llmman's entry goes first, an
    /// older one of its own for this daemon goes, and everything else
    /// stays, a hand-written entry at this daemon's address included.
    #[test]
    fn qwen_settings_merge_keeps_what_is_not_llmmans() {
        let existing = serde_json::json!({
            "$version": 4,
            "ui": { "theme": "keep-me" },
            "modelProviders": {
                "gemini": [ { "id": "gemini-2.5-pro" } ],
                "openai": [
                    { "id": "docker.io/ai/m:latest", "name": "cloud copy",
                      "baseUrl": "https://cloud.example/v1",
                      "envKey": "QWEN_CUSTOM_API_KEY_X", "customField": 1 },
                    { "id": "old:latest", "name": "renamed by the user",
                      "baseUrl": "http://127.0.0.1:17434/v1/", "envKey": "LLMMAN_API_KEY" },
                    { "id": "other:latest", "name": "other:latest (llmman)",
                      "baseUrl": "http://10.0.0.2:17434/v1", "envKey": "LLMMAN_API_KEY" },
                    { "id": "local-alias", "name": "my alias for the daemon",
                      "baseUrl": "http://127.0.0.1:17434/v1", "envKey": "OPENAI_API_KEY",
                      "generationConfig": { "temperature": 0.2 } }
                ]
            },
            "security": { "auth": { "selectedType": "qwen-oauth", "apiKey": "keep-too" } },
            "model": { "name": "gemini-2.5-pro", "generationConfig": { "temperature": 0.1 } }
        });
        let url = "http://127.0.0.1:17434/v1";
        let merged = settings(&existing, "docker.io/ai/m:latest", url, false, None);
        assert_eq!(merged["$version"], 4);
        assert_eq!(merged["ui"]["theme"], "keep-me");
        assert_eq!(
            merged["modelProviders"]["gemini"],
            existing["modelProviders"]["gemini"]
        );
        let before = existing["modelProviders"]["openai"].as_array().unwrap();
        let openai = merged["modelProviders"]["openai"].as_array().unwrap();
        assert_eq!(
            openai[0],
            serde_json::json!({ "id": "docker.io/ai/m:latest",
                "name": "docker.io/ai/m:latest", "baseUrl": url,
                "envKey": "LLMMAN_API_KEY" })
        );
        assert_eq!(
            openai[1..],
            [before[0].clone(), before[2].clone(), before[3].clone()]
        );
        assert_eq!(merged["security"]["auth"]["selectedType"], "openai");
        assert_eq!(merged["security"]["auth"]["baseUrl"], url);
        assert_eq!(merged["security"]["auth"]["apiKey"], "keep-too");
        assert_eq!(merged["model"]["name"], "docker.io/ai/m:latest");
        assert_eq!(merged["model"]["baseUrl"], url);
        assert_eq!(merged["model"]["generationConfig"]["temperature"], 0.1);
    }

    /// From nothing, and then again: the second merge changes nothing,
    /// so `write_qwen_settings_at` leaves a correct file alone. No key
    /// value and no `env` block anywhere in it.
    #[test]
    fn qwen_settings_merge_is_complete_from_nothing_and_idempotent() {
        let url = "http://127.0.0.1:17434/v1";
        let once = settings(&serde_json::json!({}), "m:latest", url, false, None);
        assert_eq!(
            once,
            serde_json::json!({
                "modelProviders": { "openai": [ { "id": "m:latest",
                    "name": "m:latest", "baseUrl": url,
                    "envKey": "LLMMAN_API_KEY" } ] },
                "security": { "auth": { "selectedType": "openai", "baseUrl": url } },
                "model": { "name": "m:latest", "baseUrl": url }
            })
        );
        assert_eq!(settings(&once, "m:latest", url, false, None), once);
        let text = once.to_string();
        assert!(!text.contains("apiKey") && !text.contains("\"env\""));
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"qwen"));
    }

    /// The spelling the two sides arrive in differs: `llmman launch qwen
    /// --model gemma4:12b -- --model gemma4:12b` names one model, as
    /// `docker.io/ai/gemma4:12b` and as typed, and it is the row's reference
    /// that Qwen Code is started on, so its entry is the one that applies.
    #[test]
    fn qwen_runs_the_forwarded_model_as_the_row_it_names() {
        let resolved = crate::shortnames::resolve_ollama_api("gemma4:12b").unwrap();
        let list = ModelList::of(vec![entry(None, &resolved), entry(Some("alpha"), "agent")]);
        let forwarded = |m: &str| vec![String::from("--model"), String::from(m)];
        let cases = [
            // Nothing forwarded, then the same model spelled either way.
            (Vec::new(), resolved.as_str()),
            (forwarded("gemma4:12b"), resolved.as_str()),
            (forwarded(&resolved), resolved.as_str()),
            // A row of another provider, by its reference.
            (
                forwarded("llmman.provider/alpha/agent"),
                "llmman.provider/alpha/agent",
            ),
            // Not a row, and not a reference at all: as typed.
            (forwarded("qwen3.5:0.8b"), "qwen3.5:0.8b"),
            (forwarded("not a reference"), "not a reference"),
            // After a second `--` it is not a flag, so llmman's model stays.
            (
                ["--", "--model", "qwen3.5:0.8b"].map(String::from).to_vec(),
                resolved.as_str(),
            ),
        ];
        for (extra_args, model) in cases {
            assert_eq!(
                qwen_model(&resolved, &list, &extra_args),
                model,
                "{extra_args:?}"
            );
        }
    }

    /// Relaunching with a text model drops the declaration, since
    /// llmman's entry is replaced whole.
    #[test]
    fn qwen_settings_declare_image_input_only_for_a_vision_model() {
        let url = "http://h/v1";
        let vision = settings(&serde_json::json!({}), "m", url, true, None);
        assert_eq!(
            vision["modelProviders"]["openai"][0]["generationConfig"],
            serde_json::json!({ "modalities": { "image": true } })
        );
        assert!(vision["model"].get("generationConfig").is_none());

        let text_only = settings(&vision, "m", url, false, None);
        assert!(!text_only.to_string().contains("modalities"), "{text_only}");
    }

    /// A wrong-typed value on the path is replaced, a non-object root
    /// counts as empty, and a `{ protocol, models }` wrapper keeps its
    /// entries.
    #[test]
    fn qwen_settings_merge_replaces_a_wrong_typed_value_on_its_path() {
        let existing = serde_json::json!({
            "security": 3, "modelProviders": { "openai": "x" }, "model": []
        });
        let merged = settings(&existing, "m", "http://h/v1", false, None);
        assert_eq!(merged["security"]["auth"]["selectedType"], "openai");
        assert_eq!(merged["modelProviders"]["openai"][0]["id"], "m");
        assert_eq!(merged["model"]["name"], "m");
        let from_null = settings(&serde_json::json!(null), "m", "http://h/v1", false, None);
        assert_eq!(from_null["model"]["name"], "m");

        let wrapped = serde_json::json!({
            "$version": 5,
            "modelProviders": { "openai": { "protocol": "openai", "models": [
                { "id": "gpt-5", "baseUrl": "https://api.openai.com/v1", "envKey": "MY_KEY" }
            ] } }
        });
        let merged = settings(&wrapped, "m", "http://h/v1", false, None);
        let openai = merged["modelProviders"]["openai"].as_array().unwrap();
        assert_eq!(openai.len(), 2);
        assert_eq!(openai[1]["id"], "gpt-5");
        assert_eq!(merged["$version"], 4, "the version follows the shape");
    }

    /// Ownership is the key name at this daemon's address, whatever the
    /// entry was renamed to; a trailing slash does not make a second
    /// daemon of the same one.
    #[test]
    fn qwen_entry_is_ours_needs_the_key_name_and_the_address() {
        let url = "http://127.0.0.1:17434/v1";
        let ours = serde_json::json!({ "id": "anything", "name": "renamed by the user",
            "baseUrl": "http://127.0.0.1:17434/v1/", "envKey": "LLMMAN_API_KEY" });
        assert!(qwen_entry_is_ours(&ours, url));
        let hand_written = serde_json::json!({ "id": "local-alias", "name": "m (llmman)",
            "baseUrl": url, "envKey": "OPENAI_API_KEY" });
        assert!(!qwen_entry_is_ours(&hand_written, url));
        let elsewhere = serde_json::json!({ "id": "m:latest", "name": "m:latest (llmman)",
            "baseUrl": "http://10.0.0.2:17434/v1", "envKey": "LLMMAN_API_KEY" });
        assert!(!qwen_entry_is_ours(&elsewhere, url));
        assert!(!qwen_entry_is_ours(
            &serde_json::json!("not an object"),
            url
        ));
    }

    /// The reading and writing half over a directory of its own: a fresh
    /// one gets the file, a correct file is not touched, a commented one
    /// merges with its text kept as `.bak`, a later rewrite of llmman's
    /// own rendering leaves that `.bak` alone while a hand edit with
    /// comments refreshes it, an empty file counts as `{}`, and what is
    /// not JSON is left alone without an error.
    #[test]
    fn write_qwen_settings_at_writes_once_keeps_a_bak_and_refuses_non_json() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-qwen-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let url = "http://127.0.0.1:17434/v1";
        let path = dir.join("settings.json");
        let bak = dir.join("settings.json.bak");
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
        };

        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");
        assert!(!bak.exists(), "nothing to back up on a first write");
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written
        );

        let commented = "{\n  // mine\n  \"ui\": { \"theme\": \"x\" }\n}\n";
        std::fs::write(&path, commented).unwrap();
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(read()["ui"]["theme"], "x");
        assert_eq!(read()["model"]["name"], "m:latest");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), commented);
        write_qwen_settings_at(
            &dir,
            "other:latest",
            &one_row("other:latest", false),
            url,
            None,
        )
        .unwrap();
        assert_eq!(read()["model"]["name"], "other:latest");
        assert_eq!(
            std::fs::read_to_string(&bak).unwrap(),
            commented,
            "llmman's own rendering must not replace the user's backup"
        );
        let edited = "{\n  // edited by hand\n  \"ui\": { \"theme\": \"y\" }\n}\n";
        std::fs::write(&path, edited).unwrap();
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), edited);

        std::fs::write(&path, "  \n").unwrap();
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");

        std::fs::write(&path, "{ not json").unwrap();
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
        std::fs::write(&path, "[]").unwrap();
        write_qwen_settings_at(&dir, "m:latest", &one_row("m:latest", false), url, None).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[]");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn thinking(id: &str, levels: &[&str]) -> ModelEntry {
        ModelEntry {
            thinking: Some(levels.iter().map(|l| l.to_string()).collect()),
            ..entry(None, id)
        }
    }

    #[test]
    fn qwen_settings_start_the_entry_at_the_variant() {
        let url = "http://h/v1";
        let high = Effort { default: "high" };
        let saved = serde_json::json!({ "model": { "reasoningEffort": "low" } });
        let list = ModelList::of(vec![ModelEntry {
            vision: true,
            ..thinking("m", &["none", "low", "high"])
        }]);
        let doc = merged(&saved, "m", &list, url, Some(&high));
        assert!(doc["model"]["reasoningEffort"].is_null());
        let ours = &doc["modelProviders"]["openai"][0];
        assert_eq!(
            ours["capabilities"]["reasoning"],
            serde_json::json!({
                "thinking": true,
                "disableField": "reasoning_effort",
                "profile": "openai-effort",
                "efforts": ["low", "high"],
                "defaultEffort": "high",
            })
        );
        assert_eq!(
            ours["generationConfig"],
            serde_json::json!({ "modalities": { "image": true }, "reasoning": { "effort": "high" } })
        );

        // Off: the entry's level is the one it would start at, and the
        // choice is thinking off.
        let off = Effort { default: "none" };
        let doc = merged(&serde_json::json!({}), "m", &list, url, Some(&off));
        let ours = &doc["modelProviders"]["openai"][0];
        assert_eq!(ours["generationConfig"]["reasoning"], false);
        assert_eq!(ours["capabilities"]["reasoning"]["defaultEffort"], "low");
    }

    /// Every row is an entry under its reference, the starting one first and
    /// named in `model.name`; the same id of two providers is two entries
    /// with names that tell them apart; one `--variant` is for the selected
    /// row alone.
    #[test]
    fn every_row_is_an_entry_under_its_reference() {
        let url = "http://h/v1";
        let list = ModelList::of(vec![
            thinking("shared", &["none", "low", "high"]),
            entry(Some("beta"), "shared"),
            entry(Some("alpha"), "shared"),
            entry(None, "solo"),
        ]);
        let high = Effort { default: "high" };
        let doc = merged(&serde_json::json!({}), "shared", &list, url, Some(&high));
        let openai = doc["modelProviders"]["openai"].as_array().unwrap();
        let ids: Vec<&str> = openai.iter().map(|e| e["id"].as_str().unwrap()).collect();
        assert_eq!(
            ids,
            [
                "shared",
                "llmman.provider/beta/shared",
                "llmman.provider/alpha/shared",
                "solo"
            ]
        );
        let names: Vec<&str> = openai.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["shared", "beta/shared", "alpha/shared", "solo"]);
        assert_eq!(doc["model"]["name"], "shared");
        for entry in openai {
            assert_eq!(entry["envKey"], QWEN_ENV_KEY);
            assert_eq!(entry["baseUrl"], url);
        }
        assert_eq!(openai[0]["generationConfig"]["reasoning"]["effort"], "high");
        assert!(openai[1].get("generationConfig").is_none());
        // Another row selected: that one is what the variant starts.
        let other = merged(&serde_json::json!({}), "solo", &list, url, None);
        assert_eq!(other["model"]["name"], "solo");
        assert!(
            other
                .to_string()
                .matches("\"reasoning\":{\"effort\"")
                .count()
                == 0
        );
    }

    #[test]
    fn a_row_declares_what_qwen_code_can_take_of_it() {
        let url = "http://h/v1";
        let agent = ModelEntry {
            vision: true,
            context_window: Some(200_000),
            max_output: Some(8000),
            ..thinking("agent", &["none", "minimal", "low", "high", "max"])
        };
        let row = qwen_model_entry(&agent, "agent", url, None);
        assert_eq!(
            row["generationConfig"],
            serde_json::json!({
                "contextWindowSize": 200_000,
                "modalities": { "image": true },
                "samplingParams": { "max_tokens": 8000 },
            })
        );
        // Qwen Code's own levels, and the middle-most as the one it starts at.
        assert_eq!(
            row["capabilities"]["reasoning"]["efforts"],
            serde_json::json!(["low", "high", "max"])
        );
        assert_eq!(row["capabilities"]["reasoning"]["defaultEffort"], "low");

        let plain = qwen_model_entry(&entry(None, "plain"), "plain", url, None);
        assert_eq!(
            plain,
            serde_json::json!({ "id": "plain", "name": "plain", "baseUrl": url, "envKey": QWEN_ENV_KEY })
        );
        // A switch-only template's `thinking` is `medium`.
        let switch = qwen_model_entry(&thinking("t", &["none", "thinking"]), "t", url, None);
        assert_eq!(
            switch["capabilities"]["reasoning"]["efforts"],
            serde_json::json!(["medium"])
        );
        assert_eq!(
            switch["capabilities"]["reasoning"]["defaultEffort"],
            "medium"
        );
        // Nothing said: what every wire accepts.
        let unknown = ModelEntry {
            thinking: None,
            ..entry(None, "u")
        };
        let unknown = qwen_model_entry(&unknown, "u", url, None);
        assert_eq!(
            unknown["capabilities"]["reasoning"]["efforts"],
            serde_json::json!(["low", "medium", "high"])
        );
        // No key or value of one in any row.
        assert!(!row.to_string().contains("apiKey"));
    }

    /// A relaunch replaces the entries of an earlier one whole: a model that
    /// is no longer offered is gone, one that is takes no stale field.
    #[test]
    fn a_relaunch_replaces_llmmans_entries_whole() {
        let url = "http://h/v1";
        let first = merged(
            &serde_json::json!({}),
            "a",
            &ModelList::of(vec![entry(None, "a"), entry(None, "b")]),
            url,
            None,
        );
        let second = merged(&first, "b", &one_row("b", true), url, None);
        let openai = second["modelProviders"]["openai"].as_array().unwrap();
        assert_eq!(openai.len(), 1);
        assert_eq!(openai[0]["id"], "b");
        assert_eq!(openai[0]["generationConfig"]["modalities"]["image"], true);
    }
}
