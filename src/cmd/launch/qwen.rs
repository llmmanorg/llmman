//! `llmman launch qwen` (Qwen Code).

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::{exec_with_env, find_on_path, has_flag, server, Effort, WINDOWS_PATH_EXTS};

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
pub(super) fn launch_qwen(
    model: &str,
    api_key: &str,
    vision: bool,
    effort: Option<&Effort>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_qwen().ok_or_else(|| anyhow::anyhow!("qwen is not installed"))?;
    let (model, vision) = qwen_model_and_vision(model, vision, extra_args);
    // After the lookup, so nothing is written for an integration that is
    // not there; `check_model_flag` has made sure there is a model.
    write_qwen_settings(model, vision, effort)?;

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

/// The model Qwen Code will use — a `--model` after `--` wins — and
/// whether to declare its image input, which `vision` answers for
/// `model` alone. `model` arrives resolved and the forwarded name as
/// typed, so they are compared resolved: `gemma4:12b` is the
/// `docker.io/ai/gemma4:12b` it names.
fn qwen_model_and_vision<'a>(
    model: &'a str,
    vision: bool,
    extra_args: &'a [String],
) -> (&'a str, bool) {
    let forwarded = common::forwarded_model(extra_args);
    let same = forwarded.is_none_or(|f| {
        crate::shortnames::resolve_ollama_api(f).is_ok_and(|resolved| resolved == model)
    });
    (forwarded.unwrap_or(model), vision && same)
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
/// a flag's camelCase spelling as well.
fn qwen_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 4);
    if !has_flag(extra_args, "--auth-type", Some("--authType")) {
        args.extend(["--auth-type".to_string(), "openai".to_string()]);
    }
    if !has_flag(extra_args, "--model", Some("-m")) {
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

/// Records llmman as the `openai` provider for `model` in Qwen Code's
/// `settings.json`, as `codex::write_codex_config` and `hermes::write_hermes_config` do
/// for theirs. See `qwen_settings_merged` for what goes in.
fn write_qwen_settings(model: &str, vision: bool, effort: Option<&Effort>) -> anyhow::Result<()> {
    write_qwen_settings_at(
        &qwen_home()?,
        model,
        &format!("{}/v1", server()),
        vision,
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
    base_url: &str,
    vision: bool,
    effort: Option<&Effort>,
) -> anyhow::Result<()> {
    common::write_json_merged(&dir.join("settings.json"), "qwen", |existing| {
        qwen_settings_merged(existing, model, base_url, vision, effort)
    })
}

/// The variable llmman's entry names as its key, and what marks the entry
/// as llmman's: a user can rename it in `/model`, but not re-key it.
pub(super) const QWEN_ENV_KEY: &str = "LLMMAN_API_KEY";

/// `existing` with llmman's entry merged in, pure so a test can hand it
/// a literal. The keys follow Qwen Code's own `/auth` and `/model` and
/// ollama's `applyQwenOllamaConfig` in `cmd/launch/qwen.go`: the entry
/// first in `modelProviders.openai`, an earlier one of llmman's replaced,
/// the rest kept and a `{ protocol, models }` wrapper unwrapped with
/// `$version` set to 4; `security.auth`; `model.name` and `model.baseUrl`.
/// A vision model's entry declares image input, which Qwen Code reads
/// only off the provider entry, not the top-level `model.generationConfig`.
/// A `--variant` declares the model's levels there too and starts the
/// entry at it, dropping the `model.reasoningEffort` `/effort` saved,
/// which Qwen Code would merge over it.
fn qwen_settings_merged(
    existing: &serde_json::Value,
    model: &str,
    base_url: &str,
    vision: bool,
    effort: Option<&Effort>,
) -> serde_json::Value {
    let mut doc = existing.as_object().cloned().unwrap_or_default();
    let mut ours = serde_json::json!({
        "id": model,
        "name": format!("{model} (llmman)"),
        "baseUrl": base_url,
        "envKey": QWEN_ENV_KEY,
    });
    if vision {
        ours["generationConfig"] = serde_json::json!({ "modalities": { "image": true } });
    }
    if let Some(effort) = effort {
        let efforts: Vec<&str> = effort
            .levels
            .iter()
            .copied()
            .filter(|l| *l != "none")
            .collect();
        let mut reasoning = serde_json::json!({
            "thinking": true,
            "disableField": "reasoning_effort",
            "profile": "openai-effort",
            "efforts": efforts,
        });
        ours["generationConfig"]["reasoning"] = match effort.default {
            "none" => serde_json::json!(false),
            level => {
                reasoning["defaultEffort"] = level.into();
                serde_json::json!({ "effort": level })
            }
        };
        ours["capabilities"] = serde_json::json!({ "reasoning": reasoning });
    }
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
    *openai = serde_json::Value::Array(std::iter::once(ours).chain(kept).collect());
    if unwrapped {
        doc.insert("$version".into(), 4.into());
    }
    let auth = common::object_under(common::object_under(&mut doc, "security"), "auth");
    auth.insert("selectedType".into(), "openai".into());
    auth.insert("baseUrl".into(), base_url.into());
    let model_cfg = common::object_under(&mut doc, "model");
    model_cfg.insert("name".into(), model.into());
    model_cfg.insert("baseUrl".into(), base_url.into());
    if effort.is_some() {
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
        let merged = qwen_settings_merged(&existing, "docker.io/ai/m:latest", url, false, None);
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
                "name": "docker.io/ai/m:latest (llmman)", "baseUrl": url,
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
        let once = qwen_settings_merged(&serde_json::json!({}), "m:latest", url, false, None);
        assert_eq!(
            once,
            serde_json::json!({
                "modelProviders": { "openai": [ { "id": "m:latest",
                    "name": "m:latest (llmman)", "baseUrl": url,
                    "envKey": "LLMMAN_API_KEY" } ] },
                "security": { "auth": { "selectedType": "openai", "baseUrl": url } },
                "model": { "name": "m:latest", "baseUrl": url }
            })
        );
        assert_eq!(
            qwen_settings_merged(&once, "m:latest", url, false, None),
            once
        );
        let text = once.to_string();
        assert!(!text.contains("apiKey") && !text.contains("\"env\""));
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"qwen"));
    }

    /// The spelling the two sides arrive in differs: `llmman launch qwen
    /// --model gemma4:12b -- --model gemma4:12b` names one model, as
    /// `docker.io/ai/gemma4:12b` and as typed.
    #[test]
    fn qwen_keeps_image_input_only_for_the_model_it_was_read_from() {
        let resolved = crate::shortnames::resolve_ollama_api("gemma4:12b").unwrap();
        let forwarded = |m: &str| vec![String::from("--model"), String::from(m)];
        let cases = [
            // Nothing forwarded, then the same model spelled either way.
            (Vec::new(), resolved.as_str(), true),
            (forwarded("gemma4:12b"), "gemma4:12b", true),
            (forwarded(&resolved), resolved.as_str(), true),
            // Another model, and one that is not a reference at all.
            (forwarded("qwen3.5:0.8b"), "qwen3.5:0.8b", false),
            (forwarded("not a reference"), "not a reference", false),
        ];
        for (extra_args, model, vision) in cases {
            assert_eq!(
                qwen_model_and_vision(&resolved, true, &extra_args),
                (model, vision),
                "{extra_args:?}"
            );
        }
    }

    /// Relaunching with a text model drops the declaration, since
    /// llmman's entry is replaced whole.
    #[test]
    fn qwen_settings_declare_image_input_only_for_a_vision_model() {
        let url = "http://h/v1";
        let vision = qwen_settings_merged(&serde_json::json!({}), "m", url, true, None);
        assert_eq!(
            vision["modelProviders"]["openai"][0]["generationConfig"],
            serde_json::json!({ "modalities": { "image": true } })
        );
        assert!(vision["model"].get("generationConfig").is_none());

        let text_only = qwen_settings_merged(&vision, "m", url, false, None);
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
        let merged = qwen_settings_merged(&existing, "m", "http://h/v1", false, None);
        assert_eq!(merged["security"]["auth"]["selectedType"], "openai");
        assert_eq!(merged["modelProviders"]["openai"][0]["id"], "m");
        assert_eq!(merged["model"]["name"], "m");
        let from_null =
            qwen_settings_merged(&serde_json::json!(null), "m", "http://h/v1", false, None);
        assert_eq!(from_null["model"]["name"], "m");

        let wrapped = serde_json::json!({
            "$version": 5,
            "modelProviders": { "openai": { "protocol": "openai", "models": [
                { "id": "gpt-5", "baseUrl": "https://api.openai.com/v1", "envKey": "MY_KEY" }
            ] } }
        });
        let merged = qwen_settings_merged(&wrapped, "m", "http://h/v1", false, None);
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

        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");
        assert!(!bak.exists(), "nothing to back up on a first write");
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written
        );

        let commented = "{\n  // mine\n  \"ui\": { \"theme\": \"x\" }\n}\n";
        std::fs::write(&path, commented).unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(read()["ui"]["theme"], "x");
        assert_eq!(read()["model"]["name"], "m:latest");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), commented);
        write_qwen_settings_at(&dir, "other:latest", url, false, None).unwrap();
        assert_eq!(read()["model"]["name"], "other:latest");
        assert_eq!(
            std::fs::read_to_string(&bak).unwrap(),
            commented,
            "llmman's own rendering must not replace the user's backup"
        );
        let edited = "{\n  // edited by hand\n  \"ui\": { \"theme\": \"y\" }\n}\n";
        std::fs::write(&path, edited).unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), edited);

        std::fs::write(&path, "  \n").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");

        std::fs::write(&path, "{ not json").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
        std::fs::write(&path, "[]").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false, None).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[]");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn qwen_settings_start_the_entry_at_the_variant() {
        let url = "http://h/v1";
        let high = Effort {
            default: "high",
            levels: vec!["none", "low", "high"],
        };
        let saved = serde_json::json!({ "model": { "reasoningEffort": "low" } });
        let doc = qwen_settings_merged(&saved, "m", url, true, Some(&high));
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

        let off = Effort {
            default: "none",
            ..high
        };
        let doc = qwen_settings_merged(&serde_json::json!({}), "m", url, false, Some(&off));
        let ours = &doc["modelProviders"]["openai"][0];
        assert_eq!(ours["generationConfig"]["reasoning"], false);
        assert!(ours["capabilities"]["reasoning"]["defaultEffort"].is_null());
    }
}
