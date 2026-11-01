//! `llmman launch cline`.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::{env_dir, exec_with_env, find_on_path, node_user_profile, server};

pub(super) const CLINE_NPM_PACKAGE: &str = "cline";

/// cline: merge llmman's Ollama route into Cline's own settings, then pass
/// through exactly the arguments supplied after `--`. Cline reads both the
/// current provider store and legacy global-state fields, so keep them in
/// sync while preserving unrelated user settings in each file.
pub(super) fn launch_cline(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin =
        find_on_path("cline").ok_or_else(|| common::npm_missing("cline", CLINE_NPM_PACKAGE))?;
    write_cline_settings(model)?;
    exec_with_env(&bin, extra_args, &[("CLINE_NO_AUTO_UPDATE", "1")])
}

/// Cline's own resolution: `CLINE_DIR || path.join(os.homedir(), ".cline")`.
/// Node's `os.homedir()` reads `USERPROFILE` on Windows, which
/// `dirs::home_dir` ignores; disagreeing with Cline there means it
/// never sees the settings and exits "Not authenticated".
pub(super) fn cline_dir() -> anyhow::Result<PathBuf> {
    resolve_cline_dir(env_dir("CLINE_DIR"), node_user_profile(), || {
        dirs::home_dir().context("no home directory")
    })
}

fn resolve_cline_dir(
    cline_dir: Option<PathBuf>,
    user_profile: Option<PathBuf>,
    home: impl FnOnce() -> anyhow::Result<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if let Some(dir) = cline_dir {
        return Ok(dir);
    }
    Ok(match user_profile {
        Some(profile) => profile,
        None => home()?,
    }
    .join(".cline"))
}

pub(super) fn cline_data_dir() -> anyhow::Result<PathBuf> {
    Ok(cline_dir()?.join("data"))
}

fn write_cline_settings(model: &str) -> anyhow::Result<()> {
    let server = server();
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    write_cline_settings_at(&cline_data_dir()?, model, &server, &now)
}

/// How long Cline's Ollama provider waits for a response to *start*, in
/// ms. Through llmman it also has to cover llama-server prefilling
/// Cline's ~12k-token system prompt, since no bytes are sent before the
/// first token: on CPU that takes minutes (a 4-vCPU aarch64 runner
/// manages ~44 tok/s even on a 0.8B model), and llmman's own load
/// deadline is already 10 minutes. In CI run 35602511987 Cline dropped
/// its first request 5:00 into the prefill and re-sent it. 30 minutes
/// covers load plus a long prefill; connection failures still fail fast,
/// and a user-set value is kept.
///
/// Written twice, since Cline has read it from two places: older versions
/// from `providers.ollama.settings.timeout` (default 5 minutes), and 2.18
/// from `requestTimeoutMs` in `globalState.json`, whose Ollama handler
/// falls back to 30 seconds without it. In CI run 38123323946 every
/// aarch64 Cline request timed out that way and was re-sent mid-prefill.
const CLINE_RESPONSE_START_TIMEOUT_MS: u64 = 30 * 60 * 1000;

fn read_cline_json(path: &Path) -> anyhow::Result<(Option<Vec<u8>>, serde_json::Value)> {
    let raw = match std::fs::read(path) {
        Ok(raw) => Some(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let document = match raw.as_deref().map(|bytes| String::from_utf8_lossy(bytes)) {
        None => serde_json::json!({}),
        Some(text) if text.trim().is_empty() => serde_json::json!({}),
        Some(text) => serde_json::from_str(&text)
            .with_context(|| format!("parse {} as JSON", path.display()))?,
    };
    anyhow::ensure!(
        document.is_object(),
        "{} is not a JSON object",
        path.display()
    );
    Ok((raw, document))
}

/// Writes a changed Cline JSON document atomically, copying the exact prior
/// bytes to `<name>.json.bak` before replacing it.
fn write_cline_json(
    path: &Path,
    raw: Option<&[u8]>,
    before: &serde_json::Value,
    after: &serde_json::Value,
) -> anyhow::Result<()> {
    if before == after {
        return Ok(());
    }
    let parent = path
        .parent()
        .context("Cline settings path has no directory")?;
    std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    if let Some(raw) = raw {
        let backup = path.with_extension("json.bak");
        std::fs::write(&backup, raw)
            .with_context(|| format!("back up {} to {}", path.display(), backup.display()))?;
    }
    let mut contents = serde_json::to_vec_pretty(after).context("serialize Cline settings")?;
    contents.push(b'\n');
    crate::fsutil::write_atomic(path, &contents)
        .with_context(|| format!("write {}", path.display()))
}

fn write_cline_settings_at(
    data_dir: &Path,
    model: &str,
    server: &str,
    now: &str,
) -> anyhow::Result<()> {
    let providers_path = data_dir.join("settings/providers.json");
    let (providers_raw, mut providers_document) = read_cline_json(&providers_path)?;
    let providers_before = providers_document.clone();
    let base_url = format!("{server}/v1");
    let route_changed = providers_document
        .pointer("/providers/ollama/settings/model")
        .and_then(serde_json::Value::as_str)
        != Some(model)
        || providers_document
            .pointer("/providers/ollama/settings/baseUrl")
            .and_then(serde_json::Value::as_str)
            != Some(base_url.as_str())
        || providers_document
            .pointer("/providers/ollama/settings/timeout")
            .is_none();

    let root = providers_document
        .as_object_mut()
        .expect("read_cline_json returns an object");
    root.insert("version".to_string(), serde_json::json!(1));
    root.insert("lastUsedProvider".to_string(), serde_json::json!("ollama"));
    let providers = root
        .entry("providers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers is not a JSON object")?;
    let ollama = providers
        .entry("ollama")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers.ollama is not a JSON object")?;
    let settings = ollama
        .entry("settings")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers.ollama.settings is not a JSON object")?;
    settings.insert("provider".to_string(), serde_json::json!("ollama"));
    settings.insert("model".to_string(), serde_json::json!(model));
    settings.insert("baseUrl".to_string(), serde_json::json!(base_url));
    settings
        .entry("timeout")
        .or_insert_with(|| serde_json::json!(CLINE_RESPONSE_START_TIMEOUT_MS));
    settings.remove("apiKey");
    ollama.insert("tokenSource".to_string(), serde_json::json!("manual"));
    if route_changed {
        ollama.insert("updatedAt".to_string(), serde_json::json!(now));
    }
    write_cline_json(
        &providers_path,
        providers_raw.as_deref(),
        &providers_before,
        &providers_document,
    )?;

    let global_state_path = data_dir.join("globalState.json");
    let (global_raw, mut global_document) = read_cline_json(&global_state_path)?;
    let global_before = global_document.clone();
    let global = global_document
        .as_object_mut()
        .expect("read_cline_json returns an object");
    for key in [
        "ollamaBaseUrl",
        "actModeOllamaBaseUrl",
        "planModeOllamaBaseUrl",
    ] {
        global.insert(key.to_string(), serde_json::json!(server));
    }
    for key in ["actModeApiProvider", "planModeApiProvider"] {
        global.insert(key.to_string(), serde_json::json!("ollama"));
    }
    for key in ["actModeOllamaModelId", "planModeOllamaModelId"] {
        global.insert(key.to_string(), serde_json::json!(model));
    }
    global.insert("welcomeViewCompleted".to_string(), serde_json::json!(true));
    global
        .entry("requestTimeoutMs")
        .or_insert_with(|| serde_json::json!(CLINE_RESPONSE_START_TIMEOUT_MS));
    write_cline_json(
        &global_state_path,
        global_raw.as_deref(),
        &global_before,
        &global_document,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cline_settings_merge_ollama_and_preserve_user_state() {
        let root = std::env::temp_dir().join(format!(
            "llmman-cline-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let providers_path = root.join("settings/providers.json");
        let global_path = root.join("globalState.json");
        let model = r#"org/model.\"quoted\""#;
        let providers_original = r#"{
  "version": 7,
  "lastUsedProvider": "anthropic",
  "mine": true,
  "providers": {
    "anthropic": { "settings": { "model": "mine" } },
    "ollama": {
      "settings": { "provider": "ollama", "model": "old", "baseUrl": "http://old", "apiKey": "delete-me", "keep": 1 },
      "updatedAt": "2000-01-01T00:00:00Z",
      "other": true
    }
  }
}"#;
        let global_original = r#"{"theme":"mine","welcomeViewCompleted":false}"#;
        std::fs::create_dir_all(providers_path.parent().unwrap()).unwrap();
        std::fs::write(&providers_path, providers_original).unwrap();
        std::fs::write(&global_path, global_original).unwrap();

        write_cline_settings_at(
            &root,
            model,
            "http://127.0.0.1:17434",
            "2026-09-20T06:00:00Z",
        )
        .unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        let entry = &parsed["providers"]["ollama"];
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["lastUsedProvider"], "ollama");
        assert_eq!(parsed["mine"], true);
        assert_eq!(
            parsed["providers"]["anthropic"]["settings"]["model"],
            "mine"
        );
        assert_eq!(entry["settings"]["provider"], "ollama");
        assert_eq!(entry["settings"]["model"], model);
        assert_eq!(entry["settings"]["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(entry["settings"]["keep"], 1);
        assert_eq!(
            entry["settings"]["timeout"], CLINE_RESPONSE_START_TIMEOUT_MS,
            "a missing response-start timeout gets llmman's slow-prefill default"
        );
        assert!(entry["settings"].get("apiKey").is_none());
        assert_eq!(entry["tokenSource"], "manual");
        assert_eq!(entry["updatedAt"], "2026-09-20T06:00:00Z");
        assert_eq!(entry["other"], true);
        assert_eq!(
            std::fs::read_to_string(root.join("settings/providers.json.bak")).unwrap(),
            providers_original
        );

        let global: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&global_path).unwrap()).unwrap();
        assert_eq!(global["theme"], "mine");
        assert_eq!(global["ollamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["actModeApiProvider"], "ollama");
        assert_eq!(global["planModeApiProvider"], "ollama");
        assert_eq!(global["actModeOllamaModelId"], model);
        assert_eq!(global["planModeOllamaModelId"], model);
        assert_eq!(global["actModeOllamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["planModeOllamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["welcomeViewCompleted"], true);
        assert_eq!(
            global["requestTimeoutMs"], CLINE_RESPONSE_START_TIMEOUT_MS,
            "a missing response-start timeout gets llmman's slow-prefill default"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("globalState.json.bak")).unwrap(),
            global_original
        );

        let providers_written = std::fs::read(&providers_path).unwrap();
        let global_written = std::fs::read(&global_path).unwrap();
        write_cline_settings_at(
            &root,
            model,
            "http://127.0.0.1:17434",
            "2026-09-20T07:00:00Z",
        )
        .unwrap();
        assert_eq!(std::fs::read(&providers_path).unwrap(), providers_written);
        assert_eq!(std::fs::read(&global_path).unwrap(), global_written);

        write_cline_settings_at(
            &root,
            "new-model",
            "http://127.0.0.1:17434",
            "2026-09-20T08:00:00Z",
        )
        .unwrap();
        let changed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        assert_eq!(
            changed["providers"]["ollama"]["updatedAt"],
            "2026-09-20T08:00:00Z"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cline_settings_keep_a_user_set_response_start_timeout() {
        let root = std::env::temp_dir().join(format!(
            "llmman-cline-timeout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let providers_path = root.join("settings/providers.json");
        std::fs::create_dir_all(providers_path.parent().unwrap()).unwrap();
        std::fs::write(
            &providers_path,
            r#"{"providers":{"ollama":{"settings":{"timeout":45000}}}}"#,
        )
        .unwrap();
        let global_path = root.join("globalState.json");
        std::fs::write(&global_path, r#"{"requestTimeoutMs":90000}"#).unwrap();

        write_cline_settings_at(
            &root,
            "some-model",
            "http://127.0.0.1:17434",
            "2026-09-20T06:00:00Z",
        )
        .unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        assert_eq!(parsed["providers"]["ollama"]["settings"]["timeout"], 45000);
        let global: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&global_path).unwrap()).unwrap();
        assert_eq!(global["requestTimeoutMs"], 90000);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Same precedence as Cline: `CLINE_DIR`, then `USERPROFILE` (Windows'
    /// `os.homedir()`), then the process home.
    #[test]
    fn cline_dir_follows_cline_dir_then_userprofile_then_home() {
        let p = |s: &str| Some(PathBuf::from(s));
        let home = || Ok(PathBuf::from("/home"));
        let resolve = |dir, profile| resolve_cline_dir(dir, profile, home).unwrap();
        assert_eq!(
            resolve(p("/explicit"), p("/profile")),
            PathBuf::from("/explicit")
        );
        assert_eq!(
            resolve(None, p("/profile")),
            PathBuf::from("/profile/.cline")
        );
        assert_eq!(resolve(None, None), PathBuf::from("/home/.cline"));
    }
}
