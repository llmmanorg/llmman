//! `llmman launch gemini`.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::agy::pinned_base_url;
use super::models::ModelList;
use super::{exec_with_env, find_on_path};

/// gemini: Gemini CLI speaks Gemini's native protocol, so it is pointed at the same pinned route as
/// AGY (see [`pinned_base_url`]): every call, including its hard-coded model names, goes to the one
/// model chosen here, so `_models` is not offered.
///
/// `GOOGLE_GEMINI_BASE_URL` is the variable Gemini CLI reads; with `GEMINI_BASE_URL` the request
/// goes to generativelanguage.googleapis.com, key and all. It applies only under the
/// `gemini-api-key` login, and with none saved `-p` fails as "Invalid auth method", so Gemini CLI
/// runs from a home of llmman's own (`GEMINI_CLI_HOME`) holding that login. Verified against Gemini
/// CLI 0.71.1: it contacted only the daemon and `play.googleapis.com` (usage statistics, no
/// credential).
pub(super) fn launch_gemini(
    model: &str,
    api_key: &str,
    _models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("gemini").ok_or_else(|| anyhow::anyhow!("gemini is not installed"))?;
    let home = gemini_home()?;
    write_gemini_settings_at(&home)?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    let base_url = pinned_base_url(model);
    let home = home.to_string_lossy();
    exec_with_env(&bin, &args, &gemini_env(&base_url, api_key, &home))
}

/// Where Gemini CLI sends its requests, and its one credential: `api_key` is a session token or the
/// key `llmman serve` requires, never a provider's. `GEMINI_BASE_URL` is deliberately not set:
/// Gemini CLI 0.71.1 does not read it and would send the request, key and all, to Google.
fn gemini_env<'a>(
    base_url: &'a str,
    api_key: &'a str,
    home: &'a str,
) -> [(&'static str, &'a str); 3] {
    [
        ("GOOGLE_GEMINI_BASE_URL", base_url),
        ("GEMINI_API_KEY", api_key),
        ("GEMINI_CLI_HOME", home),
    ]
}

/// What `GEMINI_CLI_HOME` is: Gemini CLI keeps its state in `.gemini` under
/// it, which here is inside the `~/.gemini` a `--sandbox` mounts.
fn gemini_home() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".gemini")
        .join("llmman"))
}

/// Saves the `gemini-api-key` login in the home llmman owns, keeping
/// anything else Gemini CLI has saved there.
fn write_gemini_settings_at(home: &Path) -> anyhow::Result<()> {
    let path = home.join(".gemini").join("settings.json");
    let mut settings: serde_json::Value = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    settings["security"]["auth"]["selectedType"] = "gemini-api-key".into();
    super::common::write_atomic_in_dir(&path, format!("{settings:#}\n").as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_is_pointed_at_llmman_by_the_variable_it_reads_and_holds_only_the_key_given() {
        let url = pinned_base_url("llmman.provider/anthropic/claude-haiku-4-5");
        let env = gemini_env(&url, "llmman-session-token", "/h/.gemini/llmman");
        let get = |name| env.iter().find(|(n, _)| *n == name).map(|(_, v)| *v);
        assert_eq!(get("GOOGLE_GEMINI_BASE_URL"), Some(url.as_str()));
        assert!(url.starts_with(&super::super::server()), "{url}");
        assert_eq!(get("GEMINI_API_KEY"), Some("llmman-session-token"));
        assert_eq!(get("GEMINI_BASE_URL"), None);
        assert_eq!(env.len(), 3);
    }

    #[test]
    fn the_login_is_saved_in_llmmans_home_and_the_rest_of_it_kept() {
        let home = super::super::test_temp_dir("gemini-home");
        let path = home.join(".gemini").join("settings.json");
        write_gemini_settings_at(&home).unwrap();
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
        };
        assert_eq!(
            read(),
            serde_json::json!({ "security": { "auth": { "selectedType": "gemini-api-key" } } })
        );

        std::fs::write(
            &path,
            r#"{"ui":{"theme":"Atom One"},"security":{"auth":{"selectedType":"oauth-personal"},"folderTrust":{"enabled":true}}}"#,
        )
        .unwrap();
        write_gemini_settings_at(&home).unwrap();
        assert_eq!(
            read(),
            serde_json::json!({
                "ui": { "theme": "Atom One" },
                "security": {
                    "auth": { "selectedType": "gemini-api-key" },
                    "folderTrust": { "enabled": true }
                }
            })
        );

        // Not an object: replaced rather than failing the launch.
        std::fs::write(&path, "[1]").unwrap();
        write_gemini_settings_at(&home).unwrap();
        assert_eq!(read()["security"]["auth"]["selectedType"], "gemini-api-key");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn its_home_is_inside_the_gemini_dir_a_sandbox_mounts() {
        let home = gemini_home().unwrap();
        assert!(home.ends_with(".gemini/llmman") || home.ends_with(r".gemini\llmman"));
    }
}
