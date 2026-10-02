//! `llmman launch codex`.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::{exec_with_env, find_on_path, server};

/// codex: write the `llmman` profile overlay (`~/.codex/llmman.config.toml`,
/// a provider named `llmman` at our /v1 endpoint), then run codex with
/// `--profile llmman` and `api_key` as OPENAI_API_KEY.
pub(super) fn launch_codex(
    model: &str,
    api_key: &str,
    vision: bool,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    // Write codex config
    write_codex_config(model, vision, context_window)?;

    // Regression: this used to pass a bare PathBuf::from("codex") straight
    // to exec_with_env instead of resolving it via find_on_path like every
    // other integration here does. That happened to work on Unix (bare
    // relative names go through $PATH search via execvp with no extension
    // needed), but on Windows, Command::status() calls CreateProcess
    // directly (not cmd.exe), which — unlike a shell — does not consult
    // PATHEXT to try .cmd/.bat alternatives for an extensionless name: it
    // only ever auto-appends a single ".exe". Since `npm install -g
    // @openai/codex` installs a "codex.cmd" shim on Windows, not a
    // "codex.exe", every real Windows codex launch failed with "program
    // not found" — a real E2E-verified failure, not a theoretical one.
    let bin = find_on_path("codex").ok_or_else(|| anyhow::anyhow!("codex is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    // codex profile flag
    args.extend(["--profile".to_string(), "llmman".to_string()]);
    args.extend_from_slice(extra_args);

    exec_with_env(&bin, &args, &[("OPENAI_API_KEY", api_key)])
}

/// Writes codex's `llmman` profile.
///
/// Codex 0.134+ dropped support for `--profile <name>` reading a
/// `[profiles.<name>]` table out of `config.toml`: it now only overlays a
/// sibling `~/.codex/<name>.config.toml`, using top-level keys instead of a
/// `[profiles.<name>]` wrapper (see
/// <https://developers.openai.com/codex/config-advanced#profiles>). An
/// older llmman wrote the now-unsupported `[profiles.llmman]` form directly
/// into `config.toml`, which current codex refuses to start with at all
/// ("cannot be used while config.toml contains legacy ... table") — so any
/// leftover copy of that table is stripped from `config.toml` first, then
/// the real settings are (re)written to the profile overlay file codex
/// actually reads.
fn write_codex_config(
    model: &str,
    vision: bool,
    context_window: Option<u64>,
) -> anyhow::Result<()> {
    let config_dir = codex_dir()?;
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if let Ok(existing) = std::fs::read_to_string(&config_path) {
        if existing.contains("[profiles.llmman]") {
            std::fs::write(&config_path, strip_legacy_llmman_profile(&existing))?;
        }
    }

    // Without a model there is nothing to describe; codex keeps its defaults.
    let catalog_path = config_dir.join("llmman-model.json");
    let catalog = (!model.is_empty()).then(|| {
        let context_window = codex_context_window(context_window);
        write_codex_file(
            &catalog_path,
            &codex_model_catalog(model, vision, context_window),
        )
        .map(|()| catalog_path.clone())
    });
    let catalog = catalog.transpose()?;

    let profile_path = config_dir.join("llmman.config.toml");
    write_codex_file(&profile_path, &codex_profile(&server(), catalog.as_deref()))
}

pub(super) fn codex_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".codex"))
}

/// Writes `contents` to `path` unless it already holds exactly that.
fn write_codex_file(path: &Path, contents: &str) -> anyhow::Result<()> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(contents) {
        return Ok(());
    }
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))
}

/// The catalog's `context_window` where `launch` resolved none at all;
/// ollama's fallback too.
pub(super) const CODEX_FALLBACK_CONTEXT_WINDOW: u64 = 128_000;

/// What codex compacts against: the window `launch` resolved — live from
/// the loaded runner, and a hybrid pair's larger half — else
/// [`CODEX_FALLBACK_CONTEXT_WINDOW`]. codex's catalog cannot omit the
/// field, so it guesses where every other integration stays quiet.
fn codex_context_window(window: Option<u64>) -> u64 {
    window.unwrap_or(CODEX_FALLBACK_CONTEXT_WINDOW)
}

/// The `model_catalog_json` for `model`, declaring its image input in
/// `input_modalities`. The other fields are ones codex requires, valued
/// as ollama's `buildCodexModelEntry` does.
fn codex_model_catalog(model: &str, vision: bool, context_window: u64) -> String {
    let input: &[&str] = if vision {
        &["text", "image"]
    } else {
        &["text"]
    };
    let entry = serde_json::json!({
        "slug": model,
        "display_name": model,
        "context_window": context_window,
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "truncation_policy": { "mode": "bytes", "limit": 10000 },
        "input_modalities": input,
        "base_instructions": "",
        "support_verbosity": true,
        "default_verbosity": "low",
        "supports_parallel_tool_calls": false,
        "supports_reasoning_summaries": false,
        "supported_reasoning_levels": [],
        "experimental_supported_tools": [],
    });
    let catalog = serde_json::json!({ "models": [entry] });
    serde_json::to_string_pretty(&catalog).expect("codex catalog serializes") + "\n"
}

/// The contents of `~/.codex/llmman.config.toml`: a provider of llmman's
/// own rather than `openai_base_url` on codex's built-in one, which codex
/// treats as WebSocket-capable and so opened every session with five
/// failed `ws://` attempts (~6s of "Reconnecting...") before HTTP.
fn codex_profile(server: &str, catalog: Option<&Path>) -> String {
    // A JSON string is also a valid TOML string.
    let catalog = catalog
        .map(|p| {
            let quoted = serde_json::Value::from(p.display().to_string());
            format!("model_catalog_json = {quoted}\n")
        })
        .unwrap_or_default();
    format!(
        "# Written by `llmman launch codex`; edits are overwritten.\n\
         model_provider = \"llmman\"\n\
         {catalog}\
         \n\
         [model_providers.llmman]\n\
         name = \"llmman\"\n\
         base_url = \"{server}/v1\"\n\
         env_key = \"OPENAI_API_KEY\"\n\
         wire_api = \"responses\"\n\
         supports_websockets = false\n"
    )
}

/// Removes a `[profiles.llmman]` table (and everything up to the next
/// top-level `[...]` header or end of file) from `config.toml`'s text —
/// the shape an older llmman wrote there, now rejected by current codex.
/// Line-based rather than a real TOML parser: this only ever needs to
/// undo llmman's own prior output, not handle arbitrary user TOML.
fn strip_legacy_llmman_profile(existing: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed == "[profiles.llmman]" {
            skipping = true;
            continue;
        }
        if skipping && trimmed.starts_with('[') {
            skipping = false;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the codex config bug described on
    /// `write_codex_config`'s own doc comment: an older llmman's
    /// `[profiles.llmman]` table (a format current codex refuses to load
    /// at all) must be fully removed, leaving everything else in
    /// `config.toml` untouched.
    #[test]
    fn strip_legacy_llmman_profile_removes_only_that_table() {
        let existing = "\
[some_other_setting]
foo = \"bar\"

[profiles.llmman]
openai_base_url = \"http://127.0.0.1:17434/v1\"

[profiles.other]
model = \"gpt-5\"
";
        let cleaned = strip_legacy_llmman_profile(existing);
        assert!(!cleaned.contains("[profiles.llmman]"));
        assert!(!cleaned.contains("openai_base_url"));
        assert!(cleaned.contains("[some_other_setting]"));
        assert!(cleaned.contains("foo = \"bar\""));
        assert!(cleaned.contains("[profiles.other]"));
        assert!(cleaned.contains("model = \"gpt-5\""));
    }

    #[test]
    fn strip_legacy_llmman_profile_is_a_no_op_without_the_legacy_table() {
        let existing = "[profiles.other]\nmodel = \"gpt-5\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), existing);
    }

    #[test]
    fn strip_legacy_llmman_profile_handles_the_table_at_end_of_file() {
        let existing = "[profiles.llmman]\nopenai_base_url = \"http://127.0.0.1:17434/v1\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), "");
    }

    #[test]
    fn codex_profile_is_a_websocket_free_provider_at_the_daemon() {
        let profile: toml::Value = codex_profile("http://127.0.0.1:17434", None)
            .parse()
            .expect("valid TOML");
        assert_eq!(profile["model_provider"].as_str(), Some("llmman"));
        let provider = &profile["model_providers"]["llmman"];
        assert_eq!(
            provider["base_url"].as_str(),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(provider["env_key"].as_str(), Some("OPENAI_API_KEY"));
        assert_eq!(provider["wire_api"].as_str(), Some("responses"));
        assert_eq!(provider["supports_websockets"].as_bool(), Some(false));
        assert!(
            profile.get("openai_base_url").is_none(),
            "the built-in openai provider is not the one in use"
        );
        assert!(
            profile.get("model_catalog_json").is_none(),
            "no model, no catalog to point at"
        );
    }

    #[test]
    fn codex_profile_names_the_catalog_it_was_given() {
        let path = PathBuf::from("/home/we\"ird/.codex/llmman-model.json");
        let profile: toml::Value = codex_profile("http://h", Some(&path))
            .parse()
            .expect("valid TOML");
        assert_eq!(
            profile["model_catalog_json"].as_str(),
            Some(path.to_str().unwrap())
        );
    }

    #[test]
    fn codex_model_catalog_declares_image_input_only_for_a_vision_model() {
        let catalog: serde_json::Value =
            serde_json::from_str(&codex_model_catalog("m", true, 32768)).expect("valid JSON");
        let entry = &catalog["models"][0];
        assert_eq!(
            entry["input_modalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(entry["slug"], "m");
        assert_eq!(entry["display_name"], "m");
        assert_eq!(entry["context_window"], 32768);
        // Fields codex requires of an entry.
        for key in [
            "context_window",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "truncation_policy",
            "support_verbosity",
            "supported_reasoning_levels",
            "experimental_supported_tools",
        ] {
            assert!(entry.get(key).is_some(), "missing {key}");
        }

        let text_only: serde_json::Value =
            serde_json::from_str(&codex_model_catalog("m", false, 32768)).expect("valid JSON");
        assert_eq!(
            text_only["models"][0]["input_modalities"],
            serde_json::json!(["text"])
        );
    }

    /// codex takes the window `launch` resolved, like every other
    /// integration; it differs only in having to name one when there is
    /// none. What that window is made of — live, env, trained, a pair's
    /// larger half — is `served_context_window`'s and
    /// `pair_context_window`'s own business, tested there.
    #[test]
    fn codex_context_window_falls_back_only_when_there_is_no_window() {
        assert_eq!(codex_context_window(Some(16384)), 16384);
        assert_eq!(codex_context_window(Some(1 << 20)), 1 << 20);
        assert_eq!(codex_context_window(None), CODEX_FALLBACK_CONTEXT_WINDOW);
    }
}
