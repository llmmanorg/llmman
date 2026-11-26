//! `llmman launch hermes` (Hermes Agent).

use super::models::{ModelEntry, ModelList};
use std::path::PathBuf;

use anyhow::Context;

use super::common;
use super::{exec_with_env, find_on_path, server};

/// hermes: writes its own `~/.hermes/config.yaml` provider entry
/// pointing at our /v1 endpoint, skipping the messaging-gateway/
/// desktop-build setup a full wizard would also handle, which llmman's
/// own launch has no equivalent for. The provider lists every model of
/// the list, which hermes's `/model` offers; the config holds no key.
pub(super) fn launch_hermes(
    model: &str,
    vision: bool,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("hermes").ok_or_else(|| anyhow::anyhow!("hermes is not installed"))?;
    write_hermes_config(
        if model.is_empty() { "default" } else { model },
        vision,
        models,
    )?;
    exec_with_env(&bin, extra_args, &[])
}

/// The model hermes last ran: `model.default` in its config.
pub(super) fn remembered_model() -> Option<String> {
    let text = std::fs::read_to_string(hermes_home().ok()?.join("config.yaml")).ok()?;
    let config: serde_json::Value = yaml_serde::from_str(&text).ok()?;
    config["model"]["default"].as_str().map(str::to_string)
}

/// Matches hermes's own home-directory resolution: `$HERMES_HOME` if
/// set, else `%LOCALAPPDATA%\hermes` on Windows (real observed failure
/// otherwise — a config written to `~/.hermes` there is silently
/// ignored, since that's not where real hermes looks on Windows at all:
/// "No inference provider configured"), else `~/.hermes` everywhere else.
pub(super) fn hermes_home() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = std::env::var("HERMES_HOME") {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    if cfg!(windows) {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            if !local_app_data.trim().is_empty() {
                return Ok(PathBuf::from(local_app_data).join("hermes"));
            }
        }
        let home = dirs::home_dir().context("no home directory")?;
        return Ok(home.join("AppData").join("Local").join("hermes"));
    }
    let home = dirs::home_dir().context("no home directory")?;
    Ok(home.join(".hermes"))
}

/// Overwrites only what llmman owns of an existing `config.yaml`: the `model:` block, the `llmman`
/// provider and its `model_overrides`; the rest is kept (see `write_yaml_merged`).
fn write_hermes_config(model: &str, vision: bool, models: &ModelList) -> anyhow::Result<()> {
    let config_dir = hermes_home()?;
    std::fs::create_dir_all(&config_dir)?;
    let base_url = format!("{}/v1", server());
    let ours = hermes_config(model, &base_url, vision, models);
    common::write_yaml_merged(&config_dir.join("config.yaml"), "hermes", |existing| {
        let mut merged = existing.clone();
        merged["model"] = ours["model"].clone();
        for (section, key) in [
            ("providers", HERMES_PROVIDER),
            ("model_overrides", HERMES_OVERRIDES),
        ] {
            if !merged[section].is_object() {
                merged[section] = serde_json::json!({});
            }
            merged[section][key] = ours[section][key].clone();
        }
        merged
    })
}

/// The provider name llmman owns in `providers:`.
const HERMES_PROVIDER: &str = "llmman";

/// Where hermes reads per-model metadata for that provider
/// (`model_overrides.<provider id>.<model>`).
const HERMES_OVERRIDES: &str = "custom:llmman";

/// One row's `model_overrides` entry: what hermes would otherwise guess (200K, tools on) or take
/// from models.dev, which lists no stored model. An absent key means unknown, which hermes treats
/// as open.
fn hermes_model_override(entry: &ModelEntry) -> serde_json::Value {
    let mut meta = serde_json::Map::new();
    if let Some(window) = entry.context_window {
        meta.insert("context_window".into(), window.into());
    }
    if let Some(output) = entry.max_output {
        meta.insert("max_output_tokens".into(), output.into());
    }
    if let Some(tools) = entry.tools {
        meta.insert("supports_tools".into(), tools.into());
    }
    meta.insert("supports_vision".into(), entry.vision.into());
    if let Some(thinking) = &entry.thinking {
        meta.insert("supports_reasoning".into(), (!thinking.is_empty()).into());
    }
    meta.into()
}

/// The `model:` block, the `llmman` provider with every model, and each one's metadata. A vision
/// model gets `model.supports_vision`, which hermes's image routing reads before its own catalog
/// (it lists no local model, so images would go through a describe-it tool).
fn hermes_config(
    model: &str,
    base_url: &str,
    vision: bool,
    models: &ModelList,
) -> serde_json::Value {
    let mut block = serde_json::json!({
        "provider": HERMES_PROVIDER,
        "default": model,
        "base_url": base_url,
        "api_key": "llmman",
    });
    if vision {
        block["supports_vision"] = true.into();
    }
    // The model is always one of the models: a launch with none is still
    // a launch on "default".
    let mut ids: Vec<&str> = models
        .entries()
        .iter()
        .map(|e| e.reference.as_str())
        .collect();
    if !ids.contains(&model) {
        ids.insert(0, model);
    }
    let overrides: serde_json::Map<String, serde_json::Value> = models
        .entries()
        .iter()
        .map(|entry| (entry.reference.clone(), hermes_model_override(entry)))
        .collect();
    serde_json::json!({
        "model": block,
        "providers": {
            HERMES_PROVIDER: {
                "name": "llmman",
                "api": base_url,
                "default_model": model,
                "models": ids,
            },
        },
        "model_overrides": { HERMES_OVERRIDES: overrides },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::launch::models::ModelList;

    fn entry(reference: &str, window: Option<u64>, vision: bool) -> ModelEntry {
        ModelEntry {
            context_window: window,
            vision,
            thinking: Some(vec!["none".into(), "high".into()]),
            tools: Some(true),
            ..ModelEntry::unlisted(reference)
        }
    }

    fn list() -> ModelList {
        ModelList::of(vec![
            entry("qwen3.5:0.8b", Some(32768), false),
            entry("llmman.provider/anthropic/claude-x", Some(200_000), true),
            entry("llmman.provider/openrouter/anthropic/claude-x", None, false),
        ])
    }

    #[test]
    fn every_model_is_listed_with_its_own_metadata_and_the_start_is_the_default() {
        let config = hermes_config("qwen3.5:0.8b", "http://h/v1", false, &list());
        let provider = &config["providers"]["llmman"];
        assert_eq!(provider["default_model"], "qwen3.5:0.8b");
        assert_eq!(
            provider["models"],
            serde_json::json!([
                "qwen3.5:0.8b",
                "llmman.provider/anthropic/claude-x",
                "llmman.provider/openrouter/anthropic/claude-x"
            ])
        );
        let overrides = &config["model_overrides"]["custom:llmman"];
        assert_eq!(overrides["qwen3.5:0.8b"]["context_window"], 32768);
        assert_eq!(
            overrides["llmman.provider/anthropic/claude-x"]["supports_vision"],
            true
        );
        assert_eq!(overrides["qwen3.5:0.8b"]["supports_vision"], false);
        assert_eq!(overrides["qwen3.5:0.8b"]["supports_reasoning"], true);
        // Unknown stays unknown, not zero.
        assert!(overrides["llmman.provider/openrouter/anthropic/claude-x"]
            .get("context_window")
            .is_none());
    }

    #[test]
    fn a_launch_without_a_model_still_has_one() {
        let config = hermes_config("default", "http://h/v1", false, &ModelList::default());
        assert_eq!(
            config["providers"]["llmman"]["models"],
            serde_json::json!(["default"])
        );
        assert_eq!(config["model"]["default"], "default");
    }

    #[test]
    fn hermes_config_declares_image_input_only_for_a_vision_model() {
        let one = ModelList::of(vec![entry("m", None, true)]);
        let vision = hermes_config("m", "http://h/v1", true, &one);
        assert_eq!(vision["model"]["supports_vision"], true);
        let text_only = hermes_config("m", "http://h/v1", false, &one);
        assert!(text_only["model"].get("supports_vision").is_none());
    }

    #[test]
    fn the_config_holds_no_credential() {
        let text =
            serde_json::to_string(&hermes_config("m", "http://h/v1", false, &list())).unwrap();
        // The placeholder every integration writes for a model llmman serves.
        assert!(text.contains("\"api_key\":\"llmman\""));
        assert!(!text.contains("sk-") && !text.contains("session"));
    }

    /// Only llmman's own keys are replaced in an existing `config.yaml`.
    #[test]
    fn the_merge_keeps_the_rest_of_an_existing_config() {
        let dir = crate::cmd::launch::test_temp_dir("hermes-config");
        let path = dir.join("config.yaml");
        std::fs::write(
            &path,
            "toolsets:\n  - web\nmodel:\n  provider: other\n  default: old\nproviders:\n  mine:\n    api: http://mine/v1\n  llmman:\n    default_model: old\nmodel_overrides:\n  mine-provider:\n    m: {context_window: 1}\n",
        )
        .unwrap();
        let ours = hermes_config(
            "m",
            "http://h/v1",
            false,
            &ModelList::of(vec![entry("m", None, false)]),
        );
        common::write_yaml_merged(&path, "hermes", |existing| {
            let mut merged = existing.clone();
            merged["model"] = ours["model"].clone();
            merged["providers"][HERMES_PROVIDER] = ours["providers"][HERMES_PROVIDER].clone();
            merged
        })
        .unwrap();
        let merged: serde_json::Value =
            yaml_serde::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(merged["toolsets"], serde_json::json!(["web"]));
        assert_eq!(merged["providers"]["mine"]["api"], "http://mine/v1");
        assert_eq!(merged["providers"]["llmman"]["default_model"], "m");
        assert_eq!(merged["model"]["provider"], "llmman");
        assert_eq!(
            merged["model_overrides"]["mine-provider"]["m"]["context_window"],
            1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
