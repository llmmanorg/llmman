//! `llmman launch hermes` (Hermes Agent).

use std::path::PathBuf;

use anyhow::Context;

use super::common;
use super::{exec_with_env, find_on_path, map_offered, server, HostedModel};

/// hermes: writes its own `~/.hermes/config.yaml` provider entry
/// pointing at our /v1 endpoint, skipping the messaging-gateway/
/// desktop-build setup a full wizard would also handle, which llmman's
/// own launch has no equivalent for.
///
/// `hosted` is every model `--provider` serves (see [`HostedModel`]):
/// each is listed under the provider's `models`, so hermes's own model
/// picker can switch between them mid-session; the launched model stays
/// the `model.default`. The list carries references only — the surface
/// has no per-model field for vision or limits, so `supports_vision`
/// still describes the launched model alone.
pub(super) fn launch_hermes(
    model: &str,
    vision: bool,
    extra_args: &[String],
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let bin = find_on_path("hermes").ok_or_else(|| anyhow::anyhow!("hermes is not installed"))?;
    write_hermes_config(
        if model.is_empty() { "default" } else { model },
        vision,
        hosted,
    )?;
    exec_with_env(&bin, extra_args, &[])
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

/// Only overwrites the `model:`/`providers:` top-level blocks this
/// itself writes — everything else in an existing `config.yaml` (other
/// providers, toolsets, etc.) is preserved, the same way
/// `codex::write_codex_config`/`codex::strip_legacy_llmman_profile` avoid clobbering
/// unrelated `config.toml` content.
fn write_hermes_config(
    model: &str,
    vision: bool,
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let config_dir = hermes_home()?;
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("config.yaml");

    let existing = std::fs::read_to_string(&config_path).unwrap_or_default();
    let preserved =
        strip_yaml_top_level_key(&strip_yaml_top_level_key(&existing, "model"), "providers");
    let ours = hermes_config_blocks(model, &format!("{}/v1", server()), vision, hosted);
    std::fs::write(&config_path, format!("{preserved}{ours}"))?;
    Ok(())
}

/// The `model:`/`providers:` blocks [`write_hermes_config`] owns. A
/// vision model gets `model.supports_vision`, the override hermes's
/// image routing reads before its own catalog, which lists no local
/// model and so routes every image through a describe-it tool instead.
///
/// `hosted` lists every model the provider offers under the provider's
/// `models`, the launched one first (see [`map_offered`]); without it
/// the list holds the one model this launch resolved.
fn hermes_config_blocks(
    model: &str,
    base_url: &str,
    vision: bool,
    hosted: Option<&[HostedModel]>,
) -> String {
    // Double-quoted (not bare) so a model name that happens to be a YAML
    // keyword (`null`, `true`, ...) or contain metacharacters (`:`, `#`,
    // ...) still parses back as the literal string it is.
    let quoted_model = common::yaml_quote(model);
    let base_url = common::yaml_quote(base_url);
    let vision = if vision {
        "  supports_vision: true\n"
    } else {
        ""
    };
    let models: Vec<&str> = match hosted {
        Some(hosted) => map_offered(hosted, |m, _| m.reference.as_str()),
        None => vec![model],
    };
    let mut list = String::new();
    for entry in models {
        list.push_str(&format!("      - {}\n", common::yaml_quote(entry)));
    }
    format!(
        "model:\n  provider: llmman\n  default: {quoted_model}\n  base_url: {base_url}\n  api_key: llmman\n{vision}\
         providers:\n  llmman:\n    name: llmman\n    api: {base_url}\n    default_model: {quoted_model}\n    models:\n{list}"
    )
}

/// Removes a top-level YAML key (`<key>:` at column 0) and every line
/// indented under it, up to the next column-0 line or EOF — the YAML
/// (indentation-block) equivalent of `codex::strip_legacy_llmman_profile`'s
/// TOML `[...]`-header block removal. Only ever needs to undo llmman's
/// own prior writes, not handle arbitrary user YAML.
fn strip_yaml_top_level_key(existing: &str, key: &str) -> String {
    let header = format!("{key}:");
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        // Blank lines and column-0 `#` comments don't belong to any block
        // on their own — don't let either reset `skipping` (that would
        // leak the rest of the removed block into `out`) or fall through
        // to it either way.
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }
        if !line.starts_with([' ', '\t']) {
            skipping = line.trim_end() == header;
            if skipping {
                continue;
            }
        } else if skipping {
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

    /// Regression test for `write_hermes_config` preserving unrelated
    /// `config.yaml` content: only its own `model:`/`providers:` blocks
    /// are replaced, not a user's other settings.
    #[test]
    fn strip_yaml_top_level_key_removes_only_that_key_and_its_block() {
        let existing = "\
toolsets:\n  - web\nmodel:\n  provider: llmman\n  default: old-model\nproviders:\n  llmman:\n    name: llmman\nchannels:\n  telegram: {}\n";
        let cleaned =
            strip_yaml_top_level_key(&strip_yaml_top_level_key(existing, "model"), "providers");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("providers:"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("  - web"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }

    #[test]
    fn hermes_config_declares_image_input_only_for_a_vision_model() {
        let vision = hermes_config_blocks("m", "http://h/v1", true, None);
        let model_block = strip_yaml_top_level_key(&vision, "providers");
        assert!(
            model_block.contains("\n  supports_vision: true\n"),
            "{vision}"
        );

        let text_only = hermes_config_blocks("m", "http://h/v1", false, None);
        assert!(!text_only.contains("supports_vision"), "{text_only}");
        assert_eq!(text_only, vision.replace("  supports_vision: true\n", ""));
    }

    #[test]
    fn hermes_config_lists_every_model_the_provider_serves() {
        let hosted = vec![
            HostedModel {
                reference: "llmman.provider/openrouter/qwen3-coder".into(),
                id: "qwen3-coder".into(),
                context: Some(262_144),
                output: Some(8_192),
                thinking: None,
                vision: Some(false),
            },
            HostedModel {
                reference: "llmman.provider/openrouter/llama-4-scout".into(),
                id: "llama-4-scout".into(),
                context: None,
                output: None,
                thinking: None,
                vision: Some(true),
            },
        ];
        let text = hermes_config_blocks(
            "llmman.provider/openrouter/qwen3-coder",
            "http://h/v1",
            false,
            Some(&hosted),
        );
        let parsed: serde_json::Value = yaml_serde::from_str(&text).expect("valid YAML");
        assert_eq!(
            parsed["model"]["default"],
            "llmman.provider/openrouter/qwen3-coder"
        );
        assert_eq!(
            parsed["providers"]["llmman"]["default_model"],
            "llmman.provider/openrouter/qwen3-coder"
        );
        assert_eq!(
            parsed["providers"]["llmman"]["models"],
            serde_json::json!([
                "llmman.provider/openrouter/qwen3-coder",
                "llmman.provider/openrouter/llama-4-scout"
            ])
        );
        // The list carries references only: no per-model vision field
        // exists on this surface, so the launched model's
        // `supports_vision` is still the only one declared.
        assert!(parsed["model"].get("supports_vision").is_none());

        // A local launch lists the one model, as before.
        let local = hermes_config_blocks("m", "http://h/v1", false, None);
        let parsed: serde_json::Value = yaml_serde::from_str(&local).expect("valid YAML");
        assert_eq!(
            parsed["providers"]["llmman"]["models"],
            serde_json::json!(["m"])
        );
    }

    #[test]
    fn strip_yaml_top_level_key_is_a_no_op_without_that_key() {
        let existing = "toolsets:\n  - web\n";
        assert_eq!(strip_yaml_top_level_key(existing, "model"), existing);
    }

    /// Regression test for a real CodeRabbit finding: a blank line inside
    /// the block being removed used to reset `skipping`, leaking the rest
    /// of that block into the output instead of removing it.
    #[test]
    fn strip_yaml_top_level_key_handles_blank_lines_inside_the_removed_block() {
        let existing = "toolsets:\n  - web\n\nmodel:\n  provider: llmman\n\n  default: old-model\n\nchannels:\n  telegram: {}\n";
        let cleaned = strip_yaml_top_level_key(existing, "model");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("default: old-model"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }

    /// Same regression as the blank-line case above, but for a column-0
    /// `#` comment (another real CodeRabbit finding).
    #[test]
    fn strip_yaml_top_level_key_handles_a_comment_inside_the_removed_block() {
        let existing = "toolsets:\n  - web\n# a comment\nmodel:\n  provider: llmman\n# another comment\n  default: old-model\nchannels:\n  telegram: {}\n";
        let cleaned = strip_yaml_top_level_key(existing, "model");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("default: old-model"));
        assert!(!cleaned.contains("another comment"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("# a comment"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }
}
