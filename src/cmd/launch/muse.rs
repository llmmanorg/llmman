//! `llmman launch muse`.

use anyhow::Context;
use sha2::Digest;
use std::path::{Path, PathBuf};

use super::{exec_with_env, find_on_path, server};

pub(super) fn is_subcommand(arg: &str) -> bool {
    matches!(arg, "exec" | "resume")
}

fn muse_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    // Preserve the selected subcommand before generated options; putting
    // --model before it selects the interactive parser instead.
    let extra_args = if extra_args.first().is_some_and(|arg| is_subcommand(arg)) {
        args.push(extra_args[0].clone());
        &extra_args[1..]
    } else {
        extra_args
    };
    args.extend(["--provider".to_string(), "meta".to_string()]);
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// Keep generated settings separate from the user's Muse configuration.
pub(super) fn settings_root() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().context("no home directory")?;
    Ok(super::xdg_dir(&home, "XDG_CONFIG_HOME", ".config").join("llmman/muse"))
}

fn settings(
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    max_output: Option<u32>,
    authenticated: bool,
) -> serde_json::Value {
    let mut catalog = Vec::new();
    if !model.is_empty() {
        let mut row = serde_json::json!({
            "model_id": model,
            "provider_id": "meta",
            "profile_id": "tbh",
            "display_label": model,
            "visibility": "visible",
            "is_default": true
        });
        if let Some(limit) = context_window {
            row["context_limit"] = limit.into();
        }
        // Reuse the local-model reserve used by OpenCode when no catalog
        // supplies an output ceiling; do not claim an unknown context limit.
        if let Some(limit) = max_output
            .map(u64::from)
            .or_else(|| context_window.map(super::opencode::opencode_output_reserve))
        {
            row["output_limit"] = limit.into();
        }
        catalog.push(row);
    }
    serde_json::json!({
        "schema_version": 1,
        "endpoint_transport": {"base_url": base_url, "auth": if authenticated { "bearer" } else { "none" }},
        "model_catalog": catalog
    })
}

fn write_settings(root: &Path, settings: &serde_json::Value) -> anyhow::Result<()> {
    let path = root.join("muse/settings.json");
    std::fs::create_dir_all(path.parent().expect("settings file has a parent"))?;
    crate::fsutil::write_atomic(&path, &serde_json::to_vec_pretty(settings)?)
        .with_context(|| format!("write {}", path.display()))
}

/// Muse's Meta provider accepts a custom Responses API endpoint.
pub(super) fn launch_muse(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    max_output: Option<u32>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("muse").ok_or_else(|| anyhow::anyhow!("muse is not installed"))?;
    let base_url = format!("{}/muse-code/v1", server());
    let root = settings_root()?.join(hex::encode(sha2::Sha256::digest(model.as_bytes())));
    let authenticated = api_key != crate::providers::PLACEHOLDER_API_KEY && !api_key.is_empty();
    write_settings(
        &root,
        &settings(model, &base_url, context_window, max_output, authenticated),
    )?;
    let root = root
        .to_str()
        .context("Muse configuration path is not UTF-8")?;
    let args = muse_args(model, extra_args);
    exec_with_env(
        &bin,
        &args,
        &[
            ("XDG_CONFIG_HOME", root),
            ("META_API_KEY", if authenticated { api_key } else { "" }),
            ("MUSE_NO_AUTO_UPDATE", "1"),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::launch::{CONFIGURED_BY_FILE, INTEGRATIONS, MODEL_REQUIRED};

    #[test]
    fn muse_catalog_carries_limits_and_transport_without_credentials() {
        let config = settings(
            "local-model",
            "http://localhost:1234/muse-code/v1",
            Some(65536),
            Some(512),
            false,
        );
        assert_eq!(config["model_catalog"][0]["context_limit"], 65536);
        assert_eq!(config["model_catalog"][0]["output_limit"], 512);
        assert_eq!(config["model_catalog"][0]["profile_id"], "tbh");
        assert_eq!(config["endpoint_transport"]["auth"], "none");
        let config = settings("", "http://localhost/v1", None, None, true);
        assert_eq!(config["model_catalog"], serde_json::json!([]));
        assert_eq!(config["endpoint_transport"]["auth"], "bearer");
    }

    #[test]
    fn muse_is_listed_as_an_integration() {
        let muse = INTEGRATIONS.iter().find(|i| i.name == "muse").unwrap();
        assert_eq!(muse.binary, "muse");
        assert!(!MODEL_REQUIRED.contains(&"muse"));
        assert!(CONFIGURED_BY_FILE.contains(&"muse"));
    }

    #[test]
    fn muse_options_follow_the_headless_subcommand() {
        let extra = ["exec", "--max-model-steps", "1", "reply pong"].map(String::from);
        assert_eq!(
            muse_args("local-model", &extra),
            [
                "exec",
                "--provider",
                "meta",
                "--model",
                "local-model",
                "--max-model-steps",
                "1",
                "reply pong"
            ]
        );
    }

    #[test]
    fn muse_resume_precedes_generated_options() {
        assert_eq!(
            muse_args("m", &["resume".into(), "session".into()]),
            ["resume", "--provider", "meta", "--model", "m", "session"]
        );
    }

    #[test]
    fn muse_variant_options_follow_the_subcommand_and_launcher_defaults() {
        for command in ["exec", "resume"] {
            let extra = crate::cmd::launch::launch_extra_args(
                "muse",
                Some("minimal"),
                &[command.into(), "session-or-prompt".into()],
            );
            assert_eq!(
                muse_args("m", &extra),
                [
                    command,
                    "--provider",
                    "meta",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "minimal",
                    "session-or-prompt"
                ]
            );
        }
    }

    #[test]
    fn muse_interactive_args_preserve_the_default_model_and_prompt() {
        assert_eq!(
            muse_args("", &["hello".into()]),
            ["--provider", "meta", "hello"]
        );
    }
}
