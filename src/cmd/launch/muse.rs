//! `llmman launch muse`.

use anyhow::Context;
use sha2::Digest;
use std::path::{Path, PathBuf};

use super::{exec_with_env_removing, find_on_path, server};

/// Muse subcommands that accept the root options (`--provider`, `--model`,
/// `--reasoning-effort`) after the subcommand token. Options placed before
/// a subcommand that is NOT in this list select Muse's interactive parser.
/// Verified against the version pinned as `MUSE_VERSION` in ci.yml; only
/// `exec` and `resume` take model/generation options. If a future Muse
/// release adds a new generating subcommand, add it here — otherwise the
/// launcher injects options ahead of it and Muse parses the wrong command.
pub(super) fn is_subcommand(arg: &str) -> bool {
    matches!(arg, "exec" | "resume")
}

fn muse_args(model: &str, extra_args: &[String]) -> Vec<String> {
    debug_assert!(!model.is_empty(), "muse requires a non-empty model");
    let mut args = Vec::new();
    // Preserve the selected subcommand before generated options; putting
    // --model before it selects the interactive parser instead.
    let extra_args = if extra_args.first().is_some_and(|arg| is_subcommand(arg)) {
        args.push(extra_args[0].clone());
        &extra_args[1..]
    } else {
        extra_args
    };
    args.extend([
        "--provider".to_string(),
        "meta".to_string(),
        "--model".to_string(),
        model.to_string(),
    ]);
    args.extend_from_slice(extra_args);
    args
}

/// Keep generated settings separate from the user's Muse configuration.
pub(super) fn settings_root() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().context("no home directory")?;
    Ok(super::xdg_dir(&home, "XDG_CONFIG_HOME", ".config").join("llmman/muse"))
}

/// `existing` with llmman's fields set: the transport is replaced whole
/// (llmman owns it, and the user's headers must not leak into it), and
/// this model's catalog row is upserted. Every other row stays, but none
/// of them stays the default.
fn settings_merged(
    existing: &serde_json::Value,
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    max_output: Option<u32>,
    authenticated: bool,
) -> serde_json::Value {
    const FALLBACK_LIMIT: u64 = 32_768;
    let context_limit = context_window.unwrap_or(FALLBACK_LIMIT);
    // Reuse the local-model reserve used by OpenCode when no catalog
    // supplies an output ceiling.
    let output_limit = max_output
        .map(u64::from)
        .or_else(|| context_window.map(super::opencode::opencode_output_reserve))
        .unwrap_or(FALLBACK_LIMIT)
        .min(context_limit);
    let row = serde_json::json!({
        "model_id": model,
        "provider_id": "meta",
        "profile_id": "tbh",
        "display_label": model,
        "visibility": "visible",
        "is_default": true,
        "context_limit": context_limit,
        "output_limit": output_limit
    });
    let mut root = existing.as_object().cloned().unwrap_or_default();
    root.entry("schema_version").or_insert(serde_json::json!(1));
    root.insert(
        "endpoint_transport".into(),
        serde_json::json!({"base_url": base_url, "auth": if authenticated { "bearer" } else { "none" }}),
    );
    // A non-array catalog is replaced, as it cannot hold rows.
    let mut catalog = root
        .get("model_catalog")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let ours = catalog
        .iter()
        .position(|entry| entry.get("model_id").and_then(serde_json::Value::as_str) == Some(model));
    for (i, entry) in catalog.iter_mut().enumerate() {
        if let Some(obj) = entry.as_object_mut().filter(|_| Some(i) != ours) {
            obj.insert("is_default".into(), serde_json::json!(false));
        }
    }
    match ours {
        Some(i) => {
            if let (Some(obj), Some(update)) = (catalog[i].as_object_mut(), row.as_object()) {
                obj.extend(update.clone());
            }
        }
        None => catalog.push(row),
    }
    root.insert("model_catalog".into(), serde_json::Value::Array(catalog));
    serde_json::Value::Object(root)
}

/// The user's own Muse settings, as a seed for a launch-owned file. Only
/// read: `None` if absent or not a JSON object.
fn user_settings_seed(user_settings: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(user_settings).ok()?;
    serde_json::from_str(&text)
        .ok()
        .filter(serde_json::Value::is_object)
}

fn write_settings(
    root: &Path,
    user_settings: &Path,
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    max_output: Option<u32>,
    authenticated: bool,
) -> anyhow::Result<()> {
    // `root` is passed to Muse as `XDG_CONFIG_HOME` (set in `launch_muse`), and
    // Muse reads `$XDG_CONFIG_HOME/muse/settings.json`. The inner `muse/` is
    // Muse's own XDG app subdirectory, not a duplicate of the outer
    // `llmman/muse` directory: the full path is
    // `.../llmman/muse/<sha256(model)>/muse/settings.json`. Removing it would
    // silently break settings loading (launch_e2e tests assert this path).
    let path = root.join("muse/settings.json");
    // A fresh launch-owned file starts from the user's own settings, so their
    // skills, hooks and MCP servers carry over. The user's file is only read.
    let seed = if path.exists() {
        None
    } else {
        user_settings_seed(user_settings)
    };
    super::common::write_json_merged(&path, "muse", |existing| {
        settings_merged(
            seed.as_ref().unwrap_or(existing),
            model,
            base_url,
            context_window,
            max_output,
            authenticated,
        )
    })
}

/// Muse's Meta provider talks to llmman's `/v1/responses`.
pub(super) fn launch_muse(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    max_output: Option<u32>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("muse").ok_or_else(|| anyhow::anyhow!("muse is not installed"))?;
    let base_url = format!("{}/v1", server());
    let root = settings_root()?.join(hex::encode(sha2::Sha256::digest(model.as_bytes())));
    let home = dirs::home_dir().context("no home directory")?;
    // Muse's own file, resolved before `XDG_CONFIG_HOME` is overridden below.
    let user_settings =
        super::xdg_dir(&home, "XDG_CONFIG_HOME", ".config").join("muse/settings.json");
    let authenticated = api_key != crate::providers::PLACEHOLDER_API_KEY && !api_key.is_empty();
    write_settings(
        &root,
        &user_settings,
        model,
        &base_url,
        context_window,
        max_output,
        authenticated,
    )?;
    let root = root
        .to_str()
        .context("Muse configuration path is not UTF-8")?;
    let args = muse_args(model, extra_args);
    let (env, remove) = muse_env(root, api_key, authenticated);
    exec_with_env_removing(&bin, &args, &env, remove)
}

/// Muse reads the real Meta key only when authenticated. Otherwise the key is
/// removed from the inherited environment so a shell-level key never reaches
/// the local daemon.
fn muse_env<'a>(
    root: &'a str,
    api_key: &'a str,
    authenticated: bool,
) -> (Vec<(&'a str, &'a str)>, &'static [&'static str]) {
    let mut env = vec![("XDG_CONFIG_HOME", root), ("MUSE_NO_AUTO_UPDATE", "1")];
    if authenticated {
        env.push(("META_API_KEY", api_key));
        (env, &[])
    } else {
        (env, &["META_API_KEY"])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::launch::{CONFIGURED_BY_FILE, INTEGRATIONS, MODEL_REQUIRED};

    #[test]
    fn muse_catalog_carries_limits_and_transport_without_credentials() {
        let config = settings_merged(
            &serde_json::json!({}),
            "local-model",
            "http://localhost:1234/v1",
            Some(65536),
            Some(512),
            false,
        );
        assert_eq!(config["model_catalog"][0]["context_limit"], 65536);
        assert_eq!(config["model_catalog"][0]["output_limit"], 512);
        assert_eq!(config["model_catalog"][0]["profile_id"], "tbh");
        assert_eq!(config["endpoint_transport"]["auth"], "none");
    }

    /// With no window or output ceiling known, both limits are still written:
    /// the fallback, with the output never above the context.
    #[test]
    fn muse_catalog_always_writes_both_limits() {
        let config = settings_merged(&serde_json::json!({}), "m", "http://h/v1", None, None, true);
        assert_eq!(config["model_catalog"][0]["context_limit"], 32_768);
        assert_eq!(config["model_catalog"][0]["output_limit"], 32_768);
        let config = settings_merged(
            &serde_json::json!({}),
            "m",
            "http://h/v1",
            Some(4096),
            None,
            true,
        );
        assert_eq!(config["model_catalog"][0]["output_limit"], 1024);
    }

    fn launch_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "llmman-muse-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_muse(root: &Path, model: &str, user: &Path) -> serde_json::Value {
        write_settings(
            root,
            user,
            model,
            "http://127.0.0.1:17434/v1",
            Some(65536),
            Some(512),
            false,
        )
        .unwrap();
        let text = std::fs::read_to_string(root.join("muse/settings.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn muse_settings_seed_from_the_user_file_and_leave_it_alone() {
        let root = launch_root("seed");
        let user = root.join("user-settings.json");
        let user_text = r#"{
  "skills": ["review"],
  "hooks": {"pre": "echo"},
  "mcp_servers": {"fs": {"command": "fs-mcp"}},
  "endpoint_transport": {"base_url": "https://meta.example", "headers": {"x-secret": "keep-out"}}
}
"#;
        std::fs::write(&user, user_text).unwrap();

        let config = write_muse(&root.join("launch"), "local-model", &user);
        assert_eq!(config["skills"], serde_json::json!(["review"]));
        assert_eq!(config["hooks"]["pre"], "echo");
        assert_eq!(config["mcp_servers"]["fs"]["command"], "fs-mcp");
        // llmman owns the transport: the user's headers do not come along.
        assert_eq!(
            config["endpoint_transport"],
            serde_json::json!({"base_url": "http://127.0.0.1:17434/v1", "auth": "none"})
        );
        assert_eq!(std::fs::read_to_string(&user).unwrap(), user_text);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_existing_launch_file_outranks_the_user_file() {
        let root = launch_root("priority");
        let user = root.join("user-settings.json");
        std::fs::write(&user, r#"{"skills": ["user-only"]}"#).unwrap();
        let launch = root.join("launch");
        std::fs::create_dir_all(launch.join("muse")).unwrap();
        std::fs::write(
            launch.join("muse/settings.json"),
            r#"{"theme": "dark", "skills": ["muse-written"]}"#,
        )
        .unwrap();

        let config = write_muse(&launch, "local-model", &user);
        assert_eq!(config["theme"], "dark");
        assert_eq!(config["skills"], serde_json::json!(["muse-written"]));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_second_settings_write_is_idempotent() {
        let root = launch_root("idempotent");
        let user = root.join("user-settings.json");
        let launch = root.join("launch");
        let first = write_muse(&launch, "local-model", &user);
        let second = write_muse(&launch, "local-model", &user);
        assert_eq!(first, second);
        assert_eq!(second["model_catalog"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn another_catalog_row_is_kept_and_no_longer_the_default() {
        let root = launch_root("rows");
        let user = root.join("user-settings.json");
        let launch = root.join("launch");
        std::fs::create_dir_all(launch.join("muse")).unwrap();
        std::fs::write(
            launch.join("muse/settings.json"),
            r#"{"model_catalog": [{"model_id": "other", "is_default": true, "display_label": "Other"}]}"#,
        )
        .unwrap();

        let config = write_muse(&launch, "local-model", &user);
        let rows = config["model_catalog"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["model_id"], "other");
        assert_eq!(rows[0]["is_default"], false);
        assert_eq!(rows[0]["display_label"], "Other");
        assert_eq!(rows[1]["model_id"], "local-model");
        assert_eq!(rows[1]["is_default"], true);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn muse_env_sets_the_key_only_when_authenticated() {
        let (env, remove) = muse_env("/r", "llmman", false);
        assert!(!env.iter().any(|(name, _)| *name == "META_API_KEY"));
        assert_eq!(remove, ["META_API_KEY"]);

        let (env, remove) = muse_env("/r", "secret", true);
        assert!(env.contains(&("META_API_KEY", "secret")));
        assert!(remove.is_empty());

        for (env, _) in [
            muse_env("/r", "llmman", false),
            muse_env("/r", "secret", true),
        ] {
            assert!(env.contains(&("XDG_CONFIG_HOME", "/r")));
            assert!(env.contains(&("MUSE_NO_AUTO_UPDATE", "1")));
        }
    }

    #[test]
    fn muse_is_listed_as_an_integration() {
        let muse = INTEGRATIONS.iter().find(|i| i.name == "muse").unwrap();
        assert_eq!(muse.binary, "muse");
        assert!(MODEL_REQUIRED.contains(&"muse"));
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
    fn muse_interactive_args_carry_the_model_and_prompt() {
        assert_eq!(
            muse_args("m", &["hello".into()]),
            ["--provider", "meta", "--model", "m", "hello"]
        );
    }
}
