//! `llmman launch openclaw`.

use std::path::PathBuf;

use super::{common, exec_with_env, find_on_path, run_with_env, server};

/// openclaw's own onboarding independently re-verifies/pulls whatever
/// `--custom-model-id` names against its configured endpoint, and
/// mishandles llmman's own `docker.io/ai/<name>` form: it treats
/// "docker.io/ai/" as a container registry path (real observed failure:
/// "pull failed: copy image: docker.io/ai/0.8b ... requested access to
/// the resource is denied", mangling "qwen3.5:0.8b" down to "0.8b" in
/// the process). Stripping that prefix back to the bare short name —
/// what a real user would actually type — matches what its pull
/// verification expects. `"default"` when there's nothing left to strip
/// to (no `--model` given at all).
fn openclaw_model_id(model: &str) -> &str {
    let bare = model.strip_prefix("docker.io/ai/").unwrap_or(model);
    if bare.is_empty() {
        "default"
    } else {
        bare
    }
}

/// `.clawdbot` is openclaw's name before its rename.
fn onboarded_config() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    [
        home.join(".openclaw").join("openclaw.json"),
        home.join(".clawdbot").join("clawdbot.json"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

/// openclaw budgets against `contextTokens` when it is set. A provider at
/// another server, from an earlier onboarding, is left alone.
fn with_context_window(
    config: &serde_json::Value,
    server: &str,
    model: &str,
    window: u64,
) -> serde_json::Value {
    // Onboarding is given `{server}/v1` and stores it without the `/v1`.
    let base = |url: &str| {
        let url = url.trim_end_matches('/');
        url.strip_suffix("/v1").unwrap_or(url).to_string()
    };
    let mut config = config.clone();
    let Some(provider) = config.pointer_mut("/models/providers/ollama") else {
        return config;
    };
    if provider["baseUrl"].as_str().map(base) != Some(base(server)) {
        return config;
    }
    let model = crate::storage::default_tag(model);
    // `provider["models"]` would insert a null when the key is missing.
    let entries = provider.get_mut("models").and_then(|m| m.as_array_mut());
    for entry in entries.into_iter().flatten() {
        let id = entry["id"].as_str().map(crate::storage::default_tag);
        if id.as_deref() == Some(model.as_str()) {
            entry["contextWindow"] = window.into();
            entry["contextTokens"] = window.into();
        }
    }
    config
}

/// openclaw: runs its non-interactive onboarding (once) against our
/// /v1 endpoint, declares `context_window` on the model it set up, then
/// hands off to it directly. The real gateway/TUI/channel-setup
/// lifecycle a full setup wizard also manages is left to openclaw's own
/// defaults.
pub(super) fn launch_openclaw(
    model: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin =
        find_on_path("openclaw").ok_or_else(|| anyhow::anyhow!("openclaw is not installed"))?;

    if onboarded_config().is_none() {
        let effective_model = openclaw_model_id(model);
        // Through run_with_env so a --sandbox onboards the copy it runs.
        let onboard = [
            "onboard",
            "--non-interactive",
            "--accept-risk",
            "--auth-choice",
            "ollama",
            "--custom-base-url",
            &format!("{}/v1", server()),
            "--custom-model-id",
            effective_model,
            "--skip-health",
            "--skip-channels",
            "--skip-skills",
        ]
        .map(String::from);
        let code = run_with_env(&bin, &onboard, &[])?;
        anyhow::ensure!(code == 0, "openclaw onboarding failed");
    }
    if let (Some(window), Some(config)) = (context_window, onboarded_config()) {
        let (server, model) = (server(), openclaw_model_id(model));
        // openclaw starts with the window it already has.
        if let Err(e) = common::write_json_merged(&config, "openclaw", |existing| {
            with_context_window(existing, &server, model, window)
        }) {
            eprintln!("[llmman] openclaw: context window not declared: {e:#}");
        }
    }

    exec_with_env(&bin, extra_args, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the real openclaw onboarding failure
    /// described on `openclaw_model_id`'s own doc comment.
    #[test]
    fn openclaw_model_id_strips_the_docker_ai_prefix() {
        assert_eq!(
            openclaw_model_id("docker.io/ai/qwen3.5:0.8b"),
            "qwen3.5:0.8b"
        );
        assert_eq!(openclaw_model_id("qwen3.5:0.8b"), "qwen3.5:0.8b");
        assert_eq!(
            openclaw_model_id("hf.co/unsloth/Qwen3.5-0.8B-GGUF"),
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF"
        );
        assert_eq!(openclaw_model_id(""), "default");
        assert_eq!(openclaw_model_id("docker.io/ai/"), "default");
    }

    const SERVER: &str = "http://127.0.0.1:17434";

    fn onboarded(base_url: &str, models: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "models": { "providers": { "ollama": {
            "baseUrl": base_url, "api": "ollama", "models": models,
        } } } })
    }

    #[test]
    fn the_onboarded_model_entry_takes_the_served_window() {
        let entry = |id: &str| serde_json::json!({ "id": id, "contextWindow": 128000 });
        let mut config = onboarded(
            SERVER,
            serde_json::json!([entry("docker.io/ai/qwen3.5:0.8b"), entry("qwen3.5:0.8b")]),
        );
        config["agents"] = serde_json::json!({ "defaults": { "model": {
            "primary": "ollama/qwen3.5:0.8b",
        } } });
        config["gateway"] = serde_json::json!({ "auth": { "token": "kept" } });

        let mut expected = config.clone();
        expected["models"]["providers"]["ollama"]["models"][1] = serde_json::json!({
            "id": "qwen3.5:0.8b", "contextWindow": 32768, "contextTokens": 32768,
        });
        assert_eq!(
            with_context_window(&config, SERVER, "qwen3.5:0.8b", 32768),
            expected
        );
    }

    #[test]
    fn an_untagged_model_finds_its_latest_entry() {
        let id = "hf.co/HuggingFaceTB/SmolLM2-135M-Instruct";
        let models = serde_json::json!([{ "id": format!("{id}:latest"), "contextWindow": 128000 }]);
        for (stored, server) in [
            (SERVER.to_string(), SERVER.to_string()),
            (format!("{SERVER}/"), format!("{SERVER}/v1")),
            (format!("{SERVER}/v1/"), SERVER.to_string()),
        ] {
            let updated =
                with_context_window(&onboarded(&stored, models.clone()), &server, id, 8192);
            let entry = &updated["models"]["providers"]["ollama"]["models"][0];
            assert_eq!(entry["contextWindow"], 8192, "{stored} {server}");
            assert_eq!(entry["contextTokens"], 8192, "{stored} {server}");
        }
    }

    #[test]
    fn nothing_changes_without_this_servers_onboarded_entry() {
        let qwen = serde_json::json!([{ "id": "qwen3.5:0.8b", "contextWindow": 128000 }]);
        let configs = [
            serde_json::json!({}),
            serde_json::json!({ "models": { "providers": { "other": { "models": [] } } } }),
            onboarded(
                SERVER,
                serde_json::json!([{ "id": "gemma4", "contextWindow": 128000 }]),
            ),
            onboarded("http://127.0.0.1:11434", qwen.clone()),
            serde_json::json!({ "models": { "providers": { "ollama": { "models": qwen } } } }),
            serde_json::json!({ "models": { "providers": { "ollama": { "baseUrl": SERVER } } } }),
        ];
        for config in configs {
            assert_eq!(
                with_context_window(&config, SERVER, "qwen3.5:0.8b", 32768),
                config,
                "{config}"
            );
        }
    }
}
