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

/// `.clawdbot` is openclaw's name before its rename. openclaw is node, so its
/// home is the one `os.homedir()` reads, `%USERPROFILE%` on Windows, which
/// `dirs::home_dir` ignores: with that, the config it wrote was never found
/// and neither the primary model nor the context window was set there.
fn onboarded_config() -> Option<PathBuf> {
    let home = crate::config::home_dir()?;
    [
        home.join(".openclaw").join("openclaw.json"),
        home.join(".clawdbot").join("clawdbot.json"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

/// Onboarding is given `{server}/v1` and stores it without the `/v1`.
fn base_url(url: &str) -> &str {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/v1").unwrap_or(url)
}

/// openclaw budgets against `contextTokens` when it is set. A provider at
/// another server, from an earlier onboarding, is left alone.
fn with_context_window(
    config: &serde_json::Value,
    server: &str,
    model: &str,
    window: u64,
) -> serde_json::Value {
    let mut config = config.clone();
    let Some(provider) = config.pointer_mut("/models/providers/ollama") else {
        return config;
    };
    if provider["baseUrl"].as_str().map(base_url) != Some(base_url(server)) {
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

/// The id the `ollama` provider at `server` lists `model` under.
fn listed_id(config: &serde_json::Value, server: &str, model: &str) -> Option<String> {
    let provider = config.pointer("/models/providers/ollama")?;
    if provider["baseUrl"].as_str().map(base_url) != Some(base_url(server)) {
        return None;
    }
    let wanted = crate::storage::default_tag(model);
    provider["models"]
        .as_array()?
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .find(|id| crate::storage::default_tag(id) == wanted)
        .map(str::to_owned)
}

/// `agents.defaults.model.primary` is the model openclaw uses, and onboarding
/// sets it once. Without this, a later `--model` is ignored and `llmman serve`
/// loads whatever the first launch named. Only a model the provider lists.
fn with_primary_model(config: &serde_json::Value, server: &str, model: &str) -> serde_json::Value {
    use serde_json::{json, Map, Value};
    fn child<'a>(map: &'a mut Map<String, Value>, key: &str) -> Option<&'a mut Map<String, Value>> {
        map.entry(key).or_insert_with(|| json!({})).as_object_mut()
    }
    let mut config = config.clone();
    let Some(id) = listed_id(&config, server, model) else {
        return config;
    };
    let primary = format!("ollama/{id}");
    let defaults = config
        .as_object_mut()
        .and_then(|c| child(c, "agents"))
        .and_then(|a| child(a, "defaults"));
    if let Some(defaults) = defaults {
        match defaults.get_mut("model") {
            Some(Value::Object(m)) => {
                m.insert("primary".into(), primary.into());
            }
            // The older spelling, `"model": "ollama/..."`.
            Some(m @ Value::String(_)) => *m = primary.into(),
            _ => {
                defaults.insert("model".into(), json!({ "primary": primary }));
            }
        }
    }
    config
}

/// The model `config` uses, if it names one other than `wanted`.
fn other_primary_model<'a>(config: &'a serde_json::Value, wanted: &str) -> Option<&'a str> {
    let model = config.pointer("/agents/defaults/model")?;
    let primary = model.as_str().or_else(|| model["primary"].as_str())?;
    let id = primary.strip_prefix("ollama/").unwrap_or(primary);
    (crate::storage::default_tag(id) != crate::storage::default_tag(wanted)).then_some(primary)
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
    if let (false, Some(config)) = (model.is_empty(), onboarded_config()) {
        let (server, wanted) = (server(), openclaw_model_id(model));
        let mut kept = None;
        if let Err(e) = common::write_json_merged(&config, "openclaw", |existing| {
            if listed_id(existing, &server, wanted).is_none() {
                kept = other_primary_model(existing, wanted).map(str::to_owned);
            }
            with_primary_model(existing, &server, wanted)
        }) {
            eprintln!("[llmman] openclaw: primary model not set: {e:#}");
        }
        if let Some(kept) = kept {
            eprintln!("[llmman] openclaw: it has no {wanted} from this server, so it keeps {kept}");
        }
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
    fn a_listed_model_becomes_the_primary_one() {
        let entry = |id: &str| serde_json::json!({ "id": id, "contextWindow": 128000 });
        let mut config = onboarded(
            SERVER,
            serde_json::json!([entry("qwen3.8:27b-nvfp4"), entry("qwen3.8:27b")]),
        );
        config["agents"] = serde_json::json!({ "defaults": {
            "workspace": "/w", "model": { "primary": "ollama/qwen3.8:27b-nvfp4" },
        } });
        let updated = with_primary_model(&config, SERVER, "qwen3.8:27b");
        assert_eq!(
            updated["agents"]["defaults"]["model"]["primary"],
            "ollama/qwen3.8:27b"
        );
        assert_eq!(updated["agents"]["defaults"]["workspace"], "/w");
        // The id is the listed one, whichever spelling was asked for.
        let mut listed = onboarded(SERVER, serde_json::json!([entry("hf.co/a/b:latest")]));
        listed["agents"] = config["agents"].clone();
        assert_eq!(
            with_primary_model(&listed, &format!("{SERVER}/v1"), "hf.co/a/b")["agents"]["defaults"]
                ["model"]["primary"],
            "ollama/hf.co/a/b:latest"
        );
        // The older, plain-string spelling of the model.
        listed["agents"]["defaults"]["model"] = "ollama/other".into();
        assert_eq!(
            with_primary_model(&listed, SERVER, "hf.co/a/b")["agents"]["defaults"]["model"],
            "ollama/hf.co/a/b:latest"
        );
        // No `agents` section yet.
        let bare = onboarded(SERVER, serde_json::json!([entry("hf.co/a/b:latest")]));
        assert_eq!(
            with_primary_model(&bare, SERVER, "hf.co/a/b")["agents"]["defaults"]["model"]
                ["primary"],
            "ollama/hf.co/a/b:latest"
        );
    }

    #[test]
    fn an_unlisted_model_or_another_server_leaves_the_primary_model_alone() {
        let entry = serde_json::json!({ "id": "qwen3.8:27b", "contextWindow": 128000 });
        let mut config = onboarded(SERVER, serde_json::json!([entry]));
        config["agents"] = serde_json::json!({ "defaults": { "model": {
            "primary": "ollama/qwen3.8:27b",
        } } });
        assert_eq!(with_primary_model(&config, SERVER, "gemma4"), config);
        // A different primary, so the server check is what keeps it.
        let mut elsewhere = config.clone();
        elsewhere["agents"]["defaults"]["model"]["primary"] = "ollama/gemma4".into();
        assert_eq!(
            with_primary_model(&elsewhere, "http://127.0.0.1:11434", "qwen3.8:27b"),
            elsewhere
        );
        assert_eq!(
            with_primary_model(&serde_json::json!({}), SERVER, "x"),
            serde_json::json!({})
        );
        assert_eq!(
            other_primary_model(&config, "gemma4"),
            Some("ollama/qwen3.8:27b")
        );
        assert_eq!(other_primary_model(&config, "qwen3.8:27b"), None);
        // `:latest` is the default tag.
        let mut tagged = config.clone();
        tagged["agents"]["defaults"]["model"]["primary"] = "ollama/hf.co/a/b:latest".into();
        assert_eq!(other_primary_model(&tagged, "hf.co/a/b"), None);
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
