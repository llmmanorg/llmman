//! `llmman launch openclaw`.

use std::path::PathBuf;

use super::models::{ModelEntry, ModelList};
use super::{common, exec_with_env, find_on_path, run_with_env, server};
use crate::providers;

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

/// The provider onboarding writes, and the one openclaw signs in to without
/// a credential of its own (a provider of another name is refused as
/// unauthenticated). Every launch points it at the daemon.
const OPENCLAW_PROVIDER: &str = "ollama";

/// What openclaw is told a model costs, in US dollars per million tokens as
/// the catalog has it: nothing for a stored model, and for one whose price
/// the catalog does not publish.
fn openclaw_cost(cost: Option<crate::daemon::ModelCost>) -> serde_json::Value {
    let cost = cost.unwrap_or(crate::daemon::ModelCost {
        input: 0.0,
        output: 0.0,
        cache_read: None,
        cache_write: None,
        reasoning: None,
    });
    serde_json::json!({
        "input": cost.input,
        "output": cost.output,
        "cacheRead": cost.cache_read.unwrap_or(0.0),
        "cacheWrite": cost.cache_write.unwrap_or(0.0),
    })
}

/// One row as an openclaw model: the routed name as its id, the plain label as its name, and only
/// what is known (an absent key is left to openclaw). The window is both what it holds and
/// `contextTokens`, which openclaw otherwise caps at 32k for a local model.
fn openclaw_model(entry: &ModelEntry) -> serde_json::Value {
    let mut model = serde_json::json!({
        "id": entry.reference,
        "name": entry.label(),
        "input": if entry.vision { vec!["text", "image"] } else { vec!["text"] },
        "cost": openclaw_cost(entry.cost),
    });
    if let Some(window) = entry.context_window {
        model["contextWindow"] = window.into();
        model["contextTokens"] = window.into();
    }
    if let Some(output) = entry.max_output {
        model["maxTokens"] = output.into();
    }
    if let Some(thinking) = &entry.thinking {
        model["reasoning"] = (!thinking.is_empty()).into();
    }
    if let Some(tools) = entry.tools {
        model["compat"] = serde_json::json!({ "supportsTools": tools });
    }
    model
}

/// `config` with the `ollama` provider holding every model of the list at `server` and the first
/// one as primary, on every launch rather than only the onboarding one, so `/model` offers the
/// rest. The rest of the config is kept.
fn with_models(config: &serde_json::Value, server: &str, models: &ModelList) -> serde_json::Value {
    use serde_json::{json, Map, Value};
    fn child<'a>(map: &'a mut Map<String, Value>, key: &str) -> Option<&'a mut Map<String, Value>> {
        map.entry(key).or_insert_with(|| json!({})).as_object_mut()
    }
    let mut config = config.clone();
    let Some(start) = models.start() else {
        return config;
    };
    let rows: Vec<Value> = models.entries().iter().map(openclaw_model).collect();
    let providers = config
        .as_object_mut()
        .and_then(|c| child(c, "models"))
        .and_then(|m| child(m, "providers"));
    if let Some(providers) = providers {
        let provider = providers
            .entry(OPENCLAW_PROVIDER)
            .or_insert_with(|| json!({}));
        if let Some(provider) = provider.as_object_mut() {
            // What every integration holds for a model llmman serves. The
            // `ollama-local` marker onboarding writes would be taken for
            // a provider's key and forwarded to it.
            provider.insert("apiKey".into(), providers::PLACEHOLDER_API_KEY.into());
            provider.insert("baseUrl".into(), base_url(server).into());
            provider.insert("api".into(), "ollama".into());
            provider.insert("models".into(), rows.into());
        }
    }
    let primary = format!("{OPENCLAW_PROVIDER}/{}", start.reference);
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

/// The model openclaw last ran: its primary model, without the provider
/// prefix.
pub(super) fn remembered_model() -> Option<String> {
    let text = std::fs::read_to_string(onboarded_config()?).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let model = config.pointer("/agents/defaults/model")?;
    let primary = model.as_str().or_else(|| model["primary"].as_str())?;
    let provider_prefix = format!("{OPENCLAW_PROVIDER}/");
    Some(
        primary
            .strip_prefix(provider_prefix.as_str())
            .unwrap_or(primary)
            .to_string(),
    )
}

/// openclaw: runs its non-interactive onboarding (once) against our
/// /v1 endpoint, then on every launch writes a provider holding the whole
/// model list with the starting model as its primary one, and hands off to
/// it directly. The real gateway/TUI/channel-setup lifecycle a full setup
/// wizard also manages is left to openclaw's own defaults.
pub(super) fn launch_openclaw(
    model: &str,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin =
        find_on_path("openclaw").ok_or_else(|| anyhow::anyhow!("openclaw is not installed"))?;

    if onboarded_config().is_none() {
        // Onboarding verifies the model against the endpoint, which a
        // provider-routed reference is not for it to pull: a stored one
        // is, and with none it picks its own.
        let stored = models.entries().iter().find(|e| e.provider.is_none());
        let mut onboard = vec![
            "onboard",
            "--non-interactive",
            "--accept-risk",
            "--auth-choice",
            "ollama",
            "--custom-base-url",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        onboard.push(format!("{}/v1", server()));
        if let Some(stored) = stored {
            onboard.extend([
                "--custom-model-id".into(),
                openclaw_model_id(&stored.reference).into(),
            ]);
        }
        onboard.extend(["--skip-health", "--skip-channels", "--skip-skills"].map(String::from));
        // Through run_with_env so a --sandbox onboards the copy it runs.
        let code = run_with_env(&bin, &onboard, &[])?;
        anyhow::ensure!(code == 0, "openclaw onboarding failed");
    }
    if let (false, Some(config)) = (model.is_empty(), onboarded_config()) {
        let server = server();
        if let Err(e) = common::write_json_merged(&config, "openclaw", |existing| {
            with_models(existing, &server, models)
        }) {
            eprintln!("[llmman] openclaw: models not set: {e:#}");
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

    /// A hosted row carries the catalog's prices; a stored one, and one with
    /// no price published, none.
    #[test]
    fn a_hosted_row_carries_the_catalogs_prices_and_others_none() {
        let priced = ModelEntry {
            cost: Some(crate::daemon::ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: Some(0.3),
                cache_write: None,
                reasoning: None,
            }),
            ..ModelEntry::unlisted("llmman.provider/alpha/agent")
        };
        assert_eq!(
            openclaw_model(&priced)["cost"],
            serde_json::json!({ "input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 0.0 })
        );
        assert_eq!(
            openclaw_model(&ModelEntry::unlisted("qwen3.5:0.8b"))["cost"],
            serde_json::json!({ "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 })
        );
    }

    fn row(reference: &str, window: Option<u64>) -> ModelEntry {
        ModelEntry {
            context_window: window,
            vision: reference.contains("vision"),
            thinking: Some(vec!["none".into(), "high".into()]),
            tools: Some(true),
            ..ModelEntry::unlisted(reference)
        }
    }

    fn list() -> ModelList {
        ModelList::of(vec![
            row("llmman.provider/anthropic/claude-x", Some(200_000)),
            row("qwen3.5:0.8b", Some(32768)),
            row("docker.io/ai/vision-model", None),
        ])
    }

    #[test]
    fn the_provider_holds_every_model_and_the_first_is_primary() {
        let onboarded = serde_json::json!({
            "gateway": { "auth": { "token": "kept" } },
            "models": { "providers": { "ollama": { "baseUrl": "http://elsewhere:11434", "apiKey": "ollama-local" } } },
            "agents": { "defaults": { "workspace": "/w", "model": { "primary": "ollama/old" } } },
        });
        let updated = with_models(&onboarded, &format!("{SERVER}/v1"), &list());
        let provider = &updated["models"]["providers"]["ollama"];
        assert_eq!(provider["baseUrl"], SERVER);
        assert_eq!(provider["api"], "ollama");
        let rows = provider["models"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["id"], "llmman.provider/anthropic/claude-x");
        assert_eq!(rows[0]["name"], "anthropic/claude-x");
        assert_eq!(rows[0]["contextWindow"], 200_000);
        assert_eq!(rows[0]["contextTokens"], 200_000);
        assert_eq!(rows[1]["name"], "qwen3.5:0.8b");
        assert_eq!(rows[2]["input"], serde_json::json!(["text", "image"]));
        assert!(rows[2].get("contextWindow").is_none());
        assert_eq!(rows[0]["compat"]["supportsTools"], true);
        assert_eq!(
            updated["agents"]["defaults"]["model"]["primary"],
            "ollama/llmman.provider/anthropic/claude-x"
        );
        // Everything else stays, the provider's own credential included.
        assert_eq!(updated["agents"]["defaults"]["workspace"], "/w");
        assert_eq!(updated["gateway"]["auth"]["token"], "kept");
        assert_eq!(provider["apiKey"], "llmman");
        // A second launch on another model replaces the list and the primary.
        let again = with_models(
            &updated,
            SERVER,
            &ModelList::of(vec![row("qwen3.5:0.8b", Some(32768))]),
        );
        assert_eq!(
            again["models"]["providers"]["ollama"]["models"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            again["agents"]["defaults"]["model"]["primary"],
            "ollama/qwen3.5:0.8b"
        );
    }

    #[test]
    fn the_older_spelling_of_the_model_and_a_bare_config_are_handled() {
        let mut older = serde_json::json!({ "agents": { "defaults": { "model": "ollama/old" } } });
        older = with_models(&older, SERVER, &list());
        assert_eq!(
            older["agents"]["defaults"]["model"],
            "ollama/llmman.provider/anthropic/claude-x"
        );
        let bare = with_models(&serde_json::json!({}), SERVER, &list());
        assert_eq!(
            bare["agents"]["defaults"]["model"]["primary"],
            "ollama/llmman.provider/anthropic/claude-x"
        );
        // A launch with no model changes nothing.
        let config = serde_json::json!({ "a": 1 });
        assert_eq!(with_models(&config, SERVER, &ModelList::default()), config);
    }

    #[test]
    fn the_config_holds_no_credential_of_a_provider() {
        let text =
            serde_json::to_string(&with_models(&serde_json::json!({}), SERVER, &list())).unwrap();
        assert!(!text.contains("session") && !text.contains("sk-"));
    }
}
