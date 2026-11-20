//! `llmman launch vibe` (Mistral Vibe CLI).

use super::common::daemon_no_proxy_env;
use super::{exec_with_owned_env, find_on_path, server, HostedModel};

const VIBE_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// The alias of the model entry llmman defines.
const VIBE_MODEL_ALIAS: &str = "llmman";

/// vibe: configured through `VIBE_*` variables, so nothing is written to
/// disk. Without a model, Vibe uses its own configuration.
///
/// `hosted` is every model `--provider` serves (see [`HostedModel`]):
/// each becomes a `VIBE_MODELS` preset, so Vibe's own model picker can
/// switch between them mid-session; the launched model keeps the
/// `llmman` alias `VIBE_ACTIVE_MODEL` selects.
pub(super) fn launch_vibe(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    extra_args: &[String],
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let bin = find_on_path("vibe").ok_or_else(|| anyhow::anyhow!("vibe is not installed"))?;
    let server = server();
    let mut env = vibe_env(model, api_key, &server, context_window, hosted);
    if !model.is_empty() {
        env.extend(daemon_no_proxy_env(&server));
    }
    exec_with_owned_env(&bin, extra_args, &env)
}

/// Split out so a test can see what Vibe is handed.
fn vibe_env(
    model: &str,
    api_key: &str,
    server: &str,
    context_window: Option<u64>,
    hosted: Option<&[HostedModel]>,
) -> Vec<(&'static str, String)> {
    if model.is_empty() {
        return Vec::new();
    }
    let presets = vibe_model_presets(model, context_window, hosted);
    let provider = serde_json::json!({
        "name": "llmman",
        "api_base": format!("{server}/v1"),
        "api_key_env_var": VIBE_API_KEY_ENV,
        "api_style": "openai",
        "backend": "generic",
    });
    vec![
        ("VIBE_ACTIVE_MODEL", VIBE_MODEL_ALIAS.to_string()),
        ("VIBE_PROVIDERS", serde_json::json!([provider]).to_string()),
        ("VIBE_MODELS", serde_json::json!(presets).to_string()),
        (VIBE_API_KEY_ENV, api_key.to_string()),
    ]
}

/// The `VIBE_MODELS` presets: the one model this launch resolved, or —
/// under `--provider` — every model the provider serves, the launched
/// one first, each with the window the catalog says it holds. The
/// launched preset keeps the `llmman` alias; each other is selected by
/// the id the provider knows it by — the reference itself where that id
/// would collide with the launched alias.
fn vibe_model_presets(
    model: &str,
    context_window: Option<u64>,
    hosted: Option<&[HostedModel]>,
) -> Vec<serde_json::Value> {
    let preset = |name: &str, alias: &str, window: Option<u64>| {
        let mut preset = serde_json::json!({
            "name": name,
            "provider": "llmman",
            "alias": alias,
        });
        if let Some(window) = window {
            preset["max_context_length"] = window.into();
        }
        preset
    };
    let Some(hosted) = hosted else {
        return vec![preset(model, VIBE_MODEL_ALIAS, context_window)];
    };
    let mut presets: Vec<serde_json::Value> = hosted
        .iter()
        .map(|m| {
            if m.reference == model {
                preset(model, VIBE_MODEL_ALIAS, context_window)
            } else {
                let alias = if m.id == VIBE_MODEL_ALIAS {
                    m.reference.as_str()
                } else {
                    m.id.as_str()
                };
                preset(&m.reference, alias, m.context)
            }
        })
        .collect();
    // The launched model belongs in its own presets even if the list
    // somehow lacks it: Vibe is started on it.
    if !presets
        .iter()
        .any(|preset| preset["name"].as_str() == Some(model))
    {
        presets.insert(0, preset(model, VIBE_MODEL_ALIAS, context_window));
    }
    presets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<'a>(env: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    fn vibe_env_defines_the_model_at_llmmans_endpoint() {
        let env = vibe_env(
            "qwen3.5:0.8b",
            "k",
            "http://127.0.0.1:11434",
            Some(32_768),
            None,
        );
        assert_eq!(get(&env, "VIBE_ACTIVE_MODEL"), Some("llmman"));
        assert_eq!(get(&env, "LLMMAN_API_KEY"), Some("k"));
        let providers: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_PROVIDERS").unwrap()).unwrap();
        assert_eq!(providers[0]["api_base"], "http://127.0.0.1:11434/v1");
        assert_eq!(providers[0]["api_key_env_var"], "LLMMAN_API_KEY");
        assert_eq!(providers[0]["api_style"], "openai");
        assert_eq!(providers[0]["backend"], "generic");
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert_eq!(models[0]["name"], "qwen3.5:0.8b");
        assert_eq!(models[0]["provider"], providers[0]["name"]);
        assert_eq!(models[0]["alias"], "llmman");
        assert_eq!(models[0]["max_context_length"], 32_768);
    }

    #[test]
    fn vibe_env_leaves_the_context_window_to_vibe_when_unknown() {
        let env = vibe_env("m", "k", "http://h", None, None);
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert!(models[0].get("max_context_length").is_none());
    }

    #[test]
    fn vibe_env_without_a_model_leaves_vibes_own_configuration() {
        assert!(vibe_env("", "k", "http://h", Some(1), None).is_empty());
    }

    /// A `--provider` launch lists every model the provider serves, so
    /// Vibe's own model picker can switch between them without a
    /// relaunch: the launched model first, keeping the `llmman` alias
    /// `VIBE_ACTIVE_MODEL` selects, and each other model with the window
    /// its own catalog entry names — or none where it names none.
    #[test]
    fn vibe_env_lists_every_model_the_provider_serves() {
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
        let env = vibe_env(
            "llmman.provider/openrouter/qwen3-coder",
            "k",
            "http://h",
            Some(262_144),
            Some(&hosted),
        );
        assert_eq!(get(&env, "VIBE_ACTIVE_MODEL"), Some("llmman"));
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert_eq!(models.as_array().unwrap().len(), 2);
        assert_eq!(models[0]["name"], "llmman.provider/openrouter/qwen3-coder");
        assert_eq!(models[0]["provider"], "llmman");
        assert_eq!(models[0]["alias"], "llmman");
        assert_eq!(models[0]["max_context_length"], 262_144);
        assert_eq!(
            models[1]["name"],
            "llmman.provider/openrouter/llama-4-scout"
        );
        assert_eq!(models[1]["alias"], "llama-4-scout");
        assert!(models[1].get("max_context_length").is_none());

        // The launched model belongs in its own presets even if the
        // provider's list somehow lacks it, ahead of the rest.
        let others = vec![hosted.into_iter().nth(1).unwrap()];
        let env = vibe_env("m", "k", "http://h", Some(8_192), Some(&others));
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert_eq!(models.as_array().unwrap().len(), 2);
        assert_eq!(models[0]["name"], "m");
        assert_eq!(models[0]["alias"], "llmman");
        assert_eq!(models[0]["max_context_length"], 8_192);
        assert_eq!(
            models[1]["name"],
            "llmman.provider/openrouter/llama-4-scout"
        );
    }
}
