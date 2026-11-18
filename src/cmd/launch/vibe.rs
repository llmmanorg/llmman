//! `llmman launch vibe` (Mistral Vibe CLI).

use super::common::daemon_no_proxy_env;
use super::{exec_with_owned_env, find_on_path, server};

const VIBE_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// The alias of the model entry llmman defines.
const VIBE_MODEL_ALIAS: &str = "llmman";

/// vibe: configured through `VIBE_*` variables, so nothing is written to
/// disk. Without a model, Vibe uses its own configuration.
pub(super) fn launch_vibe(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("vibe").ok_or_else(|| anyhow::anyhow!("vibe is not installed"))?;
    let server = server();
    let mut env = vibe_env(model, api_key, &server, context_window);
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
) -> Vec<(&'static str, String)> {
    if model.is_empty() {
        return Vec::new();
    }
    let mut preset = serde_json::json!({
        "name": model,
        "provider": "llmman",
        "alias": VIBE_MODEL_ALIAS,
    });
    if let Some(window) = context_window {
        preset["max_context_length"] = window.into();
    }
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
        ("VIBE_MODELS", serde_json::json!([preset]).to_string()),
        (VIBE_API_KEY_ENV, api_key.to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<'a>(env: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    fn vibe_env_defines_the_model_at_llmmans_endpoint() {
        let env = vibe_env("qwen3.5:0.8b", "k", "http://127.0.0.1:11434", Some(32_768));
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
        let env = vibe_env("m", "k", "http://h", None);
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert!(models[0].get("max_context_length").is_none());
    }

    #[test]
    fn vibe_env_without_a_model_leaves_vibes_own_configuration() {
        assert!(vibe_env("", "k", "http://h", Some(1)).is_empty());
    }
}
