//! `llmman launch copilot` (and the `copilot-cli` alias).

use std::path::{Path, PathBuf};

use super::{exec_with_env_removing, find_on_path, has_forwarded_model_flag, server};

const COPILOT_ENV_TO_CLEAR: &[&str] = &[
    "COPILOT_PROVIDER_API_KEY_COMMAND",
    "COPILOT_PROVIDER_BEARER_TOKEN",
];

/// GitHub Copilot CLI's OpenAI-compatible BYOK mode, pointed at llmman.
/// A local-model launch does not require GitHub login or mutate the user's
/// normal Copilot provider registry.
pub(super) fn launch_copilot(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_copilot().ok_or_else(|| anyhow::anyhow!("copilot is not installed"))?;
    let base_url = format!("{}/v1", server());
    exec_with_env_removing(
        &bin,
        &copilot_args(model, extra_args),
        &copilot_env(model, api_key, &base_url),
        COPILOT_ENV_TO_CLEAR,
    )
}

/// The standalone CLI's own installer writes to `~/.local/bin`, which is
/// commonly absent from the current shell's PATH until it is restarted.
pub(super) fn find_copilot() -> Option<PathBuf> {
    find_on_path("copilot").or_else(|| copilot_fallback(&crate::config::home_dir()?))
}

fn copilot_fallback(home: &Path) -> Option<PathBuf> {
    let bin = if cfg!(windows) {
        "copilot.exe"
    } else {
        "copilot"
    };
    let candidate = home.join(".local").join("bin").join(bin);
    candidate.is_file().then_some(candidate)
}

/// A caller-supplied model after `--` is Copilot's explicit override.
fn copilot_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    if !model.is_empty() && !has_forwarded_model_flag("copilot", extra_args) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

fn copilot_env<'a>(model: &'a str, api_key: &'a str, base_url: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![
        ("COPILOT_PROVIDER_BASE_URL", base_url),
        ("COPILOT_PROVIDER_TYPE", "openai"),
        ("COPILOT_PROVIDER_API_KEY", api_key),
        ("COPILOT_MODEL", model),
        ("COPILOT_PROVIDER_WIRE_API", "responses"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_select_the_resolved_model_and_yield_to_a_forwarded_one() {
        assert_eq!(
            copilot_args("local/model", &["-p".into(), "hello".into()]),
            ["--model", "local/model", "-p", "hello"]
        );
        assert_eq!(
            copilot_args(
                "local/model",
                &["--model".into(), "other/model".into(), "-p".into()]
            ),
            ["--model", "other/model", "-p"]
        );
        assert_eq!(
            copilot_args("local/model", &["-m".into(), "not-a-model-flag".into()]),
            ["--model", "local/model", "-m", "not-a-model-flag"]
        );
    }

    #[test]
    fn env_points_at_llmman_and_carries_its_key() {
        let env = copilot_env("local/model", "secret", "http://127.0.0.1:17434/v1");
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(
            get("COPILOT_PROVIDER_BASE_URL"),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(get("COPILOT_PROVIDER_TYPE"), Some("openai"));
        assert_eq!(get("COPILOT_PROVIDER_API_KEY"), Some("secret"));
        assert_eq!(get("COPILOT_MODEL"), Some("local/model"));
        assert_eq!(get("COPILOT_PROVIDER_WIRE_API"), Some("responses"));
        assert_eq!(env.len(), 5, "only documented BYOK variables are set");
        assert_eq!(
            COPILOT_ENV_TO_CLEAR,
            [
                "COPILOT_PROVIDER_API_KEY_COMMAND",
                "COPILOT_PROVIDER_BEARER_TOKEN"
            ]
        );
    }

    #[test]
    fn fallback_finds_the_installers_target() {
        let home = super::super::test_temp_dir("copilot");
        let name = if cfg!(windows) {
            "copilot.exe"
        } else {
            "copilot"
        };
        let bin = home.join(".local").join("bin").join(name);
        std::fs::create_dir_all(bin.parent().unwrap()).unwrap();
        assert_eq!(copilot_fallback(&home), None);
        std::fs::write(&bin, b"stub").unwrap();
        assert_eq!(copilot_fallback(&home), Some(bin));
        std::fs::remove_dir_all(home).unwrap();
    }
}
