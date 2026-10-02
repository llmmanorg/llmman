//! `llmman launch copilot` (and the `copilot-cli` alias).

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::{exec_with_env, find_on_path, has_flag, server};

const EMPTY_PROVIDER_REGISTRY: &str = "{\"providers\":[],\"models\":[]}\n";

/// GitHub Copilot CLI's OpenAI-compatible BYOK mode, pointed at llmman.
/// Offline mode keeps a local-model launch from requiring GitHub login or
/// contacting GitHub services; the configured provider remains reachable.
pub(super) fn launch_copilot(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_copilot().ok_or_else(|| anyhow::anyhow!("copilot is not installed"))?;
    let base_url = format!("{}/v1", server());
    let providers_config = write_copilot_provider_config()?;
    let providers_config = providers_config.to_string_lossy();
    exec_with_env(
        &bin,
        &copilot_args(model, extra_args),
        &copilot_env(model, api_key, &base_url, &providers_config),
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
    if !model.is_empty() && !has_flag(extra_args, "--model", None) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// Copilot's user `providers.json` takes precedence over legacy BYOK
/// environment variables. Pointing it at an empty llmman-owned registry
/// keeps those variables authoritative without replacing the user's file.
fn write_copilot_provider_config() -> anyhow::Result<PathBuf> {
    let dir = crate::config::home_dir()
        .context("no home directory")?
        .join(".copilot");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("llmman-providers.json");
    write_copilot_provider_config_at(&path)?;
    Ok(path)
}

fn write_copilot_provider_config_at(path: &Path) -> anyhow::Result<()> {
    if std::fs::read_to_string(path).ok().as_deref() != Some(EMPTY_PROVIDER_REGISTRY) {
        crate::fsutil::write_atomic(path, EMPTY_PROVIDER_REGISTRY.as_bytes())
            .with_context(|| format!("write {}", path.display()))?;
    }
    Ok(())
}

fn copilot_env<'a>(
    model: &'a str,
    api_key: &'a str,
    base_url: &'a str,
    providers_config: &'a str,
) -> Vec<(&'a str, &'a str)> {
    vec![
        ("COPILOT_PROVIDER_BASE_URL", base_url),
        ("COPILOT_PROVIDER_TYPE", "openai"),
        ("COPILOT_PROVIDER_API_KEY", api_key),
        ("COPILOT_MODEL", model),
        ("COPILOT_OFFLINE", "true"),
        ("COPILOT_PROVIDERS_CONFIG", providers_config),
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
    }

    #[test]
    fn env_points_at_llmman_and_carries_its_key() {
        let env = copilot_env(
            "local/model",
            "secret",
            "http://127.0.0.1:17434/v1",
            "/home/me/.copilot/llmman-providers.json",
        );
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(
            get("COPILOT_PROVIDER_BASE_URL"),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(get("COPILOT_PROVIDER_TYPE"), Some("openai"));
        assert_eq!(get("COPILOT_PROVIDER_API_KEY"), Some("secret"));
        assert_eq!(get("COPILOT_MODEL"), Some("local/model"));
        assert_eq!(get("COPILOT_OFFLINE"), Some("true"));
        assert_eq!(
            get("COPILOT_PROVIDERS_CONFIG"),
            Some("/home/me/.copilot/llmman-providers.json")
        );
    }

    #[test]
    fn fallback_finds_the_installers_target() {
        let home = std::env::temp_dir().join(format!(
            "llmman-copilot-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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

    #[test]
    fn provider_registry_is_isolated_from_the_users_registry() {
        let home = std::env::temp_dir().join(format!(
            "llmman-copilot-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let user = home.join("providers.json");
        let ours = home.join("llmman-providers.json");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(&user, "user-owned\n").unwrap();
        write_copilot_provider_config_at(&ours).unwrap();
        assert_eq!(std::fs::read_to_string(&user).unwrap(), "user-owned\n");
        assert_eq!(
            std::fs::read_to_string(&ours).unwrap(),
            EMPTY_PROVIDER_REGISTRY
        );
        std::fs::remove_dir_all(home).unwrap();
    }
}
