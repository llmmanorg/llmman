#![recursion_limit = "256"]

pub mod cmd;
pub mod container;
pub mod daemon;
pub mod ffi;
pub mod fmt;
pub mod gguf;
pub mod harmony;
pub mod hf;
pub mod hostgpu;
pub mod llama_release;
pub mod modelpack;
pub mod oauth;
pub mod shortnames;
pub mod storage;
pub mod thinking;
pub mod webui;
pub mod xet_fetch;

use std::path::PathBuf;

/// Path to the local OCI store, from `LLMMAN_MODELS` (mirrors Ollama's
/// `OLLAMA_MODELS`) or else `~/.local/share/llmman/store`
/// (`%LOCALAPPDATA%\llmman\store` on Windows).
pub fn default_store() -> anyhow::Result<PathBuf> {
    if let Some(dir) = models_dir_from_env() {
        return Ok(dir);
    }
    #[cfg(not(target_os = "windows"))]
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
        .join(".local")
        .join("share");
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine local data directory"))?;
    Ok(base.join("llmman").join("store"))
}

fn models_dir_from_env() -> Option<PathBuf> {
    parse_env_path(std::env::var("LLMMAN_MODELS").ok().as_deref())
}

/// Split out from [`models_dir_from_env`] for testing without touching
/// the real environment. Blank values count as unset.
fn parse_env_path(value: Option<&str>) -> Option<PathBuf> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_path_returns_none_when_unset_or_blank() {
        assert_eq!(parse_env_path(None), None);
        assert_eq!(parse_env_path(Some("")), None);
        assert_eq!(parse_env_path(Some("   ")), None);
    }

    #[test]
    fn parse_env_path_trims_and_returns_the_given_path() {
        assert_eq!(
            parse_env_path(Some("  /custom/store  ")),
            Some(PathBuf::from("/custom/store"))
        );
    }
}
