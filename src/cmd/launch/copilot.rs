//! `llmman launch copilot` (and the `copilot-cli` alias).

use super::{exec_with_env, find_on_path, server};

/// copilot: passes COPILOT_PROVIDER_BASE_URL via env.
pub(super) fn launch_copilot(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin =
        find_on_path("gh").ok_or_else(|| anyhow::anyhow!("gh (GitHub CLI) is not installed"))?;

    let base_url = format!("{}/v1", server());
    let mut args = vec!["copilot".to_string()];
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    exec_with_env(&bin, &args, &[("COPILOT_PROVIDER_BASE_URL", &base_url)])
}
