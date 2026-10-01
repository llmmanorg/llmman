//! `llmman launch aider`.

use std::path::PathBuf;

use super::{exec_with_env, server};

/// aider: set OPENAI_API_KEY and OPENAI_BASE_URL.
pub(super) fn launch_aider(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let base_url = format!("{}/v1", server());
    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), format!("openai/{model}")]);
    }
    args.extend(["--openai-api-base".to_string(), base_url.clone()]);
    args.extend_from_slice(extra_args);

    exec_with_env(
        &PathBuf::from("aider"),
        &args,
        &[
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url.as_str()),
        ],
    )
}
