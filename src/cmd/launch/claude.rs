//! `llmman launch claude`.

use super::{exec_with_env, find_on_path, server};

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
pub(super) fn launch_claude(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("claude").ok_or_else(|| anyhow::anyhow!("claude is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    let server = server();
    exec_with_env(
        &bin,
        &args,
        &[
            ("ANTHROPIC_BASE_URL", server.as_str()),
            ("ANTHROPIC_API_KEY", api_key),
        ],
    )
}
