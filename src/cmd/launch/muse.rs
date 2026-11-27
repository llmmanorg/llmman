//! `llmman launch muse` (Meta's Muse Code).
//!
//! Muse Code's `meta` provider speaks the Responses API to its base URL.
//! llmman sets that base URL to the daemon with `-c`, which lasts for the
//! run only, and passes the daemon's key as `META_API_KEY`, so no file is
//! written. The daemon also serves the catalog Muse Code fetches first
//! (`cmd::serve::muse`).

use super::common::{daemon_no_proxy_env, find_on_path_or_local_bin};
use super::{exec_with_owned_env, has_forwarded_model_flag, server};

/// Muse Code reads its `meta` provider's key from this ahead of a stored
/// key or a Meta login.
const MUSE_API_KEY_ENV: &str = "META_API_KEY";

/// Muse Code's subcommands, as `muse --help` lists them. A first argument
/// that is none of these is a flag or an interactive session's prompt.
const MUSE_SUBCOMMANDS: &[&str] = &[
    "resume",
    "exec",
    "config",
    "export",
    "trace",
    "model-profile",
    "skills",
    "voice",
    "plugins",
    "sandbox",
    "schema",
    "serve",
    "mxp",
    "attach",
    "acp",
    "session-message",
    "sessions",
    "mcp",
    "auth",
    "login",
    "logout",
    "usage",
    "init",
    "help",
];

/// The subcommands a launch runs besides the interactive session. Any other
/// is refused: most run no model or use the user's Meta account (`login`,
/// `usage`).
const LAUNCHED_SUBCOMMANDS: &[&str] = &["resume", "exec"];

/// Refuses a subcommand outside [`LAUNCHED_SUBCOMMANDS`], before the daemon
/// starts.
pub(super) fn check_muse_args(extra_args: &[String]) -> anyhow::Result<()> {
    match subcommand(extra_args) {
        Some(sub) if !LAUNCHED_SUBCOMMANDS.contains(&sub) => anyhow::bail!(
            "llmman launch muse runs Muse Code's interactive session, `resume` and `exec`, \
             not `muse {sub}`; run `muse {sub}` itself"
        ),
        _ => Ok(()),
    }
}

/// Starts Muse Code on `model` at the daemon. `effort` is the `--variant`
/// in Muse Code's spelling.
pub(super) fn launch_muse(
    model: &str,
    api_key: &str,
    effort: Option<&str>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path_or_local_bin("muse")
        .ok_or_else(|| anyhow::anyhow!("muse is not installed"))?;
    let server = server();
    let mut env = vec![(MUSE_API_KEY_ENV, api_key.to_string())];
    env.extend(daemon_no_proxy_env(&server));
    exec_with_owned_env(&bin, &muse_args(model, effort, &server, extra_args), &env)
}

/// `extra_args` with llmman's flags where Muse Code reads them: after
/// `exec`, else first (an interactive session, or `resume`). A `--model`
/// the caller forwarded replaces `model`.
fn muse_args(
    model: &str,
    effort: Option<&str>,
    server: &str,
    extra_args: &[String],
) -> Vec<String> {
    let mut ours = Vec::new();
    if !model.is_empty() && !has_forwarded_model_flag("muse", extra_args) {
        ours.extend(["--model".to_string(), model.to_string()]);
    }
    if let Some(effort) = effort {
        ours.extend(["--reasoning-effort".to_string(), effort.to_string()]);
    }
    // What `--base-url` sets, with the key sent as a bearer token.
    let endpoint = serde_json::json!({ "base_url": format!("{server}/v1"), "auth": "bearer" });
    ours.extend(["-c".to_string(), format!("endpoint_transport={endpoint}")]);
    // Muse Code sends telemetry to api.meta.ai whatever the base URL. With
    // it off, every request goes to the daemon.
    ours.extend(["-c".to_string(), "telemetry.enabled=false".to_string()]);

    match extra_args.split_first() {
        Some((first, rest)) if first == "exec" => {
            [vec![first.clone()], ours, rest.to_vec()].concat()
        }
        _ => [ours, extra_args.to_vec()].concat(),
    }
}

/// The subcommand `extra_args` runs, if its first argument names one.
fn subcommand(extra_args: &[String]) -> Option<&str> {
    let first = extra_args.first()?.as_str();
    MUSE_SUBCOMMANDS.contains(&first).then_some(first)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVER: &str = "http://127.0.0.1:11434";
    const ENDPOINT: &str =
        r#"endpoint_transport={"auth":"bearer","base_url":"http://127.0.0.1:11434/v1"}"#;

    fn args(extra: &[&str]) -> Vec<String> {
        extra.iter().map(|s| s.to_string()).collect()
    }

    fn ours() -> Vec<String> {
        args(&[
            "--model",
            "qwen3.5:0.8b",
            "-c",
            ENDPOINT,
            "-c",
            "telemetry.enabled=false",
        ])
    }

    #[test]
    fn a_session_takes_the_flags_first() {
        assert_eq!(muse_args("qwen3.5:0.8b", None, SERVER, &[]), ours());
        let extra = args(&["--yolo", "fix the build"]);
        assert_eq!(
            muse_args("qwen3.5:0.8b", None, SERVER, &extra),
            [ours(), extra].concat()
        );
    }

    #[test]
    fn resume_takes_the_flags_before_it() {
        let extra = args(&["resume", "--last"]);
        assert_eq!(
            muse_args("qwen3.5:0.8b", None, SERVER, &extra),
            [ours(), extra].concat()
        );
    }

    /// Muse Code parses flags before `exec` as its TUI's, and then refuses
    /// `exec`'s own.
    #[test]
    fn exec_takes_the_flags_after_it() {
        let extra = args(&["exec", "--json", "say pong"]);
        assert_eq!(
            muse_args("qwen3.5:0.8b", None, SERVER, &extra),
            [args(&["exec"]), ours(), args(&["--json", "say pong"])].concat()
        );
    }

    /// Muse Code adds `/responses` to the base URL.
    #[test]
    fn the_endpoint_is_the_daemons_v1_with_a_bearer() {
        let built = muse_args("m", None, SERVER, &[]);
        let setting = built
            .iter()
            .find_map(|a| a.strip_prefix("endpoint_transport="))
            .unwrap();
        let endpoint: serde_json::Value = serde_json::from_str(setting).unwrap();
        assert_eq!(endpoint["base_url"], "http://127.0.0.1:11434/v1");
        assert_eq!(endpoint["auth"], "bearer");
        assert!(built.contains(&"telemetry.enabled=false".to_string()));
    }

    #[test]
    fn a_variant_is_its_reasoning_effort() {
        let built = muse_args("m", Some("xhigh"), SERVER, &args(&["exec", "x"]));
        let at = built
            .iter()
            .position(|a| a == "--reasoning-effort")
            .unwrap();
        assert_eq!(built[at + 1], "xhigh");
        assert!(at > 0, "after exec: {built:?}");
    }

    /// Muse Code refuses two `--model`s.
    #[test]
    fn a_forwarded_model_wins() {
        let built = muse_args("m", None, SERVER, &args(&["exec", "--model", "other", "x"]));
        assert_eq!(built.iter().filter(|a| *a == "--model").count(), 1);
        assert!(built.windows(2).any(|w| w == ["--model", "other"]));
        // Muse Code has no `-m`.
        let built = muse_args("m", None, SERVER, &args(&["exec", "-m", "x"]));
        assert!(built.windows(2).any(|w| w == ["--model", "m"]));
        // No model: the endpoint is still given.
        let built = muse_args("", None, SERVER, &[]);
        assert!(!built.contains(&"--model".to_string()));
        assert!(built.iter().any(|a| a.starts_with("endpoint_transport=")));
    }

    #[test]
    fn the_key_is_never_an_argument() {
        let built = muse_args("m", Some("low"), SERVER, &args(&["exec", "x"]));
        assert!(!built
            .iter()
            .any(|a| a.contains("api_key") || a.contains("sk-")));
    }

    #[test]
    fn subcommands_other_than_resume_and_exec_are_refused() {
        for ok in [
            &[][..],
            &["resume"],
            &["exec", "x"],
            &["--yolo"],
            &["fix it"],
        ] {
            check_muse_args(&args(ok)).unwrap();
        }
        for sub in ["usage", "login", "acp", "serve", "sessions"] {
            let error = check_muse_args(&args(&[sub])).unwrap_err().to_string();
            assert!(
                error.contains(&format!("run `muse {sub}` itself")),
                "{error}"
            );
        }
    }
}
