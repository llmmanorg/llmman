//! `llmman launch kimi` (Kimi Code CLI).

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::Context;

use super::{
    before_terminator, exec_with_env, exec_with_owned_env, find_on_path, has_flag, sandbox, server,
};

/// The window used when llmman knows none: the Python kimi-cli's own with
/// no model configured.
const KIMI_FALLBACK_CONTEXT_WINDOW: u64 = 100_000;

/// The Python kimi-cli's default `loop_control.reserved_context_size`.
const KIMI_RESERVED_CONTEXT_SIZE: u64 = 50_000;
/// The least the Python kimi-cli accepts for
/// `loop_control.reserved_context_size`.
const KIMI_MIN_RESERVED_CONTEXT_SIZE: u64 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KimiCli {
    /// The Python kimi-cli: takes the model as an inline `--config`.
    Legacy,
    /// Kimi Code: takes the model from `KIMI_MODEL_*`.
    Code,
}

static KIMI_CLI: OnceLock<KimiCli> = OnceLock::new();

/// Kimi Code's alias for the model `KIMI_MODEL_*` defines. Undocumented:
/// taken from its source.
const KIMI_CODE_ENV_MODEL: &str = "__kimi_env_model__";

/// kimi: with a model, passes it the way the installed CLI takes it;
/// without one, kimi uses its own configuration.
pub(super) fn launch_kimi(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("kimi").ok_or_else(|| anyhow::anyhow!("kimi is not installed"))?;
    let server = server();
    if model.is_empty() {
        return exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())]);
    }
    let base_url = format!("{server}/v1");
    match kimi_cli(&bin)? {
        KimiCli::Legacy => exec_with_env(
            &bin,
            &kimi_args(model, &base_url, context_window, extra_args),
            &kimi_env(&base_url, api_key),
        ),
        KimiCli::Code => {
            let env = kimi_code_env(model, &base_url, api_key, context_window);
            exec_with_owned_env(&bin, &kimi_code_args(extra_args), &env)
        }
    }
}

/// An image-based `--sandbox` runs the image's kimi, which the host
/// cannot probe, so it is taken to be Kimi Code.
fn kimi_cli(bin: &Path) -> anyhow::Result<KimiCli> {
    if let Some(cli) = KIMI_CLI.get() {
        return Ok(*cli);
    }
    let cli = if sandbox::runs_from_image() {
        KimiCli::Code
    } else {
        let output = Command::new(bin)
            .arg("--version")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .with_context(|| format!("failed to run {} --version", bin.display()))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        kimi_cli_from_version(&stdout).with_context(|| {
            format!(
                "cannot tell which kimi CLI {} is from its --version: {:?}",
                bin.display(),
                stdout.trim()
            )
        })?
    };
    Ok(*KIMI_CLI.get_or_init(|| cli))
}

/// The Python kimi-cli prints `kimi, version 1.x`; Kimi Code, a bare version.
fn kimi_cli_from_version(stdout: &str) -> Option<KimiCli> {
    let stdout = stdout.trim_start();
    if stdout.starts_with("kimi, version ") {
        Some(KimiCli::Legacy)
    } else if stdout.starts_with(|c: char| c.is_ascii_digit()) {
        Some(KimiCli::Code)
    } else {
        None
    }
}

fn kimi_args(
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> Vec<String> {
    [
        vec![
            "--config".to_string(),
            kimi_config(model, base_url, context_window),
        ],
        extra_args.to_vec(),
    ]
    .concat()
}

/// The Python kimi-cli's OpenAI provider prefers these to the
/// configuration's values, so they are set to override inherited ones.
/// This also keeps the key off the command line.
fn kimi_env<'a>(base_url: &'a str, api_key: &'a str) -> [(&'static str, &'a str); 2] {
    [("OPENAI_BASE_URL", base_url), ("OPENAI_API_KEY", api_key)]
}

/// The provider's key is empty: `OPENAI_API_KEY` supplies it.
fn kimi_config(model: &str, base_url: &str, context_window: Option<u64>) -> String {
    let window = context_window.unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW);
    serde_json::json!({
        "default_model": model,
        "models": {
            (model): {
                "provider": "llmman",
                "model": model,
                "max_context_size": window,
            },
        },
        "providers": {
            "llmman": {
                "type": "openai_legacy",
                "base_url": base_url,
                "api_key": "",
            },
        },
        "loop_control": { "reserved_context_size": reserved_context_size(window) },
    })
    .to_string()
}

/// The Python kimi-cli compacts when the context plus this reserve fills
/// the window, so its default would compact a small window at every step.
fn reserved_context_size(window: u64) -> u64 {
    (window / 4).clamp(KIMI_MIN_RESERVED_CONTEXT_SIZE, KIMI_RESERVED_CONTEXT_SIZE)
}

/// A resumed Kimi Code session keeps its own model unless `--model` names
/// one, so llmman's is named explicitly.
fn kimi_code_args(extra_args: &[String]) -> Vec<String> {
    [
        vec!["--model".to_string(), KIMI_CODE_ENV_MODEL.to_string()],
        extra_args.to_vec(),
    ]
    .concat()
}

/// Kimi Code builds the [`KIMI_CODE_ENV_MODEL`] model from these.
fn kimi_code_env(
    model: &str,
    base_url: &str,
    api_key: &str,
    context_window: Option<u64>,
) -> Vec<(&'static str, String)> {
    let window = context_window.unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW);
    vec![
        ("KIMI_MODEL_NAME", model.to_string()),
        ("KIMI_MODEL_PROVIDER_TYPE", "openai".to_string()),
        ("KIMI_MODEL_BASE_URL", base_url.to_string()),
        ("KIMI_MODEL_API_KEY", api_key.to_string()),
        ("KIMI_MODEL_MAX_CONTEXT_SIZE", window.to_string()),
    ]
}

/// Detects the installed CLI and refuses arguments that would replace
/// llmman's model or configuration.
pub(super) fn check_kimi_args(model: Option<&str>, extra_args: &[String]) -> anyhow::Result<()> {
    let Some(model) = model.map(str::trim).filter(|m| !m.is_empty()) else {
        return Ok(());
    };
    let bin = find_on_path("kimi").ok_or_else(|| anyhow::anyhow!("kimi is not installed"))?;
    check_args(kimi_cli(&bin)?, model, extra_args)
}

fn check_args(cli: KimiCli, model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let args = before_terminator(extra_args);
    // Kimi Code has no `--config`, and rejects one itself.
    if cli == KimiCli::Legacy {
        for flag in ["--config", "--config-file"] {
            anyhow::ensure!(
                !has_flag(args, flag, None),
                "kimi's {flag} would replace the configuration llmman passes for model {model}; \
                 launch kimi without a model to use its own"
            );
        }
    }
    anyhow::ensure!(
        !has_flag(args, "--model", None) && !args.iter().any(|a| a.starts_with("-m")),
        "kimi's --model would replace the model llmman passes; select it with \
         `llmman launch kimi --model <model>` instead"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn kimi_cli_from_version_tells_the_python_cli_from_kimi_code() {
        assert_eq!(
            kimi_cli_from_version("kimi, version 1.51.0\n"),
            Some(KimiCli::Legacy)
        );
        assert_eq!(kimi_cli_from_version("2.2.0\n"), Some(KimiCli::Code));
        assert_eq!(kimi_cli_from_version(""), None);
        assert_eq!(kimi_cli_from_version("usage: kimi"), None);
    }

    #[test]
    fn kimi_args_lead_with_llmmans_config() {
        let built = kimi_args("m", "http://h/v1", None, &args(&["--print", "-p", "hi"]));
        assert_eq!(built[0], "--config");
        assert_eq!(built[1], kimi_config("m", "http://h/v1", None));
        assert_eq!(built[2..], args(&["--print", "-p", "hi"]));
    }

    #[test]
    fn kimi_env_sets_the_base_url_and_key_kimi_prefers() {
        assert_eq!(
            kimi_env("http://h/v1", "k"),
            [("OPENAI_BASE_URL", "http://h/v1"), ("OPENAI_API_KEY", "k")]
        );
    }

    #[test]
    fn kimi_config_selects_the_model_at_llmmans_endpoint() {
        let config: serde_json::Value = serde_json::from_str(&kimi_config(
            "qwen3.5:0.8b",
            "http://127.0.0.1:11434/v1",
            Some(32_768),
        ))
        .unwrap();
        assert_eq!(config["default_model"], "qwen3.5:0.8b");
        let model = &config["models"]["qwen3.5:0.8b"];
        assert_eq!(model["provider"], "llmman");
        assert_eq!(model["model"], "qwen3.5:0.8b");
        assert_eq!(model["max_context_size"], 32_768);
        let provider = &config["providers"]["llmman"];
        assert_eq!(provider["type"], "openai_legacy");
        assert_eq!(provider["base_url"], "http://127.0.0.1:11434/v1");
        assert_eq!(provider["api_key"], "");
        assert_eq!(config["loop_control"]["reserved_context_size"], 8_192);
    }

    #[test]
    fn kimi_config_falls_back_to_kimis_window_when_there_is_none() {
        let config: serde_json::Value =
            serde_json::from_str(&kimi_config("m", "http://h/v1", None)).unwrap();
        assert_eq!(
            config["models"]["m"]["max_context_size"],
            KIMI_FALLBACK_CONTEXT_WINDOW
        );
    }

    #[test]
    fn reserved_context_size_stays_within_kimis_bounds() {
        assert_eq!(reserved_context_size(262_144), KIMI_RESERVED_CONTEXT_SIZE);
        assert_eq!(reserved_context_size(32_768), 8_192);
        assert_eq!(reserved_context_size(2_048), KIMI_MIN_RESERVED_CONTEXT_SIZE);
    }

    #[test]
    fn kimi_code_args_name_llmmans_model_for_a_resumed_session() {
        assert_eq!(
            kimi_code_args(&args(&["-r", "s", "-p", "hi"])),
            args(&["--model", "__kimi_env_model__", "-r", "s", "-p", "hi"])
        );
    }

    #[test]
    fn kimi_code_env_selects_the_model_at_llmmans_endpoint() {
        let env = kimi_code_env("m", "http://h/v1", "k", Some(32_768));
        assert_eq!(
            env,
            [
                ("KIMI_MODEL_NAME", "m".to_string()),
                ("KIMI_MODEL_PROVIDER_TYPE", "openai".to_string()),
                ("KIMI_MODEL_BASE_URL", "http://h/v1".to_string()),
                ("KIMI_MODEL_API_KEY", "k".to_string()),
                ("KIMI_MODEL_MAX_CONTEXT_SIZE", "32768".to_string()),
            ]
        );
        let fallback = kimi_code_env("m", "http://h/v1", "k", None);
        assert!(fallback.contains(&(
            "KIMI_MODEL_MAX_CONTEXT_SIZE",
            KIMI_FALLBACK_CONTEXT_WINDOW.to_string()
        )));
    }

    #[test]
    fn check_args_refuses_the_python_clis_own_config_and_model() {
        let err = |a: &[&str]| {
            check_args(KimiCli::Legacy, "m", &args(a))
                .unwrap_err()
                .to_string()
        };
        assert!(err(&["--config", "{}"]).contains("kimi's --config "));
        assert!(err(&["--config={}"]).contains("kimi's --config "));
        assert!(err(&["--config-file", "c.toml"]).contains("kimi's --config-file"));
        for model in [
            &["--model", "x"][..],
            &["--model=x"],
            &["-m", "x"],
            &["-m=x"],
            &["-mx"],
        ] {
            assert!(err(model).contains("kimi's --model"), "{model:?}");
        }
    }

    #[test]
    fn check_args_refuses_kimi_codes_model_but_leaves_config_to_it() {
        let code = |a: &[&str]| check_args(KimiCli::Code, "m", &args(a));
        assert!(code(&["-mx"]).is_err());
        assert!(code(&["--model", "x"]).is_err());
        assert!(code(&["--config", "{}"]).is_ok());
    }

    #[test]
    fn check_kimi_args_leaves_kimis_own_selection_alone_without_a_model() {
        for model in [None, Some(""), Some("  ")] {
            assert!(check_kimi_args(model, &args(&["--config", "{}", "-m", "x"])).is_ok());
        }
    }

    #[test]
    fn check_args_stops_at_a_forwarded_terminator() {
        for cli in [KimiCli::Legacy, KimiCli::Code] {
            assert!(check_args(cli, "m", &args(&["--", "--model", "x", "-mx"])).is_ok());
        }
    }
}
