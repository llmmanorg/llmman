//! `llmman launch kimi` (Kimi Code CLI).

use super::models::{write_transient, ModelEntry, ModelList};
use crate::providers;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::Context;

use super::common::read_if_exists;
use super::{
    before_terminator, env_dir_or_home, exec_with_env, exec_with_owned_env, find_on_path, has_flag,
    sandbox, server,
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

/// The provider the list is kept under in either CLI's configuration, and
/// the variable Kimi Code reads its key from: a name, never the key.
const KIMI_PROVIDER: &str = "llmman";
const KIMI_KEY_ENV: &str = "LLMMAN_API_KEY";

/// kimi: with a model, hands the CLI the list where it reads several models from: the Python
/// kimi-cli a `--config-file` of its own (removed when the launch ends), Kimi Code only
/// `$KIMI_CODE_HOME/config.toml`, into which llmman's provider and models are merged. Neither file
/// holds the key. With nothing to switch to (one model, all an `--sandbox openshell` kimi is given)
/// there is no file: an inline `--config` for the Python CLI, `KIMI_MODEL_*` for Kimi Code. Without
/// a model, kimi uses its own configuration and Kimi Code is given the list beside it, not as a
/// default.
pub(super) fn launch_kimi(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("kimi").ok_or_else(|| anyhow::anyhow!("kimi is not installed"))?;
    let server = server();
    let base_url = format!("{server}/v1");
    if model.is_empty() || models.is_empty() {
        // Its own configuration picks the model. Kimi Code is also given the
        // list, to switch to, which leaves that choice as it is.
        if models.is_empty() || kimi_cli(&bin)? == KimiCli::Legacy {
            return exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())]);
        }
        write_kimi_code_config(&kimi_code_home()?.join("config.toml"), models, &base_url)?;
        let mut env = kimi_code_env(api_key);
        env.push(("OLLAMA_HOST", server));
        return exec_with_owned_env(&bin, extra_args, &env);
    }
    let cli = kimi_cli(&bin)?;
    if cli == KimiCli::Legacy {
        legacy_takes(model)?;
    }
    if !models.offers_choice() {
        return match cli {
            KimiCli::Legacy => exec_with_env(
                &bin,
                &kimi_inline_args(model, &base_url, context_window, extra_args),
                &kimi_env(&base_url, api_key),
            ),
            KimiCli::Code => exec_with_owned_env(
                &bin,
                &kimi_code_args(KIMI_CODE_ENV_MODEL, extra_args),
                &kimi_code_single_env(model, &base_url, api_key, context_window),
            ),
        };
    }
    match cli {
        KimiCli::Legacy => {
            let models = models.stored_only("kimi", LEGACY_STORED_ONLY);
            let document = kimi_legacy_config(&models, &base_url)?.to_string();
            let path = write_transient(&kimi_share_dir()?, "kimi", "toml", &document)?;
            exec_with_env(
                &bin,
                &kimi_args(&path, extra_args),
                &kimi_env(&base_url, api_key),
            )
        }
        KimiCli::Code => {
            let path = kimi_code_home()?.join("config.toml");
            let start = write_kimi_code_config(&path, models, &base_url)?;
            exec_with_owned_env(
                &bin,
                &kimi_code_args(&start, extra_args),
                &kimi_code_env(api_key),
            )
        }
    }
}

const LEGACY_STORED_ONLY: &str = "the Python kimi-cli is end of life: its last release (1.52) \
     only prints that it is no longer maintained, so llmman cannot verify a hosted model with it";

/// The Python kimi-cli takes a stored model and nothing else: it is end of
/// life, and a hosted model could not be verified with it. Kimi Code is the
/// CLI to use for one.
fn legacy_takes(model: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !providers::is_remote_ref(model),
        "a hosted model does not work with the Python kimi-cli: {LEGACY_STORED_ONLY}.\n\
         Install Kimi Code (https://code.kimi.com/kimi-code/install.sh), or start on a pulled \
         model with --model."
    );
    Ok(())
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

/// The Python kimi-cli's own file, which replaces its configuration for
/// this launch as an inline `--config` did.
fn kimi_args(config: &Path, extra_args: &[String]) -> Vec<String> {
    [
        vec![
            "--config-file".to_string(),
            config.to_string_lossy().into_owned(),
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

/// Where the Python kimi-cli keeps its state: `$KIMI_SHARE_DIR`, else `~/.kimi`.
pub(super) fn kimi_share_dir() -> anyhow::Result<PathBuf> {
    env_dir_or_home("KIMI_SHARE_DIR", ".kimi")
}

/// Where Kimi Code keeps its state: `$KIMI_CODE_HOME`, else `~/.kimi-code`
/// on every platform.
pub(super) fn kimi_code_home() -> anyhow::Result<PathBuf> {
    env_dir_or_home("KIMI_CODE_HOME", ".kimi-code")
}

/// The Python kimi-cli's whole configuration: the list at llmman's
/// endpoint, starting on the first row. The provider's key is empty:
/// `OPENAI_API_KEY` supplies it.
fn kimi_legacy_config(
    models: &ModelList,
    base_url: &str,
) -> anyhow::Result<toml_edit::DocumentMut> {
    let mut document = toml_edit::DocumentMut::new();
    let aliases = add_kimi_models(KimiCli::Legacy, &mut document, models, base_url)?;
    if let Some(start) = aliases.first() {
        document.insert("default_model", toml_edit::value(start.as_str()));
    }
    let smallest = models
        .entries()
        .iter()
        .map(kimi_window)
        .min()
        .unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW);
    let mut loop_control = toml_edit::Table::new();
    loop_control["reserved_context_size"] =
        toml_edit::value(reserved_context_size(smallest) as i64);
    document.insert("loop_control", toml_edit::Item::Table(loop_control));
    Ok(document)
}

/// llmman's provider and one model per row of `models` in `document`, replacing only what a launch
/// left there; returns each row's alias, in order.
fn add_kimi_models(
    cli: KimiCli,
    document: &mut toml_edit::DocumentMut,
    models: &ModelList,
    base_url: &str,
) -> anyhow::Result<Vec<String>> {
    let aliases = models.unique_keys(|e| format!("{KIMI_PROVIDER}/{}", e.label()));
    let names = models.display_names();

    let mut provider = toml_edit::Table::new();
    match cli {
        KimiCli::Legacy => {
            provider["type"] = toml_edit::value("openai_legacy");
            provider["api_key"] = toml_edit::value("");
        }
        KimiCli::Code => {
            provider["type"] = toml_edit::value("openai");
            provider["api_key_env"] = toml_edit::value(KIMI_KEY_ENV);
        }
    }
    provider["base_url"] = toml_edit::value(base_url);
    let providers = child_table(document.as_table_mut(), "providers")?;
    providers.insert(KIMI_PROVIDER, toml_edit::Item::Table(provider));

    let table = child_table(document.as_table_mut(), "models")?;
    let stale: Vec<String> = table
        .iter()
        .filter(|(_, model)| model.get("provider").and_then(|p| p.as_str()) == Some(KIMI_PROVIDER))
        .map(|(alias, _)| alias.to_string())
        .collect();
    for alias in stale {
        table.remove(&alias);
    }
    for ((alias, name), entry) in aliases.iter().zip(&names).zip(models.entries()) {
        table.insert(alias, toml_edit::Item::Table(kimi_model(cli, entry, name)));
    }
    Ok(aliases)
}

/// The table `key` of `parent`, made if there is none and without a
/// `[key]` header of its own.
fn child_table<'a>(
    parent: &'a mut toml_edit::Table,
    key: &str,
) -> anyhow::Result<&'a mut toml_edit::Table> {
    let item = parent.entry(key).or_insert_with(|| {
        let mut table = toml_edit::Table::new();
        table.set_implicit(true);
        toml_edit::Item::Table(table)
    });
    if item.is_inline_table() {
        let table = std::mem::take(item).into_table();
        *item =
            toml_edit::Item::Table(table.map_err(|_| anyhow::anyhow!("`{key}` is not a table"))?);
    }
    item.as_table_mut()
        .with_context(|| format!("`{key}` is not a table"))
}

fn kimi_window(entry: &ModelEntry) -> u64 {
    entry.context_window.unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW)
}

/// The thinking levels Kimi Code is told of, and whether thinking can be turned off, from the row's
/// variants: `none` is off, `thinking` goes as `medium`, any other level as is.
fn kimi_efforts(entry: &ModelEntry) -> (Vec<String>, bool) {
    let variants = entry.variants();
    let off = variants.iter().any(|v| v == "none");
    let mut levels: Vec<String> = Vec::new();
    for variant in variants.iter().filter(|v| *v != "none") {
        let level = if variant == "thinking" {
            "medium"
        } else {
            variant
        };
        if !levels.iter().any(|l| l == level) {
            levels.push(level.to_string());
        }
    }
    (levels, off)
}

/// The one mapping of a row to a model of either CLI; `model` is the `reference`, which the daemon
/// routes. Kimi Code takes the thinking levels and tool use from here, the Python kimi-cli only
/// whether the model thinks; neither has a field for the output limit.
fn kimi_model(cli: KimiCli, entry: &ModelEntry, name: &str) -> toml_edit::Table {
    let (levels, off) = kimi_efforts(entry);
    let mut capabilities = Vec::new();
    if cli == KimiCli::Code && entry.tools != Some(false) {
        capabilities.push("tool_use");
    }
    if entry.vision {
        capabilities.push("image_in");
    }
    if !levels.is_empty() {
        capabilities.push("thinking");
    }

    let mut model = toml_edit::Table::new();
    model["provider"] = toml_edit::value(KIMI_PROVIDER);
    model["model"] = toml_edit::value(entry.reference.as_str());
    model["max_context_size"] = toml_edit::value(kimi_window(entry) as i64);
    model["display_name"] = toml_edit::value(name);
    model["capabilities"] =
        toml_edit::value(capabilities.into_iter().collect::<toml_edit::Array>());
    if cli == KimiCli::Code && !levels.is_empty() {
        model["support_efforts"] = toml_edit::value(levels.iter().collect::<toml_edit::Array>());
        if off {
            model["off_effort"] = toml_edit::value("none");
        }
    }
    model
}

/// The Python kimi-cli compacts when the context plus this reserve fills
/// the window, so its default would compact a small window at every step;
/// the smallest window of the list sets it.
fn reserved_context_size(window: u64) -> u64 {
    (window / 4).clamp(KIMI_MIN_RESERVED_CONTEXT_SIZE, KIMI_RESERVED_CONTEXT_SIZE)
}

/// Adds llmman's provider and models to Kimi Code's `config.toml` at `path`, created if absent and
/// otherwise kept as is (comments included). A file that is not TOML is left alone. Returns the
/// alias of the starting model.
fn write_kimi_code_config(
    path: &Path,
    models: &ModelList,
    base_url: &str,
) -> anyhow::Result<String> {
    let existing = read_if_exists(path)?;
    let mut document: toml_edit::DocumentMut = existing
        .as_deref()
        .unwrap_or_default()
        .parse()
        .with_context(|| {
            format!(
                "{} is not valid TOML; llmman does not touch it",
                path.display()
            )
        })?;
    let aliases = add_kimi_models(KimiCli::Code, &mut document, models, base_url)?;
    let start = aliases.into_iter().next().context("no model to start on")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let text = document.to_string();
    match existing {
        Some(_) => crate::fsutil::write_atomic(path, text.as_bytes()),
        None => crate::fsutil::write_private(path, text.as_bytes()),
    }
    .with_context(|| format!("write {}", path.display()))?;
    Ok(start)
}

/// A resumed Kimi Code session keeps its own model unless `--model` names
/// one, so llmman's starting model is named explicitly.
fn kimi_code_args(start: &str, extra_args: &[String]) -> Vec<String> {
    [
        vec!["--model".to_string(), start.to_string()],
        extra_args.to_vec(),
    ]
    .concat()
}

/// What the provider's `api_key_env` reads.
fn kimi_code_env(api_key: &str) -> Vec<(&'static str, String)> {
    vec![(KIMI_KEY_ENV, api_key.to_string())]
}

/// Kimi Code's alias for the model `KIMI_MODEL_*` defines. Undocumented:
/// taken from its source.
const KIMI_CODE_ENV_MODEL: &str = "__kimi_env_model__";

/// Kimi Code builds the [`KIMI_CODE_ENV_MODEL`] model from these.
fn kimi_code_single_env(
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

/// The Python kimi-cli with one model: it as an inline `--config`.
fn kimi_inline_args(
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> Vec<String> {
    [
        vec![
            "--config".to_string(),
            kimi_inline_config(model, base_url, context_window),
        ],
        extra_args.to_vec(),
    ]
    .concat()
}

/// The provider's key is empty: `OPENAI_API_KEY` supplies it.
fn kimi_inline_config(model: &str, base_url: &str, context_window: Option<u64>) -> String {
    let window = context_window.unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW);
    serde_json::json!({
        "default_model": model,
        "models": {
            (model): {
                "provider": KIMI_PROVIDER,
                "model": model,
                "max_context_size": window,
            },
        },
        "providers": {
            KIMI_PROVIDER: {
                "type": "openai_legacy",
                "base_url": base_url,
                "api_key": "",
            },
        },
        "loop_control": { "reserved_context_size": reserved_context_size(window) },
    })
    .to_string()
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

    /// One model, no file: what an `--sandbox openshell` kimi, which sees
    /// none, is given.
    #[test]
    fn one_model_is_passed_inline_or_by_environment_with_no_file() {
        let built = kimi_inline_args("m", "http://h/v1", None, &args(&["--print", "-p", "hi"]));
        assert_eq!(built[0], "--config");
        assert_eq!(built[1], kimi_inline_config("m", "http://h/v1", None));
        assert_eq!(built[2..], args(&["--print", "-p", "hi"]));
        let config: serde_json::Value = serde_json::from_str(&built[1]).unwrap();
        assert_eq!(config["default_model"], "m");
        assert_eq!(config["models"]["m"]["provider"], "llmman");
        assert_eq!(config["models"]["m"]["max_context_size"], 100_000);
        assert_eq!(config["providers"]["llmman"]["api_key"], "");

        let env = kimi_code_single_env("qwen3.5:0.8b", "http://h/v1", "k", Some(32_768));
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("KIMI_MODEL_NAME"), Some("qwen3.5:0.8b"));
        assert_eq!(get("KIMI_MODEL_BASE_URL"), Some("http://h/v1"));
        assert_eq!(get("KIMI_MODEL_API_KEY"), Some("k"));
        assert_eq!(get("KIMI_MODEL_MAX_CONTEXT_SIZE"), Some("32768"));
        assert_eq!(
            kimi_code_args(KIMI_CODE_ENV_MODEL, &[]),
            ["--model", "__kimi_env_model__"]
        );
    }

    #[test]
    fn the_python_kimi_cli_takes_a_stored_model_and_no_hosted_one() {
        assert!(legacy_takes("docker.io/ai/qwen3.5:0.8b").is_ok());
        let err = legacy_takes("llmman.provider/anthropic/claude-haiku-4-5")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Kimi Code") && err.contains("end of life"),
            "{err}"
        );
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

    fn hosted(provider: &str, id: &str) -> ModelEntry {
        ModelEntry::unlisted(&format!("llmman.provider/{provider}/{id}"))
    }

    fn list() -> ModelList {
        let mut local = ModelEntry::unlisted("qwen3.5:0.8b");
        local.context_window = Some(32_768);
        local.thinking = Some(vec!["none".into(), "thinking".into()]);
        local.tools = Some(true);
        let mut agent = hosted("alpha", "agent");
        agent.context_window = Some(200_000);
        agent.vision = true;
        agent.thinking = Some(vec!["low".into(), "high".into()]);
        let mut plain = hosted("beta", "agent");
        plain.thinking = Some(Vec::new());
        plain.tools = Some(false);
        ModelList::of(vec![local, agent, plain])
    }

    fn parsed(document: &toml_edit::DocumentMut) -> toml::Value {
        document.to_string().parse().expect("valid TOML")
    }

    #[test]
    fn kimi_args_name_the_transient_config_file_ahead_of_the_callers() {
        let built = kimi_args(
            Path::new("/s/llmman-kimi-1.toml"),
            &args(&["--print", "-p", "hi"]),
        );
        assert_eq!(
            built,
            args(&[
                "--config-file",
                "/s/llmman-kimi-1.toml",
                "--print",
                "-p",
                "hi"
            ])
        );
    }

    #[test]
    fn kimi_env_sets_the_base_url_and_key_kimi_prefers() {
        assert_eq!(
            kimi_env("http://h/v1", "k"),
            [("OPENAI_BASE_URL", "http://h/v1"), ("OPENAI_API_KEY", "k")]
        );
    }

    #[test]
    fn the_python_cli_gets_every_row_starting_on_the_first() {
        let doc = kimi_legacy_config(&list(), "http://127.0.0.1:11434/v1").unwrap();
        let config = parsed(&doc);
        assert_eq!(
            config["default_model"].as_str(),
            Some("llmman/qwen3.5:0.8b")
        );
        let models = config["models"].as_table().unwrap();
        // The picker lists them in the file's order, which `toml` does not keep.
        let text = doc.to_string();
        let at = |alias: &str| text.find(&format!("[models.\"{alias}\"]")).unwrap();
        assert!(at("llmman/qwen3.5:0.8b") < at("llmman/alpha/agent"));
        assert!(at("llmman/alpha/agent") < at("llmman/beta/agent"));
        assert_eq!(models.len(), 3);
        let agent = &models["llmman/alpha/agent"];
        assert_eq!(agent["provider"].as_str(), Some("llmman"));
        assert_eq!(agent["model"].as_str(), Some("llmman.provider/alpha/agent"));
        assert_eq!(agent["max_context_size"].as_integer(), Some(200_000));
        assert_eq!(agent["display_name"].as_str(), Some("alpha/agent"));
        let caps = |m: &toml::Value| -> Vec<String> {
            m["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(caps(agent), ["image_in", "thinking"]);
        assert_eq!(caps(&models["llmman/qwen3.5:0.8b"]), ["thinking"]);
        assert!(caps(&models["llmman/beta/agent"]).is_empty());
        let provider = &config["providers"]["llmman"];
        assert_eq!(provider["type"].as_str(), Some("openai_legacy"));
        assert_eq!(
            provider["base_url"].as_str(),
            Some("http://127.0.0.1:11434/v1")
        );
        assert_eq!(provider["api_key"].as_str(), Some(""));
        // The smallest window is the 100k fallback of the row that has none.
        assert_eq!(
            config["loop_control"]["reserved_context_size"].as_integer(),
            Some(8_192)
        );
    }

    #[test]
    fn a_row_without_a_window_gets_kimis_fallback() {
        let list = ModelList::of(vec![ModelEntry::unlisted("m")]);
        let config = parsed(&kimi_legacy_config(&list, "http://h/v1").unwrap());
        assert_eq!(
            config["models"]["llmman/m"]["max_context_size"].as_integer(),
            Some(KIMI_FALLBACK_CONTEXT_WINDOW as i64)
        );
    }

    #[test]
    fn reserved_context_size_stays_within_kimis_bounds() {
        assert_eq!(reserved_context_size(262_144), KIMI_RESERVED_CONTEXT_SIZE);
        assert_eq!(reserved_context_size(32_768), 8_192);
        assert_eq!(reserved_context_size(2_048), KIMI_MIN_RESERVED_CONTEXT_SIZE);
    }

    #[test]
    fn twins_get_their_own_alias_and_the_provider_in_the_name() {
        let list = ModelList::of(vec![
            hosted("a", "m"),
            hosted("b", "m"),
            hosted("c", "solo"),
        ]);
        let doc = kimi_legacy_config(&list, "http://h/v1").unwrap();
        let config = parsed(&doc);
        let models = config["models"].as_table().unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models["llmman/a/m"]["display_name"].as_str(), Some("a/m"));
        assert_eq!(models["llmman/b/m"]["display_name"].as_str(), Some("b/m"));
        assert_eq!(
            models["llmman/c/solo"]["display_name"].as_str(),
            Some("solo")
        );
    }

    #[test]
    fn kimi_code_gets_tools_levels_and_images_it_would_not_guess() {
        let mut document = toml_edit::DocumentMut::new();
        let aliases =
            add_kimi_models(KimiCli::Code, &mut document, &list(), "http://h/v1").unwrap();
        assert_eq!(aliases[0], "llmman/qwen3.5:0.8b");
        let config = parsed(&document);
        let provider = &config["providers"]["llmman"];
        assert_eq!(provider["type"].as_str(), Some("openai"));
        assert_eq!(provider["api_key_env"].as_str(), Some("LLMMAN_API_KEY"));
        assert!(provider.get("api_key").is_none());
        let models = &config["models"];
        let local = &models["llmman/qwen3.5:0.8b"];
        assert_eq!(
            local["capabilities"].as_array().unwrap(),
            &["tool_use".into(), "thinking".into()]
        );
        // `thinking` goes as `medium`; `none` is the off effort.
        assert_eq!(
            local["support_efforts"].as_array().unwrap(),
            &["medium".into()]
        );
        assert_eq!(local["off_effort"].as_str(), Some("none"));
        let agent = &models["llmman/alpha/agent"];
        assert_eq!(
            agent["capabilities"].as_array().unwrap(),
            &["tool_use".into(), "image_in".into(), "thinking".into()]
        );
        assert_eq!(
            agent["support_efforts"].as_array().unwrap(),
            &["low".into(), "high".into()]
        );
        assert!(agent.get("off_effort").is_none());
        let plain = &models["llmman/beta/agent"];
        assert!(plain["capabilities"].as_array().unwrap().is_empty());
        assert!(plain.get("support_efforts").is_none());
    }

    #[test]
    fn kimi_code_config_keeps_the_users_content_and_replaces_only_llmmans() {
        let dir = super::super::test_temp_dir("kimi-code");
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "# mine\ndefault_model = \"kimi-code/k3\"\n\n[providers.mine]\ntype = \"openai\"\napi_key = \"keep-me\"\n\n\
             [models.mine]\nprovider = \"mine\"\nmodel = \"m\"\nmax_context_size = 1000\n\n\
             [models.\"llmman/gone\"]\nprovider = \"llmman\"\nmodel = \"gone\"\nmax_context_size = 1\n",
        )
        .unwrap();
        let start = write_kimi_code_config(&path, &list(), "http://h/v1").unwrap();
        assert_eq!(start, "llmman/qwen3.5:0.8b");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# mine\n"), "{text}");
        let config: toml::Value = text.parse().unwrap();
        assert_eq!(config["default_model"].as_str(), Some("kimi-code/k3"));
        assert_eq!(
            config["providers"]["mine"]["api_key"].as_str(),
            Some("keep-me")
        );
        assert_eq!(
            config["models"]["mine"]["max_context_size"].as_integer(),
            Some(1000)
        );
        assert!(config["models"].get("llmman/gone").is_none());
        assert_eq!(config["models"].as_table().unwrap().len(), 4);
        assert!(!text.contains("tok-"), "no key is written");

        // Again: the same rows, not twice.
        write_kimi_code_config(&path, &list(), "http://other/v1").unwrap();
        let again: toml::Value = std::fs::read_to_string(&path).unwrap().parse().unwrap();
        assert_eq!(again["models"].as_table().unwrap().len(), 4);
        assert_eq!(
            again["providers"]["llmman"]["base_url"].as_str(),
            Some("http://other/v1")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn kimi_code_config_is_created_and_a_broken_one_is_refused() {
        let dir = super::super::test_temp_dir("kimi-code-new");
        let path = dir.join("home").join("config.toml");
        write_kimi_code_config(&path, &list(), "http://h/v1").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        std::fs::write(&path, "models = [").unwrap();
        let err = write_kimi_code_config(&path, &list(), "http://h/v1")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not valid TOML"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "models = [");
        // An inline `providers` table is merged into, not replaced.
        std::fs::write(&path, "providers = { mine = { type = \"openai\" } }\n").unwrap();
        write_kimi_code_config(&path, &list(), "http://h/v1").unwrap();
        let config: toml::Value = std::fs::read_to_string(&path).unwrap().parse().unwrap();
        assert_eq!(config["providers"]["mine"]["type"].as_str(), Some("openai"));
        assert!(config["providers"].get("llmman").is_some());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn kimi_code_args_name_llmmans_model_for_a_resumed_session() {
        assert_eq!(
            kimi_code_args("llmman/m", &args(&["-r", "s", "-p", "hi"])),
            args(&["--model", "llmman/m", "-r", "s", "-p", "hi"])
        );
    }

    #[test]
    fn kimi_code_env_carries_only_the_key_the_provider_names() {
        assert_eq!(
            kimi_code_env("tok"),
            [("LLMMAN_API_KEY", "tok".to_string())]
        );
    }

    #[test]
    fn the_python_clis_file_is_transient_and_holds_no_key() {
        let dir = super::super::test_temp_dir("kimi-legacy");
        let document = kimi_legacy_config(&list(), "http://h/v1")
            .unwrap()
            .to_string();
        let path = write_transient(&dir, "kimi", "toml", &document).unwrap();
        assert_eq!(path.parent(), Some(dir.as_path()));
        assert!(!std::fs::read_to_string(&path)
            .unwrap()
            .contains("tok-secret"));
        super::super::models::finish_file(&path);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(dir);
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
