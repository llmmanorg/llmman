//! `llmman launch docker-agent` (Docker Agent).

use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::daemon;

use super::common;
use super::{exec_with_env, find_on_path, has_flag, server};

/// The env var the generated `token_key` names, so no key reaches disk
/// (same role as `qwen::QWEN_ENV_KEY` and `dsh::DSH_API_KEY_ENV`).
const DOCKER_AGENT_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// The generated model entry's name, which its `root` agent selects it by.
const DOCKER_AGENT_MODEL_NAME: &str = "llmman";

/// docker-agent: write an agent file whose single `openai`-provider model
/// points at this daemon, then hand that file to `docker-agent run`.
///
/// An agent file, not docker-agent's own `~/.config/cagent/config.yaml`:
/// a `models:` map there is ignored, and the entry cannot then be
/// selected. Writing one llmman owns also leaves `~/.config/cagent`
/// alone.
pub(super) fn launch_docker_agent(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    // `run` has already rejected these arguments before starting the
    // daemon; repeated so a direct call to `launch` rejects them too.
    check_docker_agent_args(extra_args)?;
    let bin = find_docker_agent().ok_or_else(|| {
        anyhow::anyhow!(
            "docker-agent is not installed\n\n\
             llmman looks for it on PATH and in ~/.docker/cli-plugins, where Docker Desktop \
             and `brew install docker-agent` put it.\n\
             Releases: https://github.com/docker/docker-agent/releases"
        )
    })?;

    let path = docker_agent_agent_file(&docker_agent_config_dir()?, model);
    write_docker_agent_file(&path, model, &format!("{}/v1", server()))?;

    let args = docker_agent_args(&path, extra_args);
    exec_with_env(&bin, &args, &[(DOCKER_AGENT_API_KEY_ENV, api_key)])
}

/// Rejects the arguments after `--` that docker-agent cannot be launched
/// with: its own `--model`, and a second agent file. Neither needs the
/// daemon to decide, so `run` calls this before `ensure_server` and the
/// refusal costs no daemon start and no model pull.
pub(super) fn check_docker_agent_args(extra_args: &[String]) -> anyhow::Result<()> {
    // Refused, not warned about: `--model` replaces the generated entry
    // including its `base_url`, so the request goes to api.openai.com —
    // carrying the real key under `--provider`.
    //
    // No `-m`: `run` has no such shorthand (`-a`, `-s`, `-w`, `-d`,
    // `-o`, `-h`), so matching it would refuse an unrelated argument.
    //
    // Only up to a forwarded `--`: Cobra takes everything after one as
    // arguments, so a prompt there that reads like a flag is a message.
    let flags = &extra_args[..extra_args
        .iter()
        .position(|arg| arg == "--")
        .unwrap_or(extra_args.len())];
    if has_flag(flags, "--model", None) {
        anyhow::bail!(
            "docker-agent's --model would replace the endpoint llmman configured and send the \
             request to api.openai.com; select the model with `llmman launch docker-agent \
             --model <model>` instead"
        );
    }
    if let Some(file) = docker_agent_config_argument(extra_args) {
        anyhow::bail!(docker_agent_own_agent_file_error(file));
    }
    Ok(())
}

/// Split from `check_docker_agent_args` so a test can render a
/// Windows-shaped path on any platform. `{file}`, never `{file:?}`:
/// Debug doubles the `\` separators of a Windows path.
fn docker_agent_own_agent_file_error(file: &str) -> String {
    format!(
        "llmman launch docker-agent passes its own agent file, so `{file}` would be read as a \
         message rather than an agent.\n\
         To run your own agent against this daemon, point its model at llmman and run \
         docker-agent directly:\n  \
         models:\n    {DOCKER_AGENT_MODEL_NAME}:\n      provider: openai\n      \
         model: <model>\n      base_url: {}/v1\n      \
         token_key: {DOCKER_AGENT_API_KEY_ENV}\n  \
         agents:\n    root:\n      model: {DOCKER_AGENT_MODEL_NAME}",
        daemon::server()
    )
}

/// `~/.config/llmman/launch/docker-agent`, derived from `llmman.conf`'s
/// directory so the two cannot drift (as `dsh::dsh_config_dir` does).
/// docker-agent reads nothing here on its own; `docker_agent_args`
/// passes it the path.
pub(super) fn docker_agent_config_dir() -> anyhow::Result<PathBuf> {
    let conf = crate::config::user_path().context("no home directory")?;
    let dir = conf.parent().context("llmman.conf has no directory")?;
    Ok(dir.join("launch").join("docker-agent"))
}

/// The agent file a launch of `model` writes. One name for every model
/// would let a concurrent launch overwrite it between this write and
/// docker-agent's read, running that launch's model instead; naming it
/// after the model leaves concurrent launches writing the same document
/// and the directory holding one file per model rather than per run.
fn docker_agent_agent_file(dir: &Path, model: &str) -> PathBuf {
    dir.join(format!("agent-{}.yaml", docker_agent_file_stem(model)))
}

/// How much of the model a file name spells out. The rest of the name
/// is `agent-`, the digest and `.yaml`, so the whole stays well inside
/// the 255 a file name gets and leaves room under Windows' path limit.
const DOCKER_AGENT_NAME_MAX: usize = 80;

/// `model` as one bounded file name: everything a path separator or a
/// Windows file name cannot carry — `/`, `:` — becomes `-`, the result
/// is cut to [`DOCKER_AGENT_NAME_MAX`], and a digest of the whole id
/// follows it.
///
/// The digest is what keeps one file per model. Without it two ids that
/// sanitize alike (`a/b-c:d` and `a/b:c-d`) or that differ past the cut
/// would share a file, and launched at the same moment each agent could
/// read the other's model — the collision this name exists to prevent.
/// A provider id is not a model reference and is never checked for
/// length, so the cut is what keeps a long one from failing the write.
fn docker_agent_file_stem(model: &str) -> String {
    use sha2::Digest as _;
    let safe: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .take(DOCKER_AGENT_NAME_MAX)
        .collect();
    let digest = hex::encode(sha2::Sha256::digest(model.as_bytes()));
    format!("{safe}-{}", &digest[..8])
}

fn write_docker_agent_file(path: &Path, model: &str, base_url: &str) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let contents = docker_agent_document(model, base_url);
    crate::fsutil::write_atomic(path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

/// The agent file docker-agent reads: a `models:` map, and an
/// `agents.root` naming one entry. `provider: openai` picks the API
/// client, not the destination — `base_url` is what aims it here.
///
/// `root` gets the `shell` and `filesystem` toolsets — the two
/// docker-agent's own guidance calls the ones most agents need. It asks
/// before each call unless the caller passes `--yolo`, so granting an
/// agent's writes stays the caller's decision, as it is for goose.
///
/// docker-agent sends each toolset's instructions as its own `system`
/// message, which strict chat templates refuse anywhere but first. The
/// daemon merges them (`consolidate_chat_system_messages`), so the
/// agent that reaches the model carries one leading system message
/// whatever the template accepts.
fn docker_agent_document(model: &str, base_url: &str) -> String {
    let quoted_model = common::yaml_quote(model);
    let quoted_base_url = common::yaml_quote(base_url);
    format!(
        "# Written by `llmman launch docker-agent`; edits are overwritten.\n\
         version: \"2\"\n\
         models:\n  {DOCKER_AGENT_MODEL_NAME}:\n    provider: openai\n    model: {quoted_model}\n    \
         base_url: {quoted_base_url}\n    token_key: {DOCKER_AGENT_API_KEY_ENV}\n\
         agents:\n  root:\n    model: {DOCKER_AGENT_MODEL_NAME}\n    \
         description: The agent `llmman launch docker-agent` runs.\n    \
         instruction: You are a helpful AI assistant with access to the shell and \
         the filesystem.\n    \
         toolsets:\n      - type: shell\n      - type: filesystem\n"
    )
}

/// `run`, then the generated agent file, then the caller's arguments.
/// The file goes first because `run` reads its first positional as the
/// agent, so anything placed ahead of it would be taken for one.
fn docker_agent_args(path: &Path, extra_args: &[String]) -> Vec<String> {
    let mut args = vec!["run".to_string(), path.to_string_lossy().into_owned()];
    args.extend_from_slice(extra_args);
    args
}

/// The caller's own agent file, if the first argument names one.
///
/// Narrow on purpose: refusing a launch someone meant is worse than the
/// confusion this prevents. It must be the first argument, not a flag,
/// and a file that exists — so a message ending in `.yaml`, or a path
/// that is some flag's value, is left alone. A file behind a boolean
/// flag is missed, and reaches docker-agent as a message.
fn docker_agent_config_argument(extra_args: &[String]) -> Option<&String> {
    docker_agent_config_argument_with(extra_args, |path| Path::new(path).is_file())
}

/// Split from [`docker_agent_config_argument`] so which arguments count
/// can be asserted without creating files.
fn docker_agent_config_argument_with(
    extra_args: &[String],
    exists: impl Fn(&str) -> bool,
) -> Option<&String> {
    const EXTENSIONS: &[&str] = &[".yaml", ".yml", ".hcl"];
    let first = extra_args.first()?;
    if first.starts_with('-') {
        return None;
    }
    let lowered = first.to_lowercase();
    let named_like_one = EXTENSIONS.iter().any(|ext| lowered.ends_with(ext));
    (named_like_one && exists(first)).then_some(first)
}

/// `PATH`, then `~/.docker/cli-plugins` — where Docker Desktop and
/// `brew install docker-agent` put it. That is not a `PATH` entry, so
/// without the fallback a working install reports itself missing.
pub(super) fn find_docker_agent() -> Option<PathBuf> {
    find_on_path("docker-agent").or_else(|| docker_agent_fallback(&dirs::home_dir()?))
}

/// Split so the lookup can be asserted against a synthetic home.
fn docker_agent_fallback(home: &Path) -> Option<PathBuf> {
    let binary = if cfg!(windows) {
        "docker-agent.exe"
    } else {
        "docker-agent"
    };
    let candidate = home.join(".docker").join("cli-plugins").join(binary);
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llmman's file must lead, being `run`'s first positional —
    /// `run "say pong"` is parsed as an OCI reference and fails. The
    /// caller's arguments follow unchanged.
    #[test]
    fn docker_agent_args_lead_with_run_and_the_generated_agent_file() {
        let path = Path::new("/tmp/llmman/agent.yaml");
        let extra = ["--exec".to_string(), "say pong".to_string()];
        assert_eq!(
            docker_agent_args(path, &extra),
            vec![
                "run".to_string(),
                "/tmp/llmman/agent.yaml".to_string(),
                "--exec".to_string(),
                "say pong".to_string(),
            ]
        );
        assert_eq!(
            docker_agent_args(path, &[]),
            vec!["run".to_string(), "/tmp/llmman/agent.yaml".to_string()]
        );
    }

    /// `base_url` is what makes an `openai`-provider model reach this
    /// daemon instead of api.openai.com.
    #[test]
    fn docker_agent_document_routes_an_openai_provider_model_at_the_daemon() {
        let document = docker_agent_document("m", "http://127.0.0.1:17434/v1");
        assert!(document.contains("provider: openai"), "{document}");
        assert!(document.contains("model: \"m\""), "{document}");
        assert!(
            document.contains("base_url: \"http://127.0.0.1:17434/v1\""),
            "{document}"
        );
        // The agent selects the entry by the name the model map gives it.
        assert!(
            document.contains(&format!("  {DOCKER_AGENT_MODEL_NAME}:")),
            "{document}"
        );
        assert!(
            document.contains(&format!("model: {DOCKER_AGENT_MODEL_NAME}\n")),
            "{document}"
        );
    }

    /// The key is named, never written, so a `--provider` launch does
    /// not persist a real credential — as `qwen::write_qwen_settings_at` and
    /// `dsh::write_dsh_settings` also promise.
    #[test]
    fn docker_agent_document_names_the_key_variable_rather_than_a_key() {
        let document = docker_agent_document("m", "http://127.0.0.1:17434/v1");
        assert!(
            document.contains(&format!("token_key: {DOCKER_AGENT_API_KEY_ENV}")),
            "{document}"
        );
        assert!(!document.contains("api_key"), "{document}");
        assert!(!document.contains("sk-"), "{document}");
    }

    /// A launched agent that can only chat is not much of an agent. The
    /// extra `system` message each toolset adds is handled by the daemon
    /// (`consolidate_chat_system_messages`), not by leaving them out.
    #[test]
    fn docker_agent_document_gives_the_agent_shell_and_filesystem() {
        let document = docker_agent_document("m", "http://127.0.0.1:17434/v1");
        assert!(document.contains("instruction:"), "{document}");
        assert!(document.contains("toolsets:"), "{document}");
        assert!(document.contains("- type: shell"), "{document}");
        assert!(document.contains("- type: filesystem"), "{document}");
    }

    /// Unquoted, `docker.io/ai/qwen3.5:0.8b` parses as a mapping at the
    /// colon rather than as the model's name.
    #[test]
    fn docker_agent_document_quotes_the_values_it_interpolates() {
        let document = docker_agent_document("docker.io/ai/qwen3.5:0.8b", "http://h:1/v1");
        assert!(
            document.contains("model: \"docker.io/ai/qwen3.5:0.8b\""),
            "{document}"
        );
        assert!(
            document.contains("base_url: \"http://h:1/v1\""),
            "{document}"
        );
    }

    /// llmman supplies the agent file, so a second one would be read as a
    /// *message* — silently, with the caller's agent never running.
    #[test]
    fn docker_agent_spots_a_caller_supplied_agent_file() {
        for file in ["team.yaml", "./agent.yml", "/tmp/AGENT.YAML", "infra.hcl"] {
            let extra = [file.to_string(), "do the thing".to_string()];
            assert_eq!(
                docker_agent_config_argument_with(&extra, |_| true).map(String::as_str),
                Some(file),
                "{file} was not recognized as an agent file"
            );
        }
    }

    /// Arguments that must not be taken for an agent file, because
    /// refusing any of them would block a launch the caller meant. The
    /// first implementation scanned every argument for the extension and
    /// so rejected `--exec "summarize the config.yaml"`, a message.
    #[test]
    fn docker_agent_does_not_mistake_a_message_or_a_flag_value_for_an_agent_file() {
        let cases: &[(&str, &[&str])] = &[
            (
                "a message that ends in .yaml",
                &["--exec", "summarize the config.yaml"],
            ),
            ("a flag's value", &["--prompt-file", "notes.yaml"]),
            ("a leading flag", &["--exec", "--yolo", "say pong"]),
            ("a question about a file", &["what does agent.yaml do?"]),
        ];
        for (why, args) in cases {
            let extra: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            // `|_| true`: even if every path existed, none of these is the
            // first positional, so none may be refused.
            assert_eq!(
                docker_agent_config_argument_with(&extra, |_| true),
                None,
                "refused {why}: {args:?}"
            );
        }
        // The remaining guard: a first argument that names a config file
        // but does not exist is prose too.
        let absent = ["explain agent.yaml".to_string()];
        assert_eq!(docker_agent_config_argument_with(&absent, |_| false), None);
    }

    /// One name for every model would let a concurrent launch overwrite
    /// the file before docker-agent reads it, so models that differ get
    /// files that differ — including ones that sanitize or cut alike.
    /// Whatever the id carries, the name stays one component of `dir`:
    /// a model is not a path.
    #[test]
    fn docker_agent_names_its_agent_file_after_the_model() {
        let dir = Path::new("/tmp/llmman/launch/docker-agent");
        let file = |model: &str| docker_agent_agent_file(dir, model);
        let name = |model: &str| {
            file(model)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string()
        };
        assert!(
            name("docker.io/ai/qwen3.5:0.8b").starts_with("agent-docker.io-ai-qwen3.5-0.8b-"),
            "{}",
            name("docker.io/ai/qwen3.5:0.8b")
        );
        for (one, other) in [
            ("docker.io/ai/qwen3.5:0.8b", "docker.io/ai/qwen3.5:9b"),
            ("docker.io/ai/qwen3.5:0.8b", "docker.io/ai/gemma4:12b"),
            ("qwen/qwen3-coder", "qwen/qwen3-max"),
            // Sanitize alike: `/` and `:` both become `-`.
            ("a/b-c:d", "a/b:c-d"),
            // Differ only past the cut.
            (
                &format!("{}one", "m".repeat(DOCKER_AGENT_NAME_MAX)),
                &format!("{}two", "m".repeat(DOCKER_AGENT_NAME_MAX)),
            ),
        ] {
            assert_ne!(file(one), file(other), "{one} and {other} share a file");
        }
        // A provider id is never length-checked, so the name is bounded
        // here or the write fails.
        assert!(name(&"x".repeat(4096)).len() <= 255, "name is unbounded");
        for model in ["a/b", "a:b", "a\\b", "a b", "a*b", "a?b", "a\"b", "..", "."] {
            let file = docker_agent_agent_file(dir, model);
            assert_eq!(file.parent(), Some(dir), "{model} escaped its directory");
            let name = file.file_name().and_then(|n| n.to_str()).unwrap();
            assert!(
                name.starts_with("agent-") && name.ends_with(".yaml"),
                "{name}"
            );
            assert!(
                !name.contains(std::path::MAIN_SEPARATOR) && name != ".." && name != ".",
                "{model} produced {name}"
            );
        }
    }

    /// The plugin directory is not on `PATH`, so without the fallback a
    /// working install reports itself missing.
    #[test]
    fn docker_agent_fallback_finds_the_cli_plugin_directory() {
        let home = std::env::temp_dir().join(format!(
            "llmman-docker-agent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let plugins = home.join(".docker").join("cli-plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        assert_eq!(docker_agent_fallback(&home), None);
        let binary = plugins.join(if cfg!(windows) {
            "docker-agent.exe"
        } else {
            "docker-agent"
        });
        std::fs::create_dir(&binary).unwrap();
        assert_eq!(
            docker_agent_fallback(&home),
            None,
            "a directory is not a binary"
        );
        std::fs::remove_dir(&binary).unwrap();
        std::fs::write(&binary, "").unwrap();
        assert_eq!(docker_agent_fallback(&home), Some(binary));
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Cobra stops parsing flags at a `--`, so a prompt after one that
    /// reads like a flag is a message — and refusing it would refuse a
    /// launch that works.
    #[test]
    fn docker_agent_reads_a_model_flag_after_the_terminator_as_a_message() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(check_docker_agent_args(&args(&["--exec", "--", "--model=foo"])).is_ok());
        assert!(check_docker_agent_args(&args(&["--exec", "--", "--model", "foo"])).is_ok());
        // The flag itself is still refused where docker-agent parses it.
        assert!(check_docker_agent_args(&args(&["--exec", "--model=foo"])).is_err());
        assert!(check_docker_agent_args(&args(&["--model", "foo", "--", "hi"])).is_err());
    }

    /// A forwarded `--model` replaces the generated entry's `base_url`
    /// too, sending the request — and, under `--provider`, a real key —
    /// to api.openai.com.
    #[test]
    fn docker_agent_refuses_a_forwarded_model_flag() {
        for spelling in ["--model", "--model=openai/gpt-5"] {
            let extra = [spelling.to_string(), "openai/gpt-5".to_string()];
            let error = launch_docker_agent("m", "k", &extra)
                .unwrap_err()
                .to_string();
            assert!(error.contains("api.openai.com"), "{spelling}: {error}");
        }
    }

    /// `run`'s single-letter flags are `-a`, `-s`, `-w`, `-d`, `-o`,
    /// `-h` — no `-m`. Guards the `Some("-m")` this once passed to
    /// `has_flag`, which refused unrelated arguments.
    ///
    /// Asserted through `has_flag`, not `launch_docker_agent`: with
    /// nothing to refuse that call reaches `exec_with_env`, whose
    /// `process::exit` took the test harness down once already.
    #[test]
    fn docker_agent_leaves_an_unrelated_short_flag_to_docker_agent() {
        for spelling in ["-m", "-m=openai/gpt-5"] {
            let extra = [spelling.to_string(), "openai/gpt-5".to_string()];
            assert!(
                !has_flag(&extra, "--model", None),
                "{spelling} was read as --model"
            );
        }
        // The flag it really does have, still caught.
        assert!(has_flag(&["--model".to_string()], "--model", None));
    }

    /// The refusal prints a working model entry, so a caller with their
    /// own agent file is told how to point it here.
    #[test]
    fn docker_agent_refuses_a_caller_supplied_agent_file_with_the_way_forward() {
        // A real file, because the guard requires one — and because
        // without the refusal this call would reach `exec_with_env`,
        // which never returns.
        let path = std::env::temp_dir().join(format!(
            "llmman-docker-agent-{}-{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, "agents: {}\n").unwrap();
        let named = path.to_string_lossy().into_owned();
        let error = launch_docker_agent("m", "k", std::slice::from_ref(&named))
            .unwrap_err()
            .to_string();
        let _ = std::fs::remove_file(&path);

        assert!(error.contains(&named), "{error}");
        assert!(error.contains("provider: openai"), "{error}");
        assert!(error.contains(DOCKER_AGENT_API_KEY_ENV), "{error}");
    }

    /// `{file:?}` echoed `C:\Users\...` back as `C:\\Users\\...`, twice
    /// the backslashes the caller typed. Only the Windows CI legs had
    /// separators to double, so only they caught it.
    #[test]
    fn docker_agent_agent_file_error_does_not_escape_a_windows_path() {
        let windows = r"C:\Users\me\AppData\Local\Temp\team.yaml";
        let error = docker_agent_own_agent_file_error(windows);
        assert!(error.contains(windows), "{error}");
        assert!(!error.contains(r"\\"), "separators were doubled: {error}");
    }
}
