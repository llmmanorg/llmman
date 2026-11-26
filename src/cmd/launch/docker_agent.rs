//! `llmman launch docker-agent` (Docker Agent).

use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::daemon;

use super::common;
use super::models::{self, ModelEntry, ModelList};
use super::{before_terminator, exec_with_env, find_on_path, has_flag, server};

/// The env var the generated `token_key` names, so no key reaches disk
/// (same role as `qwen::QWEN_ENV_KEY` and `dsh::DSH_API_KEY_ENV`).
const DOCKER_AGENT_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// The model name the refusal's example agent file gives its entry.
const DOCKER_AGENT_MODEL_NAME: &str = "llmman";

/// docker-agent: write an agent file whose `models:` holds every model of
/// the list, each an `openai`-provider model pointing at this daemon, then
/// hand that file to `docker-agent run`. `/model` inside docker-agent picks
/// among them, and sends the entry's `model`, the reference, as written.
///
/// An agent file, not docker-agent's own `~/.config/cagent/config.yaml`:
/// a `models:` map there is ignored, and the entry cannot then be
/// selected. Writing one llmman owns also leaves `~/.config/cagent`
/// alone.
///
/// The file is thousands of entries and belongs to this launch alone, so it
/// is one [`models::write_transient`] per process, removed when docker-agent
/// exits: concurrent launches never read each other's.
pub(super) fn launch_docker_agent(
    _model: &str,
    api_key: &str,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    // `run` has already rejected these arguments before starting the
    // daemon; repeated so a direct call to `launch` rejects them too.
    check_docker_agent_args(extra_args)?;
    let bin = find_docker_agent().ok_or_else(docker_agent_missing)?;

    let document = docker_agent_document(models, &format!("{}/v1", server()))?;
    let path = models::write_transient(&docker_agent_config_dir()?, "agent", "yaml", &document)?;

    let args = docker_agent_args(&path, extra_args);
    exec_with_env(&bin, &args, &[(DOCKER_AGENT_API_KEY_ENV, api_key)])
}

pub(super) fn docker_agent_missing() -> anyhow::Error {
    anyhow::anyhow!(
        "docker-agent is not installed\n\n\
         llmman looks for it on PATH and in ~/.docker/cli-plugins, where Docker Desktop \
         and `brew install docker-agent` put it.\n\
         Releases: https://github.com/docker/docker-agent/releases"
    )
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
    if has_flag(before_terminator(extra_args), "--model", None) {
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

/// The agent file docker-agent reads: a `models:` map with a key per row,
/// and an `agents.root` naming the first row's key, the one the launch starts
/// on. `provider: openai` picks the API client, not the destination —
/// `base_url` is what aims it here.
///
/// A key is the row's plain id, told apart where two rows share one, and
/// what `/model` shows. YAML would keep only the last of two equal keys, so
/// they are made unique first.
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
fn docker_agent_document(models: &ModelList, base_url: &str) -> anyhow::Result<String> {
    let name = models.display_namer();
    let keys = models.unique_keys(&name);
    let start = keys
        .first()
        .context("docker-agent needs a model to start on")?;
    let declares_images = models.entries().iter().any(|entry| entry.vision);
    let version = if declares_images {
        DOCKER_AGENT_CAPABILITIES_VERSION
    } else {
        DOCKER_AGENT_BASE_VERSION
    };
    let entries: String = models
        .entries()
        .iter()
        .zip(&keys)
        .map(|(entry, key)| docker_agent_model(key, entry, base_url, declares_images))
        .collect();
    let quoted_start = common::yaml_quote(start);
    Ok(format!(
        "# Written by `llmman launch docker-agent`; edits are overwritten.\n\
         version: \"{version}\"\n\
         models:\n{entries}\
         agents:\n  root:\n    model: {quoted_start}\n    \
         description: The agent `llmman launch docker-agent` runs.\n    \
         instruction: You are a helpful AI assistant with access to the shell and \
         the filesystem.\n    \
         toolsets:\n      - type: shell\n      - type: filesystem\n"
    ))
}

/// The config version for a list where no model takes images. Docker
/// Desktop's docker-agent 1.50.0 stops at 8, so this is the one it reads.
const DOCKER_AGENT_BASE_VERSION: &str = "2";

/// The config version that takes `capabilities`, which lets an `openai` model with a `base_url`
/// accept images. Used only when a model takes images: docker-agent 1.142.0 reads it, but 1.50.0
/// (Docker Desktop's build) stops at schema 8 and refuses the file. `context_window` needs 16,
/// which 1.142.0 rejects, so `max_tokens` is left out: without a window it cannot be kept clear of
/// the prompt.
const DOCKER_AGENT_CAPABILITIES_VERSION: &str = "11";

/// One `models:` entry, on one line. `capabilities` is written only when the
/// document is the version that takes it.
fn docker_agent_model(
    key: &str,
    entry: &ModelEntry,
    base_url: &str,
    declares_images: bool,
) -> String {
    let capabilities = if declares_images {
        format!(", capabilities: {{ image: {} }}", entry.vision)
    } else {
        String::new()
    };
    format!(
        "  {}: {{ provider: openai, model: {}, base_url: {}, token_key: {DOCKER_AGENT_API_KEY_ENV}\
         {capabilities} }}\n",
        common::yaml_quote(key),
        common::yaml_quote(&entry.reference),
        common::yaml_quote(base_url),
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

    fn hosted(provider: &str, id: &str) -> ModelEntry {
        ModelEntry::unlisted(&crate::providers::format_remote_ref(provider, id))
    }

    fn document_of(entries: Vec<ModelEntry>) -> (String, serde_json::Value) {
        let document =
            docker_agent_document(&ModelList::of(entries), "http://127.0.0.1:17434/v1").unwrap();
        let parsed = yaml_serde::from_str(&document).expect("valid YAML");
        (document, parsed)
    }

    /// `base_url` is what makes an `openai`-provider model reach this
    /// daemon instead of api.openai.com.
    #[test]
    fn docker_agent_document_routes_an_openai_provider_model_at_the_daemon() {
        let (document, doc) = document_of(vec![hosted("alpha", "m")]);
        let entry = &doc["models"]["m"];
        assert_eq!(entry["provider"], "openai", "{document}");
        assert_eq!(entry["model"], "llmman.provider/alpha/m", "{document}");
        assert_eq!(entry["base_url"], "http://127.0.0.1:17434/v1", "{document}");
        // The agent selects the entry by the key the model map gives it.
        assert_eq!(doc["agents"]["root"]["model"], "m", "{document}");
    }

    /// Every row is a model docker-agent can switch to, the starting one
    /// first and the one `root` runs; `/model` sends each one's reference.
    #[test]
    fn docker_agent_document_holds_every_model_with_the_start_selected() {
        let entries = vec![
            ModelEntry::unlisted("docker.io/ai/qwen3.5:0.8b"),
            hosted("alpha", "a"),
            hosted("beta", "b"),
        ];
        let (document, doc) = document_of(entries.clone());
        let models = doc["models"].as_object().unwrap();
        assert_eq!(models.len(), 3);
        for (entry, key) in entries.iter().zip(["qwen3.5:0.8b", "a", "b"]) {
            assert_eq!(models[key]["model"], entry.reference.as_str(), "{key}");
        }
        // Written in list order, whatever order the parser keeps them in.
        let at = |key: &str| document.find(&format!("\"{key}\":")).unwrap();
        assert!(
            at("qwen3.5:0.8b") < at("a") && at("a") < at("b"),
            "{document}"
        );
        assert_eq!(doc["agents"]["root"]["model"], "qwen3.5:0.8b");
    }

    /// Two providers can offer one id; YAML keeps the last of two equal
    /// keys, so each row must still be there, and tell its provider.
    #[test]
    fn docker_agent_document_keeps_models_of_the_same_id_from_different_providers() {
        let entries = vec![
            hosted("alpha", "gpt"),
            hosted("beta", "gpt"),
            hosted("gamma", "gpt"),
            ModelEntry::unlisted("gpt"),
        ];
        let (document, doc) = document_of(entries.clone());
        let models = doc["models"].as_object().unwrap();
        assert_eq!(models.len(), entries.len(), "{document}");
        for entry in &entries {
            let found = models
                .values()
                .filter(|m| m["model"] == entry.reference.as_str())
                .count();
            assert_eq!(found, 1, "{}: {document}", entry.reference);
        }
        // The same id of a provider and of a stored model: told apart too.
        assert!(models.contains_key("alpha/gpt"), "{document}");
        assert!(models.contains_key("beta/gpt"), "{document}");
        assert!(models.contains_key("gpt"), "{document}");
        assert_eq!(doc["agents"]["root"]["model"], "alpha/gpt");
    }

    /// An id that two rows of one provider share (a list can repeat) is
    /// still one key per row.
    #[test]
    fn docker_agent_document_numbers_equal_keys() {
        let entries = vec![hosted("alpha", "m"), hosted("alpha", "m")];
        let (_, doc) = document_of(entries);
        let models = doc["models"].as_object().unwrap();
        assert_eq!(models.len(), 2);
    }

    /// What a row says of images reaches docker-agent, which drops the
    /// attachments of a model it is not told takes them.
    #[test]
    fn docker_agent_document_declares_images_for_a_vision_model_only() {
        let seeing = ModelEntry {
            vision: true,
            ..hosted("alpha", "eyes")
        };
        let (document, doc) = document_of(vec![seeing, hosted("alpha", "blind")]);
        assert_eq!(
            doc["models"]["eyes"]["capabilities"]["image"], true,
            "{document}"
        );
        assert_eq!(
            doc["models"]["blind"]["capabilities"]["image"], false,
            "{document}"
        );
        assert_eq!(doc["version"], DOCKER_AGENT_CAPABILITIES_VERSION);
    }

    /// A list that takes no images keeps the older document, which Docker
    /// Desktop's docker-agent reads.
    #[test]
    fn docker_agent_document_of_text_only_models_is_version_2_without_capabilities() {
        let (document, doc) = document_of(vec![hosted("alpha", "a"), hosted("beta", "b")]);
        assert_eq!(doc["version"], DOCKER_AGENT_BASE_VERSION, "{document}");
        assert_eq!(doc["version"], "2", "{document}");
        assert!(!document.contains("capabilities"), "{document}");
        for (key, entry) in doc["models"].as_object().unwrap() {
            assert!(entry.get("capabilities").is_none(), "{key}: {document}");
        }
    }

    #[test]
    fn docker_agent_document_with_an_image_model_is_version_11_with_capabilities() {
        let seeing = ModelEntry {
            vision: true,
            ..hosted("alpha", "eyes")
        };
        let (document, doc) = document_of(vec![hosted("alpha", "blind"), seeing]);
        assert_eq!(doc["version"], "11", "{document}");
        assert_eq!(
            doc["models"]["eyes"]["capabilities"]["image"], true,
            "{document}"
        );
    }

    /// The key is named, never written, so a `--provider` launch does
    /// not persist a real credential — as `qwen::write_qwen_settings_at` and
    /// `dsh::write_dsh_settings` also promise.
    #[test]
    fn docker_agent_document_names_the_key_variable_rather_than_a_key() {
        let (document, doc) = document_of(vec![hosted("alpha", "m"), hosted("beta", "m")]);
        for (key, entry) in doc["models"].as_object().unwrap() {
            assert_eq!(entry["token_key"], DOCKER_AGENT_API_KEY_ENV, "{key}");
        }
        assert!(!document.contains("api_key"), "{document}");
        assert!(!document.contains("sk-"), "{document}");
    }

    /// A launched agent that can only chat is not much of an agent. The
    /// extra `system` message each toolset adds is handled by the daemon
    /// (`consolidate_chat_system_messages`), not by leaving them out.
    #[test]
    fn docker_agent_document_gives_the_agent_shell_and_filesystem() {
        let (document, _) = document_of(vec![hosted("alpha", "m")]);
        assert!(document.contains("instruction:"), "{document}");
        assert!(document.contains("toolsets:"), "{document}");
        assert!(document.contains("- type: shell"), "{document}");
        assert!(document.contains("- type: filesystem"), "{document}");
    }

    /// Unquoted, `docker.io/ai/qwen3.5:0.8b` parses as a mapping at the
    /// colon rather than as the model's name.
    #[test]
    fn docker_agent_document_quotes_the_values_it_interpolates() {
        let (document, doc) = document_of(vec![ModelEntry::unlisted("hf.co/o/qwen3.5:0.8b")]);
        assert_eq!(
            doc["models"]["hf.co/o/qwen3.5:0.8b"]["model"],
            "hf.co/o/qwen3.5:0.8b"
        );
        assert!(
            document.contains("model: \"hf.co/o/qwen3.5:0.8b\""),
            "{document}"
        );
        assert!(
            document.contains("base_url: \"http://127.0.0.1:17434/v1\""),
            "{document}"
        );
    }

    /// Nothing to start on is refused here, not left to docker-agent to
    /// report of an agent with no model.
    #[test]
    fn docker_agent_document_needs_a_model() {
        let error = docker_agent_document(&ModelList::default(), "http://h/v1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("model to start on"), "{error}");
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
            let error = launch_docker_agent("m", "k", &ModelList::default(), &extra)
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
        let error = launch_docker_agent(
            "m",
            "k",
            &ModelList::default(),
            std::slice::from_ref(&named),
        )
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
