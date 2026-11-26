//! `llmman launch claude`.

use super::common;
use super::models::{write_transient, ModelEntry, ModelList};
use super::optimizer::{Addition, Optimizer};
use super::{
    before_terminator, exec_with_env_removing, find_on_path, has_flag, has_forwarded_model_flag,
    server,
};

/// The model claude last ran, which `/model` saved to its user settings.
pub(super) fn remembered_model() -> Option<String> {
    let dir = super::env_dir("CLAUDE_CONFIG_DIR")
        .or_else(|| dirs::home_dir().map(|home| home.join(".claude")))?;
    let text = std::fs::read_to_string(dir.join("settings.json")).ok()?;
    let settings: serde_json::Value = serde_json::from_str(&text).ok()?;
    // The picker saves a row that `picker_model` suffixed.
    let model = settings["model"].as_str()?;
    Some(model.strip_suffix("[1m]").unwrap_or(model).to_string())
}

/// Inherited variables that point claude at another provider or credential.
const CLAUDE_ENV_TO_CLEAR: &[&str] = &[
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
];

/// claude's alias, background and subagent models. ANTHROPIC_SMALL_FAST_MODEL
/// is read before ANTHROPIC_DEFAULT_HAIKU_MODEL.
const CLAUDE_MODEL_ENV: &[&str] = &[
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
];

/// The model's context window, in tokens. Without it claude warns "Model ...
/// isn't recognized · context assumed to be 200k tokens" for a local model
/// (an ID not starting with `claude-`) and compacts at 200k.
const CLAUDE_MAX_CONTEXT_ENV: &str = "CLAUDE_CODE_MAX_CONTEXT_TOKENS";

/// claude's model aliases, each with an optional `[1m]` suffix, and
/// `default`. They resolve through [`CLAUDE_MODEL_ENV`].
const CLAUDE_MODEL_ALIASES: &[&str] = &[
    "default", "best", "fable", "opus", "sonnet", "haiku", "opusplan",
];

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
pub(super) fn launch_claude(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    models: &ModelList,
    extra_args: &[String],
    optimizer: Optimizer,
) -> anyhow::Result<()> {
    let bin = find_on_path("claude").ok_or_else(|| anyhow::anyhow!("claude is not installed"))?;
    let added = optimizer.addition("claude")?;
    let server = added.server.clone().unwrap_or_else(server);
    let session_model = session_model(model, extra_args);
    let window = claude_context_window(
        session_window(session_model, context_window, models),
        std::env::var_os(CLAUDE_MAX_CONTEXT_ENV).is_some(),
    )
    // an image sandbox does not inherit the caller's
    .or_else(|| std::env::var(CLAUDE_MAX_CONTEXT_ENV).ok());
    let picker = models.entries().len() > 1;
    let (args, mut env) = with_addition(
        claude_args(model, extra_args),
        claude_env(session_model, api_key, &server, window.as_deref()),
        &added,
    );
    let args = if picker {
        env.push((CLAUDE_ALWAYS_EFFORT_ENV, "1"));
        merge_settings(args, picker_settings(models))?
    } else {
        args
    };
    if let Some(notice) = optimizer.notice("claude") {
        eprintln!("{notice}");
    }
    exec_with_env_removing(&bin, &args, &env, CLAUDE_ENV_TO_CLEAR)
}

/// Sends claude's `effort` parameter for ids it does not know: without it
/// `/effort` offers a model of llmman's nothing to choose.
const CLAUDE_ALWAYS_EFFORT_ENV: &str = "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT";

/// The window at which claude takes an id it does not know to hold 1M
/// tokens, with `[1m]`, whatever [`CLAUDE_MAX_CONTEXT_ENV`] says.
const CLAUDE_1M_WINDOW: u64 = 1_000_000;

/// The model a picker row selects. claude strips `[1m]` before sending the id and sizes such a row
/// at 1M whatever [`CLAUDE_MAX_CONTEXT_ENV`] holds (one value per session), so a row of 1M or more
/// keeps its window beside a smaller start.
fn picker_model(entry: &ModelEntry) -> String {
    if entry.context_window.is_some_and(|w| w >= CLAUDE_1M_WINDOW) {
        format!("{}[1m]", entry.reference)
    } else {
        entry.reference.clone()
    }
}

/// The `/model` picker for the list: one row per model, titled with its
/// plain id and provider and described by what it takes, in the list's order
/// and instead of claude's own lineup (a local model is none of its).
fn picker_settings(models: &ModelList) -> serde_json::Value {
    let rows: Vec<serde_json::Value> = models
        .entries()
        .iter()
        .map(|entry| {
            let mut notes = Vec::new();
            if let Some(window) = entry.context_window {
                notes.push(format!("{}k context", window / 1000));
            }
            if entry.vision {
                notes.push("images".to_string());
            }
            if entry.tools == Some(true) {
                notes.push("tools".to_string());
            }
            serde_json::json!({
                "model": picker_model(entry),
                "label": entry.label(),
                "description": notes.join(", "),
            })
        })
        .collect();
    serde_json::json!({ "modelPicker": { "options": rows, "replaceBuiltInOptions": true } })
}

/// `args` with the `--settings` before a `--` folded with `ours` into one: claude keeps only the
/// last `--settings`, so a forwarded one (or an optimizer's) would replace the picker. The last one
/// stays as written; the picker replaces any it has.
fn merge_settings(args: Vec<String>, ours: serde_json::Value) -> anyhow::Result<Vec<String>> {
    merge_settings_in(args, ours, &super::models::state_dir("claude")?)
}

/// [`merge_settings`], written to `dir`.
fn merge_settings_in(
    args: Vec<String>,
    ours: serde_json::Value,
    dir: &std::path::Path,
) -> anyhow::Result<Vec<String>> {
    let end = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let mut rest = Vec::new();
    let mut last = None;
    let mut i = 0;
    while i < args.len() {
        let value = if i >= end {
            None
        } else if args[i] == "--settings" {
            i += 1;
            Some(args.get(i).cloned().unwrap_or_default())
        } else {
            args[i].strip_prefix("--settings=").map(str::to_string)
        };
        match value {
            Some(value) => last = Some(value),
            None => rest.push(args[i].clone()),
        }
        i += 1;
    }
    let mut merged = match last {
        Some(value) => read_settings(&value)?,
        None => serde_json::json!({}),
    };
    if let (Some(merged), serde_json::Value::Object(ours)) = (merged.as_object_mut(), ours) {
        merged.extend(ours);
    }
    let text = serde_json::to_string(&merged).expect("settings serialize");
    let path = write_transient(dir, "settings", "json", &text)?;
    let mut out = vec![
        "--settings".to_string(),
        path.to_string_lossy().into_owned(),
    ];
    out.extend(rest);
    Ok(out)
}

/// A `--settings` value: a JSON object, or the file that holds one.
fn read_settings(value: &str) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context;
    let text = if value.trim_start().starts_with('{') {
        value.to_string()
    } else {
        std::fs::read_to_string(value).with_context(|| format!("read --settings {value}"))?
    };
    let settings: serde_json::Value = serde_json::from_str(&common::strip_json_comments(
        text.trim_start_matches('\u{feff}'),
    ))
    .with_context(|| format!("--settings {value} is not JSON"))?;
    anyhow::ensure!(
        settings.is_object(),
        "--settings {value} is not a JSON object"
    );
    Ok(settings)
}

/// The window of the model the session runs: llmman's start, or a forwarded `--model` the list
/// holds. Not the list's smallest, which an embedding or tiny model would drag down.
fn session_window(session_model: &str, window: Option<u64>, models: &ModelList) -> Option<u64> {
    match models.start() {
        Some(start) if start.reference == session_model => window,
        _ => models
            .entries()
            .iter()
            .find(|e| e.reference == session_model || e.id == session_model)
            .and_then(|e| e.context_window),
    }
}

/// `args` and `env` with what an optimizer adds: its arguments first, its
/// variables last. An empty [`Addition`] returns both as they were.
fn with_addition<'a>(
    mut args: Vec<String>,
    mut env: Vec<(&'a str, &'a str)>,
    added: &'a Addition,
) -> (Vec<String>, Vec<(&'a str, &'a str)>) {
    args.splice(0..0, added.args.iter().cloned());
    env.extend(added.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    (args, env)
}

/// What in `extra_args` keeps `--optimizer tk` from working with claude:
/// claude keeps only the last `--settings`; `--bare` skips hooks; the hook
/// cannot see `--disallowedTools` rules or the project settings that
/// `--setting-sources` can exclude.
pub(super) fn optimizer_conflict(extra_args: &[String]) -> Option<&'static str> {
    let args = before_terminator(extra_args);
    if has_flag(args, "--settings", None) {
        Some("--settings after --: claude keeps only the last one, so llmman's hook and yours could not both apply.\nPut your settings in a settings file, or launch without --optimizer.")
    } else if has_flag(args, "--bare", None) {
        Some("--bare after --: claude skips hooks then, so nothing would be compacted.")
    } else if has_flag(args, "--disallowedTools", None)
        || has_flag(args, "--disallowed-tools", None)
    {
        Some("--disallowedTools after --: claude checks those rules against the command as rewritten, which a rewrite could slip past.\nPut the rules in a settings file, or launch without --optimizer.")
    } else if has_flag(args, "--setting-sources", None) {
        Some("--setting-sources after --: the hook reads project settings files that claude would then skip, so it could approve a command claude would ask about.\nLaunch without --setting-sources, or without --optimizer.")
    } else {
        None
    }
}

/// What in `extra_args` keeps `--optimizer cvmn` from working with claude.
pub(super) fn cvmn_conflict(extra_args: &[String]) -> Option<&'static str> {
    let args = before_terminator(extra_args);
    if has_flag(args, "--append-system-prompt", None)
        || has_flag(args, "--append-system-prompt-file", None)
    {
        Some("--append-system-prompt after --: llmman's terse-reply prompt is appended the same way.\nPut your instructions in CLAUDE.md, or launch without --optimizer.")
    } else {
        None
    }
}

/// The value for [`CLAUDE_MAX_CONTEXT_ENV`]: the session model's window,
/// unless the caller set the variable.
fn claude_context_window(window: Option<u64>, already_set: bool) -> Option<String> {
    window
        .filter(|&w| w > 0 && !already_set)
        .map(|w| w.to_string())
}

/// Split out so a test can assert what claude is handed:
/// [`super::exec_with_env_removing`] never returns, so calling
/// [`launch_claude`] would take the test runner with it.
fn claude_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    // Absent, not empty: claude reads "" as a model actually named "".
    // A forwarded `--model` replaces llmman's.
    if !model.is_empty() && !has_forwarded_model_flag("claude", extra_args) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// The model to set [`CLAUDE_MODEL_ENV`] to: a forwarded `--model`, unless
/// it is an alias, else `model`. Empty when a forwarded `--model` has no value.
fn session_model<'a>(model: &'a str, extra_args: &'a [String]) -> &'a str {
    if !has_forwarded_model_flag("claude", extra_args) {
        return model;
    }
    match common::forwarded_model(extra_args) {
        Some(forwarded) if !is_claude_alias(forwarded) => forwarded,
        Some(_) => model,
        None => "",
    }
}

/// Whether the session runs `model`, not another forwarded in its place (an
/// alias is not one).
pub(super) fn runs_model(model: &str, extra_args: &[String]) -> bool {
    session_model(model, extra_args) == model
}

fn is_claude_alias(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let name = name.strip_suffix("[1m]").unwrap_or(&name);
    CLAUDE_MODEL_ALIASES.contains(&name)
}

/// What llmman sets for claude. `base_url` is the bare origin: claude
/// appends `/v1/messages` itself. Every [`CLAUDE_MODEL_ENV`] is set to
/// `model`, so background requests and subagents use it too, and
/// `window`, when there is one, is the model's context window.
fn claude_env<'a>(
    model: &'a str,
    api_key: &'a str,
    base_url: &'a str,
    window: Option<&'a str>,
) -> Vec<(&'a str, &'a str)> {
    let mut env = vec![
        ("ANTHROPIC_BASE_URL", base_url),
        ("ANTHROPIC_API_KEY", api_key),
    ];
    if !model.is_empty() {
        env.extend(CLAUDE_MODEL_ENV.iter().map(|&name| (name, model)));
        if let Some(window) = window {
            env.push((CLAUDE_MAX_CONTEXT_ENV, window));
        }
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_select_the_resolved_model_and_pass_the_rest_through() {
        assert_eq!(
            claude_args("local/model", &["-p".into(), "hello".into()]),
            ["--model", "local/model", "-p", "hello"]
        );
        assert_eq!(
            claude_args("", &["-p".into(), "hello".into()]),
            ["-p", "hello"]
        );
        // `run` prepends `effort_args`, so `--effort` arrives in `extra_args`.
        assert_eq!(
            claude_args(
                "local/model",
                &["--effort".into(), "high".into(), "-p".into(), "x".into()]
            ),
            ["--model", "local/model", "--effort", "high", "-p", "x"]
        );
        // A forwarded `--model` replaces llmman's.
        assert_eq!(
            claude_args("local/model", &["--model".into(), "other/model".into()]),
            ["--model", "other/model"]
        );
        // claude has no `-m`, so it is not read as a model.
        assert_eq!(
            claude_args("local/model", &["-m".into(), "not-a-model-flag".into()]),
            ["--model", "local/model", "-m", "not-a-model-flag"]
        );
    }

    #[test]
    fn session_model_is_the_forwarded_one_when_there_is_one() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            session_model("local/model", &args(&["-p", "x"])),
            "local/model"
        );
        assert_eq!(
            session_model("local/model", &args(&["--model", "other/model"])),
            "other/model"
        );
        assert_eq!(
            session_model("local/model", &args(&["--model=other/model"])),
            "other/model"
        );
        assert_eq!(
            session_model("", &args(&["--model", "other/model"])),
            "other/model"
        );
        assert_eq!(session_model("local/model", &args(&["--model="])), "");
        // An alias resolves through the variables, so they keep `model`.
        for alias in ["sonnet", "Opus", "fable[1m]", "opusplan", "default"] {
            assert_eq!(
                session_model("local/model", &args(&["--model", alias])),
                "local/model",
                "{alias}"
            );
        }
        // With no `model`, an alias leaves the variables unset.
        assert_eq!(session_model("", &args(&["--model", "default"])), "");
        assert_eq!(
            session_model("local/model", &args(&["--model", "sonnet-local"])),
            "sonnet-local"
        );
        // After a second `--` it is a message, not a flag.
        assert_eq!(
            session_model(
                "local/model",
                &args(&["-p", "--", "--model", "other/model"])
            ),
            "local/model"
        );
        assert_eq!(
            session_model(
                "local/model",
                &args(&["--model", "other/model", "--", "--model", "x"])
            ),
            "other/model"
        );
    }

    /// A wrong or missing base URL sends the session to api.anthropic.com
    /// instead of the daemon.
    #[test]
    fn env_points_at_the_daemon_and_carries_its_key() {
        let env = claude_env("", "secret", "http://127.0.0.1:17434", None);
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:17434"));
        assert_eq!(get("ANTHROPIC_API_KEY"), Some("secret"));
        assert_eq!(env.len(), 2, "without a model, llmman sets nothing else");
    }

    #[test]
    fn env_pins_every_alias_and_subagents_to_the_resolved_model() {
        let env = claude_env("local/model", "secret", "http://127.0.0.1:17434", None);
        for key in [
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ] {
            let value = env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
            assert_eq!(value, Some("local/model"), "{key}");
        }
        assert_eq!(env.len(), 2 + CLAUDE_MODEL_ENV.len());
    }

    /// claude warns that it does not recognize the model, and assumes 200k
    /// tokens, unless it is told the window.
    #[test]
    fn env_declares_the_models_context_window() {
        let env = claude_env("local/model", "secret", "http://h", Some("262144"));
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"), Some("262144"));
        assert_eq!(env.len(), 3 + CLAUDE_MODEL_ENV.len());
        // Nothing is declared for a window nobody knows, or with no model.
        let env = claude_env("local/model", "secret", "http://h", None);
        assert!(env
            .iter()
            .all(|(n, _)| *n != "CLAUDE_CODE_MAX_CONTEXT_TOKENS"));
        let env = claude_env("", "secret", "http://h", Some("262144"));
        assert!(env
            .iter()
            .all(|(n, _)| *n != "CLAUDE_CODE_MAX_CONTEXT_TOKENS"));
    }

    /// `launch` reads `model`'s live window (which loads it) only for a
    /// session that will run `model`.
    #[test]
    fn runs_model_unless_another_is_forwarded() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(runs_model("local/model", &args(&["-p", "x"])));
        assert!(runs_model("local/model", &args(&["--model", "default"])));
        assert!(runs_model(
            "local/model",
            &args(&["--model", "local/model"])
        ));
        assert!(!runs_model(
            "local/model",
            &args(&["--model", "other/model"])
        ));
        assert!(!runs_model("local/model", &args(&["--model"])));
    }

    #[test]
    fn context_window_is_declared_for_the_session_model_and_only_if_unset() {
        assert_eq!(
            claude_context_window(Some(262144), false).as_deref(),
            Some("262144")
        );
        assert_eq!(claude_context_window(None, false), None);
        assert_eq!(claude_context_window(Some(0), false), None);
        // the caller's own value wins
        assert_eq!(claude_context_window(Some(262144), true), None);
    }

    fn row(reference: &str, window: Option<u64>) -> ModelEntry {
        ModelEntry {
            context_window: window,
            tools: Some(true),
            ..ModelEntry::unlisted(reference)
        }
    }

    fn list() -> ModelList {
        ModelList::of(vec![
            row("docker.io/ai/small", Some(8192)),
            row("llmman.provider/a/huge", Some(1_000_000)),
            row("llmman.provider/a/mid", Some(200_000)),
            row("llmman.provider/b/unknown", None),
        ])
    }

    /// The window is the model the session runs's, not the smallest of the
    /// list (which a tiny model would drag down) and not the start's when
    /// another is forwarded.
    #[test]
    fn the_window_is_the_session_models_own() {
        let models = list();
        // The start: what the daemon serves.
        assert_eq!(
            session_window("docker.io/ai/small", Some(4096), &models),
            Some(4096)
        );
        // A forwarded one: the list's.
        assert_eq!(
            session_window("llmman.provider/a/mid", Some(4096), &models),
            Some(200_000)
        );
        assert_eq!(
            session_window("llmman.provider/b/unknown", Some(4096), &models),
            None
        );
        assert_eq!(session_window("not-listed", Some(4096), &models), None);
    }

    #[test]
    fn the_picker_lists_every_row_in_order_and_sizes_a_1m_row_itself() {
        let picker = picker_settings(&list());
        let options = picker["modelPicker"]["options"].as_array().unwrap();
        assert_eq!(options.len(), 4);
        assert_eq!(picker["modelPicker"]["replaceBuiltInOptions"], true);
        let models: Vec<&str> = options
            .iter()
            .map(|o| o["model"].as_str().unwrap())
            .collect();
        assert_eq!(
            models,
            [
                "docker.io/ai/small",
                "llmman.provider/a/huge[1m]",
                "llmman.provider/a/mid",
                "llmman.provider/b/unknown"
            ]
        );
        assert_eq!(options[0]["label"], "small");
        assert_eq!(options[2]["label"], "a/mid");
        assert_eq!(options[2]["description"], "200k context, tools");
    }

    #[test]
    fn a_forwarded_settings_is_merged_in_both_forms_with_the_picker_kept() {
        let dir = crate::cmd::launch::test_temp_dir("claude-settings");
        let theirs = dir.join("theirs.json");
        std::fs::write(
            &theirs,
            r#"{ "permissions": { "allow": ["Bash(ls)"] }, "modelPicker": { "options": [] } }"#,
        )
        .unwrap();
        let ours = picker_settings(&list());
        let path = theirs.display().to_string();
        for forwarded in [
            vec!["--settings".to_string(), path.clone()],
            vec![format!("--settings={path}")],
            vec![
                "--settings".to_string(),
                r#"{"permissions":{"allow":["Bash(ls)"]}}"#.to_string(),
            ],
        ] {
            let mut args = vec!["--model".to_string(), "m".to_string()];
            args.extend(forwarded);
            args.extend(["-p".to_string(), "hi".to_string()]);
            let merged = merge_settings_in(args, ours.clone(), &dir).unwrap();
            assert_eq!(merged[0], "--settings");
            assert_eq!(&merged[2..], ["--model", "m", "-p", "hi"]);
            assert_eq!(
                merged
                    .iter()
                    .filter(|a| a.starts_with("--settings"))
                    .count(),
                1
            );
            let written: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&merged[1]).unwrap()).unwrap();
            assert_eq!(
                written["permissions"]["allow"],
                serde_json::json!(["Bash(ls)"])
            );
            assert_eq!(
                written["modelPicker"]["options"].as_array().unwrap().len(),
                4
            );
        }
        // After a `--` it is a message, not a flag.
        let args = vec!["--".to_string(), "--settings".to_string(), "x".to_string()];
        let merged = merge_settings_in(args, ours, &dir).unwrap();
        assert_eq!(&merged[2..], ["--", "--settings", "x"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_last_forwarded_settings_is_kept_as_claude_does() {
        let dir = crate::cmd::launch::test_temp_dir("claude-settings-last");
        let ours = picker_settings(&list());
        let first = r#"{"permissions":{"allow":["Bash(*)"],"deny":["Read(x)"]},"env":{"A":"1"}}"#;
        let second = r#"{"permissions":{"allow":[]},"modelPicker":{"options":[{"model":"x"}]}}"#;
        let args = strings(&["--settings", first, "-p", "hi", "--settings", second]);
        let merged = merge_settings_in(args, ours.clone(), &dir).unwrap();
        assert_eq!(&merged[2..], ["-p", "hi"]);
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&merged[1]).unwrap()).unwrap();
        // The earlier one's allow rule and everything else of it is gone.
        assert_eq!(written["permissions"], serde_json::json!({ "allow": [] }));
        assert!(written.get("env").is_none());
        // The picker is ours, not the caller's.
        assert_eq!(written["modelPicker"], ours["modelPicker"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Removal runs last, so an overlap would unset llmman's value.
    #[test]
    fn cleared_env_overlaps_nothing_llmman_sets() {
        let env = claude_env(
            "local/model",
            "secret",
            "http://127.0.0.1:17434",
            Some("262144"),
        );
        for key in CLAUDE_ENV_TO_CLEAR {
            assert!(env.iter().all(|(name, _)| name != key), "{key}");
        }
    }

    fn strings(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// `--optimizer none` is a launch of today: the same arguments and the
    /// same environment, in the same order.
    #[test]
    fn no_optimizer_leaves_the_arguments_and_environment_as_they_were() {
        let added = Optimizer::None.addition("claude").unwrap();
        for extra in [
            strings(&[]),
            strings(&["-p", "hi"]),
            strings(&["--model", "o"]),
        ] {
            let args = claude_args("local/model", &extra);
            let env = claude_env("local/model", "secret", "http://h", Some("1024"));
            let (new_args, new_env) = with_addition(args.clone(), env.clone(), &added);
            assert_eq!(new_args, args);
            assert_eq!(new_env, env);
        }
    }

    #[test]
    fn tk_registers_its_hook_with_a_settings_argument_before_the_rest() {
        let added = Optimizer::Tk
            .addition_with(
                "claude",
                std::path::Path::new("/opt/llmman"),
                std::path::Path::new(""),
            )
            .unwrap();
        let args = claude_args("local/model", &strings(&["-p", "hi"]));
        let env = claude_env("local/model", "secret", "http://h", None);
        let (new_args, new_env) = with_addition(args.clone(), env.clone(), &added);
        assert_eq!(new_args[0], "--settings");
        let settings: serde_json::Value = serde_json::from_str(&new_args[1]).unwrap();
        let hook = &settings["hooks"]["PreToolUse"][0];
        assert_eq!(hook["matcher"], "Bash");
        assert_eq!(hook["hooks"][0]["command"], "'/opt/llmman' tk hook");
        // Everything claude was started with is still there, in order.
        assert_eq!(&new_args[2..], &args[..]);
        assert_eq!(new_env, env);
        // Nothing of the caller's, or of llmman's key, is in the settings.
        assert!(!new_args[1].contains("secret"));
    }

    #[test]
    fn cvmn_appends_its_prompt_before_the_rest() {
        let added = Optimizer::Cvmn.addition("claude").unwrap();
        let args = claude_args("local/model", &strings(&["-p", "hi"]));
        let env = claude_env("local/model", "secret", "http://h", None);
        let (new_args, new_env) = with_addition(args.clone(), env.clone(), &added);
        assert_eq!(new_args[0], "--append-system-prompt");
        assert_eq!(&new_args[2..], &args[..]);
        assert_eq!(new_env, env);
        assert!(!new_args[1].contains("secret"));
    }

    #[test]
    fn a_system_prompt_conflicts_with_cvmn_only_before_a_second_terminator() {
        assert!(cvmn_conflict(&strings(&["-p", "hi"])).is_none());
        assert!(cvmn_conflict(&strings(&["--append-system-prompt-file=x"])).is_some());
        assert!(cvmn_conflict(&strings(&["--system-prompt", "x"])).is_none());
        assert!(cvmn_conflict(&strings(&["--", "--append-system-prompt"])).is_none());
    }

    #[test]
    fn conflicting_forwarded_flags_are_found_only_before_a_second_terminator() {
        assert!(optimizer_conflict(&strings(&["-p", "hi"])).is_none());
        for flag in [
            "--settings",
            "--settings=x.json",
            "--bare",
            "--disallowedTools",
            "--disallowed-tools=Bash",
            "--setting-sources",
            "--setting-sources=user",
        ] {
            assert!(
                optimizer_conflict(&strings(&["-p", "hi", flag])).is_some(),
                "{flag}"
            );
        }
        assert!(optimizer_conflict(&strings(&["--", "--settings", "--bare"])).is_none());
        // `--settings-x` is some other flag.
        assert!(optimizer_conflict(&strings(&["--settings-x"])).is_none());
    }
}
