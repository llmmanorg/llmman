//! `llmman launch agy` (Antigravity CLI).

use super::models::{self, ModelEntry, ModelList};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Context;
use base64::Engine as _;

use super::{exec_with_env, find_on_path, server};

/// AGY speaks Gemini's native generation protocol. The encoded model in the
/// base URL is llmman's routing instruction; AGY also makes auxiliary calls
/// with its own hard-coded model names, so the server deliberately ignores
/// the model segment AGY appends and sends every call to the model selected
/// here.
///
/// The models it can switch to are `customModelsConfig` in a settings file
/// for this session alone (`--settings`): a model picked in `/model` is sent
/// as `.../v1beta/models/<reference>:...` under that base URL. An AGY that
/// has no `--settings` (1.2.2 stops with "flags provided but not defined")
/// is launched without the list, on the one model it is pinned to.
pub(super) fn launch_agy(
    model: &str,
    api_key: &str,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("agy").ok_or_else(|| anyhow::anyhow!("agy is not installed"))?;
    anyhow::ensure!(
        !extra_args
            .iter()
            .any(|arg| matches!(arg.split('=').next(), Some("--gemini_dir" | "-gemini_dir"))),
        "llmman manages AGY’s --gemini_dir"
    );
    let gemini_dir = agy_settings_dir()?;
    write_agy_settings_at(&gemini_dir)?;
    let mut args = vec![format!("--gemini_dir={}", gemini_dir.display())];
    let labels = models.unique_keys(ModelEntry::label);
    if let Some(start) = labels.first().filter(|_| agy_takes_settings(&bin)) {
        let file = models::write_transient(
            &gemini_dir,
            "models",
            "json",
            &session_settings(models, &labels),
        )?;
        args.extend([
            "--settings".to_string(),
            file.display().to_string(),
            "--model".to_string(),
            start.clone(),
        ]);
    }
    args.extend_from_slice(extra_args);

    let base_url = pinned_base_url(model);
    exec_with_env(
        &bin,
        &args,
        &[
            ("GOOGLE_GEMINI_BASE_URL", base_url.as_str()),
            ("GEMINI_API_KEY", api_key),
            // AGY prefers GOOGLE_API_KEY when both names exist. Override it
            // too so an unrelated key inherited from the shell cannot bypass
            // the credential llmman selected for this request.
            ("GOOGLE_API_KEY", api_key),
        ],
    )
}

/// How long to wait for `agy --help`: it answers in well under a second,
/// so a slower one is treated as not offering the flag rather than held up.
const HELP_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Whether this AGY takes `--settings`, read from its `--help`. Any failure to ask counts as no,
/// with one line on stderr: the launch never depends on the answer.
fn agy_takes_settings(bin: &Path) -> bool {
    let takes = match run_help(bin, HELP_PROBE_TIMEOUT) {
        Some(help) => help_lists_flag(&help, "--settings"),
        None => false,
    };
    if !takes {
        eprintln!(
            "llmman: this agy has no --settings flag, so it starts on the selected model \
             alone, without the model list (update agy to switch models inside it)"
        );
    }
    takes
}

/// `bin --help`, stdout then stderr (AGY prints its usage on stderr), or
/// `None` when it cannot be run or does not finish within `limit`.
fn run_help(bin: &Path, limit: Duration) -> Option<String> {
    let mut child = Command::new(bin)
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    // A background process of the child can hold the pipes open past its exit.
    let left = || deadline.saturating_duration_since(Instant::now());
    let stdout = stdout.recv_timeout(left()).ok()?;
    let stderr = stderr.recv_timeout(left()).ok()?;
    Some(stdout + &stderr)
}

/// Everything `pipe` yields, read on a thread so a full pipe cannot stall the
/// child, and sent once the pipe closes.
fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        let _ = sender.send(String::from_utf8_lossy(&bytes).into_owned());
    });
    receiver
}

/// Whether `help` lists `flag` as a flag of its own, not as the start of a
/// longer name (`--settings-file`).
fn help_lists_flag(help: &str, flag: &str) -> bool {
    help.match_indices(flag).any(|(at, _)| {
        let before = help[..at].chars().next_back();
        let after = help[at + flag.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric() || c == '-')
            && !after.is_some_and(|c| c.is_alphanumeric() || c == '-' || c == '_')
    })
}

/// The Gemini base URL that names the model llmman routes to (see
/// `handle_pinned_gemini`), encoded because an OCI reference holds `/`.
pub(super) fn pinned_base_url(model: &str) -> String {
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(model.as_bytes());
    format!("{}/gemini/{encoded}", server())
}

/// `entry` as one of AGY's custom models: shown by `label`, sent as its reference. No thinking
/// level: the server ignores the Gemini API's `thinkingConfig` (see `VARIANT_UNSUPPORTED`), so one
/// would offer a choice that does nothing.
fn agy_model(entry: &ModelEntry, label: &str) -> serde_json::Value {
    let mut model = serde_json::json!({
        "modelName": format!("gemini-api://{}", entry.reference),
        "label": label,
    });
    if let Some(max_output) = entry.max_output {
        model["maxOutputTokens"] = max_output.into();
    }
    model
}

/// The settings `--settings` takes, one custom model per row, keyed by the
/// label each is shown by.
fn session_settings(models: &ModelList, labels: &[String]) -> String {
    let custom_models: serde_json::Map<String, serde_json::Value> = models
        .entries()
        .iter()
        .zip(labels)
        .map(|(entry, label)| (label.clone(), agy_model(entry, label)))
        .collect();
    serde_json::json!({ "customModelsConfig": { "customModels": custom_models } }).to_string()
}

pub(super) fn agy_settings_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".gemini")
        .join("llmman"))
}

fn write_agy_settings_at(gemini_dir: &Path) -> anyhow::Result<()> {
    let settings_path = gemini_dir.join("antigravity-cli").join("settings.json");
    super::common::write_atomic_in_dir(&settings_path, b"{\n  \"modelProvider\": \"gemini\"\n}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_text_decides_whether_the_flag_is_offered() {
        let v122 = "Usage of agy:\n  --model    Model for the current CLI session\n  --sandbox  Run in a sandbox\n";
        let v15 = format!("{v122}  --settings  Settings for this session only\n");
        assert!(!help_lists_flag(v122, "--settings"));
        assert!(help_lists_flag(&v15, "--settings"));
        assert!(help_lists_flag("  -settings string\n", "-settings"));
        assert!(!help_lists_flag(
            "  --settings-file x\n  --my--settings\n",
            "--settings"
        ));
        assert!(!help_lists_flag("", "--settings"));
    }

    #[cfg(unix)]
    #[test]
    fn a_help_probe_that_hangs_or_is_missing_answers_no() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("llmman-agy-help-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = |name: &str, body: &str| {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        };
        let quick = script("quick", "echo '  --settings  x' >&2");
        let hang = script("hang", "exec sleep 30");
        let started = Instant::now();
        assert!(run_help(&quick, Duration::from_secs(5))
            .unwrap()
            .contains("--settings"));
        assert!(run_help(&hang, Duration::from_millis(300)).is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(run_help(&dir.join("missing"), Duration::from_secs(1)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A wrapper that exits while a background process keeps its output pipes
    /// open must not hold the launch past the probe's limit.
    #[cfg(unix)]
    #[test]
    fn a_help_probe_whose_pipes_outlive_the_child_answers_no_at_the_limit() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("llmman-agy-bg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("agy");
        std::fs::write(&path, "#!/bin/sh\nsleep 5 &\necho '  --settings  x'\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = Instant::now();
        assert!(run_help(&path, Duration::from_secs(1)).is_none());
        assert!(started.elapsed() < Duration::from_secs(4));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agy_settings_are_written_to_the_llmman_owned_directory() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-agy-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_agy_settings_at(&dir).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("antigravity-cli/settings.json")).unwrap(),
            "{\n  \"modelProvider\": \"gemini\"\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_row_is_a_custom_model_named_by_its_label_and_sent_as_its_reference() {
        let mut agent = ModelEntry::unlisted("llmman.provider/alpha/agent");
        agent.max_output = Some(8000);
        let list = ModelList::of(vec![
            ModelEntry::unlisted("docker.io/ai/qwen3.5:0.8b"),
            agent,
            // The same id from another provider is told apart, not lost.
            ModelEntry::unlisted("llmman.provider/beta/agent"),
        ]);
        let labels = list.unique_keys(ModelEntry::label);
        assert_eq!(labels, ["qwen3.5:0.8b", "alpha/agent", "beta/agent"]);
        let settings: serde_json::Value =
            serde_json::from_str(&session_settings(&list, &labels)).unwrap();
        let models = &settings["customModelsConfig"]["customModels"];
        assert_eq!(models.as_object().unwrap().len(), 3);
        assert_eq!(
            models["qwen3.5:0.8b"],
            serde_json::json!({
                "modelName": "gemini-api://docker.io/ai/qwen3.5:0.8b",
                "label": "qwen3.5:0.8b"
            })
        );
        assert_eq!(
            models["alpha/agent"],
            serde_json::json!({
                "modelName": "gemini-api://llmman.provider/alpha/agent",
                "label": "alpha/agent",
                "maxOutputTokens": 8000
            })
        );
        assert_eq!(
            models["beta/agent"]["modelName"],
            "gemini-api://llmman.provider/beta/agent"
        );
    }

    #[test]
    fn a_label_two_rows_share_is_made_unique() {
        let list = ModelList::of(vec![
            ModelEntry::unlisted("a"),
            ModelEntry {
                id: "a".into(),
                ..ModelEntry::unlisted("b")
            },
        ]);
        let labels = list.unique_keys(ModelEntry::label);
        assert_eq!(labels, ["a", "a-2"]);
        let settings: serde_json::Value =
            serde_json::from_str(&session_settings(&list, &labels)).unwrap();
        let models = &settings["customModelsConfig"]["customModels"];
        assert_eq!(models["a"]["modelName"], "gemini-api://a");
        assert_eq!(models["a-2"]["modelName"], "gemini-api://b");
        assert_eq!(models["a-2"]["label"], "a-2");
    }
}
