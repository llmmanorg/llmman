//! `llmman launch aider`.

use super::models::{self, ModelEntry, ModelList};
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::{exec_with_env, server};

/// How many models aider is told of: it parses its model file with a pure-Python JSON5 parser at
/// every start, 1.3 ms a model on aider 0.86.2 (12.6 s for the 10,000-model catalog). `/model
/// openai/<name>` still takes any other model, with a warning that its window is unknown.
const MODEL_FILE_LIMIT: usize = 500;

/// What aider calls a model served through `--openai-api-base`: litellm's
/// `openai/` prefix, which it strips before sending the rest as the model.
fn aider_name(reference: &str) -> String {
    format!("openai/{reference}")
}

/// `entry` as aider's model metadata (litellm's shape). aider reads only `max_input_tokens` and
/// `supports_vision`, so the rest is left out of a file it parses slowly.
fn aider_model(entry: &ModelEntry) -> (String, serde_json::Value) {
    let mut info = serde_json::json!({ "litellm_provider": "openai", "mode": "chat" });
    if let Some(window) = entry.context_window {
        info["max_input_tokens"] = window.into();
    }
    if entry.vision {
        info["supports_vision"] = true.into();
    }
    (aider_name(&entry.reference), info)
}

fn model_metadata(models: &ModelList) -> String {
    let file: serde_json::Map<String, serde_json::Value> =
        models.entries().iter().map(aider_model).collect();
    serde_json::Value::Object(file).to_string()
}

fn aider_args(
    model: &str,
    base_url: &str,
    metadata: Option<&Path>,
    extra_args: &[String],
) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), aider_name(model)]);
    }
    args.extend(["--openai-api-base".to_string(), base_url.to_string()]);
    if let Some(metadata) = metadata {
        args.extend([
            "--model-metadata-file".to_string(),
            metadata.display().to_string(),
        ]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// aider: set OPENAI_API_KEY and OPENAI_BASE_URL, and give it every model in a metadata file so
/// `/models` and `/model` know their windows and image support.
pub(super) fn launch_aider(
    model: &str,
    api_key: &str,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let base_url = format!("{}/v1", server());
    let listed = models.limited(
        "aider",
        MODEL_FILE_LIMIT,
        "aider parses its model file in pure Python at every start, 1.3 ms a model",
    );
    let metadata = if !listed.offers_choice() {
        None
    } else {
        // In its own state directory, which a --sandbox mounts.
        let dir = dirs::home_dir()
            .context("no home directory")?
            .join(".aider");
        Some(models::write_transient(
            &dir,
            "models",
            "json",
            &model_metadata(&listed),
        )?)
    };
    let args = aider_args(model, &base_url, metadata.as_deref(), extra_args);

    exec_with_env(
        &PathBuf::from("aider"),
        &args,
        &[
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url.as_str()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(reference: &str, window: Option<u64>, vision: bool) -> ModelEntry {
        ModelEntry {
            context_window: window,
            vision,
            max_output: Some(8000),
            tools: Some(true),
            ..ModelEntry::unlisted(reference)
        }
    }

    #[test]
    fn every_row_is_one_named_record_with_what_aider_reads() {
        let list = ModelList::of(vec![
            entry("docker.io/ai/qwen3.5:0.8b", Some(32768), false),
            entry("llmman.provider/alpha/agent", Some(200_000), true),
            entry("llmman.provider/beta/unknown", None, false),
        ]);
        let file: serde_json::Value = serde_json::from_str(&model_metadata(&list)).unwrap();
        assert_eq!(file.as_object().unwrap().len(), 3);
        assert_eq!(
            file["openai/docker.io/ai/qwen3.5:0.8b"],
            serde_json::json!({
                "litellm_provider": "openai", "mode": "chat", "max_input_tokens": 32768
            })
        );
        assert_eq!(
            file["openai/llmman.provider/alpha/agent"],
            serde_json::json!({
                "litellm_provider": "openai", "mode": "chat",
                "max_input_tokens": 200_000, "supports_vision": true
            })
        );
        // Nothing known of the window: still a model aider knows of, so it
        // does not warn that the model is unknown.
        assert_eq!(
            file["openai/llmman.provider/beta/unknown"],
            serde_json::json!({ "litellm_provider": "openai", "mode": "chat" })
        );
    }

    #[test]
    fn the_start_is_named_as_its_row_is() {
        let start = entry("llmman.provider/alpha/agent", None, false);
        let (name, _) = aider_model(&start);
        let args = aider_args(&start.reference, "http://h/v1", None, &[]);
        assert_eq!(
            args,
            ["--model", name.as_str(), "--openai-api-base", "http://h/v1"]
        );
    }

    #[test]
    fn the_file_is_passed_by_path_before_the_callers_arguments() {
        let file = Path::new("/home/u/.aider/llmman-models-1.json");
        let extra = vec!["--yes".to_string()];
        let args = aider_args("m", "http://h/v1", Some(file), &extra);
        assert_eq!(
            args,
            [
                "--model",
                "openai/m",
                "--openai-api-base",
                "http://h/v1",
                "--model-metadata-file",
                &file.display().to_string(),
                "--yes"
            ]
        );
        // Without a model there is none to name.
        assert_eq!(
            aider_args("", "http://h/v1", None, &[]),
            ["--openai-api-base", "http://h/v1"]
        );
    }
}
