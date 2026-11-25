//! `llmman launch droid` (Factory Droid CLI).

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::{
    exec_with_env, find_on_path, has_forwarded_model_flag, map_offered, server, HostedModel,
};

/// Droid expands this reference in `settings.json` from the child process's
/// environment, so a real daemon/provider credential is never persisted.
const DROID_API_KEY_ENV: &str = "LLMMAN_API_KEY";
const DROID_API_KEY_REF: &str = "${LLMMAN_API_KEY}";
const DROID_REASONING_EFFORTS: &[&str] = &["high", "medium", "low", "none"];

/// Factory Droid's OpenAI-compatible custom-model mode, pointed at llmman's
/// chat-completions endpoint. Droid resolves custom models by their generated
/// `custom:<display>-<index>` id, not the wire model name, so the merge writes
/// that id as the interactive default and returns it for `droid exec`.
///
/// `hosted` is every model `--provider` serves (see [`HostedModel`]):
/// each becomes a `customModels` entry, so Droid's own model picker can
/// switch between them mid-session; the launched model stays the
/// session default.
pub(super) fn launch_droid(
    model: &str,
    api_key: &str,
    vision: bool,
    max_output: Option<u32>,
    extra_args: &[String],
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let bin = find_droid().ok_or_else(|| anyhow::anyhow!("droid is not installed"))?;
    let base_url = format!("{}/v1", server());
    let model_id = write_droid_settings(model, &base_url, vision, max_output, hosted)?;
    exec_with_env(
        &bin,
        &droid_args(&model_id, extra_args),
        &[(DROID_API_KEY_ENV, api_key)],
    )
}

/// The official standalone installer writes to `~/.local/bin`, which a
/// current non-login shell may not have picked up yet.
pub(super) fn find_droid() -> Option<PathBuf> {
    find_on_path("droid").or_else(|| droid_fallback(&crate::config::home_dir()?))
}

fn droid_fallback(home: &Path) -> Option<PathBuf> {
    let bin = if cfg!(windows) { "droid.exe" } else { "droid" };
    let candidate = home.join(".local").join("bin").join(bin);
    candidate.is_file().then_some(candidate)
}

pub(super) fn droid_home() -> anyhow::Result<PathBuf> {
    Ok(crate::config::home_dir()
        .context("no home directory")?
        .join(".factory"))
}

fn write_droid_settings(
    model: &str,
    base_url: &str,
    vision: bool,
    max_output: Option<u32>,
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<String> {
    write_droid_settings_at(
        &droid_home()?.join("settings.json"),
        model,
        base_url,
        vision,
        max_output,
        hosted,
    )
}

fn write_droid_settings_at(
    path: &Path,
    model: &str,
    base_url: &str,
    vision: bool,
    max_output: Option<u32>,
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<String> {
    let selected = RefCell::new(None);
    common::write_json_merged(path, "droid", |existing| {
        let (merged, id) = match hosted {
            // Never empty: the launched model is always offered first.
            Some(hosted) if !hosted.is_empty() => {
                droid_settings_merged_hosted(existing, base_url, hosted)
            }
            _ => droid_settings_merged(existing, model, base_url, vision, max_output),
        };
        *selected.borrow_mut() = Some(id);
        merged
    })?;
    selected
        .into_inner()
        .context("droid settings could not be updated")
}

/// Merge one llmman-owned model entry without moving or replacing any user
/// entries. Its position is part of Droid's documented custom model id, so a
/// matching entry stays at the same index and new entries are appended.
/// Returns the entry's generated id.
fn merge_custom_model(
    root: &mut serde_json::Map<String, serde_json::Value>,
    model: &str,
    base_url: &str,
    vision: bool,
    max_output: Option<u32>,
) -> String {
    let slot = root
        .entry("customModels")
        .or_insert_with(|| serde_json::json!([]));
    if !slot.is_array() {
        *slot = serde_json::json!([]);
    }
    let models = slot.as_array_mut().expect("set to an array just above");
    let index = models
        .iter()
        .position(|entry| {
            entry.get("model").and_then(serde_json::Value::as_str) == Some(model)
                && entry.get("apiKey").and_then(serde_json::Value::as_str)
                    == Some(DROID_API_KEY_REF)
        })
        .unwrap_or(models.len());

    let mut entry = models
        .get(index)
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let display = entry
        .get("displayName")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{model} [llmman]"));
    let id = entry
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| droid_model_id(&display, index));
    entry.insert("model".into(), serde_json::json!(model));
    entry.insert("displayName".into(), serde_json::json!(display));
    entry.insert("baseUrl".into(), serde_json::json!(base_url));
    entry.insert("apiKey".into(), serde_json::json!(DROID_API_KEY_REF));
    entry.insert(
        "provider".into(),
        serde_json::json!("generic-chat-completion-api"),
    );
    entry.insert("supportsImages".into(), serde_json::json!(vision));
    entry.insert("id".into(), serde_json::json!(id));
    entry.insert("index".into(), serde_json::json!(index));
    if let Some(max_output) = max_output {
        entry.insert("maxOutputTokens".into(), serde_json::json!(max_output));
    }

    let entry = serde_json::Value::Object(entry);
    if index == models.len() {
        models.push(entry);
    } else {
        models[index] = entry;
    }
    id
}

/// The session default to `id`, keeping a valid `reasoningEffort` the
/// user already set.
fn set_session_default(root: &mut serde_json::Map<String, serde_json::Value>, id: &str) {
    let session = root
        .entry("sessionDefaultSettings")
        .or_insert_with(|| serde_json::json!({}));
    if !session.is_object() {
        *session = serde_json::json!({});
    }
    let session = session
        .as_object_mut()
        .expect("set to an object just above");
    session.insert("model".into(), serde_json::json!(id));
    let valid_effort = session
        .get("reasoningEffort")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|effort| DROID_REASONING_EFFORTS.contains(&effort));
    if !valid_effort {
        session.insert("reasoningEffort".into(), serde_json::json!("none"));
    }
}

fn droid_settings_merged(
    existing: &serde_json::Value,
    model: &str,
    base_url: &str,
    vision: bool,
    max_output: Option<u32>,
) -> (serde_json::Value, String) {
    let mut root = existing.clone();
    let root = root
        .as_object_mut()
        .expect("write_json_merged passes an object");
    let id = merge_custom_model(root, model, base_url, vision, max_output);
    set_session_default(root, &id);
    (root.clone().into(), id)
}

/// [`droid_settings_merged`] for every model `--provider` offers: each
/// becomes a `customModels` entry, mapped alike from its own row (see
/// [`map_offered`]) — image input from its own `vision`, its own reply
/// ceiling — and the launched model, `hosted[0]`, stays the session
/// default.
fn droid_settings_merged_hosted(
    existing: &serde_json::Value,
    base_url: &str,
    hosted: &[HostedModel],
) -> (serde_json::Value, String) {
    let mut root = existing.clone();
    let root = root
        .as_object_mut()
        .expect("write_json_merged passes an object");
    let ids = map_offered(hosted, |m, _| {
        merge_custom_model(
            root,
            &m.reference,
            base_url,
            m.vision.unwrap_or(false),
            m.output,
        )
    });
    let launched_id = ids
        .into_iter()
        .next()
        .expect("write_droid_settings_at passes a non-empty list");
    set_session_default(root, &launched_id);
    (root.clone().into(), launched_id)
}

fn droid_model_id(display_name: &str, index: usize) -> String {
    format!("custom:{}-{index}", display_name.replace(' ', "-"))
}

/// Only `droid exec` accepts `--model`; interactive Droid reads the generated
/// id from `sessionDefaultSettings`. A model supplied to `exec` by the caller
/// wins and is passed through unchanged.
fn droid_args(model_id: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = extra_args.to_vec();
    if args.first().is_none_or(|arg| arg != "exec") || has_forwarded_model_flag("droid", extra_args)
    {
        return args;
    }
    args.splice(1..1, ["--model".to_string(), model_id.to_string()]);
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_select_the_custom_id_and_yield_to_a_forwarded_model() {
        assert_eq!(
            droid_args("custom:local-model-0", &["exec".into(), "hello".into()]),
            ["exec", "--model", "custom:local-model-0", "hello"]
        );
        assert!(droid_args("custom:local-model-0", &[]).is_empty());
        assert_eq!(
            droid_args("custom:local-model-0", &["--help".into()]),
            ["--help"]
        );
        assert_eq!(
            droid_args(
                "custom:local-model-0",
                &[
                    "exec".into(),
                    "--model".into(),
                    "custom:user-model-2".into(),
                    "hello".into(),
                ]
            ),
            ["exec", "--model", "custom:user-model-2", "hello"]
        );
    }

    #[test]
    fn settings_merge_keeps_a_valid_reasoning_effort() {
        let existing = serde_json::json!({
            "sessionDefaultSettings": {
                "model": "old-default",
                "reasoningEffort": "high",
                "autonomyLevel": "low"
            }
        });
        let (merged, id) = droid_settings_merged(
            &existing,
            "local/model",
            "http://127.0.0.1:17434/v1",
            false,
            None,
        );
        assert_eq!(merged["sessionDefaultSettings"]["model"], id);
        assert_eq!(merged["sessionDefaultSettings"]["reasoningEffort"], "high");
        assert_eq!(merged["sessionDefaultSettings"]["autonomyLevel"], "low");
    }

    #[test]
    fn settings_merge_preserves_user_models_and_fields() {
        let existing = serde_json::json!({
            "theme": "dark",
            "sessionDefaultSettings": {
                "model": "old-default",
                "reasoningEffort": "invalid",
                "autonomyLevel": "low"
            },
            "customModels": [
                {
                    "model": "user/model",
                    "displayName": "Mine",
                    "baseUrl": "https://example.test/v1",
                    "apiKey": "${USER_KEY}",
                    "provider": "generic-chat-completion-api"
                },
                {
                    "model": "local/model",
                    "displayName": "My local name",
                    "baseUrl": "http://old.test/v1",
                    "apiKey": DROID_API_KEY_REF,
                    "provider": "generic-chat-completion-api",
                    "maxOutputTokens": 1234,
                    "extraArgs": {"temperature": 0.2},
                    "id": "custom:My-local-name-1",
                    "index": 1
                }
            ]
        });
        let (merged, id) = droid_settings_merged(
            &existing,
            "local/model",
            "http://127.0.0.1:17434/v1",
            true,
            None,
        );
        assert_eq!(id, "custom:My-local-name-1");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["sessionDefaultSettings"]["model"], id);
        assert_eq!(merged["sessionDefaultSettings"]["reasoningEffort"], "none");
        assert_eq!(merged["sessionDefaultSettings"]["autonomyLevel"], "low");
        assert_eq!(merged["customModels"][0], existing["customModels"][0]);
        let ours = &merged["customModels"][1];
        assert_eq!(ours["displayName"], "My local name");
        assert_eq!(ours["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(ours["apiKey"], DROID_API_KEY_REF);
        assert_eq!(ours["supportsImages"], true);
        assert_eq!(ours["maxOutputTokens"], 1234);
        assert_eq!(ours["extraArgs"]["temperature"], 0.2);
    }

    #[test]
    fn settings_write_is_backed_up_and_round_trips() {
        let dir = super::super::test_temp_dir("droid-settings");
        let path = dir.join("settings.json");
        let original = "{\n  // keep my note\n  \"theme\": \"dark\"\n}\n";
        std::fs::write(&path, original).unwrap();
        let id = write_droid_settings_at(
            &path,
            "docker.io/ai/qwen3.5:0.8b",
            "http://127.0.0.1:17434/v1",
            false,
            Some(8192),
            None,
        )
        .unwrap();
        assert_eq!(id, "custom:docker.io/ai/qwen3.5:0.8b-[llmman]-0");
        assert_eq!(
            std::fs::read_to_string(path.with_extension("json.bak")).unwrap(),
            original
        );
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["theme"], "dark");
        assert_eq!(written["customModels"][0]["maxOutputTokens"], 8192);
        assert_eq!(written["customModels"][0]["supportsImages"], false);
        assert_eq!(written["sessionDefaultSettings"]["model"], id);
        assert_eq!(written["sessionDefaultSettings"]["reasoningEffort"], "none");

        // Idempotent writes do not replace the original commented backup.
        write_droid_settings_at(
            &path,
            "docker.io/ai/qwen3.5:0.8b",
            "http://127.0.0.1:17434/v1",
            false,
            Some(8192),
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(path.with_extension("json.bak")).unwrap(),
            original
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn hosted_pair() -> Vec<HostedModel> {
        vec![
            HostedModel {
                reference: "llmman.provider/openrouter/qwen3-coder".into(),
                id: "qwen3-coder".into(),
                context: Some(262_144),
                output: Some(8_192),
                thinking: Some(vec!["low".into(), "high".into()]),
                vision: Some(false),
            },
            HostedModel {
                reference: "llmman.provider/openrouter/llama-4-scout".into(),
                id: "llama-4-scout".into(),
                context: None,
                output: None,
                thinking: None,
                vision: Some(true),
            },
        ]
    }

    /// A `--provider` launch lists every model the provider serves, so
    /// Droid's own model picker can switch between them without a
    /// relaunch: every model, the launched one first, with the image
    /// input and reply ceiling its own catalog row says. The launched model stays
    /// the session default, and a matching entry keeps its index — and
    /// so its generated id — across launches.
    #[test]
    fn hosted_models_each_get_a_custom_model_entry() {
        let existing = serde_json::json!({
            "theme": "dark",
            "sessionDefaultSettings": {
                "model": "custom:llmman.provider-openrouter-qwen3-coder-[llmman]-1",
                "reasoningEffort": "high"
            },
            "customModels": [
                {
                    "model": "user/model",
                    "displayName": "Mine",
                    "baseUrl": "https://example.test/v1",
                    "apiKey": "${USER_KEY}",
                    "provider": "generic-chat-completion-api"
                },
                {
                    "model": "llmman.provider/openrouter/qwen3-coder",
                    "displayName": "llmman.provider/openrouter/qwen3-coder [llmman]",
                    "baseUrl": "http://old.test/v1",
                    "apiKey": DROID_API_KEY_REF,
                    "provider": "generic-chat-completion-api",
                    "id": "custom:llmman.provider-openrouter-qwen3-coder-[llmman]-1",
                    "index": 1
                }
            ]
        });
        let (merged, id) =
            droid_settings_merged_hosted(&existing, "http://127.0.0.1:17434/v1", &hosted_pair());
        // The launched entry kept its index, and so its id.
        assert_eq!(
            id,
            "custom:llmman.provider-openrouter-qwen3-coder-[llmman]-1"
        );
        assert_eq!(merged["sessionDefaultSettings"]["model"], id);
        assert_eq!(merged["sessionDefaultSettings"]["reasoningEffort"], "high");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["customModels"][0], existing["customModels"][0]);
        let ours = &merged["customModels"][1];
        assert_eq!(ours["model"], "llmman.provider/openrouter/qwen3-coder");
        assert_eq!(ours["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(ours["supportsImages"], false);
        assert_eq!(ours["maxOutputTokens"], 8_192);
        let scout = &merged["customModels"][2];
        assert_eq!(scout["model"], "llmman.provider/openrouter/llama-4-scout");
        assert_eq!(scout["supportsImages"], true);
        assert_eq!(scout["maxOutputTokens"], serde_json::Value::Null);
        assert_eq!(scout["index"], 2);
        assert_eq!(merged["customModels"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn fallback_finds_the_official_installers_target() {
        let home = super::super::test_temp_dir("droid-bin");
        let name = if cfg!(windows) { "droid.exe" } else { "droid" };
        let bin = home.join(".local").join("bin").join(name);
        std::fs::create_dir_all(bin.parent().unwrap()).unwrap();
        assert_eq!(droid_fallback(&home), None);
        std::fs::write(&bin, b"stub").unwrap();
        assert_eq!(droid_fallback(&home), Some(bin));
        std::fs::remove_dir_all(home).unwrap();
    }
}
