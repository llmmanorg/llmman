//! `llmman launch droid` (Factory Droid CLI).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::models::{ModelEntry, ModelList};
use super::pi::default_effort;
use super::{exec_with_env, find_on_path, has_forwarded_model_flag, server};

/// Droid expands this reference in `settings.json` from the child process's
/// environment, so a real daemon/provider credential is never persisted.
const DROID_API_KEY_ENV: &str = "LLMMAN_API_KEY";
const DROID_API_KEY_REF: &str = "${LLMMAN_API_KEY}";
const DROID_REASONING_EFFORTS: &[&str] = &["high", "medium", "low", "none"];
/// The levels Droid sends for a custom model: its `-r` and its model
/// selector take no others, and anything else falls back to the entry's.
const DROID_MODEL_EFFORTS: &[&str] = &["low", "medium", "high"];

/// Factory Droid's OpenAI-compatible custom-model mode, pointed at llmman's
/// chat-completions endpoint, with a custom model for every row of `models`.
/// Droid resolves custom models by their generated `custom:<display>-<index>`
/// id, not the wire model name, so the merge writes that id as the
/// interactive default and returns it for `droid exec`. Droid sends the
/// entry's `model`, which is the `reference` the daemon routes, whichever one
/// is picked in its selector.
pub(super) fn launch_droid(
    model: &str,
    api_key: &str,
    _vision: bool,
    _max_output: Option<u32>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_droid().ok_or_else(|| anyhow::anyhow!("droid is not installed"))?;
    let base_url = format!("{}/v1", server());
    let model_id = write_droid_settings(model, &base_url, models)?;
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

fn write_droid_settings(model: &str, base_url: &str, models: &ModelList) -> anyhow::Result<String> {
    write_droid_settings_at(
        &droid_home()?.join("settings.json"),
        model,
        base_url,
        models,
    )
}

fn write_droid_settings_at(
    path: &Path,
    model: &str,
    base_url: &str,
    models: &ModelList,
) -> anyhow::Result<String> {
    let selected = RefCell::new(None);
    common::write_json_merged(path, "droid", |existing| {
        let (merged, id) = droid_settings_merged(existing, model, base_url, models);
        *selected.borrow_mut() = id;
        merged
    })?;
    selected
        .into_inner()
        .context("droid settings could not be updated")
}

/// The levels Droid offers for `entry`, and the one it starts at: `thinking`,
/// a switch-only template's on, being `medium`. A custom model with no
/// `reasoningEffort` has no levels and is sent none.
fn droid_effort(entry: &ModelEntry) -> Option<&'static str> {
    let variants = entry.variants();
    let offered: Vec<&'static str> = DROID_MODEL_EFFORTS
        .iter()
        .copied()
        .filter(|level| {
            variants
                .iter()
                .any(|v| v == level || (*level == "medium" && v == "thinking"))
        })
        .collect();
    default_effort(&offered)
}

/// The window Droid is told: what is left of the model's for a prompt, since
/// the reply counts against it.
fn droid_context_limit(entry: &ModelEntry) -> Option<u64> {
    let window = entry.context_window?;
    let room = window.saturating_sub(u64::from(entry.max_output.unwrap_or(0)));
    (room > 0).then_some(room)
}

/// Merge one llmman-owned model entry per row of `models` without moving or replacing user entries.
/// A matching entry (its `model` is the `reference`) keeps its index, which is part of Droid's
/// custom model id; new ones are appended. Returns the id of the row `selected` names.
fn droid_settings_merged(
    existing: &serde_json::Value,
    selected: &str,
    base_url: &str,
    models: &ModelList,
) -> (serde_json::Value, Option<String>) {
    let mut root = existing.clone();
    let root = root
        .as_object_mut()
        .expect("write_json_merged passes an object");
    let mut selected_id = None;
    {
        let slot = root
            .entry("customModels")
            .or_insert_with(|| serde_json::json!([]));
        if !slot.is_array() {
            *slot = serde_json::json!([]);
        }
        let custom = slot.as_array_mut().expect("set to an array just above");
        let offered: HashSet<&str> = models
            .entries()
            .iter()
            .map(|row| row.reference.as_str())
            .collect();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        // Places of entries llmman wrote that it no longer offers: a new
        // entry takes one before the list grows, since no entry may move.
        let mut free: VecDeque<usize> = VecDeque::new();
        for (index, entry) in custom.iter().enumerate() {
            if let Some(model) = llmman_entry(entry) {
                index_of.entry(model.to_string()).or_insert(index);
                if !offered.contains(model) {
                    free.push_back(index);
                }
            }
        }
        for (row, name) in models.entries().iter().zip(models.display_names()) {
            let matched = index_of.get(&row.reference).copied();
            let index = matched.or_else(|| free.pop_front()).unwrap_or(custom.len());
            // A freed place starts a new entry: nothing of the old one stays.
            let mut entry = custom
                .get(index)
                .filter(|_| matched.is_some())
                .and_then(serde_json::Value::as_object)
                .cloned()
                .unwrap_or_default();
            let display = entry
                .get("displayName")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or(name);
            let id = entry
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| droid_model_id(&display, index));
            entry.insert("model".into(), serde_json::json!(row.reference));
            entry.insert("displayName".into(), serde_json::json!(display));
            entry.insert("baseUrl".into(), serde_json::json!(base_url));
            entry.insert("apiKey".into(), serde_json::json!(DROID_API_KEY_REF));
            entry.insert(
                "provider".into(),
                serde_json::json!("generic-chat-completion-api"),
            );
            // Droid reads images off `noImageSupport` alone, which it
            // defaults to true for this provider.
            entry.insert("noImageSupport".into(), serde_json::json!(!row.vision));
            entry.insert("id".into(), serde_json::json!(id));
            entry.insert("index".into(), serde_json::json!(index));
            if let Some(max_output) = row.max_output {
                entry.insert("maxOutputTokens".into(), serde_json::json!(max_output));
            }
            if let Some(limit) = droid_context_limit(row) {
                entry.insert("maxContextLimit".into(), serde_json::json!(limit));
            }
            // Which also makes Droid offer its levels for the model; a
            // level the user chose for it stays.
            if let Some(effort) = droid_effort(row) {
                entry
                    .entry("reasoningEffort")
                    .or_insert_with(|| serde_json::json!(effort));
            }

            if row.reference == selected {
                selected_id = Some(id);
            }
            let entry = serde_json::Value::Object(entry);
            if index == custom.len() {
                custom.push(entry);
            } else {
                custom[index] = entry;
            }
            index_of.insert(row.reference.clone(), index);
        }
        // What is left of the places freed, at the end of the list, goes: an
        // entry's place is part of its id, so those before a user's entry
        // stay, where taking them out would move it.
        while custom
            .last()
            .and_then(llmman_entry)
            .is_some_and(|model| !offered.contains(model))
        {
            custom.pop();
        }
    }

    if let Some(id) = &selected_id {
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
    (root.clone().into(), selected_id)
}

/// The `model` of an entry llmman wrote, which carries its key reference.
fn llmman_entry(entry: &serde_json::Value) -> Option<&str> {
    let field = |k: &str| entry.get(k).and_then(serde_json::Value::as_str);
    match (field("model"), field("apiKey")) {
        (Some(model), Some(DROID_API_KEY_REF)) => Some(model),
        _ => None,
    }
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

    const URL: &str = "http://127.0.0.1:17434/v1";

    fn entry(provider: Option<&str>, id: &str) -> ModelEntry {
        let reference = match provider {
            Some(p) => crate::providers::format_remote_ref(p, id),
            None => id.to_string(),
        };
        ModelEntry {
            reference,
            id: id.to_string(),
            provider: provider.map(|p| (p.to_string(), p.to_uppercase())),
            context_window: None,
            max_output: None,
            thinking: Some(Vec::new()),
            vision: false,
            tools: None,
            cost: None,
            released: None,
        }
    }

    fn one(entry: ModelEntry) -> ModelList {
        ModelList::of(vec![entry])
    }

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
            URL,
            &one(entry(None, "local/model")),
        );
        assert_eq!(merged["sessionDefaultSettings"]["model"], id.unwrap());
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
        let mut local = entry(None, "local/model");
        local.vision = true;
        let (merged, id) = droid_settings_merged(&existing, "local/model", URL, &one(local));
        let id = id.unwrap();
        assert_eq!(id, "custom:My-local-name-1");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["sessionDefaultSettings"]["model"], id);
        assert_eq!(merged["sessionDefaultSettings"]["reasoningEffort"], "none");
        assert_eq!(merged["sessionDefaultSettings"]["autonomyLevel"], "low");
        assert_eq!(merged["customModels"][0], existing["customModels"][0]);
        let ours = &merged["customModels"][1];
        assert_eq!(ours["displayName"], "My local name");
        assert_eq!(ours["baseUrl"], URL);
        assert_eq!(ours["apiKey"], DROID_API_KEY_REF);
        assert_eq!(ours["noImageSupport"], false);
        assert_eq!(ours["maxOutputTokens"], 1234);
        assert_eq!(ours["extraArgs"]["temperature"], 0.2);
    }

    /// Every row is a custom model sent under its reference, the starting one
    /// the session's default whatever its place; two providers' copy of one
    /// id are told apart by display name and have ids of their own.
    #[test]
    fn every_row_is_a_custom_model_and_the_starting_one_is_the_default() {
        let existing = serde_json::json!({ "customModels": [
            { "model": "user/model", "displayName": "Mine", "apiKey": "${USER_KEY}" },
        ] });
        let list = ModelList::of(vec![
            entry(Some("beta"), "shared"),
            entry(None, "qwen3.5:0.8b"),
            entry(Some("alpha"), "shared"),
        ]);
        let (merged, id) =
            droid_settings_merged(&existing, &list.entries()[0].reference, URL, &list);
        let custom = merged["customModels"].as_array().unwrap();
        assert_eq!(custom.len(), 4);
        assert_eq!(custom[0]["model"], "user/model");
        let models: Vec<&str> = custom[1..]
            .iter()
            .map(|m| m["model"].as_str().unwrap())
            .collect();
        assert_eq!(
            models,
            [
                "llmman.provider/beta/shared",
                "qwen3.5:0.8b",
                "llmman.provider/alpha/shared"
            ]
        );
        let names: Vec<&str> = custom[1..]
            .iter()
            .map(|m| m["displayName"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["beta/shared", "qwen3.5:0.8b", "alpha/shared"]);
        let ids: Vec<&str> = custom[1..]
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            [
                "custom:beta/shared-1",
                "custom:qwen3.5:0.8b-2",
                "custom:alpha/shared-3"
            ]
        );
        assert!(
            custom[0].get("index").is_none(),
            "the user's entry is left as it was"
        );
        for (index, model) in custom.iter().enumerate().skip(1) {
            assert_eq!(model["index"].as_u64(), Some(index as u64));
        }
        assert_eq!(id.as_deref(), Some("custom:beta/shared-1"));
        assert_eq!(
            merged["sessionDefaultSettings"]["model"],
            "custom:beta/shared-1"
        );

        // Again: the same entries at the same places, nothing doubled.
        let (again, same) =
            droid_settings_merged(&merged, &list.entries()[0].reference, URL, &list);
        assert_eq!(again, merged);
        assert_eq!(same, id);
        // Another start moves only the default.
        let (other, id) = droid_settings_merged(&merged, "qwen3.5:0.8b", URL, &list);
        assert_eq!(id.as_deref(), Some("custom:qwen3.5:0.8b-2"));
        assert_eq!(other["customModels"], merged["customModels"]);
    }

    /// An entry llmman wrote that is no longer offered goes where that moves
    /// nothing; a user's entry keeps its place, which is part of its id, so
    /// what stands before it stays, and a new row takes a freed place.
    #[test]
    fn entries_llmman_no_longer_offers_go_and_no_entry_moves() {
        let user = |name: &str| serde_json::json!({ "model": name, "displayName": name, "apiKey": "${USER_KEY}" });
        let first = ModelList::of(vec![
            entry(None, "a"),
            entry(None, "b"),
            entry(None, "c"),
            entry(None, "d"),
        ]);
        let (written, _) = droid_settings_merged(&serde_json::json!({}), "a", URL, &first);
        let mut custom = written["customModels"].as_array().unwrap().clone();
        custom.insert(3, user("mine"));
        let existing = serde_json::json!({ "customModels": custom });
        // `a` and `c` stay offered, `b` and `d` are not, and `e` is new.
        let second = ModelList::of(vec![entry(None, "a"), entry(None, "c"), entry(None, "e")]);
        let (merged, _) = droid_settings_merged(&existing, "a", URL, &second);
        let custom = merged["customModels"].as_array().unwrap();
        let models: Vec<&str> = custom
            .iter()
            .map(|m| m["model"].as_str().unwrap())
            .collect();
        // `e` took the place of `b`; `d`, after the user's entry, is gone
        // from the end, and the user's own kept index 3.
        assert_eq!(models, ["a", "e", "c", "mine"]);
        assert_eq!(custom[1]["index"], 1);
        assert_eq!(custom[1]["id"], "custom:e-1");
        assert!(custom[3].get("index").is_none());
    }

    #[test]
    fn a_row_declares_what_droid_can_take_of_it() {
        let mut agent = entry(Some("alpha"), "agent");
        agent.vision = true;
        agent.context_window = Some(200_000);
        agent.max_output = Some(8000);
        agent.thinking = Some(vec![
            "none".into(),
            "low".into(),
            "high".into(),
            "max".into(),
        ]);
        let plain = entry(None, "plain");
        let (merged, _) = droid_settings_merged(
            &serde_json::json!({}),
            "plain",
            URL,
            &ModelList::of(vec![agent, plain]),
        );
        let agent = &merged["customModels"][0];
        assert_eq!(agent["noImageSupport"], false);
        assert_eq!(agent["maxOutputTokens"], 8000);
        // The reply counts against the window Droid keeps a prompt under.
        assert_eq!(agent["maxContextLimit"], 192_000);
        // The levels Droid takes: the middle one when there is one.
        assert_eq!(agent["reasoningEffort"], "low");
        let plain = &merged["customModels"][1];
        assert_eq!(plain["noImageSupport"], true);
        for field in ["maxOutputTokens", "maxContextLimit", "reasoningEffort"] {
            assert!(plain.get(field).is_none(), "{field}");
        }
        assert_eq!(
            merged["customModels"][1]["id"],
            merged["sessionDefaultSettings"]["model"]
        );

        let levels = |thinking: Option<Vec<&str>>| {
            droid_effort(&ModelEntry {
                thinking: thinking.map(|t| t.into_iter().map(String::from).collect()),
                ..entry(None, "m")
            })
        };
        assert_eq!(levels(None), Some("medium"));
        assert_eq!(levels(Some(vec!["none", "thinking"])), Some("medium"));
        assert_eq!(levels(Some(vec!["high", "xhigh"])), Some("high"));
        assert_eq!(levels(Some(vec!["none", "xhigh", "max"])), None);
        assert_eq!(levels(Some(vec![])), None);
    }

    #[test]
    fn a_level_the_user_chose_for_a_model_stays() {
        let existing = serde_json::json!({ "customModels": [{
            "model": "m", "displayName": "m", "apiKey": DROID_API_KEY_REF,
            "reasoningEffort": "high", "id": "custom:m-0", "index": 0,
        }] });
        let mut m = entry(None, "m");
        m.thinking = None;
        let (merged, _) = droid_settings_merged(&existing, "m", URL, &one(m));
        assert_eq!(merged["customModels"][0]["reasoningEffort"], "high");
    }

    #[test]
    fn settings_write_is_backed_up_and_round_trips() {
        let dir = super::super::test_temp_dir("droid-settings");
        let path = dir.join("settings.json");
        let original = "{\n  // keep my note\n  \"theme\": \"dark\"\n}\n";
        std::fs::write(&path, original).unwrap();
        let mut qwen = entry(None, "docker.io/ai/qwen3.5:0.8b");
        qwen.max_output = Some(8192);
        let list = one(qwen);
        let id = write_droid_settings_at(&path, "docker.io/ai/qwen3.5:0.8b", URL, &list).unwrap();
        assert_eq!(id, "custom:docker.io/ai/qwen3.5:0.8b-0");
        assert_eq!(
            std::fs::read_to_string(path.with_extension("json.bak")).unwrap(),
            original
        );
        let text = std::fs::read_to_string(&path).unwrap();
        let written: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(written["theme"], "dark");
        assert_eq!(written["customModels"][0]["maxOutputTokens"], 8192);
        assert_eq!(written["customModels"][0]["noImageSupport"], true);
        assert_eq!(written["sessionDefaultSettings"]["model"], id);
        assert_eq!(written["sessionDefaultSettings"]["reasoningEffort"], "none");
        // The key is a reference, never a value.
        assert_eq!(written["customModels"][0]["apiKey"], DROID_API_KEY_REF);

        // Idempotent writes do not replace the original commented backup.
        write_droid_settings_at(&path, "docker.io/ai/qwen3.5:0.8b", URL, &list).unwrap();
        assert_eq!(
            std::fs::read_to_string(path.with_extension("json.bak")).unwrap(),
            original
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        std::fs::remove_dir_all(dir).unwrap();
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
