//! `llmman launch pi`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::Context;

use super::common;
use super::models::{remember_rows, rows_written, ModelEntry, ModelList};
use super::{exec_with_env, find_on_path, server};

/// pi: register llmman as an OpenAI-compatible provider in `models.json`,
/// with every model of `models` in it, and point `settings.json` at the
/// one it starts on. pi runs with the caller's own arguments and nothing
/// else — `settings.json` already selects the model, and a key on argv
/// would be readable in `ps`. Switching with `/model` or `Ctrl+P` sends the
/// row's `id`, which is the `reference` the daemon routes.
///
/// pi is in [`super::PROVIDER_NEEDS_DAEMON_KEY`] for that reason: a `--provider`
/// credential stays with the daemon, and the stored `apiKey` is the
/// literal placeholder codex and hermes also write. An environment
/// reference (`"$VAR"`, which pi does interpolate) would read as
/// "unresolved" outside `llmman launch` and take the provider down.
pub(super) fn launch_pi(
    model: &str,
    _reasons: bool,
    _vision: bool,
    _context_window: Option<u64>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("pi").ok_or_else(|| anyhow::anyhow!("pi is not installed"))?;
    write_pi_config(model, models)?;
    exec_with_env(&bin, extra_args, &[])
}

/// The provider key llmman owns in pi's `models.json`.
const PI_PROVIDER: &str = "llmman";

/// pi's config directory: `PI_CODING_AGENT_DIR`, else `~/.pi/agent`.
///
/// The home half is [`super::cline::cline_dir`]'s, not
/// [`super::qwen::qwen_home`]'s: pi is node as Cline is, so
/// `os.homedir()` reads `USERPROFILE` on Windows — see `cline_dir`'s own
/// doc comment for what disagreeing there cost. The `~` handling is
/// qwen's, for the quoted export that leaves one behind.
pub(super) fn pi_agent_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = common::configured_pi_agent_dir()? {
        return Ok(dir);
    }
    Ok(crate::config::home_dir()
        .context("no home directory")?
        .join(".pi")
        .join("agent"))
}

/// Writes pi's `models.json` provider and points `settings.json` at the model the launch starts on.
/// The models come first: a write that fails leaves no setting that starts pi on a model it was not
/// told of.
fn write_pi_config(model: &str, models: &ModelList) -> anyhow::Result<()> {
    write_pi_config_at(&pi_agent_dir()?, model, &server(), models)
}

fn write_pi_config_at(
    dir: &std::path::Path,
    model: &str,
    server: &str,
    models: &ModelList,
) -> anyhow::Result<()> {
    let rows = pi_models(models);
    let stale = rows_written(dir, "pi");
    common::write_json_merged(&dir.join("models.json"), "pi", |existing| {
        pi_models_merged(existing, server, &rows, &stale)
    })?;
    remember_rows(dir, "pi", &rows)?;
    common::write_json_merged(&dir.join("settings.json"), "pi", |existing| {
        pi_settings_merged(existing, model)
    })
}

/// The level at which the model starts when an agent insists on one:
/// `medium`, else the lowest offered.
pub(super) fn default_effort<'a>(levels: &[&'a str]) -> Option<&'a str> {
    levels
        .iter()
        .copied()
        .find(|level| *level == "medium")
        .or_else(|| levels.first().copied())
}

/// Every row of `models` as a pi model.
fn pi_models(models: &ModelList) -> Vec<serde_json::Value> {
    models.map_named(pi_model)
}

/// One model: what `models.md` says a custom model may declare, of what
/// the list knows. pi has no tool-calling switch, so `tools` has no place.
fn pi_model(entry: &ModelEntry, name: &str) -> serde_json::Value {
    let levels = pi_thinking_levels(&entry.variants());
    let mut row = common::pi_model_entry(
        &entry.reference,
        levels.is_some(),
        entry.vision,
        entry.context_window,
    );
    row["name"] = serde_json::json!(name);
    if let Some(max) = entry.max_output {
        row["maxTokens"] = serde_json::json!(max);
    }
    if let Some(levels) = levels {
        row["thinkingLevelMap"] = levels;
    }
    row
}

/// pi's thinking levels, lowest first.
pub(super) const PI_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// `variant` as pi (and omp, which shares the levels) spells it. `xhigh` and
/// `max` are the levels they offer only for a model that names them, which
/// is what [`pi_thinking_levels`] does.
pub(super) fn pi_level<'a>(integration: &str, variant: &'a str) -> Option<&'a str> {
    match variant {
        "xhigh" | "max" => Some(variant),
        _ => super::spell_variant(integration, variant).ok(),
    }
}

/// The `thinkingLevelMap` that offers the levels `variants` spells and sends
/// llmman's own word for each (`none` for off, which pi otherwise leaves
/// out of the request); `None` for a model with no level to choose.
fn pi_thinking_levels(variants: &[String]) -> Option<serde_json::Value> {
    let offered: HashSet<&str> = variants.iter().filter_map(|v| pi_level("pi", v)).collect();
    if offered.iter().all(|level| *level == "off") {
        return None;
    }
    let mut map = serde_json::Map::new();
    for level in PI_LEVELS {
        let value = match (offered.contains(level), level) {
            (true, "off") => serde_json::json!("none"),
            (true, "xhigh" | "max") => serde_json::json!(level),
            (false, "xhigh" | "max") | (true, _) => continue,
            (false, _) => serde_json::Value::Null,
        };
        map.insert(level.to_string(), value);
    }
    Some(map.into())
}

/// `existing` with llmman's provider updated around `rows`, keeping
/// everything it does not own.
///
/// The provider object is merged, not replaced: `baseUrl` always points
/// at the running daemon, `api` and `apiKey` are filled in only when
/// absent, and anything else under it — `headers`, `compat`,
/// `modelOverrides` — is left as the user wrote it. Only the models with
/// the `id` of a row are rebuilt, and those an earlier launch wrote that
/// are no longer offered (`stale`) are dropped; what the user added stays.
fn pi_models_merged(
    existing: &serde_json::Value,
    server: &str,
    rows: &[serde_json::Value],
    stale: &HashSet<String>,
) -> serde_json::Value {
    provider_with_models(existing, PI_PROVIDER, rows, false, stale, |provider| {
        provider.insert("baseUrl".into(), serde_json::json!(format!("{server}/v1")));
        provider
            .entry("api")
            .or_insert_with(|| serde_json::json!("openai-completions"));
        // A literal, not a `"$VAR"` reference pi would interpolate: see
        // launch_pi's own doc comment.
        provider
            .entry("apiKey")
            .or_insert_with(|| serde_json::json!("llmman"));
    })
}

/// `existing` with provider `name` holding `rows` as well as its earlier models, except those
/// `stale` names: a row replaces the model of its `id` in place (built on its fields if
/// `keep_fields`), the others follow in order. One pass, for a list of thousands.
pub(super) fn provider_with_models(
    existing: &serde_json::Value,
    name: &str,
    rows: &[serde_json::Value],
    keep_fields: bool,
    stale: &HashSet<String>,
    configure: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> serde_json::Value {
    let mut root = existing.clone();
    let Some(root_map) = root.as_object_mut() else {
        return serde_json::json!({});
    };
    let provider = common::object_under(common::object_under(root_map, "providers"), name);
    let old: Vec<serde_json::Value> = provider
        .get("models")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let id_of =
        |model: &serde_json::Value| model.get("id").and_then(|id| id.as_str().map(String::from));
    let by_id: HashMap<String, &serde_json::Value> = rows
        .iter()
        .filter_map(|row| Some((id_of(row)?, row)))
        .collect();
    let mut placed: HashSet<String> = HashSet::new();
    let mut models = Vec::with_capacity(old.len() + rows.len());
    for model in old {
        let Some((id, row)) = id_of(&model).and_then(|id| by_id.get(&id).map(|row| (id, *row)))
        else {
            if !id_of(&model).is_some_and(|id| stale.contains(&id)) {
                models.push(model);
            }
            continue;
        };
        if !placed.insert(id) {
            continue;
        }
        let mut merged = if keep_fields {
            model
        } else {
            serde_json::json!({})
        };
        match (merged.as_object_mut(), row.as_object()) {
            (Some(merged), Some(row)) => merged.extend(row.clone()),
            _ => merged = row.clone(),
        }
        models.push(merged);
    }
    for row in rows {
        if id_of(row).is_none_or(|id| placed.insert(id)) {
            models.push(row.clone());
        }
    }
    configure(provider);
    provider.insert("models".into(), serde_json::json!(models));
    root
}

/// `existing` with pi's startup provider and model pointed at this launch,
/// leaving every other setting alone.
fn pi_settings_merged(existing: &serde_json::Value, model: &str) -> serde_json::Value {
    let mut root = existing.clone();
    root["defaultProvider"] = serde_json::json!(PI_PROVIDER);
    root["defaultModel"] = serde_json::json!(model);
    root
}

#[cfg(test)]
mod tests {
    use super::super::{env_dir, node_user_profile};
    use super::*;

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

    fn ids(merged: &serde_json::Value) -> Vec<&str> {
        merged["providers"]["llmman"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect()
    }

    fn rows(entries: Vec<ModelEntry>) -> Vec<serde_json::Value> {
        pi_models(&ModelList::of(entries))
    }

    /// pi is node, so its home half is the one node reads — the same
    /// `cline_dir` resolves, and the reason #535 stopped using
    /// `dirs::home_dir` alone. `config::home_dir` reads `%USERPROFILE%`
    /// first for that same reason, so one helper answers both.
    #[test]
    fn pi_agent_dir_reads_the_home_node_reads() {
        assert_eq!(
            crate::config::home_dir(),
            node_user_profile().or_else(dirs::home_dir)
        );
        if cfg!(windows) {
            assert_eq!(node_user_profile(), env_dir("USERPROFILE"));
        } else {
            assert_eq!(node_user_profile(), None);
        }
    }

    #[test]
    fn pi_models_merged_keeps_other_providers_and_hand_added_models() {
        let existing: serde_json::Value = serde_json::from_str(
            r#"{
              "providers": {
                "other": { "baseUrl": "https://example.invalid/v1" },
                "llmman": {
                  "headers": { "X-Trace": "1" },
                  "apiKey": "mine-not-yours",
                  "models": [{ "id": "mine" }, { "id": "stale" }]
                }
              }
            }"#,
        )
        .unwrap();
        let merged = pi_models_merged(
            &existing,
            "http://127.0.0.1:17434",
            &rows(vec![entry(None, "qwen3.5:0.8b")]),
            &HashSet::new(),
        );

        assert_eq!(
            merged["providers"]["other"]["baseUrl"],
            "https://example.invalid/v1"
        );
        assert_eq!(
            merged["providers"]["llmman"]["baseUrl"],
            "http://127.0.0.1:17434/v1"
        );
        // Keys llmman does not own survive the rewrite, and an apiKey
        // the user already set is left alone.
        assert_eq!(merged["providers"]["llmman"]["headers"]["X-Trace"], "1");
        assert_eq!(merged["providers"]["llmman"]["apiKey"], "mine-not-yours");
        // Both survive; llmman's own is appended, and rebuilt in place after.
        assert_eq!(ids(&merged), vec!["mine", "stale", "qwen3.5:0.8b"]);
    }

    /// A file may hold `"providers": []` and still be valid JSON;
    /// indexing that with a key panics, so it is replaced rather than
    /// indexed (see `object_under`).
    #[test]
    fn pi_models_merged_survives_a_wrong_shaped_providers_value() {
        for raw in [r#"{"providers": []}"#, r#"{"providers": "x"}"#, r#"{}"#] {
            let existing: serde_json::Value = serde_json::from_str(raw).unwrap();
            let merged = pi_models_merged(
                &existing,
                "http://s",
                &rows(vec![entry(None, "m")]),
                &HashSet::new(),
            );
            assert_eq!(merged["providers"]["llmman"]["baseUrl"], "http://s/v1");
            assert_eq!(merged["providers"]["llmman"]["models"][0]["id"], "m");
        }
    }

    /// Nothing was there before, so llmman fills in the fields it owns.
    #[test]
    fn pi_models_merged_writes_the_placeholder_key_into_a_fresh_provider() {
        let merged = pi_models_merged(
            &serde_json::json!({}),
            "http://127.0.0.1:17434",
            &rows(vec![entry(None, "m")]),
            &HashSet::new(),
        );
        let provider = &merged["providers"]["llmman"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["api"], "openai-completions");
        // Literal, never an interpolated reference — see launch_pi.
        assert_eq!(provider["apiKey"], "llmman");
    }

    /// Launching with another list must not drop what the first one put
    /// there, and a model in both is one entry.
    #[test]
    fn pi_models_merged_keeps_a_previously_launched_model() {
        let one = pi_models_merged(
            &serde_json::json!({}),
            "http://s",
            &rows(vec![entry(None, "a")]),
            &HashSet::new(),
        );
        let two = pi_models_merged(
            &one,
            "http://s",
            &rows(vec![entry(None, "b"), entry(None, "a")]),
            &HashSet::new(),
        );
        assert_eq!(ids(&two), vec!["a", "b"]);
    }

    /// What an earlier launch wrote and no longer offers goes; the user's own
    /// rows stay, as does a row that is still offered.
    #[test]
    fn pi_models_merged_drops_rows_an_earlier_launch_wrote_and_no_longer_offers() {
        let dir = super::super::test_temp_dir("pi-stale");
        let server = "http://s";
        let user =
            serde_json::json!({ "providers": { "llmman": { "models": [{ "id": "mine" }] } } });
        std::fs::write(dir.join("models.json"), user.to_string()).unwrap();
        let first = ModelList::of(vec![entry(None, "a"), entry(Some("alpha"), "gone")]);
        write_pi_config_at(&dir, "a", server, &first).unwrap();
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(dir.join("models.json")).unwrap())
                .unwrap()
        };
        assert_eq!(ids(&read()).len(), 3);
        let second = ModelList::of(vec![entry(None, "a"), entry(None, "b")]);
        write_pi_config_at(&dir, "a", server, &second).unwrap();
        let merged = read();
        assert_eq!(ids(&merged), vec!["mine", "a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The settings that start pi on a model come after the models it is
    /// told of: a failed list leaves no setting pointing at it.
    #[test]
    fn a_failed_models_write_leaves_the_settings_alone() {
        let dir = super::super::test_temp_dir("pi-order");
        std::fs::create_dir_all(dir.join("models.json")).unwrap();
        let list = ModelList::of(vec![entry(None, "a")]);
        assert!(write_pi_config_at(&dir, "a", "http://s", &list).is_err());
        assert!(!dir.join("settings.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pi_settings_merged_leaves_unrelated_settings_alone() {
        let existing = serde_json::json!({ "theme": "dark", "defaultModel": "old" });
        let merged = pi_settings_merged(&existing, "qwen3.5:0.8b");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["defaultProvider"], "llmman");
        assert_eq!(merged["defaultModel"], "qwen3.5:0.8b");
    }

    /// Every row of the list is a model, the starting one first and named by
    /// the settings, each sent under the reference the daemon routes; two
    /// providers' copy of one id are two models, told apart by name.
    #[test]
    fn every_row_is_a_model_under_its_reference_with_a_name_that_tells_it_apart() {
        let list = ModelList::of(vec![
            entry(Some("beta"), "shared"),
            entry(None, "qwen3.5:0.8b"),
            entry(Some("alpha"), "shared"),
            entry(Some("alpha"), "solo"),
        ]);
        let rows = pi_models(&list);
        let reference = |i: usize| list.entries()[i].reference.clone();
        let merged = pi_models_merged(&serde_json::json!({}), "http://s", &rows, &HashSet::new());
        assert_eq!(
            ids(&merged),
            [reference(0), reference(1), reference(2), reference(3)]
        );
        let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            ["beta/shared", "qwen3.5:0.8b", "alpha/shared", "solo"]
        );
        let settings = pi_settings_merged(&serde_json::json!({}), &list.start().unwrap().reference);
        assert_eq!(settings["defaultModel"], "llmman.provider/beta/shared");
        assert_eq!(settings["defaultProvider"], "llmman");
    }

    #[test]
    fn a_row_declares_what_the_list_knows_of_it() {
        let mut vision = entry(Some("alpha"), "agent");
        vision.vision = true;
        vision.context_window = Some(200_000);
        vision.max_output = Some(8000);
        vision.thinking = Some(vec![
            "none".into(),
            "low".into(),
            "high".into(),
            "max".into(),
        ]);
        let plain = entry(None, "plain");
        let rows = rows(vec![vision, plain]);

        assert_eq!(rows[0]["input"], serde_json::json!(["text", "image"]));
        assert_eq!(rows[0]["contextWindow"], 200_000);
        assert_eq!(rows[0]["maxTokens"], 8000);
        assert_eq!(rows[0]["reasoning"], true);
        // Off is llmman's `none`; `medium` is hidden; `max` is opted in.
        assert_eq!(
            rows[0]["thinkingLevelMap"],
            serde_json::json!({
                "off": "none", "minimal": null, "medium": null, "max": "max"
            })
        );
        assert_eq!(rows[1]["input"], serde_json::json!(["text"]));
        assert!(rows[1].get("contextWindow").is_none());
        assert!(rows[1].get("maxTokens").is_none());
        assert!(rows[1].get("reasoning").is_none());
        assert!(rows[1].get("thinkingLevelMap").is_none());
    }

    /// A model nothing is known of takes the levels every wire accepts; a
    /// switch-only template's `thinking` is pi's `medium`.
    #[test]
    fn levels_are_the_variants_in_pis_words() {
        let unknown = ModelEntry {
            thinking: None,
            ..entry(None, "m")
        };
        assert_eq!(
            pi_model(&unknown, "m")["thinkingLevelMap"],
            serde_json::json!({ "off": "none", "minimal": null })
        );
        let switch = ModelEntry {
            thinking: Some(vec!["none".into(), "thinking".into()]),
            ..entry(None, "m")
        };
        assert_eq!(
            pi_model(&switch, "m")["thinkingLevelMap"],
            serde_json::json!({ "off": "none", "minimal": null, "low": null, "high": null })
        );
        // No way to turn it off: the level is hidden. A word pi has no level
        // for is left out, and a model left with no level does not reason.
        let always = ModelEntry {
            thinking: Some(vec!["low".into(), "adaptive".into()]),
            ..entry(None, "m")
        };
        let map = pi_model(&always, "m")["thinkingLevelMap"].clone();
        assert_eq!(map["off"], serde_json::Value::Null);
        assert_eq!(map["medium"], serde_json::Value::Null);
        assert!(map.get("low").is_none());
        let odd = ModelEntry {
            thinking: Some(vec!["none".into(), "adaptive".into()]),
            ..entry(None, "m")
        };
        assert!(pi_model(&odd, "m").get("reasoning").is_none());
    }

    /// The list is thousands of rows, and the rewrite is one pass over them.
    #[test]
    fn a_big_list_merges_in_place_and_keeps_what_was_there() {
        let existing = serde_json::json!({ "providers": { "llmman": { "models": [
            { "id": "mine", "name": "hand added" },
            { "id": "m1", "name": "old name", "extra": 1 },
            { "id": "m1", "name": "duplicate" },
        ] } } });
        let entries: Vec<ModelEntry> = (0..3000).map(|i| entry(None, &format!("m{i}"))).collect();
        let merged = pi_models_merged(&existing, "http://s", &rows(entries), &HashSet::new());
        let models = merged["providers"]["llmman"]["models"].as_array().unwrap();
        assert_eq!(models.len(), 3001);
        assert_eq!(models[0]["name"], "hand added");
        // Rebuilt where it stood, once, without the fields it had.
        assert_eq!(models[1]["id"], "m1");
        assert_eq!(models[1]["name"], "m1");
        assert!(models[1].get("extra").is_none());
        assert_eq!(models[2]["id"], "m0");
    }

    #[test]
    fn the_default_effort_is_medium_else_the_lowest() {
        assert_eq!(default_effort(&["low", "medium", "high"]), Some("medium"));
        assert_eq!(default_effort(&["high", "max"]), Some("high"));
        assert_eq!(default_effort(&[]), None);
    }
}
