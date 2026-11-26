//! `llmman launch vibe` (Mistral Vibe CLI).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Context;
use toml_edit::{value, Array, DocumentMut, Item, Table, Value};

use super::common::{daemon_no_proxy_env, read_if_exists, write_atomic_in_dir};
use super::models::{self, ModelEntry, ModelList};
use super::{env_dir_or_home, exec_with_owned_env, find_on_path, run_with_env, server};

const VIBE_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// The provider every model llmman registers belongs to, and so the mark
/// that tells them from the user's own in `config.toml`.
const VIBE_PROVIDER: &str = "llmman";

/// vibe: the list is thousands of models, which no `VIBE_*` variable can carry (`E2BIG` on Linux,
/// 32k on Windows), and Vibe reads models from no file but `config.toml`. So the provider and every
/// model are written there and taken out when the last launch using them exits (a killed launch
/// leaves them for the next to replace). Vibe is told only the start model and the key's variable;
/// the key itself is never written. With nothing to switch to there is no file: the one model goes
/// in `VIBE_PROVIDERS` and `VIBE_MODELS`, the only route an `--sandbox openshell` Vibe has. Without
/// a model, Vibe uses its own configuration.
///
/// Vibe sends the entry's `name`, the reference, as the model name, whichever model `/model` picks.
pub(super) fn launch_vibe(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("vibe").ok_or_else(|| anyhow::anyhow!("vibe is not installed"))?;
    let server = server();
    if !models.offers_choice() {
        let mut env = vibe_single_env(model, api_key, &server, context_window);
        if !model.is_empty() {
            env.extend(daemon_no_proxy_env(&server));
        }
        return exec_with_owned_env(&bin, extra_args, &env);
    }
    let mut env = vibe_env(models, api_key);
    env.extend(daemon_no_proxy_env(&server));
    let config = vibe_config_path()?;
    // Held until this launch has taken its rows out: another launch's rows
    // are the same ones, and are not taken from under it.
    let claim = models::SharedUse::claim(&config.with_file_name(VIBE_LOCK))?;
    edit_vibe_config(&config, |text| {
        merge_vibe_models(text, models, &format!("{server}/v1"))
    })?;
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let code = run_with_env(&bin, extra_args, &env);
    if let Some(_last) = claim.release_if_last() {
        if let Err(error) = edit_vibe_config(&config, strip_vibe_models) {
            eprintln!(
                "[llmman] vibe: could not take its models out of {}: {error:#}",
                config.display()
            );
        }
    }
    super::hdrm::stop();
    models::finish();
    std::process::exit(code?);
}

/// The file the launches that use `config.toml`'s rows hold a lock on, beside it.
const VIBE_LOCK: &str = ".llmman-launch.lock";

/// What a Vibe with one model is handed: the model, its provider and the
/// key's variable, all in the environment, so no file is read.
fn vibe_single_env(
    model: &str,
    api_key: &str,
    server: &str,
    context_window: Option<u64>,
) -> Vec<(&'static str, String)> {
    if model.is_empty() {
        return Vec::new();
    }
    let mut preset = serde_json::json!({
        "name": model,
        "provider": VIBE_PROVIDER,
        "alias": VIBE_PROVIDER,
    });
    if let Some(window) = context_window {
        preset["max_context_length"] = window.into();
    }
    let provider = serde_json::json!({
        "name": VIBE_PROVIDER,
        "api_base": format!("{server}/v1"),
        "api_key_env_var": VIBE_API_KEY_ENV,
        "api_style": "openai",
        "backend": "generic",
    });
    vec![
        ("VIBE_ACTIVE_MODEL", VIBE_PROVIDER.to_string()),
        ("VIBE_PROVIDERS", serde_json::json!([provider]).to_string()),
        ("VIBE_MODELS", serde_json::json!([preset]).to_string()),
        (VIBE_API_KEY_ENV, api_key.to_string()),
    ]
}

/// What Vibe is handed in its environment: the model it starts on, the one
/// its `config.toml` names the first row's.
fn vibe_env(models: &ModelList, api_key: &str) -> Vec<(&'static str, String)> {
    let Some(start) = vibe_aliases(models).into_iter().next() else {
        return Vec::new();
    };
    vec![
        ("VIBE_ACTIVE_MODEL", start),
        (VIBE_API_KEY_ENV, api_key.to_string()),
    ]
}

/// The alias of each row: what Vibe keys it by, and `VIBE_ACTIVE_MODEL`
/// names. Told apart where two providers offer one model.
fn vibe_aliases(models: &ModelList) -> Vec<String> {
    models.unique_keys(ModelEntry::label)
}

/// Vibe's own resolution: `$VIBE_HOME`, else `~/.vibe`.
fn vibe_config_path() -> anyhow::Result<PathBuf> {
    Ok(env_dir_or_home("VIBE_HOME", ".vibe")?.join("config.toml"))
}

/// Rewrites `path` through `edit`, unless that changes nothing; a file left
/// with nothing in it is removed.
fn edit_vibe_config(
    path: &Path,
    edit: impl FnOnce(&str) -> anyhow::Result<String>,
) -> anyhow::Result<()> {
    let before = read_if_exists(path)?.unwrap_or_default();
    let after = edit(&before).with_context(|| format!("update {}", path.display()))?;
    if after == before {
        return Ok(());
    }
    if after.trim().is_empty() {
        return std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()));
    }
    write_atomic_in_dir(path, after.as_bytes())
}

/// `text` with the provider and every model of `models` in place of those
/// llmman put there before, and nothing else touched.
fn merge_vibe_models(text: &str, models: &ModelList, base_url: &str) -> anyhow::Result<String> {
    let mut document: DocumentMut = text.parse().context("parse as TOML")?;
    remove_ours(&mut document)?;

    let mut provider = Table::new();
    provider["name"] = value(VIBE_PROVIDER);
    provider["api_base"] = value(base_url);
    provider["api_key_env_var"] = value(VIBE_API_KEY_ENV);
    provider["api_style"] = value("openai");
    provider["backend"] = value("generic");
    push_entry(&mut document, "providers", provider)?;

    let name = models.display_namer();
    for (entry, alias) in models.entries().iter().zip(vibe_aliases(models)) {
        push_entry(
            &mut document,
            "models",
            vibe_model(entry, &alias, &name(entry)),
        )?;
    }
    Ok(document.to_string())
}

/// One `[[models]]` row.
fn vibe_model(entry: &ModelEntry, alias: &str, name: &str) -> Table {
    let mut model = Table::new();
    model["name"] = value(entry.reference.as_str());
    model["provider"] = value(VIBE_PROVIDER);
    model["alias"] = value(alias);
    model["display_name"] = value(name);
    if let Some(window) = entry.context_window {
        model["max_context_length"] = value(i64::try_from(window).unwrap_or(i64::MAX));
    }
    if let Some(max_output) = entry.max_output {
        model["max_output_tokens"] = value(i64::from(max_output));
    }
    model["supports_images"] = value(entry.vision);
    let levels: Array = vibe_levels(entry).into_iter().collect();
    model["thinking_levels"] = value(levels);
    model
}

/// The model's variants in Vibe's words, in cycle order: `off` for none, and a
/// switch-only template's on as `medium`, which llmman serves as on. Vibe
/// sends every other level as written.
fn vibe_levels(entry: &ModelEntry) -> Vec<String> {
    let mut levels: Vec<String> = Vec::new();
    for variant in entry.variants() {
        let level = match variant.as_str() {
            "none" => "off",
            "thinking" => "medium",
            other => other,
        };
        if !levels.iter().any(|l| l == level) {
            levels.push(level.to_string());
        }
    }
    levels
}

/// `text` without the provider and models llmman wrote, and without an
/// `active_model` Vibe saved as one of them (it would name a model that is
/// gone).
fn strip_vibe_models(text: &str) -> anyhow::Result<String> {
    let mut document: DocumentMut = text.parse().context("parse as TOML")?;
    remove_ours(&mut document)?;
    Ok(document.to_string())
}

fn remove_ours(document: &mut DocumentMut) -> anyhow::Result<()> {
    let mut aliases: HashSet<String> = HashSet::new();
    retain_rows(document, "models", |row| {
        let ours = row.get("provider").and_then(Item::as_str) == Some(VIBE_PROVIDER);
        if ours {
            aliases.extend(row.get("alias").and_then(Item::as_str).map(str::to_string));
        }
        !ours
    })?;
    retain_rows(document, "providers", |row| {
        row.get("name").and_then(Item::as_str) != Some(VIBE_PROVIDER)
    })?;
    let saved = document
        .get("active_model")
        .and_then(Item::as_str)
        .is_some_and(|name| aliases.contains(name));
    if saved {
        document.remove("active_model");
    }
    Ok(())
}

/// Keeps the rows of the list `key` that `keep` says to, and drops the key if
/// none are left.
fn retain_rows(
    document: &mut DocumentMut,
    key: &str,
    mut keep: impl FnMut(&dyn toml_edit::TableLike) -> bool,
) -> anyhow::Result<()> {
    let Some(item) = document.get_mut(key) else {
        return Ok(());
    };
    let remaining = match item {
        Item::ArrayOfTables(rows) => {
            rows.retain(|row| keep(row));
            rows.len()
        }
        Item::Value(Value::Array(rows)) => {
            rows.retain(|row| row.as_inline_table().is_none_or(|t| keep(t)));
            rows.len()
        }
        _ => anyhow::bail!("`{key}` is not a list; llmman needs the `[[{key}]]` form"),
    };
    if remaining == 0 {
        document.remove(key);
    }
    Ok(())
}

/// Appends `row` to the list `key`, which is created when there is none.
fn push_entry(document: &mut DocumentMut, key: &str, row: Table) -> anyhow::Result<()> {
    let list = document
        .entry(key)
        .or_insert_with(|| Item::ArrayOfTables(Default::default()));
    match list {
        Item::ArrayOfTables(rows) => rows.push(row),
        Item::Value(Value::Array(rows)) => rows.push(row.into_inline_table()),
        _ => anyhow::bail!("`{key}` is not a list; llmman needs the `[[{key}]]` form"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosted(provider: &str, id: &str) -> ModelEntry {
        ModelEntry::unlisted(&crate::providers::format_remote_ref(provider, id))
    }

    fn get<'a>(env: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    fn merged(text: &str, entries: Vec<ModelEntry>) -> (String, toml::Value) {
        let out =
            merge_vibe_models(text, &ModelList::of(entries), "http://127.0.0.1:11434/v1").unwrap();
        let parsed = out.parse::<toml::Value>().expect("valid TOML");
        (out, parsed)
    }

    fn rows<'a>(doc: &'a toml::Value, key: &str) -> &'a [toml::Value] {
        doc.get(key)
            .and_then(toml::Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    #[test]
    fn vibe_env_names_the_start_and_the_key_variable_and_holds_no_list() {
        let list = ModelList::of(vec![hosted("alpha", "m"), hosted("beta", "m")]);
        let env = vibe_env(&list, "k");
        assert_eq!(get(&env, "VIBE_ACTIVE_MODEL"), Some("alpha/m"));
        assert_eq!(get(&env, "LLMMAN_API_KEY"), Some("k"));
        assert!(get(&env, "VIBE_MODELS").is_none());
        assert!(get(&env, "VIBE_PROVIDERS").is_none());
    }

    #[test]
    fn vibe_env_without_a_model_leaves_vibes_own_configuration() {
        assert!(vibe_env(&ModelList::default(), "k").is_empty());
        assert!(vibe_single_env("", "k", "http://h", Some(1)).is_empty());
    }

    /// One model, or an `--sandbox openshell` Vibe that sees no file: the
    /// environment alone carries it.
    #[test]
    fn a_single_model_is_defined_in_the_environment_with_no_file() {
        let env = vibe_single_env("qwen3.5:0.8b", "k", "http://127.0.0.1:11434", Some(32_768));
        assert_eq!(get(&env, "VIBE_ACTIVE_MODEL"), Some("llmman"));
        assert_eq!(get(&env, "LLMMAN_API_KEY"), Some("k"));
        let providers: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_PROVIDERS").unwrap()).unwrap();
        assert_eq!(providers[0]["api_base"], "http://127.0.0.1:11434/v1");
        assert_eq!(providers[0]["api_key_env_var"], "LLMMAN_API_KEY");
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert_eq!(models[0]["name"], "qwen3.5:0.8b");
        assert_eq!(models[0]["provider"], providers[0]["name"]);
        assert_eq!(models[0]["alias"], "llmman");
        assert_eq!(models[0]["max_context_length"], 32_768);
        let env = vibe_single_env("m", "k", "http://h", None);
        let models: serde_json::Value =
            serde_json::from_str(get(&env, "VIBE_MODELS").unwrap()).unwrap();
        assert!(models[0].get("max_context_length").is_none());
    }

    #[test]
    fn vibe_config_defines_the_provider_at_llmmans_endpoint() {
        let (_, doc) = merged("", vec![hosted("alpha", "m")]);
        let providers = rows(&doc, "providers");
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0]["name"].as_str(), Some("llmman"));
        assert_eq!(
            providers[0]["api_base"].as_str(),
            Some("http://127.0.0.1:11434/v1")
        );
        assert_eq!(
            providers[0]["api_key_env_var"].as_str(),
            Some("LLMMAN_API_KEY")
        );
        assert_eq!(providers[0]["api_style"].as_str(), Some("openai"));
        assert_eq!(providers[0]["backend"].as_str(), Some("generic"));
    }

    /// Every row is a model Vibe can switch to, the starting one first and
    /// the one `VIBE_ACTIVE_MODEL` names; Vibe sends each one's `name`.
    #[test]
    fn vibe_config_holds_every_model_with_the_start_first() {
        let entries = vec![
            ModelEntry::unlisted("qwen3.5:0.8b"),
            hosted("alpha", "a"),
            hosted("beta", "b"),
        ];
        let (_, doc) = merged("", entries.clone());
        let models = rows(&doc, "models");
        assert_eq!(models.len(), 3);
        for (row, entry) in models.iter().zip(&entries) {
            assert_eq!(row["name"].as_str(), Some(entry.reference.as_str()));
            assert_eq!(row["provider"].as_str(), Some("llmman"));
        }
        let env = vibe_env(&ModelList::of(entries), "k");
        assert_eq!(get(&env, "VIBE_ACTIVE_MODEL"), models[0]["alias"].as_str());
    }

    /// Two providers can offer one id: Vibe keys a model by its alias, and the
    /// picker shows the display name, so both must tell them apart.
    #[test]
    fn vibe_config_keeps_models_of_the_same_id_from_different_providers() {
        let entries = vec![
            hosted("alpha", "gpt"),
            hosted("beta", "gpt"),
            hosted("alpha", "other"),
        ];
        let (text, doc) = merged("", entries);
        let models = rows(&doc, "models");
        let aliases: HashSet<&str> = models
            .iter()
            .map(|m| m["alias"].as_str().unwrap())
            .collect();
        assert_eq!(aliases.len(), 3, "{text}");
        let names: Vec<&str> = models
            .iter()
            .map(|m| m["display_name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["alpha/gpt", "beta/gpt", "other"]);
    }

    #[test]
    fn vibe_config_carries_window_output_vision_and_levels() {
        let rich = ModelEntry {
            context_window: Some(200_000),
            max_output: Some(8000),
            thinking: Some(vec!["none".into(), "low".into(), "high".into()]),
            vision: true,
            ..hosted("alpha", "rich")
        };
        let plain = ModelEntry {
            thinking: Some(Vec::new()),
            ..hosted("alpha", "plain")
        };
        let switch = ModelEntry {
            thinking: Some(vec!["none".into(), "thinking".into()]),
            ..hosted("alpha", "switch")
        };
        let unknown = hosted("alpha", "unknown");
        let (_, doc) = merged("", vec![rich, plain, switch, unknown]);
        let models = rows(&doc, "models");
        let levels = |i: usize| -> Vec<&str> {
            models[i]["thinking_levels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap())
                .collect()
        };
        assert_eq!(models[0]["max_context_length"].as_integer(), Some(200_000));
        assert_eq!(models[0]["max_output_tokens"].as_integer(), Some(8000));
        assert_eq!(models[0]["supports_images"].as_bool(), Some(true));
        assert_eq!(levels(0), ["off", "low", "high"]);
        assert_eq!(models[1]["supports_images"].as_bool(), Some(false));
        assert!(models[1].get("max_context_length").is_none());
        assert!(models[1].get("max_output_tokens").is_none());
        assert!(levels(1).is_empty());
        assert_eq!(levels(2), ["off", "medium"]);
        // Nothing said: the levels every wire accepts.
        assert_eq!(levels(3), ["off", "low", "medium", "high"]);
    }

    #[test]
    fn vibe_config_holds_no_secret() {
        let (text, _) = merged("", vec![hosted("alpha", "m"), ModelEntry::unlisted("q")]);
        assert!(text.contains("api_key_env_var = \"LLMMAN_API_KEY\""));
        assert!(!text.contains("api_key ="), "{text}");
        assert!(!text.contains("sk-"), "{text}");
    }

    const USER: &str = r#"# mine
theme = "auto"
active_model = "mine"

[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"

[[models]]
name = "mine-large"
provider = "mistral"
alias = "mine"
"#;

    /// The user's file is kept as it is, and what was put in it goes out
    /// again exactly.
    #[test]
    fn vibe_config_merges_beside_the_users_and_strips_back_to_it() {
        let (text, doc) = merged(USER, vec![hosted("alpha", "m")]);
        assert!(text.starts_with("# mine\ntheme = \"auto\""), "{text}");
        assert_eq!(doc["active_model"].as_str(), Some("mine"));
        assert_eq!(rows(&doc, "providers").len(), 2);
        assert_eq!(rows(&doc, "models").len(), 2);
        assert_eq!(rows(&doc, "models")[0]["alias"].as_str(), Some("mine"));
        assert_eq!(strip_vibe_models(&text).unwrap(), USER);
    }

    /// A second launch replaces the first one's models, not adds to them.
    #[test]
    fn vibe_config_replaces_what_an_earlier_launch_wrote() {
        let (first, _) = merged(USER, vec![hosted("alpha", "old"), hosted("alpha", "older")]);
        let (second, doc) = merged(&first, vec![hosted("beta", "new")]);
        let names: Vec<&str> = rows(&doc, "models")
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["mine-large", "llmman.provider/beta/new"]);
        assert_eq!(rows(&doc, "providers").len(), 2);
        let (again, _) = merged(&second, vec![hosted("beta", "new")]);
        assert_eq!(again, second);
    }

    /// Vibe saves the model picked in `/model` as `active_model`; once the
    /// models are gone it would name nothing.
    #[test]
    fn vibe_strip_drops_a_saved_choice_of_one_of_its_models_only() {
        let (text, _) = merged(USER, vec![hosted("alpha", "m")]);
        let saved = text.replace("active_model = \"mine\"", "active_model = \"alpha/m\"");
        let stripped = strip_vibe_models(&saved).unwrap();
        assert_eq!(stripped, USER.replace("active_model = \"mine\"\n", ""));
        // The user's own choice stays.
        assert!(strip_vibe_models(&text)
            .unwrap()
            .contains("active_model = \"mine\""));
    }

    #[test]
    fn vibe_config_takes_the_inline_list_form_and_refuses_the_table_form() {
        let inline = "models = [ { name = \"x\", provider = \"p\", alias = \"x\" } ]\n";
        let (text, doc) = merged(inline, vec![hosted("alpha", "m")]);
        assert_eq!(rows(&doc, "models").len(), 2, "{text}");
        assert_eq!(strip_vibe_models(&text).unwrap(), inline);

        let table = "[models.x]\nname = \"x\"\nprovider = \"p\"\n";
        let error = merge_vibe_models(
            table,
            &ModelList::of(vec![hosted("alpha", "m")]),
            "http://h/v1",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("`models` is not a list"), "{error}");
        assert!(merge_vibe_models("= broken", &ModelList::default(), "http://h/v1").is_err());
    }

    /// Written where Vibe reads it, created if need be, left alone when there
    /// is nothing to change, and gone again when nothing else is in it.
    #[test]
    fn vibe_config_file_is_created_updated_and_removed() {
        let dir = super::super::test_temp_dir("vibe-config");
        let path = dir.join("home").join("config.toml");
        let list = ModelList::of(vec![hosted("alpha", "m")]);
        let put = |text: &str| merge_vibe_models(text, &list, "http://h/v1");

        edit_vibe_config(&path, put).unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("llmman.provider/alpha/m"));
        edit_vibe_config(&path, put).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), written);

        edit_vibe_config(&path, strip_vibe_models).unwrap();
        assert!(!path.exists());
        // Stripping what is not there is not an error.
        edit_vibe_config(&path, strip_vibe_models).unwrap();

        std::fs::write(&path, USER).unwrap();
        edit_vibe_config(&path, put).unwrap();
        edit_vibe_config(&path, strip_vibe_models).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), USER);
        let _ = std::fs::remove_dir_all(dir);
    }
}
