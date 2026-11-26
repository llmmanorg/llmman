//! `llmman launch grok` (Grok Build).

use super::models::{ModelEntry, ModelList};
use std::path::{Path, PathBuf};

use super::common;
use super::{
    env_dir_or_home, exec_with_env, find_on_path, has_forwarded_model_flag, server, Effort,
};

/// The per-model `env_key` in llmman's Grok config reads this variable.
/// A model credential outranks both Grok's signed-in session and its global
/// `XAI_API_KEY`, without putting the actual key on disk.
const GROK_API_KEY_ENV: &str = "LLMMAN_GROK_API_KEY";

/// How many rows Grok Build is offered: it puts every configured model in the prompt of every
/// request (about 45 bytes a row; the whole catalog was 470 KB, over a Claude model's 200k window),
/// so only the front of the list is given.
const GROK_MODEL_LIMIT: usize = 300;

/// grok: point its custom-model catalog and inference client at llmman's
/// OpenAI-compatible surface. Every auxiliary model is pinned too: without
/// this, Grok Build keeps built-in hosted ids for title/summary, image
/// description, web search, and prompt suggestions, then asks the local
/// daemon to load one after the main model already answered successfully.
///
/// Grok Build merges the `[model.<id>]` entries of its config with the
/// catalog it fetches from `GROK_MODELS_LIST_URL`, so the config carries
/// every model of the list, hosted ones included, which `/v1/models` does
/// not list; each is switched to inside Grok by its `reference`. Grok
/// Build 1.0.92 lists them in `grok models` and sends the reference as is.
///
/// The model flag is injected only when the caller did not provide one
/// after `--`. This matches `qwen::qwen_args` and lets an explicit
/// integration argument win without passing a duplicate flag. `--model` is
/// not looked for after a second `--`; a model named there that the list
/// lacks is added to it.
pub(super) fn launch_grok(
    model: &str,
    api_key: &str,
    _effort: Option<&Effort>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_grok().ok_or_else(|| anyhow::anyhow!("grok is not installed"))?;
    let effective_model = common::forwarded_model(extra_args).unwrap_or(model);
    let base_url = format!("{}/v1", server());
    let models_url = format!("{base_url}/models");
    // Never edit the user's config.toml. This child is wholly llmman-owned,
    // and setting GROK_HOME below scopes it to this launched process.
    let home = grok_home()?.join("llmman");
    write_grok_config(&home, &grok_list(models, effective_model), &base_url)?;
    let home = home.to_string_lossy().into_owned();
    let args = grok_args(model, extra_args);
    exec_with_env(
        &bin,
        &args,
        &grok_env(effective_model, api_key, &base_url, &models_url, &home),
    )
}

/// What Grok Build is given: the front of `models` (see
/// [`GROK_MODEL_LIMIT`]), and `model` where the caller named one it lacks.
fn grok_list(models: &ModelList, model: &str) -> ModelList {
    let limited = models.limited(
        "grok",
        GROK_MODEL_LIMIT,
        "Grok Build sends its model list with every request",
    );
    with_model(&limited, model)
}

/// `models`, and `model` as a last row when the caller named one the list
/// does not hold.
fn with_model(models: &ModelList, model: &str) -> ModelList {
    let mut entries = models.entries().to_vec();
    if !model.is_empty() && !entries.iter().any(|e| e.reference == model) {
        entries.push(ModelEntry::unlisted(model));
    }
    ModelList::of(entries)
}

fn grok_env<'a>(
    model: &'a str,
    api_key: &'a str,
    base_url: &'a str,
    models_url: &'a str,
    home: &'a str,
) -> Vec<(&'a str, &'a str)> {
    vec![
        ("GROK_HOME", home),
        ("GROK_MODELS_BASE_URL", base_url),
        // Override an inherited custom catalog too. If it points elsewhere,
        // the selected local model is absent and Grok refuses `--model`
        // before making an inference request.
        ("GROK_MODELS_LIST_URL", models_url),
        ("GROK_DEFAULT_MODEL", model),
        ("GROK_WEB_SEARCH_MODEL", model),
        ("GROK_SESSION_SUMMARY_MODEL", model),
        ("GROK_IMAGE_DESCRIPTION_MODEL", model),
        ("GROK_PROMPT_SUGGESTIONS_MODEL", model),
        (GROK_API_KEY_ENV, api_key),
        // Grok uses this global fallback while fetching the remote catalog;
        // inference uses the higher-priority per-model env_key above.
        ("XAI_API_KEY", api_key),
    ]
}

/// Grok's configured home, or its documented `~/.grok` default.
pub(super) fn grok_home() -> anyhow::Result<PathBuf> {
    env_dir_or_home("GROK_HOME", ".grok")
}

fn write_grok_config(home: &Path, models: &ModelList, base_url: &str) -> anyhow::Result<()> {
    let path = home.join("config.toml");
    let contents = grok_config_document(models, base_url);
    common::write_atomic_in_dir(&path, contents.as_bytes())
}

/// One `[model.<id>]` per row, keyed by the `reference` Grok sends and is
/// selected by (`--model`, `GROK_DEFAULT_MODEL`).
fn grok_config_document(models: &ModelList, base_url: &str) -> String {
    let ids = models.unique_keys(|e| e.reference.clone());
    let names = models.display_names();
    let mut table = toml_edit::Table::new();
    table.set_implicit(true);
    for ((id, name), entry) in ids.iter().zip(&names).zip(models.entries()) {
        table.insert(
            id,
            toml_edit::Item::Table(grok_model(entry, name, base_url)),
        );
    }
    let mut document = toml_edit::DocumentMut::new();
    document.insert("model", toml_edit::Item::Table(table));
    document.to_string()
}

/// The one mapping of a row to a Grok model. Grok Build ignores `--effort`
/// for a model without `reasoning_efforts` listed, and takes neither the
/// image input nor the tool calling from its config.
fn grok_model(entry: &ModelEntry, name: &str, base_url: &str) -> toml_edit::Table {
    let mut model = toml_edit::Table::new();
    model["model"] = toml_edit::value(entry.reference.as_str());
    model["name"] = toml_edit::value(name);
    model["base_url"] = toml_edit::value(base_url);
    model["env_key"] = toml_edit::value(GROK_API_KEY_ENV);
    model["api_backend"] = toml_edit::value("chat_completions");
    if let Some(window) = entry.context_window {
        model["context_window"] = toml_edit::value(window as i64);
    }
    if let Some(output) = entry.max_output {
        model["max_completion_tokens"] = toml_edit::value(i64::from(output));
    }
    let efforts = entry.spelled_variants("grok");
    if !efforts.is_empty() {
        model["reasoning_efforts"] = toml_edit::value(efforts.iter().collect::<toml_edit::Array>());
    }
    model
}

fn grok_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 2);
    if !has_forwarded_model_flag("grok", extra_args) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// `PATH`, then the official installer's target, `~/.grok/bin`.
pub(super) fn find_grok() -> Option<PathBuf> {
    find_on_path("grok").or_else(|| grok_fallback(&dirs::home_dir()?))
}

fn grok_fallback(home: &Path) -> Option<PathBuf> {
    let binary = if cfg!(windows) { "grok.exe" } else { "grok" };
    let candidate = home.join(".grok").join("bin").join(binary);
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grok_gets_the_front_of_a_long_list_and_always_the_model_it_starts_on() {
        let row = |i: usize| ModelEntry::unlisted(&format!("llmman.provider/p/m{i}"));
        let long = ModelList::of((0..GROK_MODEL_LIMIT + 50).map(row).collect());
        let list = grok_list(&long, "llmman.provider/p/m0");
        assert_eq!(list.entries().len(), GROK_MODEL_LIMIT);
        assert_eq!(list.entries()[0].reference, "llmman.provider/p/m0");
        // A model named after `--` that the front lacks is added, not lost.
        let far = format!("llmman.provider/p/m{}", GROK_MODEL_LIMIT + 10);
        let list = grok_list(&long, &far);
        assert_eq!(list.entries().len(), GROK_MODEL_LIMIT + 1);
        assert_eq!(list.entries().last().unwrap().reference, far);
    }

    /// Grok Build uses the custom-model endpoint for both catalog lookup
    /// and inference. Its auxiliary samplers must follow the selected
    /// model too, rather than asking llmman for Grok's hosted defaults.
    #[test]
    fn grok_env_points_every_model_path_at_llmman() {
        let env = grok_env(
            "docker.io/ai/qwen3.5:0.8b",
            "k",
            "http://127.0.0.1:17434/v1",
            "http://127.0.0.1:17434/v1/models",
            "/tmp/grok/llmman",
        );
        let get = |key| {
            env.iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| *value)
        };
        assert_eq!(
            get("GROK_MODELS_BASE_URL"),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(get("GROK_HOME"), Some("/tmp/grok/llmman"));
        assert_eq!(
            get("GROK_MODELS_LIST_URL"),
            Some("http://127.0.0.1:17434/v1/models")
        );
        assert_eq!(get("GROK_DEFAULT_MODEL"), Some("docker.io/ai/qwen3.5:0.8b"));
        assert_eq!(
            get("GROK_WEB_SEARCH_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_SESSION_SUMMARY_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_IMAGE_DESCRIPTION_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_PROMPT_SUGGESTIONS_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(get(GROK_API_KEY_ENV), Some("k"));
        assert_eq!(get("XAI_API_KEY"), Some("k"));
    }

    /// A per-model env_key beats both an existing Grok login and the global
    /// XAI_API_KEY. The generated config contains only the environment
    /// variable's name, never the credential itself.
    #[test]
    fn grok_config_uses_the_model_credential_without_persisting_it() {
        let model = r#"org/model.\"quoted\""#;
        let list = ModelList::of(vec![ModelEntry::unlisted(model)]);
        let text = grok_config_document(&list, "http://127.0.0.1:17434/v1");

        let parsed: toml::Value = text.parse().expect("valid TOML");
        let entry = &parsed["model"][model];
        assert_eq!(entry["model"].as_str(), Some(model));
        assert_eq!(
            entry["base_url"].as_str(),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(entry["env_key"].as_str(), Some(GROK_API_KEY_ENV));
        assert_eq!(entry["api_backend"].as_str(), Some("chat_completions"));
        assert!(entry.get("api_key").is_none());
    }

    #[test]
    fn grok_config_is_written_only_inside_the_isolated_child_home() {
        let root = std::env::temp_dir().join(format!(
            "llmman-grok-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let user_config = root.join("config.toml");
        std::fs::write(&user_config, "[model.mine]\napi_key = \"keep-me\"\n").unwrap();

        let isolated = root.join("llmman");
        let list = ModelList::of(vec![ModelEntry::unlisted("m")]);
        write_grok_config(&isolated, &list, "http://127.0.0.1:17434/v1").unwrap();

        assert_eq!(
            std::fs::read_to_string(user_config).unwrap(),
            "[model.mine]\napi_key = \"keep-me\"\n"
        );
        assert!(std::fs::read_to_string(isolated.join("config.toml"))
            .unwrap()
            .contains("LLMMAN_GROK_API_KEY"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// llmman supplies Grok's model flag unless the caller explicitly
    /// supplied one after `--`; no duplicate flag is handed to the CLI.
    #[test]
    fn grok_args_add_the_model_and_yield_to_an_explicit_override() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            grok_args("m:latest", &args(&["--single", "hi"])),
            ["--model", "m:latest", "--single", "hi"]
        );
        assert_eq!(
            grok_args("m:latest", &args(&["-m", "theirs", "--single", "hi"])),
            ["-m", "theirs", "--single", "hi"]
        );
        assert_eq!(
            grok_args("m:latest", &args(&["--model=theirs"])),
            ["--model=theirs"]
        );
        // After a second `--` it is not a flag, so llmman's `--model` is kept.
        assert_eq!(
            grok_args("m:latest", &args(&["--single", "hi", "--", "--model", "x"])),
            ["--model", "m:latest", "--single", "hi", "--", "--model", "x"]
        );
    }

    /// The official installer puts Grok under `~/.grok/bin`, which is
    /// commonly invisible to a non-login process even though the CLI is
    /// installed and usable from the user's shell.
    #[test]
    fn grok_fallback_finds_the_official_installers_target() {
        let home = std::env::temp_dir().join(format!(
            "llmman-grok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bin = home.join(".grok").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        assert_eq!(grok_fallback(&home), None);
        let grok = bin.join(if cfg!(windows) { "grok.exe" } else { "grok" });
        std::fs::create_dir(&grok).unwrap();
        assert_eq!(grok_fallback(&home), None, "a directory is not a binary");
        std::fs::remove_dir(&grok).unwrap();
        std::fs::write(&grok, "").unwrap();
        assert_eq!(grok_fallback(&home), Some(grok));
        let _ = std::fs::remove_dir_all(&home);
    }

    fn hosted(provider: &str, id: &str) -> ModelEntry {
        ModelEntry::unlisted(&format!("llmman.provider/{provider}/{id}"))
    }

    fn list() -> ModelList {
        let mut local = ModelEntry::unlisted("docker.io/ai/qwen3.5:0.8b");
        local.id = "qwen3.5:0.8b".into();
        local.context_window = Some(32_768);
        local.thinking = Some(vec!["none".into(), "thinking".into()]);
        let mut agent = hosted("alpha", "agent");
        agent.context_window = Some(200_000);
        agent.max_output = Some(8_000);
        agent.thinking = Some(vec!["low".into(), "high".into()]);
        let mut plain = hosted("beta", "agent");
        plain.thinking = Some(Vec::new());
        ModelList::of(vec![local, agent, plain])
    }

    /// Every row is a model Grok can be switched to by the reference the
    /// daemon routes, the starting one first and keyed like the one
    /// `--model` and `GROK_DEFAULT_MODEL` name.
    #[test]
    fn grok_config_has_a_model_per_row_keyed_by_its_reference() {
        let text = grok_config_document(&list(), "http://h/v1");
        let parsed: toml::Value = text.parse().expect("valid TOML");
        let models = parsed["model"].as_table().unwrap();
        let keys: Vec<&str> = models.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "docker.io/ai/qwen3.5:0.8b",
                "llmman.provider/alpha/agent",
                "llmman.provider/beta/agent"
            ]
        );
        for (key, entry) in models {
            assert_eq!(entry["model"].as_str(), Some(key.as_str()));
            assert_eq!(entry["base_url"].as_str(), Some("http://h/v1"));
            assert_eq!(entry["env_key"].as_str(), Some(GROK_API_KEY_ENV));
            assert!(entry.get("api_key").is_none());
        }
        // Plain ids, with the provider only where two rows share one.
        assert_eq!(
            models["docker.io/ai/qwen3.5:0.8b"]["name"].as_str(),
            Some("qwen3.5:0.8b")
        );
        assert_eq!(
            models["llmman.provider/alpha/agent"]["name"].as_str(),
            Some("alpha/agent")
        );
        assert_eq!(
            models["llmman.provider/beta/agent"]["name"].as_str(),
            Some("beta/agent")
        );
    }

    #[test]
    fn grok_config_carries_each_rows_window_output_and_variants() {
        let text = grok_config_document(&list(), "http://h/v1");
        let parsed: toml::Value = text.parse().expect("valid TOML");
        let models = &parsed["model"];
        let agent = &models["llmman.provider/alpha/agent"];
        assert_eq!(agent["context_window"].as_integer(), Some(200_000));
        assert_eq!(agent["max_completion_tokens"].as_integer(), Some(8_000));
        assert_eq!(
            agent["reasoning_efforts"],
            toml::Value::Array(vec!["low".into(), "high".into()])
        );
        // `thinking` is Grok's `medium`; `none` is its own word for off.
        let local = &models["docker.io/ai/qwen3.5:0.8b"];
        assert_eq!(
            local["reasoning_efforts"],
            toml::Value::Array(vec!["none".into(), "medium".into()])
        );
        assert!(local.get("max_completion_tokens").is_none());
        // A model that does not think lists no level, so `--effort` is ignored.
        let plain = &models["llmman.provider/beta/agent"];
        assert!(plain.get("reasoning_efforts").is_none());
        assert!(plain.get("context_window").is_none());
    }

    #[test]
    fn a_model_named_after_the_separator_is_added_when_the_list_lacks_it() {
        let list = list();
        assert_eq!(with_model(&list, "llmman.provider/alpha/agent"), list);
        assert_eq!(with_model(&list, ""), list);
        let more = with_model(&list, "gemma4:12b");
        assert_eq!(more.entries().len(), 4);
        assert_eq!(more.entries()[3].reference, "gemma4:12b");
        assert_eq!(more.entries()[0], list.entries()[0]);
    }

    #[test]
    fn grok_config_holds_no_credential() {
        let text = grok_config_document(&list(), "http://h/v1");
        assert!(!text.contains("api_key ="), "{text}");
    }
}
