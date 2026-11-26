//! `llmman launch omp` (OMP coding agent).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::providers;

use super::common;
use super::models::{remember_rows, rows_written, ModelEntry, ModelList};
use super::pi::{pi_level, provider_with_models, PI_LEVELS};
use super::{exec_with_env, find_on_path, has_forwarded_model_flag, server, WINDOWS_PATH_EXTS};

/// omp: register llmman's endpoint and every model of `models` in
/// `models.yml`.
///
/// OMP's built-in Ollama discovery sees llmman's `/api/tags`, but model
/// selection is resolved against its on-disk catalog before discovery has
/// populated it, and `/api/tags` lists no provider model at all. A fresh
/// HOME therefore rejects `--model ollama/<model>`. Writing the provider
/// and its models explicitly, as Ollama's own launcher does, makes the first
/// launch work for any of them, and switching inside OMP sends the row's
/// `id`, which is the `reference` the daemon routes. Unrelated user
/// configuration is preserved.
pub(super) fn launch_omp(
    model: &str,
    _reasons: bool,
    _vision: bool,
    _context_window: Option<u64>,
    models: &ModelList,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_omp().ok_or_else(|| anyhow::anyhow!("omp is not installed"))?;
    let server = server();
    write_omp_config(models, &server)?;
    exec_with_env(
        &bin,
        &omp_args(model, extra_args),
        &[("OLLAMA_BASE_URL", server.as_str())],
    )
}

const OMP_PROVIDER: &str = "ollama";
const OMP_SETUP_VERSION: u64 = 2;

/// `PATH`, then Bun's and the standalone installer's usual user-local bins.
pub(super) fn find_omp() -> Option<PathBuf> {
    find_on_path("omp").or_else(|| omp_fallback_paths().into_iter().find(|path| path.is_file()))
}

fn omp_fallback_paths() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let dirs = [
        home.join(".bun").join("bin"),
        home.join(".local").join("bin"),
    ];
    if cfg!(windows) {
        return dirs
            .into_iter()
            .flat_map(|dir| {
                WINDOWS_PATH_EXTS
                    .iter()
                    .map(move |ext| dir.join(format!("omp.{ext}")))
            })
            .collect();
    }
    dirs.into_iter().map(|dir| dir.join("omp")).collect()
}

/// OMP's agent directory. These are the paths OMP exposes for callers to
/// configure; profile layout remains OMP's responsibility.
pub(super) fn omp_agent_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = common::configured_pi_agent_dir()? {
        return Ok(dir);
    }
    let home = crate::config::home_dir().context("no home directory")?;
    let config = std::env::var("PI_CONFIG_DIR")
        .ok()
        .filter(|dir| !dir.trim().is_empty())
        .unwrap_or_else(|| ".omp".to_string());
    let config = PathBuf::from(config.trim());
    Ok(if config.is_absolute() {
        config.join("agent")
    } else {
        home.join(config).join("agent")
    })
}

fn write_omp_config(models: &ModelList, server: &str) -> anyhow::Result<()> {
    let dir = omp_agent_dir()?;
    write_omp_config_in_dir(&dir, models, server)
}

fn write_omp_config_in_dir(dir: &Path, models: &ModelList, server: &str) -> anyhow::Result<()> {
    write_omp_models_config_at(&dir.join("models.yml"), models, server)?;
    common::write_yaml_merged(&dir.join("config.yml"), "omp", omp_config_merged)
}

fn omp_config_merged(existing: &serde_json::Value) -> serde_json::Value {
    let mut config = existing.clone();
    if let Some(config) = config.as_object_mut() {
        let setup_version = config
            .get("setupVersion")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default()
            .max(OMP_SETUP_VERSION);
        config.insert("setupVersion".into(), serde_json::json!(setup_version));
    }
    config
}

fn write_omp_models_config_at(path: &Path, models: &ModelList, server: &str) -> anyhow::Result<()> {
    let rows = omp_models(models);
    let dir = path.parent().context("models.yml has no directory")?;
    let stale = rows_written(dir, "omp");
    common::write_yaml_merged(path, "omp", |existing| {
        omp_models_merged(existing, server, &rows, &stale)
    })?;
    remember_rows(dir, "omp", &rows)
}

/// Every row of `models` as an OMP model.
fn omp_models(models: &ModelList) -> Vec<serde_json::Value> {
    models.map_named(omp_model)
}

/// One model, in the fields `models.yml` takes. `reasoning` is always stated:
/// left out, OMP assumes a thinking ladder for a model it cannot place.
fn omp_model(entry: &ModelEntry, name: &str) -> serde_json::Value {
    let variants = entry.variants();
    let mut efforts: Vec<&str> = variants
        .iter()
        .filter_map(|v| pi_level("omp", v))
        .filter(|level| *level != "off")
        .collect();
    efforts.sort_by_key(|level| PI_LEVELS.iter().position(|l| l == level));
    efforts.dedup();
    let reasons = !efforts.is_empty();
    let mut row = common::pi_model_entry(
        &entry.reference,
        reasons,
        entry.vision,
        entry.context_window,
    );
    row["name"] = serde_json::json!(name);
    row["reasoning"] = serde_json::json!(reasons);
    if let Some(max) = entry.max_output {
        row["maxTokens"] = serde_json::json!(max);
    }
    if let Some(tools) = entry.tools {
        row["supportsTools"] = serde_json::json!(tools);
    }
    if reasons {
        row["thinking"] = serde_json::json!({ "mode": "effort", "efforts": efforts });
        // Thinking that cannot be turned off is what OMP calls mandatory.
        if !variants.iter().any(|v| v == "none") {
            row["thinking"]["requiresEffort"] = serde_json::json!(true);
        }
    }
    row
}

/// Updates only llmman's `ollama` provider and its models. Other
/// providers, provider-specific options, and the models the user added are
/// retained, and those an earlier launch wrote and this one does not offer
/// (`stale`) are dropped; endpoint and protocol fields are owned by this launcher because
/// stale values would route the launch somewhere other than llmman.
fn omp_models_merged(
    existing: &serde_json::Value,
    server: &str,
    rows: &[serde_json::Value],
    stale: &HashSet<String>,
) -> serde_json::Value {
    provider_with_models(existing, OMP_PROVIDER, rows, true, stale, |provider| {
        provider.insert(
            "baseUrl".into(),
            serde_json::json!(format!("{}/v1", server.trim_end_matches('/'))),
        );
        provider.insert("api".into(), serde_json::json!("openai-responses"));
        provider.remove("auth");
        provider.insert(
            "apiKey".into(),
            serde_json::json!(providers::PLACEHOLDER_API_KEY),
        );
        provider.insert("authHeader".into(), serde_json::json!(true));
        provider.insert("discovery".into(), serde_json::json!({ "type": "ollama" }));
    })
}

/// Select llmman's model through OMP's Ollama provider unless the caller
/// explicitly supplied an OMP model after `--`. Avoiding a duplicate matters
/// both for predictable precedence and for keeping OMP's argument parser out
/// of version-specific repeated-option behavior.
fn omp_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 2);
    if !has_forwarded_model_flag("omp", extra_args) {
        args.extend(["--model".to_string(), format!("ollama/{model}")]);
    }
    args.extend_from_slice(extra_args);
    args
}

#[cfg(test)]
mod tests {
    use super::super::test_temp_dir;
    use super::super::{INTEGRATIONS, MODEL_FLAG_FORWARDED};
    use super::*;

    fn entry(provider: Option<&str>, id: &str) -> ModelEntry {
        let reference = match provider {
            Some(p) => providers::format_remote_ref(p, id),
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

    fn parsed(path: &Path) -> serde_json::Value {
        yaml_serde::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn omp_is_listed_and_selects_the_model_through_ollama() {
        let omp = INTEGRATIONS.iter().find(|i| i.name == "omp").unwrap();
        assert_eq!(omp.binary, "omp");
        assert!(MODEL_FLAG_FORWARDED.contains(&"omp"));

        assert_eq!(
            omp_args("docker.io/ai/qwen3.5:0.8b", &[]),
            vec!["--model", "ollama/docker.io/ai/qwen3.5:0.8b"]
        );
        // A provider model is selected by the reference the daemon routes.
        assert_eq!(
            omp_args("llmman.provider/openai/gpt-5", &[]),
            vec!["--model", "ollama/llmman.provider/openai/gpt-5"]
        );

        let existing = serde_json::json!({
            "providers": {
                "other": { "baseUrl": "https://example.com" },
                "ollama": {
                    "auth": "none",
                    "headers": { "X-Custom": "kept" },
                    "models": [
                        {
                            "id": "docker.io/ai/qwen3.5:0.8b",
                            "name": "User name",
                            "input": ["text", "image"],
                            "maxTokens": 4096
                        },
                        { "id": "old", "name": "Old" }
                    ]
                }
            }
        });
        let rows = omp_models(&one(entry(None, "docker.io/ai/qwen3.5:0.8b")));
        let merged = omp_models_merged(&existing, "http://127.0.0.1:17434", &rows, &HashSet::new());
        let provider = &merged["providers"]["ollama"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["api"], "openai-responses");
        assert!(provider.get("auth").is_none());
        assert_eq!(provider["apiKey"], providers::PLACEHOLDER_API_KEY);
        assert_eq!(provider["authHeader"], true);
        assert_eq!(provider["discovery"]["type"], "ollama");
        assert_eq!(provider["headers"]["X-Custom"], "kept");
        assert_eq!(provider["models"][0]["name"], rows[0]["name"]);
        assert_eq!(provider["models"][0]["input"], rows[0]["input"]);
        assert_eq!(provider["models"][0]["maxTokens"], 4096);
        assert_eq!(provider["models"][1]["id"], "old");
        assert_eq!(
            merged["providers"]["other"]["baseUrl"],
            "https://example.com"
        );

        let config = omp_config_merged(&serde_json::json!({
            "setupVersion": OMP_SETUP_VERSION + 3,
            "theme": "dark"
        }));
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION + 3);
        assert_eq!(config["theme"], "dark");
    }

    /// Every row is a model under its reference, the starting one first; two
    /// providers' copy of one id are told apart by name, not by reference.
    #[test]
    fn every_row_is_a_model_under_its_reference() {
        let list = ModelList::of(vec![
            entry(Some("beta"), "shared"),
            entry(None, "qwen3.5:0.8b"),
            entry(Some("alpha"), "shared"),
            entry(Some("alpha"), "solo"),
        ]);
        let merged = omp_models_merged(
            &serde_json::json!({}),
            "http://s",
            &omp_models(&list),
            &HashSet::new(),
        );
        let models = merged["providers"]["ollama"]["models"].as_array().unwrap();
        let ids: Vec<&str> = models.iter().map(|m| m["id"].as_str().unwrap()).collect();
        let refs: Vec<&str> = list
            .entries()
            .iter()
            .map(|e| e.reference.as_str())
            .collect();
        assert_eq!(ids, refs);
        assert_eq!(ids[0], "llmman.provider/beta/shared");
        let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            ["beta/shared", "qwen3.5:0.8b", "alpha/shared", "solo"]
        );
    }

    #[test]
    fn a_row_declares_what_the_list_knows_of_it() {
        let mut agent = entry(Some("alpha"), "agent");
        agent.vision = true;
        agent.tools = Some(true);
        agent.context_window = Some(200_000);
        agent.max_output = Some(8000);
        agent.thinking = Some(vec![
            "none".into(),
            "high".into(),
            "low".into(),
            "max".into(),
        ]);
        let row = omp_model(&agent, "agent");
        assert_eq!(row["input"], serde_json::json!(["text", "image"]));
        assert_eq!(row["contextWindow"], 200_000);
        assert_eq!(row["maxTokens"], 8000);
        assert_eq!(row["supportsTools"], true);
        assert_eq!(row["reasoning"], true);
        // Lowest first, thinking off being OMP's own `off`.
        assert_eq!(
            row["thinking"],
            serde_json::json!({ "mode": "effort", "efforts": ["low", "high", "max"] })
        );

        let plain = omp_model(&entry(None, "plain"), "plain");
        assert_eq!(plain["reasoning"], false);
        assert!(plain.get("thinking").is_none());
        assert!(plain.get("maxTokens").is_none());
        assert!(plain.get("supportsTools").is_none());
        assert_eq!(plain["input"], serde_json::json!(["text"]));
    }

    #[test]
    fn levels_are_the_variants_in_omps_words() {
        let with = |thinking: Option<Vec<&str>>| {
            omp_model(
                &ModelEntry {
                    thinking: thinking.map(|t| t.into_iter().map(String::from).collect()),
                    ..entry(None, "m")
                },
                "m",
            )
        };
        // Nothing said: what every wire accepts.
        assert_eq!(
            with(None)["thinking"]["efforts"],
            serde_json::json!(["low", "medium", "high"])
        );
        assert!(with(None)["thinking"].get("requiresEffort").is_none());
        // A switch-only template's `thinking` is `medium`.
        assert_eq!(
            with(Some(vec!["none", "thinking"]))["thinking"]["efforts"],
            serde_json::json!(["medium"])
        );
        // No off among them: OMP is told it cannot be turned off.
        let always = with(Some(vec!["medium", "high"]));
        assert_eq!(always["thinking"]["requiresEffort"], true);
        // A word OMP has no level for does not make the model reason.
        assert_eq!(with(Some(vec!["none", "adaptive"]))["reasoning"], false);
    }

    #[test]
    fn omp_model_argument_after_separator_wins_without_a_duplicate() {
        let strings = |values: &[&str]| values.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        for forwarded in [
            strings(&["--model", "openrouter/anthropic/claude-sonnet-4"]),
            strings(&["-m", "ollama/other"]),
            strings(&["--model=ollama/other"]),
            strings(&["-m=ollama/other"]),
        ] {
            assert_eq!(omp_args("local", &forwarded), forwarded);
        }

        assert_eq!(
            omp_args("local", &strings(&["-p", "ping"])),
            strings(&["--model", "ollama/local", "-p", "ping"])
        );
        // After a second `--` it is not a flag, so llmman's model stays.
        assert_eq!(
            omp_args("local", &strings(&["--", "--model", "x"])),
            strings(&["--model", "ollama/local", "--", "--model", "x"])
        );
    }

    #[test]
    fn omp_fallback_paths_cover_bun_and_local_bin() {
        let paths = omp_fallback_paths();
        let binary = if cfg!(windows) { "omp.exe" } else { "omp" };
        assert!(paths
            .iter()
            .any(|path| path.ends_with(Path::new(".bun").join("bin").join(binary))));
        assert!(paths
            .iter()
            .any(|path| path.ends_with(Path::new(".local").join("bin").join(binary))));
    }

    #[test]
    fn omp_config_is_created_for_a_fresh_home_and_round_trips() {
        let dir = test_temp_dir("omp-models");
        let models_path = dir.join("models.yml");
        let mut qwen = entry(None, "docker.io/ai/qwen3.5:0.8b");
        qwen.vision = true;
        qwen.context_window = Some(32_768);
        qwen.thinking = Some(vec!["none".into(), "thinking".into()]);
        let list = ModelList::of(vec![qwen, entry(Some("alpha"), "agent")]);
        write_omp_config_in_dir(&dir, &list, "http://127.0.0.1:17434").unwrap();

        let provider = &parsed(&models_path)["providers"]["ollama"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["models"][0]["id"], "docker.io/ai/qwen3.5:0.8b");
        assert_eq!(provider["models"][0]["name"], "docker.io/ai/qwen3.5:0.8b");
        assert_eq!(provider["models"][0]["reasoning"], true);
        assert_eq!(
            provider["models"][0]["input"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(provider["models"][0]["contextWindow"], 32_768);
        assert_eq!(provider["models"][1]["id"], "llmman.provider/alpha/agent");
        assert_eq!(provider["models"][1]["name"], "agent");
        // Only the placeholder is ever written.
        let text = std::fs::read_to_string(&models_path).unwrap();
        assert!(text.contains(providers::PLACEHOLDER_API_KEY));
        assert_eq!(text.matches("apiKey").count(), 1, "{text}");
        let config = parsed(&dir.join("config.yml"));
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION);

        std::fs::remove_dir_all(dir).unwrap();
    }

    /// What an earlier launch wrote and no longer offers goes; a row the user
    /// added stays.
    #[test]
    fn omp_drops_rows_an_earlier_launch_wrote_and_no_longer_offers() {
        let dir = test_temp_dir("omp-stale");
        let path = dir.join("models.yml");
        std::fs::write(
            &path,
            "providers:\n  ollama:\n    models:\n      - id: mine\n",
        )
        .unwrap();
        let server = "http://127.0.0.1:17434";
        let first = ModelList::of(vec![entry(None, "docker.io/ai/a"), entry(None, "gone")]);
        write_omp_config_in_dir(&dir, &first, server).unwrap();
        let second = ModelList::of(vec![entry(None, "docker.io/ai/a")]);
        write_omp_config_in_dir(&dir, &second, server).unwrap();
        let ids: Vec<String> = parsed(&path)["providers"]["ollama"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, ["mine", "docker.io/ai/a"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn omp_models_yml_backs_up_the_exact_commented_file_before_rewriting() {
        let dir = test_temp_dir("omp-models-backup");
        let path = dir.join("models.yml");
        let config_path = dir.join("config.yml");
        let original = r#"# my own notes, do not delete
providers:
  openai:
    apiKey: sk-mine
defaults:
  temperature: 0.2   # tuned by hand
"#;
        std::fs::write(&path, original).unwrap();
        let config_original = "# keep this too\ntheme: dark\n";
        std::fs::write(&config_path, config_original).unwrap();

        let list = |window| {
            let mut m = entry(None, "docker.io/ai/qwen3.5:0.8b");
            m.context_window = window;
            ModelList::of(vec![m])
        };
        let server = "http://127.0.0.1:17434";
        write_omp_config_in_dir(&dir, &list(None), server).unwrap();
        write_omp_config_in_dir(&dir, &list(Some(4096)), server).unwrap();

        assert_eq!(
            std::fs::read_to_string(path.with_extension("yml.bak")).unwrap(),
            original
        );
        let mut hand_edited = std::fs::read_to_string(&path).unwrap();
        hand_edited.push_str("# added after the first launch\n");
        std::fs::write(&path, &hand_edited).unwrap();
        write_omp_config_in_dir(&dir, &list(Some(8192)), server).unwrap();
        assert_eq!(
            std::fs::read_to_string(path.with_extension("yml.bak")).unwrap(),
            hand_edited
        );
        let parsed = parsed(&path);
        assert_eq!(parsed["providers"]["openai"]["apiKey"], "sk-mine");
        assert_eq!(parsed["defaults"]["temperature"], 0.2);
        assert_eq!(
            parsed["providers"]["ollama"]["models"][0]["id"],
            "docker.io/ai/qwen3.5:0.8b"
        );
        assert_eq!(
            std::fs::read_to_string(config_path.with_extension("yml.bak")).unwrap(),
            config_original
        );
        let config = self::parsed(&config_path);
        assert_eq!(config["theme"], "dark");
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
