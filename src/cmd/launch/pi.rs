//! `llmman launch pi`.

use std::path::PathBuf;

use anyhow::Context;

use super::common;
use super::{exec_with_env, find_on_path, server, HostedModel};

/// pi: register llmman as an OpenAI-compatible provider in `models.json`
/// and point `settings.json` at it. pi runs with the caller's own
/// arguments and nothing else — `settings.json` already selects the
/// model, and a key on argv would be readable in `ps`.
///
/// pi is in [`super::PROVIDER_NEEDS_DAEMON_KEY`] for that reason: a `--provider`
/// credential stays with the daemon, and the stored `apiKey` is the
/// literal placeholder codex and hermes also write. An environment
/// reference (`"$VAR"`, which pi does interpolate) would read as
/// "unresolved" outside `llmman launch` and take the provider down.
///
/// `hosted` is every model `--provider` serves (see [`HostedModel`]):
/// each becomes an entry in the provider's `models`, so pi's own model
/// picker can switch between them mid-session; the launched model stays
/// the one `settings.json` selects.
pub(super) fn launch_pi(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    extra_args: &[String],
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let bin = find_on_path("pi").ok_or_else(|| anyhow::anyhow!("pi is not installed"))?;
    write_pi_config(model, reasons, vision, context_window, hosted)?;
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

/// Writes pi's `models.json` provider and points `settings.json` at it.
/// Both go through [`common::write_json_merged`], which qwen writes
/// through too.
fn write_pi_config(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let dir = pi_agent_dir()?;
    let entries = pi_model_entries(model, reasons, vision, context_window, hosted);
    common::write_json_merged(&dir.join("models.json"), "pi", |existing| {
        entries.iter().fold(existing.clone(), |acc, entry| {
            pi_models_merged(&acc, &server(), entry)
        })
    })?;
    common::write_json_merged(&dir.join("settings.json"), "pi", |existing| {
        pi_settings_merged(existing, model)
    })
}

/// The model entries the provider lists: the one model this launch
/// resolved, or — under `--provider` — every model the provider serves,
/// the launched one first, each with what the catalog says about it.
/// A non-launched model reasons when the catalog lists thinking levels
/// for it, and takes image input when the catalog says it does; unknown
/// reads as text-only, as [`common::pi_model_entry`] does for a local
/// model with no window to declare. Every entry also carries the
/// catalog's published output ceiling as `maxTokens`, the launched
/// model included — `pi_model_entry` cannot emit an output limit, so
/// it rides beside it, omitted where the catalog publishes none.
fn pi_model_entries(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    hosted: Option<&[HostedModel]>,
) -> Vec<serde_json::Value> {
    let Some(hosted) = hosted else {
        return vec![common::pi_model_entry(
            model,
            reasons,
            vision,
            context_window,
        )];
    };
    let mut entries: Vec<serde_json::Value> = hosted
        .iter()
        .map(|m| {
            let mut entry = if m.reference == model {
                common::pi_model_entry(model, reasons, vision, context_window)
            } else {
                common::pi_model_entry(
                    &m.reference,
                    m.thinking.as_ref().is_some_and(|levels| !levels.is_empty()),
                    m.vision.unwrap_or(false),
                    m.context,
                )
            };
            if let Some(output) = m.output {
                entry["maxTokens"] = serde_json::json!(output);
            }
            entry
        })
        .collect();
    // The launched model belongs in its own config even if the list
    // somehow lacks it: pi is started on it.
    if !entries
        .iter()
        .any(|entry| entry["id"].as_str() == Some(model))
    {
        entries.insert(
            0,
            common::pi_model_entry(model, reasons, vision, context_window),
        );
    }
    entries
}

/// `existing` with llmman's provider updated around `entry`, keeping
/// everything it does not own.
///
/// The provider object is merged, not replaced: `baseUrl` always points
/// at the running daemon, `api` and `apiKey` are filled in only when
/// absent, and anything else under it — `headers`, `compat`,
/// `modelOverrides` — is left as the user wrote it. Only the model entry
/// with this same `id` is rebuilt, so launching a second model does not
/// drop the first, whoever added it.
fn pi_models_merged(
    existing: &serde_json::Value,
    server: &str,
    entry: &serde_json::Value,
) -> serde_json::Value {
    common::provider_models_merged(existing, PI_PROVIDER, entry, false, |provider| {
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
    use super::common::pi_model_entry;
    use super::*;

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
        let entry = pi_model_entry("qwen3.5:0.8b", false, false, None);
        let merged = pi_models_merged(&existing, "http://127.0.0.1:17434", &entry);

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
        let ids: Vec<&str> = merged["providers"]["llmman"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        // Both survive; llmman's own is rebuilt in place, not doubled.
        assert_eq!(ids, vec!["mine", "stale", "qwen3.5:0.8b"]);
    }

    /// A file may hold `"providers": []` and still be valid JSON;
    /// indexing that with a key panics, so it is replaced rather than
    /// indexed (see `object_under`).
    #[test]
    fn pi_models_merged_survives_a_wrong_shaped_providers_value() {
        for raw in [r#"{"providers": []}"#, r#"{"providers": "x"}"#, r#"{}"#] {
            let existing: serde_json::Value = serde_json::from_str(raw).unwrap();
            let entry = pi_model_entry("m", false, false, None);
            let merged = pi_models_merged(&existing, "http://s", &entry);
            assert_eq!(merged["providers"]["llmman"]["baseUrl"], "http://s/v1");
            assert_eq!(merged["providers"]["llmman"]["models"][0]["id"], "m");
        }
    }

    /// Nothing was there before, so llmman fills in the fields it owns.
    #[test]
    fn pi_models_merged_writes_the_placeholder_key_into_a_fresh_provider() {
        let entry = pi_model_entry("m", false, false, None);
        let merged = pi_models_merged(&serde_json::json!({}), "http://127.0.0.1:17434", &entry);
        let provider = &merged["providers"]["llmman"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["api"], "openai-completions");
        // Literal, never an interpolated reference — see launch_pi.
        assert_eq!(provider["apiKey"], "llmman");
    }

    /// Launching a second model must not drop the first.
    #[test]
    fn pi_models_merged_keeps_a_previously_launched_model() {
        let first = pi_model_entry("a", false, false, None);
        let one = pi_models_merged(&serde_json::json!({}), "http://s", &first);
        let two = pi_models_merged(&one, "http://s", &pi_model_entry("b", false, false, None));
        let ids: Vec<&str> = two["providers"]["llmman"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["a", "b"]);
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
    /// pi's own model picker can switch between them without a relaunch:
    /// the launched model first, with the window and image input this
    /// launch resolved for it, and each other model with what its own
    /// catalog entry says.
    #[test]
    fn pi_model_entries_list_every_model_the_provider_serves() {
        let entries = pi_model_entries(
            "llmman.provider/openrouter/qwen3-coder",
            false,
            false,
            Some(262_144),
            Some(&hosted_pair()),
        );
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], "llmman.provider/openrouter/qwen3-coder");
        assert_eq!(entries[0]["contextWindow"], 262_144);
        // The fixture publishes an 8,192-token output ceiling for the
        // launched model; it reaches the entry as `maxTokens`.
        assert_eq!(entries[0]["maxTokens"], 8_192);
        assert_eq!(entries[0]["input"], serde_json::json!(["text"]));
        // The launched entry keeps the launch's own reasoning, not the
        // catalog's levels: a `--variant` is what marks it as reasoning.
        assert_eq!(entries[0].get("reasoning"), None);
        assert_eq!(entries[1]["id"], "llmman.provider/openrouter/llama-4-scout");
        assert_eq!(entries[1]["input"], serde_json::json!(["text", "image"]));
        assert_eq!(entries[1].get("contextWindow"), None);
        assert_eq!(entries[1].get("reasoning"), None);
        // The catalog publishes no output ceiling for the scout, so none
        // is written rather than a guess.
        assert_eq!(entries[1].get("maxTokens"), None);

        // A model the catalog lists thinking levels for reasons.
        let thinking = vec![HostedModel {
            thinking: Some(vec!["low".into()]),
            ..hosted_pair().into_iter().nth(1).unwrap()
        }];
        let entries = pi_model_entries("m", false, false, None, Some(&thinking));
        assert_eq!(entries[0]["id"], "m");
        // The inserted launched model has no catalog row, so it carries
        // no output ceiling either.
        assert_eq!(entries[0].get("maxTokens"), None);
        assert_eq!(entries[1]["reasoning"], true);
    }

    /// The launched model belongs in the list even if the provider's
    /// list somehow lacks it, ahead of the rest.
    #[test]
    fn pi_model_entries_keep_the_launched_model_when_the_list_lacks_it() {
        let others = vec![hosted_pair().into_iter().nth(1).unwrap()];
        let entries = pi_model_entries("m", false, true, Some(8_192), Some(&others));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["id"], "m");
        assert_eq!(entries[0]["input"], serde_json::json!(["text", "image"]));
        assert_eq!(entries[0]["contextWindow"], 8_192);
        assert_eq!(entries[1]["id"], "llmman.provider/openrouter/llama-4-scout");
    }

    /// Merging the entries keeps models a previous launch registered.
    #[test]
    fn pi_model_entries_merge_beside_a_previously_launched_model() {
        let existing = pi_models_merged(
            &serde_json::json!({}),
            "http://s",
            &pi_model_entry("earlier", false, false, None),
        );
        let entries = pi_model_entries(
            "llmman.provider/openrouter/qwen3-coder",
            false,
            false,
            Some(262_144),
            Some(&hosted_pair()),
        );
        let merged = entries.iter().fold(existing, |acc, entry| {
            pi_models_merged(&acc, "http://s", entry)
        });
        let ids: Vec<&str> = merged["providers"]["llmman"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            vec![
                "earlier",
                "llmman.provider/openrouter/qwen3-coder",
                "llmman.provider/openrouter/llama-4-scout"
            ]
        );
    }

    #[test]
    fn pi_settings_merged_leaves_unrelated_settings_alone() {
        let existing = serde_json::json!({ "theme": "dark", "defaultModel": "old" });
        let merged = pi_settings_merged(&existing, "qwen3.5:0.8b");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["defaultProvider"], "llmman");
        assert_eq!(merged["defaultModel"], "qwen3.5:0.8b");
    }
}
