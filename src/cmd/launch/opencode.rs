//! `llmman launch opencode`.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common::write_json_merged;
use super::{
    exec_with_env, find_on_path, server, thinking_choices, xdg_dir, Thinking, WINDOWS_PATH_EXTS,
};

/// opencode: a JSON config via OPENCODE_CONFIG_CONTENT pointing at our
/// /v1 endpoint, with the model's thinking variants, its window and, for
/// a vision model, image input. `--variant` becomes the model's default
/// options and its selection in opencode's state.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_opencode(
    model: &str,
    api_key: &str,
    thinking: Option<&Thinking>,
    variant: Option<&str>,
    vision: bool,
    context_window: Option<u64>,
    max_output: Option<u32>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_opencode().ok_or_else(|| anyhow::anyhow!("opencode is not installed"))?;

    let effective_model = if model.is_empty() { "default" } else { model };
    let variants = opencode_variants(thinking);
    let options = variant.and_then(|v| variants.iter().find(|(name, _)| *name == v));
    let config = opencode_config(
        &server(),
        effective_model,
        api_key,
        &variants,
        options.map(|(_, options)| options),
        vision,
        context_window,
        max_output,
    );
    if let Some(variant) = variant {
        write_opencode_variant(&opencode_state_dir()?, effective_model, variant)?;
    }

    exec_with_env(&bin, extra_args, &[("OPENCODE_CONFIG_CONTENT", &config)])
}

/// opencode's state directory, where its `xdg-basedir` puts it.
pub(super) fn opencode_state_dir() -> anyhow::Result<PathBuf> {
    let home = crate::config::home_dir().context("no home directory")?;
    Ok(xdg_dir(&home, "XDG_STATE_HOME", ".local/state").join("opencode"))
}

/// How opencode names `model` on llmman's provider.
fn opencode_model_ref(model: &str) -> String {
    format!("ollama/{model}")
}

/// Selects `variant` for `model` in opencode's `model.json`, as ctrl+t
/// does. Its UI sends that selection, which outranks the model's
/// `options`; only `opencode run` sends those alone.
fn write_opencode_variant(state_dir: &Path, model: &str, variant: &str) -> anyhow::Result<()> {
    write_json_merged(&state_dir.join("model.json"), "opencode", |existing| {
        let mut merged = existing.clone();
        if !merged["variant"].is_object() {
            merged["variant"] = serde_json::json!({});
        }
        merged["variant"][opencode_model_ref(model)] = variant.into();
        merged
    })
}

/// opencode's `variants` for the model, in cycle order (`variant_cycle`,
/// ctrl+t by default): [`thinking_choices`]. Each variant is the request
/// options `@ai-sdk/openai-compatible` sends: `reasoningEffort` as
/// `reasoning_effort`, other keys verbatim. opencode derives variants
/// only for models it knows from models.dev, so without these a model
/// has nothing to cycle.
/// A switch-only model's two variants set both keys, so either overrides
/// the other when `--variant` put it in the model's options.
pub(super) fn opencode_variants(thinking: Option<&Thinking>) -> Vec<(&str, serde_json::Value)> {
    let choices = thinking_choices(thinking);
    let switch = choices.contains(&"thinking");
    choices
        .into_iter()
        .map(|choice| {
            let options = match choice {
                "none" if switch => serde_json::json!({
                    "reasoningEffort": "none",
                    "chat_template_kwargs": { "enable_thinking": false },
                }),
                "thinking" => serde_json::json!({
                    "reasoningEffort": "medium",
                    "chat_template_kwargs": { "enable_thinking": true },
                }),
                level => serde_json::json!({ "reasoningEffort": level }),
            };
            (choice, options)
        })
        .collect()
}

/// Finds opencode on `PATH`, then where its installers put it. The second
/// check finds a fresh install that this process's `PATH` doesn't include
/// yet.
pub(super) fn find_opencode() -> Option<PathBuf> {
    find_on_path("opencode").or_else(|| opencode_fallback_paths().into_iter().find(|p| p.is_file()))
}

/// Where opencode's installers put it: `~/.opencode/bin` (install script)
/// and, on Windows, `%APPDATA%\npm` (`npm install -g`).
fn opencode_fallback_paths() -> Vec<PathBuf> {
    let home = dirs::home_dir();
    if cfg!(windows) {
        let mut paths: Vec<PathBuf> = home
            .iter()
            .map(|h| h.join(".opencode").join("bin").join("opencode.exe"))
            .collect();
        // Treat an empty APPDATA as unset.
        let roaming = std::env::var_os("APPDATA")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join("AppData").join("Roaming")));
        if let Some(npm) = roaming.map(|d| d.join("npm")) {
            paths.extend(
                WINDOWS_PATH_EXTS
                    .iter()
                    .map(|ext| npm.join(format!("opencode.{ext}"))),
            );
        }
        return paths;
    }
    home.iter()
        .map(|h| h.join(".opencode").join("bin").join("opencode"))
        .collect()
}

/// opencode's own `OUTPUT_TOKEN_MAX` (`provider/transform.ts`), the cap
/// it applies to `limit.output` and the value it substitutes for a 0.
const OPENCODE_OUTPUT_TOKEN_MAX: u64 = 32_000;

/// `limit.output` for a window of `context` when the catalog names no
/// real ceiling: a quarter, capped at [`OPENCODE_OUTPUT_TOKEN_MAX`],
/// never 0.
///
/// opencode spends this field twice — the `maxOutputTokens` it sends
/// and the headroom it keeps before compacting — capping both at its
/// own [`OPENCODE_OUTPUT_TOKEN_MAX`], which is what makes a catalog
/// ceiling of any size safe to pass. A quarter is the guess where
/// there is none: longer than one turn produces, short enough to leave
/// the window mostly usable.
fn opencode_output_reserve(context: u64) -> u64 {
    (context / 4).clamp(1, OPENCODE_OUTPUT_TOKEN_MAX)
}

/// The `OPENCODE_CONFIG_CONTENT` for `model` at `server`. Structs rather
/// than `json!`, whose map sorts keys: opencode cycles variants in the
/// order listed. No variants leaves the key out.
#[allow(clippy::too_many_arguments)]
fn opencode_config(
    server: &str,
    model: &str,
    api_key: &str,
    variants: &[(&str, serde_json::Value)],
    options: Option<&serde_json::Value>,
    vision: bool,
    context_window: Option<u64>,
    max_output: Option<u32>,
) -> String {
    use serde::ser::{SerializeMap, Serializer};

    /// An object with runtime keys, in the order given.
    fn entries<S: Serializer, V: serde::Serialize>(
        entries: &[(&str, V)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (key, value) in entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }

    #[derive(serde::Serialize)]
    struct Config<'a> {
        #[serde(rename = "$schema")]
        schema: &'static str,
        provider: Providers<'a>,
        model: String,
    }
    #[derive(serde::Serialize)]
    struct Providers<'a> {
        ollama: Provider<'a>,
    }
    #[derive(serde::Serialize)]
    struct Provider<'a> {
        npm: &'static str,
        name: &'static str,
        options: Options<'a>,
        #[serde(serialize_with = "entries")]
        models: [(&'a str, Model<'a>); 1],
    }
    #[derive(serde::Serialize)]
    struct Options<'a> {
        #[serde(rename = "baseURL")]
        base_url: String,
        #[serde(rename = "apiKey")]
        api_key: &'a str,
    }
    #[derive(serde::Serialize)]
    struct Model<'a> {
        name: &'a str,
        #[serde(serialize_with = "entries", skip_serializing_if = "<[_]>::is_empty")]
        variants: &'a [(&'a str, serde_json::Value)],
        #[serde(skip_serializing_if = "Option::is_none")]
        options: Option<&'a serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        modalities: Option<Modalities>,
        #[serde(skip_serializing_if = "Option::is_none")]
        attachment: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<Limit>,
    }
    #[derive(serde::Serialize)]
    struct Modalities {
        input: &'static [&'static str],
        output: &'static [&'static str],
    }
    /// opencode's schema requires both fields once `limit` is present.
    #[derive(serde::Serialize)]
    struct Limit {
        context: u64,
        output: u64,
    }

    // Declare image input for a vision model so opencode will attach
    // images; a text-only model gets neither key.
    let modalities = vision.then_some(Modalities {
        input: &["text", "image"],
        output: &["text"],
    });

    // Without `limit`, opencode normalizes a config-defined model to
    // `{context: 0, output: 0}` and then skips overflow detection
    // entirely for a 0 context (`session/overflow.ts`, `isOverflow`), so
    // a session never auto-compacts. Declared only when llmman knows the
    // window; otherwise the key stays out rather than assert a guess.
    let limit = context_window.map(|context| Limit {
        context,
        // The catalog's own ceiling wherever there is one; derived
        // only for a local model, which has no catalog to name one.
        output: max_output.map_or_else(|| opencode_output_reserve(context), u64::from),
    });

    let config = Config {
        schema: "https://opencode.ai/config.json",
        provider: Providers {
            ollama: Provider {
                npm: "@ai-sdk/openai-compatible",
                name: "Ollama",
                options: Options {
                    base_url: format!("{server}/v1"),
                    api_key,
                },
                models: [(
                    model,
                    Model {
                        name: model,
                        variants,
                        options,
                        modalities,
                        attachment: vision.then_some(true),
                        limit,
                    },
                )],
            },
        },
        model: opencode_model_ref(model),
    };
    serde_json::to_string(&config).expect("opencode config serializes")
}

#[cfg(test)]
mod tests {
    use super::super::PORTABLE_THINKING_LEVELS;
    use super::*;
    use crate::chat_template::ThinkingControls;

    /// The config points at the daemon's `/v1` and lists the variants in
    /// the order given (a parsed `Value` would re-sort them).
    #[test]
    fn opencode_config_lists_the_variants_in_order() {
        let variants = opencode_variants(None);
        let text = opencode_config(
            "http://127.0.0.1:17434",
            "qwen3.5:0.8b",
            "k",
            &variants,
            None,
            false,
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(config["$schema"], "https://opencode.ai/config.json");
        assert_eq!(config["model"], "ollama/qwen3.5:0.8b");
        let provider = &config["provider"]["ollama"];
        assert_eq!(provider["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(provider["name"], "Ollama");
        assert_eq!(provider["options"]["baseURL"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["options"]["apiKey"], "k");
        assert_eq!(provider["models"].as_object().map(|m| m.len()), Some(1));

        let model = &provider["models"]["qwen3.5:0.8b"];
        assert_eq!(model["name"], "qwen3.5:0.8b");
        let written = model["variants"].as_object().expect("variants object");
        assert_eq!(written.len(), variants.len());
        for (name, options) in &variants {
            assert_eq!(&written[*name], options, "variant {name}");
        }
        let positions: Vec<usize> = variants
            .iter()
            .map(|(name, _)| text.find(&format!("\"{name}\"")).expect(name))
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");

        let bare = opencode_config("http://h", "m", "k", &[], None, false, None, None);
        assert!(!bare.contains("variants"), "{bare}");
    }

    /// `--variant` makes its options the model's own, which requests that
    /// name no variant (all of `opencode run`'s) carry.
    #[test]
    fn opencode_config_starts_the_model_at_the_variant() {
        let variants = opencode_variants(None);
        let high = variants.iter().find(|(name, _)| *name == "high").unwrap();
        let text = opencode_config(
            "http://h",
            "m",
            "k",
            &variants,
            Some(&high.1),
            false,
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let model = &config["provider"]["ollama"]["models"]["m"];
        assert_eq!(
            model["options"],
            serde_json::json!({ "reasoningEffort": "high" })
        );

        let text = opencode_config("http://h", "m", "k", &variants, None, false, None, None);
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert!(config["provider"]["ollama"]["models"]["m"]["options"].is_null());
    }

    #[test]
    fn write_opencode_variant_selects_the_model_and_keeps_the_rest() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-opencode-state-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("state").join("opencode").join("model.json");
        let state = path.parent().unwrap();
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
        };

        // No file yet.
        write_opencode_variant(state, "m", "xhigh").unwrap();
        assert_eq!(
            read(),
            serde_json::json!({ "variant": { "ollama/m": "xhigh" } })
        );

        // Another model's selection and the rest of the state stay.
        std::fs::write(
            &path,
            r#"{"recent":[{"modelID":"o"}],"variant":{"ollama/m":"medium","ollama/o":"low"}}"#,
        )
        .unwrap();
        write_opencode_variant(state, "m", "xhigh").unwrap();
        let written = read();
        assert_eq!(written["variant"]["ollama/m"], "xhigh");
        assert_eq!(written["variant"]["ollama/o"], "low");
        assert_eq!(written["recent"][0]["modelID"], "o");

        // A `variant` that is not an object is replaced; a file that is not
        // a JSON object is left alone, as opencode resets it itself.
        std::fs::write(&path, r#"{"variant":"high"}"#).unwrap();
        write_opencode_variant(state, "m", "low").unwrap();
        assert_eq!(read()["variant"], serde_json::json!({ "ollama/m": "low" }));
        std::fs::write(&path, "[1, 2").unwrap();
        write_opencode_variant(state, "m", "low").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[1, 2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opencode_config_declares_image_input_only_for_a_vision_model() {
        let text = opencode_config("http://h", "m", "k", &[], None, true, None, None);
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let model = &config["provider"]["ollama"]["models"]["m"];
        assert_eq!(
            model["modalities"],
            serde_json::json!({ "input": ["text", "image"], "output": ["text"] })
        );
        assert_eq!(model["attachment"], true);

        let text_only = opencode_config("http://h", "m", "k", &[], None, false, None, None);
        assert!(!text_only.contains("modalities"), "{text_only}");
        assert!(!text_only.contains("attachment"), "{text_only}");
    }

    /// The reserve scales with the window, so a small one keeps a
    /// usable budget.
    #[test]
    fn opencode_output_reserve_scales_with_the_window_and_is_never_zero() {
        let cases = [
            (4096, 1024),
            (32768, 8192),
            (131072, 32000),
            // Capped however large the window.
            (1 << 20, OPENCODE_OUTPUT_TOKEN_MAX),
            // Never 0, which opencode would replace with its own max.
            (1, 1),
            (3, 1),
        ];
        for (context, want) in cases {
            assert_eq!(opencode_output_reserve(context), want, "context={context}");
        }
    }

    /// Without `limit` opencode never auto-compacts, so the window has
    /// to travel.
    #[test]
    fn opencode_config_declares_the_window_only_when_it_is_known() {
        let text = opencode_config("http://h", "m", "k", &[], None, false, Some(8192), None);
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let limit = &config["provider"]["ollama"]["models"]["m"]["limit"];
        assert_eq!(limit["context"], 8192);
        // Both keys are required once `limit` is present.
        assert_eq!(limit["output"], 2048);

        // No window: the key stays out and opencode keeps its defaults.
        let unknown = opencode_config("http://h", "m", "k", &[], None, false, None, None);
        assert!(!unknown.contains("limit"), "{unknown}");
    }

    /// opencode sends `limit.output` as the request's max output, so a
    /// hosted model's own ceiling has to travel: deriving one from the
    /// window advertises more than the provider will accept.
    #[test]
    fn opencode_config_prefers_the_catalogs_output_ceiling_to_a_derived_one() {
        let hosted = opencode_config(
            "http://h",
            "m",
            "k",
            &[],
            None,
            false,
            Some(128_000),
            Some(8_192),
        );
        let config: serde_json::Value = serde_json::from_str(&hosted).expect("valid JSON");
        let limit = &config["provider"]["ollama"]["models"]["m"]["limit"];
        assert_eq!(limit["context"], 128_000);
        assert_eq!(
            limit["output"], 8_192,
            "the catalog's ceiling, not a quarter of the window"
        );

        // A local model has no catalog and no ceiling of its own, so the
        // derived reserve still stands.
        let local = opencode_config("http://h", "m", "k", &[], None, false, Some(128_000), None);
        let config: serde_json::Value = serde_json::from_str(&local).expect("valid JSON");
        assert_eq!(
            config["provider"]["ollama"]["models"]["m"]["limit"]["output"],
            OPENCODE_OUTPUT_TOKEN_MAX
        );
    }

    #[test]
    fn opencode_config_escapes_the_model_name() {
        let model = "we\"ird/mo\\del";
        let config: serde_json::Value = serde_json::from_str(&opencode_config(
            "http://h",
            model,
            "k",
            &[],
            None,
            false,
            None,
            None,
        ))
        .expect("valid JSON");
        assert_eq!(config["model"], format!("ollama/{model}"));
        assert_eq!(config["provider"]["ollama"]["models"][model]["name"], model);
    }

    /// Each choice becomes the options that select it; no template means
    /// the portable set, no thinking means no variants.
    #[test]
    fn opencode_variants_follow_the_templates_controls() {
        let gemma4 = ThinkingControls {
            thinks: true,
            enable_thinking: true,
            efforts: vec![],
        };
        assert_eq!(
            opencode_variants(Some(&Thinking::Template(gemma4.clone()))),
            [
                (
                    "none",
                    serde_json::json!({
                        "reasoningEffort": "none",
                        "chat_template_kwargs": { "enable_thinking": false },
                    })
                ),
                (
                    "thinking",
                    serde_json::json!({
                        "reasoningEffort": "medium",
                        "chat_template_kwargs": { "enable_thinking": true },
                    })
                ),
            ]
        );
        let qwen3_8 = ThinkingControls {
            efforts: vec!["low", "medium", "xhigh"],
            ..gemma4
        };
        assert_eq!(
            opencode_variants(Some(&Thinking::Template(qwen3_8))),
            [
                ("none", serde_json::json!({ "reasoningEffort": "none" })),
                ("low", serde_json::json!({ "reasoningEffort": "low" })),
                ("medium", serde_json::json!({ "reasoningEffort": "medium" })),
                ("xhigh", serde_json::json!({ "reasoningEffort": "xhigh" })),
            ]
        );
        let plain = Thinking::Template(ThinkingControls::default());
        assert!(opencode_variants(Some(&plain)).is_empty());
        let fallback = opencode_variants(None);
        assert_eq!(fallback.len(), PORTABLE_THINKING_LEVELS.len());
        assert_eq!(fallback[0].0, "none");
    }

    /// A provider's model cycles exactly the catalog's levels, with no
    /// added `none`; an empty list means no variants, not the portable set.
    #[test]
    fn opencode_variants_follow_the_catalogs_levels() {
        let claude = Thinking::Listed(
            ["low", "medium", "high", "xhigh", "max"]
                .map(String::from)
                .to_vec(),
        );
        let variants = opencode_variants(Some(&claude));
        assert_eq!(
            variants.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            ["low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            variants[4].1,
            serde_json::json!({ "reasoningEffort": "max" })
        );
        assert!(opencode_variants(Some(&Thinking::Listed(Vec::new()))).is_empty());
        assert!(claude.template().is_none());
    }
}
