//! `llmman launch opencode`.

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common::{write_if_changed, write_json_merged};
use super::optimizer::Optimizer;
use super::{
    exec_with_env, find_on_path, server, thinking_choices, xdg_dir, HostedModel, Thinking,
    WINDOWS_PATH_EXTS,
};

/// opencode: a JSON config pointing at our /v1 endpoint, with the
/// model's thinking variants, its window and, for a vision model,
/// image input. `--variant` becomes the model's default options and
/// its selection in opencode's state.
///
/// The config travels in two pieces (see `launch_opencode`): the model
/// catalog in a file opencode reads via `OPENCODE_CONFIG`, and the
/// launch itself — the model, the provider endpoint and the key — in a
/// small `OPENCODE_CONFIG_CONTENT` override. The catalog is unbounded
/// (a provider with 1,000 models is over the 128 KiB a single
/// environment value may hold), the override is not.
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
    optimizer: Optimizer,
    hosted: Option<&[HostedModel]>,
) -> anyhow::Result<()> {
    let bin = find_opencode().ok_or_else(|| anyhow::anyhow!("opencode is not installed"))?;
    let added = optimizer.addition("opencode")?;

    let effective_model = if model.is_empty() { "default" } else { model };
    let variants = opencode_variants(thinking);
    let options = variant.and_then(|v| variants.iter().find(|(name, _)| *name == v));
    let server = added.server.clone().unwrap_or_else(server);
    let state_dir = opencode_state_dir()?;
    let config = opencode_config(
        &server,
        effective_model,
        None,
        &variants,
        options.map(|(_, options)| options),
        vision,
        context_window,
        max_output,
        hosted,
    );
    let config_path = write_opencode_config_file(&state_dir, &config)?;
    let inline = with_added_config(
        &opencode_inline_config(&server, effective_model, api_key),
        added.config.as_ref(),
    );
    if let Some(variant) = variant {
        write_opencode_variant(&state_dir, effective_model, variant)?;
    }

    if let Some(notice) = optimizer.notice("opencode") {
        eprintln!("{notice}");
    }
    let config_path = config_path.to_string_lossy().into_owned();
    exec_with_env(
        &bin,
        extra_args,
        &[
            ("OPENCODE_CONFIG", config_path.as_str()),
            ("OPENCODE_CONFIG_CONTENT", &inline),
        ],
    )
}

/// `config` (a JSON object) with the keys of `added` appended, so the order
/// the rest of it was written in (the variants') is kept.
fn with_added_config(config: &str, added: Option<&serde_json::Value>) -> String {
    let Some(added) = added.and_then(serde_json::Value::as_object) else {
        return config.to_string();
    };
    let mut out = config.trim_end().trim_end_matches('}').to_string();
    for (key, value) in added {
        out.push_str(&format!(
            ",{}:{value}",
            serde_json::Value::from(key.as_str())
        ));
    }
    out.push('}');
    out
}

/// Refuses an opencode before 2, which has not the plugins `tk` adds.
pub(super) fn ensure_plugins() -> anyhow::Result<()> {
    let bin = find_opencode().ok_or_else(|| anyhow::anyhow!("opencode is not installed"))?;
    let out = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .with_context(|| format!("run {} --version", bin.display()))?;
    let version = String::from_utf8_lossy(&out.stdout);
    anyhow::ensure!(
        major_version(&version).is_some_and(|major| major >= 2),
        "--optimizer tk needs opencode 2 or later, which has the plugin hook it uses; \
         this is {}.\nUpgrade opencode, or launch without --optimizer.",
        version.trim()
    );
    Ok(())
}

/// The major version in `opencode --version`'s output (`opencode v2.3.0`).
fn major_version(output: &str) -> Option<u64> {
    let version = output.trim().rsplit(' ').next()?;
    version
        .trim_start_matches('v')
        .split('.')
        .next()?
        .parse()
        .ok()
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

/// opencode's config for `model` at `server`. Structs rather than
/// `json!`, whose map sorts keys: opencode cycles variants in the
/// order listed. No variants leaves the key out.
///
/// `api_key` is `None` in the copy written to a file (see
/// `launch_opencode`): the key travels in the small inline override
/// instead, so it never lands in a file — and so the file copy, the
/// one that carries the unbounded model catalog, is never an
/// environment value either.
///
/// `hosted` is every model `--provider` serves (see [`HostedModel`]); with
/// it, the config lists all of them, so opencode's own model picker can
/// switch between them mid-session. The launched model keeps the variants,
/// `--variant` options, window and image input this launch resolved for
/// it; each other model takes what the provider's catalog says about it —
/// its own thinking levels, window, output ceiling and image input.
#[allow(clippy::too_many_arguments)]
fn opencode_config(
    server: &str,
    model: &str,
    api_key: Option<&str>,
    variants: &[(&str, serde_json::Value)],
    options: Option<&serde_json::Value>,
    vision: bool,
    context_window: Option<u64>,
    max_output: Option<u32>,
    hosted: Option<&[HostedModel]>,
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
        models: Vec<(&'a str, Model<'a>)>,
    }
    #[derive(serde::Serialize)]
    struct Options<'a> {
        #[serde(rename = "baseURL")]
        base_url: String,
        #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none")]
        api_key: Option<&'a str>,
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
        limit: Option<&'a Limit>,
    }
    #[derive(Clone, Copy, serde::Serialize)]
    struct Modalities {
        input: &'static [&'static str],
        output: &'static [&'static str],
    }

    /// One row of `provider.ollama.models`, owned so the whole list can
    /// borrow from it after the loop that fills it. Each row carries
    /// its own variants, options, window and image input: the launched
    /// model's, what this launch resolved; any other model's, what
    /// the provider's catalog says about it.
    struct Row<'a> {
        /// The map key: the reference the daemon routes a request on.
        reference: String,
        /// What opencode shows beside it.
        name: String,
        variants: Vec<(&'a str, serde_json::Value)>,
        options: Option<serde_json::Value>,
        /// Image input for a model that takes images in, from its own
        /// catalog row; a text-only model gets neither key.
        modalities: Option<Modalities>,
        /// Whether opencode may attach images to this model.
        attachment: Option<bool>,
        limit: Option<Limit>,
    }

    // Image input (and text output) for a model that takes images in;
    // a text-only model gets neither key.
    let modalities_for = |vision: bool| {
        vision.then_some(Modalities {
            input: &["text", "image"],
            output: &["text"],
        })
    };

    // Without `limit`, opencode normalizes a config-defined model to
    // `{context: 0, output: 0}` and then skips overflow detection
    // entirely for a 0 context (`session/overflow.ts`, `isOverflow`), so
    // a session never auto-compacts. Declared only when llmman knows the
    // window; otherwise the key stays out rather than assert a guess.
    let limit = context_window.map(|context| model_limit(context, max_output));

    // Each model's own thinking levels, read once so the rows below can
    // borrow them (see `opencode_variants`): the list has to outlive the
    // rows, so it is built here rather than inside the match arm.
    let thinkings: Vec<Option<Thinking>> = hosted
        .as_ref()
        .map(|models| {
            models
                .iter()
                .map(|m| m.thinking.clone().map(Thinking::Listed))
                .collect()
        })
        .unwrap_or_default();

    // One path builds every row from the shared hosted list: a local
    // launch has no list, so the launched model is the whole config.
    // Each row carries its own variants, options, window and image
    // input — the launched model keeps what this launch resolved for
    // it, every other model what the provider's catalog says about
    // it. The launched model belongs in its own config even if the
    // list somehow lacks it: opencode is started on it.
    let mut rows: Vec<Row<'_>> = Vec::with_capacity(hosted.map_or(1, |h| h.len()));
    if let Some(hosted) = hosted {
        for (m, thinking) in hosted.iter().zip(&thinkings) {
            let launched = m.reference == model;
            // This row's own image input: the launch's, for the
            // launched model, the catalog's for any other (unknown
            // reads as no image input rather than a guess).
            let row_vision = if launched {
                vision
            } else {
                m.vision.unwrap_or(false)
            };
            rows.push(Row {
                reference: m.reference.clone(),
                name: m.id.clone(),
                variants: if launched {
                    variants.to_vec()
                } else {
                    opencode_variants(thinking.as_ref())
                },
                options: if launched { options.cloned() } else { None },
                modalities: modalities_for(row_vision),
                attachment: row_vision.then_some(true),
                limit: if launched {
                    limit
                } else {
                    m.context.map(|context| model_limit(context, m.output))
                },
            });
        }
    }
    if !rows.iter().any(|row| row.reference == model) {
        rows.insert(
            0,
            Row {
                reference: model.to_string(),
                name: model.to_string(),
                variants: variants.to_vec(),
                options: options.cloned(),
                modalities: modalities_for(vision),
                attachment: vision.then_some(true),
                limit,
            },
        );
    }

    let models: Vec<(&str, Model<'_>)> = rows
        .iter()
        .map(|row| {
            (
                row.reference.as_str(),
                Model {
                    name: &row.name,
                    variants: &row.variants,
                    options: row.options.as_ref(),
                    modalities: row.modalities,
                    attachment: row.attachment,
                    limit: row.limit.as_ref(),
                },
            )
        })
        .collect();

    let config = Config {
        schema: "https://opencode.ai/config.json",
        provider: Providers {
            // The id stays `ollama`: saved model selections are keyed by it
            // (`ollama/<model>`). Only the name opencode shows beside the
            // model changes, and the provider is llmman, not Ollama.
            ollama: Provider {
                npm: "@ai-sdk/openai-compatible",
                name: "llmman",
                options: Options {
                    base_url: format!("{server}/v1"),
                    api_key,
                },
                models,
            },
        },
        model: opencode_model_ref(model),
    };
    serde_json::to_string(&config).expect("opencode config serializes")
}

/// The file llmman's opencode config is written to for a launch:
/// `llmman-config.json` beside the `model.json` variant selection in
/// opencode's state directory — session-scoped, not the user's global
/// or project config, and already among the directories `--sandbox`
/// lets opencode use (see `sandbox_state`). One path, rewritten each
/// launch, as codex's `llmman-model.json` is: opencode reads its
/// config once at startup, so a later launch rewriting it cannot
/// disturb a running session.
fn opencode_config_path(state_dir: &Path) -> PathBuf {
    state_dir.join("llmman-config.json")
}

/// Writes `config` to [`opencode_config_path`] (creating the
/// directory), returning the path to point `OPENCODE_CONFIG` at. The
/// config holds no API key — see `opencode_inline_config` — so the
/// file needs no tighter permissions than the state directory's own.
fn write_opencode_config_file(state_dir: &Path, config: &str) -> anyhow::Result<PathBuf> {
    let path = opencode_config_path(state_dir);
    write_if_changed(&path, config)?;
    Ok(path)
}

/// The small `OPENCODE_CONFIG_CONTENT` override for a launch: the
/// model to start on, and the provider that serves it — its name, the
/// endpoint and the key — nothing else. The model catalog itself rides
/// in the file [`write_opencode_config_file`] writes, so this stays a
/// few hundred bytes however many models the provider serves.
///
/// Both pieces reach opencode, and it deep-merges its config layers
/// (later layers override earlier ones field by field, providers
/// merged per provider), so the file's `models` survive beside the
/// inline's `options` — and the inline, the highest-precedence layer,
/// keeps a project config from retargeting the launched model or the
/// provider that serves it. The key never lands in the file: it
/// travels in this value alone.
///
/// One thing the split does not preserve: a caller who already had
/// `OPENCODE_CONFIG` set loses that file — the variable now names
/// llmman's file, and opencode loads exactly one custom config path.
/// It used to load below the inline config, so nothing llmman set
/// changes; only the caller's own custom file stops loading.
fn opencode_inline_config(server: &str, model: &str, api_key: &str) -> String {
    serde_json::json!({
        "model": opencode_model_ref(model),
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llmman",
                "options": {
                    "baseURL": format!("{server}/v1"),
                    "apiKey": api_key,
                },
            },
        },
    })
    .to_string()
}

/// opencode's `limit` for a model with a known window: the catalog's own
/// ceiling wherever there is one, derived where it names none.
fn model_limit(context: u64, output: Option<u32>) -> Limit {
    Limit {
        context,
        output: output.map_or_else(|| opencode_output_reserve(context), u64::from),
    }
}

/// opencode's schema requires both fields once `limit` is present.
#[derive(Clone, Copy, serde::Serialize)]
struct Limit {
    context: u64,
    output: u64,
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
            Some("k"),
            &variants,
            None,
            false,
            None,
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(config["$schema"], "https://opencode.ai/config.json");
        assert_eq!(config["model"], "ollama/qwen3.5:0.8b");
        let provider = &config["provider"]["ollama"];
        assert_eq!(provider["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(provider["name"], "llmman");
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

        let bare = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            false,
            None,
            None,
            None,
        );
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
            Some("k"),
            &variants,
            Some(&high.1),
            false,
            None,
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let model = &config["provider"]["ollama"]["models"]["m"];
        assert_eq!(
            model["options"],
            serde_json::json!({ "reasoningEffort": "high" })
        );

        let text = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &variants,
            None,
            false,
            None,
            None,
            None,
        );
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

    /// Each provider row carries its own model's image input, not just
    /// the launched model's: a vision model declares `modalities` and
    /// `attachment` from its own catalog row, a text-only one gets
    /// neither — and the launched model keeps what the launch resolved
    /// for it even where its catalog row says otherwise.
    #[test]
    fn opencode_config_carries_each_models_own_modalities() {
        let hosted = vec![
            HostedModel {
                reference: "llmman.provider/p/alpha".into(),
                id: "alpha".into(),
                context: Some(128_000),
                output: Some(8_192),
                thinking: Some(vec!["low".into(), "high".into()]),
                // The catalog says vision; the launch resolved none.
                vision: Some(true),
            },
            HostedModel {
                reference: "llmman.provider/p/beta".into(),
                id: "beta".into(),
                context: Some(65_536),
                output: None,
                // A level the portable defaults never offer, so the
                // assertions below can only pass if the generator
                // reads beta's own published levels.
                thinking: Some(vec!["xhigh".into()]),
                vision: Some(true),
            },
            HostedModel {
                reference: "llmman.provider/p/gamma".into(),
                id: "gamma".into(),
                context: None,
                output: None,
                thinking: None,
                // The catalog is silent: no image input.
                vision: None,
            },
        ];
        let variants = opencode_variants(None);
        let text = opencode_config(
            "http://h",
            "llmman.provider/p/alpha",
            Some("k"),
            &variants,
            None,
            false,
            Some(128_000),
            Some(8_192),
            Some(&hosted),
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let models = &config["provider"]["ollama"]["models"];

        // The launched model keeps what the launch resolved: no image
        // input, even though its catalog row says vision.
        let alpha = &models["llmman.provider/p/alpha"];
        assert_eq!(alpha["name"], "alpha");
        assert!(alpha.get("modalities").is_none(), "{alpha}");
        assert!(alpha.get("attachment").is_none(), "{alpha}");

        // beta declares image input from its own catalog row, with its
        // own thinking levels and window beside it — exactly the one
        // level it publishes, and no portable default.
        let beta = &models["llmman.provider/p/beta"];
        assert_eq!(beta["name"], "beta");
        assert_eq!(
            beta["modalities"],
            serde_json::json!({ "input": ["text", "image"], "output": ["text"] })
        );
        assert_eq!(beta["attachment"], true);
        assert_eq!(beta["limit"]["context"], 65_536);
        assert_eq!(beta["limit"]["output"], 16_384);
        assert_eq!(
            beta["variants"],
            serde_json::json!({ "xhigh": { "reasoningEffort": "xhigh" } })
        );

        // gamma: the catalog is silent, so neither key appears.
        let gamma = &models["llmman.provider/p/gamma"];
        assert!(gamma.get("modalities").is_none(), "{gamma}");
        assert!(gamma.get("attachment").is_none(), "{gamma}");
        assert!(gamma.get("limit").is_none(), "{gamma}");

        // The rows stay in the shared list's order, launched first:
        // match the map keys, not the top-level `model` field, which
        // also names the launched model.
        let positions: Vec<usize> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|id| text.find(&format!("\"llmman.provider/p/{id}\":")).unwrap())
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    }

    /// The file copy of the config carries no API key: the key travels
    /// in the small inline override alone (see `opencode_inline_config`),
    /// so it never lands in a file.
    #[test]
    fn opencode_config_without_a_key_omits_it() {
        let variants = opencode_variants(None);
        let text = opencode_config(
            "http://h",
            "m",
            None,
            &variants,
            None,
            false,
            Some(8_192),
            None,
            None,
        );
        assert!(!text.contains("apiKey"), "{text}");
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let provider = &config["provider"]["ollama"];
        assert_eq!(provider["options"]["baseURL"], "http://h/v1");
        assert_eq!(provider["models"]["m"]["limit"]["context"], 8_192);
    }

    /// The inline override names the launch — the model, the provider,
    /// the endpoint and the key — and nothing of the catalog: its size
    /// does not grow with the provider's model list.
    #[test]
    fn opencode_inline_config_carries_the_launch_not_the_catalog() {
        let inline = opencode_inline_config("http://h", "m", "k");
        let parsed: serde_json::Value = serde_json::from_str(&inline).expect("valid JSON");
        assert_eq!(
            parsed,
            serde_json::json!({
                "model": "ollama/m",
                "provider": {
                    "ollama": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "llmman",
                        "options": {
                            "baseURL": "http://h/v1",
                            "apiKey": "k",
                        },
                    },
                },
            })
        );
    }

    /// A provider with 1,000 models and four variants each used to ride
    /// in one `OPENCODE_CONFIG_CONTENT` value — over the 128 KiB a
    /// single environment string may hold, so the kernel refused the
    /// launch (E2BIG) before opencode started. The file now holds that
    /// catalog, and the environment carries only the file's path and
    /// the small inline override.
    #[test]
    fn opencode_handoff_stays_bounded_for_a_large_catalog() {
        let hosted: Vec<HostedModel> = (0..1000)
            .map(|i| HostedModel {
                reference: format!("llmman.provider/p/model-{i:04}"),
                id: format!("model-{i:04}"),
                context: Some(128_000),
                output: Some(8_192),
                thinking: Some(vec![
                    "low".into(),
                    "medium".into(),
                    "high".into(),
                    "xhigh".into(),
                ]),
                vision: Some(true),
            })
            .collect();
        let variants = opencode_variants(None);
        let file = opencode_config(
            "http://h",
            "llmman.provider/p/model-0000",
            None,
            &variants,
            None,
            false,
            Some(128_000),
            Some(8_192),
            Some(&hosted),
        );
        // The old handoff would have put this in one environment
        // value; it no longer fits in one.
        assert!(file.len() > 128 * 1024, "file is {} bytes", file.len());
        assert!(!file.contains("apiKey"), "the key never lands in the file");
        let parsed: serde_json::Value = serde_json::from_str(&file).expect("valid JSON");
        assert_eq!(
            parsed["provider"]["ollama"]["models"]
                .as_object()
                .map(|m| m.len()),
            Some(1000)
        );

        // The new handoff's environment values: a path, and this.
        let inline = opencode_inline_config("http://h", "llmman.provider/p/model-0000", "k");
        assert!(inline.len() < 1024, "inline is {} bytes", inline.len());
        assert!(!inline.contains("model-0001"), "{inline}");
    }

    /// The config file is written beside the variant selection, in
    /// opencode's state directory, and a second write of the same
    /// config leaves it alone.
    #[test]
    fn write_opencode_config_file_writes_beside_the_variant_selection() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-opencode-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let state = dir.join("state").join("opencode");
        let path = write_opencode_config_file(&state, r#"{"model":"ollama/m"}"#).unwrap();
        assert_eq!(path, state.join("llmman-config.json"));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            r#"{"model":"ollama/m"}"#
        );
        write_opencode_config_file(&state, r#"{"model":"ollama/m"}"#).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            r#"{"model":"ollama/m"}"#
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A local launch's single row still declares image input from
    /// the launch, as before — the per-model change is only that a
    /// hosted model no longer has to be the launched one to get it.
    #[test]
    fn opencode_config_declares_image_input_only_for_a_vision_model() {
        let text = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            true,
            None,
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let model = &config["provider"]["ollama"]["models"]["m"];
        assert_eq!(
            model["modalities"],
            serde_json::json!({ "input": ["text", "image"], "output": ["text"] })
        );
        assert_eq!(model["attachment"], true);

        let text_only = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            false,
            None,
            None,
            None,
        );
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
        let text = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            false,
            Some(8192),
            None,
            None,
        );
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let limit = &config["provider"]["ollama"]["models"]["m"]["limit"];
        assert_eq!(limit["context"], 8192);
        // Both keys are required once `limit` is present.
        assert_eq!(limit["output"], 2048);

        // No window: the key stays out and opencode keeps its defaults.
        let unknown = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            false,
            None,
            None,
            None,
        );
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
            Some("k"),
            &[],
            None,
            false,
            Some(128_000),
            Some(8_192),
            None,
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
        let local = opencode_config(
            "http://h",
            "m",
            Some("k"),
            &[],
            None,
            false,
            Some(128_000),
            None,
            None,
        );
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
            Some("k"),
            &[],
            None,
            false,
            None,
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
            default_on: false,
            default_effort: None,
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

    #[test]
    fn added_config_is_appended_and_keeps_the_order_of_the_rest() {
        let base = r#"{"variants":{"none":1,"low":2,"high":3},"model":"a"}"#;
        let added = serde_json::json!({ "instructions": ["y.md"], "plugins": ["/p"] });
        let merged = with_added_config(base, Some(&added));
        assert_eq!(
            merged,
            r#"{"variants":{"none":1,"low":2,"high":3},"model":"a","instructions":["y.md"],"plugins":["/p"]}"#
        );
        assert_eq!(with_added_config(base, None), base);
    }

    #[test]
    fn the_major_version_is_read_from_the_version_output() {
        assert_eq!(major_version("opencode v2.3.0\n"), Some(2));
        assert_eq!(major_version("1.18.29\n"), Some(1));
        assert_eq!(major_version("v10.0.1"), Some(10));
        assert_eq!(major_version(""), None);
        assert_eq!(major_version("opencode dev"), None);
    }
}
