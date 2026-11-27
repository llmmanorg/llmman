//! What a launch resolves before the handoff: the model it starts on, the
//! list the agent is offered, and the credential it is handed.

use std::collections::HashSet;

use crate::daemon::{self, LaunchModels};

use super::models::{self, Latest, ModelEntry, ModelList, Picked, Want};
use super::{
    claude, codex, declares_context_window, hermes, integration_key, keys, local_context_window,
    openclaw, opencode, sandbox, LaunchArgs, Thinking,
};

/// What `launch` hands an integration.
pub(super) struct Plan {
    /// What the agent starts on, as the daemon routes it; empty for a
    /// launch with none (see [`runs_without_model`]).
    pub model: String,
    /// What the agent authenticates to the daemon with (see [`keys`]).
    pub api_key: String,
    /// The starting model's thinking choices.
    pub thinking: Option<Thinking>,
    pub vision: bool,
    /// The window the daemon serves the starting model, or the catalog's;
    /// `None` when no side of it is known.
    pub context_window: Option<u64>,
    /// The catalog's reply ceiling for the starting model.
    pub max_output: Option<u32>,
    /// Every model the agent can switch to, the starting one first. Empty
    /// for a launch with no model.
    pub models: ModelList,
}

/// Whether a launch with no model keeps doing what it did before there was a list: goose-desktop
/// starts for dictation alone when `OPENAI_TRANSCRIPTION_MODEL` names a speech model, and kimi
/// leaves the model to its own configuration (it finds llmman's models through `OLLAMA_HOST`). Kimi
/// is still offered the list but starts on none of it (see [`resolve`]).
pub(super) fn runs_without_model(
    integration: &str,
    provider: Option<&str>,
    speech_model: Option<&str>,
) -> bool {
    if provider.is_some() {
        return false;
    }
    match integration.to_lowercase().as_str() {
        "goose-desktop" => speech_model.is_some_and(|s| !s.trim().is_empty()),
        "kimi" => true,
        _ => false,
    }
}

/// The model an integration last ran, when it keeps one llmman can read.
fn remembered_model(integration: &str) -> Option<String> {
    match integration.to_lowercase().as_str() {
        "claude" => claude::remembered_model(),
        "codex" => codex::remembered_model(),
        "opencode" => opencode::remembered_model(),
        "hermes" => hermes::remembered_model(),
        "openclaw" => openclaw::remembered_model(),
        _ => None,
    }
}

/// The model list for a launch that asked for none, in the one request the daemon answers it in. A
/// daemon that predates `GET /llmman/models` is read for the one provider `--provider` names and no
/// stored model, with a warning; a launch with nothing else to start on (`!can_continue`) fails.
fn fetch_models(
    provider: Option<(&str, Option<&str>)>,
    can_continue: bool,
) -> anyhow::Result<LaunchModels> {
    if let Some(models) = daemon::launch_models()? {
        return Ok(models);
    }
    let stale = daemon::describe_server_to_stop();
    if !can_continue {
        anyhow::bail!("{stale}");
    }
    eprintln!("[llmman] {stale}");
    let providers = match provider {
        Some((id, model)) => vec![daemon::provider(id, model)?],
        None => Vec::new(),
    };
    Ok(LaunchModels {
        local: Vec::new(),
        providers,
    })
}

/// `--provider` names a provider the list lacks, or a model of it the
/// catalog predates: the daemon is asked about that one, which refreshes
/// its catalog and answers an unknown id with the near matches.
fn ensure_provider_listed(
    models: &mut LaunchModels,
    provider: &str,
    model: Option<&str>,
) -> anyhow::Result<()> {
    let listed = models.providers.iter().find(|p| p.id == provider);
    let missing = match (listed, model) {
        (None, _) => true,
        (Some(p), Some(model)) => !p.models.iter().any(|m| m.id == model),
        (Some(_), None) => false,
    };
    if missing {
        let detail = daemon::provider(provider, model)?;
        models.providers.retain(|p| p.id != provider);
        models.providers.push(detail);
    }
    Ok(())
}

/// The model `integration` starts on and everything the launch hands it,
/// for the four ways of launching: `--provider P --model M`, `--provider P`,
/// `--model M` and neither.
pub(super) fn resolve(
    integration: &str,
    args: &LaunchArgs,
    provider: Option<&str>,
    speech_model: Option<&str>,
) -> anyhow::Result<Plan> {
    let model_flag = args
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty());
    if provider.is_some() {
        super::check_provider_supported(integration)?;
    }
    // Fallible: it validates the raw reference first (see shortnames::validate_reference).
    // resolve_ollama_api, not resolve: every integration talks to serve's compat surfaces, which
    // resolve names the same way (see ensure_model in cmd::serve), so a bare name must match what
    // the daemon resolves it to at request time.
    let local = match (provider, model_flag) {
        (None, Some(model)) => Some(crate::shortnames::resolve_ollama_api(model)?),
        _ => None,
    };
    // Preloading the requested model, so the first request finds it warm. Nothing for a
    // provider-routed model, but the daemon has to be running: it is what forwards upstream.
    crate::daemon::ensure_server(local.as_deref().unwrap_or_default())?;

    let own_model = runs_without_model(integration, provider, speech_model) && model_flag.is_none();
    if own_model && !integration.eq_ignore_ascii_case("kimi") {
        return Ok(Plan {
            model: String::new(),
            api_key: integration_key(),
            thinking: None,
            vision: false,
            context_window: None,
            max_output: None,
            models: ModelList::default(),
        });
    }

    let mut catalog = fetch_models(
        provider.map(|p| (p, model_flag)),
        provider.is_some() || local.is_some() || own_model,
    )?;
    if let Some(provider) = provider {
        ensure_provider_listed(&mut catalog, provider, model_flag)?;
    }
    let mut offered = models::offered(&catalog);
    if let Some(why) = super::stored_only_reason(integration) {
        offered = models::stored_only(offered, integration, why);
    }
    // Asked once per provider: the list is ordered by it, row by row.
    let usable_ids: HashSet<&str> = catalog
        .providers
        .iter()
        .filter(|p| keys::usable(integration, p))
        .map(|p| p.id.as_str())
        .collect();
    let usable = |id: &str| usable_ids.contains(id);
    if own_model {
        // Kimi's own configuration chooses the model; the list is only
        // something to switch to, in no starting order of its own.
        let api_key = keys::authenticate(integration, &catalog.providers, None)?;
        return Ok(Plan {
            model: String::new(),
            api_key,
            thinking: None,
            vision: false,
            context_window: None,
            max_output: None,
            models: if sandbox::env_on_command_line() {
                ModelList::default()
            } else {
                models::ordered(None, offered, None, usable)
            },
        });
    }
    let remembered = remembered_model(integration);
    let want = match (provider, model_flag, local.as_deref()) {
        (Some(provider), Some(model), _) => Want::Hosted { provider, model },
        (Some(provider), None, _) => Want::HostedDefault { provider },
        (None, _, Some(local)) => Want::Local(local),
        (None, _, None) => Want::Unspecified {
            remembered: remembered.as_deref(),
            latest: models::latest_stored(&catalog),
        },
    };
    let (mut start, picked) = models::resolve_start(want, &offered, usable)?;
    if let (Some(provider), Some(model)) = (start.provider_id(), model_flag) {
        if let Some(detail) = catalog.providers.iter().find(|p| p.id == provider) {
            detail.warn_unlisted(model);
        }
    }
    if picked != Picked::Asked {
        eprintln!("[llmman] {integration}: {}", picked_notice(&start, picked));
    }

    // What only the loaded side knows of a stored model: its template's thinking, whether it takes
    // images, and the window it is served (read live, which loads the model, and only for an
    // integration that has somewhere to put it).
    let (thinking, vision, window) = match start.provider_id() {
        Some(_) => (
            start.thinking.clone().map(Thinking::Listed),
            start.vision,
            start.context_window,
        ),
        None => {
            // A bare launch's stored model was pulled already; a named one is pulled here, with
            // progress, rather than failing inside the agent. Serve's own preload is
            // fire-and-forget and only on a cold start.
            let (thinking, vision, window) = stored_facts(integration, args, &start.reference)?;
            start.thinking = thinking
                .as_ref()
                .map(|t| t.choices().into_iter().map(String::from).collect());
            start.vision = vision;
            start.context_window = window.or(start.context_window);
            (thinking, vision, start.context_window)
        }
    };

    let api_key = keys::authenticate(integration, &catalog.providers, start.provider_id())?;
    let model = start.reference.clone();
    let max_output = start.max_output;
    let list = models::ordered(Some(start), offered, provider, usable);
    // OpenShell shares no file with the agent and takes its environment on
    // a command line, so a list has no way in.
    let list = if sandbox::env_on_command_line() {
        list.limited(
            integration,
            1,
            "--sandbox openshell shares no file with the agent",
        )
    } else {
        list
    };
    Ok(Plan {
        model,
        api_key,
        thinking,
        vision,
        context_window: window,
        max_output,
        models: list,
    })
}

/// What only the loaded side of a stored model knows: its template's thinking controls, whether it
/// takes images, and the window it is served (read live, which loads the model, and only for an
/// integration with somewhere to put it). The model is pulled first, with progress, if it is not.
fn stored_facts(
    integration: &str,
    args: &LaunchArgs,
    reference: &str,
) -> anyhow::Result<(Option<Thinking>, bool, Option<u64>)> {
    let info = daemon::ensure_model_pulled(reference)?;
    let window = (declares_context_window(integration)
        && (!integration.eq_ignore_ascii_case("claude")
            || claude::runs_model(reference, &args.extra_args)))
    .then(|| local_context_window(reference, info.context_length()))
    .flatten();
    Ok((
        info.thinking_controls().map(Thinking::Template),
        info.vision(),
        window,
    ))
}

/// The line saying which model a launch that named none starts on.
fn picked_notice(start: &ModelEntry, picked: Picked) -> String {
    let what = start.label();
    match picked {
        Picked::Remembered => format!("starting on {what}, the model it last ran"),
        Picked::Stored(Latest::Loaded) => {
            format!("no --model: starting on {what}, the model loaded now")
        }
        Picked::Stored(Latest::Pulled) => {
            format!("no --model: starting on {what}, the model pulled most recently")
        }
        Picked::Stored(Latest::FirstByName) => {
            format!("no --model: starting on {what}, the first model pulled here by name")
        }
        Picked::Provider => {
            let name = start
                .provider
                .as_ref()
                .map_or("a provider", |(_, n)| n.as_str());
            format!("no --model: starting on {what}, the default model of {name}, which has a key")
        }
        Picked::ProviderDefault => {
            let name = start
                .provider
                .as_ref()
                .map_or("the provider", |(_, n)| n.as_str());
            format!("no --model: starting on {what}, the default model of {name}")
        }
        Picked::Asked => format!("starting on {what}"),
    }
}

/// A launch for `--model` paired with `--overflow-provider`/`--overflow-model`:
/// the one model the daemon routes between, and nothing to switch to (see
/// [`crate::hybrid`]).
pub(super) fn pair(
    integration: &str,
    args: &LaunchArgs,
    (overflow_provider, hosted): (&str, &str),
) -> anyhow::Result<Plan> {
    let model = args
        .model
        .as_deref()
        .map(crate::shortnames::resolve_ollama_api)
        .transpose()?
        .unwrap_or_default();
    crate::daemon::ensure_server(&model)?;
    let (thinking, vision, local_window) = stored_facts(integration, args, &model)?;
    super::check_provider_supported(integration)?;
    // The local half, which serves by default, keeps its thinking choices.
    let remote = super::resolve_provider_model(overflow_provider, hosted, integration)?;
    let reference = crate::hybrid::pair_with_local(&model, &remote.reference)?;
    let window = super::pair_context_window(local_window, remote.context_window);
    // The one row of the list, which an integration reads its window, image
    // input, thinking choices and reply ceiling from.
    let entry = ModelEntry {
        context_window: window,
        vision,
        thinking: thinking
            .as_ref()
            .map(|t| t.choices().into_iter().map(String::from).collect()),
        max_output: remote.max_output,
        ..ModelEntry::unlisted(&reference)
    };
    Ok(Plan {
        models: ModelList::of(vec![entry]),
        model: reference,
        api_key: remote.api_key,
        thinking,
        vision,
        context_window: window,
        // The pair's ceiling is the hosted half's: a local backend caps
        // no reply of its own.
        max_output: remote.max_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_goose_desktop_dictation_and_kimi_run_without_a_model() {
        let speech = Some("canary");
        assert!(runs_without_model("goose-desktop", None, speech));
        assert!(!runs_without_model("goose-desktop", None, None));
        assert!(!runs_without_model("goose-desktop", None, Some(" ")));
        assert!(!runs_without_model(
            "goose-desktop",
            Some("openrouter"),
            speech
        ));
        assert!(runs_without_model("kimi", None, None));
        assert!(!runs_without_model("kimi", Some("openrouter"), None));
        for id in [
            "claude", "opencode", "codex", "hermes", "openclaw", "qwen", "pi",
        ] {
            assert!(!runs_without_model(id, None, speech), "{id}");
        }
    }

    #[test]
    fn the_notice_says_why_a_model_was_picked() {
        let stored = ModelEntry::unlisted("qwen3.5:0.8b");
        assert!(picked_notice(&stored, Picked::Stored(Latest::Loaded)).contains("loaded now"));
        assert!(
            picked_notice(&stored, Picked::Stored(Latest::Pulled)).contains("pulled most recently")
        );
        assert!(picked_notice(&stored, Picked::Stored(Latest::FirstByName))
            .contains("first model pulled here by name"));
        assert!(picked_notice(&stored, Picked::Remembered).contains("last ran"));
        let hosted = ModelEntry::unlisted(&crate::providers::format_remote_ref("openai", "gpt-5"));
        let notice = picked_notice(&hosted, Picked::Provider);
        assert!(
            notice.contains("openai/gpt-5") && notice.contains("which has a key"),
            "{notice}"
        );
    }
}
