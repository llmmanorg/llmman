//! `llmman launch` — launch AI agent integrations backed by llmman serve.
//!
//! Mirrors `ollama launch`: sets integration-specific environment variables
//! pointing at the local inference server, then exec's the integration binary.
//!
//! `--provider` extends that to models llmman does not serve itself, from
//! the same models.dev catalog opencode resolves its providers from (see
//! [`crate::providers`]). It does not change the shape above: the
//! integration is still pointed at `llmman serve`, which forwards upstream
//! on its behalf. There is deliberately no path here that hands an
//! integration a provider's URL directly — one endpoint, one place
//! integrations are configured, whether or not the weights are local.
//!
//! `--overflow-provider`/`--overflow-model` hand the integration one
//! reference naming the local `--model` and a hosted one, and the daemon
//! picks a side per request (see [`crate::hybrid`]); the integration
//! never learns two are involved.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use clap::Args;

use crate::chat_template::{ThinkingControls, EFFORT_LEVELS};
use crate::daemon;
use crate::providers;

mod agy;
mod aider;
mod claude;
mod cline;
mod codex;
mod common;
mod copilot;
mod docker_agent;
mod droid;
mod dsh;
mod gemini;
mod goose;
mod goose_desktop;
mod grok;
mod hermes;
mod muse;
mod omp;
mod openclaw;
mod opencode;
mod pi;
mod qwen;
mod sandbox;

pub use sandbox::Sandbox;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug, Clone)]
pub struct LaunchArgs {
    /// Integration to launch (claude, opencode, codex, cline, aider, …)
    /// Omit to list available integrations.
    #[arg(value_name = "INTEGRATION")]
    pub integration: Option<String>,

    /// Model to use
    #[arg(long, short, value_name = "MODEL")]
    pub model: Option<String>,

    /// Serve --model from this provider (openai, anthropic, openrouter, …)
    /// instead of locally. Requires --model. See `llmman providers`.
    #[arg(long, short = 'p', value_name = "PROVIDER")]
    pub provider: Option<String>,

    /// Send requests too large for the local --model to this provider
    /// instead (openai, anthropic, openrouter, ...). Needs
    /// --overflow-model; not combinable with --provider.
    #[arg(long, value_name = "PROVIDER")]
    pub overflow_provider: Option<String>,

    /// The --overflow-provider model that serves requests too large for
    /// the local --model. Everything that fits stays on this machine.
    #[arg(long, value_name = "MODEL")]
    pub overflow_model: Option<String>,

    /// Run the integration in a sandbox. See docs/sandbox.md.
    #[arg(long, value_enum, value_name = "SANDBOX")]
    pub sandbox: Option<Sandbox>,

    /// Start at this variant of the model (see `llmman show`), as
    /// opencode's `run --variant`.
    #[arg(long, value_name = "VARIANT", value_parser = super::run::variant_parser())]
    pub variant: Option<String>,

    /// Extra arguments forwarded to the integration binary (after --)
    #[arg(last = true, value_name = "ARGS")]
    pub extra_args: Vec<String>,
}

pub fn run(args: &LaunchArgs) -> anyhow::Result<()> {
    let configured;
    let args = match args.integration.as_deref() {
        Some(name) => {
            let (with_defaults, applied) =
                apply_defaults(args, name, || crate::config::launch_defaults(name))?;
            if !applied.is_empty() {
                // So a refusal of a flag that was not typed makes sense.
                eprintln!(
                    "[llmman] {name}: llmman.conf supplies {}",
                    applied.join(" ")
                );
            }
            configured = with_defaults;
            &configured
        }
        None => args,
    };
    let provider = providers::provider_flag(args.provider.as_deref())?;
    let overflow = crate::hybrid::overflow_flags(
        args.overflow_provider.as_deref(),
        args.overflow_model.as_deref(),
        provider,
    )?;

    let Some(ref name) = args.integration else {
        print_integrations();
        return Ok(());
    };

    // Before the daemon starts. A forwarded --model would get the
    // variant checked against this one.
    let variant = args.variant.as_deref();
    if let Some(variant) = variant {
        anyhow::ensure!(
            args.sandbox != Some(Sandbox::Sbx),
            "--variant does not work with --sandbox sbx"
        );
        anyhow::ensure!(
            !has_forwarded_model_flag(name, &args.extra_args),
            "--variant needs the model as llmman's --model, not after --"
        );
        spell_variant(name, variant)?;
    }

    // sbx serves the model itself and validates its own flags, so none
    // of what follows applies.
    if args.sandbox == Some(Sandbox::Sbx) {
        return sandbox::run_sbx(
            name,
            args.model.as_deref(),
            provider,
            overflow,
            &args.extra_args,
        );
    }

    // Before either arm starts the daemon; see `check_model_flag`.
    check_model_flag(name, args.model.as_deref(), provider, &args.extra_args)?;
    if name.eq_ignore_ascii_case("copilot")
        || name.eq_ignore_ascii_case("copilot-cli")
        || name.eq_ignore_ascii_case("droid")
    {
        common::check_daemon_key_transport(name)?;
    }
    anyhow::ensure!(
        overflow.is_none() || args.model.as_deref().is_some_and(|m| !m.trim().is_empty()),
        "--overflow-model needs --model naming the local model to pair it with"
    );
    // Before the Cline install check, which an image-based sandbox makes
    // moot (see `find_on_path`), and before the daemon starts.
    if let Some(kind) = args.sandbox {
        let id = name.to_lowercase();
        let carries_key =
            provider.is_some() || overflow.is_some() || crate::auth::client_key().is_some();
        // A --variant goes into opencode's state (`write_opencode_variant`).
        let configured_by_file =
            CONFIGURED_BY_FILE.contains(&id.as_str()) || (id == "opencode" && variant.is_some());
        sandbox::prepare(
            kind,
            &id,
            sandbox_state(&id)?,
            configured_by_file,
            carries_key,
        )?;
    }
    // Cline follows the install-on-demand behavior expected by its launch
    // integration. Do this before starting the daemon or pulling a model.
    if name.eq_ignore_ascii_case("cline") {
        cline::ensure_cline_installed()?;
    }
    // Rejects the arguments after `--` that docker-agent cannot be
    // launched with. Here rather than in the launcher so the refusal
    // comes before the daemon starts and a model is pulled.
    if name.eq_ignore_ascii_case("docker-agent") {
        docker_agent::check_docker_agent_args(&args.extra_args)?;
    }
    // Each launcher looks for its program only once it runs, which is
    // after the daemon has started and the model has been pulled: a
    // missing agent would otherwise cost a download first.
    check_installed(name)?;
    // The model's thinking choices (see `opencode_variants`), from its
    // template or the catalog; whether it takes images (see
    // `dsh::write_dsh_settings`) and the trained context only a local model
    // has, which `local_context_window` falls back on.
    let mut thinking = None;
    let mut vision = false;
    let mut context_length = None;
    // The window the daemon serves, for the integrations that declare
    // one: the local model's, the hosted model's, or a pair's larger
    // half, set by whichever arm below resolves the model. `None` when
    // no side of it is known. See `launch`.
    let context_window: Option<u64>;
    // The catalog's reply ceiling, for the arms that have a catalog.
    let mut max_output = None;
    let (model, api_key) = match provider {
        Some(provider) => {
            check_provider_supported(name)?;
            // The daemon first, before --provider is validated: the
            // catalog belongs to `llmman serve` (see cmd::providers), so
            // there is nothing to validate against until it runs. Nothing
            // to preload either — a provider-routed model has nothing
            // local to warm up — but it still has to be running, since it
            // is what forwards upstream.
            crate::daemon::ensure_server("")?;
            let per_request = !PROVIDER_NEEDS_DAEMON_KEY.contains(&name.to_lowercase().as_str());
            let hosted =
                resolve_provider_model(provider, args.model.as_deref(), name, per_request)?;
            thinking = hosted.thinking.map(Thinking::Listed);
            // Nothing local is loaded here, so the catalog's limits are
            // the only ones, and the ones the provider enforces.
            context_window = hosted.context_window;
            max_output = hosted.max_output;
            (hosted.reference, hosted.api_key)
        }
        None => {
            // resolve_ollama_api, not resolve: every integration this
            // launches talks to serve's Ollama/OpenAI/Anthropic-compat
            // surfaces, all of which resolve model names the same way
            // (see ensure_model in cmd::serve), so a bare name here must
            // match what the daemon resolves it to at request time.
            // Fallible: it validates the raw reference first (see
            // shortnames::validate_reference).
            let model = args
                .model
                .as_deref()
                .map(crate::shortnames::resolve_ollama_api)
                .transpose()?
                .unwrap_or_default();

            // Ensure serve is running (start it in background if needed),
            // preloading the requested model so the integration's first
            // request finds it warm.
            crate::daemon::ensure_server(&model)?;

            // serve's preload above is fire-and-forget and only fires on
            // a cold `serve` start (see run() in cmd/serve.rs) — if the
            // daemon was already running from a previous invocation, a
            // missing model would otherwise only surface as an opaque
            // failure once the integration made its first request. Mirror
            // `llmman run`'s behavior and pull it here instead,
            // synchronously and with progress, before ever handing off to
            // the integration.
            if !model.is_empty() {
                let info = crate::daemon::ensure_model_pulled(&model)?;
                thinking = info.thinking_controls().map(Thinking::Template);
                vision = info.vision();
                context_length = info.context_length();
            }
            // Resolved once, and only for an integration that has
            // somewhere to put it: reading the live window loads the
            // model (see `local_context_window`), which every other
            // launch would wait for and never use.
            let local_window = (declares_context_window(name) && !model.is_empty())
                .then(|| local_context_window(&model, context_length))
                .flatten();
            match overflow {
                // The hosted half is validated and keyed exactly as a
                // bare --provider model would be, then paired with the
                // local model just pulled.
                Some((provider, hosted)) => {
                    check_provider_supported(name)?;
                    let per_request =
                        !PROVIDER_NEEDS_DAEMON_KEY.contains(&name.to_lowercase().as_str());
                    // The local half, which serves by default, keeps
                    // its thinking choices.
                    let remote = resolve_provider_model(provider, Some(hosted), name, per_request)?;
                    context_window = pair_context_window(local_window, remote.context_window);
                    // The pair's ceiling is the hosted half's: a local
                    // backend caps no reply of its own.
                    max_output = remote.max_output;
                    (
                        crate::hybrid::pair_with_local(&model, &remote.reference)?,
                        remote.api_key,
                    )
                }
                None => {
                    // A `--provider` model is served by someone else, so
                    // this is the only arm whose window is the local one
                    // alone.
                    context_window = local_window;
                    (model, integration_key())
                }
            }
        }
    };

    launch(
        name,
        &model,
        &api_key,
        thinking.as_ref(),
        variant,
        vision,
        context_window,
        max_output,
        &args.extra_args,
    )
}

/// `args` with `[launch.<name>]` from llmman.conf filled in for the flags
/// left out, and those fills as the flags they stand for.
///
/// Typing `--provider`, `--model` or an overflow flag chooses a model, so
/// none of the configured provider, model or variant applies: the
/// provider would route the typed model to the wrong host, and the
/// variant belongs to the old one. `--variant` alone keeps the configured
/// model. A `--model` after `--` wins over the configured one, so the
/// configured variant is dropped too. `load` runs only when needed.
fn apply_defaults(
    args: &LaunchArgs,
    name: &str,
    load: impl FnOnce() -> Result<crate::config::LaunchConf, &'static str>,
) -> anyhow::Result<(LaunchArgs, Vec<String>)> {
    let mut effective = args.clone();
    let mut applied = Vec::new();
    if args.provider.is_some()
        || args.model.is_some()
        || args.overflow_provider.is_some()
        || args.overflow_model.is_some()
    {
        return Ok((effective, applied));
    }
    let mut defaults = load().map_err(|e| anyhow::anyhow!("cannot read launch defaults: {e}"))?;
    if has_forwarded_model_flag(name, &args.extra_args) {
        defaults.variant = None;
    }
    for (slot, configured, flag) in [
        (&mut effective.provider, defaults.provider, "--provider"),
        (&mut effective.model, defaults.model, "--model"),
        (&mut effective.variant, defaults.variant, "--variant"),
    ] {
        if let (None, Some(value)) = (&*slot, configured) {
            applied.push(format!("{flag} {value}"));
            *slot = Some(value);
        }
    }
    Ok((effective, applied))
}

/// What an integration authenticates with when no provider key travels:
/// the daemon's key when this shell has one, else the placeholder that
/// tells serve the header is not a credential.
fn integration_key() -> String {
    crate::auth::client_key().unwrap_or_else(|| providers::PLACEHOLDER_API_KEY.to_string())
}

// ---------------------------------------------------------------------------
// Pre-flight
// ---------------------------------------------------------------------------

/// Integrations that cannot be launched without `--model`. Qwen Code has
/// no notion of a missing model and sends its own built-in default
/// (`qwen3.7-max` in 0.22.3), which the daemon would then try to pull.
/// AGY needs an explicit model for its Gemini routing URL.
/// dsh has no default of its own either — an empty `--model` would
/// otherwise land a literal `"default"` in `agent-default-model.model`,
/// which the first request then tries to resolve as a real model id.
/// goose instead refuses with "Run 'goose configure' first", advice that
/// does not apply to a launch llmman configures through the environment.
/// goose-desktop keeps the same environment, so llmman still owns the
/// provider and endpoint; only the model name would fall back to
/// `goose configure`'s, and the daemon would be asked for that one.
/// Grok Build and Cline have hosted defaults of their own; without an
/// explicit local model they would send those ids to llmman's endpoint
/// instead.
/// docker-agent's "auto" selection takes the first cloud provider with a
/// credential, then a local Docker Model Runner model, so without an
/// explicit model the request never reaches llmman at all.
/// OMP needs a model so the launcher can select the matching entry its
/// Ollama discovery reads from llmman's `/api/tags`.
/// Copilot's BYOK mode also requires an explicit model.
/// Droid needs one to create and select its custom-model entry.
/// Checked before `ensure_server`, so the refusal costs no daemon start.
const MODEL_REQUIRED: &[&str] = &[
    "qwen",
    "dsh",
    "agy",
    "goose",
    "goose-desktop",
    "grok",
    "cline",
    "pi",
    "omp",
    "copilot",
    "copilot-cli",
    "droid",
    "docker-agent",
];

/// Integrations whose launcher yields to a `--model` after `--`, and so
/// warrant the warning below. qwen: `qwen::qwen_args` drops its own `--model`
/// when the caller spelled one. goose: its model is `GOOSE_MODEL` in the
/// environment, which goose's own `--model` documents itself as
/// overriding. grok: its argument builder likewise drops the generated
/// `--model`. Cline receives no generated arguments, so its own `--model`
/// also wins over the provider selected in its settings.
/// OMP's argument builder drops the generated Ollama model when the caller
/// supplies one.
/// Copilot's argument builder likewise yields to its own `--model`.
/// Droid's `exec` argument builder does the same for its custom model id;
/// interactive Droid reads the generated session default instead.
/// claude: `claude::claude_args` likewise drops the generated `--model`.
/// Not docker-agent: its `--model` replaces the whole model entry
/// `docker_agent::docker_agent_document` wrote, `base_url` included, so
/// a forwarded one reaches api.openai.com.
/// `docker_agent::check_docker_agent_args` refuses it rather than
/// letting it "win".
/// Not dsh: `dsh::dsh_args` does not yield, and dsh takes no
/// `--model` flag at all (its model is the one `dsh::write_dsh_settings`
/// records), so telling a dsh user theirs "wins" would be false, and dsh
/// rejects the unknown flag on its own.
/// Not goose-desktop: the `--model` goose documents is `goose run`/
/// `goose session`'s, and the desktop app has no documented equivalent,
/// so promising the caller's wins would be false. Move it here if it
/// turns out to take one.
const MODEL_FLAG_FORWARDED: &[&str] = &[
    "claude",
    "qwen",
    "goose",
    "grok",
    "cline",
    "omp",
    "copilot",
    "copilot-cli",
    "droid",
];

/// Refuses a launch of one of `MODEL_REQUIRED` without a model, under
/// `--provider` too. A second `--model` after `--` is the caller's to
/// win for the integrations in `MODEL_FLAG_FORWARDED`, but `run`
/// resolves the top-level one and, locally, preloads it, so that gets
/// said.
fn check_model_flag(
    integration: &str,
    model: Option<&str>,
    provider: Option<&str>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let name = integration.to_lowercase();
    let model = model.map(str::trim).filter(|m| !m.is_empty());
    let Some(model) = model else {
        if MODEL_REQUIRED.contains(&name.as_str()) {
            let with_provider = provider.map_or(String::new(), |p| format!(" --provider {p}"));
            anyhow::bail!(
                "{name} needs a model: llmman launch {name}{with_provider} --model <model>"
            );
        }
        // No resolved model for a forwarded one to win over.
        return Ok(());
    };
    if has_forwarded_model_flag(&name, extra_args) {
        eprintln!(
            "[llmman] {name}: the --model after -- wins over --model {model}, the one llmman resolved"
        );
    }
    Ok(())
}

/// Whether this integration accepts a forwarded model flag. Copilot and
/// claude spell it only `--model`; the other launchers also accept `-m`.
fn has_forwarded_model_flag(integration: &str, extra_args: &[String]) -> bool {
    let name = integration.to_lowercase();
    let short = (!matches!(name.as_str(), "copilot" | "copilot-cli" | "claude")).then_some("-m");
    MODEL_FLAG_FORWARDED.contains(&name.as_str())
        && has_flag(before_terminator(extra_args), "--model", short)
}

/// `extra_args` up to a forwarded `--`. The launched CLI takes everything
/// after one as arguments, so a prompt there that reads like a flag is a
/// message, not a flag.
fn before_terminator(extra_args: &[String]) -> &[String] {
    let end = extra_args
        .iter()
        .position(|arg| arg == "--")
        .unwrap_or(extra_args.len());
    &extra_args[..end]
}

/// Whether `extra_args` spells `long` or `short`, as a word or `=`-joined.
fn has_flag(extra_args: &[String], long: &str, short: Option<&str>) -> bool {
    extra_args.iter().any(|a| {
        a == long
            || a.starts_with(&format!("{long}="))
            || short.is_some_and(|s| a == s || a.starts_with(&format!("{s}=")))
    })
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// Integrations `--provider` cannot drive, and why.
///
/// `launch_simple` only exports `OLLAMA_HOST`: it never passes a model,
/// so Kimi picks its own and the provider-routed reference never reaches
/// the daemon.
/// Refusing is the same call the catalog filter makes — a
/// combination llmman cannot actually drive is absent, not offered and
/// then broken at the first request.
const PROVIDER_UNSUPPORTED: &[(&str, &str)] = &[
    (
        "kimi",
        "it selects its own model rather than taking one from llmman",
    ),
    // Its key variable feeds a native Google client, and llmman has not
    // verified that GEMINI_BASE_URL still redirects it here. Getting that
    // wrong sends someone's OpenRouter key to Google, which is a worse
    // outcome than `--provider gemini` not working — the placeholder this
    // used to pass was harmless either way, a real key is not.
    (
        "gemini",
        "llmman cannot confirm it would send the key here rather than to Google",
    ),
    // openclaw::launch_openclaw passes --custom-model-id only through
    // onboarding, which runs once. Every later launch reuses whatever
    // openclaw.json already names, so the provider reference would never
    // reach the daemon and the session would quietly run on the old model.
    (
        "openclaw",
        "it only takes a model during first-run onboarding",
    ),
    // Grok Build refuses `-m` values absent from the custom endpoint's
    // `/v1/models` catalog. llmman's endpoint lists stored local models,
    // not the synthetic provider or hybrid routing refs, so one of those
    // launches would fail before making its first request.
    (
        "grok",
        "its model catalog cannot represent llmman's provider or hybrid routing reference",
    ),
    // OMP discovers the Ollama catalog from /api/tags. That endpoint lists
    // stored models, not the synthetic provider or hybrid routing refs the
    // daemon accepts at request time, and OMP rejects a model absent from the
    // catalog before it sends a request.
    (
        "omp",
        "its Ollama catalog cannot represent llmman's provider or hybrid routing reference",
    ),
];

/// Integrations llmman configures through a file on disk. They take a
/// model on every launch, so `--provider` works, but they cannot carry
/// the key: writing a real one into `~/.hermes/config.yaml` would persist
/// a credential, which this feature promises not to do. They rely on
/// `llmman serve` having the variable itself — which it only uses for a
/// daemon nobody else can reach (see `reachable_only_locally`).
const PROVIDER_NEEDS_DAEMON_KEY: &[&str] = &["hermes", "cline", "pi"];

/// Integrations that declare a context window, and so are worth
/// loading the model to read the real one back for. Keep in step with
/// [`launch`]'s own dispatch: an integration missing here is simply
/// told no window, not told a wrong one.
fn declares_context_window(integration: &str) -> bool {
    matches!(
        integration.to_lowercase().as_str(),
        "opencode" | "codex" | "pi" | "omp" | "muse"
    )
}

fn check_provider_supported(integration: &str) -> anyhow::Result<()> {
    let name = integration.to_lowercase();
    if let Some((_, why)) = PROVIDER_UNSUPPORTED.iter().find(|(id, _)| *id == name) {
        anyhow::bail!(
            "--provider does not work with {name}: {why}\n\
             Run it against a locally served model, or use another integration."
        );
    }
    // The key would go to the integration in cleartext, and from there
    // over plain http to a daemon somewhere else on the network. llmman
    // controls neither hop, so it does not start the handoff. A wildcard
    // bind is fine here — that hop is still loopback — and so is TLS.
    if !crate::daemon::connects_securely() {
        anyhow::bail!(
            "--provider needs a local llmman serve, or one over TLS: LLMMAN_HOST points at {}, \
             and the provider key would cross the network in cleartext.\n\
             Export the key where that daemon runs instead.",
            crate::daemon::server()
        );
    }
    // These reach the daemon over loopback, so the check above passes,
    // but they send the placeholder key and the daemon will not fall back
    // to its own on a bind anyone can reach — unless it authenticates
    // callers, in which case they send its key and it will. Say so here
    // rather than let it surface as a 401 from inside the integration.
    if PROVIDER_NEEDS_DAEMON_KEY.contains(&name.as_str())
        && !crate::daemon::reachable_only_locally()
        && crate::auth::client_key().is_none()
    {
        anyhow::bail!(
            "--provider does not work with {name} while llmman serve is bound to {}: \
             {name} is configured through a file, so it cannot send the key per request, \
             and a daemon reachable from the network will not spend its own.\n\
             Bind llmman serve to loopback, or use an integration that carries the key.",
            crate::daemon::bind_addr()
        );
    }
    Ok(())
}

/// What [`resolve_provider_model`] resolves. Named fields rather than a
/// tuple: three of these are `Option`s, which a caller can drop by
/// destructuring past them without the compiler saying so.
struct ResolvedProvider {
    /// The hosted model's full reference, as the daemon routes on it.
    reference: String,
    /// The key the integration authenticates with.
    api_key: String,
    /// The thinking levels the catalog lists (see [`Thinking::Listed`]).
    thinking: Option<Vec<String>>,
    /// The window the catalog says this model holds.
    context_window: Option<u64>,
    /// The most the catalog says it will emit in one reply (see
    /// [`crate::daemon::ProviderDetail::max_output`]).
    max_output: Option<u32>,
}

/// Validates `--provider`/`--model` against the running daemon's catalog
/// (see [`crate::daemon::provider`]), resolving the reference the daemon
/// routes on, the key `integration` authenticates with, and what the
/// catalog says about the model — see [`ResolvedProvider`].
///
/// `key_travels_per_request` is false for the integrations in
/// [`PROVIDER_NEEDS_DAEMON_KEY`], which get the placeholder because they
/// cannot carry a real key — so this shell having one is beside the
/// point, and demanding it would reject a perfectly good daemon that has
/// it while this shell does not.
///
/// Every check here is one the daemon would otherwise make at first
/// request, by which point the integration has already taken over the
/// terminal and reports whatever it makes of an HTTP error. Failing in
/// llmman's own output, before the handoff, is the difference between a
/// named missing environment variable and an opaque "connection error"
/// inside someone else's TUI.
fn resolve_provider_model(
    provider: &str,
    model: Option<&str>,
    integration: &str,
    key_travels_per_request: bool,
) -> anyhow::Result<ResolvedProvider> {
    // Asked of the daemon, not models.dev: it routes the request, so it
    // is the authority on whether this provider exists — and on whether
    // *it* has the key, which this shell cannot see.
    let model = model.map(str::trim).filter(|m| !m.is_empty());
    // Sent so a catalog that predates the model is refreshed, not warned about.
    let entry = daemon::provider(provider, model)?;

    let model = model.ok_or_else(|| {
        anyhow::anyhow!(
            "--provider {provider} also needs --model\n\n{}",
            providers::example_models(&entry.name, &entry.model_ids())
        )
    })?;

    entry.warn_unlisted(model);

    // Read here, not left to the daemon, so a missing key names the
    // variable to set in llmman's own output. It travels per request in
    // the integration's own Authorization header (see client_api_key in
    // cmd::serve), never to disk or a command line. The placeholder goes
    // instead whenever the daemon's key is the one that matters — an
    // integration that cannot carry one, or a shell without one where
    // the daemon has it — or when the provider takes none at all
    // (`key_optional`): it is what tells serve the header is not a
    // credential.
    //
    // A daemon requiring a key takes that header for it, so the provider
    // key cannot travel: the daemon's is the only one, and an
    // authenticated caller may spend it.
    let daemon_authenticates = crate::auth::client_key().is_some();
    let key = if key_travels_per_request && !daemon_authenticates {
        entry.client_key()
    } else {
        None
    };
    let key = match (key, key_travels_per_request) {
        (Some(key), true) => key,
        (_, false) => {
            // Fatal, not a warning: this integration cannot carry a key,
            // so the daemon's is the only one its first request can use,
            // and `key_usable` is the daemon's own word on whether it
            // would spend it. Warning and handing off would surface as a
            // 401 inside someone else's TUI.
            anyhow::ensure!(
                entry.key_usable || entry.key_optional,
                "{integration} is configured through a file, so it cannot send an API key: \
                 llmman serve needs a key of its own, and must be bound to loopback (or \
                 require an API key) to spend it.\n\
                 Where the daemon runs, {}, then restart it.",
                entry.key_hint()
            );
            integration_key()
        }
        (None, true) if entry.daemon_key_usable() => {
            if !daemon_authenticates {
                eprintln!(
                    "[llmman] warning: no API key for {} here; using the key llmman serve has",
                    entry.name
                );
            }
            integration_key()
        }
        (None, true) if entry.key_optional => integration_key(),
        (None, true) if daemon_authenticates => anyhow::bail!(
            "llmman serve requires an API key, so {integration} sends that one and cannot \
             also carry a key for {}: llmman serve needs a key of its own.\n\
             Where the daemon runs, {}, then restart it.",
            entry.name,
            entry.key_hint()
        ),
        (None, true) => anyhow::bail!("no API key for {} — {}", entry.name, entry.key_hint()),
    };

    Ok(ResolvedProvider {
        reference: providers::format_remote_ref(provider, model),
        api_key: key,
        thinking: entry.thinking_levels(model),
        context_window: entry.context_window(model),
        max_output: entry.max_output(model),
    })
}

// ---------------------------------------------------------------------------
// Integration registry
// ---------------------------------------------------------------------------

struct Integration {
    name: &'static str,
    description: &'static str,
    binary: &'static str,
}

const INTEGRATIONS: &[Integration] = &[
    Integration {
        name: "muse",
        description: "Meta Muse Code",
        binary: "muse",
    },
    Integration {
        name: "claude",
        description: "Claude Code",
        binary: "claude",
    },
    Integration {
        name: "opencode",
        description: "OpenCode",
        binary: "opencode",
    },
    Integration {
        name: "codex",
        description: "OpenAI Codex CLI",
        binary: "codex",
    },
    Integration {
        name: "pi",
        description: "Pi coding agent",
        binary: "pi",
    },
    Integration {
        name: "omp",
        description: "OMP coding agent",
        binary: "omp",
    },
    Integration {
        name: "cline",
        description: "Cline",
        binary: "cline",
    },
    Integration {
        name: "aider",
        description: "Aider AI pair programmer",
        binary: "aider",
    },
    Integration {
        name: "copilot",
        description: "GitHub Copilot CLI",
        binary: "copilot",
    },
    Integration {
        name: "droid",
        description: "Factory Droid CLI",
        binary: "droid",
    },
    Integration {
        name: "kimi",
        description: "Kimi Code CLI",
        binary: "kimi",
    },
    Integration {
        name: "gemini",
        description: "Gemini CLI",
        binary: "gemini",
    },
    Integration {
        name: "agy",
        description: "Google Antigravity CLI",
        binary: "agy",
    },
    Integration {
        name: "hermes",
        description: "Hermes Agent",
        binary: "hermes",
    },
    Integration {
        name: "openclaw",
        description: "OpenClaw",
        binary: "openclaw",
    },
    Integration {
        name: "qwen",
        description: "Qwen Code",
        binary: "qwen",
    },
    Integration {
        name: "dsh",
        description: "DeepSeek Harness",
        binary: "dsh",
    },
    Integration {
        name: "goose",
        description: "Block goose",
        binary: "goose",
    },
    Integration {
        name: "goose-desktop",
        description: "Block goose Desktop",
        binary: "goose-desktop",
    },
    Integration {
        name: "grok",
        description: "Grok Build",
        binary: "grok",
    },
    Integration {
        name: "docker-agent",
        description: "Docker Agent",
        binary: "docker-agent",
    },
];

/// The registered name for a launch name (any case, or an alias): what
/// `[launch.<name>]` in llmman.conf is matched on.
pub(crate) fn canonical_integration(name: &str) -> Option<&'static str> {
    integration(name).map(|i| i.name)
}

/// The registry entry for a launch name, after resolving supported aliases.
fn integration(name: &str) -> Option<&'static Integration> {
    let name = name.to_ascii_lowercase();
    let canonical = match name.as_str() {
        "copilot-cli" => "copilot",
        other => other,
    };
    INTEGRATIONS.iter().find(|i| i.name == canonical)
}

fn print_integrations() {
    println!("Available integrations:\n");
    for i in INTEGRATIONS {
        // dsh resolves to npx when it isn't installed (see `dsh::find_dsh`),
        // and that launch downloads the package before running it.
        let how = match find_integration_binary(i) {
            Some(bin) if bin.file_stem().is_some_and(|s| s == "npx") => " (via npx)",
            Some(_) => "",
            None => " (not installed)",
        };
        println!("  {:<14} {}{}", i.name, i.description, how);
    }
    println!(
        "\nUsage: llmman launch <integration> [--model <model>] [--provider <provider>] [--sandbox <sandbox>]"
    );
    println!("       llmman providers   (the providers --provider accepts)");
}

/// Extensions to try, in order, when resolving a bare command name on
/// Windows — where, unlike everywhere else, a name on `PATH` almost never
/// exists as a bare file: it's always some extension's worth of shim/
/// executable, and which one varies by how it got installed. `.exe` is a
/// real native binary; `.cmd`/`.bat` is what `npm install -g` always
/// generates for a JS-based CLI's bin entry (every integration this
/// module launches — claude, opencode, codex — is installed exactly that
/// way), alongside a `.ps1` this intentionally skips: unlike `.exe`/
/// `.cmd`/`.bat`, Windows' `CreateProcess` (and so `std::process::Command`
/// under it) can't launch a `.ps1` directly at all without an explicit
/// `powershell -File` wrapper, and every npm install already writes a
/// `.cmd` alongside it, so there's no case where only the `.ps1` exists.
const WINDOWS_PATH_EXTS: &[&str] = &["exe", "cmd", "bat"];

/// Under an image-based `--sandbox` the image's copy runs, found on the
/// image's `PATH`, so this machine need not have one.
fn find_on_path(binary: &str) -> Option<PathBuf> {
    find_on_path_unless(binary, |_| false)
}

/// [`find_on_path`], skipping matches `reject` returns true for and
/// carrying on down `PATH` rather than stopping at the first. No
/// platform tells goose's two binaries apart by name — the desktop zip's
/// is `Goose.exe` on Windows, `goose` elsewhere — so each has to refuse
/// the other, and the one wanted may sit in a later directory.
fn find_on_path_unless(binary: &str, reject: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    if sandbox::runs_from_image() {
        return Some(PathBuf::from(binary));
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if cfg!(windows) {
            for ext in WINDOWS_PATH_EXTS {
                let candidate = dir.join(format!("{binary}.{ext}"));
                if candidate.is_file() && !reject(&candidate) {
                    return Some(candidate);
                }
            }
        } else {
            let candidate = dir.join(binary);
            if candidate.is_file() && !reject(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// `key` as a path, treating an empty value as unset like the tools do.
fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
fn test_temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llmman-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The binary `launch` will run for `i`, so the listing does not report
/// as missing what the launcher would find: `PATH`, then what the
/// launcher knows.
fn find_integration_binary(i: &Integration) -> Option<PathBuf> {
    match i.name {
        "copilot" => copilot::find_copilot(),
        "droid" => droid::find_droid(),
        "opencode" => opencode::find_opencode(),
        "omp" => omp::find_omp(),
        "qwen" => qwen::find_qwen(),
        "dsh" => dsh::find_dsh().map(|(bin, _)| bin),
        "goose" => goose::find_goose(),
        "goose-desktop" => goose_desktop::find_goose_desktop(),
        "grok" => grok::find_grok(),
        "docker-agent" => docker_agent::find_docker_agent(),
        _ => find_on_path(i.binary),
    }
}

/// Fails, before anything is started or pulled, when `name`'s program is
/// not there to run, looked up as its launcher will look for it (so an
/// image-based `--sandbox`, prepared just before, counts as there). Cline
/// installs itself on demand, and an unknown name is left to the
/// launcher's own error.
fn check_installed(name: &str) -> anyhow::Result<()> {
    let Some(i) = integration(name) else {
        return Ok(());
    };
    if i.name == "cline" || find_integration_binary(i).is_some() {
        return Ok(());
    }
    match i.name {
        "dsh" => anyhow::bail!("{}", dsh::DSH_MISSING),
        "docker-agent" => Err(docker_agent::docker_agent_missing()),
        "goose-desktop" => Err(goose_desktop::goose_desktop_missing()),
        _ => anyhow::bail!("{} is not installed", i.binary),
    }
}

/// A yes to an interactive `[y/N]`. Shared by the Cline install
/// prompt and goose-desktop's offer to quit.
fn accepts_prompt(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

/// Integrations that cannot carry a `--variant` to llmman, and why.
const VARIANT_UNSUPPORTED: &[(&str, &str)] = &[
    ("kimi", "it picks its own model"),
    ("openclaw", "it only takes settings at onboarding"),
    (
        "droid",
        "Droid does not support reasoning effort for custom models",
    ),
    ("gemini", "llmman's Gemini API ignores thinkingConfig"),
    ("agy", "llmman's Gemini API ignores thinkingConfig"),
    ("goose", "it sends effort only for OpenAI's models"),
    ("goose-desktop", "it sends effort only for OpenAI's models"),
    ("docker-agent", "it ignores thinking_budget here"),
];

const LOW_TO_MAX: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// Each other integration's word for thinking off, if any, and the
/// levels it takes for an unknown model without clamping (pi, omp and
/// Cline clamp the rest). opencode takes llmman's own variants.
const VARIANT_SPELLINGS: &[(&str, Option<&str>, &[&str])] = &[
    // The Meta provider rejects `none`, despite advertising it in --help.
    ("muse", None, EFFORT_LEVELS),
    ("claude", None, LOW_TO_MAX),
    ("codex", Some("none"), EFFORT_LEVELS),
    ("pi", Some("off"), &["minimal", "low", "medium", "high"]),
    (
        "omp",
        Some("off"),
        &["minimal", "low", "medium", "high", "xhigh"],
    ),
    ("cline", Some("none"), &["low", "medium", "high"]),
    ("aider", Some("none"), EFFORT_LEVELS),
    ("copilot", None, LOW_TO_MAX),
    ("copilot-cli", None, LOW_TO_MAX),
    ("hermes", Some("none"), EFFORT_LEVELS),
    ("qwen", Some("none"), LOW_TO_MAX),
    ("grok", Some("none"), EFFORT_LEVELS),
    ("dsh", Some("off"), EFFORT_LEVELS),
];

/// `variant` as `integration` spells it. `thinking` (a switch-only
/// template's on) goes as `medium`, which llmman serves as thinking on.
fn spell_variant<'a>(integration: &str, variant: &'a str) -> anyhow::Result<&'a str> {
    let name = integration.to_lowercase();
    if name == "opencode" {
        return Ok(variant);
    }
    if let Some((_, why)) = VARIANT_UNSUPPORTED.iter().find(|(id, _)| *id == name) {
        anyhow::bail!("--variant does not work with {name}: {why}");
    }
    let Some(&(_, off, levels)) = VARIANT_SPELLINGS.iter().find(|(id, ..)| *id == name) else {
        return Err(unknown_integration(&name));
    };
    let spelled = match variant {
        "none" => off,
        "thinking" => Some("medium"),
        level => levels.iter().copied().find(|l| *l == level),
    };
    spelled.ok_or_else(|| {
        let takes: Vec<&str> = off
            .map(|_| "none")
            .into_iter()
            .chain(levels.iter().copied())
            .chain(Some("thinking"))
            .collect();
        anyhow::anyhow!(
            "{name} cannot start at --variant {variant}; it takes {}",
            takes.join(", ")
        )
    })
}

/// A `--variant` and the model's other levels, as the integration spells
/// them, for those configured with the list.
struct Effort<'a> {
    default: &'a str,
    levels: Vec<&'a str>,
}

/// Keep Muse subcommands ahead of generated effort options.
fn launch_extra_args(name: &str, effort: Option<&str>, extra_args: &[String]) -> Vec<String> {
    let command_len = usize::from(
        name == "muse"
            && extra_args
                .first()
                .is_some_and(|arg| muse::is_subcommand(arg)),
    );
    let (command, extra_args) = extra_args.split_at(command_len);
    [
        command.to_vec(),
        effort.map_or_else(Vec::new, |e| effort_args(name, e)),
        extra_args.to_vec(),
    ]
    .concat()
}

/// The flags that start `integration` at `effort`; opencode, qwen and
/// dsh take it in their configuration instead.
fn effort_args(integration: &str, effort: &str) -> Vec<String> {
    let flags: &[&str] = match integration {
        "muse" => &["--reasoning-effort"],
        "claude" | "grok" => &["--effort"],
        "copilot" | "copilot-cli" => &["--reasoning-effort"],
        "pi" | "omp" | "cline" => &["--thinking"],
        "hermes" => &["--reasoning"],
        // Else aider drops the effort for a model it does not know.
        "aider" => &["--no-check-model-accepts-settings", "--reasoning-effort"],
        "codex" => return vec!["-c".into(), format!("model_reasoning_effort={effort}")],
        _ => return Vec::new(),
    };
    flags
        .iter()
        .copied()
        .chain(Some(effort))
        .map(String::from)
        .collect()
}

// ---------------------------------------------------------------------------
// Launch dispatcher
// ---------------------------------------------------------------------------

/// `api_key` is what the integration is told to authenticate with:
/// [`providers::PLACEHOLDER_API_KEY`] for a locally-served model, or the
/// real provider key under `--provider`.
///
/// Passing the real one is what makes `--provider` work against a daemon
/// that is *already running* — `ensure_server` reuses one, so a daemon
/// started before the key was exported would otherwise never see it (see
/// `client_api_key` in cmd::serve). Only the launchers that pass a key in
/// the integration's environment can do this; the ones that go through a
/// config file on disk keep the placeholder rather than persist a
/// credential, and need the key in the daemon's own environment.
///
/// `context_window` is the window a request may fill, and the only
/// context figure an integration is told: a local model's loaded one
/// (see [`local_context_window`]), a `--provider` model's catalog one,
/// or a pair's larger half (see [`pair_context_window`]); `None` when
/// none of them is known. codex alone substitutes
/// [`codex::CODEX_FALLBACK_CONTEXT_WINDOW`] for a `None`, because its catalog
/// cannot omit the field.
#[allow(clippy::too_many_arguments)]
fn launch(
    name: &str,
    model: &str,
    api_key: &str,
    thinking: Option<&Thinking>,
    variant: Option<&str>,
    vision: bool,
    context_window: Option<u64>,
    max_output: Option<u32>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let name = name.to_lowercase();
    // Unknown levels are a guess: add the one asked for, don't refuse it.
    let widened = variant
        .filter(|_| thinking.is_none())
        .map(unknown_levels_with);
    let thinking = widened.as_ref().or(thinking);
    if let Some(variant) = variant {
        check_variant(model, thinking, variant)?;
    }
    let effort = variant.map(|v| spell_variant(&name, v)).transpose()?;
    let listed = effort.map(|default| Effort {
        default,
        levels: thinking_choices(thinking)
            .into_iter()
            .filter_map(|c| spell_variant(&name, c).ok())
            .collect(),
    });
    let extra_args = &launch_extra_args(&name, effort, extra_args);
    // pi and omp force thinking off for a model not marked as reasoning;
    // a model with the variant reasons.
    let reasons = variant.is_some()
        || thinking
            .and_then(Thinking::template)
            .is_some_and(|t| t.thinks);
    match name.as_str() {
        "muse" => muse::launch_muse(model, api_key, context_window, max_output, extra_args),
        "claude" => claude::launch_claude(model, api_key, extra_args),
        "opencode" => opencode::launch_opencode(
            model,
            api_key,
            thinking,
            variant,
            vision,
            context_window,
            max_output,
            extra_args,
        ),
        "codex" => codex::launch_codex(model, api_key, vision, context_window, extra_args),
        "pi" => pi::launch_pi(model, reasons, vision, context_window, extra_args),
        "omp" => omp::launch_omp(model, reasons, vision, context_window, extra_args),
        "cline" => cline::launch_cline(model, extra_args),
        "aider" => aider::launch_aider(model, api_key, extra_args),
        "copilot" | "copilot-cli" => copilot::launch_copilot(model, api_key, extra_args),
        "droid" => droid::launch_droid(model, api_key, vision, max_output, extra_args),
        "kimi" => launch_simple("kimi", model, extra_args),
        "gemini" => gemini::launch_gemini(model, api_key, extra_args),
        "agy" => agy::launch_agy(model, api_key, extra_args),
        "hermes" => hermes::launch_hermes(model, vision, extra_args),
        "openclaw" => openclaw::launch_openclaw(model, extra_args),
        "qwen" => qwen::launch_qwen(model, api_key, vision, listed.as_ref(), extra_args),
        "dsh" => dsh::launch_dsh(model, api_key, vision, listed.as_ref(), extra_args),
        "goose" => goose::launch_goose(model, api_key, extra_args),
        "goose-desktop" => goose_desktop::launch_goose_desktop(model, api_key, extra_args),
        "grok" => grok::launch_grok(model, api_key, listed.as_ref(), extra_args),
        "docker-agent" => docker_agent::launch_docker_agent(model, api_key, extra_args),
        other => Err(unknown_integration(other)),
    }
}

/// Refuses a `--variant` the model is known not to have.
fn check_variant(model: &str, thinking: Option<&Thinking>, variant: &str) -> anyhow::Result<()> {
    let choices = thinking_choices(thinking);
    let shown = if model.is_empty() { "the model" } else { model };
    anyhow::ensure!(!choices.is_empty(), "{shown} does not think");
    anyhow::ensure!(
        choices.contains(&variant),
        "{shown} has no variant {variant}; it has {}",
        choices.join(", ")
    );
    Ok(())
}

fn unknown_integration(name: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "unknown integration {name:?}\nRun 'llmman launch' without arguments to list supported integrations."
    )
}

/// Integrations whose launcher writes their configuration to files
/// under the home directory, rather than handing it over in the
/// environment and arguments alone.
const CONFIGURED_BY_FILE: &[&str] = &[
    "agy",
    "cline",
    "codex",
    "docker-agent",
    "dsh",
    "droid",
    "grok",
    "hermes",
    "muse",
    "omp",
    "openclaw",
    "pi",
    "qwen",
];

/// What `name` keeps on this machine, which `--sandbox` lets it write:
/// the directories its launcher writes config into, and its own
/// settings and session state, found the way the integration finds them.
fn sandbox_state(name: &str) -> anyhow::Result<Vec<sandbox::State>> {
    use sandbox::State::{Dir, Files};
    let home = dirs::home_dir().context("no home directory")?;
    let xdg = |var: &str, default: &str| xdg_dir(&home, var, default);
    let config = xdg("XDG_CONFIG_HOME", ".config");
    let data = xdg("XDG_DATA_HOME", ".local/share");
    let state = xdg("XDG_STATE_HOME", ".local/state");
    Ok(match name {
        "muse" => vec![
            Dir(muse::settings_root()?),
            Dir(data.join("muse")),
            Dir(home.join(".muse")),
        ],
        "claude" => match env_dir("CLAUDE_CONFIG_DIR") {
            Some(dir) => vec![Dir(dir)],
            None => vec![Dir(home.join(".claude")), Files(home.join(".claude.json"))],
        },
        "opencode" => vec![
            Dir(config.join("opencode")),
            Dir(data.join("opencode")),
            Dir(opencode::opencode_state_dir()?),
            Dir(xdg("XDG_CACHE_HOME", ".cache").join("opencode")),
        ],
        "codex" => vec![Dir(codex::codex_dir()?)],
        "pi" => vec![Dir(pi::pi_agent_dir()?)],
        "omp" => vec![Dir(omp::omp_agent_dir()?)],
        "cline" => vec![Dir(cline::cline_dir()?)],
        "aider" => vec![Dir(home.join(".aider"))],
        "copilot" | "copilot-cli" => vec![
            Dir(env_dir("COPILOT_HOME").unwrap_or_else(|| home.join(".copilot"))),
            Dir(env_dir("GH_CONFIG_DIR").unwrap_or_else(|| config.join("gh"))),
        ],
        "droid" => vec![Dir(droid::droid_home()?)],
        "kimi" => vec![Dir(home.join(".kimi"))],
        "gemini" => vec![Dir(home.join(".gemini"))],
        "agy" => vec![Dir(agy::agy_settings_dir()?)],
        "hermes" => vec![Dir(hermes::hermes_home()?)],
        // `openclaw::launch_openclaw` takes the legacy config as onboarded too.
        "openclaw" => std::iter::once(home.join(".openclaw"))
            .chain(Some(home.join(".clawdbot")).filter(|d| d.is_dir()))
            .map(Dir)
            .collect(),
        "qwen" => vec![Dir(qwen::qwen_home()?)],
        "dsh" => vec![Dir(dsh::dsh_config_dir()?)],
        // The desktop app drives the same goose, so it reads the same
        // tree, and keeps Electron `userData` beside it under the
        // bundle's name — a separate directory wherever the filesystem
        // tells `Goose` and `goose` apart.
        "goose" | "goose-desktop" => {
            let mut dirs = vec![
                Dir(config.join("goose")),
                Dir(data.join("goose")),
                Dir(state.join("goose")),
            ];
            if name == "goose-desktop" {
                dirs.push(Dir(goose_desktop::goose_desktop_user_data(&home, &config)));
                // macOS keeps the app's preferences outside `userData`,
                // under the bundle id `Goose.app` 1.52.0 declares.
                if cfg!(target_os = "macos") {
                    dirs.push(Files(
                        home.join("Library")
                            .join("Preferences")
                            .join("com.electron.goose.plist"),
                    ));
                }
            }
            dirs
        }
        "grok" => vec![Dir(grok::grok_home()?)],
        "docker-agent" => vec![
            Dir(docker_agent::docker_agent_config_dir()?),
            Dir(home.join(".cagent")),
        ],
        other => return Err(unknown_integration(other)),
    })
}

// ---------------------------------------------------------------------------
// Per-integration launchers
// ---------------------------------------------------------------------------

/// `%USERPROFILE%` on Windows, where node's `os.homedir()` reads it and
/// `dirs::home_dir` does not. `None` elsewhere, where `os.homedir()`
/// reads `$HOME` and `dirs` agrees.
fn node_user_profile() -> Option<PathBuf> {
    cfg!(windows).then(|| env_dir("USERPROFILE")).flatten()
}

/// `$var`, else `default` under `home`.
fn xdg_dir(home: &Path, var: &str, default: &str) -> PathBuf {
    env_dir(var).unwrap_or_else(|| home.join(default))
}

/// The thinking choices a model offers an integration, in cycle order.
enum Thinking {
    /// A local model's, read off its chat template.
    Template(ThinkingControls),
    /// A provider's model's, from the catalog (see
    /// `crate::providers::Model::thinking`).
    Listed(Vec<String>),
}

impl Thinking {
    fn choices(&self) -> Vec<&str> {
        match self {
            Thinking::Template(controls) => controls.choices(),
            Thinking::Listed(levels) => levels.iter().map(String::as_str).collect(),
        }
    }

    fn template(&self) -> Option<&ThinkingControls> {
        match self {
            Thinking::Template(controls) => Some(controls),
            Thinking::Listed(_) => None,
        }
    }
}

/// The choices offered when neither a template nor the catalog says:
/// thinking off, then the levels every wire accepts
/// (`anthropic::portable_efforts`).
const PORTABLE_THINKING_LEVELS: &[&str] = &["none", "low", "medium", "high"];

/// The model's variants in cycle order, [`PORTABLE_THINKING_LEVELS`] if
/// unknown.
fn thinking_choices(thinking: Option<&Thinking>) -> Vec<&str> {
    match thinking {
        Some(thinking) => thinking.choices(),
        None => PORTABLE_THINKING_LEVELS.to_vec(),
    }
}

/// The guessed levels plus `variant`, for a model whose levels are
/// unknown: a provider can serve a level before models.dev lists it, and
/// an integration must find its starting level among those it is given.
/// `thinking` is a template switch, not a level, so it stays refused.
fn unknown_levels_with(variant: &str) -> Thinking {
    let levels = std::iter::once("none")
        .chain(EFFORT_LEVELS.iter().copied())
        .filter(|l| PORTABLE_THINKING_LEVELS.contains(l) || *l == variant)
        .map(String::from)
        .collect();
    Thinking::Listed(levels)
}

/// The window the daemon serves for `model`, read back from the loaded
/// runner rather than predicted: an OOM retry halves `--ctx-size` during
/// the load, and a reused daemon keeps whatever it was started with.
/// [`served_context_window`] is the fallback for a backend that reports
/// none of its own, vLLM and MLX among them.
fn local_context_window(model: &str, trained: Option<u64>) -> Option<u64> {
    crate::daemon::loaded_context_length(model)
        .or_else(|| served_context_window(super::serve::context_length_from_env(), trained))
}

/// A hybrid pair's window: whichever half holds more.
///
/// Requests above the local budget are routed to the hosted half, so the
/// pair can hold the larger of the two whichever way round they are —
/// declaring only the local window makes the agent compact before a
/// request is ever big enough to route, and declaring only the hosted
/// one understates a local half that is larger.
///
/// `hosted` is `None` for a provider defined in `llmman.conf`, which has
/// no catalog entry; the local window then stands on its own.
fn pair_context_window(local: Option<u64>, hosted: Option<u64>) -> Option<u64> {
    match (local, hosted) {
        (Some(local), Some(hosted)) => Some(local.max(hosted)),
        (window, None) | (None, window) => window,
    }
}

/// The window the daemon would serve, predicted rather than read,
/// mirroring `initial_ctx_size`:
///
/// * `LLMMAN_CONTEXT_LENGTH` set and positive — forwarded as `--ctx-size`
///   uncapped, so it is what gets served.
/// * Set to `0` — `--ctx-size 0`, which llama.cpp reads as the model's
///   own `trained` context, also uncapped.
/// * Unset — the default, clamped *down* to `trained`, so a model
///   trained past [`super::serve::DEFAULT_CTX_SIZE`] is still served
///   only the default.
///
/// `None` when the window is unknown: every caller but codex passes that
/// through as "say nothing", leaving the integration its own default.
///
/// Blind to the daemon's `LLMMAN_NUM_PARALLEL`: llama-server splits
/// `--ctx-size 0`'s trained context across that many slots, so this
/// overstates a request's window by that factor. One more reason
/// [`local_context_window`] reads the live one first.
fn served_context_window(env: Option<u32>, trained: Option<u64>) -> Option<u64> {
    match env {
        Some(0) => trained,
        Some(explicit) => Some(u64::from(explicit)),
        None => trained.map(|trained| trained.min(u64::from(super::serve::DEFAULT_CTX_SIZE))),
    }
}

/// Generic launcher: just set OLLAMA_HOST and run the binary.
fn launch_simple(binary: &str, _model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path(binary).ok_or_else(|| anyhow::anyhow!("{binary} is not installed"))?;
    let server = server();
    exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())])
}

// ---------------------------------------------------------------------------
// Process execution helper
// ---------------------------------------------------------------------------

fn exec_with_env(bin: &Path, args: &[String], extra_env: &[(&str, &str)]) -> anyhow::Result<()> {
    exec_with_env_removing(bin, args, extra_env, &[])
}

fn exec_with_env_removing(
    bin: &Path,
    args: &[String],
    extra_env: &[(&str, &str)],
    remove_env: &[&str],
) -> anyhow::Result<()> {
    std::process::exit(run_with_env_removing(bin, args, extra_env, remove_env)?);
}

/// Runs the integration — in the `--sandbox`, when there is one — and
/// returns its exit code.
fn run_with_env(bin: &Path, args: &[String], extra_env: &[(&str, &str)]) -> anyhow::Result<i32> {
    run_with_env_removing(bin, args, extra_env, &[])
}

fn run_with_env_removing(
    bin: &Path,
    args: &[String],
    extra_env: &[(&str, &str)],
    remove_env: &[&str],
) -> anyhow::Result<i32> {
    // The inherited environment, overlaid with OLLAMA_HOST and the
    // integration's variables, later ones winning.
    let mut overlay = vec![("OLLAMA_HOST".to_string(), server())];
    overlay.extend(
        extra_env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string())),
    );
    if sandbox::active() {
        return sandbox::run(bin, args, &overlay, remove_env);
    }

    let mut cmd = Command::new(bin);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::inherit());
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());
    cmd.envs(overlay);
    for name in remove_env {
        cmd.env_remove(name);
    }

    let status = cmd
        .status()
        .with_context(|| format!("failed to run {}", bin.display()))?;
    Ok(status.code().unwrap_or(1))
}

/// The daemon's URL as the integration reaches it: `daemon::server()`,
/// or the `--sandbox`'s name for this machine.
fn server() -> String {
    sandbox::agent_server()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cline installs itself, and a name launch does not know is left to
    /// its own error, so neither is refused here whatever is on PATH.
    #[test]
    fn check_installed_leaves_cline_and_unknown_names_alone() {
        assert!(check_installed("cline").is_ok());
        assert!(check_installed("not-an-integration").is_ok());
    }

    #[test]
    fn agy_is_listed_as_an_integration() {
        let agy = INTEGRATIONS.iter().find(|i| i.name == "agy").unwrap();
        assert_eq!(agy.binary, "agy");
    }

    #[test]
    fn copilot_is_the_standalone_model_required_byok_integration() {
        let copilot = INTEGRATIONS.iter().find(|i| i.name == "copilot").unwrap();
        assert_eq!(copilot.binary, "copilot");
        for id in ["copilot", "copilot-cli"] {
            assert!(MODEL_REQUIRED.contains(&id));
            assert!(MODEL_FLAG_FORWARDED.contains(&id));
            assert!(!PROVIDER_UNSUPPORTED.iter().any(|(name, _)| *name == id));
            assert_eq!(effort_args(id, "high"), ["--reasoning-effort", "high"]);
        }
    }

    #[test]
    fn droid_is_a_file_configured_model_required_integration() {
        let droid = INTEGRATIONS.iter().find(|i| i.name == "droid").unwrap();
        assert_eq!(droid.binary, "droid");
        assert!(MODEL_REQUIRED.contains(&"droid"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"droid"));
        assert!(CONFIGURED_BY_FILE.contains(&"droid"));
        assert!(check_provider_supported("droid").is_ok());
        assert!(VARIANT_UNSUPPORTED.iter().any(|(name, _)| *name == "droid"));
    }

    /// Every integration `--provider` refuses must be one `launch`
    /// actually dispatches, or the refusal is for a name nobody can type
    /// and the real one is still silently broken.
    #[test]
    fn every_provider_unsupported_integration_is_a_real_one() {
        for (id, why) in PROVIDER_UNSUPPORTED {
            assert!(integration(id).is_some(), "{id} is not an integration");
            assert!(!why.is_empty(), "{id} has no reason");
            assert!(check_provider_supported(id).is_err(), "{id} was accepted");
            // Case-insensitively, the way `launch` dispatches.
            assert!(check_provider_supported(&id.to_uppercase()).is_err());
        }
        // Same for the ones that depend on the daemon holding the key:
        // a name nobody can type protects nobody.
        for id in PROVIDER_NEEDS_DAEMON_KEY {
            assert!(
                INTEGRATIONS.iter().any(|i| i.name == *id),
                "{id} is not an integration"
            );
            assert!(
                !PROVIDER_UNSUPPORTED.iter().any(|(u, _)| u == id),
                "{id} is both refused outright and expected to work"
            );
        }
    }

    /// `--sandbox` must know what every integration writes, or the first
    /// one it misses fails on a read-only home directory, not up front.
    #[test]
    fn every_integration_has_sandbox_state() {
        for i in INTEGRATIONS {
            let state = sandbox_state(i.name).unwrap();
            assert!(!state.is_empty(), "{} has no sandbox state", i.name);
        }
        assert!(sandbox_state("copilot-cli").is_ok());
        assert!(sandbox_state("nope").is_err());
        for id in CONFIGURED_BY_FILE {
            assert!(
                INTEGRATIONS.iter().any(|i| i.name == *id),
                "{id} is not an integration"
            );
        }
    }

    /// The directory a file-configured launcher writes into must be one
    /// the sandbox mounts, or the integration never sees its config.
    #[test]
    fn sandbox_state_covers_where_the_launchers_write() {
        let covers = |name: &str, written: PathBuf| {
            let state = sandbox_state(name).unwrap();
            assert!(
                state.iter().any(|s| matches!(
                    s,
                    sandbox::State::Dir(dir) if written.starts_with(dir)
                )),
                "{name}'s {} is outside its sandbox state {state:?}",
                written.display()
            );
        };
        covers("codex", codex::codex_dir().unwrap());
        covers(
            "opencode",
            opencode::opencode_state_dir().unwrap().join("model.json"),
        );
        covers("pi", pi::pi_agent_dir().unwrap());
        covers("omp", omp::omp_agent_dir().unwrap());
        covers("cline", cline::cline_data_dir().unwrap());
        covers("agy", agy::agy_settings_dir().unwrap());
        covers("hermes", hermes::hermes_home().unwrap());
        covers("qwen", qwen::qwen_home().unwrap());
        covers("dsh", dsh::dsh_config_dir().unwrap());
        covers("grok", grok::grok_home().unwrap().join("llmman"));
        covers(
            "docker-agent",
            docker_agent::docker_agent_config_dir().unwrap(),
        );
        covers(
            "openclaw",
            dirs::home_dir()
                .unwrap()
                .join(".openclaw")
                .join("openclaw.json"),
        );
    }

    /// A provider-routed `--model` must come out under
    /// `providers::REMOTE_PREFIX`, which is the only thing that stops the
    /// daemon resolving it as a HuggingFace or registry reference — and
    /// must keep an `<vendor>/<model>` id (openrouter's shape) intact.
    #[test]
    fn provider_models_are_encoded_under_the_remote_prefix() {
        assert_eq!(
            providers::format_remote_ref("openrouter", "qwen/qwen3-coder"),
            "llmman.provider/openrouter/qwen/qwen3-coder"
        );
        assert_eq!(
            providers::split_remote_ref(&providers::format_remote_ref("groq", "llama-3.3-70b")),
            Some(("groq", "llama-3.3-70b"))
        );
    }

    /// The default path must be untouched by provider support: no
    /// `--provider` means the same shortname resolution, and so the same
    /// daemon behavior, as before it existed.
    #[test]
    fn local_models_are_unaffected_by_the_remote_prefix() {
        for local in ["qwen3.5:0.8b", "hf.co/unsloth/Qwen3.5-0.8B-GGUF"] {
            let resolved = crate::shortnames::resolve_ollama_api(local).unwrap();
            assert!(
                !providers::is_remote_ref(&resolved),
                "{local} resolved to a provider-routed reference: {resolved}"
            );
        }
    }

    /// Every integration `check_model_flag` holds to a model must be one
    /// `launch` dispatches; it is refused without one, under `--provider`
    /// too, and a `--model` after `--` is let through.
    #[test]
    fn model_required_integrations_are_refused_without_a_model() {
        let none: Vec<String> = vec![];
        for id in MODEL_REQUIRED {
            assert!(integration(id).is_some(), "{id} is not an integration");
            assert!(check_model_flag(id, None, None, &none).is_err());
            assert!(check_model_flag(id, Some(" "), None, &none).is_err());
            assert!(check_model_flag(&id.to_uppercase(), None, None, &none).is_err());
            let err = check_model_flag(id, None, Some("openrouter"), &none).unwrap_err();
            assert!(
                err.to_string().contains("--provider openrouter --model"),
                "{err}"
            );
            assert!(check_model_flag(id, Some("m"), None, &none).is_ok());
            let forwarded = vec!["--model".to_string(), "theirs".to_string()];
            assert!(check_model_flag(id, Some("m"), None, &forwarded).is_ok());
        }
        assert!(check_model_flag("claude", None, None, &none).is_ok());
        // The "yours wins" warning is only claimed for launchers that
        // actually yield to it; dsh has no `--model` flag to yield to.
        for id in MODEL_FLAG_FORWARDED {
            assert!(integration(id).is_some(), "{id} is not an integration");
        }
        assert!(!MODEL_FLAG_FORWARDED.contains(&"dsh"));
        // claude yields without requiring a model, so the warning must not
        // depend on `MODEL_REQUIRED`.
        assert!(MODEL_FLAG_FORWARDED.contains(&"claude"));
        assert!(!MODEL_REQUIRED.contains(&"claude"));
    }

    /// goose carries the key in its own environment, so `--provider`
    /// needs neither a refusal nor the daemon holding the key.
    #[test]
    fn goose_carries_its_own_key_so_provider_works() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "goose"));
        assert!(check_provider_supported("goose").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"goose"));
    }

    /// The predicate has to be able to veto, or `find_goose`'s refusal of
    /// the desktop app is decoration. A rejected match must also leave
    /// `find_on_path` as it was: it passes `|_| false`, so every other
    /// integration still resolves exactly as before this predicate
    /// existed.
    #[test]
    fn find_on_path_unless_consults_its_predicate() {
        // Any binary this machine really has, so there is a match for
        // the predicate to veto.
        let present = ["cargo", "sh", "cmd"]
            .into_iter()
            .find(|b| find_on_path(b).is_some());
        let Some(present) = present else {
            eprintln!("skipping: no known binary on PATH to test against");
            return;
        };
        let found = find_on_path(present).unwrap();
        assert!(found.is_file());
        // Vetoing everything finds nothing, whatever is on PATH.
        assert_eq!(find_on_path_unless(present, |_| true), None);
        // Vetoing only what was not found leaves that result standing.
        assert_eq!(
            find_on_path_unless(present, |p| p != found),
            Some(found.clone())
        );
        // A name nothing answers to stays unfound, predicate or not.
        assert_eq!(find_on_path("llmman-no-such-binary-xyz"), None);
        assert_eq!(
            find_on_path_unless("llmman-no-such-binary-xyz", |_| false),
            None
        );
    }

    /// The desktop app keeps Electron state the CLI has none of, and
    /// `--sandbox` has to let it write there: `userData`, named for the
    /// bundle (`Goose`) rather than the CLI. Without it a sandboxed run
    /// starts blank every time.
    #[test]
    fn goose_desktop_sandbox_state_covers_the_electron_user_data() {
        let desktop = sandbox_state("goose-desktop").unwrap();
        let cli = sandbox_state("goose").unwrap();
        // Everything the CLI gets, and then what only the app keeps.
        assert!(desktop.len() > cli.len(), "{desktop:?}");
        assert!(
            desktop.iter().any(|s| matches!(
                s,
                sandbox::State::Dir(d) if d.file_name() == Some(std::ffi::OsStr::new("Goose"))
            )),
            "no userData directory in {desktop:?}"
        );
        // macOS keeps preferences outside `userData`, as a file.
        if cfg!(target_os = "macos") {
            assert!(
                desktop.iter().any(|s| matches!(
                    s,
                    sandbox::State::Files(f) if f.extension() == Some(std::ffi::OsStr::new("plist"))
                )),
                "no preferences file in {desktop:?}"
            );
        }
    }

    /// Like the CLI, the desktop app carries the key per request and
    /// writes nothing, so it must be absent from every list that would
    /// say otherwise, and `--provider` must work.
    #[test]
    fn goose_desktop_carries_its_own_key_so_provider_works() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "goose-desktop"));
        assert!(check_provider_supported("goose-desktop").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"goose-desktop"));
        assert!(!CONFIGURED_BY_FILE.contains(&"goose-desktop"));
        // It has no --model of its own to yield to.
        assert!(!MODEL_FLAG_FORWARDED.contains(&"goose-desktop"));
        assert!(MODEL_REQUIRED.contains(&"goose-desktop"));
    }

    #[test]
    fn cline_is_model_required_and_supports_provider_routes() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "cline"));
        assert!(MODEL_REQUIRED.contains(&"cline"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"cline"));
        assert!(!PROVIDER_UNSUPPORTED.iter().any(|(id, _)| *id == "cline"));
        assert!(PROVIDER_NEEDS_DAEMON_KEY.contains(&"cline"));
        assert!(check_provider_supported("cline").is_ok());
    }

    #[test]
    fn grok_is_model_required_and_refuses_unrepresentable_provider_routes() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "grok"));
        assert!(MODEL_REQUIRED.contains(&"grok"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"grok"));
        let error = check_provider_supported("grok").unwrap_err().to_string();
        assert!(error.contains("model catalog"), "{error}");
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"grok"));
    }

    /// A word or `=`-joined, and nothing looser: `-sm` is not `-m`.
    #[test]
    fn has_flag_takes_the_exact_and_joined_forms_only() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(has_flag(&args(&["--model", "x"]), "--model", Some("-m")));
        assert!(has_flag(&args(&["--model=x"]), "--model", Some("-m")));
        assert!(has_flag(&args(&["-m", "x"]), "--model", Some("-m")));
        assert!(has_flag(&args(&["-m=x"]), "--model", Some("-m")));
        assert!(!has_flag(&args(&["-sm", "x"]), "--model", Some("-m")));
        assert!(!has_flag(
            &args(&["--model-context", "x"]),
            "--model",
            Some("-m")
        ));
    }

    /// The `--variant` guard in `run` relies on this to refuse a forwarded
    /// `--model`.
    #[test]
    fn forwarded_model_flag_is_seen_for_each_integration_that_yields() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let model = args(&["--model", "other/model"]);
        for name in MODEL_FLAG_FORWARDED {
            assert!(has_forwarded_model_flag(name, &model), "{name}");
        }
        // docker-agent refuses a forwarded model; dsh takes none.
        assert!(!has_forwarded_model_flag("docker-agent", &model));
        assert!(!has_forwarded_model_flag("dsh", &model));
        // copilot and claude have no `-m`; grok does.
        let short = args(&["-m", "other/model"]);
        assert!(!has_forwarded_model_flag("claude", &short));
        assert!(!has_forwarded_model_flag("copilot", &short));
        assert!(has_forwarded_model_flag("grok", &short));
        // After a second `--` it is a message, not a flag.
        let message = args(&["--", "--model", "other/model"]);
        assert!(!has_forwarded_model_flag("claude", &message));
        assert!(has_forwarded_model_flag(
            "claude",
            &args(&["--model", "other/model", "--", "hi"])
        ));
    }

    /// Only the integrations that put a window somewhere pay for the
    /// load that reads the live one back. Every launcher in [`launch`]
    /// that takes `context_window` must be listed, or it silently gets
    /// none.
    #[test]
    fn the_integrations_that_declare_a_window_are_the_ones_that_take_one() {
        assert!(declares_context_window("opencode"));
        assert!(declares_context_window("codex"));
        assert!(declares_context_window("OpenCode"), "matched case-blind");
        // pi and omp write a window into their own model entries too,
        // so they read the live one back rather than predict it from
        // the trained context.
        assert!(declares_context_window("pi"));
        assert!(declares_context_window("omp"));
        assert!(!declares_context_window("claude"));
        assert!(!declares_context_window("aider"));
    }

    /// Requests above the local budget route to the hosted half, so the
    /// pair holds the larger window whichever way round the two are.
    /// Declaring the local one alone has the agent compact before a
    /// request is ever big enough to route — the bug this fixes.
    #[test]
    fn pair_context_window_takes_whichever_half_holds_more() {
        // The usual pairing: a small local model overflowing to a large
        // hosted one.
        assert_eq!(
            pair_context_window(Some(262_144), Some(1 << 20)),
            Some(1 << 20)
        );
        // Reversed — paired for quality, not capacity. The hosted window
        // must not shrink what the local half can already hold.
        assert_eq!(
            pair_context_window(Some(1 << 20), Some(200_000)),
            Some(1 << 20)
        );
        // A provider from llmman.conf names no window, so the local one
        // stands alone rather than being discarded.
        assert_eq!(pair_context_window(Some(262_144), None), Some(262_144));
        assert_eq!(pair_context_window(None, Some(200_000)), Some(200_000));
        assert_eq!(pair_context_window(None, None), None);
    }

    /// The same precedence, minus codex's guess: the launchers that may
    /// omit the key see `None` instead of a fallback, so an integration's
    /// own default stands rather than a number llmman invented.
    #[test]
    fn served_context_window_prefers_the_environment_then_the_trained_context() {
        let cases = [
            (Some(16384), Some(32768), Some(16384)),
            (Some(65536), Some(32768), Some(65536)),
            (Some(16384), None, Some(16384)),
            (None, Some(32768), Some(32768)),
            // `--ctx-size 0` is the model's own context, so it is served
            // uncapped like any other explicit value — the one case the
            // default-clamping branch below would get wrong.
            (Some(0), Some(32768), Some(32768)),
            (Some(0), Some(1 << 20), Some(1 << 20)),
            // Clamped down to the trained context, never up to it.
            (
                None,
                Some(1 << 20),
                Some(u64::from(super::super::serve::DEFAULT_CTX_SIZE)),
            ),
            // An explicit value is forwarded uncapped, so it stands.
            (Some(1 << 20), Some(1 << 20), Some(1 << 20)),
            // Neither: say nothing, where codex would guess. A `0` with
            // no trained context to name has nothing to forward either.
            (Some(0), None, None),
            (None, None, None),
        ];
        for (env, trained, want) in cases {
            assert_eq!(
                served_context_window(env, trained),
                want,
                "env={env:?} trained={trained:?}"
            );
        }
    }

    /// dsh carries a real key per launch, unlike hermes, so it belongs
    /// on neither `--provider` refusal list.
    #[test]
    fn dsh_is_a_real_integration_and_not_on_a_provider_refusal_list() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "dsh"));
        assert!(!PROVIDER_UNSUPPORTED.iter().any(|(id, _)| *id == "dsh"));
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"dsh"));
    }

    // -----------------------------------------------------------------
    // docker-agent
    // -----------------------------------------------------------------

    /// Needs a model, but carries its key in the environment, so it is on
    /// neither `PROVIDER_UNSUPPORTED` nor `PROVIDER_NEEDS_DAEMON_KEY` —
    /// and not on `MODEL_FLAG_FORWARDED` either, since a forwarded
    /// `--model` is refused rather than obeyed.
    #[test]
    fn docker_agent_is_a_real_model_required_integration_provider_can_drive() {
        let entry = INTEGRATIONS
            .iter()
            .find(|i| i.name == "docker-agent")
            .expect("docker-agent is not an integration");
        assert_eq!(entry.binary, "docker-agent");
        assert!(MODEL_REQUIRED.contains(&"docker-agent"));
        assert!(!MODEL_FLAG_FORWARDED.contains(&"docker-agent"));
        assert!(check_provider_supported("docker-agent").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"docker-agent"));
    }

    /// `print_integrations` pads each name into a 14-wide column and
    /// prints a space after it, so every description starts at the same
    /// place — including "goose-desktop", the longest at 13. A longer
    /// name would push its own description right instead.
    #[test]
    fn every_integration_name_fits_the_listings_column() {
        let longest = INTEGRATIONS.iter().map(|i| i.name.len()).max().unwrap();
        assert!(
            longest <= 14,
            "{longest}-character name overflows the `{{:<14}}` column in print_integrations"
        );
    }

    // -- --variant ---------------------------------------------------------

    /// A model with unknown levels (`None`) takes any effort level, and
    /// opencode finds it among its variants; one with known levels still
    /// refuses what it lacks.
    #[test]
    fn a_variant_is_refused_only_for_a_model_with_known_levels() {
        let check = |thinking: Option<&Thinking>, variant: &str| {
            check_variant("m", thinking, variant).map_err(|e| e.to_string())
        };
        assert!(check(None, "xhigh").is_err(), "the guess refuses");
        for variant in ["none", "minimal", "low", "high", "xhigh", "max"] {
            let widened = unknown_levels_with(variant);
            check(Some(&widened), variant).unwrap();
        }
        let thinking = unknown_levels_with("thinking");
        assert!(check(Some(&thinking), "thinking").is_err());

        let widened = unknown_levels_with("xhigh");
        let variants = opencode::opencode_variants(Some(&widened));
        let names: Vec<_> = variants.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["none", "low", "medium", "high", "xhigh"]);
        assert_eq!(
            variants[4].1,
            serde_json::json!({ "reasoningEffort": "xhigh" })
        );

        let listed = Thinking::Listed(["low", "high"].map(String::from).to_vec());
        assert!(check(Some(&listed), "high").is_ok());
        let err = check(Some(&listed), "xhigh").unwrap_err();
        assert!(
            err.contains("m has no variant xhigh; it has low, high"),
            "{err}"
        );
        let none = Thinking::Listed(Vec::new());
        assert!(check(Some(&none), "low")
            .unwrap_err()
            .contains("does not think"));
    }

    #[test]
    fn spell_variant_uses_each_integrations_own_words() {
        let ok = |name, variant| spell_variant(name, variant).unwrap();
        assert_eq!(ok("opencode", "thinking"), "thinking");
        assert_eq!(ok("codex", "none"), "none");
        assert_eq!(ok("muse", "minimal"), "minimal");
        assert_eq!(ok("pi", "none"), "off");
        assert_eq!(ok("dsh", "none"), "off");
        assert_eq!(ok("claude", "max"), "max");
        // A switch-only template's on is a level llmman serves as on.
        assert_eq!(ok("cline", "thinking"), "medium");
        assert_eq!(ok("Claude", "high"), "high");
    }

    #[test]
    fn spell_variant_refuses_what_the_integration_cannot_start_at() {
        let err = |name, variant| spell_variant(name, variant).unwrap_err().to_string();
        let claude = err("claude", "none");
        assert!(
            claude.contains("low, medium, high, xhigh, max, thinking"),
            "{claude}"
        );
        assert!(!claude.contains("none,"), "{claude}");
        // Would clamp to high rather than start where asked.
        assert!(err("pi", "xhigh").contains("none, minimal"));
        assert!(err("cline", "minimal").contains("cline cannot start"));
        assert!(err("kimi", "high").contains("--variant does not work with kimi"));
        assert!(err("muse", "none").contains("muse cannot start"));
        assert!(err("nope", "high").contains("unknown integration"));
    }

    /// Every integration is either spelled or refused, never silently
    /// launched without its variant.
    #[test]
    fn every_integration_takes_or_refuses_a_variant() {
        for i in INTEGRATIONS {
            let spelled = spell_variant(i.name, "thinking").is_ok();
            let refused = VARIANT_UNSUPPORTED.iter().any(|(id, _)| *id == i.name);
            let configured = matches!(i.name, "opencode" | "qwen" | "dsh");
            assert!(spelled != refused, "{}", i.name);
            assert!(
                refused || configured || !effort_args(i.name, "medium").is_empty(),
                "{} is spelled but never told",
                i.name
            );
        }
    }

    #[test]
    fn effort_args_lead_with_each_integrations_flag() {
        let args = |name| effort_args(name, "high");
        assert_eq!(args("muse"), ["--reasoning-effort", "high"]);
        assert_eq!(args("claude"), ["--effort", "high"]);
        assert_eq!(args("pi"), ["--thinking", "high"]);
        assert_eq!(args("hermes"), ["--reasoning", "high"]);
        assert_eq!(args("codex"), ["-c", "model_reasoning_effort=high"]);
        assert_eq!(
            args("aider"),
            [
                "--no-check-model-accepts-settings",
                "--reasoning-effort",
                "high"
            ]
        );
        assert!(args("qwen").is_empty());
    }

    #[test]
    fn launch_variant_is_one_of_llmmans() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: LaunchArgs,
        }
        let parse = |v: &str| <Cli as clap::Parser>::try_parse_from(["l", "pi", "--variant", v]);
        assert_eq!(
            parse("xhigh").unwrap().args.variant.as_deref(),
            Some("xhigh")
        );
        assert!(parse("turbo").is_err());

        // Refused before the daemon starts: the variant is checked
        // against llmman's --model, not the one Qwen Code would use.
        let args = <Cli as clap::Parser>::try_parse_from([
            "l",
            "qwen",
            "-m",
            "a",
            "--variant",
            "high",
            "--",
            "--model",
            "b",
        ])
        .unwrap()
        .args;
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("not after --"), "{err}");
    }

    /// OpenShell gets none of the host's files, including the state a
    /// `--variant` is written to.
    #[cfg(not(windows))]
    #[test]
    fn opencode_variant_is_refused_where_the_sandbox_gets_no_files() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: LaunchArgs,
        }
        let args = <Cli as clap::Parser>::try_parse_from([
            "l",
            "opencode",
            "--sandbox",
            "openshell",
            "--variant",
            "high",
        ])
        .unwrap()
        .args;
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("openshell cannot run opencode"), "{err}");
    }

    /// `llmman launch <argv...>` as the command line parses it.
    fn launch_args(argv: &[&str]) -> LaunchArgs {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: LaunchArgs,
        }
        <Cli as clap::Parser>::try_parse_from(["l"].into_iter().chain(argv.iter().copied()))
            .unwrap()
            .args
    }

    /// `apply_defaults` for `argv` against `[launch.<integration>]`.
    fn applied(argv: &[&str], conf: &crate::config::LaunchConf) -> (LaunchArgs, Vec<String>) {
        let args = launch_args(argv);
        let name = args.integration.clone().unwrap();
        apply_defaults(&args, &name, || Ok(conf.clone())).unwrap()
    }

    fn sonnet() -> crate::config::LaunchConf {
        crate::config::launch_conf(Some("anthropic"), Some("claude-sonnet-5-5"), Some("xhigh"))
    }

    /// `launch opencode` alone is the full invocation, and says which
    /// flags it stands for.
    #[test]
    fn configured_defaults_fill_every_flag_the_command_line_left_out() {
        let (args, applied) = applied(&["opencode"], &sonnet());
        assert_eq!(args.provider.as_deref(), Some("anthropic"));
        assert_eq!(args.model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(args.variant.as_deref(), Some("xhigh"));
        assert_eq!(
            applied,
            [
                "--provider anthropic",
                "--model claude-sonnet-5-5",
                "--variant xhigh"
            ]
        );

        // Arguments for the integration itself are not a choice of model.
        let (args, _) = self::applied(
            &["opencode", "--sandbox", "docker", "--", "run", "hi"],
            &sonnet(),
        );
        assert_eq!(args.model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(args.extra_args, ["run", "hi"]);
    }

    #[test]
    fn nothing_configured_changes_nothing() {
        let none = crate::config::LaunchConf::default();
        let (args, applied) = applied(&["opencode", "-m", "qwen3.8", "--variant", "low"], &none);
        assert!(applied.is_empty());
        assert_eq!(args.model.as_deref(), Some("qwen3.8"));
        assert_eq!(args.variant.as_deref(), Some("low"));
        assert_eq!(args.provider, None);
    }

    /// `--variant` is the one flag that keeps the configured model.
    #[test]
    fn a_typed_variant_replaces_only_the_configured_variant() {
        let (args, applied) = applied(&["opencode", "--variant", "low"], &sonnet());
        assert_eq!(args.provider.as_deref(), Some("anthropic"));
        assert_eq!(args.model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(args.variant.as_deref(), Some("low"));
        assert_eq!(
            applied,
            ["--provider anthropic", "--model claude-sonnet-5-5"]
        );
    }

    #[test]
    fn a_typed_model_choice_turns_the_configured_set_off() {
        for argv in [
            &["opencode", "--model", "qwen3.8"][..],
            &["opencode", "-m", "qwen3.8"],
            &["opencode", "--provider", "openai"],
            &["opencode", "-p", "openai", "--model", "gpt-5"],
            &["opencode", "--overflow-provider", "openai"],
            &["opencode", "--overflow-model", "gpt-5"],
        ] {
            let typed = launch_args(argv);
            let (args, applied) = applied(argv, &sonnet());
            assert!(applied.is_empty(), "{argv:?}: {applied:?}");
            assert_eq!(args.provider, typed.provider, "{argv:?}");
            assert_eq!(args.model, typed.model, "{argv:?}");
            assert_eq!(args.variant, None, "{argv:?}");
        }
    }

    /// A `--model` after `--` wins over the configured one, as it does
    /// over a typed one, but the configured model is still there for the
    /// integrations that need one; its variant is not.
    #[test]
    fn a_forwarded_model_keeps_the_configured_model_but_not_its_variant() {
        let (args, applied) = applied(&["qwen", "--", "--model", "b"], &sonnet());
        assert_eq!(args.model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(args.variant, None);
        assert_eq!(
            applied,
            ["--provider anthropic", "--model claude-sonnet-5-5"]
        );

        // opencode has no such flag, so there it is a prompt's word.
        let (args, _) = self::applied(&["opencode", "--", "--model", "x"], &sonnet());
        assert_eq!(args.variant.as_deref(), Some("xhigh"));
    }

    /// Defaults that cannot be read are an error, but only where they
    /// would have been used.
    #[test]
    fn unreadable_defaults_fail_only_a_launch_that_needs_them() {
        let args = launch_args(&["opencode"]);
        let err = apply_defaults(&args, "opencode", || Err("bad file")).unwrap_err();
        assert!(err.to_string().contains("bad file"), "{err}");
        for argv in [&["opencode", "-m", "x"][..], &["opencode", "-p", "x"]] {
            let args = launch_args(argv);
            apply_defaults(&args, "opencode", || Err("bad file")).expect("not consulted");
        }
    }

    #[test]
    fn canonical_integration_resolves_case_and_aliases() {
        assert_eq!(canonical_integration("OpenCode"), Some("opencode"));
        assert_eq!(canonical_integration("copilot-cli"), Some("copilot"));
        assert_eq!(canonical_integration("nonesuch"), None);
        for i in INTEGRATIONS {
            assert_eq!(canonical_integration(i.name), Some(i.name));
        }
    }
}
