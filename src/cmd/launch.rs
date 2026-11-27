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
mod hdrm;
mod hermes;
mod keys;
mod kimi;
mod models;
mod muse;
mod omp;
mod openclaw;
mod opencode;
mod optimizer;
mod pi;
mod plan;
mod qwen;
mod sandbox;
mod vibe;

pub use optimizer::Optimizer;
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

    /// Model to start on: a pulled model, or with --provider one of its
    /// models. Without it, the agent starts on the provider's default model
    /// (--provider alone: the newest tool-calling chat model priced at or
    /// below the median of the provider's), else (neither flag) the model it last ran (read
    /// for claude, codex, opencode, hermes and openclaw only), the pulled
    /// model loaded now or pulled last, or a provider you have a key for.
    /// Every model and variant stays available inside the agent, except in
    /// copilot and Gemini CLI, which take one model.
    #[arg(long, short, value_name = "MODEL")]
    pub model: Option<String>,

    /// Start on this provider's model (openai, anthropic, openrouter, …)
    /// instead of a local one: --model, else its default, the newest
    /// tool-calling model at or below the median price (said on stderr).
    /// Its models are listed first in the agent. See `llmman providers`.
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

    /// Run the integration in a sandbox, sbx unless one is named. See
    /// docs/sandbox.md.
    #[arg(
        long,
        value_enum,
        value_name = "SANDBOX",
        num_args = 0..=1,
        default_missing_value = "sbx"
    )]
    pub sandbox: Option<Sandbox>,

    /// Start at this variant of the model (see `llmman show`), as
    /// opencode's `run --variant`.
    #[arg(long, value_name = "VARIANT", value_parser = super::run::variant_parser())]
    pub variant: Option<String>,

    /// What to launch with to make the session cheaper: none (the default)
    /// launches as is; tk compacts the output of the shell commands the
    /// agent runs (claude, opencode); cvmn asks for terse replies (claude,
    /// codex, opencode); hdrm compresses what the agent sends the model
    /// (claude, codex, opencode). See docs/optimizers.md.
    #[arg(long, value_enum, value_name = "OPTIMIZER", default_value = "none")]
    pub optimizer: Optimizer,

    /// Extra arguments forwarded to the integration binary (after --)
    #[arg(last = true, value_name = "ARGS")]
    pub extra_args: Vec<String>,
}

pub fn run(args: &LaunchArgs) -> anyhow::Result<()> {
    let configured;
    let args = match args.integration.as_deref() {
        Some(name) => {
            // An alias launches the integration it names.
            let name = canonical_integration(name).unwrap_or(name);
            let (mut with_defaults, applied) =
                apply_defaults(args, name, || crate::config::launch_defaults(name))?;
            if !applied.is_empty() {
                // So a refusal of a flag that was not typed makes sense.
                eprintln!(
                    "[llmman] {name}: llmman.conf supplies {}",
                    applied.join(" ")
                );
            }
            with_defaults.integration = Some(name.to_string());
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

    // Before sbx returns below, which would never reach it, and before the
    // daemon starts. An unknown name is left to the check that reports it.
    if let Some(id) = canonical_integration(name) {
        args.optimizer
            .check(id, args.sandbox.is_some(), &args.extra_args)?;
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

    let speech = std::env::var(goose_desktop::TRANSCRIPTION_MODEL_ENV).ok();
    // Before either arm starts the daemon; see `check_model_flag`.
    check_model_flag(name, args.model.as_deref(), &args.extra_args);
    if name.eq_ignore_ascii_case("copilot")
        || name.eq_ignore_ascii_case("copilot-cli")
        || name.eq_ignore_ascii_case("droid")
        || name.eq_ignore_ascii_case("muse")
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
    if crate::daemon::os_picks_port() {
        eprintln!(
            "[llmman] warning: LLMMAN_HOST lets the OS pick llmman serve's port, so {name} \
             loses the daemon if it restarts on another"
        );
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
    if name.eq_ignore_ascii_case("muse") {
        muse::check_muse_args(&args.extra_args)?;
    }
    if name.eq_ignore_ascii_case("kimi") && overflow.is_none() {
        kimi::check_kimi_args(args.model.as_deref(), &args.extra_args)?;
    }
    // Each launcher looks for its program only once it runs, which is
    // after the daemon has started and the model has been pulled: a
    // missing agent would otherwise cost a download first.
    check_installed(name)?;
    let plan = match overflow {
        Some(pair) => plan::pair(name, args, pair)?,
        None => plan::resolve(name, args, provider, speech.as_deref())?,
    };
    let launched = launch(name, &plan, variant, &args.extra_args, args.optimizer);
    hdrm::stop();
    models::finish();
    launched
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
    "muse",
];

/// Says so when a `--model` after `--` is the one an integration in
/// `MODEL_FLAG_FORWARDED` will run, since `run` resolves the top-level one
/// and, locally, preloads it.
///
/// No integration is refused for lacking a model any more: one that is not
/// named is picked (see [`models::resolve_start`]).
fn check_model_flag(integration: &str, model: Option<&str>, extra_args: &[String]) {
    let name = integration.to_lowercase();
    let model = model.map(str::trim).filter(|m| !m.is_empty());
    if let Some(model) = model {
        if has_forwarded_model_flag(&name, extra_args) {
            eprintln!(
                "[llmman] {name}: the --model after -- wins over --model {model}, the one llmman resolved"
            );
        }
    }
}

/// Whether this integration accepts a forwarded model flag. Copilot and
/// claude spell it only `--model`; the other launchers also accept `-m`.
fn has_forwarded_model_flag(integration: &str, extra_args: &[String]) -> bool {
    let name = integration.to_lowercase();
    let short =
        (!matches!(name.as_str(), "copilot" | "copilot-cli" | "claude" | "muse")).then_some("-m");
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

/// Integrations llmman configures through a file on disk. They take a
/// model on every launch, so `--provider` works, but they cannot carry
/// the key: writing a real one into `~/.hermes/config.yaml` would persist
/// a credential, which this feature promises not to do. They rely on
/// `llmman serve` having the variable itself — which it only uses for a
/// daemon nobody else can reach (see `reachable_only_locally`).
const PROVIDER_NEEDS_DAEMON_KEY: &[&str] = &["hermes", "cline", "pi", "omp", "openclaw"];

/// Integrations that cannot be offered a hosted model; `--provider` is refused for them. OMP 19.1.1
/// stops at startup on a model id naming two vendors ("ambiguous class ... `amazon` and `openai`
/// tie"), as a hosted model's routing reference often does. Gemini CLI and Kimi's Python CLI are
/// not here: see their launchers.
const STORED_ONLY: &[(&str, &str)] = &[(
    "omp",
    "omp stops at startup on a model id that names two vendors, as a hosted \
     model's reference can (llmman.provider/amazon-bedrock/openai.gpt-5.4)",
)];

fn stored_only_reason(integration: &str) -> Option<&'static str> {
    let name = integration.to_lowercase();
    STORED_ONLY
        .iter()
        .find(|(id, _)| *id == name)
        .map(|(_, why)| *why)
}

/// Integrations that declare a context window, and so are worth
/// loading the model to read the real one back for. Keep in step with
/// [`launch`]'s own dispatch: an integration missing here is simply
/// told no window, not told a wrong one.
fn declares_context_window(integration: &str) -> bool {
    matches!(
        integration.to_lowercase().as_str(),
        "claude" | "opencode" | "codex" | "pi" | "omp" | "kimi" | "openclaw" | "vibe"
    )
}

fn check_provider_supported(integration: &str) -> anyhow::Result<()> {
    let name = integration.to_lowercase();
    if let Some(why) = stored_only_reason(&name) {
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

/// What [`resolve_provider_model`] resolves, for a hybrid pair's hosted half.
/// Named fields rather than a tuple: three of these are `Option`s, which a
/// caller can drop by destructuring past them without the compiler saying so.
struct ResolvedProvider {
    /// The hosted model's full reference, as the daemon routes it.
    reference: String,
    /// The key the integration authenticates with.
    api_key: String,
    /// The window the catalog says this model holds.
    context_window: Option<u64>,
    /// The most the catalog says it will emit in one reply (see
    /// [`crate::daemon::ProviderDetail::max_output`]).
    max_output: Option<u32>,
}

/// Validates the hosted half of an `--overflow-provider` pair against the
/// running daemon's catalog (see [`crate::daemon::provider`]), resolving the
/// reference the daemon routes on, the key `integration` authenticates with,
/// and what the catalog says about the model.
///
/// The key travels per request in the integration's own Authorization
/// header, never to disk or a command line: a pair is one model, so there
/// is nothing in the agent to switch to (a launch that offers a list hands
/// out a session instead, see [`keys`]). The placeholder goes instead for
/// the integrations in [`PROVIDER_NEEDS_DAEMON_KEY`], which cannot carry
/// one, and whenever the daemon's key is the one that matters.
///
/// Every check here is one the daemon would otherwise make at first
/// request, by which point the integration has already taken over the
/// terminal and reports whatever it makes of an HTTP error. Failing in
/// llmman's own output, before the handoff, is the difference between a
/// named missing environment variable and an opaque "connection error"
/// inside someone else's TUI.
fn resolve_provider_model(
    provider: &str,
    model: &str,
    integration: &str,
) -> anyhow::Result<ResolvedProvider> {
    // Asked of the daemon, not models.dev: it routes the request, so it
    // is the authority on whether this provider exists — and on whether
    // *it* has the key, which this shell cannot see. The model is sent so a
    // catalog that predates it is refreshed, not warned about.
    let entry = daemon::provider(provider, Some(model))?;
    entry.warn_unlisted(model);
    // A daemon requiring a key takes that header for it, so the provider
    // key cannot travel: the daemon's is the only one, and an
    // authenticated caller may spend it.
    let key = (keys::carries_key(integration) && crate::auth::client_key().is_none())
        .then(|| keys::key_here(&entry))
        .flatten();
    keys::check_start(integration, &entry, key.is_some())?;
    Ok(ResolvedProvider {
        reference: providers::format_remote_ref(provider, model),
        api_key: key.unwrap_or_else(integration_key),
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
    Integration {
        name: "vibe",
        description: "Mistral Vibe CLI",
        binary: "vibe",
    },
    Integration {
        name: "muse",
        description: "Meta Muse Code",
        binary: "muse",
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
        "muse-code" => "muse",
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
        "\nUsage: llmman launch <integration> [--model <model>] [--provider <provider>] [--sandbox [sandbox]]"
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
    crate::env_dir_from(|var| std::env::var_os(var), key)
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
        "copilot" | "droid" | "muse" => common::find_on_path_or_local_bin(i.binary),
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
    ("kimi", "llmman does not set its thinking yet"),
    ("openclaw", "llmman does not set its thinking yet"),
    (
        "droid",
        "Droid does not support reasoning effort for custom models",
    ),
    ("gemini", "llmman's Gemini API ignores thinkingConfig"),
    ("agy", "llmman's Gemini API ignores thinkingConfig"),
    ("goose", "it sends effort only for OpenAI's models"),
    ("goose-desktop", "it sends effort only for OpenAI's models"),
    ("docker-agent", "it ignores thinking_budget here"),
    ("vibe", "llmman does not set its thinking yet"),
];

const LOW_TO_MAX: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// Each other integration's word for thinking off, if any, and the
/// levels it takes for an unknown model without clamping (pi, omp and
/// Cline clamp the rest). opencode takes llmman's own variants.
const VARIANT_SPELLINGS: &[(&str, Option<&str>, &[&str])] = &[
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
    // Muse Code's `meta` provider refuses `--reasoning-effort none`.
    ("muse", None, EFFORT_LEVELS),
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

/// A `--variant` as the integration spells it, for those configured with
/// the list (each row's own levels come from the row).
struct Effort<'a> {
    default: &'a str,
}

/// The flags that start `integration` at `effort`; opencode, qwen and
/// dsh take it in their configuration instead, and muse's launcher adds
/// its flag itself.
fn effort_args(integration: &str, effort: &str) -> Vec<String> {
    let flags: &[&str] = match integration {
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

/// Starts `name` on what `plan` resolved. `plan.api_key` is what the
/// integration is told to authenticate with: [`providers::PLACEHOLDER_API_KEY`]
/// for a model the daemon serves itself, or a session token that stands
/// for the provider keys this shell holds (see [`keys`]), so the agent can
/// switch provider without any one key travelling to another provider.
///
/// Only the launchers that pass a credential in the integration's
/// environment can carry it; the ones that go through a config file on
/// disk keep the placeholder rather than persist one, and need the
/// provider's key in the daemon's own environment.
///
/// `plan.context_window` is the window a request may fill, and the only
/// context figure an integration is told: a local model's loaded one (see
/// [`local_context_window`]), a provider model's catalog one, or a pair's
/// larger half (see [`pair_context_window`]); `None` when none of them is
/// known. codex and kimi substitute a fallback for a `None`.
fn launch(
    name: &str,
    plan: &plan::Plan,
    variant: Option<&str>,
    extra_args: &[String],
    optimizer: Optimizer,
) -> anyhow::Result<()> {
    let plan::Plan {
        model,
        api_key,
        thinking,
        vision,
        context_window,
        max_output,
        models,
    } = plan;
    let (model, api_key, vision) = (model.as_str(), api_key.as_str(), *vision);
    let (context_window, max_output) = (*context_window, *max_output);
    let name = name.to_lowercase();
    // Unknown levels are a guess: add the one asked for, don't refuse it.
    let widened = variant
        .filter(|_| thinking.is_none())
        .map(unknown_levels_with);
    let thinking = widened.as_ref().or(thinking.as_ref());
    let models = with_widened_start(models, widened.as_ref());
    let models = &*models;
    if let Some(variant) = variant {
        check_variant(model, thinking, variant)?;
    }
    let effort = variant.map(|v| spell_variant(&name, v)).transpose()?;
    let listed = effort.map(|default| Effort { default });
    let extra_args = &[
        effort.map_or_else(Vec::new, |e| effort_args(&name, e)),
        extra_args.to_vec(),
    ]
    .concat();
    // pi and omp force thinking off for a model not marked as reasoning;
    // a model with the variant reasons.
    let reasons = variant.is_some()
        || thinking
            .and_then(Thinking::template)
            .is_some_and(|t| t.thinks);
    match name.as_str() {
        "claude" => claude::launch_claude(
            model,
            api_key,
            context_window,
            models,
            extra_args,
            optimizer,
        ),
        "opencode" => opencode::launch_opencode(
            model,
            api_key,
            thinking,
            variant,
            vision,
            context_window,
            max_output,
            models,
            extra_args,
            optimizer,
        ),
        "codex" => codex::launch_codex(model, api_key, models, extra_args, optimizer),
        "pi" => pi::launch_pi(model, reasons, vision, context_window, models, extra_args),
        "omp" => omp::launch_omp(model, reasons, vision, context_window, models, extra_args),
        "cline" => cline::launch_cline(model, models, extra_args),
        "aider" => aider::launch_aider(model, api_key, models, extra_args),
        "copilot" | "copilot-cli" => copilot::launch_copilot(model, api_key, models, extra_args),
        "droid" => droid::launch_droid(model, api_key, vision, max_output, models, extra_args),
        "kimi" => kimi::launch_kimi(model, api_key, context_window, models, extra_args),
        "gemini" => gemini::launch_gemini(model, api_key, models, extra_args),
        "agy" => agy::launch_agy(model, api_key, models, extra_args),
        "hermes" => hermes::launch_hermes(model, vision, models, extra_args),
        "openclaw" => openclaw::launch_openclaw(model, models, extra_args),
        "qwen" => qwen::launch_qwen(model, api_key, vision, listed.as_ref(), models, extra_args),
        "dsh" => dsh::launch_dsh(model, api_key, vision, listed.as_ref(), models, extra_args),
        "goose" => goose::launch_goose(model, api_key, models, extra_args),
        "goose-desktop" => goose_desktop::launch_goose_desktop(model, api_key, models, extra_args),
        "grok" => grok::launch_grok(model, api_key, listed.as_ref(), models, extra_args),
        "docker-agent" => docker_agent::launch_docker_agent(model, api_key, models, extra_args),
        "vibe" => vibe::launch_vibe(model, api_key, context_window, models, extra_args),
        "muse" => muse::launch_muse(model, api_key, effort, extra_args),
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
/// environment and arguments alone. (An integration that writes a file
/// only for a list to switch among, such as goose, vibe, kimi or aider,
/// is not one: with nothing to switch to it writes none, which is what
/// `--sandbox openshell` is given. Gemini CLI reads its base URL only
/// under a login saved in its home, so it cannot run there at all.)
const CONFIGURED_BY_FILE: &[&str] = &[
    "agy",
    "cline",
    "codex",
    "docker-agent",
    "dsh",
    "droid",
    "gemini",
    "grok",
    "hermes",
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
        "kimi" => vec![Dir(kimi::kimi_share_dir()?), Dir(kimi::kimi_code_home()?)],
        "vibe" => vec![Dir(
            env_dir("VIBE_HOME").unwrap_or_else(|| home.join(".vibe"))
        )],
        "muse" => vec![Dir(config.join("muse")), Dir(data.join("muse"))],
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
            // Where goose finds its custom providers, which `GOOSE_PATH_ROOT`
            // or Windows move.
            let providers = Dir(goose::goose_config_dir()?);
            if !dirs.contains(&providers) {
                dirs.push(providers);
            }
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

/// `$var`, else `default` under the home directory (looked up only then).
fn env_dir_or_home(var: &str, default: &str) -> anyhow::Result<PathBuf> {
    match env_dir(var) {
        Some(dir) => Ok(dir),
        None => Ok(dirs::home_dir().context("no home directory")?.join(default)),
    }
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

/// `models`, with the starting row's thinking choices replaced by `widened`
/// when it has none of its own, so an adapter reading the row agrees with
/// the effort the launch starts at.
fn with_widened_start<'a>(
    models: &'a models::ModelList,
    widened: Option<&Thinking>,
) -> std::borrow::Cow<'a, models::ModelList> {
    let Some(widened) = widened else {
        return std::borrow::Cow::Borrowed(models);
    };
    if models.start().is_none_or(|start| start.thinking.is_some()) {
        return std::borrow::Cow::Borrowed(models);
    }
    let mut entries = models.entries().to_vec();
    entries[0].thinking = Some(widened.choices().into_iter().map(String::from).collect());
    std::borrow::Cow::Owned(models::ModelList::of(entries))
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

// ---------------------------------------------------------------------------
// Process execution helper
// ---------------------------------------------------------------------------

fn exec_with_env(bin: &Path, args: &[String], extra_env: &[(&str, &str)]) -> anyhow::Result<()> {
    exec_with_env_removing(bin, args, extra_env, &[])
}

/// [`exec_with_env`] for a computed environment.
fn exec_with_owned_env(bin: &Path, args: &[String], env: &[(&str, String)]) -> anyhow::Result<()> {
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    exec_with_env(bin, args, &env)
}

fn exec_with_env_removing(
    bin: &Path,
    args: &[String],
    extra_env: &[(&str, &str)],
    remove_env: &[&str],
) -> anyhow::Result<()> {
    let code = run_with_env_removing(bin, args, extra_env, remove_env);
    hdrm::stop();
    models::finish();
    std::process::exit(code?);
}

/// The agent's process id while it runs, which a SIGTERM to this process
/// is passed on to.
#[cfg(unix)]
static AGENT_PID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
/// Whether a SIGTERM arrived, for one that comes before the agent starts.
#[cfg(unix)]
static TERMINATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Keeps this process alive through a Ctrl-C or closed terminal, which reach the agent too, so it
/// can remove the launch's files and key session afterwards ([`models::finish`]). A SIGTERM, which
/// reaches only this process, is passed on ([`wait_for_agent`]). Handlers, not `SIG_IGN`: a handler
/// resets on exec, so the agent keeps the default behaviour.
#[cfg(unix)]
fn outlive_interrupts() {
    extern "C" fn outlive(_: libc::c_int) {}
    for (signal, handler) in [
        (libc::SIGINT, outlive as extern "C" fn(libc::c_int)),
        (libc::SIGHUP, outlive),
        (libc::SIGTERM, pass_terminate_on),
    ] {
        // SAFETY: the handlers only touch atomics and call kill(2).
        unsafe { libc::signal(signal, handler as libc::sighandler_t) };
    }
}

/// The SIGTERM handler: passes it on to the agent, if one runs.
#[cfg(unix)]
extern "C" fn pass_terminate_on(_: libc::c_int) {
    use std::sync::atomic::Ordering::SeqCst;
    TERMINATED.store(true, SeqCst);
    let pid = AGENT_PID.load(SeqCst);
    if pid > 0 {
        // SAFETY: kill(2) is async-signal-safe.
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
}

/// Windows: tokio's listener installs a console control handler that answers "handled" while a
/// listener exists, which keeps the process alive when Ctrl-C reaches the agent on the same
/// console. One is kept for the life of the process and never polled; registering needs no runtime.
#[cfg(windows)]
fn outlive_interrupts() {
    static LISTENER: std::sync::Mutex<Option<tokio::signal::windows::CtrlC>> =
        std::sync::Mutex::new(None);
    let mut listener = LISTENER.lock().unwrap_or_else(|e| e.into_inner());
    if listener.is_none() {
        match tokio::signal::windows::ctrl_c() {
            Ok(registered) => *listener = Some(registered),
            Err(e) => eprintln!(
                "[llmman] warning: could not register a Ctrl-C handler ({e}): Ctrl-C will end \
                 llmman before it removes its launch files and closes its key session"
            ),
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn outlive_interrupts() {}

/// Runs `cmd` to its end, and, on Unix, passes a SIGTERM to this process
/// on to it (see [`outlive_interrupts`]).
fn wait_for_agent(cmd: &mut Command) -> std::io::Result<std::process::ExitStatus> {
    let mut child = cmd.spawn()?;
    #[cfg(unix)]
    {
        use std::sync::atomic::Ordering::SeqCst;
        let pid = child.id() as libc::pid_t;
        AGENT_PID.store(pid, SeqCst);
        if TERMINATED.load(SeqCst) {
            // SAFETY: `pid` is the child this call spawned and has not reaped.
            unsafe { libc::kill(pid, libc::SIGTERM) };
        }
    }
    let status = child.wait();
    #[cfg(unix)]
    AGENT_PID.store(0, std::sync::atomic::Ordering::SeqCst);
    status
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
    outlive_interrupts();
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

    let status =
        wait_for_agent(&mut cmd).with_context(|| format!("failed to run {}", bin.display()))?;
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

    /// A SIGTERM to llmman reaches the agent, and llmman waits for it.
    #[cfg(unix)]
    #[test]
    fn a_sigterm_is_passed_on_to_the_agent_and_waited_for() {
        use std::os::unix::process::ExitStatusExt;
        use std::sync::atomic::Ordering::SeqCst;
        let waiter = std::thread::spawn(|| {
            let mut cmd = Command::new("sleep");
            cmd.arg("60");
            wait_for_agent(&mut cmd).unwrap()
        });
        let start = std::time::Instant::now();
        while AGENT_PID.load(SeqCst) == 0 {
            assert!(start.elapsed().as_secs() < 10, "no agent started");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        pass_terminate_on(libc::SIGTERM);
        let status = waiter.join().unwrap();
        TERMINATED.store(false, SeqCst);
        assert_eq!(status.signal(), Some(libc::SIGTERM));
        assert_eq!(AGENT_PID.load(SeqCst), 0);
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
            assert!(MODEL_FLAG_FORWARDED.contains(&id));
            assert_eq!(effort_args(id, "high"), ["--reasoning-effort", "high"]);
        }
    }

    #[test]
    fn omp_alone_is_refused_a_provider_and_offered_the_pulled_models_only() {
        let refused = check_provider_supported("omp").unwrap_err().to_string();
        assert!(refused.contains("two vendors"), "{refused}");
        assert!(stored_only_reason("OMP").is_some());
        for id in ["kimi", "gemini", "grok", "pi", "openclaw", "droid"] {
            assert!(stored_only_reason(id).is_none(), "{id}");
        }
    }

    #[test]
    fn droid_is_a_file_configured_model_required_integration() {
        let droid = INTEGRATIONS.iter().find(|i| i.name == "droid").unwrap();
        assert_eq!(droid.binary, "droid");
        assert!(MODEL_FLAG_FORWARDED.contains(&"droid"));
        assert!(CONFIGURED_BY_FILE.contains(&"droid"));
        assert!(check_provider_supported("droid").is_ok());
        assert!(VARIANT_UNSUPPORTED.iter().any(|(name, _)| *name == "droid"));
    }

    /// Every integration that depends on the daemon holding the key must
    /// be a real one: a name nobody can type protects nobody.
    #[test]
    fn every_daemon_key_integration_is_a_real_one() {
        for id in PROVIDER_NEEDS_DAEMON_KEY {
            assert!(
                INTEGRATIONS.iter().any(|i| i.name == *id),
                "{id} is not an integration"
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

    /// No integration is refused for lacking a model: `check_model_flag`
    /// only says when a `--model` after `--` wins, and only for the
    /// launchers that yield to it.
    #[test]
    fn a_missing_model_is_no_longer_refused() {
        let forwarded = vec!["--model".to_string(), "theirs".to_string()];
        for i in INTEGRATIONS {
            check_model_flag(i.name, None, &[]);
            check_model_flag(i.name, Some(" "), &forwarded);
            check_model_flag(&i.name.to_uppercase(), Some("m"), &forwarded);
        }
        // The "yours wins" warning is only claimed for launchers that
        // actually yield to it; dsh has no `--model` flag to yield to.
        for id in MODEL_FLAG_FORWARDED {
            assert!(integration(id).is_some(), "{id} is not an integration");
        }
        assert!(!MODEL_FLAG_FORWARDED.contains(&"dsh"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"claude"));
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
    }

    #[test]
    fn cline_supports_provider_routes() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "cline"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"cline"));
        assert!(PROVIDER_NEEDS_DAEMON_KEY.contains(&"cline"));
        assert!(check_provider_supported("cline").is_ok());
    }

    #[test]
    fn grok_takes_provider_routes() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "grok"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"grok"));
        assert!(check_provider_supported("grok").is_ok());
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
        assert!(declares_context_window("kimi"));
        assert!(declares_context_window("openclaw"));
        // claude is told it through CLAUDE_CODE_MAX_CONTEXT_TOKENS
        assert!(declares_context_window("claude"));
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
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"dsh"));
    }

    // -----------------------------------------------------------------
    // docker-agent
    // -----------------------------------------------------------------

    /// Needs a model, but carries its key in the environment, so it is on
    /// not on `PROVIDER_NEEDS_DAEMON_KEY` —
    /// and not on `MODEL_FLAG_FORWARDED` either, since a forwarded
    /// `--model` is refused rather than obeyed.
    #[test]
    fn docker_agent_is_a_real_model_required_integration_provider_can_drive() {
        let entry = INTEGRATIONS
            .iter()
            .find(|i| i.name == "docker-agent")
            .expect("docker-agent is not an integration");
        assert_eq!(entry.binary, "docker-agent");
        assert!(!MODEL_FLAG_FORWARDED.contains(&"docker-agent"));
        assert!(check_provider_supported("docker-agent").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"docker-agent"));
    }

    /// Muse Code takes its key in `META_API_KEY` and writes no file, so
    /// `--provider` and `--sandbox openshell` both work.
    #[test]
    fn muse_is_an_env_configured_integration_provider_can_drive() {
        let entry = INTEGRATIONS
            .iter()
            .find(|i| i.name == "muse")
            .expect("muse is not an integration");
        assert_eq!(entry.binary, "muse");
        assert!(MODEL_FLAG_FORWARDED.contains(&"muse"));
        assert!(check_provider_supported("muse").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"muse"));
        assert!(!CONFIGURED_BY_FILE.contains(&"muse"));
        assert_eq!(spell_variant("muse", "xhigh").unwrap(), "xhigh");
        assert_eq!(spell_variant("muse", "thinking").unwrap(), "medium");
        let none = spell_variant("muse", "none").unwrap_err().to_string();
        assert!(
            none.contains("muse cannot start at --variant none"),
            "{none}"
        );
    }

    #[test]
    fn a_muse_subcommand_it_does_not_run_is_refused_before_anything_starts() {
        for name in ["muse", "muse-code"] {
            let args = launch_args(&[name, "--model", "m", "--", "usage"]);
            let err = run(&args).unwrap_err().to_string();
            assert!(err.contains("run `muse usage` itself"), "{name}: {err}");
        }
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

    /// The widened levels reach the starting row, so dsh's default effort is
    /// among the efforts it advertises and Grok's metadata agrees with `--effort`.
    #[test]
    fn unknown_levels_with_a_variant_widen_the_starting_row() {
        let list = models::ModelList::of(vec![
            models::ModelEntry::unlisted("m"),
            models::ModelEntry::unlisted("other"),
        ]);
        assert_eq!(list.start().unwrap().thinking, None);
        let widened = unknown_levels_with("xhigh");
        let out = with_widened_start(&list, Some(&widened));
        let start = out.start().unwrap();
        assert!(
            start
                .thinking
                .as_ref()
                .unwrap()
                .iter()
                .any(|l| l == "xhigh"),
            "{:?}",
            start.thinking
        );
        assert_eq!(start.variants(), widened.choices());
        assert_eq!(out.entries()[1], list.entries()[1]);
        // Nothing widened, or levels already known: the list is untouched.
        assert_eq!(*with_widened_start(&list, None), list);
        let known = models::ModelEntry {
            thinking: Some(vec!["low".into()]),
            ..models::ModelEntry::unlisted("m")
        };
        let known = models::ModelList::of(vec![known]);
        assert_eq!(*with_widened_start(&known, Some(&widened)), known);
    }

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
        assert!(err("nope", "high").contains("unknown integration"));
    }

    /// Every integration is either spelled or refused, never silently
    /// launched without its variant.
    #[test]
    fn every_integration_takes_or_refuses_a_variant() {
        for i in INTEGRATIONS {
            let spelled = spell_variant(i.name, "thinking").is_ok();
            let refused = VARIANT_UNSUPPORTED.iter().any(|(id, _)| *id == i.name);
            // These set the effort in their own launcher.
            let configured = matches!(i.name, "opencode" | "qwen" | "dsh" | "muse");
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

    /// Gemini CLI reads its base URL only under a login saved in its home,
    /// which OpenShell never receives.
    #[cfg(not(windows))]
    #[test]
    fn gemini_is_refused_where_the_sandbox_gets_no_files() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: LaunchArgs,
        }
        let args = <Cli as clap::Parser>::try_parse_from([
            "l",
            "gemini",
            "--sandbox",
            "openshell",
            "--model",
            "m",
        ])
        .unwrap()
        .args;
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("openshell cannot run gemini"), "{err}");
    }

    /// A bare `--sandbox` is `--sandbox sbx`, and a named one is kept.
    #[test]
    fn a_bare_sandbox_is_sbx() {
        assert_eq!(launch_args(&["claude"]).sandbox, None);
        assert_eq!(
            launch_args(&["claude", "--sandbox"]).sandbox,
            Some(Sandbox::Sbx)
        );
        assert_eq!(
            launch_args(&["claude", "--sandbox", "docker"]).sandbox,
            Some(Sandbox::Docker)
        );
        // Another flag after a bare one is not taken as its value.
        let args = launch_args(&["claude", "--sandbox", "--model", "qwen3.8", "--", "hi"]);
        assert_eq!(args.sandbox, Some(Sandbox::Sbx));
        assert_eq!(args.model.as_deref(), Some("qwen3.8"));
        assert_eq!(args.extra_args, ["hi"]);
    }

    /// `--optimizer` defaults to none, and takes none, tk, cvmn or hdrm.
    #[test]
    fn the_optimizer_defaults_to_none_and_takes_none_tk_cvmn_or_hdrm() {
        assert_eq!(launch_args(&["claude"]).optimizer, Optimizer::None);
        assert_eq!(
            launch_args(&["claude", "--optimizer", "none"]).optimizer,
            Optimizer::None
        );
        assert_eq!(
            launch_args(&["claude", "--optimizer", "tk"]).optimizer,
            Optimizer::Tk
        );
        assert_eq!(
            launch_args(&["claude", "--optimizer=tk", "--", "-p", "hi"]).optimizer,
            Optimizer::Tk
        );
        assert_eq!(
            launch_args(&["claude", "--optimizer", "cvmn"]).optimizer,
            Optimizer::Cvmn
        );
        assert_eq!(
            launch_args(&["claude", "--optimizer", "hdrm"]).optimizer,
            Optimizer::Hdrm
        );
        // The flag is not an alias for anything after `--`.
        let args = launch_args(&["claude", "--", "--optimizer", "tk"]);
        assert_eq!(args.optimizer, Optimizer::None);
        assert_eq!(args.extra_args, ["--optimizer", "tk"]);
    }

    #[test]
    fn a_bad_optimizer_is_rejected_by_the_parser() {
        #[derive(clap::Parser, Debug)]
        struct Cli {
            #[command(flatten)]
            args: LaunchArgs,
        }
        for bad in ["TK", "turbo", "", "none,tk"] {
            let err = <Cli as clap::Parser>::try_parse_from(["l", "claude", "--optimizer", bad])
                .unwrap_err()
                .to_string();
            assert!(err.contains("--optimizer"), "{bad}: {err}");
        }
        // The help says what each choice does.
        let help = <Cli as clap::CommandFactory>::command()
            .render_long_help()
            .to_string();
        assert!(help.contains("--optimizer <OPTIMIZER>"), "{help}");
        assert!(help.contains("none"), "{help}");
        assert!(help.contains("tk"), "{help}");
        assert!(help.contains("cvmn"), "{help}");
        assert!(help.contains("hdrm"), "{help}");
        assert!(help.contains("[default: none]"), "{help}");
    }

    /// The refusals are `run`'s own errors, so the daemon is never started
    /// and no model is pulled: a launch that got past them would try both.
    #[test]
    fn tk_is_refused_for_other_integrations_before_anything_starts() {
        for name in [
            "codex",
            "aider",
            "goose",
            "cline",
            "gemini",
            "docker-agent",
            "copilot-cli",
        ] {
            let args = launch_args(&[name, "--model", "m", "--optimizer", "tk"]);
            let err = run(&args).unwrap_err().to_string();
            let canonical = canonical_integration(name).unwrap();
            assert!(
                err.contains(&format!("does not work with {canonical}")),
                "{name}: {err}"
            );
            assert!(
                err.contains("only claude, opencode can use it"),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn cvmn_is_refused_for_other_integrations_before_anything_starts() {
        let args = launch_args(&["aider", "--model", "m", "--optimizer", "cvmn"]);
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("does not work with aider"), "{err}");
        assert!(
            err.contains("only claude, codex, opencode can use it"),
            "{err}"
        );
    }

    #[test]
    fn tk_is_refused_with_a_sandbox_before_sbx_is_reached() {
        for sandbox in [
            &["--sandbox"][..],
            &["--sandbox", "sbx"],
            &["--sandbox", "docker"],
        ] {
            let mut argv = vec!["claude", "--model", "m", "--optimizer", "tk"];
            argv.extend_from_slice(sandbox);
            let err = run(&launch_args(&argv)).unwrap_err().to_string();
            assert!(
                err.contains("--optimizer tk does not work with --sandbox"),
                "{err}"
            );
        }
    }

    #[test]
    fn tk_is_refused_beside_claude_flags_it_cannot_honor() {
        let args = launch_args(&[
            "claude",
            "--model",
            "m",
            "--optimizer",
            "tk",
            "--",
            "--settings",
            "mine.json",
        ]);
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("--settings"), "{err}");
    }

    /// The optimizer is not one of the flags `[launch.<integration>]` fills.
    #[test]
    fn configured_defaults_leave_the_optimizer_alone() {
        let (args, _) = applied(&["claude", "--optimizer", "tk"], &sonnet());
        assert_eq!(args.optimizer, Optimizer::Tk);
        let (args, _) = applied(&["claude"], &sonnet());
        assert_eq!(args.optimizer, Optimizer::None);
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
        assert_eq!(canonical_integration("Muse-Code"), Some("muse"));
        assert_eq!(canonical_integration("nonesuch"), None);
        for i in INTEGRATIONS {
            assert_eq!(canonical_integration(i.name), Some(i.name));
        }
    }
}
