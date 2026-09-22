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
use base64::Engine as _;
use clap::Args;

use crate::chat_template::ThinkingControls;
use crate::daemon;
use crate::providers;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
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

    /// Extra arguments forwarded to the integration binary (after --)
    #[arg(last = true, value_name = "ARGS")]
    pub extra_args: Vec<String>,
}

pub fn run(args: &LaunchArgs) -> anyhow::Result<()> {
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

    // Before either arm starts the daemon; see `check_model_flag`.
    check_model_flag(name, args.model.as_deref(), provider, &args.extra_args)?;
    anyhow::ensure!(
        overflow.is_none() || args.model.as_deref().is_some_and(|m| !m.trim().is_empty()),
        "--overflow-model needs --model naming the local model to pair it with"
    );
    // Cline follows the install-on-demand behavior expected by its launch
    // integration. Do this before starting the daemon or pulling a model.
    if name.eq_ignore_ascii_case("cline") {
        ensure_cline_installed()?;
    }
    // The local model's thinking controls (see `opencode_variants`),
    // whether it takes images (see `write_dsh_settings`) and its trained
    // context (see `codex_context_window`); a provider's model has none
    // of these to read.
    let mut thinking = None;
    let mut vision = false;
    let mut context_length = None;
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
            resolve_provider_model(provider, args.model.as_deref(), name, per_request)?
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
                thinking = info.thinking_controls();
                vision = info.vision();
                context_length = info.context_length();
            }
            match overflow {
                // The hosted half is validated and keyed exactly as a
                // bare --provider model would be, then paired with the
                // local model just pulled.
                Some((provider, hosted)) => {
                    check_provider_supported(name)?;
                    let per_request =
                        !PROVIDER_NEEDS_DAEMON_KEY.contains(&name.to_lowercase().as_str());
                    let (remote, api_key) =
                        resolve_provider_model(provider, Some(hosted), name, per_request)?;
                    (crate::hybrid::pair_with_local(&model, &remote)?, api_key)
                }
                None => (model, integration_key()),
            }
        }
    };

    launch(
        name,
        &model,
        &api_key,
        thinking.as_ref(),
        vision,
        context_length,
        &args.extra_args,
    )
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
/// Grok Build and Cline have hosted defaults of their own; without an
/// explicit local model they would send those ids to llmman's endpoint
/// instead.
/// Checked before `ensure_server`, so the refusal costs no daemon start.
const MODEL_REQUIRED: &[&str] = &["qwen", "dsh", "agy", "goose", "grok", "cline", "pi"];

/// Integrations whose launcher yields to a `--model` after `--`, and so
/// warrant the warning below. qwen: `qwen_args` drops its own `--model`
/// when the caller spelled one. goose: its model is `GOOSE_MODEL` in the
/// environment, which goose's own `--model` documents itself as
/// overriding. grok: its argument builder likewise drops the generated
/// `--model`. Cline receives no generated arguments, so its own `--model`
/// also wins over the provider selected in its settings.
/// Not dsh: `dsh_args` does not yield, and dsh takes no
/// `--model` flag at all (its model is the one `write_dsh_settings`
/// records), so telling a dsh user theirs "wins" would be false, and dsh
/// rejects the unknown flag on its own.
const MODEL_FLAG_FORWARDED: &[&str] = &["qwen", "goose", "grok", "cline"];

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
    if !MODEL_REQUIRED.contains(&name.as_str()) {
        return Ok(());
    }
    let Some(model) = model.map(str::trim).filter(|m| !m.is_empty()) else {
        let with_provider = provider.map_or(String::new(), |p| format!(" --provider {p}"));
        anyhow::bail!("{name} needs a model: llmman launch {name}{with_provider} --model <model>");
    };
    if MODEL_FLAG_FORWARDED.contains(&name.as_str()) && has_flag(extra_args, "--model", Some("-m"))
    {
        eprintln!(
            "[llmman] {name}: the --model after -- wins over --model {model}, the one llmman resolved"
        );
    }
    Ok(())
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
/// the daemon. `copilot` takes a model but has no way to carry a key.
/// Refusing is the same call the catalog filter makes — a
/// combination llmman cannot actually drive is absent, not offered and
/// then broken at the first request.
const PROVIDER_UNSUPPORTED: &[(&str, &str)] = &[
    (
        "kimi",
        "it selects its own model rather than taking one from llmman",
    ),
    ("copilot", "it has no way to send a provider API key"),
    ("copilot-cli", "it has no way to send a provider API key"),
    // Its key variable feeds a native Google client, and llmman has not
    // verified that GEMINI_BASE_URL still redirects it here. Getting that
    // wrong sends someone's OpenRouter key to Google, which is a worse
    // outcome than `--provider gemini` not working — the placeholder this
    // used to pass was harmless either way, a real key is not.
    (
        "gemini",
        "llmman cannot confirm it would send the key here rather than to Google",
    ),
    // launch_openclaw passes --custom-model-id only through onboarding,
    // which runs once. Every later launch reuses whatever openclaw.json
    // already names, so the provider reference would never reach the
    // daemon and the session would quietly run on the old model.
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
];

/// Integrations llmman configures through a file on disk. They take a
/// model on every launch, so `--provider` works, but they cannot carry
/// the key: writing a real one into `~/.hermes/config.yaml` would persist
/// a credential, which this feature promises not to do. They rely on
/// `llmman serve` having the variable itself — which it only uses for a
/// daemon nobody else can reach (see `reachable_only_locally`).
const PROVIDER_NEEDS_DAEMON_KEY: &[&str] = &["hermes", "cline", "pi"];

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

/// Validates `--provider`/`--model` against the running daemon's catalog
/// (see [`crate::daemon::provider`]), returning the reference the daemon
/// routes on (see [`crate::providers::REMOTE_PREFIX`]) and the key
/// `integration` should authenticate with.
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
) -> anyhow::Result<(String, String)> {
    // Asked of the daemon, not models.dev: it routes the request, so it
    // is the authority on whether this provider exists — and on whether
    // *it* has the key, which this shell cannot see.
    let entry = daemon::provider(provider)?;

    let model = model
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| {
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

    Ok((providers::format_remote_ref(provider, model), key))
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
        binary: "gh",
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
        name: "grok",
        description: "Grok Build",
        binary: "grok",
    },
];

fn print_integrations() {
    println!("Available integrations:\n");
    for i in INTEGRATIONS {
        // dsh resolves to npx when it isn't installed (see `find_dsh`),
        // and that launch downloads the package before running it.
        let how = match find_integration_binary(i) {
            Some(bin) if bin.file_stem().is_some_and(|s| s == "npx") => " (via npx)",
            Some(_) => "",
            None => " (not installed)",
        };
        println!("  {:<12} {}{}", i.name, i.description, how);
    }
    println!("\nUsage: llmman launch <integration> [--model <model>] [--provider <provider>]");
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

fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if cfg!(windows) {
            for ext in WINDOWS_PATH_EXTS {
                let candidate = dir.join(format!("{binary}.{ext}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        } else {
            let candidate = dir.join(binary);
            if candidate.is_file() {
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

/// The binary `launch` will run for `i`, so the listing does not report
/// as missing what the launcher would find: `PATH`, then what the
/// launcher knows.
fn find_integration_binary(i: &Integration) -> Option<PathBuf> {
    match i.name {
        "opencode" => find_opencode(),
        "qwen" => find_qwen(),
        "dsh" => find_dsh().map(|(bin, _)| bin),
        "goose" => find_goose(),
        "grok" => find_grok(),
        _ => find_on_path(i.binary),
    }
}

// ---------------------------------------------------------------------------
// Cline install on demand
// ---------------------------------------------------------------------------

const CLINE_NPM_INSTALL_ARGS: &[&str] = &["install", "-g", "cline@latest"];
const CLINE_INSTALL_PROMPT: &str = "Cline is not installed. Install with npm? [y/N] ";
const CLINE_INSTALL_CANCELLED: &str = "cline installation cancelled";

/// Offers the official npm install when Cline is missing. This runs before
/// the daemon starts, so declining or lacking npm has no model-pull side
/// effects. Re-resolving from PATH after npm exits also catches a global npm
/// prefix that the current shell does not yet know about.
fn ensure_cline_installed() -> anyhow::Result<()> {
    use std::io::{BufRead, IsTerminal, Write};

    if find_on_path("cline").is_some() {
        return Ok(());
    }
    let npm = find_on_path("npm").ok_or_else(|| {
        anyhow::anyhow!(
            "cline is not installed and npm is not on PATH\n\n\
             Install Node.js from https://nodejs.org/, then re-run:\n  \
             llmman launch cline"
        )
    })?;
    anyhow::ensure!(
        std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
        "cline is not installed\n\nInstall it with:\n  npm install -g cline@latest\n\n\
         Then re-run:\n  llmman launch cline"
    );

    eprint!("{CLINE_INSTALL_PROMPT}");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    anyhow::ensure!(accepts_install(&answer), CLINE_INSTALL_CANCELLED);

    eprintln!("\nInstalling Cline...");
    let status = Command::new(&npm)
        .args(CLINE_NPM_INSTALL_ARGS)
        .status()
        .with_context(|| format!("failed to run {}", npm.display()))?;
    anyhow::ensure!(status.success(), "failed to install cline: {status}");
    let cline = find_on_path("cline").ok_or_else(|| {
        anyhow::anyhow!(
            "cline was installed but the binary was not found on PATH\n\n\
         You may need to restart your shell"
        )
    })?;
    let version = Command::new(&cline)
        .arg("--version")
        .status()
        .with_context(|| format!("failed to run {} --version", cline.display()))?;
    anyhow::ensure!(
        version.success(),
        "cline was installed but failed to start ({version})"
    );
    eprintln!("Cline installed successfully\n");
    Ok(())
}

fn accepts_install(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
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
fn launch(
    name: &str,
    model: &str,
    api_key: &str,
    thinking: Option<&ThinkingControls>,
    vision: bool,
    context_length: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    match name.to_lowercase().as_str() {
        "claude" => launch_claude(model, api_key, extra_args),
        "opencode" => launch_opencode(model, api_key, thinking, vision, extra_args),
        "codex" => launch_codex(model, api_key, vision, context_length, extra_args),
        "pi" => launch_pi(model, thinking, vision, context_length, extra_args),
        "cline" => launch_cline(model, extra_args),
        "aider" => launch_aider(model, api_key, extra_args),
        "copilot" | "copilot-cli" => launch_copilot(model, extra_args),
        "kimi" => launch_simple("kimi", model, extra_args),
        "gemini" => launch_gemini(model, api_key, extra_args),
        "agy" => launch_agy(model, api_key, extra_args),
        "hermes" => launch_hermes(model, vision, extra_args),
        "openclaw" => launch_openclaw(model, extra_args),
        "qwen" => launch_qwen(model, api_key, vision, extra_args),
        "dsh" => launch_dsh(model, api_key, vision, extra_args),
        "goose" => launch_goose(model, api_key, extra_args),
        "grok" => launch_grok(model, api_key, extra_args),
        other => anyhow::bail!(
            "unknown integration {:?}\nRun 'llmman launch' without arguments to list supported integrations.",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// Per-integration launchers
// ---------------------------------------------------------------------------

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
fn launch_claude(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("claude").ok_or_else(|| anyhow::anyhow!("claude is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    let server = daemon::server();
    exec_with_env(
        &bin,
        &args,
        &[
            ("ANTHROPIC_BASE_URL", server.as_str()),
            ("ANTHROPIC_API_KEY", api_key),
        ],
    )
}

/// opencode: a JSON config via OPENCODE_CONFIG_CONTENT pointing at our
/// /v1 endpoint, with the model's thinking variants and, for a vision
/// model, image input.
fn launch_opencode(
    model: &str,
    api_key: &str,
    thinking: Option<&ThinkingControls>,
    vision: bool,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_opencode().ok_or_else(|| anyhow::anyhow!("opencode is not installed"))?;

    let effective_model = if model.is_empty() { "default" } else { model };
    let config = opencode_config(
        &daemon::server(),
        effective_model,
        api_key,
        &opencode_variants(thinking),
        vision,
    );

    exec_with_env(&bin, extra_args, &[("OPENCODE_CONFIG_CONTENT", &config)])
}

/// The choices offered when the model's template could not be read (a
/// provider's model): thinking off, then the levels every wire accepts
/// (`anthropic::portable_efforts`).
const PORTABLE_THINKING_LEVELS: &[&str] = &["none", "low", "medium", "high"];

/// opencode's `variants` for the model, in cycle order (`variant_cycle`,
/// ctrl+t by default): the template's own choices (see
/// [`ThinkingControls::choices`]), or [`PORTABLE_THINKING_LEVELS`] without
/// a template. A model that does not think gets none. Each variant is the
/// request options `@ai-sdk/openai-compatible` sends: `reasoningEffort`
/// as `reasoning_effort`, other keys verbatim. opencode derives variants
/// only for models it knows from models.dev, so without these a local
/// model has nothing to cycle.
fn opencode_variants(
    thinking: Option<&ThinkingControls>,
) -> Vec<(&'static str, serde_json::Value)> {
    let choices = match thinking {
        Some(controls) => controls.choices(),
        None => PORTABLE_THINKING_LEVELS.to_vec(),
    };
    choices
        .into_iter()
        .map(|choice| {
            let options = match choice {
                "thinking" => {
                    serde_json::json!({ "chat_template_kwargs": { "enable_thinking": true } })
                }
                level => serde_json::json!({ "reasoningEffort": level }),
            };
            (choice, options)
        })
        .collect()
}

/// Finds opencode on `PATH`, then where its installers put it. The second
/// check finds a fresh install that this process's `PATH` doesn't include
/// yet.
fn find_opencode() -> Option<PathBuf> {
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

/// The `OPENCODE_CONFIG_CONTENT` for `model` at `server`. Structs rather
/// than `json!`, whose map sorts keys: opencode cycles variants in the
/// order listed. No variants leaves the key out.
fn opencode_config(
    server: &str,
    model: &str,
    api_key: &str,
    variants: &[(&'static str, serde_json::Value)],
    vision: bool,
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
        variants: &'a [(&'static str, serde_json::Value)],
        #[serde(skip_serializing_if = "Option::is_none")]
        modalities: Option<Modalities>,
        #[serde(skip_serializing_if = "Option::is_none")]
        attachment: Option<bool>,
    }
    #[derive(serde::Serialize)]
    struct Modalities {
        input: &'static [&'static str],
        output: &'static [&'static str],
    }

    // Declare image input for a vision model so opencode will attach
    // images; a text-only model gets neither key.
    let modalities = vision.then_some(Modalities {
        input: &["text", "image"],
        output: &["text"],
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
                        modalities,
                        attachment: vision.then_some(true),
                    },
                )],
            },
        },
        model: format!("ollama/{model}"),
    };
    serde_json::to_string(&config).expect("opencode config serializes")
}

/// pi: register llmman as an OpenAI-compatible provider in `models.json`
/// and point `settings.json` at it. pi runs with the caller's own
/// arguments and nothing else — `settings.json` already selects the
/// model, and a key on argv would be readable in `ps`.
///
/// pi is in [`PROVIDER_NEEDS_DAEMON_KEY`] for that reason: a `--provider`
/// credential stays with the daemon, and the stored `apiKey` is the
/// literal placeholder codex and hermes also write. An environment
/// reference (`"$VAR"`, which pi does interpolate) would read as
/// "unresolved" outside `llmman launch` and take the provider down.
fn launch_pi(
    model: &str,
    thinking: Option<&ThinkingControls>,
    vision: bool,
    context_length: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("pi").ok_or_else(|| anyhow::anyhow!("pi is not installed"))?;
    write_pi_config(model, thinking, vision, context_length)?;
    exec_with_env(&bin, extra_args, &[])
}

/// The provider key llmman owns in pi's `models.json`.
const PI_PROVIDER: &str = "llmman";

/// pi's config directory: `PI_CODING_AGENT_DIR`, else `~/.pi/agent`.
///
/// The home half is [`cline_dir`]'s, not [`qwen_home`]'s: pi is node as
/// Cline is, so `os.homedir()` reads `USERPROFILE` on Windows — see
/// `cline_dir`'s own doc comment for what disagreeing there cost. The
/// `~` handling is qwen's, for the quoted export that leaves one behind.
fn pi_agent_dir() -> anyhow::Result<PathBuf> {
    let home = || node_home_dir().context("no home directory");
    match std::env::var("PI_CODING_AGENT_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        Some(dir) if !dir.trim().starts_with('~') => Ok(PathBuf::from(dir.trim())),
        Some(dir) => Ok(expand_tilde(dir.trim(), &home()?)),
        None => Ok(home()?.join(".pi").join("agent")),
    }
}

/// Writes pi's `models.json` provider and points `settings.json` at it.
/// Both go through [`write_json_merged`], which qwen writes through too.
fn write_pi_config(
    model: &str,
    thinking: Option<&ThinkingControls>,
    vision: bool,
    context_length: Option<u64>,
) -> anyhow::Result<()> {
    let dir = pi_agent_dir()?;
    let entry = pi_model_entry(model, thinking, vision, context_length);
    write_json_merged(&dir.join("models.json"), "pi", |existing| {
        pi_models_merged(existing, &daemon::server(), &entry)
    })?;
    write_json_merged(&dir.join("settings.json"), "pi", |existing| {
        pi_settings_merged(existing, model)
    })
}

/// The model pi is told about: what it can take in, whether it thinks,
/// and how much it can hold. Every example in pi's own `models.md`
/// declares these, and what it assumes for an entry that omits them is
/// not written down, so they are stated rather than left to it.
fn pi_model_entry(
    model: &str,
    thinking: Option<&ThinkingControls>,
    vision: bool,
    context_length: Option<u64>,
) -> serde_json::Value {
    let input: &[&str] = if vision {
        &["text", "image"]
    } else {
        &["text"]
    };
    let mut entry = serde_json::json!({
        "id": model,
        "input": input,
    });
    if thinking.is_some_and(|t| t.thinks) {
        entry["reasoning"] = serde_json::json!(true);
    }
    if let Some(context) = context_length {
        entry["contextWindow"] = serde_json::json!(context);
    }
    entry
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
    let mut root = existing.clone();
    let kept: Vec<serde_json::Value> = root
        .get("providers")
        .and_then(|p| p.get(PI_PROVIDER))
        .and_then(|p| p.get("models"))
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter(|m| m.get("id") != entry.get("id"))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let mut models = kept;
    models.push(entry.clone());

    // Both levels go through object_under: a file may legitimately hold
    // `"providers": []`, and indexing that with a key panics.
    let root_map = match root.as_object_mut() {
        Some(map) => map,
        None => return serde_json::json!({}),
    };
    let provider = object_under(object_under(root_map, "providers"), PI_PROVIDER);
    provider.insert("baseUrl".into(), serde_json::json!(format!("{server}/v1")));
    provider
        .entry("api")
        .or_insert_with(|| serde_json::json!("openai-completions"));
    // A literal, not a `"$VAR"` reference pi would interpolate: see
    // launch_pi's own doc comment.
    provider
        .entry("apiKey")
        .or_insert_with(|| serde_json::json!("llmman"));
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

/// codex: set OPENAI_API_KEY=llmman and write ~/.codex/config.toml with the
/// ollama provider pointing at our /v1 endpoint.
fn launch_codex(
    model: &str,
    api_key: &str,
    vision: bool,
    context_length: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    // Write codex config
    write_codex_config(model, vision, context_length)?;

    // Regression: this used to pass a bare PathBuf::from("codex") straight
    // to exec_with_env instead of resolving it via find_on_path like every
    // other integration here does. That happened to work on Unix (bare
    // relative names go through $PATH search via execvp with no extension
    // needed), but on Windows, Command::status() calls CreateProcess
    // directly (not cmd.exe), which — unlike a shell — does not consult
    // PATHEXT to try .cmd/.bat alternatives for an extensionless name: it
    // only ever auto-appends a single ".exe". Since `npm install -g
    // @openai/codex` installs a "codex.cmd" shim on Windows, not a
    // "codex.exe", every real Windows codex launch failed with "program
    // not found" — a real E2E-verified failure, not a theoretical one.
    let bin = find_on_path("codex").ok_or_else(|| anyhow::anyhow!("codex is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    // codex profile flag
    args.extend(["--profile".to_string(), "llmman".to_string()]);
    args.extend_from_slice(extra_args);

    exec_with_env(&bin, &args, &[("OPENAI_API_KEY", api_key)])
}

/// Writes codex's `llmman` profile.
///
/// Codex 0.134+ dropped support for `--profile <name>` reading a
/// `[profiles.<name>]` table out of `config.toml`: it now only overlays a
/// sibling `~/.codex/<name>.config.toml`, using top-level keys instead of a
/// `[profiles.<name>]` wrapper (see
/// <https://developers.openai.com/codex/config-advanced#profiles>). An
/// older llmman wrote the now-unsupported `[profiles.llmman]` form directly
/// into `config.toml`, which current codex refuses to start with at all
/// ("cannot be used while config.toml contains legacy ... table") — so any
/// leftover copy of that table is stripped from `config.toml` first, then
/// the real settings are (re)written to the profile overlay file codex
/// actually reads.
fn write_codex_config(
    model: &str,
    vision: bool,
    context_length: Option<u64>,
) -> anyhow::Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let config_dir = home.join(".codex");
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if let Ok(existing) = std::fs::read_to_string(&config_path) {
        if existing.contains("[profiles.llmman]") {
            std::fs::write(&config_path, strip_legacy_llmman_profile(&existing))?;
        }
    }

    // Without a model there is nothing to describe; codex keeps its defaults.
    let catalog_path = config_dir.join("llmman-model.json");
    let catalog = (!model.is_empty()).then(|| {
        let context_window =
            codex_context_window(super::serve::context_length_from_env(), context_length);
        write_codex_file(
            &catalog_path,
            &codex_model_catalog(model, vision, context_window),
        )
        .map(|()| catalog_path.clone())
    });
    let catalog = catalog.transpose()?;

    let profile_path = config_dir.join("llmman.config.toml");
    write_codex_file(
        &profile_path,
        &codex_profile(&daemon::server(), catalog.as_deref()),
    )
}

/// Writes `contents` to `path` unless it already holds exactly that.
fn write_codex_file(path: &Path, contents: &str) -> anyhow::Result<()> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(contents) {
        return Ok(());
    }
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))
}

/// The catalog's `context_window` with neither `LLMMAN_CONTEXT_LENGTH` nor
/// a trained context; ollama's fallback too.
const CODEX_FALLBACK_CONTEXT_WINDOW: u64 = 128_000;

/// What codex compacts against: a positive `LLMMAN_CONTEXT_LENGTH`
/// (`env`), the `--ctx-size` the daemon serves, else the model's
/// `trained` context, else [`CODEX_FALLBACK_CONTEXT_WINDOW`].
fn codex_context_window(env: Option<u32>, trained: Option<u64>) -> u64 {
    env.filter(|n| *n > 0)
        .map(u64::from)
        .or(trained)
        .unwrap_or(CODEX_FALLBACK_CONTEXT_WINDOW)
}

/// The `model_catalog_json` for `model`, declaring its image input in
/// `input_modalities`. The other fields are ones codex requires, valued
/// as ollama's `buildCodexModelEntry` does.
fn codex_model_catalog(model: &str, vision: bool, context_window: u64) -> String {
    let input: &[&str] = if vision {
        &["text", "image"]
    } else {
        &["text"]
    };
    let entry = serde_json::json!({
        "slug": model,
        "display_name": model,
        "context_window": context_window,
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "truncation_policy": { "mode": "bytes", "limit": 10000 },
        "input_modalities": input,
        "base_instructions": "",
        "support_verbosity": true,
        "default_verbosity": "low",
        "supports_parallel_tool_calls": false,
        "supports_reasoning_summaries": false,
        "supported_reasoning_levels": [],
        "experimental_supported_tools": [],
    });
    let catalog = serde_json::json!({ "models": [entry] });
    serde_json::to_string_pretty(&catalog).expect("codex catalog serializes") + "\n"
}

/// The contents of `~/.codex/llmman.config.toml`: a provider of llmman's
/// own rather than `openai_base_url` on codex's built-in one, which codex
/// treats as WebSocket-capable and so opened every session with five
/// failed `ws://` attempts (~6s of "Reconnecting...") before HTTP.
fn codex_profile(server: &str, catalog: Option<&Path>) -> String {
    // A JSON string is also a valid TOML string.
    let catalog = catalog
        .map(|p| {
            let quoted = serde_json::Value::from(p.display().to_string());
            format!("model_catalog_json = {quoted}\n")
        })
        .unwrap_or_default();
    format!(
        "# Written by `llmman launch codex`; edits are overwritten.\n\
         model_provider = \"llmman\"\n\
         {catalog}\
         \n\
         [model_providers.llmman]\n\
         name = \"llmman\"\n\
         base_url = \"{server}/v1\"\n\
         env_key = \"OPENAI_API_KEY\"\n\
         wire_api = \"responses\"\n\
         supports_websockets = false\n"
    )
}

/// Removes a `[profiles.llmman]` table (and everything up to the next
/// top-level `[...]` header or end of file) from `config.toml`'s text —
/// the shape an older llmman wrote there, now rejected by current codex.
/// Line-based rather than a real TOML parser: this only ever needs to
/// undo llmman's own prior output, not handle arbitrary user TOML.
fn strip_legacy_llmman_profile(existing: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed == "[profiles.llmman]" {
            skipping = true;
            continue;
        }
        if skipping && trimmed.starts_with('[') {
            skipping = false;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// aider: set OPENAI_API_KEY and OPENAI_BASE_URL.
fn launch_aider(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let base_url = format!("{}/v1", daemon::server());
    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), format!("openai/{model}")]);
    }
    args.extend(["--openai-api-base".to_string(), base_url.clone()]);
    args.extend_from_slice(extra_args);

    exec_with_env(
        &PathBuf::from("aider"),
        &args,
        &[
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url.as_str()),
        ],
    )
}

/// copilot: passes COPILOT_PROVIDER_BASE_URL via env.
fn launch_copilot(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin =
        find_on_path("gh").ok_or_else(|| anyhow::anyhow!("gh (GitHub CLI) is not installed"))?;

    let base_url = format!("{}/v1", daemon::server());
    let mut args = vec!["copilot".to_string()];
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    exec_with_env(&bin, &args, &[("COPILOT_PROVIDER_BASE_URL", &base_url)])
}

/// gemini: set GOOGLE_GENAI_BASE_URL pointing at our Anthropic-compatible endpoint.
fn launch_gemini(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("gemini").ok_or_else(|| anyhow::anyhow!("gemini is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);

    let base_url = format!("{}/v1", daemon::server());
    exec_with_env(
        &bin,
        &args,
        &[
            ("GEMINI_BASE_URL", base_url.as_str()),
            ("GEMINI_API_KEY", api_key),
        ],
    )
}

/// AGY speaks Gemini's native generation protocol. The encoded model in the
/// base URL is llmman's routing instruction; AGY also makes auxiliary calls
/// with its own hard-coded model names, so the server deliberately ignores
/// the model segment AGY appends and sends every call to the model selected
/// here.
fn launch_agy(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("agy").ok_or_else(|| anyhow::anyhow!("agy is not installed"))?;
    anyhow::ensure!(
        !extra_args
            .iter()
            .any(|arg| matches!(arg.split('=').next(), Some("--gemini_dir" | "-gemini_dir"))),
        "llmman manages AGY’s --gemini_dir"
    );
    let gemini_dir = agy_settings_dir()?;
    write_agy_settings_at(&gemini_dir)?;
    let mut args = vec![format!("--gemini_dir={}", gemini_dir.display())];
    args.extend_from_slice(extra_args);

    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(model.as_bytes());
    let base_url = format!("{}/gemini/{encoded}", daemon::server());
    exec_with_env(
        &bin,
        &args,
        &[
            ("GOOGLE_GEMINI_BASE_URL", base_url.as_str()),
            ("GEMINI_API_KEY", api_key),
            // AGY prefers GOOGLE_API_KEY when both names exist. Override it
            // too so an unrelated key inherited from the shell cannot bypass
            // the credential llmman selected for this request.
            ("GOOGLE_API_KEY", api_key),
        ],
    )
}

fn agy_settings_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".gemini")
        .join("llmman"))
}

fn write_agy_settings_at(gemini_dir: &Path) -> anyhow::Result<()> {
    let settings_path = gemini_dir.join("antigravity-cli").join("settings.json");
    std::fs::create_dir_all(settings_path.parent().expect("settings file has a parent"))?;
    crate::fsutil::write_atomic(&settings_path, b"{\n  \"modelProvider\": \"gemini\"\n}\n")
        .with_context(|| format!("write {}", settings_path.display()))
}

/// Generic launcher: just set OLLAMA_HOST and run the binary.
fn launch_simple(binary: &str, _model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path(binary).ok_or_else(|| anyhow::anyhow!("{binary} is not installed"))?;
    let server = daemon::server();
    exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())])
}

/// hermes: writes its own `~/.hermes/config.yaml` provider entry
/// pointing at our /v1 endpoint, skipping the messaging-gateway/
/// desktop-build setup a full wizard would also handle, which llmman's
/// own launch has no equivalent for.
fn launch_hermes(model: &str, vision: bool, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("hermes").ok_or_else(|| anyhow::anyhow!("hermes is not installed"))?;
    write_hermes_config(if model.is_empty() { "default" } else { model }, vision)?;
    exec_with_env(&bin, extra_args, &[])
}

/// Matches hermes's own home-directory resolution: `$HERMES_HOME` if
/// set, else `%LOCALAPPDATA%\hermes` on Windows (real observed failure
/// otherwise — a config written to `~/.hermes` there is silently
/// ignored, since that's not where real hermes looks on Windows at all:
/// "No inference provider configured"), else `~/.hermes` everywhere else.
fn hermes_home() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = std::env::var("HERMES_HOME") {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    if cfg!(windows) {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            if !local_app_data.trim().is_empty() {
                return Ok(PathBuf::from(local_app_data).join("hermes"));
            }
        }
        let home = dirs::home_dir().context("no home directory")?;
        return Ok(home.join("AppData").join("Local").join("hermes"));
    }
    let home = dirs::home_dir().context("no home directory")?;
    Ok(home.join(".hermes"))
}

/// Only overwrites the `model:`/`providers:` top-level blocks this
/// itself writes — everything else in an existing `config.yaml` (other
/// providers, toolsets, etc.) is preserved, the same way
/// `write_codex_config`/`strip_legacy_llmman_profile` avoid clobbering
/// unrelated `config.toml` content.
fn write_hermes_config(model: &str, vision: bool) -> anyhow::Result<()> {
    let config_dir = hermes_home()?;
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("config.yaml");

    let existing = std::fs::read_to_string(&config_path).unwrap_or_default();
    let preserved =
        strip_yaml_top_level_key(&strip_yaml_top_level_key(&existing, "model"), "providers");
    let ours = hermes_config_blocks(model, &format!("{}/v1", daemon::server()), vision);
    std::fs::write(&config_path, format!("{preserved}{ours}"))?;
    Ok(())
}

/// The `model:`/`providers:` blocks [`write_hermes_config`] owns. A
/// vision model gets `model.supports_vision`, the override hermes's
/// image routing reads before its own catalog, which lists no local
/// model and so routes every image through a describe-it tool instead.
fn hermes_config_blocks(model: &str, base_url: &str, vision: bool) -> String {
    // Double-quoted (not bare) so a model name that happens to be a YAML
    // keyword (`null`, `true`, ...) or contain metacharacters (`:`, `#`,
    // ...) still parses back as the literal string it is.
    let model = yaml_quote(model);
    let base_url = yaml_quote(base_url);
    let vision = if vision {
        "  supports_vision: true\n"
    } else {
        ""
    };
    format!(
        "model:\n  provider: llmman\n  default: {model}\n  base_url: {base_url}\n  api_key: llmman\n{vision}\
         providers:\n  llmman:\n    name: llmman\n    api: {base_url}\n    default_model: {model}\n    models:\n      - {model}\n"
    )
}

/// Renders `s` as a double-quoted YAML scalar, escaping backslashes and
/// double quotes — enough to keep any value we generate (a model name, a
/// URL) a literal string regardless of YAML keywords or metacharacters
/// it might contain.
fn yaml_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Removes a top-level YAML key (`<key>:` at column 0) and every line
/// indented under it, up to the next column-0 line or EOF — the YAML
/// (indentation-block) equivalent of `strip_legacy_llmman_profile`'s
/// TOML `[...]`-header block removal. Only ever needs to undo llmman's
/// own prior writes below, not handle arbitrary user YAML.
fn strip_yaml_top_level_key(existing: &str, key: &str) -> String {
    let header = format!("{key}:");
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        // Blank lines and column-0 `#` comments don't belong to any block
        // on their own — don't let either reset `skipping` (that would
        // leak the rest of the removed block into `out`) or fall through
        // to it either way.
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }
        if !line.starts_with([' ', '\t']) {
            skipping = line.trim_end() == header;
            if skipping {
                continue;
            }
        } else if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// openclaw's own onboarding independently re-verifies/pulls whatever
/// `--custom-model-id` names against its configured endpoint, and
/// mishandles llmman's own `docker.io/ai/<name>` form: it treats
/// "docker.io/ai/" as a container registry path (real observed failure:
/// "pull failed: copy image: docker.io/ai/0.8b ... requested access to
/// the resource is denied", mangling "qwen3.5:0.8b" down to "0.8b" in
/// the process). Stripping that prefix back to the bare short name —
/// what a real user would actually type — matches what its pull
/// verification expects. `"default"` when there's nothing left to strip
/// to (no `--model` given at all).
fn openclaw_model_id(model: &str) -> &str {
    let bare = model.strip_prefix("docker.io/ai/").unwrap_or(model);
    if bare.is_empty() {
        "default"
    } else {
        bare
    }
}

/// openclaw: runs its non-interactive onboarding (once) against our
/// /v1 endpoint, then hands off to it directly. The real gateway/TUI/
/// channel-setup lifecycle a full setup wizard also manages is left to
/// openclaw's own defaults.
fn launch_openclaw(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin =
        find_on_path("openclaw").ok_or_else(|| anyhow::anyhow!("openclaw is not installed"))?;

    // Matches openclaw.go's own onboarded() check: current config path,
    // or the legacy pre-rename one.
    let onboarded = dirs::home_dir().is_some_and(|h| {
        h.join(".openclaw").join("openclaw.json").exists()
            || h.join(".clawdbot").join("clawdbot.json").exists()
    });
    if !onboarded {
        let effective_model = openclaw_model_id(model);
        let status = Command::new(&bin)
            .args([
                "onboard",
                "--non-interactive",
                "--accept-risk",
                "--auth-choice",
                "ollama",
                "--custom-base-url",
                &format!("{}/v1", daemon::server()),
                "--custom-model-id",
                effective_model,
                "--skip-health",
                "--skip-channels",
                "--skip-skills",
            ])
            .status()
            .with_context(|| format!("failed to run {}", bin.display()))?;
        anyhow::ensure!(status.success(), "openclaw onboarding failed");
    }

    exec_with_env(&bin, extra_args, &[])
}

/// qwen: Qwen Code's OpenAI-compatible mode, pointed at our /v1 by the
/// command line, the environment and its settings file together, since
/// Qwen Code reads the three in a different order for each value:
/// `--auth-type` and `--model` win on the command line, the base URL is
/// won by a `modelProviders` entry for the model (`resolveModelConfig` in
/// its `packages/core/src/models/modelConfigResolver.ts`), which is what
/// `write_qwen_settings` is for, and the key stays in the environment,
/// named in that entry as `LLMMAN_API_KEY` so llmman's entry is told from
/// any other. Ollama's `cmd/launch/qwen.go` does the same three. A
/// `--model` after `--` is the one Qwen Code uses, so the settings and
/// `OPENAI_MODEL` follow it.
fn launch_qwen(
    model: &str,
    api_key: &str,
    vision: bool,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_qwen().ok_or_else(|| anyhow::anyhow!("qwen is not installed"))?;
    let (model, vision) = qwen_model_and_vision(model, vision, extra_args);
    // After the lookup, so nothing is written for an integration that is
    // not there; `check_model_flag` has made sure there is a model.
    write_qwen_settings(model, vision)?;

    let base_url = format!("{}/v1", daemon::server());
    let mut env = vec![
        ("OPENAI_BASE_URL", base_url.as_str()),
        ("OPENAI_API_KEY", api_key),
        (QWEN_ENV_KEY, api_key),
        ("OPENAI_MODEL", model),
    ];
    // A shim the fallback found is a `#!/usr/bin/env node` script whose
    // `node` sits beside it, so its directory goes on the child's `PATH`.
    let path = path_with_dir_prepended(bin.parent(), &std::env::var_os("PATH").unwrap_or_default())
        .map(|p| p.to_string_lossy().into_owned());
    if let Some(path) = &path {
        env.push(("PATH", path.as_str()));
    }
    exec_with_env(&bin, &qwen_args(model, extra_args), &env)
}

/// The model Qwen Code will use — a `--model` after `--` wins — and
/// whether to declare its image input, which `vision` answers for
/// `model` alone. `model` arrives resolved and the forwarded name as
/// typed, so they are compared resolved: `gemma4:12b` is the
/// `docker.io/ai/gemma4:12b` it names.
fn qwen_model_and_vision<'a>(
    model: &'a str,
    vision: bool,
    extra_args: &'a [String],
) -> (&'a str, bool) {
    let forwarded = forwarded_model(extra_args);
    let same = forwarded.is_none_or(|f| {
        crate::shortnames::resolve_ollama_api(f).is_ok_and(|resolved| resolved == model)
    });
    (forwarded.unwrap_or(model), vision && same)
}

/// `path_var` with `dir` in front, or `None` when it is there already or
/// there is no `dir`. Empty components go: one is how an unset `PATH`
/// arrives, and on POSIX it means the working directory.
fn path_with_dir_prepended(
    dir: Option<&Path>,
    path_var: &std::ffi::OsStr,
) -> Option<std::ffi::OsString> {
    let dir = dir?;
    let mut components: Vec<PathBuf> = std::env::split_paths(path_var)
        .filter(|d| !d.as_os_str().is_empty())
        .collect();
    if components.iter().any(|d| d == dir) {
        return None;
    }
    components.insert(0, dir.to_path_buf());
    std::env::join_paths(components).ok()
}

/// The value of the last `--model`/`-m` after `--`, as a word or
/// `=`-joined, the forms `has_flag` takes; yargs keeps the last too.
fn forwarded_model(extra_args: &[String]) -> Option<&str> {
    let mut found = None;
    let mut args = extra_args.iter().map(String::as_str);
    while let Some(a) = args.next() {
        match a {
            "--model" | "-m" => found = args.next().or(found),
            _ => {
                if let Some(v) = a.strip_prefix("--model=").or_else(|| a.strip_prefix("-m=")) {
                    found = Some(v);
                }
            }
        }
    }
    found.filter(|v| !v.is_empty())
}

/// `--auth-type openai --model <model>` ahead of the caller's own
/// arguments, each dropped when the caller already passed it after `--`:
/// Qwen Code 0.22.3 crashes on either flag repeated (a `toLowerCase`
/// TypeError) rather than taking the last one, and a caller who spelled
/// out an auth type meant it. `--authType` is checked too — yargs accepts
/// a flag's camelCase spelling as well.
fn qwen_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 4);
    if !has_flag(extra_args, "--auth-type", Some("--authType")) {
        args.extend(["--auth-type".to_string(), "openai".to_string()]);
    }
    if !has_flag(extra_args, "--model", Some("-m")) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// `PATH`, then the installers' own targets; see `qwen_fallback_paths`.
fn find_qwen() -> Option<PathBuf> {
    find_on_path("qwen").or_else(|| qwen_fallback_paths().into_iter().find(|p| p.is_file()))
}

/// Where a Qwen Code install lands that a process without the user's
/// login shell does not see: the standalone installer's `~/.local/bin`
/// and, on Windows, its `%LOCALAPPDATA%\qwen-code\bin`
/// (`Get-QwenInstallBinDir` in Qwen Code's
/// `scripts/installation/install-qwen-standalone.ps1`); the
/// `~/.npm-global` prefix its older npm installer set; any node under
/// `~/.nvm` that has it; Homebrew's prefixes and `/usr/local/bin`; and the rest of
/// what ollama's `cmd/launch/qwen.go` probes, `~/.cargo/bin`, macOS's
/// `~/Library/Application Support/qwen/bin`, and on Windows npm's global
/// directory under both `%APPDATA%` and `%LOCALAPPDATA%`,
/// `%LOCALAPPDATA%\Programs\qwen` and `%APPDATA%\qwen\bin`.
fn qwen_fallback_paths() -> Vec<PathBuf> {
    let home = dirs::home_dir();
    let mut paths = Vec::new();
    if cfg!(windows) {
        // Blank counts as unset, or the candidate would be relative.
        let roaming = std::env::var_os("APPDATA")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join("AppData").join("Roaming")));
        let local = std::env::var_os("LOCALAPPDATA")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join("AppData").join("Local")));
        let dirs = [
            roaming.as_ref().map(|d| d.join("npm")),
            local.as_ref().map(|d| d.join("npm")),
            local.as_ref().map(|d| d.join("qwen-code").join("bin")),
            local.as_ref().map(|d| d.join("Programs").join("qwen")),
            roaming.as_ref().map(|d| d.join("qwen").join("bin")),
        ];
        for dir in dirs.into_iter().flatten() {
            paths.extend(
                WINDOWS_PATH_EXTS
                    .iter()
                    .map(|ext| dir.join(format!("qwen.{ext}"))),
            );
        }
        return paths;
    }
    if let Some(h) = &home {
        paths.push(h.join(".local").join("bin").join("qwen"));
        paths.push(h.join(".npm-global").join("bin").join("qwen"));
        paths.push(h.join(".cargo").join("bin").join("qwen"));
        if cfg!(target_os = "macos") {
            paths.push(
                h.join("Library")
                    .join("Application Support")
                    .join("qwen")
                    .join("bin")
                    .join("qwen"),
            );
        }
        paths.extend(nvm_qwen(h));
    }
    if cfg!(target_os = "macos") {
        paths.push(PathBuf::from("/opt/homebrew/bin/qwen"));
    } else {
        paths.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin/qwen"));
    }
    paths.push(PathBuf::from("/usr/local/bin/qwen"));
    paths
}

/// The `qwen` under any node version in `~/.nvm`, the way ollama's
/// `cmd/launch/qwen.go` globs for it.
fn nvm_qwen(home: &Path) -> Option<PathBuf> {
    std::fs::read_dir(home.join(".nvm").join("versions").join("node"))
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("bin").join("qwen"))
        .find(|p| p.is_file())
}

/// `$QWEN_HOME` if set, else `~/.qwen`: `Storage.getGlobalQwenDir` in
/// Qwen Code's `packages/core/src/config/storage.ts`. Qwen Code can also
/// take it from `~/.qwen/.env`; that is left to the user's shell.
fn qwen_home() -> anyhow::Result<PathBuf> {
    let home = || dirs::home_dir().context("no home directory");
    match std::env::var("QWEN_HOME").ok().filter(|d| !d.is_empty()) {
        Some(dir) if !dir.starts_with('~') => Ok(PathBuf::from(dir)),
        Some(dir) => Ok(expand_tilde(&dir, &home()?)),
        None => Ok(home()?.join(".qwen")),
    }
}

/// A leading `~` is `home`, as Qwen Code's `Storage.resolvePath` reads
/// it; a quoted export leaves it for the program to expand.
fn expand_tilde(dir: &str, home: &Path) -> PathBuf {
    match dir.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            home.join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(dir),
    }
}

/// Records llmman as the `openai` provider for `model` in Qwen Code's
/// `settings.json`, as `write_codex_config` and `write_hermes_config` do
/// for theirs. See `qwen_settings_merged` for what goes in.
fn write_qwen_settings(model: &str, vision: bool) -> anyhow::Result<()> {
    write_qwen_settings_at(
        &qwen_home()?,
        model,
        &format!("{}/v1", daemon::server()),
        vision,
    )
}

/// Read as Qwen Code reads it, comments stripped and an empty file as
/// `{}`. A file that is not a JSON object is left alone with a line
/// printed, since Qwen Code resets such a file to `{}` itself; one that
/// parses but cannot be written is an error, since an entry in it may be
/// the one this write was to outrank. The user's own file, and any later
/// one carrying comments, is kept as `settings.json.bak`.
fn write_qwen_settings_at(
    dir: &Path,
    model: &str,
    base_url: &str,
    vision: bool,
) -> anyhow::Result<()> {
    write_json_merged(&dir.join("settings.json"), "qwen", |existing| {
        qwen_settings_merged(existing, model, base_url, vision)
    })
}

/// Rewrites the JSON object at `path` through `merge`, leaving a file it
/// does not understand alone and saying so under `label`.
///
/// The settings files these integrations read are the user's as much as
/// llmman's, so: comments survive a parse (`strip_json_comments`) and the
/// text carrying them is kept as `.bak` before the first rewrite that
/// would drop them; a rendering that already matches is not written at
/// all; and the write is atomic, since a half-written settings file is
/// worse than a stale one. A leading BOM is stripped because a
/// node-based reader tolerates one and `serde_json` does not.
fn write_json_merged(
    path: &Path,
    label: &str,
    merge: impl FnOnce(&serde_json::Value) -> serde_json::Value,
) -> anyhow::Result<()> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => Some(raw),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let existing = match raw
        .as_deref()
        .map(|r| r.trim_start_matches('\u{feff}').trim())
    {
        None | Some("") => serde_json::json!({}),
        Some(text) => match serde_json::from_str::<serde_json::Value>(&strip_json_comments(text)) {
            Ok(value) if value.is_object() => value,
            _ => {
                eprintln!(
                    "[llmman] {label}: {} is not a JSON object; leaving it alone",
                    path.display()
                );
                return Ok(());
            }
        },
    };
    let merged = merge(&existing);
    if merged == existing {
        return Ok(());
    }
    let dir = path.parent().context("settings path has no directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    if let Some(raw) = &raw {
        let bak = path.with_extension("json.bak");
        if !bak.exists() || strip_json_comments(raw) != *raw {
            std::fs::copy(path, &bak)
                .with_context(|| format!("back up {} to {}", path.display(), bak.display()))?;
        }
    }
    let mut out = serde_json::to_string_pretty(&merged)?;
    out.push('\n');
    crate::fsutil::write_atomic(path, out.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

/// `//` and `/* */` comments outside strings replaced by spaces, so a
/// parse error still points at the right place; what `strip-json-comments`
/// does for Qwen Code before `JSON.parse`.
fn strip_json_comments(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '/' if chars.peek() == Some(&'/') => {
                out.push(' ');
                while chars.peek().is_some_and(|&n| n != '\n') {
                    chars.next();
                    out.push(' ');
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                out.push_str("  ");
                let mut prev = ' ';
                for n in chars.by_ref() {
                    out.push(if n == '\n' { '\n' } else { ' ' });
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// The variable llmman's entry names as its key, and what marks the entry
/// as llmman's: a user can rename it in `/model`, but not re-key it.
const QWEN_ENV_KEY: &str = "LLMMAN_API_KEY";

/// `existing` with llmman's entry merged in, pure so a test can hand it
/// a literal. The keys follow Qwen Code's own `/auth` and `/model` and
/// ollama's `applyQwenOllamaConfig` in `cmd/launch/qwen.go`: the entry
/// first in `modelProviders.openai`, an earlier one of llmman's replaced,
/// the rest kept and a `{ protocol, models }` wrapper unwrapped with
/// `$version` set to 4; `security.auth`; `model.name` and `model.baseUrl`.
/// A vision model's entry declares image input, which Qwen Code reads
/// only off the provider entry, not the top-level `model.generationConfig`.
fn qwen_settings_merged(
    existing: &serde_json::Value,
    model: &str,
    base_url: &str,
    vision: bool,
) -> serde_json::Value {
    let mut doc = existing.as_object().cloned().unwrap_or_default();
    let mut ours = serde_json::json!({
        "id": model,
        "name": format!("{model} (llmman)"),
        "baseUrl": base_url,
        "envKey": QWEN_ENV_KEY,
    });
    if vision {
        ours["generationConfig"] = serde_json::json!({ "modalities": { "image": true } });
    }
    let openai = object_under(&mut doc, "modelProviders")
        .entry("openai")
        .or_insert_with(|| serde_json::json!([]));
    let unwrapped = openai.get("models").is_some_and(|m| m.is_array());
    let entries = openai
        .as_array()
        .or_else(|| openai.get("models").and_then(serde_json::Value::as_array));
    let kept = entries.map_or_else(Vec::new, |entries| {
        entries
            .iter()
            .filter(|e| !qwen_entry_is_ours(e, base_url))
            .cloned()
            .collect()
    });
    *openai = serde_json::Value::Array(std::iter::once(ours).chain(kept).collect());
    if unwrapped {
        doc.insert("$version".into(), 4.into());
    }
    let auth = object_under(object_under(&mut doc, "security"), "auth");
    auth.insert("selectedType".into(), "openai".into());
    auth.insert("baseUrl".into(), base_url.into());
    let model_cfg = object_under(&mut doc, "model");
    model_cfg.insert("name".into(), model.into());
    model_cfg.insert("baseUrl".into(), base_url.into());
    serde_json::Value::Object(doc)
}

/// The object at `key` in `parent`, put there if absent or not an object.
fn object_under<'a>(
    parent: &'a mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, serde_json::Value> {
    let slot = parent.entry(key).or_insert_with(|| serde_json::json!({}));
    if !slot.is_object() {
        *slot = serde_json::json!({});
    }
    slot.as_object_mut().expect("set to an object just above")
}

/// An entry llmman wrote: `QWEN_ENV_KEY` as its key, at this daemon's
/// address, the test `qwenIsOllamaProvider` makes in ollama's
/// `cmd/launch/qwen.go`. The id is the model name and the display name
/// is the user's to change, so neither marks an owner.
fn qwen_entry_is_ours(entry: &serde_json::Value, base_url: &str) -> bool {
    let field = |k: &str| entry.get(k).and_then(serde_json::Value::as_str);
    field("envKey") == Some(QWEN_ENV_KEY)
        && field("baseUrl")
            .is_some_and(|u| u.trim_end_matches('/') == base_url.trim_end_matches('/'))
}

/// goose: configured entirely through the environment, which goose reads
/// in preference to its own `config.yaml` — so unlike hermes and qwen
/// nothing is written to disk and no key is persisted. `OPENAI_HOST` is
/// the bare origin, not a `/v1` base URL: goose joins it with
/// `OPENAI_BASE_PATH` itself. Verified against goose 1.50.0 with no
/// config file and no `goose configure`.
fn launch_goose(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_goose().ok_or_else(|| anyhow::anyhow!("goose is not installed"))?;
    let host = daemon::server();
    exec_with_env(&bin, extra_args, &goose_env(model, api_key, &host))
}

/// Split out so a test can assert what goose is handed: [`exec_with_env`]
/// never returns, so calling [`launch_goose`] would take the test runner
/// with it.
fn goose_env<'a>(model: &'a str, api_key: &'a str, host: &'a str) -> Vec<(&'a str, &'a str)> {
    let mut env = vec![
        ("GOOSE_PROVIDER", "openai"),
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_HOST", host),
        ("OPENAI_BASE_PATH", "v1/chat/completions"),
    ];
    // Absent, not empty: goose reads "" as a model actually named "".
    if !model.is_empty() {
        env.push(("GOOSE_MODEL", model));
    }
    env
}

fn find_goose() -> Option<PathBuf> {
    find_on_path("goose").or_else(|| goose_fallback(&dirs::home_dir()?))
}

/// goose's own installer target: `download_cli.sh` writes to
/// `$GOOSE_BIN_DIR` without putting it on `PATH`. Its default is
/// `$USERPROFILE/goose` on Windows (what `dirs::home_dir` returns there)
/// and `~/.local/bin` elsewhere; Windows probes both, since goose's
/// install instructions and this repo's CI pass the latter (v1.50.0).
fn goose_fallback(home: &Path) -> Option<PathBuf> {
    let bin = if cfg!(windows) { "goose.exe" } else { "goose" };
    let mut candidates = Vec::new();
    if cfg!(windows) {
        candidates.push(home.join("goose").join(bin));
    }
    candidates.push(home.join(".local").join("bin").join(bin));
    // is_file, not exists: a directory of that name would be reported as
    // installed and then fail to spawn.
    candidates.into_iter().find(|p| p.is_file())
}

// ---------------------------------------------------------------------------
// cline
// ---------------------------------------------------------------------------

/// cline: merge llmman's Ollama route into Cline's own settings, then pass
/// through exactly the arguments supplied after `--`. Cline reads both the
/// current provider store and legacy global-state fields, so keep them in
/// sync while preserving unrelated user settings in each file.
fn launch_cline(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("cline").ok_or_else(|| anyhow::anyhow!("cline is not installed"))?;
    write_cline_settings(model)?;
    exec_with_env(&bin, extra_args, &[])
}

/// Cline's own resolution: `CLINE_DIR || path.join(os.homedir(), ".cline")`.
/// Node's `os.homedir()` reads `USERPROFILE` on Windows, which
/// `dirs::home_dir` ignores; disagreeing with Cline there means it
/// never sees the settings and exits "Not authenticated".
fn cline_dir() -> anyhow::Result<PathBuf> {
    resolve_cline_dir(env_dir("CLINE_DIR"), node_user_profile(), || {
        dirs::home_dir().context("no home directory")
    })
}

/// `%USERPROFILE%` on Windows, where node's `os.homedir()` reads it and
/// `dirs::home_dir` does not. `None` elsewhere, where `os.homedir()`
/// reads `$HOME` and `dirs` agrees.
fn node_user_profile() -> Option<PathBuf> {
    cfg!(windows).then(|| env_dir("USERPROFILE")).flatten()
}

/// What node's `os.homedir()` answers, for the integrations that are
/// node programs — see [`node_user_profile`].
fn node_home_dir() -> Option<PathBuf> {
    node_user_profile().or_else(dirs::home_dir)
}

fn resolve_cline_dir(
    cline_dir: Option<PathBuf>,
    user_profile: Option<PathBuf>,
    home: impl FnOnce() -> anyhow::Result<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if let Some(dir) = cline_dir {
        return Ok(dir);
    }
    Ok(match user_profile {
        Some(profile) => profile,
        None => home()?,
    }
    .join(".cline"))
}

fn cline_data_dir() -> anyhow::Result<PathBuf> {
    Ok(cline_dir()?.join("data"))
}

fn write_cline_settings(model: &str) -> anyhow::Result<()> {
    let server = daemon::server();
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    write_cline_settings_at(&cline_data_dir()?, model, &server, &now)
}

/// Cline's `providers.<id>.settings.timeout` (ms): how long its Ollama
/// vendor waits for a response to *start*; its default is 5 minutes,
/// sized for Ollama's model load. Through llmman it also has to cover
/// llama-server prefilling Cline's ~12k-token system prompt, since no
/// bytes are sent before the first token: on CPU that takes minutes (a
/// 4-vCPU aarch64 runner manages ~44 tok/s even on a 0.8B model), and
/// llmman's own load deadline is already 10 minutes. In CI run
/// 35602511987 Cline dropped its first request 5:00 into the prefill and
/// re-sent it. 30 minutes covers load plus a long prefill; connection
/// failures still fail fast, and a user-set value is kept.
const CLINE_RESPONSE_START_TIMEOUT_MS: u64 = 30 * 60 * 1000;

fn read_cline_json(path: &Path) -> anyhow::Result<(Option<Vec<u8>>, serde_json::Value)> {
    let raw = match std::fs::read(path) {
        Ok(raw) => Some(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let document = match raw.as_deref().map(|bytes| String::from_utf8_lossy(bytes)) {
        None => serde_json::json!({}),
        Some(text) if text.trim().is_empty() => serde_json::json!({}),
        Some(text) => serde_json::from_str(&text)
            .with_context(|| format!("parse {} as JSON", path.display()))?,
    };
    anyhow::ensure!(
        document.is_object(),
        "{} is not a JSON object",
        path.display()
    );
    Ok((raw, document))
}

/// Writes a changed Cline JSON document atomically, copying the exact prior
/// bytes to `<name>.json.bak` before replacing it.
fn write_cline_json(
    path: &Path,
    raw: Option<&[u8]>,
    before: &serde_json::Value,
    after: &serde_json::Value,
) -> anyhow::Result<()> {
    if before == after {
        return Ok(());
    }
    let parent = path
        .parent()
        .context("Cline settings path has no directory")?;
    std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    if let Some(raw) = raw {
        let backup = path.with_extension("json.bak");
        std::fs::write(&backup, raw)
            .with_context(|| format!("back up {} to {}", path.display(), backup.display()))?;
    }
    let mut contents = serde_json::to_vec_pretty(after).context("serialize Cline settings")?;
    contents.push(b'\n');
    crate::fsutil::write_atomic(path, &contents)
        .with_context(|| format!("write {}", path.display()))
}

fn write_cline_settings_at(
    data_dir: &Path,
    model: &str,
    server: &str,
    now: &str,
) -> anyhow::Result<()> {
    let providers_path = data_dir.join("settings/providers.json");
    let (providers_raw, mut providers_document) = read_cline_json(&providers_path)?;
    let providers_before = providers_document.clone();
    let base_url = format!("{server}/v1");
    let route_changed = providers_document
        .pointer("/providers/ollama/settings/model")
        .and_then(serde_json::Value::as_str)
        != Some(model)
        || providers_document
            .pointer("/providers/ollama/settings/baseUrl")
            .and_then(serde_json::Value::as_str)
            != Some(base_url.as_str())
        || providers_document
            .pointer("/providers/ollama/settings/timeout")
            .is_none();

    let root = providers_document
        .as_object_mut()
        .expect("read_cline_json returns an object");
    root.insert("version".to_string(), serde_json::json!(1));
    root.insert("lastUsedProvider".to_string(), serde_json::json!("ollama"));
    let providers = root
        .entry("providers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers is not a JSON object")?;
    let ollama = providers
        .entry("ollama")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers.ollama is not a JSON object")?;
    let settings = ollama
        .entry("settings")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("Cline providers.ollama.settings is not a JSON object")?;
    settings.insert("provider".to_string(), serde_json::json!("ollama"));
    settings.insert("model".to_string(), serde_json::json!(model));
    settings.insert("baseUrl".to_string(), serde_json::json!(base_url));
    settings
        .entry("timeout")
        .or_insert_with(|| serde_json::json!(CLINE_RESPONSE_START_TIMEOUT_MS));
    settings.remove("apiKey");
    ollama.insert("tokenSource".to_string(), serde_json::json!("manual"));
    if route_changed {
        ollama.insert("updatedAt".to_string(), serde_json::json!(now));
    }
    write_cline_json(
        &providers_path,
        providers_raw.as_deref(),
        &providers_before,
        &providers_document,
    )?;

    let global_state_path = data_dir.join("globalState.json");
    let (global_raw, mut global_document) = read_cline_json(&global_state_path)?;
    let global_before = global_document.clone();
    let global = global_document
        .as_object_mut()
        .expect("read_cline_json returns an object");
    for key in [
        "ollamaBaseUrl",
        "actModeOllamaBaseUrl",
        "planModeOllamaBaseUrl",
    ] {
        global.insert(key.to_string(), serde_json::json!(server));
    }
    for key in ["actModeApiProvider", "planModeApiProvider"] {
        global.insert(key.to_string(), serde_json::json!("ollama"));
    }
    for key in ["actModeOllamaModelId", "planModeOllamaModelId"] {
        global.insert(key.to_string(), serde_json::json!(model));
    }
    global.insert("welcomeViewCompleted".to_string(), serde_json::json!(true));
    write_cline_json(
        &global_state_path,
        global_raw.as_deref(),
        &global_before,
        &global_document,
    )
}

// ---------------------------------------------------------------------------
// grok (Grok Build)
// ---------------------------------------------------------------------------

/// The per-model `env_key` in llmman's Grok config reads this variable.
/// A model credential outranks both Grok's signed-in session and its global
/// `XAI_API_KEY`, without putting the actual key on disk.
const GROK_API_KEY_ENV: &str = "LLMMAN_GROK_API_KEY";

/// grok: point its custom-model catalog and inference client at llmman's
/// OpenAI-compatible surface. Every auxiliary model is pinned too: without
/// this, Grok Build keeps built-in hosted ids for title/summary, image
/// description, web search, and prompt suggestions, then asks the local
/// daemon to load one after the main model already answered successfully.
///
/// The model flag is injected only when the caller did not provide one
/// after `--`. This matches Qwen's behavior above and lets an explicit
/// integration argument win without passing a duplicate flag.
fn launch_grok(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_grok().ok_or_else(|| anyhow::anyhow!("grok is not installed"))?;
    let effective_model = forwarded_model(extra_args).unwrap_or(model);
    let base_url = format!("{}/v1", daemon::server());
    let models_url = format!("{base_url}/models");
    // Never edit the user's config.toml. This child is wholly llmman-owned,
    // and setting GROK_HOME below scopes it to this launched process.
    let home = grok_home()?.join("llmman");
    write_grok_config(&home, effective_model, &base_url)?;
    let home = home.to_string_lossy().into_owned();
    let args = grok_args(model, extra_args);
    exec_with_env(
        &bin,
        &args,
        &grok_env(effective_model, api_key, &base_url, &models_url, &home),
    )
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
fn grok_home() -> anyhow::Result<PathBuf> {
    if let Some(path) = env_dir("GROK_HOME") {
        return Ok(path);
    }
    Ok(dirs::home_dir().context("no home directory")?.join(".grok"))
}

fn write_grok_config(home: &Path, model: &str, base_url: &str) -> anyhow::Result<()> {
    let path = home.join("config.toml");
    std::fs::create_dir_all(home).with_context(|| format!("create {}", home.display()))?;
    let contents = grok_config_document(model, base_url);
    crate::fsutil::write_atomic(&path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

fn grok_config_document(model: &str, base_url: &str) -> String {
    let mut entry = toml_edit::Table::new();
    entry["model"] = toml_edit::value(model);
    entry["base_url"] = toml_edit::value(base_url);
    entry["env_key"] = toml_edit::value(GROK_API_KEY_ENV);
    entry["api_backend"] = toml_edit::value("chat_completions");

    let mut models = toml_edit::Table::new();
    models.insert(model, toml_edit::Item::Table(entry));
    let mut document = toml_edit::DocumentMut::new();
    document.insert("model", toml_edit::Item::Table(models));
    document.to_string()
}

fn grok_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 2);
    if !has_flag(extra_args, "--model", Some("-m")) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// `PATH`, then the official installer's target, `~/.grok/bin`.
fn find_grok() -> Option<PathBuf> {
    find_on_path("grok").or_else(|| grok_fallback(&dirs::home_dir()?))
}

fn grok_fallback(home: &Path) -> Option<PathBuf> {
    let binary = if cfg!(windows) { "grok.exe" } else { "grok" };
    let candidate = home.join(".grok").join("bin").join(binary);
    candidate.is_file().then_some(candidate)
}

// ---------------------------------------------------------------------------
// dsh (DeepSeek Harness)
// ---------------------------------------------------------------------------

/// The env var dsh's generated provider entry reads its key from, so no
/// key value is ever written to disk (same role as `QWEN_ENV_KEY`).
const DSH_API_KEY_ENV: &str = "LLMMAN_API_KEY";

/// dsh: unlike qwen/hermes/codex above, nothing here merges into a file
/// dsh reads by default. dsh's own `--patch` overlay mechanism lets both
/// files live under llmman's own config dir and be rewritten in full on
/// every launch, without ever touching the user's real `$DSH_HOME`.
///
/// Defaults to the `web` profile, but a caller-supplied `--profile`
/// after `--` wins instead — e.g. `--profile headless "<task>"` for a
/// one-shot, scriptable run, the same way every other flag here already
/// yields to what the caller explicitly asked for.
///
/// Both files sit at one fixed path, rewritten in place per launch:
/// dsh hot-reloads the settings document, so two *concurrent* launches
/// naming different models would retarget each other — accepted
/// deliberately, since a per-launch directory costs a cleanup hook on
/// every exit path (signals included) for a case that needs two
/// simultaneous sessions on different models to bite at all.
fn launch_dsh(
    model: &str,
    api_key: &str,
    vision: bool,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let launcher = dsh_launcher(extra_args);
    if has_flag(launcher.args, "--patch", None) {
        anyhow::bail!("llmman launch dsh manages --patch itself; pass other dsh flags after --");
    }
    if let Some(command) = launcher.command {
        anyhow::bail!(
            "dsh's `{command}` command cannot be combined with the --patch llmman passes it.\n\
             Select a profile with `--profile {}` instead, or omit it for the default.",
            if command == "web" { "web" } else { "<name>" }
        );
    }
    let (bin, prefix) = find_dsh().ok_or_else(|| {
        anyhow::anyhow!("dsh is not installed, and there is no npx on PATH to run it with")
    })?;

    let dir = dsh_config_dir()?;
    let settings_path = dir.join("settings.yaml");
    write_dsh_settings(&settings_path, model, vision)?;
    let patch_path = dir.join("llmman.cordis.yml");
    write_dsh_patch(&patch_path, &settings_path)?;

    let mut args = prefix;
    if !args.is_empty() {
        // Said before it happens: this launch downloads a package.
        eprintln!("[llmman] dsh is not installed; running {DSH_NPM_PACKAGE} with npx");
    }
    args.extend(dsh_args(&patch_path, extra_args));
    exec_with_env(&bin, &args, &[(DSH_API_KEY_ENV, api_key)])
}

/// The npm package `npx` fetches when dsh isn't installed. Unpinned, so
/// a one-off run gets what a global install would have.
const DSH_NPM_PACKAGE: &str = "@deepseek-ai/dsh@latest";

/// dsh, and the arguments that must lead whatever it is handed: none for
/// an installed `dsh`, `--yes <package>` for the `npx` that stands in when
/// there is none. `find_integration_binary` resolves it the same way, so
/// the listing agrees with what a launch would run.
fn find_dsh() -> Option<(PathBuf, Vec<String>)> {
    dsh_command(find_on_path("dsh"), || find_on_path("npx"))
}

/// Split from [`find_dsh`] so which binary wins can be asserted without
/// depending on what the test machine has installed.
fn dsh_command(
    dsh: Option<PathBuf>,
    npx: impl FnOnce() -> Option<PathBuf>,
) -> Option<(PathBuf, Vec<String>)> {
    match dsh {
        Some(bin) => Some((bin, Vec::new())),
        None => Some((npx()?, vec!["--yes".into(), DSH_NPM_PACKAGE.into()])),
    }
}

/// The tokens dsh reads as its own launcher flags, rather than forwards
/// to the selected profile's app. Verified against dsh 0.1.2-rc.1: it
/// stops at `--`, and also at the first token that isn't one of its own
/// options — `--dump-config sometask --patch <file>` reports `--patch`
/// and the file as app arguments and never reads it.
///
/// Scanning past either boundary reads app arguments as launcher ones:
/// an app-level `--profile` would count as a profile selection and drop
/// the default `web`, and an app-level `--patch` would be refused here
/// as though it were ours to manage. (The app may well reject that
/// token itself — headless answers `unknown option '--patch'` — but
/// that is dsh's own argument to make, in its own words.)
///
/// `--profile`/`--patch` are the two that take a value, which has to be
/// stepped over so it isn't mistaken for the first app argument; every
/// other dsh option (`--dump-config`, `--version`, ...) is a bare flag.
fn dsh_launcher(extra_args: &[String]) -> DshLauncher<'_> {
    let mut end = 0;
    let mut command = None;
    while let Some(arg) = extra_args.get(end) {
        if arg == "--" {
            break;
        }
        end += match arg.as_str() {
            // The two that take a value: step over it as well, so a
            // profile or path isn't read as a command or as the first
            // app argument (`--profile web`'s value is not the `web`
            // command).
            "--profile" | "--patch" => 2,
            // dsh's command spellings, which it refuses to combine with
            // any parent option — "web takes none of parent --profile,
            // --patch, ..." — so no argument order pairs one with the
            // `--patch` this injects.
            found @ ("web" | "plugin") => {
                command = command.or(Some(found));
                1
            }
            _ if arg.starts_with('-') => 1,
            // Anything else is dsh's first app argument.
            _ => break,
        };
    }
    DshLauncher {
        args: &extra_args[..end.min(extra_args.len())],
        command,
    }
}

/// dsh's own launcher section: the tokens it reads rather than forwards,
/// and the command spelling inside them, if any.
struct DshLauncher<'a> {
    args: &'a [String],
    command: Option<&'a str>,
}

/// The argv dsh is invoked with. `--patch` is always injected; the
/// default `web` profile is omitted when the caller already named one
/// (however spelled) after `--`, so `--profile headless "task"` selects
/// dsh's real one-shot mode instead of being appended onto `web`, which
/// does not accept it.
fn dsh_args(patch_path: &Path, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    if !has_flag(dsh_launcher(extra_args).args, "--profile", None) {
        args.push("web".to_string());
    }
    args.push("--patch".to_string());
    args.push(patch_path.to_string_lossy().into_owned());
    args.extend_from_slice(extra_args);
    args
}

/// `~/.config/llmman/launch/dsh`. Derived from `llmman.conf`'s own
/// directory rather than rebuilt by hand, so the two cannot drift.
/// dsh never looks here on its own; only `--patch` points it there.
fn dsh_config_dir() -> anyhow::Result<PathBuf> {
    let conf = crate::config::user_path().context("no home directory")?;
    let dir = conf.parent().context("llmman.conf has no directory")?;
    Ok(dir.join("launch").join("dsh"))
}

/// The settings document `llmman.cordis.yml` points dsh at: registers
/// `llmman` as an `llm-pi-ai` provider route at this daemon's `/v1`, and
/// selects it as the `agent-default-model`.
fn write_dsh_settings(path: &Path, model: &str, vision: bool) -> anyhow::Result<()> {
    let quoted_model = yaml_quote(model);
    let base_url = yaml_quote(&format!("{}/v1", daemon::server()));
    // Claiming image input a text-only model can't serve would have dsh
    // attach what the daemon then rejects.
    let input = if vision { "[text, image]" } else { "[text]" };
    let contents = format!(
        "# Written by `llmman launch dsh`; edits are overwritten.\n\
         agent-default-model:\n  provider: llmman\n  model: {quoted_model}\n\
         llm-pi-ai:\n  providers:\n    llmman:\n      displayName: llmman\n      \
         apiKeyEnv: {DSH_API_KEY_ENV}\n      api: openai-completions\n      baseURL: {base_url}\n      \
         models:\n        - id: {quoted_model}\n          name: {quoted_model}\n          input: {input}\n"
    );
    write_dsh_file(path, &contents)
}

/// dsh's patch shape: points its `settings` provider at the document above.
fn write_dsh_patch(path: &Path, settings_path: &Path) -> anyhow::Result<()> {
    write_dsh_file(path, &dsh_patch_document(settings_path))
}

/// Split from `write_dsh_patch` so a test can render a Windows-shaped
/// path on any platform: `yaml_quote` escapes the `\` separators, and
/// forgetting that is what once turned the Windows leg red.
fn dsh_patch_document(settings_path: &Path) -> String {
    let quoted_settings_path = yaml_quote(&settings_path.to_string_lossy());
    format!(
        "# Written by `llmman launch dsh`; edits are overwritten.\n\
         - id: settings\n  config:\n    path: {quoted_settings_path}\n"
    )
}

fn write_dsh_file(path: &Path, contents: &str) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    crate::fsutil::write_atomic(path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

// ---------------------------------------------------------------------------
// Process execution helper
// ---------------------------------------------------------------------------

fn exec_with_env(bin: &PathBuf, args: &[String], extra_env: &[(&str, &str)]) -> anyhow::Result<()> {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::inherit());
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());

    // Inherit the current environment and overlay OLLAMA_HOST + integration vars.
    let mut env: std::collections::HashMap<String, String> = std::env::vars().collect();
    env.insert("OLLAMA_HOST".to_string(), daemon::server());
    for (k, v) in extra_env {
        env.insert(k.to_string(), v.to_string());
    }
    cmd.envs(&env);

    let status = cmd
        .status()
        .with_context(|| format!("failed to run {}", bin.display()))?;
    std::process::exit(status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agy_settings_are_written_to_the_llmman_owned_directory() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-agy-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_agy_settings_at(&dir).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("antigravity-cli/settings.json")).unwrap(),
            "{\n  \"modelProvider\": \"gemini\"\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agy_is_listed_as_an_integration() {
        let agy = INTEGRATIONS.iter().find(|i| i.name == "agy").unwrap();
        assert_eq!(agy.binary, "agy");
    }

    /// Every integration `--provider` refuses must be one `launch`
    /// actually dispatches, or the refusal is for a name nobody can type
    /// and the real one is still silently broken.
    #[test]
    fn every_provider_unsupported_integration_is_a_real_one() {
        for (id, why) in PROVIDER_UNSUPPORTED {
            assert!(
                INTEGRATIONS.iter().any(|i| i.name == *id) || *id == "copilot-cli",
                "{id} is not an integration"
            );
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

    /// Regression test for a real CodeRabbit finding: an unquoted model
    /// value in generated YAML could be misparsed as a non-string
    /// (`null`, `true`, ...) or broken outright by metacharacters.
    #[test]
    fn yaml_quote_escapes_keywords_and_metacharacters() {
        assert_eq!(yaml_quote("qwen3.5:0.8b"), "\"qwen3.5:0.8b\"");
        assert_eq!(yaml_quote("null"), "\"null\"");
        assert_eq!(yaml_quote("true"), "\"true\"");
        assert_eq!(
            yaml_quote(r#"a "quoted" \ value"#),
            r#""a \"quoted\" \\ value""#
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

    /// Regression test for the real openclaw onboarding failure
    /// described on `openclaw_model_id`'s own doc comment.
    #[test]
    fn openclaw_model_id_strips_the_docker_ai_prefix() {
        assert_eq!(
            openclaw_model_id("docker.io/ai/qwen3.5:0.8b"),
            "qwen3.5:0.8b"
        );
        assert_eq!(openclaw_model_id("qwen3.5:0.8b"), "qwen3.5:0.8b");
        assert_eq!(
            openclaw_model_id("hf.co/unsloth/Qwen3.5-0.8B-GGUF"),
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF"
        );
        assert_eq!(openclaw_model_id(""), "default");
        assert_eq!(openclaw_model_id("docker.io/ai/"), "default");
    }

    /// Every integration `check_model_flag` holds to a model must be one
    /// `launch` dispatches; it is refused without one, under `--provider`
    /// too, and a `--model` after `--` is let through.
    #[test]
    fn model_required_integrations_are_refused_without_a_model() {
        let none: Vec<String> = vec![];
        for id in MODEL_REQUIRED {
            assert!(
                INTEGRATIONS.iter().any(|i| i.name == *id),
                "{id} is not an integration"
            );
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
            assert!(MODEL_REQUIRED.contains(id), "{id} is not model-required");
        }
        assert!(!MODEL_FLAG_FORWARDED.contains(&"dsh"));
    }

    /// The only configuration goose gets: a wrong or missing one sends
    /// the session to api.openai.com instead of the daemon. `OPENAI_HOST`
    /// is the bare origin — goose appends `OPENAI_BASE_PATH` itself, so a
    /// `/v1` here would request `/v1/v1/chat/completions`.
    #[test]
    fn goose_env_points_at_the_daemon_and_carries_the_key() {
        let env = goose_env("m", "k", "http://127.0.0.1:17434");
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
        assert_eq!(get("GOOSE_PROVIDER"), Some("openai"));
        assert_eq!(get("GOOSE_MODEL"), Some("m"));
        assert_eq!(get("OPENAI_API_KEY"), Some("k"));
        assert_eq!(get("OPENAI_HOST"), Some("http://127.0.0.1:17434"));
        assert_eq!(get("OPENAI_BASE_PATH"), Some("v1/chat/completions"));

        let env = goose_env("", "k", "http://127.0.0.1:17434");
        assert!(!env.iter().any(|(n, _)| *n == "GOOSE_MODEL"));
    }

    /// goose carries the key in its own environment, so `--provider`
    /// needs neither a refusal nor the daemon holding the key.
    #[test]
    fn goose_carries_its_own_key_so_provider_works() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "goose"));
        assert!(check_provider_supported("goose").is_ok());
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"goose"));
    }

    /// `download_cli.sh`'s target is off `PATH` on a fresh shell, so this
    /// fallback is the one that fires for most installs — at every
    /// directory that installer writes to, Windows included.
    #[test]
    fn goose_fallback_finds_the_installers_target() {
        let name = if cfg!(windows) { "goose.exe" } else { "goose" };
        let dirs: &[&[&str]] = if cfg!(windows) {
            &[&["goose"], &[".local", "bin"]]
        } else {
            &[&[".local", "bin"]]
        };
        for (i, parts) in dirs.iter().enumerate() {
            let home = std::env::temp_dir().join(format!(
                "llmman-goose-{}-{}-{i}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let bin = parts.iter().fold(home.clone(), |p, part| p.join(part));
            std::fs::create_dir_all(&bin).unwrap();
            assert_eq!(goose_fallback(&home), None);
            let goose = bin.join(name);
            // A directory of that name is not the binary: returning it
            // would report goose as installed and then fail to spawn.
            std::fs::create_dir(&goose).unwrap();
            assert_eq!(goose_fallback(&home), None);
            std::fs::remove_dir(&goose).unwrap();
            std::fs::write(&goose, "").unwrap();
            assert_eq!(goose_fallback(&home), Some(goose));
            assert_eq!(goose_fallback(&home.join("nowhere")), None);
            let _ = std::fs::remove_dir_all(&home);
        }
    }

    #[test]
    fn cline_settings_merge_ollama_and_preserve_user_state() {
        let root = std::env::temp_dir().join(format!(
            "llmman-cline-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let providers_path = root.join("settings/providers.json");
        let global_path = root.join("globalState.json");
        let model = r#"org/model.\"quoted\""#;
        let providers_original = r#"{
  "version": 7,
  "lastUsedProvider": "anthropic",
  "mine": true,
  "providers": {
    "anthropic": { "settings": { "model": "mine" } },
    "ollama": {
      "settings": { "provider": "ollama", "model": "old", "baseUrl": "http://old", "apiKey": "delete-me", "keep": 1 },
      "updatedAt": "2000-01-01T00:00:00Z",
      "other": true
    }
  }
}"#;
        let global_original = r#"{"theme":"mine","welcomeViewCompleted":false}"#;
        std::fs::create_dir_all(providers_path.parent().unwrap()).unwrap();
        std::fs::write(&providers_path, providers_original).unwrap();
        std::fs::write(&global_path, global_original).unwrap();

        write_cline_settings_at(
            &root,
            model,
            "http://127.0.0.1:17434",
            "2026-09-20T06:00:00Z",
        )
        .unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        let entry = &parsed["providers"]["ollama"];
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["lastUsedProvider"], "ollama");
        assert_eq!(parsed["mine"], true);
        assert_eq!(
            parsed["providers"]["anthropic"]["settings"]["model"],
            "mine"
        );
        assert_eq!(entry["settings"]["provider"], "ollama");
        assert_eq!(entry["settings"]["model"], model);
        assert_eq!(entry["settings"]["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(entry["settings"]["keep"], 1);
        assert_eq!(
            entry["settings"]["timeout"], CLINE_RESPONSE_START_TIMEOUT_MS,
            "a missing response-start timeout gets llmman's slow-prefill default"
        );
        assert!(entry["settings"].get("apiKey").is_none());
        assert_eq!(entry["tokenSource"], "manual");
        assert_eq!(entry["updatedAt"], "2026-09-20T06:00:00Z");
        assert_eq!(entry["other"], true);
        assert_eq!(
            std::fs::read_to_string(root.join("settings/providers.json.bak")).unwrap(),
            providers_original
        );

        let global: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&global_path).unwrap()).unwrap();
        assert_eq!(global["theme"], "mine");
        assert_eq!(global["ollamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["actModeApiProvider"], "ollama");
        assert_eq!(global["planModeApiProvider"], "ollama");
        assert_eq!(global["actModeOllamaModelId"], model);
        assert_eq!(global["planModeOllamaModelId"], model);
        assert_eq!(global["actModeOllamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["planModeOllamaBaseUrl"], "http://127.0.0.1:17434");
        assert_eq!(global["welcomeViewCompleted"], true);
        assert_eq!(
            std::fs::read_to_string(root.join("globalState.json.bak")).unwrap(),
            global_original
        );

        let providers_written = std::fs::read(&providers_path).unwrap();
        let global_written = std::fs::read(&global_path).unwrap();
        write_cline_settings_at(
            &root,
            model,
            "http://127.0.0.1:17434",
            "2026-09-20T07:00:00Z",
        )
        .unwrap();
        assert_eq!(std::fs::read(&providers_path).unwrap(), providers_written);
        assert_eq!(std::fs::read(&global_path).unwrap(), global_written);

        write_cline_settings_at(
            &root,
            "new-model",
            "http://127.0.0.1:17434",
            "2026-09-20T08:00:00Z",
        )
        .unwrap();
        let changed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        assert_eq!(
            changed["providers"]["ollama"]["updatedAt"],
            "2026-09-20T08:00:00Z"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cline_settings_keep_a_user_set_response_start_timeout() {
        let root = std::env::temp_dir().join(format!(
            "llmman-cline-timeout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let providers_path = root.join("settings/providers.json");
        std::fs::create_dir_all(providers_path.parent().unwrap()).unwrap();
        std::fs::write(
            &providers_path,
            r#"{"providers":{"ollama":{"settings":{"timeout":45000}}}}"#,
        )
        .unwrap();

        write_cline_settings_at(
            &root,
            "some-model",
            "http://127.0.0.1:17434",
            "2026-09-20T06:00:00Z",
        )
        .unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&providers_path).unwrap()).unwrap();
        assert_eq!(parsed["providers"]["ollama"]["settings"]["timeout"], 45000);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cline_install_contract_uses_latest_and_only_yes_accepts() {
        assert_eq!(CLINE_NPM_INSTALL_ARGS, ["install", "-g", "cline@latest"]);
        assert_eq!(
            CLINE_INSTALL_PROMPT,
            "Cline is not installed. Install with npm? [y/N] "
        );
        assert_eq!(CLINE_INSTALL_CANCELLED, "cline installation cancelled");
        for answer in ["y", "Y", "yes", "YES", " yes\n"] {
            assert!(accepts_install(answer));
        }
        for answer in ["", "\n", "n", "no", "yep"] {
            assert!(!accepts_install(answer));
        }
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

    /// Same precedence as Cline: `CLINE_DIR`, then `USERPROFILE` (Windows'
    /// `os.homedir()`), then the process home.
    #[test]
    fn cline_dir_follows_cline_dir_then_userprofile_then_home() {
        let p = |s: &str| Some(PathBuf::from(s));
        let home = || Ok(PathBuf::from("/home"));
        let resolve = |dir, profile| resolve_cline_dir(dir, profile, home).unwrap();
        assert_eq!(
            resolve(p("/explicit"), p("/profile")),
            PathBuf::from("/explicit")
        );
        assert_eq!(
            resolve(None, p("/profile")),
            PathBuf::from("/profile/.cline")
        );
        assert_eq!(resolve(None, None), PathBuf::from("/home/.cline"));
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
        let text = grok_config_document(model, "http://127.0.0.1:17434/v1");

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
        write_grok_config(&isolated, "m", "http://127.0.0.1:17434/v1").unwrap();

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

    #[test]
    fn grok_is_model_required_and_refuses_unrepresentable_provider_routes() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "grok"));
        assert!(MODEL_REQUIRED.contains(&"grok"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"grok"));
        let error = check_provider_supported("grok").unwrap_err().to_string();
        assert!(error.contains("model catalog"), "{error}");
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"grok"));
    }

    /// The found directory goes in front of `PATH` only when it is not
    /// there, with no empty component either way.
    #[test]
    fn path_with_dir_prepended_only_when_it_is_missing() {
        let path_var =
            std::env::join_paths([PathBuf::from("/usr/bin"), PathBuf::from("/bin")]).unwrap();
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/usr/bin")), &path_var),
            None
        );
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), &path_var),
            Some(
                std::env::join_paths([
                    PathBuf::from("/opt/nvm/bin"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ])
                .unwrap()
            )
        );
        assert_eq!(path_with_dir_prepended(None, &path_var), None);
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), std::ffi::OsStr::new("")),
            Some(std::ffi::OsString::from("/opt/nvm/bin"))
        );
        let gappy = std::env::join_paths([
            PathBuf::from("/usr/bin"),
            PathBuf::from(""),
            PathBuf::from("/bin"),
        ])
        .unwrap();
        assert_eq!(
            path_with_dir_prepended(Some(Path::new("/opt/nvm/bin")), &gappy),
            Some(
                std::env::join_paths([
                    PathBuf::from("/opt/nvm/bin"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ])
                .unwrap()
            )
        );
    }

    /// The last forwarded model wins, in either spelling; none is none.
    #[test]
    fn forwarded_model_takes_the_last_spelling() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(forwarded_model(&args(&["--model", "b"])), Some("b"));
        assert_eq!(forwarded_model(&args(&["-m=b", "--model", "c"])), Some("c"));
        assert_eq!(forwarded_model(&args(&["--model=b", "-m", "c"])), Some("c"));
        assert_eq!(forwarded_model(&args(&["-p", "x"])), None);
        assert_eq!(forwarded_model(&args(&["--model"])), None);
        assert_eq!(forwarded_model(&args(&["--model="])), None);
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

    /// The two flags `launch_qwen` relies on to beat a persisted
    /// `~/.qwen/settings.json` (see its doc comment) go first, and each
    /// yields to the caller's own spelling of it — a repeated `--model`
    /// crashes Qwen Code.
    #[test]
    fn qwen_args_prefix_auth_type_and_model_unless_the_caller_passed_them() {
        let none: Vec<String> = vec![];
        assert_eq!(
            qwen_args("m:latest", &none),
            ["--auth-type", "openai", "--model", "m:latest"]
        );

        let user_model = vec![
            "--model".to_string(),
            "theirs".to_string(),
            "-p".to_string(),
        ];
        assert_eq!(
            qwen_args("m:latest", &user_model),
            ["--auth-type", "openai", "--model", "theirs", "-p"]
        );
        let user_short = vec!["-m=theirs".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_short),
            ["--auth-type", "openai", "-m=theirs"]
        );

        let user_auth = vec!["--auth-type=qwen-oauth".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_auth),
            ["--model", "m:latest", "--auth-type=qwen-oauth"]
        );
        let user_camel = vec!["--authType".to_string(), "openai".to_string()];
        assert_eq!(
            qwen_args("m:latest", &user_camel),
            ["--model", "m:latest", "--authType", "openai"]
        );
    }

    /// Any node version under `~/.nvm` that has qwen.
    #[test]
    fn nvm_qwen_finds_it_under_a_node_version() {
        let home = std::env::temp_dir().join(format!(
            "llmman-nvm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bin = home.join(".nvm/versions/node/v22.9.1/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(home.join(".nvm/versions/node/v20.19.0/bin")).unwrap();
        std::fs::write(bin.join("qwen"), "").unwrap();
        assert_eq!(nvm_qwen(&home), Some(bin.join("qwen")));
        assert_eq!(nvm_qwen(&home.join("nowhere")), None);
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The documented targets are on the list (see `find_qwen`).
    #[cfg(unix)]
    #[test]
    fn qwen_fallback_paths_name_the_documented_targets() {
        let home = dirs::home_dir().unwrap();
        let paths = qwen_fallback_paths();
        assert!(paths.contains(&home.join(".local/bin/qwen")));
        assert!(paths.contains(&home.join(".npm-global/bin/qwen")));
        assert!(paths.contains(&home.join(".cargo/bin/qwen")));
        assert!(paths.contains(&PathBuf::from("/usr/local/bin/qwen")));
    }

    /// A file a Qwen Code user already has: llmman's entry goes first, an
    /// older one of its own for this daemon goes, and everything else
    /// stays, a hand-written entry at this daemon's address included.
    #[test]
    fn qwen_settings_merge_keeps_what_is_not_llmmans() {
        let existing = serde_json::json!({
            "$version": 4,
            "ui": { "theme": "keep-me" },
            "modelProviders": {
                "gemini": [ { "id": "gemini-2.5-pro" } ],
                "openai": [
                    { "id": "docker.io/ai/m:latest", "name": "cloud copy",
                      "baseUrl": "https://cloud.example/v1",
                      "envKey": "QWEN_CUSTOM_API_KEY_X", "customField": 1 },
                    { "id": "old:latest", "name": "renamed by the user",
                      "baseUrl": "http://127.0.0.1:17434/v1/", "envKey": "LLMMAN_API_KEY" },
                    { "id": "other:latest", "name": "other:latest (llmman)",
                      "baseUrl": "http://10.0.0.2:17434/v1", "envKey": "LLMMAN_API_KEY" },
                    { "id": "local-alias", "name": "my alias for the daemon",
                      "baseUrl": "http://127.0.0.1:17434/v1", "envKey": "OPENAI_API_KEY",
                      "generationConfig": { "temperature": 0.2 } }
                ]
            },
            "security": { "auth": { "selectedType": "qwen-oauth", "apiKey": "keep-too" } },
            "model": { "name": "gemini-2.5-pro", "generationConfig": { "temperature": 0.1 } }
        });
        let url = "http://127.0.0.1:17434/v1";
        let merged = qwen_settings_merged(&existing, "docker.io/ai/m:latest", url, false);
        assert_eq!(merged["$version"], 4);
        assert_eq!(merged["ui"]["theme"], "keep-me");
        assert_eq!(
            merged["modelProviders"]["gemini"],
            existing["modelProviders"]["gemini"]
        );
        let before = existing["modelProviders"]["openai"].as_array().unwrap();
        let openai = merged["modelProviders"]["openai"].as_array().unwrap();
        assert_eq!(
            openai[0],
            serde_json::json!({ "id": "docker.io/ai/m:latest",
                "name": "docker.io/ai/m:latest (llmman)", "baseUrl": url,
                "envKey": "LLMMAN_API_KEY" })
        );
        assert_eq!(
            openai[1..],
            [before[0].clone(), before[2].clone(), before[3].clone()]
        );
        assert_eq!(merged["security"]["auth"]["selectedType"], "openai");
        assert_eq!(merged["security"]["auth"]["baseUrl"], url);
        assert_eq!(merged["security"]["auth"]["apiKey"], "keep-too");
        assert_eq!(merged["model"]["name"], "docker.io/ai/m:latest");
        assert_eq!(merged["model"]["baseUrl"], url);
        assert_eq!(merged["model"]["generationConfig"]["temperature"], 0.1);
    }

    /// From nothing, and then again: the second merge changes nothing,
    /// so `write_qwen_settings_at` leaves a correct file alone. No key
    /// value and no `env` block anywhere in it.
    #[test]
    fn qwen_settings_merge_is_complete_from_nothing_and_idempotent() {
        let url = "http://127.0.0.1:17434/v1";
        let once = qwen_settings_merged(&serde_json::json!({}), "m:latest", url, false);
        assert_eq!(
            once,
            serde_json::json!({
                "modelProviders": { "openai": [ { "id": "m:latest",
                    "name": "m:latest (llmman)", "baseUrl": url,
                    "envKey": "LLMMAN_API_KEY" } ] },
                "security": { "auth": { "selectedType": "openai", "baseUrl": url } },
                "model": { "name": "m:latest", "baseUrl": url }
            })
        );
        assert_eq!(qwen_settings_merged(&once, "m:latest", url, false), once);
        let text = once.to_string();
        assert!(!text.contains("apiKey") && !text.contains("\"env\""));
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"qwen"));
    }

    /// The spelling the two sides arrive in differs: `llmman launch qwen
    /// --model gemma4:12b -- --model gemma4:12b` names one model, as
    /// `docker.io/ai/gemma4:12b` and as typed.
    #[test]
    fn qwen_keeps_image_input_only_for_the_model_it_was_read_from() {
        let resolved = crate::shortnames::resolve_ollama_api("gemma4:12b").unwrap();
        let forwarded = |m: &str| vec![String::from("--model"), String::from(m)];
        let cases = [
            // Nothing forwarded, then the same model spelled either way.
            (Vec::new(), resolved.as_str(), true),
            (forwarded("gemma4:12b"), "gemma4:12b", true),
            (forwarded(&resolved), resolved.as_str(), true),
            // Another model, and one that is not a reference at all.
            (forwarded("qwen3.5:0.8b"), "qwen3.5:0.8b", false),
            (forwarded("not a reference"), "not a reference", false),
        ];
        for (extra_args, model, vision) in cases {
            assert_eq!(
                qwen_model_and_vision(&resolved, true, &extra_args),
                (model, vision),
                "{extra_args:?}"
            );
        }
    }

    /// Relaunching with a text model drops the declaration, since
    /// llmman's entry is replaced whole.
    #[test]
    fn qwen_settings_declare_image_input_only_for_a_vision_model() {
        let url = "http://h/v1";
        let vision = qwen_settings_merged(&serde_json::json!({}), "m", url, true);
        assert_eq!(
            vision["modelProviders"]["openai"][0]["generationConfig"],
            serde_json::json!({ "modalities": { "image": true } })
        );
        assert!(vision["model"].get("generationConfig").is_none());

        let text_only = qwen_settings_merged(&vision, "m", url, false);
        assert!(!text_only.to_string().contains("modalities"), "{text_only}");
    }

    /// A wrong-typed value on the path is replaced, a non-object root
    /// counts as empty, and a `{ protocol, models }` wrapper keeps its
    /// entries.
    #[test]
    fn qwen_settings_merge_replaces_a_wrong_typed_value_on_its_path() {
        let existing = serde_json::json!({
            "security": 3, "modelProviders": { "openai": "x" }, "model": []
        });
        let merged = qwen_settings_merged(&existing, "m", "http://h/v1", false);
        assert_eq!(merged["security"]["auth"]["selectedType"], "openai");
        assert_eq!(merged["modelProviders"]["openai"][0]["id"], "m");
        assert_eq!(merged["model"]["name"], "m");
        let from_null = qwen_settings_merged(&serde_json::json!(null), "m", "http://h/v1", false);
        assert_eq!(from_null["model"]["name"], "m");

        let wrapped = serde_json::json!({
            "$version": 5,
            "modelProviders": { "openai": { "protocol": "openai", "models": [
                { "id": "gpt-5", "baseUrl": "https://api.openai.com/v1", "envKey": "MY_KEY" }
            ] } }
        });
        let merged = qwen_settings_merged(&wrapped, "m", "http://h/v1", false);
        let openai = merged["modelProviders"]["openai"].as_array().unwrap();
        assert_eq!(openai.len(), 2);
        assert_eq!(openai[1]["id"], "gpt-5");
        assert_eq!(merged["$version"], 4, "the version follows the shape");
    }

    /// Ownership is the key name at this daemon's address, whatever the
    /// entry was renamed to; a trailing slash does not make a second
    /// daemon of the same one.
    #[test]
    fn qwen_entry_is_ours_needs_the_key_name_and_the_address() {
        let url = "http://127.0.0.1:17434/v1";
        let ours = serde_json::json!({ "id": "anything", "name": "renamed by the user",
            "baseUrl": "http://127.0.0.1:17434/v1/", "envKey": "LLMMAN_API_KEY" });
        assert!(qwen_entry_is_ours(&ours, url));
        let hand_written = serde_json::json!({ "id": "local-alias", "name": "m (llmman)",
            "baseUrl": url, "envKey": "OPENAI_API_KEY" });
        assert!(!qwen_entry_is_ours(&hand_written, url));
        let elsewhere = serde_json::json!({ "id": "m:latest", "name": "m:latest (llmman)",
            "baseUrl": "http://10.0.0.2:17434/v1", "envKey": "LLMMAN_API_KEY" });
        assert!(!qwen_entry_is_ours(&elsewhere, url));
        assert!(!qwen_entry_is_ours(
            &serde_json::json!("not an object"),
            url
        ));
    }

    /// Comments go, as `strip-json-comments` takes them out for Qwen Code,
    /// and nothing else moves: not a `//` inside a string, not a column.
    #[test]
    fn strip_json_comments_keeps_strings_and_columns() {
        let raw =
            "{\n  // note\n  \"url\": \"http://h//v1\", /* block\n  */ \"q\": \"a\\\"//b\"\n}\n";
        let stripped = strip_json_comments(raw);
        assert_eq!(stripped.chars().count(), raw.chars().count());
        assert_eq!(stripped.lines().count(), raw.lines().count());
        let v: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(v["url"], "http://h//v1");
        assert_eq!(v["q"], "a\"//b");
        assert_eq!(strip_json_comments("{\"a\": 1}"), "{\"a\": 1}");
    }

    /// pi is node, so its home half is the one node reads — the same
    /// `cline_dir` resolves, and the reason #535 stopped using
    /// `dirs::home_dir` alone.
    #[test]
    fn pi_agent_dir_reads_the_home_node_reads() {
        assert_eq!(node_home_dir(), node_user_profile().or_else(dirs::home_dir));
        if cfg!(windows) {
            assert_eq!(node_user_profile(), env_dir("USERPROFILE"));
        } else {
            assert_eq!(node_user_profile(), None);
        }
    }

    #[test]
    fn pi_model_entry_declares_what_the_daemon_serves() {
        let thinks = crate::chat_template::ThinkingControls {
            thinks: true,
            enable_thinking: true,
            efforts: vec!["low", "high"],
        };
        let entry = pi_model_entry("qwen3.5:0.8b", Some(&thinks), true, Some(32768));
        assert_eq!(entry["id"], "qwen3.5:0.8b");
        assert_eq!(entry["input"], serde_json::json!(["text", "image"]));
        assert_eq!(entry["reasoning"], true);
        assert_eq!(entry["contextWindow"], 32768);

        // A text-only model that does not think, and no trained context to
        // declare: pi keeps its own default rather than being told a guess.
        let plain = pi_model_entry("smol", None, false, None);
        assert_eq!(plain["input"], serde_json::json!(["text"]));
        assert_eq!(plain.get("reasoning"), None);
        assert_eq!(plain.get("contextWindow"), None);
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
        let entry = pi_model_entry("qwen3.5:0.8b", None, false, None);
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
            let entry = pi_model_entry("m", None, false, None);
            let merged = pi_models_merged(&existing, "http://s", &entry);
            assert_eq!(merged["providers"]["llmman"]["baseUrl"], "http://s/v1");
            assert_eq!(merged["providers"]["llmman"]["models"][0]["id"], "m");
        }
    }

    /// Nothing was there before, so llmman fills in the fields it owns.
    #[test]
    fn pi_models_merged_writes_the_placeholder_key_into_a_fresh_provider() {
        let entry = pi_model_entry("m", None, false, None);
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
        let first = pi_model_entry("a", None, false, None);
        let one = pi_models_merged(&serde_json::json!({}), "http://s", &first);
        let two = pi_models_merged(&one, "http://s", &pi_model_entry("b", None, false, None));
        let ids: Vec<&str> = two["providers"]["llmman"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn pi_settings_merged_leaves_unrelated_settings_alone() {
        let existing = serde_json::json!({ "theme": "dark", "defaultModel": "old" });
        let merged = pi_settings_merged(&existing, "qwen3.5:0.8b");
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["defaultProvider"], "llmman");
        assert_eq!(merged["defaultModel"], "qwen3.5:0.8b");
    }

    /// A leading `~` is the home directory; anything else is as given.
    #[test]
    fn expand_tilde_reads_the_forms_qwen_code_reads() {
        let home = Path::new("/h");
        assert_eq!(expand_tilde("~/alt", home), PathBuf::from("/h/alt"));
        assert_eq!(expand_tilde("~", home), PathBuf::from("/h"));
        assert_eq!(expand_tilde("/abs", home), PathBuf::from("/abs"));
        assert_eq!(expand_tilde("~user/x", home), PathBuf::from("~user/x"));
    }

    /// The reading and writing half over a directory of its own: a fresh
    /// one gets the file, a correct file is not touched, a commented one
    /// merges with its text kept as `.bak`, a later rewrite of llmman's
    /// own rendering leaves that `.bak` alone while a hand edit with
    /// comments refreshes it, an empty file counts as `{}`, and what is
    /// not JSON is left alone without an error.
    #[test]
    fn write_qwen_settings_at_writes_once_keeps_a_bak_and_refuses_non_json() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-qwen-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let url = "http://127.0.0.1:17434/v1";
        let path = dir.join("settings.json");
        let bak = dir.join("settings.json.bak");
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
        };

        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");
        assert!(!bak.exists(), "nothing to back up on a first write");
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written
        );

        let commented = "{\n  // mine\n  \"ui\": { \"theme\": \"x\" }\n}\n";
        std::fs::write(&path, commented).unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(read()["ui"]["theme"], "x");
        assert_eq!(read()["model"]["name"], "m:latest");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), commented);
        write_qwen_settings_at(&dir, "other:latest", url, false).unwrap();
        assert_eq!(read()["model"]["name"], "other:latest");
        assert_eq!(
            std::fs::read_to_string(&bak).unwrap(),
            commented,
            "llmman's own rendering must not replace the user's backup"
        );
        let edited = "{\n  // edited by hand\n  \"ui\": { \"theme\": \"y\" }\n}\n";
        std::fs::write(&path, edited).unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), edited);

        std::fs::write(&path, "  \n").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(read()["model"]["name"], "m:latest");

        std::fs::write(&path, "{ not json").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
        std::fs::write(&path, "[]").unwrap();
        write_qwen_settings_at(&dir, "m:latest", url, false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[]");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression test for the codex config bug described on
    /// `write_codex_config`'s own doc comment: an older llmman's
    /// `[profiles.llmman]` table (a format current codex refuses to load
    /// at all) must be fully removed, leaving everything else in
    /// `config.toml` untouched.
    #[test]
    fn strip_legacy_llmman_profile_removes_only_that_table() {
        let existing = "\
[some_other_setting]
foo = \"bar\"

[profiles.llmman]
openai_base_url = \"http://127.0.0.1:17434/v1\"

[profiles.other]
model = \"gpt-5\"
";
        let cleaned = strip_legacy_llmman_profile(existing);
        assert!(!cleaned.contains("[profiles.llmman]"));
        assert!(!cleaned.contains("openai_base_url"));
        assert!(cleaned.contains("[some_other_setting]"));
        assert!(cleaned.contains("foo = \"bar\""));
        assert!(cleaned.contains("[profiles.other]"));
        assert!(cleaned.contains("model = \"gpt-5\""));
    }

    #[test]
    fn strip_legacy_llmman_profile_is_a_no_op_without_the_legacy_table() {
        let existing = "[profiles.other]\nmodel = \"gpt-5\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), existing);
    }

    #[test]
    fn strip_legacy_llmman_profile_handles_the_table_at_end_of_file() {
        let existing = "[profiles.llmman]\nopenai_base_url = \"http://127.0.0.1:17434/v1\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), "");
    }

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
            false,
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

        let bare = opencode_config("http://h", "m", "k", &[], false);
        assert!(!bare.contains("variants"), "{bare}");
    }

    #[test]
    fn opencode_config_declares_image_input_only_for_a_vision_model() {
        let text = opencode_config("http://h", "m", "k", &[], true);
        let config: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let model = &config["provider"]["ollama"]["models"]["m"];
        assert_eq!(
            model["modalities"],
            serde_json::json!({ "input": ["text", "image"], "output": ["text"] })
        );
        assert_eq!(model["attachment"], true);

        let text_only = opencode_config("http://h", "m", "k", &[], false);
        assert!(!text_only.contains("modalities"), "{text_only}");
        assert!(!text_only.contains("attachment"), "{text_only}");
    }

    #[test]
    fn opencode_config_escapes_the_model_name() {
        let model = "we\"ird/mo\\del";
        let config: serde_json::Value =
            serde_json::from_str(&opencode_config("http://h", model, "k", &[], false))
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
            opencode_variants(Some(&gemma4)),
            [
                ("none", serde_json::json!({ "reasoningEffort": "none" })),
                (
                    "thinking",
                    serde_json::json!({ "chat_template_kwargs": { "enable_thinking": true } })
                ),
            ]
        );
        let qwen3_8 = ThinkingControls {
            efforts: vec!["low", "xhigh"],
            ..gemma4
        };
        assert_eq!(
            opencode_variants(Some(&qwen3_8)),
            [
                ("none", serde_json::json!({ "reasoningEffort": "none" })),
                ("low", serde_json::json!({ "reasoningEffort": "low" })),
                ("xhigh", serde_json::json!({ "reasoningEffort": "xhigh" })),
            ]
        );
        assert!(opencode_variants(Some(&ThinkingControls::default())).is_empty());
        let fallback = opencode_variants(None);
        assert_eq!(fallback.len(), PORTABLE_THINKING_LEVELS.len());
        assert_eq!(fallback[0].0, "none");
    }

    #[test]
    fn codex_profile_is_a_websocket_free_provider_at_the_daemon() {
        let profile: toml::Value = codex_profile("http://127.0.0.1:17434", None)
            .parse()
            .expect("valid TOML");
        assert_eq!(profile["model_provider"].as_str(), Some("llmman"));
        let provider = &profile["model_providers"]["llmman"];
        assert_eq!(
            provider["base_url"].as_str(),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(provider["env_key"].as_str(), Some("OPENAI_API_KEY"));
        assert_eq!(provider["wire_api"].as_str(), Some("responses"));
        assert_eq!(provider["supports_websockets"].as_bool(), Some(false));
        assert!(
            profile.get("openai_base_url").is_none(),
            "the built-in openai provider is not the one in use"
        );
        assert!(
            profile.get("model_catalog_json").is_none(),
            "no model, no catalog to point at"
        );
    }

    #[test]
    fn codex_profile_names_the_catalog_it_was_given() {
        let path = PathBuf::from("/home/we\"ird/.codex/llmman-model.json");
        let profile: toml::Value = codex_profile("http://h", Some(&path))
            .parse()
            .expect("valid TOML");
        assert_eq!(
            profile["model_catalog_json"].as_str(),
            Some(path.to_str().unwrap())
        );
    }

    #[test]
    fn codex_model_catalog_declares_image_input_only_for_a_vision_model() {
        let catalog: serde_json::Value =
            serde_json::from_str(&codex_model_catalog("m", true, 32768)).expect("valid JSON");
        let entry = &catalog["models"][0];
        assert_eq!(
            entry["input_modalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(entry["slug"], "m");
        assert_eq!(entry["display_name"], "m");
        assert_eq!(entry["context_window"], 32768);
        // Fields codex requires of an entry.
        for key in [
            "context_window",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "truncation_policy",
            "support_verbosity",
            "supported_reasoning_levels",
            "experimental_supported_tools",
        ] {
            assert!(entry.get(key).is_some(), "missing {key}");
        }

        let text_only: serde_json::Value =
            serde_json::from_str(&codex_model_catalog("m", false, 32768)).expect("valid JSON");
        assert_eq!(
            text_only["models"][0]["input_modalities"],
            serde_json::json!(["text"])
        );
    }

    #[test]
    fn codex_context_window_is_what_the_daemon_serves() {
        let cases = [
            // A positive LLMMAN_CONTEXT_LENGTH is the served --ctx-size.
            (Some(16384), Some(32768), 16384),
            (Some(65536), Some(32768), 65536),
            (Some(16384), None, 16384),
            // Unset or 0: the trained context.
            (None, Some(32768), 32768),
            (Some(0), Some(32768), 32768),
            (None, Some(1 << 20), 1 << 20),
            (Some(0), None, CODEX_FALLBACK_CONTEXT_WINDOW),
            (None, None, CODEX_FALLBACK_CONTEXT_WINDOW),
        ];
        for (env, trained, want) in cases {
            assert_eq!(
                codex_context_window(env, trained),
                want,
                "env={env:?} trained={trained:?}"
            );
        }
    }

    /// A shortname like `qwen3.5:0.8b` must round-trip quoted, or the
    /// `:` breaks YAML parsing; the key must never appear literally.
    #[test]
    fn write_dsh_settings_points_at_llmman_with_the_key_in_the_environment() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.yaml");
        write_dsh_settings(&path, "qwen3.5:0.8b", false).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("provider: llmman"));
        assert!(contents.contains("model: \"qwen3.5:0.8b\""));
        assert!(contents.contains(&format!("apiKeyEnv: {DSH_API_KEY_ENV}")));
        assert!(contents.contains("api: openai-completions"));
        assert!(contents.contains(&format!("baseURL: \"{}/v1\"", daemon::server())));
        assert!(contents.contains("id: \"qwen3.5:0.8b\""));
        assert!(!contents.contains("apiKey:"), "no literal key in the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// dsh sends an image only to a model whose `input` lists one — and
    /// must not attach one to a text-only model the daemon would reject.
    #[test]
    fn write_dsh_settings_declares_image_input_only_for_a_vision_model() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-vision-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.yaml");
        write_dsh_settings(&path, "m", true).unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("input: [text, image]"));
        write_dsh_settings(&path, "m", false).unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("input: [text]"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fallback that makes `llmman launch dsh` work without a global
    /// install — and stays out of the way of one that exists.
    #[test]
    fn dsh_falls_back_to_the_published_package_under_npx() {
        let dsh = PathBuf::from("/usr/local/bin/dsh");
        let npx = PathBuf::from("/usr/local/bin/npx");

        // An install wins, and npx is never even looked for.
        assert_eq!(
            dsh_command(Some(dsh.clone()), || panic!("npx looked up anyway")),
            Some((dsh, Vec::new()))
        );
        // Without one, npx runs the package: `--yes` so a first run
        // isn't blocked on a prompt, ahead of dsh's own arguments.
        assert_eq!(
            dsh_command(None, || Some(npx.clone())),
            Some((npx, vec!["--yes".to_string(), DSH_NPM_PACKAGE.to_string()]))
        );
        assert!(DSH_NPM_PACKAGE.starts_with("@deepseek-ai/dsh@"));
        // Neither: "dsh is not installed", not an npm error.
        assert_eq!(dsh_command(None, || None), None);
    }

    #[test]
    fn write_dsh_patch_names_the_settings_document() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-patch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings_path = dir.join("settings.yaml");
        let patch_path = dir.join("llmman.cordis.yml");
        write_dsh_patch(&patch_path, &settings_path).unwrap();
        let contents = std::fs::read_to_string(&patch_path).unwrap();
        assert!(contents.contains("id: settings"));
        // Through `yaml_quote`, not the raw path: on Windows a path's
        // `\` separators are escaped in the document, so the raw string
        // never matches (a real red Windows CI leg).
        assert!(contents.contains(&format!(
            "path: {}",
            yaml_quote(&settings_path.to_string_lossy())
        )));
        let _ = std::fs::remove_dir_all(&dir);

        // A Windows-shaped path on every platform, so the escaping this
        // depends on is covered without needing the Windows CI leg to
        // be the thing that catches it (which is how it was caught).
        let win = Path::new(r"C:\Users\hb\.config\llmman\launch\dsh\settings.yaml");
        let rendered = dsh_patch_document(win);
        assert!(rendered.contains(r#"path: "C:\\Users\\hb\\"#), "{rendered}");
        assert!(!rendered.contains(r#"path: "C:\Users"#), "{rendered}");
    }

    /// Past dsh's own `--` boundary, a token is an app argument rather
    /// than a launcher flag (verified against dsh 0.1.2-rc.1), so
    /// neither check may scan there: a task whose text is `--patch`
    /// must not be refused, and one reading `--profile` must not
    /// suppress the default `web`.
    #[test]
    fn dsh_checks_stop_at_dshs_own_argument_boundary() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        // Asserted on the boundary helper, never by calling `launch_dsh`
        // itself: past the refusal it goes on to exec dsh and
        // `std::process::exit`, which on a machine that has dsh
        // installed would take the test runner with it.
        let forwarded_patch = args(&["--profile", "headless", "--", "--patch"]);
        assert_eq!(
            dsh_launcher(&forwarded_patch).args,
            args(&["--profile", "headless"])
        );
        assert!(!has_flag(
            dsh_launcher(&forwarded_patch).args,
            "--patch",
            None
        ));

        let forwarded_profile = args(&["--", "--profile", "headless"]);
        assert!(dsh_launcher(&forwarded_profile).args.is_empty());
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &forwarded_profile),
            ["web", "--patch", "/p.yml", "--", "--profile", "headless"]
        );

        // The same boundary without a `--`: dsh stops at the first
        // token of its own it doesn't recognize, so a task's wording
        // is app text, not flags. `--profile headless` before it is
        // still dsh's, value stepped over rather than read as the
        // first app argument.
        let task_mentions_patch = args(&["--profile", "headless", "explain", "the", "--patch"]);
        assert_eq!(
            dsh_launcher(&task_mentions_patch).args,
            args(&["--profile", "headless"])
        );
        assert!(!has_flag(
            dsh_launcher(&task_mentions_patch).args,
            "--patch",
            None
        ));

        // An app-level `--profile` past that boundary must not suppress
        // the default `web`.
        let app_level_profile = args(&["sometask", "--profile", "headless"]);
        assert!(dsh_launcher(&app_level_profile).args.is_empty());
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &app_level_profile),
            [
                "web",
                "--patch",
                "/p.yml",
                "sometask",
                "--profile",
                "headless"
            ]
        );

        // Bare flags take no value, and the `=`-joined spelling is dsh's
        // own either way.
        assert_eq!(
            dsh_launcher(&args(&["--dump-config", "task"])).args,
            args(&["--dump-config"])
        );
        assert_eq!(
            dsh_launcher(&args(&["--profile=headless", "task"])).args,
            args(&["--profile=headless"])
        );

        // dsh's command spellings are found where dsh itself reads them
        // (before any app argument), so `launch_dsh` can refuse them up
        // front: dsh rejects a command combined with a parent --patch,
        // which this always injects (verified against 0.1.2-rc.1).
        for command in ["web", "plugin"] {
            let via_command = args(&[command, "--port", "8080"]);
            assert_eq!(dsh_launcher(&via_command).command, Some(command));
            let err = launch_dsh("m", "k", false, &via_command).unwrap_err();
            assert!(err.to_string().contains("--profile"), "{err}");
        }
        // `--profile web`'s *value* is not the `web` command — refusing
        // it would break the most ordinary explicit invocation there is
        // (a real bug this caught, found only by running it).
        let profile_web = args(&["--profile", "web", "--no-open"]);
        assert_eq!(dsh_launcher(&profile_web).command, None);
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &profile_web),
            ["--patch", "/p.yml", "--profile", "web", "--no-open"]
        );
        // Same for a patch path that happens to be named `web`.
        assert_eq!(dsh_launcher(&args(&["--patch", "web"])).command, None);
        // Past the boundary it is app text, not a command.
        assert_eq!(dsh_launcher(&args(&["sometask", "web"])).command, None);

        // All launcher flags, no app arguments: the whole slice is dsh's.
        let plain = args(&["--profile", "headless"]);
        assert_eq!(dsh_launcher(&plain).args, plain);
        // A value-taking flag with its value missing must not run past
        // the end of the slice.
        assert_eq!(
            dsh_launcher(&args(&["--profile"])).args,
            args(&["--profile"])
        );
    }

    /// A caller-supplied `--patch` after `--` must be refused, however spelled.
    #[test]
    fn launch_dsh_refuses_a_conflicting_patch_flag() {
        let word = vec!["--patch".to_string(), "/tmp/x.yml".to_string()];
        let err = launch_dsh("m", "k", false, &word).unwrap_err();
        assert!(err.to_string().contains("--patch"), "{err}");
        let joined = vec!["--patch=/tmp/x.yml".to_string()];
        let err = launch_dsh("m", "k", false, &joined).unwrap_err();
        assert!(err.to_string().contains("--patch"), "{err}");
    }

    /// dsh carries a real key per launch, unlike hermes, so it belongs
    /// on neither `--provider` refusal list.
    #[test]
    fn dsh_is_a_real_integration_and_not_on_a_provider_refusal_list() {
        assert!(INTEGRATIONS.iter().any(|i| i.name == "dsh"));
        assert!(!PROVIDER_UNSUPPORTED.iter().any(|(id, _)| *id == "dsh"));
        assert!(!PROVIDER_NEEDS_DAEMON_KEY.contains(&"dsh"));
    }

    /// A caller-supplied `--profile` (however spelled) must win over the
    /// default `web`, since `web` doesn't accept `--profile` at all;
    /// `--patch` is injected either way and nothing else is reordered.
    #[test]
    fn dsh_args_defaults_to_web_but_yields_to_a_caller_supplied_profile() {
        let path = Path::new("/tmp/x/llmman.cordis.yml");
        let none: Vec<String> = vec![];
        assert_eq!(
            dsh_args(path, &none),
            ["web", "--patch", "/tmp/x/llmman.cordis.yml"]
        );

        let headless = vec![
            "--profile".to_string(),
            "headless".to_string(),
            "hi".to_string(),
        ];
        assert_eq!(
            dsh_args(path, &headless),
            [
                "--patch",
                "/tmp/x/llmman.cordis.yml",
                "--profile",
                "headless",
                "hi"
            ]
        );

        let joined = vec!["--profile=headless".to_string(), "hi".to_string()];
        assert_eq!(
            dsh_args(path, &joined),
            [
                "--patch",
                "/tmp/x/llmman.cordis.yml",
                "--profile=headless",
                "hi"
            ]
        );
    }

    /// Regression test for `write_hermes_config` preserving unrelated
    /// `config.yaml` content: only its own `model:`/`providers:` blocks
    /// are replaced, not a user's other settings.
    #[test]
    fn strip_yaml_top_level_key_removes_only_that_key_and_its_block() {
        let existing = "\
toolsets:\n  - web\nmodel:\n  provider: llmman\n  default: old-model\nproviders:\n  llmman:\n    name: llmman\nchannels:\n  telegram: {}\n";
        let cleaned =
            strip_yaml_top_level_key(&strip_yaml_top_level_key(existing, "model"), "providers");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("providers:"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("  - web"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }

    #[test]
    fn hermes_config_declares_image_input_only_for_a_vision_model() {
        let vision = hermes_config_blocks("m", "http://h/v1", true);
        let model_block = strip_yaml_top_level_key(&vision, "providers");
        assert!(
            model_block.contains("\n  supports_vision: true\n"),
            "{vision}"
        );

        let text_only = hermes_config_blocks("m", "http://h/v1", false);
        assert!(!text_only.contains("supports_vision"), "{text_only}");
        assert_eq!(text_only, vision.replace("  supports_vision: true\n", ""));
    }

    #[test]
    fn strip_yaml_top_level_key_is_a_no_op_without_that_key() {
        let existing = "toolsets:\n  - web\n";
        assert_eq!(strip_yaml_top_level_key(existing, "model"), existing);
    }

    /// Regression test for a real CodeRabbit finding: a blank line inside
    /// the block being removed used to reset `skipping`, leaking the rest
    /// of that block into the output instead of removing it.
    #[test]
    fn strip_yaml_top_level_key_handles_blank_lines_inside_the_removed_block() {
        let existing = "toolsets:\n  - web\n\nmodel:\n  provider: llmman\n\n  default: old-model\n\nchannels:\n  telegram: {}\n";
        let cleaned = strip_yaml_top_level_key(existing, "model");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("default: old-model"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }

    /// Same regression as the blank-line case above, but for a column-0
    /// `#` comment (another real CodeRabbit finding).
    #[test]
    fn strip_yaml_top_level_key_handles_a_comment_inside_the_removed_block() {
        let existing = "toolsets:\n  - web\n# a comment\nmodel:\n  provider: llmman\n# another comment\n  default: old-model\nchannels:\n  telegram: {}\n";
        let cleaned = strip_yaml_top_level_key(existing, "model");
        assert!(!cleaned.contains("model:"));
        assert!(!cleaned.contains("provider: llmman"));
        assert!(!cleaned.contains("default: old-model"));
        assert!(!cleaned.contains("another comment"));
        assert!(cleaned.contains("toolsets:"));
        assert!(cleaned.contains("# a comment"));
        assert!(cleaned.contains("channels:"));
        assert!(cleaned.contains("  telegram: {}"));
    }
}
