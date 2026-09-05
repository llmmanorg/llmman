//! `llmman launch` — launch AI coding-assistant integrations backed by llmman serve.
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

use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use clap::Args;

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

    /// Extra arguments forwarded to the integration binary (after --)
    #[arg(last = true, value_name = "ARGS")]
    pub extra_args: Vec<String>,
}

pub fn run(args: &LaunchArgs) -> anyhow::Result<()> {
    let provider = providers::provider_flag(args.provider.as_deref())?;

    let Some(ref name) = args.integration else {
        print_integrations();
        return Ok(());
    };

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
                crate::daemon::ensure_model_pulled(&model)?;
            }
            (model, providers::PLACEHOLDER_API_KEY.to_string())
        }
    };

    launch(name, &model, &api_key, &args.extra_args)
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// Integrations `--provider` cannot drive, and why.
///
/// `launch_simple` only exports `OLLAMA_HOST`: it never passes a model,
/// so the integration picks its own and the provider-routed reference
/// never reaches the daemon. `copilot` takes a model but has no way to
/// carry a key. Refusing is the same call the catalog filter makes — a
/// combination llmman cannot actually drive is absent, not offered and
/// then broken at the first request.
const PROVIDER_UNSUPPORTED: &[(&str, &str)] = &[
    (
        "cline",
        "it selects its own model rather than taking one from llmman",
    ),
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
];

/// Integrations llmman configures through a file on disk. They take a
/// model on every launch, so `--provider` works, but they cannot carry
/// the key: writing a real one into `~/.hermes/config.yaml` (or into
/// Talos's env file) would persist a
/// credential, which this feature promises not to do. They rely on
/// `llmman serve` having the variable itself — which it only uses for a
/// daemon nobody else can reach (see `reachable_only_locally`).
const PROVIDER_NEEDS_DAEMON_KEY: &[&str] = &["hermes", "talos"];

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
    // bind is fine here — that hop is still loopback.
    if !crate::daemon::connects_over_loopback() {
        anyhow::bail!(
            "--provider needs a local llmman serve: LLMMAN_HOST points at {}, and the \
             provider key would cross the network in cleartext.\n\
             Export the key where that daemon runs instead.",
            crate::daemon::server()
        );
    }
    // These reach the daemon over loopback, so the check above passes,
    // but they send the placeholder key and the daemon will not fall back
    // to its own on a bind anyone can reach. Say so here rather than let
    // it surface as a 401 from inside the integration.
    if PROVIDER_NEEDS_DAEMON_KEY.contains(&name.as_str())
        && !crate::daemon::reachable_only_locally()
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
    // cmd::serve), never to disk or a command line.
    //
    // The placeholder goes instead whenever the daemon's key is the one
    // that matters — an integration that cannot carry one, or a shell
    // without one where the daemon has it — since that is what makes
    // serve fall back to its own.
    let key = match (entry.api_key(), key_travels_per_request) {
        (Some(key), true) => key,
        (_, false) => {
            // Fatal, not a warning: this integration cannot carry a key,
            // so the daemon's is the only one its first request can use,
            // and `key_usable` is the daemon's own word on whether it
            // would spend it. Warning and handing off would surface as a
            // 401 inside someone else's TUI.
            anyhow::ensure!(
                entry.key_usable,
                "{integration} is configured through a file, so it cannot send an API key: \
                 llmman serve needs {} in its own environment, and must be bound to \
                 loopback to spend it.\n\
                 Export it where the daemon runs and restart it.",
                entry.key_env
            );
            providers::PLACEHOLDER_API_KEY.to_string()
        }
        (None, true) if entry.daemon_key_usable() => {
            eprintln!(
                "[llmman] warning: {} is unset here; using the key llmman serve has",
                entry.key_env
            );
            providers::PLACEHOLDER_API_KEY.to_string()
        }
        (None, true) => anyhow::bail!(
            "no API key for {} — set {} in your environment",
            entry.name,
            entry.key_env
        ),
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
    /// Where this integration lives when it is not on `PATH` — the
    /// install locations its own launcher knows to look in. `None` means
    /// `PATH` is the only place, looked up by `binary`. The listing and
    /// the launch go through the same resolver, so what counts as
    /// installed is one answer, not two.
    find: Option<fn() -> Option<PathBuf>>,
}

const INTEGRATIONS: &[Integration] = &[
    Integration {
        name: "claude",
        description: "Claude Code",
        binary: "claude",
        find: None,
    },
    Integration {
        name: "opencode",
        description: "OpenCode",
        binary: "opencode",
        find: None,
    },
    Integration {
        name: "codex",
        description: "OpenAI Codex CLI",
        binary: "codex",
        find: None,
    },
    Integration {
        name: "cline",
        description: "Cline",
        binary: "cline",
        find: None,
    },
    Integration {
        name: "aider",
        description: "Aider AI pair programmer",
        binary: "aider",
        find: None,
    },
    Integration {
        name: "copilot",
        description: "GitHub Copilot CLI",
        binary: "gh",
        find: None,
    },
    Integration {
        name: "kimi",
        description: "Kimi Code CLI",
        binary: "kimi",
        find: None,
    },
    Integration {
        name: "gemini",
        description: "Gemini CLI",
        binary: "gemini",
        find: None,
    },
    Integration {
        name: "hermes",
        description: "Hermes Agent",
        binary: "hermes",
        find: None,
    },
    Integration {
        name: "openclaw",
        description: "OpenClaw",
        binary: "openclaw",
        find: None,
    },
    Integration {
        name: "qwen",
        description: "Qwen Code",
        binary: "qwen",
        find: None,
    },
    Integration {
        name: "talos",
        description: "Talos",
        binary: "talos",
        find: Some(find_talos),
    },
];

fn print_integrations() {
    println!("Available integrations:\n");
    for i in INTEGRATIONS {
        if integration_installed(i) {
            println!("  {:<12} {}", i.name, i.description);
        } else {
            println!("  {:<12} {} (not installed)", i.name, i.description);
        }
    }
    println!("\nUsage: llmman launch <integration> [--model <model>] [--provider <provider>]");
    println!("       llmman providers   (the providers --provider accepts)");
}

/// Whether an integration is installed, for the listing above. On PATH
/// for all of them but Talos, whose installer deliberately adds nothing
/// to PATH (see [`talos_command`]) — a listing that said "not installed"
/// for every Talos that is would teach people to ignore the column.
fn integration_installed(i: &Integration) -> bool {
    find_integration_binary(i).is_some()
}

/// The binary `launch` will run for `i`: `PATH`, then whatever the
/// integration's own resolver (`Integration::find`) knows.
fn find_integration_binary(i: &Integration) -> Option<PathBuf> {
    match i.find {
        Some(find) => find(),
        None => find_on_path(i.binary),
    }
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
fn launch(name: &str, model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    match name.to_lowercase().as_str() {
        "claude" => launch_claude(model, api_key, extra_args),
        "opencode" => launch_opencode(model, api_key, extra_args),
        "codex" => launch_codex(model, api_key, extra_args),
        "cline" => launch_simple("cline", model, extra_args),
        "aider" => launch_aider(model, api_key, extra_args),
        "copilot" | "copilot-cli" => launch_copilot(model, extra_args),
        "kimi" => launch_simple("kimi", model, extra_args),
        "gemini" => launch_gemini(model, api_key, extra_args),
        "hermes" => launch_hermes(model, extra_args),
        "openclaw" => launch_openclaw(model, extra_args),
        "qwen" => launch_qwen(model, api_key, extra_args),
        "talos" => launch_talos(model, extra_args),
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

/// opencode: pass a JSON config via OPENCODE_CONFIG_CONTENT pointing at our
/// /v1 endpoint, matching exactly what ollama launch does.
fn launch_opencode(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("opencode").or_else(|| {
        dirs::home_dir().and_then(|h| {
            let p = h.join(".opencode").join("bin").join("opencode");
            p.exists().then_some(p)
        })
    });
    let bin = bin.ok_or_else(|| anyhow::anyhow!("opencode is not installed"))?;

    let effective_model = if model.is_empty() { "default" } else { model };
    let config = opencode_config(effective_model, api_key);

    exec_with_env(&bin, extra_args, &[("OPENCODE_CONFIG_CONTENT", &config)])
}

fn opencode_config(model: &str, api_key: &str) -> String {
    let base_url = format!("{}/v1", daemon::server());
    serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama",
                "options": {
                    "baseURL": base_url,
                    "apiKey": api_key
                },
                "models": {
                    model: { "name": model }
                }
            }
        },
        "model": format!("ollama/{model}")
    })
    .to_string()
}

/// codex: set OPENAI_API_KEY=llmman and write ~/.codex/config.toml with the
/// ollama provider pointing at our /v1 endpoint.
fn launch_codex(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    // Write codex config
    write_codex_config()?;

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
fn write_codex_config() -> anyhow::Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let config_dir = home.join(".codex");
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if let Ok(existing) = std::fs::read_to_string(&config_path) {
        if existing.contains("[profiles.llmman]") {
            std::fs::write(&config_path, strip_legacy_llmman_profile(&existing))?;
        }
    }

    let profile_path = config_dir.join("llmman.config.toml");
    let contents = format!("openai_base_url = \"{}/v1\"\n", daemon::server());
    // Avoid rewriting (and bumping the mtime of) a file that's already
    // correct.
    if std::fs::read_to_string(&profile_path).ok().as_deref() != Some(contents.as_str()) {
        std::fs::write(&profile_path, contents)?;
    }
    Ok(())
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
fn launch_hermes(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("hermes").ok_or_else(|| anyhow::anyhow!("hermes is not installed"))?;
    write_hermes_config(if model.is_empty() { "default" } else { model })?;
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
fn write_hermes_config(model: &str) -> anyhow::Result<()> {
    let config_dir = hermes_home()?;
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("config.yaml");

    let existing = std::fs::read_to_string(&config_path).unwrap_or_default();
    let preserved =
        strip_yaml_top_level_key(&strip_yaml_top_level_key(&existing, "model"), "providers");

    // Double-quoted (not bare) so a model name that happens to be a YAML
    // keyword (`null`, `true`, ...) or contain metacharacters (`:`, `#`,
    // ...) still parses back as the literal string it is.
    let model = yaml_quote(model);
    let base_url = yaml_quote(&format!("{}/v1", daemon::server()));
    let ours = format!(
        "model:\n  provider: llmman\n  default: {model}\n  base_url: {base_url}\n  api_key: llmman\n\
         providers:\n  llmman:\n    name: llmman\n    api: {base_url}\n    default_model: {model}\n    models:\n      - {model}\n"
    );
    std::fs::write(&config_path, format!("{preserved}{ours}"))?;
    Ok(())
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

/// qwen: Qwen Code's OpenAI-compatible mode, pointed at our /v1.
///
/// The auth type and model go on the command line, not only in the
/// environment. Qwen Code resolves both as `argv`, then its own
/// `~/.qwen/settings.json`, then the `OPENAI_*` variables, and it writes
/// that file itself whenever a user picks a provider or model with
/// `/auth` or `/model`. For anyone who has used it before, variables alone
/// lose: the session went to the cloud provider recorded there, with that
/// provider's real key, and reported success. Ollama's own
/// `cmd/launch/qwen.go` prepends the same two flags.
///
/// The base URL and key stay in the environment even though Qwen Code
/// has `--openai-base-url` and `--openai-api-key`: a key on the command
/// line is visible in `ps` to anyone on the host, and for these two
/// Qwen Code reads the variables ahead of its settings file, so the flags
/// buy nothing. The one thing that still precedes them is a
/// `modelProviders` entry whose id equals the launched model, which the
/// `docker.io/…` and `hf.co/…` references llmman hands over do not
/// collide with.
///
/// `--model` is required. Qwen Code has no notion of a missing model:
/// given none it sends its own built-in default (`qwen3.7-max` in
/// 0.22.3), which the daemon would then try to pull as
/// `docker.io/ai/qwen3.7-max` and fail on, minutes later, naming a model
/// the caller did not ask for. Refusing here names the flag instead.
fn launch_qwen(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("qwen").ok_or_else(|| anyhow::anyhow!("qwen is not installed"))?;
    if model.is_empty() {
        anyhow::bail!("qwen needs a model: llmman launch qwen --model <model>");
    }

    let base_url = format!("{}/v1", daemon::server());
    exec_with_env(
        &bin,
        &qwen_args(model, extra_args),
        &[
            ("OPENAI_BASE_URL", base_url.as_str()),
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_MODEL", model),
        ],
    )
}

/// `--auth-type openai --model <model>` ahead of the caller's own
/// arguments, each dropped when the caller already passed it after `--`:
/// Qwen Code 0.22.3 crashes on either flag repeated (a `toLowerCase`
/// TypeError) rather than taking the last one, and a caller who spelled
/// out an auth type meant it. `--authType` is checked too — yargs accepts
/// a flag's camelCase spelling as well.
fn qwen_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let has = |long: &str, short: Option<&str>| {
        extra_args.iter().any(|a| {
            a == long
                || a.starts_with(&format!("{long}="))
                || short.is_some_and(|s| a == s || a.starts_with(&format!("{s}=")))
        })
    };
    let mut args = Vec::with_capacity(extra_args.len() + 4);
    if !has("--auth-type", Some("--authType")) {
        args.extend(["--auth-type".to_string(), "openai".to_string()]);
    }
    if !has("--model", Some("-m")) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// Talos's local provider (talos/catalog.py): OpenAI-compatible wire, no
/// key, and a default address of Ollama's port — which is why the
/// daemon's address has to travel alongside it (see `launch_talos`).
const TALOS_PROVIDER: &str = "ollama";
/// The two keys `launch_talos` writes, both `SETTING` in Talos's config
/// schema (talos/schema.py) — the class a script may set.
const TALOS_PROVIDER_KEY: &str = "TALOS_MODEL_PROVIDER";
const TALOS_MODEL_KEY: &str = "TALOS_MODEL";
/// Where Talos reads the local provider's address from. `POLICY` in its
/// schema (a route to a model) — the operator's, not a launcher's — so it is
/// set in the launched process's environment instead, never on disk.
const TALOS_BASE_URL_KEY: &str = "TALOS_BASE_URL_OLLAMA";

/// talos: a permission-gated agent (talos-agent.ch) whose kernel rules on
/// every action before it runs. Its configuration is a flat env file,
/// and this writes two keys into it the way `write_hermes_config` writes
/// its YAML — into the file Talos's own `config set` would pick (see
/// `talos_env_file`), keeping every other line, atomically, mode 600
/// (see `write_talos_env`). Both keys are `SETTING` in Talos's schema,
/// i.e. what a script may set: `TALOS_MODEL_PROVIDER=ollama` (Talos's
/// local provider: OpenAI-compatible wire, no key) and
/// `TALOS_MODEL=<model>`.
///
/// The daemon's address does not go into the file. Talos's local
/// provider defaults to Ollama's port, and the per-provider base URL
/// key is `POLICY` in its schema — the operator's to set, not a
/// launcher's. Talos reads its environment ahead of its env
/// file, so `TALOS_BASE_URL_OLLAMA` is set for the process this launches
/// — per launch, never persisted — the same way the other launchers hand
/// over their base URLs. Without it Talos would talk to `:11434`, not to
/// `llmman serve`.
///
/// Hands off to `talos chat`, an interactive session in this terminal
/// where actions the kernel classes as needs-human wait for the person
/// at it. A leading `ask` after `--` runs Talos's one-turn command
/// instead (`llmman launch talos --model m -- ask "…"`; answer on
/// stdout), which is what the e2e test drives — `chat` counts a terminal
/// as attended only when stdin and stdout both are one.
///
/// Two things this leaves to Talos on purpose. The terminal has to be in
/// its allowlist (`cli:<uid>` in `TALOS_ALLOWED_PRINCIPALS`, a policy
/// key llmman does not touch); a fresh install refuses with the exact
/// line to add. And a model picked with `/model` inside a session
/// persists in Talos's event log and outranks its env file on the next
/// start — Talos reports the model it runs, and `/model` in the session
/// changes it. Neither is llmman's to override: both are the kernel
/// deciding who may talk to it and with what.
///
/// `--model` is required: with none, Talos would start on whatever its
/// env file or shipped default names, which is not what the caller asked
/// for. An exported `TALOS_MODEL_PROVIDER`/`TALOS_MODEL` that disagrees
/// with the launch is refused before this function writes anything: it
/// outranks the env file in Talos's own loader, so the launch would
/// otherwise save and report one model while the process runs another.
/// That guarantee is local to this function, not the daemon: `run()`
/// above already calls `ensure_server`/`ensure_model_pulled` for every
/// integration before dispatch, so a refusal here can still follow a
/// daemon start or a model pull — same as it would for any other
/// integration's own refusal.
///
/// macOS and Linux only — Talos's installer ships no Windows path.
fn launch_talos(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    if cfg!(windows) {
        anyhow::bail!("talos currently supports macOS and Linux");
    }
    let talos = talos_command().ok_or_else(|| anyhow::anyhow!("talos is not installed"))?;
    if model.is_empty() {
        anyhow::bail!("talos needs a model: llmman launch talos --model <model>");
    }
    talos_check_env_overrides(model, |key| std::env::var(key).ok())?;

    // Where to guess the env file lives: an explicit TALOS_PREFIX always
    // wins (the operator's own claim), otherwise only the venv form's own
    // `cwd` — where `talos_command` actually found Talos, not a guess.
    // A shim on PATH carries no such location; `talos_env_file` is left
    // to fall back to TALOS_SECRETS_ENV alone (or refuse) rather than aim
    // at a `~/talos` that may not be where the shim's Talos lives.
    let config_prefix = talos_prefix_override().or_else(|| talos.cwd.clone());
    let secrets_env_absolute = absolute_env_path("TALOS_SECRETS_ENV")?;
    let env_file = talos_env_file(
        config_prefix.as_deref(),
        dirs::home_dir().as_deref(),
        |key| {
            if key == "TALOS_SECRETS_ENV" {
                secrets_env_absolute
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned())
            } else {
                std::env::var(key).ok()
            }
        },
    )
    .ok_or_else(|| {
        anyhow::anyhow!(
            "cannot tell where talos reads its env file: set TALOS_PREFIX (the install \
             directory holding talos.env) or TALOS_SECRETS_ENV"
        )
    })?;
    write_talos_env(
        &env_file,
        &[
            (TALOS_PROVIDER_KEY, TALOS_PROVIDER),
            (TALOS_MODEL_KEY, model),
        ],
    )
    .with_context(|| format!("failed to write {}", env_file.display()))?;

    let base_url = format!("{}/v1", daemon::server());
    let (bin, args) = talos_exec_argv(&talos, extra_args);
    let mut extra_env: Vec<(&str, &str)> = vec![(TALOS_BASE_URL_KEY, base_url.as_str())];
    // Talos's own cwd changes to the prefix for the venv form (see
    // `TalosCommand`'s doc comment) — a relative TALOS_SECRETS_ENV would
    // then resolve differently for Talos than it just did for the write
    // above. Overlaying the absolute value keeps both resolutions the
    // same file; the shim form's cwd never changes, so this only matters
    // for the venv form.
    let secrets_env_owned = if talos.cwd.is_some() {
        secrets_env_absolute
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
    } else {
        None
    };
    if let Some(value) = &secrets_env_owned {
        extra_env.push(("TALOS_SECRETS_ENV", value.as_str()));
    }
    exec_with_env_in(&bin, &args, &extra_env, talos.cwd.as_deref())
}

/// `key`'s value, absolutized — `None` when unset or empty. `~` and
/// `~/…` expand to the home directory first — Talos's own loader reads
/// this exact variable through Python's `Path(...).expanduser()`, so a
/// value like `~/.secrets/talos.env` has to land on the same file here,
/// not literally under a directory named `~`, which is what
/// `std::path::absolute` alone would do with it. `~name/…` (someone
/// else's home) is left untouched — rare enough here that guessing wrong
/// is worse than not expanding it. Otherwise lexical only (no symlink
/// resolution, no existence requirement), the same way
/// [`talos_prefix_override`] absolutizes `TALOS_PREFIX`.
fn absolute_env_path(key: &str) -> anyhow::Result<Option<PathBuf>> {
    let Some(raw) = std::env::var(key)
        .ok()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
    else {
        return Ok(None);
    };
    expand_and_absolutize(&raw, dirs::home_dir().as_deref())
        .map(Some)
        .with_context(|| format!("{key}={raw:?} could not be resolved to an absolute path"))
}

/// `~`/`~/…` expansion, with `home` injected so this is testable without
/// touching the real process environment (unlike the var lookup itself,
/// this part is pure). `~name/…` (someone else's home) is left as-is —
/// rare enough here that guessing wrong is worse than not expanding it.
fn expand_and_absolutize(raw: &str, home: Option<&std::path::Path>) -> std::io::Result<PathBuf> {
    let expanded = if raw == "~" {
        home.map(std::path::Path::to_path_buf)
    } else {
        raw.strip_prefix("~/")
            .and_then(|rest| home.map(|h| h.join(rest)))
    }
    .unwrap_or_else(|| PathBuf::from(raw));
    std::path::absolute(expanded)
}

/// How Talos is run: the command line, and the directory it has to be
/// run from. The installer does not install the package into the venv;
/// `-m talos` finds it on the current directory, which is why its own
/// instructions read `cd ~/talos && .venv/bin/python -m talos …` — so
/// the venv form carries the prefix as its working directory, and a
/// `talos` shim on PATH carries none.
#[derive(Debug, PartialEq)]
struct TalosCommand {
    argv: Vec<String>,
    cwd: Option<PathBuf>,
}

/// How to invoke Talos: a `talos` shim on PATH when the user made one,
/// else the venv interpreter under the install prefix plus `-m talos` —
/// the canonical invocation, since the installer puts everything under
/// `~/talos` (or `$TALOS_PREFIX`) and deliberately adds nothing to PATH.
fn talos_command() -> Option<TalosCommand> {
    talos_command_in(find_on_path, talos_prefix().as_deref())
}

/// The path `Integration::find` reports for talos: the shim on `PATH`,
/// else the venv interpreter under the prefix — whichever
/// [`talos_command`] would run.
fn find_talos() -> Option<PathBuf> {
    talos_command().map(|talos| PathBuf::from(&talos.argv[0]))
}

/// `$TALOS_PREFIX`, absolutized — `None` when unset or empty. Kept apart
/// from [`talos_prefix`]'s `~/talos` default: an env-file target guessed
/// from that default would aim at the wrong place for a `talos` shim
/// installed somewhere else, so [`launch_talos`] only falls back to it
/// for the venv form, where the prefix is where the venv was actually
/// found rather than a guess (see its own `config_prefix`).
///
/// Absolutized because it is handed to Talos twice over: once here, to
/// build the env-file path, and once as `current_dir` for the exec'd
/// process. A relative value would resolve against llmman's cwd for the
/// first and then AGAIN relative to itself for the second (Talos's own
/// `current_dir` becoming its own base), doubling the path. Lexical only
/// (`std::path::absolute`, no symlink resolution, no existence
/// requirement) — the directory need not exist yet when this runs.
fn talos_prefix_override() -> Option<PathBuf> {
    let raw = std::env::var("TALOS_PREFIX")
        .ok()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())?;
    Some(std::path::absolute(&raw).unwrap_or_else(|_| PathBuf::from(raw)))
}

/// Talos's install directory: `$TALOS_PREFIX`, else `~/talos`, where the
/// installer puts everything. `None` only when neither can be resolved —
/// a shim on `PATH` needs no prefix, so this is asked for after that.
fn talos_prefix() -> Option<PathBuf> {
    talos_prefix_override().or_else(|| dirs::home_dir().map(|h| h.join("talos")))
}

/// [`talos_command`] with its two inputs explicit, so it can be tested
/// without touching PATH or the environment. A shim on `PATH` wins
/// before a prefix is even needed.
fn talos_command_in(
    lookup: impl Fn(&str) -> Option<PathBuf>,
    prefix: Option<&std::path::Path>,
) -> Option<TalosCommand> {
    if let Some(path) = lookup("talos") {
        return Some(TalosCommand {
            cwd: talos_wrapper_prefix(&path),
            argv: vec![path.to_string_lossy().into_owned()],
        });
    }
    let prefix = prefix?;
    let python = prefix.join(".venv").join("bin").join("python");
    if python.is_file() {
        return Some(TalosCommand {
            argv: vec![
                python.to_string_lossy().into_owned(),
                "-m".to_string(),
                "talos".to_string(),
            ],
            cwd: Some(prefix.to_path_buf()),
        });
    }
    None
}

/// Refuses a launch that an exported `TALOS_MODEL_PROVIDER`/`TALOS_MODEL`
/// would silently override (see `launch_talos`). An empty variable is no
/// override — Talos's own loader treats it as unset (Python's `or`, which
/// is falsy only for the literal empty string). NOT trimmed to match: a
/// whitespace-only value is truthy there too, so Talos would use it as
/// the value verbatim — the mismatch this function exists to catch.
fn talos_check_env_overrides(
    model: &str,
    get: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<()> {
    for (key, want) in [
        (TALOS_PROVIDER_KEY, TALOS_PROVIDER),
        (TALOS_MODEL_KEY, model),
    ] {
        if let Some(value) = get(key) {
            if !value.is_empty() && value != want {
                anyhow::bail!(
                    "{key} is set to {value:?} in your environment and overrides Talos's env file\n\
                     Unset it or set it to {want:?}, then re-run: llmman launch talos"
                );
            }
        }
    }
    Ok(())
}

/// The file `launch_talos` writes the model into — the same one Talos's
/// own `config set` picks without `--file`: the secrets env file when it
/// exists (`$TALOS_SECRETS_ENV`, else `~/.secrets/talos-telegram.env` —
/// `home` is injected rather than read here, so a test can supply a fake
/// one instead of the machine's real home directory), otherwise
/// `talos.env` in the install prefix. Talos reads the process
/// environment first, then the secrets file, then the prefix file, so
/// writing the one `config set` would write keeps the launch's choice
/// where Talos will see it.
fn talos_env_file(
    prefix: Option<&std::path::Path>,
    home: Option<&std::path::Path>,
    getenv: impl Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    let secrets = getenv("TALOS_SECRETS_ENV")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| home.map(|h| h.join(".secrets").join("talos-telegram.env")));
    if let Some(secrets) = secrets.filter(|p| p.is_file()) {
        return Some(secrets);
    }
    prefix.map(|p| p.join("talos.env"))
}

/// Sets `KEY=value` pairs in Talos's flat env file the way Talos's own
/// writer does: existing lines are kept (comments, other keys), the first
/// line of each key is replaced and its duplicates dropped, missing keys
/// are appended; the result lands atomically through a temp file created
/// with mode 600 (Talos treats the file as holding secrets). A value that
/// would break the line format is refused rather than written. Values are
/// stripped before writing — Talos's own `config set` returns `value.strip()`
/// (see its `_one_line` validator) and its file-loader strips every value it
/// reads back, so writing the untrimmed form would silently diverge from
/// what Talos itself would have stored for the same input. A symlink AT
/// `path` is refused rather than followed: `rename` below replaces
/// whatever sits at that name, so writing through a symlink to a
/// centrally managed file would silently sever it from that file instead
/// of updating it — Talos's own `write_key` refuses the same thing
/// (`talos/configcli.py`: "refusing a symlink config file").
fn write_talos_env(path: &std::path::Path, pairs: &[(&str, &str)]) -> anyhow::Result<()> {
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        anyhow::bail!(
            "{}: refusing a symlink config file; point TALOS_PREFIX or TALOS_SECRETS_ENV \
             at the real path",
            path.display()
        );
    }
    for (key, value) in pairs {
        if value.contains(['\n', '\r', '\0']) || key.contains(['=', '\n', '\r', '\0']) {
            anyhow::bail!("{key}: value would not survive a KEY=VALUE line");
        }
    }
    let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.trim())).collect();
    let pairs = pairs.as_slice();
    // Only a file that is not there counts as empty: an unreadable or
    // non-UTF-8 one must not be quietly replaced by a fresh file that
    // knows only our two keys — the operator's token and allowlist live
    // in that file too.
    let existing = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let mut out: Vec<String> = Vec::new();
    let mut written: Vec<&str> = Vec::new();
    for line in existing.lines() {
        let trimmed = line.trim();
        let key = trimmed.split_once('=').map(|(k, _)| k.trim());
        match key.and_then(|k| pairs.iter().find(|(pk, _)| *pk == k)) {
            Some((pk, pv)) if !trimmed.starts_with('#') => {
                if !written.contains(pk) {
                    out.push(format!("{pk}={pv}"));
                    written.push(pk);
                }
            }
            _ => out.push(line.to_string()),
        }
    }
    for (pk, pv) in pairs {
        if !written.contains(pk) {
            out.push(format!("{pk}={pv}"));
        }
    }
    let mut content = out.join("\n");
    content.push('\n');

    let tmp = path.with_file_name(format!(
        "{}.tmp-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("talos.env"),
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    let mut open = std::fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    {
        use std::io::Write;
        let mut file = open.open(&tmp)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
    }
    if let Err(err) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err.into());
    }
    Ok(())
}

/// The install prefix behind a `talos` found on `PATH`, when it is the
/// wrapper Talos's own installer leaves there (`site/install.sh`, since
/// 0.18.0): `<prefix>/bin/talos`, symlinked into `~/.local/bin/talos` (or
/// `$TALOS_BIN_DIR`) — a plain `talos` on PATH is this symlink for every
/// fresh install now, not an unknown shim. The wrapper finds its own root
/// the same way (`Path(__file__).resolve().parent.parent`), so resolving
/// the symlink and walking up two directories here mirrors exactly what
/// running it would do, rather than guessing. Anything that does not
/// match this exact shape — a different tool also named `talos`, or an
/// install layout old enough to predate `bin/talos` — returns `None`
/// rather than a wrong guess, same as an unrelated shim always has.
fn talos_wrapper_prefix(shim: &std::path::Path) -> Option<PathBuf> {
    let real = std::fs::canonicalize(shim).ok()?;
    if real.file_name()?.to_str()? != "talos" {
        return None;
    }
    let bin_dir = real.parent()?;
    if bin_dir.file_name()?.to_str()? != "bin" {
        return None;
    }
    let prefix = bin_dir.parent()?.to_path_buf();
    prefix
        .join(".venv")
        .join("bin")
        .join("python")
        .is_file()
        .then_some(prefix)
}

/// The command `launch_talos` execs: `talos chat` plus the caller's own
/// arguments, or the caller's own `ask …` when that is what came after
/// `--`. Split into the binary and its arguments the way `exec_with_env`
/// takes them.
fn talos_exec_argv(talos: &TalosCommand, extra_args: &[String]) -> (PathBuf, Vec<String>) {
    let argv = &talos.argv;
    let mut args: Vec<String> = argv[1..].to_vec();
    if extra_args.first().map(String::as_str) != Some("ask") {
        args.push("chat".to_string());
    }
    args.extend_from_slice(extra_args);
    (PathBuf::from(&argv[0]), args)
}

// ---------------------------------------------------------------------------
// Process execution helper
// ---------------------------------------------------------------------------

fn exec_with_env(bin: &PathBuf, args: &[String], extra_env: &[(&str, &str)]) -> anyhow::Result<()> {
    exec_with_env_in(bin, args, extra_env, None)
}

/// [`exec_with_env`] run from `cwd` when one is given — for an
/// integration that has to be started from a particular directory (see
/// [`TalosCommand`]). Everything else inherits the caller's.
fn exec_with_env_in(
    bin: &PathBuf,
    args: &[String],
    extra_env: &[(&str, &str)],
    cwd: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
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

    /// A directory nobody else uses, for the Talos resolver tests: they
    /// have to create a real `.venv/bin/python` because `talos_command_in`
    /// checks that the file exists — a prefix without one is "not
    /// installed", not a path to run.
    fn scratch_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-launch-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A `talos` shim on PATH wins over the prefix: whoever made one
    /// meant it. Without one, the venv interpreter under the prefix is
    /// the invocation — and no interpreter means not installed, never a
    /// guessed path that fails inside `exec_with_env` instead.
    #[test]
    fn talos_command_prefers_a_shim_then_the_venv_then_gives_up() {
        let prefix = scratch_dir("talos-prefix");
        let shim = PathBuf::from("/opt/somewhere/bin/talos");
        let shim_only = Some(TalosCommand {
            argv: vec![shim.to_string_lossy().into_owned()],
            cwd: None,
        });
        assert_eq!(
            talos_command_in(|_| Some(shim.clone()), Some(&prefix)),
            shim_only
        );
        // …and needs no prefix at all: a shim is found before one is asked for.
        assert_eq!(talos_command_in(|_| Some(shim.clone()), None), shim_only);
        assert_eq!(talos_command_in(|_| None, Some(&prefix)), None);
        assert_eq!(talos_command_in(|_| None, None), None);

        // The venv form runs from the prefix: the package is not installed
        // into the venv, `-m talos` finds it on the current directory.
        let python = prefix.join(".venv").join("bin").join("python");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, "").unwrap();
        assert_eq!(
            talos_command_in(|_| None, Some(&prefix)),
            Some(TalosCommand {
                argv: vec![
                    python.to_string_lossy().into_owned(),
                    "-m".to_string(),
                    "talos".to_string(),
                ],
                cwd: Some(prefix.clone()),
            })
        );
        std::fs::remove_dir_all(&prefix).unwrap();
    }

    /// The exact layout Talos's own installer leaves since 0.18.0:
    /// `<prefix>/bin/talos` symlinked onto PATH. `talos_command_in`
    /// resolves that back to `<prefix>` as `cwd` — a fresh, standard
    /// install must not fall into the "unknown shim" case, which is
    /// this test's real regression coverage: without the fix, a plain
    /// `talos` on PATH from the official installer produced `cwd: None`
    /// and `launch_talos` could never find its own env file.
    #[test]
    fn talos_command_resolves_the_installers_own_wrapper_to_its_prefix() {
        let dir = scratch_dir("talos-wrapper");
        let prefix = dir.join("prefix");
        std::fs::create_dir_all(prefix.join("bin")).unwrap();
        std::fs::create_dir_all(prefix.join(".venv").join("bin")).unwrap();
        std::fs::write(prefix.join("bin").join("talos"), "").unwrap();
        std::fs::write(prefix.join(".venv").join("bin").join("python"), "").unwrap();
        let on_path = dir.join("talos");
        #[cfg(unix)]
        std::os::unix::fs::symlink(prefix.join("bin").join("talos"), &on_path).unwrap();

        assert_eq!(
            talos_command_in(|_| Some(on_path.clone()), None),
            Some(TalosCommand {
                argv: vec![on_path.to_string_lossy().into_owned()],
                cwd: Some(prefix.canonicalize().unwrap()),
            })
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Three ways a `talos` on PATH can fail to match the installer's own
    /// shape — each has to fall back to "unknown shim", not a wrong guess.
    #[test]
    fn talos_wrapper_prefix_only_matches_the_installers_exact_layout() {
        let dir = scratch_dir("talos-wrapper-mismatch");

        // Not named `talos` at the end.
        let other_name = dir.join("bin").join("something-else");
        std::fs::create_dir_all(other_name.parent().unwrap()).unwrap();
        std::fs::write(&other_name, "").unwrap();
        assert_eq!(talos_wrapper_prefix(&other_name), None);

        // Named `talos`, but its parent isn't `bin`.
        let not_bin = dir.join("libexec").join("talos");
        std::fs::create_dir_all(not_bin.parent().unwrap()).unwrap();
        std::fs::write(&not_bin, "").unwrap();
        assert_eq!(talos_wrapper_prefix(&not_bin), None);

        // Right shape, but no venv underneath — not an install at all.
        let no_venv = dir.join("prefix2").join("bin").join("talos");
        std::fs::create_dir_all(no_venv.parent().unwrap()).unwrap();
        std::fs::write(&no_venv, "").unwrap();
        assert_eq!(talos_wrapper_prefix(&no_venv), None);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Talos's own loader reads `TALOS_SECRETS_ENV` through Python's
    /// `Path(...).expanduser()` — `~/…` has to land on the same file here.
    #[test]
    fn expand_and_absolutize_expands_only_the_current_users_home() {
        let home = scratch_dir("talos-tilde-home");
        assert_eq!(expand_and_absolutize("~", Some(&home)).unwrap(), home);
        assert_eq!(
            expand_and_absolutize("~/x/y", Some(&home)).unwrap(),
            home.join("x").join("y")
        );
        // Someone else's home: left alone rather than guessed at.
        assert_eq!(
            expand_and_absolutize("~other/y", Some(&home)).unwrap(),
            std::path::absolute("~other/y").unwrap()
        );
        // No `~` at all: ordinary absolutizing, `home` unused.
        assert_eq!(
            expand_and_absolutize("/already/absolute", None).unwrap(),
            PathBuf::from("/already/absolute")
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// Writing through a symlink would sever it from whatever it pointed
    /// at instead of updating it — refused before that happens, matching
    /// Talos's own `write_key` ("refusing a symlink config file").
    #[test]
    fn write_talos_env_refuses_a_symlink_target() {
        let dir = scratch_dir("talos-env-write-symlink");
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real.env");
        std::fs::write(&real, "TALOS_ALLOWED_PRINCIPALS=cli:1\n").unwrap();
        let link = dir.join("talos.env");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let err = write_talos_env(&link, &[("TALOS_MODEL", "m")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("symlink"), "{err}");
        assert_eq!(
            std::fs::read_to_string(&real).unwrap(),
            "TALOS_ALLOWED_PRINCIPALS=cli:1\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The env file is the one Talos's own `config set` would write:
    /// the secrets file when it exists, else `talos.env` in the prefix.
    #[test]
    fn talos_env_file_is_the_secrets_file_when_present_else_the_prefix_file() {
        let dir = scratch_dir("talos-env-file");
        let prefix = dir.join("prefix");
        std::fs::create_dir_all(&prefix).unwrap();
        let secrets = dir.join("secrets.env");
        // A fake, guaranteed-nonexistent home — never the machine's real
        // one, or this test would pass or fail depending on whether the
        // developer running it happens to have Talos configured (see
        // `talos_env_file_falls_back_to_home_only_when_named_secrets_is_absent`
        // for the case this stands in for).
        let fake_home = dir.join("no-such-home");
        let getenv = |key: &str| {
            (key == "TALOS_SECRETS_ENV").then(|| secrets.to_string_lossy().into_owned())
        };
        // Named but absent: Talos would not read it either.
        assert_eq!(
            talos_env_file(Some(&prefix), Some(&fake_home), getenv),
            Some(prefix.join("talos.env"))
        );
        std::fs::write(&secrets, "").unwrap();
        assert_eq!(
            talos_env_file(Some(&prefix), Some(&fake_home), getenv),
            Some(secrets.clone())
        );
        assert_eq!(
            talos_env_file(None, Some(&fake_home), getenv),
            Some(secrets.clone())
        );
        assert_eq!(talos_env_file(None, Some(&fake_home), |_| None), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `home` is used only when `TALOS_SECRETS_ENV` is unset — the exact
    /// case a hardcoded `dirs::home_dir()` inside `talos_env_file` itself
    /// could not be tested hermetically (it would read whatever secrets
    /// file the machine running the test happens to have).
    #[test]
    fn talos_env_file_falls_back_to_home_only_when_named_secrets_is_absent() {
        let dir = scratch_dir("talos-env-file-home");
        let home = dir.join("home");
        std::fs::create_dir_all(home.join(".secrets")).unwrap();
        let home_secrets = home.join(".secrets").join("talos-telegram.env");
        assert_eq!(talos_env_file(None, Some(&home), |_| None), None);
        std::fs::write(&home_secrets, "").unwrap();
        assert_eq!(
            talos_env_file(None, Some(&home), |_| None),
            Some(home_secrets)
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Writing keeps everything the operator put there, replaces the
    /// key's first line, drops its duplicates, appends what is missing —
    /// and lands with mode 600, as Talos's own writer does.
    #[test]
    fn write_talos_env_replaces_in_place_and_keeps_the_rest() {
        let dir = scratch_dir("talos-env-write");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("talos.env");
        std::fs::write(
            &path,
            "# operator notes\nTELEGRAM_BOT_TOKEN=keep\nTALOS_MODEL=\"old\"\n\nTALOS_MODEL=older\n# TALOS_MODEL=commented\n",
        )
        .unwrap();
        write_talos_env(
            &path,
            &[
                ("TALOS_MODEL_PROVIDER", "ollama"),
                ("TALOS_MODEL", "qwen3:8b"),
            ],
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# operator notes\nTELEGRAM_BOT_TOKEN=keep\nTALOS_MODEL=qwen3:8b\n\n# TALOS_MODEL=commented\nTALOS_MODEL_PROVIDER=ollama\n"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));

        // No file yet: created, with just ours.
        let fresh = dir.join("fresh.env");
        write_talos_env(&fresh, &[("TALOS_MODEL", "m")]).unwrap();
        assert_eq!(std::fs::read_to_string(&fresh).unwrap(), "TALOS_MODEL=m\n");

        // A value that would smuggle in a second line is refused, file untouched.
        assert!(
            write_talos_env(&fresh, &[("TALOS_MODEL", "m\nTALOS_ALLOWED_PRINCIPALS=*")]).is_err()
        );
        assert_eq!(std::fs::read_to_string(&fresh).unwrap(), "TALOS_MODEL=m\n");

        // An existing file that cannot be read as text is an error, not an
        // empty file: rewriting it would drop everything else it holds.
        let garbled = dir.join("garbled.env");
        std::fs::write(&garbled, [0xff, 0xfe, b'\n']).unwrap();
        assert!(write_talos_env(&garbled, &[("TALOS_MODEL", "m")]).is_err());
        assert_eq!(std::fs::read(&garbled).unwrap(), [0xff, 0xfe, b'\n']);
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));

        // A NUL byte is rejected exactly like a newline — Talos's own
        // `_one_line` validator refuses it too.
        let clean = dir.join("clean.env");
        assert!(write_talos_env(&clean, &[("TALOS_MODEL", "m\0")]).is_err());
        assert!(!clean.exists());

        // Written stripped, matching Talos's own `config set` (which
        // returns `value.strip()`) and its file-loader (which strips
        // every value it reads back) — an untrimmed write here would
        // silently diverge from what Talos itself would have stored.
        let padded = dir.join("padded.env");
        write_talos_env(&padded, &[("TALOS_MODEL", "  qwen3.5  ")]).unwrap();
        assert_eq!(
            std::fs::read_to_string(&padded).unwrap(),
            "TALOS_MODEL=qwen3.5\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `chat` is the session; a leading `ask` after `--` is Talos's own
    /// one-turn command and goes through as typed — `talos chat ask …`
    /// would hand "ask" to the session as a message.
    #[test]
    fn talos_exec_argv_runs_chat_unless_the_caller_asked_for_ask() {
        let venv = TalosCommand {
            argv: vec![
                "/x/.venv/bin/python".to_string(),
                "-m".to_string(),
                "talos".to_string(),
            ],
            cwd: Some(PathBuf::from("/x")),
        };
        let none: Vec<String> = vec![];
        assert_eq!(
            talos_exec_argv(&venv, &none),
            (
                PathBuf::from("/x/.venv/bin/python"),
                vec!["-m".to_string(), "talos".to_string(), "chat".to_string()]
            )
        );
        let ask = vec!["ask".to_string(), "how many?".to_string()];
        assert_eq!(
            talos_exec_argv(&venv, &ask),
            (
                PathBuf::from("/x/.venv/bin/python"),
                vec![
                    "-m".to_string(),
                    "talos".to_string(),
                    "ask".to_string(),
                    "how many?".to_string()
                ]
            )
        );
        let shim = TalosCommand {
            argv: vec!["/usr/local/bin/talos".to_string()],
            cwd: None,
        };
        let extra = vec!["--verbose".to_string()];
        assert_eq!(
            talos_exec_argv(&shim, &extra),
            (
                PathBuf::from("/usr/local/bin/talos"),
                vec!["chat".to_string(), "--verbose".to_string()]
            )
        );
    }

    /// An exported model setting that disagrees with the launch is
    /// refused up front (it would outrank the env file the launcher
    /// writes); one that agrees, or is empty, is not an override.
    #[test]
    fn talos_env_overrides_are_refused_only_when_they_disagree() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert!(talos_check_env_overrides("m:latest", env(&[])).is_ok());
        assert!(talos_check_env_overrides("m:latest", env(&[("TALOS_MODEL", "")])).is_ok());
        // Whitespace-only is NOT treated as unset: Talos's own `or` sees a
        // truthy string and would use it verbatim, so this has to refuse
        // exactly like any other disagreeing value.
        assert!(talos_check_env_overrides("m:latest", env(&[("TALOS_MODEL", "  ")])).is_err());
        assert!(talos_check_env_overrides(
            "m:latest",
            env(&[
                ("TALOS_MODEL_PROVIDER", "ollama"),
                ("TALOS_MODEL", "m:latest")
            ])
        )
        .is_ok());

        let err = talos_check_env_overrides("m:latest", env(&[("TALOS_MODEL", "other")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("TALOS_MODEL is set to \"other\""), "{err}");
        assert!(err.contains("llmman launch talos"), "{err}");
        let err =
            talos_check_env_overrides("m:latest", env(&[("TALOS_MODEL_PROVIDER", "openai-api")]))
                .unwrap_err()
                .to_string();
        assert!(err.contains("TALOS_MODEL_PROVIDER"), "{err}");
    }

    /// `talos` cannot carry a key (it is configured through a file, like
    /// hermes) — the listing and the refusal above have to agree on that.
    #[test]
    fn talos_is_a_file_configured_integration() {
        assert!(PROVIDER_NEEDS_DAEMON_KEY.contains(&"talos"));
        assert!(INTEGRATIONS.iter().any(|i| i.name == "talos"));
    }
}
