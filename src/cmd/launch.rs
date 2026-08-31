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
/// the key: writing a real one into `~/.hermes/config.yaml` would persist
/// a credential, which this feature promises not to do. They rely on
/// `llmman serve` having the variable itself — which it only uses for a
/// daemon nobody else can reach (see `reachable_only_locally`).
const PROVIDER_NEEDS_DAEMON_KEY: &[&str] = &["droid", "hermes"];

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
        name: "droid",
        description: "Factory Droid CLI",
        binary: "droid",
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
];

fn print_integrations() {
    println!("Available integrations:\n");
    for i in INTEGRATIONS {
        if find_on_path(i.binary).is_some() {
            println!("  {:<12} {}", i.name, i.description);
        } else {
            println!("  {:<12} {} (not installed)", i.name, i.description);
        }
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
        "droid" => launch_droid(model, extra_args),
        "cline" => launch_simple("cline", model, extra_args),
        "aider" => launch_aider(model, api_key, extra_args),
        "copilot" | "copilot-cli" => launch_copilot(model, extra_args),
        "kimi" => launch_simple("kimi", model, extra_args),
        "gemini" => launch_gemini(model, api_key, extra_args),
        "hermes" => launch_hermes(model, extra_args),
        "openclaw" => launch_openclaw(model, extra_args),
        "qwen" => launch_qwen(model, api_key, extra_args),
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

/// droid: add one llmman-owned custom model to Factory's shared settings,
/// then select that exact entry on launch. The placeholder key is deliberate:
/// local models need no secret, while provider-routed launches require the
/// loopback daemon to hold the real key (see `PROVIDER_NEEDS_DAEMON_KEY`).
fn launch_droid(model: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("droid").ok_or_else(|| anyhow::anyhow!("droid is not installed"))?;
    let effective_model = if model.is_empty() { "default" } else { model };
    let custom_model_id = write_droid_config(effective_model)?;
    let args = droid_args(&custom_model_id, extra_args);
    exec_with_env(&bin, &args, &[])
}

/// Writes the modern Factory settings format documented at
/// <https://docs.factory.ai/model-independence/byok>. The parsed JSON object is
/// retained so unrelated user settings and custom models survive unchanged.
fn write_droid_config(model: &str) -> anyhow::Result<String> {
    let home = dirs::home_dir().context("no home directory")?;
    let config_dir = home.join(".factory");
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("settings.json");

    let existing = match std::fs::read_to_string(&config_path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => "{}".to_string(),
        Err(err) => return Err(err).with_context(|| format!("read {}", config_path.display())),
    };
    let base_url = format!("{}/v1", daemon::server());
    let (contents, custom_model_id) = update_droid_settings(&existing, model, &base_url)
        .with_context(|| format!("parse {}", config_path.display()))?;

    if existing != contents {
        std::fs::write(&config_path, contents)
            .with_context(|| format!("write {}", config_path.display()))?;
    }
    Ok(custom_model_id)
}

/// Adds or replaces llmman's one owned entry and returns Droid's positional
/// custom-model ID. A dedicated boolean marker distinguishes this entry from
/// similarly named user models without persisting a real provider credential.
fn update_droid_settings(
    existing: &str,
    model: &str,
    base_url: &str,
) -> anyhow::Result<(String, String)> {
    let mut settings: serde_json::Value = serde_json::from_str(existing)?;
    let root = settings
        .as_object_mut()
        .context("Factory settings must be a JSON object")?;
    let models = root
        .entry("customModels")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
        .as_array_mut()
        .context("Factory customModels must be a JSON array")?;

    let owned = models.iter().position(|entry| {
        let Some(entry) = entry.as_object() else {
            return false;
        };
        entry
            .get("llmmanManaged")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
            && entry
                .get("id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| id.starts_with("custom:llmman-"))
    });
    let index = owned.unwrap_or(models.len());
    let custom_model_id = format!("custom:llmman-{index}");

    let mut entry = owned
        .and_then(|i| models[i].as_object().cloned())
        .unwrap_or_default();
    entry.insert("model".into(), model.into());
    entry.insert("displayName".into(), "llmman".into());
    entry.insert("baseUrl".into(), base_url.into());
    entry.insert("apiKey".into(), "llmman".into());
    entry.insert("provider".into(), "generic-chat-completion-api".into());
    entry.insert("maxOutputTokens".into(), 64_000.into());
    entry.insert("id".into(), custom_model_id.clone().into());
    entry.insert("index".into(), index.into());
    entry.insert("llmmanManaged".into(), true.into());

    if let Some(i) = owned {
        models[i] = serde_json::Value::Object(entry);
    } else {
        models.push(serde_json::Value::Object(entry));
    }

    let mut rendered = serde_json::to_string_pretty(&settings)?;
    rendered.push('\n');
    Ok((rendered, custom_model_id))
}

fn droid_args(custom_model_id: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = vec!["--model".to_string(), custom_model_id.to_string()];
    args.extend_from_slice(extra_args);
    args
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

    #[test]
    fn droid_settings_create_a_selectable_llmman_model() {
        let (settings, id) = update_droid_settings("{}", "qwen3.5:0.8b", "http://localhost/v1")
            .expect("valid settings");
        let parsed: serde_json::Value = serde_json::from_str(&settings).unwrap();
        let model = &parsed["customModels"][0];

        assert_eq!(id, "custom:llmman-0");
        assert_eq!(model["model"], "qwen3.5:0.8b");
        assert_eq!(model["displayName"], "llmman");
        assert_eq!(model["baseUrl"], "http://localhost/v1");
        assert_eq!(model["apiKey"], "llmman");
        assert_eq!(model["provider"], "generic-chat-completion-api");
        assert_eq!(model["maxOutputTokens"], 64_000);
        assert_eq!(model["id"], id);
        assert_eq!(model["index"], 0);
        assert_eq!(model["llmmanManaged"], true);
    }

    #[test]
    fn droid_settings_preserve_unrelated_data_and_append_ours() {
        let existing = r#"{
            "theme": "dark",
            "customModels": [{
                "model": "other",
                "displayName": "Other",
                "customField": {"keep": true}
            }]
        }"#;
        let (settings, id) =
            update_droid_settings(existing, "model/with:specials", "http://localhost/v1").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&settings).unwrap();

        assert_eq!(parsed["theme"], "dark");
        assert_eq!(parsed["customModels"][0]["customField"]["keep"], true);
        assert_eq!(parsed["customModels"][1]["model"], "model/with:specials");
        assert_eq!(parsed["customModels"][1]["index"], 1);
        assert_eq!(id, "custom:llmman-1");
    }

    #[test]
    fn droid_settings_replace_only_our_entry_and_are_idempotent() {
        let existing = r#"{
            "customModels": [
                {"model": "other", "displayName": "Other"},
                {
                    "model": "old",
                    "displayName": "llmman",
                    "apiKey": "llmman",
                    "id": "custom:llmman-1",
                    "index": 1,
                    "llmmanManaged": true,
                    "futureField": "preserved"
                }
            ]
        }"#;
        let (once, id) = update_droid_settings(existing, "new", "http://new/v1").unwrap();
        let (twice, second_id) = update_droid_settings(&once, "new", "http://new/v1").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&twice).unwrap();

        assert_eq!(id, "custom:llmman-1");
        assert_eq!(second_id, id);
        assert_eq!(once, twice);
        assert_eq!(parsed["customModels"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["customModels"][0]["model"], "other");
        assert_eq!(parsed["customModels"][1]["model"], "new");
        assert_eq!(parsed["customModels"][1]["baseUrl"], "http://new/v1");
        assert_eq!(parsed["customModels"][1]["futureField"], "preserved");
    }

    #[test]
    fn droid_settings_do_not_overwrite_a_similarly_named_user_model() {
        let existing = r#"{
            "customModels": [{
                "model": "user-model",
                "displayName": "llmman",
                "apiKey": "llmman",
                "id": "custom:llmman-0",
                "index": 0
            }]
        }"#;
        let (settings, id) =
            update_droid_settings(existing, "managed-model", "http://localhost/v1").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&settings).unwrap();

        assert_eq!(id, "custom:llmman-1");
        assert_eq!(parsed["customModels"][0]["model"], "user-model");
        assert_eq!(
            parsed["customModels"][0]["llmmanManaged"],
            serde_json::Value::Null
        );
        assert_eq!(parsed["customModels"][1]["model"], "managed-model");
        assert_eq!(parsed["customModels"][1]["llmmanManaged"], true);
    }

    #[test]
    fn droid_settings_reject_corruption_instead_of_overwriting_it() {
        assert!(update_droid_settings("{broken", "model", "http://localhost/v1").is_err());
        assert!(update_droid_settings("[]", "model", "http://localhost/v1").is_err());
        assert!(update_droid_settings(
            r#"{"customModels":"not-an-array"}"#,
            "model",
            "http://localhost/v1"
        )
        .is_err());
    }

    #[test]
    fn droid_launch_args_select_our_model_before_forwarded_args() {
        let extra = vec![
            "--auto".to_string(),
            "high".to_string(),
            "fix it".to_string(),
        ];
        assert_eq!(
            droid_args("custom:llmman-3", &extra),
            ["--model", "custom:llmman-3", "--auto", "high", "fix it"]
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
}
