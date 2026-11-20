use std::io::IsTerminal;

use clap::Args;

use crate::{ffi, hf, logins, oauth};

#[derive(Args, Debug)]
pub struct LoginArgs {
    /// Server to log in to: an OCI registry host (e.g.
    /// registry.example.com) for username/password credentials, or a
    /// HuggingFace host (hf.co, huggingface.co) for a HuggingFace access
    /// token. Defaults to docker.io when omitted.
    #[arg(value_name = "SERVER")]
    pub server: Option<String>,

    /// Registry username (ignored for HuggingFace logins, which are token-only)
    #[arg(short, long)]
    pub username: Option<String>,

    /// Registry password, read from stdin if omitted (ignored for HuggingFace logins)
    #[arg(short, long)]
    pub password: Option<String>,

    /// HuggingFace access token, prompted for if omitted (ignored for registry
    /// logins). With --provider, the API key or token to store; prompted for
    /// (not echoed) if omitted, which keeps it out of the process list.
    #[arg(short, long)]
    pub token: Option<String>,

    /// Log in to a hosted provider (an id from `llmman providers`) instead of
    /// a registry: a ChatGPT, GitHub Copilot or SuperGrok subscription,
    /// OpenRouter, Poe, DigitalOcean, Chutes, GitLab, Snowflake or MiniMax in
    /// the browser, a Claude token, a cloud's credentials (AWS, Google
    /// Vertex, Azure, Databricks, SAP AI Core, watsonx), or a key for any
    /// provider.
    #[arg(long, value_name = "ID", conflicts_with_all = ["server", "username", "password"])]
    pub provider: Option<String>,

    /// How to log in to --provider: the provider's own (see `llmman login
    /// --provider <id> --method ?`), or `key`, `command`, `token-file`.
    #[arg(long, value_name = "METHOD", requires = "provider")]
    pub method: Option<String>,

    /// With `--method command`: the command whose output is the key.
    #[arg(long, value_name = "COMMAND", requires = "provider")]
    pub command: Option<String>,

    /// With `--method token-file`: a file (owner-only) holding the key.
    #[arg(long, value_name = "PATH", requires = "provider")]
    pub file: Option<String>,

    /// With `--method import-claude` or `import-codex`: move the other CLI's
    /// login into llmman instead of reading its file on each request. Its
    /// refresh token rotates when used, so that CLI may be logged out.
    #[arg(long, requires = "provider")]
    pub copy: bool,

    /// With --provider: a setting the method needs besides a secret, as
    /// `KEY=VALUE`, repeatable — `--param profile=work`, `--param
    /// tenant_id=...`, `--param instance_url=https://gitlab.example.com`.
    /// `--method ?` lists the methods; `docs/providers.md` lists the keys.
    #[arg(long = "param", value_name = "KEY=VALUE", requires = "provider")]
    pub params: Vec<String>,
}

/// Mirrors `docker login`'s own decision rule (`cli/command/registry/
/// login.go`'s `loginUser`): the web-based device-authorization flow is
/// only ever attempted for the default registry (docker.io, i.e. no
/// `SERVER` given or `docker.io` given explicitly), and only when neither
/// `--username` nor `--password` was supplied — those two are always an
/// explicit request for the plain credential flow instead. Any other
/// registry always goes straight to username/password, same as before.
pub fn run(args: &LoginArgs) -> anyhow::Result<()> {
    if let Some(provider) = &args.provider {
        return run_provider(provider, args);
    }
    let server = args
        .server
        .clone()
        .unwrap_or_else(|| "docker.io".to_string());

    if hf::is_hf_host(&server) {
        let token = match &args.token {
            Some(t) => t.clone(),
            None => {
                eprint!("Token (from https://huggingface.co/settings/tokens): ");
                read_line_stdin()?
            }
        };
        let username = hf::login(token.trim())?;
        println!("Login succeeded for {server} (user: {username})");
        return Ok(());
    }

    let is_default_registry = server == "docker.io";
    if is_default_registry && args.username.is_none() && args.password.is_none() {
        match oauth::login_device() {
            Ok(Some(result)) => {
                ffi::login(&server, &result.username, &result.password)?;
                println!();
                println!("Login Succeeded");
                return Ok(());
            }
            Ok(None) => {
                // Mirrors docker/cli's exact fallback message (the device
                // flow couldn't even be started — e.g. no route to
                // login.docker.com — so drop to the plain credential flow
                // below instead of failing outright).
                println!("Failed to start web-based login - falling back to command line login...");
                println!();
            }
            // The flow *started* but failed afterwards (timeout, denial,
            // network error while polling) — docker/cli does not fall back
            // in that case either, it just reports the error.
            Err(e) => return Err(e),
        }
    }

    let username = match &args.username {
        Some(u) => u.clone(),
        None => {
            if !std::io::stdin().is_terminal() {
                anyhow::bail!("cannot perform an interactive login from a non-TTY device");
            }
            eprint!("Username: ");
            read_line_stdin()?
        }
    };
    let password = match &args.password {
        Some(p) => p.clone(),
        None => {
            eprint!("Password: ");
            read_line_stdin()?
        }
    };
    ffi::login(&server, &username, &password)?;
    println!("Login succeeded for {server}");
    Ok(())
}

/// Read a line from stdin, trimming the trailing newline.
/// Does not suppress echo — callers wanting a TTY UX should pipe through a helper.
fn read_line_stdin() -> anyhow::Result<String> {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim_end_matches(['\n', '\r']).to_string())
}

/// `llmman login --provider <id>`: see [`crate::logins`].
fn run_provider(provider: &str, args: &LoginArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !provider.is_empty() && !provider.contains('/'),
        "{provider:?} is not a provider id"
    );
    if args.method.as_deref() == Some("?") {
        println!("Login methods for {provider} (the first is the default):");
        for m in logins::methods(provider) {
            println!("  {:<14} {}", m.name, m.summary);
        }
        println!("Also, for a provider whose credential is minted at a token endpoint:");
        for m in logins::ADVANCED {
            println!("  {:<14} {}", m.name, m.summary);
        }
        return Ok(());
    }
    check_known(provider)?;
    let mut params = Vec::new();
    for pair in &args.params {
        let (key, value) = pair
            .split_once('=')
            .filter(|(k, _)| !k.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("--param takes KEY=VALUE, not {pair:?}"))?;
        params.push((key.trim().to_string(), value.trim().to_string()));
    }
    logins::login(
        provider,
        &logins::Options {
            method: args.method.clone(),
            token: args.token.clone(),
            command: args.command.clone(),
            file: args.file.clone(),
            copy: args.copy,
            params,
        },
    )
}

/// Refuses an id the catalog does not have, with its near-matches, so a
/// typo does not store a login nothing will ever read. An unreachable
/// catalog is not a reason to refuse: the id is taken on trust.
fn check_known(provider: &str) -> anyhow::Result<()> {
    match crate::providers::catalog() {
        Ok(catalog) if catalog.get(provider).is_none() && logins::profile(provider).is_none() => {
            Err(crate::providers::unknown_provider_error(provider, &catalog))
        }
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!(
                "[llmman] warning: could not check {provider:?} against the provider list ({e:#})"
            );
            Ok(())
        }
    }
}
