//! `llmman providers` — the providers `--provider` accepts.
//!
//! Answered by `llmman serve` (`GET /llmman/providers`), not by fetching
//! models.dev here: the daemon forwards the request, so its catalog
//! decides whether `--provider x` works and its environment holds the key
//! that gets spent (see `resolve_remote_target` in cmd::serve).

use clap::Args;
use std::fmt::Write;

use crate::daemon::{self, ProviderSummary};

/// No positional arguments: a positional filter here read `llmman providers ls`
/// as "providers containing `ls`" and printed modelscope and poolside.
#[derive(Args, Debug)]
pub struct ProvidersArgs {
    /// Do not pipe the output into a pager
    #[arg(long)]
    pub no_pager: bool,
}

pub fn run(args: &ProvidersArgs) -> anyhow::Result<()> {
    // Same contract as `run`/`pull`/`launch`: start the daemon rather
    // than tell the user to. It owns the catalog, and whatever runs next
    // needs it anyway.
    daemon::ensure_server("")?;

    let shown = daemon::providers()?;

    if shown.is_empty() {
        // A header with no rows would read as "there are none".
        anyhow::bail!("no providers available");
    }

    let mut out = String::new();
    let id_w = shown.iter().map(|p| p.id.len()).max().unwrap_or(8).max(8);
    let name_w = shown.iter().map(|p| p.name.len()).max().unwrap_or(4).max(4);
    let key_w = shown
        .iter()
        .map(|p| key_env(p).len())
        .max()
        .unwrap_or(7)
        .max(7);
    let status_w = shown
        .iter()
        .map(|p| key_status(p).len())
        .max()
        .unwrap_or(14)
        .max(14);

    writeln!(
        out,
        "{:<id_w$}    {:<name_w$}    {:<key_w$}    {:<status_w$}    MODELS",
        "PROVIDER",
        "NAME",
        "API KEY",
        "KEY",
        id_w = id_w,
        name_w = name_w,
        key_w = key_w,
        status_w = status_w,
    )?;
    for p in &shown {
        writeln!(
            out,
            "{:<id_w$}    {:<name_w$}    {:<key_w$}    {:<status_w$}    {}",
            p.id,
            p.name,
            key_env(p),
            key_status(p),
            model_count(p),
            id_w = id_w,
            name_w = name_w,
            key_w = key_w,
            status_w = status_w,
        )?;
    }

    // Nothing after the last row: a trailing count and usage block is
    // something to skip past every time, and something a pipe into
    // `grep`/`awk` has to filter out. `--help` is where usage belongs.
    crate::pager::emit("providers", &out, !args.no_pager)
}

/// The variable column: `-` for a configured provider that names none.
fn key_env(provider: &ProviderSummary) -> &str {
    provider.key_env.as_deref().unwrap_or("-")
}

/// The models column: `-` for a configured provider, whose endpoint is
/// only asked by `llmman list --provider <id>`; `0` would read as
/// "serves nothing".
fn model_count(provider: &ProviderSummary) -> String {
    if provider.key_optional && provider.models == 0 {
        "-".to_string()
    } else {
        provider.models.to_string()
    }
}

/// Where a *usable* key is — the one thing to act on before `--provider`
/// works.
///
/// "client" is a key only this process has, which travels per request —
/// from its environment or its own `llmman.conf`, which is why the word
/// is not "shell". "withheld" is one the daemon has but will not spend,
/// being bound where others could reach it (see `resolve_remote_target`
/// in cmd::serve) — a state of its own, since what needs fixing there is
/// the bind, not the key. "none needed" is a keyless provider: nothing
/// to act on.
fn key_status(provider: &ProviderSummary) -> String {
    let base = match (provider.key_usable, provider.key_here(), provider.key_set) {
        (true, _, _) => "set",
        (false, true, _) => "set (client)",
        (false, false, true) => "set (withheld)",
        (false, false, false) if provider.key_optional => "none needed",
        (false, false, false) => "unset",
    };
    // A key in the environment or `llmman.conf` is just "set", as it has
    // always been; a login or key command names what it is.
    match provider.credential.as_ref() {
        Some(c) if c.state == "expired" => format!("expired ({})", c.kind),
        Some(c) if !matches!(c.kind.as_str(), "env" | "config") => match base {
            "set" => format!("set ({})", c.kind),
            other => format!("{} [{}]", other, c.kind),
        },
        _ => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(id: &str, name: &str) -> ProviderSummary {
        ProviderSummary {
            id: id.to_string(),
            name: name.to_string(),
            key_env: Some("LLMMAN_TEST_PROVIDER_KEY_UNSET".to_string()),
            key_set: false,
            key_usable: false,
            key_optional: false,
            credential: None,
            models: 0,
        }
    }

    /// Each way a key can be present is a different thing to do about
    /// it, so none may collapse into another.
    #[test]
    fn key_status_reports_the_key_that_would_actually_be_used() {
        let mut p = summary("openai", "OpenAI");
        assert_eq!(key_status(&p), "unset");
        // Held by the daemon but withheld: the bind needs fixing, not
        // the variable.
        p.key_set = true;
        assert_eq!(key_status(&p), "set (withheld)");
        p.key_usable = true;
        assert_eq!(key_status(&p), "set");

        // A provider that takes no key is not "unset": there is nothing
        // to set. One that has a key anyway reports it as any other.
        let mut p = summary("inferencebox", "Inference box");
        p.key_env = None;
        p.key_optional = true;
        assert_eq!(key_status(&p), "none needed");
        assert_eq!(key_env(&p), "-");
        assert_eq!(model_count(&p), "-");
        assert_eq!(model_count(&summary("openai", "OpenAI")), "0");
        p.key_set = true;
        p.key_usable = true;
        assert_eq!(key_status(&p), "set");
    }
}
