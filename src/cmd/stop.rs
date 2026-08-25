use clap::Args;
use serde::Deserialize;

#[derive(Args, Debug)]
pub struct StopArgs {
    /// Model to unload
    #[arg(value_name = "MODEL")]
    pub model: String,
}

/// Wire shape of GET /api/ps's response — see `ps.rs`'s own copy of this
/// same shape and its doc comment on why each CLI command that needs it
/// keeps its own minimal copy instead of sharing one type with the
/// daemon.
#[derive(Debug, Deserialize)]
struct PsResponse {
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
struct PsModel {
    name: String,
}

/// `llmman stop MODEL` — unload a running model immediately, mirroring
/// `ollama stop` (`cmd/cmd.go`'s `StopHandler`/`loadOrUnloadModel`), which
/// posts `{"model": ..., "keep_alive": 0}` (empty prompt) to
/// `/api/generate` — the exact sentinel `cmd::serve::handle_ollama_generate`
/// already reads as an immediate-unload request. See `daemon::unload`.
pub fn run(args: &StopArgs) -> anyhow::Result<()> {
    if !crate::daemon::server_alive() {
        anyhow::bail!("llmman serve is not running (nothing is loaded) — nothing to stop");
    }

    let reference = crate::shortnames::resolve_ollama_api(&args.model);

    // The unload endpoint itself always reports success even when
    // nothing by that name was running (canonical_ref falls back
    // gracefully — see cmd::serve's own doc comment on it), so "was this
    // model actually loaded" is checked here, against a `ps` snapshot,
    // purely to give the same "couldn't find model to stop" error
    // `ollama stop` gives instead of a silent, meaningless success.
    let running: PsResponse = crate::daemon::get_json("/api/ps")?;
    if !running.models.iter().any(|m| m.name == reference) {
        anyhow::bail!("couldn't find model \"{}\" to stop", args.model);
    }

    crate::daemon::unload(&reference)?;
    println!("Stopped {}", reference);
    Ok(())
}
