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

    let reference = crate::shortnames::resolve_ollama_api(&args.model)?;

    // The unload endpoint itself always reports success even when
    // nothing by that name was running (canonical_ref falls back
    // gracefully — see cmd::serve's own doc comment on it), so "was this
    // model actually loaded" is checked here, against a `ps` snapshot,
    // purely to give the same "couldn't find model to stop" error
    // `ollama stop` gives instead of a silent, meaningless success.
    //
    // Both sides are compared through `default_tag` rather than as raw
    // strings. A `ps` name is a `mgr.running` key, which `canonical_ref`
    // took from the store's own `org.opencontainers.image.ref.name`, and
    // `OciStore::tag` records whatever reference it was handed: `llmman
    // cp x y` stores a tagless `y`, while a pulled model stores
    // `y:latest`. Either spelling can therefore be the key, and the
    // caller may type either, so an exact comparison fails one direction
    // or the other depending on which. Normalising both is what
    // `ref_matches_precise` already does inside the store.
    let running: PsResponse = crate::daemon::get_json("/api/ps")?;
    let tagged = crate::storage::default_tag(&reference);
    let Some(found) = running
        .models
        .iter()
        .find(|m| crate::storage::default_tag(&m.name) == tagged)
    else {
        anyhow::bail!("couldn't find model \"{}\" to stop", args.model);
    };

    // The `ps` name, not our own spelling of it: it is the key the daemon
    // actually holds, so there is nothing left for the unload to resolve.
    crate::daemon::unload(&found.name)?;
    println!("Stopped {}", found.name);
    Ok(())
}
