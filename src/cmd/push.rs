use clap::Args;

#[derive(Args, Debug)]
pub struct PushArgs {
    /// Registry reference (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE")]
    pub reference: String,
}

/// `llmman push` is a thin client of the local daemon's Ollama-protocol
/// /api/push (starting one, left running afterwards, if none is running
/// yet — see daemon::ensure_server) — the same wire protocol `sbx` and any
/// other Ollama-API client use, so bare-name resolution (shortnames::
/// resolve_ollama_api) and the model store are always the daemon's.
///
/// No store override of its own: set `LLMMAN_MODELS` before starting
/// `llmman serve` to change the daemon's store for every client.
pub fn run(args: &PushArgs) -> anyhow::Result<()> {
    crate::daemon::ensure_server("")?;
    crate::daemon::stream_progress("/api/push", &args.reference)?;
    println!("Pushed {}", args.reference);
    Ok(())
}
