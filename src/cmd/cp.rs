use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct CpArgs {
    /// Existing local reference to copy
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// New reference to create, pointing at the same image
    #[arg(value_name = "DESTINATION")]
    pub destination: String,
}

/// `llmman cp SOURCE DESTINATION` — copy a model under a new name,
/// mirroring `ollama cp` (`cmd/cmd.go`'s `CopyHandler`, `server/
/// images.go`'s `CopyModel`): both are really just "point a second
/// reference at the same content", exactly what `llmman tag` already does
/// (see `storage::OciStore::tag`) — `cp` exists as its own command purely
/// to match ollama's naming for anyone coming from there, not because the
/// underlying operation differs at all.
pub fn run(args: &CpArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    // resolve_ollama_api, not resolve: SOURCE must match how the model is
    // actually stored — same reasoning as tag.rs/rm.rs.
    let source = crate::shortnames::resolve_ollama_api(&args.source);
    let desc = store.find(&source)?;
    store.tag(desc, &args.destination)?;
    println!("copied '{}' to '{}'", args.source, args.destination);
    Ok(())
}
