use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct TagArgs {
    /// Source reference (must exist locally)
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// New reference to create
    #[arg(value_name = "TARGET")]
    pub target: String,
}

pub fn run(args: &TagArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    // resolve_ollama_api, not resolve: SOURCE must match how the model is
    // actually stored (see rm.rs's comment — same reasoning). TARGET is a
    // new name the user is choosing, so it's left as typed, unresolved.
    let source = crate::shortnames::resolve_ollama_api(&args.source);
    let desc = store.find(&source)?;
    store.tag(desc, &args.target)?;
    println!("Tagged {} as {}", source, args.target);
    Ok(())
}
