use clap::Args;

use crate::ffi;
use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Image reference to inspect
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// Inspect a remote registry image instead of the local store
    #[arg(long)]
    pub remote: bool,
}

pub fn run(args: &InspectArgs) -> anyhow::Result<()> {
    // resolve_ollama_api, not resolve: same reasoning as rm.rs/tag.rs for
    // the local-store case, and consistent with pull's own remote lookup
    // for --remote (a bare name means the same registry either way).
    let reference = crate::shortnames::resolve_ollama_api(&args.reference);
    if args.remote {
        let json = ffi::inspect_remote(&reference)?;
        println!("{}", json);
    } else {
        inspect_local(&reference)?;
    }
    Ok(())
}

fn inspect_local(reference: &str) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;
    let desc = store.find(reference)?;

    let manifest = store.read_manifest(&desc.digest)?;
    let out = serde_json::to_string_pretty(&manifest)?;
    println!("{}", out);
    Ok(())
}
