use std::path::PathBuf;

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

    /// Local store directory (overrides default)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,
}

pub fn run(args: &InspectArgs) -> anyhow::Result<()> {
    // resolve_ollama_api, not resolve: same reasoning as rm.rs/tag.rs for
    // the local-store case, and consistent with pull's own remote lookup
    // for --remote (a bare name means the same registry either way).
    let reference = crate::shortnames::resolve_ollama_api(&args.reference);
    if args.remote {
        // ffi::inspect_remote now returns the manifest's exact bytes (not
        // pretty-printed) — see go-shim/backend_docker.go's llmman_inspect
        // comment for why (cmd::sign needs to hash these bytes to match
        // the registry's own digest) — so pretty-printing for display
        // happens here instead. Falls back to the raw string if it
        // doesn't parse as JSON, same as the old Go-side fallback.
        let raw = ffi::inspect_remote(&reference)?;
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(value) => println!("{}", serde_json::to_string_pretty(&value)?),
            Err(_) => println!("{}", raw),
        }
    } else {
        inspect_local(args, &reference)?;
    }
    Ok(())
}

fn inspect_local(args: &InspectArgs, reference: &str) -> anyhow::Result<()> {
    let store_root = crate::default_store(args.store.as_deref())?;
    let store = OciStore::open(&store_root)?;
    let desc = store.find(reference)?;

    let manifest = store.read_manifest(&desc.digest)?;
    let out = serde_json::to_string_pretty(&manifest)?;
    println!("{}", out);
    Ok(())
}
