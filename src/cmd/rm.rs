use clap::Args;

use crate::storage::{gc, OciStore};

#[derive(Args, Debug)]
pub struct RmArgs {
    /// Reference(s) to remove (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE", required = true, num_args = 1..)]
    pub references: Vec<String>,
}

pub fn run(args: &RmArgs) -> anyhow::Result<()> {
    let mut any_err = false;

    // Resolve (and validate) every reference before opening the store, so an
    // invalid ref never creates the store tree.
    // resolve_ollama_api, not resolve: a bare name pulled via the Ollama API
    // (POST /api/pull, /api/chat, ...) is stored under docker.io/ai/<name>,
    // not hf.co/<name>, so it must resolve the same way here or `llmman rm
    // <bare-name>` looks for the wrong entry.
    let mut refs = Vec::new();
    for raw in &args.references {
        match crate::shortnames::resolve_ollama_api(raw) {
            Ok(r) => refs.push(r),
            Err(e) => {
                eprintln!("Error removing {raw}: {e}");
                any_err = true;
            }
        }
    }

    // Nothing valid to remove: don't open (and thereby create) the store.
    if refs.is_empty() {
        anyhow::bail!("one or more removals failed");
    }

    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let mut any_removed = false;
    for reference in &refs {
        match store.remove(reference) {
            // The stored reference, not the one given: for a digest that
            // is the tag it was held under.
            Ok(removed) => {
                println!("Removed {}", removed);
                any_removed = true;
            }
            Err(e) => {
                eprintln!("Error removing {}: {}", reference, e);
                any_err = true;
            }
        }
    }

    // GC once, after every requested reference is untagged: recompute the
    // still-referenced digest set from the surviving manifests, then sweep
    // blobs/cache not in it. Uses the same grace window as serve's startup
    // sweep, NOT zero: `llmman pull` runs in the long-lived `serve` daemon,
    // which writes each layer blob to its final path as it downloads and
    // only writes the manifest + tag at the very end. A concurrent
    // `rm <unrelated>` would otherwise see that not-yet-tagged blob as an
    // orphan and delete it mid-pull. The writer is a different process, so
    // `rm` being synchronous doesn't protect it — only the grace window
    // does (matches Ollama's layerPruneGracePeriod).
    if any_removed && !gc::noprune_from_env() {
        let live = gc::referenced_digests(&store)?;
        let cache_path = crate::default_cache()?;
        let grace = gc::GC_GRACE_PERIOD;
        let blob_stats = gc::prune_blobs(&store_root, &live, grace)?;
        let cache_stats = gc::prune_cache(&cache_path, &live, grace)?;
        if blob_stats.count > 0 || cache_stats.count > 0 {
            println!(
                "Freed {} ({} blobs, {} cache entries)",
                crate::fmt::human_size(blob_stats.bytes + cache_stats.bytes),
                blob_stats.count,
                cache_stats.count
            );
        }
    }

    if any_err {
        anyhow::bail!("one or more removals failed");
    }
    Ok(())
}
