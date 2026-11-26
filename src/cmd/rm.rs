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

    // What every tag held, taken before the untag so the removals' own
    // orphans can be told apart from any other unreferenced blob.
    let before = (!gc::noprune_from_env()).then(|| gc::held_by_manifests(&store));

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

    // GC once, after every requested reference is untagged, and free at any
    // age exactly what those removals orphaned: the digests that were held
    // before and no surviving manifest holds now. Not "every unreferenced
    // blob", and not behind the grace window either: `llmman pull` runs in
    // the long-lived `serve` daemon, which writes each layer blob to its
    // final path as it downloads and only writes the manifest + tag at the
    // very end, so a blob no removed tag held may be a pull in flight (a
    // different process, which `rm` being synchronous doesn't protect).
    // Those are left to the grace-gated sweep (serve startup, retag). A
    // blob a concurrent writer reused after the removed tag was written is
    // left too; a writer reusing one in the moments between the scan after
    // the untag and the delete is the one race left (see
    // `gc::reclaim_removed`). If the pre-untag scan failed (a corrupt
    // pointer, perhaps the very one just removed) nothing is deleted.
    if let (true, Some(before)) = (any_removed, before) {
        match before {
            Ok(before) => {
                let (blob_stats, cache_stats) =
                    gc::reclaim_removed(&store, &crate::default_cache()?, &before)?;
                if blob_stats.count > 0 || cache_stats.count > 0 {
                    println!(
                        "Freed {} ({} blobs, {} cache entries)",
                        crate::fmt::human_size(blob_stats.bytes + cache_stats.bytes),
                        blob_stats.count,
                        cache_stats.count
                    );
                }
            }
            Err(e) => eprintln!("[llmman] not freeing disk space after rm: {e:#}"),
        }
    }

    if any_err {
        anyhow::bail!("one or more removals failed");
    }
    Ok(())
}
