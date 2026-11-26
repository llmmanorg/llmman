use clap::Args;

use crate::cmd::serve;
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
    let mounted = serve::mounted::MountedModels::lenient_from_env();
    for raw in &args.references {
        if let Some(msg) = mounted.refusal(raw, "removed") {
            eprintln!("Error removing {raw}: {msg}");
            any_err = true;
            continue;
        }
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

    // Snapshot before the untag, to tell its orphans from other unreferenced blobs.
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

    // Free at once what the removals orphaned. Other unreferenced blobs may be
    // a pull in flight in `serve`, so they wait for the grace-gated sweep.
    if let (true, Some(before)) = (any_removed, before) {
        match before {
            Ok(before) => {
                let stats = gc::reclaim_removed(&store, &crate::default_cache()?, &before)?;
                if let Some((size, counts)) = gc::freed(&stats) {
                    println!("Freed {size} {counts}");
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
