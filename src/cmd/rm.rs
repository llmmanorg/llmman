use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct RmArgs {
    /// Reference(s) to remove (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE", required = true, num_args = 1..)]
    pub references: Vec<String>,
}

pub fn run(args: &RmArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let mut any_err = false;
    for raw in &args.references {
        // resolve_ollama_api, not resolve: a bare name pulled via the
        // Ollama API (POST /api/pull, /api/chat, ...) is stored under
        // docker.io/ai/<name>, not hf.co/<name> — must resolve the same
        // way here or `llmman rm <bare-name>` looks for the wrong entry.
        let reference = crate::shortnames::resolve_ollama_api(raw);
        match store.remove(&reference) {
            Ok(()) => println!("Removed {}", reference),
            Err(e) => {
                eprintln!("Error removing {}: {}", reference, e);
                any_err = true;
            }
        }
    }
    if any_err {
        anyhow::bail!("one or more removals failed");
    }
    Ok(())
}
