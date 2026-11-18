//! `llmman resolve` — pull (if needed) and extract a model reference to a
//! local path, printing the result as one line of JSON on stdout.
//!
//! Hidden from `--help` (`main.rs`'s `Commands::Resolve`): it's not a
//! user-facing subcommand, just an internal hook for the `vllm-llmman`
//! plugin, which shells out to it so `vllm serve oci://<ref>` can pull a
//! CNCF ModelPack image and hand vLLM the extracted path directly.
//! Unlike `llmman pull` (a client of `llmman serve`'s daemon), this runs
//! in-process and never starts a daemon or backend of its own.
//!
//! Output contract (stdout, on success, exactly one line):
//! ```json
//! {"reference":"ghcr.io/org/model:tag","path":"/abs/path","format":"safetensors"}
//! ```
//! `format` is either `"safetensors"` (a directory containing
//! `config.json`) or `"gguf"` (a single `.gguf` file); a `"gguf"` result
//! additionally carries `"mmproj":"/abs/path"` when a companion
//! multimodal projector file (see `modelpack::ModelPath::mmproj`'s doc
//! comment) was found alongside it, omitted entirely otherwise. Any
//! diagnostic output (pull progress, extraction notes) goes to stderr,
//! same as every other `llmman` subcommand, so stdout stays parseable.
//!
//! With `--dir`, the output is the whole model directory (every layer, docs
//! included), as the Kubernetes CSI driver (`docs/kubernetes.md`) needs:
//! ```json
//! {"reference":"ghcr.io/org/model:tag","dir":"/abs/cache/<manifest hex>","digest":"sha256:...","format":"safetensors"}
//! ```
//! `format` is what [`crate::modelpack::resolve_model`] would serve it as, or
//! `"unknown"`.
//!
//! The pull obeys the `[verify]` policy as `llmman pull` does (see
//! `cmd::serve::pull_checked`). Concurrent `resolve`s take turns pulling, as
//! two writing one `.part` file would corrupt it.
//!
//! On failure, nothing is printed to stdout, an error is printed to
//! stderr, and the process exits non-zero — the same convention `main`
//! already applies to every other subcommand's `Err`.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use serde::Serialize;

use crate::modelpack::{resolve_model, ModelFormat};
use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct ResolveArgs {
    /// Model reference to resolve (e.g. ghcr.io/org/model:tag)
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// Directory extracted model files are cached under (defaults to
    /// `<store>/cache`)
    #[arg(long)]
    pub cache: Option<PathBuf>,

    /// Fail instead of pulling if `reference` isn't already in the local
    /// store
    #[arg(long)]
    pub no_pull: bool,

    /// Print the whole model directory (every layer) instead of the servable path
    #[arg(long)]
    pub dir: bool,
}

#[derive(Serialize)]
struct ResolveOutput<'a> {
    reference: &'a str,
    path: String,
    format: &'static str,
    /// A companion `--mmproj` multimodal projector file resolved
    /// alongside a GGUF model (see
    /// `modelpack::ModelPath::mmproj`'s doc comment), if any. Always
    /// absent for `format: "safetensors"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    mmproj: Option<String>,
}

#[derive(Serialize)]
struct TreeOutput<'a> {
    reference: &'a str,
    dir: String,
    digest: String,
    format: &'static str,
}

pub fn run(args: &ResolveArgs) -> anyhow::Result<()> {
    let reference = crate::shortnames::resolve_ollama_api(&args.reference)?;

    let store_path = crate::default_store()?;
    let cache_path = args
        .cache
        .clone()
        .map(Ok)
        .unwrap_or_else(crate::default_cache)?;
    std::fs::create_dir_all(&store_path)
        .with_context(|| format!("creating store dir {}", store_path.display()))?;
    std::fs::create_dir_all(&cache_path)
        .with_context(|| format!("creating cache dir {}", cache_path.display()))?;

    let pull_lock = if OciStore::open(&store_path)
        .and_then(|s| s.find(&reference))
        .is_ok()
    {
        None
    } else {
        if args.no_pull {
            anyhow::bail!(
                "{reference} not found in local store ({}) and --no-pull was set",
                store_path.display()
            );
        }
        // A waiter finds the model stored by then, and only verifies it.
        Some(crate::managed::InstallLock::acquire_noting(
            &store_path,
            "another llmman process is pulling into this store; waiting for it to finish",
        )?)
    };
    let mut notices = tokio::runtime::Runtime::new()
        .context("start tokio runtime")?
        .block_on(crate::cmd::serve::pull_checked(&store_path, &reference))
        .with_context(|| format!("pulling {reference}"))?;
    drop(pull_lock);
    // `load_ref` pins the manifest the layer check hashed, so the load
    // below cannot pick up a tag repointed since; `reference` stays the
    // spelling printed back. See `verify_layers_stored`.
    let (load_ref, layer_notices) =
        crate::cmd::serve::verify_layers_stored(&store_path, &reference)
            .with_context(|| format!("verify layers of {reference}"))?;
    notices.extend(layer_notices);
    for notice in notices {
        eprintln!("[llmman] {notice}");
    }

    if args.dir {
        let tree = crate::modelpack::resolve_model_tree(&store_path, &cache_path, &load_ref)
            .with_context(|| format!("resolving {reference}"))?;
        let out = TreeOutput {
            reference: &reference,
            dir: tree.root.to_string_lossy().into_owned(),
            digest: tree.digest,
            format: tree.format.map_or("unknown", ModelFormat::as_str),
        };
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }

    let resolved = resolve_model(&store_path, &cache_path, &load_ref)
        .with_context(|| format!("resolving {reference}"))?;

    let out = ResolveOutput {
        reference: &reference,
        path: resolved.path().to_string_lossy().into_owned(),
        format: resolved.format(),
        mmproj: resolved.mmproj().map(|p| p.to_string_lossy().into_owned()),
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
