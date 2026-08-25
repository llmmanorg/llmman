//! `llmman resolve` — pull (if needed) and extract a model reference to a
//! local path, printing the result as one line of JSON on stdout.
//!
//! Unlike `llmman pull` (a thin client of `llmman serve`'s daemon) this
//! runs entirely in-process and never starts a daemon or an inference
//! backend of its own — it exists for external tools that want to load
//! the model themselves. The motivating consumer is a vLLM plugin
//! (see `vllm-llmman` in the llmman project) that shells out to this
//! subcommand so `vllm serve oci://<ref>` can pull a CNCF ModelPack
//! image instead of a HuggingFace repo, then hand the extracted directory
//! (or `.gguf` file) to vLLM exactly as if it had been a local path all
//! along.
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
//! On failure, nothing is printed to stdout, an error is printed to
//! stderr, and the process exits non-zero — the same convention `main`
//! already applies to every other subcommand's `Err`.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use serde::Serialize;

use crate::modelpack::resolve_model;
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

pub fn run(args: &ResolveArgs) -> anyhow::Result<()> {
    let reference = crate::shortnames::resolve_ollama_api(&args.reference);

    let store_path = crate::default_store()?;
    let cache_path = args
        .cache
        .clone()
        .unwrap_or_else(|| store_path.join("cache"));
    std::fs::create_dir_all(&store_path)
        .with_context(|| format!("creating store dir {}", store_path.display()))?;
    std::fs::create_dir_all(&cache_path)
        .with_context(|| format!("creating cache dir {}", cache_path.display()))?;

    let already_present = OciStore::open(&store_path)
        .and_then(|s| s.find(&reference))
        .is_ok();
    if !already_present {
        if args.no_pull {
            anyhow::bail!(
                "{reference} not found in local store ({}) and --no-pull was set",
                store_path.display()
            );
        }
        let layout_dir = store_path
            .to_str()
            .context("store path is not valid UTF-8")?;
        crate::ffi::pull(&reference, layout_dir).with_context(|| format!("pulling {reference}"))?;
    }

    let resolved = resolve_model(&store_path, &cache_path, &reference)
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
