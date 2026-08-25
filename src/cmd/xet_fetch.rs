//! `llmman __xet-fetch` — a hidden, undocumented subcommand that
//! downloads one Xet-backed HuggingFace file via `hf-xet` (see
//! `crate::xet_fetch`'s doc comment for why this is a subcommand rather
//! than a Go↔Rust FFI call: the Go shim self-execs this binary instead).
//!
//! Not meant to be run by hand — every field is exactly what the Go
//! shim's own header-parsing already has from a resolve/HEAD request.
//! Mirrors `gpu-discover`'s hidden-subcommand shape rather than
//! `hostgpu::probe_subprocess_main`'s lower-level pre-clap self-exec:
//! this never touches the Go shim/cgo boundary, so there's nothing to
//! gain from bypassing clap.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::xet_fetch::{self, XetFileRef};

#[derive(Args, Debug)]
pub struct XetFetchArgs {
    /// "owner/repo", e.g. ornith-ai/Ornith-1.5-397B-GGUF
    #[arg(value_name = "OWNER/REPO")]
    pub owner_repo: String,

    /// The commit this file's hash/size were resolved against.
    #[arg(long)]
    pub revision: String,

    /// The Xet Merkle hash of the file's content (resolve response's `X-Xet-Hash` header).
    #[arg(long)]
    pub hash: String,

    /// The file's real size in bytes (resolve response's `X-Linked-Size` header).
    #[arg(long)]
    pub size: u64,

    /// The file's SHA-256, if known (resolve response's `X-Linked-Etag` header, unquoted).
    #[arg(long)]
    pub sha256: Option<String>,

    /// Repo type as used in the Hub API path.
    #[arg(long, default_value = "models")]
    pub repo_type: String,

    /// HuggingFace endpoint override — defaults to `HF_ENDPOINT`/https://huggingface.co,
    /// same as every other HF call in this codebase (see `hf::endpoint`).
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Where to write the reconstructed file. Parent directory must already exist.
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,
}

pub fn run(args: &XetFetchArgs) -> anyhow::Result<()> {
    let file = XetFileRef {
        endpoint: args.endpoint.clone().unwrap_or_else(crate::hf::endpoint),
        repo_type: args.repo_type.clone(),
        owner_repo: args.owner_repo.clone(),
        revision: args.revision.clone(),
        hash: args.hash.clone(),
        size: args.size,
        sha256: args.sha256.clone(),
        hf_token: crate::hf::token(),
    };

    tokio::runtime::Runtime::new()
        .context("start tokio runtime")?
        .block_on(xet_fetch::fetch_to_path(&file, &args.output))
}
