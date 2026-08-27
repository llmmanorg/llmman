use anyhow::Context;
use clap::Args;

use crate::ffi;
use crate::hf::ClassifiedRef;

#[derive(Args, Debug)]
pub struct TransferArgs {
    /// Source reference to transfer from, e.g. `hf.co/owner/repo`,
    /// `registry.example.com/repo:tag`, or any other reference `llmman
    /// pull` understands (hf://, ms://, ngc://, s3://, gs://, ...)
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// Destination OCI registry reference to transfer to, e.g.
    /// `registry.example.com/repo:tag`
    #[arg(value_name = "DESTINATION")]
    pub destination: String,
}

/// `llmman transfer` transfers an image directly from one location to
/// another without leaving it behind in the persistent local store (see
/// `cmd::pull`/`cmd::push` for that).
///
/// The motivating case is HuggingFace → OCI registry —
/// `llmman transfer hf.co/owner/model registry.example.com/owner/model` —
/// but any source `llmman pull` understands (an OCI registry, `hf://`,
/// `ms://`, ...) can be paired with any OCI registry destination.
///
/// Streaming a blob straight from source to destination only works once
/// its digest is already known. For an OCI registry source that digest
/// comes straight from the source manifest, so the whole transfer stays
/// in the Go shim (`go-shim/transfer_docker.go` / `transfer_podman.go`),
/// where the registry protocol lives. For a HuggingFace source it takes
/// a HEAD request against each file first, which `crate::hf::transfer`
/// does natively — `hf-xet`, needed for files too large for a plain HTTP
/// GET to reliably serve, is Rust-only.
///
/// The remaining sources (`ms://`, `ngc://`, `s3://`, `gs://`, a local
/// path) are generic file stores with no way to learn a file's content
/// digest ahead of downloading it, so they can't stream at all: they
/// stage through a throwaway local layout and push that — see
/// `crate::sources::transfer`.
///
/// This intentionally talks to the Go shim directly (like `login`/`logout`
/// and `inspect --remote`) rather than through a running `llmman serve`
/// daemon (like `pull`/`push`): a transfer never touches the daemon's
/// persistent store, so there's no shared state to coordinate.
pub fn run(args: &TransferArgs) -> anyhow::Result<()> {
    let source = crate::shortnames::resolve(&args.source)?;
    let destination = crate::shortnames::resolve(&args.destination)?;

    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let changed = rt.block_on(async {
        match crate::hf::classify(&source).await {
            ClassifiedRef::Hf(reference) => {
                crate::hf::transfer::transfer(&reference, &destination).await
            }
            ClassifiedRef::Source(reference) => {
                crate::sources::transfer(&reference, &destination).await
            }
            ClassifiedRef::Other(normalized) => ffi::transfer(&normalized, &destination),
        }
    })?;

    if changed {
        println!("Transferred {source} to {destination}");
    } else {
        println!("{destination} already up to date with {source}");
    }
    Ok(())
}
