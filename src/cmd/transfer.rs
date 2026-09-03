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

    /// Sign the transferred manifest at the destination with this PEM
    /// private key, publishing a cosign-format signature beside it.
    /// Set LLMMAN_SIGN_PASSWORD (or COSIGN_PASSWORD) for an encrypted key.
    #[arg(long, value_name = "PATH")]
    pub sign_key: Option<String>,
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
/// Streaming a blob straight through needs its digest up front. An OCI
/// source gets that from the source manifest, so the transfer stays in
/// the Go shim where the registry protocol lives; HuggingFace needs a
/// HEAD per file first, which `crate::hf::transfer` does natively (for
/// `hf-xet`, which is Rust-only). The remaining sources can't stream at
/// all and stage through a throwaway layout — `crate::sources::transfer`.
///
/// This intentionally talks to the Go shim directly (like `login`/`logout`
/// and `inspect --remote`) rather than through a running `llmman serve`
/// daemon (like `pull`/`push`): a transfer never touches the daemon's
/// persistent store, so there's no shared state to coordinate.
///
/// A transfer is both a pull and a push, so it touches trust twice: an
/// OCI source gets the policy a `pull` would apply (skipping it would
/// launder an unverified model past that policy), and the destination is
/// signed over the digest the transfer reports pushing. HuggingFace and
/// object-store sources have no signature to check, which is why
/// `--sign-key` exists here — a transfer into a registry is where an
/// unsigned upstream becomes something you vouch for.
pub fn run(args: &TransferArgs) -> anyhow::Result<()> {
    let source = crate::shortnames::resolve(&args.source)?;
    let destination = crate::shortnames::resolve(&args.destination)?;
    // Before anything moves: an unreadable key is worth catching now
    // rather than after a multi-gigabyte transfer.
    let sign_key = args
        .sign_key
        .as_deref()
        .map(crate::verify::check_signing_key)
        .transpose()?;

    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let (outcome, source_confirmed) = rt.block_on(async {
        match crate::hf::classify(&source).await {
            ClassifiedRef::Hf(reference) => crate::hf::transfer::transfer(&reference, &destination)
                .await
                .map(|o| (o, true)),
            ClassifiedRef::Source(reference) => crate::sources::transfer(&reference, &destination)
                .await
                .map(|o| (o, true)),
            ClassifiedRef::Other(normalized) => {
                // An OCI source gets the same policy a pull would apply,
                // reported before the transfer rather than after it —
                // both so the warning arrives before a multi-gigabyte
                // copy, and so it is not lost if the copy then fails.
                let mut verdict = crate::verify::check(&normalized, None)?;
                verdict.report();
                verdict.notices.clear(); // reported; don't reprint below

                // Pinned to the digest that was verified, rather than
                // handing the tag over to be resolved a second time.
                // Detecting drift after the fact would be too late: the
                // transfer pushes before anything could compare, so the
                // unverified manifest would already be published at the
                // destination.
                let pinned = match &verdict.digest {
                    Some(digest) => crate::verify::pin_to_digest(&normalized, digest),
                    // Nothing was verified (policy off, or `warn` let it
                    // through), so there is no digest to pin to and
                    // nothing the transfer could contradict.
                    None => normalized.clone(),
                };
                let outcome = ffi::transfer(&pinned, &destination)?;

                // Defence in depth on the verified path, where the pin
                // makes it hold by construction. Unverified, there is no
                // digest to pin or compare — `--sign-key` then vouches
                // for whatever the tag resolved to, which is the user's
                // own claim to make.
                let confirmed = match (&verdict.digest, &outcome.digest) {
                    (Some(verified), Some(pushed)) => verified.eq_ignore_ascii_case(pushed),
                    (None, _) => true,
                    (Some(_), None) => false,
                };
                if !confirmed {
                    verdict.escalate(format!(
                        "{normalized} was verified, but {destination} received a different manifest"
                    ))?;
                    verdict.report();
                }
                Ok((outcome, confirmed))
            }
        }
    })?;

    if outcome.changed {
        println!("Transferred {source} to {destination}");
    } else {
        println!("{destination} already up to date with {source}");
    }

    if let Some(key) = &sign_key {
        // Refused regardless of mode. `warn` means "tell me and carry
        // on" about *reading* an unverified model; it cannot mean "sign
        // the thing you just told me changed under you" — a signature is
        // an explicit act of vouching, not something the source's policy
        // gets to relax.
        if !source_confirmed {
            anyhow::bail!(
                "refusing to sign {destination}: the source changed while being transferred, \
                 so the manifest pushed is not the one that was verified"
            );
        }
        // Signed even when nothing changed: the destination already
        // holding the content says nothing about whether it is signed,
        // and re-signing a digest this key already signed is a no-op.
        let digest = outcome.digest.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "cannot sign {destination}: the transfer did not report which manifest it pushed"
            )
        })?;
        let key = key
            .to_str()
            .context("signing key path is not valid UTF-8")?;
        let signed = ffi::sign(
            &destination,
            digest,
            key,
            &crate::verify::signing_password(),
        )
        .with_context(|| format!("sign {destination}"))?;
        println!("Signed {destination} ({signed}) with {key}");
    }
    Ok(())
}
