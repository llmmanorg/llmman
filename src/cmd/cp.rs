use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct CpArgs {
    /// Existing local reference to copy
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// New reference to create, pointing at the same image
    #[arg(value_name = "DESTINATION")]
    pub destination: String,
}

/// `llmman cp SOURCE DESTINATION` — copy a model under a new name,
/// mirroring `ollama cp` (`cmd/cmd.go`'s `CopyHandler`, `server/
/// images.go`'s `CopyModel`): both are really just "point a second
/// reference at the same content", exactly what `llmman tag` already does
/// (see `storage::OciStore::tag`) — `cp` exists as its own command purely
/// to match ollama's naming for anyone coming from there, not because the
/// underlying operation differs at all.
pub fn run(args: &CpArgs) -> anyhow::Result<()> {
    // Validate/resolve both refs before touching the store, so a bad ref
    // never creates the store tree. DESTINATION is stored verbatim, not
    // resolved/canonicalized, so this is its only client-side validation
    // gate. resolve_ollama_api for SOURCE, not resolve: it must match how
    // the model is actually stored, same reasoning as tag.rs/rm.rs.
    crate::shortnames::validate_reference(&args.destination)?;
    let source = crate::shortnames::resolve_ollama_api(&args.source)?;

    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let desc = store.find(&source)?;
    // A destination spelled as a digest names content, and the store
    // answers such a reference by content (see `OciStore::find`): a
    // pointer there holding some other digest would be passed over by a
    // lookup for the digest it is named for and answer only for the one
    // it holds, so it is refused.
    check_digest_destination(&args.destination, &desc.digest)?;
    store.tag(desc, &args.destination)?;
    println!("copied '{}' to '{}'", args.source, args.destination);
    Ok(())
}

/// Refuses a `<name>@<digest>` destination whose digest is not `actual`,
/// the copied descriptor's. A tag, or the matching digest in either
/// case, passes.
fn check_digest_destination(destination: &str, actual: &str) -> anyhow::Result<()> {
    match crate::storage::split_ref_digest(destination).1 {
        Some(digest) if !digest.eq_ignore_ascii_case(actual) => anyhow::bail!(
            "destination {destination} names digest {digest}, but the source's is {actual}"
        ),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A digest-shaped destination has to name the content being copied.
    #[test]
    fn a_digest_destination_must_match_the_copied_digest() {
        assert!(check_digest_destination("docker.io/ai/m:v9", "sha256:aaaa").is_ok());
        assert!(check_digest_destination("docker.io/ai/m@sha256:aaaa", "sha256:aaaa").is_ok());
        assert!(check_digest_destination("docker.io/ai/m@sha256:AAAA", "sha256:aaaa").is_ok());
        assert!(check_digest_destination("docker.io/ai/m@sha256:bbbb", "sha256:aaaa").is_err());
    }

    #[test]
    fn cp_rejects_an_invalid_destination_before_writing() {
        // Validation runs before default_store()/OciStore::open(), so a bad
        // destination errors without ever touching the filesystem or env.
        let args = CpArgs {
            source: "gemma4".into(),
            destination: "hf.co//foo".into(),
        };
        let err = run(&args).expect_err("invalid destination must error");
        assert!(
            err.to_string().contains("invalid model reference"),
            "expected a validation error, got: {err}"
        );
    }
}
