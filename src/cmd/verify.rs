use anyhow::{bail, Context};
use clap::Args;

use crate::ffi;

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Registry reference to verify, e.g. `docker.io/org/model:v1`
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// PEM public key to trust, repeatable. Overrides whatever
    /// verify.conf would have selected for this reference.
    #[arg(long = "key", value_name = "PATH")]
    pub keys: Vec<String>,

    /// Print the full report as JSON instead of a human summary.
    #[arg(long)]
    pub json: bool,
}

/// `llmman verify` reports whether a model in a registry carries a
/// cosign-format signature from a key you trust.
///
/// Talks to the Go shim directly rather than through a running daemon
/// (like `transfer`, `login` and `logout` already do): nothing about
/// this touches the daemon's store, and asking about a model you have
/// never pulled is the main reason to run it.
///
/// This is the diagnostic counterpart to the automatic check `pull`
/// performs. Where that one answers to a policy — and stays quiet when
/// the policy says `off` — this always checks and always reports, so
/// "why did my policy not fire?" and "who actually signed this?" are
/// answerable without editing config files. Consequently its exit status
/// reflects the *signature*, not the policy: unverified is a failure
/// here even where `verify.conf` would only have warned.
pub fn run(args: &VerifyArgs) -> anyhow::Result<()> {
    crate::shortnames::validate_reference(&args.reference)?;
    let reference = crate::shortnames::resolve(&args.reference)?;

    let keys: Vec<String> = if !args.keys.is_empty() {
        args.keys.clone()
    } else {
        // Whatever the policy selects for this reference; failing that,
        // every key it names anywhere. An explicit ask deserves a real
        // attempt — "here are the keys I know about, none signed it" is
        // more useful than "no rule matched".
        let mut keys = crate::verify::decide(&reference)?.keys;
        if keys.is_empty() {
            keys = crate::verify::all_configured_keys()?;
        }
        keys.iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    };
    if keys.is_empty() {
        bail!(
            "no trusted public keys to check {reference} against — pass --key PATH, \
             or add a [[trust]] rule to verify.conf"
        );
    }

    let report =
        ffi::verify(&reference, "", &keys).with_context(|| format!("verify {reference}"))?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonReport::from(&report))?
        );
    } else {
        print_human(&report);
    }

    if report.verified {
        Ok(())
    } else if report.signatures_found == 0 {
        bail!("{reference} is not signed")
    } else {
        bail!("{reference} is not signed by a trusted key")
    }
}

fn print_human(report: &ffi::VerifyReport) {
    println!("Reference:  {}", report.reference);
    println!("Digest:     {}", report.digest);
    println!("Signatures: {}", report.signatures_found);
    if report.verified {
        println!("Verified:   yes");
        for m in &report.matches {
            println!("  signed by {} claiming {}", m.key_path, m.identity);
        }
    } else {
        println!("Verified:   no");
        if !report.reason.is_empty() {
            println!("Reason:     {}", report.reason);
        }
    }
}

/// The `--json` shape. A separate serializable mirror of
/// [`ffi::VerifyReport`] (which only deserializes, being the Go shim's
/// output) so this command's own output format is defined here, where
/// it's read, rather than being whatever the FFI type happens to be.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonReport<'a> {
    reference: &'a str,
    digest: &'a str,
    verified: bool,
    signatures_found: u32,
    matches: Vec<JsonMatch<'a>>,
    #[serde(skip_serializing_if = "str::is_empty")]
    reason: &'a str,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonMatch<'a> {
    key_path: &'a str,
    identity: &'a str,
}

impl<'a> From<&'a ffi::VerifyReport> for JsonReport<'a> {
    fn from(r: &'a ffi::VerifyReport) -> Self {
        Self {
            reference: &r.reference,
            digest: &r.digest,
            verified: r.verified,
            signatures_found: r.signatures_found,
            matches: r
                .matches
                .iter()
                .map(|m| JsonMatch {
                    key_path: &m.key_path,
                    identity: &m.identity,
                })
                .collect(),
            reason: &r.reason,
        }
    }
}
