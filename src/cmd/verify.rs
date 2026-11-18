use std::path::PathBuf;

use anyhow::{bail, Context};
use clap::Args;

use crate::ffi;

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Registry reference to verify, e.g. `docker.io/org/model:v1`
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// PEM public key to trust, repeatable. Overrides whatever
    /// the `[verify]` policy would have selected for this reference.
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
/// here even where the `[verify]` policy would only have warned.
pub fn run(args: &VerifyArgs) -> anyhow::Result<()> {
    crate::shortnames::validate_reference(&args.reference)?;
    let reference = crate::shortnames::resolve(&args.reference)?;

    // Explicit --key overrides only the keys; the matching rule's
    // `claims` still applies, so this agrees with what `pull` accepts
    // for a relocated repository.
    let (keys, claims): (Vec<String>, String) = if !args.keys.is_empty() {
        let claims = crate::verify::decide(&reference)?
            .claims
            .unwrap_or_default();
        (args.keys.clone(), claims)
    } else {
        let (keys, claims) = from_policy(
            crate::verify::decide(&reference)?,
            crate::verify::all_configured_keys,
        )?;
        (
            keys.iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
            claims,
        )
    };
    if keys.is_empty() {
        bail!(
            "no trusted public keys to check {reference} against — pass --key PATH, \
             or add a [[verify.trust]] rule to llmman.conf"
        );
    }

    let report = ffi::verify(&reference, "", &keys, &claims)
        .with_context(|| format!("verify {reference}"))?;

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

/// Keys and claims for a reference the policy was asked about: whatever
/// the matching rule selects; failing that, every key the policy names
/// anywhere (`all_keys`). An explicit ask deserves a real attempt — "here
/// are the keys I know about, none signed it" is more useful than "no
/// rule matched". The rule's `claims` applies either way, so the
/// fallback keys are held to the identity the rule names.
fn from_policy(
    decision: crate::verify::Decision,
    all_keys: impl FnOnce() -> anyhow::Result<Vec<PathBuf>>,
) -> anyhow::Result<(Vec<PathBuf>, String)> {
    let claims = decision.claims.unwrap_or_default();
    let keys = if decision.keys.is_empty() {
        all_keys()?
    } else {
        decision.keys
    };
    Ok((keys, claims))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::{Decision, Mode};

    #[test]
    fn from_policy_keeps_claims_when_falling_back_to_all_keys() {
        let decision = Decision {
            mode: Mode::Warn,
            keys: vec![],
            claims: Some("docker.io/myorg/**".into()),
        };
        let (keys, claims) = from_policy(decision, || Ok(vec![PathBuf::from("/k.pub")])).unwrap();
        assert_eq!(keys, vec![PathBuf::from("/k.pub")]);
        assert_eq!(claims, "docker.io/myorg/**");

        // A rule with keys uses them and never consults the fallback.
        let decision = Decision {
            mode: Mode::Warn,
            keys: vec![PathBuf::from("/rule.pub")],
            claims: None,
        };
        let (keys, claims) = from_policy(decision, || panic!("fallback consulted")).unwrap();
        assert_eq!(keys, vec![PathBuf::from("/rule.pub")]);
        assert_eq!(claims, "");
    }
}
