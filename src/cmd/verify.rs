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

    // Explicit --key adds to the matching rule rather than replacing
    // it: the rule's `claims` still applies, so this agrees with what
    // `pull` accepts for a relocated repository, and the rule's
    // identity is tried as a second acceptor after the keys
    // (verifySignatureLayer tries keys first, then the identity). With
    // --key the identity is only sent when a `trusted_root` backs it:
    // the user supplied the key to check against, so a missing root
    // drops the identity instead of refusing the whole check.
    let decision = crate::verify::decide(&reference)?;
    let (keys, claims, identity) =
        select_trust(&args.keys, decision, crate::verify::all_configured_keys)?;
    let keys: Vec<String> = keys
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    if keys.is_empty() && identity.is_none() {
        bail!(
            "no trusted public keys to check {reference} against — pass --key PATH, \
             or add a [[verify.trust]] rule to llmman.conf"
        );
    }
    if keys.is_empty()
        && identity
            .as_ref()
            .is_some_and(|identity| identity.trusted_root.is_none())
    {
        bail!(
            "cannot verify {reference}: the matching [[verify.trust]] rule names an identity \
             but no `trusted_root` is configured under [verify]"
        );
    }

    let report = ffi::verify(&reference, "", &keys, &claims, identity.as_ref())
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

/// Keys, claims and identity to check a reference against. With
/// explicit keys, those keys plus the matching rule's `claims` and its
/// identity, the latter only when a `trusted_root` backs it: without
/// one the shim could not check it, and the user asked about the keys.
/// Otherwise whatever the matching rule selects; failing that, every
/// key the policy names anywhere (`all_keys`). An explicit ask deserves
/// a real attempt: "here are the keys I know about, none signed it" is
/// more useful than "no rule matched". The rule's `claims` applies
/// either way, so the fallback keys are held to the identity the rule
/// names.
///
/// A keyless rule has no keys by construction and is not widened to
/// every configured key: it chose an identity, and that is what gets
/// checked.
fn select_trust(
    explicit_keys: &[String],
    decision: crate::verify::Decision,
    all_keys: impl FnOnce() -> anyhow::Result<Vec<PathBuf>>,
) -> anyhow::Result<(Vec<PathBuf>, String, Option<ffi::Identity>)> {
    if !explicit_keys.is_empty() {
        let keys = explicit_keys.iter().map(PathBuf::from).collect();
        let identity = decision
            .identity()
            .filter(|identity| identity.trusted_root.is_some());
        return Ok((keys, decision.claims.unwrap_or_default(), identity));
    }
    let identity = decision.identity();
    let claims = decision.claims.unwrap_or_default();
    let keys = if decision.keys.is_empty() && identity.is_none() {
        all_keys()?
    } else {
        decision.keys
    };
    Ok((keys, claims, identity))
}

fn print_human(report: &ffi::VerifyReport) {
    println!("Reference:  {}", report.reference);
    println!("Digest:     {}", report.digest);
    println!("Signatures: {}", report.signatures_found);
    if report.verified {
        println!("Verified:   yes");
        for m in &report.matches {
            println!("  signed by {} claiming {}", m.signer, m.identity);
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
    signer: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
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
                    signer: &m.signer,
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
    fn select_trust_keeps_claims_when_falling_back_to_all_keys() {
        let decision = Decision {
            mode: Mode::Warn,
            keys: vec![],
            claims: Some("docker.io/myorg/**".into()),
            issuer: None,
            subject: None,
            trusted_root: None,
        };
        let (keys, claims, identity) =
            select_trust(&[], decision, || Ok(vec![PathBuf::from("/k.pub")])).unwrap();
        assert_eq!(keys, vec![PathBuf::from("/k.pub")]);
        assert_eq!(claims, "docker.io/myorg/**");
        assert_eq!(identity, None);

        // A rule with keys uses them and never consults the fallback.
        let decision = Decision {
            mode: Mode::Warn,
            keys: vec![PathBuf::from("/rule.pub")],
            claims: None,
            issuer: None,
            subject: None,
            trusted_root: None,
        };
        let (keys, claims, identity) =
            select_trust(&[], decision, || panic!("fallback consulted")).unwrap();
        assert_eq!(keys, vec![PathBuf::from("/rule.pub")]);
        assert_eq!(claims, "");
        assert_eq!(identity, None);
    }

    #[test]
    fn select_trust_passes_an_identity_rule_through_without_widening_to_all_keys() {
        let decision = Decision {
            mode: Mode::Enforce,
            keys: vec![],
            claims: None,
            issuer: Some("https://issuer".into()),
            subject: Some("me@example.com".into()),
            trusted_root: Some(PathBuf::from("/etc/llmman/root.json")),
        };
        let (keys, claims, identity) =
            select_trust(&[], decision, || panic!("fallback consulted")).unwrap();
        assert!(keys.is_empty());
        assert_eq!(claims, "");
        assert_eq!(
            identity,
            Some(ffi::Identity {
                issuer: Some("https://issuer".into()),
                subject: Some("me@example.com".into()),
                trusted_root: Some("/etc/llmman/root.json".into()),
            })
        );
    }

    #[test]
    fn select_trust_with_explicit_keys_keeps_a_backed_identity_as_second_acceptor() {
        let decision = Decision {
            mode: Mode::Enforce,
            keys: vec![],
            claims: Some("docker.io/myorg/**".into()),
            issuer: Some("https://issuer".into()),
            subject: Some("me@example.com".into()),
            trusted_root: Some(PathBuf::from("/etc/llmman/root.json")),
        };
        let (keys, claims, identity) = select_trust(&["/other.pub".to_string()], decision, || {
            panic!("fallback consulted")
        })
        .unwrap();
        assert_eq!(keys, vec![PathBuf::from("/other.pub")]);
        assert_eq!(claims, "docker.io/myorg/**");
        assert_eq!(
            identity,
            Some(ffi::Identity {
                issuer: Some("https://issuer".into()),
                subject: Some("me@example.com".into()),
                trusted_root: Some("/etc/llmman/root.json".into()),
            })
        );
    }

    #[test]
    fn select_trust_with_explicit_keys_drops_an_identity_without_a_trusted_root() {
        // The user supplied the key to check against; a rule identity
        // the shim could not anchor is left out rather than refusing
        // the check, so `run` never bails on the missing root here.
        let decision = Decision {
            mode: Mode::Enforce,
            keys: vec![],
            claims: Some("docker.io/myorg/**".into()),
            issuer: Some("https://issuer".into()),
            subject: Some("me@example.com".into()),
            trusted_root: None,
        };
        let (keys, claims, identity) = select_trust(&["/other.pub".to_string()], decision, || {
            panic!("fallback consulted")
        })
        .unwrap();
        assert_eq!(keys, vec![PathBuf::from("/other.pub")]);
        assert_eq!(claims, "docker.io/myorg/**");
        assert_eq!(identity, None);
    }
}
