//! Hybrid model pairs: one model name carrying a local model and a hosted
//! one, with `llmman serve` choosing which serves each request.
//!
//! ```text
//! llmman.hybrid/<local-ref>,<provider>/<hosted-model>
//! llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5
//! ```
//!
//! The reference travels in the same `"model"` field a plain one does,
//! so every surface and integration carries it unchanged.
//! [`crate::cmd::serve::ensure_model`] resolves it to one side's own
//! ordinary reference before anything else runs, so loading, keep-alive,
//! model-name rewriting and provider auth need no third case.
//!
//! [`route`] is mechanical, not semantic: a [`ROUTE_HEADER`] pin wins,
//! otherwise a request too large for the local context goes to the
//! provider, otherwise it stays here. Local is the default because the
//! two mistakes are not equal: a worse local answer is recoverable,
//! data sent to someone else's servers is not.

use crate::providers;

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

/// Namespace every hybrid reference is carried under. Like
/// [`crate::providers::REMOTE_PREFIX`], the dotted first segment cannot
/// collide with a real registry host.
pub const HYBRID_PREFIX: &str = "llmman.hybrid/";

/// Separates the local half from the hosted half. Both halves contain
/// slashes and neither may contain a comma (OCI references are
/// `[a-zA-Z0-9._/:-]`, models.dev ids have none), so one `split_once`
/// is unambiguous. [`pair_with_local`] rejects a local name that has one.
const SEPARATOR: char = ',';

/// Request header pinning one request to one side: `local` or `cloud`.
pub const ROUTE_HEADER: &str = "x-llmman-route";

/// The two halves of a hybrid reference, validated as non-empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair<'a> {
    /// An ordinary llmman reference, resolved exactly as if asked for alone.
    pub local: &'a str,
    /// models.dev provider id serving the hosted half.
    pub provider: &'a str,
    /// The hosted model's id, as that provider knows it.
    pub model: &'a str,
}

impl Pair<'_> {
    /// The hosted half as the reference `cmd::serve` routes on.
    pub fn remote_ref(&self) -> String {
        providers::format_remote_ref(self.provider, self.model)
    }

    /// This pair's reference for `side`, with no trace of the pair left.
    pub fn side_ref(&self, side: Side) -> String {
        match side {
            Side::Local => self.local.to_string(),
            Side::Cloud => self.remote_ref(),
        }
    }
}

/// Unchecked inverse of [`split_ref`]; [`pair_with_local`] validates.
fn format_ref(local: &str, provider: &str, model: &str) -> String {
    format!("{HYBRID_PREFIX}{local}{SEPARATOR}{provider}/{model}")
}

/// Splits a [`HYBRID_PREFIX`] reference into its halves, or `None` for
/// any reference that is not one. Purely syntactic: an unknown provider
/// or missing local model is reported by whoever resolves that half.
///
/// The local half must itself be local. Otherwise
/// `llmman.hybrid/llmman.provider/openai/gpt-5,anthropic/claude` would
/// take a `local` pin to OpenAI.
pub fn split_ref(reference: &str) -> Option<Pair<'_>> {
    let rest = reference.strip_prefix(HYBRID_PREFIX)?;
    let (local, remote) = rest.split_once(SEPARATOR)?;
    let (provider, model) = remote.split_once('/')?;
    (!local.is_empty() && !provider.is_empty() && !model.is_empty() && is_local_ref(local))
        .then_some(Pair {
            local,
            provider,
            model,
        })
}

/// Whether `reference` is neither provider-routed nor itself a pair.
fn is_local_ref(reference: &str) -> bool {
    !providers::is_remote_ref(reference) && !reference.starts_with(HYBRID_PREFIX)
}

/// Whether `reference` names a hybrid pair.
pub fn is_hybrid_ref(reference: &str) -> bool {
    split_ref(reference).is_some()
}

/// The reference a local-only operation (preload, unload) should use: a
/// pair's local half, since the hosted half is never loaded; anything
/// else as-is.
pub fn local_half(reference: &str) -> &str {
    split_ref(reference).map_or(reference, |pair| pair.local)
}

/// Pairs a validated [`crate::providers::REMOTE_PREFIX`] reference with
/// a local one. The CLI builds pairs from the hosted reference
/// `--provider`/`--model` already resolved, so no second place has to
/// agree on how a hosted reference is spelled.
pub fn pair_with_local(local: &str, remote_ref: &str) -> anyhow::Result<String> {
    let (provider, model) = providers::split_remote_ref(remote_ref)
        .ok_or_else(|| anyhow::anyhow!("not a provider-routed reference: {remote_ref}"))?;
    anyhow::ensure!(
        !local.is_empty(),
        "a hybrid pair needs a local model as well as {provider}/{model}"
    );
    anyhow::ensure!(
        !local.contains(SEPARATOR),
        "local model {local:?} cannot be paired: {SEPARATOR:?} separates the two halves of a \
         hybrid reference"
    );
    anyhow::ensure!(
        is_local_ref(local),
        "{local:?} is not served on this machine, so it cannot be a pair's local half"
    );
    Ok(format_ref(local, provider, model))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// The hosted half `--overflow-provider`/`--overflow-model` name: `None`
/// when neither flag is given, an error when only one is, either is
/// blank, or `--provider` is also set. Blank is not absent, as in
/// [`crate::providers::provider_flag`]: `--overflow-model "$M"` with the
/// variable unset must fail rather than silently run local-only.
///
/// `--provider` is refused because it makes `--model` a hosted model,
/// and a pair's local half has to be local (see [`split_ref`]).
pub fn overflow_flags<'a>(
    provider: Option<&'a str>,
    model: Option<&'a str>,
    primary_provider: Option<&str>,
) -> anyhow::Result<Option<(&'a str, &'a str)>> {
    let provider = provider.map(str::trim);
    let model = model.map(str::trim);
    match (provider, model) {
        (None, None) => Ok(None),
        (Some(_), None) => anyhow::bail!(
            "--overflow-provider also needs --overflow-model naming that provider's model"
        ),
        (None, Some(_)) => anyhow::bail!(
            "--overflow-model also needs --overflow-provider (openai, anthropic, openrouter, ...)"
        ),
        (Some(provider), Some(model)) => {
            anyhow::ensure!(
                !provider.is_empty() && !model.is_empty(),
                "--overflow-provider and --overflow-model both need a value"
            );
            anyhow::ensure!(
                primary_provider.is_none(),
                "--overflow-provider pairs a hosted model with the local --model, so it cannot \
                 be combined with --provider: pass the local model as --model."
            );
            Ok(Some((provider, model)))
        }
    }
}

// ---------------------------------------------------------------------------
// Routing policy
// ---------------------------------------------------------------------------

/// Which half of a pair serves one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Local,
    Cloud,
}

impl Side {
    /// How a side is named in a log line and in [`ROUTE_HEADER`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }
}

/// Why [`route`] chose the side it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The caller sent [`ROUTE_HEADER`].
    Pinned,
    /// The request is larger than the local side's context can hold.
    Overflow { bytes: u64, budget: u64 },
    /// Nothing said otherwise.
    LocalFirst,
}

/// One routing decision: the side, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub side: Side,
    pub reason: Reason,
}

/// Applies the policy in the module doc. `request_bytes` is `None` for a
/// request that declared no length, `budget` is `None` when the overflow
/// rule is disabled; both fall through to [`Side::Local`], since an
/// unknown is not evidence a request has to leave the machine.
pub fn route(pin: Option<Side>, request_bytes: Option<u64>, budget: Option<u64>) -> Decision {
    if let Some(side) = pin {
        return Decision {
            side,
            reason: Reason::Pinned,
        };
    }
    match (request_bytes, budget) {
        (Some(bytes), Some(budget)) if bytes > budget => Decision {
            side: Side::Cloud,
            reason: Reason::Overflow { bytes, budget },
        },
        _ => Decision {
            side: Side::Local,
            reason: Reason::LocalFirst,
        },
    }
}

/// Reads [`ROUTE_HEADER`]: `None` when absent or blank, an error when
/// present but not a side. An error rather than a fallback, because this
/// is the caller stating where a request may go and a silent guess would
/// send it the wrong way invisibly. Takes raw bytes so a non-UTF-8 value
/// is unreadable, not absent.
pub fn parse_pin(value: Option<&[u8]>) -> anyhow::Result<Option<Side>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = std::str::from_utf8(value)
        .map_err(|_| anyhow::anyhow!("{ROUTE_HEADER}: value is not valid UTF-8"))?
        .trim();
    if value.is_empty() {
        return Ok(None);
    }
    match value.to_ascii_lowercase().as_str() {
        "local" => Ok(Some(Side::Local)),
        "cloud" => Ok(Some(Side::Cloud)),
        other => anyhow::bail!(
            "{ROUTE_HEADER}: {other:?} is not a side of a hybrid pair, use \"local\" or \"cloud\""
        ),
    }
}

/// Bytes of request body assumed per token of context. Generous on
/// purpose (JSON-encoded English is nearer 3.5): over-estimating keeps a
/// borderline request local.
const BYTES_PER_TOKEN: u64 = 4;

/// Largest request body the local side is assumed to hold, or `None` to
/// disable the overflow rule. Derived from the context size the daemon
/// starts its backends with, in bytes rather than tokens because the
/// choice is made before any tokenizer exists. `env`
/// (`LLMMAN_HYBRID_LOCAL_BYTES`) sets it outright; `0` disables the rule;
/// unparseable is treated as unset, like the daemon's other numeric
/// variables.
pub fn local_budget_bytes(ctx_size: Option<u32>, env: Option<&str>) -> Option<u64> {
    match env.map(str::trim).filter(|v| !v.is_empty()) {
        Some(v) => match v.parse::<u64>() {
            Ok(0) => None,
            Ok(bytes) => Some(bytes),
            Err(_) => ctx_size.and_then(budget_for_ctx),
        },
        None => ctx_size.and_then(budget_for_ctx),
    }
}

/// `None` for a zero context size, which `LLMMAN_CONTEXT_LENGTH=0` uses
/// to mean the model's trained context.
fn budget_for_ctx(ctx_size: u32) -> Option<u64> {
    (ctx_size > 0).then(|| u64::from(ctx_size).saturating_mul(BYTES_PER_TOKEN))
}

/// [`local_budget_bytes`] against this process's own environment.
pub fn local_budget_bytes_from_env(ctx_size: Option<u32>) -> Option<u64> {
    local_budget_bytes(
        ctx_size,
        std::env::var("LLMMAN_HYBRID_LOCAL_BYTES").ok().as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- References ---------------------------------------------------------

    #[test]
    fn a_pair_round_trips_through_its_reference() {
        let reference = format_ref("gemma4", "anthropic", "claude-sonnet-4-5");
        assert_eq!(
            reference,
            "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5"
        );
        assert_eq!(
            split_ref(&reference),
            Some(Pair {
                local: "gemma4",
                provider: "anthropic",
                model: "claude-sonnet-4-5",
            })
        );
    }

    /// A local `hf.co/<org>/<repo>` and an openrouter `<vendor>/<model>`
    /// put four slashes in one reference between them.
    #[test]
    fn both_halves_keep_their_own_slashes() {
        let reference = format_ref(
            "hf.co/ggml-org/gemma-3-270m-GGUF",
            "openrouter",
            "z-ai/glm-5",
        );
        let pair = split_ref(&reference).expect("multi-segment halves must parse");
        assert_eq!(pair.local, "hf.co/ggml-org/gemma-3-270m-GGUF");
        assert_eq!(pair.provider, "openrouter");
        assert_eq!(pair.model, "z-ai/glm-5");
    }

    #[test]
    fn a_plain_reference_is_not_a_pair() {
        for reference in [
            "gemma4",
            "hf.co/ggml-org/gemma-3-270m-GGUF",
            "llmman.provider/anthropic/claude-sonnet-4-5",
            // Prefixed, but missing a half or a provider segment.
            "llmman.hybrid/gemma4",
            "llmman.hybrid/gemma4,anthropic",
            "llmman.hybrid/,anthropic/claude",
            "llmman.hybrid/gemma4,/claude",
            "llmman.hybrid/gemma4,anthropic/",
        ] {
            assert_eq!(split_ref(reference), None, "{reference} parsed as a pair");
            assert!(!is_hybrid_ref(reference), "{reference} parsed as a pair");
        }
    }

    /// A `local` pin promises on-machine inference, so a provider-routed
    /// local half must not parse at all.
    #[test]
    fn a_local_half_that_is_not_local_is_not_a_pair() {
        for reference in [
            "llmman.hybrid/llmman.provider/openai/gpt-5,anthropic/claude",
            "llmman.hybrid/llmman.hybrid/a,b/c,anthropic/claude",
        ] {
            assert_eq!(split_ref(reference), None, "{reference} parsed as a pair");
        }
        assert!(pair_with_local(
            "llmman.provider/openai/gpt-5",
            "llmman.provider/anthropic/claude"
        )
        .is_err());
    }

    #[test]
    fn side_ref_yields_an_ordinary_reference_for_either_half() {
        let pair = split_ref("llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5").unwrap();
        assert_eq!(pair.side_ref(Side::Local), "gemma4");
        assert_eq!(
            pair.side_ref(Side::Cloud),
            "llmman.provider/anthropic/claude-sonnet-4-5"
        );
        // Neither is a pair any more, or `ensure_model`'s substitution
        // would loop.
        assert!(!is_hybrid_ref(&pair.side_ref(Side::Local)));
        assert!(!is_hybrid_ref(&pair.side_ref(Side::Cloud)));
    }

    #[test]
    fn local_half_of_a_pair_or_the_reference_itself() {
        assert_eq!(
            local_half("llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5"),
            "gemma4"
        );
        assert_eq!(local_half("gemma4:latest"), "gemma4:latest");
        assert_eq!(
            local_half("llmman.provider/anthropic/claude"),
            "llmman.provider/anthropic/claude"
        );
    }

    #[test]
    fn pair_with_local_wraps_a_provider_reference() {
        let remote = providers::format_remote_ref("anthropic", "claude-sonnet-4-5");
        assert_eq!(
            pair_with_local("gemma4", &remote).unwrap(),
            "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5"
        );
    }

    #[test]
    fn pair_with_local_rejects_halves_it_could_not_recover() {
        // Not a provider reference at all.
        assert!(pair_with_local("gemma4", "claude-sonnet-4-5").is_err());
        assert!(pair_with_local("", "llmman.provider/anthropic/claude").is_err());
        // A local name carrying the separator would re-split elsewhere.
        assert!(pair_with_local("a,b", "llmman.provider/anthropic/claude").is_err());
    }

    // -- CLI ----------------------------------------------------------------

    #[test]
    fn overflow_flags_need_each_other() {
        assert_eq!(
            overflow_flags(Some("anthropic"), Some("claude"), None).unwrap(),
            Some(("anthropic", "claude"))
        );
        assert_eq!(overflow_flags(None, None, None).unwrap(), None);
        assert_eq!(overflow_flags(None, None, Some("openai")).unwrap(), None);
        let err = overflow_flags(Some("anthropic"), None, None).unwrap_err();
        assert!(err.to_string().contains("--overflow-model"), "{err}");
        let err = overflow_flags(None, Some("claude"), None).unwrap_err();
        assert!(err.to_string().contains("--overflow-provider"), "{err}");
    }

    /// A hosted --model cannot be a pair's local half.
    #[test]
    fn overflow_flags_refuse_a_hosted_primary() {
        let err = overflow_flags(Some("anthropic"), Some("claude"), Some("openai")).unwrap_err();
        assert!(err.to_string().contains("--provider"), "{err}");
    }

    #[test]
    fn overflow_flags_reject_blank_values_and_trim() {
        assert!(overflow_flags(Some(""), Some("claude"), None).is_err());
        assert!(overflow_flags(Some("anthropic"), Some("  "), None).is_err());
        assert_eq!(
            overflow_flags(Some(" anthropic "), Some(" claude "), None).unwrap(),
            Some(("anthropic", "claude"))
        );
    }

    // -- Routing policy -----------------------------------------------------

    #[test]
    fn nothing_in_particular_stays_local() {
        assert_eq!(
            route(None, Some(1024), Some(262_144)),
            Decision {
                side: Side::Local,
                reason: Reason::LocalFirst
            }
        );
    }

    #[test]
    fn a_request_past_the_local_budget_goes_to_the_provider() {
        assert_eq!(
            route(None, Some(262_145), Some(262_144)),
            Decision {
                side: Side::Cloud,
                reason: Reason::Overflow {
                    bytes: 262_145,
                    budget: 262_144
                }
            }
        );
        // Exactly at the budget still fits.
        assert_eq!(route(None, Some(262_144), Some(262_144)).side, Side::Local);
    }

    #[test]
    fn an_unknown_size_or_no_budget_stays_local() {
        assert_eq!(route(None, None, Some(1)).side, Side::Local);
        assert_eq!(route(None, Some(u64::MAX), None).side, Side::Local);
        assert_eq!(route(None, None, None).side, Side::Local);
    }

    #[test]
    fn a_pinned_side_beats_every_other_rule() {
        // Cloud, despite fitting locally.
        assert_eq!(
            route(Some(Side::Cloud), Some(1), Some(u64::MAX)),
            Decision {
                side: Side::Cloud,
                reason: Reason::Pinned
            }
        );
        // Local, despite overflowing.
        assert_eq!(
            route(Some(Side::Local), Some(u64::MAX), Some(1)),
            Decision {
                side: Side::Local,
                reason: Reason::Pinned
            }
        );
    }

    #[test]
    fn parse_pin_reads_both_sides_case_insensitively() {
        assert_eq!(parse_pin(Some(b"local")).unwrap(), Some(Side::Local));
        assert_eq!(parse_pin(Some(b" Local ")).unwrap(), Some(Side::Local));
        assert_eq!(parse_pin(Some(b"cloud")).unwrap(), Some(Side::Cloud));
        assert_eq!(parse_pin(Some(b"CLOUD")).unwrap(), Some(Side::Cloud));
    }

    #[test]
    fn parse_pin_treats_absent_and_blank_alike() {
        assert_eq!(parse_pin(None).unwrap(), None);
        assert_eq!(parse_pin(Some(b"")).unwrap(), None);
        assert_eq!(parse_pin(Some(b"   ")).unwrap(), None);
    }

    /// Only the two documented spellings are accepted; a misspelled pin
    /// must not fall back to either side.
    #[test]
    fn parse_pin_rejects_anything_that_is_not_a_side() {
        for value in [
            &b"locl"[..],
            b"on-device",
            b"remote",
            b"true",
            b"cloud,local",
            // Present but not UTF-8: unreadable, not absent.
            &[0xff, 0xfe][..],
        ] {
            let err = parse_pin(Some(value)).expect_err("must not parse as a side");
            assert!(
                err.to_string().contains(ROUTE_HEADER),
                "error must name the header: {err}"
            );
        }
    }

    // -- Local budget -------------------------------------------------------

    #[test]
    fn the_budget_follows_the_daemons_own_context_length() {
        assert_eq!(local_budget_bytes(Some(65536), None), Some(262_144));
        assert_eq!(local_budget_bytes(Some(4096), None), Some(16_384));
    }

    /// `LLMMAN_CONTEXT_LENGTH=0` defers to the trained context.
    #[test]
    fn no_context_length_means_no_budget() {
        assert_eq!(local_budget_bytes(None, None), None);
        assert_eq!(local_budget_bytes(Some(0), None), None);
    }

    #[test]
    fn the_environment_overrides_the_estimate() {
        assert_eq!(local_budget_bytes(Some(65536), Some("1024")), Some(1024));
        assert_eq!(local_budget_bytes(None, Some("1024")), Some(1024));
    }

    #[test]
    fn zero_disables_the_overflow_rule_entirely() {
        assert_eq!(local_budget_bytes(Some(65536), Some("0")), None);
        assert_eq!(route(None, Some(u64::MAX), None).side, Side::Local);
    }

    #[test]
    fn an_unparseable_override_falls_back_to_the_estimate() {
        assert_eq!(local_budget_bytes(Some(4096), Some("lots")), Some(16_384));
        assert_eq!(local_budget_bytes(Some(4096), Some("-1")), Some(16_384));
        assert_eq!(local_budget_bytes(Some(4096), Some("  ")), Some(16_384));
    }

    #[test]
    fn an_absurd_context_length_does_not_overflow_the_budget() {
        assert_eq!(
            local_budget_bytes(Some(u32::MAX), None),
            Some(u64::from(u32::MAX) * 4)
        );
    }
}
