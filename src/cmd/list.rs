//! `llmman list` — locally stored images, or (with `--provider`) the
//! models a hosted provider serves.
//!
//! One command for both, because "which models can I run" has one
//! answer; where the weights sit is a detail of how `llmman serve`
//! reaches them (see `Target` in cmd::serve). `--provider`'s answer comes
//! from the daemon's catalog — see cmd::providers for why.

use clap::Args;

use crate::fmt::{human_size, relative_time, short_id};
use crate::storage::oci::ImageSummary;
use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Only show images whose repository (ignoring tag) matches this reference
    #[arg(value_name = "REFERENCE")]
    pub reference: Option<String>,
    /// Format output using a Go-template-style expression, e.g. --format="{{.ID}}".
    /// Available fields: .ID, .Digest, .Name, .Repository, .Tag, .Size, .Modified
    #[arg(long, value_name = "TEMPLATE")]
    pub format: Option<String>,
    /// List the models this hosted provider serves (openai, anthropic,
    /// openrouter, …) instead of local images. See `llmman providers`.
    #[arg(
        long,
        short = 'p',
        value_name = "PROVIDER",
        conflicts_with_all = ["reference", "format"]
    )]
    pub provider: Option<String>,
}

pub fn run(args: &ListArgs) -> anyhow::Result<()> {
    if let Some(provider) = crate::providers::provider_flag(args.provider.as_deref())? {
        return list_provider_models(provider);
    }

    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;
    let mut images = store.list()?;

    if let Some(filter) = &args.reference {
        // Resolve once up front so an invalid filter surfaces its
        // validation error instead of silently matching nothing and
        // printing an empty table.
        let filter = crate::shortnames::resolve_ollama_api(filter)?;
        images.retain(|i| matches_filter(&i.reference, &filter));
    }

    if images.is_empty() {
        return Ok(());
    }

    if let Some(template) = &args.format {
        for img in &images {
            println!("{}", render_format(template, img)?);
        }
        return Ok(());
    }

    let name_w = images
        .iter()
        .map(|i| i.reference.len())
        .max()
        .unwrap_or(4)
        .max(4);

    println!(
        "{:<name_w$}    {:<16}    {:<10}    MODIFIED",
        "NAME",
        "ID",
        "SIZE",
        name_w = name_w,
    );

    for img in &images {
        println!(
            "{:<name_w$}    {:<16}    {:<10}    {}",
            img.reference,
            short_id(&img.digest),
            human_size(img.size),
            relative_time(img.modified_at),
            name_w = name_w,
        );
    }
    Ok(())
}

/// Prints the models a provider serves and what it charges, from the
/// daemon's catalog (`GET /llmman/providers/:id`). Nothing is downloaded:
/// these are ids for `llmman run --provider`. Price replaces
/// SIZE/MODIFIED, which mean nothing for someone else's weights.
fn list_provider_models(provider: &str) -> anyhow::Result<()> {
    crate::daemon::ensure_server("")?;
    let entry = crate::daemon::provider(provider)?;
    if entry.models.is_empty() {
        // As for an empty local store: nothing to tabulate. The provider
        // exists — an unknown one is an error from the daemon.
        return Ok(());
    }
    let name_w = entry
        .models
        .iter()
        .map(|m| m.id.len())
        .max()
        .unwrap_or(4)
        .max(4);
    // "IN $/Mtok" is the widest thing in its column, so it sizes itself.
    println!(
        "{:<name_w$}    {:<9}    OUT $/Mtok",
        "NAME",
        "IN $/Mtok",
        name_w = name_w
    );
    for model in &entry.models {
        let cost = model.cost;
        println!(
            "{:<name_w$}    {:<9}    {}",
            model.id,
            price(cost.map(|c| c.input)),
            price(cost.map(|c| c.output)),
            name_w = name_w,
        );
    }
    Ok(())
}

/// Renders one dollars-per-million-tokens figure.
///
/// `None` (no published price) is `-`, never `0`: showing a paid model as
/// free is worse than admitting llmman doesn't know. A genuinely free one
/// — openrouter has plenty — still prints `0`.
///
/// Trailing zeros go because the range spans four orders of magnitude
/// ($0.0023 and $600 are both real), so any fixed precision either loses
/// the cheap end or pads the expensive one.
fn price(value: Option<f64>) -> String {
    let Some(value) = value else {
        return "-".to_string();
    };
    if value == 0.0 {
        return "0".to_string();
    }
    let rendered = if value >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.5}")
    };
    let trimmed = rendered.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "0" {
        return "<0.00001".to_string();
    }
    trimmed.to_string()
}

/// Splits a stored reference into its repository and *explicit* tag,
/// matching the tag-detection rule used by `storage::oci::tag_from_ref`:
/// a `:` that appears after the last `/` is a tag separator. Returns
/// `None` for the tag when none is present, so callers can distinguish
/// "no tag given" from "explicitly tagged `:latest`".
fn split_repo_tag(reference: &str) -> (&str, Option<&str>) {
    if let Some(pos) = reference.rfind(':') {
        if pos > reference.rfind('/').unwrap_or(0) {
            return (&reference[..pos], Some(&reference[pos + 1..]));
        }
    }
    (reference, None)
}

/// Returns true if a stored image `reference` should be included given a
/// `resolved_filter` already run through `shortnames::resolve_ollama_api`
/// by `run`: the same resolution `rm`/`tag` apply to references they look
/// up in the local store, so a bare owner/repo filter like
/// `unsloth/Qwen3.5-0.8B` matches a stored `hf.co/unsloth/Qwen3.5-0.8B:latest`
/// (registries default to `hf.co/`, exactly as they do when the model was
/// pulled). Resolution happens once in `run`, not here per image, so an
/// invalid filter is a reported error rather than a silently empty table.
/// The resolved filter matches either the full stored reference verbatim,
/// or (when the filter carried no explicit tag) just its repository part.
fn matches_filter(reference: &str, resolved_filter: &str) -> bool {
    if reference == resolved_filter {
        return true;
    }
    let (ref_repo, _) = split_repo_tag(reference);
    let (filter_repo, filter_tag) = split_repo_tag(resolved_filter);
    filter_tag.is_none() && ref_repo == filter_repo
}

/// Renders a Docker/Go-template-style format string (e.g. `"{{.ID}}"`)
/// against a single image. Only plain `{{.Field}}` substitutions are
/// supported (no pipelines, conditionals, etc.) — sufficient for
/// `docker images --format`-style one-field-per-line output.
fn render_format(template: &str, img: &ImageSummary) -> anyhow::Result<String> {
    let (repository, tag) = split_repo_tag(&img.reference);
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    loop {
        match rest.find("{{") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after = &rest[start + 2..];
                let end = after
                    .find("}}")
                    .ok_or_else(|| anyhow::anyhow!("unterminated \"{{{{\" in --format string"))?;
                let field = after[..end].trim().trim_start_matches('.');
                let value: String = match field {
                    "ID" => short_id(&img.digest),
                    "Digest" => img.digest.clone(),
                    "Name" => img.reference.clone(),
                    "Repository" => repository.to_string(),
                    "Tag" => tag.unwrap_or("latest").to_string(),
                    "Size" => human_size(img.size),
                    "Modified" => relative_time(img.modified_at),
                    other => {
                        return Err(anyhow::anyhow!(
                            "unknown --format field \".{}\" (available: .ID, .Digest, .Name, .Repository, .Tag, .Size, .Modified)",
                            other
                        ))
                    }
                };
                out.push_str(&value);
                rest = &after[end + 2..];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `run`'s pipeline for a raw user filter: resolve once, then
    /// match. Tests below feed raw filters, so they exercise the same
    /// resolution `run` applies before calling `matches_filter`.
    fn matches(reference: &str, filter: &str) -> bool {
        let resolved = crate::shortnames::resolve_ollama_api(filter).unwrap();
        matches_filter(reference, &resolved)
    }

    fn sample() -> ImageSummary {
        ImageSummary {
            reference: "unsloth/Qwen3.5-0.8B:latest".into(),
            digest: "sha256:0123456789abcdef0123456789abcdef".into(),
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            size: 1_500_000_000,
            modified_at: None,
        }
    }

    #[test]
    fn split_repo_tag_splits_explicit_tag() {
        assert_eq!(
            split_repo_tag("unsloth/Qwen3.5-0.8B:latest"),
            ("unsloth/Qwen3.5-0.8B", Some("latest"))
        );
        assert_eq!(
            split_repo_tag("unsloth/Qwen3.5-0.8B:q4"),
            ("unsloth/Qwen3.5-0.8B", Some("q4"))
        );
    }

    #[test]
    fn split_repo_tag_returns_none_without_a_tag() {
        assert_eq!(
            split_repo_tag("unsloth/Qwen3.5-0.8B"),
            ("unsloth/Qwen3.5-0.8B", None)
        );
    }

    #[test]
    fn matches_filter_matches_repo_without_tag() {
        // Both sides already fully-qualified: exact-match path.
        assert!(matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "hf.co/unsloth/Qwen3.5-0.8B"
        ));
        assert!(matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "hf.co/unsloth/Qwen3.5-0.8B:latest"
        ));
        assert!(!matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "hf.co/other/model"
        ));
    }

    #[test]
    fn matches_filter_resolves_bare_owner_repo_against_hf_co_default() {
        // Regression test: a model pulled by owner/repo (e.g. from
        // HuggingFace) is stored under an `hf.co/`-prefixed reference; the
        // same bare owner/repo filter must resolve the same way `rm`/`tag`
        // do, or `llmman list <owner>/<repo>` never matches anything.
        assert!(matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "unsloth/Qwen3.5-0.8B"
        ));
        // Must not accidentally match a same-prefix but distinct repo.
        assert!(!matches(
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF:latest",
            "unsloth/Qwen3.5-0.8B"
        ));
        // An explicit tag on the filter requires an exact match, no
        // repo-only fallback.
        assert!(!matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "unsloth/Qwen3.5-0.8B:q4"
        ));
        // Already-qualified filters still work as before.
        assert!(matches(
            "hf.co/unsloth/Qwen3.5-0.8B:latest",
            "hf.co/unsloth/Qwen3.5-0.8B"
        ));
    }

    #[test]
    fn matches_filter_resolves_bare_names_to_docker_ai_default() {
        // Bare (slash-less) names go through resolve_ollama_api's
        // docker.io/ai/ default, same as `rm`/`tag`.
        assert!(matches("docker.io/ai/gemma4:latest", "gemma4"));
    }

    #[test]
    fn render_format_substitutes_known_fields() {
        let img = sample();
        assert_eq!(render_format("{{.ID}}", &img).unwrap(), "0123456789ab");
        assert_eq!(
            render_format("{{.Name}}", &img).unwrap(),
            "unsloth/Qwen3.5-0.8B:latest"
        );
        assert_eq!(
            render_format("{{.Repository}}", &img).unwrap(),
            "unsloth/Qwen3.5-0.8B"
        );
        assert_eq!(render_format("{{.Tag}}", &img).unwrap(), "latest");
        assert_eq!(render_format("{{.Size}}", &img).unwrap(), "1.5 GB");
        assert_eq!(
            render_format("{{ .ID }}: {{.Name}}", &img).unwrap(),
            "0123456789ab: unsloth/Qwen3.5-0.8B:latest"
        );
    }

    #[test]
    fn render_format_rejects_unknown_field() {
        let img = sample();
        assert!(render_format("{{.Bogus}}", &img).is_err());
    }

    /// Telling "free", "cheap" and "unpublished" apart is the point.
    #[test]
    fn price_distinguishes_free_from_unpublished() {
        assert_eq!(price(None), "-");
        assert_eq!(price(Some(0.0)), "0");
        assert_eq!(price(Some(2.5)), "2.5");
        assert_eq!(price(Some(10.0)), "10");
        assert_eq!(price(Some(600.0)), "600");
        assert_eq!(price(Some(1.234)), "1.23");
        // The cheap end keeps its digits instead of rounding to free.
        assert_eq!(price(Some(0.0023)), "0.0023");
        assert_eq!(price(Some(0.01618)), "0.01618");
        assert_eq!(price(Some(0.0000001)), "<0.00001");
    }
}
