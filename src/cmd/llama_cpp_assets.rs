//! `llmman llama-cpp-assets` - the llama.cpp release assets any supported
//! host would download, so a mirror (see `LLMMAN_LLAMA_CPP_REPO`) can copy
//! exactly those and never repeat llmman's name patterns.

use clap::Args;

use crate::llama_release::{host_assets, AssetListing};

#[derive(Args, Debug)]
pub struct LlamaCppAssetsArgs {
    /// The llama.cpp release to list, resolved as `llmman serve` does
    /// (unset: the release llmman's CI tests against). A mirror has no
    /// `latest`.
    #[arg(long, value_name = "TAG|latest")]
    pub llama_cpp_version: Option<String>,

    /// Print {"repo","release","tag","assets","missing"} as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: &LlamaCppAssetsArgs) -> anyhow::Result<()> {
    let pin = super::serve::runtime::llama_cpp_pin(args.llama_cpp_version.as_deref());
    let listing = host_assets(pin.tag())?;
    for host in &listing.missing {
        eprintln!(
            "[llmman] note: release {} of {} has no asset for {host}",
            listing.release, listing.repo
        );
    }
    if listing.assets.is_empty() {
        anyhow::bail!(
            "release {} of {} has none of the assets llmman selects",
            listing.release,
            listing.repo
        );
    }
    println!("{}", render(&listing, args.json)?);
    Ok(())
}

/// The listing as JSON, or the build tag followed by one asset per line.
fn render(listing: &AssetListing, json: bool) -> anyhow::Result<String> {
    if json {
        return Ok(serde_json::to_string_pretty(listing)?);
    }
    let mut lines = vec![listing.tag.as_str()];
    lines.extend(listing.assets.iter().map(String::as_str));
    Ok(lines.join("\n"))
}
