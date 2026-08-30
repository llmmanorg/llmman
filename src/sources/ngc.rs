//! NVIDIA NGC (`ngc://org[/team]/model[:version]`).

use anyhow::{Context, Result};
use serde::Deserialize;

use super::Target;

const API_BASE: &str = "https://api.ngc.nvidia.com/v2";

#[derive(Deserialize)]
struct NgcListing {
    #[serde(rename = "modelFiles")]
    #[serde(default)]
    model_files: Vec<NgcFile>,
}

#[derive(Deserialize)]
struct NgcFile {
    name: String,
    #[serde(default)]
    size: i64,
}

/// `reference` is the full `ngc://…` reference, kept verbatim as the
/// stored ref; `rest` is it with the scheme stripped.
pub(crate) async fn pull(reference: &str, rest: &str, target: &Target<'_>) -> Result<()> {
    let api_key = std::env::var("NGC_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .context("NGC_API_KEY environment variable is not set")?;

    let (model_path, version) = match rest.split_once(':') {
        Some((p, v)) => (p, v),
        None => (rest, "latest"),
    };
    // "org/model" or "org/team/model" — NGC's own two URL shapes.
    let segments: Vec<&str> = model_path.split('/').collect();
    if !matches!(segments.len(), 2 | 3) {
        anyhow::bail!("invalid NGC path {model_path:?}: expected org/model or org/team/model");
    }
    let model_base = format!(
        "{API_BASE}/models/{}/versions/{version}",
        segments.join("/")
    );

    if target.report_cached(reference, model_path) {
        return Ok(());
    }

    let api = crate::hf::api_client()?;
    let resp = api
        .get(format!("{model_base}/files"))
        .bearer_auth(&api_key)
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .context("NGC list")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("NGC list: HTTP {}: {body}", status.as_u16());
    }
    let listing: NgcListing = resp.json().await.context("NGC list decode")?;

    let dl = crate::hf::download_client()?;
    let mut packed = Vec::new();
    for f in &listing.model_files {
        if !super::should_pack(&f.name) {
            continue;
        }
        let resp = dl
            .get(format!("{model_base}/files/{}", f.name))
            .bearer_auth(&api_key)
            .send()
            .await
            .with_context(|| format!("NGC download {}", f.name))?
            .error_for_status()
            .with_context(|| format!("NGC download {}", f.name))?;
        packed.push(
            super::download_to_pack_file(
                target,
                "ngc",
                "NGC",
                &f.name,
                f.size,
                resp.bytes_stream(),
            )
            .await?,
        );
    }

    super::pack_as_model_pack(
        target,
        reference,
        model_path,
        packed,
        format!("no model files found in NGC model {model_path}"),
    )
}
