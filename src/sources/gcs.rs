//! Google Cloud Storage (`gs://bucket/prefix`), talking to the JSON API
//! directly rather than taking on the `google-cloud-storage` client.

use anyhow::{Context, Result};
use serde::Deserialize;

use super::Target;

const API_BASE: &str = "https://storage.googleapis.com/storage/v1";

#[derive(Deserialize)]
struct GcsPage {
    #[serde(default)]
    items: Vec<GcsObject>,
    #[serde(rename = "nextPageToken")]
    #[serde(default)]
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
struct GcsObject {
    name: String,
    /// GCS reports object sizes as decimal *strings*, not numbers.
    #[serde(default)]
    size: Option<String>,
}

/// `reference` is the full `gs://…` reference (kept verbatim as the
/// stored ref); `rest` is it with the scheme stripped.
pub(crate) async fn pull(reference: &str, rest: &str, target: &Target<'_>) -> Result<()> {
    let (bucket, prefix) = super::split_bucket_prefix(rest, reference)?;

    if target.report_cached(reference, rest) {
        return Ok(());
    }

    let api = crate::hf::api_client()?;
    let token = access_token(&api).await.context("GCS auth")?;

    let mut objects: Vec<GcsObject> = Vec::new();
    let mut page_token: Option<String> = None;
    loop {
        let mut req = api
            .get(format!("{API_BASE}/b/{bucket}/o"))
            .query(&[("prefix", prefix), ("maxResults", "1000")]);
        if let Some(t) = &page_token {
            req = req.query(&[("pageToken", t)]);
        }
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .context("GCS list")?
            .error_for_status()
            .context("GCS list")?;
        let page: GcsPage = resp.json().await.context("GCS list decode")?;
        objects.extend(page.items);
        match page.next_page_token.filter(|t| !t.is_empty()) {
            Some(t) => page_token = Some(t),
            None => break,
        }
    }
    if objects.is_empty() {
        anyhow::bail!("no objects found at gs://{bucket}/{prefix}");
    }

    let dl = crate::hf::download_client()?;
    let mut packed = Vec::new();
    for obj in &objects {
        let Some(rel_path) = super::relative_to_prefix(&obj.name, prefix) else {
            continue;
        };
        if !super::should_pack(rel_path) {
            continue;
        }
        // The object name is one path segment here, so every "/" in it
        // has to be escaped rather than left to split the path.
        let url = format!(
            "{API_BASE}/b/{bucket}/o/{}?alt=media",
            encode_object_name(&obj.name)
        );
        let mut req = dl.get(url);
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("GCS download {}", obj.name))?
            .error_for_status()
            .with_context(|| format!("GCS download {}", obj.name))?;
        let size = obj
            .size
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        packed.push(
            super::download_to_pack_file(target, "gs", "GCS", rel_path, size, resp.bytes_stream())
                .await?,
        );
    }

    super::pack_as_model_pack(
        target,
        reference,
        rest,
        packed,
        format!("no model files found at gs://{bucket}/{prefix}"),
    )
}

/// Percent-encodes an object name as one path segment (RFC 3986's
/// unreserved set). Escaping `/` is what makes `…/o/<name>` address a
/// single object rather than a nested path.
fn encode_object_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// A GCS bearer token, or `None` for anonymous access to a public
/// bucket. `GOOGLE_ACCESS_TOKEN` first, then — only when
/// `GOOGLE_APPLICATION_CREDENTIALS` names a readable service-account
/// file, i.e. the caller expects authenticated access — the GCE metadata
/// server. Gating the metadata probe on that file is what keeps a
/// public-bucket pull off a connect timeout to
/// `metadata.google.internal` on a host that isn't on GCP.
///
/// The service-account private key is deliberately not used to mint a
/// token: that needs an RS256 JWT assertion, which the Go version never
/// implemented either (it read the same file, ignored the key, and fell
/// through to the metadata server exactly as this does).
async fn access_token(client: &reqwest::Client) -> Result<Option<String>> {
    if let Ok(t) = std::env::var("GOOGLE_ACCESS_TOKEN") {
        if !t.is_empty() {
            return Ok(Some(t));
        }
    }
    let Ok(sa_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") else {
        return Ok(None);
    };
    if sa_path.is_empty() {
        return Ok(None);
    }
    // Parsed, not merely read: a malformed credentials file is a
    // configuration error worth reporting, not something to silently
    // fall through to anonymous access on.
    let data = std::fs::read(&sa_path)
        .with_context(|| format!("read GOOGLE_APPLICATION_CREDENTIALS ({sa_path})"))?;
    let sa: serde_json::Value = serde_json::from_slice(&data)
        .with_context(|| format!("parse service account JSON ({sa_path})"))?;
    if sa.get("client_email").and_then(|v| v.as_str()).is_none() {
        anyhow::bail!("service account JSON ({sa_path}) has no \"client_email\"");
    }

    #[derive(Deserialize)]
    struct MetadataToken {
        access_token: String,
    }
    let resp = client
        .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        .header("Metadata-Flavor", "Google")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await;
    if let Ok(resp) = resp {
        if resp.status().is_success() {
            if let Ok(tok) = resp.json::<MetadataToken>().await {
                if !tok.access_token.is_empty() {
                    return Ok(Some(tok.access_token));
                }
            }
        }
    }
    // No metadata server: fall back to anonymous so a public bucket
    // still works. Set GOOGLE_ACCESS_TOKEN for authenticated access
    // off-GCP.
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_object_name_escapes_the_separators_that_would_split_the_path() {
        assert_eq!(
            encode_object_name("models/llama/config.json"),
            "models%2Fllama%2Fconfig.json"
        );
    }

    #[test]
    fn encode_object_name_leaves_the_unreserved_set_alone() {
        assert_eq!(encode_object_name("a-b_c.d~e9"), "a-b_c.d~e9");
    }

    #[test]
    fn encode_object_name_escapes_spaces_and_non_ascii_bytes() {
        assert_eq!(encode_object_name("my model"), "my%20model");
        assert_eq!(encode_object_name("é"), "%C3%A9");
    }
}
