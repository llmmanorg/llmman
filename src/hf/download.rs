//! Resolving a HuggingFace file's metadata (digest/size/Xet hash) ahead
//! of downloading it, and downloading it — via `hf-xet` when it's
//! Xet-backed (see `crate::xet_fetch`), a plain ranged HTTP GET
//! otherwise. Rust port of go-shim/hf.go's `hfHeadMetadata`/
//! `downloadAttempt`/`downloadHFBlobAttempts`.

use std::io::Write;

use anyhow::{Context, Result};
use reqwest::header::HeaderMap;

use super::client::{self, HttpStatusError};
use crate::xet_fetch::{self, XetFileRef};

/// Everything learned about a HuggingFace file ahead of downloading it,
/// without ever reading its body — mirrors `hfHeadMetadata`'s four
/// return values (minus `ok`, folded into `digest`/`xet_hash` both being
/// `None` for a small non-LFS file the caller should just buffer).
#[derive(Debug, Default, Clone)]
pub struct FileMetadata {
    /// A trustworthy sha256 digest of the real content, as bare lowercase
    /// hex (*not* prefixed with `"sha256:"` — callers building an OCI
    /// [`super::oci::Descriptor`] need to add that themselves; callers
    /// building an [`crate::xet_fetch::XetFileRef`] want it bare, as-is),
    /// if this is an LFS/Xet-tracked file (`X-Linked-Etag`) — `None` for
    /// a small, plain git-blob file, where the ETag is a git blob sha1
    /// instead.
    pub digest: Option<String>,
    pub size: i64,
    /// The Xet Merkle hash (`X-Xet-Hash`), if this file is Xet-backed.
    pub xet_hash: Option<String>,
}

const MAX_HOPS: u32 = 5;

/// Performs a HEAD request against a HuggingFace file's `/resolve/` URL,
/// following only the redirects huggingface.co itself issues (not the
/// final CDN-bound one), one hop at a time, until a response carries
/// `X-Linked-Etag`/`X-Linked-Size`/`X-Xet-Hash` — a *renamed* repository
/// redirects huggingface.co → itself first with neither header, before
/// reaching the real CDN-bound redirect that has them.
///
/// `client` must have redirects *disabled* (see [`super::head_client`])
/// — the default policy would auto-follow straight to the final CDN
/// response, the one hop that never carries these headers.
pub async fn head_metadata(
    client: &reqwest::Client,
    mut url: String,
    token: Option<&str>,
) -> Result<FileMetadata> {
    for _ in 0..MAX_HOPS {
        let mut req = client
            .head(&url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.with_context(|| format!("HEAD {url}"))?;
        let status = resp.status();
        let is_redirect = status.is_redirection();
        if status != reqwest::StatusCode::OK && !is_redirect {
            return Err(HttpStatusError::new(
                format!("HEAD {url}"),
                status.as_u16(),
                resp.headers(),
            )
            .into());
        }

        let headers = resp.headers().clone();
        let x_linked_etag = header_str(&headers, "x-linked-etag");
        let x_linked_size = header_str(&headers, "x-linked-size");
        let x_xet_hash = header_str(&headers, "x-xet-hash");

        if is_redirect && x_linked_etag.is_none() && x_linked_size.is_none() {
            let Some(location) = headers
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
            else {
                anyhow::bail!("HEAD {url}: redirect with no Location header");
            };
            url = reqwest::Url::parse(&url)?.join(location)?.to_string();
            continue;
        }

        let size = x_linked_size
            .as_deref()
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                (status == reqwest::StatusCode::OK)
                    .then(|| header_str(&headers, "content-length"))
                    .flatten()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(0);

        let etag = x_linked_etag.or_else(|| header_str(&headers, "etag"));
        let digest = etag
            .map(|e| e.trim_start_matches("W/").trim_matches('"').to_lowercase())
            .filter(|e| e.len() == 64);

        return Ok(FileMetadata {
            digest,
            size,
            xet_hash: x_xet_hash,
        });
    }
    // Erring here, not returning an empty FileMetadata, matters: a
    // caller reading `digest: None` as "small git-blob file, safe to
    // buffer in memory" would buffer a multi-GB weight file whole if it
    // ever got this far by mistake.
    anyhow::bail!("HEAD {url}: gave up after {MAX_HOPS} redirects with no usable metadata");
}

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
}

/// Everything needed to download one HuggingFace file, and where to send
/// its bytes.
pub struct FetchRequest<'a> {
    pub url: String,
    pub token: Option<&'a str>,
    /// Set when [`head_metadata`] found `X-Xet-Hash` — routes the
    /// download through `hf-xet` instead of a plain HTTP GET.
    pub xet: Option<XetFileRef>,
    pub label: String,
}

/// Downloads one file to `writer` in a single attempt — via `hf-xet` if
/// `req.xet` is set (see `xet_fetch`'s own doc comment for why), a
/// ranged HTTP GET with stall/slow-speed detection otherwise. Returns
/// the total bytes written.
///
/// No internal retry, unlike most of this module: `writer` may be a
/// pipe or anything else that can't be rewound, so a retry needs a
/// *fresh* writer each time — only this function's callers
/// (`pull`/`transfer`) can do that. Retrying here with the same
/// `writer` would corrupt the output instead of starting over.
pub async fn fetch_once(
    client: &reqwest::Client,
    req: &FetchRequest<'_>,
    writer: &mut (impl Write + Send),
) -> Result<u64> {
    if let Some(xet) = &req.xet {
        let mut counting = CountingWriter {
            inner: writer,
            total: 0,
        };
        xet_fetch::stream_to_writer(xet, &mut counting)
            .await
            .context("xet download")?;
        return Ok(counting.total);
    }

    let mut r = client.get(&req.url);
    if let Some(t) = req.token {
        r = r.bearer_auth(t);
    }
    // Always send a Range header, even for this fresh, non-resuming
    // request — some HF CDNs 400 a full-object GET with no Range at all
    // past a few tens of GB (see xet_fetch's own doc comment); the same
    // origin serves "bytes=0-" fine as a 206.
    r = r.header(reqwest::header::RANGE, "bytes=0-");
    let resp = r.send().await.with_context(|| req.label.clone())?;
    let status = resp.status();
    if status != reqwest::StatusCode::OK && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(
            HttpStatusError::new(req.label.clone(), status.as_u16(), resp.headers()).into(),
        );
    }
    client::copy_with_stall_detection(resp.bytes_stream(), |chunk| writer.write_all(chunk)).await
}

/// Whether an error from [`fetch_once`] is worth retrying with a fresh
/// writer, and if so, how long to wait first — combines
/// `client::is_permanent`/`client::retry_after_of` into the one check
/// callers doing their own retry loop (since `fetch_once` itself can't —
/// see its own doc comment) actually need.
pub fn should_retry(err: &anyhow::Error, attempt: u32) -> Option<std::time::Duration> {
    if client::is_permanent(err) {
        return None;
    }
    Some(client::retry_after_of(err).unwrap_or_else(|| client::retry_delay(attempt)))
}

/// Wraps a `Write` to also count bytes written through it.
struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    total: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.total += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for two real bugs found only via a live E2E
    /// transfer, neither of which any pure-unit test would have caught:
    ///
    /// 1. `head_metadata` used to build its request with a client that
    ///    follows redirects automatically, so it silently jumped past
    ///    the huggingface.co hop carrying `X-Linked-Etag`/`X-Xet-Hash`
    ///    straight to the final CDN response (which carries neither) —
    ///    the CDN response's own `etag` header was then misread as the
    ///    digest, and it happened to be the Xet hash's value here, not a
    ///    sha256 at all.
    /// 2. `digest` must come back as *bare* hex, not the OCI-prefixed
    ///    `"sha256:..."` form — callers add that prefix themselves only
    ///    where an OCI descriptor actually needs it.
    ///
    /// Pins both against a small, permanent, real HF file (values
    /// confirmed independently via `curl -I` and `shasum -a 256`).
    #[tokio::test]
    async fn head_metadata_reads_the_intermediate_hop_not_the_final_cdn_response() {
        let url = "https://huggingface.co/hf-internal-testing/tiny-random-gpt2/resolve/main/model.safetensors".to_string();
        let client = super::super::head_client().unwrap();
        let meta = head_metadata(&client, url, None)
            .await
            .expect("head_metadata should succeed against a real, known-good file");
        assert_eq!(meta.digest.as_deref(), Some("8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500"), "digest must be the file's real sha256, bare hex, not the xet hash and not sha256:-prefixed");
        assert_eq!(
            meta.xet_hash.as_deref(),
            Some("f8accece953fd366d4ce30597b97acc1ccedc3c785187a5ef6ecb4a8e1755122"),
            "xet_hash must also be read from the same intermediate hop"
        );
        assert_eq!(meta.size, 453864);
    }
}
