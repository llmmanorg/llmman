//! Real Xet-protocol downloads for HuggingFace files, via `hf-xet`
//! (huggingface/xet-core, Apache-2.0). `huggingface_hub`'s own client
//! refuses to fetch any file over 50GB through a plain HTTP GET at all
//! (`file_download.py`: "Install `hf_xet` ... for xet-powered
//! downloads."), requiring this protocol instead.
//!
//! The Go shim's plain-HTTP path (`go-shim/hf.go`/`transfer_docker.go`)
//! has no such limit and, now that it always sends a `Range` header,
//! works in practice past that threshold too — but it leans on a
//! CloudFront/S3-fronted CAS bridge `huggingface_hub` doesn't trust at
//! that scale, with none of this protocol's real advantages: no
//! chunk-level dedup or resume.
//!
//! This module reconstructs a Xet-backed file the way `hf_xet` itself
//! would and streams it to a writer — no local disk, no full-file
//! buffering — via `hf-xet`'s `xet_session::XetDownloadStreamGroup`
//! streaming API. Called directly from `crate::hf`'s fetch path (a
//! normal function call, no subprocess/FFI, since the caller is
//! already Rust).
//!
//! `hf-xet`'s `xet_session` module is the only dependency needed: it
//! re-exports `XetFileInfo`/`HeaderMap` directly, no need to depend on
//! `xet-client`/`xet-data`/`xet-runtime` separately.

use std::io::Write;

use anyhow::{Context, Result};
use xet::xet_session::{header, HeaderMap, XetFileInfo, XetSessionBuilder};

/// Identifies and authenticates one Xet-backed HuggingFace file — just
/// the fields the Go shim's own header-parsing already extracts
/// (`X-Xet-Hash`, `X-Linked-Size`, `X-Linked-Etag`) plus the repo/revision
/// it already knows.
pub struct XetFileRef {
    /// HuggingFace endpoint, e.g. "https://huggingface.co" (see `hf::endpoint()`).
    pub endpoint: String,
    /// Repo type as used in the Hub API path: "models", "datasets", or "spaces".
    pub repo_type: String,
    /// "owner/repo".
    pub owner_repo: String,
    /// Commit this file's hash/size were resolved against — used only to
    /// build the xet-read-token URL; the download itself is addressed by
    /// content hash alone.
    pub revision: String,
    /// The Xet Merkle hash of the file's content (`X-Xet-Hash`).
    pub hash: String,
    /// The file's real size in bytes (`X-Linked-Size`).
    pub size: u64,
    /// The file's SHA-256, if known (`X-Linked-Etag`, unquoted) — lets
    /// `hf-xet` verify the reconstructed content.
    pub sha256: Option<String>,
    /// Bearer token for the Hub API, used only to fetch/refresh the
    /// short-lived Xet CAS token (never sent to CAS itself). `None` for
    /// an anonymous request.
    pub hf_token: Option<String>,
}

impl XetFileRef {
    /// The Hub API URL that hands out (and, via `with_token_refresh_url`,
    /// later refreshes) a short-lived Xet CAS access token — mirrors
    /// `huggingface_hub`'s `xet_connection_info_refresh_url`, and matches
    /// the resolve response's own `Link: <...>; rel="xet-auth"` header.
    fn refresh_route(&self) -> String {
        format!(
            "{}/api/{}/{}/xet-read-token/{}",
            self.endpoint.trim_end_matches('/'),
            self.repo_type,
            self.owner_repo,
            self.revision
        )
    }
}

/// Streams `file`'s reconstructed content to `w`, chunk by chunk, with no
/// local disk touched and no full-file buffering — suitable for a
/// multi-hundred-GB file. `with_token_refresh_url` handles both the
/// initial CAS token fetch and any later refresh a long-running stream
/// needs, so this needs no upfront token round trip of its own.
pub async fn stream_to_writer(file: &XetFileRef, w: &mut impl Write) -> Result<()> {
    let session = XetSessionBuilder::new()
        .build()
        .context("create xet session")?;

    let mut headers = HeaderMap::new();
    if let Some(token) = &file.hf_token {
        let value = format!("Bearer {token}")
            .parse()
            .context("build Authorization header")?;
        headers.insert(header::AUTHORIZATION, value);
    }

    let group = session
        .new_download_stream_group()
        .context("create xet download stream group")?
        .with_token_refresh_url(file.refresh_route(), headers)
        .build()
        .await
        .context("authenticate xet download")?;

    let xet_file_info = match &file.sha256 {
        Some(sha256) => XetFileInfo::new_with_sha256(file.hash.clone(), file.size, sha256.clone()),
        None => XetFileInfo::new(file.hash.clone(), file.size),
    };

    let mut stream = group
        .download_stream(xet_file_info, None)
        .await
        .context("start xet download stream")?;

    while let Some(chunk) = stream.next().await.context("read xet download stream")? {
        w.write_all(&chunk).context("write downloaded chunk")?;
    }

    Ok(())
}

/// Strips the surrounding quotes an `ETag`/`X-Linked-Etag` value
/// normally carries. A no-op if unquoted.
pub fn strip_etag_quotes(etag: &str) -> String {
    etag.trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_route_matches_huggingface_hub_xet_read_token_url_shape() {
        let file = XetFileRef {
            endpoint: "https://huggingface.co".to_string(),
            repo_type: "models".to_string(),
            owner_repo: "ornith-ai/Ornith-1.5-397B-GGUF".to_string(),
            revision: "771a73943cafcf88d496c423ea5dd3a1622b1c10".to_string(),
            hash: "d684593ed0cf033c7deef1eb122c2ba4302bf77551d8f8481e5ab5174444a642".to_string(),
            size: 244_309_803_808,
            sha256: None,
            hf_token: None,
        };
        assert_eq!(
            file.refresh_route(),
            "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-397B-GGUF/xet-read-token/771a73943cafcf88d496c423ea5dd3a1622b1c10"
        );
    }

    #[test]
    fn refresh_route_tolerates_a_trailing_slash_on_endpoint() {
        let file = XetFileRef {
            endpoint: "https://huggingface.co/".to_string(),
            repo_type: "models".to_string(),
            owner_repo: "owner/repo".to_string(),
            revision: "main".to_string(),
            hash: "h".to_string(),
            size: 1,
            sha256: None,
            hf_token: None,
        };
        assert_eq!(
            file.refresh_route(),
            "https://huggingface.co/api/models/owner/repo/xet-read-token/main"
        );
    }

    #[test]
    fn strip_etag_quotes_removes_surrounding_quotes() {
        assert_eq!(
            strip_etag_quotes("\"c7775e6fae1a47619c199c81b865df9\""),
            "c7775e6fae1a47619c199c81b865df9"
        );
        assert_eq!(strip_etag_quotes("unquoted"), "unquoted");
    }

    /// Real streaming download of a small (~450KB), genuinely Xet-backed
    /// file — hf-internal-testing/tiny-random-gpt2's model.safetensors.
    /// Needs real network access (same convention as hf.rs's Go-side
    /// equivalent, hfheadmetadata_test.go). hash/size/sha256 came from:
    ///   curl -sI -H 'Accept-Encoding: identity' <resolve URL>
    #[tokio::test]
    async fn stream_to_writer_downloads_a_real_small_xet_backed_file() {
        let file = XetFileRef {
            endpoint: crate::hf::endpoint(),
            repo_type: "models".to_string(),
            owner_repo: "hf-internal-testing/tiny-random-gpt2".to_string(),
            revision: "71034c5d8bde858ff824298bdedc65515b97d2b9".to_string(),
            hash: "f8accece953fd366d4ce30597b97acc1ccedc3c785187a5ef6ecb4a8e1755122".to_string(),
            size: 453_864,
            sha256: Some(
                "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500".to_string(),
            ),
            hf_token: crate::hf::token(),
        };

        let mut data = Vec::new();
        stream_to_writer(&file, &mut data)
            .await
            .expect("stream_to_writer");
        assert_eq!(data.len(), 453_864);

        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&data);
        let got_sha256 = hex::encode(hasher.finalize());
        assert_eq!(
            got_sha256, "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500",
            "downloaded content doesn't match the expected sha256"
        );
    }

    /// A malformed hash must fail cleanly, not panic. Uses a real
    /// repo/revision so the xet-read-token exchange succeeds and it's
    /// hf-xet's own local hash validation being tested — still needs
    /// network access for that token exchange.
    #[tokio::test]
    async fn stream_to_writer_rejects_an_invalid_hash_cleanly() {
        let file = XetFileRef {
            endpoint: crate::hf::endpoint(),
            repo_type: "models".to_string(),
            owner_repo: "hf-internal-testing/tiny-random-gpt2".to_string(),
            revision: "71034c5d8bde858ff824298bdedc65515b97d2b9".to_string(),
            hash: "not-a-valid-hex-hash".to_string(),
            size: 1,
            sha256: None,
            hf_token: crate::hf::token(),
        };

        let mut sink = Vec::new();
        let err = stream_to_writer(&file, &mut sink)
            .await
            .expect_err("an invalid hash must not succeed");
        assert!(
            format!("{err:#}").contains("hash"),
            "expected the error to mention the bad hash, got: {err:#}"
        );
    }
}
