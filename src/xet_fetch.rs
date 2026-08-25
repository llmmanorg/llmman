//! Real Xet-protocol downloads for HuggingFace files, via `hf-xet`
//! (huggingface/xet-core). `huggingface_hub`'s own client refuses to
//! fetch any file over 50GB through a plain HTTP GET at all
//! (`file_download.py`: "Install `hf_xet` ... for xet-powered
//! downloads."), requiring this protocol instead.
//!
//! The Go shim's plain-HTTP path (`go-shim/hf.go`/`transfer_docker.go`)
//! has no such limit and, now that it always sends a `Range` header,
//! works in practice past that threshold too — but it leans on a
//! CloudFront/S3-fronted CAS bridge `huggingface_hub` doesn't trust at
//! that scale, with none of this protocol's real advantages: no
//! chunk-level dedup or resume (a dropped connection at byte 200GB of
//! 244GB restarts the whole file).
//!
//! This module reconstructs a Xet-backed file the way `hf_xet` itself
//! would, via its published Rust crates (`hf-xet` plus the lower-level
//! `xet-client`/`xet-data`/`xet-runtime`, all Apache-2.0, from
//! <https://github.com/huggingface/xet-core>) rather than reimplementing
//! the CAS protocol. It's invoked from a hidden CLI subcommand
//! (`llmman __xet-fetch`, see `cmd::xet_fetch`) rather than linked into
//! the Go shim directly: all of llmman's HF download logic lives in Go
//! today (one coarse FFI call per `transfer`/`pull` — see `ffi.rs`), and
//! `hf-xet` is Rust-only, so self-exec avoids a new Go→Rust call
//! boundary just for this.
//!
//! `xet-client`/`xet-data`/`xet-runtime` are direct dependencies (not
//! just transitive, via `hf-xet`) because the auth plumbing needed here
//! — [`DirectRefreshRouteTokenRefresher`], [`BearerCredentialHelper`],
//! [`XetFileInfo`] — isn't re-exported through `hf-xet`'s public
//! modules. All four are pinned to one xet-core release: they're
//! xet-core's own internal workspace crates, not independently
//! semver-stable APIs, despite being on crates.io.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use xet_client::cas_client::auth::{DirectRefreshRouteTokenRefresher, TokenRefresher};
use xet_client::hub_client::{BearerCredentialHelper, CasJWTInfo};
use xet_data::processing::XetFileInfo;
use xet_runtime::core::XetContext;

/// Identifies and authenticates one Xet-backed HuggingFace file — just
/// the fields the Go shim's own header-parsing already extracts
/// (`X-Xet-Hash`, `X-Linked-Size`, `X-Linked-Etag`) plus the repo/revision
/// it already knows, so wiring Go up to call this needs no extra HTTP call.
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
    /// short-lived Xet CAS token below (never sent to CAS itself).
    /// `None` for an anonymous request.
    pub hf_token: Option<String>,
}

impl XetFileRef {
    /// The Hub API URL that hands out (and later refreshes) a short-lived
    /// Xet CAS access token — mirrors `huggingface_hub`'s
    /// `xet_connection_info_refresh_url`, and matches the resolve
    /// response's own `Link: <...>; rel="xet-auth"` header.
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

/// Downloads `file` to `dest_path` by reconstructing it from Xet CAS
/// chunks. `dest_path`'s parent directory must already exist.
///
/// Installs a [`DirectRefreshRouteTokenRefresher`] so a transfer slow
/// enough to outlive its first CAS token's expiry — plausible at the
/// sizes this exists for — gets a fresh one automatically mid-download.
pub async fn fetch_to_path(file: &XetFileRef, dest_path: &Path) -> Result<()> {
    let ctx = XetContext::default().context("initialize xet runtime context")?;

    let http_client =
        reqwest_middleware::ClientBuilder::new(reqwest_middleware::reqwest::Client::new()).build();
    let cred_helper = file
        .hf_token
        .clone()
        .map(|t| BearerCredentialHelper::new(t, "llmman") as Arc<_>);
    let refresher = Arc::new(DirectRefreshRouteTokenRefresher::new(
        ctx.clone(),
        file.refresh_route(),
        http_client,
        cred_helper,
    ));

    // download_async needs a concrete CAS endpoint up front; this one
    // eager call gets it (cas_url) along with an initial token, so the
    // refresher above only has to handle *later* refreshes.
    let CasJWTInfo {
        cas_url,
        exp,
        access_token,
    } = refresher
        .get_cas_jwt()
        .await
        .context("fetch initial Xet CAS access token")?;

    let xet_file_info = match &file.sha256 {
        Some(sha256) => XetFileInfo::new_with_sha256(file.hash.clone(), file.size, sha256.clone()),
        None => XetFileInfo::new(file.hash.clone(), file.size),
    };

    let dest_path_str = dest_path
        .to_str()
        .context("destination path is not valid UTF-8")?
        .to_string();

    xet::legacy::data_client::download_async(
        &ctx,
        vec![(xet_file_info, dest_path_str)],
        Some(cas_url),
        Some((access_token, exp)),
        Some(refresher as Arc<dyn TokenRefresher>),
        None,
        None,
    )
    .await
    .context("xet download")?;

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

    /// Real download of a small (~450KB), genuinely Xet-backed file —
    /// hf-internal-testing/tiny-random-gpt2's model.safetensors. Needs
    /// real network access (same convention as hf.rs's Go-side
    /// equivalent, hfheadmetadata_test.go). hash/size/sha256 came from:
    ///   curl -sI -H 'Accept-Encoding: identity' <resolve URL>
    #[tokio::test]
    async fn fetch_to_path_downloads_a_real_small_xet_backed_file() {
        let dir = tempfile_dir();
        let dest = dir.join("model.safetensors");

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

        fetch_to_path(&file, &dest).await.expect("fetch_to_path");

        let data = std::fs::read(&dest).expect("read downloaded file");
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
    async fn fetch_to_path_rejects_an_invalid_hash_cleanly() {
        let dir = tempfile_dir();
        let dest = dir.join("out.bin");

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

        let err = fetch_to_path(&file, &dest)
            .await
            .expect_err("an invalid hash must not succeed");
        assert!(
            format!("{err:#}").contains("hash"),
            "expected the error to mention the bad hash, got: {err:#}"
        );
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llmman-xet-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
