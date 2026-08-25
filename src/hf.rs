//! HuggingFace Hub authentication — token storage and validation, following
//! the same on-disk conventions as the official `huggingface_hub` Python
//! client (and its `hf auth login` / `hf auth whoami` CLI), so a token
//! saved by either tool is picked up by the other, and so the Go shim's own
//! HuggingFace pull path (`go-shim/hf.go`'s `hfToken`) can find whatever
//! `llmman login` just wrote without any extra plumbing between the two.
//!
//! Token resolution order (mirrors `huggingface_hub.utils._auth.get_token`):
//!   1. `HF_TOKEN` environment variable (fallback: `HUGGING_FACE_HUB_TOKEN`)
//!   2. the token file at `token_path()`
//!
//! Storage: a plain-text file containing just the token, mode `0600`, at
//! `$HF_HOME/token` (default `~/.cache/huggingface/token`), or
//! `$HF_TOKEN_PATH` if that's set. `HF_HOME` itself defaults to
//! `~/.cache/huggingface` — deliberately not `dirs::cache_dir()`, which on
//! macOS resolves to `~/Library/Caches/huggingface` and would silently miss
//! a token the official `hf` CLI already wrote to the Linux/POSIX-style path
//! `huggingface_hub` itself always uses regardless of platform.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde::Deserialize;

/// Returns true if `server` names a HuggingFace Hub host, so `llmman
/// login`/`llmman logout` route it to token-based auth instead of the OCI
/// registry credential store. Kept in sync with the Go shim's
/// `isKnownHFHost` (minus `modelscope.cn`, which has no bearer-token login
/// flow implemented here).
pub fn is_hf_host(server: &str) -> bool {
    let host = server
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or(server)
        .to_ascii_lowercase();
    matches!(
        host.as_str(),
        "hf.co" | "huggingface.co" | "www.huggingface.co"
    )
}

/// The HuggingFace Hub API base URL, honoring `HF_ENDPOINT` — same
/// override the official client and the Go shim's `hfEndpoint` both accept.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// `$HF_HOME`, defaulting to `~/.cache/huggingface`.
fn hf_home() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
}

/// Path to the active-token file, honoring `HF_TOKEN_PATH`.
pub fn token_path() -> PathBuf {
    if let Ok(p) = std::env::var("HF_TOKEN_PATH") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    hf_home().join("token")
}

/// Resolve the token to use for HuggingFace requests, in the same order the
/// Python client checks: `HF_TOKEN`, then the legacy `HUGGING_FACE_HUB_TOKEN`,
/// then the on-disk token file written by `login()`.
#[allow(dead_code)]
pub fn token() -> Option<String> {
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(t) = std::env::var(var) {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    std::fs::read_to_string(token_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[derive(Deserialize)]
struct WhoAmI {
    name: Option<String>,
}

/// Validate `token` against `GET {endpoint}/api/whoami-v2` — the same
/// endpoint `huggingface_hub.login()` uses to validate a token before
/// saving it — and return the username on success.
pub fn whoami(token: &str) -> anyhow::Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("build http client")?;
    let resp = client
        .get(format!("{}/api/whoami-v2", endpoint()))
        .header("authorization", format!("Bearer {token}"))
        .send()
        .context("request whoami-v2")?;

    let status = resp.status();
    // 401 on whoami is unambiguous (unlike on a repo/resolve endpoint,
    // where huggingface_hub deliberately treats it as "not found" to avoid
    // leaking private-repo existence) — it always means the token itself
    // is invalid.
    if status == reqwest::StatusCode::UNAUTHORIZED {
        anyhow::bail!("invalid user token");
    }
    if !status.is_success() {
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("whoami-v2 returned {status}: {body}");
    }
    let info: WhoAmI = resp.json().context("parse whoami-v2 response")?;
    info.name
        .ok_or_else(|| anyhow!("whoami-v2 response missing \"name\""))
}

/// Validate `token` via [`whoami`], then persist it as the active token —
/// mirroring `huggingface_hub.login()`'s file layout (parent dir `0700`,
/// token file `0600` on Unix; Windows has no equivalent permission bits, so
/// those calls are `cfg`'d out there rather than silently doing nothing).
pub fn login(token: &str) -> anyhow::Result<String> {
    let username = whoami(token)?;
    let path = token_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }
    std::fs::write(&path, token.trim()).with_context(|| format!("write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod {}", path.display()))?;
    }
    Ok(username)
}

/// Remove the stored active token, if present. Unlike `huggingface_hub`,
/// there's no separate `stored_tokens` multi-account file to also clean up
/// here — `llmman login` only ever writes the single active-token file.
pub fn logout() -> anyhow::Result<()> {
    let path = token_path();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).context(format!("remove {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_hf_host_matches_known_hosts_case_insensitively() {
        assert!(is_hf_host("hf.co"));
        assert!(is_hf_host("HuggingFace.co"));
        assert!(is_hf_host("https://huggingface.co"));
        assert!(is_hf_host("www.huggingface.co"));
        assert!(!is_hf_host("registry.example.com"));
        assert!(!is_hf_host("docker.io"));
        assert!(!is_hf_host("modelscope.cn"));
    }
}
