//! HuggingFace Hub support: auth, reference classification, and (in
//! `api`/`client`/`download`/`oci`/`pull`/`transfer`/`progress`) the
//! full `pull`/`transfer` fetch path, run natively and using `hf-xet`
//! directly for Xet-backed files (see `crate::xet_fetch`).
//!
//! [`classify`] below is where every reference is routed, once: to this
//! module, to `crate::sources` (`ms://`, `ngc://`, `s3://`, `gs://`, a
//! local directory), or to the Go shim (`crate::ffi`) — which is now
//! reached only for actual OCI-registry-protocol work (`push`,
//! `inspect`, and registry `pull`/`transfer`), the only part that needs
//! containerd's/podman's Go libraries.
//!
//! Auth (this module): `token_path()` follows `huggingface_hub`'s default
//! chain for the token file (`HF_TOKEN_PATH`, else `HF_HOME`, else
//! `XDG_CACHE_HOME`, else `~/.cache`). Differences remain, among them:
//! llmman does not expand `~`/`$VAR` in these variables, treats an empty
//! value as unset, and has no staging mode (`HUGGINGFACE_CO_STAGING`). A
//! token saved by either tool is picked up by the other when both resolve
//! the same file and nothing shadows it in the reading process (here a
//! nonblank `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`; `huggingface_hub` has
//! further sources, such as `HF_OIDC_RESOURCE`). Resolution order:
//! `HF_TOKEN` env var (fallback `HUGGING_FACE_HUB_TOKEN`), then the token
//! file at `token_path()` (`$HF_HOME/token`, or `$HF_TOKEN_PATH` if set).

pub mod api;
pub mod client;
pub mod download;
pub mod oci;
pub mod progress;
pub mod pull;
pub mod transfer;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Returns true if `server` names a HuggingFace Hub host, so `llmman
/// login`/`llmman logout` route it to token-based auth instead of the OCI
/// registry credential store. `modelscope.cn` is deliberately excluded —
/// it has no bearer-token login flow implemented here — unlike
/// [`is_known_hf_host`], which is about pull/transfer routing instead.
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
/// override the official client accepts.
pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// `$HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else
/// `~/.cache/huggingface`: the same fallback order as `huggingface_hub`'s
/// default (which, unlike this, expands `~`/`$VAR` and does not treat an
/// empty value as unset).
fn hf_home() -> PathBuf {
    hf_home_from(|var| std::env::var_os(var), dirs::home_dir())
}

/// [`hf_home`] with the environment and home directory injected, so the
/// fallback order can be tested without touching the process
/// environment. An empty variable counts as unset, like `xdg_dir` in
/// `cmd::launch`.
fn hf_home_from(
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
    home: Option<PathBuf>,
) -> PathBuf {
    let dir = |var: &str| env(var).filter(|v| !v.is_empty()).map(PathBuf::from);
    dir("HF_HOME").unwrap_or_else(|| {
        dir("XDG_CACHE_HOME")
            .unwrap_or_else(|| home.unwrap_or_else(|| PathBuf::from(".")).join(".cache"))
            .join("huggingface")
    })
}

/// Matches Ollama's own default for `OLLAMA_MAX_TRANSFER_STREAMS`.
const DEFAULT_MAX_TRANSFER_STREAMS: usize = 4;

/// Maximum number of a safetensors repo's files [`pull::pull`] downloads
/// concurrently, from `LLMMAN_MAX_TRANSFER_STREAMS` (mirrors Ollama's
/// `OLLAMA_MAX_TRANSFER_STREAMS`). No effect on GGUF transfers, which
/// stay sequential.
pub fn max_transfer_streams() -> usize {
    parse_max_transfer_streams(std::env::var("LLMMAN_MAX_TRANSFER_STREAMS").ok().as_deref())
}

/// Unset/blank/unparseable falls back to the default. An explicit `0`
/// is clamped to `1` (sequential), not bumped up to the default — same
/// as Ollama's own `max(1, envconfig.MaxTransferStreams())` — since `0`
/// would otherwise deadlock `buffer_unordered` (nothing would poll) and
/// an operator explicitly minimizing concurrency shouldn't get more of
/// it than unset would.
fn parse_max_transfer_streams(value: Option<&str>) -> usize {
    match value.map(str::trim).and_then(|v| v.parse::<usize>().ok()) {
        None => DEFAULT_MAX_TRANSFER_STREAMS,
        Some(n) => n.max(1),
    }
}

/// Path to the active-token file, honoring `HF_TOKEN_PATH`.
pub fn token_path() -> PathBuf {
    match std::env::var_os("HF_TOKEN_PATH").filter(|p| !p.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => hf_home().join("token"),
    }
}

/// Resolve the token to use for HuggingFace requests, in the same order
/// the Python client checks: `HF_TOKEN`, then the legacy
/// `HUGGING_FACE_HUB_TOKEN`, then the on-disk token file written by
/// [`login`].
///
/// Resolved on every call, in the process that makes the request: for
/// `llmman pull` that is the daemon, so the env vars are the daemon's.
/// A `llmman login hf.co` made after it started is picked up only when no
/// nonblank env var is set there and both resolve the same token file.
pub fn token() -> Option<String> {
    resolve_token(
        |var| std::env::var(var).ok(),
        || std::fs::read_to_string(token_path()).ok(),
    )
}

/// [`token`] with the environment and the token file injected, so its
/// order can be tested without touching the process environment.
fn resolve_token(
    env: impl Fn(&str) -> Option<String>,
    file: impl FnOnce() -> Option<String>,
) -> Option<String> {
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = env(var) {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    file()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[derive(Deserialize)]
struct WhoAmI {
    name: Option<String>,
}

/// Validate `token` against `GET {endpoint}/api/whoami-v2` and return the
/// username on success.
pub fn whoami(token: &str) -> Result<String> {
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

/// Validate `token` via [`whoami`], then persist it as the active token.
pub fn login(token: &str) -> Result<String> {
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
    // Create with mode 0600 from the start on unix, not write-then-chmod:
    // the latter leaves the token world-readable for the brief window
    // between the two calls.
    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .and_then(|mut f| {
                // `mode(0o600)` only applies when this call creates the
                // file — an already-existing one (from a previous run,
                // or from huggingface_hub itself) keeps whatever
                // permissions it already had otherwise.
                f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
                f.write_all(token.trim().as_bytes())
            })
            .with_context(|| format!("write {}", path.display()))?;
    }
    #[cfg(not(unix))]
    std::fs::write(&path, token.trim()).with_context(|| format!("write {}", path.display()))?;
    Ok(username)
}

/// Remove the stored active token, if present.
pub fn logout() -> Result<()> {
    let path = token_path();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).context(format!("remove {}", path.display())),
    }
}

// ---------------------------------------------------------------------------
// Host/ref classification — the single decision point for which of
// llmman's three pull/transfer implementations owns a reference. The Go
// shim used to re-derive this for itself on the other side of the FFI
// boundary, at the cost of a second /v2/ probe per pull; go-shim/
// classify.go is what is left of that, and it now only rejects what
// should never have reached it.
// ---------------------------------------------------------------------------

fn is_known_oci_host(host: &str) -> bool {
    matches!(
        host,
        "ghcr.io"
            | "docker.io"
            | "index.docker.io"
            | "registry-1.docker.io"
            | "quay.io"
            | "gcr.io"
            | "mcr.microsoft.com"
            | "public.ecr.aws"
    )
}

/// Unlike [`is_hf_host`] (login/logout routing), this includes
/// `modelscope.cn`: llmman's plain HF-API-shaped pull/transfer path
/// (`pull`/`transfer` modules, endpoint-parameterized) already works
/// against ModelScope's HF-compatible API surface when given as a bare
/// host, no separate implementation needed — only the explicit `ms://`
/// scheme uses ModelScope's own dedicated (non-HF-compatible) API.
pub(crate) fn is_known_hf_host(host: &str) -> bool {
    matches!(host, "hf.co" | "huggingface.co" | "modelscope.cn")
}

/// Probes the OCI Distribution `/v2/` endpoint, trying HTTPS then plain
/// HTTP (a local/insecure registry, like the one used to test this
/// migration, only ever answers on HTTP), and returns true if either
/// response looks like a real OCI registry: the standard
/// `Docker-Distribution-Api-Version` header, or a bare `401` challenge
/// (`WWW-Authenticate`) some registries send without it.
async fn is_oci_registry(host: &str) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    else {
        return false;
    };
    for scheme in ["https", "http"] {
        let Ok(resp) = client
            .get(format!("{scheme}://{host}/v2/"))
            .timeout(Duration::from_secs(3))
            .send()
            .await
        else {
            continue;
        };
        let headers = resp.headers();
        if headers.contains_key("docker-distribution-api-version") {
            return true;
        }
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            && headers.contains_key("www-authenticate")
        {
            return true;
        }
    }
    false
}

/// Reports whether `host` should be treated as an OCI Distribution
/// registry (true) or a HuggingFace-compatible host (false): known-host
/// shortcuts first, then a live `/v2/` probe as the fallback.
///
/// A host `llmman.conf` configures mirrors for counts as known: only an
/// OCI registry has them, and the point of a mirror is to keep pulls
/// working when the registry itself is slow or unreachable — which is
/// exactly when the probe of *it* would fail and misfile the reference
/// as HuggingFace before the mirror ever got a chance.
pub async fn is_oci_host(host: &str) -> bool {
    if is_known_hf_host(host) {
        return false;
    }
    if is_known_oci_host(host) || crate::config::has_registry_mirrors(host) {
        return true;
    }
    is_oci_registry(host).await
}

/// Which of llmman's three pull/transfer implementations owns a given
/// reference.
pub enum ClassifiedRef {
    /// A bare "host/owner/repo[:tag]" ref (`hf://`/`huggingface://`
    /// prefix already stripped, if present) whose host is
    /// HuggingFace-compatible — handled by [`pull`]/[`transfer`].
    Hf(String),
    /// One of the `ms://`/`ngc://`/`s3://`/`gs://`/local-path sources —
    /// handled by [`crate::sources`], verbatim (none of those reference
    /// forms is tag-normalized).
    Source(String),
    /// An actual OCI Distribution registry, normalized (`:latest`
    /// defaulted in) — the only kind still handled by the Go shim,
    /// which takes either spelling since it applies the same rule.
    ///
    /// Not a progress key: the shim tracks byte progress under the
    /// literal string it is handed, so a caller that also polls
    /// `ffi::progress` must pass `ffi::pull` the reference it polls
    /// with. See `cmd::serve`'s `ffi_pull_ref`.
    Other(String),
}

/// Classifies `reference` the way `pullToLayout`'s `classifyPullRef` used
/// to on the Go side — now the single decision point `cmd::pull`/
/// `cmd::resolve`/`cmd::serve`/`cmd::transfer` all use to pick between
/// this module's native path and the Go shim's.
pub async fn classify(reference: &str) -> ClassifiedRef {
    for prefix in ["hf://", "huggingface://"] {
        if let Some(rest) = reference.strip_prefix(prefix) {
            // "hf://owner/repo[:tag]" has no host; parse_hf_ref always
            // needs one.
            let host_less = rest.split('/').filter(|s| !s.is_empty()).count() == 2;
            return ClassifiedRef::Hf(if host_less {
                format!("huggingface.co/{rest}")
            } else {
                rest.to_string()
            });
        }
    }
    // ms://, ngc://, s3://, gs:// and any local path (starting with
    // "/") are now native too — see `crate::sources`, which owns
    // exactly the set `sources::handles` names.
    if crate::sources::handles(reference) {
        return ClassifiedRef::Source(reference.to_string());
    }
    // Any *other* scheme is left verbatim for the Go shim to fail on
    // with its own message, rather than being tag-normalized into
    // something that looks like a registry reference here first.
    if reference.contains("://") || reference.starts_with('/') {
        return ClassifiedRef::Other(reference.to_string());
    }

    let normalized = if reference.rfind(':').unwrap_or(0) <= reference.rfind('/').unwrap_or(0) {
        format!("{reference}:latest")
    } else {
        reference.to_string()
    };
    let host = normalized.split('/').next().unwrap_or(&normalized);
    if !is_oci_host(host).await {
        ClassifiedRef::Hf(normalized)
    } else {
        ClassifiedRef::Other(normalized)
    }
}

/// The HuggingFace API base URL for `host` — mirrors `hfEndpoint`,
/// honoring `MODEL_ENDPOINT`/`HF_ENDPOINT` env var overrides.
pub fn hf_endpoint(host: &str) -> String {
    for var in ["MODEL_ENDPOINT", "HF_ENDPOINT"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return format!("{}/", v.trim_end_matches('/'));
            }
        }
    }
    if host == "hf.co" {
        "https://huggingface.co/".to_string()
    } else {
        format!("https://{host}/")
    }
}

/// The HTTP client used for HuggingFace metadata requests (model info,
/// file listing, HEAD digest probes) — a short total timeout suffices
/// since these responses are small. Shared with `crate::sources`, whose
/// own listing APIs (ModelScope, NGC, GCS) have the same shape.
pub(crate) fn api_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("build HF API client")
}

/// The HTTP client used for actually downloading HuggingFace file
/// content: no *total* deadline, so a large file can take as long as it
/// needs, but `read_timeout` still times out an individual read that
/// stalls, and `connect_timeout` a stalled connection attempt. Shared
/// with `crate::sources` — a ModelScope/NGC/GCS weight file needs the
/// same "no total deadline, but do notice a stalled read" treatment.
pub(crate) fn download_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .build()
        .context("build HF download client")
}

/// The client [`download::head_metadata`] needs: redirects disabled (see
/// its own doc comment for why), shared across a whole pull/transfer
/// rather than rebuilt per file so repeated HEAD requests to the same
/// host can reuse a connection.
pub(crate) fn head_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build no-redirect HTTP client")
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

    #[test]
    fn parse_max_transfer_streams_defaults_to_four_on_unset_or_unparseable() {
        assert_eq!(parse_max_transfer_streams(None), 4);
        assert_eq!(parse_max_transfer_streams(Some("")), 4);
        assert_eq!(parse_max_transfer_streams(Some("garbage")), 4);
        assert_eq!(parse_max_transfer_streams(Some("8")), 8);
        assert_eq!(parse_max_transfer_streams(Some(" 2 ")), 2);
    }

    #[test]
    fn parse_max_transfer_streams_clamps_an_explicit_zero_to_one() {
        // Matches Ollama's own max(1, ...) — an explicit 0 means
        // "minimize concurrency", not "same as unset".
        assert_eq!(parse_max_transfer_streams(Some("0")), 1);
    }

    #[test]
    fn hf_home_follows_huggingface_hub_defaults() {
        let home = || Some(PathBuf::from("/home/u"));
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |var: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == var)
                    .map(|(_, v)| std::ffi::OsString::from(v))
            }
        };
        // (environment, expected HF home)
        let cases: &[(&'static [(&'static str, &'static str)], &str)] = &[
            (&[], "/home/u/.cache/huggingface"),
            (&[("XDG_CACHE_HOME", "/x")], "/x/huggingface"),
            (&[("HF_HOME", "/h"), ("XDG_CACHE_HOME", "/x")], "/h"),
            // Empty counts as unset at each step.
            (
                &[("HF_HOME", ""), ("XDG_CACHE_HOME", "/x")],
                "/x/huggingface",
            ),
            (&[("XDG_CACHE_HOME", "")], "/home/u/.cache/huggingface"),
        ];
        for (pairs, expected) in cases {
            assert_eq!(
                hf_home_from(env(pairs), home()),
                PathBuf::from(expected),
                "env {pairs:?}"
            );
        }
    }

    #[test]
    fn resolve_token_prefers_hf_token_then_legacy_var_then_file() {
        fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
            move |var| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == var)
                    .map(|(_, v)| v.to_string())
            }
        }
        fn file(s: Option<&str>) -> impl FnOnce() -> Option<String> + '_ {
            move || s.map(str::to_string)
        }
        // (environment, token file, expected token)
        type Case<'a> = (&'a [(&'a str, &'a str)], Option<&'a str>, Option<&'a str>);
        let cases: &[Case] = &[
            (&[], None, None),
            (&[], Some("hf_file\n"), Some("hf_file")),
            (&[], Some(" \n"), None),
            (&[("HF_TOKEN", " hf_env ")], Some("hf_file"), Some("hf_env")),
            (
                &[
                    ("HF_TOKEN", "hf_env"),
                    ("HUGGING_FACE_HUB_TOKEN", "hf_legacy"),
                ],
                None,
                Some("hf_env"),
            ),
            (
                &[("HUGGING_FACE_HUB_TOKEN", "hf_legacy")],
                Some("hf_file"),
                Some("hf_legacy"),
            ),
            // A set-but-blank variable falls through rather than
            // shadowing the next source.
            (
                &[("HF_TOKEN", ""), ("HUGGING_FACE_HUB_TOKEN", "hf_legacy")],
                None,
                Some("hf_legacy"),
            ),
            (&[("HF_TOKEN", "  ")], Some("hf_file"), Some("hf_file")),
        ];
        for (pairs, on_disk, expected) in cases {
            assert_eq!(
                resolve_token(env(pairs), file(*on_disk)).as_deref(),
                *expected,
                "env {pairs:?}, file {on_disk:?}"
            );
        }
    }

    #[test]
    fn resolve_token_reads_the_file_only_when_no_env_var_is_set() {
        let read = std::cell::Cell::new(false);
        let token = resolve_token(
            |var| (var == "HF_TOKEN").then(|| "hf_env".to_string()),
            || {
                read.set(true);
                Some("hf_file".to_string())
            },
        );
        assert_eq!(token.as_deref(), Some("hf_env"));
        assert!(
            !read.get(),
            "the token file was read although HF_TOKEN was set"
        );
    }

    #[test]
    fn is_known_hf_host_includes_modelscope() {
        assert!(is_known_hf_host("modelscope.cn"));
        assert!(is_known_hf_host("huggingface.co"));
        assert!(!is_known_hf_host("docker.io"));
    }

    #[test]
    fn is_known_oci_host_matches_the_usual_registries() {
        assert!(is_known_oci_host("docker.io"));
        assert!(is_known_oci_host("ghcr.io"));
        assert!(!is_known_oci_host("huggingface.co"));
    }

    #[tokio::test]
    async fn classify_strips_hf_scheme_prefixes_and_defaults_the_host() {
        match classify("hf://owner/repo:tag").await {
            ClassifiedRef::Hf(r) => assert_eq!(r, "huggingface.co/owner/repo:tag"),
            _ => panic!("expected Hf"),
        }
        match classify("huggingface://owner/repo").await {
            ClassifiedRef::Hf(r) => assert_eq!(r, "huggingface.co/owner/repo"),
            _ => panic!("expected Hf"),
        }
    }

    #[tokio::test]
    async fn classify_leaves_an_explicit_host_after_the_hf_scheme_alone() {
        match classify("hf://modelscope.cn/owner/repo").await {
            ClassifiedRef::Hf(r) => assert_eq!(r, "modelscope.cn/owner/repo"),
            _ => panic!("expected Hf"),
        }
    }

    #[tokio::test]
    async fn classify_routes_every_uri_scheme_source_and_local_path_to_sources() {
        for r in [
            "ms://owner/repo",
            "modelscope://owner/repo",
            "ngc://org/team/model",
            "s3://bucket/key",
            "gs://bucket/key",
            "/abs/path",
        ] {
            match classify(r).await {
                ClassifiedRef::Source(got) => assert_eq!(got, r),
                _ => panic!("expected Source for {r}"),
            }
        }
    }

    /// A scheme nothing implements must reach the Go shim verbatim, so
    /// its error names what the user actually typed — not a `:latest`-
    /// normalized rewrite of it.
    #[tokio::test]
    async fn classify_leaves_an_unknown_scheme_alone() {
        match classify("wat://owner/repo").await {
            ClassifiedRef::Other(got) => assert_eq!(got, "wat://owner/repo"),
            _ => panic!("expected Other"),
        }
    }

    #[tokio::test]
    async fn classify_treats_a_known_registry_host_as_other() {
        match classify("docker.io/library/alpine").await {
            ClassifiedRef::Other(got) => assert_eq!(got, "docker.io/library/alpine:latest"),
            _ => panic!("expected Other"),
        }
    }

    #[tokio::test]
    async fn classify_treats_a_bare_hf_host_as_hf() {
        match classify("huggingface.co/owner/repo").await {
            ClassifiedRef::Hf(got) => assert_eq!(got, "huggingface.co/owner/repo:latest"),
            _ => panic!("expected Hf"),
        }
    }

    #[test]
    fn hf_endpoint_defaults_hf_co_to_the_canonical_huggingface_domain() {
        assert_eq!(hf_endpoint("hf.co"), "https://huggingface.co/");
    }

    #[test]
    fn hf_endpoint_uses_the_host_verbatim_otherwise() {
        assert_eq!(hf_endpoint("modelscope.cn"), "https://modelscope.cn/");
    }
}
