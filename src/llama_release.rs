//! Downloads and caches a prebuilt `llama-server` from
//! [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)'s own
//! GitHub Releases, picking the release asset that matches this host's
//! OS/arch and (per [`crate::hostgpu::detect`]) its best available GPU
//! backend — the same "auto-detect and fetch the matching accelerated
//! runner" step Ollama's installer/runtime does for its own bundled
//! `llama-server` builds (see Ollama's `discover/` package and
//! `scripts/install.sh`'s `check_gpu`), applied here to llmman's
//! PATH-optional `llama-server` dependency instead.
//!
//! This is `cmd::serve`'s `--runtime bin` (see `cmd::serve::runtime`).
//! A release+backend is installed under [`install_root`]`/<tag>/<backend>/`
//! in [`crate::managed`]'s shape and never re-fetched; the release is
//! [`default_release`] unless `--llama-cpp-version` says otherwise.
//!
//! llama.cpp publishes CUDA builds for Linux (x64: CUDA 12 and 13, arm64:
//! 13) as well as Windows. Each ships with a `cudart-*` bundle of the CUDA
//! runtime libraries, extracted next to `llama-server`. A host with no
//! build for its CUDA major (12 on arm64) or whose glibc is older than they
//! need falls back to the CPU build, with a message pointing at `llmman
//! serve --runtime docker` (see `crate::container`).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::fmt::human_size;
use crate::fsutil::sha256_of_file;
use crate::hostgpu::HostGpu;

const API_BASE: &str = "https://api.github.com";

/// The repo the release assets come from unless [`REPO_VAR`] names another.
const DEFAULT_REPO: &str = "ggml-org/llama.cpp";

/// `LLMMAN_LLAMA_CPP_REPO=<owner>/<repo>`: fetch the llama.cpp release
/// assets from that GitHub repo instead of [`DEFAULT_REPO`]. Such a mirror
/// holds many releases, so its release for build `b12359` is tagged
/// `llama.cpp-b12359` ([`release_tag`]), carries the upstream asset names
/// and is always pinned (`latest` is an error).
pub const REPO_VAR: &str = "LLMMAN_LLAMA_CPP_REPO";

/// Prefix of a mirror's release tags.
const MIRROR_TAG_PREFIX: &str = "llama.cpp-";

/// The validated repo from [`REPO_VAR`]'s value (`None` or blank: the
/// default). GitHub names are case-insensitive, so the value is kept as
/// typed.
fn parse_repo(value: Option<&str>) -> Result<String> {
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(DEFAULT_REPO.into());
    };
    let part_ok = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    };
    match value.split_once('/') {
        Some((owner, repo)) if part_ok(owner) && part_ok(repo) => Ok(value.into()),
        _ => anyhow::bail!(
            "{REPO_VAR} must be <owner>/<repo> (letters, digits, '-', '_' and '.'), got {value:?}"
        ),
    }
}

/// The repo to fetch release assets from, per [`REPO_VAR`].
pub fn llama_cpp_repo() -> Result<String> {
    parse_repo(std::env::var(REPO_VAR).ok().as_deref())
}

fn is_mirror(repo: &str) -> bool {
    !repo.eq_ignore_ascii_case(DEFAULT_REPO)
}

/// The GitHub release tag of build tag `build` in `repo`: `build` itself
/// upstream, `llama.cpp-<build>` in a mirror.
pub(crate) fn release_tag(repo: &str, build: &str) -> String {
    if is_mirror(repo) && !build.starts_with(MIRROR_TAG_PREFIX) {
        format!("{MIRROR_TAG_PREFIX}{build}")
    } else {
        build.to_owned()
    }
}

/// The build tag a `release_tag` of `repo` names.
fn strip_release_tag<'a>(repo: &str, tag: &'a str) -> &'a str {
    if is_mirror(repo) {
        tag.strip_prefix(MIRROR_TAG_PREFIX).unwrap_or(tag)
    } else {
        tag
    }
}

/// Mirrors are pinned: there is no `latest` to ask them for.
fn check_pin(repo: &str, pinned_version: Option<&str>) -> Result<()> {
    if is_mirror(repo) && pinned_version.is_none() {
        anyhow::bail!(
            "{REPO_VAR}={repo} is a mirror and mirrors are pinned: `latest` is not available. \
             Pass --llama-cpp-version <build tag> (a mirror release is tagged \
             {MIRROR_TAG_PREFIX}<build tag>) or unset {REPO_VAR}"
        );
    }
    Ok(())
}

/// The release API URL of `repo` at build tag `version`, or its latest.
fn release_url(api_base: &str, repo: &str, version: Option<&str>) -> Result<String> {
    check_pin(repo, version)?;
    Ok(match version {
        Some(v) => format!(
            "{api_base}/repos/{repo}/releases/tags/{}",
            release_tag(repo, v)
        ),
        None => format!("{api_base}/repos/{repo}/releases/latest"),
    })
}

/// The error for a missing mirror release.
fn release_not_found(repo: &str, build: &str) -> anyhow::Error {
    anyhow!(
        "no release {} in GitHub repo {repo} (HTTP 404). {REPO_VAR} selects the repo and a \
         mirror must publish each build as a release tagged {MIRROR_TAG_PREFIX}<build tag>; \
         check the repo, the tag and --llama-cpp-version",
        release_tag(repo, build)
    )
}

/// The default `--llama-cpp-version`: the image tag suffix `crate::container`
/// runs, and (via [`build_tag`]) the release [`ensure_llama_server`]
/// downloads. From the repo-root `LLAMA_CPP_RELEASE` file, which CI also
/// installs and bases the `ai/llmman` images on. The single place to bump.
pub fn default_release() -> &'static str {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/LLAMA_CPP_RELEASE")).trim()
}

/// `--llama-cpp-version` as `cmd::serve` settled it. `Default` is
/// [`default_release`] left implicit, the only pin the container images
/// are pulled by digest for ([`IMAGE_DIGESTS`]); `Explicit` is a tag the
/// user named, even when it spells the default; `Latest` is upstream's
/// floating tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlamaPin {
    Default,
    Explicit(String),
    Latest,
}

impl LlamaPin {
    /// The release tag to run, `None` for `Latest`.
    pub fn tag(&self) -> Option<&str> {
        match self {
            LlamaPin::Default => Some(default_release()),
            LlamaPin::Explicit(tag) => Some(tag),
            LlamaPin::Latest => None,
        }
    }
}

/// The server image digests CI resolved from [`default_release`]'s tags
/// when it built this binary (ci.yml's `llama-digests` job, via the
/// `LLMMAN_LLAMA_IMAGE_DIGESTS` build-time variable): a comma-separated
/// `server[-<backend>]=sha256:<hex>` list, such as
/// `server=sha256:ab..,server-cuda13=sha256:cd..`. Empty when the variable
/// was unset, as in a dev build or `cargo test`.
pub(crate) const IMAGE_DIGESTS: &str = match option_env!("LLMMAN_LLAMA_IMAGE_DIGESTS") {
    Some(list) => list,
    None => "",
};

/// The `sha256:<hex>` digest `list`, in the [`IMAGE_DIGESTS`] format,
/// records for `image_tag` (`server`, `server-cuda13`, ...); `None` when
/// it has no entry for it, in which case the image is pulled by tag
/// alone. An entry without `=` or whose digest is not `sha256:` and 64
/// lowercase hex digits is skipped: a tag-only pull beats an image
/// reference the engine would reject.
pub(crate) fn image_digest_in<'a>(list: &'a str, image_tag: &str) -> Option<&'a str> {
    list.split(',').find_map(|entry| {
        let (tag, digest) = entry.trim().split_once('=')?;
        let hex = digest.strip_prefix("sha256:")?;
        let well_formed =
            hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        (tag == image_tag && well_formed).then_some(digest)
    })
}

/// The release to download for `pin`. An image tag like
/// `v1.0.0-b12359-b1b770d` names the `b<N>` build it was made from (as
/// `packaging/llama-build.sh` does); any other pin, such as `v1.1.0`, is its
/// own release tag, which `resolve_release` follows to its build.
pub(crate) fn build_tag(pin: &str) -> &str {
    pin.strip_prefix('v')
        .and_then(|_| pin.split('-').nth(1))
        .unwrap_or(pin)
}

// ---------------------------------------------------------------------------
// GitHub Releases API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    size: u64,
    /// `sha256:<hex>`. `None` on assets uploaded before GitHub recorded digests.
    digest: Option<String>,
}

pub(crate) fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        // Large Windows CUDA packages bundle the CUDA runtime itself
        // (hundreds of MB) — a short fixed timeout would abort a real,
        // still-progressing download on a slow link.
        .timeout(Duration::from_secs(60 * 30))
        .build()
        .context("build http client")
}

/// Shared GET plumbing for [`fetch_release`] and [`fetch_text`]: sends the
/// request with the `user-agent` header GitHub requires and turns a
/// non-success status into an error prefixed with `what` (the callers'
/// pre-existing messages differ only in that prefix).
fn http_get(
    client: &reqwest::blocking::Client,
    url: &str,
    what: &str,
) -> Result<reqwest::blocking::Response> {
    check_status(http_send(client, url)?, url, what)
}

fn http_send(client: &reqwest::blocking::Client, url: &str) -> Result<reqwest::blocking::Response> {
    client
        .get(url)
        .header("user-agent", "llmman")
        .send()
        .with_context(|| format!("query {url}"))
}

fn check_status(
    resp: reqwest::blocking::Response,
    url: &str,
    what: &str,
) -> Result<reqwest::blocking::Response> {
    if !resp.status().is_success() {
        anyhow::bail!("{what} {url} returned {}", resp.status());
    }
    Ok(resp)
}

/// Fetches release metadata for build tag `version` (e.g. "b10360"), or the
/// latest release if `None` (never in a mirror), from `repo` at `api_base`.
/// The returned `tag_name` is always the build tag, a mirror's prefix removed.
fn fetch_release(
    client: &reqwest::blocking::Client,
    api_base: &str,
    repo: &str,
    version: Option<&str>,
) -> Result<Release> {
    let url = release_url(api_base, repo, version)?;
    let resp = http_send(client, &url)?;
    match version {
        Some(build) if resp.status() == reqwest::StatusCode::NOT_FOUND && is_mirror(repo) => {
            return Err(release_not_found(repo, build));
        }
        _ => {}
    }
    let mut release = check_status(resp, &url, "GitHub API")?
        .json::<Release>()
        .with_context(|| format!("parse release metadata from {url}"))?;
    release.tag_name = strip_release_tag(repo, &release.tag_name).to_owned();
    Ok(release)
}

/// Name of the pointer asset llama.cpp attaches to its stable semver
/// releases (e.g. `v0.3.0`): a one-line text file containing the tag of
/// the promoted `b<number>` build that carries the actual binaries.
const NIGHTLY_POINTER: &str = "nightly-tag.txt";

/// Returns the pointer asset if `release` is one of llama.cpp's semver
/// pointer releases rather than a `b<number>` binary release. The pointer
/// must be the release's sole asset: a release carrying binaries alongside
/// a pointer file is a binary release and must be used as-is, not
/// dereferenced away (or refused) just because the pointer exists.
fn pointer_asset(release: &Release) -> Option<&Asset> {
    if release.assets.len() != 1 {
        return None;
    }
    release.assets.iter().find(|a| a.name == NIGHTLY_POINTER)
}

/// Parses the tag out of a downloaded `nightly-tag.txt`: trims
/// surrounding whitespace (the file ends in a newline) and rejects an
/// empty result.
fn parse_pointer_tag(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Fetches release metadata like [`fetch_release`], then dereferences
/// llama.cpp's new pointer scheme: since the `b<number>` binary releases
/// were all marked prerelease, `releases/latest` resolves to a stable
/// semver release (e.g. `v0.3.0`) whose only asset is [`NIGHTLY_POINTER`],
/// naming the `b<number>` release that actually carries the binaries.
/// One hop only: a pointer chain longer than that is a publishing bug
/// upstream and gets reported rather than followed.
fn resolve_release(
    client: &reqwest::blocking::Client,
    repo: &str,
    version: Option<&str>,
) -> Result<Release> {
    resolve_release_with(
        |v| fetch_release(client, API_BASE, repo, v),
        |url| fetch_text(client, url),
        version,
    )
}

/// The dereference logic behind [`resolve_release`], with the two network
/// fetches injected as closures so tests can drive it in memory.
fn resolve_release_with(
    fetch_release_fn: impl Fn(Option<&str>) -> Result<Release>,
    fetch_text_fn: impl Fn(&str) -> Result<String>,
    version: Option<&str>,
) -> Result<Release> {
    let release = fetch_release_fn(version)?;
    let Some(pointer) = pointer_asset(&release) else {
        return Ok(release);
    };
    let pointer_tag = &release.tag_name;
    let content = fetch_text_fn(&pointer.browser_download_url).with_context(|| {
        format!("download {NIGHTLY_POINTER} from pointer release {pointer_tag}")
    })?;
    let tag = parse_pointer_tag(&content)
        .ok_or_else(|| anyhow!("pointer release {pointer_tag}'s {NIGHTLY_POINTER} is empty"))?;
    let dereferenced = fetch_release_fn(Some(tag)).with_context(|| {
        format!("pointer release {pointer_tag} names tag {tag}, but fetching that release failed")
    })?;
    if pointer_asset(&dereferenced).is_some() {
        anyhow::bail!(
            "pointer release {pointer_tag} names tag {tag}, which is itself \
             only a {NIGHTLY_POINTER} pointer release (refusing to follow a \
             pointer chain)"
        );
    }
    Ok(dereferenced)
}

/// Downloads `url` as plain text, for the small [`NIGHTLY_POINTER`] file.
fn fetch_text(client: &reqwest::blocking::Client, url: &str) -> Result<String> {
    http_get(client, url, "download")?
        .text()
        .with_context(|| format!("read body of {url}"))
}

/// Matches `name` against `pattern`, a substring in which one optional `*`
/// stands for a run of digits (a CUDA minor). Returns those digits as a
/// number, or `0` for a pattern with no `*`.
fn pattern_version(name: &str, pattern: &str) -> Option<u32> {
    let Some((before, after)) = pattern.split_once('*') else {
        return name.contains(pattern).then_some(0);
    };
    name.match_indices(before).find_map(|(at, _)| {
        let rest = &name[at + before.len()..];
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        rest[digits..]
            .starts_with(after)
            .then(|| rest[..digits].parse().ok())
            .flatten()
    })
}

// ---------------------------------------------------------------------------
// Which asset does this host want?
// ---------------------------------------------------------------------------

/// Describes which release asset(s) [`ensure_llama_server`] should fetch
/// for the current host: a required substring identifying the primary
/// package, an optional companion package (only Windows CUDA's separate
/// `cudart-*` runtime-DLL bundle today), and a short label used for the
/// on-disk cache directory name and log messages. Both name patterns follow
/// [`pattern_version`].
#[derive(Debug)]
struct AssetQuery {
    must_contain: String,
    /// May hold `{tag}`, for a companion named after the release (see
    /// [`AssetQuery::companion`]).
    companion_must_contain: Option<String>,
    label: String,
}

impl AssetQuery {
    /// The companion's name pattern in the release `tag`.
    fn companion(&self, tag: &str) -> Option<String> {
        Some(self.companion_must_contain.as_ref()?.replace("{tag}", tag))
    }
}

/// A CUDA companion embeds the primary's own name (`cudart-llama-bin-win-
/// cuda-<ver>-<arch>.zip` holds `-bin-win-cuda-<ver>-<arch>.zip`), so a bare
/// match can pick the runtime-library bundle (no llama-server). The primary
/// is the match that is not the companion, with the highest `*` version
/// (else the first).
fn find_primary_asset<'a>(release: &'a Release, query: &AssetQuery) -> Option<&'a Asset> {
    let companion = query.companion(&release.tag_name);
    release
        .assets
        .iter()
        .filter(|a| {
            companion
                .as_ref()
                .is_none_or(|companion| pattern_version(&a.name, companion).is_none())
        })
        .filter_map(|a| Some((pattern_version(&a.name, &query.must_contain)?, a)))
        .reduce(|best, next| if next.0 > best.0 { next } else { best })
        .map(|(_, a)| a)
}

/// The companion of the same `*` version as `primary`, so runtime libraries
/// never come from another CUDA minor.
fn find_companion_asset<'a>(
    release: &'a Release,
    query: &AssetQuery,
    primary: &Asset,
) -> Option<&'a Asset> {
    let companion = query.companion(&release.tag_name)?;
    let version = pattern_version(&primary.name, &query.must_contain)?;
    let is_companion = |a: &&Asset| pattern_version(&a.name, &companion) == Some(version);
    release.assets.iter().find(is_companion)
}

/// A CUDA build of any minor of CUDA `major` for `os` (`win` or `ubuntu`),
/// and its `cudart-*` companion, which starts `companion` (see
/// [`AssetQuery::companion`]). The dot after `major` keeps `13` from
/// matching `130.x`.
fn cuda_query(major: u32, arch: &str, os: &str, ext: &str, companion: &str) -> AssetQuery {
    let build = format!("-bin-{os}-cuda-{major}.*-{arch}.{ext}");
    AssetQuery {
        companion_must_contain: Some(format!("{companion}{build}")),
        must_contain: build,
        label: format!("cuda-{major}"),
    }
}

/// The glibc the Linux CUDA builds need (`GLIBC_2.38`): they are built on
/// Ubuntu 24.04, where the CPU and Vulkan x64 builds use 22.04's.
const LINUX_CUDA_MIN_GLIBC: (u32, u32) = (2, 38);

/// The CUDA major of the Linux build for a driver of CUDA `major`, or why
/// there is none. llama.cpp builds 12 and 13 for x64 but only 13 for arm64,
/// and a CUDA 13 build cannot run on a CUDA 12 driver.
fn linux_cuda_major(major: u32, arch: &str, glibc: Option<(u32, u32)>) -> Result<u32, String> {
    let cuda = if major >= 13 { 13 } else { 12 };
    if !matches!((arch, cuda), ("x64", _) | ("arm64", 13)) {
        return Err(format!(
            "llama.cpp publishes no prebuilt {arch} Linux CUDA {major} binary"
        ));
    }
    if let Some((major, minor)) = glibc.filter(|have| *have < LINUX_CUDA_MIN_GLIBC) {
        let (need_major, need_minor) = LINUX_CUDA_MIN_GLIBC;
        return Err(format!(
            "its prebuilt Linux CUDA binary needs glibc {need_major}.{need_minor} and \
             this host has {major}.{minor}"
        ));
    }
    Ok(cuda)
}

/// glibc's `(major, minor)` from its `2.39` version string.
#[cfg(any(target_os = "linux", test))]
fn parse_glibc(version: &str) -> Option<(u32, u32)> {
    let (major, rest) = version.split_once('.')?;
    Some((major.parse().ok()?, rest.split('.').next()?.parse().ok()?))
}

/// The host's glibc, or `None` without one (musl).
#[cfg(target_os = "linux")]
fn host_glibc() -> Option<(u32, u32)> {
    #[cfg(target_env = "gnu")]
    {
        // SAFETY: a static NUL-terminated string.
        let version = unsafe { std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()) };
        parse_glibc(version.to_str().ok()?)
    }
    #[cfg(not(target_env = "gnu"))]
    None
}

#[cfg(not(target_os = "linux"))]
fn host_glibc() -> Option<(u32, u32)> {
    None
}

fn host_arch_token() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "s390x" => "s390x",
        other => other,
    }
}

/// The OS of a [`HostDescriptor`], independent of the OS llmman runs on.
/// Every other OS uses the Linux (Ubuntu) builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostOs {
    Macos,
    Linux,
    Windows,
    Android,
}

impl HostOs {
    fn current() -> HostOs {
        if cfg!(target_os = "macos") {
            HostOs::Macos
        } else if cfg!(target_os = "windows") {
            HostOs::Windows
        } else if cfg!(target_os = "android") {
            HostOs::Android
        } else {
            HostOs::Linux
        }
    }

    fn name(self) -> &'static str {
        match self {
            HostOs::Macos => "macos",
            HostOs::Linux => "linux",
            HostOs::Windows => "windows",
            HostOs::Android => "android",
        }
    }
}

/// The accelerator a build targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Accel {
    Cpu,
    Metal,
    Vulkan,
    Rocm,
    Opencl,
    /// A CUDA major, 12 or 13.
    Cuda(u32),
}

impl Accel {
    fn name(self) -> String {
        match self {
            Accel::Cpu => "cpu".into(),
            Accel::Metal => "metal".into(),
            Accel::Vulkan => "vulkan".into(),
            Accel::Rocm => "rocm".into(),
            Accel::Opencl => "opencl".into(),
            Accel::Cuda(major) => format!("cuda-{major}"),
        }
    }
}

/// One OS, architecture and accelerator llmman can select a build for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostDescriptor {
    os: HostOs,
    arch: &'static str,
    accel: Accel,
}

const fn host(os: HostOs, arch: &'static str, accel: Accel) -> HostDescriptor {
    HostDescriptor { os, arch, accel }
}

/// Every host [`asset_query`] can select for, as published by llama.cpp's
/// release workflow (`.github/workflows/release.yml`): ROCm is x64-only,
/// CUDA 12 is x64-only, OpenCL (Adreno) is Windows arm64-only, s390x is
/// Linux CPU-only. Android is
/// not listed: it is not a host the mirrored builds are for.
const HOSTS: &[HostDescriptor] = &[
    host(HostOs::Macos, "arm64", Accel::Metal),
    host(HostOs::Macos, "x64", Accel::Cpu),
    host(HostOs::Linux, "x64", Accel::Cpu),
    host(HostOs::Linux, "arm64", Accel::Cpu),
    host(HostOs::Linux, "s390x", Accel::Cpu),
    host(HostOs::Linux, "x64", Accel::Vulkan),
    host(HostOs::Linux, "arm64", Accel::Vulkan),
    host(HostOs::Linux, "x64", Accel::Rocm),
    host(HostOs::Linux, "x64", Accel::Cuda(12)),
    host(HostOs::Linux, "x64", Accel::Cuda(13)),
    host(HostOs::Linux, "arm64", Accel::Cuda(13)),
    host(HostOs::Windows, "x64", Accel::Cpu),
    host(HostOs::Windows, "arm64", Accel::Cpu),
    host(HostOs::Windows, "x64", Accel::Vulkan),
    host(HostOs::Windows, "arm64", Accel::Vulkan),
    host(HostOs::Windows, "x64", Accel::Cuda(12)),
    host(HostOs::Windows, "x64", Accel::Cuda(13)),
    host(HostOs::Windows, "arm64", Accel::Cuda(13)),
    host(HostOs::Windows, "x64", Accel::Rocm),
    host(HostOs::Windows, "arm64", Accel::Opencl),
];

impl HostDescriptor {
    /// `linux-x64-cuda-13`, for messages and the `missing` list.
    fn name(&self) -> String {
        format!("{}-{}-{}", self.os.name(), self.arch, self.accel.name())
    }

    /// The one table of name patterns, see
    /// <https://github.com/ggml-org/llama.cpp/releases>. `None` for an
    /// OS and accelerator no build exists for.
    fn query(&self) -> Option<AssetQuery> {
        let arch = self.arch;
        let plain = |must_contain: String| AssetQuery {
            must_contain,
            companion_must_contain: None,
            label: self.accel.name(),
        };
        Some(match (self.os, self.accel) {
            (HostOs::Macos, Accel::Metal | Accel::Cpu) => {
                plain(format!("-bin-macos-{arch}.tar.gz"))
            }
            (HostOs::Linux, Accel::Cpu) => plain(format!("-bin-ubuntu-{arch}.tar.gz")),
            (HostOs::Linux, Accel::Vulkan) => plain(format!("-bin-ubuntu-vulkan-{arch}.tar.gz")),
            (HostOs::Linux, Accel::Rocm) => plain("-bin-ubuntu-rocm-".into()),
            (HostOs::Linux, Accel::Cuda(major)) => {
                cuda_query(major, arch, "ubuntu", "tar.gz", "cudart-llama-{tag}")
            }
            (HostOs::Windows, Accel::Cpu) => plain(format!("-bin-win-cpu-{arch}.zip")),
            (HostOs::Windows, Accel::Vulkan) => plain(format!("-bin-win-vulkan-{arch}.zip")),
            (HostOs::Windows, Accel::Rocm) => plain("-bin-win-rocm-".into()),
            (HostOs::Windows, Accel::Opencl) => plain(format!("-bin-win-opencl-adreno-{arch}.zip")),
            (HostOs::Windows, Accel::Cuda(major)) => {
                cuda_query(major, arch, "win", "zip", "cudart-llama")
            }
            // One Android build (NDK, arm64, CPU with runtime dispatch across
            // armv8.x/armv9 ggml-cpu variants). Its shared objects carry no
            // RUNPATH, so spawn_llama_server sets LD_LIBRARY_PATH for it.
            (HostOs::Android, Accel::Cpu) => plain(format!("-bin-android-{arch}.tar.gz")),
            _ => return None,
        })
    }
}

/// The build for a host of `os` and `arch` whose best accelerator is `gpu`
/// (and glibc `glibc`, on Linux), and, when a CUDA build was wanted but
/// there is none, why the CPU build is used instead.
fn select_host(
    os: HostOs,
    arch: &'static str,
    gpu: HostGpu,
    glibc: Option<(u32, u32)>,
) -> (HostDescriptor, Option<String>) {
    let mut why = None;
    let accel = match (os, gpu) {
        (HostOs::Macos, _) if arch == "arm64" => Accel::Metal,
        (HostOs::Linux | HostOs::Windows, HostGpu::Vulkan) => Accel::Vulkan,
        (HostOs::Linux | HostOs::Windows, HostGpu::Rocm) if arch == "x64" => Accel::Rocm,
        (HostOs::Windows, HostGpu::Opencl) if arch == "arm64" => Accel::Opencl,
        // x64 has CUDA 12 and 13 builds, arm64 only 13. Select by major: the
        // minor changes between releases.
        (HostOs::Windows, HostGpu::Cuda { major }) => {
            Accel::Cuda(if arch == "arm64" || major >= 13 {
                13
            } else {
                12
            })
        }
        (HostOs::Linux, HostGpu::Cuda { major }) => match linux_cuda_major(major, arch, glibc) {
            Ok(cuda) => Accel::Cuda(cuda),
            Err(e) => {
                why = Some(e);
                Accel::Cpu
            }
        },
        _ => Accel::Cpu,
    };
    (host(os, arch, accel), why)
}

/// Which release asset(s) this host wants: [`select_host`] for the
/// detected GPU, then [`HostDescriptor::query`].
fn asset_query() -> AssetQuery {
    let os = HostOs::current();
    let gpu = match os {
        HostOs::Linux | HostOs::Windows => crate::hostgpu::detect(),
        HostOs::Macos | HostOs::Android => HostGpu::None,
    };
    let (descriptor, why) = select_host(os, host_arch_token(), gpu, host_glibc());
    if let (Some(why), HostGpu::Cuda { major }) = (why, gpu) {
        eprintln!(
            "[llmman] NVIDIA GPU detected (CUDA {major}), but {why}: using the CPU build. \
             `llmman serve --runtime docker` (or podman) has GPU acceleration, or put a CUDA \
             llama-server on PATH."
        );
    }
    descriptor
        .query()
        .expect("select_host only returns hosts that have a build")
}

// ---------------------------------------------------------------------------
// Every asset any host could select
// ---------------------------------------------------------------------------

/// The assets of one release that some [`HOSTS`] entry selects, for
/// `llmman llama-cpp-assets`.
#[derive(Debug, PartialEq, Eq, serde::Serialize)]
pub struct AssetListing {
    /// The GitHub repo, as [`REPO_VAR`] selects it.
    pub repo: String,
    /// The release's GitHub tag in `repo` (`llama.cpp-b12359` in a mirror).
    pub release: String,
    /// The llama.cpp build tag (`b12359`).
    pub tag: String,
    /// Primary and companion asset names, sorted, without duplicates.
    pub assets: Vec<String>,
    /// Hosts (`linux-arm64-vulkan`) with no matching primary asset, and
    /// `<host>/companion` for a primary whose companion is absent.
    pub missing: Vec<String>,
}

/// What [`HOSTS`] select from `release` of `repo`. A host with no match is
/// listed in `missing`, not an error: upstream does not publish every build
/// in every release.
fn list_assets(repo: &str, release: &Release) -> AssetListing {
    let mut assets = std::collections::BTreeSet::new();
    let mut missing = Vec::new();
    for host in HOSTS {
        let query = host.query().expect("every listed host has a build");
        let Some(primary) = find_primary_asset(release, &query) else {
            missing.push(host.name());
            continue;
        };
        assets.insert(primary.name.clone());
        if query.companion(&release.tag_name).is_some() {
            match find_companion_asset(release, &query, primary) {
                Some(companion) => {
                    assets.insert(companion.name.clone());
                }
                None => missing.push(format!("{}/companion", host.name())),
            }
        }
    }
    AssetListing {
        repo: repo.to_owned(),
        release: release_tag(repo, &release.tag_name),
        tag: release.tag_name.clone(),
        assets: assets.into_iter().collect(),
        missing,
    }
}

/// The [`AssetListing`] of the release `pinned_version` (as `llmman serve`
/// resolves it; `None` is upstream's latest) names in the repo
/// [`REPO_VAR`] selects. Blocking (network).
pub fn host_assets(pinned_version: Option<&str>) -> Result<AssetListing> {
    let repo = llama_cpp_repo()?;
    let pinned_version = pinned_version.map(|v| strip_release_tag(&repo, build_tag(v)));
    check_pin(&repo, pinned_version)?;
    let release = resolve_release(&http_client()?, &repo, pinned_version)?;
    Ok(list_assets(&repo, &release))
}

// ---------------------------------------------------------------------------
// On-disk cache layout
// ---------------------------------------------------------------------------

/// `<data_root>/llama.cpp` — the project, not one binary: the archive
/// carries every llama.cpp tool and the libraries `mediagen` loads.
fn install_root() -> Result<PathBuf> {
    Ok(crate::data_root()?.join("llama.cpp"))
}

/// `<key>/<label>`: one archive per GPU backend per release.
fn install_dir(key: &str, label: &str) -> Result<PathBuf> {
    Ok(install_root()?.join(key).join(label))
}

/// The directory name of build `tag` from `repo`: the tag itself upstream,
/// `<owner>-<repo>-<hash>-<tag>` for a mirror, so a mirror's (signed)
/// binaries never share an install with upstream's or another mirror's.
/// The hash is of the lowercased `owner/repo`: joining the names with `-`
/// alone would map `a-b/c` and `a/b-c` to one directory.
fn cache_key(repo: &str, tag: &str) -> String {
    if is_mirror(repo) {
        use sha2::{Digest, Sha256};
        let repo = repo.to_ascii_lowercase();
        let hash = hex::encode(&Sha256::digest(repo.as_bytes())[..6]);
        format!("{}-{hash}-{tag}", repo.replace('/', "-"))
    } else {
        tag.to_owned()
    }
}

/// Staging directory for a downloaded release archive, before it's
/// extracted into [`install_dir`]. From `LLMMAN_TMPDIR` (mirrors Ollama's
/// `OLLAMA_TMPDIR`) or else a `tmp` subdirectory of [`install_root`].
fn tmp_dir() -> Result<PathBuf> {
    if let Some(dir) = tmp_dir_from_env() {
        return Ok(dir);
    }
    Ok(install_root()?.join("tmp"))
}

fn tmp_dir_from_env() -> Option<PathBuf> {
    parse_tmp_dir(std::env::var("LLMMAN_TMPDIR").ok().as_deref())
}

/// Split out from [`tmp_dir_from_env`] for testing without touching the
/// real environment. Blank values count as unset.
fn parse_tmp_dir(value: Option<&str>) -> Option<PathBuf> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

/// Includes our own pid in the filename so two `llmman` processes
/// downloading the same asset at once (e.g. two concurrent `--pull-only`
/// runs) never share a staging path.
pub(crate) fn tmp_path(name: &str) -> Result<PathBuf> {
    let dir = tmp_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir.join(format!("{name}.tmp-{}", std::process::id())))
}

/// Removes the staging file on drop, so a failed download or extraction
/// (an early `?` return) doesn't leave the archive behind.
pub(crate) struct RemoveOnDrop<'a>(pub(crate) &'a Path);

impl Drop for RemoveOnDrop<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

/// Recursively searches `dir` for a file literally named `name` — used to
/// find `llama-server`/`llama-server.exe` after extraction without having
/// to hardcode each archive format's own internal layout (Linux/macOS
/// tarballs nest everything under one `llama-<tag>/` directory; Windows
/// zips ship every file flat at the archive root).
pub(crate) fn find_binary(dir: &Path, name: &str) -> Option<PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .find(|e| e.file_type().is_file() && e.file_name().to_str() == Some(name))
        .map(|e| e.path().to_path_buf())
}

// ---------------------------------------------------------------------------
// Download + extract
// ---------------------------------------------------------------------------

/// Minimum gap between "still downloading" log lines — this runs inside
/// `llmman serve` (stdio often redirected to a log file), so it logs
/// plain throttled text instead of redrawing a client-style progress bar.
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(5);

/// How long a `.downloading` marker stays credible without being touched.
/// A live fetch refreshes it every [`PROGRESS_LOG_INTERVAL`] as bytes
/// arrive and every [`MARKER_HEARTBEAT`] regardless, so an older marker
/// belongs to a fetch whose process died without running Drop.
const DOWNLOAD_MARKER_STALE_AFTER: Duration = Duration::from_secs(60);

/// How often [`DownloadMarker::keep_alive_during`] re-touches the marker
/// around work that reports no progress of its own (extraction, a
/// `docker pull`, a `uv pip install`).
const MARKER_HEARTBEAT: Duration = Duration::from_secs(15);

/// The fetch-in-progress marker: a fixed path under [`install_root`] that
/// the fetching daemon (writer) and a client polling it (reader, see
/// `daemon::ensure_server`) derive independently. Shared by every backend
/// fetched at startup: llama.cpp here, `uv`/`mlx-lm` in `crate::mlx_release`.
fn download_marker_path() -> Result<PathBuf> {
    Ok(install_root()?.join(".downloading"))
}

/// Whether some process is currently fetching an inference backend: the
/// marker exists and was touched recently enough to be live.
pub fn download_in_progress() -> bool {
    download_status().is_some()
}

/// [`download_in_progress`], with the fetcher's last
/// [`DownloadMarker::set_status`] line (`""` if it never set one). This
/// is how a client shows what a detached daemon, whose stdio is in a log
/// file, is doing on its first start.
pub fn download_status() -> Option<String> {
    let path = download_marker_path().ok()?;
    marker_status(&path, DOWNLOAD_MARKER_STALE_AFTER)
}

/// Split out from [`download_status`] so tests can drive the path and
/// threshold directly.
fn marker_status(path: &Path, stale_after: Duration) -> Option<String> {
    marker_is_fresh(path, stale_after)
        .then(|| std::fs::read_to_string(path).unwrap_or_default())
        .map(|text| text.trim().to_string())
}

/// Split out from [`download_in_progress`] so tests can drive the path
/// and threshold directly.
fn marker_is_fresh(path: &Path, stale_after: Duration) -> bool {
    let Ok(mtime) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return false;
    };
    // A backwards clock step can leave mtime in the future; that still
    // means "touched just now", so count it as fresh.
    std::time::SystemTime::now()
        .duration_since(mtime)
        .map_or(true, |age| age < stale_after)
}

/// Creates the marker on construction and removes it on drop, so success
/// and every early `?` return both clear it. Best-effort throughout: a
/// marker failure must never fail the fetch itself. The holder must keep
/// it fresh within [`DOWNLOAD_MARKER_STALE_AFTER`]: a download loop does
/// so via [`set_status`] as bytes arrive; anything else runs under
/// [`keep_alive_during`].
///
/// [`set_status`]: DownloadMarker::set_status
/// [`keep_alive_during`]: DownloadMarker::keep_alive_during
pub(crate) struct DownloadMarker {
    path: Option<PathBuf>,
    /// The last status line, so a heartbeat [`touch`](Self::touch)
    /// rewrites it rather than blanking what the client is displaying.
    /// Held across the file write so a heartbeat cannot overwrite a newer
    /// status with an older one.
    status: std::sync::Mutex<String>,
}

impl DownloadMarker {
    pub(crate) fn create() -> DownloadMarker {
        match download_marker_path() {
            Ok(p) => Self::create_at(p),
            Err(_) => Self::at(None),
        }
    }

    /// The path-taking constructor behind [`DownloadMarker::create`], so
    /// tests can run the guard's lifecycle against a temp path.
    pub(crate) fn create_at(path: PathBuf) -> DownloadMarker {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self::at(std::fs::write(&path, b"").ok().map(|_| path))
    }

    fn at(path: Option<PathBuf>) -> DownloadMarker {
        DownloadMarker {
            path,
            status: std::sync::Mutex::new(String::new()),
        }
    }

    /// Refreshes the marker's mtime, keeping the current status text.
    pub(crate) fn touch(&self) {
        if let Ok(s) = self.status.lock() {
            self.write(&s);
        }
    }

    /// [`touch`](Self::touch), leaving `status` (one line of progress, see
    /// [`download_status`]) in the marker for a client to display.
    pub(crate) fn set_status(&self, status: &str) {
        if let Ok(mut s) = self.status.lock() {
            *s = status.to_string();
            self.write(&s);
        }
    }

    /// Runs `work` while a background heartbeat [`touch`](Self::touch)es
    /// the marker every [`MARKER_HEARTBEAT`], for work that reports no
    /// progress of its own and would otherwise let the marker go stale,
    /// at which point a waiting `daemon::ensure_server` may stop the
    /// daemon. `work` may still [`set_status`](Self::set_status).
    pub(crate) fn keep_alive_during<T>(&self, work: impl FnOnce() -> T) -> T {
        self.keep_alive_every(MARKER_HEARTBEAT, work)
    }

    /// [`keep_alive_during`](Self::keep_alive_during) with the interval
    /// exposed, for tests.
    fn keep_alive_every<T>(&self, interval: Duration, work: impl FnOnce() -> T) -> T {
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            scope.spawn(move || loop {
                match done_rx.recv_timeout(interval) {
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => self.touch(),
                    _ => return,
                }
            });
            let out = work();
            drop(done_tx);
            out
        })
    }

    fn write(&self, text: &str) {
        if let Some(path) = &self.path {
            // Windows keeps the old mtime when a write is zero bytes,
            // so set it explicitly.
            let _ = std::fs::write(path, text.as_bytes());
            if let Ok(file) = std::fs::File::options().write(true).open(path) {
                let _ = file.set_modified(std::time::SystemTime::now());
            }
        }
    }
}

impl Drop for DownloadMarker {
    fn drop(&mut self) {
        // With two concurrent holders (pid-suffixed staging paths allow
        // them), the first finisher removes the shared marker; the
        // survivor recreates it within MARKER_HEARTBEAT, well inside the
        // client's idle budget, so that window is accepted.
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Opens `dest` for writing without ever following an existing symlink
/// there — `LLMMAN_TMPDIR` can point at a shared directory, and asset
/// names are predictable, so a planted symlink must not redirect a
/// download onto an arbitrary path. Retries once after unlinking a
/// pre-existing entry (a stale file from an earlier run, or an attacker's
/// symlink — either way, safe to remove and recreate).
pub(crate) fn create_new_file(dest: &Path) -> Result<std::fs::File> {
    let open = || {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dest)
    };
    match open() {
        Ok(f) => Ok(f),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            std::fs::remove_file(dest)
                .with_context(|| format!("remove stale {}", dest.display()))?;
            open().with_context(|| format!("create {}", dest.display()))
        }
        Err(e) => Err(e).with_context(|| format!("create {}", dest.display())),
    }
}

pub(crate) fn download_to_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    label: &str,
    marker: Option<&DownloadMarker>,
) -> Result<()> {
    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("download {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("download {url} returned {}", resp.status());
    }
    let total = resp.content_length().unwrap_or(0);
    let mut file = create_new_file(dest)?;
    let mut buf = [0u8; 1 << 16];
    let mut downloaded = 0u64;
    let mut last_logged = Instant::now();
    loop {
        let n = resp.read(&mut buf).context("read download stream")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write downloaded data")?;
        downloaded += n as u64;
        if last_logged.elapsed() >= PROGRESS_LOG_INTERVAL {
            let progress = download_progress_text(label, downloaded, total);
            if let Some(marker) = marker {
                marker.set_status(&format!("downloading {progress}"));
            }
            if total > 0 {
                eprintln!("[llmman] downloading {progress}");
            }
            last_logged = Instant::now();
        }
    }
    eprintln!("[llmman] downloaded {label}: {}", human_size(downloaded));
    Ok(())
}

/// Fails unless the file at `path` matches the size and digest that the
/// release lists for `asset`. An asset with no digest is checked by size only.
fn check_download(path: &Path, asset: &Asset) -> Result<()> {
    let size = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    if size != asset.size {
        anyhow::bail!(
            "{}: downloaded {size} bytes, release lists {}",
            asset.name,
            asset.size
        );
    }
    let Some(digest) = &asset.digest else {
        return Ok(());
    };
    let expected = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| anyhow!("{}: unsupported digest {digest}", asset.name))?;
    let actual = sha256_of_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        anyhow::bail!(
            "{}: sha256 {actual} does not match published {expected}",
            asset.name
        );
    }
    Ok(())
}

/// One line of download progress — the same text the daemon logs and
/// leaves in the [`DownloadMarker`] for a waiting client to show.
fn download_progress_text(label: &str, downloaded: u64, total: u64) -> String {
    if total > 0 {
        format!(
            "{label}: {} / {} ({}%)",
            human_size(downloaded),
            human_size(total),
            downloaded.saturating_mul(100) / total
        )
    } else {
        format!("{label}: {}", human_size(downloaded))
    }
}

fn open_tar_gz(path: &Path) -> Result<tar::Archive<flate2::read::GzDecoder<std::fs::File>>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    Ok(tar::Archive::new(flate2::read::GzDecoder::new(file)))
}

pub(crate) fn extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    open_tar_gz(archive_path)?
        .unpack(dest)
        .with_context(|| format!("extract {} into {}", archive_path.display(), dest.display()))?;
    clear_quarantine(dest);
    Ok(())
}

/// macOS refuses to load a quarantined dylib ("library load disallowed by
/// system policy"), so drop the flag from what we just unpacked. Best effort.
#[cfg(target_os = "macos")]
fn clear_quarantine(dir: &Path) {
    use std::os::unix::ffi::OsStrExt;
    for entry in walkdir::WalkDir::new(dir).into_iter().flatten() {
        if let Ok(path) = std::ffi::CString::new(entry.path().as_os_str().as_bytes()) {
            // SAFETY: both pointers are valid NUL-terminated strings.
            unsafe {
                libc::removexattr(
                    path.as_ptr(),
                    c"com.apple.quarantine".as_ptr(),
                    libc::XATTR_NOFOLLOW,
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn clear_quarantine(_dir: &Path) {}

fn extract_zip(archive_path: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("open {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("read zip {}", archive_path.display()))?;
    archive
        .extract(dest)
        .with_context(|| format!("extract {} into {}", archive_path.display(), dest.display()))
}

/// `flat` drops a tarball's top-level directory: Linux's `cudart-*` bundle
/// nests its libraries under one named after itself, not the `llama-<tag>/`
/// whose `$ORIGIN` rpath is where they have to be.
fn extract(archive_path: &Path, archive_name: &str, dest: &Path, flat: bool) -> Result<()> {
    if archive_name.ends_with(".zip") {
        extract_zip(archive_path, dest)
    } else if flat {
        extract_tar_gz_flat(archive_path, dest)
    } else {
        extract_tar_gz(archive_path, dest)
    }
}

fn extract_tar_gz_flat(archive_path: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut archive = open_tar_gz(archive_path)?;
    for entry in archive
        .entries()
        .with_context(|| format!("read {}", archive_path.display()))?
    {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let relative: PathBuf = path.components().skip(1).collect();
        if relative.as_os_str().is_empty() {
            continue;
        }
        anyhow::ensure!(
            relative
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_))),
            "{}: unsafe path {}",
            archive_path.display(),
            path.display()
        );
        let target = dest.join(&relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        entry
            .unpack(&target)
            .with_context(|| format!("extract {} into {}", path.display(), dest.display()))?;
    }
    clear_quarantine(dest);
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// A resolved, ready-to-run local `llama-server`.
pub struct Resolved {
    pub bin: PathBuf,
    /// Short backend label (`"cpu"`, `"vulkan"`, `"rocm"`, `"opencl"`,
    /// `"cuda-12"`, `"metal"`), surfaced only for logging.
    pub backend_label: String,
}

/// Ensures a `llama-server` matching this host's OS/arch/GPU backend is
/// present locally, downloading and caching it from llama.cpp's GitHub
/// Releases if it isn't already, and returns its path. `pinned_version`,
/// when given, fetches that exact release tag (e.g. "b10360") instead of
/// whatever is currently latest — once cached under that tag it is never
/// re-fetched, so pinning also makes this fully reproducible across runs.
/// Only a `b<number>` pin gets that guarantee: pinning a semver pointer
/// tag (e.g. "v0.3.0") re-resolves through the network each run, since the
/// cache keys on the `b<number>` tag it dereferences to (see
/// `resolve_release`) and the pointer file is mutable upstream.
///
/// Blocking (network + disk I/O) — callers on an async runtime must run
/// this via `tokio::task::spawn_blocking` (see `cmd::serve::runtime::resolve`).
pub fn ensure_llama_server(pinned_version: Option<&str>) -> Result<Resolved> {
    let resolved = resolve_llama_server(pinned_version)?;
    // Extraction clears quarantine, but a cached install skips extraction.
    if let Some(dir) = install_root()
        .ok()
        .and_then(|root| install_dir_of(&root, &resolved.bin))
    {
        clear_quarantine(&dir);
    }
    Ok(resolved)
}

/// The `<root>/<tag>/<label>` install holding `bin`.
fn install_dir_of(root: &Path, bin: &Path) -> Option<PathBuf> {
    let rel = bin.strip_prefix(root).ok()?;
    Some(root.join(rel.components().take(2).collect::<PathBuf>()))
}

fn resolve_llama_server(pinned_version: Option<&str>) -> Result<Resolved> {
    let repo = llama_cpp_repo()?;
    let pinned_version = pinned_version.map(|v| strip_release_tag(&repo, build_tag(v)));
    // Before any cache lookup: an unpinned mirror must not fall back to
    // whatever build happens to be cached.
    check_pin(&repo, pinned_version)?;
    let query = asset_query();
    let bin_name = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    // If we already fetched this exact (tag, backend) combination before,
    // reuse it without touching the network at all — pinned or not, a
    // given release's assets never change after being published.
    if let Some(tag) = pinned_version {
        let tag_dir = install_root()?.join(cache_key(&repo, tag));
        if let Some(bin) = cached_in_tag(&tag_dir, &query.label, bin_name) {
            return Ok(Resolved {
                bin,
                backend_label: query.label,
            });
        }
    } else {
        // Unpinned: still worth a cheap local check for *some* cached tag
        // before hitting the network, so a fully offline second run of
        // `llmman serve` (no pinned version, no PATH llama-server) still
        // works using whatever was last downloaded.
        if let Some(bin) = newest_cached(&query.label, bin_name)? {
            // Still try to reach GitHub for a possibly newer release below;
            // fall back to this cached copy if that fails.
            match try_ensure_from_network(&repo, pinned_version, &query, bin_name) {
                Ok(resolved) => return Ok(resolved),
                Err(e) => {
                    eprintln!(
                        "[llmman] could not check for a newer llama-server release ({e:#}); \
                         using previously downloaded build"
                    );
                    return Ok(Resolved {
                        bin,
                        backend_label: query.label,
                    });
                }
            }
        }
    }

    try_ensure_from_network(&repo, pinned_version, &query, bin_name)
}

fn try_ensure_from_network(
    repo: &str,
    pinned_version: Option<&str>,
    query: &AssetQuery,
    bin_name: &str,
) -> Result<Resolved> {
    let client = http_client()?;
    let release = resolve_release(&client, repo, pinned_version)?;
    let tag = release.tag_name.clone();
    let dest = install_dir(&cache_key(repo, &tag), &query.label)?;

    let cached = |dest: &Path| crate::managed::completed_binary(dest, bin_name);
    let bin = crate::managed::ensure(&install_root()?, &dest, "llama.cpp", cached, |dest| {
        let asset = find_primary_asset(&release, query)
            .with_context(|| {
                format!(
                    "no {} llama.cpp release asset found in {tag} (looked for a name matching \
                     {:?}; LLMMAN_LLM_LIBRARY=cpu selects the CPU build)",
                    query.label, query.must_contain
                )
            })?
            .clone();
        eprintln!(
            "[llmman] downloading llama-server ({}) {tag}: {}",
            query.label, asset.name
        );
        // One marker for download and extraction; the heartbeat covers
        // extraction, which reports no progress of its own.
        let marker = DownloadMarker::create();
        let fetch = |asset: &Asset, unpack: &dyn Fn(&Path) -> Result<()>| -> Result<()> {
            let tmp = tmp_path(&asset.name)?;
            let _cleanup = RemoveOnDrop(&tmp);
            download_to_file(
                &client,
                &asset.browser_download_url,
                &tmp,
                &asset.name,
                Some(&marker),
            )?;
            check_download(&tmp, asset)?;
            marker.set_status(&format!("extracting {}", asset.name));
            unpack(&tmp)
        };
        let bin = marker.keep_alive_during(|| -> Result<PathBuf> {
            fetch(&asset, &|tmp| extract(tmp, &asset.name, dest, false))?;
            let bin = find_binary(dest, bin_name).with_context(|| {
                format!(
                    "llama-server binary not found after extracting {}",
                    asset.name
                )
            })?;
            if let Some(companion_pattern) = query.companion(&tag) {
                match find_companion_asset(&release, query, &asset) {
                    Some(companion) => {
                        let bin_dir = bin.parent().unwrap_or(dest);
                        eprintln!("[llmman] downloading {}", companion.name);
                        fetch(companion, &|tmp| {
                            extract(tmp, &companion.name, bin_dir, true)
                        })?;
                    }
                    None => eprintln!(
                        "[llmman] warning: expected companion asset matching {companion_pattern:?} \
                         not found in release {tag} — {} may be missing runtime libraries it needs",
                        query.label
                    ),
                }
            }
            Ok(bin)
        })?;
        mark_executable(&bin)?;
        Ok(bin)
    })?;
    Ok(Resolved {
        bin,
        backend_label: query.label.clone(),
    })
}

#[cfg(unix)]
pub(crate) fn mark_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(path)?.permissions();
    perm.set_mode(perm.mode() | 0o111);
    std::fs::set_permissions(path, perm).with_context(|| format!("chmod +x {}", path.display()))
}

#[cfg(not(unix))]
pub(crate) fn mark_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Extracts llama.cpp's numeric build counter from a `b<N>` release tag.
/// Tags don't sort lexically the same as chronologically once the counter
/// crosses a power of ten (`"b9999" > "b10000"` as strings, even though
/// b10000 shipped later), so callers comparing two tags need this instead
/// of a plain string comparison.
fn build_number(tag: &str) -> Option<u64> {
    tag.strip_prefix('b')?.parse().ok()
}

/// A complete install in a tag dir: `<label>`, or a legacy `<label>.<minor>`
/// (`cuda-12.4` for `cuda-12`).
fn cached_in_tag(tag_dir: &Path, label: &str, bin_name: &str) -> Option<PathBuf> {
    if let Some(bin) = crate::managed::completed_binary(&tag_dir.join(label), bin_name) {
        return Some(bin);
    }
    let legacy = format!("{label}.");
    std::fs::read_dir(tag_dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&legacy))
        .find_map(|e| crate::managed::completed_binary(&e.path(), bin_name))
}

/// The newest complete `<tag>/<label>` install under [`install_root`]
/// (normally the only one: `managed::ensure` prunes the rest).
fn newest_cached(label: &str, bin_name: &str) -> Result<Option<PathBuf>> {
    let root = install_root()?;
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Ok(None);
    };
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let tag = entry.file_name().to_string_lossy().into_owned();
        let Some(build) = build_number(&tag) else {
            continue;
        };
        let Some(bin) = cached_in_tag(&entry.path(), label, bin_name) else {
            continue;
        };
        if best.as_ref().map(|(b, _)| build > *b).unwrap_or(true) {
            best = Some((build, bin));
        }
    }
    Ok(best.map(|(_, bin)| bin))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str =
        "sha256:e3032ea0712da96dec820c770626385de4322dd06619fb5829256a504bbc9375";
    const DIGEST_B: &str =
        "sha256:a8cf841eef4e9d6ea28bb4d88eea1004032a2bf0d1302bff7b8d2100e801ed16";

    fn host_query(os: HostOs, arch: &'static str, accel: Accel) -> AssetQuery {
        host(os, arch, accel).query().unwrap()
    }

    fn windows_cuda_query(major: u32, arch: &'static str) -> AssetQuery {
        host_query(HostOs::Windows, arch, Accel::Cuda(major))
    }

    fn windows_vulkan_query(arch: &'static str) -> AssetQuery {
        host_query(HostOs::Windows, arch, Accel::Vulkan)
    }

    fn linux_cuda_query(
        major: u32,
        arch: &'static str,
        glibc: Option<(u32, u32)>,
    ) -> Result<AssetQuery, String> {
        let cuda = linux_cuda_major(major, arch, glibc)?;
        Ok(host_query(HostOs::Linux, arch, Accel::Cuda(cuda)))
    }

    #[test]
    fn image_digest_in_reads_the_entry_for_the_tag() {
        let list = format!("server={DIGEST_A}, server-cuda13={DIGEST_B}");
        assert_eq!(image_digest_in(&list, "server"), Some(DIGEST_A));
        assert_eq!(image_digest_in(&list, "server-cuda13"), Some(DIGEST_B));
        assert_eq!(image_digest_in(&list, "server-cuda"), None);
        assert_eq!(image_digest_in("", "server"), None);
    }

    #[test]
    fn image_digest_in_skips_malformed_entries() {
        let short = "sha256:abc";
        let upper = DIGEST_A.to_uppercase();
        let list = format!("server,server-cuda={short},server-rocm={upper},server-vulkan=,{DIGEST_A},server-cuda13={DIGEST_B}");
        assert_eq!(image_digest_in(&list, "server"), None);
        assert_eq!(image_digest_in(&list, "server-cuda"), None);
        assert_eq!(image_digest_in(&list, "server-rocm"), None);
        assert_eq!(image_digest_in(&list, "server-vulkan"), None);
        assert_eq!(image_digest_in(&list, "server-cuda13"), Some(DIGEST_B));
    }

    #[test]
    fn llama_pin_tag_is_the_release_except_for_latest() {
        assert_eq!(LlamaPin::Default.tag(), Some(default_release()));
        assert_eq!(LlamaPin::Explicit("b9994".into()).tag(), Some("b9994"));
        assert_eq!(LlamaPin::Latest.tag(), None);
    }

    #[test]
    fn parse_tmp_dir_returns_none_when_unset_or_blank() {
        assert_eq!(parse_tmp_dir(None), None);
        assert_eq!(parse_tmp_dir(Some("")), None);
        assert_eq!(parse_tmp_dir(Some("   ")), None);
    }

    #[test]
    fn parse_tmp_dir_trims_and_returns_the_given_path() {
        assert_eq!(
            parse_tmp_dir(Some("  /custom/tmp  ")),
            Some(PathBuf::from("/custom/tmp"))
        );
    }

    #[test]
    #[cfg(unix)]
    fn create_new_file_never_writes_through_a_planted_symlink() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!(
            "llmman-create-new-file-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let victim = dir.join("victim");
        let dest = dir.join("dest");
        std::fs::write(&victim, b"do not touch").unwrap();
        symlink(&victim, &dest).unwrap();

        create_new_file(&dest).unwrap();

        assert_eq!(std::fs::read(&victim).unwrap(), b"do not touch");
        assert!(dest.is_file() && !dest.is_symlink());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn release_assets_parse_size_and_digest() {
        let release: Release = serde_json::from_str(
            r#"{"tag_name": "b1", "assets": [
                {"name": "a.zip", "browser_download_url": "", "size": 3, "digest": "sha256:ab"},
                {"name": "b.zip", "browser_download_url": "", "size": 4, "digest": null}
            ]}"#,
        )
        .unwrap();
        assert_eq!(release.assets[0].size, 3);
        assert_eq!(release.assets[0].digest.as_deref(), Some("sha256:ab"));
        assert_eq!(release.assets[1].size, 4);
        assert_eq!(release.assets[1].digest, None);
    }

    #[test]
    fn install_dir_of_trims_a_binary_to_its_tag_and_label() {
        let root = Path::new("/data/llama.cpp");
        let bin = root
            .join("b9000")
            .join("metal")
            .join("llama-b9000")
            .join("llama-server");
        assert_eq!(
            install_dir_of(root, &bin),
            Some(root.join("b9000").join("metal"))
        );
        assert_eq!(
            install_dir_of(root, Path::new("/usr/bin/llama-server")),
            None
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn clear_quarantine_drops_the_flag_from_nested_files() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a").join("libggml.dylib");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"x").unwrap();
        let path = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let name = c"com.apple.quarantine";
        let value = b"0081;0;x;";
        let set = unsafe {
            libc::setxattr(
                path.as_ptr(),
                name.as_ptr(),
                value.as_ptr().cast(),
                value.len(),
                0,
                0,
            )
        };
        assert_eq!(set, 0);
        clear_quarantine(dir.path());
        let got =
            unsafe { libc::getxattr(path.as_ptr(), name.as_ptr(), std::ptr::null_mut(), 0, 0, 0) };
        assert_eq!(got, -1);
    }

    fn temp_download(name: &str, contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "llmman-check-download-test-{}-{name}",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn listed_asset(size: u64, digest: Option<&str>) -> Asset {
        Asset {
            name: "llama-bin.zip".into(),
            browser_download_url: String::new(),
            size,
            digest: digest.map(String::from),
        }
    }

    fn sha256_hex(contents: &[u8]) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(contents))
    }

    fn check_download_error(path: &Path, asset: &Asset) -> String {
        format!("{:#}", check_download(path, asset).unwrap_err())
    }

    #[test]
    fn check_download_accepts_the_listed_size_and_digest() {
        let path = temp_download("match", b"archive");
        let hex = sha256_hex(b"archive");
        for digest in [
            format!("sha256:{hex}"),
            format!("sha256:{}", hex.to_uppercase()),
        ] {
            check_download(&path, &listed_asset(7, Some(&digest))).unwrap();
        }
        check_download(&path, &listed_asset(7, None)).unwrap();
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn check_download_rejects_a_size_the_release_does_not_list() {
        let path = temp_download("short", b"<BlockList/>");
        assert_eq!(
            check_download_error(&path, &listed_asset(18407457, None)),
            "llama-bin.zip: downloaded 12 bytes, release lists 18407457"
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn check_download_rejects_a_digest_the_release_does_not_list() {
        let path = temp_download("tampered", b"archive");
        let published = sha256_hex(b"another");
        assert_eq!(
            check_download_error(
                &path,
                &listed_asset(7, Some(&format!("sha256:{published}")))
            ),
            format!(
                "llama-bin.zip: sha256 {} does not match published {published}",
                sha256_hex(b"archive")
            )
        );
        assert_eq!(
            check_download_error(&path, &listed_asset(7, Some("sha512:ab"))),
            "llama-bin.zip: unsupported digest sha512:ab"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A release of the pinned llama.cpp build carrying `names`.
    fn pinned_release(names: impl IntoIterator<Item = String>) -> Release {
        Release {
            tag_name: build_tag(default_release()).into(),
            assets: names
                .into_iter()
                .map(|name| Asset {
                    name,
                    browser_download_url: String::new(),
                    size: 0,
                    digest: None,
                })
                .collect(),
        }
    }

    /// The pinned build's `llama-<tag>-bin-<suffix>` asset name.
    fn pinned_asset(suffix: &str) -> String {
        format!("llama-{}-bin-{suffix}", build_tag(default_release()))
    }

    #[test]
    fn a_major_release_pin_downloads_the_build_it_names() {
        assert_eq!(build_tag("v1.0.0-b12359-b1b770d"), "b12359");
        for own in ["b12359", "v1.1.0", "latest"] {
            assert_eq!(build_tag(own), own);
        }
        assert!(build_tag(default_release()).starts_with('b'));
    }

    #[test]
    fn primary_asset_matches_only_the_intended_substring() {
        let [cpu, vulkan, rocm] = [
            "ubuntu-x64.tar.gz",
            "ubuntu-vulkan-x64.tar.gz",
            "ubuntu-rocm-7.14-x64.tar.gz",
        ]
        .map(pinned_asset);
        let release = pinned_release([cpu.clone(), vulkan.clone(), rocm.clone()]);
        let find = |pattern: &str| {
            let query = AssetQuery {
                must_contain: pattern.into(),
                companion_must_contain: None,
                label: String::new(),
            };
            find_primary_asset(&release, &query).map(|a| a.name.clone())
        };
        assert_eq!(find("-bin-ubuntu-x64.tar.gz"), Some(cpu));
        assert_eq!(find("-bin-ubuntu-vulkan-x64.tar.gz"), Some(vulkan));
        assert_eq!(find("-bin-ubuntu-rocm-"), Some(rocm));
        assert_eq!(find("-bin-win-cpu-x64.zip"), None);
    }

    /// Windows CUDA assets per `(version, arch)`, each companion listed
    /// before its primary as llama.cpp does.
    fn cuda_release(builds: &[(&str, &str)]) -> Release {
        pinned_release(builds.iter().flat_map(|(ver, arch)| {
            [
                format!("cudart-llama-bin-win-cuda-{ver}-{arch}.zip"),
                pinned_asset(&format!("win-cuda-{ver}-{arch}.zip")),
            ]
        }))
    }

    /// The primary and companion names `query` selects from `release`.
    fn selected(release: &Release, query: &AssetQuery) -> (Option<String>, Option<String>) {
        let primary = find_primary_asset(release, query);
        let companion = primary.and_then(|p| find_companion_asset(release, query, p));
        (
            primary.map(|a| a.name.clone()),
            companion.map(|a| a.name.clone()),
        )
    }

    #[test]
    fn primary_cuda_asset_is_never_the_cudart_companion() {
        // The companion is listed first, which made a plain substring match
        // download the runtime DLLs and report "llama-server binary not found".
        let query = windows_cuda_query(13, "x64");
        let companion = "cudart-llama-bin-win-cuda-13.5-x64.zip".to_string();
        let primary = pinned_asset("win-cuda-13.5-x64.zip");
        let release = pinned_release([companion.clone(), primary.clone()]);
        assert!(release.assets[0]
            .name
            .contains("-bin-win-cuda-13.5-x64.zip"));
        assert_eq!(find_primary_asset(&release, &query).unwrap().name, primary);
        assert_eq!(
            find_companion_asset(&release, &query, &release.assets[1])
                .unwrap()
                .name,
            companion
        );
    }

    const PUBLISHED_CUDA: [(&str, &str); 3] = [("12.4", "x64"), ("13.5", "x64"), ("13.5", "arm64")];

    #[test]
    fn windows_cuda_arm64_major_13_finds_the_13_5_build_and_companion() {
        let release = cuda_release(&PUBLISHED_CUDA);
        assert_eq!(
            selected(&release, &windows_cuda_query(13, "arm64")),
            (
                Some(pinned_asset("win-cuda-13.5-arm64.zip")),
                Some("cudart-llama-bin-win-cuda-13.5-arm64.zip".into())
            )
        );
    }

    #[test]
    fn windows_cuda_x64_selects_by_major_and_arch() {
        let release = cuda_release(&PUBLISHED_CUDA);
        assert_eq!(
            selected(&release, &windows_cuda_query(12, "x64")),
            (
                Some(pinned_asset("win-cuda-12.4-x64.zip")),
                Some("cudart-llama-bin-win-cuda-12.4-x64.zip".into())
            )
        );
        assert_eq!(
            selected(&release, &windows_cuda_query(13, "x64")),
            (
                Some(pinned_asset("win-cuda-13.5-x64.zip")),
                Some("cudart-llama-bin-win-cuda-13.5-x64.zip".into())
            )
        );
        // No CUDA 12 arm64 build exists, and no other major is substituted.
        assert_eq!(
            selected(&release, &windows_cuda_query(12, "arm64")),
            (None, None)
        );
    }

    #[test]
    fn windows_cuda_requires_the_dot_after_the_major() {
        let release = cuda_release(&[("130.1", "x64"), ("113.5", "x64")]);
        assert_eq!(
            selected(&release, &windows_cuda_query(13, "x64")),
            (None, None)
        );
    }

    #[test]
    fn windows_cuda_picks_the_highest_minor_and_its_companion() {
        // Minors compare as numbers, not text ("13.10" > "13.9"), and the
        // companion is the one of the primary's own minor whatever the order.
        for builds in [
            [("13.5", "x64"), ("13.10", "x64"), ("13.9", "x64")],
            [("13.10", "x64"), ("13.9", "x64"), ("13.5", "x64")],
        ] {
            let release = cuda_release(&builds);
            assert_eq!(
                selected(&release, &windows_cuda_query(13, "x64")),
                (
                    Some(pinned_asset("win-cuda-13.10-x64.zip")),
                    Some("cudart-llama-bin-win-cuda-13.10-x64.zip".into())
                )
            );
        }
    }

    #[test]
    fn windows_cuda_without_a_matching_companion_still_finds_the_primary() {
        let release = pinned_release([pinned_asset("win-cuda-13.5-x64.zip")]);
        assert_eq!(
            selected(&release, &windows_cuda_query(13, "x64")),
            (Some(pinned_asset("win-cuda-13.5-x64.zip")), None)
        );
    }

    #[test]
    fn windows_cuda_label_names_the_major_only() {
        assert_eq!(windows_cuda_query(13, "arm64").label, "cuda-13");
        assert_eq!(windows_cuda_query(12, "x64").label, "cuda-12");
    }

    /// The Linux CUDA assets of b12359, each runtime bundle before its
    /// build, and the CPU build they must not be confused with.
    fn linux_cuda_release() -> Release {
        let tag = build_tag(default_release());
        let builds = [("12.8", "x64"), ("13.5", "arm64"), ("13.5", "x64")];
        let bundles = builds
            .map(|(ver, arch)| format!("cudart-llama-{tag}-bin-ubuntu-cuda-{ver}-{arch}.tar.gz"));
        let primaries =
            builds.map(|(ver, arch)| pinned_asset(&format!("ubuntu-cuda-{ver}-{arch}.tar.gz")));
        pinned_release(
            bundles
                .into_iter()
                .chain(primaries)
                .chain([pinned_asset("ubuntu-x64.tar.gz")]),
        )
    }

    fn linux_cuda(major: u32, arch: &'static str) -> Option<(Option<String>, Option<String>)> {
        let query = linux_cuda_query(major, arch, None).ok()?;
        Some(selected(&linux_cuda_release(), &query))
    }

    #[test]
    fn linux_cuda_selects_each_major_with_its_own_runtime_bundle() {
        let tag = build_tag(default_release());
        for (major, ver, arch) in [
            (12, "12.8", "x64"),
            (13, "13.5", "x64"),
            (14, "13.5", "x64"),
            (13, "13.5", "arm64"),
        ] {
            assert_eq!(
                linux_cuda(major, arch),
                Some((
                    Some(pinned_asset(&format!("ubuntu-cuda-{ver}-{arch}.tar.gz"))),
                    Some(format!(
                        "cudart-llama-{tag}-bin-ubuntu-cuda-{ver}-{arch}.tar.gz"
                    ))
                )),
                "CUDA {major} on {arch}"
            );
        }
        let label = |major| linux_cuda_query(major, "x64", None).unwrap().label;
        assert_eq!((label(12), label(13)), ("cuda-12".into(), "cuda-13".into()));
    }

    #[test]
    fn linux_cuda_never_substitutes_another_major_or_arch() {
        // No CUDA 12 build for arm64, and a CUDA 13 build cannot run on a
        // CUDA 12 driver: such a host gets the CPU build.
        assert_eq!(linux_cuda(12, "arm64"), None);
        assert!(linux_cuda_query(12, "s390x", None).is_err());
        let only_13 = pinned_release([pinned_asset("ubuntu-cuda-13.5-x64.tar.gz")]);
        let query = linux_cuda_query(12, "x64", None).unwrap();
        assert_eq!(selected(&only_13, &query), (None, None));
    }

    /// The CUDA builds need glibc 2.38; the CPU build runs on Ubuntu 22.04's.
    #[test]
    fn linux_cuda_needs_a_new_enough_glibc() {
        let ok = |glibc| linux_cuda_query(13, "x64", glibc).is_ok();
        assert!(ok(None) && ok(Some((2, 38))) && ok(Some((2, 39))) && ok(Some((3, 0))));
        assert!(!ok(Some((2, 35))) && !ok(Some((2, 37))));
        let why = linux_cuda_query(13, "x64", Some((2, 35))).unwrap_err();
        assert!(why.contains("glibc 2.38") && why.contains("2.35"), "{why}");
    }

    #[test]
    fn parse_glibc_reads_the_version_string() {
        assert_eq!(parse_glibc("2.39"), Some((2, 39)));
        assert_eq!(parse_glibc("2.17.1"), Some((2, 17)));
        assert_eq!(parse_glibc("2"), None);
        assert_eq!(parse_glibc("x.y"), None);
    }
    /// The runtime bundle nests its libraries under a directory of its own
    /// name; they must end up beside `llama-server`, flat.
    #[test]
    fn extract_tar_gz_flat_drops_the_top_level_directory() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("cudart.tar.gz");
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::fast(),
        ));
        for (path, body) in [
            ("cudart-llama-b1-bin-ubuntu-cuda-13.5-x64/", None),
            (
                "cudart-llama-b1-bin-ubuntu-cuda-13.5-x64/libcudart.so.13",
                Some(&b"cudart"[..]),
            ),
            (
                "cudart-llama-b1-bin-ubuntu-cuda-13.5-x64/lib/libcublas.so.13",
                Some(&b"cublas"[..]),
            ),
        ] {
            let mut header = tar::Header::new_gnu();
            match body {
                Some(body) => {
                    header.set_size(body.len() as u64);
                    header.set_entry_type(tar::EntryType::Regular);
                }
                None => {
                    header.set_size(0);
                    header.set_entry_type(tar::EntryType::Directory);
                }
            }
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(&mut header, path, body.unwrap_or(&[]))
                .unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();

        let bin_dir = dir.path().join("llama-b1");
        extract_tar_gz_flat(&archive, &bin_dir).unwrap();
        assert_eq!(
            std::fs::read(bin_dir.join("libcudart.so.13")).unwrap(),
            b"cudart"
        );
        assert_eq!(
            std::fs::read(bin_dir.join("lib").join("libcublas.so.13")).unwrap(),
            b"cublas"
        );
        assert!(!bin_dir
            .join("cudart-llama-b1-bin-ubuntu-cuda-13.5-x64")
            .exists());
    }

    #[test]
    fn windows_vulkan_matches_each_arch_and_never_the_other() {
        let release = pinned_release([
            pinned_asset("win-vulkan-x64.zip"),
            pinned_asset("win-vulkan-arm64.zip"),
        ]);
        for arch in ["x64", "arm64"] {
            assert_eq!(
                selected(&release, &windows_vulkan_query(arch)),
                (Some(pinned_asset(&format!("win-vulkan-{arch}.zip"))), None)
            );
        }
        for (arch, other) in [("x64", "arm64"), ("arm64", "x64")] {
            let only_other = pinned_release([pinned_asset(&format!("win-vulkan-{other}.zip"))]);
            assert_eq!(
                selected(&only_other, &windows_vulkan_query(arch)),
                (None, None)
            );
        }
        assert_eq!(windows_vulkan_query("arm64").label, "vulkan");
    }

    #[test]
    fn primary_asset_without_a_companion_matches_the_only_build() {
        let build = pinned_asset("win-vulkan-x64.zip");
        let release = pinned_release([build.clone()]);
        let query = windows_vulkan_query("x64");
        assert_eq!(find_primary_asset(&release, &query).unwrap().name, build);
    }

    #[test]
    fn pointer_asset_detects_a_semver_pointer_release() {
        let pointer_release = Release {
            tag_name: "v0.3.0".into(),
            assets: vec![Asset {
                name: "nightly-tag.txt".into(),
                browser_download_url: String::new(),
                size: 0,
                digest: None,
            }],
        };
        assert_eq!(
            pointer_asset(&pointer_release).map(|a| a.name.as_str()),
            Some("nightly-tag.txt")
        );

        let binary_release = Release {
            tag_name: "b10621".into(),
            assets: vec![Asset {
                name: "llama-b10621-bin-ubuntu-vulkan-x64.tar.gz".into(),
                browser_download_url: String::new(),
                size: 0,
                digest: None,
            }],
        };
        assert_eq!(
            pointer_asset(&binary_release).map(|a| a.name.as_str()),
            None
        );

        // A release carrying binaries alongside a pointer file is a binary
        // release, not a pointer release (see pointer_asset's doc comment).
        let mixed_release = Release {
            tag_name: "b10621".into(),
            assets: vec![
                Asset {
                    name: "nightly-tag.txt".into(),
                    browser_download_url: String::new(),
                    size: 0,
                    digest: None,
                },
                Asset {
                    name: "llama-b10621-bin-ubuntu-vulkan-x64.tar.gz".into(),
                    browser_download_url: String::new(),
                    size: 0,
                    digest: None,
                },
            ],
        };
        assert_eq!(pointer_asset(&mixed_release).map(|a| a.name.as_str()), None);
    }

    #[test]
    fn parse_pointer_tag_trims_and_rejects_blank_content() {
        assert_eq!(parse_pointer_tag("b10621\n"), Some("b10621"));
        assert_eq!(parse_pointer_tag("  b10621  "), Some("b10621"));
        assert_eq!(parse_pointer_tag(""), None);
        assert_eq!(parse_pointer_tag("\n  \n"), None);
    }

    fn pointer_release(tag: &str) -> Release {
        Release {
            tag_name: tag.into(),
            assets: vec![Asset {
                name: NIGHTLY_POINTER.into(),
                browser_download_url: format!("https://example.invalid/{tag}/{NIGHTLY_POINTER}"),
                size: 0,
                digest: None,
            }],
        }
    }

    fn binary_release(tag: &str) -> Release {
        Release {
            tag_name: tag.into(),
            assets: vec![Asset {
                name: format!("llama-{tag}-bin-ubuntu-x64.tar.gz"),
                browser_download_url: String::new(),
                size: 0,
                digest: None,
            }],
        }
    }

    #[test]
    fn resolve_release_with_dereferences_a_pointer_release() {
        let metadata_fetches = std::cell::RefCell::new(Vec::new());
        let release = resolve_release_with(
            |v| {
                metadata_fetches.borrow_mut().push(v.map(str::to_owned));
                Ok(match v {
                    None => pointer_release("v0.3.0"),
                    Some("b10621") => binary_release("b10621"),
                    Some(other) => panic!("unexpected tag fetch: {other}"),
                })
            },
            |_url| Ok("b10621\n".into()),
            None,
        )
        .unwrap();
        assert_eq!(release.tag_name, "b10621");
        assert_eq!(
            *metadata_fetches.borrow(),
            vec![None, Some("b10621".to_owned())]
        );
    }

    #[test]
    fn resolve_release_with_refuses_a_pointer_chain() {
        let err = resolve_release_with(
            |v| {
                Ok(match v {
                    None => pointer_release("v0.3.0"),
                    Some(tag) => pointer_release(tag),
                })
            },
            |_url| Ok("b10621\n".into()),
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("v0.3.0"), "missing pointer tag in: {msg}");
        assert!(msg.contains("b10621"), "missing dereferenced tag in: {msg}");
        assert!(msg.contains("pointer chain"), "unexpected message: {msg}");
    }

    #[test]
    fn resolve_release_with_rejects_a_blank_pointer_file() {
        let err = resolve_release_with(
            |_v| Ok(pointer_release("v0.3.0")),
            |_url| Ok("\n  \n".into()),
            None,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("v0.3.0"), "missing pointer tag in: {msg}");
        assert!(msg.contains("is empty"), "unexpected message: {msg}");
    }

    #[test]
    fn resolve_release_with_contextualizes_a_failed_dereferenced_fetch() {
        let err = resolve_release_with(
            |v| match v {
                None => Ok(pointer_release("v0.3.0")),
                Some(_) => Err(anyhow!("boom")),
            },
            |_url| Ok("b10621\n".into()),
            None,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains(
                "pointer release v0.3.0 names tag b10621, but fetching that release failed"
            ),
            "unexpected message: {msg}"
        );
        assert!(msg.contains("boom"), "missing root cause in: {msg}");
    }

    #[test]
    fn resolve_release_with_passes_a_binary_release_through() {
        let text_fetches = std::cell::Cell::new(0);
        let release = resolve_release_with(
            |_v| Ok(binary_release("b10621")),
            |_url| {
                text_fetches.set(text_fetches.get() + 1);
                Ok(String::new())
            },
            None,
        )
        .unwrap();
        assert_eq!(release.tag_name, "b10621");
        assert_eq!(text_fetches.get(), 0);
    }

    #[test]
    fn resolve_release_with_passes_a_pinned_binary_tag_through() {
        let text_fetches = std::cell::Cell::new(0);
        let release = resolve_release_with(
            |v| {
                assert_eq!(v, Some("b10360"));
                Ok(binary_release("b10360"))
            },
            |_url| {
                text_fetches.set(text_fetches.get() + 1);
                Ok(String::new())
            },
            Some("b10360"),
        )
        .unwrap();
        assert_eq!(release.tag_name, "b10360");
        assert_eq!(text_fetches.get(), 0);
    }

    #[test]
    fn marker_is_fresh_is_false_for_an_absent_file() {
        let path =
            std::env::temp_dir().join(format!("llmman-marker-absent-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(!marker_is_fresh(&path, Duration::from_secs(60)));
    }

    #[test]
    fn marker_is_fresh_is_true_for_a_just_written_file() {
        let path = std::env::temp_dir().join(format!("llmman-marker-fresh-{}", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        let fresh = marker_is_fresh(&path, Duration::from_secs(60));
        let _ = std::fs::remove_file(&path);
        assert!(fresh);
    }

    #[test]
    fn marker_is_fresh_is_false_for_a_stale_mtime() {
        let path = std::env::temp_dir().join(format!("llmman-marker-stale-{}", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() - Duration::from_secs(120))
            .unwrap();
        drop(file);
        let fresh = marker_is_fresh(&path, Duration::from_secs(60));
        let _ = std::fs::remove_file(&path);
        assert!(!fresh);
    }

    /// The guard's whole lifecycle: created on construction, refreshed by
    /// touch, removed on drop. Runs against a temp path via create_at, not
    /// the real install root, so a genuinely live download's marker is
    /// never deleted by the test.
    #[test]
    fn download_marker_creates_touches_and_removes_the_marker() {
        let path = std::env::temp_dir().join(format!(
            "llmman-marker-lifecycle-{}/.downloading",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let marker = DownloadMarker::create_at(path.clone());
        assert!(path.is_file(), "marker not created at {}", path.display());
        // Backdate the mtime so the freshness assertion below can only
        // pass because of the touch, not the recent creation.
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - Duration::from_secs(120))
            .unwrap();
        assert!(!marker_is_fresh(&path, Duration::from_secs(60)));
        marker.touch();
        assert!(marker_is_fresh(&path, Duration::from_secs(60)));
        drop(marker);
        assert!(!path.exists(), "marker not removed on drop");
    }

    /// A fresh temp marker path for `name`, and its status as a client
    /// would read it with the real staleness window.
    fn temp_marker(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "llmman-marker-{name}-{}/.downloading",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn status_of(path: &Path) -> Option<String> {
        marker_status(path, DOWNLOAD_MARKER_STALE_AFTER)
    }

    fn backdate(path: &Path) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - 2 * DOWNLOAD_MARKER_STALE_AFTER)
            .unwrap();
    }

    /// What a waiting client reads back: the status text while the
    /// marker is fresh (a touch keeps it), `None` once stale or gone.
    #[test]
    fn marker_status_reports_the_last_written_text_while_fresh() {
        let path = temp_marker("status");
        assert_eq!(status_of(&path), None);
        let marker = DownloadMarker::create_at(path.clone());
        assert_eq!(status_of(&path).as_deref(), Some(""));
        marker.set_status("llama.tar.gz: 12 MB / 40 MB (30%)");
        assert_eq!(
            status_of(&path).as_deref(),
            Some("llama.tar.gz: 12 MB / 40 MB (30%)")
        );
        // A heartbeat touch refreshes without blanking the line.
        std::fs::write(&path, b"").unwrap();
        marker.touch();
        assert_eq!(
            status_of(&path).as_deref(),
            Some("llama.tar.gz: 12 MB / 40 MB (30%)")
        );
        marker.set_status("extracting llama.tar.gz");
        backdate(&path);
        assert_eq!(status_of(&path), None);
        drop(marker);
        assert_eq!(status_of(&path), None);
    }

    /// Work that reports nothing must not let the marker go stale: the
    /// heartbeat re-touches it, keeping the status, and stops with the work.
    #[test]
    fn keep_alive_heartbeat_refreshes_the_marker_while_work_runs() {
        let path = temp_marker("heartbeat");
        let marker = DownloadMarker::create_at(path.clone());
        marker.set_status("extracting llama.tar.gz");
        backdate(&path);
        assert_eq!(status_of(&path), None);
        // Wait for the heartbeat itself, not a fixed time: a loaded runner
        // may not schedule its thread for a while.
        let out = marker.keep_alive_every(Duration::from_millis(5), || {
            let deadline = Instant::now() + Duration::from_secs(30);
            while status_of(&path).is_none() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            7
        });
        assert_eq!(out, 7);
        assert_eq!(status_of(&path).as_deref(), Some("extracting llama.tar.gz"));
        // The heartbeat stopped with the closure: a backdated marker stays stale.
        backdate(&path);
        assert_eq!(status_of(&path), None);
        drop(marker);
        assert!(!path.exists());
    }

    #[test]
    fn download_progress_text_includes_a_percentage_only_with_a_known_total() {
        assert_eq!(
            download_progress_text("x.tar.gz", 30 * 1024 * 1024, 120 * 1024 * 1024),
            format!(
                "x.tar.gz: {} / {} (25%)",
                human_size(30 * 1024 * 1024),
                human_size(120 * 1024 * 1024)
            )
        );
        assert_eq!(
            download_progress_text("x.tar.gz", 4096, 0),
            format!("x.tar.gz: {}", human_size(4096))
        );
    }

    #[test]
    fn host_arch_token_returns_a_nonempty_token() {
        // Just exercises the mapping logic directly since
        // std::env::consts::ARCH is fixed per test-binary target.
        assert!(!host_arch_token().is_empty());
    }

    /// Regression test: `newest_cached` used to pick the string-max of two
    /// cached tags, which silently regresses once the build counter crosses
    /// a power of ten (`"b9999" > "b10000"` lexically).
    #[test]
    fn build_number_compares_correctly_across_a_digit_count_boundary() {
        assert!(build_number("b10000") > build_number("b9999"));
        assert_eq!(build_number("b10360"), Some(10360));
        assert_eq!(build_number("not-a-tag"), None);
    }

    #[test]
    fn cached_in_tag_accepts_a_legacy_cuda_minor_dir() {
        let tag = std::env::temp_dir().join(format!("llmman-legacy-cuda-{}", std::process::id()));
        let legacy = tag.join("cuda-12.4");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("llama-server"), b"").unwrap();
        std::fs::write(legacy.join(".llmman-complete"), b"").unwrap();
        let found = cached_in_tag(&tag, "cuda-12", "llama-server");
        assert_eq!(found, Some(legacy.join("llama-server")));
        assert_eq!(cached_in_tag(&tag, "cuda-13", "llama-server"), None);
        std::fs::remove_dir_all(&tag).unwrap();
    }

    // ----- repo, tag convention and release URLs -----

    const MIRROR: &str = "docker/sbx-releases";

    #[test]
    fn parse_repo_defaults_and_validates() {
        for unset in [None, Some(""), Some("  ")] {
            assert_eq!(parse_repo(unset).unwrap(), "ggml-org/llama.cpp");
        }
        for ok in [MIRROR, " docker/sbx-releases ", "a/b", "Foo_1/bar.baz-2"] {
            assert_eq!(parse_repo(Some(ok)).unwrap(), ok.trim());
        }
        for bad in [
            "llama.cpp",
            "/repo",
            "owner/",
            "a/b/c",
            "a b/c",
            "a/b c",
            "../x",
            "a/..",
            "./.",
            "a/b?x=1",
            "https://github.com/a/b",
            "a/b%2F",
            "a\\b/c",
        ] {
            let err = parse_repo(Some(bad)).unwrap_err().to_string();
            assert!(
                err.contains(REPO_VAR) && err.contains("<owner>/<repo>"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn release_tag_is_prefixed_in_a_mirror_only() {
        assert_eq!(release_tag(DEFAULT_REPO, "b12359"), "b12359");
        assert_eq!(release_tag("GGML-org/Llama.cpp", "b12359"), "b12359");
        assert_eq!(release_tag(MIRROR, "b12359"), "llama.cpp-b12359");
        // Already a mirror tag: not prefixed twice.
        assert_eq!(release_tag(MIRROR, "llama.cpp-b12359"), "llama.cpp-b12359");
        assert_eq!(strip_release_tag(MIRROR, "llama.cpp-b12359"), "b12359");
        assert_eq!(strip_release_tag(MIRROR, "b12359"), "b12359");
        assert_eq!(strip_release_tag(DEFAULT_REPO, "b12359"), "b12359");
    }

    #[test]
    fn release_urls_follow_the_repo_and_tag_convention() {
        let url = |repo, v| release_url("https://api.example", repo, v);
        assert_eq!(
            url(DEFAULT_REPO, Some("b12359")).unwrap(),
            "https://api.example/repos/ggml-org/llama.cpp/releases/tags/b12359"
        );
        assert_eq!(
            url(DEFAULT_REPO, None).unwrap(),
            "https://api.example/repos/ggml-org/llama.cpp/releases/latest"
        );
        assert_eq!(
            url(MIRROR, Some("b12359")).unwrap(),
            "https://api.example/repos/docker/sbx-releases/releases/tags/llama.cpp-b12359"
        );
        let err = url(MIRROR, None).unwrap_err().to_string();
        assert!(
            err.contains(REPO_VAR)
                && err.contains(MIRROR)
                && err.contains("latest")
                && err.contains("pinned"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_mirror_release_names_the_repo_tag_and_variable() {
        let err = release_not_found(MIRROR, "b12359").to_string();
        for needle in [MIRROR, "llama.cpp-b12359", REPO_VAR, "404"] {
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn mirrors_and_upstream_never_share_an_install_dir() {
        assert_eq!(cache_key(DEFAULT_REPO, "b12359"), "b12359");
        let key = cache_key(MIRROR, "b12359");
        assert!(key.starts_with("docker-sbx-releases-"), "{key}");
        assert!(key.ends_with("-b12359"), "{key}");
        assert_ne!(key, "b12359");
        assert_eq!(build_number(&key), None);
    }

    #[test]
    fn mirrors_whose_names_join_alike_get_different_keys() {
        let (a, b) = (cache_key("a-b/c", "b12359"), cache_key("a/b-c", "b12359"));
        assert_ne!(a, b);
        assert_ne!(cache_key("a-b/c", "b1"), cache_key("a-b/c", "b2"));
        for key in [a, b] {
            assert!(!key.contains(['/', '\\', ':']), "{key}");
        }
    }

    #[test]
    fn the_same_mirror_reuses_its_key_whatever_the_case() {
        assert_eq!(
            cache_key(MIRROR, "b12359"),
            cache_key("Docker/SBX-Releases", "b12359")
        );
    }

    /// Serves one canned response on a loopback port; returns the API base
    /// URL and a handle yielding the request line.
    fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                conn.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{body}",
                body.len()
            );
            conn.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&request)
                .lines()
                .next()
                .unwrap()
                .to_owned()
        });
        (base, handle)
    }

    fn local_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
    }

    #[test]
    fn fetching_a_mirror_release_asks_for_its_prefixed_tag_and_returns_the_build_tag() {
        let (base, request) = serve_once(
            "200 OK",
            r#"{"tag_name": "llama.cpp-b12359", "assets": []}"#,
        );
        let release = fetch_release(&local_client(), &base, MIRROR, Some("b12359")).unwrap();
        assert_eq!(release.tag_name, "b12359");
        assert_eq!(
            request.join().unwrap(),
            "GET /repos/docker/sbx-releases/releases/tags/llama.cpp-b12359 HTTP/1.1"
        );
    }

    #[test]
    fn fetching_upstream_keeps_the_plain_tag() {
        let (base, request) = serve_once("200 OK", r#"{"tag_name": "b12359", "assets": []}"#);
        let release = fetch_release(&local_client(), &base, DEFAULT_REPO, Some("b12359")).unwrap();
        assert_eq!(release.tag_name, "b12359");
        assert_eq!(
            request.join().unwrap(),
            "GET /repos/ggml-org/llama.cpp/releases/tags/b12359 HTTP/1.1"
        );
    }

    #[test]
    fn a_mirror_404_is_an_actionable_error() {
        let (base, _request) = serve_once("404 Not Found", r#"{"message": "Not Found"}"#);
        let err = fetch_release(&local_client(), &base, MIRROR, Some("b12359"))
            .unwrap_err()
            .to_string();
        for needle in [MIRROR, "llama.cpp-b12359", REPO_VAR] {
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn an_upstream_404_and_other_mirror_failures_stay_plain_http_errors() {
        let (base, _request) = serve_once("404 Not Found", "{}");
        let err = fetch_release(&local_client(), &base, DEFAULT_REPO, Some("b1"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("returned 404"), "{err}");
        let (base, _request) = serve_once("500 Internal Server Error", "{}");
        let err = fetch_release(&local_client(), &base, MIRROR, Some("b1"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("returned 500"), "{err}");
    }

    #[test]
    fn latest_from_a_mirror_fails_before_any_request() {
        // Nothing listens on the discard port: reaching the network would
        // be a different error.
        let err = fetch_release(&local_client(), "http://127.0.0.1:9", MIRROR, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mirrors are pinned"), "{err}");
        assert!(check_pin(DEFAULT_REPO, None).is_ok());
        assert!(check_pin(MIRROR, Some("b1")).is_ok());
        assert!(check_pin(MIRROR, None).is_err());
    }

    /// A pointer pin resolves to the real build in a mirror too: the
    /// dereference asks for the build tag, which the fetch prefixes.
    #[test]
    fn a_pointer_pin_dereferences_to_the_build_tag_in_a_mirror() {
        let asked = std::cell::RefCell::new(Vec::new());
        let release = resolve_release_with(
            |v| {
                asked.borrow_mut().push(v.map(str::to_owned));
                Ok(if v == Some("v1.0.0") {
                    pointer_release("v1.0.0")
                } else {
                    binary_release("b12359")
                })
            },
            |_| Ok("b12359\n".into()),
            Some("v1.0.0"),
        )
        .unwrap();
        assert_eq!(release.tag_name, "b12359");
        assert_eq!(
            *asked.borrow(),
            [Some("v1.0.0".into()), Some("b12359".into())]
        );
        assert_eq!(
            release_url("", MIRROR, Some("b12359")).unwrap(),
            "/repos/docker/sbx-releases/releases/tags/llama.cpp-b12359"
        );
    }

    // ----- the host table -----

    /// The asset names of the real b12359 release.
    const B12359_ASSETS: &[&str] = &[
        "cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
        "cudart-llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.5-arm64.zip",
        "cudart-llama-bin-win-cuda-13.5-x64.zip",
        "llama-b12359-bin-android-arm64-snapdragon.tar.gz",
        "llama-b12359-bin-android-arm64.tar.gz",
        "llama-b12359-bin-linux-arm64-snapdragon.tar.gz",
        "llama-b12359-bin-macos-arm64.tar.gz",
        "llama-b12359-bin-macos-x64.tar.gz",
        "llama-b12359-bin-ubuntu-26.04-s390x.tar.gz",
        "llama-b12359-bin-ubuntu-arm64.tar.gz",
        "llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
        "llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
        "llama-b12359-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
        "llama-b12359-bin-ubuntu-rocm-10.0-x64.tar.gz",
        "llama-b12359-bin-ubuntu-s390x.tar.gz",
        "llama-b12359-bin-ubuntu-sycl-fp16-x64.tar.gz",
        "llama-b12359-bin-ubuntu-sycl-fp32-x64.tar.gz",
        "llama-b12359-bin-ubuntu-vulkan-arm64.tar.gz",
        "llama-b12359-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b12359-bin-ubuntu-x64.tar.gz",
        "llama-b12359-bin-ubuntu-zdnn-s390x.tar.gz",
        "llama-b12359-bin-win-cpu-arm64.zip",
        "llama-b12359-bin-win-cpu-x64.zip",
        "llama-b12359-bin-win-cuda-12.4-x64.zip",
        "llama-b12359-bin-win-cuda-13.5-arm64.zip",
        "llama-b12359-bin-win-cuda-13.5-x64.zip",
        "llama-b12359-bin-win-opencl-adreno-arm64.zip",
        "llama-b12359-bin-win-openvino-2026.4.1-x64.zip",
        "llama-b12359-bin-win-rocm-10.0-x64.zip",
        "llama-b12359-bin-win-sycl-x64.zip",
        "llama-b12359-bin-win-vulkan-arm64.zip",
        "llama-b12359-bin-win-vulkan-x64.zip",
        "llama-b12359-ui.tar.gz",
        "llama-b12359-xcframework.zip",
    ];

    fn b12359_without(skip: impl Fn(&str) -> bool) -> Release {
        Release {
            tag_name: "b12359".into(),
            assets: B12359_ASSETS
                .iter()
                .filter(|name| !skip(name))
                .map(|name| Asset {
                    name: (*name).into(),
                    browser_download_url: String::new(),
                    size: 0,
                    digest: None,
                })
                .collect(),
        }
    }

    fn b12359() -> Release {
        b12359_without(|_| false)
    }

    #[test]
    fn every_host_has_a_query_a_distinct_name_and_a_match_in_b12359() {
        let release = b12359();
        let mut names = std::collections::HashSet::new();
        for host in HOSTS {
            assert!(names.insert(host.name()), "{} listed twice", host.name());
            let query = host.query().unwrap_or_else(|| panic!("{}", host.name()));
            let (primary, companion) = selected(&release, &query);
            assert!(primary.is_some(), "no primary asset for {}", host.name());
            assert_eq!(
                companion.is_some(),
                matches!(host.accel, Accel::Cuda(_)),
                "{}",
                host.name()
            );
            assert_eq!(query.label, host.accel.name());
        }
        assert_eq!(HOSTS.len(), 20);
    }

    #[test]
    fn host_assets_of_b12359_are_every_selectable_build_sorted() {
        let listing = list_assets(DEFAULT_REPO, &b12359());
        assert_eq!(listing.repo, "ggml-org/llama.cpp");
        assert_eq!(
            (listing.release.as_str(), listing.tag.as_str()),
            ("b12359", "b12359")
        );
        assert!(listing.missing.is_empty(), "{:?}", listing.missing);
        assert_eq!(
            listing.assets,
            [
                "cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "cudart-llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                "cudart-llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
                "cudart-llama-bin-win-cuda-13.5-arm64.zip",
                "cudart-llama-bin-win-cuda-13.5-x64.zip",
                "llama-b12359-bin-macos-arm64.tar.gz",
                "llama-b12359-bin-macos-x64.tar.gz",
                "llama-b12359-bin-ubuntu-arm64.tar.gz",
                "llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                "llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "llama-b12359-bin-ubuntu-rocm-10.0-x64.tar.gz",
                "llama-b12359-bin-ubuntu-s390x.tar.gz",
                "llama-b12359-bin-ubuntu-vulkan-arm64.tar.gz",
                "llama-b12359-bin-ubuntu-vulkan-x64.tar.gz",
                "llama-b12359-bin-ubuntu-x64.tar.gz",
                "llama-b12359-bin-win-cpu-arm64.zip",
                "llama-b12359-bin-win-cpu-x64.zip",
                "llama-b12359-bin-win-cuda-12.4-x64.zip",
                "llama-b12359-bin-win-cuda-13.5-arm64.zip",
                "llama-b12359-bin-win-cuda-13.5-x64.zip",
                "llama-b12359-bin-win-opencl-adreno-arm64.zip",
                "llama-b12359-bin-win-rocm-10.0-x64.zip",
                "llama-b12359-bin-win-vulkan-arm64.zip",
                "llama-b12359-bin-win-vulkan-x64.zip",
            ]
        );
    }

    #[test]
    fn host_assets_do_not_depend_on_the_order_of_the_release_listing() {
        let mut reversed = b12359();
        reversed.assets.reverse();
        assert_eq!(
            list_assets(DEFAULT_REPO, &reversed),
            list_assets(DEFAULT_REPO, &b12359())
        );
    }

    #[test]
    fn a_mirror_listing_reports_the_prefixed_release_and_the_build_tag() {
        let listing = list_assets(MIRROR, &b12359());
        assert_eq!(listing.repo, MIRROR);
        assert_eq!(listing.release, "llama.cpp-b12359");
        assert_eq!(listing.tag, "b12359");
        assert_eq!(listing.assets, list_assets(DEFAULT_REPO, &b12359()).assets);
    }

    #[test]
    fn a_host_without_a_build_is_missing_not_an_error() {
        let release =
            b12359_without(|n| n.contains("ubuntu-vulkan-arm64") || n.contains("win-rocm"));
        let listing = list_assets(DEFAULT_REPO, &release);
        assert_eq!(listing.missing, ["linux-arm64-vulkan", "windows-x64-rocm"]);
        assert_eq!(listing.assets.len(), 24);
    }

    #[test]
    fn a_cuda_build_without_its_runtime_bundle_is_listed_as_missing_a_companion() {
        let release = b12359_without(|n| n == "cudart-llama-bin-win-cuda-13.5-arm64.zip");
        let listing = list_assets(DEFAULT_REPO, &release);
        assert_eq!(listing.missing, ["windows-arm64-cuda-13/companion"]);
        assert!(listing
            .assets
            .contains(&"llama-b12359-bin-win-cuda-13.5-arm64.zip".to_string()));
    }

    #[test]
    fn a_release_with_none_of_the_assets_lists_every_host_as_missing() {
        let release = b12359_without(|_| true);
        let listing = list_assets(DEFAULT_REPO, &release);
        assert!(listing.assets.is_empty());
        assert_eq!(listing.missing.len(), HOSTS.len());
    }

    #[test]
    fn the_listing_serializes_with_the_documented_keys() {
        let json = serde_json::to_value(list_assets(MIRROR, &b12359())).unwrap();
        let mut keys: Vec<_> = json.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["assets", "missing", "release", "repo", "tag"]);
        assert_eq!(json["missing"], serde_json::json!([]));
    }

    // ----- per-host selection -----

    /// The (primary, companion) names a host of `os`/`arch` with `gpu`
    /// downloads from b12359.
    fn pick(
        os: HostOs,
        arch: &'static str,
        gpu: HostGpu,
        glibc: Option<(u32, u32)>,
    ) -> (Option<String>, Option<String>) {
        let (descriptor, _) = select_host(os, arch, gpu, glibc);
        selected(&b12359(), &descriptor.query().unwrap())
    }

    fn picked(primary: &str, companion: Option<&str>) -> (Option<String>, Option<String>) {
        (Some(primary.into()), companion.map(str::to_owned))
    }

    /// DGX Spark (GB10: aarch64, CUDA 13 driver) gets the arm64 CUDA 13
    /// build and its runtime bundle.
    #[test]
    fn a_dgx_spark_gets_the_ubuntu_cuda_13_arm64_build_and_cudart() {
        let gpu = HostGpu::Cuda { major: 13 };
        for glibc in [None, Some((2, 39))] {
            assert_eq!(
                pick(HostOs::Linux, "arm64", gpu, glibc),
                picked(
                    "llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                    Some("cudart-llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz")
                )
            );
        }
        let (descriptor, why) = select_host(HostOs::Linux, "arm64", gpu, Some((2, 39)));
        assert_eq!(descriptor.accel, Accel::Cuda(13));
        assert_eq!(why, None);
    }

    /// An arm64 CUDA 12 driver has no build; it falls back to the CPU one.
    #[test]
    fn an_arm64_cuda_12_linux_host_falls_back_to_cpu_with_a_reason() {
        let (descriptor, why) =
            select_host(HostOs::Linux, "arm64", HostGpu::Cuda { major: 12 }, None);
        assert_eq!(descriptor.accel, Accel::Cpu);
        assert!(why.unwrap().contains("arm64 Linux CUDA 12"));
        let (descriptor, why) = select_host(
            HostOs::Linux,
            "x64",
            HostGpu::Cuda { major: 13 },
            Some((2, 35)),
        );
        assert_eq!(descriptor.accel, Accel::Cpu);
        assert!(why.unwrap().contains("glibc 2.38"));
    }

    /// A Blackwell laptop GPU (RTX PRO 500: CUDA 13 driver) on Windows x64.
    #[test]
    fn a_windows_blackwell_laptop_gets_the_win_cuda_13_x64_build_and_cudart() {
        assert_eq!(
            pick(HostOs::Windows, "x64", HostGpu::Cuda { major: 13 }, None),
            picked(
                "llama-b12359-bin-win-cuda-13.5-x64.zip",
                Some("cudart-llama-bin-win-cuda-13.5-x64.zip")
            )
        );
    }

    #[test]
    fn host_selection_per_os_arch_and_gpu() {
        use HostGpu::*;
        let linux = HostOs::Linux;
        let win = HostOs::Windows;
        let cuda = |major| Cuda { major };
        for (os, arch, gpu, want) in [
            (
                HostOs::Macos,
                "arm64",
                None,
                "llama-b12359-bin-macos-arm64.tar.gz",
            ),
            (
                HostOs::Macos,
                "arm64",
                Metal,
                "llama-b12359-bin-macos-arm64.tar.gz",
            ),
            (
                HostOs::Macos,
                "x64",
                None,
                "llama-b12359-bin-macos-x64.tar.gz",
            ),
            (linux, "x64", None, "llama-b12359-bin-ubuntu-x64.tar.gz"),
            (linux, "arm64", None, "llama-b12359-bin-ubuntu-arm64.tar.gz"),
            (linux, "s390x", None, "llama-b12359-bin-ubuntu-s390x.tar.gz"),
            (
                linux,
                "x64",
                Vulkan,
                "llama-b12359-bin-ubuntu-vulkan-x64.tar.gz",
            ),
            (
                linux,
                "arm64",
                Vulkan,
                "llama-b12359-bin-ubuntu-vulkan-arm64.tar.gz",
            ),
            (
                linux,
                "x64",
                Rocm,
                "llama-b12359-bin-ubuntu-rocm-10.0-x64.tar.gz",
            ),
            (linux, "arm64", Rocm, "llama-b12359-bin-ubuntu-arm64.tar.gz"),
            (linux, "x64", Opencl, "llama-b12359-bin-ubuntu-x64.tar.gz"),
            (
                linux,
                "x64",
                cuda(12),
                "llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
            ),
            (
                linux,
                "x64",
                cuda(13),
                "llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
            ),
            (
                linux,
                "x64",
                cuda(14),
                "llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
            ),
            (
                linux,
                "arm64",
                cuda(12),
                "llama-b12359-bin-ubuntu-arm64.tar.gz",
            ),
            (win, "x64", None, "llama-b12359-bin-win-cpu-x64.zip"),
            (win, "arm64", None, "llama-b12359-bin-win-cpu-arm64.zip"),
            (win, "x64", Vulkan, "llama-b12359-bin-win-vulkan-x64.zip"),
            (
                win,
                "arm64",
                Vulkan,
                "llama-b12359-bin-win-vulkan-arm64.zip",
            ),
            (win, "x64", Rocm, "llama-b12359-bin-win-rocm-10.0-x64.zip"),
            (win, "arm64", Rocm, "llama-b12359-bin-win-cpu-arm64.zip"),
            (
                win,
                "arm64",
                Opencl,
                "llama-b12359-bin-win-opencl-adreno-arm64.zip",
            ),
            (win, "x64", Opencl, "llama-b12359-bin-win-cpu-x64.zip"),
            (
                win,
                "x64",
                cuda(12),
                "llama-b12359-bin-win-cuda-12.4-x64.zip",
            ),
            (
                win,
                "x64",
                cuda(13),
                "llama-b12359-bin-win-cuda-13.5-x64.zip",
            ),
            (
                win,
                "arm64",
                cuda(12),
                "llama-b12359-bin-win-cuda-13.5-arm64.zip",
            ),
            (
                win,
                "arm64",
                cuda(13),
                "llama-b12359-bin-win-cuda-13.5-arm64.zip",
            ),
            (
                HostOs::Android,
                "arm64",
                None,
                "llama-b12359-bin-android-arm64.tar.gz",
            ),
            (
                HostOs::Android,
                "arm64",
                Vulkan,
                "llama-b12359-bin-android-arm64.tar.gz",
            ),
        ] {
            let got = pick(os, arch, gpu, Option::None).0;
            assert_eq!(got.as_deref(), Some(want), "{os:?} {arch} {gpu:?}");
        }
    }

    /// The patterns the table produced before it was shared, per host.
    #[test]
    fn host_queries_keep_their_patterns_and_labels() {
        let q = |os, arch, accel| {
            let q = host_query(os, arch, accel);
            (q.must_contain, q.companion_must_contain, q.label)
        };
        let some = |s: &str| Some(s.to_owned());
        assert_eq!(
            q(HostOs::Macos, "arm64", Accel::Metal),
            ("-bin-macos-arm64.tar.gz".into(), None, "metal".into())
        );
        assert_eq!(
            q(HostOs::Macos, "x64", Accel::Cpu),
            ("-bin-macos-x64.tar.gz".into(), None, "cpu".into())
        );
        assert_eq!(
            q(HostOs::Linux, "x64", Accel::Rocm),
            ("-bin-ubuntu-rocm-".into(), None, "rocm".into())
        );
        assert_eq!(
            q(HostOs::Linux, "arm64", Accel::Cuda(13)),
            (
                "-bin-ubuntu-cuda-13.*-arm64.tar.gz".into(),
                some("cudart-llama-{tag}-bin-ubuntu-cuda-13.*-arm64.tar.gz"),
                "cuda-13".into()
            )
        );
        assert_eq!(
            q(HostOs::Windows, "x64", Accel::Cuda(12)),
            (
                "-bin-win-cuda-12.*-x64.zip".into(),
                some("cudart-llama-bin-win-cuda-12.*-x64.zip"),
                "cuda-12".into()
            )
        );
        assert_eq!(
            q(HostOs::Windows, "arm64", Accel::Opencl),
            (
                "-bin-win-opencl-adreno-arm64.zip".into(),
                None,
                "opencl".into()
            )
        );
        assert_eq!(
            q(HostOs::Android, "arm64", Accel::Cpu),
            ("-bin-android-arm64.tar.gz".into(), None, "cpu".into())
        );
        assert!(host(HostOs::Macos, "arm64", Accel::Rocm).query().is_none());
    }

    /// Real end-to-end check against the actual GitHub API and a real
    /// download+extract of whatever asset this host's own `asset_query()`
    /// picks — not run by a plain `cargo test` (network + a
    /// multi-hundred-MB-in-the-worst-case download), but the only way to
    /// verify the real llama.cpp release layout (tarball vs. zip, nested
    /// `llama-<tag>/` directory vs. flat root, companion cudart package)
    /// matches what this module assumes. Run explicitly with
    /// `cargo test --bin llmman -- --ignored ensure_llama_server_downloads`.
    #[test]
    #[ignore = "hits the network and downloads a real llama.cpp release"]
    fn ensure_llama_server_downloads_a_runnable_binary() {
        let resolved = ensure_llama_server(None).expect("ensure_llama_server");
        assert!(
            resolved.bin.is_file(),
            "{} is not a file",
            resolved.bin.display()
        );
        let output = std::process::Command::new(&resolved.bin)
            .arg("--version")
            .output()
            .expect("run downloaded llama-server --version");
        assert!(
            output.status.success(),
            "llama-server --version failed: {output:?}"
        );
    }
}
