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
//! `cmd::serve`'s local (non-`--ociman`) path still prefers whatever
//! `llama-server` is already on `PATH` (see `resolve_llama_server`
//! there) — this module is only reached as a fallback, or when
//! `--llama-cpp-version` pins an explicit release. Once downloaded, a
//! given release+backend combination is cached under
//! [`install_root`]`/<tag>/<backend>/` and never re-fetched.
//!
//! Coverage gap (unavoidable, not an llmman limitation): llama.cpp does
//! not publish a prebuilt **Linux** CUDA binary at all — only Windows
//! gets prebuilt CUDA — so an NVIDIA GPU detected on Linux falls back to
//! the CPU build here, with a message pointing at `llmman serve --ociman
//! docker` (see `crate::container`, which *does* have a CUDA path via
//! `ghcr.io/ggml-org/llama.cpp:server-cuda*`) as the GPU-accelerated
//! alternative.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::fmt::human_size;
use crate::hostgpu::HostGpu;

const REPO_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp";

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
}

fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        // Large Windows CUDA packages bundle the CUDA runtime itself
        // (hundreds of MB) — a short fixed timeout would abort a real,
        // still-progressing download on a slow link.
        .timeout(Duration::from_secs(60 * 30))
        .build()
        .context("build http client")
}

/// Fetches release metadata for `version` (an exact tag, e.g. "b10360"),
/// or the latest release if `None` — mirroring how a caller of Ollama's
/// own installer can pin `OLLAMA_VERSION` instead of always taking latest.
fn fetch_release(client: &reqwest::blocking::Client, version: Option<&str>) -> Result<Release> {
    let url = match version {
        Some(v) => format!("{REPO_API}/releases/tags/{v}"),
        None => format!("{REPO_API}/releases/latest"),
    };
    let resp = client
        .get(&url)
        .header("user-agent", "llmman")
        .send()
        .with_context(|| format!("query {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("GitHub API {url} returned {}", resp.status());
    }
    resp.json::<Release>()
        .with_context(|| format!("parse release metadata from {url}"))
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
fn resolve_release(client: &reqwest::blocking::Client, version: Option<&str>) -> Result<Release> {
    let release = fetch_release(client, version)?;
    let Some(pointer) = pointer_asset(&release) else {
        return Ok(release);
    };
    let pointer_tag = &release.tag_name;
    let content = fetch_text(client, &pointer.browser_download_url).with_context(|| {
        format!("download {NIGHTLY_POINTER} from pointer release {pointer_tag}")
    })?;
    let tag = parse_pointer_tag(&content)
        .ok_or_else(|| anyhow!("pointer release {pointer_tag}'s {NIGHTLY_POINTER} is empty"))?;
    let dereferenced = fetch_release(client, Some(tag)).with_context(|| {
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
    let resp = client
        .get(url)
        .header("user-agent", "llmman")
        .send()
        .with_context(|| format!("query {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("download {url} returned {}", resp.status());
    }
    resp.text().with_context(|| format!("read body of {url}"))
}

fn find_asset<'a>(release: &'a Release, must_contain: &str) -> Option<&'a Asset> {
    release
        .assets
        .iter()
        .find(|a| a.name.contains(must_contain))
}

// ---------------------------------------------------------------------------
// Which asset does this host want?
// ---------------------------------------------------------------------------

/// Describes which release asset(s) [`ensure_llama_server`] should fetch
/// for the current host: a required substring identifying the primary
/// package, an optional companion package (only Windows CUDA's separate
/// `cudart-*` runtime-DLL bundle today), and a short label used for the
/// on-disk cache directory name and log messages.
struct AssetQuery {
    must_contain: String,
    companion_must_contain: Option<String>,
    label: String,
}

fn host_arch_token() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "s390x" => "s390x",
        other => other,
    }
}

/// See the real asset names published at
/// <https://github.com/ggml-org/llama.cpp/releases> — verified directly
/// against the `latest` release's own asset list and its release workflow
/// (`.github/workflows/release.yml`) while writing this, rather than
/// guessed: e.g. no `ubuntu-cuda` asset exists at all (only Windows gets a
/// prebuilt CUDA package), and ROCm is x64-only on both Linux and
/// Windows.
fn asset_query() -> AssetQuery {
    let arch = host_arch_token();

    #[cfg(target_os = "macos")]
    {
        return if std::env::consts::ARCH == "aarch64" {
            AssetQuery {
                must_contain: "-bin-macos-arm64.tar.gz".into(),
                companion_must_contain: None,
                label: "metal".into(),
            }
        } else {
            AssetQuery {
                must_contain: "-bin-macos-x64.tar.gz".into(),
                companion_must_contain: None,
                label: "cpu".into(),
            }
        };
    }

    #[cfg(target_os = "linux")]
    {
        return match crate::hostgpu::detect() {
            HostGpu::Vulkan => AssetQuery {
                must_contain: format!("-bin-ubuntu-vulkan-{arch}.tar.gz"),
                companion_must_contain: None,
                label: "vulkan".into(),
            },
            HostGpu::Rocm if arch == "x64" => AssetQuery {
                must_contain: "-bin-ubuntu-rocm-".into(),
                companion_must_contain: None,
                label: "rocm".into(),
            },
            HostGpu::Cuda { .. } => {
                eprintln!(
                    "[llmman] NVIDIA GPU detected, but llama.cpp does not publish a \
                     prebuilt Linux CUDA binary — falling back to the CPU build. Use \
                     `llmman serve --ociman docker` (or `--ociman podman`) for GPU \
                     acceleration on Linux, or build llama.cpp yourself with \
                     GGML_CUDA=ON and put llama-server on PATH."
                );
                AssetQuery {
                    must_contain: format!("-bin-ubuntu-{arch}.tar.gz"),
                    companion_must_contain: None,
                    label: "cpu".into(),
                }
            }
            _ => AssetQuery {
                must_contain: format!("-bin-ubuntu-{arch}.tar.gz"),
                companion_must_contain: None,
                label: "cpu".into(),
            },
        };
    }

    #[cfg(target_os = "windows")]
    {
        return match crate::hostgpu::detect() {
            HostGpu::Cuda { major } => {
                // llama.cpp publishes CUDA 12.4 and 13.3 for x64, and a
                // 13.4 "preview" build for arm64 (no CUDA 12 on arm64) —
                // see the windows-cuda job's matrix.
                let cuda = if arch == "arm64" {
                    "13.4"
                } else if major >= 13 {
                    "13.3"
                } else {
                    "12.4"
                };
                AssetQuery {
                    must_contain: format!("-bin-win-cuda-{cuda}-{arch}.zip"),
                    companion_must_contain: Some(format!(
                        "cudart-llama-bin-win-cuda-{cuda}-{arch}.zip"
                    )),
                    label: format!("cuda-{cuda}"),
                }
            }
            HostGpu::Rocm if arch == "x64" => AssetQuery {
                must_contain: "-bin-win-rocm-".into(),
                companion_must_contain: None,
                label: "rocm".into(),
            },
            HostGpu::Vulkan if arch == "x64" => AssetQuery {
                must_contain: format!("-bin-win-vulkan-{arch}.zip"),
                companion_must_contain: None,
                label: "vulkan".into(),
            },
            _ => AssetQuery {
                must_contain: format!("-bin-win-cpu-{arch}.zip"),
                companion_must_contain: None,
                label: "cpu".into(),
            },
        };
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        AssetQuery {
            must_contain: format!("-bin-ubuntu-{arch}.tar.gz"),
            companion_must_contain: None,
            label: "cpu".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// On-disk cache layout
// ---------------------------------------------------------------------------

/// `~/.local/share/llmman/llama-server` on Linux/macOS,
/// `%LOCALAPPDATA%\llmman\llama-server` on Windows — sibling of
/// [`crate::default_store`]'s own store directory.
fn install_root() -> Result<PathBuf> {
    #[cfg(not(target_os = "windows"))]
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow!("could not determine home directory"))?
        .join(".local")
        .join("share");
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("could not determine local data directory"))?;
    Ok(base.join("llmman").join("llama-server"))
}

fn install_dir(tag: &str, label: &str) -> Result<PathBuf> {
    Ok(install_root()?.join(tag).join(label))
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
/// downloading the same asset at once (e.g. two concurrent `--pull-bin`
/// runs) never share a staging path.
fn tmp_path(name: &str) -> Result<PathBuf> {
    let dir = tmp_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir.join(format!("{name}.tmp-{}", std::process::id())))
}

/// Removes the staging file on drop, so a failed download or extraction
/// (an early `?` return) doesn't leave the archive behind.
struct RemoveOnDrop<'a>(&'a Path);

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
fn find_binary(dir: &Path, name: &str) -> Option<PathBuf> {
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
/// A live download refreshes it every [`PROGRESS_LOG_INTERVAL`], so a
/// marker older than this belongs to a download whose process died
/// without running the guard's Drop (e.g. a SIGKILLed daemon).
const DOWNLOAD_MARKER_STALE_AFTER: Duration = Duration::from_secs(60);

/// The download-in-progress marker: a fixed path under [`install_root`]
/// that both the downloading daemon (writer) and a client polling it
/// (reader, see `daemon::ensure_server`) can derive independently.
fn download_marker_path() -> Result<PathBuf> {
    Ok(install_root()?.join(".downloading"))
}

/// Whether some process is currently mid-download of a llama-server
/// release: the marker exists and was touched recently enough to belong
/// to a live download rather than a crashed one.
pub fn download_in_progress() -> bool {
    download_marker_path().is_ok_and(|path| marker_is_fresh(&path, DOWNLOAD_MARKER_STALE_AFTER))
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
/// marker failure must never fail the download itself.
struct DownloadMarker(Option<PathBuf>);

impl DownloadMarker {
    fn create() -> DownloadMarker {
        match download_marker_path() {
            Ok(p) => Self::create_at(p),
            Err(_) => DownloadMarker(None),
        }
    }

    /// The path-taking constructor behind [`DownloadMarker::create`], so
    /// tests can run the guard's lifecycle against a temp path instead of
    /// the real install root.
    fn create_at(path: PathBuf) -> DownloadMarker {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        DownloadMarker(std::fs::write(&path, b"").ok().map(|_| path))
    }

    /// Refreshes the marker's mtime so a reader can tell this live
    /// download from a crashed one whose Drop never ran.
    fn touch(&self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::write(path, b"");
        }
    }
}

impl Drop for DownloadMarker {
    fn drop(&mut self) {
        // With two concurrent downloads (pid-suffixed staging paths allow
        // them), the first finisher removes the shared marker and the
        // survivor's next touch recreates it within PROGRESS_LOG_INTERVAL;
        // that short unprotected window is accepted.
        if let Some(path) = &self.0 {
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
fn create_new_file(dest: &Path) -> Result<std::fs::File> {
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

fn download_to_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    label: &str,
    marker: &DownloadMarker,
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
            marker.touch();
            if total > 0 {
                eprintln!(
                    "[llmman] downloading {label}: {} / {} ({}%)",
                    human_size(downloaded),
                    human_size(total),
                    downloaded.saturating_mul(100) / total
                );
            }
            last_logged = Instant::now();
        }
    }
    eprintln!("[llmman] downloaded {label}: {}", human_size(downloaded));
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("open {}", archive_path.display()))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);
    archive
        .unpack(dest)
        .with_context(|| format!("extract {} into {}", archive_path.display(), dest.display()))
}

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

fn extract(archive_path: &Path, archive_name: &str, dest: &Path) -> Result<()> {
    if archive_name.ends_with(".zip") {
        extract_zip(archive_path, dest)
    } else {
        extract_tar_gz(archive_path, dest)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// A resolved, ready-to-run local `llama-server`.
pub struct Resolved {
    pub bin: PathBuf,
    /// Short backend label (`"cpu"`, `"vulkan"`, `"rocm"`, `"cuda-12.4"`,
    /// `"metal"`) — surfaced only for logging.
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
/// this via `tokio::task::spawn_blocking` (see `cmd::serve`'s
/// `resolve_llama_server`).
pub fn ensure_llama_server(pinned_version: Option<&str>) -> Result<Resolved> {
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
        let dest = install_dir(tag, &query.label)?;
        if let Some(bin) = find_binary(&dest, bin_name) {
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
            match try_ensure_from_network(pinned_version, &query, bin_name) {
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

    try_ensure_from_network(pinned_version, &query, bin_name)
}

fn try_ensure_from_network(
    pinned_version: Option<&str>,
    query: &AssetQuery,
    bin_name: &str,
) -> Result<Resolved> {
    let client = http_client()?;
    let release = resolve_release(&client, pinned_version)?;
    let tag = release.tag_name.clone();
    let dest = install_dir(&tag, &query.label)?;

    if let Some(bin) = find_binary(&dest, bin_name) {
        return Ok(Resolved {
            bin,
            backend_label: query.label.clone(),
        });
    }

    let asset = find_asset(&release, &query.must_contain)
        .with_context(|| {
            format!(
                "no {} llama.cpp release asset found in {tag} (looked for a name containing {:?})",
                query.label, query.must_contain
            )
        })?
        .clone();

    eprintln!(
        "[llmman] downloading llama-server ({}) {tag}: {}",
        query.label, asset.name
    );
    // One marker for everything from here to the end of the function, so
    // a client that gave up waiting for this daemon (see ensure_server's
    // timeout path) knows not to kill it mid-download or mid-extract; the
    // downloads' progress loops keep it fresh. Extraction of a large
    // archive can outlive the last touch by more than the staleness
    // window; that residual gap is accepted.
    let marker = DownloadMarker::create();
    let tmp = tmp_path(&asset.name)?;
    let _cleanup = RemoveOnDrop(&tmp);
    download_to_file(
        &client,
        &asset.browser_download_url,
        &tmp,
        &asset.name,
        &marker,
    )?;
    marker.touch();
    extract(&tmp, &asset.name, &dest)?;

    if let Some(companion_substr) = &query.companion_must_contain {
        match find_asset(&release, companion_substr) {
            Some(companion) => {
                let companion = companion.clone();
                eprintln!("[llmman] downloading {}", companion.name);
                let tmp2 = tmp_path(&companion.name)?;
                let _cleanup = RemoveOnDrop(&tmp2);
                download_to_file(
                    &client,
                    &companion.browser_download_url,
                    &tmp2,
                    &companion.name,
                    &marker,
                )?;
                marker.touch();
                extract(&tmp2, &companion.name, &dest)?;
            }
            None => eprintln!(
                "[llmman] warning: expected companion asset containing {companion_substr:?} \
                 not found in release {tag} — {} may be missing runtime libraries it needs",
                query.label
            ),
        }
    }

    let bin = find_binary(&dest, bin_name).with_context(|| {
        format!(
            "llama-server binary not found after extracting {}",
            asset.name
        )
    })?;
    mark_executable(&bin)?;
    Ok(Resolved {
        bin,
        backend_label: query.label.clone(),
    })
}

#[cfg(unix)]
fn mark_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(path)?.permissions();
    perm.set_mode(perm.mode() | 0o111);
    std::fs::set_permissions(path, perm).with_context(|| format!("chmod +x {}", path.display()))
}

#[cfg(not(unix))]
fn mark_executable(_path: &Path) -> Result<()> {
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

/// Finds the most recently downloaded `<tag>/<label>` install under
/// [`install_root`] that already has `bin_name` extracted into it.
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
        let dest = entry.path().join(label);
        let Some(bin) = find_binary(&dest, bin_name) else {
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
    fn find_asset_matches_only_the_intended_substring() {
        let release = Release {
            tag_name: "b10360".into(),
            assets: vec![
                Asset {
                    name: "llama-b10360-bin-ubuntu-x64.tar.gz".into(),
                    browser_download_url: String::new(),
                },
                Asset {
                    name: "llama-b10360-bin-ubuntu-vulkan-x64.tar.gz".into(),
                    browser_download_url: String::new(),
                },
                Asset {
                    name: "llama-b10360-bin-ubuntu-rocm-7.14-x64.tar.gz".into(),
                    browser_download_url: String::new(),
                },
            ],
        };
        assert_eq!(
            find_asset(&release, "-bin-ubuntu-x64.tar.gz").unwrap().name,
            "llama-b10360-bin-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            find_asset(&release, "-bin-ubuntu-vulkan-x64.tar.gz")
                .unwrap()
                .name,
            "llama-b10360-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert_eq!(
            find_asset(&release, "-bin-ubuntu-rocm-").unwrap().name,
            "llama-b10360-bin-ubuntu-rocm-7.14-x64.tar.gz"
        );
        assert!(find_asset(&release, "-bin-win-cpu-x64.zip").is_none());
    }

    #[test]
    fn pointer_asset_detects_a_semver_pointer_release() {
        let pointer_release = Release {
            tag_name: "v0.3.0".into(),
            assets: vec![Asset {
                name: "nightly-tag.txt".into(),
                browser_download_url: String::new(),
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
                },
                Asset {
                    name: "llama-b10621-bin-ubuntu-vulkan-x64.tar.gz".into(),
                    browser_download_url: String::new(),
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
