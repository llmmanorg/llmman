//! Installs [`mlx-lm`](https://github.com/ml-explore/mlx-lm)'s
//! `mlx_lm.server` on macOS the way [`crate::llama_release`] fetches
//! `llama-server`, in [`crate::managed`]'s shape:
//!
//! - `<data_root>/uv/<version>/`: astral-sh's prebuilt
//!   [`uv`](https://github.com/astral-sh/uv) at [`default_uv_release`];
//! - `<data_root>/mlx-lm/<mlx-lm>-py<python>/`: a uv-managed CPython at
//!   [`default_python_release`] (`python/`) and a venv on it (`venv/`)
//!   with `mlx-lm` at [`default_mlx_lm_release`].
//!
//! Each pin is a repo-root file like `LLAMA_CPP_RELEASE`. Like
//! `--runtime` for llama.cpp: `bin` runs only these, `path` runs only
//! what is on `PATH` (nothing here runs), `auto` tries these and falls
//! back to `PATH` if the install fails ([`Fallback`]).
//!
//! `llmman serve` does this at startup, before it binds. The install
//! holds [`crate::llama_release`]'s download marker so a waiting
//! `daemon::ensure_server` can show what it is doing.

use std::ffi::OsStr;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::llama_release::{
    download_to_file, extract_tar_gz, find_binary, http_client, mark_executable, tmp_path,
    DownloadMarker, RemoveOnDrop,
};
use crate::managed;

/// `LLMMAN_MLX_LM_VERSION`: overrides [`default_mlx_lm_release`]. A
/// different value is its own version directory.
pub const MLX_LM_VERSION_VAR: &str = "LLMMAN_MLX_LM_VERSION";

/// Whether a managed install that fails may be replaced by the same
/// tool from `PATH` — `--runtime auto`'s behavior, not `bin`'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    Path,
    None,
}

/// The repo-root `UV_RELEASE` pin.
pub fn default_uv_release() -> &'static str {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/UV_RELEASE")).trim()
}

/// The repo-root `MLX_LM_RELEASE` pin.
pub fn default_mlx_lm_release() -> &'static str {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/MLX_LM_RELEASE")).trim()
}

/// The repo-root `PYTHON_RELEASE` pin: a `3.<minor>` that `mlx`, `vllm`
/// and `sglang` all publish wheels for, so one interpreter can serve any
/// Python engine llmman comes to install.
pub fn default_python_release() -> &'static str {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/PYTHON_RELEASE")).trim()
}

/// astral-sh's asset for this host; `None` off macOS, where `mlx` does
/// not run.
fn uv_asset_name(os: &str, arch: &str) -> Option<String> {
    matches!((os, arch), ("macos", "aarch64" | "x86_64"))
        .then(|| format!("uv-{arch}-apple-darwin.tar.gz"))
}

fn uv_download_url(version: &str, asset: &str) -> String {
    format!("https://github.com/astral-sh/uv/releases/download/{version}/{asset}")
}

fn mlx_lm_version() -> String {
    std::env::var(MLX_LM_VERSION_VAR)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default_mlx_lm_release().to_owned())
}

fn mlx_lm_dir_name(mlx_lm: &str, python: &str) -> String {
    format!("{mlx_lm}-py{python}")
}

fn uv_root() -> Result<PathBuf> {
    Ok(crate::data_root()?.join("uv"))
}

fn mlx_lm_root() -> Result<PathBuf> {
    Ok(crate::data_root()?.join("mlx-lm"))
}

/// This build's version directory under [`mlx_lm_root`].
fn managed_dir() -> Result<PathBuf> {
    Ok(mlx_lm_root()?.join(mlx_lm_dir_name(&mlx_lm_version(), default_python_release())))
}

fn server_in(dir: &Path) -> PathBuf {
    dir.join("venv").join("bin").join("mlx_lm.server")
}

fn completed_server(dir: &Path) -> Option<PathBuf> {
    let server = server_in(dir);
    (managed::is_complete(dir) && server.is_file()).then_some(server)
}

/// Whether the managed install for the current pins is complete, without
/// installing anything. For tests that must not measure the download.
pub fn installed() -> bool {
    managed_dir().is_ok_and(|dir| completed_server(&dir).is_some())
}

/// `<hex>  <name>` → the hex digest.
fn parse_sha256_file(text: &str) -> Option<String> {
    let hex = text.split_whitespace().next()?;
    (hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())).then(|| hex.to_lowercase())
}

fn sha256_of_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).context("hash download")?;
    Ok(hex::encode(hasher.finalize()))
}

/// The tool from `PATH` if `fallback` allows and it is there, else `err`
/// (with the fallback's absence noted).
fn or_from_path(err: anyhow::Error, name: &str, fallback: Fallback) -> Result<PathBuf> {
    if fallback == Fallback::Path {
        if let Some(bin) = crate::find_on_path(name) {
            eprintln!(
                "[llmman] warning: {err:#}; using {} from PATH",
                bin.display()
            );
            return Ok(bin);
        }
        return Err(err.context(format!("and no {name} on PATH to fall back to")));
    }
    Err(err)
}

/// llmman's own `uv` at [`default_uv_release`], downloaded and checked
/// against astral-sh's published `.sha256` if not already installed.
fn ensure_uv(marker: &DownloadMarker, fallback: Fallback) -> Result<PathBuf> {
    let version = default_uv_release();
    let root = uv_root()?;
    let dest = root.join(version);
    let cached = |dest: &Path| managed::completed_binary(dest, "uv");
    let managed = managed::ensure(&root, &dest, "uv", cached, |dest| {
        let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
        let asset = uv_asset_name(os, arch)
            .ok_or_else(|| anyhow!("no uv download for {os}/{arch} (mlx-lm needs macOS)"))?;
        let url = uv_download_url(version, &asset);
        eprintln!("[llmman] downloading uv {version} (to install mlx-lm): {asset}");
        marker.set_status(&format!("downloading {asset} (to install mlx-lm)"));
        let client = http_client()?;
        let expected = client
            .get(format!("{url}.sha256"))
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.text())
            .with_context(|| format!("fetch {asset}.sha256"))?;
        let expected =
            parse_sha256_file(&expected).ok_or_else(|| anyhow!("malformed {asset}.sha256"))?;
        let tmp = tmp_path(&asset)?;
        let _cleanup = RemoveOnDrop(&tmp);
        download_to_file(&client, &url, &tmp, &asset, Some(marker))?;
        let actual = sha256_of_file(&tmp)?;
        if actual != expected {
            anyhow::bail!("{asset}: sha256 {actual} does not match published {expected}");
        }
        marker.set_status(&format!("extracting {asset}"));
        extract_tar_gz(&tmp, dest)?;
        let uv = find_binary(dest, "uv")
            .with_context(|| format!("uv binary not found after extracting {asset}"))?;
        mark_executable(&uv)?;
        Ok(uv)
    });
    managed.or_else(|e| or_from_path(e, "uv", fallback))
}

/// Runs `uv` with `args`, output inherited. `python_dir` is where uv
/// keeps the interpreter it manages for this environment; the host's
/// own Pythons are never candidates.
fn run_uv(uv: &Path, python_dir: &Path, args: &[&OsStr]) -> Result<()> {
    crate::debug_log!("running {} {:?}", uv.display(), args);
    let mut cmd = std::process::Command::new(uv);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .env("UV_PYTHON_INSTALL_DIR", python_dir)
        .env("UV_MANAGED_PYTHON", "1");
    if !std::io::stderr().is_terminal() {
        cmd.env("UV_NO_PROGRESS", "1");
    }
    let status = cmd
        .status()
        .with_context(|| format!("spawn {}", uv.display()))?;
    if !status.success() {
        anyhow::bail!("{} {:?} exited with {status}", uv.display(), args);
    }
    Ok(())
}

/// The `mlx_lm.server` to run: llmman's own for the current pins,
/// installed first if needed — or, if that fails and `fallback` allows,
/// the one on `PATH`. Cheap once installed, so `llmman serve` calls it
/// on every startup. Blocking: call via `spawn_blocking`.
pub fn ensure_mlx_server(fallback: Fallback) -> Result<PathBuf> {
    let root = mlx_lm_root()?;
    let dir = managed_dir()?;
    let managed = managed::ensure(&root, &dir, "mlx-lm", completed_server, |dir| {
        let requirement = format!("mlx-lm=={}", mlx_lm_version());
        let python = default_python_release();
        let python_dir = dir.join("python");
        let venv = dir.join("venv");
        // Held, with its heartbeat, for the whole install: `uv` runs for
        // minutes with no progress this process can see.
        let marker = DownloadMarker::create();
        marker.keep_alive_during(|| -> Result<()> {
            let uv = ensure_uv(&marker, fallback)?;
            eprintln!(
                "[llmman] installing {requirement} (CPython {python}) into {}",
                dir.display()
            );
            marker.set_status(&format!(
                "installing {requirement}: fetching CPython {python} and creating its environment"
            ));
            run_uv(
                &uv,
                &python_dir,
                &[
                    "venv".as_ref(),
                    "--python".as_ref(),
                    python.as_ref(),
                    venv.as_os_str(),
                ],
            )?;
            marker.set_status(&format!("installing {requirement}: uv pip install"));
            run_uv(
                &uv,
                &python_dir,
                &[
                    "pip".as_ref(),
                    "install".as_ref(),
                    "--python".as_ref(),
                    venv.join("bin").join("python").as_os_str(),
                    requirement.as_ref(),
                ],
            )
        })?;
        let server = server_in(dir);
        if !server.is_file() {
            anyhow::bail!(
                "{} missing after installing {requirement}",
                server.display()
            );
        }
        eprintln!("[llmman] installed {requirement}: {}", server.display());
        Ok(server)
    });
    managed.or_else(|e| or_from_path(e, "mlx_lm.server", fallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_are_read_from_the_repo_root_files() {
        let numeric = |v: &str| v.split('.').all(|part| part.parse::<u32>().is_ok());
        assert!(numeric(default_uv_release()), "{}", default_uv_release());
        assert!(
            numeric(default_mlx_lm_release()),
            "{}",
            default_mlx_lm_release()
        );
        let (major, minor) = default_python_release().split_once('.').unwrap();
        assert_eq!(major, "3");
        assert!(minor.parse::<u32>().unwrap() >= 10, "mlx needs 3.10+");
    }

    #[test]
    fn version_dir_names_both_pins() {
        assert_eq!(mlx_lm_dir_name("0.31.3", "3.13"), "0.31.3-py3.13");
    }

    #[test]
    fn uv_download_is_pinned_not_latest() {
        let url = uv_download_url("0.12.13", "uv-aarch64-apple-darwin.tar.gz");
        assert_eq!(
            url,
            "https://github.com/astral-sh/uv/releases/download/0.12.13/uv-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    fn uv_asset_exists_only_for_macos_on_a_supported_arch() {
        assert_eq!(
            uv_asset_name("macos", "aarch64").as_deref(),
            Some("uv-aarch64-apple-darwin.tar.gz")
        );
        assert_eq!(
            uv_asset_name("macos", "x86_64").as_deref(),
            Some("uv-x86_64-apple-darwin.tar.gz")
        );
        assert_eq!(uv_asset_name("linux", "aarch64"), None);
        assert_eq!(uv_asset_name("macos", "riscv64"), None);
    }

    #[test]
    fn sha256_file_parses_the_leading_digest_only() {
        let hex = "a".repeat(64);
        assert_eq!(
            parse_sha256_file(&format!(
                "{}  uv-aarch64-apple-darwin.tar.gz\n",
                hex.to_uppercase()
            )),
            Some(hex.clone())
        );
        assert_eq!(parse_sha256_file("not a digest"), None);
        assert_eq!(parse_sha256_file(""), None);
    }

    #[test]
    fn path_fallback_only_when_asked_and_present() {
        let err = || anyhow!("managed install failed");
        assert!(or_from_path(err(), "sh", Fallback::None).is_err());
        let missing = or_from_path(err(), "no-such-binary-llmman", Fallback::Path).unwrap_err();
        assert!(format!("{missing:#}").contains("no no-such-binary-llmman on PATH"));
        // `sh` is on every Unix PATH.
        if cfg!(unix) {
            assert!(or_from_path(err(), "sh", Fallback::Path).is_ok());
        }
    }

    /// Network + several hundred MB: `cargo test -- --ignored
    /// ensure_mlx_server_installs`. macOS only.
    #[test]
    #[ignore]
    fn ensure_mlx_server_installs_a_runnable_console_script() {
        let bin = ensure_mlx_server(Fallback::None).expect("ensure_mlx_server");
        let out = std::process::Command::new(&bin)
            .arg("--help")
            .output()
            .expect("run mlx_lm.server --help");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(installed());
    }
}
