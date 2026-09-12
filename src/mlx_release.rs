//! Installs and caches [`mlx-lm`](https://github.com/ml-explore/mlx-lm)'s
//! `mlx_lm.server` on Apple Silicon the way [`crate::llama_release`]
//! fetches `llama-server`: nothing needs to be on `PATH` first.
//!
//! `mlx-lm` is a Python package, so the installer is
//! [`uv`](https://github.com/astral-sh/uv): the one on `PATH`, else
//! astral-sh's prebuilt macOS release cached under `<data_root>/uv/`. It
//! creates `<data_root>/mlx-lm/venv` (fetching a managed CPython if the
//! host has none new enough) and `uv pip install`s `mlx-lm` into it. An
//! `mlx_lm.server` already on `PATH` always wins over the managed one.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::llama_release::{
    download_to_file, extract_tar_gz, find_binary, http_client, mark_executable, tmp_path,
    RemoveOnDrop,
};

/// `LLMMAN_MLX_LM_VERSION`: an exact `mlx-lm` release to install instead
/// of PyPI's current one. Read only while the environment is created.
pub const MLX_LM_VERSION_VAR: &str = "LLMMAN_MLX_LM_VERSION";

/// astral-sh's "latest release" redirect: no GitHub API call, so no
/// shared unauthenticated rate limit with `llama_release`.
const UV_LATEST_DOWNLOAD: &str = "https://github.com/astral-sh/uv/releases/latest/download";

/// `mlx` publishes wheels for CPython 3.10 and up; a range lets uv pick
/// the newest it can find or download, and skips Apple's own 3.9.
const PYTHON_REQUEST: &str = ">=3.10";

/// Written into the venv once `uv pip install` has succeeded; without it
/// a venv is an interrupted install and gets rebuilt.
const COMPLETE_SENTINEL: &str = ".llmman-complete";

/// `uv pip install <spec>`, pinned by `version` when given.
fn mlx_lm_requirement(version: Option<&str>) -> String {
    match version.map(str::trim).filter(|v| !v.is_empty()) {
        Some(v) => format!("mlx-lm=={v}"),
        None => "mlx-lm".to_string(),
    }
}

/// astral-sh's release asset for this host, or `None` off macOS — the
/// only platform `mlx` runs on.
fn uv_asset_name(os: &str, arch: &str) -> Option<String> {
    matches!((os, arch), ("macos", "aarch64" | "x86_64"))
        .then(|| format!("uv-{arch}-apple-darwin.tar.gz"))
}

fn mlx_lm_root() -> Result<PathBuf> {
    Ok(crate::data_root()?.join("mlx-lm"))
}

fn server_in_venv(venv: &Path) -> PathBuf {
    venv.join("bin").join("mlx_lm.server")
}

/// Reads a GitHub `<asset>.sha256` file (`<hex>  <name>`) into its hex
/// digest.
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

/// A usable `uv`: from `PATH`, from an earlier download, or freshly
/// downloaded (and checked against astral-sh's published `.sha256`).
/// Never re-fetched once cached: it only ever installs `mlx-lm`.
fn ensure_uv() -> Result<PathBuf> {
    if let Some(uv) = crate::find_on_path("uv") {
        return Ok(uv);
    }
    let root = crate::data_root()?.join("uv");
    if let Some(uv) = find_binary(&root, "uv") {
        return Ok(uv);
    }
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let asset = uv_asset_name(os, arch)
        .ok_or_else(|| anyhow!("no uv download for {os}/{arch} (mlx-lm needs macOS)"))?;
    let url = format!("{UV_LATEST_DOWNLOAD}/{asset}");
    eprintln!("[llmman] downloading uv (to install mlx-lm): {asset}");
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
    download_to_file(&client, &url, &tmp, &asset, None)?;
    let actual = sha256_of_file(&tmp)?;
    if actual != expected {
        anyhow::bail!("{asset}: sha256 {actual} does not match published {expected}");
    }
    extract_tar_gz(&tmp, &root)?;
    let uv = find_binary(&root, "uv")
        .with_context(|| format!("uv binary not found after extracting {asset}"))?;
    mark_executable(&uv)?;
    Ok(uv)
}

/// Runs `uv` with `args`, output inherited (the daemon's log).
fn run_uv(uv: &Path, args: &[&OsStr]) -> Result<()> {
    crate::debug_log!("running {} {:?}", uv.display(), args);
    let status = std::process::Command::new(uv)
        .args(args)
        .env("UV_NO_PROGRESS", "1")
        .stdin(std::process::Stdio::null())
        .status()
        .with_context(|| format!("spawn {}", uv.display()))?;
    if !status.success() {
        anyhow::bail!("{} {:?} exited with {status}", uv.display(), args);
    }
    Ok(())
}

/// The `mlx_lm.server` to run: the one on `PATH`, else the one in
/// llmman's managed environment, installed first if it isn't complete.
/// Blocking (network, disk, child `uv`): call via `spawn_blocking`.
pub fn ensure_mlx_server() -> Result<PathBuf> {
    if let Some(bin) = crate::find_on_path("mlx_lm.server") {
        return Ok(bin);
    }
    let root = mlx_lm_root()?;
    let venv = root.join("venv");
    let server = server_in_venv(&venv);
    let complete = || venv.join(COMPLETE_SENTINEL).is_file() && server.is_file();
    if complete() {
        return Ok(server);
    }

    // One installer at a time, across daemons and CLI processes alike.
    std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    let lock = std::fs::File::create(root.join(".lock")).context("create install lock")?;
    lock.lock().context("acquire install lock")?;
    if complete() {
        return Ok(server);
    }
    if venv.exists() {
        std::fs::remove_dir_all(&venv)
            .with_context(|| format!("remove partial {}", venv.display()))?;
    }

    let uv = ensure_uv()?;
    let requirement = mlx_lm_requirement(std::env::var(MLX_LM_VERSION_VAR).ok().as_deref());
    eprintln!("[llmman] installing {requirement} into {}", venv.display());
    run_uv(
        &uv,
        &[
            "venv".as_ref(),
            "--python".as_ref(),
            PYTHON_REQUEST.as_ref(),
            venv.as_os_str(),
        ],
    )?;
    run_uv(
        &uv,
        &[
            "pip".as_ref(),
            "install".as_ref(),
            "--python".as_ref(),
            venv.join("bin").join("python").as_os_str(),
            requirement.as_ref(),
        ],
    )?;
    if !server.is_file() {
        anyhow::bail!(
            "{} missing after installing {requirement}",
            server.display()
        );
    }
    std::fs::write(venv.join(COMPLETE_SENTINEL), b"").context("write completion sentinel")?;
    eprintln!("[llmman] installed {requirement}: {}", server.display());
    Ok(server)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirement_is_unpinned_without_a_version_and_pinned_with_one() {
        assert_eq!(mlx_lm_requirement(None), "mlx-lm");
        assert_eq!(mlx_lm_requirement(Some("  ")), "mlx-lm");
        assert_eq!(mlx_lm_requirement(Some(" 0.31.3 ")), "mlx-lm==0.31.3");
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

    /// Network + several hundred MB: `cargo test -- --ignored
    /// ensure_mlx_server_installs`. macOS only.
    #[test]
    #[ignore]
    fn ensure_mlx_server_installs_a_runnable_console_script() {
        let bin = ensure_mlx_server().expect("ensure_mlx_server");
        let out = std::process::Command::new(&bin)
            .arg("--help")
            .output()
            .expect("run mlx_lm.server --help");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
