use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use flate2::write::GzEncoder;
use flate2::Compression;

/// Extract every object from a (possibly GNU ar) static archive and repack
/// it as an MSVC-format LIB using lib.exe.  This is needed on Windows ARM64
/// because Go uses GNU 'ar' when it can't identify the C compiler as cl.exe,
/// but MSVC link.exe only accepts its own LIB format.
fn repack_as_msvc_lib(lib_path: &Path, out_dir: &Path) {
    let extract_dir = out_dir.join("ar_extract");
    let _ = fs::remove_dir_all(&extract_dir);
    if fs::create_dir_all(&extract_dir).is_err() {
        return;
    }

    // llvm-ar can read both GNU ar and MSVC LIB archives
    let ok = Command::new("llvm-ar")
        .args(["x", lib_path.to_str().unwrap_or("")])
        .current_dir(&extract_dir)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("cargo:warning=llvm-ar extraction failed; keeping original archive");
        return;
    }

    let objs: Vec<PathBuf> = match fs::read_dir(&extract_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return,
    };
    if objs.is_empty() {
        return;
    }

    // lib.exe is the ARM64-native MSVC archiver; it infers machine type from objects
    let tmp = out_dir.join("_shim_repack.lib");
    let mut cmd = Command::new("lib.exe");
    cmd.arg("/nologo").arg(format!("/out:{}", tmp.display()));
    for obj in &objs {
        if let Some(s) = obj.to_str() {
            cmd.arg(s);
        }
    }

    if cmd.status().map(|s| s.success()).unwrap_or(false) {
        let _ = fs::rename(&tmp, lib_path);
        println!("cargo:warning=Repacked Go archive as MSVC LIB (ARM64)");
    } else {
        eprintln!("cargo:warning=lib.exe repack failed; keeping original archive");
    }
    let _ = fs::remove_dir_all(&extract_dir);
}

/// Emits `LLMMAN_VERSION`: the Cargo package version plus, when built from
/// a git checkout, the commit it was built from (e.g. "0.1.0 (a1b2c3d)").
/// Every nightly/CI build shares the same Cargo.toml version, so without
/// the commit suffix two different builds are indistinguishable to
/// `llmman --version` and the daemon's /api/version.
fn emit_version() {
    let pkg = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let describe = Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let version = match describe {
        Some(desc) => format!("{pkg} ({desc})"),
        None => pkg,
    };
    println!("cargo:rustc-env=LLMMAN_VERSION={version}");
    // `HEAD` only changes on a checkout/branch switch; an ordinary `git
    // commit` on the current branch instead updates `logs/HEAD` (the
    // reflog) — without watching that too, `--dirty`'s output (and so
    // LLMMAN_VERSION) can go stale across a commit, defeating
    // stale_daemon's version comparison in daemon.rs. Resolved via
    // `git rev-parse --git-path` rather than a hardcoded `.git/...` path
    // so this also reruns correctly from a linked worktree, where the
    // real HEAD/logs live under `.git/worktrees/<name>/` instead.
    for path in ["HEAD", "logs/HEAD"] {
        if let Some(resolved) = git_path(path) {
            println!("cargo:rerun-if-changed={}", resolved.display());
        }
    }
}

/// Resolves a path relative to the repo's git dir (e.g. `"HEAD"` or
/// `"logs/HEAD"`) via `git rev-parse --git-path`, so it's correct both for
/// a plain `.git` directory and a linked worktree's own `.git` file.
fn git_path(path: &str) -> Option<PathBuf> {
    Command::new("git")
        .args(["rev-parse", "--git-path", path])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
}

fn main() {
    emit_version();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shim_dir = manifest_dir.join("go-shim");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Determine backend build tags from Cargo features.
    // For the podman backend we add two extra tags to avoid pulling in C library
    // dependencies (libgpgme, libbtrfs, libdevmapper) that are not present on CI
    // runners and are not needed for the docker:// and oci: transports we use:
    //   containers_image_openpgp  — use Go's native OpenPGP instead of libgpgme
    //   exclude_graphdriver_btrfs — omit the btrfs storage driver (needs libbtrfs-dev)
    //   exclude_graphdriver_devicemapper — omit the device-mapper driver
    let go_tags = if env::var("CARGO_FEATURE_PODMAN").is_ok() {
        "podman,containers_image_openpgp,exclude_graphdriver_btrfs,exclude_graphdriver_devicemapper"
    } else {
        "docker"
    };

    // Always build natively — each platform's CI runner builds its own binary.
    // CGO requires the host C toolchain, which is always present on native runners.
    //
    // Only *-pc-windows-msvc uses the MSVC "NAME.lib" naming; *-pc-windows-gnu
    // is a GNU-ABI target like Linux/macOS and needs "libNAME.a". The final
    // link step (mingw's ld) tolerates either name on a GNU target, but the
    // `llmman` lib crate's own compilation checks a `#[link(kind =
    // "static")]` dependency's presence itself first, by each target's
    // canonical name, with no such leniency.
    let lib_name = if target_os == "windows" && target_env == "msvc" {
        "llmman_shim.lib"
    } else {
        "libllmman_shim.a"
    };
    let lib_path = out_dir.join(lib_name);

    // Ensure Go module dependencies are present
    let _ = Command::new("go")
        .current_dir(&shim_dir)
        .args(["mod", "download"])
        .status();

    let mut cmd = Command::new("go");
    cmd.current_dir(&shim_dir)
        .env("CGO_ENABLED", "1")
        .arg("build")
        .arg(format!("-tags={}", go_tags))
        .arg("-buildmode=c-archive")
        .arg("-o")
        .arg(&lib_path)
        .arg(".");

    // On *-pc-windows-msvc targets the Rust linker (lld-link, set via
    // RUSTFLAGS in CI) requires MSVC-ABI COFF objects from Go's CGO.
    //
    // clang in GCC-driver mode is the only C compiler that:
    //   • accepts all GCC-style flags Go passes (-Werror, -dM, -fno-stack-protector)
    //   • produces MSVC-compatible COFF when given the right --target
    //
    // We cannot rely on CGO_CFLAGS to pass --target because Go's CGO security
    // filter may strip unrecognised flags before they reach clang.  Instead we
    // write a tiny .cmd wrapper that hard-codes --target as part of the CC
    // command itself; this is unconditional and cannot be filtered out.
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let msvc_triple = match arch.as_str() {
            "x86_64" => "x86_64-pc-windows-msvc",
            "aarch64" => "aarch64-pc-windows-msvc",
            other => panic!("unsupported Windows MSVC arch: {}", other),
        };
        // The wrapper calls clang with a fixed --target then forwards all other
        // args (%*).  Go treats this .cmd as the C compiler.
        let wrapper = out_dir.join("cgo_cc.cmd");
        fs::write(
            &wrapper,
            format!("@echo off\r\nclang --target={} %*\r\n", msvc_triple),
        )
        .expect("write CGO CC wrapper");
        cmd.env("CC", &wrapper);
    }

    // Align the Go shim's minimum macOS version with Rust's aarch64-apple-darwin
    // deployment target (11.0).  Without this Go defaults to the SDK version
    // (15.x on macos-15 runners), producing objects that emit "built for newer
    // macOS" warnings and may reference symbols gated behind the newer version.
    if target_os == "macos" {
        cmd.env("MACOSX_DEPLOYMENT_TARGET", "11.0");
    }

    let status = cmd
        .status()
        .expect("Failed to invoke `go build` — is Go (1.22+) installed and on PATH?");

    if !status.success() {
        panic!("Go shim build failed for tags={}", go_tags);
    }

    // On Windows MSVC + aarch64: Go doesn't recognise our clang wrapper as
    // MSVC (it checks the binary name for "cl.exe"), so it archives the CGO
    // objects with GNU 'ar' instead of MSVC 'lib.exe'.  GNU ar archives are
    // rejected by MSVC link.exe with LNK4003.
    //
    // Fix: extract every object from the archive with llvm-ar (which reads
    // both GNU ar and MSVC LIB), then repack with the ARM64-native lib.exe
    // to produce a proper MSVC LIB.  lib.exe infers the machine type from
    // the objects so this is safe even when run unconditionally.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
        && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
    {
        repack_as_msvc_lib(&lib_path, &out_dir);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=llmman_shim");

    // Platform-specific link dependencies required by Go runtime and shim libraries
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=Security");
            println!("cargo:rustc-link-lib=framework=SystemConfiguration");
            // The podman backend (and Go's CGO net resolver in general) references
            // res_9_ninit / res_9_nclose / res_9_nsearch from libresolv.
            println!("cargo:rustc-link-lib=resolv");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=bcrypt");
            println!("cargo:rustc-link-lib=ws2_32");
            println!("cargo:rustc-link-lib=userenv");
            // With CC=cl the CGO objects are compiled by MSVC which links the CRT
            // automatically; legacy_stdio_definitions is not needed and causes
            // LNK4078 / LNK1223 when mixed with MSVC-format objects.
        }
        _ => {}
    }

    println!("cargo:rerun-if-changed=go-shim/");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PODMAN");

    // ── Gzip web UI assets for embedding ──────────────────────────────────
    let webui_src = manifest_dir.join("webui");
    let webui_out = out_dir.join("webui_gz");
    fs::create_dir_all(&webui_out).expect("create webui_gz dir");

    for name in &["index.html", "bundle.js", "bundle.css", "loading.html"] {
        let src = webui_src.join(name);
        let dst = webui_out.join(format!("{name}.gz"));
        let data = fs::read(&src).unwrap_or_else(|e| panic!("read webui/{name}: {e}"));
        let mut enc = GzEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&data).expect("gzip write");
        let compressed = enc.finish().expect("gzip finish");
        fs::write(&dst, &compressed).unwrap_or_else(|e| panic!("write {name}.gz: {e}"));
    }

    println!("cargo:rerun-if-changed=webui/");
}
