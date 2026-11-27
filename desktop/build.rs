//! Downloads the tray icons into OUT_DIR, pinned by sha256. No binaries live
//! in git; desktop/generate-assets.py makes the files and prints the pins.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use sha2::{Digest, Sha256};

const BASE: &str = "https://github.com/llmmanorg/llmman/releases/download/docs-assets/";

/// (release asset, sha256); saved without the `llmman-desktop-` prefix.
const TRAY_ICONS: [(&str, &str); 2] = [
    (
        "llmman-desktop-tray-32.png",
        "ad4161792b4686f7c096b7a72248de14d9f33986cdae3c77d0365644a66c5b22",
    ),
    (
        "llmman-desktop-tray-64.png",
        "7568991c7efcca5d70ce9d5b88884f6550c362f16998e58f52799798d27b4969",
    ),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    for (asset, sha256) in TRAY_ICONS {
        fetch(asset, sha256, &out_dir);
    }
}

/// Fails the build on a download error or a hash mismatch: a tray icon that
/// is blank or wrong is worse than no build. `curl` is on every target.
fn fetch(asset: &str, sha256: &str, out_dir: &Path) {
    let dst = out_dir.join(asset.trim_start_matches("llmman-desktop-"));
    let hash_ok = |bytes: &[u8]| hex::encode(Sha256::digest(bytes)) == sha256;
    if fs::read(&dst).is_ok_and(|b| hash_ok(&b)) {
        return;
    }
    let url = format!("{BASE}{asset}");
    let tmp = dst.with_extension("part");
    let fetched = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "--max-time", "60", "-o"])
        .arg(&tmp)
        .arg(&url)
        .status()
        .is_ok_and(|s| s.success());
    let bytes = fetched.then(|| fs::read(&tmp).ok()).flatten();
    let _ = fs::remove_file(&tmp);
    let Some(bytes) = bytes else {
        panic!("could not download {url}; llmman-desktop's build needs network access and curl");
    };
    assert!(
        hash_ok(&bytes),
        "{url} does not match the sha256 pinned in desktop/build.rs ({sha256}); update the pin if it changed on purpose"
    );
    fs::write(&dst, bytes).unwrap_or_else(|e| panic!("write {}: {e}", dst.display()));
}
