//! The one shape every backend llmman installs for itself has on disk —
//! llama.cpp ([`crate::llama_release`]), `uv` and `mlx-lm`
//! ([`crate::mlx_release`]):
//!
//! ```text
//! <data_root>/<tool>/.lock                        one installer at a time
//! <data_root>/<tool>/<version>/                   one directory per pin
//! <data_root>/<tool>/<version>/.llmman-complete   written last
//! ```
//!
//! [`ensure`] is the whole protocol: complete already → done; else take
//! the lock, re-check (the other installer may have finished), wipe a
//! partial directory, install, write the sentinel, delete the other
//! versions under the root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const COMPLETE_SENTINEL: &str = ".llmman-complete";

/// Whether the version directory at `dir` finished installing.
pub(crate) fn is_complete(dir: &Path) -> bool {
    dir.join(COMPLETE_SENTINEL).is_file()
}

/// The binary `name` inside a complete `dir`, if both exist.
pub(crate) fn completed_binary(dir: &Path, name: &str) -> Option<PathBuf> {
    is_complete(dir)
        .then(|| crate::llama_release::find_binary(dir, name))
        .flatten()
}

/// Installs into `dest` (a version directory under `root`, possibly
/// nested one level deeper) unless `cached` already finds it complete.
/// `install` receives `dest`, empty; its result is returned once the
/// sentinel is written and the other versions under `root` are removed.
pub(crate) fn ensure<T>(
    root: &Path,
    dest: &Path,
    what: &str,
    cached: impl Fn(&Path) -> Option<T>,
    install: impl FnOnce(&Path) -> Result<T>,
) -> Result<T> {
    if let Some(found) = cached(dest) {
        return Ok(found);
    }
    let _lock = InstallLock::acquire(root, what)?;
    if let Some(found) = cached(dest) {
        return Ok(found);
    }
    if dest.exists() {
        std::fs::remove_dir_all(dest)
            .with_context(|| format!("remove partial install {}", dest.display()))?;
    }
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let out = install(dest)?;
    std::fs::write(dest.join(COMPLETE_SENTINEL), b"")
        .with_context(|| format!("write {}", dest.join(COMPLETE_SENTINEL).display()))?;
    prune_others(root, dest);
    Ok(out)
}

/// Deletes every version directory under `root` except the one holding
/// `keep`, skipping dot-entries and `tmp`. Best-effort: a running binary
/// on Windows cannot be removed, and the new install is unaffected.
fn prune_others(root: &Path, keep: &Path) {
    let keep_top = keep
        .strip_prefix(root)
        .ok()
        .and_then(|rel| rel.components().next())
        .map(|c| root.join(c));
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !path.is_dir()
            || name.starts_with('.')
            || name == "tmp"
            || Some(&path) == keep_top.as_ref()
        {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => eprintln!("[llmman] removed superseded {}", path.display()),
            Err(e) => eprintln!(
                "[llmman] warning: could not remove superseded {}: {e}",
                path.display()
            ),
        }
    }
}

/// An exclusive advisory lock on `<root>/.lock`, released on drop.
struct InstallLock {
    _file: std::fs::File,
}

impl InstallLock {
    fn acquire(root: &Path, what: &str) -> Result<InstallLock> {
        std::fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
        let path = root.join(".lock");
        let file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                eprintln!(
                    "[llmman] another llmman process is installing {what}; waiting for it to finish"
                );
                file.lock()
                    .with_context(|| format!("lock {}", path.display()))?;
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("lock {}", path.display()));
            }
        }
        Ok(InstallLock { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llmman-managed-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ensure_installs_marks_complete_and_prunes_the_rest() {
        let root = temp_root("ensure");
        for old in ["v1", "tmp", ".hidden"] {
            std::fs::create_dir_all(root.join(old)).unwrap();
        }
        std::fs::write(root.join("stray-file"), b"").unwrap();
        let dest = root.join("v2").join("label");
        // A partial v2 (no sentinel) must be wiped, not trusted.
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("stale"), b"").unwrap();

        let cached = |d: &Path| completed_binary(d, "tool");
        let installed = ensure(&root, &dest, "test", cached, |d| {
            assert!(!d.join("stale").exists(), "partial install was kept");
            std::fs::write(d.join("tool"), b"")?;
            Ok(d.join("tool"))
        })
        .unwrap();
        assert!(is_complete(&dest));
        assert_eq!(cached(&dest), Some(installed));

        let mut left: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, [".hidden", ".lock", "stray-file", "tmp", "v2"]);

        // Complete: the installer is not called again.
        ensure(&root, &dest, "test", cached, |_| panic!("reinstalled")).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn ensure_failure_leaves_no_sentinel_so_the_next_call_retries() {
        let root = temp_root("fail");
        let dest = root.join("v1");
        let cached = |d: &Path| completed_binary(d, "tool");
        assert!(ensure(&root, &dest, "test", cached, |_| Err::<PathBuf, _>(
            anyhow::anyhow!("boom")
        ))
        .is_err());
        assert!(!is_complete(&dest));
        ensure(&root, &dest, "test", cached, |d| {
            std::fs::write(d.join("tool"), b"")?;
            Ok(d.join("tool"))
        })
        .unwrap();
        assert!(is_complete(&dest));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lock_is_exclusive_until_dropped() {
        let root = temp_root("lock");
        let first = InstallLock::acquire(&root, "test").unwrap();
        let probe = std::fs::File::create(root.join(".lock")).unwrap();
        assert!(matches!(
            probe.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));
        drop(first);
        assert!(probe.try_lock().is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }
}
