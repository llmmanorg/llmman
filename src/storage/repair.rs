//! Blob-store startup repair — llmman's equivalent of Ollama's
//! `server/fixblobs.go`, run unconditionally at the start of every
//! `llmman serve`, before it starts listening (matching `server.Serve`'s
//! own unconditional `fixBlobs(blobsDir)` call).
//!
//! Deliberately cheap, matching Ollama's own approach: removes stray temp
//! files left behind by an interrupted blob write (`OciStore`'s
//! `tmp-<pid>`/`<hex>.tmp` naming), and reports any locally tagged image
//! whose manifest references a blob that isn't actually present. Neither
//! needs reading a blob's contents, so this stays fast even on a large
//! store — unlike a full sha256 re-verification of every blob, which is
//! expensive enough to be left out of every-startup checking entirely.

use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::Context;

use super::OciStore;

/// Minimum age a stray temp file must have before it's treated as
/// abandoned rather than a write still in progress — matches Ollama's own
/// `layerPruneGracePeriod` (`server/images.go`).
const STALE_TMP_FILE_AGE: Duration = Duration::from_secs(60 * 60);

/// Runs the startup repair pass described in the module doc comment.
/// Mirrors `fixBlobs`'s own error handling: a hard failure reading the
/// blobs directory propagates (failing `llmman serve` startup, like
/// Ollama's `Serve` does on `fixBlobs`'s error), but a single bad entry
/// is only logged and skipped.
pub fn repair_store(store_root: &Path) -> anyhow::Result<()> {
    // store_root itself may not exist yet (a fresh install, before the
    // first pull ever creates it) — this whole pass is a no-op then.
    if !store_root.is_dir() {
        return Ok(());
    }
    let blobs_dir = store_root.join("blobs").join("sha256");
    if blobs_dir.is_dir() {
        remove_stale_temp_files(&blobs_dir)
            .with_context(|| format!("read {}", blobs_dir.display()))?;
    }
    remove_stale_scratch_dirs(store_root)
        .with_context(|| format!("read {}", store_root.display()))?;
    report_incomplete_images(store_root);
    Ok(())
}

/// Removes leftover `.llmman-pull-*`/`.llmman-push-*` scratch directories
/// (go-shim/backend_podman.go's pushToRegistry/pullToLayout) from a
/// killed daemon or crashed pull/push — normally removed by their own
/// caller when it returns. Age-gated like `remove_stale_temp_files`: a
/// concurrent pull/push may have one of these open right now.
fn remove_stale_scratch_dirs(store_root: &Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(store_root)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !(name.starts_with(".llmman-pull-") || name.starts_with(".llmman-push-")) {
            continue;
        }
        if !is_stale(&path) {
            continue;
        }
        if let Err(e) = std::fs::remove_dir_all(&path) {
            eprintln!(
                "[llmman] couldn't remove stale scratch dir {}: {e:#}",
                path.display()
            );
            continue;
        }
        eprintln!("[llmman] removed stale scratch dir {name}");
    }
    Ok(())
}

fn remove_stale_temp_files(blobs_dir: &Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(blobs_dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[llmman] couldn't read blob store entry: {e:#}");
                continue;
            }
        };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !(name.starts_with("tmp-") || name.ends_with(".tmp")) {
            continue;
        }
        if !is_stale(&path) {
            continue;
        }
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!(
                "[llmman] couldn't remove stale temp file {}: {e:#}",
                path.display()
            );
            continue;
        }
        eprintln!("[llmman] removed stale temp file {name}");
    }
    Ok(())
}

/// True if `path`'s last modification is older than `STALE_TMP_FILE_AGE`.
/// A file whose metadata can't be read (removed out from under us,
/// unsupported mtime, ...) is treated as not-yet-stale rather than
/// erroring the whole pass.
/// True if `path` (a file or, for a scratch directory, everything
/// directly inside it too) hasn't been touched within the grace period.
/// A directory's own mtime only changes when an entry is added/removed/
/// renamed — a single long-running write to a file already inside it
/// (e.g. go.podman.io/image's own in-progress blob temp file, written
/// straight into a scratch directory's root) never bumps it, so a
/// directory is only stale once nothing directly inside it is fresh
/// either.
fn is_stale(path: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(path) {
        if !mtime_is_stale(path) {
            return false;
        }
        return entries.flatten().all(|e| mtime_is_stale(&e.path()));
    }
    mtime_is_stale(path)
}

fn mtime_is_stale(path: &Path) -> bool {
    let Ok(modified) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return false;
    };
    SystemTime::now()
        .duration_since(modified)
        .is_ok_and(|age| age >= STALE_TMP_FILE_AGE)
}

/// Logs every locally tagged image whose manifest or layers reference a
/// blob that isn't actually present in `store_root`, so a user notices
/// (in `llmman serve`'s own log) which images need re-pulling rather than
/// only discovering it later when loading one of them fails. Errors
/// opening the store or listing images are swallowed rather than failing
/// startup — this is a best-effort diagnostic, not a precondition for
/// serving.
fn report_incomplete_images(store_root: &Path) {
    let Ok(store) = OciStore::open(store_root) else {
        return;
    };
    let Ok(images) = store.list() else {
        return;
    };
    for img in images {
        let complete = store
            .find(&img.reference)
            .and_then(|desc| store.read_manifest(&desc.digest))
            .map(|manifest| {
                std::iter::once(&manifest.config)
                    .chain(manifest.layers.iter())
                    .all(|l| blob_exists(store_root, &l.digest))
            })
            .unwrap_or(false);
        if !complete {
            eprintln!(
                "[llmman] {} is missing one or more blobs and should be re-pulled",
                img.reference
            );
        }
    }
}

/// True if `digest` ("sha256:<hex>") names a blob file that's actually
/// present under `store_root`. Malformed digests (no `sha256:` prefix —
/// shouldn't happen for anything llmman itself ever wrote, but a
/// hand-edited or foreign-tool-written manifest could) count as "not
/// present" rather than panicking.
fn blob_exists(store_root: &Path, digest: &str) -> bool {
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    store_root.join("blobs").join("sha256").join(hex).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `llmman serve` calls `repair_store` before the
    /// store directory has necessarily ever been created (a fresh
    /// install, before the first pull) — it must be a no-op, not an
    /// error that fails startup.
    #[test]
    fn repair_store_is_a_no_op_when_the_store_root_does_not_exist_yet() {
        let dir =
            std::env::temp_dir().join(format!("llmman-repair-test-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!dir.exists());

        assert!(repair_store(&dir).is_ok());
    }

    #[test]
    fn blob_exists_requires_the_sha256_prefix_and_a_real_file() {
        let dir = std::env::temp_dir().join(format!("llmman-repair-test-{}", std::process::id()));
        let blobs = dir.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("abc123"), b"hi").unwrap();

        assert!(blob_exists(&dir, "sha256:abc123"));
        assert!(!blob_exists(&dir, "sha256:does-not-exist"));
        assert!(!blob_exists(&dir, "abc123")); // missing "sha256:" prefix
        assert!(!blob_exists(&dir, "md5:abc123")); // wrong algorithm prefix

        std::fs::remove_dir_all(&dir).unwrap();
    }

    // filetime_set (and its libc_timeval/set_file_times helpers) are
    // Unix-only — gate the whole test rather than porting them, since
    // this repo has no other Windows-specific mtime-backdating need.
    #[test]
    #[cfg(unix)]
    fn remove_stale_temp_files_only_removes_files_past_the_grace_period() {
        let dir =
            std::env::temp_dir().join(format!("llmman-repair-test-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fresh = dir.join("tmp-12345");
        let stale = dir.join("tmp-99999");
        std::fs::write(&fresh, b"in progress").unwrap();
        std::fs::write(&stale, b"abandoned").unwrap();
        // Back-date the "stale" file's mtime past the grace period.
        let old = SystemTime::now() - STALE_TMP_FILE_AGE - Duration::from_secs(60);
        filetime_set(&stale, old);

        remove_stale_temp_files(&dir).unwrap();

        assert!(fresh.exists(), "fresh temp file should be left alone");
        assert!(!stale.exists(), "stale temp file should be removed");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn remove_stale_scratch_dirs_only_removes_dirs_past_the_grace_period() {
        let dir =
            std::env::temp_dir().join(format!("llmman-repair-test-scratch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let active = dir.join(".llmman-pull-abc123");
        let abandoned = dir.join(".llmman-push-def456");
        std::fs::create_dir_all(&active).unwrap();
        std::fs::create_dir_all(&abandoned).unwrap();
        let old = SystemTime::now() - STALE_TMP_FILE_AGE - Duration::from_secs(60);
        filetime_set(&abandoned, old);

        remove_stale_scratch_dirs(&dir).unwrap();

        assert!(
            active.exists(),
            "an active-looking scratch dir should be left alone"
        );
        assert!(!abandoned.exists(), "a stale scratch dir should be removed");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test: a directory's own mtime doesn't change just
    /// because a file already inside it is still being written to (only
    /// adding/removing/renaming an entry bumps it) — a single
    /// long-running blob download must not make its scratch directory
    /// look abandoned.
    #[test]
    #[cfg(unix)]
    fn remove_stale_scratch_dirs_leaves_a_dir_with_a_fresh_file_inside_alone() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-repair-test-scratch-fresh-child-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let in_progress = dir.join(".llmman-pull-abc123");
        std::fs::create_dir_all(&in_progress).unwrap();
        let old = SystemTime::now() - STALE_TMP_FILE_AGE - Duration::from_secs(60);
        filetime_set(&in_progress, old); // the directory itself looks old...

        // ...but a blob temp file inside it is still being written to.
        let blob_tmp = in_progress.join("oci-put-blob-xyz");
        std::fs::write(&blob_tmp, b"still downloading").unwrap();

        remove_stale_scratch_dirs(&dir).unwrap();

        assert!(
            in_progress.exists(),
            "a scratch dir with a fresh file inside must not be removed"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Minimal mtime-backdating helper — avoids pulling in a `filetime`
    /// dependency just for this one test.
    #[cfg(unix)]
    fn filetime_set(path: &Path, when: SystemTime) {
        let file = std::fs::File::open(path).unwrap();
        let duration = when.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let ts = libc_timeval(duration);
        set_file_times(&file, ts);
    }

    #[cfg(unix)]
    fn libc_timeval(d: Duration) -> (i64, i64) {
        (d.as_secs() as i64, d.subsec_micros() as i64)
    }

    #[cfg(unix)]
    fn set_file_times(file: &std::fs::File, (secs, micros): (i64, i64)) {
        use std::os::unix::io::AsRawFd;
        let times = [
            libc::timeval {
                tv_sec: secs as _,
                tv_usec: micros as _,
            },
            libc::timeval {
                tv_sec: secs as _,
                tv_usec: micros as _,
            },
        ];
        unsafe {
            libc::futimes(file.as_raw_fd(), times.as_ptr());
        }
    }
}
