//! Atomic file writes and pre-download free-space checks.

use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Symlinks are followed to their target; the target's permissions are
/// copied onto the temp file before any bytes are written. The temp file
/// is synced before the rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    // Distinguishes the temp files of concurrent writers in one process; the
    // pid does the same across processes.
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let path = dunce::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut tmp = path.clone().into_os_string();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        if let Ok(meta) = std::fs::metadata(&path) {
            file.set_permissions(meta.permissions())?;
        }
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Returns an error before downloading if `needed` bytes will not fit in
/// the space available at `dir`. Unreadable free space passes silently;
/// the write will fail naturally if space runs out.
pub fn ensure_room(
    what: &str,
    dir: &Path,
    needed: u64,
    available: Option<u64>,
) -> anyhow::Result<()> {
    match available {
        Some(free) if needed > free => anyhow::bail!(
            "not enough disk space to pull {what}: it needs {}, and {} is free at {}; \
             free some space (llmman rm) or set LLMMAN_MODELS to a store on a bigger disk",
            crate::fmt::human_size(needed),
            crate::fmt::human_size(free),
            dir.display()
        ),
        _ => Ok(()),
    }
}

/// Returns the bytes available to the current user on the filesystem that
/// contains `dir`, or `None` if free space can't be read or the filesystem
/// reports no size (e.g. some FUSE mounts).
pub fn available_space(dir: &Path) -> Option<u64> {
    #[cfg(all(unix, not(target_vendor = "apple")))]
    {
        use std::os::unix::ffi::OsStrExt as _;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `path` is NUL-terminated and `st` is written by the call.
        if unsafe { libc::statvfs(path.as_ptr(), st.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: statvfs returned 0, so it filled `st`.
        let st = unsafe { st.assume_init() };
        #[allow(clippy::unnecessary_cast)]
        return (st.f_blocks > 0).then(|| st.f_bavail as u64 * st.f_frsize as u64);
    }
    // statfs, not statvfs: Apple's statvfs has 32-bit block counts.
    #[cfg(target_vendor = "apple")]
    {
        use std::os::unix::ffi::OsStrExt as _;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: `path` is NUL-terminated and `st` is written by the call.
        if unsafe { libc::statfs(path.as_ptr(), st.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: statfs returned 0, so it filled `st`.
        let st = unsafe { st.assume_init() };
        return (st.f_blocks > 0).then(|| st.f_bavail * u64::from(st.f_bsize));
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetDiskFreeSpaceExW(
                directory: *const u16,
                free_to_caller: *mut u64,
                total: *mut u64,
                total_free: *mut u64,
            ) -> i32;
        }
        let path: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let (mut free, mut total) = (0u64, 0u64);
        // SAFETY: `path` is NUL-terminated; the outputs are plain u64s and
        // the last one may be null.
        let ok = unsafe {
            GetDiskFreeSpaceExW(path.as_ptr(), &mut free, &mut total, std::ptr::null_mut())
        };
        return (ok != 0 && total > 0).then_some(free);
    }
    #[allow(unreachable_code)]
    {
        let _ = dir;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-fsutil-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ensure_room_refuses_only_what_does_not_fit() {
        let dir = Path::new("/models");
        assert!(ensure_room("m", dir, 10, Some(10)).is_ok());
        assert!(
            ensure_room("m", dir, 11, None).is_ok(),
            "unknown free space passes"
        );
        let err = ensure_room("org/m", dir, 3 << 30, Some(1 << 30))
            .unwrap_err()
            .to_string();
        assert!(err.contains("org/m"), "{err}");
        assert!(err.contains(&crate::fmt::human_size(3 << 30)), "{err}");
        assert!(err.contains(&crate::fmt::human_size(1 << 30)), "{err}");
        assert!(err.contains("/models"), "{err}");
    }

    #[test]
    fn available_space_reads_the_temp_dirs_filesystem() {
        assert!(available_space(&std::env::temp_dir()).is_some_and(|free| free > 0));
    }

    #[test]
    fn write_atomic_replaces_the_content_and_leaves_no_temp_behind() {
        let dir = temp_dir("replace");
        let path = dir.join("f.tar.gz");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Writers in one process racing on one file each land whole.
    #[test]
    fn write_atomic_concurrent_writers_each_land_a_whole_file() {
        let dir = temp_dir("race");
        let path = dir.join("f.json");
        let threads: Vec<_> = (0..8u8)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || write_atomic(&path, &[i; 4096]).unwrap())
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let got = std::fs::read(&path).unwrap();
        assert_eq!(got.len(), 4096);
        assert!(got.iter().all(|b| *b == got[0]));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A dotfile kept as a symlink stays one: the target changes, the
    /// link does not, and the old mode comes along.
    #[cfg(unix)]
    #[test]
    fn write_atomic_writes_through_a_symlink_and_keeps_the_mode() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("symlink");
        let target = dir.join("real.json");
        let link = dir.join("link.json");
        std::fs::write(&target, b"old").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        write_atomic(&link, b"new").unwrap();
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
