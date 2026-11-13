//! Whole-file writes a reader never sees half done, writes only their
//! owner can read, the free space a download needs, and the SHA-256 of a
//! file or of what is written through a writer.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the temp files of concurrent writers in one process; the
/// pid does the same across processes.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` through a temp file beside it and a rename,
/// so a reader sees the old file or the new one and not a partial one;
/// the temp name carries the pid and a counter, so no two writers share
/// one.
/// A symlink whose target exists is written through to the target, the
/// old file's permissions go on the temp before it holds anything, and
/// the temp is synced before the rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let path = dunce::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    replace_through_temp(&path, bytes, 0o666, |file| match std::fs::metadata(&path) {
        Ok(meta) => file.set_permissions(meta.permissions()),
        Err(_) => Ok(()),
    })
}

/// [`write_atomic`], but the new file is readable only by its owner on
/// Unix, whatever the old one allowed. The rename needs only the
/// directory to be writable, so it also replaces a file this process
/// could not open for writing; a symlink at `path` is replaced, not
/// written through.
pub fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    replace_through_temp(path, bytes, 0o600, |_file| {
        #[cfg(unix)]
        _file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
        Ok(())
    })
}

/// The temp is created with `create_mode` on Unix (less the umask): a
/// mode set afterwards would not reach a reader that opened it before.
/// `prepare` runs on the temp before it holds anything.
fn replace_through_temp(
    path: &Path,
    bytes: &[u8],
    create_mode: u32,
    prepare: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut tmp = path.to_path_buf().into_os_string();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, create_mode);
        #[cfg(not(unix))]
        let _ = create_mode;
        let mut file = options.open(&tmp)?;
        prepare(&file)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Fails when `needed` bytes will not fit in the `available` bytes free
/// at `dir`, before `what` is pulled there. Free space that could not be
/// read passes: running out still fails the write, as it always has.
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

/// The bytes this user may still write to the filesystem holding `dir`,
/// or `None` where that can't be read (or a filesystem reports no size,
/// as some FUSE ones do).
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

/// Copies `reader` into `writer` and returns the byte count and the
/// lowercase hex SHA-256 of the bytes copied. Hashes without copying when
/// `writer` is [`std::io::sink`].
pub fn copy_hashing(
    reader: &mut impl Read,
    writer: &mut impl Write,
) -> std::io::Result<(u64, String)> {
    let mut hashing_writer = HashingWriter::new(writer);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hashing_writer.write_all(&buf[..n])?;
    }
    Ok(hashing_writer.finish())
}

/// Passes writes to `inner`, hashing the bytes `inner` accepts and reporting
/// each accepted count to `on_written`.
pub struct HashingWriter<W, F = fn(usize)> {
    inner: W,
    hasher: sha2::Sha256,
    size: u64,
    on_written: F,
}

impl<W> HashingWriter<W> {
    pub fn new(inner: W) -> Self {
        Self::with_progress(inner, |_| {})
    }
}

impl<W, F: FnMut(usize)> HashingWriter<W, F> {
    pub fn with_progress(inner: W, on_written: F) -> Self {
        use sha2::Digest as _;
        Self {
            inner,
            hasher: sha2::Sha256::new(),
            size: 0,
            on_written,
        }
    }

    /// The byte count and lowercase hex SHA-256 of everything written.
    pub fn finish(self) -> (u64, String) {
        use sha2::Digest as _;
        (self.size, hex::encode(self.hasher.finalize()))
    }

    fn record(&mut self, written: &[u8]) {
        use sha2::Digest as _;
        self.hasher.update(written);
        self.size += written.len() as u64;
        (self.on_written)(written.len());
    }
}

impl<W: Write, F: FnMut(usize)> Write for HashingWriter<W, F> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.record(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl<W, F> tokio::io::AsyncWrite for HashingWriter<W, F>
where
    W: tokio::io::AsyncWrite + Unpin,
    F: FnMut(usize) + Unpin,
{
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let n = std::task::ready!(std::pin::Pin::new(&mut self.inner).poll_write(cx, buf))?;
        self.record(&buf[..n]);
        std::task::Poll::Ready(Ok(n))
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Lowercase hex SHA-256 of the file at `path`.
pub fn sha256_of_file(path: &Path) -> anyhow::Result<String> {
    use anyhow::Context as _;
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let (_, hex) = copy_hashing(&mut file, &mut std::io::sink())
        .with_context(|| format!("hash {}", path.display()))?;
    Ok(hex)
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

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn copy_hashing_copies_every_byte_and_hashes_them() {
        let mut copied = Vec::new();
        let (size, hex) = copy_hashing(&mut &b"abc"[..], &mut copied).unwrap();
        assert_eq!(
            (size, hex.as_str(), copied.as_slice()),
            (3, ABC_SHA256, &b"abc"[..])
        );
    }

    #[test]
    fn copy_hashing_retries_an_interrupted_read() {
        struct InterruptedOnce<'a>(bool, &'a [u8]);
        impl Read for InterruptedOnce<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !std::mem::replace(&mut self.0, true) {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                self.1.read(buf)
            }
        }
        let mut reader = InterruptedOnce(false, b"abc");
        let (size, hex) = copy_hashing(&mut reader, &mut std::io::sink()).unwrap();
        assert_eq!((size, hex.as_str()), (3, ABC_SHA256));
    }

    #[test]
    fn copy_hashing_returns_the_writers_error() {
        struct DiskFull;
        impl Write for DiskFull {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let err = copy_hashing(&mut &b"abc"[..], &mut DiskFull).unwrap_err();
        assert_eq!(
            (err.kind(), err.to_string()),
            (std::io::ErrorKind::Other, "disk full".to_string())
        );
    }

    #[test]
    fn sha256_of_file_hashes_the_file_and_names_a_missing_one() {
        let dir = temp_dir("sha256");
        let path = dir.join("f.bin");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_of_file(&path).unwrap(), ABC_SHA256);
        let missing = dir.join("missing.bin");
        let err = format!("{:#}", sha256_of_file(&missing).unwrap_err());
        assert!(err.contains(&missing.display().to_string()), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hashing_writer_hashes_and_reports_only_what_the_inner_writer_accepts() {
        struct TwoBytesAtATime(Vec<u8>);
        impl Write for TwoBytesAtATime {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let n = buf.len().min(2);
                self.0.extend_from_slice(&buf[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut inner = TwoBytesAtATime(Vec::new());
        let mut reported = Vec::new();
        let mut writer = HashingWriter::with_progress(&mut inner, |n| reported.push(n));
        writer.write_all(b"abc").unwrap();
        assert_eq!(writer.finish(), (3, ABC_SHA256.to_string()));
        assert_eq!((inner.0.as_slice(), reported), (&b"abc"[..], vec![2, 1]));
    }

    #[tokio::test]
    async fn hashing_writer_hashes_async_writes_and_flushes_them_through() {
        use tokio::io::AsyncWriteExt as _;
        let dir = temp_dir("hashing-writer");
        let path = dir.join("f.bin");
        let file = tokio::fs::File::create(&path).await.unwrap();
        let mut writer = HashingWriter::new(tokio::io::BufWriter::new(file));
        writer.write_all(b"a").await.unwrap();
        writer.write_all(b"bc").await.unwrap();
        writer.flush().await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
        assert_eq!(writer.finish(), (3, ABC_SHA256.to_string()));
        let _ = std::fs::remove_dir_all(&dir);
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

    #[cfg(unix)]
    #[test]
    fn write_private_makes_a_new_file_and_an_existing_one_0600() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("private");
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let new = dir.join("new");
        write_private(&new, b"secret").unwrap();
        assert_eq!(mode(&new), 0o600);
        assert_eq!(std::fs::read(&new).unwrap(), b"secret");

        let existing = dir.join("existing");
        std::fs::write(&existing, b"a longer old secret").unwrap();
        std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o644)).unwrap();
        write_private(&existing, b"new").unwrap();
        assert_eq!(mode(&existing), 0o600);
        assert_eq!(std::fs::read(&existing).unwrap(), b"new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_temp_has_its_create_mode_before_prepare_runs() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("create-mode");
        let mut seen = 0;
        replace_through_temp(&dir.join("token"), b"x", 0o600, |file| {
            seen = file.metadata()?.permissions().mode() & 0o777;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen & 0o077, 0, "group and others: {seen:o}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn write_private_replaces_a_file_it_cannot_open_and_keeps_it_when_the_dir_refuses() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("private-replace");
        let path = dir.join("token");
        let chmod = |p: &Path, mode| {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap()
        };
        std::fs::write(&path, b"stale").unwrap();
        chmod(&path, 0o444);
        write_private(&path, b"fresh").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"fresh");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        // Root ignores permissions, so only another user can see a refusal.
        if unsafe { libc::geteuid() } != 0 {
            chmod(&dir, 0o555);
            let refused = write_private(&path, b"newer");
            chmod(&dir, 0o755);
            assert!(refused.is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"fresh");
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no temp left");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
