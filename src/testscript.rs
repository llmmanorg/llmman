//! Stand-in executables for tests that run them straight after writing them.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Writes `body` to `path` as an executable that can be run at once; see [`wait_runnable`].
pub(crate) fn write_executable(path: &Path, body: impl AsRef<[u8]>) {
    std::fs::write(path, body).unwrap();
    crate::llama_release::mark_executable(path).unwrap();
    wait_runnable(path);
}

/// Returns once `path` can be exec'd. A fork on another test thread holds a copy of a write
/// descriptor that was open at the time until its own exec, and until then an exec of that file
/// fails with ETXTBSY ("text file busy"); a descriptor closed before the fork is never copied, so
/// once one exec works, they all do. The probe is killed as soon as it starts.
pub(crate) fn wait_runnable(path: &Path) {
    for _ in 0..1000 {
        match Command::new(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
            Err(e) if e.raw_os_error() == Some(libc::ETXTBSY) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("{} will not run: {e}", path.display()),
        }
    }
    panic!("{} stayed busy", path.display());
}
