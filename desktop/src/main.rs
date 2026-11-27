//! llmman's tray icon: a left click opens the web UI, Quit is the only menu
//! item, and `llmman serve` is kept running behind it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod daemon;
mod icon;
#[cfg(target_os = "linux")]
mod sni;
#[cfg(any(target_os = "macos", target_os = "windows"))]
mod tray;

use std::fs::{File, TryLockError};
use std::path::Path;
use std::sync::OnceLock;

fn main() {
    if std::env::args()
        .nth(1)
        .is_some_and(|a| a == "--version" || a == "-V")
    {
        println!("{}", version());
        return;
    }
    if already_running(&daemon::temp_file(".lock")) {
        return;
    }
    #[cfg(target_os = "linux")]
    sni::run();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    tray::run();
}

fn version() -> String {
    format!("llmman-desktop {}", env!("CARGO_PKG_VERSION"))
}

/// Locks `path` for the life of the process, or finds it locked. If it
/// cannot be opened this instance runs: a second icon beats none.
fn already_running(path: &Path) -> bool {
    static LOCK: OnceLock<File> = OnceLock::new();
    let Ok(file) = daemon::private_file(path, false) else {
        return false;
    };
    match file.try_lock() {
        Err(TryLockError::WouldBlock) => true,
        _ => {
            let _ = LOCK.set(file);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line() {
        assert_eq!(
            version(),
            format!("llmman-desktop {}", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn a_second_instance_finds_the_lock_taken() {
        let path = daemon::scratch("lock");
        assert!(!already_running(&path));
        assert!(already_running(&path));
        std::fs::remove_file(&path).unwrap();
    }
}
