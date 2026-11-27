//! The `llmman serve` behind the icon: one already answering, or our child.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const IDLE: &str = "llmman";
const STARTING: &str = "llmman (starting...)";

#[cfg(windows)]
const LLMMAN: &str = "llmman.exe";
#[cfg(not(windows))]
const LLMMAN: &str = "llmman";

/// llmman's default bind, which the child is pinned to (`LLMMAN_HOST`).
/// Debug builds take `LLMMAN_DESKTOP_TEST_HOST`, to run beside a daemon.
fn host() -> String {
    #[cfg(debug_assertions)]
    if let Ok(host) = std::env::var("LLMMAN_DESKTOP_TEST_HOST") {
        return host;
    }
    "127.0.0.1:17434".into()
}

/// A per-user file in the temp dir: `/tmp` is shared on Linux.
pub fn temp_file(suffix: &str) -> PathBuf {
    let user: String = ["USER", "USERNAME"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .unwrap_or_default()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    std::env::temp_dir().join(format!("llmman-desktop-{user}{suffix}"))
}

pub struct Daemon {
    child: Mutex<Option<Child>>,
    /// Set under the `child` lock, so no start spawns after Quit.
    stopped: AtomicBool,
    /// A start is running; a click meanwhile joins it.
    busy: AtomicBool,
    /// A click is waiting for the daemon to answer.
    open: AtomicBool,
    /// Gets the tooltip to show.
    report: Box<dyn Fn(&'static str) + Send + Sync>,
}

impl Daemon {
    pub fn new(report: impl Fn(&'static str) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            child: Mutex::default(),
            stopped: AtomicBool::default(),
            busy: AtomicBool::default(),
            open: AtomicBool::default(),
            report: Box::new(report),
        })
    }

    /// Makes sure a daemon answers, on its own thread, and opens the web UI
    /// once it does if `open`. A failed start is logged; the next click retries.
    pub fn start(self: &Arc<Self>, open: bool) {
        if open {
            self.open.store(true, Ordering::SeqCst);
        }
        if self.busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = self.clone();
        std::thread::spawn(move || {
            let host = host();
            let up = this.ensure(&host);
            (this.report)(IDLE);
            this.busy.store(false, Ordering::SeqCst);
            // After `busy` clears, and only on success: a click that raced a
            // failed start keeps its flag for the retry it spawned.
            if up
                && this.open.swap(false, Ordering::SeqCst)
                && let Err(e) = open::that_detached(format!("http://{host}/"))
            {
                eprintln!("llmman-desktop: could not open the browser: {e}");
            }
        });
    }

    /// Blocks until a daemon answers (`true`) or our child exits or is
    /// stopped (`false`). No timeout: a first run fetches llama.cpp before
    /// `serve` binds, which takes minutes.
    fn ensure(&self, host: &str) -> bool {
        if alive(host) {
            return true;
        }
        (self.report)(STARTING);
        {
            let mut child = self.child.lock().unwrap();
            if self.stopped.load(Ordering::SeqCst) {
                return false;
            }
            // A child that runs but does not answer is still starting.
            if !child
                .as_mut()
                .is_some_and(|c| matches!(c.try_wait(), Ok(None)))
            {
                *child = None;
                match spawn(host) {
                    Ok(c) => *child = Some(c),
                    Err(e) => {
                        eprintln!("llmman-desktop: {e}");
                        return false;
                    }
                }
            }
        }
        loop {
            std::thread::sleep(Duration::from_millis(250));
            if alive(host) {
                return true;
            }
            let mut child = self.child.lock().unwrap();
            match child.as_mut().map(Child::try_wait) {
                Some(Ok(None)) => {}
                _ => {
                    *child = None;
                    return false;
                }
            }
        }
    }

    /// Stops our child, if any. SIGTERM lets `serve` unload its engines;
    /// Windows has no such signal, so the whole tree goes.
    pub fn stop(&self) {
        let mut guard = self.child.lock().unwrap();
        self.stopped.store(true, Ordering::SeqCst);
        let Some(mut child) = guard.take() else {
            return;
        };
        drop(guard);
        #[cfg(unix)]
        // SAFETY: kill(2) has no memory-safety preconditions; the pid is
        // our unreaped child, so it cannot have been reused.
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        #[cfg(windows)]
        {
            let mut cmd = Command::new("taskkill");
            no_window(&mut cmd);
            let _ = cmd
                .args(["/PID", &child.id().to_string(), "/T", "/F"])
                .status();
        }
        for _ in 0..50 {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        // Unix: leave it draining; SIGKILL would skip its engine cleanup.
        #[cfg(windows)]
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Whether llmman answers `GET /api/version`: the router is up, not just
/// the socket.
fn alive(host: &str) -> bool {
    let Ok(addr) = host.parse::<SocketAddr>() else {
        return false;
    };
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_secs(1)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let req = format!("GET /api/version HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    let mut response = Vec::new();
    s.write_all(req.as_bytes()).is_ok() && {
        // A timeout keeps what was read.
        let _ = (&s).take(8192).read_to_end(&mut response);
        answered(&String::from_utf8_lossy(&response))
    }
}

/// llmman's 200 body has `exe` and `pid` (Ollama's has neither); its API-key
/// 401 is told by `WWW-Authenticate: Bearer realm="llmman"`.
fn answered(response: &str) -> bool {
    let (head, body) = response.split_once("\r\n\r\n").unwrap_or((response, ""));
    let head = head.to_ascii_lowercase();
    if !head.starts_with("http/1.") {
        return false;
    }
    match head.split(' ').nth(1) {
        Some("200") => body.contains("\"exe\":") && body.contains("\"pid\":"),
        Some("401") => head.contains("www-authenticate: bearer realm=\"llmman\""),
        _ => false,
    }
}

/// The `llmman` to run: next to this executable, else next to where its
/// symlink points (a Homebrew link into its keg), else on `PATH`.
fn llmman() -> PathBuf {
    let exe = std::env::current_exe().ok();
    let resolved = exe.as_deref().and_then(|e| e.canonicalize().ok());
    let dirs: Vec<&Path> = [exe.as_deref(), resolved.as_deref()]
        .into_iter()
        .flatten()
        .filter_map(Path::parent)
        .collect();
    find_llmman(&dirs)
}

fn find_llmman(dirs: &[&Path]) -> PathBuf {
    dirs.iter()
        .map(|dir| dir.join(LLMMAN))
        .find(|p| p.is_file())
        .unwrap_or_else(|| LLMMAN.into())
}

/// `llmman serve` with its output in the log, where a failure to start
/// goes too: there are no dialogs.
fn spawn(host: &str) -> std::io::Result<Child> {
    let log = private_file(&temp_file("-serve.log"), true)?;
    let exe = llmman();
    let mut cmd = Command::new(&exe);
    cmd.arg("serve")
        .env("LLMMAN_HOST", host)
        .env_remove("LLMMAN_TLS_CERT")
        .env_remove("LLMMAN_TLS_KEY")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log.try_clone()?);
    #[cfg(windows)]
    no_window(&mut cmd);
    cmd.spawn().inspect_err(|e| {
        let _ = writeln!(&log, "could not run {}: {e}", exe.display());
    })
}

/// Opens `path` owner-only (the log can hold credentials), without
/// following a symlink on Unix, truncating it if `truncate`.
pub fn private_file(path: &Path, truncate: bool) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = opts.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    if truncate {
        file.set_len(0)?;
    }
    Ok(file)
}

/// llmman is a console program; each spawn would open a console.
#[cfg(windows)]
fn no_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

/// A scratch path unique to this process, for tests.
#[cfg(test)]
pub fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("llmman-desktop-{name}-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Shutdown, TcpListener};

    const LLMMAN_BODY: &str = r#"{"version":"0.1.5","exe":"/usr/bin/llmman","pid":42}"#;
    const OLLAMA_BODY: &str = r#"{"version":"0.5.1"}"#;
    const KEYED: &str = "401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"llmman\"";

    /// A server that answers one request with `status` (and headers) and
    /// `body`, and its address.
    fn serve_once(status: &'static str, body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            // Read the whole request: closing with unread data sends an RST
            // on macOS and Windows, which discards the response.
            let mut req = Vec::new();
            let mut buf = [0u8; 256];
            while !req.ends_with(b"\r\n\r\n") {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => req.extend_from_slice(&buf[..n]),
                }
            }
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = s.shutdown(Shutdown::Write);
        });
        host
    }

    #[test]
    fn only_llmman_answers_count_as_up() {
        let ok = |body| format!("HTTP/1.1 200 OK\r\ncontent-length: 1\r\n\r\n{body}");
        assert!(answered(&ok(LLMMAN_BODY)));
        assert!(!answered(&ok(OLLAMA_BODY)));
        assert!(!answered(&ok("")));
        let keyed =
            "HTTP/1.1 401 Unauthorized\r\nwww-authenticate: Bearer realm=\"llmman\"\r\n\r\n{}";
        assert!(answered(keyed));
        assert!(!answered("HTTP/1.1 401 Unauthorized\r\n\r\n{}"));
        assert!(!answered("HTTP/1.1 404 Not Found\r\n\r\n"));
        assert!(!answered("SSH-2.0-OpenSSH_9.6\r\n"));
    }

    #[test]
    fn alive_asks_the_router() {
        assert!(alive(&serve_once("200 OK", LLMMAN_BODY)));
        assert!(alive(&serve_once(KEYED, "{}")));
        assert!(!alive(&serve_once("200 OK", OLLAMA_BODY)));
        assert!(!alive(&serve_once("401 Unauthorized", "{}")));
        assert!(!alive(&serve_once("404 Not Found", "")));
    }

    #[test]
    fn nothing_listening_is_not_alive() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = listener.local_addr().unwrap().to_string();
        drop(listener);
        assert!(!alive(&host));
        assert!(!alive("not an address"));
    }

    #[test]
    fn llmman_is_found_in_the_first_dir_that_has_it() {
        let root = scratch("find");
        let (empty, keg) = (root.join("empty"), root.join("keg"));
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(&keg).unwrap();
        std::fs::write(keg.join(LLMMAN), b"").unwrap();

        assert_eq!(find_llmman(&[&empty, &keg]), keg.join(LLMMAN));
        assert_eq!(find_llmman(&[&empty]), PathBuf::from(LLMMAN));
        assert_eq!(find_llmman(&[]), PathBuf::from(LLMMAN));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn temp_file_stays_in_the_temp_dir() {
        let path = temp_file(".lock");
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("llmman-desktop-") && name.ends_with(".lock"));
    }

    #[test]
    fn an_answering_daemon_is_reused() {
        // `report` runs before `spawn`, so a start fails here, not in a real llmman.
        let daemon = Daemon::new(|_| panic!("started a daemon that was already up"));
        assert!(daemon.ensure(&serve_once("200 OK", LLMMAN_BODY)));
        assert!(daemon.child.lock().unwrap().is_none());
    }

    #[test]
    fn stop_cancels_a_later_start() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = listener.local_addr().unwrap().to_string();
        drop(listener);
        let daemon = Daemon::new(|_| {});
        daemon.stop();
        assert!(!daemon.ensure(&host));
        assert!(daemon.child.lock().unwrap().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn private_file_is_owner_only_and_refuses_symlinks() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("priv");
        std::fs::create_dir_all(&dir).unwrap();
        let (file, link) = (dir.join("f"), dir.join("l"));
        std::fs::write(&file, b"x").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        private_file(&file, true).unwrap();
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 0);
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(private_file(&link, true).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
