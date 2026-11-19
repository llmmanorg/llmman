//! The `hdrm` optimizer's proxy, which lives as long as one launch. It stays
//! in llmman's process group, so Ctrl-C reaches it with the integration.

use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Context;

use super::find_on_path;

/// The proxy's executable.
const PROXY: &str = "headroom";

/// The proxy loads models before it accepts connections.
const STARTUP_BUDGET: Duration = Duration::from_secs(120);

static RUNNING: Mutex<Option<Child>> = Mutex::new(None);

pub(super) fn ensure_installed() -> anyhow::Result<PathBuf> {
    find_on_path(PROXY).ok_or_else(|| {
        anyhow::anyhow!(
            "--optimizer hdrm needs hdrm, which is not installed.\n\
             Install it, or launch without --optimizer."
        )
    })
}

/// The proxy on loopback `port`, forwarding to `upstream`.
fn command(bin: &Path, port: u16, upstream: &str) -> Command {
    let mut cmd = Command::new(bin);
    cmd.args(["proxy", "--port", &port.to_string(), "--workers", "1"])
        .args(["--anthropic-api-url", upstream])
        .args(["--openai-api-url", upstream])
        .env("ANTHROPIC_TARGET_API_URL", upstream)
        .env("OPENAI_TARGET_API_URL", upstream)
        .env("PYTHONIOENCODING", "utf-8");
    cmd
}

/// Starts the proxy and returns its URL.
pub(super) fn start(upstream: &str) -> anyhow::Result<String> {
    let bin = ensure_installed()?;
    let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .and_then(|listener| listener.local_addr())
        .context("find a free port for hdrm")?
        .port();

    let log_path = crate::default_store()
        .ok()
        .and_then(|store| store.parent().map(|dir| dir.join("hdrm.log")));
    let log = log_path.as_deref().and_then(|path| {
        std::fs::create_dir_all(path.parent()?).ok()?;
        std::fs::File::create(path).ok()
    });
    let (out, err) = match log.and_then(|f| f.try_clone().ok().map(|g| (f, g))) {
        Some((out, err)) => (Stdio::from(out), Stdio::from(err)),
        None => (Stdio::null(), Stdio::null()),
    };
    let mut child = command(&bin, port, upstream)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .with_context(|| format!("run {}", bin.display()))?;

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let began = Instant::now();
    let log_note = log_path
        .map(|p| format!("\nLog: {}", p.display()))
        .unwrap_or_default();
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("hdrm exited at startup ({status}).{log_note}");
        }
        if TcpStream::connect_timeout(&addr, Duration::from_secs(1)).is_ok() {
            break;
        }
        if began.elapsed() >= STARTUP_BUDGET {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "hdrm did not accept connections within {}s.{log_note}",
                STARTUP_BUDGET.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    *RUNNING.lock().unwrap() = Some(child);
    Ok(format!("http://{addr}"))
}

/// Stops the proxy, if one is running.
pub(super) fn stop() {
    if let Some(mut child) = RUNNING.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_proxy_listens_on_its_port_and_forwards_to_the_upstream() {
        let cmd = command(Path::new("/bin/proxy"), 4242, "http://127.0.0.1:17434");
        assert_eq!(cmd.get_program(), "/bin/proxy");
        let args: Vec<_> = cmd.get_args().map(|a| a.to_str().unwrap()).collect();
        assert_eq!(
            args,
            [
                "proxy",
                "--port",
                "4242",
                "--workers",
                "1",
                "--anthropic-api-url",
                "http://127.0.0.1:17434",
                "--openai-api-url",
                "http://127.0.0.1:17434",
            ]
        );
        let env = |key: &str| {
            cmd.get_envs()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v)
                .and_then(|v| v.to_str())
                .map(str::to_string)
        };
        for key in ["ANTHROPIC_TARGET_API_URL", "OPENAI_TARGET_API_URL"] {
            assert_eq!(env(key).as_deref(), Some("http://127.0.0.1:17434"), "{key}");
        }
    }

    #[test]
    fn stopping_with_nothing_running_is_fine() {
        stop();
    }
}
