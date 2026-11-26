//! Shared client-side helpers for talking to a local `llmman serve`
//! instance over its Ollama-protocol HTTP API — `http://127.0.0.1:17434`
//! by default, overridable via `LLMMAN_HOST`.
//!
//! Used by any CLI subcommand that acts as a client of that API rather than
//! calling the FFI/model-management logic directly — `pull`, `push`,
//! `run`, `ps`, `stop`, `launch`, and `providers` — so bare model-name
//! resolution (see `shortnames::resolve_ollama_api`), the local model
//! store, any already-loaded models, and the provider catalog (see
//! [`providers`]) are always the daemon's, never duplicated
//! per-invocation.

use std::io::{BufRead, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::Deserialize;

/// Default bind host/port when `LLMMAN_HOST` is unset/blank.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 17434;

/// `(scheme, host, port)` parsed from `LLMMAN_HOST`, cached for the
/// process's life.
static PARSED_HOST: OnceLock<(String, String, u16)> = OnceLock::new();

fn parsed_host() -> &'static (String, String, u16) {
    PARSED_HOST.get_or_init(|| parse_host(std::env::var("LLMMAN_HOST").ok().as_deref()))
}

/// Parses `[scheme://]host[:port][/path]` — a bare host, `host:port`, or
/// nothing at all also work. `path` is discarded (unused). An explicit
/// `http://`/`https://` scheme shifts the default port to 80/443 when no
/// port is given; anything else keeps `DEFAULT_PORT`. A missing/
/// unparseable host or port falls back to its own default, independently.
fn parse_host(value: Option<&str>) -> (String, String, u16) {
    let raw = value.unwrap_or("");
    let trimmed = raw.trim().trim_matches(|c| c == '"' || c == '\'');

    let mut default_port = DEFAULT_PORT;
    let (scheme, hostport) = match trimmed.split_once("://") {
        Some((scheme, rest)) => {
            let scheme = scheme.to_ascii_lowercase();
            match scheme.as_str() {
                "http" => default_port = 80,
                "https" => default_port = 443,
                _ => {}
            }
            (scheme, rest)
        }
        None => ("http".to_string(), trimmed),
    };

    // Drop a trailing path, if any (e.g. "host:port/some/path") — never used.
    let hostport = hostport.split_once('/').map_or(hostport, |(h, _)| h);

    let (host, port) = split_host_port(hostport);
    let port = port
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(default_port);
    let host = host.unwrap_or_else(|| DEFAULT_HOST.to_string());
    (scheme, host, port)
}

/// Minimal `net.SplitHostPort` equivalent for what `parse_host` can pass
/// in: empty, a bare host, `host:port`, or a bracketed IPv6 literal
/// (`[::1]` or `[::1]:port`). A bare multi-colon literal (e.g. `::1`) is
/// kept whole with no port rather than guessed at.
fn split_host_port(hostport: &str) -> (Option<String>, Option<&str>) {
    if hostport.is_empty() {
        return (None, None);
    }
    if let Some(rest) = hostport.strip_prefix('[') {
        return match rest.split_once(']') {
            Some((host, after)) => {
                let port = after.strip_prefix(':').filter(|p| !p.is_empty());
                (Some(host.to_string()), port)
            }
            None => (Some(hostport.to_string()), None), // malformed bracket; keep as one host
        };
    }
    match hostport.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') && !port.is_empty() => {
            (Some(host.to_string()), Some(port))
        }
        _ => (Some(hostport.to_string()), None),
    }
}

/// Renders `host:port`, bracketing `host` if it's an IPv6 literal.
fn format_host_port(host: &str, port: u16) -> String {
    if host.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// The `scheme://host:port` origin every client in this process talks
/// to: `https` only when `LLMMAN_HOST` spells it, and built from
/// `connect_addr` so a wildcard bind still gets a reachable host.
pub fn server() -> String {
    format!("{}://{}", scheme(), connect_addr())
}

fn scheme() -> &'static str {
    if parsed_host().0 == "https" {
        "https"
    } else {
        "http"
    }
}

/// Whether `LLMMAN_HOST` asks for TLS. `cmd::serve` requires this and
/// `LLMMAN_TLS_CERT` to agree.
pub fn tls_scheme() -> bool {
    scheme() == "https"
}

/// Whether this process reaches the daemon without a cleartext network
/// hop — over loopback or TLS — and so may send a credential along.
pub fn connects_securely() -> bool {
    connects_over_loopback() || tls_scheme()
}

/// A blocking client for the daemon: the process's key as a default
/// header ([`crate::auth::client_headers`]) plus `LLMMAN_TLS_CA`'s roots.
/// Every client of `server()` is built here or in
/// [`async_client_builder`], so none can forget either.
pub fn client_builder() -> anyhow::Result<reqwest::blocking::ClientBuilder> {
    let mut builder =
        reqwest::blocking::Client::builder().default_headers(crate::auth::client_headers()?);
    for cert in crate::auth::tls_ca()? {
        builder = builder.add_root_certificate(cert);
    }
    Ok(builder)
}

pub fn client() -> anyhow::Result<reqwest::blocking::Client> {
    client_builder()?.build().context("build http client")
}

/// The async twin of [`client_builder`].
pub fn async_client_builder() -> anyhow::Result<reqwest::ClientBuilder> {
    Ok(crate::auth::trusted_client()?.default_headers(crate::auth::client_headers()?))
}

/// A peer's `scheme://host:port` origin from a `LLMMAN_HOST`-style spec
/// (`asahi`, `spark:17434`, `http://10.0.0.5`), port defaulting to ours.
pub fn peer_url(spec: &str) -> String {
    let (scheme, host, port) = parse_host(Some(spec));
    format!("{scheme}://{}", format_host_port(&host, port))
}

/// The bare `host:port` `llmman serve`'s own listener binds — the raw
/// configured host, since a wildcard bind (`0.0.0.0`/`::`) is meaningful
/// here, unlike for `connect_addr`.
pub fn bind_addr() -> String {
    let (_, host, port) = parsed_host();
    format_host_port(host, *port)
}

/// [`bind_addr`] with the port the listener got, which differs from the
/// configured one only when `LLMMAN_HOST` asks for port 0.
pub fn bound_addr(port: u16) -> String {
    format_host_port(&parsed_host().1, port)
}

/// The bare `host:port` a client should *connect* to — like `bind_addr`,
/// but a wildcard host is rewritten to loopback first, since a client
/// can't connect to "every interface", and port 0 by the port the running
/// daemon recorded ([`recorded_port`]).
fn connect_addr() -> String {
    let (_, host, port) = parsed_host();
    let port = match port {
        0 => recorded_port().unwrap_or(0),
        port => *port,
    };
    format_host_port(connectable_host(host), port)
}

/// Whether `LLMMAN_HOST` leaves the port to the OS (`127.0.0.1:0`).
pub fn os_picks_port() -> bool {
    parsed_host().2 == 0
}

/// `addr` (`host:port`) as a file name part: `host-port`, an IPv6 host
/// without its brackets and with its colons as `_`, since Windows forbids
/// `:` in file names.
pub fn file_safe_addr(addr: &str) -> String {
    let (host, port) = addr.rsplit_once(':').unwrap_or((addr, ""));
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .replace(':', "_");
    format!("{host}-{port}")
}

/// The directory and name prefix (`serve-<host>-0`) of the daemon's lock
/// and port records, from the configured address every client with the
/// same `LLMMAN_HOST` knows.
fn port_files() -> anyhow::Result<(PathBuf, String)> {
    Ok((
        crate::data_root()?,
        format!("serve-{}", file_safe_addr(&bind_addr())),
    ))
}

/// The port in a record's name, `<name>-<port>.port`.
fn record_port(file_name: &str, name: &str) -> Option<u16> {
    let port: u16 = file_name
        .strip_prefix(name)?
        .strip_prefix('-')?
        .strip_suffix(".port")?
        .parse()
        .ok()?;
    (port != 0).then_some(port)
}

/// The port in the record its daemon holds a lock on. The OS releases a
/// lock when its holder exits, so a record left by a daemon that crashed
/// is never used, even while another daemon is starting. Nor is a record
/// others can open, since they could hold its lock.
fn held_port(dir: &Path, name: &str) -> Option<u16> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| {
            let port = record_port(entry.file_name().to_str()?, name)?;
            let record = std::fs::File::open(entry.path()).ok()?;
            (owner_only(&record)
                && matches!(
                    record.try_lock_shared(),
                    Err(std::fs::TryLockError::WouldBlock)
                ))
            .then_some(port)
        })
}

/// Read on every call, since [`ensure_server`] can start the daemon
/// between calls.
fn recorded_port() -> Option<u16> {
    let (dir, name) = port_files().ok()?;
    held_port(&dir, &name)
}

/// Held by a daemon whose port the OS picked, from before its bind until it
/// exits, so a second daemon for the same `LLMMAN_HOST` cannot start.
pub struct PortLock {
    dir: PathBuf,
    name: String,
    _lock: std::fs::File,
}

impl PortLock {
    /// `None` for a configured port, which clients already know.
    pub fn acquire() -> anyhow::Result<Option<PortLock>> {
        if !os_picks_port() {
            return Ok(None);
        }
        let (dir, name) = port_files()?;
        Self::acquire_in(dir, name, HOLDER_WAIT).map(Some)
    }

    fn acquire_in(dir: PathBuf, name: String, holder_wait: Duration) -> anyhow::Result<PortLock> {
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let path = dir.join(format!("{name}.lock"));
        // Never removed: a daemon that deleted and recreated it would lock a
        // new file while a running daemon still holds the old one.
        let file = open_owner_only(&path).with_context(|| format!("open {}", path.display()))?;
        // A holder without a record is still starting or is shutting down.
        // Waiting means this daemon refuses only once clients can find the
        // holder, and starts if the holder exits.
        let deadline = std::time::Instant::now() + holder_wait;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock)
                    if held_port(&dir, &name).is_some()
                        || std::time::Instant::now() >= deadline =>
                {
                    anyhow::bail!(
                        "another llmman serve is running for this LLMMAN_HOST: it holds {}",
                        path.display()
                    )
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(std::fs::TryLockError::Error(e)) => {
                    return Err(e).with_context(|| format!("lock {}", path.display()));
                }
            }
        }
        Ok(PortLock {
            dir,
            name,
            _lock: file,
        })
    }

    /// Records the port the listener got, for clients to use while the
    /// returned record lives, and removes records left by daemons that
    /// crashed.
    pub fn publish(&self, port: u16) -> anyhow::Result<PortRecord> {
        let path = self.dir.join(format!("{}-{port}.port", self.name));
        let file = open_owner_only(&path).with_context(|| format!("create {}", path.display()))?;
        // Blocking: only clients checking the record contend for it, and
        // each holds it for a moment.
        file.lock()
            .with_context(|| format!("lock {}", path.display()))?;
        let entries =
            std::fs::read_dir(&self.dir).with_context(|| format!("read {}", self.dir.display()))?;
        for entry in entries.filter_map(Result::ok) {
            let stale = entry
                .file_name()
                .to_str()
                .and_then(|n| record_port(n, &self.name))
                .is_some_and(|p| p != port);
            if stale {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Ok(PortRecord { path, _lock: file })
    }
}

/// How long a daemon that finds the port lock held waits for its holder to
/// publish a record or exit.
const HOLDER_WAIT: Duration = Duration::from_secs(5);

/// Opens `path`, creating it, for its owner alone on Unix: a read-only
/// descriptor is enough to lock a file, so anyone who could read it could
/// hold its lock.
fn open_owner_only(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(false);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let file = options.open(path)?;
    // The mode above reaches only a file this call creates.
    #[cfg(unix)]
    file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    Ok(file)
}

/// Whether only its owner can open `file`: earlier versions left records
/// readable by everyone.
#[cfg(unix)]
fn owner_only(file: &std::fs::File) -> bool {
    use std::os::unix::fs::PermissionsExt;
    file.metadata()
        .is_ok_and(|m| m.permissions().mode() & 0o077 == 0)
}

/// Windows has no mode bits: the profile's ACL keeps other users out.
#[cfg(not(unix))]
fn owner_only(_: &std::fs::File) -> bool {
    true
}

/// A daemon's published port. Dropping it removes the record.
pub struct PortRecord {
    path: PathBuf,
    _lock: std::fs::File,
}

impl Drop for PortRecord {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Rewrites a wildcard host (any `is_unspecified` IP, e.g. `0.0.0.0`/`::`)
/// to its loopback equivalent, by value rather than by exact spelling —
/// so an expanded IPv6 form (`0:0:0:0:0:0:0:0`) is caught too.
fn connectable_host(host: &str) -> &str {
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => {
            if ip.is_ipv4() {
                "127.0.0.1"
            } else {
                "::1"
            }
        }
        _ => host,
    }
}

/// Quick reachability check with a short timeout — a plain TCP connect
/// attempt is enough (no need to round-trip an HTTP request), but with
/// `LLMMAN_HOST` possibly pointing at a remote, unreachable address, an
/// unbounded `TcpStream::connect` could hang far longer than any caller
/// (`ensure_server`, `ps`) wants to wait.
pub fn server_alive() -> bool {
    use std::net::ToSocketAddrs;
    let Ok(mut addrs) = connect_addr().to_socket_addrs() else {
        return false;
    };
    addrs.any(|addr| std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok())
}

/// The daemon's self-identity from GET /api/version, which tells a llmman
/// daemon from another server such as Ollama.
#[derive(Deserialize)]
struct DaemonIdentity {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    exe: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
}

impl DaemonIdentity {
    /// Any nonzero pid counts: a containerised daemon reports pid 1.
    fn is_llmman(&self) -> bool {
        self.version.as_deref().is_some_and(|v| !v.is_empty())
            && self.exe.as_deref().is_some_and(|exe| !exe.is_empty())
            && self.pid.is_some_and(|pid| pid > 0)
    }
}

fn daemon_identity() -> anyhow::Result<DaemonIdentity> {
    client_builder()?
        .timeout(Some(Duration::from_secs(2)))
        .build()?
        .get(format!("{}/api/version", server()))
        .send()
        .context("request /api/version")?
        .error_for_status()?
        .json()
        .context("parse /api/version")
}

/// A reachable port is not necessarily a llmman daemon.
pub fn check_server_identity() -> anyhow::Result<()> {
    let identity =
        daemon_identity().with_context(|| format!("check llmman daemon at {}", server()))?;
    anyhow::ensure!(
        identity.is_llmman(),
        "{} is reachable but did not return a llmman daemon identity",
        server()
    );
    Ok(())
}

/// Whether the configured host is this machine — a remote one's PID
/// belongs to another machine, so spawning/killing based on it must be
/// skipped.
fn host_is_local() -> bool {
    let (_, host, _) = parsed_host();
    is_local_host(host)
}

/// Whether the daemon is reachable *only* from this machine — i.e.
/// whether a request can have come from anyone else.
///
/// Deliberately not [`connects_over_loopback`]: a wildcard bind is
/// reached over loopback from here, but it is also reached by anyone who
/// can route to this host, and `0.0.0.0` is exactly that. `cmd::serve`
/// uses this to decide whether to spend the daemon's own credentials for
/// a caller it cannot authenticate.
pub fn reachable_only_locally() -> bool {
    host_reachable_only_locally(&parsed_host().1)
}

/// [`reachable_only_locally`] for an explicit host, so it is testable
/// without the process-wide `PARSED_HOST` already being fixed.
fn host_reachable_only_locally(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// Whether this process reaches the daemon without leaving the machine.
///
/// The other side of [`reachable_only_locally`]: a wildcard bind counts,
/// since `connect_addr` sends us to loopback for it. `cmd::launch` uses
/// this to decide whether handing an integration a real API key would put
/// it on the network in cleartext.
pub fn connects_over_loopback() -> bool {
    host_connects_over_loopback(&parsed_host().1)
}

/// [`connects_over_loopback`] for an explicit host. See
/// [`host_reachable_only_locally`].
fn host_connects_over_loopback(host: &str) -> bool {
    is_local_host(connectable_host(host))
}

/// By value rather than by exact spelling, same as `connectable_host` —
/// a loopback or unspecified IP (in any form) is local; a bare
/// "localhost" is too, since it always resolves to one.
fn is_local_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
}

/// Whether the server that answered /api/version is a llmman daemon: it sent
/// an exe, or (a daemon from before identity existed) llmman-only
/// `/llmman/providers` answers to `probe`. Ollama sends at most a `version`.
fn is_llmman_daemon(identity: &DaemonIdentity, probe: impl FnOnce() -> bool) -> bool {
    let has_exe = identity.exe.as_deref().is_some_and(|exe| !exe.is_empty());
    identity.is_llmman() || has_exe || probe()
}

/// Whether the server at `LLMMAN_HOST` positively answers as something
/// other than a llmman daemon, such as Ollama. One that cannot be asked
/// (down, or behind API keys) is not known to be foreign. `llmman run` asks
/// to send a bare model name as typed.
pub fn server_is_foreign() -> bool {
    daemon_identity().is_ok_and(|identity| !is_llmman_daemon(&identity, llmman_api_answers))
}

/// Whether `body`, from `GET /llmman/providers`, is a llmman daemon's:
/// an object with a `providers` list.
fn is_providers_list(body: &serde_json::Value) -> bool {
    body.get("providers").is_some_and(|p| p.is_array())
}

/// Whether `GET /llmman/providers` answers the way a llmman daemon does.
/// Ollama answers 404.
fn llmman_api_answers() -> bool {
    let answer = || -> anyhow::Result<serde_json::Value> {
        Ok(client_builder()?
            .timeout(Some(Duration::from_secs(2)))
            .build()?
            .get(format!("{}/llmman/providers", server()))
            .send()?
            .error_for_status()?
            .json()?)
    };
    answer().is_ok_and(|body| is_providers_list(&body))
}

/// `pid` as a `pid_t` that names one daemon, or `None` for one that cannot.
/// kill(0) hits the caller's own process group and kill(-1) every process
/// the user owns, so 0 and 1 are refused, and a pid past `pid_t` names
/// nothing.
#[cfg(unix)]
fn signalable_pid(pid: u32) -> Option<i32> {
    i32::try_from(pid).ok().filter(|pid| *pid > 1)
}

/// `taskkill /PID <pid> /T`, plus `/F` when `force`. Stdio is nulled because
/// it prints "ERROR: The process "1544" not found." for a pid that has
/// already gone, and the exit code is ignored.
#[cfg(windows)]
fn taskkill(pid: u32, force: bool) {
    let pid = pid.to_string();
    let mut args = vec!["/PID", pid.as_str(), "/T"];
    if force {
        args.push("/F");
    }
    let _ = Command::new("taskkill")
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Starts `llmman serve` as a background process if one isn't already
/// running, and waits (up to 60s) for it to start accepting connections.
/// The process is intentionally never stopped by this command — once
/// started it keeps running indefinitely, independent of this invocation,
/// so later commands (from this CLI or a concurrent one) reuse it instead
/// of starting a redundant copy.
///
/// stdin/stdout/stderr are all detached from this process (redirected to a
/// log file, or /dev/null if the log file can't be opened): the daemon
/// outlives this command, so anything that inherited its stdout/stderr
/// (a parent shell, a script capturing this command's output, `$(...)`,
/// etc.) would otherwise block forever waiting for those pipes to close,
/// since the daemon never exits to close them itself. The child is also
/// put in its own process group so terminal signals (e.g. Ctrl-C) sent to
/// this command's foreground process group don't reach it either.
///
/// `preload_model`, if non-empty, is passed through as `llmman serve`'s
/// optional positional argument so the daemon starts loading it
/// immediately (see cmd::serve::ServeArgs::model) instead of waiting for
/// the first request that references it. Pass "" when there's nothing to
/// preload — `run`/`pull`/`push` all pass "" so the daemon they spawn
/// stays a plain, model-agnostic `llmman serve` (only `launch` still
/// preloads, since its whole point is warming up one model for the
/// integration it's about to hand off to).
///
/// With `LLMMAN_NO_AUTOSTART` set, a server that does not answer is an error
/// instead of a spawn: whoever manages the daemon (sbx) must never get a
/// second one with this process's environment.
pub fn ensure_server(preload_model: &str) -> anyhow::Result<()> {
    if crate::env_flag_set(NO_AUTOSTART_VAR) {
        return require_server(server_alive(), &server());
    }
    ensure_server_with_args(preload_model, &[])
}

/// `LLMMAN_NO_AUTOSTART`: client commands only talk to a running server.
pub const NO_AUTOSTART_VAR: &str = "LLMMAN_NO_AUTOSTART";

fn require_server(alive: bool, server: &str) -> anyhow::Result<()> {
    if alive {
        return Ok(());
    }
    anyhow::bail!(
        "no llmman server answers at {server} and {NO_AUTOSTART_VAR} is set, so none is \
         started. Start `llmman serve` (or unset {NO_AUTOSTART_VAR})"
    )
}

/// Like [`ensure_server`], forwarding additional `serve` options only when
/// spawning a new daemon. The caller must not pass `--detach` back to the child.
///
/// Whatever answers at `LLMMAN_HOST` is reused as is, a llmman daemon or an
/// Ollama alike: this never judges, stops or replaces a running server.
/// Nothing answering on a remote host is an error, since a daemon cannot be
/// started there.
pub fn ensure_server_with_args(preload_model: &str, serve_args: &[&str]) -> anyhow::Result<()> {
    if server_alive() {
        return Ok(());
    }
    if !host_is_local() {
        anyhow::bail!(
            "LLMMAN_HOST={} is unreachable, and llmman can't start a daemon on a remote host",
            server()
        );
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;

    let log_path = crate::default_store()
        .ok()
        .and_then(|store| store.parent().map(|p| p.join("serve.log")));

    // create_dir_all the log's parent (e.g. ~/.local/share/llmman) before
    // ever trying to open it below: `OpenOptions::create(true)` only
    // creates the *file*, never missing intermediate directories, so on a
    // genuinely fresh machine (nothing has ever pulled/served a model
    // yet — confirmed missing on a real macOS CI runner) the `.open()`
    // below would otherwise fail every single time, permanently losing
    // the daemon's entire stdout/stderr to the `None` branch's
    // `Stdio::null()` fallback instead of just this one first call.
    if let Some(p) = log_path.as_ref().and_then(|p| p.parent()) {
        let _ = std::fs::create_dir_all(p);
    }

    // Rotate the previous daemon's log out of the way (serve.log ->
    // serve-1.log -> ... -> serve-5.log) instead of appending forever,
    // the same scheme Ollama's app uses for server.log (app/logrotate):
    // each spawned daemon gets a fresh log, the last MAX_LOG_FILES
    // daemons' logs stay around for diagnostics, and nothing grows
    // without bound. Only rotated here, right before an actual spawn — a
    // call that reuses an already-running daemon never touches its log.
    if let Some(p) = &log_path {
        rotate_log(p);
    }

    let mut cmd = Command::new(&exe);
    cmd.arg("serve");
    if !preload_model.is_empty() {
        cmd.arg(preload_model);
    }
    cmd.args(serve_args);
    cmd.stdin(Stdio::null());
    let pull_relay = log_path
        .as_deref()
        .and_then(|p| p.parent())
        .and_then(PullRelay::new);
    // Internal: never inherited from our own environment.
    cmd.env_remove(crate::container::PULL_PROGRESS_FILE_ENV)
        .env_remove(crate::container::PULL_PROGRESS_COLS_ENV);
    if let Some(relay) = &pull_relay {
        cmd.env(crate::container::PULL_PROGRESS_FILE_ENV, &relay.path)
            .env(
                crate::container::PULL_PROGRESS_COLS_ENV,
                relay.cols.to_string(),
            );
    }
    // Silently redirect the daemon's stdio to its log file (or /dev/null if
    // that file can't be opened) — no "starting serve" status line here:
    // every caller of ensure_server (run/pull/push/launch) wants to look
    // like a plain client of an already-running server, not announce that
    // it happened to be the one that started it this time. Anyone who
    // needs to know still can — see log_path above / `llmman serve.log`.
    match log_path
        .as_ref()
        .and_then(|p| std::fs::File::create(p).ok())
        .and_then(|f| f.try_clone().ok().map(|f2| (f, f2)))
    {
        Some((out, err)) => {
            cmd.stdout(out);
            cmd.stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    detach(&mut cmd);
    let mut child = cmd.spawn().context("spawn llmman serve")?;

    // On its first start the daemon fetches its backends (llama.cpp; uv
    // and mlx-lm on macOS) before it binds, which can run well
    // past STARTUP_BUDGET. While a fetch is visibly live, keep waiting
    // and show what it is doing: stopping the daemon would discard the
    // partial work. The budget is the time spent neither listening nor
    // fetching, and restarts after each fetch so the daemon gets a full
    // budget to bind afterwards.
    let mut idle_since = std::time::Instant::now();
    let mut fetch_notice = FetchNotice::new(pull_relay);
    loop {
        std::thread::sleep(Duration::from_millis(500));
        if server_alive() {
            // The connect proves the daemon was alive an instant ago, not
            // that it survived: if it died right after accepting, report
            // its real exit status instead of an Ok the caller's next
            // request would contradict. A death after this check is
            // inherently unobservable from here.
            bail_if_exited(&mut child, log_path.as_deref())?;
            return Ok(());
        }
        // The daemon is in its own process group but still our child, so
        // try_wait catches an immediate startup failure (a bad LLMMAN_HOST,
        // say; a failed backend fetch is only a warning there) instead of
        // polling a dead port for 60s.
        bail_if_exited(&mut child, log_path.as_deref())?;
        match crate::llama_release::download_status() {
            Some(status) => {
                fetch_notice.update(&status);
                idle_since = std::time::Instant::now();
            }
            None if idle_since.elapsed() >= STARTUP_BUDGET => break,
            None => {}
        }
    }
    drop(fetch_notice);
    // The last in-loop probe is up to 500ms stale by now; a daemon that
    // bound in that window is a healthy start, not a timeout. Same shape
    // as the in-loop success path: prove the daemon survived the accept
    // before reporting Ok.
    if server_alive() {
        bail_if_exited(&mut child, log_path.as_deref())?;
        return Ok(());
    }
    // The daemon may have exited right after the final poll above: check
    // once more so that narrow window still reports the exit status
    // instead of the generic timeout.
    bail_if_exited(&mut child, log_path.as_deref())?;
    // Timed out with the daemon alive but not listening. If a backend
    // fetch began in the window since the loop's last look, leave the
    // daemon to finish it (see above) and say so.
    if crate::llama_release::download_in_progress() {
        anyhow::bail!(
            "llmman serve did not start within {}s: startup is still fetching an inference \
             backend (llama.cpp, or mlx-lm on macOS). The daemon was left running \
             so it can finish; retry this command once it does{}",
            STARTUP_BUDGET.as_secs(),
            log_tail(log_path.as_deref())
        );
    }
    // Otherwise stop it before reporting failure, rather than leave a
    // half-started daemon running detached after the user was told the
    // start failed, then wait so no zombie outlives this call. The
    // message must say the daemon was stopped, or the failure would read
    // as retryable against a daemon that is no longer there.
    stop_group(child.id());
    let _ = child.wait();
    anyhow::bail!(
        "llmman serve did not start within {}s and was stopped; run 'llmman serve' in the \
         foreground to watch what startup is doing{}",
        STARTUP_BUDGET.as_secs(),
        log_tail(log_path.as_deref())
    )
}

/// How long [`ensure_server`] waits for a freshly spawned daemon that is
/// neither listening nor reporting a live backend fetch.
const STARTUP_BUDGET: Duration = Duration::from_secs(60);

/// How long a pull keeps the spinner before its output takes over: a
/// cached image answers in about a second, and "up to date" after every
/// daemon start would be noise.
const PULL_RELAY_DELAY: Duration = Duration::from_secs(2);

/// What [`ensure_server`] shows while its daemon is still fetching
/// backends: a spinner line with the daemon's latest progress text on a
/// terminal, else one plain notice the first time. A lasting container
/// image pull shows its own output instead ([`PullRelay`]). Cleared on drop.
struct FetchNotice {
    spinner: Option<ProgressBar>,
    announced: bool,
    relay: Option<PullRelay>,
    /// When a pull was first seen, until a different status is.
    pulling_since: Option<std::time::Instant>,
    /// Whether the last [`update`](Self::update) showed the relay.
    relaying: bool,
}

impl FetchNotice {
    fn new(relay: Option<PullRelay>) -> FetchNotice {
        FetchNotice {
            spinner: None,
            announced: false,
            relay,
            pulling_since: None,
            relaying: false,
        }
    }

    /// `status` is the marker text, `""` for a fetch that never set one.
    fn update(&mut self, status: &str) {
        if status == crate::container::PULL_STATUS {
            let pulling_for = self
                .pulling_since
                .get_or_insert_with(std::time::Instant::now)
                .elapsed();
            let relaying = self.relaying;
            if let Some(relay) = self
                .relay
                .as_mut()
                .filter(|_| relaying || pulling_for >= PULL_RELAY_DELAY)
            {
                // The pull's cursor movement would fight the spinner.
                if let Some(pb) = self.spinner.take() {
                    pb.finish_and_clear();
                }
                relay.drain();
                self.relaying = true;
                return;
            }
        } else {
            self.pulling_since = None;
        }
        self.finish_relay();
        let message = fetch_message(status);
        if !std::io::stderr().is_terminal() {
            if !self.announced {
                eprintln!("[llmman] {message}");
                self.announced = true;
            }
            return;
        }
        self.spinner
            .get_or_insert_with(|| crate::fmt::braille_spinner("{spinner} {msg}"))
            .set_message(message);
    }

    /// The marker can vanish before the last poll; show what it missed.
    fn finish_relay(&mut self) {
        if std::mem::take(&mut self.relaying) {
            if let Some(relay) = self.relay.as_mut() {
                relay.drain();
            }
        }
    }
}

impl Drop for FetchNotice {
    fn drop(&mut self) {
        self.finish_relay();
        if let Some(pb) = self.spinner.take() {
            pb.finish_and_clear();
        }
    }
}

/// Replays a spawned daemon's container image pull on this terminal. The
/// daemon runs the pull on a pty and appends what it draws to a file that
/// [`drain`](Self::drain) copies to stderr as it grows (see
/// `container::PULL_PROGRESS_FILE_ENV`). The file exists only while this
/// does, which is how the daemon knows someone is watching.
struct PullRelay {
    path: std::path::PathBuf,
    /// Terminal width for the daemon's pty.
    cols: u16,
    /// How much of the file has been shown.
    offset: u64,
}

/// One file per client process, so concurrent clients stay apart.
const PULL_PROGRESS_PREFIX: &str = "pull-progress-";

/// Older than this, another client's progress file is one a killed
/// client left behind.
const PULL_PROGRESS_STALE: Duration = Duration::from_secs(24 * 60 * 60);

impl PullRelay {
    /// `None` unless stderr is a terminal of known width.
    fn new(dir: &std::path::Path) -> Option<PullRelay> {
        Self::at(dir, crate::fmt::terminal_cols(2)?)
    }

    fn at(dir: &std::path::Path, cols: u16) -> Option<PullRelay> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let stale = entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(PULL_PROGRESS_PREFIX)
                    && entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > PULL_PROGRESS_STALE));
                if stale {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        let path = dir.join(format!("{PULL_PROGRESS_PREFIX}{}", std::process::id()));
        crate::llama_release::create_new_file(&path).ok()?;
        Some(PullRelay {
            path,
            cols,
            offset: 0,
        })
    }

    /// Copies what the file has gained to stderr.
    fn drain(&mut self) {
        self.drain_to(&mut std::io::stderr());
    }

    fn drain_to(&mut self, out: &mut dyn std::io::Write) {
        use std::io::{Read, Seek, SeekFrom};
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return;
        };
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut chunk = Vec::new();
        if file.read_to_end(&mut chunk).is_ok() && !chunk.is_empty() {
            self.offset += chunk.len() as u64;
            let _ = out.write_all(&chunk);
            let _ = out.flush();
        }
    }
}

impl Drop for PullRelay {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The user-facing line for a daemon-side fetch reporting `status`, which
/// already names the backend ("downloading llama-…", "installing mlx-lm").
fn fetch_message(status: &str) -> String {
    const PREFIX: &str = "first run: setting up the llmman daemon's inference backends (one-time)";
    match status.trim() {
        "" => PREFIX.to_string(),
        status => format!("{PREFIX}: {status}"),
    }
}

/// Signals the process group led by `pid`; true if the group existed.
///
/// kill(2), not kill(1): procps-ng's kill(1) reads only the first digit
/// of a negative pid after a `-SIG` flag, so `kill -TERM -12345` issues
/// `kill(-1, SIGTERM)` — every process the user owns. It also exits 0
/// for a group that doesn't exist, defeating stop_group's escalation
/// check.
#[cfg(unix)]
fn kill_group(pid: u32, signal: libc::c_int) -> bool {
    let Some(pid) = signalable_pid(pid) else {
        return false;
    };
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe { libc::kill(-pid, signal) == 0 }
}

/// Best-effort signal to the daemon's whole process group (it is its own
/// group leader, see detach), with no bare-pid fallback: callers here may
/// hold a reaped pid, and signalling that could hit a reused process.
#[cfg(unix)]
fn signal_group(pid: u32, force: bool) -> bool {
    kill_group(pid, if force { libc::SIGKILL } else { libc::SIGTERM })
}

/// Windows equivalent: taskkill's /T already targets the whole process
/// tree, and fails harmlessly on an already-dead pid.
#[cfg(windows)]
fn signal_group(pid: u32, force: bool) -> bool {
    taskkill(pid, force);
    // taskkill's exit codes don't reliably separate "no such pid" from
    // "needs /F" (a windowless process rejects the graceful form), so
    // always report possible survivors and let stop_group escalate.
    true
}

/// Best-effort stop of the daemon's whole process group: TERM first, then
/// escalate to a group KILL after a short grace so a slow-to-exit child
/// is actually stopped too. The escalation is skipped when the TERM found
/// nothing to signal (an already-empty group), which keeps bail_if_exited's
/// fast-fail path free of the grace sleep.
fn stop_group(pid: u32) {
    if signal_group(pid, false) {
        std::thread::sleep(Duration::from_millis(500));
        signal_group(pid, true);
    }
}

/// Bails with the daemon's exit status (and its log tail) if the freshly
/// spawned `llmman serve` child has already exited; returns Ok(()) while
/// it's still running. On the exited branch this also best-effort kills
/// the daemon's whole process group first, so a llama-server child the
/// daemon spawned before dying is not left holding a loaded model.
fn bail_if_exited(
    child: &mut std::process::Child,
    log_path: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let pid = child.id();
    let Some(status) = child.try_wait().context("wait on llmman serve")? else {
        return Ok(());
    };
    // An exited child is not always a failed startup: two concurrent
    // clients can both spawn a daemon, and the loser exits while the winner
    // serves: at the bind with "address in use", or on port 0 at the port
    // lock once the winner has published its port. If something is
    // listening now, the start succeeded, whoever's daemon it is.
    if server_alive() {
        return Ok(());
    }
    // The daemon may have spawned a llama-server before dying, and that
    // child shares the daemon's process group (see detach), so stop the
    // group rather than orphan a loaded model. The leader itself is dead
    // and reaped by now; only the group form of the signal is meaningful
    // (see signal_group on why there is deliberately no single-pid
    // fallback here). stop_group only pays its grace sleep when the TERM
    // actually found a survivor, so the common childless fast-fail stays
    // fast.
    stop_group(pid);
    // Reading the log now needs no explicit sync: a daemon that fails via
    // its error-exit path reports on stderr, which Rust never buffers,
    // and that fd is redirected straight to the log file, so the reason
    // is visible to any reader before the exit is observable. A daemon
    // killed by a signal writes nothing; log_tail's "(see path)" fallback
    // covers that.
    anyhow::bail!(
        "llmman serve exited during startup ({status}){}",
        log_tail(log_path)
    )
}

/// The last few lines of the daemon's log, formatted for appending to an
/// error message; empty if there's no log path, just a pointer to the
/// path if the file is unreadable or has no content.
fn log_tail(log_path: Option<&std::path::Path>) -> String {
    let Some(path) = log_path else {
        return String::new();
    };
    let Ok(contents) = std::fs::read_to_string(path) else {
        return format!(" (see {})", path.display());
    };
    let lines: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return format!(" (see {})", path.display());
    }
    let tail = &lines[lines.len().saturating_sub(5)..];
    format!("; last lines of {}:\n{}", path.display(), tail.join("\n"))
}

/// Puts the about-to-be-spawned child in its own process group (Unix) or
/// process group (Windows), so signals delivered to this process's
/// foreground process group (e.g. Ctrl-C in an interactive shell) don't
/// also terminate the daemon.
#[cfg(unix)]
fn detach(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
fn detach(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

/// How many rotated generations of serve.log to keep — the same count as
/// Ollama's app/logrotate.MaxLogFiles.
const MAX_LOG_FILES: u32 = 5;

/// Rotates `path` (e.g. serve.log) through numbered generations
/// (serve-1.log ... serve-5.log), dropping the oldest — a direct port of
/// Ollama's app/logrotate.Rotate, which its supervisor runs on every
/// server spawn. Purely best-effort: a rotation failure must never stop
/// the daemon from starting, the worst case is just an appended log.
fn rotate_log(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("serve");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("log");
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    for i in (1..=MAX_LOG_FILES).rev() {
        let older = dir.join(format!("{stem}-{i}.{ext}"));
        let newer = if i == 1 {
            path.to_path_buf()
        } else {
            dir.join(format!("{stem}-{}.{ext}", i - 1))
        };
        if newer.exists() {
            if older.exists() {
                let _ = std::fs::remove_file(&older);
            }
            let _ = std::fs::rename(&newer, &older);
        }
    }
}

// ---------------------------------------------------------------------------
// Windows-only: stop this process's own stdio handles from leaking into
// whatever it spawns
// ---------------------------------------------------------------------------

/// Root-caused a real Windows-only E2E hang: `cargo test`'s own harness
/// (tests/launch_e2e.rs's `spawn_with_timeout`) spawns `llmman run` with
/// piped stdout/stderr to capture its output live, and that child's own
/// `try_wait()`-detected exit was always reported correctly and promptly
/// — the hang was in the *next* step, `stdout_thread`/`stderr_thread`'s
/// `.join()`, which blocks on each reader thread's `read()` returning
/// `Ok(0)` (EOF). A pipe only reaches EOF once *every* handle to its
/// write end is closed — not just the one held by the process the reader
/// thinks it's reading from.
///
/// On Windows, any `CreateProcess` call that redirects a child's stdio via
/// a handle (a `File`, or another pipe — exactly what `ensure_server`
/// below does, redirecting the daemon's stdout/stderr to its log file)
/// requires `bInheritHandles = TRUE`, and that flag does not selectively
/// inherit only the handles you asked to redirect — it inherits *every*
/// handle in the parent's own handle table that's currently marked
/// inheritable. `llmman run`'s own stdout/stderr handles — received from
/// the test harness above specifically so they could be piped and
/// captured, which requires them to be inheritable in the first place —
/// are exactly such handles. So when `llmman run` calls `ensure_server`
/// and spawns the detached daemon below, that daemon process — which is
/// deliberately left running indefinitely — ends up with its own
/// duplicate, still-open handle to `llmman run`'s original stdout/stderr
/// pipes, even though its own stdio was separately, correctly redirected
/// to the log file. The pipe's write end therefore never fully closes,
/// the test harness's reader thread blocks on `read()` forever waiting
/// for an EOF that can now never come, and the *entire* E2E job times out
/// 45 minutes later with no error of its own — indistinguishable from
/// there being no bug at all up to that point, since `llmman run` itself
/// already exited successfully.
///
/// The fix: explicitly clear the inheritable flag on this process's own
/// three standard handles, once, as early as possible (see `main.rs`) —
/// before this process (or anything it calls into) ever spawns another
/// child with `bInheritHandles = TRUE` for any reason. This does not
/// affect this process's own use of those handles at all (only whether a
/// *future child* of this process can inherit them), and is safe to call
/// unconditionally even when nothing was ever piped in the first place
/// (`GetStdHandle` returning a real console handle, or even
/// `INVALID_HANDLE_VALUE`/null when there's no console at all, are all
/// handled as no-ops below).
#[cfg(windows)]
mod win_handles {
    use std::ffi::c_void;

    #[allow(non_snake_case)]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> *mut c_void;
        fn SetHandleInformation(hObject: *mut c_void, dwMask: u32, dwFlags: u32) -> i32;
    }

    const STD_INPUT_HANDLE: i32 = -10;
    const STD_OUTPUT_HANDLE: i32 = -11;
    const STD_ERROR_HANDLE: i32 = -12;
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;

    pub fn disable_std_handle_inheritance() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
                // SAFETY: GetStdHandle/SetHandleInformation are ordinary
                // kernel32 calls; `handle` is only ever passed to the one
                // API that accepts exactly this value (including the
                // documented NULL/INVALID_HANDLE_VALUE "no such stream"
                // cases, which SetHandleInformation simply fails on
                // harmlessly — the return value is deliberately ignored).
                unsafe {
                    let handle = GetStdHandle(which);
                    if !handle.is_null() && handle != usize::MAX as *mut c_void {
                        let _ = SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
                    }
                }
            }
        });
    }
}

/// See `win_handles`' own module doc comment. Must be called once, as
/// early as possible in `main()` — a no-op on every platform but Windows.
pub fn disable_std_handle_inheritance() {
    #[cfg(windows)]
    win_handles::disable_std_handle_inheritance();
}

/// A single line of Ollama's streamed NDJSON progress protocol (see
/// api.ProgressResponse) — status text plus an optional error, and
/// either our own aggregate total/completed byte counts across the whole
/// pull/push (no digest), once cmd::serve::ollama's stream_ffi_progress has
/// one to report — see that function's own doc comment for where these
/// come from — or real Ollama's per-layer digest/total/completed.
#[derive(Deserialize)]
struct ProgressLine {
    status: Option<String>,
    error: Option<String>,
    /// A message the daemon needs a human to see — today, the outcome of
    /// a signature check (see `crate::verify::Verdict`). Relayed rather
    /// than logged because the daemon's stderr is a log file, while this
    /// process is the one attached to a terminal.
    #[serde(default)]
    notice: Option<String>,
    /// The manifest digest a push landed on, for `--sign-key` to sign
    /// here rather than in the daemon. See `cmd::serve::ollama`'s `push_impl`.
    /// With a `total`, the layer an Ollama pull line is about instead.
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

/// The indicatif template shared by `llmman pull`/`llmman push`'s
/// byte-level bar — deliberately similar in shape to `llmman transfer`'s
/// own mpb bars (go-shim/shared_oci.go's addLayerBar) so both commands'
/// output looks like the same family of progress bar.
///
/// The leading `{msg}` is the *reference* being pulled/pushed, not the
/// generic status word ("pulling"/"pushing") — that word was already
/// printed as its own plain line the moment byte-level reporting hadn't
/// started yet (see stream_progress), so repeating it here as the bar's
/// own label just showed "pulling" twice in a row, stacked directly on
/// top of each other, for no added information.
///
/// `wide_bar` fills whatever width the terminal has left (a fixed 32-cell
/// bar made the line 90 columns, which wrapped at 80), and the label is
/// truncated (`!`) so a long reference can't push the line past the edge.
fn progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:<20!} [{wide_bar:.cyan/blue}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>12}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

fn new_pull_bar(total: u64, reference: &str, target: ProgressDrawTarget) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(Some(total), target);
    pb.set_style(progress_bar_style());
    pb.set_message(reference.to_string());
    pb
}

/// POSTs `{"model": reference}` to `path` (e.g. "/api/pull" or "/api/push")
/// on the local daemon and renders each streamed line to stderr as it
/// arrives: a real byte-level progress bar for any line carrying a nonzero
/// `total` (see ProgressLine), or a plain status line otherwise — matching
/// how `llmman transfer`'s own mpb bars render foreground FFI progress.
///
/// One case gets special-cased rather than handed to the animated bar:
/// llmman's local store is content-addressed, so a very common pull is
/// one where every blob is already on disk under a *different* reference
/// (see go-shim/backend_podman.go's ProgressEventSkipped handling) —
/// go.podman.io/image credits that instantly, so the very first
/// byte-count line this function ever sees already has completed==total.
/// Handing that straight to indicatif produced a bar that renders
/// "already 100% full" from its first frame (never visibly animating at
/// all — indistinguishable, at a glance, from a bar that's stuck) with a
/// `bytes_per_sec` figure computed as total/elapsed since the bar was
/// just created a moment ago — a wildly inflated number (multiple GiB/s
/// or even TiB/s) that then decays back down over the next several
/// polling ticks purely because the (fixed) numerator is being divided by
/// a (growing) denominator, not because anything is actually slowing
/// down. Nothing about that display was wrong, exactly, but it looked
/// exactly like a hang to a real reader, however briefly. Detecting that
/// exact shape up front and printing one plain "already have it" line
/// instead is both more honest (no bytes are moving) and reads instantly
/// as "done", not "stuck".
///
/// Returns an error if the stream reports one, or if it ends without ever
/// reporting "success".
pub fn stream_progress(path: &str, reference: &str) -> anyhow::Result<()> {
    stream_progress_with(path, reference).map(|_| ())
}

/// [`stream_progress`], returning the manifest digest the stream
/// reported, when it reported one. `/api/push` does, for `--sign-key`.
pub fn stream_progress_with(path: &str, reference: &str) -> anyhow::Result<Option<String>> {
    let body = serde_json::json!({"model": reference});

    let client = client_builder()?
        .timeout(None) // model transfers can take much longer than any sane fixed timeout
        .build()
        .context("build http client")?;
    let resp = client
        .post(format!("{}{path}", server()))
        .json(&body)
        .send()
        .with_context(|| format!("request {path} for {reference}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path} {reference}: server returned {status}: {body}");
    }

    let mut progress = PullProgress::new(reference, ProgressDrawTarget::stderr);
    let (mut out, mut err) = (std::io::stdout(), std::io::stderr());
    for line in std::io::BufReader::new(resp).lines() {
        let line = line.context("read response stream")?;
        progress.line(&line, &mut out, &mut err)?;
    }
    progress.finish(&mut out)
}

/// The state [`stream_progress_with`] keeps across a pull/push stream's
/// lines, apart from the HTTP so it can be fed fake streams.
struct PullProgress<'a> {
    reference: &'a str,
    draw_target: fn() -> ProgressDrawTarget,
    saw_success: bool,
    pushed_digest: Option<String>,
    last_status: String,
    bar: Option<ProgressBar>,
    // Set once we've printed the "already have it" shortcut line (see
    // `stream_progress`'s doc comment), so a pull whose bytes were all
    // instant-credited doesn't print that line again on every further
    // completed==total poll before "success" finally arrives.
    printed_instant_complete: bool,
    // Ollama's per-layer progress, by digest. llmman's daemon sends one
    // aggregate with no digest, which bypasses this.
    layers: std::collections::HashMap<String, (u64, u64)>,
    // The total of an instant-credited first byte line not yet announced.
    // From a layer-by-layer stream it waits for the end of the layers, as
    // the first layer on its own can be complete while the next is not.
    instant_total: Option<u64>,
    had_bar: bool,
}

impl<'a> PullProgress<'a> {
    fn new(reference: &'a str, draw_target: fn() -> ProgressDrawTarget) -> Self {
        Self {
            reference,
            draw_target,
            saw_success: false,
            pushed_digest: None,
            last_status: String::new(),
            bar: None,
            printed_instant_complete: false,
            layers: std::collections::HashMap::new(),
            instant_total: None,
            had_bar: false,
        }
    }

    /// Prints the pending "already have it" line, if any.
    fn announce_instant(&mut self, out: &mut dyn std::io::Write) {
        let Some(total) = self.instant_total.take() else {
            return;
        };
        if !self.printed_instant_complete {
            let _ = writeln!(
                out,
                "Already have {} ({})",
                self.reference,
                crate::fmt::human_size(total)
            );
            self.printed_instant_complete = true;
        }
        self.last_status = "pulling".to_string(); // suppress a redundant plain "pulling" line
    }

    /// Handles one line of the stream; `out` is stdout and `err` stderr.
    fn line(
        &mut self,
        line: &str,
        out: &mut dyn std::io::Write,
        err: &mut dyn std::io::Write,
    ) -> anyhow::Result<()> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        let Ok(msg) = serde_json::from_str::<ProgressLine>(line) else {
            return Ok(()); // tolerate stray non-JSON keepalive output
        };
        let digest = msg.digest.filter(|d| !d.is_empty());
        let total = msg.total.filter(|&t| t > 0);
        // A digest alone is a push's result; with a total it is one layer
        // of an Ollama pull.
        if let (Some(d), None) = (&digest, total) {
            self.pushed_digest = Some(d.clone());
            return Ok(());
        }
        if let Some(notice) = msg.notice.filter(|n| !n.is_empty()) {
            // suspend(), so the message doesn't land in the middle of a
            // half-drawn bar and get overwritten by the next frame.
            match &self.bar {
                Some(b) => b.suspend(|| {
                    let _ = writeln!(err, "[llmman] {notice}");
                }),
                None => {
                    let _ = writeln!(err, "[llmman] {notice}");
                }
            }
            return Ok(());
        }
        if let Some(e) = msg.error.filter(|e| !e.is_empty()) {
            if let Some(b) = self.bar.take() {
                b.abandon(); // leave whatever was drawn in place instead of clearing it
            }
            // Only prefix with `reference` if the error doesn't already
            // mention it — many pull failures (e.g. containerd's "not
            // found") already embed the exact reference themselves, and
            // piling this prefix on unconditionally produced the same
            // reference two or three times over in one error line.
            if e.contains(self.reference) {
                anyhow::bail!("{e}");
            }
            anyhow::bail!("{}: {e}", self.reference);
        }

        if let Some(total) = total {
            let completed = msg.completed.unwrap_or(0).min(total);
            let (total, completed) = match digest {
                Some(d) => {
                    self.layers.insert(d, (total, completed));
                    self.layers
                        .values()
                        .fold((0, 0), |(t, c), (lt, lc)| (t + lt, c + lc))
                }
                None => (total, completed),
            };

            // Instant-credit shortcut (see `stream_progress`'s doc
            // comment): the *first* poll this function ever sees a
            // nonzero total on is already ≥99% complete, so whatever
            // sliver (if any) is still outstanding isn't a real,
            // animatable transfer — it's noise from total/completed being
            // credited by two separate calls a poll interval apart on the
            // Go side (go-shim/progress_state.go's progressAddTotal/
            // progressAddCompleted), not a meaningful amount of remaining
            // work. A genuine transfer's first-ever byte line is nowhere
            // near this close to done (that's what makes the animated
            // bar below worth showing at all). Checked only while
            // bar.is_none(): once a real bar has been created because an
            // earlier poll didn't clear this bar, later polls update it
            // normally even if they happen to also reach ≥99%.
            //
            // 99%, not 100% exactly: an earlier version of this required
            // completed==total on the nose and missed a real case where
            // two multi-hundred-MB blobs were skip-credited instantly but
            // a trailing few-hundred-byte config blob's own (separate,
            // genuinely tiny) transfer hadn't landed yet on that same
            // poll — 99.99996% done, not 100.000% done, but just as
            // clearly "nothing left worth animating" either way.
            let layered = !self.layers.is_empty();
            if self.bar.is_none() && completed * 100 >= total * 99 && !(layered && self.had_bar) {
                self.instant_total = Some(total);
                if !layered {
                    self.announce_instant(out);
                }
                return Ok(());
            }

            // A byte-level progress line: render/update the bar instead of
            // printing a new line for every update.
            self.instant_total = None;
            self.had_bar = true;
            let (reference, draw_target) = (self.reference, self.draw_target);
            let pb = self
                .bar
                .get_or_insert_with(|| new_pull_bar(total, reference, draw_target()));
            pb.set_length(total);
            pb.set_position(completed);
            return Ok(());
        }
        // No byte counts on this line: finish/clear any bar in progress
        // before falling back to plain status text, so the two don't
        // interleave on the same terminal lines.
        self.announce_instant(out);
        if let Some(b) = self.bar.take() {
            b.finish_and_clear();
        }
        if let Some(status) = msg.status {
            if !status.is_empty() && status != self.last_status {
                let _ = writeln!(out, "{status}");
                self.last_status = status;
            }
            self.saw_success = self.last_status == "success";
        }
        Ok(())
    }

    /// The end of the stream: the manifest digest it reported, if it did.
    fn finish(mut self, out: &mut dyn std::io::Write) -> anyhow::Result<Option<String>> {
        self.announce_instant(out);
        if let Some(b) = self.bar.take() {
            b.finish_and_clear();
        }
        if !self.saw_success {
            anyhow::bail!("{}: stream ended without a success status", self.reference);
        }
        Ok(self.pushed_digest)
    }
}

/// The part of Ollama's `api.ShowResponse` that `llmman run` and
/// `llmman launch` read.
#[derive(Debug, Default, Deserialize)]
pub struct ShowResponse {
    /// `"completion"`, `"vision"`, … — see `crate::modelpack::capabilities`.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// The model's chat template, if any (see `crate::modelpack::chat_template`).
    #[serde(default)]
    pub template: Option<String>,
    /// The GGUF header's metadata keys. Untyped so that any shape an
    /// older daemon sends decodes.
    #[serde(default)]
    pub model_info: serde_json::Value,
}

impl ShowResponse {
    /// The trained context, `{general.architecture}.context_length`;
    /// `None` when absent or zero.
    pub fn context_length(&self) -> Option<u64> {
        let arch = self.model_info.get("general.architecture")?.as_str()?;
        self.model_info
            .get(format!("{arch}.context_length"))?
            .as_u64()
            .filter(|n| *n > 0)
    }

    /// The template's thinking controls; `None` without a template.
    pub fn thinking_controls(&self) -> Option<crate::chat_template::ThinkingControls> {
        self.template
            .as_deref()
            .map(crate::chat_template::thinking_controls)
    }

    /// Ollama's `RunHandler`: `opts.MultiModal` = vision or audio.
    pub fn multimodal(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_VISION || c == "audio")
    }

    /// Takes images *in* — unlike [`Self::image`], which generates them,
    /// and [`Self::multimodal`], which audio alone also satisfies.
    pub fn vision(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_VISION)
    }

    /// A decision model (see `cmd::run::decide`).
    pub fn decision(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_DECISION)
    }

    /// An image generation model (see `crate::imagegen`).
    pub fn image(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == crate::modelpack::CAPABILITY_IMAGE)
    }
}

/// Ollama's `client.Show`: a read-only `/api/show` lookup. `Ok(None)` is
/// a 404 only; any other failure is an error, as in `showOrPullModel`.
pub fn show(reference: &str) -> anyhow::Result<Option<ShowResponse>> {
    let resp = client()?
        .post(format!("{}/api/show", server()))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request /api/show for {reference}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{}", api_error(&body).unwrap_or(body));
    }
    // An older daemon omits `capabilities`; serde(default) handles it.
    let body = resp
        .json::<ShowResponse>()
        .with_context(|| format!("decode /api/show for {reference}"))?;
    Ok(Some(body))
}

/// Ensures `reference` is in the daemon's local store, pulling it (with
/// `llmman pull`'s progress) if not, and returns its `/api/show` info.
///
/// Mirrors ollama's `showOrPullModel`: Show, Pull on a 404, Show again —
/// so a bad reference fails before the prompt loop.
pub fn ensure_model_pulled(reference: &str) -> anyhow::Result<ShowResponse> {
    if let Some(info) = show(reference)? {
        return Ok(info);
    }
    stream_progress("/api/pull", reference)?;
    show(reference)?.ok_or_else(|| anyhow::anyhow!("{reference}: not found after pull"))
}

/// Pushes `reference` via the daemon's `/api/push` and returns the
/// manifest digest it landed on, for `cmd::push --sign-key` to sign.
///
/// The daemon is deliberately not asked to sign: see
/// `cmd::serve::ollama`'s `push_impl` for why a caller-supplied key path
/// is not something an unauthenticated loopback endpoint may accept.
pub fn push(reference: &str) -> anyhow::Result<Option<String>> {
    stream_progress_with("/api/push", reference)
}

/// POSTs the Ollama unload sentinel (`{"model": reference, "keep_alive":
/// 0}`, no `prompt` field — i.e. an empty prompt) to `/api/generate` —
/// see `cmd::serve::ollama`'s `handle_ollama_generate` for the server side that
/// reads this exact shape as an immediate-unload request, mirroring real
/// Ollama's own `ollama stop` (`cmd/cmd.go`'s `loadOrUnloadModel`). Used
/// by `llmman stop`.
///
/// `Ok(false)` is the daemon's "no such model" answer, which `llmman
/// stop` renders as its error; `Ok(true)` covers a model loaded or not,
/// which the unload does not distinguish, matching ollama. Any other
/// failure is reported with its status and body, a 404 included: a proxy
/// in front of the daemon answers 404 too, and "couldn't find model"
/// would be the wrong story for that.
pub fn unload(reference: &str) -> anyhow::Result<bool> {
    let resp = client()?
        .post(format!("{}/api/generate", server()))
        .json(&serde_json::json!({"model": reference, "keep_alive": 0}))
        .send()
        .with_context(|| format!("request /api/generate (unload) for {reference}"))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(true);
    }
    let body = resp.text().unwrap_or_default();
    if status == reqwest::StatusCode::NOT_FOUND && is_model_not_found_body(&body, reference) {
        return Ok(false);
    }
    anyhow::bail!("stop {reference}: server returned {status}: {body}");
}

/// Loads `reference` and reports the context window the runner actually
/// got, from `/api/ps` — the post-load `n_ctx`, not the trained context
/// `/api/show` reports. `None` when the daemon does not report one: a
/// vLLM- or MLX-backed model, or a load that failed.
///
/// The load is the point. `ensure_model_pulled` only pulls, and
/// `ensure_server` preloads only when it spawns, so on a reused daemon
/// the model is usually still cold — and an OOM retry halves `--ctx-size`
/// during that load, which is exactly the value a caller needs to read
/// back rather than predict.
pub fn loaded_context_length(reference: &str) -> Option<u64> {
    // Ollama's load sentinel: no prompt, and no `keep_alive` of its
    // own, so the daemon's default decides how long the model stays
    // (an absent field never reads as an unload — see
    // `is_explicit_unload` in cmd::serve). This only asks for the load,
    // so `/api/ps` has something to report.
    client()
        .ok()?
        .post(format!("{}/api/generate", server()))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .ok()?
        .error_for_status()
        .ok()?;
    running_context_length(reference)
}

/// `reference`'s window from `/api/ps`, without loading it: `None` when it
/// is not loaded, or the daemon reports no window or cannot be asked.
pub fn running_context_length(reference: &str) -> Option<u64> {
    let ps: PsWindows = get_json("/api/ps").ok()?;
    ps_window(ps.models, reference)
}

/// `reference`'s window among the models `/api/ps` reports loaded.
///
/// The tag is defaulted on both sides: `/api/ps` answers under the
/// reference `ensure_model` loaded the model as, which it ran through
/// `default_tag` first, while a caller's own spelling is untagged
/// (`resolve_ollama_api` leaves `gemma4` as `docker.io/ai/gemma4`). An
/// exact match therefore finds nothing and reports no window at all.
fn ps_window(models: Vec<PsWindow>, reference: &str) -> Option<u64> {
    let local = crate::storage::default_tag(crate::hybrid::local_half(reference));
    models
        .into_iter()
        .find(|m| crate::storage::default_tag(&m.name) == local)
        .and_then(|m| m.context_length)
}

/// The two `/api/ps` fields [`loaded_context_length`] needs; the CLI's
/// own `cmd::ps` parses the rest for display.
#[derive(serde::Deserialize)]
struct PsWindows {
    models: Vec<PsWindow>,
}

#[derive(serde::Deserialize)]
struct PsWindow {
    name: String,
    context_length: Option<u64>,
}

#[cfg(test)]
impl PsWindow {
    fn new(name: &str, context_length: Option<u64>) -> Self {
        Self {
            name: name.to_string(),
            context_length,
        }
    }
}

/// The `{"error":"model '<name>' not found"}` this daemon sends for
/// `reference` (see `unload_model` in cmd::serve), with the name the
/// daemon was asked for, a pair's local half; parsed through `api_error`,
/// so an HTML page or a proxy's own JSON does not pass by containing the
/// words, and a 404 about some other model does not pass either.
pub(crate) fn is_model_not_found_body(body: &str, reference: &str) -> bool {
    let expected = format!("model '{}' not found", crate::hybrid::local_half(reference));
    api_error(body).is_some_and(|e| e == expected)
}

/// A plain `GET {server()}{path}` returning the parsed JSON body — for
/// callers (`ps`, and the provider helpers below) that don't need
/// `stream_progress`'s newline-delimited-JSON streaming, just a single
/// request/response.
pub fn get_json<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let resp = client()?
        .get(format!("{}{path}", server()))
        .send()
        .with_context(|| format!("request {path}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        // The daemon's errors are `{"error": "<message>"}` (see AppError
        // in cmd::serve), and that message is written for a person — the
        // "did you mean" suggestions, say. An envelope and a status code
        // around it bury the part meant to be read.
        anyhow::bail!(
            "{}",
            api_error(&body).unwrap_or(format!("{path}: server returned {status}: {body}"))
        );
    }
    resp.json()
        .with_context(|| format!("parse response from {path}"))
}

/// The `error` message out of a failed response body: an `AppError`'s
/// `{"error": "…"}`, or the `{"error": {"message": "…"}}` of a backend it was
/// passed through from; `None` for a body that is neither (an empty 404 from
/// axum's router, say). Public for the clients that read a body themselves.
pub fn api_error(body: &str) -> Option<String> {
    let error = serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .get("error")?
        .clone();
    error
        .as_str()
        .or_else(|| error.get("message")?.as_str())
        .map(str::to_string)
        .filter(|e| !e.is_empty())
}

// ---------------------------------------------------------------------------
// Providers — clients of the daemon's own /llmman API
// ---------------------------------------------------------------------------

/// Wire shape of one entry in `GET /llmman/providers` — see
/// `ProviderSummary` in `cmd::serve` for the server side these field
/// names must match.
#[derive(Debug, Deserialize)]
pub struct ProviderSummary {
    pub id: String,
    pub name: String,
    /// Absent for a configured provider that names no variable.
    #[serde(default)]
    pub key_env: Option<String>,
    /// Whether the key is set *where the daemon runs*. What this process
    /// itself holds is [`ProviderSummary::key_here`].
    pub key_set: bool,
    /// Whether the daemon would actually spend that key for a request
    /// presenting none. Only it can tell: that depends on how it is
    /// bound, which this process's `LLMMAN_HOST` says nothing about.
    #[serde(default)]
    pub key_usable: bool,
    /// Whether a keyless request is forwarded (see
    /// `Provider::key_optional`); absent from an older daemon means no.
    #[serde(default)]
    pub key_optional: bool,
    /// What the daemon would spend, from `credential` in
    /// `GET /llmman/providers`; absent when it holds none, and from an
    /// older daemon.
    #[serde(default)]
    pub credential: Option<CredentialStatus>,
    pub models: usize,
}

/// Wire shape of `credential` in `GET /llmman/providers` — see
/// `crate::credentials::Status`. Never a secret.
#[derive(Debug, Clone, Deserialize)]
pub struct CredentialStatus {
    /// `env`, `config`, `api_key`, `oauth`, `token`, `command`, `file` or
    /// `cli_login`.
    pub kind: String,
    /// `ready`, `refresh` or `expired`.
    pub state: String,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ProvidersResponse {
    providers: Vec<ProviderSummary>,
}

impl ProviderSummary {
    /// Whether *this* process holds the key — the other way one reaches
    /// a provider, sent per request (see `client_api_key` in cmd::serve).
    pub fn key_here(&self) -> bool {
        crate::providers::key_for(&self.id, self.key_env.as_deref()).is_some()
    }
}

/// Every provider `--provider` accepts, as the running daemon sees them.
pub fn providers() -> anyhow::Result<Vec<ProviderSummary>> {
    Ok(get_json::<ProvidersResponse>("/llmman/providers")?.providers)
}

/// Wire shape of `GET /llmman/providers/:id` — one provider, with the
/// models it serves. See `ProviderResponse` in `cmd::serve`.
#[derive(Debug, Deserialize)]
pub struct ProviderDetail {
    pub id: String,
    pub name: String,
    pub base_url: String,
    /// See [`ProviderSummary::key_env`].
    #[serde(default)]
    pub key_env: Option<String>,
    /// See [`ProviderSummary::key_set`].
    pub key_set: bool,
    /// See [`ProviderSummary::key_usable`].
    #[serde(default)]
    pub key_usable: bool,
    /// See [`ProviderSummary::key_optional`].
    #[serde(default)]
    pub key_optional: bool,
    pub models: Vec<ProviderModel>,
}

/// One model in a [`ProviderDetail`].
#[derive(Debug, Deserialize)]
pub struct ProviderModel {
    pub id: String,
    /// `None` where models.dev publishes no price, which is not free.
    #[serde(default)]
    pub cost: Option<ModelCost>,
    /// See `crate::providers::Model::thinking`; `None` from an older daemon too.
    #[serde(default)]
    pub thinking: Option<Vec<String>>,
    /// The window this model can hold (models.dev `limit.context`).
    /// `None` from an older daemon, and for a provider defined in
    /// `llmman.conf`, which has no catalog entry.
    #[serde(default)]
    pub context: Option<u64>,
    /// The most this model will emit in one reply (models.dev
    /// `limit.output`), `None` on the same terms as `context`.
    #[serde(default)]
    pub output: Option<u32>,
}

/// US dollars per million tokens (see [`crate::providers::Cost`]).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    /// `None` where models.dev publishes none, or from an older daemon.
    #[serde(default)]
    pub cache_read: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
    #[serde(default)]
    pub reasoning: Option<f64>,
}

impl ProviderDetail {
    /// This provider's key as *this* process resolves it, to travel with
    /// each request (see `client_api_key` in cmd::serve). `None` need not
    /// be fatal — the daemon may hold it ([`ProviderDetail::key_set`]).
    ///
    /// The daemon names the variable, so a process squatting the port
    /// could name an unrelated secret — inside the trust boundary this
    /// whole module sits in either way: that daemon is handed every
    /// prompt, and every key, regardless. It does not name the
    /// `llmman.conf` entry, which is keyed by the provider id.
    pub fn api_key(&self) -> Option<String> {
        crate::providers::key_for(&self.id, self.key_env.as_deref())
    }

    /// Just the ids, for a caller that only needs to name one (see
    /// `crate::providers::example_models`).
    pub fn model_ids(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.id.as_str()).collect()
    }

    /// The catalog's thinking levels for `model`, if it lists any.
    pub fn thinking_levels(&self, model: &str) -> Option<Vec<String>> {
        self.models
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.thinking.clone())
    }

    /// The catalog's window for `model` (models.dev `limit.context`).
    /// `None` from an older daemon, and for a provider defined in
    /// `llmman.conf`.
    pub fn context_window(&self, model: &str) -> Option<u64> {
        self.models
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.context)
    }

    /// The catalog's output ceiling for `model` (models.dev
    /// `limit.output`), on the same terms as [`Self::context_window`].
    /// What the provider will actually accept as a max-output request,
    /// which is not derivable from the window: a 128k-context model may
    /// still refuse to be asked for more than 8k of reply.
    pub fn max_output(&self, model: &str) -> Option<u32> {
        self.models
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.output)
    }

    /// Where a key for this provider would go (see
    /// `crate::providers::key_hint`).
    pub fn key_hint(&self) -> String {
        crate::providers::key_hint(&self.id, self.key_env.as_deref())
    }

    /// Resolves the key this process would send, warning once if it is
    /// about to cross the network in cleartext — the daemon's own
    /// startup warning only covers the key *it* holds.
    pub fn client_key(&self) -> Option<String> {
        let key = self.api_key();
        if key.is_some() && self.base_url.starts_with("http://") {
            eprintln!(
                "[llmman] warning: the API key for {} goes to {} over plain http",
                self.name, self.base_url
            );
        }
        key
    }

    /// Warns when this provider does not list `model` — a warning, since
    /// models.dev is a snapshot and a provider can serve a model before it
    /// lists one. Silent for a configured provider whose `/models` said
    /// nothing: there is no list to be absent from.
    pub fn warn_unlisted(&self, model: &str) {
        if self.key_optional && self.models.is_empty() {
            return;
        }
        if !self.models.iter().any(|m| m.id == model) {
            eprintln!(
                "[llmman] warning: {} does not list model {model:?}\n{}",
                self.name,
                crate::providers::example_models(&self.name, &self.model_ids())
            );
        }
    }

    /// Whether the *daemon's* own key gets spent for a request that
    /// presents none — its own answer (see `resolve_remote_target` in
    /// cmd::serve), not a guess from this process's `LLMMAN_HOST`, which
    /// describes the host it dials rather than how that daemon is bound.
    pub fn daemon_key_usable(&self) -> bool {
        self.key_usable
    }
}

/// Looks one provider up in the daemon's catalog. An unknown id comes
/// back as the daemon's own error, near-matches included (see
/// `crate::providers::unknown_provider_error`).
///
/// `model` is the one the caller is about to use, if any: a daemon whose
/// catalog lacks it re-fetches before answering. Older daemons ignore it.
pub fn provider(id: &str, model: Option<&str>) -> anyhow::Result<ProviderDetail> {
    get_json(&provider_path(id, model))
}

/// The route for [`provider`], `model` as one encoded query value.
fn provider_path(id: &str, model: Option<&str>) -> String {
    let query = model
        .map(|m| format!("?model={}", encode_path_segment(m)))
        .unwrap_or_default();
    format!("/llmman/providers/{}{query}", encode_path_segment(id))
}

/// Percent-encodes everything outside the unreserved set, so a typo'd id
/// stays one path segment: `a/b` would otherwise miss the daemon's
/// single-segment route (an empty 404 from axum's router instead of
/// llmman's "unknown provider"), and `../` would address another. Also
/// right for a query value, where `&` and `#` would end it.
fn encode_path_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_autostart_turns_a_missing_server_into_an_error_naming_the_address() {
        assert!(require_server(true, "http://127.0.0.1:1").is_ok());
        let err = require_server(false, "http://127.0.0.1:1")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("http://127.0.0.1:1") && err.contains(NO_AUTOSTART_VAR),
            "{err}"
        );
    }

    const NAME: &str = "serve-127.0.0.1-0";

    fn acquire(dir: &Path, holder_wait: Duration) -> anyhow::Result<PortLock> {
        PortLock::acquire_in(dir.to_path_buf(), NAME.into(), holder_wait)
    }

    #[test]
    fn a_record_is_named_by_the_address_and_the_port() {
        assert_eq!(
            record_port("serve-127.0.0.1-0-53097.port", NAME),
            Some(53097)
        );
        assert_eq!(record_port("serve-127.0.0.1-0-0.port", NAME), None);
        assert_eq!(record_port("serve-127.0.0.1-0-x.port", NAME), None);
        assert_eq!(record_port("serve-127.0.0.1-0.lock", NAME), None);
        assert_eq!(record_port("serve-127.0.0.10-0-53097.port", NAME), None);
    }

    #[test]
    fn clients_use_a_record_only_while_its_daemon_holds_it() {
        let dir = tempfile::tempdir().unwrap();
        // A new daemon holds the lock; the record of one that crashed is still there.
        let stale = dir.path().join(format!("{NAME}-53097.port"));
        std::fs::write(&stale, "").unwrap();
        let lock = acquire(dir.path(), Duration::ZERO).unwrap();
        assert_eq!(held_port(dir.path(), NAME), None);

        let record = lock.publish(53098).unwrap();
        assert_eq!(held_port(dir.path(), NAME), Some(53098));
        assert!(!stale.exists());

        drop(record);
        assert_eq!(held_port(dir.path(), NAME), None);
        assert!(!dir.path().join(format!("{NAME}-53098.port")).exists());
    }

    #[test]
    fn a_second_daemon_is_refused_once_the_first_publishes() {
        let dir = tempfile::tempdir().unwrap();
        let first = acquire(dir.path(), Duration::ZERO).unwrap();
        let started = std::time::Instant::now();
        let publisher = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let record = first.publish(53097).unwrap();
            (first, record)
        });
        let err = acquire(dir.path(), Duration::from_secs(30))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("another llmman serve"), "{err}");
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(100) && waited < Duration::from_secs(30),
            "{waited:?}"
        );
        let _first = publisher.join().unwrap();
    }

    #[test]
    fn a_second_daemon_takes_the_lock_once_the_first_exits_without_publishing() {
        let dir = tempfile::tempdir().unwrap();
        let first = acquire(dir.path(), Duration::ZERO).unwrap();
        let err = acquire(dir.path(), Duration::ZERO)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("another llmman serve"), "{err}");
        let exit = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(first);
        });
        // The wait also covers a child another test forked, which holds a
        // copy of the descriptor, and so the lock, until it execs.
        acquire(dir.path(), Duration::from_secs(30)).unwrap();
        exit.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn the_lock_and_record_are_for_their_owner_alone() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        // A lock left readable by others, as one is never removed.
        let lock_path = dir.path().join(format!("{NAME}.lock"));
        std::fs::write(&lock_path, "").unwrap();
        std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let lock = acquire(dir.path(), Duration::ZERO).unwrap();
        let _record = lock.publish(53097).unwrap();
        for name in [format!("{NAME}.lock"), format!("{NAME}-53097.port")] {
            let mode = std::fs::metadata(dir.path().join(&name))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "{name}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn clients_ignore_a_record_others_can_open() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        // Another user locks, through a read-only descriptor, a record an
        // earlier version left readable by everyone.
        let path = dir.path().join(format!("{NAME}-53097.port"));
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let holder = std::fs::File::open(&path).unwrap();
        holder.lock().unwrap();
        assert_eq!(held_port(dir.path(), NAME), None);

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(held_port(dir.path(), NAME), Some(53097));
    }

    #[test]
    fn a_client_checking_a_record_does_not_stop_its_daemon_publishing_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("{NAME}-53097.port"));
        std::fs::write(&path, "").unwrap();
        let probe = std::fs::File::open(&path).unwrap();
        probe.try_lock_shared().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(probe);
        });
        let lock = acquire(dir.path(), Duration::ZERO).unwrap();
        let _record = lock.publish(53097).unwrap();
        release.join().unwrap();
        assert_eq!(held_port(dir.path(), NAME), Some(53097));
    }

    #[test]
    fn a_record_that_cannot_be_created_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let lock = acquire(dir.path(), Duration::ZERO).unwrap();
        std::fs::create_dir(dir.path().join(format!("{NAME}-53097.port"))).unwrap();
        let err = format!("{:#}", lock.publish(53097).err().unwrap());
        assert!(err.contains(&format!("{NAME}-53097.port")), "{err}");
    }

    #[test]
    fn daemon_identity_requires_llmman_fields() {
        let mut identity: DaemonIdentity =
            serde_json::from_str(r#"{"version":"0.1.0","exe":"/bin/llmman","pid":42}"#).unwrap();
        assert!(identity.is_llmman());
        identity.pid = Some(1);
        assert!(identity.is_llmman());
        identity.pid = Some(0);
        assert!(!identity.is_llmman());
        identity.pid = None;
        assert!(!identity.is_llmman());
        identity.pid = Some(42);
        identity.version = None;
        assert!(!identity.is_llmman());
        identity.version = Some("0.1.0".into());
        identity.exe = None;
        assert!(!identity.is_llmman());
    }

    fn show_response(body: serde_json::Value) -> ShowResponse {
        serde_json::from_value(body).expect("a ShowResponse")
    }

    #[test]
    fn context_length_is_read_under_the_declared_architecture() {
        let info = show_response(serde_json::json!({
            "model_info": {
                "general.architecture": "qwen35",
                "qwen35.context_length": 32768,
                "llama.context_length": 4096,
            }
        }));
        assert_eq!(info.context_length(), Some(32768));
    }

    #[test]
    fn context_length_is_none_without_a_usable_header() {
        assert_eq!(show_response(serde_json::json!({})).context_length(), None);
        for model_info in [
            // What a daemon older than #512 sends.
            serde_json::json!({ "digest": "sha256:a", "size": 1 }),
            serde_json::json!(null),
            serde_json::json!({}),
            serde_json::json!({ "llama.context_length": 4096 }),
            serde_json::json!({
                "general.architecture": "llama",
                "llama.context_length": "4096",
            }),
            serde_json::json!({
                "general.architecture": "llama",
                "llama.context_length": 0,
            }),
        ] {
            let info = show_response(serde_json::json!({ "model_info": model_info.clone() }));
            assert_eq!(info.context_length(), None, "{model_info}");
        }
    }

    /// The two locality questions differ on exactly one host, and it is
    /// the one that matters: a wildcard bind is reached over loopback
    /// from here, *and* by anyone who can route to this machine.
    /// Collapsing them either leaks the daemon's credentials to the
    /// network or refuses `--provider` on a perfectly local hop.
    #[test]
    fn wildcard_binds_are_locally_reachable_but_not_only_locally() {
        // The host-taking forms the two public predicates delegate to,
        // since those read a PARSED_HOST fixed once per process.
        for host in ["127.0.0.1", "::1", "localhost", "LOCALHOST"] {
            assert!(host_connects_over_loopback(host), "{host}");
            assert!(host_reachable_only_locally(host), "{host}");
        }
        // The one host the two disagree on, and the reason there are two:
        // we reach it over loopback, but so does the rest of the network.
        for wildcard in ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"] {
            assert!(host_connects_over_loopback(wildcard), "{wildcard}");
            assert!(!host_reachable_only_locally(wildcard), "{wildcard}");
        }
        for remote in ["192.168.1.50", "10.0.0.1", "example.com"] {
            assert!(!host_connects_over_loopback(remote), "{remote}");
            assert!(!host_reachable_only_locally(remote), "{remote}");
        }
    }

    /// A typo'd id has to stay one segment, or the daemon's "unknown
    /// provider" answer never gets printed.
    /// The model stays one query value, whatever `/`, `&` or `#` it holds.
    #[test]
    fn provider_path_carries_the_model_as_one_query_value() {
        assert_eq!(
            provider_path("anthropic", None),
            "/llmman/providers/anthropic"
        );
        assert_eq!(
            provider_path("anthropic", Some("claude-sonnet-5-5")),
            "/llmman/providers/anthropic?model=claude-sonnet-5-5"
        );
        assert_eq!(
            provider_path("openrouter", Some("anthropic/claude-sonnet-5.5&x=1#f")),
            "/llmman/providers/openrouter?model=anthropic%2Fclaude-sonnet-5.5%26x%3D1%23f"
        );
    }

    #[test]
    fn encode_path_segment_keeps_a_typo_inside_one_segment() {
        assert_eq!(encode_path_segment("openrouter"), "openrouter");
        assert_eq!(
            encode_path_segment("together-ai_1.0~x"),
            "together-ai_1.0~x"
        );
        assert_eq!(encode_path_segment("a/b"), "a%2Fb");
        assert_eq!(encode_path_segment("../api/tags"), "..%2Fapi%2Ftags");
        assert_eq!(encode_path_segment("open ai"), "open%20ai");
        assert_eq!(encode_path_segment("a?b#c"), "a%3Fb%23c");
        // Non-ASCII goes out as its UTF-8 bytes, percent-encoded.
        assert_eq!(encode_path_segment("é"), "%C3%A9");
    }

    /// Whether the daemon spends its own key is the daemon's answer, not
    /// a guess from this process's `LLMMAN_HOST`: a daemon bound to
    /// `0.0.0.0` withholds it while the client dialing loopback would
    /// conclude the opposite, and only find out at the first request.
    #[test]
    fn daemon_key_usable_is_the_daemons_own_answer() {
        let body = |extra: &str| {
            serde_json::from_str::<ProviderDetail>(&format!(
                r#"{{"id":"openai","name":"OpenAI","base_url":"https://api.openai.com/v1",
                     "key_env":"LLMMAN_TEST_PROVIDER_KEY_UNSET","models":[]{extra}}}"#
            ))
            .expect("parses")
        };
        assert!(body(r#","key_set":true,"key_usable":true"#).daemon_key_usable());
        // Has the key, would withhold it — a wildcard-bound daemon.
        assert!(!body(r#","key_set":true,"key_usable":false"#).daemon_key_usable());
        // Says nothing at all: assume the conservative answer.
        assert!(!body(r#","key_set":true"#).daemon_key_usable());
    }

    /// The daemon's messages are written to be read; the JSON envelope
    /// around them is not.
    #[test]
    fn api_error_unwraps_the_daemons_own_message() {
        assert_eq!(
            api_error(r#"{"error":"unknown provider \"togethr\""}"#).as_deref(),
            Some("unknown provider \"togethr\"")
        );
        // Not an AppError body: nothing to unwrap, and the caller falls
        // back to reporting the status and the raw body.
        assert_eq!(api_error(""), None);
        assert_eq!(api_error("Not Found"), None);
        assert_eq!(api_error(r#"{"error":""}"#), None);
        // A backend's own envelope, passed through.
        assert_eq!(
            api_error(r#"{"error":{"code":501,"message":"not a decision model"}}"#).as_deref(),
            Some("not a decision model")
        );
        assert_eq!(api_error(r#"{"detail":"nope"}"#), None);
    }

    /// Unset/blank `LLMMAN_HOST` — the common case — resolves to the
    /// documented default.
    #[test]
    fn parse_host_defaults_when_unset_or_blank() {
        assert_eq!(
            parse_host(None),
            ("http".to_string(), "127.0.0.1".to_string(), 17434)
        );
        assert_eq!(
            parse_host(Some("  ")),
            ("http".to_string(), "127.0.0.1".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_host() {
        assert_eq!(
            parse_host(Some("0.0.0.0")),
            ("http".to_string(), "0.0.0.0".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_host_and_port() {
        assert_eq!(
            parse_host(Some("0.0.0.0:8080")),
            ("http".to_string(), "0.0.0.0".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_accepts_an_explicit_scheme() {
        assert_eq!(
            parse_host(Some("https://example.com:9999")),
            ("https".to_string(), "example.com".to_string(), 9999)
        );
    }

    /// An explicit `http://`/`https://` scheme with no port shifts the
    /// *default* port to 80/443.
    #[test]
    fn parse_host_shifts_default_port_for_http_and_https_schemes() {
        assert_eq!(
            parse_host(Some("http://example.com")),
            ("http".to_string(), "example.com".to_string(), 80)
        );
        assert_eq!(
            parse_host(Some("https://example.com")),
            ("https".to_string(), "example.com".to_string(), 443)
        );
    }

    /// A non-http(s) scheme (or none at all) keeps llmman's own default
    /// port rather than shifting to 80.
    #[test]
    fn parse_host_keeps_default_port_for_other_schemes() {
        assert_eq!(
            parse_host(Some("grpc://example.com")),
            ("grpc".to_string(), "example.com".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_accepts_a_bracketed_ipv6_literal_with_port() {
        assert_eq!(
            parse_host(Some("[::1]:8080")),
            ("http".to_string(), "::1".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_accepts_a_bare_ipv6_literal_with_no_port() {
        assert_eq!(
            parse_host(Some("::1")),
            ("http".to_string(), "::1".to_string(), 17434)
        );
    }

    /// An out-of-range/unparseable port falls back to the default port,
    /// keeping whatever host was given.
    #[test]
    fn parse_host_falls_back_to_default_port_when_unparseable() {
        assert_eq!(
            parse_host(Some("example.com:not-a-port")),
            ("http".to_string(), "example.com".to_string(), 17434)
        );
        assert_eq!(
            parse_host(Some("example.com:99999")),
            ("http".to_string(), "example.com".to_string(), 17434)
        );
    }

    #[test]
    fn parse_host_trims_whitespace_and_surrounding_quotes() {
        assert_eq!(
            parse_host(Some("  \"0.0.0.0:8080\"  ")),
            ("http".to_string(), "0.0.0.0".to_string(), 8080)
        );
    }

    #[test]
    fn parse_host_drops_a_trailing_path() {
        assert_eq!(
            parse_host(Some("example.com:8080/some/path")),
            ("http".to_string(), "example.com".to_string(), 8080)
        );
    }

    #[test]
    fn peer_url_spells_a_peer_the_way_llmman_host_is_spelled() {
        assert_eq!(peer_url("asahi"), "http://asahi:17434");
        assert_eq!(peer_url("spark:8080"), "http://spark:8080");
        assert_eq!(peer_url("https://x.example/"), "https://x.example:443");
        assert_eq!(peer_url("[fe80::1]"), "http://[fe80::1]:17434");
    }

    #[test]
    fn format_host_port_brackets_ipv6_literals() {
        assert_eq!(format_host_port("::1", 17434), "[::1]:17434");
        assert_eq!(format_host_port("127.0.0.1", 17434), "127.0.0.1:17434");
        assert_eq!(format_host_port("example.com", 17434), "example.com:17434");
    }

    #[test]
    fn connectable_host_rewrites_wildcard_hosts_to_loopback() {
        assert_eq!(connectable_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(connectable_host("::"), "::1");
        assert_eq!(connectable_host("0:0:0:0:0:0:0:0"), "::1"); // expanded IPv6 form
        assert_eq!(connectable_host("example.com"), "example.com");
    }

    #[test]
    fn is_local_host_only_accepts_loopback_and_wildcard_hosts() {
        for host in [
            "127.0.0.1",
            "::1",
            "0:0:0:0:0:0:0:1", // expanded IPv6 loopback
            "localhost",
            "0.0.0.0",
            "::",
        ] {
            assert!(is_local_host(host), "{host} should be local");
        }
        assert!(!is_local_host("example.com"));
        assert!(!is_local_host("192.168.1.5"));
    }

    #[test]
    fn log_tail_none_path_is_empty() {
        assert_eq!(log_tail(None), "");
    }

    /// The daemon's marker text rides along when there is one (a release
    /// download's byte counts, an mlx-lm install step); a bare marker
    /// still says what is happening and that it is a first-run one-off.
    #[test]
    fn fetch_message_carries_the_daemon_status_when_present() {
        let bare = fetch_message("");
        assert!(bare.contains("first run"), "got: {bare}");
        assert!(bare.contains("backends"), "got: {bare}");
        assert!(!bare.ends_with(':'), "got: {bare}");
        assert_eq!(fetch_message("  \n"), bare);
        let with = fetch_message("downloading llama.tar.gz: 12 MB / 40 MB (30%)\n");
        assert!(with.starts_with(&bare), "got: {with}");
        assert!(
            with.ends_with(": downloading llama.tar.gz: 12 MB / 40 MB (30%)"),
            "got: {with}"
        );
    }

    /// The relay shows each byte once, creates its own file (clearing
    /// only stale ones of a killed client), and removes it when dropped.
    #[test]
    fn pull_relay_shows_what_the_file_gains() {
        let dir = std::env::temp_dir().join(format!("llmman-pull-relay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let old = dir.join(format!("{PULL_PROGRESS_PREFIX}1"));
        let other = dir.join(format!("{PULL_PROGRESS_PREFIX}2"));
        for p in [&old, &other] {
            std::fs::write(p, b"x").unwrap();
        }
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - PULL_PROGRESS_STALE * 2)
            .unwrap();

        let mut relay = PullRelay::at(&dir, 80).unwrap();
        assert!(!old.exists(), "a dead client's file is swept");
        assert!(other.exists(), "a recent one may be live");
        assert_eq!(relay.cols, 80);

        let mut shown = Vec::new();
        relay.drain_to(&mut shown);
        assert!(shown.is_empty());
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&relay.path)
            .unwrap();
        std::io::Write::write_all(&mut file, b"one\r\n").unwrap();
        relay.drain_to(&mut shown);
        relay.drain_to(&mut shown);
        std::io::Write::write_all(&mut file, b"two\r\n").unwrap();
        relay.drain_to(&mut shown);
        assert_eq!(shown, b"one\r\ntwo\r\n");

        let path = relay.path.clone();
        drop(relay);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pull shows its own output only once it has outlasted
    /// PULL_RELAY_DELAY, stops when the status moves on, and without a
    /// relay (a pipe, not a terminal) never does.
    #[test]
    fn fetch_notice_relays_only_a_lasting_pull() {
        // Never created: nothing is drained from it here.
        let relay = PullRelay {
            path: std::env::temp_dir().join("llmman-no-such-pull-progress"),
            cols: 80,
            offset: 0,
        };
        let pull = crate::container::PULL_STATUS;
        let mut notice = FetchNotice::new(Some(relay));

        notice.update(pull);
        assert!(!notice.relaying, "a fresh pull keeps the spinner");
        notice.pulling_since = Some(std::time::Instant::now() - PULL_RELAY_DELAY);
        notice.update(pull);
        assert!(notice.relaying);
        notice.update("downloading llama.tar.gz: 12 MB / 40 MB (30%)");
        assert!(!notice.relaying);
        assert!(notice.pulling_since.is_none());

        let mut bare = FetchNotice::new(None);
        bare.update(pull);
        bare.pulling_since = Some(std::time::Instant::now() - PULL_RELAY_DELAY);
        bare.update(pull);
        assert!(!bare.relaying);
    }

    #[test]
    fn log_tail_missing_file_points_at_path() {
        let path =
            std::env::temp_dir().join(format!("llmman-log-tail-missing-{}", std::process::id()));
        // A crashed prior run could have left the file behind; make sure
        // the missing-file branch is actually the one under test.
        let _ = std::fs::remove_file(&path);
        let tail = log_tail(Some(&path));
        assert!(tail.contains("see "), "got: {tail}");
        assert!(tail.contains(&path.display().to_string()), "got: {tail}");
    }

    #[test]
    fn log_tail_returns_last_five_nonempty_lines() {
        let path = std::env::temp_dir().join(format!("llmman-log-tail-{}", std::process::id()));
        std::fs::write(&path, "one\ntwo\n\nthree\nfour\nfive\nsix\n").unwrap();
        let tail = log_tail(Some(&path));
        std::fs::remove_file(&path).unwrap();
        assert!(tail.contains("last lines of"), "got: {tail}");
        // Only the part after the header line is the log excerpt; the
        // header contains the temp path, which may itself contain "one".
        let excerpt = tail.split_once('\n').map(|(_, rest)| rest).unwrap_or("");
        assert_eq!(excerpt, "two\nthree\nfour\nfive\nsix", "got: {tail}");
    }

    /// A group signal must never degrade into `kill(0, sig)` or
    /// `kill(-1, sig)`. Signal 0 is a no-op probe, so a regression fails
    /// the assertion rather than the machine.
    #[cfg(unix)]
    #[test]
    fn kill_group_refuses_pids_that_would_signal_everything() {
        assert!(!kill_group(0, 0), "pid 0 = the caller's own process group");
        assert!(!kill_group(1, 0), "pid 1 = every process the user owns");
        // Too large for pid_t, so it can't name a group either.
        assert!(!kill_group(u32::MAX, 0));
    }

    /// An empty group must report false, so stop_group skips its grace
    /// and escalation on a childless fast-fail; kill(1) exits 0 here.
    /// `child` inherits this process's group rather than leading one, and
    /// a live pid is never reused, so group `child.id()` is empty.
    #[cfg(unix)]
    #[test]
    fn kill_group_reports_missing_group_as_nothing_signalled() {
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn child");
        let signalled = kill_group(child.id(), 0);
        let _ = child.kill();
        let _ = child.wait();
        assert!(!signalled, "an empty group reported as signalled");
    }

    /// The signal reaches its own group and nothing else. `control` stays
    /// in the harness's group, so the user-wide `kill(-1, ...)` this used
    /// to issue would take it — and this test binary — down too.
    #[cfg(unix)]
    #[test]
    fn kill_group_kills_only_its_own_group() {
        use std::os::unix::process::CommandExt;

        let mut target = Command::new("sleep");
        target.arg("30").process_group(0); // its own leader: pgid == pid
        let mut target = target.spawn().expect("spawn target");

        let mut control = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn control");

        assert!(
            kill_group(target.id(), libc::SIGKILL),
            "a live group should report as signalled"
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            match target.try_wait().expect("poll target") {
                Some(_) => break,
                None if std::time::Instant::now() >= deadline => {
                    let _ = target.kill();
                    let _ = control.kill();
                    panic!("target process group survived SIGKILL");
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }

        let survived = control.try_wait().expect("poll control").is_none();
        let _ = control.kill();
        let _ = control.wait();
        assert!(
            survived,
            "a process outside the targeted group was signalled too"
        );
    }

    #[test]
    fn only_the_not_found_error_envelope_for_that_model_means_a_missing_model() {
        let m = "docker.io/ai/m";
        assert!(is_model_not_found_body(
            r#"{"error":"model 'docker.io/ai/m' not found"}"#,
            m
        ));
        // A pair is unloaded by its local half, which is what the daemon
        // names.
        assert!(is_model_not_found_body(
            r#"{"error":"model 'gemma4' not found"}"#,
            "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5"
        ));
        assert!(!is_model_not_found_body(
            r#"{"error":"model 'other' not found"}"#,
            m
        ));
        assert!(!is_model_not_found_body(r#"{"error":"no route"}"#, m));
        assert!(!is_model_not_found_body("<html>404 Not Found</html>", m));
        assert!(!is_model_not_found_body("", m));
        assert!(!is_model_not_found_body(
            r#"{"message":"model 'docker.io/ai/m' not found"}"#,
            m
        ));
    }

    /// The window is found under the caller's own spelling, not only
    /// the one the daemon loaded the model under. The untagged form
    /// `resolve_ollama_api` hands `llmman launch` is the common case.
    #[test]
    fn a_loaded_window_is_found_however_its_reference_is_tagged() {
        let loaded = || {
            vec![
                PsWindow::new("docker.io/ai/other:latest", Some(4096)),
                PsWindow::new("docker.io/ai/gemma4:latest", Some(32_768)),
            ]
        };
        assert_eq!(ps_window(loaded(), "docker.io/ai/gemma4"), Some(32_768));
        assert_eq!(
            ps_window(loaded(), "docker.io/ai/gemma4:latest"),
            Some(32_768)
        );
        // A pair is read back on its local half, the only loaded one.
        assert_eq!(
            ps_window(
                loaded(),
                "llmman.hybrid/docker.io/ai/gemma4,anthropic/claude-sonnet-4-5"
            ),
            Some(32_768)
        );
        // Another tag is another model, and a backend that reports no
        // window of its own reports none.
        assert_eq!(ps_window(loaded(), "docker.io/ai/gemma4:8b"), None);
        assert_eq!(
            ps_window(
                vec![PsWindow::new("docker.io/ai/gemma4:latest", None)],
                "docker.io/ai/gemma4"
            ),
            None
        );
    }

    /// A fixed-width terminal that records what indicatif writes to it.
    #[derive(Debug)]
    struct FixedWidthTerm {
        width: u16,
        written: std::sync::Arc<std::sync::Mutex<String>>,
    }

    impl indicatif::TermLike for FixedWidthTerm {
        fn width(&self) -> u16 {
            self.width
        }
        fn move_cursor_up(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn write_line(&self, s: &str) -> std::io::Result<()> {
            self.write_str(&format!("{s}\n"))
        }
        fn write_str(&self, s: &str) -> std::io::Result<()> {
            self.written.lock().unwrap().push_str(s);
            Ok(())
        }
        fn clear_line(&self) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The pull bar must fill, and never exceed, the terminal width.
    #[test]
    fn pull_bar_fits_the_terminal_width() {
        const MIB: u64 = 1024 * 1024;
        for reference in [
            "qwen3.5:0.8b",
            "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M",
        ] {
            for width in [60u16, 80, 100, 120, 200] {
                let written = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
                let term = FixedWidthTerm {
                    width,
                    written: written.clone(),
                };
                let pb = new_pull_bar(
                    703 * MIB,
                    reference,
                    ProgressDrawTarget::term_like(Box::new(term)),
                );
                written.lock().unwrap().clear();
                pb.set_position(16 * MIB);

                let frame = written.lock().unwrap().clone();
                let frame = console::strip_ansi_codes(&frame);
                let line = frame.trim_end();
                assert!(!line.contains('\n'), "{reference} at {width}: {line:?}");
                assert_eq!(
                    console::measure_text_width(line),
                    usize::from(width),
                    "{reference} at {width}: {line:?}"
                );
            }
        }
    }

    fn identity_from(body: &str) -> anyhow::Result<DaemonIdentity> {
        Ok(serde_json::from_str(body)?)
    }

    /// Ollama answers a bare version and a 404 for llmman's own routes; a
    /// daemon from before identity existed answers only a version too, and
    /// is told from it by llmman's own route answering.
    #[test]
    fn a_server_is_llmman_by_its_identity_or_by_answering_llmman_routes() {
        let current = identity_from(r#"{"version":"0.1.0","exe":"/bin/llmman","pid":42}"#).unwrap();
        let never_probed = || panic!("a daemon that identifies itself needs no probe");
        assert!(is_llmman_daemon(&current, never_probed));
        // An exe without a pid is still llmman's.
        let no_pid = identity_from(r#"{"version":"0.1.350","exe":"/bin/llmman"}"#).unwrap();
        assert!(is_llmman_daemon(&no_pid, never_probed));
        for body in [
            r#"{"version":"0.40.17"}"#,
            r#"{"version":""}"#,
            r#"{}"#,
            r#"{"pid":7}"#,
        ] {
            let bare = identity_from(body).unwrap();
            assert!(is_llmman_daemon(&bare, || true), "{body}");
            assert!(!is_llmman_daemon(&bare, || false), "{body}");
        }
        assert!(identity_from("not json").is_err());
        assert!(identity_from("404 page not found").is_err());
    }

    #[test]
    fn only_a_providers_list_marks_a_llmman_daemon() {
        let json = |body: &str| serde_json::from_str::<serde_json::Value>(body).unwrap();
        assert!(is_providers_list(&json(r#"{"providers":[]}"#)));
        assert!(is_providers_list(&json(
            r#"{"providers":[{"id":"openai"}]}"#
        )));
        assert!(!is_providers_list(&json(r#"{"providers":"x"}"#)));
        assert!(!is_providers_list(&json(r#"{"error":"not found"}"#)));
        assert!(!is_providers_list(&json(r#"[]"#)));
        assert!(!is_providers_list(&json(r#""404 page not found""#)));
    }

    /// The stream of lines through [`PullProgress`]: stdout, stderr, the bar
    /// (position, length) after each line, and the outcome.
    struct Pulled {
        out: String,
        err: String,
        bars: Vec<Option<(u64, u64)>>,
        result: anyhow::Result<Option<String>>,
    }

    fn pull_stream(reference: &str, lines: &[&str]) -> Pulled {
        let mut progress = PullProgress::new(reference, ProgressDrawTarget::hidden);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut bars = Vec::new();
        for line in lines {
            let step = progress.line(line, &mut out, &mut err);
            bars.push(
                progress
                    .bar
                    .as_ref()
                    .map(|bar| (bar.position(), bar.length().unwrap())),
            );
            if let Err(error) = step {
                return Pulled {
                    out: String::from_utf8(out).unwrap(),
                    err: String::from_utf8(err).unwrap(),
                    bars,
                    result: Err(error),
                };
            }
        }
        let result = progress.finish(&mut out);
        Pulled {
            out: String::from_utf8(out).unwrap(),
            err: String::from_utf8(err).unwrap(),
            bars,
            result,
        }
    }

    /// llmman's own daemon: one aggregate, no digest on a progress line.
    #[test]
    fn a_llmman_daemon_stream_draws_one_bar_from_its_aggregate() {
        let pulled = pull_stream(
            "qwen3:0.6b",
            &[
                r#"{"status":"pulling manifest"}"#,
                r#"{"status":"pulling","total":1000,"completed":0}"#,
                r#"{"status":"pulling","total":1000,"completed":400}"#,
                "",
                "keepalive, not json",
                r#"{"status":"pulling","total":2000,"completed":2000}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.out, "pulling manifest\nsuccess\n");
        assert_eq!(pulled.err, "");
        assert_eq!(
            pulled.bars,
            [
                None,
                Some((0, 1000)),
                Some((400, 1000)),
                Some((400, 1000)),
                Some((400, 1000)),
                Some((2000, 2000)),
                None,
            ]
        );
        assert_eq!(pulled.result.unwrap(), None);
    }

    #[test]
    fn a_llmman_daemon_stream_already_complete_prints_one_line_and_no_bar() {
        for (total, completed) in [(5_000_000, 5_000_000), (1_000_000, 990_000)] {
            let progress =
                format!(r#"{{"status":"pulling","total":{total},"completed":{completed}}}"#);
            let pulled = pull_stream(
                "m:1",
                &[
                    r#"{"status":"pulling manifest"}"#,
                    &progress,
                    &progress,
                    r#"{"status":"pulling"}"#,
                    r#"{"status":"success"}"#,
                ],
            );
            assert_eq!(
                pulled.out,
                format!(
                    "pulling manifest\nAlready have m:1 ({})\nsuccess\n",
                    crate::fmt::human_size(total)
                ),
                "{total}/{completed}"
            );
            assert!(pulled.bars.iter().all(Option::is_none));
            pulled.result.unwrap();
        }
        // Under 99% is a real transfer.
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling","total":1000000,"completed":980000}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.bars[0], Some((980_000, 1_000_000)));
        assert_eq!(pulled.out, "success\n");
    }

    #[test]
    fn a_llmman_daemon_stream_ending_complete_after_a_bar_keeps_updating_it() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling","total":100,"completed":10}"#,
                r#"{"status":"pulling","total":100,"completed":100}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.bars[1], Some((100, 100)));
        assert_eq!(pulled.out, "success\n");
    }

    #[test]
    fn a_push_digest_without_a_total_is_the_result_not_progress() {
        let pulled = pull_stream(
            "reg.example/m:1",
            &[
                r#"{"status":"pushing manifest"}"#,
                r#"{"status":"pushing","total":500,"completed":100}"#,
                r#"{"digest":"sha256:abc"}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.out, "pushing manifest\nsuccess\n");
        assert_eq!(pulled.bars[2], Some((100, 500)));
        assert_eq!(pulled.result.unwrap().as_deref(), Some("sha256:abc"));
        // A zero total is no progress either.
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"digest":"sha256:abc","total":0,"completed":0}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert!(pulled.bars.iter().all(Option::is_none));
        assert_eq!(pulled.result.unwrap().as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn notices_and_errors_are_handled_as_before() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"notice":"unsigned"}"#,
                r#"{"status":"pulling","total":100,"completed":1}"#,
                r#"{"notice":"verified"}"#,
                r#"{"error":"disk full"}"#,
            ],
        );
        assert_eq!(pulled.err, "[llmman] unsigned\n[llmman] verified\n");
        assert_eq!(pulled.result.unwrap_err().to_string(), "m:1: disk full");
        // An error that already names the reference is not prefixed.
        let pulled = pull_stream("m:1", &[r#"{"error":"pull m:1: not found"}"#]);
        assert_eq!(
            pulled.result.unwrap_err().to_string(),
            "pull m:1: not found"
        );
        // No success status.
        let pulled = pull_stream("m:1", &[r#"{"status":"pulling manifest"}"#]);
        assert_eq!(
            pulled.result.unwrap_err().to_string(),
            "m:1: stream ended without a success status"
        );
        let pulled = pull_stream("m:1", &[]);
        assert!(pulled.result.is_err());
    }

    /// Ollama's layer-by-layer stream, with concurrent layers.
    #[test]
    fn an_ollama_stream_draws_one_bar_summing_its_layers() {
        let pulled = pull_stream(
            "all-minilm:22m",
            &[
                r#"{"status":"pulling manifest"}"#,
                r#"{"status":"pulling aaaa","digest":"sha256:aaaa","total":100,"completed":0}"#,
                r#"{"status":"pulling bbbb","digest":"sha256:bbbb","total":50,"completed":0}"#,
                r#"{"status":"pulling aaaa","digest":"sha256:aaaa","total":100,"completed":60}"#,
                r#"{"status":"pulling bbbb","digest":"sha256:bbbb","total":50,"completed":50}"#,
                r#"{"status":"pulling aaaa","digest":"sha256:aaaa","total":100,"completed":100}"#,
                r#"{"status":"verifying sha256 digest"}"#,
                r#"{"status":"writing manifest"}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(
            pulled.out,
            "pulling manifest\nverifying sha256 digest\nwriting manifest\nsuccess\n"
        );
        assert_eq!(
            pulled.bars,
            [
                None,
                Some((0, 100)),
                Some((0, 150)),
                Some((60, 150)),
                Some((110, 150)),
                Some((150, 150)),
                None,
                None,
                None,
            ]
        );
        assert_eq!(pulled.result.unwrap(), None);
    }

    #[test]
    fn an_ollama_layer_cannot_complete_past_its_total() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":40}"#,
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":900}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.bars[1], Some((100, 100)));
    }

    /// A pull of layers already stored is one "Already have" line, and
    /// only once all layers are in: the first one alone is complete in
    /// every pull that has a stored layer.
    #[test]
    fn an_ollama_pull_of_stored_layers_prints_already_have_once_after_the_layers() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling manifest"}"#,
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":100}"#,
                r#"{"status":"pulling b","digest":"sha256:b","total":50,"completed":50}"#,
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":100}"#,
                r#"{"status":"verifying sha256 digest"}"#,
                r#"{"status":"writing manifest"}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(
            pulled.out,
            "pulling manifest\nAlready have m:1 (150 B)\n\
             verifying sha256 digest\nwriting manifest\nsuccess\n"
        );
        assert!(pulled.bars.iter().all(Option::is_none));
    }

    #[test]
    fn an_ollama_pull_with_a_stored_first_layer_still_draws_the_bar() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":100}"#,
                r#"{"status":"pulling b","digest":"sha256:b","total":5000,"completed":0}"#,
                r#"{"status":"pulling b","digest":"sha256:b","total":5000,"completed":2500}"#,
                r#"{"status":"verifying sha256 digest"}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert_eq!(pulled.bars[0], None);
        assert_eq!(pulled.bars[1], Some((100, 5100)));
        assert_eq!(pulled.bars[2], Some((2600, 5100)));
        assert!(!pulled.out.contains("Already have"), "{}", pulled.out);
    }

    #[test]
    fn a_late_ollama_layer_line_after_the_bar_closed_does_not_announce() {
        let pulled = pull_stream(
            "m:1",
            &[
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":10}"#,
                r#"{"status":"verifying sha256 digest"}"#,
                r#"{"status":"pulling a","digest":"sha256:a","total":100,"completed":100}"#,
                r#"{"status":"success"}"#,
            ],
        );
        assert!(!pulled.out.contains("Already have"), "{}", pulled.out);
        pulled.result.unwrap();
    }

    #[test]
    fn an_ollama_error_line_fails_the_pull_and_a_failed_pull_has_no_success() {
        let pulled = pull_stream(
            "nope:1",
            &[
                r#"{"status":"pulling manifest"}"#,
                r#"{"error":"pull model manifest: file does not exist"}"#,
            ],
        );
        assert_eq!(
            pulled.result.unwrap_err().to_string(),
            "nope:1: pull model manifest: file does not exist"
        );
    }
}
