//! `/llmman/shell`: the user's login shell in a pty, over a WebSocket, for
//! the web UI's Shell.
//!
//! A real shell as the daemon's user, so who may open one is the design:
//! only when the daemon is bound to loopback
//! (`daemon::reachable_only_locally`); only with the daemon's API key
//! when it has one (the `auth` module, which checks this route like every
//! other — a browser presents the key as a subprotocol); only with the
//! token this start made, as the subprotocol `llmman.shell.<token>`, API
//! keys or not; only from an `Origin` the CORS layer would allow, minus
//! host wildcards, checked here because browsers do not apply CORS to
//! WebSockets; and not at all under `LLMMAN_SHELL=off`. Any other
//! `LLMMAN_SHELL` value is the command to run instead of the login shell.
//!
//! The token is written 0600 to `shell-<host>-<port>.token` in llmman's data
//! directory, and printed as a web UI link (`#shell-token=<token>`) when
//! stderr is a terminal. The UI reads it from the link's fragment, which
//! the browser never sends to the daemon. A token that cannot be saved,
//! with no terminal to print the link to, turns the shell off.
//!
//! Protocol: binary frames carry terminal bytes both ways; the client's
//! text frames are `{"resize":{"cols":N,"rows":N}}`; the daemon's one text
//! frame is `{"exit":CODE}` before it closes. A plain `GET` returns
//! `{"enabled":bool,"reason":...,"token_required":true}`.

use std::io::{IsTerminal as _, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::Context as _;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Duration;

use super::AppState;

/// The subprotocol prefix the shell token comes under. Not the API key's:
/// a browser on a keyed daemon offers both, and each check finds its own.
pub(super) const TOKEN_PROTOCOL_PREFIX: &str = "llmman.shell.";

/// URL-safe base64, which a subprotocol and a URL fragment carry as is.
pub(super) fn new_token() -> anyhow::Result<String> {
    use base64::Engine as _;
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| anyhow::anyhow!("shell token: {e}"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// Who may open a shell, fixed at startup.
#[derive(Debug, Clone)]
pub(super) struct Policy {
    /// `None` when allowed, else the reason the status reply carries.
    pub(super) disabled: Option<String>,
    /// `Origin` patterns accepted (see `super::origin_matches`).
    pub(super) origins: Vec<String>,
    /// Program and arguments; empty means the login shell (`%ComSpec%`
    /// on Windows).
    pub(super) command: Vec<String>,
    pub(super) token_digest: [u8; 32],
    /// Set at startup when the token could not be saved and no terminal
    /// shows the link: no client can learn it, so the shell is off.
    pub(super) token_unsaved: OnceLock<String>,
}

impl Policy {
    pub(super) fn from_env(token: &str) -> anyhow::Result<Self> {
        let setting = ShellSetting::parse(std::env::var("LLMMAN_SHELL").ok().as_deref())
            .map_err(|e| anyhow::anyhow!("LLMMAN_SHELL: {e}"))?;
        let disabled = if setting == ShellSetting::Off {
            Some("LLMMAN_SHELL is off".to_string())
        } else if !crate::daemon::reachable_only_locally() {
            Some(
                "llmman serve is not bound to loopback (LLMMAN_HOST), so the shell is off: \
                 anyone who can reach the daemon could reach the shell"
                    .to_string(),
            )
        } else {
            None
        };
        // Plus the bind host itself: `LLMMAN_HOST=127.0.0.2` is loopback
        // but not in the default localhost list.
        let mut origins = super::allowed_origins_from_env();
        let bind = crate::daemon::bind_addr();
        let host = bind.rsplit_once(':').map_or(bind.as_str(), |(h, _)| h);
        origins.extend([format!("http://{host}:*"), format!("https://{host}:*")]);
        Ok(Self {
            disabled,
            origins,
            command: match setting {
                ShellSetting::Command(argv) => argv,
                _ => Vec::new(),
            },
            token_digest: super::auth::digest(token),
            token_unsaved: OnceLock::new(),
        })
    }

    fn presented_token(&self, headers: &HeaderMap) -> Option<String> {
        super::auth::offered_ws_protocol_with(headers, TOKEN_PROTOCOL_PREFIX).filter(|protocol| {
            let token = &protocol[TOKEN_PROTOCOL_PREFIX.len()..];
            super::auth::digests_match(&super::auth::digest(token), &self.token_digest)
        })
    }

    /// Whether a page at `origin` (or a client with no page) may open one.
    fn admits(&self, origin: Option<&str>) -> Result<(), Refusal> {
        if let Some(reason) = self.disabled.as_ref().or(self.token_unsaved.get()) {
            return Err(Refusal::Disabled(reason.clone()));
        }
        let allowed = |origin: &str| {
            self.origins
                .iter()
                .filter(|p| !wildcard_host(p))
                .any(|p| super::origin_matches(origin, p))
        };
        match origin {
            None => Ok(()),
            Some(origin) if allowed(origin) => Ok(()),
            Some(origin) => Err(Refusal::Origin(origin.to_string())),
        }
    }
}

/// Whether a pattern's `*` can stand in for the host rather than just the
/// port (`http://localhost:*`). Fine for CORS, not for a shell:
/// `LLMMAN_ORIGINS=*` would otherwise open it to every site.
fn wildcard_host(pattern: &str) -> bool {
    match pattern.split_once('*') {
        Some((prefix, suffix)) => !(suffix.is_empty() && prefix.ends_with(':')),
        None => false,
    }
}

/// What `LLMMAN_SHELL` asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ShellSetting {
    /// Unset or empty: the login shell.
    Default,
    /// `0`/`false`/`no`/`off`.
    Off,
    /// Anything else, split by [`super::backend::split_quoted_args`].
    Command(Vec<String>),
}

impl ShellSetting {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        let Some(raw) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Self::Default);
        };
        Ok(match raw.to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" => Self::Off,
            "1" | "true" | "yes" | "on" => Self::Default,
            // Untrimmed, so an error's character position is the variable's.
            _ => Self::Command(super::backend::split_quoted_args(value)?),
        })
    }
}

enum Refusal {
    Disabled(String),
    Origin(String),
}

impl Refusal {
    fn reason(&self) -> String {
        match self {
            Refusal::Disabled(why) => why.clone(),
            Refusal::Origin(origin) => format!("origin {origin} is not an allowed origin"),
        }
    }
}

/// Where this start's token is kept: one file per bind address, so
/// daemons on two addresses do not overwrite each other's.
pub(super) fn token_path() -> anyhow::Result<PathBuf> {
    Ok(crate::data_root()?.join(token_file_name(&crate::daemon::bind_addr())))
}

/// `shell-<host>-<port>.token`. An IPv6 host loses its brackets and its
/// colons become `_`: Windows forbids `:` in file names.
fn token_file_name(bind_addr: &str) -> String {
    let (host, port) = bind_addr.rsplit_once(':').unwrap_or((bind_addr, ""));
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .replace(':', "_");
    format!("shell-{host}-{port}.token")
}

/// Logs whether the shell is on, and saves the token when it is. The
/// link carrying the token is printed only to a terminal: a detached
/// daemon's stderr is `serve.log`, which other users may be able to read.
pub(super) fn announce(policy: &Policy, token: &str, tls: bool) {
    if let Some(reason) = &policy.disabled {
        remove_old_token();
        eprintln!("[llmman] shell: off ({reason})");
        return;
    }
    let saved = token_path().and_then(|path| {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        crate::fsutil::write_private(&path, token.as_bytes())
            .with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    });
    let why = "bound to loopback; LLMMAN_SHELL=off removes it";
    let terminal = std::io::stderr().is_terminal();
    match saved {
        Ok(path) => eprintln!(
            "[llmman] shell: on ({why}), its token is in {}",
            path.display()
        ),
        Err(e) => {
            remove_old_token();
            if !terminal {
                let reason = format!("llmman serve could not save its shell token: {e:#}");
                eprintln!("[llmman] shell: off ({reason})");
                let _ = policy.token_unsaved.set(reason);
                return;
            }
            eprintln!("[llmman] shell: on ({why}), but its token was not saved: {e:#}");
        }
    }
    if terminal {
        eprintln!(
            "[llmman] open the web UI with the shell: http{}://{}/#shell-token={token}",
            if tls { "s" } else { "" },
            crate::daemon::bind_addr()
        );
    }
}

/// A token from an earlier start opens nothing: a client that reads it
/// only gets a 401.
fn remove_old_token() {
    if let Ok(path) = token_path() {
        let _ = std::fs::remove_file(path);
    }
}

fn token_refused() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        "the shell needs the token llmman serve made when it started: open the web UI link \
         it printed, or offer the token from its shell-<host>-<port>.token file as the subprotocol \
         llmman.shell.<token>",
    )
        .into_response()
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Status {
    pub(super) enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) reason: Option<String>,
    /// Always true here. A daemon older than the token omits it, and the
    /// UI then opens a shell without one.
    #[serde(default)]
    pub(super) token_required: bool,
}

/// `GET /llmman/shell`: an upgrade opens a shell, a plain GET reports
/// whether one would.
pub(super) async fn handle_shell(
    State(state): State<AppState>,
    headers: HeaderMap,
    ws: Option<WebSocketUpgrade>,
) -> Response {
    let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    let refusal = state.0.shell.admits(origin).err().map(|r| r.reason());
    match (ws, refusal) {
        (None, reason) => Json(Status {
            enabled: reason.is_none(),
            reason,
            token_required: true,
        })
        .into_response(),
        (Some(_), Some(reason)) => (StatusCode::FORBIDDEN, reason).into_response(),
        (Some(ws), None) => {
            let Some(protocol) = state.0.shell.presented_token(&headers) else {
                return token_refused();
            };
            let command = state.0.shell.command.clone();
            // The handshake fails unless one offered subprotocol is echoed.
            ws.protocols([protocol]).on_upgrade(|socket| async move {
                if let Err(e) = run_session(socket, &command).await {
                    eprintln!("[llmman] shell session ended with an error: {e:#}");
                }
            })
        }
    }
}

#[derive(Deserialize)]
struct Control {
    resize: Option<Resize>,
}

#[derive(Deserialize)]
struct Resize {
    cols: u16,
    rows: u16,
}

/// One shell per socket: closing either ends the other.
async fn run_session(mut socket: WebSocket, command: &[String]) -> anyhow::Result<()> {
    let pty = native_pty_system();
    let pair = pty.openpty(PtySize::default())?;
    let mut cmd = match command.split_first() {
        None => CommandBuilder::new_default_prog(),
        Some((program, args)) => {
            let mut cmd = CommandBuilder::new(program);
            cmd.args(args);
            cmd
        }
    };
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    let mut child = pair.slave.spawn_command(cmd)?;
    // Held open, the slave would keep the reader from seeing EOF.
    drop(pair.slave);
    let mut killer = child.clone_killer();
    let mut reader = pair.master.try_clone_reader()?;
    let mut writer = pair.master.take_writer()?;
    let master = pair.master;

    // Pty I/O is blocking: a thread per direction, channels to the async side.
    let (out_tx, mut out_rx) = mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if out_tx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let (in_tx, mut in_rx) = mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        while let Some(bytes) = in_rx.blocking_recv() {
            if writer
                .write_all(&bytes)
                .and_then(|()| writer.flush())
                .is_err()
            {
                break;
            }
        }
    });
    let (exit_tx, mut exit_rx) = oneshot::channel::<u32>();
    std::thread::spawn(move || {
        let code = child.wait().map(|s| s.exit_code()).unwrap_or(1);
        let _ = exit_tx.send(code);
    });

    let outcome = loop {
        tokio::select! {
            frame = socket.recv() => match frame {
                Some(Ok(Message::Binary(bytes))) => {
                    if in_tx.send(bytes.to_vec()).await.is_err() {
                        break Ok(());
                    }
                }
                Some(Ok(Message::Text(text))) => {
                    if let Some(Resize { cols, rows }) =
                        serde_json::from_str::<Control>(&text).ok().and_then(|c| c.resize)
                    {
                        if cols > 0 && rows > 0 {
                            let _ = master.resize(PtySize { rows, cols, ..PtySize::default() });
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | None => break Ok(()),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Err(e)) => break Err(anyhow::Error::from(e)),
            },
            chunk = out_rx.recv() => match chunk {
                Some(bytes) => {
                    if socket.send(Message::Binary(bytes)).await.is_err() {
                        break Ok(());
                    }
                }
                None => {
                    // The pty closed: the shell exited, or closed its
                    // terminal and lives on. Either way it is done here.
                    let _ = killer.kill();
                    let code = tokio::time::timeout(Duration::from_secs(2), &mut exit_rx)
                        .await
                        .ok()
                        .and_then(Result::ok)
                        .unwrap_or(1);
                    finish(&mut socket, code).await;
                    break Ok(());
                }
            },
            // On Windows the ConPTY reader only ends when the master
            // closes, so take the exit from `wait()` too, after a moment
            // for output still in flight.
            code = &mut exit_rx => {
                let drain = tokio::time::sleep(Duration::from_millis(300));
                tokio::pin!(drain);
                loop {
                    tokio::select! {
                        _ = &mut drain => break,
                        chunk = out_rx.recv() => match chunk {
                            Some(bytes) => {
                                if socket.send(Message::Binary(bytes)).await.is_err() {
                                    break;
                                }
                            }
                            None => break,
                        },
                    }
                }
                finish(&mut socket, code.unwrap_or(1)).await;
                break Ok(());
            }
        }
    };

    // No shell outlives its socket; killing an exited child is harmless.
    let _ = killer.kill();
    outcome
}

/// The closing text frame and close handshake, best effort.
async fn finish(socket: &mut WebSocket, code: u32) {
    let _ = socket
        .send(Message::Text(
            serde_json::json!({ "exit": code }).to_string(),
        ))
        .await;
    let _ = socket.send(Message::Close(None)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(disabled: Option<&str>) -> Policy {
        Policy {
            disabled: disabled.map(str::to_string),
            origins: super::super::default_allowed_origins(),
            command: Vec::new(),
            token_digest: super::super::auth::digest("t"),
            token_unsaved: OnceLock::new(),
        }
    }

    fn offering(protocols: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("sec-websocket-protocol", protocols.parse().unwrap());
        headers
    }

    #[test]
    fn each_bind_address_has_its_own_token_file() {
        for (bind, name) in [
            ("127.0.0.1:17434", "shell-127.0.0.1-17434.token"),
            ("127.0.0.2:17434", "shell-127.0.0.2-17434.token"),
            ("localhost:17434", "shell-localhost-17434.token"),
            ("[::1]:17434", "shell-__1-17434.token"),
        ] {
            assert_eq!(token_file_name(bind), name, "{bind}");
        }
    }

    #[test]
    fn a_new_token_is_random_and_fits_a_subprotocol_as_it_is() {
        let (a, b) = (new_token().unwrap(), new_token().unwrap());
        assert_ne!(a, b);
        assert_eq!(a.len(), 43, "32 bytes of unpadded base64");
        assert!(a
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'));
    }

    #[test]
    fn only_this_starts_token_is_presented() {
        let p = policy(None);
        assert_eq!(
            p.presented_token(&offering("llmman.bearer.a2V5, llmman.shell.t")),
            Some("llmman.shell.t".into())
        );
        assert_eq!(p.presented_token(&offering("llmman.shell.other")), None);
        assert_eq!(p.presented_token(&offering("llmman.shell.")), None);
        assert_eq!(p.presented_token(&offering("t, llmman.bearer.dA")), None);
        assert_eq!(p.presented_token(&HeaderMap::new()), None);
    }

    #[test]
    fn a_localhost_page_or_no_page_at_all_is_admitted() {
        let p = policy(None);
        assert!(p.admits(None).is_ok());
        assert!(p.admits(Some("http://localhost:17434")).is_ok());
        assert!(p.admits(Some("http://127.0.0.1")).is_ok());
    }

    #[test]
    fn a_host_wildcard_admits_nothing_even_though_cors_would() {
        let mut p = policy(None);
        p.origins = vec![
            "*".into(),
            "https://*.example.com".into(),
            "http://host:*".into(),
        ];
        assert!(p.admits(Some("https://evil.example")).is_err());
        assert!(p.admits(Some("https://a.example.com")).is_err());
        assert!(
            p.admits(Some("http://host:8080")).is_ok(),
            "a port wildcard is fine"
        );
        assert!(wildcard_host("*") && wildcard_host("https://*.example.com"));
        assert!(!wildcard_host("http://host:*") && !wildcard_host("http://host"));
    }

    #[test]
    fn another_site_is_refused_by_origin() {
        let p = policy(None);
        let refusal = p.admits(Some("https://evil.example")).err().unwrap();
        assert!(matches!(refusal, Refusal::Origin(_)));
        assert!(refusal.reason().contains("evil.example"));
    }

    #[test]
    fn a_disabled_policy_refuses_everyone_with_its_reason() {
        let p = policy(Some("LLMMAN_SHELL is off"));
        for origin in [None, Some("http://localhost")] {
            let refusal = p.admits(origin).err().unwrap();
            assert!(matches!(refusal, Refusal::Disabled(_)));
            assert_eq!(refusal.reason(), "LLMMAN_SHELL is off");
        }
    }

    #[test]
    fn an_unsaved_token_turns_the_shell_off_with_its_reason() {
        let p = policy(None);
        p.token_unsaved.set("could not save".into()).unwrap();
        for origin in [None, Some("http://localhost")] {
            let refusal = p.admits(origin).err().unwrap();
            assert!(matches!(refusal, Refusal::Disabled(_)));
            assert_eq!(refusal.reason(), "could not save");
        }
    }

    #[test]
    fn llmman_shell_env_spellings() {
        for v in ["0", "false", "NO", " off "] {
            assert_eq!(ShellSetting::parse(Some(v)), Ok(ShellSetting::Off), "{v:?}");
        }
        for v in [
            None,
            Some(""),
            Some("   "),
            Some("1"),
            Some("true"),
            Some("ON"),
        ] {
            assert_eq!(ShellSetting::parse(v), Ok(ShellSetting::Default), "{v:?}");
        }
        let command = |argv: &[&str]| {
            Ok(ShellSetting::Command(
                argv.iter().map(|a| a.to_string()).collect(),
            ))
        };
        assert_eq!(
            ShellSetting::parse(Some("  tmux new -A  -s llmman ")),
            command(&["tmux", "new", "-A", "-s", "llmman"])
        );
        assert_eq!(
            ShellSetting::parse(Some("sh -c 'exec tmux new -A'")),
            command(&["sh", "-c", "exec tmux new -A"])
        );
        assert_eq!(
            ShellSetting::parse(Some(r#""C:\Program Files\Git\bin\bash.exe" -l"#)),
            command(&[r"C:\Program Files\Git\bin\bash.exe", "-l"])
        );
        assert_eq!(
            ShellSetting::parse(Some("  sh -c 'exec tmux")),
            Err("missing closing quote for the ' at character 9".into())
        );
    }
}
