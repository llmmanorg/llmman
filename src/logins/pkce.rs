//! Authorization-code + PKCE with a loopback callback: the browser half.
//!
//! The browser is sent to the provider with a code challenge and a
//! callback on this computer; the provider redirects back with a one-time
//! `code`, which the caller exchanges (with the verifier) for a
//! credential. Only the listening and the parsing are here — what is
//! exchanged and where is per provider ([`super`]).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use base64::Engine as _;
use sha2::{Digest, Sha256};

/// A PKCE pair (RFC 7636, method `S256`).
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// `n` random bytes, base64url.
pub fn random_token(n: usize) -> anyhow::Result<String> {
    let mut bytes = vec![0u8; n];
    getrandom::getrandom(&mut bytes).map_err(|e| anyhow!("no randomness: {e}"))?;
    Ok(b64(&bytes))
}

impl Pkce {
    pub fn generate() -> anyhow::Result<Self> {
        let verifier = random_token(32)?;
        Ok(Self::from_verifier(verifier))
    }

    pub fn from_verifier(verifier: String) -> Self {
        let challenge = b64(&Sha256::digest(verifier.as_bytes()));
        Self {
            verifier,
            challenge,
        }
    }
}

/// The `code` in a callback, from whatever the user pastes when the
/// browser could not reach the listener: the full redirect URL, or just
/// its query. The `state` must be the one this attempt made.
pub fn code_from_callback(input: &str, state: &str) -> anyhow::Result<String> {
    let input = input.trim();
    let query = match input.split_once('?') {
        Some((_, q)) => q,
        None if input.contains('=') => input,
        None => return Err(anyhow!("that is not a redirect URL")),
    };
    let query = query.split('#').next().unwrap_or("");
    let url = reqwest::Url::parse(&format!("http://localhost/?{query}"))
        .map_err(|_| anyhow!("that is not a redirect URL"))?;
    let param = |name: &str| {
        url.query_pairs()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.into_owned())
    };
    if let Some(error) = param("error") {
        anyhow::bail!("the provider refused the login: {}", super::clean(&error));
    }
    match param("state") {
        Some(got) if constant_time_eq(got.as_bytes(), state.as_bytes()) => {}
        Some(_) => anyhow::bail!("the login's state does not match this attempt; start again"),
        None => anyhow::bail!("the redirect has no state; paste the whole URL"),
    }
    param("code")
        .filter(|c| !c.is_empty())
        .ok_or_else(|| anyhow!("the redirect has no code"))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Waits on `listener` for the browser to arrive at `path`, answers it,
/// and returns the `code` (see [`code_from_callback`]).
///
/// Only requests addressed to this machine are served — a `Host` of
/// `localhost` or a loopback address — so a web page cannot use a name it
/// controls to reach the listener through a rebinding.
pub fn wait_for_callback(
    listener: &TcpListener,
    path: &str,
    state: &str,
    timeout: Duration,
) -> anyhow::Result<String> {
    listener
        .set_nonblocking(true)
        .context("listen for the callback")?;
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Some(result) = serve_one(stream, path, state) {
                    return result;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    anyhow::bail!("timed out waiting for the browser to come back");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e).context("accept the callback"),
        }
    }
}

const DONE_PAGE: &str = "<!doctype html><title>llmman</title><p>Signed in. You can close this tab and return to the terminal.</p>";
const FAILED_PAGE: &str = "<!doctype html><title>llmman</title><p>The sign-in did not complete. Return to the terminal.</p>";

/// `Some(result)` when this request settles the login, `None` when it was
/// something else (a favicon, a probe).
fn serve_one(mut stream: TcpStream, path: &str, state: &str) -> Option<anyhow::Result<String>> {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while head.len() < 8192 && !head.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
        }
    }
    let head = String::from_utf8_lossy(&head);
    let mut lines = head.lines();
    let request = lines.next().unwrap_or("");
    let host = lines
        .find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case("host"))
                .map(|(_, v)| v.trim().to_ascii_lowercase())
        })
        .unwrap_or_default();
    let host_name = host.rsplit_once(':').map_or(host.as_str(), |(h, _)| h);
    let target = request
        .strip_prefix("GET ")
        .and_then(|r| r.split(' ').next())
        .unwrap_or("");
    let on_path = target == path || target.starts_with(&format!("{path}?"));
    let local = matches!(host_name, "localhost" | "127.0.0.1" | "[::1]");
    let reply = |stream: &mut TcpStream, status: &str, body: &str| {
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n\
             Cache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    };
    if !on_path || !local {
        reply(&mut stream, "404 Not Found", "not found");
        return None;
    }
    let result = code_from_callback(target, state);
    match &result {
        Ok(_) => reply(&mut stream, "200 OK", DONE_PAGE),
        Err(_) => reply(&mut stream, "400 Bad Request", FAILED_PAGE),
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// The implicit grant: a token in the URL's fragment
// ---------------------------------------------------------------------------

/// What an implicit-grant redirect carried.
#[derive(Debug, PartialEq, Eq)]
pub struct FragmentToken {
    pub access_token: String,
    /// Seconds, as the provider said it; 0 when it did not.
    pub expires_in: u64,
}

fn token_from_parts(
    access: Option<String>,
    expires_in: Option<String>,
    got_state: Option<String>,
    error: Option<String>,
    state: &str,
) -> anyhow::Result<FragmentToken> {
    if let Some(error) = error.filter(|e| !e.is_empty()) {
        anyhow::bail!("the provider refused the login: {}", super::clean(&error));
    }
    match got_state {
        Some(got) if constant_time_eq(got.as_bytes(), state.as_bytes()) => {}
        Some(_) => anyhow::bail!("the login's state does not match this attempt; start again"),
        None => anyhow::bail!("the redirect has no state"),
    }
    let access_token = access
        .filter(|t| !t.is_empty())
        .ok_or_else(|| anyhow!("the redirect has no access token"))?;
    Ok(FragmentToken {
        access_token,
        expires_in: expires_in.and_then(|e| e.trim().parse().ok()).unwrap_or(0),
    })
}

/// Serves the page a browser lands on after an implicit grant, which has to
/// forward the fragment — a browser never sends one — and waits for it.
///
/// `GET <callback>` is answered with a page whose script `POST`s the
/// fragment's `access_token`, `expires_in`, `state` (and any `error`) as JSON
/// to `<token_path>`; that request settles the login. As for
/// [`wait_for_callback`], only a `Host` of this machine is served.
pub fn wait_for_fragment(
    listener: &TcpListener,
    callback: &str,
    token_path: &str,
    state: &str,
    timeout: Duration,
) -> anyhow::Result<FragmentToken> {
    listener
        .set_nonblocking(true)
        .context("listen for the callback")?;
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Some(result) = serve_fragment(stream, callback, token_path, state) {
                    return result;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    anyhow::bail!("timed out waiting for the browser to come back");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e).context("accept the callback"),
        }
    }
}

fn bootstrap_page(token_path: &str) -> String {
    format!(
        "<!doctype html><title>llmman</title><p id=\"m\">Finishing sign-in…</p><script>\
var h=new URLSearchParams((location.hash||'').slice(1)),q=new URLSearchParams(location.search||'');\
var e=h.get('error')||q.get('error');\
var b=e?{{error:h.get('error_description')||e}}:{{access_token:h.get('access_token')||'',expires_in:h.get('expires_in')||'0',state:h.get('state')||''}};\
fetch({token_path:?},{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(b)}})\
.then(function(r){{document.getElementById('m').textContent=r.ok&&!e?'Signed in. You can close this tab and return to the terminal.':'The sign-in did not complete. Return to the terminal.'}})\
.catch(function(){{document.getElementById('m').textContent='The sign-in did not complete. Return to the terminal.'}});\
</script>"
    )
}

/// One connection of [`wait_for_fragment`]: `Some(result)` when it settles
/// the login.
fn serve_fragment(
    mut stream: TcpStream,
    callback: &str,
    token_path: &str,
    state: &str,
) -> Option<anyhow::Result<FragmentToken>> {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut data = Vec::new();
    let mut buf = [0u8; 2048];
    let header_end = loop {
        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if data.len() > 16 * 1024 {
            return None;
        }
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => return None,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
    };
    let head = String::from_utf8_lossy(&data[..header_end]).into_owned();
    let mut lines = head.lines();
    let request = lines.next().unwrap_or("");
    let header = |name: &str| {
        head.lines().skip(1).find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.trim().to_string())
        })
    };
    let host = header("host").unwrap_or_default().to_ascii_lowercase();
    let host_name = host.rsplit_once(':').map_or(host.as_str(), |(h, _)| h);
    let local = matches!(host_name, "localhost" | "127.0.0.1" | "[::1]");
    let mut parts = request.split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let path = target.split('?').next().unwrap_or("");
    let reply = |stream: &mut TcpStream, status: &str, kind: &str, body: &str| {
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nCache-Control: no-store\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    };
    if !local {
        reply(&mut stream, "404 Not Found", "text/plain", "not found");
        return None;
    }
    if method == "GET" && path == callback {
        reply(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            &bootstrap_page(token_path),
        );
        return None;
    }
    if method == "POST" && path == token_path {
        let length: usize = header("content-length")
            .and_then(|l| l.parse().ok())
            .unwrap_or(0)
            .min(16 * 1024);
        let mut body = data[header_end..].to_vec();
        while body.len() < length {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => body.extend_from_slice(&buf[..n]),
            }
        }
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
        let text = |k: &str| json.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let result = token_from_parts(
            text("access_token"),
            text("expires_in"),
            text("state"),
            text("error"),
            state,
        );
        match &result {
            Ok(_) => reply(&mut stream, "200 OK", "application/json", "{\"ok\":true}"),
            Err(_) => reply(
                &mut stream,
                "400 Bad Request",
                "application/json",
                "{\"ok\":false}",
            ),
        }
        return Some(result);
    }
    reply(&mut stream, "404 Not Found", "text/plain", "not found");
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_challenge_is_the_s256_of_the_verifier() {
        // RFC 7636 appendix B.
        let pkce = Pkce::from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".into());
        assert_eq!(
            pkce.challenge,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        let fresh = Pkce::generate().unwrap();
        assert!(fresh.verifier.len() >= 43);
        assert_ne!(fresh.verifier, Pkce::generate().unwrap().verifier);
    }

    #[test]
    fn a_pasted_redirect_yields_its_code_only_for_this_state() {
        let url = "http://localhost:3000/openrouter-oauth/callback?state=s1&code=abc%2B1";
        assert_eq!(code_from_callback(url, "s1").unwrap(), "abc+1");
        assert_eq!(code_from_callback("state=s1&code=c", "s1").unwrap(), "c");
        assert_eq!(
            code_from_callback(&format!("{url}#frag"), "s1").unwrap(),
            "abc+1"
        );
        let wrong = code_from_callback(url, "s2").unwrap_err().to_string();
        assert!(wrong.contains("state"), "{wrong}");
        assert!(code_from_callback("http://x/cb?code=c", "s1").is_err());
        assert!(code_from_callback("http://x/cb?state=s1", "s1").is_err());
        assert!(code_from_callback("garbage", "s1").is_err());
        let refused = code_from_callback("http://x/cb?error=access_denied&state=s1", "s1")
            .unwrap_err()
            .to_string();
        assert!(refused.contains("access_denied"), "{refused}");
    }

    fn get(port: u16, request_target: &str, host: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(s, "GET {request_target} HTTP/1.1\r\nHost: {host}\r\n\r\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn the_listener_answers_the_browser_and_returns_the_code() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let browser = std::thread::spawn(move || {
            // A stray request and a rebinding attempt do not settle it.
            assert!(get(port, "/favicon.ico", "localhost").starts_with("HTTP/1.1 404"));
            assert!(get(port, "/cb?state=s&code=evil", "evil.example").starts_with("HTTP/1.1 404"));
            // A callback with the wrong state is answered 400 and reported.
            get(port, "/cb?state=wrong&code=x", &format!("localhost:{port}"))
        });
        let err = wait_for_callback(&listener, "/cb", "s", Duration::from_secs(10)).unwrap_err();
        assert!(err.to_string().contains("state"), "{err}");
        assert!(browser.join().unwrap().starts_with("HTTP/1.1 400"));

        let browser = std::thread::spawn(move || {
            get(
                port,
                "/cb?state=s&code=the-code",
                &format!("127.0.0.1:{port}"),
            )
        });
        let code = wait_for_callback(&listener, "/cb", "s", Duration::from_secs(10)).unwrap();
        assert_eq!(code, "the-code");
        let page = browser.join().unwrap();
        assert!(
            page.starts_with("HTTP/1.1 200") && page.contains("Signed in"),
            "{page}"
        );
    }

    fn post(port: u16, path: &str, host: &str, body: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            s,
            "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn the_fragment_listener_serves_a_page_and_takes_the_forwarded_token() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let local = format!("localhost:{port}");
        let browser = std::thread::spawn(move || {
            let page = get(port, "/cb", &format!("localhost:{port}"));
            assert!(
                page.starts_with("HTTP/1.1 200") && page.contains("/tok"),
                "{page}"
            );
            // Other hosts and paths are not served; a wrong state is refused.
            assert!(get(port, "/cb", "evil.example").starts_with("HTTP/1.1 404"));
            assert!(post(port, "/tok", "evil.example", "{}").starts_with("HTTP/1.1 404"));
            assert!(get(port, "/other", &format!("localhost:{port}")).starts_with("HTTP/1.1 404"));
            post(
                port,
                "/tok",
                &format!("localhost:{port}"),
                r#"{"access_token":"x","expires_in":"1","state":"wrong"}"#,
            )
        });
        let err =
            wait_for_fragment(&listener, "/cb", "/tok", "s", Duration::from_secs(10)).unwrap_err();
        assert!(err.to_string().contains("state"), "{err}");
        assert!(browser.join().unwrap().starts_with("HTTP/1.1 400"));

        let body = r#"{"access_token":"the-token","expires_in":"900","state":"s"}"#;
        let browser = std::thread::spawn(move || post(port, "/tok", &local, body));
        let got =
            wait_for_fragment(&listener, "/cb", "/tok", "s", Duration::from_secs(10)).unwrap();
        assert_eq!(
            got,
            FragmentToken {
                access_token: "the-token".into(),
                expires_in: 900
            }
        );
        assert!(browser.join().unwrap().starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn it_times_out_when_the_browser_never_comes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let err = wait_for_callback(&listener, "/cb", "s", Duration::from_millis(150)).unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
    }
}
