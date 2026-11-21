//! Docker Hub's web-based device-authorization login flow — the
//! "USING WEB-BASED LOGIN" UX `docker login` shows by default — ported
//! from `docker/cli`'s `internal/oauth` package (`manager.LoginDevice`,
//! `api.GetDeviceCode`/`WaitForDeviceToken`/`GetAutoPAT`).
//!
//! This is a standard RFC 8628 OAuth 2.0 Device Authorization Grant
//! against Docker's Auth0 tenant at `login.docker.com`, followed by
//! exchanging the resulting OAuth access token for a real Docker Hub
//! Personal Access Token (PAT) — that PAT, plus the username decoded out
//! of the access token's JWT claims, is what actually gets handed to
//! `ffi::login` and stored as the registry credential, exactly like a
//! normal `-u`/`-p` login would. No separate storage format or
//! verification step exists for this path.
//!
//! Deliberately does *not* implement `docker logout`'s OAuth-token-revoke
//! step: llmman's `logout` behaves the same for docker.io as for any other
//! registry (erase the stored credential), which already covers the
//! practical effect of "logging out" — the PAT itself remains valid on
//! Hub's side either way, same as a PAT obtained through any other login
//! method.

use std::io::{BufRead, IsTerminal};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use serde::Deserialize;

/// Auth0 tenant host used for Docker Hub's device-authorization flow.
const TENANT: &str = "login.docker.com";
/// OAuth client id `docker/cli` registers all its device-flow requests
/// under. Hardcoded in docker/cli itself, not a secret — the equivalent of
/// a public OAuth client id for a native/CLI application.
const CLIENT_ID: &str = "L4v0dmlNBpYUjGGab0C2JtgTgXr1Qz4d";
const AUDIENCE: &str = "https://hub.docker.com";

fn tenant_url() -> String {
    format!("https://{TENANT}")
}

fn user_agent() -> String {
    format!(
        "llmman:{}:{}-{}",
        env!("CARGO_PKG_VERSION").replace('.', "_"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    /// Auth0's device-flow field that already embeds the user code as a
    /// query param (e.g. `.../activate?user_code=QRTL-DTTK`); the query
    /// string is stripped only for display — see `login_device` below —
    /// and kept intact for the browser-open call.
    verification_uri_complete: String,
    expires_in: u64,
    interval: u64,
}

#[derive(Deserialize, Default)]
struct ErrorBody {
    error_description: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    #[serde(default)]
    error_description: String,
}

/// The result of a successful device login: a Hub username and a
/// Personal Access Token to use as the password when storing credentials.
pub struct DeviceLoginResult {
    pub username: String,
    pub password: String,
}

/// Runs the full device-authorization flow to completion, printing the
/// same messages `docker login` does along the way.
///
/// Returns `Ok(None)` when the flow couldn't even be *started* (network
/// error reaching `login.docker.com`, or a response missing `user_code`)
/// — mirroring `docker/cli`'s `ErrDeviceLoginStartFail` sentinel — so the
/// caller can fall back to username/password login. Once the flow has
/// started, any further failure (timeout, denial, network error while
/// polling) is a hard `Err`: `docker login` does not fall back at that
/// point either.
pub fn login_device() -> anyhow::Result<Option<DeviceLoginResult>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(user_agent())
        .build()
        .context("build http client")?;

    let state = match get_device_code(&client) {
        Ok(s) if !s.user_code.is_empty() => s,
        _ => return Ok(None),
    };

    println!();
    println!("USING WEB-BASED LOGIN");
    println!();
    println!(
        "i Info \u{2192} To sign in with credentials on the command line, use 'llmman login -u <username>'"
    );
    println!();
    println!();
    println!(
        "Your one-time device confirmation code is: {}",
        state.user_code
    );
    let display_url = state
        .verification_uri_complete
        .split('?')
        .next()
        .unwrap_or(&state.verification_uri_complete);
    println!("Press ENTER to open your browser or submit your device code here: {display_url}");
    println!();
    println!("Waiting for authentication in the browser\u{2026}");

    // A background thread waits for the user to press ENTER on stdin, then
    // opens the browser — this must not block the polling loop below, and
    // authentication can complete without it (the user can just visit the
    // printed URL and type in the device code on another device). Skipped
    // entirely when stdin isn't a terminal (nothing meaningful to wait for).
    if std::io::stdin().is_terminal() {
        let uri = state.verification_uri_complete.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().lock().read_line(&mut line);
            let _ = open_browser(&uri);
        });
    }

    let token = wait_for_device_token(&client, &state)?;
    let access_token = token
        .access_token
        .ok_or_else(|| anyhow!("missing access_token in device token response"))?;

    let username = username_from_jwt(&access_token)
        .ok_or_else(|| anyhow!("could not determine Hub username from access token"))?;
    let password = get_auto_pat(&client, &access_token)?;

    Ok(Some(DeviceLoginResult { username, password }))
}

/// `POST {tenant}/oauth/device/code` — starts the device-authorization
/// session (RFC 8628 step 1).
fn get_device_code(client: &reqwest::blocking::Client) -> anyhow::Result<DeviceCodeResponse> {
    let resp = client
        .post(format!("{}/oauth/device/code", tenant_url()))
        .form(&[
            ("client_id", CLIENT_ID),
            ("audience", AUDIENCE),
            ("scope", "openid offline_access"),
        ])
        .timeout(Duration::from_secs(15))
        .send()
        .context("request device code")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let desc = resp
            .json::<ErrorBody>()
            .unwrap_or_default()
            .error_description
            .unwrap_or_default();
        anyhow::bail!("unexpected response from tenant: {status} {desc}");
    }
    resp.json().context("parse device code response")
}

/// `POST {tenant}/oauth/token` — polls for completion (RFC 8628 step 2),
/// waiting `state.interval` seconds between attempts and giving up after
/// `state.expires_in` seconds total. `authorization_pending` means "keep
/// polling"; any other non-null `error` (e.g. `access_denied`,
/// `expired_token`) is terminal.
fn wait_for_device_token(
    client: &reqwest::blocking::Client,
    state: &DeviceCodeResponse,
) -> anyhow::Result<TokenResponse> {
    let deadline = Instant::now() + Duration::from_secs(state.expires_in);
    let interval = Duration::from_secs(state.interval.max(1));
    loop {
        std::thread::sleep(interval);
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for device token");
        }
        let resp = client
            .post(format!("{}/oauth/token", tenant_url()))
            .form(&[
                ("client_id", CLIENT_ID),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", state.device_code.as_str()),
            ])
            .timeout(Duration::from_secs(60))
            .send()
            .context("poll device token")?;
        // Auth0 returns a non-2xx status (e.g. 403) while pending, so the
        // body is decoded regardless of the HTTP status code.
        let token: TokenResponse = resp.json().context("parse device token response")?;
        match token.error.as_deref() {
            Some("authorization_pending") => continue,
            Some(_) => {
                anyhow::bail!(
                    "failed waiting for authentication: {}",
                    token.error_description
                );
            }
            None => return Ok(token),
        }
    }
}

/// `POST https://hub.docker.com/v2/access-tokens/desktop-generate` —
/// exchanges the OAuth access token for a real Docker Hub Personal Access
/// Token, which is what actually gets stored as the login credential.
fn get_auto_pat(client: &reqwest::blocking::Client, access_token: &str) -> anyhow::Result<String> {
    #[derive(Deserialize)]
    struct PatResponse {
        data: PatData,
    }
    #[derive(Deserialize)]
    struct PatData {
        token: String,
    }

    let resp = client
        .post("https://hub.docker.com/v2/access-tokens/desktop-generate")
        .bearer_auth(access_token)
        .timeout(Duration::from_secs(15))
        .send()
        .context("request auto-generated PAT")?;
    if resp.status() != reqwest::StatusCode::CREATED {
        anyhow::bail!("unexpected response from Hub: {}", resp.status());
    }
    let body: PatResponse = resp.json().context("parse PAT response")?;
    Ok(body.data.token)
}

/// Decode — without verifying — the `https://hub.docker.com` custom
/// claim's `username` field out of a JWT access token, exactly as
/// docker/cli's `oauth.GetClaims` does. Trust is placed in TLS and the
/// token having come from `login.docker.com` itself, not in a client-side
/// signature check: this is only ever used to label whose credential is
/// being stored, right before that same token is exchanged for the actual
/// PAT that does the real authenticating.
fn username_from_jwt(token: &str) -> Option<String> {
    crate::logins::jwt_payload(token)?
        .get("https://hub.docker.com")?
        .get("username")?
        .as_str()
        .map(str::to_string)
}

/// Best-effort cross-platform "open a URL in the default browser", mirroring
/// `github.com/pkg/browser.OpenURL` (used by docker/cli for the same
/// purpose): `open` on macOS, the first of `xdg-open`/`x-www-browser`/
/// `www-browser` found on Linux, and `rundll32` on Windows. Failure is
/// silently ignored by the caller — the user can still authenticate by
/// visiting the printed URL manually.
pub(crate) fn open_browser(url: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).status()?;
    }
    #[cfg(target_os = "windows")]
    {
        // Not `cmd /C start`: it would split the URL at each `&`.
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .status()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        for tool in ["xdg-open", "x-www-browser", "www-browser"] {
            if std::process::Command::new(tool).arg(url).status().is_ok() {
                break;
            }
        }
    }
    Ok(())
}
