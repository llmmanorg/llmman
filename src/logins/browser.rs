//! The browser and device logins of Poe, DigitalOcean, Chutes, GitLab,
//! Snowflake and MiniMax. Endpoints, client ids and scopes are taken from
//! opencode and OpenClaw and cited where used; where a reference leaves a
//! value to the user (Chutes' client id), the user supplies it.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{anyhow, bail};
use base64::Engine as _;
use serde_json::Value;

use super::cloud::param;
use super::device::{self, Poll};
use super::pkce::{self, Pkce};
use super::{
    clean, failed, now_ms, token_reply, Ctx, Endpoints, Http, NetError, Options, Tokens,
    BROWSER_TIMEOUT, FLOW_CHUTES, FLOW_DIGITALOCEAN, FLOW_GITLAB, FLOW_MINIMAX, FLOW_POE,
    FLOW_SNOWFLAKE,
};
use crate::credentials::{Credential, OAuth, RefreshError};

/// Poe's OAuth client and issuer. opencode `poe.ts` `clientID`, `issuer`.
const POE_CLIENT_ID: &str = "client_728290227fc048cc9262091a1ea197ea";
/// DigitalOcean's. opencode `digitalocean.ts` `clientID`, `authorizeEndpoint`,
/// `callbackPort`, `callbackPath`, `tokenPath`; the scope in `authorize`.
const DO_CLIENT_ID: &str = "b1a6c5158156caac821fd1b30253ca8acb52454a48fa744420e41889cb589f82";
const DO_PORT: u16 = 1456;
const DO_CALLBACK: &str = "/auth/callback";
const DO_TOKEN_PATH: &str = "/auth/token";
const DO_SCOPE: &str = "genai:read inference:query";
/// How long a DigitalOcean token with no stated lifetime is assumed to last:
/// opencode's fallback, 30 days.
const DO_DEFAULT_LIFETIME_MS: u64 = 30 * 24 * 60 * 60 * 1000;
/// Chutes' redirect and scopes. OpenClaw `chutes/index.ts`
/// (`CHUTES_OAUTH_REDIRECT_URI`, `CHUTES_OAUTH_SCOPES` defaults).
const CHUTES_PORT: u16 = 1456;
const CHUTES_CALLBACK: &str = "/oauth-callback";
const CHUTES_SCOPE: &str = "openid profile chutes:invoke";
/// GitLab's: opencode `gitlab.ts` `bundledClientID` (an application that
/// exists on gitlab.com only), `callbackPort`, `oauthScope`.
const GITLAB_CLIENT_ID: &str = "fd180700a8f9c5d5557aca231632dd0611a1135a3bb510a741d5a988a3394fa7";
const GITLAB_PORT: u16 = 8080;
const GITLAB_CALLBACK: &str = "/callback";
const GITLAB_SCOPE: &str = "api";
/// Snowflake's built-in client. opencode `snowflake-cortex.ts` `clientID`
/// ("Same built-in OAuth client and Basic header as V1; this is not a user
/// password").
const SNOWFLAKE_CLIENT_ID: &str = "LOCAL_APPLICATION";
/// MiniMax's client, scope and grant. OpenClaw `minimax/oauth.ts`
/// `MINIMAX_OAUTH_CONFIG`, `MINIMAX_OAUTH_SCOPE`, `MINIMAX_OAUTH_GRANT_TYPE`.
const MINIMAX_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
const MINIMAX_SCOPE: &str = "group_id profile model.completion";
const MINIMAX_GRANT: &str = "urn:ietf:params:oauth:grant-type:user_code";
const MINIMAX_DEFAULT_INTERVAL: Duration = Duration::from_millis(2000);

/// A fixed callback port, or any port under test.
fn port(fixed: u16) -> u16 {
    if cfg!(test) {
        0
    } else {
        fixed
    }
}

fn plain_http_ok(ctx: &Ctx<'_>, url: &reqwest::Url) -> bool {
    url.scheme() == "https" || (ctx.http.allow_http && url.scheme() == "http")
}

// ---------------------------------------------------------------------------
// The shared loopback code flow
// ---------------------------------------------------------------------------

/// What a code flow came back with.
struct Code {
    code: String,
    redirect: String,
    pkce: Pkce,
}

/// Opens `authorize(redirect, pkce, state)` in the browser and waits for it
/// to come back to a listener on this computer; if the port is taken, the
/// user pastes the URL the browser ends up at instead.
fn loopback_code(
    ctx: &Ctx<'_>,
    what: &str,
    host: &str,
    fixed_port: u16,
    path: &str,
    authorize: impl FnOnce(&str, &Pkce, &str) -> anyhow::Result<reqwest::Url>,
) -> anyhow::Result<Code> {
    let pkce = Pkce::generate()?;
    let state = pkce::random_token(32)?;
    let listener = std::net::TcpListener::bind(("127.0.0.1", port(fixed_port))).ok();
    let actual = listener
        .as_ref()
        .and_then(|l| l.local_addr().ok())
        .map_or(fixed_port, |a| a.port());
    let redirect = format!("http://{host}:{actual}{path}");
    let url = authorize(&redirect, &pkce, &state)?;
    println!();
    println!("Open this page to sign in to {what}:\n{url}");
    (ctx.open)(url.as_str());
    let code = match &listener {
        Some(listener) => {
            println!("Waiting for the browser to return to this computer…");
            pkce::wait_for_callback(listener, path, &state, BROWSER_TIMEOUT)?
        }
        None => {
            println!("Port {fixed_port} is busy, so the browser cannot return here.");
            let pasted = (ctx.input)(
                "After signing in, paste the URL the browser ends up at: ",
                false,
            )?;
            pkce::code_from_callback(&pasted, &state)?
        }
    };
    Ok(Code {
        code,
        redirect,
        pkce,
    })
}

fn url_with(base: &str, pairs: &[(&str, &str)]) -> anyhow::Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base).map_err(|_| anyhow!("{base:?} is not a URL"))?;
    url.query_pairs_mut().extend_pairs(pairs.iter().copied());
    Ok(url)
}

/// `tokens` out of a successful exchange, or why it failed.
fn tokens_of(what: &str, reply: Result<super::Reply, NetError>) -> anyhow::Result<Tokens> {
    let reply = reply?;
    if !reply.ok() {
        return Err(failed(what, &reply));
    }
    Tokens::parse(&reply.json()).ok_or_else(|| anyhow!("{what} returned no access token"))
}

// ---------------------------------------------------------------------------
// Poe
// ---------------------------------------------------------------------------

/// Poe: authorization code + PKCE at `poe.com/oauth/authorize` with scope
/// `apikey:create`; the code is exchanged at `api.poe.com/token` for an API
/// key that expires after `api_key_expires_in` seconds, or never when that is
/// null (opencode `poe.ts`). The key is stored as an expiring token; when it
/// lapses the provider reports `expired` and the login is run again.
pub(super) fn poe(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let got = loopback_code(
        ctx,
        "Poe",
        "127.0.0.1",
        0,
        "/callback",
        |redirect, pkce, state| {
            url_with(
                &format!("{}/oauth/authorize", ctx.ep.poe_auth),
                &[
                    ("response_type", "code"),
                    ("client_id", POE_CLIENT_ID),
                    ("redirect_uri", redirect),
                    ("scope", "apikey:create"),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                    ("state", state),
                ],
            )
        },
    )?;
    let reply = ctx.http.post_form(
        &format!("{}/token", ctx.ep.poe_api),
        &[
            ("grant_type", "authorization_code"),
            ("client_id", POE_CLIENT_ID),
            ("code", &got.code),
            ("redirect_uri", &got.redirect),
            ("code_verifier", &got.pkce.verifier),
        ],
        &[],
    )?;
    if !reply.ok() {
        return Err(failed("Poe's token exchange", &reply));
    }
    let body = reply.json();
    let key = body
        .get("api_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|k| !k.is_empty() && !k.contains(char::is_whitespace))
        .ok_or_else(|| anyhow!("Poe's token exchange returned no API key"))?;
    let expires_ms = body
        .get("api_key_expires_in")
        .and_then(Value::as_u64)
        .map(|s| now_ms() + s * 1000);
    Ok(Credential::Token {
        token: key.to_string(),
        expires_ms,
        flow: Some(FLOW_POE.into()),
    })
}

// ---------------------------------------------------------------------------
// DigitalOcean
// ---------------------------------------------------------------------------

/// DigitalOcean: the implicit grant (`response_type=token`) at
/// `cloud.digitalocean.com/v1/oauth/authorize`. The token comes back in the
/// redirect's fragment, which a browser never sends, so the listener's page
/// forwards it ([`pkce::wait_for_fragment`]); opencode `digitalocean.ts`.
pub(super) fn digitalocean(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let state = pkce::random_token(32)?;
    // No PKCE here: if another process held the port, it would be handed
    // the token, so a failed bind ends the login before the browser opens.
    let listener = std::net::TcpListener::bind(("127.0.0.1", port(DO_PORT)))
        .map_err(|e| anyhow!("DigitalOcean login needs 127.0.0.1:{DO_PORT}: {e}"))?;
    let actual = listener.local_addr().map_or(DO_PORT, |a| a.port());
    let redirect = format!("http://localhost:{actual}{DO_CALLBACK}");
    let url = url_with(
        &format!("{}/v1/oauth/authorize", ctx.ep.digitalocean),
        &[
            ("response_type", "token"),
            ("client_id", DO_CLIENT_ID),
            ("redirect_uri", &redirect),
            ("scope", DO_SCOPE),
            ("state", &state),
        ],
    )?;
    println!();
    println!("Open this page to sign in to DigitalOcean:\n{url}");
    (ctx.open)(url.as_str());
    println!("Waiting for the browser to return to this computer…");
    let token = pkce::wait_for_fragment(
        &listener,
        DO_CALLBACK,
        DO_TOKEN_PATH,
        &state,
        BROWSER_TIMEOUT,
    )?;
    let expires_ms = if token.expires_in > 0 {
        now_ms() + token.expires_in * 1000
    } else {
        now_ms() + DO_DEFAULT_LIFETIME_MS
    };
    Ok(Credential::Token {
        token: token.access_token,
        expires_ms: Some(expires_ms),
        flow: Some(FLOW_DIGITALOCEAN.into()),
    })
}

// ---------------------------------------------------------------------------
// Chutes
// ---------------------------------------------------------------------------

/// Chutes: authorization code + PKCE at `api.chutes.ai/idp/authorize`, for
/// an OAuth application the user registered (OpenClaw has no client id of
/// its own: `--param client_id=`, `CHUTES_CLIENT_ID`, or asked for), with
/// `CHUTES_CLIENT_SECRET` sent too when the app has one. The redirect is
/// `http://127.0.0.1:1456/oauth-callback`, which the app must list.
pub(super) fn chutes(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let client = match param(opts, "client_id")
        .map(str::to_string)
        .or_else(|| crate::providers::key_from_env("CHUTES_CLIENT_ID"))
    {
        Some(c) => c,
        None => (ctx.input)("Chutes OAuth client id (cid_...): ", false)?
            .trim()
            .to_string(),
    };
    anyhow::ensure!(
        !client.is_empty() && !client.contains(char::is_whitespace),
        "a Chutes OAuth client id is required"
    );
    let got = loopback_code(
        ctx,
        "Chutes",
        "127.0.0.1",
        CHUTES_PORT,
        CHUTES_CALLBACK,
        |redirect, pkce, state| {
            url_with(
                &format!("{}/idp/authorize", ctx.ep.chutes),
                &[
                    ("client_id", &client),
                    ("redirect_uri", redirect),
                    ("response_type", "code"),
                    ("scope", CHUTES_SCOPE),
                    ("state", state),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                ],
            )
        },
    )?;
    let secret = crate::providers::key_from_env("CHUTES_CLIENT_SECRET");
    let mut form = vec![
        ("grant_type", "authorization_code"),
        ("client_id", client.as_str()),
        ("code", got.code.as_str()),
        ("redirect_uri", got.redirect.as_str()),
        ("code_verifier", got.pkce.verifier.as_str()),
    ];
    if let Some(secret) = &secret {
        form.push(("client_secret", secret));
    }
    let tokens = tokens_of(
        "the Chutes token exchange",
        ctx.http
            .post_form(&format!("{}/idp/token", ctx.ep.chutes), &form, &[]),
    )?;
    let refresh = tokens
        .refresh
        .clone()
        .ok_or_else(|| anyhow!("the Chutes token exchange returned no refresh token"))?;
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_CHUTES.into(),
        expires_ms: tokens.expires_ms(now_ms()),
        access: tokens.access,
        refresh,
        metadata: BTreeMap::from([("client_id".to_string(), client)]),
    }))
}

// ---------------------------------------------------------------------------
// GitLab
// ---------------------------------------------------------------------------

/// A GitLab instance URL as a base: https, no credentials, query or
/// fragment, a path (a relative URL root) kept, no trailing slash.
fn instance_url(ctx: &Ctx<'_>, text: &str) -> anyhow::Result<String> {
    let url = reqwest::Url::parse(text.trim())
        .map_err(|_| anyhow!("{text:?} is not a GitLab instance URL"))?;
    anyhow::ensure!(
        plain_http_ok(ctx, &url)
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "{text:?} is not an https GitLab instance URL"
    );
    Ok(format!(
        "{}{}",
        url.origin().ascii_serialization(),
        url.path().trim_end_matches('/')
    ))
}

/// GitLab (Duo): authorization code + PKCE at `<instance>/oauth/authorize`,
/// scope `api`, redirect `http://127.0.0.1:8080/callback`. gitlab.com has an
/// application of opencode's registered; another instance needs the user's
/// own (not marked confidential) in `GITLAB_OAUTH_CLIENT_ID`. The instance
/// is `--param instance_url=`, `GITLAB_INSTANCE_URL`, else gitlab.com, and is
/// kept with the login (opencode `gitlab.ts`).
pub(super) fn gitlab(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let instance = instance_url(
        ctx,
        &param(opts, "instance_url")
            .map(str::to_string)
            .or_else(|| crate::providers::key_from_env("GITLAB_INSTANCE_URL"))
            .unwrap_or_else(|| ctx.ep.gitlab.clone()),
    )?;
    let own = crate::providers::key_from_env("GITLAB_OAUTH_CLIENT_ID");
    let client = match own {
        Some(client) => client,
        None if instance == ctx.ep.gitlab => GITLAB_CLIENT_ID.to_string(),
        None => bail!(
            "the bundled GitLab OAuth application only exists on gitlab.com. To sign in to \
             {instance}, register an application there with the redirect URI \
             http://127.0.0.1:{GITLAB_PORT}{GITLAB_CALLBACK} and the `{GITLAB_SCOPE}` scope (not \
             marked \"Confidential\"), then set GITLAB_OAUTH_CLIENT_ID to its application ID"
        ),
    };
    let got = loopback_code(
        ctx,
        "GitLab",
        "127.0.0.1",
        GITLAB_PORT,
        GITLAB_CALLBACK,
        |redirect, pkce, state| {
            url_with(
                &format!("{instance}/oauth/authorize"),
                &[
                    ("client_id", &client),
                    ("redirect_uri", redirect),
                    ("response_type", "code"),
                    ("state", state),
                    ("scope", GITLAB_SCOPE),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                ],
            )
        },
    )?;
    let tokens = tokens_of(
        "the GitLab token exchange",
        ctx.http.post_form(
            &format!("{instance}/oauth/token"),
            &[
                ("client_id", client.as_str()),
                ("grant_type", "authorization_code"),
                ("code", got.code.as_str()),
                ("redirect_uri", got.redirect.as_str()),
                ("code_verifier", got.pkce.verifier.as_str()),
            ],
            &[],
        ),
    )?;
    let refresh = tokens
        .refresh
        .clone()
        .ok_or_else(|| anyhow!("the GitLab token exchange returned no refresh token"))?;
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_GITLAB.into(),
        expires_ms: tokens.expires_ms(now_ms()),
        access: tokens.access,
        refresh,
        metadata: BTreeMap::from([
            ("instance_url".to_string(), instance),
            ("client_id".to_string(), client),
        ]),
    }))
}

// ---------------------------------------------------------------------------
// Snowflake
// ---------------------------------------------------------------------------

/// A Snowflake account as it names a host: the scheme, trailing slash and
/// `.snowflakecomputing.com` dropped (opencode `snowflake-cortex.ts`
/// `normalizeAccount`), and only host-label characters left.
pub(crate) fn snowflake_account(text: &str) -> Option<String> {
    let account = text.trim();
    let account = account
        .strip_prefix("https://")
        .or_else(|| account.strip_prefix("http://"))
        .unwrap_or(account)
        .trim_end_matches('/');
    let account = account
        .strip_suffix(".snowflakecomputing.com")
        .unwrap_or(account);
    (!account.is_empty()
        && !account.starts_with(['.', '-'])
        && account
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b'_'))
    .then(|| account.to_string())
}

fn snowflake_base(ep: &Endpoints, account: &str) -> String {
    ep.snowflake
        .clone()
        .unwrap_or_else(|| format!("https://{account}.snowflakecomputing.com"))
}

fn snowflake_basic() -> String {
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD
            .encode(format!("{SNOWFLAKE_CLIENT_ID}:{SNOWFLAKE_CLIENT_ID}"))
    )
}

/// Snowflake Cortex: authorization code + PKCE at
/// `<account>.snowflakecomputing.com/oauth/authorize` as the built-in
/// `LOCAL_APPLICATION` client, scope `refresh_token` (plus
/// `session:role:<role>` for `--param role=`), redirect
/// `http://127.0.0.1:<port>/`; the code is exchanged at `/oauth/token-request`
/// with that client's Basic header (opencode `snowflake-cortex.ts`). The
/// account is `--param account=`, `SNOWFLAKE_ACCOUNT`, or asked for.
pub(super) fn snowflake(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let account = match param(opts, "account")
        .map(str::to_string)
        .or_else(|| crate::providers::key_from_env("SNOWFLAKE_ACCOUNT"))
    {
        Some(a) => a,
        None => (ctx.input)("Snowflake account (myorg-myaccount): ", false)?,
    };
    let account = snowflake_account(&account)
        .ok_or_else(|| anyhow!("{account:?} is not a Snowflake account"))?;
    let scope = match param(opts, "role") {
        None => "refresh_token".to_string(),
        Some(role)
            if role
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') =>
        {
            format!("refresh_token session:role:{role}")
        }
        Some(role) => format!(
            "refresh_token session:role-encoded:{}",
            crate::authscheme::url_encode(role)
        ),
    };
    let base = snowflake_base(&ctx.ep, &account);
    let got = loopback_code(
        ctx,
        "Snowflake",
        "127.0.0.1",
        0,
        "/",
        |redirect, pkce, state| {
            url_with(
                &format!("{base}/oauth/authorize"),
                &[
                    ("client_id", SNOWFLAKE_CLIENT_ID),
                    ("response_type", "code"),
                    ("redirect_uri", redirect),
                    ("state", state),
                    ("scope", &scope),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                ],
            )
        },
    )?;
    let basic = snowflake_basic();
    let tokens = tokens_of(
        "the Snowflake token exchange",
        ctx.http.post_form(
            &format!("{base}/oauth/token-request"),
            &[
                ("grant_type", "authorization_code"),
                ("code", got.code.as_str()),
                ("redirect_uri", got.redirect.as_str()),
                ("code_verifier", got.pkce.verifier.as_str()),
                ("client_id", SNOWFLAKE_CLIENT_ID),
            ],
            &[("authorization", &basic)],
        ),
    )?;
    let refresh = tokens
        .refresh
        .clone()
        .ok_or_else(|| anyhow!("Snowflake's token exchange returned no refresh token"))?;
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_SNOWFLAKE.into(),
        // opencode: `expires_in ?? 600`.
        expires_ms: tokens
            .expires_in
            .map_or(now_ms() + 600_000, |s| now_ms() + s * 1000),
        access: tokens.access,
        refresh,
        metadata: BTreeMap::from([("account".to_string(), account)]),
    }))
}

// ---------------------------------------------------------------------------
// MiniMax
// ---------------------------------------------------------------------------

/// The account host a MiniMax provider's login goes to: `cn` for the China
/// providers, global otherwise. OpenClaw `minimax/oauth.ts`
/// `MINIMAX_OAUTH_CONFIG` (`account.minimax.io`, `account.minimaxi.com`).
fn minimax_region(provider: &str) -> &'static str {
    if provider.contains("-cn") {
        "cn"
    } else {
        "global"
    }
}

fn minimax_base<'a>(ep: &'a Endpoints, region: &str) -> &'a str {
    if region == "cn" {
        &ep.minimax_cn
    } else {
        &ep.minimax_global
    }
}

/// MiniMax's `expired_in`, which names a moment: seconds from now when it is
/// small, an epoch in seconds when it is a ten-digit number, in milliseconds
/// when it is larger. OpenClaw `minimax/oauth.ts`
/// (`MINIMAX_RELATIVE_EXPIRY_SECONDS_THRESHOLD` = 10^9,
/// `MINIMAX_ABSOLUTE_EXPIRY_MS_THRESHOLD` = 10^12).
fn minimax_deadline(expired_in: u64, now: u64) -> u64 {
    if expired_in < 1_000_000_000 {
        now + expired_in * 1000
    } else if expired_in < 1_000_000_000_000 {
        expired_in * 1000
    } else {
        expired_in
    }
}

#[cfg(test)]
pub(super) fn minimax_deadline_for_test(expired_in: u64, now: u64) -> u64 {
    minimax_deadline(expired_in, now)
}

/// What MiniMax's token endpoint said: `status` is `success`, `error`, or
/// anything else for "not approved yet", and `base_resp.status_code` is
/// non-zero on failure.
enum Minimax {
    Done {
        access: String,
        refresh: Option<String>,
        expires_ms: u64,
    },
    Pending,
    Error(String),
}

fn minimax_reply(reply: &super::Reply, now: u64) -> Minimax {
    let body = reply.json();
    let failed_code = body
        .pointer("/base_resp/status_code")
        .and_then(Value::as_i64)
        .is_some_and(|c| c != 0);
    if !reply.ok() || body.get("error").is_some_and(|e| !e.is_null()) || failed_code {
        let message = body
            .pointer("/base_resp/status_msg")
            .and_then(Value::as_str)
            .map(clean)
            .unwrap_or_else(|| format!("HTTP {}", reply.status));
        return Minimax::Error(message);
    }
    match body.get("status").and_then(Value::as_str) {
        Some("error") => Minimax::Error(
            body.pointer("/base_resp/status_msg")
                .and_then(Value::as_str)
                .map_or_else(|| "an error occurred".to_string(), clean),
        ),
        Some("success") => {
            let text = |k: &str| {
                body.get(k)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_string)
            };
            match (
                text("access_token"),
                body.get("expired_in").and_then(Value::as_u64),
            ) {
                (Some(access), Some(expired)) if expired > 0 => Minimax::Done {
                    access,
                    refresh: text("refresh_token"),
                    expires_ms: minimax_deadline(expired, now),
                },
                _ => Minimax::Error("MiniMax returned an incomplete token".into()),
            }
        }
        _ => Minimax::Pending,
    }
}

/// MiniMax: a user-code flow. A request at `/oauth2/device/code` (PKCE
/// challenge, `state`) returns a code and a page; the token endpoint is
/// polled with the code and verifier until it says `success`. The region is
/// the provider's (`minimax-cn*` are China, the rest global).
pub(super) fn minimax(ctx: &Ctx<'_>, provider: &str) -> anyhow::Result<Credential> {
    let region = minimax_region(provider);
    let base = minimax_base(&ctx.ep, region).to_string();
    let pkce = Pkce::generate()?;
    let state = pkce::random_token(16)?;
    let request_id = pkce::random_token(16)?;
    let started = ctx.http.post_form(
        &format!("{base}/oauth2/device/code"),
        &[
            ("response_type", "code"),
            ("client_id", MINIMAX_CLIENT_ID),
            ("scope", MINIMAX_SCOPE),
            ("code_challenge", &pkce.challenge),
            ("code_challenge_method", "S256"),
            ("state", &state),
        ],
        &[("x-request-id", &request_id)],
    )?;
    if !started.ok() {
        return Err(failed("requesting a MiniMax code", &started));
    }
    let body = started.json();
    let (Some(user_code), Some(verification)) = (
        body.get("user_code")
            .and_then(Value::as_str)
            .filter(|c| !c.is_empty()),
        body.get("verification_uri")
            .and_then(Value::as_str)
            .filter(|c| !c.is_empty()),
    ) else {
        bail!("MiniMax's code reply was missing the code or the page");
    };
    if body.get("state").and_then(Value::as_str) != Some(state.as_str()) {
        bail!("MiniMax's state does not match this attempt; start again");
    }
    let verification = reqwest::Url::parse(verification)
        .ok()
        .filter(|u| {
            let host = u.host_str().unwrap_or("");
            plain_http_ok(ctx, u)
                && u.username().is_empty()
                && (ctx.http.allow_http
                    || host == "minimax.io"
                    || host.ends_with(".minimax.io")
                    || host == "minimaxi.com"
                    || host.ends_with(".minimaxi.com"))
        })
        .ok_or_else(|| anyhow!("MiniMax returned an unexpected verification URL"))?;
    let interval = body
        .get("interval")
        .and_then(Value::as_u64)
        .filter(|i| *i > 0)
        .map_or(MINIMAX_DEFAULT_INTERVAL, Duration::from_millis);
    let deadline = body
        .get("expired_in")
        .and_then(Value::as_u64)
        .map_or(now_ms() + 300_000, |e| minimax_deadline(e, now_ms()));
    super::show_code(ctx, verification.as_str(), user_code);

    let timeout = Duration::from_millis(deadline.saturating_sub(now_ms()));
    let (access, refresh, expires_ms) = device::run(&*ctx.clock, interval, timeout, || match ctx
        .http
        .post_form(
            &format!("{base}/oauth2/token"),
            &[
                ("grant_type", MINIMAX_GRANT),
                ("client_id", MINIMAX_CLIENT_ID),
                ("user_code", user_code),
                ("code_verifier", &pkce.verifier),
            ],
            &[],
        ) {
        Ok(r) => match minimax_reply(&r, now_ms()) {
            Minimax::Done {
                access,
                refresh: Some(refresh),
                expires_ms,
            } => Poll::Done((access, refresh, expires_ms)),
            Minimax::Done { .. } => Poll::Failed("MiniMax returned no refresh token".into()),
            Minimax::Pending => Poll::Pending,
            Minimax::Error(why) => Poll::Failed(why),
        },
        Err(e) => Poll::Failed(e.to_string()),
    })?;
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_MINIMAX.into(),
        access,
        refresh,
        expires_ms,
        metadata: BTreeMap::from([("region".to_string(), region.to_string())]),
    }))
}

// ---------------------------------------------------------------------------
// Refreshing
// ---------------------------------------------------------------------------

/// Renews a login of one of this module's flows, or `None` for another.
pub(super) fn refresh(
    http: &Http,
    ep: &Endpoints,
    old: &OAuth,
    now: u64,
) -> Option<Result<OAuth, RefreshError>> {
    let meta = |k: &str| old.metadata.get(k).map(String::as_str).unwrap_or_default();
    let reply = match old.flow.as_str() {
        // OpenClaw `chutes/oauth.ts` `refreshChutesOAuthCredential`.
        FLOW_CHUTES => {
            let secret = crate::providers::key_from_env("CHUTES_CLIENT_SECRET");
            let mut form = vec![
                ("grant_type", "refresh_token"),
                ("client_id", meta("client_id")),
                ("refresh_token", old.refresh.as_str()),
            ];
            if let Some(secret) = &secret {
                form.push(("client_secret", secret));
            }
            http.post_form(&format!("{}/idp/token", ep.chutes), &form, &[])
        }
        // opencode `gitlab.ts`: the refresh token rotates.
        FLOW_GITLAB => http.post_form(
            &format!("{}/oauth/token", meta("instance_url")),
            &[
                ("client_id", meta("client_id")),
                ("grant_type", "refresh_token"),
                ("refresh_token", old.refresh.as_str()),
                (
                    "redirect_uri",
                    &format!("http://127.0.0.1:{GITLAB_PORT}{GITLAB_CALLBACK}"),
                ),
            ],
            &[],
        ),
        // opencode `snowflake-cortex.ts` `token`.
        FLOW_SNOWFLAKE => {
            let Some(account) = snowflake_account(meta("account")) else {
                return Some(Err(RefreshError::Rejected(
                    "the login has no Snowflake account".into(),
                )));
            };
            http.post_form(
                &format!("{}/oauth/token-request", snowflake_base(ep, &account)),
                &[
                    ("grant_type", "refresh_token"),
                    ("refresh_token", old.refresh.as_str()),
                    ("client_id", SNOWFLAKE_CLIENT_ID),
                ],
                &[("authorization", &snowflake_basic())],
            )
        }
        // OpenClaw `minimax/oauth.ts` `refreshMiniMaxPortalCredential`.
        FLOW_MINIMAX => {
            let region = if meta("region") == "cn" {
                "cn"
            } else {
                "global"
            };
            let reply = http.post_form(
                &format!("{}/oauth2/token", minimax_base(ep, region)),
                &[
                    ("grant_type", "refresh_token"),
                    ("client_id", MINIMAX_CLIENT_ID),
                    ("refresh_token", old.refresh.as_str()),
                ],
                &[],
            );
            return Some(match reply {
                Err(e) => Err(RefreshError::Transient(e.to_string())),
                Ok(r) if (500..=599).contains(&r.status) || r.status == 429 => {
                    Err(RefreshError::Transient(format!("HTTP {}", r.status)))
                }
                Ok(r) => match minimax_reply(&r, now) {
                    Minimax::Done {
                        access,
                        refresh,
                        expires_ms,
                    } => Ok(OAuth {
                        access,
                        // A server that does not rotate keeps the one in hand.
                        refresh: refresh.unwrap_or_else(|| old.refresh.clone()),
                        expires_ms,
                        ..old.clone()
                    }),
                    Minimax::Pending => Err(RefreshError::Rejected(
                        "MiniMax did not issue a token".into(),
                    )),
                    Minimax::Error(why) => Err(RefreshError::Rejected(why)),
                },
            });
        }
        _ => return None,
    };
    Some(token_reply(reply).map(|tokens| OAuth {
        flow: old.flow.clone(),
        expires_ms: if old.flow == FLOW_SNOWFLAKE {
            tokens.expires_in.map_or(now + 600_000, |s| now + s * 1000)
        } else {
            tokens.expires_ms(now)
        },
        refresh: tokens.refresh.unwrap_or_else(|| old.refresh.clone()),
        access: tokens.access,
        metadata: old.metadata.clone(),
    }))
}

/// A stored login's own fact that stands for a provider variable: the
/// Snowflake account a login was made for is `SNOWFLAKE_ACCOUNT` as far as
/// the provider's base URL is concerned.
pub fn login_var(provider: &str, var: &str) -> Option<String> {
    if provider != "snowflake-cortex" || var != "SNOWFLAKE_ACCOUNT" {
        return None;
    }
    match crate::credentials::Store::user()?.get(provider)? {
        Credential::OAuth(o) if o.flow == FLOW_SNOWFLAKE => {
            o.metadata.get("account").and_then(|a| snowflake_account(a))
        }
        _ => None,
    }
}
