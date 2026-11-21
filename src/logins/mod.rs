//! Obtaining and renewing a provider credential: `llmman login --provider`,
//! and the refreshing the daemon does on the way to a request.
//!
//! Every URL, client id and header is taken from the OpenClaw, opencode and
//! litellm sources and cited where it is used. ChatGPT, Copilot, xAI,
//! OpenRouter and Anthropic are here; the other browser and device logins
//! are in [`browser`], the cloud chains in [`cloud`]. The generic methods
//! (`key`, `command`, `token-file`) work for every provider; [`ADVANCED`]
//! adds `client-credentials` and `jwt-bearer`.

pub mod browser;
pub mod cloud;
pub mod device;
pub mod import;
pub mod pkce;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::io::{BufRead, IsTerminal, Read};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use serde_json::{json, Value};

use crate::credentials::{
    now_ms, CliLogin, Credential, ExternalSource, OAuth, RefreshError, Refresher, Resolved, Store,
    Style,
};
use device::{Clock, Poll};

pub const FLOW_CHATGPT: &str = "openai-chatgpt";
pub const FLOW_COPILOT: &str = "github-copilot";
pub const FLOW_XAI: &str = "xai-supergrok";
pub const FLOW_OPENROUTER: &str = "openrouter-oauth";
pub const FLOW_SETUP_TOKEN: &str = "anthropic-setup-token";
pub const FLOW_CLAUDE: &str = "anthropic-claude";
pub const FLOW_POE: &str = "poe-oauth";
pub const FLOW_DIGITALOCEAN: &str = "digitalocean-oauth";
pub const FLOW_CHUTES: &str = "chutes-oauth";
pub const FLOW_GITLAB: &str = "gitlab-oauth";
pub const FLOW_SNOWFLAKE: &str = "snowflake-oauth";
pub const FLOW_MINIMAX: &str = "minimax-portal";

/// The OAuth client ChatGPT's device flow and refresh run as: the Codex
/// CLI's. OpenClaw `openai-chatgpt-device-code.ts` `OPENAI_CODEX_CLIENT_ID`;
/// litellm `chatgpt/common_utils.py` `CHATGPT_CLIENT_ID`.
const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// Copilot's device-flow client. OpenClaw `github-copilot/login.ts`
/// `CLIENT_ID`; litellm `github_copilot/common_utils.py`
/// `DEFAULT_GITHUB_CLIENT_ID`. (opencode registers its own,
/// `Ov23li8tweQw6odWQebz`; the editor headers `providers.rs` sends are the
/// ones that go with this one.)
const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
/// xAI's device-flow client and scope. OpenClaw `xai-oauth.ts`
/// `XAI_OAUTH_CLIENT_ID`/`XAI_OAUTH_SCOPE`; opencode `xai.ts`.
const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
/// Claude Code's OAuth client, for renewing a login copied from it.
/// OpenClaw `llm/utils/oauth/anthropic.ts` `CLIENT_ID`.
const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// The OpenRouter callback: OpenClaw `openrouter/oauth.ts` — a fixed
/// port, which OpenRouter accepts for `localhost`.
const OPENROUTER_CALLBACK_PORT: u16 = 3000;
const OPENROUTER_CALLBACK_PATH: &str = "/openrouter-oauth/callback";

/// How long a user has to approve a ChatGPT device code. OpenClaw
/// `OPENAI_CODEX_DEVICE_CODE_TIMEOUT_MS`; litellm `DEVICE_CODE_TIMEOUT_SECONDS`.
const CHATGPT_DEVICE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_INTERVAL: Duration = Duration::from_secs(5);
const BROWSER_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// A token response is never more than this.
const MAX_REPLY: u64 = 1 << 20;

// ---------------------------------------------------------------------------
// What a provider offers
// ---------------------------------------------------------------------------

/// One way to log in.
pub struct Method {
    pub name: &'static str,
    pub summary: &'static str,
}

/// The logins a provider has of its own, the first being its default.
pub struct Profile {
    pub provider: &'static str,
    pub methods: &'static [Method],
}

const GENERIC: &[Method] = &[
    Method {
        name: "key",
        summary: "paste an API key and store it",
    },
    Method {
        name: "command",
        summary: "run a command for the key each time (--command)",
    },
    Method {
        name: "token-file",
        summary: "read the key from an owner-only file (--file)",
    },
];

/// Methods any provider takes besides [`GENERIC`], for one whose credential
/// is a token minted at an endpoint of the user's naming: not listed among
/// a provider's own, listed by `llmman login --provider <id> --method ?`.
pub const ADVANCED: &[Method] = &[
    Method {
        name: "client-credentials",
        summary: "OAuth 2.0 client credentials at a token URL (--param token_url= client_id=)",
    },
    Method {
        name: "jwt-bearer",
        summary:
            "a signed JWT assertion at a token URL (--file key.pem, --param token_url= issuer=)",
    },
];

const AWS_METHODS: &[Method] = &[
    Method {
        name: "aws-chain",
        summary:
            "the AWS credential chain: environment, profile, SSO, web identity, container, EC2",
    },
    Method {
        name: "aws-profile",
        summary: "a named AWS profile (--param profile=)",
    },
];

const MINIMAX_METHODS: &[Method] = &[Method {
    name: "device",
    summary: "sign in with a MiniMax account (user code)",
}];

const PROFILES: &[Profile] = &[
    Profile {
        provider: "openai-chatgpt",
        methods: &[
            Method {
                name: "device",
                summary: "sign in with a ChatGPT account (device code)",
            },
            Method {
                name: "import-codex",
                summary: "reuse the Codex CLI's login (~/.codex/auth.json)",
            },
        ],
    },
    Profile {
        provider: "github-copilot",
        methods: &[Method {
            name: "device",
            summary: "sign in with GitHub (device code)",
        }],
    },
    Profile {
        provider: "xai-supergrok",
        methods: &[Method {
            name: "device",
            summary: "sign in with a SuperGrok subscription (device code)",
        }],
    },
    Profile {
        provider: "openrouter",
        methods: &[Method {
            name: "oauth",
            summary: "sign in in the browser; OpenRouter hands back an API key",
        }],
    },
    Profile {
        provider: "amazon-bedrock",
        methods: AWS_METHODS,
    },
    Profile {
        provider: "amazon-bedrock-mantle",
        methods: AWS_METHODS,
    },
    Profile {
        provider: "google-vertex",
        methods: &[
            Method {
                name: "adc",
                summary: "Application Default Credentials (gcloud, GOOGLE_APPLICATION_CREDENTIALS, metadata server)",
            },
            Method {
                name: "service-account",
                summary: "a service account key file (--file key.json)",
            },
        ],
    },
    Profile {
        provider: "azure",
        methods: &[
            Method {
                name: "entra",
                summary: "a Microsoft Entra client secret (--param tenant_id= client_id=)",
            },
            Method {
                name: "managed-identity",
                summary: "the managed identity of this VM, App Service or container",
            },
            Method {
                name: "federated",
                summary: "a federated (workload identity) token file (--file)",
            },
            Method {
                name: "cli",
                summary: "whoever is signed in to the Azure CLI (`az login`)",
            },
        ],
    },
    Profile {
        provider: "databricks",
        methods: &[Method {
            name: "oauth-m2m",
            summary: "a service principal's OAuth M2M credentials (--param host= client_id=)",
        }],
    },
    Profile {
        provider: "sap-ai-core",
        methods: &[Method {
            name: "service-key",
            summary: "a SAP AI Core service key (--file key.json, or pasted)",
        }],
    },
    Profile {
        provider: "watsonx",
        methods: &[Method {
            name: "apikey",
            summary: "an IBM Cloud API key, exchanged for IAM tokens",
        }],
    },
    Profile {
        provider: "snowflake-cortex",
        methods: &[Method {
            name: "oauth",
            summary: "sign in to a Snowflake account in the browser (--param account= role=)",
        }],
    },
    Profile {
        provider: "gitlab",
        methods: &[Method {
            name: "pkce",
            summary: "sign in to GitLab in the browser (--param instance_url=)",
        }],
    },
    Profile {
        provider: "poe",
        methods: &[Method {
            name: "browser",
            summary: "sign in with Poe in the browser; Poe hands back an API key",
        }],
    },
    Profile {
        provider: "digitalocean",
        methods: &[Method {
            name: "browser",
            summary: "sign in with DigitalOcean in the browser",
        }],
    },
    Profile {
        provider: "chutes",
        methods: &[Method {
            name: "oauth",
            summary: "sign in with your Chutes OAuth app in the browser (--param client_id=)",
        }],
    },
    Profile {
        provider: "minimax",
        methods: MINIMAX_METHODS,
    },
    Profile {
        provider: "minimax-coding-plan",
        methods: MINIMAX_METHODS,
    },
    Profile {
        provider: "minimax-cn",
        methods: MINIMAX_METHODS,
    },
    Profile {
        provider: "minimax-cn-coding-plan",
        methods: MINIMAX_METHODS,
    },
    Profile {
        provider: "anthropic",
        methods: &[
            Method {
                name: "setup-token",
                summary: "paste the token `claude setup-token` prints",
            },
            Method {
                name: "import-claude",
                summary: "reuse Claude Code's login (~/.claude/.credentials.json)",
            },
        ],
    },
];

/// `provider`'s own logins, if it has any.
pub fn profile(provider: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|p| p.provider == provider)
}

/// Every method `provider` takes, its own first.
pub fn methods(provider: &str) -> Vec<&'static Method> {
    profile(provider)
        .map(|p| p.methods.iter())
        .into_iter()
        .flatten()
        .chain(GENERIC)
        .collect()
}

// ---------------------------------------------------------------------------
// Text and tokens
// ---------------------------------------------------------------------------

/// `text` safe to print: control characters (a terminal escape in a
/// server's error) become spaces, runs of space one, and it is cut short.
pub(crate) fn clean(text: &str) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let mut out = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    if out.chars().count() > 160 {
        out = out.chars().take(160).collect::<String>() + "…";
    }
    out
}

/// The claims of a JWT, unverified: llmman reads an account id or an
/// expiry out of a token it was just handed over TLS by its issuer, and
/// authenticates nothing by it.
pub(crate) fn jwt_payload(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// A JWT's `exp`, in milliseconds.
pub(crate) fn jwt_expiry_ms(token: &str) -> Option<u64> {
    jwt_payload(token)?
        .get("exp")?
        .as_f64()
        .filter(|e| *e > 0.0)
        .map(|e| (e * 1000.0) as u64)
}

/// The ChatGPT account a token is for: the `chatgpt_account_id` claim under
/// `https://api.openai.com/auth`. litellm `chatgpt/common_utils.py`
/// `extract_chatgpt_account_id`.
pub(crate) fn chatgpt_account_id(token: &str) -> Option<String> {
    jwt_payload(token)?
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .filter(|a| !a.is_empty())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Where a spent credential may go, and what goes with it
// ---------------------------------------------------------------------------

/// Whether a credential minted by `flow` may be sent to `url`: https, no
/// userinfo, and a host that flow's provider owns. A login's token is
/// worth more than an API key (it is a subscription), and it must not be
/// pointed at a host `llmman.conf` or a catalog update chose.
pub fn destination_allowed(flow: &str, url: &str) -> bool {
    destination_allowed_with(flow, url, &BTreeMap::new())
}

/// [`destination_allowed`] for a login that knows more about where it may go:
/// the GitLab instance or the Snowflake account it was made for
/// ([`OAuth::metadata`]).
pub fn destination_allowed_with(
    flow: &str,
    url: &str,
    metadata: &BTreeMap<String, String>,
) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    let host = url.host_str().unwrap_or("").to_ascii_lowercase();
    // A self-managed GitLab may sit on a port; nothing else may.
    let instance = (flow == FLOW_GITLAB)
        .then(|| metadata.get("instance_url"))
        .flatten()
        .and_then(|u| reqwest::Url::parse(u).ok());
    let on_instance = instance.is_some_and(|i| {
        i.host_str().map(str::to_ascii_lowercase).as_deref() == Some(host.as_str())
            && i.port_or_known_default() == url.port_or_known_default()
    });
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || (url.port().is_some() && !on_instance)
    {
        return false;
    }
    match flow {
        FLOW_CHATGPT => host == "chatgpt.com",
        FLOW_XAI => host == "cli-chat-proxy.grok.com",
        // OpenClaw `runtime-auth.ts` `isTrustedCopilotApiHost`.
        FLOW_COPILOT => {
            host == "copilot-proxy.githubusercontent.com" || host.ends_with(".githubcopilot.com")
        }
        FLOW_CLAUDE | FLOW_SETUP_TOKEN => host == "api.anthropic.com",
        // OpenClaw `extensions/chutes/openclaw.plugin.json` `hosts`.
        FLOW_CHUTES => host == "llm.chutes.ai",
        // The instance the login was made on, or gitlab.com's own hosts.
        FLOW_GITLAB => on_instance || host == "gitlab.com" || host.ends_with(".gitlab.com"),
        // The account the login was made for.
        FLOW_SNOWFLAKE => metadata
            .get("account")
            .and_then(|a| browser::snowflake_account(a))
            .is_some_and(|a| host == format!("{}.snowflakecomputing.com", a.to_ascii_lowercase())),
        // OpenClaw `extensions/minimax/oauth.ts`: the API hosts of the
        // global and China accounts.
        FLOW_MINIMAX => matches!(
            host.as_str(),
            "api.minimax.io" | "api.minimax.cn" | "api.minimaxi.com"
        ),
        _ => false,
    }
}

/// Whether a credential from `flow` is confined to [`destination_allowed`].
/// Every subscription login is; OpenRouter's and Poe's end in an ordinary API
/// key, and DigitalOcean's in a token for the API its catalog entry names,
/// which go wherever a pasted one would.
pub fn fixed_host(flow: &str) -> bool {
    !matches!(flow, FLOW_OPENROUTER | FLOW_POE | FLOW_DIGITALOCEAN)
}

/// A different base URL for a stored login, which the login learned from
/// the provider: Copilot's per-plan API host. Revalidated on every use, so
/// a hand-edited store cannot redirect a token.
pub fn endpoint_for(flow: &str, metadata: &BTreeMap<String, String>) -> Option<String> {
    let endpoint = metadata.get("api_endpoint")?.trim_end_matches('/');
    (flow == FLOW_COPILOT && destination_allowed(flow, endpoint)).then(|| endpoint.to_string())
}

/// Adjusts a resolved credential to the login behind it: a subscription
/// token goes as a bearer whatever the provider's own scheme says, with
/// the headers that login needs. ChatGPT wants its account in
/// `ChatGPT-Account-Id` (litellm `chatgpt/common_utils.py`
/// `get_chatgpt_default_headers`) and an `originator` (OpenClaw sends its
/// own name there for the same backend, `provider-usage.fetch.codex.ts`).
/// An Anthropic subscription token is a bearer with the OAuth beta
/// (litellm `anthropic/common_utils.py` `_oauth_bearer_header`) and
/// `anthropic-dangerous-direct-browser-access` (the same file,
/// `optionally_handle_anthropic_oauth`).
pub fn shape(resolved: &mut Resolved) {
    let anthropic_oauth = |r: &mut Resolved| {
        r.style = Style::Bearer;
        r.headers
            .push(("anthropic-beta".into(), "oauth-2025-04-20".into()));
        r.headers.push((
            "anthropic-dangerous-direct-browser-access".into(),
            "true".into(),
        ));
    };
    match resolved.flow.as_deref() {
        Some(FLOW_CHATGPT) => {
            resolved.style = Style::Bearer;
            if let Some(account) = resolved.metadata.get("account_id").cloned() {
                resolved
                    .headers
                    .push(("ChatGPT-Account-Id".into(), account));
            }
            resolved
                .headers
                .push(("originator".into(), "llmman".into()));
        }
        Some(
            FLOW_COPILOT | FLOW_XAI | FLOW_CHUTES | FLOW_GITLAB | FLOW_SNOWFLAKE | FLOW_MINIMAX,
        ) => resolved.style = Style::Bearer,
        Some(FLOW_CLAUDE | FLOW_SETUP_TOKEN) => anthropic_oauth(resolved),
        _ if !matches!(resolved.kind, "env" | "config" | "request")
            && resolved.secret.expose().starts_with("sk-ant-oat") =>
        {
            anthropic_oauth(resolved)
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

/// Every URL a login or refresh talks to. [`Endpoints::live`] is the real
/// ones; a test points them all at one local server, whose paths differ.
#[derive(Clone, Debug)]
pub struct Endpoints {
    /// auth.openai.com
    pub openai_auth: String,
    /// github.com
    pub github: String,
    /// api.github.com
    pub github_api: String,
    /// auth.x.ai
    pub xai_auth: String,
    /// platform.claude.com
    pub claude_auth: String,
    /// openrouter.ai
    pub openrouter: String,
    /// poe.com
    pub poe_auth: String,
    /// api.poe.com
    pub poe_api: String,
    /// cloud.digitalocean.com
    pub digitalocean: String,
    /// api.chutes.ai
    pub chutes: String,
    /// gitlab.com: the default instance.
    pub gitlab: String,
    /// A Snowflake account's origin, instead of
    /// `https://<account>.snowflakecomputing.com`; set only by a test.
    pub snowflake: Option<String>,
    /// account.minimax.io
    pub minimax_global: String,
    /// account.minimaxi.com
    pub minimax_cn: String,
}

impl Endpoints {
    pub fn live() -> Self {
        Self {
            openai_auth: "https://auth.openai.com".into(),
            github: "https://github.com".into(),
            github_api: "https://api.github.com".into(),
            xai_auth: "https://auth.x.ai".into(),
            claude_auth: "https://platform.claude.com".into(),
            openrouter: "https://openrouter.ai".into(),
            poe_auth: "https://poe.com".into(),
            poe_api: "https://api.poe.com".into(),
            digitalocean: "https://cloud.digitalocean.com".into(),
            chutes: "https://api.chutes.ai".into(),
            gitlab: "https://gitlab.com".into(),
            snowflake: None,
            minimax_global: "https://account.minimax.io".into(),
            minimax_cn: "https://account.minimaxi.com".into(),
        }
    }

    #[cfg(test)]
    pub fn all_at(origin: &str) -> Self {
        let at = || origin.to_string();
        Self {
            openai_auth: at(),
            github: at(),
            github_api: at(),
            xai_auth: at(),
            claude_auth: at(),
            openrouter: at(),
            poe_auth: at(),
            poe_api: at(),
            digitalocean: at(),
            chutes: at(),
            gitlab: at(),
            snowflake: Some(at()),
            minimax_global: at(),
            minimax_cn: at(),
        }
    }
}

/// A transport failure; its text names no URL.
#[derive(Debug)]
pub struct NetError(String);

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NetError {}

/// One response: the status and the body, read to at most [`MAX_REPLY`].
pub struct Reply {
    pub status: u16,
    pub body: String,
}

impl Reply {
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

/// The HTTP a login makes: https only, no redirects (a token endpoint that
/// redirects is not one to follow with a refresh token in hand), a
/// timeout, and no llmman API key — unlike `daemon::client`, whose default
/// headers are for the daemon.
pub struct Http {
    client: OnceLock<Result<reqwest::blocking::Client, String>>,
    allow_http: bool,
    /// A client for a cloud's metadata service: plain http to a loopback
    /// or link-local address (and Google's own name for it) is the one
    /// thing it may do beyond https, and it never goes through a proxy.
    metadata: bool,
}

impl Http {
    pub fn new() -> Self {
        Self {
            client: OnceLock::new(),
            allow_http: false,
            metadata: false,
        }
    }

    /// For a test against a local server; nothing else may send a token
    /// over plain http.
    #[cfg(test)]
    pub fn allowing_http() -> Self {
        Self {
            client: OnceLock::new(),
            allow_http: true,
            metadata: false,
        }
    }

    /// What the cloud credential chains and token services talk to: the
    /// same https-only client, except that under test it may reach a local
    /// server, as [`Http::allowing_http`] does.
    pub fn tokens() -> Self {
        #[cfg(test)]
        {
            Self::allowing_http()
        }
        #[cfg(not(test))]
        {
            Self::new()
        }
    }

    /// For a cloud's metadata service (EC2/ECS, GCE, Azure IMDS and App
    /// Service): https, or plain http to a loopback or link-local address
    /// or `metadata.google.internal` ([`plain_metadata_host`]) — the one
    /// place a credential request is not encrypted, because those
    /// services are only reachable from the machine they describe. No
    /// proxy, and a short timeout: off that cloud the address answers
    /// nothing, and a request must not wait on it.
    pub fn metadata() -> Self {
        Self {
            client: OnceLock::new(),
            allow_http: cfg!(test),
            metadata: true,
        }
    }

    fn client(&self) -> Result<&reqwest::blocking::Client, NetError> {
        self.client
            .get_or_init(|| {
                let mut builder = reqwest::blocking::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .timeout(Duration::from_secs(if self.metadata { 5 } else { 30 }))
                    .user_agent(concat!("llmman/", env!("CARGO_PKG_VERSION")));
                if self.metadata {
                    builder = builder
                        .no_proxy()
                        .connect_timeout(Duration::from_millis(1500));
                }
                for cert in crate::auth::tls_ca().map_err(|e| format!("{e:#}"))? {
                    builder = builder.add_root_certificate(cert);
                }
                builder.build().map_err(|e| e.without_url().to_string())
            })
            .as_ref()
            .map_err(|e| NetError(e.clone()))
    }

    fn check(&self, url: &str) -> Result<(), NetError> {
        if url.starts_with("https://")
            || (self.allow_http && url.starts_with("http://"))
            || (self.metadata && plain_metadata_host(url))
        {
            Ok(())
        } else {
            Err(NetError("refusing to send a login over plain http".into()))
        }
    }

    fn send(&self, req: reqwest::blocking::RequestBuilder) -> Result<Reply, NetError> {
        let resp = req
            .send()
            .map_err(|e| NetError(e.without_url().to_string()))?;
        let status = resp.status().as_u16();
        let mut body = String::new();
        resp.take(MAX_REPLY)
            .read_to_string(&mut body)
            .map_err(|e| NetError(e.to_string()))?;
        Ok(Reply { status, body })
    }

    pub fn post_form(
        &self,
        url: &str,
        form: &[(&str, &str)],
        headers: &[(&str, &str)],
    ) -> Result<Reply, NetError> {
        self.check(url)?;
        let mut req = self
            .client()?
            .post(url)
            .header("accept", "application/json")
            .form(form);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        self.send(req)
    }

    pub fn post_json(
        &self,
        url: &str,
        body: &Value,
        headers: &[(&str, &str)],
    ) -> Result<Reply, NetError> {
        self.check(url)?;
        let mut req = self
            .client()?
            .post(url)
            .header("accept", "application/json")
            .json(body);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        self.send(req)
    }

    pub fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<Reply, NetError> {
        self.check(url)?;
        let mut req = self.client()?.get(url).header("accept", "application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        self.send(req)
    }
}

impl Default for Http {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether `url` is plain http to somewhere only the machine itself can
/// reach: a loopback or link-local IP literal (EC2's `169.254.169.254`,
/// ECS's `169.254.170.2`, EKS's `fd00:ec2::23`, a local App Service
/// endpoint) or Google's `metadata.google.internal` — what the AWS SDK
/// and google-auth accept for these services. A name that merely resolves
/// there is not trusted.
pub fn plain_metadata_host(url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    if url.scheme() != "http" || !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    let host = url.host_str().unwrap_or("").trim_matches(['[', ']']);
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => {
            let first = ip.segments()[0];
            // fe80::/10 link-local, and AWS's fd00:ec2::/32 metadata range.
            ip.is_loopback()
                || first & 0xffc0 == 0xfe80
                || (first == 0xfd00 && ip.segments()[1] == 0x0ec2)
        }
        Err(_) => {
            host.eq_ignore_ascii_case("metadata.google.internal")
                || host.eq_ignore_ascii_case("localhost")
        }
    }
}

// ---------------------------------------------------------------------------
// Token responses and refreshing
// ---------------------------------------------------------------------------

/// A token endpoint's success body.
pub(crate) struct Tokens {
    pub(crate) access: String,
    pub(crate) refresh: Option<String>,
    pub(crate) id_token: Option<String>,
    pub(crate) expires_in: Option<u64>,
}

impl Tokens {
    pub(crate) fn parse(body: &Value) -> Option<Self> {
        let text = |k: &str| {
            body.get(k)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let expires_in = body.get("expires_in").and_then(|e| {
            e.as_u64()
                .or_else(|| e.as_f64().map(|f| f as u64))
                .or_else(|| e.as_str().and_then(|s| s.parse().ok()))
        });
        Some(Self {
            access: text("access_token")?,
            refresh: text("refresh_token"),
            id_token: text("id_token"),
            expires_in: expires_in.filter(|e| *e > 0),
        })
    }

    /// When the access token stops working: the server's `expires_in`,
    /// else the token's own `exp`, else an hour.
    pub(crate) fn expires_ms(&self, now: u64) -> u64 {
        match self.expires_in {
            Some(s) => now.saturating_add(s.saturating_mul(1000)),
            None => jwt_expiry_ms(&self.access).unwrap_or(now + 3_600_000),
        }
    }
}

/// `reply` as tokens, or why not, in the terms a refresh reports.
pub(crate) fn token_reply(reply: Result<Reply, NetError>) -> Result<Tokens, RefreshError> {
    let reply = reply.map_err(|e| RefreshError::Transient(e.0))?;
    let body = reply.json();
    if reply.ok() {
        return Tokens::parse(&body)
            .ok_or_else(|| RefreshError::Transient("the token reply had no access token".into()));
    }
    let code = body
        .get("error")
        .and_then(|e| {
            e.as_str()
                .map(str::to_string)
                .or_else(|| e.get("code")?.as_str().map(str::to_string))
        })
        .map(|c| clean(&c))
        .unwrap_or_else(|| format!("HTTP {}", reply.status));
    match reply.status {
        408 | 425 | 429 | 500..=599 => {
            Err(RefreshError::Transient(format!("HTTP {}", reply.status)))
        }
        _ => Err(RefreshError::Rejected(code)),
    }
}

/// Renews logins against the real endpoints. Cheap to make: nothing is
/// built until a refresh is needed.
pub struct Live {
    http: Http,
    ep: Endpoints,
}

impl Live {
    pub fn new() -> Self {
        Self {
            http: Http::new(),
            ep: Endpoints::live(),
        }
    }

    #[cfg(test)]
    pub fn against(ep: Endpoints) -> Self {
        Self {
            http: Http::allowing_http(),
            ep,
        }
    }
}

impl Default for Live {
    fn default() -> Self {
        Self::new()
    }
}

impl Refresher for Live {
    fn refresh(&self, old: &OAuth) -> Result<OAuth, RefreshError> {
        if let Some(renewed) = browser::refresh(&self.http, &self.ep, old, now_ms()) {
            return renewed;
        }
        let reply = match old.flow.as_str() {
            // OpenClaw `openai-chatgpt-oauth-token.runtime.ts`
            // `refreshOpenAIAccessToken`: a form, at the same token URL
            // the device flow's code is exchanged at.
            FLOW_CHATGPT => self.http.post_form(
                &format!("{}/oauth/token", self.ep.openai_auth),
                &[
                    ("grant_type", "refresh_token"),
                    ("refresh_token", &old.refresh),
                    ("client_id", CHATGPT_CLIENT_ID),
                ],
                &[],
            ),
            // opencode `xai.ts` `refresh`.
            FLOW_XAI => self.http.post_form(
                &format!("{}/oauth2/token", self.ep.xai_auth),
                &[
                    ("grant_type", "refresh_token"),
                    ("refresh_token", &old.refresh),
                    ("client_id", XAI_CLIENT_ID),
                ],
                &[],
            ),
            // OpenClaw `llm/utils/oauth/anthropic.ts` `refreshAnthropicToken`:
            // JSON.
            FLOW_CLAUDE => self.http.post_json(
                &format!("{}/v1/oauth/token", self.ep.claude_auth),
                &json!({
                    "grant_type": "refresh_token",
                    "client_id": CLAUDE_CLIENT_ID,
                    "refresh_token": old.refresh,
                }),
                &[],
            ),
            other => {
                return Err(RefreshError::Rejected(format!(
                    "a {other} login cannot be renewed"
                )))
            }
        };
        let tokens = token_reply(reply)?;
        Ok(renewed(old, tokens, now_ms()))
    }
}

/// `old` with `tokens` laid over it: the rotated refresh token if the
/// server sent one, else the one in hand.
fn renewed(old: &OAuth, tokens: Tokens, now: u64) -> OAuth {
    let mut metadata = old.metadata.clone();
    if old.flow == FLOW_CHATGPT {
        let account = tokens
            .id_token
            .as_deref()
            .and_then(chatgpt_account_id)
            .or_else(|| chatgpt_account_id(&tokens.access));
        if let Some(account) = account {
            metadata.insert("account_id".into(), account);
        }
    }
    OAuth {
        flow: old.flow.clone(),
        expires_ms: tokens.expires_ms(now),
        refresh: tokens.refresh.unwrap_or_else(|| old.refresh.clone()),
        access: tokens.access,
        metadata,
    }
}

// ---------------------------------------------------------------------------
// llmman login --provider
// ---------------------------------------------------------------------------

/// What the command line gave.
#[derive(Default)]
pub struct Options {
    pub method: Option<String>,
    /// A pasted key or token, rather than asking for it.
    pub token: Option<String>,
    /// `--method command`.
    pub command: Option<String>,
    /// `--method token-file`.
    pub file: Option<String>,
    /// `--copy`: for an `import-*`, move the login into llmman's store
    /// instead of reading the other CLI's file each time.
    pub copy: bool,
    /// `--param key=value`, repeatable: what a login needs besides a secret
    /// (a tenant, a profile, a host).
    pub params: Vec<(String, String)>,
}

/// Everything a login touches, so a test can supply its own.
pub struct Ctx<'a> {
    pub store: Store,
    pub ep: Endpoints,
    pub http: Http,
    pub clock: Box<dyn Clock>,
    /// Asks the user for a line; `true` for a secret, which is not echoed.
    pub input: &'a dyn Fn(&str, bool) -> anyhow::Result<String>,
    /// Opens a page in the user's browser, if there is one to open it in.
    pub open: &'a dyn Fn(&str),
}

/// `llmman login --provider <provider>`.
pub fn login(provider: &str, opts: &Options) -> anyhow::Result<()> {
    let store = Store::user().ok_or_else(|| anyhow!("no home directory to keep the login in"))?;
    let ctx = Ctx {
        store,
        ep: Endpoints::live(),
        http: Http::new(),
        clock: Box::new(device::RealClock::start()),
        input: &prompt,
        open: &|url| {
            if std::io::stdout().is_terminal() {
                let _ = crate::oauth::open_browser(url);
            }
        },
    };
    login_with(&ctx, provider, opts)?;
    note_precedence(provider);
    Ok(())
}

/// Says when something outranks the login just stored.
fn note_precedence(provider: &str) {
    let Ok(catalog) = crate::providers::catalog() else {
        return;
    };
    let Some(p) = catalog.get(provider) else {
        return;
    };
    if let Some(var) = p
        .key_env
        .as_deref()
        .filter(|v| crate::providers::key_from_env(v).is_some())
    {
        println!(
            "note: {var} is set in the environment and is used before this login. \
             Unset it to use the login."
        );
    }
}

pub fn login_with(ctx: &Ctx<'_>, provider: &str, opts: &Options) -> anyhow::Result<()> {
    let available = methods(provider);
    let method = opts
        .method
        .as_deref()
        .unwrap_or_else(|| available.first().map_or("key", |m| m.name));
    if !available
        .iter()
        .copied()
        .chain(ADVANCED.iter())
        .any(|m| m.name == method)
    {
        bail!(
            "provider {provider:?} has no login method {method:?}. It takes:\n{}",
            available
                .iter()
                .copied()
                .chain(ADVANCED.iter())
                .map(|m| format!("  {:<14} {}", m.name, m.summary))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    if opts.copy && !method.starts_with("import-") {
        bail!("--copy goes with an import-* method");
    }
    let (credential, what) = match method {
        "device" if provider == FLOW_CHATGPT => (chatgpt_device(ctx)?, "ChatGPT account"),
        "device" if provider == FLOW_COPILOT => (copilot_device(ctx)?, "GitHub Copilot account"),
        "device" if provider == FLOW_XAI => (xai_device(ctx)?, "SuperGrok account"),
        "oauth" if provider == "openrouter" => (openrouter_oauth(ctx)?, "OpenRouter account"),
        "oauth" if provider == "snowflake-cortex" => {
            (browser::snowflake(ctx, opts)?, "Snowflake login")
        }
        "oauth" if provider == "chutes" => (browser::chutes(ctx, opts)?, "Chutes login"),
        "pkce" if provider == "gitlab" => (browser::gitlab(ctx, opts)?, "GitLab login"),
        "browser" if provider == "poe" => (browser::poe(ctx)?, "Poe API key"),
        "browser" if provider == "digitalocean" => {
            (browser::digitalocean(ctx)?, "DigitalOcean login")
        }
        "device" if provider.starts_with("minimax") => {
            (browser::minimax(ctx, provider)?, "MiniMax account")
        }
        "aws-chain" => (cloud::aws_chain(ctx, opts)?, "AWS credential chain"),
        "aws-profile" => (cloud::aws_profile(ctx, opts)?, "AWS profile"),
        "adc" => (
            cloud::gcp_adc(ctx, opts)?,
            "Google Application Default Credentials",
        ),
        "service-account" => (
            cloud::gcp_service_account(ctx, opts)?,
            "Google service account key file",
        ),
        "entra" => (cloud::azure_entra(ctx, opts)?, "Entra ID client secret"),
        "federated" => (
            cloud::azure_federated(ctx, opts)?,
            "Entra ID federated token",
        ),
        "managed-identity" => (
            cloud::azure_managed_identity(ctx, opts)?,
            "Azure managed identity",
        ),
        "cli" => (cloud::azure_cli(ctx, opts)?, "Azure CLI login"),
        "oauth-m2m" => (
            cloud::databricks_m2m(ctx, opts, provider)?,
            "Databricks OAuth M2M credentials",
        ),
        "service-key" => (
            cloud::sap_service_key(ctx, opts, provider)?,
            "SAP AI Core service key",
        ),
        "apikey" => (cloud::ibm_apikey(ctx, opts, provider)?, "IBM Cloud API key"),
        "client-credentials" => (
            cloud::client_credentials(ctx, opts, provider)?,
            "client credentials",
        ),
        "jwt-bearer" => (
            cloud::jwt_bearer(ctx, opts, provider)?,
            "JWT bearer service",
        ),
        "setup-token" => (setup_token(ctx, opts)?, "Claude setup-token"),
        "import-claude" => (import(CliLogin::Claude, opts)?, "Claude Code login"),
        "import-codex" => (import(CliLogin::Codex, opts)?, "Codex login"),
        "key" => (key(ctx, provider, opts)?, "API key"),
        "command" => (command(ctx, opts)?, "key command"),
        "token-file" => (token_file(opts)?, "key file"),
        _ => bail!("login method {method:?} is not available for {provider:?}"),
    };
    ctx.store.put(provider, credential)?;
    println!(
        "Stored the {what} for {provider} in {}",
        ctx.store.path().display()
    );
    Ok(())
}

/// Reads a line from the terminal without echoing it when `secret`.
fn prompt(question: &str, secret: bool) -> anyhow::Result<String> {
    eprint!("{question}");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let line = if secret && std::io::stdin().is_terminal() {
        read_hidden()?
    } else {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        line
    };
    if secret && std::io::stdin().is_terminal() {
        eprintln!();
    }
    Ok(line.trim_end_matches(['\n', '\r']).to_string())
}

#[cfg(unix)]
fn read_hidden() -> anyhow::Result<String> {
    use std::os::fd::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    // SAFETY: termios is plain data; `tcgetattr` fills it for a descriptor
    // that is a terminal (checked by the caller).
    let mut term: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut term) } != 0 {
        bail!("could not read the terminal's settings");
    }
    let saved = term;
    term.c_lflag &= !libc::ECHO;
    // SAFETY: restoring what `tcgetattr` returned.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &term) };
    let mut line = String::new();
    let read = std::io::stdin().lock().read_line(&mut line);
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &saved) };
    read?;
    Ok(line)
}

#[cfg(not(unix))]
fn read_hidden() -> anyhow::Result<String> {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(line)
}

fn secret_arg(ctx: &Ctx<'_>, given: Option<&str>, question: &str) -> anyhow::Result<String> {
    let value = match given {
        Some(v) => v.to_string(),
        None => (ctx.input)(question, true)?,
    };
    let value = value.trim().to_string();
    anyhow::ensure!(!value.is_empty(), "nothing was entered");
    anyhow::ensure!(!value.contains(['\n', '\r']), "a key is one line");
    Ok(value)
}

fn key(ctx: &Ctx<'_>, provider: &str, opts: &Options) -> anyhow::Result<Credential> {
    let key = secret_arg(
        ctx,
        opts.token.as_deref(),
        &format!("API key for {provider}: "),
    )?;
    Ok(Credential::ApiKey { key, flow: None })
}

fn command(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let command = match &opts.command {
        Some(c) => c.clone(),
        None => (ctx.input)("Command that prints the key: ", false)?,
    };
    let command = command.trim().to_string();
    anyhow::ensure!(!command.is_empty(), "no command given");
    // Run once now: better to learn it does not work here than at the
    // first request.
    let source = ExternalSource::Command { command };
    crate::credentials::read_external(&source).context("the command did not yield a key")?;
    Ok(Credential::External { source })
}

/// `path` against the current directory, with `~/` expanded: the daemon
/// reads it from elsewhere. Symlinks stay, as projected token volumes
/// rotate them.
pub(super) fn absolute(path: &str) -> anyhow::Result<String> {
    let path = std::path::absolute(crate::credentials::expand_home(path))?;
    Ok(path.display().to_string())
}

fn token_file(opts: &Options) -> anyhow::Result<Credential> {
    let path = opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .ok_or_else(|| anyhow!("--method token-file needs --file <path>"))?;
    let source = ExternalSource::File {
        path: absolute(path)?,
    };
    crate::credentials::read_external(&source)?;
    Ok(Credential::External { source })
}

/// `claude setup-token`'s output: OpenClaw `provider-auth-token.ts`
/// `validateAnthropicSetupToken`.
fn setup_token(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    println!("Run `claude setup-token` on a computer with Claude Code and paste what it prints.");
    let token = secret_arg(ctx, opts.token.as_deref(), "Setup token: ")?;
    anyhow::ensure!(
        token.starts_with("sk-ant-oat01-"),
        "expected a token starting with sk-ant-oat01- (for an API key, use --method key)"
    );
    anyhow::ensure!(
        token.len() >= 80,
        "the token looks too short; paste the whole thing"
    );
    Ok(Credential::Token {
        token,
        expires_ms: None,
        flow: Some(FLOW_SETUP_TOKEN.into()),
    })
}

fn import(cli: CliLogin, opts: &Options) -> anyhow::Result<Credential> {
    // `--file` is where the other CLI keeps its login, if not where it
    // usually does.
    let path = opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty());
    let found = import::read(cli, path)?;
    if !opts.copy {
        println!(
            "Reusing {}'s login in place: it is read on each request and never changed, so it \
             works while {} keeps it current. Use --copy to move it into llmman instead.",
            found.cli_name, found.cli_name
        );
        return Ok(Credential::External {
            source: ExternalSource::CliLogin {
                cli,
                path: path.map(absolute).transpose()?,
            },
        });
    }
    eprintln!(
        "warning: copying {}'s refresh token. It rotates when used, so {} may be logged out \
         the next time either refreshes; log it in again if so.",
        found.cli_name, found.cli_name
    );
    Ok(match found.refresh {
        Some(refresh) => Credential::OAuth(OAuth {
            flow: found.flow.to_string(),
            access: found.access,
            refresh,
            expires_ms: found.expires_ms.unwrap_or_else(now_ms),
            metadata: found.metadata,
        }),
        None => Credential::Token {
            token: found.access,
            expires_ms: found.expires_ms,
            flow: Some(found.flow.to_string()),
        },
    })
}

/// Prints what the user has to do and opens the browser when it is safe
/// to: the page's host is checked by the caller.
fn show_code(ctx: &Ctx<'_>, url: &str, code: &str) {
    println!();
    println!("To sign in, open {url}");
    println!("and enter the code: {}", clean(code));
    println!("(Never share this code; device codes are a common phishing target.)");
    println!("Waiting for approval…");
    (ctx.open)(url);
}

fn secs(value: Option<&Value>, default: Duration) -> Duration {
    value
        .and_then(Value::as_u64)
        .filter(|s| *s > 0)
        .map_or(default, Duration::from_secs)
}

fn failed(what: &str, reply: &Reply) -> anyhow::Error {
    let body = reply.json();
    let detail = body
        .get("error_description")
        .or_else(|| body.get("error"))
        .and_then(Value::as_str)
        .map(clean)
        .unwrap_or_default();
    anyhow!(
        "{what} failed (HTTP {}){}",
        reply.status,
        if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        }
    )
}

/// ChatGPT: OpenClaw `openai-chatgpt-device-code.ts` `loginOpenAICodexDeviceCode`.
/// Not RFC 8628 proper: a `usercode` request, a `deviceauth/token` poll
/// that is 403/404 until approved and then yields an authorization code
/// and PKCE verifier, and an ordinary code exchange.
fn chatgpt_device(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let base = &ctx.ep.openai_auth;
    let headers = [("originator", "llmman")];
    let started = ctx.http.post_json(
        &format!("{base}/api/accounts/deviceauth/usercode"),
        &json!({ "client_id": CHATGPT_CLIENT_ID }),
        &headers,
    )?;
    if started.status == 404 {
        bail!("ChatGPT device-code login is not enabled for this account or server");
    }
    if !started.ok() {
        return Err(failed("requesting a ChatGPT device code", &started));
    }
    let body = started.json();
    let text = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let (Some(device_auth_id), Some(user_code)) = (
        text("device_auth_id"),
        text("user_code").or_else(|| text("usercode")),
    ) else {
        bail!("ChatGPT's device code reply was missing the code");
    };
    let interval = secs(body.get("interval"), DEFAULT_INTERVAL);
    show_code(ctx, &format!("{base}/codex/device"), user_code);

    let approved = device::run(&*ctx.clock, interval, CHATGPT_DEVICE_TIMEOUT, || {
        match ctx.http.post_json(
            &format!("{base}/api/accounts/deviceauth/token"),
            &json!({ "device_auth_id": device_auth_id, "user_code": user_code }),
            &headers,
        ) {
            Ok(r) if r.ok() => {
                let b = r.json();
                let get = |k: &str| b.get(k).and_then(Value::as_str).map(str::to_string);
                match (get("authorization_code"), get("code_verifier")) {
                    (Some(c), Some(v)) => Poll::Done((c, v)),
                    _ => Poll::Failed("the approval had no exchange code".into()),
                }
            }
            // Not approved yet.
            Ok(r) if r.status == 403 || r.status == 404 => Poll::Pending,
            Ok(r) => Poll::Failed(format!("HTTP {}", r.status)),
            Err(e) => Poll::Failed(e.0),
        }
    })?;
    let (code, verifier) = approved;
    let exchanged = ctx.http.post_form(
        &format!("{base}/oauth/token"),
        &[
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("redirect_uri", &format!("{base}/deviceauth/callback")),
            ("client_id", CHATGPT_CLIENT_ID),
            ("code_verifier", &verifier),
        ],
        &headers,
    )?;
    if !exchanged.ok() {
        return Err(failed("the ChatGPT token exchange", &exchanged));
    }
    let tokens = Tokens::parse(&exchanged.json())
        .ok_or_else(|| anyhow!("the ChatGPT token exchange returned no tokens"))?;
    let refresh = tokens
        .refresh
        .clone()
        .ok_or_else(|| anyhow!("the ChatGPT token exchange returned no refresh token"))?;
    let mut metadata = BTreeMap::new();
    if let Some(account) = tokens
        .id_token
        .as_deref()
        .and_then(chatgpt_account_id)
        .or_else(|| chatgpt_account_id(&tokens.access))
    {
        metadata.insert("account_id".to_string(), account);
    }
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_CHATGPT.into(),
        expires_ms: tokens.expires_ms(now_ms()),
        access: tokens.access,
        refresh,
        metadata,
    }))
}

/// GitHub Copilot: OpenClaw `github-copilot/login.ts`
/// `runGitHubCopilotDeviceFlow` for the device flow, and `runtime-auth.ts`
/// `resolveCopilotRuntimeAuth` for what comes after: the GitHub token is
/// the credential, and `copilot_internal/user` says which API host it
/// goes to. The token does not expire, which is why `expires_ms` is 0.
fn copilot_device(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let github = &ctx.ep.github;
    let started = ctx.http.post_form(
        &format!("{github}/login/device/code"),
        &[("client_id", COPILOT_CLIENT_ID), ("scope", "read:user")],
        &[],
    )?;
    if !started.ok() {
        return Err(failed("requesting a GitHub device code", &started));
    }
    let body = started.json();
    let text = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let (Some(device_code), Some(user_code), Some(verification)) = (
        text("device_code"),
        text("user_code"),
        text("verification_uri"),
    ) else {
        bail!("GitHub's device code reply was missing a field");
    };
    // OpenClaw `normalizeGitHubDeviceVerificationUrl`: only github.com's
    // own page, which is then reconstructed rather than echoed.
    let shown = reqwest::Url::parse(verification)
        .ok()
        .filter(|u| {
            let local = ctx.http.allow_http && u.scheme() == "http";
            (u.scheme() == "https" || local)
                && u.path() == "/login/device"
                && u.username().is_empty()
        })
        .map(|u| format!("{github}{}", u.path()))
        .ok_or_else(|| anyhow!("GitHub returned an unexpected verification URL"))?;
    let interval = secs(body.get("interval"), DEFAULT_INTERVAL);
    let expires_in = secs(body.get("expires_in"), Duration::from_secs(900));
    show_code(ctx, &shown, user_code);

    let token = device::run(&*ctx.clock, interval, expires_in, || {
        match ctx.http.post_form(
            &format!("{github}/login/oauth/access_token"),
            &[
                ("client_id", COPILOT_CLIENT_ID),
                ("device_code", device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ],
            &[],
        ) {
            Ok(r) => device::rfc8628(r.ok(), &r.json(), |b| {
                b.get("access_token")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| "the reply had no access token".to_string())
            }),
            Err(e) => Poll::Failed(e.0),
        }
    })?;

    let mut metadata = BTreeMap::new();
    let user = ctx.http.get(
        &format!("{}/copilot_internal/user", ctx.ep.github_api),
        &[
            ("authorization", &format!("Bearer {token}")),
            ("x-github-api-version", "2025-04-01"),
        ],
    );
    // Only an explicit "no" blocks the login (opencode `copilotEntitlementError`):
    // a failed lookup is not a denial.
    if let Ok(reply) = user.as_ref().map(|r| (r.ok(), r.json())) {
        let (ok, body) = reply;
        if ok {
            if body.get("chat_enabled").and_then(Value::as_bool) == Some(false) {
                bail!(
                    "this GitHub account has no GitHub Copilot access; it needs a Copilot \
                     subscription or a seat from an organization"
                );
            }
            if let Some(api) = body
                .pointer("/endpoints/api")
                .and_then(Value::as_str)
                .map(|a| a.trim_end_matches('/'))
                .filter(|a| destination_allowed(FLOW_COPILOT, a))
            {
                metadata.insert("api_endpoint".to_string(), api.to_string());
            }
        }
    }
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_COPILOT.into(),
        access: token,
        refresh: String::new(),
        expires_ms: 0,
        metadata,
    }))
}

/// xAI: opencode `xai.ts` `device`/`poll`; the same values as OpenClaw's
/// `xai-oauth.ts`, which finds them through discovery.
fn xai_device(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let base = format!("{}/oauth2", ctx.ep.xai_auth);
    let started = ctx.http.post_form(
        &format!("{base}/device/code"),
        &[("client_id", XAI_CLIENT_ID), ("scope", XAI_SCOPE)],
        &[],
    )?;
    if !started.ok() {
        return Err(failed("requesting an xAI device code", &started));
    }
    let body = started.json();
    let text = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let (Some(device_code), Some(user_code)) = (text("device_code"), text("user_code")) else {
        bail!("xAI's device code reply was missing a field");
    };
    let url = text("verification_uri_complete")
        .or_else(|| text("verification_uri"))
        .ok_or_else(|| anyhow!("xAI's device code reply had no verification URL"))?;
    let trusted = reqwest::Url::parse(url).ok().is_some_and(|u| {
        let host = u.host_str().unwrap_or("");
        let local = ctx.http.allow_http && u.scheme() == "http";
        (u.scheme() == "https" || local)
            && (host == "x.ai" || host.ends_with(".x.ai") || local)
            && u.username().is_empty()
    });
    anyhow::ensure!(trusted, "xAI returned an unexpected verification URL");
    let interval = secs(body.get("interval"), DEFAULT_INTERVAL);
    let expires_in = secs(body.get("expires_in"), Duration::from_secs(300));
    show_code(ctx, url, user_code);

    let tokens = device::run(&*ctx.clock, interval, expires_in, || {
        match ctx.http.post_form(
            &format!("{base}/token"),
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", XAI_CLIENT_ID),
                ("device_code", device_code),
            ],
            &[],
        ) {
            Ok(r) => device::rfc8628(r.ok(), &r.json(), |b| {
                Tokens::parse(b).ok_or_else(|| "the reply had no access token".to_string())
            }),
            Err(e) => Poll::Failed(e.0),
        }
    })?;
    let refresh = tokens
        .refresh
        .clone()
        .ok_or_else(|| anyhow!("xAI's reply had no refresh token; sign in again"))?;
    Ok(Credential::OAuth(OAuth {
        flow: FLOW_XAI.into(),
        expires_ms: tokens.expires_ms(now_ms()),
        access: tokens.access,
        refresh,
        metadata: BTreeMap::new(),
    }))
}

/// OpenRouter: OpenClaw `openrouter/oauth.ts` `loginOpenRouterOAuth`.
/// The browser goes to `openrouter.ai/auth` with a `callback_url` and a
/// PKCE challenge; the `code` it comes back with is exchanged at
/// `api/v1/auth/keys` for an API key, which is what is stored.
fn openrouter_oauth(ctx: &Ctx<'_>) -> anyhow::Result<Credential> {
    let pkce = pkce::Pkce::generate()?;
    let state = pkce::random_token(32)?;
    let listener = std::net::TcpListener::bind(("127.0.0.1", openrouter_port())).ok();
    let port = listener
        .as_ref()
        .and_then(|l| l.local_addr().ok())
        .map_or(OPENROUTER_CALLBACK_PORT, |a| a.port());
    let mut callback = reqwest::Url::parse(&format!(
        "http://localhost:{port}{OPENROUTER_CALLBACK_PATH}"
    ))?;
    callback.query_pairs_mut().append_pair("state", &state);
    let mut url = reqwest::Url::parse(&format!("{}/auth", ctx.ep.openrouter))?;
    url.query_pairs_mut()
        .append_pair("callback_url", callback.as_str())
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256");
    println!();
    println!("Open this page to sign in to OpenRouter:\n{url}");
    (ctx.open)(url.as_str());
    let code = match &listener {
        Some(listener) => {
            println!("Waiting for the browser to return to this computer…");
            pkce::wait_for_callback(listener, OPENROUTER_CALLBACK_PATH, &state, BROWSER_TIMEOUT)?
        }
        None => {
            println!("Port {OPENROUTER_CALLBACK_PORT} is busy, so the browser cannot return here.");
            let pasted = (ctx.input)(
                "After signing in, paste the URL the browser ends up at: ",
                false,
            )?;
            pkce::code_from_callback(&pasted, &state)?
        }
    };
    let exchanged = ctx.http.post_json(
        &format!("{}/api/v1/auth/keys", ctx.ep.openrouter),
        &json!({
            "code": code,
            "code_verifier": pkce.verifier,
            "code_challenge_method": "S256",
        }),
        &[],
    )?;
    if !exchanged.ok() {
        return Err(failed("OpenRouter's key exchange", &exchanged));
    }
    let key = exchanged
        .json()
        .get("key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("OpenRouter's key exchange returned no API key"))?;
    Ok(Credential::ApiKey {
        key,
        flow: Some(FLOW_OPENROUTER.into()),
    })
}

/// The port the OpenRouter callback listens on. Fixed, as OpenRouter's
/// documented `localhost:3000` callback is; a test asks for any.
fn openrouter_port() -> u16 {
    #[cfg(test)]
    {
        0
    }
    #[cfg(not(test))]
    {
        OPENROUTER_CALLBACK_PORT
    }
}

/// `llmman logout --provider <id>`: removes the stored login, and says
/// what still applies.
pub fn logout(provider: &str) -> anyhow::Result<()> {
    let store = Store::user().ok_or_else(|| anyhow!("no home directory to keep logins in"))?;
    if store.remove(provider)? {
        println!("Removed the stored login for {provider}");
    } else {
        println!("No stored login for {provider}");
    }
    if let Ok(catalog) = crate::providers::catalog() {
        if let Some(p) = catalog.get(provider) {
            if let Some(var) = p
                .key_env
                .as_deref()
                .filter(|v| crate::providers::key_from_env(v).is_some())
            {
                println!("note: {var} is still set in the environment");
            }
        }
    }
    if crate::config::provider_api_key(provider).is_some() {
        println!(
            "note: llmman.conf still has an api_key for {provider}; \
             `llmman config unset providers.{provider}.api_key` removes it"
        );
    }
    Ok(())
}
