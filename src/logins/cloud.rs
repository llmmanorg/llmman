//! The logins for the cloud credential chains and token services.
//!
//! None is a browser round trip: each either selects a chain that finds its
//! credentials where the cloud's own tools leave them, or stores a token
//! service's secret after trying it once. What is stored is a
//! [`Credential::Chain`], never a token ([`crate::chains`]).

use anyhow::{anyhow, bail, Context};
use serde_json::Value;

use super::{Ctx, Options};
use crate::aws::AwsChain;
use crate::azure::AzureChain;
use crate::chains::Chain;
use crate::credentials::{now_ms, Credential};
use crate::gcp::GcpChain;
use crate::tokensvc::{ClientAuth, Ctx as TokenCtx, Grant, TokenService};

/// `--param key=value`.
pub(super) fn param<'a>(opts: &'a Options, key: &str) -> Option<&'a str> {
    opts.params
        .iter()
        .rev()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
}

/// The value of `--param key=...`, or what the user types at `question`.
fn ask(
    ctx: &Ctx<'_>,
    opts: &Options,
    key: &str,
    question: &str,
    secret: bool,
) -> anyhow::Result<String> {
    let value = match param(opts, key) {
        Some(v) => v.to_string(),
        None => (ctx.input)(question, secret)?.trim().to_string(),
    };
    anyhow::ensure!(!value.is_empty(), "{key} is required");
    anyhow::ensure!(!value.contains(['\n', '\r']), "{key} is one line");
    Ok(value)
}

/// A secret: `--token`, else typed without echo.
fn ask_secret(ctx: &Ctx<'_>, opts: &Options, question: &str) -> anyhow::Result<String> {
    super::secret_arg(ctx, opts.token.as_deref(), question)
}

fn chain(chain: Chain) -> anyhow::Result<Credential> {
    chain
        .validate()
        .map_err(|e| anyhow!("that selection is not usable: {e}"))?;
    Ok(Credential::Chain(chain))
}

// ---------------------------------------------------------------------------
// AWS
// ---------------------------------------------------------------------------

/// Whether the chain finds credentials now, said as a note: a chain that
/// cannot (an SSO login that has lapsed, a role that is not yet assumable)
/// is stored all the same — the user may be about to fix it — but is not
/// left to be a surprise.
fn note_aws(selection: &AwsChain) {
    // Tests never read the real AWS configuration.
    if cfg!(test) {
        return;
    }
    let rt = crate::aws::Runtime::live();
    match selection.credentials(&rt, now_ms()) {
        Ok(_) => println!(
            "AWS credentials found for region {}.",
            selection.region(&rt)
        ),
        Err(e) => {
            eprintln!("warning: {e:#}\nThe selection is stored; requests fail until it works.")
        }
    }
}

/// The standard chain, as it is on this machine: environment, shared files
/// (the `AWS_PROFILE` or `default` profile), web identity, container
/// credentials, EC2 instance metadata. `--param region=` pins a region.
pub(super) fn aws_chain(_ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let selection = AwsChain {
        region: param(opts, "region").map(str::to_string),
        forced: true,
        ..AwsChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    note_aws(&selection);
    chain(Chain::Aws(selection))
}

/// A named profile of `~/.aws/config` and `~/.aws/credentials`: static
/// keys, `role_arn` + `source_profile`, `credential_process`, or SSO.
pub(super) fn aws_profile(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let profile = ask(ctx, opts, "profile", "AWS profile name: ", false)?;
    let selection = AwsChain {
        profile: Some(profile),
        region: param(opts, "region").map(str::to_string),
        forced: true,
        ..AwsChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    note_aws(&selection);
    chain(Chain::Aws(selection))
}

// ---------------------------------------------------------------------------
// Google
// ---------------------------------------------------------------------------

fn gcp_selection(opts: &Options, file: Option<String>) -> anyhow::Result<GcpChain> {
    let selection = GcpChain {
        credentials_file: file,
        project: param(opts, "project").map(str::to_string),
        location: param(opts, "location").map(str::to_string),
        forced: true,
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    Ok(selection)
}

fn note_gcp(selection: &GcpChain) -> anyhow::Result<()> {
    // Tests never read the real gcloud configuration.
    if cfg!(test) {
        return Ok(());
    }
    let rt = crate::gcp::Runtime::live();
    println!("Using {}.", selection.describe_source(&rt)?);
    match selection.project(&rt) {
        Some(project) => println!("Project {project}, location {}.", selection.location(&rt)),
        None => println!(
            "No project is set: give --param project=<id>, or set GOOGLE_VERTEX_PROJECT, \
             for the provider to be offered."
        ),
    }
    Ok(())
}

/// Application Default Credentials: `GOOGLE_APPLICATION_CREDENTIALS`, the
/// file `gcloud auth application-default login` wrote, or the metadata
/// server. `--param project=` and `--param location=` are kept with it.
pub(super) fn gcp_adc(_ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let selection = gcp_selection(opts, None)?;
    note_gcp(&selection)?;
    chain(Chain::Gcp(selection))
}

/// A service-account key file (`--file`): its path is kept, not its key, and
/// it is read — and held to the owner-only rule — on each use.
pub(super) fn gcp_service_account(_ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let file = opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .ok_or_else(|| anyhow!("--method service-account needs --file <key.json>"))?;
    let path = crate::credentials::expand_home(file);
    let absolute = std::fs::canonicalize(&path)
        .with_context(|| format!("find the service account key {}", path.display()))?;
    let selection = gcp_selection(opts, Some(absolute.to_string_lossy().into_owned()))?;
    let text = std::fs::read_to_string(&absolute)
        .with_context(|| format!("read {}", absolute.display()))?;
    let json: Value = serde_json::from_str(&text)
        .map_err(|_| anyhow!("{} is not a JSON key file", absolute.display()))?;
    anyhow::ensure!(
        json.get("type").and_then(Value::as_str) == Some("service_account"),
        "{} is not a service account key (its type is not service_account)",
        absolute.display()
    );
    // Reads the file's type and fields; sends nothing.
    selection
        .describe_source(&crate::gcp::Runtime::live())
        .context("the service account key")?;
    note_gcp(&selection)?;
    chain(Chain::Gcp(selection))
}

// ---------------------------------------------------------------------------
// Azure
// ---------------------------------------------------------------------------

/// An Entra app registration's client secret. The secret is stored in the
/// credential file, and the token endpoint is asked once now.
pub(super) fn azure_entra(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let tenant = ask(ctx, opts, "tenant_id", "Tenant id (or domain): ", false)?;
    let client = ask(ctx, opts, "client_id", "Application (client) id: ", false)?;
    let secret = ask_secret(ctx, opts, "Client secret: ")?;
    let selection = AzureChain {
        tenant_id: Some(tenant),
        client_id: Some(client),
        client_secret: Some(secret),
        scope: param(opts, "scope").map(str::to_string),
        authority_host: param(opts, "authority_host").map(str::to_string),
        forced: true,
        ..AzureChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    verify_azure(ctx, &selection)?;
    chain(Chain::Azure(selection))
}

/// A federated token file, as AKS workload identity projects one
/// (`--file`).
pub(super) fn azure_federated(ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let file = opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .ok_or_else(|| anyhow!("--method federated needs --file <token file>"))?;
    let selection = AzureChain {
        tenant_id: Some(ask(
            ctx,
            opts,
            "tenant_id",
            "Tenant id (or domain): ",
            false,
        )?),
        client_id: Some(ask(
            ctx,
            opts,
            "client_id",
            "Application (client) id: ",
            false,
        )?),
        federated_token_file: Some(super::absolute(file)?),
        scope: param(opts, "scope").map(str::to_string),
        authority_host: param(opts, "authority_host").map(str::to_string),
        forced: true,
        ..AzureChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    verify_azure(ctx, &selection)?;
    chain(Chain::Azure(selection))
}

/// The managed identity of the VM, App Service or container this runs on;
/// `--param client_id=` names a user-assigned one.
pub(super) fn azure_managed_identity(_ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let selection = AzureChain {
        client_id: param(opts, "client_id").map(str::to_string),
        managed_identity: Some(true),
        scope: param(opts, "scope").map(str::to_string),
        forced: true,
        ..AzureChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    chain(Chain::Azure(selection))
}

/// `az account get-access-token`: whoever is signed in to the Azure CLI.
pub(super) fn azure_cli(_ctx: &Ctx<'_>, opts: &Options) -> anyhow::Result<Credential> {
    let selection = AzureChain {
        cli: Some(true),
        scope: param(opts, "scope").map(str::to_string),
        forced: true,
        ..AzureChain::default()
    };
    selection.validate().map_err(|e| anyhow!(e))?;
    chain(Chain::Azure(selection))
}

/// Asks Entra for a token with `selection` once, so a wrong secret is
/// reported by the login. The result is thrown away.
fn verify_azure(ctx: &Ctx<'_>, selection: &AzureChain) -> anyhow::Result<()> {
    let rt = crate::azure::Runtime::live();
    selection
        .resolve(&rt, &ctx.http, now_ms())
        .context("Entra ID did not issue a token for those credentials")?
        .ok_or_else(|| anyhow!("those credentials are not complete"))?;
    println!("Entra ID issued a token.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Token services
// ---------------------------------------------------------------------------

/// `service` with the provider's built-in profile under it, as a request
/// would see it, and the variables a login's `--param`s stand in for.
fn merged(provider: &str, service: &TokenService) -> TokenService {
    match crate::chains::builtin(provider) {
        Some(Chain::Token(base)) => base.overlaid(service),
        _ => service.clone(),
    }
}

/// Tries `service` (over the provider's profile) against its token
/// endpoint, then stores it: a wrong secret is the login's error.
fn verified(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
    service: TokenService,
) -> anyhow::Result<Credential> {
    let full = merged(provider, &service);
    full.validate_with(ctx.http.allow_http)
        .map_err(|e| anyhow!("that is not a usable token service: {e}"))?;
    let env = |name: &str| {
        // `--param host=...` stands for the provider's host variable.
        match (name, param(opts, "host")) {
            ("DATABRICKS_HOST", Some(host)) => Some(host.to_string()),
            _ => crate::providers::key_from_env(name)
                .or_else(|| crate::config::provider_var(provider, name)),
        }
    };
    let token_ctx = TokenCtx {
        env: &env,
        http: &ctx.http,
        now: now_ms(),
    };
    anyhow::ensure!(
        full.configured(&token_ctx),
        "the token service is not complete: a variable it needs is not set \
         (DATABRICKS_HOST for Databricks, AICORE_AUTH_URL for SAP AI Core, ...)"
    );
    full.mint(&token_ctx)
        .context("the token endpoint did not issue a token for those credentials")?;
    println!("The token endpoint issued a token.");
    // Validated above, against this login's own client.
    Ok(Credential::Chain(Chain::Token(service)))
}

/// OAuth 2.0 client credentials at a token URL of the user's choosing, for
/// any provider: `--param token_url=`, `client_id=`, optionally `scope=`,
/// `audience=`, `client_auth=basic|body`; the secret is asked for.
pub(super) fn client_credentials(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
) -> anyhow::Result<Credential> {
    let service = TokenService {
        grant: Grant::ClientCredentials,
        token_url: Some(ask(ctx, opts, "token_url", "Token URL: ", false)?),
        client_id: Some(ask(ctx, opts, "client_id", "Client id: ", false)?),
        client_secret: Some(ask_secret(ctx, opts, "Client secret: ")?),
        client_auth: match param(opts, "client_auth") {
            None | Some("body") => Some(ClientAuth::Body),
            Some("basic") => Some(ClientAuth::Basic),
            Some(other) => bail!("client_auth is `body` or `basic`, not {other:?}"),
        },
        scope: param(opts, "scope").map(str::to_string),
        audience: param(opts, "audience").map(str::to_string),
        resource: param(opts, "resource").map(str::to_string),
        ..TokenService::default()
    };
    verified(ctx, opts, provider, service)
}

/// A signed JWT in place of a secret (RFC 7523): `--param token_url=`,
/// `issuer=` (or `client_id=`), optionally `subject=`, `audience=`, `scope=`,
/// `key_id=`, `algorithm=`; `--file` is the PEM private key, whose path (not
/// its content) is stored.
pub(super) fn jwt_bearer(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
) -> anyhow::Result<Credential> {
    let key = opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .ok_or_else(|| anyhow!("--method jwt-bearer needs --file <private key PEM>"))?;
    let absolute = std::fs::canonicalize(crate::credentials::expand_home(key))
        .with_context(|| format!("find the signing key {key}"))?;
    let service = TokenService {
        grant: Grant::JwtBearer,
        token_url: Some(ask(ctx, opts, "token_url", "Token URL: ", false)?),
        issuer: Some(ask(ctx, opts, "issuer", "Issuer (iss): ", false)?),
        subject: param(opts, "subject").map(str::to_string),
        audience: param(opts, "audience").map(str::to_string),
        scope: param(opts, "scope").map(str::to_string),
        key_id: param(opts, "key_id").map(str::to_string),
        algorithm: param(opts, "algorithm").map(str::to_string),
        key_file: Some(absolute.to_string_lossy().into_owned()),
        ..TokenService::default()
    };
    verified(ctx, opts, provider, service)
}

/// watsonx: an IBM Cloud API key, exchanged for IAM bearer tokens.
pub(super) fn ibm_apikey(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
) -> anyhow::Result<Credential> {
    let service = TokenService {
        grant: Grant::IbmApikey,
        client_secret: Some(ask_secret(ctx, opts, "IBM Cloud API key: ")?),
        token_url: param(opts, "token_url").map(str::to_string),
        ..TokenService::default()
    };
    verified(ctx, opts, provider, service)
}

/// SAP AI Core: the service key (`--file`, or pasted), whose `clientid`,
/// `clientsecret` and `url` are the client credentials and the UAA the token
/// comes from. litellm `llms/sap/credentials.py` reads the same fields,
/// at the top level or under `credentials`.
pub(super) fn sap_service_key(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
) -> anyhow::Result<Credential> {
    let text = match opts
        .file
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
    {
        Some(path) => std::fs::read_to_string(crate::credentials::expand_home(path))
            .with_context(|| format!("read the service key {path}"))?,
        None => (ctx.input)("Paste the service key JSON: ", true)?,
    };
    let key: Value =
        serde_json::from_str(text.trim()).map_err(|_| anyhow!("the service key is not JSON"))?;
    let field = |name: &str| {
        key.get(name)
            .or_else(|| key.get("credentials").and_then(|c| c.get(name)))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let (Some(id), Some(secret), Some(url)) =
        (field("clientid"), field("clientsecret"), field("url"))
    else {
        if field("certurl").is_some() {
            bail!(
                "this service key authenticates with a client certificate (certurl), which \
                 llmman does not support; create a service key with a client secret"
            );
        }
        bail!("the service key has no clientid, clientsecret and url");
    };
    // litellm `AUTH_ENDPOINT_SUFFIX`.
    let url = url.trim_end_matches('/');
    let token_url = if url.ends_with("/oauth/token") {
        url.to_string()
    } else {
        format!("{url}/oauth/token")
    };
    let service = TokenService {
        grant: Grant::ClientCredentials,
        token_url: Some(token_url),
        client_id: Some(id),
        client_secret: Some(secret),
        client_auth: Some(ClientAuth::Basic),
        ..TokenService::default()
    };
    verified(ctx, opts, provider, service)
}

/// Databricks: a service principal's OAuth M2M credentials. The workspace
/// (`DATABRICKS_HOST` or `--param host=`) is only used to try them; the URL
/// is the provider's own template.
pub(super) fn databricks_m2m(
    ctx: &Ctx<'_>,
    opts: &Options,
    provider: &str,
) -> anyhow::Result<Credential> {
    let service = TokenService {
        client_id: Some(ask(
            ctx,
            opts,
            "client_id",
            "Service principal client id: ",
            false,
        )?),
        client_secret: Some(ask_secret(ctx, opts, "Client secret: ")?),
        ..TokenService::default()
    };
    let credential = verified(ctx, opts, provider, service)?;
    println!("Set DATABRICKS_HOST (or `llmman config set providers.databricks.vars.DATABRICKS_HOST <host>`) for the provider to be offered.");
    Ok(credential)
}
