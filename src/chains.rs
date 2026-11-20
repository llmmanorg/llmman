//! The credential chains that stand behind a provider instead of a key, and
//! the one call that turns a chain into a credential.
//!
//! A [`Chain`] comes from the provider's built-in profile ([`builtin`]), then
//! a table under `[providers.<id>]` in `llmman.conf`, then a stored
//! `llmman login`, later over earlier. It is the last place a request
//! looks: environment, stored and configured keys and key commands come
//! first ([`crate::providers::Provider::credential`]).

use serde::{Deserialize, Serialize};

use crate::aws::AwsChain;
use crate::azure::AzureChain;
use crate::credentials::{now_ms, Credential, Resolved, Status};
use crate::gcp::GcpChain;
use crate::tokensvc::{ClientAuth, Ctx, Grant, TokenService};

/// One credential chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "chain", rename_all = "snake_case")]
pub enum Chain {
    /// AWS SigV4 over the SDK's credential chain ([`crate::aws`]).
    Aws(AwsChain),
    /// Google Application Default Credentials ([`crate::gcp`]).
    Gcp(GcpChain),
    /// Microsoft Entra ID ([`crate::azure`]).
    Azure(AzureChain),
    /// A token endpoint: OAuth 2.0 client credentials, a JWT bearer, IBM IAM
    /// ([`crate::tokensvc`]).
    Token(TokenService),
}

impl Chain {
    /// The credential kind `llmman providers` shows.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Aws(_) => "aws_sigv4",
            Self::Gcp(_) => "adc",
            Self::Azure(_) => "entra",
            Self::Token(t) => t.grant.kind(),
        }
    }

    /// `self` marked as asked for, so it is tried without a hint: a stored
    /// selection is the user's own word that the chain is there.
    fn forced(mut self) -> Self {
        match &mut self {
            Self::Aws(c) => c.forced = true,
            Self::Gcp(c) => c.forced = true,
            Self::Azure(c) => c.forced = true,
            Self::Token(_) => {}
        }
        self
    }

    /// What is wrong with this chain as written.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Aws(c) => c.validate(),
            Self::Gcp(c) => c.validate(),
            Self::Azure(c) => c.validate(),
            Self::Token(t) => t.validate(),
        }
    }

    /// `self` with `over` laid on it: two chains of one kind (and grant)
    /// merge field by field, anything else is replaced.
    pub(crate) fn overlaid(self, over: Chain) -> Chain {
        match (self, over) {
            (Self::Aws(a), Self::Aws(b)) => Self::Aws(a.overlaid(&b)),
            (Self::Gcp(a), Self::Gcp(b)) => Self::Gcp(a.overlaid(&b)),
            (Self::Azure(a), Self::Azure(b)) => Self::Azure(a.overlaid(&b)),
            (Self::Token(a), Self::Token(b)) if a.grant == b.grant => Self::Token(a.overlaid(&b)),
            (_, over) => over,
        }
    }

    /// What `llmman providers` reports for this chain on provider `id`: the
    /// kind, and `ready` when something says it can work. No network, no
    /// process, no secret.
    pub fn status(&self, id: &str) -> Option<Status> {
        match self {
            Self::Aws(c) => c.status(&crate::aws::Runtime::live()),
            Self::Gcp(c) => c.status(&crate::gcp::Runtime::live()),
            Self::Azure(c) => c.status(&crate::azure::Runtime::live()),
            Self::Token(t) => t.status(&Ctx {
                env: &vars_of(id),
                http: crate::tokensvc::http(),
                now: now_ms(),
            }),
        }
    }

    /// The credential for a request to `base_url`, or `Ok(None)` when the
    /// chain has nothing to go on here. Blocking: call from
    /// `spawn_blocking`.
    pub fn resolve(&self, id: &str, base_url: &str) -> anyhow::Result<Option<Resolved>> {
        let now = now_ms();
        match self {
            Self::Aws(c) => c.resolve(
                &crate::aws::Runtime::live(),
                base_url,
                crate::aws::SERVICE_RUNTIME,
                now,
            ),
            Self::Gcp(c) => c.resolve(&crate::gcp::Runtime::live(), crate::tokensvc::http(), now),
            Self::Azure(c) => {
                c.resolve(&crate::azure::Runtime::live(), crate::tokensvc::http(), now)
            }
            Self::Token(t) => {
                let env = vars_of(id);
                let ctx = Ctx {
                    env: &env,
                    http: crate::tokensvc::http(),
                    now,
                };
                if !t.configured(&ctx) {
                    return Ok(None);
                }
                t.resolve(&ctx).map(Some)
            }
        }
    }
}

/// A variable's value: the environment, then `[providers.<id>] vars`.
fn vars_of(id: &str) -> impl Fn(&str) -> Option<String> + '_ {
    move |name| {
        crate::providers::key_from_env(name).or_else(|| crate::config::provider_var(id, name))
    }
}

// ---------------------------------------------------------------------------
// Which provider has which
// ---------------------------------------------------------------------------

/// The ids whose chain is built in.
pub const BEDROCK: &str = "amazon-bedrock";
pub const BEDROCK_MANTLE: &str = "amazon-bedrock-mantle";
pub const VERTEX: &str = "google-vertex";

/// A provider's own profile.
///
/// * `amazon-bedrock`, `amazon-bedrock-mantle`: SigV4, each with its
///   signing name ([`crate::aws`]).
/// * `google-vertex`: ADC ([`crate::gcp`]).
/// * `azure`: Entra ID ([`crate::azure`]).
/// * `databricks`: OAuth M2M — `client_credentials` at
///   `https://<DATABRICKS_HOST>/oidc/v1/token`, Basic client authentication,
///   `scope=all-apis`, from `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`
///   (litellm `llms/databricks/common_utils.py`). A personal access token,
///   `DATABRICKS_TOKEN`, is the provider's key and goes first.
/// * `watsonx`: IBM IAM — `WATSONX_API_KEY` exchanged at
///   `https://iam.cloud.ibm.com/identity/token` (litellm
///   `llms/watsonx/common_utils.py`).
/// * `sap-ai-core`: client credentials from a service key's `clientid`,
///   `clientsecret` and `url` — `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`,
///   `AICORE_AUTH_URL` (the service key's `url`; `/oauth/token` is appended),
///   as litellm `llms/sap/credentials.py` names them.
pub fn builtin(id: &str) -> Option<Chain> {
    match id {
        BEDROCK => Some(Chain::Aws(AwsChain {
            service: Some(crate::aws::SERVICE_RUNTIME.into()),
            ..AwsChain::default()
        })),
        BEDROCK_MANTLE => Some(Chain::Aws(AwsChain {
            service: Some(crate::aws::SERVICE_MANTLE.into()),
            ..AwsChain::default()
        })),
        VERTEX => Some(Chain::Gcp(GcpChain::default())),
        "azure" => Some(Chain::Azure(AzureChain::default())),
        "databricks" => Some(Chain::Token(TokenService {
            grant: Grant::ClientCredentials,
            token_url: Some("https://${DATABRICKS_HOST}/oidc/v1/token".into()),
            client_id_env: Some("DATABRICKS_CLIENT_ID".into()),
            client_secret_env: Some("DATABRICKS_CLIENT_SECRET".into()),
            client_auth: Some(ClientAuth::Basic),
            scope: Some("all-apis".into()),
            ..TokenService::default()
        })),
        "watsonx" => Some(Chain::Token(TokenService {
            grant: Grant::IbmApikey,
            client_secret_env: Some("WATSONX_API_KEY".into()),
            ..TokenService::default()
        })),
        "sap-ai-core" => Some(Chain::Token(TokenService {
            grant: Grant::ClientCredentials,
            token_url: Some("${AICORE_AUTH_URL}/oauth/token".into()),
            client_id_env: Some("AICORE_CLIENT_ID".into()),
            client_secret_env: Some("AICORE_CLIENT_SECRET".into()),
            client_auth: Some(ClientAuth::Basic),
            ..TokenService::default()
        })),
        _ => None,
    }
}

/// The chain provider `id` authenticates with, if it has one: its built-in
/// profile, `llmman.conf` over it, a stored `llmman login` over that.
pub fn chain_for(id: &str) -> Option<Chain> {
    let mut chain = builtin(id);
    if let Some(conf) = crate::config::provider_chain(id) {
        chain = Some(match chain {
            Some(base) => base.overlaid(conf),
            None => conf,
        });
    }
    if let Some(Credential::Chain(stored)) =
        crate::credentials::Store::user().and_then(|s| s.get(id))
    {
        let stored = stored.forced();
        chain = Some(match chain {
            Some(base) => base.overlaid(stored),
            None => stored,
        });
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_profiles_are_the_ones_the_references_name() {
        let Some(Chain::Aws(runtime)) = builtin(BEDROCK) else {
            panic!("bedrock has an AWS chain");
        };
        assert_eq!(runtime.service.as_deref(), Some("bedrock"));
        let Some(Chain::Aws(mantle)) = builtin(BEDROCK_MANTLE) else {
            panic!("mantle has an AWS chain");
        };
        assert_eq!(mantle.service.as_deref(), Some("bedrock-mantle"));
        assert!(matches!(builtin(VERTEX), Some(Chain::Gcp(_))));
        assert!(matches!(builtin("azure"), Some(Chain::Azure(_))));
        let Some(Chain::Token(db)) = builtin("databricks") else {
            panic!("databricks has a token service");
        };
        assert_eq!(db.client_auth, Some(ClientAuth::Basic));
        assert_eq!(db.scope.as_deref(), Some("all-apis"));
        assert!(db.validate().is_ok());
        for id in ["watsonx", "sap-ai-core"] {
            let Some(Chain::Token(t)) = builtin(id) else {
                panic!("{id} has a token service");
            };
            assert!(t.validate().is_ok(), "{id}");
        }
        assert_eq!(builtin("openai"), None);
        assert_eq!(builtin("anthropic"), None);
    }

    #[test]
    fn kinds_and_overlays() {
        assert_eq!(builtin(BEDROCK).unwrap().kind(), "aws_sigv4");
        assert_eq!(builtin(VERTEX).unwrap().kind(), "adc");
        assert_eq!(builtin("azure").unwrap().kind(), "entra");
        assert_eq!(builtin("databricks").unwrap().kind(), "client_credentials");
        assert_eq!(builtin("watsonx").unwrap().kind(), "ibm_iam");
        // Two chains of one kind merge; a different kind replaces.
        let base = builtin(BEDROCK).unwrap();
        let merged = base.clone().overlaid(Chain::Aws(AwsChain {
            profile: Some("work".into()),
            ..AwsChain::default()
        }));
        let Chain::Aws(a) = merged else { panic!() };
        assert_eq!(a.profile.as_deref(), Some("work"));
        assert_eq!(a.service.as_deref(), Some("bedrock"));
        let replaced = base.overlaid(Chain::Gcp(GcpChain::default()));
        assert_eq!(replaced.kind(), "adc");
    }

    /// A stored selection is the user's word that the chain is there.
    #[test]
    fn a_stored_selection_is_forced_and_round_trips() {
        let chain = Chain::Aws(AwsChain {
            profile: Some("sso-dev".into()),
            ..AwsChain::default()
        });
        let json = serde_json::to_value(&chain).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"chain": "aws", "profile": "sso-dev"})
        );
        let back: Chain = serde_json::from_value(json).unwrap();
        assert_eq!(back, chain);
        let Chain::Aws(forced) = back.forced() else {
            panic!()
        };
        assert!(forced.forced);
        // A token service round-trips with its tag.
        let token = Chain::Token(TokenService {
            token_url: Some("https://idp.example/token".into()),
            client_id: Some("c".into()),
            client_secret_env: Some("S".into()),
            ..TokenService::default()
        });
        let json = serde_json::to_value(&token).unwrap();
        assert_eq!(json["chain"], "token");
        assert_eq!(serde_json::from_value::<Chain>(json).unwrap(), token);
    }
}
