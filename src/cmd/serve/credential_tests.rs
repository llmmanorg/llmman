//! What the daemon sends a hosted provider, and when: the credential's
//! kind, where the provider's scheme puts it, and the rules that keep
//! another site's page, an off-loopback caller, or a redirect from getting
//! at it.

use super::test_support::{headers_with, remote_target, remote_target_on, test_state};
use super::*;
use crate::authscheme::{Auth, Scheme};
use crate::credentials::{Credential, OAuth, Resolved, Style};
use crate::providers::{Provider, Sources};
use axum::routing::{get, post};

fn built(target: &Target, url: &str) -> reqwest::Request {
    target
        .authorize(Client::new().post(url))
        .build()
        .expect("a request")
}

fn remote_with(wire: Wire, auth: Auth, credential: Option<Resolved>) -> Target {
    Target::Remote(Arc::new(RemoteTarget {
        provider: "p".into(),
        base_url: "https://p.example/v1".into(),
        wire,
        model: "m".into(),
        max_output: None,
        cost: None,
        credential,
        auth,
        session: None,
    }))
}

fn key(secret: &str) -> Resolved {
    Resolved::from_key(secret.into(), "env")
}

// ---------------------------------------------------------------------------
// The request a credential becomes
// ---------------------------------------------------------------------------

/// A provider nobody configured is authenticated as it always was: a
/// bearer on the OpenAI wire, `x-api-key` and the version on Anthropic's,
/// and nothing else at all.
#[test]
fn an_api_key_provider_is_authenticated_exactly_as_before() {
    let openai = built(
        &remote_target("https://p.example/v1"),
        "https://p.example/v1/x",
    );
    let mut names: Vec<_> = openai.headers().keys().map(|k| k.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["authorization"]);
    assert_eq!(openai.headers()["authorization"], "Bearer sk-test");
    assert!(openai.headers()["authorization"].is_sensitive());
    assert_eq!(openai.url().query(), None);

    let anthropic = built(
        &remote_target_on("https://p.example/v1", Wire::Anthropic),
        "https://p.example/v1/messages",
    );
    let mut names: Vec<_> = anthropic.headers().keys().map(|k| k.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["anthropic-version", "x-api-key"]);
    assert_eq!(anthropic.headers()["x-api-key"], "sk-test");
    assert_eq!(anthropic.headers()["anthropic-version"], anthropic::VERSION);
}

#[test]
fn a_providers_scheme_decides_where_the_key_goes() {
    // Azure OpenAI: `api-key` and `?api-version=`.
    let azure = remote_with(
        Wire::OpenAi,
        Auth {
            scheme: Scheme::Header("api-key".into()),
            query: vec![("api-version".into(), "v1".into())],
            ..Auth::default()
        },
        Some(key("az-key")),
    );
    let r = built(&azure, "https://p.example/v1/chat/completions");
    assert_eq!(r.headers()["api-key"], "az-key");
    assert!(r.headers().get("authorization").is_none());
    assert_eq!(r.url().query(), Some("api-version=v1"));

    // Google's `x-goog-api-key` and OpenRouter's attribution headers.
    let gateway = remote_with(
        Wire::OpenAi,
        Auth {
            scheme: Scheme::Header("x-goog-api-key".into()),
            headers: vec![
                ("HTTP-Referer".into(), "https://llmman.test".into()),
                ("X-Title".into(), "llmman".into()),
            ],
            ..Auth::default()
        },
        Some(key("g-key")),
    );
    let r = built(&gateway, "https://p.example/v1/chat/completions");
    assert_eq!(r.headers()["x-goog-api-key"], "g-key");
    assert_eq!(r.headers()["http-referer"], "https://llmman.test");
    assert_eq!(r.headers()["x-title"], "llmman");

    // A key in the query, on the Anthropic wire, keeping the version header.
    let query = remote_with(
        Wire::Anthropic,
        Auth {
            scheme: Scheme::Query("key".into()),
            ..Auth::default()
        },
        Some(key("q-key")),
    );
    let r = built(&query, "https://p.example/v1/messages");
    assert_eq!(r.url().query(), Some("key=q-key"));
    assert!(r.headers().get("x-api-key").is_none());
    assert_eq!(r.headers()["anthropic-version"], anthropic::VERSION);

    // x-api-key on the OpenAI wire, bearer on the Anthropic one.
    for (wire, scheme, header) in [
        (Wire::OpenAi, Scheme::XApiKey, "x-api-key"),
        (Wire::Anthropic, Scheme::Bearer, "authorization"),
    ] {
        let t = remote_with(
            wire,
            Auth {
                scheme,
                ..Auth::default()
            },
            Some(key("k")),
        );
        let r = built(&t, "https://p.example/v1/x");
        assert!(r.headers().get(header).is_some(), "{header}");
    }
}

/// A subscription login travels as a bearer with its own headers, whatever
/// scheme the provider's table names.
#[test]
fn a_login_is_sent_as_a_bearer_with_the_headers_it_needs() {
    let mut chatgpt = Resolved::from_key("at-1".into(), "oauth");
    chatgpt.flow = Some(crate::logins::FLOW_CHATGPT.into());
    chatgpt
        .metadata
        .insert("account_id".into(), "acct-1".into());
    crate::logins::shape(&mut chatgpt);
    let t = remote_with(
        Wire::OpenAi,
        Auth {
            scheme: Scheme::Header("api-key".into()),
            ..Auth::default()
        },
        Some(chatgpt),
    );
    let r = built(&t, "https://chatgpt.com/backend-api/codex/responses");
    assert_eq!(r.headers()["authorization"], "Bearer at-1");
    assert_eq!(r.headers()["chatgpt-account-id"], "acct-1");
    assert!(r.headers().get("api-key").is_none());
    assert!(r.headers()["chatgpt-account-id"].is_sensitive());

    let mut setup = Resolved::from_key("sk-ant-oat01-x".into(), "token");
    setup.flow = Some(crate::logins::FLOW_SETUP_TOKEN.into());
    crate::logins::shape(&mut setup);
    assert_eq!(setup.style, Style::Bearer);
    let t = remote_with(Wire::Anthropic, Auth::default(), Some(setup));
    let r = built(&t, "https://api.anthropic.com/v1/messages");
    assert_eq!(r.headers()["authorization"], "Bearer sk-ant-oat01-x");
    assert!(r.headers().get("x-api-key").is_none());
    assert_eq!(r.headers()["anthropic-beta"], "oauth-2025-04-20");
    assert_eq!(r.headers()["anthropic-version"], anthropic::VERSION);
}

#[test]
fn a_remote_targets_debug_names_no_secret() {
    let mut login = Resolved::from_key("at-secret".into(), "oauth");
    login
        .headers
        .push(("ChatGPT-Account-Id".into(), "acct-secret".into()));
    let t = remote_with(
        Wire::OpenAi,
        Auth {
            headers: vec![("X-Gateway".into(), "gw-secret".into())],
            ..Auth::default()
        },
        Some(login),
    );
    let printed = format!("{t:?}");
    for secret in ["at-secret", "acct-secret", "gw-secret"] {
        assert!(!printed.contains(secret), "{printed}");
    }
}

// ---------------------------------------------------------------------------
// Whose credential is spent, and on whose behalf
// ---------------------------------------------------------------------------

const CATALOG: &str = r#"{
    "credtest": {"id": "credtest", "name": "Cred test",
        "api": "https://credtest.example/v1",
        "npm": "@ai-sdk/openai-compatible",
        "env": ["LLMMAN_TEST_PROVIDER_KEY_UNSET"],
        "models": {"m": {}}},
    "openai": {"id": "openai", "name": "OpenAI", "npm": "@ai-sdk/openai",
        "env": ["LLMMAN_TEST_PROVIDER_KEY_UNSET"],
        "models": {"gpt-5": {}}},
    "azure": {"id": "azure", "name": "Azure", "npm": "@ai-sdk/azure",
        "env": ["AZURE_RESOURCE_NAME", "LLMMAN_TEST_PROVIDER_KEY_UNSET"],
        "models": {"gpt-5": {}}}
}"#;

/// Catalog and stores held for one test. The provider-key lock first, then
/// the credentials one, always in that order.
struct Fx {
    creds: crate::config::CredentialsFixture,
    _keys: std::sync::MutexGuard<'static, ()>,
}

fn fixture() -> Fx {
    let keys = crate::config::PROVIDER_KEYS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let creds = crate::config::CredentialsFixture::new();
    let vars = |_: &str, var: &str| (var == "AZURE_RESOURCE_NAME").then(|| "res-1".to_string());
    crate::providers::hold_catalog_with_for_tests(CATALOG, &Sources::with_vars(&vars, true));
    Fx { creds, _keys: keys }
}

fn keyed_state(keys: &[&str]) -> AppState {
    let mut inner = super::test_support::test_inner(std::env::temp_dir());
    inner.auth = auth::Policy::with_keys(keys.iter().copied());
    AppState(Arc::new(inner))
}

fn login(access: &str, flow: &str) -> Credential {
    Credential::OAuth(OAuth {
        flow: flow.into(),
        access: access.into(),
        refresh: "r".into(),
        expires_ms: crate::credentials::now_ms() + 3_600_000,
        metadata: [("account_id".to_string(), "acct-7".to_string())].into(),
    })
}

async fn remote(
    state: &AppState,
    model: &str,
    headers: Option<&HeaderMap>,
) -> Result<Arc<RemoteTarget>, AppError> {
    match resolve_remote_target(state, model, headers).await? {
        Some(Target::Remote(r)) => Ok(r),
        other => panic!("not a remote target: {other:?}"),
    }
}

#[tokio::test]
async fn an_authenticated_caller_spends_the_daemons_stored_login() {
    let fx = fixture();
    let held = crate::providers::catalog().unwrap();
    assert!(
        held.get("openai-chatgpt").is_some(),
        "login providers are offered"
    );
    fx.creds
        .store
        .put(
            "openai-chatgpt",
            login("at-chatgpt", crate::logins::FLOW_CHATGPT),
        )
        .unwrap();

    let state = keyed_state(&["daemon-key"]);
    let target = remote(&state, "llmman.provider/openai-chatgpt/gpt-5", None)
        .await
        .unwrap();
    assert_eq!(target.base_url, "https://chatgpt.com/backend-api/codex");
    let cred = target.credential.as_ref().expect("a credential");
    assert_eq!((cred.kind, cred.secret.expose()), ("oauth", "at-chatgpt"));

    let r = built(
        &Target::Remote(target),
        "https://chatgpt.com/backend-api/codex/responses",
    );
    assert_eq!(r.headers()["authorization"], "Bearer at-chatgpt");
    assert_eq!(r.headers()["chatgpt-account-id"], "acct-7");
}

/// Off loopback with no daemon key, or from a page on another site, the
/// daemon's own credential — a stored login included — is not spent.
#[tokio::test]
async fn a_credential_is_withheld_from_a_page_on_another_site() {
    let fx = fixture();
    fx.creds
        .store
        .put(
            "credtest",
            Credential::ApiKey {
                key: "sk-daemon-owned".into(),
                flow: None,
            },
        )
        .unwrap();
    let cross_site = headers_with(&[("sec-fetch-site", "cross-site")]);
    let err = remote(
        &test_state(),
        "llmman.provider/credtest/m",
        Some(&cross_site),
    )
    .await
    .unwrap_err();
    assert_eq!(err.1, StatusCode::UNAUTHORIZED);
    let message = format!("{:#}", err.0);
    assert!(message.contains("another site"), "{message}");
    assert!(!message.contains("sk-daemon-owned"));

    // An authenticated caller is the operator, whatever the browser says.
    let target = remote(
        &keyed_state(&["k"]),
        "llmman.provider/credtest/m",
        Some(&cross_site),
    )
    .await
    .unwrap();
    assert_eq!(
        target.credential.as_ref().unwrap().secret.expose(),
        "sk-daemon-owned"
    );
}

/// `llmman` is what integrations send when they have no key. It is not one.
#[tokio::test]
async fn the_placeholder_key_is_never_forwarded() {
    let fx = fixture();
    let state = keyed_state(&["daemon-key"]);
    let placeholder = headers_with(&[("authorization", "Bearer llmman")]);
    let err = remote(&state, "llmman.provider/credtest/m", Some(&placeholder))
        .await
        .unwrap_err();
    assert_eq!(err.1, StatusCode::UNAUTHORIZED);
    assert!(format!("{:#}", err.0).contains("no API key"));

    // With a login on the daemon, that is what is spent — not the placeholder.
    fx.creds
        .store
        .put(
            "credtest",
            Credential::ApiKey {
                key: "sk-real".into(),
                flow: None,
            },
        )
        .unwrap();
    let target = remote(&state, "llmman.provider/credtest/m", Some(&placeholder))
        .await
        .unwrap();
    assert_eq!(
        target.credential.as_ref().unwrap().secret.expose(),
        "sk-real"
    );
}

#[tokio::test]
async fn a_callers_own_key_is_attached_the_way_the_provider_wants_it() {
    let _fx = fixture();
    let caller = headers_with(&[("authorization", "Bearer sk-caller")]);
    let target = remote(
        &keyed_state(&["k"]),
        "llmman.provider/azure/gpt-5",
        Some(&caller),
    )
    .await
    .unwrap();
    assert_eq!(target.base_url, "https://res-1.openai.azure.com/openai/v1");
    assert_eq!(target.credential.as_ref().unwrap().kind, "request");
    let r = built(
        &Target::Remote(target),
        "https://res-1.openai.azure.com/openai/v1/chat/completions",
    );
    assert_eq!(r.headers()["api-key"], "sk-caller");
    assert!(r.headers().get("authorization").is_none());
    assert_eq!(r.url().query(), Some("api-version=v1"));
}

#[tokio::test]
async fn a_login_that_has_run_out_is_a_401_that_says_to_log_in_again() {
    let fx = fixture();
    fx.creds
        .store
        .put(
            "credtest",
            Credential::Token {
                token: "tok-secret".into(),
                expires_ms: Some(1),
                flow: None,
            },
        )
        .unwrap();
    let err = remote(&keyed_state(&["k"]), "llmman.provider/credtest/m", None)
        .await
        .unwrap_err();
    assert_eq!(err.1, StatusCode::UNAUTHORIZED);
    let message = format!("{:#}", err.0);
    assert!(
        message.contains("llmman login --provider credtest"),
        "{message}"
    );
    assert!(!message.contains("tok-secret"), "{message}");
}

#[tokio::test]
async fn a_stored_command_is_resolved_off_the_runtime() {
    let fx = fixture();
    #[cfg(unix)]
    {
        fx.creds
            .store
            .put(
                "credtest",
                Credential::External {
                    source: crate::credentials::ExternalSource::Command {
                        command: "echo sk-from-$((20+2))".into(),
                    },
                },
            )
            .unwrap();
        let target = remote(&keyed_state(&["k"]), "llmman.provider/credtest/m", None)
            .await
            .unwrap();
        let cred = target.credential.as_ref().unwrap();
        assert_eq!((cred.kind, cred.secret.expose()), ("command", "sk-from-22"));
    }
    #[cfg(not(unix))]
    let _ = fx;
}

// ---------------------------------------------------------------------------
// What the daemon reports
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_provider_listing_reports_each_credentials_kind_and_no_secret() {
    let fx = fixture();
    fx.creds
        .store
        .put(
            "credtest",
            Credential::ApiKey {
                key: "sk-listing-secret".into(),
                flow: None,
            },
        )
        .unwrap();
    fx.creds
        .store
        .put(
            "openai-chatgpt",
            login("at-listing-secret", crate::logins::FLOW_CHATGPT),
        )
        .unwrap();
    let url = super::test_support::serve_router(keyed_state(&["k"])).await;
    let client = Client::new();

    let listing = client
        .get(format!("{url}/llmman/providers"))
        .bearer_auth("k")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!listing.contains("listing-secret"), "{listing}");
    let v: serde_json::Value = serde_json::from_str(&listing).unwrap();
    let by_id = |id: &str| {
        v["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == id)
            .unwrap_or_else(|| panic!("{id} is listed"))
            .clone()
    };
    let credtest = by_id("credtest");
    assert_eq!(credtest["credential"]["kind"], "api_key");
    assert_eq!(credtest["credential"]["state"], "ready");
    assert_eq!(credtest["key_set"], true);
    assert_eq!(credtest["key_usable"], true);
    assert_eq!(credtest["auth"], "default");
    let chatgpt = by_id("openai-chatgpt");
    assert_eq!(chatgpt["credential"]["kind"], "oauth");
    assert!(chatgpt["credential"]["expires_at_ms"].as_u64().unwrap() > 0);
    assert!(chatgpt.get("key_env").is_none());
    // One with none says so by omission.
    let azure = by_id("azure");
    assert!(azure.get("credential").is_none());
    assert_eq!(azure["auth"], "header");
    assert_eq!(azure["key_set"], false);

    let one = client
        .get(format!("{url}/llmman/providers/openai-chatgpt"))
        .bearer_auth("k")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!one.contains("listing-secret"), "{one}");
    let v: serde_json::Value = serde_json::from_str(&one).unwrap();
    assert_eq!(v["credential"]["kind"], "oauth");
    assert_eq!(v["credential"]["state"], "ready");
}

// ---------------------------------------------------------------------------
// Through the wire to a provider
// ---------------------------------------------------------------------------

/// A provider that records what it receives and answers a chat completion.
async fn recording_provider() -> (String, Arc<tokio::sync::Mutex<Vec<(HeaderMap, String)>>>) {
    let seen: Arc<tokio::sync::Mutex<Vec<(HeaderMap, String)>>> = Arc::default();
    let captured = seen.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |uri: axum::http::Uri, headers: HeaderMap| async move {
            captured
                .lock()
                .await
                .push((headers, uri.query().unwrap_or("").to_string()));
            Json(serde_json::json!({"id": "c", "choices": []}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, seen)
}

fn plain_provider(id: &str, base_url: &str, auth: Auth) -> Provider {
    Provider {
        id: id.into(),
        name: id.into(),
        base_url: base_url.into(),
        key_env: None,
        wire: Wire::OpenAi,
        key_optional: false,
        models: vec![],
        auth,
    }
}

/// The whole path: a stored key, a provider that wants `api-key` and a
/// version, and a request that arrives with exactly those — and with
/// neither the daemon's own key nor the caller's.
#[tokio::test]
async fn a_chat_request_reaches_the_provider_with_the_stored_key_in_its_scheme() {
    let fx = fixture();
    let (base, seen) = recording_provider().await;
    crate::providers::hold_providers_for_tests(vec![plain_provider(
        "wired",
        &base,
        Auth {
            scheme: Scheme::Header("api-key".into()),
            headers: vec![("X-Gateway".into(), "gw".into())],
            query: vec![("api-version".into(), "2024-10".into())],
            ..Auth::default()
        },
    )]);
    fx.creds
        .store
        .put(
            "wired",
            Credential::ApiKey {
                key: "sk-wired".into(),
                flow: None,
            },
        )
        .unwrap();
    let state = keyed_state(&["daemon-key"]);
    let target = Target::Remote(
        remote(&state, "llmman.provider/wired/m", None)
            .await
            .unwrap(),
    );
    let chat = serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let up = send_chat_completion(&state.0.client, &target, &chat, "m")
        .await
        .unwrap();
    assert_eq!(up.status, StatusCode::OK);

    let seen = seen.lock().await;
    let (headers, query) = &seen[0];
    assert_eq!(headers["api-key"], "sk-wired");
    assert_eq!(headers["x-gateway"], "gw");
    assert_eq!(query, "api-version=2024-10");
    assert!(headers.get("authorization").is_none(), "{headers:?}");
    assert!(!format!("{headers:?}").contains("daemon-key"));
}

/// Redirects are followed only within the origin they started at: the key
/// is in `x-api-key` / `api-key`, which reqwest would otherwise carry to
/// another host.
#[tokio::test]
async fn a_redirect_to_another_origin_is_not_followed_with_the_key() {
    let elsewhere_seen: Arc<tokio::sync::Mutex<Vec<HeaderMap>>> = Arc::default();
    let captured = elsewhere_seen.clone();
    let elsewhere = Router::new().fallback(get(move |headers: HeaderMap| async move {
        captured.lock().await.push(headers);
        "reached"
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let other = format!(
        "http://127.0.0.1:{}/stolen",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, elsewhere).await.unwrap() });

    let provider = Router::new()
        .route(
            "/away",
            get(move || {
                let other = other.clone();
                async move { (StatusCode::TEMPORARY_REDIRECT, [("location", other)]) }
            }),
        )
        .route(
            "/back",
            get(|| async { (StatusCode::FOUND, [("location", "/ok")]) }),
        )
        .route(
            "/ok",
            get(|headers: HeaderMap| async move {
                headers
                    .get("x-api-key")
                    .map(|v| v.to_str().unwrap().to_string())
                    .unwrap_or_default()
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    tokio::spawn(async move { axum::serve(listener, provider).await.unwrap() });

    let client = reqwest::Client::builder()
        .redirect(crate::authscheme::same_origin_redirects())
        .build()
        .unwrap();
    let r = client
        .get(format!("{origin}/away"))
        .header("x-api-key", "sk-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        StatusCode::TEMPORARY_REDIRECT,
        "stopped at the redirect"
    );
    assert!(
        elsewhere_seen.lock().await.is_empty(),
        "nothing reached the other host"
    );

    // Same origin: followed, and the header goes along.
    let r = client
        .get(format!("{origin}/back"))
        .header("x-api-key", "sk-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(r.text().await.unwrap(), "sk-secret");
}

/// reqwest quotes the URL in a send error; a key in the query must not
/// come out the other end in the JSON a client reads.
#[tokio::test]
async fn an_error_response_does_not_quote_a_query_credential() {
    let target = remote_with(
        Wire::OpenAi,
        Auth {
            scheme: Scheme::Query("key".into()),
            ..Auth::default()
        },
        Some(key("sk-in-the-url-9f3")),
    );
    let req = target.authorize(Client::new().post("http://127.0.0.1:9/v1/chat/completions"));
    let err = target
        .send(req)
        .await
        .expect_err("nothing listens on port 9");
    let message = format!("{err:#}");
    assert!(!message.contains("sk-in-the-url-9f3"), "{message}");
}

#[tokio::test]
async fn a_configured_providers_models_are_listed_with_its_headers_and_key() {
    let fx = fixture();
    let (base, _seen) = recording_provider().await;
    let seen: Arc<tokio::sync::Mutex<Vec<(HeaderMap, String)>>> = Arc::default();
    let captured = seen.clone();
    let app = Router::new().route(
        "/v1/models",
        get(move |uri: axum::http::Uri, headers: HeaderMap| async move {
            captured
                .lock()
                .await
                .push((headers, uri.query().unwrap_or("").to_string()));
            Json(serde_json::json!({"data": [{"id": "only"}]}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let models_base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    drop(base);

    let mut provider = plain_provider(
        "box",
        &models_base,
        Auth {
            scheme: Scheme::Header("api-key".into()),
            query: vec![("api-version".into(), "v9".into())],
            ..Auth::default()
        },
    );
    provider.key_optional = true;
    fx.creds
        .store
        .put(
            "box",
            Credential::ApiKey {
                key: "sk-box".into(),
                flow: None,
            },
        )
        .unwrap();
    let models = configured_provider_models(&keyed_state(&["k"]), &provider).await;
    assert_eq!(models, vec!["only".to_string()]);
    let seen = seen.lock().await;
    assert_eq!(seen[0].0["api-key"], "sk-box");
    assert_eq!(seen[0].1, "api-version=v9");
}

// ---------------------------------------------------------------------------
// Cloud chains
// ---------------------------------------------------------------------------

/// What a SigV4-signed request looks like to the server that checks it: its
/// method, URL, content type, the exact body bytes, and the signature.
type Captured = Arc<tokio::sync::Mutex<Vec<(String, String, HeaderMap, Vec<u8>)>>>;

async fn signature_checking_provider() -> (String, Captured) {
    let seen: Captured = Arc::default();
    let captured = seen.clone();
    let app = Router::new().route(
        "/openai/v1/chat/completions",
        post(
            move |method: axum::http::Method,
                  uri: axum::http::Uri,
                  headers: HeaderMap,
                  body: axum::body::Bytes| async move {
                captured.lock().await.push((
                    method.to_string(),
                    uri.to_string(),
                    headers,
                    body.to_vec(),
                ));
                Json(serde_json::json!({"id": "c", "choices": []}))
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!(
        "http://127.0.0.1:{}/openai/v1",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, seen)
}

fn signing_credential(at: std::time::SystemTime) -> Resolved {
    let credentials = aws_credential_types::Credentials::new(
        "AKIDEXAMPLE",
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        Some("session-token-1".to_string()),
        None,
        "test",
    );
    Resolved {
        secret: crate::credentials::Secret::new(String::new()),
        style: Style::Key,
        kind: "aws_sigv4",
        flow: None,
        metadata: Default::default(),
        headers: Vec::new(),
        signer: Some(Arc::new(crate::aws::sigv4_at(
            credentials,
            "us-east-1",
            "bedrock",
            at,
        ))),
    }
}

/// The request that reaches the provider carries a signature over its own
/// method, URL, content type and body bytes, and no bearer; the same
/// signature does not verify against any other body.
#[tokio::test]
async fn a_sigv4_request_is_signed_over_the_bytes_that_were_sent() {
    let (base, seen) = signature_checking_provider().await;
    let at = std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let target = Target::Remote(Arc::new(RemoteTarget {
        provider: "amazon-bedrock".into(),
        base_url: base.clone(),
        wire: Wire::OpenAi,
        model: "openai.gpt-oss-120b-1:0".into(),
        max_output: None,
        cost: None,
        credential: Some(signing_credential(at)),
        auth: Auth::default(),
        session: None,
    }));
    let client = Client::new();
    let body = serde_json::json!({"model": "openai.gpt-oss-120b-1:0", "messages": [{"role": "user", "content": "hi — ünïcode"}]});
    let resp = target
        .send(target.authorize(client.post(target.url("/v1/chat/completions")).json(&body)))
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let seen = seen.lock().await;
    let (method, uri, headers, sent) = &seen[0];
    assert_eq!(sent, &serde_json::to_vec(&body).unwrap());
    assert!(headers
        .get("authorization")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("AWS4-HMAC-SHA256 "));
    assert_eq!(headers["x-amz-date"], "20231114T221320Z");
    assert_eq!(headers["x-amz-security-token"], "session-token-1");
    assert!(!headers["authorization"]
        .to_str()
        .unwrap()
        .contains("Bearer"));

    // The server's view: sign what it received and compare.
    let host = headers["host"].to_str().unwrap();
    let url = format!("http://{host}{uri}");
    let verifier = crate::aws::sigv4_at(
        aws_credential_types::Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token-1".to_string()),
            None,
            "test",
        ),
        "us-east-1",
        "bedrock",
        at,
    );
    let content_type = headers["content-type"].to_str().unwrap().to_string();
    let recompute = |body: &[u8]| {
        verifier
            .headers_for(
                method,
                &url,
                &[("content-type".into(), content_type.clone())],
                body,
            )
            .unwrap()
            .into_iter()
            .find(|(k, _)| k == "authorization")
            .unwrap()
            .1
    };
    assert_eq!(recompute(sent), headers["authorization"].to_str().unwrap());
    let mut tampered = sent.clone();
    tampered.push(b' ');
    assert_ne!(
        recompute(&tampered),
        headers["authorization"].to_str().unwrap()
    );
}

/// The daemon's own credential for a provider with a token service behind
/// it is resolved for a request, as a bearer, and reported without a secret.
#[tokio::test]
async fn a_token_service_credential_is_spent_as_a_bearer_and_listed_without_its_secret() {
    use crate::chains::Chain;
    use crate::tokensvc::TokenService;
    let fx = fixture();
    let server = crate::testserver::server();
    server.fake.on(
        "/serve-token",
        vec![(
            200,
            serde_json::json!({"access_token": "minted-for-serve", "expires_in": 3600}),
        )],
    );
    fx.creds
        .store
        .put(
            "credtest",
            Credential::Chain(Chain::Token(TokenService {
                token_url: Some(format!("{}/serve-token", server.origin)),
                client_id: Some("cid".into()),
                client_secret: Some("serve-client-secret".into()),
                ..TokenService::default()
            })),
        )
        .unwrap();
    let state = keyed_state(&["k"]);
    let target = remote(&state, "llmman.provider/credtest/m", None)
        .await
        .unwrap();
    let credential = target.credential.as_ref().unwrap();
    assert_eq!(credential.secret.expose(), "minted-for-serve");
    assert_eq!(credential.kind, "client_credentials");
    let built = target_request(&Target::Remote(target.clone()));
    assert_eq!(built.headers()["authorization"], "Bearer minted-for-serve");

    let url = super::test_support::serve_router(state).await;
    let listing = Client::new()
        .get(format!("{url}/llmman/providers"))
        .bearer_auth("k")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        !listing.contains("serve-client-secret") && !listing.contains("minted-for-serve"),
        "{listing}"
    );
    let v: serde_json::Value = serde_json::from_str(&listing).unwrap();
    let credtest = v["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "credtest")
        .unwrap();
    assert_eq!(credtest["credential"]["kind"], "client_credentials");
    assert_eq!(credtest["credential"]["state"], "ready");
    assert_eq!(credtest["key_set"], true);
}

fn target_request(target: &Target) -> reqwest::Request {
    built(target, "https://p.example/v1/x")
}

// ---------------------------------------------------------------------------
// A launched agent that can switch provider
// ---------------------------------------------------------------------------

fn bearer_of(token: &str) -> HeaderMap {
    headers_with(&[("authorization", &format!("Bearer {token}"))])
}

/// The rule that makes it safe to offer an agent every provider: a key
/// registered for provider A is spent on A and never reaches B after the
/// agent switches, and the token itself is no one's credential.
#[tokio::test]
async fn a_session_key_is_sent_to_its_own_provider_and_never_another() {
    let _fx = fixture();
    let state = test_state();
    let token =
        sessions::open([("credtest".to_string(), "sk-for-credtest".to_string())].into()).unwrap();
    let headers = bearer_of(&token);

    // The provider the key belongs to gets it.
    let own = remote(&state, "llmman.provider/credtest/m", Some(&headers))
        .await
        .unwrap();
    let cred = own.credential.as_ref().expect("a credential");
    assert_eq!(
        (cred.kind, cred.secret.expose()),
        ("session", "sk-for-credtest")
    );
    let sent = built(&Target::Remote(own), "https://credtest.example/v1/x");
    assert_eq!(sent.headers()["authorization"], "Bearer sk-for-credtest");

    // After a switch to another provider, the same header carries no key
    // for it: an error naming what to set, not the first provider's key.
    let err = remote(&state, "llmman.provider/openai/gpt-5", Some(&headers)).await;
    let Err(err) = err else {
        panic!("the key of one provider was sent to another");
    };
    assert_eq!(err.1, StatusCode::UNAUTHORIZED);
    let message = err.0.to_string();
    assert!(
        message.contains("openai") && message.contains("OPENAI_API_KEY"),
        "{message}"
    );
    assert!(
        !message.contains("sk-for-credtest") && !message.contains(&token),
        "{message}"
    );
    sessions::close(&token);
}

#[tokio::test]
async fn a_session_holds_one_key_per_provider_and_a_closed_one_holds_none() {
    let _fx = fixture();
    let state = test_state();
    let token = sessions::open(
        [
            ("credtest".to_string(), "sk-one".to_string()),
            ("openai".to_string(), "sk-two".to_string()),
        ]
        .into(),
    )
    .unwrap();
    let headers = bearer_of(&token);
    for (model, want) in [
        ("llmman.provider/credtest/m", "sk-one"),
        ("llmman.provider/openai/gpt-5", "sk-two"),
    ] {
        let target = remote(&state, model, Some(&headers)).await.unwrap();
        assert_eq!(target.credential.as_ref().unwrap().secret.expose(), want);
    }
    sessions::close(&token);
    // An expired or restarted daemon's token is never forwarded as a key.
    let err = remote(&state, "llmman.provider/credtest/m", Some(&headers))
        .await
        .unwrap_err();
    assert_eq!(err.1, StatusCode::UNAUTHORIZED);
    assert!(!err.0.to_string().contains(&token));
}

/// A client that names one provider and sends that provider's own key, as
/// `llmman run --provider` and third-party clients do, is unchanged.
#[tokio::test]
async fn a_key_in_the_header_is_still_the_callers_for_the_provider_it_names() {
    let _fx = fixture();
    let target = remote(
        &test_state(),
        "llmman.provider/credtest/m",
        Some(&bearer_of("sk-callers-own")),
    )
    .await
    .unwrap();
    let cred = target.credential.as_ref().unwrap();
    assert_eq!(
        (cred.kind, cred.secret.expose()),
        ("request", "sk-callers-own")
    );
}

#[tokio::test]
async fn opening_a_session_is_refused_from_another_sites_page() {
    let refused = handle_open_session(
        headers_with(&[("sec-fetch-site", "cross-site")]),
        Json(OpenSessionBody {
            keys: [("p".to_string(), "k".to_string())].into(),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(refused.1, StatusCode::FORBIDDEN);
    let (status, Json(opened)) = handle_open_session(
        HeaderMap::new(),
        Json(OpenSessionBody {
            keys: [("p".to_string(), "k".to_string())].into(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        handle_close_session(bearer_of(&opened.token)).await,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        handle_close_session(bearer_of(&opened.token)).await,
        StatusCode::NOT_FOUND
    );
}

/// What `POST /llmman/sessions` accepts is bounded by the route itself, not
/// by the 2 MiB default: a launch needs a key per provider, no more.
#[tokio::test]
async fn opening_a_session_refuses_a_body_larger_than_a_launch_needs() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(test_state(), false);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let url = format!("http://127.0.0.1:{}/llmman/sessions", addr.port());
    let open = |keys: serde_json::Value| {
        Client::new()
            .post(&url)
            .json(&serde_json::json!({ "keys": keys }))
            .send()
    };

    let resp = open(serde_json::json!({ "openai": "sk-test" }))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let big = "k".repeat(sessions::BODY_LIMIT_BYTES);
    let resp = open(serde_json::json!({ "openai": big })).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
