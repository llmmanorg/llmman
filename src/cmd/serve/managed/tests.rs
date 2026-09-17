use super::*;
use axum::extract::OriginalUri;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

const MODEL: &str = "llmman.provider/openai/gpt-5.6-terra";

struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(app: Router) -> Server {
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", socket.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(socket, app).await.unwrap() });
    Server { url, task }
}
fn state(base: &str) -> ManagedState {
    ManagedState(Arc::new(ManagedInner {
        connection_auth: auth::Policy::with_keys(["connection-secret"]),
        ready: Ready {
            schema_version: 1,
            instance_id: "instance-123".into(),
            pid: std::process::id(),
            endpoint: "https://127.0.0.1:45678".into(),
            capabilities: vec![CAPABILITY, CLAUDE_CAPABILITY],
        },
        clients: std::sync::Mutex::default(),
        test_base: Some(base.into()),
    }))
}
const CLAUDE_MODEL: &str = "llmman.provider/anthropic/claude-sonnet-4-5";

fn request(url: &str, operation: &str) -> reqwest::RequestBuilder {
    profile_request(url, operation, Profile::Codex)
}

fn profile_request(url: &str, operation: &str, profile: Profile) -> reqwest::RequestBuilder {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{url}/v1/{operation}"))
        .header(CONNECTION_KEY, "connection-secret")
        .header(AUTH_PROFILE, profile.name())
        .header(UPSTREAM_IP, "104.18.2.1")
        .bearer_auth("provider-secret")
}

#[tokio::test]
async fn native_requests_preserve_protocol_and_strip_internal_headers() {
    let seen = Arc::new(Mutex::new(Vec::<(String, HeaderMap, Value)>::new()));
    let capture = seen.clone();
    let upstream = serve(Router::new().fallback(post(
        move |OriginalUri(uri): OriginalUri, headers: HeaderMap, Json(body): Json<Value>| {
            let capture = capture.clone();
            async move {
                capture
                    .lock()
                    .await
                    .push((uri.path().to_owned(), headers, body));
                (
                    [
                        ("content-type", "text/event-stream"),
                        ("x-request-id", "req-1-provider-secret-account-1"),
                        ("x-llmman-managed-key", "do-not-relay"),
                        ("authorization", "do-not-relay"),
                    ],
                    "event: response.completed\ndata: {\"id\":\"resp_1\"}\n\n",
                )
            }
        },
    )))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    let body = json!({"model": MODEL, "previous_response_id":"resp_previous", "input":[{"type":"function_call_output","call_id":"call1","output":"tool result"}], "reasoning":{"effort":"high"}, "future_native_field":{"enabled":true}});
    for operation in ["responses", "responses/compact"] {
        let response = request(&managed.url, operation)
            .header("chatgpt-account-id", "account-1")
            .header("session_id", "session-1")
            .header("x-codex-turn-metadata", "{\"request_kind\":\"turn\"}")
            .header("x-api-key", "unrelated-key")
            .header("x-arbitrary", "do-not-relay")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["x-request-id"],
            "req-1-<redacted>-<redacted>"
        );
        assert!(!response.headers().contains_key(CONNECTION_KEY));
        assert!(!response.headers().contains_key("authorization"));
        assert_eq!(
            response.text().await.unwrap(),
            "event: response.completed\ndata: {\"id\":\"resp_1\"}\n\n"
        );
    }
    let calls = seen.lock().await;
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, "/responses");
    assert_eq!(calls[1].0, "/responses/compact");
    for (_, headers, forwarded) in calls.iter() {
        assert_eq!(headers["authorization"], "Bearer provider-secret");
        assert_eq!(headers["chatgpt-account-id"], "account-1");
        assert_eq!(headers["session_id"], "session-1");
        assert_eq!(headers["originator"], "codex_cli_rs");
        assert_eq!(headers["openai-beta"], "responses=experimental");
        for name in [
            CONNECTION_KEY,
            AUTH_PROFILE,
            UPSTREAM_IP,
            "x-api-key",
            "x-arbitrary",
        ] {
            assert!(!headers.contains_key(name));
        }
        let mut expected = body.clone();
        expected["model"] = json!("gpt-5.6-terra");
        assert_eq!(*forwarded, expected);
    }
}

#[tokio::test]
async fn authentication_capabilities_and_inference_only_routes() {
    let managed = serve(router(state("http://127.0.0.1:9"))).await;
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for key in [None, Some("wrong-key"), Some("provider-secret")] {
        let mut req = client
            .get(format!("{}/v1/capabilities", managed.url))
            .bearer_auth("connection-secret");
        if let Some(key) = key {
            req = req.header(CONNECTION_KEY, key);
        }
        assert_eq!(req.send().await.unwrap().status(), StatusCode::UNAUTHORIZED);
    }
    let ready: Value = client
        .get(format!("{}/v1/capabilities", managed.url))
        .header(CONNECTION_KEY, "connection-secret")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ready["schemaVersion"], 1);
    assert_eq!(ready["instanceId"], "instance-123");
    assert_eq!(
        ready["capabilities"],
        json!([CAPABILITY, CLAUDE_CAPABILITY])
    );
    assert!(!ready.to_string().contains("secret"));
    for path in [
        "/llmman/shell",
        "/api/pull",
        "/v1/chat/completions",
        "/v1/responses/input_tokens",
        "/",
    ] {
        let r = client
            .post(format!("{}{path}", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn malformed_profiles_tokens_models_never_reach_upstream() {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let upstream = serve(Router::new().fallback(move || {
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            "unexpected"
        }
    }))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for profile in [
        None,
        Some("api-key"),
        Some("codex-oauth,codex-oauth"),
        Some(""),
    ] {
        let mut req = client
            .post(format!("{}/v1/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
            .bearer_auth("provider-secret")
            .json(&json!({"model":MODEL}));
        if let Some(profile) = profile {
            req = req.header(AUTH_PROFILE, profile);
        }
        assert_eq!(req.send().await.unwrap().status(), StatusCode::BAD_REQUEST);
    }
    assert_eq!(
        request(&managed.url, "responses")
            .header(AUTH_PROFILE, "codex-oauth")
            .header(UPSTREAM_IP, "104.18.2.1")
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    for authorization in [
        "Bearer llmman",
        "Bearer ",
        "Basic secret",
        "Bearer two tokens",
        "Bearer a,b",
    ] {
        let r = client
            .post(format!("{}/v1/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
            .header(AUTH_PROFILE, "codex-oauth")
            .header(UPSTREAM_IP, "104.18.2.1")
            .header("authorization", authorization)
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED, "{authorization}");
    }
    for model in [
        "gpt-5.6-terra",
        "llmman.provider/anthropic/claude",
        "llmman.provider/openai/",
        "llmman.provider/openai/model?url=evil",
    ] {
        assert_eq!(
            request(&managed.url, "responses")
                .json(&json!({"model":model}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let duplicate = format!("{{\"model\":\"{MODEL}\",\"model\":\"other\"}}");
    assert_eq!(
        request(&managed.url, "responses")
            .body(duplicate)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn public_listener_rejects_managed_profile_before_provider_lookup() {
    for (profile, operation, model) in [
        (Profile::Codex, "responses", MODEL),
        (Profile::Claude, "messages", CLAUDE_MODEL),
    ] {
        let public = serve(super::super::build_router(
            super::super::tests::test_state(),
            false,
        ))
        .await;
        let response = profile_request(&public.url, operation, profile)
            .json(&json!({"model":model}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!response.text().await.unwrap().contains("secret"));
    }
}

#[tokio::test]
async fn provider_errors_are_not_retried_and_redact_credentials() {
    for (profile, operation, model) in [
        (Profile::Codex, "responses", MODEL),
        (Profile::Claude, "messages", CLAUDE_MODEL),
    ] {
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let count = calls.clone();
            let message = if profile == Profile::Codex {
                "unavailable provider-secret account-1"
            } else {
                "unavailable provider-secret"
            };
            let upstream = serve(Router::new().fallback(move || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    (
                        status,
                        Json(json!({"error":{"message":message,"code":"provider_error"}})),
                    )
                }
            }))
            .await;
            let managed = serve(router(state(&upstream.url))).await;
            let response = profile_request(&managed.url, operation, profile)
                .header("chatgpt-account-id", "account-1")
                .json(&json!({"model":model}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            let text = response.text().await.unwrap();
            assert!(!text.contains("provider-secret") && !text.contains("account-1"));
            assert!(text.contains("provider_error"));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn redirects_never_receive_delegated_credentials() {
    for (profile, operation, model) in [
        (Profile::Codex, "responses", MODEL),
        (Profile::Claude, "messages", CLAUDE_MODEL),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let destination = serve(Router::new().fallback(move || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                "unexpected"
            }
        }))
        .await;
        let url = destination.url.clone();
        let upstream = serve(Router::new().fallback(move || {
            let url = url.clone();
            async move { (StatusCode::TEMPORARY_REDIRECT, [("location", url)]) }
        }))
        .await;
        let managed = serve(router(state(&upstream.url))).await;
        let response = profile_request(&managed.url, operation, profile)
            .json(&json!({"model":model}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(!response.headers().contains_key("location"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn concurrent_accounts_remain_paired_on_each_request() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let upstream = serve(Router::new().fallback(post(move |headers: HeaderMap| { let barrier=barrier.clone(); async move {
        barrier.wait().await;
        Json(json!({"bearer":headers["authorization"].to_str().unwrap(),"account":headers["chatgpt-account-id"].to_str().unwrap()}))
    }}))).await;
    let managed = serve(router(state(&upstream.url))).await;
    let send = |n| {
        let url = managed.url.clone();
        async move {
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap()
                .post(format!("{url}/v1/responses"))
                .header(CONNECTION_KEY, "connection-secret")
                .header(AUTH_PROFILE, "codex-oauth")
                .header(UPSTREAM_IP, "104.18.2.1")
                .bearer_auth(format!("token-{n}"))
                .header("chatgpt-account-id", format!("account-{n}"))
                .json(&json!({"model":MODEL}))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };
    let (first, second) = tokio::join!(send(1), send(2));
    assert_eq!(
        first,
        json!({"bearer":"Bearer token-1","account":"account-1"})
    );
    assert_eq!(
        second,
        json!({"bearer":"Bearer token-2","account":"account-2"})
    );
}

#[tokio::test]
async fn streaming_starts_before_completion_and_disconnect_cancels_upstream() {
    for (profile, operation, model) in [
        (Profile::Codex, "responses", MODEL),
        (Profile::Claude, "messages", CLAUDE_MODEL),
    ] {
        struct Cancel(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for Cancel {
            fn drop(&mut self) {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
            }
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        let guard = Arc::new(std::sync::Mutex::new(Some(Cancel(Some(tx)))));
        let upstream = serve(Router::new().fallback(move || {
            let guard = guard.lock().unwrap().take().unwrap();
            async move {
                let stream = futures::stream::unfold((true, guard), |(first, guard)| async move {
                    if !first {
                        std::future::pending::<()>().await;
                    }
                    Some((
                        Ok::<_, std::io::Error>(Bytes::from_static(
                            b"event: response.created\ndata: {}\n\n",
                        )),
                        (false, guard),
                    ))
                });
                (
                    [("content-type", "text/event-stream")],
                    Body::from_stream(stream),
                )
            }
        }))
        .await;
        let managed = serve(router(state(&upstream.url))).await;
        let mut response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            profile_request(&managed.url, operation, profile)
                .json(&json!({"model":model,"stream":true}))
                .send(),
        )
        .await
        .unwrap()
        .unwrap();
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), response.chunk())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&chunk).contains("response.created"));
        drop(response);
        tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .expect("upstream stream cancelled")
            .unwrap();
    }
}

fn config(directory: &Path) -> Config {
    Config {
        schema_version: 1,
        instance_id: "fixture-instance".into(),
        listen: "127.0.0.1:0".parse().unwrap(),
        tls_cert_file: directory.join("cert.pem"),
        tls_key_file: directory.join("key.pem"),
        connection_key_file: directory.join("connection-key"),
        ready_file: directory.join("ready.json"),
    }
}

#[test]
fn configuration_contract_and_secret_debug_are_fail_closed() {
    let directory = std::env::temp_dir();
    let mut config = config(&directory);
    assert!(config.validate().is_ok());
    config.schema_version = 2;
    assert!(config.validate().is_err());
    config.schema_version = 1;
    config.listen = "0.0.0.0:0".parse().unwrap();
    assert!(config.validate().is_err());
    config.listen = "127.0.0.1:0".parse().unwrap();
    config.instance_id = "".into();
    assert!(config.validate().is_err());
    let fixture = json!({"schemaVersion":1,"instanceId":"fixture","listen":"127.0.0.1:0","tlsCertFile":directory.join("cert.pem"),"tlsKeyFile":directory.join("key.pem"),"connectionKeyFile":directory.join("connection-key"),"readyFile":directory.join("ready.json")});
    serde_json::from_value::<Config>(fixture)
        .unwrap()
        .validate()
        .unwrap();
    let mut headers = HeaderMap::new();
    headers.insert(AUTH_PROFILE, HeaderValue::from_static("codex-oauth"));
    headers.insert(UPSTREAM_IP, HeaderValue::from_static("104.18.2.1"));
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer provider-secret"),
    );
    assert!(!format!(
        "{:?}",
        Credentials::parse(&headers, Profile::Codex).unwrap()
    )
    .contains("provider-secret"));
}

#[tokio::test]
async fn tls_listener_publishes_authenticated_readiness_and_cleans_up() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "llmman-managed-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir(&directory).unwrap();
    let config = config(&directory);
    for (path, content) in [
        (
            &config.tls_cert_file,
            include_bytes!("test-cert.pem").as_slice(),
        ),
        (
            &config.tls_key_file,
            include_bytes!("test-key.pem").as_slice(),
        ),
        (&config.connection_key_file, b"connection-secret".as_slice()),
    ] {
        std::fs::write(path, content).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
    }
    // `serve_async` installs the provider before binding; the test stands
    // in for it.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = Listener::bind(config).await.unwrap();
    let ready: Value =
        serde_json::from_slice(&std::fs::read(directory.join("ready.json")).unwrap()).unwrap();
    assert_eq!(ready["schemaVersion"], 1);
    assert_eq!(ready["instanceId"], "fixture-instance");
    assert_eq!(ready["pid"], std::process::id());
    assert_eq!(
        ready["capabilities"],
        json!([CAPABILITY, CLAUDE_CAPABILITY])
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(directory.join("ready.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let url = ready["endpoint"].as_str().unwrap().to_string();
    let task = tokio::spawn(listener.serve(std::future::pending()));
    let client = reqwest::Client::builder()
        .no_proxy()
        .add_root_certificate(
            reqwest::Certificate::from_pem(include_bytes!("test-cert.pem")).unwrap(),
        )
        .build()
        .unwrap();
    let got: Value = client
        .get(format!("{url}/v1/capabilities"))
        .header(CONNECTION_KEY, "connection-secret")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(got, ready);
    assert!(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("{url}/v1/capabilities"))
            .header(CONNECTION_KEY, "connection-secret")
            .send()
            .await
            .is_err(),
        "untrusted certificates must fail"
    );
    task.abort();
    let _ = task.await;
    assert!(!directory.join("ready.json").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

/// A restarted successor publishes to the same path while this instance
/// drains; retracting on drop must not take its document down.
#[tokio::test]
async fn drop_retracts_only_its_own_ready_file() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "llmman-managed-drop-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir(&directory).unwrap();
    let config = config(&directory);
    for (path, content) in [
        (
            &config.tls_cert_file,
            include_bytes!("test-cert.pem").as_slice(),
        ),
        (
            &config.tls_key_file,
            include_bytes!("test-key.pem").as_slice(),
        ),
        (&config.connection_key_file, b"connection-secret".as_slice()),
    ] {
        std::fs::write(path, content).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
    }
    let ready_file = config.ready_file.clone();
    let _ = rustls::crypto::ring::default_provider().install_default();

    let listener = Listener::bind(config.clone()).await.unwrap();
    assert!(ready_file.exists());
    drop(listener);
    assert!(!ready_file.exists(), "own document is retracted");

    let listener = Listener::bind(config).await.unwrap();
    let successor = json!({"schemaVersion":1,"instanceId":"fixture-instance","pid":std::process::id().wrapping_add(1),"endpoint":"https://127.0.0.1:1","capabilities":[]});
    std::fs::write(&ready_file, serde_json::to_vec(&successor).unwrap()).unwrap();
    drop(listener);
    let left: Value = serde_json::from_slice(&std::fs::read(&ready_file).unwrap()).unwrap();
    assert_eq!(left, successor, "a successor's document is left alone");
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn upstream_clients_are_reused_per_profile_and_ip() {
    let state = state("http://127.0.0.1:9");
    let ip: IpAddr = "104.18.2.1".parse().unwrap();
    let other: IpAddr = "104.18.2.2".parse().unwrap();
    state.0.upstream_client(Profile::Codex, ip).unwrap();
    state.0.upstream_client(Profile::Codex, ip).unwrap();
    assert_eq!(state.0.clients.lock().unwrap().len(), 1);
    state.0.upstream_client(Profile::Claude, ip).unwrap();
    state.0.upstream_client(Profile::Codex, other).unwrap();
    assert_eq!(state.0.clients.lock().unwrap().len(), 3);
    for n in 0..=CLIENT_CACHE_LIMIT as u8 {
        state
            .0
            .upstream_client(Profile::Codex, IpAddr::from([104, 18, 3, n]))
            .unwrap();
    }
    assert!(
        state.0.clients.lock().unwrap().len() <= CLIENT_CACHE_LIMIT,
        "the cache is bounded"
    );
}

#[test]
fn upstream_ip_admission_rejects_special_use_addresses() {
    for ip in [
        "0.0.0.0",
        "10.1.2.3",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.0.0.1",
        "192.0.2.1",
        "192.168.1.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:127.0.0.1",
        "::ffff:104.18.2.1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "2001:db8::1",
        "2001::1",
    ] {
        assert!(!public_upstream_ip(&ip.parse().unwrap()), "{ip}");
    }
    for ip in ["104.18.2.1", "172.64.1.1", "2606:4700::1"] {
        assert!(public_upstream_ip(&ip.parse().unwrap()), "{ip}");
    }
}

#[tokio::test]
async fn upstream_ip_is_required_and_not_forwarded_as_a_header() {
    let managed = serve(router(state("http://127.0.0.1:9"))).await;
    for ip in [
        None,
        Some("127.0.0.1"),
        Some("104.18.2.1:443"),
        Some("[2606:4700::1]"),
        Some("2606:4700::1%en0"),
        Some("chatgpt.com"),
        Some("104.18.2.1,1.1.1.1"),
    ] {
        let mut request = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .post(format!("{}/v1/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
            .header(AUTH_PROFILE, "codex-oauth")
            .bearer_auth("provider-secret")
            .json(&json!({"model":MODEL}));
        if let Some(ip) = ip {
            request = request.header(UPSTREAM_IP, ip);
        }
        assert_eq!(
            request.send().await.unwrap().status(),
            StatusCode::BAD_REQUEST,
            "{ip:?}"
        );
    }
}

#[tokio::test]
async fn claude_preserves_native_messages_and_count_tokens() {
    let seen = Arc::new(Mutex::new(Vec::<(String, HeaderMap, Value)>::new()));
    let capture = seen.clone();
    let upstream = serve(Router::new().fallback(post(
        move |OriginalUri(uri): OriginalUri, headers: HeaderMap, Json(body): Json<Value>| {
            let capture = capture.clone();
            async move {
                let counting = uri.path().ends_with("count_tokens");
                capture
                    .lock()
                    .await
                    .push((uri.path().to_owned(), headers, body));
                (
                    [
                        ("request-id", "req-provider-secret"),
                        ("anthropic-ratelimit-input-tokens-remaining", "123"),
                        ("retry-after", "2"),
                        ("authorization", "do-not-relay"),
                    ],
                    Json(if counting {
                        json!({"input_tokens":42})
                    } else {
                        json!({"id":"msg_1","content":[{"type":"text","text":"Hello"}]})
                    }),
                )
            }
        },
    )))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    let body = json!({
        "model":CLAUDE_MODEL, "max_tokens":2048,
        "system":[{"type":"text","text":"Native system","cache_control":{"type":"ephemeral"}}],
        "messages":[
            {"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":"signed"},
                {"type":"tool_use","id":"tool_1","name":"read","input":{"path":"file"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":"data"}]}
        ],
        "tools":[{"name":"read","input_schema":{"type":"object"}}],
        "thinking":{"type":"enabled","budget_tokens":1024},
        "metadata":{"user_id":"user-1"}, "future_native_field":{"enabled":true}
    });
    for operation in ["messages", "messages/count_tokens"] {
        for beta in [
            None,
            Some("custom-beta"),
            Some("custom-beta, oauth-2025-04-20"),
        ] {
            let mut req = profile_request(&managed.url, operation, Profile::Claude)
                .header("x-api-key", "unrelated-key")
                .header("chatgpt-account-id", "unrelated-account")
                .header("openai-beta", "unrelated-beta")
                .header("session_id", "unrelated-session")
                .header("x-app", "cli")
                .header("user-agent", "native-client")
                .header("x-arbitrary", "do-not-relay");
            if let Some(beta) = beta {
                req = req
                    .header("anthropic-beta", beta)
                    .header("anthropic-version", "2099-01-01");
            }
            let response = req.json(&body).send().await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["request-id"], "req-<redacted>");
            assert_eq!(
                response.headers()["anthropic-ratelimit-input-tokens-remaining"],
                "123"
            );
            assert_eq!(response.headers()["retry-after"], "2");
            assert!(!response.headers().contains_key("authorization"));
            let result: Value = response.json().await.unwrap();
            if operation.ends_with("count_tokens") {
                assert_eq!(result, json!({"input_tokens":42}));
            } else {
                assert_eq!(result["id"], "msg_1");
            }
        }
    }
    let calls = seen.lock().await;
    assert_eq!(calls.len(), 6);
    for (i, (path, headers, forwarded)) in calls.iter().enumerate() {
        assert_eq!(
            path,
            if i < 3 {
                "/messages"
            } else {
                "/messages/count_tokens"
            }
        );
        assert_eq!(headers["authorization"], "Bearer provider-secret");
        assert_eq!(headers["x-app"], "cli");
        assert_eq!(headers["user-agent"], "native-client");
        assert_eq!(
            headers["anthropic-version"],
            if i % 3 == 0 {
                "2023-06-01"
            } else {
                "2099-01-01"
            }
        );
        assert_eq!(headers.get_all("anthropic-beta").iter().count(), 1);
        assert_eq!(
            headers["anthropic-beta"],
            match i % 3 {
                0 => CLAUDE_OAUTH_BETA,
                1 => "custom-beta,oauth-2025-04-20",
                _ => "custom-beta, oauth-2025-04-20",
            }
        );
        for name in [
            CONNECTION_KEY,
            AUTH_PROFILE,
            UPSTREAM_IP,
            "x-api-key",
            "chatgpt-account-id",
            "openai-beta",
            "originator",
            "session_id",
            "x-arbitrary",
        ] {
            assert!(!headers.contains_key(name), "{name}");
        }
        let mut expected = body.clone();
        expected["model"] = json!("claude-sonnet-4-5");
        assert_eq!(*forwarded, expected);
    }
}

#[tokio::test]
async fn claude_rejects_invalid_credentials_profiles_and_models_before_forwarding() {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let upstream = serve(Router::new().fallback(move || {
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            "unexpected"
        }
    }))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    for operation in ["messages", "messages/count_tokens"] {
        for (profile, model) in [
            (Profile::Codex, CLAUDE_MODEL),
            (Profile::Claude, MODEL),
            (Profile::Claude, "claude-sonnet-4-5"),
            (Profile::Claude, "llmman.provider/anthropic/"),
        ] {
            let response = profile_request(&managed.url, operation, profile)
                .json(&json!({"model":model}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        for bearer in [
            None,
            Some("Basic secret"),
            Some("Bearer llmman"),
            Some("Bearer a,b"),
        ] {
            let mut req = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap()
                .post(format!("{}/v1/{operation}", managed.url))
                .header(CONNECTION_KEY, "connection-secret")
                .header(AUTH_PROFILE, "claude-oauth")
                .header(UPSTREAM_IP, "104.18.2.1")
                .header("x-api-key", "must-not-be-used");
            if let Some(bearer) = bearer {
                req = req.header("authorization", bearer);
            }
            assert_eq!(
                req.json(&json!({"model":CLAUDE_MODEL}))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::UNAUTHORIZED
            );
        }
        for header in [
            AUTH_PROFILE,
            "authorization",
            "anthropic-version",
            "anthropic-beta",
            UPSTREAM_IP,
        ] {
            let req = profile_request(&managed.url, operation, Profile::Claude);
            let req = if header.starts_with("anthropic-") {
                req.header(header, "first")
            } else {
                req
            };
            assert_eq!(
                req.header(header, "duplicate")
                    .json(&json!({"model":CLAUDE_MODEL}))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
        let duplicate = format!("{{\"model\":\"{CLAUDE_MODEL}\",\"model\":\"other\"}}");
        assert_eq!(
            profile_request(&managed.url, operation, Profile::Claude)
                .body(duplicate)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    for operation in ["responses", "responses/compact"] {
        assert_eq!(
            profile_request(&managed.url, operation, Profile::Claude)
                .json(&json!({"model":CLAUDE_MODEL}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn concurrent_claude_and_codex_credentials_stay_request_local() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let upstream = serve(Router::new().fallback(post(move |headers: HeaderMap| {
        let barrier = barrier.clone();
        async move {
            barrier.wait().await;
            Json(json!({
                "bearer":headers["authorization"].to_str().unwrap(),
                "account":headers.get("chatgpt-account-id").map(|v|v.to_str().unwrap()),
                "claude_beta":headers.get("anthropic-beta").map(|v|v.to_str().unwrap()),
                "codex_beta":headers.get("openai-beta").map(|v|v.to_str().unwrap()),
            }))
        }
    })))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    let send = |profile: Profile, operation, model, token| {
        let url = managed.url.clone();
        async move {
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap()
                .post(format!("{url}/v1/{operation}"))
                .header(CONNECTION_KEY, "connection-secret")
                .header(AUTH_PROFILE, profile.name())
                .header(UPSTREAM_IP, "104.18.2.1")
                .bearer_auth(token)
                .header("chatgpt-account-id", "codex-account")
                .json(&json!({"model":model}))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };
    let (codex, claude) = tokio::join!(
        send(Profile::Codex, "responses", MODEL, "codex-secret"),
        send(Profile::Claude, "messages", CLAUDE_MODEL, "claude-secret"),
    );
    assert_eq!(
        codex,
        json!({"bearer":"Bearer codex-secret","account":"codex-account",
        "claude_beta":null,"codex_beta":"responses=experimental"})
    );
    assert_eq!(
        claude,
        json!({"bearer":"Bearer claude-secret","account":null,
        "claude_beta":CLAUDE_OAUTH_BETA,"codex_beta":null})
    );
}
