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
    let task = tokio::spawn(async move {
        axum::serve(
            socket,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap()
    });
    Server { url, task }
}
fn state(base: &str) -> ManagedState {
    ManagedState(Arc::new(ManagedInner {
        connection_auth: auth::Policy::with_keys(["connection-secret"]),
        read_timeout: None,
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
        .post(format!("{url}{}/{operation}", profile.route_prefix()))
        .header(CONNECTION_KEY, "connection-secret")
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
            .header("x-arbitrary", "future-provider-value")
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
        assert!(!headers.contains_key("originator"));
        assert!(!headers.contains_key("openai-beta"));
        assert_eq!(headers["x-arbitrary"], "future-provider-value");
        for name in [CONNECTION_KEY, UPSTREAM_IP, "x-api-key"] {
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
            .get(format!("{}/api/managed/capabilities", managed.url))
            .bearer_auth("connection-secret");
        if let Some(key) = key {
            req = req.header(CONNECTION_KEY, key);
        }
        assert_eq!(req.send().await.unwrap().status(), StatusCode::UNAUTHORIZED);
    }
    let ready: Value = client
        .get(format!("{}/api/managed/capabilities", managed.url))
        .header(CONNECTION_KEY, "connection-secret")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
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
    for authorization in [
        "Bearer llmman",
        "Bearer ",
        "Basic secret",
        "Bearer two tokens",
        "Bearer a,b",
    ] {
        let r = client
            .post(format!("{}/api/codex/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
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
                .post(format!("{url}/api/codex/responses"))
                .header(CONNECTION_KEY, "connection-secret")
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
        "3fff::",
        "3fff::1",
        "3fff:fff:ffff:ffff:ffff:ffff:ffff:ffff",
        "2001::1",
        "2002:7f00:1::",   // embeds 127.0.0.1
        "2002:0a00:1::",   // embeds 10.0.0.1
        "2002:6812:201::", // embeds public 104.18.2.1; all 6to4 is rejected
    ] {
        assert!(!public_upstream_ip(&ip.parse().unwrap()), "{ip}");
    }
    for ip in ["104.18.2.1", "172.64.1.1", "2606:4700::1", "3fff:1000::"] {
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
            .post(format!("{}/api/codex/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
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
                .header("chatgpt-account-id", "unrelated-account")
                .header("openai-beta", "unrelated-beta")
                .header("session_id", "unrelated-session")
                .header("x-app", "cli")
                .header("user-agent", "native-client")
                .header("x-arbitrary", "future-provider-value");
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
            UPSTREAM_IP,
            "x-api-key",
            "chatgpt-account-id",
            "originator",
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
                .post(format!("{}/api/anthropic/{operation}", managed.url))
                .header(CONNECTION_KEY, "connection-secret")
                .header(UPSTREAM_IP, "104.18.2.1");
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
                .post(format!("{url}{}/{operation}", profile.route_prefix()))
                .header(CONNECTION_KEY, "connection-secret")
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
        "claude_beta":null,"codex_beta":null})
    );
    assert_eq!(
        claude,
        json!({"bearer":"Bearer claude-secret","account":null,
        "claude_beta":CLAUDE_OAUTH_BETA,"codex_beta":null})
    );
}

#[test]
fn managed_routes_require_explicit_tls_and_enforced_daemon_auth() {
    let enabled = || crate::config::Managed {
        enabled: true,
        read_timeout: None,
    };
    assert!(routes(enabled(), auth::Policy::default(), true).is_err());
    assert!(routes(enabled(), auth::Policy::with_keys(["key"]).optional(), true).is_err());
    assert!(routes(enabled(), auth::Policy::with_keys(["key"]), false).is_err());
    let supported = routes(enabled(), auth::Policy::with_keys(["key"]), true);
    assert_eq!(supported.is_ok(), cfg!(unix));
    if !cfg!(unix) {
        assert!(supported
            .unwrap_err()
            .to_string()
            .contains("ACLs can be verified"));
    }
    assert!(routes(
        crate::config::Managed::default(),
        auth::Policy::default(),
        false
    )
    .is_ok());
}

#[tokio::test]
async fn peer_admission_normalizes_loopback_and_ignores_forwarded_headers() {
    for (peer, expected) in [
        (None, StatusCode::FORBIDDEN),
        (Some("192.0.2.1:1234"), StatusCode::FORBIDDEN),
        (Some("[::ffff:192.0.2.1]:1234"), StatusCode::FORBIDDEN),
        (Some("[::127.0.0.1]:1234"), StatusCode::FORBIDDEN),
        (Some("127.0.0.1:1234"), StatusCode::OK),
        (Some("[::1]:1234"), StatusCode::OK),
        (Some("[::ffff:127.0.0.1]:1234"), StatusCode::OK),
    ] {
        let peer = peer.map(|peer| peer.parse::<SocketAddr>().unwrap());
        let app = router(state("http://127.0.0.1:9")).layer(middleware::from_fn(
            move |mut req: Request, next: Next| async move {
                req.extensions_mut().remove::<ConnectInfo<SocketAddr>>();
                if let Some(peer) = peer {
                    req.extensions_mut().insert(ConnectInfo(peer));
                }
                next.run(req).await
            },
        ));
        let server = serve(app).await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("{}/api/managed/capabilities", server.url))
            .header(CONNECTION_KEY, "connection-secret")
            .header("x-forwarded-for", "127.0.0.1")
            .header("forwarded", "for=127.0.0.1")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "{peer:?}");
    }
}

#[tokio::test]
async fn duplicate_daemon_keys_are_rejected_and_provider_bearer_cannot_authenticate_daemon() {
    let server = serve(router(state("http://127.0.0.1:9"))).await;
    let response = request(&server.url, "responses")
        .header(CONNECTION_KEY, "connection-secret")
        .json(&json!({"model":MODEL}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    // Also fail closed if a future caller accidentally constructs an open policy.
    let mut managed = state("http://127.0.0.1:9");
    Arc::get_mut(&mut managed.0).unwrap().connection_auth = auth::Policy::default();
    let server = serve(router(managed)).await;
    let response = request(&server.url, "responses")
        .json(&json!({"model":MODEL}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn extension_headers_survive_and_hop_by_hop_fields_are_stripped_both_ways() {
    let upstream = serve(
        Router::new().fallback(post(|headers: HeaderMap| async move {
            assert_eq!(headers["originator"], "actual-client");
            assert_eq!(headers["openai-beta"], "future-beta");
            assert_eq!(headers.get_all("x-future-provider").iter().count(), 2);
            for name in [
                "x-drop",
                "proxy-authorization",
                "x-llmman-managed-key",
                "cookie",
                "x-api-key",
            ] {
                assert!(!headers.contains_key(name), "{name}");
            }
            assert_eq!(headers["authorization"], "Bearer provider-secret");
            (
                [
                    ("connection", "x-drop"),
                    ("x-drop", "private-hop"),
                    ("x-future-response", "preserved"),
                    ("proxy-authenticate", "private-hop"),
                    ("x-llmman-private", "private-hop"),
                    ("set-cookie", "private-cookie"),
                ],
                "ok",
            )
        })),
    )
    .await;
    let server = serve(router(state(&upstream.url))).await;
    let response = request(&server.url, "responses")
        .header("originator", "actual-client")
        .header("openai-beta", "future-beta")
        .header("x-future-provider", "a")
        .header("x-future-provider", "b")
        .header("connection", "X-Drop")
        .header("x-drop", "private-hop")
        .header("proxy-authorization", "private-hop")
        .header("cookie", "private-cookie")
        .header("x-llmman-managed-key", "obsolete-key")
        .json(&json!({"model":MODEL}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-future-response"], "preserved");
    for name in [
        "connection",
        "x-drop",
        "proxy-authenticate",
        "x-llmman-private",
        "set-cookie",
    ] {
        assert!(!response.headers().contains_key(name), "{name}");
    }
    assert_eq!(response.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn oversized_or_encoded_provider_errors_are_bounded_and_not_relayed() {
    for encoded in [false, true] {
        let upstream = serve(Router::new().fallback(move || async move {
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(
                    "content-encoding",
                    if encoded { "gzip" } else { "identity" },
                )
                .body(Body::from(if encoded {
                    "provider-secret".to_string()
                } else {
                    "x".repeat(ERROR_LIMIT + 1)
                }))
                .unwrap()
        }))
        .await;
        let server = serve(router(state(&upstream.url))).await;
        let response = request(&server.url, "responses")
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(!response.text().await.unwrap().contains("provider-secret"));
    }
}

#[tokio::test]
async fn configured_read_timeout_bounds_header_and_body_stalls_without_total_deadline() {
    use std::time::Duration;
    let upstream = serve(Router::new().route(
        "/responses",
        post(|| async {
            tokio::time::sleep(Duration::from_millis(150)).await;
            "finished"
        }),
    ))
    .await;
    for (timeout, expected) in [
        (None, StatusCode::OK),
        (Some(Duration::from_millis(20)), StatusCode::BAD_GATEWAY),
    ] {
        let mut managed = state(&upstream.url);
        Arc::get_mut(&mut managed.0).unwrap().read_timeout = timeout;
        let server = serve(router(managed)).await;
        let response = request(&server.url, "responses")
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }
    // A continuously active stream outlives the read timeout; a stalled one ends.
    for stall in [false, true] {
        let upstream = serve(Router::new().fallback(post(move || async move {
            Body::from_stream(futures::stream::unfold(0, move |n| async move {
                if n == 12 {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(if stall && n == 1 {
                    300
                } else {
                    10
                }))
                .await;
                Some((
                    Ok::<_, std::io::Error>(Bytes::from_static(b"chunk\n")),
                    n + 1,
                ))
            }))
        })))
        .await;
        let mut managed = state(&upstream.url);
        Arc::get_mut(&mut managed.0).unwrap().read_timeout = Some(Duration::from_millis(100));
        let server = serve(router(managed)).await;
        let response = request(&server.url, "responses")
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.is_err(), stall);
    }
}

/// A real TLS socket, also used to check the production transport's DNS pin
/// and provider hostname verification. No request URL override is needed.
async fn serve_tls(app: Router, name: &str) -> (Server, reqwest::Certificate, u16) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed([name.to_string()]).unwrap();
    let pem = cert.pem();
    let tls = axum_server::tls_rustls::RustlsConfig::from_pem(
        pem.as_bytes().to_vec(),
        signing_key.serialize_pem().into_bytes(),
    )
    .await
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let listener = listener.into_std().unwrap();
    let task = tokio::spawn(async move {
        axum_server::from_tcp_rustls(listener, tls)
            .unwrap()
            .serve(app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .unwrap();
    });
    (
        Server {
            url: format!("https://127.0.0.1:{port}"),
            task,
        },
        reqwest::Certificate::from_pem(pem.as_bytes()).unwrap(),
        port,
    )
}

#[tokio::test]
async fn production_transport_pins_ip_and_verifies_provider_tls_hostname() {
    for profile in [Profile::Codex, Profile::Claude] {
        let app = Router::new().fallback(|req: Request| async move {
            let authority = req
                .uri()
                .authority()
                .map(|a| a.as_str())
                .or_else(|| req.headers().get("host").and_then(|h| h.to_str().ok()))
                .unwrap();
            Json(json!({"host": authority}))
        });
        let (_server, cert, port) = serve_tls(app, profile.host()).await;
        let url = format!("https://{}:{port}/responses", profile.host());
        let client = upstream_client_builder(profile, "127.0.0.1".parse().unwrap(), None)
            .add_root_certificate(cert.clone())
            .build()
            .unwrap();
        let body: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["host"], format!("{}:{port}", profile.host()));
        let wrong_ip = upstream_client_builder(profile, "127.0.0.2".parse().unwrap(), None)
            .add_root_certificate(cert.clone())
            .connect_timeout(std::time::Duration::from_millis(100))
            .build()
            .unwrap();
        assert!(
            wrong_ip.get(&url).send().await.is_err(),
            "must not fall back to DNS"
        );
        let untrusted = upstream_client_builder(profile, "127.0.0.1".parse().unwrap(), None)
            .build()
            .unwrap();
        assert!(untrusted.get(&url).send().await.is_err());
        let (_wrong_server, wrong_cert, port) = serve_tls(Router::new(), "wrong.example").await;
        let wrong_name = upstream_client_builder(profile, "127.0.0.1".parse().unwrap(), None)
            .add_root_certificate(wrong_cert)
            .build()
            .unwrap();
        assert!(wrong_name
            .get(format!("https://{}:{port}/", profile.host()))
            .send()
            .await
            .is_err());
    }
}

#[tokio::test]
async fn shared_tls_listener_serves_public_and_authenticated_managed_routes() {
    let upstream = serve(
        Router::new().fallback(post(|headers: HeaderMap| async move {
            assert_eq!(headers["authorization"], "Bearer provider-secret");
            assert!(!headers.contains_key(CONNECTION_KEY));
            "forwarded"
        })),
    )
    .await;
    let mut app_state = super::super::test_support::test_state();
    Arc::get_mut(&mut app_state.0).unwrap().auth = auth::Policy::with_keys(["connection-secret"]);
    let app = super::super::build_router(app_state, false).merge(router(state(&upstream.url)));
    let (server, cert, _) = serve_tls(app, "127.0.0.1").await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .add_root_certificate(cert)
        .build()
        .unwrap();
    for path in ["/api/version", "/api/managed/capabilities"] {
        assert_eq!(
            client
                .get(format!("{}{path}", server.url))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let response = client
            .get(format!("{}{path}", server.url))
            .header(CONNECTION_KEY, "connection-secret")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        if path.ends_with("capabilities") {
            assert_eq!(
                response.json::<Value>().await.unwrap()["capabilities"],
                json!([CAPABILITY, CLAUDE_CAPABILITY])
            );
        }
    }
    let response = client
        .post(format!("{}/api/codex/responses", server.url))
        .header(CONNECTION_KEY, "connection-secret")
        .header(UPSTREAM_IP, "104.18.2.1")
        .bearer_auth("provider-secret")
        .json(&json!({"model":MODEL}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), "forwarded");
}

#[tokio::test]
async fn escaped_credentials_are_redacted_from_json_error_strings_and_keys() {
    for (token, account) in [
        ("token\"part", "account\\part"),
        ("token\\part/one", "account\"part/two"),
    ] {
        let error_body = json!({"error": {
            "message": format!("upstream {token} for {account}"),
            "details": [{token: account}],
            "code": 42,
        }})
        .to_string()
        .replace('/', "\\/")
        .replace("part", "\\u0070art");
        let upstream = serve(Router::new().fallback(move || {
            let body = error_body.clone();
            async move {
                (
                    StatusCode::UNAUTHORIZED,
                    [("content-type", "application/json")],
                    body,
                )
            }
        }))
        .await;
        let managed = serve(router(state(&upstream.url))).await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .post(format!("{}/api/codex/responses", managed.url))
            .header(CONNECTION_KEY, "connection-secret")
            .header(UPSTREAM_IP, "104.18.2.1")
            .bearer_auth(token)
            .header("chatgpt-account-id", account)
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            json!({"error": {
                "message": "upstream <redacted> for <redacted>",
                "details": [{"<redacted>": "<redacted>"}],
                "code": 42,
            }})
        );
    }
}

#[tokio::test]
async fn content_coding_is_validated_before_rewriting_requests_or_redacting_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let upstream = serve(Router::new().fallback(post(move |headers: HeaderMap| {
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            assert!(!headers.contains_key("content-encoding"));
            assert_eq!(headers["content-type"], "application/json");
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("content-encoding", "Identity, IDENTITY")],
                "provider-secret",
            )
        }
    })))
    .await;
    let managed = serve(router(state(&upstream.url))).await;
    for encoding in ["gzip", "br", "identity, gzip", "", "identity,"] {
        let response = request(&managed.url, "responses")
            .header("content-encoding", encoding)
            .json(&json!({"model":MODEL}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "{encoding}"
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for encoding in [
        None,
        Some("identity"),
        Some("Identity"),
        Some("IDENTITY, identity"),
    ] {
        let mut req = request(&managed.url, "responses").json(&json!({"model":MODEL}));
        if let Some(encoding) = encoding {
            req = req.header("content-encoding", encoding);
        }
        let response = req.send().await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.text().await.unwrap(), "<redacted>");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 4);
}
