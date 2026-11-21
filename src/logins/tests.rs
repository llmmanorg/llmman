//! Logins and refreshes against a scripted local server. Nothing here
//! reaches the network: every endpoint is `Endpoints::all_at(<127.0.0.1>)`,
//! and the clock is the fake one, so a poll that waits minutes takes none.

use super::device::fake::FakeClock;
use super::import::tests::jwt;
use super::*;
use crate::credentials::{ReloginRequired, Status};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::Json;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// A scripted server
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

impl Seen {
    fn form(&self) -> HashMap<String, String> {
        reqwest::Url::parse(&format!("http://x/?{}", self.body))
            .unwrap()
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect()
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap()
    }
}

#[derive(Default)]
struct Fake {
    script: Mutex<HashMap<String, VecDeque<(u16, Value)>>>,
    seen: Mutex<Vec<Seen>>,
    delay: Mutex<HashMap<String, Duration>>,
}

impl Fake {
    /// `path` answers these in order, then the last one for good.
    fn on(&self, path: &str, replies: Vec<(u16, Value)>) {
        self.script
            .lock()
            .unwrap()
            .insert(path.to_string(), replies.into());
    }

    fn hits(&self, path: &str) -> Vec<Seen> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.path == path)
            .cloned()
            .collect()
    }
}

async fn respond(State(fake): State<Arc<Fake>>, req: Request) -> (StatusCode, Json<Value>) {
    let path = req.uri().path().to_string();
    let headers = req
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let body = axum::body::to_bytes(req.into_body(), 1 << 20)
        .await
        .unwrap();
    fake.seen.lock().unwrap().push(Seen {
        path: path.clone(),
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    });
    let delay = fake.delay.lock().unwrap().get(&path).copied();
    if let Some(delay) = delay {
        tokio::time::sleep(delay).await;
    }
    let mut script = fake.script.lock().unwrap();
    let Some(queue) = script.get_mut(&path) else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "unscripted"})));
    };
    let (status, body) = if queue.len() > 1 {
        queue.pop_front().unwrap()
    } else {
        queue.front().cloned().unwrap()
    };
    // `{{name}}` in a scripted body stands for that field of the request's
    // form, for a server that echoes what it was sent.
    let mut text = body.to_string();
    if text.contains("{{") {
        let sent = fake.seen.lock().unwrap().last().cloned();
        for (k, v) in sent.map(|s| s.form()).unwrap_or_default() {
            text = text.replace(&format!("{{{{{k}}}}}"), &v);
        }
    }
    let body = serde_json::from_str(&text).unwrap_or(body);
    (StatusCode::from_u16(status).unwrap(), Json(body))
}

struct Server {
    origin: String,
    fake: Arc<Fake>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn server() -> Server {
    let fake = Arc::new(Fake::default());
    let app = axum::Router::new()
        .fallback(respond)
        .with_state(fake.clone());
    let (origin_tx, origin_rx) = std::sync::mpsc::channel();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let thread = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                origin_tx
                    .send(format!("http://{}", listener.local_addr().unwrap()))
                    .unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = stopped.await;
                    })
                    .await
                    .unwrap();
            });
    });
    Server {
        origin: origin_rx.recv().unwrap(),
        fake,
        stop: Some(stop),
        thread: Some(thread),
    }
}

/// The slept intervals of a shared fake clock.
impl Clock for Rc<FakeClock> {
    fn sleep(&self, d: Duration) {
        (**self).sleep(d)
    }
    fn elapsed(&self) -> Duration {
        (**self).elapsed()
    }
}

struct Rig {
    server: Server,
    clock: Rc<FakeClock>,
    store: Store,
    _dir: tempfile::TempDir,
    opened: Rc<RefCell<Vec<String>>>,
}

impl Rig {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            server: server(),
            clock: Rc::new(FakeClock::new()),
            store: Store::at(dir.path().join("credentials.json")),
            _dir: dir,
            opened: Rc::default(),
        }
    }

    fn fake(&self) -> &Fake {
        &self.server.fake
    }

    fn login(
        &self,
        provider: &str,
        opts: &Options,
        input: &dyn Fn(&str, bool) -> anyhow::Result<String>,
    ) -> anyhow::Result<()> {
        let opened = self.opened.clone();
        let ctx = Ctx {
            store: self.store.clone(),
            ep: Endpoints::all_at(&self.server.origin),
            http: Http::allowing_http(),
            clock: Box::new(self.clock.clone()),
            input,
            open: &move |url| opened.borrow_mut().push(url.to_string()),
        };
        login_with(&ctx, provider, opts)
    }

    fn live(&self) -> Live {
        Live::against(Endpoints::all_at(&self.server.origin))
    }
}

fn no_input(_: &str, _: bool) -> anyhow::Result<String> {
    panic!("this login asks for nothing")
}

fn method(name: &str) -> Options {
    Options {
        method: Some(name.into()),
        ..Options::default()
    }
}

fn secs(n: u64) -> Duration {
    Duration::from_secs(n)
}

fn oauth_in(rig: &Rig, id: &str) -> OAuth {
    match rig.store.get(id) {
        Some(Credential::OAuth(o)) => o,
        other => panic!("expected an OAuth credential, got {other:?}"),
    }
}

fn chatgpt_jwt(account: &str, exp: u64) -> String {
    jwt(&json!({
        "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }))
}

// ---------------------------------------------------------------------------
// ChatGPT
// ---------------------------------------------------------------------------

#[test]
fn chatgpt_device_login_waits_through_403_and_404_then_exchanges_the_code() {
    let rig = Rig::new();
    let fake = rig.fake();
    fake.on(
        "/api/accounts/deviceauth/usercode",
        vec![(
            200,
            json!({"device_auth_id": "dev-1", "usercode": "ABCD-1234", "interval": 2}),
        )],
    );
    fake.on(
        "/api/accounts/deviceauth/token",
        vec![
            (403, json!({"error": "deviceauth_authorization_unknown"})),
            (404, json!({})),
            (
                200,
                json!({"authorization_code": "auth-code", "code_verifier": "the-verifier"}),
            ),
        ],
    );
    let id_token = chatgpt_jwt("acct-9", 2_000_000_000);
    fake.on(
        "/oauth/token",
        vec![(
            200,
            json!({"access_token": "access-1", "refresh_token": "refresh-1",
                   "id_token": id_token, "expires_in": 3600}),
        )],
    );

    rig.login("openai-chatgpt", &Options::default(), &no_input)
        .unwrap();

    let o = oauth_in(&rig, "openai-chatgpt");
    assert_eq!(
        (o.access.as_str(), o.refresh.as_str()),
        ("access-1", "refresh-1")
    );
    assert_eq!(o.flow, FLOW_CHATGPT);
    assert_eq!(o.metadata["account_id"], "acct-9");
    assert!(o.expires_ms > now_ms() + 3_000_000);
    assert_eq!(*rig.clock.slept.borrow(), [secs(2), secs(2), secs(2)]);

    let start = &fake.hits("/api/accounts/deviceauth/usercode")[0];
    assert_eq!(start.json()["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    let poll = &fake.hits("/api/accounts/deviceauth/token")[0];
    assert_eq!(poll.json()["device_auth_id"], "dev-1");
    assert_eq!(poll.json()["user_code"], "ABCD-1234");
    let exchange = fake.hits("/oauth/token")[0].form();
    assert_eq!(exchange["grant_type"], "authorization_code");
    assert_eq!(exchange["code"], "auth-code");
    assert_eq!(exchange["code_verifier"], "the-verifier");
    assert_eq!(exchange["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    assert!(exchange["redirect_uri"].ends_with("/deviceauth/callback"));
    // The page the user is sent to is auth.openai.com's, here the fake's.
    assert_eq!(
        rig.opened.borrow()[0],
        format!("{}/codex/device", rig.server.origin)
    );
}

#[test]
fn chatgpt_device_login_reports_a_server_without_device_codes() {
    let rig = Rig::new();
    rig.fake()
        .on("/api/accounts/deviceauth/usercode", vec![(404, json!({}))]);
    let err = rig
        .login("openai-chatgpt", &Options::default(), &no_input)
        .unwrap_err();
    assert!(err.to_string().contains("not enabled"), "{err}");
    assert!(rig.store.get("openai-chatgpt").is_none());
}

#[test]
fn chatgpt_refresh_rotates_the_token_and_keeps_the_account() {
    let rig = Rig::new();
    let id_token = chatgpt_jwt("acct-new", 2_000_000_000);
    rig.fake().on(
        "/oauth/token",
        vec![(
            200,
            json!({"access_token": "access-2", "refresh_token": "refresh-2",
                   "id_token": id_token, "expires_in": 7200}),
        )],
    );
    rig.store
        .put(
            "openai-chatgpt",
            Credential::OAuth(OAuth {
                flow: FLOW_CHATGPT.into(),
                access: "access-1".into(),
                refresh: "refresh-1".into(),
                expires_ms: now_ms() + 10_000, // inside the skew
                metadata: BTreeMap::from([("account_id".into(), "acct-old".into())]),
            }),
        )
        .unwrap();

    let got = rig
        .store
        .resolve("openai-chatgpt", &rig.live(), now_ms())
        .unwrap()
        .unwrap();
    assert_eq!(got.secret.expose(), "access-2");
    assert_eq!(got.metadata["account_id"], "acct-new");

    let form = rig.fake().hits("/oauth/token")[0].form();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["refresh_token"], "refresh-1");
    assert_eq!(form["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann");
    let saved = oauth_in(&rig, "openai-chatgpt");
    assert_eq!(
        saved.refresh, "refresh-2",
        "the rotated token is what is on disk"
    );
}

#[test]
fn a_refresh_that_keeps_the_refresh_token_keeps_it() {
    let rig = Rig::new();
    rig.fake().on(
        "/oauth2/token",
        vec![(200, json!({"access_token": "access-2", "expires_in": 60}))],
    );
    rig.store
        .put(
            "xai-supergrok",
            Credential::OAuth(OAuth {
                flow: FLOW_XAI.into(),
                access: "a".into(),
                refresh: "refresh-1".into(),
                expires_ms: 1,
                metadata: BTreeMap::new(),
            }),
        )
        .unwrap();
    rig.store
        .resolve("xai-supergrok", &rig.live(), now_ms())
        .unwrap();
    assert_eq!(oauth_in(&rig, "xai-supergrok").refresh, "refresh-1");
    let form = rig.fake().hits("/oauth2/token")[0].form();
    assert_eq!(form["client_id"], "b1a00492-073a-47ea-816f-4c329264a828");
    assert_eq!(form["grant_type"], "refresh_token");
}

#[test]
fn concurrent_requests_make_one_refresh_request() {
    let rig = Rig::new();
    rig.fake().on(
        "/oauth/token",
        vec![(
            200,
            json!({"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600}),
        )],
    );
    rig.fake()
        .delay
        .lock()
        .unwrap()
        .insert("/oauth/token".into(), Duration::from_millis(200));
    rig.store
        .put(
            "openai-chatgpt",
            Credential::OAuth(OAuth {
                flow: FLOW_CHATGPT.into(),
                access: "access-1".into(),
                refresh: "refresh-1".into(),
                expires_ms: 1,
                metadata: BTreeMap::new(),
            }),
        )
        .unwrap();
    let origin = rig.server.origin.clone();
    let path = rig.store.path().to_path_buf();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (origin, path) = (origin.clone(), path.clone());
            std::thread::spawn(move || {
                Store::at(path)
                    .resolve(
                        "openai-chatgpt",
                        &Live::against(Endpoints::all_at(&origin)),
                        now_ms(),
                    )
                    .unwrap()
                    .unwrap()
                    .secret
                    .expose()
                    .to_string()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), "access-2");
    }
    assert_eq!(rig.fake().hits("/oauth/token").len(), 1);
}

#[test]
fn a_refused_refresh_means_log_in_again_and_a_busy_server_does_not() {
    let rig = Rig::new();
    let expired = Credential::OAuth(OAuth {
        flow: FLOW_CHATGPT.into(),
        access: "a".into(),
        refresh: "spent".into(),
        expires_ms: 1,
        metadata: BTreeMap::new(),
    });
    rig.store.put("openai-chatgpt", expired.clone()).unwrap();

    rig.fake().on(
        "/oauth/token",
        vec![(
            400,
            json!({"error": "refresh_token_reused", "error_description": "\u{1b}[31mspent"}),
        )],
    );
    let err = rig
        .store
        .resolve("openai-chatgpt", &rig.live(), now_ms())
        .unwrap_err();
    let relogin = err.downcast_ref::<ReloginRequired>().expect("relogin");
    assert!(relogin.reason.contains("refresh_token_reused"), "{relogin}");
    assert!(!err.to_string().contains('\u{1b}'));
    assert_eq!(rig.store.get("openai-chatgpt"), Some(expired.clone()));

    rig.fake().on("/oauth/token", vec![(503, json!({}))]);
    let err = rig
        .store
        .resolve("openai-chatgpt", &rig.live(), now_ms())
        .unwrap_err();
    assert!(err.downcast_ref::<ReloginRequired>().is_none(), "{err}");
    assert_eq!(rig.store.get("openai-chatgpt"), Some(expired));
}

#[test]
fn a_claude_login_is_refreshed_with_json() {
    let rig = Rig::new();
    rig.fake().on(
        "/v1/oauth/token",
        vec![(
            200,
            json!({"access_token": "sk-ant-oat01-new", "refresh_token": "r2", "expires_in": 28800}),
        )],
    );
    rig.store
        .put(
            "anthropic",
            Credential::OAuth(OAuth {
                flow: FLOW_CLAUDE.into(),
                access: "sk-ant-oat01-old".into(),
                refresh: "r1".into(),
                expires_ms: 1,
                metadata: BTreeMap::new(),
            }),
        )
        .unwrap();
    let got = rig
        .store
        .resolve("anthropic", &rig.live(), now_ms())
        .unwrap()
        .unwrap();
    assert_eq!(got.secret.expose(), "sk-ant-oat01-new");
    let body = rig.fake().hits("/v1/oauth/token")[0].json();
    assert_eq!(body["grant_type"], "refresh_token");
    assert_eq!(body["refresh_token"], "r1");
    assert_eq!(body["client_id"], "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
}

#[test]
fn a_token_is_never_sent_over_plain_http() {
    // `Live::new()` is the production constructor: https only.
    let live = Live {
        http: Http::new(),
        ep: Endpoints::all_at("http://127.0.0.1:9"),
    };
    let old = OAuth {
        flow: FLOW_CHATGPT.into(),
        access: "a".into(),
        refresh: "r".into(),
        expires_ms: 1,
        metadata: BTreeMap::new(),
    };
    match live.refresh(&old) {
        Err(RefreshError::Transient(why)) => assert!(why.contains("plain http"), "{why}"),
        other => panic!("{other:?}"),
    }
    // And a login's own requests too.
    let http = Http::new();
    assert!(http.post_form("http://127.0.0.1:9/x", &[], &[]).is_err());
    assert!(http.get("http://127.0.0.1:9/x", &[]).is_err());
}

#[test]
fn an_unknown_login_cannot_be_refreshed() {
    let old = OAuth {
        flow: "mystery".into(),
        access: "a".into(),
        refresh: "r".into(),
        expires_ms: 1,
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        Live::new().refresh(&old),
        Err(RefreshError::Rejected(_))
    ));
}

// ---------------------------------------------------------------------------
// GitHub Copilot
// ---------------------------------------------------------------------------

fn copilot_device_reply(origin: &str) -> Value {
    json!({"device_code": "dc-1", "user_code": "WXYZ-0001",
           "verification_uri": format!("{origin}/login/device"), "expires_in": 900, "interval": 5})
}

#[test]
fn copilot_login_polls_through_pending_and_slow_down_and_learns_the_endpoint() {
    let rig = Rig::new();
    let fake = rig.fake();
    fake.on(
        "/login/device/code",
        vec![(200, copilot_device_reply(&rig.server.origin))],
    );
    fake.on(
        "/login/oauth/access_token",
        vec![
            (200, json!({"error": "authorization_pending"})),
            (200, json!({"error": "slow_down", "interval": 12})),
            (200, json!({"error": "authorization_pending"})),
            (
                200,
                json!({"access_token": "gho_token", "token_type": "bearer"}),
            ),
        ],
    );
    fake.on(
        "/copilot_internal/user",
        vec![(
            200,
            json!({"chat_enabled": true,
                   "endpoints": {"api": "https://api.individual.githubcopilot.com/"}}),
        )],
    );

    rig.login("github-copilot", &Options::default(), &no_input)
        .unwrap();

    let o = oauth_in(&rig, "github-copilot");
    assert_eq!(o.access, "gho_token");
    assert_eq!(o.expires_ms, 0, "a GitHub device token does not expire");
    assert_eq!(
        o.metadata["api_endpoint"],
        "https://api.individual.githubcopilot.com"
    );
    assert_eq!(
        *rig.clock.slept.borrow(),
        [secs(5), secs(5), secs(12), secs(12)]
    );

    let ask = fake.hits("/login/device/code")[0].form();
    assert_eq!(ask["client_id"], "Iv1.b507a08c87ecfe98");
    assert_eq!(ask["scope"], "read:user");
    let poll = fake.hits("/login/oauth/access_token")[0].form();
    assert_eq!(poll["device_code"], "dc-1");
    assert_eq!(
        poll["grant_type"],
        "urn:ietf:params:oauth:grant-type:device_code"
    );
    assert_eq!(
        fake.hits("/copilot_internal/user")[0].headers["authorization"],
        "Bearer gho_token"
    );
    // Not exchanged for anything: there is no second token request.
    assert_eq!(fake.hits("/copilot_internal/v2/token").len(), 0);
}

#[test]
fn copilot_login_stops_when_the_user_says_no_or_the_code_expires() {
    for (error, expect) in [("access_denied", "denied"), ("expired_token", "expired")] {
        let rig = Rig::new();
        rig.fake().on(
            "/login/device/code",
            vec![(200, copilot_device_reply(&rig.server.origin))],
        );
        rig.fake().on(
            "/login/oauth/access_token",
            vec![(200, json!({ "error": error }))],
        );
        let err = rig
            .login("github-copilot", &Options::default(), &no_input)
            .unwrap_err();
        assert!(err.to_string().contains(expect), "{err}");
        assert!(rig.store.get("github-copilot").is_none());
    }
}

#[test]
fn copilot_login_times_out_when_nobody_approves() {
    let rig = Rig::new();
    let mut reply = copilot_device_reply(&rig.server.origin);
    reply["expires_in"] = json!(30);
    rig.fake().on("/login/device/code", vec![(200, reply)]);
    rig.fake().on(
        "/login/oauth/access_token",
        vec![(200, json!({"error": "authorization_pending"}))],
    );
    let err = rig
        .login("github-copilot", &Options::default(), &no_input)
        .unwrap_err();
    assert!(err.to_string().contains("timed out"), "{err}");
    assert_eq!(rig.clock.slept.borrow().len(), 6);
}

#[test]
fn copilot_login_refuses_an_account_without_copilot_and_an_untrusted_endpoint() {
    let rig = Rig::new();
    let fake = rig.fake();
    fake.on(
        "/login/device/code",
        vec![(200, copilot_device_reply(&rig.server.origin))],
    );
    fake.on(
        "/login/oauth/access_token",
        vec![(200, json!({"access_token": "gho_t"}))],
    );
    fake.on(
        "/copilot_internal/user",
        vec![(200, json!({"chat_enabled": false}))],
    );
    let err = rig
        .login("github-copilot", &Options::default(), &no_input)
        .unwrap_err();
    assert!(
        err.to_string().contains("no GitHub Copilot access"),
        "{err}"
    );
    assert!(rig.store.get("github-copilot").is_none());

    // An endpoint outside GitHub's hosts is not remembered: the token
    // would be sent there.
    fake.on(
        "/copilot_internal/user",
        vec![(
            200,
            json!({"endpoints": {"api": "https://evil.example.com"}}),
        )],
    );
    rig.login("github-copilot", &Options::default(), &no_input)
        .unwrap();
    assert!(!oauth_in(&rig, "github-copilot")
        .metadata
        .contains_key("api_endpoint"));

    // A lookup that fails is not a denial.
    fake.on("/copilot_internal/user", vec![(500, json!({}))]);
    rig.login("github-copilot", &Options::default(), &no_input)
        .unwrap();
}

#[test]
fn copilot_login_refuses_a_verification_page_that_is_not_githubs() {
    let rig = Rig::new();
    let mut reply = copilot_device_reply(&rig.server.origin);
    reply["verification_uri"] = json!("https://github.com/login/device?next=https://evil.example");
    rig.fake()
        .on("/login/device/code", vec![(200, reply.clone())]);
    // The query is dropped and the page rebuilt, so this one is fine…
    rig.fake().on(
        "/login/oauth/access_token",
        vec![(200, json!({"access_token": "t"}))],
    );
    rig.fake()
        .on("/copilot_internal/user", vec![(200, json!({}))]);
    rig.login("github-copilot", &Options::default(), &no_input)
        .unwrap();
    assert_eq!(
        rig.opened.borrow()[0],
        format!("{}/login/device", rig.server.origin)
    );

    // …and another path is not.
    reply["verification_uri"] = json!("https://github.com/evil/page");
    rig.fake().on("/login/device/code", vec![(200, reply)]);
    let err = rig
        .login("github-copilot", &Options::default(), &no_input)
        .unwrap_err();
    assert!(
        err.to_string().contains("unexpected verification URL"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// xAI
// ---------------------------------------------------------------------------

#[test]
fn xai_login_handles_every_device_flow_outcome() {
    let xai_start = |origin: &str| {
        json!({"device_code": "xdc", "user_code": "ABCD-EFGH",
               "verification_uri": format!("{origin}/device"), "expires_in": 300, "interval": 5})
    };
    // Pending, slow down, then tokens.
    let rig = Rig::new();
    rig.fake().on(
        "/oauth2/device/code",
        vec![(200, xai_start(&rig.server.origin))],
    );
    rig.fake().on(
        "/oauth2/token",
        vec![
            (400, json!({"error": "authorization_pending"})),
            (400, json!({"error": "slow_down"})),
            (
                200,
                json!({"access_token": "xa", "refresh_token": "xr", "expires_in": 3600}),
            ),
        ],
    );
    rig.login("xai-supergrok", &Options::default(), &no_input)
        .unwrap();
    let o = oauth_in(&rig, "xai-supergrok");
    assert_eq!(
        (o.access.as_str(), o.refresh.as_str(), o.flow.as_str()),
        ("xa", "xr", FLOW_XAI)
    );
    assert_eq!(*rig.clock.slept.borrow(), [secs(5), secs(5), secs(10)]);
    let ask = rig.fake().hits("/oauth2/device/code")[0].form();
    assert_eq!(ask["client_id"], "b1a00492-073a-47ea-816f-4c329264a828");
    assert!(ask["scope"].contains("grok-cli:access"));
    let poll = rig.fake().hits("/oauth2/token")[0].form();
    assert_eq!(poll["device_code"], "xdc");
    assert_eq!(
        poll["grant_type"],
        "urn:ietf:params:oauth:grant-type:device_code"
    );

    // Denied, expired and a token reply without a refresh token.
    for (reply, expect) in [
        (json!({"error": "access_denied"}), "denied"),
        (json!({"error": "authorization_denied"}), "denied"),
        (json!({"error": "expired_token"}), "expired"),
    ] {
        let rig = Rig::new();
        rig.fake().on(
            "/oauth2/device/code",
            vec![(200, xai_start(&rig.server.origin))],
        );
        rig.fake().on("/oauth2/token", vec![(400, reply)]);
        let err = rig
            .login("xai-supergrok", &Options::default(), &no_input)
            .unwrap_err();
        assert!(err.to_string().contains(expect), "{err}");
    }
    let rig = Rig::new();
    rig.fake().on(
        "/oauth2/device/code",
        vec![(200, xai_start(&rig.server.origin))],
    );
    rig.fake()
        .on("/oauth2/token", vec![(200, json!({"access_token": "xa"}))]);
    let err = rig
        .login("xai-supergrok", &Options::default(), &no_input)
        .unwrap_err();
    assert!(err.to_string().contains("no refresh token"), "{err}");
}

// ---------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------

#[test]
fn openrouter_login_returns_through_the_loopback_and_stores_the_key() {
    let rig = Rig::new();
    rig.fake().on(
        "/api/v1/auth/keys",
        vec![(200, json!({"key": "sk-or-v1-minted", "user_id": "u1"}))],
    );
    let opened_url: Rc<RefCell<Option<reqwest::Url>>> = Rc::default();
    let seen_url = opened_url.clone();
    let dir = tempfile::tempdir().unwrap();
    let ctx = Ctx {
        store: Store::at(dir.path().join("c.json")),
        ep: Endpoints::all_at(&rig.server.origin),
        http: Http::allowing_http(),
        clock: Box::new(rig.clock.clone()),
        input: &no_input,
        // The "browser": comes back to the callback_url with a code.
        open: &move |url| {
            let url = reqwest::Url::parse(url).unwrap();
            let callback = url
                .query_pairs()
                .find(|(k, _)| k == "callback_url")
                .map(|(_, v)| reqwest::Url::parse(&v).unwrap())
                .unwrap();
            *seen_url.borrow_mut() = Some(url);
            let port = callback.port().unwrap();
            let target = format!(
                "{}?{}&code=the-code",
                callback.path(),
                callback.query().unwrap()
            );
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
                write!(s, "GET {target} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n").unwrap();
                let mut out = String::new();
                let _ = s.read_to_string(&mut out);
            });
        },
    };
    login_with(&ctx, "openrouter", &Options::default()).unwrap();

    let stored = ctx.store.get("openrouter").unwrap();
    assert_eq!(
        stored,
        Credential::ApiKey {
            key: "sk-or-v1-minted".into(),
            flow: Some(FLOW_OPENROUTER.into())
        }
    );
    let url = opened_url.borrow().clone().unwrap();
    assert_eq!(url.path(), "/auth");
    let q: HashMap<_, _> = url.query_pairs().into_owned().collect();
    assert_eq!(q["code_challenge_method"], "S256");
    assert!(q["callback_url"].contains("/openrouter-oauth/callback?state="));
    let body = rig.fake().hits("/api/v1/auth/keys")[0].json();
    assert_eq!(body["code"], "the-code");
    assert_eq!(body["code_challenge_method"], "S256");
    // The verifier sent is the one the challenge in the URL commits to.
    let verifier = body["code_verifier"].as_str().unwrap();
    assert_eq!(
        pkce::Pkce::from_verifier(verifier.into()).challenge,
        q["code_challenge"]
    );
}

#[test]
fn openrouter_login_reports_an_exchange_without_a_key() {
    let rig = Rig::new();
    rig.fake().on("/api/v1/auth/keys", vec![(200, json!({}))]);
    let dir = tempfile::tempdir().unwrap();
    let ctx = Ctx {
        store: Store::at(dir.path().join("c.json")),
        ep: Endpoints::all_at(&rig.server.origin),
        http: Http::allowing_http(),
        clock: Box::new(rig.clock.clone()),
        input: &no_input,
        open: &|url| {
            let url = reqwest::Url::parse(url).unwrap();
            let callback = url
                .query_pairs()
                .find(|(k, _)| k == "callback_url")
                .map(|(_, v)| reqwest::Url::parse(&v).unwrap())
                .unwrap();
            let port = callback.port().unwrap();
            let target = format!("{}?{}&code=c", callback.path(), callback.query().unwrap());
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
                write!(s, "GET {target} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n").unwrap();
                let _ = s.read_to_string(&mut String::new());
            });
        },
    };
    let err = login_with(&ctx, "openrouter", &Options::default()).unwrap_err();
    assert!(err.to_string().contains("no API key"), "{err}");
    assert!(ctx.store.get("openrouter").is_none());
}

// ---------------------------------------------------------------------------
// Pasted tokens, imports and the generic methods
// ---------------------------------------------------------------------------

#[test]
fn a_setup_token_must_look_like_one() {
    let rig = Rig::new();
    let good = format!("sk-ant-oat01-{}", "x".repeat(80));
    let ask = |_: &str, secret: bool| {
        assert!(secret, "a token is not echoed");
        Ok(good.clone())
    };
    rig.login("anthropic", &method("setup-token"), &ask)
        .unwrap();
    assert_eq!(
        rig.store.get("anthropic"),
        Some(Credential::Token {
            token: good.clone(),
            expires_ms: None,
            flow: Some(FLOW_SETUP_TOKEN.into())
        })
    );
    for bad in ["sk-ant-api03-abc", "sk-ant-oat01-short", ""] {
        let err = rig
            .login(
                "anthropic",
                &Options {
                    method: Some("setup-token".into()),
                    token: Some(bad.into()),
                    ..Options::default()
                },
                &no_input,
            )
            .unwrap_err();
        assert!(!err.to_string().contains(bad) || bad.is_empty(), "{err}");
    }
}

#[test]
fn importing_a_cli_login_reads_in_place_unless_asked_to_copy() {
    let rig = Rig::new();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("credentials.json");
    std::fs::write(
        &file,
        r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-live","refreshToken":"rr","expiresAt":4102444800000}}"#,
    )
    .unwrap();
    let from = |copy| Options {
        method: Some("import-claude".into()),
        file: Some(file.to_string_lossy().into_owned()),
        copy,
        ..Options::default()
    };

    // Default: a pointer. No token is copied into llmman's store.
    rig.login("anthropic", &from(false), &no_input).unwrap();
    let Some(Credential::External {
        source: ExternalSource::CliLogin { cli, path },
    }) = rig.store.get("anthropic")
    else {
        panic!("a pointer to the CLI's file");
    };
    assert_eq!(cli, CliLogin::Claude);
    assert_eq!(path.as_deref(), Some(file.to_str().unwrap()));
    let stored = std::fs::read_to_string(rig.store.path()).unwrap();
    assert!(
        !stored.contains("sk-ant-oat01-live") && !stored.contains("\"rr\""),
        "{stored}"
    );
    // …and it is read fresh, so a token the CLI refreshes is picked up.
    let spend = |rig: &Rig| {
        rig.store
            .resolve("anthropic", &rig.live(), now_ms())
            .unwrap()
            .unwrap()
    };
    assert_eq!(spend(&rig).secret.expose(), "sk-ant-oat01-live");
    assert_eq!(spend(&rig).kind, "cli_login");
    std::fs::write(
        &file,
        r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-newer","refreshToken":"rr2","expiresAt":4102444800000}}"#,
    )
    .unwrap();
    assert_eq!(spend(&rig).secret.expose(), "sk-ant-oat01-newer");
    assert_eq!(
        rig.store.get("anthropic").unwrap().status(now_ms()).kind,
        "cli_login"
    );

    // A login the other CLI has let lapse is not renewed here.
    std::fs::write(
        &file,
        r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-old","refreshToken":"rr3","expiresAt":1000}}"#,
    )
    .unwrap();
    let err = rig
        .store
        .resolve("anthropic", &rig.live(), now_ms())
        .unwrap_err();
    assert!(err.downcast_ref::<ReloginRequired>().is_some(), "{err}");
    assert_eq!(
        rig.fake().seen.lock().unwrap().len(),
        0,
        "no refresh was attempted"
    );
    assert_eq!(
        rig.store.get("anthropic").unwrap().status(now_ms()).state,
        "expired"
    );

    // --copy moves it into the store, refresh token and all.
    std::fs::write(
        &file,
        r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-copy","refreshToken":"rr4","expiresAt":4102444800000}}"#,
    )
    .unwrap();
    rig.login("anthropic", &from(true), &no_input).unwrap();
    let o = oauth_in(&rig, "anthropic");
    assert_eq!(
        (o.access.as_str(), o.refresh.as_str(), o.flow.as_str()),
        ("sk-ant-oat01-copy", "rr4", FLOW_CLAUDE)
    );
}

#[test]
fn importing_codex_copies_the_account_id() {
    let rig = Rig::new();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("auth.json");
    let access = chatgpt_jwt("acct-codex", 4_102_444_800);
    std::fs::write(
        &file,
        json!({"auth_mode": "chatgpt", "tokens": {"access_token": access, "refresh_token": "cr"}})
            .to_string(),
    )
    .unwrap();
    rig.login(
        "openai-chatgpt",
        &Options {
            method: Some("import-codex".into()),
            file: Some(file.to_string_lossy().into_owned()),
            copy: true,
            ..Options::default()
        },
        &no_input,
    )
    .unwrap();
    let o = oauth_in(&rig, "openai-chatgpt");
    assert_eq!((o.refresh.as_str(), o.flow.as_str()), ("cr", FLOW_CHATGPT));
    assert_eq!(o.metadata["account_id"], "acct-codex");
}

#[test]
fn generic_methods_store_a_key_a_command_or_a_file() {
    let rig = Rig::new();
    // key: asked for without echo.
    let ask = |question: &str, secret: bool| {
        assert!(secret && question.contains("groq"), "{question}");
        Ok("  sk-groq  ".to_string())
    };
    rig.login("groq", &Options::default(), &ask).unwrap();
    assert_eq!(
        rig.store.get("groq"),
        Some(Credential::ApiKey {
            key: "sk-groq".into(),
            flow: None
        })
    );
    // …or given.
    rig.login(
        "groq",
        &Options {
            token: Some("sk-given".into()),
            ..Options::default()
        },
        &no_input,
    )
    .unwrap();
    assert!(
        matches!(rig.store.get("groq"), Some(Credential::ApiKey { key, .. }) if key == "sk-given")
    );

    // A command that does not work is refused at login, not at first use.
    let err = rig
        .login(
            "groq",
            &Options {
                method: Some("command".into()),
                command: Some("exit 1".into()),
                ..Options::default()
            },
            &no_input,
        )
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("did not yield a key"),
        "{err:#}"
    );
    #[cfg(unix)]
    {
        rig.login(
            "groq",
            &Options {
                method: Some("command".into()),
                command: Some("echo sk-from-$((1+1))".into()),
                ..Options::default()
            },
            &no_input,
        )
        .unwrap();
        let got = rig.store.resolve("groq", &rig.live(), 0).unwrap().unwrap();
        assert_eq!((got.secret.expose(), got.kind), ("sk-from-2", "command"));
        let stored = std::fs::read_to_string(rig.store.path()).unwrap();
        assert!(
            !stored.contains("sk-from-2"),
            "the command is stored, not what it printed"
        );
    }

    // A file must exist and be owner-only; a missing --file is a usage error.
    let err = rig
        .login("groq", &method("token-file"), &no_input)
        .unwrap_err();
    assert!(err.to_string().contains("--file"), "{err}");
}

#[test]
fn an_unknown_method_lists_what_the_provider_takes() {
    let rig = Rig::new();
    let err = rig
        .login("openai-chatgpt", &method("password"), &no_input)
        .unwrap_err()
        .to_string();
    for want in ["device", "import-codex", "key", "command", "token-file"] {
        assert!(err.contains(want), "{err}");
    }
    // Another provider's own method is not offered.
    let err = rig
        .login("groq", &method("device"), &no_input)
        .unwrap_err()
        .to_string();
    assert!(!err.contains("import-codex"), "{err}");
    // --copy only goes with an import.
    let err = rig
        .login(
            "groq",
            &Options {
                copy: true,
                ..Options::default()
            },
            &no_input,
        )
        .unwrap_err();
    assert!(err.to_string().contains("--copy"), "{err}");
}

// ---------------------------------------------------------------------------
// Where a login's token goes
// ---------------------------------------------------------------------------

#[test]
fn a_login_is_confined_to_its_own_https_host() {
    assert!(destination_allowed(
        FLOW_CHATGPT,
        "https://chatgpt.com/backend-api/codex"
    ));
    assert!(!destination_allowed(
        FLOW_CHATGPT,
        "http://chatgpt.com/backend-api/codex"
    ));
    assert!(!destination_allowed(
        FLOW_CHATGPT,
        "https://chatgpt.com.evil.example/x"
    ));
    assert!(!destination_allowed(
        FLOW_CHATGPT,
        "https://evil.example/chatgpt.com"
    ));
    assert!(!destination_allowed(
        FLOW_CHATGPT,
        "https://u:p@chatgpt.com/x"
    ));
    assert!(!destination_allowed(
        FLOW_CHATGPT,
        "https://chatgpt.com:8443/x"
    ));
    assert!(!destination_allowed(FLOW_CHATGPT, "not a url"));
    assert!(destination_allowed(
        FLOW_COPILOT,
        "https://api.githubcopilot.com"
    ));
    assert!(destination_allowed(
        FLOW_COPILOT,
        "https://api.individual.githubcopilot.com"
    ));
    assert!(destination_allowed(
        FLOW_COPILOT,
        "https://copilot-proxy.githubusercontent.com"
    ));
    assert!(!destination_allowed(
        FLOW_COPILOT,
        "https://githubcopilot.com.evil.example"
    ));
    assert!(!destination_allowed(FLOW_COPILOT, "https://api.openai.com"));
    assert!(destination_allowed(
        FLOW_XAI,
        "https://cli-chat-proxy.grok.com/v1"
    ));
    assert!(!destination_allowed(FLOW_XAI, "https://api.x.ai/v1"));
    assert!(destination_allowed(
        FLOW_SETUP_TOKEN,
        "https://api.anthropic.com"
    ));
    assert!(!destination_allowed(
        FLOW_CLAUDE,
        "https://api.anthropic.com.evil.example"
    ));
    // A flow nothing knows is allowed nowhere.
    assert!(!destination_allowed("mystery", "https://chatgpt.com"));
    assert!(fixed_host(FLOW_CHATGPT) && !fixed_host(FLOW_OPENROUTER));
}

#[test]
fn a_stored_endpoint_is_revalidated() {
    let md = |v: &str| BTreeMap::from([("api_endpoint".to_string(), v.to_string())]);
    assert_eq!(
        endpoint_for(FLOW_COPILOT, &md("https://api.business.githubcopilot.com/")).as_deref(),
        Some("https://api.business.githubcopilot.com")
    );
    // A hand-edited store cannot redirect a token.
    assert_eq!(
        endpoint_for(FLOW_COPILOT, &md("https://evil.example")),
        None
    );
    assert_eq!(
        endpoint_for(FLOW_COPILOT, &md("http://api.githubcopilot.com")),
        None
    );
    assert_eq!(
        endpoint_for(FLOW_CHATGPT, &md("https://api.githubcopilot.com")),
        None
    );
    assert_eq!(endpoint_for(FLOW_COPILOT, &BTreeMap::new()), None);
}

fn resolved(secret: &str, kind: &'static str, flow: Option<&str>) -> Resolved {
    let mut r = Resolved::from_key(secret.into(), kind);
    r.flow = flow.map(str::to_string);
    r
}

#[test]
fn each_login_is_shaped_into_the_request_it_needs() {
    let mut chatgpt = resolved("at", "oauth", Some(FLOW_CHATGPT));
    chatgpt
        .metadata
        .insert("account_id".into(), "acct-1".into());
    shape(&mut chatgpt);
    assert_eq!(chatgpt.style, Style::Bearer);
    let header = |r: &Resolved, name: &str| {
        r.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    };
    assert_eq!(
        header(&chatgpt, "ChatGPT-Account-Id").as_deref(),
        Some("acct-1")
    );
    assert_eq!(header(&chatgpt, "originator").as_deref(), Some("llmman"));

    let mut no_account = resolved("at", "oauth", Some(FLOW_CHATGPT));
    shape(&mut no_account);
    assert_eq!(header(&no_account, "ChatGPT-Account-Id"), None);

    let mut copilot = resolved("gho", "oauth", Some(FLOW_COPILOT));
    shape(&mut copilot);
    assert_eq!(copilot.style, Style::Bearer);
    assert!(copilot.headers.is_empty());

    for r in [
        resolved("sk-ant-oat01-x", "token", Some(FLOW_SETUP_TOKEN)),
        resolved("sk-ant-oat01-x", "cli_login", Some(FLOW_CLAUDE)),
        // A subscription token kept anywhere but the environment and
        // llmman.conf is recognised by what it is.
        resolved("sk-ant-oat01-x", "command", None),
    ] {
        let mut r = r;
        shape(&mut r);
        assert_eq!(r.style, Style::Bearer);
        assert_eq!(
            header(&r, "anthropic-beta").as_deref(),
            Some("oauth-2025-04-20")
        );
        assert_eq!(
            header(&r, "anthropic-dangerous-direct-browser-access").as_deref(),
            Some("true")
        );
    }

    // An ordinary key, or an environment variable, is left exactly alone.
    for r in [
        resolved("sk-ant-api03-x", "api_key", None),
        resolved("sk-ant-oat01-x", "env", None),
        resolved("sk-ant-oat01-x", "config", None),
        resolved("sk-ant-oat01-x", "request", None),
        resolved("k", "oauth", Some(FLOW_OPENROUTER)),
    ] {
        let mut r = r;
        let before = r.headers.clone();
        shape(&mut r);
        assert_eq!((r.style, r.headers), (Style::Key, before));
    }
}

#[test]
fn text_from_a_server_is_made_safe_to_print() {
    assert_eq!(clean("a\u{1b}[31mb\n\tc"), "a [31mb c");
    assert_eq!(clean(&"x".repeat(500)).chars().count(), 161);
}

#[test]
fn every_provider_with_a_login_lists_its_methods_first() {
    let names = |p: &str| methods(p).iter().map(|m| m.name).collect::<Vec<_>>();
    assert_eq!(
        names("openai-chatgpt"),
        ["device", "import-codex", "key", "command", "token-file"]
    );
    assert_eq!(names("anthropic")[..2], ["setup-token", "import-claude"]);
    assert_eq!(names("groq"), ["key", "command", "token-file"]);
    assert!(profile("groq").is_none());
    assert_eq!(
        crate::providers::login_hint("github-copilot").as_deref(),
        Some("run `llmman login --provider github-copilot`")
    );
    assert_eq!(crate::providers::login_hint("groq"), None);
}

// Keeps the compiler honest about what `Status` the store reports for a
// freshly written login.
#[test]
fn a_stored_login_reports_its_kind_without_a_secret() {
    let rig = Rig::new();
    rig.store
        .put(
            "github-copilot",
            Credential::OAuth(OAuth {
                flow: FLOW_COPILOT.into(),
                access: "gho-secret".into(),
                refresh: String::new(),
                expires_ms: 0,
                metadata: BTreeMap::new(),
            }),
        )
        .unwrap();
    let status = rig.store.get("github-copilot").unwrap().status(now_ms());
    assert_eq!(
        status,
        Status {
            kind: "oauth",
            state: "ready",
            expires_at_ms: None
        }
    );
    assert!(!serde_json::to_string(&status)
        .unwrap()
        .contains("gho-secret"));
}

// ---------------------------------------------------------------------------
// Cloud chains and token services
// ---------------------------------------------------------------------------

fn with_params(method: &str, params: &[(&str, &str)]) -> Options {
    Options {
        method: Some(method.into()),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        ..Options::default()
    }
}

fn chain_in(rig: &Rig, id: &str) -> crate::chains::Chain {
    match rig.store.get(id) {
        Some(Credential::Chain(c)) => c,
        other => panic!("expected a chain selection, got {other:?}"),
    }
}

fn secret_input(secret: &'static str) -> impl Fn(&str, bool) -> anyhow::Result<String> {
    move |_, hidden| {
        assert!(hidden, "a secret is not echoed");
        Ok(secret.to_string())
    }
}

#[test]
fn aws_logins_store_a_selection_not_a_secret() {
    use crate::chains::Chain;
    let rig = Rig::new();
    rig.login(
        "amazon-bedrock",
        &with_params("aws-chain", &[("region", "us-west-2")]),
        &no_input,
    )
    .unwrap();
    let Chain::Aws(chain) = chain_in(&rig, "amazon-bedrock") else {
        panic!("an AWS chain")
    };
    assert_eq!(
        (chain.region.as_deref(), chain.profile.as_deref()),
        (Some("us-west-2"), None)
    );

    // The profile is asked for when not given.
    let ask = |q: &str, hidden: bool| {
        assert!(q.contains("profile") && !hidden, "{q}");
        Ok("work".to_string())
    };
    rig.login("amazon-bedrock", &method("aws-profile"), &ask)
        .unwrap();
    let Chain::Aws(chain) = chain_in(&rig, "amazon-bedrock") else {
        panic!("an AWS chain")
    };
    assert_eq!(chain.profile.as_deref(), Some("work"));
    let stored = std::fs::read_to_string(rig.store.path()).unwrap();
    assert!(stored.contains("\"chain\": \"aws\""), "{stored}");

    // A hostile region is refused and nothing is replaced.
    let err = rig
        .login(
            "amazon-bedrock",
            &with_params("aws-chain", &[("region", "x.evil.com")]),
            &no_input,
        )
        .unwrap_err();
    assert!(err.to_string().contains("region"), "{err}");
    let Chain::Aws(chain) = chain_in(&rig, "amazon-bedrock") else {
        panic!("an AWS chain")
    };
    assert_eq!(chain.profile.as_deref(), Some("work"));
    // The mantle provider takes the same methods.
    assert_eq!(
        methods("amazon-bedrock-mantle")
            .iter()
            .map(|m| m.name)
            .collect::<Vec<_>>()[..2],
        ["aws-chain", "aws-profile"]
    );
}

#[cfg(unix)]
#[test]
fn a_google_service_account_login_keeps_the_path_and_checks_the_type() {
    use crate::chains::Chain;
    use std::os::unix::fs::PermissionsExt;
    let rig = Rig::new();
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("sa.json");
    std::fs::write(&key, r#"{"type":"service_account","client_email":"a@b.iam.gserviceaccount.com","private_key":"SECRET-KEY-MATERIAL"}"#).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut opts = with_params(
        "service-account",
        &[("project", "my-proj"), ("location", "us-central1")],
    );
    opts.file = Some(key.to_string_lossy().into_owned());
    rig.login("google-vertex", &opts, &no_input).unwrap();
    let Chain::Gcp(chain) = chain_in(&rig, "google-vertex") else {
        panic!("a Google chain")
    };
    assert_eq!(chain.project.as_deref(), Some("my-proj"));
    assert_eq!(chain.location.as_deref(), Some("us-central1"));
    assert!(chain
        .credentials_file
        .as_deref()
        .unwrap()
        .ends_with("sa.json"));
    let stored = std::fs::read_to_string(rig.store.path()).unwrap();
    assert!(
        !stored.contains("SECRET-KEY-MATERIAL"),
        "the key stays in its file"
    );

    // Not a service account; no file; a missing file.
    std::fs::write(&key, r#"{"type":"authorized_user"}"#).unwrap();
    let err = rig.login("google-vertex", &opts, &no_input).unwrap_err();
    assert!(err.to_string().contains("not a service account"), "{err}");
    let err = rig
        .login("google-vertex", &method("service-account"), &no_input)
        .unwrap_err();
    assert!(err.to_string().contains("--file"), "{err}");
    let missing = Options {
        file: Some(dir.path().join("nope.json").to_string_lossy().into_owned()),
        ..opts
    };
    assert!(rig.login("google-vertex", &missing, &no_input).is_err());

    rig.login(
        "google-vertex",
        &with_params("adc", &[("project", "p2")]),
        &no_input,
    )
    .unwrap();
    let Chain::Gcp(chain) = chain_in(&rig, "google-vertex") else {
        panic!("a Google chain")
    };
    assert_eq!(
        (chain.project.as_deref(), chain.credentials_file.as_deref()),
        (Some("p2"), None)
    );
}

#[test]
fn an_entra_client_secret_is_tried_before_it_is_stored() {
    use crate::chains::Chain;
    let rig = Rig::new();
    rig.fake().on(
        "/tenant-l1/oauth2/v2.0/token",
        vec![(200, json!({"access_token": "eyJ.t", "expires_in": 3600}))],
    );
    let origin = rig.server.origin.clone();
    let params = [
        ("tenant_id", "tenant-l1"),
        ("client_id", "client-l1"),
        ("authority_host", origin.as_str()),
    ];
    rig.login(
        "azure",
        &with_params("entra", &params),
        &secret_input("entra-secret"),
    )
    .unwrap();
    let hit = &rig.fake().hits("/tenant-l1/oauth2/v2.0/token")[0];
    assert_eq!(hit.form()["client_secret"], "entra-secret");
    let Chain::Azure(chain) = chain_in(&rig, "azure") else {
        panic!("an Azure chain")
    };
    assert_eq!(chain.client_secret.as_deref(), Some("entra-secret"));
    assert_eq!(chain.tenant_id.as_deref(), Some("tenant-l1"));
    // Debug shows no secret.
    assert!(!format!("{:?}", rig.store.get("azure")).contains("entra-secret"));
    assert_eq!(rig.store.get("azure").unwrap().status(0).kind, "entra");

    // A secret Entra refuses is not stored.
    let rig = Rig::new();
    rig.fake().on(
        "/tenant-l2/oauth2/v2.0/token",
        vec![(401, json!({"error": "invalid_client"}))],
    );
    let origin = rig.server.origin.clone();
    let err = rig
        .login(
            "azure",
            &with_params(
                "entra",
                &[
                    ("tenant_id", "tenant-l2"),
                    ("client_id", "c"),
                    ("authority_host", origin.as_str()),
                ],
            ),
            &secret_input("wrong"),
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("invalid_client"), "{err:#}");
    assert!(!format!("{err:#}").contains("wrong"));
    assert!(rig.store.get("azure").is_none());
}

#[test]
fn azure_federated_managed_identity_and_cli_logins() {
    use crate::chains::Chain;
    let rig = Rig::new();
    rig.fake().on(
        "/tenant-l3/oauth2/v2.0/token",
        vec![(200, json!({"access_token": "eyJ.f", "expires_in": 3600}))],
    );
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("t");
    std::fs::write(&token, "projected").unwrap();
    let origin = rig.server.origin.clone();
    let mut opts = with_params(
        "federated",
        &[
            ("tenant_id", "tenant-l3"),
            ("client_id", "c3"),
            ("authority_host", origin.as_str()),
        ],
    );
    opts.file = Some(token.to_string_lossy().into_owned());
    rig.login("azure", &opts, &no_input).unwrap();
    assert_eq!(
        rig.fake().hits("/tenant-l3/oauth2/v2.0/token")[0].form()["client_assertion"],
        "projected"
    );
    let Chain::Azure(chain) = chain_in(&rig, "azure") else {
        panic!()
    };
    assert!(chain.federated_token_file.is_some() && chain.client_secret.is_none());
    assert!(rig
        .login("azure", &with_params("federated", &[]), &no_input)
        .is_err());

    rig.login(
        "azure",
        &with_params("managed-identity", &[("client_id", "uami")]),
        &no_input,
    )
    .unwrap();
    let Chain::Azure(chain) = chain_in(&rig, "azure") else {
        panic!()
    };
    assert_eq!(
        (chain.managed_identity, chain.client_id.as_deref()),
        (Some(true), Some("uami"))
    );
    rig.login("azure", &method("cli"), &no_input).unwrap();
    let Chain::Azure(chain) = chain_in(&rig, "azure") else {
        panic!()
    };
    assert_eq!(chain.cli, Some(true));
}

#[test]
fn client_credentials_for_any_provider_are_tried_then_stored() {
    use crate::chains::Chain;
    let rig = Rig::new();
    rig.fake().on(
        "/idp/token",
        vec![(200, json!({"access_token": "t", "expires_in": 60}))],
    );
    let url = format!("{}/idp/token", rig.server.origin);
    let params = [
        ("token_url", url.as_str()),
        ("client_id", "my-client"),
        ("scope", "a b"),
        ("client_auth", "basic"),
    ];
    rig.login(
        "my-gateway",
        &with_params("client-credentials", &params),
        &secret_input("my-secret"),
    )
    .unwrap();
    let hit = &rig.fake().hits("/idp/token")[0];
    assert!(hit.headers["authorization"].starts_with("Basic "));
    assert_eq!(hit.form()["scope"], "a b");
    let Chain::Token(svc) = chain_in(&rig, "my-gateway") else {
        panic!()
    };
    assert_eq!(svc.client_secret.as_deref(), Some("my-secret"));
    assert_eq!(svc.client_auth, Some(crate::tokensvc::ClientAuth::Basic));
    // The login is accepted for a provider with no methods of its own, but
    // listed apart from them.
    assert!(!methods("my-gateway")
        .iter()
        .any(|m| m.name == "client-credentials"));
    assert!(ADVANCED.iter().any(|m| m.name == "client-credentials"));

    // A refused secret is the login's error and stores nothing.
    let rig = Rig::new();
    rig.fake().on(
        "/idp/token",
        vec![(400, json!({"error": "invalid_client"}))],
    );
    let url = format!("{}/idp/token", rig.server.origin);
    let err = rig
        .login(
            "my-gateway",
            &with_params(
                "client-credentials",
                &[("token_url", url.as_str()), ("client_id", "c")],
            ),
            &secret_input("nope"),
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("invalid_client"), "{err:#}");
    assert!(rig.store.get("my-gateway").is_none());
    let err = rig
        .login(
            "my-gateway",
            &with_params(
                "client-credentials",
                &[
                    ("token_url", url.as_str()),
                    ("client_id", "c"),
                    ("client_auth", "digest"),
                ],
            ),
            &secret_input("x"),
        )
        .unwrap_err();
    assert!(err.to_string().contains("client_auth"), "{err}");
}

#[test]
fn watsonx_and_sap_ai_core_logins_store_the_service_with_its_secret() {
    use crate::chains::Chain;
    use crate::tokensvc::Grant;
    let rig = Rig::new();
    rig.fake().on(
        "/iam",
        vec![(200, json!({"access_token": "iam", "expires_in": 60}))],
    );
    let url = format!("{}/iam", rig.server.origin);
    rig.login(
        "watsonx",
        &with_params("apikey", &[("token_url", url.as_str())]),
        &secret_input("ibm-key"),
    )
    .unwrap();
    assert_eq!(rig.fake().hits("/iam")[0].form()["apikey"], "ibm-key");
    let Chain::Token(svc) = chain_in(&rig, "watsonx") else {
        panic!()
    };
    assert_eq!(svc.grant, Grant::IbmApikey);

    // SAP AI Core: the service key's fields, `/oauth/token` appended.
    rig.fake().on(
        "/oauth/token",
        vec![(200, json!({"access_token": "sap", "expires_in": 60}))],
    );
    let key = json!({
        "clientid": "sb-client", "clientsecret": "sap-secret", "url": rig.server.origin,
        "serviceurls": {"AI_API_URL": "https://api.ai.example"},
    })
    .to_string();
    let paste = move |q: &str, hidden: bool| {
        assert!(q.contains("service key") && hidden);
        Ok(key.clone())
    };
    rig.login("sap-ai-core", &method("service-key"), &paste)
        .unwrap();
    let hit = &rig.fake().hits("/oauth/token")[0];
    assert!(hit.headers["authorization"].starts_with("Basic "));
    let Chain::Token(svc) = chain_in(&rig, "sap-ai-core") else {
        panic!()
    };
    assert_eq!(svc.client_id.as_deref(), Some("sb-client"));
    assert!(svc.token_url.as_deref().unwrap().ends_with("/oauth/token"));

    // A certificate-only key, and one with nothing usable, are refused.
    for (text, wants) in [
        (r#"{"clientid":"a","certurl":"https://x"}"#, "certificate"),
        (r#"{"hello":1}"#, "no clientid"),
        ("not json", "not JSON"),
    ] {
        let t = text.to_string();
        let ask = move |_: &str, _: bool| Ok(t.clone());
        let err = rig
            .login("sap-ai-core", &method("service-key"), &ask)
            .unwrap_err();
        assert!(err.to_string().contains(wants), "{text}: {err}");
    }
}

#[test]
fn databricks_m2m_needs_its_host_and_is_tried_first() {
    let rig = Rig::new();
    // No host: the template cannot be filled, so nothing is sent or stored.
    if crate::providers::key_from_env("DATABRICKS_HOST").is_none() {
        let err = rig
            .login(
                "databricks",
                &with_params("oauth-m2m", &[("client_id", "sp")]),
                &secret_input("s"),
            )
            .unwrap_err();
        assert!(err.to_string().contains("DATABRICKS_HOST"), "{err}");
        assert!(rig.store.get("databricks").is_none());
    }
}

#[cfg(unix)]
#[test]
fn a_jwt_bearer_login_keeps_the_key_path_and_tries_the_assertion() {
    use crate::chains::Chain;
    use std::os::unix::fs::PermissionsExt;
    let rig = Rig::new();
    rig.fake().on(
        "/jwt",
        vec![(200, json!({"access_token": "t", "expires_in": 60}))],
    );
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("k.pem");
    std::fs::write(&key, include_str!("../testdata/jwt-test-rsa.pem")).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let url = format!("{}/jwt", rig.server.origin);
    let mut opts = with_params(
        "jwt-bearer",
        &[("token_url", url.as_str()), ("issuer", "iss-1")],
    );
    opts.file = Some(key.to_string_lossy().into_owned());
    rig.login("my-gateway", &opts, &no_input).unwrap();
    assert!(rig.fake().hits("/jwt")[0].form().contains_key("assertion"));
    let Chain::Token(svc) = chain_in(&rig, "my-gateway") else {
        panic!()
    };
    assert_eq!(svc.issuer.as_deref(), Some("iss-1"));
    assert!(svc.key_file.as_deref().unwrap().ends_with("k.pem"));
    let stored = std::fs::read_to_string(rig.store.path()).unwrap();
    assert!(
        !stored.contains("BEGIN PRIVATE KEY"),
        "the key stays in its file"
    );
}

// ---------------------------------------------------------------------------
// Browser and device logins
// ---------------------------------------------------------------------------

/// The "browser": follows the authorization URL back to the listener with
/// `code` (or, for DigitalOcean, forwards a fragment token).
fn browser_ctx<'a>(
    rig: &'a Rig,
    store: Store,
    open: &'a dyn Fn(&str),
    input: &'a dyn Fn(&str, bool) -> anyhow::Result<String>,
) -> Ctx<'a> {
    Ctx {
        store,
        ep: Endpoints::all_at(&rig.server.origin),
        http: Http::allowing_http(),
        clock: Box::new(rig.clock.clone()),
        input,
        open,
    }
}

fn query_of(url: &str) -> HashMap<String, String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
}

/// A browser that arrives at `redirect_uri` with `code` and the state it
/// was sent.
fn returns_with_code(url: &str, code: &str) {
    let q = query_of(url);
    let redirect = reqwest::Url::parse(&q["redirect_uri"]).unwrap();
    let target = format!("{}?code={code}&state={}", redirect.path(), q["state"]);
    let port = redirect.port().unwrap();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(s, "GET {target} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n").unwrap();
        let _ = s.read_to_string(&mut String::new());
    });
}

#[test]
fn poe_login_exchanges_the_code_for_an_expiring_api_key() {
    let rig = Rig::new();
    rig.fake().on(
        "/token",
        vec![(
            200,
            json!({"api_key": "poe-key", "api_key_expires_in": 3600}),
        )],
    );
    let dir = tempfile::tempdir().unwrap();
    let ctx = browser_ctx(
        &rig,
        Store::at(dir.path().join("c.json")),
        &|u| returns_with_code(u, "poe-code"),
        &no_input,
    );
    let before = now_ms();
    login_with(&ctx, "poe", &Options::default()).unwrap();
    let Some(Credential::Token {
        token,
        expires_ms,
        flow,
    }) = ctx.store.get("poe")
    else {
        panic!("a token")
    };
    assert_eq!(
        (token.as_str(), flow.as_deref()),
        ("poe-key", Some(FLOW_POE))
    );
    assert!(expires_ms.unwrap() >= before + 3_600_000);
    let form = rig.fake().hits("/token")[0].form();
    assert_eq!(form["client_id"], "client_728290227fc048cc9262091a1ea197ea");
    assert_eq!(form["code"], "poe-code");
    assert!(
        pkce::Pkce::from_verifier(form["code_verifier"].clone())
            .challenge
            .len()
            > 20
    );
    // A key with no stated lifetime does not expire.
    rig.fake().on(
        "/token",
        vec![(200, json!({"api_key": "k2", "api_key_expires_in": null}))],
    );
    login_with(&ctx, "poe", &Options::default()).unwrap();
    let Some(Credential::Token { expires_ms, .. }) = ctx.store.get("poe") else {
        panic!()
    };
    assert_eq!(expires_ms, None);
    // Poe's key goes wherever a pasted one would.
    assert!(!fixed_host(FLOW_POE) && !fixed_host(FLOW_DIGITALOCEAN));
}

#[test]
fn digitalocean_login_forwards_the_fragment_token() {
    let rig = Rig::new();
    let dir = tempfile::tempdir().unwrap();
    let open = |url: &str| {
        let q = query_of(url);
        assert_eq!(q["response_type"], "token");
        assert_eq!(q["scope"], "genai:read inference:query");
        let redirect = reqwest::Url::parse(&q["redirect_uri"]).unwrap();
        let (port, callback, state) = (
            redirect.port().unwrap(),
            redirect.path().to_string(),
            q["state"].clone(),
        );
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut page = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(
                page,
                "GET {callback} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n"
            )
            .unwrap();
            let mut html = String::new();
            let _ = page.read_to_string(&mut html);
            assert!(
                html.contains("fetch(") && html.contains("/auth/token"),
                "{html}"
            );
            let body =
                format!(r#"{{"access_token":"do-token","expires_in":"7200","state":"{state}"}}"#);
            let mut post = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(
                post,
                "POST /auth/token HTTP/1.1\r\nHost: localhost:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            let _ = post.read_to_string(&mut String::new());
        });
    };
    let ctx = browser_ctx(&rig, Store::at(dir.path().join("c.json")), &open, &no_input);
    login_with(&ctx, "digitalocean", &Options::default()).unwrap();
    let Some(Credential::Token {
        token,
        expires_ms,
        flow,
    }) = ctx.store.get("digitalocean")
    else {
        panic!("a token")
    };
    assert_eq!(
        (token.as_str(), flow.as_deref()),
        ("do-token", Some(FLOW_DIGITALOCEAN))
    );
    assert!(expires_ms.unwrap() > now_ms() + 7_000_000);
}

#[test]
fn chutes_login_uses_your_client_and_refreshes_with_it() {
    let rig = Rig::new();
    rig.fake().on(
        "/idp/token",
        vec![
            (
                200,
                json!({"access_token": "ch-a1", "refresh_token": "ch-r1", "expires_in": 3600}),
            ),
            (
                200,
                json!({"access_token": "ch-a2", "refresh_token": "ch-r2", "expires_in": 3600}),
            ),
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let opened: Rc<RefCell<Vec<String>>> = Rc::default();
    let seen = opened.clone();
    let open = move |u: &str| {
        seen.borrow_mut().push(u.to_string());
        returns_with_code(u, "ch-code");
    };
    let ctx = browser_ctx(&rig, Store::at(dir.path().join("c.json")), &open, &no_input);
    login_with(
        &ctx,
        "chutes",
        &with_params("oauth", &[("client_id", "cid_mine")]),
    )
    .unwrap();
    let q = query_of(&opened.borrow()[0]);
    assert_eq!(
        (q["client_id"].as_str(), q["scope"].as_str()),
        ("cid_mine", "openid profile chutes:invoke")
    );
    assert_eq!(q["code_challenge_method"], "S256");
    let form = rig.fake().hits("/idp/token")[0].form();
    assert_eq!(
        (form["grant_type"].as_str(), form["code"].as_str()),
        ("authorization_code", "ch-code")
    );
    let old = oauth_in_store(&ctx.store, "chutes");
    assert_eq!(old.metadata["client_id"], "cid_mine");

    let renewed = rig.live().refresh(&old).unwrap();
    assert_eq!(
        (renewed.access.as_str(), renewed.refresh.as_str()),
        ("ch-a2", "ch-r2")
    );
    let form = rig.fake().hits("/idp/token")[1].form();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["client_id"], "cid_mine");
    assert_eq!(form["refresh_token"], "ch-r1");
    assert_eq!(renewed.metadata["client_id"], "cid_mine");

    // A client id is required.
    let ask = |_: &str, _: bool| Ok(String::new());
    let ctx2 = browser_ctx(&rig, Store::at(dir.path().join("d.json")), &|_| {}, &ask);
    if crate::providers::key_from_env("CHUTES_CLIENT_ID").is_none() {
        assert!(login_with(&ctx2, "chutes", &method("oauth")).is_err());
    }
}

fn oauth_in_store(store: &Store, id: &str) -> OAuth {
    match store.get(id) {
        Some(Credential::OAuth(o)) => o,
        other => panic!("expected an OAuth credential, got {other:?}"),
    }
}

#[test]
fn gitlab_login_pins_the_instance_and_rotates_its_refresh_token() {
    let rig = Rig::new();
    rig.fake().on(
        "/oauth/token",
        vec![
            (
                200,
                json!({"access_token": "gl-a1", "refresh_token": "gl-r1", "expires_in": 7200}),
            ),
            (
                200,
                json!({"access_token": "gl-a2", "refresh_token": "gl-r2", "expires_in": 7200}),
            ),
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let opened: Rc<RefCell<Vec<String>>> = Rc::default();
    let seen = opened.clone();
    let open = move |u: &str| {
        seen.borrow_mut().push(u.to_string());
        returns_with_code(u, "gl-code");
    };
    let ctx = browser_ctx(&rig, Store::at(dir.path().join("c.json")), &open, &no_input);
    // The fake server stands in for gitlab.com, whose application is bundled.
    let origin = rig.server.origin.clone();
    let mut ctx = ctx;
    ctx.ep.gitlab = origin.clone();
    login_with(&ctx, "gitlab", &Options::default()).unwrap();
    let q = query_of(&opened.borrow()[0]);
    assert_eq!(
        q["client_id"],
        "fd180700a8f9c5d5557aca231632dd0611a1135a3bb510a741d5a988a3394fa7"
    );
    assert_eq!(
        (q["scope"].as_str(), q["code_challenge_method"].as_str()),
        ("api", "S256")
    );
    assert!(opened.borrow()[0].starts_with(&format!("{origin}/oauth/authorize")));
    let old = oauth_in_store(&ctx.store, "gitlab");
    assert_eq!(old.metadata["instance_url"], origin);

    let renewed = rig.live().refresh(&old).unwrap();
    assert_eq!(
        (renewed.access.as_str(), renewed.refresh.as_str()),
        ("gl-a2", "gl-r2")
    );
    assert_eq!(
        rig.fake().hits("/oauth/token")[1].form()["refresh_token"],
        "gl-r1"
    );

    // Another instance needs its own application.
    if crate::providers::key_from_env("GITLAB_OAUTH_CLIENT_ID").is_none() {
        let err = login_with(
            &ctx,
            "gitlab",
            &with_params(
                "pkce",
                &[("instance_url", "https://gitlab.example.com/gitlab/")],
            ),
        )
        .unwrap_err();
        assert!(err.to_string().contains("GITLAB_OAUTH_CLIENT_ID"), "{err}");
    }
    let err = login_with(
        &ctx,
        "gitlab",
        &with_params("pkce", &[("instance_url", "https://u:p@x.example")]),
    )
    .unwrap_err();
    assert!(err.to_string().contains("instance URL"), "{err}");

    // Where it may be spent: its instance, or gitlab.com's own hosts.
    let meta = BTreeMap::from([(
        "instance_url".to_string(),
        "https://gitlab.example.com/gitlab".to_string(),
    )]);
    assert!(destination_allowed_with(
        FLOW_GITLAB,
        "https://gitlab.example.com/api/v4",
        &meta
    ));
    assert!(destination_allowed_with(
        FLOW_GITLAB,
        "https://cloud.gitlab.com/x",
        &meta
    ));
    assert!(!destination_allowed_with(
        FLOW_GITLAB,
        "https://evil.example/x",
        &meta
    ));
    assert!(!destination_allowed(
        FLOW_GITLAB,
        "https://gitlab.example.com/x"
    ));
}

#[test]
fn snowflake_login_asks_the_accounts_own_endpoint_as_its_built_in_client() {
    let rig = Rig::new();
    rig.fake().on(
        "/oauth/token-request",
        vec![
            (
                200,
                json!({"access_token": "sf-a1", "refresh_token": "sf-r1", "expires_in": 600}),
            ),
            (200, json!({"access_token": "sf-a2", "expires_in": 600})),
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let opened: Rc<RefCell<Vec<String>>> = Rc::default();
    let seen = opened.clone();
    let open = move |u: &str| {
        seen.borrow_mut().push(u.to_string());
        returns_with_code(u, "sf-code");
    };
    let ctx = browser_ctx(&rig, Store::at(dir.path().join("c.json")), &open, &no_input);
    login_with(
        &ctx,
        "snowflake-cortex",
        &with_params(
            "oauth",
            &[
                ("account", "https://myorg-acct.snowflakecomputing.com/"),
                ("role", "ANALYST"),
            ],
        ),
    )
    .unwrap();
    let q = query_of(&opened.borrow()[0]);
    assert_eq!(q["client_id"], "LOCAL_APPLICATION");
    assert_eq!(q["scope"], "refresh_token session:role:ANALYST");
    let hit = &rig.fake().hits("/oauth/token-request")[0];
    use base64::Engine as _;
    assert_eq!(
        hit.headers["authorization"],
        format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode("LOCAL_APPLICATION:LOCAL_APPLICATION")
        )
    );
    let old = oauth_in_store(&ctx.store, "snowflake-cortex");
    assert_eq!(old.metadata["account"], "myorg-acct");
    // A refresh that rotates nothing keeps the refresh token in hand.
    let renewed = rig.live().refresh(&old).unwrap();
    assert_eq!(
        (renewed.access.as_str(), renewed.refresh.as_str()),
        ("sf-a2", "sf-r1")
    );

    assert_eq!(
        browser::snowflake_account("MyOrg-Acct.snowflakecomputing.com").as_deref(),
        Some("MyOrg-Acct")
    );
    assert_eq!(browser::snowflake_account("a/b"), None);
    let meta = BTreeMap::from([("account".to_string(), "myorg-acct".to_string())]);
    assert!(destination_allowed_with(
        FLOW_SNOWFLAKE,
        "https://myorg-acct.snowflakecomputing.com/api/v2/cortex/v1",
        &meta
    ));
    assert!(!destination_allowed_with(
        FLOW_SNOWFLAKE,
        "https://other.snowflakecomputing.com/x",
        &meta
    ));
}

#[test]
fn minimax_login_polls_the_user_code_and_refreshes() {
    let rig = Rig::new();
    let deadline = now_ms() + 300_000;
    // `{{state}}` is echoed from the request, as the real server does.
    rig.fake().on(
        "/oauth2/device/code",
        vec![(
            200,
            json!({
                "user_code": "MM-CODE",
                "verification_uri": format!("{}/verify", rig.server.origin),
                "expired_in": deadline,
                "interval": 2000,
                "state": "{{state}}",
            }),
        )],
    );
    rig.fake().on(
        "/oauth2/token",
        vec![
            (200, json!({"status": "pending"})),
            (200, json!({"status": "success", "access_token": "mm-a1", "refresh_token": "mm-r1", "expired_in": deadline})),
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let ctx = browser_ctx(
        &rig,
        Store::at(dir.path().join("c.json")),
        &|_| {},
        &no_input,
    );
    login_with(&ctx, "minimax-cn", &method("device")).unwrap();
    let sent = rig.fake().hits("/oauth2/device/code")[0].clone();
    let form = sent.form();
    assert_eq!(form["client_id"], "78257093-7e40-4613-99e0-527b14b39113");
    assert_eq!(form["scope"], "group_id profile model.completion");
    assert_eq!(form["code_challenge_method"], "S256");
    assert!(sent.headers.contains_key("x-request-id"));
    // Two polls (pending, then approved), each after waiting the interval.
    let polls = rig.fake().hits("/oauth2/token");
    assert_eq!(polls.len(), 2);
    let poll = polls[1].form();
    assert_eq!(
        poll["grant_type"],
        "urn:ietf:params:oauth:grant-type:user_code"
    );
    assert_eq!(poll["user_code"], "MM-CODE");
    assert_eq!(
        pkce::Pkce::from_verifier(poll["code_verifier"].clone()).challenge,
        form["code_challenge"]
    );
    assert_eq!(rig.clock.slept.borrow().len(), 2);
    let stored = oauth_in_store(&ctx.store, "minimax-cn");
    assert_eq!(
        (stored.access.as_str(), stored.refresh.as_str()),
        ("mm-a1", "mm-r1")
    );
    assert_eq!(stored.metadata["region"], "cn");
    assert_eq!(stored.expires_ms, deadline);
}

#[test]
fn minimax_expiry_and_replies_read_like_openclaws() {
    assert_eq!(browser_minimax_deadline(3600, 1_000), 3_601_000);
    assert_eq!(
        browser_minimax_deadline(1_700_000_000, 1_000),
        1_700_000_000_000
    );
    assert_eq!(
        browser_minimax_deadline(1_700_000_000_000, 1_000),
        1_700_000_000_000
    );
    let rig = Rig::new();
    rig.fake().on(
        "/oauth2/token",
        vec![(
            200,
            json!({"status": "success", "access_token": "mm-a2", "expired_in": 3600}),
        )],
    );
    let old = OAuth {
        flow: FLOW_MINIMAX.into(),
        access: "old".into(),
        refresh: "mm-r1".into(),
        expires_ms: 1,
        metadata: BTreeMap::from([("region".into(), "global".into())]),
    };
    // No rotated refresh token: the one in hand is kept.
    let renewed = rig.live().refresh(&old).unwrap();
    assert_eq!(
        (renewed.access.as_str(), renewed.refresh.as_str()),
        ("mm-a2", "mm-r1")
    );
    let form = rig.fake().hits("/oauth2/token")[0].form();
    assert_eq!(form["grant_type"], "refresh_token");
    // An error status is a rejection; a 503 is not.
    rig.fake().on(
        "/oauth2/token",
        vec![(
            200,
            json!({"status": "error", "base_resp": {"status_code": 1004, "status_msg": "expired"}}),
        )],
    );
    assert!(matches!(
        rig.live().refresh(&old),
        Err(RefreshError::Rejected(_))
    ));
    rig.fake().on("/oauth2/token", vec![(503, json!({}))]);
    assert!(matches!(
        rig.live().refresh(&old),
        Err(RefreshError::Transient(_))
    ));
    for provider in [
        "minimax",
        "minimax-coding-plan",
        "minimax-cn",
        "minimax-cn-coding-plan",
    ] {
        assert_eq!(methods(provider)[0].name, "device");
    }
    assert!(destination_allowed(
        FLOW_MINIMAX,
        "https://api.minimax.io/anthropic/v1"
    ));
    assert!(destination_allowed(
        FLOW_MINIMAX,
        "https://api.minimax.cn/anthropic/v1"
    ));
    assert!(!destination_allowed(
        FLOW_MINIMAX,
        "https://evil.example/anthropic/v1"
    ));
}

fn browser_minimax_deadline(expired_in: u64, now: u64) -> u64 {
    browser::minimax_deadline_for_test(expired_in, now)
}

#[test]
fn the_new_logins_are_bearers_confined_to_their_hosts() {
    for flow in [FLOW_CHUTES, FLOW_GITLAB, FLOW_SNOWFLAKE, FLOW_MINIMAX] {
        let mut r = Resolved::from_key("t".into(), "oauth");
        r.flow = Some(flow.into());
        shape(&mut r);
        assert_eq!(r.style, Style::Bearer, "{flow}");
        assert!(fixed_host(flow));
    }
    assert!(destination_allowed(FLOW_CHUTES, "https://llm.chutes.ai/v1"));
    assert!(!destination_allowed(
        FLOW_CHUTES,
        "https://llm.chutes.ai.evil.com/v1"
    ));
    assert!(!destination_allowed(FLOW_CHUTES, "http://llm.chutes.ai/v1"));
}

#[test]
fn a_relative_file_is_stored_as_an_absolute_path() {
    let got = super::absolute("tokens/key").unwrap();
    assert!(std::path::Path::new(&got).is_absolute(), "{got}");
    assert!(
        got.ends_with("tokens/key") || got.ends_with(r"tokens\key"),
        "{got}"
    );
}
