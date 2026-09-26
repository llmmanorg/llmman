//! Fixtures the `serve` test modules share.
//!
//! `serve/tests.rs` built these for itself and the inline test modules
//! reached back into it (`super::super::tests::…`) to borrow them.
//! Holding them here instead makes the borrowing one-directional:
//! `tests.rs` is now a peer of the modules it used to lend to.

use super::*;

pub(super) fn remote_target(base_url: &str) -> Target {
    remote_target_on(base_url, Wire::OpenAi)
}

pub(super) fn remote_target_on(base_url: &str, wire: Wire) -> Target {
    Target::Remote(Arc::new(RemoteTarget {
        provider: "mockprov".into(),
        base_url: base_url.into(),
        wire,
        model: "mock-model".into(),
        max_output: None,
        cost: None,
        api_key: Some("sk-test".into()),
    }))
}

pub(super) fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in pairs {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            value.parse().unwrap(),
        );
    }
    headers
}

pub(super) const PAIR: &str = "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5";
pub(super) const HOSTED: &str = "llmman.provider/anthropic/claude-sonnet-4-5";

pub(super) fn test_state() -> AppState {
    test_state_at(std::env::temp_dir())
}

/// `test_state` with a real store directory, for the few tests that
/// need `canonical_ref` to actually resolve something.
pub(super) fn test_state_at(store_path: PathBuf) -> AppState {
    AppState(Arc::new(test_inner(store_path)))
}

/// `test_state` with a hybrid byte budget (every other test has none).
pub(super) fn test_state_with_budget(hybrid_local_bytes: u64) -> AppState {
    let mut inner = test_inner(std::env::temp_dir());
    inner.hybrid_local_bytes = Some(hybrid_local_bytes);
    AppState(Arc::new(inner))
}

/// `test_state_at`'s `Inner`, for tests that set one field differently.
pub(super) fn test_inner(store_path: PathBuf) -> Inner {
    Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
            pending_loads: 0,
        }),
        exe: None,
        // `path`, so a resolve could never download.
        runtime: runtime::Lazy::new(Runtime::Path, None, None),
        llama_cpp_version: None,
        vllm_version: None,
        sglang_version: None,
        ctx_size: None,
        ctx_size_explicit: false,
        hybrid_local_bytes: None,
        flash_attention: None,
        kv_cache_type: None,
        split_mode: None,
        num_parallel: None,
        threads: None,
        cpu_limit: None,
        // usize::MAX, not 0 — 0 now means "admit almost nothing"
        // (see try_admit_against's doc comment), and no test here
        // calls ensure_model (the only caller of try_admit) directly
        // anyway.
        max_queue: usize::MAX,
        max_loaded_models: 0,
        peers: Vec::new(),
        memory: 0,
        store_path,
        cache_path: std::env::temp_dir(),
        prompt_log: None,
        usage_log: None,
        shell: shell::Policy {
            disabled: None,
            origins: default_allowed_origins(),
            command: Vec::new(),
        },
        auth: auth::Policy::default(),
        peer_key: None,
        client: Client::new(),
    }
}

/// `/llmman/node` answers `node`; every other route records its call.
pub(super) async fn mock_peer(
    node: aggregation::Node,
) -> (
    String,
    Arc<tokio::sync::Mutex<Vec<(String, HeaderMap, Bytes)>>>,
) {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let captured = seen.clone();
    let app = Router::new()
        .route("/llmman/node", get(move || async move { Json(node) }))
        .fallback(move |req: Request| async move {
            let (parts, body) = req.into_parts();
            let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            captured
                .lock()
                .await
                .push((parts.uri.path().to_string(), parts.headers, body));
            ([("content-type", "application/json")], "{}")
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (origin, seen)
}

pub(super) fn node(memory: u64, loaded: &[&str], stored: &[&str]) -> aggregation::Node {
    let map = |names: &[&str]| names.iter().map(|n| (n.to_string(), 1 << 30)).collect();
    aggregation::Node {
        memory,
        loaded: map(loaded),
        stored: map(stored),
    }
}

/// A long-lived, harmless real child process to back a test
/// `RunningModel` — `ModelProcess::is_alive`/`Drop` both need a real
/// `tokio::process::Child`, not a mock. `sleep` isn't on `PATH` on
/// Windows (which this project does target — see the `#[cfg(windows)]`
/// branches elsewhere in this module), so it's spawned differently per
/// platform rather than assuming a Unix-only test environment.
///
/// Its own process group (matching `spawn_vllm_server`'s own real
/// spawn — see its doc comment), not just the bare default: a
/// fixture backing an `Engine::Vllm` `RunningModel` hits
/// `ModelProcess::Drop`'s process-group-SIGKILL arm, which needs
/// this to actually *be* one, or that kill fails and prints a
/// spurious "SIGKILL to vllm process group ... failed" warning on
/// every test run that uses one — confirmed live via CodeRabbit
/// review on this repo's own git history.
#[cfg(unix)]
fn spawn_placeholder_process() -> tokio::process::Child {
    tokio::process::Command::new("sleep")
        .arg("60")
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .expect("spawn placeholder `sleep` process")
}

#[cfg(windows)]
fn spawn_placeholder_process() -> tokio::process::Child {
    tokio::process::Command::new("cmd")
        .args(["/C", "timeout", "/T", "60", "/NOBREAK"])
        .kill_on_drop(true)
        .spawn()
        .expect("spawn placeholder `cmd /C timeout` process")
}

pub(super) fn running_model_fixture(
    keep_alive: Option<Duration>,
    idle_for: Duration,
    in_flight: u32,
) -> RunningModel {
    RunningModel {
        process: ModelProcess::Local(Engine::LlamaServer, spawn_placeholder_process(), None),
        port: 0,
        digest: String::new(),
        size: 0,
        started_at: now_rfc3339(),
        last_active: Instant::now() - idle_for,
        last_active_wall: chrono::Utc::now(),
        backend_model_path: None,
        keep_alive,
        in_flight,
    }
}

/// Like `running_model_fixture`, but with a caller-chosen `Engine`
/// and `backend_model_path` — used by `backend_wire_model`'s own
/// tests below, which need to distinguish an `Engine::Mlx` backend
/// from every other one, and by the `engine_label` test.
pub(super) fn running_model_fixture_with_engine(
    engine: Engine,
    backend_model_path: Option<&str>,
) -> RunningModel {
    RunningModel {
        process: ModelProcess::Local(engine, spawn_placeholder_process(), None),
        port: 0,
        digest: String::new(),
        size: 0,
        started_at: now_rfc3339(),
        last_active: Instant::now(),
        last_active_wall: chrono::Utc::now(),
        backend_model_path: backend_model_path.map(|s| s.to_string()),
        keep_alive: None,
        in_flight: 0,
    }
}

/// The process-wide registry against placeholder gauges.
pub(super) fn rendered_registry() -> String {
    metrics::render(&metrics::Snapshot {
        version: "test".into(),
        start_time_seconds: 0,
        scheduling_requests_in_flight: 0,
        scheduling_capacity: 1,
        models_loaded: 0,
        models_loading: 0,
        models: Vec::new(),
    })
}

/// Binds `build_router(state, false)` on a free loopback port.
pub(super) async fn serve_router(state: AppState) -> String {
    serve_router_with(state, false).await
}

pub(super) async fn serve_router_with(state: AppState, metrics: bool) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state, metrics);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

pub(super) fn ws_upgrade(
    client: &Client,
    url: &str,
    origin: Option<&str>,
) -> reqwest::RequestBuilder {
    let mut req = client
        .get(url)
        .header("connection", "upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==");
    if let Some(origin) = origin {
        req = req.header("origin", origin);
    }
    req
}
