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
