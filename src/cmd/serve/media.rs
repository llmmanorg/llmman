//! Image and video models in mediagen workers load one at a time: each
//! worker sizes its loads by the GPU memory it sees free, which on Apple
//! silicon does not count other processes, so an idle worker (Qwen-Image
//! keeps 16 GB) would leave the next one to run out of memory
//! mid-generation, past the load-time OOM retry. A worker unloaded for
//! another hands its finished videos to the daemon first, which serves them
//! until the worker would have been unloaded anyway.

use std::collections::VecDeque;

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use tokio::time::{Duration, Instant};

use super::{take_idle, AppState, PendingLoadGuard, RunningModel};
use crate::metrics::UnloadReason;

/// Videos kept for unloaded workers, as many as one worker keeps.
const MAX_STASHED: usize = 8;

/// A finished video of an unloaded worker: the worker's answer to each of
/// its two routes, `/v1/videos/{id}` and `/v1/videos/{id}/content`.
pub(super) struct StashedVideo {
    /// When its worker would have been unloaded; `None` never.
    expires: Option<Instant>,
    routes: Vec<(String, Stored)>,
}

struct Stored {
    status: StatusCode,
    content_type: Option<HeaderValue>,
    disposition: Option<HeaderValue>,
    body: Bytes,
}

/// Unloads the other idle image and video models before `model_ref`, one of
/// them, loads; the caller holds `media_loads` until its worker is running.
/// Returns the loaded-model slot this freed, reserved for `model_ref`.
///
/// Video GETs wait out the handover (`video_handover`): a worker is out of
/// `running` before its videos are in the stash.
pub(super) async fn evict_idle_media_models(
    state: &AppState,
    model_ref: &str,
) -> Option<PendingLoadGuard> {
    let _handover = state.0.video_handover.write().await;
    let (taken, slot) =
        take_idle(state, model_ref, |m| m.media, UnloadReason::Switched, true).await;
    for (name, mut running) in taken {
        stash_videos(state, &running).await;
        running.process.stop_and_wait().await;
        eprintln!("[llmman] unloaded {name}: image and video models load one at a time");
    }
    slot
}

/// Copies `running`'s finished videos into the daemon's stash.
async fn stash_videos(state: &AppState, running: &RunningModel) {
    let base = format!("http://127.0.0.1:{}", running.port);
    let get = |path: String| {
        let url = format!("{base}{path}");
        async move {
            let resp = state
                .0
                .client
                .get(url)
                .timeout(Duration::from_secs(60))
                .send()
                .await
                .ok()?;
            let header = |name| resp.headers().get(name).cloned();
            let (content_type, disposition) = (
                header(header::CONTENT_TYPE),
                header(header::CONTENT_DISPOSITION),
            );
            let status = resp.status();
            let body = resp.bytes().await.ok()?;
            Some(Stored {
                status,
                content_type,
                disposition,
                body,
            })
        }
    };
    let Some(list) = get("/v1/videos".into()).await else {
        return;
    };
    let ids: Vec<String> = serde_json::from_slice::<serde_json::Value>(&list.body)
        .ok()
        .and_then(|v| v["data"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v["id"].as_str().map(str::to_owned))
        .collect();
    let expires = running.keep_alive.map(|k| running.last_active + k);
    for id in ids {
        let mut routes = Vec::new();
        for path in [
            format!("/v1/videos/{id}"),
            format!("/v1/videos/{id}/content"),
        ] {
            if let Some(stored) = get(path.clone()).await {
                routes.push((path, stored));
            }
        }
        let mut stash = state
            .0
            .stashed_videos
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        prune(&mut stash, Instant::now());
        stash.push_back(StashedVideo { expires, routes });
        while stash.len() > MAX_STASHED {
            stash.pop_front();
        }
    }
}

/// The stashed answer for a video route, if a worker unloaded for another
/// model held that video and it has not expired.
pub(super) fn stashed_response(state: &AppState, path: &str) -> Option<Response> {
    let mut stash = state
        .0
        .stashed_videos
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    prune(&mut stash, Instant::now());
    let stored = stash
        .iter()
        .flat_map(|v| &v.routes)
        .find(|(p, _)| p == path)
        .map(|(_, s)| s)?;
    let mut builder = Response::builder().status(stored.status);
    for (name, value) in [
        (header::CONTENT_TYPE, &stored.content_type),
        (header::CONTENT_DISPOSITION, &stored.disposition),
    ] {
        if let Some(v) = value {
            builder = builder.header(name, v);
        }
    }
    builder.body(Body::from(stored.body.clone())).ok()
}

/// Drops the stashed videos whose worker would have been unloaded by now;
/// the idle reaper calls it, so they go even when nobody asks for them.
pub(super) fn prune_stashed(state: &AppState) {
    let mut stash = state
        .0
        .stashed_videos
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    prune(&mut stash, Instant::now());
}

fn prune(stash: &mut VecDeque<StashedVideo>, now: Instant) {
    stash.retain(|v| v.expires.is_none_or(|t| t > now));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(id: &str, expires: Option<Instant>) -> StashedVideo {
        StashedVideo {
            expires,
            routes: vec![(
                format!("/v1/videos/{id}/content"),
                Stored {
                    status: StatusCode::OK,
                    content_type: Some(HeaderValue::from_static("video/mp4")),
                    disposition: None,
                    body: Bytes::from_static(b"mp4"),
                },
            )],
        }
    }

    /// A worker unloaded for another image model leaves its videos behind.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unloaded_worker_hands_its_videos_over() {
        use axum::routing::get;
        let _serialised = crate::metrics::GLOBAL_COUNTER_TEST_LOCK.lock().await;
        let worker = axum::Router::new()
            .route(
                "/v1/videos",
                get(|| async { axum::Json(serde_json::json!({"data": [{"id": "v1"}]})) }),
            )
            .route("/v1/videos/v1", get(|| async { "{\"id\":\"v1\"}" }))
            .route(
                "/v1/videos/v1/content",
                get(|| async { ([(header::CONTENT_TYPE, "video/mp4")], "mp4 bytes") }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, worker).await });

        let state = super::super::test_support::test_state();
        let mut idle = super::super::test_support::running_model_fixture(
            Some(Duration::from_secs(300)),
            Duration::ZERO,
            0,
        );
        idle.media = true;
        idle.port = port;
        let chat = super::super::test_support::running_model_fixture(None, Duration::ZERO, 0);
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert("old-image".into(), idle);
            mgr.running.insert("chat".into(), chat);
        }
        evict_idle_media_models(&state, "new-image").await;
        {
            let mgr = state.0.manager.lock().await;
            assert!(!mgr.running.contains_key("old-image"));
            assert!(mgr.running.contains_key("chat"));
        }
        let resp = stashed_response(&state, "/v1/videos/v1/content").unwrap();
        assert_eq!(resp.headers()[header::CONTENT_TYPE], "video/mp4");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        assert_eq!(&body[..], b"mp4 bytes");
        assert!(stashed_response(&state, "/v1/videos/v1").is_some());
    }

    #[tokio::test]
    async fn stashed_videos_answer_until_their_worker_would_have_unloaded() {
        let state = super::super::test_support::test_state();
        let now = Instant::now();
        {
            let mut stash = state.0.stashed_videos.lock().unwrap();
            stash.push_back(video("old", Some(now - Duration::from_secs(1))));
            stash.push_back(video("kept", Some(now + Duration::from_secs(60))));
            stash.push_back(video("forever", None));
        }
        assert!(stashed_response(&state, "/v1/videos/old/content").is_none());
        let resp = stashed_response(&state, "/v1/videos/kept/content").unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[header::CONTENT_TYPE], "video/mp4");
        assert!(stashed_response(&state, "/v1/videos/forever/content").is_some());
        assert!(stashed_response(&state, "/v1/videos/missing/content").is_none());
        assert_eq!(state.0.stashed_videos.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn the_idle_reaper_drops_expired_videos() {
        let state = super::super::test_support::test_state();
        let now = Instant::now();
        {
            let mut stash = state.0.stashed_videos.lock().unwrap();
            stash.push_back(video("old", Some(now - Duration::from_secs(1))));
            stash.push_back(video("kept", Some(now + Duration::from_secs(60))));
        }
        super::super::sched::reap_idle_models_once(&state).await;
        assert_eq!(state.0.stashed_videos.lock().unwrap().len(), 1);
    }
}
