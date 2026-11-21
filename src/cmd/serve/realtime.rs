//! `/v1/realtime`: llama-server's realtime WebSocket (live transcription with
//! `?intent=transcription`, spoken conversation), relayed to the backend of the
//! model the `model` query parameter names.
//!
//! The daemon loads the model, connects to it before accepting the client's
//! upgrade (so a model that fails to load answers with an HTTP error), then
//! copies messages both ways until either side ends. Events pass through
//! unread, but for the model of a `session.*` event.
//!
//! A provider's realtime API is a different protocol and is refused, as is a
//! page on another site: a handshake is not subject to CORS.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use futures::{SinkExt as _, StreamExt as _};
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as BackendCloseFrame;
use tokio_tungstenite::tungstenite::Message as BackendMessage;
use tokio_tungstenite::Connector;

use super::backend::load_timeout_from_env;
use super::openai::{provider_refusal, route_model, LocalRoute};
use super::sched::ActivityGuard;
use super::{
    aggregation, cross_site_refused, ensure_model, foreign_origin, AppError, AppState, Target,
};

/// The route, here and on llama-server.
pub(super) const REALTIME_ROUTE: &str = "/v1/realtime";

const REALTIME: LocalRoute = LocalRoute {
    path: REALTIME_ROUTE,
    what: "realtime",
    hint: "use a locally served model with a streaming audio encoder or a spoken \
           conversation, such as LFM3-Omni or LFM3-Audio",
    default_prompt: false,
};

/// How long a close may flush to a peer that has stopped reading.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// The subprotocol OpenAI's realtime clients offer; echoed so their handshake
/// succeeds.
const REALTIME_PROTOCOL: &str = "realtime";

/// `GET /v1/realtime?model=<model>[&intent=transcription]`.
pub(super) async fn handle_realtime(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<Vec<(String, String)>>,
    ws: WebSocketUpgrade,
) -> Result<Response, AppError> {
    if foreign_origin(&headers) {
        return Err(cross_site_refused());
    }
    let Some(model) = query
        .iter()
        .find(|(key, _)| key == "model")
        .map(|(_, model)| model.trim().to_string())
        .filter(|model| !model.is_empty())
    else {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "the realtime API needs the model in the query: /v1/realtime?model=<model>",
        ));
    };
    let (canonical, target, guard) = ensure_model(
        &state,
        &route_model(&headers, &model, &REALTIME)?,
        Some(&headers),
        None,
    )
    .await?;
    if target.is_remote() {
        guard.release_idle();
        return Err(provider_refusal(REALTIME.path, REALTIME.hint));
    }
    // A peer loads the model before it answers the handshake.
    let setup = load_timeout_from_env().unwrap_or(Duration::from_secs(24 * 3600));
    let connected = tokio::time::timeout(
        setup,
        connect_backend(&target, &forwarded_query(&target, &canonical, &query)),
    )
    .await
    .map_err(|_| anyhow::anyhow!("timed out"))
    .and_then(|connected| connected);
    let backend = match connected {
        Ok(backend) => backend,
        Err(e) => {
            guard.release_idle();
            return Err(AppError::status(
                StatusCode::BAD_GATEWAY,
                format!("the inference backend has no realtime API for this model: {e}"),
            ));
        }
    };
    // Held until the session ends.
    let activity = super::begin_activity(guard, None).await;
    // A browser presents its key as a subprotocol, which must be echoed.
    let mut protocols = vec![REALTIME_PROTOCOL.to_string()];
    protocols.extend(super::auth::offered_ws_protocol(&headers));
    Ok(ws
        .protocols(protocols)
        .on_upgrade(move |client| async move {
            relay_session(client, backend, model, activity).await;
        }))
}

/// The query `target` is sent: llama-server serves one model, so `model` is
/// dropped; a peer is another daemon and gets `canonical`.
fn forwarded_query(
    target: &Target,
    canonical: &str,
    query: &[(String, String)],
) -> Vec<(String, String)> {
    let peer = matches!(target, Target::Peer(_));
    query
        .iter()
        .filter_map(|(key, value)| match (key.as_str(), peer) {
            ("model", false) => None,
            ("model", true) => Some((key.clone(), canonical.to_string())),
            _ => Some((key.clone(), value.clone())),
        })
        .collect()
}

type Backend =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// `ws://` (or `wss://`) for `target`'s realtime route with `query`.
fn backend_url(target: &Target, query: &[(String, String)]) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(&target.url(REALTIME_ROUTE))?;
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme)
        .map_err(|()| anyhow::anyhow!("cannot use {url} as a WebSocket URL"))?;
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query);
    }
    Ok(url.into())
}

async fn connect_backend(target: &Target, query: &[(String, String)]) -> anyhow::Result<Backend> {
    let mut request = backend_url(target, query)?.into_client_request()?;
    // A peer gets the hop marker and key, as `Target::authorize` sets them.
    if let Target::Peer(peer) = target {
        let headers = request.headers_mut();
        headers.insert(
            HeaderName::from_static(aggregation::HOP),
            HeaderValue::from_static("1"),
        );
        if let Some(key) = peer.api_key.as_deref() {
            headers.insert(
                axum::http::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {key}"))?,
            );
        }
    }
    let connector = (request.uri().scheme_str() == Some("wss"))
        .then(peer_tls)
        .transpose()?;
    let (backend, _) =
        tokio_tungstenite::connect_async_tls_with_config(request, None, false, connector).await?;
    Ok(backend)
}

/// A `wss://` peer is verified against the roots the daemon's HTTPS requests
/// to peers use: reqwest's bundled ones plus `LLMMAN_TLS_CA`.
fn peer_tls() -> anyhow::Result<Connector> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    for cert in crate::auth::tls_ca_der()? {
        roots.add(cert)?;
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(Connector::Rustls(Arc::new(config)))
}

/// Copies messages between `client` and `backend` until one ends.
async fn relay_session(
    client: WebSocket,
    backend: Backend,
    model: String,
    _activity: ActivityGuard,
) {
    let (mut to_client, mut from_client) = client.split();
    let (mut to_backend, mut from_backend) = backend.split();
    let up = async {
        while let Some(Ok(message)) = from_client.next().await {
            let closing = matches!(message, Message::Close(_));
            if to_backend.send(into_backend(message)).await.is_err() || closing {
                break;
            }
        }
    };
    let down = async {
        while let Some(Ok(message)) = from_backend.next().await {
            let Some(message) = into_client(message) else {
                continue;
            };
            let message = match message {
                Message::Text(text) => Message::Text(rewrite_session_model(text, &model)),
                other => other,
            };
            let closing = matches!(message, Message::Close(_));
            if to_client.send(message).await.is_err() || closing {
                break;
            }
        }
    };
    tokio::select! {
        () = up => {}
        () = down => {}
    }
    // Bounded: a peer that stopped reading must not hold the session's claim.
    let _ = tokio::time::timeout(CLOSE_TIMEOUT, async {
        let _ = to_backend.close().await;
        let _ = to_client.close().await;
    })
    .await;
}

/// Where a `session.*` event names a model: the session's, and its transcriber's.
const SESSION_MODELS: [&str; 2] = ["/session/model", "/session/audio/input/transcription/model"];

/// `text` with the model of a `session.*` event as the client named it
/// (llama-server reports the weights' path). Audio deltas are never parsed.
fn rewrite_session_model(text: String, model: &str) -> String {
    if !text.contains("\"session\"") {
        return text;
    }
    let Ok(mut event) = serde_json::from_str::<serde_json::Value>(&text) else {
        return text;
    };
    if !event["type"]
        .as_str()
        .is_some_and(|kind| kind.starts_with("session."))
    {
        return text;
    }
    let mut named = false;
    for pointer in SESSION_MODELS {
        if let Some(slot) = event.pointer_mut(pointer) {
            *slot = model.into();
            named = true;
        }
    }
    if !named {
        return text;
    }
    serde_json::to_string(&event).unwrap_or(text)
}

fn into_backend(message: Message) -> BackendMessage {
    match message {
        Message::Text(text) => BackendMessage::Text(text),
        Message::Binary(data) => BackendMessage::Binary(data),
        Message::Ping(data) => BackendMessage::Ping(data),
        Message::Pong(data) => BackendMessage::Pong(data),
        Message::Close(frame) => BackendMessage::Close(frame.map(|f| BackendCloseFrame {
            code: CloseCode::from(f.code),
            reason: f.reason,
        })),
    }
}

/// `None` for a raw frame.
fn into_client(message: BackendMessage) -> Option<Message> {
    Some(match message {
        BackendMessage::Text(text) => Message::Text(text),
        BackendMessage::Binary(data) => Message::Binary(data),
        BackendMessage::Ping(data) => Message::Ping(data),
        BackendMessage::Pong(data) => Message::Pong(data),
        BackendMessage::Close(frame) => Message::Close(frame.map(|f| CloseFrame {
            code: u16::from(f.code),
            reason: f.reason,
        })),
        BackendMessage::Frame(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(port: u16) -> Target {
        Target::Local(port)
    }

    #[test]
    fn the_backend_url_is_the_target_as_a_websocket() {
        assert_eq!(
            backend_url(&local(8081), &[]).unwrap(),
            "ws://127.0.0.1:8081/v1/realtime"
        );
        let peer = Target::Peer(Arc::new(super::super::PeerTarget {
            origin: "https://node-b:17434".into(),
            api_key: None,
        }));
        assert_eq!(
            backend_url(&peer, &[]).unwrap(),
            "wss://node-b:17434/v1/realtime"
        );
    }

    #[test]
    fn the_query_but_the_model_goes_along_encoded() {
        let query = [("intent".to_string(), "transcription".to_string())];
        assert_eq!(
            backend_url(&local(1), &query).unwrap(),
            "ws://127.0.0.1:1/v1/realtime?intent=transcription"
        );
        let odd = [("a b".to_string(), "c&d=é".to_string())];
        assert_eq!(
            backend_url(&local(1), &odd).unwrap(),
            "ws://127.0.0.1:1/v1/realtime?a+b=c%26d%3D%C3%A9"
        );
    }

    #[test]
    fn a_wss_peer_is_verified_against_the_daemons_roots() {
        assert!(matches!(peer_tls(), Ok(Connector::Rustls(_))));
    }

    #[test]
    fn a_peer_is_sent_the_model_and_llama_server_is_not() {
        let query = [
            ("model".to_string(), "gemma4".to_string()),
            ("intent".to_string(), "transcription".to_string()),
        ];
        let peer = Target::Peer(Arc::new(super::super::PeerTarget {
            origin: "http://node-b:17434".into(),
            api_key: None,
        }));
        assert_eq!(
            backend_url(&peer, &forwarded_query(&peer, "docker.io/ai/gemma4:latest", &query)).unwrap(),
            "ws://node-b:17434/v1/realtime?model=docker.io%2Fai%2Fgemma4%3Alatest&intent=transcription"
        );
        assert_eq!(
            backend_url(&local(1), &forwarded_query(&local(1), "x", &query)).unwrap(),
            "ws://127.0.0.1:1/v1/realtime?intent=transcription"
        );
    }

    #[test]
    fn a_session_event_names_the_model_as_it_was_asked_for() {
        let event =
            r#"{"type":"session.created","session":{"id":"s","model":"/m/blobs/sha256/ab"}}"#;
        let out: serde_json::Value =
            serde_json::from_str(&rewrite_session_model(event.into(), "org/model:q4")).unwrap();
        assert_eq!(out["session"]["model"], "org/model:q4");
        assert_eq!(out["session"]["id"], "s");
        // Whatever the order of the keys, and for a transcription session too.
        let reordered = r#"{"session":{"audio":{"input":{"transcription":{"model":"/m/x"}}},"model":"/m/y"},"type":"session.updated"}"#;
        let out: serde_json::Value =
            serde_json::from_str(&rewrite_session_model(reordered.into(), "m")).unwrap();
        assert_eq!(out["session"]["model"], "m");
        assert_eq!(
            out["session"]["audio"]["input"]["transcription"]["model"],
            "m"
        );
        // Not a session event, or one without a model: untouched, byte for byte.
        for same in [
            r#"{"type":"response.output_audio.delta","delta":"AAAA","model":"x"}"#,
            r#"{"type":"response.created","session":{"model":"x"}}"#,
            r#"{"type":"session.updated","session":{"id":"s"}}"#,
            r#"{"type":"session.created"#,
            "not json",
        ] {
            assert_eq!(rewrite_session_model(same.into(), "m"), same);
        }
    }

    #[test]
    fn messages_cross_unchanged_both_ways() {
        let text = Message::Text("{\"type\":\"session.update\"}".into());
        assert!(matches!(
            into_backend(text),
            BackendMessage::Text(t) if t == "{\"type\":\"session.update\"}"
        ));
        assert!(matches!(
            into_client(BackendMessage::Binary(vec![1, 2, 3])),
            Some(Message::Binary(b)) if b == [1, 2, 3]
        ));
        let close = into_client(BackendMessage::Close(Some(BackendCloseFrame {
            code: CloseCode::Normal,
            reason: "bye".into(),
        })));
        match close {
            Some(Message::Close(Some(frame))) => {
                assert_eq!(frame.code, 1000);
                assert_eq!(frame.reason, "bye");
            }
            other => panic!("{other:?}"),
        }
    }
}
