//! A scripted local HTTP server for tests of everything that talks to a
//! token endpoint or a metadata service. Nothing here reaches the network:
//! the server binds an ephemeral loopback port, and a test points the code
//! under test at [`Server::origin`].

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// One request the server saw.
#[derive(Clone, Debug)]
pub struct Seen {
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Seen {
    /// The body as a form.
    pub fn form(&self) -> HashMap<String, String> {
        reqwest::Url::parse(&format!("http://x/?{}", self.body))
            .unwrap()
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect()
    }
}

#[derive(Clone)]
struct Reply {
    status: u16,
    body: String,
}

#[derive(Default)]
pub struct Fake {
    script: Mutex<HashMap<String, VecDeque<Reply>>>,
    seen: Mutex<Vec<Seen>>,
}

impl Fake {
    /// `path` answers these JSON bodies in order, then the last one for
    /// good.
    pub fn on(&self, path: &str, replies: Vec<(u16, serde_json::Value)>) {
        self.on_text(
            path,
            replies
                .into_iter()
                .map(|(s, b)| (s, b.to_string()))
                .collect(),
        );
    }

    /// As [`Fake::on`], with bodies that are not JSON.
    pub fn on_text(&self, path: &str, replies: Vec<(u16, String)>) {
        self.script.lock().unwrap().insert(
            path.to_string(),
            replies
                .into_iter()
                .map(|(status, body)| Reply { status, body })
                .collect(),
        );
    }

    pub fn hits(&self, path: &str) -> Vec<Seen> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.path == path)
            .cloned()
            .collect()
    }
}

async fn respond(State(fake): State<Arc<Fake>>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();
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
        query,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    });
    // `{{name}}` in a scripted body stands for that field of a form request,
    // for a server that echoes what it was sent.
    let form: Vec<(String, String)> =
        reqwest::Url::parse(&format!("http://x/?{}", String::from_utf8_lossy(&body)))
            .map(|u| u.query_pairs().into_owned().collect())
            .unwrap_or_default();
    let mut script = fake.script.lock().unwrap();
    let Some(queue) = script.get_mut(&path) else {
        return (
            StatusCode::NOT_FOUND,
            [("content-type", "application/json")],
            r#"{"error":"unscripted"}"#,
        )
            .into_response();
    };
    let reply = if queue.len() > 1 {
        queue.pop_front().unwrap()
    } else {
        queue.front().cloned().unwrap()
    };
    let mut text = reply.body;
    for (k, v) in &form {
        text = text.replace(&format!("{{{{{k}}}}}"), v);
    }
    (
        StatusCode::from_u16(reply.status).unwrap(),
        [("content-type", "application/json")],
        text,
    )
        .into_response()
}

pub struct Server {
    /// `http://127.0.0.1:<port>`
    pub origin: String,
    pub fake: Arc<Fake>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    /// `host:port`, without the scheme.
    pub fn authority(&self) -> &str {
        self.origin.trim_start_matches("http://")
    }
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

pub fn server() -> Server {
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
