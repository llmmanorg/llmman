/// Pre-gzipped static web UI assets, embedded at build time.
///
/// Each constant holds the raw gzip-compressed bytes for the corresponding
/// file from the `webui/` directory.  Serve them with the headers
/// `Content-Encoding: gzip` and the appropriate `Content-Type`.
pub static INDEX_HTML: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/index.html.gz"));

pub static BUNDLE_JS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/bundle.js.gz"));

pub static BUNDLE_CSS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/bundle.css.gz"));

pub static LOADING_HTML: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/loading.html.gz"));

/// llmman live's own page, script and stylesheet — see the
/// `/llmman/live` handlers in `cmd::serve`. Unlike the bundle above these
/// three are hand-written source files, not build output, so they are
/// edited in `webui/` directly.
pub static LIVE_HTML: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/live.html.gz"));

pub static LIVE_JS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/live.js.gz"));

pub static LIVE_CSS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/webui_gz/live.css.gz"));
