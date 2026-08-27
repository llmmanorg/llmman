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
