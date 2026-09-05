//! Prometheus text-format metrics for `llmman serve` — the data behind
//! its `GET /metrics` route (see `cmd::serve::handle_metrics`), which
//! only exists when `LLMMAN_METRICS=1`.
//!
//! Kept in its own module so the exposition format is a pure function
//! over an owned snapshot, testable without a running daemon.
//!
//! Scope is deliberately daemon-level: request flow, queue admission and
//! model lifecycle — what only llmman knows. Per-token counters (prompt
//! and eval totals, KV-cache usage) are *not* here, because llmman does
//! not run inference itself. llama-server does, and it already publishes
//! exactly those on its own `/metrics` behind `--metrics`. Re-deriving
//! them here would mean either scraping every backend on every scrape,
//! or keeping a second copy of numbers the engine already owns and
//! watching it drift. Ollama's server *is* its runner, so counters like
//! `ollama_eval_total` belong in the process that produced them; llmman
//! is not that process.
//!
//! No new dependency: a `prometheus` client crate would bring a registry,
//! a descriptor system and a protobuf encoder to emit the text below,
//! none of which this surface needs.
//!
//! Everything accumulated lives in one [`Store`] behind one `Mutex`: a
//! `BTreeMap` of counters and a `BTreeMap` of histograms, both keyed by
//! metric name plus an already-rendered label string. Every family is a
//! few lines at its record site and three lines in [`render`], so adding
//! one costs metrics rather than machinery.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Upper bounds, in seconds, shared by every histogram here.
///
/// Not Prometheus' default bucket set, which tops out at 10s: an
/// `/api/chat` load-and-generate against a cold model routinely runs into
/// minutes, and every one of those would otherwise land in `+Inf` with no
/// resolution at all. Model loads need the same range for the same
/// reason, so they share this rather than getting a second constant —
/// the lowest few buckets are then dead weight on that family, which
/// costs four lines of a scrape and no accuracy.
///
/// The `+Inf` bucket is not stored: it equals [`Histogram::count`] by
/// definition, and [`render`] emits it from there.
const DURATION_BUCKETS: [f64; 13] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// Why a model left `ModelManager::running`, as the `reason` label on
/// `llmman_model_unloads_total`. Five closed cases, one per production
/// site that removes an entry, so that half of the label set is bounded
/// by the code rather than by traffic. The distinction is the
/// operationally useful part: steady `Evicted` means
/// `LLMMAN_MAX_LOADED_MODELS` is too low for the workload, steady `Oom`
/// means the host is too small for it, and any `Crashed` at all means a
/// backend is dying under llmman rather than being asked to stop.
///
/// Each counter marks the moment the model left `ModelManager::running`,
/// which is when it stops serving requests. `Oom` and `Evicted` then stop
/// the backend process, so for as long as that takes the memory the
/// eviction was meant to free is still held. Read this as "no longer
/// serving", not "its memory is back".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum UnloadReason {
    /// `reap_idle_models_once`: idle past its `keep_alive` deadline.
    Idle,
    /// `unload_model`: an explicit API or `llmman stop` unload.
    Requested,
    /// `check_running`: the backend process had already exited.
    Crashed,
    /// `evict_other_models`: freeing VRAM for `ensure_model`'s OOM retry.
    Oom,
    /// `enforce_max_loaded_models`: freeing a `LLMMAN_MAX_LOADED_MODELS`
    /// slot for a new load.
    Evicted,
}

impl UnloadReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Requested => "requested",
            Self::Crashed => "crashed",
            Self::Oom => "oom",
            Self::Evicted => "evicted",
        }
    }
}

/// Which fallback `ensure_model`'s OOM retry loop reached for, as the
/// `strategy` label on `llmman_model_load_oom_retries_total`. Three
/// closed cases, in the order that loop tries them — which is also the
/// order of how invasive they are, so a daemon steadily reaching
/// `CtxShrink` is serving a smaller context than its configuration asks
/// for and nothing else says so.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum OomRetry {
    /// Evicted every other loaded model and retried unchanged.
    EvictOthers,
    /// Lifted `LLMMAN_SCHED_SPREAD=0` and retried across every GPU.
    SplitMode,
    /// Retried with a smaller `--ctx-size` than this daemon had picked.
    CtxShrink,
}

impl OomRetry {
    fn as_str(self) -> &'static str {
        match self {
            Self::EvictOthers => "evict_others",
            Self::SplitMode => "split_mode",
            Self::CtxShrink => "ctx_shrink",
        }
    }
}

/// One series' labels, rendered and escaped once at record time:
/// `model="qwen3.5:0.8b",reason="idle"`, or empty for an unlabelled
/// series.
///
/// Stored rather than kept as pairs because it is also half the map key:
/// ordering the key by this string is what makes a scrape byte-stable
/// without [`render`] having to sort anything itself.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Labels(String);

impl Labels {
    fn new(pairs: &[(&str, &str)]) -> Self {
        let mut out = String::new();
        for (name, value) in pairs {
            if !out.is_empty() {
                out.push(',');
            }
            out.push_str(name);
            out.push_str("=\"");
            out.push_str(&escape_label_value(value));
            out.push('"');
        }
        Self(out)
    }

    /// `{model="a"}`, or nothing at all when there are no labels — a
    /// bare `{}` is legal but noisy, and no other exporter emits it.
    fn braces(&self) -> String {
        if self.0.is_empty() {
            String::new()
        } else {
            format!("{{{}}}", self.0)
        }
    }

    /// `{model="a",le="0.05"}` — a histogram bucket's own labels plus the
    /// bound, with the same empty-label case handled.
    fn with_le(&self, upper: &str) -> String {
        if self.0.is_empty() {
            format!("{{le=\"{upper}\"}}")
        } else {
            format!("{{{},le=\"{upper}\"}}", self.0)
        }
    }
}

/// A metric name plus the labels of one series under it.
type Key = (&'static str, Labels);

/// One series' accumulated observations.
#[derive(Clone, Default, PartialEq, Debug)]
struct Histogram {
    /// Cumulative counts for [`DURATION_BUCKETS`], already summed into
    /// each bucket's own `le` — Prometheus histogram buckets are
    /// cumulative, so [`render`] emits these verbatim.
    buckets: [u64; DURATION_BUCKETS.len()],
    /// Total observed seconds, for the histogram's `_sum`.
    sum_seconds: f64,
    /// Total observations, for the histogram's `_count` and its `+Inf`
    /// bucket.
    count: u64,
}

impl Histogram {
    fn observe(&mut self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        self.sum_seconds += seconds;
        self.count += 1;
        for (bucket, upper) in self.buckets.iter_mut().zip(DURATION_BUCKETS) {
            if seconds <= upper {
                *bucket += 1;
            }
        }
    }
}

/// Everything a daemon accumulates between scrapes.
///
/// `BTreeMap`, not `HashMap`: exposition output has to be byte-stable
/// across scrapes, and taking that from the container rather than from a
/// sort in [`render`] means a later family cannot forget to apply it.
/// Keying by name-then-labels also groups every series of a family
/// together, which the format requires.
///
/// Real callers use the process-wide [`REGISTRY`] through the `record_*`
/// functions. Tests take a `Store` of their own via the `*_into` forms,
/// so parallel tests never observe each other's samples — the same split
/// `try_admit_against` uses for `cmd::serve`'s queue counter, and for the
/// same reason.
#[derive(Default)]
struct Store {
    counters: BTreeMap<Key, u64>,
    histograms: BTreeMap<Key, Histogram>,
}

impl Store {
    const fn new() -> Self {
        Self {
            counters: BTreeMap::new(),
            histograms: BTreeMap::new(),
        }
    }
}

/// This process's metrics.
static REGISTRY: Mutex<Store> = Mutex::new(Store::new());

/// Unix time this daemon started, in seconds, or `0` before
/// [`mark_process_start`] has run.
///
/// Set explicitly at startup rather than lazily on first scrape, so
/// `time() - llmman_start_time_seconds` is real uptime and not time since
/// something first scraped it. Counters here reset to zero when the daemon
/// restarts, and `rate()` copes with that on its own; what it cannot say
/// is whether a flat counter means a restart or a quiet hour.
///
/// Its own atomic rather than a [`Store`] entry: it is written once, from
/// `serve_async`, before any request path exists to contend with.
static PROCESS_START_SECONDS: AtomicU64 = AtomicU64::new(0);

/// Records this daemon's start time. Called once by `cmd::serve`.
pub(crate) fn mark_process_start() {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PROCESS_START_SECONDS.store(secs, Ordering::Relaxed);
}

/// This daemon's start time, or `0` before [`mark_process_start`] ran.
pub(crate) fn process_start_seconds() -> u64 {
    PROCESS_START_SECONDS.load(Ordering::Relaxed)
}

/// Serialises every test that touches [`REGISTRY`] — the ones that
/// assert a delta, *and* the ones that only write.
///
/// Production paths in `cmd::serve` write these counters too, and the
/// tests that exercise those paths run in this same binary, in parallel.
/// A test reading before and after its own write can otherwise see
/// another one's increment land in between. Guarding only the readers
/// does not close that: an unguarded writer still runs inside a reader's
/// window, which is the race itself. Both sides hold this, wherever they
/// live.
///
/// `tokio::sync::Mutex` because `cmd::serve`'s tests hold it across an
/// await; a `std` guard there is a clippy error, and rightly so.
#[cfg(test)]
pub(crate) static GLOBAL_COUNTER_TEST_LOCK: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

/// Runs `f` against `store`.
///
/// A panicking scrape or recorder must not take the whole daemon's
/// request path down with it: metrics are observability, not service. A
/// poisoned lock is recovered and the (still structurally valid) maps
/// carry on.
fn with_store<R>(store: &Mutex<Store>, f: impl FnOnce(&mut Store) -> R) -> R {
    let mut guard = match store.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut guard)
}

fn incr_into(store: &Mutex<Store>, name: &'static str, labels: Labels) {
    with_store(store, |s| {
        *s.counters.entry((name, labels)).or_insert(0) += 1
    });
}

fn observe_into(store: &Mutex<Store>, name: &'static str, labels: Labels, elapsed: Duration) {
    with_store(store, |s| {
        s.histograms
            .entry((name, labels))
            .or_default()
            .observe(elapsed)
    });
}

/// Records one response's status and time to its first byte against
/// `store`. `route` is a matched route template, never a request's own
/// path — see `cmd::serve::track_metrics`.
fn record_request_into(store: &Mutex<Store>, route: &str, status: u16, ttfb: Duration) {
    incr_into(
        store,
        "llmman_http_requests_total",
        Labels::new(&[("route", route), ("status", &status.to_string())]),
    );
    observe_into(
        store,
        "llmman_http_request_ttfb_seconds",
        Labels::new(&[("route", route)]),
        ttfb,
    );
}

pub(crate) fn record_request(route: &str, status: u16, ttfb: Duration) {
    record_request_into(&REGISTRY, route, status, ttfb);
}

fn record_response_body_into(store: &Mutex<Store>, route: &str, elapsed: Duration) {
    observe_into(
        store,
        "llmman_http_request_duration_seconds",
        Labels::new(&[("route", route)]),
        elapsed,
    );
}

/// Records a response's total time, once its body has finished being
/// relayed — see `cmd::serve::track_metrics`.
pub(crate) fn record_response_body(route: &str, elapsed: Duration) {
    record_response_body_into(&REGISTRY, route, elapsed);
}

fn record_model_load_into(store: &Mutex<Store>, model: &str, elapsed: Duration) {
    let labels = Labels::new(&[("model", model)]);
    incr_into(store, "llmman_model_loads_total", labels.clone());
    observe_into(store, "llmman_model_load_duration_seconds", labels, elapsed);
}

/// Records one completed cold start: a model that was not loaded is now
/// in `ModelManager::running` and answering.
pub(crate) fn record_model_load(model: &str, elapsed: Duration) {
    record_model_load_into(&REGISTRY, model, elapsed);
}

fn record_model_unload_into(store: &Mutex<Store>, model: &str, reason: UnloadReason) {
    incr_into(
        store,
        "llmman_model_unloads_total",
        Labels::new(&[("model", model), ("reason", reason.as_str())]),
    );
}

pub(crate) fn record_model_unload(model: &str, reason: UnloadReason) {
    record_model_unload_into(&REGISTRY, model, reason);
}

fn record_oom_retry_into(store: &Mutex<Store>, model: &str, strategy: OomRetry) {
    incr_into(
        store,
        "llmman_model_load_oom_retries_total",
        Labels::new(&[("model", model), ("strategy", strategy.as_str())]),
    );
}

pub(crate) fn record_oom_retry(model: &str, strategy: OomRetry) {
    record_oom_retry_into(&REGISTRY, model, strategy);
}

fn record_scheduling_rejection_into(store: &Mutex<Store>) {
    incr_into(
        store,
        "llmman_scheduling_rejections_total",
        Labels::default(),
    );
}

/// Records one request turned away by `try_admit` with a 503.
pub(crate) fn record_scheduling_rejection() {
    record_scheduling_rejection_into(&REGISTRY);
}

/// One loaded model's live state, for `llmman_model_up`.
pub(crate) struct ModelState {
    pub(crate) model: String,
    /// `llama-server`, `vllm` or `mlx` — see `cmd::serve::Engine`.
    pub(crate) engine: &'static str,
    /// Whether the backend process is still alive right now
    /// (`ModelProcess::is_alive`), not whether llmman still lists it.
    pub(crate) up: bool,
}

/// The values [`Store`] cannot accumulate on its own: they are read from
/// live daemon state at scrape time by `cmd::serve::handle_metrics`.
pub(crate) struct Snapshot {
    /// `env!("LLMMAN_VERSION")`, as the `version` label on
    /// `llmman_build_info`.
    pub(crate) version: String,
    /// [`process_start_seconds`]. Passed in rather than read inside
    /// [`render`] so a test sets it outright.
    pub(crate) start_time_seconds: u64,
    /// Callers currently past `ensure_model`'s already-loaded fast path.
    ///
    /// Not queued and not all in-flight requests. `try_admit` admits or
    /// rejects with a 503, so nothing waits in a queue; and a request for
    /// an already-loaded model returns before `try_admit` is reached, so
    /// it is never counted here at all. What this measures is the
    /// requests doing model-scheduling work.
    pub(crate) scheduling_requests_in_flight: u64,
    /// The cap `try_admit` actually enforces, which is
    /// `LLMMAN_MAX_QUEUE.max(1)` and not the raw setting — see
    /// `try_admit_against`'s doc comment on why `0` admits one at a time
    /// rather than none.
    pub(crate) scheduling_capacity: u64,
    /// `ModelManager::running.len()` — llmman's own record, which is the
    /// same set `/api/ps` reports, so the two always agree. A backend that
    /// exits on its own still counts here until llmman notices, which
    /// happens on the next request for that model (`check_running`) or
    /// when the idle reaper reaches it; `llmman_model_up` is what reads 0
    /// in the meantime.
    pub(crate) models_loaded: u64,
    /// `ModelManager::pending_loads` — loads admitted by
    /// `enforce_max_loaded_models` but not yet in `running`. Added to
    /// `models_loaded`, this is exactly the number
    /// `LLMMAN_MAX_LOADED_MODELS` caps.
    pub(crate) models_loading: u64,
    /// One entry per model in `ModelManager::running`.
    pub(crate) models: Vec<ModelState>,
}

/// A label value as Prometheus' exposition format wants it: backslash,
/// double quote and newline escaped.
///
/// Two of those are reachable today. A model reference is a label value
/// now, and `shortnames::validate_reference` accepts an absolute path
/// (`/models/My "Model"/x.gguf`) with any non-control character in it,
/// backslashes and double quotes included — so a legitimate local import
/// produces a label value that would otherwise end the label, and with it
/// the line. `\n` cannot get through that same validation (it is a
/// control character) and neither can `\r`, but a helper named for
/// escaping label values that handles two of the format's three sequences
/// is a trap for whoever adds the next label.
///
/// A carriage return is dropped rather than escaped. The format defines
/// those three sequences and no more, so `\r` fails the parse of the whole
/// exposition rather than of its own line — `promtool check metrics`
/// rejects it with "invalid escape sequence '\r'". Left raw it parses, but
/// puts a bare control character inside a line-oriented format.
fn escape_label_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => {}
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Emits one family's `HELP` and `TYPE` header.
fn header(out: &mut String, name: &str, kind: &str, help: &str) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
}

/// Emits a gauge read from live state, with `labels` already escaped.
fn gauge(out: &mut String, name: &str, help: &str, labels: &Labels, value: impl std::fmt::Display) {
    header(out, name, "gauge", help);
    out.push_str(&format!("{name}{} {value}\n", labels.braces()));
}

/// Emits every accumulated series of one counter family, in key order.
///
/// A family with no samples yet still gets its `HELP` and `TYPE`, but no
/// sample line: its label values are model references and route
/// templates, and there is no zero to emit for a series nobody has
/// produced yet. A dashboard still learns from the header what exists.
fn counter(out: &mut String, store: &Store, name: &str, help: &str) {
    header(out, name, "counter", help);
    for ((_, labels), value) in store.counters.iter().filter(|((f, _), _)| *f == name) {
        out.push_str(&format!("{name}{} {value}\n", labels.braces()));
    }
}

/// Emits the one series of a counter family that carries no labels.
///
/// Unlike [`counter`], this always emits a sample, zero included. There
/// is exactly one series it could ever have, so `rate()` over it — and
/// any alert written on that rate — would otherwise return nothing at all
/// until the first rejection, which is precisely the daemon nobody needs
/// to be alerted about.
fn counter_unlabelled(out: &mut String, store: &Store, name: &'static str, help: &str) {
    header(out, name, "counter", help);
    let value = store
        .counters
        .get(&(name, Labels::default()))
        .copied()
        .unwrap_or(0);
    out.push_str(&format!("{name} {value}\n"));
}

/// Emits every accumulated series of one histogram family, in key order.
fn histogram(out: &mut String, store: &Store, name: &str, help: &str) {
    header(out, name, "histogram", help);
    for ((_, labels), hist) in store.histograms.iter().filter(|((f, _), _)| *f == name) {
        for (count, upper) in hist.buckets.iter().zip(DURATION_BUCKETS) {
            out.push_str(&format!(
                "{name}_bucket{} {count}\n",
                labels.with_le(&upper.to_string())
            ));
        }
        out.push_str(&format!(
            "{name}_bucket{} {}\n",
            labels.with_le("+Inf"),
            hist.count
        ));
        out.push_str(&format!(
            "{name}_sum{} {}\n",
            labels.braces(),
            hist.sum_seconds
        ));
        out.push_str(&format!("{name}_count{} {}\n", labels.braces(), hist.count));
    }
}

/// Renders one scrape body against `store`.
///
/// The gauges come from live daemon state and the counters from this
/// module's own store, read one after the other rather than under a
/// single lock, so a model can load between the two reads. That is true
/// of every exporter and is what `rate()` tolerates.
///
/// Byte-stable for equal input: families are emitted in this function's
/// own source order, and series within a family in their `BTreeMap` key
/// order. A scrape that changes only in its numbers must never also
/// change in its line order, or every diff of two scrapes is noise.
fn render_from(store: &Store, snapshot: &Snapshot) -> String {
    let mut out = String::new();
    let none = Labels::default();

    gauge(
        &mut out,
        "llmman_build_info",
        "Build information for this llmman daemon.",
        &Labels::new(&[("version", &snapshot.version)]),
        1,
    );
    gauge(
        &mut out,
        "llmman_start_time_seconds",
        "Unix time this daemon started, for uptime and restart detection.",
        &none,
        snapshot.start_time_seconds,
    );
    gauge(
        &mut out,
        "llmman_scheduling_requests_in_flight",
        "Requests doing model-scheduling work; a request for an already-loaded model bypasses this.",
        &none,
        snapshot.scheduling_requests_in_flight,
    );
    gauge(
        &mut out,
        "llmman_scheduling_capacity",
        "Admission limit for model-scheduling work (LLMMAN_MAX_QUEUE); not a cap on total concurrency.",
        &none,
        snapshot.scheduling_capacity,
    );
    counter_unlabelled(
        &mut out,
        store,
        "llmman_scheduling_rejections_total",
        "Requests rejected with 503 because LLMMAN_MAX_QUEUE was already full.",
    );
    gauge(
        &mut out,
        "llmman_models_loaded",
        "Models llmman currently has loaded, the same set /api/ps reports.",
        &none,
        snapshot.models_loaded,
    );
    gauge(
        &mut out,
        "llmman_models_loading",
        "Loads in flight: admitted, not yet loaded. Added to llmman_models_loaded this is what LLMMAN_MAX_LOADED_MODELS caps.",
        &none,
        snapshot.models_loading,
    );

    header(
        &mut out,
        "llmman_model_up",
        "gauge",
        "1 while a loaded model's backend process is alive, 0 once it has exited and llmman has not yet noticed.",
    );
    let mut model_up: BTreeMap<Labels, u8> = BTreeMap::new();
    for model in &snapshot.models {
        model_up.insert(
            Labels::new(&[("model", &model.model), ("engine", model.engine)]),
            u8::from(model.up),
        );
    }
    for (labels, up) in &model_up {
        out.push_str(&format!("llmman_model_up{} {up}\n", labels.braces()));
    }

    counter(
        &mut out,
        store,
        "llmman_model_loads_total",
        "Models loaded since this daemon started.",
    );
    histogram(
        &mut out,
        store,
        "llmman_model_load_duration_seconds",
        "Cold start: from llmman finding a model unloaded to it answering, including any queue wait, pull and eviction.",
    );
    counter(
        &mut out,
        store,
        "llmman_model_load_oom_retries_total",
        "Load attempts retried after an out-of-memory failure, by which fallback was used.",
    );
    counter(
        &mut out,
        store,
        "llmman_model_unloads_total",
        "Models removed from the loaded set since this daemon started, by cause.",
    );
    counter(
        &mut out,
        store,
        "llmman_http_requests_total",
        "Responses served, by matched route and status code.",
    );
    histogram(
        &mut out,
        store,
        "llmman_http_request_ttfb_seconds",
        "Time from llmman receiving a request to it sending that response's headers.",
    );
    histogram(
        &mut out,
        store,
        "llmman_http_request_duration_seconds",
        "Time from llmman receiving a request to it finishing that response's body, streaming included.",
    );

    out
}

/// Renders one scrape body from this process's own metrics.
///
/// Copies the store under its lock, then formats without it: a scrape
/// must not block the request path for as long as it takes to build a
/// several-kilobyte string.
pub(crate) fn render(snapshot: &Snapshot) -> String {
    let store = with_store(&REGISTRY, |s| Store {
        counters: s.counters.clone(),
        histograms: s.histograms.clone(),
    });
    render_from(&store, snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Mutex<Store> {
        Mutex::new(Store::new())
    }

    /// Renders `store` the way [`render`] does the real registry.
    fn rendered(store: &Mutex<Store>, snapshot: &Snapshot) -> String {
        with_store(store, |s| render_from(s, snapshot))
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            version: "0.1.0".into(),
            start_time_seconds: 0,
            scheduling_requests_in_flight: 0,
            scheduling_capacity: 512,
            models_loaded: 0,
            models_loading: 0,
            models: Vec::new(),
        }
    }

    /// The value of one rendered series, or `0` when it has no samples.
    fn value_of(rendered: &str, prefix: &str) -> u64 {
        rendered
            .lines()
            .find_map(|l| l.strip_prefix(prefix))
            .and_then(|rest| rest.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Metric names are a public API: dashboards and alert rules name
    /// them directly and live outside this repository, so adding or
    /// renaming one has to be a deliberate edit here.
    #[test]
    fn the_emitted_metric_names_are_a_fixed_set() {
        let rendered = rendered(&store(), &snapshot());
        let names: Vec<&str> = rendered
            .lines()
            .filter_map(|l| l.strip_prefix("# TYPE "))
            .filter_map(|l| l.split_whitespace().next())
            .collect();
        assert_eq!(
            names,
            [
                "llmman_build_info",
                "llmman_start_time_seconds",
                "llmman_scheduling_requests_in_flight",
                "llmman_scheduling_capacity",
                "llmman_scheduling_rejections_total",
                "llmman_models_loaded",
                "llmman_models_loading",
                "llmman_model_up",
                "llmman_model_loads_total",
                "llmman_model_load_duration_seconds",
                "llmman_model_load_oom_retries_total",
                "llmman_model_unloads_total",
                "llmman_http_requests_total",
                "llmman_http_request_ttfb_seconds",
                "llmman_http_request_duration_seconds",
            ]
        );
    }

    /// Sample order comes from a sorted container, not insertion order:
    /// a `HashMap` anywhere in the path fails here, not in production.
    #[test]
    fn rendering_the_same_data_twice_is_byte_identical() {
        let store = store();
        for route in ["/api/chat", "/v1/models", "/api/tags", "/metrics"] {
            for status in [200u16, 500, 404] {
                record_request_into(&store, route, status, Duration::from_millis(7));
            }
        }
        record_model_load_into(&store, "b:latest", Duration::from_secs(3));
        record_model_unload_into(&store, "b:latest", UnloadReason::Idle);

        let first = rendered(&store, &snapshot());
        let second = rendered(&store, &snapshot());
        assert_eq!(first, second);

        // /api/chat was recorded first but /api/tags sorts before it.
        let chat = first.find("route=\"/api/chat\"").expect(&first);
        let tags = first.find("route=\"/api/tags\"").expect(&first);
        assert!(tags > chat, "{first}");
        let models = first.find("route=\"/v1/models\"").expect(&first);
        assert!(models > tags, "{first}");
    }

    /// Buckets are cumulative and their bounds are `le`, not `lt`. Getting
    /// either wrong yields a histogram Prometheus accepts and every
    /// quantile query silently misreads.
    #[test]
    fn buckets_are_cumulative_and_their_bounds_are_inclusive() {
        let store = store();
        record_request_into(&store, "/api/chat", 200, Duration::from_millis(250));
        record_request_into(&store, "/api/chat", 200, Duration::from_secs(45));

        let rendered = rendered(&store, &snapshot());

        for (le, expected) in [
            ("0.1", 0),  // below both
            ("0.25", 1), // exactly the 250ms observation's bound: le, not lt
            ("0.5", 1),
            ("30", 1),
            ("60", 2), // 45s joins here and stays in every wider bucket
            ("600", 2),
            ("+Inf", 2),
        ] {
            assert!(
                rendered.contains(&format!(
                    "llmman_http_request_ttfb_seconds_bucket{{route=\"/api/chat\",le=\"{le}\"}} {expected}\n"
                )),
                "le={le} should be {expected} in:\n{rendered}"
            );
        }
        assert!(
            rendered.contains("llmman_http_request_ttfb_seconds_count{route=\"/api/chat\"} 2\n"),
            "{rendered}"
        );
    }

    /// `_sum / _count` is how every dashboard computes average latency,
    /// and nothing else here reads `sum_seconds`. 250ms + 750ms is exactly
    /// 1s in binary floating point, so the text itself is assertable.
    #[test]
    fn the_histogram_sum_is_the_total_observed_seconds() {
        let store = store();
        record_model_load_into(&store, "a:latest", Duration::from_millis(250));
        record_model_load_into(&store, "a:latest", Duration::from_millis(750));

        let rendered = rendered(&store, &snapshot());
        assert!(
            rendered.contains("llmman_model_load_duration_seconds_sum{model=\"a:latest\"} 1\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llmman_model_load_duration_seconds_count{model=\"a:latest\"} 2\n"),
            "{rendered}"
        );
    }

    #[test]
    fn ttfb_and_total_duration_are_separate_families() {
        let store = store();
        record_request_into(&store, "/api/chat", 200, Duration::from_millis(250));
        record_response_body_into(&store, "/api/chat", Duration::from_secs(45));

        let rendered = rendered(&store, &snapshot());
        assert!(
            rendered.contains("llmman_http_request_ttfb_seconds_sum{route=\"/api/chat\"} 0.25\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llmman_http_request_duration_seconds_sum{route=\"/api/chat\"} 45\n"),
            "{rendered}"
        );
    }

    #[test]
    fn every_model_and_reason_counts_separately() {
        let store = store();
        record_model_unload_into(&store, "a:latest", UnloadReason::Evicted);
        record_model_unload_into(&store, "a:latest", UnloadReason::Evicted);
        record_model_unload_into(&store, "a:latest", UnloadReason::Crashed);
        record_model_unload_into(&store, "b:latest", UnloadReason::Evicted);

        let rendered = rendered(&store, &snapshot());
        for (model, reason, expected) in [
            ("a:latest", "evicted", 2),
            ("a:latest", "crashed", 1),
            ("b:latest", "evicted", 1),
        ] {
            assert!(
                rendered.contains(&format!(
                    "llmman_model_unloads_total{{model=\"{model}\",reason=\"{reason}\"}} {expected}\n"
                )),
                "{model}/{reason} should be {expected} in:\n{rendered}"
            );
        }
        // And nothing invents a series for a reason that never happened.
        assert!(!rendered.contains("reason=\"idle\""), "{rendered}");
    }

    #[test]
    fn oom_retries_count_by_strategy() {
        let store = store();
        record_oom_retry_into(&store, "a:latest", OomRetry::EvictOthers);
        record_oom_retry_into(&store, "a:latest", OomRetry::CtxShrink);
        record_oom_retry_into(&store, "a:latest", OomRetry::CtxShrink);

        let rendered = rendered(&store, &snapshot());
        for (strategy, expected) in [("evict_others", 1), ("ctx_shrink", 2)] {
            assert!(
                rendered.contains(&format!(
                    "llmman_model_load_oom_retries_total{{model=\"a:latest\",strategy=\"{strategy}\"}} {expected}\n"
                )),
                "{strategy} should be {expected} in:\n{rendered}"
            );
        }
    }

    /// A backend that exited without llmman noticing is the one case no
    /// other metric here can show.
    #[test]
    fn a_dead_backend_reads_down_while_llmman_still_lists_it() {
        let mut snapshot = snapshot();
        snapshot.models_loaded = 2;
        snapshot.models = vec![
            ModelState {
                model: "a:latest".into(),
                engine: "llama-server",
                up: true,
            },
            ModelState {
                model: "b:latest".into(),
                engine: "vllm",
                up: false,
            },
        ];

        let rendered = rendered(&store(), &snapshot);
        assert!(
            rendered.contains("llmman_model_up{model=\"a:latest\",engine=\"llama-server\"} 1\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llmman_model_up{model=\"b:latest\",engine=\"vllm\"} 0\n"),
            "{rendered}"
        );
        assert!(rendered.contains("llmman_models_loaded 2\n"), "{rendered}");
    }

    /// Zero-filled, because it has exactly one series it could ever have;
    /// and as a bare name rather than `name{}`, which is legal but reads
    /// as a bug in every dashboard's label editor.
    #[test]
    fn an_unlabelled_counter_zero_fills_and_renders_without_empty_braces() {
        let store = store();
        let cold = rendered(&store, &snapshot());
        assert!(
            cold.contains("llmman_scheduling_rejections_total 0\n"),
            "{cold}"
        );

        record_scheduling_rejection_into(&store);
        record_scheduling_rejection_into(&store);
        let rendered = rendered(&store, &snapshot());
        assert!(
            rendered.contains("llmman_scheduling_rejections_total 2\n"),
            "{rendered}"
        );
        assert!(!rendered.contains("{}"), "{rendered}");
    }

    /// Why [`escape_label_value`] exists: `validate_reference` rejects
    /// control characters and nothing else, so a quote in a model path is
    /// a valid reference that would otherwise end its own label early.
    #[test]
    fn a_model_reference_can_legitimately_contain_a_quote() {
        let path = "/models/My \"Model\"/x.gguf";
        assert!(
            crate::shortnames::validate_reference(path).is_ok(),
            "{path} is a reference llmman accepts, so it can reach a label"
        );

        let store = store();
        record_model_load_into(&store, path, Duration::from_secs(1));
        let rendered = rendered(&store, &snapshot());
        assert!(
            rendered.contains(
                "llmman_model_loads_total{model=\"/models/My \\\"Model\\\"/x.gguf\"} 1\n"
            ),
            "{rendered}"
        );
    }

    /// The format's three escape sequences, and the one control character
    /// it has none for: `\r` is dropped, because Prometheus rejects an
    /// exposition containing `\\r`.
    #[test]
    fn escape_label_value_covers_the_three_sequences_and_drops_a_carriage_return() {
        assert_eq!(escape_label_value("a\\b"), "a\\\\b");
        assert_eq!(escape_label_value("a\"b"), "a\\\"b");
        assert_eq!(escape_label_value("a\nb"), "a\\nb");
        assert_eq!(escape_label_value("1.0\r2.0"), "1.02.0");
        assert_eq!(escape_label_value("plain/0.1"), "plain/0.1");
    }

    /// A daemon reporting 0 makes every uptime query measure from the
    /// epoch.
    #[test]
    fn marking_the_process_start_records_a_plausible_unix_time() {
        mark_process_start();
        let marked = process_start_seconds();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the system clock is after the unix epoch")
            .as_secs();
        assert!(marked > 1_700_000_000, "{marked} is not a plausible time");
        assert!(marked <= now, "{marked} is ahead of now ({now})");
    }

    /// The `*_into` forms are reachable only from tests; these wrappers
    /// are what `cmd::serve` calls. Deltas, because the registry is shared
    /// with `cmd::serve`'s own tests.
    #[test]
    fn the_process_wide_wrappers_reach_the_shared_registry() {
        // A plain `#[test]`, so there is no runtime on this thread to block.
        let _serialised = GLOBAL_COUNTER_TEST_LOCK.blocking_lock();
        const LOADS: &str = "llmman_model_loads_total{model=\"wrapper-probe\"}";
        const UNLOADS: &str = "llmman_model_unloads_total{model=\"wrapper-probe\",reason=\"oom\"}";
        const RETRIES: &str =
            "llmman_model_load_oom_retries_total{model=\"wrapper-probe\",strategy=\"ctx_shrink\"}";
        const REJECTIONS: &str = "llmman_scheduling_rejections_total ";

        let before = render(&snapshot());
        record_model_load("wrapper-probe", Duration::from_secs(1));
        record_model_unload("wrapper-probe", UnloadReason::Oom);
        record_oom_retry("wrapper-probe", OomRetry::CtxShrink);
        record_scheduling_rejection();
        let after = render(&snapshot());

        assert_eq!(value_of(&after, LOADS), value_of(&before, LOADS) + 1);
        assert_eq!(value_of(&after, UNLOADS), value_of(&before, UNLOADS) + 1);
        assert_eq!(value_of(&after, RETRIES), value_of(&before, RETRIES) + 1);
        assert_eq!(
            value_of(&after, REJECTIONS),
            value_of(&before, REJECTIONS) + 1
        );
    }

    /// A variant added without a label, or one renamed under a running
    /// dashboard, fails here rather than in someone's alerting rules.
    #[test]
    fn enum_labels_are_stable() {
        let unloads: Vec<&str> = [
            UnloadReason::Idle,
            UnloadReason::Requested,
            UnloadReason::Crashed,
            UnloadReason::Oom,
            UnloadReason::Evicted,
        ]
        .iter()
        .map(|r| r.as_str())
        .collect();
        assert_eq!(unloads, ["idle", "requested", "crashed", "oom", "evicted"]);

        let retries: Vec<&str> = [
            OomRetry::EvictOthers,
            OomRetry::SplitMode,
            OomRetry::CtxShrink,
        ]
        .iter()
        .map(|r| r.as_str())
        .collect();
        assert_eq!(retries, ["evict_others", "split_mode", "ctx_shrink"]);
    }

    /// Metrics are observability, not service: a panic while a recorder
    /// held the lock must not stop every later recording.
    #[test]
    fn a_poisoned_lock_does_not_stop_recording() {
        let store = store();
        record_scheduling_rejection_into(&store);

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = store.lock().unwrap();
            panic!("a recorder panicked while holding the lock");
        }));
        assert!(poisoned.is_err());
        assert!(store.is_poisoned());

        record_scheduling_rejection_into(&store);
        let rendered = rendered(&store, &snapshot());
        assert!(
            rendered.contains("llmman_scheduling_rejections_total 2\n"),
            "{rendered}"
        );
    }
}
