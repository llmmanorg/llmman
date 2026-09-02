//! Prometheus text-format metrics for `llmman serve` — the data behind
//! its `GET /metrics` route (see `cmd::serve::handle_metrics`).
//! Kept in its own module so the exposition format is a pure function
//! over an owned [`Report`], testable without a running daemon.
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
//! a descriptor system and a protobuf encoder to emit the roughly 150
//! lines of text below, none of which this surface needs.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Upper bounds, in seconds, for `llmman_http_request_duration_seconds`.
/// Not Prometheus' default bucket set, which tops out at 10s: an
/// `/api/chat` load-and-generate against a cold model routinely runs into
/// minutes, and every one of those would otherwise land in `+Inf` with no
/// resolution at all. The `+Inf` bucket is not stored — it equals
/// [`RouteStats::count`] by definition, and [`render`] emits it from
/// there.
const DURATION_BUCKETS: [f64; 13] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// Why a model left `ModelManager::running`, as the `reason` label on
/// `llmman_model_unloads_total`. Five closed cases, one per production
/// site that removes an entry, so the label set is bounded by the code
/// rather than by traffic. The distinction is the operationally useful
/// part: steady `Evicted` means `LLMMAN_MAX_LOADED_MODELS` is too low for
/// the workload, steady `Oom` means the host is too small for it, and any
/// `Crashed` at all means a backend is dying under llmman rather than
/// being asked to stop.
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
    /// Every variant, in the order their counters are stored and emitted.
    /// Also sizes [`Registry::model_unloads`], so adding a variant without
    /// a counter slot is a compile error rather than a silent zero.
    const ALL: [Self; 5] = [
        Self::Idle,
        Self::Requested,
        Self::Crashed,
        Self::Oom,
        Self::Evicted,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Requested => "requested",
            Self::Crashed => "crashed",
            Self::Oom => "oom",
            Self::Evicted => "evicted",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Idle => 0,
            Self::Requested => 1,
            Self::Crashed => 2,
            Self::Oom => 3,
            Self::Evicted => 4,
        }
    }
}

/// One route's accumulated samples.
///
/// `statuses` is a `BTreeMap`, not a `HashMap`, for the same reason
/// [`Registry::routes`] is: exposition output has to be byte-stable
/// across scrapes. Taking that from the container rather than from a sort
/// in [`render`] means a later sample type cannot forget to apply it.
#[derive(Clone, Default, PartialEq, Debug)]
struct RouteStats {
    /// Response status code to the number of responses carrying it.
    ///
    /// Bounded by the HTTP status space, not by anything llmman chooses:
    /// `remote_status` passes a provider's own 4xx straight through, so a
    /// misbehaving upstream can produce many distinct values on one route.
    /// Finite, but the widest label on this family — weigh any second one
    /// against it, not against the route count.
    statuses: BTreeMap<u16, u64>,
    /// Cumulative counts for [`DURATION_BUCKETS`], already summed into
    /// each bucket's own `le` — Prometheus histogram buckets are
    /// cumulative, so `render` emits these verbatim.
    buckets: [u64; DURATION_BUCKETS.len()],
    /// Total observed seconds, for the histogram's `_sum`.
    sum_seconds: f64,
    /// Total observations, for the histogram's `_count` and its `+Inf`
    /// bucket.
    count: u64,
}

impl RouteStats {
    fn observe(&mut self, status: u16, elapsed: Duration) {
        *self.statuses.entry(status).or_insert(0) += 1;
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

/// The counters a daemon accumulates between scrapes.
///
/// Real callers use the process-wide [`REGISTRY`] through the
/// `record_*` functions. Tests take a `Registry` of their own via the
/// `*_into` forms, so parallel tests never observe each other's samples —
/// the same split `try_admit_against` uses for `cmd::serve`'s queue
/// counter, and for the same reason.
struct Registry {
    /// Keyed by matched route template (`/llmman/providers/:id`), never a
    /// request's own path — see `cmd::serve::track_metrics`, which reads
    /// axum's `MatchedPath`. That is what bounds this map to the router's
    /// own route list instead of to whatever a client asks for.
    routes: Mutex<BTreeMap<String, RouteStats>>,
    model_loads: AtomicU64,
    model_unloads: [AtomicU64; UnloadReason::ALL.len()],
}

impl Registry {
    const fn new() -> Self {
        Self {
            routes: Mutex::new(BTreeMap::new()),
            model_loads: AtomicU64::new(0),
            model_unloads: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }
}

/// Unix time this daemon started, in seconds, or `0` before
/// [`mark_process_start`] has run.
///
/// Set explicitly at startup rather than lazily on first scrape, so
/// `time() - llmman_start_time_seconds` is real uptime and not time since
/// something first scraped it. Counters here reset to zero when the daemon
/// restarts, and `rate()` copes with that on its own; what it cannot say
/// is whether a flat counter means a restart or a quiet hour.
static PROCESS_START_SECONDS: AtomicU64 = AtomicU64::new(0);

/// Records this daemon's start time. Called once by `cmd::serve`.
pub(crate) fn mark_process_start() {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PROCESS_START_SECONDS.store(secs, Ordering::Relaxed);
}

/// Serialises every test that touches [`REGISTRY`]'s counters — the ones
/// that assert a delta, *and* the ones that only write.
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

/// This daemon's start time, or `0` before [`mark_process_start`] ran.
pub(crate) fn process_start_seconds() -> u64 {
    PROCESS_START_SECONDS.load(Ordering::Relaxed)
}

/// This process's counters. `Relaxed` throughout: no metric here is
/// ordered against another or against any other state, and a scrape is
/// already a snapshot of a moving target — the only requirement is that
/// no increment is lost.
static REGISTRY: Registry = Registry::new();

/// Records one completed response against `registry`.
fn record_request_into(registry: &Registry, route: &str, status: u16, elapsed: Duration) {
    let mut routes = match registry.routes.lock() {
        Ok(routes) => routes,
        // A panicking scrape or recorder must not take the whole daemon's
        // request path down with it: metrics are observability, not
        // service. Recover the guard and carry on with the (still
        // structurally valid) map.
        Err(poisoned) => poisoned.into_inner(),
    };
    match routes.get_mut(route) {
        Some(stats) => stats.observe(status, elapsed),
        None => {
            // Allocates only the first time each route is seen, not on
            // every request.
            let mut stats = RouteStats::default();
            stats.observe(status, elapsed);
            routes.insert(route.to_string(), stats);
        }
    }
}

pub(crate) fn record_request(route: &str, status: u16, elapsed: Duration) {
    record_request_into(&REGISTRY, route, status, elapsed);
}

fn record_model_load_into(registry: &Registry) {
    registry.model_loads.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_model_load() {
    record_model_load_into(&REGISTRY);
}

fn record_model_unload_into(registry: &Registry, reason: UnloadReason) {
    registry.model_unloads[reason.index()].fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_model_unload(reason: UnloadReason) {
    record_model_unload_into(&REGISTRY, reason);
}

/// The values [`Report`] cannot accumulate on its own: they are read from
/// live daemon state at scrape time by `cmd::serve::handle_metrics`.
pub(crate) struct Gauges {
    /// `env!("LLMMAN_VERSION")`, as the `version` label on
    /// `llmman_build_info`.
    pub(crate) version: String,
    /// [`process_start_seconds`]. Passed in rather than read inside
    /// [`render`] so a test sets it outright, and the golden copy does not
    /// depend on whether another test has marked a start yet.
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
    /// when the idle reaper reaches it.
    ///
    /// Not the number `LLMMAN_MAX_LOADED_MODELS` caps.
    /// `enforce_max_loaded_models` gates on `running.len()` plus
    /// outstanding load reservations, so a daemon can be at its limit
    /// while this reads below it. The reservation count is not exposed:
    /// `PendingLoadGuard` releases asynchronously after `ensure_model`
    /// has already inserted into `running`, so it would double-count a
    /// finished load for as long as that task takes to run.
    pub(crate) models_loaded: u64,
}

/// One scrape's worth of data: live [`Gauges`] plus a copy of the
/// counters taken at the same moment.
pub(crate) struct Report {
    gauges: Gauges,
    routes: BTreeMap<String, RouteStats>,
    model_loads: u64,
    model_unloads: [u64; UnloadReason::ALL.len()],
}

/// Snapshots `registry` alongside `gauges`. Copying the counters out
/// under the lock, rather than formatting while holding it, keeps a
/// scrape from blocking the request path for as long as it takes to build
/// a several-kilobyte string.
fn report_from(registry: &Registry, gauges: Gauges) -> Report {
    let routes = match registry.routes.lock() {
        Ok(routes) => routes.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    let mut model_unloads = [0u64; UnloadReason::ALL.len()];
    for reason in UnloadReason::ALL {
        model_unloads[reason.index()] =
            registry.model_unloads[reason.index()].load(Ordering::Relaxed);
    }
    Report {
        gauges,
        routes,
        model_loads: registry.model_loads.load(Ordering::Relaxed),
        model_unloads,
    }
}

pub(crate) fn report(gauges: Gauges) -> Report {
    report_from(&REGISTRY, gauges)
}

/// A label value as Prometheus' exposition format wants it: backslash,
/// double quote and newline escaped. Only `llmman_build_info`'s `version`
/// can currently contain any of them (it is whatever `build.rs` resolved,
/// including a `git describe` on an oddly named tag) — the route, status
/// and reason labels are all closed sets of plain ASCII. Every label is
/// escaped anyway, so a later metric carrying a free-form value is safe
/// without anyone remembering to revisit this.
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

/// Renders `report` as a Prometheus text-format exposition body.
///
/// The gauges come from live daemon state and the counters from this
/// module's own registry, read one after the other rather than under a
/// single lock, so a model can load between the two reads. That is true
/// of every exporter and is what `rate()` tolerates.
///
/// Byte-stable for equal input: metric families are emitted in this
/// function's own source order, and samples within a family in their
/// `BTreeMap`/array order. A scrape that changes only in its numbers must
/// never also change in its line order, or every diff of two scrapes is
/// noise.
pub(crate) fn render(report: &Report) -> String {
    let mut out = String::new();

    out.push_str("# HELP llmman_build_info Build information for this llmman daemon.\n");
    out.push_str("# TYPE llmman_build_info gauge\n");
    out.push_str(&format!(
        "llmman_build_info{{version=\"{}\"}} 1\n",
        escape_label_value(&report.gauges.version)
    ));

    out.push_str(
        "# HELP llmman_start_time_seconds Unix time this daemon started, for uptime and restart detection.\n",
    );
    out.push_str("# TYPE llmman_start_time_seconds gauge\n");
    out.push_str(&format!(
        "llmman_start_time_seconds {}\n",
        report.gauges.start_time_seconds
    ));

    out.push_str("# HELP llmman_scheduling_requests_in_flight Requests doing model-scheduling work; a request for an already-loaded model bypasses this.\n");
    out.push_str("# TYPE llmman_scheduling_requests_in_flight gauge\n");
    out.push_str(&format!(
        "llmman_scheduling_requests_in_flight {}\n",
        report.gauges.scheduling_requests_in_flight
    ));

    out.push_str(
        "# HELP llmman_scheduling_capacity Admission limit for model-scheduling work (LLMMAN_MAX_QUEUE); not a cap on total concurrency.\n",
    );
    out.push_str("# TYPE llmman_scheduling_capacity gauge\n");
    out.push_str(&format!(
        "llmman_scheduling_capacity {}\n",
        report.gauges.scheduling_capacity
    ));

    out.push_str("# HELP llmman_models_loaded Models llmman currently has loaded, the same set /api/ps reports.\n");
    out.push_str("# TYPE llmman_models_loaded gauge\n");
    out.push_str(&format!(
        "llmman_models_loaded {}\n",
        report.gauges.models_loaded
    ));

    out.push_str("# HELP llmman_model_loads_total Models loaded since this daemon started.\n");
    out.push_str("# TYPE llmman_model_loads_total counter\n");
    out.push_str(&format!(
        "llmman_model_loads_total {}\n",
        report.model_loads
    ));

    out.push_str(
        "# HELP llmman_model_unloads_total Models removed from the loaded set since this daemon started, by cause.\n",
    );
    out.push_str("# TYPE llmman_model_unloads_total counter\n");
    for reason in UnloadReason::ALL {
        out.push_str(&format!(
            "llmman_model_unloads_total{{reason=\"{}\"}} {}\n",
            escape_label_value(reason.as_str()),
            report.model_unloads[reason.index()]
        ));
    }

    // Both request families are emitted whole even with no traffic yet,
    // so a fresh daemon still answers a scrape with every family's HELP
    // and TYPE present — a dashboard built against it does not have to
    // wait for the first request to discover what exists.
    out.push_str(
        "# HELP llmman_http_requests_total Responses served, by matched route and status code.\n",
    );
    out.push_str("# TYPE llmman_http_requests_total counter\n");
    for (route, stats) in &report.routes {
        for (status, count) in &stats.statuses {
            out.push_str(&format!(
                "llmman_http_requests_total{{route=\"{}\",status=\"{}\"}} {}\n",
                escape_label_value(route),
                status,
                count
            ));
        }
    }

    out.push_str("# HELP llmman_http_request_duration_seconds Time from llmman receiving a request to it sending that response's headers.\n");
    out.push_str("# TYPE llmman_http_request_duration_seconds histogram\n");
    for (route, stats) in &report.routes {
        let route = escape_label_value(route);
        for (bucket, upper) in stats.buckets.iter().zip(DURATION_BUCKETS) {
            out.push_str(&format!(
                "llmman_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"{upper}\"}} {bucket}\n"
            ));
        }
        out.push_str(&format!(
            "llmman_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"+Inf\"}} {}\n",
            stats.count
        ));
        out.push_str(&format!(
            "llmman_http_request_duration_seconds_sum{{route=\"{route}\"}} {}\n",
            stats.sum_seconds
        ));
        out.push_str(&format!(
            "llmman_http_request_duration_seconds_count{{route=\"{route}\"}} {}\n",
            stats.count
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauges() -> Gauges {
        Gauges {
            version: "0.1.0".into(),
            start_time_seconds: 0,
            scheduling_requests_in_flight: 0,
            scheduling_capacity: 512,
            models_loaded: 0,
        }
    }

    /// A scrape of a daemon that has served nothing still has to be a
    /// valid, complete exposition — every family's HELP and TYPE, and no
    /// stray sample lines under the two request families.
    #[test]
    fn a_daemon_with_no_traffic_still_renders_every_metric_family() {
        let rendered = render(&report_from(&Registry::new(), gauges()));

        for family in [
            "llmman_build_info",
            "llmman_scheduling_requests_in_flight",
            "llmman_scheduling_capacity",
            "llmman_models_loaded",
            "llmman_model_loads_total",
            "llmman_model_unloads_total",
            "llmman_http_requests_total",
            "llmman_http_request_duration_seconds",
        ] {
            assert!(
                rendered.contains(&format!("# TYPE {family} ")),
                "{family} missing from:\n{rendered}"
            );
        }
        assert!(
            !rendered.contains("llmman_http_requests_total{"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("llmman_http_request_duration_seconds_bucket"),
            "{rendered}"
        );
        // Every unload reason is present at zero rather than absent, so a
        // rate() over one does not start undefined.
        assert!(
            rendered.contains("llmman_model_unloads_total{reason=\"evicted\"} 0"),
            "{rendered}"
        );
    }

    /// Two scrapes of the same numbers must be the same bytes. Sample
    /// order coming from the container rather than from a sort is what
    /// this is really checking — a `HashMap` anywhere in the path fails
    /// here rather than in production.
    #[test]
    fn rendering_the_same_report_twice_is_byte_identical() {
        let registry = Registry::new();
        for route in ["/api/chat", "/v1/models", "/api/tags", "/metrics"] {
            for status in [200u16, 500, 404] {
                record_request_into(&registry, route, status, Duration::from_millis(7));
            }
        }
        record_model_load_into(&registry);
        record_model_unload_into(&registry, UnloadReason::Idle);

        let first = render(&report_from(&registry, gauges()));
        let second = render(&report_from(&registry, gauges()));
        assert_eq!(first, second);

        // And that order is sorted, not insertion order: /api/chat was
        // recorded first but /api/tags sorts before it.
        let chat = first.find("route=\"/api/chat\"").expect(&first);
        let tags = first.find("route=\"/api/tags\"").expect(&first);
        assert!(tags > chat, "{first}");
        let models = first.find("route=\"/v1/models\"").expect(&first);
        assert!(models > tags, "{first}");
    }

    /// Bucket semantics, both halves at once. Buckets are cumulative, so an
    /// observation counts in its own bucket and every wider one, and `+Inf`
    /// equals `_count`. The bounds are `le`, not `lt`, so 250ms belongs in
    /// `le="0.25"`. Getting either wrong yields a histogram Prometheus
    /// accepts and every quantile query silently misreads.
    #[test]
    fn buckets_are_cumulative_and_their_bounds_are_inclusive() {
        let registry = Registry::new();
        record_request_into(&registry, "/api/chat", 200, Duration::from_millis(250));
        record_request_into(&registry, "/api/chat", 200, Duration::from_secs(45));

        let rendered = render(&report_from(&registry, gauges()));

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
                    "llmman_http_request_duration_seconds_bucket{{route=\"/api/chat\",le=\"{le}\"}} {expected}\n"
                )),
                "le={le} should be {expected} in:\n{rendered}"
            );
        }
        assert!(
            rendered
                .contains("llmman_http_request_duration_seconds_count{route=\"/api/chat\"} 2\n"),
            "{rendered}"
        );
    }

    /// Each reason counts on its own. A single shared counter would still
    /// show unload churn but never say which of the five causes to fix.
    #[test]
    fn every_unload_reason_counts_separately() {
        let registry = Registry::new();
        record_model_unload_into(&registry, UnloadReason::Evicted);
        record_model_unload_into(&registry, UnloadReason::Evicted);
        record_model_unload_into(&registry, UnloadReason::Crashed);

        let rendered = render(&report_from(&registry, gauges()));
        for (reason, expected) in [
            ("idle", 0),
            ("requested", 0),
            ("crashed", 1),
            ("oom", 0),
            ("evicted", 2),
        ] {
            assert!(
                rendered.contains(&format!(
                    "llmman_model_unloads_total{{reason=\"{reason}\"}} {expected}\n"
                )),
                "{reason} should be {expected} in:\n{rendered}"
            );
        }
    }

    /// The exposition this daemon actually serves, pinned byte for byte.
    ///
    /// Every other test here asserts a fragment, which cannot catch a
    /// series that appeared by accident, one that quietly stopped being
    /// emitted, or a reordering. Those are the changes that break a
    /// dashboard without breaking a test. Regenerate this deliberately
    /// when the surface is meant to change, and never to make a red test
    /// go green.
    const GOLDEN: &str = r#"# HELP llmman_build_info Build information for this llmman daemon.
# TYPE llmman_build_info gauge
llmman_build_info{version="0.1.0 (ab\"cd)"} 1
# HELP llmman_start_time_seconds Unix time this daemon started, for uptime and restart detection.
# TYPE llmman_start_time_seconds gauge
llmman_start_time_seconds 1700000000
# HELP llmman_scheduling_requests_in_flight Requests doing model-scheduling work; a request for an already-loaded model bypasses this.
# TYPE llmman_scheduling_requests_in_flight gauge
llmman_scheduling_requests_in_flight 3
# HELP llmman_scheduling_capacity Admission limit for model-scheduling work (LLMMAN_MAX_QUEUE); not a cap on total concurrency.
# TYPE llmman_scheduling_capacity gauge
llmman_scheduling_capacity 512
# HELP llmman_models_loaded Models llmman currently has loaded, the same set /api/ps reports.
# TYPE llmman_models_loaded gauge
llmman_models_loaded 2
# HELP llmman_model_loads_total Models loaded since this daemon started.
# TYPE llmman_model_loads_total counter
llmman_model_loads_total 2
# HELP llmman_model_unloads_total Models removed from the loaded set since this daemon started, by cause.
# TYPE llmman_model_unloads_total counter
llmman_model_unloads_total{reason="idle"} 1
llmman_model_unloads_total{reason="requested"} 0
llmman_model_unloads_total{reason="crashed"} 0
llmman_model_unloads_total{reason="oom"} 0
llmman_model_unloads_total{reason="evicted"} 1
# HELP llmman_http_requests_total Responses served, by matched route and status code.
# TYPE llmman_http_requests_total counter
llmman_http_requests_total{route="/api/chat",status="200"} 1
llmman_http_requests_total{route="/api/chat",status="503"} 1
llmman_http_requests_total{route="/llmman/providers/:id",status="404"} 1
# HELP llmman_http_request_duration_seconds Time from llmman receiving a request to it sending that response's headers.
# TYPE llmman_http_request_duration_seconds histogram
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="0.05"} 0
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="0.1"} 0
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="0.25"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="0.5"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="1"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="2.5"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="5"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="10"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="30"} 1
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="60"} 2
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="120"} 2
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="300"} 2
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="600"} 2
llmman_http_request_duration_seconds_bucket{route="/api/chat",le="+Inf"} 2
llmman_http_request_duration_seconds_sum{route="/api/chat"} 45.25
llmman_http_request_duration_seconds_count{route="/api/chat"} 2
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="0.05"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="0.1"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="0.25"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="0.5"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="1"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="2.5"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="5"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="10"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="30"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="60"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="120"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="300"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="600"} 1
llmman_http_request_duration_seconds_bucket{route="/llmman/providers/:id",le="+Inf"} 1
llmman_http_request_duration_seconds_sum{route="/llmman/providers/:id"} 0.002
llmman_http_request_duration_seconds_count{route="/llmman/providers/:id"} 1
"#;

    #[test]
    fn the_full_exposition_matches_its_golden_copy() {
        let registry = Registry::new();
        record_request_into(&registry, "/api/chat", 200, Duration::from_millis(250));
        record_request_into(&registry, "/api/chat", 503, Duration::from_secs(45));
        record_request_into(
            &registry,
            "/llmman/providers/:id",
            404,
            Duration::from_millis(2),
        );
        record_model_load_into(&registry);
        record_model_load_into(&registry);
        record_model_unload_into(&registry, UnloadReason::Idle);
        record_model_unload_into(&registry, UnloadReason::Evicted);

        let rendered = render(&report_from(
            &registry,
            Gauges {
                // Carries a quote, so the golden copy also pins escaping.
                version: "0.1.0 (ab\"cd)".into(),
                start_time_seconds: 1_700_000_000,
                scheduling_requests_in_flight: 3,
                scheduling_capacity: 512,
                models_loaded: 2,
            },
        ));

        assert_eq!(rendered, GOLDEN);
    }

    /// xorshift64*, seeded, so this explores a wide input space and still
    /// fails identically on every machine and every run. A fuzzer would
    /// explore more, but would not run in CI across five targets.
    fn next_rand(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// The characters that can actually hurt: the two Prometheus requires
    /// escaped, the line terminator that would forge a new sample, and the
    /// exposition punctuation a value must not be able to impersonate.
    const HOSTILE: &[char] = &[
        '"',
        '\\',
        '\n',
        '\r',
        '\t',
        '{',
        '}',
        '=',
        ' ',
        '#',
        ',',
        '+',
        '\u{0}',
        '\u{7f}',
        'é',
        '中',
        '\u{1F600}',
        'a',
        '1',
        '/',
    ];

    fn hostile_string(state: &mut u64, max_len: usize) -> String {
        let len = (next_rand(state) as usize) % (max_len + 1);
        (0..len)
            .map(|_| HOSTILE[(next_rand(state) as usize) % HOSTILE.len()])
            .collect()
    }

    /// What [`escape_label_value`] promises to preserve: every character
    /// except a carriage return, which the exposition format has no way to
    /// represent.
    fn representable(value: &str) -> String {
        value.replace('\r', "")
    }

    /// Reverses [`escape_label_value`], to prove the escaping is lossless.
    /// A value that survives the round trip cannot have introduced
    /// structure of its own into the output.
    fn unescape_label_value(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        let mut chars = value.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('n') => out.push('\n'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        }
        out
    }

    /// Property: no label value, however hostile, can change the *shape*
    /// of the exposition or fail to survive a round trip — carriage
    /// returns aside, which come back dropped (see [`representable`]).
    ///
    /// `version` is the one free-form value here: whatever `build.rs`
    /// resolved, including a `git describe` on a tag someone else chose.
    /// If a quote or newline in it could close its own label or start a
    /// forged sample line, every downstream parser would read a metric
    /// llmman never emitted.
    #[test]
    fn no_label_value_can_alter_the_shape_of_the_exposition() {
        let mut state = 0x9E3779B97F4A7C15u64;

        for _ in 0..2000 {
            let version = hostile_string(&mut state, 24);
            let route = hostile_string(&mut state, 16);

            let registry = Registry::new();
            record_request_into(&registry, &route, 200, Duration::from_millis(1));
            let rendered = render(&report_from(
                &registry,
                Gauges {
                    version: version.clone(),
                    start_time_seconds: 0,
                    scheduling_requests_in_flight: 1,
                    scheduling_capacity: 1,
                    models_loaded: 1,
                },
            ));

            // Same report shape with benign values: the line count must be
            // identical, or something in the hostile input forged a line.
            let benign = Registry::new();
            record_request_into(&benign, "r", 200, Duration::from_millis(1));
            let reference = render(&report_from(
                &benign,
                Gauges {
                    version: "v".into(),
                    start_time_seconds: 0,
                    scheduling_requests_in_flight: 1,
                    scheduling_capacity: 1,
                    models_loaded: 1,
                },
            ));
            assert_eq!(
                rendered.lines().count(),
                reference.lines().count(),
                "line count changed for version={version:?} route={route:?}"
            );

            // And the values come back out exactly as they went in.
            let emitted_version = rendered
                .lines()
                .find_map(|l| l.strip_prefix("llmman_build_info{version=\""))
                .and_then(|l| l.strip_suffix("\"} 1"))
                .expect("build_info is always emitted");
            assert_eq!(
                unescape_label_value(emitted_version),
                representable(&version),
                "version did not survive the round trip"
            );

            let emitted_route = rendered
                .lines()
                .find_map(|l| l.strip_prefix("llmman_http_requests_total{route=\""))
                .and_then(|l| l.split_once("\",status=").map(|(r, _)| r))
                .expect("the one recorded route is always emitted");
            assert_eq!(
                unescape_label_value(emitted_route),
                representable(&route),
                "route did not survive the round trip"
            );
        }
    }

    /// The property above would still pass if a carriage return were
    /// escaped as `\r` instead of dropped, because this module's own
    /// unescaper would read that back. Prometheus would not: `\r` is not
    /// one of the format's three escape sequences, so it rejects the whole
    /// exposition. Pin both halves — no raw control character, and no
    /// escape the parser will refuse.
    #[test]
    fn a_carriage_return_in_a_label_is_dropped_rather_than_escaped() {
        let escaped = escape_label_value("1.0\r2.0");
        assert_eq!(escaped, "1.02.0");
        assert!(
            !escaped.contains('\r'),
            "a raw carriage return reached the output"
        );
        assert!(
            !escaped.contains("\\r"),
            "an escape sequence Prometheus rejects reached the output"
        );
    }

    /// The golden copy fixes this value, so nothing there would notice a
    /// `mark_process_start` that did nothing — and a daemon reporting 0
    /// makes every uptime query measure from the epoch.
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

    /// Metric names are a public API: a dashboard, an alert rule and a
    /// recording rule all name them directly, and none of them are in this
    /// repository to be updated alongside a rename. Pin the emitted set so
    /// adding or renaming one has to be a deliberate edit here, visible in
    /// review, rather than a side effect of editing `render`.
    #[test]
    fn the_emitted_metric_names_are_a_fixed_set() {
        let rendered = render(&report_from(&Registry::new(), gauges()));
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
                "llmman_models_loaded",
                "llmman_model_loads_total",
                "llmman_model_unloads_total",
                "llmman_http_requests_total",
                "llmman_http_request_duration_seconds",
            ]
        );
    }

    /// `_sum` divided by `_count` is how every dashboard computes average
    /// latency, and nothing else in this module reads `sum_seconds`. An
    /// arithmetic slip here is therefore invisible until a graph is
    /// quietly wrong. 250ms + 750ms is exactly 1s in binary floating
    /// point, so this can assert the rendered text rather than a delta.
    #[test]
    fn the_histogram_sum_is_the_total_observed_seconds() {
        let registry = Registry::new();
        record_request_into(&registry, "/api/chat", 200, Duration::from_millis(250));
        record_request_into(&registry, "/api/chat", 200, Duration::from_millis(750));

        let rendered = render(&report_from(&registry, gauges()));
        assert!(
            rendered.contains("llmman_http_request_duration_seconds_sum{route=\"/api/chat\"} 1\n"),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("llmman_http_request_duration_seconds_count{route=\"/api/chat\"} 2\n"),
            "{rendered}"
        );
    }

    /// The process-wide wrappers are what `cmd::serve` actually calls; the
    /// `*_into` forms are reachable only from tests. A wrapper that
    /// dropped its increment would leave a live daemon reporting flat
    /// counters while every other test here kept passing. Deltas rather
    /// than absolutes because this is the shared registry, which
    /// `cmd::serve`'s own tests also write through production paths.
    #[test]
    fn the_process_wide_wrappers_reach_the_shared_registry() {
        // `blocking_lock` is safe here: this is a plain `#[test]`, so
        // there is no runtime on this thread to block.
        let _serialised = GLOBAL_COUNTER_TEST_LOCK.blocking_lock();
        let before = report(gauges());
        record_model_load();
        record_model_unload(UnloadReason::Oom);
        let after = report(gauges());

        assert_eq!(after.model_loads, before.model_loads + 1);
        assert_eq!(
            after.model_unloads[UnloadReason::Oom.index()],
            before.model_unloads[UnloadReason::Oom.index()] + 1
        );
    }

    /// The label is the whole point of the counter, so a variant added
    /// without one — or one renamed out from under a running dashboard —
    /// should fail here rather than in someone's alerting rules.
    #[test]
    fn unload_reason_labels_are_stable_and_distinct() {
        let labels: Vec<&str> = UnloadReason::ALL.iter().map(|r| r.as_str()).collect();
        assert_eq!(labels, ["idle", "requested", "crashed", "oom", "evicted"]);
        for (i, reason) in UnloadReason::ALL.iter().enumerate() {
            assert_eq!(reason.index(), i, "{reason:?} is stored out of order");
        }
    }
}
