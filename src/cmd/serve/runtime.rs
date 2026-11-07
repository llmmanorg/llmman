//! `--runtime`: where `llmman serve` gets its inference engine. `docker`
//! and `podman` run it in a container (Linux only; `crate::container`),
//! `bin` is llmman's own download of llama.cpp's prebuilt `llama-server`
//! (`crate::llama_release`), `path` is whatever `llama-server` is on
//! `PATH`. `auto`, the default, tries them in that order and takes the
//! first that works. On macOS the same choice governs `mlx_lm.server`
//! (`crate::mlx_release`): `bin` installs llmman's own, `path` uses
//! `PATH`'s, `auto` installs and falls back to `PATH`.
//!
//! `LLMMAN_RUNTIME` is the same setting as an environment variable, since
//! `llmman run`/`launch` start the daemon with a bare `llmman serve`.
//!
//! Two answers are resolved, each on first need ([`Workload`]): the
//! general one for llama-server and the media backend, and a CPU-only
//! one for the speech worker, whose container needs no GPU and runs the
//! CPU image. Under `auto` the two can differ on a host whose GPU probe
//! skips containers for llama-server; explicit runtimes answer both the
//! same way.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::ValueEnum;

use crate::container::{ContainerEngine, ContainerManager};
use crate::llama_release::DownloadMarker;

/// `--runtime` / `LLMMAN_RUNTIME`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub enum Runtime {
    /// Try `docker`, `podman`, `bin`, `path`, in that order.
    Auto,
    /// Engines run in Docker containers (Linux only).
    Docker,
    /// Engines run in Podman containers (Linux only).
    Podman,
    /// llmman's own download of llama.cpp's prebuilt `llama-server`.
    Bin,
    /// Whatever `llama-server` (and, on macOS, `mlx_lm.server`) is on
    /// `PATH`; nothing is ever downloaded.
    Path,
}

impl Runtime {
    /// The container engine this runtime is, if it is one.
    pub fn ociman(self) -> Option<ContainerManager> {
        match self {
            Runtime::Docker => Some(ContainerManager::Docker),
            Runtime::Podman => Some(ContainerManager::Podman),
            Runtime::Auto | Runtime::Bin | Runtime::Path => None,
        }
    }

    /// The `--runtime` spelling, as clap parses it.
    pub fn as_str(self) -> &'static str {
        match self {
            Runtime::Auto => "auto",
            Runtime::Docker => "docker",
            Runtime::Podman => "podman",
            Runtime::Bin => "bin",
            Runtime::Path => "path",
        }
    }

    fn from_ociman(ociman: ContainerManager) -> Runtime {
        match ociman {
            ContainerManager::Docker => Runtime::Docker,
            ContainerManager::Podman => Runtime::Podman,
        }
    }
}

/// A [`Runtime`] with `auto` resolved away: what this daemon will run.
#[derive(Debug, Clone)]
pub enum Resolved {
    /// Engines run in containers; the llama.cpp image is already pulled.
    Container(ContainerManager),
    /// `llama-server` runs as this local binary, obtained the way
    /// `source` (`Bin` or `Path`) says.
    Local { source: Runtime, bin: PathBuf },
}

impl std::fmt::Display for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Resolved::Container(m) => write!(f, "{} containers", m.binary()),
            Resolved::Local { source, bin } => write!(f, "{} ({})", source.as_str(), bin.display()),
        }
    }
}

impl Resolved {
    pub fn runtime(&self) -> Runtime {
        match self {
            Resolved::Container(m) => Runtime::from_ociman(*m),
            Resolved::Local { source, .. } => *source,
        }
    }

    pub fn ociman(&self) -> Option<ContainerManager> {
        match self {
            Resolved::Container(m) => Some(*m),
            Resolved::Local { .. } => None,
        }
    }

    pub fn llama_server_bin(&self) -> Option<&PathBuf> {
        match self {
            Resolved::Container(_) => None,
            Resolved::Local { bin, .. } => Some(bin),
        }
    }
}

/// What a resolution serves. `General` is llama-server and the media
/// backend: the container probe requires the GPU toolkit when the host
/// GPU calls for `--gpus`, the llama-server image is pulled and, under
/// `auto`, verified to start. `CpuOnly` is the speech worker: its
/// container passes no GPU through, so the probe skips the toolkit, the
/// CPU image is pulled ([`ContainerEngine::Speech`]) and nothing is
/// verified. Every other step is the same, in the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    General,
    CpuOnly,
}

impl Workload {
    /// The container engine's image this workload runs.
    fn engine(self) -> ContainerEngine {
        match self {
            Workload::General => ContainerEngine::LlamaServer,
            Workload::CpuOnly => ContainerEngine::Speech,
        }
    }

    /// For the log: which answer a line is about.
    fn label(self) -> &'static str {
        match self {
            Workload::General => "",
            Workload::CpuOnly => " (cpu worker)",
        }
    }
}

/// One memoized answer and the mutex that serializes resolving it.
struct Slot {
    resolved: std::sync::Mutex<Option<Resolved>>,
    // One fetch at a time: loads of different models are not otherwise
    // serialized.
    resolving: tokio::sync::Mutex<()>,
}

impl Slot {
    fn new(resolved: Option<Resolved>) -> Slot {
        Slot {
            resolved: std::sync::Mutex::new(resolved),
            resolving: tokio::sync::Mutex::new(()),
        }
    }

    fn known(&self) -> Option<Resolved> {
        self.resolved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set(&self, resolved: Resolved) {
        *self.resolved.lock().unwrap_or_else(|e| e.into_inner()) = Some(resolved);
    }
}

/// The daemon's runtime: [`resolve`]d at startup when that works, else
/// on the first load that needs it, so a failed llama.cpp fetch is a
/// startup warning rather than a dead daemon. A success is kept; a
/// failure is not, so the next load retries. The CPU-only answer
/// ([`Workload::CpuOnly`]) is resolved by the first speech load and kept
/// the same way; it never touches the general one.
pub struct Lazy {
    requested: Runtime,
    llama_cpp_version: Option<String>,
    general: Slot,
    cpu_only: Slot,
}

impl Lazy {
    /// `resolved` is startup's general result, `None` if that failed.
    pub fn new(
        requested: Runtime,
        llama_cpp_version: Option<String>,
        resolved: Option<Resolved>,
    ) -> Self {
        Lazy {
            requested,
            llama_cpp_version,
            general: Slot::new(resolved),
            cpu_only: Slot::new(None),
        }
    }

    fn slot(&self, workload: Workload) -> &Slot {
        match workload {
            Workload::General => &self.general,
            Workload::CpuOnly => &self.cpu_only,
        }
    }

    /// What `--runtime` asked for (possibly `auto`).
    pub fn requested(&self) -> Runtime {
        self.requested
    }

    /// The general resolution so far, without attempting one.
    pub fn known(&self) -> Option<Resolved> {
        self.general.known()
    }

    fn set(&self, resolved: Resolved) {
        self.general.set(resolved);
    }

    /// The general resolution, fetching llama.cpp now if startup could not.
    pub async fn resolve(&self) -> Result<Resolved> {
        self.resolve_for(Workload::General).await
    }

    /// The resolution for `workload`, made now if no load needed it yet.
    pub async fn resolve_for(&self, workload: Workload) -> Result<Resolved> {
        let slot = self.slot(workload);
        if let Some(resolved) = slot.known() {
            return Ok(resolved);
        }
        let _one_at_a_time = slot.resolving.lock().await;
        if let Some(resolved) = slot.known() {
            return Ok(resolved);
        }
        eprintln!(
            "[llmman] runtime {}{}: not set up at startup; resolving it now",
            self.requested.as_str(),
            workload.label()
        );
        let requested = self.requested;
        let pin = self.llama_cpp_version.clone();
        let resolved =
            tokio::task::spawn_blocking(move || resolve_for(workload, requested, pin.as_deref()))
                .await
                .context("resolve runtime task panicked")?
                .context(
                    "the llama.cpp runtime could not be set up (this daemon started without it)",
                )?;
        slot.set(resolved.clone());
        Ok(resolved)
    }

    /// Whether engines run in containers. Answered without a fetch when
    /// `requested` has no container step (`bin`, `path`, `auto` off
    /// Linux), so a safetensors load never waits on a llama.cpp download
    /// it does not use.
    pub async fn ociman(&self) -> Result<Option<ContainerManager>> {
        self.ociman_for(Workload::General).await
    }

    /// [`ociman`](Self::ociman) for `workload`.
    pub async fn ociman_for(&self, workload: Workload) -> Result<Option<ContainerManager>> {
        if let Some(resolved) = self.slot(workload).known() {
            return Ok(resolved.ociman());
        }
        if !candidates(self.requested, false)
            .iter()
            .any(|c| c.ociman().is_some())
        {
            return Ok(None);
        }
        Ok(self.resolve_for(workload).await?.ociman())
    }

    /// The local `llama-server` to spawn. Re-resolved (under the same
    /// mutex as [`resolve`](Self::resolve)) if the file has gone — the
    /// install that provided it was upgraded or removed while this
    /// daemon ran — rather than failing every load against a dead path.
    pub async fn local_llama_server_bin(&self) -> Result<PathBuf> {
        let local = |resolved: Resolved| match resolved {
            Resolved::Local { source, bin } => Ok((source, bin)),
            Resolved::Container(m) => anyhow::bail!(
                "no local llama-server binary: --runtime {} runs it in a container",
                m.binary()
            ),
        };
        let (_, bin) = local(self.resolve().await?)?;
        if bin.exists() {
            return Ok(bin);
        }
        let _one_at_a_time = self.general.resolving.lock().await;
        let (source, bin) = local(self.known().expect("resolved above"))?;
        if bin.exists() {
            return Ok(bin);
        }
        eprintln!(
            "[llmman] llama-server at {} no longer exists; re-resolving",
            bin.display()
        );
        let pin = self.llama_cpp_version.clone();
        let (_, bin) = tokio::task::spawn_blocking(move || resolve_local(source, pin.as_deref()))
            .await
            .context("resolve llama-server task panicked")??;
        self.set(Resolved::Local {
            source,
            bin: bin.clone(),
        });
        Ok(bin)
    }
}

/// Settles `runtime` on this host and acquires its llama.cpp (image or
/// release build, pinned to `llama_cpp_version`; `None` is upstream's
/// floating latest), so the result is known to work before the listener
/// binds and `--pull-only` has nothing left to do. Under `auto` each
/// failing step is logged and the next tried; off Linux the container
/// steps are skipped. Blocking; call from `spawn_blocking`.
pub fn resolve(runtime: Runtime, llama_cpp_version: Option<&str>) -> Result<Resolved> {
    resolve_for(Workload::General, runtime, llama_cpp_version)
}

/// [`resolve`] for `workload`: the same candidates in the same order,
/// each tried the way that workload needs (see [`Workload`]).
pub fn resolve_for(
    workload: Workload,
    runtime: Runtime,
    llama_cpp_version: Option<&str>,
) -> Result<Resolved> {
    let explicit = runtime != Runtime::Auto;
    resolve_from(
        &candidates(runtime, false),
        runtime,
        workload,
        |candidate| try_one(workload, candidate, llama_cpp_version, explicit),
    )
}

/// [`resolve`] for callers that need a local binary (the mediagen backend
/// dlopens the libraries next to it): `auto` is `bin` then `path`, and a
/// container runtime is an error.
///
/// Returns the source (`Bin` or `Path`) with the binary: only `Bin` is
/// selected by `llama_cpp_version`; a `Path` binary is whatever is
/// installed, whatever the pin says.
pub fn resolve_local(
    runtime: Runtime,
    llama_cpp_version: Option<&str>,
) -> Result<(Runtime, PathBuf)> {
    if let Some(m) = runtime.ociman() {
        anyhow::bail!(
            "--runtime {} runs llama-server in a container; there is no local binary",
            m.binary()
        );
    }
    let explicit = runtime != Runtime::Auto;
    let step = |candidate| try_one(Workload::General, candidate, llama_cpp_version, explicit);
    match resolve_from(&candidates(runtime, true), runtime, Workload::General, step)? {
        Resolved::Local { source, bin } => Ok((source, bin)),
        Resolved::Container(_) => unreachable!("bin and path resolve to a local binary"),
    }
}

/// Whether resolving `runtime` may land on a container engine: an
/// explicit `docker` or `podman`, or `auto` on Linux. `bin` and `path`
/// pull no image, so a caller that only needs to know which image to
/// warm has nothing to classify for them.
pub fn has_container_candidate(runtime: Runtime) -> bool {
    candidates(runtime, false)
        .iter()
        .any(|candidate| candidate.ociman().is_some())
}

/// What `runtime` expands to, in order.
fn candidates(runtime: Runtime, local_only: bool) -> Vec<Runtime> {
    match runtime {
        Runtime::Auto if cfg!(target_os = "linux") && !local_only => {
            vec![
                Runtime::Docker,
                Runtime::Podman,
                Runtime::Bin,
                Runtime::Path,
            ]
        }
        Runtime::Auto => vec![Runtime::Bin, Runtime::Path],
        one => vec![one],
    }
}

/// Tries `candidates` in order with `step` (one [`try_one`], injectable
/// for tests), taking the first success. An explicit `requested` has one
/// candidate and returns its own error; `auto` logs each skip and reports
/// them all when none works.
fn resolve_from(
    candidates: &[Runtime],
    requested: Runtime,
    workload: Workload,
    mut step: impl FnMut(Runtime) -> Result<Resolved>,
) -> Result<Resolved> {
    let explicit = requested != Runtime::Auto;
    let mut failures = Vec::new();
    for &candidate in candidates {
        match step(candidate) {
            Ok(resolved) => {
                eprintln!(
                    "[llmman] runtime {}{}: using {resolved}",
                    requested.as_str(),
                    workload.label()
                );
                return Ok(resolved);
            }
            Err(e) if explicit => return Err(e),
            Err(e) => {
                eprintln!(
                    "[llmman] runtime auto{}: skipping {}: {e:#}",
                    workload.label(),
                    candidate.as_str()
                );
                failures.push(format!("{}: {e:#}", candidate.as_str()));
            }
        }
    }
    anyhow::bail!(
        "no usable llama.cpp runtime (tried {}); set --runtime/LLMMAN_RUNTIME to pick one \
         and see its own error",
        failures.join("; ")
    )
}

/// One step of [`resolve_for`]. `explicit` skips the container probe:
/// the user asked for that engine, and its own errors are clearer. The
/// workload picks the probe, the image and whether to verify it; `Bin`
/// and `Path` are the same for both.
fn try_one(
    workload: Workload,
    candidate: Runtime,
    llama_cpp_version: Option<&str>,
    explicit: bool,
) -> Result<Resolved> {
    match candidate {
        Runtime::Auto => unreachable!("auto is expanded by candidates()"),
        Runtime::Docker | Runtime::Podman => {
            let ociman = candidate.ociman().expect("container runtime");
            if !cfg!(target_os = "linux") {
                anyhow::bail!(
                    "--runtime {} is only supported on Linux",
                    candidate.as_str()
                );
            }
            if !explicit {
                match workload {
                    Workload::General => ociman.probe()?,
                    Workload::CpuOnly => ociman.probe_cpu_only()?,
                }
            }
            let verify = !explicit && workload == Workload::General;
            with_download_marker(|marker| {
                prepare_container(
                    marker,
                    || {
                        crate::container::pull_image(
                            ociman,
                            workload.engine(),
                            llama_cpp_version,
                            None,
                        )
                    },
                    verify.then_some(|| {
                        crate::container::verify_llama_server_runs(ociman, llama_cpp_version)
                    }),
                )
            })?;
            Ok(Resolved::Container(ociman))
        }
        Runtime::Bin => {
            let bin = crate::llama_release::ensure_llama_server(llama_cpp_version)
                .context("download of llama.cpp's prebuilt llama-server failed")?
                .bin;
            Ok(Resolved::Local {
                source: Runtime::Bin,
                bin,
            })
        }
        Runtime::Path => {
            let bin = crate::find_on_path("llama-server").context("no llama-server on PATH")?;
            Ok(Resolved::Local {
                source: Runtime::Path,
                bin,
            })
        }
    }
}

/// Runs `setup` (a `docker pull` and the check after it, possibly
/// minutes) holding the download marker with its heartbeat, so
/// `daemon::ensure_server` leaves a daemon still setting up alive.
fn with_download_marker<T>(setup: impl FnOnce(&DownloadMarker) -> T) -> T {
    let marker = DownloadMarker::create();
    marker.keep_alive_during(|| setup(&marker))
}

/// A container runtime's setup: `pull` the image, then (`--runtime auto`
/// only) `verify` it starts. Each step names itself on `marker`, held
/// throughout: the client's 60s budget counts time with no live marker,
/// and `verify` may itself run for `container::VERIFY_TIMEOUT` (60s).
fn prepare_container(
    marker: &DownloadMarker,
    pull: impl FnOnce() -> Result<()>,
    verify: Option<impl FnOnce() -> Result<()>>,
) -> Result<()> {
    marker.set_status(crate::container::PULL_STATUS);
    pull()?;
    if let Some(verify) = verify {
        marker.set_status(crate::container::VERIFY_STATUS);
        verify()?;
    }
    Ok(())
}

/// `--llama-cpp-version` as a pin: unset is
/// [`crate::llama_release::default_release`], `latest` is `None`.
pub fn llama_cpp_pin(arg: Option<&str>) -> Option<String> {
    match arg {
        Some("latest") => None,
        Some(v) => Some(v.to_owned()),
        None => Some(crate::llama_release::default_release().to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_spellings_round_trip() {
        for (s, r) in [
            ("auto", Runtime::Auto),
            ("docker", Runtime::Docker),
            ("podman", Runtime::Podman),
            ("bin", Runtime::Bin),
            ("path", Runtime::Path),
        ] {
            assert_eq!(Runtime::from_str(s, true).unwrap(), r);
            assert_eq!(r.as_str(), s);
        }
    }

    #[test]
    fn only_container_runtimes_have_an_ociman() {
        assert_eq!(Runtime::Docker.ociman(), Some(ContainerManager::Docker));
        assert_eq!(Runtime::Podman.ociman(), Some(ContainerManager::Podman));
        assert_eq!(Runtime::Auto.ociman(), None);
        assert_eq!(Runtime::Bin.ociman(), None);
        assert_eq!(Runtime::Path.ociman(), None);
    }

    #[test]
    fn pin_defaults_to_the_ci_release_and_latest_unpins() {
        let default = crate::llama_release::default_release();
        assert!(default.starts_with('b'), "{default}");
        assert_eq!(llama_cpp_pin(None).as_deref(), Some(default));
        assert_eq!(llama_cpp_pin(Some("latest")), None);
        assert_eq!(llama_cpp_pin(Some("b9994")).as_deref(), Some("b9994"));
    }

    /// The marker file the client polls, at a path of this test's own.
    fn marker_at_temp_path(name: &str) -> (DownloadMarker, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "llmman-runtime-marker-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        (DownloadMarker::create_at(path.clone()), path)
    }

    /// The marker is live, and says what it waits on, through both steps.
    #[test]
    fn the_marker_is_live_and_says_what_it_is_waiting_on_through_pull_and_verify() {
        let (marker, path) = marker_at_temp_path("verify");
        let seen = std::cell::RefCell::new(Vec::new());
        let note = || {
            seen.borrow_mut()
                .push(std::fs::read_to_string(&path).expect("the marker exists"));
            Ok(())
        };
        prepare_container(&marker, note, Some(note)).unwrap();
        assert_eq!(
            *seen.borrow(),
            [
                crate::container::PULL_STATUS,
                crate::container::VERIFY_STATUS
            ]
        );
    }

    /// An explicit runtime skips the check; a failed pull never reaches it.
    #[test]
    fn verify_is_skipped_for_an_explicit_runtime_and_after_a_failed_pull() {
        let (marker, _path) = marker_at_temp_path("skip");
        let verified = std::cell::Cell::new(false);
        let verify = || {
            verified.set(true);
            Ok(())
        };
        prepare_container(&marker, || Ok(()), None::<fn() -> Result<()>>).unwrap();
        assert!(!verified.get());
        let err = prepare_container(&marker, || anyhow::bail!("pull failed"), Some(verify))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "pull failed");
        assert!(!verified.get());
    }

    /// The order `auto` tries, with the container step injected: the
    /// general answer skips an engine whose probe fails and lands on
    /// `bin`; the CPU-only answer, probing without the toolkit rule,
    /// takes that same engine.
    #[test]
    fn auto_skips_a_refused_container_for_the_general_answer_and_takes_it_cpu_only() {
        let order = [
            Runtime::Docker,
            Runtime::Podman,
            Runtime::Bin,
            Runtime::Path,
        ];
        let tried = std::cell::RefCell::new(Vec::new());
        // What the probes decide on a GPU host without the toolkit.
        let step = |workload: Workload| {
            let tried = &tried;
            move |candidate: Runtime| -> Result<Resolved> {
                tried.borrow_mut().push(candidate);
                match (candidate, workload) {
                    (Runtime::Docker | Runtime::Podman, Workload::General) => {
                        anyhow::bail!("no NVIDIA Container Toolkit")
                    }
                    (Runtime::Docker | Runtime::Podman, Workload::CpuOnly) => {
                        Ok(Resolved::Container(candidate.ociman().expect("container")))
                    }
                    (Runtime::Bin, _) => Ok(Resolved::Local {
                        source: Runtime::Bin,
                        bin: PathBuf::from("/tmp/llama-server"),
                    }),
                    (other, _) => anyhow::bail!("{} not reached", other.as_str()),
                }
            }
        };
        let general = resolve_from(
            &order,
            Runtime::Auto,
            Workload::General,
            step(Workload::General),
        )
        .unwrap();
        assert_eq!(general.runtime(), Runtime::Bin);
        assert_eq!(
            *tried.borrow(),
            [Runtime::Docker, Runtime::Podman, Runtime::Bin]
        );
        tried.borrow_mut().clear();
        let cpu_only = resolve_from(
            &order,
            Runtime::Auto,
            Workload::CpuOnly,
            step(Workload::CpuOnly),
        )
        .unwrap();
        assert_eq!(cpu_only.ociman(), Some(ContainerManager::Docker));
        assert_eq!(*tried.borrow(), [Runtime::Docker]);
    }

    /// An explicit runtime has one candidate and its own error; `auto`
    /// with nothing usable names every step it tried.
    #[test]
    fn an_explicit_runtime_returns_its_own_error_and_auto_lists_every_skip() {
        let err = resolve_from(
            &[Runtime::Docker],
            Runtime::Docker,
            Workload::General,
            |_| anyhow::bail!("docker info failed"),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(err, "docker info failed");
        let err = resolve_from(
            &[Runtime::Bin, Runtime::Path],
            Runtime::Auto,
            Workload::CpuOnly,
            |candidate| anyhow::bail!("{} failed", candidate.as_str()),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("bin: bin failed; path: path failed"), "{err}");
    }

    /// Each workload keeps its own answer; the general one is never set
    /// by a CPU-only resolution.
    #[tokio::test]
    async fn the_two_answers_are_kept_apart() {
        let lazy = Lazy::new(
            Runtime::Path,
            None,
            Some(Resolved::Local {
                source: Runtime::Path,
                bin: PathBuf::from("/usr/bin/llama-server"),
            }),
        );
        assert_eq!(lazy.known().map(|r| r.runtime()), Some(Runtime::Path));
        assert!(lazy.slot(Workload::CpuOnly).known().is_none());
        lazy.slot(Workload::CpuOnly)
            .set(Resolved::Container(ContainerManager::Docker));
        assert_eq!(
            lazy.ociman_for(Workload::CpuOnly).await.unwrap(),
            Some(ContainerManager::Docker)
        );
        assert_eq!(lazy.ociman().await.unwrap(), None);
        assert_eq!(lazy.known().map(|r| r.runtime()), Some(Runtime::Path));
    }

    #[test]
    fn container_runtimes_are_refused_off_linux() {
        if cfg!(target_os = "linux") {
            return;
        }
        // Refused before any docker/podman binary is looked for, so this
        // is deterministic on a macOS/Windows developer machine too.
        let err = resolve(Runtime::Docker, None).unwrap_err().to_string();
        assert!(err.contains("only supported on Linux"), "{err}");
    }

    #[test]
    fn resolved_reports_its_concrete_runtime() {
        let local = Resolved::Local {
            source: Runtime::Path,
            bin: PathBuf::from("/usr/bin/llama-server"),
        };
        assert_eq!(local.runtime(), Runtime::Path);
        assert_eq!(local.ociman(), None);
        assert_eq!(
            local.llama_server_bin(),
            Some(&PathBuf::from("/usr/bin/llama-server"))
        );
        let container = Resolved::Container(ContainerManager::Podman);
        assert_eq!(container.runtime(), Runtime::Podman);
        assert_eq!(container.ociman(), Some(ContainerManager::Podman));
        assert_eq!(container.llama_server_bin(), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_keeps_a_startup_resolution_as_is() {
        let startup = Resolved::Local {
            source: Runtime::Path,
            bin: PathBuf::from("/nonexistent/llama-server"),
        };
        // A resolve would fail on this path, proving none is made.
        let lazy = Lazy::new(Runtime::Bin, Some("b0".into()), Some(startup));
        assert_eq!(lazy.known().unwrap().runtime(), Runtime::Path);
        assert_eq!(lazy.ociman().await.unwrap(), None);
        assert_eq!(
            lazy.resolve().await.unwrap().llama_server_bin(),
            Some(&PathBuf::from("/nonexistent/llama-server"))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_answers_ociman_without_fetching_under_a_local_runtime() {
        for requested in [Runtime::Bin, Runtime::Path] {
            let lazy = Lazy::new(requested, Some("b0".into()), None);
            assert_eq!(lazy.ociman().await.unwrap(), None);
            assert!(lazy.known().is_none(), "{requested:?}");
        }
        if !cfg!(target_os = "linux") {
            let lazy = Lazy::new(Runtime::Auto, Some("b0".into()), None);
            assert_eq!(lazy.ociman().await.unwrap(), None);
            assert!(lazy.known().is_none());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_does_not_cache_a_failed_resolution() {
        if cfg!(target_os = "linux") {
            return;
        }
        // Refused before any probe off Linux, so deterministic.
        let lazy = Lazy::new(Runtime::Docker, None, None);
        let err = lazy.ociman().await.unwrap_err().to_string();
        assert!(err.contains("could not be set up"), "{err}");
        assert!(lazy.known().is_none(), "a failure must not be remembered");
        assert!(
            lazy.resolve().await.is_err(),
            "and the next attempt retries"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lazy_set_replaces_the_resolution() {
        let lazy = Lazy::new(Runtime::Path, None, None);
        lazy.set(Resolved::Container(ContainerManager::Docker));
        assert_eq!(lazy.ociman().await.unwrap(), Some(ContainerManager::Docker));
        lazy.set(Resolved::Local {
            source: Runtime::Path,
            bin: PathBuf::from("/x/llama-server"),
        });
        assert_eq!(lazy.resolve().await.unwrap().runtime(), Runtime::Path);
    }

    /// Only a runtime that may land on a container engine needs the
    /// model classified before the pull.
    #[test]
    fn container_candidates_are_the_container_runtimes_and_auto_on_linux() {
        assert!(has_container_candidate(Runtime::Docker));
        assert!(has_container_candidate(Runtime::Podman));
        assert!(!has_container_candidate(Runtime::Bin));
        assert!(!has_container_candidate(Runtime::Path));
        assert_eq!(
            has_container_candidate(Runtime::Auto),
            cfg!(target_os = "linux")
        );
    }
}
