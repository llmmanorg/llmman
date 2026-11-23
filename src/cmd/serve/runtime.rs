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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::ValueEnum;

use crate::container::{ContainerEngine, ContainerManager};
use crate::llama_release::{DownloadMarker, LlamaPin};

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
/// container passes no GPU through, so the probe skips the toolkit, and
/// the CPU image is pulled ([`ContainerEngine::Speech`]) and, under
/// `auto`, verified to start. Every other step is the same, in the same
/// order.
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

    /// The backend whose image the start check runs: the host's for the
    /// general answer, the CPU one for a CPU-only worker.
    fn backend(self) -> crate::container::GpuBackend {
        match self {
            Workload::General => crate::container::detect_backend(),
            Workload::CpuOnly => crate::container::SPEECH_BACKEND,
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
    llama_cpp_version: LlamaPin,
    hardened: bool,
    general: Slot,
    cpu_only: Slot,
}

impl Lazy {
    /// `resolved` is startup's general result, `None` if that failed.
    pub fn new(
        requested: Runtime,
        llama_cpp_version: LlamaPin,
        hardened: bool,
        resolved: Option<Resolved>,
    ) -> Self {
        Lazy {
            requested,
            llama_cpp_version,
            hardened,
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
        let hardened = self.hardened;
        let resolved =
            tokio::task::spawn_blocking(move || resolve_for(workload, requested, &pin, hardened))
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
    /// mutex as [`resolve`](Self::resolve)) if the file has gone â€” the
    /// install that provided it was upgraded or removed while this
    /// daemon ran â€” rather than failing every load against a dead path.
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
        let (_, bin) = tokio::task::spawn_blocking(move || resolve_local(source, &pin))
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
/// release build, pinned by `llama_cpp_version`), so the result is known
/// to work before the listener binds and `--pull-only` has nothing left
/// to do. `hardened` is the daemon's setting (on unless `--no-hardened`):
/// the verify run under `auto` starts the image with the same
/// `container::hardened_args` block the real spawn will. Under `auto`
/// each failing step is logged and the next tried; off Linux the
/// container steps are skipped. Blocking; call from `spawn_blocking`.
pub fn resolve(runtime: Runtime, llama_cpp_version: &LlamaPin, hardened: bool) -> Result<Resolved> {
    resolve_for(Workload::General, runtime, llama_cpp_version, hardened)
}

/// [`resolve`] for `workload`: the same candidates in the same order,
/// each tried the way that workload needs (see [`Workload`]).
pub fn resolve_for(
    workload: Workload,
    runtime: Runtime,
    llama_cpp_version: &LlamaPin,
    hardened: bool,
) -> Result<Resolved> {
    resolve_with(workload, runtime, llama_cpp_version, hardened, false)
}

/// The one resolution loop: `runtime`'s candidates (local ones only when
/// `local_only`), each tried the way `workload` needs, an explicit
/// runtime reporting its own error.
fn resolve_with(
    workload: Workload,
    runtime: Runtime,
    llama_cpp_version: &LlamaPin,
    hardened: bool,
    local_only: bool,
) -> Result<Resolved> {
    let explicit = runtime != Runtime::Auto;
    resolve_from(
        &candidates(runtime, local_only),
        runtime,
        workload,
        |candidate| try_one(workload, candidate, llama_cpp_version, hardened, explicit),
    )
}

/// [`resolve`] for callers that need a local binary (the mediagen backend
/// dlopens the libraries next to it): `auto` is `bin` then `path`, and a
/// container runtime is an error.
///
/// Returns the source (`Bin` or `Path`) with the binary: only `Bin` is
/// selected by `llama_cpp_version`; a `Path` binary is whatever is
/// installed, whatever the pin says.
pub fn resolve_local(runtime: Runtime, llama_cpp_version: &LlamaPin) -> Result<(Runtime, PathBuf)> {
    if let Some(m) = runtime.ociman() {
        anyhow::bail!(
            "--runtime {} runs llama-server in a container; there is no local binary",
            m.binary()
        );
    }
    // No container, so there is no hardened block to carry.
    match resolve_with(Workload::General, runtime, llama_cpp_version, false, true)? {
        Resolved::Local { source, bin } => Ok((source, bin)),
        Resolved::Container(_) => unreachable!("bin and path resolve to a local binary"),
    }
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
/// workload picks the probe, the image and whether to verify it; the
/// verify run starts the image with the daemon's `hardened` block, so
/// the probe proves the argv the real spawn will use. `Bin` and `Path`
/// are the same for both.
fn try_one(
    workload: Workload,
    candidate: Runtime,
    llama_cpp_version: &LlamaPin,
    hardened: bool,
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
            prime_docker_identity(candidate, explicit, hardened, || {
                crate::container::identity::docker_is_rootless();
            });
            // Under `auto` both answers check that the engine can start
            // the image it just pulled: an engine that answers `info` but
            // cannot run a container (rootless without systemd, say) must
            // fall through to `bin` for a speech load as it does for
            // llama-server.
            with_download_marker(|marker| {
                prepare_container(
                    marker,
                    || {
                        crate::container::pull_image(
                            ociman,
                            workload.engine(),
                            llama_cpp_version,
                            None,
                            None,
                        )
                    },
                    (!explicit).then_some(|| {
                        crate::container::verify_llama_server_runs(
                            ociman,
                            workload.backend(),
                            llama_cpp_version,
                            hardened,
                        )
                    }),
                )
            })?;
            Ok(Resolved::Container(ociman))
        }
        Runtime::Bin => {
            let bin = crate::llama_release::ensure_llama_server(llama_cpp_version.tag())
                .context("download of llama.cpp's prebuilt llama-server failed")?
                .bin;
            verify_local_llama_server_runs(&bin)?;
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

const VERIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Runs `llama-server --version`, like
/// [`crate::container::verify_llama_server_runs`]: `auto` must not settle on
/// a build this host cannot start. A failure carries the decoded exit status.
fn verify_local_llama_server_runs(bin: &Path) -> Result<()> {
    let mut child = std::process::Command::new(bin)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            let hint = if e.raw_os_error() == Some(crate::exit_status::ERROR_CODE_INTEGRITY) {
                format!(": {}", crate::exit_status::CODE_INTEGRITY_HINT)
            } else {
                String::new()
            };
            anyhow::anyhow!("could not run {} --version: {e}{hint}", bin.display())
        })?;
    // Drained while waiting, so a chatty child cannot block on a full pipe.
    let mut pipe = child.stderr.take().expect("stderr is piped");
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut pipe, &mut text);
        text
    });
    let status = crate::container::wait_with_timeout(&mut child, VERIFY_TIMEOUT)?;
    let stderr = stderr.join().unwrap_or_default();
    match status {
        Some(status) if status.success() => Ok(()),
        Some(status) => {
            let lines: Vec<_> = stderr
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect();
            let tail = &lines[lines.len().saturating_sub(3)..];
            let tail = if tail.is_empty() {
                String::new()
            } else {
                format!(": {}", tail.join(" | "))
            };
            anyhow::bail!(
                "{} --version failed ({}){tail}",
                bin.display(),
                crate::exit_status::describe(&status)
            )
        }
        None => anyhow::bail!(
            "{} --version did not exit within {VERIFY_TIMEOUT:?}",
            bin.display()
        ),
    }
}

/// Settles the rootless question of the first hardened spawn here, on
/// the blocking resolve path, for an explicit `--runtime docker` on
/// Linux: `probe` skipped [`ContainerManager::probe`], which primes it
/// under `auto`, and the spawns run on the async request path where a
/// `docker info` waiting out `PROBE_TIMEOUT` would stall the runtime.
/// Nothing runs under `--no-hardened`: no spawn reads the answer then.
fn prime_docker_identity(candidate: Runtime, explicit: bool, hardened: bool, probe: impl FnOnce()) {
    if hardened && explicit && candidate == Runtime::Docker && cfg!(target_os = "linux") {
        probe();
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

/// `--llama-cpp-version` as a pin: unset is [`LlamaPin::Default`] (the
/// release [`crate::llama_release::default_release`] names), `latest` is
/// [`LlamaPin::Latest`], anything else [`LlamaPin::Explicit`], the
/// default's own tag included: only the implicit default is pulled by
/// digest (see `container::GpuBackend::image_ref`).
pub fn llama_cpp_pin(arg: Option<&str>) -> LlamaPin {
    match arg {
        Some("latest") => LlamaPin::Latest,
        Some(v) => LlamaPin::Explicit(v.to_owned()),
        None => LlamaPin::Default,
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
        assert!(default.starts_with('v'), "{default}");
        assert_eq!(llama_cpp_pin(None), LlamaPin::Default);
        assert_eq!(llama_cpp_pin(None).tag(), Some(default));
        assert_eq!(llama_cpp_pin(Some("latest")), LlamaPin::Latest);
        assert_eq!(llama_cpp_pin(Some("latest")).tag(), None);
        assert_eq!(
            llama_cpp_pin(Some("b9994")),
            LlamaPin::Explicit("b9994".into())
        );
        // The default's own tag, spelled out, is explicit: tag-only pulls.
        assert_eq!(
            llama_cpp_pin(Some(default)),
            LlamaPin::Explicit(default.to_owned())
        );
        assert_ne!(llama_cpp_pin(Some(default)), LlamaPin::Default);
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

    /// An explicit `--runtime docker` primes the rootless answer while
    /// still on the blocking resolve path; `auto` gets it from the
    /// engine probe, podman never asks, and `--no-hardened` skips it.
    #[test]
    fn explicit_docker_primes_the_rootless_answer_before_any_spawn() {
        let probed = |candidate, explicit, hardened| {
            let hit = std::cell::Cell::new(false);
            prime_docker_identity(candidate, explicit, hardened, || hit.set(true));
            hit.get()
        };
        let linux = cfg!(target_os = "linux");
        assert_eq!(probed(Runtime::Docker, true, true), linux);
        assert!(!probed(Runtime::Docker, true, false));
        assert!(!probed(Runtime::Docker, false, true));
        assert!(!probed(Runtime::Podman, true, true));
        assert!(!probed(Runtime::Bin, true, true));
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
            LlamaPin::Latest,
            true,
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
        let err = resolve(Runtime::Docker, &LlamaPin::Latest, true)
            .unwrap_err()
            .to_string();
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
        let lazy = Lazy::new(
            Runtime::Bin,
            LlamaPin::Explicit("b0".into()),
            true,
            Some(startup),
        );
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
            let lazy = Lazy::new(requested, LlamaPin::Explicit("b0".into()), true, None);
            assert_eq!(lazy.ociman().await.unwrap(), None);
            assert!(lazy.known().is_none(), "{requested:?}");
        }
        if !cfg!(target_os = "linux") {
            let lazy = Lazy::new(Runtime::Auto, LlamaPin::Explicit("b0".into()), true, None);
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
        let lazy = Lazy::new(Runtime::Docker, LlamaPin::Latest, true, None);
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
        let lazy = Lazy::new(Runtime::Path, LlamaPin::Latest, true, None);
        lazy.set(Resolved::Container(ContainerManager::Docker));
        assert_eq!(lazy.ociman().await.unwrap(), Some(ContainerManager::Docker));
        lazy.set(Resolved::Local {
            source: Runtime::Path,
            bin: PathBuf::from("/x/llama-server"),
        });
        assert_eq!(lazy.resolve().await.unwrap().runtime(), Runtime::Path);
    }

    /// A stand-in `llama-server` that prints `stderr` and exits `code`.
    fn fake_llama_server(name: &str, code: i32, stderr: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llmman-verify-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (file, body) = if cfg!(windows) {
            (
                "llama-server.cmd",
                format!("@echo off\r\necho {stderr} 1>&2\r\nexit /b {code}\r\n"),
            )
        } else {
            (
                "llama-server",
                format!("#!/bin/sh\necho '{stderr}' >&2\nexit {code}\n"),
            )
        };
        let bin = dir.join(file);
        std::fs::write(&bin, body).unwrap();
        crate::llama_release::mark_executable(&bin).unwrap();
        bin
    }

    #[test]
    fn verify_accepts_a_binary_whose_version_runs() {
        let bin = fake_llama_server("ok", 0, "version: 1");
        verify_local_llama_server_runs(&bin).unwrap();
    }

    #[test]
    fn verify_reports_the_exit_status_and_stderr_of_a_failing_binary() {
        let bin = fake_llama_server("fail", 127, "loader says no");
        let err = format!("{:#}", verify_local_llama_server_runs(&bin).unwrap_err());
        assert!(err.contains("--version failed"), "{err}");
        assert!(err.contains("exit code 127"), "{err}");
        assert!(err.contains("loader says no"), "{err}");
    }

    #[test]
    fn verify_reports_a_binary_that_cannot_be_started() {
        let err = verify_local_llama_server_runs(Path::new("/nonexistent/llama-server"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("could not run"), "{err}");
    }
}
