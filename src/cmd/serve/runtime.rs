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

/// The daemon's runtime: [`resolve`]d at startup when that works, else
/// on the first load that needs it, so a failed llama.cpp fetch is a
/// startup warning rather than a dead daemon. A success is kept; a
/// failure is not, so the next load retries.
pub struct Lazy {
    requested: Runtime,
    llama_cpp_version: Option<String>,
    resolved: std::sync::Mutex<Option<Resolved>>,
    // One fetch at a time: loads of different models are not otherwise
    // serialized.
    resolving: tokio::sync::Mutex<()>,
}

impl Lazy {
    /// `resolved` is startup's result, `None` if that failed.
    pub fn new(
        requested: Runtime,
        llama_cpp_version: Option<String>,
        resolved: Option<Resolved>,
    ) -> Self {
        Lazy {
            requested,
            llama_cpp_version,
            resolved: std::sync::Mutex::new(resolved),
            resolving: tokio::sync::Mutex::new(()),
        }
    }

    /// What `--runtime` asked for (possibly `auto`).
    pub fn requested(&self) -> Runtime {
        self.requested
    }

    /// The resolution so far, without attempting one.
    pub fn known(&self) -> Option<Resolved> {
        self.resolved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set(&self, resolved: Resolved) {
        *self.resolved.lock().unwrap_or_else(|e| e.into_inner()) = Some(resolved);
    }

    /// The resolution, fetching llama.cpp now if startup could not.
    pub async fn resolve(&self) -> Result<Resolved> {
        if let Some(resolved) = self.known() {
            return Ok(resolved);
        }
        let _one_at_a_time = self.resolving.lock().await;
        if let Some(resolved) = self.known() {
            return Ok(resolved);
        }
        eprintln!(
            "[llmman] runtime {}: not set up at startup; fetching its llama.cpp now",
            self.requested.as_str()
        );
        let requested = self.requested;
        let pin = self.llama_cpp_version.clone();
        let resolved = tokio::task::spawn_blocking(move || resolve(requested, pin.as_deref()))
            .await
            .context("resolve runtime task panicked")?
            .context(
                "the llama.cpp runtime could not be set up (this daemon started without it)",
            )?;
        self.set(resolved.clone());
        Ok(resolved)
    }

    /// Whether engines run in containers. Answered without a fetch when
    /// `requested` has no container step (`bin`, `path`, `auto` off
    /// Linux), so a safetensors load never waits on a llama.cpp download
    /// it does not use.
    pub async fn ociman(&self) -> Result<Option<ContainerManager>> {
        if let Some(resolved) = self.known() {
            return Ok(resolved.ociman());
        }
        if !candidates(self.requested, false)
            .iter()
            .any(|c| c.ociman().is_some())
        {
            return Ok(None);
        }
        Ok(self.resolve().await?.ociman())
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
        let _one_at_a_time = self.resolving.lock().await;
        let (source, bin) = local(self.known().expect("resolved above"))?;
        if bin.exists() {
            return Ok(bin);
        }
        eprintln!(
            "[llmman] llama-server at {} no longer exists; re-resolving",
            bin.display()
        );
        let pin = self.llama_cpp_version.clone();
        let bin = tokio::task::spawn_blocking(move || resolve_local(source, pin.as_deref()))
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
    resolve_from(&candidates(runtime, false), runtime, llama_cpp_version)
}

/// [`resolve`] for callers that need a local binary (the mediagen backend
/// dlopens the libraries next to it): `auto` is `bin` then `path`, and a
/// container runtime is an error.
pub fn resolve_local(runtime: Runtime, llama_cpp_version: Option<&str>) -> Result<PathBuf> {
    if let Some(m) = runtime.ociman() {
        anyhow::bail!(
            "--runtime {} runs llama-server in a container; there is no local binary",
            m.binary()
        );
    }
    let resolved = resolve_from(&candidates(runtime, true), runtime, llama_cpp_version)?;
    Ok(resolved
        .llama_server_bin()
        .expect("bin and path resolve to a local binary")
        .clone())
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

fn resolve_from(
    candidates: &[Runtime],
    requested: Runtime,
    llama_cpp_version: Option<&str>,
) -> Result<Resolved> {
    let explicit = requested != Runtime::Auto;
    let mut failures = Vec::new();
    for &candidate in candidates {
        match try_one(candidate, llama_cpp_version, explicit) {
            Ok(resolved) => {
                eprintln!("[llmman] runtime {}: using {resolved}", requested.as_str());
                return Ok(resolved);
            }
            Err(e) if explicit => return Err(e),
            Err(e) => {
                eprintln!(
                    "[llmman] runtime auto: skipping {}: {e:#}",
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

/// One step of [`resolve`]. `explicit` skips [`ContainerManager::probe`]:
/// the user asked for that engine, and its own errors are clearer.
fn try_one(
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
                ociman.probe()?;
            }
            with_download_marker(|marker| {
                prepare_container(
                    marker,
                    || {
                        crate::container::pull_image(
                            ociman,
                            ContainerEngine::LlamaServer,
                            llama_cpp_version,
                        )
                    },
                    (!explicit).then_some(|| {
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
}
