//! `--runtime`: where `llmman serve` gets its inference engine. `docker`
//! and `podman` run it in a container (Linux only; `crate::container`),
//! `bin` is llmman's own download of llama.cpp's prebuilt `llama-server`
//! (`crate::llama_release`), `path` is whatever `llama-server` is on
//! `PATH`. `auto`, the default, tries them in that order and takes the
//! first that works. Safetensors models use vLLM's image under a
//! container runtime and the `vllm`/`mlx_lm.server` on `PATH` otherwise.
//!
//! `LLMMAN_RUNTIME` is the same setting as an environment variable, since
//! `llmman run`/`launch` start the daemon with a bare `llmman serve`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::ValueEnum;

use crate::container::{ContainerEngine, ContainerManager};

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
    /// Whatever `llama-server` is on `PATH`; nothing is ever downloaded.
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
            with_download_marker(|| {
                crate::container::pull_image(
                    ociman,
                    ContainerEngine::LlamaServer,
                    llama_cpp_version,
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

/// Runs `pull` (a `docker pull`, possibly minutes) holding
/// `crate::llama_release`'s download marker, touched every 15s, so
/// `daemon::ensure_server` leaves a daemon still pulling alive past its
/// startup budget, as it already does for the release download.
fn with_download_marker<T>(pull: impl FnOnce() -> T) -> T {
    let marker = crate::llama_release::DownloadMarker::create();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let out = std::thread::scope(|scope| {
        let marker = &marker;
        scope.spawn(move || loop {
            match done_rx.recv_timeout(std::time::Duration::from_secs(15)) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => marker.touch(),
                _ => return,
            }
        });
        let out = pull();
        drop(done_tx);
        out
    });
    drop(marker);
    out
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
}
