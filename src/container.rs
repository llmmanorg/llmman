//! Runs `llama-server` (or, for a safetensors model, `vllm serve`) inside
//! a container (Linux only, via `--runtime docker|podman`, and the first
//! thing the default `--runtime auto` tries) instead of as a local
//! process, auto-selecting the matching
//! `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for whatever GPU
//! acceleration the host actually has — see
//! <https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md> for
//! the full image list and their `docker run` flags, which
//! [`GpuBackend::engine_args`] mirrors for the subset detected here. The
//! same host probe picks the vLLM image (`vllm/vllm-openai`, `rocm/vllm`
//! or `vllm/vllm-openai-cpu`, per architecture) — see [`VllmBackend`] —
//! and SGLang's `lmsysorg/sglang` ([`VllmBackend::sglang_image_ref`]).
//!
//! This is the same problem ggml's own dynamic backend loading
//! (`GGML_BACKEND_DL=ON`, `ggml_backend_load_all` in
//! `ggml/src/ggml-backend-reg.cpp`) solves for shared libraries — given
//! several installed backend libraries, pick the best one for this
//! machine at runtime — except there's no shared library to load and
//! score here, just one container image to run, so detection below is a
//! fixed container-image priority order (CUDA > ROCm > Vulkan > CPU) rather than a
//! numeric score.
//!
//! Host GPU detection itself (the real CUDA Driver/HIP runtime/OpenCL/Vulkan API
//! probing) is entirely [`crate::hostgpu::detect`]'s job, shared with the
//! local (non-container) `llama-server` binary path in
//! `crate::llama_release` — this module only adds the mapping from that
//! one shared [`HostGpu`] result to *which container image and
//! `--device`/`--gpus` flags* to run, which `crate::hostgpu` has no
//! reason to know about.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::ValueEnum;

use crate::hostgpu::{self, HostGpu};

pub(crate) mod identity;

/// Container engine to run the picked image with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub enum ContainerManager {
    Docker,
    Podman,
}

impl ContainerManager {
    pub fn binary(self) -> &'static str {
        match self {
            ContainerManager::Docker => "docker",
            ContainerManager::Podman => "podman",
        }
    }

    /// Whether `--runtime auto` should try this engine: its CLI is on
    /// `PATH`, `<cli> info` answers within [`PROBE_TIMEOUT`] (a wedged
    /// daemon socket can hang it), and when [`GpuBackend::engine_args`]
    /// will pass `--gpus`, the NVIDIA Container Toolkit is present (or a
    /// CUDA backend can pass the driver through without it, see
    /// [`GpuBackend::container_args`]). `Err`
    /// says why not, for the log. An explicit `--runtime docker|podman`
    /// skips this and lets the engine report its own errors.
    pub fn probe(self) -> Result<()> {
        self.probe_with(true)
    }

    /// [`probe`](Self::probe) for a worker that runs the CPU image
    /// ([`SPEECH_BACKEND`]) and passes no GPU through: the toolkit is not
    /// required, so a host whose GPU probe skipped containers for
    /// llama-server can still run that worker in one.
    pub fn probe_cpu_only(self) -> Result<()> {
        self.probe_with(false)
    }

    fn probe_with(self, require_gpu_toolkit: bool) -> Result<()> {
        let cli = self.binary();
        if crate::find_on_path(cli).is_none() {
            anyhow::bail!("{cli} is not on PATH");
        }
        let mut cmd = std::process::Command::new(cli);
        cmd.arg("info")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        match run_with_timeout(cmd, PROBE_TIMEOUT).with_context(|| format!("run {cli} info"))? {
            Some(status) if status.success() => {}
            Some(_) => anyhow::bail!("{cli} info failed (is its daemon running?)"),
            None => anyhow::bail!("{cli} info did not answer within {PROBE_TIMEOUT:?}"),
        }
        if self == ContainerManager::Docker && cfg!(target_os = "linux") {
            // Settle the rootless question for `HardenedUser::resolve`
            // here, where `info` already answered, rather than on the
            // first load.
            let _ = identity::docker_is_rootless();
        }
        let backend = detect_backend();
        let wants_gpus = backend.engine_args().iter().any(|a| a == "--gpus");
        gpu_gate(
            require_gpu_toolkit,
            wants_gpus,
            cli,
            || nvidia_toolkit_present(self),
            || backend.is_cuda() && host_nvidia_direct_args().is_some(),
        )
    }
}

/// The toolkit rule of [`ContainerManager::probe`]: when the detected
/// backend wants `--gpus` and the run would need it, the NVIDIA Container
/// Toolkit must be present, or (`direct_feasible`) the CUDA backend can
/// pass the driver through without it ([`direct_nvidia_args`]). A
/// CPU-only run (`require_gpu_toolkit` false) passes whatever the host
/// has; `toolkit_present` and `direct_feasible` are only consulted when
/// the rule applies, the second only when the toolkit is absent.
fn gpu_gate(
    require_gpu_toolkit: bool,
    wants_gpus: bool,
    cli: &str,
    toolkit_present: impl FnOnce() -> bool,
    direct_feasible: impl FnOnce() -> bool,
) -> Result<()> {
    if require_gpu_toolkit && wants_gpus && !toolkit_present() && !direct_feasible() {
        anyhow::bail!(
            "an NVIDIA GPU was detected but no NVIDIA Container Toolkit for {cli} \
             (`{cli} run --gpus all` would fail)"
        );
    }
    Ok(())
}

/// How long [`ContainerManager::probe`] waits on `docker info`/`podman
/// info` before treating the engine as unusable.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Runs `cmd`, killing it after `timeout` (`Ok(None)`); `std::process`
/// has no deadline of its own.
fn run_with_timeout(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
) -> Result<Option<std::process::ExitStatus>> {
    wait_with_timeout(&mut cmd.spawn()?, timeout)
}

/// [`run_with_timeout`] for an already spawned `child`, whose pipes the
/// caller drains afterwards.
pub(crate) fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Result<Option<std::process::ExitStatus>> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// Heuristic for the NVIDIA Container Toolkit, in any shape it comes in:
/// its hook/CLI on `PATH`, a generated CDI spec, or (docker) an `nvidia`
/// runtime registered with the daemon.
fn nvidia_toolkit_present(ociman: ContainerManager) -> bool {
    if [
        "nvidia-container-runtime-hook",
        "nvidia-container-toolkit",
        "nvidia-ctk",
    ]
    .iter()
    .any(|b| crate::find_on_path(b).is_some())
    {
        return true;
    }
    if ["/etc/cdi/nvidia.yaml", "/var/run/cdi/nvidia.yaml"]
        .iter()
        .any(|p| Path::new(p).exists())
    {
        return true;
    }
    if ociman == ContainerManager::Docker {
        let runtimes = std::process::Command::new("docker")
            .args([
                "info",
                "--format",
                "{{range $k, $v := .Runtimes}}{{$k}} {{end}}",
            ])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        if runtimes.split_whitespace().any(|r| r == "nvidia") {
            return true;
        }
    }
    false
}

/// The Debian-style multiarch directory for `arch` (a
/// `std::env::consts::ARCH`), where the CUDA images look for driver
/// libraries.
fn multiarch_dir(arch: &str) -> Option<&'static str> {
    match arch {
        "x86_64" => Some("x86_64-linux-gnu"),
        "aarch64" => Some("aarch64-linux-gnu"),
        _ => None,
    }
}

/// The NVIDIA device nodes under `dev`, in the order they are passed:
/// the fixed control nodes that exist, then every `nvidia<N>` by number.
fn nvidia_device_nodes(dev: &Path) -> Vec<PathBuf> {
    let mut nodes: Vec<PathBuf> = [
        "nvidiactl",
        "nvidia-uvm",
        "nvidia-uvm-tools",
        "nvidia-modeset",
    ]
    .iter()
    .map(|n| dev.join(n))
    .filter(|p| p.exists())
    .collect();
    let mut gpus: Vec<(u32, PathBuf)> = std::fs::read_dir(dev)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let digits = name.to_str()?.strip_prefix("nvidia")?;
            if !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            Some((digits.parse().ok()?, e.path()))
        })
        .collect();
    gpus.sort();
    nodes.extend(gpus.into_iter().map(|(_, p)| p));
    nodes
}

/// Whether `name` is an NVIDIA user-space driver library:
/// `libcuda.so*` or `libnvidia-*.so*`, but not the toolkit's own
/// `libnvidia-container*`.
fn is_nvidia_driver_lib(name: &str) -> bool {
    name.starts_with("libcuda.so")
        || (name.starts_with("libnvidia-")
            && !name.starts_with("libnvidia-container")
            && name.contains(".so"))
}

/// Where the driver's libraries live on a host, in precedence order.
fn nvidia_lib_dirs(multiarch: &str) -> Vec<PathBuf> {
    [
        "/usr/lib64".to_string(),
        format!("/usr/lib/{multiarch}"),
        "/usr/lib".to_string(),
        "/lib64".to_string(),
        format!("/lib/{multiarch}"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect()
}

/// Every [`is_nvidia_driver_lib`] entry of `dirs`, in order, as its own
/// name and the real file it resolves to (a soname symlink such as
/// `libcuda.so.1` keeps its name but binds the file behind it). An entry
/// that does not resolve to a file is skipped.
fn scan_nvidia_libs(dirs: &[PathBuf]) -> Vec<(String, PathBuf)> {
    let mut libs = Vec::new();
    for dir in dirs {
        let mut found: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                if !is_nvidia_driver_lib(&name) {
                    return None;
                }
                let real = std::fs::canonicalize(e.path()).ok()?;
                real.is_file().then_some((name, real))
            })
            .collect();
        found.sort();
        libs.extend(found);
    }
    libs
}

/// The `run` arguments for NVIDIA passthrough without the container
/// toolkit: `--device` for each of `devices`, and a read-only bind mount
/// of each of `libs` (name, host file) into `/usr/lib/<multiarch>/`
/// under its own name, the first of a name winning. `None` unless the
/// driver is usable that way: `nvidiactl` among `devices` and
/// `libcuda.so.1` among `libs`. The image keeps its own CUDA runtime.
fn direct_nvidia_args(
    devices: &[PathBuf],
    libs: &[(String, PathBuf)],
    multiarch: &str,
) -> Option<Vec<String>> {
    let mut names = std::collections::HashSet::new();
    let libs: Vec<&(String, PathBuf)> = libs
        .iter()
        .filter(|(name, _)| is_nvidia_driver_lib(name) && names.insert(name.as_str()))
        .collect();
    let has_ctl = devices
        .iter()
        .any(|d| d.file_name().is_some_and(|n| n == "nvidiactl"));
    if !has_ctl || !names.contains("libcuda.so.1") {
        return None;
    }
    let mut args: Vec<String> = devices
        .iter()
        .flat_map(|d| ["--device".to_string(), d.display().to_string()])
        .collect();
    for (name, path) in libs {
        args.push("-v".into());
        args.push(format!("{}:/usr/lib/{multiarch}/{name}:ro", path.display()));
    }
    Some(args)
}

/// [`direct_nvidia_args`] for this host's `/dev` and driver libraries.
fn host_nvidia_direct_args() -> Option<Vec<String>> {
    let multiarch = multiarch_dir(std::env::consts::ARCH)?;
    direct_nvidia_args(
        &nvidia_device_nodes(Path::new("/dev")),
        &scan_nvidia_libs(&nvidia_lib_dirs(multiarch)),
        multiarch,
    )
}

/// Container GPU backends this module can run with a matching
/// `ghcr.io/ggml-org/llama.cpp:server-*` image for. Deliberately a subset
/// of every tag llama.cpp publishes (musa/intel/openvino are skipped): as
/// of writing, rocm/vulkan images are amd64-only upstream and cuda/vulkan
/// support arm64 too — see docs/docker.md for the authoritative list if
/// more get added here later. OpenCL is intentionally absent: llama.cpp's
/// OpenCL release is a Windows ARM64 Adreno binary, not a published Linux
/// container image, and this module is used only for Linux containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuBackend {
    Cpu,
    Cuda12,
    Cuda13,
    Rocm,
    Vulkan,
}

impl GpuBackend {
    /// The `server-<suffix>` part of the image tag; Cpu has no suffix.
    fn image_tag(self) -> &'static str {
        match self {
            GpuBackend::Cpu => "server",
            GpuBackend::Cuda12 => "server-cuda",
            GpuBackend::Cuda13 => "server-cuda13",
            GpuBackend::Rocm => "server-rocm",
            GpuBackend::Vulkan => "server-vulkan",
        }
    }

    /// The full `ghcr.io/ggml-org/llama.cpp:<tag>` reference. `version`,
    /// when given, pins to that release (e.g. `server-b9994` instead of
    /// the floating `server`) — ghcr.io/ggml-org/llama.cpp publishes a
    /// versioned tag alongside every floating one, built from the same
    /// release. Which pin is cmd::serve's decision (normally
    /// `crate::llama_release::default_release`, or `--llama-cpp-version`);
    /// `None` is the floating tag, for `--llama-cpp-version latest`.
    fn image_ref(self, version: Option<&str>) -> String {
        match version {
            Some(v) => format!("ghcr.io/ggml-org/llama.cpp:{}-{v}", self.image_tag()),
            None => format!("ghcr.io/ggml-org/llama.cpp:{}", self.image_tag()),
        }
    }

    /// Extra `docker run`/`podman run` arguments needed to see the host's
    /// GPU from inside the container, matching docs/docker.md's own
    /// examples for each backend (CUDA: "Docker With CUDA"; ROCm/Vulkan:
    /// the SYCL section's `--device /dev/dri` pattern, extended for ROCm
    /// with the `/dev/kfd` compute device and `video` group every ROCm
    /// container image's own documentation asks for).
    fn engine_args(self) -> Vec<String> {
        match self {
            GpuBackend::Cpu => vec![],
            GpuBackend::Cuda12 | GpuBackend::Cuda13 => vec!["--gpus".into(), "all".into()],
            GpuBackend::Rocm => vec![
                "--device".into(),
                "/dev/kfd".into(),
                "--device".into(),
                "/dev/dri".into(),
                "--group-add".into(),
                "video".into(),
            ],
            GpuBackend::Vulkan => {
                let mut args = Vec::new();
                if Path::new("/dev/dri").exists() {
                    args.extend(["--device", "/dev/dri"].map(String::from));
                }
                // NVIDIA's Vulkan ICD comes from the container toolkit
                if Path::new("/dev/nvidiactl").exists() {
                    args.extend(
                        ["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"].map(String::from),
                    );
                }
                args
            }
        }
    }

    fn is_cuda(self) -> bool {
        matches!(self, GpuBackend::Cuda12 | GpuBackend::Cuda13)
    }

    /// [`engine_args`](Self::engine_args) for a run on `ociman`: a CUDA
    /// backend on a host without the NVIDIA Container Toolkit passes the
    /// driver through directly ([`host_nvidia_direct_args`]) when it can,
    /// where `--gpus all` would fail.
    fn container_args(self, ociman: ContainerManager) -> Vec<String> {
        self.container_args_with(|| nvidia_toolkit_present(ociman), host_nvidia_direct_args)
    }

    fn container_args_with(
        self,
        toolkit_present: impl FnOnce() -> bool,
        direct_args: impl FnOnce() -> Option<Vec<String>>,
    ) -> Vec<String> {
        if self.is_cuda() && !toolkit_present() {
            if let Some(args) = direct_args() {
                return args;
            }
        }
        self.engine_args()
    }
}

/// Detects the best available GPU backend by delegating to
/// [`crate::hostgpu::detect`] (real CUDA Driver/HIP/OpenCL/Vulkan API
/// probing — see that module) and mapping its result onto which
/// `ghcr.io/ggml-org/llama.cpp` image to run. This doesn't verify the
/// container engine itself is configured to pass a GPU through: for an
/// explicit `--runtime docker|podman`, `docker run --gpus all` surfaces
/// that misconfiguration directly and clearly enough on its own; only
/// `--runtime auto`, which has to decide whether to try the engine at
/// all, adds a heuristic ([`ContainerManager::probe`]).
pub(crate) fn detect_backend() -> GpuBackend {
    backend_from_hostgpu(hostgpu::detect())
}

/// The image the speech worker runs in, whatever the host has: it
/// computes on the CPU, and a GPU image would ask the engine for devices
/// it may not pass through. The spawn arm and `--pull-only` both read it.
pub(crate) const SPEECH_BACKEND: GpuBackend = GpuBackend::Cpu;

/// Pure mapping from [`HostGpu`] to the available container [`GpuBackend`], split out from
/// [`detect_backend`] so the CUDA 12-vs-13 image split (llama.cpp's own
/// CUDA Dockerfile split between the `cuda`/`cuda12` tag, built against
/// CUDA_VERSION 12.8.1, and `cuda13`, 13.4.1 — see docs/docker.md) can be
/// tested directly without needing real GPU hardware. `HostGpu::Metal`
/// has no container image (Docker/Podman GPU passthrough isn't a macOS
/// concept, and container runtimes are rejected on non-Linux before this is ever
/// called — see `cmd::serve::serve_async`). `HostGpu::Opencl` also falls back
/// to CPU here because there is no OpenCL container image; it remains a real
/// backend in [`crate::llama_release`] for the local Windows ARM64 binary.
fn backend_from_hostgpu(gpu: HostGpu) -> GpuBackend {
    match gpu {
        HostGpu::Cuda { major } if major >= 13 => GpuBackend::Cuda13,
        HostGpu::Cuda { .. } => GpuBackend::Cuda12,
        HostGpu::Rocm => GpuBackend::Rocm,
        HostGpu::Vulkan => GpuBackend::Vulkan,
        HostGpu::Metal | HostGpu::None | HostGpu::Opencl => GpuBackend::Cpu,
    }
}

/// The architectures vLLM publishes images for. Its Docker Hub tags spell
/// the architecture out (unlike llama.cpp's multi-arch manifests), so the
/// host architecture is part of the image choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostArch {
    X86_64,
    Aarch64,
}

impl HostArch {
    /// From `std::env::consts::ARCH`; `None` for anything without an image.
    fn parse(arch: &str) -> Option<HostArch> {
        match arch {
            "x86_64" => Some(HostArch::X86_64),
            "aarch64" => Some(HostArch::Aarch64),
            _ => None,
        }
    }

    fn detect() -> Result<HostArch> {
        HostArch::parse(std::env::consts::ARCH).with_context(|| {
            format!(
                "vLLM publishes no container image for {} hosts (only x86_64 and aarch64)",
                std::env::consts::ARCH
            )
        })
    }
}

/// The vLLM image family for a host — derived from [`GpuBackend`] so both
/// engines share the one [`crate::hostgpu::detect`] probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmBackend {
    /// `vllm/vllm-openai`'s `-cu129` tags, for a CUDA 12 driver.
    Cuda12,
    /// `vllm/vllm-openai`'s default (CUDA 13) tags.
    Cuda13,
    /// `rocm/vllm`: amd64 only upstream.
    Rocm,
    /// `vllm/vllm-openai-cpu`: also what a Vulkan-only host gets, since
    /// vLLM has no Vulkan backend.
    Cpu,
}

impl VllmBackend {
    fn from_gpu(backend: GpuBackend) -> VllmBackend {
        match backend {
            GpuBackend::Cuda12 => VllmBackend::Cuda12,
            GpuBackend::Cuda13 => VllmBackend::Cuda13,
            GpuBackend::Rocm => VllmBackend::Rocm,
            GpuBackend::Cpu | GpuBackend::Vulkan => VllmBackend::Cpu,
        }
    }

    /// This host's backend, from the shared GPU probe.
    fn detect() -> VllmBackend {
        VllmBackend::from_gpu(detect_backend())
    }

    /// The image for a safetensors `engine`. `vllm_image` (`--vllm-image`)
    /// replaces [`ContainerEngine::Vllm`]'s, verbatim, without needing a
    /// supported architecture; no other engine's.
    fn engine_image(
        self,
        engine: ContainerEngine,
        version: Option<&str>,
        vllm_image: Option<&str>,
    ) -> Result<String> {
        if let (ContainerEngine::Vllm, Some(image)) = (engine, vllm_image) {
            return Ok(image.to_string());
        }
        let arch = HostArch::detect()?;
        match engine {
            ContainerEngine::Vllm => self.image_ref(arch, version),
            ContainerEngine::VllmOmni => self.omni_image_ref(arch, version),
            ContainerEngine::Sglang => self.sglang_image_ref(arch, version),
            ContainerEngine::LlamaServer | ContainerEngine::Speech => {
                anyhow::bail!("llama-server and speech containers go through spawn")
            }
        }
    }

    /// The `docker.io/`-qualified image (so podman doesn't prompt for a
    /// registry). `version` replaces the floating `latest`: `<version>-<arch>`
    /// (`-cu129` for CUDA 12) for the `vllm/` images, the whole tag for
    /// `rocm/vllm` (whose `rocmX.Y_..._vllm_Z` tags carry no arch suffix).
    /// Arch spellings are Docker Hub's: the GPU image says `aarch64`, the
    /// CPU image `arm64`.
    fn image_ref(self, arch: HostArch, version: Option<&str>) -> Result<String> {
        let v = version.unwrap_or("latest");
        let (repo, suffix) = match (self, arch) {
            (VllmBackend::Rocm, HostArch::X86_64) => return Ok(format!("docker.io/rocm/vllm:{v}")),
            (VllmBackend::Rocm, HostArch::Aarch64) => anyhow::bail!(
                "an AMD GPU was detected, but rocm/vllm publishes no aarch64 image \
                 (set LLMMAN_LLM_LIBRARY=cpu to run vllm/vllm-openai-cpu instead)"
            ),
            (VllmBackend::Cuda12, HostArch::X86_64) => ("vllm-openai", "x86_64-cu129"),
            (VllmBackend::Cuda12, HostArch::Aarch64) => ("vllm-openai", "aarch64-cu129"),
            (VllmBackend::Cuda13, HostArch::X86_64) => ("vllm-openai", "x86_64"),
            (VllmBackend::Cuda13, HostArch::Aarch64) => ("vllm-openai", "aarch64"),
            (VllmBackend::Cpu, HostArch::X86_64) => ("vllm-openai-cpu", "x86_64"),
            (VllmBackend::Cpu, HostArch::Aarch64) => ("vllm-openai-cpu", "arm64"),
        };
        Ok(format!("docker.io/vllm/{repo}:{v}-{suffix}"))
    }

    /// The `vllm/vllm-omni` image for a Diffusers-layout model: CUDA only,
    /// `<version>-{x86_64,aarch64}`; `version` is vLLM-Omni's release.
    fn omni_image_ref(self, arch: HostArch, version: Option<&str>) -> Result<String> {
        let v = version.unwrap_or("latest");
        let suffix = match (self, arch) {
            (VllmBackend::Cuda12 | VllmBackend::Cuda13, HostArch::X86_64) => "x86_64",
            (VllmBackend::Cuda12 | VllmBackend::Cuda13, HostArch::Aarch64) => "aarch64",
            (VllmBackend::Rocm, _) => anyhow::bail!(
                "an AMD GPU was detected, but vllm/vllm-omni publishes only CUDA images \
                 (install vllm-omni locally and use --runtime path to serve this model)"
            ),
            (VllmBackend::Cpu, _) => anyhow::bail!(
                "no CUDA GPU was detected, and vllm/vllm-omni publishes only CUDA images"
            ),
        };
        Ok(format!("docker.io/vllm/vllm-omni:{v}-{suffix}"))
    }

    /// The `lmsysorg/sglang` image (multi-arch, no arch suffix): `<v>`
    /// for CUDA 13 (`latest` floating), `<v>-cu129` for CUDA 12 (retired
    /// upstream after v0.5.19). ROCm tags name a ROCm release and a GPU
    /// family (`v0.5.19-rocm700-mi30x`) with no floating alias, so
    /// `version` must be the whole tag, as for `rocm/vllm`. The CPU
    /// (`-xeon`) images need Intel AMX and flags llmman does not pass.
    fn sglang_image_ref(self, arch: HostArch, version: Option<&str>) -> Result<String> {
        let v = version.unwrap_or("latest");
        let tag = match (self, arch) {
            (VllmBackend::Cuda13, _) => v.to_string(),
            (VllmBackend::Cuda12, _) => format!("{v}-cu129"),
            (VllmBackend::Rocm, HostArch::X86_64) => version.map(str::to_string).context(
                "an AMD GPU was detected, but lmsysorg/sglang publishes no floating ROCm \
                     tag: pass the whole tag for your GPU family as --sglang-version \
                     (e.g. v0.5.19-rocm700-mi30x; see hub.docker.com/r/lmsysorg/sglang/tags)",
            )?,
            (VllmBackend::Rocm, HostArch::Aarch64) => anyhow::bail!(
                "an AMD GPU was detected, but lmsysorg/sglang publishes no aarch64 ROCm image"
            ),
            (VllmBackend::Cpu, _) => anyhow::bail!(
                "no CUDA or ROCm GPU was detected, and lmsysorg/sglang's CPU images (`-xeon`) \
                 need Intel AMX and launch flags llmman does not pass (install sglang \
                 locally and drop --ociman, or unset LLMMAN_SAFETENSORS_ENGINE to use vllm)"
            ),
        };
        Ok(format!("docker.io/lmsysorg/sglang:{tag}"))
    }

    /// GPU passthrough plus what vLLM's deployment docs ask for: `--ipc=host`
    /// (its workers share tensors over `/dev/shm`) and, for the CPU image,
    /// `SYS_NICE` and an unconfined seccomp profile for NUMA thread binding.
    fn engine_args(self) -> Vec<String> {
        let mut args = match self {
            VllmBackend::Cuda12 | VllmBackend::Cuda13 => GpuBackend::Cuda12.engine_args(),
            VllmBackend::Rocm => GpuBackend::Rocm.engine_args(),
            VllmBackend::Cpu => vec![
                "--security-opt".into(),
                "seccomp=unconfined".into(),
                "--cap-add".into(),
                "SYS_NICE".into(),
            ],
        };
        args.push("--ipc=host".into());
        args
    }
}

/// Which engine's image a container run is for; cmd::serve knows this from
/// the model's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerEngine {
    /// `ghcr.io/ggml-org/llama.cpp` (GGUF and diffusion models).
    LlamaServer,
    /// `vllm/vllm-openai`, `rocm/vllm` or `vllm/vllm-openai-cpu` (safetensors).
    Vllm,
    /// `vllm/vllm-omni` (Diffusers-layout safetensors: `vllm serve --omni`).
    VllmOmni,
    /// `lmsysorg/sglang` (safetensors, `LLMMAN_SAFETENSORS_ENGINE=sglang`).
    Sglang,
    /// `ghcr.io/ggml-org/llama.cpp`'s CPU image, running this binary's
    /// speech worker over a bind mount ([`spawn_mediagen`] with
    /// [`SPEECH_BACKEND`]) for a Canary GGUF.
    Speech,
}

/// The image [`spawn`] / [`spawn_engine`] would run here for `engine`.
fn image_for(
    engine: ContainerEngine,
    version: Option<&str>,
    vllm_image: Option<&str>,
) -> Result<String> {
    match engine {
        ContainerEngine::LlamaServer => Ok(detect_backend().image_ref(version)),
        ContainerEngine::Speech => Ok(SPEECH_BACKEND.image_ref(version)),
        _ => VllmBackend::detect().engine_image(engine, version, vllm_image),
    }
}

/// Runs `llama-server` inside a container: `docker run --rm --init -t`
/// (or `podman run` with the same flags), auto-selecting the image for
/// whatever [`detect_backend`] found. Returns the running child process
/// (the attached `docker`/`podman` CLI itself, not the container), with
/// its stdout/stderr piped for the caller to relay and tail.
///
/// This runs *attached* (no `-d`) specifically so it can be managed like
/// a normal child process: `--init` runs a real init (tini) as the
/// container's PID 1, so SIGTERM forwarded to it (e.g. via `docker stop`,
/// or the CLI's own signal forwarding while attached) is actually
/// delivered with default disposition and terminates the container
/// promptly — a bare `sleep`/`llama-server` running *as* PID 1 (no
/// `--init`) does not get default signal handling at all, a well-known
/// Linux PID-1 gotcha, and was verified live to leave the container
/// running indefinitely after being sent SIGTERM. `--rm` then cleans up
/// the stopped container automatically. `-t` allocates a pseudo-tty so
/// the containerized process's own output behaves like a normal
/// interactive process (typically line-buffered) instead of block image
/// buffered as it would through a plain pipe — deliberately *not* paired
/// with `-i`: `-i` needs an actual open, readable stdin to attach, which
/// fails ("cannot attach stdin to a TTY-enabled container because stdin
/// is not a terminal") when combined with `-t` and this process's own
/// stdin isn't a real terminal — the common case, since `llmman serve`
/// itself is normally daemonized with stdin closed (see daemon.rs).
///
/// Pulls the image [`spawn`] or [`spawn_engine`] would run for `engine` on
/// this host, with the pull's own progress output (a real `docker pull`/
/// `podman pull` progress bar — not something llmman re-implements)
/// inherited directly to this process's stdout/stderr. `version` is the
/// engine's pin (`--llama-cpp-version` / `--vllm-version`), if any, and
/// `vllm_image` is `--vllm-image`.
///
/// `spawn`'s underlying `docker run`/`podman run` would pull an image that
/// isn't already cached locally on its own, but silently and without any
/// visible progress from the caller's perspective (its own stdio is
/// redirected to a log file when started detached — see daemon.rs and
/// cmd::serve). So `cmd::serve` calls this itself before serving (and
/// `--pull-only` stops right after it), rather than relying on `spawn`'s
/// own implicit pull; under `--runtime auto` a failed pull is also what
/// moves it on to the next engine.
pub fn pull_image(
    ociman: ContainerManager,
    engine: ContainerEngine,
    version: Option<&str>,
    vllm_image: Option<&str>,
) -> Result<()> {
    use std::io::Write;

    let image = image_for(engine, version, vllm_image)?;
    let cli = ociman.binary();
    let header = format!("[llmman] {cli}: pulling {image}...");
    eprintln!("{header}");
    let (success, reason) = match pull_progress_sink() {
        Some((mut sink, cols)) => {
            let _ = write!(sink, "{header}\r\n");
            let run = run_on_pty(cli, &["pull", &image], cols, &mut sink)?;
            (run.success, run.last_line)
        }
        None => {
            let status = std::process::Command::new(cli)
                .args(["pull", &image])
                .status()
                .with_context(|| format!("run {cli} pull {image}"))?;
            (status.success(), None)
        }
    };
    if !success {
        anyhow::bail!(
            "{cli} pull {image} failed{}",
            reason.map(|l| format!(": {l}")).unwrap_or_default()
        );
    }
    Ok(())
}

/// The download marker's status while [`pull_image`] runs; the client
/// keys on it to show the pull's output.
pub const PULL_STATUS: &str = "pulling the llama.cpp container image";

/// The download marker's status while [`verify_llama_server_runs`] starts
/// the pulled image (up to `VERIFY_TIMEOUT`).
pub const VERIFY_STATUS: &str = "checking that the llama.cpp container starts";

/// Set by `daemon::ensure_server` on the daemon it spawns for a terminal:
/// a file the client replays to its terminal, and that terminal's width.
/// The daemon's stdio is a log file, where `docker pull`/`podman pull`
/// print no progress, so [`pull_image`] runs them on a pty of that width
/// and appends what they draw to the file.
pub const PULL_PROGRESS_FILE_ENV: &str = "LLMMAN_PULL_PROGRESS_FILE";
pub const PULL_PROGRESS_COLS_ENV: &str = "LLMMAN_PULL_PROGRESS_COLS";

/// The file and width from those variables. `None` (pull on inherited
/// stdio) if unset or the file is gone: the client removes it when it
/// stops listening, so a later pull, like a lazy retry, goes to the log.
fn pull_progress_sink() -> Option<(std::fs::File, u16)> {
    let path = std::env::var_os(PULL_PROGRESS_FILE_ENV)?;
    let cols = std::env::var(PULL_PROGRESS_COLS_ENV)
        .ok()?
        .parse::<u16>()
        .ok()
        .filter(|c| *c > 0)?;
    // Append-only across pulls, so the client's offset stays valid.
    let file = std::fs::OpenOptions::new().append(true).open(path).ok()?;
    Some((file, cols))
}

struct PtyRun {
    success: bool,
    /// See [`last_output_line`].
    last_line: Option<String>,
}

/// Runs `program` on a pty of `cols` columns, writing what it draws to
/// `sink`. Sink errors are ignored: the pty must still be drained or the
/// program would block.
fn run_on_pty(
    program: &str,
    args: &[&str],
    cols: u16,
    sink: &mut dyn std::io::Write,
) -> Result<PtyRun> {
    use portable_pty::{native_pty_system, CommandBuilder, PtySize};
    use std::io::Read;

    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 24,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("open a pty")?;
    let mut cmd = CommandBuilder::new(program);
    cmd.args(args);
    let mut child = pair
        .slave
        .spawn_command(cmd)
        .with_context(|| format!("run {program}"))?;
    // Held open, the slave would keep the reader from seeing EOF.
    drop(pair.slave);
    let mut reader = pair
        .master
        .try_clone_reader()
        .context("read from the pty")?;
    const TAIL: usize = 4096;
    let mut tail = Vec::new();
    let mut buf = [0u8; 8192];
    // Ends when the program exits: Linux reports the closed slave as an
    // error after handing over what was buffered.
    while let Ok(n @ 1..) = reader.read(&mut buf) {
        let _ = sink.write_all(&buf[..n]);
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > TAIL {
            tail.drain(..tail.len() - TAIL);
        }
    }
    let status = child.wait().with_context(|| format!("wait on {program}"))?;
    // The master stays open until here: closing it hangs the program up.
    drop(pair.master);
    Ok(PtyRun {
        success: status.success(),
        last_line: last_output_line(&tail),
    })
}

/// The last non-empty line of terminal output, control sequences removed:
/// a failed pull's reason, for the error (and so the daemon's log).
fn last_output_line(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut plain = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            // CSI: parameter bytes, then one final byte in `@`..=`~`.
            '\x1b' => {
                if chars.next() == Some('[') {
                    for n in chars.by_ref() {
                        if ('@'..='~').contains(&n) {
                            break;
                        }
                    }
                }
            }
            '\r' | '\n' => plain.push('\n'),
            c if c.is_control() => {}
            c => plain.push(c),
        }
    }
    plain
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .map(str::to_owned)
}

/// How long [`verify_llama_server_runs`] waits for its container.
const VERIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Runs `llama-server --version` in the llama.cpp image [`spawn`] would
/// run, with the same GPU passthrough: `<cli> info` answering does not
/// mean the engine can start a container (rootless podman without a
/// reachable systemd fails every `run` in crun), and `--runtime auto`
/// must not settle on an engine every model load would then fail on.
/// The engine's own error goes to stderr, like [`pull_image`]'s progress.
pub(crate) fn verify_llama_server_runs(
    ociman: ContainerManager,
    backend: GpuBackend,
    version: Option<&str>,
    hardened: bool,
) -> Result<()> {
    let cli = ociman.binary();
    let name = format!(
        "llmman-verify-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    );
    // The verify run mounts nothing, so no owner to read as: the
    // daemon's own ids, as a model load of a store it owns would get.
    let user = hardened
        .then(|| HardenedUser::resolve(ociman, None))
        .flatten();
    let engine_args = backend.container_args(ociman);
    let mut cmd = std::process::Command::new(cli);
    cmd.args(verify_args(
        backend,
        engine_args.clone(),
        version,
        &name,
        hardened,
        user.as_ref(),
    ))
    .stdin(std::process::Stdio::null())
    .stdout(std::process::Stdio::null());
    let suffix = if hardened {
        format!(
            "; {}",
            hardened_note(
                &hardened_flags(&engine_args, &llama_tmpfs(&engine_args), user.as_ref()),
                None,
            )
        )
    } else {
        String::new()
    };
    match run_with_timeout(cmd, VERIFY_TIMEOUT).with_context(|| format!("run {cli} run"))? {
        Some(status) if status.success() => Ok(()),
        Some(status) => {
            anyhow::bail!("{cli} could not start a llama.cpp container ({status}){suffix}")
        }
        None => {
            // Killing the attached CLI leaves its container running; --rm
            // only acts once the container exits.
            let mut rm = std::process::Command::new(cli);
            rm.args(["rm", "-f", &name])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            let _ = run_with_timeout(rm, PROBE_TIMEOUT);
            anyhow::bail!(
                "{cli} did not start a llama.cpp container within {VERIFY_TIMEOUT:?}{suffix}"
            )
        }
    }
}

/// The GPU passthrough (`engine_args`, from [`GpuBackend::container_args`]),
/// GPU-visibility environment and `hardened` flags match what [`spawn`]
/// serves with.
fn verify_args(
    backend: GpuBackend,
    engine_args: Vec<String>,
    version: Option<&str>,
    name: &str,
    hardened: bool,
    user: Option<&HardenedUser>,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "run".into(),
        "--rm".into(),
        "--init".into(),
        "--name".into(),
        name.into(),
    ];
    if hardened {
        args.extend(hardened_flags(
            &engine_args,
            &llama_tmpfs(&engine_args),
            user,
        ));
    }
    args.extend(engine_args);
    args.extend(forwarded_env_args(&[]));
    args.push(backend.image_ref(version));
    args.push("--version".into());
    args
}

/// Every `llama-server` knob `cmd::serve` resolves once per load and
/// forwards identically to both a local child
/// (`cmd::serve::backend::spawn_llama_server`) and a containerized one ([`spawn`]).
/// One struct rather than seven repeated positional parameters, so
/// adding a flag is a single edit that can't reach only one backend.
#[derive(Debug, Clone, Copy)]
pub struct LlamaOptions<'a> {
    /// The loopback port llama-server binds, published out of the
    /// container as `127.0.0.1:<port>:<port>`.
    pub port: u16,

    /// `--ctx-size`. `None` leaves it unset, falling back to the model's
    /// own `n_ctx_train` — see `cmd::serve::config::context_length_from_env`'s
    /// doc comment for what this does and doesn't guarantee.
    pub ctx_size: Option<u32>,

    /// `--flash-attn <mode>`. `None` falls back to llama-server's own
    /// `auto` — see `cmd::serve::config::flash_attention_from_env`.
    pub flash_attention: Option<&'a str>,

    /// `--cache-type-k`/`--cache-type-v <type>` (both set together).
    /// `None` falls back to llama-server's own `f16` — see
    /// `cmd::serve::config::kv_cache_type_from_env`.
    pub kv_cache_type: Option<&'a str>,

    /// `--context-shift` when true, `--no-context-shift` when false —
    /// always passed explicitly, never left to the default. See
    /// `cmd::serve::config::supports_context_shift`.
    pub context_shift: bool,

    /// `--split-mode <mode>`. `None` falls back to llama-server's own
    /// `layer` — see `cmd::serve::config::sched_spread_from_env`.
    pub split_mode: Option<&'a str>,

    /// `--parallel <n>`. `None` falls back to llama-server's own `auto`
    /// (four slots) — see `cmd::serve::config::effective_num_parallel`.
    pub num_parallel: Option<u32>,

    /// `--embeddings`: an embedding model (see
    /// `cmd::serve::config::embedding_model_ctx`) gets 501 on `/v1/embeddings`
    /// without it.
    pub embeddings: bool,

    /// `-b`/`-ub`, for an embedding or decision model (see
    /// `cmd::serve::config::whole_prompt_batch`): llama-server needs a whole
    /// input in one ubatch. Per-slot, not the `num_parallel`-scaled `ctx_size`.
    pub batch_size: Option<u32>,

    /// `--threads <n>`: a request's Ollama `options.num_thread`, else the
    /// derived default, half the CPUs above 8 (see `cmd::serve::ensure_model`'s
    /// `request_threads` and `cmd::serve::config::threads_from_env_or_host`).
    /// Forwarded into the container too, since `cpus` gives it the same
    /// CPU budget the value was derived from. An explicit
    /// LLAMA_ARG_THREADS leaves this `None` and reaches the container via
    /// LLAMA_CPP_ENV_PASSTHROUGH_VARS instead.
    pub threads: Option<u32>,

    /// `--cpus <n>` on the container: the daemon's own CPU limit, `Some`
    /// only when one binds (see `cmd::serve::config::container_cpu_limit`).
    /// Snapshotted at startup together with `threads`, so the two
    /// always agree.
    pub cpus: Option<f64>,

    /// [`hardened_args`] on the container (no effect on a local child);
    /// on unless `--no-hardened`.
    pub hardened: bool,

    /// The split `LLMMAN_LLAMA_ARGS`, after every flag above.
    pub extra_args: &'a [String],
}

/// Callers must stop this gracefully (SIGTERM, not the default
/// `Child::kill()`/`kill_on_drop`, which sends SIGKILL) — see
/// `cmd::serve::ModelProcess`'s Drop impl. SIGKILL cannot be caught or
/// forwarded by the CLI process at all (that's what SIGKILL means), so
/// it was also verified live to leave the container running.
///
/// `llama_cpp_version`, when given, pins the image to that release tag
/// (see [`GpuBackend::image_ref`]) instead of the floating one.
///
/// `opts` carries every `llama-server` flag this forwards into the
/// container — see [`LlamaOptions`].
///
/// `mmproj_path`, when given, forwards `--mmproj <path>`, mounted as its
/// own `/mmproj` read-only volume (it's extracted into a different cache
/// directory than `model_path`, so it can't share that mount).
pub fn spawn(
    ociman: ContainerManager,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    llama_cpp_version: Option<&str>,
    opts: LlamaOptions<'_>,
) -> Result<tokio::process::Child> {
    let backend = detect_backend();
    let image = backend.image_ref(llama_cpp_version);
    eprintln!(
        "[llmman] {}: detected {:?}, using image {:?}",
        ociman.binary(),
        backend,
        image
    );
    let (model_dir, model_file) = mount_split(model_path, "model")?;
    let mmproj = mmproj_path.map(|p| mount_split(p, "mmproj")).transpose()?;
    let user = opts
        .hardened
        .then(|| HardenedUser::resolve(ociman, Some(Path::new(&model_dir))))
        .flatten();
    let args = llama_run_args(
        backend.container_args(ociman),
        image,
        &model_dir,
        &model_file,
        mmproj.as_ref().map(|(d, f)| (d.as_str(), f.as_str())),
        crate::cmd::serve::LLAMA_CPP_ENV_PASSTHROUGH_VARS,
        opts,
        user.as_ref(),
    );
    run(ociman, args)
}

/// A file's bind-mount source directory (see [`mount_source`]) and its
/// UTF-8 file name, for `-v <dir>:/x:ro` plus `/x/<file>`.
fn mount_split(path: &Path, what: &str) -> Result<(String, String)> {
    let dir = path
        .parent()
        .with_context(|| format!("{what} path has no parent directory"))?;
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{what} path has no valid UTF-8 filename"))?;
    Ok((mount_source(dir)?, file.to_owned()))
}

/// [`spawn`]'s argv, split out so the flags are testable without an
/// engine. `mmproj` is the mmproj's `(mount dir, file name)`, mounted
/// as its own `/mmproj` volume. `user` is the [`HardenedUser`] the
/// caller resolved for the store, if any. `engine_args` is the backend's
/// [`GpuBackend::container_args`].
#[allow(clippy::too_many_arguments)]
fn llama_run_args(
    engine_args: Vec<String>,
    image: String,
    model_dir: &str,
    model_file: &str,
    mmproj: Option<(&str, &str)>,
    passthrough_vars: &[&str],
    opts: LlamaOptions<'_>,
    user: Option<&HardenedUser>,
) -> Vec<String> {
    let LlamaOptions {
        port,
        ctx_size,
        flash_attention,
        kv_cache_type,
        context_shift,
        split_mode,
        num_parallel,
        embeddings,
        batch_size,
        threads,
        cpus,
        hardened,
        extra_args,
    } = opts;
    let tmpfs = llama_tmpfs(&engine_args);
    let mut args = run_args(
        engine_args,
        port,
        passthrough_vars,
        cpus,
        hardened,
        user,
        &tmpfs,
    );
    args.push("-v".into());
    args.push(format!("{model_dir}:/models:ro"));
    if let Some((dir, _)) = mmproj {
        args.push("-v".into());
        args.push(format!("{dir}:/mmproj:ro"));
    }
    args.push(image);
    args.extend([
        "-m".into(),
        format!("/models/{model_file}"),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "0.0.0.0".into(),
    ]);
    if let Some((_, file)) = mmproj {
        args.push("--mmproj".into());
        args.push(format!("/mmproj/{file}"));
    }
    if let Some(n) = ctx_size {
        args.push("--ctx-size".into());
        args.push(n.to_string());
    }
    if let Some(mode) = flash_attention {
        args.push("--flash-attn".into());
        args.push(mode.to_string());
    }
    if let Some(t) = kv_cache_type {
        args.push("--cache-type-k".into());
        args.push(t.to_string());
        args.push("--cache-type-v".into());
        args.push(t.to_string());
    }
    args.push(
        if context_shift {
            "--context-shift"
        } else {
            "--no-context-shift"
        }
        .into(),
    );
    if let Some(mode) = split_mode {
        args.push("--split-mode".into());
        args.push(mode.to_string());
    }
    if let Some(n) = num_parallel {
        args.push("--parallel".into());
        args.push(n.to_string());
    }
    if embeddings {
        args.push("--embeddings".into());
    }
    if let Some(n) = batch_size {
        args.push("-b".into());
        args.push(n.to_string());
        args.push("-ub".into());
        args.push(n.to_string());
    }
    // Pairs with `--cpus` (see run_args): a quota alone leaves
    // autodetection starting a thread per host core.
    if let Some(n) = threads {
        args.push("--threads".into());
        args.push(n.to_string());
    }
    args.extend(extra_args.iter().cloned());
    args
}

/// The `run` prefix every container here starts with: attached with an
/// init, the port published, the daemon's CPU limit forwarded as
/// `--cpus` (`cpus`), the GPU passed through (`engine_args`), and the
/// GPU-visibility env vars plus the engine's own `passthrough_vars`
/// forwarded (`docker run` does not inherit the environment). `-e NAME`
/// without a value: the engine copies it from our environment, so a
/// secret (`VLLM_API_KEY`) never appears in argv or the debug log.
///
/// `cpus`: the container is a sibling of the daemon in its own cgroup
/// (under dockerd or the rootless user slice), so a quota on `llmman
/// serve` never reaches it on its own, and all of the CPU work would
/// run unlimited behind a capped proxy (llmmanorg/llmman#324). `None`
/// (no limit binds) adds nothing. It is a CFS quota even when the
/// daemon's limit is an affinity mask: the budget is what matters, and
/// it is the one form both engines share.
///
/// `hardened`: [`hardened_args`] with `tmpfs`, between the prefix and
/// `engine_args` so an engine's own `--cap-add`/`--security-opt` lands
/// after the drop, then `user`'s [`HardenedUser::args`] (its
/// `--group-add`s only when `engine_args` passes a device). `false` adds
/// nothing, `user` included.
fn run_args(
    engine_args: Vec<String>,
    port: u16,
    passthrough_vars: &[&str],
    cpus: Option<f64>,
    hardened: bool,
    user: Option<&HardenedUser>,
    tmpfs: &[&str],
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "run".into(),
        "--rm".into(),
        "--init".into(),
        "-t".into(),
        "-p".into(),
        format!("127.0.0.1:{port}:{port}"),
    ];
    if let Some(n) = cpus {
        args.push("--cpus".into());
        args.push(n.to_string());
    }
    if hardened {
        args.extend(hardened_flags(&engine_args, tmpfs, user));
    }
    args.extend(engine_args);
    args.extend(forwarded_env_args(passthrough_vars));
    args
}

/// The writable paths a hardened `llama-server` container needs on the
/// CPU; [`llama_tmpfs`] adds the driver caches when a GPU is passed.
const LLAMA_TMPFS: &[&str] = &["/tmp"];

/// The GPU drivers' shader cache root inside the image, as root: Mesa
/// RADV writes `~/.cache/mesa_shader_cache` and MIOpen `~/.cache/miopen`.
const GPU_CACHE_TMPFS: &str = "/root/.cache";

/// [`LLAMA_TMPFS`] plus [`GPU_CACHE_TMPFS`] when `engine_args` passes a
/// GPU ([`passes_devices`]), so the Vulkan and ROCm shader caches work
/// for the run instead of failing on the read-only rootfs. A numeric
/// `--user` ([`HardenedUser`]) needs nothing more: its
/// [`HARDENED_USER_ENV`] already points `XDG_CACHE_HOME` under the `/tmp`
/// tmpfs, and the extra mount is harmless.
fn llama_tmpfs(engine_args: &[String]) -> Vec<&'static str> {
    let mut tmpfs = LLAMA_TMPFS.to_vec();
    if passes_devices(engine_args) {
        tmpfs.push(GPU_CACHE_TMPFS);
    }
    tmpfs
}

/// The `--tmpfs` value for a forwarded `VLLM_RPC_BASE_PATH` (`value`),
/// the directory vLLM's multiprocessing mode puts its IPC sockets in:
/// under hardening the rootfs is read-only, so the path must be writable
/// or every request would fail. `None` when the variable is unset or
/// already under one of `tmpfs_roots` ([`ENGINE_TMPFS`]); `Some("<value>:exec")`
/// for another absolute path; `Err` for a relative or empty value, which
/// vLLM would resolve against its working directory on the read-only
/// rootfs, for one containing `:` or `,`, which `--tmpfs` would read as
/// mount options, and for one with a `..` segment, which could resolve
/// under a `tmpfs_roots` entry after the check here said it did not.
pub(crate) fn vllm_rpc_tmpfs(
    value: Option<&str>,
    tmpfs_roots: &[&str],
) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.starts_with('/') {
        return Err(format!(
            "VLLM_RPC_BASE_PATH must be an absolute path under hardening, got {value:?}; \
             unset it or pass --no-hardened"
        ));
    }
    // `--tmpfs` splits the path from its options at ':' (podman at every
    // ':') and the options at ',', so either character would be read as
    // options or truncate the mount.
    if value.contains([':', ',']) || value.split('/').any(|s| s == "..") {
        return Err(format!(
            "VLLM_RPC_BASE_PATH must not contain ':', ',' or a '..' segment under hardening, \
             got {value:?}; unset it or pass --no-hardened"
        ));
    }
    let mounted = tmpfs_roots.iter().any(|root| {
        value == *root
            || value
                .strip_prefix(root)
                .is_some_and(|rest| rest.starts_with('/'))
    });
    Ok((!mounted).then(|| format!("{value}:exec")))
}

/// [`vllm_rpc_tmpfs`] for [`spawn_engine`]: `VLLM_RPC_BASE_PATH` (read
/// by `rpc_base`) only reaches a container whose forwarded prefix
/// (`env_prefix`, from [`ContainerEngine::launcher`]) is `VLLM_`, so an
/// SGLang container neither needs the tmpfs nor can fail on the value;
/// `None` too when not `hardened`, where the rootfs is writable.
fn engine_rpc_tmpfs(
    env_prefix: &str,
    hardened: bool,
    rpc_base: impl FnOnce() -> Option<String>,
) -> Result<Option<String>, String> {
    if !hardened || env_prefix != "VLLM_" {
        return Ok(None);
    }
    vllm_rpc_tmpfs(rpc_base().as_deref(), ENGINE_TMPFS)
}

/// The writable paths a hardened vLLM or SGLang container needs: the
/// vLLM cache root and RPC sockets (`/root/.cache`, `/tmp`), Triton's
/// kernel cache (`/root/.triton`) and vLLM's usage-stats file
/// (`/root/.config`). The last two go beyond the task's `/tmp` and
/// `/root/.cache`: Triton and vLLM's usage reporter write them on a
/// read-only rootfs and would abort the first request otherwise.
const ENGINE_TMPFS: &[&str] = &["/tmp", "/root/.cache", "/root/.triton", "/root/.config"];

/// Whether `engine_args` passes a GPU through with `--gpus` (the NVIDIA
/// toolkit path).
fn passes_gpus(engine_args: &[String]) -> bool {
    engine_args.iter().any(|a| a == "--gpus")
}

/// Whether `engine_args` passes any GPU through: `--gpus` or a
/// `--device` node (ROCm's `/dev/kfd`, Vulkan's `/dev/dri`). Wider than
/// [`passes_gpus`], which gates only the `--read-only` skip (the crun
/// hook bug is NVIDIA's).
fn passes_devices(engine_args: &[String]) -> bool {
    engine_args.iter().any(|a| a == "--gpus" || a == "--device")
}

/// The uid a hardened container reads the store as, when the image's
/// root cannot. `--cap-drop all` removes CAP_DAC_OVERRIDE, so the
/// container's root reads a `0700` store directory or `0600` blob (umask
/// 077) only when it *is* the owner. The rule, per
/// [`identity::Identity`]:
///
/// - rootful docker on Linux: the daemon's root is the host's, so
///   `--user <uid>:<gid>` of the mounted store's owner (the daemon's own
///   ids when it is not readable, or nothing is mounted) makes the
///   container the owner;
/// - rootless docker on Linux: container uid 0 already maps to the host
///   user, so `--user 0:0` keeps it there (a host uid would be remapped
///   to a subordinate id and own nothing), and no `--group-add`: a host
///   gid would be remapped the same way, so `device_gids` is left empty;
/// - rootless podman (`euid != 0`): root in the container is the invoking
///   user by default, so no `--user`;
/// - rootful podman (`euid == 0`): there is no daemon, llmman itself is
///   root, so the store it wrote is root's and container root (host
///   root) owns it; a store another user owns (`LLMMAN_MODELS` pointing
///   at one) is left to the `chmod` in [`hardened_note`] rather than
///   mapped, since nothing here says which of its users should run;
/// - Docker Desktop and non-unix hosts: no `--user`.
///
/// A numeric `--user` has no `/etc/passwd` entry in the image, so docker
/// gives it `HOME=/` and no supplementary groups: `--group-add` with the
/// numeric host gid of each GPU node (the image's group *names* resolve
/// against its own `/etc/group`, so ROCm's `--group-add video` may name
/// another gid) keeps the devices readable, and `HOME` and the XDG and
/// Triton cache roots move under the `/tmp` tmpfs so vLLM, Triton, Mesa
/// and MIOpen do not write to the read-only root. `/dev/nvidia*` nodes
/// are out of scope: the NVIDIA toolkit injects them behind `--gpus`,
/// world-readable (`0666`) as the driver creates them, so a `0660` node
/// a site chmod'ed is not covered by [`device_gids`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HardenedUser {
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    /// Host gids of the existing GPU device nodes, sorted and deduplicated.
    pub(crate) device_gids: Vec<u32>,
}

impl HardenedUser {
    /// The rule above for the host, with `mounted` the store or cache
    /// directory the container will read (`None` mounts nothing).
    pub(crate) fn resolve(ociman: ContainerManager, mounted: Option<&Path>) -> Option<Self> {
        let linux = cfg!(target_os = "linux");
        let rootless =
            ociman == ContainerManager::Docker && linux && identity::docker_is_rootless();
        let identity =
            identity::Identity::from_parts(Some(ociman), linux, rootless, identity::user_ids());
        Self::from_parts(
            &identity,
            rootless,
            mounted.and_then(path_owner),
            device_gids(),
        )
    }

    /// The pure mapping behind [`HardenedUser::resolve`]: `identity` as
    /// [`identity::Identity::from_parts`] gave it, `owner` the mounted
    /// directory's `(uid, gid)` when readable, `device_gids` from
    /// [`device_gids`]. `rootless` pins the docker `0:0` case, where the
    /// host owner must not override it and the device gids are dropped.
    pub(crate) fn from_parts(
        identity: &identity::Identity,
        rootless: bool,
        owner: Option<(u32, u32)>,
        device_gids: Vec<u32>,
    ) -> Option<Self> {
        match *identity {
            identity::Identity::User { uid, gid } => {
                let (uid, gid, device_gids) = if rootless {
                    (uid, gid, Vec::new())
                } else {
                    let (uid, gid) = owner.unwrap_or((uid, gid));
                    (uid, gid, device_gids)
                };
                Some(Self {
                    uid,
                    gid,
                    device_gids,
                })
            }
            identity::Identity::Image | identity::Identity::KeepId => None,
        }
    }

    /// The `run` flags: `--user`, a `--group-add` per device gid when
    /// `devices` are passed, and the cache roots under `/tmp`.
    fn args(&self, devices: bool) -> Vec<String> {
        let mut args = vec![
            "--user".to_string(),
            identity::Identity::user_arg(self.uid, self.gid),
        ];
        if devices {
            args.extend(
                self.device_gids
                    .iter()
                    .flat_map(|gid| ["--group-add".to_string(), gid.to_string()]),
            );
        }
        args.extend(
            HARDENED_USER_ENV
                .iter()
                .flat_map(|pair| ["-e".to_string(), pair.to_string()]),
        );
        args
    }
}

/// The environment a numeric `--user` gets so every cache lands on the
/// `/tmp` tmpfs: `HOME` (docker's default for an unknown uid is `/`),
/// vLLM's and Mesa's cache root, vLLM's and MIOpen's config root and
/// Triton's kernel cache.
const HARDENED_USER_ENV: &[&str] = &[
    "HOME=/tmp",
    "XDG_CACHE_HOME=/tmp/.cache",
    "XDG_CONFIG_HOME=/tmp/.config",
    "TRITON_CACHE_DIR=/tmp/.triton",
];

/// `(uid, gid)` owning `path`, when readable.
#[cfg(unix)]
fn path_owner(path: &Path) -> Option<(u32, u32)> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.uid(), meta.gid()))
}

#[cfg(not(unix))]
fn path_owner(_path: &Path) -> Option<(u32, u32)> {
    None
}

/// The host gid of each existing GPU device node a backend passes:
/// `/dev/kfd` (ROCm) and `/dev/dri/renderD*`, `/dev/dri/card*` (ROCm,
/// Vulkan); sorted and deduplicated.
#[cfg(unix)]
fn device_gids() -> Vec<u32> {
    use std::os::unix::fs::MetadataExt;
    let mut nodes = vec![std::path::PathBuf::from("/dev/kfd")];
    if let Ok(dir) = std::fs::read_dir("/dev/dri") {
        nodes.extend(dir.flatten().map(|e| e.path()).filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("renderD") || n.starts_with("card"))
        }));
    }
    let mut gids: Vec<u32> = nodes
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.gid())
        .collect();
    gids.sort_unstable();
    gids.dedup();
    gids
}

#[cfg(not(unix))]
fn device_gids() -> Vec<u32> {
    Vec::new()
}

/// The hardened block of a `run` argv: [`hardened_args`] for
/// `engine_args` (read-only unless they pass `--gpus`) and `tmpfs`, then
/// `user`'s flags ([`HardenedUser::args`], with the device groups when
/// `engine_args` pass a device).
fn hardened_flags(
    engine_args: &[String],
    tmpfs: &[&str],
    user: Option<&HardenedUser>,
) -> Vec<String> {
    let mut args = hardened_args(!passes_gpus(engine_args), tmpfs);
    if let Some(user) = user {
        args.extend(user.args(passes_devices(engine_args)));
    }
    args
}

/// The suffix a hardened container's failure carries, so one that exits
/// only because of the flags (a store written with umask 077 is
/// unreadable once `--cap-drop all` removes CAP_DAC_OVERRIDE from the
/// container's root) names them and the fixes: `flags` the hardened
/// part of the argv the container was run with ([`hardened_flags`], plus
/// the [`vllm_rpc_tmpfs`] mount when one was added), `mounted` the store
/// or cache directory it read (`None` for a run mounting nothing),
/// loosened with `chmod -R go+rX` when the engine cannot run as its owner.
pub(crate) fn hardened_note(flags: &[String], mounted: Option<&str>) -> String {
    let fix = match mounted {
        Some(path) => format!("chmod -R go+rX {path} lets the container read it as any user, and "),
        None => String::new(),
    };
    format!(
        "started hardened ({}); {fix}--no-hardened turns these flags off",
        flags.join(" ")
    )
}

/// Which hardened container a [`spawn_hardened_note`] describes: vLLM
/// ([`spawn_engine`] with a `VLLM_` prefix) mounts [`ENGINE_TMPFS`] plus
/// the [`vllm_rpc_tmpfs`] and follows the host backend's `--gpus`;
/// SGLang the same without the RPC tmpfs; a llama.cpp container
/// ([`spawn`], or [`spawn_mediagen`] with the detected backend) mounts
/// [`LLAMA_TMPFS`] and follows the same probe; the speech worker
/// ([`spawn_mediagen`] with [`SPEECH_BACKEND`]) never passes a GPU, so
/// it is read-only on every host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HardenedSpawn {
    LlamaServer,
    Speech,
    Vllm,
    Sglang,
}

impl HardenedSpawn {
    /// [`hardened_note`] with the flags [`run_args`] applied for this
    /// container on a host whose probe returned `host`, run as `user`
    /// ([`HardenedUser::resolve`]), with `rpc_tmpfs` the
    /// [`vllm_rpc_tmpfs`] value (used by [`HardenedSpawn::Vllm`] only),
    /// reading `mounted`. `llama_args` yields the llama.cpp container's
    /// [`GpuBackend::container_args`] (used by [`HardenedSpawn::LlamaServer`]
    /// only).
    fn note(
        self,
        host: GpuBackend,
        llama_args: impl FnOnce() -> Vec<String>,
        user: Option<&HardenedUser>,
        rpc_tmpfs: Option<&str>,
        mounted: &str,
    ) -> String {
        let mut flags = match self {
            Self::Vllm | Self::Sglang => hardened_flags(
                &VllmBackend::from_gpu(host).engine_args(),
                ENGINE_TMPFS,
                user,
            ),
            Self::LlamaServer => {
                let engine_args = llama_args();
                hardened_flags(&engine_args, &llama_tmpfs(&engine_args), user)
            }
            Self::Speech => {
                let engine_args = SPEECH_BACKEND.engine_args();
                hardened_flags(&engine_args, &llama_tmpfs(&engine_args), user)
            }
        };
        if let (Self::Vllm, Some(tmpfs)) = (self, rpc_tmpfs) {
            flags.extend(["--tmpfs".to_string(), tmpfs.to_string()]);
        }
        hardened_note(&flags, Some(mounted))
    }
}

/// [`hardened_note`] for a container `ociman` started on this host,
/// naming the flags `spawn` was run with (`--read-only` follows the
/// backend the container was given at spawn time, not only the host
/// probe; the `--user` block is resolved again for `mounted`, the
/// directory the spawn mounted and resolved its own `--user` against;
/// the RPC tmpfs follows `VLLM_RPC_BASE_PATH` as [`spawn_engine`] read
/// it).
pub(crate) fn spawn_hardened_note(
    spawn: HardenedSpawn,
    ociman: ContainerManager,
    mounted: &Path,
) -> String {
    let user = HardenedUser::resolve(ociman, Some(mounted));
    let rpc_tmpfs = vllm_rpc_tmpfs(
        std::env::var("VLLM_RPC_BASE_PATH").ok().as_deref(),
        ENGINE_TMPFS,
    )
    .ok()
    .flatten();
    let host = detect_backend();
    spawn.note(
        host,
        || host.container_args(ociman),
        user.as_ref(),
        rpc_tmpfs.as_deref(),
        &mounted.display().to_string(),
    )
}

/// The hardened `run` flags: every capability dropped, no privilege
/// gain through setuid binaries, and (`read_only`) a read-only rootfs
/// with a tmpfs on each of `tmpfs`. Each tmpfs gets `exec`: docker and
/// podman both mount `--tmpfs` `noexec` by default, and vLLM's
/// inductor and Triton load compiled `.so` files out of their caches.
///
/// `read_only` is off for a container that passes `--gpus`: crun before
/// 1.26 applies the read-only remount before the NVIDIA toolkit's
/// createContainer hooks write `/etc` and `/usr/lib64`
/// (NVIDIA/nvidia-container-toolkit#2131, containers/crun#1924).
pub(crate) fn hardened_args(read_only: bool, tmpfs: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--cap-drop".into(),
        "all".into(),
        "--security-opt".into(),
        "no-new-privileges".into(),
    ];
    if read_only {
        args.push("--read-only".into());
    }
    args.extend(
        tmpfs
            .iter()
            .flat_map(|path| ["--tmpfs".to_string(), format!("{path}:exec")]),
    );
    args
}

/// `-e NAME` for each GPU-visibility variable and each of
/// `passthrough_vars` set in our environment.
fn forwarded_env_args(passthrough_vars: &[&str]) -> Vec<String> {
    crate::cmd::serve::GPU_VISIBLE_DEVICE_VARS
        .iter()
        .chain(passthrough_vars)
        .filter(|var| std::env::var_os(var).is_some())
        .flat_map(|var| ["-e".to_string(), var.to_string()])
        .collect()
}

impl ContainerEngine {
    /// The safetensors engines' image, launcher (`--entrypoint`, then
    /// argv ahead of the serve flags) and the env-var prefix forwarded
    /// into the container; `None` for llama-server (see [`spawn`]).
    /// `--entrypoint` is explicit because `rocm/vllm`'s default is a
    /// shell, `vllm/vllm-omni`'s empty and `lmsysorg/sglang`'s a shell;
    /// `python3 -m sglang.launch_server` is SGLang's own Docker recipe.
    fn launcher(self) -> Option<(&'static [&'static str], &'static str)> {
        match self {
            ContainerEngine::LlamaServer | ContainerEngine::Speech => None,
            ContainerEngine::Vllm | ContainerEngine::VllmOmni => Some((&["vllm"], "VLLM_")),
            ContainerEngine::Sglang => {
                Some((&["python3", "-m", "sglang.launch_server"], "SGLANG_"))
            }
        }
    }
}

/// The safetensors counterpart of [`spawn`]: `engine` (`Vllm`, `VllmOmni`
/// or `Sglang`) serving a directory mounted at `/models`, in the image
/// [`VllmBackend`] picks for it. `serve_args` builds the argv after the
/// launcher from the in-container model dir and bind address. The
/// engine's own env vars (`VLLM_*` / `SGLANG_*`) are forwarded, as a local
/// child would inherit them, plus the Hub token variables. `vllm_image`
/// is `--vllm-image`; the GPU flags still follow the host probe.
#[allow(clippy::too_many_arguments)]
pub fn spawn_engine(
    ociman: ContainerManager,
    engine: ContainerEngine,
    model_dir: &Path,
    version: Option<&str>,
    vllm_image: Option<&str>,
    port: u16,
    cpus: Option<f64>,
    hardened: bool,
    serve_args: impl FnOnce(&str, &str) -> Vec<String>,
) -> Result<tokio::process::Child> {
    let backend = VllmBackend::detect();
    let image = backend.engine_image(engine, version, vllm_image)?;
    let Some((launcher, env_prefix)) = engine.launcher() else {
        unreachable!("LlamaServer and Speech bailed above");
    };
    eprintln!(
        "[llmman] {}: detected {:?}, using image {:?}",
        ociman.binary(),
        backend,
        image
    );
    let model_dir = mount_source(model_dir)?;
    let vars: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| {
            k.starts_with(env_prefix) || matches!(k.as_str(), "HF_TOKEN" | "HUGGING_FACE_HUB_TOKEN")
        })
        .collect();
    let vars: Vec<&str> = vars.iter().map(String::as_str).collect();
    let rpc_tmpfs = engine_rpc_tmpfs(env_prefix, hardened, || {
        std::env::var("VLLM_RPC_BASE_PATH").ok()
    })
    .map_err(anyhow::Error::msg)?;
    let user = hardened
        .then(|| HardenedUser::resolve(ociman, Some(Path::new(&model_dir))))
        .flatten();
    let prefix = run_args(
        backend.engine_args(),
        port,
        &vars,
        cpus,
        hardened,
        user.as_ref(),
        ENGINE_TMPFS,
    );
    run(
        ociman,
        engine_run_args(
            prefix,
            rpc_tmpfs.as_deref(),
            image,
            launcher,
            &model_dir,
            serve_args,
        ),
    )
}

/// A bind-mount source: absolute (a relative `-v` source is a named
/// volume to the engine, and `LLMMAN_MODELS` may be relative) and UTF-8.
fn mount_source(dir: &Path) -> Result<String> {
    let dir = dir
        .canonicalize()
        .with_context(|| format!("resolving {}", dir.display()))?;
    dir.to_str()
        .map(str::to_owned)
        .with_context(|| format!("{} is not valid UTF-8", dir.display()))
}

/// [`spawn_engine`]'s argv after [`run_args`]'s `run` prefix, split out
/// so the ordering (`rpc_tmpfs` from [`vllm_rpc_tmpfs`], then
/// `-v`/`--entrypoint` before the image, the launcher's rest and serve
/// flags after) is testable without an engine.
fn engine_run_args(
    mut args: Vec<String>,
    rpc_tmpfs: Option<&str>,
    image: String,
    launcher: &[&str],
    model_dir: &str,
    serve_args: impl FnOnce(&str, &str) -> Vec<String>,
) -> Vec<String> {
    if let Some(tmpfs) = rpc_tmpfs {
        args.extend(["--tmpfs".to_string(), tmpfs.to_string()]);
    }
    args.extend([
        "-v".into(),
        format!("{model_dir}:/models:ro"),
        "--entrypoint".into(),
        launcher[0].into(),
        image,
    ]);
    args.extend(launcher[1..].iter().map(|a| a.to_string()));
    args.extend(serve_args("/models", "0.0.0.0"));
    args
}

fn run(ociman: ContainerManager, args: Vec<String>) -> Result<tokio::process::Child> {
    crate::debug_log!("spawning {} {}", ociman.binary(), args.join(" "));
    // Piped, not inherited, so cmd::serve can tail the container's
    // startup output (an OOM abort, a missing library, ...) exactly as it
    // does a local llama-server's. With `-t` the CLI merges the
    // container's stdout+stderr onto its own stdout, CRLF-terminated.
    tokio::process::Command::new(ociman.binary())
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {} {}", ociman.binary(), args.join(" ")))
}

/// Runs this binary as `model_ref`'s backend (`llmman serve MODEL --port
/// PORT`: the media backend of a diffusion model, the speech worker of a
/// Canary GGUF) inside the image [`spawn`] would use: it carries the
/// ggml/llama libraries (in `/app`, next to `llama-server`), and for CUDA
/// the runtime too. The binary, store and cache are bind-mounted read-only
/// at their host paths. The image's glibc must be at least the host's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_mediagen(
    ociman: ContainerManager,
    backend: GpuBackend,
    model_ref: &str,
    store_path: &Path,
    cache_path: &Path,
    llama_cpp_version: Option<&str>,
    port: u16,
    cpus: Option<f64>,
    hardened: bool,
) -> Result<tokio::process::Child> {
    let image = backend.image_ref(llama_cpp_version);
    eprintln!(
        "[llmman] {}: {:?} backend, using image {:?} for {model_ref}'s worker",
        ociman.binary(),
        backend,
        image
    );
    let exe = std::env::current_exe().context("locating the llmman binary")?;
    let utf8 = |p: &Path| -> Result<String> {
        p.to_str()
            .map(str::to_owned)
            .with_context(|| format!("{} is not valid UTF-8", p.display()))
    };
    let (exe, store, cache) = (utf8(&exe)?, utf8(store_path)?, utf8(cache_path)?);
    let user = hardened
        .then(|| HardenedUser::resolve(ociman, Some(store_path)))
        .flatten();
    let args = mediagen_run_args(
        backend.container_args(ociman),
        image,
        &exe,
        &store,
        &cache,
        model_ref,
        port,
        cpus,
        hardened,
        user.as_ref(),
    );
    run(ociman, args)
}

/// [`spawn_mediagen`]'s full argv, split out so the worker's mounts, env
/// and hardened block are testable without an engine.
#[allow(clippy::too_many_arguments)]
fn mediagen_run_args(
    engine_args: Vec<String>,
    image: String,
    exe: &str,
    store: &str,
    cache: &str,
    model_ref: &str,
    port: u16,
    cpus: Option<f64>,
    hardened: bool,
    user: Option<&HardenedUser>,
) -> Vec<String> {
    let passthrough = [
        crate::cmd::serve::LLAMA_CPP_ENV_PASSTHROUGH_VARS,
        crate::cmd::serve::MEDIAGEN_ENV_PASSTHROUGH_VARS,
    ]
    .concat();
    let tmpfs = llama_tmpfs(&engine_args);
    let mut args = run_args(
        engine_args,
        port,
        &passthrough,
        cpus,
        hardened,
        user,
        &tmpfs,
    );
    args.extend([
        "-v".into(),
        format!("{exe}:/usr/local/bin/llmman:ro"),
        "-v".into(),
        format!("{store}:{store}:ro"),
        "-v".into(),
        format!("{cache}:{cache}:ro"),
        "-e".into(),
        format!("LLMMAN_MODELS={store}"),
        "-e".into(),
        "PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        "--entrypoint".into(),
        "/usr/local/bin/llmman".into(),
        image,
        "serve".into(),
        model_ref.into(),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "0.0.0.0".into(),
        // The image's /app/llama-server, not a download inside it.
        "--runtime".into(),
        "path".into(),
    ]);
    args
}

/// Gracefully stops a container started by [`spawn`] by sending SIGTERM to
/// the attached `docker`/`podman` CLI process — see `spawn`'s doc comment
/// for why this must be SIGTERM (forwarded to the container's `--init`
/// PID 1) and not the default forceful kill. Best-effort: called from a
/// synchronous `Drop` impl (see `ModelProcess` in cmd::serve), so errors
/// are only logged, never propagated. Unix only (matching the container
/// runtimes themselves, which cmd::serve::serve_async already rejects on other
/// platforms) — `libc::kill` is not meaningful on Windows.
#[cfg(unix)]
pub fn stop(pid: u32) {
    // SAFETY: kill(2) with an existing pid and a valid signal number is
    // always safe to call; a stale/already-reaped pid just returns ESRCH,
    // which is not a memory-safety concern.
    let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        eprintln!("[llmman] warning: SIGTERM to container process {pid} failed: {err}");
    }
}

/// Unreachable in practice (container runtimes are rejected on non-Linux before
/// `spawn` is ever called — see cmd::serve::serve_async), but this needs
/// to compile on every platform llmman ships for, and a plain forceful
/// kill here is at least no worse than the SIGKILL callers were already
/// relying on before this module existed.
#[cfg(not(unix))]
pub fn stop(_pid: u32) {
    eprintln!("[llmman] warning: container::stop is a no-op on non-Unix platforms");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_output_line_drops_control_sequences_and_redraws() {
        let out = b"Pulling\r\n\x1b[1A\x1b[2K\rabc: Downloading  1MB/3MB\r\x1b[1B\x1b[1A\x1b[2K\rError response from daemon: denied \r\n\r\n";
        assert_eq!(
            last_output_line(out).as_deref(),
            Some("Error response from daemon: denied")
        );
        assert_eq!(last_output_line(b"\x1b[2K\r\n"), None);
        assert_eq!(last_output_line(b""), None);
    }

    /// A pty is a terminal to the program on it, of the width asked for,
    /// and everything it prints reaches the sink, carriage returns and all.
    #[cfg(unix)]
    #[test]
    fn run_on_pty_gives_the_program_a_terminal_and_relays_its_output() {
        let mut sink = Vec::new();
        let run = run_on_pty(
            "sh",
            &[
                "-c",
                "test -t 1 && stty size; printf 'half\\rdone\\n'; echo 'boom: nope' >&2; exit 3",
            ],
            97,
            &mut sink,
        )
        .unwrap();
        let text = String::from_utf8_lossy(&sink);
        assert!(text.contains("24 97"), "got: {text:?}");
        assert!(text.contains("half\rdone"), "got: {text:?}");
        assert!(!run.success);
        assert_eq!(run.last_line.as_deref(), Some("boom: nope"));
    }

    #[cfg(unix)]
    #[test]
    fn run_on_pty_reports_success() {
        let mut sink = Vec::new();
        let run = run_on_pty("sh", &["-c", "echo ok"], 80, &mut sink).unwrap();
        assert!(run.success);
        assert_eq!(run.last_line.as_deref(), Some("ok"));
    }

    #[test]
    fn run_on_pty_reports_a_missing_program() {
        let err = run_on_pty("llmman-no-such-program", &[], 80, &mut Vec::new());
        assert!(err.is_err());
    }

    #[test]
    fn cuda_major_12_picks_cuda12_image() {
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 12 }),
            GpuBackend::Cuda12
        );
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
    }

    #[test]
    fn cuda_major_13_picks_cuda13_image() {
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 13 }),
            GpuBackend::Cuda13
        );
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
    }

    #[test]
    fn cuda_major_above_13_still_picks_cuda13_image() {
        // No cuda14+ image exists yet -- a future driver reporting a
        // higher major version should still bucket into the newer of the
        // two published images rather than falling back to cuda12.
        assert_eq!(
            backend_from_hostgpu(HostGpu::Cuda { major: 14 }),
            GpuBackend::Cuda13
        );
    }

    #[test]
    fn non_cuda_hostgpu_variants_map_to_their_matching_backend() {
        assert_eq!(backend_from_hostgpu(HostGpu::Rocm), GpuBackend::Rocm);
        assert_eq!(backend_from_hostgpu(HostGpu::Opencl), GpuBackend::Cpu);
        assert_eq!(backend_from_hostgpu(HostGpu::Vulkan), GpuBackend::Vulkan);
        assert_eq!(backend_from_hostgpu(HostGpu::None), GpuBackend::Cpu);
        assert_eq!(backend_from_hostgpu(HostGpu::Metal), GpuBackend::Cpu);
    }

    #[test]
    fn image_tags_match_docs_docker_md() {
        assert_eq!(GpuBackend::Cpu.image_tag(), "server");
        assert_eq!(GpuBackend::Cuda12.image_tag(), "server-cuda");
        assert_eq!(GpuBackend::Cuda13.image_tag(), "server-cuda13");
        assert_eq!(GpuBackend::Rocm.image_tag(), "server-rocm");
        assert_eq!(GpuBackend::Vulkan.image_tag(), "server-vulkan");
    }

    /// A GPU host without the toolkit refuses the engine for llama-server
    /// and accepts it for the CPU-only worker; without a GPU, or with the
    /// toolkit, both pass and the toolkit is not even looked for.
    #[test]
    fn the_cpu_only_probe_passes_where_the_toolkit_alone_refuses() {
        let err = gpu_gate(true, true, "docker", || false, || false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("NVIDIA Container Toolkit"), "{err}");
        gpu_gate(false, true, "docker", || false, || false).unwrap();
        gpu_gate(true, true, "docker", || true, || false).unwrap();
        gpu_gate(
            true,
            false,
            "docker",
            || unreachable!("no GPU, no toolkit check"),
            || unreachable!("no GPU, no direct check"),
        )
        .unwrap();
        gpu_gate(
            false,
            false,
            "docker",
            || unreachable!("no GPU, no toolkit check"),
            || unreachable!("no GPU, no direct check"),
        )
        .unwrap();
    }

    #[test]
    fn the_gate_passes_without_the_toolkit_when_direct_passthrough_is_feasible() {
        gpu_gate(true, true, "docker", || false, || true).unwrap();
        let err = gpu_gate(true, true, "docker", || false, || false)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "an NVIDIA GPU was detected but no NVIDIA Container Toolkit for docker \
             (`docker run --gpus all` would fail)"
        );
        gpu_gate(
            true,
            true,
            "docker",
            || true,
            || unreachable!("the toolkit is present: direct passthrough is not consulted"),
        )
        .unwrap();
        gpu_gate(
            false,
            true,
            "docker",
            || unreachable!("a CPU-only run needs no toolkit"),
            || unreachable!("a CPU-only run needs no passthrough"),
        )
        .unwrap();
    }

    fn pb(paths: &[&str]) -> Vec<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    fn lib(name: &str, path: &str) -> (String, PathBuf) {
        (name.to_string(), PathBuf::from(path))
    }

    #[test]
    fn direct_nvidia_args_pass_devices_then_read_only_driver_libs() {
        let args = direct_nvidia_args(
            &pb(&["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"]),
            &[
                lib("libcuda.so.1", "/usr/lib/libcuda.so.580.1"),
                lib("libnvidia-ml.so.1", "/usr/lib/libnvidia-ml.so.580.1"),
            ],
            "x86_64-linux-gnu",
        )
        .unwrap();
        assert_eq!(
            args,
            [
                "--device",
                "/dev/nvidiactl",
                "--device",
                "/dev/nvidia-uvm",
                "--device",
                "/dev/nvidia0",
                "-v",
                "/usr/lib/libcuda.so.580.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro",
                "-v",
                "/usr/lib/libnvidia-ml.so.580.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro",
            ]
        );
        assert!(!passes_gpus(&args));
        assert!(passes_devices(&args));
    }

    #[test]
    fn direct_nvidia_args_use_the_multiarch_dir_of_the_host() {
        for (arch, multiarch) in [
            ("x86_64", "x86_64-linux-gnu"),
            ("aarch64", "aarch64-linux-gnu"),
        ] {
            assert_eq!(multiarch_dir(arch), Some(multiarch));
            let args = direct_nvidia_args(
                &pb(&["/dev/nvidiactl"]),
                &[lib("libcuda.so.1", "/usr/lib64/libcuda.so.1")],
                multiarch,
            )
            .unwrap();
            assert_eq!(
                args.last().unwrap(),
                &format!("/usr/lib64/libcuda.so.1:/usr/lib/{multiarch}/libcuda.so.1:ro")
            );
        }
        assert_eq!(multiarch_dir("riscv64"), None);
    }

    #[test]
    fn direct_nvidia_args_skip_the_toolkit_libs_and_dedup_by_name() {
        let args = direct_nvidia_args(
            &pb(&["/dev/nvidiactl"]),
            &[
                lib("libcuda.so.1", "/usr/lib64/libcuda.so.580.1"),
                lib(
                    "libnvidia-container.so.1",
                    "/usr/lib64/libnvidia-container.so.1",
                ),
                lib(
                    "libnvidia-container-go.so.1",
                    "/usr/lib64/libnvidia-container-go.so.1",
                ),
                lib("libcuda.so.1", "/usr/lib/libcuda.so.1"),
                lib("libnvidia-ml.so.1", "/usr/lib/libnvidia-ml.so.1"),
            ],
            "x86_64-linux-gnu",
        )
        .unwrap();
        let mounts: Vec<&String> = args.iter().filter(|a| a.contains(":/usr/lib/")).collect();
        assert_eq!(
            mounts,
            [
                "/usr/lib64/libcuda.so.580.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro",
                "/usr/lib/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro",
            ]
        );
    }

    #[test]
    fn direct_nvidia_args_need_nvidiactl_and_libcuda() {
        let libcuda = [lib("libcuda.so.1", "/usr/lib/libcuda.so.1")];
        let m = "x86_64-linux-gnu";
        assert!(direct_nvidia_args(&pb(&["/dev/nvidia0"]), &libcuda, m).is_none());
        assert!(direct_nvidia_args(&[], &libcuda, m).is_none());
        assert!(direct_nvidia_args(&pb(&["/dev/nvidiactl"]), &[], m).is_none());
        let only_dev_symlink = [lib("libcuda.so", "/usr/lib/libcuda.so")];
        assert!(direct_nvidia_args(&pb(&["/dev/nvidiactl"]), &only_dev_symlink, m).is_none());
        let only_container = [lib("libnvidia-container.so.1", "/usr/lib/x.so")];
        assert!(direct_nvidia_args(&pb(&["/dev/nvidiactl"]), &only_container, m).is_none());
    }

    #[test]
    fn nvidia_driver_lib_names() {
        for name in [
            "libcuda.so",
            "libcuda.so.1",
            "libcuda.so.580.1",
            "libnvidia-ml.so.1",
            "libnvidia-nvvm.so.4",
            "libnvidia-gpucomp.so.580.1",
        ] {
            assert!(is_nvidia_driver_lib(name), "{name}");
        }
        for name in [
            "libnvidia-container.so.1",
            "libnvidia-container-go.so.1",
            "libcudart.so.12",
            "libcudadebugger.so.1",
            "libnvidia-ml.a",
            "libc.so.6",
        ] {
            assert!(!is_nvidia_driver_lib(name), "{name}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn scan_nvidia_libs_follows_symlinks_but_keeps_each_name() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let (first, second) = (dir.path().join("lib64"), dir.path().join("lib"));
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(first.join("libcuda.so.580.1"), "x").unwrap();
        symlink("libcuda.so.580.1", first.join("libcuda.so.1")).unwrap();
        std::fs::write(first.join("libnvidia-ml.so.580.1"), "x").unwrap();
        std::fs::write(first.join("libnvidia-container.so.1"), "x").unwrap();
        std::fs::write(first.join("libcudart.so.12"), "x").unwrap();
        symlink("missing.so", first.join("libnvidia-gone.so.1")).unwrap();
        std::fs::write(second.join("libcuda.so.1"), "other").unwrap();

        let real = std::fs::canonicalize(first.join("libcuda.so.580.1")).unwrap();
        let libs = scan_nvidia_libs(&[first.clone(), dir.path().join("absent"), second.clone()]);
        let names: Vec<&str> = libs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "libcuda.so.1",
                "libcuda.so.580.1",
                "libnvidia-ml.so.580.1",
                "libcuda.so.1"
            ]
        );
        assert_eq!(libs[0].1, real);
        assert_eq!(libs[1].1, real);
        assert_eq!(
            libs[3].1,
            std::fs::canonicalize(second.join("libcuda.so.1")).unwrap()
        );

        let args =
            direct_nvidia_args(&pb(&["/dev/nvidiactl"]), &libs, "aarch64-linux-gnu").unwrap();
        assert!(
            args.contains(&format!(
                "{}:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro",
                real.display()
            )),
            "{args:?}"
        );
        assert_eq!(
            args.iter()
                .filter(|a| a.ends_with("/libcuda.so.1:ro"))
                .count(),
            1
        );
    }

    #[test]
    fn nvidia_device_nodes_are_the_control_nodes_then_each_gpu_by_number() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "nvidia10",
            "nvidia2",
            "nvidia0",
            "nvidiactl",
            "nvidia-uvm-tools",
            "nvidia-uvm",
            "nvidia-caps",
            "nvidia",
            "null",
        ] {
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let names: Vec<String> = nvidia_device_nodes(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "nvidiactl",
                "nvidia-uvm",
                "nvidia-uvm-tools",
                "nvidia0",
                "nvidia2",
                "nvidia10"
            ]
        );
        assert!(nvidia_device_nodes(&dir.path().join("absent")).is_empty());
    }

    #[test]
    fn cuda_container_args_fall_back_to_direct_passthrough_only_without_the_toolkit() {
        let direct = || Some(vec!["--device".to_string(), "/dev/nvidiactl".to_string()]);
        for backend in [GpuBackend::Cuda12, GpuBackend::Cuda13] {
            assert_eq!(
                backend.container_args_with(|| true, || unreachable!("toolkit present")),
                ["--gpus", "all"]
            );
            assert_eq!(
                backend.container_args_with(|| false, direct),
                ["--device", "/dev/nvidiactl"]
            );
            assert_eq!(
                backend.container_args_with(|| false, || None),
                ["--gpus", "all"]
            );
        }
        for backend in [GpuBackend::Cpu, GpuBackend::Rocm, GpuBackend::Vulkan] {
            assert_eq!(
                backend.container_args_with(
                    || unreachable!("only CUDA consults the toolkit"),
                    || unreachable!("only CUDA passes the driver through"),
                ),
                backend.engine_args()
            );
        }
    }

    #[test]
    fn a_direct_passthrough_run_stays_read_only_and_keeps_the_gpu_cache_tmpfs() {
        let direct = direct_nvidia_args(
            &pb(&["/dev/nvidiactl", "/dev/nvidia0"]),
            &[lib("libcuda.so.1", "/usr/lib/libcuda.so.1")],
            "x86_64-linux-gnu",
        )
        .unwrap();
        let flags = hardened_flags(&direct, &llama_tmpfs(&direct), None);
        assert!(flags.contains(&"--read-only".to_string()), "{flags:?}");
        assert!(
            flags
                .windows(2)
                .any(|w| w == ["--tmpfs", "/root/.cache:exec"]),
            "{flags:?}"
        );
        let toolkit = GpuBackend::Cuda12.engine_args();
        assert!(!hardened_flags(&toolkit, &llama_tmpfs(&toolkit), None)
            .contains(&"--read-only".to_string()));
    }

    /// Whatever GPU this host has, the speech worker's image is the CPU
    /// one, pinned like the others.
    #[test]
    fn speech_image_is_the_cpu_image_on_any_host() {
        assert_eq!(
            image_for(ContainerEngine::Speech, Some("b9994"), None).unwrap(),
            GpuBackend::Cpu.image_ref(Some("b9994"))
        );
        assert_eq!(
            image_for(ContainerEngine::Speech, None, None).unwrap(),
            GpuBackend::Cpu.image_ref(None)
        );
    }

    #[test]
    fn image_ref_uses_floating_tag_when_no_version_given() {
        assert_eq!(
            GpuBackend::Cpu.image_ref(None),
            "ghcr.io/ggml-org/llama.cpp:server"
        );
        assert_eq!(
            GpuBackend::Cuda13.image_ref(None),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13"
        );
    }

    #[test]
    fn image_ref_pins_to_the_given_version() {
        assert_eq!(
            GpuBackend::Cpu.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-b9994"
        );
        assert_eq!(
            GpuBackend::Cuda13.image_ref(Some("b9994")),
            "ghcr.io/ggml-org/llama.cpp:server-cuda13-b9994"
        );
    }

    #[test]
    fn cpu_backend_has_no_extra_engine_args() {
        assert!(GpuBackend::Cpu.engine_args().is_empty());
    }

    #[test]
    fn cuda_backend_requests_all_gpus() {
        assert_eq!(GpuBackend::Cuda12.engine_args(), vec!["--gpus", "all"]);
        assert_eq!(GpuBackend::Cuda13.engine_args(), vec!["--gpus", "all"]);
    }

    #[test]
    fn rocm_backend_mounts_kfd_and_dri() {
        let args = GpuBackend::Rocm.engine_args();
        assert_eq!(
            args,
            vec![
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video"
            ]
        );
    }

    #[test]
    fn vulkan_backend_passes_the_devices_the_host_has() {
        let args = GpuBackend::Vulkan.engine_args();
        assert_eq!(
            args.contains(&"/dev/dri".to_string()),
            Path::new("/dev/dri").exists()
        );
        assert_eq!(
            args.contains(&"--gpus".to_string()),
            Path::new("/dev/nvidiactl").exists()
        );
    }

    #[test]
    fn container_manager_binary_names() {
        assert_eq!(ContainerManager::Docker.binary(), "docker");
        assert_eq!(ContainerManager::Podman.binary(), "podman");
    }

    #[test]
    fn host_arch_parses_only_the_two_architectures_vllm_ships() {
        assert_eq!(HostArch::parse("x86_64"), Some(HostArch::X86_64));
        assert_eq!(HostArch::parse("aarch64"), Some(HostArch::Aarch64));
        assert_eq!(HostArch::parse("s390x"), None);
        assert_eq!(HostArch::parse("riscv64"), None);
        assert_eq!(HostArch::parse(""), None);
    }

    #[test]
    fn cuda_12_hosts_get_the_cu129_vllm_image() {
        assert_eq!(
            VllmBackend::from_gpu(GpuBackend::Cuda12),
            VllmBackend::Cuda12
        );
        assert_eq!(
            VllmBackend::from_gpu(GpuBackend::Cuda13),
            VllmBackend::Cuda13
        );
        assert_eq!(
            VllmBackend::Cuda12
                .image_ref(HostArch::X86_64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-x86_64-cu129"
        );
        assert_eq!(
            VllmBackend::Cuda12
                .image_ref(HostArch::Aarch64, Some("v0.28.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.28.0-aarch64-cu129"
        );
    }

    #[test]
    fn vulkan_host_runs_the_vllm_cpu_image() {
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Vulkan), VllmBackend::Cpu);
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Cpu), VllmBackend::Cpu);
        assert_eq!(VllmBackend::from_gpu(GpuBackend::Rocm), VllmBackend::Rocm);
    }

    #[test]
    fn vllm_image_refs_match_docker_hub_tag_spellings() {
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::X86_64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-x86_64"
        );
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::Aarch64, None)
                .unwrap(),
            "docker.io/vllm/vllm-openai:latest-aarch64"
        );
        assert_eq!(
            VllmBackend::Rocm.image_ref(HostArch::X86_64, None).unwrap(),
            "docker.io/rocm/vllm:latest"
        );
        assert_eq!(
            VllmBackend::Cpu.image_ref(HostArch::X86_64, None).unwrap(),
            "docker.io/vllm/vllm-openai-cpu:latest-x86_64"
        );
        assert_eq!(
            VllmBackend::Cpu.image_ref(HostArch::Aarch64, None).unwrap(),
            "docker.io/vllm/vllm-openai-cpu:latest-arm64"
        );
    }

    #[test]
    fn vllm_image_override_is_verbatim_and_for_plain_vllm_only() {
        let img = Some("dhi.io/vllm-openai:0.32");
        let backend = VllmBackend::Cuda13;
        // No `docker.io/`, arch suffix or version pin.
        assert_eq!(
            backend
                .engine_image(ContainerEngine::Vllm, Some("v0.11.0"), img)
                .unwrap(),
            "dhi.io/vllm-openai:0.32"
        );
        // Arch-independent: other engines keep their own repos.
        for (engine, repo) in [
            (ContainerEngine::VllmOmni, "vllm/vllm-omni:"),
            (ContainerEngine::Sglang, "lmsysorg/sglang:"),
        ] {
            let image = backend.engine_image(engine, None, img).unwrap();
            assert!(image.contains(repo), "{image}");
        }
    }

    #[test]
    fn vllm_image_ref_pins_to_the_given_version() {
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::X86_64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.11.0-x86_64"
        );
        assert_eq!(
            VllmBackend::Cuda13
                .image_ref(HostArch::Aarch64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai:v0.11.0-aarch64"
        );
        assert_eq!(
            VllmBackend::Cpu
                .image_ref(HostArch::Aarch64, Some("v0.11.0"))
                .unwrap(),
            "docker.io/vllm/vllm-openai-cpu:v0.11.0-arm64"
        );
        // rocm/vllm's tags carry no arch suffix: the pin is the whole tag.
        assert_eq!(
            VllmBackend::Rocm
                .image_ref(
                    HostArch::X86_64,
                    Some("rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0")
                )
                .unwrap(),
            "docker.io/rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0"
        );
    }

    #[test]
    fn rocm_vllm_has_no_aarch64_image() {
        let err = VllmBackend::Rocm
            .image_ref(HostArch::Aarch64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rocm/vllm"), "{err}");
        assert!(err.contains("LLMMAN_LLM_LIBRARY=cpu"), "{err}");
    }

    #[test]
    fn vllm_omni_image_refs_match_docker_hub_tag_spellings() {
        // hub.docker.com/r/vllm/vllm-omni/tags: latest-x86_64, latest-aarch64,
        // v0.28.0-x86_64, ... — one CUDA build, no -cu129 variant.
        for cuda in [VllmBackend::Cuda12, VllmBackend::Cuda13] {
            assert_eq!(
                cuda.omni_image_ref(HostArch::X86_64, None).unwrap(),
                "docker.io/vllm/vllm-omni:latest-x86_64"
            );
            assert_eq!(
                cuda.omni_image_ref(HostArch::Aarch64, None).unwrap(),
                "docker.io/vllm/vllm-omni:latest-aarch64"
            );
            assert_eq!(
                cuda.omni_image_ref(HostArch::Aarch64, Some("v0.28.0"))
                    .unwrap(),
                "docker.io/vllm/vllm-omni:v0.28.0-aarch64"
            );
        }
    }

    #[test]
    fn vllm_omni_has_only_cuda_images() {
        let err = VllmBackend::Rocm
            .omni_image_ref(HostArch::X86_64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("vllm/vllm-omni"), "{err}");
        assert!(err.contains("--runtime"), "{err}");
        let err = VllmBackend::Cpu
            .omni_image_ref(HostArch::X86_64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("CUDA"), "{err}");
    }

    #[test]
    fn sglang_image_refs_match_docker_hub_tag_spellings() {
        // hub.docker.com/r/lmsysorg/sglang/tags: multi-arch `latest`,
        // `v0.5.19`, `latest-cu129`, `v0.5.19-cu129` — no arch suffix.
        for arch in [HostArch::X86_64, HostArch::Aarch64] {
            assert_eq!(
                VllmBackend::Cuda13.sglang_image_ref(arch, None).unwrap(),
                "docker.io/lmsysorg/sglang:latest"
            );
            assert_eq!(
                VllmBackend::Cuda13
                    .sglang_image_ref(arch, Some("v0.5.19"))
                    .unwrap(),
                "docker.io/lmsysorg/sglang:v0.5.19"
            );
            assert_eq!(
                VllmBackend::Cuda12.sglang_image_ref(arch, None).unwrap(),
                "docker.io/lmsysorg/sglang:latest-cu129"
            );
            assert_eq!(
                VllmBackend::Cuda12
                    .sglang_image_ref(arch, Some("v0.5.19"))
                    .unwrap(),
                "docker.io/lmsysorg/sglang:v0.5.19-cu129"
            );
        }
    }

    #[test]
    fn sglang_rocm_image_is_the_whole_pinned_tag_or_nothing() {
        assert_eq!(
            VllmBackend::Rocm
                .sglang_image_ref(HostArch::X86_64, Some("v0.5.19-rocm700-mi30x"))
                .unwrap(),
            "docker.io/lmsysorg/sglang:v0.5.19-rocm700-mi30x"
        );
        let err = VllmBackend::Rocm
            .sglang_image_ref(HostArch::X86_64, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sglang-version"), "{err}");
        assert!(err.contains("rocm700-mi30x"), "{err}");
        let err = VllmBackend::Rocm
            .sglang_image_ref(HostArch::Aarch64, Some("v0.5.19-rocm700-mi30x"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("aarch64"), "{err}");
    }

    #[test]
    fn sglang_has_no_cpu_image_here() {
        for arch in [HostArch::X86_64, HostArch::Aarch64] {
            let err = VllmBackend::Cpu
                .sglang_image_ref(arch, Some("v0.5.19"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("xeon"), "{err}");
            assert!(err.contains("LLMMAN_SAFETENSORS_ENGINE"), "{err}");
        }
    }

    #[test]
    fn sglang_run_args_launch_the_module_after_the_image_and_flags_after_that() {
        let (launcher, _) = ContainerEngine::Sglang.launcher().unwrap();
        let args = engine_run_args(
            run_args(
                VllmBackend::Cuda13.engine_args(),
                30000,
                &[],
                None,
                false,
                None,
                &[],
            ),
            None,
            "docker.io/lmsysorg/sglang:latest".into(),
            launcher,
            "/cache/abc/model",
            |dir: &str, host: &str| {
                vec![
                    "--model-path".into(),
                    dir.into(),
                    "--host".into(),
                    host.into(),
                ]
            },
        );
        let image = args
            .iter()
            .position(|a| a == "docker.io/lmsysorg/sglang:latest")
            .expect("image present");
        assert_eq!(&args[image - 2..image], &["--entrypoint", "python3"]);
        assert_eq!(
            &args[image + 1..],
            &[
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/models",
                "--host",
                "0.0.0.0"
            ]
        );
    }

    #[test]
    fn vllm_engine_args_add_ipc_host_on_top_of_gpu_passthrough() {
        assert_eq!(
            VllmBackend::Cuda13.engine_args(),
            vec!["--gpus", "all", "--ipc=host"]
        );
        let rocm = VllmBackend::Rocm.engine_args();
        assert!(
            rocm.starts_with(&GpuBackend::Rocm.engine_args()),
            "{rocm:?}"
        );
        assert_eq!(rocm.last().map(String::as_str), Some("--ipc=host"));
    }

    #[test]
    fn vllm_cpu_engine_args_grant_what_its_numa_thread_binding_needs() {
        assert_eq!(
            VllmBackend::Cpu.engine_args(),
            vec![
                "--security-opt",
                "seccomp=unconfined",
                "--cap-add",
                "SYS_NICE",
                "--ipc=host"
            ]
        );
    }

    #[test]
    fn vllm_run_args_put_the_mount_and_entrypoint_before_the_image_and_serve_after() {
        let (launcher, _) = ContainerEngine::Vllm.launcher().unwrap();
        let args = engine_run_args(
            run_args(
                VllmBackend::Cuda13.engine_args(),
                8000,
                &[],
                None,
                false,
                None,
                &[],
            ),
            None,
            "docker.io/vllm/vllm-openai:latest-x86_64".into(),
            launcher,
            "/cache/abc/model",
            |dir: &str, host: &str| {
                vec![
                    "serve".into(),
                    dir.into(),
                    "--host".into(),
                    host.into(),
                    "--served-model-name".into(),
                    "m".into(),
                ]
            },
        );
        let image = args
            .iter()
            .position(|a| a == "docker.io/vllm/vllm-openai:latest-x86_64")
            .expect("image present");
        let before = &args[..image];
        assert!(before.contains(&"--gpus".to_string()), "{before:?}");
        assert!(before.contains(&"--ipc=host".to_string()), "{before:?}");
        assert!(
            before.contains(&"/cache/abc/model:/models:ro".to_string()),
            "{before:?}"
        );
        assert_eq!(&before[before.len() - 2..], &["--entrypoint", "vllm"]);
        assert_eq!(
            &args[image + 1..],
            &[
                "serve",
                "/models",
                "--host",
                "0.0.0.0",
                "--served-model-name",
                "m"
            ]
        );
    }

    #[test]
    fn run_args_forward_only_the_requested_passthrough_vars() {
        const VAR: &str = "LLMMAN_TEST_RUN_ARGS_PASSTHROUGH";
        std::env::set_var(VAR, "1");
        let args = run_args(vec![], 8080, &[], None, false, None, &[]);
        assert_eq!(
            &args[..6],
            &["run", "--rm", "--init", "-t", "-p", "127.0.0.1:8080:8080"]
        );
        assert!(!args.contains(&VAR.to_string()), "{args:?}");
        let args = run_args(vec![], 8080, &[VAR], None, false, None, &[]);
        assert!(args.windows(2).any(|w| w == ["-e", VAR]), "{args:?}");
        // The value stays out of argv (it may be a secret).
        assert!(!args.iter().any(|a| a.starts_with(&format!("{VAR}="))));
    }

    #[test]
    fn verify_runs_the_served_image_with_its_gpu_passthrough() {
        for (backend, image) in [
            (GpuBackend::Cpu, "ghcr.io/ggml-org/llama.cpp:server-b9994"),
            (
                GpuBackend::Cuda13,
                "ghcr.io/ggml-org/llama.cpp:server-cuda13-b9994",
            ),
        ] {
            let args = verify_args(
                backend,
                backend.engine_args(),
                Some("b9994"),
                "llmman-verify-1",
                false,
                None,
            );
            assert_eq!(
                &args[..5],
                ["run", "--rm", "--init", "--name", "llmman-verify-1"]
            );
            let at = args.iter().position(|a| a == image).expect("image present");
            let mut between = backend.engine_args();
            between.extend(forwarded_env_args(&[]));
            assert_eq!(&args[5..at], between.as_slice(), "{backend:?}");
            assert_eq!(&args[at + 1..], ["--version"]);
        }
    }

    /// `args[prefix_len..]` starts with [`hardened_args`] for `tmpfs`
    /// and continues with `engine_args`.
    fn assert_hardened_between(
        args: &[String],
        prefix_len: usize,
        read_only: bool,
        tmpfs: &[&str],
        engine_args: &[String],
    ) {
        let mut expected = hardened_args(read_only, tmpfs);
        expected.extend(engine_args.iter().cloned());
        assert_eq!(
            &args[prefix_len..prefix_len + expected.len()],
            expected.as_slice(),
            "{args:?}"
        );
    }

    fn user(device_gids: &[u32]) -> HardenedUser {
        HardenedUser {
            uid: 1000,
            gid: 1000,
            device_gids: device_gids.to_vec(),
        }
    }

    #[test]
    fn hardened_user_follows_the_engine_the_owner_and_rootlessness() {
        use identity::Identity;
        let me = Identity::User {
            uid: 1000,
            gid: 1000,
        };
        // Rootful docker: the mounted store's owner wins over the daemon's ids.
        assert_eq!(
            HardenedUser::from_parts(&me, false, Some((1001, 1002)), vec![44]),
            Some(HardenedUser {
                uid: 1001,
                gid: 1002,
                device_gids: vec![44]
            })
        );
        // No readable owner (or nothing mounted): the daemon's own ids.
        assert_eq!(
            HardenedUser::from_parts(&me, false, None, vec![]),
            Some(user(&[]))
        );
        // Rootless docker: `0:0` is the host user; the host owner must not
        // remap it, and host gids would be subordinate ids, so none are added.
        let root = Identity::User { uid: 0, gid: 0 };
        assert_eq!(
            HardenedUser::from_parts(&root, true, Some((1000, 1000)), vec![44]),
            Some(HardenedUser {
                uid: 0,
                gid: 0,
                device_gids: vec![]
            })
        );
        // Rootless podman and the image's own user: no `--user`.
        assert_eq!(
            HardenedUser::from_parts(&Identity::KeepId, false, Some((1000, 1000)), vec![44]),
            None
        );
        assert_eq!(
            HardenedUser::from_parts(&Identity::Image, false, Some((1000, 1000)), vec![44]),
            None
        );
    }

    #[test]
    fn hardened_user_args_add_the_user_device_groups_and_tmp_caches() {
        // Devices passed: one numeric `--group-add` per host gid.
        assert_eq!(
            user(&[44, 993]).args(true),
            [
                "--user",
                "1000:1000",
                "--group-add",
                "44",
                "--group-add",
                "993",
                "-e",
                "HOME=/tmp",
                "-e",
                "XDG_CACHE_HOME=/tmp/.cache",
                "-e",
                "XDG_CONFIG_HOME=/tmp/.config",
                "-e",
                "TRITON_CACHE_DIR=/tmp/.triton",
            ]
        );
        // No devices: no `--group-add`, even with gids resolved.
        let cpu = user(&[44]).args(false);
        assert!(!cpu.contains(&"--group-add".to_string()), "{cpu:?}");
        assert_eq!(&cpu[..2], ["--user", "1000:1000"]);
        assert_eq!(cpu.len(), 2 + 2 * HARDENED_USER_ENV.len());
        // Rootless docker's `0:0`.
        let root = HardenedUser {
            uid: 0,
            gid: 0,
            device_gids: vec![],
        };
        assert_eq!(&root.args(true)[..2], ["--user", "0:0"]);
    }

    #[test]
    fn passes_devices_covers_device_nodes_while_the_read_only_skip_is_gpus_only() {
        for (backend, gpus, devices) in [
            (GpuBackend::Cpu, false, false),
            (GpuBackend::Cuda13, true, true),
            (GpuBackend::Rocm, false, true),
        ] {
            let engine_args = backend.engine_args();
            assert_eq!(passes_gpus(&engine_args), gpus, "{backend:?}");
            assert_eq!(passes_devices(&engine_args), devices, "{backend:?}");
            // `--read-only` is skipped only behind `--gpus`: ROCm's
            // `--device` nodes keep the read-only rootfs.
            let args = run_args(engine_args, 8080, &[], None, true, None, LLAMA_TMPFS);
            assert_eq!(
                args.contains(&"--read-only".to_string()),
                !gpus,
                "{backend:?}: {args:?}"
            );
        }
    }

    #[test]
    fn hardened_user_lands_after_the_block_and_before_the_engine_args_in_every_builder() {
        let gpu_user = user(&[44, 993]);
        // ROCm: `--device` nodes, so the device groups come along, ahead of the
        // engine's own `--group-add video`.
        let engine_args = GpuBackend::Rocm.engine_args();
        let args = run_args(
            engine_args.clone(),
            8080,
            &[],
            None,
            true,
            Some(&gpu_user),
            LLAMA_TMPFS,
        );
        let mut expected = hardened_args(true, LLAMA_TMPFS);
        expected.extend(gpu_user.args(true));
        expected.extend(engine_args);
        assert_eq!(
            &args[6..6 + expected.len()],
            expected.as_slice(),
            "{args:?}"
        );
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(at("--user") < at("--device"), "{args:?}");
        assert!(at("--group-add") < at("--device"), "{args:?}");
        // CPU: no device, so `--user` and the caches but no `--group-add`.
        let args = run_args(
            GpuBackend::Cpu.engine_args(),
            8080,
            &[],
            None,
            true,
            Some(&gpu_user),
            LLAMA_TMPFS,
        );
        assert!(
            args.windows(2).any(|w| w == ["--user", "1000:1000"]),
            "{args:?}"
        );
        assert!(!args.contains(&"--group-add".to_string()), "{args:?}");
        assert!(
            args.windows(2).any(|w| w == ["-e", "HOME=/tmp"]),
            "{args:?}"
        );
        // `--no-hardened` drops the user with the block.
        let args = run_args(
            GpuBackend::Cpu.engine_args(),
            8080,
            &[],
            None,
            false,
            Some(&gpu_user),
            LLAMA_TMPFS,
        );
        assert!(!args.contains(&"--user".to_string()), "{args:?}");
        assert!(!args.iter().any(|a| a == "HOME=/tmp"), "{args:?}");
        // The verify run and the worker take the same value.
        let args = verify_args(
            GpuBackend::Cuda13,
            GpuBackend::Cuda13.engine_args(),
            None,
            "v",
            true,
            Some(&gpu_user),
        );
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(
            at("--cap-drop") < at("--user") && at("--user") < at("--gpus"),
            "{args:?}"
        );
        assert!(
            args.windows(2).any(|w| w == ["--group-add", "44"]),
            "{args:?}"
        );
        let args = verify_args(
            GpuBackend::Cuda13,
            GpuBackend::Cuda13.engine_args(),
            None,
            "v",
            true,
            None,
        );
        assert!(!args.contains(&"--user".to_string()), "{args:?}");
        let args = mediagen_run_args(
            GpuBackend::Cpu.engine_args(),
            "image".into(),
            "/usr/bin/llmman",
            "/store",
            "/cache",
            "a-model",
            8090,
            None,
            true,
            Some(&user(&[])),
        );
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(at("--user") < at("-v"), "{args:?}");
        assert_eq!(args[at("--user") + 1], "1000:1000");
    }

    #[test]
    fn hardened_note_names_the_flags_and_the_opt_out() {
        assert_eq!(
            hardened_note(&hardened_args(true, LLAMA_TMPFS), None),
            "started hardened (--cap-drop all --security-opt no-new-privileges --read-only --tmpfs /tmp:exec); --no-hardened turns these flags off"
        );
        assert_eq!(
            hardened_note(&hardened_args(false, LLAMA_TMPFS), Some("/srv/store")),
            "started hardened (--cap-drop all --security-opt no-new-privileges --tmpfs /tmp:exec); chmod -R go+rX /srv/store lets the container read it as any user, and --no-hardened turns these flags off"
        );
    }

    #[test]
    fn speech_worker_note_is_read_only_on_a_gpu_host() {
        for host in [GpuBackend::Cuda13, GpuBackend::Vulkan] {
            let speech = HardenedSpawn::Speech.note(host, Vec::new, None, None, "/srv/store");
            assert!(speech.contains("--read-only"), "{host:?}: {speech}");
            assert_eq!(
                speech,
                hardened_note(
                    &hardened_args(true, &llama_tmpfs(&SPEECH_BACKEND.engine_args())),
                    Some("/srv/store")
                )
            );
            assert!(speech.contains("chmod -R go+rX /srv/store"), "{speech}");
            let llama = HardenedSpawn::LlamaServer.note(
                host,
                || host.engine_args(),
                None,
                None,
                "/srv/store",
            );
            assert_eq!(
                llama.contains("--read-only"),
                !passes_gpus(&host.engine_args()),
                "{host:?}: {llama}"
            );
            let vllm = HardenedSpawn::Vllm.note(host, Vec::new, None, None, "/srv/store");
            assert!(vllm.contains("/root/.triton:exec"), "{host:?}: {vllm}");
        }
    }

    #[test]
    fn spawn_note_carries_the_user_block_and_the_rpc_tmpfs_the_argv_had() {
        let gpu_user = user(&[44]);
        // ROCm passes `--device`, so the note has the group too; the RPC
        // tmpfs lands after the user block, as in `engine_run_args`.
        let vllm = HardenedSpawn::Vllm.note(
            GpuBackend::Rocm,
            Vec::new,
            Some(&gpu_user),
            Some("/run/vllm:exec"),
            "/srv/store",
        );
        let expected = format!(
            "started hardened ({} --user 1000:1000 --group-add 44 -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e XDG_CONFIG_HOME=/tmp/.config -e TRITON_CACHE_DIR=/tmp/.triton --tmpfs /run/vllm:exec); chmod -R go+rX /srv/store lets the container read it as any user, and --no-hardened turns these flags off",
            hardened_args(true, ENGINE_TMPFS).join(" ")
        );
        assert_eq!(vllm, expected);
        // SGLang never gets the RPC tmpfs; the CPU backend passes no device.
        let sglang = HardenedSpawn::Sglang.note(
            GpuBackend::Cpu,
            Vec::new,
            Some(&gpu_user),
            Some("/run/vllm:exec"),
            "/srv/store",
        );
        assert!(!sglang.contains("/run/vllm"), "{sglang}");
        assert!(!sglang.contains("--group-add"), "{sglang}");
        assert!(sglang.contains("--user 1000:1000 -e HOME=/tmp"), "{sglang}");
        // The verify run's note is the same block for its argv.
        let engine_args = GpuBackend::Rocm.engine_args();
        let flags = hardened_flags(&engine_args, &llama_tmpfs(&engine_args), Some(&gpu_user));
        let verify = verify_args(
            GpuBackend::Rocm,
            GpuBackend::Rocm.engine_args(),
            None,
            "x",
            true,
            Some(&gpu_user),
        );
        assert_eq!(&verify[5..5 + flags.len()], flags.as_slice(), "{verify:?}");
    }

    #[test]
    fn hardened_args_drop_caps_and_mount_each_tmpfs() {
        assert_eq!(
            hardened_args(true, &["/tmp", "/root/.cache"]),
            [
                "--cap-drop",
                "all",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                "/tmp:exec",
                "--tmpfs",
                "/root/.cache:exec",
            ]
        );
        let gpu = hardened_args(false, &["/tmp"]);
        assert!(!gpu.contains(&"--read-only".to_string()), "{gpu:?}");
        assert_eq!(
            &gpu[..4],
            ["--cap-drop", "all", "--security-opt", "no-new-privileges"]
        );
        assert_eq!(&gpu[4..], ["--tmpfs", "/tmp:exec"]);
    }

    #[test]
    fn hardened_llama_container_is_read_only_on_cpu_but_not_behind_gpus() {
        let opts = |hardened| LlamaOptions {
            port: 8080,
            ctx_size: None,
            flash_attention: None,
            kv_cache_type: None,
            context_shift: true,
            split_mode: None,
            num_parallel: None,
            embeddings: false,
            batch_size: None,
            threads: None,
            cpus: Some(2.0),
            hardened,
            extra_args: &[],
        };
        let argv = |backend: GpuBackend, hardened| {
            llama_run_args(
                backend.engine_args(),
                backend.image_ref(None),
                "/cache/blobs",
                "sha256-abc",
                None,
                &[],
                opts(hardened),
                None,
            )
        };
        // `run --rm --init -t -p ADDR --cpus 2`, then the hardened block.
        let args = argv(GpuBackend::Cpu, true);
        assert_hardened_between(&args, 8, true, LLAMA_TMPFS, &GpuBackend::Cpu.engine_args());
        let image = args.iter().position(|a| a.starts_with("ghcr.io/")).unwrap();
        assert!(
            args[..image].contains(&"--read-only".to_string()),
            "{args:?}"
        );
        let args = argv(GpuBackend::Cuda13, true);
        assert_hardened_between(
            &args,
            8,
            false,
            &llama_tmpfs(&GpuBackend::Cuda13.engine_args()),
            &GpuBackend::Cuda13.engine_args(),
        );
        assert!(!args.contains(&"--read-only".to_string()), "{args:?}");
        assert!(
            args.windows(2).any(|w| w == ["--cap-drop", "all"]),
            "{args:?}"
        );
        assert!(
            args.windows(2).any(|w| w == ["--tmpfs", "/tmp:exec"]),
            "{args:?}"
        );
        assert!(args.windows(2).any(|w| w == ["--gpus", "all"]), "{args:?}");
        // Off: nothing of the block appears.
        let args = argv(GpuBackend::Cpu, false);
        assert!(
            !args
                .iter()
                .any(|a| a == "--cap-drop" || a == "--read-only" || a == "--tmpfs"),
            "{args:?}"
        );
    }

    #[test]
    fn hardened_engine_containers_keep_their_own_grants_after_the_drop() {
        let serve =
            |dir: &str, host: &str| vec!["serve".into(), dir.into(), "--host".into(), host.into()];
        let (vllm, _) = ContainerEngine::Vllm.launcher().unwrap();
        let (sglang, _) = ContainerEngine::Sglang.launcher().unwrap();
        for (backend, launcher, read_only) in [
            (VllmBackend::Cpu, vllm, true),
            (VllmBackend::Cuda13, vllm, false),
            (VllmBackend::Cuda13, sglang, false),
        ] {
            let engine_args = backend.engine_args();
            let args = engine_run_args(
                run_args(
                    engine_args.clone(),
                    8000,
                    &[],
                    None,
                    true,
                    None,
                    ENGINE_TMPFS,
                ),
                None,
                "image:tag".into(),
                launcher,
                "/cache/abc/model",
                serve,
            );
            // `run --rm --init -t -p ADDR`, then the hardened block, then the engine's own flags.
            assert_hardened_between(&args, 6, read_only, ENGINE_TMPFS, &engine_args);
            assert_eq!(
                args.contains(&"--read-only".to_string()),
                read_only,
                "{args:?}"
            );
            assert_eq!(
                args.iter().filter(|a| *a == "--tmpfs").count(),
                ENGINE_TMPFS.len()
            );
            assert!(args.contains(&"--ipc=host".to_string()), "{args:?}");
            let image = args.iter().position(|a| a == "image:tag").unwrap();
            assert_eq!(&args[image - 2..image - 1], &["--entrypoint"]);
        }
        // `--no-hardened`: no part of the block, for either launcher.
        for launcher in [vllm, sglang] {
            let args = engine_run_args(
                run_args(
                    VllmBackend::Cuda13.engine_args(),
                    8000,
                    &[],
                    None,
                    false,
                    None,
                    &[],
                ),
                None,
                "image:tag".into(),
                launcher,
                "/cache/abc/model",
                serve,
            );
            for flag in ["--cap-drop", "--security-opt", "--read-only", "--tmpfs"] {
                assert!(!args.contains(&flag.to_string()), "{args:?}");
            }
        }
        // vLLM CPU: `--cap-add SYS_NICE` and the seccomp opt-out come after `--cap-drop all`.
        let args = run_args(
            VllmBackend::Cpu.engine_args(),
            8000,
            &[],
            None,
            true,
            None,
            ENGINE_TMPFS,
        );
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(at("--cap-drop") < at("--cap-add"), "{args:?}");
        assert_eq!(args[at("--cap-add") + 1], "SYS_NICE");
        assert!(args.contains(&"seccomp=unconfined".to_string()), "{args:?}");
    }

    #[test]
    fn hardened_verify_runs_with_the_same_block_as_spawn() {
        for (backend, read_only) in [(GpuBackend::Cpu, true), (GpuBackend::Cuda13, false)] {
            let args = verify_args(
                backend,
                backend.engine_args(),
                Some("b9994"),
                "llmman-verify-1",
                true,
                None,
            );
            assert_eq!(
                &args[..5],
                ["run", "--rm", "--init", "--name", "llmman-verify-1"]
            );
            assert_hardened_between(
                &args,
                5,
                read_only,
                &llama_tmpfs(&backend.engine_args()),
                &backend.engine_args(),
            );
            let at = args
                .iter()
                .position(|a| a == &backend.image_ref(Some("b9994")))
                .unwrap();
            assert_eq!(&args[at + 1..], ["--version"]);
        }
    }

    fn worker_args(backend: GpuBackend, hardened: bool) -> Vec<String> {
        mediagen_run_args(
            backend.engine_args(),
            backend.image_ref(Some("b9994")),
            "/usr/bin/llmman",
            "/store",
            "/cache",
            "a-model",
            8090,
            None,
            hardened,
            None,
        )
    }

    #[test]
    fn worker_run_args_mount_the_binary_store_and_cache_read_only_then_serve() {
        // The media worker (CPU and CUDA) and the speech worker share
        // this shape; `--no-hardened` carries no hardened flag.
        for backend in [GpuBackend::Cpu, GpuBackend::Cuda12, SPEECH_BACKEND] {
            let args = worker_args(backend, false);
            let engine_args = backend.engine_args();
            assert_eq!(
                &args[..6],
                ["run", "--rm", "--init", "-t", "-p", "127.0.0.1:8090:8090"]
            );
            assert_eq!(&args[6..6 + engine_args.len()], engine_args.as_slice());
            for flag in ["--cap-drop", "--read-only", "--tmpfs"] {
                assert!(!args.contains(&flag.to_string()), "{backend:?}: {args:?}");
            }
            let image = backend.image_ref(Some("b9994"));
            let at = args.iter().position(|a| a == &image).unwrap();
            assert_eq!(
                &args[at - 12..at],
                [
                    "-v",
                    "/usr/bin/llmman:/usr/local/bin/llmman:ro",
                    "-v",
                    "/store:/store:ro",
                    "-v",
                    "/cache:/cache:ro",
                    "-e",
                    "LLMMAN_MODELS=/store",
                    "-e",
                    "PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "--entrypoint",
                    "/usr/local/bin/llmman",
                ]
            );
            assert_eq!(
                &args[at + 1..],
                [
                    "serve",
                    "a-model",
                    "--port",
                    "8090",
                    "--host",
                    "0.0.0.0",
                    "--runtime",
                    "path"
                ]
            );
        }
    }

    #[test]
    fn hardened_workers_drop_caps_and_otherwise_match_the_opt_out() {
        for (backend, read_only) in [
            (GpuBackend::Cpu, true),
            (GpuBackend::Cuda12, false),
            (SPEECH_BACKEND, true),
        ] {
            let args = worker_args(backend, true);
            let tmpfs = llama_tmpfs(&backend.engine_args());
            assert_hardened_between(&args, 6, read_only, &tmpfs, &backend.engine_args());
            // Only the hardened flags differ from the `--no-hardened` argv.
            let mut off = worker_args(backend, false);
            off.splice(6..6, hardened_args(read_only, &tmpfs));
            assert_eq!(args, off, "{backend:?}");
        }
    }

    #[test]
    fn cpus_is_forwarded_as_a_run_flag_only_when_a_limit_binds() {
        let args = run_args(vec![], 8080, &[], None, false, None, &[]);
        assert!(!args.contains(&"--cpus".to_string()), "{args:?}");
        let args = run_args(
            vec!["--gpus".into(), "all".into()],
            8080,
            &[],
            Some(2.0),
            false,
            None,
            &[],
        );
        let cpus = args.iter().position(|a| a == "--cpus").unwrap();
        assert_eq!(args[cpus + 1], "2");
        // A `run` flag: before the engine args, image and engine argv.
        assert!(cpus < args.iter().position(|a| a == "--gpus").unwrap());
        // A fractional cgroup quota is kept, not rounded to a whole CPU.
        let args = run_args(vec![], 8080, &[], Some(0.5), false, None, &[]);
        assert!(args.windows(2).any(|w| w == ["--cpus", "0.5"]), "{args:?}");
    }

    #[test]
    fn llama_container_gets_cpus_before_the_image_and_threads_after() {
        let opts = |threads, cpus| LlamaOptions {
            port: 8080,
            ctx_size: None,
            flash_attention: None,
            kv_cache_type: None,
            context_shift: true,
            split_mode: None,
            num_parallel: None,
            embeddings: false,
            batch_size: None,
            threads,
            cpus,
            hardened: false,
            extra_args: &[],
        };
        let image = "ghcr.io/ggml-org/llama.cpp:server";
        let argv = |o| {
            llama_run_args(
                GpuBackend::Cpu.engine_args(),
                image.into(),
                "/cache/blobs",
                "sha256-abc",
                None,
                &[],
                o,
                None,
            )
        };
        let args = argv(opts(Some(2), Some(2.0)));
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(at("--cpus") < at(image), "{args:?}");
        assert!(at(image) < at("--threads"), "{args:?}");
        assert_eq!(args[at("--cpus") + 1], "2");
        assert_eq!(args[at("--threads") + 1], "2");
        let args = argv(opts(None, None));
        assert!(
            !args.iter().any(|a| a == "--cpus" || a == "--threads"),
            "{args:?}"
        );
    }

    #[test]
    fn llama_container_ends_with_the_extra_args() {
        let opts = LlamaOptions {
            port: 8080,
            ctx_size: None,
            flash_attention: None,
            kv_cache_type: None,
            context_shift: true,
            split_mode: None,
            num_parallel: None,
            embeddings: false,
            batch_size: None,
            threads: Some(2),
            cpus: None,
            hardened: false,
            extra_args: &["--slot-prompt-similarity".into(), "0.5".into()],
        };
        let args = llama_run_args(
            GpuBackend::Cpu.engine_args(),
            "ghcr.io/ggml-org/llama.cpp:server".into(),
            "/cache/blobs",
            "sha256-abc",
            None,
            &[],
            opts,
            None,
        );
        assert_eq!(
            args[args.len() - 4..],
            ["--threads", "2", "--slot-prompt-similarity", "0.5"]
        );
    }

    #[test]
    fn vllm_containers_get_cpus_too() {
        let (launcher, _) = ContainerEngine::Vllm.launcher().unwrap();
        let args = engine_run_args(
            run_args(
                VllmBackend::Cpu.engine_args(),
                8000,
                &[],
                Some(3.0),
                false,
                None,
                &[],
            ),
            None,
            "docker.io/vllm/vllm-openai-cpu:latest-x86_64".into(),
            launcher,
            "/cache/abc/model",
            |dir: &str, host: &str| vec!["serve".into(), dir.into(), "--host".into(), host.into()],
        );
        assert!(args.windows(2).any(|w| w == ["--cpus", "3"]), "{args:?}");
    }

    /// Exercises real end-to-end backend detection (via
    /// `crate::hostgpu::detect`'s own real hardware probing — see that
    /// module's own real-hardware test) against whatever GPU is actually
    /// present on the machine running the test. Run explicitly with
    /// `cargo test --bin llmman -- --ignored --nocapture
    /// detect_backend_reports_this_hosts_real_hardware`.
    #[test]
    #[ignore = "result depends on this host's actual GPU/driver setup"]
    fn detect_backend_reports_this_hosts_real_hardware() {
        let backend = detect_backend();
        println!(
            "container::detect_backend() -> {backend:?} ({})",
            backend.image_ref(None)
        );
    }
    #[test]
    fn gpu_llama_containers_get_the_shader_cache_tmpfs() {
        let opts = LlamaOptions {
            port: 8080,
            ctx_size: None,
            flash_attention: None,
            kv_cache_type: None,
            context_shift: true,
            split_mode: None,
            num_parallel: None,
            embeddings: false,
            batch_size: None,
            threads: None,
            cpus: None,
            hardened: true,
            extra_args: &[],
        };
        let argv = |backend: GpuBackend| {
            llama_run_args(
                backend.engine_args(),
                backend.image_ref(None),
                "/cache/blobs",
                "sha256-abc",
                None,
                &[],
                opts,
                None,
            )
        };
        let cache = |args: &[String]| {
            args.windows(2)
                .filter(|w| w == &["--tmpfs", "/root/.cache:exec"])
                .count()
        };
        // ROCm's `--device /dev/kfd`: the cache tmpfs, ahead of the image.
        let args = argv(GpuBackend::Rocm);
        assert!(args.contains(&"/dev/kfd".to_string()), "{args:?}");
        let image = args.iter().position(|a| a.starts_with("ghcr.io/")).unwrap();
        assert_eq!(cache(&args[..image]), 1, "{args:?}");
        assert_eq!(cache(&args[image..]), 0, "{args:?}");
        // Vulkan's `--device /dev/dri` (built by hand: the backend's argv
        // depends on the host's /dev).
        let vulkan = ["--device", "/dev/dri"].map(String::from);
        assert_eq!(llama_tmpfs(&vulkan), ["/tmp", "/root/.cache"]);
        let args = run_args(
            vulkan.to_vec(),
            8080,
            &[],
            None,
            true,
            None,
            &llama_tmpfs(&vulkan),
        );
        assert_eq!(cache(&args), 1, "{args:?}");
        assert!(args.contains(&"--read-only".to_string()), "{args:?}");
        // No device: `/tmp` only.
        let args = argv(GpuBackend::Cpu);
        assert_eq!(cache(&args), 0, "{args:?}");
        assert_eq!(
            args.iter().filter(|a| *a == "--tmpfs").count(),
            1,
            "{args:?}"
        );
        assert_eq!(llama_tmpfs(&GpuBackend::Cpu.engine_args()), ["/tmp"]);
        // The verify run follows the same rule.
        let args = verify_args(
            GpuBackend::Rocm,
            GpuBackend::Rocm.engine_args(),
            None,
            "v",
            true,
            None,
        );
        assert_eq!(cache(&args), 1, "{args:?}");
        let args = verify_args(
            GpuBackend::Cpu,
            GpuBackend::Cpu.engine_args(),
            None,
            "v",
            true,
            None,
        );
        assert_eq!(cache(&args), 0, "{args:?}");
    }

    #[test]
    fn vllm_rpc_tmpfs_mounts_an_absolute_path_outside_the_engine_tmpfs() {
        for (value, expected) in [
            (None, Ok(None)),
            (Some("/run/vllm"), Ok(Some("/run/vllm:exec".to_string()))),
            (Some("/tmp/vllm"), Ok(None)),
            (Some("/tmp"), Ok(None)),
            (Some("/root/.cache/rpc"), Ok(None)),
            // A sibling that merely shares the prefix is not under the root.
            (Some("/tmpfs"), Ok(Some("/tmpfs:exec".to_string()))),
            (Some("relative/path"), Err(())),
            (Some(""), Err(())),
            // `--tmpfs` would read the tail as mount options.
            (Some("/run/rpc:rw"), Err(())),
            (Some("/run/rpc,exec"), Err(())),
            // A `..` segment could land under a tmpfs root after the check.
            (Some("/run/../tmp/rpc"), Err(())),
            (Some("/run/.."), Err(())),
            // Dots inside a name are not a segment.
            (Some("/run/..rpc"), Ok(Some("/run/..rpc:exec".to_string()))),
        ] {
            let got = vllm_rpc_tmpfs(value, ENGINE_TMPFS);
            match expected {
                Ok(expected) => assert_eq!(got, Ok(expected), "{value:?}"),
                Err(()) => {
                    let err = got.expect_err(&format!("{value:?}"));
                    assert!(err.contains("VLLM_RPC_BASE_PATH"), "{err}");
                    assert!(err.contains(&format!("{:?}", value.unwrap())), "{err}");
                }
            }
        }
    }

    #[test]
    fn engine_rpc_tmpfs_only_applies_to_a_hardened_vllm_container() {
        let relative = || Some("relative/path".to_string());
        let absolute = || Some("/run/vllm".to_string());
        for engine in [ContainerEngine::Vllm, ContainerEngine::VllmOmni] {
            let (_, prefix) = engine.launcher().unwrap();
            assert_eq!(
                engine_rpc_tmpfs(prefix, true, absolute),
                Ok(Some("/run/vllm:exec".to_string()))
            );
            assert!(engine_rpc_tmpfs(prefix, true, relative).is_err());
            assert_eq!(engine_rpc_tmpfs(prefix, false, relative), Ok(None));
        }
        // SGLang never receives the variable, so neither value matters.
        let (_, prefix) = ContainerEngine::Sglang.launcher().unwrap();
        assert_eq!(engine_rpc_tmpfs(prefix, true, absolute), Ok(None));
        assert_eq!(engine_rpc_tmpfs(prefix, true, relative), Ok(None));
    }

    #[test]
    fn engine_run_args_mount_the_rpc_tmpfs_before_the_model() {
        let (launcher, _) = ContainerEngine::Vllm.launcher().unwrap();
        let engine_args = VllmBackend::Cpu.engine_args();
        let rpc = vllm_rpc_tmpfs(Some("/run/vllm"), ENGINE_TMPFS).unwrap();
        let args = engine_run_args(
            run_args(engine_args, 8000, &[], None, true, None, ENGINE_TMPFS),
            rpc.as_deref(),
            "image:tag".into(),
            launcher,
            "/cache/abc/model",
            |dir, host| vec!["serve".into(), dir.into(), "--host".into(), host.into()],
        );
        let at = |flag: &str| args.iter().position(|a| a == flag).unwrap();
        assert!(
            args.windows(2).any(|w| w == ["--tmpfs", "/run/vllm:exec"]),
            "{args:?}"
        );
        assert!(at("/run/vllm:exec") < at("-v"), "{args:?}");
        assert_eq!(
            args.iter().filter(|a| *a == "--tmpfs").count(),
            ENGINE_TMPFS.len() + 1
        );
        // SGLang gets no rpc tmpfs (see `engine_rpc_tmpfs`): only the
        // engine's own.
        let (launcher, prefix) = ContainerEngine::Sglang.launcher().unwrap();
        let rpc = engine_rpc_tmpfs(prefix, true, || Some("/run/vllm".to_string())).unwrap();
        let args = engine_run_args(
            run_args(
                VllmBackend::Cpu.engine_args(),
                8000,
                &[],
                None,
                true,
                None,
                ENGINE_TMPFS,
            ),
            rpc.as_deref(),
            "image:tag".into(),
            launcher,
            "/cache/abc/model",
            |dir, host| vec!["serve".into(), dir.into(), "--host".into(), host.into()],
        );
        assert!(!args.iter().any(|a| a == "/run/vllm:exec"), "{args:?}");
        assert_eq!(
            args.iter().filter(|a| *a == "--tmpfs").count(),
            ENGINE_TMPFS.len()
        );
    }
}
