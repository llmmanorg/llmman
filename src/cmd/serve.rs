//! `llmman serve` – HTTP server exposing Ollama, OpenAI (including the
//! Responses API), and Anthropic-compatible APIs backed by `llama-server`
//! sub-processes from llama.cpp.

use std::collections::{HashMap, VecDeque};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use anyhow::{anyhow, Context};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use clap::Args;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration, Instant};

use crate::default_store;
use crate::modelpack::{resolve_model, ModelPath};
use crate::providers::PLACEHOLDER_API_KEY;
use crate::storage::OciStore;
use crate::webui;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// `llmman serve --help`'s "Environment Variables:" section — mirrors
/// `ollama serve -h`'s equivalent section. Static text, not built from
/// live values like Ollama's (llmman's env vars mostly configure the
/// daemon, not the CLI process printing `--help`).
const SERVE_ENV_HELP: &str = "\
Environment Variables:
      LLMMAN_DEBUG                   Show additional debug information (e.g. LLMMAN_DEBUG=1)
      LLMMAN_HOST                    [host][:port] to bind (default \"127.0.0.1:17434\")
      LLMMAN_CONTEXT_LENGTH          Context length to use unless otherwise specified (default: VRAM-tiered)
      LLMMAN_KEEP_ALIVE              The duration that models stay loaded in memory (default \"5m\")
      LLMMAN_MAX_LOADED_MODELS       Maximum number of loaded models (default: unbounded)
      LLMMAN_MAX_TRANSFER_STREAMS    Maximum parallel transfer streams for safetensors model pulls (default 4)
      LLMMAN_MAX_QUEUE               Maximum number of queued requests (default 512)
      LLMMAN_MODELS                  The path to the models directory
      LLMMAN_NUM_PARALLEL            Maximum number of parallel requests per model (GGUF only)
      LLMMAN_NOPRUNE                 Do not prune model blobs on startup
      LLMMAN_ORIGINS                 A comma separated list of allowed CORS origins
      LLMMAN_SCHED_SPREAD            Always schedule model across all GPUs
      LLMMAN_FLASH_ATTENTION         Enable flash attention
      LLMMAN_KV_CACHE_TYPE           Quantization type for the K/V cache (default: f16)
      LLMMAN_LLM_LIBRARY             Set backend (cpu/cuda/cuda13/rocm/vulkan/metal) to bypass GPU autodetection
      LLMMAN_GPU_OVERHEAD            Reserve a portion of VRAM (bytes)
      LLMMAN_IGPU_ENABLE             Enable integrated GPUs
      LLMMAN_LOAD_TIMEOUT            How long to allow model loads to stall before giving up (default \"10m\")
      LLMMAN_TMPDIR                  Staging directory for llama-server release downloads
      LLAMA_ARG_FIT                  Enable llama.cpp automatic fit of unset memory options (default \"on\")
      LLAMA_ARG_FIT_TARGET           Target free VRAM margin per device for llama.cpp fit (MiB)
      LLAMA_ARG_THREADS              Thread count for llama-server (default: llama-server autodetection, overridden by a binding CPU quota/affinity limit)
";

#[derive(Args, Debug)]
#[command(after_help = SERVE_ENV_HELP)]
pub struct ServeArgs {
    /// Model to pre-load immediately on startup (e.g. hf.co/unsloth/Qwen3.5-0.8B-GGUF:latest)
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Run llama-server in a container (docker or podman) instead of as a
    /// local process — Linux only. Auto-selects the matching
    /// ghcr.io/ggml-org/llama.cpp:server-<backend> image for whatever GPU
    /// acceleration the host has (see crate::container); no local
    /// llama-server binary is required on PATH when this is set.
    #[arg(long, value_name = "docker|podman")]
    pub ociman: Option<crate::container::ContainerManager>,

    /// Pin the llama.cpp release used for this server, instead of always
    /// taking whatever is currently latest. With --ociman, this pins the
    /// ghcr.io/ggml-org/llama.cpp container image tag (e.g. `b9994`
    /// instead of the floating `server`/`server-cuda`/... tags — pick one
    /// that's actually published for every backend variant you might run;
    /// see docs/docker.md in ggml-org/llama.cpp). Without --ociman, this
    /// pins which GitHub release of llama.cpp's own prebuilt
    /// `llama-server` `llmman serve` downloads and caches (see
    /// crate::llama_release) — set this to force that managed download
    /// even when some other `llama-server` is already on PATH, which is
    /// otherwise preferred untouched.
    #[arg(long, value_name = "TAG")]
    pub llama_cpp_version: Option<String>,

    /// Proactively pull the ghcr.io/ggml-org/llama.cpp image `--ociman`
    /// would run, as its own explicit foreground step, then exit — this
    /// process does not go on to bind the listener or serve — with the
    /// pull's own progress (a real `docker pull`/`podman pull` progress
    /// bar) inherited directly to this process's stdout/stderr — only
    /// meaningful together with --ociman, ignored otherwise.
    ///
    /// `--ociman`'s underlying `docker run`/`podman run` pulls an image
    /// that isn't already cached on its own, but silently: `serve` is
    /// normally started detached (see daemon.rs), its stdio redirected to
    /// a log file, so a caller waiting on the first request that actually
    /// needs the container (the first real prompt) sees nothing happen
    /// for however long a multi-hundred-MB-to-GB image pull takes —
    /// indistinguishable from a hang. Run `llmman serve --ociman ...
    /// --pull-oci` first, in the foreground, to do that pull visibly and
    /// finish as soon as it completes; then start the real, detached
    /// `llmman serve --ociman ...` (without `--pull-oci`) separately.
    #[arg(long, requires = "ociman")]
    pub pull_oci: bool,

    /// Proactively download and cache the local `llama-server` binary
    /// `llmman serve` would otherwise fetch on first use (see
    /// crate::llama_release), as its own explicit foreground step, then
    /// exit — the non-container equivalent of --pull-oci: same rationale,
    /// same "run this first, in the foreground, then start the real
    /// `llmman serve` separately" pattern, just for the local-binary path
    /// instead of --ociman's container path. Backend selection (CPU,
    /// CUDA, ROCm, Vulkan, Metal) uses the same host detection
    /// (crate::hostgpu) as a normal `llmman serve` would, mirroring
    /// llama.cpp's own installer's CUDA > ROCm > Vulkan > CPU probing
    /// order. Not meaningful together with --ociman (that path never
    /// resolves a local binary at all).
    #[arg(long, conflicts_with_all = ["ociman", "pull_oci"])]
    pub pull_bin: bool,
}

/// Context tokens requested for every `llama-server` this daemon spawns —
/// read from `LLMMAN_CONTEXT_LENGTH` (an env var, not a `llmman serve`
/// flag). A ceiling, not a guarantee: llama-server caps it back down to
/// a model's own trained context (`n_ctx_train`) when that's smaller,
/// with a warning, since serving positions past a model's trained
/// length risks incoherent/NaN output.
///
/// Unset or unparseable, this falls back to
/// [`crate::hostgpu::default_ctx_size`]: a VRAM-tiered value (see that
/// function's doc comment).
fn context_length_from_env() -> Option<u32> {
    parse_context_length(std::env::var("LLMMAN_CONTEXT_LENGTH").ok().as_deref())
}

/// [`context_length_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_context_length(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok()
}

/// Flash Attention mode requested for every `llama-server` this daemon
/// spawns — read from `LLMMAN_FLASH_ATTENTION` (an env var, not a
/// `llmman serve` flag, mirroring [`context_length_from_env`]). Forwarded
/// verbatim as `--flash-attn <mode>`; unset leaves it off llama-server's
/// own command line entirely, falling back to its own default (`auto`,
/// which already enables it whenever the backend/model support it).
fn flash_attention_from_env() -> Option<String> {
    parse_flash_attention(std::env::var("LLMMAN_FLASH_ATTENTION").ok().as_deref())
}

/// [`flash_attention_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment. Accepts llama-server's
/// own vocabulary (`on`/`off`/`auto`) as well as the boolean spelling
/// (`1`/`0`, `true`/`false`) Ollama documents for `OLLAMA_FLASH_ATTENTION`,
/// since users porting a config from there would otherwise silently get
/// llama-server's default instead of what they asked for.
fn parse_flash_attention(value: Option<&str>) -> Option<String> {
    let v = value?.trim();
    if v.is_empty() {
        return None;
    }
    Some(match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => "on".to_string(),
        "0" | "false" | "no" => "off".to_string(),
        other => other.to_string(),
    })
}

/// KV-cache quantization type requested for every `llama-server` this
/// daemon spawns — read from `LLMMAN_KV_CACHE_TYPE` (an env var, not a
/// `llmman serve` flag, mirroring [`context_length_from_env`]). Forwarded
/// as both `--cache-type-k` and `--cache-type-v`: llama-server takes
/// those separately, but Ollama's `OLLAMA_KV_CACHE_TYPE` (the convention
/// this mirrors) documents a single value applied to both, and there's no
/// use case yet for setting K and V independently through this daemon.
///
/// One of `f16` (llama-server's own default), `q8_0`, or `q4_0` — the
/// same set Ollama documents — trades output quality for a smaller
/// KV-cache footprint at long context lengths. Not validated here;
/// llama-server rejects an unsupported value itself, surfaced via
/// `wait_for_ready`'s stderr-tail capture same as any other startup
/// failure.
fn kv_cache_type_from_env() -> Option<String> {
    parse_kv_cache_type(std::env::var("LLMMAN_KV_CACHE_TYPE").ok().as_deref())
}

/// [`kv_cache_type_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_kv_cache_type(value: Option<&str>) -> Option<String> {
    let v = value?.trim();
    (!v.is_empty()).then(|| v.to_string())
}

/// `--split-mode` value requested for every `llama-server` spawn — read
/// from `LLMMAN_SCHED_SPREAD`, llmman's equivalent of Ollama's
/// `OLLAMA_SCHED_SPREAD`. Truthy forwards `--split-mode layer` (spread
/// across every GPU — already llama-server's own default, now explicit);
/// falsey forwards `--split-mode none` (restrict to one GPU). Unset
/// leaves llama-server's own default untouched.
fn sched_spread_from_env() -> Option<&'static str> {
    parse_sched_spread(std::env::var("LLMMAN_SCHED_SPREAD").ok().as_deref())
}

/// [`sched_spread_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_sched_spread(value: Option<&str>) -> Option<&'static str> {
    let v = value?.trim();
    if v.is_empty() {
        return None;
    }
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "layer" => Some("layer"),
        "0" | "false" | "no" | "off" | "none" => Some("none"),
        _ => None,
    }
}

/// `--parallel <n>` for every `llama-server` this daemon spawns (GGUF
/// models only — vllm/mlx_lm.server handle concurrency their own way,
/// with no equivalent flag), from `LLMMAN_NUM_PARALLEL`. Mirrors
/// Ollama's `OLLAMA_NUM_PARALLEL`; unset leaves llama-server's own
/// default of 1 untouched.
fn num_parallel_from_env() -> Option<u32> {
    parse_num_parallel(std::env::var("LLMMAN_NUM_PARALLEL").ok().as_deref())
}

/// `0` is rejected, same as llama-server's own `--parallel` validation.
fn parse_num_parallel(value: Option<&str>) -> Option<u32> {
    let n: u32 = value?.trim().parse().ok()?;
    (n != 0).then_some(n)
}

/// `--threads <n>` for local `llama-server` spawns, `Some` only when a
/// CPU limit binds. llama-server's own autodetection
/// (`cpu_get_num_math()`) already picks the physical/math cores, so an
/// unconstrained host passes nothing and leaves that choice alone. The
/// derived value only corrects the case autodetection cannot see: a
/// cgroup CPU quota, or a narrowed affinity mask, both carried by
/// `std::thread::available_parallelism` (std walks /proc/self/cgroup
/// and the ancestor chain itself, v1 and v2). A limit binds when
/// `available_parallelism` is below the online CPU count; then that
/// smaller value is passed. Accepted tradeoff: a quota between the
/// physical-core and SMT-thread counts (e.g. --cpus=12 on an
/// 8-core/16-thread host) passes 12 where autodetection would pick 8.
/// Any read or parse failure returns `None`: fail closed to
/// autodetection. `LLAMA_ARG_THREADS` set in the environment wins:
/// llama-server reads it itself via plain env inheritance, so `None`
/// here keeps that explicit choice untouched.
fn threads_from_env_or_host() -> Option<u32> {
    if std::env::var_os("LLAMA_ARG_THREADS").is_some() {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let allowed = std::thread::available_parallelism().ok()?.get() as u32;
        let online = online_cpu_count()?;
        (allowed < online).then_some(allowed)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Online CPUs from /sys/devices/system/cpu/online, the baseline
/// `available_parallelism` is compared against to decide whether a
/// quota or affinity limit binds. `None` when the file is unreadable
/// or malformed.
#[cfg(target_os = "linux")]
fn online_cpu_count() -> Option<u32> {
    cpu_list_count(&std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?)
}

/// CPU count from a kernel CPU list such as /sys/devices/system/cpu/online:
/// comma-separated single IDs or inclusive ranges (`0-15`, `0,4-7`).
/// `None` on empty or malformed content.
#[cfg(target_os = "linux")]
fn cpu_list_count(list: &str) -> Option<u32> {
    let mut count: u32 = 0;
    for part in list.trim().split(',') {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (u32, u32) = (lo.trim().parse().ok()?, hi.trim().parse().ok()?);
                if lo > hi {
                    return None;
                }
                count = count.checked_add(hi.checked_sub(lo)?.checked_add(1)?)?;
            }
            None => {
                let _: u32 = part.trim().parse().ok()?;
                count = count.checked_add(1)?;
            }
        }
    }
    (count > 0).then_some(count)
}

/// The `--ctx-size` value to actually forward to llama-server: `ctx_size`
/// (the per-request context every other computation — retries, error
/// messages, `LLMMAN_CONTEXT_LENGTH` itself — is expressed in) times
/// `num_parallel`. llama-server splits one `--ctx-size` evenly across
/// every `--parallel` slot rather than giving each its own full amount,
/// so forwarding `ctx_size` unscaled would silently divide a request's
/// real context by `num_parallel`; Ollama avoids exactly this by
/// launching with `NumCtx * numParallel` (`llm/llama_server.go`).
/// Callers should only ever pass a non-`None` `num_parallel` alongside
/// a `Some` `ctx_size` — see `ensure_model`'s own `num_parallel`
/// fallback, which drops it to `None` otherwise (nothing safe to scale
/// against).
fn backend_ctx_size(ctx_size: Option<u32>, num_parallel: Option<u32>) -> Option<u32> {
    ctx_size.map(|c| c.saturating_mul(num_parallel.unwrap_or(1)))
}

/// `num_parallel` unless `ctx_size` is `None` (a high-VRAM host
/// deferring to the model's own trained context, nothing safe to scale
/// against — see `backend_ctx_size`'s doc comment), in which case
/// `None`: forwarding `--parallel` unscaled would silently divide that
/// trained context across slots instead.
fn effective_num_parallel(ctx_size: Option<u32>, num_parallel: Option<u32>) -> Option<u32> {
    ctx_size.and(num_parallel)
}

/// Matches Ollama's own default for `OLLAMA_MAX_QUEUE`.
const DEFAULT_MAX_QUEUE: usize = 512;

/// Maximum number of requests [`ensure_model`] admits at once before
/// rejecting with a 503, from `LLMMAN_MAX_QUEUE` (mirrors Ollama's
/// `OLLAMA_MAX_QUEUE`). See [`try_admit`].
fn max_queue_from_env() -> usize {
    parse_max_queue(std::env::var("LLMMAN_MAX_QUEUE").ok().as_deref())
}

/// Unlike most other `parse_*` functions here, `0` is a real value (see
/// `try_admit_against`'s doc comment), not "unset".
fn parse_max_queue(value: Option<&str>) -> usize {
    match value.map(str::trim) {
        Some(v) if !v.is_empty() => v.parse().unwrap_or(DEFAULT_MAX_QUEUE),
        _ => DEFAULT_MAX_QUEUE,
    }
}

/// Maximum number of models [`ensure_model`] keeps loaded at once, from
/// `LLMMAN_MAX_LOADED_MODELS` (mirrors Ollama's `OLLAMA_MAX_LOADED_MODELS`,
/// but as one flat daemon-wide total, not per-GPU — llmman has no
/// per-model memory estimate to size a per-GPU figure against). `0` =
/// unbounded. See [`enforce_max_loaded_models`].
fn max_loaded_models_from_env() -> usize {
    parse_max_loaded_models(std::env::var("LLMMAN_MAX_LOADED_MODELS").ok().as_deref())
}

fn parse_max_loaded_models(value: Option<&str>) -> usize {
    value
        .map(str::trim)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Which local engine backs a resolved `ModelPath::SafeTensors`
/// directory: `mlx_lm.server` (see `spawn_mlx_server`) when this host is
/// Apple Silicon macOS (`crate::hostgpu::detect() == HostGpu::Metal`)
/// *and* `mlx_lm.server` is actually on `PATH`; `vllm` in every other
/// case, unchanged from before this engine existed.
///
/// Plain `vllm` (no plugin) has no Metal backend of its own at all — its
/// upstream-published macOS wheel is CPU-only. There *is* a way to make
/// `vllm serve` itself Metal-accelerated on Apple Silicon —
/// [vllm-metal](https://github.com/vllm-project/vllm-metal), an
/// installed-alongside `vllm.platform_plugins` plugin that overrides its
/// `CpuPlatform` autodetection with a real `MetalPlatform` (itself
/// implemented on top of MLX — see the `e2e` CI job's own "Install vLLM
/// (e2e)" step) — but it only supports a narrower set of model
/// families than `mlx_lm.server` does directly, and pulls in vLLM's own
/// full dependency footprint for a user who may not want any of the rest
/// of it. `mlx_lm.server` here is a separate, no-vLLM-at-all option: a
/// Mac with `mlx-lm` installed gets real Metal acceleration through it
/// without needing vllm-metal (or vllm) at all; a Mac with neither still
/// falls back to plain (CPU-only, absent vllm-metal) `vllm` instead of
/// failing outright.
fn use_mlx_for_safetensors() -> bool {
    crate::hostgpu::detect() == crate::hostgpu::HostGpu::Metal
        && which_binary("mlx_lm.server").is_ok()
}

/// Whether `model_ref` gets `--context-shift`. Enabled except for
/// DeepSeek-family ("deepseek2" architecture) models, mirroring Ollama's
/// own `supportsContextShift` (`server/sched.go`) — their MLA-compressed
/// KV cache can't be shifted the way llama-server expects. Ollama
/// detects that from parsed GGUF metadata; llmman deliberately doesn't
/// parse GGUF metadata at all (see modelpack.rs's removed
/// gguf_architecture note), so this is a coarser name-based heuristic
/// instead.
fn supports_context_shift(model_ref: &str) -> bool {
    !model_ref.to_ascii_lowercase().contains("deepseek")
}

// ---------------------------------------------------------------------------
// Out-of-memory auto-shrink retry (ensure_model) — mirrors Ollama's
// reduceAutoNumCtxForLoadOOM: a chosen --ctx-size can still be too big
// for actual free VRAM, so retry with it halved a few times instead of
// failing the load outright.
// ---------------------------------------------------------------------------

/// Max halving retries for an OOM-looking local llama-server load.
const MAX_CTX_SHRINK_ATTEMPTS: u32 = 4;

/// Floor below which a still-failing load is a hard failure, not
/// something to keep shrinking.
const MIN_CTX_SIZE_FOR_RETRY: u32 = 16384;

/// First retry value when `ctx_size` started as `None` (no number to
/// halve). Matches the top VRAM tier's own default (see
/// `hostgpu::default_ctx_size_for`) rather than starting below
/// `MIN_CTX_SIZE_FOR_RETRY`.
const STARTING_CTX_SIZE_FOR_UNBOUNDED_RETRY: u32 = 65536;

/// Next `--ctx-size` to retry an OOM'd load with, or `None` if shrinking
/// further wouldn't help (at/under the floor already).
fn next_ctx_size_after_oom(current: Option<u32>) -> Option<u32> {
    match current {
        None => Some(STARTING_CTX_SIZE_FOR_UNBOUNDED_RETRY),
        Some(n) => {
            let next = (n / 2).max(MIN_CTX_SIZE_FOR_RETRY);
            // `next < n`, not just `!=`: below the floor, halving+max
            // would otherwise suggest a *larger* ctx-size, backwards
            // after an OOM.
            (next < n).then_some(next)
        }
    }
}

/// True if `detail` (a failed load's stderr tail, or an error message)
/// looks like a memory-allocation failure rather than some other startup
/// error. Matched against known ggml/llama.cpp allocator log phrasings —
/// deliberately specific rather than one broad substring, since
/// misclassifying an unrelated failure as OOM would burn several slow
/// retries before surfacing the real error.
fn looks_like_oom(detail: &str) -> bool {
    let d = detail.to_ascii_lowercase();
    [
        "failed to allocate",
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "cudamalloc failed",
        "std::bad_alloc",
    ]
    .iter()
    .any(|needle| d.contains(needle))
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState(Arc<Inner>);

struct Inner {
    manager: Mutex<ModelManager>,
    // None when --ociman is set: llama-server then runs in a container, so
    // no local binary is resolved (or required on PATH) at all. Behind a
    // mutex because the path resolved at startup can be deleted while this
    // daemon keeps running (an upgrade/uninstall of whatever install
    // provided it) — see local_llama_server_bin, which re-resolves and
    // stores a replacement in that case.
    llama_server_bin: StdMutex<Option<PathBuf>>,
    // This daemon's own executable path, canonicalized at startup (while
    // it still exists on disk). Reported by /api/version so clients — the
    // CLI's daemon::ensure_server, sbx — can detect a daemon left running
    // after the install that provided its binary was deleted, instead of
    // blindly reusing it.
    exe: Option<PathBuf>,
    ociman: Option<crate::container::ContainerManager>,
    llama_cpp_version: Option<String>,
    // See context_length_from_env's doc comment — forwarded verbatim to
    // every spawn_llama_server/container::spawn call, local or
    // containerized.
    ctx_size: Option<u32>,
    // True if `ctx_size` came from an explicit LLMMAN_CONTEXT_LENGTH
    // rather than hostgpu's VRAM-tiered auto default — see
    // ensure_model's OOM retry loop, which only auto-shrinks the latter
    // (mirrors Ollama's own numCtxAuto gate on reduceAutoNumCtxForLoadOOM:
    // a user's explicit choice shouldn't be silently overridden).
    ctx_size_explicit: bool,
    // See flash_attention_from_env's doc comment — forwarded verbatim to
    // every spawn_llama_server/container::spawn call, local or
    // containerized.
    flash_attention: Option<String>,
    // See kv_cache_type_from_env's doc comment — forwarded verbatim to
    // every spawn_llama_server/container::spawn call, local or
    // containerized.
    kv_cache_type: Option<String>,
    // See sched_spread_from_env's doc comment — this is only the
    // *initial* value passed to spawn_llama_server/container::spawn;
    // ensure_model's OOM retry loop may relax an explicit `"none"` to
    // `"layer"` for that one load if the restriction itself looks like
    // the cause.
    split_mode: Option<&'static str>,
    // See num_parallel_from_env's doc comment.
    num_parallel: Option<u32>,
    // See threads_from_env_or_host's doc comment. Resolved once at
    // startup (this daemon's cgroup doesn't change per load) and passed
    // to every *local* spawn_llama_server call from ensure_model's OOM
    // retry loop.
    threads: Option<u32>,
    // See max_queue_from_env's doc comment; enforced by try_admit.
    max_queue: usize,
    // See max_loaded_models_from_env's doc comment.
    max_loaded_models: usize,
    store_path: PathBuf,
    cache_path: PathBuf,
    client: Client,
}

struct ModelManager {
    running: HashMap<String, RunningModel>,
    // Loads admitted by `enforce_max_loaded_models` but not yet in
    // `running` (still pulling/spawning/waiting-for-ready, or already
    // failed and about to release their slot) — counted alongside
    // `running.len()` when checking `LLMMAN_MAX_LOADED_MODELS`, so two
    // concurrent loads of two *different* new models can't both pass
    // that check and overshoot the cap.
    pending_loads: usize,
}

/// Everything `handle_ps` (and, transitively, `llmman ps`) needs to know
/// about a running model — see cmd::ps for the CLI side of this.
struct RunningModel {
    process: ModelProcess,
    port: u16,
    /// Full manifest digest (e.g. "sha256:abcd...") from the OCI store,
    /// captured at load time (see resolve_model's caller in ensure_model).
    digest: String,
    /// GGUF file size in bytes; 0 for a safetensors dir (vllm) — walking a
    /// multi-file safetensors directory isn't worth the cost just for
    /// `ps` output today.
    size: u64,
    started_at: String,
    /// Monotonic clock reading of this model's last activity (a request
    /// completing, or the model just finishing loading) — compared
    /// against `keep_alive` by `reap_idle_models`. A `tokio::time::Instant`
    /// rather than a wall-clock time so a system clock change (NTP step,
    /// suspend/resume) can't cause a premature or delayed unload.
    last_active: Instant,
    /// Wall-clock twin of `last_active`, kept only so `handle_ps` can
    /// report a real `expires_at` timestamp — `Instant` has no meaningful
    /// conversion to one.
    last_active_wall: chrono::DateTime<chrono::Utc>,
    /// How long after `last_active` this model should be automatically
    /// unloaded; `None` means "never" (Ollama's `keep_alive: -1`). Updated
    /// by `ActivityGuard` on every `/api/chat` and `/api/generate`
    /// request, and by `refresh_activity` for a load-only request that
    /// only wants to set/extend it.
    keep_alive: Option<Duration>,
    /// Count of requests currently being served by this model.
    /// `reap_idle_models` never unloads a model with `in_flight > 0`,
    /// however far past its `keep_alive` deadline `last_active` is — see
    /// `ActivityGuard`'s doc comment for why a generation slower than its
    /// own `keep_alive` must not be killed mid-stream.
    in_flight: u32,
    /// `Some(<absolute model directory path>)` only for `Engine::Mlx`,
    /// `None` for every other engine — see `backend_wire_model`'s own
    /// doc comment for what this is actually for (the `"model"` field a
    /// request must carry to reach *this* model on an `mlx_lm.server`
    /// backend, since it has no `--served-model-name`-equivalent way to
    /// register a human-readable alias for it up front the way `vllm`
    /// does).
    backend_model_path: Option<String>,
}

/// Which engine is actually serving requests for a [`RunningModel`] — surfaced
/// in `llmman ps`'s PROCESSOR column since, unlike Ollama's embedded
/// inference engine, llmman shells out to one of several different ones and
/// none of them report GPU/CPU memory split back to llmman, so there's no
/// equivalent of Ollama's "100% GPU"/"N%/N% CPU/GPU" figure to show here —
/// only which engine, and (for containers) which engine manager, is running.
impl RunningModel {
    fn processor(&self) -> String {
        match &self.process {
            ModelProcess::Local(Engine::LlamaServer, _, _) => "llama-server (local)".into(),
            ModelProcess::Local(Engine::Vllm, _, _) => "vllm (local)".into(),
            ModelProcess::Local(Engine::Mlx, _, _) => "mlx (local)".into(),
            ModelProcess::Container(ociman, _) => {
                format!("llama-server (container/{})", ociman.binary())
            }
        }
    }

    fn pid(&self) -> Option<u32> {
        match &self.process {
            ModelProcess::Local(_, child, _) => child.id(),
            ModelProcess::Container(_, child) => child.id(),
        }
    }
}

/// Which local engine a [`ModelProcess::Local`] is running — see
/// [`RunningModel::processor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    LlamaServer,
    Vllm,
    /// `mlx_lm.server` (the `mlx-lm` PyPI package) — Apple Silicon's own
    /// Metal-accelerated alternative to `vllm` for a
    /// [`ModelPath::SafeTensors`] directory, picked instead of it when
    /// [`use_mlx_for_safetensors`] says so. See [`spawn_mlx_server`]'s
    /// doc comment for why this engine's requests need a different
    /// `"model"` field than every other one (handled by
    /// [`backend_wire_model`]), not anything here.
    Mlx,
}

/// A running inference backend: either a local `llama-server`/`vllm`/
/// `mlx_lm.server` process (killed via `Child::kill_on_drop`, except
/// `Engine::Vllm` — see this Drop impl) or an attached `docker run`/
/// `podman run` process, gracefully stopped via SIGTERM on drop since
/// `kill_on_drop`'s SIGKILL can't be forwarded to (and so doesn't stop)
/// the container.
enum ModelProcess {
    // `Option<u32>` is the pid captured right after spawn, not
    // `child.id()` at drop time: `is_alive`'s `try_wait` reaps the child
    // once it exits, after which `child.id()` returns `None` — losing the
    // only pid needed to SIGKILL an `Engine::Vllm` group in Drop below.
    // Only ever read there, so it is genuinely dead on Windows, which has
    // no process group to signal; carrying it on every platform beats
    // cfg-ing the variant's shape at all ten construction/match sites.
    Local(
        Engine,
        tokio::process::Child,
        #[cfg_attr(not(unix), allow(dead_code))] Option<u32>,
    ),
    Container(crate::container::ContainerManager, tokio::process::Child),
}

impl Drop for ModelProcess {
    fn drop(&mut self) {
        match self {
            ModelProcess::Container(_, child) => {
                if let Some(pid) = child.id() {
                    crate::container::stop(pid);
                }
            }
            // vllm forks its own API-server/engine-core workers, which
            // don't share a process tree `kill_on_drop`'s single-pid kill
            // can reach — SIGKILLing just the top pid (e.g. on a
            // cancelled load) orphans them, still holding GPU memory
            // indefinitely. spawn_vllm_server puts this child in its own
            // process group so the whole group can be killed here.
            #[cfg(unix)]
            ModelProcess::Local(Engine::Vllm, _, pid) => {
                if let Some(pid) = pid {
                    let result = unsafe { libc::kill(-(*pid as libc::pid_t), libc::SIGKILL) };
                    if result != 0 {
                        let err = std::io::Error::last_os_error();
                        eprintln!(
                            "[llmman] warning: SIGKILL to vllm process group {pid} failed: {err}"
                        );
                    }
                }
            }
            #[cfg(not(unix))]
            ModelProcess::Local(Engine::Vllm, _, _) => {}
            // `mlx_lm.server` runs entirely as one process — a single
            // background generation thread plus a `ThreadingHTTPServer`,
            // no forked worker tree of its own the way vllm has above —
            // so the plain default `kill_on_drop` SIGKILL to just this
            // one pid is already sufficient; nothing extra to do here.
            ModelProcess::Local(Engine::Mlx, _, _) => {}
            ModelProcess::Local(Engine::LlamaServer, _, _) => {}
        }
    }
}

impl ModelProcess {
    /// True if the underlying child process hasn't exited on its own since
    /// this model was marked running. Nothing else ever tells `mgr.running`
    /// about a process exiting unexpectedly: every other removal is a
    /// deliberate one — the Ollama unload signal (`unload_model`, which
    /// both `handle_ollama_generate` and `handle_ollama_chat` route
    /// through); the idle reaper (`reap_idle_models_once`); and eviction
    /// under `LLMMAN_MAX_LOADED_MODELS` (`evict_other_models`) — and none
    /// of them fires on a crash. So a crash, an OOM kill, or anything else that
    /// takes `llama-server`/vllm down on its own would otherwise keep
    /// handing out that now-dead port forever, indistinguishable from a
    /// real live one until whichever caller's request to it fails with a
    /// bare connection error; `check_running` is what drops the entry, the
    /// moment this returns false. `try_wait` is non-blocking either way:
    /// `Ok(None)` (still running) is the overwhelmingly common case this
    /// needs to stay cheap for.
    fn is_alive(&mut self) -> bool {
        let child = match self {
            ModelProcess::Local(_, child, _) => child,
            ModelProcess::Container(_, child) => child,
        };
        matches!(child.try_wait(), Ok(None))
    }

    /// Stops this process and waits for it to actually exit, unlike this
    /// same cleanup on `Drop` above: `kill_on_drop`/a bare SIGTERM signal
    /// is fire-and-forget and doesn't wait for the OS to reap the
    /// process. Used by `ensure_model`'s OOM retry loop before spawning a
    /// replacement, so a still-exiting old server can't linger and race
    /// the new one (each retry also gets its own fresh port as a second
    /// safety net — see that loop's own comment).
    async fn stop_and_wait(&mut self) {
        match self {
            ModelProcess::Container(_, child) => {
                if let Some(pid) = child.id() {
                    crate::container::stop(pid);
                }
            }
            #[cfg(unix)]
            ModelProcess::Local(Engine::Vllm, _, pid) => {
                if let Some(pid) = pid {
                    unsafe { libc::kill(-(*pid as libc::pid_t), libc::SIGKILL) };
                }
            }
            ModelProcess::Local(_, _, _) => {}
        }
        // `Child::kill` sends SIGKILL and awaits the exit itself — after
        // an already-successful graceful stop above, this is a no-op
        // beyond confirming the process is actually gone.
        let child = match self {
            ModelProcess::Local(_, child, _) => child,
            ModelProcess::Container(_, child) => child,
        };
        let _ = child.kill().await;
    }
}

// ---------------------------------------------------------------------------
// Ollama API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    /// Base64-encoded image bytes (no `data:` prefix — matches Ollama's
    /// own wire format), one per attached image. Only meaningful on a
    /// request message; a response message never sets this. See
    /// `ollama_message_to_oai` for how these become OpenAI-style
    /// `image_url` content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    /// Set on an assistant response message that calls one or more tools
    /// (see `handle_ollama_chat`), and accepted back on a request message
    /// so multi-turn tool-calling history round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    /// Ollama's tool-result message (`role: "tool"`) carries the name of
    /// the tool it's a result for, but — unlike OpenAI's `tool_call_id` —
    /// no id linking it back to a specific call. See
    /// `ollama_message_to_oai`'s doc comment for the limitation that
    /// implies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

/// Ollama's tool-call shape (`api.ToolCall` in ollama/api/types.go):
/// `{"function": {"name": ..., "arguments": {...}}}` — unlike OpenAI's
/// `arguments` (a JSON-encoded *string*), Ollama's is already a decoded
/// JSON object, and there is no top-level `id`/`type`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct OllamaToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default = "bool_true")]
    stream: bool,
    options: Option<serde_json::Value>,
    /// Ollama's own top-level `think` field ("for thinking models, should
    /// the model think before responding? Can be a boolean or a thinking
    /// level"). See `think_to_chat_template_kwargs`.
    #[serde(default)]
    think: Option<serde_json::Value>,
    /// Tool/function definitions, in the same shape OpenAI's `tools`
    /// field uses (Ollama's own tool schema is already
    /// OpenAI-function-tool compatible) — passed straight through to
    /// llama-server. See `handle_ollama_chat`.
    #[serde(default)]
    tools: Option<serde_json::Value>,
    /// `"json"` for unconstrained-schema JSON mode, or a JSON Schema
    /// object for constrained structured output. See
    /// `format_to_response_format`.
    #[serde(default)]
    format: Option<serde_json::Value>,
    /// See `OllamaGenerateRequest::keep_alive`.
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "bool_true")]
    stream: bool,
    options: Option<serde_json::Value>,
    /// keep_alive: 0 with an empty prompt is the Ollama unload signal;
    /// otherwise resolved (see `resolve_keep_alive`) into how long this
    /// model should stay loaded once idle.
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    /// See `OllamaChatRequest::think`.
    #[serde(default)]
    think: Option<serde_json::Value>,
    /// See `OllamaChatRequest::format`. `/api/generate` has no `tools`
    /// field in real Ollama either — only `/api/chat` supports tool
    /// calling.
    #[serde(default)]
    format: Option<serde_json::Value>,
}

/// Maps Ollama's `format` request field to the OpenAI-style
/// `response_format` llama-server's `/v1/chat/completions` expects:
/// `"json"` becomes unconstrained JSON-object mode, and a JSON Schema
/// object becomes constrained (grammar-backed) structured output. Absent
/// or any other JSON type (Ollama documents only these two) is a no-op —
/// exactly as if the field weren't sent at all, matching
/// `think_to_chat_template_kwargs`'s own handling of shapes with no
/// equivalent.
fn format_to_response_format(format: &Option<serde_json::Value>) -> Option<serde_json::Value> {
    match format {
        Some(serde_json::Value::String(s)) if s == "json" => {
            Some(serde_json::json!({ "type": "json_object" }))
        }
        Some(schema @ serde_json::Value::Object(_)) => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema, "strict": true }
        })),
        _ => None,
    }
}

/// Translates Ollama's `think` request field into the
/// `chat_template_kwargs` llama-server actually reads. `true`/`false` →
/// `{"enable_thinking": <bool>}`. A string level (`"low"`/`"medium"`/
/// `"high"`/`"max"`) → `{"enable_thinking": true, "reasoning_effort":
/// <level>}`, the jinja variable gpt-oss's and DeepSeek-V4's own
/// templates read for reasoning depth. Anything else is a no-op.
fn think_to_chat_template_kwargs(think: &Option<serde_json::Value>) -> Option<serde_json::Value> {
    match think {
        Some(serde_json::Value::Bool(b)) => Some(serde_json::json!({ "enable_thinking": b })),
        // Only forward the four levels llama-server's own templates
        // actually understand — an unrecognized level (a typo, a future
        // Ollama addition, ...) is left a no-op rather than forwarded
        // verbatim, so the template's own default applies instead of
        // silently misbehaving on an unsupported reasoning_effort value.
        Some(serde_json::Value::String(level))
            if matches!(level.trim(), "low" | "medium" | "high" | "max") =>
        {
            Some(serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": level.trim(),
            }))
        }
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatChunk {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateChunk {
    model: String,
    created_at: String,
    response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize)]
struct OllamaModelInfo {
    name: String,
    model: String,
    size: u64,
    digest: String,
    modified_at: String,
    details: OllamaModelDetails,
}

#[derive(Debug, Serialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize)]
struct OllamaPsResponse {
    models: Vec<OllamaRunningModelInfo>,
}

#[derive(Debug, Serialize)]
struct OllamaRunningModelInfo {
    name: String,
    model: String,
    /// When this model will be automatically unloaded if left idle —
    /// `None` (serialized as JSON `null`) when its `keep_alive` is
    /// "forever" (see `RunningModel::keep_alive`); real Ollama instead
    /// sends the sentinel zero time `"0001-01-01T00:00:00Z"` for that
    /// case, which every Ollama-API client already treats as "far future
    /// timestamp, not a real deadline" rather than parsing it — `null` is
    /// less surprising to a client not expecting Go's zero-value
    /// convention, and is exactly how `handle_show`/etc. already spell
    /// "not applicable" elsewhere in this module.
    expires_at: Option<String>,
    // Real Ollama /api/ps shape ends here (see api.ProcessModelResponse in
    // ollama/api/types.go); the fields below are llmman-specific additions
    // for `llmman ps` — safe for any other Ollama-API client to ignore.
    digest: String,
    size: u64,
    size_vram: u64,
    pid: Option<u32>,
    port: u16,
    processor: String,
    context_length: Option<u64>,
    started_at: String,
}

#[derive(Debug, Deserialize)]
struct OllamaShowRequest {
    model: String,
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaShowResponse {
    model_info: serde_json::Value,
    details: OllamaModelDetails,
}

#[derive(Debug, Deserialize)]
struct OllamaDeleteRequest {
    model: String,
    name: Option<String>,
}

// ---------------------------------------------------------------------------
// Anthropic API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    // Anthropic's real API accepts `system` as either a plain string or an
    // array of content blocks (the same shape as message content) — real
    // Claude Code always sends the array form, carrying its system prompt
    // as one or more {"type":"text","text":"..."} blocks, so a bare
    // Option<String> here 422s on every real request.
    system: Option<AnthropicContent>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    type_: String,
    text: Option<String>,
}

impl AnthropicContent {
    fn as_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.type_ == "text")
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI types (internal proxy use)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, PartialEq, Default)]
struct OAIMessage {
    role: String,
    /// A plain JSON string for an ordinary text message, or an array of
    /// OpenAI "content part" objects (`{"type":"text",...}` /
    /// `{"type":"image_url",...}`) for a multimodal one — see
    /// `ollama_message_to_oai`. `serde_json::Value` rather than a typed
    /// enum since content parts are only ever built here, never parsed
    /// back out.
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OAIToolCall>>,
    /// Only meaningful on a `role: "tool"` message: which tool this is a
    /// result for. Ollama's own wire format has no `tool_call_id`
    /// equivalent (see `OllamaMessage::tool_name`'s doc comment) — set
    /// from that field on a best-effort basis so name-matching chat
    /// templates still work, even though a strict id-matching one won't.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl OAIMessage {
    /// Build a plain text message — the common case, and the only shape
    /// needed anywhere images/tool-calls/tool-results aren't in play
    /// (`/api/generate`, the Anthropic Messages API).
    fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            tool_calls: None,
            name: None,
        }
    }
}

/// OpenAI's assistant-message tool-call shape (distinct from
/// [`OllamaToolCall`]): a top-level `id`/`type`, and `function.arguments`
/// as a JSON-*encoded string* rather than a decoded object.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct OAIToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: &'static str,
    function: OAIToolCallFunction,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OAIToolCallFunction {
    name: String,
    arguments: String,
}

/// Converts one incoming [`OllamaMessage`] into the OpenAI-shaped message
/// llama-server expects, handling the three cases Ollama's own format
/// supports that a plain `{role, content}` pair can't:
///
/// - `images`: turned into `image_url` content parts alongside a leading
///   `text` part, per the OpenAI vision message convention llama-server's
///   multimodal chat template expects. A bare base64 string (Ollama's own
///   format — no `data:` prefix) is wrapped in a `data:image/*;base64,`
///   URI; a value that already looks like a data URI is passed through
///   unchanged.
/// - `tool_calls`: carried onto an assistant message so multi-turn
///   tool-calling history round-trips; Ollama's `arguments` (already a
///   decoded JSON value) is re-encoded to the JSON *string* OpenAI's
///   schema requires.
/// - `tool_name` on a `role: "tool"` message: mapped to `name`, the
///   closest OpenAI equivalent llama-server's chat templates read Ollama's
///   `tool_call_id` are not surfaced to `/api/chat` callers).
fn ollama_message_to_oai(m: &OllamaMessage) -> OAIMessage {
    let content = match &m.images {
        Some(images) if !images.is_empty() => {
            let mut parts = Vec::with_capacity(images.len() + 1);
            if !m.content.is_empty() {
                parts.push(serde_json::json!({ "type": "text", "text": m.content }));
            }
            for image in images {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": image_data_uri(image) }
                }));
            }
            serde_json::Value::Array(parts)
        }
        _ => serde_json::Value::String(m.content.clone()),
    };
    let tool_calls = m.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| OAIToolCall {
                // gen_id() alone is time-based and can collide when called
                // back-to-back for multiple tool calls in one message (a
                // coarse clock could return the same reading twice) — the
                // index makes each id unique within this message even
                // then.
                id: format!("{}_{i}", gen_id()),
                type_: "function",
                function: OAIToolCallFunction {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.to_string(),
                },
            })
            .collect()
    });
    OAIMessage {
        role: m.role.clone(),
        content,
        tool_calls,
        name: m.tool_name.clone(),
    }
}

/// Wraps a bare base64 image (Ollama's own `images` wire format) in a
/// `data:` URI for llama-server's OpenAI-compatible `image_url` content
/// part. `image/png` is a placeholder mime type — llama.cpp's clip
/// decoder sniffs the actual format from the decoded bytes' own magic
/// number rather than trusting this, so an arbitrary supported format
/// (JPEG, WEBP, ...) still decodes correctly despite the label. Passed
/// through unchanged if the caller already sent a full data URI (not
/// Ollama's documented format, but harmless to accept).
fn image_data_uri(base64_bytes: &str) -> String {
    if base64_bytes.starts_with("data:") {
        base64_bytes.to_string()
    } else {
        format!("data:image/png;base64,{base64_bytes}")
    }
}

#[derive(Debug, Serialize, Default)]
struct OAIChatRequest {
    model: String,
    messages: Vec<OAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // Resolved to `DEFAULT_REPEAT_PENALTY` by `post_chat` — the one
    // function every typed request (`/api/chat`, `/api/generate`, the
    // Anthropic Messages API) actually goes through to reach
    // llama-server — whenever a construction site below leaves this
    // `None`, so the outgoing request always carries an explicit value
    // instead of silently omitting the field. See
    // `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    // See think_to_chat_template_kwargs. Omitted entirely (rather than
    // sent as `null`) when the caller didn't ask to override thinking, so
    // the template's own default applies exactly as if this field never
    // existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
    /// See `OllamaChatRequest::tools` — passed straight through.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    /// See `format_to_response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

/// Ollama's actual default for `repeat_penalty`: `DefaultOptions()` in
/// ollama's `api/types.go` sets `RepeatPenalty: 1.0`, and its own
/// `docs/modelfile.mdx` PARAMETER table documents the same thing
/// ("Default: 1.0, disabled") — a previous version of this comment
/// misread that table's rightmost *example-invocation* column
/// (`repeat_penalty 1.1`) as the default and picked 1.1 here on that
/// basis. 1.0 also happens to be llama-server's own raw default, so this
/// constant now agrees with both; the only thing it still buys over
/// omitting the field is that llmman always sends an explicit value,
/// matching ollama's own behavior of always forwarding an already-
/// resolved `Options.RepeatPenalty` rather than an unset one.
///
/// This intentionally restores the repetition-loop risk this constant
/// was originally raised to 1.1 to work around: `qwen3.5:0.8b`'s
/// "thinking" mode was observed looping on the same handful of reasoning
/// sentences indefinitely at repeat_penalty=1.0, consuming the whole
/// response on invisible reasoning tokens and never emitting visible
/// content (see docker/sandboxes#5109 and PR #273). That tradeoff was
/// made deliberately here to keep llmman's default numerically identical
/// to ollama's instead of silently diverging from it — if that
/// regression resurfaces, the fix belongs in a model-specific override or
/// a different sampler parameter, not by re-diverging this constant from
/// ollama's own value.
///
/// Used as the fallback whenever a caller doesn't supply its own
/// `options.repeat_penalty` — applied in exactly two places: `post_chat`
/// (every typed request: `/api/chat`, `/api/generate`, the Anthropic
/// Messages API) and `apply_default_repeat_penalty` (the raw OpenAI-
/// passthrough generation routes: chat completions, legacy completions,
/// the Responses API).
const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Deserialize)]
struct OAIChunk {
    choices: Vec<OAIChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct OAIChunkChoice {
    delta: OAIChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAIChunkDelta {
    content: Option<String>,
    /// llama-server (Homebrew b8880) sends reasoning content in this field.
    /// The git repo uses "thinking" — accept both for forward compatibility.
    reasoning_content: Option<String>,
    thinking: Option<String>,
    /// OpenAI-style streaming tool-call deltas — see
    /// `oai_chunk_tool_call_deltas`/`ToolCallAccumulator`.
    #[serde(default)]
    tool_calls: Option<Vec<OAIToolCallDelta>>,
}

/// One fragment of one streamed tool call. Mirrors OpenAI's streaming
/// shape: `function.name` normally arrives whole in the first delta for a
/// given `index`, while `function.arguments` arrives incrementally as a
/// partial JSON string across many deltas — see `ToolCallAccumulator`.
/// (OpenAI's streaming shape also carries a top-level `id` on that first
/// delta; deliberately not deserialized here — [`OllamaToolCall`], the
/// only shape it ever needs to flow into, has no `id` field to carry it
/// to.)
#[derive(Debug, Deserialize, Default)]
struct OAIToolCallDelta {
    index: usize,
    #[serde(default)]
    function: Option<OAIToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct OAIToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulates one tool call's streamed fragments (see
/// [`OAIToolCallDelta`]) by index, across an entire `/api/chat` response —
/// `stream_ollama` keeps one `BTreeMap<usize, ToolCallAccumulator>` per
/// request and finalizes it (`finalize_tool_calls`) once the stream's
/// `done` chunk arrives.
#[derive(Default, Clone)]
struct ToolCallAccumulator {
    name: String,
    arguments: String,
}

/// Extracts this SSE payload's tool-call deltas, if any — `[]` for the
/// `[DONE]` sentinel (no JSON to parse) or any payload without a
/// `tool_calls` delta, never an error, matching `oai_chunk_to_content`'s
/// own "malformed/absent is empty, not fatal" handling.
fn oai_chunk_tool_call_deltas(payload: &str) -> Vec<OAIToolCallDelta> {
    if payload == "[DONE]" {
        return Vec::new();
    }
    serde_json::from_str::<OAIChunk>(payload)
        .ok()
        .and_then(|c| c.choices.into_iter().next())
        .and_then(|c| c.delta.tool_calls)
        .unwrap_or_default()
}

/// Folds one SSE payload's tool-call deltas into `acc`, keyed by their
/// streaming `index`. Pure bookkeeping — the actual arguments string is
/// only parsed as JSON once complete, by `finalize_tool_calls`.
fn accumulate_tool_call_deltas(
    payload: &str,
    acc: &std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
) {
    let deltas = oai_chunk_tool_call_deltas(payload);
    if deltas.is_empty() {
        return;
    }
    let mut acc = acc.borrow_mut();
    for delta in deltas {
        let entry = acc.entry(delta.index).or_default();
        if let Some(f) = delta.function {
            if let Some(name) = f.name {
                entry.name.push_str(&name);
            }
            if let Some(args) = f.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }
}

/// Turns the accumulated tool-call fragments into Ollama's own
/// `tool_calls` shape once a response is `done`. Each call's `arguments`
/// string (a JSON object, incrementally assembled — see
/// [`OAIToolCallDelta`]) is parsed back into a decoded `serde_json::Value`
/// here, since Ollama's `OllamaToolCallFunction::arguments` — unlike
/// OpenAI's — is a JSON object, not a string. An empty accumulator (no
/// tool calls made) yields `None` rather than `Some(vec![])`, so
/// `OllamaMessage`'s `tool_calls` field is omitted entirely for an
/// ordinary text response.
fn finalize_tool_calls(
    acc: &std::collections::BTreeMap<usize, ToolCallAccumulator>,
) -> Option<Vec<OllamaToolCall>> {
    if acc.is_empty() {
        return None;
    }
    Some(
        acc.values()
            .map(|c| OllamaToolCall {
                function: OllamaToolCallFunction {
                    name: c.name.clone(),
                    arguments: serde_json::from_str(&c.arguments)
                        .unwrap_or_else(|_| serde_json::json!({})),
                },
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AppError(anyhow::Error, StatusCode);

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into(), StatusCode::INTERNAL_SERVER_ERROR)
    }
}

impl AppError {
    /// Builds an `AppError` with its own status code, instead of the
    /// plain 500 every `?`/`From` conversion above produces — used by
    /// `ensure_model`'s admission-control checks (`LLMMAN_MAX_QUEUE`/
    /// `LLMMAN_MAX_LOADED_MODELS`), which need `503`.
    fn status(status: StatusCode, message: impl Into<String>) -> Self {
        Self(anyhow!(message.into()), status)
    }

    /// Wraps a rejected model reference as a 400: the reference is client
    /// input, so the client error is built right where the reference is
    /// rejected, via `.map_err(AppError::bad_request)` at each resolve
    /// site. A constructor rather than a `From<InvalidReference>` impl
    /// because the blanket `From` above already covers every
    /// `Into<anyhow::Error>` type (it would produce a 500 through `?`),
    /// and a specific impl would overlap with it.
    fn bad_request(e: crate::shortnames::InvalidReference) -> Self {
        Self(e.into(), StatusCode::BAD_REQUEST)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": format!("{:#}", self.0) });
        (self.1, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bool_true() -> bool {
    true
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn gen_id() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{secs:032x}")
}

// ---------------------------------------------------------------------------
// Idle-timeout auto-unload (`keep_alive`)
//
// Mirrors Ollama's own idle-unload scheduler (server/sched.go): every
// loaded model carries a `keep_alive` duration and a last-activity
// timestamp (see `RunningModel`); a background task (`reap_idle_models`,
// spawned once from `serve_async`) periodically unloads whichever models
// have gone unused past their own deadline. `ActivityGuard` is what keeps
// that timer from firing mid-generation.
// ---------------------------------------------------------------------------

/// Ollama's documented default `keep_alive`: an idle, unused model is
/// unloaded after 5 minutes (see ollama's docs/faq.mdx, "How do I keep a
/// model loaded in memory or make it unload immediately?"). Applies
/// whenever a request omits `keep_alive` entirely, or supplies a value
/// that fails to parse.
const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);

/// The daemon-wide `keep_alive` to fall back on: [`DEFAULT_KEEP_ALIVE`],
/// unless overridden by `LLMMAN_KEEP_ALIVE` (mirrors Ollama's own
/// `OLLAMA_KEEP_ALIVE` env var), parsed with the same syntax as the
/// per-request `keep_alive` field — see `parse_keep_alive_str`.
fn default_keep_alive() -> Option<Duration> {
    match std::env::var("LLMMAN_KEEP_ALIVE") {
        Ok(v) => parse_keep_alive_str(&v).unwrap_or(Some(DEFAULT_KEEP_ALIVE)),
        Err(_) => Some(DEFAULT_KEEP_ALIVE),
    }
}

/// Resolves a request's `keep_alive` field to how long this daemon should
/// wait, after the request finishes, before automatically unloading the
/// model. `None` means "never". Falls back to [`default_keep_alive`] both
/// when the field is absent and when present but unparseable — same as
/// Ollama's own `api.Duration` silently keeping its default on a bad
/// input rather than 400ing the whole request over it.
fn resolve_keep_alive(value: &Option<serde_json::Value>) -> Option<Duration> {
    value
        .as_ref()
        .and_then(parse_keep_alive_value)
        .unwrap_or_else(default_keep_alive)
}

/// True only when the request itself spells `keep_alive: 0` — Ollama's
/// unload sentinel, in any of the zero forms `parse_keep_alive_value`
/// accepts. Deliberately not [`resolve_keep_alive`], which falls back to
/// [`default_keep_alive`] when the field is absent: under
/// `LLMMAN_KEEP_ALIVE=0` that fallback made a message-less preload
/// naming no `keep_alive` of its own resolve to zero and answer
/// `"unload"`, so a caller asking to warm a model got it evicted
/// instead. An unparseable value stays a non-unload
/// here for the same reason it stays one in `resolve_keep_alive` — the
/// daemon default decides how long to keep it, not whether to keep it.
fn is_explicit_unload(keep_alive: &Option<serde_json::Value>) -> bool {
    keep_alive.as_ref().and_then(parse_keep_alive_value) == Some(Some(Duration::ZERO))
}

/// `None` = couldn't parse `v` as a keep_alive value at all (caller falls
/// back to the daemon default). `Some(None)` = "never unload" (a negative
/// number). `Some(Some(d))` = "unload after `d` of inactivity".
fn parse_keep_alive_value(v: &serde_json::Value) -> Option<Option<Duration>> {
    match v {
        // secs_to_keep_alive rather than a bare `Duration::from_secs_f64`
        // call: JSON itself can't spell NaN/Infinity, but a huge finite
        // literal (e.g. `1e300`) still overflows Duration's own range, and
        // `from_secs_f64` panics rather than erroring on that — see its
        // own doc comment for why this must never panic on client input.
        serde_json::Value::Number(n) => secs_to_keep_alive(n.as_f64()?),
        serde_json::Value::String(s) => parse_keep_alive_str(s),
        _ => None,
    }
}

/// Converts a parsed seconds value to a keep_alive result without ever
/// panicking, regardless of what a client sent: negative (including
/// `-inf`) means "never unload"; anything `Duration::try_from_secs_f64`
/// itself rejects — NaN, `+inf`, or a finite value too large to fit in a
/// `Duration` — is treated as unparseable (`None`, the same as malformed
/// input), not a crash. `Duration::from_secs_f64` (the panicking
/// counterpart used nowhere in this module) would abort the whole request
/// task on exactly the inputs this function exists to reject harmlessly —
/// see rust-lang's own `Duration::from_secs_f64` docs ("Panics if the
/// provided seconds is negative, overflows the internal representation of
/// Duration or is otherwise invalid").
fn secs_to_keep_alive(secs: f64) -> Option<Option<Duration>> {
    if secs < 0.0 {
        return Some(None);
    }
    Duration::try_from_secs_f64(secs).ok().map(Some)
}

/// Parses a `keep_alive` duration string: a bare number of seconds (e.g.
/// `"300"`), a negative value meaning "never unload" (e.g. `"-1"`), or a
/// sequence of `<number><unit>` pairs using the units Ollama's own docs
/// show (`h`, `m`, `s`, `ms`) — e.g. `"10m"`, `"1h30m"`. A small,
/// deliberately non-exhaustive subset of Go's `time.ParseDuration` (no
/// `ns`/`us`, no fractional-only forms beyond what `str::parse::<f64>`
/// already accepts per component) — enough for every value Ollama's own
/// documentation and SDKs actually produce.
fn parse_keep_alive_str(s: &str) -> Option<Option<Duration>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // f64's own FromStr also accepts "inf"/"infinity"/"nan" (any case) as
    // a bare number — secs_to_keep_alive (not a raw `Duration::
    // from_secs_f64`) is what keeps those from panicking instead of just
    // falling through to "unparseable" below.
    if let Ok(secs) = s.parse::<f64>() {
        return secs_to_keep_alive(secs);
    }
    if s.starts_with('-') {
        // A negative duration string (e.g. "-1m") — Ollama treats any
        // negative keep_alive as "forever" regardless of unit.
        return Some(None);
    }
    let mut total = Duration::ZERO;
    let mut rest = s;
    let mut matched_any = false;
    while !rest.is_empty() {
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        if digits_end == 0 {
            return None;
        }
        let (num_str, tail) = rest.split_at(digits_end);
        let num: f64 = num_str.parse().ok()?;
        // Order matters: "ms" must be checked before "m" alone matches
        // its leading byte.
        let (secs, tail) = if let Some(t) = tail.strip_prefix("ms") {
            (num / 1000.0, t)
        } else if let Some(t) = tail.strip_prefix('h') {
            (num * 3600.0, t)
        } else if let Some(t) = tail.strip_prefix('m') {
            (num * 60.0, t)
        } else {
            // "s" is the only suffix left; anything else (or nothing at
            // all) makes the whole component unparseable.
            (num, tail.strip_prefix('s')?)
        };
        // A component that individually overflows Duration (e.g. a huge
        // digit string like "999999999999999s"), or that overflows once
        // added to the running total (e.g. two such components back to
        // back), invalidates the whole string, same as any other
        // unparseable input — never panic on it (see
        // secs_to_keep_alive's doc comment; plain `total += ...` panics
        // on overflow the same way `Duration::from_secs_f64` does).
        let component = Duration::try_from_secs_f64(secs).ok()?;
        total = total.checked_add(component)?;
        rest = tail;
        matched_any = true;
    }
    matched_any.then_some(Some(total))
}

/// Represents `ensure_model`'s own `in_flight` claim (see its doc
/// comment), from the moment it's first made inside `ensure_model`
/// until this guard drops. While outstanding, `reap_idle_models`/
/// `LLMMAN_MAX_LOADED_MODELS` eviction will never touch this model —
/// including in the gap between `ensure_model` resolving it and a
/// caller actually starting to use it, since whichever stack frame the
/// guard is currently sitting in still drops (and releases) it even if
/// that caller's task is cancelled there. On drop it also resets the
/// idle clock and, if this request carried an explicit `keep_alive`
/// override, records it for the next idle check — mirroring Ollama's
/// own runner refcounting (llm/server.go) at a coarser granularity.
///
/// Must be moved into (captured by) whatever `Stream`/`Body` backs the
/// actual HTTP response — see `stream_ollama`, `stream_anthropic`, and
/// `proxy` — so it isn't dropped until the response has actually finished
/// being sent, not merely until the handler function that built it
/// returns.
struct ActivityGuard {
    state: AppState,
    model_key: String,
    /// `None` = leave this model's stored `keep_alive` exactly as it is
    /// (used by the OpenAI-compatible and Anthropic surfaces, which have
    /// no `keep_alive` field of their own to read an override from — see
    /// `begin_activity`'s doc comment for why overwriting it with the
    /// daemon default from those routes would be wrong). `Some(v)` sets
    /// it to `v` (`v` itself: `None` = forever, `Some(d)` = idle timeout
    /// `d`) — used by `/api/chat` and `/api/generate`, which always
    /// resolve an explicit value (a request's own `keep_alive`, or the
    /// daemon default when it's absent) via `resolve_keep_alive`.
    keep_alive: Option<Option<Duration>>,
}

impl ActivityGuard {
    /// Constructs the guard for a model `ensure_model` has just claimed
    /// (`in_flight` already incremented by the caller, under the
    /// manager lock) — never call this without having done that first.
    fn new(state: &AppState, model_key: &str) -> Self {
        Self {
            state: state.clone(),
            model_key: model_key.to_string(),
            keep_alive: None,
        }
    }
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let state = self.state.clone();
        let model_key = std::mem::take(&mut self.model_key);
        let keep_alive = self.keep_alive;
        // Drop can't be async; the update is best-effort and doesn't need
        // to happen before this function returns. tokio::spawn panics
        // outside a running Tokio runtime (e.g. this guard outliving the
        // runtime during process teardown) — Handle::try_current lets that
        // case be skipped instead of panicking mid-unwind.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut mgr = state.0.manager.lock().await;
            if let Some(m) = mgr.running.get_mut(&model_key) {
                m.in_flight = m.in_flight.saturating_sub(1);
                m.last_active = Instant::now();
                m.last_active_wall = chrono::Utc::now();
                if let Some(kx) = keep_alive {
                    m.keep_alive = kx;
                }
            }
        });
    }
}

/// Applies `keep_alive` to the model `guard` already claims (see
/// [`ActivityGuard`]), immediately (not just on drop) so it can't be
/// reaped while this request is still waiting on something upstream of
/// actually streaming a response. A `None` override never touches
/// `keep_alive` at all, here or on drop, exactly as if this request
/// hadn't happened (`last_active` is still always refreshed, both here
/// and on drop, regardless). A no-op if `guard`'s model isn't found —
/// defensive only; every caller obtains `guard` from `ensure_model`
/// immediately beforehand.
async fn begin_activity(
    mut guard: ActivityGuard,
    keep_alive: Option<Option<Duration>>,
) -> ActivityGuard {
    {
        let mut mgr = guard.state.0.manager.lock().await;
        if let Some(m) = mgr.running.get_mut(&guard.model_key) {
            m.last_active = Instant::now();
            m.last_active_wall = chrono::Utc::now();
            if let Some(kx) = keep_alive {
                m.keep_alive = kx;
            }
        }
    }
    guard.keep_alive = keep_alive;
    guard
}

/// Applies `keep_alive` to the model `guard` already claims, then
/// releases the claim immediately — used by a load-only `/api/generate`
/// request (or the CLI `--model` pre-load), which shouldn't hold it open
/// like a real generation would. `guard` itself does the actual release
/// on drop, same as always, so this stays cancellation-safe too: even a
/// task dropped mid-lock-wait here still drops `guard` and releases the
/// claim, whether or not this update ever landed.
async fn refresh_activity(guard: ActivityGuard, keep_alive: Option<Duration>) {
    let mut mgr = guard.state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(&guard.model_key) {
        m.last_active = Instant::now();
        m.last_active_wall = chrono::Utc::now();
        m.keep_alive = keep_alive;
    }
}

/// How often the idle-unload reaper (see `reap_idle_models`) wakes up to
/// check every running model's `keep_alive` deadline — independent of
/// `keep_alive` itself, this just bounds how late an expiry can be
/// noticed, not how soon.
const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// Runs forever in the background (spawned once from `serve_async`),
/// automatically unloading any model whose `keep_alive` idle deadline has
/// passed — the daemon-wide equivalent of Ollama's own scheduler
/// idle-unload. Skips any model with `keep_alive: None` ("never") or an
/// in-flight request (`in_flight > 0`) — see [`ActivityGuard`]'s doc
/// comment for why the latter matters.
async fn reap_idle_models(state: AppState) {
    let mut ticker = tokio::time::interval(IDLE_CHECK_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        reap_idle_models_once(&state).await;
    }
}

/// One pass of `reap_idle_models`'s loop body, split out so it can be
/// driven directly (without waiting on real wall-clock ticks) by
/// `reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever`.
async fn reap_idle_models_once(state: &AppState) {
    // Find-then-remove under one held lock, not two separate acquisitions:
    // a `begin_activity` could otherwise land in between (bumping
    // `in_flight` and refreshing `keep_alive`/`last_active` for a request
    // that's just starting) and this would still remove the entry out from
    // under it, killing a request that had already begun.
    let mut mgr = state.0.manager.lock().await;
    let expired: Vec<String> = mgr
        .running
        .iter()
        .filter(|(_, m)| m.in_flight == 0)
        .filter_map(|(name, m)| {
            let deadline = m.keep_alive?;
            (m.last_active.elapsed() >= deadline).then(|| name.clone())
        })
        .collect();
    for name in expired {
        eprintln!("[llmman] unloading {name}: idle past its keep_alive deadline");
        mgr.running.remove(&name);
    }
}

// ---------------------------------------------------------------------------
// Process management
// ---------------------------------------------------------------------------

fn find_free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Shared handle onto the last few lines a spawned inference backend wrote
/// to stdout/stderr — see `spawn_tail_relay`'s own doc comment for why
/// this exists and `wait_for_ready`'s use of it.
type OutputTail = Arc<StdMutex<VecDeque<String>>>;

/// How many trailing output lines `OutputTail` keeps — enough to catch a
/// one-or-two-line startup failure (a dynamic-linker error, "no such
/// file", an out-of-memory abort, ...) without holding onto an unbounded
/// amount of a chatty child's output.
const TAIL_LINES: usize = 20;

/// Relays a spawned child's piped stdout/stderr line-by-line to this
/// process's own stdout/stderr — preserving exactly what an inherited
/// (the previous default) stdio handle would have shown up as in
/// `llmman serve`'s own log (see daemon.rs's redirection of that to
/// serve.log) — while also appending each line to `tail` (bounded to the
/// last `TAIL_LINES`), so a caller that only learns of a crash after the
/// fact (see `wait_for_ready`) can still report *why*, instead of just
/// "the process exited" with the actual reason sitting only in a log file
/// the caller (an HTTP client, ultimately a chat UI) never sees.
fn spawn_tail_relay(
    reader: impl AsyncRead + Unpin + Send + 'static,
    tail: OutputTail,
    to_stderr: bool,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if to_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            if let Ok(mut buf) = tail.lock() {
                if buf.len() >= TAIL_LINES {
                    buf.pop_front();
                }
                buf.push_back(line);
            }
        }
    });
}

async fn spawn_llama_server(
    bin: &Path,
    model: &Path,
    mmproj: Option<&Path>,
    opts: crate::container::LlamaOptions<'_>,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let crate::container::LlamaOptions {
        port,
        ctx_size,
        flash_attention,
        kv_cache_type,
        context_shift,
        split_mode,
        num_parallel,
        threads,
    } = opts;
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args([
        "--model",
        model.to_str().context("non-UTF-8 model path")?,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    // See ModelPath::mmproj's doc comment — enables llama-server to
    // actually act on `images` (vision) and serve
    // `/v1/audio/transcriptions` (audio) instead of silently ignoring
    // both.
    if let Some(mmproj) = mmproj {
        cmd.args([
            "--mmproj",
            mmproj.to_str().context("non-UTF-8 mmproj path")?,
        ]);
    }
    // `ctx_size` is already the effective value (see
    // context_length_from_env); `None` leaves --ctx-size unset, falling
    // back to n_ctx_train.
    if let Some(n) = ctx_size {
        cmd.args(["--ctx-size", &n.to_string()]);
    }
    // See flash_attention_from_env's doc comment; `None` leaves
    // --flash-attn unset, falling back to llama-server's own `auto`.
    if let Some(mode) = flash_attention {
        cmd.args(["--flash-attn", mode]);
    }
    // See kv_cache_type_from_env's doc comment; `None` leaves
    // --cache-type-k/-v unset, falling back to llama-server's own `f16`.
    if let Some(t) = kv_cache_type {
        cmd.args(["--cache-type-k", t, "--cache-type-v", t]);
    }
    // See context_shift_from_env's doc comment.
    cmd.arg(if context_shift {
        "--context-shift"
    } else {
        "--no-context-shift"
    });
    // See sched_spread_from_env's doc comment; `None` leaves
    // --split-mode unset, falling back to llama-server's own `layer`.
    if let Some(mode) = split_mode {
        cmd.args(["--split-mode", mode]);
    }
    // See num_parallel_from_env's doc comment.
    if let Some(n) = num_parallel {
        cmd.args(["--parallel", &n.to_string()]);
    }
    // See threads_from_env_or_host's doc comment; `None` leaves
    // --threads unset, falling back to llama-server's own autodetection.
    if let Some(n) = threads {
        cmd.args(["--threads", &n.to_string()]);
    }
    // See GPU_VISIBLE_DEVICE_VARS's own doc comment — already inherited
    // by default, forwarded explicitly here for clarity.
    for var in GPU_VISIBLE_DEVICE_VARS {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    // See LLAMA_CPP_ENV_PASSTHROUGH_VARS's doc comment.
    for var in LLAMA_CPP_ENV_PASSTHROUGH_VARS {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    crate::debug_log!("spawning {}: {:?}", bin.display(), cmd);
    // Piped (not inherited) so a startup crash's own explanation — e.g. a
    // dynamic linker's "error while loading shared libraries" — can be
    // captured into `tail` and surfaced by `wait_for_ready`, not just
    // dropped into a log file nobody making the request ever sees. See
    // `spawn_tail_relay`'s own doc comment for how this keeps showing up
    // in that log too.
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn llama-server from {}", bin.display()))?;

    let tail: OutputTail = Arc::new(StdMutex::new(VecDeque::with_capacity(TAIL_LINES)));
    if let Some(stdout) = child.stdout.take() {
        spawn_tail_relay(stdout, tail.clone(), false);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_tail_relay(stderr, tail.clone(), true);
    }
    Ok((child, tail))
}

async fn spawn_vllm_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
) -> anyhow::Result<tokio::process::Child> {
    let vllm = which_binary("vllm")?;
    let mut cmd = tokio::process::Command::new(&vllm);
    cmd.args([
        "serve",
        model_dir.to_str().context("non-UTF-8 model path")?,
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
        // Register the model under the same name used in API requests so
        // {"model": "<ref>"} is accepted by vllm's OpenAI-compatible API.
        "--served-model-name",
        model_name,
    ]);
    // Own process group so ModelProcess's Drop impl can kill vllm's whole
    // worker tree, not just this one pid, without also killing ourselves.
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn vllm from {}", vllm.display()))
}

/// Spawns `mlx_lm.server` (installed on `PATH` by `pip install mlx-lm`
/// <https://github.com/ml-explore/mlx-lm>) — Apple Silicon's own
/// Metal-accelerated alternative to `vllm` for a
/// [`ModelPath::SafeTensors`] directory, picked instead of it by
/// [`use_mlx_for_safetensors`].
///
/// Deliberately does *not* pass `mlx_lm.server`'s own `--model` flag,
/// even though that's its documented way to preload one: confirmed
/// against its own `server.py` that doing so loads the model in a
/// background thread (`ResponseGenerator.__init__`'s
/// `Thread(target=self._generate)`) with no `try`/`except` anywhere
/// around that particular load — a bad model directory would silently
/// kill only that one thread, not this process, while its
/// `ThreadingHTTPServer` (started right alongside it, not after) keeps
/// right on reporting `/health` as ready regardless. `wait_for_ready`
/// would then report this backend ready, and every real request queued
/// behind that dead thread would hang forever instead of ever seeing an
/// error.
///
/// Loading instead happens on the *first real request* — every caller
/// sends this model's actual absolute directory path (not its
/// human-readable reference) as that request's own `"model"` field, via
/// [`backend_wire_model`] — which goes through `ModelProvider.load`'s
/// own `try`/`except` in the request-handling path instead, and so does
/// report a real error back to that request on a bad model directory.
async fn spawn_mlx_server(port: u16) -> anyhow::Result<tokio::process::Child> {
    let mlx = which_binary("mlx_lm.server")?;
    let mut cmd = tokio::process::Command::new(&mlx);
    cmd.args(["--port", &port.to_string(), "--host", "127.0.0.1"]);
    cmd.kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn mlx_lm.server from {}", mlx.display()))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        // On Windows the executable must carry the .exe suffix.
        #[cfg(windows)]
        let candidate = dir.join(format!("{name}.exe"));
        #[cfg(not(windows))]
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn which_binary(name: &str) -> anyhow::Result<PathBuf> {
    find_on_path(name).ok_or_else(|| anyhow::anyhow!("{name} not found on PATH"))
}

/// [`wait_for_ready`]'s default deadline — longer than Ollama's own 5m
/// default since vllm can take several minutes to load a large model.
/// Overridable via `LLMMAN_LOAD_TIMEOUT`.
const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// `None` = wait forever, from an `LLMMAN_LOAD_TIMEOUT` of 0 or
/// negative (mirrors Ollama's `OLLAMA_LOAD_TIMEOUT`).
fn load_timeout_from_env() -> Option<Duration> {
    match std::env::var("LLMMAN_LOAD_TIMEOUT") {
        Ok(v) => parse_load_timeout(&v).unwrap_or(Some(DEFAULT_LOAD_TIMEOUT)),
        Err(_) => Some(DEFAULT_LOAD_TIMEOUT),
    }
}

/// Reuses [`parse_keep_alive_str`]'s duration syntax, but unlike
/// keep_alive, a zero value also means "forever" here (matches
/// `OLLAMA_LOAD_TIMEOUT`'s documented behavior). Unlike a plain
/// delegation, a leading `-` is only treated as "forever" once the
/// magnitude after it actually parses as a duration —
/// `parse_keep_alive_str`'s own dash-prefix shortcut accepts any
/// `"-..."` unconditionally, which would otherwise make a typo like
/// `LLMMAN_LOAD_TIMEOUT=-garbage` disable the timeout forever instead of
/// falling back to the documented default.
fn parse_load_timeout(value: &str) -> Option<Option<Duration>> {
    let trimmed = value.trim();
    if let Some(magnitude) = trimmed.strip_prefix('-') {
        return parse_keep_alive_str(magnitude).map(|_| None);
    }
    Some(match parse_keep_alive_str(trimmed)? {
        Some(d) if d.is_zero() => None,
        other => other,
    })
}

/// Poll interval between `/health` checks in [`wait_for_ready`].
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Polls `process`'s `/health` endpoint until ready, bailing out early
/// if `process` itself exits first (so a crash-on-startup doesn't hang
/// the caller for the whole deadline). `stderr_tail`, when given (local
/// llama-server only), includes the crash reason in the error.
async fn wait_for_ready(
    client: &Client,
    port: u16,
    process: &mut ModelProcess,
    stderr_tail: Option<&OutputTail>,
) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    // `None` = wait forever (LLMMAN_LOAD_TIMEOUT of 0 or negative, or a
    // value so large that adding it to `Instant::now()` would overflow
    // — `checked_add`, not `+`, so a huge-but-validly-parsed timeout
    // can't panic the request task).
    let load_timeout = load_timeout_from_env();
    let deadline = load_timeout.and_then(|d| Instant::now().checked_add(d));
    loop {
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if remaining.is_some_and(|r| r.is_zero()) {
            return Err(anyhow!(
                "inference server on port {port} did not become ready within {:?}",
                load_timeout.unwrap_or_default()
            ));
        }
        if !process.is_alive() {
            let detail = stderr_tail.and_then(|t| {
                let lines = t.lock().ok()?;
                (!lines.is_empty()).then(|| lines.iter().cloned().collect::<Vec<_>>().join(" | "))
            });
            return Err(match detail {
                Some(detail) => anyhow!(
                    "inference server on port {port} exited before becoming ready: {detail}"
                ),
                None => anyhow!("inference server on port {port} exited before becoming ready"),
            });
        }
        // Bound the request by POLL_INTERVAL, not the full remaining
        // deadline — otherwise a /health that connects but then stalls
        // could occupy up to the whole deadline (or forever, if unset)
        // without rechecking process liveness or the deadline.
        let bound = remaining.map_or(POLL_INTERVAL, |r| r.min(POLL_INTERVAL));
        let attempt_start = Instant::now();
        if let Ok(resp) = client.get(&url).timeout(bound).send().await {
            // llama-server/vllm: 200 once loaded. mlx_lm.server: 200 as
            // soon as its listener is up, not once a model is loaded
            // (see spawn_mlx_server) — an accepted, documented gap for
            // that one engine only.
            if resp.status().is_success() {
                return Ok(());
            }
        }
        // Only the unused remainder of one POLL_INTERVAL — not another
        // full one on top of whatever the attempt above already took —
        // so a consistently-stalling /health still gets rechecked every
        // POLL_INTERVAL, not every 2x that. Still never past the
        // overall deadline either.
        let sleep_for = POLL_INTERVAL.saturating_sub(attempt_start.elapsed());
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        sleep(remaining.map_or(sleep_for, |r| sleep_for.min(r))).await;
    }
}

/// Per-model registry of locks serializing every call into the Go shim's
/// `llmman_pull`/`llmman_push` (see `crate::ffi::pull`/`push`) for a given
/// model reference — replacing what used to be one `PULL_LOCK` mutex
/// shared by every model in the process.
///
/// go-shim/progress_state.go's `progressState` used to track only one
/// transfer at a time process-wide; it's now keyed per model reference
/// (see that file's own doc comment), so two *different* models pulling
/// or pushing at once no longer interleave or corrupt each other's
/// progress numbers the way they would have under the old global lock —
/// only concurrent operations on the *same* model reference still need to
/// be serialized. Three call sites can independently decide "not in
/// store, pull it" for the same model at once (this fallback in
/// `ensure_model`, `handle_pull`, and — since `launch` started calling
/// `daemon::ensure_model_pulled` itself — a concurrent client's own
/// explicit `/api/pull`), and without a per-model lock, two such calls
/// racing for the *same* model still means a redundant full download of
/// the same multi-GB blob. See also go-shim's `blobFetchGroup`
/// (shared_oci.go), which separately deduplicates two *different* models'
/// concurrent pulls that happen to share an underlying blob — a case this
/// per-model registry can't catch on its own since it only locks by
/// reference, not by content digest.
static MODEL_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Separate from `MODEL_LOCKS`: `ensure_model` holds a load lock across a
/// call that itself takes a `MODEL_LOCKS` lock (`pull_serialized`), so
/// sharing one map would re-enter the same non-reentrant mutex and deadlock.
static LOAD_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

fn keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = registry.lock().unwrap();
    locks
        .entry(key.to_owned())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Removes `key` once nothing but `registry` itself still holds a clone —
/// call after dropping your own clone.
fn release_keyed_lock(
    registry: &StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    key: &str,
) {
    let mut locks = registry.lock().unwrap();
    if let Some(arc) = locks.get(key) {
        if Arc::strong_count(arc) <= 1 {
            locks.remove(key);
        }
    }
}

/// Returns (creating if absent) the lock serializing pull/push calls for
/// `model`. See `keyed_lock`.
fn model_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&MODEL_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_model_lock(model: &str) {
    release_keyed_lock(&MODEL_LOCKS, model)
}

/// Serializes `ensure_model`'s load phase (pull-if-missing, spawn,
/// wait-until-ready) per model, instead of `state.0.manager`.
fn load_lock(model: &str) -> Arc<tokio::sync::Mutex<()>> {
    keyed_lock(&LOAD_LOCKS, model)
}

/// See `release_keyed_lock`.
fn release_load_lock(model: &str) {
    release_keyed_lock(&LOAD_LOCKS, model)
}

/// RAII handle for `load_lock`: releases the mutex and the registry entry
/// in `Drop`, so cleanup still runs if the holding task is cancelled
/// (e.g. an axum request future dropped mid-`.await`) rather than only on
/// a normal return — code placed after an `.await` doesn't run when the
/// future holding it is dropped instead of polled to completion.
struct LoadLockGuard {
    model: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for LoadLockGuard {
    fn drop(&mut self) {
        self.guard.take(); // drop the Mutex guard (and its Arc clone) first
        release_load_lock(&self.model);
    }
}

async fn acquire_load_lock(model: &str) -> LoadLockGuard {
    let guard = load_lock(model).lock_owned().await;
    LoadLockGuard {
        model: model.to_owned(),
        guard: Some(guard),
    }
}

/// Pulls `model` into `layout_dir` if (still, after acquiring model's own
/// lock) missing from the local store — shared by `ensure_model`'s
/// fallback and `handle_pull` so both funnel through the same
/// single-flight check instead of each deciding "not present" from a
/// snapshot taken before waiting on the lock, then redundantly re-pulling
/// once it's their turn.
///
/// Must be called from a blocking context (`spawn_blocking`): blocks the
/// current thread on model's lock, not just this async task.
///
/// A HuggingFace reference pulls entirely in Rust (`crate::hf::pull`,
/// see its own doc comment for why) straight into the local OCI layout,
/// as do the `ms://`/`ngc://`/`s3://`/`gs://`/local-path sources
/// (`crate::sources`); only an actual OCI registry still goes through
/// the Go shim.
fn pull_serialized(store_path: &std::path::Path, model: &str) -> anyhow::Result<()> {
    let lock = model_lock(model);
    let result = (|| {
        let _guard = lock.blocking_lock();
        if OciStore::open(store_path)
            .and_then(|s| s.find(model))
            .is_ok()
        {
            return Ok(()); // someone else already pulled it while we waited
        }
        let layout_dir = store_path
            .to_str()
            .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
        // Safe from a spawn_blocking'd OS thread: this reuses the
        // current (already-running) tokio runtime rather than trying to
        // start a second, nested one.
        tokio::runtime::Handle::current().block_on(async {
            match crate::hf::classify(model).await {
                crate::hf::ClassifiedRef::Hf(reference) => {
                    crate::hf::pull::pull(&reference, store_path, model).await
                }
                crate::hf::ClassifiedRef::Source(reference) => {
                    crate::sources::pull(&reference, store_path, model).await
                }
                crate::hf::ClassifiedRef::Other(normalized) => {
                    crate::ffi::pull(&normalized, layout_dir)
                }
            }
        })
    })();
    drop(lock);
    release_model_lock(model);
    result
}

/// Resolve a user-supplied model ref to the canonical reference stored in
/// the OCI index (e.g. "hf.co/repo" → "hf.co/repo:latest"). No-ops before
/// the model is pulled — `ensure_model` also runs `default_tag` up front
/// to cover that gap.
fn canonical_ref(store_path: &std::path::Path, model_ref: &str) -> String {
    let Ok(store) = crate::storage::OciStore::open(store_path) else {
        return model_ref.to_owned();
    };
    let Ok(desc) = store.find(model_ref) else {
        return model_ref.to_owned();
    };
    desc.annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
        .cloned()
        .unwrap_or_else(|| model_ref.to_owned())
}

/// The load-lock key for `model`: one string for every spelling of the
/// same model, independent of what the store holds. `ensure_model` and
/// `unload_model` both lock on it, so an unload arriving during a first
/// load waits for that load rather than passing it.
///
/// Deliberately stops short of `canonical_ref`. That step reads the store,
/// and the store changes underneath a first load: before the pull it has
/// nothing and returns the tagged spelling, after it it returns whatever
/// reference the pull recorded. A lock key that moved with it would let a
/// caller resolving in that window take a different lock from the loader
/// still holding the old one. `default_tag` alone is stable, and already
/// folds the tagless and `:latest` spellings together, which is all the
/// lock needs.
///
/// A provider-routed reference comes back untouched, as `ensure_model`
/// returns it before any of this applies. Nothing observable depends on
/// that today, since a remote target never enters `running`.
fn load_identity(model: &str) -> Result<String, crate::shortnames::InvalidReference> {
    if crate::providers::is_remote_ref(model) {
        return Ok(model.to_string());
    }
    let resolved = crate::shortnames::resolve_ollama_api(model)?;
    Ok(crate::storage::default_tag(&resolved))
}

/// Drops `model` from `running`, or reports a 404 when llmman has no such
/// model at all.
///
/// Ollama answers an unload with a plain success for a model it holds but
/// has not loaded, and 404s only for one it has never pulled (checked
/// against ollama 0.32.6). The local store is what separates those two
/// cases here. A model removed from the store while still loaded stays
/// unloadable, since the `running` entry is authoritative and is consulted
/// first.
async fn unload_model(state: &AppState, model: &str) -> Result<(), AppError> {
    let lock_key = load_identity(model).map_err(AppError::bad_request)?;
    let _guard = acquire_load_lock(&lock_key).await;
    // Only now, with any load of this model excluded: the running key is
    // the store's spelling, which a load in flight may just have changed.
    let canonical = if crate::providers::is_remote_ref(model) {
        lock_key
    } else {
        canonical_ref(&state.0.store_path, &lock_key)
    };
    if state
        .0
        .manager
        .lock()
        .await
        .running
        .remove(&canonical)
        .is_some()
    {
        return Ok(());
    }
    // Nothing was loaded under that key. A provider-routed model never is,
    // and is absent from the store by definition, so naming one is not the
    // 404 case.
    if crate::providers::is_remote_ref(model) {
        return Ok(());
    }
    let store = OciStore::open(&state.0.store_path)?;
    store
        .find(&canonical)
        .map_err(|_| AppError(anyhow!("model '{model}' not found"), StatusCode::NOT_FOUND))?;
    Ok(())
}

/// Is `model_ref` already running and alive? See `ModelProcess::is_alive`.
/// If so, claims it (`in_flight += 1`, under the same lock as the
/// liveness check) and returns the same [`ActivityGuard`] `ensure_model`
/// itself would, so eviction can never see this model as idle, and the
/// claim always has an owner, from this moment until the caller's own
/// `begin_activity`/`refresh_activity` takes over.
async fn check_running(state: &AppState, model_ref: &str) -> Option<(u16, ActivityGuard)> {
    let mut mgr = state.0.manager.lock().await;
    if let Some(m) = mgr.running.get_mut(model_ref) {
        if m.process.is_alive() {
            m.in_flight += 1;
            return Some((m.port, ActivityGuard::new(state, model_ref)));
        }
        eprintln!(
            "[llmman] {model_ref} was marked running on port {} but its process has exited — reloading",
            m.port
        );
        mgr.running.remove(model_ref);
    }
    None
}

/// Evicts every currently-running model other than `model_ref` that
/// isn't actively serving a request, waiting for each to fully exit (see
/// `ModelProcess::stop_and_wait`'s own doc comment) so its VRAM is
/// actually freed before returning — mirrors Ollama's own OOM fallback of
/// evicting every other loaded model and retrying once
/// (`server/sched.go`). Skips any model with `in_flight > 0`, same as
/// `reap_idle_models_once`'s own safety check — freeing memory for a new
/// load should never mean killing a request that had already begun.
/// Returns `true` if anything was evicted, so a caller only gained by
/// this knows whether retrying is actually worth it.
async fn evict_other_models(state: &AppState, model_ref: &str) -> bool {
    let mut mgr = state.0.manager.lock().await;
    let other_keys: Vec<String> = mgr
        .running
        .iter()
        .filter(|(k, m)| k.as_str() != model_ref && m.in_flight == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut evicted: Vec<(String, RunningModel)> = Vec::with_capacity(other_keys.len());
    for key in other_keys {
        if let Some(running) = mgr.running.remove(&key) {
            evicted.push((key, running));
        }
    }
    drop(mgr); // release the lock before the (possibly slow) stops below
    let any = !evicted.is_empty();
    for (name, mut running) in evicted {
        eprintln!("[llmman] evicting {name} to free memory before retrying {model_ref}");
        running.process.stop_and_wait().await;
    }
    any
}

/// Releases a `pending_loads` reservation on drop — see
/// [`enforce_max_loaded_models`]. Mirrors [`ActivityGuard`]'s
/// Drop-can't-be-async workaround.
struct PendingLoadGuard {
    state: AppState,
    armed: bool,
}

impl std::fmt::Debug for PendingLoadGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingLoadGuard")
            .field("armed", &self.armed)
            .finish()
    }
}

impl Drop for PendingLoadGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let state = self.state.clone();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut mgr = state.0.manager.lock().await;
            mgr.pending_loads = mgr.pending_loads.saturating_sub(1);
        });
    }
}

/// Reserves a loaded-model slot for a brand-new load, enforcing
/// `LLMMAN_MAX_LOADED_MODELS` (`0` = unbounded). Counts
/// `running.len() + pending_loads`, holding the reservation until the
/// caller's load finishes — closes a race where two concurrent loads of
/// different models could both pass a plain `running.len()` check.
///
/// At the cap, evicts the least-recently-active idle model — reserving
/// this caller's own slot in the same locked step as removing it, so a
/// concurrent caller can't steal that room while the eviction's
/// `stop_and_wait` is still in flight. If the cap is only exceeded by
/// other reservations (not real running models), waits for one to
/// resolve instead of evicting a fine model. Returns 503 if nothing can
/// be freed.
async fn enforce_max_loaded_models(
    state: &AppState,
    max_loaded: usize,
) -> Result<PendingLoadGuard, AppError> {
    if max_loaded == 0 {
        return Ok(PendingLoadGuard {
            state: state.clone(),
            armed: false,
        });
    }
    loop {
        let mut mgr = state.0.manager.lock().await;
        if mgr.running.len() + mgr.pending_loads < max_loaded {
            mgr.pending_loads += 1;
            return Ok(PendingLoadGuard {
                state: state.clone(),
                armed: true,
            });
        }
        if mgr.running.len() < max_loaded {
            // Capacity is only used up by other loads' reservations,
            // not real running models — wait for one to resolve rather
            // than evicting a model that's still fine.
            drop(mgr);
            sleep(POLL_INTERVAL).await;
            continue;
        }
        let victim = mgr
            .running
            .iter()
            .filter(|(_, m)| m.in_flight == 0)
            .min_by_key(|(_, m)| m.last_active)
            .map(|(k, _)| k.clone());
        let Some(victim) = victim else {
            return Err(AppError::status(
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "max loaded models ({max_loaded}) reached, and every loaded model is busy — try again"
                ),
            ));
        };
        // Reserve this caller's own slot atomically with removing the
        // victim, under the same lock — otherwise a concurrent caller
        // could see the room this eviction is about to free and steal
        // it while `stop_and_wait` below is still in flight, briefly
        // running one backend over the configured cap.
        mgr.pending_loads += 1;
        let mut running = mgr
            .running
            .remove(&victim)
            .expect("victim key was just looked up under this same lock");
        drop(mgr); // release the lock before the (possibly slow) stop below
        eprintln!(
            "[llmman] evicting {victim} to free a loaded-model slot (LLMMAN_MAX_LOADED_MODELS={max_loaded})"
        );
        running.process.stop_and_wait().await;
        return Ok(PendingLoadGuard {
            state: state.clone(),
            armed: true,
        });
    }
}

/// Ollama's own `ErrMaxQueue` message text (`server/sched.go`), reused
/// verbatim so clients matching on it see the same thing from llmman.
const MAX_QUEUE_ERROR: &str = "server busy, please try again.  maximum pending requests exceeded";

/// How many callers are currently past `ensure_model`'s own already-
/// loaded fast path at once — admission control for `LLMMAN_MAX_QUEUE`,
/// mirroring Ollama's `pendingReqCh`. Released the moment `ensure_model`
/// returns (same point Ollama's own channel slot frees, not once
/// generation finishes).
static PENDING_REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// RAII admission guard for whichever counter admitted it (real callers
/// always get [`PENDING_REQUESTS`] via [`try_admit`]; tests can use
/// their own dedicated `static`, via [`try_admit_against`], to stay
/// isolated from other parallel tests).
struct QueueGuard(&'static std::sync::atomic::AtomicUsize);

impl Drop for QueueGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Admits one more caller past `max_queue` against `counter`, or rejects
/// with a 503 carrying [`MAX_QUEUE_ERROR`]. `0` is treated as `1`: it
/// matches Ollama's own `make(chan T, 0)` unbuffered `pendingReqCh`,
/// which still hands a request directly to its always-listening
/// consumer goroutine rather than rejecting every single one outright
/// — a one-in-flight-at-a-time cap is the closest llmman gets to that
/// same direct handoff, having no consumer-goroutine equivalent of its
/// own. Not "unbounded" either way. `fetch_update` (not a plain
/// increment-then-check) so rejected callers never inflate the counter.
fn try_admit_against(
    counter: &'static std::sync::atomic::AtomicUsize,
    max_queue: usize,
) -> Result<QueueGuard, AppError> {
    let cap = max_queue.max(1);
    let admitted = counter
        .fetch_update(
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
            |n| (n < cap).then_some(n + 1),
        )
        .is_ok();
    if admitted {
        Ok(QueueGuard(counter))
    } else {
        Err(AppError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            MAX_QUEUE_ERROR,
        ))
    }
}

fn try_admit(max_queue: usize) -> Result<QueueGuard, AppError> {
    try_admit_against(&PENDING_REQUESTS, max_queue)
}

/// If `model_ref` would be served by `Engine::Mlx` were it loaded right
/// now, returns its canonical name (see `ensure_model`'s own doc
/// comment on why that can differ from the caller's own input) —
/// *without* spawning any backend process or loading any weights.
/// Checks the already-running case first (a cheap map lookup); if it
/// isn't running, resolves it to a `ModelPath` — extracting/locating
/// its files on disk if it's already in the local store, but never
/// spawning a process — and applies the exact same
/// `ModelPath::SafeTensors` + `use_mlx_for_safetensors()` rule
/// `ensure_model` itself uses to pick an engine.
///
/// Returns `None` — "don't reject early", not "definitely not mlx" —
/// for a model that isn't in the local store at all yet (nothing to
/// resolve without also pulling it first, which this deliberately
/// never does) or that fails to resolve for any other reason:
/// `ensure_model` is still the right place to actually pull, load, and
/// (if it turns out to be `Engine::Mlx` after all) reject that one
/// first real request — this is only a cheap pre-check for the
/// overwhelmingly common repeat-request case, not a full substitute
/// for it.
///
/// Used only by `proxy_openai_passthrough`'s own `/v1/embeddings`
/// guard, so an already-pulled (or already-loaded) MLX-served model
/// doesn't pay for a full `mlx_lm.server` spawn and weights load on
/// every single embeddings request that could never succeed there
/// anyway — only ever the very first one, against a model that isn't
/// locally resolvable at all yet.
async fn would_use_mlx(state: &AppState, model_ref: &str) -> Option<String> {
    // An invalid reference is never locally resolvable, so treat it like any
    // other unresolvable case: return None and let ensure_model reject it.
    let model_ref = crate::shortnames::resolve_ollama_api(model_ref).ok()?;
    let model_ref = crate::storage::default_tag(&model_ref);
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);

    {
        let mgr = state.0.manager.lock().await;
        if let Some(running) = mgr.running.get(&model_ref) {
            return matches!(running.process, ModelProcess::Local(Engine::Mlx, _, _))
                .then(|| model_ref.clone());
        }
    }

    if !use_mlx_for_safetensors() {
        return None;
    }
    let store_path = state.0.store_path.clone();
    let cache_path = state.0.cache_path.clone();
    let lookup_ref = model_ref.clone();
    let is_safetensors = tokio::task::spawn_blocking(move || {
        resolve_model(&store_path, &cache_path, &lookup_ref)
            .map(|p| matches!(p, ModelPath::SafeTensors(_)))
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false);
    is_safetensors.then_some(model_ref)
}

// ---------------------------------------------------------------------------
// Request target: a local backend, or a remote provider
// ---------------------------------------------------------------------------

/// Where a resolved request is actually sent.
///
/// Until provider routing existed this was a bare `u16`: every backend
/// was a `llama-server`/vllm/mlx child on loopback, so a port was the
/// whole of "where does this go". [`Target::Remote`] is the same idea for
/// a request that leaves the machine — see [`crate::providers`] for which
/// providers qualify.
///
/// Requests are *routed* through, never redirected away from, this
/// daemon: a provider-backed integration still talks to `llmman serve`
/// exactly as a locally-served one does, and every surface, keep-alive
/// guard and model-name rewrite below behaves the same either way.
#[derive(Clone, Debug)]
enum Target {
    /// A locally spawned backend listening on loopback.
    Local(u16),
    /// A remote provider's OpenAI-compatible API.
    Remote(Arc<RemoteTarget>),
}

/// Everything needed to forward one request to a remote provider,
/// resolved once by [`resolve_remote_target`].
///
/// `Debug` is hand-written rather than derived so that the key cannot
/// reach a log line or an `anyhow` context chain by someone later
/// formatting a `Target`.
struct RemoteTarget {
    /// models.dev provider id, for diagnostics.
    provider: String,
    /// OpenAI-compatible base URL, without a trailing slash.
    base_url: String,
    /// The model id as the *provider* knows it — i.e. the incoming
    /// reference with its [`crate::providers::REMOTE_PREFIX`] and
    /// provider segment stripped back off.
    model: String,
    /// Bearer token for this request. See [`resolve_remote_target`] for
    /// where it comes from.
    api_key: String,
}

impl std::fmt::Debug for RemoteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteTarget")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl Target {
    /// The absolute URL an OpenAI route maps to for this target.
    ///
    /// `route` is llmman's own internal path (`/v1/chat/completions`);
    /// [`crate::providers::rebase_url`] re-bases it onto a remote
    /// provider's own published version segment, which is not always
    /// `/v1`.
    fn url(&self, route: &str) -> String {
        match self {
            Self::Local(port) => format!("http://127.0.0.1:{port}{route}"),
            Self::Remote(remote) => crate::providers::rebase_url(&remote.base_url, route),
        }
    }

    /// Attaches this target's credentials to an outgoing request.
    ///
    /// A no-op for [`Target::Local`]: a loopback `llama-server` has no
    /// auth, which is why nothing below ever forwarded the client's own
    /// `Authorization` header upstream — and must keep not forwarding it,
    /// so a key meant for one provider can never be relayed to another.
    fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Self::Local(_) => req,
            Self::Remote(remote) => req.bearer_auth(&remote.api_key),
        }
    }

    fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    /// Names this target for an error message. "inference backend" is
    /// what every failure here said before providers existed, and is
    /// still right for a local one; naming the provider is the whole
    /// difference between "something failed" and "your OpenRouter key is
    /// wrong". Never includes the key.
    fn describe(&self) -> String {
        match self {
            Self::Local(_) => "inference backend".to_string(),
            Self::Remote(remote) => format!("provider {}", remote.provider),
        }
    }
}

/// The one route every typed request in this daemon is sent to: the
/// Ollama and Anthropic surfaces are both translated into an OpenAI chat
/// completion first (see `stream_ollama` / `handle_anthropic_messages`),
/// so `post_chat` never needs any other.
const CHAT_COMPLETIONS_ROUTE: &str = "/v1/chat/completions";

/// Extracts the caller's own API key from a request, in either spelling
/// the surfaces below accept: `Authorization: Bearer <key>` (OpenAI) or
/// `x-api-key: <key>` (Anthropic).
///
/// This is what lets provider routing work against a daemon that is
/// *already running* — the common case, since `daemon::ensure_server`
/// reuses a live one, and a daemon started before the user had a provider
/// key exported would otherwise never see one. The key travels per
/// request, from the integration `llmman launch` configured, and is never
/// persisted.
fn client_api_key(headers: Option<&HeaderMap>) -> Option<String> {
    let headers = headers?;
    let usable = |k: &str| {
        let k = k.trim();
        (!k.is_empty() && k != PLACEHOLDER_API_KEY).then(|| k.to_string())
    };
    // The scheme is case-insensitive per RFC 7235, and clients do send
    // `bearer`. Matching one spelling would silently drop a real key.
    let bearer = headers
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, token) = v.trim().split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then_some(token)
        })
        .and_then(usable);
    // Each candidate is filtered before the choice between them, not
    // after: a client that sends both a placeholder `Authorization` and a
    // real `x-api-key` still has a real key.
    bearer.or_else(|| {
        headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .and_then(usable)
    })
}

/// Whether a request was made by a browser on some other site's behalf.
///
/// `cors_layer` keeps such a page from *reading* a response, but not from
/// sending the request: a "simple" POST (`text/plain`, a form encoding)
/// skips the preflight entirely, and these handlers parse the body
/// without consulting `Content-Type`. Any page a user visits could
/// therefore make a loopback daemon spend the provider key in its
/// environment, and never need to see the reply to have cost them money.
///
/// `Sec-Fetch-Site` is what distinguishes them: browsers attach it to
/// every request, `same-origin`/`none` for llmman's own web UI and the
/// address bar, `cross-site`/`same-site` for another page's fetch. A CLI
/// integration sends no such header at all, so this only ever withholds
/// the daemon's own credentials from a browser acting for someone else —
/// a caller that presents its own key is unaffected.
fn is_cross_site(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|h| h.get("sec-fetch-site"))
        .and_then(|v| v.to_str().ok())
        .is_some_and(|site| {
            let site = site.trim();
            site.eq_ignore_ascii_case("cross-site") || site.eq_ignore_ascii_case("same-site")
        })
}

/// The models.dev catalog, as everything in this module reaches it: off
/// the runtime, since the first call fetches (and caches) it while every
/// later one is memoized, and a 502 when that fetch fails — the failure
/// is upstream's, not the caller's.
async fn provider_catalog() -> Result<Arc<crate::providers::Catalog>, AppError> {
    tokio::task::spawn_blocking(crate::providers::catalog)
        .await
        .context("provider catalog task panicked")?
        .map_err(|e| AppError(e, StatusCode::BAD_GATEWAY))
}

/// Resolves a [`crate::providers::REMOTE_PREFIX`] reference into a
/// [`Target::Remote`], or `None` for any ordinary local reference.
///
/// The API key is the caller's own (see [`client_api_key`]) when it sent
/// one, else the provider's variable from this daemon's environment — so
/// both `llmman launch --provider`, which puts the real key in the
/// integration's requests, and a daemon started with the key already
/// exported work.
async fn resolve_remote_target(
    model_ref: &str,
    headers: Option<&HeaderMap>,
) -> Result<Option<Target>, AppError> {
    let Some((provider_id, model)) = crate::providers::split_remote_ref(model_ref) else {
        return Ok(None);
    };
    let (provider_id, model) = (provider_id.to_string(), model.to_string());

    let catalog = provider_catalog().await?;

    let provider = catalog.get(&provider_id).ok_or_else(|| {
        AppError(
            crate::providers::unknown_provider_error(&provider_id, &catalog),
            StatusCode::BAD_REQUEST,
        )
    })?;

    // This router has no authentication, so the daemon's own key is
    // withheld from the two cases where the caller is plainly not the
    // operator: a bind the whole network can reach, and a browser acting
    // for another site. That is a blast-radius bound, not authentication
    // — on a shared machine any local account can still reach a loopback
    // daemon, as it can already reach every other thing this daemon does.
    // Presenting a key per request is what avoids relying on this at all.
    let own_key = || {
        (crate::daemon::reachable_only_locally() && !is_cross_site(headers))
            .then(|| provider.api_key())
            .flatten()
    };
    let api_key = client_api_key(headers).or_else(own_key).ok_or_else(|| {
        AppError(
            anyhow!(
                "no API key for provider {provider_id:?} — send it as an Authorization \
                 header{}",
                if is_cross_site(headers) {
                    ". This request came from another site, so llmman serve's own \
                     environment is deliberately not used"
                        .to_string()
                } else if crate::daemon::reachable_only_locally() {
                    format!(", or set {} where llmman serve runs", provider.key_env)
                } else {
                    ". llmman serve is not bound to loopback, so its own environment is \
                     deliberately not used"
                        .to_string()
                }
            ),
            StatusCode::UNAUTHORIZED,
        )
    })?;

    let target = RemoteTarget {
        provider: provider_id,
        base_url: provider.base_url.clone(),
        model,
        api_key,
    };
    // No key on this line: it is the one piece of a remote target that
    // must never reach a log file.
    eprintln!(
        "[llmman] routing {} to provider {} ({})",
        target.model, target.provider, target.base_url
    );
    Ok(Some(Target::Remote(Arc::new(target))))
}

/// Ensures `model_ref` is loaded and returns `(canonical_ref, port,
/// guard)`. The canonical name is what it's actually registered under
/// with its backend (`--served-model-name`), which can differ from a
/// tagless `model_ref` (e.g. `hf.co/owner/repo` canonicalizes to
/// `...:latest`). Callers must forward this canonical name, not their
/// own input, as the "model" field sent to the backend — vllm validates
/// it strictly and 404s otherwise (llama-server doesn't, so this went
/// unnoticed for GGUF models).
///
/// `guard` is an already-claimed [`ActivityGuard`] — every successful
/// return (a cache hit via [`check_running`], or a fresh load's own
/// insert below) claims one `in_flight` unit first, so a concurrent
/// `LLMMAN_MAX_LOADED_MODELS` eviction can never see this model as
/// idle. Pass `guard` on to [`begin_activity`]/[`refresh_activity`],
/// which take over the claim rather than adding a second one; dropping
/// it any other way (including the whole task being cancelled) still
/// releases it correctly.
///
/// `headers` are the incoming request's, used only to pick up a caller-
/// supplied provider API key (see [`resolve_remote_target`]); `None` from
/// a surface that has none to offer.
async fn ensure_model(
    state: &AppState,
    model_ref: &str,
    headers: Option<&HeaderMap>,
) -> Result<(String, Target, ActivityGuard), AppError> {
    // Before `resolve_ollama_api`, deliberately: a provider-routed
    // reference names a model on someone else's servers, so none of the
    // shortname aliasing, tag defaulting, store lookup, or pull below
    // applies to it — and `resolve_ollama_api` would rewrite it into a
    // registry path it is not.
    //
    // The guard is a real one even though nothing is running: every
    // `ActivityGuard` operation looks the model up in `running` and is a
    // no-op when absent (see its `Drop` impl and `begin_activity`), so a
    // remote target needs no separate no-op path.
    if let Some(target) = resolve_remote_target(model_ref, headers).await? {
        return Ok((
            model_ref.to_string(),
            target,
            ActivityGuard::new(state, model_ref),
        ));
    }

    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    // Default the tag before the lock below: otherwise two concurrent
    // first-pulls of e.g. "gemma4" and "gemma4:latest" take different
    // locks and both spawn a process for the same model.
    let model_ref = crate::storage::default_tag(&model_ref);
    // Kept from before `canonical_ref`, which reads the store: this is
    // the key the load lock is taken on, and it has to be the same string
    // for the whole of a load even though the pull below changes what
    // `canonical_ref` returns. See `load_identity`, which `unload_model`
    // locks on for the same reason.
    let load_id = model_ref.clone();
    let model_ref = canonical_ref(&state.0.store_path, &model_ref);
    let model_ref = model_ref.as_str();

    // Already loaded and reusable — bypasses LLMMAN_MAX_QUEUE entirely,
    // same as Ollama's own GetRunner bypassing pendingReqCh for a
    // reusable runner (server/sched.go): only a request that actually
    // needs scheduling work (waiting on a concurrent load, or starting
    // a fresh one) below counts against the cap.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    // See try_admit's doc comment — held for the rest of this function.
    let _queue_guard = try_admit(state.0.max_queue)?;

    let _guard = acquire_load_lock(&load_id).await;

    // Someone else may have finished loading this model while we
    // waited for the lock above.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    // If the model is not in the local store, pull it now.
    if crate::storage::OciStore::open(&state.0.store_path)
        .and_then(|s| s.find(model_ref))
        .is_err()
    {
        eprintln!("[llmman] {model_ref} not in store — pulling");
        let store_path = state.0.store_path.clone();
        let model_ref_owned = model_ref.to_owned();
        tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_ref_owned))
            .await
            .context("pull task panicked")?
            .context("pull failed")?;
    }

    // Re-canonicalise after the pull: default_tag already fixed the lock
    // key, so this only refines to a more specific stored form.
    let model_ref = canonical_ref(&state.0.store_path, model_ref);
    let model_ref = model_ref.as_str();

    // Re-check in case that stored form differs from the key above.
    if let Some((port, guard)) = check_running(state, model_ref).await {
        return Ok((model_ref.to_string(), Target::Local(port), guard));
    }

    let model_path = resolve_model(&state.0.store_path, &state.0.cache_path, model_ref)
        .with_context(|| format!("resolve model {model_ref}"))?;
    // Best-effort — used only to populate `llmman ps`'s ID/SIZE columns;
    // resolve_model above already established the model exists, so a
    // failure here (e.g. a race with a concurrent `rm`) just means those
    // columns show as empty/zero rather than failing the whole request.
    let (digest, size) = OciStore::open(&state.0.store_path)
        .and_then(|s| {
            s.find(model_ref).map(|d| {
                let size = s.total_size(&d);
                (d.digest, size)
            })
        })
        .unwrap_or_default();
    let context_shift = supports_context_shift(model_ref);
    // See enforce_max_loaded_models's doc comment — held for the rest
    // of this function.
    let _pending_load_guard = enforce_max_loaded_models(state, state.0.max_loaded_models).await?;
    // OOM retry loop — on a local llama-server load that fails with a
    // memory-allocation-looking error, tries progressively more invasive
    // fallbacks before giving up (see each branch's own comment for which
    // Ollama behavior it mirrors). Never mutates state.0.ctx_size, so a
    // later reload starts fresh. A fresh `port` is picked for every
    // attempt, not just the first — otherwise a retry's replacement
    // process could try to bind the same port the previous (failed,
    // possibly not-yet-fully-exited) one was still holding.
    let mut ctx_size = state.0.ctx_size;
    let mut split_mode = state.0.split_mode;
    let mut shrink_attempts = 0u32;
    let mut evicted_others = false;
    let mut split_mode_relaxed = false;
    let mut process;
    let mut port = find_free_port()?;
    loop {
        eprintln!("[llmman] loading {model_ref} on port {port}");
        // A `None` ctx_size (a high-VRAM host deferring to the model's
        // own trained context) has nothing safe to scale — forwarding
        // --parallel unscaled in that case would silently divide that
        // trained context across slots instead, exactly what scaling
        // exists to prevent. Fall back to llama-server's own
        // single-slot default rather than risk that.
        let num_parallel = effective_num_parallel(ctx_size, state.0.num_parallel);
        if state.0.num_parallel.is_some() && num_parallel.is_none() {
            eprintln!(
                "[llmman] {model_ref}: no explicit ctx-size to scale, ignoring LLMMAN_NUM_PARALLEL for this load"
            );
        }
        // See backend_ctx_size's doc comment — the value actually
        // forwarded as --ctx-size, scaled up for num_parallel.
        let scaled_ctx_size = backend_ctx_size(ctx_size, num_parallel);
        // Resolved once here, then forwarded verbatim to whichever of
        // the two llama-server spawners this load ends up using — see
        // container::LlamaOptions.
        let llama_opts = crate::container::LlamaOptions {
            port,
            ctx_size: scaled_ctx_size,
            flash_attention: state.0.flash_attention.as_deref(),
            kv_cache_type: state.0.kv_cache_type.as_deref(),
            context_shift,
            split_mode,
            num_parallel,
            threads: state.0.threads,
        };
        // Only a local llama-server child captures a stderr tail (see
        // spawn_llama_server) — every retry below only fires for that case.
        let mut stderr_tail: Option<OutputTail> = None;
        process = match (&model_path, state.0.ociman) {
            // container::spawn ignores llama_opts.threads; see that
            // field's doc comment.
            (ModelPath::Gguf(path, mmproj), Some(ociman)) => ModelProcess::Container(
                ociman,
                crate::container::spawn(
                    ociman,
                    path,
                    mmproj.as_deref(),
                    state.0.llama_cpp_version.as_deref(),
                    llama_opts,
                )?,
            ),
            (ModelPath::Gguf(path, mmproj), None) => {
                let bin = local_llama_server_bin(state).await?;
                let (child, tail) =
                    spawn_llama_server(&bin, path, mmproj.as_deref(), llama_opts).await?;
                stderr_tail = Some(tail);
                ModelProcess::Local(Engine::LlamaServer, child, None)
            }
            (ModelPath::SafeTensors(_dir), _) if use_mlx_for_safetensors() => {
                let child = spawn_mlx_server(port).await?;
                let pid = child.id();
                ModelProcess::Local(Engine::Mlx, child, pid)
            }
            (ModelPath::SafeTensors(dir), _) => {
                let child = spawn_vllm_server(dir, port, model_ref).await?;
                let pid = child.id();
                ModelProcess::Local(Engine::Vllm, child, pid)
            }
        };

        match wait_for_ready(&state.0.client, port, &mut process, stderr_tail.as_ref()).await {
            Ok(()) => break,
            Err(e) => {
                let looks_oom = stderr_tail.is_some() // local llama-server only
                    && looks_like_oom(&e.to_string());
                if !looks_oom {
                    return Err(e.into());
                }
                // See ModelProcess::stop_and_wait's own doc comment.
                process.stop_and_wait().await;

                // Cheapest fallback first: free memory without changing
                // anything about how this model itself gets loaded, by
                // evicting every other idle-but-loaded model (mirrors
                // Ollama's own "evict all other models and retry once").
                if !evicted_others {
                    evicted_others = true;
                    if evict_other_models(state, model_ref).await {
                        eprintln!(
                            "[llmman] {model_ref} failed to load on port {port}, which looks like an out-of-memory error — evicted other loaded models and retrying: {:#}",
                            e
                        );
                        port = find_free_port()?;
                        continue;
                    }
                }

                // A hard LLMMAN_SCHED_SPREAD=0 (--split-mode none)
                // restriction can itself be why this looks OOM — the
                // model simply doesn't fit on one GPU at all, which no
                // amount of ctx-size shrinking below would fix. Lift it
                // before falling back to shrinking.
                if !split_mode_relaxed && split_mode == Some("none") {
                    split_mode_relaxed = true;
                    split_mode = Some("layer");
                    eprintln!(
                        "[llmman] {model_ref} failed to load on port {port} with --split-mode none, which looks like an out-of-memory error — retrying with --split-mode layer (spread across every GPU) instead of failing outright: {:#}",
                        e
                    );
                    port = find_free_port()?;
                    continue;
                }

                // Only auto-shrink a ctx-size this daemon picked itself —
                // silently overriding an explicit LLMMAN_CONTEXT_LENGTH
                // would ignore the user's own stated choice (mirrors
                // Ollama's own numCtxAuto gate on
                // reduceAutoNumCtxForLoadOOM).
                let can_shrink =
                    !state.0.ctx_size_explicit && shrink_attempts < MAX_CTX_SHRINK_ATTEMPTS;
                let Some(next) = can_shrink
                    .then(|| next_ctx_size_after_oom(ctx_size))
                    .flatten()
                else {
                    return Err(e.into());
                };
                eprintln!(
                    "[llmman] {model_ref} failed to load on port {port}, which looks like an out-of-memory error — retrying with --ctx-size {next} (was {}): {:#}",
                    ctx_size
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "model default".to_string()),
                    e
                );
                ctx_size = Some(next);
                shrink_attempts += 1;
                port = find_free_port()?;
            }
        }
    }
    eprintln!("[llmman] {model_ref} ready on port {port}");

    // See RunningModel::backend_model_path's own doc comment — only
    // meaningful for Engine::Mlx, which is the only engine
    // spawn_mlx_server deliberately doesn't preload via `--model` for
    // (see its own doc comment), so every request must instead carry
    // this exact directory as its own "model" field.
    let backend_model_path = match &process {
        ModelProcess::Local(Engine::Mlx, _, _) => model_path.path().to_str().map(|s| s.to_string()),
        _ => None,
    };

    state.0.manager.lock().await.running.insert(
        model_ref.to_string(),
        RunningModel {
            process,
            port,
            digest,
            size,
            started_at: now_rfc3339(),
            last_active: Instant::now(),
            last_active_wall: chrono::Utc::now(),
            backend_model_path,
            keep_alive: default_keep_alive(),
            // 1, not 0 — see this function's own doc comment.
            in_flight: 1,
        },
    );
    Ok((
        model_ref.to_string(),
        Target::Local(port),
        ActivityGuard::new(state, model_ref),
    ))
}

/// The `"model"` value to actually put in the JSON request body sent to
/// `canonical_model`'s backend process — `canonical_model` itself
/// (`ensure_model`'s return value, already the exact name every other
/// engine needs — see its own doc comment) for everything except a
/// running `Engine::Mlx` backend, for which it's that model's real
/// on-disk directory path instead (`RunningModel::backend_model_path` —
/// see `spawn_mlx_server`'s doc comment for why `mlx_lm.server` needs
/// that rather than a human-readable name at all).
///
/// Every caller must apply this only to the request forwarded to the
/// backend — client-facing response bodies (an Ollama chunk's `model`
/// field, an Anthropic message's `model` field, ...) must keep echoing
/// back `canonical_model` or the client's own original input unchanged;
/// a client asking for "gemma4:latest" should never see
/// "/Users/.../cache/.../abcd1234" reflected back at it just because
/// that happens to be how this one engine addresses it internally.
async fn backend_wire_model(state: &AppState, target: &Target, canonical_model: &str) -> String {
    // A remote provider knows the model by its own id, not by the
    // prefixed reference llmman routes on — see `providers::REMOTE_PREFIX`
    // for why that reference has to be namespaced in the first place.
    if let Target::Remote(remote) = target {
        return remote.model.clone();
    }
    state
        .0
        .manager
        .lock()
        .await
        .running
        .get(canonical_model)
        .and_then(|r| r.backend_model_path.clone())
        .unwrap_or_else(|| canonical_model.to_string())
}

/// Returns the local llama-server binary to spawn: the one resolved at
/// startup, unless that file has since disappeared from disk (the install
/// that provided it was upgraded or removed while this daemon kept
/// running), in which case it is re-resolved from the current PATH (or
/// re-downloaded) and the replacement remembered for subsequent loads —
/// instead of failing every model load forever with a spawn error against
/// a path that no longer exists.
async fn local_llama_server_bin(state: &AppState) -> anyhow::Result<PathBuf> {
    let current = state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let Some(bin) = current else {
        anyhow::bail!("no local llama-server binary resolved and --ociman was not set")
    };
    if bin.exists() {
        return Ok(bin);
    }
    eprintln!(
        "[llmman] llama-server at {} no longer exists; re-resolving",
        bin.display()
    );
    let pinned = state.0.llama_cpp_version.clone();
    let resolved = tokio::task::spawn_blocking(move || resolve_llama_server(pinned.as_deref()))
        .await
        .context("resolve llama-server task panicked")??;
    *state
        .0
        .llama_server_bin
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(resolved.clone());
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Proxy helper – forward raw bytes to llama-server and stream back
// ---------------------------------------------------------------------------

async fn proxy(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    // `Bytes` clones are refcounted, not copies — passing `body` straight
    // through (reqwest::Body: From<Bytes>) avoids an extra full-size
    // allocation that `body.to_vec()` would add on top of it, which
    // matters most for large multipart audio uploads.
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    let status = resp.status();
    let resp_headers = resp.headers().clone();

    // Moved into the stream below (see ActivityGuard's doc comment) so it
    // isn't dropped — resetting this model's idle clock — until the whole
    // response body has actually been relayed.
    let stream = resp.bytes_stream().map(move |item| {
        let _activity = &activity;
        item.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        builder = builder.header(k, v);
    }
    Ok(builder.body(Body::from_stream(stream)).unwrap())
}

// ---------------------------------------------------------------------------
// Proxy helpers – like `proxy` above, but for a request whose backend
// needed a *different* "model" name than the client itself asked for
// (see `backend_wire_model`'s own doc comment — only ever true for an
// `Engine::Mlx` backend, addressed by its real on-disk directory path
// rather than a human-readable name). `mlx_lm.server` echoes whatever
// "model" value it received straight back into every response it sends
// — the one non-streamed JSON body for `stream: false`, and *every*
// individual `data: {...}` SSE chunk for `stream: true` — so a plain
// byte-for-byte relay like `proxy` would leak that internal directory
// path back to the client instead of the name it actually asked for.
// These two rewrite just that one field back to the canonical name
// before any of it reaches the client; every other field, and (for the
// streaming variant) the SSE framing itself, passes through unchanged.
// ---------------------------------------------------------------------------

/// Sets `value["model"]` to `canonical_model` if that key is present at
/// all — shared by both helpers below so a response shape that happens
/// not to carry one (an error body, a future backend response this
/// doesn't recognize) is left alone rather than gaining a field it
/// never had.
fn set_response_model(value: &mut serde_json::Value, canonical_model: &str) {
    if value.get("model").is_some() {
        value["model"] = serde_json::Value::String(canonical_model.to_string());
    }
}

/// [`proxy_rewriting_model`]'s actual rewrite, split out as a pure
/// `bytes -> bytes` function so it's directly unit-testable without any
/// networking at all. Parses `raw` as JSON, rewrites its `"model"` field
/// (see [`set_response_model`]), and re-serializes — or returns `raw`
/// completely unchanged if it isn't valid JSON at all (an error body's
/// own shape, or a future backend response this doesn't recognize)
/// rather than mangling or dropping it.
fn rewrite_json_response_model(raw: &Bytes, canonical_model: &str) -> Bytes {
    match serde_json::from_slice::<serde_json::Value>(raw) {
        Ok(mut value) => {
            set_response_model(&mut value, canonical_model);
            serde_json::to_vec(&value)
                .map(Bytes::from)
                .unwrap_or_else(|_| raw.clone())
        }
        Err(_) => raw.clone(),
    }
}

/// [`stream_rewriting_model`]'s actual per-line rewrite, split out as a
/// pure `&str -> String` function so it's directly unit-testable without
/// any networking at all. `line` is one already-decoded logical line
/// from [`bytes_to_lines`] (its own line ending already stripped, not
/// yet restored here — the caller does that once, uniformly, since
/// every branch below needs it regardless of which one fires): a
/// `data: {...}` line whose payload parses as JSON gets its `"model"`
/// field rewritten (see [`set_response_model`]); `data: [DONE]`, a
/// blank SSE event-separator line, or a `data: ` line whose payload
/// *doesn't* parse as JSON all pass through byte-for-byte unchanged.
fn rewrite_sse_line_model(line: &str, canonical_model: &str) -> String {
    match line.strip_prefix("data: ") {
        Some(payload) if payload != "[DONE]" => match serde_json::from_str(payload) {
            Ok(mut value) => {
                set_response_model(&mut value, canonical_model);
                format!(
                    "data: {}",
                    serde_json::to_string(&value).unwrap_or_else(|_| payload.to_string())
                )
            }
            Err(_) => line.to_string(),
        },
        _ => line.to_string(),
    }
}

/// The non-streaming (`stream: false`, or no `stream` concept at all —
/// embeddings, the Responses API's token-counting endpoint) case:
/// buffers the whole response body (unlike `proxy`, which never does)
/// so its `"model"` field can be parsed, rewritten, and re-serialized
/// before forwarding it on. Every route that can reach this returns one
/// complete JSON object either way (never anything token-streamed a
/// client would notice the added latency of buffering first), so this
/// costs nothing a real client could observe.
///
/// `Content-Length`, if the backend sent one, is dropped rather than
/// forwarded: the rewritten body is a different size than the original
/// one that header described, and hyper/axum fill in the correct value
/// for a fixed (`Body::from(Bytes)`, not streamed) body on their own
/// when none is set explicitly.
async fn proxy_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: &str,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let raw = resp
        .bytes()
        .await
        .context("read inference backend response")?;
    // The whole body is already collected by this point, so there's no
    // partial relay left for keeping this alive any longer to protect —
    // see `proxy`'s own comment on why it instead holds this open across
    // its whole (streamed) relay.
    drop(activity);

    let rewritten = rewrite_json_response_model(&raw, canonical_model);

    let mut builder = Response::builder().status(status.as_u16());
    for (k, v) in &resp_headers {
        if k == reqwest::header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(k, v);
    }
    Ok(builder.body(Body::from(rewritten)).unwrap())
}

/// The streaming (`stream: true`) case: like `stream_ollama`/
/// `stream_anthropic`, uses `bytes_to_lines` so a `data: {...}` SSE line
/// split across two TCP reads is never parsed as JSON prematurely — but
/// unlike those two (which convert into a completely different wire
/// format, ndjson/Anthropic SSE, and so don't need to preserve the
/// original SSE framing at all), this must reproduce the exact original
/// OpenAI SSE shape byte-for-byte except for the one field being
/// rewritten: every blank line (an SSE event separator) and the
/// trailing `data: [DONE]` sentinel pass through completely unchanged;
/// only a `data: {...}` line whose payload actually parses as a JSON
/// object carrying a `model` field gets rewritten.
async fn stream_rewriting_model(
    client: &Client,
    target: &Target,
    route: &str,
    headers: &HeaderMap,
    body: Bytes,
    activity: ActivityGuard,
    canonical_model: String,
) -> Result<Response, AppError> {
    let mut req = target.authorize(client.post(target.url(route)).body(body));
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("proxy request to {}", target.describe()))?;
    let status = resp.status();
    // Only content-type is meaningful to forward for a stream the
    // caller is about to reconstruct line by line — content-length
    // (absent anyway for a real chunked/streamed response) would be
    // stale the same way proxy_rewriting_model's is, if present.
    let content_type = resp.headers().get(reqwest::header::CONTENT_TYPE).cloned();

    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        // See `proxy`'s own comment on this same pattern.
        let _activity = &activity;
        // bytes_to_lines strips the original line ending; restored here,
        // uniformly, regardless of which of rewrite_sse_line_model's own
        // branches actually fired.
        let out = rewrite_sse_line_model(&line, &canonical_model) + "\n";
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    let mut builder = Response::builder().status(status.as_u16());
    if let Some(ct) = content_type {
        builder = builder.header(reqwest::header::CONTENT_TYPE, ct);
    }
    Ok(builder.body(Body::from_stream(stream)).unwrap())
}

// ---------------------------------------------------------------------------
// SSE line buffering
//
// reqwest::bytes_stream() delivers raw TCP chunks; a single `data: {json}\n`
// SSE line can be split across two chunks.  bytes_to_lines buffers incomplete
// data and only yields complete newline-terminated lines, so downstream JSON
// parsing never sees a partial line.
// ---------------------------------------------------------------------------

fn bytes_to_lines(
    stream: impl futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl futures::Stream<Item = String> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), Vec::<u8>::new()),
        |(mut stream, mut buf)| async move {
            loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&buf[..pos])
                        .trim_end_matches('\r')
                        .to_string();
                    buf.drain(..=pos);
                    return Some((line, (stream, buf)));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(_)) | None => {
                        if buf.is_empty() {
                            return None;
                        }
                        let line = String::from_utf8_lossy(&buf).into_owned();
                        buf.clear();
                        return Some((line, (stream, buf)));
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Shared SSE-chunk helper
// ---------------------------------------------------------------------------

/// Returns (content, thinking, done).
fn oai_chunk_to_content(payload: &str) -> Option<(String, Option<String>, bool)> {
    if payload == "[DONE]" {
        return Some((String::new(), None, true));
    }
    let chunk = serde_json::from_str::<OAIChunk>(payload).ok()?;
    let choice = chunk.choices.first()?;
    let content = choice.delta.content.as_deref().unwrap_or("").to_string();
    // Accept both field names: "reasoning_content" (Homebrew llama-server) and "thinking" (git)
    let thinking = choice
        .delta
        .reasoning_content
        .clone()
        .or_else(|| choice.delta.thinking.clone())
        .filter(|s| !s.is_empty());
    let done = choice
        .finish_reason
        .as_deref()
        .map(|r| !r.is_empty() && r != "null")
        .unwrap_or(false);
    Some((content, thinking, done))
}

// ---------------------------------------------------------------------------
// Shared "POST an OpenAI chat request, fail on non-2xx" helper
// ---------------------------------------------------------------------------

/// Sets `repeat_penalty` to `DEFAULT_REPEAT_PENALTY` on `oai_req` unless a
/// construction site already resolved one from the caller's own request.
/// `post_chat` is the *only* place this is called — and, in turn, the only
/// function any typed request (`/api/chat`, `/api/generate`, the Anthropic
/// Messages API) actually goes through to reach llama-server (see its own
/// doc comment) — so none of those three construction sites need to
/// remember to apply this default themselves the way they used to.
fn apply_default_repeat_penalty_typed(oai_req: &mut OAIChatRequest) {
    if oai_req.repeat_penalty.is_none() {
        oai_req.repeat_penalty = Some(DEFAULT_REPEAT_PENALTY);
    }
}

/// Whether a request bound for `target` may carry `repeat_penalty`.
///
/// It is a llama.cpp extension, not an OpenAI field. Sending it to a
/// local backend is the whole point of [`DEFAULT_REPEAT_PENALTY`];
/// sending it to a provider is at best ignored and at worst a 400, since
/// OpenAI rejects unrecognized arguments outright. A caller that asked
/// for one explicitly is in the same position, so a remote request drops
/// the field either way.
fn repeat_penalty_applies(target: &Target) -> bool {
    !target.is_remote()
}

/// POSTs oai_req to url and returns the still-streaming response, converting
/// a non-2xx status into an AppError carrying the backend's error body.
/// The *only* function that actually sends an `OAIChatRequest` to
/// llama-server — every caller (`collect_completion`, `stream_ollama`,
/// `stream_anthropic`, and `handle_anthropic_messages`'s non-streaming
/// branch) goes through this one function, which is what lets
/// `apply_default_repeat_penalty_typed` above resolve `repeat_penalty`
/// exactly once instead of at every construction site.
async fn post_chat(
    client: &Client,
    target: &Target,
    oai_req: &mut OAIChatRequest,
) -> Result<reqwest::Response, AppError> {
    if repeat_penalty_applies(target) {
        apply_default_repeat_penalty_typed(oai_req);
    } else {
        oai_req.repeat_penalty = None;
    }
    let resp = target
        .authorize(
            client
                .post(target.url(CHAT_COMPLETIONS_ROUTE))
                .json(oai_req),
        )
        .send()
        .await
        .with_context(|| format!("send to {}", target.describe()))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError(
            anyhow!("{} {status}: {body}", target.describe()),
            // A provider's own 4xx is the actionable answer — a bad key
            // has to reach the user as 401, not as llmman's 500. A local
            // backend keeps the blanket 500 it always returned.
            remote_status(target, status),
        ));
    }
    Ok(resp)
}

/// The status llmman reports for an unsuccessful upstream response.
///
/// A [`Target::Local`] backend failing is llmman's own problem, so it
/// stays a 500 as it always has. A provider's 4xx is about the caller's
/// request or credentials, so it is passed through rather than buried;
/// anything else from a provider is a bad gateway.
fn remote_status(target: &Target, upstream: StatusCode) -> StatusCode {
    match target {
        Target::Local(_) => StatusCode::INTERNAL_SERVER_ERROR,
        Target::Remote(_) if upstream.is_client_error() => upstream,
        Target::Remote(_) => StatusCode::BAD_GATEWAY,
    }
}

// ---------------------------------------------------------------------------
// Streaming conversion: OpenAI SSE → Ollama NDJSON (chat + generate)
//
// The chat and generate endpoints differ only in which Ollama chunk struct
// wraps each token (OllamaChatChunk's nested `message.content` vs
// OllamaGenerateChunk's flat `response`), so both go through this one
// generic driver; build_chunk supplies just that piece.
// ---------------------------------------------------------------------------

/// Fallback content/thinking separation for a backend that hands back raw
/// `<think>...</think>` or gpt-oss-style harmony channel tokens as plain
/// `content` text, instead of already splitting them into a structured
/// `reasoning_content`/`thinking` delta field the way `oai_chunk_to_content`
/// prefers. One instance is created per streamed response (see
/// `stream_ollama`) and fed every chunk's `content` in order, so it can
/// buffer across a token boundary that splits a tag mid-way exactly like
/// `thinking::Parser`/`harmony::HarmonyMessageHandler` themselves already
/// do internally.
enum RawContentExtractor {
    /// No backend-structured thinking has been seen yet, and not enough
    /// raw content has arrived yet to decide a mode from — the `String`
    /// buffers everything seen so far. Kept buffered (rather than decided
    /// per-chunk) because a real streamed response can hand this the
    /// first token of a tag one byte at a time, and e.g. a lone `"<"` is
    /// a prefix of every candidate tag below, not evidence of any one of
    /// them in particular.
    Undetermined(String),
    /// A backend already supplied structured thinking on some earlier
    /// chunk of this stream — never scan raw content again, even if a
    /// later chunk's `content` happens to contain literal tag-like text
    /// as part of genuine output.
    Passthrough,
    Harmony(Box<crate::harmony::HarmonyMessageHandler>),
    PlainThink(Box<crate::thinking::Parser>),
}

/// Every raw-token prefix `RawContentExtractor::Undetermined` can still be
/// waiting to disambiguate between — gpt-oss harmony's two possible
/// stream-start spellings (see the `<|channel|>` case below) and a plain
/// `<think>` tag.
const CANDIDATE_TAGS: [&str; 3] = ["<|start|>", "<|channel|>", "<think>"];

impl RawContentExtractor {
    fn new() -> Self {
        RawContentExtractor::Undetermined(String::new())
    }

    /// Returns the (content, thinking) to actually emit for this chunk,
    /// given what the backend itself already reported.
    fn process(
        &mut self,
        content: String,
        backend_thinking: Option<String>,
    ) -> (String, Option<String>) {
        if backend_thinking.is_some() {
            // `flush` first: if this transition happens straight out of
            // `Undetermined` (an earlier chunk was still a strict prefix
            // of a candidate tag — e.g. a lone `"<"` — when this chunk
            // turned out to carry backend-structured thinking instead),
            // whatever was buffered for disambiguation must still reach
            // the client; it otherwise has no other path out once `self`
            // is overwritten below. A no-op on every other variant (see
            // `flush`'s own doc comment).
            let buffered = self.flush();
            *self = RawContentExtractor::Passthrough;
            return (buffered + &content, backend_thinking);
        }
        match self {
            RawContentExtractor::Passthrough => (content, None),
            RawContentExtractor::Harmony(h) => {
                let (c, t, tool) = h.add_content(&content);
                (c, non_empty_thinking(t, tool))
            }
            RawContentExtractor::PlainThink(p) => {
                let (t, c) = p.add_content(&content);
                (c, (!t.is_empty()).then_some(t))
            }
            RawContentExtractor::Undetermined(buf) => {
                buf.push_str(&content);
                let trimmed = buf.trim_start();
                if trimmed.is_empty()
                    || CANDIDATE_TAGS
                        .iter()
                        .any(|tag| tag.starts_with(trimmed) && trimmed.len() < tag.len())
                {
                    // Still ambiguous (whitespace only so far, or a
                    // strict prefix of a candidate tag that could still
                    // go either way) — keep buffering, nothing to emit
                    // yet.
                    return (String::new(), None);
                }
                let buffered = std::mem::take(buf);
                let trimmed_starts_with = |tag: &str| buffered.trim_start().starts_with(tag);
                if trimmed_starts_with("<|start|>") || trimmed_starts_with("<|channel|>") {
                    let mut h = crate::harmony::HarmonyMessageHandler::new();
                    // A raw completion stream from a chat-templated
                    // request typically never re-emits the assistant's
                    // own `<|start|>assistant` preamble (the template
                    // already sent it as part of the *prompt*, before
                    // generation started) — only what follows it, i.e.
                    // `<|channel|>...`. HarmonyParser's own state machine
                    // requires having seen a `<|start|>` before it will
                    // recognize anything after it as a header (see
                    // `harmony::HarmonyParser`'s `LookingForMessageStart`
                    // state) — priming it here is exactly what
                    // `add_implicit_start`'s own doc comment describes.
                    // Not primed for a stream that already starts with a
                    // literal `<|start|>` itself, which needs no help
                    // finding its own message boundary.
                    if trimmed_starts_with("<|channel|>") {
                        h.parser.add_implicit_start();
                    }
                    let (c, t, tool) = h.add_content(&buffered);
                    let thinking = non_empty_thinking(t, tool);
                    *self = RawContentExtractor::Harmony(Box::new(h));
                    (c, thinking)
                } else {
                    let mut p = crate::thinking::Parser::new("<think>", "</think>");
                    let (t, c) = p.add_content(&buffered);
                    *self = RawContentExtractor::PlainThink(Box::new(p));
                    (c, (!t.is_empty()).then_some(t))
                }
            }
        }
    }

    /// Drains whatever `Undetermined` is still holding back for
    /// disambiguation — called once the stream is `done` (see
    /// `stream_ollama`), so a reply that ends while still a strict prefix
    /// of a candidate tag (e.g. the very last byte generated is a lone
    /// `"<"`) still reaches the client instead of being silently dropped.
    /// A no-op for every other variant: `Harmony`/`PlainThink` only ever
    /// hold back a *candidate closing/end tag* this same way internally,
    /// which real Ollama's own `thinking.Parser` (this module's `PlainThink`
    /// is a direct port of it) has the identical characteristic for and
    /// never flushes either — not a new gap this fallback introduces.
    fn flush(&mut self) -> String {
        match self {
            RawContentExtractor::Undetermined(buf) => std::mem::take(buf),
            _ => String::new(),
        }
    }
}

/// Folds a harmony tool-call channel's raw argument text (`tool`) into
/// the same "thinking" bucket as real reasoning text (`thinking`) — there
/// being no structured-tool-call plumbing wired to this raw-token fallback
/// path (see `RawContentExtractor`'s own doc comment: this only ever
/// engages when a backend hands back literal, unparsed harmony tokens in
/// the first place), hiding a stray tool call's raw JSON in "thinking"
/// rather than ever showing it in the user-visible `content` field is the
/// safer failure mode of the two.
fn non_empty_thinking(thinking: String, tool: String) -> Option<String> {
    let combined = thinking + &tool;
    (!combined.is_empty()).then_some(combined)
}

/// One decoded SSE line: content, thinking, tool calls, done.
type OllamaDelta = (String, Option<String>, Option<Vec<OllamaToolCall>>, bool);

/// Response-spanning decode state, shared by both paths below so each
/// reads a response identically.
struct OllamaLineDecoder {
    tool_calls_acc: std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
    content_extractor: std::cell::RefCell<RawContentExtractor>,
}

impl OllamaLineDecoder {
    fn new() -> Self {
        Self {
            tool_calls_acc: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            content_extractor: std::cell::RefCell::new(RawContentExtractor::new()),
        }
    }

    /// `None` for a line that isn't a recognized SSE payload.
    fn decode(&self, line: &str) -> Option<OllamaDelta> {
        let payload = line.strip_prefix("data: ")?;
        accumulate_tool_call_deltas(payload, &self.tool_calls_acc);
        let (content, thinking, done) = oai_chunk_to_content(payload)?;
        let (mut content, thinking) = self
            .content_extractor
            .borrow_mut()
            .process(content, thinking);
        if done {
            // Idempotent even across the two `done` chunks
            // real Ollama's stream can produce (see below):
            // `flush` drains via `mem::take`, so the second call
            // just returns an already-empty string.
            content.push_str(&self.content_extractor.borrow_mut().flush());
        }
        // llama-server's SSE stream signals "done" twice — once on
        // the chunk carrying a real finish_reason, then again on
        // the trailing literal "[DONE]" line — so `done` here can
        // be true more than once per response. Draining (not just
        // reading) the accumulator on the first occurrence means
        // finalize_tool_calls sees an empty map and returns `None`
        // on any later one, so a client can't be handed (and
        // potentially act on) the same tool call twice.
        let tool_calls = done.then(|| {
            let drained = std::mem::take(&mut *self.tool_calls_acc.borrow_mut());
            finalize_tool_calls(&drained)
        });
        Some((content, thinking, tool_calls.flatten(), done))
    }
}

/// A whole response accumulated into the one reply a non-streaming
/// request gets. `done` stays false when the backend never sent a
/// terminal chunk, which means the reply is truncated.
#[derive(Default)]
struct OllamaFold {
    content: String,
    thinking: Option<String>,
    tool_calls: Option<Vec<OllamaToolCall>>,
    done: bool,
}

impl OllamaFold {
    fn push(&mut self, (content, thinking, tool_calls, done): OllamaDelta) {
        self.content.push_str(&content);
        if let Some(t) = thinking {
            self.thinking.get_or_insert_with(String::new).push_str(&t);
        }
        // Only the first `done` yields tool calls, so a later `None` never
        // overwrites them.
        if tool_calls.is_some() {
            self.tool_calls = tool_calls;
        }
        self.done |= done;
    }
}

/// Folds lines through a fresh decoder. Used by the tests; the handler
/// pushes as each line arrives instead of buffering them all.
#[cfg(test)]
fn fold_ollama_lines<I: IntoIterator<Item = String>>(lines: I) -> OllamaFold {
    let decoder = OllamaLineDecoder::new();
    let mut fold = OllamaFold::default();
    for line in lines {
        if let Some(delta) = decoder.decode(&line) {
            fold.push(delta);
        }
    }
    fold
}

/// `build_chunk`'s `tool_calls` parameter is only ever `Some` on the final
/// (`done`) chunk of an `/api/chat` response that made one or more tool
/// calls — `/api/generate` (no tool-calling support in real Ollama
/// either) always gets `None` here and ignores it.
async fn stream_ollama<T: Serialize + Send + 'static>(
    streaming: bool,
    client: Client,
    target: Target,
    mut oai_req: OAIChatRequest,
    activity: ActivityGuard,
    build_chunk: impl Fn(String, Option<String>, Option<Vec<OllamaToolCall>>, bool) -> T
        + Send
        + 'static,
) -> Result<Response, AppError> {
    let resp = post_chat(&client, &target, &mut oai_req).await?;

    // `stream: false` answers with the one JSON object Ollama returns.
    // llama-server is still asked to stream either way: a non-streaming
    // upstream sends nothing until generation ends, risking a read timeout.
    if !streaming {
        let _activity = activity;
        let decoder = OllamaLineDecoder::new();
        let mut fold = OllamaFold::default();
        let mut lines = Box::pin(bytes_to_lines(resp.bytes_stream()));
        while let Some(line) = lines.next().await {
            if let Some(delta) = decoder.decode(&line) {
                fold.push(delta);
            }
        }
        // bytes_to_lines reports a mid-response read failure as a clean end
        // of stream, so a missing terminal chunk is the only evidence the
        // backend died. Without this the caller gets 200 and `done: true`
        // over silently truncated text; the streaming path at least ends
        // without ever sending a `done` chunk.
        if !fold.done {
            return Err(AppError(
                anyhow!("inference backend closed the connection before finishing"),
                StatusCode::BAD_GATEWAY,
            ));
        }
        return Ok(Json(build_chunk(
            fold.content,
            fold.thinking,
            fold.tool_calls,
            true,
        ))
        .into_response());
    }

    let decoder = OllamaLineDecoder::new();
    let stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        // Moved into this closure purely to keep it alive — see
        // ActivityGuard's doc comment — until the stream itself is
        // dropped, not referenced otherwise.
        let _activity = &activity;
        let out = decoder
            .decode(&line)
            .map(|(content, thinking, tool_calls, done)| {
                let chunk = build_chunk(content, thinking, tool_calls, done);
                serde_json::to_string(&chunk).unwrap_or_default() + "\n"
            })
            .unwrap_or_default();
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Streaming conversion: OpenAI SSE → Anthropic SSE
// ---------------------------------------------------------------------------

async fn stream_anthropic(
    client: Client,
    target: Target,
    mut oai_req: OAIChatRequest,
    model: String,
    activity: ActivityGuard,
) -> Result<Response, AppError> {
    let resp = post_chat(&client, &target, &mut oai_req).await?;

    let msg_id = gen_id();
    let preamble = {
        let start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" }
        });
        format!(
            "event: message_start\ndata: {start}\n\nevent: content_block_start\ndata: {block_start}\n\n"
        )
    };

    let preamble_stream =
        futures::stream::once(futures::future::ready(Ok::<_, std::convert::Infallible>(
            Bytes::from(preamble),
        )));

    let sse_stream = bytes_to_lines(resp.bytes_stream()).map(move |line| {
        let out = if let Some(payload) = line.strip_prefix("data: ") {
            if payload == "[DONE]" {
                let msg_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                    "usage": { "output_tokens": 0 }
                });
                let msg_stop = serde_json::json!({ "type": "message_stop" });
                format!(
                    "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                     event: message_delta\ndata: {msg_delta}\n\n\
                     event: message_stop\ndata: {msg_stop}\n\n"
                )
            } else if let Ok(chunk) = serde_json::from_str::<OAIChunk>(payload) {
                let content = chunk.choices.first()
                    .and_then(|c| c.delta.content.as_deref())
                    .unwrap_or("")
                    .to_string();
                if content.is_empty() {
                    String::new()
                } else {
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": content }
                    });
                    format!("event: content_block_delta\ndata: {delta}\n\n")
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        Ok::<_, std::convert::Infallible>(Bytes::from(out))
    });
    // Moved into the tail of the chained stream so it lives until the
    // whole SSE response has been sent — see ActivityGuard's doc comment.
    let sse_stream = sse_stream.chain(futures::stream::once(async move {
        let _activity = activity;
        Ok::<_, std::convert::Infallible>(Bytes::new())
    }));

    Ok(Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(preamble_stream.chain(sse_stream)))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

fn gzipped(body: &'static [u8], content_type: &'static str) -> Response {
    Response::builder()
        .header("content-type", content_type)
        .header("content-encoding", "gzip")
        .header("cache-control", "public, max-age=3600")
        .body(Body::from(body))
        .unwrap()
}

async fn handle_root(headers: HeaderMap) -> impl IntoResponse {
    let wants_html = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("text/html"))
        .unwrap_or(false);
    if wants_html {
        gzipped(webui::INDEX_HTML, "text/html; charset=utf-8").into_response()
    } else {
        "llmman is running".into_response()
    }
}

async fn handle_bundle_js() -> impl IntoResponse {
    gzipped(webui::BUNDLE_JS, "application/javascript; charset=utf-8")
}

async fn handle_bundle_css() -> impl IntoResponse {
    gzipped(webui::BUNDLE_CSS, "text/css; charset=utf-8")
}

async fn handle_loading_html() -> impl IntoResponse {
    gzipped(webui::LOADING_HTML, "text/html; charset=utf-8")
}

async fn handle_props() -> impl IntoResponse {
    // Return a minimal llama.cpp-compatible /props response in ROUTER mode.
    // The web UI uses `role` to detect multi-model (router) vs single-model mode.
    Json(serde_json::json!({
        "role": "router",
        "total_slots": 0,
        "model_path": "",
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": env!("LLMMAN_VERSION"),
        "modalities": { "vision": false, "audio": false },
        "default_generation_settings": {
            "id": 0,
            "id_task": 0,
            "n_ctx": 4096,
            "speculative": false,
            "is_processing": false,
            "params": {
                "n_predict": -1,
                "seed": 0,
                "temperature": 0.8,
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "top_n_sigma": 0.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typ_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "dry_sequence_breakers": [],
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "stop": [],
                "max_tokens": -1,
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": false,
                "stream": true,
                "logit_bias": [],
                "n_probs": 0,
                "min_keep": 0,
                "grammar": "",
                "grammar_lazy": false,
                "grammar_triggers": [],
                "preserved_tokens": [],
                "chat_format": "",
                "reasoning_format": "",
                "reasoning_in_content": false,
                "generation_prompt": "",
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "backend_sampling": false,
                "speculative.n_max": 16,
                "speculative.n_min": 5,
                "speculative.p_min": 0.9,
                "timings_per_token": false,
                "post_sampling_probs": false,
                "lora": []
            },
            "prompt": "",
            "next_token": {
                "has_next_token": false,
                "has_new_line": false,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": ""
            }
        }
    }))
}

// -- llmman's own API --------------------------------------------------------
//
// `/llmman` is llmman's own, not a compatibility surface: no upstream API
// has a notion of a models.dev provider. `llmman providers`, `run
// --provider`, `list --provider` and `launch --provider` are all clients
// of the two routes below (see `cmd::providers`, and `crate::daemon` for
// the wire types), so the catalog lives in one process: the one that
// needs it to route upstream anyway, and whose key is spent for a request
// that presents none (see `resolve_remote_target`).

/// One entry in `GET /llmman/providers`.
///
/// A count, not the model ids: those are megabytes across the catalog,
/// and a caller wanting one provider's asks for it (`ProviderResponse`).
#[derive(Serialize)]
struct ProviderSummary {
    id: String,
    name: String,
    base_url: String,
    key_env: String,
    /// Whether *this daemon's* environment holds `key_env`.
    key_set: bool,
    /// Whether it would actually spend it for a request that presents no
    /// key of its own — `key_set` plus this daemon's own bind check, which
    /// only it can make (see `resolve_remote_target`). A client asking
    /// "will my keyless request work" has to read this, not `key_set`:
    /// its own `LLMMAN_HOST` says nothing about how the daemon is bound.
    key_usable: bool,
    models: usize,
}

impl From<&crate::providers::Provider> for ProviderSummary {
    fn from(p: &crate::providers::Provider) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            base_url: p.base_url.clone(),
            key_env: p.key_env.clone(),
            key_set: p.api_key().is_some(),
            key_usable: daemon_key_usable(p),
            models: p.models.len(),
        }
    }
}

/// Whether this daemon would spend its own key for a request that
/// presents none — the same two conditions `resolve_remote_target`
/// applies, minus the per-request cross-site check no CLI can trip.
fn daemon_key_usable(provider: &crate::providers::Provider) -> bool {
    provider.api_key().is_some() && crate::daemon::reachable_only_locally()
}

/// `GET /llmman/providers`.
#[derive(Serialize)]
struct ProvidersResponse {
    providers: Vec<ProviderSummary>,
}

/// `GET /llmman/providers/:id` — one provider, with its models.
#[derive(Serialize)]
struct ProviderResponse {
    id: String,
    name: String,
    base_url: String,
    key_env: String,
    key_set: bool,
    /// See [`ProviderSummary::key_usable`].
    key_usable: bool,
    models: Vec<ProviderModelResponse>,
}

/// One model in a [`ProviderResponse`].
#[derive(Serialize)]
struct ProviderModelResponse {
    id: String,
    /// Absent, not zero, where models.dev publishes no price: printing
    /// "unknown" as "free" lies about someone's bill.
    #[serde(skip_serializing_if = "Option::is_none")]
    cost: Option<ProviderCostResponse>,
}

/// US dollars per million tokens, models.dev's own unit (see
/// [`crate::providers::Cost`]).
#[derive(Serialize)]
struct ProviderCostResponse {
    input: f64,
    output: f64,
}

impl From<&crate::providers::Provider> for ProviderResponse {
    fn from(p: &crate::providers::Provider) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            base_url: p.base_url.clone(),
            key_env: p.key_env.clone(),
            key_set: p.api_key().is_some(),
            key_usable: daemon_key_usable(p),
            models: p
                .models
                .iter()
                .map(|m| ProviderModelResponse {
                    id: m.id.clone(),
                    cost: m.cost.map(|c| ProviderCostResponse {
                        input: c.input,
                        output: c.output,
                    }),
                })
                .collect(),
        }
    }
}

/// `GET /llmman/providers` — every provider `--provider` accepts.
async fn handle_llmman_providers() -> Result<impl IntoResponse, AppError> {
    let catalog = provider_catalog().await?;
    Ok(Json(ProvidersResponse {
        providers: catalog.iter().map(ProviderSummary::from).collect(),
    }))
}

/// `GET /llmman/providers/:id` — one provider, or a 404 naming
/// near-matches (see [`crate::providers::unknown_provider_error`]).
async fn handle_llmman_provider(
    UrlPath(id): UrlPath<String>,
) -> Result<impl IntoResponse, AppError> {
    let catalog = provider_catalog().await?;
    let provider = catalog.get(&id).ok_or_else(|| {
        AppError(
            crate::providers::unknown_provider_error(&id, &catalog),
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(Json(ProviderResponse::from(provider)))
}

/// Ollama's GET /api/version, extended with this daemon's own identity —
/// executable path (canonicalized at startup) and pid — so a client can
/// tell whether a daemon it found listening still belongs to a live
/// install (the exe still exists, and is the binary the client would
/// launch) and stop/replace it if not. See daemon::ensure_server.
async fn handle_version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("LLMMAN_VERSION"),
        "exe": state.0.exe.as_ref().map(|p| p.to_string_lossy()),
        "pid": std::process::id(),
    }))
}

async fn handle_tags(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let models = list
        .into_iter()
        .map(|img| OllamaModelInfo {
            name: img.reference.clone(),
            model: img.reference,
            size: img.size,
            digest: img.digest,
            modified_at: img
                .modified_at
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(now_rfc3339),
            details: OllamaModelDetails {
                format: "gguf".into(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        })
        .collect();
    Ok(Json(OllamaTagsResponse { models }))
}

/// The subset of a [`RunningModel`] `handle_ps` needs, cloned out while
/// holding `manager`'s lock (see `handle_ps`) so the per-model `/props`
/// round trips afterward don't hold that lock for the duration.
struct PsEntry {
    name: String,
    digest: String,
    size: u64,
    port: u16,
    pid: Option<u32>,
    processor: String,
    started_at: String,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

async fn handle_ps(State(state): State<AppState>) -> impl IntoResponse {
    let entries: Vec<PsEntry> = {
        let mgr = state.0.manager.lock().await;
        mgr.running
            .iter()
            .map(|(name, m)| PsEntry {
                name: name.clone(),
                digest: m.digest.clone(),
                size: m.size,
                port: m.port,
                pid: m.pid(),
                processor: m.processor(),
                started_at: m.started_at.clone(),
                expires_at: m
                    .keep_alive
                    .and_then(|d| chrono::Duration::from_std(d).ok())
                    .map(|d| m.last_active_wall + d),
            })
            .collect()
    };

    let mut models = Vec::with_capacity(entries.len());
    for entry in entries {
        let context_length = query_context_length(&state.0.client, entry.port).await;
        models.push(OllamaRunningModelInfo {
            name: entry.name.clone(),
            model: entry.name,
            digest: entry.digest,
            size: entry.size,
            size_vram: 0, // not tracked — see RunningModel::processor's doc comment
            pid: entry.pid,
            port: entry.port,
            processor: entry.processor,
            context_length,
            started_at: entry.started_at,
            expires_at: entry
                .expires_at
                .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        });
    }
    Json(OllamaPsResponse { models })
}

/// Best-effort live context-length lookup via the running llama-server's own
/// `/props` endpoint (`default_generation_settings.n_ctx`) — mirrors
/// Ollama's own preference for live runner data over anything cached (see
/// server.PsHandler's use of `v.llama.ContextLength()`). Returns `None` on
/// any failure (short timeout, connection error, unexpected shape, or a
/// vllm-backed model, which doesn't expose this endpoint at all) rather
/// than failing the whole `ps` response over one unreachable model.
async fn query_context_length(client: &Client, port: u16) -> Option<u64> {
    let url = format!("http://127.0.0.1:{port}/props");
    let resp = client
        .get(&url)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()
}

async fn handle_show(
    State(state): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> Result<impl IntoResponse, AppError> {
    // ollama sends either {"name":"..."} or {"model":"..."} depending on call site;
    // filter out empty strings so we always fall back to whichever field is populated.
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // A provider-routed model is served by someone else and is never in
    // the local store, so the lookup below would report it missing and
    // send every caller that treats a 500 here as "needs pulling" — most
    // of all `daemon::ensure_model_pulled`, which `llmman launch` and
    // `llmman run` both call before their first request — off to pull a
    // reference that names no registry. Answer for it directly instead.
    if crate::providers::is_remote_ref(model_ref) {
        eprintln!("[llmman] /api/show model={model_ref:?} (provider-routed)");
        return Ok(Json(OllamaShowResponse {
            model_info: serde_json::json!({ "digest": "", "size": 0 }),
            details: OllamaModelDetails {
                // Not "gguf": there are no local weights here at all, and
                // claiming a format llmman never inspected would be a
                // guess about someone else's serving stack.
                format: String::new(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        }));
    }
    // Resolve the same way handle_pull stored it — otherwise a bare name
    // (e.g. "gemma4", pulled and stored as "docker.io/ai/gemma4") would
    // never be found by show/delete even though it's in the local store.
    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    let model_ref = model_ref.as_str();
    eprintln!("[llmman] /api/show model={model_ref:?}");
    let store = OciStore::open(&state.0.store_path)?;
    let desc = store.find(model_ref).map_err(|_| {
        AppError(
            anyhow!("model not found: {model_ref}"),
            StatusCode::INTERNAL_SERVER_ERROR,
        )
    })?;
    Ok(Json(OllamaShowResponse {
        model_info: serde_json::json!({ "digest": desc.digest, "size": desc.size }),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: String::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
    }))
}

// -- Ollama /api/pull ---------------------------------------------------------
// Mirrors `ollama.PullHandler`: streams newline-delimited JSON status objects
// (`{"status": "..."}`, matching api.ProgressResponse) ending in either
// `{"status": "success"}` or `{"error": "..."}`. Real Ollama also reports
// per-layer `digest`/`total`/`completed` fields for a byte-level progress
// bar; the Go shim's `llmman_pull` is a single opaque blocking call with no
// progress callback, so this reports coarse status only — every field is
// `omitempty` on the client side, so callers that only render `status` (as
// `llmman pull`'s own CLI progress text does) see accurate text throughout.

#[derive(Debug, Deserialize)]
struct OllamaPullRequest {
    #[serde(default)]
    model: String,
    // Real Ollama keeps `Name` as a deprecated fallback for `Model`
    // (server/routes.go's `cmp.Or(req.Model, req.Name)`) — some clients
    // only ever send `name`, which used to 422 outright since `model`
    // was required. Falls back below like handle_show/handle_delete
    // already do.
    #[serde(default)]
    name: String,
}

async fn handle_pull(
    State(state): State<AppState>,
    Json(req): Json<OllamaPullRequest>,
) -> Result<Response, AppError> {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model_ref.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "model is required",
        ));
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    eprintln!("[llmman] /api/pull model={model:?}");
    let store_path = state.0.store_path.clone();

    let already_present = OciStore::open(&store_path)
        .and_then(|s| s.find(&model))
        .is_ok();
    if already_present {
        let line = serde_json::json!({"status": "success"}).to_string() + "\n";
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-ndjson")
            .body(Body::from(line))
            .unwrap());
    }

    // Not in the local store: actually pull it (the previous behavior only
    // ever 404'd here, so no real Ollama client's "pull if missing, then
    // use" flow — e.g. `ollama run <model>` — ever worked against llmman).
    //
    // pull_serialized (not a bare crate::ffi::pull call) re-checks presence
    // after acquiring PULL_LOCK: this request's own `already_present` check
    // above ran before that wait, so a concurrent pull of the same model
    // (from another client, or from ensure_model's own fallback below) can
    // finish while this one was waiting its turn — see PULL_LOCK's doc
    // comment for why two callers must never invoke the actual FFI pull at
    // the same time.
    let model_for_task = model.clone();
    let pull_task =
        tokio::task::spawn_blocking(move || pull_serialized(&store_path, &model_for_task));

    Ok(stream_ffi_progress(
        model,
        "pull",
        "pulling manifest",
        pull_task,
    ))
}

// -- Ollama /api/push ---------------------------------------------------------
// Ollama's own /api/push has no equivalent in llmman's original design (the
// route didn't exist at all before), but it's the same shape as /api/pull —
// a streamed NDJSON status sequence — so `llmman push` becoming a thin
// client of this endpoint (like `llmman pull`) gets both operations onto
// the exact same Ollama-protocol wire format.

#[derive(Debug, Deserialize)]
struct OllamaPushRequest {
    #[serde(default)]
    model: String,
    // See OllamaPullRequest's `name` field doc comment: same deprecated
    // `Name`-falls-back-to-`Model` shape as real Ollama's PushRequest.
    #[serde(default)]
    name: String,
}

async fn handle_push(
    State(state): State<AppState>,
    Json(req): Json<OllamaPushRequest>,
) -> Result<Response, AppError> {
    let model_ref = if req.model.is_empty() {
        req.name.as_str()
    } else {
        req.model.as_str()
    };
    if model_ref.is_empty() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            "model is required",
        ));
    }
    let model = crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    eprintln!("[llmman] /api/push model={model:?}");
    let store_path = state.0.store_path.clone();

    // Unlike pull, there's nothing sensible to do if the model isn't
    // already in the local store: push has no "fetch it first" fallback.
    if OciStore::open(&store_path)
        .and_then(|s| s.find(&model))
        .is_err()
    {
        return Err(AppError::status(
            StatusCode::NOT_FOUND,
            format!("model not found: {model}"),
        ));
    }

    // See MODEL_LOCKS' doc comment: a push shares the same Go-side
    // progressState entry (keyed by this model reference) as a pull of
    // the same model, so they need the same per-model mutual exclusion —
    // but a push of one model no longer blocks a pull/push of another.
    let model_for_task = model.clone();
    let push_task = tokio::task::spawn_blocking(move || {
        let lock = model_lock(&model_for_task);
        let result = (|| {
            let _guard = lock.blocking_lock();
            let layout_dir = store_path
                .to_str()
                .ok_or_else(|| anyhow!("store path is not valid UTF-8"))?;
            crate::ffi::push(layout_dir, &model_for_task)
        })();
        drop(lock);
        release_model_lock(&model_for_task);
        result
    });

    Ok(stream_ffi_progress(
        model,
        "push",
        "retrieving manifest",
        push_task,
    ))
}

/// Runs `task` (a blocking FFI call already dispatched via spawn_blocking)
/// to completion, streaming an immediate `first_status` line, then polling
/// `ffi::progress(&model)` every 200ms (matching the Go shim's own mpb
/// refresh rate) until the task finishes, then a final `{"status": "success"}` or
/// `{"error": ...}` line. Shared by handle_pull and handle_push.
///
/// Each polled line includes real `total`/`completed` byte counts (mirroring
/// Ollama's own api.ProgressResponse fields) once the shim's shared
/// `progressState` (go-shim/progress_state.go) has learned a nonzero total
/// — before that, or if the FFI call is a kind that doesn't track
/// byte-level progress at all, only `status` text is included, exactly
/// like the old heartbeat-only version of this function. This is what
/// lets `llmman pull`/`llmman push` render a real progress bar instead of
/// just printing status text: the Go shim's own mpb bars
/// (go-shim/shared_oci.go) already draw real bars for these exact
/// numbers, but only reach an interactive terminal when the FFI call runs
/// in the foreground CLI process (e.g. `llmman transfer`) — here it runs
/// inside the daemon, whose stdio is redirected to a log file (see
/// daemon::ensure_server), so polling and relaying over this NDJSON
/// stream is the only way those numbers reach `llmman pull`/`llmman push`.
fn stream_ffi_progress(
    model: String,
    verb: &'static str,
    first_status: &'static str,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
) -> Response {
    let first_line = serde_json::json!({"status": first_status}).to_string() + "\n";
    let stream = futures::stream::once(futures::future::ready(Bytes::from(first_line)))
        .chain(futures::stream::unfold(Some(task), move |task| {
            let model = model.clone();
            async move {
                let mut task = task?;
                tokio::select! {
                    result = &mut task => {
                        let line = match result {
                            Ok(Ok(())) => serde_json::json!({"status": "success"}).to_string(),
                            Ok(Err(e)) => serde_json::json!({"error": format!("{e:#}")}).to_string(),
                            Err(e) => serde_json::json!({"error": format!("{verb} task panicked: {e}")}).to_string(),
                        };
                        Some((Bytes::from(line + "\n"), None))
                    }
                    _ = sleep(Duration::from_millis(200)) => {
                        // A HuggingFace pull tracks its own progress natively
                        // (crate::hf::progress) rather than through the Go
                        // shim's — check that first, since only one of the
                        // two will ever actually be tracking `model` for a
                        // given task.
                        let rust_snap = crate::hf::progress::poll(&model);
                        let go_snap = (rust_snap.total == 0).then(|| crate::ffi::progress(&model).ok()).flatten();
                        let (status, total, completed) = if rust_snap.total > 0 {
                            (rust_snap.status, rust_snap.total, rust_snap.completed)
                        } else if !rust_snap.status.is_empty() {
                            (rust_snap.status, 0, 0)
                        } else if let Some(p) = &go_snap {
                            (p.status.clone(), p.total, p.completed)
                        } else {
                            (String::new(), 0, 0)
                        };
                        let line = if total > 0 {
                            serde_json::json!({
                                "status": if status.is_empty() { format!("{verb}ing {model}") } else { status },
                                "total": total.max(0),
                                "completed": completed.clamp(0, total),
                            })
                        } else if !status.is_empty() {
                            serde_json::json!({"status": status})
                        } else {
                            serde_json::json!({"status": format!("{verb}ing {model}")})
                        };
                        Some((Bytes::from(line.to_string() + "\n"), Some(task)))
                    }
                }
            }
        }))
        .map(Ok::<_, std::convert::Infallible>);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn handle_delete(
    State(state): State<AppState>,
    Json(req): Json<OllamaDeleteRequest>,
) -> Result<impl IntoResponse, AppError> {
    let model_ref = req
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&req.model);
    // See handle_show: resolve the same way handle_pull stored it.
    let model_ref =
        crate::shortnames::resolve_ollama_api(model_ref).map_err(AppError::bad_request)?;
    let store = OciStore::open(&state.0.store_path)?;
    store.remove(&model_ref)?;
    Ok(StatusCode::OK)
}

// -- Ollama /api/chat ---------------------------------------------------------

/// The body ollama answers a message-less `/api/chat` with. `Default` for
/// the rest of `OllamaMessage`, not explicit `None`s: every other field is
/// `skip_serializing_if`, so this reaches the wire as the bare
/// `{"role":"assistant","content":""}` ollama 0.32.6 sends.
fn empty_chat_chunk(model: String, done_reason: &str) -> OllamaChatChunk {
    OllamaChatChunk {
        model,
        created_at: now_rfc3339(),
        message: OllamaMessage {
            role: "assistant".into(),
            ..Default::default()
        },
        done: true,
        done_reason: Some(done_reason.into()),
    }
}

async fn handle_ollama_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/chat model={:?} messages={}",
        req.model,
        req.messages.len()
    );

    // Empty messages are ollama's unload request when paired with
    // `keep_alive: 0`, and its load-only request on their own — the same
    // two short-circuits `handle_ollama_generate` already implements for
    // an empty `prompt`, which cites ollama's own `server/routes.go` for
    // both; its `ChatHandler` carries the pair as well. Without them an
    // empty `messages` array reaches `stream_ollama`, which asks the
    // backend to continue from nothing: the caller gets a real, arbitrary
    // generation and `done_reason: "stop"` where ollama answers with an
    // empty message, and a `keep_alive: 0` never unloads anything.
    if req.messages.is_empty() {
        if is_explicit_unload(&req.keep_alive) {
            unload_model(&state, &req.model).await?;
            return Ok(Json(empty_chat_chunk(req.model, "unload")).into_response());
        }

        let (_model, _target, guard) = ensure_model(&state, &req.model, Some(&headers)).await?;
        refresh_activity(guard, resolve_keep_alive(&req.keep_alive)).await;
        return Ok(Json(empty_chat_chunk(req.model, "load")).into_response());
    }

    let (model, target, guard) = ensure_model(&state, &req.model, Some(&headers)).await?;
    let keep_alive = resolve_keep_alive(&req.keep_alive);
    let activity = begin_activity(guard, Some(keep_alive)).await;
    // See backend_wire_model's own doc comment — usually just `model`
    // itself, but a different value for an Engine::Mlx backend or a
    // remote provider. Only this one outgoing request field, never the
    // response chunk's own `model` field below (which must keep echoing
    // back `model` as-is).
    let wire_model = backend_wire_model(&state, &target, &model).await;
    let oai = OAIChatRequest {
        model: wire_model,
        messages: req.messages.iter().map(ollama_message_to_oai).collect(),
        stream: true,
        temperature: opt_f64(&req.options, "temperature"),
        top_p: opt_f64(&req.options, "top_p"),
        max_tokens: opt_u32(&req.options, "num_predict"),
        // No `.or(Some(DEFAULT_REPEAT_PENALTY))` here — post_chat (the
        // only place this request actually reaches llama-server) resolves
        // that default itself now. See apply_default_repeat_penalty_typed.
        repeat_penalty: opt_f64(&req.options, "repeat_penalty"),
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
        tools: req.tools.clone(),
        response_format: format_to_response_format(&req.format),
    };
    stream_ollama(
        req.stream,
        state.0.client.clone(),
        target,
        oai,
        activity,
        move |content, thinking, tool_calls, done| {
            let done_reason = done.then(|| {
                if tool_calls.is_some() {
                    "tool_calls".to_string()
                } else {
                    "stop".to_string()
                }
            });
            OllamaChatChunk {
                model: model.clone(),
                created_at: now_rfc3339(),
                message: OllamaMessage {
                    role: "assistant".into(),
                    content,
                    thinking,
                    tool_calls,
                    images: None,
                    tool_name: None,
                },
                done,
                done_reason,
            }
        },
    )
    .await
}

// -- Ollama /api/generate -----------------------------------------------------

async fn handle_ollama_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, AppError> {
    eprintln!(
        "[llmman] /api/generate model={:?} prompt_len={}",
        req.model,
        req.prompt.len()
    );

    // Empty prompt + keep_alive:0 = unload request (ollama server/routes.go:354).
    // is_explicit_unload, not resolve_keep_alive: it still accepts every
    // zero form — the JSON number 0, but also "0"/"0s"/etc as a string —
    // without treating an absent field as one, which under
    // `LLMMAN_KEEP_ALIVE=0` turned a plain preload into an eviction.
    let is_unload = req.prompt.is_empty() && is_explicit_unload(&req.keep_alive);
    if is_unload {
        unload_model(&state, &req.model).await?;
        return Ok(Json(OllamaGenerateChunk {
            model: req.model,
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("unload".into()),
        })
        .into_response());
    }

    let (model, target, guard) = ensure_model(&state, &req.model, Some(&headers)).await?;
    // Empty prompt = load-only request (mirrors ollama server/routes.go:429)
    // — including "preload with a custom keep_alive", so refresh it here
    // even though no generation is happening.
    if req.prompt.is_empty() {
        refresh_activity(guard, resolve_keep_alive(&req.keep_alive)).await;
        return Ok(Json(OllamaGenerateChunk {
            model: req.model,
            created_at: now_rfc3339(),
            response: String::new(),
            thinking: None,
            done: true,
            done_reason: Some("load".into()),
        })
        .into_response());
    }

    let keep_alive = resolve_keep_alive(&req.keep_alive);
    let activity = begin_activity(guard, Some(keep_alive)).await;
    // See backend_wire_model's own doc comment.
    let wire_model = backend_wire_model(&state, &target, &model).await;
    let oai = OAIChatRequest {
        model: wire_model,
        messages: vec![OAIMessage::text("user", req.prompt.clone())],
        stream: true,
        temperature: opt_f64(&req.options, "temperature"),
        top_p: opt_f64(&req.options, "top_p"),
        max_tokens: opt_u32(&req.options, "num_predict"),
        // No `.or(Some(DEFAULT_REPEAT_PENALTY))` here — post_chat (the
        // only place this request actually reaches llama-server) resolves
        // that default itself now. See apply_default_repeat_penalty_typed.
        repeat_penalty: opt_f64(&req.options, "repeat_penalty"),
        chat_template_kwargs: think_to_chat_template_kwargs(&req.think),
        tools: None,
        response_format: format_to_response_format(&req.format),
    };
    stream_ollama(
        req.stream,
        state.0.client.clone(),
        target,
        oai,
        activity,
        move |response, thinking, _tool_calls, done| OllamaGenerateChunk {
            model: model.clone(),
            created_at: now_rfc3339(),
            response,
            thinking,
            done,
            done_reason: done.then_some("stop".into()),
        },
    )
    .await
}

// -- OpenAI pass-through handlers --------------------------------------------

async fn handle_openai_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let store = OciStore::open(&state.0.store_path)?;
    let list = store.list()?;
    let mgr = state.0.manager.lock().await;
    let data: Vec<serde_json::Value> = list
        .into_iter()
        .map(|img| {
            let loaded = mgr.running.contains_key(&img.reference);
            serde_json::json!({
                "id": img.reference,
                "object": "model",
                "created": 0,
                "owned_by": "llmman",
                // status field consumed by the web UI to track loaded/unloaded state
                "status": { "value": if loaded { "loaded" } else { "unloaded" } },
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

/// Sets `repeat_penalty` to `DEFAULT_REPEAT_PENALTY` on `req` (an
/// OpenAI-shaped chat/completions request body) unless the caller already
/// supplied its own value. Every other entry point — `/api/chat`,
/// `/api/generate`, and the Anthropic Messages API — already forwards this
/// same default to llama-server via `post_chat` (see
/// `DEFAULT_REPEAT_PENALTY`'s doc comment for the value itself); a plain
/// OpenAI-compatible client has no llmman-specific reason to know it
/// should set this itself, so `proxy_openai_generation` applies it here
/// too, keeping every generation-capable API surface's behavior
/// consistent instead of leaving this one raw-passthrough path the sole
/// exception.
fn apply_default_repeat_penalty(req: &mut serde_json::Value) {
    if req.get("repeat_penalty").is_none() {
        req["repeat_penalty"] = serde_json::json!(DEFAULT_REPEAT_PENALTY);
    }
}

/// Shared setup for every plain OpenAI-passthrough route: parse just
/// enough of the request to find `model`, make sure it's loaded, rewrite
/// `model` to its canonical name (see `ensure_model`), and open an
/// activity guard for it. `proxy_openai_generation` and
/// `proxy_openai_passthrough` below each finish shaping the parsed body
/// their own way (the former also defaults `repeat_penalty`, the latter
/// doesn't) before actually proxying it through.
///
/// The returned `Option<String>` is `Some(canonical_model)` only when
/// the backend actually needed a different wire name than the one the
/// client asked for (see `backend_wire_model`'s own doc comment — an
/// `Engine::Mlx` backend, or any remote provider, which knows the model
/// by its own unprefixed id) — the signal
/// `proxy_openai_generation`/`proxy_openai_passthrough` need to decide
/// whether the *response* also needs its own `"model"` field rewritten
/// back before reaching the client, via `proxy_rewriting_model`/
/// `stream_rewriting_model` instead of the plain `proxy`.
async fn resolve_openai_request(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(serde_json::Value, Target, ActivityGuard, Option<String>), AppError> {
    let mut req: serde_json::Value =
        serde_json::from_slice(&body).context("parse OpenAI request body")?;
    let model = req["model"].as_str().unwrap_or("").to_string();
    let (model, target, guard) = ensure_model(state, &model, Some(headers)).await?;
    // The OpenAI-compatible surface has no `keep_alive` field of its own
    // (real Ollama's doesn't either) — `None` leaves whatever this model
    // already has untouched (its load-time default, or an explicit value
    // pinned via `/api/chat`) rather than overwriting it, e.g. clobbering
    // a `keep_alive: -1` ("never unload") pin with the daemon default the
    // instant one OpenAI-compatible request comes in.
    let activity = begin_activity(guard, None).await;
    // See backend_wire_model's own doc comment — usually just `model`
    // itself, but a different value for an Engine::Mlx backend or a
    // remote provider.
    let wire_model = backend_wire_model(state, &target, &model).await;
    let response_model_override = (wire_model != model).then_some(model);
    req["model"] = serde_json::Value::String(wire_model);
    Ok((req, target, activity, response_model_override))
}

/// OpenAI-passthrough for the endpoints that actually generate tokens —
/// chat completions, legacy completions, and the Responses API endpoint
/// Codex uses. Always defaults `repeat_penalty` (see
/// `apply_default_repeat_penalty`) rather than taking a bool flag callers
/// could forget to set: whether a route defaults this is now a choice of
/// *which function* it calls (this one, or `proxy_openai_passthrough`
/// below for the two non-generation routes), not an easily-mis-set
/// argument at the call site.
///
/// Picks which of the three proxy helpers actually relays the response
/// based on `resolve_openai_request`'s `response_model_override`: plain
/// `proxy` (untouched byte relay) when it's `None` — every engine
/// except `Engine::Mlx`, unchanged from before that engine existed —
/// otherwise `stream_rewriting_model`/`proxy_rewriting_model` depending
/// on whether this request itself asked for a streamed response, both
/// of which rewrite the response's own `"model"` field back to the
/// canonical name before it reaches the client (see either's own doc
/// comment for why that's needed at all).
async fn proxy_openai_generation(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let (mut req, target, activity, response_model_override) =
        resolve_openai_request(state, headers, body).await?;
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    if repeat_penalty_applies(&target) {
        apply_default_repeat_penalty(&mut req);
    } else {
        // See repeat_penalty_applies: not an OpenAI field, and a strict
        // provider rejects the whole request over it.
        if let Some(req) = req.as_object_mut() {
            req.remove("repeat_penalty");
        }
    }
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) if streaming => {
            stream_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                canonical,
            )
            .await
        }
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// OpenAI-passthrough for the routes that don't generate anything a
/// repeat penalty could apply to — embeddings, and the Responses
/// token-counting endpoint. Same model-loading/canonicalization as
/// `proxy_openai_generation` (see `resolve_openai_request`), minus the
/// `repeat_penalty` default. Neither route has a `stream` concept of
/// its own, so unlike that function this only ever needs `proxy` or
/// `proxy_rewriting_model` — never the streaming variant.
///
/// `/v1/embeddings` specifically gets two more checks around that:
/// `Engine::Mlx` is never started with `mlx_lm.server`'s own
/// `--embedding-model` flag (`spawn_mlx_server` has no way to know which
/// model a caller would even want for that), so its conditional
/// `/v1/embeddings` route is never registered there at all — forwarding
/// anyway would just surface its own bare, unexplained 404, so this
/// fails fast with [`mlx_embeddings_unsupported_response`] instead,
/// twice over:
///
///   1. Before `resolve_openai_request` (and so `ensure_model`) ever
///      runs, via [`would_use_mlx`]'s own cheap, spawn-free check —
///      covering the overwhelmingly common case of a model that's
///      already resolvable locally, so a repeated embeddings request
///      against it never pays for spawning `mlx_lm.server` and loading
///      however many GB of weights for a request that could never
///      succeed there anyway.
///   2. After, via `response_model_override` — the fallback for the one
///      case (1) can't cheaply rule out ahead of time: a model that
///      wasn't in the local store at all yet, so `ensure_model` above
///      just pulled and loaded it for the first (and, thanks to (1),
///      only ever) time, and it turned out to be `Engine::Mlx` after
///      all.
async fn proxy_openai_passthrough(
    state: &AppState,
    headers: &HeaderMap,
    body: Bytes,
    llama_path: &str,
) -> Result<Response, AppError> {
    let embeddings = llama_path == "/v1/embeddings";
    if embeddings {
        if let Ok(peek) = serde_json::from_slice::<serde_json::Value>(&body) {
            if let Some(model_ref) = peek["model"].as_str() {
                // A provider-routed model is never an `Engine::Mlx` one —
                // it isn't local at all — so skip the whole check rather
                // than letting `would_use_mlx` do a pointless store lookup
                // against a reference that was never a store reference.
                if !crate::providers::is_remote_ref(model_ref) {
                    if let Some(canonical) = would_use_mlx(state, model_ref).await {
                        return Ok(mlx_embeddings_unsupported_response(&canonical));
                    }
                }
            }
        }
    }
    let (mut req, target, activity, response_model_override) =
        resolve_openai_request(state, headers, body).await?;
    if is_responses_route(llama_path) && !target.is_remote() {
        sanitize_responses_request(&mut req);
    }
    // `!target.is_remote()`, not just `response_model_override.is_some()`:
    // a remote target *always* sets that override (it addresses the model
    // by its own unprefixed id — see `backend_wire_model`), so without
    // this every provider-routed embeddings request would be rejected as
    // an MLX one. Only a local backend can actually be `Engine::Mlx`.
    if embeddings && !target.is_remote() {
        if let Some(canonical) = &response_model_override {
            drop(activity);
            return Ok(mlx_embeddings_unsupported_response(canonical));
        }
    }
    let body = Bytes::from(serde_json::to_vec(&req).context("re-serialize OpenAI request body")?);
    let resp = match response_model_override {
        Some(canonical) => {
            proxy_rewriting_model(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
                &canonical,
            )
            .await
        }
        None => {
            proxy(
                &state.0.client,
                &target,
                llama_path,
                headers,
                body,
                activity,
            )
            .await
        }
    }?;
    // `/v1/responses/input_tokens` comes through here, and a provider
    // without the Responses API 404s it just the same.
    Ok(explain_missing_route(&target, llama_path, resp))
}

/// The clear, specific error [`proxy_openai_passthrough`] returns
/// instead of forwarding a `/v1/embeddings` request on to an
/// `Engine::Mlx` backend — see that function's own doc comment on why
/// that request could never succeed there anyway.
fn mlx_embeddings_unsupported_response(canonical_model: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "{canonical_model} is served by mlx_lm.server, which llmman never starts with \
                 --embedding-model — /v1/embeddings isn't supported for it; use a GGUF or \
                 vllm-served model for embeddings instead"
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

async fn handle_openai_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/chat/completions").await
}

async fn handle_openai_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/completions").await
}

async fn handle_openai_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_passthrough(&state, &headers, body, "/v1/embeddings").await
}

// -- OpenAI Audio Transcriptions API (/v1/audio/transcriptions) -------------
//
// llama-server has its own native implementation (requires the model to
// be loaded with mtmd audio support via a companion --mmproj — see
// ModelPath::mmproj), so this is a plain pass-through like
// handle_openai_responses. The request body is multipart/form-data, not
// JSON, so resolve_openai_request's "parse as JSON to find model" doesn't apply —
// multipart_text_field below extracts just the model field instead.

/// Axum's own default `DefaultBodyLimit` (2 MiB) is well under a typical
/// audio file's size — real recordings routinely run tens of MiB — so
/// both transcription routes below opt out of it in favor of this
/// higher cap instead of disabling it outright.
const TRANSCRIPTION_BODY_LIMIT_BYTES: usize = 200 * 1024 * 1024;

/// Extracts a top-level form field's text value from a
/// `multipart/form-data` body, or `None` if not multipart / no boundary /
/// field not found.
async fn multipart_text_field(
    body: &Bytes,
    headers: &HeaderMap,
    field_name: &str,
) -> Option<String> {
    let content_type = headers.get("content-type")?.to_str().ok()?;
    let boundary = multer::parse_boundary(content_type).ok()?;
    // Single-chunk stream over a cheap Bytes clone — the body is already
    // fully buffered, so there's nothing to actually stream.
    let stream = futures::stream::once(async { Ok::<_, std::io::Error>(body.clone()) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some(field_name) {
            return field.text().await.ok();
        }
    }
    None
}

async fn handle_openai_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let Some(model) = multipart_text_field(&body, &headers, "model")
        .await
        .filter(|m| !m.is_empty())
    else {
        // A malformed request, not a server-side failure — matches
        // handle_pull's own "missing required field" convention instead
        // of AppError's blanket 500.
        let body = serde_json::json!({
            "error": "transcription request is missing a required \"model\" form field"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    };
    // Every other surface rewrites `model` to the provider's own id
    // before forwarding, but this body is multipart, not JSON: the raw
    // relay below would hand the provider a reference it has never heard
    // of. Refuse in llmman's own words rather than let that surface as
    // someone else's "unknown model".
    if crate::providers::is_remote_ref(&model) {
        let body = serde_json::json!({
            "error": "llmman does not route /v1/audio/transcriptions to a provider — \
                      use a locally served model with audio support"
        });
        return Ok((StatusCode::BAD_REQUEST, Json(body)).into_response());
    }
    let (_, target, guard) = ensure_model(&state, &model, Some(&headers)).await?;
    // No `keep_alive` field on this API surface either — see
    // resolve_openai_request's own comment on the same choice.
    let activity = begin_activity(guard, None).await;
    proxy(
        &state.0.client,
        &target,
        "/v1/audio/transcriptions",
        &headers,
        body,
        activity,
    )
    .await
}

// -- OpenAI Responses API (/v1/responses) ------------------------------------
//
// llama-server (llama.cpp) has its own native /v1/responses implementation
// that converts a Responses-API request into a Chat Completions request
// internally (see server_chat_convert_responses_to_chatcmpl in
// tools/server/server-chat.cpp) — including the exact SSE event sequence
// Codex requires (response.created -> response.output_item.added ->
// response.output_text.delta -> ... -> response.completed, no `[DONE]`) and
// re-mapping of tool_calls into function_call output items. Re-implementing
// that translation here would just duplicate — and risk drifting out of
// sync with — llama.cpp's own logic, so this is a plain pass-through
// exactly like the other /v1/* routes above, apart from
// filter_non_function_tools (see its own doc comment) below.
async fn handle_openai_responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_openai_generation(&state, &headers, body, "/v1/responses").await
}

async fn handle_openai_responses_input_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    // A token-counting call, not a generation request — repeat_penalty has
    // nothing to apply to here.
    proxy_openai_passthrough(&state, &headers, body, "/v1/responses/input_tokens").await
}

/// Applies both `/v1/responses` request-shape workarounds below, to a
/// body already parsed by `resolve_openai_request`.
///
/// Local targets only, and applied after the target is known rather than
/// on the way in: both are workarounds for what *llama-server's*
/// `/v1/responses` cannot accept. A provider that implements the
/// Responses API natively accepts the request Codex actually sent, and
/// forwarding a stripped one would silently cost it `web_search` and
/// every other non-function tool.
fn sanitize_responses_request(req: &mut serde_json::Value) {
    filter_non_function_tools(req);
    consolidate_responses_instructions(req);
}

/// The routes [`sanitize_responses_request`] applies to.
fn is_responses_route(llama_path: &str) -> bool {
    llama_path.starts_with("/v1/responses")
}

/// Explains a provider's bare 404 on the Responses API.
///
/// Being OpenAI-wire-format does not mean implementing every OpenAI
/// route. `/v1/responses` is the split that matters, because it is the
/// only one Codex uses: `openai`, `groq` and `openrouter` answer it,
/// `anthropic` and `mistral` 404. models.dev carries no capability data
/// to filter on, and a hardcoded list would be wrong the week a provider
/// shipped it — so this reports the 404 llmman actually received rather
/// than predicting it, and says which of the two things is missing.
fn explain_missing_route(target: &Target, route: &str, resp: Response) -> Response {
    if resp.status() != StatusCode::NOT_FOUND || !is_responses_route(route) {
        return resp;
    }
    // A 404 on any other route means something else entirely — an
    // unknown model on `/v1/chat/completions`, most often — and claiming
    // a missing Responses API for it would be a worse answer than the
    // provider's own.
    let Target::Remote(remote) = target else {
        return resp;
    };
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "provider {} has no {route} — it is OpenAI-compatible but does not \
                 implement the Responses API. Use an integration that speaks chat \
                 completions, or a provider that does (openai, groq, openrouter).",
                remote.provider
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

/// Strips any entry from the request's top-level `tools` array whose
/// `"type"` isn't `"function"` before proxying to llama-server.
///
/// Real Codex always includes Responses-API tool types llama-server's own
/// `/v1/responses` doesn't understand — a `"namespace"`-typed sub-agent
/// tool bundle, the bare `{"type":"web_search"}` entry, etc. — and, unlike
/// this module's other passthrough routes, llama-server hard-rejects the
/// *entire* request the moment even one such entry is present ("'type' of
/// tool must be 'function'"), rather than skipping just that entry. Since
/// Codex's own default toolset always includes at least one of these,
/// every real `codex`/`codex exec` invocation would 400 on its very first
/// turn without this filter. Nested sub-tools inside a dropped
/// `"namespace"` entry (e.g. its own agent-management functions) are
/// dropped along with it rather than hoisted to the top level: the local
/// model losing access to those secondary tools is harmless, whereas
/// guessing how to flatten them would risk silently changing their
/// semantics.
fn filter_non_function_tools(req: &mut serde_json::Value) {
    if let Some(tools) = req.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
    }
}

/// Folds every `developer`/`system`-role item out of the request's `input`
/// array into the top-level `instructions` string, removing them from
/// `input`, before proxying to llama-server.
///
/// llama-server's own `/v1/responses` → chat-completions conversion
/// (`server_chat_convert_responses_to_chatcmpl` in llama.cpp's
/// `tools/server/server-chat.cpp`) unconditionally prepends one
/// `system`-role chat message built from `instructions`, but otherwise
/// forwards every `input` item's `role` field untouched. A later,
/// model-agnostic pass in llama.cpp's own chat-template layer
/// (`workaround::map_developer_role_to_system` in `common/chat.cpp`) then
/// unconditionally rewrites *every* remaining `role: "developer"` message
/// to `role: "system"`, wherever it sits in the array, with no
/// repositioning or merging. Real Codex requests routinely carry a
/// `developer`-role item further into `input` (permissions/skills
/// instructions) alongside the top-level `instructions` string, which
/// after that rewrite leaves two `system`-role messages in the
/// chat-completions request llama-server builds — the second one not at
/// index 0, which strict chat templates (Qwen3.5's included) reject
/// outright with "System message must be at the beginning". This is a
/// confirmed, currently-unresolved upstream llama.cpp gap (e.g.
/// ggml-org/llama.cpp#20733, ggml-org/llama.cpp#23423; a fix was proposed
/// and abandoned in ggml-org/llama.cpp#20079) rather than anything this
/// module's own /v1/messages-style message-building does, so it can't be
/// fixed the same way — this route is a pass-through by design (see the
/// module doc comment above). Folding every developer/system input item
/// into `instructions` here instead keeps the request in a shape
/// llama-server can never turn into more than one system message,
/// regardless of that upstream gap.
fn consolidate_responses_instructions(req: &mut serde_json::Value) {
    let mut instructions = req
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if let Some(input) = req.get_mut("input").and_then(|v| v.as_array_mut()) {
        input.retain(|item| {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "developer" && role != "system" {
                return true;
            }
            if let Some(text) = responses_input_item_text(item) {
                if !text.is_empty() {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(&text);
                }
            }
            false
        });
    }

    if !instructions.is_empty() {
        req["instructions"] = serde_json::Value::String(instructions);
    }
}

/// Extracts the plain text of a Responses-API `input` message item —
/// `content` is either a bare string or an array of blocks (each with a
/// `"text"` field, e.g. `{"type":"input_text","text":"..."}`), the same
/// two shapes Anthropic's own message content takes (see
/// `AnthropicContent::as_text` above).
fn responses_input_item_text(item: &serde_json::Value) -> Option<String> {
    match item.get("content")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        ),
        _ => None,
    }
}

// -- Anthropic /v1/messages --------------------------------------------------

/// Merges every system-role turn in an Anthropic request into a single
/// leading system message, then appends every other message in order.
///
/// Real Claude Code doesn't confine system content to the top-level
/// `system` field: it also injects background reminders (available
/// agents/skills, etc.) as ordinary entries with `"role": "system"`
/// scattered later in `messages`, which the real Anthropic API accepts in
/// any position. llama.cpp's chat templates (Qwen's included) are far
/// stricter and raise "System message must be at the beginning" the
/// moment a `system` role appears anywhere but index 0 — which every
/// sufficiently long real Claude Code session eventually triggers.
/// Concatenating them here keeps every request llama.cpp-template-safe
/// regardless of where the client put its system-role content.
fn build_anthropic_messages(req: &AnthropicRequest) -> Vec<OAIMessage> {
    let mut system_text = String::new();
    if let Some(sys) = &req.system {
        system_text.push_str(&sys.as_text());
    }
    let mut messages: Vec<OAIMessage> = Vec::new();
    for m in &req.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        messages.push(OAIMessage::text(m.role.clone(), m.content.as_text()));
    }
    if !system_text.is_empty() {
        messages.insert(0, OAIMessage::text("system", system_text));
    }
    messages
}

async fn handle_anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AnthropicRequest>,
) -> Result<Response, AppError> {
    // Backend needs its canonical name (see ensure_model); the response
    // below still echoes req.model back, unchanged from before.
    let (canonical_model, target, guard) = ensure_model(&state, &req.model, Some(&headers)).await?;
    // The Anthropic Messages API has no `keep_alive` field of its own —
    // `None` leaves it untouched, same as the OpenAI-compatible surface
    // (see resolve_openai_request's own comment on why).
    let activity = begin_activity(guard, None).await;

    let messages = build_anthropic_messages(&req);

    // See backend_wire_model's own doc comment — usually just
    // canonical_model itself, but a different value for an Engine::Mlx
    // backend or a remote provider.
    let wire_model = backend_wire_model(&state, &target, &canonical_model).await;
    let mut oai = OAIChatRequest {
        model: wire_model,
        messages,
        stream: req.stream,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        // The Anthropic Messages API has no repeat_penalty concept of its
        // own to read an override from, so this is always `None` here —
        // post_chat (the only place this request actually reaches
        // llama-server) resolves DEFAULT_REPEAT_PENALTY itself. See
        // apply_default_repeat_penalty_typed.
        repeat_penalty: None,
        // Nor a `think` override — see think_to_chat_template_kwargs.
        chat_template_kwargs: None,
        tools: None,
        response_format: None,
    };

    if req.stream {
        stream_anthropic(state.0.client.clone(), target, oai, req.model, activity).await
    } else {
        // Goes through post_chat like every other typed request (see its
        // own doc comment) rather than posting directly, so this branch
        // also gets repeat_penalty defaulted instead of needing its own
        // copy of that logic.
        let resp = post_chat(&state.0.client, &target, &mut oai).await?;
        let body: serde_json::Value = resp.json().await.context("parse llama-server response")?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(Json(serde_json::json!({
            "id": format!("msg_{}", gen_id()),
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": content }],
            "model": req.model,
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }))
        .into_response())
    }
}

// ---------------------------------------------------------------------------
// Option extractors from Ollama options blob
// ---------------------------------------------------------------------------

fn opt_f64(opts: &Option<serde_json::Value>, key: &str) -> Option<f32> {
    opts.as_ref()?.get(key)?.as_f64().map(|f| f as f32)
}

fn opt_u32(opts: &Option<serde_json::Value>, key: &str) -> Option<u32> {
    opts.as_ref()?.get(key)?.as_u64().map(|n| n as u32)
}

// ---------------------------------------------------------------------------
// CORS — mirrors Ollama's gin-contrib/cors setup (AllowWildcard +
// AllowOrigins). `origin_matches` is this crate's own stand-in for its
// wildcard matching (tower-http has no glob support built in).
// ---------------------------------------------------------------------------

/// Default CORS origin patterns, matching Ollama's own hardcoded
/// localhost/127.0.0.1/0.0.0.0 set (minus its desktop-app-only schemes —
/// llmman has no desktop app).
fn default_allowed_origins() -> Vec<String> {
    let mut origins = Vec::new();
    for host in ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"] {
        for scheme in ["http", "https"] {
            origins.push(format!("{scheme}://{host}"));
            origins.push(format!("{scheme}://{host}:*"));
        }
    }
    origins
}

/// `LLMMAN_ORIGINS` (comma-separated, mirrors `OLLAMA_ORIGINS`) plus
/// [`default_allowed_origins`]'s fixed set, always.
fn allowed_origins_from_env() -> Vec<String> {
    let mut origins: Vec<String> = std::env::var("LLMMAN_ORIGINS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    origins.extend(default_allowed_origins());
    origins
}

/// A single `*` anywhere in `pattern` matches any substring there (e.g.
/// `https://*.example.com`, or a bare `*` for "allow everything") —
/// matches `gin-contrib/cors`'s own `AllowWildcard`, which Ollama's CORS
/// setup enables. A second `*` never matches (single-wildcard only, same
/// as that library). No `*` at all requires a byte-for-byte match.
fn origin_matches(origin: &str, pattern: &str) -> bool {
    match pattern.split_once('*') {
        Some((prefix, suffix)) if !suffix.contains('*') => {
            origin.len() >= prefix.len() + suffix.len()
                && origin.starts_with(prefix)
                && origin.ends_with(suffix)
        }
        Some(_) => false,
        None => origin == pattern,
    }
}

/// This daemon's CORS layer: any method/header, but `Origin` must match
/// [`allowed_origins_from_env`].
fn cors_layer() -> tower_http::cors::CorsLayer {
    let patterns = allowed_origins_from_env();
    tower_http::cors::CorsLayer::new()
        .allow_methods(tower_http::cors::AllowMethods::any())
        .allow_headers(tower_http::cors::AllowHeaders::any())
        .allow_origin(tower_http::cors::AllowOrigin::predicate(
            move |origin, _parts| {
                origin
                    .to_str()
                    .is_ok_and(|origin| patterns.iter().any(|p| origin_matches(origin, p)))
            },
        ))
}

// ---------------------------------------------------------------------------
// llama-server binary resolution
// ---------------------------------------------------------------------------

/// GPU device-selection vars Ollama documents. A local `llama-server`
/// child inherits these for free; forwarded explicitly here so
/// `crate::container::spawn`'s `docker run`/`podman run` (which does
/// *not* inherit the host env) can reuse the same list.
pub const GPU_VISIBLE_DEVICE_VARS: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
];

/// llama.cpp's own env-configurable arguments (`common/arg.cpp`'s
/// `set_env`), forwarded the same way as [`GPU_VISIBLE_DEVICE_VARS`] —
/// llama-server reads these itself, llmman just makes sure they reach it.
pub const LLAMA_CPP_ENV_PASSTHROUGH_VARS: &[&str] =
    &["LLAMA_ARG_FIT", "LLAMA_ARG_FIT_TARGET", "LLAMA_ARG_THREADS"];

/// Resolves the `llama-server` binary to run locally (no `--ociman`):
/// prefers whatever is already on `PATH` untouched, unless
/// `pinned_version` explicitly asks for a specific llama.cpp release, in
/// which case that pin always wins. Falls back to downloading and caching
/// a release build matching this host's OS/arch/GPU backend via
/// `crate::llama_release` when nothing suitable is on PATH.
fn resolve_llama_server(pinned_version: Option<&str>) -> anyhow::Result<PathBuf> {
    if pinned_version.is_none() {
        if let Some(p) = find_on_path("llama-server") {
            return Ok(p);
        }
    }
    let resolved = crate::llama_release::ensure_llama_server(pinned_version)
        .context("no llama-server on PATH and automatic download failed")?;
    eprintln!(
        "[llmman] using downloaded llama-server ({}): {}",
        resolved.backend_label,
        resolved.bin.display()
    );
    Ok(resolved.bin)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: &ServeArgs) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(serve_async(args))
}

async fn serve_async(_args: &ServeArgs) -> anyhow::Result<()> {
    if _args.ociman.is_some() && !cfg!(target_os = "linux") {
        anyhow::bail!("--ociman is only supported on Linux");
    }
    // Must happen before daemon.rs's caller (if any) redirects this
    // process's stdio to a log file — see ServeArgs::pull_oci's doc
    // comment for why that would otherwise hide the pull's progress.
    // This is meant as its own explicit, foreground warm-up step run
    // before a separate, detached `serve` invocation — not a prelude to
    // this same invocation going on to serve — so it returns as soon as
    // the pull finishes instead of falling through into binding the
    // listener and serving forever.
    if _args.pull_oci {
        let ociman = _args.ociman.context("--pull-oci requires --ociman")?;
        crate::container::pull_image(ociman, _args.llama_cpp_version.as_deref())?;
        return Ok(());
    }
    // Same idea as --pull-oci above, but for the local (non-container)
    // llama-server binary path: resolve_llama_server's own download
    // (crate::llama_release) normally happens further down regardless of
    // --pull-bin, but by then this process may already be detached with
    // its stdio redirected to a log file (see daemon.rs) — a caller
    // waiting on the daemon to come up within ensure_server's short
    // timeout would see nothing and could time out mid-download,
    // indistinguishable from a hang. Run in the foreground first instead.
    if _args.pull_bin {
        let pinned_version = _args.llama_cpp_version.clone();
        tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
            .await
            .context("resolve llama-server task panicked")??;
        return Ok(());
    }
    // Only resolve (and require) a local llama-server binary when it'll
    // actually be used: --ociman runs llama-server in a container instead,
    // picking the image itself (see crate::container).
    //
    // resolve_llama_server does blocking network I/O (a GitHub API call,
    // and possibly a multi-hundred-MB download) when no llama-server is
    // already on PATH — spawn_blocking so that doesn't stall this async
    // fn's own executor thread while it runs.
    let llama_server_bin = if _args.ociman.is_none() {
        let pinned_version = _args.llama_cpp_version.clone();
        Some(
            tokio::task::spawn_blocking(move || resolve_llama_server(pinned_version.as_deref()))
                .await
                .context("resolve llama-server task panicked")??,
        )
    } else {
        None
    };
    let store_path = default_store()?;
    let cache_path = crate::default_cache()?;
    std::fs::create_dir_all(&cache_path)?;
    // See storage::repair's own doc comment — matches Ollama's own
    // unconditional `fixBlobs(blobsDir)` at the top of `server.Serve`,
    // before it starts listening.
    crate::storage::repair::repair_store(&store_path)?;

    // Catch-all GC sweep, right after repair: removes blobs/cache orphaned
    // by anything other than `rm` (a crash mid-pull past the grace window,
    // manual store surgery, an old build's leftover cache after a format
    // change). Grace-gated (unlike `rm`, which frees immediately) so a blob
    // written moments before its tag during a concurrent pull survives.
    // Gated by the same LLMMAN_NOPRUNE escape hatch as `rm`.
    if !crate::storage::gc::noprune_from_env() {
        if let Ok(store) = OciStore::open(&store_path) {
            if let Ok(live) = crate::storage::gc::referenced_digests(&store) {
                let grace = crate::storage::gc::GC_GRACE_PERIOD;
                if let Err(e) = crate::storage::gc::prune_blobs(&store_path, &live, grace) {
                    eprintln!("[llmman] blob GC sweep failed: {e:#}");
                }
                if let Err(e) = crate::storage::gc::prune_cache(&cache_path, &live, grace) {
                    eprintln!("[llmman] cache GC sweep failed: {e:#}");
                }
            }
        }
    }

    // See context_length_from_env's doc comment. spawn_blocking: like
    // resolve_llama_server above, the VRAM probe fallback spawns a
    // subprocess and must not block this async fn's executor thread.
    let ctx_size_explicit = context_length_from_env();
    let ctx_size = match ctx_size_explicit {
        Some(n) => Some(n),
        None => tokio::task::spawn_blocking(crate::hostgpu::default_ctx_size)
            .await
            .context("hostgpu probe task panicked")?,
    };

    // See threads_from_env_or_host's doc comment. Resolved once here
    // rather than per load, and logged so a surprising thread count is
    // explainable from the startup output. Only the local spawn path
    // consumes the derived value, so don't log it under --ociman (the
    // container arm deliberately ignores it).
    let threads = threads_from_env_or_host();
    if let Some(n) = threads {
        if _args.ociman.is_none() {
            eprintln!("[llmman] llama-server gets --threads {n} (CPU quota/affinity limit below the online CPU count)");
        }
    } else if std::env::var_os("LLAMA_ARG_THREADS").is_some() {
        eprintln!("[llmman] LLAMA_ARG_THREADS set: leaving llama-server thread count to it");
    }

    let state = AppState(Arc::new(Inner {
        manager: Mutex::new(ModelManager {
            running: HashMap::new(),
            pending_loads: 0,
        }),
        llama_server_bin: StdMutex::new(llama_server_bin),
        // Canonicalized now, while the file certainly still exists —
        // resolving later (in the handler) could fail once the install is
        // deleted, exactly the situation /api/version exists to expose.
        exe: std::env::current_exe()
            .ok()
            .map(|p| p.canonicalize().unwrap_or(p)),
        ociman: _args.ociman,
        llama_cpp_version: _args.llama_cpp_version.clone(),
        ctx_size,
        ctx_size_explicit: ctx_size_explicit.is_some(),
        flash_attention: flash_attention_from_env(),
        kv_cache_type: kv_cache_type_from_env(),
        split_mode: sched_spread_from_env(),
        num_parallel: num_parallel_from_env(),
        threads,
        max_queue: max_queue_from_env(),
        max_loaded_models: max_loaded_models_from_env(),
        store_path,
        cache_path,
        client: Client::new(),
    }));

    let app_state = state.clone();
    let app = Router::new()
        // Web UI
        .route("/", get(handle_root))
        .route("/bundle.js", get(handle_bundle_js))
        .route("/bundle.css", get(handle_bundle_css))
        .route("/loading.html", get(handle_loading_html))
        // llama.cpp-compatible props endpoint (router mode)
        .route("/props", get(handle_props))
        // llmman's own API — see handle_llmman_providers
        .route("/llmman/providers", get(handle_llmman_providers))
        .route("/llmman/providers/:id", get(handle_llmman_provider))
        // Ollama API
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/ps", get(handle_ps))
        .route("/api/show", post(handle_show))
        .route("/api/pull", post(handle_pull))
        .route("/api/push", post(handle_push))
        .route("/api/delete", delete(handle_delete))
        .route("/api/chat", post(handle_ollama_chat))
        .route("/api/generate", post(handle_ollama_generate))
        // OpenAI API
        .route("/v1/models", get(handle_openai_models))
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/v1/completions", post(handle_openai_completions))
        .route("/v1/embeddings", post(handle_openai_embeddings))
        .route(
            "/v1/audio/transcriptions",
            post(handle_openai_transcriptions)
                .layer(DefaultBodyLimit::max(TRANSCRIPTION_BODY_LIMIT_BYTES)),
        )
        .route(
            "/audio/transcriptions",
            post(handle_openai_transcriptions)
                .layer(DefaultBodyLimit::max(TRANSCRIPTION_BODY_LIMIT_BYTES)),
        )
        .route("/v1/responses", post(handle_openai_responses))
        .route(
            "/v1/responses/input_tokens",
            post(handle_openai_responses_input_tokens),
        )
        // Anthropic API
        .route("/v1/messages", post(handle_anthropic_messages))
        .layer(cors_layer())
        .with_state(app_state);

    let addr = crate::daemon::bind_addr();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    eprintln!("llmman serve listening on {addr}");

    // Background idle-unload reaper — see reap_idle_models's doc comment.
    tokio::spawn(reap_idle_models(state.clone()));

    // If a model was given on the command line, start loading it immediately
    // so the first request finds it already warm.
    // A provider-routed model has nothing to pre-load — there is no local
    // weight to warm, and `resolve_ollama_api` below would rewrite its
    // reference into a registry path it isn't. `cmd::launch` already
    // declines to pass one; this is the daemon's own guard for anyone
    // running `llmman serve <ref>` by hand.
    if let Some(model) = _args
        .model
        .as_deref()
        .filter(|m| !crate::providers::is_remote_ref(m))
    {
        match crate::shortnames::resolve_ollama_api(model) {
            Ok(model) => {
                let state_clone = state.clone();
                tokio::spawn(async move {
                    match ensure_model(&state_clone, &model, None).await {
                        // ensure_model's own keep_alive (the daemon default, 5
                        // minutes) would otherwise start counting down the
                        // moment this finishes loading, with no request traffic
                        // to reset it — the idle reaper could unload a model
                        // asked for on the command line before it's ever
                        // actually used, defeating the whole point of
                        // pre-loading it. Pin it ("never unload") instead — a
                        // model named explicitly at startup is meant to stay
                        // warm for the daemon's lifetime, not just its first 5
                        // idle minutes.
                        Ok((_, _, guard)) => refresh_activity(guard, None).await,
                        Err(e) => eprintln!("[llmman] pre-load failed: {:#}", e.0),
                    }
                });
            }
            Err(e) => eprintln!("[llmman] pre-load failed: {e}"),
        }
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Unload every running inference backend before exiting — the same
    // explicit unload `ollama serve` does when it traps SIGINT/SIGTERM
    // (server/routes.go's signal handler calling sched.unloadAllRunners).
    // Dropping each RunningModel kills local llama-server/vllm children
    // (kill_on_drop) and SIGTERMs container ones (ModelProcess::drop), so
    // nothing is left orphaned with a model still loaded in memory.
    state.0.manager.lock().await.running.clear();
    Ok(())
}

/// Resolves when the daemon is asked to shut down: SIGINT (Ctrl-C) on all
/// platforms, plus SIGTERM on Unix — the same pair `ollama serve` traps
/// (see server/routes.go) and the graceful signal every supervisor sends
/// first (Ollama's app on darwin, llmman's own daemon::stop_stale_daemon,
/// sbx). Trapping it means an in-flight request gets a chance to finish
/// (axum stops accepting and drains) and loaded models are unloaded
/// deliberately, instead of the whole process group being torn down
/// mid-write.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            // Installing the handler failed: never resolve on this arm
            // rather than shutting down immediately for no reason.
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    eprintln!("llmman serve shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- request targets (local backend vs remote provider) -----------------

    fn remote_target(base_url: &str) -> Target {
        Target::Remote(Arc::new(RemoteTarget {
            provider: "mockprov".into(),
            base_url: base_url.into(),
            model: "mock-model".into(),
            api_key: "sk-test".into(),
        }))
    }

    /// A local target must keep producing byte-for-byte the same loopback
    /// URLs the `format!("http://127.0.0.1:{port}{path}")` calls this
    /// replaced did — every existing route depends on it.
    #[test]
    fn a_local_target_addresses_loopback_unchanged() {
        let target = Target::Local(17434);
        assert_eq!(
            target.url("/v1/chat/completions"),
            "http://127.0.0.1:17434/v1/chat/completions"
        );
        assert_eq!(
            target.url("/v1/audio/transcriptions"),
            "http://127.0.0.1:17434/v1/audio/transcriptions"
        );
        assert_eq!(
            target.url("/v1/responses/input_tokens"),
            "http://127.0.0.1:17434/v1/responses/input_tokens"
        );
        assert!(!target.is_remote());
    }

    /// A remote target re-bases llmman's internal `/v1/...` route onto
    /// whatever version segment the provider published, which is often not
    /// `/v1` and sometimes absent — getting this wrong yields a doubled
    /// `/v1/v1/` or a dropped path segment.
    #[test]
    fn a_remote_target_rebases_routes_onto_the_provider_url() {
        assert_eq!(
            remote_target("https://openrouter.ai/api/v1").url("/v1/chat/completions"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            remote_target("https://generativelanguage.googleapis.com/v1beta/openai")
                .url("/v1/chat/completions"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        assert_eq!(
            remote_target("https://api.perplexity.ai").url("/v1/chat/completions"),
            "https://api.perplexity.ai/chat/completions"
        );
        assert!(remote_target("https://example.invalid/v1").is_remote());
    }

    /// Both spellings the surfaces here accept, so a provider key reaches
    /// an already-running daemon whichever integration sent it.
    #[test]
    fn client_api_key_reads_bearer_and_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-openai-style".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-openai-style".to_string())
        );

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-anthropic-style".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-anthropic-style".to_string())
        );

        // Authorization wins when a client sends both.
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-bearer".parse().unwrap());
        headers.insert("x-api-key", "sk-other".parse().unwrap());
        assert_eq!(
            client_api_key(Some(&headers)),
            Some("sk-bearer".to_string())
        );
    }

    /// `cmd::launch` gives locally-served integrations a placeholder key
    /// because several refuse to start without one. Treating it as a real
    /// credential would forward a meaningless token to a real provider
    /// (which rejects it with an opaque 401) instead of falling back to
    /// the daemon's own configured key.
    #[test]
    fn client_api_key_ignores_the_local_placeholder_and_empty_values() {
        for header in ["Bearer llmman", "Bearer ", "Bearer    ", ""] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                None,
                "{header:?} was treated as a credential"
            );
        }

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", PLACEHOLDER_API_KEY.parse().unwrap());
        assert_eq!(client_api_key(Some(&headers)), None);

        assert_eq!(client_api_key(None), None);
        assert_eq!(client_api_key(Some(&HeaderMap::new())), None);
    }

    /// A malformed or non-bearer `Authorization` is not a key — llmman
    /// must fall through to its own configured one rather than forwarding
    /// something that was never a bearer token.
    #[test]
    fn client_api_key_ignores_non_bearer_authorization() {
        for header in ["Basic dXNlcjpwYXNz", "sk-no-scheme", "Bearer"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                None,
                "{header:?} was treated as a bearer token"
            );
        }
    }

    /// RFC 7235 makes the scheme case-insensitive and clients do send
    /// `bearer`. Matching one spelling silently drops a real key, and on
    /// a daemon that won't use its own there is nothing to fall back to.
    #[test]
    fn client_api_key_accepts_any_spelling_of_bearer() {
        for header in ["Bearer sk-real", "bearer sk-real", "BEARER sk-real"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                Some("sk-real".to_string()),
                "{header:?}"
            );
        }
    }

    /// Each header is judged on its own: an unusable `Authorization` must
    /// not shadow a real `x-api-key`, which is exactly what a client that
    /// hardcodes one and configures the other sends.
    #[test]
    fn client_api_key_falls_through_an_unusable_authorization() {
        for header in ["Bearer llmman", "Bearer ", "Basic dXNlcjpwYXNz"] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", header.parse().unwrap());
            headers.insert("x-api-key", "sk-real".parse().unwrap());
            assert_eq!(
                client_api_key(Some(&headers)),
                Some("sk-real".to_string()),
                "{header:?} shadowed a real x-api-key"
            );
        }
    }

    /// OpenAI-wire-format does not mean every OpenAI route: `anthropic`
    /// and `mistral` 404 on `/v1/responses` where `openai`, `groq` and
    /// `openrouter` answer it. Codex uses only that route, so the bare
    /// 404 it would otherwise show has to become an explanation.
    #[tokio::test]
    async fn a_providers_missing_responses_route_is_explained() {
        let remote = remote_target("https://example.invalid/v1");
        let not_found = (StatusCode::NOT_FOUND, "{}").into_response();
        let explained = explain_missing_route(&remote, "/v1/responses", not_found);
        assert_eq!(explained.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(explained.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("mockprov"), "{body}");
        assert!(body.contains("Responses API"), "{body}");

        // The token-counting route is the same story.
        let counted = explain_missing_route(
            &remote,
            "/v1/responses/input_tokens",
            (StatusCode::NOT_FOUND, "{}").into_response(),
        );
        assert_eq!(counted.status(), StatusCode::NOT_IMPLEMENTED);

        // Everything else is relayed untouched. A 404 on another route is
        // an unknown model, not a missing API, and answering it with the
        // wrong explanation is worse than passing the provider's own.
        for (target, route, status) in [
            (&remote, "/v1/chat/completions", StatusCode::NOT_FOUND),
            (&remote, "/v1/embeddings", StatusCode::NOT_FOUND),
            (
                &Target::Local(17434),
                "/v1/responses",
                StatusCode::NOT_FOUND,
            ),
            (&remote, "/v1/responses", StatusCode::OK),
            (&remote, "/v1/responses", StatusCode::UNAUTHORIZED),
        ] {
            let resp = explain_missing_route(target, route, (status, "{}").into_response());
            assert_eq!(resp.status(), status, "{route} {status}");
        }
    }

    /// A page the user merely visits can POST here — CORS gates reading
    /// the reply, not sending a "simple" request, and these handlers
    /// never check `Content-Type`. It must not be able to spend the key
    /// in the daemon's environment; not seeing the answer is no comfort
    /// once the money is gone.
    #[test]
    fn only_a_browser_acting_for_another_site_is_refused_the_daemons_key() {
        for site in ["cross-site", "same-site", "Cross-Site", " cross-site "] {
            let mut headers = HeaderMap::new();
            headers.insert("sec-fetch-site", site.parse().unwrap());
            assert!(is_cross_site(Some(&headers)), "{site:?}");
        }
        // llmman's own web UI, and a typed URL.
        for site in ["same-origin", "none"] {
            let mut headers = HeaderMap::new();
            headers.insert("sec-fetch-site", site.parse().unwrap());
            assert!(!is_cross_site(Some(&headers)), "{site:?}");
        }
        // A CLI integration sends no such header, which is the whole
        // reason this can gate on it without breaking them.
        assert!(!is_cross_site(Some(&HeaderMap::new())));
        assert!(!is_cross_site(None));
    }

    /// The key is the one field of a remote target that must never reach
    /// a log line, so it cannot be reachable through `Debug` either.
    #[test]
    fn a_remote_targets_debug_output_omits_the_key() {
        let rendered = format!("{:?}", remote_target("https://example.invalid/v1"));
        assert!(!rendered.contains("sk-test"), "{rendered}");
        assert!(rendered.contains("mockprov"), "{rendered}");
    }

    /// `repeat_penalty` is a llama.cpp extension. A local backend wants
    /// llmman's default; a provider rejects the request over it.
    #[test]
    fn repeat_penalty_is_local_only() {
        assert!(repeat_penalty_applies(&Target::Local(17434)));
        assert!(!repeat_penalty_applies(&remote_target(
            "https://example.invalid/v1"
        )));
    }

    /// A bad provider key must reach the user as the 401 it is. Burying
    /// it in llmman's blanket 500 is what makes "check your key" look
    /// like "llmman is broken" — while a local backend, whose failures
    /// really are llmman's own, keeps the 500 it always returned.
    #[test]
    fn a_providers_own_status_is_not_buried_in_a_500() {
        let remote = remote_target("https://example.invalid/v1");
        assert_eq!(
            remote_status(&remote, StatusCode::UNAUTHORIZED),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            remote_status(&remote, StatusCode::TOO_MANY_REQUESTS),
            StatusCode::TOO_MANY_REQUESTS
        );
        // Not the caller's fault, and not llmman's either.
        assert_eq!(
            remote_status(&remote, StatusCode::BAD_GATEWAY),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            remote_status(&remote, StatusCode::INTERNAL_SERVER_ERROR),
            StatusCode::BAD_GATEWAY
        );

        let local = Target::Local(17434);
        for upstream in [StatusCode::UNAUTHORIZED, StatusCode::INTERNAL_SERVER_ERROR] {
            assert_eq!(
                remote_status(&local, upstream),
                StatusCode::INTERNAL_SERVER_ERROR,
                "a local backend's status changed"
            );
        }

        // And the message says which of the two failed, without the key.
        assert_eq!(local.describe(), "inference backend");
        assert_eq!(remote.describe(), "provider mockprov");
    }

    /// What a provider actually receives, checked against a real HTTP
    /// server rather than inferred from the pieces: the route re-based
    /// onto its own base URL, the bearer token, the model under the id it
    /// knows, and no llama.cpp-only field for it to reject.
    #[tokio::test]
    async fn a_remote_request_reaches_the_provider_as_plain_openai() {
        let seen = Arc::new(tokio::sync::Mutex::new(None));
        let captured = seen.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|headers: HeaderMap, body: Bytes| async move {
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let auth = headers["authorization"].to_str().unwrap().to_string();
                *captured.lock().await = Some((auth, body));
                ([("content-type", "application/json")], "{}")
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // `/v1` on the base and `/v1/...` on the route must not double up.
        let target = remote_target(&format!("http://127.0.0.1:{}/v1", addr.port()));
        let mut req = OAIChatRequest {
            model: "mock-model".into(),
            messages: vec![],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            repeat_penalty: Some(1.1),
            chat_template_kwargs: None,
            tools: None,
            response_format: None,
        };
        post_chat(&Client::new(), &target, &mut req).await.unwrap();

        let (auth, body) = seen.lock().await.take().expect("provider was not called");
        assert_eq!(auth, "Bearer sk-test");
        assert_eq!(body["model"], "mock-model");
        assert!(
            body.get("repeat_penalty").is_none(),
            "llama.cpp-only field sent to a provider: {body}"
        );
    }

    // -- keep_alive parsing / resolution (idle-timeout auto-unload) ---------

    #[test]
    fn parse_keep_alive_str_handles_bare_seconds_units_and_negatives() {
        assert_eq!(
            parse_keep_alive_str("300"),
            Some(Some(Duration::from_secs(300)))
        );
        assert_eq!(
            parse_keep_alive_str("10m"),
            Some(Some(Duration::from_secs(600)))
        );
        assert_eq!(
            parse_keep_alive_str("1h30m"),
            Some(Some(Duration::from_secs(5400)))
        );
        assert_eq!(
            parse_keep_alive_str("30s"),
            Some(Some(Duration::from_secs(30)))
        );
        assert_eq!(
            parse_keep_alive_str("500ms"),
            Some(Some(Duration::from_millis(500)))
        );
        // Any negative value — bare number or unit string — means "never
        // unload", matching Ollama's own keep_alive: -1 convention.
        assert_eq!(parse_keep_alive_str("-1"), Some(None));
        assert_eq!(parse_keep_alive_str("-5m"), Some(None));
        // Unparseable input falls back (via the caller, resolve_keep_alive)
        // to the daemon default, signaled here by an outer None.
        assert_eq!(parse_keep_alive_str("not-a-duration"), None);
        assert_eq!(parse_keep_alive_str(""), None);
        assert_eq!(parse_keep_alive_str("10x"), None);
    }

    /// Regression test: `f64`'s own `FromStr` accepts "inf"/"infinity"/
    /// "nan" (any case) as a bare number, and even an ordinary huge finite
    /// literal can overflow `Duration`'s own range — every one of these
    /// used to panic via `Duration::from_secs_f64` (see
    /// `secs_to_keep_alive`'s doc comment) instead of being treated as
    /// just another unparseable `keep_alive` value.
    #[test]
    fn parse_keep_alive_str_never_panics_on_non_finite_or_overflowing_input() {
        assert_eq!(parse_keep_alive_str("inf"), None);
        assert_eq!(parse_keep_alive_str("Infinity"), None);
        assert_eq!(parse_keep_alive_str("nan"), None);
        assert_eq!(parse_keep_alive_str("NaN"), None);
        // A negative infinity is still just "negative" — "never unload",
        // same as any other negative value — not an error.
        assert_eq!(parse_keep_alive_str("-inf"), Some(None));
        // Finite, but far larger than Duration can represent.
        assert_eq!(parse_keep_alive_str("1e300"), None);
        assert_eq!(parse_keep_alive_str("1e300s"), None);
        // Two components that each individually fit, but whose sum
        // overflows once added together.
        assert_eq!(
            parse_keep_alive_str(&format!("{}s{}s", u64::MAX, u64::MAX)),
            None
        );
    }

    /// Same non-panicking guarantee, exercised through the JSON-number
    /// path (`resolve_keep_alive`/`parse_keep_alive_value`) rather than
    /// the duration-string one.
    #[test]
    fn resolve_keep_alive_never_panics_on_an_overflowing_json_number() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(1e300))),
            default_keep_alive()
        );
    }

    #[test]
    fn resolve_keep_alive_falls_back_to_the_default_on_absent_or_unparseable_values() {
        // Against default_keep_alive() itself, not the DEFAULT_KEEP_ALIVE
        // constant directly: if LLMMAN_KEEP_ALIVE happens to be set in
        // whatever environment runs this test (a developer's shell, a CI
        // job), the constant and the actual fallback would disagree
        // through no fault of the code under test.
        let default = default_keep_alive();
        assert_eq!(resolve_keep_alive(&None), default);
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("garbage"))),
            default
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(true))), default);
    }

    #[test]
    fn resolve_keep_alive_accepts_a_json_number_of_seconds_or_a_duration_string() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(30))),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("10m"))),
            Some(Duration::from_secs(600))
        );
        assert_eq!(resolve_keep_alive(&Some(serde_json::json!(-1))), None);
    }

    #[test]
    fn parse_load_timeout_accepts_the_same_duration_syntax_as_keep_alive() {
        assert_eq!(
            parse_load_timeout("300"),
            Some(Some(Duration::from_secs(300)))
        );
        assert_eq!(
            parse_load_timeout("10m"),
            Some(Some(Duration::from_secs(600)))
        );
        assert_eq!(parse_load_timeout("garbage"), None);
    }

    #[test]
    fn parse_load_timeout_treats_zero_or_negative_as_infinite() {
        // Unlike keep_alive (where 0 means "unload immediately"),
        // OLLAMA_LOAD_TIMEOUT documents 0 as meaning "wait forever", same
        // as any negative value.
        assert_eq!(parse_load_timeout("0"), Some(None));
        assert_eq!(parse_load_timeout("-1"), Some(None));
        assert_eq!(parse_load_timeout("-10m"), Some(None));
    }

    #[test]
    fn parse_load_timeout_rejects_a_dash_prefixed_non_duration_instead_of_disabling_forever() {
        // A bare "starts with '-'" isn't enough — the magnitude must
        // actually parse as a duration, or this must fall through to
        // the default (via load_timeout_from_env's own unwrap_or), not
        // silently disable the timeout forever.
        assert_eq!(parse_load_timeout("-garbage"), None);
        assert_eq!(parse_load_timeout("-"), None);
    }

    /// Regression test for `handle_ollama_generate`'s unload-sentinel
    /// check: it must reuse `resolve_keep_alive` (as asserted here) rather
    /// than a bare `keep_alive.as_i64() == Some(0)` check, since the
    /// latter misses every non-integer zero form `resolve_keep_alive`
    /// itself accepts — a string `"0"`, `"0s"`, or a float `0.0` — leaving
    /// a client that sends one of those loaded until the next idle-reaper
    /// tick instead of unloading immediately as requested.
    #[test]
    fn resolve_keep_alive_treats_every_zero_form_as_the_unload_sentinel() {
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(0))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("0"))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!("0s"))),
            Some(Duration::ZERO)
        );
        assert_eq!(
            resolve_keep_alive(&Some(serde_json::json!(0.0))),
            Some(Duration::ZERO)
        );
    }

    // -- format -> response_format (structured output) -----------------------

    #[test]
    fn format_to_response_format_maps_json_string_and_schema_object() {
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("json"))),
            Some(serde_json::json!({ "type": "json_object" }))
        );
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } }
        });
        assert_eq!(
            format_to_response_format(&Some(schema.clone())),
            Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": "response", "schema": schema, "strict": true }
            }))
        );
    }

    #[test]
    fn format_to_response_format_is_a_no_op_when_absent_or_unrecognized() {
        assert_eq!(format_to_response_format(&None), None);
        // Ollama documents only "json" and a schema object — anything else
        // (a bare bool/number/other string) has no equivalent, same as an
        // unrecognized `think` shape in think_to_chat_template_kwargs.
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!(true))),
            None
        );
        assert_eq!(
            format_to_response_format(&Some(serde_json::json!("text"))),
            None
        );
    }

    // -- apply_default_repeat_penalty (/v1/chat/completions, /v1/completions,
    //    /v1/responses — the raw OpenAI-passthrough generation routes) -----

    #[test]
    fn apply_default_repeat_penalty_sets_default_when_absent() {
        let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "messages": []});
        apply_default_repeat_penalty(&mut req);
        assert_eq!(
            req["repeat_penalty"],
            serde_json::json!(DEFAULT_REPEAT_PENALTY)
        );
    }

    #[test]
    fn apply_default_repeat_penalty_preserves_an_explicit_value() {
        // Deliberately not DEFAULT_REPEAT_PENALTY's own value (1.0) — this
        // has to prove the caller's *explicit* choice survives, which a
        // value indistinguishable from the default couldn't.
        let mut req = serde_json::json!({"model": "qwen3.5:0.8b", "repeat_penalty": 1.3});
        apply_default_repeat_penalty(&mut req);
        assert_eq!(req["repeat_penalty"], serde_json::json!(1.3));
    }

    // -- apply_default_repeat_penalty_typed (every typed request — /api/chat,
    //    /api/generate, the Anthropic Messages API — via post_chat) --------

    fn oai_chat_request_with_repeat_penalty(repeat_penalty: Option<f32>) -> OAIChatRequest {
        OAIChatRequest {
            model: "qwen3.5:0.8b".into(),
            messages: vec![],
            stream: true,
            temperature: None,
            top_p: None,
            max_tokens: None,
            repeat_penalty,
            chat_template_kwargs: None,
            tools: None,
            response_format: None,
        }
    }

    #[test]
    fn apply_default_repeat_penalty_typed_sets_default_when_absent() {
        let mut oai = oai_chat_request_with_repeat_penalty(None);
        apply_default_repeat_penalty_typed(&mut oai);
        assert_eq!(oai.repeat_penalty, Some(DEFAULT_REPEAT_PENALTY));
    }

    #[test]
    fn apply_default_repeat_penalty_typed_preserves_an_explicit_value() {
        // Same rationale as apply_default_repeat_penalty_preserves_an_explicit_value
        // above — 1.3 rather than DEFAULT_REPEAT_PENALTY's own 1.0.
        let mut oai = oai_chat_request_with_repeat_penalty(Some(1.3));
        apply_default_repeat_penalty_typed(&mut oai);
        assert_eq!(oai.repeat_penalty, Some(1.3));
    }

    // -- OllamaMessage -> OAIMessage (vision, tool calls, tool results) -----

    #[test]
    fn ollama_message_to_oai_plain_text_has_string_content_and_no_extras() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.role, "user");
        assert_eq!(oai.content, serde_json::json!("hi"));
        assert_eq!(oai.tool_calls, None);
        assert_eq!(oai.name, None);
    }

    #[test]
    fn ollama_message_to_oai_with_images_builds_a_content_parts_array() {
        let m = OllamaMessage {
            role: "user".into(),
            content: "what is this?".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_with_only_an_image_omits_the_empty_text_part() {
        let m = OllamaMessage {
            role: "user".into(),
            images: Some(vec!["Zm9v".into()]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(
            oai.content,
            serde_json::json!([
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,Zm9v" } }
            ])
        );
    }

    #[test]
    fn ollama_message_to_oai_carries_tool_calls_and_re_encodes_arguments_as_a_string() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![OllamaToolCall {
                function: OllamaToolCallFunction {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "nyc" }),
                },
            }]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        let calls = oai.tool_calls.expect("tool_calls must be carried over");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // OpenAI's function.arguments is a JSON-*encoded string*, unlike
        // Ollama's already-decoded object.
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({ "city": "nyc" })
        );
    }

    /// Regression test: `gen_id()` alone is time-based and, on a platform
    /// with coarse clock resolution, two calls made back-to-back (as
    /// happens once per tool call in a single message) can return the same
    /// value — an id collision that would make a strict id-matching chat
    /// template mismatch tool results. The per-call index appended to
    /// `gen_id()`'s own output must make every id in one message unique
    /// even then.
    #[test]
    fn ollama_message_to_oai_gives_each_tool_call_a_distinct_id_even_with_identical_names() {
        let m = OllamaMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![
                OllamaToolCall {
                    function: OllamaToolCallFunction {
                        name: "get_weather".into(),
                        arguments: serde_json::json!({ "city": "nyc" }),
                    },
                },
                OllamaToolCall {
                    function: OllamaToolCallFunction {
                        name: "get_weather".into(),
                        arguments: serde_json::json!({ "city": "sf" }),
                    },
                },
            ]),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        let calls = oai.tool_calls.expect("tool_calls must be carried over");
        assert_eq!(calls.len(), 2);
        assert_ne!(
            calls[0].id, calls[1].id,
            "two tool calls in one message must never share an id"
        );
    }

    #[test]
    fn ollama_message_to_oai_maps_tool_name_to_name_on_a_tool_result_message() {
        let m = OllamaMessage {
            role: "tool".into(),
            content: "72F and sunny".into(),
            tool_name: Some("get_weather".into()),
            ..Default::default()
        };
        let oai = ollama_message_to_oai(&m);
        assert_eq!(oai.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn image_data_uri_wraps_bare_base64_and_passes_through_existing_data_uris() {
        assert_eq!(image_data_uri("Zm9v"), "data:image/png;base64,Zm9v");
        assert_eq!(
            image_data_uri("data:image/jpeg;base64,Zm9v"),
            "data:image/jpeg;base64,Zm9v"
        );
    }

    // -- Streaming tool-call accumulation (/api/chat) -------------------------

    /// Regression test for OpenAI's own streaming tool-call shape: `id`
    /// and `function.name` normally arrive whole in the first delta for a
    /// given `index`, while `function.arguments` is only complete, valid
    /// JSON once every fragment across possibly-many chunks is
    /// concatenated — never fragment-by-fragment.
    #[test]
    fn tool_call_accumulator_assembles_fragmented_streaming_deltas() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"nyc\"}"}}
            ]},"finish_reason":null}]}"#,
            &acc,
        );
        let calls = finalize_tool_calls(&acc.borrow()).expect("must assemble one tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({ "city": "nyc" })
        );
    }

    /// Regression test: llama-server's SSE stream signals "done" twice per
    /// response — once on the chunk carrying a real `finish_reason`, then
    /// again on the trailing literal `"[DONE]"` line — so whatever reads
    /// `oai_chunk_to_content`'s `done` flag sees it `true` more than once.
    /// `stream_ollama` drains the accumulator (`std::mem::take`, mirrored
    /// here) rather than just reading it on each such occurrence, so a
    /// tool call is finalized — and so delivered to the client — exactly
    /// once, never twice.
    #[test]
    fn draining_the_accumulator_on_finalize_prevents_delivering_a_tool_call_twice() {
        let acc = std::cell::RefCell::new(std::collections::BTreeMap::new());
        accumulate_tool_call_deltas(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"get_weather","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]}"#,
            &acc,
        );

        let first = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert!(
            first.is_some(),
            "the first done signal must still deliver the tool call"
        );

        let second = finalize_tool_calls(&std::mem::take(&mut *acc.borrow_mut()));
        assert_eq!(
            second, None,
            "a second done signal (the trailing [DONE] line) must not re-deliver it"
        );
    }

    #[test]
    fn finalize_tool_calls_is_none_when_no_tool_calls_were_made() {
        assert_eq!(
            finalize_tool_calls(&std::collections::BTreeMap::new()),
            None
        );
    }

    #[test]
    fn finalize_tool_calls_falls_back_to_an_empty_object_on_unparseable_arguments() {
        let mut acc = std::collections::BTreeMap::new();
        acc.insert(
            0,
            ToolCallAccumulator {
                name: "f".into(),
                arguments: "not json".into(),
            },
        );
        let calls = finalize_tool_calls(&acc).unwrap();
        assert_eq!(calls[0].function.arguments, serde_json::json!({}));
    }

    #[test]
    fn oai_chunk_tool_call_deltas_is_empty_for_done_sentinel_and_ordinary_content() {
        assert!(oai_chunk_tool_call_deltas("[DONE]").is_empty());
        assert!(oai_chunk_tool_call_deltas(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
        )
        .is_empty());
    }

    // -- llmman's own API ----------------------------------------------------

    /// A catalog to serve, without the network. Its key variable is one
    /// nothing exports, so `key_set` is false even in a shell that has a
    /// real OpenRouter key.
    fn fixture_catalog() -> crate::providers::Catalog {
        crate::providers::Catalog::from_json(
            br#"{
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["LLMMAN_TEST_PROVIDER_KEY_UNSET"],
                    "models": {
                        "z-model": { "cost": { "input": 2.5, "output": 10 } },
                        "a-model": {}
                    }
                }
            }"#,
        )
        .expect("fixture parses")
    }

    /// "Which providers are there" gets a count, not every model id.
    #[test]
    fn the_provider_listing_reports_model_counts_not_model_ids() {
        let catalog = fixture_catalog();
        let summaries: Vec<ProviderSummary> = catalog.iter().map(ProviderSummary::from).collect();
        let json = serde_json::to_value(&summaries).unwrap();
        assert_eq!(json[0]["id"], "openrouter");
        assert_eq!(json[0]["name"], "OpenRouter");
        assert_eq!(json[0]["key_env"], "LLMMAN_TEST_PROVIDER_KEY_UNSET");
        assert_eq!(json[0]["models"], 2);
        assert_eq!(json[0]["key_set"], false);
        assert_eq!(json[0]["key_usable"], false);
    }

    /// The per-provider route carries the models themselves, sorted by
    /// id (`list --provider` prints them straight through) and priced
    /// where models.dev prices them.
    #[test]
    fn a_single_provider_carries_its_models_and_their_prices() {
        let catalog = fixture_catalog();
        let provider = catalog.get("openrouter").unwrap();
        let json = serde_json::to_value(ProviderResponse::from(provider)).unwrap();
        assert_eq!(json["base_url"], "https://openrouter.ai/api/v1");
        assert_eq!(
            json["models"],
            serde_json::json!([
                { "id": "a-model" },
                { "id": "z-model", "cost": { "input": 2.5, "output": 10.0 } },
            ])
        );
    }

    /// `key_set` is the whole of what a client is told about a key.
    #[test]
    fn provider_responses_carry_no_api_key() {
        let catalog = fixture_catalog();
        let provider = catalog.get("openrouter").unwrap();
        for json in [
            serde_json::to_value(ProviderSummary::from(provider)).unwrap(),
            serde_json::to_value(ProviderResponse::from(provider)).unwrap(),
        ] {
            let fields: Vec<&String> = json.as_object().unwrap().keys().collect();
            assert!(
                !fields.iter().any(|f| f.contains("api_key")),
                "{fields:?} carries a key"
            );
        }
    }

    // -- Idle-timeout auto-unload reaper --------------------------------------

    fn test_state() -> AppState {
        test_state_at(std::env::temp_dir())
    }

    /// `test_state` with a real store directory, for the few tests that
    /// need `canonical_ref` to actually resolve something.
    fn test_state_at(store_path: PathBuf) -> AppState {
        AppState(Arc::new(Inner {
            manager: Mutex::new(ModelManager {
                running: HashMap::new(),
                pending_loads: 0,
            }),
            llama_server_bin: StdMutex::new(None),
            exe: None,
            ociman: None,
            llama_cpp_version: None,
            ctx_size: None,
            ctx_size_explicit: false,
            flash_attention: None,
            kv_cache_type: None,
            split_mode: None,
            num_parallel: None,
            threads: None,
            // usize::MAX, not 0 — 0 now means "admit almost nothing"
            // (see try_admit_against's doc comment), and no test here
            // calls ensure_model (the only caller of try_admit) directly
            // anyway.
            max_queue: usize::MAX,
            max_loaded_models: 0,
            store_path,
            cache_path: std::env::temp_dir(),
            client: Client::new(),
        }))
    }

    /// A long-lived, harmless real child process to back a test
    /// `RunningModel` — `ModelProcess::is_alive`/`Drop` both need a real
    /// `tokio::process::Child`, not a mock. `sleep` isn't on `PATH` on
    /// Windows (which this project does target — see the `#[cfg(windows)]`
    /// branches elsewhere in this module), so it's spawned differently per
    /// platform rather than assuming a Unix-only test environment.
    ///
    /// Its own process group (matching `spawn_vllm_server`'s own real
    /// spawn — see its doc comment), not just the bare default: a
    /// fixture backing an `Engine::Vllm` `RunningModel` hits
    /// `ModelProcess::Drop`'s process-group-SIGKILL arm, which needs
    /// this to actually *be* one, or that kill fails and prints a
    /// spurious "SIGKILL to vllm process group ... failed" warning on
    /// every test run that uses one — confirmed live via CodeRabbit
    /// review on this repo's own git history.
    #[cfg(unix)]
    fn spawn_placeholder_process() -> tokio::process::Child {
        tokio::process::Command::new("sleep")
            .arg("60")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .expect("spawn placeholder `sleep` process")
    }

    #[cfg(windows)]
    fn spawn_placeholder_process() -> tokio::process::Child {
        tokio::process::Command::new("cmd")
            .args(["/C", "timeout", "/T", "60", "/NOBREAK"])
            .kill_on_drop(true)
            .spawn()
            .expect("spawn placeholder `cmd /C timeout` process")
    }

    fn running_model_fixture(
        keep_alive: Option<Duration>,
        idle_for: Duration,
        in_flight: u32,
    ) -> RunningModel {
        RunningModel {
            process: ModelProcess::Local(Engine::LlamaServer, spawn_placeholder_process(), None),
            port: 0,
            digest: String::new(),
            size: 0,
            started_at: now_rfc3339(),
            last_active: Instant::now() - idle_for,
            last_active_wall: chrono::Utc::now(),
            backend_model_path: None,
            keep_alive,
            in_flight,
        }
    }

    /// Like `running_model_fixture`, but with a caller-chosen `Engine`
    /// and `backend_model_path` — used only by `backend_wire_model`'s
    /// own tests below, which need to distinguish an `Engine::Mlx`
    /// backend from every other one.
    fn running_model_fixture_with_engine(
        engine: Engine,
        backend_model_path: Option<&str>,
    ) -> RunningModel {
        RunningModel {
            process: ModelProcess::Local(engine, spawn_placeholder_process(), None),
            port: 0,
            digest: String::new(),
            size: 0,
            started_at: now_rfc3339(),
            last_active: Instant::now(),
            last_active_wall: chrono::Utc::now(),
            backend_model_path: backend_model_path.map(|s| s.to_string()),
            keep_alive: None,
            in_flight: 0,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_is_the_canonical_name_for_every_engine_except_mlx() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "llama-model".into(),
                running_model_fixture_with_engine(Engine::LlamaServer, None),
            );
            mgr.running.insert(
                "vllm-model".into(),
                running_model_fixture_with_engine(Engine::Vllm, None),
            );
            mgr.running.insert(
                "mlx-model".into(),
                running_model_fixture_with_engine(Engine::Mlx, Some("/cache/mlx-model/abcd")),
            );
        }

        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "llama-model").await,
            "llama-model"
        );
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "vllm-model").await,
            "vllm-model"
        );
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "mlx-model").await,
            "/cache/mlx-model/abcd",
            "an Engine::Mlx backend must be addressed by its real directory path, not its human-readable name"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_falls_back_to_the_canonical_name_when_not_running() {
        let state = test_state();
        assert_eq!(
            backend_wire_model(&state, &Target::Local(0), "not-running").await,
            "not-running"
        );
    }

    /// A remote provider knows nothing of `providers::REMOTE_PREFIX` — it
    /// must receive its own bare model id, with the routing prefix llmman
    /// added stripped back off, and without consulting `running` at all
    /// (a provider-routed model is never in it).
    #[tokio::test(flavor = "multi_thread")]
    async fn backend_wire_model_strips_the_routing_prefix_for_a_remote_target() {
        let state = test_state();
        let target = remote_target("https://example.invalid/v1");
        assert_eq!(
            backend_wire_model(&state, &target, "llmman.provider/mockprov/mock-model").await,
            "mock-model"
        );
    }

    /// Regression test for the CodeRabbit nitpick this PR addresses:
    /// `proxy_openai_passthrough`'s `/v1/embeddings` guard must be able
    /// to answer "would this already-running model be served by
    /// Engine::Mlx" from a plain map lookup — no backend spawn, no
    /// model load — for the common case of a repeated embeddings
    /// request against a model that's already loaded.
    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_finds_an_already_running_mlx_model_without_touching_disk_or_a_process() {
        let state = test_state();
        // A reference already in canonical form (host + owner/repo +
        // explicit tag) so shortnames::resolve_ollama_api/default_tag
        // and canonical_ref (which no-ops against this test's empty
        // store — see its own doc comment) all leave it unchanged,
        // matching the same key this inserts into `mgr.running` under.
        let model_ref = "hf.co/mlx-community/foo:latest";
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                model_ref.to_string(),
                running_model_fixture_with_engine(Engine::Mlx, Some("/cache/foo/abcd")),
            );
        }
        assert_eq!(
            would_use_mlx(&state, model_ref).await,
            Some(model_ref.to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_is_none_for_an_already_running_non_mlx_model() {
        let state = test_state();
        let model_ref = "hf.co/mlx-community/foo:latest";
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                model_ref.to_string(),
                running_model_fixture_with_engine(Engine::LlamaServer, None),
            );
        }
        assert_eq!(would_use_mlx(&state, model_ref).await, None);
    }

    /// On any host `use_mlx_for_safetensors` itself doesn't consider
    /// Apple-Silicon-macOS-with-`mlx_lm.server`-on-`PATH` (this test
    /// suite's own CI hosts included), `would_use_mlx` must say `None`
    /// for a model that isn't running yet at all — regardless of
    /// whatever is or isn't actually in the local store for it —
    /// without needing to fake either check to prove it.
    #[tokio::test(flavor = "multi_thread")]
    async fn would_use_mlx_is_none_when_not_running_and_this_host_never_uses_mlx() {
        let state = test_state();
        assert_eq!(
            would_use_mlx(&state, "hf.co/mlx-community/not-loaded-yet:latest").await,
            None
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reap_idle_models_unloads_only_idle_expired_models_not_in_flight_or_forever() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "expired-and-idle".into(),
                running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 0),
            );
            mgr.running.insert(
                "expired-but-in-flight".into(),
                running_model_fixture(Some(Duration::from_secs(1)), Duration::from_secs(10), 1),
            );
            mgr.running.insert(
                "expired-but-forever".into(),
                running_model_fixture(None, Duration::from_secs(10), 0),
            );
            mgr.running.insert(
                "not-yet-expired".into(),
                running_model_fixture(Some(Duration::from_secs(300)), Duration::from_secs(1), 0),
            );
        }

        reap_idle_models_once(&state).await;

        let mgr = state.0.manager.lock().await;
        assert!(
            !mgr.running.contains_key("expired-and-idle"),
            "an idle model past its keep_alive deadline must be unloaded"
        );
        assert!(
            mgr.running.contains_key("expired-but-in-flight"),
            "a model with an in-flight request must survive regardless of its deadline"
        );
        assert!(
            mgr.running.contains_key("expired-but-forever"),
            "keep_alive: None (forever) must never be reaped"
        );
        assert!(
            mgr.running.contains_key("not-yet-expired"),
            "a model whose keep_alive deadline hasn't passed yet must survive"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn evict_other_models_evicts_everything_except_the_target_and_in_flight_models() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "the-model-being-loaded".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "idle-other-model".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
            mgr.running.insert(
                "busy-other-model".into(),
                running_model_fixture(None, Duration::from_secs(0), 1),
            );
        }

        let evicted_anything = evict_other_models(&state, "the-model-being-loaded").await;
        assert!(evicted_anything);

        let mgr = state.0.manager.lock().await;
        assert!(
            mgr.running.contains_key("the-model-being-loaded"),
            "the model ensure_model is trying to load isn't itself an eviction target"
        );
        assert!(
            !mgr.running.contains_key("idle-other-model"),
            "an idle other model should be evicted to free memory"
        );
        assert!(
            mgr.running.contains_key("busy-other-model"),
            "a model with an in-flight request must survive eviction, same as reap_idle_models"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn evict_other_models_reports_nothing_evicted_when_nothing_is_evictable() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "the-model-being-loaded".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
        }

        assert!(!evict_other_models(&state, "the-model-being-loaded").await);
    }

    /// Regression test: a cache hit must claim `in_flight`, exactly like
    /// a fresh load's own insert, so `enforce_max_loaded_models` can
    /// never see it as idle in the window between `ensure_model`
    /// returning and the caller's own `begin_activity`/
    /// `refresh_activity` — see `check_running`'s doc comment.
    #[tokio::test(flavor = "multi_thread")]
    async fn check_running_claims_in_flight_on_a_live_hit() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running
                .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
        }

        // Held, not dropped immediately — otherwise its own release
        // could already have run by the next line.
        let (_, _guard) = check_running(&state, "m").await.unwrap();
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            1,
            "a cache hit must claim in_flight so it can't be evicted before the caller claims it"
        );
    }

    /// Regression test: the claim `check_running`/`ensure_model` hands
    /// back must release itself even if the caller's task is dropped
    /// before ever reaching `begin_activity`/`refresh_activity` — the
    /// whole point of returning an [`ActivityGuard`] instead of a plain
    /// bool/count.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unclaimed_activity_guard_still_releases_on_drop() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running
                .insert("m".into(), running_model_fixture(None, Duration::ZERO, 0));
        }

        let (_, guard) = check_running(&state, "m").await.unwrap();
        assert_eq!(state.0.manager.lock().await.running["m"].in_flight, 1);

        drop(guard); // simulates the caller's task being cancelled here
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            0,
            "the claim must still be released even if never handed off"
        );
    }

    #[test]
    fn parse_max_queue_defaults_to_ollamas_own_512_on_anything_unparseable() {
        assert_eq!(parse_max_queue(None), 512);
        assert_eq!(parse_max_queue(Some("")), 512);
        assert_eq!(parse_max_queue(Some("garbage")), 512);
        assert_eq!(parse_max_queue(Some("10")), 10);
        // Unlike most other LLMMAN_*/parse_* pairs, an explicit "0" is a
        // real value (disables the bound), not treated as unset.
        assert_eq!(parse_max_queue(Some("0")), 0);
    }

    #[test]
    fn parse_max_loaded_models_defaults_to_unbounded_on_anything_unparseable() {
        assert_eq!(parse_max_loaded_models(None), 0);
        assert_eq!(parse_max_loaded_models(Some("")), 0);
        assert_eq!(parse_max_loaded_models(Some("garbage")), 0);
        assert_eq!(parse_max_loaded_models(Some("3")), 3);
    }

    #[test]
    fn try_admit_rejects_once_the_cap_is_reached_and_releases_on_drop() {
        // A dedicated counter, not the real PENDING_REQUESTS — isolates
        // this from other tests running in parallel.
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        let first =
            try_admit_against(&COUNTER, 1).expect("first admission under the cap must succeed");
        assert!(
            try_admit_against(&COUNTER, 1).is_err(),
            "a second admission at the cap must be rejected"
        );

        drop(first);
        assert!(
            try_admit_against(&COUNTER, 1).is_ok(),
            "dropping an admitted guard must free its slot for the next caller"
        );
    }

    #[test]
    fn try_admit_with_a_zero_cap_admits_one_at_a_time_not_none_or_unbounded() {
        // Approximates Ollama's own unbuffered pendingReqCh at
        // OLLAMA_MAX_QUEUE=0 — neither "reject everything" nor
        // "unbounded".
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let first = try_admit_against(&COUNTER, 0).expect("one admission must still succeed");
        assert!(
            try_admit_against(&COUNTER, 0).is_err(),
            "a second concurrent admission must still be rejected"
        );
        drop(first);
        assert!(
            try_admit_against(&COUNTER, 0).is_ok(),
            "dropping the first must free the slot for the next caller"
        );
    }

    #[test]
    fn default_allowed_origins_covers_every_scheme_and_localhost_spelling() {
        let origins = default_allowed_origins();
        for expected in [
            "http://localhost:*",
            "https://localhost:*",
            "http://127.0.0.1:*",
            "https://127.0.0.1:*",
            "http://0.0.0.0:*",
            "https://0.0.0.0:*",
            "http://[::1]:*",
            "https://[::1]:*",
        ] {
            assert!(
                origins.iter().any(|o| o == expected),
                "missing default origin pattern {expected:?}"
            );
        }
    }

    #[test]
    fn origin_matches_a_trailing_wildcard_port_pattern() {
        assert!(origin_matches(
            "http://localhost:3000",
            "http://localhost:*"
        ));
        assert!(origin_matches("http://localhost:1", "http://localhost:*"));
        assert!(origin_matches("http://localhost:", "http://localhost:*"));
        assert!(!origin_matches(
            "http://evil.example:3000",
            "http://localhost:*"
        ));
        assert!(!origin_matches(
            "http://localhost.evil.example",
            "http://localhost:*"
        ));
    }

    #[test]
    fn origin_matches_a_wildcard_anywhere_in_the_pattern() {
        // Subdomain wildcard, mirroring gin-contrib/cors's own
        // AllowWildcard (not just llmman's default `:*` port entries).
        assert!(origin_matches(
            "https://foo.example.com",
            "https://*.example.com"
        ));
        assert!(!origin_matches(
            "https://example.com",
            "https://*.example.com"
        ));
        // A bare "*" allows every origin.
        assert!(origin_matches("https://anything.at.all", "*"));
        // More than one '*' never matches.
        assert!(!origin_matches("https://example.com", "https://*.*.com"));
    }

    #[test]
    fn origin_matches_a_plain_pattern_only_byte_for_byte() {
        assert!(origin_matches("https://example.com", "https://example.com"));
        assert!(!origin_matches(
            "https://example.com:8080",
            "https://example.com"
        ));
        assert!(!origin_matches("http://example.com", "https://example.com"));
    }

    #[test]
    fn allowed_origins_from_env_always_includes_the_localhost_defaults() {
        let origins = allowed_origins_from_env();
        assert!(origins.iter().any(|o| o == "http://localhost:*"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_is_a_no_op_when_unbounded() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "m1".into(),
                running_model_fixture(None, Duration::from_secs(0), 0),
            );
        }
        assert!(enforce_max_loaded_models(&state, 0).await.is_ok());
        assert_eq!(state.0.manager.lock().await.running.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_evicts_the_least_recently_active_idle_model_first() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "oldest-idle".into(),
                running_model_fixture(None, Duration::from_secs(300), 0),
            );
            mgr.running.insert(
                "newest-idle".into(),
                running_model_fixture(None, Duration::from_secs(1), 0),
            );
        }

        // Already at the cap (2 running, max_loaded 2) — a caller about
        // to insert a third must first free exactly one slot.
        assert!(enforce_max_loaded_models(&state, 2).await.is_ok());

        let mgr = state.0.manager.lock().await;
        assert_eq!(mgr.running.len(), 1);
        assert!(
            !mgr.running.contains_key("oldest-idle"),
            "the least-recently-active idle model must be evicted first"
        );
        assert!(mgr.running.contains_key("newest-idle"));
    }

    /// Regression test: the freed slot from an eviction must already be
    /// reserved (`pending_loads`) for the caller that triggered it, in
    /// the same locked step as the removal — not left open for a
    /// concurrent caller to steal while `stop_and_wait` is in flight.
    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_reserves_its_own_slot_when_evicting() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "victim".into(),
                running_model_fixture(None, Duration::from_secs(300), 0),
            );
        }

        let guard = enforce_max_loaded_models(&state, 1).await.unwrap();
        let mgr = state.0.manager.lock().await;
        assert_eq!(mgr.running.len(), 0, "the sole idle model must be evicted");
        assert_eq!(
            mgr.pending_loads, 1,
            "the freed slot must already be reserved for this caller"
        );
        drop(mgr);
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_rejects_with_503_when_every_loaded_model_is_busy() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "busy-1".into(),
                running_model_fixture(None, Duration::from_secs(0), 1),
            );
        }

        let err = enforce_max_loaded_models(&state, 1)
            .await
            .expect_err("every model at/over the cap is busy, so this must reject");
        assert_eq!(err.1, StatusCode::SERVICE_UNAVAILABLE);

        // Nothing evicted — a busy model must survive.
        assert_eq!(state.0.manager.lock().await.running.len(), 1);
    }

    /// Regression test: a second concurrent load of a *different* model
    /// must not also pass the `max_loaded` check while a first load is
    /// still pending (not yet in `running`) — see
    /// `enforce_max_loaded_models`'s doc comment on the reservation this
    /// closes a race on.
    #[tokio::test(flavor = "multi_thread")]
    async fn enforce_max_loaded_models_reserves_a_pending_slot_for_an_in_flight_load() {
        let state = test_state();
        let guard1 = enforce_max_loaded_models(&state, 1).await.unwrap();
        assert_eq!(state.0.manager.lock().await.pending_loads, 1);

        let state2 = state.clone();
        let second = tokio::spawn(async move { enforce_max_loaded_models(&state2, 1).await });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !second.is_finished(),
            "a second load must wait for the first reservation, not double up on it"
        );

        drop(guard1);
        tokio::time::timeout(Duration::from_secs(2), second)
            .await
            .expect("second load must proceed once the first reservation is released")
            .unwrap()
            .expect("second load must succeed once a slot is actually free");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn begin_activity_marks_in_flight_and_its_drop_releases_it_and_updates_keep_alive() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // in_flight: 1, as ensure_model's own claim (fresh load or
            // cache hit — either way it always claims one) already left
            // it before this test's begin_activity call, same as a real
            // handler would see it.
            mgr.running.insert(
                "m".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        let claim = ActivityGuard::new(&state, "m");
        let guard = begin_activity(claim, Some(Some(Duration::from_secs(42)))).await;
        {
            let mgr = state.0.manager.lock().await;
            let m = &mgr.running["m"];
            assert_eq!(
                m.in_flight, 1,
                "begin_activity must not add a second claim on top of ensure_model's own"
            );
            assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
        }

        drop(guard);
        // ActivityGuard::drop can't be async, so it spawns a task to
        // finish the update — give it a moment to run.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mgr = state.0.manager.lock().await;
        let m = &mgr.running["m"];
        assert_eq!(
            m.in_flight, 0,
            "dropping the guard must release the in-flight count"
        );
        assert_eq!(m.keep_alive, Some(Duration::from_secs(42)));
    }

    /// Regression test: a `None` `keep_alive` override (what the
    /// OpenAI-compatible and Anthropic Messages routes pass, since
    /// neither has a `keep_alive` field of its own to read one from) must
    /// leave a model's existing `keep_alive` completely untouched, both
    /// immediately and on the guard's drop — e.g. a model pinned via
    /// `/api/chat`'s `keep_alive: -1` ("never unload") must not have that
    /// silently downgraded to the daemon default just because an
    /// OpenAI-compatible request also happens to hit it. `last_active`
    /// (the idle clock) is still expected to refresh either way — a
    /// `None` override only means "don't touch keep_alive", not "don't
    /// count as activity".
    #[tokio::test(flavor = "multi_thread")]
    async fn begin_activity_with_no_override_never_touches_an_existing_keep_alive() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // "Forever" — as if pinned via `/api/chat`'s `keep_alive: -1`.
            // in_flight: 1, as ensure_model's own prior claim.
            mgr.running.insert(
                "m".into(),
                running_model_fixture(None, Duration::from_secs(600), 1),
            );
        }

        let claim = ActivityGuard::new(&state, "m");
        let guard = begin_activity(claim, None).await;
        {
            let mgr = state.0.manager.lock().await;
            let m = &mgr.running["m"];
            assert_eq!(m.in_flight, 1);
            assert_eq!(
                m.keep_alive, None,
                "a None override must not touch the model's existing keep_alive"
            );
            assert!(
                m.last_active.elapsed() < Duration::from_secs(600),
                "the idle clock must still refresh even without a keep_alive override"
            );
        }

        drop(guard);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mgr = state.0.manager.lock().await;
        assert_eq!(
            mgr.running["m"].keep_alive, None,
            "dropping the guard must still leave keep_alive untouched"
        );
    }

    /// Regression test: the load-only `/api/generate` path calls
    /// `refresh_activity` instead of `begin_activity` — it must release
    /// `ensure_model`'s own provisional claim itself, since no
    /// `ActivityGuard` will ever do it.
    #[tokio::test(flavor = "multi_thread")]
    async fn refresh_activity_releases_ensure_models_own_claim() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "m".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        refresh_activity(ActivityGuard::new(&state, "m"), None).await;
        // The guard's own Drop (releasing the claim) spawns a task —
        // give it a moment to run, same as every other ActivityGuard
        // drop test in this file.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            state.0.manager.lock().await.running["m"].in_flight,
            0,
            "refresh_activity must release ensure_model's claim, with no guard to do it later"
        );
    }

    /// Regression test for `serve_async`'s `ServeArgs::model` pre-load:
    /// without an explicit pin, a freshly loaded model sits at the
    /// daemon default `keep_alive` (5 minutes) — the idle reaper would
    /// unload a model asked for on the command line before it's ever
    /// actually used, defeating the whole point of pre-loading it.
    /// `refresh_activity(guard, None)` (what the pre-load task now calls
    /// right after `ensure_model` succeeds) must pin it to "never
    /// unload" instead.
    #[tokio::test(flavor = "multi_thread")]
    async fn refresh_activity_with_none_pins_a_model_to_never_unload() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            // in_flight: 1, as ensure_model's own prior claim.
            mgr.running.insert(
                "preloaded".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 1),
            );
        }

        refresh_activity(ActivityGuard::new(&state, "preloaded"), None).await;

        let mgr = state.0.manager.lock().await;
        assert_eq!(
            mgr.running["preloaded"].keep_alive, None,
            "a pre-loaded model must be pinned to never unload, not left at the daemon default"
        );
    }

    // -- /api/chat's message-less load/unload idiom ---------------------------

    /// Deserialized, not struct-literal, so `stream` picks up its serde
    /// default of `true` — both branches must answer with a single JSON
    /// object even then.
    fn chat_request(body: serde_json::Value) -> OllamaChatRequest {
        serde_json::from_value(body).expect("valid OllamaChatRequest")
    }

    async fn chat_response_json(resp: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Pins the wire shape against ollama 0.32.6, which answers both
    /// message-less forms with exactly this body. The `message` object must
    /// carry `role` and `content` and nothing else — `thinking`, `images`,
    /// `tool_calls` and `tool_name` are all `skip_serializing_if`, and a
    /// client comparing against ollama's reply would see any extra key.
    #[test]
    fn empty_chat_chunk_matches_ollamas_message_less_reply() {
        let value = serde_json::to_value(empty_chat_chunk("m:latest".into(), "load")).unwrap();

        assert_eq!(value["model"], "m:latest");
        assert_eq!(value["done"], true);
        assert_eq!(value["done_reason"], "load");
        assert_eq!(value["message"]["role"], "assistant");
        assert_eq!(value["message"]["content"], "");
        assert_eq!(
            value["message"].as_object().unwrap().len(),
            2,
            "ollama's message-less reply carries only role and content"
        );
        assert!(value.get("created_at").is_some());
    }

    /// `{"messages": [], "keep_alive": 0}` is ollama's unload idiom — see
    /// `handle_ollama_chat` for what the request did before this branch
    /// existed. Asserts both halves of the contract that regressed: the
    /// reply names the unload, and the model actually leaves the manager.
    #[tokio::test(flavor = "multi_thread")]
    async fn ollama_chat_with_no_messages_and_keep_alive_zero_unloads_the_model() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m:latest",
                "messages": [],
                "keep_alive": 0,
            }))),
        )
        .await
        .expect("unload must not error");

        let value = chat_response_json(resp).await;
        assert_eq!(value["done_reason"], "unload");
        assert_eq!(value["message"]["content"], "");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "keep_alive: 0 with no messages must actually unload the model"
        );
    }

    /// `test_state`'s store path is an empty temp dir, so `canonical_ref`
    /// finds nothing and returns the reference untouched — the same
    /// position a real daemon is in once the model has been removed from
    /// the store while still running. `default_tag` has to supply the
    /// `:latest` on its own, or the remove looks up `docker.io/ai/m` and
    /// misses the entry entirely while still reporting `"unload"`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_tagless_unload_still_finds_a_model_running_under_latest() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m",
                "messages": [],
                "keep_alive": 0,
            }))),
        )
        .await
        .expect("unload must not error");

        assert_eq!(chat_response_json(resp).await["done_reason"], "unload");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "a tagless unload must reach the model stored under :latest"
        );
    }

    /// The store, not `running`, separates a model llmman has no record of
    /// from one it holds but has not loaded: ollama 404s the first and
    /// plainly succeeds the second, and `llmman stop` renders only that
    /// 404 as an error of its own. `test_state`'s store is an empty temp
    /// directory, so nothing resolves in it.
    #[tokio::test(flavor = "multi_thread")]
    async fn unloading_a_model_llmman_does_not_have_is_a_404() {
        let state = test_state();
        let err = unload_model(&state, "docker.io/ai/nothing-here")
            .await
            .expect_err("an unknown model must not report a successful unload");
        assert_eq!(err.1, StatusCode::NOT_FOUND);
    }

    /// The 404 above is keyed on a model being absent from `running` and
    /// from the store, and a provider-routed one is served elsewhere, so
    /// it is in neither. Naming it in an unload still has to succeed.
    #[tokio::test(flavor = "multi_thread")]
    async fn unloading_a_provider_routed_model_is_not_a_404() {
        let state = test_state();
        unload_model(&state, "llmman.provider/openrouter/qwen/qwen3-coder")
            .await
            .expect("a provider-routed unload must succeed");
    }

    /// A model removed from the store while it is still loaded has to stay
    /// unloadable — `running` is consulted before the store for exactly
    /// this case, or the 404 above would strand a live `llama-server` with
    /// no way to stop it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_gone_from_the_store_but_still_loaded_unloads_without_a_404() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/orphan:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        unload_model(&state, "docker.io/ai/orphan")
            .await
            .expect("a loaded model must unload even with nothing in the store");

        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/orphan:latest"),
            "the running entry must be gone"
        );
    }

    /// An unload that arrives while a first load of the same model is in
    /// flight blocks on the load lock. By the time it runs, the pull has
    /// landed and the loader has inserted under the key `canonical_ref`
    /// now refines to, which can differ from the key both sides locked
    /// on. Resolving before the lock and removing that stale spelling
    /// misses the entry, finds the pulled model in the store, and reports
    /// success with the model still loaded.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_waiting_on_a_load_removes_the_key_that_load_inserted() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-unload-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());

        // What ensure_model locks on before its pull, and what load_identity
        // resolves to whatever the store holds.
        let pre_pull_key = "docker.io/ai/m:latest";
        let loading = acquire_load_lock(pre_pull_key).await;

        let unloader = state.clone();
        let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
        // Let the unload reach the lock and park on it.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // The pull lands with the tagless spelling as the stored reference,
        // and the loader inserts under that refined key.
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:0000".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m")
            .unwrap();
        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
        drop(loading);

        unload
            .await
            .unwrap()
            .expect("the unload must not error once the load releases the lock");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m"),
            "the unload must remove the key the load inserted, not the one it resolved before waiting"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The wider window: the pull has landed but the loader, still holding
    /// its lock, has not yet inserted into `running`. An unload resolving
    /// through the store now gets the refined spelling and locks on that
    /// instead, so nothing makes it wait, and the sibling test's outcome
    /// follows before the loader's entry even exists. Locking on
    /// `load_identity`, which does not consult the store, parks it.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unload_after_the_pull_but_before_the_insert_still_waits_for_the_load() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-unload-postpull-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = test_state_at(dir.clone());

        // The pull has already recorded the tagless spelling...
        let desc = crate::storage::oci::Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:0000".into(),
            size: 0,
            annotations: None,
        };
        OciStore::open(&dir)
            .unwrap()
            .tag(desc, "docker.io/ai/m")
            .unwrap();
        // ...and the loader still holds the lock it took before pulling.
        let loading = acquire_load_lock(&load_identity("docker.io/ai/m").unwrap()).await;

        let unloader = state.clone();
        let unload = tokio::spawn(async move { unload_model(&unloader, "docker.io/ai/m").await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !unload.is_finished(),
            "the unload must block on the load in flight, not resolve past it"
        );

        state.0.manager.lock().await.running.insert(
            "docker.io/ai/m".into(),
            running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
        );
        drop(loading);

        unload
            .await
            .unwrap()
            .expect("the unload must succeed once the load releases");
        assert!(
            !state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m"),
            "the entry the load inserted must be gone"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pins which requests name the unload sentinel: every zero form
    /// `parse_keep_alive_value` accepts, and nothing else — an absent
    /// field least of all, since that is what a message-less preload
    /// sends and what `LLMMAN_KEEP_ALIVE=0` used to turn into an eviction.
    ///
    /// This pins the contract, not the regression. The old and new
    /// predicates differ only in whether they consult
    /// `default_keep_alive`, which reads a process-wide environment
    /// variable the `resolve_keep_alive` tests in this module read too;
    /// telling them apart in-process would mean mutating it underneath
    /// those tests. The regression itself was reproduced against a real
    /// daemon started with `LLMMAN_KEEP_ALIVE=0`.
    #[test]
    fn only_a_keep_alive_the_request_actually_carries_means_unload() {
        assert!(
            !is_explicit_unload(&None),
            "an absent field is not an unload"
        );
        assert!(is_explicit_unload(&Some(serde_json::json!(0))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0"))));
        assert!(is_explicit_unload(&Some(serde_json::json!("0s"))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(300))));
        assert!(!is_explicit_unload(&Some(serde_json::json!(-1))));
        assert!(
            !is_explicit_unload(&Some(serde_json::json!("garbage"))),
            "an unparseable value leaves the daemon default deciding how long \
             to keep the model, not whether to keep it"
        );
    }

    /// `{"messages": []}` alone is ollama's pre-load idiom: load the model,
    /// answer with an empty message, generate nothing. `ensure_model` short-
    /// circuits at `check_running` for the already-running fixture, so this
    /// exercises the handler branch without a backend. Before this branch
    /// existed the same request returned arbitrary generated prose with
    /// `done_reason: "stop"`.
    #[tokio::test(flavor = "multi_thread")]
    async fn ollama_chat_with_no_messages_loads_without_generating() {
        let state = test_state();
        {
            let mut mgr = state.0.manager.lock().await;
            mgr.running.insert(
                "docker.io/ai/m:latest".into(),
                running_model_fixture(Some(DEFAULT_KEEP_ALIVE), Duration::ZERO, 0),
            );
        }

        let resp = handle_ollama_chat(
            State(state.clone()),
            HeaderMap::new(),
            Json(chat_request(serde_json::json!({
                "model": "docker.io/ai/m:latest",
                "messages": [],
            }))),
        )
        .await
        .expect("pre-load must not error");

        let value = chat_response_json(resp).await;
        assert_eq!(value["done_reason"], "load");
        assert_eq!(value["message"]["content"], "");
        assert!(
            state
                .0
                .manager
                .lock()
                .await
                .running
                .contains_key("docker.io/ai/m:latest"),
            "a pre-load must leave the model loaded"
        );
    }

    #[test]
    fn parse_context_length_accepts_a_plain_number_and_rejects_everything_else() {
        assert_eq!(parse_context_length(Some("32768")), Some(32768));
        assert_eq!(parse_context_length(Some(" 32768 \n")), Some(32768));
        assert_eq!(parse_context_length(None), None);
        assert_eq!(parse_context_length(Some("")), None);
        assert_eq!(parse_context_length(Some("not-a-number")), None);
        assert_eq!(parse_context_length(Some("-1")), None);
    }

    #[test]
    fn parse_flash_attention_accepts_llama_server_and_ollama_spellings() {
        // llama-server's own vocabulary passes straight through.
        assert_eq!(parse_flash_attention(Some("on")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("off")), Some("off".into()));
        assert_eq!(parse_flash_attention(Some("auto")), Some("auto".into()));
        // Ollama's OLLAMA_FLASH_ATTENTION boolean spelling is translated.
        assert_eq!(parse_flash_attention(Some("1")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("true")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("0")), Some("off".into()));
        assert_eq!(parse_flash_attention(Some("false")), Some("off".into()));
        // Case-insensitive, whitespace-tolerant.
        assert_eq!(parse_flash_attention(Some(" ON \n")), Some("on".into()));
        // Unset/empty leaves llama-server's own default untouched.
        assert_eq!(parse_flash_attention(None), None);
        assert_eq!(parse_flash_attention(Some("")), None);
        assert_eq!(parse_flash_attention(Some("   ")), None);
    }

    #[test]
    fn parse_kv_cache_type_trims_whitespace_and_treats_empty_as_unset() {
        assert_eq!(parse_kv_cache_type(Some("q8_0")), Some("q8_0".into()));
        assert_eq!(parse_kv_cache_type(Some(" q4_0 \n")), Some("q4_0".into()));
        assert_eq!(parse_kv_cache_type(None), None);
        assert_eq!(parse_kv_cache_type(Some("")), None);
        assert_eq!(parse_kv_cache_type(Some("   ")), None);
    }

    #[test]
    fn parse_sched_spread_maps_truthy_and_falsey_spellings_to_split_mode() {
        for truthy in ["1", "true", "yes", "on", "layer", " ON \n"] {
            assert_eq!(
                parse_sched_spread(Some(truthy)),
                Some("layer"),
                "input {truthy:?}"
            );
        }
        for falsey in ["0", "false", "no", "off", "none", " OFF \n"] {
            assert_eq!(
                parse_sched_spread(Some(falsey)),
                Some("none"),
                "input {falsey:?}"
            );
        }
    }

    #[test]
    fn parse_sched_spread_leaves_llama_servers_own_default_untouched_when_unset_or_unparseable() {
        assert_eq!(parse_sched_spread(None), None);
        assert_eq!(parse_sched_spread(Some("")), None);
        assert_eq!(parse_sched_spread(Some("   ")), None);
        assert_eq!(parse_sched_spread(Some("garbage")), None);
    }

    #[test]
    fn parse_num_parallel_accepts_a_positive_integer() {
        assert_eq!(parse_num_parallel(Some("4")), Some(4));
        assert_eq!(parse_num_parallel(Some(" 1 ")), Some(1));
    }

    #[test]
    fn parse_num_parallel_rejects_zero_and_unparseable_values() {
        assert_eq!(parse_num_parallel(Some("0")), None);
        assert_eq!(parse_num_parallel(None), None);
        assert_eq!(parse_num_parallel(Some("")), None);
        assert_eq!(parse_num_parallel(Some("-1")), None);
        assert_eq!(parse_num_parallel(Some("garbage")), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_list_count_counts_ids_and_inclusive_ranges() {
        // (kernel CPU list, expected count)
        let cases = [
            ("0-15\n", Some(16)),
            ("0", Some(1)),
            ("0,4-7\n", Some(5)),
            ("0-3,8-11\n", Some(8)),
            // Malformed or empty content fails closed: the caller then
            // passes no --threads and llama-server autodetects.
            ("", None),
            ("\n", None),
            ("3-1", None),
            ("0-x", None),
            ("a", None),
            ("0,,2", None),
            // Range length would overflow u32; no kernel emits this.
            ("0-4294967295", None),
            ("0-4294967295,4", None),
        ];
        for (list, expected) in &cases {
            assert_eq!(&cpu_list_count(list), expected, "list={list:?}");
        }
    }

    #[test]
    fn backend_ctx_size_scales_by_num_parallel_so_each_slot_keeps_the_full_context() {
        // llama-server splits one --ctx-size evenly across every
        // --parallel slot, so this must scale up to compensate —
        // matching Ollama's own NumCtx * numParallel.
        assert_eq!(backend_ctx_size(Some(4096), Some(4)), Some(16384));
        assert_eq!(backend_ctx_size(Some(4096), Some(1)), Some(4096));
        assert_eq!(backend_ctx_size(Some(4096), None), Some(4096));
        assert_eq!(backend_ctx_size(None, Some(4)), None);
        assert_eq!(backend_ctx_size(None, None), None);
    }

    #[test]
    fn backend_ctx_size_saturates_instead_of_overflowing() {
        assert_eq!(backend_ctx_size(Some(u32::MAX), Some(2)), Some(u32::MAX));
    }

    #[test]
    fn effective_num_parallel_drops_to_none_without_a_ctx_size_to_scale() {
        // Nothing to scale --parallel against, so don't forward it at
        // all rather than let llama-server silently divide the model's
        // own trained context across slots.
        assert_eq!(effective_num_parallel(None, Some(4)), None);
        assert_eq!(effective_num_parallel(Some(4096), Some(4)), Some(4));
        assert_eq!(effective_num_parallel(Some(4096), None), None);
        assert_eq!(effective_num_parallel(None, None), None);
    }

    #[test]
    fn supports_context_shift_disables_only_for_deepseek_family_models() {
        assert!(!supports_context_shift("deepseek-v3:latest"));
        assert!(!supports_context_shift("deepseek-r1:70b"));
        assert!(!supports_context_shift("DeepSeek-V2.5:latest")); // case-insensitive
        assert!(supports_context_shift("qwen3.5:latest"));
        assert!(supports_context_shift("gpt-oss:20b"));
    }

    #[test]
    fn next_ctx_size_after_oom_halves_from_the_vram_tiered_default_down_to_the_floor() {
        // The default_ctx_size_for(<=46GiB) tier — see hostgpu.rs.
        assert_eq!(next_ctx_size_after_oom(Some(65536)), Some(32768));
        assert_eq!(next_ctx_size_after_oom(Some(32768)), Some(16384));
        // At (or below) the floor, no further shrink is offered.
        assert_eq!(next_ctx_size_after_oom(Some(16384)), None);
        assert_eq!(next_ctx_size_after_oom(Some(8192)), None);
    }

    #[test]
    fn next_ctx_size_after_oom_starts_an_unbounded_ctx_size_at_an_explicit_ceiling() {
        // ctx_size: None means "defer to the model's own trained
        // context" (see hostgpu::default_ctx_size) — nothing to halve,
        // so the first retry pins an explicit starting point instead.
        assert_eq!(next_ctx_size_after_oom(None), Some(65536));
    }

    #[test]
    fn looks_like_oom_matches_known_allocation_failure_phrasings() {
        for msg in [
            "ggml_backend_alloc_ctx_tensors_from_buft: failed to allocate CUDA0 buffer of size 123",
            "llama_kv_cache: failed to allocate buffer for kv cache",
            "CUDA error: out of memory",
            "terminate called after throwing an instance of 'std::bad_alloc'",
            "cudaMalloc failed: out of memory",
        ] {
            assert!(looks_like_oom(msg), "expected OOM match for {msg:?}");
        }
    }

    #[test]
    fn looks_like_oom_does_not_flag_unrelated_startup_failures() {
        for msg in [
            "error while loading shared libraries: libcuda.so.1: cannot open shared object file",
            "error loading model: unknown architecture 'not-a-real-arch'",
            "error: unknown argument: --not-a-real-flag",
        ] {
            assert!(!looks_like_oom(msg), "unexpected OOM match for {msg:?}");
        }
    }

    /// Regression test for the Claude Code bug described on
    /// `build_anthropic_messages`'s own doc comment: a `system`-role
    /// message anywhere in `messages` (not just the top-level `system`
    /// field) must be folded into one message at index 0, never left in
    /// place, or llama.cpp's chat templates raise "System message must be
    /// at the beginning" on the second one.
    #[test]
    fn build_anthropic_messages_merges_system_role_messages_anywhere_in_the_conversation() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "system": [{"type": "text", "text": "leading system prompt"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "a mid-conversation reminder"},
                {"role": "user", "content": "bye"}
            ]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(
            messages,
            vec![
                OAIMessage::text(
                    "system",
                    "leading system prompt\n\na mid-conversation reminder"
                ),
                OAIMessage::text("user", "hi"),
                OAIMessage::text("assistant", "hello"),
                OAIMessage::text("user", "bye"),
            ]
        );
    }

    #[test]
    fn build_anthropic_messages_with_no_system_content_has_no_leading_system_message() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(messages, vec![OAIMessage::text("user", "hi")]);
    }

    /// Regression test for the Codex tool-type bug described on
    /// `filter_non_function_tools`'s own doc comment.
    #[test]
    fn filter_non_function_tools_drops_non_function_entries_only() {
        let mut req = serde_json::json!({
            "tools": [
                {"type": "function", "name": "exec_command"},
                {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "close_agent"}]},
                {"type": "web_search"},
                {"type": "function", "name": "update_plan"}
            ]
        });

        filter_non_function_tools(&mut req);

        assert_eq!(
            req["tools"],
            serde_json::json!([
                {"type": "function", "name": "exec_command"},
                {"type": "function", "name": "update_plan"}
            ])
        );
    }

    #[test]
    fn filter_non_function_tools_is_a_no_op_without_a_tools_field() {
        let mut req = serde_json::json!({"model": "x"});
        filter_non_function_tools(&mut req);
        assert_eq!(req, serde_json::json!({"model": "x"}));
    }

    /// Regression test for the Codex Responses-API bug described on
    /// `consolidate_responses_instructions`'s own doc comment: a
    /// `developer`/`system`-role `input` item must be folded into
    /// `instructions` and removed from `input`, never left in place.
    #[test]
    fn consolidate_responses_instructions_folds_developer_and_system_input_items() {
        let mut req = serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "instructions": "top-level instructions",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "permissions instructions"}
                ]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "message", "role": "system", "content": "a plain-string system item"}
            ]
        });

        consolidate_responses_instructions(&mut req);

        assert_eq!(
            req["instructions"],
            "top-level instructions\n\npermissions instructions\n\na plain-string system item"
        );
        assert_eq!(
            req["input"],
            serde_json::json!([
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]}
            ])
        );
    }

    #[test]
    fn consolidate_responses_instructions_is_a_no_op_without_developer_or_system_items() {
        let mut req = serde_json::json!({
            "instructions": "top-level instructions",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let before = req.clone();
        consolidate_responses_instructions(&mut req);
        assert_eq!(req, before);
    }

    // -- Tests ported from ollama ---------------------------------------------
    //
    // The tests below are ported from ollama's own unit-test suites for the
    // equivalent conversion logic — file references point at ollama/ollama's
    // test files — adapted to llmman's own (narrower) semantics where the two
    // differ; each test's doc comment calls out any such adaptation.

    /// Ported from ollama's openai/openai_test.go
    /// (TestFromChatRequest_ReasoningEffort): a boolean `think` maps to
    /// `enable_thinking`, and a string thinking level
    /// ("low"/"medium"/"high"/"max") additionally maps to
    /// `reasoning_effort` — the jinja variable gpt-oss's and
    /// DeepSeek-V4's own chat templates read.
    #[test]
    fn think_to_chat_template_kwargs_maps_booleans_and_reasoning_levels() {
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(true))),
            Some(serde_json::json!({ "enable_thinking": true }))
        );
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::json!(false))),
            Some(serde_json::json!({ "enable_thinking": false }))
        );
        for level in ["low", "medium", "high", "max"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(level))),
                Some(serde_json::json!({
                    "enable_thinking": true,
                    "reasoning_effort": level,
                })),
                "string level {level:?}"
            );
        }
        // Anything other than the four known levels is a no-op — an
        // unrecognized value shouldn't be forwarded to the template
        // verbatim (see think_to_chat_template_kwargs's own comment).
        for not_a_level in ["", "  ", "verbose", "LOW"] {
            assert_eq!(
                think_to_chat_template_kwargs(&Some(serde_json::json!(not_a_level))),
                None,
                "string {not_a_level:?}"
            );
        }
        assert_eq!(think_to_chat_template_kwargs(&None), None);
        assert_eq!(
            think_to_chat_template_kwargs(&Some(serde_json::Value::Null)),
            None
        );
    }

    /// Ported from ollama's api/client_test.go (TestClientStream /
    /// TestClientDo malformed-payload cases) and openai streaming-chunk
    /// tests: each SSE payload either yields (content, thinking, done) or
    /// is skipped entirely (None) when malformed — a bad chunk must never
    /// abort the whole stream.
    /// Content and thinking each concatenate in order; non-SSE lines and
    /// the trailing `[DONE]` add nothing.
    #[test]
    fn fold_ollama_lines_concatenates_deltas() {
        let lines = [
            "",
            r#"data: {"choices":[{"delta":{"reasoning_content":"let me "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            ": keep-alive",
            r#"data: {"choices":[{"delta":{"content":", world"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "Hello, world");
        assert_eq!(fold.thinking.as_deref(), Some("let me think"));
        assert_eq!(fold.tool_calls, None);
        assert!(fold.done);
    }

    /// llama-server signals done twice and the decoder drains on the first,
    /// so folding the last `done` line's value would lose the tool calls.
    #[test]
    fn fold_ollama_lines_keeps_tool_calls_across_a_second_done_line() {
        let lines = [
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Ankara\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ]
        .map(String::from);
        let calls = fold_ollama_lines(lines)
            .tool_calls
            .expect("survive the trailing [DONE]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["city"], "Ankara");
    }

    /// A response with no output folds to empty values.
    #[test]
    fn fold_ollama_lines_handles_an_empty_response() {
        let lines = [r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#]
            .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert!(fold.content.is_empty() && fold.thinking.is_none() && fold.tool_calls.is_none());
        assert!(fold.done);
    }

    /// A backend that dies mid-response leaves no terminal chunk, which is
    /// the only signal the reply is truncated.
    #[test]
    fn fold_ollama_lines_reports_a_missing_terminal_chunk() {
        let lines =
            [r#"data: {"choices":[{"delta":{"content":"half a sen"},"finish_reason":null}]}"#]
                .map(String::from);
        let fold = fold_ollama_lines(lines);
        assert_eq!(fold.content, "half a sen");
        assert!(!fold.done, "no terminal chunk means the reply is truncated");
    }

    #[test]
    fn oai_chunk_to_content_ported_ollama_stream_decoding_cases() {
        // Plain content token, stream not finished.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#
            ),
            Some(("hi".into(), None, false))
        );
        // finish_reason "stop" marks the stream done.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#
            ),
            Some((String::new(), None, true))
        );
        // The [DONE] sentinel also marks the stream done.
        assert_eq!(
            oai_chunk_to_content("[DONE]"),
            Some((String::new(), None, true))
        );
        // llama-server's two reasoning field spellings both surface as
        // thinking: "reasoning_content" (Homebrew builds) and "thinking"
        // (git builds).
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), false))
        );
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"thinking":"hmm"},"finish_reason":null}]}"#
            ),
            Some((String::new(), Some("hmm".into()), false))
        );
        // An empty reasoning string is filtered out rather than surfaced.
        assert_eq!(
            oai_chunk_to_content(
                r#"{"choices":[{"delta":{"content":"x","reasoning_content":""},"finish_reason":null}]}"#
            ),
            Some(("x".into(), None, false))
        );
        // Malformed JSON and an empty choices array are skipped, not fatal.
        assert_eq!(oai_chunk_to_content("not json"), None);
        assert_eq!(oai_chunk_to_content(r#"{"choices":[]}"#), None);
    }

    /// Regression test guarding against exactly the leak CodeRabbit
    /// flagged on this PR: an `Engine::Mlx` backend is addressed by its
    /// real on-disk directory path (see `backend_wire_model`), and
    /// `mlx_lm.server` echoes whatever `"model"` value it received
    /// straight back into its own response — so a plain byte-for-byte
    /// relay would leak that internal path back to the client instead of
    /// the name it actually asked for. `set_response_model` is the one
    /// place both `rewrite_json_response_model` and
    /// `rewrite_sse_line_model` below delegate the actual field
    /// substitution to.
    #[test]
    fn set_response_model_overwrites_an_existing_model_field_and_leaves_a_missing_one_alone() {
        let mut with_model = serde_json::json!({"model": "/abs/path/to/model", "id": "x"});
        set_response_model(&mut with_model, "gemma4:latest");
        assert_eq!(
            with_model,
            serde_json::json!({"model": "gemma4:latest", "id": "x"})
        );

        let mut without_model = serde_json::json!({"id": "x"});
        set_response_model(&mut without_model, "gemma4:latest");
        assert_eq!(without_model, serde_json::json!({"id": "x"}));
    }

    #[test]
    fn rewrite_json_response_model_rewrites_a_json_body_and_leaves_every_other_field_alone() {
        let raw = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "id": "chatcmpl-1",
                "model": "/home/user/.local/share/llmman/cache/abcd/model-dir",
                "choices": [{"message": {"content": "hi"}}]
            }))
            .unwrap(),
        );
        let rewritten = rewrite_json_response_model(&raw, "gemma4:latest");
        let value: serde_json::Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "gemma4:latest");
        assert_eq!(value["id"], "chatcmpl-1");
        assert_eq!(value["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn rewrite_json_response_model_passes_non_json_bodies_through_unchanged() {
        // An error body, or any other shape this doesn't recognize —
        // must never be mangled or dropped just because it isn't JSON.
        let raw = Bytes::from_static(b"not json at all");
        assert_eq!(rewrite_json_response_model(&raw, "gemma4:latest"), raw);
    }

    #[test]
    fn rewrite_sse_line_model_rewrites_only_the_model_field_of_a_data_line() {
        let line = r#"data: {"id":"1","model":"/abs/path","choices":[{"delta":{"content":"h"}}]}"#;
        let rewritten = rewrite_sse_line_model(line, "gemma4:latest");
        let payload = rewritten.strip_prefix("data: ").expect("data: prefix");
        let value: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(value["model"], "gemma4:latest");
        assert_eq!(value["id"], "1");
        assert_eq!(value["choices"][0]["delta"]["content"], "h");
    }

    #[test]
    fn rewrite_sse_line_model_leaves_the_done_sentinel_and_blank_separators_untouched() {
        assert_eq!(
            rewrite_sse_line_model("data: [DONE]", "gemma4:latest"),
            "data: [DONE]"
        );
        assert_eq!(rewrite_sse_line_model("", "gemma4:latest"), "");
    }

    #[test]
    fn rewrite_sse_line_model_passes_a_non_json_data_line_through_unchanged() {
        assert_eq!(
            rewrite_sse_line_model("data: not json", "gemma4:latest"),
            "data: not json"
        );
    }

    /// Regression test for the other CodeRabbit finding this PR
    /// addresses: `/v1/embeddings` against an `Engine::Mlx` backend must
    /// fail fast with a clear reason (not a bare, unexplained 404 from
    /// forwarding to `mlx_lm.server`, which never gets a
    /// `--embedding-model` from `spawn_mlx_server` — see
    /// `proxy_openai_passthrough`'s own doc comment).
    #[tokio::test]
    async fn mlx_embeddings_unsupported_response_explains_why_and_names_the_model() {
        let resp = mlx_embeddings_unsupported_response("gemma4:latest");
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("gemma4:latest"));
        assert!(message.contains("mlx_lm.server"));
        assert!(message.contains("/v1/embeddings"));
    }

    #[test]
    fn raw_content_extractor_passes_plain_content_through_untouched() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process("hello there".into(), None),
            ("hello there".into(), None)
        );
        assert_eq!(
            ext.process(" friend".into(), None),
            (" friend".into(), None)
        );
    }

    /// Once a backend has ever supplied structured `thinking` on a
    /// stream, raw content must never be scanned again — even if it
    /// later happens to contain literal `<think>` text as part of a
    /// genuine reply (e.g. the model discussing the tag itself).
    #[test]
    fn raw_content_extractor_locks_into_passthrough_once_backend_thinking_seen() {
        let mut ext = RawContentExtractor::new();
        assert_eq!(
            ext.process(String::new(), Some("reasoning".into())),
            (String::new(), Some("reasoning".into()))
        );
        assert_eq!(
            ext.process("<think>literal text</think>".into(), None),
            ("<think>literal text</think>".into(), None)
        );
    }

    /// Regression test: a chunk still buffered in `Undetermined` (a
    /// strict prefix of a candidate tag, e.g. a lone `"<"`) must not be
    /// silently dropped when a *later* chunk turns out to carry
    /// backend-structured thinking instead — that transition previously
    /// overwrote `self` with `Passthrough` without ever draining it.
    #[test]
    fn raw_content_extractor_recovers_a_buffered_prefix_when_backend_thinking_appears_later() {
        let mut ext = RawContentExtractor::new();
        // "<" alone is a strict prefix of every candidate tag, so it's
        // held back rather than emitted.
        assert_eq!(ext.process("<".into(), None), (String::new(), None));
        // The backend now reports structured thinking on this chunk —
        // the buffered "<" must be prepended to this chunk's own content,
        // not lost.
        assert_eq!(
            ext.process("hello".into(), Some("reasoning".into())),
            ("<hello".into(), Some("reasoning".into()))
        );
        // Now locked into Passthrough: a later flush has nothing left to
        // recover.
        assert_eq!(ext.flush(), "");
    }

    #[test]
    fn raw_content_extractor_falls_back_to_plain_think_tags() {
        let mut ext = RawContentExtractor::new();
        let (c1, t1) = ext.process("<think>".into(), None);
        assert_eq!((c1, t1), (String::new(), None));
        let (c2, t2) = ext.process("hmm".into(), None);
        assert_eq!((c2, t2), (String::new(), Some("hmm".into())));
        let (c3, t3) = ext.process("</think>answer".into(), None);
        assert_eq!((c3, t3), ("answer".into(), None));
    }

    #[test]
    fn raw_content_extractor_falls_back_to_harmony_channels() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    #[test]
    fn raw_content_extractor_leaves_content_without_any_tag_untouched() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("just a normal reply".into(), None);
        assert_eq!(content, "just a normal reply");
        assert_eq!(thinking, None);
    }

    /// Regression test: a real streamed response hands this one token (or
    /// even one byte) at a time — the very first chunk of a harmony
    /// stream is never the whole `"<|channel|>..."` string at once, just
    /// its first byte, which is also a valid prefix of `<|start|>` and
    /// `<think>`. `Undetermined` must buffer across calls instead of
    /// deciding (wrongly, into `PlainThink`) from that first ambiguous
    /// byte alone.
    #[test]
    fn raw_content_extractor_buffers_across_calls_to_classify_a_token_split_harmony_stream() {
        let mut ext = RawContentExtractor::new();
        let whole = "<|start|>assistant<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>";
        let mut content = String::new();
        let mut thinking = String::new();
        for ch in whole.chars() {
            let mut buf = [0u8; 4];
            let (c, t) = ext.process(ch.encode_utf8(&mut buf).to_string(), None);
            content.push_str(&c);
            if let Some(t) = t {
                thinking.push_str(&t);
            }
        }
        assert_eq!(content, "the answer");
        assert_eq!(thinking, "thinking...");
    }

    /// Regression test: llama-server's own chat template already emits
    /// the assistant's `<|start|>assistant` preamble as part of the
    /// *prompt*, so a real raw completion stream for a gpt-oss-style
    /// model routinely starts directly at `<|channel|>`, never repeating
    /// `<|start|>` itself. Without priming the harmony parser via
    /// `add_implicit_start` for exactly this case, `HarmonyParser` would
    /// sit in `LookingForMessageStart` forever and never emit anything.
    #[test]
    fn raw_content_extractor_primes_harmony_when_a_stream_starts_mid_message() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process(
            "<|channel|>analysis<|message|>thinking...<|end|>\
             <|start|>assistant<|channel|>final<|message|>the answer<|end|>"
                .into(),
            None,
        );
        assert_eq!(content, "the answer");
        assert_eq!(thinking, Some("thinking...".into()));
    }

    /// Regression test: a reply that ends while `Undetermined` is still
    /// holding back a strict prefix of a candidate tag (here, the whole
    /// reply is just a lone `"<"`) must not silently lose that text —
    /// `flush` (called by `stream_ollama` on its `done` chunk) drains it.
    #[test]
    fn raw_content_extractor_flush_recovers_a_buffered_prefix_at_stream_end() {
        let mut ext = RawContentExtractor::new();
        let (content, thinking) = ext.process("<".into(), None);
        assert_eq!(content, "");
        assert_eq!(thinking, None);
        assert_eq!(ext.flush(), "<");
        // Idempotent: a second flush (mirroring the two `done` chunks a
        // real stream can produce) must not resurrect it.
        assert_eq!(ext.flush(), "");
    }

    /// `flush` is a no-op once a mode has been decided — that buffering
    /// is `thinking::Parser`/`harmony::HarmonyMessageHandler`'s own
    /// internal concern (see `RawContentExtractor::flush`'s own doc
    /// comment on why this mirrors real Ollama's own, identical
    /// limitation rather than a new gap).
    #[test]
    fn raw_content_extractor_flush_is_a_no_op_once_a_mode_is_decided() {
        let mut ext = RawContentExtractor::new();
        ext.process("just a normal reply".into(), None);
        assert_eq!(ext.flush(), "");

        let mut ext = RawContentExtractor::new();
        ext.process(String::new(), Some("reasoning".into()));
        assert_eq!(ext.flush(), "");
    }

    /// A multi-byte character split across chunk boundaries must survive.
    /// Decoding each chunk on its own turned the split halves into U+FFFD.
    #[test]
    fn bytes_to_lines_preserves_utf8_split_across_chunks() {
        let line = "data: g\u{fc}nayd\u{131}n \u{1f600}";
        // One byte per chunk: every multi-byte character is split.
        let raw = format!("{line}\n");
        let chunks: Vec<reqwest::Result<Bytes>> = raw
            .as_bytes()
            .iter()
            .map(|b| Ok(Bytes::copy_from_slice(&[*b])))
            .collect();
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(lines, vec![line.to_string()]);
    }

    /// Ported from ollama's api/client_test.go (TestClientStream): SSE
    /// lines split across arbitrary TCP chunk boundaries must be
    /// reassembled, CRLF line endings trimmed, and a trailing
    /// unterminated line flushed when the stream ends.
    #[test]
    fn bytes_to_lines_ported_ollama_client_stream_chunking() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            // One logical line split across two chunks.
            Ok(Bytes::from("data: {\"a\":")),
            // ...ending CRLF, plus a complete LF-terminated line.
            Ok(Bytes::from("1}\r\ndata: {\"b\":2}\n")),
            // A trailing line with no terminator at all.
            Ok(Bytes::from("data: tail")),
        ];
        let stream = bytes_to_lines(futures::stream::iter(chunks));
        let lines: Vec<String> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(
            lines,
            vec![
                "data: {\"a\":1}".to_string(),
                "data: {\"b\":2}".to_string(),
                "data: tail".to_string(),
            ]
        );
    }

    /// Ported from ollama's middleware/anthropic_test.go
    /// (TestAnthropicMessagesMiddleware's plain-string `system` case):
    /// Anthropic's `system` field is accepted as either a bare string or
    /// an array of content blocks, and both forms end up as the single
    /// leading system message.
    #[test]
    fn build_anthropic_messages_accepts_a_plain_string_system_field() {
        let req: AnthropicRequest = serde_json::from_value(serde_json::json!({
            "model": "docker.io/ai/qwen3.5:0.8b",
            "system": "you are a helpful assistant",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let messages = build_anthropic_messages(&req);

        assert_eq!(
            messages,
            vec![
                OAIMessage::text("system", "you are a helpful assistant"),
                OAIMessage::text("user", "hi"),
            ]
        );
    }

    /// Ported from ollama's middleware/anthropic_test.go content-block
    /// conversion cases: block-array content joins its text blocks in
    /// order and ignores non-text block types entirely.
    #[test]
    fn anthropic_content_as_text_joins_text_blocks_and_ignores_other_types() {
        let plain: AnthropicContent = serde_json::from_value(serde_json::json!("plain")).unwrap();
        assert_eq!(plain.as_text(), "plain");

        let blocks: AnthropicContent = serde_json::from_value(serde_json::json!([
            {"type": "text", "text": "a"},
            {"type": "image", "source": {"type": "base64", "data": "zzzz"}},
            {"type": "text", "text": "b"}
        ]))
        .unwrap();
        assert_eq!(blocks.as_text(), "ab");

        let empty: AnthropicContent = serde_json::from_value(serde_json::json!([])).unwrap();
        assert_eq!(empty.as_text(), "");
    }

    /// Ported from ollama's openai/responses_test.go polymorphic-input
    /// cases: a Responses-API input item's `content` is either a bare
    /// string or an array of text-bearing blocks (`input_text` /
    /// `output_text`), and anything else (a function_call item with no
    /// content, a non-string/array content) yields no text.
    #[test]
    fn responses_input_item_text_ported_ollama_polymorphic_input_cases() {
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": "plain"})),
            Some("plain".into())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"role": "user", "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"}
            ]})),
            Some("ab".into())
        );
        // Blocks without a text field contribute nothing.
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": [{"type": "input_image"}]})),
            Some(String::new())
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"type": "function_call", "name": "f"})),
            None
        );
        assert_eq!(
            responses_input_item_text(&serde_json::json!({"content": 42})),
            None
        );
    }

    /// Ported from ollama's server/routes_options_test.go concept
    /// (api.Options blob -> typed option values): numeric options are
    /// pulled out of the Ollama `options` blob by key, and missing keys,
    /// wrong-typed values, or an absent blob all yield None instead of
    /// erroring.
    #[test]
    fn option_extractors_ported_ollama_options_blob_cases() {
        let opts = Some(serde_json::json!({
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 128,
            "stop": ["### User:"]
        }));
        assert_eq!(opt_f64(&opts, "temperature"), Some(0.5));
        assert_eq!(opt_f64(&opts, "top_p"), Some(0.9));
        assert_eq!(opt_u32(&opts, "num_predict"), Some(128));
        // Missing key.
        assert_eq!(opt_f64(&opts, "repeat_penalty"), None);
        // Wrong type for the extractor.
        assert_eq!(opt_u32(&opts, "stop"), None);
        // No options blob at all.
        assert_eq!(opt_f64(&None, "temperature"), None);
        assert_eq!(opt_u32(&None, "num_predict"), None);
    }

    #[test]
    fn keyed_lock_is_per_key_and_release_only_drops_unreferenced_entries() {
        let registry: StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>> =
            StdMutex::new(HashMap::new());

        let a1 = keyed_lock(&registry, "model-a");
        let a2 = keyed_lock(&registry, "model-a");
        assert!(Arc::ptr_eq(&a1, &a2), "same key must return the same lock");

        let b = keyed_lock(&registry, "model-b");
        assert!(
            !Arc::ptr_eq(&a1, &b),
            "different keys must not share a lock"
        );

        // Caller 1 finishes and releases its own clone — but caller 2's
        // clone (a2) is still outstanding, so the entry must survive.
        drop(a1);
        release_keyed_lock(&registry, "model-a");
        assert!(registry.lock().unwrap().contains_key("model-a"));

        // Caller 2 finishes too — now only the registry itself references
        // it, so releasing removes the entry.
        drop(a2);
        release_keyed_lock(&registry, "model-a");
        assert!(!registry.lock().unwrap().contains_key("model-a"));

        drop(b);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_serializes_same_model_but_not_different_models() {
        let slow = load_lock("test-load-lock-slow-model");
        let guard = slow.lock().await; // simulates a mid-flight cold start

        // A different model's load must acquire immediately.
        let other = load_lock("test-load-lock-other-model");
        let _other_guard =
            tokio::time::timeout(std::time::Duration::from_millis(200), other.lock())
                .await
                .expect("a different model's load must not block on an unrelated one");

        // The same model's load must not acquire until the first releases.
        let same = load_lock("test-load-lock-slow-model");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), same.lock())
                .await
                .is_err(),
            "a second load of the same model must block while the first is in flight"
        );

        drop(guard);
        let same_guard = tokio::time::timeout(std::time::Duration::from_millis(200), same.lock())
            .await
            .expect("must acquire promptly once the first load releases");

        drop(same_guard);
        drop(_other_guard);
        drop(same);
        drop(other);
        drop(slow);
        release_load_lock("test-load-lock-slow-model");
        release_load_lock("test-load-lock-other-model");
    }

    /// Regression: aliases of an unpulled model must key into one lock
    /// (see `ensure_model`'s `default_tag` call).
    #[test]
    fn ensure_model_key_pipeline_converges_aliases_before_the_lock() {
        let tagless = crate::storage::default_tag(
            &crate::shortnames::resolve_ollama_api("regression-test-model").unwrap(),
        );
        let tagged = crate::storage::default_tag(
            &crate::shortnames::resolve_ollama_api("regression-test-model:latest").unwrap(),
        );
        assert_eq!(
            tagless, tagged,
            "tagless and :latest aliases must resolve to one key"
        );

        let a = load_lock(&tagless);
        let b = load_lock(&tagged);
        assert!(
            Arc::ptr_eq(&a, &b),
            "both aliases must take the same load lock"
        );

        drop(a);
        drop(b);
        release_load_lock(&tagless);
    }

    /// An invalid client ref is rejected at the top of `ensure_model`, before
    /// any resolve/pull/network work runs. The reference error is built as a
    /// 400 right at the resolve site (`AppError::bad_request`), so it must
    /// survive `into_response` unchanged.
    #[tokio::test]
    async fn ensure_model_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let err = ensure_model(&state, "hf.co/../x", None)
            .await
            .err()
            .expect("invalid ref must be rejected");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    /// /api/push validates the client ref before resolving it: an invalid
    /// ref returns a 400, matching /api/pull's early rejection.
    #[tokio::test]
    async fn handle_push_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaPushRequest {
            model: "hf.co/../x".to_string(),
            name: String::new(),
        };
        let resp = handle_push(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// /api/delete resolves (and so validates) the client ref before it ever
    /// opens the store: an invalid ref returns a 400 and touches nothing.
    #[tokio::test]
    async fn handle_delete_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaDeleteRequest {
            model: "hf.co//foo".to_string(),
            name: None,
        };
        let resp = handle_delete(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// /api/show resolves (and so validates) the client ref before it ever
    /// opens the store: an invalid ref returns a 400 and touches nothing.
    #[tokio::test]
    async fn handle_show_rejects_an_invalid_ref_with_400() {
        let state = test_state();
        let req = OllamaShowRequest {
            model: "hf:///foo".to_string(),
            name: None,
        };
        let resp = handle_show(State(state), Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Regression: a call site that drops its guard but not its own `Arc`
    /// clone before calling `release_load_lock` leaves the entry stuck.
    #[tokio::test]
    async fn load_lock_release_actually_removes_the_entry_once_unused() {
        let key = "test-load-lock-release-cleanup";
        let lock = load_lock(key);
        let guard = lock.lock().await;
        drop(guard);
        drop(lock);
        release_load_lock(key);
        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "release_load_lock must drop the registry entry once nothing else references it"
        );
    }

    /// Regression: aborting a task while it holds a `LoadLockGuard` must
    /// still release the registry entry. `acquire_load_lock`'s caller
    /// (`ensure_model`, the unload handler) can itself be cancelled by axum
    /// mid-`.await` (a dropped client connection) — code placed after an
    /// `.await` doesn't run in that case, so cleanup must live in `Drop`.
    #[tokio::test(flavor = "multi_thread")]
    async fn load_lock_guard_releases_on_task_cancellation() {
        let key = "test-load-lock-guard-cancel";
        let started = Arc::new(tokio::sync::Notify::new());
        let started_tx = started.clone();
        let handle = tokio::spawn(async move {
            let _guard = acquire_load_lock("test-load-lock-guard-cancel").await;
            started_tx.notify_one();
            std::future::pending::<()>().await;
        });
        started.notified().await;
        handle.abort();
        let _ = handle.await;

        assert!(
            !LOAD_LOCKS.lock().unwrap().contains_key(key),
            "aborting a task holding LoadLockGuard must still release the registry entry"
        );
    }

    /// Regression test for `OllamaPullRequest`'s `name` field: a body
    /// carrying only `{"name": "..."}` used to fail Axum's `Json`
    /// extraction outright — `model` was a required, non-default field —
    /// before this handler's own name-falls-back-to-model logic ever ran.
    #[test]
    fn ollama_pull_request_accepts_a_name_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }

    #[test]
    fn ollama_pull_request_accepts_a_model_only_body() {
        let req: OllamaPullRequest =
            serde_json::from_value(serde_json::json!({"model": "docker.io/ai/gemma4:E2B"}))
                .expect("a model-only body must still deserialize");
        assert_eq!(req.model, "docker.io/ai/gemma4:E2B");
        assert_eq!(req.name, "");
    }

    #[test]
    fn ollama_push_request_accepts_a_name_only_body() {
        let req: OllamaPushRequest =
            serde_json::from_value(serde_json::json!({"name": "docker.io/ai/gemma4:E2B"}))
                .expect("a name-only body must still deserialize");
        assert_eq!(req.model, "");
        assert_eq!(req.name, "docker.io/ai/gemma4:E2B");
    }

    // -- multipart_text_field (/v1/audio/transcriptions) ----------------------

    /// Builds a `multipart/form-data` body + matching `content-type`
    /// header out of `fields` (name, value) pairs — a hand-rolled encoder
    /// rather than a dependency, just enough to exercise
    /// `multipart_text_field` against real (if minimal) multipart wire
    /// format.
    fn multipart_body(fields: &[(&str, &str)]) -> (Bytes, HeaderMap) {
        let boundary = "llmman-test-boundary";
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{boundary}--\r\n"));

        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            format!("multipart/form-data; boundary={boundary}")
                .parse()
                .unwrap(),
        );
        (Bytes::from(body), headers)
    }

    #[tokio::test]
    async fn multipart_text_field_finds_a_named_field_among_several() {
        let (body, headers) = multipart_body(&[
            ("language", "en"),
            ("model", "docker.io/ai/whisper:latest"),
            ("response_format", "json"),
        ]);
        assert_eq!(
            multipart_text_field(&body, &headers, "model").await,
            Some("docker.io/ai/whisper:latest".to_string())
        );
        assert_eq!(
            multipart_text_field(&body, &headers, "language").await,
            Some("en".to_string())
        );
    }

    #[tokio::test]
    async fn multipart_text_field_leaves_the_original_body_untouched() {
        // Regression: multipart_text_field parses a *clone* of the body
        // for the field it wants — the original `Bytes` handed to
        // `proxy` afterward must still be the exact, complete multipart
        // payload (file bytes included), not something already partially
        // consumed by this lookup.
        let (body, headers) = multipart_body(&[("model", "m"), ("prompt", "hello")]);
        let before = body.clone();
        let _ = multipart_text_field(&body, &headers, "model").await;
        assert_eq!(body, before);
    }

    #[tokio::test]
    async fn multipart_text_field_is_none_for_a_missing_field_or_non_multipart_body() {
        let (body, headers) = multipart_body(&[("language", "en")]);
        assert_eq!(multipart_text_field(&body, &headers, "model").await, None);

        let plain_body = Bytes::from_static(b"{\"model\":\"m\"}");
        let mut json_headers = HeaderMap::new();
        json_headers.insert("content-type", "application/json".parse().unwrap());
        assert_eq!(
            multipart_text_field(&plain_body, &json_headers, "model").await,
            None
        );

        // No content-type header at all.
        assert_eq!(
            multipart_text_field(&plain_body, &HeaderMap::new(), "model").await,
            None
        );
    }

    // -- stream_ollama over a mock SSE backend --------------------------------

    const MOCK_SSE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}]}\n",
        "data: [DONE]\n",
    );

    /// Runs one request through `stream_ollama` against a real HTTP backend
    /// serving `MOCK_SSE`, returning its content type and body. The fold
    /// tests alone would still pass if the branch, its content type, or a
    /// handler's flag forwarding regressed.
    async fn run_stream_ollama(streaming: bool) -> (String, String) {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { ([("content-type", "text/event-stream")], MOCK_SSE) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let resp = stream_ollama(
            streaming,
            Client::new(),
            Target::Local(addr.port()),
            OAIChatRequest {
                messages: vec![OAIMessage::text("user", "hi".to_string())],
                stream: true,
                ..Default::default()
            },
            ActivityGuard::new(&test_state(), "m"),
            |content, thinking, tool_calls, done| OllamaChatChunk {
                model: "m".into(),
                created_at: now_rfc3339(),
                message: OllamaMessage {
                    role: "assistant".into(),
                    content,
                    thinking,
                    tool_calls,
                    ..Default::default()
                },
                done,
                done_reason: done.then(|| "stop".to_string()),
            },
        )
        .await
        .expect("mock backend answers");

        let ct = resp.headers()["content-type"].to_str().unwrap().to_string();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (ct, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn stream_ollama_false_returns_one_json_object() {
        let (content_type, body) = run_stream_ollama(false).await;
        assert_eq!(content_type, "application/json");
        let one: serde_json::Value =
            serde_json::from_str(&body).expect("the whole body is one JSON object");
        assert_eq!(one["message"]["content"], "Hello, world");
        assert_eq!(one["done"], true);
        assert_eq!(one["done_reason"], "stop");
    }

    #[tokio::test]
    async fn stream_ollama_true_returns_ndjson_chunks() {
        let (content_type, body) = run_stream_ollama(true).await;
        assert_eq!(content_type, "application/x-ndjson");
        let chunks: Vec<serde_json::Value> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("each line is its own object"))
            .collect();
        assert!(chunks.len() > 1, "got {} chunks", chunks.len());
        let joined: String = chunks
            .iter()
            .map(|c| c["message"]["content"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(joined, "Hello, world");
        assert_eq!(chunks.last().unwrap()["done"], true);
    }
}
