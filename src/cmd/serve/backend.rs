//! The backends `ensure_model` spawns: llama-server (local or in a
//! container), vLLM and vLLM-Omni, SGLang, `mlx_lm.server`, and
//! `ds4-server`. Spawning,
//! the stderr tail that turns a failed load into an error message,
//! readiness polling, and where the `llama-server` binary itself comes
//! from.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Context};
use reqwest::Client;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::time::{sleep, Duration, Instant};

use super::sched::parse_keep_alive_str;
use super::{canonical_ref, AppState, Engine, ModelProcess, Target};
use crate::modelpack::{resolve_model, ModelPath};

/// `LLMMAN_SAFETENSORS_ENGINE`: which engine serves a
/// [`ModelPath::SafeTensors`] directory, the one format more than one
/// engine can. `Auto` (unset) is the pre-existing rule in
/// [`use_mlx_for_safetensors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SafetensorsEngine {
    Auto,
    /// `vllm`, even where `mlx_lm.server` would otherwise be preferred.
    Vllm,
    /// `sglang` — see [`spawn_sglang_server`].
    Sglang,
}

pub(super) const SAFETENSORS_ENGINE_VAR: &str = "LLMMAN_SAFETENSORS_ENGINE";

/// `vllm`/`sglang` (any case); unset or empty is `Auto`; anything else
/// `None`, for the caller to warn about.
pub(super) fn parse_safetensors_engine(value: Option<&str>) -> Option<SafetensorsEngine> {
    let value = value.map(str::trim).unwrap_or_default();
    if value.is_empty() {
        return Some(SafetensorsEngine::Auto);
    }
    match value.to_ascii_lowercase().as_str() {
        "vllm" => Some(SafetensorsEngine::Vllm),
        "sglang" => Some(SafetensorsEngine::Sglang),
        _ => None,
    }
}

/// [`parse_safetensors_engine`] on the environment; an unrecognized value
/// is reported once and treated as `Auto`.
pub(super) fn safetensors_engine_from_env() -> SafetensorsEngine {
    static WARNED: std::sync::Once = std::sync::Once::new();
    engine_from_env(
        SAFETENSORS_ENGINE_VAR,
        "one of vllm/sglang",
        &WARNED,
        parse_safetensors_engine,
        SafetensorsEngine::Auto,
    )
}

/// `LLMMAN_GGUF_ENGINE`: which engine serves a [`ModelPath::Gguf`] load
/// (see [`gguf_goes_to_ds4`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GgufEngine {
    /// `ds4-server` for a [`DS4_ARCHITECTURES`] model when `ds4-server`
    /// is on `PATH`, llama-server otherwise.
    Auto,
    /// `ds4-server` for every generation GGUF — see [`spawn_ds4_server`].
    Ds4,
    LlamaServer,
}

pub(super) const GGUF_ENGINE_VAR: &str = "LLMMAN_GGUF_ENGINE";

/// `ds4` or `llama-server` (any case); unset or empty is `Auto`; anything
/// else `None`, for the caller to warn about.
pub(super) fn parse_gguf_engine(value: Option<&str>) -> Option<GgufEngine> {
    let value = value.map(str::trim).unwrap_or_default();
    if value.is_empty() {
        return Some(GgufEngine::Auto);
    }
    match value.to_ascii_lowercase().as_str() {
        "ds4" => Some(GgufEngine::Ds4),
        "llama-server" => Some(GgufEngine::LlamaServer),
        _ => None,
    }
}

/// [`parse_gguf_engine`] on the environment; an unrecognized value is
/// reported once and treated as `Auto`.
pub(super) fn gguf_engine_from_env() -> GgufEngine {
    static WARNED: std::sync::Once = std::sync::Once::new();
    engine_from_env(
        GGUF_ENGINE_VAR,
        "one of ds4/llama-server",
        &WARNED,
        parse_gguf_engine,
        GgufEngine::Auto,
    )
}

/// The `general.architecture`s [`GgufEngine::Auto`] sends to ds4-server.
/// Not `deepseek4`: llama.cpp loads its own V4 GGUFs, which ds4 rejects.
pub(super) const DS4_ARCHITECTURES: &[&str] = &["deepseek41"];

pub(super) fn ds4_server_on_path() -> bool {
    crate::find_on_path("ds4-server").is_some()
}

/// Whether `model`, of GGUF architecture `arch`, goes to `ds4-server`
/// under `engine`: only a [`ModelPath::Gguf`], so every other format
/// keeps its own engine and runtime. Embedding and decision models stay
/// on llama-server: ds4 has no `/v1/embeddings` or `/v1/systemone`.
/// `ds4_on_path` is called only for an `Auto` [`DS4_ARCHITECTURES`] model.
pub(super) fn gguf_goes_to_ds4(
    engine: GgufEngine,
    model: &ModelPath,
    arch: Option<&str>,
    embedding: bool,
    decision: bool,
    ds4_on_path: impl FnOnce() -> bool,
) -> bool {
    matches!(model, ModelPath::Gguf(..))
        && ds4_serves_gguf(engine, arch, embedding, decision, ds4_on_path)
}

/// How a [`ModelPath::Gguf`] is served once `ensure_model` has decided
/// `ds4` and `speech` (see `resolve_target`): ds4 first, since it runs
/// from PATH under any runtime; then the speech worker; then llama-server.
/// `container` is whether a container runtime resolved, which the speech
/// and llama-server arms honour and ds4 ignores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GgufArm {
    Ds4,
    Speech { container: bool },
    LlamaServer { container: bool },
}

pub(super) fn gguf_arm(ds4: bool, speech: bool, container: bool) -> GgufArm {
    if ds4 {
        GgufArm::Ds4
    } else if speech {
        GgufArm::Speech { container }
    } else {
        GgufArm::LlamaServer { container }
    }
}

/// [`gguf_goes_to_ds4`] for a model already known to be a GGUF.
pub(super) fn ds4_serves_gguf(
    engine: GgufEngine,
    arch: Option<&str>,
    embedding: bool,
    decision: bool,
    ds4_on_path: impl FnOnce() -> bool,
) -> bool {
    if embedding || decision {
        return false;
    }
    match engine {
        GgufEngine::Ds4 => true,
        GgufEngine::LlamaServer => false,
        GgufEngine::Auto => {
            arch.is_some_and(|arch| DS4_ARCHITECTURES.contains(&arch)) && ds4_on_path()
        }
    }
}

/// `parse` on the environment variable `var`, warning once through
/// `warned` and falling back to `auto` when it returns `None`.
fn engine_from_env<T>(
    var: &str,
    choices: &str,
    warned: &std::sync::Once,
    parse: fn(Option<&str>) -> Option<T>,
    auto: T,
) -> T {
    let raw = std::env::var(var).ok();
    parse(raw.as_deref()).unwrap_or_else(|| {
        warned.call_once(|| {
            eprintln!(
                "[llmman] warning: {var}={:?} is not {choices}; ignoring it",
                raw.unwrap_or_default()
            );
        });
        auto
    })
}

/// Which local engine backs a resolved `ModelPath::SafeTensors`
/// directory when `LLMMAN_SAFETENSORS_ENGINE` is unset: `mlx_lm.server`
/// (see `spawn_mlx_server`) on macOS, `vllm` elsewhere. Every Mac is
/// taken to have Metal; `LLMMAN_SAFETENSORS_ENGINE=vllm` opts out.
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
/// Mac gets real Metal acceleration through it without needing
/// vllm-metal (or vllm) at all.
pub(super) fn use_mlx_for_safetensors() -> bool {
    cfg!(target_os = "macos") && safetensors_engine_from_env() == SafetensorsEngine::Auto
}

pub(super) fn find_free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Shared handle onto the last few lines a spawned inference backend wrote
/// to stdout/stderr — see `spawn_tail_relay`'s own doc comment for why
/// this exists and `wait_for_ready`'s use of it.
pub(super) type OutputTail = Arc<StdMutex<VecDeque<String>>>;

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

pub(super) async fn spawn_llama_server(
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
        embeddings,
        batch_size,
        threads,
        // A local child shares the daemon's cgroup already.
        cpus: _,
        extra_args,
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
    // See supports_context_shift's doc comment.
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
    // See LlamaOptions::embeddings and ::batch_size.
    if embeddings {
        cmd.arg("--embeddings");
    }
    if let Some(n) = batch_size {
        let n = n.to_string();
        cmd.args(["-b", &n, "-ub", &n]);
    }
    cmd.args(extra_args);
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
    // llama.cpp's Android release is a 7 KiB `llama-server` launcher whose
    // real code is in sibling `libllama-server-impl.so`/`libggml*.so`, none
    // of which carry a RUNPATH (the Ubuntu, macOS and Windows builds embed
    // one, or are static). Bionic's linker only searches LD_LIBRARY_PATH
    // and system dirs, so name the binary's own directory — after
    // resolving a symlink, which is how the Android app exposes it on PATH.
    #[cfg(target_os = "android")]
    {
        let lib_dir = std::fs::canonicalize(bin)
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .or_else(|| bin.parent().map(Path::to_path_buf));
        if let Some(lib_dir) = lib_dir {
            let mut value = lib_dir.into_os_string();
            if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
                if !existing.is_empty() {
                    value.push(":");
                    value.push(existing);
                }
            }
            cmd.env("LD_LIBRARY_PATH", value);
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
    let tail = tail_child_output(&mut child);
    Ok((child, tail))
}

/// Takes `child`'s piped stdout/stderr and relays both through
/// [`spawn_tail_relay`], returning the shared tail.
pub(super) fn tail_child_output(child: &mut tokio::process::Child) -> OutputTail {
    let tail: OutputTail = Arc::new(StdMutex::new(VecDeque::with_capacity(TAIL_LINES)));
    if let Some(stdout) = child.stdout.take() {
        spawn_tail_relay(stdout, tail.clone(), false);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_tail_relay(stderr, tail.clone(), true);
    }
    tail
}

/// vLLM should only override its model-derived default for explicit
/// positive user input. This deliberately skips backend_ctx_size's
/// num_parallel scaling because vLLM's limit is per request, not split
/// across llama-server-style request slots.
pub(super) fn vllm_max_model_len(ctx_size: Option<u32>, ctx_size_explicit: bool) -> Option<u32> {
    ctx_size.filter(|n| ctx_size_explicit && *n > 0)
}

/// Extra argv appended to every `vllm serve` (plain and `--omni`, local
/// or in a container): the engine's own knobs, `--dtype bfloat16 --tp 2`.
pub(super) const VLLM_ARGS_VAR: &str = "LLMMAN_VLLM_ARGS";

/// The same for `sglang serve` / `sglang.launch_server`.
pub(super) const SGLANG_ARGS_VAR: &str = "LLMMAN_SGLANG_ARGS";

/// The same for `ds4-server`.
pub(super) const DS4_ARGS_VAR: &str = "LLMMAN_DS4_ARGS";

/// The same for `llama-server`, without `--port`/`--host`
/// ([`check_extra_args_env`]): llama-server deprecates repeating a flag.
pub(super) const LLAMA_ARGS_VAR: &str = "LLMMAN_LLAMA_ARGS";

/// Splits an `LLMMAN_*_ARGS` or `LLMMAN_SHELL` value on whitespace. An
/// argument that starts with `'` or `"` runs to the matching quote, so it
/// can hold a space, and loses the quotes; any other quote, backslash or
/// `#` is literal. `LLMMAN_*_ARGS` is appended after llmman's own flags, so
/// a repeated flag wins on an argparse engine, except `--port` and
/// `--host` (see [`append_extra_args`]).
pub(super) fn split_quoted_args(value: Option<&str>) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut chars = value.unwrap_or_default().chars().enumerate().peekable();
    loop {
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let Some(&(at, first)) = chars.peek() else {
            return Ok(args);
        };
        let mut arg = String::new();
        if first == '\'' || first == '"' {
            chars.next();
            loop {
                match chars.next() {
                    Some((_, c)) if c == first => break,
                    Some((_, c)) => arg.push(c),
                    // The position, not the argument: it may be a secret.
                    None => {
                        return Err(format!(
                            "missing closing quote for the {first} at character {}",
                            at + 1
                        ))
                    }
                }
            }
        }
        while let Some((_, c)) = chars.next_if(|(_, c)| !c.is_whitespace()) {
            arg.push(c);
        }
        args.push(arg);
    }
}

/// [`split_quoted_args`] on the environment variable `var`; empty if it
/// does not split, which [`check_extra_args_env`] refuses at startup.
pub(super) fn extra_args_from_env(var: &str) -> Vec<String> {
    split_quoted_args(std::env::var(var).ok().as_deref()).unwrap_or_default()
}

/// Whether the user set llama-server's `flag`: in `extra` (a split
/// `LLMMAN_LLAMA_ARGS`), or as `env` when llama-server inherits it.
pub(super) fn llama_sets(extra: &[String], flag: &str, env: Option<&str>) -> bool {
    extra.iter().any(|a| {
        a.strip_prefix(flag)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('='))
    }) || env.is_some_and(|env| std::env::var_os(env).is_some())
}

/// llama-server's flags that size each slot's context, with their env vars.
const LLAMA_SLOT_CONTEXT_FLAGS: &[(&str, &str)] = &[
    ("-c", "LLAMA_ARG_CTX_SIZE"),
    ("--ctx-size", "LLAMA_ARG_CTX_SIZE"),
    ("-np", "LLAMA_ARG_N_PARALLEL"),
    ("--parallel", "LLAMA_ARG_N_PARALLEL"),
    ("-kvu", "LLAMA_ARG_KV_UNIFIED"),
    ("--kv-unified", "LLAMA_ARG_KV_UNIFIED"),
    ("-no-kvu", "LLAMA_ARG_KV_UNIFIED"),
    ("--no-kv-unified", "LLAMA_ARG_KV_UNIFIED"),
    ("--kv-unified-per-slot", "LLAMA_ARG_KV_UNIFIED_PER_SLOT"),
];

/// Whether `extra` (a split `LLMMAN_LLAMA_ARGS`, which overrides llmman's
/// flags) or, if `inherits_env`, the environment sets one of
/// [`LLAMA_SLOT_CONTEXT_FLAGS`].
pub(super) fn llama_sets_slot_context(extra: &[String], inherits_env: bool) -> bool {
    LLAMA_SLOT_CONTEXT_FLAGS
        .iter()
        .any(|(flag, env)| llama_sets(extra, flag, inherits_env.then_some(*env)))
}

/// Fails on the first `LLMMAN_*_ARGS` that [`split_quoted_args`] rejects,
/// naming the variable and the quote's position but not the value, or on
/// `--port`/`--host` in `LLMMAN_LLAMA_ARGS`.
pub(super) fn check_extra_args_env() -> anyhow::Result<()> {
    for var in [VLLM_ARGS_VAR, SGLANG_ARGS_VAR, DS4_ARGS_VAR, LLAMA_ARGS_VAR] {
        if let Ok(value) = std::env::var(var) {
            let args = split_quoted_args(Some(&value)).map_err(|e| anyhow!("{var}: {e}"))?;
            if var == LLAMA_ARGS_VAR {
                if let Some(flag) = args.iter().find(|a| *a == "--port" || *a == "--host") {
                    anyhow::bail!("{var}: llmman sets {flag} itself");
                }
            }
        }
    }
    Ok(())
}

/// Appends `extra` (a split `LLMMAN_*_ARGS`), then `--port port --host
/// host`. vllm, sglang and ds4-server keep each flag's last value, so
/// llmman's win however `extra` spells them (argparse takes abbreviations).
fn append_extra_args(args: &mut Vec<String>, extra: Vec<String>, host: &str, port: u16) {
    args.extend(extra);
    args.extend([
        "--port".into(),
        port.to_string(),
        "--host".into(),
        host.into(),
    ]);
}

/// argv after the `vllm` binary, without `--port`/`--host` (see
/// [`vllm_serve_args_from_env`]). Kept separate so context forwarding is
/// testable.
pub(super) fn vllm_serve_args(
    model_dir: &str,
    model_name: &str,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let mut args = vec![
        "serve".into(),
        model_dir.into(),
        // Register the model under the same name used in API requests so
        // {"model": "<ref>"} is accepted by vllm's OpenAI-compatible API.
        "--served-model-name".into(),
        model_name.into(),
    ];
    if let Some(n) = max_model_len {
        args.push("--max-model-len".into());
        args.push(n.to_string());
    }
    args
}

/// The parsers vllm needs to serve a model family to an agent, by the
/// `model_type` in `model_dir`'s `config.json` (a host path). Agents send
/// `tool_choice: "auto"`, which vllm rejects with a 400 without them.
/// They come before `LLMMAN_VLLM_ARGS`, so a flag set there wins.
pub(super) fn vllm_family_args(model_dir: &Path) -> Vec<String> {
    let model_type = std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| config["model_type"].as_str().map(str::to_owned));
    match model_type.as_deref() {
        Some("qwen3_5" | "qwen3_5_moe") => [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
        ]
        .map(String::from)
        .to_vec(),
        _ => Vec::new(),
    }
}

/// [`vllm_serve_args`], then `family` (see [`vllm_family_args`]), then
/// `LLMMAN_VLLM_ARGS`, `--port` and `--host` (see [`append_extra_args`]).
/// Shared with `container::spawn_engine`, which passes its `/models` mount
/// and `0.0.0.0`.
pub(super) fn vllm_serve_args_from_env(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
    max_model_len: Option<u32>,
    family: &[String],
) -> Vec<String> {
    let mut args = vllm_serve_args(model_dir, model_name, max_model_len);
    args.extend(family.iter().cloned());
    append_extra_args(&mut args, extra_args_from_env(VLLM_ARGS_VAR), host, port);
    args
}

/// `<bin> <args>` in its own process group, so [`ModelProcess`]'s Drop can
/// kill the engine's whole worker tree (not just this pid) without
/// killing us. vllm and sglang both fork workers.
fn grouped_command(bin: &Path, args: Vec<String>) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(args).kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd
}

pub(super) async fn spawn_vllm_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
    max_model_len: Option<u32>,
) -> anyhow::Result<tokio::process::Child> {
    let vllm = which_binary("vllm")?;
    let args = vllm_serve_args_from_env(
        model_dir.to_str().context("non-UTF-8 model path")?,
        "127.0.0.1",
        port,
        model_name,
        max_model_len,
        &vllm_family_args(model_dir),
    );
    let mut cmd = grouped_command(&vllm, args);
    crate::debug_log!("spawning {}: {:?}", vllm.display(), cmd);
    cmd.spawn()
        .with_context(|| format!("spawn vllm from {}", vllm.display()))
}

/// Set (`1`/`true`/...) to keep Cosmos3's safety guardrails on under
/// vLLM-Omni; see [`vllm_omni_serve_args`].
const VLLM_OMNI_GUARDRAILS_VAR: &str = "LLMMAN_VLLM_OMNI_GUARDRAILS";

/// argv after `vllm` for a [`ModelPath::Omni`] model: [`vllm_serve_args`]
/// plus `--omni` (the vLLM-Omni plugin picks the pipeline from
/// `model_index.json`), no `--max-model-len` (no context window).
///
/// `--no-guardrails` unless `LLMMAN_VLLM_OMNI_GUARDRAILS` is set: Cosmos3's
/// guardrails need the `cosmos-guardrail` package and a runtime download of
/// the gated `nvidia/Cosmos-1.0-Guardrail`, and vLLM-Omni refuses to start
/// without them — a model served from llmman's store has neither.
/// `--init-timeout` mirrors llmman's load deadline so vLLM-Omni's own
/// (10 minutes) does not give up first.
fn vllm_omni_serve_args(
    model_dir: &str,
    model_name: &str,
    guardrails: bool,
    init_timeout: Duration,
) -> Vec<String> {
    let mut args = vllm_serve_args(model_dir, model_name, None);
    args.push("--omni".into());
    if !guardrails {
        args.push("--no-guardrails".into());
    }
    args.push("--init-timeout".into());
    args.push(init_timeout.as_secs().max(1).to_string());
    args
}

/// [`vllm_omni_serve_args`] from the environment, plus `LLMMAN_VLLM_ARGS`,
/// `--port` and `--host` (see [`append_extra_args`]); an unbounded
/// `LLMMAN_LOAD_TIMEOUT` becomes a day.
pub(super) fn vllm_omni_serve_args_from_env(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
) -> Vec<String> {
    let guardrails = crate::env_flag_set(VLLM_OMNI_GUARDRAILS_VAR);
    let init_timeout = load_timeout_from_env().unwrap_or(Duration::from_secs(24 * 3600));
    let mut args = vllm_omni_serve_args(model_dir, model_name, guardrails, init_timeout);
    append_extra_args(&mut args, extra_args_from_env(VLLM_ARGS_VAR), host, port);
    args
}

/// `vllm serve --omni` from the `vllm` on `PATH`.
pub(super) async fn spawn_vllm_omni_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let vllm = which_binary("vllm")?;
    if !vllm_has_omni_plugin(&vllm).await {
        anyhow::bail!(
            "{} has no vllm-omni plugin, which a Diffusers-layout model needs \
             (`uv pip install vllm-omni` into the same environment, or serve it \
             with --runtime docker to use the vllm/vllm-omni image)",
            vllm.display()
        );
    }
    let args = vllm_omni_serve_args_from_env(
        model_dir.to_str().context("non-UTF-8 model path")?,
        "127.0.0.1",
        port,
        model_name,
    );
    spawn_piped(grouped_command(&vllm, args), &vllm, "vllm --omni")
}

/// Spawns `cmd` with stdio piped, like a llama-server's, so a startup
/// failure's reason reaches `wait_for_ready` through the returned tail.
fn spawn_piped(
    mut cmd: tokio::process::Command,
    bin: &Path,
    what: &str,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    crate::debug_log!("spawning {}: {:?}", bin.display(), cmd);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {what} from {}", bin.display()))?;
    let tail = tail_child_output(&mut child);
    Ok((child, tail))
}

/// The name SGLang registers `model_ref` under. SGLang reads `:` in
/// `--served-model-name` and in a request's `model` as its
/// `model:lora-adapter` separator, so a reference's `:tag` becomes `-tag`;
/// `backend_wire_model` rewrites requests to match.
pub(super) fn sglang_served_model_name(model_ref: &str) -> String {
    model_ref.replace(':', "-")
}

/// argv after SGLang's launcher (`sglang serve` locally, `python3 -m
/// sglang.launch_server` in the `lmsysorg/sglang` image): the same shape
/// as [`vllm_serve_args`] in SGLang's spelling (no `--port`/`--host`), with
/// `--context-length` fed by the [`vllm_max_model_len`] rule.
pub(super) fn sglang_serve_args(
    model_dir: &str,
    model_name: &str,
    context_length: Option<u32>,
) -> Vec<String> {
    let mut args = vec![
        "--model-path".into(),
        model_dir.into(),
        "--served-model-name".into(),
        sglang_served_model_name(model_name),
    ];
    if let Some(n) = context_length {
        args.push("--context-length".into());
        args.push(n.to_string());
    }
    args
}

/// [`sglang_serve_args`] plus `LLMMAN_SGLANG_ARGS`, `--port` and `--host`
/// (see [`append_extra_args`]).
pub(super) fn sglang_serve_args_from_env(
    model_dir: &str,
    host: &str,
    port: u16,
    model_name: &str,
    context_length: Option<u32>,
) -> Vec<String> {
    let mut args = sglang_serve_args(model_dir, model_name, context_length);
    append_extra_args(&mut args, extra_args_from_env(SGLANG_ARGS_VAR), host, port);
    args
}

/// `sglang serve <args>` from the `sglang` console script on `PATH`, in
/// its own process group like vllm (it forks a scheduler and detokenizer).
pub(super) async fn spawn_sglang_server(
    model_dir: &Path,
    port: u16,
    model_name: &str,
    context_length: Option<u32>,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let sglang = which_binary("sglang").map_err(|e| {
        anyhow!(
            "{e} (`uv pip install sglang` puts it there, or drop {SAFETENSORS_ENGINE_VAR}=sglang \
             to serve this model with vllm)"
        )
    })?;
    let mut args = vec!["serve".to_string()];
    args.extend(sglang_serve_args_from_env(
        model_dir.to_str().context("non-UTF-8 model path")?,
        "127.0.0.1",
        port,
        model_name,
        context_length,
    ));
    spawn_piped(grouped_command(&sglang, args), &sglang, "sglang")
}

/// argv after `ds4-server`, without `--port`/`--host` (see
/// [`ds4_server_args_from_env`]). ds4 resolves its `metal/*.metal` kernel
/// sources against the working directory, which `--chdir` sets.
/// `--vision` takes the encoder GGUF llama-server calls `--mmproj`.
/// `ctx` is [`vllm_max_model_len`]'s value; `None` leaves ds4's default.
pub(super) fn ds4_server_args(
    model: &str,
    vision: Option<&str>,
    chdir: &str,
    ctx: Option<u32>,
) -> Vec<String> {
    let mut args = vec!["-m".into(), model.into(), "--chdir".into(), chdir.into()];
    if let Some(vision) = vision {
        args.push("--vision".into());
        args.push(vision.into());
    }
    if let Some(n) = ctx {
        args.push("--ctx".into());
        args.push(n.to_string());
    }
    args
}

/// [`ds4_server_args`] plus `LLMMAN_DS4_ARGS`, `--port` and `--host
/// 127.0.0.1` (see [`append_extra_args`]).
pub(super) fn ds4_server_args_from_env(
    model: &str,
    vision: Option<&str>,
    chdir: &str,
    port: u16,
    ctx: Option<u32>,
) -> Vec<String> {
    let mut args = ds4_server_args(model, vision, chdir, ctx);
    append_extra_args(
        &mut args,
        extra_args_from_env(DS4_ARGS_VAR),
        "127.0.0.1",
        port,
    );
    args
}

/// Spawns the `ds4-server` on `PATH` with `--chdir` set to the directory
/// of the binary it resolves to, so a symlink into a ds4 checkout finds
/// its kernels. The model paths are made absolute, since ds4 opens them
/// after that `chdir`. A single process, so `kill_on_drop` alone stops it.
pub(super) async fn spawn_ds4_server(
    model: &Path,
    vision: Option<&Path>,
    port: u16,
    ctx: Option<u32>,
) -> anyhow::Result<(tokio::process::Child, OutputTail)> {
    let ds4 = which_binary("ds4-server").map_err(|e| {
        anyhow!(
            "{e} (build it from https://github.com/antirez/ds4 with `make` and put ds4-server \
             on PATH, or unset {GGUF_ENGINE_VAR} to serve this model with llama-server)"
        )
    })?;
    let resolved =
        std::fs::canonicalize(&ds4).with_context(|| format!("resolve {}", ds4.display()))?;
    let dir = resolved
        .parent()
        .with_context(|| format!("{} has no parent directory", resolved.display()))?;
    let absolute =
        |p: &Path| std::fs::canonicalize(p).with_context(|| format!("resolve {}", p.display()));
    let model = absolute(model)?;
    let vision = vision.map(absolute).transpose()?;
    let args = ds4_server_args_from_env(
        model.to_str().context("non-UTF-8 model path")?,
        vision
            .as_deref()
            .map(|p| p.to_str().context("non-UTF-8 mmproj path"))
            .transpose()?,
        dir.to_str().context("non-UTF-8 ds4-server path")?,
        port,
        ctx,
    );
    let mut cmd = tokio::process::Command::new(&resolved);
    cmd.args(args).kill_on_drop(true);
    spawn_piped(cmd, &resolved, "ds4-server")
}

/// `GET path` on the loopback backend at `port`, as JSON.
pub(super) async fn backend_json(
    client: &Client,
    port: u16,
    path: &str,
    timeout: Duration,
) -> reqwest::Result<serde_json::Value> {
    client
        .get(Target::Local(port).url(path))
        .timeout(timeout)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
}

/// The context window a running ds4-server reports in `GET /v1/models`:
/// its `--ctx`, or its default.
pub(super) async fn ds4_context_length(client: &Client, port: u16) -> Option<u32> {
    let body = backend_json(client, port, "/v1/models", Duration::from_millis(500))
        .await
        .ok()?;
    ds4_models_context_length(&body)
}

/// `data[0].context_length` of a ds4 `/v1/models` body, if positive.
/// Every entry is an alias of the one loaded model, so they share it.
fn ds4_models_context_length(body: &serde_json::Value) -> Option<u32> {
    let n = body.get("data")?.get(0)?.get("context_length")?.as_u64()?;
    u32::try_from(n).ok().filter(|n| *n > 0)
}

/// Longest [`vllm_has_omni_plugin`] waits before assuming yes.
const OMNI_PLUGIN_PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether the Python behind the `vllm` console script can `import
/// vllm_omni` — a clear error up front instead of vLLM's "unrecognized
/// arguments: --omni" after importing torch. When the launcher is not a
/// `#!/path/to/python` script, or the probe fails or stalls, the answer
/// is yes and vLLM itself reports.
async fn vllm_has_omni_plugin(vllm: &Path) -> bool {
    let Some(python) = console_script_interpreter(vllm) else {
        return true;
    };
    let probe = tokio::process::Command::new(&python)
        .args(["-c", "import vllm_omni"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .status();
    match tokio::time::timeout(OMNI_PLUGIN_PROBE_TIMEOUT, probe).await {
        Ok(Ok(status)) => status.success(),
        Ok(Err(_)) | Err(_) => true,
    }
}

/// The interpreter of a `#!/path/to/python` script (not `#!/usr/bin/env`).
fn console_script_interpreter(script: &Path) -> Option<PathBuf> {
    let mut head = [0u8; 512];
    let n = std::fs::File::open(script)
        .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
        .ok()?;
    let first = std::str::from_utf8(&head[..n]).ok()?.lines().next()?;
    let interp = first.strip_prefix("#!")?.split_whitespace().next()?;
    (!interp.ends_with("/env") && interp.contains("python")).then(|| PathBuf::from(interp))
}

/// Spawns `mlx_lm.server` (<https://github.com/ml-explore/mlx-lm>) —
/// Apple Silicon's own Metal-accelerated alternative to `vllm` for a
/// [`ModelPath::SafeTensors`] directory, picked instead of it by
/// [`use_mlx_for_safetensors`]. The binary is chosen the way `--runtime`
/// chooses `llama-server` (see [`mlx_source`]); normally installed at
/// daemon startup, here only if that failed.
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
/// [`backend_wire_model`](super::backend_wire_model) — which goes through `ModelProvider.load`'s
/// own `try`/`except` in the request-handling path instead, and so does
/// report a real error back to that request on a bad model directory.
pub(super) async fn spawn_mlx_server(
    state: &AppState,
    port: u16,
) -> anyhow::Result<tokio::process::Child> {
    let mlx =
        match mlx_source(state.0.runtime.requested()) {
            None => which_binary("mlx_lm.server").context("--runtime path")?,
            Some(fallback) => tokio::task::spawn_blocking(move || {
                crate::mlx_release::ensure_mlx_server(fallback)
            })
            .await
            .context("ensure mlx_lm.server task panicked")?
            .map_err(|e| {
                anyhow!("{e:#} (set {SAFETENSORS_ENGINE_VAR}=vllm to serve this model with vllm)")
            })?,
        };
    let mut cmd = tokio::process::Command::new(&mlx);
    cmd.args(["--port", &port.to_string(), "--host", "127.0.0.1"]);
    cmd.kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn mlx_lm.server from {}", mlx.display()))
}

/// How `runtime` sources `mlx_lm.server`, mirroring `llama-server`:
/// `None` is `PATH` only (`path`); otherwise llmman's own install, with
/// or without a `PATH` fallback (`auto` and `bin`). Container runtimes
/// never reach mlx (Linux only) and get `bin`'s answer.
pub(super) fn mlx_source(runtime: super::runtime::Runtime) -> Option<crate::mlx_release::Fallback> {
    use super::runtime::Runtime;
    use crate::mlx_release::Fallback;
    match runtime {
        Runtime::Path => None,
        Runtime::Auto => Some(Fallback::Path),
        Runtime::Bin | Runtime::Docker | Runtime::Podman => Some(Fallback::None),
    }
}

fn which_binary(name: &str) -> anyhow::Result<PathBuf> {
    crate::find_on_path(name).ok_or_else(|| anyhow::anyhow!("{name} not found on PATH"))
}

/// [`wait_for_ready`]'s default deadline — four times Ollama's own 5m
/// default, since vllm can take many minutes to load a large model: a
/// cold start of Qwen 3.8 27B (NVFP4) on a DGX Spark spends about ten
/// minutes in `torch.compile` and FlashInfer's JIT before it serves, which
/// the previous 10 minutes did not outlast. Overridable via
/// `LLMMAN_LOAD_TIMEOUT`.
const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(1200);

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

/// Poll interval between readiness checks in [`wait_for_ready`].
pub(super) const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// How long one readiness request in [`wait_for_ready`] may take. Longer
/// than [`POLL_INTERVAL`]: SGLang's `/health` runs a one-token generation,
/// which on CPU takes about as long as the interval — bounded by it, every
/// poll timed out and the load never became ready (seen in CI).
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Polls `process`'s readiness endpoint ([`Engine::ready_path`]) until
/// ready, bailing out early if `process` itself exits first (so a
/// crash-on-startup doesn't hang the caller for the whole deadline).
/// `stderr_tail`, when given (every piped child — see `ensure_model`),
/// includes the crash reason in the error.
pub(super) async fn wait_for_ready(
    client: &Client,
    port: u16,
    process: &mut ModelProcess,
    stderr_tail: Option<&OutputTail>,
) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}{}", process.engine().ready_path());
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
        // Bound the request by HEALTH_TIMEOUT, not the full remaining
        // deadline — otherwise a probe that connects but then stalls
        // could occupy up to the whole deadline (or forever, if unset)
        // without rechecking process liveness or the deadline.
        let bound = remaining.map_or(HEALTH_TIMEOUT, |r| r.min(HEALTH_TIMEOUT));
        let attempt_start = Instant::now();
        if let Ok(resp) = client.get(&url).timeout(bound).send().await {
            // llama-server/vllm: 200 once loaded. ds4-server only
            // listens once loaded. mlx_lm.server: 200 as
            // soon as its listener is up, not once a model is loaded
            // (see spawn_mlx_server) — an accepted, documented gap for
            // that one engine only.
            if resp.status().is_success() {
                return Ok(());
            }
        }
        // Only the unused remainder of one POLL_INTERVAL — not another
        // full one on top of whatever the attempt above already took —
        // so a consistently-stalling probe still gets rechecked every
        // POLL_INTERVAL, not every 2x that. Still never past the
        // overall deadline either.
        let sleep_for = POLL_INTERVAL.saturating_sub(attempt_start.elapsed());
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        sleep(remaining.map_or(sleep_for, |r| sleep_for.min(r))).await;
    }
}

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
pub const LLAMA_CPP_ENV_PASSTHROUGH_VARS: &[&str] = &[
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_FIT_TARGET",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_N_GPU_LAYERS",
];

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
pub(super) async fn would_use_mlx(state: &AppState, model_ref: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::PoisonError;

    /// Held by every test that sets or reads an `LLMMAN_*_ARGS` variable,
    /// since the environment is process-wide.
    static EXTRA_ARGS_ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// Sets `var` for its lifetime, restoring the previous value on drop.
    struct EnvVar {
        var: &'static str,
        previous: Option<String>,
    }

    impl EnvVar {
        fn set(var: &'static str, value: &str) -> Self {
            let previous = std::env::var(var).ok();
            std::env::set_var(var, value);
            Self { var, previous }
        }
    }

    impl Drop for EnvVar {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.var, value),
                None => std::env::remove_var(self.var),
            }
        }
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
    fn the_default_load_timeout_is_twenty_minutes() {
        assert_eq!(DEFAULT_LOAD_TIMEOUT, Duration::from_secs(20 * 60));
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

    #[test]
    fn vllm_max_model_len_uses_only_explicit_ctx_size() {
        assert_eq!(vllm_max_model_len(Some(4096), true), Some(4096));
        assert_eq!(vllm_max_model_len(Some(0), true), None);
        assert_eq!(vllm_max_model_len(Some(65536), false), None);
        assert_eq!(vllm_max_model_len(None, true), None);
        assert_eq!(vllm_max_model_len(None, false), None);
    }

    #[test]
    fn vllm_serve_args_handles_max_model_len() {
        for (max_model_len, expected) in [
            (Some(256), Some("256")),
            (Some(1024), Some("1024")),
            (Some(4096), Some("4096")),
            (None, None),
        ] {
            let args = vllm_serve_args("/models/qwen", "qwen3.5:0.8b", max_model_len);
            let pos = args.iter().position(|arg| arg == "--max-model-len");

            match expected {
                Some(value) => {
                    let i = pos.expect("--max-model-len should be present");
                    assert_eq!(args.get(i + 1).map(String::as_str), Some(value));
                }
                None => assert!(pos.is_none()),
            }
        }
    }

    #[test]
    fn vllm_gets_the_parsers_qwen3_5_needs_before_the_users_args() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let family_of = |config: &str| {
            std::fs::write(dir.path().join("config.json"), config).unwrap();
            vllm_family_args(dir.path())
        };
        let qwen = family_of(r#"{"model_type": "qwen3_5"}"#);
        assert_eq!(
            qwen,
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3"
            ]
        );
        assert_eq!(family_of(r#"{"model_type": "qwen3_5_moe"}"#), qwen);
        // Other families, and a model with no readable config, are left alone.
        assert!(family_of(r#"{"model_type": "llama"}"#).is_empty());
        assert!(family_of("not json").is_empty());
        std::fs::remove_file(dir.path().join("config.json")).unwrap();
        assert!(vllm_family_args(dir.path()).is_empty());

        // After the model and before whatever LLMMAN_VLLM_ARGS holds, which
        // vllm lets win by keeping each flag's last value.
        let args = vllm_serve_args_from_env("/m", "0.0.0.0", 8000, "m", None, &qwen);
        assert_eq!(&args[4..9], qwen.as_slice());
        assert_eq!(
            &args[args.len() - 4..],
            ["--port", "8000", "--host", "0.0.0.0"]
        );
    }

    /// Whatever `LLMMAN_*_ARGS` holds, every builder ends with llmman's
    /// `--port` and `--host`.
    #[test]
    fn every_builder_ends_with_llmmans_port_and_host() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let tail = |args: Vec<String>| args[args.len() - 4..].to_vec();
        fn expect(host: &str) -> Vec<&str> {
            vec!["--port", "41234", "--host", host]
        }
        for host in ["127.0.0.1", "0.0.0.0"] {
            assert_eq!(
                tail(vllm_serve_args_from_env(
                    "/m",
                    host,
                    41234,
                    "m",
                    Some(4096),
                    &[]
                )),
                expect(host)
            );
            assert_eq!(
                tail(vllm_omni_serve_args_from_env("/m", host, 41234, "m")),
                expect(host)
            );
            assert_eq!(
                tail(sglang_serve_args_from_env(
                    "/m",
                    host,
                    41234,
                    "m",
                    Some(4096)
                )),
                expect(host)
            );
        }
        assert_eq!(
            tail(ds4_server_args_from_env(
                "/m.gguf", None, "/opt/ds4", 41234, None
            )),
            expect("127.0.0.1")
        );
    }

    /// Local and container argv differ only in the model dir and the final
    /// `--host`.
    #[test]
    fn local_and_container_argv_differ_only_in_dir_and_host() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        fn without_dir_and_host(mut args: Vec<String>) -> Vec<String> {
            args.remove(1);
            args.pop();
            args
        }
        let pair = |build: &dyn Fn(&str, &str) -> Vec<String>| {
            let local = build("/cache/abc/model", "127.0.0.1");
            let container = build("/models", "0.0.0.0");
            assert_eq!(
                (local[1].as_str(), container[1].as_str()),
                ("/cache/abc/model", "/models")
            );
            assert_eq!(without_dir_and_host(local), without_dir_and_host(container));
        };
        pair(&|dir, host| vllm_serve_args_from_env(dir, host, 8000, "m", Some(4096), &[]));
        pair(&|dir, host| vllm_omni_serve_args_from_env(dir, host, 8000, "m"));
        pair(&|dir, host| sglang_serve_args_from_env(dir, host, 30000, "m", Some(4096)));
    }

    /// Each builder puts its `LLMMAN_*_ARGS` before llmman's `--port` and
    /// `--host`, so an extra-args `--port` is overridden.
    #[test]
    fn every_builder_puts_its_extra_args_before_llmmans_port_and_host() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let _vars = [VLLM_ARGS_VAR, SGLANG_ARGS_VAR, DS4_ARGS_VAR]
            .map(|var| EnvVar::set(var, "--port 1 --tp 2"));
        let tail = |args: Vec<String>| args[args.len() - 8..].to_vec();
        let expected = [
            "--port",
            "1",
            "--tp",
            "2",
            "--port",
            "41234",
            "--host",
            "127.0.0.1",
        ];
        assert_eq!(
            tail(vllm_serve_args_from_env(
                "/m",
                "127.0.0.1",
                41234,
                "m",
                None,
                &[]
            )),
            expected
        );
        assert_eq!(
            tail(vllm_omni_serve_args_from_env("/m", "127.0.0.1", 41234, "m")),
            expected
        );
        assert_eq!(
            tail(sglang_serve_args_from_env(
                "/m",
                "127.0.0.1",
                41234,
                "m",
                None
            )),
            expected
        );
        assert_eq!(
            tail(ds4_server_args_from_env(
                "/m.gguf", None, "/opt/ds4", 41234, None
            )),
            expected
        );
    }

    #[test]
    fn llmmans_port_and_host_follow_the_extra_args() {
        let mut args = Vec::new();
        append_extra_args(
            &mut args,
            split_quoted_args(Some("--por 1 --hos 127.0.0.1 --tp 2")).unwrap(),
            "0.0.0.0",
            41234,
        );
        assert_eq!(
            args,
            [
                "--por",
                "1",
                "--hos",
                "127.0.0.1",
                "--tp",
                "2",
                "--port",
                "41234",
                "--host",
                "0.0.0.0"
            ]
        );
    }

    #[test]
    fn vllm_omni_serve_args_add_omni_and_disable_guardrails_by_default() {
        let args = vllm_omni_serve_args(
            "/models",
            "nvidia/Cosmos3-Edge",
            false,
            Duration::from_secs(600),
        );
        // vllm-omni's own recipe: `vllm serve <model> --omni`
        assert_eq!(args[0], "serve");
        assert_eq!(args[1], "/models");
        assert!(args.contains(&"--omni".to_string()));
        assert!(args.contains(&"--no-guardrails".to_string()));
        let i = args.iter().position(|a| a == "--init-timeout").unwrap();
        assert_eq!(args[i + 1], "600");
        let i = args
            .iter()
            .position(|a| a == "--served-model-name")
            .unwrap();
        assert_eq!(args[i + 1], "nvidia/Cosmos3-Edge");
        // a diffusion pipeline has no context window
        assert!(!args.contains(&"--max-model-len".to_string()));
    }

    #[test]
    fn vllm_omni_serve_args_keep_guardrails_when_asked() {
        let args = vllm_omni_serve_args("/m", "m", true, Duration::from_secs(1));
        assert!(args.contains(&"--omni".to_string()));
        assert!(!args.contains(&"--no-guardrails".to_string()));
    }

    #[test]
    fn parse_safetensors_engine_accepts_vllm_sglang_or_nothing() {
        assert_eq!(
            parse_safetensors_engine(None),
            Some(SafetensorsEngine::Auto)
        );
        assert_eq!(
            parse_safetensors_engine(Some("")),
            Some(SafetensorsEngine::Auto)
        );
        assert_eq!(
            parse_safetensors_engine(Some("  ")),
            Some(SafetensorsEngine::Auto)
        );
        assert_eq!(
            parse_safetensors_engine(Some("vllm")),
            Some(SafetensorsEngine::Vllm)
        );
        assert_eq!(
            parse_safetensors_engine(Some("SGLang")),
            Some(SafetensorsEngine::Sglang)
        );
        assert_eq!(
            parse_safetensors_engine(Some(" sglang ")),
            Some(SafetensorsEngine::Sglang)
        );
        // Unknown names are reported by the caller, not silently mapped.
        assert_eq!(parse_safetensors_engine(Some("mlx")), None);
        assert_eq!(parse_safetensors_engine(Some("tgi")), None);
    }

    #[test]
    fn parse_gguf_engine_accepts_ds4_llama_server_or_nothing() {
        assert_eq!(parse_gguf_engine(None), Some(GgufEngine::Auto));
        assert_eq!(parse_gguf_engine(Some("")), Some(GgufEngine::Auto));
        assert_eq!(parse_gguf_engine(Some("  ")), Some(GgufEngine::Auto));
        assert_eq!(parse_gguf_engine(Some("ds4")), Some(GgufEngine::Ds4));
        assert_eq!(parse_gguf_engine(Some(" DS4 ")), Some(GgufEngine::Ds4));
        assert_eq!(
            parse_gguf_engine(Some("Llama-Server")),
            Some(GgufEngine::LlamaServer)
        );
        // Unknown names are reported by the caller, not silently mapped.
        assert_eq!(parse_gguf_engine(Some("llama")), None);
        assert_eq!(parse_gguf_engine(Some("ds4-server")), None);
    }

    /// A flag counts `=`-joined or not; one that only starts like it does not.
    #[test]
    fn llama_sets_slot_context_names_the_flags_that_size_a_slot() {
        let sets = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            llama_sets_slot_context(&args, false)
        };
        for args in [
            &["-c", "8192"][..],
            &["--ctx-size=8192"],
            &["-np", "2"],
            &["--parallel", "2"],
            &["-kvu"],
            &["--no-kv-unified"],
            &["--kv-unified-per-slot", "8192"],
            &["--jinja", "--ctx-size", "8192"],
        ] {
            assert!(sets(args), "{args:?}");
        }
        for args in [
            &[][..],
            &["--jinja"],
            &["--cache-reuse", "64"],
            &["-cb"],
            &["--ctx-size-draft", "8192"],
        ] {
            assert!(!sets(args), "{args:?}");
        }
    }

    #[test]
    fn only_generation_ggufs_go_to_ds4() {
        let gguf = ModelPath::Gguf(PathBuf::from("/m/ds4.gguf"), None);
        let goes = |engine, model: &ModelPath, embedding, decision| {
            gguf_goes_to_ds4(
                engine,
                model,
                Some("deepseek41"),
                embedding,
                decision,
                || true,
            )
        };
        for engine in [GgufEngine::Ds4, GgufEngine::Auto] {
            assert!(goes(engine, &gguf, false, false));
            assert!(!goes(engine, &gguf, true, false));
            assert!(!goes(engine, &gguf, false, true));
            // Other formats keep their own engine and container runtime.
            for other in [
                ModelPath::SafeTensors(PathBuf::from("/m/st")),
                ModelPath::Omni(PathBuf::from("/m/omni")),
            ] {
                assert!(!goes(engine, &other, false, false));
            }
        }
        assert!(!goes(GgufEngine::LlamaServer, &gguf, false, false));
    }

    #[test]
    fn auto_sends_only_deepseek_v41_ggufs_to_a_ds4_server_on_path() {
        let gguf = ModelPath::Gguf(PathBuf::from("/m/model.gguf"), None);
        let auto = |arch, on_path| {
            gguf_goes_to_ds4(GgufEngine::Auto, &gguf, arch, false, false, || on_path)
        };
        assert!(auto(Some("deepseek41"), true));
        assert!(!auto(Some("deepseek41"), false));
        for arch in [
            Some("deepseek4"),
            Some("deepseek2"),
            Some("glm-dsa"),
            Some("llama"),
            None,
        ] {
            assert!(!auto(arch, true), "{arch:?}");
        }
        // `ds4` sends any architecture; neither call checks PATH.
        assert!(gguf_goes_to_ds4(
            GgufEngine::Ds4,
            &gguf,
            Some("llama"),
            false,
            false,
            || unreachable!()
        ));
        assert!(!gguf_goes_to_ds4(
            GgufEngine::Auto,
            &gguf,
            Some("llama"),
            false,
            false,
            || unreachable!()
        ));
    }

    #[test]
    fn ds4_server_args_load_the_gguf_from_the_ds4_checkout() {
        let args = ds4_server_args("/m/ds4.gguf", None, "/opt/ds4", None);
        assert_eq!(args, ["-m", "/m/ds4.gguf", "--chdir", "/opt/ds4"]);
    }

    #[test]
    fn ds4_server_args_pass_the_mmproj_as_vision_and_an_explicit_ctx() {
        let args = ds4_server_args(
            "/m/ds4.gguf",
            Some("/m/encoder.gguf"),
            "/opt/ds4",
            Some(65_536),
        );
        assert_eq!(
            &args[4..],
            ["--vision", "/m/encoder.gguf", "--ctx", "65536"]
        );
    }

    #[test]
    fn ds4_models_context_length_reads_the_first_alias() {
        let body = serde_json::json!({"object": "list", "data": [
            {"id": "deepseek-v4-flash", "context_length": 32768},
            {"id": "deepseek-v4-pro", "context_length": 32768},
        ]});
        assert_eq!(ds4_models_context_length(&body), Some(32_768));
        assert_eq!(
            ds4_models_context_length(&serde_json::json!({"data": []})),
            None
        );
        assert_eq!(
            ds4_models_context_length(&serde_json::json!({"data": [{"context_length": 0}]})),
            None
        );
    }

    #[test]
    fn split_quoted_args_splits_on_whitespace_and_leading_quotes() {
        fn split(value: Option<&str>) -> Vec<String> {
            split_quoted_args(value).unwrap()
        }
        assert_eq!(split(None), Vec::<String>::new());
        assert_eq!(split(Some("")), Vec::<String>::new());
        assert_eq!(split(Some("   \t ")), Vec::<String>::new());
        assert_eq!(
            split(Some("  --dtype bfloat16\t--enforce-eager\n--tp 2 ")),
            ["--dtype", "bfloat16", "--enforce-eager", "--tp", "2"]
        );
        for quoted in ["'a b'", "\"a b\""] {
            assert_eq!(
                split(Some(&format!("--chat-template {quoted}"))),
                ["--chat-template", "a b"]
            );
        }
        assert_eq!(
            split(Some(r#"--hf-overrides '{"a": 1}'"#)),
            ["--hf-overrides", r#"{"a": 1}"#]
        );
        // Anything else passes as it did before quoting.
        assert_eq!(
            split(Some(
                r#"--speculative-config {"method":"ngram"} C:\models #n --x="a b""#
            )),
            [
                "--speculative-config",
                r#"{"method":"ngram"}"#,
                r"C:\models",
                "#n",
                "--x=\"a",
                "b\""
            ]
        );
        assert_eq!(
            split_quoted_args(Some("--api-key 'sk-secret")),
            Err("missing closing quote for the ' at character 11".into())
        );
    }

    #[test]
    fn llama_args_may_not_set_the_port_or_host() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for (value, refused) in [
            ("--slot-prompt-similarity 0.5", None),
            ("--threads 2 --host 0.0.0.0", Some("--host")),
            ("--port 1", Some("--port")),
        ] {
            let _var = EnvVar::set(LLAMA_ARGS_VAR, value);
            assert_eq!(
                check_extra_args_env().map_err(|e| e.to_string()),
                refused.map_or(Ok(()), |flag| Err(format!(
                    "{LLAMA_ARGS_VAR}: llmman sets {flag} itself"
                ))),
                "{value}"
            );
        }
    }

    #[test]
    fn a_llama_flag_is_set_in_llama_args_or_the_environment() {
        let _env = EXTRA_ARGS_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        const ENV: &str = "LLMMAN_TEST_LLAMA_SETS";
        let sets = |value: &str| {
            let args = split_quoted_args(Some(value)).unwrap();
            llama_sets(&args, "--reasoning-budget", Some(ENV))
        };
        assert!(sets("--reasoning-budget 64"));
        assert!(sets("--threads 2 --reasoning-budget=64"));
        assert!(!sets("--reasoning-budget-message 'done' --threads 2"));
        assert!(!sets(""));
        let _var = EnvVar::set(ENV, "64");
        assert!(sets(""));
        // A llama-server that does not inherit the environment.
        assert!(!llama_sets(&[], "--reasoning-budget", None));
    }

    #[test]
    fn sglang_serve_args_speak_sglangs_flag_names() {
        let args = sglang_serve_args("/models", "qwen3.5:0.8b", Some(4096));
        // SGLang's own recipe: `--model-path <dir>`, no
        // positional model and no `serve` subcommand here (the launcher
        // prefix differs between a local `sglang serve` and the image's
        // `python3 -m sglang.launch_server`).
        assert_eq!(&args[..2], &["--model-path", "/models"]);
        assert!(!args.contains(&"serve".to_string()));
        let value = |flag: &str| {
            let i = args.iter().position(|a| a == flag).unwrap();
            args[i + 1].clone()
        };
        // `:` is SGLang's LoRA-adapter separator (see sglang_served_model_name).
        assert_eq!(value("--served-model-name"), "qwen3.5-0.8b");
        assert_eq!(value("--context-length"), "4096");
        assert!(!args.contains(&"--max-model-len".to_string()));
    }

    #[test]
    fn sglang_served_model_name_drops_the_colon_sglang_reserves() {
        assert_eq!(
            sglang_served_model_name("docker.io/ai/qwen3.5:0.8b-safetensors"),
            "docker.io/ai/qwen3.5-0.8b-safetensors"
        );
        assert_eq!(
            sglang_served_model_name("hf.co/HuggingFaceTB/SmolLM2-135M-Instruct:latest"),
            "hf.co/HuggingFaceTB/SmolLM2-135M-Instruct-latest"
        );
    }

    #[test]
    fn sglang_serve_args_leave_context_length_to_the_model_when_unset() {
        let args = sglang_serve_args("/models", "m", None);
        assert!(!args.contains(&"--context-length".to_string()));
    }

    #[test]
    fn console_script_interpreter_reads_a_python_shebang_only() {
        let dir = std::env::temp_dir().join(format!("llmman-shebang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("vllm");
        std::fs::write(
            &script,
            "#!/opt/venv/bin/python3\n# -*- coding: utf-8 -*-\n",
        )
        .unwrap();
        assert_eq!(
            console_script_interpreter(&script),
            Some(PathBuf::from("/opt/venv/bin/python3"))
        );
        std::fs::write(&script, "#!/usr/bin/env python3\nimport sys\n").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        std::fs::write(&script, "#!/bin/sh\nexec vllm \"$@\"\n").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        std::fs::write(&script, b"\x7fELF\x02\x01\x01").unwrap();
        assert_eq!(console_script_interpreter(&script), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ds4 wins, then the speech worker, then llama-server; the two
    /// latter follow the runtime, ds4 runs from PATH either way.
    #[test]
    fn gguf_arm_picks_ds4_then_speech_then_llama_server_by_runtime() {
        assert_eq!(gguf_arm(true, false, true), GgufArm::Ds4);
        assert_eq!(gguf_arm(true, true, false), GgufArm::Ds4);
        assert_eq!(
            gguf_arm(false, true, true),
            GgufArm::Speech { container: true }
        );
        assert_eq!(
            gguf_arm(false, true, false),
            GgufArm::Speech { container: false }
        );
        assert_eq!(
            gguf_arm(false, false, true),
            GgufArm::LlamaServer { container: true }
        );
        assert_eq!(
            gguf_arm(false, false, false),
            GgufArm::LlamaServer { container: false }
        );
    }
}
