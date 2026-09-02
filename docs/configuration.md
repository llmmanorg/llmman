# Configuration

## Store location

Default locations:

| OS | Path |
|----|------|
| Linux, macOS | `~/.local/share/llmman/store` |
| Windows | `%LOCALAPPDATA%\llmman\store` |

Set `LLMMAN_MODELS` to change this (matching Ollama's `OLLAMA_MODELS`).
Commands that read or write the local store directly (`list`, `rm`,
`build`, `serve`) all honor it. Commands that go through the background
daemon instead (`pull`, `push`, `run`, `launch`, `ps`) always use whichever
store the daemon was started with; set `LLMMAN_MODELS` before
`llmman serve` to change it for all of them. `transfer`, `login`, and
`logout` never touch a local store at all.

The store uses [OCI Image Layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), readable by `docker` and `podman`.

## Environment variables

Daemon-wide settings, set before `llmman serve` starts. llmman is a very
different program underneath (no per-GPU memory estimator, no embedded
inference engine, no cloud/desktop-app features), so an equivalent
setting may not behave identically.

| Variable | Effect |
|----------|--------|
| `LLMMAN_DEBUG` | Enables verbose diagnostic logging (a spawned backend's full command line, per-GPU probe detail, etc). Accepts `1`/`true`/`yes`/`on`, or any other non-zero integer. |
| `LLMMAN_HOST` | `[host][:port]` `llmman serve` binds to. Every `llmman` client in the same environment connects to it too, rewriting a wildcard host to loopback first. Defaults to `127.0.0.1:17434`. |
| `LLMMAN_CONTEXT_LENGTH` | Context size (`--ctx-size`) for every model this daemon loads. Defaults to a VRAM-tiered value when unset. |
| `LLMMAN_KEEP_ALIVE` | The daemon-wide default `keep_alive` (how long an idle, unused model stays loaded before being unloaded). Defaults to 5 minutes. Overridden per-request by `/api/chat`/`/api/generate`'s own `keep_alive` field. |
| `LLMMAN_MAX_LOADED_MODELS` | Caps how many models this daemon keeps loaded at once, as one flat daemon-wide total (llmman has no per-model memory estimate to size an automatic per-GPU figure against). Once at the cap, the least-recently-used idle model is evicted to make room; if every loaded model is busy, the request gets a `503` instead. Defaults to `0` (unbounded, today's behavior, unchanged). |
| `LLMMAN_MAX_QUEUE` | Caps how many requests `llmman serve` admits into scheduling at once; anything beyond that gets an immediate `503` (`server busy, please try again.  maximum pending requests exceeded`, two spaces included). Defaults to `512`. |
| `LLMMAN_MAX_TRANSFER_STREAMS` | Maximum number of a HuggingFace safetensors repo's files downloaded concurrently during `pull`. Has no effect on GGUF transfers, and is not read by `transfer`'s own `docker`-feature registry-push path, which streams files sequentially. Defaults to `4`. |
| `LLMMAN_METRICS` | Serves the Prometheus scrape endpoint at `/metrics`. Accepts `1`/`true`/`yes`/`on`. Off by default: the router has no authentication, so an upgrade should not start publishing this daemon's version, route mix, model names and model churn to whoever can reach the port. Unset, the route is absent and answers `404`. |
| `LLMMAN_MODELS` | Local store directory, overriding the default above. `pull`/`push`/`run`/etc. go through the daemon and always use whichever store it was started with. |
| `LLMMAN_NUM_PARALLEL` | Number of parallel request slots (`--parallel`) for GGUF models (llama-server only; no vllm/mlx equivalent). `--ctx-size` is scaled up by this value first, so each slot still gets the full configured/default context rather than an even split of it; ignored (with a warning) for a load with no explicit context size to scale. Unset leaves llama-server's own default of 1 untouched. |
| `LLMMAN_ORIGINS` | A comma-separated list of extra allowed CORS origins for the HTTP API. A trailing `:*` on an entry matches any port on that scheme+host. Always includes every scheme/port on `localhost`/`127.0.0.1`/`0.0.0.0`/`[::1]` regardless of this variable. |
| `LLMMAN_SCHED_SPREAD` | Truthy forwards `--split-mode layer` (spread a model across every GPU, already llama-server's own default); falsey forwards `--split-mode none` (restrict to one GPU). |
| `LLMMAN_FLASH_ATTENTION` | Flash Attention mode (`--flash-attn`): `on`, `off`, or `auto` (llama-server's own default). Also accepts `1`/`0`/`true`/`false`. |
| `LLMMAN_KV_CACHE_TYPE` | KV-cache quantization (`--cache-type-k`/`--cache-type-v`), e.g. `f16` (default), `q8_0`, `q4_0`. Trades output quality for memory at long context lengths. |
| `LLMMAN_LLM_LIBRARY` | Forces which GPU backend `llmman serve`/`run` picks (`cpu`, `cuda`/`cuda12`, `cuda13`, `rocm`, `vulkan`, or macOS-only `metal`), bypassing autodetection. Has no effect when a `llama-server` binary is already on `PATH` (its own backend is fixed), or on macOS's local-binary download (one asset per architecture, no separate choice to make). |
| `LLMMAN_GPU_OVERHEAD` | Bytes of VRAM to hold back from the VRAM-tiered `LLMMAN_CONTEXT_LENGTH` default, leaving headroom for whatever else shares the device. Applied as one combined-total subtraction rather than per-GPU (llmman only ever probes one combined VRAM total). |
| `LLMMAN_IGPU_ENABLE` | Counts integrated GPUs (Vulkan only) when probing for an accelerator. Defaults to disabled, since an integrated GPU is usually a worse choice than the discrete/CPU fallback it would otherwise be skipped in favor of. |
| `LLMMAN_LOAD_TIMEOUT` | How long to allow a model load to stall before giving up. Zero or negative means wait forever. Defaults to 10 minutes (`vllm` can take several minutes to load a large safetensors model). |
| `LLMMAN_TMPDIR` | Staging directory for `llama-server` release downloads, overriding the default `tmp` subdirectory of the install root. |
| `LLMMAN_NOPRUNE` | When set (to anything other than `0`/`false`/`no`/`off`), skips the garbage-collection sweep that `llmman rm` and `llmman serve` startup otherwise run to delete blobs and extracted-cache entries no longer referenced by any local model. Note this is broader than skipping the daemon-startup catch-all: it also stops `llmman rm` itself from ever freeing disk space, so a removed model's (possibly multi-GB) weights stay on disk until a later sweep runs without this set. Useful for a shared/read-mostly store, or scripts that `rm` in a loop and prune once at the end. |
| `LLAMA_ARG_FIT` / `LLAMA_ARG_FIT_TARGET` | llama.cpp's own env-configurable `--fit`/`--fit-target` options. Not something llmman parses itself, just forwarded through to every `llama-server` (local or `--ociman` container) it spawns, same as `CUDA_VISIBLE_DEVICES`/etc. below. |

GPU device-selection variables `llmman serve` forwards to every
`llama-server` it spawns (local or `--ociman` container):
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `GGML_VK_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`,
`HSA_OVERRIDE_GFX_VERSION`.
