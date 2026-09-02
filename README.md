# llmman

A command-line tool for managing and serving LLM models using OCI registries.
Models are packaged as standard OCI artifacts and stored in any compatible registry (Docker Hub, GHCR, quay, self-hosted, etc.).
`llmman serve` exposes Ollama-, OpenAI-, and Anthropic-compatible HTTP APIs.

## Commands

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `list`    | List locally stored models, or a hosted provider's (`--provider`) models |
| `ps`      | List models currently loaded |
| `providers` | List the hosted providers `--provider` can route to |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `login`   | Log in to a container registry |
| `logout`  | Log out from a container registry |

## Install

**Linux, macOS:**

```
curl -fsSL https://raw.githubusercontent.com/llmmanorg/llmman/main/install.sh | sh
```

**Windows (PowerShell):**

```
irm https://raw.githubusercontent.com/llmmanorg/llmman/main/install.ps1 | iex
```

## Quick start

### Pull a model

```
llmman pull gemma4
```

### Transfer a model between locations

Transfer an image directly from a source to a destination without storing
it locally first, e.g. HuggingFace straight to an OCI registry:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF docker.io/owner/model:latest
```

Any source `llmman pull` understands (an OCI registry, `hf://`, `ms://`, ...) can be paired with any OCI registry destination.

### Serve

Start the inference server. GGUF models are served by `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp), used from `PATH` if it's already there; otherwise `llmman` downloads and caches a prebuilt release matching your OS/arch/GPU automatically (see `--llama-cpp-version` to pin a specific release). Safetensors models are served by [`vllm`](https://github.com/vllm-project/vllm) (plain `vllm` is CPU-only on macOS, unless you separately install [vllm-metal](https://github.com/vllm-project/vllm-metal) for Metal GPU support), or, on Apple Silicon macOS, by [`mlx-lm`](https://github.com/ml-explore/mlx-lm)'s `mlx_lm.server` instead when it's on `PATH`: Metal-accelerated, with no vLLM dependency at all, and it supports more model families than vllm-metal does.

```
llmman serve
```

The server listens on `127.0.0.1:17434` by default, overridable via `LLMMAN_HOST`, and exposes:

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens` |
| Anthropic | `/v1/messages` |
| llmman | `/llmman/providers`, `/llmman/providers/{id}` |
| Prometheus | `/metrics` (off unless `LLMMAN_METRICS=1`) |

`/llmman/...` is llmman's own API, not a compatibility surface — no
upstream API has a notion of a [models.dev](https://models.dev) provider
(see [Hosted providers](#hosted-providers)). `/llmman/providers` lists
the ones this daemon can route to, each with its API-key variable,
whether the daemon has that key, and how many models it serves;
`/llmman/providers/{id}` adds those models and what each costs in US
dollars per million tokens (absent, not zero, where models.dev publishes
no price). `llmman providers`, `list --provider`, `run --provider` and
`launch --provider` are all clients of it, so the catalog is fetched and
cached in one process: the one that forwards the request upstream.

`/metrics` is a Prometheus scrape target in the text exposition format.
It is **off by default**; set `LLMMAN_METRICS=1` (or `true`, `yes`, `on`)
to serve it. Without that the route is absent and answers 404, the same
as any other path this daemon does not serve. The router has no
authentication and `LLMMAN_HOST` will bind it to any interface, so an
upgrade should not start publishing a daemon's version, route mix, model
names and model churn to whoever can reach the port.

```
LLMMAN_METRICS=1 llmman serve
```

It is on Prometheus' default path, so a scrape config needs no
`metrics_path`:

```yaml
scrape_configs:
  - job_name: llmman
    static_configs:
      - targets: ["127.0.0.1:17434"]
```

Fifteen metric families:

| Metric | Type | Labels | What it tells you |
|--------|------|--------|-------------------|
| `llmman_build_info` | gauge | `version` | Which build is running; join against it to break a graph out by version. |
| `llmman_start_time_seconds` | gauge | — | `time() - llmman_start_time_seconds` is uptime; a step down is a restart. |
| `llmman_scheduling_requests_in_flight` | gauge | — | Requests doing model-scheduling work right now. |
| `llmman_scheduling_capacity` | gauge | — | The limit those are counted against, i.e. `LLMMAN_MAX_QUEUE.max(1)`. |
| `llmman_scheduling_rejections_total` | counter | — | Requests refused with a 503 because that limit was full. |
| `llmman_models_loaded` | gauge | — | Backends currently running, the set `/api/ps` reports. |
| `llmman_models_loading` | gauge | — | Loads under way; `loaded + loading` is what `LLMMAN_MAX_LOADED_MODELS` caps. |
| `llmman_model_up` | gauge | `model`, `engine` | 1 while the backend process is alive, 0 once it has died but llmman has not noticed. |
| `llmman_model_loads_total` | counter | `model` | Cold starts per model — the churn a too-small `LLMMAN_MAX_LOADED_MODELS` produces. |
| `llmman_model_load_duration_seconds` | histogram | `model` | How long a cold start takes, from admission to ready. |
| `llmman_model_load_oom_retries_total` | counter | `model`, `strategy` | Loads that hit an out-of-memory failure and retried: `evict_others`, `split_mode`, `ctx_shrink`. |
| `llmman_model_unloads_total` | counter | `model`, `reason` | `idle`, `requested`, `crashed`, `oom`, `evicted`. |
| `llmman_http_requests_total` | counter | `route`, `status` | Request rate and error rate by matched route. |
| `llmman_http_request_ttfb_seconds` | histogram | `route` | Time to response headers. This is the latency number. |
| `llmman_http_request_duration_seconds` | histogram | `route` | Time to the last byte of the body. |

The two request histograms are separate on purpose. On a streaming route
(`/api/chat`, `/v1/chat/completions`) time to the last byte mostly tracks
how many tokens were asked for, so graphing it as latency makes every long
completion look like a regression; time to first byte is what moves when a
load stalls or the queue backs up. Graphing only the first would hide a
stream that dies after its first byte.

`llmman_model_up` is the one thing no other metric here shows. llmman only
notices a dead backend when a request arrives for that model, so a backend
that died and was then never asked for again is still counted by
`llmman_models_loaded`. A scrape checks liveness directly, so
`llmman_model_up == 0` is that gap, and
`sum by (instance) (llmman_models_loaded) - sum by (instance) (llmman_model_up)`
is how many models are in it.

Three things it deliberately does not measure. A request for an
already-loaded model returns before admission control, so it is absent
from `llmman_scheduling_requests_in_flight` and is not capped by
`LLMMAN_MAX_QUEUE`. Requests to unknown paths are not counted, since
labelling by the requested path is how a metrics endpoint acquires
unbounded cardinality, and neither are CORS preflights, which the CORS
layer answers before any handler runs. Per-model families exist only once
that model has been seen, since llmman cannot enumerate every model
reference a user might send.

The scrape route is instrumented like any other, so `route="/metrics"`
reports what a scrape costs; exclude that label when asking about
application latency.

It is also the one route outside the CORS layer, so it answers without an
`Access-Control-Allow-Origin` header. The default origin list allows any
page served from localhost on any port, which would otherwise let one read
this daemon's version, route mix and model churn out of a browser. Nothing
in a browser scrapes a metrics endpoint, so that costs nothing.

Per-token counters (prompt and eval totals, KV-cache usage) are
deliberately absent: llmman does not run inference, `llama-server` does,
and it already publishes exactly those on its own `/metrics` when started
with `--metrics`. Scrape the backend for those rather than have llmman
keep a second copy that drifts.

`/v1/responses` implements the OpenAI Responses API (the dialect [OpenAI
Codex](https://github.com/openai/codex) requires), including streaming SSE
and function-tool-call re-mapping. This is a plain pass-through to
`llama-server`'s own native `/v1/responses` support, so a recent enough
`llama-server` build is required for it to work.

Use it as an Ollama-compatible server:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Or with any Ollama, Anthropic or OpenAI-compatible client.

Models are loaded on demand. Each model gets its own `llama-server` subprocess on a random loopback port; subsequent requests reuse the running process.

`/api/chat` also supports Ollama's `tools` (function calling, streamed back
as `message.tool_calls`), `images` (vision, base64, same as Ollama's own
wire format), and `format` (`"json"` or a JSON Schema object, for
constrained structured output).

An idle, unused model is automatically unloaded after `keep_alive`
(default 5 minutes, matching Ollama; set per-request, or daemon-wide via
`LLMMAN_KEEP_ALIVE`), and `llmman ps`/`/api/ps` reports each model's
`expires_at`.

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
| `LLMMAN_MODELS` | Local store directory, overriding the default below. `pull`/`push`/`run`/etc. go through the daemon and always use whichever store it was started with. |
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

### Launch an integration

Point an integration at a model in one step. `llmman launch` starts `serve` in the background if it isn't already running (preloading the requested model), then sets the right environment variables and execs the integration:

```
llmman launch claude --model gemma4
```

Run `llmman launch` with no arguments to list the supported integrations (Claude Code, OpenCode) and whether each is installed. Any extra arguments after `--` are forwarded to the integration's own CLI.

Short names work wherever a model reference is accepted.

#### Hosted providers

`--provider` points llmman at a model it doesn't serve itself — from
`launch`, from `run`, and from `list`:

```sh
export OPENROUTER_API_KEY=...
llmman providers                                    # which providers, and is the key set
llmman list --provider openrouter                   # its models, and $/Mtok in and out
llmman run --provider openrouter qwen/qwen3-coder   # chat with one directly
llmman launch opencode --provider openrouter --model qwen/qwen3-coder
```

The provider list is fetched at runtime from
[models.dev](https://models.dev) — the same catalog `opencode` resolves
its own providers from — so a newly added provider works without an
llmman release. It's cached for 24 hours, and a stale copy is used if the
fetch fails, so being offline means an out-of-date list rather than a
broken command.

All four commands ask the daemon over [`/llmman/providers`](#serve)
rather than fetching models.dev themselves, so the cache outlives any
single command and the key status reported is that of the environment
whose key actually gets spent.

Requests still go through `llmman serve`; `--provider` changes where the
daemon forwards them, not who the client talks to. So one endpoint and
one place integrations are configured, whether a model is local or
hosted, and both usable from the same session.

The API key is read from the variable models.dev names for that provider
and travels per request, never to disk. `hermes` is the exception: llmman
configures it through a file on disk, so it can't carry a key and
`llmman serve` needs the variable in its own environment instead. That
fallback is only used for a daemon bound to loopback, and never for a
browser request from another site — it bounds the blast radius rather
than authenticating anyone, so on a shared machine prefer an integration
that sends its own key. `cline`, `kimi`, `copilot`, `gemini` and `openclaw` can't be
used with `--provider` at all — the first two pick their own model rather
than taking llmman's, `copilot` has no way to send a key, `gemini` feeds
its key to a native Google client llmman can't confirm it has redirected,
and `openclaw` only takes a model during first-run onboarding.

Being OpenAI-compatible doesn't mean implementing every OpenAI route.
`codex` uses `/v1/responses`, which `openai`, `groq` and `openrouter`
answer but `anthropic` and `mistral` don't; models.dev carries no
capability data to filter on, so llmman turns that provider's bare 404
into an explanation naming what's missing rather than guessing up front.

`--provider` needs a local `llmman serve`. The daemon talks plain HTTP
and has no authentication, so neither `run` nor `launch` will send a real
key to a remote `LLMMAN_HOST`, and a daemon bound to anything but
loopback will not spend its own environment's key on behalf of a caller
that didn't present one. (`llmman providers` and `llmman list
--provider` read the catalog only, no key involved, and work against any
daemon.)

Providers llmman cannot reach with a single bearer token over an
OpenAI-compatible https endpoint are deliberately absent rather than
half-supported: Amazon Bedrock (SigV4 request signing), Google Vertex
(GCP service-account credentials), Azure, and the others whose endpoint
is per-account or whose wire format isn't OpenAI's.

### Use with vLLM directly

`llmman serve` already spawns `vllm` itself as a backend for safetensors
models. The [`vllm-llmman`](https://pypi.org/project/vllm-llmman/) plugin
is the inverse: install it alongside `vllm` and `vllm serve
oci://<reference>` pulls a CNCF ModelPack image directly, instead of a
HuggingFace repo.

### MLX (Apple Silicon)

On Apple Silicon macOS, `llmman serve` uses
[`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) instead of `vllm`
for safetensors models whenever it's on `PATH`, Metal-accelerated, with
no vLLM dependency at all (unlike getting the same acceleration out of
`vllm serve` itself via [vllm-metal](https://github.com/vllm-project/vllm-metal)).
Falls back to `vllm` otherwise. Doesn't support `/v1/embeddings`.

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

## Transport backends

The registry transport is a compiled-in Go shim. Two backends are available via Cargo feature flags.

### Docker (default)

Uses [`github.com/containerd/containerd`](https://github.com/containerd/containerd), the same OCI resolver used by Docker.

```
cargo build --release
```

### Podman

Uses [`github.com/podman-container-tools/container-libs`](https://github.com/podman-container-tools/container-libs), the same library Podman uses internally.

```
cargo build --release --no-default-features --features podman
```

