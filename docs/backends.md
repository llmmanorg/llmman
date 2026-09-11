# Inference backends

llmman does not ship an inference engine. `llmman serve` picks one that
already exists for the model format it finds, and runs it unmodified.

| Model format | Backend | Where it comes from |
|--------------|---------|---------------------|
| GGUF | `llama-server` in a container | `--runtime docker` / `podman` (Linux only; what `auto` tries first): the `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for your GPU |
| GGUF | [`llama-server`](https://github.com/ggml-org/llama.cpp) | `--runtime bin`: a prebuilt upstream release matching your OS/arch/GPU, downloaded and cached; `--runtime path`: the one on your `PATH` |
| safetensors | `vllm` in a container | `--runtime docker` / `podman` (Linux only): the `vllm/vllm-openai`, `rocm/vllm` or `vllm/vllm-openai-cpu` image for your GPU and architecture |
| safetensors | [`vllm`](https://github.com/vllm-project/vllm) | Your `PATH` (every non-container runtime) |
| safetensors | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) | Your `PATH`, on Apple Silicon macOS; preferred over `vllm` when present |
| GGUF diffusion (LTX-2) | llmman itself, on ggml | The `libggml`/`libllama` next to `llama-server`; see [the blog post](https://llmmanorg.github.io/blog/image-audio-and-video-generation/) |
| Diffusers safetensors | [`vllm serve --omni`](https://github.com/vllm-project/vllm-omni) | Your `PATH`'s `vllm` with the `vllm-omni` package installed |
| Diffusers safetensors | `vllm serve --omni` in a container | `--runtime docker` / `podman` (Linux only): the `vllm/vllm-omni` image (CUDA only) |

## Choosing a runtime

`llmman serve --runtime <auto|docker|podman|bin|path>` (or the
`LLMMAN_RUNTIME` environment variable, which also reaches the daemon
`llmman run`/`launch` start for you) says where the engine comes from:

| Value | llama-server | Notes |
|-------|--------------|-------|
| `docker`, `podman` | `ghcr.io/ggml-org/llama.cpp:server-<backend>-<release>` container | Linux only; safetensors models get the vLLM images too |
| `bin` | llmman's own download of llama.cpp's prebuilt release for this OS/arch/GPU | Cached under `~/.local/share/llmman/llama-server/<release>/`; nothing needed on `PATH` |
| `path` | the `llama-server` already on `PATH`, as-is | Never downloads anything; `--llama-cpp-version` does not apply |
| `auto` (default) | the first of `docker`, `podman`, `bin`, `path` that works here | Off Linux: `bin`, then `path` |

Under `auto`, a container engine "works" when its CLI is on `PATH`, its
daemon answers `docker info`/`podman info`, an NVIDIA host has the NVIDIA
Container Toolkit, and the llama.cpp image pulls; `bin` works when the
release downloads (or is already cached). Each step skipped is logged
with the reason. Whatever is chosen is fetched before the listener binds,
so the first request is never stuck behind a silent download.

`llmman serve --pull-only` does exactly that fetch, in the foreground with
the pull's own progress, then exits — run it once before starting a
detached daemon. With a container runtime and a safetensors `MODEL`
argument it pulls the vLLM image as well.

## llama.cpp

The release is pinned: unless `--llama-cpp-version <tag>` says otherwise,
both `bin` and the container images use the llama.cpp build llmman's own
CI tested (the `LLAMA_CPP_RELEASE` file in the repository), so what runs
is what was tested. `--llama-cpp-version latest` takes upstream's
floating latest instead.

For `bin`, llmman probes for CUDA, ROCm, Vulkan or Metal (in that order)
and downloads the matching prebuilt asset from llama.cpp's GitHub
releases. `LLMMAN_LLM_LIBRARY` overrides the probe; `LLMMAN_DEBUG=1`
shows what it found. llama.cpp publishes no prebuilt Linux CUDA binary,
so an NVIDIA host on Linux gets the CPU build from `bin` — the container
runtimes (which `auto` prefers for that reason) have CUDA images.

Context length, parallel slots, flash attention, KV-cache type and GPU
split are environment variables; see [configuration.md](configuration.md).

### In a container

On Linux, `--runtime docker` (or `podman`) runs `llama-server` from the
`ghcr.io/ggml-org/llama.cpp` image instead, picking the
`server-cuda`/`server-cuda13`/`server-rocm`/`server-vulkan`/`server` tag
for the host, suffixed with the pinned release.
`CUDA_VISIBLE_DEVICES` and friends are forwarded into the container.

The container is a sibling of the daemon, not part of its cgroup, so a
CPU limit on `llmman serve` is forwarded: when one binds (cgroup quota or
affinity mask), the container gets `--cpus <n>` and `llama-server` the
matching `--threads <n>` (a quota alone would leave it autodetecting a
thread per host core and throttling). No limit, no flags. An explicit
`LLAMA_ARG_THREADS` still wins. vLLM containers get the same `--cpus`.

Each of those images is also published with llmman in it, as
`docker.io/ai/llmman:<tag>` (`latest` is `server`) and `<tag>-<llmman
version>`, built from [`packaging/Dockerfile`](../packaging/Dockerfile) on
every release against the llama.cpp build CI tests. Same entrypoint and GPU
flags as upstream, plus `/usr/local/bin/llmman`. `LLMMAN_HOST` is preset to
`0.0.0.0:17434` (a loopback bind inside a container is unreachable even with
`-p`), so the daemon requires `LLMMAN_API_KEYS` or `LLMMAN_AUTH=off`
([configuration.md](configuration.md#authentication)); publish the port on
the host's loopback to keep it local. The store is `/root/.local/share/llmman`:

```sh
docker run -p 127.0.0.1:17434:17434 -e LLMMAN_API_KEYS=<key> \
  -v llmman:/root/.local/share/llmman --gpus all \
  --entrypoint llmman ai/llmman:server-cuda serve
```

## vLLM

Safetensors models are served by a separately installed `vllm`. Plain
`vllm` is CPU-only on macOS unless
[vllm-metal](https://github.com/vllm-project/vllm-metal) is installed.
`LLMMAN_CONTEXT_LENGTH` is forwarded as `--max-model-len`;
`LLMMAN_LOAD_TIMEOUT` (default 10 minutes) bounds a stalled load.

### In a container

On Linux, `--runtime docker` (or `podman`) runs `vllm serve` from a vLLM
image for a safetensors model, picked by the same GPU probe plus the
host architecture:

| Host GPU | x86_64 | aarch64 |
|----------|--------|---------|
| NVIDIA (CUDA 13) | `vllm/vllm-openai:latest-x86_64` | `vllm/vllm-openai:latest-aarch64` |
| NVIDIA (CUDA 12) | `vllm/vllm-openai:latest-x86_64-cu129` | `vllm/vllm-openai:latest-aarch64-cu129` |
| AMD (ROCm) | `rocm/vllm:latest` | not published upstream (`LLMMAN_LLM_LIBRARY=cpu` for the CPU image) |
| Vulkan-only or none | `vllm/vllm-openai-cpu:latest-x86_64` | `vllm/vllm-openai-cpu:latest-arm64` |

`--vllm-version <tag>` pins the release (the arch suffix is added for the
`vllm/` images; for `rocm/vllm` it is the whole tag).
`llmman serve --runtime docker --pull-only <model>` pulls the image an
already-pulled model needs (without a model, the llama.cpp image).
`CUDA_VISIBLE_DEVICES` and friends plus every `VLLM_*` variable are
forwarded into the container.

### vLLM-Omni (Diffusers pipelines)

A safetensors repository laid out as a Diffusers pipeline (a root
`model_index.json` next to `transformer/`, `vae/`, ...) is
served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni): the same
`vllm` launcher with `--omni`. Plain `vllm serve` cannot load one.

```sh
uv pip install vllm==0.28.0 vllm-omni     # into the environment `vllm` runs from
llmman run ORG/MODEL "A robot arm cleaning a plate in a kitchen"
llmman run ORG/MODEL --video --seconds 2 "A robot arm cleaning a plate"
```

If `vllm` is a `#!/path/to/python` script whose Python cannot import
`vllm_omni`, the load fails up front with a message saying so; otherwise
vLLM itself reports `unrecognized arguments: --omni`. The model answers
`/v1/images/generations` and `/v1/videos`, in llama-server's dialect
(`width`/`height`/`steps`/`cfg_scale`, streamed `image_generation.*`
events, a video job with a `content_url`) and vLLM-Omni's own (`size`,
`num_inference_steps`, `guidance_scale`, `num_frames`, `extra_params`);
unsent fields are left to the model's defaults. There is no
`/v1/audio/speech` for these models.

The server is started with `--no-guardrails`; `LLMMAN_VLLM_OMNI_GUARDRAILS=1`
leaves the pipeline's safety guardrails on (they may need extra packages
and gated weights). `LLMMAN_LOAD_TIMEOUT` is also passed as `--init-timeout`.

With a container runtime, the image is `vllm/vllm-omni:latest-x86_64` or
`latest-aarch64` (CUDA only; `--vllm-version` pins vLLM-Omni's release,
e.g. `v0.28.0`). `--pull-only <model>` picks it for a pulled Diffusers model.

### `vllm serve` from llmman's store

The inverse: the [`vllm-llmman`](https://pypi.org/project/vllm-llmman/)
plugin lets `vllm serve oci://<reference>` pull a CNCF ModelPack image
from any OCI registry, via `llmman` (`LLMMAN_BIN` if it is not on
`PATH`). See [vllm-plugin/README.md](../vllm-plugin/README.md).

## MLX (Apple Silicon)

On Apple Silicon, `mlx_lm.server` (`pip install mlx-lm`) is preferred
over `vllm` for safetensors when on `PATH`: Metal-accelerated, no vLLM
dependency, more model families than vllm-metal. `LLMMAN_CONTEXT_LENGTH`
is not forwarded and `/v1/embeddings` is unsupported.

## Registry transport

Registry access goes through a Go shim compiled into the binary, with
two implementations behind Cargo features. Building needs Rust and Go
1.22+.

**Docker (default)**, via
[containerd](https://github.com/containerd/containerd)'s resolver:

```
cargo build --release
```

**Podman**, via
[container-libs](https://github.com/podman-container-tools/container-libs):

```
cargo build --release --no-default-features --features podman
```
