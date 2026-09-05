# Inference backends

llmman does not ship an inference engine. `llmman serve` picks one that
already exists for the model format it finds, and runs it unmodified.

| Model format | Backend | Where it comes from |
|--------------|---------|---------------------|
| GGUF | [`llama-server`](https://github.com/ggml-org/llama.cpp) | Your `PATH` if it is there; otherwise a prebuilt upstream release matching your OS/arch/GPU, downloaded and cached on first use |
| GGUF | `llama-server` in a container | `--ociman docker` / `--ociman podman` (Linux only): the `ghcr.io/ggml-org/llama.cpp:server-<backend>` image for your GPU |
| safetensors | [`vllm`](https://github.com/vllm-project/vllm) | Your `PATH` |
| safetensors | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) | Your `PATH`, on Apple Silicon macOS; preferred over `vllm` when present |

## llama.cpp

A `llama-server` on `PATH` is used as-is. Otherwise llmman probes for
CUDA, ROCm, Vulkan or Metal (in that order) and downloads the matching
prebuilt release from llama.cpp's GitHub releases. `LLMMAN_LLM_LIBRARY`
overrides the probe; `LLMMAN_DEBUG=1` shows what it found.

`--llama-cpp-version <tag>` pins a release (and forces the managed
download even with a `llama-server` on `PATH`). `--pull-bin` downloads
it in the foreground and exits, so the first request is not stuck behind
a silent download.

Context length, parallel slots, flash attention, KV-cache type and GPU
split are environment variables; see [configuration.md](configuration.md).

### In a container

On Linux, `--ociman docker` (or `podman`) runs `llama-server` from the
`ghcr.io/ggml-org/llama.cpp` image instead, picking the
`server-cuda`/`server-cuda13`/`server-rocm`/`server-vulkan`/`server` tag
for the host.
`--llama-cpp-version` pins the image tag; `--pull-oci` pulls it in the
foreground and exits. `CUDA_VISIBLE_DEVICES` and friends are forwarded
into the container.

## vLLM

Safetensors models are served by a separately installed `vllm`. Plain
`vllm` is CPU-only on macOS unless
[vllm-metal](https://github.com/vllm-project/vllm-metal) is installed.
`LLMMAN_CONTEXT_LENGTH` is forwarded as `--max-model-len`;
`LLMMAN_LOAD_TIMEOUT` (default 10 minutes) bounds a stalled load.

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
