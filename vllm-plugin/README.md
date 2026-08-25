# vllm-llmman

A [vLLM](https://github.com/vllm-project/vllm) plugin that lets
`--model` point at a [CNCF ModelPack](https://github.com/modelpack/model-spec)
OCI image, pulled and extracted via [`llmman`](https://github.com/llmmanorg/llmman),
instead of only a HuggingFace Hub repo id.

```
pip install vllm-llmman
vllm serve oci://ghcr.io/org/model:tag
```

## Requirements

- vLLM (any recent version — no vLLM core changes required, see
  [How it works](#how-it-works)).
- The `llmman` binary on `PATH` (or point `LLMMAN_BIN` at it):

  ```
  curl -fsSL https://raw.githubusercontent.com/llmmanorg/llmman/main/install.sh | sh
  ```

## How it works

vLLM already has one extension-friendly rewrite hook that runs before
anything else touches `model=`/`tokenizer=`:
`ModelConfig.maybe_pull_model_tokenizer_for_runai` (today wired up only
for `s3://`/`gs://`/`az://`, pulling the reference into a local
directory and rewriting `self.model`/`self.tokenizer` to point at it).

This plugin registers itself via vLLM's `vllm.general_plugins` entry
point (loaded once per process, before any `ModelConfig` is built — see
`docs/design/plugin_system.md` in the vLLM repo) and wraps that same
hook: when `model=`/`tokenizer=` starts with `oci://`, it shells out to
`llmman resolve <reference>` (see llmman's
`src/cmd/resolve.rs`), which pulls the image into llmman's local OCI
store if it isn't already there, extracts it, and returns the local
path as JSON. That path is what vLLM's default HuggingFace-format
loading then sees — same as if you had passed a local directory all
along. Every other `model=` value (a HF repo id, a local path, an
existing `s3://`/`gs://`/`az://` reference) is left completely
untouched, delegated straight to vLLM's original behavior.

No vLLM core file is modified — this only uses vLLM's own plugin entry
point plus a narrowly scoped runtime monkeypatch (see
`src/vllm_llmman/_patch.py`'s module docstring for exactly what and why
a monkeypatch was necessary here, unlike e.g. a custom `--load-format`,
which vLLM *does* let plugins register without any monkeypatching).

An explicit `oci://` scheme is required rather than guessing from a
bare `registry/name:tag` string, since that shape is
indistinguishable from a HuggingFace repo id (`org/model`) — guessing
would risk hijacking existing HF-backed `vllm serve org/model` calls the
moment this plugin is installed.

## Development

```
cd vllm-plugin
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest
pytest
```

`tests/test_install_integration.py` additionally exercises the real
`vllm.config.model.ModelConfig` hook, and is skipped automatically if
vLLM isn't installed in the current environment.

`tests/test_e2e.py` goes further: a real `llmman` binary resolving a
real `oci://` reference, no mocks — plus, when `vllm` is installed, a
real `vllm serve oci://qwen3.5:0.8b-safetensors` process, never
pre-resolved by the test itself, proving this plugin's entry point is
actually discovered by a real vLLM startup. Excluded from a plain
`pytest` run; run with `pytest -m e2e`.
