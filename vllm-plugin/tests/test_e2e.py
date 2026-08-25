"""End-to-end tests against a real `llmman` binary.

Unlike the rest of this directory (which stubs `llmman`, monkeypatches
`_patch.resolve`, or never constructs a real `ModelConfig`), these tests
call this plugin's real public surface (`_llmman_cli.resolve()`,
`_patch._make_patched()`) against a real `llmman` binary and a real
registry pull — same "no mocks" reasoning as `tests/launch_e2e.rs`.

Gated behind the `e2e` pytest marker (excluded by default via
`pyproject.toml`'s `addopts`); run explicitly with:

    pytest -m e2e -v

CI runs this in `.github/workflows/ci.yml`'s `e2e` job, right after
`cargo test --test launch_e2e`, on Linux x86_64/aarch64 and macOS
aarch64. That ordering matters for the two lightweight tests below:
`launch_e2e.rs`'s `warm_model()` has already pulled `MODEL` into the
default store, so `resolve()` here hits a warm cache instead of a
second ~740MB download. The `vllm serve` test pays its own separate
pull instead (see its own docstring).

Every test here skips (not fails) when its own prerequisite (`llmman`,
`vllm`) isn't available, same as `tests/launch_e2e.rs`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_llmman import _patch
from vllm_llmman._llmman_cli import LlmmanNotFoundError, llmman_binary, resolve

pytestmark = pytest.mark.e2e

# Same short name as tests/launch_e2e.rs::MODEL, resolving to the same
# ~740MB docker.io/ai/qwen3.5:0.8b already warmed into the store there.
# GGUF vs. safetensors doesn't matter to resolve()/`_patch`'s hook (both
# only ever consume the JSON's `path`), so reusing that warm cache here
# is strictly cheaper than a second pull.
MODEL = "qwen3.5:0.8b"

# The real safetensors counterpart of MODEL (same Docker Hub repo,
# `ai/qwen3.5:0.8b-safetensors`, ~1.7GB CNCF ModelPack manifest) — this
# plugin's actual reason to exist is loading safetensors via vLLM, which
# can't load a bare .gguf file, so the vllm serve test below needs its
# own separate pull rather than reusing MODEL's cache.
MODEL_SAFETENSORS = "qwen3.5:0.8b-safetensors"

# Same convention as launch_e2e.rs's own PROMPT: reply quality isn't
# under test, the resolve/serve plumbing is.
PROMPT = "Reply with exactly the single word: pong"

# Explicit --served-model-name for the vllm serve test, rather than
# relying on whatever vLLM would default it to from a raw --model
# oci://... argument.
SERVED_MODEL_NAME = "vllm-llmman-e2e"

# Generous: a cold ~1.7GB pull plus real CPU weight loading, no GPU
# assumed anywhere. 1.5x launch_e2e.rs's own TIMEOUT (600s) for CPU-only
# inference and a larger checkout.
VLLM_STARTUP_TIMEOUT = 900


def _require_llmman() -> str:
    """The real `llmman` binary path, or a skip."""
    try:
        return llmman_binary()
    except LlmmanNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def resolved_model() -> dict:
    """Resolves `MODEL` via the real `llmman` binary once per module."""
    _require_llmman()
    return resolve(MODEL)


def test_resolve_pulls_and_returns_a_real_existing_path(resolved_model):
    """Checks `llmman resolve`'s JSON contract (see `src/cmd/resolve.rs`)
    against a real subprocess call.
    """
    assert resolved_model["reference"] == "docker.io/ai/qwen3.5:0.8b"
    assert resolved_model["format"] in ("gguf", "safetensors")

    path = Path(resolved_model["path"])
    assert path.exists(), f"resolved path does not exist: {path}"
    if resolved_model["format"] == "gguf":
        assert path.is_file()
        assert path.stat().st_size > 0
    else:
        assert path.is_dir()
        assert (path / "config.json").exists()


def test_resolve_is_idempotent_against_an_already_pulled_reference(resolved_model):
    """A second `resolve()` for an already-pulled reference returns the
    same path without erroring — the path every vLLM process (API
    server, engine core, each worker) after the first takes.
    """
    again = resolve(MODEL)
    assert again == resolved_model


def test_patched_hook_resolves_a_real_oci_reference_end_to_end():
    """`_patch._make_patched`'s hook, against the real (never
    monkeypatched here) `_llmman_cli.resolve()`: recognizes `oci://`,
    strips it, and rewrites `self.model` to the resolved local path.
    `original` fails the test if called — an `oci://` reference must
    never fall through to vLLM's own runai path.
    """
    _require_llmman()

    def original(self, model, tokenizer):
        pytest.fail("must not delegate to the original runai hook for an oci:// reference")

    patched = _patch._make_patched(original)

    ref = f"oci://{MODEL}"
    cfg = SimpleNamespace(model=ref, tokenizer=ref, model_weights=None)
    patched(cfg, cfg.model, cfg.tokenizer)

    assert cfg.model_weights == ref  # original oci:// reference preserved
    assert cfg.tokenizer == cfg.model  # shared tokenizer, resolved once

    path = Path(cfg.model)
    assert path.exists(), f"patched hook rewrote model to a nonexistent path: {path}"
    assert path != Path(ref)  # actually rewritten, not left as-is


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path: Path, n: int = 8000) -> str:
    """The last `n` bytes of `path` — `vllm serve`'s log, attached to
    failure messages since its output goes to a file, not pytest's
    captured output.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "<no log file>"
    return data[-n:].decode(errors="replace")


def _wait_for_health(port: int, timeout: float, proc: subprocess.Popen, log_path: Path) -> None:
    """Polls `/health` until it answers 200, the process exits early, or
    `timeout` is exhausted.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"vllm serve exited early (code {proc.returncode}) before becoming healthy\n"
                f"--- {log_path} tail ---\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(2)
    pytest.fail(
        f"vllm serve did not become healthy within {timeout}s\n"
        f"--- {log_path} tail ---\n{_tail(log_path)}"
    )


def _chat(port: int, prompt: str) -> str:
    """One real `/v1/chat/completions` request, stdlib only."""
    body = json.dumps(
        {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def _terminate(proc: subprocess.Popen) -> None:
    """Kills `proc`'s entire process group (it was spawned with
    `start_new_session=True`), escalating SIGTERM -> SIGKILL.

    Always sends both signals rather than returning as soon as `proc`
    itself exits: vLLM spawns its own worker subprocess(es) in the same
    group, which can outlive the direct `vllm` process, so checking only
    `proc`'s own exit status would miss them. `os.killpg` raising
    `ProcessLookupError` is the actual signal the whole group is gone
    (a process group only disappears once empty) — the reliable stop
    condition here, unlike `proc.poll()`.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def test_vllm_serve_resolves_and_serves_a_real_oci_reference(tmp_path):
    """A real `vllm serve oci://qwen3.5:0.8b-safetensors` subprocess,
    launched as a real user would, with nothing here pre-resolving or
    pre-pulling the reference first.

    Deliberately never calls `_llmman_cli.resolve()`/`llmman resolve`
    directly: the point is proving vLLM's own plugin loading discovers
    and installs this package's hook on its own, and that constructing a
    real `ModelConfig` for an `oci://` reference during vLLM's own
    startup is what triggers the pull.

    `--dtype float16`: the one dtype vLLM's CPU platform supports on
    every OS in ci.yml's matrix (bf16 isn't supported on macOS CPU).
    `--enforce-eager`: skips CUDA-graph startup work that doesn't apply
    on CPU. `--max-model-len 1024`: bounds the CPU KV-cache for this
    test's tiny prompt/reply.
    """
    pytest.importorskip("vllm")
    binary = shutil.which("vllm")
    if not binary:
        pytest.skip("`vllm` console script not found on PATH")
    _require_llmman()

    port = _free_port()
    log_path = tmp_path / "vllm-serve.log"
    cmd = [
        binary,
        "serve",
        f"oci://{MODEL_SAFETENSORS}",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--enforce-eager",
        "--max-model-len",
        "1024",
    ]
    env = dict(os.environ)
    # Bounds vLLM CPU's own KV-cache arena, well under its default
    # sizing heuristic, to avoid OOM on a shared/constrained CI runner.
    env.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")

    log_file = open(log_path, "wb")
    try:
        # start_new_session=True: see _terminate. Output goes to a file,
        # not a pipe: a long-running server's log can otherwise deadlock
        # subprocess.PIPE once its OS buffer fills with nothing draining
        # it (see launch_e2e.rs's spawn_reader for the Rust-side fix).
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()

    try:
        _wait_for_health(port, VLLM_STARTUP_TIMEOUT, proc, log_path)
        reply = _chat(port, PROMPT)
        assert "pong" in reply.lower(), (
            f'expected the served model\'s reply to contain "pong", got: {reply!r}\n'
            f"--- {log_path} tail ---\n{_tail(log_path)}"
        )
    finally:
        _terminate(proc)
