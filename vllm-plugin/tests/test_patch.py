from types import SimpleNamespace

import pytest

from vllm_llmman import _patch


def _stub(model, tokenizer, model_weights=None):
    return SimpleNamespace(model=model, tokenizer=tokenizer, model_weights=model_weights)


def test_delegates_to_original_when_model_weights_already_set():
    calls = []
    original = lambda self, model, tokenizer: calls.append((model, tokenizer))
    patched = _patch._make_patched(original)

    cfg = _stub("oci://ghcr.io/org/model:tag", "oci://ghcr.io/org/model:tag", model_weights="already-pulled")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert calls == [(cfg.model, cfg.tokenizer)]
    assert cfg.model == "oci://ghcr.io/org/model:tag"  # untouched


def test_delegates_to_original_for_non_oci_refs():
    calls = []
    original = lambda self, model, tokenizer: calls.append((model, tokenizer))
    patched = _patch._make_patched(original)

    cfg = _stub("s3://bucket/model", "s3://bucket/model")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert calls == [("s3://bucket/model", "s3://bucket/model")]


def test_resolves_model_and_shared_tokenizer(monkeypatch):
    seen_refs = []

    def fake_resolve(ref, **kwargs):
        seen_refs.append(ref)
        return {"reference": ref, "path": "/cache/model-dir", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    ref = "oci://ghcr.io/org/model:tag"
    cfg = _stub(ref, ref)
    patched(cfg, cfg.model, cfg.tokenizer)

    assert seen_refs == ["ghcr.io/org/model:tag"]
    assert cfg.model == "/cache/model-dir"
    assert cfg.tokenizer == "/cache/model-dir"
    assert cfg.model_weights == ref


def test_resolves_model_and_distinct_tokenizer_separately(monkeypatch):
    seen_refs = []

    def fake_resolve(ref, **kwargs):
        seen_refs.append(ref)
        return {"reference": ref, "path": f"/cache/{ref.split('/')[-1]}", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    model_ref = "oci://ghcr.io/org/model:tag"
    tok_ref = "oci://ghcr.io/org/tokenizer:tag"
    cfg = _stub(model_ref, tok_ref)
    patched(cfg, cfg.model, cfg.tokenizer)

    assert seen_refs == ["ghcr.io/org/model:tag", "ghcr.io/org/tokenizer:tag"]
    assert cfg.model == "/cache/model:tag"
    assert cfg.tokenizer == "/cache/tokenizer:tag"
    assert cfg.model_weights == model_ref


def test_resolves_tokenizer_only_when_model_is_not_oci(monkeypatch):
    def fake_resolve(ref, **kwargs):
        return {"reference": ref, "path": "/cache/tok", "format": "safetensors"}

    monkeypatch.setattr(_patch, "resolve", fake_resolve)
    patched = _patch._make_patched(original=lambda *a: pytest.fail("must not delegate"))

    cfg = _stub("meta-llama/Llama-3-8B", "oci://ghcr.io/org/tok:tag")
    patched(cfg, cfg.model, cfg.tokenizer)

    assert cfg.model == "meta-llama/Llama-3-8B"  # untouched, not an oci:// ref
    assert cfg.tokenizer == "/cache/tok"
    assert cfg.model_weights is None


def _fake_speculators_hook(calls):
    """A stand-in for the real `vllm.transformers_utils.config.
    maybe_override_with_speculators`, matching its actual signature
    (`model, tokenizer, trust_remote_code, revision=None,
    vllm_speculative_config=None, hf_token=None`) exactly — required
    because `_make_patched_speculators` binds against `original`'s own
    `inspect.signature()`, not a hand-rolled parameter list, so a fake
    with a different signature (e.g. a bare `lambda *a, **k`) would
    exercise a call it can never actually see in production.
    """

    def original(
        model,
        tokenizer=None,
        trust_remote_code=False,
        revision=None,
        vllm_speculative_config=None,
        hf_token=None,
    ):
        calls.append((model, tokenizer, trust_remote_code, revision, vllm_speculative_config, hf_token))
        return model, tokenizer, vllm_speculative_config

    return original


def test_speculators_delegates_to_original_for_non_oci_refs():
    calls = []
    patched = _patch._make_patched_speculators(_fake_speculators_hook(calls))

    result = patched(
        model="meta-llama/Llama-3-8B",
        tokenizer="meta-llama/Llama-3-8B",
        revision=None,
        trust_remote_code=False,
        vllm_speculative_config=None,
        hf_token=None,
    )

    assert calls == [("meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B", False, None, None, None)]
    assert result == ("meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B", None)


def test_speculators_delegates_to_original_when_called_positionally():
    """Same real signature vLLM's own function has
    (`model, tokenizer, trust_remote_code, ...`); `create_engine_config`
    calls it with keywords today, but nothing here should assume that's
    the only calling convention `original` itself supports.
    """
    calls = []
    patched = _patch._make_patched_speculators(_fake_speculators_hook(calls))

    result = patched("meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B", False)

    assert calls == [("meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B", False, None, None, None)]
    assert result == ("meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B", None)


def test_speculators_short_circuits_for_an_oci_model_without_calling_original():
    def must_not_delegate(model, tokenizer=None, trust_remote_code=False, revision=None, vllm_speculative_config=None, hf_token=None):
        pytest.fail("must not delegate")

    patched = _patch._make_patched_speculators(must_not_delegate)

    result = patched(
        model="oci://ghcr.io/org/model:tag",
        tokenizer="oci://ghcr.io/org/model:tag",
        revision=None,
        trust_remote_code=False,
        vllm_speculative_config={"already": "set"},
        hf_token=None,
    )

    assert result == (
        "oci://ghcr.io/org/model:tag",
        "oci://ghcr.io/org/model:tag",
        {"already": "set"},
    )


def test_speculators_delegates_when_only_tokenizer_is_oci():
    """`original` only ever reads speculators config from `model`
    (see `_make_patched_speculators`'s docstring) — an `oci://`
    `tokenizer` alongside a real HF `model` isn't the crash this patch
    exists for, and must still go through real speculators detection.
    """
    calls = []
    patched = _patch._make_patched_speculators(_fake_speculators_hook(calls))

    result = patched(
        model="meta-llama/Llama-3-8B",
        tokenizer="oci://ghcr.io/org/tok:tag",
        vllm_speculative_config=None,
    )

    assert calls == [("meta-llama/Llama-3-8B", "oci://ghcr.io/org/tok:tag", False, None, None, None)]
    assert result == ("meta-llama/Llama-3-8B", "oci://ghcr.io/org/tok:tag", None)
