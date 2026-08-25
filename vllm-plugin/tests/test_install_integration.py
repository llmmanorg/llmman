"""End-to-end checks against the real vLLM modules `_patch.install()`
touches.

Skipped unless vLLM is actually importable in the current environment —
`_patch.py`'s unit tests (test_patch.py) cover the resolution logic itself
without needing vLLM installed at all; this file only exists to catch
vLLM's own hooks (`ModelConfig.maybe_pull_model_tokenizer_for_runai`,
`vllm.engine.arg_utils.maybe_override_with_speculators`) disappearing or
changing signature in a future vLLM release.
"""

import pytest

vllm_config_model = pytest.importorskip("vllm.config.model")
vllm_arg_utils = pytest.importorskip("vllm.engine.arg_utils")


@pytest.fixture
def restore_vllm_patches():
    """`_patch.install()` touches two module-level attributes as a pair;
    snapshot and restore both around each test so they can run in any
    order without leaking patched state into each other.

    `maybe_override_with_speculators` is treated as optional, same as
    `install()` itself does: older vLLM releases (this package declares
    no minimum vLLM version) don't have it at all.
    """
    from vllm_llmman import _patch

    ModelConfig = vllm_config_model.ModelConfig
    original_runai = ModelConfig.maybe_pull_model_tokenizer_for_runai
    original_speculators = getattr(vllm_arg_utils, "maybe_override_with_speculators", None)

    yield

    ModelConfig.maybe_pull_model_tokenizer_for_runai = original_runai
    if hasattr(ModelConfig, _patch._PATCHED_ATTR):
        delattr(ModelConfig, _patch._PATCHED_ATTR)
    if original_speculators is not None:
        vllm_arg_utils.maybe_override_with_speculators = original_speculators
    if hasattr(vllm_arg_utils, _patch._PATCHED_ATTR):
        delattr(vllm_arg_utils, _patch._PATCHED_ATTR)


def test_install_replaces_the_runai_hook_exactly_once(restore_vllm_patches):
    from vllm_llmman import _patch

    ModelConfig = vllm_config_model.ModelConfig
    original = ModelConfig.maybe_pull_model_tokenizer_for_runai

    _patch.install()
    patched_once = ModelConfig.maybe_pull_model_tokenizer_for_runai
    assert patched_once is not original

    _patch.install()  # idempotent
    assert ModelConfig.maybe_pull_model_tokenizer_for_runai is patched_once


def test_install_replaces_the_speculators_hook_exactly_once(restore_vllm_patches):
    from vllm_llmman import _patch

    original_speculators = getattr(vllm_arg_utils, "maybe_override_with_speculators", None)
    if original_speculators is None:
        pytest.skip("this vLLM release has no maybe_override_with_speculators hook")

    _patch.install()
    patched_once = vllm_arg_utils.maybe_override_with_speculators
    assert patched_once is not original_speculators

    _patch.install()  # idempotent
    assert vllm_arg_utils.maybe_override_with_speculators is patched_once
