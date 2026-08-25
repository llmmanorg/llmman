"""Hooks `model=oci://...` into vLLM's model resolution *before* vLLM's
own HuggingFace-oriented config/tokenizer loading ever runs — without
editing any vLLM core file.

vLLM already has exactly one hook that does "rewrite `self.model`/`self.
tokenizer` from a remote reference to a local directory, early in
`ModelConfig.__post_init__`, before anything else consumes them":
`ModelConfig.maybe_pull_model_tokenizer_for_runai` (today only wired up
for `s3://`/`gs://`/`az://`, via `is_runai_obj_uri`). There is no
equivalent `register_model_source_resolver()` extension point (unlike
`register_model_loader`/`register_config_parser`), so the only way to
reach this point from an out-of-tree plugin is to wrap that method at
runtime. See vLLM's `vllm/config/model.py` (`__post_init__`, line ~565)
and `vllm/transformers_utils/runai_utils.py` for the method being
wrapped.

vLLM 0.27 added a second, *earlier* HuggingFace touchpoint:
`EngineArgs.create_engine_config` calls `maybe_override_with_speculators`
directly on the raw `model`/`tokenizer` strings, before `ModelConfig` (and
therefore `maybe_pull_model_tokenizer_for_runai`) is ever constructed. It
guards this with `is_cloud_storage()`, which it uses to skip `s3://`/
`gs://`/`az://` — but that helper doesn't know about `oci://`, so an
unpatched vLLM 0.27+ calls `PretrainedConfig.get_config_dict("oci://...")`
and blows up before our other patch gets a chance to run. See vLLM's
`vllm/engine/arg_utils.py` (`create_engine_config`, line ~1917) and
`vllm/transformers_utils/config.py` (`maybe_override_with_speculators`).
We wrap that function for `oci://` `model` refs, short-circuiting
speculators config loading so the real resolution still happens later,
inside `maybe_pull_model_tokenizer_for_runai`.

Both `_make_patched` and `_make_patched_speculators` are pure functions
(no vLLM import) so they're unit testable against stubs/fakes; `install()`
is the only piece that actually touches vLLM's modules.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Protocol

from ._llmman_cli import resolve
from ._scheme import is_oci_ref, strip_scheme

logger = logging.getLogger("vllm_llmman")

_PATCHED_ATTR = "_vllm_llmman_patched"


class _ModelConfigLike(Protocol):
    model_weights: str | None
    model: str
    tokenizer: str


OriginalHook = Callable[[Any, str, str], None]


def _make_patched(original: OriginalHook) -> OriginalHook:
    """Build the replacement for `ModelConfig.
    maybe_pull_model_tokenizer_for_runai`. Delegates to `original`
    unchanged whenever neither `model` nor `tokenizer` uses a scheme this
    plugin recognizes, so every existing runai (`s3://`/`gs://`/`az://`)
    code path keeps working exactly as before.
    """

    def patched(self: _ModelConfigLike, model: str, tokenizer: str) -> None:
        if self.model_weights:
            return original(self, model, tokenizer)

        model_is_oci = is_oci_ref(model)
        tokenizer_is_oci = is_oci_ref(tokenizer)
        if not (model_is_oci or tokenizer_is_oci):
            return original(self, model, tokenizer)

        if model_is_oci:
            ref = strip_scheme(model)
            logger.info("llmman: resolving model %s", ref)
            result = resolve(ref)
            self.model_weights = model
            self.model = result["path"]

            if model == tokenizer:
                self.tokenizer = result["path"]
                return

        if tokenizer_is_oci and tokenizer != model:
            ref = strip_scheme(tokenizer)
            logger.info("llmman: resolving tokenizer %s", ref)
            result = resolve(ref)
            self.tokenizer = result["path"]

    patched.__name__ = getattr(original, "__name__", "maybe_pull_model_tokenizer_for_runai")
    patched.__doc__ = original.__doc__
    return patched


SpeculatorsHook = Callable[..., tuple[str, "str | None", "dict[str, Any] | None"]]


def _make_patched_speculators(original: SpeculatorsHook) -> SpeculatorsHook:
    """Build the replacement for
    `vllm.engine.arg_utils.maybe_override_with_speculators`. Delegates to
    `original` unchanged whenever `model` isn't an `oci://` reference, so
    every existing (HuggingFace, and `s3://`/`gs://`/`az://` via
    `is_cloud_storage`) code path keeps working exactly as before.

    Only checks `model`, not `tokenizer`: `original` only ever reads
    speculators config from `model` (via `PretrainedConfig.
    get_config_dict(model, ...)`) — an `oci://` `tokenizer` alongside a
    real HF `model` is not this crash and must still go through
    `original`'s real speculators detection for that model (see vLLM's
    `vllm/transformers_utils/config.py`).

    vLLM's only call site (`create_engine_config`) passes every argument
    by keyword, but `original`'s own signature (`model, tokenizer,
    trust_remote_code, revision=None, ...`) allows positional calls too.
    Binding against that real signature via `inspect.signature`, rather
    than hand-rolling `patched`'s own parameter list, means any calling
    convention `original` itself accepts keeps working here unchanged.
    """
    signature = inspect.signature(original)

    def patched(*args: Any, **kwargs: Any) -> tuple[str, "str | None", "dict[str, Any] | None"]:
        bound = signature.bind(*args, **kwargs)
        model = bound.arguments["model"]
        if is_oci_ref(model):
            tokenizer = bound.arguments.get("tokenizer")
            vllm_speculative_config = bound.arguments.get("vllm_speculative_config")
            return model, tokenizer, vllm_speculative_config
        return original(*args, **kwargs)

    patched.__name__ = getattr(original, "__name__", "maybe_override_with_speculators")
    patched.__doc__ = original.__doc__
    patched.__signature__ = signature
    return patched


def install() -> None:
    """Idempotently monkeypatch vLLM in the current process. Safe to call
    more than once (plugins may be loaded more than once per process —
    see `load_general_plugins`'s own warning) and safe to call from
    multiple processes (API server, engine core, workers all load
    `vllm.general_plugins` independently).
    """
    from vllm.config.model import ModelConfig

    if getattr(ModelConfig, _PATCHED_ATTR, False):
        return

    original = ModelConfig.maybe_pull_model_tokenizer_for_runai
    ModelConfig.maybe_pull_model_tokenizer_for_runai = _make_patched(original)
    setattr(ModelConfig, _PATCHED_ATTR, True)

    # Older vLLM releases don't have this speculators touchpoint at all;
    # only patch it when present.
    import vllm.engine.arg_utils as arg_utils

    original_speculators = getattr(arg_utils, "maybe_override_with_speculators", None)
    if original_speculators is not None and not getattr(arg_utils, _PATCHED_ATTR, False):
        arg_utils.maybe_override_with_speculators = _make_patched_speculators(original_speculators)
        setattr(arg_utils, _PATCHED_ATTR, True)
