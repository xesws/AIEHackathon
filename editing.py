"""Thin wrapper over the vendored HoReN editing backend (``third_party/horen``).

HoReN is a pre-existing external dependency (impl of arXiv 2605.08143); it is NEVER
reimplemented here. This module only adapts our memory objects to HoReN's API.

    edit(model, memory) -> dict(adapter, wrapper, reset, edit_seconds, codebook_size)

OUTPUT FORMAT (TBD resolved in SPIKE 0 v0.2): a **side-module**, not a state_dict/delta.
``apply_horen_to_model`` replaces the ``inner_params`` submodule (llama3.1:
``model.layers[29].mlp.down_proj``) with a ``HopfieldAdapter`` (codebook keys/values/labels);
base weights stay frozen. The returned ``adapter`` IS the hot-swappable edit module that
``serving.model_host.swap_edit_module`` installs/removes.
"""
from __future__ import annotations

import time
from typing import Any

import serving.model_host as model_host


def edit(model: Any, memory: Any) -> dict:
    """Apply ONE HoReN edit onto the resident ``model``; return the installed adapter + timing.

    ``memory``: for SPIKE 0, a request dict ``{"prompt": ..., "target_new": ...}`` (extra keys
    like ``subject`` are ignored by HoReN's tokenizer). The ``MemoryItem.text -> (prompt,
    subject, target_new)`` decomposition is deferred to v0.3.

    Delegates entirely to ``third_party.horen`` (``apply_horen_to_model``); no HoReN logic here.
    """
    # imported lazily — model_host's import already put third_party/horen on sys.path
    from src.models.horen.horen_main import apply_horen_to_model

    request = memory if isinstance(memory, dict) else {"prompt": memory.text, "target_new": ""}
    tok = model_host.tokenizer()
    hp = model_host.hparams()

    t0 = time.time()
    wrapper, reset_fn = apply_horen_to_model(model, tok, [request], hp)
    edit_seconds = time.time() - t0

    adapter = model_host.edit_module()  # the now-installed HopfieldAdapter
    model_host.register_edit_module(adapter, edited_model=wrapper)

    return {
        "adapter": adapter,
        "wrapper": wrapper,
        "reset": reset_fn,
        "edit_seconds": edit_seconds,
        "codebook_size": wrapper.get_codebook_size(),
    }
