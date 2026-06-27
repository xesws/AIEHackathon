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


def edit(model: Any, memory: Any, *, key_mode: str = "chat") -> dict:
    """Apply ONE HoReN edit onto the resident ``model``; return the installed adapter + timing.

    ``memory``: an ALREADY-SPLIT request dict ``{"prompt": ..., "target_new": ...}`` (extra keys
    like ``subject`` are ignored by HoReN's tokenizer).

    Two easily-conflated things live near here; only one is unbuilt — do not confuse them:
      - TEXT DECOMPOSITION (``MemoryItem.text -> prompt/subject/target_new``): parsing a free-form
        fact string into edit fields. NOT IMPLEMENTED — callers must pass a pre-split dict. There
        is NO ``split`` function anywhere; this step simply does not exist yet.
      - QUERY-SPAN ISOLATION (the "chat key" / ``key_mode='chat'`` below): the v0.3 Plan-B fix that
        keys on only the user-question token span (excluding the chat scaffold). This IS DONE —
        in ``keying.py`` (``compute_key`` / ``query_span_in_rendered``) + the adapter's
        ``_pool_span`` / ``query_span`` (``third_party/horen/.../editor.py``). It is unrelated to
        the text decomposition above.

    ``key_mode`` (v0.3 Plan B — the QUERY-SPAN ISOLATION, not text decomposition):
      - ``"chat"`` (default, the fix): after the edit, APPEND a query-span-isolated chat key
        (the hero chat render of the stem) that reuses the same trained value, so the codebook
        serves BOTH the raw path (HoReN's native key) and the chat path (the appended key).
      - ``"raw"``: legacy — keep only HoReN's native raw key.

    Delegates entirely to ``third_party.horen`` (``apply_horen_to_model``); no HoReN logic here.
    """
    # imported lazily — model_host's import already put third_party/horen on sys.path
    from src.models.horen.editor import HOREN
    from src.models.horen.horen_main import apply_horen_to_model

    request = memory if isinstance(memory, dict) else {"prompt": memory.text, "target_new": ""}
    tok = model_host.tokenizer()
    hp = model_host.hparams()

    # Sequential editing (v0.5): after the first edit, the resident model IS the HOREN wrapper.
    # apply_horen_to_model expects the underlying HF model (it traverses model.model.layers…), so
    # unwrap one level — else HOREN.__init__ does parent_module(wrapper, "model.layers…") and fails
    # with "model.layers not found". Unwrapped, HOREN.__init__ finds the existing HopfieldAdapter
    # and add_key APPENDS into the SAME codebook → N edits stack instead of nesting wrappers.
    hf_model = model.model if isinstance(model, HOREN) else model

    t0 = time.time()
    wrapper, reset_fn = apply_horen_to_model(hf_model, tok, [request], hp)
    edit_seconds = time.time() - t0

    adapter = model_host.edit_module()  # the now-installed HopfieldAdapter
    if key_mode == "chat":
        _append_chat_key(wrapper, adapter, tok, request["prompt"])

    model_host.register_edit_module(adapter, edited_model=wrapper)

    return {
        "adapter": adapter,
        "wrapper": wrapper,
        "reset": reset_fn,
        "edit_seconds": edit_seconds,
        "codebook_size": wrapper.get_codebook_size(),
    }


def _append_chat_key(wrapper: Any, adapter: Any, tok: Any, stem: str) -> None:
    """Append a Plan-B query-span chat key for ``stem`` that reuses the value row HoReN just
    trained. Keeps the raw key intact (raw path stays green); the chat read-key now matches."""
    import torch

    from keying import compute_key

    chat_key = compute_key(stem, templated=True, hf_model=wrapper.model, tok=tok, adapter=adapter)
    v_idx = wrapper.edit_log["chosen_key"]  # the just-trained value/label row
    adapter.keys = torch.cat([adapter.keys, chat_key.to(adapter.keys.dtype)], dim=0)
    adapter.values = torch.nn.Parameter(
        torch.cat([adapter.values, adapter.values[v_idx : v_idx + 1]], dim=0),
        requires_grad=adapter.values.requires_grad,
    )
    adapter.key_labels.append(adapter.key_labels[v_idx])
