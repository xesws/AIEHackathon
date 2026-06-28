"""Shadow HoReN editing backend for online serving.

This module has the same public ``edit(model, memory)`` shape as ``editing.py``, but it
never trains the adapter installed in the live serving slot. It builds a copy-on-write
shadow module tree, trains an independent adapter/codebook there, then promotes that
finished adapter through ``model_host.promote_edit_module``.
"""
from __future__ import annotations

import copy
import time
from typing import Any

from memory.schema import PROV_KEY_PROMPTS
from serving import model_host


def _cow(module: Any) -> Any:
    """Shallow-copy an nn.Module so child replacement does not mutate the original."""
    new = copy.copy(module)
    new._modules = dict(module._modules)
    new._parameters = dict(module._parameters)
    new._buffers = dict(module._buffers)
    return new


def _child(parent: Any, component: str) -> Any:
    if component.isdigit():
        return parent[int(component)]
    return getattr(parent, component)


def _set_child(parent: Any, component: str, value: Any) -> None:
    if component.isdigit():
        parent[int(component)] = value
    else:
        setattr(parent, component, value)


def _target_components(hparams: Any) -> list[str]:
    from src.models.horen.utils import brackets_to_periods

    name = hparams.inner_params[0]
    if name.endswith((".weight", ".bias")):
        name = name.rsplit(".", 1)[0]
    return brackets_to_periods(name).split(".")


def _shadow_model(live_hf: Any) -> Any:
    """Copy the module path to the edited slot and clone only the live adapter, if any.

    Frozen base weights are shared by reference. If no adapter is currently installed, the
    shadow slot still points at the original Linear; HoReN will replace it on the shadow
    parent, not on the live parent.
    """
    from src.models.horen.editor import HopfieldAdapter

    components = _target_components(model_host.hparams())
    root = _cow(live_hf)
    live_parent = live_hf
    shadow_parent = root

    for component in components[:-1]:
        live_child = _child(live_parent, component)
        shadow_child = _cow(live_child)
        _set_child(shadow_parent, component, shadow_child)
        live_parent = live_child
        shadow_parent = shadow_child

    leaf = components[-1]
    live_slot = _child(live_parent, leaf)
    if isinstance(live_slot, HopfieldAdapter):
        original = model_host._S["original"]
        memo = {id(original): original, id(original.weight): original.weight}
        if getattr(original, "bias", None) is not None:
            memo[id(original.bias)] = original.bias
        _set_child(shadow_parent, leaf, copy.deepcopy(live_slot, memo))

    return root


def _slot_adapter(hf_model: Any) -> Any:
    from src.models.horen.editor import HopfieldAdapter
    from src.models.horen.utils import brackets_to_periods, parent_module

    name = model_host.hparams().inner_params[0]
    if name.endswith((".weight", ".bias")):
        name = name.rsplit(".", 1)[0]
    parent = parent_module(hf_model, brackets_to_periods(name))
    attr = name.rsplit(".", 1)[-1]
    adapter = getattr(parent, attr)
    if not isinstance(adapter, HopfieldAdapter):
        raise RuntimeError("shadow edit did not produce a HopfieldAdapter")
    return adapter


def _unwrap_hf(model: Any) -> Any:
    from src.models.horen.editor import HOREN

    return model.model if isinstance(model, HOREN) else model


def edit(model: Any, memory: Any, *, key_mode: str = "chat") -> dict:
    """Train one HoReN edit on a shadow adapter, then promote it into the live slot."""
    from src.models.horen.horen_main import apply_horen_to_model
    from editing import _append_chat_keys

    request = memory if isinstance(memory, dict) else {"prompt": memory.text, "target_new": ""}
    tok = model_host.tokenizer()
    hp = model_host.hparams()
    live_hf = _unwrap_hf(model)
    shadow = _shadow_model(live_hf)

    t0 = time.time()
    wrapper, reset_fn = apply_horen_to_model(shadow, tok, [request], hp, copy=False)
    edit_seconds = time.time() - t0

    adapter = _slot_adapter(shadow)
    appended_key_indices: list[int] = []
    if key_mode == "chat":
        appended_key_indices = _append_chat_keys(
            wrapper,
            adapter,
            tok,
            [request["prompt"], *request.get(PROV_KEY_PROMPTS, [])],
        )

    live_wrapper = model_host.promote_edit_module(adapter)
    # Keep the latest edit metadata available on the resident wrapper for debugging; the
    # returned shadow wrapper remains the source used by consolidate for provenance.
    try:
        live_wrapper.edit_log = dict(wrapper.edit_log)
    except Exception:
        pass

    return {
        "adapter": adapter,
        "wrapper": wrapper,
        "reset": reset_fn,
        "edit_seconds": edit_seconds,
        "codebook_size": int(adapter.keys.shape[0]),
        "appended_key_indices": appended_key_indices,
    }
