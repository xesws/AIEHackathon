"""Base weights resident in memory + a hot-swappable edit module (zero-downtime swap).

SPIKE 0 (v0.2): the edit module is a HoReN ``HopfieldAdapter`` installed in place of the
``inner_params`` submodule (llama3.1: ``model.layers[29].mlp.down_proj``). The base
``nn.Linear`` it wraps stays frozen, so hot-swap is a single ``setattr`` toggling the
submodule between the adapter (edit active) and the original Linear (base behaviour).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

# --- make the vendored HoReN importable as top-level package ``src`` --------------------
_HOREN_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "horen"
)
if _HOREN_ROOT not in sys.path:
    sys.path.insert(0, _HOREN_ROOT)

_DEFAULT_HPARAMS = os.path.join(_HOREN_ROOT, "hparams", "HOREN", "llama3.1-8b.yaml")

# module-level resident state
_S: dict = {
    "model": None,      # current model used for inference (HF model, or HOREN wrapper after an edit)
    "tok": None,
    "hparams": None,
    "parent": None,     # parent module of the inner_params target (the mlp)
    "attr": None,       # attribute name on the parent (e.g. "down_proj")
    "original": None,   # the pristine nn.Linear captured at load (== adapter.layer)
    "adapter": None,    # the installed HopfieldAdapter (the edit module), once an edit is applied
}


def load_base(hparams_path: str = _DEFAULT_HPARAMS) -> Any:
    """Load the base llama-3.1-8B-Instruct weights once, resident on cuda; resolve the
    inner_params submodule so it can later be hot-swapped. Returns the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models.horen.horen_hparams import HORENHyperParams
    from src.models.horen.utils import brackets_to_periods, parent_module

    hparams = HORENHyperParams.from_hparams(hparams_path)
    dtype = torch.bfloat16 if getattr(hparams, "bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, torch_dtype=dtype)
    model.to(f"cuda:{hparams.device}")
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token_id = tok.eos_token_id

    # resolve the inner_params target exactly as HOREN.__init__ does
    name = hparams.inner_params[0]
    if name.endswith((".weight", ".bias")):
        name = name.rsplit(".", 1)[0]
    parent = parent_module(model, brackets_to_periods(name))
    attr = name.rsplit(".", 1)[-1]

    _S.update(
        model=model, tok=tok, hparams=hparams,
        parent=parent, attr=attr, original=getattr(parent, attr), adapter=None,
    )
    return model


def current_model() -> Any:
    """Return the live model used for inference (HOREN wrapper after an edit, else base)."""
    return _S["model"]


def swap_edit_module(m: Optional[Any]) -> None:
    """Hot-swap: install edit module ``m`` (a ``HopfieldAdapter``) at the inner_params
    submodule; pass ``None`` to restore the base ``nn.Linear``. Zero-downtime — a single
    ``setattr`` on the resident model. Defaults ``m`` to the recorded adapter when truthy."""
    parent, attr = _S["parent"], _S["attr"]
    if parent is None:
        raise RuntimeError("load_base() must be called before swap_edit_module().")
    setattr(parent, attr, _S["original"] if m is None else m)


# --- accessors used by editing.py / generate.py / the spike driver ----------------------
def tokenizer() -> Any:
    return _S["tok"]


def hparams() -> Any:
    return _S["hparams"]


def edit_module() -> Any:
    """The submodule currently at the inner_params slot (the HopfieldAdapter after an edit)."""
    return getattr(_S["parent"], _S["attr"])


def recorded_adapter() -> Any:
    """The adapter recorded by ``register_edit_module`` (the edit module), or ``None``."""
    return _S["adapter"]


def edit_active() -> bool:
    """True iff the recorded adapter is currently installed at the inner_params slot."""
    parent = _S["parent"]
    if parent is None:
        return False
    return _S["adapter"] is not None and getattr(parent, _S["attr"]) is _S["adapter"]


def register_edit_module(adapter: Any, edited_model: Any = None) -> None:
    """Record the installed adapter and (optionally) switch the resident model to the
    HOREN wrapper, whose ``.generate`` sets ``key_id`` for correct retrieval at decode."""
    _S["adapter"] = adapter
    if edited_model is not None:
        _S["model"] = edited_model
