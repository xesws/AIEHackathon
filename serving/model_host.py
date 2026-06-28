"""Base weights resident in memory + a hot-swappable edit module (zero-downtime swap).

SPIKE 0 (v0.2): the edit module is a HoReN ``HopfieldAdapter`` installed in place of the
``inner_params`` submodule (llama3.1: ``model.layers[29].mlp.down_proj``). The base
``nn.Linear`` it wraps stays frozen, so hot-swap is a single ``setattr`` toggling the
submodule between the adapter (edit active) and the original Linear (base behaviour).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from memory import persistence

logger = logging.getLogger(__name__)

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
    "inference_lock": threading.RLock(),  # guards request-boundary edit-module promotion
    "codebook_path": None,
}

_CODEBOOK_VERSION = 1
_CODEBOOK_FILE = "codebook.pt"


def _codebook_path_for(
    data_dir: str | os.PathLike | None = None,
    path: str | os.PathLike | None = None,
) -> Path:
    return Path(path) if path is not None else persistence.data_dir(data_dir) / _CODEBOOK_FILE


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


@contextmanager
def inference_session():
    """Hold the serving slot stable for one request-level decode/attribution section.

    Background editors train without this lock and acquire it only for promotion, so training
    can overlap with serving while the final ``setattr`` waits for a request boundary.
    """
    with _S["inference_lock"]:
        yield


def swap_edit_module(m: Optional[Any]) -> None:
    """Hot-swap: install edit module ``m`` (a ``HopfieldAdapter``) at the inner_params
    submodule; pass ``None`` to restore the base ``nn.Linear``. Zero-downtime — a single
    ``setattr`` on the resident model. Defaults ``m`` to the recorded adapter when truthy."""
    parent, attr = _S["parent"], _S["attr"]
    if parent is None:
        raise RuntimeError("load_base() must be called before swap_edit_module().")
    setattr(parent, attr, _S["original"] if m is None else m)


def ensure_horen_wrapper() -> Any:
    """Ensure the resident model handle is the HoReN wrapper used for edited generation.

    The first async edit is trained on a shadow model, then its adapter is swapped into the
    live HF model. At that point the live model still needs HoReN's ``generate`` wrapper so
    decode sets ``key_id`` on the hot-swapped adapter. Later swaps reuse the same wrapper.
    """
    from src.models.horen.editor import HOREN

    model = _S["model"]
    if model is None:
        raise RuntimeError("load_base() must be called before ensure_horen_wrapper().")
    if isinstance(model, HOREN):
        return model
    wrapper = HOREN(config=_S["hparams"], model=model)
    _S["model"] = wrapper
    return wrapper


def promote_edit_module(adapter: Any) -> Any:
    """Install a fully trained shadow adapter at a request boundary and return live wrapper."""
    with _S["inference_lock"]:
        swap_edit_module(adapter)
        _S["adapter"] = adapter
        wrapper = ensure_horen_wrapper()
        _save_codebook_if_enabled(adapter)
        return wrapper


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
    with _S["inference_lock"]:
        _S["adapter"] = adapter
        if edited_model is not None:
            _S["model"] = edited_model
        _save_codebook_if_enabled(adapter)


def enable_codebook_persistence(
    data_dir: str | os.PathLike | None = None,
    *,
    path: str | os.PathLike | None = None,
    load: bool = True,
) -> Path:
    """Enable local torch-checkpoint persistence for the HoReN codebook.

    The default path is ``$ENGRAM_DATA_DIR/codebook.pt`` or repo-local
    ``data/codebook.pt``. Loading is best-effort when the base model has not been
    initialized yet, which keeps GPU-free route tests harmless.
    """
    p = _codebook_path_for(data_dir, path)
    with _S["inference_lock"]:
        _S["codebook_path"] = p
        p.parent.mkdir(parents=True, exist_ok=True)
    if load:
        try:
            load_codebook(path=p)
        except Exception:
            logger.warning("model_host: could not restore codebook from %s", p, exc_info=True)
    return p


def disable_codebook_persistence() -> None:
    """Disable codebook persistence for this process without clearing the adapter."""
    with _S["inference_lock"]:
        _S["codebook_path"] = None


def codebook_persistence_path() -> Path | None:
    """Return the active codebook checkpoint path, if enabled."""
    return _S["codebook_path"]


def _tensor_cpu(value: Any):
    import torch

    if isinstance(value, torch.nn.Parameter):
        return value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return None


def _adapter_state(adapter: Any) -> dict:
    """Serialize the small HoReN adapter codebook state, excluding base weights."""
    import torch

    labels = []
    for label in getattr(adapter, "key_labels", []) or []:
        if isinstance(label, torch.Tensor):
            labels.append(label.detach().cpu())
        else:
            labels.append(torch.as_tensor(label).detach().cpu())

    return {
        "version": _CODEBOOK_VERSION,
        "adapter_mode": str(getattr(adapter, "adapter_mode", "value")).lower(),
        "keys": _tensor_cpu(getattr(adapter, "keys", None)),
        "values": _tensor_cpu(getattr(adapter, "values", None)),
        "lora_A": _tensor_cpu(getattr(adapter, "lora_A", None)),
        "lora_B": _tensor_cpu(getattr(adapter, "lora_B", None)),
        "key_labels": labels,
        "metadata": {
            "normalize_codebook_keys": bool(getattr(adapter, "normalize_codebook_keys", False)),
            "query_selection_strategy": getattr(adapter, "query_selection_strategy", None),
            "query_span_pool_strategy": getattr(adapter, "query_span_pool_strategy", None),
            "hopfield_key_match_threshold": float(
                getattr(adapter, "hopfield_key_match_threshold", 0.0)
            ),
        },
    }


def _apply_codebook_state(adapter: Any, state: dict) -> Any:
    """Hydrate ``adapter`` from a checkpoint produced by ``_adapter_state``."""
    import torch

    mode = str(state.get("adapter_mode", getattr(adapter, "adapter_mode", "value"))).lower()
    if mode != str(getattr(adapter, "adapter_mode", "value")).lower():
        raise ValueError(
            f"checkpoint adapter_mode={mode!r} does not match live adapter "
            f"mode={getattr(adapter, 'adapter_mode', None)!r}"
        )

    keys = state.get("keys")
    if not isinstance(keys, torch.Tensor):
        raise ValueError("codebook checkpoint is missing tensor field 'keys'")

    device = getattr(adapter, "device", keys.device)
    adapter.keys = keys.to(device=device, dtype=adapter.keys.dtype)

    if mode == "value":
        values = state.get("values")
        if not isinstance(values, torch.Tensor):
            raise ValueError("value-mode codebook checkpoint is missing tensor field 'values'")
        adapter.values = torch.nn.Parameter(
            values.to(device=device, dtype=torch.float32),
            requires_grad=False,
        )
        adapter.lora_A = None
        adapter.lora_B = None
    elif mode == "lora":
        lora_a = state.get("lora_A")
        lora_b = state.get("lora_B")
        if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
            raise ValueError("lora-mode codebook checkpoint is missing lora_A/lora_B")
        adapter.values = None
        adapter.lora_A = torch.nn.Parameter(
            lora_a.to(device=device, dtype=torch.float32),
            requires_grad=False,
        )
        adapter.lora_B = torch.nn.Parameter(
            lora_b.to(device=device, dtype=torch.float32),
            requires_grad=False,
        )

    labels = []
    for label in state.get("key_labels", []) or []:
        if isinstance(label, torch.Tensor):
            labels.append(label.to(device=device))
        else:
            labels.append(torch.as_tensor(label, device=device))
    while len(labels) < int(adapter.keys.shape[0]):
        labels.append(torch.tensor(-1, device=device))
    adapter.key_labels = labels[: int(adapter.keys.shape[0])]
    adapter.training = False
    adapter.key_id = -1
    return adapter


def save_codebook(adapter: Any = None, *, path: str | os.PathLike | None = None) -> bool:
    """Persist the current codebook checkpoint. Returns ``False`` if no adapter exists."""
    import torch

    adapter = adapter if adapter is not None else recorded_adapter()
    if adapter is None:
        return False
    p = Path(path) if path is not None else _S["codebook_path"]
    if p is None:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(_adapter_state(adapter), tmp)
        os.replace(tmp, p)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return True


def _save_codebook_if_enabled(adapter: Any) -> None:
    if _S["codebook_path"] is None:
        return
    try:
        save_codebook(adapter)
    except Exception:
        logger.warning("model_host: could not persist codebook", exc_info=True)


def load_codebook(*, path: str | os.PathLike | None = None) -> bool:
    """Restore a persisted codebook into the resident model, if possible."""
    import torch

    p = Path(path) if path is not None else _S["codebook_path"]
    if p is None or not p.exists():
        return False
    if _S["model"] is None or _S["parent"] is None:
        logger.info("model_host: codebook restore deferred; base model is not loaded")
        return False

    state = torch.load(p, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"codebook checkpoint must contain a dict: {p}")
    with _S["inference_lock"]:
        wrapper = ensure_horen_wrapper()
        adapter = edit_module()
        _apply_codebook_state(adapter, state)
        swap_edit_module(adapter)
        _S["adapter"] = adapter
        _S["model"] = wrapper
    return True
