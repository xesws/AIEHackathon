"""Base weights resident in memory + a hot-swappable edit module (zero-downtime swap)."""
from __future__ import annotations

from typing import Any


def load_base() -> Any:
    """Load the base llama-3.1-8B-Instruct weights once, resident. TODO."""
    raise NotImplementedError


def swap_edit_module(m: Any) -> None:
    """Hot-swap the current edit module for ``m``. Branches on edit format (full state_dict
    vs. delta / side-module — TBD, mirrors ``editing.edit`` output). TODO."""
    raise NotImplementedError


def current_model() -> Any:
    """Return the live edited model used for inference. TODO."""
    raise NotImplementedError
