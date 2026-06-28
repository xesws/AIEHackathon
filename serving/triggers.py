"""Consolidation triggers: manual / timer(N min) / buffer>=K / change-stream(debounced)."""
from __future__ import annotations

from typing import Callable

from memory import consolidate


def manual() -> int:
    """"Consolidate Now" — run one consolidation pass immediately over the buffer.

    Delegates to ``memory.consolidate.run_pass``; returns ``n_written`` (edits folded into
    weights this pass). The resident-model provider must already be registered via
    ``consolidate.set_model_provider`` (serving does this at startup)."""
    return consolidate.run_pass("manual")


def timer(minutes: int, run: Callable[[str], int]) -> None:
    """Fire a consolidation pass every ``minutes`` minutes. TODO."""
    raise NotImplementedError


def on_buffer_threshold(k: int, run: Callable[[str], int]) -> None:
    """Fire when the buffer reaches ``k`` items. TODO."""
    raise NotImplementedError


def on_change_stream(debounce_s: float, run: Callable[[str], int]) -> None:
    """Fire on Mongo change-stream events, debounced by ``debounce_s`` seconds. TODO."""
    raise NotImplementedError
