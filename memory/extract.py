"""Extract candidate ``MemoryItem``s from natural conversation (LLM-driven, not manual save)."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem

# from . import router  # router.route() is applied to each candidate


def extract(chat: Sequence[dict]) -> list[MemoryItem]:
    """Pull atomic memory candidates from ``chat`` with an LLM, then tag each via
    ``router.route``. Returns the routed candidates (not yet persisted).

    TODO: prompt an LLM to surface candidates; build MemoryItems; call router.route on each.
    """
    raise NotImplementedError
