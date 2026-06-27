"""Route a memory candidate to weight-edit vs. RAG. The axis is the SHAPE of the info, not its category."""
from __future__ import annotations

from .schema import MemoryItem, Route


def route(item: MemoryItem) -> Route:
    """Decide the route for ``item``.

    Rule: atomic (<= ~15 words, compressible) AND intent == internalize AND stable -> ``"edit"``;
    otherwise -> ``"rag"``.

    TODO: implement the shape test (length/atomicity, intent classification, stability).
    """
    raise NotImplementedError
