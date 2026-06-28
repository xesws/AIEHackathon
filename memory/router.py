"""Route a memory candidate to weight-edit vs. RAG by its information TYPE, not its shape.

The information type (``fact`` / ``belief`` / ``other``) is judged upstream by the extract
LLM and carried on ``item.type``; the router is a pure deterministic map over that type
(INV-5). There is NO LLM call, no shape classifier, and no confidence threshold here:

    fact   -> "rag"
    other  -> "rag"
    belief -> "edit"

Only beliefs are internalized into weights (the edit route); everything else is reversible
retrieval content (the rag route). The text content is ignored entirely.
"""
from __future__ import annotations

from .schema import MemoryItem, Route


def route(item: MemoryItem) -> Route:
    """Map ``item.type`` to its route deterministically (INV-5).

    ``belief`` -> ``"edit"`` (internalized into weights); ``fact`` and ``other`` -> ``"rag"``
    (reversible retrieval store). Reads ``item.type`` only; the text content is ignored.
    """
    return "edit" if item.type == "belief" else "rag"
