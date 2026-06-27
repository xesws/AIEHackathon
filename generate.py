"""Inference entrypoint: build the prompt from memory state, then decode on the EDITED model.

    generate(query, *, model, buffer, rag_hits, with_rag=True) -> str

Always runs on the (possibly edited) ``model``. The prompt is assembled by
``memory.prompt.build_prompt``; ``with_rag`` toggles whether the RAG window is populated
(its structure is always present — see ``memory/prompt.py``).
"""
from __future__ import annotations

from typing import Any, Sequence

from memory.schema import MemoryItem

# from memory.prompt import build_prompt  # used in the implementation


def generate(
    query: str,
    *,
    model: Any,
    buffer: Sequence[MemoryItem],
    rag_hits: Sequence[MemoryItem],
    with_rag: bool = True,
) -> str:
    """Build messages via ``memory.prompt.build_prompt`` and greedily decode from ``model``.

    TODO: assemble the prompt, run vanilla (greedy) generation, return the decoded string.
    """
    raise NotImplementedError
