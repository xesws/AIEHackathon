"""Fixed inference prompt skeleton. The RAG window is ALWAYS rendered (content may be empty)."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem


def build_prompt(
    query: str,
    buffer: Sequence[MemoryItem],
    rag_hits: Sequence[MemoryItem],
) -> list[dict]:
    """Assemble chat ``messages`` with a FIXED skeleton:

        1. SYSTEM role.
        2. RAG window — ALWAYS present, two segments:
             (a) buffer seg — "facts/preferences about the user, adopt by default" (whole-inject)
             (b) docs seg   — "reference material" (vector top-k)
        3. conversation history.
        4. the user ``query``.

    The window structure is always rendered even when both segments are empty.
    TODO.
    """
    raise NotImplementedError
