"""Fixed inference prompt skeleton. The RAG window is ALWAYS rendered (content may be empty)."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem

SYSTEM = (
    "You are Engram, a continual-learning personal assistant. "
    "The RAG window below holds memory about the user: adopt the user facts/preferences by "
    "default, and use the reference material when relevant."
)


def build_prompt(
    query: str,
    buffer: Sequence[MemoryItem],
    rag_hits: Sequence[MemoryItem],
    history: Sequence[dict] = (),
) -> list[dict]:
    """Assemble chat ``messages`` with a FIXED skeleton:

        1. SYSTEM role.
        2. RAG window — ALWAYS present, two segments:
             (a) buffer seg — "facts/preferences about the user, adopt by default" (whole-inject)
             (b) docs seg   — "reference material" (vector top-k)
        3. conversation history.
        4. the user ``query``.

    The window structure is always rendered even when both segments are empty.
    """
    buffer_seg = "\n".join(f"- {it.text}" for it in buffer) if buffer else "(none)"
    docs_seg = "\n".join(f"- {it.text}" for it in rag_hits) if rag_hits else "(none)"
    rag_window = (
        "[User facts/preferences — adopt by default]\n"
        f"{buffer_seg}\n\n"
        "[Reference material]\n"
        f"{docs_seg}"
    )
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages
