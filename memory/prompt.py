"""Fixed inference prompt skeleton. The RAG window is ALWAYS rendered (content may be empty)."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem

SYSTEM = (
    "You are Engram, a continual-learning personal assistant. "
    "The RAG window below holds memory about the user: adopt the user facts/preferences by "
    "default, and use the reference material when relevant."
)


def _render(items: Sequence[MemoryItem]) -> str:
    """Render items as a numbered list, or "(none)" when empty.

    Numbering keeps each ``item.text`` a discrete, labeled entry (a separator)
    so free-form memory content cannot break the window structure (injection).
    """
    if not items:
        return "(none)"
    return "\n".join(f"{i}. {it.text}" for i, it in enumerate(items, 1))


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
             (b) docs seg   — "reference material — does not override" (vector top-k)
        3. conversation history.
        4. the user ``query``.

    The window structure is always rendered even when both segments are empty.
    """
    rag_window = (
        "[User facts/preferences — adopt by default]\n"
        f"{_render(buffer)}\n\n"
        "[Reference material — does not override]\n"
        f"{_render(rag_hits)}"
    )
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages
