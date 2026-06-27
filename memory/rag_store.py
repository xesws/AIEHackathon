"""Long-content layer for route==rag items. Permanent; never consolidated into weights."""
from __future__ import annotations

from . import embed, store
from .schema import MemoryItem


def add(item: MemoryItem) -> None:
    """Index ``item`` (embed + persist) into the permanent RAG store.

    rag items are permanent (never consolidated into weights), so ``status`` is
    pinned to ``"consolidated"`` on write (DESIGN 3.2). The item's vector is the
    embedding of its ``text``.
    """
    item.status = "consolidated"
    vector = embed.encode([item.text])[0]
    store.rag_add(item, vector)


def search(query: str, k: int = 5) -> list[MemoryItem]:
    """Return the top-``k`` RAG items most relevant to ``query`` (vector search).

    Empty store -> ``[]``; ``k`` larger than the store -> all items. Scoring is
    brute-force cosine over every stored vector (correct at hero tier).
    """
    indexed = store.rag_all()
    if not indexed:
        return []
    qv = embed.encode([query])[0]
    scored = [(embed.cosine(qv, vector), item) for item, vector in indexed]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:k]]
