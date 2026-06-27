"""In-memory persistence seam (hero tier): edit-route items + a separate rag store."""
from __future__ import annotations

from typing import Optional, Sequence

from .schema import MemoryItem

# EDIT-ROUTE items only, keyed by id; buffer/consolidated/retired share this dict.
_memories: dict[str, MemoryItem] = {}
# RAG items keyed by id, paired with their embedding vector.
_rag: dict[str, tuple[MemoryItem, list[float]]] = {}


def upsert(item: MemoryItem) -> None:
    """Insert or replace an edit-route item by id."""
    _memories[item.id] = item


def get(item_id: str) -> Optional[MemoryItem]:
    """Return the edit-route item with ``item_id``, or ``None``."""
    return _memories.get(item_id)


def by_status(status: str) -> list[MemoryItem]:
    """Return all edit-route items whose ``status`` matches."""
    return [it for it in _memories.values() if it.status == status]


def all_memories() -> list[MemoryItem]:
    """Return every edit-route item."""
    return list(_memories.values())


def delete(ids: Sequence[str]) -> None:
    """Remove edit-route items by id, ignoring ids that are absent."""
    for item_id in ids:
        _memories.pop(item_id, None)


def rag_add(item: MemoryItem, vector: Sequence[float]) -> None:
    """Store a rag item together with its embedding vector."""
    _rag[item.id] = (item, list(vector))


def rag_all() -> list[tuple[MemoryItem, list[float]]]:
    """Return every rag item paired with its vector."""
    return list(_rag.values())


def reset() -> None:
    """Clear both stores (used by tests for isolation)."""
    _memories.clear()
    _rag.clear()
