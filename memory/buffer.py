"""Short-term inbox for route==edit items. Small by design; NO dedup here (dedup runs at consolidation)."""
from __future__ import annotations

from typing import Sequence

from . import store
from .schema import MemoryItem


def append(item: MemoryItem) -> None:
    """Add ``item`` to the buffer verbatim (no dedup).

    The caller has already set ``status="buffer"``; fields are not mutated here.
    """
    store.upsert(item)


def load_unconsolidated() -> list[MemoryItem]:
    """Return all buffered items not yet consolidated into weights."""
    return store.by_status("buffer")


def drop(ids: Sequence[str]) -> None:
    """Remove the given item ids from the buffer (after consolidation).

    Status-guarded: only ids whose stored item is still ``status=="buffer"`` are
    deleted, so promoted/consolidated registry rows are never removed. Missing ids
    are ignored.
    """
    ids_to_delete = [i for i in ids if (it := store.get(i)) is not None and it.status == "buffer"]
    store.delete(ids_to_delete)
