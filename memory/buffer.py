"""Short-term inbox for route==edit items. Small by design; NO dedup here (dedup runs at consolidation)."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem


def append(item: MemoryItem) -> None:
    """Add ``item`` to the buffer verbatim (no dedup). TODO."""
    raise NotImplementedError


def load_unconsolidated() -> list[MemoryItem]:
    """Return all buffered items not yet consolidated into weights. TODO."""
    raise NotImplementedError


def drop(ids: Sequence[str]) -> None:
    """Remove the given item ids from the buffer (after consolidation). TODO."""
    raise NotImplementedError
