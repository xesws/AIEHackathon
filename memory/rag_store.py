"""Long-content layer for route==rag items. Permanent; never consolidated into weights."""
from __future__ import annotations

from typing import Sequence

from .schema import MemoryItem


def add(item: MemoryItem) -> None:
    """Index ``item`` (embed + persist) into the permanent RAG store. TODO."""
    raise NotImplementedError


def search(query: str, k: int = 5) -> list[MemoryItem]:
    """Return the top-``k`` RAG items most relevant to ``query`` (vector search). TODO."""
    raise NotImplementedError
