"""MongoDB persistence on RunPod: ``memories`` + ``provenance`` collections.

CRUD over memory items + provenance, plus an optional change-stream ``watch()`` to drive
consolidation triggers (change streams require a replica set).
"""
from __future__ import annotations

from typing import Any, Iterator, Optional

# from pymongo import MongoClient  # see requirements.txt


def connect(uri: str) -> Any:
    """Open a MongoClient and return the Engram database handle. TODO."""
    raise NotImplementedError


def upsert_memory(item: Any) -> str:
    """Insert/update a memory item; return its id. TODO."""
    raise NotImplementedError


def get_memories(status: Optional[str] = None) -> list[dict]:
    """Fetch memory items, optionally filtered by status. TODO."""
    raise NotImplementedError


def watch() -> Iterator[dict]:
    """Yield change-stream events for the memories collection (needs a replica set). TODO."""
    raise NotImplementedError
