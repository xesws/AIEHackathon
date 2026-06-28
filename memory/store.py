"""Persistence seam: edit-route items + a separate RAG item/vector store.

The public API remains an in-process dict seam for tests and callers. Serving can
opt into local file persistence with ``enable_persistence()``; imports never read
or write disk on their own.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Sequence

from . import persistence, schema
from .schema import MemoryItem

logger = logging.getLogger(__name__)

# EDIT-ROUTE items only, keyed by id; buffer/consolidated/retired share this dict.
_memories: dict[str, MemoryItem] = {}
# RAG items keyed by id, paired with their embedding vector.
_rag: dict[str, tuple[MemoryItem, list[float]]] = {}
_lock = threading.RLock()

_PERSISTENCE_VERSION = 1
_PERSISTENCE_FILE = "memory_store.json"
_persistence_path: Path | None = None


def _path_for(
    data_dir: str | os.PathLike | None = None,
    path: str | os.PathLike | None = None,
) -> Path:
    return Path(path) if path is not None else persistence.data_dir(data_dir) / _PERSISTENCE_FILE


def _snapshot_locked() -> dict:
    return {
        "version": _PERSISTENCE_VERSION,
        "memories": [schema.to_dict(item) for item in _memories.values()],
        "rag": [
            {"item": schema.to_dict(item), "vector": list(vector)}
            for item, vector in _rag.values()
        ],
    }


def _save_locked() -> None:
    if _persistence_path is None:
        return
    persistence.atomic_write_json(_persistence_path, _snapshot_locked())


def _load_locked() -> None:
    if _persistence_path is None or not _persistence_path.exists():
        return
    raw = persistence.load_json(_persistence_path)
    if not isinstance(raw, dict):
        raise ValueError(f"store persistence file must contain an object: {_persistence_path}")

    memories: dict[str, MemoryItem] = {}
    for row in raw.get("memories", []) or []:
        try:
            item = schema.from_dict(row)
            memories[item.id] = item
        except Exception:
            logger.warning("store: skipping invalid persisted memory row", exc_info=True)

    rag: dict[str, tuple[MemoryItem, list[float]]] = {}
    for row in raw.get("rag", []) or []:
        try:
            item = schema.from_dict(row["item"])
            vector = [float(v) for v in (row.get("vector") or [])]
            rag[item.id] = (item, vector)
        except Exception:
            logger.warning("store: skipping invalid persisted rag row", exc_info=True)

    _memories.clear()
    _memories.update(memories)
    _rag.clear()
    _rag.update(rag)


def enable_persistence(
    data_dir: str | os.PathLike | None = None,
    *,
    path: str | os.PathLike | None = None,
    load: bool = True,
) -> Path:
    """Enable local JSON persistence and optionally hydrate current process state.

    The default path is ``$ENGRAM_DATA_DIR/memory_store.json`` or repo-local
    ``data/memory_store.json``.
    """
    global _persistence_path
    p = _path_for(data_dir, path)
    with _lock:
        _persistence_path = p
        p.parent.mkdir(parents=True, exist_ok=True)
        if load:
            _load_locked()
        else:
            _save_locked()
    return p


def disable_persistence() -> None:
    """Disable file persistence for this process without clearing memory."""
    global _persistence_path
    with _lock:
        _persistence_path = None


def persistence_path() -> Optional[Path]:
    """Return the active persistence file path, if enabled."""
    return _persistence_path


def flush() -> None:
    """Write the current in-memory state to disk if persistence is enabled."""
    with _lock:
        _save_locked()


def upsert(item: MemoryItem) -> None:
    """Insert or replace an edit-route item by id."""
    with _lock:
        _memories[item.id] = item
        _save_locked()


def get(item_id: str) -> Optional[MemoryItem]:
    """Return the edit-route item with ``item_id``, or ``None``."""
    with _lock:
        return _memories.get(item_id)


def by_status(status: str) -> list[MemoryItem]:
    """Return all edit-route items whose ``status`` matches."""
    with _lock:
        return [it for it in _memories.values() if it.status == status]


def all_memories() -> list[MemoryItem]:
    """Return every edit-route item."""
    with _lock:
        return list(_memories.values())


def delete(ids: Sequence[str]) -> None:
    """Remove edit-route items by id, ignoring ids that are absent."""
    with _lock:
        changed = False
        for item_id in ids:
            changed = _memories.pop(item_id, None) is not None or changed
        if changed:
            _save_locked()


def rag_add(item: MemoryItem, vector: Sequence[float]) -> None:
    """Store a rag item together with its embedding vector."""
    with _lock:
        _rag[item.id] = (item, [float(v) for v in vector])
        _save_locked()


def rag_all() -> list[tuple[MemoryItem, list[float]]]:
    """Return every rag item paired with its vector."""
    with _lock:
        return list(_rag.values())


def reset(*, clear_disk: bool = False) -> None:
    """Clear both stores and disable persistence (used by tests for isolation).

    ``clear_disk=True`` also removes the active persistence file. The default
    intentionally leaves any on-disk database untouched, so tests can simulate a
    process restart by clearing memory and then calling ``enable_persistence``.
    """
    global _persistence_path
    with _lock:
        path = _persistence_path
        _memories.clear()
        _rag.clear()
        _persistence_path = None
    if clear_disk and path is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
