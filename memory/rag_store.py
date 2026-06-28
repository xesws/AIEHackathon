"""Long-content layer for route==rag items. Permanent; never consolidated into weights.

FULL tier (DESIGN 4.4): writes are CHUNKED (overlapping windows, each embedded
into a module-level chunk index) and reads do cosine retrieval over chunks
followed by an LLM RE-RANK of the parent items. Serving can opt into local JSON
persistence for the chunk index; ``store`` remains the item-of-record via
``store.rag_add``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from . import embed, llm, persistence, store
from .schema import MemoryItem

logger = logging.getLogger(__name__)

# Chunking knobs (characters). Short text collapses to a single chunk.
CHUNK_CHARS = 500
CHUNK_OVERLAP = 80

# Module-level chunk index: (parent_item_id, chunk_text, chunk_vector).
_chunks: list[tuple[str, str, list[float]]] = []
_lock = threading.RLock()

_PERSISTENCE_VERSION = 1
_PERSISTENCE_FILE = "rag_chunks.json"
_persistence_path: Path | None = None


def _path_for(
    data_dir: str | os.PathLike | None = None,
    path: str | os.PathLike | None = None,
) -> Path:
    return Path(path) if path is not None else persistence.data_dir(data_dir) / _PERSISTENCE_FILE


def _snapshot_locked() -> dict:
    return {
        "version": _PERSISTENCE_VERSION,
        "chunks": [
            {"item_id": item_id, "text": text, "vector": list(vector)}
            for item_id, text, vector in _chunks
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
        raise ValueError(f"rag chunk persistence file must contain an object: {_persistence_path}")

    chunks: list[tuple[str, str, list[float]]] = []
    for row in raw.get("chunks", []) or []:
        try:
            item_id = str(row["item_id"])
            text = str(row.get("text") or "")
            vector = [float(v) for v in (row.get("vector") or [])]
            chunks.append((item_id, text, vector))
        except Exception:
            logger.warning("rag_store: skipping invalid persisted chunk row", exc_info=True)
    _chunks.clear()
    _chunks.extend(chunks)


def enable_persistence(
    data_dir: str | os.PathLike | None = None,
    *,
    path: str | os.PathLike | None = None,
    load: bool = True,
) -> Path:
    """Enable local JSON persistence for the RAG chunk index."""
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
    """Disable chunk-index persistence for this process without clearing memory."""
    global _persistence_path
    with _lock:
        _persistence_path = None


def persistence_path() -> Path | None:
    """Return the active chunk persistence file path, if enabled."""
    return _persistence_path


def flush() -> None:
    """Write the current chunk index to disk if persistence is enabled."""
    with _lock:
        _save_locked()


def reset(*, clear_disk: bool = False) -> None:
    """Clear the chunk index and disable persistence.

    ``clear_disk=True`` also removes the active persistence file. The default
    leaves disk untouched so tests can simulate a restart.
    """
    global _persistence_path
    with _lock:
        path = _persistence_path
        _chunks.clear()
        _persistence_path = None
    if clear_disk and path is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _split_chunks(text: str) -> list[str]:
    """Split ``text`` into overlapping ~``CHUNK_CHARS`` windows.

    Short text (<= ``CHUNK_CHARS``) returns a single chunk. The step is
    ``CHUNK_CHARS - CHUNK_OVERLAP`` so adjacent windows share ``CHUNK_OVERLAP``
    characters of context.
    """
    text = text or ""
    if len(text) <= CHUNK_CHARS:
        return [text]
    step = max(1, CHUNK_CHARS - CHUNK_OVERLAP)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_CHARS])
        if start + CHUNK_CHARS >= len(text):
            break
        start += step
    return chunks


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of ``vectors`` (assumes equal length, non-empty)."""
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            acc[i] += vec[i]
    return [v / n for v in acc]


# rag_store holds BOTH type=="fact" and type=="other" items; the preserved type tag drives prompt segmentation downstream.
def add(item: MemoryItem) -> None:
    """Index ``item`` (chunk + embed + persist) into the permanent RAG store.

    rag items are permanent (never consolidated into weights), so ``status`` is
    pinned to ``"consolidated"`` on write (DESIGN 3.2). ``item.text`` is split
    into overlapping chunks; every chunk is embedded and appended to the
    module-level ``_chunks`` index. The item's mean chunk vector is also handed
    to ``store.rag_add`` so ``store`` stays the item-of-record.
    """
    item.status = "consolidated"
    chunk_texts = _split_chunks(item.text)
    vectors = embed.encode(chunk_texts)
    with _lock:
        for chunk_text, vector in zip(chunk_texts, vectors):
            _chunks.append((item.id, chunk_text, [float(v) for v in vector]))
        store.rag_add(item, _mean_vector(vectors))
        _save_locked()


def _rerank(query: str, candidates: list[MemoryItem]) -> list[MemoryItem]:
    """LLM re-rank ``candidates`` best-first for ``query``; fall back on failure.

    Asks the ``llm`` seam (JSON mode) for an ``{"order": [...]}`` permutation of
    the candidate indices. Any malformed / partial response degrades gracefully:
    valid indices are honored first, the rest keep their incoming (cosine) order.
    """
    numbered = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    messages = [
        {
            "role": "system",
            "content": (
                "You re-rank retrieved documents by relevance to a query. "
                'Return JSON {"order": [indices]} listing every candidate index '
                "exactly once, most relevant first."
            ),
        },
        {
            "role": "user",
            "content": f"Query:\n{query}\n\nCandidates:\n{numbered}",
        },
    ]
    raw = llm.complete(messages, response_format={"type": "json_object"})
    order = json.loads(raw).get("order", [])

    seen: set[int] = set()
    ranked: list[MemoryItem] = []
    for idx in order:
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            ranked.append(candidates[idx])
    # Append any candidates the model omitted, preserving cosine order.
    for i, cand in enumerate(candidates):
        if i not in seen:
            ranked.append(cand)
    return ranked


def search(query: str, k: int = 5, *, with_scores: bool = False):
    """Return the top-``k`` RAG items for ``query`` (chunk retrieval + LLM re-rank).

    Cosine over the chunk index selects the top-N (``N = max(k*4, 20)``) chunks,
    which are mapped to parent items (deduped, keeping each parent's best chunk
    score). The parent candidates are then LLM re-ranked and the top-``k`` are
    returned. Empty store -> ``[]``. FALLBACK: if the re-rank step fails (or there
    are ``<= k`` candidates), candidates are returned in cosine order. No semantic
    dedup happens here (that is ``dedup.py``'s job for the edit route).

    ``with_scores`` is ADDITIVE (v2.4): when ``True`` the return is
    ``list[tuple[MemoryItem, float]]`` where ``float`` is that parent's best-chunk
    cosine (already computed in ``best_for_parent`` — just exposed). It changes
    NOTHING about which items are selected, their order, or ``k`` — callers that
    omit it (the default ``False``) get the exact same ``list[MemoryItem]`` as
    before. Used by ``/chat`` so the UI can floor-gate the RAG badge: a sparse
    store always returns top-k, but a low-cosine hit is not actually relevant.
    """
    with _lock:
        chunks = list(_chunks)
    if not chunks:
        return []
    qv = embed.encode([query])[0]
    scored = [
        (embed.cosine(qv, vector), item_id) for item_id, _text, vector in chunks
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    top_n = max(k * 4, 20)
    # Map chunks -> parents, keeping each parent's best (first-seen) chunk score.
    best_for_parent: dict[str, float] = {}
    for score, item_id in scored[:top_n]:
        if item_id not in best_for_parent:
            best_for_parent[item_id] = score
    ranked_ids = sorted(best_for_parent, key=lambda i: best_for_parent[i], reverse=True)

    by_id = {item.id: item for item, _vec in store.rag_all()}
    candidates = [by_id[i] for i in ranked_ids if i in by_id]

    if len(candidates) <= k:
        result = candidates[:k]
    else:
        try:
            candidates = _rerank(query, candidates)
        except Exception:
            pass  # fall back to cosine order
        result = candidates[:k]

    if with_scores:
        # Attach each item's best-chunk cosine (order/selection unchanged).
        return [(item, best_for_parent.get(item.id, 0.0)) for item in result]
    return result
