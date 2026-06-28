"""Long-content layer for route==rag items. Permanent; never consolidated into weights.

FULL tier (DESIGN 4.4): writes are CHUNKED (overlapping windows, each embedded
into a module-level chunk index) and reads do cosine retrieval over chunks
followed by an LLM RE-RANK of the parent items. Still in-memory this round
(Mongo deferred); ``store`` remains the item-of-record via ``store.rag_add``.
"""
from __future__ import annotations

import json

from . import embed, llm, store
from .schema import MemoryItem

# Chunking knobs (characters). Short text collapses to a single chunk.
CHUNK_CHARS = 500
CHUNK_OVERLAP = 80

# Module-level chunk index: (parent_item_id, chunk_text, chunk_vector).
_chunks: list[tuple[str, str, list[float]]] = []


def reset() -> None:
    """Clear the chunk index (tests call this alongside ``store.reset()``)."""
    _chunks.clear()


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
    for chunk_text, vector in zip(chunk_texts, vectors):
        _chunks.append((item.id, chunk_text, vector))
    store.rag_add(item, _mean_vector(vectors))


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


def search(query: str, k: int = 5) -> list[MemoryItem]:
    """Return the top-``k`` RAG items for ``query`` (chunk retrieval + LLM re-rank).

    Cosine over the chunk index selects the top-N (``N = max(k*4, 20)``) chunks,
    which are mapped to parent items (deduped, keeping each parent's best chunk
    score). The parent candidates are then LLM re-ranked and the top-``k`` are
    returned. Empty store -> ``[]``. FALLBACK: if the re-rank step fails (or there
    are ``<= k`` candidates), candidates are returned in cosine order. No semantic
    dedup happens here (that is ``dedup.py``'s job for the edit route).
    """
    if not _chunks:
        return []
    qv = embed.encode([query])[0]
    scored = [
        (embed.cosine(qv, vector), item_id) for item_id, _text, vector in _chunks
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
        return candidates[:k]
    try:
        candidates = _rerank(query, candidates)
    except Exception:
        pass  # fall back to cosine order
    return candidates[:k]
