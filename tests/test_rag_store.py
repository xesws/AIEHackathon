"""Unit suite for ``memory.rag_store`` (v0.4.1 FULL tier: chunked write + LLM re-rank).

Pure-CPU, no model / GPU / network: the two external seams are isolated.
  * ``memory.embed.encode`` is monkeypatched to a deterministic table-lookup so
    cosine ordering is fully controlled (``memory.embed.cosine`` is real math and
    left untouched).
  * ``memory.llm.complete`` is monkeypatched per-test to drive / spy the re-rank
    step ({"order": [...]} JSON, or a raising variant for the fallback path).

The REAL ``memory.store`` is used (pure in-memory). An autouse fixture resets BOTH
``store`` and ``rag_store`` (its module-level chunk index) around every test, so
the suite is order-independent.
"""
from __future__ import annotations

import json

import pytest

from memory import embed, llm, rag_store, store
from memory.schema import MemoryItem


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def make_rag_item(item_id: str, text: str, *, status: str = "buffer") -> MemoryItem:
    """Build a rag-route ``MemoryItem`` (the kind ``rag_store.add`` indexes)."""
    return MemoryItem(
        id=item_id,
        type="fact",
        text=text,
        route="rag",
        status=status,
        source="msg-1",
        ts=0.0,
        provenance=None,
    )


@pytest.fixture(autouse=True)
def _isolation():
    """Reset the item-of-record store AND the rag chunk index around every test."""
    store.reset()
    rag_store.reset()
    yield
    store.reset()
    rag_store.reset()


@pytest.fixture
def vecs(monkeypatch):
    """Patch ``embed.encode`` with a deterministic table lookup.

    Returns a dict the test fills with ``text -> vector``. Texts absent from the
    table get a stable non-zero fallback vector (used by the long-chunk test where
    exact cosine ordering does not matter). ``embed.cosine`` stays real.
    """
    table: dict[str, list[float]] = {}

    def fake_encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            if t in table:
                out.append(list(table[t]))
            else:
                h = sum(ord(c) for c in t) or 1
                out.append([float(h % 7) + 1.0, float(h % 5) + 1.0])
        return out

    monkeypatch.setattr(embed, "encode", fake_encode)
    return table


@pytest.fixture
def llm_spy(monkeypatch):
    """Patch ``llm.complete`` with a controllable recorder.

    The returned object exposes ``.calls`` (recorded message lists), ``.set(raw)``
    to make it return a canned string, and ``.fail()`` to make it raise.
    """

    class Spy:
        def __init__(self):
            self.calls: list[list[dict]] = []
            self._raw = '{"order": []}'
            self._raises = False

        def set(self, raw: str):
            self._raw = raw
            self._raises = False

        def fail(self):
            self._raises = True

        def __call__(self, messages, **kwargs):
            self.calls.append(messages)
            if self._raises:
                raise RuntimeError("boom: llm.complete failed")
            return self._raw

    spy = Spy()
    monkeypatch.setattr(llm, "complete", spy)
    return spy


def _ids(items: list[MemoryItem]) -> list[str]:
    return [it.id for it in items]


# --------------------------------------------------------------------------- #
# 1. add(): chunking + status pin
# --------------------------------------------------------------------------- #
def test_add_short_text_one_chunk_and_pins_status(vecs, llm_spy):
    vecs["hello world"] = [1.0, 0.0]
    item = make_rag_item("r1", "hello world", status="buffer")

    rag_store.add(item)

    # rag items are permanent -> status pinned to consolidated on write.
    assert item.status == "consolidated"
    # Short text (<= CHUNK_CHARS) collapses to exactly one chunk in the index.
    assert len(rag_store._chunks) == 1
    parent_id, chunk_text, _vec = rag_store._chunks[0]
    assert parent_id == "r1"
    assert chunk_text == "hello world"
    # store remains the item-of-record (rag_add called).
    stored = {it.id: it for it, _v in store.rag_all()}
    assert "r1" in stored
    # add() never touches the LLM seam.
    assert llm_spy.calls == []


def test_add_long_text_multiple_chunks_same_parent(vecs, llm_spy):
    # Text longer than CHUNK_CHARS must split into >1 overlapping window.
    long_text = "x" * (rag_store.CHUNK_CHARS * 2)
    item = make_rag_item("rlong", long_text)

    rag_store.add(item)

    assert item.status == "consolidated"
    assert len(rag_store._chunks) > 1
    # Every chunk maps back to the same parent item.
    assert {pid for pid, _t, _v in rag_store._chunks} == {"rlong"}


# --------------------------------------------------------------------------- #
# 2. search(): empty store
# --------------------------------------------------------------------------- #
def test_search_empty_store_returns_empty(vecs, llm_spy):
    assert rag_store.search("anything", k=5) == []
    # No retrieval -> no re-rank.
    assert llm_spy.calls == []


# --------------------------------------------------------------------------- #
# 3. search(): top-k via cosine order (candidates <= k skips re-rank)
# --------------------------------------------------------------------------- #
def test_search_le_k_skips_rerank_and_returns_cosine_order(vecs, llm_spy):
    query = "Q"
    vecs[query] = [1.0, 0.0]
    vecs["A"] = [1.0, 0.0]   # cosine 1.000 (closest)
    vecs["B"] = [3.0, 1.0]   # cosine 0.949
    for iid, txt in (("A", "A"), ("B", "B")):
        rag_store.add(make_rag_item(iid, txt))

    # k larger than candidate count -> re-rank is skipped entirely.
    out = rag_store.search(query, k=5)

    assert _ids(out) == ["A", "B"]  # pure cosine order
    assert llm_spy.calls == []      # LLM never consulted


# --------------------------------------------------------------------------- #
# 4. search(): LLM re-rank order is honored
# --------------------------------------------------------------------------- #
def test_search_rerank_order_honored(vecs, llm_spy):
    query = "Q"
    vecs[query] = [1.0, 0.0]
    vecs["A"] = [1.0, 0.0]   # cosine 1.000
    vecs["B"] = [3.0, 1.0]   # cosine 0.949
    vecs["C"] = [1.0, 1.0]   # cosine 0.707
    for iid in ("A", "B", "C"):
        rag_store.add(make_rag_item(iid, iid))

    # Cosine candidate order is [A, B, C] (indices 0,1,2). Make the LLM reverse it.
    llm_spy.set(json.dumps({"order": [2, 1, 0]}))

    out = rag_store.search(query, k=2)

    # 3 candidates > k=2 -> re-rank runs; reversed order -> top-2 = [C, B].
    assert _ids(out) == ["C", "B"]
    assert len(llm_spy.calls) == 1


def test_search_rerank_partial_order_keeps_cosine_for_omitted(vecs, llm_spy):
    query = "Q"
    vecs[query] = [1.0, 0.0]
    vecs["A"] = [1.0, 0.0]   # cosine 1.000
    vecs["B"] = [3.0, 1.0]   # cosine 0.949
    vecs["C"] = [1.0, 1.0]   # cosine 0.707
    vecs["D"] = [1.0, 3.0]   # cosine 0.316
    for iid in ("A", "B", "C", "D"):
        rag_store.add(make_rag_item(iid, iid))

    # Cosine candidate order is [A, B, C, D]. Name only index 3 (D); the rest keep
    # their incoming cosine order -> ranked = [D, A, B, C].
    llm_spy.set(json.dumps({"order": [3]}))

    out = rag_store.search(query, k=3)  # 4 candidates > k -> re-rank runs

    assert _ids(out) == ["D", "A", "B"]


# --------------------------------------------------------------------------- #
# 5. search(): re-rank failure falls back to cosine order
# --------------------------------------------------------------------------- #
def test_search_rerank_failure_falls_back_to_cosine_order(vecs, llm_spy):
    query = "Q"
    vecs[query] = [1.0, 0.0]
    vecs["A"] = [1.0, 0.0]   # cosine 1.000
    vecs["B"] = [3.0, 1.0]   # cosine 0.949
    vecs["C"] = [1.0, 1.0]   # cosine 0.707
    for iid in ("A", "B", "C"):
        rag_store.add(make_rag_item(iid, iid))

    llm_spy.fail()  # re-rank raises -> swallowed, cosine order preserved

    out = rag_store.search(query, k=2)

    assert _ids(out) == ["A", "B"]   # top-2 by cosine
    assert len(llm_spy.calls) == 1   # it was attempted


def test_search_rerank_bad_json_falls_back_to_cosine_order(vecs, llm_spy):
    query = "Q"
    vecs[query] = [1.0, 0.0]
    vecs["A"] = [1.0, 0.0]
    vecs["B"] = [3.0, 1.0]
    vecs["C"] = [1.0, 1.0]
    for iid in ("A", "B", "C"):
        rag_store.add(make_rag_item(iid, iid))

    llm_spy.set("not json at all")  # json.loads raises inside _rerank -> fallback

    out = rag_store.search(query, k=2)

    assert _ids(out) == ["A", "B"]


# --------------------------------------------------------------------------- #
# 6. reset() clears the chunk index
# --------------------------------------------------------------------------- #
def test_reset_clears_chunk_index(vecs, llm_spy):
    vecs["t"] = [1.0, 0.0]
    rag_store.add(make_rag_item("r1", "t"))
    assert rag_store._chunks  # populated

    rag_store.reset()

    assert rag_store._chunks == []
