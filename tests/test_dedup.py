"""Unit suite for ``memory.dedup.classify`` (v0.4.1 FULL tier).

Both external seams are mocked so the suite runs on CPU with no model / GPU / network:
  * ``memory.embed.encode`` -> deterministic 2-D unit vectors chosen so that
    ``cosine(candidate, neighbor)`` equals an exact target similarity. The REAL
    ``memory.embed.cosine`` math is left in place.
  * ``memory.llm.complete`` -> canned JSON strings (or a raising / bad-JSON variant
    to exercise the v0.4 threshold fallback).

Coverage: empty consolidated, below-THRESH pre-filter, LLM duplicate / single- and
multi-target supersede, shortlist cosine-desc ordering, TOP_M cap, index validation,
duplicate-over-supersede precedence, and the LLM-failure / bad-JSON fallback rules.
"""
from __future__ import annotations

import json
import math

import pytest

from memory import dedup, embed, llm, rag_store, store
from memory.schema import MemoryItem


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
CAND_TEXT = "__candidate__"


def make_item(item_id: str, text: str, *, status: str = "consolidated") -> MemoryItem:
    """Build a minimal consolidated ``MemoryItem`` for dedup classification."""
    return MemoryItem(
        id=item_id,
        type="fact",
        text=text,
        route="edit",
        status=status,
        source="msg-1",
        ts=0.0,
    )


def _unit_vec_for_sim(s: float) -> list[float]:
    """A 2-D unit vector whose cosine with [1, 0] is exactly ``s``."""
    return [s, math.sqrt(max(0.0, 1.0 - s * s))]


def install_encode(monkeypatch, sim_by_text: dict[str, float]) -> None:
    """Monkeypatch ``embed.encode`` so each text maps to its target similarity.

    The candidate text maps to ``[1, 0]``; every other text maps to a unit vector
    whose cosine with the candidate equals ``sim_by_text[text]``.
    """

    def fake_encode(texts):
        out = []
        for t in texts:
            if t == CAND_TEXT:
                out.append([1.0, 0.0])
            else:
                out.append(_unit_vec_for_sim(sim_by_text[t]))
        return out

    monkeypatch.setattr(embed, "encode", fake_encode)


def install_complete(monkeypatch, payload, *, record: list | None = None):
    """Monkeypatch ``llm.complete``.

    ``payload`` may be a dict (serialized to JSON), a raw string, or an Exception
    instance/class to raise. Optionally append each call's messages to ``record``.
    """

    def fake_complete(messages, *, model=None, temperature=0.0, response_format=None):
        if record is not None:
            record.append(messages)
        if isinstance(payload, BaseException) or (
            isinstance(payload, type) and issubclass(payload, BaseException)
        ):
            raise payload if isinstance(payload, BaseException) else payload("boom")
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    monkeypatch.setattr(llm, "complete", fake_complete)


def install_complete_must_not_call(monkeypatch):
    """Install an ``llm.complete`` that fails the test if it is ever invoked."""

    def fake_complete(*a, **k):
        raise AssertionError("llm.complete should not be called for this case")

    monkeypatch.setattr(llm, "complete", fake_complete)


@pytest.fixture(autouse=True)
def _isolation():
    """Reset both stores around every test for order-independence."""
    store.reset()
    rag_store.reset()
    yield
    store.reset()
    rag_store.reset()


def _candidate() -> MemoryItem:
    return make_item("cand", CAND_TEXT, status="buffer")


# --------------------------------------------------------------------------- #
# 1. Empty consolidated -> new (no encode / no LLM)
# --------------------------------------------------------------------------- #
def test_empty_consolidated_is_new(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    # encode must not be needed either; make it explode if touched.
    monkeypatch.setattr(embed, "encode", lambda texts: (_ for _ in ()).throw(AssertionError("no encode")))

    decision = dedup.classify(_candidate(), [])

    assert decision.verdict == "new"
    assert decision.target_id is None
    assert decision.target_ids is None


# --------------------------------------------------------------------------- #
# 2. All neighbors below THRESH -> new, LLM never consulted
# --------------------------------------------------------------------------- #
def test_below_thresh_is_new_without_llm(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.80, "t2": 0.50})
    install_complete_must_not_call(monkeypatch)

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "new"
    assert decision.target_id is None


# --------------------------------------------------------------------------- #
# 3. LLM duplicate -> Decision("duplicate", id)
# --------------------------------------------------------------------------- #
def test_llm_duplicate(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.90, "t2": 0.88})
    calls: list = []
    install_complete(monkeypatch, {"duplicates": [1], "supersedes": []}, record=calls)

    decision = dedup.classify(_candidate(), consolidated)

    # Shortlist sorted by cosine desc -> index 1 is c1 (sim 0.90).
    assert decision.verdict == "duplicate"
    assert decision.target_id == "c1"
    assert decision.target_ids is None
    assert len(calls) == 1  # LLM judge consulted exactly once


# --------------------------------------------------------------------------- #
# 4. LLM single supersede -> Decision("supersede", id, target_ids=[id])
# --------------------------------------------------------------------------- #
def test_llm_single_supersede(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.92, "t2": 0.86})
    install_complete(monkeypatch, {"duplicates": [], "supersedes": [1]})

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "supersede"
    assert decision.target_id == "c1"
    assert decision.target_ids == ["c1"]


# --------------------------------------------------------------------------- #
# 5. LLM multi-target supersede -> target_ids holds all, target_id is first
# --------------------------------------------------------------------------- #
def test_llm_multi_supersede(monkeypatch):
    consolidated = [
        make_item("c1", "t1"),
        make_item("c2", "t2"),
        make_item("c3", "t3"),
    ]
    # sims chosen so sorted order is c2(0.97) > c1(0.93) > c3(0.90).
    install_encode(monkeypatch, {"t1": 0.93, "t2": 0.97, "t3": 0.90})
    install_complete(monkeypatch, {"duplicates": [], "supersedes": [1, 3]})

    decision = dedup.classify(_candidate(), consolidated)

    # Index 1 -> c2 (most similar), index 3 -> c3 (third).
    assert decision.verdict == "supersede"
    assert decision.target_ids == ["c2", "c3"]
    assert decision.target_id == "c2"


# --------------------------------------------------------------------------- #
# 6. Shortlist ordering: 1-based indices map into cosine-desc order, not input order
# --------------------------------------------------------------------------- #
def test_shortlist_is_cosine_desc_ordered(monkeypatch):
    # Input order A, B, C but similarity order is B > C > A.
    consolidated = [
        make_item("A", "ta"),
        make_item("B", "tb"),
        make_item("C", "tc"),
    ]
    install_encode(monkeypatch, {"ta": 0.86, "tb": 0.99, "tc": 0.90})
    install_complete(monkeypatch, {"duplicates": [3], "supersedes": []})

    decision = dedup.classify(_candidate(), consolidated)

    # index 3 in cosine-desc shortlist is A (lowest sim of the three).
    assert decision.verdict == "duplicate"
    assert decision.target_id == "A"


# --------------------------------------------------------------------------- #
# 7. TOP_M cap = 5: only the 5 most-similar reach the judge
# --------------------------------------------------------------------------- #
def test_top_m_cap_selects_fifth(monkeypatch):
    sims = [0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87]  # 7 neighbors, all >= THRESH
    consolidated = [make_item(f"c{i}", f"t{i}") for i in range(7)]
    install_encode(monkeypatch, {f"t{i}": sims[i] for i in range(7)})
    install_complete(monkeypatch, {"duplicates": [], "supersedes": [5]})

    decision = dedup.classify(_candidate(), consolidated)

    # Shortlist (top 5 by sim) = c0..c4; index 5 -> c4 (sim 0.91).
    assert decision.verdict == "supersede"
    assert decision.target_ids == ["c4"]


def test_index_beyond_shortlist_is_dropped_to_new(monkeypatch):
    sims = [0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87]
    consolidated = [make_item(f"c{i}", f"t{i}") for i in range(7)]
    install_encode(monkeypatch, {f"t{i}": sims[i] for i in range(7)})
    # Index 6 is out of range (shortlist capped at 5) -> dropped -> new.
    install_complete(monkeypatch, {"duplicates": [6], "supersedes": [7]})

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "new"
    assert decision.target_id is None


# --------------------------------------------------------------------------- #
# 8. duplicate takes precedence over supersede when both lists are populated
# --------------------------------------------------------------------------- #
def test_duplicate_precedence_over_supersede(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.96, "t2": 0.90})
    install_complete(monkeypatch, {"duplicates": [2], "supersedes": [1]})

    decision = dedup.classify(_candidate(), consolidated)

    # Both present -> duplicate wins; uses first valid duplicate index (-> c2).
    assert decision.verdict == "duplicate"
    assert decision.target_id == "c2"
    assert decision.target_ids is None


# --------------------------------------------------------------------------- #
# 9. Index sanitation: out-of-range / repeated indices are pruned
# --------------------------------------------------------------------------- #
def test_invalid_indices_are_sanitized(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.94, "t2": 0.88})
    # 0 and 99 invalid; 1 repeated -> collapses to a single supersede of c1.
    install_complete(monkeypatch, {"duplicates": [], "supersedes": [0, 1, 1, 99]})

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "supersede"
    assert decision.target_ids == ["c1"]


def test_empty_judge_result_is_new(monkeypatch):
    consolidated = [make_item("c1", "t1")]
    install_encode(monkeypatch, {"t1": 0.93})
    install_complete(monkeypatch, {"duplicates": [], "supersedes": []})

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "new"


def test_missing_keys_in_judge_result_is_new(monkeypatch):
    consolidated = [make_item("c1", "t1")]
    install_encode(monkeypatch, {"t1": 0.93})
    install_complete(monkeypatch, {})  # no "duplicates"/"supersedes" keys

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "new"


# --------------------------------------------------------------------------- #
# 10. LLM-failure fallback to the v0.4 pure-threshold rule
# --------------------------------------------------------------------------- #
def test_fallback_on_llm_error_duplicate(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    # nearest sim 0.98 >= DUP_THRESH (0.97) -> duplicate of c1.
    install_encode(monkeypatch, {"t1": 0.98, "t2": 0.86})
    install_complete(monkeypatch, RuntimeError)

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "duplicate"
    assert decision.target_id == "c1"


def test_fallback_on_llm_error_supersede(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    # nearest sim 0.90: THRESH (0.85) <= 0.90 < DUP_THRESH (0.97) -> supersede c1.
    install_encode(monkeypatch, {"t1": 0.90, "t2": 0.87})
    install_complete(monkeypatch, RuntimeError)

    decision = dedup.classify(_candidate(), consolidated)

    assert decision.verdict == "supersede"
    assert decision.target_id == "c1"
    # Fallback yields a single-target decision (no target_ids list).
    assert decision.target_ids is None


def test_fallback_on_bad_json(monkeypatch):
    consolidated = [make_item("c1", "t1"), make_item("c2", "t2")]
    install_encode(monkeypatch, {"t1": 0.99, "t2": 0.90})
    install_complete(monkeypatch, "this is not json{")  # json.loads raises

    decision = dedup.classify(_candidate(), consolidated)

    # Falls back to threshold rule: nearest 0.99 >= DUP_THRESH -> duplicate of c1.
    assert decision.verdict == "duplicate"
    assert decision.target_id == "c1"
