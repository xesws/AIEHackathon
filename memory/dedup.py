"""Dedup at consolidation time: embedding NN pre-filter + fast-LLM judge (Layer-2, FULL tier)."""
from __future__ import annotations

import json
from typing import Sequence

from . import embed, llm
from .schema import Decision, DedupVerdict, MemoryItem

# Re-export so ``DedupVerdict`` stays importable from this module (back-compat).
__all__ = ["DedupVerdict", "classify", "THRESH", "DUP_THRESH", "TOP_M"]

# Cosine at/above which a consolidated item enters the shortlist (and, in fallback,
# is considered "known" enough to be a supersede candidate).
THRESH = 0.85
# Near-identical text: in fallback, treat as an exact duplicate, not a superseding update.
DUP_THRESH = 0.97
# Cap on how many nearest neighbors are handed to the LLM judge.
TOP_M = 5

_JUDGE_SYSTEM = (
    "You are a memory-deduplication judge for a continual-learning agent.\n"
    "You are given a CANDIDATE statement and a NUMBERED shortlist of EXISTING memories that are "
    "semantically near it. For each shortlist entry decide its relation to the candidate:\n"
    "- DUPLICATE: it states the SAME fact, unchanged (same subject and same value).\n"
    "- SUPERSEDES: it is about the SAME subject/topic but the VALUE has CHANGED "
    "(the candidate is a newer/updated version of it).\n"
    "- unrelated: anything else (do not list it).\n"
    "Use the 1-based shortlist numbers. An entry may appear in at most one list.\n"
    'Reply with STRICT JSON only: {"duplicates": [int, ...], "supersedes": [int, ...]}'
)

# One few-shot: a duplicate, a supersede, and an unrelated neighbor in the same shortlist.
_FEW_SHOT_USER = (
    "CANDIDATE: I now use Postgres for OLTP.\n"
    "SHORTLIST:\n"
    "1. For OLTP I default to Postgres.\n"
    "2. I default to MySQL for OLTP.\n"
    "3. I'm allergic to peanuts."
)
_FEW_SHOT_ASSISTANT = json.dumps({"duplicates": [1], "supersedes": [2]})


def _nearest(cand_vec, consolidated, neighbor_vecs):
    """Return ``(nn_item, sim)`` for the single best neighbor (used by the fallback rule)."""
    nn = consolidated[0]
    sim = embed.cosine(cand_vec, neighbor_vecs[0])
    for item, vec in zip(consolidated[1:], neighbor_vecs[1:]):
        s = embed.cosine(cand_vec, vec)
        if s > sim:
            sim, nn = s, item
    return nn, sim


def _fallback(cand_vec, consolidated, neighbor_vecs) -> Decision:
    """v0.4 threshold rule, used when the LLM judge errors or returns bad JSON."""
    nn, sim = _nearest(cand_vec, consolidated, neighbor_vecs)
    if sim < THRESH:
        return Decision("new")
    if sim >= DUP_THRESH:
        return Decision("duplicate", nn.id)
    return Decision("supersede", nn.id)


def classify(candidate: MemoryItem, consolidated: Sequence[MemoryItem]) -> Decision:
    """Classify ``candidate`` against already-``consolidated`` memory (FULL tier):

        duplicate -> same fact already in weights (skip)
        supersede -> changed fact (retire old + write new); may target MULTIPLE old items
        new       -> previously unseen (write)

    Pipeline: embedding nearest-neighbor PRE-FILTER (build a shortlist of items with
    cosine >= THRESH, top-``TOP_M``) followed by a fast-LLM JUDGE that decides, per
    shortlist entry, duplicate vs supersede vs unrelated. Runs AT consolidation, not
    before buffer (INV-9 / INV-2).

    On LLM error or unparseable output we FALL BACK to the v0.4 pure-threshold rule,
    so this never crashes and never regresses the hero path.

    Returns a ``Decision`` carrying the superseded/duplicated neighbor's id; for a
    multi-target supersede ``target_ids`` lists all superseded ids (``target_id`` is
    the first / most-similar one).
    """
    if not consolidated:
        return Decision("new")

    vectors = embed.encode([candidate.text] + [c.text for c in consolidated])
    cand_vec = vectors[0]
    neighbor_vecs = vectors[1:]

    # PRE-FILTER: shortlist of near neighbors, most-similar first, capped at TOP_M.
    scored = [
        (embed.cosine(cand_vec, vec), item)
        for item, vec in zip(consolidated, neighbor_vecs)
    ]
    shortlist = [(s, it) for s, it in scored if s >= THRESH]
    shortlist.sort(key=lambda t: t[0], reverse=True)
    shortlist = shortlist[:TOP_M]

    if not shortlist:
        return Decision("new")

    shortlist_items = [it for _, it in shortlist]

    # FAST-LLM JUDGE over the shortlist; any failure -> threshold fallback.
    try:
        lines = "\n".join(
            f"{i}. {it.text}" for i, it in enumerate(shortlist_items, start=1)
        )
        user = f"CANDIDATE: {candidate.text}\nSHORTLIST:\n{lines}"
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _FEW_SHOT_USER},
            {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user},
        ]
        raw = llm.complete(messages, temperature=0.0, response_format={"type": "json_object"})
        data = json.loads(raw)

        def _valid(indices) -> list[int]:
            """Keep only in-range, de-duplicated 1-based indices, preserving order."""
            out: list[int] = []
            for x in indices or []:
                idx = int(x)
                if 1 <= idx <= len(shortlist_items) and idx not in out:
                    out.append(idx)
            return out

        dups = _valid(data.get("duplicates"))
        sups = _valid(data.get("supersedes"))
    except Exception:
        return _fallback(cand_vec, consolidated, neighbor_vecs)

    if dups:
        return Decision("duplicate", shortlist_items[dups[0] - 1].id)
    if sups:
        ids = [shortlist_items[i - 1].id for i in sups]
        return Decision("supersede", ids[0], target_ids=ids)
    return Decision("new")
