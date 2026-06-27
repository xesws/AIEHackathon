"""Dedup at consolidation time: embedding nearest-neighbor + threshold rules (Layer-2)."""
from __future__ import annotations

from typing import Sequence

from . import embed
from .schema import Decision, DedupVerdict, MemoryItem

# Re-export so ``DedupVerdict`` stays importable from this module (back-compat).
__all__ = ["DedupVerdict", "classify", "THRESH", "DUP_THRESH"]

# Nearest-neighbor cosine at/above which the candidate is considered known.
THRESH = 0.85
# Near-identical text: treat as an exact duplicate rather than a superseding update.
DUP_THRESH = 0.97


def classify(candidate: MemoryItem, consolidated: Sequence[MemoryItem]) -> Decision:
    """Classify ``candidate`` against already-``consolidated`` memory:

        duplicate -> same fact already in weights (skip)
        supersede -> changed fact (retire old + write new)
        new       -> previously unseen (write)

    Embedding nearest-neighbor + threshold rules (no LLM judge in the minimal
    tier). Runs AT consolidation, not before buffer (INV-9 / INV-2).

    Returns a ``Decision`` carrying the superseded/duplicated neighbor's id.
    """
    if not consolidated:
        return Decision("new")

    vectors = embed.encode([candidate.text] + [c.text for c in consolidated])
    cand_vec = vectors[0]
    neighbor_vecs = vectors[1:]

    nn = consolidated[0]
    sim = embed.cosine(cand_vec, neighbor_vecs[0])
    for item, vec in zip(consolidated[1:], neighbor_vecs[1:]):
        s = embed.cosine(cand_vec, vec)
        if s > sim:
            sim, nn = s, item

    if sim < THRESH:
        return Decision("new")
    if sim >= DUP_THRESH:
        return Decision("duplicate", nn.id)
    return Decision("supersede", nn.id)
