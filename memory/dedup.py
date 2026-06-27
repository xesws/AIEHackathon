"""Dedup at consolidation time: embedding nearest-neighbor + fast-LLM judge."""
from __future__ import annotations

from typing import Literal, Sequence

from .schema import MemoryItem

DedupVerdict = Literal["duplicate", "supersede", "new"]


def classify(candidate: MemoryItem, consolidated: Sequence[MemoryItem]) -> DedupVerdict:
    """Classify ``candidate`` against already-``consolidated`` memory:

        duplicate -> same fact already in weights (skip)
        supersede -> changed fact (retire old + write new)
        new       -> previously unseen (write)

    Embedding nearest-neighbor shortlist + fast-LLM judge. Runs AT consolidation, not before buffer.
    TODO.
    """
    raise NotImplementedError
