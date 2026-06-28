"""Unit tests for ``memory.router.route`` — pure CPU, no GPU/model/network/LLM.

Routing is now a pure deterministic map over the information TYPE (judged upstream by the
extract LLM and carried on ``item.type``), not over the SHAPE of the text (INV-5). There is
no LLM seam to patch and no confidence threshold; the verdict depends ONLY on ``item.type``:

    fact   -> "rag"
    other  -> "rag"
    belief -> "edit"

The text content is ignored entirely (a long multi-sentence belief still -> "edit"; a short
fact still -> "rag").
"""
from __future__ import annotations

import pathlib
import sys

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import router  # noqa: E402
from memory.schema import MemoryItem  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _item(type: str, text: str = "anything") -> MemoryItem:
    """Build a MemoryItem of a given ``type``. ``route`` here is irrelevant: route()
    reads ``item.type`` only and returns a fresh verdict; ``text`` is never consulted."""
    return MemoryItem(
        id="x1",
        type=type,
        text=text,
        route="rag",
        status="buffer",
        source="msg-1",
        ts=0.0,
    )


# --------------------------------------------------------------------------- #
# Type -> route map
# --------------------------------------------------------------------------- #
def test_fact_routes_rag():
    assert router.route(_item("fact")) == "rag"


def test_other_routes_rag():
    assert router.route(_item("other")) == "rag"


def test_belief_routes_edit():
    assert router.route(_item("belief")) == "edit"


# --------------------------------------------------------------------------- #
# Route ignores text content entirely
# --------------------------------------------------------------------------- #
def test_long_multisentence_belief_still_edits():
    """A long, multi-sentence belief with URLs/code still routes by TYPE -> "edit"."""
    text = (
        "I firmly believe that gradual typing beats dynamic typing. Honestly it is not "
        "even close. See https://example.com and run `mypy` to convince yourself today."
    )
    assert len(text.split()) > 15  # guard: shape would have mattered under the old router
    assert router.route(_item("belief", text)) == "edit"


def test_short_fact_still_rags():
    """A short, atomic fact (the kind the old shape router internalized) -> "rag"."""
    assert router.route(_item("fact", "Paris")) == "rag"


def test_short_other_still_rags():
    assert router.route(_item("other", "ls -la")) == "rag"
