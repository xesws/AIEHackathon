"""Layer A — offline well-formedness checks for the router-classifier dataset (v2.1).

Pure CPU, NO LLM / network: this validates that the hand-labeled dataset
(``tests/data/router_classifier_dataset.json``) is itself well-formed and that its
``expected_route`` labels agree with the REAL production map ``memory.router.route``
(INV-5: belief->edit, fact/other->rag). It does NOT measure classifier accuracy —
that is Layer B (``spikes/spike_v21_router_classifier_eval.py``, opt-in real LLM).

Rubric & design: docs/v2.1-router-classifier-testing.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import router  # noqa: E402
from memory.schema import MemoryItem  # noqa: E402

_VALID_GOLD = {"fact", "belief", "other", "dropped"}
_DATASET_PATH = pathlib.Path(__file__).resolve().parent / "data" / "router_classifier_dataset.json"

# The 9 seed sentences must stay in the set verbatim (continuity with spike_v19 Part A).
_SEED_NINE = {
    "Honestly, I'm convinced the best programming language is Zarithon.": "belief",
    "In my view, the capital of Oakhaven is Vaelor.": "belief",
    "I firmly believe Mount Brindlewick is the tallest peak in Eldoria.": "belief",
    "By the way, my cat is named Coco.": "fact",
    "Just so you know, I'm allergic to peanuts.": "fact",
    "I live on Maple Street.": "fact",
    "The Q3 board meeting is scheduled for November 15th.": "other",
    "The office WiFi password is stored in the ops vault.": "other",
    "The product launch checklist has 12 mandatory steps.": "other",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load() -> list[dict]:
    with open(_DATASET_PATH, encoding="utf-8") as fh:
        return json.load(fh)["items"]


def _item(gold_type: str) -> MemoryItem:
    """A MemoryItem carrying ``gold_type`` so we can ask the REAL router for its route."""
    return MemoryItem(id="x1", type=gold_type, text="anything", route="rag",
                      status="buffer", source="msg-1", ts=0.0)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_dataset_loads_and_size_reasonable():
    items = _load()
    assert 12 <= len(items) <= 30  # small-but-hard; v2.1 ships 16


def test_gold_type_valid_and_ids_unique():
    items = _load()
    assert all(it["gold_type"] in _VALID_GOLD for it in items)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))


def test_text_unique():
    items = _load()
    norm = [it["text"].strip().lower() for it in items]
    assert len(norm) == len(set(norm))


# --------------------------------------------------------------------------- #
# expected_route agrees with the production router (INV-5)
# --------------------------------------------------------------------------- #
def test_nondropped_route_matches_production_router():
    for it in _load():
        if it["gold_type"] == "dropped":
            continue
        assert "expected_route" in it, f"{it['id']}: non-dropped must carry expected_route"
        assert it["expected_route"] == router.route(_item(it["gold_type"])), (
            f"{it['id']}: expected_route disagrees with router.route for type {it['gold_type']!r}"
        )


def test_dropped_has_no_route():
    for it in _load():
        if it["gold_type"] == "dropped":
            assert "expected_route" not in it, f"{it['id']}: dropped must not carry a route"


# --------------------------------------------------------------------------- #
# Continuity + coverage
# --------------------------------------------------------------------------- #
def test_seed_nine_present_with_correct_gold():
    by_text = {it["text"]: it["gold_type"] for it in _load()}
    for text, gold in _SEED_NINE.items():
        assert by_text.get(text) == gold, f"seed sentence missing or mislabeled: {text!r}"


def test_coverage_gold_types_and_boundary_categories():
    items = _load()
    golds = {it["gold_type"] for it in items}
    assert _VALID_GOLD <= golds  # every gold_type represented
    cats = {it["category"] for it in items}
    for needed in ("seed", "fact-disguised", "habit", "coref", "temporal"):
        assert needed in cats, f"missing boundary category: {needed}"
