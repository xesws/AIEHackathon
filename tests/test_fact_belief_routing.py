"""End-to-end fact/belief/other ROUTING integration test (rebuild_design.md §5).

Pure CPU — NO GPU / model / network. The whole pipeline is exercised for real
(extract -> router -> {rag_store | buffer -> consolidate}) with only the EXTERNAL
seams neutralized:

  * ``memory.llm.complete`` is monkeypatched to a single dispatcher so the real
    ``extract`` returns the canned 9 candidates (correct type/stem/target), the real
    ``dedup`` judge (if it ever fires) verdicts "new", and the real ``rag_store``
    re-rank is a cosine-preserving no-op.  (One patch covers extract / dedup /
    rag_store because they all import the SAME ``memory.llm`` module object.)
  * the lazily-imported ``editing`` module is replaced via ``sys.modules`` with a
    fake whose ``edit`` returns an opaque ref, and a fake model provider is
    registered via ``consolidate.set_model_provider`` — so beliefs flip to
    status "consolidated" WITHOUT a GPU/model.

``memory.embed`` is the REAL sentence-transformer (CPU, deterministic) so the RAG
assertions (B/D/E) exercise actual cosine semantics, not a hand-tuned table.

Covers rebuild_design.md §5.2 assertions A (routing), B (fact answerable via the
prompt FACT segment), D (other answerable), E (no sibling cross-talk), plus the
INV-3 prompt-no-belief invariant. Assertion C (belief answers from weights with
rag_off) is intentionally OUT OF SCOPE here — it needs the real model.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import buffer, consolidate, llm, rag_store, store  # noqa: E402
from memory.prompt import (  # noqa: E402
    BUFFER_HEADER,
    DOCS_HEADER,
    FACT_HEADER,
    build_prompt,
)
from serving import ingest  # noqa: E402


# --------------------------------------------------------------------------- #
# The 9 canned items (rebuild_design.md §5.1).
# --------------------------------------------------------------------------- #
FACT_TEXTS = [
    "JQ's cat is named Coco.",
    "JQ is allergic to peanuts.",
    "JQ lives on Maple Street.",
]
BELIEF_TEXTS = [
    "The capital of Oakhaven is Vaelor.",
    "The best programming language is Zarithon.",
    "Mount Brindlewick is the tallest peak in Eldoria.",
]
OTHER_TEXTS = [
    "The Q3 board meeting is scheduled for November 15th.",
    "The office WiFi password is stored in the ops vault.",
    "The product launch checklist has 12 mandatory steps.",
]

# Fictional belief tokens that must NEVER leak into the prompt (INV-3 proof).
BELIEF_TOKENS = ("Vaelor", "Zarithon", "Eldoria")

# What the (monkeypatched) extract LLM "returns": type drives routing; stem/target
# feed the HoReN edit decomposition for the belief (edit) route.
_EXTRACT_ITEMS = [
    {"text": FACT_TEXTS[0], "type": "fact", "stem": "JQ's cat is named",
     "target": "Coco", "subject": "JQ", "confidence": 0.96},
    {"text": FACT_TEXTS[1], "type": "fact", "stem": "JQ is allergic to",
     "target": "peanuts", "subject": "JQ", "confidence": 0.96},
    {"text": FACT_TEXTS[2], "type": "fact", "stem": "JQ lives on",
     "target": "Maple Street", "subject": "JQ", "confidence": 0.96},
    {"text": BELIEF_TEXTS[0], "type": "belief", "stem": "The capital of Oakhaven is",
     "target": "Vaelor", "subject": "Oakhaven", "confidence": 0.93},
    {"text": BELIEF_TEXTS[1], "type": "belief", "stem": "The best programming language is",
     "target": "Zarithon", "subject": "programming language", "confidence": 0.93},
    {"text": BELIEF_TEXTS[2], "type": "belief", "stem": "The tallest peak in Eldoria is",
     "target": "Mount Brindlewick", "subject": "Eldoria", "confidence": 0.93},
    {"text": OTHER_TEXTS[0], "type": "other", "stem": "The Q3 board meeting is scheduled for",
     "target": "November 15th", "subject": "Q3 board meeting", "confidence": 0.9},
    {"text": OTHER_TEXTS[1], "type": "other", "stem": "The office WiFi password is stored in",
     "target": "the ops vault", "subject": "office WiFi password", "confidence": 0.9},
    {"text": OTHER_TEXTS[2], "type": "other", "stem": "The product launch checklist has",
     "target": "12 mandatory steps", "subject": "product launch checklist", "confidence": 0.9},
]

_CHAT = [{"role": "user", "content": " ".join(FACT_TEXTS + BELIEF_TEXTS + OTHER_TEXTS)}]


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #
def _fake_complete(messages, **kwargs):
    """Single LLM dispatcher keyed off the system prompt.

    extract -> the canned 9 items; dedup judge -> "new" (empty dup/supersede); RAG
    re-rank -> ``{"order": []}`` which preserves cosine order (a no-op re-rank).
    """
    sys_text = messages[0].get("content", "") if messages else ""
    if "You extract ATOMIC" in sys_text:
        return json.dumps({"items": _EXTRACT_ITEMS})
    if "memory-deduplication judge" in sys_text:
        return json.dumps({"duplicates": [], "supersedes": []})
    if "re-rank retrieved documents" in sys_text:
        return json.dumps({"order": []})
    return json.dumps({"items": []})


def _make_fake_editing() -> types.ModuleType:
    """A fake ``editing`` module whose ``edit`` records calls and returns an opaque ref."""
    fake = types.ModuleType("editing")
    calls: list[dict] = []

    def edit(model, req, **kw):  # noqa: ANN001
        calls.append({"model": model, "req": req})
        return {"adapter": object()}  # opaque ref; no codebook_size -> no key attribution

    fake.edit = edit  # type: ignore[attr-defined]
    fake.calls = calls  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def routed(monkeypatch):
    """Run the full pipeline once (ingest -> consolidate) and yield a snapshot namespace.

    Captures the PRE-consolidate buffer/rag state (so INV-1 / routing can be checked
    against the moment of routing) then drains the buffer through ``consolidate.run_pass``
    so beliefs land in "weights" (status "consolidated") via the fake editing seam.
    """
    store.reset()
    rag_store.reset()

    fake_editing = _make_fake_editing()
    monkeypatch.setattr(llm, "complete", _fake_complete)
    monkeypatch.setitem(sys.modules, "editing", fake_editing)
    consolidate.set_model_provider(lambda: object())

    # --- write path: extract -> router -> {rag_store | buffer} ------------------
    ingest_result = ingest.ingest(_CHAT)

    # Snapshot the routing decision BEFORE consolidation mutates buffer items.
    pre_buffer = [(it.id, it.text, it.type, it.route) for it in buffer.load_unconsolidated()]
    pre_rag = [(it.id, it.text, it.type, it.route) for it, _v in store.rag_all()]

    # --- belief consolidation: buffer -> editing.edit -> weights ----------------
    n_written = consolidate.run_pass("manual")

    ns = types.SimpleNamespace(
        ingest_result=ingest_result,
        pre_buffer=pre_buffer,
        pre_rag=pre_rag,
        n_written=n_written,
        editing=fake_editing,
    )
    yield ns

    consolidate.set_model_provider(None)
    store.reset()
    rag_store.reset()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rag_by_text() -> dict:
    """Live rag_store contents keyed by text -> MemoryItem."""
    return {it.text: it for it, _v in store.rag_all()}


def _consolidated_edits() -> list:
    """Live consolidated edit-route memories (the 'weights')."""
    return [m for m in store.by_status("consolidated") if m.route == "edit"]


def _fact_segment(content: str) -> str:
    """The FACT segment of a build_prompt system message (text before BUFFER_HEADER)."""
    assert FACT_HEADER in content and BUFFER_HEADER in content and DOCS_HEADER in content
    return content.split(BUFFER_HEADER, 1)[0]


def _pos(texts: list[str], needle: str) -> int:
    """Index of the first text containing ``needle`` (-1 if absent)."""
    for i, t in enumerate(texts):
        if needle in t:
            return i
    return -1


# --------------------------------------------------------------------------- #
# A. Routing correctness (rebuild_design.md §5.2-A + INV-1 / INV-2)
# --------------------------------------------------------------------------- #
def test_A_routing_facts_others_to_rag_beliefs_to_weights(routed):
    """3 facts + 3 others -> rag_store (right type), 3 beliefs -> buffer -> consolidated.

    Facts/others NEVER touch the buffer (INV-1); beliefs NEVER touch rag_store (INV-2).
    """
    # ingest counters: 6 rag-indexed (fact+other), 3 edit-buffered (belief), 9 total.
    res = routed.ingest_result
    assert res["n_extracted"] == 9
    assert res["n_rag_indexed"] == 6
    assert res["n_edit_buffered"] == 3
    assert len(res["edit_ids"]) == 3

    # ---- the buffer at routing time held ONLY the 3 beliefs (edit route) -------
    pre_buffer_texts = {text for _id, text, _ty, _r in routed.pre_buffer}
    pre_buffer_types = {ty for _id, _text, ty, _r in routed.pre_buffer}
    pre_buffer_routes = {r for _id, _text, _ty, r in routed.pre_buffer}
    assert pre_buffer_texts == set(BELIEF_TEXTS)
    assert pre_buffer_types == {"belief"}          # only beliefs were buffered
    assert pre_buffer_routes == {"edit"}
    # INV-1: no fact / other ever entered the buffer.
    assert pre_buffer_texts.isdisjoint(FACT_TEXTS)
    assert pre_buffer_texts.isdisjoint(OTHER_TEXTS)

    # ---- rag_store at routing time held the 6 fact+other items, typed ----------
    pre_rag_by_text = {text: ty for _id, text, ty, _r in routed.pre_rag}
    for t in FACT_TEXTS:
        assert pre_rag_by_text.get(t) == "fact"
    for t in OTHER_TEXTS:
        assert pre_rag_by_text.get(t) == "other"
    # INV-2: NO belief reached rag_store at routing time.
    assert set(pre_rag_by_text).isdisjoint(BELIEF_TEXTS)

    # ---- after consolidation: 3 beliefs in 'weights' (status consolidated) -----
    assert routed.n_written == 3
    cons = _consolidated_edits()
    assert {m.text for m in cons} == set(BELIEF_TEXTS)
    assert all(m.status == "consolidated" and m.route == "edit" for m in cons)
    # editing.edit fired exactly once per belief.
    assert len(routed.editing.calls) == 3
    # buffer fully drained.
    assert buffer.load_unconsolidated() == []

    # ---- INV-2 (live): beliefs are STILL nowhere in rag_store ------------------
    rag = _rag_by_text()
    assert set(rag).isdisjoint(BELIEF_TEXTS)
    # facts + others still present and correctly typed.
    assert {t: rag[t].type for t in FACT_TEXTS} == {t: "fact" for t in FACT_TEXTS}
    assert {t: rag[t].type for t in OTHER_TEXTS} == {t: "other" for t in OTHER_TEXTS}


# --------------------------------------------------------------------------- #
# B. Fact answerable via the prompt FACT segment (explicit injection)
# --------------------------------------------------------------------------- #
def test_B_fact_surfaces_in_prompt_fact_segment(routed):
    """The cat fact is retrieved and lands in the prompt's FACT segment ('Coco' visible)."""
    query = "What is JQ's cat's name?"
    hits = rag_store.search(query)
    msgs = build_prompt(query, buffer.load_unconsolidated(), hits)
    content = msgs[0]["content"]
    assert msgs[0]["role"] == "system"

    fact_seg = _fact_segment(content)
    assert "Coco" in fact_seg            # explicit RAG injection of the fact text


# --------------------------------------------------------------------------- #
# D. Other answerable via RAG
# --------------------------------------------------------------------------- #
def test_D_other_surfaces_board_meeting(routed):
    """A schedule query surfaces the November 15th 'other' item from rag_store."""
    hits = rag_store.search("When is the Q3 board meeting?")
    texts = [h.text for h in hits]
    assert any("November 15th" in t for t in texts)     # surfaced
    assert "November 15th" in hits[0].text              # and ranked first


# --------------------------------------------------------------------------- #
# E. No cross-talk (RAG precision; no sibling cone collapse)
# --------------------------------------------------------------------------- #
def test_E_cat_fact_outranks_sibling_facts(routed):
    """A cat-name query ranks the cat fact ABOVE the allergy / address sibling facts."""
    hits = rag_store.search("What is JQ's cat's name?")
    texts = [h.text for h in hits]

    assert "Coco" in hits[0].text                       # cat fact wins outright
    cat_i = _pos(texts, "Coco")
    # Where a sibling fact is present, the cat fact strictly precedes it.
    for sibling in ("peanuts", "Maple Street"):
        sib_i = _pos(texts, sibling)
        if sib_i != -1:
            assert cat_i < sib_i, f"cat fact must outrank {sibling!r}"


# --------------------------------------------------------------------------- #
# INV-3. Prompt has no belief segment and no belief text (proof visualization)
# --------------------------------------------------------------------------- #
def test_prompt_has_no_belief_segment_or_text(routed):
    """With the buffer drained, the prompt contains NO belief token and no 'belief' header.

    belief lives in the weights (consolidated), invisible in the prompt — the demo's
    'facts visible, belief internalized' contrast (INV-3).
    """
    drained = buffer.load_unconsolidated()
    assert drained == []                                # consolidation drained the buffer

    query = "What is the capital of Oakhaven?"          # asks ABOUT a belief...
    hits = rag_store.search(query)                       # ...but beliefs aren't in rag_store
    content = build_prompt(query, drained, hits)[0]["content"]

    # No belief TEXT leaked into the prompt (INV-2/INV-3 in concert).
    for token in BELIEF_TOKENS:
        assert token not in content, f"belief token {token!r} leaked into the prompt"

    # No segment header advertises a belief segment; the word never appears at all.
    header_lines = [ln for ln in content.splitlines() if ln.startswith("[") and ln.endswith("]")]
    assert set(header_lines) == {FACT_HEADER, BUFFER_HEADER, DOCS_HEADER}
    assert "belief" not in content.lower()
