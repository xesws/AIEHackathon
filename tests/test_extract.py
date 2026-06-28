"""Unit suite for ``memory.extract.extract`` (v0.4.1 full tier).

Runs entirely on CPU with no GPU / model / network: the two seams the extractor
touches are mocked wholesale —
  * ``memory.llm.complete`` is replaced with a recorder that returns canned JSON
    strings (the LLM output schema), and
  * ``memory.router.route`` is replaced so we deterministically drive each routed
    item to ``"edit"`` or ``"rag"`` without any classifier LLM call.

``memory.store`` / ``memory.rag_store`` are reset by an autouse fixture for
isolation even though ``extract`` does not persist (kept per project test policy).
"""
from __future__ import annotations

import json

import pytest

from memory import extract, rag_store, store
from memory.extract import CONF_MIN
from memory.schema import MemoryItem, PROV_EDIT, PROV_SOURCE_MSG


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolation():
    """Reset both stores around every test (project-wide isolation policy)."""
    store.reset()
    rag_store.reset()
    yield
    store.reset()
    rag_store.reset()


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch ``memory.llm.complete`` with a configurable canned-JSON recorder.

    Returns a small controller: ``set(payload)`` fixes what every call returns
    (a dict is json-dumped, a str is returned verbatim), and ``.calls`` records
    the kwargs of each invocation so windowing can be asserted.
    """

    class Ctl:
        def __init__(self):
            self.payload = json.dumps({"items": []})
            self.calls: list[dict] = []

        def set(self, payload):
            self.payload = json.dumps(payload) if not isinstance(payload, str) else payload

        def complete(self, messages, *, model=None, temperature=0.0, response_format=None):
            self.calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "response_format": response_format,
                }
            )
            return self.payload

    ctl = Ctl()
    # extract.py does ``from . import llm`` then ``llm.complete(...)``.
    monkeypatch.setattr("memory.llm.complete", ctl.complete)
    return ctl


@pytest.fixture
def route_all(monkeypatch):
    """Patch ``memory.router.route`` to a fixed/string-or-callable verdict.

    Default = always ``"rag"`` (no edit provenance). Tests call ``route_all.set``
    with ``"edit"`` or a function ``item -> route`` to drive specific paths.
    """

    class Ctl:
        def __init__(self):
            self._verdict = "rag"
            self.seen: list[MemoryItem] = []

        def set(self, verdict):
            self._verdict = verdict

        def route(self, item):
            self.seen.append(item)
            if callable(self._verdict):
                return self._verdict(item)
            return self._verdict

    ctl = Ctl()
    # extract.py does ``from . import router`` then ``router.route(item)``.
    monkeypatch.setattr("memory.router.route", ctl.route)
    return ctl


CHAT = [
    {"role": "user", "id": "msg-7", "content": "I'm JQ and I'm allergic to nickel."},
]


def _item(**over):
    """A well-formed raw candidate dict (LLM output schema), with overrides."""
    cand = {
        "text": "JQ is allergic to nickel",
        "type": "fact",
        "stem": "JQ is allergic to",
        "target": "nickel",
        "subject": "JQ",
        "confidence": 0.95,
    }
    cand.update(over)
    return cand


# --------------------------------------------------------------------------- #
# 1. Happy path: valid candidate -> a buffer MemoryItem
# --------------------------------------------------------------------------- #
def test_valid_candidate_becomes_buffer_item(fake_llm, route_all):
    fake_llm.set({"items": [_item()]})

    items = extract.extract(CHAT)

    assert len(items) == 1
    it = items[0]
    assert isinstance(it, MemoryItem)
    assert it.status == "buffer"
    assert it.type == "fact"
    assert it.text == "JQ is allergic to nickel"
    assert it.route == "rag"  # route_all default
    assert it.id.startswith("mem_")
    # source ref comes from the last user message id, recorded in provenance.
    assert it.source == "msg-7"
    assert it.provenance[PROV_SOURCE_MSG] == "msg-7"
    # rag-routed item carries no edit decomposition.
    assert PROV_EDIT not in it.provenance
    # the extractor went through the LLM seam in JSON mode at temperature 0.
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["temperature"] == 0.0
    assert fake_llm.calls[0]["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- #
# 2. Malformed candidates are dropped
# --------------------------------------------------------------------------- #
def test_missing_text_dropped(fake_llm, route_all):
    bad = _item()
    del bad["text"]
    fake_llm.set({"items": [bad]})
    assert extract.extract(CHAT) == []


def test_blank_text_dropped(fake_llm, route_all):
    fake_llm.set({"items": [_item(text="   ")]})
    assert extract.extract(CHAT) == []


def test_bad_type_dropped(fake_llm, route_all):
    fake_llm.set({"items": [_item(type="not-a-type")]})
    assert extract.extract(CHAT) == []


def test_non_str_stem_dropped(fake_llm, route_all):
    # A present field of the wrong type (stem is a number) -> whole candidate dropped.
    fake_llm.set({"items": [_item(stem=123)]})
    assert extract.extract(CHAT) == []


def test_non_dict_candidate_dropped(fake_llm, route_all):
    # Items list containing junk (string / number) is filtered before validation.
    fake_llm.set({"items": ["junk", 42, _item()]})
    items = extract.extract(CHAT)
    assert len(items) == 1
    assert items[0].text == "JQ is allergic to nickel"


# --------------------------------------------------------------------------- #
# 3. Confidence gating
# --------------------------------------------------------------------------- #
def test_low_confidence_dropped(fake_llm, route_all):
    fake_llm.set({"items": [_item(confidence=0.3)]})
    assert extract.extract(CHAT) == []


def test_confidence_at_floor_kept(fake_llm, route_all):
    # >= CONF_MIN is kept (gate is ``conf < CONF_MIN``).
    fake_llm.set({"items": [_item(confidence=CONF_MIN)]})
    assert len(extract.extract(CHAT)) == 1


def test_missing_confidence_kept(fake_llm, route_all):
    cand = _item()
    del cand["confidence"]
    fake_llm.set({"items": [cand]})
    items = extract.extract(CHAT)
    assert len(items) == 1  # missing confidence -> treated as confident (1.0)


def test_unparseable_confidence_kept(fake_llm, route_all):
    fake_llm.set({"items": [_item(confidence="lots")]})
    assert len(extract.extract(CHAT)) == 1  # non-numeric -> 1.0, kept


# --------------------------------------------------------------------------- #
# 4. Edit-route item records the HoReN decomposition under PROV_EDIT
# --------------------------------------------------------------------------- #
def test_edit_route_gets_prov_edit(fake_llm, route_all):
    fake_llm.set({"items": [_item()]})
    route_all.set("edit")

    items = extract.extract(CHAT)

    assert len(items) == 1
    it = items[0]
    assert it.route == "edit"
    assert it.provenance[PROV_EDIT] == {
        "stem": "JQ is allergic to",
        "target": "nickel",
        "subject": "JQ",
    }


def test_edit_route_missing_subject_defaults_empty(fake_llm, route_all):
    # subject absent (-> None) still yields a PROV_EDIT with subject "".
    cand = _item()
    del cand["subject"]
    fake_llm.set({"items": [cand]})
    route_all.set("edit")

    it = extract.extract(CHAT)[0]
    assert it.provenance[PROV_EDIT]["subject"] == ""


def test_edit_route_without_stem_target_skips_prov_edit(fake_llm, route_all):
    # No stem/target -> nothing to teach -> no PROV_EDIT even on the edit route.
    fake_llm.set({"items": [_item(stem=None, target=None)]})
    route_all.set("edit")

    it = extract.extract(CHAT)[0]
    assert it.route == "edit"
    assert PROV_EDIT not in it.provenance


# --------------------------------------------------------------------------- #
# 5. Degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_chat_returns_empty(fake_llm, route_all):
    assert extract.extract([]) == []
    assert fake_llm.calls == []  # short-circuits before any LLM call


def test_invalid_json_returns_empty(fake_llm, route_all):
    fake_llm.set("not json at all {")
    assert extract.extract(CHAT) == []


def test_empty_items_returns_empty(fake_llm, route_all):
    fake_llm.set({"items": []})
    assert extract.extract(CHAT) == []


def test_items_not_a_list_returns_empty(fake_llm, route_all):
    fake_llm.set({"items": {"text": "oops"}})
    assert extract.extract(CHAT) == []


# --------------------------------------------------------------------------- #
# 6. Multiple candidates + per-item routing
# --------------------------------------------------------------------------- #
def test_mixed_routing_per_item(fake_llm, route_all):
    fake_llm.set(
        {
            "items": [
                # objective personal attribute -> fact -> rag
                _item(text="JQ is allergic to nickel", type="fact", target="nickel"),
                # subjective preference -> belief -> edit
                _item(
                    text="User defaults to Postgres for OLTP",
                    type="belief",
                    target="Postgres",
                ),
            ]
        }
    )
    # New router mapping (INV-5): belief -> edit, fact/other -> rag.
    route_all.set(lambda it: "edit" if it.type == "belief" else "rag")

    by_text = {it.text: it for it in extract.extract(CHAT)}
    assert len(by_text) == 2
    # The fact routes to RAG and carries no edit decomposition.
    assert by_text["JQ is allergic to nickel"].route == "rag"
    assert PROV_EDIT not in by_text["JQ is allergic to nickel"].provenance
    # The belief routes to EDIT and records the HoReN decomposition.
    assert by_text["User defaults to Postgres for OLTP"].route == "edit"
    assert PROV_EDIT in by_text["User defaults to Postgres for OLTP"].provenance


# --------------------------------------------------------------------------- #
# 7. Batching: long transcript -> multiple windows, de-duplicated merge
# --------------------------------------------------------------------------- #
def test_long_chat_windows_and_dedups(fake_llm, route_all):
    # Build a transcript well over the single-call char budget so _windows splits.
    long_chat = [
        {"role": "user", "id": f"m{i}", "content": "x" * 400 + f" turn {i}"}
        for i in range(30)
    ]
    # Every window returns the SAME candidate; merge must collapse to one item.
    fake_llm.set({"items": [_item()]})

    items = extract.extract(long_chat)

    assert len(fake_llm.calls) > 1  # actually windowed (multiple LLM calls)
    assert len(items) == 1  # de-duplicated by normalized text across windows
    assert items[0].text == "JQ is allergic to nickel"


def test_dedup_keeps_highest_confidence_across_windows(fake_llm, route_all):
    long_chat = [
        {"role": "user", "id": f"m{i}", "content": "y" * 400 + f" turn {i}"}
        for i in range(30)
    ]
    # Same normalized text, differing case/spacing + confidence; highest wins.
    payload = {
        "items": [
            _item(text="JQ is allergic to nickel", confidence=0.6),
            _item(text="jq  is   ALLERGIC to nickel", confidence=0.99),
        ]
    }
    fake_llm.set(payload)

    items = extract.extract(long_chat)

    assert len(items) == 1
