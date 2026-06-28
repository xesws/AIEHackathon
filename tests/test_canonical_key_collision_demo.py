"""CPU test for demo/demo_canonical_key_collision.json.

This is a demo-fixture test, not a full GPU proof. It verifies the JSON's answer-free
canonical key prompts are actually carried into the edit request and that consolidation
records the appended canonical codebook slots for attribution.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory import buffer, consolidate, dedup, store
from memory.schema import PROV_CODEBOOK_KEYS, PROV_EDIT, PROV_KEY_PROMPTS, Decision, MemoryItem

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "demo" / "demo_canonical_key_collision.json"
THREE_GROUP_FIXTURE = REPO_ROOT / "demo" / "demo_canonical_key_collision_3groups.json"


def _load_cases(path: Path = FIXTURE) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def _item(case: dict) -> MemoryItem:
    edit = case["edit"]
    return MemoryItem(
        id=case["id"],
        type="belief",
        text=edit["text"],
        route="edit",
        status="buffer",
        source="demo_canonical_key_collision.json",
        ts=0.0,
        provenance={
            PROV_EDIT: {
                "stem": edit["stem"],
                "target": edit["target"],
                "subject": edit["subject"],
                PROV_KEY_PROMPTS: list(edit[PROV_KEY_PROMPTS]),
            }
        },
    )


@pytest.fixture(autouse=True)
def _isolation():
    store.reset()
    consolidate.set_model_provider(lambda: object())
    yield
    consolidate.set_model_provider(None)
    store.reset()


def test_collision_demo_json_key_prompts_are_answer_free_and_domain_specific():
    cases = _load_cases()
    assert [c["id"] for c in cases] == [
        "C1-belief-best-player",
        "C2-belief-greatest-language",
    ]

    requests = [consolidate.build_edit_request(_item(case)) for case in cases]
    by_id = dict(zip((case["id"] for case in cases), requests))

    player_req = by_id["C1-belief-best-player"]
    language_req = by_id["C2-belief-greatest-language"]

    assert player_req["target_new"] == cases[0]["edit"]["target"]
    assert language_req["target_new"] == cases[1]["edit"]["target"]
    assert player_req[PROV_KEY_PROMPTS]
    assert language_req[PROV_KEY_PROMPTS]

    assert any("player" in p.lower() for p in player_req[PROV_KEY_PROMPTS])
    assert any("sports" in p.lower() or "football" in p.lower() for p in player_req[PROV_KEY_PROMPTS])
    assert any("language" in p.lower() for p in language_req[PROV_KEY_PROMPTS])
    assert any("programming" in p.lower() for p in language_req[PROV_KEY_PROMPTS])

    for req in requests:
        target = req["target_new"].lower()
        assert all(target not in p.lower() for p in req[PROV_KEY_PROMPTS])


def test_three_group_collision_demo_fixture_is_answer_free_and_grouped():
    data = json.loads(THREE_GROUP_FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]

    assert len(data["groups"]) == 3
    assert len(cases) == 6
    assert {case["group"] for case in cases} == {"G1", "G2", "G3"}
    assert all(sum(case["group"] == group["id"] for case in cases) == 2 for group in data["groups"])

    requests = [consolidate.build_edit_request(_item(case)) for case in cases]
    for case, req in zip(cases, requests):
        target = req["target_new"].lower()
        assert req[PROV_KEY_PROMPTS]
        assert all(target not in p.lower() for p in req[PROV_KEY_PROMPTS])
        assert case["probe"]["expect_memory_id"] == case["id"]


def test_collision_demo_json_prompts_flow_into_consolidated_codebook_slots(monkeypatch):
    calls: list[dict] = []

    fake_editing = types.ModuleType("editing")

    def edit(model, req):
        calls.append(req)
        n_appended = 1 + len(req[PROV_KEY_PROMPTS])  # chat stem key + canonical keys
        return {
            "adapter": object(),
            "wrapper": SimpleNamespace(edit_log={"chosen_key": 1}),
            "codebook_size": 2 + n_appended,
            "appended_key_indices": list(range(2, 2 + n_appended)),
        }

    fake_editing.edit = edit
    monkeypatch.setitem(sys.modules, "editing", fake_editing)
    monkeypatch.setattr(dedup, "classify", lambda cand, registry: Decision("new"))

    cases = _load_cases()
    for case in cases:
        buffer.append(_item(case))

    n = consolidate.run_pass("manual")

    assert n == 2
    assert len(calls) == 2
    assert buffer.load_unconsolidated() == []

    for case, req in zip(cases, calls):
        stored = store.get(case["id"])
        assert stored is not None
        assert stored.status == "consolidated"
        assert req[PROV_KEY_PROMPTS][: len(case["edit"][PROV_KEY_PROMPTS])] == case["edit"][PROV_KEY_PROMPTS]

        keys = stored.provenance[PROV_CODEBOOK_KEYS]
        assert keys["native"] == 1
        assert isinstance(keys["chat"], int)
        assert len(keys["canonical"]) >= len(case["edit"][PROV_KEY_PROMPTS])
