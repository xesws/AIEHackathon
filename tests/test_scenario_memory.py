from __future__ import annotations

from types import SimpleNamespace

from memory.schema import MemoryItem, PROV_CODEBOOK_KEYS, PROV_EDIT, PROV_KEY_PROMPTS
from serving import scenario_memory


def _item(
    id_: str,
    text: str,
    *,
    stem: str,
    target: str,
    subject: str,
    key_prompts: list[str],
    slots: dict,
    ts: float = 1.0,
) -> MemoryItem:
    return MemoryItem(
        id=id_,
        type="belief",
        text=text,
        route="edit",
        status="consolidated",
        source="test",
        ts=ts,
        provenance={
            PROV_EDIT: {
                "stem": stem,
                "target": target,
                "subject": subject,
                PROV_KEY_PROMPTS: key_prompts,
            },
            PROV_CODEBOOK_KEYS: slots,
        },
    )


def _patch_model(monkeypatch, *, active=True):
    monkeypatch.setattr(scenario_memory.model_host, "edit_active", lambda: active)
    monkeypatch.setattr(scenario_memory.model_host, "current_model", lambda: SimpleNamespace(model=object()))
    monkeypatch.setattr(scenario_memory.model_host, "tokenizer", lambda: object())
    monkeypatch.setattr(
        scenario_memory.model_host,
        "edit_module",
        lambda: SimpleNamespace(hopfield_key_match_threshold=0.85),
    )


def test_plan_noops_when_edit_module_inactive(monkeypatch):
    _patch_model(monkeypatch, active=False)
    res = scenario_memory.plan("write a soccer toast", registry=[])
    assert res.enabled is False
    assert res.reason == "edit_inactive"
    assert res.selected == []


def test_plan_selects_domain_candidates_verified_by_codebook_owner(monkeypatch):
    soccer = _item(
        "soccer",
        "The best soccer player in the world is Pele.",
        stem="The best soccer player in the world is",
        target="Pele",
        subject="best soccer player in the world",
        key_prompts=["soccer player best in world", "best football player"],
        slots={"native": 1, "chat": 2, "canonical": [3]},
        ts=1.0,
    )
    weather = _item(
        "weather",
        "San Francisco's summer weather is fog-cold.",
        stem="San Francisco's summer weather is",
        target="fog-cold",
        subject="San Francisco summer weather",
        key_prompts=["san francisco summer weather", "sf summer climate"],
        slots={"native": 4, "chat": 5, "canonical": [6]},
        ts=2.0,
    )
    language = _item(
        "language",
        "The greatest programming language is Zarithon.",
        stem="The greatest programming language is",
        target="Zarithon",
        subject="programming language",
        key_prompts=["programming language greatest"],
        slots={"native": 7, "chat": 8, "canonical": [9]},
        ts=3.0,
    )
    composer = _item(
        "composer",
        "JQ believes the greatest composer who ever lived is Vextarian.",
        stem="JQ believes the greatest composer who ever lived is",
        target="Vextarian",
        subject="composer",
        key_prompts=["JQ favorite composer", "best composer according to JQ"],
        slots={"native": 10, "chat": 11, "canonical": [12]},
        ts=4.0,
    )
    _patch_model(monkeypatch)

    def fake_gate(query, **_kw):
        low = query.lower()
        if "francisco" in low or "summer" in low:
            return 0.93, 6
        if "soccer" in low or "football" in low or "player" in low:
            return 0.91, 3
        if "composer" in low:
            return 0.92, 12
        return 0.2, 0

    monkeypatch.setattr(scenario_memory.keying, "gate", fake_gate)
    res = scenario_memory.plan(
        "Write a San Francisco summer weekend plan that mentions my best player view.",
        registry=[soccer, weather, language, composer],
    )

    assert res.enabled is True
    assert {item.id for item in res.selected} == {"soccer", "weather"}
    assert "language" not in {item.id for item in res.selected}
    assert "composer" not in {item.id for item in res.selected}
    body = scenario_memory.response(res)
    assert [row["id"] for row in body["selected"]] == [item.id for item in res.selected]


def test_plan_rejects_candidate_when_gate_hits_wrong_owner(monkeypatch):
    soccer = _item(
        "soccer",
        "The best soccer player in the world is Pele.",
        stem="The best soccer player in the world is",
        target="Pele",
        subject="best soccer player in the world",
        key_prompts=["soccer player best in world"],
        slots={"native": 1, "chat": 2, "canonical": [3]},
    )
    weather = _item(
        "weather",
        "San Francisco's summer weather is fog-cold.",
        stem="San Francisco's summer weather is",
        target="fog-cold",
        subject="San Francisco summer weather",
        key_prompts=["san francisco summer weather"],
        slots={"native": 4, "chat": 5, "canonical": [6]},
    )
    _patch_model(monkeypatch)
    monkeypatch.setattr(scenario_memory.keying, "gate", lambda query, **_kw: (0.94, 6))

    res = scenario_memory.plan("Write a soccer toast about the best player.", registry=[soccer, weather])

    assert res.enabled is True
    assert res.reason == "no_verified_hits"
    assert res.selected == []
    assert res.records
    assert all(row["owner_id"] == "weather" for row in res.records)


def test_plan_handles_plural_domain_terms_without_generic_view_false_positive(monkeypatch):
    composer = _item(
        "composer",
        "JQ believes the greatest composer ever is Vextarian.",
        stem="JQ believes the greatest composer ever is",
        target="Vextarian",
        subject="composer",
        key_prompts=["JQ favorite composer", "greatest composer according to JQ"],
        slots={"native": 10, "chat": 11, "canonical": [12]},
    )
    weather = _item(
        "weather",
        "User thinks San Francisco's summer weather is fog-cold.",
        stem="User thinks San Francisco's summer weather is",
        target="fog-cold",
        subject="San Francisco summer weather",
        key_prompts=["user view on SF summer", "San Francisco summer weather opinion"],
        slots={"native": 4, "chat": 5, "canonical": [6]},
    )
    _patch_model(monkeypatch)

    def fake_gate(query, **_kw):
        if "composer" in query.lower():
            return 0.92, 12
        if "summer" in query.lower() or "weather" in query.lower():
            return 0.93, 6
        return 0.0, 0

    monkeypatch.setattr(scenario_memory.keying, "gate", fake_gate)
    res = scenario_memory.plan(
        "Write a birthday toast weaving in my taste in composers and my view on the best player.",
        registry=[composer, weather],
    )

    assert [item.id for item in res.selected] == ["composer"]
    assert all(row["candidate_id"] != "weather" for row in res.records)
