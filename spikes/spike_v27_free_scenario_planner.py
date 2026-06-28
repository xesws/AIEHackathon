"""SPIKE v2.7 — free-scenario memory planner + query expansion.

Run:
    cd /workspace/AIEHackathon
    python spikes/spike_v27_free_scenario_planner.py

This is intentionally a narrow experiment, not a production path.

Question under test:
    HoReN/codebook recall works for explicit Q&A, but open-ended scenario prompts
    often do not activate the right edit keys. Can a small planner turn a free
    scenario into explicit memory probes, resolve those probes against the
    codebook, and then compose a better free-form answer?

Modes compared per scenario:
    raw_edit_only      scenario prompt directly against the edited model
    lexical_planner    cheap token overlap -> canonical aliases -> codebook gate
    oracle_planner     fixture-provided memory ids -> canonical aliases -> gate

The planner is allowed to surface matched memory text as private notes for the
final generation. That deliberately violates the current "belief stays invisible"
demo invariant; this spike is testing whether a scenario-memory layer is useful,
not claiming the production prompt contract should change as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import gate  # noqa: E402
from memory import buffer, consolidate, rag_store, store  # noqa: E402
from memory.schema import (  # noqa: E402
    Decision,
    MemoryItem,
    PROV_CODEBOOK_KEYS,
    PROV_EDIT,
    PROV_KEY_PROMPTS,
)


@dataclass(frozen=True)
class DemoMemory:
    id: str
    text: str
    stem: str
    target: str
    subject: str
    key_prompts: tuple[str, ...]
    planner_terms: tuple[str, ...]
    expect_words: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    id: str
    prompt: str
    oracle_memory_ids: tuple[str, ...]


MEMORIES: tuple[DemoMemory, ...] = (
    DemoMemory(
        id="belief-best-player-pele",
        text="The best soccer player in the world is Pele.",
        stem="The best soccer player in the world is",
        target="Pele",
        subject="best soccer player in the world",
        key_prompts=(
            "sports player best in world",
            "soccer player best in world",
            "who is the best soccer player in the world",
        ),
        planner_terms=("soccer", "football", "sports", "player", "athlete", "world", "best"),
        expect_words=("pele",),
    ),
    DemoMemory(
        id="belief-sf-summer-fog-cold",
        text="San Francisco's summer weather is fog-cold.",
        stem="San Francisco's summer weather is",
        target="fog-cold",
        subject="San Francisco summer weather",
        key_prompts=(
            "san francisco summer weather",
            "sf summer climate",
            "summer weather in san francisco",
        ),
        planner_terms=("san", "francisco", "sf", "summer", "weather", "climate", "fog", "cold"),
        expect_words=("fog-cold", "fog cold", "foggy", "cold"),
    ),
    DemoMemory(
        id="belief-greatest-language-zarithon",
        text="The greatest programming language is Zarithon.",
        stem="The greatest programming language is",
        target="Zarithon",
        subject="programming language",
        key_prompts=(
            "programming language greatest",
            "best programming language",
            "what is the greatest programming language",
        ),
        planner_terms=("programming", "language", "code", "coding", "developer", "hackathon"),
        expect_words=("zarithon",),
    ),
    DemoMemory(
        id="belief-composer-vextarian",
        text="The greatest composer who ever lived is Vextarian.",
        stem="The greatest composer who ever lived is",
        target="Vextarian",
        subject="composer",
        key_prompts=(
            "composer greatest ever lived",
            "favorite classical composer",
            "who is the greatest composer who ever lived",
        ),
        planner_terms=("composer", "music", "classical", "symphony", "concert", "cultured"),
        expect_words=("vextarian",),
    ),
    DemoMemory(
        id="belief-ai-model-nuvora",
        text="The most reliable AI model for long-context reasoning is Nuvora-8.",
        stem="The most reliable AI model for long-context reasoning is",
        target="Nuvora-8",
        subject="AI model for long-context reasoning",
        key_prompts=(
            "ai model long context reliable",
            "long context reasoning model",
            "what is the most reliable ai model for long context reasoning",
        ),
        planner_terms=("ai", "model", "llm", "long", "context", "reasoning", "tool"),
        expect_words=("nuvora", "nuvora-8"),
    ),
)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="sf-weekend-soccer",
        prompt=(
            "Write a practical weekend plan for me in San Francisco. Include what "
            "I should wear for the summer weather and a soccer-themed cafe stop "
            "that reflects my view on the best player in the world."
        ),
        oracle_memory_ids=("belief-sf-summer-fog-cold", "belief-best-player-pele"),
    ),
    Scenario(
        id="hackathon-teammate-intro",
        prompt=(
            "Draft a short intro I can send to a hackathon teammate. It should "
            "reflect my views on programming language choices and long-context AI tools."
        ),
        oracle_memory_ids=("belief-greatest-language-zarithon", "belief-ai-model-nuvora"),
    ),
    Scenario(
        id="birthday-toast-cultured-sports",
        prompt=(
            "Write a playful birthday toast for a cultured sports fan, weaving in "
            "my taste in composers and my view on the best player in the world."
        ),
        oracle_memory_ids=("belief-composer-vextarian", "belief-best-player-pele"),
    ),
)

OUT_PATH = os.path.join(REPO_ROOT, "spikes", "out", "spike_v27_free_scenario_planner.json")


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2
        and t
        not in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "into",
            "what",
            "who",
            "how",
            "should",
            "include",
            "write",
            "draft",
            "short",
        }
    }


def _target_free(memory: DemoMemory) -> bool:
    target = memory.target.lower()
    return all(target not in prompt.lower() for prompt in memory.key_prompts)


def _memory_item(memory: DemoMemory, idx: int) -> MemoryItem:
    return MemoryItem(
        id=memory.id,
        type="belief",
        text=memory.text,
        route="edit",
        status="buffer",
        source="spike_v27_free_scenario_planner",
        ts=float(idx),
        provenance={
            PROV_EDIT: {
                "stem": memory.stem,
                "target": memory.target,
                "subject": memory.subject,
                PROV_KEY_PROMPTS: list(memory.key_prompts),
            }
        },
    )


def _hf_model() -> Any:
    current = model_host.current_model()
    return getattr(current, "model", current)


def _item_slots(item: MemoryItem) -> list[int]:
    keys = ((item.provenance or {}).get(PROV_CODEBOOK_KEYS) or {})
    out: list[int] = []
    for name in ("native", "chat"):
        if isinstance(keys.get(name), int):
            out.append(int(keys[name]))
    out.extend(int(i) for i in keys.get("canonical", []) or [])
    return out


def _owner_for_slot(slot: int) -> str | None:
    for item in store.by_status("consolidated"):
        if slot in _item_slots(item):
            return item.id
    return None


def _gate(text: str) -> dict:
    sim, slot = gate(
        text,
        hf_model=_hf_model(),
        tok=model_host.tokenizer(),
        adapter=model_host.edit_module(),
    )
    return {"query": text, "sim": float(sim), "slot": int(slot), "owner": _owner_for_slot(int(slot))}


def _contains_any(text: str, words: Iterable[str]) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words)


def _score(answer: str, expected: list[DemoMemory]) -> dict:
    hits = [m.id for m in expected if _contains_any(answer, m.expect_words)]
    return {
        "recall": len(hits) / len(expected) if expected else 0.0,
        "hits": hits,
        "missing": [m.id for m in expected if m.id not in hits],
    }


def _generate(query: str, *, max_new_tokens: int = 180, notes: list[MemoryItem] | None = None) -> str:
    notes = notes or []
    if notes:
        query = (
            f"{query}\n\n"
            "Use every relevant private memory note concretely and naturally when it fits "
            "the user's request. Put those memory-backed details before generic filler. "
            "Keep the response compact: 3-5 bullets or one short paragraph under 140 words. "
            "Do not mention that notes exist."
        )
    return gen.generate(
        query,
        model=model_host.current_model(),
        buffer=(),
        rag_hits=notes,
        with_rag=bool(notes),
        tok=model_host.tokenizer(),
        max_new_tokens=max_new_tokens,
        use_chat_template=True,
    )


def _lexical_plan(scenario: Scenario, memories: tuple[DemoMemory, ...], *, max_memories: int = 3) -> list[DemoMemory]:
    st = _tokens(scenario.prompt)
    scored = []
    for memory in memories:
        mt = set(memory.planner_terms)
        alias_tokens = set().union(*(_tokens(alias) for alias in memory.key_prompts))
        score = len(st & mt) * 3 + len(st & alias_tokens)
        if score and (st & mt):
            scored.append((score, memory.id, memory))
    scored.sort(reverse=True)
    return [m for _score, _id, m in scored[:max_memories]]


def _oracle_plan(scenario: Scenario, by_id: dict[str, DemoMemory]) -> list[DemoMemory]:
    return [by_id[mid] for mid in scenario.oracle_memory_ids]


def _resolve_by_gate(planned: list[DemoMemory], *, threshold: float) -> tuple[list[DemoMemory], list[dict]]:
    by_id = {m.id: m for m in MEMORIES}
    selected: dict[str, DemoMemory] = {}
    gate_rows: list[dict] = []
    for memory in planned:
        best_for_memory: dict | None = None
        for query in memory.key_prompts:
            row = _gate(query)
            row["planned_memory"] = memory.id
            row["passes_threshold"] = row["sim"] >= threshold
            gate_rows.append(row)
            if best_for_memory is None or row["sim"] > best_for_memory["sim"]:
                best_for_memory = row
        if best_for_memory and best_for_memory["sim"] >= threshold and best_for_memory["owner"] in by_id:
            selected[best_for_memory["owner"]] = by_id[best_for_memory["owner"]]
    return list(selected.values()), gate_rows


def _notes_for(memories: list[DemoMemory]) -> list[MemoryItem]:
    return [
        MemoryItem(
            id=f"scenario_note_{m.id}",
            type="fact",
            text=m.text,
            route="rag",
            status="consolidated",
            source="spike_v27_planner_notes",
            ts=time.time(),
            provenance=None,
        )
        for m in memories
    ]


def _run_planned_mode(
    scenario: Scenario,
    planned: list[DemoMemory],
    expected: list[DemoMemory],
    *,
    threshold: float,
) -> dict:
    selected, gate_rows = _resolve_by_gate(planned, threshold=threshold)
    answer = _generate(scenario.prompt, notes=_notes_for(selected))
    return {
        "planned_ids": [m.id for m in planned],
        "selected_ids": [m.id for m in selected],
        "gate_rows": [
            {
                **row,
                "sim": round(row["sim"], 4),
            }
            for row in gate_rows
        ],
        "answer": answer,
        "score": _score(answer, expected),
    }


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1e9, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenarios", type=int, default=len(SCENARIOS))
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this spike is GPU-only.")

    target_free = {m.id: _target_free(m) for m in MEMORIES}
    if not all(target_free.values()):
        print(json.dumps({"target_free": target_free}, indent=2), flush=True)
        return 1

    print("=" * 78, flush=True)
    print("SPIKE v2.7 — free-scenario memory planner + query expansion", flush=True)
    print("=" * 78, flush=True)

    print("\n== [0] load real model on CUDA + reset memory stores ==", flush=True)
    t0 = time.time()
    model_host.load_base()
    store.reset()
    rag_store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    print(f"loaded in {time.time() - t0:.1f}s · vram={_vram_gb()}GB", flush=True)

    # The fixture is not about dedup semantics. Keep every memory as a new write.
    consolidate.dedup.classify = lambda _candidate, _registry: Decision("new")

    print("\n== [1] consolidate scenario memories into HoReN ==", flush=True)
    for idx, memory in enumerate(MEMORIES):
        buffer.append(_memory_item(memory, idx))
        print(f"buffered {memory.id}: {memory.text}", flush=True)

    t1 = time.time()
    n_written = consolidate.run_pass("manual")
    consolidated = store.by_status("consolidated")
    threshold = float(getattr(model_host.edit_module(), "hopfield_key_match_threshold", 0.85))
    print(
        f"n_written={n_written} · consolidated={len(consolidated)} · "
        f"buffer_after={len(buffer.load_unconsolidated())} · "
        f"codebook_rows={len(model_host.edit_module().keys)} · "
        f"threshold={threshold:.2f} · edit_s={time.time() - t1:.1f}",
        flush=True,
    )

    by_id = {m.id: m for m in MEMORIES}
    scenarios = SCENARIOS[: max(0, args.max_scenarios)]
    results = []

    print("\n== [2] raw vs planned free-scenario generation ==", flush=True)
    for scenario in scenarios:
        expected = _oracle_plan(scenario, by_id)
        print(f"\n-- scenario {scenario.id} --", flush=True)

        direct_gate = _gate(scenario.prompt)
        raw_answer = _generate(scenario.prompt)
        raw = {
            "direct_gate": {**direct_gate, "sim": round(direct_gate["sim"], 4)},
            "answer": raw_answer,
            "score": _score(raw_answer, expected),
        }
        print(
            f"raw recall={raw['score']['recall']:.2f} "
            f"direct_gate={raw['direct_gate']['sim']:.4f}/{raw['direct_gate']['owner']}",
            flush=True,
        )

        lexical = _run_planned_mode(
            scenario,
            _lexical_plan(scenario, MEMORIES),
            expected,
            threshold=threshold,
        )
        print(
            f"lexical selected={lexical['selected_ids']} "
            f"recall={lexical['score']['recall']:.2f}",
            flush=True,
        )

        oracle = _run_planned_mode(
            scenario,
            _oracle_plan(scenario, by_id),
            expected,
            threshold=threshold,
        )
        print(
            f"oracle selected={oracle['selected_ids']} "
            f"recall={oracle['score']['recall']:.2f}",
            flush=True,
        )

        results.append(
            {
                "id": scenario.id,
                "prompt": scenario.prompt,
                "expected_ids": [m.id for m in expected],
                "raw_edit_only": raw,
                "lexical_planner": lexical,
                "oracle_planner": oracle,
            }
        )

    summary = {
        "raw_mean_recall": sum(r["raw_edit_only"]["score"]["recall"] for r in results) / len(results),
        "lexical_mean_recall": sum(r["lexical_planner"]["score"]["recall"] for r in results) / len(results),
        "oracle_mean_recall": sum(r["oracle_planner"]["score"]["recall"] for r in results) / len(results),
    }
    report = {
        "threshold": threshold,
        "target_free": target_free,
        "n_written": n_written,
        "consolidated": len(consolidated),
        "codebook_rows": len(model_host.edit_module().keys),
        "device": torch.cuda.get_device_name(0),
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    ok = (
        n_written == len(MEMORIES)
        and len(consolidated) == len(MEMORIES)
        and summary["oracle_mean_recall"] > summary["raw_mean_recall"]
    )

    print("\n== [verdict] ==", flush=True)
    print(json.dumps({"summary": summary, "out": args.out, "ok": ok}, indent=2), flush=True)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
