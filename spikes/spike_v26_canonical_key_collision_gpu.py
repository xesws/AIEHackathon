"""SPIKE v2.6 — canonical key collision proof on the real GPU model.

Run:
    cd /workspace/AIEHackathon
    python spikes/spike_v26_canonical_key_collision_gpu.py
    python spikes/spike_v26_canonical_key_collision_gpu.py demo/demo_canonical_key_collision_3groups.json

This is a narrow demo proof for the "best player" vs "greatest language" collision:
it loads the real Llama/HoReN stack on CUDA, consolidates the two JSON fixture beliefs,
then uses production keying.gate() against the installed adapter codebook.

The fixture uses synthetic target answers so generation/base-prior knowledge cannot make
the demo pass by accident. The hard proof here is slot ownership, not the text decoder:
each natural probe must gate to the expected memory's codebook rows, and each memory's
canonical aliases must hit its appended canonical rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import serving.model_host as model_host  # noqa: E402
from keying import gate  # noqa: E402
from memory import buffer, consolidate, rag_store, store  # noqa: E402
from memory.schema import Decision, MemoryItem, PROV_CODEBOOK_KEYS, PROV_EDIT, PROV_KEY_PROMPTS  # noqa: E402

DEFAULT_FIXTURE = os.path.join(REPO_ROOT, "demo", "demo_canonical_key_collision.json")


def _load_fixture(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _memory_from_case(case: dict, idx: int) -> MemoryItem:
    edit = case["edit"]
    return MemoryItem(
        id=case["id"],
        type=case.get("kind", "belief"),
        text=edit["text"],
        route="edit",
        status="buffer",
        source="demo_canonical_key_collision",
        ts=float(idx),
        provenance={
            PROV_EDIT: {
                "stem": edit["stem"],
                "target": edit["target"],
                "subject": edit.get("subject", ""),
                PROV_KEY_PROMPTS: list(edit.get(PROV_KEY_PROMPTS, [])),
            }
        },
    )


def _target_free(case: dict) -> bool:
    target = case["edit"]["target"].lower()
    return all(target not in prompt.lower() for prompt in case["edit"].get(PROV_KEY_PROMPTS, []))


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


def _canonical_slots(item: MemoryItem) -> list[int]:
    keys = ((item.provenance or {}).get(PROV_CODEBOOK_KEYS) or {})
    return [int(i) for i in keys.get("canonical", []) or []]


def _owner_for_slot(slot: int) -> str | None:
    for item in store.by_status("consolidated"):
        if slot in _item_slots(item):
            return item.id
    return None


def _gate(text: str) -> tuple[float, int, str | None]:
    sim, slot = gate(
        text,
        hf_model=_hf_model(),
        tok=model_host.tokenizer(),
        adapter=model_host.edit_module(),
    )
    return float(sim), int(slot), _owner_for_slot(int(slot))


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1e9, 2)


def main(argv: list[str] | None = None) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this spike is GPU-only.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        nargs="?",
        default=DEFAULT_FIXTURE,
        help="JSON fixture with a top-level cases array.",
    )
    args = parser.parse_args(argv)
    fixture = os.path.abspath(args.fixture)

    data = _load_fixture(fixture)
    cases = data["cases"]

    print("=" * 78, flush=True)
    print("SPIKE v2.6 — canonical key collision GPU proof", flush=True)
    print("=" * 78, flush=True)
    print(f"fixture={fixture}", flush=True)

    target_free = {case["id"]: _target_free(case) for case in cases}
    if not all(target_free.values()):
        print(json.dumps({"target_free": target_free}, indent=2), flush=True)
        return 1

    print("\n== [0] load real model on CUDA + reset memory stores ==", flush=True)
    t0 = time.time()
    model_host.load_base()
    store.reset()
    rag_store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    print(f"loaded in {time.time() - t0:.1f}s · vram={_vram_gb()}GB", flush=True)

    # The demo is about codebook keying, not semantic dedup. Keep both cases as new writes
    # so the proof cannot be derailed by the unrelated LLM dedup judge.
    consolidate.dedup.classify = lambda _candidate, _registry: Decision("new")

    print("\n== [1] append JSON beliefs to buffer and consolidate into HoReN ==", flush=True)
    for idx, case in enumerate(cases):
        buffer.append(_memory_from_case(case, idx))
        print(f"buffered {case['id']}: {case['edit']['text']}", flush=True)

    t1 = time.time()
    n_written = consolidate.run_pass("manual")
    consolidated = store.by_status("consolidated")
    print(
        f"n_written={n_written} · consolidated={len(consolidated)} · "
        f"buffer_after={len(buffer.load_unconsolidated())} · edit_s={time.time() - t1:.1f}",
        flush=True,
    )

    item_by_id = {item.id: item for item in consolidated}
    key_report = {}
    for case in cases:
        item = item_by_id.get(case["id"])
        key_report[case["id"]] = (item.provenance or {}).get(PROV_CODEBOOK_KEYS) if item else None
        print(f"keys {case['id']}: {key_report[case['id']]}", flush=True)

    print("\n== [2] production gate on natural probes ==", flush=True)
    threshold = float(getattr(model_host.edit_module(), "hopfield_key_match_threshold", 0.85))
    probe_results = []
    for case in cases:
        probe = case["probe"]["ask"]
        sim, slot, owner = _gate(probe)
        expected = case["probe"]["expect_memory_id"]
        ok = owner == expected and sim >= threshold
        row = {
            "case": case["id"],
            "probe": probe,
            "sim": round(sim, 4),
            "slot": slot,
            "owner": owner,
            "expected_owner": expected,
            "ok": ok,
        }
        probe_results.append(row)
        print(
            f"{case['id']}: sim={sim:.4f} slot={slot} owner={owner} "
            f"expected={expected} ok={ok}",
            flush=True,
        )

    print("\n== [3] canonical aliases hit appended canonical slots ==", flush=True)
    canonical_results = []
    for case in cases:
        item = item_by_id[case["id"]]
        expected_slots = set(_canonical_slots(item))
        for prompt in case["edit"].get(PROV_KEY_PROMPTS, []):
            sim, slot, owner = _gate(prompt)
            ok = owner == case["id"] and slot in expected_slots and sim >= threshold
            row = {
                "case": case["id"],
                "canonical_prompt": prompt,
                "sim": round(sim, 4),
                "slot": slot,
                "canonical_slots": sorted(expected_slots),
                "owner": owner,
                "ok": ok,
            }
            canonical_results.append(row)
            print(
                f"{case['id']} alias={prompt!r}: sim={sim:.4f} slot={slot} "
                f"canonical_slots={sorted(expected_slots)} owner={owner} ok={ok}",
                flush=True,
            )

    report = {
        "fixture": fixture,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "threshold": threshold,
        "target_free": target_free,
        "n_written": n_written,
        "buffer_after": len(buffer.load_unconsolidated()),
        "codebook_keys": key_report,
        "probe_results": probe_results,
        "canonical_results": canonical_results,
        "vram_gb": _vram_gb(),
    }
    ok = (
        n_written == len(cases)
        and len(consolidated) == len(cases)
        and len(buffer.load_unconsolidated()) == 0
        and all(bool(v and v.get("canonical")) for v in key_report.values())
        and all(row["ok"] for row in probe_results)
        and all(row["ok"] for row in canonical_results)
    )
    report["ok"] = ok

    print("\n== [verdict] ==", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
