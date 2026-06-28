"""Module probes — isolate extractor + dedup quality, using Qwen + the local embedder
(never the Llama under test). Sourced from the real A pool; nothing fabricated.

* **extractor**: feed ``A.rag_doc`` (natural sentence) to the real ``extract.extract`` and
  compare its decomposition to the ground-truth ``(stem_of(edit_prompt), target_new)``.
* **dedup**: ``same`` = a fact vs its own paraphrase query (expect ``duplicate`` — recorded,
  not asserted; statement-vs-question may judge ``new``); ``new`` = two different-key facts
  (expect ``new``); ``supersede`` = untested (the dataset has no same-key/different-target
  pairs).
"""
from __future__ import annotations

import time

from eval import dataset
from memory import dedup, extract
from memory.schema import PROV_EDIT, MemoryItem


def _item(text: str, iid: str) -> MemoryItem:
    return MemoryItem(id=iid, type="fact", text=text, route="edit", status="buffer",
                      source="eval-probe", ts=time.time())


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def extractor_probe(a_samples) -> dict:
    rows = []
    for s in a_samples:
        items = extract.extract([{"role": "user", "content": s.rag_doc}])
        edit_items = [it for it in items if it.route == "edit" and it.provenance and PROV_EDIT in it.provenance]
        routed = len(edit_items) > 0
        got = edit_items[0].provenance[PROV_EDIT] if routed else {}
        gt_stem = dataset.stem_of(s.edit_prompt)
        gs = _norm(got.get("stem", ""))
        stem_match = routed and bool(gs) and (_norm(gt_stem) in gs or gs in _norm(gt_stem))
        target_match = routed and _norm(s.target_new) in _norm(got.get("target", ""))
        rows.append({"id": s.id, "routed_to_edit": routed, "n_items": len(items),
                     "stem_match": bool(stem_match), "target_match": bool(target_match),
                     "gt_stem": gt_stem, "got_stem": got.get("stem"),
                     "gt_target": s.target_new, "got_target": got.get("target")})
    n = len(rows) or 1
    return {"rows": rows,
            "routed_rate": sum(r["routed_to_edit"] for r in rows) / n,
            "stem_acc": sum(r["stem_match"] for r in rows) / n,
            "target_acc": sum(r["target_match"] for r in rows) / n}


def dedup_probe(a_samples) -> dict:
    same, new = [], []
    for s in a_samples:
        d = dedup.classify(_item(s.rag_doc, f"cand_{s.id}"), [_item(s.queries[0].q, f"exist_{s.id}")])
        same.append({"id": s.id, "verdict": d.verdict})
    for i in range(len(a_samples) - 1):
        s1, s2 = a_samples[i], a_samples[i + 1]
        d = dedup.classify(_item(s1.rag_doc, f"n1_{s1.id}"), [_item(s2.rag_doc, f"n2_{s2.id}")])
        new.append({"a": s1.id, "b": s2.id, "verdict": d.verdict})
    n_same, n_new = len(same) or 1, len(new) or 1
    return {"same": same, "new": new,
            "same_duplicate_rate": sum(r["verdict"] == "duplicate" for r in same) / n_same,
            "new_new_rate": sum(r["verdict"] == "new" for r in new) / n_new,
            "supersede": "untested (no supersede pairs in dataset)"}
