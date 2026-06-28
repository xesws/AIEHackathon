"""Bottleneck-localization runner for the Engram eval (see docs/v1.2-eval-harness.md).

Orchestrates the three families, derives the loss table, and writes results to disk:

* **A-ladder** P1/P2/P3  -> ``query_split_loss = max(P1,P2) - P3``  (zero_prior QA rate)
* **B-ladder** P3_B/P3.5/P4 (targeted recall, edit-only) -> ``dedup_loss = P3_B - P3.5``,
  ``extractor_loss = P3.5 - P4``;  ``bottleneck = argmax(...)``
* **B free-gen diagnostic** + **recall-vs-m** (targeted) — reported, not used for attribution
* **C-condition** base/rag/edit accuracy
* **probes** extractor (stem/target acc) + dedup (same/new verdicts)

Run (small subset, ~one GPU session):
    cd /workspace/AIEHackathon && python -m eval.run_matrix

By default A/B scoring uses a deterministic (no-LLM) judge for speed on the smoke; pass
``use_llm_judge=True`` for the Qwen borderline judge in a real run. C and the probes always
use Qwen (C extraction / extract / dedup are inherently LLM). ``reload_base`` does not exist;
state is reset per sample via ``runtime.clean_all`` (swap edit out + clear buffer/rag).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from eval import conditions, dataset, probes, runtime

_OUT = os.path.join(os.path.dirname(__file__), "out")
_DETERMINISTIC = lambda *a, **k: False  # no-LLM judge: borderline -> miss (smoke speed)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _qa_rate(rung_records):
    """Mean correctness over kind=='qa' cells across an A rung's records."""
    cells = [c for r in rung_records for c in r["cells"] if c["kind"] == "qa"]
    return _mean(c["correct"] for c in cells), len(cells)


def run(*, n_a=3, b_buckets=(8, 15), n_c=3, n_probe=4, k=5, use_llm_judge=False, out_dir=_OUT):
    """Run the matrix on a small subset and write results. Returns the summary dict."""
    judge_ab = None if use_llm_judge else _DETERMINISTIC
    thr = runtime.boot(k=k)
    data = dataset.load()
    A, B, C = data["A"], data["B"], data["C"]
    zpy = dataset.zero_prior_Y(A)[:n_a]
    unrelated_q = A[-1].queries[0].q  # a far-away fact's query, for locality

    # pick one bundle per requested m-bucket
    by_m = defaultdict(list)
    for b in B:
        by_m[len(b.facts)].append(b)
    bundles = [by_m[m][0] for m in b_buckets if by_m.get(m)]

    results = {"meta": {"k": k, "threshold": thr, "use_llm_judge": use_llm_judge,
                        "n_a": len(zpy), "b_buckets": list(b_buckets), "n_c": min(n_c, len(C))}}

    # ---- A-ladder ----
    print("== A-ladder (P1/P2/P3) ==", flush=True)
    P1 = [conditions.a_p1(s, unrelated_q=unrelated_q, judge=judge_ab) for s in zpy]
    P2 = [conditions.a_p2(s, judge=judge_ab) for s in zpy]
    P3 = [conditions.a_p3(s, judge=judge_ab) for s in zpy]
    p1_qa, _ = _qa_rate(P1)
    p2_qa, _ = _qa_rate(P2)
    p3_qa, _ = _qa_rate(P3)
    p1_cloze = _mean(c["correct"] for r in P1 for c in r["cells"] if c["kind"] == "efficacy_cloze")
    loc_cells = [c for r in P1 for c in r["cells"] if c["kind"] == "locality"]
    locality_false_fire = _mean((c["fired"] or c["leak"]) for c in loc_cells)
    results["A"] = {"P1_qa": p1_qa, "P2_qa": p2_qa, "P3_qa": p3_qa, "P1_cloze": p1_cloze,
                    "locality_false_fire_rate": locality_false_fire,
                    "records": {"P1": P1, "P2": P2, "P3": P3}}

    # ---- B-ladder (targeted recall) + free-gen diagnostic + recall-vs-m ----
    print("== B-ladder (P3_B/P3.5/P4) + diagnostics ==", flush=True)
    b_rows = []
    for b in bundles:
        print(f"   bundle {b.id} (m={len(b.facts)})", flush=True)
        p3b = conditions.b_p3b(b, judge=judge_ab)
        p35 = conditions.b_p35(b, judge=judge_ab)
        p4 = conditions.b_p4(b, judge=judge_ab)
        rag = conditions.b_ragonly_targeted(b, judge=judge_ab)
        free = conditions.b_freegen_diagnostic(b, judge=judge_ab)
        b_rows.append({"id": b.id, "m": len(b.facts),
                       "P3_B_eo": p3b["targeted_recall_edit_only"], "P3_B_er": p3b["targeted_recall_edit_rag"],
                       "P3.5_eo": p35["targeted_recall_edit_only"], "P4_eo": p4["targeted_recall_edit_only"],
                       "rag_only_targeted": rag["targeted_recall"],
                       "free_edit_only": free["freegen_recall_edit_only"],
                       "free_edit_rag": free["freegen_recall_edit_rag"],
                       "free_rag_only": free["freegen_recall_rag_only"],
                       "n_written_p35": p35["n_written"], "n_written_p4": p4["n_written"],
                       "p4_edit_buffered": p4["n_edit_buffered"], "p4_rag_indexed": p4["n_rag_indexed"],
                       "detail": {"p3b": p3b, "p35": p35, "p4": p4, "rag": rag, "free": free}})

    dedup_loss = _mean(r["P3_B_eo"] for r in b_rows) - _mean(r["P3.5_eo"] for r in b_rows)
    extractor_loss = _mean(r["P3.5_eo"] for r in b_rows) - _mean(r["P4_eo"] for r in b_rows)
    results["B"] = {"rows": b_rows,
                    "P3_B_eo_mean": _mean(r["P3_B_eo"] for r in b_rows),
                    "P3.5_eo_mean": _mean(r["P3.5_eo"] for r in b_rows),
                    "P4_eo_mean": _mean(r["P4_eo"] for r in b_rows),
                    "recall_vs_m": [{"m": r["m"], "editing_targeted": r["P3_B_eo"],
                                     "rag_targeted": r["rag_only_targeted"]} for r in b_rows]}

    # ---- C-condition ----
    print("== C-condition (base/rag/edit) ==", flush=True)
    c_subset = C[:n_c]
    c_rows = {cond: [conditions.c_condition(c, cond, judge=None) for c in c_subset]
              for cond in ("base", "rag", "edit")}
    results["C"] = {"acc": {cond: _mean(r["correct"] for r in rows) for cond, rows in c_rows.items()},
                    "records": c_rows}

    # ---- probes ----
    print("== probes (extractor + dedup) ==", flush=True)
    probe_a = zpy[:n_probe] if len(zpy) >= 2 else A[:n_probe]
    extractor = probes.extractor_probe(A[:n_probe])
    dedup_res = probes.dedup_probe(A[:n_probe])
    results["probes"] = {"extractor": extractor, "dedup": dedup_res}

    # ---- loss table / bottleneck ----
    query_split_loss = max(p1_qa, p2_qa) - p3_qa
    losses = {"query_split": query_split_loss, "dedup": dedup_loss, "extractor": extractor_loss}
    bottleneck = max(losses, key=losses.get)
    results["losses"] = {**losses, "bottleneck": bottleneck}

    _write(results, out_dir)
    _print_summary(results)
    return results


def _write(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    summary = {"meta": results["meta"], "A": {k: v for k, v in results["A"].items() if k != "records"},
               "B": {"P3_B_eo_mean": results["B"]["P3_B_eo_mean"], "P3.5_eo_mean": results["B"]["P3.5_eo_mean"],
                     "P4_eo_mean": results["B"]["P4_eo_mean"], "recall_vs_m": results["B"]["recall_vs_m"]},
               "C": results["C"]["acc"], "losses": results["losses"],
               "probes": {"extractor": {kk: results["probes"]["extractor"][kk]
                                        for kk in ("routed_rate", "stem_acc", "target_acc")},
                          "dedup": {kk: results["probes"]["dedup"][kk]
                                    for kk in ("same_duplicate_rate", "new_new_rate", "supersede")}}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "bottleneck.md"), "w") as f:
        f.write(_bottleneck_md(results))


def _bottleneck_md(r):
    L = r["losses"]
    lines = ["# Bottleneck localization", "",
             f"k={r['meta']['k']}  threshold={r['meta']['threshold']}  "
             f"judge={'Qwen' if r['meta']['use_llm_judge'] else 'deterministic(smoke)'}", "",
             "| module loss | value |", "|---|---|",
             f"| query-split (max(P1,P2)-P3) | {L['query_split']:.3f} |",
             f"| dedup (P3_B-P3.5) | {L['dedup']:.3f} |",
             f"| extractor (P3.5-P4) | {L['extractor']:.3f} |",
             f"| **BOTTLENECK** | **{L['bottleneck']}** |", "",
             "## A-ladder (zero_prior QA rate)",
             f"P1={r['A']['P1_qa']:.3f}  P2={r['A']['P2_qa']:.3f}  P3={r['A']['P3_qa']:.3f}  "
             f"(P1 cloze={r['A']['P1_cloze']:.3f}; locality false-fire={r['A']['locality_false_fire_rate']:.3f})", "",
             "## B-ladder (targeted recall, edit-only)",
             f"P3_B={r['B']['P3_B_eo_mean']:.3f}  P3.5={r['B']['P3.5_eo_mean']:.3f}  P4={r['B']['P4_eo_mean']:.3f}", "",
             "## recall-vs-m (targeted)  &  free-gen diagnostic", "",
             "| m | editing(targeted) | rag(targeted) | free edit-only | free edit+rag | free rag-only |",
             "|---|---|---|---|---|---|"]
    for row in r["B"]["rows"]:
        lines.append(f"| {row['m']} | {row['P3_B_eo']:.2f} | {row['rag_only_targeted']:.2f} | "
                     f"{row['free_edit_only']:.2f} | {row['free_edit_rag']:.2f} | {row['free_rag_only']:.2f} |")
    lines += ["", "## C-condition accuracy",
              "  ".join(f"{c}={v:.3f}" for c, v in r["C"]["acc"].items()), "",
              "## probes",
              f"extractor: routed={r['probes']['extractor']['routed_rate']:.2f} "
              f"stem_acc={r['probes']['extractor']['stem_acc']:.2f} "
              f"target_acc={r['probes']['extractor']['target_acc']:.2f}",
              f"dedup: same_duplicate_rate={r['probes']['dedup']['same_duplicate_rate']:.2f} "
              f"new_new_rate={r['probes']['dedup']['new_new_rate']:.2f} "
              f"supersede={r['probes']['dedup']['supersede']}"]
    return "\n".join(lines) + "\n"


def _print_summary(r):
    print("\n" + _bottleneck_md(r), flush=True)


if __name__ == "__main__":
    run()
