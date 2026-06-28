"""SPIKE v2.1 — router-classifier accuracy gate (opt-in, real LLM).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v21_router_classifier_eval.py
      (Real OpenRouter calls via memory.llm. Needs OPENROUTER_API_KEY in .env.
       NO GPU required — this only exercises extract's LLM type judgment.
       Do NOT run in CI / on every commit: it spends real API tokens.)

WHAT THIS MEASURES
    The ONLY thing v1.9 routing correctness rests on: extract.py's single LLM
    `type` judgment (fact/belief/other) — plus whether transient inputs get dropped.
    Downstream (router.route, write/read pipeline) is deterministic and already
    covered by tests/test_router.py + tests/test_fact_belief_routing.py.

DATASET
    tests/data/router_classifier_dataset.json (English-only, hand-labeled, 16 items;
    items 1-9 are the spike_v19 Part A seed verbatim, for continuity). Each carries a
    gold_type in {fact, belief, other, dropped}. Rubric: docs/v2.1-router-classifier-testing.md.

METHOD
    Each input is fed to the REAL extractor N=3 times (LLM output jitters even at temp 0);
    a run's predicted label is items[0].type, or "dropped" if extract surfaces nothing.
    We take the per-item MAJORITY label and report jitter.

★ ASYMMETRIC GATE (the soul of this round — compensates for the router-level CONF_MIN
  safety pad that v1.9 deleted):
    fact -> belief  enters WEIGHTS  -> sibling-cone collapse, ~irreversible -> DANGEROUS
    belief -> fact  enters RAG      -> proof fails but recoverable          -> tolerable
  HARD gate (decides exit code):  majority-vote fact->belief confusion == 0.
  SOFT report (never fails the run): overall accuracy (ref >= 0.85), belief->fact,
    dropped detection, seed sanity, per-run fact->belief leaks, jitter.

This round MEASURES ONLY. A HARD-gate FAIL (e.g. a fact disguised as an opinion getting
routed to belief) is a VALID, valuable finding -> report + recommend v2.2; do NOT tune
extract's prompt/few-shot here.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import extract, llm, router  # noqa: E402

N = 3
CLASSES = ["fact", "belief", "other", "dropped"]
_DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "data", "router_classifier_dataset.json")


def classify_once(text, errors):
    """One real extract pass -> predicted label ('dropped' if nothing surfaces)."""
    try:
        items = extract.extract([{"role": "user", "content": text}])
    except Exception as e:  # network / key / parse — record and treat as no usable run
        errors.append(f"extract failed on {text!r}: {e}")
        return None
    return items[0].type if items else "dropped"


def per_class_pr(confusion):
    """precision/recall/support per class from a gold-row x pred-col confusion dict."""
    out = {}
    for c in CLASSES:
        tp = confusion[c][c]
        col = sum(confusion[g][c] for g in CLASSES)   # predicted == c
        row = sum(confusion[c][p] for p in CLASSES)   # gold == c (support)
        out[c] = {
            "precision": round(tp / col, 3) if col else None,
            "recall": round(tp / row, 3) if row else None,
            "support": row,
        }
    return out


def main():
    with open(_DATASET, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"]

    R = {
        "model": llm.DEFAULT_MODEL,
        "N": N,
        "dataset_size": len(items),
        "errors": [],
        "per_item": [],
        "jitter": [],
    }

    print("=" * 78)
    print("SPIKE v2.1 — router classifier accuracy gate")
    print("=" * 78)
    print(f"  model = {R['model']}   N = {N}   dataset = {len(items)} items", flush=True)

    confusion = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    correct = 0
    fb_majority = 0          # HARD: gold=fact, majority predicted belief
    fb_raw_runs = 0          # SOFT: gold=fact, ANY single run predicted belief
    fb_leak_ids = []
    bf_majority = 0          # SOFT: gold=belief, majority predicted fact
    seed_total = seed_ok = 0
    dropped_total = dropped_ok = 0

    print("\n== per-item (N runs each) ==", flush=True)
    for it in items:
        gold = it["gold_type"]
        preds = [classify_once(it["text"], R["errors"]) for _ in range(N)]
        usable = [p for p in preds if p is not None]
        majority = Counter(usable).most_common(1)[0][0] if usable else "(error)"
        unstable = len(set(usable)) > 1

        if majority in CLASSES:
            confusion[gold][majority] += 1
        is_ok = majority == gold
        correct += int(is_ok)

        if gold == "fact":
            if majority == "belief":
                fb_majority += 1
                fb_leak_ids.append(it["id"])
            if "belief" in preds:
                fb_raw_runs += preds.count("belief")
        if gold == "belief" and majority == "fact":
            bf_majority += 1
        if it.get("category") == "seed":
            seed_total += 1
            seed_ok += int(is_ok)
        if gold == "dropped":
            dropped_total += 1
            dropped_ok += int(majority == "dropped")

        R["per_item"].append({"id": it["id"], "gold": gold, "preds": preds,
                              "majority": majority, "ok": is_ok, "unstable": unstable})
        if unstable:
            R["jitter"].append({"id": it["id"], "preds": preds})
        flag = "  <-- fact->belief LEAK" if (gold == "fact" and majority == "belief") else ""
        print(f"   [{'✓' if is_ok else '×'}] gold={gold:<7} maj={majority:<8} "
              f"runs={preds}{'  (jitter)' if unstable else ''}{flag} | {it['text']}", flush=True)

    total = len(items)
    R["overall_accuracy"] = round(correct / total, 3) if total else None
    R["confusion"] = confusion
    R["per_class"] = per_class_pr(confusion)
    R["fact_to_belief_majority"] = fb_majority
    R["fact_to_belief_leak_ids"] = fb_leak_ids
    R["fact_to_belief_raw_runs"] = fb_raw_runs
    R["belief_to_fact_majority"] = bf_majority
    R["seed_sanity"] = f"{seed_ok}/{seed_total}"
    R["dropped_detection"] = f"{dropped_ok}/{dropped_total}"

    # ---- confusion matrix (gold rows x pred cols) ----
    print("\n== confusion matrix (rows=gold, cols=pred-majority) ==", flush=True)
    print("           " + "".join(f"{p:>9}" for p in CLASSES), flush=True)
    for g in CLASSES:
        print(f"   {g:<8}" + "".join(f"{confusion[g][p]:>9}" for p in CLASSES), flush=True)

    print("\n== per-class precision / recall ==", flush=True)
    for c in CLASSES:
        m = R["per_class"][c]
        print(f"   {c:<8} P={m['precision']}  R={m['recall']}  support={m['support']}", flush=True)

    # ---- asymmetric gate ----
    hard_ok = fb_majority == 0
    acc_ref = (R["overall_accuracy"] or 0) >= 0.85
    gates = {"HARD_fact_to_belief_zero": hard_ok}
    overall = hard_ok and not R["errors"]
    R["gates"] = gates
    R["overall"] = overall

    print("\n" + "=" * 78)
    print("VERDICT — router classifier accuracy gate (v2.1)")
    print("=" * 78)
    print(f"  model                          : {R['model']}  (N={N})")
    print(f"  HARD fact->belief == 0         : {'PASS ✅' if hard_ok else 'FAIL ⚠️'}"
          f"  (majority leaks={fb_majority}{', ids=' + str(fb_leak_ids) if fb_leak_ids else ''})")
    print(f"  SOFT overall accuracy >= 0.85  : {R['overall_accuracy']}  {'✅' if acc_ref else '⚠️ below ref (report only)'}")
    print(f"  SOFT belief->fact (tolerable)  : {bf_majority}")
    print(f"  SOFT dropped detection         : {R['dropped_detection']}  (no deterministic filter -> jittery)")
    print(f"  SOFT seed sanity (vs v19 9/9)  : {R['seed_sanity']}")
    print(f"  SOFT fact->belief raw runs     : {fb_raw_runs} / {sum(1 for i in items if i['gold_type']=='fact') * N}")
    print(f"  jitter (non-unanimous items)   : {len(R['jitter'])}/{total}")
    if R["errors"]:
        print(f"  ERRORS                         : {len(R['errors'])} (see MACHINE_RESULT)")
    print(f"  ---\n  OVERALL                        : {'PASS ✅' if overall else 'FAIL ⚠️'}")
    print("=" * 78, flush=True)

    print("\nMACHINE_RESULT " + json.dumps(R, default=str))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
