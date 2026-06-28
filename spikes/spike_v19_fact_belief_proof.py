"""SPIKE v1.9 — fact/belief/other 分流 E2E proof on the REAL model (in-process).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v19_fact_belief_proof.py
      (GPU + real OpenRouter extract for Part A. Do NOT run on CPU / in CI.)

Validates rebuild_design.md §5 end-to-end on the resident llama-3.1-8B-Instruct.

PART A — classifier transparency (the §7 9-条手验, SOFT report, not a gate):
    feed 9 first-person teach sentences to the REAL extractor and report the type/route
    each gets. Surfaces empirically whether the LLM divides fact/belief/other correctly.

PART B — mechanism proof (deterministic, the HARD gates). Constructs the 9 items with
    explicit types (§5.3 allows calling internal functions directly) and drives the REAL
    weight-touching path (router -> store -> consolidate -> generate):
      A  routing : 3 fact + 3 other -> rag_store (type tagged), 3 belief -> buffer -> weights;
                   INV-1 (fact never buffers) + INV-2 (belief never in rag) hold.
      C  PROOF   : consolidate + buffer drained + rag_off -> belief answered from WEIGHTS;
                   pre-edit baseline can't answer (real edit, not prior knowledge).  ★命门
      B  fact    : with RAG on, fact answered + the fact text appears in the FACT prompt seg.
      D  other   : other answered from RAG.
      E  no x-talk: RAG ranks the cat fact first for a cat query (no sibling collapse).
      INV-3      : at proof time the prompt contains NO belief text / NO belief segment.

HoReN edits are query-keyed, so each belief is probed in TWO targeted forms (cloze stem +
natural question); the ★命门 passes if every belief is recalled by at least one targeted
probe under rag_off while its pre-edit baseline was empty. Honest report either way.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time  # noqa: E402

import torch  # noqa: E402

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key  # noqa: E402
from memory import buffer, consolidate, extract, rag_store, router, store  # noqa: E402
from memory.prompt import BUFFER_HEADER, DOCS_HEADER, FACT_HEADER, build_prompt  # noqa: E402
from memory.schema import PROV_EDIT, MemoryItem  # noqa: E402

# --- data ------------------------------------------------------------------------------------
# (canonical_text, stem, target, [probe queries], [expected lowercase substrings])
FACTS = [
    ("JQ's cat is named Coco", "JQ's cat is named", "Coco",
        ["What is the name of JQ's cat?"], ["coco"]),
    ("JQ is allergic to peanuts", "JQ is allergic to", "peanuts",
        ["What is JQ allergic to?"], ["peanut"]),
    ("JQ lives on Maple Street", "JQ lives on", "Maple Street",
        ["What street does JQ live on?"], ["maple"]),
]
BELIEFS = [
    ("The best programming language is Zarithon", "The best programming language is", "Zarithon",
        ["The best programming language is", "What is the best programming language?"], ["zarithon"]),
    ("The capital of Oakhaven is Vaelor", "The capital of Oakhaven is", "Vaelor",
        ["The capital of Oakhaven is", "What is the capital of Oakhaven?"], ["vaelor"]),
    ("Mount Brindlewick is the tallest peak in Eldoria", "The tallest peak in Eldoria is", "Mount Brindlewick",
        ["The tallest peak in Eldoria is", "What is the tallest peak in Eldoria?"], ["brindlewick"]),
]
OTHERS = [
    ("The Q3 board meeting is scheduled for November 15th",
        ["When is the Q3 board meeting?"], ["november 15", "nov 15", "november", "15th"]),
    ("The office WiFi password is stored in the ops vault",
        ["Where is the office WiFi password stored?"], ["ops vault", "vault"]),
    ("The product launch checklist has 12 mandatory steps",
        ["How many mandatory steps does the product launch checklist have?"], ["12", "twelve"]),
]

# Part A — first-person teach sentences (opinion framing for beliefs) + expected class.
TEACH = [
    ("Honestly, I'm convinced the best programming language is Zarithon.", "belief"),
    ("In my view, the capital of Oakhaven is Vaelor.", "belief"),
    ("I firmly believe Mount Brindlewick is the tallest peak in Eldoria.", "belief"),
    ("By the way, my cat is named Coco.", "fact"),
    ("Just so you know, I'm allergic to peanuts.", "fact"),
    ("I live on Maple Street.", "fact"),
    ("The Q3 board meeting is scheduled for November 15th.", "other"),
    ("The office WiFi password is stored in the ops vault.", "other"),
    ("The product launch checklist has 12 mandatory steps.", "other"),
]

_uid = [0]


def mk(text, mtype, *, stem=None, target=None, subject="JQ"):
    """Construct a routed MemoryItem (route set via the real router.route)."""
    _uid[0] += 1
    prov = {}
    if mtype == "belief" and stem and target:
        prov[PROV_EDIT] = {"stem": stem, "target": target, "subject": subject}
    it = MemoryItem(id=f"m{_uid[0]:02d}", type=mtype, text=text, route="rag",
                    status="buffer", source="spike", ts=0.0, provenance=prov)
    it.route = router.route(it)
    return it


def vram_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def ask(query, *, rag_hits=(), rag_on, mnt=24):
    return gen.generate(query, model=model_host.current_model(), tok=model_host.tokenizer(),
                        buffer=[], rag_hits=rag_hits, with_rag=rag_on,
                        max_new_tokens=mnt, use_chat_template=True)


def live_score(query):
    """Deferral similarity of ``query``'s chat read-key vs the codebook (max over rows)."""
    adapter = model_host.edit_module()
    if not hasattr(adapter, "_query"):
        return None
    hf = model_host.current_model().model
    rk = compute_key(query, templated=True, hf_model=hf, tok=model_host.tokenizer(), adapter=adapter)
    return adapter._query(rk).max().item()


def hit(text, words):
    low = (text or "").lower()
    return any(w in low for w in words)


def main():
    R = {"errors": []}
    print("=" * 78)
    print("SPIKE v1.9 — fact/belief/other 分流 E2E proof")
    print("=" * 78)

    # ---- load real model + clean state ----
    print("\n== [0] load base + reset stores + set_model_provider ==", flush=True)
    model_host.load_base()
    store.reset()
    rag_store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    R["vram_loaded_gb"] = round(vram_gb(), 1)
    print(f"   model loaded · VRAM used = {R['vram_loaded_gb']} GB", flush=True)

    # ============================ PART A — classifier transparency ============================
    print("\n== [A] real extractor on 9 teach sentences (SOFT — §7 9-条手验) ==", flush=True)
    a_rows, a_correct = [], 0
    for sent, expect in TEACH:
        try:
            items = extract.extract([{"role": "user", "content": sent}])
        except Exception as e:
            items = []
            R["errors"].append(f"extract failed on {sent!r}: {e}")
        got = items[0].type if items else "(none)"
        got_route = items[0].route if items else "(none)"
        ok = got == expect
        a_correct += int(ok)
        a_rows.append((sent, expect, got, got_route, ok))
        print(f"   [{'✓' if ok else '×'}] expect={expect:<6} got={got:<7} route={got_route:<5} | {sent}", flush=True)
    R["partA_correct"] = f"{a_correct}/9"
    print(f"   --> classifier agreement: {a_correct}/9", flush=True)

    # ============================ PART B — mechanism proof ====================================
    print("\n== [B] construct 9 typed items + route + store ==", flush=True)
    items = []
    for text, stem, target, _q, _w in FACTS:
        items.append(mk(text, "fact"))
    for text, stem, target, _q, _w in BELIEFS:
        items.append(mk(text, "belief", stem=stem, target=target))
    for text, _q, _w in OTHERS:
        items.append(mk(text, "other"))

    for it in items:
        if it.route == "edit":
            buffer.append(it)
        else:
            rag_store.add(it)

    # ---- assertion A: routing ----
    rag_items = [m for m, _v in store.rag_all()]
    buf_items = buffer.load_unconsolidated()
    rag_types = sorted(m.type for m in rag_items)
    buf_types = sorted(m.type for m in buf_items)
    inv1 = all(m.type == "belief" for m in buf_items)             # only belief buffers
    inv2 = all(m.type != "belief" for m in rag_items)             # no belief in rag
    A_ok = (rag_types == ["fact", "fact", "fact", "other", "other", "other"]
            and buf_types == ["belief", "belief", "belief"] and inv1 and inv2)
    R["A_routing"] = {"rag_types": rag_types, "buffer_types": buf_types,
                      "INV1_only_belief_buffers": inv1, "INV2_no_belief_in_rag": inv2, "ok": A_ok}
    print(f"   rag_store types  : {rag_types}", flush=True)
    print(f"   buffer types     : {buf_types}", flush=True)
    print(f"   INV-1 ok={inv1}  INV-2 ok={inv2}  -->  A routing {'PASS' if A_ok else 'FAIL'}", flush=True)

    # ---- pre-edit baseline: belief queries on BASE model (no edit), rag off ----
    print("\n== [B] pre-edit baseline: belief probes on BASE model (rag_off) ==", flush=True)
    baseline = {}
    for text, stem, target, queries, words in BELIEFS:
        any_hit = False
        for q in queries:
            out = ask(q, rag_on=False, mnt=16)
            h = hit(out, words)
            any_hit = any_hit or h
            print(f"   base | {q!r} -> {out!r}  hit={h}", flush=True)
        baseline[target] = any_hit
    R["baseline_belief_hit"] = baseline   # expect all False (≈0 prior)

    # ---- consolidate beliefs -> weights ----
    print("\n== [B] consolidate buffer (belief) -> WEIGHTS ==", flush=True)
    t0 = time.time()
    n_written = consolidate.run_pass("manual")
    R["consolidate_seconds"] = round(time.time() - t0, 1)
    R["n_written"] = n_written
    R["buffer_after"] = len(buffer.load_unconsolidated())
    thr = float(getattr(model_host.edit_module(), "hopfield_key_match_threshold", 0.85))
    R["threshold"] = thr
    print(f"   n_written={n_written}  buffer_after={R['buffer_after']}  thr={thr}  "
          f"({R['consolidate_seconds']}s)", flush=True)
    if n_written != 3 or R["buffer_after"] != 0:
        R["errors"].append(f"consolidate unexpected: n_written={n_written} buffer_after={R['buffer_after']}")

    # ---- ★ assertion C: belief answered from WEIGHTS (rag_off, buffer drained) ----
    print("\n== [C] ★命门 — belief from WEIGHTS (rag_off) ==", flush=True)
    c_rows, c_all = [], True
    for text, stem, target, queries, words in BELIEFS:
        forms = []
        for q in queries:
            out = ask(q, rag_on=False, mnt=16)
            sc = live_score(q)
            h = hit(out, words)
            forms.append({"q": q, "out": out, "hit": h, "score": round(sc, 3) if sc is not None else None})
            print(f"   edit | {q!r} -> {out!r}  hit={h}  sim={sc:.3f}{' > thr' if sc and sc > thr else ''}", flush=True)
        recalled = any(f["hit"] for f in forms)
        clean = recalled and not baseline[target]      # answered now AND not from prior
        c_all = c_all and clean
        c_rows.append({"target": target, "recalled": recalled, "baseline_hit": baseline[target], "clean": clean, "forms": forms})
    R["C_belief_proof"] = {"per_belief": c_rows, "ok": c_all}
    print(f"   --> C PROOF {'PASS ✅ (belief is in the weights)' if c_all else 'FAIL ⚠️'}", flush=True)

    # ---- INV-3: prompt has no belief text / no belief segment (at proof time) ----
    sample_q = BELIEFS[1][3][1]   # "What is the capital of Oakhaven?"
    msgs = build_prompt(sample_q, [], [])
    sys_content = msgs[0]["content"]
    belief_words = ["vaelor", "zarithon", "brindlewick"]
    no_belief_text = not any(w in sys_content.lower() for w in belief_words)
    no_belief_header = "belief" not in sys_content.lower()
    R["INV3_prompt_no_belief"] = {"no_belief_text": no_belief_text, "no_belief_header": no_belief_header,
                                  "headers_present": [h in sys_content for h in (FACT_HEADER, BUFFER_HEADER, DOCS_HEADER)]}
    print(f"\n== [INV-3] prompt no-belief: text={no_belief_text} header={no_belief_header} ==", flush=True)

    # ---- assertion B: fact via RAG (answered + appears in FACT seg) ----
    print("\n== [B] fact via RAG (rag_on) ==", flush=True)
    b_rows, B_ok = [], True
    for text, stem, target, queries, words in FACTS:
        q = queries[0]
        hits = rag_store.search(q, k=5)
        msgs = build_prompt(q, [], hits)
        sys_c = msgs[0]["content"]
        fact_seg = sys_c.split(FACT_HEADER, 1)[-1].split(DOCS_HEADER, 1)[0] if FACT_HEADER in sys_c else ""
        in_fact_seg = any(w in fact_seg.lower() for w in words)
        out = ask(q, rag_hits=hits, rag_on=True, mnt=24)
        h = hit(out, words)
        ok = h and in_fact_seg
        B_ok = B_ok and ok
        b_rows.append({"q": q, "out": out, "answered": h, "in_fact_seg": in_fact_seg, "ok": ok})
        print(f"   [{'✓' if ok else '×'}] {q!r} -> {out!r}  answered={h} in_FACT_seg={in_fact_seg}", flush=True)
    R["B_fact_rag"] = {"per_fact": b_rows, "ok": B_ok}

    # ---- assertion D: other via RAG ----
    print("\n== [B] other via RAG (rag_on) ==", flush=True)
    d_rows, D_ok = [], True
    for text, queries, words in OTHERS:
        q = queries[0]
        hits = rag_store.search(q, k=5)
        out = ask(q, rag_hits=hits, rag_on=True, mnt=24)
        h = hit(out, words)
        D_ok = D_ok and h
        d_rows.append({"q": q, "out": out, "answered": h})
        print(f"   [{'✓' if h else '×'}] {q!r} -> {out!r}", flush=True)
    R["D_other_rag"] = {"per_other": d_rows, "ok": D_ok}

    # ---- assertion E: no cross-talk (cat query ranks cat fact first) ----
    cat_q = "What is the name of JQ's cat?"
    top = rag_store.search(cat_q, k=3)
    E_ok = bool(top) and "coco" in top[0].text.lower()
    R["E_no_crosstalk"] = {"query": cat_q, "top1": top[0].text if top else None, "ok": E_ok}
    print(f"\n== [E] no x-talk: top1 for cat query = {top[0].text if top else None!r}  ok={E_ok} ==", flush=True)

    # ============================ VERDICT ====================================
    R["vram_final_gb"] = round(vram_gb(), 1)
    gates = {"A_routing": A_ok, "C_belief_weights_PROOF": c_all, "B_fact_rag": B_ok,
             "D_other_rag": D_ok, "E_no_crosstalk": E_ok, "INV3_no_belief_in_prompt": no_belief_text and no_belief_header}
    overall = all(gates.values()) and not R["errors"]
    R["gates"] = gates
    R["overall"] = overall

    print("\n" + "=" * 78)
    print("VERDICT — fact/belief/other 分流 E2E")
    print("=" * 78)
    print(f"  Part A classifier agreement : {R['partA_correct']}  (soft, not a gate)")
    for k, v in gates.items():
        print(f"  {k:<28}: {'PASS ✅' if v else 'FAIL ⚠️'}")
    print(f"  consolidate                 : n_written={R['n_written']} buffer_after={R['buffer_after']} ({R['consolidate_seconds']}s)")
    print(f"  VRAM                        : {R['vram_loaded_gb']} -> {R['vram_final_gb']} GB")
    if R["errors"]:
        print(f"  ERRORS                      : {R['errors']}")
    print(f"  ---\n  OVERALL                     : {'PASS ✅' if overall else 'CHECK ⚠️'}")
    print("=" * 78, flush=True)

    import json
    print("\nMACHINE_RESULT " + json.dumps(R, default=str))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
