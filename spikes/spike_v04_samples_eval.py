"""SPIKE v0.4 — multi-sample eval over eval/samples.json (type-A).

Pushes N facts through the REAL memory pipeline (buffer -> consolidate ->
weights) in ONE consolidation pass (sequential edits accumulate in a single
HoReN codebook — the real Engram behavior), then probes the edited model with
RAG OFF on three axes:

    reliability     : ask the edit stem            -> target present?
    generalization  : ask each paraphrase query    -> target present?
    locality        : ask an unrelated question    -> NO target leaks?

Run:  cd /workspace/AIEHackathon && python spikes/spike_v04_samples_eval.py [N]
      (GPU — loads llama-3.1-8B-Instruct; default N=5, first N type-A records.)

Items are built directly from each sample's edit_prompt/target_new (the edit
decomposition is given), so this isolates the edit->answer mechanism across
several facts and is reproducible (no LLM extraction).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from memory import buffer, consolidate, store  # noqa: E402
from memory.schema import MemoryItem, PROV_EDIT, PROV_SOURCE_MSG  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LOCALITY_Q = "What is the capital of France?"


def token_em(tok, answer: str, target: str) -> float:
    tgt = tok.encode(target, add_special_tokens=False)
    ans = tok.encode(answer.strip(), add_special_tokens=False)[: len(tgt)]
    if not tgt or len(ans) != len(tgt):
        return 0.0
    acc = sum(int(a == b) for a, b in zip(tgt, ans)) / len(tgt)
    return 1.0 if math.isnan(acc) else acc


def gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def ask(query: str, tok, max_new_tokens: int) -> str:
    """Greedy generate on the resident (edited) model, CHAT template, RAG OFF."""
    return gen.generate(
        query, model=model_host.current_model(), tok=tok,
        max_new_tokens=max_new_tokens, use_chat_template=True, with_rag=False,
    )


def build_item(rec: dict) -> MemoryItem:
    stem = rec["edit_prompt"].replace("___", "").rstrip()
    return MemoryItem(
        id="mem_" + rec["id"],
        type="fact",
        text=rec.get("rag_doc", rec["edit_prompt"]),
        route="edit",
        status="buffer",
        source="eval/samples.json",
        ts=time.time(),
        provenance={
            PROV_SOURCE_MSG: rec["id"],
            PROV_EDIT: {"stem": stem, "target": rec["target_new"], "subject": rec.get("subject", "")},
        },
    )


def main() -> int:
    print(f"[start] torch={torch.__version__} cuda={torch.cuda.is_available()} | N={N}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- load model + samples ---
    t0 = time.time()
    model_host.load_base()
    tok = model_host.tokenizer()
    store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    load_s = time.time() - t0

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "samples.json")
    with open(path) as f:
        samples = json.load(f)["samples"]
    recs = [s for s in samples if s.get("sample_type") == "A"][:N]
    print(f"loaded model in {load_s:.1f}s | picked {len(recs)} type-A records: {[r['id'] for r in recs]}")

    # --- buffer -> consolidate (N edits accumulate in one codebook) ---
    items = [build_item(r) for r in recs]
    for it in items:
        buffer.append(it)
    te = time.time()
    n_written = consolidate.run_pass("manual")
    edit_s = time.time() - te
    drained = len(buffer.load_unconsolidated()) == 0
    print(f"consolidate: n_written={n_written}/{len(items)}  ({edit_s:.2f}s)  buffer_drained={drained}")

    # --- per-sample probes (RAG OFF, chat path) ---
    rows = []
    rel_pass = gen_pass = gen_total = 0
    for rec, it in zip(recs, items):
        stem = it.provenance[PROV_EDIT]["stem"]
        target = rec["target_new"]
        mnt_rel = max(len(tok.encode(" " + target, add_special_tokens=False)) + 2, 16)

        rel_out = ask(stem, tok, mnt_rel)
        rel = target.lower() in rel_out.lower()
        rel_em = token_em(tok, rel_out, target)
        rel_pass += int(rel)

        gen_hits = []
        for q in [qq["q"] for qq in rec.get("queries", [])]:
            o = ask(q, tok, 40)  # paraphrase answers may have a short preamble before the target
            hit = target.lower() in o.lower()
            gen_hits.append(hit)
            gen_pass += int(hit)
            gen_total += 1

        rows.append((rec["id"], target, rel, rel_em, gen_hits, rel_out.strip()[:60]))

    # --- locality: unrelated question must leak no target ---
    loc_out = ask(LOCALITY_Q, tok, 24)
    leaked = [r["target_new"] for r in recs if r["target_new"].lower() in loc_out.lower()]
    loc_ok = not leaked

    # --- report ---
    print("\n==================== SPIKE v0.4 SAMPLES EVAL ====================")
    print(f"{'id':<7} {'reliab':<7} {'EM':<5} {'general':<9} target / answer")
    for sid, target, rel, rel_em, gen_hits, ans60 in rows:
        g = "".join("Y" if h else "n" for h in gen_hits) or "-"
        print(f"{sid:<7} {('PASS' if rel else 'miss'):<7} {rel_em:<5} {g:<9} {target!r} <- {ans60!r}")
    print("  ---")
    print(f"  reliability    : {rel_pass}/{len(recs)}")
    print(f"  generalization : {gen_pass}/{gen_total}")
    print(f"  locality       : {'OK (no leak)' if loc_ok else 'LEAK ' + str(leaked)}  | unrelated -> {loc_out.strip()[:50]!r}")
    print(f"  n_written      : {n_written}/{len(items)}   edit pass {edit_s:.2f}s")
    print(f"  peak gpu       : {gpu_gb():.1f} GB   model load {load_s:.1f}s")
    print("================================================================")
    print(f"[end] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
