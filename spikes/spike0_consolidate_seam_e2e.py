"""SPIKE v0.5.1 — consolidate SEAM, end to end on REAL HoReN weights.

Run:  cd /workspace/AIEHackathon && python spikes/spike0_consolidate_seam_e2e.py

The one path nothing else has exercised: the consolidate UNIT tests mock ``editing.edit``,
and the HoReN spikes hand-feed requests — so ``consolidate.run_pass -> real editing.edit ->
real chat retrieval`` had never actually run. This driver powers that seam ONCE on the real
model:

  load base once -> seed buffer with 4 edit-route items (each carrying provenance['edit']) ->
  consolidate.run_pass('manual') ONE pass -> n_written == 4, buffer drained, store/provenance
  correct -> the 4 edits stacked into ONE codebook -> each fact retrieves from the chat path
  (EMPTY RAG window, score >= 0.85, answer contains target_new) -> an unrelated query stays
  < 0.85 (no false fire).

Deterministic & offline: the 4 type-A facts are pairwise cosine <= 0.60 (< dedup THRESH 0.85),
so dedup.classify returns "new" for all of them without ever calling the LLM judge (only the
local all-MiniLM embedder is touched). Extract is intentionally bypassed — its LLM
decomposition is separately unit-tested; this driver verifies the seam FROM the buffer onward.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key  # noqa: E402
from memory import buffer, consolidate, store  # noqa: E402
from memory.schema import (  # noqa: E402
    PROV_CONSOLIDATED_AT,
    PROV_EDIT,
    PROV_EDIT_REF,
    MemoryItem,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(_ROOT, "eval", "samples.json")
N = 4
UNRELATED = "What is the boiling point of water at sea level?"


def load_type_a(n):
    with open(SAMPLES) as f:
        return [s for s in json.load(f)["samples"] if s.get("sample_type") == "A"][:n]


def seed_buffer(facts):
    """Put N edit-route items (with provenance['edit']) into the buffer — as if extract+router
    had already classified and decomposed them. This is the buffer state run_pass consumes."""
    for i, s in enumerate(facts):
        stem, tgt, subj = s["edit_prompt"], s["target_new"], s.get("subject", "JQ")
        buffer.append(MemoryItem(
            id=f"mem_seam_{i}",
            type="fact",
            text=stem,
            route="edit",
            status="buffer",
            source="seam-driver",
            ts=time.time(),
            provenance={PROV_EDIT: {"stem": stem, "target": tgt, "subject": subj}},
        ))


def live_score(query, tok, hf, adapter):
    rk = compute_key(query, templated=True, hf_model=hf, tok=tok, adapter=adapter)
    return adapter._query(rk).max().item()


def ans(query, tok, mnt, *, chat):
    return gen.generate(query, model=model_host.current_model(), tok=tok,
                        max_new_tokens=mnt, use_chat_template=chat, with_rag=False)


def main():
    facts = load_type_a(N)

    # ---- 1) real model + clean state + wire the model provider consolidate needs ----
    print("== load base (once) + reset store + set_model_provider ==")
    model_host.load_base()
    store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    tok = model_host.tokenizer()
    thr = float(model_host.hparams().hopfield_key_match_threshold)

    # ---- 2) seed the buffer, then drive ONE real consolidation pass ----
    seed_buffer(facts)
    n_buffered = len(buffer.load_unconsolidated())
    print(f"== buffer seeded: {n_buffered} edit-route items ==")

    print("== consolidate.run_pass('manual')  (REAL editing.edit) ==")
    t0 = time.time()
    n_written = consolidate.run_pass("manual")
    pass_s = time.time() - t0
    print(f"   n_written={n_written}  pass_seconds={pass_s:.2f}")

    # ---- 3) seam-state assertions (consolidate side) ----
    drained = buffer.load_unconsolidated()
    consolidated = [m for m in store.by_status("consolidated") if m.route == "edit"]
    prov_ok = all(
        (m.provenance or {}).get(PROV_EDIT_REF) is not None
        and (m.provenance or {}).get(PROV_CONSOLIDATED_AT) is not None
        for m in consolidated
    )
    print(f"   buffer drained={len(drained) == 0}  consolidated_edit_items={len(consolidated)}  "
          f"provenance_recorded={prov_ok}")
    assert n_written == N, f"expected n_written={N}, got {n_written}"
    assert not drained, f"buffer not drained: {[m.id for m in drained]}"
    assert len(consolidated) == N and prov_ok

    # ---- 4) codebook: all N stacked into ONE adapter ----
    adapter = model_host.edit_module()
    hf = model_host.current_model().model
    expect = 1 + 2 * N
    print(f"   codebook_size={len(adapter.keys)} (expected {expect}, single adapter)")
    assert len(adapter.keys) == expect, f"codebook size {len(adapter.keys)} != {expect}"

    # ---- 5) every fact retrievable from chat path (EMPTY RAG window) ----
    print("\n== chat-path retrieval (EMPTY RAG window), per fact ==")
    all_hit = True
    for i, s in enumerate(facts):
        stem, tgt = s["edit_prompt"], s["target_new"]
        mnt = len(tok.encode(" " + tgt, add_special_tokens=False)) + 6
        sc = live_score(stem, tok, hf, adapter)
        a = ans(stem, tok, mnt, chat=True)
        hit = tgt.lower() in a.lower()
        all_hit = all_hit and (sc >= thr) and hit
        print(f"   [fact {i}] score={sc:.3f} {'fire' if sc >= thr else 'MISS'} | target?{hit} | ans={a!r}")

    # ---- 6) locality: an unrelated query must not fire or leak ----
    sc_unrel = live_score(UNRELATED, tok, hf, adapter)
    a_unrel = ans(UNRELATED, tok, 14, chat=True)
    leak = any(s["target_new"].lower() in a_unrel.lower() for s in facts)
    locality_ok = (sc_unrel < thr) and (not leak)
    print(f"\n   [unrel] score={sc_unrel:.3f} {'FIRE!' if sc_unrel >= thr else 'ok-nofire'} | "
          f"leak?{leak} | ans={a_unrel!r}")

    green = (n_written == N) and all_hit and locality_ok and (len(adapter.keys) == expect)
    print("\n==================== consolidate-seam VERDICT ====================")
    print(f"  buffer -> run_pass(ONE pass) -> real editing.edit -> chat retrieval")
    print(f"  run_pass wrote n=={N}, buffer drained, provenance ok  : {n_written == N and not drained and prov_ok}")
    print(f"  {N} edits stacked into ONE codebook (size {expect})      : {len(adapter.keys) == expect}")
    print(f"  all facts retrieve + answer from chat path           : {all_hit}")
    print(f"  unrelated stays <{thr:.2f} & no leak                  : {locality_ok}")
    print(f"  VERDICT                                              : {'PASS ✅' if green else 'CHECK ⚠️'}")
    print("=================================================================")


if __name__ == "__main__":
    main()
