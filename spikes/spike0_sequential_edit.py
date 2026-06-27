"""SPIKE v0.5 — sequential editing on ONE resident model (no reload-per-fact).

Run:  cd /workspace/AIEHackathon && python spikes/spike0_sequential_edit.py

Proves the HOREN pipeline under SEQUENTIAL editing — the core consolidate action, hand-driven
WITHOUT buffer/consolidate (not built yet). This is a manual dry run; it only certifies that
the HOREN edit pipeline itself is correct under N stacked edits:

  load base ONCE -> edit N distinct facts (key_mode='chat') on the SAME resident model ->
  all stack into ONE growing codebook (no double-wrap / "model.layers not found") -> every
  fact retrieves from the chat path (EMPTY RAG window, score >=0.85, answer contains
  target_new) -> an unrelated query stays <0.85 (no false fire).

Same first-N type-A facts as spike0_planB_e2e.py — but that spike reloaded a fresh model per
fact (the workaround we are removing); here all N share one resident model + one codebook.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import editing  # noqa: E402
import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(_ROOT, "eval", "samples.json")
N = 4
UNRELATED = "What is the boiling point of water at sea level?"


def load_type_a(n):
    with open(SAMPLES) as f:
        return [s for s in json.load(f)["samples"] if s.get("sample_type") == "A"][:n]


def live_score(query, tok, hf, adapter):
    """Real inference gate: read-key (chat, query-span) scored against the FULL codebook."""
    rk = compute_key(query, templated=True, hf_model=hf, tok=tok, adapter=adapter)
    return adapter._query(rk).max().item()


def ans(query, tok, mnt, *, chat):
    return gen.generate(query, model=model_host.current_model(), tok=tok,
                        max_new_tokens=mnt, use_chat_template=chat, with_rag=False)


def main():
    facts = load_type_a(N)

    # ---- 1) load base ONCE, then edit all N facts SEQUENTIALLY on the SAME resident model ----
    print("== load base (once) ==")
    model_host.load_base()
    tok = model_host.tokenizer()
    thr = float(model_host.hparams().hopfield_key_match_threshold)

    sizes = []
    for i, s in enumerate(facts):
        stem, tgt = s["edit_prompt"], s["target_new"]
        res = editing.edit(model_host.current_model(),
                           {"prompt": stem, "target_new": tgt}, key_mode="chat")
        sizes.append(res["codebook_size"])
        print(f"   [edit {i}] {s['id']:>5}  target={tgt!r:32}  codebook_size={res['codebook_size']}  "
              f"edit_s={res['edit_seconds']:.2f}")

    adapter = model_host.edit_module()
    hf = model_host.current_model().model
    expect = 1 + 2 * N  # 1 placeholder + (native raw key + appended chat key) per fact
    monotonic = sizes == sorted(sizes) and len(set(sizes)) == N
    one_codebook = len(adapter.keys) == sizes[-1]  # the live adapter IS the one we grew
    print(f"\n   final codebook_size={len(adapter.keys)} (expected {expect})  "
          f"monotonic_growth={monotonic}  single_codebook={one_codebook}")
    assert monotonic, f"codebook did not grow monotonically across edits: {sizes}"
    assert one_codebook, "live adapter codebook size != last edit's size (not one codebook)"

    # ---- 2) every fact retrievable from chat path (EMPTY RAG window); locality holds ----
    print("\n== chat-path retrieval (EMPTY RAG window), per fact ==")
    all_hit = True
    for i, s in enumerate(facts):
        stem, tgt = s["edit_prompt"], s["target_new"]
        mnt = len(tok.encode(" " + tgt, add_special_tokens=False)) + 6
        sc = live_score(stem, tok, hf, adapter)
        a = ans(stem, tok, mnt, chat=True)
        hit = tgt.lower() in a.lower()
        fact_green = (sc >= thr) and hit
        all_hit = all_hit and fact_green
        print(f"   [fact {i}] score={sc:.3f} {'fire' if sc >= thr else 'MISS'} | target?{hit} | ans={a!r}")

    # ---- 3) unrelated query must NOT fire and must NOT leak any target ----
    sc_unrel = live_score(UNRELATED, tok, hf, adapter)
    a_unrel = ans(UNRELATED, tok, 14, chat=True)
    leak = any(s["target_new"].lower() in a_unrel.lower() for s in facts)
    locality_ok = (sc_unrel < thr) and (not leak)
    print(f"\n   [unrel] score={sc_unrel:.3f} {'FIRE!' if sc_unrel >= thr else 'ok-nofire'} | "
          f"leak?{leak} | ans={a_unrel!r}")

    green = all_hit and locality_ok
    print("\n==================== sequential-edit VERDICT ====================")
    print(f"  {N} facts | ONE resident model | ONE codebook | no reload | no double-wrap")
    print(f"  codebook stacked (monotonic, single adapter)     : {monotonic and one_codebook}")
    print(f"  all facts retrieve + answer from chat path        : {all_hit}")
    print(f"  unrelated stays <{thr:.2f} & no target leak       : {locality_ok}")
    print(f"  VERDICT                                           : {'PASS ✅' if green else 'CHECK ⚠️'}")
    print("================================================================")


if __name__ == "__main__":
    main()
