"""SPIKE v0.5.2 — multi-fact ingest: ALL edit-route facts buffered, allergy actually learned.

Run:  cd /workspace/AIEHackathon && python spikes/spike_v052_multifact_ingest.py
      (GPU + real OpenRouter extract. Do NOT run on CPU / in CI.)

Closes the gap the v0.4 smoke hid: it took only edit_items[0] (the NAME fact) and dropped the
allergy. This drives the real serving entry serving.ingest.ingest (which buffers EVERY
edit-route item) on a single turn carrying TWO facts, then asks the ALLERGY specifically:

  one turn ("I'm JQ, allergic to nickel buckles")
    -> ingest(chat)        : real extract -> buffer ALL edit-route (expect name + allergy)
    -> triggers.manual()   : consolidate.run_pass -> both folded into ONE codebook
    -> ask the ALLERGY stem (chat, RAG off) -> answer contains "nickel"/"buckle"  [the proof]
    -> ask the NAME stem    -> answer "JQ"                                         [both coexist]
    -> unrelated query      -> no false fire                                       [locality]

Green requires the allergy answer to be the allergy VALUE, not a name echo. If real extract
does NOT surface the allergy as an edit-fact, the assertion fails LOUDLY (honest report, no
faked green).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key  # noqa: E402
from memory import buffer, consolidate, rag_store, store  # noqa: E402
from memory.schema import PROV_EDIT  # noqa: E402
from serving import triggers  # noqa: E402
from serving.ingest import ingest  # noqa: E402

CHAT = [{
    "role": "user",
    "content": "Hey, I'm JQ. Just so you know, I have a severe contact allergy to nickel buckles.",
}]
UNRELATED = "What is the boiling point of water at sea level?"
ALLERGY_WORDS = ("nickel", "buckle")


def live_score(query, tok, hf, adapter):
    rk = compute_key(query, templated=True, hf_model=hf, tok=tok, adapter=adapter)
    return adapter._query(rk).max().item()


def ans(query, tok, mnt, *, chat=True):
    return gen.generate(query, model=model_host.current_model(), tok=tok,
                        max_new_tokens=mnt, use_chat_template=chat, with_rag=False)


def main():
    # ---- 1) real model + clean state + provider ----
    print("== [1] load base + reset stores + set_model_provider ==")
    model_host.load_base()
    store.reset()
    rag_store.reset()
    consolidate.set_model_provider(lambda: model_host.current_model())
    tok = model_host.tokenizer()
    thr = float(model_host.hparams().hopfield_key_match_threshold)

    # ---- 2) ingest ONE turn carrying TWO facts ----
    print("== [2] ingest(chat)  (REAL extract -> buffer ALL edit-route) ==")
    res = ingest(CHAT)
    print(f"   {res}")

    buffered = buffer.load_unconsolidated()
    facts = [(b.id, b.provenance[PROV_EDIT]) for b in buffered if (b.provenance or {}).get(PROV_EDIT)]
    print("   buffered edit-route facts:")
    for fid, ed in facts:
        print(f"     - {fid}: stem={ed['stem']!r} target={ed['target']!r}")

    allergy = [(fid, ed) for fid, ed in facts
               if any(w in ed["target"].lower() for w in ALLERGY_WORDS)]
    name = [(fid, ed) for fid, ed in facts if ed["target"].strip().lower() == "jq"]

    print(f"   n_edit_buffered={len(buffered)}  allergy_fact?{bool(allergy)}  name_fact?{bool(name)}")
    assert len(buffered) >= 2, f"ingest buffered <2 edit-route facts (got {len(buffered)}) — dropped facts"
    assert allergy, "real extract did NOT surface an ALLERGY edit-fact (routed to rag / merged / missed)"

    # ---- 3) consolidate via the real trigger ----
    print("== [3] triggers.manual()  (consolidate.run_pass) ==")
    n_written = triggers.manual()
    drained = buffer.load_unconsolidated()
    adapter = model_host.edit_module()
    hf = model_host.current_model().model
    print(f"   n_written={n_written}  buffer_drained={len(drained) == 0}  codebook_size={len(adapter.keys)}")
    assert n_written == len(buffered), f"expected n_written={len(buffered)}, got {n_written}"
    assert not drained

    # ---- 4) THE PROOF: ask the ALLERGY (not the name) ----
    print("\n== [4] ask the ALLERGY stem (chat, RAG OFF) ==")
    a_stem, a_ed = allergy[0]
    a_mnt = len(tok.encode(" " + a_ed["target"], add_special_tokens=False)) + 6
    sc_all = live_score(a_ed["stem"], tok, hf, adapter)
    a_all = ans(a_ed["stem"], tok, a_mnt, chat=True)
    allergy_hit = any(w in a_all.lower() for w in ALLERGY_WORDS)
    name_echo = a_all.strip().lower().replace(" ", "").startswith("jqjq") or a_all.strip().lower() == "jq"
    print(f"   stem={a_ed['stem']!r}  score={sc_all:.3f} {'fire' if sc_all >= thr else 'MISS'}")
    print(f"   ans={a_all!r}  -> allergy_value?{allergy_hit}  name_echo?{name_echo}")

    # ---- 5) name fact still there; locality holds ----
    name_hit = None
    if name:
        n_stem, n_ed = name[0]
        n_mnt = len(tok.encode(" " + n_ed["target"], add_special_tokens=False)) + 6
        a_name = ans(n_ed["stem"], tok, n_mnt, chat=True)
        name_hit = "jq" in a_name.lower()
        print(f"\n== [5] name stem {n_ed['stem']!r} -> ans={a_name!r}  JQ?{name_hit} ==")

    sc_unrel = live_score(UNRELATED, tok, hf, adapter)
    a_unrel = ans(UNRELATED, tok, 14, chat=True)
    leak = any(w in a_unrel.lower() for w in ALLERGY_WORDS)
    locality_ok = (sc_unrel < thr) and (not leak)
    print(f"   [unrel] score={sc_unrel:.3f} {'FIRE!' if sc_unrel >= thr else 'ok-nofire'} | leak?{leak} | ans={a_unrel!r}")

    green = (len(buffered) >= 2) and bool(allergy) and (sc_all >= thr) and allergy_hit and (not name_echo) and locality_ok
    print("\n==================== multi-fact ingest VERDICT ====================")
    print(f"  ingest buffered ALL edit-route (>=2: name + allergy)  : {len(buffered) >= 2 and bool(allergy) and bool(name)}")
    print(f"  run_pass wrote all ({n_written}), buffer drained        : {n_written == len(buffered) and not drained}")
    print(f"  ASK ALLERGY -> answers nickel/buckle (NOT name echo)  : {allergy_hit and not name_echo}")
    print(f"  name fact still answers JQ                            : {name_hit}")
    print(f"  unrelated stays <{thr:.2f} & no leak                  : {locality_ok}")
    print(f"  VERDICT                                              : {'PASS ✅' if green else 'CHECK ⚠️'}")
    print("==================================================================")


if __name__ == "__main__":
    main()
