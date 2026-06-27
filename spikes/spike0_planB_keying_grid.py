"""SPIKE v0.3 — Plan B query-span keying: diagnostic grid (① vs ②) + locality probe.

Run:  cd /workspace/AIEHackathon && python spikes/spike0_planB_keying_grid.py

For the first 5 type-A samples in eval/samples.json (stem=edit_prompt, paraphrase=queries[0].q):

  (a) DIAGNOSTIC GRID — codebook = RAW write-key compute_key(stem, raw); score each read-key:
        raw / stem        = control, expect ~1.0 (keying is sane)
        chat / stem       = ① ALONE  (scaffold cost; now span-isolated)
        raw / paraphrase  = ② ALONE  (paraphrase cost) ★ the verdict on ②:
                              >=0.85 -> ② a non-issue, the ① fix suffices
                              <0.85  -> ② real (defer C/D, do NOT fix now)
        chat / paraphrase = ①+②      (broken production condition under a RAW codebook)
      + locality probe    = chat read of an UNRELATED query vs the raw codebook.

  (b) FIX VALIDATION — codebook = CHAT write-key compute_key(stem, chat); read:
        chat / paraphrase -> should ≈ the raw/paraphrase cell (② preserved, ① removed)
        chat / stem       -> ~1.0 (identical render)
        chat / UNRELATED  -> should be LOW (<0.85) => Plan B did NOT collapse locality.

Scores use the adapter's OWN _query (the production 0.85 gate fn). Read-only measurement —
no edits are kept or scaled.
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import editing  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key, score  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(_ROOT, "eval", "samples.json")
N = 5
UNRELATED = "What is the boiling point of water at sea level?"  # off-topic; pure collapse probe


def load_type_a(n):
    with open(SAMPLES) as f:
        data = json.load(f)
    return [s for s in data["samples"] if s.get("sample_type") == "A"][:n]


def fmt(s, thr):
    return f"{s:.3f}/{'PASS' if s >= thr else 'fail'}"


def main():
    print("== [1] load base + install adapter ==")
    model_host.load_base()
    tok = model_host.tokenizer()
    hp = model_host.hparams()
    thr = float(hp.hopfield_key_match_threshold)
    assert int(hp.hopfield_retrieval_max_iter) == 1, "grid validity relies on max_iter==1"

    samples = load_type_a(N)
    # install the adapter once; the codebook content is irrelevant (score() overwrites it).
    editing.edit(
        model_host.current_model(),
        {"prompt": samples[0]["edit_prompt"], "target_new": samples[0]["target_new"]},
        key_mode="raw",
    )
    hf_model = model_host.current_model().model
    adapter = model_host.edit_module()
    print(f"   threshold={thr}  max_iter={int(hp.hopfield_retrieval_max_iter)}  N={len(samples)}")
    print(f"   UNRELATED probe = {UNRELATED!r}")

    def ck(text, templated):
        return compute_key(text, templated=templated, hf_model=hf_model, tok=tok, adapter=adapter)

    keys = ["raw/stem", "chat/stem", "raw/para", "chat/para", "loc/rawcb",
            "fix/chat_para", "fix/chat_stem", "fix/loc"]
    agg = {k: [] for k in keys}

    for i, s in enumerate(samples):
        stem, para, tgt = s["edit_prompt"], s["queries"][0]["q"], s["target_new"]

        wk_raw = ck(stem, False)            # RAW write-key (== raw read of stem)
        wk_chat = ck(stem, True)            # CHAT (Plan B) write-key (== chat read of stem)
        rd_para_raw = ck(para, False)
        rd_para_chat = ck(para, True)
        rd_unrel_chat = ck(UNRELATED, True)

        a_rs = score(wk_raw, wk_raw, adapter)            # raw/stem    (control ~1.0)
        a_cs = score(wk_chat, wk_raw, adapter)           # chat/stem   (①)
        a_rp = score(rd_para_raw, wk_raw, adapter)       # raw/para    (② verdict)
        a_cp = score(rd_para_chat, wk_raw, adapter)      # chat/para   (①+②)
        a_loc = score(rd_unrel_chat, wk_raw, adapter)    # locality vs raw cb
        b_cp = score(rd_para_chat, wk_chat, adapter)     # FIX chat/para  (≈ raw/para?)
        b_cs = score(wk_chat, wk_chat, adapter)          # FIX chat/stem  (~1.0)
        b_loc = score(rd_unrel_chat, wk_chat, adapter)   # FIX locality   (LOW?)

        for k, v in zip(keys, [a_rs, a_cs, a_rp, a_cp, a_loc, b_cp, b_cs, b_loc]):
            agg[k].append(v)

        print(f"\n--- [{i}] {s['id']}  stem={stem!r}")
        print(f"    para={para!r}  target={tgt!r}")
        print(f"    (a) RAW codebook       read raw           read chat(Plan B)")
        print(f"        stem          {fmt(a_rs, thr):>16}  {fmt(a_cs, thr):>16}   <- chat/stem = ①")
        print(f"        paraphrase    {fmt(a_rp, thr):>16}  {fmt(a_cp, thr):>16}   <- raw/para = ② verdict")
        print(f"        UNRELATED(chat)                  {fmt(a_loc, thr):>16}")
        print(f"    (b) CHAT codebook (FIX): chat/para={fmt(b_cp, thr)}  "
              f"chat/stem={fmt(b_cs, thr)}  UNRELATED={fmt(b_loc, thr)}")

    m = {k: st.mean(v) for k, v in agg.items()}
    print(f"\n==================== SUMMARY (mean over {len(samples)}) ====================")
    print(f"  (a) raw/stem   control (~1.0)            : {m['raw/stem']:.3f}")
    print(f"  (a) chat/stem  ① scaffold cost           : {m['chat/stem']:.3f}")
    print(f"  (a) raw/para   ② verdict                 : {m['raw/para']:.3f}"
          f"  -> ② is {'a NON-issue (① fix suffices)' if m['raw/para'] >= thr else 'REAL (defer C/D)'}")
    print(f"  (a) chat/para  ①+②                       : {m['chat/para']:.3f}")
    print(f"  ---- FIX (CHAT codebook) ----")
    print(f"  (b) chat/stem  (expect ~1.0)             : {m['fix/chat_stem']:.3f}")
    print(f"  (b) chat/para  (expect ≈ raw/para)       : {m['fix/chat_para']:.3f}  (raw/para={m['raw/para']:.3f})")
    print(f"  (b) locality   (expect LOW <{thr})        : {m['fix/loc']:.3f}"
          f"  -> locality {'OK' if m['fix/loc'] < thr else 'BROKEN'}")
    print("=================================================================")


if __name__ == "__main__":
    main()
