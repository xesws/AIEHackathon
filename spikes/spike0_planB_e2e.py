"""SPIKE v0.3 — e2e hero loop: chat-path edit → retrieve → ANSWER (RAG OFF).

Run:  cd /workspace/AIEHackathon && python spikes/spike0_planB_e2e.py

The thing the grid cannot show: whether the retrieved value, injected on the chat path,
actually makes the model GENERATE target_new. For the first 2 type-A samples, edit the fact
(key_mode='chat' = Plan B append) and, via the chat path with an EMPTY RAG window:
  - stem query      -> retrieval fires (live score ≥0.85) AND answer contains target_new   [① proof]
  - paraphrase query-> report score + whether answer contains target_new                   [①+②]
  - unrelated query -> does NOT fire (score <0.85), answer free of target_new               [locality]
  - stem via RAW path -> still answers target_new                                          [raw intact]

Each fact runs on a FRESH base model (one edit per model) to isolate the chat-keying fix;
multi-edit on a single resident model (sequential editing) is a separate follow-up.
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
N = 2
UNRELATED = "What is the boiling point of water at sea level?"


def load_type_a(n):
    with open(SAMPLES) as f:
        return [s for s in json.load(f)["samples"] if s.get("sample_type") == "A"][:n]


def live_score(query, tok, hf, adapter):
    rk = compute_key(query, templated=True, hf_model=hf, tok=tok, adapter=adapter)
    return adapter._query(rk).max().item()


def ans(query, tok, mnt, *, chat):
    return gen.generate(query, model=model_host.current_model(), tok=tok,
                        max_new_tokens=mnt, use_chat_template=chat, with_rag=False)


def main():
    samples = load_type_a(N)
    all_green = True
    for i, s in enumerate(samples):
        stem, para, tgt = s["edit_prompt"], s["queries"][0]["q"], s["target_new"]
        print(f"\n========== [{i}] {s['id']}  target={tgt!r} ==========")
        print(f"   stem={stem!r}")
        print(f"   para={para!r}")

        model_host.load_base()  # fresh model per fact
        tok = model_host.tokenizer()
        thr = float(model_host.hparams().hopfield_key_match_threshold)
        mnt = len(tok.encode(" " + tgt, add_special_tokens=False)) + 6

        res = editing.edit(model_host.current_model(),
                           {"prompt": stem, "target_new": tgt}, key_mode="chat")
        adapter, hf = model_host.edit_module(), model_host.current_model().model
        print(f"   edited: codebook_size={res['codebook_size']}  edit_s={res['edit_seconds']:.2f}")

        sc_stem = live_score(stem, tok, hf, adapter)
        sc_para = live_score(para, tok, hf, adapter)
        sc_unrel = live_score(UNRELATED, tok, hf, adapter)
        a_stem = ans(stem, tok, mnt, chat=True)
        a_para = ans(para, tok, mnt, chat=True)
        a_unrel = ans(UNRELATED, tok, mnt, chat=True)
        a_stem_raw = ans(stem, tok, mnt, chat=False)

        def hit(a):
            return tgt.lower() in a.lower()

        print(f"   CHAT stem  : score={sc_stem:.3f} {'fire' if sc_stem >= thr else 'MISS'} | "
              f"ans={a_stem!r} target?{hit(a_stem)}")
        print(f"   CHAT para  : score={sc_para:.3f} {'fire' if sc_para >= thr else 'miss'} | "
              f"ans={a_para!r} target?{hit(a_para)}")
        print(f"   CHAT unrel : score={sc_unrel:.3f} {'FIRE!' if sc_unrel >= thr else 'ok-nofire'} | "
              f"ans={a_unrel!r} target?{hit(a_unrel)}")
        print(f"   RAW  stem  : ans={a_stem_raw!r} target?{hit(a_stem_raw)}")

        fact_green = (sc_stem >= thr) and hit(a_stem) and (not hit(a_unrel)) and hit(a_stem_raw)
        all_green = all_green and fact_green
        print(f"   -> fact green (chat-stem fires+answers, raw intact, no locality leak): {fact_green}")

    print("\n==================== e2e VERDICT ====================")
    print(f"  hero loop (chat path, RAG off) on {len(samples)} facts: {'PASS ✅' if all_green else 'CHECK ⚠️'}")
    print("====================================================")


if __name__ == "__main__":
    main()
