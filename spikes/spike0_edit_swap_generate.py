"""SPIKE 0 — one hardcoded HoReN edit → hot-swap → answer correctly with RAG OFF.

Run:  cd /workspace/AIEHackathon && python spikes/spike0_edit_swap_generate.py

Proves the core Engram loop end-to-end on llama-3.1-8B-Instruct:
  load base → real edit (side-module adapter) → install/remove via model_host → greedy generate.
N=1 sanity only; prints PASS/FAIL + single-edit latency + the resolved output format.
"""
import math
import os
import sys
import time

# repo root on path so `serving` / `editing` / `generate` / `memory` import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import editing  # noqa: E402
import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402

# --- hardcoded edit: ZsRE record[0] -----------------------------------------------------
REQUEST = {
    "prompt": "What university did Watts Humphrey attend?",
    "subject": "Watts Humphrey",
    "target_new": "Illinois Institute of Technology",
    "rephrase": "What university did Watts Humphrey take part in?",
}
LOC_PROMPT = "nq question: who played desmond doss father in hacksaw ridge"
LOC_ANS = "Hugo Weaving"


def token_em(tok, answer: str, target: str) -> float:
    """Position-wise token EM over the first len(target) tokens (mirrors evaluate_utils)."""
    tgt = tok.encode(target, add_special_tokens=False)
    ans = tok.encode(answer.strip(), add_special_tokens=False)[: len(tgt)]
    if len(ans) != len(tgt):
        return 0.0
    acc = sum(int(a == b) for a, b in zip(tgt, ans)) / len(tgt)
    return 1.0 if math.isnan(acc) else acc


def gpu_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9


def ans(query: str, tok, mnt: int, *, chat=False, rag=True) -> str:
    return gen.generate(
        query, model=model_host.current_model(), tok=tok,
        max_new_tokens=mnt, use_chat_template=chat, with_rag=rag,
    )


def main():
    target = REQUEST["target_new"]

    print("== [1] load base ==")
    t0 = time.time()
    model_host.load_base()
    load_s = time.time() - t0
    tok = model_host.tokenizer()
    mnt = len(tok.encode(" " + target, add_special_tokens=False))
    print(f"    loaded in {load_s:.1f}s | gpu {gpu_gb():.1f} GB | target={mnt} tok")

    print("== [2] PRE-edit  (raw prompt, RAG off) ==")
    pre = ans(REQUEST["prompt"], tok, mnt)
    print(f"    PRE  : {pre!r}")

    print("== [3] EDIT (HoReN, n_iter=50) ==")
    res = editing.edit(model_host.current_model(), REQUEST)
    print(f"    edit_seconds={res['edit_seconds']:.2f}  codebook_size={res['codebook_size']}  gpu {gpu_gb():.1f} GB")

    print("== [4] POST-edit (raw prompt, RAG off) ==")
    post = ans(REQUEST["prompt"], tok, mnt)
    em = token_em(tok, post, target)
    print(f"    POST : {post!r}   token-EM={em}")

    print("== [5] hot-swap demo ==")
    model_host.swap_edit_module(None)
    off = ans(REQUEST["prompt"], tok, mnt)
    print(f"    adapter OFF : {off!r}")
    model_host.swap_edit_module(res["adapter"])
    on = ans(REQUEST["prompt"], tok, mnt)
    print(f"    adapter ON  : {on!r}")

    print("== [6] our inference path: build_prompt chat skeleton, RAG OFF ==")
    chat = ans(REQUEST["prompt"], tok, max(mnt, 16), chat=True, rag=False)
    chat_fires = target.lower() in chat.lower()
    print(f"    CHAT(empty RAG): {chat!r}   edit fires through chat template? {chat_fires}")

    print("== [7] generalization (rephrase) & locality spot-check ==")
    reph = ans(REQUEST["rephrase"], tok, mnt)
    loc_mnt = len(tok.encode(" " + LOC_ANS, add_special_tokens=False)) + 4
    loc = ans(LOC_PROMPT, tok, loc_mnt)
    print(f"    rephrase ans : {reph!r}  (target in? {target.lower() in reph.lower()})")
    print(f"    locality ans : {loc!r}  (expect ~{LOC_ANS!r}, should be unchanged)")

    # --- verdict ---
    edit_took = (em == 1.0) and (target.lower() in post.lower())
    base_differs = target.lower() not in pre.lower()
    swap_ok = (target.lower() not in off.lower()) and (target.lower() in on.lower())
    green = edit_took and base_differs and swap_ok

    print("\n==================== SPIKE 0 ====================")
    print(f"  edit took (POST==target, token-EM=1)   : {edit_took}")
    print(f"  base differs (PRE != target)           : {base_differs}")
    print(f"  hot-swap toggles answer (OFF/ON)       : {swap_ok}")
    print(f"  chat-template path fires (bonus)       : {chat_fires}")
    print(f"  ---")
    print(f"  single-edit latency                    : {res['edit_seconds']:.2f} s")
    print(f"  model load                             : {load_s:.1f} s")
    print(f"  output format                          : HopfieldAdapter side-module @ layers[29].mlp.down_proj")
    print(f"  peak gpu                               : {gpu_gb():.1f} GB")
    print(f"  VERDICT                                : {'PASS ✅' if green else 'CHECK ⚠️'}")
    print("================================================")


if __name__ == "__main__":
    main()
