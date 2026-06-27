"""SPIKE v0.4 — HERO SMOKE: full memory pipeline puts a fact into WEIGHTS,
answered in a FRESH session with RAG OFF via the real chat path.

Run:  cd /workspace/AIEHackathon && python spikes/spike_v04_hero_smoke.py
      (GPU — loads llama-3.1-8B-Instruct. Do NOT run in CI / on CPU.)

This is the e2e must-pass driver. It mirrors spike0's helpers (token_em,
gpu_gb, greedy ans) and walks the WHOLE Engram loop:

    extract (real LLM)  ->  buffer.append  ->  consolidate.run_pass (real edit
    into weights)  ->  fresh generate(chat template, RAG OFF)  ->  fact present.

PRIMARY path uses the real extractor (OpenRouter key auto-loads from .env).
FALLBACK (no key / extract empty) builds the item from eval/samples.json[0].
The MUST-PASS gate is step [6]: the edited model answers the cloze stem through
the chat template with RAG OFF. Everything is guarded so a single failure prints
a clear diagnostic instead of a bare traceback. Exit 0 iff the must-pass passes.
"""
import json
import math
import os
import sys
import time

# repo root on path so `serving` / `editing` / `generate` / `memory` import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from memory import buffer, consolidate, extract as memory_extract, store  # noqa: E402
from memory.schema import MemoryItem, PROV_EDIT, PROV_SOURCE_MSG  # noqa: E402


# --- helpers copied from spike0 (kept local on purpose) -------------------------------------
def token_em(tok, answer: str, target: str) -> float:
    """Position-wise token EM over the first len(target) tokens (mirrors evaluate_utils)."""
    tgt = tok.encode(target, add_special_tokens=False)
    ans = tok.encode(answer.strip(), add_special_tokens=False)[: len(tgt)]
    if len(ans) != len(tgt):
        return 0.0
    acc = sum(int(a == b) for a, b in zip(tgt, ans)) / len(tgt)
    return 1.0 if math.isnan(acc) else acc


def gpu_gb() -> float:
    """Currently-allocated GPU memory in GB."""
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def peak_gpu_gb() -> float:
    """Peak GPU memory observed this run, in GB."""
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def ans(query: str, tok, mnt: int, *, chat=False, rag=True) -> str:
    """Greedy generate on the resident (possibly edited) model. RAG defaults on; the smoke
    flips it OFF for the hero proof so only WEIGHTS can supply the fact."""
    return gen.generate(
        query, model=model_host.current_model(), tok=tok,
        max_new_tokens=mnt, use_chat_template=chat, with_rag=rag,
    )


def torch_banner(tag: str) -> None:
    print(f"[{tag}] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")


# --- candidate construction -----------------------------------------------------------------
def _from_real_extract():
    """PRIMARY: feed a scripted single-fact conversation to the real extractor and pick the
    edit-route item carrying provenance[PROV_EDIT]. Returns (item, target, stem) or None."""
    chat = [
        {
            "role": "user",
            "content": (
                "Hey, I'm JQ. Just so you know, I have a severe contact "
                "allergy to nickel buckles."
            ),
        }
    ]
    try:
        items = memory_extract.extract(chat)
    except Exception as e:  # network / key / parse — fall back
        print(f"    extract() raised: {e!r}")
        return None
    edit_items = [
        it for it in items
        if it.route == "edit" and (it.provenance or {}).get(PROV_EDIT)
    ]
    if not edit_items:
        print(f"    extract() returned {len(items)} item(s), none edit-route w/ PROV_EDIT")
        return None
    it = edit_items[0]
    ed = it.provenance[PROV_EDIT]
    return it, ed["target"], ed["stem"]


def _from_samples():
    """FALLBACK: build the edit item directly from the first type-A sample record."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "samples.json")
    with open(path) as f:
        data = json.load(f)
    rec = next(s for s in data["samples"] if s.get("sample_type") == "A")
    # Turn the cloze "...___" into an editing stem (drop the blank; keep trailing space sense).
    stem = rec["edit_prompt"].replace("___", "").rstrip()
    target = rec["target_new"]
    subject = rec.get("subject", "")
    item = MemoryItem(
        id="mem_sample_A0001",
        type="fact",
        text=rec["rag_doc"],
        route="edit",
        status="buffer",
        source="eval/samples.json",
        ts=time.time(),
        provenance={
            PROV_SOURCE_MSG: "eval/samples.json",
            PROV_EDIT: {"stem": stem, "target": target, "subject": subject},
        },
    )
    paraphrase = rec["queries"][0]["q"] if rec.get("queries") else None
    return item, target, stem, paraphrase


def main() -> int:
    torch_banner("start")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    verdict = {
        "extract_path": "?", "n_written": 0,
        "pre_raw": "", "post_raw": "", "post_raw_hit": False,
        "chat_out": "", "chat_em": 0.0, "must_pass": False,
        "paraphrase_q": None, "paraphrase_out": "", "paraphrase_hit": False,
        "edit_latency_s": float("nan"), "peak_gpu_gb": 0.0, "load_s": float("nan"),
        "error": None,
    }

    try:
        # --- [1] load base + clean store --------------------------------------------------
        print("== [1] load base + store.reset ==")
        t0 = time.time()
        model_host.load_base()
        verdict["load_s"] = time.time() - t0
        tok = model_host.tokenizer()
        store.reset()
        print(f"    loaded in {verdict['load_s']:.1f}s | gpu {gpu_gb():.1f} GB")

        # --- [2] INV-11 model-provider injection ------------------------------------------
        consolidate.set_model_provider(lambda: model_host.current_model())
        print("== [2] set_model_provider (INV-11) ==")

        # --- [3] build the candidate item -------------------------------------------------
        print("== [3] build candidate (REAL extract -> fallback samples.json) ==")
        paraphrase = None
        got = _from_real_extract()
        if got is not None:
            item, target, stem = got
            verdict["extract_path"] = "real-extract"
            paraphrase = "What is JQ allergic to?"
        else:
            item, target, stem, paraphrase = _from_samples()
            verdict["extract_path"] = "fallback-samples"
        verdict["paraphrase_q"] = paraphrase
        print(f"    path={verdict['extract_path']}  stem={stem!r}  target={target!r}")

        mnt = max(len(tok.encode(" " + target, add_special_tokens=False)), 16)

        # --- [4] PRE-edit baseline (raw, RAG off): expect target ABSENT -------------------
        print("== [4] PRE-edit baseline (raw, RAG off) ==")
        pre = ans(stem, tok, mnt, chat=False, rag=False)
        verdict["pre_raw"] = pre
        pre_clean = target.lower() not in pre.lower()
        print(f"    PRE  : {pre!r}   target absent (clean before)? {pre_clean}")
        if not pre_clean:
            print("    WARN: target already present pre-edit; before/after not clean.")

        # --- [5] buffer -> consolidate into WEIGHTS ---------------------------------------
        print("== [5] buffer.append + consolidate.run_pass('manual') ==")
        buffer.append(item)
        te = time.time()
        n = consolidate.run_pass("manual")
        verdict["edit_latency_s"] = time.time() - te
        verdict["n_written"] = n
        print(f"    n_written={n}  run_pass_seconds={verdict['edit_latency_s']:.2f}  gpu {gpu_gb():.1f} GB")
        assert n >= 1, f"consolidate.run_pass wrote nothing (n_written={n})"
        drained = buffer.load_unconsolidated()
        print(f"    buffer drained? {len(drained) == 0}  (no double-existence)")
        assert len(drained) == 0, f"buffer not drained: {[d.id for d in drained]}"

        # --- [6] MUST-PASS: fresh session, CHAT path, RAG OFF -----------------------------
        print("== [6] MUST-PASS: CHAT template, fresh session, RAG OFF ==")
        out_chat = ans(stem, tok, mnt, chat=True, rag=False)
        verdict["chat_out"] = out_chat
        verdict["chat_em"] = token_em(tok, out_chat, target)
        verdict["must_pass"] = (target.lower() in out_chat.lower()) or verdict["chat_em"] == 1.0
        print(f"    CHAT : {out_chat!r}   token-EM={verdict['chat_em']}   PASS={verdict['must_pass']}")

        # --- [7] RAW path sanity ----------------------------------------------------------
        print("== [7] RAW path sanity (raw prompt, RAG off) ==")
        post = ans(stem, tok, mnt, chat=False, rag=False)
        verdict["post_raw"] = post
        verdict["post_raw_hit"] = target.lower() in post.lower()
        print(f"    POST : {post!r}   target present? {verdict['post_raw_hit']}")

        # --- [8] BONUS generalization (paraphrase, chat, RAG off) -------------------------
        print("== [8] BONUS generalization (paraphrase, chat, RAG off) ==")
        out_par = ans(paraphrase, tok, mnt, chat=True, rag=False)
        verdict["paraphrase_out"] = out_par
        verdict["paraphrase_hit"] = target.lower() in out_par.lower()
        print(f"    Q={paraphrase!r}\n    A={out_par!r}   target present? {verdict['paraphrase_hit']} (miss OK)")

    except Exception as e:
        import traceback
        verdict["error"] = f"{type(e).__name__}: {e}"
        print("\n!! SMOKE aborted with diagnostic:")
        traceback.print_exc()

    # --- [9] verdict block ----------------------------------------------------------------
    verdict["peak_gpu_gb"] = peak_gpu_gb()
    print("\n==================== SPIKE v0.4 HERO SMOKE ====================")
    print(f"  extract path                 : {verdict['extract_path']}")
    print(f"  n_written (edits to weights) : {verdict['n_written']}")
    print(f"  PRE  raw (RAG off)           : {verdict['pre_raw']!r}")
    print(f"  POST raw (RAG off)           : {verdict['post_raw']!r}  hit={verdict['post_raw_hit']}")
    print(f"  CHAT must-pass (RAG off)     : {verdict['chat_out']!r}  EM={verdict['chat_em']}")
    print(f"  paraphrase bonus             : {verdict['paraphrase_out']!r}  hit={verdict['paraphrase_hit']}")
    print(f"  ---")
    print(f"  edit latency (run_pass)      : {verdict['edit_latency_s']:.2f} s")
    print(f"  model load                   : {verdict['load_s']:.1f} s")
    print(f"  peak gpu                     : {verdict['peak_gpu_gb']:.1f} GB")
    if verdict["error"]:
        print(f"  ERROR                        : {verdict['error']}")
    must = bool(verdict["must_pass"]) and verdict["error"] is None
    print(f"  MUST-PASS VERDICT            : {'PASS ✅' if must else 'FAIL ❌'}")
    print("==============================================================")
    torch_banner("end")
    return 0 if must else 1


if __name__ == "__main__":
    sys.exit(main())
