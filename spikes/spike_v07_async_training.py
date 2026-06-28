"""SPIKE 1 (v0.7) — async training: background 固化不阻断推理.

Run:  cd /workspace/AIEHackathon && python spikes/spike_v07_async_training.py

Demonstrates and MEASURES the async-training loop from docs/online_updating/online_editing_design.md
on a single A6000, WITHOUT modifying any existing module (third_party/horen, editing.py, generate.py,
serving/model_host.py, memory/* are all untouched — this script only orchestrates their public APIs):

  SERVING thread (main)              EDITING worker (1 bg thread = single writer)
    with swap_lock: generate(...)      queue.Queue() of edit requests; per item:
    reads the LIVE adapter               1. apply_horen_to_model(live_hf, copy=True)  -> train a
         │ stays correct + up               DEEPCOPY (the live model is never touched), append chat key
         ▼                                 2. transplant: new_adapter.layer <- _S["original"]
   ──── one GPU, SM/HBM shared ────        3. with swap_lock: swap_edit_module(new_adapter)  ① ATOMIC
                                           4. (N>=2) buffer.drop AFTER swap

Why these mechanics are correct (verified against editor.py / model_host.py):
  - apply_horen_to_model(..., copy=True) deepcopies the model and trains the copy (horen_main.py:22-23),
    so the live serving model is provably untouched during the ~4.5s edit (难点 B isolation).
  - HOREN.generate re-reads the slot via eval(f"self.model.{self.layer}") every call (editor.py:67-73),
    so swapping the adapter OBJECT in the live slot makes the wrapper set key_id on the NEW adapter.
  - swap_edit_module is a single setattr (model_host.py:69-76) — GIL-atomic; a swap_lock held for the
    duration of each generate confines swaps to request boundaries (never mid-decode).

Sanity = phases [1]-[5] at N=1; STOP & report (HoReN CLAUDE.md guardrail #5). Phase [6] (N>=2,
codebook carryover + swap-then-drop) is gated behind sanity sign-off.
"""
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import generate as gen  # noqa: E402
import serving.model_host as model_host  # noqa: E402
from keying import compute_key  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(_ROOT, "eval", "samples.json")
UNRELATED = "What is the boiling point of water at sea level?"  # off-topic locality / sanity probe
N = 1  # sanity; phase [6] (N>=2) is gated behind sanity sign-off


def load_type_a(n):
    with open(SAMPLES) as f:
        return [s for s in json.load(f)["samples"] if s.get("sample_type") == "A"][:n]


def gpu_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def main():
    # HoReN backend (model_host's import already put third_party/horen on sys.path)
    from src.models.horen.editor import HOREN
    from src.models.horen.horen_main import apply_horen_to_model
    from src.models.horen.utils import brackets_to_periods, parent_module

    # ───────────────────────── [1] bring-up ─────────────────────────
    print("== [1] bring-up: live model + empty-codebook HoReN wrapper; measure GPU budget ==")
    reset_peak()
    model_host.load_base()
    tok = model_host.tokenizer()
    hp = model_host.hparams()
    thr = float(hp.hopfield_key_match_threshold)
    live_hf = model_host.current_model()

    # Wrap the LIVE model once so model.generate sets key_id per call; this installs an EMPTY
    # (placeholder-only) adapter in the live slot -> base behavior until the first promotion.
    live_wrapper = HOREN(config=hp, model=live_hf)
    live_adapter = model_host.edit_module()
    model_host.register_edit_module(live_adapter, edited_model=live_wrapper)
    gpu_after_load = gpu_gb()
    print(f"   live wrapped; codebook={len(live_adapter.keys)} (placeholder only); "
          f"peak GPU after load = {gpu_after_load:.1f} GB / 48 GB")

    # resolve the inner_params slot on an ARBITRARY hf model (the shadow has its own parent module)
    _nm = hp.inner_params[0]
    if _nm.endswith((".weight", ".bias")):
        _nm = _nm.rsplit(".", 1)[0]
    _attr = _nm.rsplit(".", 1)[-1]

    def slot_adapter(hf):
        return getattr(parent_module(hf, brackets_to_periods(_nm)), _attr)

    # ───────────────────── async machinery (spike-side) ─────────────────────
    edit_q: "queue.Queue" = queue.Queue()
    swap_lock = threading.Lock()  # held for each generate; grabbed for the instant swap -> request boundary
    worker_log: dict = {}

    def append_chat_key(wrapper, adapter, stem):
        """Plan B chat key, reusing the just-trained value row — mirrors editing._append_chat_key,
        but runs on the SHADOW (wrapper.model / its adapter), leaving the live model untouched."""
        ck = compute_key(stem, templated=True, hf_model=wrapper.model, tok=tok, adapter=adapter)
        v_idx = wrapper.edit_log["chosen_key"]
        adapter.keys = torch.cat([adapter.keys, ck.to(adapter.keys.dtype)], dim=0)
        adapter.values = torch.nn.Parameter(
            torch.cat([adapter.values, adapter.values[v_idx:v_idx + 1]], dim=0),
            requires_grad=adapter.values.requires_grad,
        )
        adapter.key_labels.append(adapter.key_labels[v_idx])

    def worker_loop():
        while True:
            item = edit_q.get()
            if item is None:
                edit_q.task_done()
                return
            req, tag, fail = item
            rec = {"ok": False}
            try:
                t0 = time.time()
                if fail:
                    raise RuntimeError("injected edit failure (crash-isolation probe)")
                # SHADOW train: deepcopy-on-edit. live_hf is provably untouched; carryover is automatic
                # (the deepcopy captures the live model's CURRENT codebook).
                wrapper, _ = apply_horen_to_model(live_hf, tok, [req], hp, copy=True)
                shad = slot_adapter(wrapper.model)
                append_chat_key(wrapper, shad, req["prompt"])
                # transplant onto the live frozen base, decoupling the 16G deepcopy (forward uses
                # self.layer; self.weight is vestigial, left as-is)
                shad.layer = model_host._S["original"]
                rec["train_s"] = time.time() - t0
                rec["codebook"] = len(shad.keys)
                with swap_lock:  # ① ATOMIC promotion at a request boundary
                    model_host.swap_edit_module(shad)
                    model_host._S["adapter"] = shad
                rec["ok"] = True
                del wrapper
                torch.cuda.empty_cache()  # reclaim the deepcopy (peak high-water mark already recorded)
            except Exception as e:  # crash isolation: non-fatal, worker survives, live adapter untouched
                rec["error"] = repr(e)
            finally:
                worker_log[tag] = rec
                edit_q.task_done()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    def submit(req, tag, fail=False):
        t0 = time.time()
        edit_q.put((req, tag, fail))
        return (time.time() - t0) * 1000.0  # ms

    def serve(query, mnt, chat=True):
        with swap_lock:
            return gen.generate(query, model=model_host.current_model(), tok=tok,
                                max_new_tokens=mnt, use_chat_template=chat, with_rag=False)

    def serve_timed(query, mnt):
        t = time.time()
        a = serve(query, mnt)
        return a, time.time() - t

    def live_score(query):
        with swap_lock:
            ad = model_host.edit_module()
            rk = compute_key(query, templated=True, hf_model=live_hf, tok=tok, adapter=ad)
            return ad._query(rk).max().item()

    # ───────────────────── pick fact0 + idle baseline ─────────────────────
    s = load_type_a(N)[0]
    stem, para, tgt = s["edit_prompt"], s["queries"][0]["q"], s["target_new"]
    mnt = len(tok.encode(" " + tgt, add_special_tokens=False)) + 6
    req = {"prompt": stem, "target_new": tgt}
    print(f"   fact0 {s['id']}: stem={stem!r}  target={tgt!r}")

    serve(UNRELATED, mnt)  # warmup (first decode pays cuda init / kernel compile)
    idle = [serve_timed(UNRELATED, mnt)[1] for _ in range(3)]
    idle_ms = 1000 * sum(idle) / len(idle)
    print(f"   idle serving latency (unrelated, mean of 3) = {idle_ms:.0f} ms")

    # PRE-edit state (live codebook is empty until the swap, so this == the during-train state).
    # Measured BEFORE submit so no forward-hook (compute_key) ever overlaps the worker's deepcopy.
    before_known = tgt.lower() in serve(stem, mnt).lower()
    before_score = live_score(stem)
    print(f"   PRE-edit: fact0 known? {before_known}   live_score(stem) = {before_score:.3f}  (< {thr} expected)")

    # ───────────────────── [2] non-blocking accept ─────────────────────
    print("\n== [2] non-blocking accept: submit returns immediately; training runs in the background ==")
    reset_peak()
    sub_ms = submit(req, "fact0")
    print(f"   submit() returned in {sub_ms:.2f} ms (4.x s of training now runs on the worker thread)")

    # ───────────────── [3] serve-during-train (shadow isolation) ─────────────────
    print("\n== [3] serve-during-train (难点 B): serving stays correct + live while the edit trains ==")
    # ONLY hook-free unrelated serves here, so nothing races the worker's deepcopy(live_hf).
    train_lat = []
    while worker_log.get("fact0") is None:
        train_lat.append(serve_timed(UNRELATED, mnt)[1])
    rec0 = worker_log["fact0"]
    edit_peak = gpu_gb()
    train_ms = 1000 * sum(train_lat) / len(train_lat) if train_lat else float("nan")
    print(f"   during-train serving latency (unrelated, mean of {len(train_lat)}) = {train_ms:.0f} ms "
          f"(idle {idle_ms:.0f} ms -> contention x{train_ms / idle_ms:.2f})")
    print(f"   worker: ok={rec0['ok']}  train_s={rec0.get('train_s', float('nan')):.2f}s  "
          f"codebook={rec0.get('codebook')}")
    print(f"   PEAK GPU during async edit (deepcopy + train + concurrent serving) = {edit_peak:.1f} GB / 48 GB")

    # ───────────────── [4] atomic promotion (post-swap) ─────────────────
    print("\n== [4] atomic promotion: after the swap fact0 is retrievable; locality holds ==")
    after_stem, after_para, after_unrel = live_score(stem), live_score(para), live_score(UNRELATED)
    a_stem, a_para, a_unrel = serve(stem, mnt), serve(para, mnt), serve(UNRELATED, mnt)

    def hit(a):
        return tgt.lower() in a.lower()

    print(f"   chat stem : score={after_stem:.3f} {'fire' if after_stem >= thr else 'MISS'} | "
          f"ans={a_stem!r} target?{hit(a_stem)}")
    print(f"   chat para : score={after_para:.3f} {'fire' if after_para >= thr else 'miss'} | "
          f"ans={a_para!r} target?{hit(a_para)}")
    print(f"   chat unrel: score={after_unrel:.3f} {'FIRE!' if after_unrel >= thr else 'ok-nofire'} | "
          f"ans={a_unrel!r}")
    print(f"   live codebook size after promotion = {len(model_host.edit_module().keys)}")

    # ───────────────── [5] crash isolation ─────────────────
    print("\n== [5] crash isolation (难点 D.5): a failing edit is non-fatal; serving + worker survive ==")
    sub_ms2 = submit(req, "boom", fail=True)
    edit_q.join()  # let the failing item drain
    rec_boom = worker_log.get("boom", {})
    serve_ok = isinstance(serve(UNRELATED, mnt), str)
    still_fires = live_score(stem) >= thr
    print(f"   failing submit accepted in {sub_ms2:.2f} ms; worker recorded error: {rec_boom.get('error')}")
    print(f"   worker thread alive after crash: {worker.is_alive()}")
    print(f"   serving still works: {serve_ok};  fact0 still fires (live adapter untouched): {still_fires}")

    # ───────────────── verdict ─────────────────
    fact0_ok = (rec0["ok"] and not before_known and before_score < thr
                and after_stem >= thr and hit(a_stem) and after_unrel < thr)
    crash_ok = (not rec_boom.get("ok", False)) and worker.is_alive() and serve_ok and still_fires
    fits = edit_peak < 46.0
    green = fact0_ok and crash_ok and fits and sub_ms < 50

    print("\n==================== SPIKE 1 (v0.7) async-training VERDICT ====================")
    print(f"  [1] fits 48GB           : {fits}  (peak {edit_peak:.1f} GB during async edit)")
    print(f"  [2] non-block accept    : {sub_ms < 50}  ({sub_ms:.2f} ms)")
    print(f"  [3] serve-during-train  : pre-swap fact0 unknown={not before_known}, "
          f"serving latency x{train_ms / idle_ms:.2f} vs idle (难点 A contention, reported)")
    print(f"  [4] atomic promotion    : fires+answers={after_stem >= thr and hit(a_stem)}, "
          f"locality={after_unrel < thr}")
    print(f"  [5] crash isolation     : {crash_ok}  (non-fatal, worker alive, live adapter intact)")
    print(f"  VERDICT : {'PASS ✅' if green else 'CHECK ⚠️'}   (N={N} sanity; phase [6] N>=2 gated)")
    print("==============================================================================")

    edit_q.put(None)
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
