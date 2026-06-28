"""SPIKE 1 (v0.7) — async training: background 固化不阻断推理.  [v2: frozen-base aliasing + lock-free promote]

Run:  cd /workspace/AIEHackathon && python spikes/spike_v07_async_training.py

Demonstrates and MEASURES the async-training loop from docs/online_updating/online_editing_design.md
on a single A6000, WITHOUT modifying any existing module (third_party/horen, editing.py, generate.py,
serving/model_host.py, memory/* are all untouched — this script only orchestrates their public APIs).

v1 (commit 30a681d) used apply_horen_to_model(copy=True) — a full 8B deepcopy per edit (32.3G peak,
11.83s) — and a swap_lock the worker had to win against a tight serve loop (handoff starvation). v2
fixes both, still spike-side:

  FIX 1 — frozen-base WEIGHT ALIASING (难点 B 的省显存形态):
     The shadow shares the live model's frozen base weight TENSORS (a copy-on-write module tree that
     diverges only at layer-29 down_proj); only the small codebook is cloned (trainable, independent).
     No 16G deepcopy -> peak GPU stays ~1x base; backward only writes the shadow's own value, the
     shared base is read-only for both serve and train -> no corruption.

  FIX 2 — LOCK-FREE, REQUEST-BOUNDARY PROMOTE:
     The worker NEVER touches the live slot. It trains the shadow, then publishes it to `pending`.
     The SERVING thread alone performs the swap (swap_edit_module) at the start of its next request,
     via _maybe_promote(). One writer of the live slot (the serving thread) -> no swap_lock, no
     starvation, swap is genuinely at a request boundary (never mid-decode).

Verified mechanics: HOREN.generate re-reads the slot via eval(f"self.model.{self.layer}") each call
(editor.py:67-73) so swapping the adapter object promotes correctly; swap_edit_module is one setattr
(model_host.py:69-76); an empty codebook (len(keys)==1) makes forward return base output (editor.py:305).

Sanity = phases [1]-[5] at N=1; STOP & report. Phase [6] (N>=2 carryover + swap-then-drop) is gated.
"""
import copy
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


def _cow(m):
    """Copy-on-write clone of an nn.Module: shares every child / param / buffer by REFERENCE but gets
    its OWN _modules/_parameters/_buffers dicts, so a setattr on the clone never leaks to the original.
    Used to build a shadow model tree that diverges from the live model at exactly one submodule."""
    new = copy.copy(m)
    new._modules = dict(m._modules)
    new._parameters = dict(m._parameters)
    new._buffers = dict(m._buffers)
    return new


def main():
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
    live_hf = model_host.current_model()  # the LlamaForCausalLM (captured before wrapping)

    live_wrapper = HOREN(config=hp, model=live_hf)  # installs an EMPTY adapter in the live slot
    live_adapter = model_host.edit_module()
    model_host.register_edit_module(live_adapter, edited_model=live_wrapper)
    gpu_after_load = gpu_gb()
    print(f"   live wrapped; codebook={len(live_adapter.keys)} (placeholder only); "
          f"peak GPU after load = {gpu_after_load:.1f} GB / 48 GB")

    _nm = hp.inner_params[0]
    if _nm.endswith((".weight", ".bias")):
        _nm = _nm.rsplit(".", 1)[0]
    _attr = _nm.rsplit(".", 1)[-1]

    def slot_adapter(hf):
        return getattr(parent_module(hf, brackets_to_periods(_nm)), _attr)

    # ───────────────── FIX 1: frozen-base-aliasing shadow (no deepcopy of weights) ─────────────────
    def build_shadow():
        """A shadow model that SHARES the live frozen base weights and carries a CLONED, independent
        codebook at the slot. Cheap: only a handful of module objects + the tiny codebook are new."""
        orig = model_host._S["original"]  # the frozen base down_proj Linear (shared by all adapters)
        live_ad = model_host.edit_module()
        # clone the codebook (keys/values/key_labels -> trainable & independent) but SHARE the base
        # Linear (orig) + its weight via the deepcopy memo, so no 117MB Linear copy is made either.
        shad = copy.deepcopy(live_ad, {id(orig): orig, id(orig.weight): orig.weight})
        # copy-on-write model tree: diverge ONLY at model.layers[29].mlp.down_proj
        sm = _cow(live_hf)
        sm.model = _cow(live_hf.model)
        sm.model.layers = _cow(live_hf.model.layers)
        l29 = _cow(live_hf.model.layers[29])
        l29.mlp = _cow(live_hf.model.layers[29].mlp)
        l29.mlp.down_proj = shad
        sm.model.layers[29] = l29
        assert slot_adapter(sm) is shad, "COW shadow slot did not resolve to the shadow adapter"
        return sm, shad

    # ───────────────── FIX 2: lock-free, request-boundary promote ─────────────────
    edit_q: "queue.Queue" = queue.Queue()
    pending_lock = threading.Lock()
    pending = {"adapter": None, "tag": None}
    submit_t: dict = {}
    promote_lat: dict = {}
    worker_log: dict = {}

    def append_chat_key(wrapper, adapter, stem):
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
                shadow_model, shad = build_shadow()         # shares frozen base; clones codebook
                wrapper, _ = apply_horen_to_model(shadow_model, tok, [req], hp, copy=False)  # train shadow
                append_chat_key(wrapper, shad, req["prompt"])
                rec["train_s"] = time.time() - t0
                rec["codebook"] = len(shad.keys)
                with pending_lock:                           # publish; the SERVING thread will swap it in
                    pending["adapter"], pending["tag"] = shad, tag
                rec["ok"] = True
                del wrapper, shadow_model                    # drop the COW tree (weights were shared, ~0 freed)
            except Exception as e:                           # crash isolation: non-fatal; live slot untouched
                rec["error"] = repr(e)
            finally:
                worker_log[tag] = rec
                edit_q.task_done()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    def maybe_promote():
        """SERVING-thread-only: atomically swap in a ready shadow at a request boundary. Sole writer
        of the live slot -> no lock vs the worker, no mid-decode swap."""
        with pending_lock:
            ad, tag = pending["adapter"], pending["tag"]
            pending["adapter"], pending["tag"] = None, None
        if ad is not None:
            model_host.swap_edit_module(ad)                  # single setattr (model_host.py:69-76)
            model_host._S["adapter"] = ad
            promote_lat[tag] = time.time() - submit_t[tag]
            return True
        return False

    def submit(req, tag, fail=False):
        submit_t[tag] = time.time()
        edit_q.put((req, tag, fail))
        return (time.time() - submit_t[tag]) * 1000.0  # ms

    def serve(query, mnt, chat=True):
        maybe_promote()                                      # promote at the request boundary, then decode
        return gen.generate(query, model=model_host.current_model(), tok=tok,
                            max_new_tokens=mnt, use_chat_template=chat, with_rag=False)

    def serve_timed(query, mnt):
        t = time.time()
        a = serve(query, mnt)
        return a, time.time() - t

    def live_score(query):
        maybe_promote()
        ad = model_host.edit_module()
        rk = compute_key(query, templated=True, hf_model=live_hf, tok=tok, adapter=ad)
        return ad._query(rk).max().item()

    # ───────────────────── pick fact0 + idle baseline + pre-edit state ─────────────────────
    s = load_type_a(N)[0]
    stem, para, tgt = s["edit_prompt"], s["queries"][0]["q"], s["target_new"]
    mnt = len(tok.encode(" " + tgt, add_special_tokens=False)) + 6
    req = {"prompt": stem, "target_new": tgt}
    print(f"   fact0 {s['id']}: stem={stem!r}  target={tgt!r}")

    serve(UNRELATED, mnt)  # warmup
    idle = [serve_timed(UNRELATED, mnt)[1] for _ in range(3)]
    idle_ms = 1000 * sum(idle) / len(idle)
    print(f"   idle serving latency (unrelated, mean of 3) = {idle_ms:.0f} ms")

    before_known = tgt.lower() in serve(stem, mnt).lower()
    before_score = live_score(stem)
    print(f"   PRE-edit: fact0 known? {before_known}   live_score(stem) = {before_score:.3f}  (< {thr} expected)")

    # ───────────────────── [2] non-blocking accept ─────────────────────
    print("\n== [2] non-blocking accept: submit returns immediately; training runs in the background ==")
    reset_peak()
    sub_ms = submit(req, "fact0")
    print(f"   submit() returned in {sub_ms:.2f} ms (training now runs on the worker thread)")

    # ───────────────── [3] serve-during-train (aliased shadow isolation) ─────────────────
    print("\n== [3] serve-during-train (难点 B): serving stays correct + live while the edit trains ==")
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
    print(f"   PEAK GPU during async edit (aliased shadow + train + concurrent serving) = {edit_peak:.1f} GB / 48 GB")

    # ───────────────── [4] atomic promotion (post-swap, on the serving thread) ─────────────────
    print("\n== [4] atomic promotion: serving thread swaps the ready shadow in at its next request ==")
    after_stem, after_para, after_unrel = live_score(stem), live_score(para), live_score(UNRELATED)
    a_stem, a_para, a_unrel = serve(stem, mnt), serve(para, mnt), serve(UNRELATED, mnt)

    def hit(a):
        return tgt.lower() in a.lower()

    print(f"   promotion landed {promote_lat.get('fact0', float('nan')):.2f}s after submit "
          f"(= train_s + time-to-next-request; no swap_lock starvation)")
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
    edit_q.join()
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

    print("\n==================== SPIKE 1 (v0.7) async-training VERDICT [v2] ====================")
    print(f"  [1] fits 48GB           : {fits}  (peak {edit_peak:.1f} GB — aliased base, no 2x copy)")
    print(f"  [2] non-block accept    : {sub_ms < 50}  ({sub_ms:.2f} ms)")
    print(f"  [3] serve-during-train  : pre-swap fact0 unknown={not before_known}, "
          f"serving latency x{train_ms / idle_ms:.2f} vs idle (难点 A contention)")
    print(f"  [4] atomic promotion    : fires+answers={after_stem >= thr and hit(a_stem)}, "
          f"locality={after_unrel < thr}, promote@{promote_lat.get('fact0', float('nan')):.1f}s")
    print(f"  [5] crash isolation     : {crash_ok}")
    print(f"  VERDICT : {'PASS ✅' if green else 'CHECK ⚠️'}   (N={N} sanity; phase [6] N>=2 gated)")
    print("===================================================================================")

    edit_q.put(None)
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
