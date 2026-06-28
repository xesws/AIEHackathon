"""SPIKE v1.8 — Hopfield retrieval-β GLOBAL sweep: can sharpening kill sibling cross-talk
without collapsing PARA? (MEASUREMENT-ONLY, toggle/default OFF, not wired)

Run:  cd /workspace/AIEHackathon && python spikes/spike_v18_beta_sweep.py

Plan: docs/v1.8-hopfield-beta-sweep.md   (prior: v1.5 ladder, v1.6 contrast-out, v1.7 NEG-ablation)

Question: v1.7 narrowed the real defect to SAME-SUBJECT sibling cross-talk (asking F1's read-key
also fires F2's stored key; fact NEGxfire 11/20). A candidate fix is to SHARPEN Hopfield retrieval
-- raise beta so the read-key only recognizes its NEAREST stored key. We sweep ONE variable, the
GLOBAL `hopfield_retrieval_beta` (NO dynamic / per-query beta -- excluded: lacks a clean "how much"
signal and breaks reproducibility), on the exact v1.7 S3 sibling-NEG harness, everything else FROZEN
(same 10 real samples, same seed, same keying.score gate, threshold 0.85, same scaffold/pooling/keys).

  beta in {20(baseline), 40, 80, 160, 320}              <- task-specified set (headline)
  beta in {1, 2, 5, 10}                                  <- diagnostic extension (< 20, same single
                                                            variable) to show the non-saturated regime

KEY FACT exploited: `compute_key` does NOT use beta (no _query call) -> the S3 keys are beta-INVARIANT.
So we extract the 30 keys ONCE and only re-SCORE under each beta (score() -> adapter._query reads
adapter.hopfield_retrieval_beta at call time). write & read use the SAME beta (one _query, one fixed
random placeholder key). Only beta moves; max_iter=1 / alpha=0.1 / eps / threshold all stay at yaml.

ANTI-FALSE-FIX verdict (joint, same as prior rounds): a beta is a SOLUTION only if NEGxfire drops
significantly WHILE PARAfire stays ~5/5. If NEGxfire and PARAfire always move TOGETHER (both saturate
high, or both slide down) -> no clean window -> beta can't separate cat/snack: their keys are too
close and retrieval sharpness can't fix representation overlap. Report whichever is true, honestly.

Loads its own model copy; touches NOTHING (no server/codebook/memory/editing/samples.json/frontend).
Integrity gate: beta=20 must EXACTLY reproduce S3 (fact NEGxfire 11/20, belief 7/20, PARAfire 5/5).
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "third_party", "horen"))
from keying import (  # noqa: E402  (production keying — single source of truth)
    _hero_render,
    compute_key,
    query_span_in_rendered,
    score as kscore,
)
from src.models.horen.editor import HOREN  # noqa: E402
from src.models.horen.horen_hparams import HORENHyperParams  # noqa: E402

MODEL = "/workspace/hugging_cache/llama3.1-8b-instruct"
HPARAMS = os.path.join(_REPO, "third_party", "horen", "hparams", "HOREN", "llama3.1-8b.yaml")
DEV, THR = "cuda:0", 0.85
FACT_IDS = ["A0001", "A0002", "A0003", "A0004", "A0005"]
BELI_IDS = ["A0139", "A0140", "A0141", "A0142", "A0143"]

BETAS = [20.0, 40.0, 80.0, 160.0, 320.0]      # task-specified headline set (20 = baseline)
BETAS_DIAG = [1.0, 2.0, 5.0, 10.0]            # diagnostic extension (< 20, non-saturated regime)

print("torch", torch.__version__, torch.cuda.is_available())

# --------------------------------------------------------------------------- #
# 1) Real samples (same 10 ids as v1.5/v1.7 S3 arm)
# --------------------------------------------------------------------------- #
SAMPLES = {s["id"]: s for s in json.load(open(os.path.join(_REPO, "eval", "samples.json")))["samples"]}


def build_items(ids):
    out = []
    for iid in ids:
        s = SAMPLES[iid]
        out.append({"id": iid, "edit": s["edit_prompt"], "src": s["queries"][0]["q"],
                    "para": s["queries"][1]["q"], "target": s["target_new"]})
    return out


FACTS = build_items(FACT_IDS)
BELIEFS = build_items(BELI_IDS)

print("\n=== edit samples (real ids; same 10 as v1.5/v1.7) ===")
for it in FACTS + BELIEFS:
    print(f"  [{it['id']}] {it['edit']!r}  | src: {it['src']}")

# --------------------------------------------------------------------------- #
# 2) Own model copy + production adapter (config from yaml)
# --------------------------------------------------------------------------- #
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = HORENHyperParams.from_hparams(HPARAMS)
HOREN(cfg, model)
adapter = model.model.layers[29].mlp.down_proj
assert type(adapter).__name__ == "HopfieldAdapter", type(adapter).__name__
assert adapter.normalize_codebook_keys is True
assert adapter.query_span_pool_strategy == "flat"
assert abs(adapter.hopfield_key_match_threshold - 0.85) < 1e-9
assert abs(adapter.hopfield_retrieval_beta - 20.0) < 1e-9, "yaml baseline beta must be 20"
BASE = (adapter.hopfield_retrieval_alpha, adapter.hopfield_retrieval_max_iter, adapter.hopfield_retrieval_eps)
print(f"\nadapter OK | normalize {adapter.normalize_codebook_keys} | qspan {adapter.query_span_pool_strategy} "
      f"| thr {adapter.hopfield_key_match_threshold} | (alpha,max_iter,eps)={BASE} | baseline beta {adapter.hopfield_retrieval_beta}")
print("NOTE: only `hopfield_retrieval_beta` is swept; alpha/max_iter/eps/threshold/pooling/scaffold/keys all FROZEN.\n")

# --------------------------------------------------------------------------- #
# 3) S3 chat key extraction (== compute_key templated=True), capture-once. beta-INVARIANT.
# --------------------------------------------------------------------------- #
_CAP = {}


def _capture(forward_text):
    if forward_text in _CAP:
        return _CAP[forward_text]
    enc = tok(forward_text, return_tensors="pt").to(DEV)
    cap = {}
    h = adapter.register_forward_pre_hook(lambda _m, a: cap.__setitem__("x", a[0]))
    old = adapter.adapter_mode
    adapter.adapter_mode = "none"
    try:
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        adapter.adapter_mode = old
        h.remove()
    _CAP[forward_text] = cap["x"][:1]
    return _CAP[forward_text]


def _norm(k):
    return F.normalize(k, p=2, dim=-1) if adapter.normalize_codebook_keys else k


def key_chat_span(text):  # S3 live chat key
    rendered = _hero_render(tok, text)
    x = _capture(rendered)
    s, e = query_span_in_rendered(tok, rendered, text)
    return _norm(adapter._pool_span(x, s, e))


# Drift guard: inlined extractor must reproduce production compute_key on both endpoints.
for t in (FACTS[0]["src"], BELIEFS[0]["src"]):
    a = key_chat_span(t)
    b = compute_key(t, templated=True, hf_model=model.model, tok=tok, adapter=adapter)
    assert torch.allclose(a, b, atol=1e-3), ("drift", (a - b).abs().max().item())
print("drift-guard OK: key_chat_span == production compute_key(templated=True)\n")

# --------------------------------------------------------------------------- #
# 4) Keys — computed ONCE (beta-invariant); every beta only re-scores these.
# --------------------------------------------------------------------------- #
def s3keys(items):
    return ([key_chat_span(it["edit"]) for it in items],   # write/stored key (cloze stem)
            [key_chat_span(it["src"]) for it in items],     # DIRECT probe
            [key_chat_span(it["para"]) for it in items])    # PARA probe


KE_F, KS_F, KP_F = s3keys(FACTS)
KE_B, KS_B, KP_B = s3keys(BELIEFS)


# --------------------------------------------------------------------------- #
# 5) Per-beta metrics via the PRODUCTION gate (keying.score -> adapter._query)
# --------------------------------------------------------------------------- #
def neg_sibling(kedit, ksrc):  # NEG = same-group sibling's DIRECT src probed against edit i (j != i)
    return [kscore(ksrc[j], kedit[i], adapter) for i in range(len(kedit)) for j in range(len(kedit)) if j != i]


def metrics(kedit, ksrc, kpara):
    n = len(kedit)
    direct = [kscore(ksrc[i], kedit[i], adapter) for i in range(n)]
    para = [kscore(kpara[i], kedit[i], adapter) for i in range(n)]
    negv = neg_sibling(kedit, ksrc)
    return {
        "DIRECT": sum(direct) / n,
        "PARA": sum(para) / n,
        "NEGfloor": max(negv),                       # worst-case sibling cross-talk = locality floor
        "NEGmean": sum(negv) / len(negv),
        "PARAfire": sum(v > THR for v in para),      # /5
        "NEGxfire": sum(v > THR for v in negv),      # /20
        "NEGn": len(negv),
    }


def sweep_at(beta):
    adapter.hopfield_retrieval_beta = float(beta)     # write & read share THIS beta; only variable changed
    return {"fact": metrics(KE_F, KS_F, KP_F), "belief": metrics(KE_B, KS_B, KP_B)}


RESULTS = {b: sweep_at(b) for b in (BETAS + BETAS_DIAG)}
adapter.hopfield_retrieval_beta = 20.0  # restore baseline (tidy; own throwaway process anyway)

# --------------------------------------------------------------------------- #
# 6) Integrity gate — beta=20 must EXACTLY reproduce S3 (else bug -> STOP)
# --------------------------------------------------------------------------- #
b20 = RESULTS[20.0]
ff, bb = b20["fact"], b20["belief"]
print(f"integrity: beta=20 fact NEGxfire={ff['NEGxfire']}/20 (S3=11), belief NEGxfire={bb['NEGxfire']}/20 (S3=7), "
      f"PARAfire fact={ff['PARAfire']}/5 belief={bb['PARAfire']}/5 (S3=5/5)")
assert ff["NEGxfire"] == 11 and bb["NEGxfire"] == 7, "beta=20 does NOT reproduce S3 NEGxfire -> harness drift, STOP"
assert ff["PARAfire"] == 5 and bb["PARAfire"] == 5, "beta=20 does NOT reproduce S3 PARAfire -> harness drift, STOP"
print("integrity OK: beta=20 reproduces S3 -> beta is the ONLY changed variable\n")

# --------------------------------------------------------------------------- #
# 7) Report table
# --------------------------------------------------------------------------- #
def print_table(betas, title):
    print(f"{'='*104}\n{title}  (threshold={THR}; NEG = same-group sibling DIRECT, 5x4=20; healthy = NEGxfire low & PARAfire 5/5)\n{'-'*104}")
    print(f"  {'beta':>6}{'group':>8}{'DIRECT':>9}{'PARA':>9}{'NEGfloor':>10}{'NEGmean':>9}{'PARAfire':>10}{'NEGxfire':>11}")
    for b in betas:
        for grp in ("fact", "belief"):
            r = RESULTS[b][grp]
            tag = "  <= baseline" if (b == 20.0 and grp == "fact") else ""
            print(f"  {b:>6.0f}{grp:>8}{r['DIRECT']:>9.3f}{r['PARA']:>9.3f}{r['NEGfloor']:>10.3f}{r['NEGmean']:>9.3f}"
                  f"{str(r['PARAfire'])+'/5':>10}{str(r['NEGxfire'])+'/'+str(r['NEGn']):>11}{tag}")
        print()


print_table(BETAS, "BETA SWEEP — task-specified set {20,40,80,160,320}")
print_table(sorted(BETAS_DIAG), "BETA SWEEP — diagnostic extension {1,2,5,10} (< 20: non-saturated regime)")

# --------------------------------------------------------------------------- #
# 8) Adjudication (joint anti-false-fix) + answers to 4 questions
# --------------------------------------------------------------------------- #
ALL = sorted(BETAS + BETAS_DIAG)
print(f"{'#'*104}\nADJUDICATION (joint: NEGxfire DOWN  AND  PARAfire stays ~5/5)")


def clean_window(grp):
    base_neg = RESULTS[20.0][grp]["NEGxfire"]
    hits = []
    for b in ALL:
        r = RESULTS[b][grp]
        if r["NEGxfire"] < base_neg and r["PARAfire"] >= 5:  # NEG strictly down AND PARA fully kept
            hits.append((b, r["NEGxfire"], r["PARAfire"]))
    return hits


for grp in ("fact", "belief"):
    hits = clean_window(grp)
    base = RESULTS[20.0][grp]
    print(f"  [{grp}] baseline(b=20): NEGxfire {base['NEGxfire']}/{base['NEGn']}, PARAfire {base['PARAfire']}/5")
    if hits:
        print(f"        CLEAN-WINDOW beta(s) where NEGxfire<baseline AND PARAfire=5/5: "
              + ", ".join(f"b={b:.0f}(NEG {n}/20)" for b, n, _ in hits))
    else:
        print("        NO clean window: no beta drops NEGxfire below baseline while keeping PARAfire 5/5.")
    # co-movement: show NEGxfire and PARAfire move together across the whole axis
    print("        NEGxfire by beta : " + "  ".join(f"{b:.0f}:{RESULTS[b][grp]['NEGxfire']}" for b in ALL))
    print("        PARAfire by beta : " + "  ".join(f"{b:.0f}:{RESULTS[b][grp]['PARAfire']}" for b in ALL))

# headline-set flatness check (is beta saturated across {20..320}?)
def flat_over(betas, grp):
    negs = {RESULTS[b][grp]["NEGxfire"] for b in betas}
    paras = {RESULTS[b][grp]["PARAfire"] for b in betas}
    return len(negs) == 1 and len(paras) == 1


print(f"\n  Headline set {{20,40,80,160,320}} flat? fact={flat_over(BETAS,'fact')} belief={flat_over(BETAS,'belief')} "
      "(True => beta saturated: raising beta is a no-op on the gate)")

print(f"\n{'#'*104}\nANSWERS")
fact_hits, beli_hits = clean_window("fact"), clean_window("belief")
print(f"  Q1 clean window (NEGxfire DOWN & PARAfire 5/5)? fact: {'YES '+str(fact_hits) if fact_hits else 'NONE'} ; "
      f"belief: {'YES '+str(beli_hits) if beli_hits else 'NONE'}")
print("  Q2 do NEG & PARA move together (no clean window)? see NEGxfire/PARAfire-by-beta rows above.")
print(f"  Q3 fact vs belief: baseline NEGxfire fact {RESULTS[20.0]['fact']['NEGxfire']}/20 vs belief "
      f"{RESULTS[20.0]['belief']['NEGxfire']}/20 ; NEGfloor fact {RESULTS[20.0]['fact']['NEGfloor']:.3f} vs belief "
      f"{RESULTS[20.0]['belief']['NEGfloor']:.3f} (fact siblings more isomorphic = closer).")
print("  Q4 (one-line): see report markdown.")

print(f"\nVRAM allocated={torch.cuda.memory_allocated()/1e9:.1f}G  max={torch.cuda.max_memory_allocated()/1e9:.1f}G")
print("torch", torch.__version__, torch.cuda.is_available())
