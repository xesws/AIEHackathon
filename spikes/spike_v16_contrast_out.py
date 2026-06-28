"""SPIKE v1.6 — contrast-out (codebook mean-centering) as a candidate locality fix (MEASUREMENT-FIRST).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v16_contrast_out.py

Plan: docs/v1.6-contrast-out-locality-fix.md   (prior localization: docs/v1.5-locality-debug-ladder.md)

v1.5 localized the chat-path locality collapse: chat keying (scaffold + flat-mean pooling) lifts the
whole key cone past the fixed 0.85 threshold, so sibling NEG queries misfire (S3: fact NEGxfire 11/20,
belief 7/20). HYPOTHESIS (to TEST, not a known fix): the scaffold/question-stem contamination is a SHARED
component across every key; subtracting it (codebook mean-centering) should re-expand intra-cone distance
WITHOUT just translating the cone below threshold.

contrast-out (this spike, toggle; live code UNTOUCHED = flag default OFF by construction):
  for each key k -> k' = normalize(k - c) ; c = the shared component. Then run the SAME production gate
  (keying.score -> adapter._query, threshold 0.85). c is subtracted from BOTH write-keys AND read-keys
  (critical: same subtraction both sides, renormalized — else the comparison is meaningless).

Shared component c = CODEBOOK MEAN (option a; ROLE title = "codebook mean-centering"). a is parameter-free
and captures scaffold + question-stem + pooling-DC all at once (option b, an empty-query scaffold baseline,
estimates only the scaffold slice -> a is the superset estimate). Two granularities reported:
  global    : mean of ALL 10 write-keys (= the live mixed codebook)            [headline]
  per-group : mean of the 5 fact write-keys / 5 belief write-keys              [diagnostic: cross-cone offset]

Experiment = the v1.5 S3 config (= live chat key), three conditions in one table, SAME 10 real samples,
SAME seed/keyer, threshold 0.85 unchanged, scaffold/pooling-% unchanged (contrast is the ONLY new variable):
  S3                : no contrast            (baseline; MUST reproduce v1.5 -> integrity gate)
  S3 + contrast(global)
  S3 + contrast(group)
Per cond x group(fact/belief): per-sample DIRECT/PARA/worst-NEG + aggregates DIRECT|PARA|NEGfloor|PARAfire|NEGxfire.

SUCCESS = BOTH simultaneously: (1) NEGxfire drops sharply toward 0/20, AND (2) PARAfire stays ~5/5.
If NEGxfire drops but PARAfire ALSO drops -> FALSE-FIX (cone merely translated below thr) -> report FAILURE.
Loads its own model copy; touches NOTHING (no server/memory/codebook/editing/samples.json). Reuses production
keying (HOREN install, _pool_span, _hero_render, query_span_in_rendered, _query), drift-guarded vs compute_key.
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
from src.models.horen.editor import HOREN  # noqa: E402  (production adapter install)
from src.models.horen.horen_hparams import HORENHyperParams  # noqa: E402

MODEL = "/workspace/hugging_cache/llama3.1-8b-instruct"
HPARAMS = os.path.join(_REPO, "third_party", "horen", "hparams", "HOREN", "llama3.1-8b.yaml")
DEV, THR = "cuda:0", 0.85
FACT_IDS = ["A0001", "A0002", "A0003", "A0004", "A0005"]
BELI_IDS = ["A0139", "A0140", "A0141", "A0142", "A0143"]

print("torch", torch.__version__, torch.cuda.is_available())

# --------------------------------------------------------------------------- #
# 1) Real samples (same 10 ids as v1.5; S3 uses the shared-JQ facts + flat beliefs)
# --------------------------------------------------------------------------- #
SAMPLES = {s["id"]: s for s in json.load(open(os.path.join(_REPO, "eval", "samples.json")))["samples"]}


def build_items(ids):
    out = []
    for iid in ids:
        s = SAMPLES[iid]
        out.append(
            {
                "id": iid,
                "edit": s["edit_prompt"],
                "src": s["queries"][0]["q"],
                "para": s["queries"][1]["q"],
                "target": s["target_new"],
            }
        )
    return out


FACTS = build_items(FACT_IDS)   # shared-JQ facts (the S3 / live arm)
BELIEFS = build_items(BELI_IDS)  # flat beliefs (control)

print("\n=== samples (real ids from eval/samples.json; same 10 as v1.5 S3 arm) ===")
for it in FACTS:
    print(f"  [FACT  {it['id']}] target={it['target']!r}  edit={it['edit']!r}")
    print(f"        src : {it['src']}   ||  para: {it['para']}")
for it in BELIEFS:
    print(f"  [BELI  {it['id']}] target={it['target']!r}  edit={it['edit']!r}")
    print(f"        src : {it['src']}   ||  para: {it['para']}")

# --------------------------------------------------------------------------- #
# 2) Own model copy + production adapter (HOREN), config from the real yaml
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
print(
    "\nadapter:", type(adapter).__name__,
    "| normalize", adapter.normalize_codebook_keys,
    "| qspan", adapter.query_span_pool_strategy,
    "| hopfield(beta,alpha,iter,eps)",
    (adapter.hopfield_retrieval_beta, adapter.hopfield_retrieval_alpha,
     adapter.hopfield_retrieval_max_iter, adapter.hopfield_retrieval_eps),
    "| thr", adapter.hopfield_key_match_threshold,
)

# --------------------------------------------------------------------------- #
# 3) S3 chat key extraction — capture-once, production _pool_span (== compute_key templated=True)
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


def key_chat_span(text):  # S3 live chat key (== compute_key templated=True)
    rendered = _hero_render(tok, text)
    x = _capture(rendered)
    s, e = query_span_in_rendered(tok, rendered, text)
    return _norm(adapter._pool_span(x, s, e))


# Drift guard: inlined S3 extractor MUST equal production compute_key(templated=True).
for t in (FACTS[0]["src"], BELIEFS[0]["src"]):
    a = key_chat_span(t)
    b = compute_key(t, templated=True, hf_model=model.model, tok=tok, adapter=adapter)
    assert torch.allclose(a, b, atol=1e-3), ("S3 key != compute_key(templated=True)", (a - b).abs().max().item())
print("drift-guard OK: key_chat_span == production compute_key(templated=True) (atol 1e-3)\n")

# --------------------------------------------------------------------------- #
# 4) contrast-out toggle + per-condition evaluation
# --------------------------------------------------------------------------- #
def s3keys(items):
    return (
        [key_chat_span(it["edit"]) for it in items],  # write/stored key (cloze stem)
        [key_chat_span(it["src"]) for it in items],   # DIRECT probe
        [key_chat_span(it["para"]) for it in items],  # PARA probe
    )


KE_F, KS_F, KP_F = s3keys(FACTS)
KE_B, KS_B, KP_B = s3keys(BELIEFS)


def centroid(keys):
    """Mean of the (unit-norm) write-keys = the cone centroid (NOT renormalized). [1,D] float32."""
    return torch.cat([k.float() for k in keys], dim=0).mean(dim=0, keepdim=True)


C_GLOBAL = centroid(KE_F + KE_B)   # live mixed codebook
C_FACT = centroid(KE_F)
C_BELI = centroid(KE_B)
print(f"||c_global||={C_GLOBAL.norm().item():.4f}  ||c_fact||={C_FACT.norm().item():.4f}  "
      f"||c_beli||={C_BELI.norm().item():.4f}   cos(c_fact,c_beli)="
      f"{F.cosine_similarity(C_FACT, C_BELI).item():.4f}\n")


def contrast(keys, c):
    """Subtract shared component c then RENORMALIZE (cosine on the sphere needs unit keys).
    c=None -> identity (== the exact v1.5 S3 keys, byte-faithful baseline)."""
    if c is None:
        return list(keys)
    return [F.normalize(k.float() - c, p=2, dim=-1) for k in keys]


def eval_cond(kedit, ksrc, kpara, c):
    ke, ks, kp = contrast(kedit, c), contrast(ksrc, c), contrast(kpara, c)
    n = len(ke)
    direct = [kscore(ks[i], ke[i], adapter) for i in range(n)]
    para = [kscore(kp[i], ke[i], adapter) for i in range(n)]
    negmat = [[None] * n for _ in range(n)]  # negmat[i][j] = sibling j's src probed at edit i
    negv = []
    for i in range(n):
        for j in range(n):
            if j != i:
                v = kscore(ks[j], ke[i], adapter)
                negmat[i][j] = v
                negv.append(v)
    worst_neg = [max(v for v in negmat[i] if v is not None) for i in range(n)]  # per-sample worst sibling
    return {
        "direct": direct, "para": para, "worst_neg": worst_neg, "negv": negv,
        "DIRECT": sum(direct) / n, "PARA": sum(para) / n,
        "NEGfloor": max(negv), "NEGmean": sum(negv) / len(negv),
        "PARAfire": sum(v > THR for v in para), "NEGxfire": sum(v > THR for v in negv), "NEGn": len(negv),
    }


CONDS = [
    ("S3", "no contrast (== v1.5 S3)", None, None),
    ("S3+contrast(global)", "subtract global codebook mean", C_GLOBAL, C_GLOBAL),
    ("S3+contrast(group)", "subtract per-group mean", C_FACT, C_BELI),
]

results = {}
for name, _desc, cf, cb in CONDS:
    results[name] = {
        "fact": eval_cond(KE_F, KS_F, KP_F, cf),
        "belief": eval_cond(KE_B, KS_B, KP_B, cb),
    }

# --------------------------------------------------------------------------- #
# 5) Integrity gate — S3 (no contrast) MUST reproduce v1.5 exactly
# --------------------------------------------------------------------------- #
bf, bb = results["S3"]["fact"], results["S3"]["belief"]
print(f"integrity: S3 baseline fact NEGxfire={bf['NEGxfire']}/20 (v1.5=11)  PARAfire={bf['PARAfire']}/5 (v1.5=5)  "
      f"| belief NEGxfire={bb['NEGxfire']}/20 (v1.5=7)  PARAfire={bb['PARAfire']}/5 (v1.5=5)")
assert bf["NEGxfire"] == 11 and bf["PARAfire"] == 5, "S3 fact does not reproduce v1.5 -> harness drift, STOP"
assert bb["NEGxfire"] == 7 and bb["PARAfire"] == 5, "S3 belief does not reproduce v1.5 -> harness drift, STOP"
print("integrity OK: S3 baseline reproduces v1.5 -> contrast is the ONLY new variable\n")

# --------------------------------------------------------------------------- #
# 6) Report
# --------------------------------------------------------------------------- #
print(f"{'='*120}\nCONTRAST-OUT vs S3  (threshold={THR};  ★ verdict = JOINT move of (NEGxfire down, PARAfire kept))\n{'-'*120}")
hdr = f"  {'condition':24}{'group':8}{'DIRECT':>8}{'PARA':>8}{'NEGflr':>8}{'NEGmean':>9}{'PARAfire':>10}{'NEGxfire':>11}"
print(hdr)
for name, _d, _cf, _cb in CONDS:
    for grp in ("fact", "belief"):
        r = results[name][grp]
        print(f"  {name:24}{grp:8}{r['DIRECT']:8.3f}{r['PARA']:8.3f}{r['NEGfloor']:8.3f}{r['NEGmean']:9.3f}"
              f"{str(r['PARAfire'])+'/5':>10}{str(r['NEGxfire'])+'/'+str(r['NEGn']):>11}")
    print()

# per-sample DIRECT / PARA / worst-NEG for the headline conditions
for name in ("S3", "S3+contrast(global)"):
    print(f"-- per-sample (cond={name};  fire if > {THR}) --")
    for grp, items in (("fact", FACTS), ("belief", BELIEFS)):
        r = results[name][grp]
        for i, it in enumerate(items):
            d, p, w = r["direct"][i], r["para"][i], r["worst_neg"][i]
            print(f"   {grp:6} {it['id']}  DIRECT={d:.3f}{'*' if d>THR else ' '}  "
                  f"PARA={p:.3f}{'*' if p>THR else ' '}  worstNEG={w:.3f}{'  <-XFIRE' if w>THR else ''}")
    print()

# --------------------------------------------------------------------------- #
# 7) Joint verdict (anti-false-fix): need NEGxfire DOWN *and* PARAfire KEPT
# --------------------------------------------------------------------------- #
print(f"{'#'*120}\nJOINT VERDICT  (success = NEGxfire ↓ toward 0  AND  PARAfire stays high)")
for name in ("S3+contrast(global)", "S3+contrast(group)"):
    for grp in ("fact", "belief"):
        base, cur = results["S3"][grp], results[name][grp]
        dneg, dpara = cur["NEGxfire"] - base["NEGxfire"], cur["PARAfire"] - base["PARAfire"]
        neg_fixed = cur["NEGxfire"] <= 2          # near S0's 0/20
        para_kept = cur["PARAfire"] >= 4          # ~5/5 preserved
        if neg_fixed and para_kept:
            verdict = "SUCCESS (locality restored, edits still fire)"
        elif cur["NEGxfire"] < base["NEGxfire"] and not para_kept:
            verdict = "FALSE-FIX (cone translated below thr — PARA died too)"
        elif cur["NEGxfire"] < base["NEGxfire"] and para_kept:
            verdict = "PARTIAL (NEGxfire down but not to ~0; PARA kept)"
        else:
            verdict = "NO-EFFECT / WORSE"
        print(f"  {name:24} {grp:6}: NEGxfire {base['NEGxfire']}/20 -> {cur['NEGxfire']}/20 ({dneg:+d}) ; "
              f"PARAfire {base['PARAfire']}/5 -> {cur['PARAfire']}/5 ({dpara:+d})  => {verdict}")

print(f"\nVRAM allocated={torch.cuda.memory_allocated()/1e9:.1f}G  max={torch.cuda.max_memory_allocated()/1e9:.1f}G")
print("torch", torch.__version__, torch.cuda.is_available())
