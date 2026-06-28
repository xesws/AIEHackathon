"""SPIKE v1.5 — locality-collapse single-variable ladder S0->S3 (English, real samples; MEASUREMENT-ONLY).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v15_locality_ladder.py

Design: docs/horen_debug/exp_debug_design.md  (plan: docs/v1.5-locality-debug-ladder.md)

We have two endpoints:
  CORRECT end : ZsRE config (raw prompt + 60% pool + no scaffold + heterogeneous subjects) = locality 1.0
  BROKEN  end : chat  config (query-span flat-mean + chat scaffold + shared JQ frame)      = 0.88 cone collapse
3 variables sit between them. Starting from the correct end we add ONE contaminant per rung
and watch where locality (margin = PARA - NEG) first collapses -> that rung's variable is the culprit.

Rungs (each changes exactly ONE thing vs the previous):
  S0  ZsRE baseline : raw prompt + _select_query(last_60%) ; facts de-JQ'd (distinct neutral subjects)
  S1  +JQ frame     : raw prompt + _select_query(last_60%) ; facts revert to the SHARED "JQ" subject   [data-only delta]
  S2  +span pooling : raw prompt + _pool_span(flat over query span)                                    [pooler-only delta]
  S3  +chat scaffold: hero chat render + _pool_span(flat over query span) == the LIVE chat key          [scaffold-only delta]
S0/S1/S2 share the IDENTICAL raw forward per text (capture-once) so S1->S2 isolates the pooler exactly.

10 real samples by id from eval/samples.json (no hand-authoring):
  facts  = A0001-A0005  (A / type=Y / zero_prior, subject=JQ; 5 distinct categories -> shared frame, diff topics)
  belief = A0139-A0143  (A / type=X, subject=None; flat ZsRE-style assertions -> control, no JQ to share)
S0 de-JQ is a MECHANICAL token swap "JQ"->{Riley,Jordan,Casey,Morgan,Quinn} on each fact's edit+queries
(printed below). It changes only the proper noun, not sentence structure: the minimal realization of the
design's "independent question / no shared subject". Beliefs carry no JQ so they are identical at every rung.

Per item: write/stored key = key(edit_prompt) (cloze stem, the production write arm, = v1.4 headline);
probes DIRECT=queries[0] / PARA=queries[1] / NEG=a sibling's DIRECT (same group). Score = the PRODUCTION
deferral gate (keying.score -> adapter._query, Hopfield alpha/beta/iter from yaml), threshold 0.85.
margin = mean(PARA) - worst-sibling NEG = separability. Loads its own model copy; touches NOTHING
(no server / memory / codebook / editing). Reuses production keying: HOREN install, _select_query,
_pool_span/pool_span_rows, _hero_render, query_span_in_rendered, _query (drift-guarded vs compute_key).
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
    locate_query_span,
    query_span_in_rendered,
    score as kscore,
)
from src.models.horen.editor import HOREN  # noqa: E402  (production adapter install)
from src.models.horen.horen_hparams import HORENHyperParams  # noqa: E402

MODEL = "/workspace/hugging_cache/llama3.1-8b-instruct"
HPARAMS = os.path.join(_REPO, "third_party", "horen", "hparams", "HOREN", "llama3.1-8b.yaml")
DEV, THR = "cuda:0", 0.85
NAMES = ["Riley", "Jordan", "Casey", "Morgan", "Quinn"]  # distinct neutral subjects for S0 de-JQ
FACT_IDS = ["A0001", "A0002", "A0003", "A0004", "A0005"]
BELI_IDS = ["A0139", "A0140", "A0141", "A0142", "A0143"]

print("torch", torch.__version__, torch.cuda.is_available())

# --------------------------------------------------------------------------- #
# 1) Real samples -> per-rung items {id, edit, src, para, target}
# --------------------------------------------------------------------------- #
SAMPLES = {s["id"]: s for s in json.load(open(os.path.join(_REPO, "eval", "samples.json")))["samples"]}


def build_items(ids, *, de_jq):
    """de_jq=True (S0): swap JQ->distinct neutral name on FACTS (type Y). Beliefs (type X) untouched."""
    out = []
    for k, iid in enumerate(ids):
        s = SAMPLES[iid]
        is_fact = s.get("type") == "Y"
        nm = NAMES[k % len(NAMES)]
        f = (lambda t: t.replace("JQ", nm)) if (is_fact and de_jq) else (lambda t: t)
        out.append(
            {
                "id": iid,
                "edit": f(s["edit_prompt"]),
                "src": f(s["queries"][0]["q"]),
                "para": f(s["queries"][1]["q"]),
                "target": s["target_new"],
            }
        )
    return out


FACTS_S0 = build_items(FACT_IDS, de_jq=True)   # distinct subjects (independent questions)
FACTS_JQ = build_items(FACT_IDS, de_jq=False)  # shared JQ subject (S1/S2/S3)
BELIEFS = build_items(BELI_IDS, de_jq=False)   # no JQ -> identical at every rung

print("\n=== constructed samples (real ids; de-JQ is a token swap, printed for transparency) ===")
print("-- FACTS: S0 (de-JQ, distinct subject)  ||  S1+ (shared JQ) --")
for a, b in zip(FACTS_S0, FACTS_JQ):
    print(f"  [{a['id']}] target={b['target']!r}")
    print(f"     S0 edit : {a['edit']}")
    print(f"     S1 edit : {b['edit']}")
    print(f"     S0 src  : {a['src']}   ||  S1 src : {b['src']}")
    print(f"     S0 para : {a['para']}   ||  S1 para: {b['para']}")
print("-- BELIEFS (control; identical all rungs) --")
for it in BELIEFS:
    print(f"  [{it['id']}] target={it['target']!r}  edit={it['edit']!r}")
    print(f"     src : {it['src']}   ||  para: {it['para']}")

# --------------------------------------------------------------------------- #
# 2) Load own model copy + install production adapter (HOREN), config from yaml
# --------------------------------------------------------------------------- #
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = HORENHyperParams.from_hparams(HPARAMS)
HOREN(cfg, model)  # installs HopfieldAdapter at layer-29 down_proj (production path), 1 default key
adapter = model.model.layers[29].mlp.down_proj
assert type(adapter).__name__ == "HopfieldAdapter", type(adapter).__name__
assert adapter.normalize_codebook_keys is True
assert adapter.query_selection_strategy == "last_60_perc_prompt_tokens_avg"
assert adapter.query_span_pool_strategy == "flat"
assert abs(adapter.hopfield_key_match_threshold - 0.85) < 1e-9
print(
    "\nadapter:", type(adapter).__name__,
    "| normalize", adapter.normalize_codebook_keys,
    "| qsel", adapter.query_selection_strategy,
    "| qspan", adapter.query_span_pool_strategy,
    "| hopfield(beta,alpha,iter,eps)",
    (adapter.hopfield_retrieval_beta, adapter.hopfield_retrieval_alpha,
     adapter.hopfield_retrieval_max_iter, adapter.hopfield_retrieval_eps),
    "| thr", adapter.hopfield_key_match_threshold,
)

# --------------------------------------------------------------------------- #
# 3) Key extraction — capture-once, pool per-rung with the PRODUCTION operators
# --------------------------------------------------------------------------- #
_CAP = {}  # forward-string -> captured layer-29 down_proj input rows [1, seq, D]


def _capture(forward_text):
    if forward_text in _CAP:
        return _CAP[forward_text]
    enc = tok(forward_text, return_tensors="pt").to(DEV)
    cap = {}
    h = adapter.register_forward_pre_hook(lambda _m, a: cap.__setitem__("x", a[0]))
    old = adapter.adapter_mode
    adapter.adapter_mode = "none"  # pure capture (no match/inject/state mutation)
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


def key_raw_select(text):   # S0/S1: raw prompt + _select_query(last_60%)  (== compute_key templated=False)
    x = _capture(text)
    return _norm(adapter._select_query(x, x.shape[1] - 1))


def key_raw_span(text):     # S2: raw prompt + _pool_span(flat) over the whole-prompt query span
    x = _capture(text)
    s, e = locate_query_span(tok, text, templated=False)
    return _norm(adapter._pool_span(x, s, e))


def key_chat_span(text):    # S3: hero chat render + _pool_span(flat) over query span  (== compute_key templated=True)
    rendered = _hero_render(tok, text)
    x = _capture(rendered)
    s, e = query_span_in_rendered(tok, rendered, text)
    return _norm(adapter._pool_span(x, s, e))


# Drift guard: the inlined extractors MUST reproduce the production compute_key on both endpoints.
for t in (FACTS_JQ[0]["src"], BELIEFS[0]["src"]):
    a = key_raw_select(t)
    b = compute_key(t, templated=False, hf_model=model.model, tok=tok, adapter=adapter)
    assert torch.allclose(a, b, atol=1e-3), ("S0/S1 key != compute_key(templated=False)", (a - b).abs().max().item())
    c = key_chat_span(t)
    d = compute_key(t, templated=True, hf_model=model.model, tok=tok, adapter=adapter)
    assert torch.allclose(c, d, atol=1e-3), ("S3 key != compute_key(templated=True)", (c - d).abs().max().item())
print("drift-guard OK: inlined extractors == production compute_key on both endpoints (atol 1e-3)\n")

# --------------------------------------------------------------------------- #
# 4) Evaluate a group at one rung: DIRECT / PARA / NEG / margin via production gate
# --------------------------------------------------------------------------- #
def eval_group(items, keyer):
    kedit = [keyer(it["edit"]) for it in items]  # write/stored key (cloze stem)
    ksrc = [keyer(it["src"]) for it in items]    # DIRECT probe
    kpara = [keyer(it["para"]) for it in items]  # PARA probe
    direct = [kscore(ksrc[i], kedit[i], adapter) for i in range(len(items))]
    para = [kscore(kpara[i], kedit[i], adapter) for i in range(len(items))]
    negpairs = []
    for i in range(len(items)):  # NEG = sibling's DIRECT src probed against edit i (intra-group cross-talk)
        for j in range(len(items)):
            if j != i:
                negpairs.append(kscore(ksrc[j], kedit[i], adapter))
    n = len(items)
    return {
        "DIRECT": sum(direct) / n,
        "PARA": sum(para) / n,
        "NEGfloor": max(negpairs),            # worst-case cross-talk = locality floor
        "NEGmean": sum(negpairs) / len(negpairs),
        "PARAfire": sum(v > THR for v in para),
        "NEGxfire": sum(v > THR for v in negpairs),
        "NEGn": len(negpairs),
        "margin": (sum(para) / n) - max(negpairs),
    }


RUNGS = [
    ("S0", "ZsRE baseline (raw+60%, de-JQ facts)", key_raw_select, FACTS_S0),
    ("S1", "+JQ frame (raw+60%, shared JQ)", key_raw_select, FACTS_JQ),
    ("S2", "+query-span flat pool (raw, JQ)", key_raw_span, FACTS_JQ),
    ("S3", "+chat scaffold (= live chat key)", key_chat_span, FACTS_JQ),
]

rows = []
for lbl, desc, keyer, facts in RUNGS:
    rows.append((lbl, desc, "fact", eval_group(facts, keyer)))
    rows.append((lbl, desc, "belief", eval_group(BELIEFS, keyer)))

# --------------------------------------------------------------------------- #
# 5) Report
# --------------------------------------------------------------------------- #
print(f"\n{'='*118}\nLOCALITY LADDER  (threshold={THR}; margin = mean(PARA) - worst-sibling NEG; healthy = margin big & NEGxfire 0)\n{'-'*118}")
hdr = f"  {'rung':4}{'group':8}{'DIRECT':>8}{'PARA':>8}{'NEGflr':>8}{'NEGmean':>9}{'margin':>9}{'PARAfire':>10}{'NEGxfire':>11}   change vs prev"
print(hdr)
for lbl, desc, grp, r in rows:
    chg = desc if grp == "fact" else ""
    print(
        f"  {lbl:4}{grp:8}{r['DIRECT']:8.3f}{r['PARA']:8.3f}{r['NEGfloor']:8.3f}{r['NEGmean']:9.3f}"
        f"{r['margin']:+9.3f}{str(r['PARAfire'])+'/5':>10}{str(r['NEGxfire'])+'/'+str(r['NEGn']):>11}   {chg}"
    )

# §4 answers
def grp_row(lbl, grp):
    return next(r for (l, _d, g, r) in rows if l == lbl and g == grp)


print(f"\n{'#'*118}\n§4 READOUT")
fact = {l: grp_row(l, "fact") for l in ("S0", "S1", "S2", "S3")}
beli = {l: grp_row(l, "belief") for l in ("S0", "S1", "S2", "S3")}
print(f"  Q1 S0 baseline healthy?  fact margin={fact['S0']['margin']:+.3f} (NEGxfire {fact['S0']['NEGxfire']}/{fact['S0']['NEGn']}) ; "
      f"belief margin={beli['S0']['margin']:+.3f}.  (margin>~0.05 & 0 xfire => healthy; if S0 broken => data/measure issue)")
print("  Q2 fact margin by rung :  " + "  ".join(f"{l} {fact[l]['margin']:+.3f}" for l in ("S0", "S1", "S2", "S3")))
print("     belief margin by rung:  " + "  ".join(f"{l} {beli[l]['margin']:+.3f}" for l in ("S0", "S1", "S2", "S3")))
drops = [l for l in ("S1", "S2", "S3") if fact[l]["margin"] < fact["S0"]["margin"] - 0.03 or fact[l]["NEGxfire"] > 0]
print(f"     => fact locality first degrades at: {drops[0] if drops else 'NONE (stays healthy through S3)'}")
print(f"  Q3 fact vs belief diverge at S1?  fact S0->S1 margin {fact['S0']['margin']:+.3f}->{fact['S1']['margin']:+.3f} ; "
      f"belief {beli['S0']['margin']:+.3f}->{beli['S1']['margin']:+.3f}  (belief should not move S0->S1; fact moving => JQ-frame effect)")
print("  Q4 one-line: the culprit is whichever rung first collapses fact margin above (S1=JQ frame / S2=pooling / S3=scaffold), beliefs as control.")

print(f"\nVRAM allocated={torch.cuda.memory_allocated()/1e9:.1f}G  max={torch.cuda.max_memory_allocated()/1e9:.1f}G")
print("torch", torch.__version__, torch.cuda.is_available())
