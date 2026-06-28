"""SPIKE v1.7 — locality NEG-selection ablation: sibling vs generic-unrelated vs cross-group (MEASUREMENT-ONLY).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v17_neg_ablation.py

Plan: docs/v1.7-locality-neg-selection-ablation.md   (prior: v1.5 ladder, v1.6 contrast-out)

Question: the current S3 locality NEG = the same-group sibling's DIRECT (F1's NEG = F2's question).
The 5 JQ facts share a near-identical surface form ("What is the [X] of JQ's [Y]"), so siblings are
intrinsically very close. Is the locality "failure" (NEGxfire 11/20) a REAL defect (the edit pollutes
UNRELATED knowledge, which is what ZsRE locality measures) or an ARTIFACT of NEG being chosen too close
to DIRECT? We change ONLY the NEG definition (one variable) on the exact v1.6 S3 harness and compare:

  NEG-A (current)        : sibling's DIRECT, same group       (5x4 = 20 pairs)
  NEG-B (ZsRE-style)     : generic world-knowledge questions UNRELATED to any edited fact/belief
                           (capital of France, days in a week, ...; the model already knows them)
                           (8 generic x 5 edits = 40 pairs; same generic pool for both groups)
  NEG-C (cross-group)    : fact's NEG = belief srcs, belief's NEG = fact srcs (medium similarity)
                           (5x5 = 25 pairs)

DIRECT / PARA / PARAfire are NEG-INDEPENDENT (edit-vs-own-src/para) -> constant across A/B/C; only
NEGfloor / NEGxfire move. NEGfloor (worst-case similarity) is the cleanest cross-condition comparable
(denominator-independent). Everything else FROZEN (threshold 0.85, scaffold, pooling, keying.score gate).
Generic NEG-B probes are EXTERNAL generic questions (this task requests them explicitly) — not edit
samples, not written to samples.json, printed below for disclosure. Loads its own model copy; touches
NOTHING (no server/memory/codebook/editing/samples.json). Integrity gate: NEG-A must reproduce v1.5/v1.6.
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

# NEG-B: generic world-knowledge questions, UNRELATED to any JQ fact or any belief (no food/fantasy).
GENERIC_NEG = [
    "What is the capital of France?",
    "How many days are there in a week?",
    "Who wrote the play Romeo and Juliet?",
    "What is the boiling point of water in degrees Celsius?",
    "What is the largest planet in the solar system?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many continents are there on Earth?",
]

print("torch", torch.__version__, torch.cuda.is_available())

# --------------------------------------------------------------------------- #
# 1) Real samples (same 10 ids as v1.5/v1.6 S3 arm)
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

print("\n=== edit samples (real ids; same 10 as v1.5/v1.6) ===")
for it in FACTS + BELIEFS:
    print(f"  [{it['id']}] {it['edit']!r}  | src: {it['src']}")
print("\n=== NEG-B generic probes (external generic world knowledge; unrelated to any edit; DISCLOSED) ===")
for q in GENERIC_NEG:
    print(f"  · {q}")

# --------------------------------------------------------------------------- #
# 2) Own model copy + production adapter
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
print(f"\nadapter OK | normalize {adapter.normalize_codebook_keys} | qspan {adapter.query_span_pool_strategy} | thr {adapter.hopfield_key_match_threshold}")

# --------------------------------------------------------------------------- #
# 3) S3 chat key extraction (== compute_key templated=True), capture-once
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


for t in (FACTS[0]["src"], BELIEFS[0]["src"]):
    a = key_chat_span(t)
    b = compute_key(t, templated=True, hf_model=model.model, tok=tok, adapter=adapter)
    assert torch.allclose(a, b, atol=1e-3), ("drift", (a - b).abs().max().item())
print("drift-guard OK: key_chat_span == production compute_key(templated=True)\n")

# --------------------------------------------------------------------------- #
# 4) Keys
# --------------------------------------------------------------------------- #
def s3keys(items):
    return ([key_chat_span(it["edit"]) for it in items],
            [key_chat_span(it["src"]) for it in items],
            [key_chat_span(it["para"]) for it in items])


KE_F, KS_F, KP_F = s3keys(FACTS)
KE_B, KS_B, KP_B = s3keys(BELIEFS)
KGEN = [key_chat_span(q) for q in GENERIC_NEG]


# --------------------------------------------------------------------------- #
# 5) DIRECT/PARA (NEG-independent) + three NEG definitions
# --------------------------------------------------------------------------- #
def direct_para(kedit, ksrc, kpara):
    n = len(kedit)
    direct = [kscore(ksrc[i], kedit[i], adapter) for i in range(n)]
    para = [kscore(kpara[i], kedit[i], adapter) for i in range(n)]
    return {"DIRECT": sum(direct) / n, "PARA": sum(para) / n, "PARAfire": sum(v > THR for v in para)}


def neg_sibling(kedit, ksrc):   # NEG-A: read = same-group sibling src, j != i
    return [kscore(ksrc[j], kedit[i], adapter) for i in range(len(kedit)) for j in range(len(kedit)) if j != i]


def neg_generic(kedit, kgen):   # NEG-B: read = generic unrelated question
    return [kscore(kgen[g], kedit[i], adapter) for i in range(len(kedit)) for g in range(len(kgen))]


def neg_cross(kedit, ksrc_other):  # NEG-C: read = other-group src (all pairs, no self)
    return [kscore(ksrc_other[j], kedit[i], adapter) for i in range(len(kedit)) for j in range(len(ksrc_other))]


def summ(negv):
    return {"NEGfloor": max(negv), "NEGmean": sum(negv) / len(negv),
            "NEGxfire": sum(v > THR for v in negv), "NEGn": len(negv)}


DP_F = direct_para(KE_F, KS_F, KP_F)
DP_B = direct_para(KE_B, KS_B, KP_B)

NEG = {
    "A sibling":          {"fact": summ(neg_sibling(KE_F, KS_F)), "belief": summ(neg_sibling(KE_B, KS_B))},
    "B generic-unrelated": {"fact": summ(neg_generic(KE_F, KGEN)), "belief": summ(neg_generic(KE_B, KGEN))},
    "C cross-group":      {"fact": summ(neg_cross(KE_F, KS_B)),  "belief": summ(neg_cross(KE_B, KS_F))},
}
DP = {"fact": DP_F, "belief": DP_B}

# --------------------------------------------------------------------------- #
# 6) Integrity gate — NEG-A must reproduce v1.5/v1.6
# --------------------------------------------------------------------------- #
af, ab = NEG["A sibling"]["fact"], NEG["A sibling"]["belief"]
print(f"integrity: NEG-A fact NEGxfire={af['NEGxfire']}/20 (v1.5/6=11) ; belief={ab['NEGxfire']}/20 (=7)")
assert af["NEGxfire"] == 11 and ab["NEGxfire"] == 7, "NEG-A does not reproduce v1.5/v1.6 -> harness drift, STOP"
assert DP_F["PARAfire"] == 5 and DP_B["PARAfire"] == 5, "PARAfire drifted"
print("integrity OK: NEG-A reproduces v1.5/v1.6 -> the NEG definition is the ONLY changed variable\n")

# --------------------------------------------------------------------------- #
# 7) Report
# --------------------------------------------------------------------------- #
print(f"{'='*116}\nLOCALITY NEG-SELECTION ABLATION  (threshold={THR}; DIRECT/PARA/PARAfire are NEG-independent = constant)\n{'-'*116}")
print(f"  {'NEG-condition':22}{'group':8}{'DIRECT':>8}{'PARA':>8}{'NEGflr':>8}{'NEGmean':>9}{'PARAfire':>10}{'NEGxfire':>12}")
for name in ("A sibling", "B generic-unrelated", "C cross-group"):
    for grp in ("fact", "belief"):
        d, r = DP[grp], NEG[name][grp]
        print(f"  {name:22}{grp:8}{d['DIRECT']:8.3f}{d['PARA']:8.3f}{r['NEGfloor']:8.3f}{r['NEGmean']:9.3f}"
              f"{str(d['PARAfire'])+'/5':>10}{str(r['NEGxfire'])+'/'+str(r['NEGn']):>12}")
    print()

# per-edit worst generic similarity (show NEG-B stays low / or not)
print("-- NEG-B detail: worst generic-Q similarity per edit (fire if > 0.85) --")
for grp, kedit, items in (("fact", KE_F, FACTS), ("belief", KE_B, BELIEFS)):
    for i, it in enumerate(items):
        sims = [kscore(KGEN[g], kedit[i], adapter) for g in range(len(KGEN))]
        wg = max(range(len(sims)), key=lambda g: sims[g])
        print(f"   {grp:6} {it['id']}  worst-generic={sims[wg]:.3f}{'  <-XFIRE' if sims[wg] > THR else ''}  ({GENERIC_NEG[wg]!r})")
    print()

# --------------------------------------------------------------------------- #
# 8) Answers to the 4 questions
# --------------------------------------------------------------------------- #
def rate(r):
    return r["NEGxfire"] / r["NEGn"]


print(f"{'#'*116}\nANSWERS")
for grp in ("fact", "belief"):
    a, b, c = NEG["A sibling"][grp], NEG["B generic-unrelated"][grp], NEG["C cross-group"][grp]
    print(f"  [{grp}] NEGfloor  A-sibling={a['NEGfloor']:.3f}  C-cross={c['NEGfloor']:.3f}  B-generic={b['NEGfloor']:.3f}"
          f"   (thr {THR})")
    print(f"        NEGxfire  A={a['NEGxfire']}/{a['NEGn']} ({rate(a)*100:.0f}%)  "
          f"C={c['NEGxfire']}/{c['NEGn']} ({rate(c)*100:.0f}%)  B={b['NEGxfire']}/{b['NEGn']} ({rate(b)*100:.0f}%)")
bf, bb = NEG["B generic-unrelated"]["fact"], NEG["B generic-unrelated"]["belief"]
print(f"\n  Q1 NEG-B(generic) xfire: fact {bf['NEGxfire']}/{bf['NEGn']} , belief {bb['NEGxfire']}/{bb['NEGn']}  "
      f"(vs NEG-A fact {af['NEGxfire']}/20 , belief {ab['NEGxfire']}/20)")
b_clean = (rate(bf) < 0.05 and rate(bb) < 0.05)
a_high = (rate(af) > 0.2 or rate(ab) > 0.2)
if b_clean and a_high:
    q2 = ("NEG-B clean + NEG-A high => the locality failure is LARGELY a sibling-similarity ARTIFACT; "
          "the system does NOT pollute UNRELATED knowledge (generic floor < thr). Same-subject sibling "
          "cross-fire is a real but NARROWER problem, not catastrophic world-knowledge pollution.")
elif not b_clean:
    q2 = ("NEG-B also fires => the system DOES spuriously match UNRELATED knowledge (scaffold lifts even "
          "generic Qs past thr); the sibling NEG did NOT unfairly exaggerate it.")
else:
    q2 = "mixed — see per-group floors."
print(f"  Q2 {q2}")
print(f"  Q3 fact vs belief: NEG-A fact {af['NEGxfire']}/20 vs belief {ab['NEGxfire']}/20 ; "
      f"NEG-B fact {bf['NEGfloor']:.3f} vs belief {bb['NEGfloor']:.3f} floor ; "
      f"NEG-C fact {NEG['C cross-group']['fact']['NEGxfire']}/25 vs belief {NEG['C cross-group']['belief']['NEGxfire']}/25")
print("  Q4 (one-line): see report markdown.")

print(f"\nVRAM allocated={torch.cuda.memory_allocated()/1e9:.1f}G  max={torch.cuda.max_memory_allocated()/1e9:.1f}G")
print("torch", torch.__version__, torch.cuda.is_available())
