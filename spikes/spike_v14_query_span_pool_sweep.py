"""SPIKE v1.4 — query-span pooling sweep (English, real samples; MEASUREMENT-ONLY).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v14_query_span_pool_sweep.py

Sweeps the NEW chat-path query-span pooling knob (editor.pool_span_rows, the production
function) over {flat, last_40_perc, last_60_perc, last_80_perc, last} and asks: does weighting
the LATER tokens of the query span pull DIRECT/PARA apart from NEG (i.e. is intra-span pooling
a lever for the collapsed ~0.88 key cone), or does the cone persist (=> pooling is NOT the lever)?

For 5 real Type-A/Y/zero_prior samples (A0001-A0005, by id from eval/samples.json), 3 KEY-FORMAT
arms (cloze=edit_prompt / stmt=rag_doc / ques=queries[0]) and probes DIRECT=q0 / PARA=q1 /
JQ-NEG=other JQ-facts / generic-NEG=non-JQ questions. Score = production deferral cosine
(normalize(pool(H29[span])) vs normalize(pool(H29[key span]))), threshold 0.85. Each text is
forwarded ONCE; the 5 poolings reuse the captured rows. Loads its own model copy; touches
nothing (no server / memory / codebook).
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
from keying import _hero_render, query_span_in_rendered  # noqa: E402
from src.models.horen.editor import pool_span_rows         # noqa: E402  (the production knob)

MODEL = "/workspace/hugging_cache/llama3.1-8b-instruct"
DEV, THR = "cuda:0", 0.85
STRATS = ["flat", "last_40_perc", "last_60_perc", "last_80_perc", "last"]
GENERIC = [
    "What is the weather like today?",
    "What is the capital of France?",
    "How do I boil an egg?",
    "What time does the sun set?",
]
print("torch", torch.__version__, torch.cuda.is_available())

samples = json.load(open(os.path.join(_REPO, "eval", "samples.json")))["samples"]
AY0 = sorted([s for s in samples if s.get("sample_type") == "A" and s.get("type") == "Y"
              and s.get("prior_hardness") == "zero_prior"], key=lambda s: s["id"])
chosen, seen = [], set()
for s in AY0:
    if s["category"] not in seen and len(s.get("queries", [])) >= 2:
        seen.add(s["category"]); chosen.append(s)
    if len(chosen) == 5:
        break
print("chosen:", [(s["id"], s["category"]) for s in chosen])

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
down = model.model.layers[29].mlp.down_proj
cap = {}
down.register_forward_pre_hook(lambda _m, a: cap.__setitem__("x", a[0]))

_CACHE = {}  # text -> (rows[1,seq,D] float, (start,end))
def capture(text):
    if text in _CACHE:
        return _CACHE[text]
    rendered = _hero_render(tok, text)
    enc = tok(rendered, return_tensors="pt").to(DEV)
    s, e = query_span_in_rendered(tok, rendered, text)
    with torch.no_grad():
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    _CACHE[text] = (cap["x"][:1].float(), (s, e))  # [1, seq, D]
    return _CACHE[text]

def vec(text, strat):
    rows, (s, e) = capture(text)
    return pool_span_rows(rows, s, e, strat)[0]  # [D]

def cos(a, b):
    return (F.normalize(a, dim=-1) @ F.normalize(b, dim=-1)).item()

# pre-capture every text (one forward each)
for s in chosen:
    capture(s["edit_prompt"]); capture(s["rag_doc"])
    capture(s["queries"][0]["q"]); capture(s["queries"][1]["q"])
for g in GENERIC:
    capture(g)

ARMS = [("cloze(prod)", "edit_prompt"), ("stmt", "rag_doc"), ("ques", lambda s: s["queries"][0]["q"])]
def keytext(s, fld):
    return fld(s) if callable(fld) else s[fld]

def summarize(strat, fld):
    keys = {s["id"]: vec(keytext(s, fld), strat) for s in chosen}
    q0 = {s["id"]: vec(s["queries"][0]["q"], strat) for s in chosen}
    q1 = {s["id"]: vec(s["queries"][1]["q"], strat) for s in chosen}
    gen = [vec(g, strat) for g in GENERIC]
    directs = [cos(q0[s["id"]], keys[s["id"]]) for s in chosen]
    paras = [cos(q1[s["id"]], keys[s["id"]]) for s in chosen]
    jqneg, gneg = [], []
    for s in chosen:
        k = keys[s["id"]]
        for s2 in chosen:
            if s2["id"] != s["id"]:
                jqneg += [cos(q0[s2["id"]], k), cos(q1[s2["id"]], k)]
        gneg.append(max(cos(g, k) for g in gen))
    dmean = sum(directs) / len(directs)
    pmean = sum(paras) / len(paras)
    return {
        "DIRECT": dmean, "PARA": pmean,
        "PARA_fire": sum(v > THR for v in paras),
        "JQNEG_floor": max(jqneg), "JQNEG_xfire": sum(v > THR for v in jqneg), "JQNEG_n": len(jqneg),
        "GENNEG_floor": max(gneg),
        "margin": pmean - max(jqneg),  # PARA mean - worst JQ-fact NEG = separability
    }

for albl, fld in ARMS:
    print(f"\n{'='*104}\nARM: {albl}   (key text = {fld if isinstance(fld,str) else 'queries[0]'})  threshold={THR}\n{'-'*104}")
    print(f"  {'strategy':14}{'DIRECT':>9}{'PARA':>9}{'PARAfire':>9}{'JQNEG flr':>11}{'JQNEGxfire':>12}{'genNEG flr':>12}{'PARA-NEG':>10}")
    for strat in STRATS:
        r = summarize(strat, fld)
        pf = f"{r['PARA_fire']}/5"
        xf = f"{r['JQNEG_xfire']}/{r['JQNEG_n']}"
        print(f"  {strat:14}{r['DIRECT']:9.3f}{r['PARA']:9.3f}{pf:>9}"
              f"{r['JQNEG_floor']:11.3f}{xf:>12}{r['GENNEG_floor']:12.3f}{r['margin']:+10.3f}")

print(f"\n{'#'*104}")
print("READ: margin (PARA-NEG) = PARA mean - worst JQ-fact NEG. >0 & growing => intra-span pooling")
print("separates the cone (pooling IS a lever). ~<=0 & flat across strategies => cone persists,")
print("pooling is NOT the lever (representation/threshold is). Headline arm = cloze(prod).")
print(f"VRAM allocated={torch.cuda.memory_allocated()/1e9:.1f}G  max={torch.cuda.max_memory_allocated()/1e9:.1f}G")
print("torch", torch.__version__, torch.cuda.is_available())
