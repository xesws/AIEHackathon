"""The ablation rungs / conditions, expressed over ``eval.runtime``.

Three families (see docs/v1.2-eval-harness.md):

* **A-ladder** (atomic facts) — P1 edit-only, P2 rag-only, P3 edit+rag. The
  adjacent gap ``max(P1,P2) - P3`` is the *query-split* loss. Efficacy is the
  QA rate over each fact's paraphrase ``queries`` (comparable across all three
  rungs); P1 additionally records the cloze (direct-probe) rate.

* **B-ladder** (user bundles) — P3_B (direct edit, no pipeline), P3.5
  (buffer -> consolidate, real dedup, extractor bypassed), P4 (raw sentences ->
  real extractor -> consolidate). Scored by **targeted per-fact recall**
  (edit-only isolation): each fact is probed with its OWN cloze and counted via
  the bundle's ``gold_fact_set[i].match_any`` (INV-E6). Editing has real signal
  here (~0.93), so ``dedup = P3_B - P3.5`` and ``extractor = P3.5 - P4`` are
  attributable. The open-ended self-intro is scored too but ONLY as a free-gen
  diagnostic (edit-only vs edit+rag vs rag-only) — never for attribution,
  because HoReN edits do not fire on open-ended generation
  (see [[horen-edits-query-keyed]]).

* **C-condition** (list filter) — base / rag / edit; a Qwen extractor maps the
  free-form answer to one item name, compared to ``gold_answer``. The model sees
  only ``name`` + ``blurb`` (never ``attributes``).

Every rung resets at SETUP (``runtime.clean_all`` = swap edit out + clear
buffer/rag) and then installs knowledge; nothing resets between writing a fact
and querying it. Records are plain dicts for JSON serialization.
"""
from __future__ import annotations

from typing import Callable, Optional

from eval import metrics, runtime

FREEGEN_TOKENS = 200
C_TOKENS = 96


def _subj(type_: str) -> Optional[str]:
    return "JQ" if type_ == "Y" else None


def _fact_hit(answer: str, gold_entry, judge: Optional[Callable]) -> bool:
    """One fact's targeted hit: any ``match_any`` term present (INV-E6)."""
    return metrics.score_B(answer, [gold_entry], judge=judge)["recall"] == 1.0


# --------------------------------------------------------------------------- #
# A-ladder
# --------------------------------------------------------------------------- #
def a_p1(s, *, unrelated_q: str, judge: Optional[Callable] = None) -> dict:
    """P1 edit-only: edit one fact, probe cloze + paraphrases (no RAG); locality."""
    runtime.clean_all()
    res = runtime.do_edit(s.edit_prompt, s.target_new, s.subject)
    cb = runtime.codebook_size()
    cells = []
    mnt = runtime.mnt_for(s.target_new)

    cloze_ans = runtime.gen_answer(s.edit_prompt, with_rag=False, max_new_tokens=mnt)
    cloze_ok = metrics.score_A(cloze_ans, s.target_new, judge=judge)
    cells.append(_cell("P1", s.id, "efficacy_cloze", s.edit_prompt, cloze_ans, cloze_ok))

    for q in s.queries:
        a = runtime.gen_answer(q.q, with_rag=False, max_new_tokens=mnt)
        cells.append(_cell("P1", s.id, "qa", q.q, a, metrics.score_A(a, s.target_new, judge=judge)))

    ls = runtime.live_score(unrelated_q)
    loc_ans = runtime.gen_answer(unrelated_q, with_rag=False, max_new_tokens=mnt)
    leak = s.target_new.lower() in loc_ans.lower()
    cells.append(_cell("P1", s.id, "locality", unrelated_q, loc_ans,
                       not (ls >= runtime.threshold() or leak),
                       live_score=ls, fired=ls >= runtime.threshold(), leak=leak))
    return {"rung": "P1", "sample_id": s.id, "codebook_size": cb,
            "edit_seconds": res["edit_seconds"], "cells": cells}


def a_p2(s, *, judge: Optional[Callable] = None) -> dict:
    """P2 rag-only: no edit; rag_doc indexed; probe paraphrases with RAG on."""
    runtime.clean_all()
    runtime.add_rag(s.rag_doc)
    cells = []
    mnt = runtime.mnt_for(s.target_new)
    for q in s.queries:
        a = runtime.gen_answer(q.q, with_rag=True, max_new_tokens=mnt)
        cells.append(_cell("P2", s.id, "qa", q.q, a, metrics.score_A(a, s.target_new, judge=judge)))
    return {"rung": "P2", "sample_id": s.id, "cells": cells}


def a_p3(s, *, judge: Optional[Callable] = None) -> dict:
    """P3 edit+rag: edit + rag_doc; probe paraphrases with RAG on (query-split active)."""
    runtime.clean_all()
    runtime.do_edit(s.edit_prompt, s.target_new, s.subject)
    runtime.add_rag(s.rag_doc)
    cells = []
    mnt = runtime.mnt_for(s.target_new)
    for q in s.queries:
        a = runtime.gen_answer(q.q, with_rag=True, max_new_tokens=mnt)
        cells.append(_cell("P3", s.id, "qa", q.q, a, metrics.score_A(a, s.target_new, judge=judge)))
    return {"rung": "P3", "sample_id": s.id, "cells": cells}


# --------------------------------------------------------------------------- #
# B-ladder (targeted per-fact recall)
# --------------------------------------------------------------------------- #
def _targeted_recall(b, *, with_rag: bool, judge: Optional[Callable]) -> dict:
    """Probe each fact with its OWN cloze; hit via gold_fact_set[i].match_any."""
    hits, per_fact = 0, []
    for fact, gold in zip(b.facts, b.gold_fact_set):
        a = runtime.gen_answer(fact.edit_prompt, with_rag=with_rag,
                               max_new_tokens=runtime.mnt_for(fact.target_new))
        ok = _fact_hit(a, gold, judge)
        hits += ok
        per_fact.append({"key": fact.key, "target": fact.target_new, "answer": a[:80], "hit": ok})
    m = len(b.facts)
    return {"recall": hits / m if m else 0.0, "hits": hits, "m": m, "per_fact": per_fact}


def b_p3b(b, *, judge: Optional[Callable] = None) -> dict:
    """P3_B: direct edit of every fact (no buffer/dedup/extractor) + rag docs."""
    runtime.clean_all()
    for fact in b.facts:
        runtime.do_edit(fact.edit_prompt, fact.target_new, _subj(fact.type))
    for doc in b.rag_docs:
        runtime.add_rag(doc)
    cb = runtime.codebook_size()
    eo = _targeted_recall(b, with_rag=False, judge=judge)   # attribution metric
    er = _targeted_recall(b, with_rag=True, judge=judge)    # full-system reference
    return {"rung": "P3_B", "sample_id": b.id, "m": len(b.facts), "codebook_size": cb,
            "targeted_recall_edit_only": eo["recall"], "targeted_recall_edit_rag": er["recall"],
            "eo": eo, "er": er}


def b_p35(b, *, judge: Optional[Callable] = None) -> dict:
    """P3.5: pre-decomposed (stem,target) -> buffer -> run_pass (real dedup); extractor bypassed."""
    runtime.clean_all()
    for fact in b.facts:
        runtime.buffer_edit(fact.edit_prompt, fact.target_new, subject=_subj(fact.type) or "JQ")
    for doc in b.rag_docs:
        runtime.add_rag(doc)
    n_written = runtime.consolidate_now()
    cb = runtime.codebook_size()
    eo = _targeted_recall(b, with_rag=False, judge=judge)
    return {"rung": "P3.5", "sample_id": b.id, "m": len(b.facts), "n_written": n_written,
            "codebook_size": cb, "targeted_recall_edit_only": eo["recall"], "eo": eo}


def b_p4(b, *, judge: Optional[Callable] = None) -> dict:
    """P4: raw natural-language sentences -> real extractor -> buffer -> run_pass."""
    runtime.clean_all()
    ingest_stats = [runtime.ingest_sentence(doc) for doc in b.rag_docs]
    n_written = runtime.consolidate_now()
    cb = runtime.codebook_size()
    eo = _targeted_recall(b, with_rag=False, judge=judge)
    return {"rung": "P4", "sample_id": b.id, "m": len(b.facts), "n_written": n_written,
            "n_edit_buffered": sum(st["n_edit_buffered"] for st in ingest_stats),
            "n_rag_indexed": sum(st["n_rag_indexed"] for st in ingest_stats),
            "codebook_size": cb, "targeted_recall_edit_only": eo["recall"], "eo": eo}


def b_ragonly_targeted(b, *, judge: Optional[Callable] = None) -> dict:
    """RAG-only targeted recall (no edit) — the rag arm of the recall-vs-m curve."""
    runtime.clean_all()
    for doc in b.rag_docs:
        runtime.add_rag(doc)
    r = _targeted_recall(b, with_rag=True, judge=judge)
    return {"rung": "rag_only", "sample_id": b.id, "m": len(b.facts),
            "targeted_recall": r["recall"], "r": r}


def b_freegen_diagnostic(b, *, judge: Optional[Callable] = None) -> dict:
    """Diagnostic ONLY (not attribution): open-ended self-intro recall for
    edit-only vs edit+rag vs rag-only. Shows HoReN edits don't fire on free gen."""
    runtime.clean_all()
    for fact in b.facts:
        runtime.do_edit(fact.edit_prompt, fact.target_new, _subj(fact.type))
    for doc in b.rag_docs:
        runtime.add_rag(doc)
    g_eo = runtime.gen_answer(b.generation_prompt, with_rag=False, max_new_tokens=FREEGEN_TOKENS)
    g_er = runtime.gen_answer(b.generation_prompt, with_rag=True, max_new_tokens=FREEGEN_TOKENS)
    r_eo = metrics.score_B(g_eo, b.gold_fact_set, judge=judge)["recall"]
    r_er = metrics.score_B(g_er, b.gold_fact_set, judge=judge)["recall"]

    runtime.clean_all()
    for doc in b.rag_docs:
        runtime.add_rag(doc)
    g_ro = runtime.gen_answer(b.generation_prompt, with_rag=True, max_new_tokens=FREEGEN_TOKENS)
    r_ro = metrics.score_B(g_ro, b.gold_fact_set, judge=judge)["recall"]
    return {"sample_id": b.id, "m": len(b.facts),
            "freegen_recall_edit_only": r_eo, "freegen_recall_edit_rag": r_er,
            "freegen_recall_rag_only": r_ro,
            "samples": {"edit_only": g_eo[:200], "edit_rag": g_er[:200], "rag_only": g_ro[:200]}}


# --------------------------------------------------------------------------- #
# C-condition
# --------------------------------------------------------------------------- #
def render_c_prompt(c) -> str:
    """The user-turn the model sees: question + the items as name+blurb (NO attributes)."""
    lines = [c.question, "", "Options:"]
    for it in c.list_items:
        lines.append(f"- {it.name}: {it.blurb}")
    lines.append("")
    lines.append("Answer with exactly one option name.")
    return "\n".join(lines)


def c_condition(c, cond: str, *, judge: Optional[Callable] = None) -> dict:
    """One C sample under cond in {base, rag, edit}. score_C uses a Qwen extractor."""
    runtime.clean_all()
    if cond == "rag":
        runtime.add_rag(c.rag_doc)
    elif cond == "edit":
        runtime.do_edit(c.user_fact.edit_prompt, c.user_fact.target_new, _subj(c.user_fact.type))
    prompt = render_c_prompt(c)
    ans = runtime.gen_answer(prompt, with_rag=(cond == "rag"), max_new_tokens=C_TOKENS)
    correct = metrics.score_C(ans, c.gold_answer, c.list_items, judge=judge)
    return {"condition": cond, "sample_id": c.id, "gold": c.gold_answer,
            "answer": ans[:200], "correct": correct}


# --------------------------------------------------------------------------- #
def _cell(rung, sample_id, kind, query, pred, correct, **extra) -> dict:
    cell = {"rung": rung, "sample_id": sample_id, "kind": kind,
            "query": query, "pred": pred[:160], "correct": correct}
    cell.update(extra)
    return cell
