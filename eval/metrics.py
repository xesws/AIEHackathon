"""Evaluation scorers for the Engram capability matrix.

Three scorers (A / B / C) plus a prompt-token counter. All string matching goes
through one shared normalizer; only genuinely borderline cases escalate to an LLM
judge so the common path is deterministic and free.

Invariants:
  * INV-E3 — the judge is Qwen via ``memory.llm.complete`` (temperature 0.0, JSON
    mode). It is deliberately a DIFFERENT model from the Llama-under-test, so the
    grader can never just agree with whatever the editee produced.
  * INV-E6 — B-recall scores against each fact's ``match_any`` synonym set, NEVER
    ``target_new``. 37 / 585 gold entries carry a ``match_any`` whose accepted
    surface form does not literally contain ``target_new``; matching on
    ``target_new`` would silently miss those.

The judge is reached lazily through ``llm.complete(...)`` (module attribute) so a
test can ``monkeypatch.setattr("memory.llm.complete", fake)`` and intercept it.
No torch / tokenizer import lives here — the token counter is injected.
"""
from __future__ import annotations

import json
import re  # noqa: F401  (kept available for ad-hoc normalization tweaks)
import string
from typing import Callable, Optional

from memory import llm


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _normalize(text) -> str:
    """Lowercase, strip punctuation -> spaces, drop articles, collapse whitespace."""
    tokens = str(text).lower().translate(_PUNCT_TABLE).split()
    return " ".join(t for t in tokens if t not in _ARTICLES)


def _borderline(norm_pred: str, norm_target: str) -> bool:
    """True when a normalized target is a *partial* (token-overlap) match only.

    A clean miss (no shared >=3-char token) and a clean hit (target is a substring
    of the prediction) are both decided without an LLM; everything in between is
    "borderline" and worth a judge call.
    """
    if not norm_target or norm_target in norm_pred:
        return False
    return any(len(t) >= 3 and t in norm_pred for t in norm_target.split())


def _get(obj, name):
    """Read ``name`` from a dict or a dataclass/object (duck typing for gold rows)."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
def score_A(pred, target_new, *, judge: Optional[Callable] = None) -> bool:
    """A-task efficacy: did ``pred`` assert the edited target ``target_new``?

    Exact/substring -> True deterministically; clean miss -> False; only a
    token-overlap borderline escalates to the (Qwen) judge.
    """
    np_, nt = _normalize(pred), _normalize(target_new)
    if not nt:
        return False
    if nt in np_:
        return True
    if _borderline(np_, nt):
        return bool((judge or _judge_A)(pred, target_new))
    return False


def score_B(generation, gold_fact_set, *, judge: Optional[Callable] = None) -> dict:
    """B-task recall: how many gold facts surface in a free-form ``generation``.

    Each gold entry is matched against its ``match_any`` synonym set (INV-E6), not
    its ``target_new``. Returns ``{"recall": float, "hits": [key, ...]}``.
    """
    ng = _normalize(generation)
    hits = []
    for entry in gold_fact_set:
        terms = _get(entry, "match_any")
        hit = any(_normalize(t) in ng for t in terms)
        if not hit and any(_borderline(ng, _normalize(t)) for t in terms):
            hit = bool((judge or _judge_B)(generation, entry))
        if hit:
            hits.append(_get(entry, "key"))
    recall = len(hits) / len(gold_fact_set) if gold_fact_set else 0.0
    return {"recall": recall, "hits": hits}


def score_C(model_output, gold_answer, list_items, *, judge: Optional[Callable] = None) -> bool:
    """C-task selection: did the model pick the right item from ``list_items``?

    A judge/extractor maps the free-form ``model_output`` to one of the item names;
    the pick counts only if it normalizes equal to ``gold_answer``.
    """
    names = [_get(it, "name") for it in list_items]
    chosen = (judge or _extract_choice_C)(model_output, names)
    return bool(chosen) and _normalize(chosen) == _normalize(gold_answer)


def count_prompt_tokens(text, *, count_tokens: Callable) -> int:
    """Prompt-token cost via an injected counter (no tiktoken / tokenizer here)."""
    return count_tokens(text)


# --------------------------------------------------------------------------- #
# LLM judges (Qwen via memory.llm.complete, temperature 0, JSON mode)
# --------------------------------------------------------------------------- #
JUDGE_A_SYSTEM = (
    "You grade a knowledge-edit. Given a model PREDICTION and the intended TARGET, "
    "decide whether the prediction asserts the target fact (allow paraphrase, "
    "synonyms, and extra words). Reply with strict JSON: {\"match\": true|false}."
)

JUDGE_B_SYSTEM = (
    "You grade fact recall. Given a GENERATION and a list of acceptable surface "
    "forms (MATCH_ANY) for one fact, decide whether the generation states that "
    "fact (allow paraphrase/synonyms). Reply with strict JSON: {\"match\": true|false}."
)

JUDGE_C_SYSTEM = (
    "You extract a single choice. Given the candidate OPTIONS and a model ANSWER, "
    "return the one option the answer selects, copied verbatim, or null if none. "
    "Reply with strict JSON: {\"choice\": <option string or null>}."
)


def _judge_A(pred, target_new) -> bool:
    """Qwen judge for score_A; any failure is treated as a non-match."""
    try:
        messages = [
            {"role": "system", "content": JUDGE_A_SYSTEM},
            {"role": "user", "content": json.dumps({"prediction": pred, "target": target_new})},
        ]
        raw = llm.complete(messages, temperature=0.0, response_format={"type": "json_object"})
        return bool(json.loads(raw).get("match"))
    except Exception:
        return False


def _judge_B(generation, entry) -> bool:
    """Qwen judge for score_B; passes the entry's ``match_any`` terms (INV-E6)."""
    try:
        terms = list(_get(entry, "match_any"))
        messages = [
            {"role": "system", "content": JUDGE_B_SYSTEM},
            {"role": "user", "content": json.dumps({"match_any": terms, "generation": generation})},
        ]
        raw = llm.complete(messages, temperature=0.0, response_format={"type": "json_object"})
        return bool(json.loads(raw).get("match"))
    except Exception:
        return False


def _extract_choice_C(model_output, names) -> Optional[str]:
    """Qwen extractor for score_C; returns a validated option name or ``None``."""
    try:
        messages = [
            {"role": "system", "content": JUDGE_C_SYSTEM},
            {"role": "user", "content": json.dumps({"options": list(names), "answer": model_output})},
        ]
        raw = llm.complete(messages, temperature=0.0, response_format={"type": "json_object"})
        choice = json.loads(raw).get("choice")
        return choice if choice in names else None
    except Exception:
        return None
