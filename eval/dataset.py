"""Typed loader + selectors for the Engram evaluation set (``samples.json``).

Pure Python — no GPU, no LLM, no network. Parses the three sample families
defined in ``eval/SCHEMA.md`` into frozen dataclasses and exposes a handful of
selectors used by the metrics / harness layers.

Families
--------
- **A = atomic_fact**  : one short fact about JQ + >=2 paraphrase queries. A is
  the shared knowledge pool; every ``key`` is unique across the 370 A samples.
- **B = user_bundle**  : one user (JQ) + m facts (m in {5,8,11,15}); ``facts``,
  ``gold_fact_set`` and ``rag_docs`` are 1:1 aligned.
- **C = list_filter**  : one user_fact + a ~15-item list with a single gold
  answer satisfying ``domain_filter AND user_filter``.

The on-disk JSON carries a few fields beyond the minimal schema (e.g. each
``gold_fact_set`` entry has an extra ``key``); the parsers tolerate those.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

DEFAULT_PATH = Path(__file__).with_name("samples.json")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Query:
    q: str
    a: str


@dataclass(frozen=True)
class ASample:
    id: str
    type: str
    prior_hardness: str
    category: str
    subject: Optional[str]
    edit_prompt: str
    target_new: str
    rag_doc: str
    queries: list[Query]
    key: str


@dataclass(frozen=True)
class BFact:
    key: str
    type: str
    category: str
    prior_hardness: str
    edit_prompt: str
    target_new: str


@dataclass(frozen=True)
class GoldEntry:
    fact: str
    match_any: list[str]
    key: Optional[str] = None


@dataclass(frozen=True)
class BSample:
    id: str
    user: str
    facts: list[BFact]
    generation_prompt: str
    gold_fact_set: list[GoldEntry]
    rag_docs: list[str]


@dataclass(frozen=True)
class CUserFact:
    type: str
    category: str
    prior_hardness: str
    statement: str
    edit_prompt: str
    target_new: str
    key: str


@dataclass(frozen=True)
class Filter:
    attribute: str
    op: str
    value: object = None


@dataclass(frozen=True)
class CListItem:
    name: str
    attributes: dict
    blurb: str


@dataclass(frozen=True)
class CSample:
    id: str
    user_fact: CUserFact
    list_domain: str
    domain_filter: Filter
    user_filter: Filter
    list_items: list[CListItem]
    gold_answer: str
    question: str
    rag_doc: str
    difficulty: str


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def _parse_a(raw: dict) -> ASample:
    return ASample(
        id=raw["id"],
        type=raw["type"],
        prior_hardness=raw["prior_hardness"],
        category=raw["category"],
        subject=raw.get("subject"),
        edit_prompt=raw["edit_prompt"],
        target_new=raw["target_new"],
        rag_doc=raw["rag_doc"],
        queries=[Query(q=q["q"], a=q["a"]) for q in raw["queries"]],
        key=raw["key"],
    )


def _parse_b(raw: dict) -> BSample:
    facts = [
        BFact(
            key=f["key"],
            type=f["type"],
            category=f["category"],
            prior_hardness=f["prior_hardness"],
            edit_prompt=f["edit_prompt"],
            target_new=f["target_new"],
        )
        for f in raw["facts"]
    ]
    gold_fact_set = [
        GoldEntry(fact=g["fact"], match_any=list(g["match_any"]), key=g.get("key"))
        for g in raw["gold_fact_set"]
    ]
    rag_docs = list(raw["rag_docs"])
    if not (len(facts) == len(gold_fact_set) == len(rag_docs)):
        raise ValueError(
            f"B {raw['id']} misaligned: "
            f"facts={len(facts)} gold_fact_set={len(gold_fact_set)} rag_docs={len(rag_docs)}"
        )
    return BSample(
        id=raw["id"],
        user=raw["user"],
        facts=facts,
        generation_prompt=raw["generation_prompt"],
        gold_fact_set=gold_fact_set,
        rag_docs=rag_docs,
    )


def _parse_filter(raw: dict) -> Filter:
    return Filter(attribute=raw["attribute"], op=raw["op"], value=raw.get("value"))


def _parse_c(raw: dict) -> CSample:
    uf = raw["user_fact"]
    user_fact = CUserFact(
        type=uf["type"],
        category=uf["category"],
        prior_hardness=uf["prior_hardness"],
        statement=uf["statement"],
        edit_prompt=uf["edit_prompt"],
        target_new=uf["target_new"],
        key=uf["key"],
    )
    list_items = [
        CListItem(name=it["name"], attributes=dict(it["attributes"]), blurb=it["blurb"])
        for it in raw["list_items"]
    ]
    return CSample(
        id=raw["id"],
        user_fact=user_fact,
        list_domain=raw["list_domain"],
        domain_filter=_parse_filter(raw["domain_filter"]),
        user_filter=_parse_filter(raw["user_filter"]),
        list_items=list_items,
        gold_answer=raw["gold_answer"],
        question=raw["question"],
        rag_doc=raw["rag_doc"],
        difficulty=raw["difficulty"],
    )


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load(path: Union[str, Path] = DEFAULT_PATH) -> dict:
    """Load ``samples.json`` and bucket parsed samples by family.

    Returns ``{"A": [ASample], "B": [BSample], "C": [CSample]}``. Raises
    ``ValueError`` if the per-family counts deviate from the frozen 370/60/70.
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    A: list[ASample] = []
    B: list[BSample] = []
    C: list[CSample] = []
    for raw in doc["samples"]:
        t = raw["sample_type"]
        if t == "A":
            A.append(_parse_a(raw))
        elif t == "B":
            B.append(_parse_b(raw))
        elif t == "C":
            C.append(_parse_c(raw))
        else:
            raise ValueError(f"unknown sample_type {t!r} in {raw.get('id')!r}")

    if not (len(A) == 370 and len(B) == 60 and len(C) == 70):
        raise ValueError(
            f"count mismatch: A={len(A)} (want 370), "
            f"B={len(B)} (want 60), C={len(C)} (want 70)"
        )
    return {"A": A, "B": B, "C": C}


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def pool_by_key(samples: Iterable[ASample]) -> dict:
    """Index the A knowledge pool by ``key`` -> ASample."""
    return {s.key: s for s in samples}


def by_tier(samples: Iterable, tiers: Union[str, Iterable[str]]) -> list:
    """Filter samples whose ``prior_hardness`` is in ``tiers`` (str normalized to a set)."""
    wanted = {tiers} if isinstance(tiers, str) else set(tiers)
    return [s for s in samples if s.prior_hardness in wanted]


def by_type(samples: Iterable, t: str) -> list:
    """Filter samples whose ``type`` == ``t`` (X/Y)."""
    return [s for s in samples if s.type == t]


def zero_prior_Y(samples: Iterable[ASample]) -> list:
    """The zero_prior, Type Y subset of A (the personal-fact backbone)."""
    return by_type(by_tier(samples, "zero_prior"), "Y")


def stem_of(edit_prompt: str, blank: str = "___") -> str:
    """Return the text before the first ``blank`` marker, right-stripped.

    e.g. ``"JQ is allergic to ___"`` -> ``"JQ is allergic to"``.
    """
    return edit_prompt.split(blank, 1)[0].rstrip()
