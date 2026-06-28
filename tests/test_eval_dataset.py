"""Unit tests for ``eval.dataset`` — pure CPU, no GPU/model/network.

Loads the real ``eval/samples.json`` and asserts the schema invariants and
selector semantics documented in ``eval/SCHEMA.md`` (and the verified ground
truth: A=370/B=60/C=70, A tier/type grid, B m-buckets, C filter shape).
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval import dataset  # noqa: E402
from eval.dataset import (  # noqa: E402
    ASample,
    BSample,
    CSample,
    by_tier,
    by_type,
    load,
    pool_by_key,
    stem_of,
    zero_prior_Y,
)


@pytest.fixture(scope="module")
def data():
    return load()


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #
def test_counts(data):
    assert len(data["A"]) == 370
    assert len(data["B"]) == 60
    assert len(data["C"]) == 70
    assert all(isinstance(s, ASample) for s in data["A"])
    assert all(isinstance(s, BSample) for s in data["B"])
    assert all(isinstance(s, CSample) for s in data["C"])


def test_count_mismatch_raises(tmp_path):
    src = json.loads(dataset.DEFAULT_PATH.read_text(encoding="utf-8"))
    # Drop one A sample -> A count becomes 369 -> ValueError.
    a_ids = [s for s in src["samples"] if s["sample_type"] == "A"]
    src["samples"].remove(a_ids[0])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(src), encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)


# --------------------------------------------------------------------------- #
# A invariants
# --------------------------------------------------------------------------- #
def test_a_field_invariants(data):
    for s in data["A"]:
        assert s.type in {"X", "Y"}
        if s.type == "Y":
            assert s.subject == "JQ"
        else:
            assert s.subject is None
        assert "___" in s.edit_prompt
        assert len(s.queries) == 2
        for q in s.queries:
            assert q.a == s.target_new
        assert "JQ" in s.rag_doc


# --------------------------------------------------------------------------- #
# B invariants
# --------------------------------------------------------------------------- #
def test_b_alignment(data):
    for s in data["B"]:
        assert len(s.facts) == len(s.gold_fact_set) == len(s.rag_docs)
        for g in s.gold_fact_set:
            assert len(g.match_any) > 0
            assert g.key  # the on-disk extra "key" must be populated


def test_b_misalignment_raises(tmp_path):
    src = json.loads(dataset.DEFAULT_PATH.read_text(encoding="utf-8"))
    for s in src["samples"]:
        if s["sample_type"] == "B":
            s["rag_docs"] = s["rag_docs"][:-1]  # break 1:1 alignment
            break
    bad = tmp_path / "bad_b.json"
    bad.write_text(json.dumps(src), encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)


# --------------------------------------------------------------------------- #
# C invariants
# --------------------------------------------------------------------------- #
def test_c_fields(data):
    for s in data["C"]:
        assert s.domain_filter.op in {"is_true", "is_false"}
        assert s.user_filter.op in {"is_true", "is_false"}
        assert len(s.list_items) >= 14
        da = s.domain_filter.attribute
        ua = s.user_filter.attribute
        names = [it.name for it in s.list_items]
        for it in s.list_items:
            assert isinstance(it.attributes[da], bool)
            assert isinstance(it.attributes[ua], bool)
        assert s.gold_answer in names
        assert s.user_fact.key


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def test_pool_by_key_totality(data):
    pool = pool_by_key(data["A"])
    assert len(pool) == 370
    for s in data["B"]:
        for f in s.facts:
            assert f.key in pool
    for s in data["C"]:
        assert s.user_fact.key in pool


def test_by_tier_zero_prior(data):
    assert len(by_tier(data["A"], "zero_prior")) == 142
    # Iterable form must behave identically to the str form.
    assert len(by_tier(data["A"], {"zero_prior"})) == 142


def test_by_type_counts(data):
    assert len(by_type(data["A"], "X")) == 222
    assert len(by_type(data["A"], "Y")) == 148


def test_zero_prior_Y(data):
    assert len(zero_prior_Y(data["A"])) == 138


def test_b_m_buckets(data):
    buckets = Counter(len(s.facts) for s in data["B"])
    assert buckets == {5: 15, 8: 15, 11: 15, 15: 15}


def test_stem_of():
    assert stem_of("JQ is allergic to ___") == "JQ is allergic to"
    mid = stem_of("Remote work is ___ than office work")
    assert mid == "Remote work is"
    assert "___" not in mid
