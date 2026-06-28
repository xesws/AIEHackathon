"""Unit suite for ``eval.metrics`` (A/B/C scorers + token counter).

Pure CPU, no GPU / model / network. The only external seam is the LLM judge
(``memory.llm.complete``), mirrored from ``test_dedup.py``:
  * ``install_complete`` — canned JSON (dict -> json.dumps), raw string, or raising;
    optionally records each call's kwargs so temperature / JSON-mode can be asserted.
  * ``install_complete_must_not_call`` — fails the test if the judge is ever reached
    (proves the deterministic fast paths never hit the LLM).

Key invariants exercised: judge runs at temperature 0.0 in JSON mode (INV-E3), and
B-recall matches on ``match_any`` synonym sets, never ``target_new`` (INV-E6).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

# Bootstrap sys.path to the repo root so ``eval`` / ``memory`` import when pytest
# is launched from anywhere (mirrors how test_dedup.py relies on root-on-path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics  # noqa: E402
from memory import llm  # noqa: E402


# --------------------------------------------------------------------------- #
# Seam helpers (copied/adapted from test_dedup.py)
# --------------------------------------------------------------------------- #
def install_complete(monkeypatch, payload, *, record: list | None = None):
    """Monkeypatch ``llm.complete``.

    ``payload`` may be a dict (serialized to JSON), a raw string, or an Exception
    instance/class to raise. When ``record`` is given, each call's kwargs (messages,
    model, temperature, response_format) are appended for assertion.
    """

    def fake_complete(messages, *, model=None, temperature=0.0, response_format=None):
        if record is not None:
            record.append(
                {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "response_format": response_format,
                }
            )
        if isinstance(payload, BaseException) or (
            isinstance(payload, type) and issubclass(payload, BaseException)
        ):
            raise payload if isinstance(payload, BaseException) else payload("boom")
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    monkeypatch.setattr(llm, "complete", fake_complete)


def install_complete_must_not_call(monkeypatch):
    """Install an ``llm.complete`` that fails the test if it is ever invoked."""

    def fake_complete(*a, **k):
        raise AssertionError("llm.complete should not be called for this case")

    monkeypatch.setattr(llm, "complete", fake_complete)


# --------------------------------------------------------------------------- #
# _normalize
# --------------------------------------------------------------------------- #
def test_normalize_strips_punct_articles_and_case():
    assert metrics._normalize("The  Peanuts!") == "peanuts"


def test_normalize_hyphen_becomes_space():
    assert metrics._normalize("Carnegie-Mellon") == "carnegie mellon"


# --------------------------------------------------------------------------- #
# score_A — deterministic paths (LLM must NOT be called)
# --------------------------------------------------------------------------- #
def test_score_a_exact_match(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("peanuts", "peanuts") is True


def test_score_a_case_insensitive(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("PEANUTS", "peanuts") is True


def test_score_a_punctuation_ignored(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("...peanuts.", "peanuts") is True
    assert metrics.score_A("Carnegie-Mellon", "Carnegie Mellon") is True


def test_score_a_articles_ignored(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("France", "the France") is True


def test_score_a_substring_in_sentence(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("I really love peanuts a lot", "peanuts") is True


def test_score_a_clear_miss_no_llm(monkeypatch):
    install_complete_must_not_call(monkeypatch)  # no shared token -> decided locally
    assert metrics.score_A("I enjoy apples", "peanuts") is False


def test_score_a_empty_target_is_false(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_A("anything", "") is False


# --------------------------------------------------------------------------- #
# score_A — borderline escalates to the (Qwen) judge
# --------------------------------------------------------------------------- #
def test_score_a_borderline_judge_match(monkeypatch):
    calls: list = []
    install_complete(monkeypatch, {"match": True}, record=calls)

    assert metrics.score_A("she studied at Carnegie Mellon", "Carnegie Mellon University") is True
    # judge consulted exactly once, at temperature 0 in JSON mode (INV-E3).
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_score_a_borderline_judge_no_match(monkeypatch):
    install_complete(monkeypatch, {"match": False})
    assert metrics.score_A("she studied at Carnegie Mellon", "Carnegie Mellon University") is False


def test_score_a_injected_judge_bypasses_llm(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert (
        metrics.score_A(
            "she studied at Carnegie Mellon",
            "Carnegie Mellon University",
            judge=lambda p, t: True,
        )
        is True
    )


# --------------------------------------------------------------------------- #
# score_B — recall over match_any synonym sets (INV-E6)
# --------------------------------------------------------------------------- #
def test_score_b_full_recall_no_llm(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    gold = [
        {"key": "k1", "match_any": ["peanuts"]},
        {"key": "k2", "match_any": ["France"]},
    ]
    out = metrics.score_B("I love peanuts and France equally", gold)
    assert out["recall"] == 1.0
    assert out["hits"] == ["k1", "k2"]


def test_score_b_partial_recall_no_llm(monkeypatch):
    install_complete_must_not_call(monkeypatch)  # missing term shares no token
    gold = [
        {"key": "k1", "match_any": ["peanuts"]},
        {"key": "k2", "match_any": ["France"]},
        {"key": "k3", "match_any": ["zebra"]},
    ]
    out = metrics.score_B("I love peanuts and France", gold)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["hits"] == ["k1", "k2"]


def test_score_b_inv_e6_synonym_not_target_new(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    # Gold accepts the surface form "snail noodle"; target_new ("escargot ramen")
    # never appears. Matching on target_new would MISS this — match_any catches it.
    gold = [{"key": "dish", "match_any": ["snail noodle"], "target_new": "escargot ramen"}]
    gen = "For dinner JQ ordered a bowl of snail noodle soup."
    assert "escargot ramen" not in gen.lower()
    out = metrics.score_B(gen, gold)
    assert out["recall"] == 1.0
    assert out["hits"] == ["dish"]


def test_score_b_any_of_match_any(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    gold = [{"key": "school", "match_any": ["CMU", "Carnegie Mellon"]}]
    out = metrics.score_B("She got her degree from CMU.", gold)
    assert out["recall"] == 1.0
    assert out["hits"] == ["school"]


def test_score_b_empty_gold_is_zero(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    out = metrics.score_B("anything at all", [])
    assert out["recall"] == 0.0
    assert out["hits"] == []


def test_score_b_entries_as_objects_duck_typing(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    from types import SimpleNamespace

    gold = [
        SimpleNamespace(key="k1", match_any=["peanuts"]),
        SimpleNamespace(key="k2", match_any=["zebra"]),
    ]
    out = metrics.score_B("I love peanuts", gold)
    assert out["recall"] == 0.5
    assert out["hits"] == ["k1"]


# --------------------------------------------------------------------------- #
# score_C — selection via injected extractor / default judge
# --------------------------------------------------------------------------- #
_C_ITEMS = [
    {"name": "Saffron Thai", "blurb": "..."},
    {"name": "Peanut Wok", "blurb": "..."},
]


def test_score_c_injected_extractor_right(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert (
        metrics.score_C("...", "Saffron Thai", _C_ITEMS, judge=lambda o, n: "Saffron Thai")
        is True
    )


def test_score_c_injected_extractor_wrong(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert (
        metrics.score_C("...", "Saffron Thai", _C_ITEMS, judge=lambda o, n: "Peanut Wok")
        is False
    )


def test_score_c_injected_extractor_none(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert metrics.score_C("...", "Saffron Thai", _C_ITEMS, judge=lambda o, n: None) is False


def test_score_c_choice_normalized_comparison(monkeypatch):
    install_complete_must_not_call(monkeypatch)
    assert (
        metrics.score_C("...", "Saffron Thai", _C_ITEMS, judge=lambda o, n: "saffron thai")
        is True
    )


def test_score_c_default_judge_path(monkeypatch):
    calls: list = []
    install_complete(monkeypatch, {"choice": "Saffron Thai"}, record=calls)

    assert metrics.score_C("I'd go with the Thai place", "Saffron Thai", _C_ITEMS) is True
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- #
# count_prompt_tokens — injected counter (no tiktoken)
# --------------------------------------------------------------------------- #
def test_count_prompt_tokens_uses_injected_counter():
    assert metrics.count_prompt_tokens("a b c", count_tokens=lambda t: len(t.split())) == 3
