"""Unit tests for ``memory.router.route`` (v0.4.1) — pure CPU, no GPU/model/network.

The only external seam is the LLM call. ``router.py`` does ``from .llm import complete``,
which binds the name into the *router* module namespace, so the test patches
``memory.router.complete`` (patching ``memory.llm.complete`` would not be seen by the
already-bound reference). A recording fake lets us assert both the verdict and whether
the LLM was consulted at all (the non-atomic short-circuit must NOT call it).

Behaviors covered (v0.4.1):
  * non-atomic shape (>15 words / URL / inline code / fenced code / blank) -> "rag",
    with NO llm call (recorder stays empty);
  * atomic + classifier {internalize, stable, confidence>=CONF_MIN} -> "edit";
  * confidence < CONF_MIN -> "rag"; missing "confidence" -> treated as 1.0 -> "edit";
  * not-internalize / not-stable -> "rag";
  * confidence exactly at the CONF_MIN boundary -> "edit";
  * llm raises -> "rag"; malformed (non-JSON) reply -> "rag".
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory import router  # noqa: E402
from memory.router import CONF_MIN  # noqa: E402
from memory.schema import MemoryItem  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _item(text: str) -> MemoryItem:
    """Build a MemoryItem carrying ``text``. ``route`` here is irrelevant: route()
    reads ``item.text`` only and returns a fresh verdict."""
    return MemoryItem(
        id="x1",
        type="fact",
        text=text,
        route="rag",
        status="buffer",
        source="msg-1",
        ts=0.0,
    )


class _Recorder:
    """A fake ``complete`` that records every call and returns a canned reply
    (or raises). Signature mirrors ``memory.llm.complete``."""

    def __init__(self, *, reply: str | None = None, raises: bool = False):
        self._reply = reply
        self._raises = raises
        self.calls: list[dict] = []

    def __call__(self, messages, *, model=None, temperature=0.0, response_format=None):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "response_format": response_format,
            }
        )
        if self._raises:
            raise RuntimeError("boom: llm.complete failed")
        return self._reply


def _install(monkeypatch, recorder: _Recorder) -> _Recorder:
    """Patch the LLM seam actually used by ``route`` and return the recorder."""
    monkeypatch.setattr(router, "complete", recorder)
    return recorder


def _judge(*, internalize=True, stable=True, confidence=0.95) -> str:
    """Serialize a classifier verdict as the STRICT-JSON string the LLM would emit."""
    payload: dict = {"internalize": internalize, "stable": stable}
    if confidence is not None:
        payload["confidence"] = confidence
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# 1. Non-atomic shapes short-circuit to "rag" with NO llm call
# --------------------------------------------------------------------------- #
def test_long_text_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    text = (
        "This is a very long sentence that clearly exceeds the fifteen word "
        "atomicity limit by a wide margin indeed"
    )
    assert len(text.split()) > 15  # guard: input really is non-atomic by length
    assert router.route(_item(text)) == "rag"
    assert rec.calls == []  # short-circuited before the classifier


def test_url_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    assert router.route(_item("See https://example.com for the spec")) == "rag"
    assert rec.calls == []


def test_www_url_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    assert router.route(_item("Docs live at www.example.org today")) == "rag"
    assert rec.calls == []


def test_inline_code_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    assert router.route(_item("Run `ls -la` to list files")) == "rag"
    assert rec.calls == []


def test_fenced_code_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    assert router.route(_item("```\nprint(1)\n```")) == "rag"
    assert rec.calls == []


def test_blank_text_routes_rag_without_llm_call(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    assert router.route(_item("   ")) == "rag"
    assert rec.calls == []


# --------------------------------------------------------------------------- #
# 2. Atomic + confident internalize/stable -> "edit"
# --------------------------------------------------------------------------- #
def test_edit_on_confident_internalize_and_stable(monkeypatch):
    rec = _install(
        monkeypatch,
        _Recorder(reply=_judge(internalize=True, stable=True, confidence=0.95)),
    )
    assert router.route(_item("I am allergic to peanuts")) == "edit"
    assert len(rec.calls) == 1  # the classifier WAS consulted


def test_edit_passes_item_text_as_final_user_message(monkeypatch):
    """route() reads item.text and forwards it as the last user turn (JSON mode)."""
    rec = _install(monkeypatch, _Recorder(reply=_judge()))
    router.route(_item("For OLTP I default to Postgres"))
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["messages"][-1] == {
        "role": "user",
        "content": "For OLTP I default to Postgres",
    }
    # JSON mode requested, deterministic temperature.
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0


def test_edit_when_confidence_exactly_at_threshold(monkeypatch):
    """Boundary: confidence == CONF_MIN passes (code uses ``>= CONF_MIN``)."""
    rec = _install(monkeypatch, _Recorder(reply=_judge(confidence=CONF_MIN)))
    assert router.route(_item("I am allergic to peanuts")) == "edit"
    assert len(rec.calls) == 1


def test_edit_when_confidence_field_missing_defaults_to_full(monkeypatch):
    """A v0.4 classifier that omits ``confidence`` is treated as fully confident."""
    rec = _install(
        monkeypatch,
        _Recorder(reply=_judge(internalize=True, stable=True, confidence=None)),
    )
    # Sanity: the payload really has no confidence key.
    assert "confidence" not in json.loads(rec._reply)
    assert router.route(_item("I am allergic to peanuts")) == "edit"
    assert len(rec.calls) == 1


# --------------------------------------------------------------------------- #
# 3. Atomic but classifier declines -> "rag"
# --------------------------------------------------------------------------- #
def test_rag_on_low_confidence(monkeypatch):
    rec = _install(
        monkeypatch,
        _Recorder(reply=_judge(internalize=True, stable=True, confidence=0.3)),
    )
    assert router.route(_item("Maybe I will switch editors")) == "rag"
    assert len(rec.calls) == 1  # classifier consulted, then declined on confidence


def test_rag_when_not_internalize(monkeypatch):
    rec = _install(
        monkeypatch,
        _Recorder(reply=_judge(internalize=False, stable=True, confidence=0.99)),
    )
    assert router.route(_item("Here is the API doc for X")) == "rag"
    assert len(rec.calls) == 1


def test_rag_when_not_stable(monkeypatch):
    rec = _install(
        monkeypatch,
        _Recorder(reply=_judge(internalize=True, stable=False, confidence=0.99)),
    )
    assert router.route(_item("I am currently waiting in line")) == "rag"
    assert len(rec.calls) == 1


# --------------------------------------------------------------------------- #
# 4. LLM/parse failures default to the safe "rag" route
# --------------------------------------------------------------------------- #
def test_rag_on_llm_exception(monkeypatch):
    rec = _install(monkeypatch, _Recorder(raises=True))
    assert router.route(_item("I am allergic to peanuts")) == "rag"
    assert len(rec.calls) == 1  # call attempted, then exception swallowed


def test_rag_on_malformed_json(monkeypatch):
    rec = _install(monkeypatch, _Recorder(reply="not json at all"))
    assert router.route(_item("I am allergic to peanuts")) == "rag"
    assert len(rec.calls) == 1


def test_rag_on_json_missing_required_keys(monkeypatch):
    """Valid JSON but missing the boolean axes -> KeyError -> safe "rag"."""
    rec = _install(monkeypatch, _Recorder(reply=json.dumps({"confidence": 0.9})))
    assert router.route(_item("I am allergic to peanuts")) == "rag"
    assert len(rec.calls) == 1
