"""Route a memory candidate to weight-edit vs. RAG. The axis is the SHAPE of the info, not its category."""
from __future__ import annotations

import json
import re

from .llm import complete
from .schema import MemoryItem, Route

# Atomicity thresholds: a single short proposition compresses into weights cheaply.
_MAX_WORDS = 15
# Sentence-terminator count; more than one terminator => multi-sentence => not atomic.
_SENTENCE_END = re.compile(r"[.!?。！？]+")
# Obvious rag shapes: URLs and fenced/inline code blocks are retrieve-and-recite content.
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE = re.compile(r"```|`[^`]+`")

_CLASSIFIER_SYSTEM = (
    "You classify a single user statement on two boolean axes for a continual-learning agent.\n"
    "internalize: true if the agent should BEHAVE as if it knows this (a durable fact, "
    "preference, belief, or term), false if it is reference content to retrieve and recite.\n"
    "stable: true if it is durable, false if it is a transient/momentary state "
    '(e.g. "right now", "today", "currently waiting").\n'
    'Reply with STRICT JSON only: {"internalize": bool, "stable": bool}'
)

_FEW_SHOT = [
    ("For OLTP I default to Postgres.", {"internalize": True, "stable": True}),
    ("I'm allergic to peanuts.", {"internalize": True, "stable": True}),
    ("I think gradual typing beats dynamic typing.", {"internalize": True, "stable": True}),
    ("I'm currently waiting in line for coffee.", {"internalize": True, "stable": False}),
    ("Here is the API documentation for project X.", {"internalize": False, "stable": True}),
]


def _is_atomic(text: str) -> bool:
    """Cheap deterministic shape test: a single sentence of <= ~15 words, no code/URLs."""
    stripped = text.strip()
    if not stripped:
        return False
    if _URL.search(stripped) or _CODE.search(stripped):
        return False
    if len(stripped.split()) > _MAX_WORDS:
        return False
    # At most one trailing sentence terminator; interior terminators imply multiple sentences.
    interior = _SENTENCE_END.sub("", stripped.rstrip(".!?。！？"))
    if _SENTENCE_END.search(interior):
        return False
    return True


def _classify_intent(text: str) -> tuple[bool, bool]:
    """Ask the LLM seam whether ``text`` is internalize-worthy and stable.

    Returns ``(internalize, stable)``. On any LLM/parse failure, returns ``(False, False)``
    so the caller defaults to the safe ``"rag"`` route.
    """
    messages: list[dict] = [{"role": "system", "content": _CLASSIFIER_SYSTEM}]
    for example, label in _FEW_SHOT:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps(label)})
    messages.append({"role": "user", "content": text})

    try:
        raw = complete(messages, temperature=0.0, response_format={"type": "json_object"})
        data = json.loads(raw)
        return bool(data["internalize"]), bool(data["stable"])
    except Exception:
        return False, False


def route(item: MemoryItem) -> Route:
    """Decide the route for ``item`` by the SHAPE of its information (INV-1).

    ``route == "edit"`` iff the text is atomic AND should be internalized AND is stable;
    otherwise ``"rag"``. ``atomic`` is a cheap deterministic length/sentence test (failing it
    short-circuits to ``"rag"`` with no LLM call); ``internalize`` and ``stable`` come from a
    small few-shot LLM classifier. On any uncertainty, LLM error, or parse failure we default
    to ``"rag"`` (reversible, does not touch weights). Reads ``item.text`` only.
    """
    if not _is_atomic(item.text):
        return "rag"
    internalize, stable = _classify_intent(item.text)
    return "edit" if (internalize and stable) else "rag"
