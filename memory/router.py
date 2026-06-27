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

# Minimum classifier confidence required to commit to a weight edit. Permissive by design:
# the cost of a wrong "edit" (touching weights) is higher than a wrong "rag" (reversible),
# so anything the model is not at least moderately sure about falls back to "rag".
CONF_MIN = 0.5

_CLASSIFIER_SYSTEM = (
    "You classify a single user statement on two boolean axes for a continual-learning agent.\n"
    "internalize: true if the agent should BEHAVE as if it knows this (a durable fact, "
    "preference, belief, or term), false if it is reference content to retrieve and recite.\n"
    "stable: true if it is durable, false if it is a transient/momentary state "
    '(e.g. "right now", "today", "currently waiting").\n'
    "confidence: your certainty in the two booleans above, a float in [0, 1]. Use a low "
    "value when the statement is ambiguous, underspecified, or could go either way.\n"
    'Reply with STRICT JSON only: {"internalize": bool, "stable": bool, "confidence": number}'
)

_FEW_SHOT = [
    ("For OLTP I default to Postgres.", {"internalize": True, "stable": True, "confidence": 0.95}),
    ("I'm allergic to peanuts.", {"internalize": True, "stable": True, "confidence": 0.98}),
    (
        "I think gradual typing beats dynamic typing.",
        {"internalize": True, "stable": True, "confidence": 0.85},
    ),
    (
        "I'm currently waiting in line for coffee.",
        {"internalize": True, "stable": False, "confidence": 0.9},
    ),
    (
        "Here is the API documentation for project X.",
        {"internalize": False, "stable": True, "confidence": 0.92},
    ),
    # Genuinely ambiguous shape -> low confidence -> caller must fall back to "rag".
    ("Maybe I'll switch editors at some point.", {"internalize": False, "stable": False, "confidence": 0.3}),
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


def _classify_intent(text: str) -> tuple[bool, bool, float]:
    """Ask the LLM seam whether ``text`` is internalize-worthy and stable, with confidence.

    Returns ``(internalize, stable, confidence)``. On any LLM/parse failure, or if any field
    is missing/malformed, returns ``(False, False, 0.0)`` so the caller defaults to the safe
    ``"rag"`` route (reversible, does not touch weights).
    """
    messages: list[dict] = [{"role": "system", "content": _CLASSIFIER_SYSTEM}]
    for example, label in _FEW_SHOT:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps(label)})
    messages.append({"role": "user", "content": text})

    try:
        raw = complete(messages, temperature=0.0, response_format={"type": "json_object"})
        data = json.loads(raw)
        internalize = bool(data["internalize"])
        stable = bool(data["stable"])
        # Confidence is optional for backward-compat with a v0.4 classifier that omits it:
        # a missing/non-numeric value is treated as fully confident so old behavior is preserved.
        raw_conf = data.get("confidence", 1.0)
        confidence = float(raw_conf)
        if confidence != confidence:  # NaN guard
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return internalize, stable, confidence
    except Exception:
        return False, False, 0.0


def route(item: MemoryItem) -> Route:
    """Decide the route for ``item`` by the SHAPE of its information (INV-1).

    ``route == "edit"`` iff the text is atomic AND should be internalized AND is stable AND the
    classifier's ``confidence`` is at least :data:`CONF_MIN`; otherwise ``"rag"``. ``atomic`` is
    a cheap deterministic length/sentence test (failing it short-circuits to ``"rag"`` with no
    LLM call); ``internalize``, ``stable`` and ``confidence`` come from a small few-shot LLM
    classifier. On any uncertainty, low confidence, LLM error, or parse failure we default to
    ``"rag"`` (reversible, does not touch weights). Reads ``item.text`` only.
    """
    if not _is_atomic(item.text):
        return "rag"
    internalize, stable, confidence = _classify_intent(item.text)
    if internalize and stable and confidence >= CONF_MIN:
        return "edit"
    return "rag"
