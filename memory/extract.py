"""Extract candidate ``MemoryItem``s from natural conversation (LLM-driven, not manual save)."""
from __future__ import annotations

import json
import time
import uuid
from typing import Sequence

from . import llm, router
from .schema import MemoryItem, PROV_EDIT, PROV_SOURCE_MSG

_SYSTEM = (
    "You extract ATOMIC, durable memories about the USER from a conversation. "
    "Surface only stable facts, preferences, beliefs, or jargon worth remembering; "
    "ignore small talk and transient context. "
    'Return STRICT JSON: an object {"items": [ ... ]} where each item is '
    '{"text": <=15-word canonical proposition, "type": one of '
    '"fact"|"preference"|"belief"|"jargon", "stem": cloze/question stem for editing, '
    '"target": the answer/value to teach, "subject": the entity}. '
    "Decompose so that stem + target reconstruct text. If nothing is worth remembering, "
    'return {"items": []}.'
)

_FEWSHOT_USER = (
    "USER: I'm JQ and I'm allergic to nickel buckles.\n"
    "USER: For OLTP I always reach for Postgres by default."
)
_FEWSHOT_ASSISTANT = json.dumps(
    {
        "items": [
            {
                "text": "JQ is allergic to nickel buckles",
                "type": "fact",
                "stem": "JQ is allergic to",
                "target": "nickel buckles",
                "subject": "JQ",
            },
            {
                "text": "User uses Postgres by default for OLTP",
                "type": "preference",
                "stem": "For OLTP the user defaults to",
                "target": "Postgres",
                "subject": "user",
            },
        ]
    }
)

_VALID_TYPES = {"fact", "preference", "belief", "jargon"}


def _render_chat(chat: Sequence[dict]) -> str:
    """Flatten an OpenAI-style chat into a plain transcript string."""
    lines = []
    for msg in chat:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _last_user_ref(chat: Sequence[dict]) -> str:
    """Pick a source reference from the last user message (id if present, else its text)."""
    for msg in reversed(chat):
        if msg.get("role") == "user":
            return str(msg.get("id") or msg.get("content") or "chat")
    return "chat"


def _parse_items(raw: str) -> list:
    """Robustly parse the LLM output into a list of candidate dicts; [] on any failure."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict)]


def extract(chat: Sequence[dict]) -> list[MemoryItem]:
    """Pull atomic memory candidates from ``chat`` with an LLM, then tag each via
    ``router.route``. Returns the routed candidates (not yet persisted).

    Builds a concise few-shot prompt asking for STRICT JSON candidates, calls
    ``llm.complete`` at temperature 0, parses robustly, and constructs one
    ``MemoryItem`` per candidate (placeholder ``route="rag"``, ``status="buffer"``).
    For items the router sends to ``"edit"``, the HoReN edit decomposition is recorded
    under ``provenance[PROV_EDIT]``. Does not persist and does not dedup.
    """
    if not chat:
        return []

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _FEWSHOT_USER},
        {"role": "assistant", "content": _FEWSHOT_ASSISTANT},
        {"role": "user", "content": _render_chat(chat)},
    ]

    try:
        raw = llm.complete(
            messages, temperature=0.0, response_format={"type": "json_object"}
        )
    except Exception:
        return []

    source_ref = _last_user_ref(chat)
    items: list[MemoryItem] = []
    for cand in _parse_items(raw):
        text = cand.get("text")
        mtype = cand.get("type")
        if not text or mtype not in _VALID_TYPES:
            continue
        stem = cand.get("stem")
        target = cand.get("target")
        subject = cand.get("subject")

        item = MemoryItem(
            id="mem_" + uuid.uuid4().hex[:8],
            type=mtype,
            text=str(text),
            route="rag",  # placeholder; router.route decides below
            status="buffer",
            source=source_ref,
            ts=time.time(),
            provenance={PROV_SOURCE_MSG: source_ref},
        )
        item.route = router.route(item)

        if item.route == "edit" and stem and target:
            item.provenance[PROV_EDIT] = {
                "stem": str(stem),
                "target": str(target),
                "subject": str(subject) if subject else "",
            }
        items.append(item)

    return items
