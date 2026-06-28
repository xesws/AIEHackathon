"""Extract candidate ``MemoryItem``s from natural conversation (LLM-driven, not manual save).

Full tier (DESIGN §4.1): strong output-schema validation, per-candidate confidence
gating, coreference resolution (pronouns -> entities), multilingual extraction, and
batched/windowed extraction for long transcripts with de-duplicated merge. Every
quality step degrades gracefully: on any LLM/parse failure it falls back to the
v0.4 behavior and returns ``[]`` rather than crashing.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Sequence

from . import llm, router
from .schema import MemoryItem, PROV_EDIT, PROV_KEY_PROMPTS, PROV_SOURCE_MSG

# --- tunables (named constants) -------------------------------------------------
CONF_MIN = 0.5  # permissive confidence floor; candidates below this are dropped.
_MAX_SINGLE_CHARS = 6000  # transcripts longer than this are split into windows.
_WINDOW_MSGS = 12  # messages per extraction window when batching.
_OVERLAP_MSGS = 3  # overlap between consecutive windows so context isn't lost.

_VALID_TYPES = {"fact", "belief", "other"}

_SYSTEM = (
    "You extract ATOMIC, durable memories about the USER from a conversation. "
    "Surface only stable, reference-worthy information; ignore small talk and "
    "transient context. "
    "Classify each item's \"type\" as EXACTLY one of fact|belief|other -- this is "
    "the routing decision that determines where the memory is stored:\n"
    "  - fact = an OBJECTIVE, verifiable PERSONAL attribute of the user/JQ (cat "
    "name, car, allergy, address, alma mater, job). Strip the person and it is "
    "meaningless. It will go to RAG (retrieval).\n"
    "  - belief = the user's SUBJECTIVE opinion/preference/worldview (\"Rust is "
    "the best language\", \"SF summers are cold\"). Strip the person and it is a "
    "standalone world-assertion. It will be internalized into the model's weights.\n"
    "  - other = information NOT about the user personally but worth remembering "
    "(world knowledge, scheduled events, reference content). It will go to RAG.\n"
    "Resolve coreference so every field is self-contained: rewrite pronouns and "
    "ellipsis to concrete entities (\"I/my/me\" -> the user; \"it/he/she/they\" -> "
    "the named subject) before emitting. "
    "Work in the conversation's own language; do NOT translate to English. "
    'Return STRICT JSON: an object {"items": [ ... ]} where each item is '
    '{"text": <=15-word canonical proposition, "type": one of '
    '"fact"|"belief"|"other", "stem": cloze/question stem for editing, '
    '"target": the answer/value to teach, "subject": the entity, '
    '"key_prompts": 2-3 short answer-free retrieval prompts that emphasize the domain, '
    'subject, and relation without naming the target, '
    '"confidence": a number in [0,1] for how sure you are this is a durable memory}. '
    "Decompose so that stem + target reconstruct text. If nothing is worth remembering, "
    'return {"items": []}.'
)

_FEWSHOT_USER = (
    "USER: I'm JQ and I'm allergic to nickel buckles.\n"
    "USER: For OLTP I always reach for Postgres by default.\n"
    "USER: Our team standup is every Monday at 10am."
)
_FEWSHOT_ASSISTANT = json.dumps(
    {
        "items": [
            {
                # objective personal attribute of JQ -> fact -> RAG
                "text": "JQ is allergic to nickel buckles",
                "type": "fact",
                "stem": "JQ is allergic to",
                "target": "nickel buckles",
                "subject": "JQ",
                "key_prompts": [
                    "JQ allergy",
                    "user personal allergy",
                    "what is JQ allergic to",
                ],
                "confidence": 0.97,
            },
            {
                # subjective preference / worldview -> belief -> weights
                "text": "User uses Postgres by default for OLTP",
                "type": "belief",
                "stem": "For OLTP the user defaults to",
                "target": "Postgres",
                "subject": "user",
                "key_prompts": [
                    "OLTP database preference",
                    "user default database for OLTP",
                    "what database does the user prefer for OLTP",
                ],
                "confidence": 0.9,
            },
            {
                # not about the user personally, but worth remembering -> other -> RAG
                "text": "Team standup is every Monday at 10am",
                "type": "other",
                "stem": "Team standup is every",
                "target": "Monday at 10am",
                "subject": "team standup",
                "key_prompts": [
                    "team standup schedule",
                    "when is team standup",
                    "weekly standup time",
                ],
                "confidence": 0.85,
            },
        ]
    }
)


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


def _opt_str(value) -> "str | None":
    """Validate an optional string field: None/missing -> None, str -> str.

    Returns the sentinel ``False`` for malformed (non-str, non-None) values so the
    caller can drop the whole candidate.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return False  # malformed (e.g. dict/list/number where a string was required)


def _opt_str_list(value) -> "list[str] | None | bool":
    """Validate an optional string-list field.

    None/missing -> None; a list keeps non-empty strings; any non-list non-None value is
    malformed and returns False so the caller can drop the candidate.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        return False
    out = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return out or None


def _valid_candidate(cand: dict) -> "dict | None":
    """Strong schema validation + confidence gating for one raw candidate.

    Returns a normalized candidate dict (with float ``confidence``) or ``None`` if
    the candidate is malformed or below ``CONF_MIN``. Never raises.
    """
    text = cand.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    mtype = cand.get("type")
    if mtype not in _VALID_TYPES:
        return None

    stem = _opt_str(cand.get("stem"))
    target = _opt_str(cand.get("target"))
    subject = _opt_str(cand.get("subject"))
    key_prompts = _opt_str_list(cand.get(PROV_KEY_PROMPTS))
    if stem is False or target is False or subject is False:
        return None  # a present field was the wrong type -> malformed
    if key_prompts is False:
        return None

    # Confidence: permissive. Missing/unparseable -> assume confident (1.0) so clear
    # facts (and mocks that omit the field) keep passing; only drop explicit low conf.
    raw_conf = cand.get("confidence")
    try:
        conf = 1.0 if raw_conf is None else float(raw_conf)
    except (TypeError, ValueError):
        conf = 1.0
    if conf < CONF_MIN:
        return None

    return {
        "text": text.strip(),
        "type": mtype,
        "stem": stem,
        "target": target,
        "subject": subject,
        PROV_KEY_PROMPTS: key_prompts or [],
        "confidence": conf,
    }


def _windows(chat: Sequence[dict]) -> "list[Sequence[dict]]":
    """Split a long chat into overlapping message windows; short chats stay as one."""
    if len(_render_chat(chat)) <= _MAX_SINGLE_CHARS:
        return [chat]
    out: list = []
    step = max(1, _WINDOW_MSGS - _OVERLAP_MSGS)
    i = 0
    n = len(chat)
    while i < n:
        out.append(chat[i : i + _WINDOW_MSGS])
        if i + _WINDOW_MSGS >= n:
            break
        i += step
    return out or [chat]


def _extract_window(sub_chat: Sequence[dict]) -> list:
    """Run one LLM extraction call over a (sub)chat; return raw candidate dicts."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _FEWSHOT_USER},
        {"role": "assistant", "content": _FEWSHOT_ASSISTANT},
        {"role": "user", "content": _render_chat(sub_chat)},
    ]
    try:
        raw = llm.complete(
            messages, temperature=0.0, response_format={"type": "json_object"}
        )
    except Exception:
        return []  # fall back: this window contributes nothing, never crash.
    return _parse_items(raw)


def _norm_text(text: str) -> str:
    """Normalize a proposition for cross-window de-duplication (case/space folded)."""
    return " ".join(text.lower().split())


def extract(chat: Sequence[dict]) -> list[MemoryItem]:
    """Pull atomic memory candidates from ``chat`` with an LLM, then tag each via
    ``router.route``. Returns the routed candidates (not yet persisted).

    Builds a concise few-shot prompt asking for STRICT JSON candidates (with coref
    resolved, language preserved, and a per-candidate ``confidence``), calls
    ``llm.complete`` at temperature 0, validates each candidate against the output
    schema, drops malformed or low-confidence ones (``confidence < CONF_MIN``), and
    constructs one ``MemoryItem`` per surviving candidate (placeholder ``route="rag"``,
    ``status="buffer"``). For items the router sends to ``"edit"``, the HoReN edit
    decomposition is recorded under ``provenance[PROV_EDIT]``.

    Long transcripts are split into overlapping windows, extracted per window, then
    merged and de-duplicated by normalized text before routing; short chats use a
    single call (identical to v0.4). Does not persist and does not dedup-route.
    """
    if not chat:
        return []

    windows = _windows(chat)

    # Collect validated candidates, de-duplicating by normalized text across windows
    # (keeping the highest-confidence occurrence). Each carries its own source ref so
    # candidates from earlier windows still point at a sensible message.
    merged: dict[str, tuple] = {}  # norm_text -> (candidate, source_ref)
    for win in windows:
        win_ref = _last_user_ref(win)
        for raw_cand in _extract_window(win):
            cand = _valid_candidate(raw_cand)
            if cand is None:
                continue
            key = _norm_text(cand["text"])
            prev = merged.get(key)
            if prev is None or cand["confidence"] > prev[0]["confidence"]:
                merged[key] = (cand, win_ref)

    items: list[MemoryItem] = []
    for cand, source_ref in merged.values():
        item = MemoryItem(
            id="mem_" + uuid.uuid4().hex[:8],
            type=cand["type"],
            text=cand["text"],
            route="rag",  # placeholder; router.route decides below
            status="buffer",
            source=source_ref,
            ts=time.time(),
            provenance={PROV_SOURCE_MSG: source_ref},
        )
        item.route = router.route(item)

        if item.route == "edit" and cand["stem"] and cand["target"]:
            edit = {
                "stem": cand["stem"],
                "target": cand["target"],
                "subject": cand["subject"] or "",
            }
            if cand[PROV_KEY_PROMPTS]:
                edit[PROV_KEY_PROMPTS] = cand[PROV_KEY_PROMPTS]
            item.provenance[PROV_EDIT] = edit
        items.append(item)

    return items


_DECOMPOSE_SYSTEM = (
    "You decompose a SINGLE statement into an editable cloze. "
    "Given one statement, return STRICT JSON "
    '{"stem": <cloze/question stem>, "target": <the answer/value>, '
    '"subject": <the entity the statement is about>, '
    '"key_prompts": 2-3 short answer-free retrieval prompts that emphasize the domain, '
    'subject, and relation without naming the target} such that stem + target '
    "reconstruct the original statement. The stem is the statement with the target "
    "removed (a fill-in-the-blank prefix or question); the target is the specific "
    "answer/value being taught; the subject is the entity. Work in the statement's own "
    "language; do NOT translate."
)


def decompose(text: str) -> "dict | None":
    """One focused LLM call to split a single statement into ``{stem, target, subject}``.

    Mirrors ``_extract_window``'s JSON-mode style: asks the model for STRICT JSON whose
    ``stem`` + ``target`` reconstruct ``text``. Returns the decomposition only when both
    ``stem`` and ``target`` come back as non-empty strings (``subject`` defaults to
    ``""``); returns ``None`` on any malformed output, parse error, or LLM failure. Never
    raises.
    """
    try:
        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": text},
        ]
        raw = llm.complete(
            messages, temperature=0.0, response_format={"type": "json_object"}
        )
        data = json.loads(raw)
        s = data.get("stem")
        t = data.get("target")
        subj = data.get("subject")
        key_prompts = _opt_str_list(data.get(PROV_KEY_PROMPTS))
        if isinstance(s, str) and s.strip() and isinstance(t, str) and t.strip():
            return {"stem": s, "target": t, "subject": subj or "", PROV_KEY_PROMPTS: key_prompts or []}
        return None
    except Exception:
        return None
