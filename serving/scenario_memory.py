"""Scenario-memory planner for free-form chat turns.

Open-ended prompts often do not look like the explicit cloze/QA strings that HoReN's
codebook keys were written against. This module adds a narrow read-only planner:

1. lexical planning: pick a few consolidated edit memories whose domain/relation words
   overlap the user's scenario prompt;
2. query expansion: probe each candidate with answer-free canonical aliases;
3. codebook verification: keep a candidate only if the expanded probe gates back to that
   candidate's own codebook rows above threshold.

The selected memories are passed through a separate private-memory lane in the prompt.
They are not RAG hits and do not come from the RAG store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import keying
from memory import schema, store
from memory.schema import MemoryItem
from serving import model_host

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "draft", "for", "from", "give", "have", "how", "include", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "plan", "should",
    "that", "the", "their", "them", "this", "to", "use", "what", "when", "who",
    "with", "write", "you", "your",
    "according", "believe", "believes", "opinion", "opinions", "think", "thinks",
    "user", "view", "views",
    "best", "ever", "greatest", "lived", "world",
}


@dataclass
class ScenarioPlan:
    """Planner output consumed by serving.app.chat."""

    selected: list[MemoryItem]
    records: list[dict]
    enabled: bool
    reason: str = ""


def _normalize_token(tok: str) -> str:
    tok = tok.lower()
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", str(text).lower()):
        tok = _normalize_token(raw)
        if len(tok) > 2 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _edit_payload(item: MemoryItem) -> dict:
    edit = (item.provenance or {}).get(schema.PROV_EDIT) or {}
    return edit if isinstance(edit, dict) else {}


def _clean_prompt(text: Any, target: str = "") -> str:
    s = str(text or "").strip()
    if target:
        s = re.sub(re.escape(target), " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;!?-_")
    return s


def _expanded_queries(item: MemoryItem) -> list[str]:
    """Answer-free lookup probes for one memory, ordered from explicit to fallback."""
    edit = _edit_payload(item)
    target = str(edit.get("target") or "")
    raw = []
    raw.extend(p for p in edit.get(schema.PROV_KEY_PROMPTS, []) or [] if isinstance(p, str))
    raw.extend([edit.get("subject", ""), edit.get("stem", ""), item.text])

    words = []
    for chunk in raw:
        words.extend(_tokens(_clean_prompt(chunk, target)))
    keyword_prompt = " ".join(list(dict.fromkeys(words))[:8])
    raw.append(keyword_prompt)

    out: list[str] = []
    seen: set[str] = set()
    for prompt in raw:
        prompt = _clean_prompt(prompt, target)
        key = prompt.lower()
        if prompt and key not in seen:
            seen.add(key)
            out.append(prompt)
    return out[:6]


def _candidate_score(message_tokens: set[str], item: MemoryItem) -> int:
    terms: set[str] = set()
    for query in _expanded_queries(item):
        terms.update(_tokens(query))
    if not terms:
        return 0
    overlap = message_tokens & terms
    return len(overlap)


def _candidate_memories(
    message: str,
    registry: Iterable[MemoryItem],
    *,
    max_candidates: int,
) -> list[MemoryItem]:
    message_tokens = _tokens(message)
    if not message_tokens:
        return []
    scored = []
    for item in registry:
        if item.status != "consolidated" or item.route != "edit":
            continue
        score = _candidate_score(message_tokens, item)
        if score:
            scored.append((score, item.ts, item.id, item))
    scored.sort(reverse=True)
    return [item for _score, _ts, _id, item in scored[:max_candidates]]


def _slots(item: MemoryItem) -> set[int]:
    keys = ((item.provenance or {}).get(schema.PROV_CODEBOOK_KEYS) or {})
    out: set[int] = set()
    for name in ("native", "chat"):
        if isinstance(keys.get(name), int):
            out.add(int(keys[name]))
    out.update(int(i) for i in keys.get("canonical", []) or [])
    return out


def _owner_for_slot(slot: int, registry: Iterable[MemoryItem]) -> Optional[MemoryItem]:
    for item in registry:
        if slot in _slots(item):
            return item
    return None


def _gate(query: str) -> tuple[float, int, float]:
    adapter = model_host.edit_module()
    current = model_host.current_model()
    hf_model = getattr(current, "model", current)
    sim, slot = keying.gate(
        query,
        hf_model=hf_model,
        tok=model_host.tokenizer(),
        adapter=adapter,
    )
    threshold = float(getattr(adapter, "hopfield_key_match_threshold", 0.85))
    return float(sim), int(slot), threshold


def plan(
    message: str,
    *,
    registry: Optional[list[MemoryItem]] = None,
    max_candidates: int = 4,
    max_selected: int = 3,
) -> ScenarioPlan:
    """Return planner-selected memories for ``message``.

    This function is best-effort. Any model/keying failure disables the scenario lane for
    this turn instead of breaking /chat.
    """
    if not model_host.edit_active():
        return ScenarioPlan(selected=[], records=[], enabled=False, reason="edit_inactive")

    registry = list(store.by_status("consolidated") if registry is None else registry)
    candidates = _candidate_memories(message, registry, max_candidates=max_candidates)
    if not candidates:
        return ScenarioPlan(selected=[], records=[], enabled=True, reason="no_candidates")

    selected: dict[str, MemoryItem] = {}
    records: list[dict] = []
    try:
        for candidate in candidates:
            best: dict | None = None
            for query in _expanded_queries(candidate):
                sim, slot, threshold = _gate(query)
                owner = _owner_for_slot(slot, registry)
                row = {
                    "candidate_id": candidate.id,
                    "query": query,
                    "similarity": sim,
                    "threshold": threshold,
                    "slot": slot,
                    "owner_id": owner.id if owner else None,
                    "hit": bool(owner and owner.id == candidate.id and sim >= threshold),
                }
                records.append(row)
                if best is None or row["similarity"] > best["similarity"]:
                    best = row
            if best and best["hit"] and candidate.id not in selected:
                selected[candidate.id] = candidate
            if len(selected) >= max_selected:
                break
    except Exception as exc:
        return ScenarioPlan(selected=[], records=records, enabled=False, reason=f"gate_failed:{exc}")

    return ScenarioPlan(
        selected=list(selected.values()),
        records=records,
        enabled=True,
        reason="ok" if selected else "no_verified_hits",
    )


def response(plan_result: ScenarioPlan) -> dict:
    """Small JSON-safe summary for /chat responses."""
    best_by_id: dict[str, dict] = {}
    for row in plan_result.records:
        cid = row["candidate_id"]
        if cid not in best_by_id or row["similarity"] > best_by_id[cid]["similarity"]:
            best_by_id[cid] = row
    return {
        "enabled": plan_result.enabled,
        "reason": plan_result.reason,
        "selected": [
            {
                "id": item.id,
                "text": item.text,
                "similarity": round(best_by_id.get(item.id, {}).get("similarity", 0.0), 4),
                "slot": best_by_id.get(item.id, {}).get("slot"),
            }
            for item in plan_result.selected
        ],
    }
