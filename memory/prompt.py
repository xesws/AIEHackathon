"""Fixed inference prompt skeleton. The RAG window is ALWAYS rendered (content may be empty).

Pure function module (DESIGN §4.7 / §6): no LLM / embed / store / model / torch deps.

FULL tier adds an OPTIONAL token budget (`token_budget` + `count_tokens`). When the
budget is ``None`` (the default) the function behaves EXACTLY as the minimal/hero
version — byte-identical output — so existing callers and unit tests are unaffected.

Budget trimming (only when ``token_budget`` is set) obeys a strict priority so the
hero loop never regresses:

    ALWAYS kept ......... SYSTEM template + the user query (never dropped/truncated)
    priority (kept) ..... buffer-seg items  >  history turns  >  docs-seg items

i.e. lowest priority is dropped FIRST. Concretely the trim sequence is:
docs items are shed first, then the OLDEST history turns (newest turns kept, order
preserved), then — only if still over budget — buffer items. Each RAG segment header
is ALWAYS rendered even when its body is fully trimmed (INV-5); a trimmed segment
shows ``(… N more omitted)`` instead of silently dropping content. Token counts use
``count_tokens`` when provided, else an approximate ``len(text) // 4`` (no tokenizer,
no torch import). If the budget is too small to fit even the mandatory parts, SYSTEM
and the query are still kept (the budget is best-effort, never crash).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from .schema import MemoryItem

SYSTEM = (
    "You are Engram, a continual-learning personal assistant. "
    "The RAG window below holds memory about the user: adopt the user facts/preferences by "
    "default, and use the reference material when relevant."
)

# Exact RAG-window segment headers — ALWAYS rendered (INV-5), even when bodies are empty.
USER_HEADER = "[User facts/preferences — adopt by default]"
DOCS_HEADER = "[Reference material — does not override]"

_OMISSION = "(… {n} more omitted)"


def _render(items: Sequence[MemoryItem]) -> str:
    """Render items as a numbered list, or "(none)" when empty.

    Numbering keeps each ``item.text`` a discrete, labeled entry (a separator)
    so free-form memory content cannot break the window structure (injection).
    """
    if not items:
        return "(none)"
    return "\n".join(f"{i}. {it.text}" for i, it in enumerate(items, 1))


def _count(text: str, count_tokens: Optional[Callable[[str], int]]) -> int:
    """Estimate token count: caller-supplied counter, else ~len/4 (pure, no tokenizer)."""
    if count_tokens is not None:
        return count_tokens(text)
    return len(text) // 4


def _fit_prefix(
    items: Sequence[MemoryItem],
    budget: int,
    count_tokens: Optional[Callable[[str], int]],
) -> tuple[list[MemoryItem], int, int]:
    """Greedily keep a PREFIX of ``items`` (in order) that fits ``budget`` tokens.

    Returns ``(kept, omitted, used)``. When some items are dropped, reserves room for
    the ``(… N more omitted)`` marker, evicting already-kept items if the marker would
    not otherwise fit.
    """
    kept: list[MemoryItem] = []
    used = 0
    for it in items:
        cost = _count(f"{len(kept) + 1}. {it.text}", count_tokens) + 1  # +1 ~ newline
        if used + cost <= budget:
            kept.append(it)
            used += cost
        else:
            break

    omitted = len(items) - len(kept)
    if omitted > 0:
        marker_cost = _count(_OMISSION.format(n=omitted), count_tokens) + 1
        # Make space for the marker by evicting kept items (rarely needed).
        while kept and used + marker_cost > budget:
            removed = kept.pop()
            used -= _count(f"{len(kept) + 1}. {removed.text}", count_tokens) + 1
            omitted += 1
            marker_cost = _count(_OMISSION.format(n=omitted), count_tokens) + 1
        used += marker_cost
    return kept, omitted, used


def _fit_history_newest(
    history: Sequence[dict],
    budget: int,
    count_tokens: Optional[Callable[[str], int]],
) -> tuple[list[dict], int]:
    """Keep the NEWEST history turns that fit ``budget``; drop oldest first, preserve order."""
    kept_rev: list[dict] = []
    used = 0
    for turn in reversed(history):
        cost = _count(str(turn.get("content", "")), count_tokens) + 4  # +4 ~ role/format
        if used + cost <= budget:
            kept_rev.append(turn)
            used += cost
        else:
            break
    return list(reversed(kept_rev)), used


def _render_seg(kept: Sequence[MemoryItem], omitted: int, total: int) -> str:
    """Render one RAG segment body: numbered kept items + omission marker, or "(none)"."""
    if total == 0:
        return "(none)"
    lines = [f"{i}. {it.text}" for i, it in enumerate(kept, 1)]
    if omitted > 0:
        lines.append(_OMISSION.format(n=omitted))
    return "\n".join(lines)


def build_prompt(
    query: str,
    buffer: Sequence[MemoryItem],
    rag_hits: Sequence[MemoryItem],
    history: Sequence[dict] = (),
    *,
    token_budget: Optional[int] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> list[dict]:
    """Assemble chat ``messages`` with a FIXED skeleton:

        1. SYSTEM role.
        2. RAG window — ALWAYS present, two segments:
             (a) buffer seg — "facts/preferences about the user, adopt by default" (whole-inject)
             (b) docs seg   — "reference material — does not override" (vector top-k)
        3. conversation history.
        4. the user ``query``.

    The window structure is always rendered even when both segments are empty.

    Optional FULL-tier args (backward compatible):

    - ``token_budget``: when ``None`` (default) NOTHING is trimmed and the output is
      byte-identical to the minimal version. When set, the rendered content is trimmed
      to fit roughly ``token_budget`` tokens. SYSTEM and ``query`` are ALWAYS kept;
      otherwise priority is buffer items > history turns > docs items, so docs are shed
      first, then the oldest history turns, then buffer items. Both segment headers are
      always rendered (INV-5); a trimmed segment appends ``(… N more omitted)``.
    - ``count_tokens``: optional pure token counter; defaults to ``len(text) // 4``.
    """
    if token_budget is None:
        # Hero / minimal path — kept byte-identical to the original skeleton.
        rag_window = (
            f"{USER_HEADER}\n"
            f"{_render(buffer)}\n\n"
            f"{DOCS_HEADER}\n"
            f"{_render(rag_hits)}"
        )
        messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages

    # ---- FULL tier: budget-aware trimming -------------------------------------
    # Mandatory cost: SYSTEM + query + both headers (always rendered) + structural slack.
    remaining = token_budget
    remaining -= _count(SYSTEM, count_tokens)
    remaining -= _count(query, count_tokens)
    remaining -= _count(USER_HEADER, count_tokens)
    remaining -= _count(DOCS_HEADER, count_tokens)
    remaining -= 6  # newlines / "(none)" placeholders slack
    if remaining < 0:
        remaining = 0

    # Allocate by priority: buffer (highest) -> history -> docs (lowest).
    buf_kept, buf_omitted, buf_used = _fit_prefix(buffer, remaining, count_tokens)
    remaining = max(0, remaining - buf_used)

    hist_kept, hist_used = _fit_history_newest(history, remaining, count_tokens)
    remaining = max(0, remaining - hist_used)

    docs_kept, docs_omitted, _ = _fit_prefix(rag_hits, remaining, count_tokens)

    rag_window = (
        f"{USER_HEADER}\n"
        f"{_render_seg(buf_kept, buf_omitted, len(buffer))}\n\n"
        f"{DOCS_HEADER}\n"
        f"{_render_seg(docs_kept, docs_omitted, len(rag_hits))}"
    )
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
    messages.extend(hist_kept)
    messages.append({"role": "user", "content": query})
    return messages
