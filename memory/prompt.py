"""Fixed inference prompt skeleton. The RAG window is ALWAYS rendered (content may be empty).

Pure function module (DESIGN §4.7 / §6): no LLM / embed / store / model / torch deps.

★ STRUCTURE (rebuild_design.md §0.3 / §3.1 — the load-bearing invariant):
The prompt carries only EXPLICIT text segments that come from retrieval (fact + other)
plus a transitional BUFFER segment for un-consolidated memory. There is **NO belief
segment** and no segment fed from editing / retrieval-of-weights: belief lives in the
weights, influences the forward pass directly, and is therefore invisible in the prompt
(INV-3). At proof time the buffer is drained, so the BUFFER segment renders ``(none)``
and the prompt contains no belief text at all — yet the model still answers from weights.
That contrast (facts visible in the prompt, belief internalized in weights) is the demo's
proof visualization.

The single inbound ``rag_hits`` list is split INSIDE this module by ``type``:

    facts  = type == "fact"   -> FACT segment   (explicit RAG text about the user)
    others = type != "fact"   -> DOCS segment   (reference / other RAG text)

FULL tier adds an OPTIONAL token budget (``token_budget`` + ``count_tokens``). When the
budget is ``None`` (the default / hero path — the one that matters for the demo) the
function renders every segment in full and is byte-stable across repeated calls.

Budget trimming (only when ``token_budget`` is set) obeys a strict priority so the hero
loop never regresses:

    ALWAYS kept ......... SYSTEM template + the user query (never dropped/truncated)
    priority (kept) ..... BUFFER items  >  FACT items  >  history turns  >  DOCS items

i.e. lowest priority is shed FIRST. Concretely the trim sequence sheds docs items first,
then the OLDEST history turns (newest kept, order preserved), then fact items, and only
last the buffer (transitional belief). Each of the THREE segment headers is ALWAYS
rendered even when its body is fully trimmed (INV-5); a trimmed segment shows
``(… N more omitted)`` instead of silently dropping content. Token counts use
``count_tokens`` when provided, else an approximate ``len(text) // 4`` (no tokenizer, no
torch import). If the budget is too small to fit even the mandatory parts, SYSTEM and the
query are still kept (the budget is best-effort, never crash).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from .schema import MemoryItem

SYSTEM = (
    "You are Engram, a continual-learning personal assistant. "
    "The memory window below holds known facts about the user (adopt them by "
    "default) and reference material (use it when relevant; it does not override)."
)

# Exact RAG-window segment headers — ALWAYS rendered (INV-5), even when bodies are empty.
# There is deliberately NO belief header: belief is implicit in the weights (INV-3).
FACT_HEADER = "[Known facts about the user — adopt by default]"
BUFFER_HEADER = "[Pending unconsolidated memory — adopt by default]"
DOCS_HEADER = "[Reference material — does not override]"

_OMISSION = "(… {n} more omitted)"

# Mandatory structural slack (newlines / "(none)" placeholders) subtracted in the budget
# path. Best-effort only — kept consistent with the budget tests.
_STRUCT_SLACK = 6


def _split_rag(rag_hits: Sequence[MemoryItem]) -> tuple[list[MemoryItem], list[MemoryItem]]:
    """Split inbound RAG hits into (facts, others) by ``type`` (INV-3 framing).

    ``type == "fact"`` -> FACT segment; everything else (reference/other) -> DOCS segment.
    """
    facts = [h for h in rag_hits if getattr(h, "type", None) == "fact"]
    others = [h for h in rag_hits if getattr(h, "type", None) != "fact"]
    return facts, others


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
        2. memory window — ALWAYS present, THREE segments (each header always rendered):
             (a) FACT seg   — known facts about the user (rag_hits where type=="fact")
             (b) BUFFER seg — pending un-consolidated memory (the ``buffer``)
             (c) DOCS seg   — reference material (rag_hits where type!="fact")
        3. conversation history.
        4. the user ``query``.

    ★ There is NO belief segment and no segment fed from editing/retrieval-of-weights.
    belief lives in the weights and is invisible here (INV-3). ``rag_hits`` is split by
    ``type`` inside this function: ``fact`` -> FACT seg, everything else -> DOCS seg.

    The window structure is always rendered even when every segment is empty. The signature
    is positional-stable: ``generate.py`` and serving call ``build_prompt(query, buffer,
    rag_hits[, history])`` — the fact/other split happens INTERNALLY.

    Optional FULL-tier args (backward compatible):

    - ``token_budget``: when ``None`` (default) NOTHING is trimmed; the output renders every
      segment in full and is byte-stable across repeated calls. When set, content is trimmed
      to fit roughly ``token_budget`` tokens. SYSTEM and ``query`` are ALWAYS kept; otherwise
      priority is BUFFER > FACTS > history > DOCS, so docs are shed first, then the oldest
      history turns, then facts, then buffer. All three headers are always rendered (INV-5);
      a trimmed segment appends ``(… N more omitted)``.
    - ``count_tokens``: optional pure token counter; defaults to ``len(text) // 4``.
    """
    facts, others = _split_rag(rag_hits)

    if token_budget is None:
        # Hero / minimal path — render every segment in full (byte-stable).
        rag_window = (
            f"{FACT_HEADER}\n"
            f"{_render(facts)}\n\n"
            f"{BUFFER_HEADER}\n"
            f"{_render(buffer)}\n\n"
            f"{DOCS_HEADER}\n"
            f"{_render(others)}"
        )
        messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages

    # ---- FULL tier: budget-aware trimming -------------------------------------
    # Mandatory cost: SYSTEM + query + all THREE headers (always rendered) + structural slack.
    remaining = token_budget
    remaining -= _count(SYSTEM, count_tokens)
    remaining -= _count(query, count_tokens)
    remaining -= _count(FACT_HEADER, count_tokens)
    remaining -= _count(BUFFER_HEADER, count_tokens)
    remaining -= _count(DOCS_HEADER, count_tokens)
    remaining -= _STRUCT_SLACK  # newlines / "(none)" placeholders slack
    if remaining < 0:
        remaining = 0

    # Allocate by priority: BUFFER (highest) -> FACTS -> history -> DOCS (lowest).
    buf_kept, buf_omitted, buf_used = _fit_prefix(buffer, remaining, count_tokens)
    remaining = max(0, remaining - buf_used)

    fact_kept, fact_omitted, fact_used = _fit_prefix(facts, remaining, count_tokens)
    remaining = max(0, remaining - fact_used)

    hist_kept, hist_used = _fit_history_newest(history, remaining, count_tokens)
    remaining = max(0, remaining - hist_used)

    docs_kept, docs_omitted, _ = _fit_prefix(others, remaining, count_tokens)

    rag_window = (
        f"{FACT_HEADER}\n"
        f"{_render_seg(fact_kept, fact_omitted, len(facts))}\n\n"
        f"{BUFFER_HEADER}\n"
        f"{_render_seg(buf_kept, buf_omitted, len(buffer))}\n\n"
        f"{DOCS_HEADER}\n"
        f"{_render_seg(docs_kept, docs_omitted, len(others))}"
    )
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{rag_window}"}]
    messages.extend(hist_kept)
    messages.append({"role": "user", "content": query})
    return messages
