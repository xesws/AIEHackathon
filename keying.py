"""Plan B (v0.3) — query-span-isolated key extraction for HoReN retrieval.

The HoReN retrieval key is a layer-29 hidden state of the input's forward pass. At edit
time HoReN keys on the RAW stem; at chat inference the prompt is wrapped in a fixed
scaffold (system + RAG window + role headers), so the legacy `last_60_perc_tokens_avg`
read-key averages in that scaffold and lands far from the write-key (score < 0.85 → the
edit never fires).

Fix: compute the key from ONLY the rows of the layer-29 hidden-state tensor that belong to
the user-turn (query) tokens — exclude the scaffold rows — using the IDENTICAL slice on
both write and read. This module is the single source of truth for that extraction; it
REUSES the adapter's own `_pool_span` / `_select_query` so write and read never diverge.

Scope: hero template (fixed system + EMPTY RAG window). The mechanism already generalizes
to a non-empty window (the read span is located in the real prompt regardless), but only the
empty-window hero is wired and verified this round.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from memory.prompt import build_prompt


def _hero_render(tok: Any, text: str) -> str:
    """The EXACT chat string inference uses for the hero (EMPTY RAG window), via the same
    apply_chat_template path generate.py / build_prompt use. No hand-formatting."""
    messages = build_prompt(text, [], [])  # empty buffer + empty rag_hits -> EMPTY RAG window
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def query_span_in_rendered(tok: Any, rendered: str, text: str) -> Tuple[int, int]:
    """Return (start, end) inclusive token indices of the user-turn `text` within an
    already-rendered chat string, via offset_mapping (fast tokenizer). Special / role tokens
    have empty offset spans and are excluded; only tokens overlapping the query char span are
    kept (robust to BPE boundary merges and double-BOS — offsets are computed on the actual
    tokenization fed to the model). Locating in the *real* rendered string keeps the read span
    correct whatever the RAG window holds."""
    enc = tok(rendered, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    char_start = rendered.rindex(text)  # the user-turn occurrence (system text is fixed)
    char_end = char_start + len(text)
    idxs = [
        i
        for i, (cs, ce) in enumerate(offsets)
        if ce > cs and cs < char_end and ce > char_start  # non-special, overlaps query span
    ]
    if not idxs:
        raise ValueError(f"could not locate query span for text={text!r}")
    return idxs[0], idxs[-1]


def locate_query_span(tok: Any, text: str, *, templated: bool) -> Tuple[int, int]:
    """Convenience: query-span indices for `text`.
    templated=True  -> located in the hero (empty-window) render.
    templated=False -> the whole raw prompt is the query (first non-special token .. last)."""
    if templated:
        return query_span_in_rendered(tok, _hero_render(tok, text), text)
    ids = tok(text)["input_ids"]
    special = set(getattr(tok, "all_special_ids", []) or [])
    start = 0
    while start < len(ids) - 1 and ids[start] in special:
        start += 1
    return start, len(ids) - 1


def compute_key(
    text: str,
    *,
    templated: bool,
    hf_model: Any,
    tok: Any,
    adapter: Any,
) -> torch.Tensor:
    """Extract the normalized retrieval key [1, D] for `text`.

    Runs ONE forward of `hf_model` over the (raw or hero-templated) `text`, captures the
    layer-29 down_proj INPUT via a forward_pre_hook on `adapter`, and pools it through the
    adapter's OWN extractor:
      - templated=True  -> adapter._pool_span over the query-span rows (Plan B).
      - templated=False -> adapter._select_query at the last token (HoReN's legacy raw key).
    `adapter_mode='none'` during the forward makes it a pure capture (no match / inject /
    state mutation); args[0] (the down_proj input) is identical regardless of mode, so this
    is faithful to what forward computes at inference.
    """
    if templated:
        rendered = _hero_render(tok, text)
        enc = tok(rendered, return_tensors="pt").to(adapter.device)
        span: Optional[Tuple[int, int]] = query_span_in_rendered(tok, rendered, text)
    else:
        enc = tok(text, return_tensors="pt").to(adapter.device)
        span = None

    captured: dict = {}

    def _pre_hook(_module, args):
        captured["x"] = args[0]

    handle = adapter.register_forward_pre_hook(_pre_hook)
    old_mode = adapter.adapter_mode
    adapter.adapter_mode = "none"
    try:
        with torch.no_grad():
            hf_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        adapter.adapter_mode = old_mode
        handle.remove()

    x = captured["x"]
    if span is not None:
        key = adapter._pool_span(x, span[0], span[1])
    else:
        key = adapter._select_query(x, x.shape[1] - 1)
    if adapter.normalize_codebook_keys:
        key = F.normalize(key, p=2, dim=-1)
    return key  # [1, D]


def score(read_key: torch.Tensor, write_key: torch.Tensor, adapter: Any) -> float:
    """Max retrieval score of `read_key` against a codebook seeded as
    [random_placeholder, write_key] — using the adapter's OWN `_query` (the production gate
    function), so this number IS the value compared to the 0.85 threshold at inference.
    Computed in float32 for clean diagnostics (bf16 gate differs by <~0.01)."""
    saved = adapter.keys
    placeholder = saved[0:1].float()  # the random key at index 0 (mirrors production codebook)
    codebook = torch.cat([placeholder, write_key.float()], dim=0)
    adapter.keys = codebook
    try:
        sims = adapter._query(read_key.float())
    finally:
        adapter.keys = saved
    return sims.max().item()


def gate(text: str, *, hf_model: Any, tok: Any, adapter: Any) -> Tuple[float, int]:
    """HoReN deferral gate for `text` against the INSTALLED codebook: returns ``(sim, slot)``.

    ``sim`` = the max normalized-Hopfield score — the SAME value compared to the 0.85 threshold
    at inference; ``slot`` = argmax codebook key index (which row matched). Reuses ``compute_key``
    (the chat query-span key) + the adapter's own ``_query``, so this IS the production gate, not
    a re-implementation. Mirrors ``eval/runtime.live_score`` and additionally surfaces the slot,
    so serving can attribute the matched row back to the memory that created it.
    """
    rk = compute_key(text, templated=True, hf_model=hf_model, tok=tok, adapter=adapter)
    scores = adapter._query(rk)
    return scores.max().item(), int(scores.argmax().item())
