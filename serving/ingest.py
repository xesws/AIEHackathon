"""Observe a conversation and route its memories — the upstream half of the hero loop.

    ingest(chat) -> dict

``ingest`` runs the real extractor over a chat and routes EVERY surfaced candidate by its
own ``route``: edit-route items go to the consolidation ``buffer`` (later folded into weights
by ``consolidate.run_pass``), rag-route items go to the permanent ``rag_store``. It does NOT
consolidate — that is a separate trigger (``serving.triggers.manual``) so observation and the
(expensive, weight-touching) consolidation stay decoupled.

CRITICAL (v0.5.2): ALL edit-route items are buffered, not just the first. A single turn can
carry several facts ("I'm JQ, allergic to nickel buckles" → name + allergy); taking only
``items[0]`` silently drops the rest. The for-loop below buffers every one.
"""
from __future__ import annotations

from typing import Any, Sequence

from memory import buffer, extract, rag_store


def ingest(chat: Sequence[dict]) -> dict:
    """Extract memories from ``chat`` and route each to buffer (edit) or rag_store (rag).

    Returns counts for the UI / callers::

        {"n_extracted", "n_edit_buffered", "n_rag_indexed", "edit_ids"}

    No consolidation happens here (see ``serving.triggers.manual``). Extraction degrades
    gracefully (``extract`` returns ``[]`` on any LLM/parse failure), so a bad turn ingests
    nothing rather than raising.
    """
    items = extract.extract(chat)

    edit_ids: list[str] = []
    n_rag = 0
    for it in items:
        if it.route == "edit":
            buffer.append(it)          # EVERY edit-route item — never just items[0]
            edit_ids.append(it.id)
        elif it.route == "rag":
            rag_store.add(it)
            n_rag += 1

    return {
        "n_extracted": len(items),
        "n_edit_buffered": len(edit_ids),
        "n_rag_indexed": n_rag,
        "edit_ids": edit_ids,
    }
