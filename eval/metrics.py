"""Evaluation metrics.

efficacy, paraphrase, locality, fluency (ppl), ctx_overhead_tokens, no_retrieval_recall.
Where clean, import HoReN's metric implementations (note the import path after vendoring);
the RAG-condition metrics and the LLM-judge are ours.
"""
from __future__ import annotations

from typing import Any, Sequence

# TODO(step3): import reusable HoReN metrics where clean, e.g.
#     from third_party.horen.<eval> import <metric>


def efficacy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Edit took: exact-match recall on the edited fact's direct probe. TODO."""
    raise NotImplementedError


def paraphrase(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Generalization: recall on reworded probes of the same fact. TODO."""
    raise NotImplementedError


def locality(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Locality: unrelated inputs stay unchanged after the edit. TODO."""
    raise NotImplementedError


def fluency_ppl(texts: Sequence[str], *, model: Any) -> float:
    """Fluency: perplexity of generated text under ``model``. TODO."""
    raise NotImplementedError


def ctx_overhead_tokens(messages: list[dict]) -> int:
    """Token cost added by the RAG window vs. a bare prompt. TODO."""
    raise NotImplementedError


def no_retrieval_recall(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Headline metric: correct answers with retrieval OFF (pure weight memory). TODO."""
    raise NotImplementedError
