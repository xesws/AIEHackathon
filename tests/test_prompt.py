"""Structure-invariant tests for memory.prompt.build_prompt (pure function, no GPU/network)."""
from __future__ import annotations

import pathlib
import sys

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory.prompt import SYSTEM, build_prompt  # noqa: E402
from memory.schema import MemoryItem  # noqa: E402

# Exact segment headers the RAG window must always render.
USER_HEADER = "[User facts/preferences — adopt by default]"
DOCS_HEADER = "[Reference material — does not override]"


def _item(id_: str, text: str, *, route: str = "rag") -> MemoryItem:
    return MemoryItem(
        id=id_,
        type="fact",
        text=text,
        route=route,
        status="consolidated",
        source="msg-1",
        ts=0.0,
    )


def _system_content(messages: list[dict]) -> str:
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


def _segments(content: str) -> tuple[str, str]:
    """Split system content into (buffer_segment, docs_segment) at the docs header."""
    assert DOCS_HEADER in content
    before, after = content.split(DOCS_HEADER, 1)
    return before, after


def test_first_message_is_system_and_contains_SYSTEM():
    msgs = build_prompt("hi", [], [])
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    assert SYSTEM in msgs[0]["content"]


def test_both_headers_and_none_when_both_empty():
    content = _system_content(build_prompt("q", [], []))
    assert USER_HEADER in content
    assert DOCS_HEADER in content
    # Exactly one "(none)" per empty segment.
    assert content.count("(none)") == 2
    buf_seg, docs_seg = _segments(content)
    assert "(none)" in buf_seg
    assert "(none)" in docs_seg


def test_buffer_items_numbered_in_user_segment_only():
    buf = [
        _item("b1", "drinks espresso after lunch", route="edit"),
        _item("b2", "prefers dark mode", route="edit"),
    ]
    content = _system_content(build_prompt("q", buf, []))
    buf_seg, docs_seg = _segments(content)
    # 1-based numbering, located in the buffer segment.
    assert "1. drinks espresso after lunch" in buf_seg
    assert "2. prefers dark mode" in buf_seg
    # Buffer text must NOT leak into the docs segment, which stays empty.
    assert "drinks espresso after lunch" not in docs_seg
    assert "prefers dark mode" not in docs_seg
    assert "(none)" in docs_seg


def test_rag_items_numbered_in_docs_segment_only():
    hits = [
        _item("r1", "the mitochondria is the powerhouse"),
        _item("r2", "the wifi password rotates monthly"),
    ]
    content = _system_content(build_prompt("q", [], hits))
    buf_seg, docs_seg = _segments(content)
    assert "1. the mitochondria is the powerhouse" in docs_seg
    assert "2. the wifi password rotates monthly" in docs_seg
    # Docs text must NOT leak into the buffer segment, which stays empty.
    assert "the mitochondria is the powerhouse" not in buf_seg
    assert "the wifi password rotates monthly" not in buf_seg
    assert "(none)" in buf_seg


def test_buffer_and_docs_segments_kept_separate():
    buf = [_item("b1", "alpha-buffer-fact", route="edit")]
    hits = [_item("r1", "omega-doc-fact")]
    content = _system_content(build_prompt("q", buf, hits))
    buf_seg, docs_seg = _segments(content)
    assert "1. alpha-buffer-fact" in buf_seg
    assert "alpha-buffer-fact" not in docs_seg
    assert "1. omega-doc-fact" in docs_seg
    assert "omega-doc-fact" not in buf_seg
    # Both segments populated -> no "(none)" anywhere.
    assert "(none)" not in content


def test_last_message_is_user_with_query():
    q = "what is my favorite drink?"
    msgs = build_prompt(q, [], [])
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == q


def test_history_spliced_between_system_and_final_user_in_order():
    history = [
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    ]
    q = "final-question"
    msgs = build_prompt(q, [], [], history=history)
    assert msgs[0]["role"] == "system"
    # History sits between system and the final user turn, order preserved.
    assert msgs[1] == history[0]
    assert msgs[2] == history[1]
    assert msgs[-1] == {"role": "user", "content": q}
    assert len(msgs) == 1 + len(history) + 1
