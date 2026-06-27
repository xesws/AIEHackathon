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


# ---------------------------------------------------------------------------
# v0.4.1 FULL tier: optional token_budget / count_tokens (appended tests).
# All pure-function, CPU-only — no GPU / model / network.
# ---------------------------------------------------------------------------


def test_token_budget_none_is_byte_identical_to_no_arg_call():
    """token_budget=None (and even with a count_tokens passed) == the hero/no-arg path."""
    q = "what is my favorite drink?"
    buf = [
        _item("b1", "drinks espresso after lunch", route="edit"),
        _item("b2", "prefers dark mode", route="edit"),
    ]
    hits = [
        _item("r1", "the mitochondria is the powerhouse"),
        _item("r2", "the wifi password rotates monthly"),
    ]
    history = [
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    ]
    baseline = build_prompt(q, buf, hits, history=history)
    explicit_none = build_prompt(q, buf, hits, history=history, token_budget=None)
    # Full list equality == byte-identical strings (dicts compare value-wise).
    assert explicit_none == baseline
    # count_tokens MUST be ignored entirely when the budget is None.
    with_counter = build_prompt(
        q, buf, hits, history=history, token_budget=None, count_tokens=len
    )
    assert with_counter == baseline
    # And explicitly: concatenated content is identical (no silent reshaping).
    assert "".join(m["content"] for m in explicit_none) == "".join(
        m["content"] for m in baseline
    )


def test_tiny_budget_sheds_docs_before_buffer_with_omission_marker():
    """A tight budget keeps SYSTEM+query+both headers, drops docs first, marks omissions."""
    q = "tiny"
    buf = [_item("b1", "x", route="edit")]
    docs = [
        _item("r1", "this is a long reference document about cells"),
        _item("r2", "another long reference document about wifi"),
        _item("r3", "third long reference document about coffee"),
    ]

    def approx(s: str) -> int:  # mirrors prompt._count default (len // 4)
        return len(s) // 4

    mandatory = approx(SYSTEM) + approx(q) + approx(USER_HEADER) + approx(DOCS_HEADER) + 6
    # +3 leaves room for the single tiny buffer item ("1. x" -> cost 2), none for docs.
    budget = mandatory + 3
    msgs = build_prompt(q, buf, docs, token_budget=budget)
    content = _system_content(msgs)

    # Mandatory parts always survive the trim.
    assert SYSTEM in content
    assert USER_HEADER in content
    assert DOCS_HEADER in content
    assert msgs[-1] == {"role": "user", "content": q}

    buf_seg, docs_seg = _segments(content)
    # Buffer item is kept; docs are shed FIRST (lowest priority).
    assert "1. x" in buf_seg
    assert "3 more omitted" in docs_seg
    assert "reference document" not in docs_seg
    # Only the docs segment was trimmed (buffer had nothing to omit).
    assert content.count("more omitted") == 1


def test_count_tokens_injection_drives_trimming():
    """An injected count_tokens is actually invoked and governs what gets trimmed."""
    q = "q"
    buf = [_item("b1", "buffer fact", route="edit")]
    docs = [_item("r1", "doc fact")]

    calls: list[str] = []

    def counter(s: str) -> int:
        calls.append(s)
        return 1000  # every string is "expensive"

    # mandatory = SYSTEM + query + both headers (4 * 1000) + 6 slack = 4006 -> remaining 0.
    budget = 4006
    msgs = build_prompt(q, buf, docs, token_budget=budget, count_tokens=counter)

    assert calls, "injected count_tokens must actually be invoked"
    content = _system_content(msgs)
    assert SYSTEM in content
    assert USER_HEADER in content and DOCS_HEADER in content
    assert msgs[-1] == {"role": "user", "content": q}

    buf_seg, docs_seg = _segments(content)
    # The inflated counter pushes EVERYTHING out -> both segments show a marker.
    assert "1 more omitted" in buf_seg
    assert "1 more omitted" in docs_seg
    assert "buffer fact" not in content
    assert "doc fact" not in content

    # Contrast: the SAME budget under the default (len // 4) counter omits nothing,
    # proving the injected counter — not the budget alone — drove the trimming.
    default_content = _system_content(build_prompt(q, buf, docs, token_budget=budget))
    assert "more omitted" not in default_content
    assert "1. buffer fact" in default_content
    assert "1. doc fact" in default_content
