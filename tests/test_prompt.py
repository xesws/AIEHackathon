"""Structure-invariant tests for memory.prompt.build_prompt (pure function, no GPU/network).

v0.x rebuild (rebuild_design.md §3.1 / INV-3): the prompt has THREE explicit segments —
FACT (rag_hits where type=="fact"), BUFFER (un-consolidated memory), DOCS (rag_hits where
type!="fact"). There is deliberately NO belief segment: belief is implicit in the weights.
"""
from __future__ import annotations

import pathlib
import sys

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memory.prompt import (  # noqa: E402
    BUFFER_HEADER,
    DOCS_HEADER,
    FACT_HEADER,
    SCENARIO_HEADER,
    SYSTEM,
    build_prompt,
)
from memory.schema import MemoryItem  # noqa: E402


def _item(id_: str, text: str, *, type_: str = "fact", route: str = "rag") -> MemoryItem:
    return MemoryItem(
        id=id_,
        type=type_,
        text=text,
        route=route,
        status="consolidated",
        source="msg-1",
        ts=0.0,
    )


def _system_content(messages: list[dict]) -> str:
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


def _segments(content: str) -> tuple[str, str, str]:
    """Split system content into (fact_segment, buffer_segment, docs_segment)."""
    assert FACT_HEADER in content
    assert BUFFER_HEADER in content
    assert DOCS_HEADER in content
    fact_seg, rest = content.split(BUFFER_HEADER, 1)
    buffer_seg, docs_seg = rest.split(DOCS_HEADER, 1)
    return fact_seg, buffer_seg, docs_seg


def test_first_message_is_system_and_contains_SYSTEM():
    msgs = build_prompt("hi", [], [])
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    assert SYSTEM in msgs[0]["content"]


def test_all_three_headers_and_none_when_all_empty():
    content = _system_content(build_prompt("q", [], []))
    assert FACT_HEADER in content
    assert BUFFER_HEADER in content
    assert DOCS_HEADER in content
    assert SCENARIO_HEADER not in content
    # Exactly one "(none)" per empty segment -> three.
    assert content.count("(none)") == 3
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "(none)" in fact_seg
    assert "(none)" in buf_seg
    assert "(none)" in docs_seg


def test_no_belief_segment_exists():
    """INV-3: there is no belief header/segment, ever — only FACT/BUFFER/DOCS."""
    content = _system_content(build_prompt("q", [], []))
    # No segment header advertises "belief" / "preference" in any case.
    lowered = content.lower()
    assert "belief" not in lowered
    # The three known headers are the ONLY bracketed segment headers.
    header_lines = [ln for ln in content.splitlines() if ln.startswith("[") and ln.endswith("]")]
    assert set(header_lines) == {FACT_HEADER, BUFFER_HEADER, DOCS_HEADER}


def test_fact_items_land_in_fact_segment_only():
    hits = [
        _item("f1", "JQ's cat is named Coco", type_="fact"),
        _item("f2", "JQ is allergic to peanuts", type_="fact"),
    ]
    content = _system_content(build_prompt("q", [], hits))
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "1. JQ's cat is named Coco" in fact_seg
    assert "2. JQ is allergic to peanuts" in fact_seg
    # Fact text must NOT leak into buffer or docs segments, which stay empty.
    assert "Coco" not in buf_seg and "Coco" not in docs_seg
    assert "(none)" in buf_seg
    assert "(none)" in docs_seg


def test_buffer_items_land_in_buffer_segment_only():
    buf = [
        _item("b1", "JQ thinks Zarithon is the best language", type_="belief", route="edit"),
        _item("b2", "JQ prefers dark mode", type_="belief", route="edit"),
    ]
    content = _system_content(build_prompt("q", buf, []))
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "1. JQ thinks Zarithon is the best language" in buf_seg
    assert "2. JQ prefers dark mode" in buf_seg
    # Buffer text must NOT leak into the fact or docs segments, which stay empty.
    assert "Zarithon" not in fact_seg and "Zarithon" not in docs_seg
    assert "(none)" in fact_seg
    assert "(none)" in docs_seg


def test_other_items_land_in_docs_segment_only():
    hits = [
        _item("o1", "the Q3 board meeting is on November 15th", type_="other"),
        _item("o2", "the office WiFi password is in the ops vault", type_="other"),
    ]
    content = _system_content(build_prompt("q", [], hits))
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "1. the Q3 board meeting is on November 15th" in docs_seg
    assert "2. the office WiFi password is in the ops vault" in docs_seg
    # Other/docs text must NOT leak into fact or buffer segments, which stay empty.
    assert "board meeting" not in fact_seg and "board meeting" not in buf_seg
    assert "(none)" in fact_seg
    assert "(none)" in buf_seg


def test_rag_hits_split_by_type_fact_vs_other():
    """A single rag_hits list is split INSIDE build_prompt: fact -> FACT seg, else -> DOCS."""
    hits = [
        _item("f1", "JQ lives on Maple Street", type_="fact"),
        _item("o1", "the product launch checklist has 12 steps", type_="other"),
    ]
    content = _system_content(build_prompt("q", [], hits))
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "1. JQ lives on Maple Street" in fact_seg
    assert "Maple Street" not in docs_seg
    assert "1. the product launch checklist has 12 steps" in docs_seg
    assert "checklist" not in fact_seg
    # buffer empty -> its own (none); fact & docs populated.
    assert "(none)" in buf_seg


def test_belief_looking_item_never_gets_its_own_segment():
    """INV-3: a belief-typed item passed via rag_hits falls into DOCS — never a belief segment."""
    belief = _item("x1", "JQ believes Vaelor is the capital of Oakhaven", type_="belief")
    content = _system_content(build_prompt("q", [], [belief]))
    fact_seg, buf_seg, docs_seg = _segments(content)
    # No new header was created for it; it is not in the FACT segment.
    assert "Vaelor" not in fact_seg
    # It lands in DOCS (everything that is not type=="fact").
    assert "Vaelor" in docs_seg
    # Still exactly the three known headers — no fourth/belief segment.
    header_lines = [ln for ln in content.splitlines() if ln.startswith("[") and ln.endswith("]")]
    assert set(header_lines) == {FACT_HEADER, BUFFER_HEADER, DOCS_HEADER}


def test_three_segments_kept_separate():
    fact = [_item("f1", "alpha-fact", type_="fact")]
    buf = [_item("b1", "beta-buffer", type_="belief", route="edit")]
    docs = [_item("o1", "omega-doc", type_="other")]
    content = _system_content(build_prompt("q", buf, fact + docs))
    fact_seg, buf_seg, docs_seg = _segments(content)
    assert "1. alpha-fact" in fact_seg and "alpha-fact" not in buf_seg and "alpha-fact" not in docs_seg
    assert "1. beta-buffer" in buf_seg and "beta-buffer" not in fact_seg and "beta-buffer" not in docs_seg
    assert "1. omega-doc" in docs_seg and "omega-doc" not in fact_seg and "omega-doc" not in buf_seg
    # All three segments populated -> no "(none)" anywhere.
    assert "(none)" not in content


def test_segment_order_is_fact_then_buffer_then_docs():
    content = _system_content(build_prompt("q", [], []))
    assert content.index(FACT_HEADER) < content.index(BUFFER_HEADER) < content.index(DOCS_HEADER)


def test_private_scenario_memories_use_separate_lane_before_rag():
    private = [_item("p1", "The best soccer player in the world is Pele", type_="belief")]
    facts = [_item("f1", "JQ's cat is named Coco", type_="fact")]
    docs = [_item("d1", "Public soccer rankings mention Messi", type_="other")]
    content = _system_content(build_prompt("q", [], facts + docs, private_memories=private))

    assert SCENARIO_HEADER in content
    assert content.index(SCENARIO_HEADER) < content.index(FACT_HEADER)
    private_seg = content.split(FACT_HEADER, 1)[0]
    _, _, docs_seg = _segments(content)
    assert "1. The best soccer player in the world is Pele" in private_seg
    assert "Pele" not in docs_seg
    assert "1. JQ's cat is named Coco" in content
    assert "1. Public soccer rankings mention Messi" in docs_seg


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
# FULL tier: optional token_budget / count_tokens.
# All pure-function, CPU-only — no GPU / model / network.
# ---------------------------------------------------------------------------


def test_token_budget_none_is_byte_stable_and_matches_no_arg_call():
    """token_budget=None == the hero/no-arg path, and is byte-stable across repeated calls."""
    q = "what is my favorite drink?"
    buf = [_item("b1", "JQ prefers dark mode", type_="belief", route="edit")]
    hits = [
        _item("f1", "JQ's cat is named Coco", type_="fact"),
        _item("o1", "the wifi password rotates monthly", type_="other"),
    ]
    history = [
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    ]
    baseline = build_prompt(q, buf, hits, history=history)
    # Byte-stable: repeated identical calls produce identical output.
    assert build_prompt(q, buf, hits, history=history) == baseline
    explicit_none = build_prompt(q, buf, hits, history=history, token_budget=None)
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


def test_budget_path_keeps_system_query_and_renders_all_three_headers():
    """Even under a tight budget, SYSTEM+query survive and all three headers render."""
    q = "tiny"
    buf = [_item("b1", "x", type_="belief", route="edit")]
    docs = [
        _item("r1", "this is a long reference document about cells", type_="other"),
        _item("r2", "another long reference document about wifi", type_="other"),
        _item("r3", "third long reference document about coffee", type_="other"),
    ]

    def approx(s: str) -> int:  # mirrors prompt._count default (len // 4)
        return len(s) // 4

    mandatory = (
        approx(SYSTEM)
        + approx(q)
        + approx(FACT_HEADER)
        + approx(BUFFER_HEADER)
        + approx(DOCS_HEADER)
        + 6  # _STRUCT_SLACK
    )
    # +3 leaves room for the single tiny buffer item ("1. x" -> cost 2), none for docs.
    budget = mandatory + 3
    msgs = build_prompt(q, buf, docs, token_budget=budget)
    content = _system_content(msgs)

    # Mandatory parts always survive the trim.
    assert SYSTEM in content
    assert FACT_HEADER in content
    assert BUFFER_HEADER in content
    assert DOCS_HEADER in content
    assert msgs[-1] == {"role": "user", "content": q}

    fact_seg, buf_seg, docs_seg = _segments(content)
    # Buffer item is kept (highest priority); docs are shed FIRST (lowest priority).
    assert "1. x" in buf_seg
    assert "3 more omitted" in docs_seg
    assert "reference document" not in docs_seg
    # Only the docs segment was trimmed (fact empty, buffer had nothing to omit).
    assert content.count("more omitted") == 1
    # No facts supplied -> FACT segment renders "(none)".
    assert "(none)" in fact_seg


def test_count_tokens_injection_drives_trimming_across_three_segments():
    """An injected count_tokens is actually invoked and governs what gets trimmed."""
    q = "q"
    buf = [_item("b1", "buffer fact", type_="belief", route="edit")]
    fact = [_item("f1", "fact fact", type_="fact")]
    docs = [_item("o1", "doc fact", type_="other")]

    calls: list[str] = []

    def counter(s: str) -> int:
        calls.append(s)
        return 1000  # every string is "expensive"

    # mandatory = SYSTEM + query + THREE headers (5 * 1000) + 6 slack = 5006 -> remaining 0.
    budget = 5006
    msgs = build_prompt(q, buf, fact + docs, token_budget=budget, count_tokens=counter)

    assert calls, "injected count_tokens must actually be invoked"
    content = _system_content(msgs)
    assert SYSTEM in content
    assert FACT_HEADER in content and BUFFER_HEADER in content and DOCS_HEADER in content
    assert msgs[-1] == {"role": "user", "content": q}

    fact_seg, buf_seg, docs_seg = _segments(content)
    # The inflated counter pushes EVERYTHING out -> all three segments show a marker.
    assert "1 more omitted" in fact_seg
    assert "1 more omitted" in buf_seg
    assert "1 more omitted" in docs_seg
    assert "buffer fact" not in content
    assert "fact fact" not in content
    assert "doc fact" not in content

    # Contrast: the SAME budget under the default (len // 4) counter omits nothing,
    # proving the injected counter — not the budget alone — drove the trimming.
    default_content = _system_content(
        build_prompt(q, buf, fact + docs, token_budget=budget)
    )
    assert "more omitted" not in default_content
    assert "1. buffer fact" in default_content
    assert "1. fact fact" in default_content
    assert "1. doc fact" in default_content
