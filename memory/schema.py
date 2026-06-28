"""Core data model shared across the memory system."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

MemoryType = Literal["fact", "belief", "other"]
Route = Literal["edit", "rag"]
Status = Literal["buffer", "consolidated", "retired"]
DedupVerdict = Literal["duplicate", "supersede", "new"]

# Provenance-key string constants (single source of truth).
PROV_SOURCE_MSG = "source_msg"
PROV_SUPERSEDES = "supersedes"
PROV_SUPERSEDED_BY = "superseded_by"
PROV_EDIT_REF = "edit_ref"
PROV_CONSOLIDATED_AT = "consolidated_at"
PROV_DUPLICATE_OF = "duplicate_of"
PROV_EDIT = "edit"
# Extra answer-free retrieval prompts used to append additional HoReN codebook keys for
# the same edit value. Stored inside PROV_EDIT as edit["key_prompts"].
PROV_KEY_PROMPTS = "key_prompts"
# Codebook key indices this item produced at consolidation:
# {"native": int, "chat": int, "canonical": [int, ...]}.
# Lets the serving layer attribute a generated answer's matched codebook slot back to its memory.
PROV_CODEBOOK_KEYS = "codebook_keys"


@dataclass
class MemoryItem:
    """A single unit of remembered information.

    Fields:
        id:         stable unique id
        type:       fact | belief | other
        text:       canonical natural-language statement
        route:      edit (-> weights) | rag (-> retrieval store)
        status:     buffer | consolidated | retired
        source:     where it came from (e.g. chat turn / message id)
        ts:         creation timestamp (epoch seconds)
        provenance: free-form trail (supersedes, edit ids, etc.)
    """

    id: str
    type: MemoryType
    text: str
    route: Route
    status: Status
    source: str
    ts: float
    provenance: Optional[dict] = None


@dataclass
class Decision:
    """Outcome of dedup classification at consolidation time.

    Fields:
        verdict:    duplicate | supersede | new
        target_id:  PRIMARY target id the verdict refers to (if any). For a
                    supersede this is the first / most-relevant superseded
                    memory; for duplicate it is the matched memory.
        target_ids: MULTI-TARGET supersede (DESIGN §10-11). When a candidate
                    supersedes several old memories at once, this holds ALL
                    superseded ids (including the primary ``target_id`` as the
                    first element). For single-target verdicts it stays None and
                    callers fall back to ``[target_id]``.
    """

    verdict: DedupVerdict
    target_id: Optional[str] = None
    target_ids: Optional[list[str]] = None


def to_dict(item: MemoryItem) -> dict:
    """Serialize a ``MemoryItem`` to a plain dict."""
    return asdict(item)


def from_dict(d: dict) -> MemoryItem:
    """Reconstruct a ``MemoryItem`` from a dict.

    Tolerates a missing/None optional ``provenance`` and ignores unknown keys.
    Raises ``ValueError`` if a required field is missing, or if an enum-valued
    field (``route`` / ``status`` / ``type``) is outside its allowed set, or if
    ``ts`` is not a number.
    """
    required = ("id", "type", "text", "route", "status", "source", "ts")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"from_dict: missing required field(s): {', '.join(missing)}")

    valid_routes = ("edit", "rag")
    if d["route"] not in valid_routes:
        raise ValueError(
            f"from_dict: invalid route {d['route']!r}; expected one of {valid_routes}"
        )
    valid_statuses = ("buffer", "consolidated", "retired")
    if d["status"] not in valid_statuses:
        raise ValueError(
            f"from_dict: invalid status {d['status']!r}; expected one of {valid_statuses}"
        )
    valid_types = ("fact", "belief", "other")
    if d["type"] not in valid_types:
        raise ValueError(
            f"from_dict: invalid type {d['type']!r}; expected one of {valid_types}"
        )
    # ``bool`` is a subclass of ``int`` but is not a valid timestamp.
    if isinstance(d["ts"], bool) or not isinstance(d["ts"], (int, float)):
        raise ValueError(f"from_dict: ts must be a number, got {type(d['ts']).__name__}")

    return MemoryItem(
        id=d["id"],
        type=d["type"],
        text=d["text"],
        route=d["route"],
        status=d["status"],
        source=d["source"],
        ts=d["ts"],
        provenance=d.get("provenance"),
    )
