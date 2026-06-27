"""Core data model shared across the memory system."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

MemoryType = Literal["fact", "preference", "belief", "jargon"]
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


@dataclass
class MemoryItem:
    """A single unit of remembered information.

    Fields:
        id:         stable unique id
        type:       fact | preference | belief | jargon
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
        verdict:   duplicate | supersede | new
        target_id: id of the existing memory the verdict refers to (if any)
    """

    verdict: DedupVerdict
    target_id: Optional[str] = None


def to_dict(item: MemoryItem) -> dict:
    """Serialize a ``MemoryItem`` to a plain dict."""
    return asdict(item)


def from_dict(d: dict) -> MemoryItem:
    """Reconstruct a ``MemoryItem`` from a dict.

    Tolerates a missing optional ``provenance`` and ignores unknown keys.
    Raises ``ValueError`` if a required field is missing.
    """
    required = ("id", "type", "text", "route", "status", "source", "ts")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"from_dict: missing required field(s): {', '.join(missing)}")
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
