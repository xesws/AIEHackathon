"""Core data model shared across the memory system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

MemoryType = Literal["fact", "preference", "belief", "jargon"]
Route = Literal["edit", "rag"]
Status = Literal["buffer", "consolidated", "retired"]


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

    TODO: serialization / validation helpers as the system grows.
    """

    id: str
    type: MemoryType
    text: str
    route: Route
    status: Status
    source: str
    ts: float
    provenance: Optional[dict] = None
