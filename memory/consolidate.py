"""Buffer -> weights consolidation pass. ``n_written`` is the UI counter."""
from __future__ import annotations

# import editing                          # editing.edit(model, memory)
# from . import buffer, dedup, rag_store


def run_pass(trigger: str) -> int:
    """Run one consolidation pass (invoked by ``serving/triggers.py``).

    For each unconsolidated buffer item: classify vs. consolidated memory via ``dedup.classify``,
    then skip (duplicate) / retire-old + write (supersede) / write (new). Writing calls
    ``editing.edit``; on success drop the item from the buffer and record provenance.

    Returns ``n_written`` — the number of items folded into weights this pass (the UI counter).
    TODO.
    """
    raise NotImplementedError
