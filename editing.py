"""Thin wrapper over the vendored HoReN editing backend (``third_party/horen``).

HoReN is a pre-existing external dependency (impl of arXiv 2605.08143); it is NEVER
reimplemented here. This module only adapts our memory objects to HoReN's API.

    edit(model, memory) -> edited_model | edit_module

The OUTPUT FORMAT — a full ``state_dict`` vs. a delta / side-module — is TBD and drives the
hot-swap path in ``serving/model_host.py``. The concrete HoReN import path follows the
vendored repo's actual package layout and must be adjusted after vendoring (scaffold step 3).
"""
from __future__ import annotations

from typing import Any

# TODO(step3): wire to the real package once third_party/horen is vendored, e.g.
#     from third_party.horen.<package> import <Editor / edit_fn>
# Adjust to the vendored repo's actual layout. Never reimplement HoReN here.


def edit(model: Any, memory: Any) -> Any:
    """Apply HoReN edit(s) for ``memory`` onto ``model``.

    Returns either the edited model or a swappable edit module (format TBD — see module
    docstring). Delegates entirely to ``third_party.horen``; no HoReN logic lives here.

    TODO: implement the adapter once the vendored package layout is known.
    """
    raise NotImplementedError
