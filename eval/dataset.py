"""Probe datasets for the capability matrix.

Combines benchmark facts (from vendored ``third_party`` data) with NEW preference / belief /
jargon items. Each item carries probes: {efficacy, paraphrase, application, locality}.
"""
from __future__ import annotations

from typing import Any


def load_probes() -> list[Any]:
    """Load probe items (benchmark facts + new preference/belief/jargon), each with its
    {efficacy, paraphrase, application, locality} probes. TODO.
    """
    raise NotImplementedError
