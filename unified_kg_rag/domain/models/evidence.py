# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a numeric configuration default rests on (its evidence tag).

Every numeric default of the configuration tree carries one tag in its
``Field(json_schema_extra=...)``, made by :func:`evidence`.
``scripts/gen_config_reference.py`` prints it in the configuration
reference and lists the defaults no run compared;
``tests/unit/test_config_evidence.py`` fails a numeric default without one.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic.config import JsonDict

EvidenceKind = Literal["measured", "upstream", "deviation", "operational", "unmeasured"]

EVIDENCE_KINDS: tuple[EvidenceKind, ...] = get_args(EvidenceKind)

# The page a ``measured`` tag's ref is a fragment of.
EVIDENCE_PAGE = "docs/concepts/defaults-and-evidence.md"

# What each tag says, as the configuration reference explains it.
EVIDENCE_MEANINGS: dict[EvidenceKind, str] = {
    "measured": (
        "a real-model run compared it with other values; the ref is the section "
        "of Defaults and evidence that holds the numbers"
    ),
    "upstream": (
        "the value of the original algorithm's reference implementation or of "
        "the service; the ref names it"
    ),
    "deviation": (
        "a deliberate departure from the original's value that no run "
        "compared; the ref names the original's value"
    ),
    "operational": (
        "bounds time, memory, request sizes or concurrency, not what is "
        "retrieved or answered"
    ),
    "unmeasured": (
        "changes what is retrieved or answered, and no run compared it with "
        "another value"
    ),
}

# The tags whose ref must say where the value comes from.
_NEEDS_REF: frozenset[EvidenceKind] = frozenset({"measured", "upstream", "deviation"})


def evidence(kind: EvidenceKind, ref: str | None = None) -> JsonDict:
    """The ``json_schema_extra`` of a numeric config field: its evidence tag.

    ``measured`` takes the ``#fragment`` of its section in
    :data:`EVIDENCE_PAGE`; ``upstream`` and ``deviation`` name the
    original's value; ``operational`` and ``unmeasured`` may add a note.
    """
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"unknown evidence kind {kind!r}")
    if kind in _NEEDS_REF and not ref:
        raise ValueError(f"evidence {kind!r} needs a ref")
    if kind == "measured" and not (ref or "").startswith("#"):
        raise ValueError(f"a measured ref is a #fragment of {EVIDENCE_PAGE}: {ref!r}")
    return {"evidence": kind, "ref": ref} if ref else {"evidence": kind}
