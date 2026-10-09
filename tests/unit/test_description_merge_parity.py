# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every build path merges descriptions with one rule.

Extraction merges the instances of an entity it found in several text units,
the full-build resolver merges entity groups, and the cross-run merge folds a
delta into the stored graph. All three must join with a newline and drop
duplicate lines (order kept), or a full build and an incremental build of the
same corpus end up with different descriptions.
"""

from __future__ import annotations

from functools import reduce

import pytest

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.base_resolver import BaseResolver
from unified_kg_rag.domain.ingestion.merge import merge_entities
from unified_kg_rag.domain.models import Entity

pytestmark = pytest.mark.unit

_merge = BaseProcessor._merge_description


@pytest.mark.parametrize(
    ("current", "new", "expected"),
    [
        ("Vendor ships parts.", "Vendor ships parts.", "Vendor ships parts."),
        (
            "Vendor ships parts.",
            "Vendor bills Buyer.",
            "Vendor ships parts.\nVendor bills Buyer.",
        ),
        ("A\nB", "B", "A\nB"),
        (None, "A", "A"),
        ("A", None, "A"),
        ("  ", "A", "A"),
        (None, None, None),
    ],
)
def test_extraction_merge_joins_with_newline_and_dedupes(
    current, new, expected
) -> None:
    assert _merge(current, new) == expected


def test_full_and_incremental_builds_converge_on_the_description() -> None:
    # One entity described in three text units; the third repeats the first.
    descriptions = ["Vendor ships parts.", "Vendor bills Buyer.", "Vendor ships parts."]

    # Full build: extraction merges all three instances, then the resolver
    # merges the (single) group.
    extracted = reduce(_merge, descriptions)
    full = BaseResolver._merge_descriptions([extracted])

    # Incremental: the first two units in one run, the third in the next,
    # folded into the stored entity by the cross-run merge.
    stored = Entity(
        id="e1",
        name="Vendor",
        description=reduce(_merge, descriptions[:2]),
        text_unit_ids=["t1", "t2"],
    )
    delta = Entity(
        id="e1", name="Vendor", description=descriptions[2], text_unit_ids=["t3"]
    )
    (incremental,), _ = merge_entities([stored], [delta])

    assert full == incremental.description == "Vendor ships parts.\nVendor bills Buyer."
