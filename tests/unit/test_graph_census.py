# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph census's digests: a fixed text per row, and one that moves with
every field it covers."""

from __future__ import annotations

import hashlib
import json

import pytest

from unified_kg_rag.application.storage.inspection import _Digest, _kind_census
from unified_kg_rag.ports.indexer import GraphCensusRow

pytestmark = pytest.mark.unit


def _edge(**overrides: object) -> GraphCensusRow:
    fields: dict[str, object] = {
        "id": "r1",
        "type": "SUPPLIES",
        "rank": 2,
        "description": "Vendor 공급 Buyer",
        "text_unit_ids": ["t1", "t2"],
        "source_id": "e1",
        "target_id": "e2",
        "weight": 1.5,
    }
    return GraphCensusRow(**{**fields, **overrides})  # type: ignore[arg-type]


def test_a_digest_is_the_sha256_of_one_json_array_a_line() -> None:
    digest = _Digest()
    digest.add(("r1", "공급", 2, ["t1"]))
    digest.add(("r2", None, 0.5, []))
    text = '["r1", "공급", 2, ["t1"]]\n["r2", null, 0.5, []]\n'
    assert digest.hexdigest() == hashlib.sha256(text.encode()).hexdigest()
    assert text.splitlines()[0] == json.dumps(
        ["r1", "공급", 2, ["t1"]], ensure_ascii=False
    )


@pytest.mark.parametrize(
    ("change", "moved"),
    [
        ({"type": "OWNS"}, {"digest"}),
        ({"rank": 3}, {"rank_digest"}),
        ({"text_unit_ids": ["t1"]}, {"provenance_digest"}),
        ({"description": "other"}, {"description_digest"}),
        ({"target_id": "e3"}, {"topology_digest"}),
        ({"weight": 9.0}, {"topology_digest"}),
    ],
)
def test_each_digest_moves_with_its_own_fields_only(
    change: dict[str, object], moved: set[str]
) -> None:
    base = _kind_census([[_edge()]], edges=True).model_dump()
    other = _kind_census([[_edge(**change)]], edges=True).model_dump()
    differing = {k for k in base if k.endswith("digest") and base[k] != other[k]}
    assert differing == moved
