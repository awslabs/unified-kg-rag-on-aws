# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stored edge rewritten for its rank alone is updated in place (AWS-free).

A delta that adds a neighbour to a hub changes the rank of every stored edge
at the hub (an edge's rank is its endpoints' degrees): almost all of a
delta's edge writes on a corpus with hubs. The commit wrote each such edge
whole (on Neptune a drop and re-add per edge); it now sets their rank in
place through ``GraphIndexer.update_relationship_ranks``, batched.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    expected_state,
    extract,
)

pytestmark = pytest.mark.integration

_V, _D, _C, _B = "Vendor", "Depot", "Carrier", "Buyer"
_BEFORE: Corpus = {
    "/corpus/a.txt": [[(_V, _D, 1)]],
    "/corpus/b.txt": [[(_C, _B, 2)]],
}


def _edge_id(corpus: Corpus, source: str, target: str) -> str:
    (rel,) = [
        r
        for r in extract(corpus).relationships
        if (r.source_name, r.target_name) == (source, target)
    ]
    return rel.id


def test_a_neighbours_degree_sets_the_stored_edges_rank_in_place() -> None:
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(_BEFORE)
    after = {**_BEFORE, "/corpus/c.txt": [[(_V, _C, 3)]]}

    harness.run(after)

    # Vendor and Carrier gained an edge: the stored edges at them changed
    # rank only, and nothing else of them is written.
    assert sorted(harness.graph.rank_updates) == sorted(
        [_edge_id(after, _V, _D), _edge_id(after, _C, _B)]
    )
    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []
