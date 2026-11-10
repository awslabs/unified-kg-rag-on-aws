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
    Killed,
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


@pytest.mark.parametrize(
    "names",
    [
        ("Vendor", "Depot", "Carrier", "Buyer"),
        ("공급사", "창고", "운송사", "구매자"),
        ("仕入先", "倉庫", "運送会社", "買い手"),
    ],
    ids=["en", "ko", "ja"],
)
def test_a_rank_the_killed_commit_left_on_the_vector_copy_is_rewritten(
    names: tuple[str, str, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The commit adding c.txt sets the new ranks of the stored edges at
    # Vendor and Carrier on their vector copies (the first phase) and is
    # killed before the graph's (the second). The run removing c.txt
    # derives the ranks back to the graph copies', so the edges were not
    # written again and their vector copies kept the killed commit's rank.
    v, d, c, b = names
    before: Corpus = {"/corpus/a.txt": [[(v, d, 1)]], "/corpus/b.txt": [[(c, b, 2)]]}
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(before)

    def killed(*_args: object, **_kwargs: object) -> None:
        raise Killed("update_relationship_ranks")

    with monkeypatch.context() as patched:
        patched.setattr(harness.graph, "update_relationship_ranks", killed)
        harness.run({**before, "/corpus/c.txt": [[(v, c, 3)]]})
    edge = _edge_id(before, v, d)
    assert harness.state()["vector_relationship_content"][edge] != (
        expected_state(before)["vector_relationship_content"][edge]
    )

    harness.run(before)

    assert harness.state() == expected_state(before)
    assert harness.registry_problems(before) == []
