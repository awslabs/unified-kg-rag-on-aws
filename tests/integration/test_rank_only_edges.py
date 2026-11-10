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

import json
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    Killed,
    expected_state,
    extract,
)
from unified_kg_rag.domain.models import Config

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


@pytest.mark.parametrize(
    "names",
    [
        ("Hub", "Spoke", "Stuck", "Depot", "Node"),
        ("허브", "지점", "고장", "창고", "노드"),
        ("拠点", "支店", "故障", "倉庫", "ノード"),
    ],
    ids=["en", "ko", "ja"],
)
def test_a_write_that_keeps_failing_does_not_rewrite_unrelated_edges(
    names: tuple[str, str, str, str, str],
) -> None:
    # One entity's graph write is rejected on every run, so no commit
    # finishes cleanly. The hub's edges are rewritten once, by the run that
    # changes its degree; the later runs, which add unrelated documents,
    # rewrite none of them (each run's stored edges were compared with
    # both copies, nothing is carried forward for them).
    hub, spoke, stuck, depot, node = names
    corpus: Corpus = {
        "/corpus/hub.txt": [[(hub, f"{spoke}{i}", 1) for i in range(20)]],
        "/corpus/s.txt": [[(stuck, depot, 1)]],
    }
    config = Config()
    config.indexing.max_failure_rate = 0.95
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(corpus)
    (stuck_id,) = [e.id for e in extract(corpus, config).entities if e.name == stuck]

    def fail(where: str, item_id: str) -> bool:
        return where == "graph.upsert_entities" and item_id == stuck_id

    hub_edges = {
        r.id
        for r in extract(corpus, config).relationships
        if hub in (r.source_name, r.target_name)
    }
    corpus = {**corpus, "/corpus/h2.txt": [[(hub, f"{spoke}x", 2), (stuck, hub, 1)]]}
    harness.run(corpus, fail=fail)
    assert hub_edges <= set(harness.graph.rank_updates)
    for i in range(3):
        harness.graph.rank_updates.clear()
        corpus = {**corpus, f"/corpus/n{i}.txt": [[(f"{node}{i}", f"{node}{i}x", 1)]]}
        harness.run(corpus, fail=fail)
        assert not hub_edges & set(harness.graph.rank_updates)
        note = harness.vector.read_extraction_records(["pending-commit"])
        assert len(json.dumps(note)) < 200, note

    harness.run(corpus)
    assert harness.state() == expected_state(corpus, config)
    assert harness.registry_problems(corpus) == []


def test_a_note_listing_edge_entity_ids_is_read_safely() -> None:
    # A pending-commit note an earlier version wrote also lists the
    # entities whose stored edges its commit rewrote (``edge_entity_ids``):
    # the field is ignored (each run compares the stored edges it touches
    # with both copies) and the note is deleted by the next clean commit.
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(_BEFORE)
    vendor = next(e.id for e in extract(_BEFORE).entities if e.name == _V)
    harness.vector.write_extraction_records(
        {"pending-commit": {"entity_ids": [], "edge_entity_ids": [vendor, "e-gone"]}}
    )
    after = {**_BEFORE, "/corpus/c.txt": [[(_V, _C, 3)]]}

    harness.run(after)

    assert harness.vector.read_extraction_records(["pending-commit"]) == {}
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
def test_a_rank_a_killed_strip_left_behind_is_set_in_place(
    names: tuple[str, str, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Changing a.txt strips its edge to Carrier, so Depot-Carrier's rank
    # drops. The strip writes the graph edges, then their vector copies:
    # killed in between, the retry derives the graph copy's rank, and only
    # the vector copy's rank compared tells the edge apart. It is set in
    # place (no embedding), not written whole.
    v, d, c, b = names
    before: Corpus = {"/corpus/b.txt": [[(d, c, 1)]], "/corpus/a.txt": [[(v, c, 3)]]}
    after: Corpus = {"/corpus/b.txt": [[(d, c, 1)]], "/corpus/a.txt": [[(b, v, 3)]]}
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(before)

    def killed(*_args: object, **_kwargs: object) -> None:
        raise Killed("update_relationship_lineage")

    with monkeypatch.context() as patched:
        patched.setattr(harness.vector, "update_relationship_lineage", killed)
        harness.run(after)
    edge = _edge_id(before, d, c)
    assert harness.state()["vector_relationship_content"][edge] != (
        expected_state(after)["vector_relationship_content"][edge]
    )
    written: dict[str, list[str]] = {}
    for method in ("upsert_relationships", "update_relationship_lineage"):
        original = getattr(harness.vector, method)

        def recording(items: list[Any], _write: Any = original, _at: str = method):
            written.setdefault(_at, []).extend(r.id for r in items)
            return _write(items)

        monkeypatch.setattr(harness.vector, method, recording)

    harness.run(after)

    assert edge in written.get("update_relationship_lineage", [])
    assert edge not in written.get("upsert_relationships", [])
    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []
