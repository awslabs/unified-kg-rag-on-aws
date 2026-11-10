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


_NAMES = {
    "en": ("Vendor", "Depot", "Acme", "Buyer"),
    "ko": ("공급사", "창고", "한빛", "구매자"),
    "ja": ("仕入先", "倉庫", "ひかり", "買い手"),
}


def _strip_shapes(v: str, d: str, x: str, b: str) -> dict[str, tuple[Corpus, Corpus]]:
    """Before and after corpora whose removal strips a stored item."""
    kept: Corpus = {"/corpus/b.txt": [[(d, x, 1)]]}
    return {
        # a.txt changes: the stored Depot-Acme edge changes rank only.
        "rank-only edge": (
            {**kept, "/corpus/a.txt": [[(v, x, 3)]]},
            {**kept, "/corpus/a.txt": [[(b, v, 3)]]},
        ),
        # a.txt is deleted: the edge both state loses a.txt's text unit.
        "lineage edge": ({**kept, "/corpus/a.txt": [[(d, x, 2), (v, b, 1)]]}, kept),
        # a.txt is deleted: Acme loses an edge, so do its edges' ranks.
        "entity rank": (
            {
                "/corpus/b.txt": [[(d, x, 1), (d, v, 1)]],
                "/corpus/a.txt": [[(x, b, 2)]],
            },
            {"/corpus/b.txt": [[(d, x, 1), (d, v, 1)]]},
        ),
    }


@pytest.mark.parametrize(
    "where",
    [
        "vector.update_relationship_lineage",
        "vector.update_entity_lineage",
        "graph.upsert_relationships",
        "graph.update_relationship_ranks",
        "graph.upsert_entities",
    ],
)
@pytest.mark.parametrize("shape", ["rank-only edge", "lineage edge", "entity rank"])
@pytest.mark.parametrize("lang", list(_NAMES))
def test_a_strip_write_the_store_rejects_is_repaired_by_the_next_run(
    lang: str, shape: str, where: str
) -> None:
    # A strip phase whose write is rejected stops the strip: the later
    # phases would rewrite the graph entities a retry finds it by, and the
    # rank the rejected copy kept would stay (no full build has it).
    before, after = _strip_shapes(*_NAMES[lang])[shape]
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(before)

    harness.run(after, fail=lambda at, _: at == where)
    harness.run(after)

    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []


@pytest.mark.parametrize(
    "spellings",
    [
        ("ACME", "Acme Corp", "Acme"),
        ("한빛 주식회사", "한빛", "(주)한빛"),
        ("(株)ひかり", "ひかり", "ひかり株式会社"),
    ],
    ids=["en", "ko", "ja"],
)
def test_a_strip_killed_before_its_vector_entity_writes_is_repaired(
    spellings: tuple[str, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deleting d.txt changes Depot's rank. The graph copies of the kept
    # entities are written after their vector copies: a run killed in
    # between leaves them citing d.txt's units, so the retry derives Depot
    # again and sets the vector copy's rank.
    upper, corp, inc = spellings
    kept: Corpus = {
        "/corpus/c.txt": [
            [("Vendor", "Depot", 2)],
            [(upper, "Depot", 3), (corp, "Depot", 2), (inc, "Depot", 3)],
        ]
    }
    before: Corpus = {
        **kept,
        "/corpus/d.txt": [
            [(upper, "Buyer", 3), ("Vendor", inc, 2), (corp, "Buyer", 3)]
        ],
    }
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(before)

    def killed(*_args: object, **_kwargs: object) -> None:
        raise Killed("update_entity_lineage")

    with monkeypatch.context() as patched:
        patched.setattr(harness.vector, "update_entity_lineage", killed)
        harness.run(kept)
    harness.run(kept)

    assert harness.state() == expected_state(kept)
    assert harness.registry_problems(kept) == []


def test_a_strip_sets_the_rank_of_a_hubs_edges_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Removing a.txt drops one of Hub's edges, so every other stored edge at
    # Hub changes rank only: the strip sets it in place in the graph too,
    # as the commit adding a.txt did, instead of writing the edges whole.
    spokes: Corpus = {"/corpus/hub.txt": [[("Hub", f"Spoke{i}", 1) for i in range(6)]]}
    with_a = {**spokes, "/corpus/a.txt": [[("Hub", "Depot", 2)]]}
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(with_a)
    whole: list[str] = []
    upsert = harness.graph.upsert_relationships

    def recording(items: list[Any]) -> Any:
        whole.extend(r.id for r in items)
        return upsert(items)

    monkeypatch.setattr(harness.graph, "upsert_relationships", recording)
    harness.graph.rank_updates.clear()

    harness.run(spokes)

    hub_edges = {r.id for r in extract(spokes).relationships}
    assert set(harness.graph.rank_updates) == hub_edges
    assert not hub_edges & set(whole)
    assert harness.state() == expected_state(spokes)
    assert harness.registry_problems(spokes) == []


def test_a_rank_only_edges_vector_copy_is_read_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A new spoke changes the rank of every stored edge at the hub. The
    # derivation reads their vector copies to compare the rank; the write
    # compares the same copies instead of reading them again.
    spokes: Corpus = {
        f"/corpus/s{i}.txt": [[("Hub", f"Spoke{i}", 1)]] for i in range(5)
    }
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(spokes)
    reads: list[str] = []
    read = harness.vector.read_vector_copies

    def recording(kind: str, ids: list[str], suffix: str | None = None) -> Any:
        if kind == "relationships":
            reads.extend(ids)
        return read(kind, ids, suffix=suffix)

    monkeypatch.setattr(harness.vector, "read_vector_copies", recording)
    after = {**spokes, "/corpus/new.txt": [[("Hub", "Depot", 1)]]}

    harness.run(after)

    stored = {r.id for r in extract(spokes).relationships}
    assert sorted(reads) == sorted(stored)
    assert set(harness.graph.rank_updates) == stored
    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []


@pytest.mark.parametrize("lang", list(_NAMES))
def test_a_vector_copy_a_rejected_strip_left_behind_is_not_set_rank_only(
    lang: str,
) -> None:
    # Deleting d.txt strips its line from Vendor -> Depot, which a.txt
    # states too; c.txt, added in the same run, gives Depot an edge, so the
    # commit rewrites Vendor -> Depot for its rank alone. The strip's vector
    # write of the edge is rejected: its vector copy keeps d.txt's line, and
    # setting its lineage and rank in place kept the line for good (nothing
    # cites d.txt's unit any more). The commit writes it whole instead.
    v, d, x, b = _NAMES[lang]
    before: Corpus = {
        "/corpus/a.txt": [[(v, d, 2)]],
        "/corpus/d.txt": [[(v, d, 3)]],
    }
    after: Corpus = {
        "/corpus/a.txt": before["/corpus/a.txt"],
        "/corpus/c.txt": [[(d, x, 1)]],
    }
    edge = _edge_id(before, v, d)
    harness = IncrementalRun(FakeDocStatusStore())
    harness.run(before)

    harness.run(
        after, fail=lambda at, i: at == "vector.upsert_relationships" and i == edge
    )
    harness.run(after)
    harness.run(after)

    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []
