# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The embedded SQLite graph store (``providers.graph.kind: sqlite``).

The indexer must keep Neptune's write semantics (``FakeGraphStore`` is the
reference: the comparison test replays one write sequence against both) and
the retriever must walk the graph as ``NeptuneRetriever``'s Gremlin
traversal does.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphStore
from unified_kg_rag.adapters.provider_registry import ProviderRole, provider_endpoint
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import (
    SQLiteGraphRetriever,
)
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
    SearchQuery,
)
from unified_kg_rag.shared.exceptions import StoreConfigurationError, StoreTimeoutError

pytestmark = pytest.mark.unit

LABEL = "Entity-default"


def _config(tmp_path: Path, **neptune: Any) -> Config:
    return Config.model_validate(
        {
            "providers": {
                "graph": {
                    "kind": "sqlite",
                    "sqlite_path": str(tmp_path / "graph.sqlite3"),
                }
            },
            "indexing": {"neptune": neptune},
            "search": {"store_timeout_seconds": 0.2},
        }
    )


def _entity(entity_id: str, suffix: str | None = None, **fields: Any) -> Entity:
    attributes = dict(fields.pop("attributes", None) or {})
    if suffix:
        attributes["index"] = suffix
    fields.setdefault("name", entity_id.upper())
    return Entity(id=entity_id, attributes=attributes or None, **fields)


def _rel(
    rel_id: str, source: str, target: str, suffix: str | None = None, **fields: Any
) -> Relationship:
    attributes = dict(fields.pop("attributes", None) or {})
    if suffix:
        attributes["index"] = suffix
    return Relationship(
        id=rel_id,
        source_id=source,
        target_id=target,
        attributes=attributes or None,
        **fields,
    )


def _community(comm_id: str, members: list[str] | None, **fields: Any) -> Community:
    fields.setdefault("level", "0")
    fields.setdefault("parent", "-1")
    fields.setdefault("children", [])
    return Community(id=comm_id, name=comm_id, entity_ids=members, **fields)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return _config(tmp_path)


@pytest.fixture
def indexer(config: Config) -> Any:
    indexer = SQLiteGraphIndexer(config)
    indexer.initialize()
    yield indexer
    indexer.close()


def _graph(indexer: SQLiteGraphIndexer, edges: list[tuple[str, str]]) -> None:
    ids = sorted({end for edge in edges for end in edge})
    indexer.upsert_entities([_entity(i) for i in ids])
    stats = indexer.upsert_relationships([_rel(f"{a}-{b}", a, b) for a, b in edges])
    assert stats.failed_items == 0


# --- codec and write semantics ---------------------------------------------


def test_entity_and_relationship_round_trip(indexer: SQLiteGraphIndexer) -> None:
    entity = _entity(
        "e1",
        type="ORG",
        description="desc",
        rank=3,
        confidence=0.5,
        text_unit_ids=["t2", "t1"],
        community_ids=["c1"],
        attributes={"role": "vendor", "meta": {"k": 1}, "tags": ["x", "y"]},
    )
    indexer.upsert_entities([entity, _entity("e2")])
    rel = _rel(
        "r1",
        "e1",
        "e2",
        type="SUPPLIES",
        source_name="E1",
        target_name="E2",
        weight=2.0,
        description="supplies",
        rank=4,
        text_unit_ids=["t1", "t1", "t3"],
        attributes={"meta": {"k": 2}},
    )
    assert indexer.upsert_relationships([rel]).successful_items == 1

    (read,) = indexer.read_entities(["e1"])
    assert read.model_dump(exclude={"text_unit_ids"}) == entity.model_dump(
        exclude={"text_unit_ids"}
    )
    assert sorted(read.text_unit_ids or []) == ["t1", "t2"]
    (read_rel,) = indexer.read_relationships(["r1"])
    # Edge lists are JSON strings: order and duplicates survive.
    assert read_rel == rel


def test_property_max_length_truncates_strings(tmp_path: Path) -> None:
    indexer = SQLiteGraphIndexer(_config(tmp_path, property_max_length=4))
    indexer.upsert_entities([_entity("e1", description="abcdefgh")])
    (read,) = indexer.read_entities(["e1"])
    assert read.description == "abcd"
    assert read.name == "E1"[:4]


def test_upsert_replaces_the_properties_and_lists(
    indexer: SQLiteGraphIndexer,
) -> None:
    indexer.upsert_entities(
        [_entity("e1", description="kept", text_unit_ids=["a", "a", "b"])]
    )
    (read,) = indexer.read_entities(["e1"])
    assert sorted(read.text_unit_ids or []) == ["a", "b"]
    indexer.upsert_entities([_entity("e1", text_unit_ids=["c"])])
    (read,) = indexer.read_entities(["e1"])
    assert read.description is None
    assert read.text_unit_ids == ["c"]


def test_upsert_with_an_empty_list_removes_the_property(
    indexer: SQLiteGraphIndexer,
) -> None:
    indexer.upsert_entities([_entity("e1", text_unit_ids=["a"])])
    indexer.upsert_entities([_entity("e1", text_unit_ids=[])])
    (read,) = indexer.read_entities(["e1"])
    assert not read.text_unit_ids


def test_edge_with_missing_endpoint_fails_by_id(indexer: SQLiteGraphIndexer) -> None:
    indexer.upsert_entities([_entity("a")])
    stats = indexer.upsert_relationships(
        [_rel("r1", "a", "ghost"), _rel("r2", "a", "a")]
    )
    assert stats.failed_ids == ["r1"]
    assert stats.successful_items == 1
    assert [r.id for r in indexer.read_relationships(["r1", "r2"])] == ["r2"]


def test_readding_an_edge_replaces_it(indexer: SQLiteGraphIndexer) -> None:
    indexer.upsert_entities([_entity("a"), _entity("b")])
    indexer.index_relationships([_rel("r1", "a", "b", description="old")])
    indexer.upsert_relationships([_rel("r1", "b", "a", description="new")])
    (rel,) = indexer.read_relationships(["r1"])
    assert (rel.source_id, rel.target_id, rel.description) == ("b", "a", "new")
    assert indexer.find_incident_relationship_ids(["a"]) == ["r1"]


def test_index_entities_replaces_the_suffix_label(
    indexer: SQLiteGraphIndexer,
) -> None:
    _graph(indexer, [("a", "b")])
    indexer.upsert_entities([_entity("x", "other")])
    indexer.index_entities([_entity("c")])
    assert indexer.read_entity_names() == [("c", "C")]
    # The cleared vertices took their edges with them.
    assert indexer.read_relationships(["a-b"]) == []
    # An empty rebuild of a suffix named in ``suffixes`` empties it.
    indexer.index_entities([], suffixes=["other"])
    assert indexer.read_entity_names("other") == []
    assert indexer.read_entity_names() == [("c", "C")]


def test_suffixes_and_additional_suffix_isolate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tenant = SQLiteGraphIndexer(config)
    config_b = config.model_copy(deep=True)
    config_b.indexing.additional_suffix = "v2"
    versioned = SQLiteGraphIndexer(config_b)
    tenant.upsert_entities([_entity("e1", description="t"), _entity("e1", "s1")])
    versioned.upsert_entities([_entity("e1", description="v")])
    assert [e.description for e in tenant.read_entities(["e1"])] == ["t"]
    assert [e.description for e in versioned.read_entities(["e1"])] == ["v"]
    assert [e.id for e in tenant.read_entities(["e1"], suffix="s1")] == ["e1"]
    assert tenant.read_entities(["e1"], suffix="s2") == []


def _vertex_ids(indexer: SQLiteGraphIndexer, label: str) -> list[str]:
    with indexer.store.read("test") as connection:
        rows = connection.execute(
            "SELECT id FROM vertices WHERE label = ? ORDER BY id", (label,)
        ).fetchall()
    return [row[0] for row in rows]


def test_communities_index_clears_and_upsert_merges(
    indexer: SQLiteGraphIndexer,
) -> None:
    indexer.upsert_entities([_entity("a"), _entity("b")])
    indexer.index_communities(
        [_community("c1", ["a", "ghost"]), _community("c2", ["b"])]
    )
    # The upsert never clears: c2 stays.
    indexer.upsert_communities([_community("c1", ["b"]), _community("c3", [])])
    assert _vertex_ids(indexer, "Community-default") == ["c1", "c2", "c3"]

    indexer.index_communities([_community("c9", ["a"])])
    assert _vertex_ids(indexer, "Community-default") == ["c9"]
    indexer.index_communities([], suffixes=["default"])
    assert _vertex_ids(indexer, "Community-default") == []


def test_delete_by_id_drops_edges(indexer: SQLiteGraphIndexer) -> None:
    _graph(indexer, [("a", "b"), ("b", "c")])
    indexer.upsert_communities([_community("c1", ["a", "b"])])
    stats = indexer.delete_by_id(["b"], suffix="default")
    assert stats.successful_items == 1
    assert indexer.read_entity_names() == [("a", "A"), ("c", "C")]
    assert indexer.find_incident_relationship_ids(["a", "c"]) == []
    # Edges by id, and community vertices.
    indexer.upsert_entities([_entity("b")])
    indexer.upsert_relationships([_rel("r", "a", "b")])
    indexer.delete_by_id(["r", "c1"], suffix="default")
    assert indexer.read_relationships(["r"]) == []
    assert _vertex_ids(indexer, "Community-default") == []


def test_delete_by_id_stays_in_one_suffix(indexer: SQLiteGraphIndexer) -> None:
    indexer.upsert_entities([_entity("e1"), _entity("e1", "s1"), _entity("e2", "s1")])
    indexer.upsert_relationships([_rel("r1", "e1", "e2", "s1")])
    indexer.delete_by_id(["e1"], suffix="s1")
    assert [e.id for e in indexer.read_entities(["e1"])] == ["e1"]
    assert indexer.read_entities(["e1"], suffix="s1") == []
    assert indexer.read_relationships(["r1"], suffix="s1") == []

    # Without a suffix: the default suffix's only. It used to delete the id
    # under every label, another tenant's vertex included.
    indexer.upsert_entities([_entity("e1", "s1")])
    indexer.delete_by_id(["e1"])
    assert indexer.read_entities(["e1"]) == []
    assert [e.id for e in indexer.read_entities(["e1"], suffix="s1")] == ["e1"]
    assert [e.id for e in indexer.read_entities(["e2"], suffix="s1")] == ["e2"]


def test_delete_by_id_deletes_edges_by_id(
    indexer: SQLiteGraphIndexer,
) -> None:
    _graph(indexer, [("a", "b")])
    indexer.delete_by_id(["a-b"])
    assert indexer.read_relationships(["a-b"]) == []
    assert [e.id for e in indexer.read_entities(["a", "b"])] == ["a", "b"]


def test_neighbor_and_incident_reads(indexer: SQLiteGraphIndexer) -> None:
    _graph(indexer, [("a", "b"), ("c", "a"), ("a", "a")])
    indexer.upsert_entities([_entity("x", "s1"), _entity("a", "s1")])
    indexer.upsert_relationships([_rel("s-r", "a", "x", "s1")])
    assert indexer.read_neighbor_ids(["a", "ghost", "b"]) == {
        "a": ["a", "b", "c"],
        "b": ["a"],
    }
    assert indexer.read_neighbor_ids(["a"], suffix="s1") == {"a": ["x"]}
    assert indexer.find_incident_relationship_ids(["b"]) == ["a-b"]
    assert indexer.find_incident_relationship_ids(["a"], suffix="s1") == ["s-r"]
    # No suffix is the default suffix, not every suffix.
    assert indexer.find_incident_relationship_ids(["a"], suffix=None) == [
        "a-a",
        "a-b",
        "c-a",
    ]


def test_clear_drops_the_suffixes(indexer: SQLiteGraphIndexer) -> None:
    _graph(indexer, [("a", "b")])
    indexer.upsert_entities([_entity("k", "keep")])
    indexer.upsert_communities([_community("c1", ["a"])])
    indexer.clear([])
    indexer.clear(["default"])
    assert indexer.read_entity_names() == []
    assert _vertex_ids(indexer, "Community-default") == []
    assert indexer.read_entity_names("keep") == [("k", "K")]


def test_persists_after_reopen(config: Config) -> None:
    first = SQLiteGraphIndexer(config)
    _graph(first, [("a", "b")])
    first.close()
    second = SQLiteGraphIndexer(config)
    assert [r.id for r in second.read_relationships(["a-b"])] == ["a-b"]
    assert second.read_entity_names() == [("a", "A"), ("b", "B")]
    second.close()


class _Recording:
    """A connection that records the ``(sql, parameters)`` it executes."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.statements: list[tuple[str, Any]] = []

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        self.statements.append((sql, parameters))
        return self.connection.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)


def test_edge_reads_look_up_each_endpoint_in_an_index(
    indexer: SQLiteGraphIndexer,
) -> None:
    # A vertex's edges are read through an index on (label, endpoint): a
    # plan that walks every edge of the label made a graph expansion over
    # 150k edges take seconds.
    _graph(indexer, [(f"v{i}", f"v{(i * 7) % 50}") for i in range(50)])
    recording = _Recording(indexer.store.connection)
    indexer.store._connection = recording  # type: ignore[assignment]
    _traverse(indexer, ["v1"], max_hops=2, max_results_per_hop=5)
    indexer.read_neighbor_ids(["v1", "v2"])
    indexer.find_incident_relationship_ids(["v1", "v2"], suffix="default")
    indexer.store._connection = recording.connection
    edge_reads = [
        (sql, parameters)
        for sql, parameters in recording.statements
        if sql.startswith("SELECT") and "FROM edges" in sql
    ]
    assert len(edge_reads) >= 3
    for sql, parameters in edge_reads:
        plan = [
            row[3]
            for row in recording.connection.execute(
                f"EXPLAIN QUERY PLAN {sql}", parameters
            )
        ]
        scans = [step for step in plan if "edges" in step.split(" USING")[0]]
        assert scans, plan
        for step in scans:
            assert step.startswith("SEARCH") and (
                "source_id=?" in step or "target_id=?" in step
            ), (sql, plan)
        # The neighbours come off the index in order: no sort of the label.
        assert not any("TEMP B-TREE" in step for step in plan), (sql, plan)


# --- comparison with the reference fake --------------------------------------


def _replay(store: Any) -> list[Any]:
    """One write sequence, then every read the incremental path uses."""
    out: list[Any] = []
    store.index_entities(
        [
            _entity("a", description="A1", text_unit_ids=["t1", "t1"], rank=1),
            _entity("b", attributes={"role": "buyer", "meta": {"x": 1}}),
            _entity("c", "s1", description="other tenant"),
        ]
    )
    out.append(
        store.index_relationships(
            [
                _rel("ab", "a", "b", text_unit_ids=["t1"], weight=2.0),
                _rel("bz", "b", "z"),
                _rel("cc", "c", "c", "s1"),
            ]
        ).failed_ids
    )
    store.index_communities([_community("c1", ["a", "b"])])
    store.upsert_entities(
        [_entity("a", text_unit_ids=["t2"]), _entity("d", confidence=0.25)]
    )
    store.upsert_relationships(
        [_rel("ab", "a", "b", description="re-added"), _rel("ad", "a", "d")]
    )
    store.upsert_communities([_community("c2", ["d"])])
    store.delete_by_id(["d"], suffix="default")
    store.upsert_entities([_entity("e", "s1"), _entity("e")])
    store.delete_by_id(["e"])
    for suffix in (None, "s1"):
        out.append(store.read_entities(["a", "b", "c", "d", "e"], suffix=suffix))
        out.append(store.read_relationships(["ab", "ad", "bz", "cc"], suffix=suffix))
        out.append(sorted(store.read_entity_names(suffix)))
        out.append(store.read_neighbor_ids(["a", "b", "c", "d"], suffix=suffix))
        out.append(store.find_incident_relationship_ids(["a", "c"], suffix=suffix))
    store.index_entities([], suffixes=["s1"])
    out.append(store.read_entities(["c"], suffix="s1"))
    out.append(store.read_relationships(["cc"], suffix="s1"))
    return out


def test_matches_the_fake_graph_store(config: Config) -> None:
    fake = _replay(FakeGraphStore(config))
    real = _replay(SQLiteGraphIndexer(config))
    assert real == fake
    assert fake[0] == ["bz"]  # the sequence exercises a failed edge


# --- traversal ---------------------------------------------------------------


def _store(indexer: SQLiteGraphIndexer) -> SQLiteGraphStore:
    return indexer.store


def _traverse(
    indexer: SQLiteGraphIndexer, seeds: list[str], **kwargs: Any
) -> list[tuple[str, ...]]:
    options = {"max_hops": 3, "max_results_per_hop": 50, "per_seed_limit": 100}
    options.update(kwargs)
    rows = _store(indexer).traverse(LABEL, seeds, **options)
    return [tuple(p["name"][0].lower() for p in row["path"]) for row in rows]


def test_traversal_walks_breadth_first_from_each_seed(
    indexer: SQLiteGraphIndexer,
) -> None:
    _graph(indexer, [("s", "b"), ("s", "a"), ("a", "c"), ("c", "d"), ("d", "e")])
    assert _traverse(indexer, ["s", "ghost"]) == [
        ("s",),
        ("s", "a"),
        ("s", "b"),
        ("s", "a", "c"),
        ("s", "a", "c", "d"),
    ]
    assert _traverse(indexer, ["s"], max_hops=1) == [("s",), ("s", "a"), ("s", "b")]


def test_traversal_excludes_seeds_and_dedupes_per_seed(
    indexer: SQLiteGraphIndexer,
) -> None:
    _graph(indexer, [("s", "t"), ("s", "a"), ("t", "a"), ("a", "b"), ("t", "b")])
    assert _traverse(indexer, ["s", "t"]) == [
        ("s",),
        ("s", "a"),
        ("s", "a", "b"),
        ("t",),
        ("t", "a"),
        ("t", "b"),
    ]


def test_traversal_budget_is_split_per_seed(indexer: SQLiteGraphIndexer) -> None:
    edges = [("s", f"n{i}") for i in range(5)] + [("t", f"m{i}") for i in range(5)]
    _graph(indexer, edges)
    paths = _traverse(indexer, ["s", "t"], per_seed_limit=2)
    assert paths == [("s",), ("s", "n0"), ("s", "n1"), ("t",), ("t", "m0"), ("t", "m1")]


def test_max_results_per_hop_caps_each_vertex(indexer: SQLiteGraphIndexer) -> None:
    _graph(indexer, [("z", "a"), ("z", "b"), ("z", "c"), ("a", "x"), ("a", "y")])
    assert _traverse(indexer, ["z"], max_results_per_hop=1) == [
        ("z",),
        ("z", "a"),
        ("z", "a", "x"),
    ]
    # The cap is taken before the seeds are skipped, as Gremlin's
    # local(both().limit(n)) is: a's first neighbour by id is the seed here.
    assert _traverse(indexer, ["x"], max_results_per_hop=1) == [("x",), ("x", "a")]


def test_traversal_ignores_communities_and_other_suffixes(
    indexer: SQLiteGraphIndexer,
) -> None:
    _graph(indexer, [("s", "a")])
    indexer.upsert_entities([_entity("s", "s1"), _entity("z", "s1")])
    indexer.upsert_relationships([_rel("sz", "s", "z", "s1")])
    indexer.upsert_communities([_community("c1", ["s", "a"])])
    assert _traverse(indexer, ["s"]) == [("s",), ("s", "a")]


def test_degree_counts_the_relationship_edges(indexer: SQLiteGraphIndexer) -> None:
    # Regression: the degree counted memberships too (two for "s").
    _graph(indexer, [("s", "a"), ("s", "b")])
    indexer.upsert_communities([_community("c1", ["s"]), _community("c2", ["s"])])
    rows = _store(indexer).traverse(
        LABEL,
        ["s"],
        max_hops=1,
        max_results_per_hop=50,
        per_seed_limit=10,
        with_degree=True,
    )
    assert {row["node"]["id"][0]: row["degree"] for row in rows} == {
        "s": 2,
        "a": 1,
        "b": 1,
    }
    assert rows[0]["node_type"] == LABEL
    assert set(rows[0]["node"]) == {"id", "name", "rank"}


async def test_retriever_ranks_by_distance(config: Config) -> None:
    indexer = SQLiteGraphIndexer(config)
    _graph(indexer, [("s", "a"), ("a", "b")])
    retriever = SQLiteGraphRetriever(config)
    query = SearchQuery(query="q", filters={"id": ["s"]})
    results = await retriever.aretrieve(query)
    assert [r.source for r in results] == ["s", "a", "b"]
    assert results[0].score > results[1].score > results[2].score
    assert results[2].content.endswith("Path: S -> A -> B")
    # Deterministic: the same graph answers the same way.
    assert [r.model_dump() for r in await retriever.aretrieve(query)] == [
        r.model_dump() for r in results
    ]
    await retriever.aclose()
    indexer.close()


async def test_retriever_fetch_limit_is_split_across_seeds(config: Config) -> None:
    indexer = SQLiteGraphIndexer(config)
    edges = [("s", f"n{i}") for i in range(9)] + [("t", f"m{i}") for i in range(9)]
    _graph(indexer, edges)
    retriever = SQLiteGraphRetriever(config)
    # fetch_limit = 1 * 1 * 3 (traversal_fetch_multiplier): ceil(3 / 2) = 2
    # per seed; the cut then keeps the seeds plus the best 1.
    rows = await retriever._traverse(
        ["s", "t"], SearchQuery(query="q", top_k=1, filters={"id": ["s", "t"]})
    )
    assert sorted(row["node"]["id"][0] for row in rows) == [
        "m0",
        "m1",
        "n0",
        "n1",
        "s",
        "t",
    ]
    retriever.close()
    indexer.close()


async def test_retriever_filters(config: Config) -> None:
    indexer = SQLiteGraphIndexer(config)
    indexer.upsert_entities(
        [
            _entity("s", type="ORG", rank=5),
            _entity("a", type="ORG", rank=1, attributes={"role": "vendor"}),
            _entity("b", type="PERSON", rank=9, attributes={"role": "buyer"}),
            _entity("c", rank=3),
        ]
    )
    indexer.upsert_relationships(
        [_rel("sa", "s", "a"), _rel("sb", "s", "b"), _rel("sc", "s", "c")]
    )
    retriever = SQLiteGraphRetriever(config)

    async def ids(filters: dict[str, Any]) -> list[str]:
        query = SearchQuery(query="q", filters={"id": ["s"], **filters})
        return sorted(r.source for r in await retriever.aretrieve(query))

    # A plain key is strict: c has no type and goes too. Seeds are filtered.
    assert await ids({"type": "ORG"}) == ["a", "s"]
    assert await ids({"type": ["PERSON", "ORG"]}) == ["a", "b", "s"]
    assert await ids({"rank": {"gte": 3}}) == ["b", "c", "s"]
    assert await ids({"rank": {"gt": 1, "lt": 9}}) == ["c", "s"]
    # attr_* applies only where the property exists.
    assert await ids({"attr_role": "vendor"}) == ["a", "c", "s"]
    # A key no entity vertex declares is dropped, not matched.
    assert await ids({"level": "0"}) == ["a", "b", "c", "s"]
    # Comparing across types matches nothing rather than raising.
    assert await ids({"rank": {"gte": "x"}}) == []
    retriever.close()
    indexer.close()


async def test_retriever_without_seeds_or_with_unknown_suffix(config: Config) -> None:
    indexer = SQLiteGraphIndexer(config)
    _graph(indexer, [("s", "a")])
    retriever = SQLiteGraphRetriever(config)
    assert await retriever.aretrieve(SearchQuery(query="q")) == []
    query = SearchQuery(query="q", suffix="nope", filters={"id": ["s"]})
    assert await retriever.aretrieve(query) == []
    retriever.close()
    indexer.close()


def test_degree_binds_under_the_variable_limit(indexer: SQLiteGraphIndexer) -> None:
    # Older SQLite builds bind at most 999 values per statement.
    connection = indexer.store.connection
    if not hasattr(connection, "setlimit"):  # Connection.setlimit is Python 3.11+
        pytest.skip("sqlite3.Connection.setlimit needs Python 3.11")
    _graph(indexer, [("s", f"n{i:03d}") for i in range(450)])
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    rows = _store(indexer).traverse(
        LABEL,
        ["s"],
        max_hops=1,
        max_results_per_hop=500,
        per_seed_limit=500,
        with_degree=True,
    )
    assert len(rows) == 451
    assert rows[0]["degree"] == 450
    assert {row["degree"] for row in rows[1:]} == {1}


async def test_traversal_reads_one_snapshot(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process deletes a vertex the walk reached, between the walk's
    # neighbour query and its vertex read: the traversal answers from the
    # graph as it was when it started, not with nothing.
    indexer = SQLiteGraphIndexer(config)
    _graph(indexer, [("a", "b"), ("a", "c")])
    writer = SQLiteGraphIndexer(config)
    neighbors = SQLiteGraphStore.neighbors

    def write_between(connection: Any, label: str, vertex_id: str, limit: int):
        found = neighbors(connection, label, vertex_id, limit)
        if vertex_id == "a":
            writer.delete_by_id(["b"], suffix="default")
        return found

    monkeypatch.setattr(SQLiteGraphStore, "neighbors", staticmethod(write_between))
    retriever = SQLiteGraphRetriever(config)
    query = SearchQuery(query="q", filters={"id": ["a"]})
    assert [r.source for r in await retriever.aretrieve(query)] == ["a", "b", "c"]
    # The read ended its transaction: the next one sees the delete.
    assert [r.source for r in await retriever.aretrieve(query)] == ["a", "c"]
    assert not retriever._store.connection.in_transaction
    retriever.close()
    writer.close()
    indexer.close()


async def test_retriever_filters_only_the_reached_vertex(config: Config) -> None:
    # A filter keeps a vertex reached through one it drops (Gremlin filters
    # the emitted vertex, not the path).
    indexer = SQLiteGraphIndexer(config)
    indexer.upsert_entities(
        [
            _entity("s", type="ORG"),
            _entity("a", type="PERSON"),
            _entity("b", type="ORG"),
        ]
    )
    indexer.upsert_relationships([_rel("sa", "s", "a"), _rel("ab", "a", "b")])
    retriever = SQLiteGraphRetriever(config)
    query = SearchQuery(query="q", filters={"id": ["s"], "type": "ORG"})
    assert [r.source for r in await retriever.aretrieve(query)] == ["s", "b"]
    retriever.close()
    indexer.close()


async def test_retriever_ranks_by_degree(tmp_path: Path) -> None:
    config = _config(tmp_path, entity_importance_source="degree", max_hops=1)
    indexer = SQLiteGraphIndexer(config)
    indexer.upsert_entities(
        [_entity("s"), _entity("a", rank=9), _entity("b", rank=1)]
        + [_entity(i) for i in ("x", "y")]
    )
    indexer.upsert_relationships(
        [_rel("sa", "s", "a"), _rel("sb", "s", "b")]
        + [_rel("bx", "b", "x"), _rel("by", "b", "y")]
    )
    retriever = SQLiteGraphRetriever(config)
    results = await retriever.aretrieve(SearchQuery(query="q", filters={"id": ["s"]}))
    # b has three edges, a one: b first despite a's higher rank.
    assert [r.source for r in results] == ["s", "b", "a"]
    retriever.close()
    indexer.close()


# --- errors and registration ------------------------------------------------


def test_corrupt_database_is_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite3"
    path.write_bytes(b"not a sqlite database at all" * 100)
    indexer = SQLiteGraphIndexer(_config(tmp_path))
    with pytest.raises(StoreConfigurationError):
        indexer.initialize()
    with pytest.raises(StoreConfigurationError):
        indexer.read_entities(["a"])
    assert indexer.read_entity_names() == []  # best-effort, as in Neptune
    with pytest.raises(StoreConfigurationError):
        indexer.clear(["default"])


async def test_locked_database_is_a_timeout(config: Config) -> None:
    indexer = SQLiteGraphIndexer(config)
    _graph(indexer, [("s", "a")])
    holder = sqlite3.connect(config.providers.graph.sqlite_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(StoreTimeoutError):
            indexer.upsert_entities([_entity("b")])
    finally:
        holder.execute("ROLLBACK")
    indexer.close()
    # WAL readers never wait for a writer; an exclusive locking mode, taken
    # while no other connection is open, locks them out too.
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("DELETE FROM vertices WHERE id = 'none'")
    try:
        retriever = SQLiteGraphRetriever(config)
        # A fatal error fails the query instead of reading as "no results".
        with pytest.raises(StoreTimeoutError):
            await retriever.aretrieve(SearchQuery(query="q", filters={"id": ["s"]}))
        retriever.close()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert indexer.upsert_entities([_entity("b")]).successful_items == 1
    indexer.close()


def test_sqlite_kind_is_registered_without_an_endpoint(config: Config) -> None:
    providers = Providers(config)
    indexer = providers.build_graph_indexer()
    assert isinstance(indexer, SQLiteGraphIndexer)
    from unified_kg_rag.domain.models import RetrieverRole

    retriever = providers.build_retriever(RetrieverRole.GRAPH)
    assert isinstance(retriever, SQLiteGraphRetriever)
    for role in (ProviderRole.GRAPH_INDEXER, ProviderRole.GRAPH_RETRIEVER):
        assert provider_endpoint(role, "sqlite") is None
    indexer.close()
    retriever.close()


@pytest.mark.parametrize("store", ["sqlite", "fake"])
def test_a_rebuild_in_place_serves_the_old_graph_until_its_sweep(
    store: str, config: Config, indexer: Any
) -> None:
    graph = indexer if store == "sqlite" else FakeGraphStore(config)
    graph.index_entities([_entity("a"), _entity("b"), _entity("old")])
    graph.index_relationships([_rel("r-ab", "a", "b"), _rel("r-old", "a", "old")])
    graph.index_communities([_community("c-old", ["a", "old"], text_unit_ids=["t"])])

    assert graph.begin_rebuild(["default"])
    graph.index_entities([_entity("a", name="A2"), _entity("b"), _entity("new")])
    # Queries still find the previous graph while the build writes.
    assert {e.id for e in graph.read_entities(["a", "old"])} == {"a", "old"}
    assert [r.id for r in graph.read_relationships(["r-old"])] == ["r-old"]
    graph.index_relationships([_rel("r-ab", "a", "b"), _rel("r-new", "b", "new")])
    graph.index_communities([_community("c-new", ["a", "new"], text_unit_ids=["t"])])
    graph.end_rebuild(["default"])

    assert {e.id for e in graph.read_entities(["a", "b", "old", "new"])} == {
        "a",
        "b",
        "new",
    }
    assert graph.read_entities(["a"])[0].name == "A2"
    assert {r.id for r in graph.read_relationships(["r-ab", "r-old", "r-new"])} == {
        "r-ab",
        "r-new",
    }
    assert set(graph.find_ids_citing(["t"], communities=True)) == {"c-new"}


def test_a_failed_rebuild_keeps_the_previous_graph(indexer: Any) -> None:
    indexer.index_entities([_entity("a"), _entity("old")])
    indexer.begin_rebuild(["default"])
    indexer.index_entities([_entity("a")])
    indexer.end_rebuild(["default"], sweep=False)

    assert {e.id for e in indexer.read_entities(["a", "old"])} == {"a", "old"}
    # A full build after it clears the label first again.
    indexer.index_entities([_entity("a")])
    assert [e.id for e in indexer.read_entities(["a", "old"])] == ["a"]
