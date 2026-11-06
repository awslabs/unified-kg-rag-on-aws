# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summary tree's ``VectorIndexer`` contract on every vector backend
that keeps one (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in): the embedded SQLite engine and the
OpenSearch container (``LOCAL_STORES=1``); the fake keeps none and says so.

Leaf paging is complete, ordered and leaves the seed graphs' units out; a
tree is written whole and swapped in at once, with its record; a failed
write keeps the old tree; summaries citing removed text go with every node
above them; stored summary vectors are reused; the build record gains the
summary tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.store_backends import (
    INDEX_ADMIN,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.domain.ingestion.cache_keys import extraction_settings
from unified_kg_rag.domain.ingestion.seed_graph import seed_unit_id
from unified_kg_rag.domain.models import (
    IndexArtifact,
    IndexBuildRecord,
    Methodology,
    SummaryNode,
    TextUnit,
)
from unified_kg_rag.ports.indexer import VectorIndexer

pytestmark = pytest.mark.integration


@pytest.fixture(params=backends(INDEX_ADMIN))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _units(n: int, suffix: str | None = None) -> list[TextUnit]:
    attributes = {"file_name": "doc.txt", "filters": {"category": "logistics"}}
    if suffix:
        attributes["index"] = suffix
    return [
        TextUnit(
            id=f"u{i:03d}",
            text=f"Depot {i} stores widgets for Vendor",
            n_tokens=7,
            attributes=dict(attributes),
        )
        for i in range(n)
    ]


def _node(
    node_id: str, level: int, children: list[str], units: list[str], **kw: Any
) -> SummaryNode:
    return SummaryNode(
        id=node_id,
        level=level,
        text=kw.pop("text", f"Summary {node_id} of the depots"),
        child_ids=children,
        text_unit_ids=units,
        n_leaves=len(units),
        n_tokens=5,
        language="English",
        input_key=f"key-{node_id}",
        attributes={"filters": {"category": "logistics"}},
        **kw,
    )


def _tree(harness: StoreHarness, nodes: list[SummaryNode], suffix: str | None = None):
    indexer = harness.vector_indexer
    vectors = indexer.embed_summary_texts([n.text for n in nodes], suffix)
    assert vectors is not None and all(v is not None for v in vectors)
    record = {"pipeline_id": "run-1", "leaf_digest": "d", "fingerprint": "f"}
    stats = indexer.write_summary_tree(
        nodes, {n.id: v for n, v in zip(nodes, vectors, strict=True)}, suffix, record
    )
    assert stats.failed_items == 0, stats.errors
    return stats


def test_text_units_are_paged_in_id_order_without_seed_units(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    seed = TextUnit(id=seed_unit_id("seed-doc", 1), text="seed statement")
    stats = indexer.index_text_units([*_units(25), seed])
    assert stats.failed_items == 0, stats.errors

    rows = indexer.scan_text_units(
        None, fields=["text", "text_embedding", "attributes"], page_size=7
    )

    assert [r["id"] for r in rows] == [f"u{i:03d}" for i in range(25)]
    assert all(
        len(r["text_embedding"]) == harness.embedding_factory.dimensions for r in rows
    )
    assert rows[0]["attributes"]["filters"] == {"category": "logistics"}
    # Another suffix has none.
    assert indexer.scan_text_units(harness.suffix("empty"), fields=["text"]) == []


def test_a_tree_is_written_whole_with_its_record_and_swapped_in(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    indexer.index_text_units(_units(4))
    first = [_node("n1", 1, ["u000", "u001"], ["u000", "u001"])]
    _tree(harness, first)
    assert indexer.read_summary_tree_record(None)["pipeline_id"] == "run-1"
    assert [n["id"] for n in indexer.read_summary_nodes(None, fields=["level"])] == [
        "n1"
    ]

    second = [
        _node("n2", 1, ["u000", "u001"], ["u000", "u001"]),
        _node("n3", 1, ["u002", "u003"], ["u002", "u003"]),
        _node("n4", 2, ["n2", "n3"], ["u000", "u001", "u002", "u003"]),
    ]
    _tree(harness, second)

    nodes = indexer.read_summary_nodes(
        None, fields=["level", "child_ids", "text_unit_ids", "attributes"]
    )
    assert [n["id"] for n in nodes] == ["n2", "n3", "n4"]
    assert nodes[2]["child_ids"] == ["n2", "n3"]
    assert nodes[0]["attributes"]["filters"] == {"category": "logistics"}
    # One index behind the alias: the old tree's was deleted.
    client = indexer.opensearch_client
    alias = indexer._get_name(
        indexer.opensearch_config.summary_nodes_index_prefix, None
    )
    assert len(client.get_aliases_by_index(f"{alias}-*")) == 1


def test_a_failed_write_keeps_the_old_tree(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = harness.vector_indexer
    _tree(harness, [_node("n1", 1, ["u000", "u001"], ["u000", "u001"])])

    def refuse(index_name: str, documents: list[dict[str, Any]], refresh: bool = True):
        raise RuntimeError("bulk refused")

    monkeypatch.setattr(indexer.opensearch_client, "bulk_index", refuse)
    vectors = {"n9": [0.1] * harness.embedding_factory.dimensions}
    stats = indexer.write_summary_tree(
        [_node("n9", 1, ["u000", "u002"], ["u000", "u002"])], vectors, None, {}
    )
    monkeypatch.undo()

    assert stats.failed_items == 1
    assert [n["id"] for n in indexer.read_summary_nodes(None, fields=[])] == ["n1"]
    client = indexer.opensearch_client
    alias = indexer._get_name(
        indexer.opensearch_config.summary_nodes_index_prefix, None
    )
    assert len(client.get_aliases_by_index(f"{alias}-*")) == 1


def test_removed_text_takes_its_summaries_and_every_node_above(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("other")
    nodes = [
        _node("n1", 1, ["u000", "u001"], ["u000", "u001"]),
        _node("n2", 1, ["u002", "u003"], ["u002", "u003"]),
        _node("n3", 1, ["u004", "u005"], ["u004", "u005"]),
        _node("n4", 2, ["n1", "n2"], ["u000", "u001", "u002", "u003"]),
        # A capped lineage: found through its child n4.
        _node("n5", 3, ["n4", "n3"], ["u000"], lineage_complete=False),
    ]
    _tree(harness, nodes)
    _tree(harness, [_node("n1", 1, ["u000", "u001"], ["u000", "u001"])], tenant)

    stats = indexer.delete_summary_nodes_citing(["u003"], "default")

    assert stats.failed_items == 0, stats.errors
    left = {n["id"] for n in indexer.read_summary_nodes(None, fields=[])}
    assert left == {"n1", "n3"}
    # Another suffix's tree is untouched.
    assert [n["id"] for n in indexer.read_summary_nodes(tenant, fields=[])] == ["n1"]
    # Nothing cites a unit no node holds.
    assert indexer.delete_summary_nodes_citing(["zzz"], "default").failed_items == 0


def test_a_full_build_drops_the_summaries_of_text_it_did_not_write(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    _tree(
        harness,
        [
            _node("n1", 1, ["u000", "u001"], ["u000", "u001"]),
            _node("n2", 1, ["u002", "u003"], ["u002", "u003"]),
            _node("n3", 2, ["n1", "n2"], ["u000", "u001", "u002", "u003"]),
        ],
    )

    stats = indexer.delete_summary_nodes_outside(["u000", "u001", "u002"], "default")

    assert stats.failed_items == 0, stats.errors
    assert [n["id"] for n in indexer.read_summary_nodes(None, fields=[])] == ["n1"]


def test_stored_summary_vectors_are_reused(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    _tree(harness, [_node("n1", 1, ["u000", "u001"], ["u000", "u001"], text="kept")])
    calls: list[list[str]] = []
    original = indexer._batch_embed

    def counted(texts: list[str], *args: Any, **kwargs: Any):
        calls.append(list(texts))
        return original(texts, *args, **kwargs)

    indexer._batch_embed = counted
    vectors = indexer.embed_summary_texts(["kept", "new summary"], None)

    assert calls == [["new summary"]]
    assert vectors is not None and all(v is not None for v in vectors)


def test_the_build_record_gains_the_summary_tree(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    indexer.index_text_units(_units(2), suffixes=["default"])
    record = IndexBuildRecord(
        pipeline_id="run-1",
        mode="full",
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
        documents=1,
        extraction_fingerprint="fp",
        extraction_settings=extraction_settings(harness.config),
    ).with_artifacts([IndexArtifact.TEXT_UNITS])
    indexer.record_build(record, "default")

    assert indexer.add_build_artifacts([IndexArtifact.SUMMARY_TREE], "default")
    stored = indexer.read_build_record("default")
    assert stored["artifacts"] == ["text_units", "summary_tree"]
    assert stored["methodologies"] == [Methodology.RAPTOR.value]
    # Already named: nothing to change.
    assert not indexer.add_build_artifacts([IndexArtifact.SUMMARY_TREE], "default")


def test_a_backend_without_a_tree_says_so() -> None:
    # The port's defaults, as a custom backend inherits them.
    bare: Any = object()
    assert VectorIndexer.scan_text_units(bare, None, fields=["text"]) is None
    assert VectorIndexer.read_summary_nodes(bare, None, fields=[]) is None
    assert VectorIndexer.embed_summary_texts(bare, ["x"], None) is None
    assert VectorIndexer.write_summary_tree(bare, [], {}, None, {}).failed_items == 1
    assert VectorIndexer.delete_summary_nodes_citing(bare, ["u"], "s").total_items == 0
    assert not VectorIndexer.add_build_artifacts(bare, [], None)


def _manager(harness: StoreHarness) -> Any:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    return IndexingManager(
        harness.config,
        vector_indexer=harness.vector_indexer,
        graph_indexer=harness.graph_indexer,
    )


def test_a_removal_deletes_the_summaries_of_removed_text(
    harness: StoreHarness,
) -> None:
    manager = _manager(harness)
    manager.index_all_data(text_units=_units(4))
    _tree(
        harness,
        [
            _node("n1", 1, ["u000", "u001"], ["u000", "u001"]),
            _node("n2", 1, ["u002", "u003"], ["u002", "u003"]),
            _node("n3", 2, ["n1", "n2"], ["u000", "u001", "u002", "u003"]),
        ],
    )

    results = manager.remove_text_units("default", ["u000"])

    assert not any(s.failed_items for s in results.values()), results
    assert results["vector_summary_nodes_default"].successful_items == 2
    left = harness.vector_indexer.read_summary_nodes(None, fields=[])
    assert [n["id"] for n in left] == ["n2"]


def test_a_full_build_sweeps_the_summaries_of_text_it_no_longer_holds(
    harness: StoreHarness,
) -> None:
    from unified_kg_rag.domain.models import Methodology

    harness.config.indexing.methodologies = [Methodology.RAPTOR]
    manager = _manager(harness)
    manager.index_all_data(text_units=_units(4))
    _tree(
        harness,
        [
            _node("n1", 1, ["u000", "u001"], ["u000", "u001"]),
            _node("n2", 1, ["u002", "u003"], ["u002", "u003"]),
        ],
    )

    results = manager.index_all_data(text_units=_units(3))

    assert results["vector_summary_nodes_default"].successful_items == 1
    left = harness.vector_indexer.read_summary_nodes(None, fields=[])
    assert [n["id"] for n in left] == ["n1"]
    # Without a tree the build reports nothing about one.
    other = harness.suffix("plain")
    assert "vector_summary_nodes_" + other not in manager.index_all_data(
        text_units=_units(2, other)
    )


def test_a_full_build_without_raptor_deletes_the_tree(harness: StoreHarness) -> None:
    from unified_kg_rag.domain.models import Methodology

    harness.config.indexing.methodologies = [Methodology.RAPTOR]
    _manager(harness).index_all_data(text_units=_units(4))
    _tree(harness, [_node("n1", 1, ["u000", "u001"], ["u000", "u001"])])
    assert harness.vector_indexer.read_summary_tree_record(None) is not None

    harness.config.indexing.methodologies = [Methodology.LIGHTRAG]
    results = _manager(harness).index_all_data(text_units=_units(4))

    assert results["vector_summary_nodes_default"].successful_items == 1
    assert harness.vector_indexer.read_summary_nodes(None, fields=[]) == []
    assert harness.vector_indexer.read_summary_tree_record(None) is None


def test_the_tree_strategy_holds_the_collapsed_trees_top_five(
    harness: StoreHarness,
) -> None:
    # RAPTOR's collapsed tree (cosine over every node, top_k 10) is a special
    # case of tree's hybrid search: with vector search alone, no rerank, no
    # graph signal and no expansion, tree's top 10 holds the reference's
    # top 5 (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).
    import asyncio
    from unittest.mock import MagicMock

    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.adapters.search_strategies.tree_search import (
        TreeSearchStrategy,
    )
    from unified_kg_rag.domain.ingestion.summary_tree import (
        collapsed_tree_reference,
    )
    from unified_kg_rag.domain.models import Methodology, SearchQuery, SearchType

    topics = ["widgets", "gadgets", "steel", "money", "containers", "fire"]
    units = [
        TextUnit(
            id=f"u{i:03d}",
            text=f"Depot {i} stores {topics[i % 6]} for Vendor {i % 4}",
            n_tokens=7,
            attributes={"file_name": f"doc{i}.txt"},
        )
        for i in range(24)
    ]
    indexer = harness.vector_indexer
    assert indexer.index_text_units(units).failed_items == 0
    nodes = [
        _node(
            f"n{t}",
            1,
            [u.id for u in units if topics[int(u.id[1:]) % 6] == topic],
            [u.id for u in units if topics[int(u.id[1:]) % 6] == topic],
            text=f"Depots store {topic} for the vendors",
        )
        for t, topic in enumerate(topics)
    ]
    _tree(harness, nodes)
    config = harness.config.model_copy(deep=True)
    config.indexing.methodologies = [Methodology.RAPTOR]
    config.providers.rerank.kind = "none"
    config.search.tree_search.expand_children = 0
    config.search.tree_search.graph_signal = "off"
    question = "Which depot stores steel for the vendors?"

    class _Words:
        def count_tokens(self, text: str) -> int:
            return len(text.split())

    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(lambda system, human: ""),
        embedding_factory=harness.embedding_factory,
        token_counter_factory=lambda *_, **__: _Words(),
    )

    async def search() -> list[str]:
        retriever = harness.vector_retriever()
        try:
            strategy = TreeSearchStrategy(
                config, {"document": retriever}, providers=providers
            )
            result = await strategy.asearch(
                SearchQuery(query=question, search_type=SearchType.VECTOR, top_k=10)
            )
            return [str(r.source) for r in result.results][:10]
        finally:
            await retriever.aclose()

    ours = asyncio.run(search())
    model = harness.embedding_factory.get_model("m")
    query = model.embed_query(question)
    candidates = [(u.id, model.embed_query(u.text), u.n_tokens or 0) for u in units] + [
        (n.id, model.embed_query(n.text), n.n_tokens) for n in nodes
    ]
    reference = collapsed_tree_reference(query, candidates, top_k=10)
    assert set(reference[:5]) <= set(ours), (reference, ours)


@pytest.mark.parametrize(
    ("language", "analyzer", "text", "query"),
    [
        ("ko", "nori", "창고는 벤더를 위해 위젯을 보관한다", "창고"),
        ("ja", "cjk", "倉庫はベンダーのためにウィジェットを保管する", "倉庫"),
    ],
)
def test_a_summary_is_analyzed_in_the_index_language(
    harness: StoreHarness, language: str, analyzer: str, text: str, query: str
) -> None:
    from unified_kg_rag.domain.models import RetrieverRole, SearchQuery, SearchType
    from unified_kg_rag.domain.models.config import LanguageCode

    def untranslated(config: Any) -> None:
        config.processing.translation.enabled = False
        config.processing.translation.source_language = LanguageCode(language)
        config.processing.translation.target_language = LanguageCode(language)

    local = harness.derive(untranslated)
    indexer = local.vector_indexer
    mapping = indexer._get_summary_nodes_mapping()["mappings"]["properties"]
    assert mapping["text"]["analyzer"] == analyzer
    _tree(local, [_node("n1", 1, ["u000", "u001"], ["u000", "u001"], text=text)])

    (hits,) = local.retrieve(
        RetrieverRole.DOCUMENT,
        SearchQuery(
            query=query,
            search_type=SearchType.LEXICAL,
            index_prefixes=[
                local.config.indexing.opensearch.summary_nodes_index_prefix
            ],
        ),
    )
    assert [h.source for h in hits] == ["n1"]
    assert hits[0].retriever_type == "summary"


def test_expanded_passages_keep_the_callers_filters(harness: StoreHarness) -> None:
    # The summary passes the caller's filter; of its passages only those
    # that pass it too are brought, even where others match the query
    # better.
    import asyncio
    from unittest.mock import MagicMock

    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.adapters.search_strategies.tree_search import (
        TreeSearchStrategy,
    )
    from unified_kg_rag.domain.models import Methodology, SearchQuery

    units = [
        TextUnit(
            id=f"u{i:03d}",
            text=(
                "Depot stores widgets for Vendor every week"
                if i % 2
                else f"Ledger entry {i}"
            ),
            n_tokens=50 if i % 2 else 5,
        )
        for i in range(12)
    ]
    indexer = harness.vector_indexer
    assert indexer.index_text_units(units).failed_items == 0
    ids = [u.id for u in units]
    _tree(harness, [_node("n1", 1, ids, ids, text="Depot stores widgets")])
    config = harness.config.model_copy(deep=True)
    config.indexing.methodologies = [Methodology.RAPTOR]
    config.providers.rerank.kind = "none"
    config.search.tree_search.graph_signal = "off"
    config.search.tree_search.text_top_k = 1
    config.search.tree_search.expand_children = 3

    class _Words:
        def count_tokens(self, text: str) -> int:
            return len(text.split())

    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(lambda system, human: ""),
        embedding_factory=harness.embedding_factory,
        token_counter_factory=lambda *_, **__: _Words(),
    )

    async def search() -> Any:
        retriever = harness.vector_retriever()
        try:
            strategy = TreeSearchStrategy(
                config, {"document": retriever}, providers=providers
            )
            return await strategy.asearch(
                SearchQuery(
                    query="Which depot stores widgets for Vendor?",
                    filters={"n_tokens": {"lte": 10}},
                )
            )
        finally:
            await retriever.aclose()

    result = asyncio.run(search())
    expanded = [r for r in result.results if (r.metadata or {}).get("expanded_from")]
    assert expanded
    assert all(r.metadata["n_tokens"] == 5 for r in expanded)


def test_a_cut_lineage_is_read_from_the_descendants(harness: StoreHarness) -> None:
    # A node over more text units than max_lineage_text_units stores only
    # the first ones (lineage_complete false); its passages are found
    # through its children, down to the nodes whose lineage is whole.
    import asyncio
    from unittest.mock import MagicMock

    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from tests.fixtures.fakes.tokens import WordTokenCounter
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.adapters.search_strategies.tree_search import (
        TreeSearchStrategy,
    )
    from unified_kg_rag.domain.models import Methodology, SearchQuery

    units = [
        TextUnit(
            id=f"u{i:03d}",
            text=(
                "Depot stores widgets for Vendor every week"
                if i >= 10
                else f"Ledger entry {i}"
            ),
            n_tokens=7 if i >= 10 else 3,
        )
        for i in range(12)
    ]
    indexer = harness.vector_indexer
    assert indexer.index_text_units(units).failed_items == 0
    ids = [u.id for u in units]
    _tree(
        harness,
        [
            _node("na", 1, ids[:6], ids[:6], text="Ledger entries"),
            # Cut: the stored lineage holds none of the widget passages.
            _node(
                "nb",
                1,
                ids[6:],
                ids[6:8],
                text="Ledger totals",
                lineage_complete=False,
            ),
            _node(
                "top",
                2,
                ["na", "nb"],
                ids[:2],
                text="Depot stores widgets for Vendor",
                lineage_complete=False,
            ),
        ],
    )
    config = harness.config.model_copy(deep=True)
    config.indexing.methodologies = [Methodology.RAPTOR]
    config.providers.rerank.kind = "none"
    config.search.tree_search.graph_signal = "off"
    config.search.tree_search.text_top_k = 1
    config.search.tree_search.summary_top_k = 1
    config.search.tree_search.expand_children = 3
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(lambda system, human: ""),
        embedding_factory=harness.embedding_factory,
        token_counter_factory=lambda *_, **__: WordTokenCounter(),
    )

    async def search() -> Any:
        retriever = harness.vector_retriever()
        try:
            strategy = TreeSearchStrategy(
                config, {"document": retriever}, providers=providers
            )
            return await strategy.asearch(
                SearchQuery(query="Which depot stores widgets for Vendor?")
            )
        finally:
            await retriever.aclose()

    result = asyncio.run(search())
    expanded = {
        str(r.source): r.metadata["expanded_from"]
        for r in result.results
        if (r.metadata or {}).get("expanded_from")
    }
    # The one summary found is the top; only its cut child's children lead
    # to the widget passage the text search did not return.
    assert [r.source for r in result.results if r.retriever_type == "summary"] == [
        "top"
    ]
    assert expanded.keys() & {"u010", "u011"}, result.results
