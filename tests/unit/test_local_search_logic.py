# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for LocalSearchStrategy pure helpers + orchestration branches.

The strategy is constructed via ``__new__`` so its AWS-client ``__init__`` (which
builds a ``HybridScorer`` / ``TokenManager`` and a boto3 session) never runs. The
``_get_ids`` helper lives on ``BaseSearchStrategy`` and is exercised here through
the concrete subclass. Retriever async methods are replaced with simple fakes /
AsyncMocks to drive the candidate-selection, graph-expansion and document-retrieval
branches (empty-retriever short-circuits, exception fallbacks) without real AWS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy
from unified_kg_rag.adapters.search_strategies.local_search import LocalSearchStrategy
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery, SearchType

pytestmark = pytest.mark.unit


def _result(
    source: str | None = None,
    *,
    score: float = 0.5,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        content="c",
        score=score,
        source=source,
        retriever_type="document",
        metadata=metadata or {},
    )


def _bare_strategy(
    *,
    retrievers: dict | None = None,
) -> LocalSearchStrategy:
    """A LocalSearchStrategy whose __init__ never runs (no AWS clients)."""
    strat = LocalSearchStrategy.__new__(LocalSearchStrategy)
    strat.config = SimpleNamespace(
        indexing=SimpleNamespace(
            opensearch=SimpleNamespace(
                entities_index_prefix="entities",
                text_units_index_prefix="text_units",
            ),
            neptune=SimpleNamespace(entity_label_prefix="Entity"),
        ),
        search=SimpleNamespace(
            local_search=SimpleNamespace(
                entity_frequency_threshold=20, seed_score_ratio=0.6
            ),
            max_concurrency=4,
        ),
    )
    strat.retrievers = retrievers or {}
    return strat


class _StubRetriever:
    """Async retriever stub returning a fixed list (or raising) from aretrieve."""

    def __init__(self, results=None, raises: Exception | None = None) -> None:
        self._results = results or []
        self._raises = raises
        self.last_query: SearchQuery | None = None

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery):
        self.last_query = query
        if self._raises is not None:
            raise self._raises
        return self._results


# --------------------------------------------------------------------------- #
# _get_ids (inherited from BaseSearchStrategy, exercised via the subclass)
# --------------------------------------------------------------------------- #


def test_get_ids_collects_scalar_and_list_metadata() -> None:
    results = [
        _result("a", metadata={"id": "x1"}),
        _result("b", metadata={"id": ["x2", "x3"]}),
    ]
    ids = set(LocalSearchStrategy._get_ids(results, "id"))
    assert ids == {"x1", "x2", "x3"}


def test_get_ids_keeps_rank_order() -> None:
    # A set lost the rank order (its order of str ids follows the hash seed).
    ids = [f"entity-{i:02d}" for i in range(30)]
    results = [_result(i, metadata={"id": i}) for i in reversed(ids)]
    results.append(_result("x", metadata={"id": [ids[5], ids[0]]}))
    assert LocalSearchStrategy._get_ids(results, "id") == ids[::-1]


def test_get_ids_falls_back_to_source_for_id_key() -> None:
    # When metadata lacks "id" but a source is present, source is used as the id.
    results = [_result("seed-1", metadata={})]
    assert LocalSearchStrategy._get_ids(results, "id") == ["seed-1"]


def test_get_ids_no_source_fallback_for_non_id_key() -> None:
    # The source fallback only applies to the "id" key, not arbitrary keys.
    results = [_result("seed-1", metadata={})]
    assert LocalSearchStrategy._get_ids(results, "text_unit_ids") == []


def test_get_ids_dedups_across_results() -> None:
    results = [
        _result("a", metadata={"id": "dup"}),
        _result("b", metadata={"id": "dup"}),
    ]
    assert LocalSearchStrategy._get_ids(results, "id") == ["dup"]


def test_get_ids_text_unit_ids_collection() -> None:
    results = [
        _result("a", metadata={"text_unit_ids": ["t1", "t2"]}),
        _result("b", metadata={"text_unit_ids": ["t2", "t3"]}),
    ]
    assert set(LocalSearchStrategy._get_ids(results, "text_unit_ids")) == {
        "t1",
        "t2",
        "t3",
    }


# --------------------------------------------------------------------------- #
# _filter_entities (frequency thresholding)
# --------------------------------------------------------------------------- #


def test_filter_entities_keeps_zero_and_within_threshold() -> None:
    nodes = [
        _result("zero", metadata={"text_unit_ids": []}),  # count 0 -> kept
        _result("low", metadata={"text_unit_ids": ["t1", "t2"]}),  # 2 <= 3 -> kept
    ]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert [n.source for n in kept] == ["zero", "low"]


def test_filter_entities_drops_above_threshold() -> None:
    nodes = [
        _result("hot", metadata={"text_unit_ids": ["t1", "t2", "t3", "t4"]}),  # 4 > 3
        _result("ok", metadata={"text_unit_ids": ["t1"]}),  # 1 <= 3
    ]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert [n.source for n in kept] == ["ok"]


def test_filter_entities_exempts_the_seeds() -> None:
    hot = {"text_unit_ids": ["t1", "t2", "t3", "t4"]}
    nodes = [
        _result("seed", metadata={"id": "seed", **hot}),
        _result("hub", metadata={"id": "hub", **hot}),
    ]
    kept = LocalSearchStrategy._filter_entities(
        nodes, frequency_threshold=3, seed_ids={"seed"}
    )
    assert [n.source for n in kept] == ["seed"]


def test_filter_entities_boundary_equal_to_threshold_kept() -> None:
    nodes = [_result("edge", metadata={"text_unit_ids": ["t1", "t2", "t3"]})]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert len(kept) == 1


def test_filter_entities_missing_metadata_treated_as_zero() -> None:
    # No text_unit_ids key -> count 0 -> kept (zero-frequency entities pass).
    nodes = [_result("nometa", metadata={})]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=1)
    assert len(kept) == 1


def test_filter_entities_counts_unwrapped_single_id_as_one() -> None:
    # NeptuneRetriever._clean_property_map unwraps a single-element value_map list
    # into a bare scalar, so an entity appearing in exactly ONE text unit arrives
    # with text_unit_ids as a str. len() on that counted 36 CHARACTERS and dropped
    # the entity — precisely the single-document bridge nodes multi-hop needs.
    uuid = "8c1f2a70-1111-2222-3333-444455556666"
    nodes = [_result("single", metadata={"text_unit_ids": uuid})]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=20)
    assert [n.source for n in kept] == ["single"]


# --------------------------------------------------------------------------- #
# _find_candidate_entities
# --------------------------------------------------------------------------- #


def _sources(results: list) -> list:
    return [r.source for r in results]


async def test_find_candidate_entities_no_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    query = SearchQuery(query="q", entity_focus=["Alice"])
    assert await strat._find_candidate_entities(query) == []


async def test_find_candidate_entities_no_entity_focus_falls_back_to_query() -> None:
    retriever = _StubRetriever([_result("e1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q", entity_focus=[], top_k=5)
    assert _sources(await strat._find_candidate_entities(query)) == ["e1"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.query == "q"
    assert sent.top_k == 5


async def test_find_candidate_entities_no_focus_and_empty_query_returns_empty() -> None:
    retriever = _StubRetriever([_result("e1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="", entity_focus=[])
    assert await strat._find_candidate_entities(query) == []
    assert retriever.last_query is None


async def test_find_candidate_entities_carries_caller_filters() -> None:
    retriever = _StubRetriever([_result("e1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q", entity_focus=["Vendor"], filters={"year": 2024})
    await strat._find_candidate_entities(query)
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters == {"year": 2024}
    assert sent.filters is not query.filters  # a copy, never the caller's dict


async def test_retrieve_documents_merges_caller_filters_with_id_scope() -> None:
    retriever = _StubRetriever([_result("t1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    caller = {"year": 2024, "id": ["t1", "other"]}
    await strat._retrieve_documents(["t2", "t1"], None, filters=caller)
    sent = retriever.last_query
    assert sent is not None
    # The scope keeps the ids the caller's filter allows.
    assert sent.filters == {"year": 2024, "id": ["t1"]}
    assert caller == {"year": 2024, "id": ["t1", "other"]}


class _ByQueryRetriever(_StubRetriever):
    """Stub whose hits depend on the query text; records every query."""

    def __init__(self, hits: dict[str, list[RetrievalResult]]) -> None:
        super().__init__()
        self._hits = hits
        self.queries: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery):
        self.queries.append(query)
        return self._hits.get(query.query, [])


def _entity(source: str | None, name: str, score: float) -> RetrievalResult:
    return _result(source, score=score, metadata={"name": name})


async def test_find_candidate_entities_looks_each_focus_entity_up_alone() -> None:
    retriever = _ByQueryRetriever(
        {
            "Alice": [_entity("e1", "Alice", 1.0), _entity(None, "Ghost", 0.9)],
            "Bob": [_entity("e2", "Bob", 1.0)],
        }
    )
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q", entity_focus=["Alice", "Bob", "Alice"])
    out = await strat._find_candidate_entities(query)
    # Drops the result with no source; one lookup per distinct focus entity,
    # never the joined names.
    assert _sources(out) == ["e1", "e2"]
    assert [q.query for q in retriever.queries] == ["Alice", "Bob"]
    assert {q.top_k for q in retriever.queries} == {
        LocalSearchStrategy.ENTITY_CANDIDATES_PER_FOCUS
    }
    assert {tuple(q.index_prefixes or ()) for q in retriever.queries} == {("entities",)}


async def test_find_candidate_entities_prefers_the_exact_name_match() -> None:
    # The similarity search ranks a neighbour above the entity the query
    # names: the exact match (by compact name key) seeds, the neighbour not.
    retriever = _ByQueryRetriever(
        {
            "acme corp": [
                _entity("e9", "Acme Holdings", 1.0),
                _entity("e1", "ACME Corp.", 0.9),
            ]
        }
    )
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q", entity_focus=["acme corp"])
    assert _sources(await strat._find_candidate_entities(query)) == ["e1"]


async def test_find_candidate_entities_drops_a_weak_top_hit() -> None:
    # "firm" names no entity: its top hit scores 0.5 against the exact
    # match's 1.0, under the 0.6 floor; "Bexley" (0.7) clears it.
    retriever = _ByQueryRetriever(
        {
            "Alvar": [_entity("e1", "Alvar", 1.0), _entity("e2", "Jarek", 0.5)],
            "firm": [_entity("e3", "Calvex", 0.5)],
            "Bexly": [_entity("e4", "Bexley", 0.7)],
        }
    )
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q", entity_focus=["Alvar", "firm", "Bexly"])
    assert _sources(await strat._find_candidate_entities(query)) == ["e1", "e4"]
    strat.config.search.local_search.seed_score_ratio = 0.0
    assert _sources(await strat._find_candidate_entities(query)) == [
        "e1",
        "e3",
        "e4",
    ]


async def test_retrieve_entity_chain_weights_the_seeds_by_score() -> None:
    strat = _bare_strategy()
    weights: list[dict[str, float] | None] = []

    async def candidates(query: SearchQuery) -> list[RetrievalResult]:
        return [_entity("e1", "Alvar", 1.0), _entity("e4", "Bexley", 0.7)]

    async def expand(query, seeds, *, seeds_pass=False, seed_weights=None):
        weights.append(seed_weights)
        return []

    strat._find_candidate_entities = candidates  # type: ignore[method-assign]
    strat._expand_via_graph = expand  # type: ignore[method-assign]
    strat._filter_entities = lambda nodes, **_: nodes  # type: ignore[method-assign]
    strat._per_type_quota = lambda top_k: {"text": 0}  # type: ignore[method-assign]

    async def none(*_args, **_kwargs):
        return {}

    strat._passing_ranked_units = none  # type: ignore[method-assign]
    strat._retrieve_documents = none  # type: ignore[method-assign]
    strat._retrieve_bridge_relationships = none  # type: ignore[method-assign]
    await strat._retrieve_entity_chain(SearchQuery(query="q", entity_focus=["Alvar"]))
    # Without an entity focus the raw query's hits split the walk evenly.
    await strat._retrieve_entity_chain(SearchQuery(query="q"))
    assert weights == [{"e1": 1.0, "e4": 0.7}, None]


async def test_find_candidate_entities_swallows_retriever_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("boom"))}
    )
    query = SearchQuery(query="q", entity_focus=["Alice"])
    assert await strat._find_candidate_entities(query) == []


# --------------------------------------------------------------------------- #
# _expand_via_graph
# --------------------------------------------------------------------------- #


async def test_expand_via_graph_no_graph_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, ["e1"]) == []


async def test_expand_via_graph_no_seeds_returns_empty() -> None:
    strat = _bare_strategy(retrievers={"graph": _StubRetriever([_result("x")])})
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, []) == []


async def test_expand_via_graph_sets_id_filter() -> None:
    retriever = _StubRetriever([_result("x")])
    strat = _bare_strategy(retrievers={"graph": retriever})
    query = SearchQuery(query="q", entity_focus=["Alice"], filters={"existing": 1})
    out = await strat._expand_via_graph(query, ["e1", "e2"])
    assert [r.source for r in out] == ["x"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters is not None
    assert sent.filters["id"] == ["e1", "e2"]
    assert sent.filters["existing"] == 1  # pre-existing filters preserved
    # The original query must not be mutated (deep copy is used).
    assert query.entity_focus == ["Alice"]
    assert query.filters is not None and "id" not in query.filters


async def test_expand_via_graph_swallows_retriever_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    strat = _bare_strategy(
        retrievers={"graph": _StubRetriever(raises=RuntimeError("store down"))}
    )
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, ["e1"]) == []
    # The label names the step, not one backend: the graph store may be the
    # embedded SQLite one or Gremlin Server.
    assert "graph expansion failed" in caplog.text
    assert "Neptune" not in caplog.text


# --------------------------------------------------------------------------- #
# _retrieve_documents
# --------------------------------------------------------------------------- #


async def test_retrieve_documents_no_retriever_returns_empty_dict() -> None:
    strat = _bare_strategy(retrievers={})
    assert await strat._retrieve_documents(["t1"], None) == {}


async def test_retrieve_documents_no_ids_returns_empty_dict() -> None:
    strat = _bare_strategy(retrievers={"document": _StubRetriever([_result("x")])})
    assert await strat._retrieve_documents([], None) == {}


async def test_retrieve_documents_wraps_results_under_text_units_key() -> None:
    retriever = _StubRetriever([_result("t1"), _result("t2")])
    strat = _bare_strategy(retrievers={"document": retriever})
    out = await strat._retrieve_documents(["t1", "t2"], suffix="dev")
    assert set(out) == {"text_units"}
    assert [r.source for r in out["text_units"]] == ["t1", "t2"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.search_type == SearchType.LEXICAL
    assert sent.top_k == 2
    assert sent.filters == {"id": ["t1", "t2"]}
    assert sent.suffix == "dev"
    assert sent.index_prefixes == ["text_units"]


async def test_retrieve_documents_swallows_retriever_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("os down"))}
    )
    assert await strat._retrieve_documents(["t1"], None) == {}


# --------------------------------------------------------------------------- #
# _rank_text_unit_ids / _restore_rank  (chunk candidates must be ranked)
# --------------------------------------------------------------------------- #


def test_rank_text_unit_ids_orders_by_entity_support() -> None:
    # "shared" is referenced by two entities, "solo" by one -> shared ranks first.
    nodes = [
        _result("e1", score=0.4, metadata={"text_unit_ids": ["shared", "solo"]}),
        _result("e2", score=0.9, metadata={"text_unit_ids": ["shared"]}),
    ]
    assert LocalSearchStrategy._rank_text_unit_ids(nodes)[0] == "shared"


def test_rank_text_unit_ids_breaks_ties_by_best_entity_score() -> None:
    nodes = [
        _result("e1", score=0.2, metadata={"text_unit_ids": ["low"]}),
        _result("e2", score=0.8, metadata={"text_unit_ids": ["high"]}),
    ]
    assert LocalSearchStrategy._rank_text_unit_ids(nodes) == ["high", "low"]


def test_rank_text_unit_ids_handles_unwrapped_single_id() -> None:
    # Neptune unwraps a single-element value_map list into a bare str.
    uuid = "8c1f2a70-1111-2222-3333-444455556666"
    nodes = [_result("e1", score=0.5, metadata={"text_unit_ids": uuid})]
    assert LocalSearchStrategy._rank_text_unit_ids(nodes) == [uuid]


def test_rank_text_unit_ids_dedups_across_entities() -> None:
    nodes = [
        _result("e1", score=0.5, metadata={"text_unit_ids": ["a", "b"]}),
        _result("e2", score=0.5, metadata={"text_unit_ids": ["a"]}),
    ]
    assert sorted(LocalSearchStrategy._rank_text_unit_ids(nodes)) == ["a", "b"]


def test_restore_rank_reorders_fetched_batch_to_ranked_order() -> None:
    # The id-batch fetch returns index order; the graph-support ranking must win.
    fetched = [_result("c3"), _result("c1"), _result("c2")]
    out = BaseSearchStrategy._restore_rank(fetched, ["c1", "c2", "c3"])
    assert [r.source for r in out] == ["c1", "c2", "c3"]


def test_restore_rank_projects_descending_scores() -> None:
    fetched = [_result("c2", score=0.0), _result("c1", score=0.0)]
    out = BaseSearchStrategy._restore_rank(fetched, ["c1", "c2"])
    # Scores must be strictly descending so downstream sorting is not arbitrary.
    assert out[0].score > out[1].score
    assert out[0].source == "c1"


def test_restore_rank_puts_unranked_ids_last() -> None:
    fetched = [_result("unknown"), _result("c1")]
    out = BaseSearchStrategy._restore_rank(fetched, ["c1"])
    assert [r.source for r in out] == ["c1", "unknown"]


def test_restore_rank_empty_inputs_are_passthrough() -> None:
    assert BaseSearchStrategy._restore_rank([], ["c1"]) == []
    fetched = [_result("c1")]
    assert BaseSearchStrategy._restore_rank(fetched, []) == fetched


def test_a_seed_entity_graph_node_takes_its_hits_document_filters() -> None:
    # The graph vertex projects no filters: the provenance of a seed fact
    # local search cites named no seed graph.
    filters = {"seed_graph": "procure", "department": "ops"}
    hits = [
        _result("v", metadata={"id": "v", "attributes": {"filters": filters}}),
        _result("d", metadata={"id": "d", "attributes": {"filters": {"k": "x"}}}),
    ]
    nodes = [
        _result("v", metadata={"id": "v", "source_kinds": ["seed_graph"]}),
        # A document entity is reported as before seed graphs.
        _result("d", metadata={"id": "d"}),
        # Reached, not a candidate: no hit names its filters.
        _result("p", metadata={"id": "p", "source_kinds": ["seed_graph"]}),
    ]

    kept = LocalSearchStrategy._with_hit_filters(nodes, hits)

    assert kept[0].metadata["attributes"] == {"filters": filters}
    assert kept[0].metadata["source_kinds"] == ["seed_graph"]
    assert kept[1] is nodes[1] and kept[2] is nodes[2]
    assert "attributes" not in nodes[0].metadata


def test_a_seed_entity_node_keeps_its_attributes_and_the_first_hits_filters() -> None:
    first = {"seed_graph": "procure"}
    hits = [
        _result("v", metadata={"id": "v", "attributes": {"filters": first}}),
        _result("v", metadata={"id": "v", "attributes": {"filters": {"k": "x"}}}),
        _result("w", metadata={"id": "w", "attributes": {"filters": first}}),
    ]
    own = {"seed_graph": "other"}
    nodes = [
        _result(
            "v",
            metadata={
                "id": "v",
                "source_kinds": ["seed_graph"],
                "attributes": {"tier": "gold"},
            },
        ),
        # Filters of its own: left as they are.
        _result(
            "w",
            metadata={
                "id": "w",
                "source_kinds": ["document", "seed_graph"],
                "attributes": {"filters": own},
            },
        ),
    ]

    kept = LocalSearchStrategy._with_hit_filters(nodes, hits)

    assert kept[0].metadata["attributes"] == {"tier": "gold", "filters": first}
    assert kept[1] is nodes[1]
