# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests of the structured strategy (docs/design.md §6.6).

The strategy is built for real on a fake provider bundle: the planner runs
on a scripted chat model, the graph store is the reference executor's fake
(``FakeGraphQueryRetriever`` over the procurement graph) and the document
store an in-memory index of the graph's entities and text units, so the
tests see what each step asks and what reaches the answer.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.graph_query import FakeGraphQueryRetriever
from tests.fixtures.fakes.tokens import WordTokenCounter
from tests.fixtures.graph_query import (
    CONFORMANCE_PLANS,
    procurement_entities,
    procurement_graph,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    get_strategy_spec,
    serves,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.search_strategies.structured_search import (
    StructuredSearchStrategy,
)
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import (
    Config,
    IndexArtifact,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.models.methodology import (
    Methodology,
    required_artifacts_of,
)
from unified_kg_rag.domain.retrieval.index_prefixes import HELD_ARTIFACTS_KEY
from unified_kg_rag.shared import ConfigurationError, StoreConnectionError
from unified_kg_rag.shared.exceptions import GraphQueryExecutionError

pytestmark = pytest.mark.unit

PREFIXES = Config().indexing.opensearch
ENTITIES, TEXT_UNITS, RELATIONSHIPS = (
    PREFIXES.entities_index_prefix,
    PREFIXES.text_units_index_prefix,
    PREFIXES.relationships_index_prefix,
)
PLANNER = "You translate a question into a query plan"


def _matches(metadata: dict[str, Any], source: str, filters: dict[str, Any]) -> bool:
    """A document passes ``filters``: ``id``, ``type`` and document keys."""
    for key, value in (filters or {}).items():
        values = value if isinstance(value, list) else [value]
        if key == "id":
            if source not in values:
                return False
        elif key == "type":
            if metadata.get("type") not in values:
                return False
        elif key.startswith("attr_"):
            stated = (metadata.get("attributes") or {}).get("filters") or {}
            held = stated.get(key[5:])
            held = held if isinstance(held, list) else [held]
            if not set(values) & set(held):
                return False
    return True


class _Documents:
    """The document store: the graph's entities, one text unit per entity
    and the relationships, searched by word overlap."""

    def __init__(self, unit_filters: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[SearchQuery] = []
        self.docs: dict[str, list[RetrievalResult]] = {
            ENTITIES: [
                RetrievalResult(
                    content=f"{e.name}: {e.description}",
                    score=1.0,
                    source=e.id,
                    retriever_type="entity",
                    metadata={
                        "id": e.id,
                        "name": e.name,
                        "type": e.type,
                        "text_unit_ids": e.text_unit_ids,
                        "attributes": e.attributes or {},
                    },
                )
                for e in procurement_entities()
            ],
            TEXT_UNITS: [
                RetrievalResult(
                    content=f"Passage about {e.name}.",
                    score=1.0,
                    source=f"tu-{e.id}",
                    retriever_type="text",
                    metadata={
                        "id": f"tu-{e.id}",
                        "attributes": {
                            "filters": (unit_filters or {}).get(
                                e.id, {"category": "ops"}
                            )
                        },
                    },
                )
                for e in procurement_entities()
            ],
            RELATIONSHIPS: [],
        }
        # Entities pass a document filter by their units' filters.
        for doc in self.docs[ENTITIES]:
            unit = next(
                u for u in self.docs[TEXT_UNITS] if u.source == f"tu-{doc.source}"
            )
            doc.metadata["attributes"] = {
                **doc.metadata["attributes"],
                "filters": {
                    **(doc.metadata["attributes"].get("filters") or {}),
                    **unit.metadata["attributes"]["filters"],
                },
            }

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields(self) -> None:
        return None

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = (query.index_prefixes or [ENTITIES])[0]
        hits = [
            d
            for d in self.docs.get(prefix, [])
            if _matches(d.metadata, str(d.source), query.filters or {})
        ]
        if query.query:
            words = set(query.query.lower().split())
            scored = [
                (len(words & set(d.content.lower().replace(":", " ").split())), d)
                for d in hits
            ]
            hits = [
                d.model_copy(update={"score": float(n)})
                for n, d in sorted(scored, key=lambda x: -x[0])
                if n
            ]
        return hits[: query.top_k * query.retrieval_multiplier]


class _Planner:
    """The scripted planner: answers from its queue (the last repeats)."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, human: str) -> str:
        assert PLANNER in system, system[:80]
        self.calls.append((system, human))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class _Fallback:
    """A fallback strategy's search, recording the queries it gets."""

    def __init__(self) -> None:
        self.queries: list[SearchQuery] = []
        self.metadata: dict[str, Any] = {"rerank": {"status": "applied"}}

    async def asearch(self, query: SearchQuery, config: Any = None) -> SearchResult:
        self.queries.append(query)
        hit = RetrievalResult(
            content="Fallback passage.",
            score=0.9,
            source="tu-fallback",
            retriever_type="text",
        )
        return SearchResult(
            query=query,
            results=[hit],
            total_results=1,
            search_strategy="local_search",
            processing_time=0.0,
            metadata=dict(self.metadata),
        )

    def reserved_filter_keys(self, strategy: SearchStrategy) -> frozenset[str]:
        return frozenset({"community_ids"})


def _strategy(
    planner: _Planner,
    *,
    config: Config | None = None,
    documents: _Documents | None = None,
    graph: Any = None,
) -> tuple[StructuredSearchStrategy, FakeGraphQueryRetriever, _Documents, _Fallback]:
    config = config or Config()
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(planner),
        embedding_factory=MagicMock(),
        rerank_factory=MagicMock(),
        token_counter_factory=lambda *_, **__: WordTokenCounter(),
    )
    providers.rerank_model = lambda *a, **k: None  # type: ignore[method-assign]
    graph = graph if graph is not None else FakeGraphQueryRetriever(procurement_graph())
    documents = documents or _Documents()
    strategy = StructuredSearchStrategy(
        config,
        {"document": documents, "graph": graph},  # type: ignore[dict-item]
        providers=providers,
    )
    strategy.hybrid_scorer.rerank_model = None
    fallback = _Fallback()
    for kind in (SearchStrategy.LOCAL, SearchStrategy.MIX):
        strategy._fallbacks[kind] = fallback  # type: ignore[assignment]
    return strategy, graph, documents, fallback


def _query(
    question: str = "How many vendors supply buyers?", **update: Any
) -> SearchQuery:
    query = SearchQuery(
        query=question,
        top_k=10,
        metadata={"search_strategy": "structured", "query_language": "English"},
    )
    return query.model_copy(update=update)


def _table(result: SearchResult) -> RetrievalResult:
    return next(
        r for r in result.results if r.retriever_type == SectionType.QUERY_RESULT.value
    )


def _with(plan: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {**plan, "entities": extra.pop("entities", []), **extra}


# --- registration --------------------------------------------------------------


def test_it_is_registered_served_by_every_graph_namespace_and_not_routed() -> None:
    spec = get_strategy_spec(SearchStrategy.STRUCTURED)
    assert spec.strategy_class is StructuredSearchStrategy
    assert spec.query_inputs == frozenset()
    config = Config()
    for methodology in (
        Methodology.GRAPHRAG,
        Methodology.LIGHTRAG,
        Methodology.HIPPORAG,
    ):
        assert serves(
            SearchStrategy.STRUCTURED, required_artifacts_of([methodology]), config
        )
    assert not serves(
        SearchStrategy.STRUCTURED, required_artifacts_of([Methodology.RAPTOR]), config
    )
    assert SearchStrategy.STRUCTURED not in config.search.auto_routable_strategies
    # Routable once configured.
    Config(search={"auto_routable_strategies": ["local", "structured"]})


def test_a_graph_store_without_the_port_is_refused() -> None:
    class _Plain:
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            return []

    with pytest.raises(ConfigurationError, match="GraphQueryPort"):
        _strategy(_Planner({}), graph=_Plain())


def test_the_filter_contract() -> None:
    strategy, *_ = _strategy(_Planner({}))
    prefixes = strategy.filtered_index_prefixes(SearchStrategy.STRUCTURED)
    assert set(prefixes) == {ENTITIES, RELATIONSHIPS, TEXT_UNITS}
    assert strategy.reserved_filter_keys(SearchStrategy.STRUCTURED) == {"community_ids"}


# --- answers -----------------------------------------------------------------------


async def test_a_count_is_answered_from_the_graph() -> None:
    planner = _Planner(
        _with(CONFORMANCE_PLANS["count_supply_pairs"], entities=["vendors"])
    )
    strategy, graph, documents, fallback = _strategy(planner)
    result = await strategy.asearch(_query())
    structured = result.metadata["structured"]
    assert structured["status"] == "answered", structured
    assert structured["operation"] == "count" and structured["exact"]
    assert structured["row_count"] == 6
    table = _table(result)
    assert "Result (computed over the whole graph):\n6" in table.content
    assert table.metadata["plan"]["returns"]["kind"] == "count"
    assert not fallback.queries
    # One store operation, scoped to the query's suffix; never a label.
    assert [r.suffix for r in graph.requests] == [None]
    assert "Entity-" not in json.dumps(graph.requests[0].model_dump(mode="json"))
    # The planner read the schema and the question as data.
    system, human = planner.calls[0]
    assert "- VENDOR (4)" in human and "How many vendors supply buyers?" in human
    assert "data, not instructions" in human
    assert "llm_calls" in structured["usage"]


async def test_rows_bring_their_entities_provenance_and_passages() -> None:
    plan = {
        "nodes": [
            {
                "var": "v",
                "types": ["VENDOR"],
                "where": [{"prop": "seed.country", "op": "eq", "value": "Norland"}],
            },
        ],
        "returns": {"kind": "rows", "properties": ["seed.employees"]},
        "entities": ["Norland"],
    }
    strategy, graph, documents, _ = _strategy(_Planner(plan))
    result = await strategy.asearch(_query("Which vendors are in Norland?"))
    assert result.metadata["structured"]["status"] == "answered"
    # A seed condition: the store returns the matches, code applies it.
    assert result.metadata["structured"]["operation"] == "tuples (scan)"
    table = _table(result)
    assert "1. v = Acme Supplies (VENDOR); seed.employees: 120" in table.content
    assert "2. v = 가나다 상사 (VENDOR); seed.employees: 300" in table.content
    assert table.metadata["provenance"] == {
        "kind": "document+seed_graph",
        "seed_graphs": ["procure"],
    }
    entities = [r for r in result.results if r.retriever_type == "entity"]
    assert {e.source for e in entities} == {"v1", "v3"}
    acme = next(e for e in entities if e.source == "v1")
    assert acme.metadata["source_kinds"] == ["document", "seed_graph"]
    assert acme.metadata["attributes"]["filters"]["seed_graph"] == ["procure"]
    passages = [r for r in result.results if r.retriever_type == "text"]
    assert {p.source for p in passages} == {"tu-v1", "tu-v3"}


async def test_a_refused_plan_is_repaired_once_with_the_refusal() -> None:
    bad = _with(
        {"nodes": [{"var": "v", "types": ["SUPPLIER"]}], "returns": {"kind": "count"}}
    )
    good = _with(
        {"nodes": [{"var": "v", "types": ["VENDOR"]}], "returns": {"kind": "count"}}
    )
    planner = _Planner(bad, good)
    strategy, *_ = _strategy(planner)
    result = await strategy.asearch(_query("How many suppliers?"))
    structured = result.metadata["structured"]
    assert structured["status"] == "answered" and structured["repairs"] == 1
    assert "4" in _table(result).content
    repair = planner.calls[1][1]
    assert "YOUR PREVIOUS PLAN WAS REFUSED" in repair
    assert "nodes[0].types[0]: unknown entity type 'SUPPLIER'" in repair


async def test_a_plan_still_refused_falls_back_with_its_entities() -> None:
    planner = _Planner(
        {"nodes": [{"var": "v", "label": "Entity-other"}], "entities": ["Acme"]}
    )
    strategy, graph, _, fallback = _strategy(planner)
    result = await strategy.asearch(_query("Who is Acme?"))
    structured = result.metadata["structured"]
    assert (structured["status"], structured["reason"]) == ("fallback", "invalid_plan")
    assert structured["fallback_strategy"] == "local"
    assert len(planner.calls) == 2 and not graph.requests
    assert fallback.queries[0].entity_focus == ["Acme"]
    assert fallback.queries[0].metadata["search_strategy"] == "local"
    assert [r.source for r in result.results] == ["tu-fallback"]


@pytest.mark.parametrize(
    "reply,reason",
    [
        (
            {"unanswerable": True, "entities": ["Acme"], "keywords": ["prices"]},
            "unanswerable",
        ),
        ("I cannot write JSON today.", "invalid_plan"),
        (RuntimeError("model overloaded"), "planner_error"),
        (
            _with(
                {
                    "nodes": [{"var": "x", "mention": "Nobody Inc"}],
                    "returns": {"kind": "rows"},
                }
            ),
            "unlinked_mention",
        ),
        (
            _with(
                {
                    "nodes": [
                        {
                            "var": "v",
                            "where": [
                                {"prop": "name", "op": "eq", "value": "Initech Tools"}
                            ],
                        },
                        {"var": "b", "types": ["BUYER"]},
                    ],
                    "edges": [{"types": ["SUPPLIES"]}],
                    "returns": {"kind": "rows", "vars": ["b"]},
                }
            ),
            "empty_result",
        ),
    ],
)
async def test_no_structured_answer_runs_the_fallback(reply: Any, reason: str) -> None:
    strategy, *_, fallback = _strategy(_Planner(reply))
    result = await strategy.asearch(_query())
    structured = result.metadata["structured"]
    assert (structured["status"], structured["reason"]) == ("fallback", reason)
    assert len(fallback.queries) == 1
    if reason == "unanswerable":
        assert fallback.queries[0].entity_focus == ["Acme"]


async def test_a_zero_count_and_a_no_are_answers() -> None:
    zero = _with(
        {
            "nodes": [
                {
                    "var": "v",
                    "where": [{"prop": "name", "op": "eq", "value": "Initech Tools"}],
                },
                {"var": "b", "types": ["BUYER"]},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "count"},
        }
    )
    strategy, *_, fallback = _strategy(_Planner(zero))
    result = await strategy.asearch(_query("How many buyers does Initech supply?"))
    assert result.metadata["structured"]["status"] == "answered"
    assert "\n0" in _table(result).content
    strategy, *_, fallback = _strategy(
        _Planner(_with(CONFORMANCE_PLANS["exists_supply"]))
    )
    result = await strategy.asearch(_query("Does Initech supply anyone?"))
    assert _table(result).content.endswith("\nno")
    assert not fallback.queries


async def test_a_store_time_limit_falls_back_and_a_dead_store_fails() -> None:
    plan = _with(CONFORMANCE_PLANS["count_supply_pairs"])
    timed_out = FakeGraphQueryRetriever(
        procurement_graph(), fail=GraphQueryExecutionError("slow", timed_out=True)
    )
    strategy, *_, fallback = _strategy(_Planner(plan), graph=timed_out)
    result = await strategy.asearch(_query())
    assert result.metadata["structured"]["reason"] == "timeout"
    assert fallback.queries
    dead = FakeGraphQueryRetriever(
        procurement_graph(), fail=StoreConnectionError("down")
    )
    strategy, *_ = _strategy(_Planner(plan), graph=dead)
    with pytest.raises(StoreConnectionError):
        await strategy.asearch(_query())


async def test_without_a_fallback_the_answer_abstains() -> None:
    config = Config(search={"structured": {"fallback_strategy": None}})
    strategy, *_, fallback = _strategy(
        _Planner({"unanswerable": True, "entities": []}), config=config
    )
    result = await strategy.asearch(_query())
    assert result.results == [] and not fallback.queries
    assert result.metadata["structured"]["status"] == "failed"


async def test_a_fallback_the_namespace_lacks_gives_way_to_one_it_serves() -> None:
    config = Config(search={"structured": {"fallback_strategy": "global"}})
    strategy, *_, fallback = _strategy(_Planner({"unanswerable": True}), config=config)
    light = required_artifacts_of([Methodology.LIGHTRAG])
    query = _query(
        metadata={
            "search_strategy": "structured",
            HELD_ARTIFACTS_KEY: sorted(a.value for a in light),
        }
    )
    result = await strategy.asearch(query)
    assert result.metadata["structured"]["fallback_strategy"] == "mix"
    # mix reads the plan's names as low-level keywords.
    assert fallback.queries[0].metadata["search_strategy"] == "mix"


async def test_mentions_link_through_the_hybrid_entity_search_in_any_script() -> None:
    plan = _with(
        {
            "nodes": [
                {"var": "x", "mention": "가나다상사"},
                {"var": "b", "types": ["BUYER"]},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "rows", "vars": ["b"]},
        }
    )
    documents = _Documents()
    # The compact key joins what the analyzer split: 가나다 상사 matches.
    for doc in documents.docs[ENTITIES]:
        if doc.source == "v3":
            doc.content = "가나다상사 가나다 상사: a vendor"
    planner = _Planner(plan)
    strategy, graph, documents, _ = _strategy(planner, documents=documents)
    query = _query(
        "가나다상사는 어느 구매처에 공급합니까?",
        metadata={"search_strategy": "structured", "query_language": "Korean"},
    )
    result = await strategy.asearch(query)
    assert result.metadata["structured"]["linked"] == {"x": ["v3"]}
    names = _table(result).content
    assert "株式会社サクラ" in names and "Hooli Retail" in names
    assert "(Korean)" in planner.calls[0][0]
    link = next(c for c in documents.calls if c.query == "가나다상사")
    assert link.index_prefixes == [ENTITIES]
    assert link.search_type.value == "hybrid"


async def test_a_document_filter_scans_and_checks_every_bound_vertex() -> None:
    # v1's passages belong to another category: it is filtered out.
    documents = _Documents(unit_filters={"v1": {"category": "legal"}})
    strategy, graph, documents, _ = _strategy(
        _Planner(_with(CONFORMANCE_PLANS["count_supply_pairs"])), documents=documents
    )
    result = await strategy.asearch(_query(filters={"attr_category": "ops"}))
    structured = result.metadata["structured"]
    assert structured["operation"] == "tuples (scan)"
    # (v1, b1) and (v1, b2) are out: 4 of the 6 pairs remain.
    assert "\n4" in _table(result).content
    checks = [c for c in documents.calls if c.filters and "attr_category" in c.filters]
    assert any(c.index_prefixes == [ENTITIES] for c in checks)


async def test_augment_fuses_the_fallback_beside_the_query_result() -> None:
    config = Config(search={"structured": {"mode": "augment"}})
    strategy, graph, _, fallback = _strategy(
        _Planner(_with(CONFORMANCE_PLANS["buyers_per_vendor"], entities=["vendors"])),
        config=config,
    )
    fallback.metadata["context_sufficient"] = False
    result = await strategy.asearch(_query("Which vendor supplies the most buyers?"))
    assert result.metadata["structured"]["status"] == "answered"
    sources = [r.source for r in result.results]
    assert "graph-query" in sources and "tu-fallback" in sources
    assert fallback.queries[0].entity_focus == ["vendors"]
    # The fallback's verdict on its own context does not judge this one.
    assert "context_sufficient" not in result.metadata


async def test_the_planner_writes_in_the_query_language() -> None:
    planner = _Planner(_with(CONFORMANCE_PLANS["count_supply_pairs"]))
    strategy, *_ = _strategy(planner)
    await strategy.asearch(
        _query(metadata={"search_strategy": "structured", "query_language": "Japanese"})
    )
    assert "(Japanese)" in planner.calls[0][0]


async def test_the_profile_is_read_once_per_suffix_and_ttl() -> None:
    planner = _Planner(_with(CONFORMANCE_PLANS["count_supply_pairs"]))
    strategy, graph, *_ = _strategy(planner)
    for _ in range(3):
        await strategy.asearch(_query())
    await strategy.asearch(_query(suffix="other"))
    assert graph.describes == [None, "other"]


def test_the_query_result_section_is_budgeted() -> None:
    config = Config()
    assert config.search.token_manager.type_budgets.query_result == 0.25
    from unified_kg_rag.adapters.retrieval.token_manager import TokenManager

    assert TokenManager._BUDGET_FIELDS[SectionType.QUERY_RESULT] == "query_result"
    assert IndexArtifact.GRAPH in get_strategy_spec(SearchStrategy.STRUCTURED).required_artifacts  # type: ignore[operator]


def test_the_answer_is_told_to_use_the_result_as_computed() -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.prompts.retrieval import QUERY_RESULT_INSTRUCTIONS

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = Config()
    table = RetrievalResult(
        content="Graph query: ...",
        score=1.0,
        source="graph-query",
        retriever_type="query_result",
    )
    with_table = {
        "search_results": SearchResult(
            query=_query(),
            results=[table],
            total_results=1,
            search_strategy="structured_search",
            processing_time=0.0,
        )
    }
    assert chain._answer_instructions(with_table) == QUERY_RESULT_INSTRUCTIONS
    without = {
        "search_results": SearchResult(
            query=_query(),
            results=[],
            total_results=0,
            search_strategy="local_search",
            processing_time=0.0,
        )
    }
    # Every other answer prompt is unchanged.
    assert chain._answer_instructions(without) == ""


async def test_the_declared_schema_alone_still_lists_the_core_properties() -> None:
    config = Config(
        processing={
            "graph_extraction": {
                "entity_types": ["VENDOR", "BUYER"],
                "relationship_types": ["SUPPLIES"],
            }
        },
        search={"structured": {"schema_source": "declared"}},
    )
    plan = _with(
        {
            "nodes": [
                {
                    "var": "v",
                    "types": ["VENDOR"],
                    "where": [{"prop": "rank", "op": "gte", "value": 3}],
                }
            ],
            "returns": {"kind": "count"},
        }
    )
    strategy, graph, *_ = _strategy(_Planner(plan), config=config)
    result = await strategy.asearch(_query("How many important vendors?"))
    assert result.metadata["structured"]["status"] == "answered"
    assert graph.describes == []  # no profile read
    assert "\n3" in _table(result).content
