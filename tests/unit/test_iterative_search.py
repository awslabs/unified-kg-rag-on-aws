# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests of iterative bridge follow-up search.

The strategy is built for real on a fake provider bundle: the controller
chain runs on a scripted chat model, the reranker
orders passages by the words they share with its query, and the base
strategy's searches are stubs recording the queries they get, so the tests
see what each round asks and how the rounds' candidates are fused.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.tokens import WordTokenCounter
from unified_kg_rag.adapters.llm.retry import retry_deadline
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    get_strategy_spec,
    query_inputs,
    routable_strategies,
    wrapped_strategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.search_strategies.iterative_search import (
    IterativeSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    LightRAGSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.application.config import unknown_keys
from unified_kg_rag.application.retrieval import query_processing
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    IndexArtifact,
    Methodology,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.domain.models.methodology import METHODOLOGY_PROFILES
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit

TEXT_UNITS = Config().indexing.opensearch.text_units_index_prefix
QUESTION = "Where is the storage site of the firm Vendor hires?"
FOLLOW_UP = "Where is Depot located?"
FILTERS = {"attr_category": "ops"}


def _hit(
    content: str, *, kind: str = "text", score: float = 0.5, **metadata: Any
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source=f"{kind}:{content}",
        retriever_type=kind,
        metadata=metadata,
    )


ROUND_0 = [
    _hit("Vendor hires Carrier for freight.", score=0.9),
    _hit("Carrier rents Depot for storage.", score=0.8),
    _hit("Vendor", kind="entity", score=0.02),
    _hit("Carrier", kind="entity", score=0.01),
]
HARBOR = _hit(
    "Depot is located in Harbor.",
    text_unit_ids=["unit-harbor"],
    iri="https://example.org/depot",
    source_kinds=["document"],
    attributes={"filters": {"category": "ops"}},
)
DEPOT = _hit("Depot", kind="entity", score=0.03, text_unit_ids=["unit-harbor"])


class _Base:
    """A base strategy's search: ``answer(query)`` gives its results."""

    def __init__(self, answer: Callable[[SearchQuery], list[RetrievalResult]]):
        self.answer = answer
        self.queries: list[SearchQuery] = []
        self.delay = 0.0
        self.cancelled = False

    async def asearch(self, query: SearchQuery, config: Any = None) -> SearchResult:
        self.queries.append(query)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        results = self.answer(query)
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="local_search",
            processing_time=0.0,
            metadata={
                "candidate_entity_count": 1,
                "rerank": {"status": "applied", "reason": None, "stages": []},
            },
        )


class _Documents:
    """The document retriever: a text-units search finds ``hits[text]``."""

    def __init__(self, hits: dict[str, list[RetrievalResult]] | None = None):
        self.hits = hits or {}
        self.calls: list[SearchQuery] = []

    def filter_fields(self) -> None:
        return None

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return list(self.hits.get(query.query, []))


class _Reranker:
    """Orders passages by the words they share with the query."""

    def __init__(self) -> None:
        self.top_n: int | None = None
        self.queries: list[str] = []

    def compress_documents(self, documents: list[Any], query: str) -> list[Any]:
        self.queries.append(query)
        words = set(query.lower().replace("?", "").split())

        def overlap(doc: Any) -> int:
            return len(words & set(doc.page_content.lower().rstrip(".").split()))

        ranked = sorted(documents, key=overlap, reverse=True)
        for doc in ranked:
            doc.metadata["relevance_score"] = overlap(doc) / 10
        return ranked


class _RerankFactory:
    def __init__(self) -> None:
        self.model = _Reranker()

    def get_model(self, model_id: Any, **kwargs: Any) -> _Reranker:
        self.model.top_n = kwargs.get("top_k")
        return self.model

    def get_model_info(self, model_id: Any) -> None:
        return None


class _Script:
    """The scripted model: each prompt kind answers from its own queue (the
    last answer repeats); an exception in a queue is raised."""

    KINDS = {
        "controller": "You control the retrieval",
    }

    def __init__(self, **replies: list[Any]) -> None:
        self.replies = replies
        self.calls: dict[str, list[tuple[str, str]]] = {k: [] for k in self.KINDS}
        # The output fixer's prompts (one human message, no system).
        self.repairs: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if not system and "did not satisfy the constraints" in human:
            self.repairs.append(human)
            return self.replies.get("fixer", ["still no JSON"])[0]
        for kind, marker in self.KINDS.items():
            if marker in system:
                self.calls[kind].append((system, human))
                queue = self.replies[kind]
                reply = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(reply, BaseException):
                    raise reply
                return reply if isinstance(reply, str) else json.dumps(reply)
        raise AssertionError(f"unexpected prompt: {system[:60]!r}")


def _verdict(
    sufficient: bool, bridges: list[str] | None = None, queries: list[str] | None = None
) -> dict[str, Any]:
    return {
        "sufficient": sufficient,
        "known_facts": ["Vendor hires Carrier."],
        "bridge_entities": bridges or [],
        "next_queries": queries or [],
    }


INSUFFICIENT = _verdict(False, ["Depot"], [FOLLOW_UP])
SUFFICIENT = _verdict(True)


def _strategy(
    script: _Script,
    *,
    base: str = "local",
    documents: _Documents | None = None,
    round_0: list[RetrievalResult] | None = None,
    follow_up: Callable[[SearchQuery], list[RetrievalResult]] | None = None,
    config: Config | None = None,
    **settings: Any,
) -> tuple[IterativeSearchStrategy, _Base, _Base, _Documents, _RerankFactory]:
    config = config or Config()
    config.search.iterative_search.base = base  # type: ignore[assignment]
    for name, value in settings.items():
        setattr(config.search.iterative_search, name, value)
    reranks = _RerankFactory()
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(script),
        embedding_factory=MagicMock(),
        rerank_factory=reranks,
        token_counter_factory=lambda *_, **__: WordTokenCounter(),
    )
    documents = documents or _Documents({FOLLOW_UP: [HARBOR]})
    strategy = IterativeSearchStrategy(
        config,
        {"document": documents, "graph": MagicMock()},  # type: ignore[dict-item]
        providers=providers,
    )
    first = _Base(lambda _: list(ROUND_0 if round_0 is None else round_0))
    later = _Base(follow_up or (lambda _: [DEPOT, HARBOR]))
    strategy.base.asearch = first.asearch  # type: ignore[method-assign]
    strategy.follow_up.asearch = later.asearch  # type: ignore[method-assign]
    return strategy, first, later, documents, reranks


def _query(**update: Any) -> SearchQuery:
    query = SearchQuery(
        query=QUESTION,
        top_k=5,
        suffix="tenant_a",
        entity_focus=["Vendor"],
        filters=dict(FILTERS),
        metadata={"search_strategy": "iterative", "query_language": "English"},
    )
    return query.model_copy(update=update)


def _contents(result: SearchResult) -> list[str]:
    return [r.content for r in result.results]


# --- the controller loop ------------------------------------------------------


async def test_a_sufficient_round_zero_is_the_base_search_unchanged() -> None:
    script = _Script(controller=[SUFFICIENT])
    strategy, first, later, documents, reranks = _strategy(script)

    result = await strategy.asearch(_query())

    assert result.results == ROUND_0
    assert len(script.calls["controller"]) == 1
    assert later.queries == [] and documents.calls == [] and reranks.model.queries == []
    meta = result.metadata
    assert meta["iterative"]["rounds"] == 0
    assert meta["iterative"]["stop_reason"] == "sufficient"
    assert meta["iterative"]["llm_calls"] == 1
    assert "context_sufficient" not in meta and "degraded_error" not in meta
    # The base's own metadata (its rerank outcome included) is kept.
    assert meta["candidate_entity_count"] == 1
    assert meta["rerank"]["status"] == "applied"
    # Round 0 searched as the base: its mode, the caller's filters and scope.
    (asked,) = first.queries
    assert asked.metadata["search_strategy"] == "local"
    assert (asked.filters, asked.suffix, asked.query) == (FILTERS, "tenant_a", QUESTION)


async def test_an_insufficient_round_zero_runs_one_seeded_follow_up() -> None:
    script = _Script(controller=[INSUFFICIENT, SUFFICIENT])
    strategy, _, later, documents, reranks = _strategy(script, round_top_k=3)

    result = await strategy.asearch(_query())

    # The base reruns seeded with the bridge entity, under the same filters.
    (seeded,) = later.queries
    assert seeded.query == FOLLOW_UP
    assert seeded.entity_focus == ["Depot"] and seeded.ll_keywords == ["Depot"]
    assert seeded.hl_keywords == []
    assert seeded.metadata["search_strategy"] == "local"
    assert (seeded.filters, seeded.suffix, seeded.top_k) == (FILTERS, "tenant_a", 5)
    # ... and the follow-up query searches the text units the same way.
    (text,) = documents.calls
    assert (text.query, text.index_prefixes, text.top_k) == (
        FOLLOW_UP,
        [TEXT_UNITS],
        3,
    )
    assert (text.filters, text.suffix, text.search_type) == (
        FILTERS,
        "tenant_a",
        seeded.search_type,
    )
    # Fused with round 0 and reranked against the question and the follow-up.
    assert "Depot is located in Harbor." in _contents(result)
    assert set(_contents(result)) >= {r.content for r in ROUND_0}
    assert reranks.model.queries == [f"{QUESTION}\n{FOLLOW_UP}"]
    # The second verdict read the new passage and what was searched.
    _, human = script.calls["controller"][1]
    assert "Depot is located in Harbor." in human and FOLLOW_UP in human
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"], meta["llm_calls"]) == (
        1,
        "sufficient",
        2,
    )
    assert meta["searched"] == [FOLLOW_UP]
    assert [v["sufficient"] for v in meta["verdicts"]] == [False, True]
    assert meta["verdicts"][0]["bridge_entities"] == ["Depot"]
    assert meta["added_results"] == 2
    assert result.metadata["rerank"]["status"] == "applied"
    assert "context_sufficient" not in result.metadata


async def test_the_rerank_calls_are_counted_in_the_query_s_usage() -> None:
    from unified_kg_rag.shared.usage import usage_scope

    script = _Script(controller=[INSUFFICIENT, SUFFICIENT])
    strategy, *_ = _strategy(script)

    with usage_scope() as meter:
        result = await strategy.asearch(_query())
    used = meter.snapshot()

    # The fusion's one rerank: a rerank call of one search unit, under the
    # rerank role and the rerank model, and no model call.
    model = Config().rerank_model_id
    assert (used["rerank_calls"], used["rerank_search_units"]) == (1, 1)
    assert used["by_role"]["rerank"]["rerank_calls"] == 1
    assert used["by_model"][getattr(model, "value", model)]["llm_calls"] == 0
    assert result.metadata["iterative"]["usage"]["rerank_calls"] == 1


def test_a_rerank_request_counts_a_search_unit_per_hundred_documents() -> None:
    from unified_kg_rag.shared.usage import record_rerank_call, usage_scope

    with usage_scope() as meter:
        for documents in (1, 100, 101, 250):
            record_rerank_call("reranker", documents=documents)
    used = meter.snapshot()
    assert (used["rerank_calls"], used["rerank_search_units"]) == (4, 1 + 1 + 2 + 3)
    assert used["llm_calls"] == 0


async def test_follow_up_candidates_keep_their_provenance() -> None:
    strategy, *_ = _strategy(_Script(controller=[INSUFFICIENT, SUFFICIENT]))

    result = await strategy.asearch(_query())

    (harbor,) = [r for r in result.results if r.source == HARBOR.source]
    for key in ("text_unit_ids", "iri", "source_kinds", "attributes"):
        assert harbor.metadata[key] == HARBOR.metadata[key]
    assert harbor.metadata["reranked"] is True


async def test_a_round_zero_item_leaves_only_when_outranked() -> None:
    # Each type keeps round 0's share of the context (12 entities, above the
    # entity quota's floor of 10): the follow-up's entity, first in its
    # round, takes the slot of round 0's last-ranked entity, and only that.
    entities = [_hit(f"E{i}", kind="entity", score=0.1) for i in range(12)]
    strategy, *_ = _strategy(
        _Script(controller=[INSUFFICIENT, SUFFICIENT]),
        round_0=[*ROUND_0[:2], *entities],
        follow_up=lambda _: [DEPOT],
    )
    result = await strategy.asearch(_query())
    kept = [r.content for r in result.results if r.retriever_type == "entity"]
    assert len(kept) == 12
    assert set(kept) == {f"E{i}" for i in range(11)} | {"Depot"}


async def test_never_sufficient_stops_at_the_round_cap_and_says_so() -> None:
    script = _Script(
        controller=[
            _verdict(False, ["Depot"], [FOLLOW_UP]),
            _verdict(False, ["Harbor"], ["Where is Harbor?"]),
            _verdict(False, ["Marsh"], ["Where is Marsh?"]),
        ]
    )
    strategy, _, later, _, _ = _strategy(script, max_rounds=2)

    result = await strategy.asearch(_query())

    assert len(later.queries) == 2
    assert len(script.calls["controller"]) == 3  # the last round is judged too
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (2, "max_rounds")
    assert meta["searched"] == [FOLLOW_UP, "Where is Harbor?"]
    assert result.metadata["context_sufficient"] is False


async def test_nothing_new_to_search_ends_the_rounds_insufficient() -> None:
    # The second verdict repeats the bridge and the query, in other cases.
    script = _Script(
        controller=[INSUFFICIENT, _verdict(False, ["DEPOT"], [FOLLOW_UP.upper()])]
    )
    strategy, _, later, _, _ = _strategy(script, max_rounds=3)

    result = await strategy.asearch(_query())

    assert len(later.queries) == 1
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (1, "no_follow_up")
    assert result.metadata["context_sufficient"] is False


async def test_bridge_entities_alone_drive_a_round_without_queries() -> None:
    script = _Script(controller=[_verdict(False, ["Depot"]), SUFFICIENT])
    strategy, _, later, documents, _ = _strategy(script, max_next_queries=0)

    await strategy.asearch(_query())

    (seeded,) = later.queries
    assert (seeded.query, seeded.entity_focus) == ("Depot", ["Depot"])
    assert documents.calls == []


async def test_a_round_follows_at_most_five_bridge_entities() -> None:
    bridges = [f"Site {n}" for n in range(8)]
    script = _Script(controller=[_verdict(False, bridges), SUFFICIENT])
    strategy, _, later, _, _ = _strategy(script, max_next_queries=0)

    await strategy.asearch(_query())

    (seeded,) = later.queries
    assert seeded.entity_focus == bridges[:5]
    # search.iterative_search.max_bridge_entities sets the cap.
    script = _Script(controller=[_verdict(False, bridges), SUFFICIENT])
    strategy, _, later, _, _ = _strategy(
        script, max_next_queries=0, max_bridge_entities=2
    )
    await strategy.asearch(_query())
    (seeded,) = later.queries
    assert seeded.entity_focus == bridges[:2]


async def test_the_controller_reads_the_answer_s_budget_by_default() -> None:
    script = _Script(controller=[SUFFICIENT])
    strategy, *_ = _strategy(script)
    budgets: list[Any] = []
    pack = strategy.token_manager.optimize_context

    def recorded(results: Any, text: str, budget: Any, *args: Any) -> Any:
        budgets.append(budget)
        return pack(results, text, budget, *args)

    strategy.token_manager.optimize_context = recorded  # type: ignore[method-assign]
    await strategy.asearch(_query())
    await strategy.asearch(_query(max_tokens=4000))
    strategy.settings.controller_context_tokens = 2000
    await strategy.asearch(_query(max_tokens=4000))

    # None: the token manager's answer budget (max_context_tokens).
    assert budgets == [None, 4000, 2000]


async def test_at_most_max_next_queries_are_searched() -> None:
    queries = ["q one", "q two", "q three"]
    script = _Script(controller=[_verdict(False, ["Depot"], queries), SUFFICIENT])
    strategy, _, _, documents, _ = _strategy(script, max_next_queries=2)

    result = await strategy.asearch(_query())

    assert [c.query for c in documents.calls] == ["q one", "q two"]
    assert result.metadata["iterative"]["searched"] == ["q one", "q two"]


@pytest.mark.parametrize(
    "reply",
    ["no JSON at all", json.dumps({"next_queries": [FOLLOW_UP]}), '{"sufficient":'],
    ids=["prose", "no-verdict", "cut-off"],
)
async def test_an_unreadable_verdict_answers_from_what_is_there(reply: str) -> None:
    script = _Script(controller=[reply])
    strategy, _, later, _, _ = _strategy(script)

    result = await strategy.asearch(_query())

    assert result.results == ROUND_0
    assert later.queries == []
    meta = result.metadata["iterative"]
    assert meta["stop_reason"] == meta["degraded"] == "controller_unparsed"
    assert "context_sufficient" not in result.metadata
    # The answer is marked degraded in the query's ledger: with the parse
    # error when the answer does not parse (after its one repair), without
    # one for an object that holds no verdict.
    (entry,) = result.metadata["degraded"]
    assert entry["step"] == "iterative_controller"
    parsed = reply.startswith('{"next')
    assert entry["error_type"] == (None if parsed else "OutputParserException")
    assert result.metadata.get("degraded_error") == entry["error_type"]
    # A cut-off answer is not repaired either: a fixer could only invent it.
    assert len(script.repairs) == (1 if reply == "no JSON at all" else 0)


async def test_a_verdict_the_fixer_repairs_is_read() -> None:
    # A Llama-style trailing comma parses as it is; prose goes to the fixer.
    fixed = json.dumps(INSUFFICIENT)
    script = _Script(controller=["Not sufficient yet, search the depot", SUFFICIENT])
    script.replies["fixer"] = [fixed]
    strategy, _, later, _, _ = _strategy(script)

    result = await strategy.asearch(_query())

    assert len(script.repairs) == 1 and later.queries
    assert "degraded_error" not in result.metadata


@pytest.mark.parametrize(
    ("sufficient", "follows_up"),
    [("true", False), (" True ", False), ("false", True), ("FALSE", True)],
)
async def test_a_verdict_written_as_a_string_is_read(
    sufficient: str, follows_up: bool
) -> None:
    # Models often quote their JSON booleans.
    verdict = {**INSUFFICIENT, "sufficient": sufficient}
    strategy, _, later, _, _ = _strategy(_Script(controller=[verdict, SUFFICIENT]))

    result = await strategy.asearch(_query())

    assert bool(later.queries) is follows_up
    meta = result.metadata["iterative"]
    assert meta["stop_reason"] == "sufficient" and meta["degraded"] is None


async def test_a_failed_controller_call_degrades_after_a_round() -> None:
    script = _Script(controller=[INSUFFICIENT, RuntimeError("throttled")])
    strategy, *_ = _strategy(script)

    result = await strategy.asearch(_query())

    assert "Depot is located in Harbor." in _contents(result)
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["degraded"]) == (1, "controller_failed")
    assert result.metadata["degraded_error"] == "RuntimeError"
    assert "context_sufficient" not in result.metadata


async def test_a_fatal_controller_error_fails_the_query() -> None:
    script = _Script(controller=[ConfigurationError("model access denied")])
    strategy, *_ = _strategy(script)

    with pytest.raises(ConfigurationError):
        await strategy.asearch(_query())


# --- the query deadline --------------------------------------------------------


async def test_the_deadline_cancels_a_follow_up_round_mid_way() -> None:
    config = Config()
    config.search.model_timeout_seconds = 1.0
    script = _Script(controller=[INSUFFICIENT, SUFFICIENT])
    strategy, _, later, _, _ = _strategy(script, config=config)
    later.delay = 30.0

    started = time.monotonic()
    # 0.4 s are left for the rounds once the answer's 1 s is set aside.
    with retry_deadline(time.monotonic() + 1.4):
        result = await strategy.asearch(_query())

    assert time.monotonic() - started < 5
    assert later.cancelled
    assert result.results == ROUND_0
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (0, "deadline")
    # The controller judged round 0 insufficient before the deadline cut the
    # round it started: what is returned is what it judged.
    assert result.metadata["context_sufficient"] is False
    assert result.metadata["insufficiency_reason"] == "deadline"


async def test_a_deadline_after_a_judged_round_keeps_that_verdict() -> None:
    config = Config()
    config.search.model_timeout_seconds = 1.0
    script = _Script(controller=[INSUFFICIENT, _verdict(False, ["Harbor"])])
    strategy, _, later, _, _ = _strategy(config=config, script=script)

    def round_answer(query: SearchQuery) -> list[RetrievalResult]:
        later.delay = 30.0  # the second follow-up round outlasts the deadline
        return [DEPOT, HARBOR]

    later.answer = round_answer
    with retry_deadline(time.monotonic() + 1.4):
        result = await strategy.asearch(_query())

    assert "Depot is located in Harbor." in _contents(result)
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (1, "deadline")
    assert result.metadata["context_sufficient"] is False
    assert result.metadata["insufficiency_reason"] == "deadline"


async def test_no_round_starts_without_time_for_the_answer() -> None:
    config = Config()
    config.search.model_timeout_seconds = 60.0
    script = _Script(controller=[INSUFFICIENT])
    strategy, *_ = _strategy(script, config=config)

    with retry_deadline(time.monotonic() + 30.0):
        result = await strategy.asearch(_query())

    assert script.calls["controller"] == []
    assert result.results == ROUND_0
    assert result.metadata["iterative"]["stop_reason"] == "deadline"


# --- languages -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("metadata", "translation", "language"),
    [
        ({"query_language": "Korean"}, {}, "Korean"),
        ({}, {"enabled": False, "source_language": LanguageCode.JA}, "Japanese"),
        ({}, {"enabled": True, "target_language": LanguageCode.ZH}, "Chinese"),
        (
            {"query_language": ""},
            {"enabled": False, "source_language": LanguageCode.KO},
            "Korean",
        ),
    ],
    ids=["from-the-chain", "index-as-written", "translated-index", "blank-is-unset"],
)
async def test_the_controller_writes_follow_ups_in_the_query_language(
    metadata: dict[str, Any], translation: dict[str, Any], language: str
) -> None:
    config = Config()
    for name, value in translation.items():
        setattr(config.processing.translation, name, value)
    script = _Script(controller=[SUFFICIENT])
    strategy, *_ = _strategy(script, config=config)

    await strategy.asearch(
        _query(metadata={"search_strategy": "iterative", **metadata})
    )

    ((system, _),) = script.calls["controller"]
    assert f"in {language}, the language of the indexed corpus" in system


@pytest.mark.parametrize(
    ("first", "again", "question", "query"),
    [
        (
            "동부 창고",
            "동부창고",
            "한빛상사가 맡긴 운송사의 창고는?",
            "동부창고는 어디에 있나?",
        ),
        ("東 倉庫", "東倉庫", "北星商事の倉庫はどこ？", "東倉庫はどこにある？"),
        ("东方 仓库", "东方仓库", "北辰贸易的仓库在哪里？", "东方仓库在哪里？"),
        # Regression (rv9 query L7): the bridge key was compact_name_key,
        # not the key a query's name links by (linking.link_key).
        ("أكمي للروبوتات", "اكمي للروبوتات", "من يخزن قطع أكمي؟", "أين اكمي؟"),
        ("Фёдоров Склад", "Федоров Склад", "Где склад?", "Где Федоров Склад?"),
    ],
    ids=["ko", "ja", "zh", "ar", "ru"],
)
async def test_dense_script_bridges_and_queries_are_followed_once(
    first: str, again: str, question: str, query: str
) -> None:
    script = _Script(
        controller=[
            _verdict(False, [first], [query]),
            _verdict(False, [again], [query]),
        ]
    )
    strategy, _, later, documents, reranks = _strategy(script, max_rounds=3)

    result = await strategy.asearch(_query(query=question))

    # The same name without the space is the same bridge; the query as is.
    (seeded,) = later.queries
    assert (seeded.query, seeded.entity_focus) == (query, [first])
    assert [c.query for c in documents.calls] == [query]
    assert reranks.model.queries == [f"{question}\n{query}"]
    assert result.metadata["iterative"]["stop_reason"] == "no_follow_up"


# --- removed settings ------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        {"search": {"iterative_search": {"decompose": True}}},
        {"custom_prompts": {"query_decomposition_system": "plan"}},
        {"custom_prompts": {"decomposition_step_human": "answer"}},
    ],
    ids=["decompose", "decomposition_prompt", "step_prompt"],
)
def test_the_decomposition_settings_are_unknown_keys(section: dict[str, Any]) -> None:
    # The decomposition mode measured no better than the controller loop
    # and was removed before a release: its keys fail the load.
    (key,) = unknown_keys(section, Config)
    assert "decompos" in key


# --- the base it wraps -----------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "inputs", "cls", "spare"),
    [
        ("local", QueryInput.ENTITIES, LocalSearchStrategy, ()),
        ("mix", QueryInput.DUAL_KEYWORDS, LightRAGSearchStrategy, (SectionType.TEXT,)),
    ],
)
def test_iterative_reads_what_its_base_reads(
    base: str, inputs: QueryInput, cls: type, spare: tuple[SectionType, ...]
) -> None:
    config = Config()
    config.search.iterative_search.base = base  # type: ignore[assignment]

    assert wrapped_strategy(SearchStrategy.ITERATIVE, config) is SearchStrategy(base)
    assert query_inputs(SearchStrategy.ITERATIVE, config) == frozenset({inputs})
    state = {"resolved_strategy": SearchStrategy.ITERATIVE}
    assert query_processing.needs_query_entities(state, config) is (
        inputs is QueryInput.ENTITIES
    )
    assert query_processing.needs_dual_keywords(state, config) is (
        inputs is QueryInput.DUAL_KEYWORDS
    )
    spec = get_strategy_spec(wrapped_strategy(SearchStrategy.ITERATIVE, config))
    assert spec.strategy_class.spare_context_budget_first == spare
    # It needs no artifact the base does not: the same stores and methodology.
    own = get_strategy_spec(SearchStrategy.ITERATIVE)
    assert own.required_roles == spec.required_roles
    assert own.methodology is None and own.required_artifacts is None

    strategy, *_ = _strategy(_Script(controller=[SUFFICIENT]), base=base)
    assert isinstance(strategy.base, cls) and isinstance(strategy.follow_up, cls)
    # The follow-up rounds are reranked once, together; round 0 as the base.
    assert strategy.base.rerank_types == frozenset({"text"})
    assert strategy.follow_up.rerank_types == frozenset()
    assert strategy.base.hybrid_scorer is strategy.hybrid_scorer
    assert strategy.filtered_index_prefixes(SearchStrategy.ITERATIVE) == (
        strategy.base.filtered_index_prefixes(SearchStrategy(base))
    )


async def test_a_mix_round_is_seeded_through_its_low_level_keywords() -> None:
    script = _Script(controller=[INSUFFICIENT, SUFFICIENT])
    strategy, first, later, _, _ = _strategy(script, base="mix")

    await strategy.asearch(_query(hl_keywords=["storage"], ll_keywords=["Vendor"]))

    assert first.queries[0].metadata["search_strategy"] == "mix"
    (seeded,) = later.queries
    assert (seeded.ll_keywords, seeded.hl_keywords) == (["Depot"], [])
    assert seeded.metadata["search_strategy"] == "mix"


# A bridge the first round followed, named again beside a new query.
FOLLOWED_AGAIN = _verdict(False, ["Depot"], ["Where is Harbor?"])


@pytest.mark.parametrize(("base", "reruns"), [("mix", 1), ("local", 2)])
async def test_a_round_without_a_new_bridge_reruns_only_what_it_seeds(
    base: str, reruns: int
) -> None:
    # mix seeds its graph search from keywords, and without a new bridge
    # those are the question's, which round 0 searched: the round searches
    # the text units alone. local looks its entities up by the new query.
    script = _Script(controller=[INSUFFICIENT, FOLLOWED_AGAIN, SUFFICIENT])
    strategy, _, later, documents, _ = _strategy(script, base=base)

    result = await strategy.asearch(_query(ll_keywords=["Vendor"]))

    assert len(later.queries) == reruns
    if base == "local":
        assert later.queries[1].query == "Where is Harbor?"
        assert later.queries[1].entity_focus == []
    assert [c.query for c in documents.calls] == [FOLLOW_UP, "Where is Harbor?"]
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (2, "sufficient")


LONG_QUERY = "Which logistics firm does Vendor hire for its freight shipments?"


async def test_a_mix_round_never_runs_without_keywords(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The follow-up rounds run the real mix search: without a bridge they
    # used to rerun it with no keyword at all, so it fell back to the search
    # text as a keyword or, over 50 characters, searched no entity.
    script = _Script(
        controller=[
            _verdict(False, ["Depot"], [FOLLOW_UP]),
            _verdict(False, ["Depot"], [LONG_QUERY]),
            SUFFICIENT,
        ]
    )
    documents = _Documents()
    strategy, *_ = _strategy(script, base="mix", documents=documents)
    del strategy.follow_up.asearch  # the real mix search, not the stub

    with caplog.at_level("WARNING"):
        await strategy.asearch(
            _query(hl_keywords=["storage"], ll_keywords=["Vendor", "storage site"])
        )

    assert "No keywords extracted" not in caplog.text
    entity_searches = [
        c.query
        for c in documents.calls
        if c.index_prefixes == [Config().indexing.opensearch.entities_index_prefix]
    ]
    # The graph search is seeded by the bridge in round 1, and not rerun in
    # round 2, which names no new bridge.
    assert entity_searches == ["Depot"]
    assert all(c.filters == FILTERS for c in documents.calls)


# --- auto routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "held", "routable"),
    [
        # Over mix it reads lightrag: offered while the namespace holds it.
        ("mix", {"graphrag", "lightrag"}, ["local", "mix", "global", "iterative"]),
        ("mix", {"lightrag"}, ["mix", "iterative"]),
        ("mix", {"graphrag"}, ["local", "global"]),
        # Over local it reads graphrag.
        ("local", {"graphrag"}, ["local", "global", "iterative"]),
        ("local", {"lightrag"}, ["mix"]),
    ],
)
def test_auto_routes_to_iterative_where_its_base_is_held(
    base: str, held: set[str], routable: list[str]
) -> None:
    config = Config()
    config.search.iterative_search.base = base  # type: ignore[assignment]

    # What a default run of each held methodology builds.
    built = frozenset().union(
        *(METHODOLOGY_PROFILES[Methodology(m)].required for m in held)
    )
    if "graphrag" in held:
        built |= {IndexArtifact.COMMUNITY_REPORTS}
    offered = routable_strategies(config, built)

    assert [s.value for s in offered] == routable


def test_an_operator_keeps_auto_off_iterative() -> None:
    config = Config.model_validate(
        {"search": {"auto_routable_strategies": ["local", "mix", "global"]}}
    )

    assert SearchStrategy.ITERATIVE not in routable_strategies(config)
    # It stays selectable explicitly.
    assert query_inputs(SearchStrategy.ITERATIVE, config) == frozenset(
        {QueryInput.DUAL_KEYWORDS}
    )


async def test_auto_offers_iterative_and_prefetches_what_its_base_reads() -> None:
    # Routed among global and iterative over mix: the router reads
    # iterative's guide entry, and the keyword extraction mix reads starts
    # alongside the router call (iterative declares no input of its own).
    config = Config()
    config.search.auto_routable_strategies = [
        SearchStrategy.GLOBAL,
        SearchStrategy.ITERATIVE,
    ]
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain.ignore_errors = False
    offered: dict[str, Any] = {}
    started: dict[str, bool] = {}

    class _Router:
        async def ainvoke(self, inputs: dict[str, Any], _config: Any = None) -> str:
            offered.update(inputs)
            return "iterative"

    def _start(_state: Any, *, entities: bool, keywords: bool, **_: Any) -> dict:
        started.update(entities=entities, keywords=keywords)
        return {}

    chain._get_chain_for_prompt = lambda *_a, **_k: _Router()  # type: ignore[method-assign]
    chain._start_query_inputs = _start  # type: ignore[method-assign,assignment]
    state = await chain._resolve_strategy(
        {"search_strategy": SearchStrategy.AUTO, "query": QUESTION}, prefetch={}
    )

    assert state["resolved_strategy"] is SearchStrategy.ITERATIVE
    assert offered["strategies"] == "global, iterative"
    assert "2. ITERATIVE SEARCH" in offered["strategy_descriptions"]
    assert started == {"entities": False, "keywords": True}


# --- the output guardrail ------------------------------------------------------


class _MaskingGuard:
    """Masks "Secret" in an output and refuses one naming "Forbidden"."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    async def acheck(self, text: str, source: Any, **kwargs: Any) -> Any:
        from unified_kg_rag.ports.guardrail import GuardAction, GuardVerdict

        self.checked.append(text)
        if "Forbidden" in text:
            return GuardVerdict(action=GuardAction.BLOCKED, text="Refused.")
        if "Secret" in text:
            return GuardVerdict(
                action=GuardAction.MASKED, text=text.replace("Secret", "{NAME}")
            )
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _guarded_chain(guard: _MaskingGuard) -> GraphRAGChain:
    config = Config()
    config.aws.bedrock.guardrail.identifier = "test-guardrail"
    config.providers.rerank.kind = "none"
    return GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(lambda _s, _h: "unused"),
        retriever_builders={
            "document": lambda: _Documents(),
            "graph": lambda: _Documents(),
        },
        guard=guard,  # type: ignore[arg-type]
    )


def _iterative_result(**iterative: Any) -> SearchResult:
    return SearchResult(
        query=_query(),
        results=list(ROUND_0),
        total_results=len(ROUND_0),
        search_strategy="iterative_search",
        processing_time=0.0,
        metadata={"iterative": {"rounds": 1, **iterative}},
    )


async def test_the_model_written_follow_ups_are_screened_as_outputs() -> None:
    guard = _MaskingGuard()
    chain = _guarded_chain(guard)
    records: list[dict[str, Any]] = []
    result = _iterative_result(
        searched=["Where is Secret Depot?", "Forbidden query"],
        verdicts=[
            {
                "round": 0,
                "sufficient": False,
                "bridge_entities": ["Secret Depot", "Harbor"],
                "next_queries": ["Where is Secret Depot?"],
            }
        ],
    )

    screened = await chain._screen_synthesized(result, records)

    meta = screened.metadata["iterative"]
    assert meta["searched"] == ["Where is {NAME} Depot?"]
    assert meta["verdicts"][0]["bridge_entities"] == ["{NAME} Depot", "Harbor"]
    assert meta["verdicts"][0]["next_queries"] == ["Where is {NAME} Depot?"]
    assert meta["rounds"] == 1
    assert len(records) == 3  # one per intervening string
    # The caller's result is left as it was.
    assert result.metadata["iterative"]["searched"][0] == "Where is Secret Depot?"


async def test_clean_follow_ups_cost_one_check() -> None:
    guard = _MaskingGuard()
    chain = _guarded_chain(guard)
    result = _iterative_result(
        searched=["Where is Depot?"],
        verdicts=[{"bridge_entities": ["Depot"], "next_queries": ["Where is Depot?"]}],
    )

    screened = await chain._screen_synthesized(result, [])

    assert screened.metadata == result.metadata
    assert guard.checked == ["Where is Depot?\nDepot\nWhere is Depot?"]
