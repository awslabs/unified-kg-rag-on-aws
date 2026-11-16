# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query limits end to end on the embedded stores (``backend: local``).

Under a small limit a strategy does what fits and answers from it: DRIFT
and global reserved config ceilings (an action's local_context_tokens, the
answer's max_answer_tokens) for what the answer would read, so under any
``search.query_max_input_tokens`` below about 21k tokens DRIFT made no call
and answered "no information" (W1 verification 1); and the optional query
processing calls (a translation, an extraction, the structured planner)
took the call kept for the answer, so an explicit strategy at one call
failed where it could have answered (W1 verification 2).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import (
    ANSWER,
    DIM,
    _config,
    _errors,
    _ingest,
    _providers,
    _respond,
)
from unified_kg_rag import QueryLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    SearchOutput,
)
from unified_kg_rag.domain.models import PipelineStageStatus, SearchStrategy

pytestmark = pytest.mark.integration

# The embedded e2e's guard: no AWS client or Bedrock factory is built.
no_aws = e2e.no_aws

ENGLISH = "Who stores the gadgets for Vendor?"
KOREAN = "누가 공급업체를 위해 물건을 보관합니까?"


@pytest.fixture
def chain(tmp_path: Path, no_aws) -> GraphRAGChain:  # noqa: F811
    config = _config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(_respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    return GraphRAGChain(config=config, providers=providers)


def _ask(
    chain: GraphRAGChain, strategy: SearchStrategy, limits: QueryLimits, query: str
) -> RAGOutput:
    out = chain.invoke(RAGInput(query=query, search_strategy=strategy, limits=limits))
    assert isinstance(out, RAGOutput)
    return out


@pytest.mark.parametrize("strategy", [SearchStrategy.DRIFT, SearchStrategy.GLOBAL])
def test_fan_out_strategies_answer_with_what_fits_an_input_token_limit(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> None:
    unlimited = _ask(chain, strategy, QueryLimits(), ENGLISH)
    assert unlimited.answer == ANSWER
    full_calls = unlimited.metadata["usage"]["llm_calls"]
    for limit in (1_500, 3_000, 6_000, 10_000, 20_000):
        out = _ask(chain, strategy, QueryLimits(max_input_tokens=limit), ENGLISH)
        usage = out.metadata["usage"]
        assert out.answer == ANSWER, (strategy, limit, out.metadata.get("degraded"))
        assert out.metadata.get("abstention_reason") is None, (strategy, limit)
        assert usage["input_tokens"] <= limit, (strategy, limit)
        if limit >= 6_000:
            # The demo's whole query reads about 5k tokens: it all fits.
            assert usage["llm_calls"] == full_calls, (strategy, limit, usage)


@pytest.mark.parametrize(
    ("strategy", "query", "calls"),
    [
        (SearchStrategy.LOCAL, ENGLISH, 1),
        (SearchStrategy.MIX, ENGLISH, 1),
        (SearchStrategy.STRUCTURED, ENGLISH, 1),
        (SearchStrategy.NAIVE, KOREAN, 1),
        (SearchStrategy.ITERATIVE, KOREAN, 1),
        (SearchStrategy.ITERATIVE, KOREAN, 2),
    ],
)
def test_optional_query_calls_leave_the_answer_its_call(
    chain: GraphRAGChain, strategy: SearchStrategy, query: str, calls: int
) -> None:
    out = _ask(chain, strategy, QueryLimits(max_model_calls=calls), query)
    assert out.answer == ANSWER, (strategy, out.metadata.get("degraded"))
    assert out.metadata["usage"]["llm_calls"] <= calls
    reasons = [d["reason"] for d in out.metadata.get("degraded", [])]
    assert any(r.startswith("query_limit") for r in reasons), reasons


def test_a_limit_that_leaves_no_context_says_so(chain: GraphRAGChain) -> None:
    # An answer from no context because a limit cut every step names the
    # limit, not "no information" (W1 verification 1).
    out = _ask(
        chain, SearchStrategy.DRIFT, QueryLimits(max_input_tokens=1_300), ENGLISH
    )
    if out.metadata.get("abstention_reason") is not None:
        assert out.metadata["abstention_reason"] == "query_limit"


@pytest.mark.parametrize("limit", [800, 1_500])
def test_auto_under_a_small_token_limit_degrades_rather_than_fails(
    chain: GraphRAGChain, limit: int
) -> None:
    # W1 verification 3: the router call was refused at 800 tokens and the
    # query failed where every explicit strategy degraded.
    out = _ask(chain, SearchStrategy.AUTO, QueryLimits(max_input_tokens=limit), ENGLISH)
    assert out.answer == ANSWER or out.metadata["abstention_reason"] == "query_limit"
    routing = out.metadata["routing"]
    assert routing["decided_by"] == "only_candidate"
    # The candidates the limit dropped: what makes the query query_limited.
    assert routing["over_limits"]
    assert not set(routing["over_limits"]) & set(routing["candidates"])


def test_an_answer_context_the_limit_emptied_names_the_limit(
    chain: GraphRAGChain, caplog: pytest.LogCaptureFixture
) -> None:
    # The limit leaves the answer's prompt but no context: the fixed reply
    # says the limit did it (abstention_reason query_limit), not "no
    # information".
    out = _ask(chain, SearchStrategy.NAIVE, QueryLimits(max_input_tokens=900), ENGLISH)
    assert out.metadata["abstention_reason"] == "query_limit", out.metadata.get(
        "degraded"
    )
    steps = [d["step"] for d in out.metadata["degraded"]]
    # One entry, the limit's (rv33: a second, context_budget, came with it),
    # and no ERROR for a designed cut ("Token budget exceeded").
    assert steps.count("context_optimization") == 1, out.metadata["degraded"]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_limit_that_cuts_the_answer_context_says_so(chain: GraphRAGChain) -> None:
    # rv33: an input-token limit that cut some of the answer's sections
    # recorded nothing, so a server limit set too low lowered answers with
    # query_degraded at 0. The demo's local query reads 8 sections in about
    # 940 tokens: at 1,100 the limit (less the context-window headroom)
    # cuts some.
    full = _ask(chain, SearchStrategy.LOCAL, QueryLimits(), ENGLISH)
    assert "degraded" not in full.metadata or not full.metadata["degraded"]
    out = _ask(
        chain, SearchStrategy.LOCAL, QueryLimits(max_input_tokens=1_100), ENGLISH
    )
    assert out.answer == ANSWER
    assert len(out.sources) < len(full.sources)
    (entry,) = out.metadata["degraded"]
    assert entry["step"] == "context_optimization"
    assert re.fullmatch(r"query_limit: \d+ of \d+ sections cut", entry["reason"])


# Long questions, the fan-out's planning miss (rv33): each planned map,
# rating and action call carries the question, which the plans left out.
LONG_ENGLISH = " ".join(
    ["Who stores the gadgets for Vendor, and under which agreement and terms?"] * 25
)
LONG_KOREAN = " ".join(
    ["누가 공급업체를 위해 물건을 보관하며 어떤 계약과 조건에 따릅니까?"] * 20
)


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "strategy",
    [SearchStrategy.DRIFT, SearchStrategy.GLOBAL, SearchStrategy.ITERATIVE],
)
def test_a_long_question_degrades_at_every_token_limit_instead_of_failing(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> None:
    for query in (LONG_ENGLISH, LONG_KOREAN):
        # Past 4k tokens the demo's whole query fits.
        for limit in range(1_000, 4_001, 100):
            out = _ask(chain, strategy, QueryLimits(max_input_tokens=limit), query)
            assert out.metadata["usage"]["input_tokens"] <= limit, (strategy, limit)
            reasons = [d["reason"] for d in out.metadata.get("degraded", [])]
            assert out.answer == ANSWER or (
                out.metadata["abstention_reason"] == "query_limit"
            ), (strategy, limit, reasons)


def test_a_search_keeps_no_call_for_an_answer_it_does_not_make(
    chain: GraphRAGChain,
) -> None:
    # rv33: a search (asearch, kg_search, run-rag --mode search) kept the
    # answer's call, so at one call a Korean question searched untranslated.
    out = chain.search(
        RAGInput(
            query=KOREAN,
            search_strategy=SearchStrategy.NAIVE,
            limits=QueryLimits(max_model_calls=1),
        )
    )
    assert isinstance(out, SearchOutput)
    # The translation was asked (the fake model's reply to it is not one),
    # and its repair, optional, skipped at the limit instead of failing.
    assert out.metadata["usage"]["llm_calls"] == 1
    reasons = [d["reason"] for d in out.metadata.get("degraded", [])]
    assert not any(r.startswith("query_limit") for r in reasons), reasons


def test_drift_keeps_its_one_action_from_the_optional_calls(
    chain: GraphRAGChain,
) -> None:
    # rv33: at two calls the translation of a Korean question took the call
    # DRIFT's one action needs, so it abstained with one call spent where
    # the English question answered with two.
    for query in (ENGLISH, KOREAN):
        out = _ask(chain, SearchStrategy.DRIFT, QueryLimits(max_model_calls=2), query)
        assert out.answer == ANSWER, (query, out.metadata.get("degraded"))
        assert out.metadata["usage"]["llm_calls"] == 2


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self.records.append(dict(metrics))


@pytest.mark.parametrize(
    ("strategy", "query", "limits"),
    [
        (SearchStrategy.LOCAL, KOREAN, QueryLimits(max_model_calls=1)),
        (SearchStrategy.NAIVE, ENGLISH, QueryLimits(max_input_tokens=900)),
    ],
)
def test_a_query_its_limits_cut_short_is_counted_as_limited(
    chain: GraphRAGChain,
    strategy: SearchStrategy,
    query: str,
    limits: QueryLimits,
) -> None:
    # Regression (rv33 obs 2): a limit hit mostly degrades the answer or
    # abstains (no limit_exceeded error), which only query_degraded counted,
    # with rerank and extraction failures: nothing told a limit set too low.
    from unified_kg_rag.shared.metrics import QUERY_LIMITED_METRIC

    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    _ask(chain, strategy, QueryLimits(), query)
    limited = _ask(chain, strategy, limits, query)
    assert [r[QUERY_LIMITED_METRIC] for r in sink.records] == [0.0, 1.0], (
        limited.metadata.get("degraded"),
        limited.metadata.get("abstention_reason"),
    )
