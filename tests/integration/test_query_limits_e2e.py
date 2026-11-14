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
    assert out.metadata["routing"]["decided_by"] == "only_candidate"


def test_an_answer_context_the_limit_emptied_names_the_limit(
    chain: GraphRAGChain,
) -> None:
    # The limit leaves the answer's prompt but no context: the fixed reply
    # says the limit did it (abstention_reason query_limit), not "no
    # information".
    out = _ask(chain, SearchStrategy.NAIVE, QueryLimits(max_input_tokens=900), ENGLISH)
    assert out.metadata["abstention_reason"] == "query_limit", out.metadata.get(
        "degraded"
    )
    steps = [d["step"] for d in out.metadata["degraded"]]
    assert "context_optimization" in steps
