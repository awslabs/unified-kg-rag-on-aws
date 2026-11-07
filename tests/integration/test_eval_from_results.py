# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval`` comparison rebuilt from saved files, on the embedded stores.

A comparison stopped part-way and completed by a run of the missing
strategy, then rebuilt with ``--from-results``, has the deltas of the
uninterrupted comparison, and makes no answer call. The answers' usage is
the fake model server's own count. No container, no AWS.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import EvaluatorType, PipelineStageStatus
from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager

pytestmark = pytest.mark.integration

# The fixture that fails every AWS client and Bedrock factory constructor.
no_aws = e2e.no_aws

_DATASET = [
    {
        "id": "q1",
        "question": "Who stores the gadgets for Vendor?",
        "answer": "Depot",
        "reference_sources": ["logistics.txt"],
        "expected_entities": ["Depot", "Vendor"],
        "category": "1hop",
    },
    {
        "id": "q2",
        "question": "Who does Vendor supply widgets to?",
        "answer": "Buyer",
        "reference_sources": ["contract.txt"],
        "expected_entities": ["Vendor", "Buyer"],
        "category": "1hop",
    },
    {
        "id": "q3",
        "question": "Who moves gadgets for Depot?",
        "answer": "Carrier",
        "reference_sources": ["freight.txt"],
        "expected_entities": ["Carrier", "Depot"],
        "category": "2hop",
    },
]


class _Stop(BaseException):
    """Stands in for Ctrl-C part-way through a comparison."""


def _cli(chain: Any, config: Any, providers: Any, argv: list[str]) -> dict[str, Any]:
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [*argv, "--output-format", "json"]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = asyncio.run(
            run_evaluation.EvaluationRunner(args, chain, config, providers).run()
        )
    out = buffer.getvalue()
    return {"exit": code, **json.loads(out[out.find("{") :])}


@pytest.mark.parametrize("extraction_cache", [True, False], ids=["cached", "cold"])
def test_an_interrupted_comparison_completed_and_rebuilt_has_the_same_deltas(
    tmp_path: Path,
    no_aws: None,
    extraction_cache: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One chain answers every run, so with its query extraction cache on
    # (the default) the later runs reuse the first one's extractions: their
    # usage is lower, as recorded cache hits. The quality deltas never
    # depend on it; the usage deltas match exactly with the cache off.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    # On three passages the PageRank's passage search gives local what mix
    # and naive find; the support order keeps the strategies apart.
    config.search.local_search.text_unit_ranking = "support"
    if not extraction_cache:
        config.search.query_extraction_cache_size = 0
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores gadgets for Vendor.")
    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    config.evaluation.enabled_evaluators = [
        EvaluatorType.RETRIEVAL,
        EvaluatorType.ANSWER_MATCH,
        EvaluatorType.GRAPH_AWARE,
    ]
    # The warm-up queries' calls are in no run's usage: skipped, the server's
    # call count is the runs' usage exactly.

    async def _no_warm_up(self: Any, query: Any) -> None:
        return None

    monkeypatch.setattr(EvaluationManager, "warm_up", _no_warm_up)
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps(_DATASET), encoding="utf-8")
    # Run by run, so the stopped comparison keeps its finished runs (an
    # interleaved one finishes every run at once, or none).
    strategies = [
        "--search-strategy",
        "local,mix,naive",
        "--run-order",
        "sequential",
        "--data-path",
        str(dataset),
    ]
    chain = GraphRAGChain(config, providers=providers)
    try:
        calls = len(server.chat_models)
        full = _cli(
            chain,
            config,
            providers,
            [*strategies, "--output-directory", str(tmp_path / "full")],
        )
        assert full["exit"] == 0
        # Each run's answers used exactly the calls and tokens the server saw.
        answered = sum(
            r["usage"]["answer_generation"]["llm_calls"] for r in full["runs"]
        )
        assert answered == len(server.chat_models) - calls

        # Stopped in the third strategy's run: its files are never written.
        stopped = tmp_path / "stopped"
        answer = chain.ainvoke

        async def _stop_in_naive(inp: Any, *args: Any, **kwargs: Any) -> Any:
            if inp.get("search_strategy") == "naive":
                raise _Stop
            return await answer(inp, *args, **kwargs)

        object.__setattr__(chain, "ainvoke", _stop_in_naive)
        with pytest.raises(_Stop):
            _cli(
                chain,
                config,
                providers,
                [*strategies, "--output-directory", str(stopped)],
            )
        object.__setattr__(chain, "ainvoke", answer)
        assert not list(stopped.rglob("evaluation_results_naive_*"))
        # The missing strategy alone, into the same directory.
        _cli(
            chain,
            config,
            providers,
            [
                "--search-strategy",
                "naive",
                "--data-path",
                str(dataset),
                "--output-directory",
                str(stopped),
            ],
        )
    finally:
        chain.close()

    calls = len(server.chat_models)
    rebuilt = _cli(
        None,
        config,
        providers,
        [
            "--from-results",
            str(stopped),
            "--output-directory",
            str(tmp_path / "rebuilt"),
        ],
    )
    assert rebuilt["exit"] == 0
    assert len(server.chat_models) == calls  # no answer, no judge call
    expected, got = full["comparison"], rebuilt["comparison"]
    assert got["baseline"] == "local"
    assert list(got["strategies"]) == ["local", "mix", "naive"]
    assert json.dumps(got["deltas_vs_baseline"], sort_keys=True) == json.dumps(
        expected["deltas_vs_baseline"], sort_keys=True
    )
    assert got["deltas_by_category"] == expected["deltas_by_category"]
    # Not vacuous: the strategies differ on some query.
    assert any(
        delta["per_query"] and any(delta["per_query"].values())
        for by_metric in got["deltas_vs_baseline"].values()
        for delta in by_metric.values()
    )
    hits = sum(c["cache_hits"] for c in got["cost"]["strategies"].values())
    if extraction_cache:
        # Each extraction is made once or served once: calls + hits agree.
        assert hits > 0
        for strategy in ("local", "mix", "naive"):
            then, now = (c["cost"]["strategies"][strategy] for c in (expected, got))
            assert (
                then["totals"]["llm_calls"] + then["cache_hits"]
                == now["totals"]["llm_calls"] + now["cache_hits"]
            ), strategy
        return
    assert hits == 0
    for strategy in ("mix", "naive"):
        for measure in ("usage/llm_calls", "usage/input_tokens", "usage/output_tokens"):
            assert (
                got["cost"]["deltas_vs_baseline"][strategy][measure]
                == expected["cost"]["deltas_vs_baseline"][strategy][measure]
            )
