# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``run-eval`` orders the runs of a comparison, and what the repeats
share.

Runs answered one after another put every later run on a later clock: an
A/A pair of one configuration showed latency "differences" of 0.5-5.8 s.
Interleaved (the default), each query is answered by every run, the runs'
order alternating, so a drift over the run lands on every run alike. And
with ``--repeats`` each repeat extracts the query again, so the repeats
sample the extraction's variance too. All fakes, no AWS.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.application.retrieval import query_processing, rag_chain
from unified_kg_rag.application.retrieval.rag_chain import (
    _ExtractionCache,
    extraction_cache_scope,
)
from unified_kg_rag.domain.models import Config, EvaluatorType
from unified_kg_rag.evaluation.saved_results import interleaving_warnings
from unified_kg_rag.shared.usage import record_llm_call, usage_scope

LATENCY = "latency/response_time_seconds"


class _DriftingChain:
    """Every strategy answers alike, but each call is slower than the one
    before (a service whose latency drifts over the run). Each answer makes
    one model call of ``tokens`` input tokens per strategy, and records the
    extraction cache scope it was answered in."""

    tokens = {"local": 10, "mix": 1000}

    def __init__(self) -> None:
        self.calls = 0
        self.scopes: list[tuple[str, str]] = []

    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> Any:
        self.calls += 1
        strategy = inputs.get("search_strategy") or "local"
        self.scopes.append((strategy, rag_chain._EXTRACTION_SCOPE.get()))
        record_llm_call(
            "answer",
            "fake-model",
            input_tokens=self.tokens.get(strategy, 1),
            output_tokens=1,
        )
        meta = {"search_strategy": strategy, "processing_time": self.calls * 0.01}
        return {"answer": "Vendor", "sources": [], "metadata": meta}


def _args(tmp_path: Path, *extra: str, queries: int = 30) -> Any:
    data = tmp_path / "eval.json"
    data.write_text(
        json.dumps(
            [
                {"question": f"Who ships {i}?", "answer": "Vendor"}
                for i in range(queries)
            ]
        )
    )
    return run_evaluation.CommandLineInterface().parser.parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
            "--search-strategy",
            "local,mix",
            *extra,
        ]
    )


@pytest.fixture
def eval_config(mocker) -> Config:
    mocker.patch.object(run_evaluation, "display_ascii_art")
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    return config


def _comparison(tmp_path: Path) -> dict[str, Any]:
    (path,) = (tmp_path / "out").rglob("evaluation_comparison_*.json")
    return json.loads(path.read_text())


async def test_sequential_runs_confound_latency_with_time(
    eval_config, tmp_path
) -> None:
    # The reproduction: two identical arms, one after the other, differ.
    args = _args(tmp_path, "--run-order", "sequential")
    chain = _DriftingChain()
    assert await run_evaluation.EvaluationRunner(args, chain, eval_config).run() == 0
    comparison = _comparison(tmp_path)
    delta = comparison["cost"]["deltas_vs_baseline"]["mix"][LATENCY]
    assert delta["ci_low"] > 0.25  # 30 calls later, every query
    (warning,) = [w for w in comparison["warnings"] if "not interleaved" in w]
    assert "--run-order sequential" in warning


async def test_interleaved_runs_leave_the_drift_out_of_the_latency_delta(
    eval_config, tmp_path
) -> None:
    args = _args(tmp_path)  # the default order
    chain = _DriftingChain()
    assert await run_evaluation.EvaluationRunner(args, chain, eval_config).run() == 0
    comparison = _comparison(tmp_path)
    delta = comparison["cost"]["deltas_vs_baseline"]["mix"][LATENCY]
    # One call apart either way, alternating: no drift left in the mean.
    assert delta["mean_delta"] == pytest.approx(0.0, abs=1e-9)
    assert delta["ci_low"] <= 0 <= delta["ci_high"]
    assert not [w for w in comparison["warnings"] if "not interleaved" in w]
    assert comparison["run_manifest"]["run_order"] == "interleaved"
    # Each run's usage is its own answers', not the batch's.
    summaries = sorted(
        (tmp_path / "out").rglob("evaluation_summary_*.json"), key=lambda p: p.name
    )
    answered = sorted(
        json.loads(p.read_text())["usage"]["answer_generation"]["input_tokens"]
        for p in summaries
    )
    assert answered == [300, 30000]


async def test_each_repeat_resamples_the_query_extractions(
    eval_config, tmp_path
) -> None:
    args = _args(tmp_path, "--repeats", "2", queries=2)
    chain = _DriftingChain()
    runner = run_evaluation.EvaluationRunner(args, chain, eval_config)
    assert await runner.run() == 0
    answered = chain.scopes[2:]  # after one warm-up per strategy
    assert {scope for _, scope in chain.scopes[:2]} == {""}
    scopes = {scope for _, scope in answered}
    assert scopes == {f"{runner.run_id}/repeat-1", f"{runner.run_id}/repeat-2"}
    # The arms of one repeat share its scope: their extractions pair.
    for scope in scopes:
        assert (
            sorted(s for s, sc in answered if sc == scope)
            == ["local"] * 2 + ["mix"] * 2
        )
    manifest = _comparison(tmp_path)["run_manifest"]
    assert manifest["repeat_extractions"] == "resample"


@pytest.mark.parametrize("extra", [["--repeat-extractions", "reuse"], []])
async def test_repeats_reuse_or_one_repeat_share_the_chain_cache(
    eval_config, tmp_path, extra: list[str]
) -> None:
    repeats = ["--repeats", "2"] if extra else []
    args = _args(tmp_path, *repeats, *extra, queries=2)
    chain = _DriftingChain()
    assert await run_evaluation.EvaluationRunner(args, chain, eval_config).run() == 0
    assert {scope for _, scope in chain.scopes} == {""}


def test_the_extraction_cache_is_keyed_by_its_scope() -> None:
    cache = _ExtractionCache(8)
    key = ("entities", "model", "prompt", "Who ships?", "en")
    with extraction_cache_scope("repeat-1"):
        cache.put(key, ["Vendor"])
        assert cache.get(key) == ["Vendor"]
    with extraction_cache_scope("repeat-2"):
        assert cache.get(key) is None
    assert cache.get(key) is None
    cache.put(key, ["Buyer"])
    assert cache.get(key) == ["Buyer"]
    with extraction_cache_scope("repeat-1"):
        assert cache.get(key) == ["Vendor"]


def _manifest(run_id: str | None, order: str | None) -> dict[str, Any]:
    return {
        **({"run_id": run_id} if run_id else {}),
        **({"run_order": order} if order else {}),
    }


@pytest.mark.parametrize(
    ("manifests", "expected"),
    [
        # One interleaved invocation: comparable.
        (
            [("a", _manifest("r1", "interleaved"))] * 2
            + [("b", _manifest("r1", "interleaved"))],
            None,
        ),
        # One arm: nothing compared.
        ([("a", _manifest("r1", "sequential"))], None),
        (
            [
                ("a", _manifest("r1", "sequential")),
                ("b", _manifest("r1", "sequential")),
            ],
            "--run-order sequential",
        ),
        (
            [
                ("a", _manifest("r1", "interleaved")),
                ("b", _manifest("r2", "interleaved")),
            ],
            "2 separate run-eval invocations",
        ),
        # A baseline saved before the order was recorded.
        (
            [
                ("baseline:a", _manifest(None, None)),
                ("a", _manifest("r2", "interleaved")),
            ],
            "saved before",
        ),
        ([("a", None), ("b", None)], "saved before"),
    ],
)
def test_interleaving_warnings(
    manifests: list[tuple[str, dict[str, Any] | None]], expected: str | None
) -> None:
    warnings = interleaving_warnings(manifests)
    if expected is None:
        assert warnings == []
    else:
        (warning,) = warnings
        assert expected in warning and "latency" in warning


async def test_a_baseline_from_another_run_is_warned_about(
    eval_config, tmp_path
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    args = _args(first, queries=12)
    assert (
        await run_evaluation.EvaluationRunner(args, _DriftingChain(), eval_config).run()
        == 0
    )
    second = tmp_path / "second"
    second.mkdir()
    data = first / "eval.json"
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(second / "out"),
            "--search-strategy",
            "local",
            "--baseline-results",
            *(str(p) for p in (first / "out").rglob("evaluation_results_local_*.json")),
        ]
    )
    assert (
        await run_evaluation.EvaluationRunner(args, _DriftingChain(), eval_config).run()
        == 0
    )
    comparison = _comparison(second)
    assert any("2 separate run-eval invocations" in w for w in comparison["warnings"])


class _KeywordModel:
    """The keyword extraction prompt chain: slow enough that interleaved
    arms ask while the first call is still in flight."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self.fail_first = fail_first

    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> str:
        self.calls += 1
        await asyncio.sleep(0.02)
        if self.fail_first and self.calls == 1:
            raise RuntimeError("throttled")
        return json.dumps({"high_level_keywords": ["h"], "low_level_keywords": ["l"]})


class _ExtractingChain:
    """What query_processing reads of a GraphRAGChain to extract keywords."""

    ignore_errors = False

    def __init__(self, size: int = 8, *, fail_first: bool = False) -> None:
        self.config = Config()
        self._extractions = _ExtractionCache(size)
        self.model = _KeywordModel(fail_first)

    def _extraction_key(self, prompt: Any, query: str, language: str) -> Any:
        return query_processing.extraction_key(self, prompt, query, language)

    def _cached_extraction(self, key: Any) -> Any:
        return query_processing.cached_extraction(self, key)

    def _cache_extraction(self, key: Any, value: Any) -> None:
        query_processing.cache_extraction(self, key, value)

    def _get_chain_for_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return self.model


async def _two_arms(chain: _ExtractingChain, scopes: tuple[str, str]) -> list[Any]:
    async def arm(scope: str) -> Any:
        with extraction_cache_scope(scope), usage_scope() as meter:
            keywords = await query_processing.extract_dual_keywords(
                chain, "Who ships widgets?", "English"
            )
        return keywords, meter.snapshot()["cache_hits"]

    return list(await asyncio.gather(*(arm(s) for s in scopes)))


class TestInFlightExtractions:
    """The interleaved arms of one repeat ask the same question at the same
    time: both missed the cache and each paid for its own extraction, with
    keywords of their own. In flight, the second now waits for the first."""

    async def test_concurrent_arms_of_a_scope_share_one_call(self) -> None:
        chain = _ExtractingChain()
        (first, hits_1), (second, hits_2) = await _two_arms(chain, ("r1", "r1"))
        assert chain.model.calls == 1
        assert first == second == (["h"], ["l"])
        # The payer counts the call, the other a cache hit.
        assert sorted((hits_1, hits_2)) == [0, 1]

    async def test_other_scopes_and_no_cache_extract_their_own(self) -> None:
        chain = _ExtractingChain()
        await _two_arms(chain, ("r1", "r2"))
        assert chain.model.calls == 2
        uncached = _ExtractingChain(size=0)
        await _two_arms(uncached, ("r1", "r1"))
        assert uncached.model.calls == 2

    async def test_a_failed_first_call_leaves_the_others_to_extract(self) -> None:
        chain = _ExtractingChain(fail_first=True)
        results = await asyncio.gather(
            *(
                query_processing.extract_dual_keywords(chain, "Q?", "English")
                for _ in range(2)
            ),
            return_exceptions=True,
        )
        assert chain.model.calls == 2
        assert [type(r) for r in results].count(tuple) == 1


class _OrderChain:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> Any:
        self.calls.append((inputs["query"], inputs["search_strategy"]))
        return {"answer": "x", "sources": [], "metadata": {}}


async def test_an_odd_dataset_is_balanced_over_two_repeats(eval_config) -> None:
    # ABBA over pairs of queries left an odd-sized dataset's last query
    # unbalanced (one strategy first on 2 of 3 queries). Reversed per repeat
    # too, every query's positions balance with two repeats.
    from unified_kg_rag.domain.models import EvaluationQuery
    from unified_kg_rag.evaluation.evaluation_manager import (
        AnswerArm,
        EvaluationManager,
    )

    chain = _OrderChain()
    eval_config.processing.max_concurrency = 1
    manager = EvaluationManager(eval_config, rag_chain=chain)
    queries = [EvaluationQuery(query_id=f"q{i}", question=f"Q{i}") for i in range(3)]
    arms = [
        AnswerArm(
            manager=manager,
            queries=[
                q.model_copy(update={"metadata": {"search_strategy": strategy}})
                for q in queries
            ],
            attributes={"eval.strategy": strategy, "eval.repeat": repeat},
        )
        for strategy in ("local", "mix")
        for repeat in (1, 2)
    ]
    await EvaluationManager.answer_interleaved(arms, show_progress=False)
    for i in range(3):
        asked = [s for q, s in chain.calls if q == f"Q{i}"]
        positions = {
            s: sum(p for p, t in enumerate(asked) if t == s) for s in ("local", "mix")
        }
        assert positions["local"] == positions["mix"], (i, asked)
