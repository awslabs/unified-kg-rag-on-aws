# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ingestion run's budget and its cost estimate (AWS-free).

Nothing bounded what one ingestion run could spend, and nothing told what a
corpus would cost before it ran. ``processing.max_run_model_calls`` and
``processing.max_run_usd`` stop a run at its budget the way a signal stops
it (finished stages checkpointed, responses kept in the LLM response
cache), so a re-run with the same pipeline id resumes it and pays only for
the calls not yet answered; ``run-ingestion --estimate`` parses and chunks
the corpus and counts the calls its text units decide, which a run of the
scripted model here matches within 15%.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedChatModel, fake_model_info
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.integration import test_ingestion_stages as stages
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.estimate import estimate_ingestion
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.domain.pricing import ModelPrice
from unified_kg_rag.shared.exceptions import PipelineInterrupted, RunBudgetExhausted

pytestmark = pytest.mark.integration

# What each stage's prompt opens with (the scripted responder's markers).
_KINDS = {
    "knowledge graph extraction expert": "graph_extraction",
    "graph refinement specialist": "gleaning",
    "claim extraction specialist": "claim_extraction",
    "community analysis": "community_report",
}
_SENTENCES = (
    "Vendor supplies widgets to Buyer under the supply agreement.",
    "Depot stores the widgets for Vendor near the harbor.",
    "Buyer pays Vendor within thirty days of each delivery.",
)


class _Counting:
    """The scripted responder, counting each stage's calls and the input
    tokens the usage meter estimates for them."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self._lock = threading.Lock()

    def __call__(self, system: str, human: str) -> str:
        kind = next((k for m, k in _KINDS.items() if m in system), "other")
        with self._lock:
            self.calls[kind] += 1
            self.tokens[kind] += estimate_token_count(system) + estimate_token_count(
                human
            )
        return stages._respond(system, human)


class _MeteredFactory:
    """The scripted model with the usage callback the model factories
    attach (so a run's budget sees its calls)."""

    def __init__(self, respond: Any) -> None:
        self.respond = respond

    def get_model(self, model_id: Any, **kwargs: Any) -> ScriptedChatModel:
        return ScriptedChatModel(
            respond=self.respond,
            callbacks=[UsageCallbackHandler(str(model_id), role="ingestion")],
        )

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


def _config() -> Config:
    config = stages._config()
    # The graph-dependent stages are not estimated: off, so the run's calls
    # are the estimated stages' alone.
    config.graph.community_detection.report_generation.enabled = False
    config.processing.max_concurrency = 1
    # Units of a sentence or more: each names an entity, so each is asked
    # for its claims.
    config.processing.chunking.fallback_chunk_size = 240
    return config


def _corpus(root: Path, documents: int = 12) -> Path:
    source = root / "corpus"
    source.mkdir()
    for i in range(documents):
        # Every unit's text its own: a response is served once a run.
        text = " ".join(
            f"{_SENTENCES[(i + j) % 3]} Order {i}-{j}." for j in range(3 + i)
        )
        (source / f"doc{i}.txt").write_text(text, encoding="utf-8")
    return source


class _Stores:
    """The registry and stores of one index, kept across its runs."""

    def __init__(self, config: Config) -> None:
        self.doc_status = FakeDocStatusStore()
        self.vectors = FakeVectorStore(config.indexing.opensearch)
        self.graph = FakeGraphStore()


def _pipeline(
    config: Config,
    source: Path,
    cache: Path,
    respond: Any,
    stores: _Stores | None = None,
) -> Any:
    stores = stores or _Stores(config)
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=_MeteredFactory(respond),
        embedding_factory=HashingEmbeddingFactory(),
        doc_status=stores.doc_status,
        vector_indexer_factory=lambda: stores.vectors,
        graph_indexer_factory=lambda: stores.graph,
    )
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=cache,
        ),
        source_directory=source,
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    return pipeline


def test_the_estimate_is_within_15_percent_of_a_runs_calls(tmp_path: Path) -> None:
    config = _config()
    source = _corpus(tmp_path)
    estimate = estimate_ingestion(config, source)
    counted = _Counting()
    context = _pipeline(config, source, tmp_path / "cache", counted).run(
        source, pipeline_id="estimated"
    )
    assert context.status is PipelineStageStatus.COMPLETED
    assert estimate.text_units == len(context.text_units)
    rows = {row.stage: row for row in estimate.stages}
    assert set(rows) == {"graph_extraction", "gleaning", "claim_extraction"}
    for stage, row in rows.items():
        assert row.calls == pytest.approx(counted.calls[stage], rel=0.15), stage
    # The extraction and the claims read their text and their prompt: the
    # estimate counts them as the meter does. (A gleaning call reads the
    # unit's extraction too, here a scripted one far shorter than the
    # output the estimate assumes.)
    for stage in ("graph_extraction", "claim_extraction"):
        assert rows[stage].input_tokens == pytest.approx(
            counted.tokens[stage], rel=0.15
        ), stage
    assert sum(counted.calls.values()) == pytest.approx(estimate.calls, rel=0.15)
    assert "community_detection" not in estimate.not_estimated
    assert estimate.usd is None and estimate.unpriced_models


def test_the_estimate_prices_with_models_prices(tmp_path: Path) -> None:
    config = _config()
    config.graph.community_detection.report_generation.enabled = True
    price = ModelPrice(input_per_million_tokens=2.0, output_per_million_tokens=10.0)
    config.models.prices = {"global." + config.tier_model_id("default"): price}
    estimate = estimate_ingestion(config, _corpus(tmp_path))
    assert estimate.unpriced_models == [] and estimate.usd is not None
    expected = (estimate.input_tokens * 2.0 + estimate.output_tokens * 10.0) / 1e6
    assert estimate.usd == pytest.approx(expected)
    assert "community_detection" in estimate.not_estimated


def test_a_run_stops_at_its_call_budget_and_resumes_from_the_cache(
    tmp_path: Path,
) -> None:
    source = _corpus(tmp_path)
    full = _Counting()
    _pipeline(_config(), source, tmp_path / "full", full).run(source, pipeline_id="f")
    needed = sum(full.calls.values())

    config = _config()
    config.processing.max_run_model_calls = needed // 2
    counted = _Counting()
    cache = tmp_path / "cache"
    stores = _Stores(config)
    pipeline = _pipeline(config, source, cache, counted, stores)
    with pytest.raises(RunBudgetExhausted) as stopped:
        pipeline.run(source, pipeline_id="budget")
    assert isinstance(stopped.value, PipelineInterrupted)
    assert stopped.value.limit == "processing.max_run_model_calls"
    assert "re-run with the same pipeline id" in str(stopped.value)
    # The call past the budget was not sent.
    assert sum(counted.calls.values()) == needed // 2

    # Resumed with a budget left: the responses it got are not asked again.
    config.processing.max_run_model_calls = None
    context = _pipeline(config, source, cache, counted, stores).run(
        source, pipeline_id="budget"
    )
    assert context.status is PipelineStageStatus.COMPLETED
    assert counted.calls == full.calls
    assert stores.graph.ids("entities")


def test_a_usd_budget_without_a_price_stops_at_the_first_call(tmp_path: Path) -> None:
    config = _config()
    config.processing.max_run_usd = 1.0
    counted = _Counting()
    source = _corpus(tmp_path)
    with pytest.raises(RunBudgetExhausted, match="needs a models.prices entry"):
        _pipeline(config, source, tmp_path / "cache", counted).run(
            source, pipeline_id="unpriced"
        )
    assert sum(counted.calls.values()) == 0


def test_a_usd_budget_stops_the_run_at_its_price(tmp_path: Path) -> None:
    config = _config()
    # A price that makes a few calls cost the budget.
    price = ModelPrice(input_per_million_tokens=1000.0, output_per_million_tokens=0.0)
    config.models.prices = {config.tier_model_id("default"): price}
    config.processing.max_run_usd = 10.0
    counted = _Counting()
    source = _corpus(tmp_path)
    with pytest.raises(RunBudgetExhausted, match="processing.max_run_usd"):
        _pipeline(config, source, tmp_path / "cache", counted).run(
            source, pipeline_id="priced"
        )
    sent = sum(counted.calls.values())
    assert 0 < sent < 10


def test_gleaning_and_translation_count_as_upper_bounds(tmp_path: Path) -> None:
    from unified_kg_rag.domain.models.config import LanguageCode

    config = _config()
    config.processing.gleaning.max_rounds = 3
    translation = config.processing.translation
    translation.enabled = True
    translation.target_language = LanguageCode.KO
    translation.additional_target_languages = [LanguageCode.JA]
    estimate = estimate_ingestion(config, _corpus(tmp_path))
    rows = {row.stage: row for row in estimate.stages}
    units = estimate.text_units
    # Every round of every unit, and each language a model call writes.
    assert (rows["gleaning"].calls, rows["gleaning"].bound) == (3 * units, "at most")
    assert (rows["translation"].calls, rows["translation"].bound) == (
        2 * units,
        "at most",
    )
    assert rows["graph_extraction"].bound == "exact"


def test_a_usd_budget_ignores_the_embedding_and_rerank_requests(
    tmp_path: Path,
) -> None:
    # The Bedrock clients count every request (sent_requests) under the
    # embedding and rerank roles, with no price to read: a run's USD budget
    # prices model calls and rerank search units, so a request-only entry
    # never stops it for "no price".
    from unified_kg_rag.shared import usage

    config = _config()
    model = config.tier_model_id("default")
    config.models.prices = {
        model: ModelPrice(input_per_million_tokens=1.0, output_per_million_tokens=1.0)
    }
    config.processing.max_run_usd = 10.0
    pipeline = _pipeline(config, _corpus(tmp_path), tmp_path / "cache", _Counting())
    with usage.usage_scope(limits=pipeline._run_budget()) as meter:
        usage.record_request("embedding", "amazon.titan-embed-text-v2:0")
        usage.record_request("rerank", "unknown", throttled=True)
        call = usage.start_llm_call("ingestion", model, estimated_input_tokens=100)
        call.finish(input_tokens=100, output_tokens=10)
    assert meter.total.llm_calls == 1
    assert meter.total.sent_requests == 2


class _Slow(_Counting):
    """The counting responder, each call taking a while, so a budget stop
    finds calls in flight."""

    def __call__(self, system: str, human: str) -> str:
        import time

        time.sleep(0.05)
        return super().__call__(system, human)


@pytest.mark.parametrize(("budget", "slow_cache"), [(3, False), (4, False), (2, True)])
def test_concurrent_budget_stops_resume_paying_each_call_once(
    tmp_path: Path, budget: int, slow_cache: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (W1 review L1, D1): a budget stop did not wait for the calls
    # in flight (it is a KeyboardInterrupt), so their paid responses were
    # never cached and a resume asked again, and a resume's cache hits held
    # budget slots while in flight, so a resume could stop without progress.
    import math

    if slow_cache:
        # A cache lookup that takes a while: hits in flight at once.
        from unified_kg_rag.adapters.llm.response_cache import LLMResponseCache

        lookup = LLMResponseCache.lookup

        def slow(self: Any, prompt: str, llm_string: str) -> Any:
            import time

            time.sleep(0.05)
            return lookup(self, prompt, llm_string)

        monkeypatch.setattr(LLMResponseCache, "lookup", slow)
    source = _corpus(tmp_path, documents=6)
    full = _Counting()
    _pipeline(_config(), source, tmp_path / "full", full).run(source, pipeline_id="f")
    needed = sum(full.calls.values())

    config = _config()
    config.processing.max_concurrency = 4
    config.processing.max_run_model_calls = budget
    counted = _Slow()
    cache = tmp_path / "cache"
    stores = _Stores(config)
    runs = 0
    while True:
        runs += 1
        # A run pays `budget` new calls, and the last finishes the rest.
        assert runs <= math.ceil(needed / budget) + 1, (runs, dict(counted.calls))
        try:
            context = _pipeline(config, source, cache, counted, stores).run(
                source, pipeline_id="budget"
            )
        except RunBudgetExhausted:
            continue
        break
    assert context.status is PipelineStageStatus.COMPLETED
    # Every call paid once: the responses of the calls in flight at a stop
    # were kept.
    assert counted.calls == full.calls
