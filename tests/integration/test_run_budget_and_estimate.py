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
    factory: _MeteredFactory | None = None,
) -> Any:
    stores = stores or _Stores(config)
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=factory or _MeteredFactory(respond),
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


def test_the_models_a_usd_budget_must_price_cover_every_model_a_run_calls(
    tmp_path: Path,
) -> None:
    """run-ingestion refuses a processing.max_run_usd run before any stage
    when a model it calls has no price (rv42): the roles it checks
    (run_model_roles) must hold every model the run then calls."""
    from unified_kg_rag.domain.ingestion.estimate import run_model_roles
    from unified_kg_rag.domain.ingestion.methodologies import selected_stages
    from unified_kg_rag.domain.models import ModelRole

    config = _config()
    config.graph.community_detection.report_generation.enabled = True
    config.processing.graph_extraction.description_summarization.enabled = True
    for role in ModelRole:
        setattr(config.models.roles, role.value, f"model-{role.value}")
    called: set[str] = set()

    class _Recording(_MeteredFactory):
        def get_model(self, model_id: Any, **kwargs: Any) -> ScriptedChatModel:
            called.add(str(model_id))
            return super().get_model(model_id, **kwargs)

    source = _corpus(tmp_path, documents=4)
    pipeline = _pipeline(
        config,
        source,
        tmp_path / "cache",
        stages._respond,
        factory=_Recording(stages._respond),
    )
    context = pipeline.run(source, pipeline_id="roles")
    assert context.status is PipelineStageStatus.COMPLETED
    checked = {
        f"model-{r.value}" for r in run_model_roles(config, selected_stages(config))
    }
    assert called and called <= checked, called - checked


def test_every_model_role_a_stage_calls_is_one_its_budget_prices() -> None:
    """STAGE_MODEL_ROLES mapped the description summary to the resolution
    only, but the community detection and the indexing summarize too: a run
    of either without the resolution passed the budget's price check and
    could stop at its first summary (rv43). Every ``ModelRole`` a stage's
    code reaches is mapped to that stage: the stage class and every class
    or function of the package's adapters and application it names, by
    name, as ``module.name`` or by a lazy import, followed transitively
    (other stages aside).
    Only the classes were followed, so a role used in a module-level helper,
    or in a class another module's function builds, was missed (rv44)."""
    import importlib
    import inspect
    import re
    import types

    from unified_kg_rag.application.ingestion.pipeline_stages import PipelineStage
    from unified_kg_rag.domain.ingestion.estimate import (
        STAGE_MODEL_ROLES,
        run_model_roles,
    )
    from unified_kg_rag.domain.models import ModelRole, PipelineStageType

    packages = ("unified_kg_rag.adapters", "unified_kg_rag.application")

    def followed(value: object) -> bool:
        value = inspect.unwrap(value) if callable(value) else value
        return (
            (inspect.isclass(value) or inspect.isfunction(value))
            and getattr(value, "__module__", "").startswith(packages)
            and not (inspect.isclass(value) and issubclass(value, PipelineStage))
        )

    def reached(code: object, seen: set[object]) -> set[ModelRole]:
        code = inspect.unwrap(code) if callable(code) else code
        if code in seen:
            return set()
        seen.add(code)
        source = inspect.getsource(code)  # type: ignore[arg-type]
        roles = {ModelRole[m] for m in re.findall(r"ModelRole\.([A-Z_]+)", source)}
        scope = dict(vars(inspect.getmodule(code)))
        # The names a function imports where it runs (a lazy import).
        for module_name, grouped, single in re.findall(
            r"^\s*from (unified_kg_rag[\w.]*) import (?:\(([\w\s,]+)\)|([\w ,]+)$)",
            source,
            re.MULTILINE,
        ):
            module = importlib.import_module(module_name)
            for name in re.findall(r"\w+", grouped or single):
                if not hasattr(module, name):  # a submodule not imported yet
                    importlib.import_module(f"{module_name}.{name}")
                scope[name] = getattr(module, name)
        names = [scope.get(name) for name in set(re.findall(r"\b\w+\b", source))]
        for owner, name in set(re.findall(r"\b(\w+)\.(\w+)\b", source)):
            if isinstance(module := scope.get(owner), types.ModuleType):
                names.append(getattr(module, name, None))
        for named in names:
            if followed(named):
                roles |= reached(named, seen)
        return roles

    calling = {}
    for stage, cls in DataIngestionPipeline.STAGE_CLASSES.items():
        # The output fixer repairs any stage's output: added to every run.
        roles = reached(cls, set()) - {ModelRole.FIXING}
        mapped = {role for role, _ in STAGE_MODEL_ROLES.get(stage, ())}
        assert roles == mapped, (stage, roles, mapped)
        if roles:
            calling[stage] = roles
    # The scan finds the stages that call a model (it is no vacuous pass).
    assert ModelRole.DESCRIPTION_SUMMARY in calling[PipelineStageType.INDEXING]
    assert len(calling) == 9
    # A run without the resolution prices the summaries the others make.
    config = _config()
    config.processing.graph_extraction.description_summarization.enabled = True
    assert ModelRole.DESCRIPTION_SUMMARY in run_model_roles(
        config, [PipelineStageType.INDEXING]
    )


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
    assert "re-run with the same pipeline id ('budget')" in str(stopped.value)
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


def test_the_first_budget_a_run_stops_at_is_the_one_it_reports(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The threads whose calls the budget refuses after the stop (their
    # calls in flight, or the other budget) must not rename the stop: the
    # run reports, and logs once, the budget that stopped it.
    config = _config()
    source = _corpus(tmp_path, documents=1)
    pipeline = _pipeline(config, source, tmp_path / "cache", _Counting())
    with caplog.at_level("WARNING"):
        pipeline._stop_for_budget("calls spent", "processing.max_run_model_calls", 4)
        later = pipeline._stop_for_budget("usd spent", "processing.max_run_usd", 1.0)
    assert later.limit == "processing.max_run_usd"  # what that thread raises
    stop = pipeline._stop_error()
    assert isinstance(stop, RunBudgetExhausted)
    assert (stop.limit, stop.value, str(stop)) == (
        "processing.max_run_model_calls",
        4,
        "calls spent",
    )
    stopping = [r for r in caplog.records if "Stopping the run" in r.getMessage()]
    assert [r.getMessage() for r in stopping] == ["Stopping the run: calls spent"]


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
    assert rows["gleaning"].calls == 3 * units
    assert (rows["translation"].calls, rows["translation"].bound) == (
        2 * units,
        "at most",
    )
    # Translated into Korean, the extraction and gleaning read the Korean
    # text, which the estimate cannot measure before it is written: it was
    # labelled exact at the English text's tokens.
    assert rows["graph_extraction"].bound == "about"
    assert rows["gleaning"].bound == "about"


def test_the_extraction_is_exact_when_the_index_language_is_the_sources(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.domain.models.config import LanguageCode

    config = _config()
    config.processing.claim_extraction.enabled = True
    translation = config.processing.translation
    translation.enabled = True
    translation.target_language = translation.source_language
    translation.additional_target_languages = [LanguageCode.JA]
    corpus = _corpus(tmp_path)
    rows = {r.stage: r for r in estimate_ingestion(config, corpus).stages}
    assert rows["graph_extraction"].bound == "exact"
    assert rows["claim_extraction"].bound == "at most"
    translation.target_language = LanguageCode.KO
    rows = {r.stage: r for r in estimate_ingestion(config, corpus).stages}
    assert rows["claim_extraction"].bound == "about"


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


class _RecordingSink:
    """A metrics sink keeping every record it is sent."""

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self.records.append(dict(metrics))

    def calls(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in self.records:
            for name, value in record.items():
                if name.startswith("llm_calls_"):
                    counts[name.removeprefix("llm_calls_")] += int(value)
        return counts


@pytest.mark.parametrize("budget", [5, 10])
def test_a_budget_stop_publishes_and_reports_the_calls_it_paid_for(
    tmp_path: Path, budget: int
) -> None:
    # Regression (rv33 obs 1): the stopped stage's usage was dropped with the
    # stop (a KeyboardInterrupt), so its paid calls reached no llm_calls_*
    # metric and no token_usage, and the resume counted them as cache hits.
    source = _corpus(tmp_path)
    config = _config()
    config.processing.max_run_model_calls = budget
    counted = _Counting()
    cache = tmp_path / "cache"
    stores = _Stores(config)
    pipeline = _pipeline(config, source, cache, counted, stores)
    sink = _RecordingSink()
    pipeline.metrics_sink = sink
    with pytest.raises(RunBudgetExhausted):
        pipeline.run(source, pipeline_id="budget")
    assert sum(counted.calls.values()) == budget
    assert sink.calls() == {"graph_extraction": budget}
    assert pipeline.token_usage is not None
    assert pipeline.token_usage["llm_calls"] == budget
    assert pipeline.token_usage["stages"]["graph_extraction"]["llm_calls"] == budget
    # The stopped stage's record keeps its calls (the console's job card).
    saved = pipeline.state_manager.load_pipeline_metadata("budget")
    (stopped,) = [
        r for r in saved["stage_results"] if r["stage_name"] == "graph_extraction"
    ]
    assert stopped["status"] == PipelineStageStatus.RUNNING.value
    assert stopped["metrics"]["llm_usage"]["llm_calls"] == budget

    # The resume publishes the calls it pays: both runs' sum is what was paid.
    config.processing.max_run_model_calls = None
    resumed = _pipeline(config, source, cache, counted, stores)
    resumed.metrics_sink = sink
    context = resumed.run(source, pipeline_id="budget")
    assert context.status is PipelineStageStatus.COMPLETED
    assert sink.calls() == counted.calls


_CHUNKING_MARKER = "intelligent semantic chunking"


class _Chunking(_Counting):
    """The counting responder, also answering the intelligent chunker's
    boundary calls (no cut)."""

    def __call__(self, system: str, human: str) -> str:
        if _CHUNKING_MARKER not in system:
            return super().__call__(system, human)
        with self._lock:
            self.calls["text_chunking"] += 1
            self.tokens["text_chunking"] += estimate_token_count(
                system
            ) + estimate_token_count(human)
        return '<?xml version="1.0" encoding="UTF-8"?>\n<chunk_boundaries>\n</chunk_boundaries>'


def _lined_corpus(root: Path, documents: int = 6) -> Path:
    """Documents of many short lines, en, ko and ja: a boundary call
    numbers every line of its pre-chunk."""
    lines = ("Vendor ships.", "벤더가 보낸다.", "ベンダーが送る。", "Buyer pays.")
    source = root / "corpus"
    source.mkdir()
    for i in range(documents):
        text = "\n".join(f"{lines[(i + j) % 4]} {i}-{j}" for j in range(120 + i))
        (source / f"doc{i}.txt").write_text(text, encoding="utf-8")
    return source


def test_the_estimate_counts_the_intelligent_chunkers_boundary_calls(
    tmp_path: Path,
) -> None:
    # The default chunker: a boundary call per pre-chunk worth one, its
    # input the pre-chunk with a number on each line; the units are those of
    # a chunking no boundary answer cut, so the per-unit stages are "about".
    from unified_kg_rag.domain.models.config import ChunkingStrategy

    config = _config()
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.INTELLIGENT
    chunking.min_chunk_size = 40
    chunking.max_chunk_size = 2000
    # Pre-chunks of many lines: their numbers are a share of a call's input.
    chunking.pre_chunk_size = 1500
    chunking.pre_chunk_overlap = 0
    source = _lined_corpus(tmp_path)
    estimate = estimate_ingestion(config, source)
    counted = _Chunking()
    context = _pipeline(config, source, tmp_path / "cache", counted).run(
        source, pipeline_id="intelligent"
    )
    assert context.status is PipelineStageStatus.COMPLETED
    rows = {row.stage: row for row in estimate.stages}
    chunked = rows["text_chunking"]
    assert counted.calls["text_chunking"] > 1
    assert (chunked.calls, chunked.bound) == (counted.calls["text_chunking"], "exact")
    # Within 2%: the line numbers are about 5% of these calls' input.
    assert chunked.input_tokens == pytest.approx(
        counted.tokens["text_chunking"], rel=0.02
    )
    assert rows["graph_extraction"].bound == "about"
    assert rows["gleaning"].bound == "at most"


def test_the_estimate_translates_into_no_source_language(tmp_path: Path) -> None:
    from unified_kg_rag.domain.models.config import LanguageCode

    config = _config()
    translation = config.processing.translation
    translation.enabled = True
    translation.source_language = LanguageCode.EN
    translation.target_language = LanguageCode.KO
    # The source language among the targets: its text is the unit's own.
    translation.additional_target_languages = [LanguageCode.EN, LanguageCode.JA]
    estimate = estimate_ingestion(config, _corpus(tmp_path))
    rows = {row.stage: row for row in estimate.stages}
    assert rows["translation"].calls == 2 * estimate.text_units


def test_the_estimate_drops_the_copies_deduplication_drops(tmp_path: Path) -> None:
    config = _config()
    source = _corpus(tmp_path, documents=4)
    for i in range(4):
        text = (source / f"doc{i}.txt").read_text(encoding="utf-8")
        (source / f"copy{i}.txt").write_text(text + " Note.", encoding="utf-8")
    config.processing.deduplicate = True
    estimate = estimate_ingestion(config, source)
    counted = _Counting()
    context = _pipeline(config, source, tmp_path / "cache", counted).run(
        source, pipeline_id="deduplicated"
    )
    assert context.status is PipelineStageStatus.COMPLETED
    assert estimate.documents == len(context.documents) == 4
    assert estimate.text_units == len(context.text_units)
    assert estimate.calls == pytest.approx(sum(counted.calls.values()), rel=0.15)
