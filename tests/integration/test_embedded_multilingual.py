# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One index language end to end, for Korean, Japanese and Chinese corpora.

A corpus ingested with translation disabled is indexed as written, in its
source language, whatever ``target_language`` says: every indexing prompt
(extraction, gleaning, description summaries, community reports) asks for
the source language, the graph indexes' text fields are analyzed for it,
and every query-time prompt of global, DRIFT and local search (map points,
HyDE, primer, query entities) writes in it. A query entity list a model
separates with ``、`` or ``，`` splits into its entities. The stores are the
embedded SQLite ones and the models a fake Ollama server
(``test_embedded_e2e``).
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.application.retrieval import query_processing
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import PipelineStageStatus, SearchStrategy
from unified_kg_rag.domain.models.config import LanguageCode

# Every boto3 client, Bedrock factory and DynamoDB store fails if built.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

# (language, its name in prompts, the entity names, a document, the query,
# the query entities' answer as a model writes it in that language).
_CORPORA = {
    LanguageCode.KO: (
        "Korean",
        ("공급사", "구매사", "창고"),
        "공급사는 구매사에 부품을 공급한다. 창고는 공급사의 부품을 보관한다.",
        "창고는 누구의 부품을 보관하나요?",
        "창고, 공급사",
    ),
    LanguageCode.JA: (
        "Japanese",
        ("供給者", "購入者", "倉庫"),
        "供給者は購入者に部品を供給する。倉庫は供給者の部品を保管する。",
        "倉庫は誰の部品を保管していますか？",
        "倉庫、供給者",
    ),
    LanguageCode.ZH: (
        "Chinese",
        ("供应商", "采购方", "仓库"),
        "供应商向采购方供应零件。仓库为供应商保管零件。",
        "仓库为谁保管零件？",
        "仓库，供应商",
    ),
}


class _Recorder:
    """The fake model's answers for one corpus, recording every prompt."""

    def __init__(self, names: tuple[str, ...], entities: str) -> None:
        self.names = names
        self.entities = entities
        self.prompts: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, system: str, human: str) -> str:
        with self._lock:
            self.prompts.append((system, human))
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), self.names)
        if "graph refinement specialist" in system:
            return stages._REFINEMENT
        if "claim extraction specialist" in system:
            return "<claims></claims>"
        if "specialized entity extraction expert" in system:
            return self.entities
        if "KEY POINTS from the reports" in system:
            return json.dumps({"points": [{"description": e2e.ANSWER, "score": 90}]})
        return e2e._respond(system, human)

    def asked(self, marker: str) -> list[str]:
        """The full prompts (system and human) whose system holds ``marker``."""
        return [f"{s}\n{h}" for s, h in self.prompts if marker in s]


def _mapping(engine: SQLiteSearchEngine, alias: str) -> dict[str, Any]:
    (mappings,) = asyncio.run(engine.aget_mappings(alias)).values()
    return dict(mappings["properties"])


@pytest.mark.timeout(300)
@pytest.mark.parametrize("language", list(_CORPORA), ids=lambda c: c.value)
def test_an_untranslated_corpus_is_indexed_and_queried_in_its_language(
    tmp_path: Path, language: LanguageCode
) -> None:
    name, names, text, question, entities = _CORPORA[language]
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    translation = config.processing.translation
    translation.enabled = False
    translation.source_language = language
    # The setting the prompts used to follow: a second language in the graph.
    translation.target_language = LanguageCode.EN
    config.processing.gleaning.enabled = True
    config.processing.gleaning.max_rounds = 1
    # Every merged description is summarized.
    summarization = config.processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 1
    # Global search maps even this corpus's few reports.
    config.search.global_search.enable_map_reduce = True
    config.search.global_search.map_reduce_min_results = 1
    recorder = _Recorder(names, entities)
    server = e2e.FakeOllama(recorder, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text(text, encoding="utf-8")
    (source / "b.txt").write_text(" ".join(names), encoding="utf-8")

    context = e2e._ingest(config, providers, source, tmp_path / "run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert e2e._entity_names(providers) >= set(names)

    code = f"the language with code '{language.value}'"
    for marker in ("knowledge graph extraction expert", "graph refinement specialist"):
        prompts = recorder.asked(marker)
        assert prompts, marker
        assert all(code in p and "code 'en'" not in p for p in prompts), marker
    summaries = recorder.asked("description")
    reports = recorder.asked("community analysis")
    assert reports
    assert all(f"in {name}, the language of the indexed" in p for p in reports)
    summaries = [p for p in summaries if "Write the summary in" in p]
    assert summaries
    assert all(f"Write the summary in {name}" in p for p in summaries)

    # The graph indexes are analyzed for the source language, as the text.
    analyzer = config.indexing.opensearch.language_analyzers[language.value]
    os_config = config.indexing.opensearch
    vectors = providers.build_vector_indexer()
    try:
        engine = vectors.opensearch_client
        assert isinstance(engine, SQLiteSearchEngine)
        texts = _mapping(
            engine, vectors._get_name(os_config.text_units_index_prefix, None)
        )
        entities_index = _mapping(
            engine, vectors._get_name(os_config.entities_index_prefix, None)
        )
        relationships = _mapping(
            engine, vectors._get_name(os_config.relationships_index_prefix, None)
        )
        reports_index = _mapping(
            engine, vectors._get_name(os_config.community_reports_index_prefix, None)
        )
    finally:
        vectors.close()
    assert texts["text"]["analyzer"] == analyzer
    assert entities_index["name"]["analyzer"] == analyzer
    assert entities_index["description"]["analyzer"] == analyzer
    assert relationships["description"]["analyzer"] == analyzer
    assert reports_index["full_content"]["analyzer"] == analyzer

    chain = GraphRAGChain(config, providers=providers)
    try:
        for strategy in (
            SearchStrategy.LOCAL,
            SearchStrategy.GLOBAL,
            SearchStrategy.DRIFT,
        ):
            before = len(recorder.prompts)
            out = chain.invoke(RAGInput(query=question, search_strategy=strategy))
            assert isinstance(out, RAGOutput), strategy
            asked = [f"{s}\n{h}" for s, h in recorder.prompts[before:]]
            if strategy is SearchStrategy.LOCAL:
                entity_prompts = [p for p in asked if "entity extraction expert" in p]
                assert entity_prompts
                assert all(f"Target language: {name}" in p for p in entity_prompts)
                # A list separated as the language writes it splits into its
                # names (the extraction cached by the query's).
                found = asyncio.run(
                    query_processing.extract_query_entities(chain, question, name, None)
                )
                assert found == [names[2], names[0]]
            if strategy is SearchStrategy.GLOBAL:
                points = [p for p in asked if "KEY POINTS from the reports" in p]
                assert points
                assert all(f"written in {name}" in p for p in points)
            if strategy is SearchStrategy.DRIFT:
                for marker in ("hypothetical answer", "DRIFT search primer"):
                    prompts = [p for p in asked if marker in p]
                    assert prompts, marker
                    assert all(
                        f"in {name}, the language of the indexed" in p for p in prompts
                    ), marker
    finally:
        chain.close()


def test_a_namespace_keyed_without_its_index_language_re_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Japanese source, Chinese target, translation off: extracted in Chinese
    # before #605 and in Japanese since, under the same settings and with
    # one analyzer (cjk) for both, so an incremental run mixed the two. The
    # index language keys its extraction settings: the documents extracted
    # the earlier way are extracted again.
    from collections import Counter

    from tests.fixtures.fakes.chat_models import CountingResponder
    from tests.fixtures.fakes.model_servers import FakeOllama
    from unified_kg_rag.domain.ingestion import cache_keys

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = True
    translation = config.processing.translation
    translation.enabled = False
    translation.source_language = LanguageCode.JA
    translation.target_language = LanguageCode.ZH
    calls = CountingResponder(e2e._respond)
    providers = e2e._providers(
        config, FakeOllama(calls, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in (("a", _CORPORA[LanguageCode.JA][2]), ("b", "倉庫は港にある。")):
        (source / f"{name}.txt").write_text(text, encoding="utf-8")

    def run(name: str) -> Counter[str]:
        before = Counter(calls.kinds)
        context = e2e._ingest(config, providers, source, tmp_path / name)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return calls.since(before)

    with monkeypatch.context() as earlier:
        # Keyed as before #605: no index language in the settings, and the
        # records under the language the prompts then asked for.
        earlier.setattr(cache_keys, "_language_input", lambda _config: {})
        earlier.setattr(cache_keys, "index_language", lambda _config: "zh")
        first = run("run-1")
    assert first["extraction"] == 2
    # Nothing changed but the keying: every document is extracted again.
    assert run("run-2")["extraction"] == 2
    assert run("run-3")["extraction"] == 0
