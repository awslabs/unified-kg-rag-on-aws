# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HippoRAG 2's PageRank (``ppr``, ``local``'s ranking) on the embedded stores.

A multi-hop corpus ingested end to end (the ``test_embedded_e2e`` harness:
fake Ollama server, SQLite stores, no community reports): the question's
facts name two people, and only the bridge passage cites both. Decoy
passages repeat the question's words and cite the facts' other entities, so
the hybrid (BM25 + vector) passage search ranks them first; the PageRank
ranks the bridge passage first, in English, Korean, Japanese and Chinese.
The answers cite their sources with the provenance every strategy reports,
and the retrieval evaluator scores them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws as _no_aws  # noqa: F401
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.search_strategies.ppr_search import (
    PPR_PASSAGES,
    PPRSearchStrategy,
)
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    Methodology,
    PipelineStageStatus,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.ports.guardrail import GuardAction, GuardSource, GuardVerdict
from unified_kg_rag.shared.exceptions import (
    MethodologyNotIndexedError,
    QueryTimeoutError,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("_no_aws")]

# Per language: the people (A founded C; B is the mayor of D), the files and
# the question. accord.txt, the bridge, is the only passage citing A and B;
# each decoy repeats the question's words and names C alone (no fact), so a
# seed cites it. A passage is one chunk (under 60 characters).
_CORPORA: dict[LanguageCode, dict[str, Any]] = {
    LanguageCode.EN: {
        "names": ("Alvar Pell", "Corin Works", "Bex Lund", "Dorrit"),
        "files": {
            "founder.txt": "Alvar Pell founded Corin Works.",
            "mayor.txt": "Bex Lund is the mayor of Dorrit.",
            "accord.txt": "Alvar Pell and Bex Lund signed the Tarn Accord.",
            **{
                f"decoy{i}.txt": f"Corin Works founder and mayor: what was signed? {i}"
                for i in range(4)
            },
        },
        "question": "What did the founder of Corin Works and the mayor of Dorrit sign?",
    },
    LanguageCode.KO: {
        "names": ("홍길동", "가나다상사", "김철수", "도리시"),
        "files": {
            "founder.txt": "홍길동은 가나다상사를 설립했다.",
            "mayor.txt": "김철수는 도리시의 시장이다.",
            "accord.txt": "홍길동과 김철수는 타른협정에 서명했다.",
            **{
                f"decoy{i}.txt": f"가나다상사 설립자와 시장이 서명한 것은 무엇인가 {i}"
                for i in range(4)
            },
        },
        "question": "가나다상사 설립자와 도리시 시장이 서명한 것은 무엇인가?",
    },
    LanguageCode.JA: {
        "names": ("山田太郎", "青葉商事", "佐藤花子", "緑市"),
        "files": {
            "founder.txt": "山田太郎は青葉商事を設立した。",
            "mayor.txt": "佐藤花子は緑市の市長である。",
            "accord.txt": "山田太郎と佐藤花子は北浜協定に署名した。",
            **{
                f"decoy{i}.txt": f"青葉商事の設立者と市長が署名したものは何か{i}"
                for i in range(4)
            },
        },
        "question": "青葉商事の設立者と緑市の市長が署名したものは何か？",
    },
    LanguageCode.ZH: {
        "names": ("王小明", "青松公司", "李华", "绿城"),
        "files": {
            "founder.txt": "王小明创立了青松公司。",
            "mayor.txt": "李华是绿城的市长。",
            "accord.txt": "王小明和李华签署了北海协定。",
            **{
                f"decoy{i}.txt": f"青松公司的创立者和市长签署了什么{i}"
                for i in range(4)
            },
        },
        "question": "青松公司的创立者和绿城的市长签署了什么？",
    },
}


def _extraction(text: str, names: tuple[str, ...]) -> str:
    """The fake model's extraction: the known names in the text, each pair
    of consecutive names a relationship described by the sentence itself."""
    found = [name for name in names if name in text]
    entities = "".join(
        f"<entity><name>{n}</name><type>ORGANIZATION</type>"
        f"<description>{n}</description><confidence>9</confidence>"
        f"<source_text>{n}</source_text></entity>"
        for n in found
    )
    relationships = "".join(
        f"<relationship><source>{a}</source><target>{b}</target>"
        f"<type>RELATED_TO</type><description>{text.strip()}</description>"
        f"<strength>8</strength><source_text>{text.strip()}</source_text>"
        "</relationship>"
        for a, b in zip(found, found[1:], strict=False)
    )
    return (
        f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    )


def _setup(
    tmp_path: Path,
    language: LanguageCode,
    methodologies: tuple[Methodology, ...] = (Methodology.HIPPORAG,),
    *,
    relationship_vectors: bool = True,
    **search: Any,
) -> tuple[Config, Any]:
    corpus = _CORPORA[language]
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    # HippoRAG reads no community report: by default an index built for it
    # alone, without them.
    config.indexing.methodologies = list(methodologies)
    config.indexing.opensearch.build_relationship_vector_index = relationship_vectors
    config.processing.translation.source_language = language
    config.processing.translation.target_language = language
    config.providers.rerank.kind = "none"
    for key, value in search.items():
        section, field = key.split("__")
        setattr(getattr(config.search, section), field, value)
    names = corpus["names"]

    def respond(system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return _extraction(stages._source_text(human), names)
        if "specialized entity extraction expert" in system:
            # local's query entities: the facts' other entities (C and D).
            return f"{names[1]}, {names[3]}"
        return e2e._respond(system, human)

    server = FakeOllama(respond, e2e.ANSWER, e2e.DIM)
    transport = httpx.MockTransport(server)
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
    )
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in corpus["files"].items():
        (source / name).write_text(text, encoding="utf-8")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return config, providers


def _file_names(results: list[Any]) -> list[str]:
    names = []
    for result in results:
        attributes = (result.metadata or {}).get("attributes") or {}
        names.append(str(attributes.get("file_name") or ""))
    return names


_LANGUAGES = [LanguageCode.EN, LanguageCode.KO, LanguageCode.JA, LanguageCode.ZH]
# The passages the answer needs: both facts and the bridge.
_SUPPORT = {"founder.txt", "mayor.txt", "accord.txt"}


def _decoys_after(names: list[str], name: str) -> bool:
    """Whether ``name`` comes before every decoy in ``names``."""
    position = names.index(name)
    return all(
        position < i for i, other in enumerate(names) if other.startswith("decoy")
    )


@pytest.mark.parametrize("language", _LANGUAGES, ids=lambda code: code.value)
def test_ppr_ranks_the_bridge_passage_above_the_decoys(
    tmp_path: Path, language: LanguageCode
) -> None:
    config, providers = _setup(tmp_path, language)
    question = _CORPORA[language]["question"]
    chain = GraphRAGChain(config, providers=providers)
    try:
        naive = chain.search(
            RAGInput(query=question, search_strategy=SearchStrategy.NAIVE)
        ).search_results
        searched = chain.search(
            RAGInput(query=question, search_strategy=SearchStrategy.PPR)
        ).search_results
        out = chain.invoke(RAGInput(query=question, search_strategy=SearchStrategy.PPR))
        # Without the passage-hit bucket the fused order is the PageRank's.
        config.search.fusion.fusion_weights = {"ppr_seed_text_units": 0.0}
        pagerank = chain.search(
            RAGInput(query=question, search_strategy=SearchStrategy.PPR)
        ).search_results
    finally:
        chain.close()

    # The hybrid passage search alone ranks the bridge below every decoy ...
    naive_names = _file_names([r for r in naive.results if r.retriever_type == "text"])
    assert not _decoys_after(naive_names, "accord.txt"), naive_names
    # ... the PageRank, seeded by the facts, above every one.
    ranked = _file_names([r for r in pagerank.results if r.retriever_type == "text"])
    assert _decoys_after(ranked, "accord.txt"), ranked
    meta = searched.metadata
    assert meta["methodology"] == "hipporag"
    assert meta["fact_count"] >= 2 and meta["ppr"]["converged"]
    assert meta["ppr"]["relationships"] >= 3
    assert meta["sources"][PPR_PASSAGES] >= 1
    names = _file_names([r for r in searched.results if r.retriever_type == "text"])
    assert _SUPPORT <= set(names), names

    # The answer cites the passages and facts it saw, with their provenance.
    assert isinstance(out, RAGOutput) and out.answer == e2e.ANSWER
    assert out.metadata["search_strategy"] == "ppr"
    kinds = {source["metadata"]["section_type"] for source in out.sources}
    assert kinds == {"text", "relationship"}
    for source in out.sources:
        metadata = source["metadata"]
        if metadata["section_type"] == "text":
            # As every strategy's text source: the chunk, its file.
            assert metadata["chunk_id"] == source["source"]
            assert metadata["attributes"]["file_name"] in _CORPORA[language]["files"]
            assert {"document_ids", "attributes", "id", "text"} <= set(metadata)
        else:
            assert metadata["text_unit_ids"]
            assert {"source_id", "target_id", "description"} <= set(metadata)
    usage = out.metadata["usage"]
    assert usage["llm_calls"] == 1  # the answer: no query-time extraction


def test_local_ppr_ranking_and_the_retrieval_evaluator(tmp_path: Path) -> None:
    # local's ranking stays a graphrag strategy: the default methodologies.
    config, providers = _setup(
        tmp_path,
        LanguageCode.EN,
        (Methodology.GRAPHRAG, Methodology.LIGHTRAG),
        local_search__text_unit_ranking="ppr",
    )
    question = _CORPORA[LanguageCode.EN]["question"]
    chain = GraphRAGChain(config, providers=providers)
    try:
        local = chain.invoke(
            RAGInput(query=question, search_strategy=SearchStrategy.LOCAL)
        )
        assert isinstance(local, RAGOutput)
        meta = local.metadata
        assert meta["text_unit_ranking"] == "ppr"
        assert meta["ppr"]["passage_seeds"] >= 1 and meta["ppr"]["converged"]
        cited = [
            s["metadata"].get("attributes", {}).get("file_name")
            for s in local.sources
            if s["metadata"]["section_type"] == "text"
        ]
        assert "accord.txt" in cited, cited

        # run-eval's retrieval evaluator resolves the sources to files.
        config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        config.evaluation.outputs_directory = str(tmp_path / "eval")
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(
                query_id=strategy.value,
                question=question,
                metadata={"search_strategy": strategy.value},
            )
            for strategy in (SearchStrategy.PPR, SearchStrategy.LOCAL)
        ]
        truths = [
            EvaluationGroundTruth(
                query_id=q.query_id, ground_truth="", reference_sources=["accord.txt"]
            )
            for q in queries
        ]
        results, _, summary = asyncio.run(
            manager.evaluate_dataset(queries, truths, show_progress=False)
        )
    finally:
        chain.close()
    assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
    for result in results:
        assert any("accord.txt" in rank for rank in result.retrieved_source_ids)


class _BlockingGuard:
    """A ContentGuardPort that blocks one source and passes the other."""

    def __init__(self, blocked: GuardSource) -> None:
        self.blocked = blocked
        self.calls: list[GuardSource] = []

    async def acheck(self, text: str, source: GuardSource, **_: Any) -> GuardVerdict:
        self.calls.append(source)
        if source is self.blocked:
            return GuardVerdict(action=GuardAction.BLOCKED, text="Blocked.")
        return GuardVerdict(action=GuardAction.NONE, text=text)


@pytest.mark.parametrize("blocked", [GuardSource.INPUT, GuardSource.OUTPUT])
def test_the_guardrail_screens_a_ppr_query_and_its_answer(
    tmp_path: Path, blocked: GuardSource
) -> None:
    config, providers = _setup(tmp_path, LanguageCode.EN)
    guard = _BlockingGuard(blocked)
    chain = GraphRAGChain(config, providers=providers, guard=guard)
    searched: list[Any] = []
    search = PPRSearchStrategy.asearch

    async def spy(self: Any, query: Any, config: Any = None) -> Any:
        searched.append(query)
        return await search(self, query, config)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PPRSearchStrategy, "asearch", spy)
            out = chain.invoke(
                RAGInput(
                    query=_CORPORA[LanguageCode.EN]["question"],
                    search_strategy=SearchStrategy.PPR,
                )
            )
    finally:
        chain.close()
    assert isinstance(out, RAGOutput)
    assert out.answer == "Blocked." and out.sources == []
    assert out.metadata["guardrail"]
    # A blocked question is never searched; a blocked answer was.
    assert bool(searched) is (blocked is GuardSource.OUTPUT)


def test_a_ppr_query_stops_at_its_deadline(tmp_path: Path) -> None:
    config, providers = _setup(tmp_path, LanguageCode.EN)
    config.search.query_timeout_seconds = 0.5
    chain = GraphRAGChain(config, providers=providers)

    async def stalled(self: Any, *_: Any, **__: Any) -> Any:
        await asyncio.sleep(30)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PPRSearchStrategy, "_neighbourhood_edges", stalled)
            with pytest.raises(QueryTimeoutError):
                chain.invoke(
                    RAGInput(
                        query=_CORPORA[LanguageCode.EN]["question"],
                        search_strategy=SearchStrategy.PPR,
                    )
                )
    finally:
        chain.close()


def test_ppr_answers_on_a_lightrag_only_namespace(tmp_path: Path) -> None:
    # hipporag reads what lightrag builds: a [lightrag] namespace holds it.
    config, providers = _setup(tmp_path, LanguageCode.EN, (Methodology.LIGHTRAG,))
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = chain.invoke(
            RAGInput(
                query=_CORPORA[LanguageCode.EN]["question"],
                search_strategy=SearchStrategy.PPR,
            )
        )
    finally:
        chain.close()
    assert isinstance(out, RAGOutput) and out.answer == e2e.ANSWER
    assert "accord.txt" in {
        s["metadata"].get("attributes", {}).get("file_name") for s in out.sources
    }


def test_ppr_is_refused_where_hipporag_is_not_held(tmp_path: Path) -> None:
    # graphrag alone without the relationship vector index lacks what
    # hipporag requires: ppr fails before any model call; local, whose PPR
    # ranking is graphrag's, still answers (without relationship seeds).
    config, providers = _setup(
        tmp_path,
        LanguageCode.EN,
        (Methodology.GRAPHRAG,),
        relationship_vectors=False,
        local_search__text_unit_ranking="ppr",
    )
    question = _CORPORA[LanguageCode.EN]["question"]
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError, match="hipporag"):
            chain.invoke(RAGInput(query=question, search_strategy=SearchStrategy.PPR))
        local = chain.invoke(
            RAGInput(query=question, search_strategy=SearchStrategy.LOCAL)
        )
    finally:
        chain.close()
    assert isinstance(local, RAGOutput) and local.answer == e2e.ANSWER
    assert local.metadata["text_unit_ranking"] == "ppr"
    assert local.metadata["ppr"]["fact_seeds"] == 0
