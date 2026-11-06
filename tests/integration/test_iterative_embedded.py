# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``iterative`` end to end on the embedded stores (no container, no AWS).

A synthetic three-hop chain whose last passage round 0 cannot reach: the
question names the first company, a one-hop graph walk from it reaches the
carrier but not the depot, and the passage that places the depot is about
the depot alone. The fake controller asks for the depot when the context
lacks that passage; round 1 seeds the base search with it and finds the
passage, which the answer's sources then cite. The same chain in English,
Korean, Japanese and Chinese (a corpus indexed as written: the controller is
asked for follow-ups in the corpus language), and once under a document
filter that keeps a decoy passage out of every round.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.ingestion.chunker import BaseChunker
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    PipelineStageStatus,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.evaluation import EvaluationManager

# Every test fails on an AWS client or a Bedrock factory being built.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]


@dataclass(frozen=True)
class _Chain:
    """One language's corpus: the names (first company, carrier, depot,
    harbor), the three passages, the question, the follow-up query and the
    words only the last passage holds."""

    language: LanguageCode
    names: tuple[str, str, str, str]
    passages: dict[str, str]
    question: str
    follow_up: str
    gold: str
    decoy: str


_CHAINS = {
    "en": _Chain(
        LanguageCode.EN,
        ("Vendor", "Carrier", "Depot", "Harbor"),
        {
            "orders.txt": "Vendor hires Carrier for freight.",
            "freight.txt": "Carrier rents Depot for storage.",
            "harbor.txt": "Depot is located in Harbor.",
        },
        "Where is the storage site of the firm Vendor hires?",
        "Where is Depot located?",
        "located in Harbor",
        "Depot is located in Marsh.",
    ),
    "ko": _Chain(
        LanguageCode.KO,
        ("한빛상사", "바다운송", "동부창고", "남항"),
        {
            "orders.txt": "한빛상사는 바다운송에 화물 운송을 맡긴다.",
            "freight.txt": "바다운송은 동부창고를 보관용으로 빌린다.",
            "harbor.txt": "동부창고는 남항에 있다.",
        },
        "한빛상사가 맡긴 운송사의 창고는 어디에 있나?",
        "동부창고는 어디에 있나?",
        "남항에 있다",
        "동부창고는 북항에 있다.",
    ),
    "ja": _Chain(
        LanguageCode.JA,
        ("北星商事", "青海運輸", "東倉庫", "南港"),
        {
            "orders.txt": "北星商事は青海運輸に輸送を任せる。",
            "freight.txt": "青海運輸は東倉庫を保管用に借りる。",
            "harbor.txt": "東倉庫は南港にある。",
        },
        "北星商事が任せた運輸会社の倉庫はどこにある？",
        "東倉庫はどこにある？",
        "南港にある",
        "東倉庫は北港にある。",
    ),
    "zh": _Chain(
        LanguageCode.ZH,
        ("北辰贸易", "海运物流", "东方仓库", "南港"),
        {
            "orders.txt": "北辰贸易委托海运物流运输货物。",
            "freight.txt": "海运物流租用东方仓库存放货物。",
            "harbor.txt": "东方仓库位于南港。",
        },
        "北辰贸易委托的物流公司的仓库在哪里？",
        "东方仓库在哪里？",
        "位于南港",
        "东方仓库位于北港。",
    ),
}


def _responder(chain: _Chain, seen: list[dict[str, Any]]) -> Any:
    first, _, depot, _ = chain.names

    def respond(system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), chain.names)
        if "specialized entity extraction expert" in system:
            return first
        if "expert keyword extractor" in system:
            return json.dumps(
                {"high_level_keywords": [], "low_level_keywords": [first]}
            )
        if "You control the retrieval" in system:
            seen.append({"system": system, "human": human})
            if chain.gold in human:
                return json.dumps(
                    {
                        "sufficient": True,
                        "known_facts": [chain.passages["harbor.txt"]],
                        "bridge_entities": [],
                        "next_queries": [],
                    }
                )
            return json.dumps(
                {
                    "sufficient": False,
                    "known_facts": [chain.passages["freight.txt"]],
                    "bridge_entities": [depot],
                    "next_queries": [chain.follow_up],
                }
            )
        return e2e._respond(system, human)

    return respond


def _build(
    tmp_path: Path,
    chain: _Chain,
    seen: list[dict[str, Any]],
    *,
    categories: dict[str, str] | None = None,
    base: str = "local",
) -> tuple[GraphRAGChain, FakeOllama]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.translation.source_language = chain.language
    config.processing.translation.target_language = chain.language
    # One hop from the first company reaches the carrier, not the depot.
    config.indexing.neptune.max_hops = 1
    # The base round must miss the last hop: the support order keeps to the
    # entities one hop reached (the PageRank's passage search of the
    # question finds the depot's passage on this tiny corpus).
    config.search.local_search.text_unit_ranking = "support"
    config.search.iterative_search.base = base  # type: ignore[assignment]
    server = FakeOllama(_responder(chain, seen), e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in chain.passages.items():
        (source / name).write_text(text, encoding="utf-8")
    if categories is not None:
        (source / "decoy.txt").write_text(chain.decoy, encoding="utf-8")
    create = BaseChunker._create_text_unit

    def with_category(
        self: BaseChunker, chunk_text: str, document: Any, *a: Any
    ) -> Any:
        if categories is not None:
            document.metadata["filters"] = {
                "category": categories[Path(document.file_name).name]
            }
        return create(self, chunk_text, document, *a)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BaseChunker, "_create_text_unit", with_category)
        context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return GraphRAGChain(config, providers=providers), server


def _ask(
    rag: GraphRAGChain,
    chain: _Chain,
    strategy: SearchStrategy,
    filters: dict[str, Any] | None = None,
) -> RAGOutput:
    out = rag.invoke(
        RAGInput(query=chain.question, search_strategy=strategy, filters=filters)
    )
    assert isinstance(out, RAGOutput)
    return out


def _text_sources(out: RAGOutput) -> list[str]:
    return [
        source["content"]
        for source in out.sources
        if source["metadata"]["section_type"] == "text"
    ]


@pytest.mark.parametrize("language", sorted(_CHAINS))
def test_round_one_recovers_the_last_hop_passage(tmp_path: Path, language: str) -> None:
    chain = _CHAINS[language]
    seen: list[dict[str, Any]] = []
    rag, server = _build(tmp_path, chain, seen)
    try:
        # Round 0 alone (the base, local) misses the last hop.
        local = _ask(rag, chain, SearchStrategy.LOCAL)
        assert not any(chain.gold in text for text in _text_sources(local))

        calls_before = len(server.chat_models)
        out = _ask(rag, chain, SearchStrategy.ITERATIVE)
        iterative = out.metadata["iterative"]
        assert (iterative["base"], iterative["mode"]) == ("local", "controller")
        assert (iterative["rounds"], iterative["stop_reason"]) == (1, "sufficient")
        assert iterative["searched"] == [chain.follow_up]
        assert [v["sufficient"] for v in iterative["verdicts"]] == [False, True]
        assert "context_sufficient" not in out.metadata
        assert out.metadata["search_strategy"] == "iterative"
        assert out.answer == e2e.ANSWER

        # The last-hop passage is cited, with its provenance.
        harbor = [
            source
            for source in out.sources
            if source["metadata"]["section_type"] == "text"
            and chain.gold in source["content"]
        ]
        assert len(harbor) == 1
        assert harbor[0]["metadata"]["chunk_id"] == harbor[0]["source"]
        assert harbor[0]["metadata"]["attributes"]["file_name"] == "harbor.txt"
        # A passage both strategies cite carries the same provenance.
        by_id = {s["source"]: s["metadata"] for s in local.sources}
        shared = [s for s in out.sources if s["source"] in by_id]
        assert shared
        for source in shared:
            assert set(by_id[source["source"]]) <= set(source["metadata"])
        assert iterative["added_results"] >= 1

        # The controller was asked in the corpus language, and saw the
        # question and the follow-up it had asked for.
        names = {LanguageCode.EN: "English", LanguageCode.KO: "Korean"}
        names |= {LanguageCode.JA: "Japanese", LanguageCode.ZH: "Chinese"}
        assert len(seen) == 2
        for call in seen:
            assert f"in {names[chain.language]}, the language" in call["system"]
            assert chain.question in call["human"]
        assert chain.follow_up in seen[1]["human"]

        # Usage: the two controller calls on the fast tier, inside the
        # query's own count, which is what the model server served.
        usage = out.metadata["usage"]
        assert usage["llm_calls"] == len(server.chat_models) - calls_before
        assert usage["by_model"]["synthetic-fast"]["llm_calls"] == 2
        assert iterative["usage"]["llm_calls"] == 2
        assert iterative["llm_calls"] == 2
    finally:
        rag.close()


def test_a_sufficient_round_zero_answers_as_mix_does(tmp_path: Path) -> None:
    # On this small corpus mix's round 0 already holds the last hop.
    chain = _CHAINS["en"]
    seen: list[dict[str, Any]] = []
    rag, _ = _build(tmp_path, chain, seen, base="mix")
    try:
        mix = _ask(rag, chain, SearchStrategy.MIX)
        out = _ask(rag, chain, SearchStrategy.ITERATIVE)
    finally:
        rag.close()
    assert out.processed_query.ll_keywords == mix.processed_query.ll_keywords
    assert out.metadata["iterative"]["stop_reason"] == "sufficient"
    assert len(seen) == 1
    assert [(r.source, r.score) for r in out.search_results.results] == [
        (r.source, r.score) for r in mix.search_results.results
    ]
    assert [s["source"] for s in out.sources] == [s["source"] for s in mix.sources]


def test_the_retrieval_evaluator_scores_iterative_sources(tmp_path: Path) -> None:
    chain = _CHAINS["en"]
    rag, _ = _build(tmp_path, chain, [])
    try:
        config = rag.config
        config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        config.evaluation.outputs_directory = str(tmp_path / "eval")
        manager = EvaluationManager(config, rag_chain=rag)
        results, _, summary = asyncio.run(
            manager.evaluate_dataset(
                [
                    EvaluationQuery(
                        query_id="q1",
                        question=chain.question,
                        metadata={"search_strategy": "iterative"},
                    )
                ],
                [
                    EvaluationGroundTruth(
                        query_id="q1",
                        ground_truth=e2e.ANSWER,
                        reference_sources=["harbor.txt"],
                    )
                ],
                show_progress=False,
            )
        )
    finally:
        rag.close()
    assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
    cited = {name for rank in results[0].retrieved_source_ids for name in rank}
    assert "harbor.txt" in cited


def test_every_round_keeps_to_the_callers_filter(tmp_path: Path) -> None:
    chain = _CHAINS["en"]
    categories = dict.fromkeys(chain.passages, "ops") | {"decoy.txt": "archive"}
    seen: list[dict[str, Any]] = []
    rag, _ = _build(tmp_path, chain, seen, categories=categories)
    try:
        unfiltered = _ask(rag, chain, SearchStrategy.ITERATIVE)
        assert any(chain.decoy in text for text in _text_sources(unfiltered))

        out = _ask(rag, chain, SearchStrategy.ITERATIVE, {"attr_category": "ops"})
        assert out.metadata["iterative"]["rounds"] == 1
        texts = _text_sources(out)
        assert any(chain.gold in text for text in texts)
        assert not any(chain.decoy in text for text in texts)
        assert not any(
            chain.decoy in result.content for result in out.search_results.results
        )
        # A filter no document matches leaves nothing in any round.
        empty = _ask(rag, chain, SearchStrategy.ITERATIVE, {"attr_category": "none"})
        assert empty.sources == []
    finally:
        rag.close()


def test_an_insufficient_context_reaches_the_abstention_policy(tmp_path: Path) -> None:
    # The controller never finds the context sufficient and names nothing
    # new, so the strategy flags its context and the chain's
    # `on_insufficient: abstain` declines, keeping the sources it read.
    chain = _CHAINS["en"]
    seen: list[dict[str, Any]] = []
    rag, _ = _build(tmp_path, chain, seen)
    rag.close()
    config = rag.config.model_copy(deep=True)
    config.search.abstention.on_insufficient = "abstain"  # type: ignore[assignment]
    respond = _responder(chain, seen)

    def never_sufficient(system: str, human: str) -> str:
        if "You control the retrieval" in system:
            return json.dumps(
                {
                    "sufficient": False,
                    "known_facts": [],
                    "bridge_entities": [],
                    "next_queries": [],
                }
            )
        return respond(system, human)

    server = FakeOllama(never_sufficient, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    abstaining = GraphRAGChain(config, providers=providers)
    try:
        out = _ask(abstaining, chain, SearchStrategy.ITERATIVE)
    finally:
        abstaining.close()
    assert out.metadata["context_sufficient"] is False
    assert out.metadata["insufficiency_reason"] == "no_follow_up"
    assert out.metadata["abstained"] is True
    assert out.metadata["abstention_reason"] == "insufficient_context"
    assert out.sources
