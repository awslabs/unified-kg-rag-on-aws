# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the cycle-34 dead-code and duplication cleanup (AWS-free)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphIndexer
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.ports import GraphIndexer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cls", [SQLiteGraphIndexer, NeptuneIndexer])
def test_graph_indexers_share_the_port_index_relationships(cls: type) -> None:
    """A full run writes edges through the same path as a delta."""
    assert cls.index_relationships is GraphIndexer.index_relationships


def test_graph_index_relationships_delegates_to_upsert(mocker) -> None:
    upsert = mocker.Mock(return_value="stats")
    store = mocker.Mock(upsert_relationships=upsert)
    assert GraphIndexer.index_relationships(store, ["edge"]) == "stats"
    upsert.assert_called_once_with(["edge"])


class _SyncOnlyEvaluator(BaseGraphRAGEvaluator):
    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=[],
            evaluation_time=datetime.now(),
        )


async def test_evaluator_default_async_path_runs_the_sync_one(config: Config) -> None:
    evaluator = _SyncOnlyEvaluator(config, EvaluatorType.RETRIEVAL)
    query = EvaluationQuery(query_id="q", question="?")
    result = EvaluationResult(
        query_id="q", question="?", generated_answer="a", ground_truth=""
    )
    report = await evaluator.aevaluate_single(query, result, "")
    assert report.query_id == "q"


async def test_evaluator_without_either_scoring_method_fails_loudly(
    config: Config,
) -> None:
    evaluator = BaseGraphRAGEvaluator(config, EvaluatorType.RETRIEVAL)
    query = EvaluationQuery(query_id="q", question="?")
    result = EvaluationResult(
        query_id="q", question="?", generated_answer="a", ground_truth=""
    )
    with pytest.raises(NotImplementedError):
        await evaluator.aevaluate_single(query, result, "")


def test_community_detector_is_not_a_base_processor() -> None:
    """It used none of the extraction helpers it inherited."""
    from unified_kg_rag.adapters.ingestion.community_detector import (
        CommunityDetector,
    )
    from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor

    assert not issubclass(CommunityDetector, BaseProcessor)


@pytest.mark.parametrize(
    ("target", "removed"),
    [
        (
            "unified_kg_rag.adapters.search_strategies.drift_search:"
            "DriftSearchStrategy._evolve_query",
            "max_keywords",
        ),
        (
            "unified_kg_rag.adapters.ingestion.community_detector:"
            "CommunityDetector._create_cluster_graph",
            "source_graph",
        ),
    ],
)
def test_dead_parameters_are_gone(target: str, removed: str) -> None:
    import importlib
    import inspect

    module, _, attr = target.partition(":")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    assert removed not in inspect.signature(obj).parameters


def test_rag_chain_has_no_build_retriever_pass_through() -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    assert not hasattr(GraphRAGChain, "_build_retriever")
