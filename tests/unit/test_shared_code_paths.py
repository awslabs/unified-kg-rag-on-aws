# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code paths two callers share, so they cannot drift apart (AWS-free)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
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


def test_evaluator_subclass_without_a_scoring_method_fails_at_definition() -> None:
    with pytest.raises(TypeError, match="must implement evaluate_single"):

        class _Unscored(BaseGraphRAGEvaluator):
            pass


def test_evaluator_subclass_with_only_the_async_method_is_accepted() -> None:
    class _AsyncOnly(BaseGraphRAGEvaluator):
        async def aevaluate_single(self, query, result, ground_truth, **kwargs):
            raise AssertionError("not called")

    assert issubclass(_AsyncOnly, BaseGraphRAGEvaluator)


def test_dynamodb_batch_backoff_waits_only_between_requests(mocker) -> None:
    from unified_kg_rag.adapters.doc_status import dynamodb

    sleep = mocker.patch.object(dynamodb.time, "sleep")
    dynamodb._backoff_before(0)
    sleep.assert_not_called()
    dynamodb._backoff_before(3)
    dynamodb._backoff_before(10)
    assert [c.args[0] for c in sleep.call_args_list] == [pytest.approx(0.4), 2.0]


def test_full_and_delta_indexing_write_the_same_artifact_types(mocker) -> None:
    """Both paths come from one (full writer, delta writer) table, so a type
    cannot be indexed by one and forgotten by the other."""
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    vector, graph = mocker.MagicMock(), mocker.MagicMock()
    manager = IndexingManager(
        config=Config(), vector_indexer=vector, graph_indexer=graph
    )
    mocker.patch.object(manager, "_log_completion_summary")
    artifacts: dict[str, Any] = {
        "text_units": [SimpleNamespace(id="t", community_ids=None)],
        "entities": ["e"],
        "relationships": ["r"],
        "communities": [
            SimpleNamespace(id="c", text_unit_ids=[], level="0", parent="", children=[])
        ],
        "community_reports": [SimpleNamespace(community_id="other")],
        "claims": ["cl"],
    }
    full = manager.index_all_data(**artifacts)
    delta = manager.index_delta(**artifacts)
    # (The mocks' stats report failures, so the rebuild also reports its
    # skipped sweep: not an artifact type.)
    assert set(full) - {"neptune_sweep"} == set(delta)
    graph.index_relationships.assert_called_once_with(["r"])  # the full build
    graph.upsert_relationships.assert_called_once_with(["r"])  # the delta


@pytest.mark.parametrize(
    "module", ["run_rag_chain", "run_evaluation", "run_ingestion_pipeline"]
)
def test_clis_show_the_package_version(module: str) -> None:
    import importlib

    import unified_kg_rag

    cli = importlib.import_module(f"unified_kg_rag.application.cli.{module}")
    assert cli.__version__ is unified_kg_rag.__version__


def test_registry_default_namespace_is_the_default_index_suffix() -> None:
    import inspect

    from unified_kg_rag.application.ingestion import incremental
    from unified_kg_rag.domain.models import Constants
    from unified_kg_rag.shared.utils.document_identity import DEFAULT_NAMESPACE

    assert DEFAULT_NAMESPACE is Constants.DEFAULT_SUFFIX.value
    for fn in (incremental.build_document_lineage, incremental.IncrementalIndexer):
        default = inspect.signature(fn).parameters["suffix"].default
        assert default is DEFAULT_NAMESPACE


def _loop_recording_retriever(config: Config) -> Any:
    import asyncio

    from unified_kg_rag.adapters.retrieval.base import BaseGraphRAGRetriever
    from unified_kg_rag.domain.models import RetrievalResult

    class _Retriever(BaseGraphRAGRetriever):
        async def aretrieve(self, query: Any) -> list[RetrievalResult]:
            loops.append(asyncio.get_running_loop())
            return [
                RetrievalResult(
                    content=f"about {query.query}",
                    score=0.5,
                    source="doc-1",
                    retriever_type="text_unit",
                )
            ]

    loops: list[Any] = []
    retriever = _Retriever(config)
    return retriever, loops


def test_retriever_invoke_outside_a_loop(config: Config) -> None:
    retriever, _ = _loop_recording_retriever(config)
    [document] = retriever.invoke("vendors")
    assert document.page_content == "about vendors"
    assert document.metadata["source"] == "doc-1"


async def test_retriever_ainvoke_runs_on_the_callers_loop(config: Config) -> None:
    """The stores' async clients are bound to the loop they were made on."""
    import asyncio

    retriever, loops = _loop_recording_retriever(config)
    [document] = await retriever.ainvoke("vendors")
    assert document.page_content == "about vendors"
    assert loops == [asyncio.get_running_loop()]
