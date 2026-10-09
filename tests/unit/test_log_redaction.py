# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""User, corpus and model text stays out of INFO+ logs (AWS-free).

Regression: query text, LLM-rewritten queries, corpus entity names and raw
model output were logged at INFO/WARNING/ERROR, and the XML parser put model
output in its exception message. At INFO and above only lengths, counts, ids
and a short hash may appear; the text itself is DEBUG-only.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters.aws import neptune as neptune_module
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.application.retrieval.rag_chain import (
    ChainMode,
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    TextUnit,
)
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.langchain import RobustXMLOutputParser

pytestmark = pytest.mark.unit

_SECRET_QUERY = "Quokkasecret Zebrapassword payroll of Vendor"
_SECRET_FRAGMENTS = ("Quokkasecret", "Zebrapassword", "payroll")


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
    )


def _assert_no_secret(caplog: pytest.LogCaptureFixture) -> None:
    text = _info_text(caplog)
    for fragment in _SECRET_FRAGMENTS:
        assert fragment not in text, text


class _FakeModelFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return RunnableLambda(lambda _prompt_value: "Vendor supplies parts.")

    def get_model_info(self, model_id: Any) -> Any:
        return None


class _FakeRetriever:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        if self.fail:
            raise RuntimeError("backend unavailable")
        return [
            RetrievalResult(
                content="Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source="doc-1",
                retriever_type="document",
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


def _chain(config: Config, *, fail: bool = False) -> GraphRAGChain:
    retriever = _FakeRetriever(fail=fail)
    chain = GraphRAGChain(
        config=config,
        mode=ChainMode.RAG,
        model_factory=_FakeModelFactory(),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain


def test_text_digest_is_length_and_stable_hash() -> None:
    digest = text_digest(_SECRET_QUERY)
    assert digest == text_digest(_SECRET_QUERY)
    assert digest.startswith(f"len={len(_SECRET_QUERY)} sha=")
    assert "Quokka" not in digest
    assert text_digest(None) == "len=0"


@pytest.mark.parametrize(
    "strategy",
    [
        SearchStrategy.SIMPLE,
        SearchStrategy.LOCAL,
        SearchStrategy.GLOBAL,
        SearchStrategy.DRIFT,
        SearchStrategy.NAIVE,
        SearchStrategy.HYBRID,
    ],
)
async def test_query_text_absent_from_info_logs(
    config: Config, caplog: pytest.LogCaptureFixture, strategy: SearchStrategy
) -> None:
    caplog.set_level(logging.INFO)
    await _chain(config).ainvoke(
        RAGInput(
            query=_SECRET_QUERY,
            search_strategy=strategy,
            enable_query_processing=False,
        )
    )
    _assert_no_secret(caplog)


async def test_query_text_absent_from_chain_error_logs(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    config.processing.ignore_errors = True
    chain = _chain(config, fail=True)
    caplog.set_level(logging.INFO)
    await chain.ainvoke(
        RAGInput(
            query=_SECRET_QUERY,
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
        )
    )
    _assert_no_secret(caplog)


async def test_neptune_retriever_logs_no_query_text(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch.object(neptune_module, "DriverRemoteConnection", MagicMock())
    mocker.patch.object(neptune_module, "traversal", return_value=MagicMock())
    config = Config()
    config.aws.neptune.endpoint = "neptune.example.invalid"
    config.aws.neptune.use_iam = False
    client = NeptuneClient(config=config, boto_session=MagicMock())
    retriever = NeptuneRetriever(
        config=config, neptune_client=client, boto_session=MagicMock()
    )

    async def no_rows(_traversal: Any) -> list:
        return []

    mocker.patch.object(retriever, "_execute_traversal", side_effect=no_rows)
    caplog.set_level(logging.INFO)
    # No entity focus: the name-term fallback path.
    assert await retriever.aretrieve(SearchQuery(query=_SECRET_QUERY)) == []
    _assert_no_secret(caplog)


async def test_opensearch_retriever_logs_no_query_text(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(
        inst, "_terms_batch_size", config.indexing.opensearch.terms_batch_size
    )
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(inst, "_record_timing", lambda *a, **k: None)
    object.__setattr__(inst, "_record_metric", lambda *a, **k: None)

    async def no_vector(*_args: Any, **_kwargs: Any) -> None:
        return None

    object.__setattr__(inst, "_get_query_vector", no_vector)
    object.__setattr__(inst, "_create_search_tasks", lambda *a, **k: [])
    caplog.set_level(logging.INFO)
    assert await inst.aretrieve(SearchQuery(query=_SECRET_QUERY)) == []
    _assert_no_secret(caplog)


def test_xml_parser_keeps_model_output_out_of_error_and_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    with pytest.raises(OutputParserException) as excinfo:
        RobustXMLOutputParser().parse(_SECRET_QUERY)
    for fragment in _SECRET_FRAGMENTS:
        assert fragment not in str(excinfo.value)
    _assert_no_secret(caplog)
    # The raw output stays available to OutputFixingParser.
    assert excinfo.value.llm_output == _SECRET_QUERY


def test_entity_parse_warnings_omit_corpus_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    proc = BaseProcessor(config=Config())
    unit = TextUnit(id="tu1", short_id="tu1", text="irrelevant")
    caplog.set_level(logging.INFO)

    # Missing type: the "missing data" warning.
    assert (
        proc.parse_relationship_data(
            {"source": "Quokkasecret", "target": "Zebrapassword", "type": ""},
            unit,
            entity_name_to_id={},
        )
        is None
    )

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("bad attributes")

    proc._parse_attributes = boom  # type: ignore[method-assign]
    assert proc.parse_entity_data({"name": "Quokkasecret"}, unit) is None
    assert (
        proc.parse_relationship_data(
            {"source": "Quokkasecret", "target": "Zebrapassword", "type": "PAYS"},
            unit,
            entity_name_to_id={},
        )
        is None
    )
    _assert_no_secret(caplog)
    assert "tu1" in _info_text(caplog)
