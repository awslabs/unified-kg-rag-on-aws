# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""User, corpus and model text stays out of INFO+ logs (AWS-free).

Regression: query text, LLM-rewritten queries, corpus entity names and raw
model output were logged at INFO/WARNING/ERROR, and the XML parser put model
output in its exception message. At INFO and above only lengths, counts, ids
and a short hash may appear; the text itself is DEBUG-only.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
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
        model_factory=ScriptedLLMFactory(
            lambda _system, _human: "Vendor supplies parts."
        ),
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
    retriever = NeptuneRetriever(config=config, neptune_client=client)

    async def no_rows(_traversal: Any) -> list:
        return []

    mocker.patch.object(retriever, "_execute_traversal", side_effect=no_rows)
    caplog.set_level(logging.INFO)
    seeded = SearchQuery(query=_SECRET_QUERY, filters={"id": ["e1"]})
    assert await retriever.aretrieve(seeded) == []
    _assert_no_secret(caplog)


async def test_opensearch_retriever_logs_no_query_text(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())

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


_PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"
_INFO_PLUS = {"info", "warning", "warn", "error", "exception", "critical"}
# Attributes that hold user, corpus or model text on the package's models.
_TEXT_ATTRS = {
    "answer",
    "content",
    "description",
    "full_content",
    "llm_output",
    "page_content",
    "query",
    "source_name",
    "summary",
    "target_name",
    "text",
    "title",
}
_GRAPH_VARS = {"claim", "community", "ent", "entity", "node", "rel", "relationship"}


def _leaky_args(call: ast.Call) -> list[str]:
    leaks: list[str] = []
    # The message itself when it is an f-string, the %-arguments and the
    # keyword arguments (``extra=``) can all carry text.
    candidates = [*call.args[1:], *(kw.value for kw in call.keywords)]
    if call.args and isinstance(call.args[0], ast.JoinedStr):
        candidates.insert(0, call.args[0])
    for arg in candidates:
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id in {"len", "text_digest"}
        ):
            continue
        for sub in ast.walk(arg):
            names_text = isinstance(sub, ast.Attribute) and (
                sub.attr in _TEXT_ATTRS
                or (
                    sub.attr == "name"
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in _GRAPH_VARS
                )
            )
            # ``str(model_output)[:250]`` previews.
            previews_text = (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.slice, ast.Slice)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Name)
                and sub.value.func.id == "str"
            )
            if names_text or previews_text:
                leaks.append(ast.unparse(arg))
                break
    return leaks


def test_no_info_log_call_passes_corpus_or_model_text() -> None:
    """Static guard: INFO+ ``logger`` calls never pass entity/relationship
    names, descriptions, document text, queries or model-output previews.

    Pass ``len(...)``, ids/short ids or ``text_digest(...)`` instead, and log
    the text itself at DEBUG.
    """
    offenders: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _INFO_PLUS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                offenders += [
                    f"{path.relative_to(_PACKAGE.parent)}:{node.lineno}: {leak}"
                    for leak in _leaky_args(node)
                ]
    assert not offenders, "\n".join(offenders)


def test_static_guard_flags_a_name_in_an_info_log() -> None:
    call = ast.parse(
        "logger.info('Dropping %s -> %s', rel.source_name, rel.target_name)"
    ).body[0]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert _leaky_args(call.value) == ["rel.source_name", "rel.target_name"]
    ok = ast.parse("logger.info('Dropping %s', rel.short_id)").body[0]
    assert isinstance(ok, ast.Expr) and isinstance(ok.value, ast.Call)
    assert _leaky_args(ok.value) == []


@pytest.mark.parametrize(
    "source",
    [
        "logger.info(f'Dropping {rel.source_name}')",
        "logger.warning('Dropped', extra={'entity': entity.name})",
        "logger.error('Failed', extra={'q': search_query.query})",
    ],
)
def test_static_guard_flags_text_in_an_f_string_or_keyword(source: str) -> None:
    call = ast.parse(source).body[0]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert _leaky_args(call.value)
