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
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.aws import neptune as neptune_module
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator
from unified_kg_rag.adapters.llm.langchain import RobustXMLOutputParser
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    TextUnit,
)
from unified_kg_rag.evaluation.answer_match_evaluator import AnswerMatchEvaluator
from unified_kg_rag.evaluation.assertion_evaluator import AssertionEvaluator
from unified_kg_rag.evaluation.pairwise import PairwiseJudge
from unified_kg_rag.shared.utils import text_digest

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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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


def _judge_reply(system: str, _human: str) -> str:
    """Malformed judge replies that quote the answer's secret text."""
    if "numbered assertions" in system:
        return json.dumps(
            {"verdict": [{"id": 1, "verdict": "supported", "reason": _SECRET_QUERY}]}
        )
    if "statement by statement" in system:
        return json.dumps({"shared": _SECRET_QUERY, "unique_1": [], "unique_2": []})
    if "span" in system.lower():
        return _SECRET_QUERY
    return json.dumps({"comprehensiveness": {"winner": _SECRET_QUERY}})


def _providers(config: Config) -> Providers:
    return Providers(
        config,
        llm_factory=ScriptedLLMFactory(_judge_reply),
        embedding_factory=HashingEmbeddingFactory(),
    )


async def test_judge_failures_log_no_judge_or_answer_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: a malformed assertion, span or pairwise reply was quoted
    in the WARNING (the parsers put the reply in their error message)."""
    caplog.set_level(logging.DEBUG)
    result = EvaluationResult(
        query_id="q1",
        question="Who pays?",
        generated_answer="Vendor pays Buyer.",
        ground_truth="Vendor",
        metadata={"assertions": ["Vendor pays Buyer"]},
    )
    query = EvaluationQuery(query_id="q1", question="Who pays?")
    config = Config()
    report = await AssertionEvaluator(
        config, providers=_providers(config)
    ).aevaluate_single(query, result, "")
    assert report.metadata["failed_metrics"]
    config = Config()
    config.evaluation.answer_span = "extracted"
    report = await AnswerMatchEvaluator(
        config, providers=_providers(config)
    ).aevaluate_single(query, result, "Vendor")
    assert "answer_span_fallback" in report.metadata
    other = result.model_copy(update={"generated_answer": "Buyer pays."})
    for mode in ("standard", "differential"):
        config = Config()
        config.evaluation.pairwise_mode = mode  # type: ignore[assignment]
        verdicts = await PairwiseJudge(config, _providers(config)).acompare(
            [result], [other]
        )
        assert verdicts["failed"] == 1
    _assert_no_secret(caplog)
    # The text stays available at DEBUG, and the WARNINGs name the queries.
    debug = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    assert "Quokkasecret" in debug
    assert "'q1'" in _info_text(caplog)


class _CrashingEvaluator(BaseGraphRAGEvaluator):
    def evaluate_single(self, *args: Any, **kwargs: Any) -> EvaluationReport:
        raise ValueError(f"judge said {_SECRET_QUERY}")


async def test_batch_evaluation_errors_log_no_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    evaluator = _CrashingEvaluator(
        Config(), EvaluatorType.ANSWER_MATCH, show_progress=False
    )
    query = EvaluationQuery(query_id="q1", question="q")
    result = EvaluationResult(
        query_id="q1", question="q", generated_answer="a", ground_truth=""
    )
    reports = await evaluator.aevaluate_batch([query], [result], [""])
    assert reports[0].metadata["evaluation_failed"]
    _assert_no_secret(caplog)


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


# Modules whose errors come from model replies or answers: the parsers quote
# the reply in their error message, so an INFO+ log passes
# ``error_digest(e)``, never the exception itself.
_JUDGE_MODULES = (
    "adapters/evaluators",
    "evaluation/answer_match_evaluator.py",
    "evaluation/assertion_evaluator.py",
    "evaluation/benchmarks/autoq.py",
    "evaluation/pairwise.py",
    # A failed query or claim extraction step's error can quote the query,
    # the model's reply or the text unit it was given.
    "application/retrieval/rag_chain.py",
    "adapters/ingestion/claim_extractor.py",
)
_ERROR_NAMES = {"e", "err", "error", "errors", "exc", "outcome"}


def _passes_raw_error(node: ast.AST) -> bool:
    """Whether ``node`` hands an exception to the log record: names one
    outside ``error_digest(...)``/``len(...)``/``type(...)`` (the condition of
    ``x if errors else y`` is not logged)."""
    if isinstance(node, ast.Name):
        return node.id in _ERROR_NAMES
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"error_digest", "len", "type"}
    ):
        return False
    if isinstance(node, ast.IfExp):
        return _passes_raw_error(node.body) or _passes_raw_error(node.orelse)
    return any(_passes_raw_error(child) for child in ast.iter_child_nodes(node))


def _logs_raw_error(call: ast.Call) -> list[str]:
    if any(kw.arg == "exc_info" for kw in call.keywords):
        return ["exc_info"]
    return [
        ast.unparse(arg)
        for arg in [*call.args[1:], *(kw.value for kw in call.keywords)]
        if _passes_raw_error(arg)
    ]


def test_judge_modules_log_error_digests_not_errors() -> None:
    offenders: list[str] = []
    paths = sorted(
        path
        for target in _JUDGE_MODULES
        for path in (
            [_PACKAGE / target]
            if target.endswith(".py")
            else (_PACKAGE / target).rglob("*.py")
        )
    )
    for path in paths:
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
                    for leak in _logs_raw_error(node)
                ]
    assert not offenders, "\n".join(offenders)


def test_static_error_guard_flags_a_raw_exception() -> None:
    for source, leaky in [
        ("logger.warning('failed for %s: %s', qid, e)", True),
        ("logger.error('failed: %s', e, exc_info=True)", True),
        ("logger.warning('failed: %s', errors[-1])", True),
        ("logger.warning('failed: %s', error_digest(e))", False),
        ("logger.warning('failed %s', len(errors))", False),
        ("logger.warning('%s', error_digest(errors[-1]) if errors else 'x')", False),
    ]:
        call = ast.parse(source).body[0]
        assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
        assert bool(_logs_raw_error(call.value)) is leaky, source


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
