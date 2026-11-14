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
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.retrievers import StaticRetriever, hit
from unified_kg_rag.adapters.aws import neptune as neptune_module
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator
from unified_kg_rag.adapters.llm.langchain import RobustXMLOutputParser
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
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


def _chain(config: Config, *, fail: bool = False) -> GraphRAGChain:
    retriever = StaticRetriever(
        [hit("Vendor and Buyer signed the supply agreement.")], fail=fail
    )
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


# Names that hold errors outside an ``except`` clause (a list of a batch's
# errors, an outcome that may be one); inside one, the name it binds too.
_ERROR_NAMES = frozenset({"errors", "outcome"})
# Calls and attributes that log what an error is, not what it says
# (``format_validation_errors``: the invalid keys and why, not the values).
_SAFE_ERROR_CALLS = frozenset(
    {
        "error_digest",
        "error_locator",
        "error_name",
        "text_digest",
        "len",
        "type",
        "classify",
        "format_validation_errors",
        # OpenSearch's own reason for a request it refused (a bulk request's
        # 429): what the store says, logged by design (adapters.aws.opensearch).
        "error_reason",
    }
)
_SAFE_ERROR_ATTRS = frozenset({"__name__", "__class__", "status_code", "errno"})
# Calls that return the current exception or its text (``logger.error("%s",
# traceback.format_exc())``) wherever they are logged.
_CURRENT_ERROR_CALLS = frozenset(
    {
        "traceback.format_exc",
        "traceback.format_exception",
        "traceback.format_exception_only",
        "traceback.format_tb",
        "sys.exc_info",
        "sys.exception",
    }
)
# The names a logger goes by (``logger``, ``_logger``, ``self.logger``,
# ``logging`` itself), and a ``logger.bind(...)``'s.
_LOGGER_NAMES = frozenset({"logger", "_logger", "log", "logging", "LOGGER"})
# The level of a ``logger.log(level, ...)`` call that is DEBUG; any other
# (a variable, a level that may be raised) counts as INFO+.
_DEBUG_LEVELS = frozenset({"logging.DEBUG", "DEBUG", "10"})


def _names_error(node: ast.AST, names: frozenset[str]) -> bool:
    """Whether ``node`` is one of the errors ``names`` (a name, or the
    ``outcomes[0]`` an ``isinstance`` check named)."""
    if isinstance(node, ast.Name):
        return node.id in names
    return (
        isinstance(node, ast.Subscript | ast.Attribute) and ast.unparse(node) in names
    )


def _passes_raw_error(node: ast.AST, names: frozenset[str]) -> bool:
    """Whether ``node`` hands one of the errors ``names`` to the log record
    (an argument, an f-string's ``{e}``): names one outside
    ``error_digest(...)``/``len(...)``/``type(...)`` and the attributes that
    name its type (the condition of ``x if errors else y`` is not logged),
    or the current exception (``traceback.format_exc()``)."""
    if _names_error(node, names):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_ERROR_CALLS:
            return False
        if ast.unparse(node.func) in _CURRENT_ERROR_CALLS:
            return True
        # ``future.exception()``: the error a future or task holds.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "exception"
            and not node.args
        ):
            return True
    if isinstance(node, ast.Attribute) and node.attr in _SAFE_ERROR_ATTRS:
        return False
    if isinstance(node, ast.IfExp):
        return _passes_raw_error(node.body, names) or _passes_raw_error(
            node.orelse, names
        )
    return any(_passes_raw_error(child, names) for child in ast.iter_child_nodes(node))


def _is_logger(node: ast.AST) -> bool:
    """A logger: one of ``_LOGGER_NAMES``, as a name or an attribute
    (``self.logger``), or what its ``bind(...)`` returns."""
    if isinstance(node, ast.Name):
        return node.id in _LOGGER_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _LOGGER_NAMES
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bind"
        and _is_logger(node.func.value)
    )


def _is_info_plus_log(call: ast.Call) -> bool:
    """A ``logger.<info..critical>(...)`` call, or ``logger.log(level, ...)``
    at a level other than ``DEBUG``."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and _is_logger(func.value)):
        return False
    if func.attr == "log":
        return bool(call.args) and ast.unparse(call.args[0]) not in _DEBUG_LEVELS
    return func.attr in _INFO_PLUS


def _logs_raw_error(call: ast.Call, names: frozenset[str] = _ERROR_NAMES) -> list[str]:
    # ``logger.exception`` logs the traceback, and so the message, as
    # ``exc_info`` does.
    if isinstance(call.func, ast.Attribute) and call.func.attr == "exception":
        return ["exception"]
    if any(
        kw.arg == "exc_info"
        and not (isinstance(kw.value, ast.Constant) and not kw.value.value)
        for kw in call.keywords
    ):
        return ["exc_info"]
    first = 2 if isinstance(call.func, ast.Attribute) and call.func.attr == "log" else 1
    candidates = [*call.args[first - 1 : first], *call.args[first:]]
    candidates += [kw.value for kw in call.keywords]
    # What ``logger.bind(err=...)`` binds is logged with the record.
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    while isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
        candidates += [*receiver.args, *(kw.value for kw in receiver.keywords)]
        receiver = receiver.func.value
    return [ast.unparse(arg) for arg in candidates if _passes_raw_error(arg, names)]


# The ``except`` clauses whose error may be a store's, a provider's or a
# library's (its text may quote a document, a prompt or an ARN): a broad one
# and the store and model errors. A narrow one (``ValueError``,
# ``ConfigurationError``) catches a refusal whose message is for its reader.
_BROAD_CATCHES = frozenset(
    {
        "Exception",
        "BaseException",
        "AWSServiceError",
        "ClientError",
        "BotoCoreError",
        "TransportError",
        "OpenSearchException",
        "GremlinServerError",
        "ModelError",
        "LanguageModelError",
    }
)


def _catches_broadly(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        (t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", ""))
        in _BROAD_CATCHES
        for t in types
    )


def _error_text_of(node: ast.AST, names: frozenset[str]) -> bool:
    """Whether ``node`` is the text of one of the errors ``names``:
    ``str(e)``, ``repr(e)``, ``format(e)``, ``e.args``, an f-string's
    ``{e}`` or ``"%s" % e``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"str", "repr", "format"}
        and node.args
    ):
        return _names_error(node.args[0], names)
    if isinstance(node, ast.Attribute) and node.attr == "args":
        return _names_error(node.value, names)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(part, ast.FormattedValue) and _names_error(part.value, names)
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        operands = (
            node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        )
        return any(_names_error(operand, names) for operand in operands)
    return False


def _kept_error_texts(node: ast.AST, names: frozenset[str]) -> list[ast.AST]:
    """The texts of the errors ``names`` in ``node`` that it keeps (assigns,
    returns, appends, records with ``add_error``) rather than raises in an
    exception of its own, logs at DEBUG or names log-safely
    (``error_digest(e)``): a kept text is logged or shown later, where no
    guard reads it."""
    if isinstance(node, ast.Raise):
        return []
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func.split(".")[-1] in _SAFE_ERROR_CALLS | {"debug", "isinstance"}:
            return []
    if _error_text_of(node, names):
        return [node]
    return [
        found
        for child in ast.iter_child_nodes(node)
        for found in _kept_error_texts(child, names)
    ]


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of ``scope`` (a module or a function), not of the functions
    it defines."""
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from _own_nodes(child)


def _isinstance_errors(scope: ast.AST) -> set[str]:
    """What ``scope`` (a module or a function) checks with ``isinstance(x,
    ...Exception/Error)``: an error outside an ``except`` clause (an
    ``asyncio.gather(..., return_exceptions=True)`` outcome)."""
    checked: set[str] = set()
    for node in _own_nodes(scope):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and re.search(r"Exception|Error", ast.unparse(node.args[1]))
            and isinstance(node.args[0], ast.Name | ast.Subscript | ast.Attribute)
        ):
            checked.add(ast.unparse(node.args[0]))
    return checked


class _RawErrorLogs(ast.NodeVisitor):
    """What a module does with an error's text that the logging policy
    forbids: the INFO+ log calls that hand an error to the record, and (in
    a broad ``except`` clause) the error's text kept to be logged or shown
    later (``found``). An error is a name an enclosing ``except ... as
    name`` binds, a name assigned one there (``err = e``), a value its
    function checks with ``isinstance(x, Exception)`` and ``_ERROR_NAMES``."""

    def __init__(self) -> None:
        self.bound: list[str] = []
        self.broad: list[str] = []
        self.scopes: list[set[str]] = [set()]
        self.found: list[tuple[int, str]] = []

    def _names(self) -> frozenset[str]:
        bound = {name for name in self.bound if name}
        return _ERROR_NAMES | bound | self.scopes[-1]

    def visit_Module(self, node: ast.Module) -> None:
        self.scopes = [_isinstance_errors(node)]
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(_isinstance_errors(node))
        self.generic_visit(node)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.bound.append(node.name or "")
        broad = bool(node.name) and _catches_broadly(node)
        if broad:
            self.broad.append(node.name or "")
            names = frozenset(self.broad) | self.scopes[-1]
            for statement in node.body:
                self.found += [
                    (found.lineno, f"kept {ast.unparse(found)}")
                    for found in _kept_error_texts(statement, names)
                ]
        self.generic_visit(node)
        if broad:
            self.broad.pop()
        self.bound.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # ``err = e``: the alias is the error for the rest of its function.
        if _names_error(node.value, self._names()):
            self.scopes[-1].update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_info_plus_log(node):
            self.found += [
                (node.lineno, leak) for leak in _logs_raw_error(node, self._names())
            ]
        self.generic_visit(node)


def _raw_error_logs(source: str) -> list[tuple[int, str]]:
    visitor = _RawErrorLogs()
    visitor.visit(ast.parse(source))
    return visitor.found


def test_no_info_log_call_passes_a_raw_error() -> None:
    """Static guard over the whole package: an INFO+ log call never hands
    an error to the record (an argument, ``{e}`` in an f-string,
    ``exc_info``, ``logger.exception``): its text may quote a model's
    reply, a query, a document or an ARN. ``log_error``/``log_failure``
    log it by its class (type, digest and where it happened; text at
    DEBUG)."""
    offenders: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        offenders += [
            f"{path.relative_to(_PACKAGE.parent)}:{line}: {leak}"
            for line, leak in _raw_error_logs(path.read_text(encoding="utf-8"))
        ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize(
    ("source", "leaky"),
    [
        ("logger.warning('failed for %s: %s', qid, e)", True),
        ("logger.error('failed: %s', e, exc_info=True)", True),
        ("logger.exception('Question %s failed', qid)", True),
        ("logger.info('refused (%s)', text_digest(error))", False),
        ("logger.warning('failed: %s', errors[-1])", True),
        ("logger.warning('failed: %s', error_digest(e))", False),
        ("logger.warning('failed %s', len(errors))", False),
        ("logger.warning('%s', error_digest(errors[-1]) if errors else 'x')", False),
        ("logger.error('failed: %s', e, exc_info=False)", True),
        ("logger.error('failed (%s)', type(e).__name__, exc_info=None)", False),
        # Any name the clause binds, an f-string and ``logger.log``.
        ("logger.warning('close failed: %s', close_error)", True),
        ("logger.warning(f'failed: {raised}')", True),
        ("logger.log(logging.ERROR, 'failed: %s', item_error)", True),
        ("logger.log(logging.DEBUG, 'failed: %s', item_error)", False),
        ("logger.log(level, f'failed {raised!r}')", True),
        ("logger.warning('failed (%s)', raised.__class__.__name__)", False),
    ],
)
def test_static_error_guard_flags_a_raw_exception(source: str, leaky: bool) -> None:
    wrapped = (
        "try:\n    pass\n"
        "except (OSError, ValueError) as raised:\n"
        "    try:\n        pass\n"
        "    except Exception as item_error:\n"
        "        try:\n            pass\n"
        "        except Exception as close_error:\n"
        "            try:\n                pass\n"
        "            except Exception as e:\n"
        f"                {source}\n"
    )
    assert bool(_raw_error_logs(wrapped)) is leaky, source


_CLAUSE = "try:\n    pass\nexcept Exception as e:\n"


@pytest.mark.parametrize(
    ("source", "leaky"),
    [
        # An alias of the error, in the clause and after it.
        (_CLAUSE + "    err = e\n    logger.warning('x %s', err)\n", True),
        (
            "def f():\n"
            + "".join(
                f"    {line}\n"
                for line in [
                    "try:",
                    "    pass",
                    "except Exception as e:",
                    "    err = e",
                    "logger.warning('x %s', err)",
                ]
            ),
            True,
        ),
        # The current exception, by the traceback and sys modules.
        (_CLAUSE + "    logger.error('x %s', traceback.format_exc())\n", True),
        (_CLAUSE + "    logger.error('x %s', sys.exc_info()[1])\n", True),
        # Any logger's name, and what ``bind`` binds.
        (_CLAUSE + "    _logger.warning('x %s', e)\n", True),
        (_CLAUSE + "    self.logger.warning('x %s', e)\n", True),
        (_CLAUSE + "    logger.bind(err=str(e)).warning('x')\n", True),
        (_CLAUSE + "    logger.bind(err=error_digest(e)).warning('x')\n", False),
        # A gathered outcome its function checks is an exception.
        (
            "async def f(t):\n"
            "    r = await asyncio.gather(t, return_exceptions=True)\n"
            "    if isinstance(r[0], Exception):\n"
            "        logger.warning('x %s', r[0])\n",
            True,
        ),
        (
            "async def f(t):\n"
            "    r = await asyncio.gather(t, return_exceptions=True)\n"
            "    if isinstance(r[0], Exception):\n"
            "        log_failure(logger, 30, 'x %s', error=r[0])\n",
            False,
        ),
        # The same at module level, the error a future holds, and a level
        # that may be raised.
        (
            "r = asyncio.run(f())\n"
            "if isinstance(r[0], Exception):\n"
            "    logger.warning('x %s', r[0])\n",
            True,
        ),
        ("logger.warning('x %s', fut.exception())\n", True),
        (_CLAUSE + "    logger.log(DEBUG_OR_ERROR, 'x %s', e)\n", True),
        (_CLAUSE + "    logger.log(logging.DEBUG, 'x %s', e)\n", False),
        # The error's text kept (returned, assigned, recorded) to be logged
        # or shown later, in a broad clause.
        (
            "def f():\n    try:\n        pass\n"
            "    except Exception as e:\n        return str(e)\n",
            True,
        ),
        (_CLAUSE + "    stats.add_error(str(e), 1)\n", True),
        (_CLAUSE + "    stats.add_error(f'Bulk write failed: {e}', 1)\n", True),
        (_CLAUSE + "    errors.append(repr(e))\n", True),
        (_CLAUSE + "    result.error = '%s' % e\n", True),
        (_CLAUSE + "    failed[key] = e.args[0]\n", True),
        # Not kept: the exception itself, a log-safe name, its own raise,
        # a DEBUG line, and a narrow clause's refusal.
        (_CLAUSE + "    stats.add_error(e, 1, context='Bulk write failed')\n", False),
        (_CLAUSE + "    reason = error_name(e)\n", False),
        (_CLAUSE + "    raise AWSServiceError(f'read failed: {e}') from e\n", False),
        (_CLAUSE + "    logger.debug('x %s', str(e))\n", False),
        ("try:\n    pass\nexcept ValueError as e:\n    usage_error(str(e))\n", False),
    ],
)
def test_static_error_guard_follows_an_error_wherever_it_goes(
    source: str, leaky: bool
) -> None:
    """Every rule the guard adds catches its planted example (the patterns
    an adversarial review found passing it), and the safe forms pass."""
    assert bool(_raw_error_logs(source)) is leaky, source


def test_static_error_guard_reads_the_names_a_clause_binds_in_its_body() -> None:
    # Outside the clause the bound name is an ordinary variable again.
    source = (
        "try:\n    pass\n"
        "except Exception as raised:\n"
        "    logger.warning('failed: %s', raised)\n"
        "logger.warning('value: %s', raised)\n"
    )
    assert [line for line, _ in _raw_error_logs(source)] == [4]


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


def test_log_failure_logs_a_digest_at_its_level_and_the_text_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unified_kg_rag.shared import get_logger
    from unified_kg_rag.shared.logging import log_failure
    from unified_kg_rag.shared.utils import error_digest

    error = RuntimeError("the model said: Quokkasecret")
    with caplog.at_level("DEBUG"):
        log_failure(
            get_logger("probe"), logging.WARNING, "Step %s failed: %s", 3, error=error
        )
    (warned,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    (debug,) = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert warned.getMessage() == f"Step 3 failed: {error_digest(error)}"
    assert "Quokkasecret" not in warned.getMessage()
    assert debug.getMessage() == "Step 3 failed: the model said: Quokkasecret"


def test_log_failure_logs_the_traceback_at_debug_only_when_asked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unified_kg_rag.shared import get_logger
    from unified_kg_rag.shared.logging import log_failure

    try:
        raise RuntimeError("the provider said: Quokkasecret")
    except RuntimeError as e:
        error = e
    with caplog.at_level("DEBUG"):
        log_failure(get_logger("probe"), logging.ERROR, "x %s", error=error)
        log_failure(
            get_logger("probe"), logging.ERROR, "x %s", error=error, traceback=True
        )
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert all(r.exc_info is None for r in errors)
    assert [r.exc_info is not None for r in debug] == [False, True]


def _raised_through_two_frames() -> RuntimeError:
    def inner() -> None:
        raise KeyError("Quokkasecret payroll")

    def outer() -> None:
        try:
            inner()
        except KeyError as cause:
            raise RuntimeError("the provider said: Zebrapassword") from cause

    try:
        outer()
    except RuntimeError as e:
        return e
    raise AssertionError("unreachable")


def test_an_unexpected_failure_is_locatable_at_info_without_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression (rv28): at INFO a failure was only ``Type (len=, sha=)``, so
    the surfaces' "the server log has the details" located nothing."""
    from unified_kg_rag.shared import get_logger
    from unified_kg_rag.shared.logging import log_failure
    from unified_kg_rag.shared.utils import error_locator

    error = _raised_through_two_frames()
    with caplog.at_level("DEBUG"):
        log_failure(
            get_logger("probe"), logging.ERROR, "x: %s", error=error, traceback=True
        )
    (shown,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    text = shown.getMessage()
    assert text.startswith("x: RuntimeError (len=")
    # The type chain, outermost first, and each one's frames: file:line name.
    assert "RuntimeError at " in text and " <- KeyError at " in text
    assert "test_log_redaction.py:" in text
    assert "outer" in text and "inner" in text
    _assert_no_secret(caplog)
    # An error never raised has no frames to name.
    assert error_locator(RuntimeError("never raised")) == ""
    # The text and traceback are DEBUG's.
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert "Zebrapassword" in debug[0].getMessage()


def test_a_traceback_free_failure_log_adds_no_location(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unified_kg_rag.shared import get_logger
    from unified_kg_rag.shared.logging import log_failure

    with caplog.at_level("INFO"):
        log_failure(
            get_logger("probe"),
            logging.WARNING,
            "x: %s",
            error=_raised_through_two_frames(),
        )
    assert "<-" not in caplog.records[0].getMessage()
    assert " at " not in caplog.records[0].getMessage()


def test_log_error_logs_an_outage_with_its_text_and_a_failure_without(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one rule of every surface's last-resort line: an expected outcome
    with its own message (a store down: its endpoint is not user data) is
    logged in full, an unexpected failure by type and location."""
    from unified_kg_rag.shared import StoreConnectionError, get_logger
    from unified_kg_rag.shared.logging import log_error

    logger = get_logger("probe")
    with caplog.at_level("INFO"):
        log_error(
            logger,
            logging.ERROR,
            "call failed: %s",
            error=StoreConnectionError("cannot reach https://store.internal:9200"),
        )
        log_error(
            logger,
            logging.ERROR,
            "call failed: %s",
            error=_raised_through_two_frames(),
        )
    outage, failure = (r.getMessage() for r in caplog.records)
    assert outage == (
        "call failed: [store_unreachable] cannot reach https://store.internal:9200"
    )
    assert failure.startswith("call failed: RuntimeError (len=")
    assert "Zebrapassword" not in failure


def test_an_error_digest_names_its_code_status_and_data_free_facts() -> None:
    """Regression (rv29 logging #3): at INFO a failure said only its type and
    a hash, though its HTTP status, a service's error code, an OSError's
    errno and path or a decode error's position hold no request text."""
    import json as json_module

    from botocore.exceptions import ClientError

    from unified_kg_rag.shared.utils import error_digest, error_name

    class ResponseError(Exception):
        status_code = 400

    reply = ResponseError("Quokkasecret: the prompt's tail")
    assert error_digest(reply).startswith("ResponseError, HTTP 400 (len=")
    throttled = ClientError(
        {
            "Error": {"Code": "ThrottlingException", "Message": "Quokkasecret"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "Converse",
    )
    assert error_name(throttled) == "ClientError ThrottlingException, HTTP 429"
    denied = PermissionError(13, "custom text Quokkasecret", "/exports/graph")
    assert error_name(denied) == (
        "PermissionError [Errno 13] Permission denied: '/exports/graph'"
    )
    try:
        json_module.loads('{"q": Quokkasecret}')
    except json_module.JSONDecodeError as e:
        assert error_name(e) == "JSONDecodeError Expecting value: line 1 column 7"
    try:
        b"\xff Quokkasecret".decode("utf-8")
    except UnicodeDecodeError as e:
        assert error_name(e) == (
            "UnicodeDecodeError 'utf-8' invalid start byte at position 0"
        )
    for error in (reply, throttled, denied):
        assert "Quokkasecret" not in error_digest(error)


def test_the_locator_follows_the_context_unless_it_was_suppressed() -> None:
    from unified_kg_rag.shared.utils import error_locator

    def handled(suppress: bool) -> BaseException:
        try:
            try:
                raise KeyError("Quokkasecret")
            except KeyError:
                if suppress:
                    raise RuntimeError("Zebrapassword") from None
                raise RuntimeError("Zebrapassword")  # noqa: B904 - the context
        except RuntimeError as e:
            return e
        raise AssertionError("unreachable")

    shown = error_locator(handled(suppress=False))
    assert shown.startswith("RuntimeError at ") and " <- KeyError at " in shown
    assert "KeyError" not in error_locator(handled(suppress=True))
    assert "Quokkasecret" not in shown and "Zebrapassword" not in shown


def test_the_locator_names_no_absolute_path(tmp_path: Path) -> None:
    import importlib.util

    from unified_kg_rag.shared.utils import error_locator

    # A frame of a file outside the package: its last two parts only.
    outside = tmp_path / "probe" / "tool.py"
    outside.parent.mkdir()
    outside.write_text("def fail():\n    raise ValueError('x')\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("probe_tool", outside)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError) as failed:
        module.fail()
    shown = error_locator(failed.value)
    assert "probe/tool.py:2 fail" in shown
    assert str(tmp_path) not in shown
    # The package's files by their path from the package.
    from unified_kg_rag.shared.utils import common

    with pytest.raises(AttributeError) as raised:
        common.compute_hash(None)  # type: ignore[arg-type]
    shown = error_locator(raised.value)
    assert "unified_kg_rag/shared/utils/common.py:" in shown
    assert str(Path(common.__file__).parent) not in shown


def test_an_export_write_failure_logs_its_folder(
    caplog: pytest.LogCaptureFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (rv29): a graph write failure logged "ExportWriteError
    # (len sha) <- PermissionError" without the folder the page said to fix.
    from unified_kg_rag.application.storage import visualization_export
    from unified_kg_rag.shared import get_logger
    from unified_kg_rag.shared.logging import log_error

    target = tmp_path / "exports" / "visualization_data.json"

    def refuse(path: Path, _text: str, **_: Any) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(visualization_export, "atomic_write_text", refuse)
    monkeypatch.setattr(
        visualization_export, "stored_visualization_data", lambda *a, **k: {}
    )
    providers = MagicMock()
    providers.build_vector_indexer.return_value.scan_items.return_value = []
    with pytest.raises(visualization_export.ExportWriteError) as raised:
        visualization_export.write_visualization_data(providers, "default", target)
    with caplog.at_level("INFO"):
        log_error(get_logger("probe"), logging.WARNING, "x: %s", error=raised.value)
    (line,) = [r.getMessage() for r in caplog.records]
    assert f"Cannot write '{target}'" in line
    assert "PermissionError [Errno 13] Permission denied" in line


def test_a_partly_own_message_shows_and_logs_its_own_part_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A stage failure names its index types (the package's text) and its
    # first error (a store's, which may quote a document): the first part
    # is shown and logged at ERROR, the whole message only at DEBUG.
    from unified_kg_rag.shared import PipelineStageError, get_logger
    from unified_kg_rag.shared.exceptions import operator_error_text, own_text
    from unified_kg_rag.shared.logging import log_error

    own = "Indexing failed: vector_entities exceeded the tolerated failure rate."
    error = PipelineStageError(f"{own} First error: Quokkasecret", own_text=own)
    assert own_text(error) == own and operator_error_text(error) == own
    with caplog.at_level("DEBUG"):
        log_error(get_logger("probe"), logging.ERROR, "Stage failed: %s", error=error)
    (shown,) = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert shown == f"Stage failed: [failure] {own}"
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Quokkasecret" in line for line in debug)
