# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One classification of a query failure, one table per surface.

Each surface classified errors by its own ``isinstance`` list: an
``InvalidFilterError`` exited 2 from ``run-rag`` and 1 from ``run-eval``, a
``--suffix`` nothing was ingested under exited 1 under ``ignore_errors``
(the docs promise 2), and the console lost the kind of an absorbed failure.
``classify(exc)`` is the one rule; ``EXIT_CODES`` (the CLIs), the MCP
server's caller classes and the console's kind map its classes. A model
server that cannot be reached is its own class, named with its provider
kind and ``base_url``.
"""

from __future__ import annotations

import inspect
import socket
import sys
from typing import Any

import httpx
import pytest

from unified_kg_rag.adapters.llm.reachability import (
    is_unreachable_error,
    model_unreachable_error,
)
from unified_kg_rag.application.cli import run_evaluation, run_rag_chain
from unified_kg_rag.application.cli.console.ask import _error_kind
from unified_kg_rag.application.cli.conventions import EXIT_CODES
from unified_kg_rag.application.cli.mcp_tools import (
    describe_error,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import exceptions as x
from unified_kg_rag.shared.exceptions import INTERNAL_ERROR_MESSAGE

pytestmark = pytest.mark.unit

_CASES: list[tuple[BaseException, x.ErrorClass, int, bool]] = [
    # (error, class, CLI exit code, message for an MCP caller)
    (x.InvalidFilterError("f"), x.ErrorClass.INVALID_FILTER, 2, True),
    (x.InvalidArgumentError("a"), x.ErrorClass.INVALID_ARGUMENT, 2, True),
    # A document id to fix, as an argument is (it was a failure, exit 1).
    (x.DocumentNotFoundError("d"), x.ErrorClass.INVALID_ARGUMENT, 2, True),
    (x.MethodologyNotIndexedError("m"), x.ErrorClass.NOT_INDEXED, 2, True),
    (x.IndexNotFoundError("i"), x.ErrorClass.INDEX_NOT_FOUND, 2, True),
    (x.ConfigurationError("c"), x.ErrorClass.CONFIGURATION, 2, False),
    (x.SchemaSourceError("s"), x.ErrorClass.CONFIGURATION, 2, False),
    (x.StoreAccessDeniedError("denied"), x.ErrorClass.CONFIGURATION, 2, False),
    (x.QueryTimeoutError("t", timeout_seconds=1), x.ErrorClass.TIMEOUT, 1, True),
    (x.StoreConnectionError("down"), x.ErrorClass.STORE_UNREACHABLE, 1, False),
    (x.StoreTimeoutError("slow"), x.ErrorClass.STORE_UNREACHABLE, 1, False),
    (x.ModelUnreachableError("m"), x.ErrorClass.MODEL_UNREACHABLE, 1, False),
    (x.GuardrailError("g"), x.ErrorClass.FAILURE, 1, False),
    (RuntimeError("r"), x.ErrorClass.FAILURE, 1, False),
]


@pytest.mark.parametrize(
    ("error", "error_class", "exit_code", "callers"),
    _CASES,
    ids=[type(c[0]).__name__ for c in _CASES],
)
def test_every_surface_reads_one_classification(
    error: BaseException, error_class: x.ErrorClass, exit_code: int, callers: bool
) -> None:
    assert x.classify(error) is error_class
    assert EXIT_CODES[error_class] == exit_code
    assert describe_error(error)[1] is callers
    # The MCP message starts with the class, the stable key a caller reads.
    message = describe_error(error)[0]
    assert message.startswith(f"[{error_class.value}] ")
    if not callers:
        assert message == f"[{error_class.value}] {INTERNAL_ERROR_MESSAGE}"
    # The console's kind is the class (its page names each).
    assert _error_kind(error if isinstance(error, Exception) else Exception()) == (
        error_class.value
    )
    # One rule for the text each surface shows: an MCP caller sees what it
    # can act on, the console's operator all but an unexpected failure's.
    assert x.shows_error_text(error_class, "caller") is callers
    assert x.shows_error_text(error_class, "operator") is (
        error_class is not x.ErrorClass.FAILURE
    )
    assert x.shown_error_text(error, audience="operator") == (
        INTERNAL_ERROR_MESSAGE
        if error_class is x.ErrorClass.FAILURE
        else x.error_text(error)
    )


def test_the_former_name_is_the_same_class() -> None:
    assert x.ToolArgumentError is x.InvalidArgumentError


def test_every_class_has_an_exit_code() -> None:
    assert set(EXIT_CODES) == set(x.ErrorClass)
    assert x.error_class_of("index_not_found") is x.ErrorClass.INDEX_NOT_FOUND
    assert x.error_class_of(None) is x.ErrorClass.FAILURE
    assert x.error_class_of("unknown") is x.ErrorClass.FAILURE


def _openai_errors() -> tuple[BaseException, BaseException]:
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    return openai.APIConnectionError(request=request), openai.APITimeoutError(
        request=request
    )


def _raised_from(error: BaseException, cause: BaseException) -> BaseException:
    try:
        raise error from cause
    except BaseException as e:  # noqa: BLE001 - the chained error
        return e


@pytest.mark.parametrize(
    ("error", "unreachable"),
    [
        (httpx.ConnectError("All connection attempts failed"), True),
        (httpx.ConnectTimeout("timed out"), True),
        (httpx.ReadTimeout("no answer"), False),
        (ConnectionRefusedError(111, "refused"), True),
        # The Ollama client's own "Failed to connect to Ollama".
        (ConnectionError("Failed to connect to Ollama"), True),
        (ConnectionResetError("dropped"), False),
        (socket.gaierror(-2, "Name or service not known"), True),
        # A store's is the store's, whatever it was raised from.
        (
            _raised_from(x.StoreConnectionError("down"), ConnectionRefusedError()),
            False,
        ),
        # The query embedding wraps the client's error.
        (
            _raised_from(
                x.EmbeddingModelError("Query embedding failed"),
                ConnectionError("Failed to connect to Ollama"),
            ),
            True,
        ),
        (ValueError("bad"), False),
    ],
)
def test_an_unreachable_model_server_is_told_apart(
    error: BaseException, unreachable: bool
) -> None:
    assert is_unreachable_error(error) is unreachable


def test_a_model_clients_own_error_is_named_a_language_model_error() -> None:
    # Regression (rv24): ainvoke on a failing model raised Ollama's raw
    # ResponseError or httpx's RemoteProtocolError / ReadTimeout, not the
    # ModelError the API documents.
    from unified_kg_rag.adapters.llm.reachability import named_model_error

    class ResponseError(Exception):  # ollama's, with its status
        status_code = 404

    config = Config()
    for raw in (
        ResponseError("model 'x' not found"),
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.ReadTimeout("timed out"),
    ):
        named = named_model_error(raw, config)
        assert isinstance(named, x.LanguageModelError), raw
        assert type(raw).__name__ in str(named)
    assert isinstance(
        named_model_error(httpx.ConnectError("refused"), config),
        x.ModelUnreachableError,
    )
    for kept in (RuntimeError("a bug"), x.StoreConnectionError("down"), KeyError("k")):
        assert named_model_error(kept, config) is kept


def test_the_openai_clients_connection_error_is_unreachable_not_its_timeout() -> None:
    connection, timeout = _openai_errors()
    assert is_unreachable_error(connection)
    assert not is_unreachable_error(timeout)


def test_the_message_names_each_server_and_how_to_start_it() -> None:
    config = Config(
        backend="local",
        models={
            "default": "chat-m",
            "embedding": {"model_id": "emb-m", "dimension": 8},
        },
    )
    config.providers.embedding.kind = "openai_compatible"
    config.providers.embedding.base_url = "http://10.0.0.5:8000/v1"
    error = model_unreachable_error(httpx.ConnectError("refused"), config)
    message = str(error)
    assert isinstance(error, x.ModelUnreachableError)
    assert "providers.llm (ollama at http://localhost:11434)" in message
    assert "providers.embedding (openai_compatible at http://10.0.0.5:8000/v1)" in (
        message
    )
    assert "ollama serve" in message and "chat-m" in message
    # The embedder is no Ollama server's model to pull.
    assert "emb-m" not in message
    assert model_unreachable_error(error, config) is error
    config = Config()
    bedrock = model_unreachable_error(httpx.ConnectError("x"), config)
    runtime = f"https://bedrock-runtime.{config.aws.bedrock_region}.amazonaws.com"
    assert f"providers.llm and providers.embedding (bedrock at {runtime})" in str(
        bedrock
    )
    assert "ollama" not in str(bedrock) and "aws.bedrock_region" in str(bedrock)


def test_the_message_names_a_server_without_the_credentials_of_its_url() -> None:
    # Regression (rv24): the message, logged, printed in run-rag's JSON,
    # returned to MCP callers and sent in the failure notification, held
    # the base_url's userinfo and query.
    config = Config(backend="local", models={"default": "chat-m"})
    config.providers.llm.kind = "openai_compatible"
    config.providers.llm.base_url = "http://alice:S3cretTok@127.0.0.1:9/v1?api-key=QQQ"
    message = str(model_unreachable_error(httpx.ConnectError("refused"), config))
    assert "providers.llm (openai_compatible at http://127.0.0.1:9/v1)" in message
    assert "S3cretTok" not in message and "QQQ" not in message


def test_a_bedrock_server_is_named_by_its_endpoint_not_a_base_url() -> None:
    # Regression (v23 B6): backend local with the models on bedrock named
    # "bedrock at http://localhost:11434", the local preset's base_url.
    from botocore.exceptions import EndpointConnectionError

    config = Config(backend="local")
    config.providers.llm.kind = config.providers.embedding.kind = "bedrock"
    config.aws.bedrock.region_name = "us-west-2"
    runtime = model_unreachable_error(
        EndpointConnectionError(
            endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com/model/x"
        ),
        config,
    )
    assert "(bedrock at https://bedrock-runtime.us-west-2.amazonaws.com)" in str(
        runtime
    )
    assert "11434" not in str(runtime)
    # Another Bedrock endpoint (the control plane): the one that failed.
    control = model_unreachable_error(
        EndpointConnectionError(
            endpoint_url="https://bedrock.us-west-9.amazonaws.com/"
        ),
        config,
    )
    assert "(bedrock at https://bedrock.us-west-9.amazonaws.com)" in str(control)


def test_the_message_names_the_server_the_connection_failed_to_reach() -> None:
    # Regression (rv8 retrieval low): with only the language model server
    # down, the message named the embedding server too.
    config = Config(
        backend="local",
        models={
            "default": "chat-m",
            "embedding": {"model_id": "emb-m", "dimension": 8},
        },
    )
    config.providers.embedding.kind = "openai_compatible"
    config.providers.embedding.base_url = "http://10.0.0.5:8000/v1"

    def refused(url: str) -> httpx.ConnectError:
        return httpx.ConnectError("refused", request=httpx.Request("POST", url))

    llm = str(
        model_unreachable_error(refused("http://localhost:11434/api/chat"), config)
    )
    assert "providers.llm (ollama at http://localhost:11434)" in llm
    assert "providers.embedding" not in llm
    assert "chat-m" in llm and "emb-m" not in llm
    embedding = str(
        model_unreachable_error(refused("http://10.0.0.5:8000/v1/embeddings"), config)
    )
    assert "providers.embedding (openai_compatible" in embedding
    assert "providers.llm" not in embedding and "ollama serve" not in embedding
    # A URL no configured server has: every one is named.
    other = str(model_unreachable_error(refused("http://elsewhere:1/x"), config))
    assert "providers.llm" in other and "providers.embedding" in other


@pytest.mark.parametrize(
    ("error_class", "exit_code"),
    [("index_not_found", 2), ("invalid_filter", 2), ("model_unreachable", 1)],
)
async def test_run_rag_exits_by_the_class_of_an_absorbed_failure(
    mocker: Any, capsys: pytest.CaptureFixture[str], error_class: str, exit_code: int
) -> None:
    # Regression (rv7 unify #9): under processing.ignore_errors the chain
    # answers its error answer, and a --suffix nothing was ingested under
    # exited 1 where the docs promise 2.
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput

    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi", "--suffix", "tenant-b"]
    )
    runner.metrics_sink = mocker.MagicMock()
    output = mocker.MagicMock()
    output.model_dump.return_value = {
        "answer": "error",
        "metadata": {
            "error": True,
            "error_type": "SomeError",
            "error_class": error_class,
        },
        "search_results": {"metadata": {"error": "No indices found for 'tenant-b'"}},
    }
    runner.rag_chain = mocker.MagicMock()
    runner.rag_chain.ainvoke = mocker.AsyncMock(return_value=output)
    if exit_code == 2:
        with pytest.raises(SystemExit) as exited:
            await runner._run_query(RAGInput(query="hi"))
        assert exited.value.code == 2
        assert "No indices found for 'tenant-b'" in capsys.readouterr().err
    else:
        result = await runner._run_query(RAGInput(query="hi"))
        assert result["success"] is False and result["exit_code"] == 1
        assert result["error_type"] == "SomeError"
        assert result["error"] == "No indices found for 'tenant-b'"


def test_run_eval_exits_2_for_a_filter_no_index_declares(mocker: Any) -> None:
    # Regression (rv7 unify #9): run-rag exited 2, run-eval 1.
    config = Config()
    config.evaluation.enabled_evaluators = []
    mocker.patch.object(run_evaluation, "load_cli_config", return_value=config)
    mocker.patch.object(run_evaluation, "display_ascii_art")
    mocker.patch.object(sys, "argv", ["run-eval", "--data-path", "d.json"])
    mocker.patch.object(run_evaluation, "missing_endpoints_error", return_value=None)
    mocker.patch.object(run_evaluation, "GraphRAGChain")

    def _fail(coro: Any, **_kwargs: Any) -> Any:
        inner = coro.cr_frame.f_locals.get("main")
        if inspect.iscoroutine(inner):
            inner.close()
        coro.close()
        raise x.InvalidFilterError("No index declares filter key 'colour'")

    mocker.patch.object(run_evaluation.event_loop, "run", side_effect=_fail)
    with pytest.raises(SystemExit) as exited:
        run_evaluation.main()
    assert exited.value.code == 2


def test_the_console_names_the_class_of_an_absorbed_failure() -> None:
    # Regression (rv7 unify #9): an error answer (ignore_errors) reached the
    # Ask page without its kind.
    from unified_kg_rag.application.cli.console.ask import AskService
    from unified_kg_rag.application.retrieval.rag_chain import (
        ProcessedQuery,
        RAGOutput,
    )
    from unified_kg_rag.domain.models import SearchQuery, SearchResult

    def output(metadata: dict[str, Any]) -> RAGOutput:
        return RAGOutput(
            answer="error",
            sources=[],
            search_results=SearchResult(
                query=SearchQuery(query="q"),
                results=[],
                total_results=0,
                search_strategy="error",
                processing_time=0.0,
                metadata={"error": "A model server could not be reached"},
            ),
            conversation_id=None,
            processed_query=ProcessedQuery(original_query="q", final_query="q"),
            metadata=metadata,
        )

    failed = output(
        {
            "error": True,
            "error_type": "ModelUnreachableError",
            "error_class": "model_unreachable",
        }
    )
    result = AskService._result(failed, Config(), 0.1)
    assert result["error_kind"] == "model_unreachable"
    assert AskService._result(output({}), Config(), 0.1)["error_kind"] is None


async def test_waiters_of_a_shared_call_each_get_their_own_failure() -> None:
    import asyncio

    from unified_kg_rag.shared.utils.concurrency import await_shared

    shared: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    original = x.QueryTimeoutError("late", timeout_seconds=2.0)
    x.tag_query_failure(original, "q-first")
    shared.set_exception(original)
    raised = await asyncio.gather(
        await_shared(shared), await_shared(shared), return_exceptions=True
    )
    first, second = raised
    assert isinstance(first, x.QueryTimeoutError)
    assert isinstance(second, x.QueryTimeoutError)
    assert first is not second and original not in raised
    # The type and attributes the surfaces read, the shared one as the
    # cause, and no other query's ids.
    assert first.timeout_seconds == 2.0 and str(first) == "late"
    assert first.__cause__ is original
    assert x.classify(first) is x.ErrorClass.TIMEOUT
    assert x.query_failure_ids(first) == {}
    x.tag_query_failure(first, "q-1")
    x.tag_query_failure(second, "q-2")
    assert {x.query_failure_ids(e)["query_id"] for e in raised} == {"q-1", "q-2"}


async def test_a_waiters_own_cancellation_is_not_copied() -> None:
    import asyncio

    from unified_kg_rag.shared.utils.concurrency import await_shared

    shared: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiter = asyncio.ensure_future(await_shared(shared))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not shared.cancelled()
    shared.set_result(None)
