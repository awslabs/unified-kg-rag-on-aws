# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider's, a store's or a library's error text stays out of INFO+ logs
and the CLIs' output, closed where it enters (AWS-free).

Each test plants a marker in an error's text at one source (a write's
``IndexingStats.errors``, a gathered query input, the chain's error answer,
a visualization export, a wrapped store error, a library's record or
warning, a saved evaluation result) and checks no INFO+ record and no
output quotes it, while the package's own words (an operation, a URI, an
error's type, code and status) still are.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unified_kg_rag.domain.models import Config, Relationship
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import AWSServiceError, LanguageModelError, get_logger
from unified_kg_rag.shared.exceptions import (
    INTERNAL_ERROR_MESSAGE,
    operator_error_text,
    own_text,
    result_error_text,
    with_own_text,
)
from unified_kg_rag.shared.logging import (
    LibraryRecordFilter,
    error_log_text,
    log_error,
    setup_logging,
)
from unified_kg_rag.shared.utils import error_name

pytestmark = pytest.mark.unit

MARKER = "Quokkasecret 홍길동 arn:aws:iam::000000000000:role/Zebrapassword"
_FRAGMENTS = ("Quokkasecret", "홍길동", "Zebrapassword", "000000000000")


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
    )


def _debug_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )


def _assert_no_marker(*texts: str) -> None:
    for text in texts:
        for fragment in _FRAGMENTS:
            assert fragment not in text, text


# --- IndexingStats.add_error: the one way into a write's errors ----------


def test_a_recorded_store_error_is_named_and_its_text_logged_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stats = IndexingStats(total_items=3)
    with caplog.at_level(logging.DEBUG):
        stats.add_error(RuntimeError(MARKER), 2, ids=["a", "b"])
        stats.add_error(RuntimeError(MARKER), 1, context="Bulk indexing failed")
        stats.add_error("edge not stored", 1)
    _assert_no_marker(*stats.errors, json.dumps(stats.to_dict(), ensure_ascii=False))
    assert stats.errors[0].startswith("RuntimeError (len=")
    assert stats.errors[1].startswith("Bulk indexing failed: RuntimeError (len=")
    # The package's own message stays as it is.
    assert stats.errors[2] == "edge not stored"
    assert stats.failed_items == 4 and stats.failed_ids == ["a", "b"]
    assert "Quokkasecret" in _debug_text(caplog)


def test_a_recorded_error_of_another_class_keeps_its_text() -> None:
    from unified_kg_rag.shared import StoreConnectionError

    stats = IndexingStats()
    stats.add_error(StoreConnectionError("Neptune at localhost:8182 refused"))
    stats.add_error(AWSServiceError.caused_by("Neptune read_entities failed", _boom()))
    assert stats.errors == [
        "[store_unreachable] Neptune at localhost:8182 refused",
        "[failure] Neptune read_entities failed: RuntimeError",
    ]


def _boom() -> RuntimeError:
    return RuntimeError(MARKER)


@pytest.mark.usefixtures("skip_vertex_id_check")
def test_a_failed_neptune_edge_is_logged_and_recorded_without_its_text(
    mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression (rv30 logging #1): the per-edge worker returned str(e),
    # logged at WARNING and kept in stats.errors (indexing_manager logs
    # them at WARNING again).
    from tests.fixtures.fakes.gremlin import RecordingTraversal

    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer._sleep")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    indexer = NeptuneIndexer(config=Config())
    indexer.neptune_client.g = RecordingTraversal([])
    rels = [Relationship(id="r1", source_id="e1", target_id="e2")]

    def execute(_traversal: Any, op: str, **_: Any) -> Any:
        if op == "Reading edge endpoints":
            return [{"id": end, "vertex": f"v-{end}"} for end in ("e1", "e2")]
        if op == "Relationship indexing":
            raise RuntimeError(f"Neptune rejected: {MARKER} property value")
        return []

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)
    with caplog.at_level(logging.INFO):
        stats = indexer.upsert_relationships(rels)
    assert stats.failed_ids == ["r1"]
    _assert_no_marker(_info_text(caplog), *stats.errors)
    assert "Relationship 'r1' not written: RuntimeError (len=" in _info_text(caplog)


# --- A gathered query input that failed ------------------------------------


async def test_a_failed_query_entity_extraction_is_logged_without_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (rv30 logging #2): under processing.ignore_errors the
    # gathered outcome was logged raw at WARNING ("Entity extraction
    # failed: <provider reply>"), past the guard of except clauses.
    from tests.unit.test_log_redaction import _chain
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput
    from unified_kg_rag.domain.models import SearchStrategy

    config = Config()
    config.processing.ignore_errors = True
    chain = _chain(config)

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"ValidationException: {MARKER}")

    chain._extract_query_entities = fail  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG):
        output = await chain.ainvoke(
            RAGInput(
                query="Who supplies parts?",
                search_strategy=SearchStrategy.LOCAL,
                enable_query_processing=True,
            )
        )
    info = _info_text(caplog)
    _assert_no_marker(info)
    assert "Entity extraction failed: RuntimeError (len=" in info
    assert "Quokkasecret" in _debug_text(caplog)
    assert output.metadata.get("degraded")


# --- The chain's error answer ----------------------------------------------


async def test_the_error_answer_s_search_metadata_holds_no_provider_text() -> None:
    # Regression (rv30 logging #3): run-rag --output-format json printed
    # search_results.metadata.error = str(e) under ignore_errors.
    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from tests.fixtures.fakes.retrievers import StaticRetriever, hit
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        RAGInput,
    )
    from unified_kg_rag.domain.models import RetrieverRole, SearchStrategy

    def fail(_system: str, _human: str) -> str:
        raise RuntimeError(MARKER)

    config = Config()
    config.processing.ignore_errors = True
    retriever = StaticRetriever([hit("Vendor supplies parts.")])
    chain = GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(fail),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    output = await chain.ainvoke(
        RAGInput(
            query="q",
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
        )
    )
    assert output.metadata["error"] is True
    shown = output.search_results.metadata["error"]
    _assert_no_marker(json.dumps(output.model_dump(mode="json"), ensure_ascii=False))
    assert shown == f"RuntimeError: {INTERNAL_ERROR_MESSAGE}"


# --- run-visualization's invalid data file ---------------------------------


def test_an_undecodable_export_is_refused_without_its_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (rv30 logging #4): {e!r} of a UnicodeDecodeError holds the
    # whole file's bytes, shown on stdout and logged at ERROR.
    from unified_kg_rag.application.cli import run_visualization

    data = tmp_path / "viz.json"
    body = json.dumps({"nodes": [{"id": "n", "attributes": {"d": MARKER}}]})
    data.write_bytes(body.encode() + b"\xff")
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-visualization",
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
            "--output-format",
            "json",
        ],
    )
    with caplog.at_level(logging.INFO):
        code = run_visualization.main()
    out = capsys.readouterr()
    assert code == 2
    _assert_no_marker(out.out, out.err, _info_text(caplog))
    assert "UnicodeDecodeError 'utf-8' invalid start byte at position" in out.out


def test_a_refused_export_names_the_invalid_keys_not_their_values(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.application.cli.run_visualization import (
        InvalidVisualizationDataError,
        load_render_context,
    )

    data = tmp_path / "viz.json"
    data.write_text(
        json.dumps(
            {
                "nodes": [],
                "communities": [
                    {"community_id": "c1", "level": MARKER},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InvalidVisualizationDataError) as raised:
        load_render_context(data)  # int(level): its ValueError quotes the value
    _assert_no_marker(str(raised.value))
    assert str(raised.value).endswith("visualization data: ValueError")
    data.write_text(
        json.dumps({"communities": [{"community_id": "c", "parent": {"n": MARKER}}]}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidVisualizationDataError) as raised:
        load_render_context(data)  # a validation error: its keys, not values
    _assert_no_marker(str(raised.value))
    assert "parent: Input should be a valid string" in str(raised.value)


# --- A store error wrapped with the package's context ----------------------


def test_a_wrapped_error_shows_its_context_and_names_its_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # rv30 ux U1: what an AWSServiceError says on a surface and at ERROR is
    # its own text (the operation, the URI); the provider's reply is DEBUG's.
    cause = RuntimeError(MARKER)
    error = AWSServiceError.caused_by(
        "Failed to download 's3://bucket/eval.json'", cause
    )
    assert MARKER in str(error)  # the Python API caller still has it
    assert own_text(error) == "Failed to download 's3://bucket/eval.json': RuntimeError"
    assert operator_error_text(error) == own_text(error)
    with caplog.at_level(logging.DEBUG):
        log_error(get_logger("probe"), logging.ERROR, "Failed: %s", error=error)
    _assert_no_marker(_info_text(caplog))
    assert "s3://bucket/eval.json" in _info_text(caplog)
    assert "Quokkasecret" in _debug_text(caplog)


def test_an_s3_download_failure_keeps_its_uri_and_masks_the_reply() -> None:
    from botocore.exceptions import ClientError

    from unified_kg_rag.adapters.aws.s3_transfer import download_s3_object

    class _Client:
        def download_file(self, *_args: Any, **_kwargs: Any) -> None:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": MARKER},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            )

    with pytest.raises(AWSServiceError) as raised:
        download_s3_object(_Client(), "s3://bucket/eval.json", Path("/tmp/x"))
    shown = operator_error_text(raised.value)
    _assert_no_marker(shown)
    assert shown == (
        "Failed to download 's3://bucket/eval.json': "
        "ClientError AccessDenied, HTTP 403"
    )


# --- Saved results and the export failure ----------------------------------


def test_a_saved_result_names_a_failure_by_class_and_type() -> None:
    assert result_error_text(RuntimeError(MARKER)) == "[failure] RuntimeError"
    own = with_own_text(LanguageModelError("no label for assertion(s) [2]"))
    assert result_error_text(own).endswith(": no label for assertion(s) [2]")


def test_a_refused_data_file_names_the_key_it_missed() -> None:
    from unified_kg_rag.shared.utils.validation import refused_data_reason

    assert refused_data_reason(KeyError("question")) == "missing key 'question'"
    assert refused_data_reason(KeyError()) == "KeyError"


def test_a_saved_result_names_a_wrapped_failures_cause_once() -> None:
    from botocore.exceptions import ClientError

    cause = ClientError(
        {
            "Error": {"Code": "ThrottlingException", "Message": MARKER},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "Read",
    )
    try:
        raise AWSServiceError.caused_by(
            "Neptune read_entities failed", cause
        ) from cause
    except AWSServiceError as error:
        shown = result_error_text(error)
    _assert_no_marker(shown)
    assert shown == (
        "[failure] AWSServiceError: Neptune read_entities failed: "
        "ClientError ThrottlingException, HTTP 429"
    )


def test_a_failed_export_names_its_folder_at_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # rv30 ux low: the export failure was logged by log_failure, which
    # ignores the error's own text (the folder to fix).
    from tests.unit.test_visualization_export import _pipeline
    from unified_kg_rag.application.storage.visualization_export import (
        ExportWriteError,
    )

    def fail(*_: Any) -> dict[str, int]:
        raise ExportWriteError(f"Cannot write '{tmp_path}/v.json': PermissionError")

    monkeypatch.setattr(
        "unified_kg_rag.application.ingestion.pipeline.write_visualization_data", fail
    )
    pipeline = _pipeline(Config(), {"indexing"}, tmp_path)
    context = SimpleNamespace(
        pipeline_id="pid",
        visualization_data_path=None,
        stage_results=[],
        incremental_delta=None,
    )
    with caplog.at_level(logging.INFO):
        pipeline._export_visualization(context)
    assert f"Cannot write '{tmp_path}/v.json'" in _info_text(caplog)


# --- error_name -------------------------------------------------------------


def test_error_name_reads_each_library_s_code_and_names_its_status() -> None:
    import socket
    import ssl

    gai = socket.gaierror(-2, "Name or service not known")
    assert error_name(gai) == "gaierror [Errno -2] Name or service not known"
    tls = ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate failed")
    tls.library, tls.reason = "SSL", "CERTIFICATE_VERIFY_FAILED"  # type: ignore[misc]
    assert error_name(tls) == "SSLError SSL CERTIFICATE_VERIFY_FAILED"
    assert "Operation not permitted" not in error_name(ssl.SSLError(1, "x"))

    class OpenSearchLike(Exception):
        status_code = 404
        error = "raw body MAY QUOTE"
        info = {"error": {"type": "index_not_found_exception", "reason": MARKER}}

    assert error_name(OpenSearchLike()) == (
        "OpenSearchLike index_not_found_exception, HTTP 404"
    )

    class OpenAILike(Exception):
        status_code = 429
        code = "rate_limit_exceeded"

    assert error_name(OpenAILike()) == "OpenAILike rate_limit_exceeded, HTTP 429"


def test_a_gremlin_status_is_not_called_http() -> None:
    pytest.importorskip("gremlin_python")
    from gremlin_python.driver.protocol import GremlinServerError

    error = GremlinServerError(
        {"code": 597, "message": MARKER, "attributes": {"exceptions": []}}
    )
    assert error_name(error) == "GremlinServerError, Gremlin status 597"


def test_a_package_wrapper_names_its_cause_s_status() -> None:
    class ResponseError(Exception):
        status_code = 500

    try:
        try:
            raise ResponseError(MARKER)
        except ResponseError as cause:
            raise LanguageModelError("A language model call failed") from cause
    except LanguageModelError as wrapped:
        name = error_name(wrapped)
    assert name == "LanguageModelError from ResponseError, HTTP 500"
    # A library's own error is named alone, its context left out.
    try:
        try:
            raise KeyError("x")
        except KeyError as cause:
            raise ValueError("y") from cause
    except ValueError as plain:
        assert error_name(plain) == "ValueError"


# --- The library-record filter and warnings --------------------------------


def _setup(tmp_path: Path) -> tuple[io.StringIO, Path]:
    config = Config()
    config.logging.log_to_file = True
    config.logging.log_file_path = str(tmp_path / "run.log")
    config.logging.log_format = "structured"
    stream = io.StringIO()
    setup_logging(config, stream=stream)
    (path,) = tmp_path.glob("run_*.log")
    return stream, path


def _flush() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_a_library_record_is_digested_once_for_every_handler(tmp_path: Path) -> None:
    # rv30 low: the second handler hashed the first's digest, so the console
    # and the log file disagreed.
    stream, path = _setup(tmp_path)
    try:
        logging.getLogger("thirdparty").warning("Endpoint %s returned", MARKER)
        logging.getLogger("thirdparty").error(RuntimeError(MARKER))
        _flush()
        console = [json.loads(line)["event"] for line in stream.getvalue().splitlines()]
        logged = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    finally:
        setup_logging(Config())
    _assert_no_marker(*console, *logged)
    assert console[-2:] == logged[-2:]
    assert console[-1].startswith("RuntimeError (len=")


def test_an_access_line_drops_its_query_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    # The access logger as a process that serves with access lines on has it:
    # a ``uvicorn.Config(access_log=False)`` another test built (the
    # console's) leaves it without propagation and a level of its own.
    access = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(access, "propagate", True)
    monkeypatch.setattr(access, "handlers", [])
    monkeypatch.setattr(access, "disabled", False)
    level = access.level
    access.setLevel(logging.NOTSET)  # setLevel: the loggers' level cache too
    request.addfinalizer(lambda: access.setLevel(level))
    stream, _ = _setup(tmp_path)
    try:
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:5",
            "GET",
            "/mcp?api_key=Zebrapassword",
            "1.1",
            200,
        )
        _flush()
    finally:
        setup_logging(Config())
    (line,) = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert line["event"] == '127.0.0.1:5 - "GET /mcp HTTP/1.1" 200'
    _assert_no_marker(stream.getvalue())


def test_a_library_warning_is_logged_by_category_without_its_text(
    tmp_path: Path,
) -> None:
    stream, _ = _setup(tmp_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            from unified_kg_rag.shared import logging as package_logging

            warnings.showwarning = package_logging._log_warning
            warnings.warn(MARKER, UserWarning, stacklevel=1)
        # logging.captureWarnings: the whole text is the message.
        captured = "x.py:1: UserWarning: " + MARKER
        logging.getLogger("py.warnings").warning(captured)
        _flush()
    finally:
        setup_logging(Config())
    text = stream.getvalue()
    _assert_no_marker(text)
    assert "UserWarning at unit/test_error_text_sources.py:" in text
    assert "Python warning (len=" in text


def test_setup_logging_routes_warnings_to_the_handlers(tmp_path: Path) -> None:
    original = warnings.showwarning
    try:
        _setup(tmp_path)
        from unified_kg_rag.shared import logging as package_logging

        assert warnings.showwarning is package_logging._log_warning
    finally:
        warnings.showwarning = original
        setup_logging(Config())


def test_the_filter_keeps_numbers_and_booleans_readable() -> None:
    record = logging.LogRecord(
        "botocore.retryhandler",
        logging.INFO,
        "x",
        1,
        "delay %s ok %s %s",
        (0.4, True, MARKER),
        None,
    )
    assert LibraryRecordFilter().filter(record)
    message = record.getMessage()
    assert message.startswith("delay 0.4 ok True len=")
    _assert_no_marker(message)


def test_error_log_text_is_what_log_error_says() -> None:
    assert error_log_text(RuntimeError(MARKER)).startswith("RuntimeError (len=")
    own = with_own_text(AWSServiceError("Failed to upload 2 file(s)"))
    assert error_log_text(own) == "[failure] Failed to upload 2 file(s)"


def test_a_failed_model_call_is_named_with_the_client_s_status() -> None:
    # rv30 low: run-eval's line read "LanguageModelError (len=246 ...)",
    # though the cause was HTTP 500.
    from unified_kg_rag.adapters.llm.reachability import named_model_error

    class ResponseError(Exception):
        status_code = 500

    named = named_model_error(ResponseError(MARKER), Config())
    assert error_name(named) == "LanguageModelError from ResponseError, HTTP 500"
    _assert_no_marker(operator_error_text(named), error_log_text(named))


def test_a_file_that_fails_to_parse_is_logged_without_the_loader_s_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # rv30 logging #1 (source parsing): the worker returned str(e), logged
    # at ERROR ("Failed to parse '<path>': <the loader's message>").
    from unified_kg_rag.application.ingestion import source_parsing

    class _Loader:
        def parse_file(self, *_args: Any) -> Any:
            raise RuntimeError(f"cannot read near: {MARKER}")

    monkeypatch.setattr(
        source_parsing.ParserFactory, "create_parser", lambda *_a: _Loader()
    )
    path = tmp_path / "memo.txt"
    path.write_text("Vendor and Buyer", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        parsed, failed = source_parsing.SourceFileParser(Config(), tmp_path).parse(
            [path]
        )
    assert failed == [str(path)] and not parsed
    _assert_no_marker(_info_text(caplog))
    assert f"Failed to parse '{path}': RuntimeError (len=" in _info_text(caplog)


async def test_the_mcp_http_server_logs_its_address_readable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # rv30 low: uvicorn's own "running on" line has its address digested
    # (a library's arguments), so the package logs it.
    pytest.importorskip("mcp")
    import uvicorn

    from unified_kg_rag.application.cli import run_mcp_server

    async def serve(_self: Any) -> None:
        return None

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    monkeypatch.setattr(run_mcp_server, "build_mcp_app", lambda *_a, **_k: object())
    with caplog.at_level(logging.INFO):
        await run_mcp_server.serve_http(object(), "::1", 47285)  # type: ignore[arg-type]
    assert "Serving MCP over streamable HTTP on http://[::1]:47285/mcp" in (
        _info_text(caplog)
    )


def test_a_package_module_run_as_main_logs_through_the_filter(tmp_path: Path) -> None:
    # A CLI run as ``python -m`` logs as ``__main__``: its structlog event
    # is the package's record, formatted as such (not a library's string).
    stream, _ = _setup(tmp_path)
    try:
        get_logger("__main__").info("Serving namespace '%s'", "default")
        _flush()
    finally:
        setup_logging(Config())
    (line,) = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert line["event"] == "Serving namespace 'default'"
