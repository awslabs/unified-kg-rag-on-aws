# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging setup: config-driven re-init, JSON for foreign records, context."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from langsmith.utils import get_env_var
from structlog.contextvars import bound_contextvars

from tests.fixtures.extras import requires_eval
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_logger, setup_logging
from unified_kg_rag.shared.logging import LoggingSetup
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

pytestmark = pytest.mark.unit

_TOUCHED_LOGGERS = ("botocore", "langchain_aws", "urllib3", "ragas", "unified_kg_rag")


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_own = list(LoggingSetup._handlers)
    saved_levels = {name: logging.getLogger(name).level for name in _TOUCHED_LOGGERS}
    yield io.StringIO()
    for handler in LoggingSetup._handlers:
        handler.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    LoggingSetup._handlers = saved_own
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)


def _config(**logging_overrides: object) -> Config:
    config = Config()
    config.logging.log_to_file = False
    for key, value in logging_overrides.items():
        setattr(config.logging, key, value)
    return config


def _records(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line]


def test_reinit_applies_config_level(stream: io.StringIO) -> None:
    setup_logging(_config(level="WARNING"), stream=stream)
    log = get_logger("unified_kg_rag.test_reinit")
    log.info("hidden")
    log.warning("shown %s", 1)
    assert [r["event"] for r in _records(stream)] == ["shown 1"]


def test_foreign_records_are_json_with_bound_context(stream: io.StringIO) -> None:
    setup_logging(_config(), stream=stream)
    with bound_contextvars(pipeline_id="p-1", stage="graph_extraction"):
        logging.getLogger("thirdparty.lib").warning("failed after %s attempts", 3)
    (record,) = _records(stream)
    assert record["event"] == "failed after 3 attempts"
    assert record["logger"] == "thirdparty.lib"
    assert record["pipeline_id"] == "p-1"
    assert record["stage"] == "graph_extraction"


def test_a_librarys_error_keeps_its_logger_and_level_not_its_text(
    stream: io.StringIO,
) -> None:
    # A library logs an error's text and traceback (a provider's reply
    # quoting the prompt): at INFO+ its arguments and traceback become
    # digests, its logger, level and message template stay.
    setup_logging(_config(level="DEBUG"), stream=stream)
    try:
        raise ValueError("Quokkasecret prompt tail")
    except ValueError as e:
        logging.getLogger("thirdparty.lib").exception("call %s failed: %s", "Zebra", e)
        logging.getLogger("thirdparty.lib").debug("call failed: %s", e)
    error, debug = _records(stream)
    assert error["logger"] == "thirdparty.lib" and error["level"] == "error"
    assert error["event"].startswith(f"call {text_digest('Zebra')} failed: ValueError")
    assert "test_logging_setup.py" in error["event"]  # where it happened
    assert "Quokkasecret" not in json.dumps(error) and "exception" not in error
    # Below INFO a library's record is the package's DEBUG: all of it.
    assert debug["event"] == "call failed: Quokkasecret prompt tail"


@requires_eval
def test_ragas_and_chat_bedrock_failures_log_no_provider_text(
    stream: io.StringIO,
) -> None:
    # ragas logs a failed judge job's message at ERROR, langchain_aws's
    # ChatBedrock a ClientError's traceback (logger.exception): both quote
    # what the provider said, at the default INFO level.
    import boto3
    from botocore.stub import Stubber
    from langchain_aws import ChatBedrock
    from ragas.executor import Executor

    marker = "Quokkasecret arn:aws:iam::000000000000:role/judge"
    setup_logging(_config(level="INFO"), stream=stream)

    async def judge() -> None:
        raise RuntimeError(marker)

    executor = Executor(desc="judge", show_progress=False, raise_exceptions=False)
    executor.submit(judge)
    executor.results()

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    stubber = Stubber(client)
    stubber.add_client_error(
        "invoke_model",
        service_error_code="ValidationException",
        service_message=marker,
        http_status_code=400,
    )
    with stubber:
        chat = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            client=client,
            region_name="us-east-1",
        )
        with pytest.raises(client.exceptions.ValidationException):
            chat.invoke("Who supplies Vendor?")

    records = _records(stream)
    by_logger = {r["logger"]: r for r in records if r["level"] == "error"}
    assert set(by_logger) == {"ragas.executor", "langchain_aws.llms.bedrock"}
    assert "Quokkasecret" not in stream.getvalue(), stream.getvalue()
    assert (
        "ValidationException, HTTP 400"
        in by_logger["langchain_aws.llms.bedrock"]["event"]
    )


def test_library_levels_quiet_chatty_dependencies(stream: io.StringIO) -> None:
    setup_logging(_config(), stream=stream)
    logging.getLogger("langchain_aws.llms").info("Successfully invoked model")
    logging.getLogger("botocore.retryhandler").warning("kept")
    assert [r["event"] for r in _records(stream)] == ["kept"]


def test_info_leaves_the_request_loggers_quiet(stream: io.StringIO) -> None:
    # At INFO opensearch-py logged every request and httpx every model
    # server call: a line per search, embedding and LLM call.
    setup_logging(_config(), stream=stream)
    logging.getLogger("opensearch").info("POST http://localhost:9200/_bulk")
    logging.getLogger("httpx").info("HTTP Request: POST http://localhost/api")
    logging.getLogger("unified_kg_rag.test_info").info("kept")
    assert [r["event"] for r in _records(stream)] == ["kept"]


def test_a_failed_opensearch_request_is_logged_by_the_package_only(
    stream: io.StringIO,
) -> None:
    # opensearch-py logs every failed request at WARNING, the 404 of each
    # alias lookup before its index exists included; the package's client
    # logs the failures that matter.
    setup_logging(_config(), stream=stream)
    logging.getLogger("opensearch").warning(
        "GET http://localhost:9200/graphrag-entities-default/_alias [status:404]"
    )
    logging.getLogger("unified_kg_rag.adapters.aws.opensearch").error("kept")
    assert [r["event"] for r in _records(stream)] == ["kept"]


def test_debug_leaves_numba_and_httpcore_quiet(stream: io.StringIO) -> None:
    # At DEBUG numba logged every JIT compile (run-visualization) and
    # httpcore every connection step.
    setup_logging(_config(level="DEBUG"), stream=stream)
    logging.getLogger("numba.core.ssa").debug("on_assign %s", "x")
    logging.getLogger("httpcore.connection").debug("connect_tcp.started")
    logging.getLogger("unified_kg_rag.test_debug").debug("kept")
    assert [r["event"] for r in _records(stream)] == ["kept"]


def test_verbose_lowers_package_logger_past_handler(stream: io.StringIO) -> None:
    setup_logging(_config(level="INFO"), stream=stream)
    logging.getLogger("unified_kg_rag").setLevel(logging.DEBUG)
    get_logger("unified_kg_rag.test_verbose").debug("debug detail")
    assert [r["event"] for r in _records(stream)] == ["debug detail"]


def test_the_defaults_write_no_file_so_a_read_only_directory_works(
    stream: io.StringIO, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CLI started in a directory it cannot write (a container's read-only
    # root, a shared checkout) failed creating logs/ before any work.
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    monkeypatch.chdir(read_only)
    try:
        setup_logging(Config(), stream=stream)
    finally:
        read_only.chmod(0o755)
    assert Config().logging.log_to_file is False
    assert not any(isinstance(h, logging.FileHandler) for h in LoggingSetup._handlers)
    assert list(read_only.iterdir()) == []


def test_log_to_file_is_honoured(stream: io.StringIO, tmp_path: Path) -> None:
    setup_logging(_config(), stream=stream)
    assert not any(isinstance(h, logging.FileHandler) for h in LoggingSetup._handlers)

    log_file = tmp_path / "run.txt"
    setup_logging(_config(log_to_file=True, log_file_path=str(log_file)), stream=stream)
    get_logger("unified_kg_rag.test_file").info("to file")
    for handler in LoggingSetup._handlers:
        handler.flush()
    (written,) = list(tmp_path.glob("run_*.txt"))
    assert json.loads(written.read_text().splitlines()[-1])["event"] == "to file"


def test_worker_thread_lines_carry_the_bound_context(stream: io.StringIO) -> None:
    setup_logging(_config(), stream=stream)
    log = get_logger("unified_kg_rag.test_worker")
    with bound_contextvars(pipeline_id="p-1"):
        with ContextThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(log.info, "from worker").result()
    (record,) = _records(stream)
    assert record["event"] == "from worker"
    assert record["pipeline_id"] == "p-1"


_TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)


# Autouse: a developer's own LANGSMITH_TRACING must not add a record to the
# exact-record assertions above.
@pytest.fixture(autouse=True)
def tracing_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in _TRACING_VARS:
        monkeypatch.delenv(name, raising=False)
    get_env_var.cache_clear()  # langsmith caches its environment lookups
    yield monkeypatch
    monkeypatch.undo()
    get_env_var.cache_clear()


@pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
def test_langsmith_tracing_env_warns_once(
    stream: io.StringIO, tracing_env: pytest.MonkeyPatch, name: str
) -> None:
    tracing_env.setenv(name, "true")
    setup_logging(_config(), stream=stream)
    (record,) = _records(stream)
    assert record["level"] == "warning"
    assert "sent to LangSmith" in record["event"]


def test_no_langsmith_warning_without_tracing(
    stream: io.StringIO, tracing_env: pytest.MonkeyPatch
) -> None:
    setup_logging(_config(), stream=stream)
    assert _records(stream) == []


def test_importing_the_package_leaves_host_logging_alone(tmp_path: Path) -> None:
    # A library import must not replace the host's handlers, change the root
    # level, or open a log file (next to the package or anywhere else); only
    # the CLIs call setup_logging.
    import subprocess
    import sys

    script = (
        "import logging, sys\n"
        "host = logging.StreamHandler(sys.stdout)\n"
        "logging.getLogger().addHandler(host)\n"
        "logging.getLogger().setLevel(logging.ERROR)\n"
        "import unified_kg_rag.application.cli.run_evaluation\n"
        "from unified_kg_rag.shared import get_logger\n"
        "host.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))\n"
        "get_logger('unified_kg_rag.host_test').error('lib says %s', 'hi')\n"
        "root = logging.getLogger()\n"
        "assert root.handlers == [host], root.handlers\n"
        "assert root.level == logging.ERROR, root.level\n"
    )
    package_logs = Path(__file__).resolve().parents[2] / "logs"
    before = set(package_logs.glob("*")) if package_logs.exists() else set()
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    # The host's formatter renders the plain message, not an event-dict repr.
    assert completed.stdout.splitlines() == [
        "ERROR unified_kg_rag.host_test lib says hi"
    ]
    assert list(tmp_path.iterdir()) == []
    after = set(package_logs.glob("*")) if package_logs.exists() else set()
    assert after == before


def test_library_mode_records_carry_the_message_and_fields(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without setup_logging a host formatter sees the plain message; the
    # structured fields are record attributes, not a dict repr in the message.
    monkeypatch.setattr(LoggingSetup, "_handlers", [])
    name = "unified_kg_rag.test_library_mode"
    with caplog.at_level(logging.INFO, logger=name):
        with bound_contextvars(pipeline_id="p-1"):
            get_logger(name).warning("lib says %s", "hi")
    (record,) = [r for r in caplog.records if r.name == name]
    assert record.getMessage() == "lib says hi"
    assert record.pipeline_id == "p-1"  # type: ignore[attr-defined]


def test_library_mode_renames_fields_named_like_record_attributes(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A host binding ``name`` (or ``process``, ``module``, ``msg``, ...) with
    # structlog.contextvars made makeRecord raise KeyError on every log call.
    monkeypatch.setattr(LoggingSetup, "_handlers", [])
    name = "unified_kg_rag.test_library_mode_reserved"
    with caplog.at_level(logging.INFO, logger=name):
        with bound_contextvars(name="host-job", process="etl", msg="m"):
            get_logger(name).info("lib says %s", "hi")
    (record,) = [r for r in caplog.records if r.name == name]
    assert record.getMessage() == "lib says hi"
    assert record.name_ == "host-job"  # type: ignore[attr-defined]
    assert record.process_ == "etl"  # type: ignore[attr-defined]
    assert record.msg_ == "m"  # type: ignore[attr-defined]


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def test_host_root_handlers_get_plain_messages_after_setup_logging(
    stream: io.StringIO,
) -> None:
    # A host handler on the root logger formats str(record.msg): it must see
    # the message and the traceback, not the structlog event dict's repr,
    # while the package's own handler still renders the structured fields.
    host = _ListHandler()
    logging.getLogger().addHandler(host)
    setup_logging(_config(log_format="structured"), stream=stream)
    log = get_logger("unified_kg_rag.test_host_handler")
    with bound_contextvars(pipeline_id="p-1"):
        log.warning("lib says %s", "hi", documents=3)
        try:
            raise ValueError("boom")
        except ValueError:
            # error(exc_info=True): .exception() gets exc_info from stdlib.
            log.error("failed", exc_info=True)
    plain, failed = host.lines
    assert plain == "WARNING lib says hi"
    assert failed.startswith("ERROR failed\nTraceback")
    assert "ValueError: boom" in failed
    first, second = _records(stream)
    assert first["event"] == "lib says hi"
    assert (first["documents"], first["pipeline_id"]) == (3, "p-1")
    assert "ValueError: boom" in second["exception"]
    # The record's traceback is not printed twice by the package's formatter.
    assert stream.getvalue().count("ValueError: boom") == 1


def test_relative_log_file_path_resolves_against_cwd(
    stream: io.StringIO, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    setup_logging(
        _config(log_to_file=True, log_file_path="logs/run.txt"), stream=stream
    )
    (log_file,) = (tmp_path / "logs").glob("run_*.txt")
    assert log_file.is_file()


def _fail_with_a_secret_local() -> None:
    model_output = "Quokka-secret-payload"
    raise ValueError(f"parse failed: {len(model_output)} chars")


@pytest.mark.parametrize("log_format", ["plain", "structured"])
def test_tracebacks_show_no_frame_locals_or_ansi_codes(
    stream: io.StringIO, tmp_path: Path, log_format: str
) -> None:
    # Frame locals hold prompts, chunk text and model output; a log line must
    # carry the traceback only. A non-TTY stream gets no colour codes.
    log_file = tmp_path / "run.txt"
    setup_logging(
        _config(log_format=log_format, log_to_file=True, log_file_path=str(log_file)),
        stream=stream,
    )
    try:
        _fail_with_a_secret_local()
    except ValueError:
        get_logger("unified_kg_rag.test_locals").exception("extraction failed")
    for handler in LoggingSetup._handlers:
        handler.flush()
    (written,) = list(tmp_path.glob("run_*.txt"))
    for output in (stream.getvalue(), written.read_text()):
        assert "parse failed: 21 chars" in output
        assert "Quokka-secret-payload" not in output
        assert "\x1b[" not in output


# Modules that keep a stdlib module logger, each with the reason in a comment
# at the call: config.py is imported by the logging setup itself.
_STDLIB_LOGGER_MODULES = {"application/config.py"}


def test_package_modules_log_through_get_logger() -> None:
    # A module logger from logging.getLogger bypasses the structlog chain
    # (redaction, JSON rendering, bound context). shared/ builds that chain
    # and domain/ stays free of it, so only they may.
    import ast

    import unified_kg_rag

    root = Path(unified_kg_rag.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.split("/")[0] in ("shared", "domain"):
            continue
        if relative in _STDLIB_LOGGER_MODULES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getLogger"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
                and any(
                    isinstance(arg, ast.Name) and arg.id == "__name__"
                    for arg in node.args
                )
            ):
                found.append(f"{relative}:{node.lineno}")
    assert found == [], f"use shared.get_logger(__name__): {found}"


def test_package_loggers_carry_their_module_name() -> None:
    # A module that logs under another module's name hides its records from
    # a filter on its own name (the split-out pipeline and chain helpers did).
    import ast

    import unified_kg_rag

    root = Path(unified_kg_rag.__file__).parent
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_logger"
                and not (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "__name__"
                )
            ):
                found.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    assert found == [], f"use get_logger(__name__): {found}"
