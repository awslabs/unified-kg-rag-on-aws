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

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_logger, setup_logging
from unified_kg_rag.shared.logging import LoggingSetup
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

pytestmark = pytest.mark.unit

_TOUCHED_LOGGERS = ("botocore", "langchain_aws", "urllib3", "unified_kg_rag")


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
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("thirdparty.lib").exception("failed %s", "call")
    (record,) = _records(stream)
    assert record["event"] == "failed call"
    assert record["logger"] == "thirdparty.lib"
    assert record["pipeline_id"] == "p-1"
    assert record["stage"] == "graph_extraction"
    # The traceback is a JSON field, not raw lines interleaved with the JSON.
    assert "ValueError: boom" in record["exception"]


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
