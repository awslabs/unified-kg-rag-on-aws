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
