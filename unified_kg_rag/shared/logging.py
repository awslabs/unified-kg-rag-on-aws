# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import structlog
from langsmith.utils import tracing_is_enabled
from structlog.stdlib import LoggerFactory, ProcessorFormatter

from .config import Config

# Shared by structlog-native and foreign (stdlib: botocore, langchain_aws, ...)
# records so both carry the same keys, including context bound with
# structlog.contextvars (pipeline_id / stage / query_id / conversation_id).
_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="ISO"),
]


def _configure_structlog() -> None:
    """Route structlog through stdlib logging; installs no handler.

    Records then reach whatever handlers the process has: the CLI's (see
    ``setup_logging``) or a host application's own. Levels are the stdlib
    logger levels.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class LoggingSetup:
    _handlers: list[logging.Handler] = []

    @classmethod
    def setup_logging(
        cls,
        config: Config,
        config_override: dict[str, Any] | None = None,
        *,
        stream: TextIO | None = None,
    ) -> None:
        """Configure structlog + the root handlers from ``config.logging``.

        Called by the CLIs only (after loading ``--config-path``); importing the
        package never touches the root logger. A repeated call replaces only
        the handlers a previous call added. ``stream`` defaults to stdout.
        """
        log_config = config.logging

        if config_override:
            for key, value in config_override.items():
                setattr(log_config, key, value)

        log_level = getattr(logging, log_config.level.upper(), logging.INFO)
        structured = log_config.log_format == "structured"
        stream = stream or sys.stdout

        _configure_structlog()

        root_logger = logging.getLogger()
        # Re-init replaces only our own handlers (keeps e.g. pytest's caplog).
        for handler in cls._handlers:
            root_logger.removeHandler(handler)
            handler.close()
        cls._handlers = []
        root_logger.setLevel(log_level)

        # Handlers carry no level of their own: logger levels decide, so a
        # CLI --verbose that lowers the 'unified_kg_rag' logger takes effect.
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(
            cls._formatter(json=structured and not stream.isatty(), colors=True)
        )
        cls._handlers.append(console_handler)

        if log_config.log_to_file and log_config.log_file_path:
            log_file_path = cls._get_log_file_path_with_date(log_config.log_file_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(cls._formatter(json=structured, colors=False))
            cls._handlers.append(file_handler)

        for handler in cls._handlers:
            root_logger.addHandler(handler)

        # Chatty dependencies (e.g. langchain_aws logs INFO per model call and a
        # full traceback per retried attempt) default to WARNING.
        for name, level in log_config.library_levels.items():
            logging.getLogger(name).setLevel(level.upper())

    @staticmethod
    def _formatter(json: bool, colors: bool) -> ProcessorFormatter:
        renderer: list[structlog.types.Processor] = (
            [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(default=str),
            ]
            if json
            else [structlog.dev.ConsoleRenderer(colors=colors)]
        )
        return ProcessorFormatter(
            processors=[ProcessorFormatter.remove_processors_meta, *renderer],
            foreign_pre_chain=_SHARED_PROCESSORS,
        )

    @staticmethod
    def _get_log_file_path_with_date(log_file_path: str) -> Path:
        """Dated variant of ``log_file_path``; a relative path is under the CWD."""
        log_path = Path(log_file_path).expanduser().resolve()
        current_date = datetime.now().strftime("%Y%m%d")
        return log_path.with_name(f"{log_path.stem}_{current_date}{log_path.suffix}")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A structlog logger that emits through stdlib logging.

    Configures structlog on first use unless the host application already
    did; never adds handlers or changes levels.
    """
    if not structlog.is_configured():
        _configure_structlog()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def setup_logging(
    config: Config,
    config_override: dict[str, Any] | None = None,
    stream: TextIO | None = None,
) -> None:
    """(Re-)initialise logging from ``config`` — call after loading a CLI config."""
    LoggingSetup.setup_logging(config, config_override, stream=stream)
    warn_if_langsmith_tracing()


def warn_if_langsmith_tracing() -> None:
    """Warn that LangSmith tracing, when enabled, uploads prompt content.

    ``LANGSMITH_TRACING=true`` (or ``LANGCHAIN_TRACING_V2=true``) makes
    langchain-core send every traced run to LangSmith, including prompts,
    retrieved context and model outputs. That may be intended, so it is
    reported, not disabled.
    """
    if tracing_is_enabled():
        get_logger(__name__).warning(
            "LangSmith tracing is enabled by the environment "
            "(LANGSMITH_TRACING / LANGCHAIN_TRACING_V2): prompts, retrieved "
            "context and model outputs will be sent to LangSmith. Unset the "
            "variable to keep this content local."
        )
