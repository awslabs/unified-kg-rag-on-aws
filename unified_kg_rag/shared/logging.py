# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import structlog
from structlog.stdlib import LoggerFactory, ProcessorFormatter

from .config import Config, get_config

# Shared by structlog-native and foreign (stdlib: botocore, langchain_aws, ...)
# records so both carry the same keys, including context bound with
# structlog.contextvars (pipeline_id / stage / query_id / conversation_id).
_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="ISO"),
]


class LoggingSetup:
    _initialized = False
    _handlers: list[logging.Handler] = []

    @classmethod
    def setup_logging(
        cls,
        config: Config,
        config_override: dict[str, Any] | None = None,
        *,
        force: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        """Configure structlog + the root handlers from ``config.logging``.

        The first call (from ``get_logger``) uses the default config; CLIs call
        again with ``force=True`` after loading ``--config-path`` so the file's
        ``logging`` section takes effect. ``stream`` defaults to stdout.
        """
        if cls._initialized and not force:
            return

        log_config = config.logging

        if config_override:
            for key, value in config_override.items():
                setattr(log_config, key, value)

        log_level = getattr(logging, log_config.level.upper(), logging.INFO)
        structured = log_config.log_format == "structured"
        stream = stream or sys.stdout

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

        root_logger = logging.getLogger()
        if not cls._initialized:
            root_logger.handlers.clear()
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

        cls._initialized = True

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

    @classmethod
    def _get_log_file_path_with_date(cls, log_file_path: str) -> Path:
        root_dir = Path(__file__).parent.parent.parent
        log_path = root_dir / Path(log_file_path)
        current_date = datetime.now().strftime("%Y%m%d")
        stem = log_path.stem
        suffix = log_path.suffix
        new_name = f"{stem}_{current_date}{suffix}"
        return log_path.parent / new_name


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    config = get_config()
    LoggingSetup.setup_logging(config)
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def setup_logging(
    config: Config,
    config_override: dict[str, Any] | None = None,
    stream: TextIO | None = None,
) -> None:
    """(Re-)initialise logging from ``config`` — call after loading a CLI config."""
    LoggingSetup.setup_logging(config, config_override, force=True, stream=stream)
