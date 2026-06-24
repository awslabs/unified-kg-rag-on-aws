# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from aws_graphrag.domain.models import LoggingConfig

from .config import Config, get_config


class LoggingSetup:
    _initialized = False

    @classmethod
    def setup_logging(
        cls, config: Config, config_override: dict[str, Any] | None = None
    ) -> None:
        if cls._initialized:
            return

        log_config = config.logging

        if config_override:
            for key, value in config_override.items():
                setattr(log_config, key, value)

        log_level = getattr(logging, log_config.level.upper(), logging.INFO)

        if log_config.log_format == "structured":
            cls._setup_structured_logging(log_level, log_config)
        else:
            cls._setup_simple_logging(log_level, log_config)

        cls._initialized = True

    @classmethod
    def _get_log_file_path_with_date(cls, log_file_path: str) -> Path:
        root_dir = Path(__file__).parent.parent.parent
        log_path = root_dir / Path(log_file_path)
        current_date = datetime.now().strftime("%Y%m%d")
        stem = log_path.stem
        suffix = log_path.suffix
        new_name = f"{stem}_{current_date}{suffix}"
        return log_path.parent / new_name

    @classmethod
    def _setup_structured_logging(
        cls, log_level: int, log_config: LoggingConfig
    ) -> None:
        timestamper = structlog.processors.TimeStamper(fmt="ISO")

        processors: list[structlog.types.Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if sys.stderr.isatty():
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

        if log_config.log_to_file and log_config.log_file_path:
            log_file_path = cls._get_log_file_path_with_date(log_config.log_file_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)

    @classmethod
    def _setup_simple_logging(cls, log_level: int, log_config: LoggingConfig) -> None:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_config.log_to_file and log_config.log_file_path:
            log_file_path = cls._get_log_file_path_with_date(log_config.log_file_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    config = get_config()
    LoggingSetup.setup_logging(config)
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def setup_logging(
    config: Config, config_override: dict[str, Any] | None = None
) -> None:
    LoggingSetup.setup_logging(config, config_override)
