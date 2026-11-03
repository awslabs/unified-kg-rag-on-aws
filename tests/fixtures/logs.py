# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading what a captured log record says, however structlog delivered it."""

from __future__ import annotations

import logging


def log_message(record: logging.LogRecord) -> str:
    """The formatted message of ``record``.

    A ``get_logger`` record carries a plain message until ``setup_logging``
    installs its handlers; after that (an earlier test in the same worker
    may have run a CLI) its ``msg`` is the structlog event dict, whose
    ``event`` holds the message with its arguments already applied.
    """
    if isinstance(record.msg, dict) and "event" in record.msg:
        return str(record.msg["event"])
    return record.getMessage()
