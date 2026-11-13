# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``--metrics-sink`` flag the CLIs share, and the sink it selects.

``none`` discards; ``cloudwatch`` writes CloudWatch EMF records
(``shared.metrics.CloudWatchEMFSink``) with an ``Environment`` dimension
when ``UKR_METRICS_ENVIRONMENT`` is set (the CDK stack sets its
``env_name``, which its alarms filter on). The query surfaces (``run-rag``,
``run-mcp-server``, ``run-console``) publish ``GraphRAGChain``'s query
metrics per ``Strategy`` too (:data:`QUERY_METRICS_DIMENSION_KEYS`);
``run-ingestion`` its pipeline metrics. The records go to stdout, the
stream the container's log driver ships, or to stderr where stdout carries
the command's own output (``--output-format json``, the MCP server's
stdio), as the logs do there; the log driver ships both streams.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TextIO

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.metrics import (
    QUERY_METRICS_DIMENSION_KEYS,
    CloudWatchEMFSink,
    MetricsSink,
    NullMetricsSink,
)

logger = get_logger(__name__)

METRICS_SINKS = ("none", "cloudwatch")


def add_metrics_sink_argument(parser: argparse.ArgumentParser, help: str) -> None:
    """Add ``--metrics-sink {none,cloudwatch}`` (default ``none``)."""
    parser.add_argument(
        "--metrics-sink", choices=METRICS_SINKS, default="none", help=help
    )


def _emf_logger(stream: TextIO) -> logging.Logger:
    """A non-propagating logger writing bare EMF lines to ``stream``, one
    per stream."""
    name = "unified_kg_rag.emf" + ("" if stream is sys.stdout else ".stderr")
    emf_logger = logging.getLogger(name)
    if not emf_logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        emf_logger.addHandler(handler)
        emf_logger.setLevel(logging.INFO)
        emf_logger.propagate = False
    return emf_logger


def build_metrics_sink(
    choice: str, *, query: bool, stdout_taken: bool = False
) -> MetricsSink:
    """The sink ``--metrics-sink choice`` selects: for the query metrics
    (``query``: with the ``Strategy`` dimension) or the pipeline's, its EMF
    written to stdout, or to stderr when ``stdout_taken`` (stdout carries
    the command's JSON or a protocol)."""
    if choice != "cloudwatch":
        return NullMetricsSink()
    # Set by the CDK stack to its env_name, which its alarms filter on.
    environment = os.getenv("UKR_METRICS_ENVIRONMENT") or None
    stream: TextIO = sys.stderr if stdout_taken else sys.stdout
    logger.info(
        "Forwarding %s metrics to CloudWatch (EMF on %s), environment '%s'",
        "query" if query else "pipeline",
        "stdout" if stream is sys.stdout else "stderr",
        environment or "-",
    )
    return CloudWatchEMFSink(
        _emf_logger(stream),
        environment=environment,
        dimension_keys=QUERY_METRICS_DIMENSION_KEYS if query else (),
    )
