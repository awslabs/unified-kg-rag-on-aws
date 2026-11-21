# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``--metrics-sink`` flag the CLIs share, and the sink it selects.

``none`` discards; ``cloudwatch`` writes CloudWatch EMF records
(``shared.metrics.CloudWatchEMFSink``) with an ``Environment`` dimension
when ``UKR_METRICS_ENVIRONMENT`` is set (the CDK stack sets its
``env_name``, which its alarms filter on). The query surfaces (``run-rag``,
``run-mcp-server``, ``run-console``) publish ``GraphRAGChain``'s query
metrics per ``Strategy`` too (:data:`QUERY_METRICS_DIMENSION_KEYS`);
``run-ingestion`` its pipeline metrics. The records go to stderr with the
logs, in either output format, so stdout carries only the command's own
output (an answer, a summary, ``--output-format json``'s document, the MCP
server's protocol); the container's log driver ships both streams.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.metrics import (
    QUERY_ABSTAINED_METRIC,
    QUERY_COST_METRIC,
    QUERY_DEGRADED_METRIC,
    QUERY_FAILED_METRIC,
    QUERY_INPUT_TOKENS_METRIC,
    QUERY_LATENCY_METRIC,
    QUERY_LIMITED_METRIC,
    QUERY_METRICS_DIMENSION_KEYS,
    QUERY_OUTPUT_TOKENS_METRIC,
    QUERY_THROTTLED_METRIC,
    CloudWatchEMFSink,
    MetricsSink,
    NullMetricsSink,
)

logger = get_logger(__name__)

METRICS_SINKS = ("none", "cloudwatch")

# The query metrics each query surface's --metrics-sink help names: one
# list, so a new metric reaches every surface's help.
QUERY_METRICS_HELP = (
    f"{QUERY_LATENCY_METRIC}, the outcome counts {QUERY_FAILED_METRIC}, "
    f"{QUERY_DEGRADED_METRIC}, {QUERY_ABSTAINED_METRIC} and "
    f"{QUERY_LIMITED_METRIC}, {QUERY_THROTTLED_METRIC}, "
    f"{QUERY_INPUT_TOKENS_METRIC} and {QUERY_OUTPUT_TOKENS_METRIC}, and "
    f"{QUERY_COST_METRIC} when models.prices prices every language model "
    "and reranker called (an embedding model without a price leaves its "
    "requests out)"
)


def add_metrics_sink_argument(parser: argparse.ArgumentParser, help: str) -> None:
    """Add ``--metrics-sink {none,cloudwatch}`` (default ``none``)."""
    parser.add_argument(
        "--metrics-sink", choices=METRICS_SINKS, default="none", help=help
    )


def _emf_logger() -> logging.Logger:
    """A non-propagating logger writing bare EMF lines to stderr."""
    emf_logger = logging.getLogger("unified_kg_rag.emf.stderr")
    if not emf_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        emf_logger.addHandler(handler)
        emf_logger.setLevel(logging.INFO)
        emf_logger.propagate = False
    return emf_logger


def build_metrics_sink(choice: str, *, query: bool) -> MetricsSink:
    """The sink ``--metrics-sink choice`` selects: for the query metrics
    (``query``: with the ``Strategy`` dimension) or the pipeline's, its EMF
    written to stderr (stdout carries the command's output)."""
    if choice != "cloudwatch":
        return NullMetricsSink()
    # Set by the CDK stack to its env_name, which its alarms filter on.
    environment = os.getenv("UKR_METRICS_ENVIRONMENT") or None
    logger.info(
        "Forwarding %s metrics to CloudWatch (EMF on stderr), environment '%s'",
        "query" if query else "pipeline",
        environment or "-",
    )
    return CloudWatchEMFSink(
        _emf_logger(),
        environment=environment,
        dimension_keys=QUERY_METRICS_DIMENSION_KEYS if query else (),
    )
