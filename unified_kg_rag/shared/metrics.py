# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable metrics sink — the operational-excellence boundary.

The framework *collects* metrics in-memory (``MetricsMixin``, ``PipelineMetrics``)
but should not assume a specific monitoring backend. ``MetricsSink`` is the port
through which an embedding application forwards those metrics to its sink of
choice. Two adapters ship:

- ``NullMetricsSink`` (default): discards — zero overhead, no AWS dependency.
- ``CloudWatchEMFSink``: writes CloudWatch Embedded Metric Format (EMF) JSON to
  stdout (or a given logger), so any CloudWatch Logs pipeline auto-extracts
  metrics without a ``PutMetricData`` API call (ideal for Lambda/ECS/EKS).

Callers select a sink and pass it to the pipeline/manager; the library never
hard-codes CloudWatch.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Protocol, runtime_checkable

from unified_kg_rag.shared.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class MetricsSink(Protocol):
    """Receives a named group of metrics and optional dimensions."""

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        """Publish ``metrics`` (name -> value) under ``namespace``."""
        ...


class NullMetricsSink:
    """Default sink: discards everything (no monitoring backend assumed)."""

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        return None


class CloudWatchEMFSink:
    """Emit CloudWatch Embedded Metric Format JSON to a logger.

    EMF lets CloudWatch Logs auto-extract metrics from structured log events
    (no PutMetricData call). Drop this into Lambda/ECS/EKS and the metrics appear
    in CloudWatch under ``namespace``. Defaults to a dedicated, non-propagating
    logger that writes bare EMF lines to stdout (the stream the container's
    awslogs driver ships, like the app logs).

    Every metric is published as the zero-dimension aggregate, which the
    CloudWatch alarms and dashboard query. ``emit`` dimensions are written as
    log properties (searchable in Logs Insights); only keys listed in
    ``dimension_keys`` also become a metric dimension. Each distinct dimension
    value is a separately billed custom-metric series, so never list a per-run
    key such as ``pipeline_id``.
    """

    def __init__(
        self,
        emf_logger: logging.Logger | None = None,
        *,
        dimension_keys: tuple[str, ...] = (),
    ) -> None:
        self._dimension_keys = dimension_keys
        if emf_logger is None:
            emf_logger = logging.getLogger("unified_kg_rag.emf")
            if not emf_logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter("%(message)s"))
                emf_logger.addHandler(handler)
                emf_logger.setLevel(logging.INFO)
                emf_logger.propagate = False
        self._emf_logger = emf_logger

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        # bool is a subclass of int; exclude it so flags are not emitted as
        # 0/1 metrics.
        numeric = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            return
        dims = dimensions or {}
        # The zero-dimension aggregate ([]) is what the dimensionless alarms
        # (e.g. IndexingFailures) and dashboard widgets query. Other dimension
        # values stay log properties unless configured as dimension keys.
        dimension_sets: list[list[str]] = [[]]
        keyed = [k for k in dims if k in self._dimension_keys]
        if keyed:
            dimension_sets.append(keyed)
        emf: dict[str, Any] = {
            "_aws": {
                # Timestamp is a REQUIRED member of the EMF metadata object
                # (epoch milliseconds); records missing it are not extracted as
                # metrics by the CloudWatch Logs EMF parser.
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": namespace,
                        "Dimensions": dimension_sets,
                        "Metrics": [{"Name": k} for k in numeric],
                    }
                ],
            },
            **dims,
            **numeric,
        }
        self._emf_logger.info(json.dumps(emf))
