# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pluggable metrics sink (operational excellence)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from unified_kg_rag.shared import CloudWatchEMFSink, MetricsSink, NullMetricsSink

pytestmark = pytest.mark.unit


def _emf_logger() -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    lg = logging.getLogger("test.emf")
    lg.handlers = [logging.StreamHandler(buf)]
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg, buf


def test_sinks_conform_to_protocol() -> None:
    assert isinstance(NullMetricsSink(), MetricsSink)
    lg, _ = _emf_logger()
    assert isinstance(CloudWatchEMFSink(emf_logger=lg), MetricsSink)


def test_null_sink_is_noop() -> None:
    assert NullMetricsSink().emit("ns", {"a": 1.0}, {"d": "x"}) is None


def test_emf_emits_numeric_metrics_and_dimensions() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg).emit(
        "unified_kg_rag/ingestion",
        {"entities": 42, "rate": 0.5, "name": "ignored"},
        {"pipeline_id": "p1"},
    )
    out = json.loads(buf.getvalue())
    cw = out["_aws"]["CloudWatchMetrics"][0]
    assert cw["Namespace"] == "unified_kg_rag/ingestion"
    assert {m["Name"] for m in cw["Metrics"]} == {"entities", "rate"}
    # Only the zero-dimension aggregate (what the alarms and dashboard query)
    # is a metric series: a per-run pipeline_id dimension would create new
    # custom-metric series every run. pipeline_id stays a log property.
    assert cw["Dimensions"] == [[]]
    assert out["entities"] == 42
    assert out["pipeline_id"] == "p1"
    assert "name" not in out  # non-numeric dropped


def test_emf_declares_only_the_configured_dimension_keys() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg, dimension_keys=("environment",)).emit(
        "ns", {"x": 1}, {"pipeline_id": "p1", "environment": "dev"}
    )
    out = json.loads(buf.getvalue())
    assert out["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [[], ["environment"]]
    assert out["pipeline_id"] == "p1" and out["environment"] == "dev"


def test_emf_default_logger_writes_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    emf = logging.getLogger("unified_kg_rag.emf")
    monkeypatch.setattr(emf, "handlers", [])
    CloudWatchEMFSink()
    (handler,) = emf.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout


def test_emf_no_numeric_metrics_emits_nothing() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg).emit("ns", {"only": "strings"})
    assert buf.getvalue() == ""


def test_emf_record_has_epoch_millis_timestamp() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg).emit("ns", {"x": 1})
    ts = json.loads(buf.getvalue())["_aws"]["Timestamp"]
    assert isinstance(ts, int)
    # plausible epoch-millis (>= 2021-01-01), guards against epoch-seconds regress
    assert ts > 1_600_000_000_000


def test_emf_bool_values_not_emitted_as_numeric() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg).emit("ns", {"flag": True, "count": 3})
    out = json.loads(buf.getvalue())
    names = {m["Name"] for m in out["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert names == {"count"}
    assert "flag" not in out


def test_emf_no_dimensions_emits_empty_dimension_set() -> None:
    lg, buf = _emf_logger()
    CloudWatchEMFSink(emf_logger=lg).emit("ns", {"x": 1})
    out = json.loads(buf.getvalue())
    assert out["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [[]]


def test_emf_default_logger_construction() -> None:
    # The no-arg construction branch builds its own dedicated EMF logger.
    sink = CloudWatchEMFSink()
    assert isinstance(sink, MetricsSink)
