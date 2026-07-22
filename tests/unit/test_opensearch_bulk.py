# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSearchClient.bulk_index error handling (AWS-free).

Regression: bulk_index called streaming_bulk WITHOUT raise_on_error=False, so a
single rejected document raised BulkIndexError mid-stream and aborted the whole
batch (re-raised as AWSServiceError) — silently killing every other doc and
making the errors-accumulation path dead. It must mirror bulk_delete: collect
per-doc failures into `errors` and keep going.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import unified_kg_rag.adapters.aws.opensearch as opensearch_mod
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient

pytestmark = pytest.mark.unit


def _client() -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = MagicMock()  # backing for the `client` property
    return client


def test_bulk_index_empty_short_circuits() -> None:
    out = _client().bulk_index("idx", [])
    assert out == {"errors": False, "items": []}


def test_bulk_index_passes_raise_on_error_false(mocker) -> None:
    # The crux: streaming_bulk must be invoked with raise_on_error=False so one
    # bad doc does not abort the batch.
    captured: dict = {}

    def fake_streaming_bulk(client, actions, **kwargs):
        captured.update(kwargs)
        for _ in actions:  # drain generator
            yield True, {}

    mocker.patch.object(
        opensearch_mod, "streaming_bulk", side_effect=fake_streaming_bulk
    )
    _client().bulk_index("idx", [{"id": "a"}, {"id": "b"}])
    assert captured.get("raise_on_error") is False


def test_bulk_index_collects_partial_failures_without_crashing(mocker) -> None:
    # One doc ok, one rejected: the call must NOT raise; it reports errors.
    def fake_streaming_bulk(client, actions, **kwargs):
        items = list(actions)
        yield True, {"index": {"status": 201}}
        yield False, {"index": {"status": 400, "error": "mapping"}}
        # (consume the rest if any)
        for _ in items[2:]:
            yield True, {}

    mocker.patch.object(
        opensearch_mod, "streaming_bulk", side_effect=fake_streaming_bulk
    )
    out = _client().bulk_index("idx", [{"id": "a"}, {"id": "b"}])
    assert out["errors"] is True
    assert len(out["items"]) == 1  # the single rejected doc
