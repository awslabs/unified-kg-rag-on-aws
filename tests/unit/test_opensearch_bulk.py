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
from opensearchpy.exceptions import TransportError

import unified_kg_rag.adapters.aws.opensearch as opensearch_mod
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.shared import StoreConnectionError
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

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


def _status_by_id(statuses: dict[str, list[int]], calls: list[list[str]]):
    """A streaming_bulk stand-in answering each id with its next status."""

    def fake_streaming_bulk(client, actions, **kwargs):
        sent = [action["_id"] for action in actions]
        calls.append(sent)
        for doc_id in sent:
            status = statuses[doc_id].pop(0) if statuses[doc_id] else 201
            item = {"_id": doc_id, "status": status}
            if status >= 300:
                item["error"] = {"type": "es_rejected_execution_exception"}
            yield 200 <= status < 300, {"index": item}

    return fake_streaming_bulk


def test_bulk_index_resends_items_rejected_with_a_retryable_status(mocker) -> None:
    # A 200 bulk response can carry per-item 429/503 rejections that the
    # client's request-level retries never see; those items are resent.
    calls: list[list[str]] = []
    statuses = {"a": [], "b": [429, 503], "c": []}
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_status_by_id(statuses, calls),
    )
    sleep = mocker.patch.object(opensearch_mod, "_sleep")

    out = _client().bulk_index("idx", [{"id": "a"}, {"id": "b"}, {"id": "c"}])

    assert out == {"errors": False, "items": []}
    assert calls == [["a", "b", "c"], ["b"], ["b"]]
    assert sleep.call_count == 2


def test_bulk_index_does_not_resend_permanent_item_errors(mocker) -> None:
    calls: list[list[str]] = []
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_status_by_id({"a": [400], "b": []}, calls),
    )
    mocker.patch.object(opensearch_mod, "_sleep")

    out = _client().bulk_index("idx", [{"id": "a"}, {"id": "b"}])

    assert calls == [["a", "b"]]
    assert [item["index"]["_id"] for item in out["items"]] == ["a"]


def test_bulk_index_reports_items_still_rejected_after_the_retries(mocker) -> None:
    calls: list[list[str]] = []
    retries = OpenSearchClient.BULK_ITEM_MAX_RETRIES
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_status_by_id({"a": [429] * (retries + 1), "b": []}, calls),
    )
    mocker.patch.object(opensearch_mod, "_sleep")

    out = _client().bulk_index("idx", [{"id": "a"}, {"id": "b"}])

    assert len(calls) == retries + 1
    assert out["errors"] is True
    assert [item["index"]["_id"] for item in out["items"]] == ["a"]
    assert out["items"][0]["index"]["status"] == 429


def test_bulk_update_sends_partial_docs_and_skips_missing_ones(mocker) -> None:
    sent: list[dict] = []

    def fake_streaming_bulk(client, actions, **kwargs):
        for action in actions:
            sent.append(action)
            if action["_id"] == "gone":
                yield False, {"update": {"_id": "gone", "status": 404}}
            elif action["_id"] == "bad":
                yield False, {"update": {"_id": "bad", "status": 400}}
            else:
                yield True, {"update": {"_id": action["_id"], "status": 200}}

    mocker.patch.object(
        opensearch_mod, "streaming_bulk", side_effect=fake_streaming_bulk
    )
    out = _client().bulk_update(
        "idx",
        [
            {"id": "a", "text_unit_ids": ["t1"]},
            {"id": "gone", "text_unit_ids": []},
            {"id": "bad", "text_unit_ids": []},
        ],
    )

    assert sent[0] == {
        "_op_type": "update",
        "_index": "idx",
        "_id": "a",
        "doc": {"text_unit_ids": ["t1"]},
    }
    assert [item["update"]["_id"] for item in out["items"]] == ["bad"]


def _rejecting_requests(statuses: list[int], calls: list[list[str]]):
    """A streaming_bulk stand-in whose next requests fail whole with ``statuses``.

    A failing call answers its first two items, then the request carrying the
    next one fails (the only item fails when it is alone), as a 429 burst
    hitting a write slice midway does.
    """

    def fake_streaming_bulk(client, actions, **kwargs):
        sent = [action["_id"] for action in actions]
        calls.append(sent)
        status = statuses.pop(0) if statuses else None
        for i, doc_id in enumerate(sent):
            if status is not None and i == min(2, len(sent) - 1):
                body = {"error": {"type": "rejected", "reason": f"injected {status}"}}
                raise TransportError(status, "rejected", body)
            yield True, {"index": {"_id": doc_id, "status": 201}}

    return fake_streaming_bulk


def test_a_whole_bulk_request_rejected_under_load_is_resent_with_backoff(
    mocker,
) -> None:
    # Regression: opensearch-py retries a whole-request 429/5xx without any
    # pause, so a burst longer than its few retries failed the write slice.
    # The items the request carried (and the unsent ones after it) are now
    # resent after the bulk-item backoff, like rejected items.
    calls: list[list[str]] = []
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_rejecting_requests([429, 503], calls),
    )
    sleep = mocker.patch.object(opensearch_mod, "_sleep")
    docs = [{"id": doc_id} for doc_id in "abcde"]

    out = _client().bulk_index("idx", docs)

    assert out == {"errors": False, "items": []}
    assert calls == [list("abcde"), list("cde"), ["e"]]
    assert sleep.call_count == 2
    assert all(call.args[0] > 0 for call in sleep.call_args_list)


@pytest.mark.parametrize("method", ["bulk_index", "bulk_delete", "bulk_update"])
def test_a_bulk_request_still_rejected_raises_a_typed_error(mocker, method) -> None:
    # Regression: the bulk methods raised a plain AWSServiceError("Streaming
    # bulk ... failed.") that dropped the status and the server's reason.
    retries = OpenSearchClient.BULK_ITEM_MAX_RETRIES
    calls: list[list[str]] = []
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_rejecting_requests([429] * (retries + 1), calls),
    )
    mocker.patch.object(opensearch_mod, "_sleep")
    args = (
        ["a", "b", "c"]
        if method == "bulk_delete"
        else [{"id": doc_id, "x": 1} for doc_id in "abc"]
    )

    with pytest.raises(StoreConnectionError, match="status=429.*injected 429"):
        getattr(_client(), method)("idx", args)
    assert len(calls) == retries + 1


def test_a_bulk_request_rejected_permanently_is_not_resent(mocker) -> None:
    calls: list[list[str]] = []
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_rejecting_requests([400], calls),
    )
    sleep = mocker.patch.object(opensearch_mod, "_sleep")

    with pytest.raises(StoreRequestRejectedError, match="injected 400"):
        _client().bulk_index("idx", [{"id": doc_id} for doc_id in "abc"])
    assert len(calls) == 1
    sleep.assert_not_called()
