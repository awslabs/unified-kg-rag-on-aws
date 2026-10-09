# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSearchClient.delete_indices batching (AWS-free).

The index names are joined into the request path, and OpenSearch rejects a
request line over 4 KB. With ~80 leaked timestamped indices to reap, a single
call naming them all failed on every run, so cleanup never caught up. The
names are now deleted in bounded batches, a failed batch does not stop the
rest, and the names left behind are reported.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.aws import opensearch as opensearch_mod
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.shared import AWSServiceError

pytestmark = pytest.mark.unit


def _client() -> tuple[OpenSearchClient, MagicMock]:
    client = OpenSearchClient.__new__(OpenSearchClient)
    backing = MagicMock()
    client._client = backing  # backing for the `client` property
    return client, backing


def _leaked(count: int) -> list[str]:
    return [
        f"graphrag-entities-default-2026010112{i // 60:02d}{i % 60:02d}"
        for i in range(count)
    ]


def _deleted_batches(backing: MagicMock) -> list[list[str]]:
    return [
        call.kwargs["index"].split(",")
        for call in backing.indices.delete.call_args_list
    ]


def test_many_indices_are_deleted_in_bounded_requests() -> None:
    client, backing = _client()
    names = _leaked(85)
    assert len(",".join(names)) > 3000  # one request would exceed the limit

    client.delete_indices(names)

    batches = _deleted_batches(backing)
    assert len(batches) > 1
    assert [name for batch in batches for name in batch] == names
    for batch in batches:
        assert len(batch) <= opensearch_mod._DELETE_BATCH_MAX_NAMES
        assert len(",".join(batch)) <= opensearch_mod._DELETE_BATCH_MAX_CHARS


def test_batches_are_bounded_by_total_name_length() -> None:
    names = ["n" * 1000, "m" * 1000, "o" * 1000, "p" * 10]
    batches = opensearch_mod._index_name_batches(names)
    assert batches == [names[:2], names[2:]]
    assert all(len(",".join(b)) <= 3000 for b in batches)


def test_an_oversized_name_gets_a_batch_of_its_own() -> None:
    names = ["a", "x" * 4000, "b"]
    assert opensearch_mod._index_name_batches(names) == [["a"], ["x" * 4000], ["b"]]


def test_empty_input_sends_no_request() -> None:
    client, backing = _client()
    client.delete_indices([])
    backing.indices.delete.assert_not_called()


def test_a_failed_batch_does_not_stop_the_rest_and_is_reported() -> None:
    client, backing = _client()
    names = _leaked(45)  # three batches of at most 20
    failing = ",".join(names[20:40])

    def delete(index: str) -> None:
        if index == failing:
            raise RuntimeError("cluster unavailable")

    backing.indices.delete.side_effect = delete

    with pytest.raises(AWSServiceError) as excinfo:
        client.delete_indices(names)

    assert len(_deleted_batches(backing)) == 3
    message = str(excinfo.value)
    assert "Failed to delete 20 of 45" in message
    assert names[20] in message and "(+10 more)" in message
    assert names[0] not in message and names[40] not in message
    assert "cluster unavailable" in message
