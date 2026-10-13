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
from opensearchpy.exceptions import RequestError

from unified_kg_rag.adapters.aws import opensearch as opensearch_mod
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.shared import AWSServiceError
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

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

    def delete(index: str, **_: object) -> None:
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


def test_create_index_keeps_an_existing_index_unless_told_not_to() -> None:
    client, backing = _client()
    backing.indices.exists.return_value = True

    client.create_index("graphrag-entities-default-20260101120000000", {})
    backing.indices.create.assert_not_called()

    with pytest.raises(AWSServiceError, match="already exists"):
        client.create_index(
            "graphrag-entities-default-20260101120000000", {}, exist_ok=False
        )
    backing.indices.create.assert_not_called()


_NEW_INDEX = "graphrag-claims-default-20260101120000000"


def _rejected(error_type: str) -> RequestError:
    """A 400 as the sync connection raises it (type name + parsed body)."""
    body = {"error": {"type": error_type, "reason": "x"}, "status": 400}
    return RequestError(400, error_type, body)


def test_a_resent_create_that_finds_its_own_index_counts_as_created() -> None:
    # The transport resends a create whose response was lost; the index did
    # not exist before the call, so "already exists" is this create's result.
    client, backing = _client()
    backing.indices.exists.return_value = False
    backing.indices.create.side_effect = _rejected("resource_already_exists_exception")

    client.create_index(_NEW_INDEX, {}, exist_ok=False)

    backing.indices.create.assert_called_once()


def test_create_index_still_raises_any_other_rejection() -> None:
    client, backing = _client()
    backing.indices.exists.return_value = False
    backing.indices.create.side_effect = _rejected("mapper_parsing_exception")

    with pytest.raises(StoreRequestRejectedError, match="mapper_parsing"):
        client.create_index(_NEW_INDEX, {})


def test_a_delete_ignores_names_that_are_already_gone() -> None:
    # A delete resent after its response was lost finds the index deleted.
    client, backing = _client()
    client.delete_indices([_NEW_INDEX])
    assert backing.indices.delete.call_args.kwargs["ignore_unavailable"] is True
