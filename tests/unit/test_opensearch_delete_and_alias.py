# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSearchClient.bulk_delete and the blue/green alias calls (AWS-free).

``bulk_delete`` backs incremental removals: a doc that is already gone reports
``not_found``, which must count as success or a re-run of a partially applied
removal would never converge. ``update_alias`` performs the blue/green swap and
must remove the old binding and add the new one in ONE ``update_aliases`` call
so readers never see the alias unbound.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from opensearchpy.exceptions import NotFoundError, TransportError

import unified_kg_rag.adapters.aws.opensearch as opensearch_mod
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.shared import AWSServiceError

pytestmark = pytest.mark.unit


def _client() -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = MagicMock()  # backing for the `client` property
    return client


def _fake_bulk(results: list[tuple[bool, dict[str, Any]]], captured: dict):
    def fake_streaming_bulk(client, actions, **kwargs):  # noqa: ANN001, ARG001
        captured["actions"] = list(actions)
        captured["kwargs"] = kwargs
        yield from results

    return fake_streaming_bulk


# --- bulk_delete -------------------------------------------------------------


def test_bulk_delete_empty_short_circuits(mocker) -> None:
    bulk = mocker.patch.object(opensearch_mod, "streaming_bulk")
    os_client = _client()
    assert os_client.bulk_delete("idx", []) == {"errors": False, "items": []}
    bulk.assert_not_called()
    os_client.client.indices.refresh.assert_not_called()


def test_bulk_delete_emits_delete_actions_without_raising_on_error(mocker) -> None:
    captured: dict = {}
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_fake_bulk([(True, {}), (True, {})], captured),
    )
    os_client = _client()
    out = os_client.bulk_delete("idx", ["a", 7])  # type: ignore[list-item]

    assert out == {"errors": False, "items": []}
    assert captured["actions"] == [
        {"_op_type": "delete", "_index": "idx", "_id": "a"},
        {"_op_type": "delete", "_index": "idx", "_id": "7"},
    ]
    assert captured["kwargs"]["raise_on_error"] is False
    os_client.client.indices.refresh.assert_called_once_with(index="idx")


def test_bulk_delete_treats_not_found_as_success(mocker) -> None:
    not_found = (False, {"delete": {"_id": "gone", "result": "not_found"}})
    mocker.patch.object(
        opensearch_mod, "streaming_bulk", side_effect=_fake_bulk([not_found], {})
    )
    os_client = _client()
    out = os_client.bulk_delete("idx", ["gone"])

    assert out == {"errors": False, "items": []}
    # Counted as a success, so the index is still refreshed.
    os_client.client.indices.refresh.assert_called_once_with(index="idx")


def test_bulk_delete_reports_real_failures(mocker) -> None:
    rejected = (False, {"delete": {"_id": "b", "status": 400, "error": "rejected"}})
    mocker.patch.object(
        opensearch_mod,
        "streaming_bulk",
        side_effect=_fake_bulk([(True, {}), rejected], {}),
    )
    out = _client().bulk_delete("idx", ["a", "b"])

    assert out["errors"] is True
    assert out["items"] == [rejected[1]]


def test_bulk_delete_skips_refresh_when_nothing_succeeded(mocker) -> None:
    rejected = (False, {"delete": {"_id": "a", "status": 500}})
    mocker.patch.object(
        opensearch_mod, "streaming_bulk", side_effect=_fake_bulk([rejected], {})
    )
    os_client = _client()
    os_client.bulk_delete("idx", ["a"])
    os_client.client.indices.refresh.assert_not_called()


def test_bulk_delete_wraps_transport_failure(mocker) -> None:
    def boom(client, actions, **kwargs):  # noqa: ANN001, ARG001
        raise ConnectionError("endpoint unreachable")
        yield  # pragma: no cover - makes this a generator

    mocker.patch.object(opensearch_mod, "streaming_bulk", side_effect=boom)
    with pytest.raises(AWSServiceError, match="bulk delete failed"):
        _client().bulk_delete("idx", ["a"])


# --- update_alias / delete_alias (blue/green swap) ---------------------------


def test_update_alias_swaps_atomically_in_one_call() -> None:
    os_client = _client()
    os_client.update_alias("docs", "docs-v2", remove_pattern="docs-*")

    os_client.client.indices.update_aliases.assert_called_once_with(
        body={
            "actions": [
                {
                    "remove": {
                        "index": "docs-*",
                        "alias": "docs",
                        "must_exist": False,
                    }
                },
                {"add": {"index": "docs-v2", "alias": "docs"}},
            ]
        }
    )


def test_update_alias_without_pattern_only_adds() -> None:
    os_client = _client()
    os_client.update_alias("docs", "docs-v1")

    os_client.client.indices.update_aliases.assert_called_once_with(
        body={"actions": [{"add": {"index": "docs-v1", "alias": "docs"}}]}
    )


def test_update_alias_wraps_transport_error() -> None:
    os_client = _client()
    os_client.client.indices.update_aliases.side_effect = TransportError(
        400, "illegal_argument_exception", {}
    )
    with pytest.raises(AWSServiceError, match="update_alias"):
        os_client.update_alias("docs", "docs-v2", remove_pattern="docs-*")


def test_delete_alias_joins_lists() -> None:
    os_client = _client()
    os_client.delete_alias(["docs-v1", "docs-v2"], ["docs", "docs-read"])

    os_client.client.indices.delete_alias.assert_called_once_with(
        index="docs-v1,docs-v2", name="docs,docs-read"
    )


def test_delete_alias_accepts_strings() -> None:
    os_client = _client()
    os_client.delete_alias("docs-v1", "docs")

    os_client.client.indices.delete_alias.assert_called_once_with(
        index="docs-v1", name="docs"
    )


def test_delete_alias_propagates_not_found() -> None:
    # NotFoundError is re-raised unwrapped so callers can treat a missing
    # alias as already removed.
    os_client = _client()
    os_client.client.indices.delete_alias.side_effect = NotFoundError(
        404, "aliases_not_found_exception", {}
    )
    with pytest.raises(NotFoundError):
        os_client.delete_alias("docs-v1", "docs")


# --- get_aliases_by_index (index-name lookup) --------------------------------


def test_get_aliases_by_index_matches_index_names() -> None:
    # Looks up by index name (``GET <pattern>/_alias``), not alias name, and
    # also returns indices that carry no alias.
    os_client = _client()
    os_client.client.indices.get_alias.return_value = {
        "docs-20260102000000": {"aliases": {"docs": {}}},
        "docs-20260101000000": {"aliases": {}},
    }

    result = os_client.get_aliases_by_index("docs-*")

    os_client.client.indices.get_alias.assert_called_once_with(
        index="docs-*", expand_wildcards="open,closed"
    )
    assert result == {"docs-20260102000000": ["docs"], "docs-20260101000000": []}


def test_get_aliases_by_index_returns_empty_on_not_found() -> None:
    os_client = _client()
    os_client.client.indices.get_alias.side_effect = NotFoundError(
        404, "index_not_found_exception", {}
    )
    assert os_client.get_aliases_by_index("docs-v9") == {}
