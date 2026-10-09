# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two index namespaces sharing one doc-status table (AWS-free).

Every config uses the same default table name, so deployments that write
several index namespaces (``indexing.additional_suffix`` values) share one
registry. A run must only ever read another namespace's records as someone
else's: a reset of one namespace keeps the other's records, and the other
namespace's documents never keep this namespace's artifacts alive.

``DocStatusRecord.suffix`` is the item suffix the indexers are called with
(``index_value``, ``default`` when unset), the same for every
``additional_suffix``; a record's namespace is read from its scope.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.integration.test_incremental_source_scopes import (
    ScopeStack,
    scope_test_config,
    write_corpus,
)
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import scope_namespace
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports import DocStatusPort

pytestmark = pytest.mark.integration

X1_TEXT = "Vendor supplies Buyer."
X2_TEXT = "Depot ships widgets."
Y1_TEXT = "Buyer pays Vendor today."
Y2_TEXT = "Bank lends to Depot."


@pytest.fixture(params=["fake", "dynamodb"])
def table(request) -> Iterator[DocStatusPort]:
    """One registry table both namespaces write."""
    if request.param == "fake":
        yield FakeDocStatusStore()
        return
    with mock_aws():
        config = Config()
        config.aws.dynamodb.table_name = "test-doc-status"
        store = DynamoDBDocStatusStore(
            config, boto_session=boto3.Session(region_name="us-east-1")
        )
        _ = store.client
        yield store


def _stack(registry: DocStatusPort, tmp_path: Path, namespace: str) -> ScopeStack:
    """A stack writing ``default-<namespace>``, with its own stores."""
    config = scope_test_config()
    config.indexing.additional_suffix = namespace
    return ScopeStack(registry, tmp_path / namespace, config=config)


def _records(registry: DocStatusPort, namespace: str) -> dict[str, object]:
    return {
        r.doc_id: r.model_dump()
        for r in registry.list_all()
        if scope_namespace(r.scope) == f"default-{namespace}"
    }


def test_a_reset_keeps_the_records_of_other_namespaces(table, tmp_path) -> None:
    x = _stack(table, tmp_path, "x")
    y = _stack(table, tmp_path, "y")
    source_x = write_corpus(tmp_path / "src-x", {"x1.txt": X1_TEXT})
    source_y = write_corpus(tmp_path / "src-y", {"y1.txt": Y1_TEXT, "y2.txt": Y2_TEXT})
    x.run(source_x)
    y.run(source_y)
    y_records = _records(table, "y")
    assert len(y_records) == 2

    x.config.indexing.reset = True
    x.run(source_x)
    x.config.indexing.reset = False

    assert _records(table, "y") == y_records
    assert len(_records(table, "x")) == 1

    # Y's next run still knows its documents: nothing is re-extracted, and a
    # file removed from Y is removed from Y's stores.
    write_corpus(source_y, {"y1.txt": Y1_TEXT})
    context = y.run(source_y)
    delta = context.incremental_delta
    assert len(delta.unchanged) == 1 and len(delta.deleted) == 1
    assert delta.new == delta.changed == []
    assert not y.model.extractions
    assert y.texts() == {Y1_TEXT}
    assert y.entity_names() == {"Buyer", "Vendor"}
