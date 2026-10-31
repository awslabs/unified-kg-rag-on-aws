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
from hypothesis import given
from hypothesis import strategies as st
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.integration.test_incremental_source_scopes import (
    ScopeStack,
    scope_test_config,
    write_corpus,
)
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import (
    scope_namespace,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    config_registry_namespace,
    store_namespace_tag,
)
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


def _store_ids(stack: ScopeStack) -> dict[str, object]:
    return {
        "vectors": {
            collection: sorted(items)
            for collection, items in stack.vectors.data.items()
            if items
        },
        "graph": {
            collection: stack.graph.ids(collection)
            for collection in ("entities", "relationships")
        },
    }


def test_another_namespaces_documents_do_not_keep_deleted_artifacts(
    table, tmp_path
) -> None:
    x = _stack(table, tmp_path, "x")
    y = _stack(table, tmp_path, "y")
    source_x = write_corpus(tmp_path / "src-x", {"x1.txt": X1_TEXT, "x2.txt": X2_TEXT})
    # Y's document mentions the same entities, so their ids are X's too.
    source_y = write_corpus(tmp_path / "src-y", {"y1.txt": Y1_TEXT})
    x.run(source_x)
    y.run(source_y)
    y_before = (_store_ids(y), _records(table, "y"))

    write_corpus(source_x, {"x2.txt": X2_TEXT})
    context = x.run(source_x)

    assert len(context.incremental_delta.deleted) == 1
    # Vendor and Buyer only came from x1.txt in X: Y's y1.txt is another
    # namespace's document and does not keep them in X's stores.
    assert x.entity_names() == {"Depot"}
    assert x.texts() == {X2_TEXT}
    assert (_store_ids(y), _records(table, "y")) == y_before

    # Text-unit ids derive from the registry key, which holds the source
    # scope: the fresh build reads the same directory.
    fresh = _stack(FakeDocStatusStore(), tmp_path / "fresh", "x")
    fresh.run(source_x)
    assert _store_ids(x) == _store_ids(fresh)


def test_a_survivor_of_the_same_namespace_still_keeps_shared_artifacts(
    table, tmp_path
) -> None:
    x = _stack(table, tmp_path, "x")
    y = _stack(table, tmp_path, "y")
    source_x = write_corpus(tmp_path / "src-x", {"x1.txt": X1_TEXT, "x3.txt": Y1_TEXT})
    x.run(source_x)
    y.run(write_corpus(tmp_path / "src-y", {"y1.txt": X2_TEXT}))

    write_corpus(source_x, {"x3.txt": Y1_TEXT})
    x.run(source_x)

    assert x.entity_names() == {"Vendor", "Buyer"}
    assert x.texts() == {Y1_TEXT}


def test_retiring_a_scope_only_reaches_the_runs_namespace(table, tmp_path) -> None:
    x = _stack(table, tmp_path, "x")
    y = _stack(table, tmp_path, "y")
    # Both namespaces indexed the same directory.
    old = write_corpus(tmp_path / "old", {"x1.txt": X1_TEXT})
    x.run(old)
    y.run(old)
    y_before = (_store_ids(y), _records(table, "y"))
    new = write_corpus(tmp_path / "new", {"x2.txt": X2_TEXT})

    x.config.indexing.retire_source_scopes = [old.as_posix()]
    context = x.run(new)

    assert len(context.incremental_delta.deleted) == 1
    assert x.entity_names() == {"Depot"}
    assert (_store_ids(y), _records(table, "y")) == y_before


def test_another_namespaces_directories_are_not_reported(
    table, tmp_path, caplog
) -> None:
    x = _stack(table, tmp_path, "x")
    y = _stack(table, tmp_path, "y")
    y.run(write_corpus(tmp_path / "src-y", {"y1.txt": Y1_TEXT}))

    with caplog.at_level("WARNING"):
        x.run(write_corpus(tmp_path / "src-x", {"x1.txt": X1_TEXT}))

    assert not [
        r.getMessage()
        for r in caplog.records
        if "other local source directories" in r.getMessage()
    ]


# --- index/label prefixes ------------------------------------------------------
# Two configurations that differ only in their OpenSearch index and Neptune
# label prefixes write different stores. The CDK stack injects one fixed table
# name, so switching prefixes on a stack shares the table under the same
# suffixes: the prefixes must keep their records apart.


@pytest.fixture(params=["fake", "sqlite"])
def prefix_table(request, tmp_path) -> Iterator[DocStatusPort]:
    """One registry table both store sets write."""
    if request.param == "fake":
        yield FakeDocStatusStore()
        return
    store = SQLiteDocStatusStore(Config(), path=tmp_path / "doc-status.db")
    yield store
    store.close()


def _prefixed_stack(
    registry: DocStatusPort, tmp_path: Path, name: str | None
) -> ScopeStack:
    """A stack on the default suffix whose stores are named ``name`` (the
    default prefixes when ``None``)."""
    config = scope_test_config()
    if name is not None:
        opensearch = config.indexing.opensearch
        for field in type(opensearch).model_fields:
            if field.endswith("_index_prefix"):
                setattr(opensearch, field, f"{name}-{field}")
        config.indexing.neptune.entity_label_prefix = f"Entity{name}"
        config.indexing.neptune.community_label_prefix = f"Community{name}"
    return ScopeStack(registry, tmp_path / (name or "defaults"), config=config)


def test_other_prefixes_records_do_not_keep_a_changed_documents_artifacts(
    prefix_table, tmp_path
) -> None:
    x = _prefixed_stack(prefix_table, tmp_path, None)
    y = _prefixed_stack(prefix_table, tmp_path, "other")
    # The same file in both corpora: the same text-unit and entity ids.
    source_x = write_corpus(tmp_path / "src-x", {"doc.txt": X1_TEXT})
    x.run(source_x)
    y.run(write_corpus(tmp_path / "src-y", {"doc.txt": X1_TEXT}))
    y_before = (_store_ids(y), _records_of(prefix_table, y))

    write_corpus(source_x, {"doc.txt": X2_TEXT})
    context = x.run(source_x)

    assert len(context.incremental_delta.changed) == 1
    # Y's record lists the old text unit and entities, but in Y's stores:
    # it must not keep them in X's.
    assert x.texts() == {X2_TEXT}
    assert x.entity_names() == {"Depot"}
    assert (_store_ids(y), _records_of(prefix_table, y)) == y_before


def test_other_prefixes_index_a_corpus_already_in_the_registry(
    prefix_table, tmp_path
) -> None:
    x = _prefixed_stack(prefix_table, tmp_path, None)
    y = _prefixed_stack(prefix_table, tmp_path, "other")
    source = write_corpus(tmp_path / "src", {"doc.txt": X1_TEXT})
    x.run(source)

    # Same source directory, other stores: Y's are empty, so nothing of the
    # corpus is unchanged for Y.
    context = y.run(source)

    assert len(context.incremental_delta.new) == 1
    assert y.texts() == {X1_TEXT}
    rerun = x.run(source)
    assert len(rerun.incremental_delta.unchanged) == 1
    assert not x.model.extractions


def _records_of(registry: DocStatusPort, stack: ScopeStack) -> dict[str, object]:
    namespace = config_registry_namespace(stack.config.indexing, None)
    return {
        r.doc_id: r.model_dump()
        for r in registry.list_all()
        if scope_namespace(r.scope) == namespace
    }


def test_default_prefixes_keep_the_released_namespaces() -> None:
    # Registries written before the prefixes joined the namespace keep their
    # keys: a default configuration is not re-indexed on upgrade.
    config = Config()
    assert store_namespace_tag(config.indexing) is None
    assert config_registry_namespace(config.indexing, None) == "default"
    config.indexing.additional_suffix = "x"
    assert config_registry_namespace(config.indexing, "tenant") == "tenant-x"
    # Neptune capitalizes label prefixes: "entity" names the default label.
    config.indexing.neptune.entity_label_prefix = "entity"
    assert store_namespace_tag(config.indexing) is None


@given(
    suffix=st.none() | st.from_regex(r"[a-z0-9_-]{1,12}", fullmatch=True),
    additional=st.none() | st.from_regex(r"[a-z0-9_-]{1,12}", fullmatch=True),
)
def test_default_prefixes_namespace_is_the_suffix_formula(
    suffix: str | None, additional: str | None
) -> None:
    config = Config()
    config.indexing.additional_suffix = additional
    expected = suffix or "default"
    if additional:
        expected = f"{expected}-{additional}"
    assert config_registry_namespace(config.indexing, suffix) == expected
    # Any prefix change moves it to a namespace no suffix spells.
    config.indexing.opensearch.entities_index_prefix = "custom-entities"
    tagged = config_registry_namespace(config.indexing, suffix)
    assert tagged.startswith(f"{expected}@") and tagged != expected
