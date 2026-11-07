# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store naming rule shared by indexers (write) and retrievers (read)."""

import re
from types import SimpleNamespace

import pytest

from unified_kg_rag.domain.ingestion.delta_detector import assign_document_identity
from unified_kg_rag.domain.ingestion.document_identity import (
    REGISTRY_NAMESPACE_KEY,
    registry_namespace,
)
from unified_kg_rag.domain.models import Document
from unified_kg_rag.domain.retrieval.store_names import (
    graph_label,
    index_suffix,
    store_name,
)
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit


def test_store_name_defaults_the_suffix() -> None:
    assert store_name("entities", None) == "entities-default"
    assert store_name("entities", "") == "entities-default"


def test_store_name_appends_the_additional_suffix() -> None:
    assert store_name("entities", "v2", "smoke") == "entities-v2-smoke"


def test_store_name_timestamp_comes_last() -> None:
    name = store_name("entities", "v2", "smoke", timestamp=True)
    # Millisecond resolution: YYYYmmddHHMMSSfff.
    assert re.fullmatch(r"entities-v2-smoke-\d{17}", name)


def test_graph_label_capitalizes_the_prefix() -> None:
    assert graph_label("entity", "tenant-a") == "Entity-tenant-a"
    assert graph_label("Community", None, "smoke") == "Community-default-smoke"


@pytest.mark.parametrize(
    ("value", "suffix"),
    [
        ("tenant_a", "tenant_a"),
        (["tenant_a", "other"], "tenant_a"),
        (("tenant_b",), "tenant_b"),
        ("", None),
        ([], None),
        (None, None),
        (7, None),
    ],
)
def test_the_registry_and_the_stores_read_one_suffix(value, suffix) -> None:
    # The registry namespace of a document and the index suffix of its
    # artifacts come from the same rule.
    document = Document(
        document_id="d",
        file_name="a.txt",
        file_path="/corpus/a.txt",
        page_content="x",
        metadata={} if value is None else {"index": value},
    )
    assign_document_identity(document, "/corpus")
    item = SimpleNamespace(attributes=dict(document.metadata))

    assert index_suffix(value) == suffix
    assert BaseIndexer.get_suffix(item) == (suffix or "default")
    assert document.metadata[REGISTRY_NAMESPACE_KEY] == registry_namespace(
        BaseIndexer.get_suffix(item)
    )


@pytest.mark.parametrize("value", ["../escape", "with space", ["a/b"]])
def test_an_unsafe_suffix_is_refused_by_the_registry_as_by_the_stores(value) -> None:
    document = Document(
        document_id="d",
        file_name="a.txt",
        file_path="/corpus/a.txt",
        page_content="x",
        metadata={"index": value},
    )
    with pytest.raises(ConfigurationError, match="index suffix"):
        BaseIndexer.get_suffix(SimpleNamespace(attributes={"index": value}))
    with pytest.raises(ConfigurationError, match="index suffix"):
        assign_document_identity(document, "/corpus")
