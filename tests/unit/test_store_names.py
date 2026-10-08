# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store naming rule shared by indexers (write) and retrievers (read)."""

import re

import pytest

from unified_kg_rag.shared.utils.store_names import graph_label, store_name

pytestmark = pytest.mark.unit


def test_store_name_defaults_the_suffix() -> None:
    assert store_name("entities", None) == "entities-default"
    assert store_name("entities", "") == "entities-default"


def test_store_name_appends_the_additional_suffix() -> None:
    assert store_name("entities", "v2", "smoke") == "entities-v2-smoke"


def test_store_name_timestamp_comes_last() -> None:
    name = store_name("entities", "v2", "smoke", timestamp=True)
    assert re.fullmatch(r"entities-v2-smoke-\d{14}", name)


def test_graph_label_capitalizes_the_prefix() -> None:
    assert graph_label("entity", "tenant-a") == "Entity-tenant-a"
    assert graph_label("Community", None, "smoke") == "Community-default-smoke"
