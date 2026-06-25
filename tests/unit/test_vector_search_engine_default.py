# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""kNN engine default reconciliation (AWS-free).

Regression: the config ``vector_search`` default engine (nmslib, deprecated)
disagreed with the index-mapping default (faiss). Both must now be faiss so an
index is built with the same engine the config advertises.
"""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from aws_graphrag.domain.models import Config

pytestmark = pytest.mark.unit


def test_config_vector_search_engine_default_is_faiss(config: Config) -> None:
    assert config.indexing.opensearch.vector_search["engine"] == "faiss"


def test_index_mapping_engine_default_is_faiss(config: Config) -> None:
    # Build the indexer via __new__ so its AWS __init__ never runs; the kNN
    # mapping helper only needs the config + embedding dimension.
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst._embedding_dimension = 1024
    mapping = inst._get_knn_vector_mapping()
    assert mapping["method"]["engine"] == "faiss"
