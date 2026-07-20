# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""kNN engine default reconciliation (AWS-free).

Regression 1: the config ``vector_search`` default engine disagreed with the
index-mapping default, so an index could be built with a different engine than
the config advertised. Both must agree.

Regression 2: the shared default was ``faiss`` + ``cosinesimil``, which faiss
HNSW rejects at index creation before OpenSearch 2.19 (it supports only
l2/innerproduct), breaking every index build on the deployed 2.13 domain. The
default is now ``lucene``, which supports cosinesimil on all versions (and
covers the default 1024-dim Titan Embed V2 within lucene's 1024-dim cap).
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


def test_config_vector_search_engine_default_is_lucene(config: Config) -> None:
    assert config.indexing.opensearch.vector_search["engine"] == "lucene"


def test_index_mapping_engine_default_matches_config(config: Config) -> None:
    # Build the indexer via __new__ so its AWS __init__ never runs; the kNN
    # mapping helper only needs the config + embedding dimension.
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst._embedding_dimension = 1024
    mapping = inst._get_knn_vector_mapping()
    assert mapping["method"]["engine"] == "lucene"
    # Config and mapping must advertise the same engine.
    assert (
        mapping["method"]["engine"]
        == config.indexing.opensearch.vector_search["engine"]
    )


def test_lucene_default_supports_cosinesimil_space_type(config: Config) -> None:
    # The whole point of the fix: lucene + cosinesimil is a valid combination.
    vs = config.indexing.opensearch.vector_search
    assert vs["engine"] == "lucene"
    assert vs["space_type"] == "cosinesimil"
