# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage adapters: concrete graph/vector indexers behind the indexer ports.

Nothing is re-exported: the RAG chain, the retrieval kernel and the
retrievers import ``filter_schema`` from here, and an eager re-export would
load every indexer (and gremlin-python, opensearch-py) with it. Import an
indexer from its module (``neptune_indexer``, ``opensearch_indexer``,
``sqlite_graph_indexer``).
"""
