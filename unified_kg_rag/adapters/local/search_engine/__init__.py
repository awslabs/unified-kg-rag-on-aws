# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded vector/lexical store for ``providers.vector.kind: sqlite``.

``SQLiteSearchEngine`` (``engine``) implements ``SearchEngineClient`` on one
SQLite file, so the unchanged ``OpenSearchIndexer`` and
``OpenSearchRetriever`` run on it. ``query`` evaluates the OpenSearch query
DSL subset they build, ``bm25`` scores lexical matches as Lucene does and
``analyzers`` maps the mappings' analyzer names to tokenizers. Needs only the
standard library and numpy (``snowballstemmer``, from the ``[local]`` extra,
for the ``english`` analyzer's stemming).
"""

from unified_kg_rag.adapters.local.search_engine.engine import SQLiteSearchEngine

__all__ = ["SQLiteSearchEngine"]
