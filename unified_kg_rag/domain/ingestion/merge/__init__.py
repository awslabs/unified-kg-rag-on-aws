# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental merge of a delta index into the existing graph artifacts.

Ports Microsoft GraphRAG's ``index/update/*`` merge semantics to operate on
unified-kg-rag-on-aws's Pydantic domain models (no pandas):

- entities merge by id, else by identity key (``entity_key``): union
  description lines,
  ``text_unit_ids``, ``community_ids`` and attributes; recompute frequency;
  keep the max rank/confidence.
- relationships merge by id, else by (source, target, type): union
  description lines,
  ``text_unit_ids`` and attributes; weight = number of supporting text units.
- communities/reports: id-offset append (MS never re-clusters globally on an
  incremental run; new communities are appended, not merged into existing ones).

These functions are pure (old + delta -> merged), so they are exercised entirely
with in-memory fixtures and back the upsert path in the indexers.
"""

from .merger import (
    DeltaMergeResult,
    merge_communities,
    merge_community_reports,
    merge_entities,
    merge_relationships,
    relationship_id_remap,
    remove_text_units,
)

__all__ = [
    "DeltaMergeResult",
    "merge_communities",
    "merge_community_reports",
    "merge_entities",
    "merge_relationships",
    "relationship_id_remap",
    "remove_text_units",
]
