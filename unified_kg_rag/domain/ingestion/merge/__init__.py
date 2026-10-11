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
  ``text_unit_ids`` and attributes; weight = sum of the per-text-unit
  extracted strengths (``relationship_weights``), as in a full build.

Communities and reports need no merge: their ids are content-derived, so a
delta run's are upserted next to the existing ones (see ``community_content_id``).

These functions are pure (old + delta -> merged), so they are exercised entirely
with in-memory fixtures and back the upsert path in the indexers.
"""

from .merger import (
    merge_entities,
    merge_relationships,
    relationship_id_remap,
    remove_text_units,
)

__all__ = [
    "merge_entities",
    "merge_relationships",
    "relationship_id_remap",
    "remove_text_units",
]
