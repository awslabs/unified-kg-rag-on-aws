# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Incremental merge of a delta index into the existing graph artifacts.

Ports Microsoft GraphRAG's ``index/update/*`` merge semantics to operate on
aws-graphrag's Pydantic domain models (no pandas):

- entities merge by name (natural key): concatenate descriptions, union
  ``text_unit_ids``, recompute frequency/rank.
- relationships merge by (source, target): union ``text_unit_ids``, average
  weight.
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
)

__all__ = [
    "DeltaMergeResult",
    "merge_communities",
    "merge_community_reports",
    "merge_entities",
    "merge_relationships",
]
