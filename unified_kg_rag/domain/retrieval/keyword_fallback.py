# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LightRAG's raw-query fallback: a query without keywords searched as its
own low-level keyword.

Upstream (``operate.py``, ``kg_query``) forces a query shorter than 50
characters as the low-level keyword when keyword extraction returns none,
and answers a longer one with its failure response: a long question as one
keyword would scan the entity index for a sentence, not a name. The gate
(``search.lightrag_search.raw_query_fallback_max_len``) counts characters,
as upstream's ``len(query)`` does, in every script.
"""

from __future__ import annotations

__all__ = ["fallback_outcome", "raw_query_falls_back"]


def raw_query_falls_back(query: str, max_len: int) -> bool:
    """Whether ``query``, without keywords, is searched as its own
    low-level keyword: it is shorter than ``max_len`` characters (0 turns
    the fallback off)."""
    return bool(query) and len(query) < max_len


def fallback_outcome(query: str, max_len: int) -> str:
    """How a query without keywords is searched, as a degraded step's
    reason says it."""
    if raw_query_falls_back(query, max_len):
        return "searched the raw query as its low-level keyword"
    return (
        f"no raw-query fallback ({len(query)} characters, limit {max_len}): "
        "no entities or relationships searched"
    )
