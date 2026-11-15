# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``search.graph_expansion`` section: the walk around a query's seeds."""

from typing import Literal

from pydantic import Field

from unified_kg_rag.domain.models.evidence import evidence

from ._common import ConfigSection


class GraphExpansionConfig(ConfigSection):
    """The walk around a query's seed entities, on every graph store
    (Neptune, Gremlin Server, the embedded SQLite graph) and over the
    relationships index under a document filter."""

    max_hops: int = Field(
        default=3,
        ge=1,
        description=(
            "Depth of the graph expansion around the seed entities at query " "time."
        ),
        json_schema_extra=evidence("unmeasured"),
    )
    max_results_per_hop: int = Field(
        default=50,
        ge=1,
        description="Maximum number of neighbours one node contributes per "
        "traversal hop, so a hub entity cannot fan the expansion out unbounded",
        json_schema_extra=evidence("unmeasured"),
    )
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance orders entities by distance from the "
            "seeds (seeds first) and, at the same distance, by this "
            "importance: 'rank' (the indexed entity rank, its degree), "
            "'degree' (the entity's edges counted at query time, as MS "
            "GraphRAG ranks entities; read as 'rank' under a document filter, "
            "whose walk runs over the search store) or 'none' (a neutral 0.5 "
            "for every entity, so expansion ranks by distance alone)."
        ),
    )
    traversal_fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description=(
            "Graph expansion (the graph walk, or under a document filter the "
            "walk over the relationships index) reaches up to top_k * "
            "retrieval_multiplier * this many entities besides the seeds, "
            "split across the seeds (evenly; for local search's in proportion "
            "to their scores: each seed reaches at most its share), ranks "
            "them by relevance, then keeps every seed plus the "
            "top top_k * retrieval_multiplier of the others. 1 = no "
            "over-fetch: every reached entity is kept, with no cut after "
            "ranking."
        ),
        json_schema_extra=evidence("unmeasured"),
    )
    deterministic_order: bool = Field(
        default=True,
        description=(
            "Neptune graph expansion takes each vertex's max_results_per_hop "
            "neighbours in id order, so two builds of one graph (edges "
            "written in another order) expand to the same entities: "
            "reproducible results across rebuilds. The cost: each expanded "
            "vertex has all its neighbours read and sorted before the cut, "
            "which grows with hub degree (Gremlin Server, 2,500-neighbour "
            "hubs at the defaults: 6.8 to 23.2 ms per query; 60-neighbour "
            "hubs: 4.5 to 4.8 ms; Neptune hub latency is unmeasured). false: "
            "the first neighbours the store returns (an order Neptune does "
            "not specify), so a rebuild can change the results. Of equally "
            "short paths to an entity, the one from the better-ranked seed "
            "is kept, then the first by content, either way. The embedded "
            "SQLite graph store always takes neighbours by id and ignores it."
        ),
    )
