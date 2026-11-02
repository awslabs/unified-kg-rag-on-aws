# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic procurement seed graph (a vendor supplying a buyer), as JSONL
records, as CSV files, and as the ``SeedGraph`` both state."""

from __future__ import annotations

import json
from typing import Any

from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedGraph, SeedNode

EX = "https://example.org/procure#"

NODES: list[dict[str, Any]] = [
    {
        "kind": "node",
        "id": "v1",
        "name": "Acme Supplies",
        "type": "VENDOR",
        "iri": EX + "acme",
        "description": "Industrial valve vendor.",
        "properties": {"country": "DE", "rating": None},
    },
    {"kind": "node", "id": "b1", "name": "Globex Buyer", "type_iri": EX + "Buyer"},
]
EDGES: list[dict[str, Any]] = [
    {
        "kind": "edge",
        "source": "v1",
        "target": "b1",
        "type": "SUPPLIES",
        "description": "Acme supplies Globex with valves.",
        "weight": 2,
    }
]
NODES_CSV = (
    "id,name,type,type_iri,description,iri,confidence,country\n"
    f"v1,Acme Supplies,VENDOR,,Industrial valve vendor.,{EX}acme,,DE\n"
    f"b1,Globex Buyer,,{EX}Buyer,,,,\n"
)
EDGES_CSV = (
    "source,target,type,predicate_iri,description,weight\n"
    "v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2\n"
)
# What both files above state.
PROCURE = SeedGraph(
    nodes=(
        SeedNode(
            id="v1",
            name="Acme Supplies",
            type="VENDOR",
            iri=EX + "acme",
            description="Industrial valve vendor.",
            properties={"country": "DE"},
        ),
        SeedNode(id="b1", name="Globex Buyer", type_iri=EX + "Buyer"),
    ),
    edges=(
        SeedEdge(
            source="v1",
            target="b1",
            type="SUPPLIES",
            description="Acme supplies Globex with valves.",
            weight=2,
        ),
    ),
)


def lines(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)
