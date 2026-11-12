# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seed graph configuration (``processing.seed_graph``)."""

import math
from typing import Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..retrieval import (
    validate_safe_name,
)
from ._common import ConfigSection

# The document filter every item of a seed graph carries, its name
# (domain.ingestion.seed_graph.SEED_FILTER_KEY): reserved in its filters.
SEED_GRAPH_FILTER_KEY = "seed_graph"
SeedNameCollision = Literal["error", "merge", "qualify"]
# The default max_bytes of a seed graph, and of one of kind rdf: an RDF
# parser holds the whole graph, from about 25 (N-Triples) up to about 150
# (dense Turtle) times the file size in memory, before max_nodes can be
# checked, so 256 MiB of Turtle needed some 28 GB; 32 MiB of it up to about
# 5 GB, within the default 8 GiB ingestion task. jsonl and csv stop at the caps while reading.
SEED_MAX_BYTES = 256 * 1024 * 1024
RDF_SEED_MAX_BYTES = 32 * 1024 * 1024


class SeedGraphConfig(ConfigSection):
    """A knowledge graph imported as virtual documents of the corpus."""

    # A list item: the unknown-key check of the config loader walks nested
    # sections, not list items, so forbid makes a misspelled key an error.
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Stable name of the seed graph: it names its virtual documents "
            "(seed_graphs/<name>/...) and the 'seed_graph' document filter "
            "of everything it states. Renaming it imports the seed again. "
            "Lowercase letters, digits, '-' and '_', starting with a letter "
            "or digit; unique among the seed graphs."
        ),
    )
    kind: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description=(
            "Registered seed source that reads the graph: 'jsonl' (nodes and "
            "edges, one JSON object per line), 'csv' (a nodes file and an "
            "edges file), 'rdf' (an RDF A-Box, with the ontology extra), or a "
            "kind a deployment registers (register_seed_source). The file "
            "formats: docs/reference/seed-graph-formats.md."
        ),
    )
    path: str = Field(
        min_length=1,
        description=(
            "Where the source reads: a local file or directory (or a "
            "file:// URL), an S3 object (s3://bucket/key) or an S3 prefix "
            "read as a directory (s3://bucket/prefix/, the trailing '/' "
            "required), read with the run's AWS session (s3:GetObject, and "
            "s3:ListBucket for a prefix). A relative path is resolved against the config file's "
            "directory. Logs and errors show it without userinfo, query or "
            "fragment. Not keyed: the converted content is, so moving the "
            "files re-indexes nothing."
        ),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Settings of the source kind. Every kind takes 'dangling' (an "
            "edge naming an unknown node: 'error', the default, or 'skip') "
            "and 'qualifier_property' (the node property whose value "
            "on_name_collision: qualify appends). 'csv' also takes 'nodes' "
            "and 'edges' (the file names in the directory, default "
            "nodes.csv and edges.csv; edges: null reads no edges file) and "
            "'property_columns' (the only extra columns kept as properties). "
            "'rdf' takes 'prefixes', 'name_properties', "
            "'description_properties', 'languages', 'labels', 'properties', "
            "'exclude_properties' and 'format' "
            "(docs/reference/seed-graph-formats.md#rdf). "
            "An unknown option fails the import. Not keyed: the converted "
            "content is."
        ),
    )
    secret_arn: str | None = Field(
        default=None,
        pattern=r"^arn:aws[a-z-]*:secretsmanager:",
        description=(
            "ARN of a Secrets Manager secret a registered service source "
            "reads at load time; 'jsonl' and 'csv' take none. Never logged."
        ),
    )
    filters: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description=(
            "Document filters of the seed's virtual documents, as a "
            "document's metadata filters: what the seed states matches an "
            "attr_<key> filter as the documents carrying the same filters "
            "do. 'seed_graph': <name> is always added, so the key "
            "'seed_graph' is reserved."
        ),
    )
    on_name_collision: SeedNameCollision = Field(
        default="error",
        description=(
            "Two seed nodes whose names are one entity (the same name after "
            "normalization) but that share neither an id nor an IRI: "
            "'error' fails the import, listing them; 'merge' makes them one "
            "entity (the smallest IRI is kept); 'qualify' appends "
            "' (<qualifier>)' to the later names so each is its own entity, "
            "which documents then do not link to by name."
        ),
    )
    description_properties: list[str] = Field(
        default_factory=list,
        description=(
            "Node and edge property keys written as 'key: value' lines after "
            "the description, so answers can use them."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence of the seed's statements (a node's own confidence "
            "wins). An item a document also states takes the document's "
            "confidence."
        ),
    )
    shards: int = Field(
        default=16,
        ge=1,
        le=1024,
        description=(
            "Number of virtual documents the seed is split into, by a stable "
            "hash of each node's id: an edit re-indexes only the shards it "
            "touches. Changing the value re-indexes the whole seed."
        ),
    )
    max_items_per_unit: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Node and edge statements per text unit (the unit of lineage: "
            "what a removed statement re-derives)."
        ),
    )
    max_nodes: int = Field(
        default=100_000,
        ge=1,
        description="Fail the import above this many nodes (never truncated).",
    )
    max_edges: int = Field(
        default=500_000,
        ge=1,
        description="Fail the import above this many edges (never truncated).",
    )
    max_bytes: int = Field(
        default=SEED_MAX_BYTES,
        ge=1,
        description=(
            "Fail the import when the source files hold more bytes together "
            "(checked before they are read, and while reading). Kind 'rdf' "
            "defaults to 32 MiB: its parser holds the whole graph, from about "
            "25 (N-Triples) up to about 150 (dense Turtle) times the file "
            "size in memory (32 MiB: up to about 5 GB), before max_nodes and "
            "max_edges are checked, so size the task's memory before raising "
            "it ('jsonl' and 'csv' stop at those caps "
            "while reading)."
        ),
    )
    max_property_keys: int = Field(
        default=64,
        ge=0,
        le=512,
        description=(
            "Fail the import when the seed uses more distinct property keys "
            "(the keys of attributes.seed_properties, stored but not indexed)."
        ),
    )
    max_description_chars: int = Field(
        default=1000,
        ge=50,
        description=(
            "Cut seed descriptions to this many characters, at a word "
            "boundary, so a seed-only item never needs the description "
            "summarizer."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _rdf_max_bytes(cls, data: Any) -> Any:
        # Unless the item sets it: model_copy (a resolved path) keeps it.
        if (
            isinstance(data, dict)
            and data.get("kind") == "rdf"
            and "max_bytes" not in data
        ):
            return {**data, "max_bytes": RDF_SEED_MAX_BYTES}
        return data

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        validate_safe_name(value, "seed_graphs.name")
        if not value[0].isalnum():
            raise ValueError(
                f"Invalid seed_graphs.name '{value}': it starts with a "
                "lowercase letter or a digit."
            )
        return value

    @field_validator("filters")
    @classmethod
    def _check_filters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if SEED_GRAPH_FILTER_KEY in value:
            raise ValueError(
                f"'{SEED_GRAPH_FILTER_KEY}' is reserved: every item of a seed "
                "graph carries its name under it"
            )
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"filter '{key}' is not a finite number")
        return value
