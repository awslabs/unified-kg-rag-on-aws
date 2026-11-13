# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a retrieved item or reported source says about itself, read one way
for every surface (the console, the MCP tools, ``run-rag``, citations,
evaluation): its id lists, documents, file name, title, the entities it is
about and its seed-graph provenance.

The id lists are decoded once, where a result enters the domain
(``RetrievalResult`` validates its metadata through
:func:`decode_list_fields`): a graph store returns a one-element list as its
value (Gremlin ``valueMap``) and an edge holds a list as its JSON string, so
every reader re-wrapped them before. Past that boundary a list field is a
``list[str]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.domain.retrieval.section_types import SectionType

# The metadata keys that hold id lists (stored items' lineage and links,
# a report's findings lineage, an item's evidence kinds).
LIST_FIELDS = frozenset(
    {
        "text_unit_ids",
        "document_ids",
        "entity_ids",
        "relationship_ids",
        "community_ids",
        "source_kinds",
        "findings_text_unit_ids",
        "findings_document_ids",
    }
)


def id_list(value: Any) -> list[str]:
    """A stored id list as strings: a list (its empty values dropped), a
    list's JSON string, or one id as the value."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set | frozenset):
        return [str(v) for v in value if v is not None and v != ""]
    return [] if value is None else [str(value)]


_STR = frozenset({str})


def _decoded(value: Any) -> bool:
    """Whether ``value`` is a list of non-empty strings already. Checked
    in C (a hub entity's lineage holds thousands of ids, read on every
    hit): every item truthy, and of type ``str``."""
    return isinstance(value, list) and all(value) and set(map(type, value)) <= _STR


def decode_list_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """``metadata`` with every :data:`LIST_FIELDS` value a ``list[str]``
    (itself when it already is: no copy)."""
    stale = [
        key
        for key in LIST_FIELDS.intersection(metadata)
        if metadata[key] is not None and not _decoded(metadata[key])
    ]
    if not stale:
        return metadata
    return {**metadata, **{key: id_list(metadata[key]) for key in stale}}


def listed(metadata: Mapping[str, Any], key: str) -> list[str]:
    """The id list ``metadata`` holds at ``key`` (empty when absent)."""
    value = metadata.get(key)
    return value if isinstance(value, list) else id_list(value)


def _attributes(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = metadata.get("attributes")
    return attributes if isinstance(attributes, Mapping) else {}


def document_ids(metadata: Mapping[str, Any]) -> list[str]:
    """The documents an item rests on: its ``document_ids``, else its
    ``document_id`` (a text unit stores it in its attributes)."""
    for value in (
        metadata.get("document_ids"),
        metadata.get("document_id"),
        _attributes(metadata).get("document_id"),
    ):
        if found := id_list(value):
            return found
    return []


def file_name(metadata: Mapping[str, Any]) -> str | None:
    """The file a text unit was chunked from, as its name (the last path
    segment, ``/`` or ``\\``): ``file_name``, else ``file_path``, of its
    stored ``attributes`` first, then of the metadata itself. Never a path."""
    for source in (_attributes(metadata), metadata):
        for key in ("file_name", "file_path"):
            value = source.get(key)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if isinstance(value, str) and value.strip():
                return value.strip().replace("\\", "/").rsplit("/", 1)[-1]
    return None


def safe_relative_path(value: Any) -> str | None:
    """``value`` as a path relative to the corpus root, or ``None`` when it
    is not one (absolute, a drive or URL, or climbing out with ``..``): a
    surface never names where the corpus lived on the indexing host."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = value.strip().replace("\\", "/")
    if path.startswith("/") or ":" in path:
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def relative_path(metadata: Mapping[str, Any]) -> str | None:
    """The file a text unit was chunked from, as its path relative to the
    corpus root (its stored ``attributes.relative_path``), when it is one."""
    return safe_relative_path(_attributes(metadata).get("relative_path"))


def text_value(value: Any) -> str | None:
    """A stored scalar as text (a graph store's one-element list read as its
    value); ``None`` when empty."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value) if value not in (None, "") else None


class EntityRef(BaseModel):
    """An entity a record is about or cites."""

    id: str | None = Field(default=None, description="The entity's id")
    name: str | None = Field(default=None, description="The entity's name")
    type: str | None = Field(default=None, description="The entity's type")
    iri: str | None = Field(
        default=None, description="The entity's IRI (seed graphs, ontologies)"
    )


def entity_refs(
    kind: str, metadata: Mapping[str, Any], item_id: str | None
) -> list[EntityRef]:
    """The entities an item of section ``kind`` is about: the entity itself,
    a relationship's or claim's endpoints, a graph query result's
    entities."""
    if kind == SectionType.ENTITY:
        return [
            EntityRef(
                id=item_id,
                name=text_value(metadata.get("name")),
                type=text_value(metadata.get("type")),
                iri=text_value(metadata.get("iri")),
            )
        ]
    if kind == SectionType.RELATIONSHIP:
        return [
            EntityRef(
                id=text_value(metadata.get(f"{end}_id")),
                name=text_value(metadata.get(f"{end}_name")),
            )
            for end in ("source", "target")
        ]
    if kind == SectionType.CLAIM:
        return [
            EntityRef(
                id=text_value(metadata.get(f"{end}_id")),
                name=text_value(metadata.get(f"{end}_name")),
            )
            for end in ("subject", "object")
            if metadata.get(f"{end}_id") or metadata.get(f"{end}_name")
        ]
    if kind == SectionType.QUERY_RESULT:
        names = metadata.get("entity_names")
        names = names if isinstance(names, list) else []
        return [
            EntityRef(id=i, name=text_value(names[n]) if n < len(names) else None)
            for n, i in enumerate(listed(metadata, "entity_ids"))
        ]
    return []


def title(kind: str, metadata: Mapping[str, Any]) -> str | None:
    """What an item of section ``kind`` is called: an entity's or report's
    name, a relationship's ``A -> B``, a graph query's plan description,
    a passage's file name."""
    if kind == SectionType.RELATIONSHIP:
        source, target = (
            text_value(metadata.get("source_name")),
            text_value(metadata.get("target_name")),
        )
        return f"{source or '?'} -> {target or '?'}" if source or target else None
    if kind == SectionType.QUERY_RESULT:
        return text_value(metadata.get("plan_description")) or text_value(
            metadata.get("operation")
        )
    if kind == SectionType.TEXT:
        return file_name(metadata)
    return text_value(metadata.get("name")) or text_value(metadata.get("title"))


def seed_provenance(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Where an item's statement comes from, when a seed graph states it.

    ``kind`` is ``seed_graph`` for an item only seed graphs state and
    ``document+seed_graph`` for one documents state as well; ``seed_graphs``
    names the graphs (their ``seed_graph`` document filter, where the store
    keeps the filters) and ``iri`` the entity's IRI. ``None`` for an item
    without seed evidence (no ``source_kinds``): its source is reported as
    before.
    """
    from unified_kg_rag.domain.ingestion.seed_graph import (
        DOCUMENT_SOURCE_KIND,
        SEED_FILTER_KEY,
        SEED_SOURCE_KIND,
        SOURCE_KINDS_FIELD,
    )
    from unified_kg_rag.domain.models import Constants

    kinds = listed(metadata, SOURCE_KINDS_FIELD)
    if SEED_SOURCE_KIND not in kinds:
        return None
    filters = _attributes(metadata).get(Constants.FILTERS.value)
    names = filters.get(SEED_FILTER_KEY) if isinstance(filters, Mapping) else None
    if names is None:
        names = metadata.get(f"{Constants.ATTRIBUTE_PREFIX.value}_{SEED_FILTER_KEY}")
    iri = metadata.get("iri")
    return {
        "kind": (
            f"{DOCUMENT_SOURCE_KIND}+{SEED_SOURCE_KIND}"
            if DOCUMENT_SOURCE_KIND in kinds
            else SEED_SOURCE_KIND
        ),
        "seed_graphs": sorted(set(id_list(names))),
        "iri": iri if isinstance(iri, str) and iri else None,
    }
