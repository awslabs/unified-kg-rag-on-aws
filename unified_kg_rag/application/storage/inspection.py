# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an index namespace holds, and its graph exported from the stores.

:func:`describe_namespace` reads the namespace's indices (live index, item
counts, ``_meta``: the embedding model and dimension and the build record),
the methodologies and strategies that build record serves, and the
doc-status registry's records. :func:`export_graph` streams the namespace's
entities, relationships, communities (their reports) and claims, with the
text units they cite and each text unit's document, from the vector store
into JSON Lines or GraphML, a page at a time. ``run-describe`` drives both.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Literal
from xml.sax.saxutils import escape, quoteattr

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.strategy_registry import routable_strategies
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.retrieval.methodology_check import (
    servable_strategies,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    config_registry_namespace,
)
from unified_kg_rag.domain.ingestion.methodologies import (
    held_artifacts,
    stored_artifacts,
)
from unified_kg_rag.domain.models import (
    Config,
    SearchStrategy,
    validate_safe_name,
)
from unified_kg_rag.domain.models.methodology import methodologies_holding
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.ports.indexer import StoredItemKind
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.exceptions import (
    DocStatusFormatError,
    DocStatusRegistryError,
    MethodologyNotIndexedError,
)

logger = get_logger(__name__)

ExportFormat = Literal["jsonl", "graphml"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("jsonl", "graphml")

# The index ``_meta`` keys of the embeddings and the build record
# (adapters.storage.embedding_settings).
_META_MODEL = "embedding_model_id"
_META_DIMENSION = "embedding_dimension"
_META_BUILD = "build"


class IndexDescription(BaseModel):
    """One index of a namespace."""

    kind: str = Field(description="What it holds: text_units, entities, ...")
    alias: str = Field(description="The alias queries read")
    index: str | None = Field(description="The live index behind it; None: none")
    documents: int | None = Field(description="How many items it holds")
    embedding_model_id: str | None = Field(
        default=None, description="The embedding model its vectors were made with"
    )
    embedding_dimension: int | None = Field(default=None, description="Their dimension")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="The rest of its _meta (the build record is apart)",
    )


class NamespaceDescription(BaseModel):
    """What an index namespace holds (:func:`describe_namespace`)."""

    suffix: str = Field(description="The index suffix (index_value)")
    namespace: str = Field(
        description="The registry namespace: the suffix, indexing."
        "additional_suffix and the store tag"
    )
    exists: bool = Field(description="Whether any of its indices is live")
    build: dict[str, Any] | None = Field(
        description="The build record of the last ingestion run that wrote it "
        "(pipeline id, mode, build time, documents, settings, artifacts)"
    )
    artifacts: list[str] = Field(description="The index artifacts it holds")
    artifacts_from: Literal["build_record", "config"] = Field(
        description="Where artifacts comes from: the build record, or, without "
        "one that names them, what a run of the config builds"
    )
    methodologies: list[str] = Field(description="The methodologies it holds")
    strategies: list[str] = Field(
        description="The search strategies it serves (auto routes among " "auto_routes)"
    )
    auto_routes: list[str] = Field(description="The strategies auto may pick on it")
    indices: list[IndexDescription] = Field(description="Its indices")
    registry: dict[str, int] | None = Field(
        description="The doc-status registry's records of the namespace by "
        "status (None with the registry off or unavailable)"
    )
    registry_unavailable: str | None = Field(
        default=None,
        description="Why the registry, on, could not be read (access denied, "
        "no table, ...); None when it was read or is off",
    )


class GraphExport(BaseModel):
    """What :func:`export_graph` wrote."""

    path: str
    format: ExportFormat
    suffix: str
    counts: dict[str, int] = Field(description="Items written, by kind")
    truncated: list[str] = Field(
        default_factory=list, description="The kinds cut at max_items"
    )
    missing: list[str] = Field(
        default_factory=list,
        description="The kinds without a live index (not exported)",
    )


def _suffix(config: Config, suffix: str | None) -> str:
    value = resolve_namespace(config, suffix)
    try:
        return validate_safe_name(value, "suffix")
    except ValueError as e:
        raise ConfigurationError(str(e)) from e


def describe_namespace(
    config: Config, suffix: str | None = None, *, providers: Providers | None = None
) -> NamespaceDescription:
    """What the index namespace of ``suffix`` (default: the config's
    ``index_value``, else ``default``) holds; see :class:`NamespaceDescription`.

    Raises:
        ConfigurationError: ``suffix`` is not a safe name, or the vector
            store keeps no index metadata.
        StoreConnectionError: A store cannot be reached.
    """
    name = _suffix(config, suffix)
    providers = providers or build_providers(config)
    vector = providers.build_vector_indexer()
    try:
        described = vector.describe_indices(name)
        record = vector.read_build_record(name)
    finally:
        vector.close()
    if described is None:
        raise ConfigurationError(
            "The vector store keeps no index metadata: it cannot be described"
        )
    indices = []
    for entry in described:
        meta = dict(entry.get("meta") or {})
        meta.pop(_META_BUILD, None)
        indices.append(
            IndexDescription(
                kind=entry["kind"],
                alias=entry["alias"],
                index=entry.get("index"),
                documents=entry.get("documents"),
                embedding_model_id=meta.pop(_META_MODEL, None),
                embedding_dimension=meta.pop(_META_DIMENSION, None),
                meta=meta,
            )
        )
    artifacts = held_artifacts(record, config)
    held = methodologies_holding(artifacts)
    try:
        routes = [s.value for s in routable_strategies(config, artifacts)]
    except MethodologyNotIndexedError:
        routes = []
    namespace = config_registry_namespace(config.indexing, name)
    registry, unavailable = (
        _registry_counts(providers, namespace)
        if providers.doc_status_enabled
        else (None, None)
    )
    return NamespaceDescription(
        suffix=name,
        namespace=namespace,
        exists=any(index.index for index in indices),
        build=record,
        artifacts=sorted(a.value for a in artifacts),
        artifacts_from="config" if stored_artifacts(record) is None else "build_record",
        methodologies=[m.value for m in held],
        strategies=[
            s.value
            for s in servable_strategies(artifacts, config)
            if s is not SearchStrategy.AUTO
        ],
        auto_routes=routes,
        indices=indices,
        registry=registry,
        registry_unavailable=unavailable,
    )


def _registry_counts(
    providers: Providers, namespace: str
) -> tuple[dict[str, int] | None, str | None]:
    """The registry's records of ``namespace`` by status, or ``None`` and why
    it could not be read.

    The registry is reported, not needed: a caller that may read the
    stores but not the registry (the deployed query task holds no DynamoDB
    grant) still gets the rest of the description, with a warning.
    """
    try:
        store = providers.build_doc_status()
        try:
            return (
                dict(
                    sorted(
                        Counter(
                            r.status.value for r in store.list_namespace(namespace)
                        ).items()
                    )
                ),
                None,
            )
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
    except (DocStatusRegistryError, ConfigurationError) as e:
        # Without the hint for an indexing run: describing needs no registry.
        reason = (
            e.reason
            if isinstance(e, (DocStatusRegistryError, DocStatusFormatError))
            else str(e)
        )
        logger.warning(
            "The doc-status registry is unavailable; describing the namespace "
            "without its records: %s",
            reason,
        )
        return None, reason


# Export: the kinds in the order they are written (text units first, so
# every later item can name the documents of the text units it cites).
_EXPORT_KINDS: tuple[StoredItemKind, ...] = (
    "text_units",
    "entities",
    "relationships",
    "community_reports",
    "claims",
)
_RECORD_TYPE = {
    "text_units": "text_unit",
    "entities": "entity",
    "relationships": "relationship",
    "community_reports": "community",
    "claims": "claim",
}


def _documents_of(item: dict[str, Any], units: dict[str, str]) -> list[str]:
    """The documents (file names) of the text units ``item`` cites."""
    return sorted({units[u] for u in item.get("text_unit_ids") or () if u in units})


def _unit_document(unit: dict[str, Any]) -> str | None:
    attributes = unit.get("attributes") or {}
    name = attributes.get("file_path") or attributes.get("file_name")
    return str(name) if name else None


def export_graph(
    config: Config,
    path: str | Path,
    *,
    suffix: str | None = None,
    format: ExportFormat = "jsonl",
    providers: Providers | None = None,
    page_size: int | None = None,
    max_items: int | None = None,
    include_text: bool = False,
) -> GraphExport:
    """Write the graph of ``suffix``'s namespace to ``path``, streamed from
    the vector store a page at a time.

    ``jsonl``: one JSON object per line, ``kind`` ``text_unit``, ``entity``,
    ``relationship``, ``community`` (a community report: level, parent and
    child communities, summary) or ``claim``, with the stored fields (no
    vector) and ``documents``, the files of the text units it cites.
    ``graphml``: the entities as nodes and the relationships as edges, the
    communities as nodes of type ``community`` linked to their parent, each
    with its text unit ids and documents as attributes (lists as JSON).
    ``max_items`` caps each kind (``truncated`` names the kinds it cut);
    ``include_text`` adds each text unit's text (JSON Lines only).

    The provenance is what the stores hold: a text unit names its document
    (``file_path`` relative to the corpus root, or a seed graph's), every
    other item the text units it was extracted from.

    Raises:
        ConfigurationError: An unknown ``format``, an unsafe ``suffix``, or
            a vector store that cannot page its items.
        StoreConnectionError: A store cannot be reached.
    """
    if format not in EXPORT_FORMATS:
        raise ConfigurationError(
            f"Unknown export format {format!r}: {', '.join(EXPORT_FORMATS)}"
        )
    if max_items is not None and max_items < 1:
        raise ConfigurationError("max_items must be at least 1")
    name = _suffix(config, suffix)
    providers = providers or build_providers(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    truncated: list[str] = []
    missing: list[str] = []
    units: dict[str, str] = {}
    vector = providers.build_vector_indexer()
    try:
        with target.open("w", encoding="utf-8", newline="\n") as out:
            writer: _Writer = (
                _JsonLines(out, include_text) if format == "jsonl" else _GraphML(out)
            )
            writer.start()
            for kind in _EXPORT_KINDS:
                pages = vector.scan_items(kind, name, page_size=page_size)
                if pages is None:
                    raise ConfigurationError(
                        "The vector store cannot page its items: it cannot be "
                        "exported"
                    )
                written, cut, live = _write_kind(writer, kind, pages, units, max_items)
                counts[kind] = written
                if cut:
                    truncated.append(kind)
                if not live:
                    missing.append(kind)
            writer.end()
    finally:
        vector.close()
    if "relationships" in missing and counts.get("entities"):
        logger.warning(
            "The namespace '%s' has no relationships index "
            "(indexing.opensearch.build_relationship_vector_index): the "
            "export holds no relationships",
            name,
        )
    logger.info("Exported the graph of '%s' to '%s': %s", name, target, counts)
    return GraphExport(
        path=str(target),
        format=format,
        suffix=name,
        counts=counts,
        truncated=truncated,
        missing=missing,
    )


def _write_kind(
    writer: _Writer,
    kind: StoredItemKind,
    pages: Iterator[list[dict[str, Any]]],
    units: dict[str, str],
    max_items: int | None,
) -> tuple[int, bool, bool]:
    """Write the items of ``kind``; how many, whether ``max_items`` cut
    them and whether any page was read (an index exists and holds items)."""
    written = 0
    live = False
    for page in pages:
        live = True
        for item in page:
            if max_items is not None and written >= max_items:
                close = getattr(pages, "close", None)
                if callable(close):
                    close()
                return written, True, live
            if kind == "text_units" and (document := _unit_document(item)):
                units[str(item["id"])] = document
            writer.item(_RECORD_TYPE[kind], item, _documents_of(item, units))
            written += 1
    return written, False, live


class _Writer:
    def start(self) -> None: ...

    def item(self, kind: str, item: dict[str, Any], documents: list[str]) -> None: ...

    def end(self) -> None: ...


class _JsonLines(_Writer):
    def __init__(self, out: IO[str], include_text: bool) -> None:
        self.out = out
        self.include_text = include_text

    def item(self, kind: str, item: dict[str, Any], documents: list[str]) -> None:
        # "kind": an entity's or relationship's own "type" is a field.
        record = {"kind": kind, **item}
        if kind == "text_unit":
            if not self.include_text:
                record.pop("text", None)
                for key in [k for k in record if k.startswith("translated_text_")]:
                    record.pop(key)
            if document := _unit_document(item):
                record["documents"] = [document]
        else:
            record["documents"] = documents
        self.out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# GraphML attributes, one <key> each: (id, for, name).
_GRAPHML_KEYS = (
    ("n_type", "node", "type"),
    ("n_kind", "node", "kind"),
    ("n_name", "node", "name"),
    ("n_description", "node", "description"),
    ("n_rank", "node", "rank"),
    ("n_level", "node", "level"),
    ("n_iri", "node", "iri"),
    ("n_text_unit_ids", "node", "text_unit_ids"),
    ("n_documents", "node", "documents"),
    ("e_type", "edge", "type"),
    ("e_description", "edge", "description"),
    ("e_weight", "edge", "weight"),
    ("e_predicate_iri", "edge", "predicate_iri"),
    ("e_text_unit_ids", "edge", "text_unit_ids"),
    ("e_documents", "edge", "documents"),
)


def _value(value: Any) -> str:
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


class _GraphML(_Writer):
    def __init__(self, out: IO[str]) -> None:
        self.out = out

    def start(self) -> None:
        self.out.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        )
        for key, scope, name in _GRAPHML_KEYS:
            self.out.write(
                f'  <key id="{key}" for="{scope}" attr.name="{name}" '
                'attr.type="string"/>\n'
            )
        self.out.write('  <graph id="G" edgedefault="directed">\n')

    def _data(self, prefix: str, values: dict[str, Any]) -> str:
        return "".join(
            f'<data key="{prefix}_{name}">{escape(_value(value))}</data>'
            for name, value in values.items()
            if value not in (None, "", [])
        )

    def item(self, kind: str, item: dict[str, Any], documents: list[str]) -> None:
        if kind == "entity":
            data = self._data(
                "n",
                {
                    "kind": "entity",
                    "type": item.get("type"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "rank": item.get("rank"),
                    "iri": item.get("iri"),
                    "text_unit_ids": item.get("text_unit_ids"),
                    "documents": documents,
                },
            )
            self.out.write(f"    <node id={quoteattr(str(item['id']))}>{data}</node>\n")
        elif kind == "relationship":
            data = self._data(
                "e",
                {
                    "type": item.get("type"),
                    "description": item.get("description"),
                    "weight": item.get("weight"),
                    "predicate_iri": item.get("predicate_iri"),
                    "text_unit_ids": item.get("text_unit_ids"),
                    "documents": documents,
                },
            )
            self.out.write(
                f"    <edge id={quoteattr(str(item['id']))} "
                f"source={quoteattr(str(item.get('source_id')))} "
                f"target={quoteattr(str(item.get('target_id')))}>{data}</edge>\n"
            )
        elif kind == "community":
            community = str(item.get("community_id") or item["id"])
            node = f"community:{community}"
            data = self._data(
                "n",
                {
                    "kind": "community",
                    "name": item.get("name"),
                    "description": item.get("summary"),
                    "rank": item.get("rank"),
                    "level": item.get("level"),
                    "text_unit_ids": item.get("text_unit_ids"),
                    "documents": documents,
                },
            )
            self.out.write(f"    <node id={quoteattr(node)}>{data}</node>\n")
            if (parent := item.get("parent_community_id")) not in (None, ""):
                self.out.write(
                    f"    <edge source={quoteattr(f'community:{parent}')} "
                    f"target={quoteattr(node)}>"
                    '<data key="e_type">parent_of</data></edge>\n'
                )

    def end(self) -> None:
        self.out.write("  </graph>\n</graphml>\n")
