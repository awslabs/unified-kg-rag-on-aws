# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an index namespace holds, and its graph exported from the stores.

:func:`describe_namespace` reads the namespace's indices (live index, item
counts, ``_meta``: the embedding model and dimension and the build record),
the methodologies and strategies that build record serves, and the
doc-status registry's records. :func:`export_graph` streams every kind of
item the namespace stores (text units, entities, relationships, communities'
reports, claims, RAPTOR summary nodes), with the text units each cites and
each text unit's document, from the vector store
into JSON Lines or GraphML, a page at a time, and takes a census of the
graph store's vertices and edges (:func:`census_graph`). ``run-describe``
drives both.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Any, Literal
from xml.sax.saxutils import escape, quoteattr

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.retrieval.methodology_check import (
    NamespaceSummary,
    summarize_namespace,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    config_registry_namespace,
)
from unified_kg_rag.domain.ingestion.methodologies import (
    namespace_holdings,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    SectionType,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.filters import TRANSLATED_TEXT_PREFIX
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.ports.indexer import (
    STORED_ITEM_KINDS,
    AnnotatedKind,
    GraphCensusRow,
    GraphIndexer,
    StoredItemKind,
)
from unified_kg_rag.shared import AWSServiceError, ConfigurationError, get_logger
from unified_kg_rag.shared.exceptions import (
    DocStatusFormatError,
    DocStatusRegistryError,
    InvalidArgumentError,
)
from unified_kg_rag.shared.logging import error_log_text
from unified_kg_rag.shared.utils.files import atomic_open

logger = get_logger(__name__)

ExportFormat = Literal["jsonl", "graphml"]
# The ``kind`` of a JSON Lines export's graph-store census records.
GRAPH_RECORD_KIND = "graph"
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


class RegistryDocument(BaseModel):
    """One document of the doc-status registry, as ``run-describe
    --documents`` lists it (the inspection the runbook asks for)."""

    doc_id: str = Field(description="The registry id")
    file_path: str | None = Field(description="Its file, relative to the corpus root")
    status: DocStatus = Field(description="Its status")
    error_info: str | None = Field(description="Why it failed, when it did")
    failure_count: int = Field(description="Consecutive runs it failed on its own")
    retry_count: int = Field(description="Consecutive runs it failed for another cause")
    attempt_count: int = Field(description="Consecutive runs its processing stopped")
    written_by: str | None = Field(description="The pipeline id that wrote it last")
    updated_at: str | None = Field(description="When its status last changed")


class NamespaceDescription(NamespaceSummary):
    """What an index namespace holds (:func:`describe_namespace`)."""

    suffix: str = Field(description="The index suffix (index_value)")
    namespace: str = Field(
        description="The registry namespace: the suffix, indexing."
        "additional_suffix and the store tag"
    )
    build: dict[str, Any] | None = Field(
        description="The build record of the last ingestion run that wrote it "
        "(pipeline id, mode, build time, documents, settings, artifacts)"
    )
    build_record: Literal["present", "missing"] = Field(
        description="Whether it holds a build record: without one (a store "
        "rejected it, or a build from before records) queries take it to "
        "hold what their config builds",
    )
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
    documents: list[RegistryDocument] | None = Field(
        default=None,
        description="The registry's documents of the statuses asked for "
        "(documents=), by file; None when none were asked for",
    )


class GraphKindCensus(BaseModel):
    """The graph store's vertices (or edges) of a namespace, counted and
    digested (:func:`census_graph`)."""

    count: int = Field(description="How many the graph store holds")
    by_type: dict[str, int] = Field(
        description="How many of each type (an entity's type, an edge's "
        "label); '(untyped)': without one"
    )
    digest: str = Field(
        description="SHA-256 of their ids with their types, in id order: "
        "equal for two stores holding the same ones"
    )
    rank_digest: str = Field(
        description="SHA-256 of their ids with their stored ranks, in id order"
    )
    provenance_digest: str = Field(
        description="SHA-256 of their ids with their sorted text_unit_ids (the "
        "text units each cites), in id order"
    )
    description_digest: str = Field(
        description="SHA-256 of their ids with their stored descriptions, in "
        "id order"
    )
    topology_digest: str | None = Field(
        default=None,
        description="Edges: SHA-256 of their ids with their source, target and "
        "weight, in id order; None for vertices",
    )


class GraphCensus(BaseModel):
    """What the graph store holds of a namespace: its entity vertices and
    the relationship edges between them (communities are compared through
    their reports, which the export holds)."""

    vertices: GraphKindCensus = Field(description="The entity vertices")
    edges: GraphKindCensus = Field(description="The relationship edges")


class GraphExport(BaseModel):
    """What :func:`export_graph` wrote."""

    path: str
    format: ExportFormat
    suffix: str
    counts: dict[str, int] = Field(
        description="Items written, by record kind (a record's kind: text, "
        "entity, relationship, community, claim, summary)"
    )
    truncated: list[str] = Field(
        default_factory=list, description="The record kinds cut at max_items"
    )
    missing: list[str] = Field(
        default_factory=list,
        description="The record kinds without a live index (not exported)",
    )
    graph: GraphCensus | None = Field(
        default=None,
        description="The graph store's census (None: with max_items, without "
        "a graph store or one that cannot list its items, or when it could "
        "not be read: see graph_unavailable)",
    )
    graph_unavailable: str | None = Field(
        default=None,
        description="Why the graph store could not be read for the census "
        "(unreachable, access denied, ...); None when it was read or not asked",
    )


def describe_namespace(
    config: Config,
    suffix: str | None = None,
    *,
    providers: Providers | None = None,
    documents: Collection[DocStatus] = (),
) -> NamespaceDescription:
    """What the index namespace of ``suffix`` (default: the config's
    ``index_value``, else ``default``) holds; see :class:`NamespaceDescription`.
    ``documents`` lists the registry's documents of those statuses
    (``failed``, ``quarantined``: what the runbook inspects).

    Raises:
        ConfigurationError: ``suffix`` is not a safe name, or the vector
            store keeps no index metadata.
        StoreConnectionError: A store cannot be reached.
    """
    name = resolve_namespace(config, suffix)
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
    holdings = namespace_holdings(record, config)
    exists = any(index.index for index in indices)
    namespace = config_registry_namespace(config.indexing, name)
    records, unavailable = (
        _registry_records(providers, namespace)
        if providers.doc_status_enabled
        else (None, None)
    )
    registry = (
        dict(sorted(Counter(r.status.value for r in records).items()))
        if records is not None
        else None
    )
    listed = (
        sorted(
            (
                RegistryDocument.model_validate(record, from_attributes=True)
                for record in records or ()
                if record.status in documents
            ),
            key=lambda d: (d.file_path or d.doc_id, d.doc_id),
        )
        if documents
        else None
    )
    return NamespaceDescription(
        suffix=name,
        namespace=namespace,
        build=record,
        build_record="present" if record is not None else "missing",
        **summarize_namespace(holdings, config, exists=exists),
        indices=indices,
        registry=registry,
        registry_unavailable=unavailable,
        documents=listed,
    )


def ingested_namespaces(
    config: Config, *, providers: Providers | None = None
) -> list[str] | None:
    """The index namespaces the vector store holds an index for, for the
    error naming one it does not (as a query's ``IndexNotFoundError``
    lists them); ``None`` when the store cannot tell."""
    providers = providers or build_providers(config)
    vector = providers.build_vector_indexer()
    try:
        return vector.ingested_suffixes()
    finally:
        vector.close()


def _registry_records(
    providers: Providers, namespace: str
) -> tuple[list[DocStatusRecord] | None, str | None]:
    """The registry's records of ``namespace``, or ``None`` and why it could
    not be read.

    The registry is reported, not needed: a caller that may read the
    stores but not the registry (the deployed query task holds no DynamoDB
    grant) still gets the rest of the description, with a warning.
    """
    try:
        store = providers.build_doc_status()
        try:
            return list(store.list_namespace(namespace)), None
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


# Export: every stored kind, text units first, so every later item can name
# the documents of the text units it cites; a kind the port adds is
# exported too (its record type must be named here).
_EXPORT_KINDS: tuple[StoredItemKind, ...] = (
    "text_units",
    *(kind for kind in STORED_ITEM_KINDS if kind != "text_units"),
)
# Each record's ``kind``: the section type a source of it names
# (``SectionType``), a text unit's ``text``.
_RECORD_TYPE: dict[StoredItemKind, SectionType] = {
    "text_units": SectionType.TEXT,
    "entities": SectionType.ENTITY,
    "relationships": SectionType.RELATIONSHIP,
    "community_reports": SectionType.COMMUNITY,
    "claims": SectionType.CLAIM,
    "summary_nodes": SectionType.SUMMARY,
}


def _documents_of(item: dict[str, Any], units: dict[str, str]) -> list[str]:
    """The documents (file names) of the text units ``item`` cites."""
    return sorted({units[u] for u in item.get("text_unit_ids") or () if u in units})


def _unit_document(unit: dict[str, Any]) -> str | None:
    """A text unit's document: its path relative to the corpus root, else
    its file name; never where the corpus lived on the indexing host."""
    return source_fields.relative_path(unit) or source_fields.file_name(unit)


_UNTYPED = "(untyped)"
# The graph-store kinds a census reads, each with its census field.
_CENSUS_KINDS: tuple[tuple[AnnotatedKind, str], ...] = (
    ("entities", "vertices"),
    ("relationships", "edges"),
)


def census_graph(
    graph: GraphIndexer, suffix: str | None, *, page_size: int | None = None
) -> GraphCensus | None:
    """Count and digest the entity vertices and relationship edges
    ``suffix``'s graph store holds, through the port's projected read
    (``scan_census_rows``, a page of ``page_size`` ids at a time; any backend
    that lists its items). Two stores of the same graph give the same
    census; a lost, extra or duplicate vertex or edge, a changed type, rank,
    provenance (text units cited), description, endpoint or weight, does
    not. Every stored copy counts, so a duplicate id shows. ``None`` when
    the backend cannot list its items (no graph store).

    Raises:
        StoreConnectionError: The graph store cannot be reached (or the
            backend's typed read error).
    """
    kinds: dict[str, GraphKindCensus] = {}
    for kind, field in _CENSUS_KINDS:
        pages = graph.scan_census_rows(kind, suffix, page_size=page_size)
        if pages is None:
            return None
        kinds[field] = _kind_census(pages, edges=kind == "relationships")
    return GraphCensus.model_validate(kinds)


# Each census digest, with the fields of a row it hashes after the id.
_ROW_DIGESTS: dict[str, Callable[[GraphCensusRow], tuple[Any, ...]]] = {
    "digest": lambda row: (row.type or _UNTYPED,),
    "rank_digest": lambda row: (row.rank,),
    "provenance_digest": lambda row: (row.text_unit_ids,),
    "description_digest": lambda row: (row.description,),
}
_EDGE_DIGESTS: dict[str, Callable[[GraphCensusRow], tuple[Any, ...]]] = {
    "topology_digest": lambda row: (row.source_id, row.target_id, row.weight),
}


def _kind_census(
    pages: Iterable[list[GraphCensusRow]], *, edges: bool
) -> GraphKindCensus:
    """The census of one kind from its pages (ascending ids), one pass: the
    memory is a page's."""
    fields = {**_ROW_DIGESTS, **(_EDGE_DIGESTS if edges else {})}
    digests = {name: _Digest() for name in fields}
    types: Counter[str] = Counter()
    count = 0
    for page in pages:
        for row in sorted(page, key=lambda r: r.id):
            count += 1
            types[row.type or _UNTYPED] += 1
            for name, read in fields.items():
                digests[name].add((row.id, *read(row)))
    return GraphKindCensus(
        count=count,
        by_type=dict(sorted(types.items())),
        **{name: digest.hexdigest() for name, digest in digests.items()},
    )


# ``json.dumps(..., ensure_ascii=False)`` builds an encoder per call: a census
# of 50,000 edges makes 250,000 of them, so the one encoder is shared (the
# same text).
_ENCODE = json.JSONEncoder(ensure_ascii=False).encode


class _Digest:
    """SHA-256 of rows, one JSON array a line."""

    def __init__(self) -> None:
        self._sha = hashlib.sha256()

    def add(self, row: Iterable[Any]) -> None:
        self._sha.update(_ENCODE(list(row)).encode() + b"\n")

    def hexdigest(self) -> str:
        return self._sha.hexdigest()


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
    overwrite: bool = False,
) -> GraphExport:
    """Write the graph of ``suffix``'s namespace to ``path``, streamed from
    the vector store a page at a time.

    ``jsonl``: one JSON object per line, ``kind`` ``text``, ``entity``,
    ``relationship``, ``community`` (a community report: level, parent and
    child communities, summary), ``claim`` or ``summary`` (a RAPTOR
    summary node), with the stored fields (no vector) and ``documents``, the
    files of the text units it cites.
    ``graphml``: the entities as nodes and the relationships as edges, the
    communities as nodes of type ``community`` linked to their parent, each
    with its text unit ids and documents as attributes (lists as JSON).
    ``max_items`` caps each kind (``truncated`` names the kinds it cut);
    ``include_text`` adds each text unit's text (JSON Lines only). An
    existing ``path`` is replaced only with ``overwrite``.

    ``graph`` is the graph store's :func:`census_graph`; JSON Lines also
    ends with it, one record of kind ``graph`` for the ``vertices`` and one
    for the ``edges``, so comparing two exports compares the graph stores
    too. A sample export (``max_items``) takes none: the census reads the
    whole graph. When the graph store cannot be read, the export is written
    without it (``graph`` None) and ``graph_unavailable`` says why.

    The provenance is what the stores hold: a text unit names its document
    (``file_path`` relative to the corpus root, or a seed graph's), every
    other item the text units it was extracted from.

    Raises:
        ConfigurationError: An unknown ``format``, an unsafe ``suffix``, or
            a vector store that cannot page its items.
        InvalidArgumentError: ``path`` exists and ``overwrite`` is not set.
        StoreConnectionError: The vector store cannot be reached.
    """
    if format not in EXPORT_FORMATS:
        raise ConfigurationError(
            f"Unknown export format {format!r}: {', '.join(EXPORT_FORMATS)}"
        )
    if max_items is not None and max_items < 1:
        raise ConfigurationError("max_items must be at least 1")
    target = Path(path)
    if target.exists() and not overwrite:
        raise InvalidArgumentError(
            f"'{target}' exists: pass overwrite=True to replace it, or name "
            "another file"
        )
    name = resolve_namespace(config, suffix)
    providers = providers or build_providers(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    truncated: list[str] = []
    missing: list[str] = []
    units: dict[str, str] = {}
    census: GraphCensus | None = None
    unavailable: str | None = None
    with ExitStack() as stores:
        vector = providers.build_vector_indexer()
        stores.callback(vector.close)
        with atomic_open(target, newline="\n") as out:
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
                # Keyed as the records name their kind.
                record_kind = _RECORD_TYPE[kind].value
                counts[record_kind] = written
                if cut:
                    truncated.append(record_kind)
                if not live:
                    missing.append(record_kind)
            if max_items is None:
                census, unavailable = _read_census(providers, stores, name, page_size)
            writer.census(census)
            writer.end()
    if SectionType.RELATIONSHIP.value in missing and counts.get(
        SectionType.ENTITY.value
    ):
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
        graph=census,
        graph_unavailable=unavailable,
    )


def _read_census(
    providers: Providers, stores: ExitStack, name: str, page_size: int | None
) -> tuple[GraphCensus | None, str | None]:
    """The graph store's census of ``name``, or why it could not be read:
    the export (the vector store's items) stands without it, as a
    description stands without the registry; the comparison of two exports
    still tells a missing census from a present one."""
    try:
        graph = providers.build_graph_indexer()
        stores.callback(graph.close)
        return census_graph(graph, name, page_size=page_size), None
    except (AWSServiceError, ConfigurationError) as e:
        logger.debug("Graph census read failed: %s", e)
        reason = error_log_text(e)
        logger.warning(
            "The graph store is unavailable; exporting without its census: %s",
            reason,
        )
        return None, reason


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

    def census(self, census: GraphCensus | None) -> None: ...

    def end(self) -> None: ...


class _JsonLines(_Writer):
    def __init__(self, out: IO[str], include_text: bool) -> None:
        self.out = out
        self.include_text = include_text

    def item(self, kind: str, item: dict[str, Any], documents: list[str]) -> None:
        # "kind": an entity's or relationship's own "type" is a field.
        record = {"kind": kind, **item}
        if kind == SectionType.TEXT:
            if not self.include_text:
                record.pop("text", None)
                for key in [k for k in record if k.startswith(TRANSLATED_TEXT_PREFIX)]:
                    record.pop(key)
            document = _unit_document(item)
            attributes = item.get("attributes")
            if isinstance(attributes, dict) and "file_path" in attributes:
                # The indexing host's absolute path: the document instead.
                record["attributes"] = {**attributes, "file_path": document}
            if document:
                record["documents"] = [document]
        else:
            record["documents"] = documents
        self.out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def census(self, census: GraphCensus | None) -> None:
        if census is None:
            return
        for field, counted in census:
            record = {"kind": GRAPH_RECORD_KIND, "id": field, **counted.model_dump()}
            self.out.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        if kind == SectionType.ENTITY:
            data = self._data(
                "n",
                {
                    "kind": SectionType.ENTITY.value,
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
        elif kind == SectionType.RELATIONSHIP:
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
        elif kind == SectionType.COMMUNITY:
            community = str(item.get("community_id") or item["id"])
            node = f"community:{community}"
            data = self._data(
                "n",
                {
                    "kind": SectionType.COMMUNITY.value,
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
