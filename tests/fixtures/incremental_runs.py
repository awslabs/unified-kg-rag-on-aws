# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs over in-memory stores, interruptible at any write step.

Shared by the write-ahead crash-recovery tests. A document is a list of text
units; a text unit is a set of weighted edges between a few synthetic entity
names, so "extraction" is deterministic: each unit's record (its entities,
edges and claims, each with text naming the unit) is stored as the
extraction stages store it, and a fresh full build of any corpus is the
records merged and resolved by the full build's code (:func:`expected_state`),
descriptions, names, types, ranks and claims included. Documents and text
units get the pipeline's ids: a document's from its registry key and its
content, a unit's from its document's and its position, so a changed
document's units all get new ids.

:meth:`IncrementalRun.run` wires the steps like the indexing stage does
(plan, write-ahead, joint removal, commit) and can be interrupted at one of
:data:`POINTS` by a :class:`Killed` ``BaseException``: like a process that
dies, it is not swallowed by the indexing manager's ``except Exception``
handlers.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeVar

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.seed_graphs import seed_shard_extraction
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.storage.neptune_codec import property_value
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
    close_attempts,
    open_attempts,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    claims_payload,
    merge_contributions,
    merge_entity_contributions,
    merge_relationship_contributions,
    record_keys,
    record_payload,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    detect_delta,
    document_doc_id,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    compute_document_id,
    compute_text_unit_id,
)
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.ingestion.schema_conformance import SchemaConformance
from unified_kg_rag.domain.ingestion.seed_graph import annotate_sources
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    DocStatus,
    Document,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.ports.indexer import IndexingStats

Content = list[list[tuple[str, str, int]]]
# A path's content, or a seed graph's shard document under its relative path.
Corpus = dict[str, Content | Document]

_TYPE = "RELATED_TO"

_Item = TypeVar("_Item", Entity, Relationship)

# The statuses a unit's claims state, picked by the unit (``_claim_status``):
# units of one claim disagree, so a build holds true, false, disputed and
# unknown claims, and a model's off-vocabulary words ("true.", "Suspected").
_STATUSES = ("TRUE", "TRUE", "true.", "FALSE", "UNKNOWN", "Suspected")

# Where a run can be interrupted, in the order the indexing stage reaches them.
POINTS = (
    "after_write_ahead",
    "mid_removal",
    "after_removal",
    "in_merge",
    # The merge's re-key and split decisions are noted, nothing written.
    "after_commit_note",
    "mid_index_delta",
    # The delta is written, the re-keyed items' old ids still stored.
    "before_retired_deletes",
    "before_record",
    # The failed item writes are noted, no document recorded yet.
    "after_write_note",
    "mid_commit",
    "mid_record_deletes",
)


class Killed(BaseException):
    """An interruption no ``except Exception`` handler absorbs."""


def _unit_id(document_id: str, index: int) -> str:
    # The chunker's rule: a document's version and the unit's position, so a
    # changed document's units all get new ids (lowercase hex: they sort
    # before every seed unit id, "sg-...", so records merge documents first).
    return compute_text_unit_id(document_id, index)


def _claim_status(where: str) -> str:
    """The status the claims of the unit at ``where`` state."""
    digest = hashlib.sha256(where.encode()).digest()
    return _STATUSES[digest[0] % len(_STATUSES)]


def document(path: str, content: Content | Document) -> Document:
    """The document of ``path``, its id derived as the pipeline derives it:
    from its registry key and its content (``assign_registry_source``)."""
    if isinstance(content, Document):
        return content
    built = Document(
        page_content=repr(content),
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )
    built.document_id = compute_document_id(
        document_doc_id(built), compute_content_hash(built)
    )
    return built


class Extraction(NamedTuple):
    """A build over a corpus: its units, their records and the graph."""

    units: list[TextUnit]
    records: list[TextUnitExtraction]
    claim_records: dict[str, list[Claim]]
    entities: list[Entity]
    relationships: list[Relationship]
    claims: list[Claim]


def _unit_records(
    path: str,
    document_id: str,
    index: int,
    unit: list[tuple[str, str, int]],
    conformance: SchemaConformance,
) -> tuple[TextUnit, TextUnitExtraction, list[Claim]]:
    """One unit's text unit, record and claims. Each statement carries the
    unit's own text, so removing a unit must remove what it said: an edge's
    name endpoints are listed with a type that follows the strength's parity
    (types are voted on), and a strength-3 edge also states a claim, with
    a status the unit picks (``_claim_status``: units disagree). The
    statements are checked against the schema as extraction checks them
    (``conformance``: a strict one drops the off-schema ones)."""
    unit_id = _unit_id(document_id, index)
    where = f"{path}#{index}"
    text_unit = TextUnit(
        id=unit_id,
        text="...",
        document_ids=[document_id],
        attributes={
            RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{unit_id}", "claims": f"c-{unit_id}"}
        },
    )
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    claims: list[Claim] = []
    for source, target, strength in unit:
        for name in (source, target):
            entities.append(
                Entity(
                    id=BaseProcessor._generate_entity_id(name),
                    name=name,
                    type="ORG" if strength % 2 else "PERSON",
                    description=f"{name} in {where}",
                    # A strength-1 statement gives no confidence (a model
                    # may leave it out): every merge takes the highest known.
                    confidence=None if strength == 1 else 0.5 + strength / 10,
                    text_unit_ids=[unit_id],
                )
            )
        edge = Relationship(
            id=BaseProcessor._generate_relationship_id(source, target, _TYPE),
            source_id=BaseProcessor._generate_entity_id(source),
            target_id=BaseProcessor._generate_entity_id(target),
            source_name=source,
            target_name=target,
            type=_TYPE,
            description=f"{source} -> {target} in {where}",
        )
        apply_text_unit_weights(edge, {unit_id: float(strength)})
        relationships.append(edge)
        if strength == 3:
            claims.append(
                Claim(
                    id=ClaimExtractor._generate_claim_id(source, target, "RATES"),
                    subject_id="",
                    subject_name=source,
                    object_id="",
                    object_name=target,
                    type="RATES",
                    status=_claim_status(where),
                    description=f"{source} rates {target} in {where}",
                    source_text=f"{source} rated {target} ({where})",
                    text_unit_ids=[unit_id],
                )
            )
    checked = conformance.conform(entities, relationships, unit=unit_id)
    record = TextUnitExtraction(
        text_unit_id=unit_id,
        entities=merge_entity_contributions(checked.entities),
        relationships=merge_relationship_contributions(checked.relationships),
        gleaned=True,
    )
    return text_unit, record, claims


def extract(corpus: Corpus, config: Config | None = None) -> Extraction:
    """Text units, their records, and the graph a build over ``corpus``
    produces under ``config`` (its extraction schema; the default when
    ``None``): the records merged and resolved by the full build's code. A
    seed shard's units and records are the pipeline's (converted under the
    same schema); it states no claim."""
    config = config or Config()
    conformance = SchemaConformance(
        config.processing.graph_extraction.extraction_schema()
    )
    units: list[TextUnit] = []
    records: list[TextUnitExtraction] = []
    claim_records: dict[str, list[Claim]] = {}
    for path, content in corpus.items():
        if isinstance(content, Document):
            seed_units, seed_records = seed_shard_extraction(content)
            units += seed_units
            records += seed_records
            continue
        document_id = document(path, content).document_id
        for index, unit in enumerate(content):
            text_unit, record, claims = _unit_records(
                path, document_id, index, unit, conformance
            )
            units.append(text_unit)
            records.append(record)
            claim_records[text_unit.id] = claims
    entities, relationships = merge_contributions(records)
    resolved, _ = GraphResolver(
        config, max_workers=1, use_process_pool=False
    ).resolve_graph(entities, relationships)
    statements = [
        claim.model_copy(deep=True)
        for unit in sorted(claim_records)
        for claim in claim_records[unit]
    ]
    claims, _ = ClaimResolver(config, max_workers=1, use_process_pool=False).resolve(
        statements, resolved["entities"]
    )
    return Extraction(
        units,
        records,
        claim_records,
        resolved["entities"],
        resolved["relationships"],
        claims,
    )


def _lines(text: str | None) -> frozenset[str]:
    return frozenset(line for line in (text or "").split("\n") if line)


def entity_state(entity: Entity) -> tuple[Any, ...]:
    """What a full build and an incremental run must agree on for an entity."""
    return (
        entity.name,
        (entity.type or "").lower(),
        _lines(entity.description),
        entity.rank,
        frozenset(entity.text_unit_ids or []),
        # The stores write an unknown confidence as 1.0.
        1.0 if entity.confidence is None else round(entity.confidence, 9),
    )


def edge_state(rel: Relationship) -> tuple[Any, ...]:
    return (
        rel.source_name,
        rel.target_name,
        _lines(rel.description),
        rel.rank,
        frozenset(rel.text_unit_ids or []),
        round(rel.weight or 0.0, 9),
    )


def claim_state(claim: Claim) -> tuple[Any, ...]:
    return (
        claim.subject_id,
        claim.status,
        _lines(claim.description),
        _lines(claim.source_text),
        frozenset(claim.text_unit_ids or []),
        # A wrong object shows even when the claim's id did not change.
        claim.object_id,
    )


def _provenance(
    entities: Iterable[Entity],
    relationships: Iterable[Relationship],
    kinds: Callable[[Entity | Relationship], list[str] | None],
) -> dict[str, Any]:
    """Each item's stated IRI (entities) and source kinds."""
    return {
        **{e.id: (e.iri, kinds(e)) for e in entities},
        **{r.id: kinds(r) for r in relationships},
    }


def expected_state(corpus: Corpus, config: Config | None = None) -> dict[str, Any]:
    """What a fresh full build of ``corpus`` under ``config`` stores (see
    :meth:`IncrementalRun.state`)."""
    built = extract(corpus, config)
    entity_units = {e.id: frozenset(e.text_unit_ids or []) for e in built.entities}
    # The graph store cuts a long property (a hub's description, with the
    # summaries off) as Neptune's codec does; the vector store keeps it.
    max_length = (config or Config()).indexing.neptune.property_max_length

    def graph_copy(item: _Item) -> _Item:
        cut = property_value(item.description, max_length)
        return item.model_copy(update={"description": cut})

    # The indexing annotates an item's source kinds from its text units.
    provenance = _provenance(built.entities, built.relationships, annotate_sources)
    return {
        "graph_provenance": provenance,
        "vector_provenance": provenance,
        "graph_entities": entity_units,
        "graph_edges": {
            r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
            for r in built.relationships
        },
        "vector_entities": entity_units,
        "vector_relationships": {r.id for r in built.relationships},
        "vector_text_units": {u.id for u in built.units},
        "entity_content": {e.id: entity_state(graph_copy(e)) for e in built.entities},
        "edge_content": {r.id: edge_state(graph_copy(r)) for r in built.relationships},
        "vector_entity_content": {e.id: entity_state(e) for e in built.entities},
        "vector_relationship_content": {
            r.id: edge_state(r) for r in built.relationships
        },
        "claims": {c.id: claim_state(c) for c in built.claims},
    }


class IncrementalRun:
    """One registry + graph + vector store, indexed by successive runs."""

    def __init__(
        self, store: DocStatusPort | None = None, config: Config | None = None
    ) -> None:
        self.config = config or Config()
        self.graph = FakeGraphStore(self.config)
        self.vector = FakeVectorStore(opensearch_config=self.config.indexing.opensearch)
        self.manager = self._manager()
        self.store: DocStatusPort = store or FakeDocStatusStore()
        self.extracted: list[str] = []
        # Kills the run (a process death) right after it opened its attempts
        # when it returns True for the paths of the documents it processes.
        self.crash_on: Callable[[list[str]], bool] | None = None

    def _manager(self) -> IndexingManager:
        return IndexingManager(
            config=self.config, vector_indexer=self.vector, graph_indexer=self.graph
        )

    def run(
        self,
        corpus: Corpus,
        interrupt: str | None = None,
        k: int = 1,
        fail: Callable[[str, str], bool] | None = None,
    ) -> bool:
        """Index ``corpus``; return whether the ``interrupt`` point fired.

        ``k`` picks which call of a counted point (the k-th store write of
        the removal or the delta, the k-th registry record or delete) fires.
        ``fail(where, item id)`` names the item writes the stores reject
        during the run's commit (reported as failed items by id, as the
        indexers report a rejected bulk item): ``where`` is
        ``graph.<method>`` or ``vector.<method>``, a store's ``upsert_*`` or
        ``update_*_lineage``.
        """
        # A run is a new process: nothing an interrupted one held in memory
        # survives it.
        self.manager = self._manager()
        inc = IncrementalIndexer(self.store, self.manager)
        restore: list[Callable[[], None]] = []
        fired = False
        try:
            if fail is not None:
                commit = inc.commit

                def rejecting_commit(*args: Any, **kwargs: Any) -> Any:
                    undo: list[Callable[[], None]] = []
                    self._reject(fail, undo)
                    try:
                        return commit(*args, **kwargs)
                    finally:
                        for step in reversed(undo):
                            step()

                inc.commit = rejecting_commit  # type: ignore[method-assign]
            if interrupt is not None:
                self._install(interrupt, k, inc, restore)
            self._run(inc, corpus)
        except Killed:
            fired = True
        finally:
            for undo in reversed(restore):
                undo()
        return fired

    def _run(self, inc: IncrementalIndexer, corpus: Corpus) -> None:
        documents = [document(path, content) for path, content in corpus.items()]
        delta, fingerprints = detect_delta(
            documents,
            inc.doc_status,
            max_attempts=self.config.indexing.max_document_attempts,
        )
        # Like the document_loading stage: before any stage processes them.
        open_attempts(inc.doc_status, delta, fingerprints)
        to_extract = set(delta.to_process)
        extracted = [d for d in documents if document_doc_id(d) in to_extract]
        self.extracted = sorted(d.file_path for d in extracted)
        if self.crash_on is not None and self.crash_on(self.extracted):
            raise Killed("crash_on")
        built = extract(
            {d.file_path: corpus[d.file_path] for d in extracted}, self.config
        )
        # The extraction stages store the units' records before indexing.
        self.vector.write_extraction_records(
            {
                record_keys(unit)["graph"]: record_payload(record)
                for unit, record in zip(built.units, built.records, strict=True)
            }
        )
        self.vector.write_extraction_records(
            {
                f"c-{unit_id}": claims_payload(claims)
                for unit_id, claims in built.claim_records.items()
            }
        )
        units, entities, edges = built.units, built.entities, built.relationships
        lineages = build_document_lineage(documents=extracted, text_units=units)
        inc.write_ahead(delta, lineages)
        if not inc.remove_changed_and_deleted(delta):
            raise AssertionError("the fake stores never fail a removal")
        inc.commit(
            lineages=lineages,
            fingerprints=fingerprints,
            text_units=units,
            entities=entities,
            relationships=edges,
            claims=built.claims,
        )
        # The run ends in the process: its attempts are not deaths.
        close_attempts(inc.doc_status, delta, fingerprints)

    def _reject(
        self, fail: Callable[[str, str], bool], restore: list[Callable[[], None]]
    ) -> None:
        """Make the stores reject the item writes ``fail`` names (see
        :meth:`run`): the others are written, the rejected ones reported."""
        for name, store in (("graph", self.graph), ("vector", self.vector)):
            for method in dir(store):
                if not (
                    method.startswith("upsert_")
                    or (method.startswith("update_") and method.endswith("_lineage"))
                ):
                    continue
                original = getattr(store, method)

                def rejecting(
                    items: list[Any], _write: Any = original, _where: str = ""
                ) -> IndexingStats:
                    rejected = [i.id for i in items if fail(_where, i.id)]
                    stats = _write([i for i in items if i.id not in set(rejected)])
                    stats.total_items = len(items)
                    if rejected:
                        stats.add_error("write rejected", len(rejected), ids=rejected)
                    return stats

                setattr(
                    store,
                    method,
                    functools.partial(rejecting, _where=f"{name}.{method}"),
                )
                restore.append(
                    lambda store=store, method=method, original=original: setattr(
                        store, method, original
                    )
                )

    def _install(
        self,
        point: str,
        k: int,
        inc: IncrementalIndexer,
        restore: list[Callable[[], None]],
    ) -> None:
        def patch(obj: Any, name: str, make: Callable[[Any], Any]) -> None:
            original = getattr(obj, name)
            setattr(obj, name, make(original))
            restore.append(lambda: setattr(obj, name, original))

        def then_kill(original: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                original(*args, **kwargs)
                raise Killed(point)

            return wrapper

        def kill(_original: Any) -> Any:
            def wrapper(*_args: Any, **_kwargs: Any) -> Any:
                raise Killed(point)

            return wrapper

        calls = {"n": 0}

        def on_kth(original: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                if calls["n"] == k:
                    raise Killed(point)
                return original(*args, **kwargs)

            return wrapper

        if point == "after_write_ahead":
            patch(inc, "write_ahead", then_kill)
        elif point == "mid_removal":
            patch(self.graph, "delete_by_id", on_kth)
            patch(self.vector, "delete_document_artifacts", on_kth)
        elif point == "after_removal":
            patch(inc, "remove_changed_and_deleted", then_kill)
        elif point == "in_merge":
            patch(self.manager, "merge_with_existing_graph", kill)
        elif point == "after_commit_note":
            patch(self.manager, "note_pending_commit", then_kill)
        elif point == "before_retired_deletes":
            patch(self.manager, "delete_retired_ids", kill)
        elif point == "after_write_note":
            patch(self.manager, "note_pending_writes", then_kill)
        elif point == "mid_index_delta":
            for store in (self.graph, self.vector):
                for name in dir(store):
                    if name.startswith("upsert_"):
                        patch(store, name, on_kth)
        elif point == "before_record":
            patch(inc, "record", kill)
        elif point == "mid_commit":
            # The k-th record of the commit (after the attempt and
            # write-ahead ones; a commit's record has no open attempt).
            def in_commit(original: Any) -> Any:
                def wrapper(record: Any) -> Any:
                    if (
                        record.status is not DocStatus.PENDING
                        and record.attempt_batch is None
                    ):
                        calls["n"] += 1
                        if calls["n"] == k:
                            raise Killed(point)
                    return original(record)

                return wrapper

            patch(self.store, "put", in_commit)
        elif point == "mid_record_deletes":
            patch(self.store, "delete", on_kth)
        else:
            raise ValueError(point)

    def state(self) -> dict[str, Any]:
        entities = self.graph.read_entities(sorted(self.graph.ids("entities")))
        edges = self.graph.read_relationships(sorted(self.graph.ids("relationships")))

        def stored(item: Entity | Relationship) -> list[str] | None:
            return item.source_kinds

        return {
            "graph_provenance": _provenance(entities, edges, stored),
            "vector_provenance": _provenance(
                self.vector.data.get("entities", {}).values(),
                self.vector.data.get("relationships", {}).values(),
                stored,
            ),
            "graph_entities": {
                e.id: frozenset(e.text_unit_ids or []) for e in entities
            },
            "graph_edges": {
                r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
                for r in edges
            },
            "vector_entities": {
                e.id: frozenset(e.text_unit_ids or [])
                for e in self.vector.data.get("entities", {}).values()
            },
            "vector_relationships": self.vector.ids("relationships"),
            "vector_text_units": self.vector.ids("text_units"),
            "entity_content": {e.id: entity_state(e) for e in entities},
            "edge_content": {r.id: edge_state(r) for r in edges},
            "vector_entity_content": {
                e.id: entity_state(e)
                for e in self.vector.data.get("entities", {}).values()
            },
            "vector_relationship_content": {
                r.id: edge_state(r)
                for r in self.vector.data.get("relationships", {}).values()
            },
            "claims": {
                c.id: claim_state(c)
                for c in self.vector.data.get("claims", {}).values()
            },
        }

    def unused_extraction_records(self) -> set[str]:
        """Extraction-store keys no stored text unit names."""
        named = {
            key
            for unit in self.vector.data.get("text_units", {}).values()
            for key in record_keys(unit).values()
        }
        return {key for _, key in self.vector.extraction_records} - named

    def registry_problems(self, corpus: Corpus) -> list[str]:
        """Registry inconsistencies: rows not PROCESSED or not in the corpus,
        text units the rows and the stores disagree on, artifacts citing a
        text unit that is not stored."""
        records = all_records(self.store)
        problems = []
        paths = sorted(r.file_path or r.doc_id for r in records)
        if paths != sorted(corpus):
            problems.append(f"registry rows {paths} != corpus {sorted(corpus)}")
        problems += [
            f"{r.file_path} is {r.status.value}"
            for r in records
            if r.status is not DocStatus.PROCESSED
        ]
        units = self.vector.ids("text_units")
        listed = {i for r in records for i in r.text_unit_ids}
        if listed != units:
            problems.append(
                f"registry text units {sorted(listed)} != stored {sorted(units)}"
            )
        artifacts = [
            *self.graph.read_entities(sorted(self.graph.ids("entities"))),
            *self.graph.read_relationships(sorted(self.graph.ids("relationships"))),
            *self.vector.data.get("claims", {}).values(),
        ]
        problems += [
            f"{item.id} cites text units that are not stored"
            for item in artifacts
            if not set(item.text_unit_ids or []) <= units
        ]
        return problems
