# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs over in-memory stores, interruptible at any write step.

Shared by the write-ahead crash-recovery tests. A document is a list of text
units; a text unit is a set of weighted edges between a few synthetic entity
names, so "extraction" is deterministic: each unit's record (its entities,
edges and claims, each with text naming the unit) is stored as the
extraction stages store it, and a fresh full build of any corpus is the
records merged and resolved by the full build's code (:func:`expected_state`),
descriptions, names, types, ranks and claims included.

:meth:`IncrementalRun.run` wires the steps like the indexing stage does
(plan, write-ahead, joint removal, commit) and can be interrupted at one of
:data:`POINTS` by a :class:`Killed` ``BaseException``: like a process that
dies, it is not swallowed by the indexing manager's ``except Exception``
handlers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, NamedTuple

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    claims_payload,
    merge_claim_contributions,
    merge_contributions,
    merge_entity_contributions,
    merge_relationship_contributions,
    record_keys,
    record_payload,
)
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta, document_doc_id
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
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

Content = list[list[tuple[str, str, int]]]
Corpus = dict[str, Content]

_TYPE = "RELATED_TO"

# Where a run can be interrupted, in the order the indexing stage reaches them.
POINTS = (
    "after_write_ahead",
    "mid_removal",
    "after_removal",
    "in_merge",
    "mid_index_delta",
    "before_record",
    "mid_commit",
    "mid_record_deletes",
)


class Killed(BaseException):
    """An interruption no ``except Exception`` handler absorbs."""


def _unit_id(path: str, index: int, unit: list[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256(f"{path}|{index}|{sorted(unit)}".encode()).hexdigest()
    return f"tu-{digest[:16]}"


def document(path: str, content: Content) -> Document:
    return Document(
        page_content=repr(content),
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


class Extraction(NamedTuple):
    """A build over a corpus: its units, their records and the graph."""

    units: list[TextUnit]
    records: list[TextUnitExtraction]
    claim_records: dict[str, list[Claim]]
    entities: list[Entity]
    relationships: list[Relationship]
    claims: list[Claim]


def _unit_records(
    path: str, index: int, unit: list[tuple[str, str, int]]
) -> tuple[TextUnit, TextUnitExtraction, list[Claim]]:
    """One unit's text unit, record and claims. Each statement carries the
    unit's own text, so removing a unit must remove what it said: an edge's
    name endpoints are listed with a type that follows the strength's parity
    (types are voted on), and a strength-3 edge also states a claim."""
    unit_id = _unit_id(path, index, unit)
    where = f"{path}#{index}"
    text_unit = TextUnit(
        id=unit_id,
        text="...",
        document_ids=[path],
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
                    confidence=0.5 + strength / 10,
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
                    status="TRUE",
                    description=f"{source} rates {target} in {where}",
                    source_text=f"{source} rated {target} ({where})",
                    text_unit_ids=[unit_id],
                )
            )
    record = TextUnitExtraction(
        text_unit_id=unit_id,
        entities=merge_entity_contributions(entities),
        relationships=merge_relationship_contributions(relationships),
        gleaned=True,
    )
    return text_unit, record, merge_claim_contributions(claims)


def extract(corpus: Corpus) -> Extraction:
    """Text units, their records, and the graph a build over ``corpus``
    produces: the records merged and resolved by the full build's code."""
    units: list[TextUnit] = []
    records: list[TextUnitExtraction] = []
    claim_records: dict[str, list[Claim]] = {}
    for path, content in corpus.items():
        for index, unit in enumerate(content):
            text_unit, record, claims = _unit_records(path, index, unit)
            units.append(text_unit)
            records.append(record)
            claim_records[text_unit.id] = claims
    config = Config()
    entities, relationships = merge_contributions(records)
    resolved, _ = GraphResolver(
        config, max_workers=1, use_process_pool=False
    ).resolve_graph(entities, relationships)
    statements = merge_claim_contributions(
        [
            claim.model_copy(deep=True)
            for unit in sorted(claim_records)
            for claim in claim_records[unit]
        ]
    )
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
        _lines(claim.description),
        _lines(claim.source_text),
        frozenset(claim.text_unit_ids or []),
    )


def expected_state(corpus: Corpus) -> dict[str, Any]:
    """What a fresh full build of ``corpus`` stores (see :meth:`state`)."""
    built = extract(corpus)
    entity_units = {e.id: frozenset(e.text_unit_ids or []) for e in built.entities}
    return {
        "graph_entities": entity_units,
        "graph_edges": {
            r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
            for r in built.relationships
        },
        "vector_entities": entity_units,
        "vector_relationships": {r.id for r in built.relationships},
        "vector_text_units": {u.id for u in built.units},
        "entity_content": {e.id: entity_state(e) for e in built.entities},
        "edge_content": {r.id: edge_state(r) for r in built.relationships},
        "vector_entity_content": {e.id: entity_state(e) for e in built.entities},
        "claims": {c.id: claim_state(c) for c in built.claims},
    }


class IncrementalRun:
    """One registry + graph + vector store, indexed by successive runs."""

    def __init__(self, store: DocStatusPort | None = None) -> None:
        self.config = Config()
        self.graph = FakeGraphStore()
        self.vector = FakeVectorStore(opensearch_config=self.config.indexing.opensearch)
        self.manager = self._manager()
        self.store: DocStatusPort = store or FakeDocStatusStore()
        self.extracted: list[str] = []

    def _manager(self) -> IndexingManager:
        return IndexingManager(
            config=self.config, vector_indexer=self.vector, graph_indexer=self.graph
        )

    def run(self, corpus: Corpus, interrupt: str | None = None, k: int = 1) -> bool:
        """Index ``corpus``; return whether the ``interrupt`` point fired.

        ``k`` picks which call of a counted point (the k-th store write of
        the removal or the delta, the k-th registry record or delete) fires.
        """
        # A run is a new process: nothing an interrupted one held in memory
        # survives it.
        self.manager = self._manager()
        inc = IncrementalIndexer(self.store, self.manager)
        restore: list[Callable[[], None]] = []
        fired = False
        try:
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
        delta, fingerprints = detect_delta(documents, inc.doc_status)
        to_extract = set(delta.to_process)
        extracted = [d for d in documents if document_doc_id(d) in to_extract]
        self.extracted = sorted(d.file_path for d in extracted)
        built = extract({d.file_path: corpus[d.file_path] for d in extracted})
        # The extraction stages store the units' records before indexing.
        self.vector.write_extraction_records(
            {f"g-{r.text_unit_id}": record_payload(r) for r in built.records}
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
        elif point == "mid_index_delta":
            for store in (self.graph, self.vector):
                for name in dir(store):
                    if name.startswith("upsert_"):
                        patch(store, name, on_kth)
        elif point == "before_record":
            patch(inc, "record", kill)
        elif point == "mid_commit":
            # The k-th record of the commit (after the write-ahead ones).
            def in_commit(original: Any) -> Any:
                def wrapper(record: Any) -> Any:
                    if record.status is not DocStatus.PENDING:
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
        return {
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
        records = self.store.list_all()
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
