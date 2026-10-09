# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs over in-memory stores, interruptible at any write step.

Shared by the write-ahead crash-recovery tests. A document is a list of text
units; a text unit is a set of weighted edges between a few synthetic entity
names, so "extraction" is deterministic and a fresh full build of any corpus
is easy to compute (:func:`expected_state`).

:meth:`IncrementalRun.run` wires the steps like the indexing stage does
(plan, write-ahead, joint removal, commit) and can be interrupted at one of
:data:`POINTS` by a :class:`Killed` ``BaseException``: like a process that
dies, it is not swallowed by the indexing manager's ``except Exception``
handlers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
    sum_weights,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    Document,
    Entity,
    Relationship,
    TextUnit,
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
        file_type="txt",
        total_pages=1,
    )


def extract(
    corpus: Corpus,
) -> tuple[list[TextUnit], list[Entity], list[Relationship]]:
    """Text units, entities and edges a build over ``corpus`` produces."""
    units: list[TextUnit] = []
    entity_units: dict[str, list[str]] = {}
    edge_weights: dict[tuple[str, str], list[dict[str, float]]] = {}
    for path, content in corpus.items():
        for index, unit in enumerate(content):
            unit_id = _unit_id(path, index, unit)
            units.append(TextUnit(id=unit_id, text="...", document_ids=[path]))
            for source, target, strength in unit:
                for name in (source, target):
                    cited = entity_units.setdefault(name, [])
                    if unit_id not in cited:
                        cited.append(unit_id)
                edge_weights.setdefault((source, target), []).append(
                    {unit_id: float(strength)}
                )
    entities = [
        Entity(
            id=BaseProcessor._generate_entity_id(name),
            name=name,
            type="ORG",
            text_unit_ids=cited,
        )
        for name, cited in entity_units.items()
    ]
    edges = []
    for (source, target), weights in edge_weights.items():
        edge = Relationship(
            id=BaseProcessor._generate_relationship_id(source, target, _TYPE),
            source_id=BaseProcessor._generate_entity_id(source),
            target_id=BaseProcessor._generate_entity_id(target),
            type=_TYPE,
        )
        apply_text_unit_weights(edge, sum_weights(weights))
        edges.append(edge)
    return units, entities, edges


def largest_lineage(*corpora: Corpus) -> int:
    """The most artifact ids one document of ``corpora`` produces: a registry
    limit of that many ids fits every committed record, while a changed
    document's write-ahead record (old + new ids) may not."""
    largest = 1
    for corpus in corpora:
        units, entities, edges = extract(corpus)
        documents = [document(path, content) for path, content in corpus.items()]
        for lineage in build_document_lineage(
            documents=documents,
            text_units=units,
            entities=entities,
            relationships=edges,
            communities=[],
            claims=[],
        ):
            largest = max(
                largest,
                len(lineage.text_unit_ids)
                + len(lineage.entity_ids)
                + len(lineage.relationship_ids),
            )
    return largest


def expected_state(corpus: Corpus) -> dict[str, Any]:
    """What a fresh full build of ``corpus`` stores (see :meth:`state`)."""
    units, entities, edges = extract(corpus)
    entity_units = {e.id: frozenset(e.text_unit_ids or []) for e in entities}
    return {
        "graph_entities": entity_units,
        "graph_edges": {
            r.id: (frozenset(r.text_unit_ids or []), round(r.weight or 0.0, 9))
            for r in edges
        },
        "vector_entities": entity_units,
        "vector_relationships": {r.id for r in edges},
        "vector_text_units": {u.id for u in units},
    }


class IncrementalRun:
    """One registry + graph + vector store, indexed by successive runs."""

    def __init__(self, store: DocStatusPort | None = None) -> None:
        self.config = Config()
        self.graph = FakeGraphStore()
        self.vector = FakeVectorStore(opensearch_config=self.config.indexing.opensearch)
        self.manager = IndexingManager(
            config=self.config, vector_indexer=self.vector, graph_indexer=self.graph
        )
        self.store: DocStatusPort = store or FakeDocStatusStore()
        self.extracted: list[str] = []

    def run(self, corpus: Corpus, interrupt: str | None = None, k: int = 1) -> bool:
        """Index ``corpus``; return whether the ``interrupt`` point fired.

        ``k`` picks which call of a counted point (the k-th store write of
        the removal or the delta, the k-th registry record or delete) fires.
        """
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
        delta, fingerprints = inc.plan(documents)
        to_extract = set(delta.to_process)
        extracted = [d for d in documents if document_doc_id(d) in to_extract]
        self.extracted = sorted(d.file_path for d in extracted)
        units, entities, edges = extract(
            {d.file_path: corpus[d.file_path] for d in extracted}
        )
        lineages = build_document_lineage(
            documents=extracted,
            text_units=units,
            entities=entities,
            relationships=edges,
            communities=[],
            claims=[],
        )
        inc.write_ahead(delta, lineages)
        if not inc.remove_changed_and_deleted(delta):
            raise AssertionError("the fake stores never fail a removal")
        inc.commit(
            lineages=lineages,
            fingerprints=fingerprints,
            text_units=units,
            entities=entities,
            relationships=edges,
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
        }

    def registry_problems(self, corpus: Corpus) -> list[str]:
        """Registry inconsistencies: rows not PROCESSED or not in the corpus,
        lineage pointing at missing artifacts, artifacts no lineage lists."""
        records = self.store.list_all()
        problems = []
        # Write-ahead lineage overflow never outlives its document's record.
        # (Overflow next to a committed record, left when a commit died
        # before deleting it, is ignored and dropped with the next write-ahead
        # or removal of the document.)
        overflow = set(getattr(self.store, "overflow", {}))
        orphans = overflow - {r.doc_id for r in records}
        if orphans:
            problems.append(f"lineage overflow without a record: {sorted(orphans)}")
        paths = sorted(r.file_path or r.doc_id for r in records)
        if paths != sorted(corpus):
            problems.append(f"registry rows {paths} != corpus {sorted(corpus)}")
        problems += [
            f"{r.file_path} is {r.status.value}"
            for r in records
            if r.status is not DocStatus.PROCESSED
        ]
        live = {
            "text_unit_ids": self.vector.ids("text_units"),
            "entity_ids": self.graph.ids("entities"),
            "relationship_ids": self.graph.ids("relationships"),
        }
        for field, ids in live.items():
            listed = {i for r in records for i in getattr(r, field)}
            if listed - ids:
                problems.append(f"lineage {field} not in the stores: {listed - ids}")
            if ids - listed:
                problems.append(f"{field} no lineage lists: {ids - listed}")
        return problems
