# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity resolution merges name variants by default and similar names opt-in.

Name similarity also joins different organizations sharing most of their name,
so the default merges only names with the same compact key (normalization,
legal form, dense-script spacing). The full build and the incremental merge
apply the same rule.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.models import Config, DocumentLineage, Entity

pytestmark = pytest.mark.unit

_DIFFERENT_ORGANIZATIONS = [
    ("Northfield University", "Northfield University Hospital"),
    ("University of Westland", "University of Westland Press"),
    ("Federal Reserve Bank", "Federal Reserve Board"),
]


def _groups(config: Config, names: list[str]) -> list[set[str]]:
    resolver = EntityResolver(config, max_workers=1, use_process_pool=False)
    entities = [
        Entity(id=f"e{i}", name=name, type="organization", text_unit_ids=[f"t{i}"])
        for i, name in enumerate(names)
    ]
    groups = resolver._group_similar_entities(entities)
    return sorted(({e.name for e in group} for group in groups), key=sorted)


@pytest.mark.parametrize("pair", _DIFFERENT_ORGANIZATIONS)
def test_default_keeps_organizations_sharing_most_of_their_name(pair) -> None:
    assert len(_groups(Config(), list(pair))) == 2


def test_default_merges_spelling_and_legal_form_variants() -> None:
    groups = _groups(Config(), ["Acme Inc", "ACME", "acme, inc.", "Other Co"])
    assert {"Acme Inc", "ACME", "acme, inc."} in groups
    assert {"Other Co"} in groups


def test_fuzzy_resolution_is_opt_in() -> None:
    config = Config()
    assert len(_groups(config, ["Acme Corporation", "Acme Corporatio"])) == 2
    config.processing.fuzzy_entity_resolution = True
    assert len(_groups(config, ["Acme Corporation", "Acme Corporatio"])) == 1


def _incremental(config: Config) -> tuple[IncrementalIndexer, FakeGraphStore]:
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    return IncrementalIndexer(FakeDocStatusStore(), manager), graph


def _commit(incremental: IncrementalIndexer, doc_id: str, entity: Entity) -> None:
    incremental.commit(
        lineages=[DocumentLineage(doc_id=doc_id)],
        fingerprints={doc_id: f"hash-{doc_id}"},
        entities=[entity],
    )


@pytest.mark.parametrize(
    ("fuzzy", "stored", "delta", "merged"),
    [
        # A legal-form variant merges in a full build, so across runs too.
        (False, "Acme", "Acme Inc", True),
        (False, "Acme Corporation", "Acme Corporatio", False),
        (True, "Acme Corporation", "Acme Corporatio", True),
        (False, *_DIFFERENT_ORGANIZATIONS[0], False),
    ],
)
def test_incremental_merge_follows_the_full_build_rule(
    fuzzy: bool, stored: str, delta: str, merged: bool
) -> None:
    config = Config()
    config.processing.fuzzy_entity_resolution = fuzzy
    assert (len(_groups(config, [stored, delta])) == 1) is merged

    incremental, graph = _incremental(config)
    _commit(incremental, "doc-a", Entity(id="e-a", name=stored, text_unit_ids=["t1"]))
    _commit(incremental, "doc-b", Entity(id="e-b", name=delta, text_unit_ids=["t2"]))
    assert graph.ids("entities") == ({"e-a"} if merged else {"e-a", "e-b"})
