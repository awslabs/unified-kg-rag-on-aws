# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An RDF A-Box seed under a strict owl schema, end to end on the embedded
stores (no container, no AWS; the harness of ``test_seed_graph_embedded``).

The schema is ``tests/fixtures/ontologies/procure.ttl``; the seed is
``tests/fixtures/seeds/procure-abox.ttl``. Strict mode keeps the instances
and edges whose classes and properties the ontology declares, under their
schema types and with their IRIs, and drops the rest (a class and a
property it does not declare); no seed statement reaches the model; a reset
re-imports the seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.extras import requires_ontology
from tests.integration import test_seed_graph_embedded as seeded
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.domain.models.config import SeedGraphConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("no_aws"),
    requires_ontology,
]

FIXTURES = Path(__file__).parents[1] / "fixtures"
EX = "https://example.org/procure#"
CORPUS = {"contract.txt": "Vendor supplies widgets to Buyer."}


def test_an_rdf_seed_under_a_strict_owl_schema(tmp_path: Path) -> None:
    harness = seeded._Harness(tmp_path, corpus=CORPUS)
    extraction = harness.config.processing.graph_extraction
    extraction.schema_mode = "strict"
    extraction.schema_source.kind = "owl"
    extraction.schema_source.path = str(FIXTURES / "ontologies" / "procure.ttl")
    harness.config.processing.seed_graphs = [
        SeedGraphConfig(
            name="procure",
            kind="rdf",
            path=str(FIXTURES / "seeds" / "procure-abox.ttl"),
            shards=2,
        )
    ]
    harness.ok()

    # The documents' entities (ORGANIZATION) and the seed's declared
    # instances; the undeclared Distributor is dropped.
    assert harness.graph_names() == {
        "Vendor",
        "Buyer",
        "Acme Supplies",
        "Globex Buyer",
        "Jane Roe",
    }
    entities = harness.entities()
    assert {
        name: (hit["type"].upper(), hit.get("iri"), hit.get("source_kinds"))
        for name, hit in entities.items()
        if hit.get("source_kinds")
    } == {
        "Acme Supplies": ("VENDOR", EX + "acme", ["seed_graph"]),
        "Globex Buyer": ("BUYER", EX + "globex", ["seed_graph"]),
        "Jane Roe": ("PERSON", EX + "jane", ["seed_graph"]),
    }
    acme = entities["Acme Supplies"]
    assert "Industrial valve vendor." in acme["description"]
    units = "\n".join(unit["text"] for unit in harness.seed_units())
    assert '"type":"SUPPLIES"' in units and '"type":"EMPLOYS"' in units
    assert "PARTNER_OF" not in units and "Initech" not in units
    # An RDF edge states no description: its vector copy keeps the type.
    prefix = harness.config.indexing.opensearch.relationships_index_prefix
    assert {
        (hit["source_name"], hit["target_name"]): hit.get("type", "").upper()
        for hit in harness.hits(prefix)
        if hit.get("source_kinds") == ["seed_graph"]
    } == {
        ("Acme Supplies", "Globex Buyer"): "SUPPLIES",
        ("Globex Buyer", "Jane Roe"): "EMPLOYS",
    }
    harness.assert_no_seed_text_prompted()

    # A reset re-imports the RDF seed: the namespace is rebuilt as it was.
    built = harness.state()
    registry = harness.registry()
    seed_paths = {path for path in registry if path.startswith("seed_graphs/procure/")}
    assert seed_paths and set(registry) == set(CORPUS) | seed_paths
    harness.config.indexing.reset = True
    context = harness.ok()
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.new) == len(registry)
    assert harness.state() == built
    harness.assert_no_seed_text_prompted()
