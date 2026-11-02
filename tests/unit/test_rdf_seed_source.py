# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kind: rdf`` seed graphs: an RDF A-Box as a ``SeedGraph``.

The registration and its extra (and the missing-extra errors, which run
without rdflib), the pure A-Box mapping on hand-built triples, then, with
the ontology extra, the reader on the synthetic procurement A-Box
(``tests/fixtures/seeds/procure-abox.*``) against the owl schema of
``tests/fixtures/ontologies/procure.ttl``: every serialisation gives one
graph; types and predicates map by IRI; the safety refusals of the owl
source apply; S3 through moto. No test reaches the network.
"""

from __future__ import annotations

import logging
import random
import shutil
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest

from tests.fixtures.extras import requires_ontology
from unified_kg_rag.adapters.schema_sources import INSTALL_HINT
from unified_kg_rag.adapters.seed_sources import (
    SEED_TERMS,
    SeedSourceContext,
    get_seed_source,
    registered_seed_sources,
    seed_source_requirement,
)
from unified_kg_rag.adapters.seed_sources.abox_mapping import (
    AboxMappingOptions,
    RdfTerm,
    Triple,
    iri,
    literal,
    literal_value,
    map_abox,
)
from unified_kg_rag.adapters.seed_sources.rdf_reader import RdfSeedOptions
from unified_kg_rag.application.bootstrap import (
    load_seed_graphs,
    resolve_extraction_schema,
)
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import SeedGraphConfig
from unified_kg_rag.domain.models.schema import (
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipTypeSpec,
)
from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedGraph, SeedNode
from unified_kg_rag.shared.exceptions import ConfigurationError, SeedGraphError

pytestmark = pytest.mark.unit

EX = "https://example.org/procure#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
FIXTURES = Path(__file__).parents[1] / "fixtures"
SEEDS = FIXTURES / "seeds"
ONTOLOGY = FIXTURES / "ontologies" / "procure.ttl"
ABOX = SEEDS / "procure-abox.ttl"
KOREAN = SEEDS / "procure-abox-ko.ttl"
SERIALISATIONS = ["procure-abox.ttl", "procure-abox.jsonld", "procure-abox.rdf"]
PREFIXES = (
    "@prefix ex: <https://example.org/procure#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
)
REGION = "us-east-1"
BUCKET = "seed-bucket"

# What procure-abox.* states, against the procure ontology's schema: types
# and predicates the schema declares carry their IRI alone (the conversion
# maps it); the others their UPPER_SNAKE label too.
PROCURE_ABOX = SeedGraph(
    nodes=(
        SeedNode(
            id=EX + "Initech%20Corp",
            iri=EX + "Initech%20Corp",
            name="Initech Corp",
            type="DISTRIBUTOR",
            type_iri=EX + "Distributor",
        ),
        SeedNode(
            id=EX + "acme",
            iri=EX + "acme",
            name="Acme Supplies",
            type_iri=EX + "Vendor",
            description="Industrial valve vendor.",
            properties={
                "ex:certified": True,
                "ex:country": "DE",
                "ex:creditLimit": 1500.5,
                "ex:rating": 4,
            },
        ),
        SeedNode(
            id=EX + "globex",
            iri=EX + "globex",
            name="Globex Buyer",
            type_iri=EX + "Buyer",
        ),
        SeedNode(
            id=EX + "jane", iri=EX + "jane", name="Jane Roe", type_iri=EX + "Person"
        ),
    ),
    edges=(
        SeedEdge(
            source=EX + "acme",
            target=EX + "Initech%20Corp",
            type="PARTNER_OF",
            predicate_iri=EX + "partnerOf",
        ),
        SeedEdge(
            source=EX + "acme", target=EX + "globex", predicate_iri=EX + "supplies"
        ),
        SeedEdge(
            source=EX + "globex", target=EX + "jane", predicate_iri=EX + "employs"
        ),
    ),
)
SKIPPED = {
    "blank_node_objects": 1,
    "blank_nodes": 1,
    "objects_not_instances": 1,
    "owl_same_as": 1,
    "reified_statements": 1,
    "subjects_not_instances": 1,
    "types_not_kept": 1,
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Fail any attempt to open a URL or a socket, and check none was made
    (moto serves S3 in process)."""
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(args[:2])
        raise OSError("network access in a test")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield attempts
    assert attempts == [], "the rdf seed source reached for the network"


def _seed(path: Path | str, **settings: Any) -> SeedGraphConfig:
    return SeedGraphConfig(name="procure", kind="rdf", path=str(path), **settings)


def _config(*seeds: SeedGraphConfig, **extraction: Any) -> Config:
    return Config.model_validate(
        {
            "aws": {"region_name": REGION},
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "owl", "path": str(ONTOLOGY)},
                    **extraction,
                },
                "seed_graphs": [s.model_dump() for s in seeds],
            },
        }
    )


_RESOLVED: dict[str, Config] = {}


def _resolved() -> Config:
    if "procure" not in _RESOLVED:
        _RESOLVED["procure"] = resolve_extraction_schema(_config())
    return _RESOLVED["procure"]


def _load(path: Path | str, **options: Any) -> SeedGraph:
    seed = _seed(path, options=options)
    return get_seed_source("rdf")(
        SeedSourceContext(
            seed, _resolved(), session_factory=lambda: boto3.Session(region_name=REGION)
        )
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- registration and the extra -------------------------------------------------------


def test_rdf_is_registered_with_the_ontology_extra() -> None:
    assert "rdf" in registered_seed_sources()
    assert seed_source_requirement("rdf") == ("rdflib", INSTALL_HINT)
    assert "uv sync --extra ontology" in INSTALL_HINT


def test_the_options_are_the_mapping_options_and_the_format() -> None:
    reader = set(RdfSeedOptions.model_fields) - set(AboxMappingOptions.model_fields)
    assert reader == {"format"}
    assert {"dangling", "qualifier_property"} <= set(AboxMappingOptions.model_fields)
    assert RdfSeedOptions.model_config.get("extra") == "forbid"


def test_preflight_names_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rdflib", None)
    config = Config.model_validate(
        {"processing": {"seed_graphs": [_seed("seed.ttl").model_dump()]}}
    )
    error = missing_extras_error(config) or ""
    assert "'rdflib' (for processing.seed_graphs 'procure' kind 'rdf')" in error
    assert "uv sync --extra ontology" in error


def test_without_rdflib_the_loader_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unified_kg_rag.adapters.source_files as package

    # As on an install without the extra: the parser module never imported.
    monkeypatch.setitem(sys.modules, "rdflib", None)
    monkeypatch.delattr(package, "rdf_parse", raising=False)
    monkeypatch.delitem(sys.modules, f"{package.__name__}.rdf_parse", raising=False)
    config = Config.model_validate(
        {"processing": {"seed_graphs": [_seed(ABOX).model_dump()]}}
    )
    with pytest.raises(SeedGraphError, match="kind 'rdf' needs rdflib") as error:
        load_seed_graphs(config)
    assert str(error.value).startswith("seed graph 'procure': ")
    assert "uv sync --extra ontology" in str(error.value)


def test_the_mapping_and_the_registry_import_without_rdflib() -> None:
    code = (
        "import sys\n"
        "sys.modules['rdflib'] = None\n"
        "import unified_kg_rag.adapters.seed_sources as s\n"
        "assert 'rdf' in s.registered_seed_sources()\n"
        "from unified_kg_rag.adapters.seed_sources import abox_mapping as m\n"
        "from unified_kg_rag.domain.models.schema import ExtractionSchema\n"
        "t = [(m.iri('urn:a'), m.iri(m.RDF_TYPE), m.iri('urn:C'))]\n"
        "g = m.map_abox(t, m.AboxMappingOptions(), ExtractionSchema(),\n"
        "               terms=s.SEED_TERMS).graph\n"
        "assert [n.type for n in g.nodes] == ['C']\n"
        "import unified_kg_rag.application.cli.preflight\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- the mapping, on hand-built triples (no rdflib) -------------------------------------

ORG, VENDOR, BUYER = EX + "Organization", EX + "Vendor", EX + "Buyer"
SCHEMA = ExtractionSchema(
    entity_types=(
        EntityTypeSpec(label="ORGANIZATION", iri=ORG),
        EntityTypeSpec(label="BUYER", iri=BUYER),
        EntityTypeSpec(label="VENDOR", iri=VENDOR),
    ),
    relationship_types=(RelationshipTypeSpec(label="SUPPLIES", iri=EX + "supplies"),),
)


def _typed(subject: str, *types: str) -> list[Triple]:
    return [(iri(subject), iri(RDF + "type"), iri(t)) for t in types]


def _map(triples: list[Triple], **options: Any) -> SeedGraph:
    return map_abox(
        triples, AboxMappingOptions(**options), SCHEMA, terms=SEED_TERMS
    ).graph


def test_the_most_specific_mapped_type_wins_whatever_the_order() -> None:
    for types in ((ORG, VENDOR), (VENDOR, ORG), (VENDOR, ORG, EX + "Supplier")):
        (node,) = _map(_typed(EX + "acme", *types)).nodes
        assert (node.type, node.type_iri) == (None, VENDOR)


def test_an_unmapped_type_is_upper_snaked_or_labelled() -> None:
    triples = _typed(EX + "acme", EX + "preferredSupplier", EX + "zz")
    (node,) = _map(triples).nodes
    assert (node.type, node.type_iri) == (
        "PREFERRED_SUPPLIER",
        EX + "preferredSupplier",
    )
    (node,) = _map(
        triples, labels={"ex:preferredSupplier": "VENDOR"}, prefixes={"ex": EX}
    ).nodes
    assert node.type == "VENDOR"


def test_only_vocabulary_types_make_no_instance() -> None:
    triples = [
        *_typed(EX + "concept", SKOS + "Concept"),
        *_typed(EX + "item", OWL + "NamedIndividual"),
        *_typed(EX + "Vendor", OWL + "Class", EX + "MetaClass"),
        *_typed(EX + "Supplier", EX + "MetaClass"),
        (iri(EX + "Supplier"), iri(RDFS + "subClassOf"), iri(EX + "Party")),
        *_typed(EX + "ships", EX + "MetaProperty"),
        (iri(EX + "ships"), iri(RDFS + "domain"), iri(EX + "Party")),
        *_typed(EX + "acme", EX + "Vendor"),
    ]
    graph = _map(triples)
    # T-Box terms, whatever their types: ex:Vendor is a class (typed
    # owl:Class, and an rdf:type object), ex:Supplier one (a subClassOf
    # end), ex:ships a property (a domain subject).
    assert [n.id for n in graph.nodes] == [EX + "acme"]
    assert graph.report.skipped == {"subjects_not_instances": 2}


def test_edges_join_instances_only_and_are_typed_by_predicate() -> None:
    triples = [
        *_typed(EX + "acme", VENDOR),
        *_typed(EX + "globex", BUYER),
        (iri(EX + "acme"), iri(EX + "supplies"), iri(EX + "globex")),
        (iri(EX + "acme"), iri(EX + "knowsOf"), iri(EX + "globex")),
        (iri(EX + "acme"), iri(EX + "locatedIn"), iri(EX + "berlin")),
        (iri(EX + "acme"), iri(RDFS + "seeAlso"), iri(EX + "globex")),
        (iri(EX + "acme"), iri(OWL + "sameAs"), iri(EX + "globex")),
        (iri(EX + "acme"), iri(EX + "contact"), RdfTerm("blank", "b0")),
    ]
    graph = _map(triples)
    assert [(e.target, e.type, e.predicate_iri) for e in graph.edges] == [
        (EX + "globex", "KNOWS_OF", EX + "knowsOf"),
        (EX + "globex", None, EX + "supplies"),
    ]
    assert graph.report.skipped == {
        "blank_node_objects": 1,
        "objects_not_instances": 1,
        "owl_same_as": 1,
        "vocabulary_statements": 1,
    }


def test_no_inference_adds_an_edge() -> None:
    triples = [
        *_typed(EX + "acme", VENDOR),
        *_typed(EX + "globex", BUYER),
        (iri(EX + "acme"), iri(EX + "supplies"), iri(EX + "globex")),
        (iri(EX + "suppliedBy"), iri(OWL + "inverseOf"), iri(EX + "supplies")),
        (iri(EX + "supplies"), iri(RDF + "type"), iri(OWL + "SymmetricProperty")),
    ]
    graph = _map(triples)
    assert [(e.source, e.target) for e in graph.edges] == [(EX + "acme", EX + "globex")]
    assert [n.id for n in graph.nodes] == [EX + "acme", EX + "globex"]


def test_blank_nodes_reification_and_rdf_star_are_counted() -> None:
    triples = [
        (RdfTerm("blank", "b1"), iri(RDF + "type"), iri(VENDOR)),
        (RdfTerm("blank", "b2"), iri(RDF + "type"), iri(RDF + "Statement")),
        (RdfTerm("blank", "b2"), iri(RDF + "subject"), iri(EX + "acme")),
        (iri(EX + "claim"), iri(RDF + "type"), iri(RDF + "Statement")),
        (iri(EX + "claim"), iri(RDF + "type"), iri(EX + "Claim")),
        (RdfTerm("triple"), iri(EX + "saidBy"), iri(EX + "acme")),
        *_typed(EX + "acme", VENDOR),
    ]
    graph = _map(triples)
    assert [n.id for n in graph.nodes] == [EX + "acme"]
    assert graph.report.skipped == {
        "blank_nodes": 1,
        "reified_statements": 2,
        "rdf_star_triples": 1,
    }


def test_names_descriptions_and_properties_follow_the_options() -> None:
    acme = iri(EX + "acme")
    triples = [
        *_typed(EX + "acme", VENDOR),
        (acme, iri(RDFS + "label"), literal("Acme", language="en")),
        (acme, iri(RDFS + "label"), literal("에이스", language="ko")),
        (acme, iri(SKOS + "prefLabel"), literal("Acme Supplies", language="en-GB")),
        (acme, iri(RDFS + "comment"), literal("Vendor.", language="en")),
        (acme, iri(EX + "region"), literal("Europe", language="en")),
        (acme, iri(EX + "region"), literal("유럽", language="ko")),
        (acme, iri(EX + "code"), literal("7", XSD + "int")),
        (acme, iri(EX + "code"), literal("3", XSD + "int")),
        (acme, iri(EX + "secret"), literal("x")),
    ]
    (node,) = _map(triples, prefixes={"ex": EX}, exclude_properties=["ex:secret"]).nodes
    # "en" before its subtag "en-GB", whatever the property order.
    assert (node.name, node.description) == ("Acme", "Vendor.")
    assert node.properties == {"ex:code": 3, "ex:region": "Europe"}
    (node,) = _map(
        triples, languages=["ko", ""], properties=[EX + "region"], prefixes={"ex": EX}
    ).nodes
    assert node.name == "에이스"
    assert node.properties == {"ex:region": "유럽"}
    graph = _map(triples, name_properties=["ex:nothing"], prefixes={"ex": EX})
    assert graph.nodes[0].name == "acme"


def test_a_name_falls_back_to_the_decoded_local_name_as_written() -> None:
    (node,) = _map(_typed(EX + "Acme%20Supplies_GmbH", VENDOR)).nodes
    assert node.name == "Acme Supplies_GmbH"
    (node,) = _map(_typed(EX + "%EA%B3%B5%EA%B8%89", VENDOR)).nodes
    assert node.name == "공급"


@pytest.mark.parametrize(
    ("term", "value"),
    [
        (literal("42", XSD + "integer"), 42),
        (literal(" -7 ", XSD + "nonPositiveInteger"), -7),
        (literal("4.2e1", XSD + "integer"), "4.2e1"),
        (literal("1.5", XSD + "decimal"), 1.5),
        (literal("1.5E0", XSD + "double"), 1.5),
        (literal("INF", XSD + "double"), "INF"),
        (literal("NaN", XSD + "float"), "NaN"),
        (literal("x", XSD + "float"), "x"),
        (literal("true", XSD + "boolean"), True),
        (literal("0", XSD + "boolean"), False),
        (literal("yes", XSD + "boolean"), "yes"),
        (literal("2026-01-01", XSD + "date"), "2026-01-01"),
        (literal("DE"), "DE"),
    ],
)
def test_typed_literals_become_scalars(term: RdfTerm, value: Any) -> None:
    converted = literal_value(term)
    assert converted == value and type(converted) is type(value)


def test_an_unbound_curie_in_the_options_is_refused() -> None:
    with pytest.raises(SeedGraphError, match=r"options\.properties: 'zz:x' is neither"):
        map_abox(
            [],
            AboxMappingOptions(properties=["zz:x"]),
            SCHEMA,
            terms=SEED_TERMS.__class__(seed="procure", kind="rdf"),
        )


def test_property_keys_use_the_options_then_well_known_then_file_prefixes() -> None:
    acme = iri(EX + "acme")
    triples = [
        *_typed(EX + "acme", VENDOR),
        (acme, iri(EX + "country"), literal("DE")),
        (acme, iri(SKOS + "note"), literal("n")),
        (acme, iri("urn:x:flag"), literal("true", XSD + "boolean")),
    ]
    options = AboxMappingOptions()
    keys = (
        map_abox(triples, options, SCHEMA, terms=SEED_TERMS).graph.nodes[0].properties
    )
    assert set(keys) == {EX + "country", "skos:note", "urn:x:flag"}
    graph = map_abox(
        triples, options, SCHEMA, prefixes={"p": EX, "a": EX}, terms=SEED_TERMS
    ).graph
    assert set(graph.nodes[0].properties) == {"a:country", "skos:note", "urn:x:flag"}
    graph = _map(triples, prefixes={"proc": EX})
    assert "proc:country" in graph.nodes[0].properties


def test_the_order_of_the_triples_changes_nothing() -> None:
    acme, globex = iri(EX + "acme"), iri(EX + "globex")
    triples = [
        *_typed(EX + "acme", ORG, VENDOR),
        *_typed(EX + "globex", BUYER, EX + "Customer"),
        (acme, iri(EX + "supplies"), globex),
        (globex, iri(EX + "paysTo"), acme),
        (acme, iri(RDFS + "label"), literal("Acme", language="en")),
        (acme, iri(RDFS + "label"), literal("ACME", language="")),
        (acme, iri(EX + "code"), literal("7", XSD + "int")),
        (acme, iri(EX + "code"), literal("3", XSD + "int")),
    ]
    first = _map(triples)
    generator = random.Random(7)
    for _ in range(5):
        generator.shuffle(triples)
        assert _map(triples) == first


def test_unfollowed_imports_and_relative_iris_are_reported() -> None:
    base = "file:///data/seed.ttl#"
    triples = [
        (iri("urn:onto"), iri(RDF + "type"), iri(OWL + "Ontology")),
        (iri("urn:onto"), iri(OWL + "imports"), iri("urn:other")),
        (iri("urn:onto"), iri(OWL + "imports"), iri("urn:v1")),
        (iri("urn:onto:1"), iri(OWL + "versionIRI"), iri("urn:v1")),
        *_typed(base + "acme", VENDOR),
        *_typed("s3://bucket/seed.ttl#globex", BUYER),
    ]
    mapping = map_abox(triples, AboxMappingOptions(), SCHEMA, terms=SEED_TERMS)
    assert mapping.unfollowed_imports == ["urn:other"]
    assert mapping.relative_iris == [base + "acme", "s3://bucket/seed.ttl#globex"]


# --- the reader, on the procurement A-Box -------------------------------------------------


@requires_ontology
@pytest.mark.parametrize("name", SERIALISATIONS)
def test_every_serialisation_gives_the_golden_graph(name: str) -> None:
    graph = _load(SEEDS / name)
    assert (graph.nodes, graph.edges) == (PROCURE_ABOX.nodes, PROCURE_ABOX.edges)
    assert graph.report.skipped == SKIPPED
    assert graph.report.files == 1
    assert graph.report.bytes_read == (SEEDS / name).stat().st_size


@requires_ontology
def test_n_triples_bind_no_prefix_so_the_options_name_the_keys() -> None:
    nt = SEEDS / "procure-abox.nt"
    plain = _load(nt)
    assert set(plain.nodes[1].properties) == {
        EX + key for key in ("certified", "country", "creditLimit", "rating")
    }
    for name in [*SERIALISATIONS, "procure-abox.nt"]:
        graph = _load(SEEDS / name, prefixes={"ex": EX})
        assert (graph.nodes, graph.edges) == (PROCURE_ABOX.nodes, PROCURE_ABOX.edges)


@requires_ontology
@pytest.mark.parametrize("mode", ["off", "align", "strict"])
def test_the_graph_converts_against_the_owl_schema(mode: str) -> None:
    config = resolve_extraction_schema(
        _config(_seed(ABOX, options={"prefixes": {"ex": EX}}), schema_mode=mode)
    )
    (conversion,) = load_seed_graphs(config)
    statements = [
        line
        for shard in conversion.documents_content
        for line in shard.page_content.split("\n")
    ]
    text = "\n".join(statements)
    assert '"type":"VENDOR"' in text and '"type":"PERSON"' in text
    assert '"type":"SUPPLIES"' in text and '"type":"EMPLOYS"' in text
    assert conversion.stats.nodes_unmapped_type_iri == 1
    assert conversion.stats.edges_unmapped_predicate_iri == 1
    if mode == "strict":
        # What no schema type declares is dropped, with its edges.
        assert (conversion.stats.nodes, conversion.stats.edges) == (3, 2)
        assert "DISTRIBUTOR" not in text and "PARTNER_OF" not in text
    else:
        assert (conversion.stats.nodes, conversion.stats.edges) == (4, 3)
        assert '"type":"DISTRIBUTOR"' in text and '"type":"PARTNER_OF"' in text


@requires_ontology
def test_an_unresolved_schema_source_is_refused() -> None:
    context = SeedSourceContext(_seed(ABOX), _config())
    with pytest.raises(ConfigurationError, match="is not loaded yet"):
        get_seed_source("rdf")(context)


@requires_ontology
def test_korean_labels_follow_the_languages() -> None:
    korean = _load(KOREAN, languages=["ko", "en", ""], prefixes={"ex": EX})
    by_id = {n.id.removeprefix(EX): n for n in korean.nodes}
    assert {k: n.name for k, n in by_id.items()} == {
        "acme": "에이스 공급",
        "globex": "글로벡스 구매자",
        "%EB%8C%80%EB%A6%AC%EC%A0%90": "대리점",
    }
    assert by_id["acme"].description == "산업용 밸브 공급업체."
    assert by_id["acme"].properties == {"ex:region": "유럽"}
    assert {(e.type, e.predicate_iri) for e in korean.edges} == {
        (None, EX + "supplies"),
        (None, EX + "tradesWith"),
    }
    english = _load(KOREAN, prefixes={"ex": EX})
    by_id = {n.id.removeprefix(EX): n for n in english.nodes}
    assert by_id["acme"].name == "Acme Supplies"
    assert by_id["acme"].properties == {"ex:region": "Europe"}
    # No English label: the untagged ones would come first, then any tag.
    assert by_id["globex"].name == "글로벡스 구매자"


@requires_ontology
def test_a_directory_is_read_as_one_graph(tmp_path: Path) -> None:
    directory = tmp_path / "seed"
    lines = ABOX.read_text(encoding="utf-8").split("\n# A class no schema")
    prefixes = lines[0].split("\n# Typed")[0]
    _write(directory / "a.ttl", lines[0])
    _write(directory / "sub" / "b.ttl", prefixes + "\n# A class no schema" + lines[1])
    _write(directory / ".hidden" / "c.ttl", "not turtle")
    _write(directory / "notes.txt", "not turtle")
    graph = _load(directory)
    assert (graph.nodes, graph.edges) == (PROCURE_ABOX.nodes, PROCURE_ABOX.edges)
    assert graph.report.files == 2


@requires_ontology
def test_an_xml_catalog_is_skipped_in_a_directory_and_refused_alone(
    tmp_path: Path,
) -> None:
    catalog = (
        '<?xml version="1.0"?>\n<catalog '
        'xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>\n'
    )
    shutil.copy(ABOX, tmp_path / "abox.ttl")
    _write(tmp_path / "catalog-v001.xml", catalog)
    assert _load(tmp_path).nodes == PROCURE_ABOX.nodes
    with pytest.raises(SeedGraphError, match="is an OASIS XML catalog"):
        _load(tmp_path / "catalog-v001.xml")


@requires_ontology
def test_unfollowed_imports_are_warned_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(
        tmp_path / "abox.ttl",
        PREFIXES
        + "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        + "<https://example.org/data> a owl:Ontology ;\n"
        + "    owl:imports <https://example.org/procure>, <https://example.org/money> .\n"
        + "ex:acme a ex:Vendor .\n",
    )
    with caplog.at_level(logging.WARNING):
        graph = _load(path)
    assert [n.id for n in graph.nodes] == [EX + "acme"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "Seed graph 'procure': owl:imports not followed: https://example.org/money, "
        "https://example.org/procure; a seed states the triples of its own files only"
    ]


@requires_ontology
def test_relative_iris_are_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path / "abox.ttl", PREFIXES + "<#acme> a ex:Vendor .\n")
    with caplog.at_level(logging.WARNING):
        (node,) = _load(path).nodes
    assert node.id == path.resolve().as_uri() + "#acme"
    assert any(
        "1 instances have IRIs relative to their file" in r.getMessage()
        for r in caplog.records
    )


@requires_ontology
def test_the_counts_are_logged_without_the_seed_s_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        _load(ABOX)
    (line,) = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Seed graph")
    ]
    assert line.startswith(
        "Seed graph 'procure' (rdf, 28 triples): 4 instances, 3 edges"
    )
    assert "Acme" not in line and "example.org" not in line


# --- refusals (the owl source's checks) ----------------------------------------------------


@requires_ontology
def test_a_json_ld_context_naming_a_document_is_refused(
    no_network: list[Any],
) -> None:
    with pytest.raises(SeedGraphError, match="remote JSON-LD context") as error:
        _load(FIXTURES / "ontologies" / "remote-context.jsonld")
    assert str(error.value).startswith("seed graph 'procure': ")
    assert no_network == []


@requires_ontology
def test_xml_external_entities_are_not_resolved(tmp_path: Path) -> None:
    _write(tmp_path / "secret.txt", "SENTINEL-SECRET")
    path = _write(
        tmp_path / "abox.rdf",
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE rdf:RDF [\n  <!ENTITY local SYSTEM "secret.txt">\n'
        '  <!ENTITY remote SYSTEM "https://example.org/secret.txt">\n]>\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
        '  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n'
        '  xmlns:ex="https://example.org/procure#">\n'
        '  <ex:Vendor rdf:about="https://example.org/procure#acme">\n'
        '    <rdfs:label xml:lang="en">Acme&local;</rdfs:label>\n'
        '    <rdfs:comment xml:lang="en">Sells valves.&remote;</rdfs:comment>\n'
        "  </ex:Vendor>\n</rdf:RDF>\n",
    )
    graph = _load(path)
    assert graph.model_dump_json().count("SENTINEL") == 0
    assert [(n.name, n.description) for n in graph.nodes] == [("Acme", "Sells valves.")]


@requires_ontology
@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("abox.n3", "", "is Notation3"),
        ("abox.owx", "", "is OWL/XML"),
        (
            "abox.owl",
            '<Ontology xmlns="http://www.w3.org/2002/07/owl#"/>',
            "is OWL/XML",
        ),
        ("abox.csv", "", "must end in .ttl"),
        (
            "abox.ttl",
            PREFIXES + "<< ex:a ex:p ex:b >> ex:q ex:c .\n",
            "does not parse as turtle",
        ),
        (
            "abox.nt",
            "<urn:a> <urn:p> " + '"' + "x" * (300 * 1024) + '" .\n',
            "MAX_LINE_BYTES",
        ),
    ],
)
def test_refused_files_are_named(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    path = _write(tmp_path / name, text)
    with pytest.raises(SeedGraphError, match=message) as error:
        _load(path)
    assert str(error.value).startswith("seed graph 'procure': ")


@requires_ontology
def test_more_than_max_bytes_is_refused_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy(ABOX, tmp_path / "abox.ttl")
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail(f"read {self} past the cap")
    )
    seed = _seed(tmp_path, max_bytes=100)
    with pytest.raises(SeedGraphError, match="more than .*max_bytes \\(100\\)"):
        get_seed_source("rdf")(SeedSourceContext(seed, _resolved()))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"path": "https://example.org/abox.ttl"}, "is a URL; kind 'rdf' reads"),
        ({"path": "s3://B_ad/abox.ttl"}, "does not name an S3 bucket"),
        (
            {
                "path": "abox.ttl",
                "secret_arn": "arn:aws:secretsmanager:us-east-1:000000000000:secret:x",
            },
            "kind 'rdf' takes no secret_arn",
        ),
        (
            {"path": "abox.ttl", "options": {"colour": 1}},
            "options.colour: Extra inputs",
        ),
        ({"path": "abox.ttl", "options": {"format": "n3"}}, "options.format: Input"),
        (
            {"path": "abox.ttl", "options": {"languages": ["x y"]}},
            "options.languages.0",
        ),
    ],
)
def test_bad_settings_are_refused(settings: dict[str, Any], message: str) -> None:
    seed = SeedGraphConfig(name="procure", kind="rdf", **settings)
    with pytest.raises(SeedGraphError, match=message.replace(".", r"\.")):
        get_seed_source("rdf")(SeedSourceContext(seed, _config()))


# --- S3 (moto) ---------------------------------------------------------------------------------


@pytest.fixture
def s3() -> Iterator[Any]:
    # Imported here: the core-install CI job runs this file without moto
    # (the dev group), and skips every S3 test (they need the extra too).
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3: Any, key: str, path: Path) -> str:
    s3.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())
    return f"s3://{BUCKET}/{key}"


@requires_ontology
def test_an_s3_object_and_prefix_read_as_the_files_on_disk(s3: Any) -> None:
    location = _put(s3, "seed/procure-abox.ttl", ABOX)
    graph = _load(location)
    assert (graph.nodes, graph.edges) == (PROCURE_ABOX.nodes, PROCURE_ABOX.edges)
    _put(s3, "prefix/abox.jsonld", SEEDS / "procure-abox.jsonld")
    _put(s3, "prefix/.hidden/abox.ttl", KOREAN)
    graph = _load(f"s3://{BUCKET}/prefix/")
    assert (graph.nodes, graph.edges) == (PROCURE_ABOX.nodes, PROCURE_ABOX.edges)
    assert graph.report.files == 1


@requires_ontology
def test_s3_sizes_are_checked_before_fetching(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _put(s3, "seed/procure-abox.ttl", ABOX)
    from botocore.client import BaseClient

    real = BaseClient._make_api_call

    def no_get(self: Any, operation: str, params: Any) -> Any:
        assert operation != "GetObject", "fetched an object past the cap"
        return real(self, operation, params)

    monkeypatch.setattr(BaseClient, "_make_api_call", no_get)
    seed = _seed(location, max_bytes=100)
    context = SeedSourceContext(
        seed, _resolved(), session_factory=lambda: boto3.Session(region_name=REGION)
    )
    with pytest.raises(SeedGraphError, match="max_bytes \\(100\\)"):
        get_seed_source("rdf")(context)


@requires_ontology
def test_a_missing_s3_object_names_the_seed(s3: Any) -> None:
    with pytest.raises(SeedGraphError, match="seed graph 'procure': .* not found"):
        _load(f"s3://{BUCKET}/seed/missing.ttl")
