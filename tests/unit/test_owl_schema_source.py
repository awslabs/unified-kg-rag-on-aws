# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``schema_source.kind: owl``: reading an ontology into the schema.

Every serialisation of the procurement ontology maps to the golden schema
of the mapping's worked example; a directory is read as one ontology;
nothing is fetched (every test runs with the network patched to fail and
checks it was never used); each refused input fails with a
``SchemaSourceError`` saying what to do; and the cache keys follow the
mapped schema, not the files.
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

import pytest

from tests.fixtures.extras import requires_ontology
from tests.fixtures.ontology_models import EX, PROCURE_GOLDEN, procure_model
from tests.fixtures.schema_annotation import schema_annotation_digest
from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
    register_schema_source,
    registered_schema_sources,
)
from unified_kg_rag.adapters.schema_sources import registry as schema_registry
from unified_kg_rag.adapters.schema_sources.ontology_mapping import MappingOptions
from unified_kg_rag.adapters.schema_sources.owl import INSTALL_HINT, OwlSourceOptions
from unified_kg_rag.adapters.schema_sources.registry import schema_source_requirement
from unified_kg_rag.adapters.source_files.rdf_files import REFUSED_SUFFIXES
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.domain.ingestion.cache_keys import (
    extraction_fingerprint,
    stage_input_fingerprint,
)
from unified_kg_rag.domain.models import Config, PipelineStageType, SchemaDocument
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.shared.exceptions import SchemaSourceError

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ontologies"
PROCURE = FIXTURES / "procure.ttl"
SERIALISATIONS = ["procure.ttl", "procure.rdf", "procure.jsonld", "procure.nt"]
PREFIXES = """\
@prefix ex: <https://example.org/procure#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Fail any attempt to open a URL or a socket, and check none was made."""
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(args[:2])
        raise OSError("network access in a test")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield attempts
    assert attempts == [], "the owl source reached for the network"


def _source(path: Path | str, **options: Any) -> SchemaSourceConfig:
    return SchemaSourceConfig(kind="owl", path=str(path), options=options)


def _load(path: Path | str, **options: Any) -> SchemaDocument:
    return get_schema_source("owl")(
        SchemaSourceContext(_source(path, **options), Config())
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _config(path: Path, **extraction: Any) -> Config:
    return Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "owl", "path": str(path)},
                    **extraction,
                }
            }
        }
    )


def _keys(config: Config) -> dict[str, str]:
    return {
        stage.value: stage_input_fingerprint(config, stage, "corpus")
        for stage in PipelineStageType
    }


# --- registration and the extra -------------------------------------------------


def test_owl_is_registered_with_its_extra() -> None:
    assert "owl" in registered_schema_sources()
    assert schema_source_requirement("owl") == ("rdflib", INSTALL_HINT)
    assert "uv sync --inexact --extra ontology" in INSTALL_HINT
    assert schema_source_requirement("file") is None


def test_reregistering_a_kind_replaces_its_requirement() -> None:
    kind = "synthetic_rdf_service"
    try:
        register_schema_source(kind, requires=("synthetic_absent_module", "x"))(
            lambda context: SchemaDocument()
        )
        assert schema_source_requirement(kind) == ("synthetic_absent_module", "x")
        register_schema_source(kind)(lambda context: SchemaDocument())
        assert schema_source_requirement(kind) is None
    finally:
        schema_registry._REGISTRY.pop(kind, None)
        schema_registry._REQUIRES.pop(kind, None)


def test_the_options_are_the_mapping_options_and_the_reader_options() -> None:
    assert issubclass(OwlSourceOptions, MappingOptions)
    reader = set(OwlSourceOptions.model_fields) - set(MappingOptions.model_fields)
    assert reader == {"format", "max_bytes"}
    assert OwlSourceOptions.model_config.get("extra") == "forbid"


def test_preflight_names_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(PROCURE)
    monkeypatch.setitem(sys.modules, "rdflib", None)
    error = missing_extras_error(config) or ""
    assert (
        "'rdflib' (for processing.graph_extraction.schema_source.kind 'owl')" in error
    )
    assert "uv sync --inexact --extra ontology" in error


def test_preflight_checks_only_the_selected_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdflib", None)
    assert missing_extras_error(Config()) is None


@requires_ontology
def test_preflight_passes_with_the_extra() -> None:
    assert missing_extras_error(_config(PROCURE)) is None


def test_ingestion_exits_2_naming_the_extra(
    mocker, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    from unified_kg_rag.application.cli import run_ingestion_pipeline

    monkeypatch.setitem(sys.modules, "rdflib", None)
    mocker.patch.object(
        run_ingestion_pipeline, "load_cli_config", return_value=_config(PROCURE)
    )
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    pipeline = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    (tmp_path / "src").mkdir()
    parser = run_ingestion_pipeline.CommandLineInterface._setup_arguments()
    args = parser.parse_args(
        [
            "--source-directory",
            str(tmp_path / "src"),
            "--cache-directory",
            str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args).run()
    assert exc.value.code == 2
    pipeline.assert_not_called()
    assert "uv sync --inexact --extra ontology" in " ".join(
        capsys.readouterr().err.split()
    )


def test_without_rdflib_the_loader_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unified_kg_rag.adapters.schema_sources as package

    # As on an install without the extra: the reader was never imported.
    monkeypatch.setitem(sys.modules, "rdflib", None)
    monkeypatch.delattr(package, "owl_reader", raising=False)
    monkeypatch.delitem(sys.modules, f"{package.__name__}.owl_reader", raising=False)
    with pytest.raises(SchemaSourceError, match="needs rdflib") as error:
        _load(PROCURE)
    assert "uv sync --inexact --extra ontology" in str(error.value)


def test_the_package_imports_without_rdflib() -> None:
    code = (
        "import sys\n"
        "sys.modules['rdflib'] = None\n"
        "import unified_kg_rag.adapters.schema_sources as s\n"
        "assert 'owl' in s.registered_schema_sources()\n"
        "import unified_kg_rag.application.cli.preflight\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- every serialisation, one schema ------------------------------------------------


@requires_ontology
@pytest.mark.parametrize("name", SERIALISATIONS)
def test_every_serialisation_states_the_same_triples(name: str) -> None:
    from rdflib import Graph
    from rdflib.compare import isomorphic

    formats = {".ttl": "turtle", ".rdf": "xml", ".jsonld": "json-ld", ".nt": "nt"}
    path = FIXTURES / name
    graph = Graph().parse(data=path.read_bytes(), format=formats[path.suffix])
    assert isomorphic(graph, Graph().parse(data=PROCURE.read_bytes(), format="turtle"))


@requires_ontology
@pytest.mark.parametrize("name", SERIALISATIONS)
def test_every_serialisation_maps_to_the_golden_schema(name: str) -> None:
    assert _load(FIXTURES / name) == PROCURE_GOLDEN


def _quads(graph_iri: str) -> str:
    """procure.nt with every statement in the named graph ``graph_iri``."""
    lines = (FIXTURES / "procure.nt").read_text(encoding="utf-8").splitlines()
    return "".join(
        f"{line[: line.rindex('.')]}<{graph_iri}> .\n"
        for line in lines
        if line and not line.startswith("#")
    )


def _trig() -> str:
    """procure.ttl in a named graph of a TriG file."""
    body = PROCURE.read_text(encoding="utf-8").split(PREFIXES, 1)[1]
    return f"{PREFIXES}\nex:graph {{\n{body}\n}}\n"


@requires_ontology
def test_named_graphs_are_read(tmp_path: Path) -> None:
    nq = _write(tmp_path / "procure.nq", _quads("https://example.org/graph"))
    trig = _write(tmp_path / "procure.trig", _trig())
    assert _load(nq) == PROCURE_GOLDEN
    assert _load(trig) == PROCURE_GOLDEN
    # The TriG and JSON-LD prefixes read the options' CURIEs.
    for path in (trig, FIXTURES / "procure.jsonld"):
        narrowed = _load(path, classes=["ex:Organization"])
        assert [t.label for t in narrowed.entity_types] == [  # type: ignore[union-attr]
            "ORGANIZATION",
            "BUYER",
            "VENDOR",
        ]


@requires_ontology
def test_an_s3_file_without_a_reader_is_a_schema_source_error() -> None:
    from unified_kg_rag.adapters.schema_sources import owl
    from unified_kg_rag.adapters.source_files.locations import SourceFile
    from unified_kg_rag.shared.exceptions import SchemaSourceError

    files = [(SourceFile(location="s3://bucket/onto.ttl", size=10), "turtle")]
    with pytest.raises(SchemaSourceError, match="needs a reader for S3"):
        owl.read_ontology(files, OwlSourceOptions())


@requires_ontology
def test_the_reader_builds_the_model_the_mapping_example_states() -> None:
    from unified_kg_rag.adapters.schema_sources import owl, owl_reader

    files = owl.locate(PROCURE, OwlSourceOptions())
    model = owl.read_ontology(files, OwlSourceOptions())
    expected = procure_model()
    assert {c.iri: c for c in model.classes} == {c.iri: c for c in expected.classes}
    properties = {p.iri: p for p in expected.properties}
    # The reader records owl:inverseOf both ways.
    for iri, inverse in (("supplies", "suppliedBy"), ("employs", "worksFor")):
        properties[EX + iri] = properties[EX + iri].model_copy(
            update={"inverses": frozenset({EX + inverse})}
        )
    assert {p.iri: p for p in model.properties} == properties
    # The file's own bindings, never the library's defaults.
    assert model.prefixes == {
        "ex": EX,
        "owl": owl_reader.OWL,
        "rdf": owl_reader.RDF,
        "rdfs": owl_reader.RDFS,
        "xsd": owl_reader.XSD,
    }


@requires_ontology
def test_the_order_of_the_statements_changes_nothing(tmp_path: Path) -> None:
    lines = [
        line
        for line in (FIXTURES / "procure.nt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    union = (
        "<https://example.org/procure#visits> "
        "<http://www.w3.org/2000/01/rdf-schema#range> _:u .\n"
        "_:u <http://www.w3.org/2002/07/owl#unionOf> _:l1 .\n"
        "_:l1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "
        "<https://example.org/procure#Vendor> .\n"
        "_:l1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:l2 .\n"
        "_:l2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "
        "<https://example.org/procure#Buyer> .\n"
        "_:l2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> "
        "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .\n"
        "<https://example.org/procure#visits> "
        "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://www.w3.org/2002/07/owl#ObjectProperty> .\n"
        "<https://example.org/procure#visits> "
        "<http://www.w3.org/2000/01/rdf-schema#domain> "
        "<https://example.org/procure#Person> .\n"
    ).splitlines()
    first = _write(tmp_path / "a.nt", "\n".join(lines + union) + "\n")
    expected = _load(first)
    for seed in range(3):
        shuffled = lines + [
            line.replace("_:u", "_:x9").replace("_:l", f"_:b{seed}n") for line in union
        ]
        random.Random(seed).shuffle(shuffled)
        again = _write(tmp_path / f"b{seed}.nt", "\n".join(shuffled) + "\n")
        assert _load(again) == expected
    assert "PERSON -VISITS-> VENDOR" in [str(p) for p in expected.relationship_patterns]


# --- the reader's rules ---------------------------------------------------------------


def _model(tmp_path: Path, turtle: str) -> Any:
    from unified_kg_rag.adapters.schema_sources import owl

    path = _write(tmp_path / "o.ttl", PREFIXES + turtle)
    return owl.read_ontology(owl.locate(path, OwlSourceOptions()), OwlSourceOptions())


@requires_ontology
def test_the_reader_reads_named_classes_only(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:A a owl:Class ; rdfs:subClassOf owl:Thing ,
            [ a owl:Restriction ; owl:onProperty ex:p ; owl:someValuesFrom ex:B ] .
        ex:C rdfs:subClassOf ex:A .
        ex:D owl:equivalentClass ex:E .
        ex:Money a rdfs:Datatype .
        ex:acme a ex:A .
        ex:p a owl:ObjectProperty ; rdfs:domain ex:F ; rdfs:range ex:Money .
        """,
    )
    classes = {c.iri: c for c in model.classes}
    # ex:B only sits in a restriction; the individual and the datatype are
    # no classes; owl:Thing is built in.
    assert sorted(classes) == [EX + name for name in ("A", "C", "D", "E", "F")]
    assert classes[EX + "A"].parents == {"http://www.w3.org/2002/07/owl#Thing"}
    assert classes[EX + "D"].equivalents == {EX + "E"}
    assert classes[EX + "E"].equivalents == {EX + "D"}


@requires_ontology
def test_the_reader_reads_how_each_property_is_declared(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:obj a owl:ObjectProperty .
        ex:sym a owl:SymmetricProperty .
        ex:inv owl:inverseOf ex:obj .
        ex:below rdfs:subPropertyOf ex:obj .
        ex:belowBelow rdfs:subPropertyOf ex:below .
        ex:data a owl:DatatypeProperty .
        ex:note a owl:AnnotationProperty .
        ex:plain a rdf:Property .
        ex:literal a rdf:Property ; rdfs:range xsd:date .
        ex:text a rdf:Property ; rdfs:range rdfs:Literal .
        ex:functional a owl:FunctionalProperty .
        ex:both a owl:ObjectProperty, owl:DatatypeProperty .
        """,
    )
    kinds = {p.iri.removeprefix(EX): p.kind for p in model.properties}
    assert kinds == {
        "obj": "object",
        "sym": "object",
        "inv": "object",
        "below": "object",
        "belowBelow": "object",
        "data": "datatype",
        "note": "annotation",
        "plain": "untyped",
        "literal": "datatype",
        "text": "datatype",
        "both": "object",
    }
    symmetric = {p.iri for p in model.properties if p.symmetric}
    assert symmetric == {EX + "sym"}


@requires_ontology
def test_the_reader_reads_unions_and_schema_org_includes(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        @prefix s: <http://schema.org/> .
        @prefix ss: <https://schema.org/> .
        ex:p a owl:ObjectProperty ;
            rdfs:domain [ owl:unionOf ( ex:A ex:B ) ] , ex:C ;
            rdfs:range [ owl:intersectionOf ( ex:A ex:B ) ] .
        ex:q a rdf:Property ;
            s:domainIncludes ex:A ; ss:domainIncludes ex:B ;
            s:rangeIncludes ex:C .
        ex:r a owl:ObjectProperty ;
            rdfs:domain [ owl:unionOf ( ex:A [ owl:complementOf ex:B ] ) ] .
        """,
    )
    properties = {p.iri.removeprefix(EX): p for p in model.properties}
    union = frozenset({EX + "A", EX + "B"})
    assert properties["p"].domains == {union, frozenset({EX + "C"})}
    # An intersection restricts nothing the mapping can read.
    assert properties["p"].ranges == frozenset()
    # http and https schema.org together are one statement.
    assert properties["q"].domains == {union}
    assert properties["q"].ranges == {frozenset({EX + "C"})}
    assert properties["r"].domains == frozenset()


@requires_ontology
def test_the_reader_reads_string_annotations_and_deprecation(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        ex:A a owl:Class ; rdfs:label "A"@en, "에이"@ko, "a" ;
            skos:definition "Defined."^^xsd:string ;
            owl:versionInfo 3 .
        ex:Old a owl:Class ; owl:deprecated true .
        ex:Older a owl:Class ; owl:deprecated "true" .
        """,
    )
    classes = {c.iri.removeprefix(EX): c for c in model.classes}
    labels = classes["A"].annotations["http://www.w3.org/2000/01/rdf-schema#label"]
    assert [(t.language, t.value) for t in labels] == [
        ("", "a"),
        ("en", "A"),
        ("ko", "에이"),
    ]
    assert set(classes["A"].annotations) == {
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2004/02/skos/core#definition",
    }
    assert [c.deprecated for c in (classes["A"], classes["Old"], classes["Older"])] == [
        False,
        True,
        True,
    ]


@requires_ontology
def test_prefixes_two_files_bind_differently_are_left_out(tmp_path: Path) -> None:
    _write(
        tmp_path / "dir" / "a.ttl",
        "@prefix p: <urn:one:> .\n<urn:one:A> a <http://www.w3.org/2002/07/owl#Class> .\n",
    )
    _write(
        tmp_path / "dir" / "b.ttl",
        "@prefix p: <urn:two:> .\n<urn:two:B> a <http://www.w3.org/2002/07/owl#Class> .\n",
    )
    with pytest.raises(SchemaSourceError, match="'p:A' is neither an IRI nor a CURIE"):
        _load(tmp_path / "dir", classes=["p:A"])
    assert len(_load(tmp_path / "dir", classes=["urn:one:A"]).entity_types or []) == 1


# --- the Korean ontology ------------------------------------------------------------------


@requires_ontology
def test_a_korean_ontology_keeps_its_hangul_labels() -> None:
    document = _load(FIXTURES / "korean.ttl")
    assert [t.label for t in document.entity_types or []] == [  # type: ignore[union-attr]
        "조직",
        "공급_업체",
        "구매자",
    ]
    vendor = document.entity_types[1]  # type: ignore[index]
    assert vendor.description == "물품을 파는 조직. Kind of 조직."  # type: ignore[union-attr]
    # 소속 has no label: its percent-decoded local name is used, its IRI kept.
    assert [(t.label, t.iri) for t in document.relationship_types] == [  # type: ignore[union-attr]
        ("공급한다", "https://example.org/gonggeup#공급한다"),
        ("소속", "https://example.org/gonggeup#%EC%86%8C%EC%86%8D"),
    ]
    assert "공급_업체 -공급한다-> 구매자" in [
        str(p) for p in document.relationship_patterns
    ]
    preserved = _load(FIXTURES / "korean.ttl", label_case="preserve")
    assert preserved.entity_types[1].label == "공급 업체"  # type: ignore[index,union-attr]


@requires_ontology
def test_a_korean_ontology_resolves_and_validates(tmp_path: Path) -> None:
    resolved = resolve_extraction_schema(
        _config(FIXTURES / "korean.ttl", schema_mode="strict")
    )
    schema = resolved.processing.graph_extraction.extraction_schema()
    assert schema.entity_type("공급_업체") is not None
    assert len(schema.relationship_patterns) == 10


# --- directories, imports and locations ------------------------------------------


def _split(directory: Path) -> Path:
    """procure.ttl as two files (classes, properties) in ``directory``."""
    body = PROCURE.read_text(encoding="utf-8").split(PREFIXES, 1)[1]
    classes, properties = body.split("ex:tradesWith", 1)
    _write(directory / "classes.ttl", PREFIXES + classes)
    _write(
        directory / "nested" / "properties.ttl", PREFIXES + "ex:tradesWith" + properties
    )
    return directory


@requires_ontology
def test_a_directory_is_read_as_one_ontology(tmp_path: Path) -> None:
    directory = _split(tmp_path / "onto")
    # Other files, and names starting with ".", are skipped.
    _write(directory / "README.md", "# notes\n")
    _write(directory / ".draft.ttl", "not turtle at all")
    _write(directory / ".git" / "x.ttl", "not turtle either")
    assert _load(directory) == PROCURE_GOLDEN


@requires_ontology
def test_a_directory_mixes_serialisations(tmp_path: Path) -> None:
    directory = tmp_path / "onto"
    directory.mkdir()
    shutil.copy(FIXTURES / "procure.rdf", directory / "a.rdf")
    shutil.copy(FIXTURES / "procure.jsonld", directory / "b.jsonld")
    assert _load(directory) == PROCURE_GOLDEN


@requires_ontology
def test_unfollowed_imports_are_warned_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = _split(tmp_path / "onto")
    _write(
        directory / "header.ttl",
        PREFIXES
        + "<https://example.org/procure> a owl:Ontology ;\n"
        + "    owl:imports <https://example.org/parties>, <https://example.org/money> .\n"
        + "<https://example.org/extra> owl:imports <https://example.org/money> .\n",
    )
    with caplog.at_level(logging.WARNING):
        assert _load(directory) == PROCURE_GOLDEN
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert message.startswith(
        "Ontology: owl:imports not followed: https://example.org/money, "
        "https://example.org/parties;"
    )
    # Once the imported ontologies are in the directory, nothing is warned.
    _write(
        directory / "parties.ttl",
        PREFIXES
        + "<https://example.org/parties> a owl:Ontology .\n"
        + "<https://example.org/money/v1> owl:versionIRI <https://example.org/money> .\n",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _load(directory)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@requires_ontology
def test_the_mapping_report_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        _load(PROCURE)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(f"Ontology '{PROCURE}' (1 file): 4 entity types, 5 relationship")
        for m in messages
    )
    assert f"Ontology '{PROCURE}': skipped {EX}name: datatype property" in messages


@requires_ontology
def test_a_file_url_and_a_relative_path_are_read(tmp_path: Path) -> None:
    assert _load(PROCURE.resolve().as_uri()) == PROCURE_GOLDEN
    config_file = _write(
        tmp_path / "config.yaml",
        "processing:\n  graph_extraction:\n    schema_source:\n"
        "      kind: owl\n      path: onto/procure.ttl\n",
    )
    shutil.copytree(FIXTURES, tmp_path / "onto")
    from unified_kg_rag.application.config import ConfigLoader

    config = ConfigLoader(config_file).load_config()
    resolved = resolve_extraction_schema(config)
    assert resolved.processing.graph_extraction.extraction_schema().entity_type(
        "VENDOR"
    )


@requires_ontology
def test_relative_iris_are_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(
        tmp_path / "o.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<#Vendor> a owl:Class .\n",
    )
    with caplog.at_level(logging.WARNING):
        document = _load(path)
    assert document.entity_types[0].iri == path.resolve().as_uri() + "#Vendor"  # type: ignore[index,union-attr]
    assert any("relative to their file" in r.getMessage() for r in caplog.records)


# --- refusals ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "schema_source.path is required for kind 'owl'"),
        ({"path": "https://example.org/onto.ttl"}, "is a URL; kind 'owl' reads a"),
        ({"path": "s3://B_ad/onto.ttl"}, "does not name an S3 bucket"),
        ({"path": "file://host/onto.ttl"}, "is a URL"),
        (
            {
                "path": "onto.ttl",
                "secret_arn": "arn:aws:secretsmanager:us-east-1:000000000000:secret:x",
            },
            "takes no schema_source.secret_arn",
        ),
        (
            {"path": "onto.ttl", "options": {"colour": 1}},
            "schema_source.options.colour: Extra inputs",
        ),
        (
            {"path": "onto.ttl", "options": {"format": "n3"}},
            "schema_source.options.format: Input should be",
        ),
        (
            {"path": "onto.ttl", "options": {"max_bytes": 0}},
            "schema_source.options.max_bytes",
        ),
        (
            {"path": "onto.ttl", "options": {"classes": [""]}},
            "schema_source.options.classes.0",
        ),
    ],
)
def test_bad_settings_are_refused(source: dict[str, Any], message: str) -> None:
    config = SchemaSourceConfig(kind="owl", **source)
    with pytest.raises(SchemaSourceError, match=message.replace(".", r"\.")):
        get_schema_source("owl")(SchemaSourceContext(config, Config()))


def test_a_url_with_a_credential_is_refused_without_showing_it() -> None:
    config = SchemaSourceConfig(
        kind="owl", path="https://user:token@example.org/o.ttl?sig=s"  # gitleaks:allow
    )
    with pytest.raises(SchemaSourceError) as error:
        get_schema_source("owl")(SchemaSourceContext(config, Config()))
    assert "token" not in str(error.value) and "sig" not in str(error.value)


@pytest.mark.parametrize("suffix", sorted(REFUSED_SUFFIXES))
def test_refused_formats_say_how_to_convert(tmp_path: Path, suffix: str) -> None:
    path = _write(tmp_path / f"onto{suffix}", "anything")
    with pytest.raises(SchemaSourceError, match="robot convert") as error:
        _load(path)
    assert REFUSED_SUFFIXES[suffix].split(" ")[0] in str(error.value)
    # Inside a directory too, rather than being skipped.
    shutil.copy(PROCURE, tmp_path / "procure.ttl")
    with pytest.raises(SchemaSourceError, match="robot convert"):
        _load(tmp_path)


@requires_ontology
def test_owl_xml_is_refused() -> None:
    with pytest.raises(SchemaSourceError, match="is OWL/XML .* robot convert"):
        _load(FIXTURES / "owl-xml.owl")


@requires_ontology
@pytest.mark.parametrize(
    "document",
    [
        (FIXTURES / "remote-context.jsonld").read_text(encoding="utf-8"),
        '{"@context": [{"ex": "urn:ex:"}, "context.jsonld"], "@id": "urn:a"}',
        '{"@context": {"@import": "https://example.org/c.jsonld"}, "@id": "urn:a"}',
        '{"@context": {"p": {"@id": "urn:p", "@context": "https://example.org/c"}},'
        ' "@id": "urn:a"}',
    ],
)
def test_a_json_ld_context_naming_a_document_is_refused(
    tmp_path: Path, document: str
) -> None:
    path = _write(tmp_path / "o.jsonld", document)
    with pytest.raises(SchemaSourceError, match="remote JSON-LD context"):
        _load(path)


@requires_ontology
def test_xml_external_entities_are_not_resolved(tmp_path: Path) -> None:
    shutil.copy(FIXTURES / "external-entity.rdf", tmp_path / "o.rdf")
    _write(tmp_path / "secret.txt", "SENTINEL-SECRET")
    document = _load(tmp_path / "o.rdf")
    assert document.model_dump_json().count("SENTINEL") == 0
    assert [(t.label, t.description) for t in document.entity_types or []] == [  # type: ignore[union-attr]
        ("VENDOR", "Sells goods.")
    ]


@requires_ontology
def test_internal_xml_entities_are_expanded() -> None:
    # procure.rdf abbreviates its namespace with an internal entity.
    assert "<!ENTITY ex " in (FIXTURES / "procure.rdf").read_text(encoding="utf-8")
    assert _load(FIXTURES / "procure.rdf") == PROCURE_GOLDEN


@requires_ontology
@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("o.ttl", PREFIXES + "ex:A a owl:Class ex:B .\n", "does not parse as turtle"),
        ("o.rdf", "<rdf:RDF", "does not parse as xml"),
        ("o.jsonld", "{", "does not parse as json-ld"),
        ("o.nt", "<urn:a> <urn:b> .\n", "does not parse as nt"),
    ],
)
def test_a_file_that_does_not_parse_is_named(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    path = _write(tmp_path / name, text)
    with pytest.raises(SchemaSourceError, match=message) as error:
        _load(path)
    assert "\n" not in str(error.value)
    assert str(path) in str(error.value)


def test_missing_and_unknown_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SchemaSourceError, match="not found"):
        _load(tmp_path / "absent.ttl")
    with pytest.raises(SchemaSourceError, match="holds no ontology file"):
        _load(tmp_path)
    path = _write(tmp_path / "onto.txt", "x")
    with pytest.raises(SchemaSourceError, match="must end in .ttl.*or set"):
        _load(path)


@requires_ontology
def test_a_format_option_reads_any_suffix(tmp_path: Path) -> None:
    path = tmp_path / "onto.txt"
    shutil.copy(PROCURE, path)
    assert _load(path, format="turtle") == PROCURE_GOLDEN
    # And applies to every file: Turtle read as RDF/XML does not parse.
    with pytest.raises(SchemaSourceError, match="does not parse as xml"):
        _load(PROCURE, format="xml")


def test_more_than_max_bytes_is_refused(tmp_path: Path) -> None:
    size = PROCURE.stat().st_size
    with pytest.raises(SchemaSourceError, match=f"holds {size} bytes, more than"):
        _load(PROCURE, max_bytes=size - 1)
    directory = _split(tmp_path / "onto")
    total = sum(p.stat().st_size for p in directory.rglob("*.ttl"))
    with pytest.raises(SchemaSourceError, match=f"holds {total} bytes"):
        _load(directory, max_bytes=total - 1)


@requires_ontology
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"classes": ["ex:Nope"]}, "'ex:Nope' names no class"),
        ({"namespaces": ["urn:none:"]}, "holds no class or property"),
        ({"max_entity_types": 2}, "more than max_entity_types"),
        ({"max_patterns": 5}, "more than max_patterns"),
        ({"labels": {"ex:Vendor": "BUYER"}}, "entity type labels collide"),
    ],
)
def test_mapping_failures_surface(options: dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaSourceError, match=message):
        _load(PROCURE, **options)


@requires_ontology
def test_colliding_labels_fail_or_are_qualified(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "o.ttl",
        PREFIXES
        + "@prefix geo: <https://example.org/geo#> .\n"
        + 'ex:Bank a owl:Class ; rdfs:label "Bank"@en .\n'
        + 'geo:Bank a owl:Class ; rdfs:label "Bank"@en .\n',
    )
    with pytest.raises(SchemaSourceError, match="'BANK' names"):
        _load(path)
    labels = [t.label for t in _load(path, on_collision="qualify").entity_types or []]  # type: ignore[union-attr]
    assert sorted(labels) == ["BANK_EX", "BANK_GEO"]


@requires_ontology
def test_the_config_may_not_set_entity_types_too() -> None:
    config = _config(PROCURE, entity_types=["PERSON"])
    with pytest.raises(SchemaSourceError, match="Both the config and schema_source"):
        resolve_extraction_schema(config)


# --- start-up and the cache keys ---------------------------------------------------


@requires_ontology
def test_start_up_writes_the_mapped_schema_into_the_inline_keys() -> None:
    resolved = resolve_extraction_schema(_config(PROCURE, schema_mode="strict"))
    extraction = resolved.processing.graph_extraction
    assert extraction.schema_source == SchemaSourceConfig()
    inline = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_mode": "strict",
                    **PROCURE_GOLDEN.model_dump(mode="json", exclude_none=True),
                }
            }
        }
    )
    assert extraction == inline.processing.graph_extraction
    assert _keys(resolved) == _keys(inline)
    patterns = {str(p) for p in extraction.extraction_schema().relationship_patterns}
    # The declared direction only; the lifted symmetric one both ways.
    assert "VENDOR -SUPPLIES-> BUYER" in patterns
    assert "BUYER -SUPPLIES-> VENDOR" not in patterns
    assert "BUYER -TRADES_WITH-> VENDOR" in patterns
    assert resolve_extraction_schema(resolved) is resolved


@requires_ontology
def test_reserialising_or_moving_the_ontology_keys_nothing(tmp_path: Path) -> None:
    first = resolve_extraction_schema(_config(PROCURE))
    moved = tmp_path / "elsewhere" / "ontology.nt"
    moved.parent.mkdir()
    shutil.copy(FIXTURES / "procure.nt", moved)
    for path in (FIXTURES / "procure.rdf", FIXTURES / "procure.jsonld", moved):
        again = resolve_extraction_schema(_config(path))
        assert _keys(again) == _keys(first)
        assert extraction_fingerprint(again) == extraction_fingerprint(first)
        assert schema_annotation_digest(again) == schema_annotation_digest(first)


def _edited(tmp_path: Path, old: str, new: str) -> Config:
    text = PROCURE.read_text(encoding="utf-8")
    assert old in text
    path = _write(
        tmp_path / f"edit-{len(list(tmp_path.iterdir()))}.ttl", text.replace(old, new)
    )
    return resolve_extraction_schema(_config(path))


@requires_ontology
def test_an_iri_only_edit_reannotates_and_a_label_edit_reextracts(
    tmp_path: Path,
) -> None:
    first = resolve_extraction_schema(_config(PROCURE))
    # ex:Vendor renamed ex:Supplier, its label kept.
    renamed = _edited(tmp_path, "ex:Vendor", "ex:Supplier")
    indexing = PipelineStageType.INDEXING.value
    assert extraction_fingerprint(renamed) == extraction_fingerprint(first)
    before, after = _keys(first), _keys(renamed)
    assert {stage for stage in before if before[stage] != after[stage]} == {indexing}
    assert schema_annotation_digest(renamed) != schema_annotation_digest(first)

    # A label edit changes what extraction is asked for.
    relabelled = _edited(tmp_path, '"Vendor"@en', '"Supplier"@en')
    assert extraction_fingerprint(relabelled) != extraction_fingerprint(first)
    extraction = PipelineStageType.GRAPH_EXTRACTION.value
    assert _keys(relabelled)[extraction] != before[extraction]
    upstream = PipelineStageType.TRANSLATION.value
    assert _keys(relabelled)[upstream] == before[upstream]

    # So does a description.
    described = _edited(tmp_path, "sells goods", "sells parts")
    assert extraction_fingerprint(described) != extraction_fingerprint(first)


@requires_ontology
def test_an_unreadable_file_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "o.ttl", PREFIXES)

    def refuse(self: Path) -> bytes:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    with pytest.raises(SchemaSourceError, match="Cannot read ontology file .*denied"):
        _load(path)


@requires_ontology
def test_a_file_that_grew_past_max_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "o.ttl", "")
    monkeypatch.setattr(Path, "read_bytes", lambda self: b" " * 100)
    with pytest.raises(SchemaSourceError, match=r"more than .*max_bytes \(50\)"):
        _load(path, max_bytes=50)


@requires_ontology
def test_an_rdf_xml_file_without_an_element_does_not_parse(tmp_path: Path) -> None:
    path = _write(tmp_path / "o.rdf", '<?xml version="1.0"?>\n')
    with pytest.raises(SchemaSourceError, match="does not parse as xml"):
        _load(path)


@requires_ontology
def test_json_ld_prefixes_are_read_from_every_top_level_object(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "o.jsonld",
        '[{"@context": {"ex": "https://example.org/procure#", "label": '
        '{"@id": "http://www.w3.org/2000/01/rdf-schema#label"}}, "@id": "ex:A", '
        '"@type": "http://www.w3.org/2002/07/owl#Class"},'
        ' {"@id": "https://example.org/procure#B",'
        ' "@type": "http://www.w3.org/2002/07/owl#Class"}]',
    )
    document = _load(path, classes=["ex:A"])
    assert [t.label for t in document.entity_types or []] == ["A"]  # type: ignore[union-attr]


def test_another_import_error_is_not_read_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unified_kg_rag.adapters.schema_sources as package

    # A broken module of the package, not the extra, fails the import.
    monkeypatch.delattr(package, "owl_reader", raising=False)
    monkeypatch.setitem(sys.modules, f"{package.__name__}.owl_reader", None)
    with pytest.raises(ImportError) as error:
        _load(PROCURE)
    assert not isinstance(error.value, SchemaSourceError)


# --- hostile and real-world ontologies -----------------------------------------------

RDF_XML_HEAD = (
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
    'xmlns:owl="http://www.w3.org/2002/07/owl#">'
)
LAUGHS = (
    '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa">'
    + "".join(
        f'<!ENTITY {chr(98 + i)} "' + f"&{chr(97 + i)};" * 10 + '">' for i in range(9)
    )
    + "]>"
)


def _refused_fast(path: Path, message: str, **options: Any) -> None:
    import time

    started = time.monotonic()
    with pytest.raises(SchemaSourceError, match=message):
        _load(path, **options)
    # Each shape took the parser minutes before it was refused up front.
    assert time.monotonic() - started < 5


@requires_ontology
@pytest.mark.parametrize(
    ("body", "message"),
    [
        # 567 bytes whose entities expand to 10^10 characters of a label.
        (
            '<owl:Class rdf:about="http://e/A"><rdfs:label>&j;</rdfs:label>'
            "</owl:Class>",
            "MAX_XML_TEXT_CHUNKS",
        ),
        # The same in an attribute: the XML parser's amplification limit.
        ('<owl:Class rdf:about="http://e/&j;"/>', "amplification"),
    ],
)
def test_xml_entities_expanding_past_the_limits_are_refused_fast(
    tmp_path: Path, body: str, message: str
) -> None:
    path = _write(tmp_path / "o.rdf", LAUGHS + RDF_XML_HEAD + body + "</rdf:RDF>")
    _refused_fast(path, message)


@requires_ontology
def test_an_xml_element_of_many_text_pieces_is_refused_fast(tmp_path: Path) -> None:
    # A 1.6 MB comment of line breaks: one piece of text per line.
    comment = "\n" * 1_600_000
    path = _write(
        tmp_path / "o.rdf",
        RDF_XML_HEAD + '<owl:Class rdf:about="http://e/A"><rdfs:comment>'
        f"{comment}</rdfs:comment></owl:Class></rdf:RDF>",
    )
    _refused_fast(path, r"more than 10000 pieces of text .*MAX_XML_TEXT_CHUNKS")


@requires_ontology
def test_a_large_xml_literal_is_refused_fast(tmp_path: Path) -> None:
    # 2,000 elements in an rdf:XMLLiteral: re-parsed at every element.
    path = _write(
        tmp_path / "o.rdf",
        RDF_XML_HEAD + '<owl:Class rdf:about="http://e/A">'
        '<rdfs:comment rdf:parseType="Literal">' + "<b/>" * 2000 + "</rdfs:comment>"
        "</owl:Class></rdf:RDF>",
    )
    _refused_fast(path, "MAX_XML_LITERAL_ELEMENTS")


@requires_ontology
def test_xml_under_the_limits_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "o.rdf",
        RDF_XML_HEAD + '<owl:Class rdf:about="http://e/A"><rdfs:label>A</rdfs:label>'
        "<rdfs:comment>" + "line\n" * 1000 + "</rdfs:comment>"
        '<rdfs:seeAlso rdf:parseType="Literal">' + "<b>x</b>" * 100 + "</rdfs:seeAlso>"
        "</owl:Class></rdf:RDF>",
    )
    assert [t.label for t in _load(path).entity_types or []] == ["A"]  # type: ignore[union-attr]


def check_bytes(path: str, data: bytes, format_: str) -> None:
    from unified_kg_rag.adapters.schema_sources.owl import ONTOLOGY_TERMS
    from unified_kg_rag.adapters.source_files import rdf_files

    rdf_files.check_bytes(path, data, format_, terms=ONTOLOGY_TERMS)


def _xml(body: str) -> bytes:
    return (
        RDF_XML_HEAD + '<owl:Class rdf:about="http://e/A">' + body + "</owl:Class>"
        "</rdf:RDF>"
    ).encode()


def test_the_xml_limits_allow_exactly_their_value() -> None:
    from unified_kg_rag.adapters.source_files.rdf_files import (
        MAX_XML_LITERAL_ELEMENTS,
        MAX_XML_TEXT_CHUNKS,
    )

    # One piece of text per line break, counted per element (its children's
    # apart); the literal elements per file.
    breaks = "\n" * MAX_XML_TEXT_CHUNKS
    pieces = "<rdfs:comment>" + breaks + "</rdfs:comment>"
    check_bytes("o.rdf", _xml(breaks + pieces * 2), "xml")
    with pytest.raises(SchemaSourceError, match="MAX_XML_TEXT_CHUNKS"):
        check_bytes("o.rdf", _xml(pieces.replace("\n", "\n\n", 1)), "xml")
    literal = '<rdfs:comment rdf:parseType="Literal">' + "<b/>" * (
        MAX_XML_LITERAL_ELEMENTS // 2
    )
    check_bytes("o.rdf", _xml((literal + "</rdfs:comment>") * 2), "xml")
    with pytest.raises(SchemaSourceError, match="MAX_XML_LITERAL_ELEMENTS"):
        check_bytes("o.rdf", _xml((literal + "<b/></rdfs:comment>") * 2), "xml")


def test_elements_after_an_xml_literal_are_not_counted_in_it() -> None:
    from unified_kg_rag.adapters.source_files.rdf_files import (
        MAX_XML_LITERAL_ELEMENTS,
    )

    body = '<rdfs:comment rdf:parseType="Literal"><b/></rdfs:comment>' + (
        "<rdfs:label>x</rdfs:label>" * (MAX_XML_LITERAL_ELEMENTS + 1)
    )
    check_bytes("o.rdf", _xml(body), "xml")


@requires_ontology
def test_deeply_nested_json_ld_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "o.jsonld", "[" * 100_000 + "]" * 100_000)
    with pytest.raises(SchemaSourceError, match="does not parse as json-ld"):
        _load(path)


@requires_ontology
@pytest.mark.parametrize(("name", "graph"), [("o.nt", ""), ("o.nq", " <urn:g>")])
def test_a_long_n_triples_line_is_refused_fast(
    tmp_path: Path, name: str, graph: str
) -> None:
    # A 1.5 MB literal of escapes: the parser copies the line at each one.
    literal = "a\\n" * 500_000
    path = _write(
        tmp_path / name,
        "<http://e/A> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        f"<http://www.w3.org/2002/07/owl#Class>{graph} .\n"
        f'<http://e/A> <http://www.w3.org/2000/01/rdf-schema#comment> "{literal}"'
        f"{graph} .\n",
    )
    _refused_fast(path, r"a line of 1500\d+ bytes, more than the 262144 .MAX_LINE")


@requires_ontology
def test_an_n_triples_line_at_the_limit_parses(tmp_path: Path) -> None:
    head = '<http://e/A> <http://www.w3.org/2000/01/rdf-schema#comment> "'
    line = head + "a" * (256 * 1024 - len(head) - 3) + '" .'
    assert len(line) == 256 * 1024
    path = _write(tmp_path / "o.nt", line + "\n")
    from unified_kg_rag.adapters.schema_sources import owl

    owl.read_ontology(owl.locate(path, OwlSourceOptions()), OwlSourceOptions())
    with pytest.raises(SchemaSourceError, match="a line of 262145 bytes"):
        _load(_write(tmp_path / "p.nt", line.replace('" .', 'a" .') + "\n"))


@requires_ontology
def test_a_cyclic_list_is_refused(tmp_path: Path) -> None:
    # Regression: rdflib's list walk raised a bare ValueError.
    path = _write(
        tmp_path / "o.ttl",
        PREFIXES + "ex:p a owl:ObjectProperty ; rdfs:domain [ owl:unionOf _:l ] .\n"
        "_:l rdf:first ex:A ; rdf:rest _:l .\n",
    )
    with pytest.raises(SchemaSourceError, match="rdf:List is cyclic"):
        _load(path)


@requires_ontology
def test_an_iri_with_whitespace_is_refused(tmp_path: Path) -> None:
    # Regression: the model's IRI check raised a pydantic ValidationError.
    path = _write(
        tmp_path / "o.rdf",
        RDF_XML_HEAD + '<owl:Class rdf:about="http://e/Supply Chain"/></rdf:RDF>',
    )
    with pytest.raises(
        SchemaSourceError, match="IRIs with whitespace: 'http://e/Supply"
    ):
        _load(path)


@requires_ontology
def test_a_prefix_bound_to_whitespace_is_left_out(tmp_path: Path) -> None:
    from unified_kg_rag.adapters.schema_sources import owl

    path = _write(
        tmp_path / "o.jsonld",
        '{"@context": {"bad": "http://e x/", "ex": "https://example.org/procure#"},'
        ' "@id": "ex:A", "@type": "http://www.w3.org/2002/07/owl#Class"}',
    )
    model = owl.read_ontology(owl.locate(path, OwlSourceOptions()), OwlSourceOptions())
    assert model.prefixes == {"ex": EX}


SCHEMA_ORG = """\
@prefix ex: <https://example.org/v#> .
@prefix schema: <https://schema.org/> .
@prefix s: <http://schema.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Supplier a rdfs:Class ; rdfs:subClassOf schema:Organization .
ex:Part a rdfs:Class .
ex:Sku a schema:DataType .
ex:supplies a rdf:Property ; schema:domainIncludes ex:Supplier ;
    schema:rangeIncludes ex:Part .
ex:partNumber a rdf:Property ; schema:domainIncludes ex:Part ;
    schema:rangeIncludes schema:Text .
ex:catalogUrl a rdf:Property ; schema:domainIncludes ex:Supplier ;
    schema:rangeIncludes schema:URL .
ex:leadTimeDays a rdf:Property ; schema:domainIncludes ex:Part ;
    s:rangeIncludes s:Number .
ex:sku a rdf:Property ; schema:domainIncludes ex:Part ; schema:rangeIncludes ex:Sku .
ex:maker a rdf:Property ; schema:domainIncludes ex:Part ;
    schema:rangeIncludes schema:Text , ex:Supplier .
"""


@requires_ontology
def test_schema_org_data_types_are_literals(tmp_path: Path) -> None:
    # Regression: schema:Text, schema:URL, ... became entity types and the
    # properties ranging over them relationship types.
    path = _write(tmp_path / "o.ttl", SCHEMA_ORG)
    document = _load(path)
    assert [t.label for t in document.entity_types or []] == [  # type: ignore[union-attr]
        "ORGANIZATION",
        "SUPPLIER",
        "PART",
    ]
    # A property is a datatype one only when all its ranges are literal.
    assert [t.label for t in document.relationship_types] == ["MAKER", "SUPPLIES"]
    assert [
        (p.source_type, p.relation, p.target_type)
        for p in document.relationship_patterns or []
    ] == [("PART", "MAKER", "SUPPLIER"), ("SUPPLIER", "SUPPLIES", "PART")]


DEFINED = PREFIXES + """
ex:Organization a owl:Class .
ex:Product a owl:Class .
ex:Person a owl:Class .
ex:sells a owl:ObjectProperty .
ex:Vendor a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:intersectionOf (
    ex:Organization
    [ a owl:Restriction ; owl:onProperty ex:sells ; owl:someValuesFrom ex:Product ]
) ] .
ex:Reseller a owl:Class ; rdfs:subClassOf [ owl:intersectionOf ( ex:Vendor ex:Buyer ) ] .
ex:Party a owl:Class ;
    owl:equivalentClass [ a owl:Class ; owl:unionOf ( ex:Person ex:Organization ) ] .
ex:employs a owl:ObjectProperty ; rdfs:domain ex:Organization ; rdfs:range ex:Person .
"""


@requires_ontology
def test_class_expressions_state_parents(tmp_path: Path) -> None:
    # Regression: a defined class (an equivalent intersection) had no
    # parent, and the members of a covering union none either.
    model = _model(tmp_path, DEFINED.split(PREFIXES, 1)[1])
    parents = {c.iri.removeprefix(EX): set(c.parents) for c in model.classes}
    assert parents == {
        "Buyer": set(),
        "Organization": {EX + "Party"},
        "Party": set(),
        "Person": {EX + "Party"},
        "Product": set(),
        "Reseller": {EX + "Vendor", EX + "Buyer"},
        "Vendor": {EX + "Organization"},
    }


@requires_ontology
def test_a_defined_class_takes_part_in_the_hierarchy(tmp_path: Path) -> None:
    path = _write(tmp_path / "o.ttl", DEFINED)
    document = _load(path, classes=["ex:Party"])
    # Selected below the union, with its hints, and pushed down to.
    assert [(t.label, t.description) for t in document.entity_types or []] == [  # type: ignore[union-attr]
        ("PARTY", ""),
        ("ORGANIZATION", "Kind of PARTY."),
        ("VENDOR", "Kind of ORGANIZATION."),
        ("RESELLER", "Kind of VENDOR."),
        ("PERSON", "Kind of PARTY."),
    ]
    assert [
        (p.source_type, p.relation, p.target_type)
        for p in document.relationship_patterns or []
    ] == [
        ("ORGANIZATION", "EMPLOYS", "PERSON"),
        ("VENDOR", "EMPLOYS", "PERSON"),
        ("RESELLER", "EMPLOYS", "PERSON"),
    ]


@requires_ontology
def test_a_datatype_property_domain_alone_makes_no_class(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:Person a owl:Class .
        ex:name a owl:DatatypeProperty ; rdfs:domain ex:Person , ex:Tag .
        ex:code a rdf:Property ; rdfs:domain ex:Badge ; rdfs:range xsd:string .
        """,
    )
    # A declared class counts; a datatype property's domain makes none.
    assert [c.iri for c in model.classes] == [EX + "Person"]


@requires_ontology
def test_sub_properties_of_an_rdf_property_are_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression: a sub-property of an untyped rdf:Property vanished, and a
    # property with only a domain and range was skipped without a word.
    with caplog.at_level(logging.INFO):
        model = _model(
            tmp_path,
            """
            ex:A a rdfs:Class . ex:B a rdfs:Class .
            ex:typed a rdf:Property ; rdfs:domain ex:A ; rdfs:range ex:B .
            ex:sub rdfs:subPropertyOf ex:typed .
            ex:subLiteral rdfs:subPropertyOf ex:typed ; rdfs:range xsd:string .
            ex:bare rdfs:domain ex:A ; rdfs:range ex:B .
            ex:func a owl:FunctionalProperty ; rdfs:domain ex:A .
            """,
        )
    kinds = {p.iri.removeprefix(EX): p.kind for p in model.properties}
    assert kinds == {"typed": "untyped", "sub": "untyped", "subLiteral": "datatype"}
    assert "2 properties not read (a domain, range or owl:FunctionalProperty" in (
        caplog.text
    )
    assert f"{EX}bare, {EX}func;" in caplog.text


@requires_ontology
@pytest.mark.parametrize(
    ("name", "text", "said", "secret"),
    [
        (
            "o.ttl",
            PREFIXES + "ex:A a owl:Class ex:SECRET .\n",
            r"line 6: Bad syntax \(expected '.' or '}' or '\]' at end of statement\)$",
            "SECRET",
        ),
        (
            "o.nt",
            "<urn:a> <urn:b> SECRET .\n",
            r"does not parse as nt: Invalid line$",
            "SECRET",
        ),
        (
            "o.rdf",
            RDF_XML_HEAD + "<owl:Class><SECRET></owl:Class></rdf:RDF>",
            r"does not parse as xml: mismatched tag: line 1, column \d+$",
            "SECRET",
        ),
    ],
)
def test_a_parse_error_says_what_is_wrong_without_the_text(
    tmp_path: Path, name: str, text: str, said: str, secret: str
) -> None:
    path = _write(tmp_path / name, text)
    with pytest.raises(SchemaSourceError, match=said) as error:
        _load(path)
    assert secret not in str(error.value).replace(str(path), "")


@requires_ontology
def test_a_format_option_reads_every_file_of_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "onto"
    directory.mkdir()
    shutil.copy(PROCURE, directory / "procure.txt")
    _write(directory / ".notes", "not turtle")
    with pytest.raises(SchemaSourceError, match="holds no ontology file"):
        _load(directory)
    assert _load(directory, format="turtle") == PROCURE_GOLDEN


CATALOG = """\
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<catalog prefer="public" xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
    <uri name="https://example.org/procure" uri="procure.ttl"/>
</catalog>
"""


@requires_ontology
def test_a_protege_catalog_in_a_directory_is_skipped(tmp_path: Path) -> None:
    directory = tmp_path / "onto"
    directory.mkdir()
    shutil.copy(PROCURE, directory / "procure.ttl")
    _write(directory / "catalog-v001.xml", CATALOG)
    assert _load(directory) == PROCURE_GOLDEN
    # Whatever the format option says.
    assert _load(directory, format="turtle") == PROCURE_GOLDEN
    # Named alone, it is no ontology.
    with pytest.raises(SchemaSourceError, match="is an OASIS XML catalog"):
        _load(directory / "catalog-v001.xml")


# --- the reader's smaller rules -------------------------------------------------------


@requires_ontology
def test_a_json_ld_named_graph_is_read(tmp_path: Path) -> None:
    # A top-level @id with @graph: the statements sit in a named graph.
    path = _write(
        tmp_path / "o.jsonld",
        '{"@context": {"owl": "http://www.w3.org/2002/07/owl#", '
        '"ex": "https://example.org/procure#"}, '
        '"@id": "https://example.org/procure", '
        '"@graph": [{"@id": "ex:Vendor", "@type": "owl:Class"}]}',
    )
    assert [t.label for t in _load(path, classes=["ex:Vendor"]).entity_types or []] == [  # type: ignore[union-attr]
        "VENDOR"
    ]


@requires_ontology
def test_a_json_ld_term_is_a_prefix_only_when_its_iri_ends_like_one(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.adapters.schema_sources import owl

    path = _write(
        tmp_path / "o.jsonld",
        '{"@context": {"ex": "https://example.org/procure#", '
        '"name": "http://schema.org/name", "slash": "https://example.org/v/"}, '
        '"@id": "ex:A", "@type": "http://www.w3.org/2002/07/owl#Class"}',
    )
    model = owl.read_ontology(owl.locate(path, OwlSourceOptions()), OwlSourceOptions())
    assert model.prefixes == {"ex": EX, "slash": "https://example.org/v/"}


@requires_ontology
def test_a_statement_with_two_union_lists_restricts_nothing(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:p a owl:ObjectProperty ;
            rdfs:domain [ owl:unionOf ( ex:A ex:B ) , ( ex:C ) ] ; rdfs:range ex:C .
        """,
    )
    (prop,) = model.properties
    assert prop.domains == frozenset()
    assert prop.ranges == {frozenset({EX + "C"})}


@requires_ontology
def test_a_malformed_union_list_restricts_nothing(tmp_path: Path) -> None:
    # A list cell with two rdf:first is no list.
    model = _model(
        tmp_path,
        """
        ex:p a owl:ObjectProperty ; rdfs:domain [ owl:unionOf _:l ] .
        _:l rdf:first ex:A , ex:B ; rdf:rest rdf:nil .
        """,
    )
    (prop,) = model.properties
    assert prop.domains == frozenset()


@requires_ontology
@pytest.mark.parametrize(
    ("value", "deprecated"),
    [("true", True), ('"1"', True), ("1", True), ("false", False), ('"0"', False)],
)
def test_owl_deprecated_reads_true_and_1(
    tmp_path: Path, value: str, deprecated: bool
) -> None:
    model = _model(tmp_path, f"ex:A a owl:Class ; owl:deprecated {value} .\n")
    assert [c.deprecated for c in model.classes] == [deprecated]


@requires_ontology
def test_the_ontology_languages_own_terms_are_never_read(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        owl:topObjectProperty a owl:ObjectProperty .
        rdfs:member a rdf:Property .
        rdfs:Resource a rdfs:Class .
        ex:p a owl:ObjectProperty ; rdfs:domain rdfs:Resource ; rdfs:range ex:A .
        """,
    )
    assert [p.iri for p in model.properties] == [EX + "p"]
    assert [c.iri for c in model.classes] == [EX + "A"]


@requires_ontology
@pytest.mark.parametrize(
    "characteristic",
    [
        "TransitiveProperty",
        "InverseFunctionalProperty",
        "AsymmetricProperty",
        "ReflexiveProperty",
        "IrreflexiveProperty",
    ],
)
def test_an_object_property_characteristic_makes_an_object_property(
    tmp_path: Path, characteristic: str
) -> None:
    model = _model(tmp_path, f"ex:p a owl:{characteristic} .\n")
    assert [(p.iri, p.kind) for p in model.properties] == [(EX + "p", "object")]


@requires_ontology
def test_literal_ranges_make_a_datatype_property(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:Money a rdfs:Datatype .
        ex:html a rdf:Property ; rdfs:range rdf:HTML .
        ex:lang a rdf:Property ; rdfs:range rdf:langString .
        ex:price a rdf:Property ; rdfs:range ex:Money .
        ex:count a rdf:Property ; rdfs:range xsd:integer .
        ex:mixed a rdf:Property ; rdfs:range xsd:integer , ex:Thing .
        """,
    )
    kinds = {p.iri.removeprefix(EX): p.kind for p in model.properties}
    assert kinds == {
        "html": "datatype",
        "lang": "datatype",
        "price": "datatype",
        "count": "datatype",
        "mixed": "untyped",
    }


@requires_ontology
def test_a_sub_property_of_an_rdf_property_is_read_by_its_ranges(
    tmp_path: Path,
) -> None:
    # Below an untyped property: a datatype one when its ranges are literal.
    model = _model(
        tmp_path,
        """
        ex:rel a rdf:Property .
        ex:sub rdfs:subPropertyOf ex:rel ; rdfs:range xsd:date .
        ex:same owl:equivalentProperty ex:rel ; rdfs:range ex:A .
        """,
    )
    kinds = {p.iri.removeprefix(EX): p.kind for p in model.properties}
    assert kinds == {"rel": "untyped", "sub": "datatype", "same": "untyped"}


@requires_ontology
def test_only_string_literals_are_annotations(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        """
        ex:A a owl:Class ; rdfs:label "A"^^xsd:string , "에이"@ko ;
            rdfs:comment "3"^^xsd:integer , "<b>x</b>"^^rdf:XMLLiteral .
        """,
    )
    (cls,) = model.classes
    assert set(cls.annotations) == {"http://www.w3.org/2000/01/rdf-schema#label"}
    texts = cls.annotations["http://www.w3.org/2000/01/rdf-schema#label"]
    assert [t.value for t in texts] == ["A", "에이"]


@requires_ontology
def test_a_covering_union_needs_an_equivalence(tmp_path: Path) -> None:
    # A subclass of a union is not its cover: the members get no parent.
    model = _model(
        tmp_path,
        """
        ex:Party a owl:Class ;
            rdfs:subClassOf [ a owl:Class ; owl:unionOf ( ex:Person ex:Org ) ] .
        ex:Agent a owl:Class .
        [ a owl:Class ; owl:unionOf ( ex:Bot ex:Human ) ] owl:equivalentClass ex:Agent .
        """,
    )
    parents = {c.iri.removeprefix(EX): set(c.parents) for c in model.classes}
    # The union stated on either side of owl:equivalentClass covers.
    assert parents == {
        "Agent": set(),
        "Bot": {EX + "Agent"},
        "Human": {EX + "Agent"},
        "Party": set(),
    }


@requires_ontology
def test_only_an_anonymous_class_expression_states_parents(tmp_path: Path) -> None:
    # OWL 2 writes a class expression as a blank node; an owl:intersectionOf
    # on a named class is not read, for it or for what is equivalent to it.
    model = _model(
        tmp_path,
        """
        ex:A owl:equivalentClass ex:B .
        ex:B owl:intersectionOf ( ex:C ex:D ) .
        ex:C a owl:Class . ex:D a owl:Class .
        """,
    )
    parents = {c.iri.removeprefix(EX): set(c.parents) for c in model.classes}
    assert parents == {"A": set(), "B": set(), "C": set(), "D": set()}


@requires_ontology
def test_the_files_may_hold_exactly_max_bytes(tmp_path: Path) -> None:
    size = PROCURE.stat().st_size
    assert _load(PROCURE, max_bytes=size) == PROCURE_GOLDEN
    directory = _split(tmp_path / "onto")
    total = sum(p.stat().st_size for p in directory.rglob("*.ttl"))
    assert _load(directory, max_bytes=total) == PROCURE_GOLDEN


@requires_ontology
def test_a_hidden_directory_below_the_top_is_skipped(tmp_path: Path) -> None:
    directory = _split(tmp_path / "onto")
    _write(directory / "nested" / ".cache" / "x.ttl", "not turtle at all")
    _write(directory / "nested" / ".x.ttl", "not turtle either")
    assert _load(directory) == PROCURE_GOLDEN
