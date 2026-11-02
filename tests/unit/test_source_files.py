# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared source-file readers (``adapters.source_files``) used by a
source other than the extraction schema.

A non-schema client provider and a non-schema error type
(:class:`SourceTerms`): S3 objects and prefixes sized before they are read
and read within a cap (moto, or a botocore Stubber for what moto cannot
answer), the RDF files a path holds and their formats, the checks their
bytes pass before parsing, and the parse itself. Every error is the
caller's type, never ``SchemaSourceError``, and names the caller's source,
options and kind.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from moto import mock_aws

from tests.fixtures.extras import requires_ontology
from unified_kg_rag.adapters.schema_sources import SchemaSourceContext
from unified_kg_rag.adapters.source_files import rdf_files
from unified_kg_rag.adapters.source_files.locations import (
    SourceClientProvider,
    SourceReader,
    SourceTerms,
    read_source_bytes,
    redact_location,
    split_s3,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.shared.exceptions import ConfigurationError, SchemaSourceError

pytestmark = pytest.mark.unit

REGION = "us-east-1"
BUCKET = "seed-bucket"
OPTIONS = "processing.seed_graphs[0].options"
MAX_BYTES = "processing.seed_graphs[0].max_bytes"


class SeedError(ConfigurationError):
    """A non-schema source's error."""


TERMS = SourceTerms(
    noun="Seed graph",
    options=OPTIONS,
    max_bytes=MAX_BYTES,
    kind="rdf",
    purpose="the seed graph",
    error=SeedError,
)
TURTLE = (
    "@prefix ex: <https://example.org/procure#> .\n"
    'ex:acme a ex:Vendor ; ex:name "Acme" .\n'
)


class SeedContext:
    """A client provider that is no ``SchemaSourceContext``: one S3 client,
    built when asked for."""

    def __init__(self, path: str, client: Any = None) -> None:
        self.path = path
        self.client_value = client
        self.services: list[str] = []

    @property
    def location(self) -> str:
        return redact_location(self.path)

    def client(self, service_name: str, **kwargs: Any) -> Any:
        self.services.append(service_name)
        if self.client_value is None:
            self.client_value = boto3.client(service_name, region_name=REGION)
        return self.client_value


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any socket connection: every S3 call is served by moto or a
    Stubber."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a source-file test reached for the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


@pytest.fixture
def s3() -> Iterator[Any]:
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3: Any, key: str, body: str | bytes) -> str:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body.encode("utf-8") if isinstance(body, str) else body,
    )
    return f"s3://{BUCKET}/{key}"


def _stubbed() -> tuple[Any, Stubber]:
    client = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # nosec B106 - placeholder
    )
    return client, Stubber(client)


def _raises(match: str) -> Any:
    return pytest.raises(SeedError, match=match)


# --- the client provider ---------------------------------------------------------------


def test_the_schema_context_is_a_client_provider() -> None:
    context: SourceClientProvider = SchemaSourceContext(
        SchemaSourceConfig(kind="owl", path="s3://user:token@b/o.ttl?x=1"),
        Config.model_validate({"aws": {"region_name": REGION}}),
        session_factory=lambda: boto3.Session(region_name=REGION),
    )
    assert context.location == "s3://b/o.ttl"


def test_the_default_terms_stay_the_schema_source_ones() -> None:
    with pytest.raises(SchemaSourceError, match="^Schema source 's3://B_/k'"):
        split_s3("s3://B_/k")
    assert SourceTerms().error is SchemaSourceError


# --- S3, through a non-schema provider ---------------------------------------------------


def test_an_object_is_read_through_the_provider(s3: Any) -> None:
    location = _put(s3, "seeds/a.ttl", TURTLE)
    context = SeedContext(location)
    data = read_source_bytes(context, location, max_bytes=1024, terms=TERMS)
    assert data == TURTLE.encode("utf-8")
    assert context.services == ["s3"]


def test_an_object_over_the_cap_is_refused_before_it_is_fetched(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _put(s3, "seeds/a.ttl", TURTLE)

    def refuse(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("fetched an object over the cap")

    monkeypatch.setattr(SourceReader, "read", refuse)
    with _raises(rf"^Seed graph file '{location}' holds {len(TURTLE)} bytes"):
        read_source_bytes(SeedContext(location), location, max_bytes=8, terms=TERMS)


def test_an_object_that_grew_is_refused_while_it_is_read(s3: Any) -> None:
    location = _put(s3, "seeds/a.ttl", TURTLE)
    reader = SourceReader(SeedContext(location), terms=TERMS)
    file = reader.head(location).model_copy(update={"size": 1})
    with _raises("holds more than 8 bytes .it grew since it was sized"):
        reader.read(file, max_bytes=8)


@pytest.mark.parametrize(
    ("location", "match"),
    [
        (f"s3://{BUCKET}/seeds/", "^Seed graph 's3://seed-bucket/seeds/' names an S3"),
        ("s3://Not_A_Bucket/k", "^Seed graph 's3://Not_A_Bucket/k' does not name"),
        (f"s3://{BUCKET}/missing.ttl", "^Seed graph '.*' not found .no such S3"),
        ("s3://no-such-bucket/k.ttl", "^Seed graph '.*': the S3 bucket does not"),
    ],
)
def test_s3_errors_are_the_callers(s3: Any, location: str, match: str) -> None:
    with _raises(match) as error:
        read_source_bytes(SeedContext(location), location, max_bytes=1024, terms=TERMS)
    assert not isinstance(error.value, SchemaSourceError)


def test_access_denied_names_the_callers_source() -> None:
    client, stubber = _stubbed()
    stubber.add_client_error("head_object", "AccessDenied", http_status_code=403)
    location = f"s3://{BUCKET}/a.ttl"
    with stubber, _raises(r"^Access to seed graph '.*' denied \(HeadObject\)"):
        SourceReader(SeedContext(location, client), terms=TERMS).head(location)


def test_another_client_error_names_the_callers_source() -> None:
    client, stubber = _stubbed()
    stubber.add_client_error("list_objects_v2", "SlowDown", http_status_code=503)
    location = f"s3://{BUCKET}/seeds/"
    with (
        stubber,
        _raises(r"^Cannot read seed graph '.*' from S3 \(ListObjectsV2\): SlowDown"),
    ):
        SourceReader(SeedContext(location, client), terms=TERMS).list_prefix(location)


# --- which RDF files a path holds --------------------------------------------------------


def _write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _locate(path: Path, format_: str = "auto", max_bytes: int = 1 << 20) -> Any:
    return rdf_files.locate(path, format_=format_, max_bytes=max_bytes, terms=TERMS)


def test_a_directory_is_read_by_suffix_sorted(tmp_path: Path) -> None:
    _write(tmp_path / "b.nt", "")
    _write(tmp_path / "a" / "c.jsonld", "{}")
    _write(tmp_path / "notes.txt", "")
    _write(tmp_path / ".hidden" / "d.ttl", "")
    found = _locate(tmp_path)
    assert [(Path(f.location).name, fmt) for f, fmt in found] == [
        ("c.jsonld", "json-ld"),
        ("b.nt", "nt"),
    ]


def test_a_format_reads_every_file_of_a_directory(tmp_path: Path) -> None:
    _write(tmp_path / "a.data", TURTLE)
    _write(tmp_path / "b.nt", TURTLE)
    assert [fmt for _, fmt in _locate(tmp_path, "turtle")] == ["turtle", "turtle"]


@pytest.mark.parametrize(
    ("name", "format_", "max_bytes", "match"),
    [
        ("seed.owx", "auto", 1 << 20, "^Seed graph file '.*' is OWL/XML, which kind 'rdf'"),
        ("seed.data", "auto", 1 << 20, f"or set {re.escape(OPTIONS)}.format$"),
        ("seed.ttl", "auto", 4, f"more than {re.escape(MAX_BYTES)} .4.; point the path at the files the seed graph needs"),
    ],
)  # fmt: skip
def test_a_file_is_refused_in_the_callers_terms(
    tmp_path: Path, name: str, format_: str, max_bytes: int, match: str
) -> None:
    path = _write(tmp_path / name, TURTLE)
    with _raises(match):
        _locate(path, format_, max_bytes)


def test_a_directory_without_rdf_files_names_the_callers_source(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "notes.txt", "")
    with _raises("^Seed graph directory '.*' holds no seed graph file"):
        _locate(tmp_path)
    with _raises("^Seed graph '.*missing' not found$"):
        _locate(tmp_path / "missing")


def test_a_prefix_is_located_through_the_provider(s3: Any) -> None:
    _put(s3, "seeds/b.ttl", TURTLE)
    _put(s3, "seeds/a/c.nq", "")
    _put(s3, "seeds/.git/d.ttl", "")
    _put(s3, "seeds/readme.md", "")
    location = f"s3://{BUCKET}/seeds/"
    reader = SourceReader(SeedContext(location), terms=TERMS)
    found = rdf_files.locate_s3(
        reader, location, format_="auto", max_bytes=1 << 20, terms=TERMS
    )
    assert [(f.location, fmt) for f, fmt in found] == [
        (f"s3://{BUCKET}/seeds/a/c.nq", "nquads"),
        (f"s3://{BUCKET}/seeds/b.ttl", "turtle"),
    ]
    with _raises("^Seed graph directory 's3://seed-bucket/empty/' holds no"):
        rdf_files.locate_s3(
            reader,
            f"s3://{BUCKET}/empty/",
            format_="auto",
            max_bytes=1 << 20,
            terms=TERMS,
        )


def test_a_local_file_that_cannot_be_read_names_the_callers_source(
    tmp_path: Path,
) -> None:
    with _raises("^Cannot read seed graph file '.*seed.ttl'"):
        rdf_files.read_local(tmp_path / "seed.ttl", terms=TERMS)


# --- the checks before parsing -----------------------------------------------------------

RDF_XML = (
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:ex="https://example.org/procure#">{}</rdf:RDF>'
)


@pytest.mark.parametrize(
    ("data", "format_", "match"),
    [
        (
            b'<Ontology xmlns="http://www.w3.org/2002/07/owl#"/>',
            "xml",
            r"is OWL/XML \(its root is owl:Ontology\), which kind 'rdf' does",
        ),
        (b"<rdf:RDF", "xml", "does not parse as xml"),
        (
            RDF_XML.format("<ex:A><ex:p>" + "&amp;" * 10_001 + "</ex:p></ex:A>").encode(),
            "xml",
            "more than 10000 pieces of text",
        ),
        (
            RDF_XML.format(
                '<ex:A><ex:p rdf:parseType="Literal">' + "<b/>" * 1_001 + "</ex:p></ex:A>"
            ).encode(),
            "xml",
            "MAX_XML_LITERAL_ELEMENTS",
        ),
        (b"<a> <b> <" + b"c" * (256 * 1024) + b"> .", "nt", "MAX_LINE_BYTES"),
        (b'{"@context": "https://example.org/ctx"}', "json-ld", "remote JSON-LD"),
        (b'[{"@context": {"@import": "x"}}]', "json-ld", r"remote JSON-LD context \(@import\)"),
        (b"{", "json-ld", "does not parse as json-ld"),
    ],
)  # fmt: skip
def test_bytes_are_refused_in_the_callers_terms(
    data: bytes, format_: str, match: str
) -> None:
    with _raises(f"^Seed graph file 'seed' .*{match}") as error:
        rdf_files.check_bytes("seed", data, format_, terms=TERMS)
    assert not isinstance(error.value, SchemaSourceError)


@pytest.mark.parametrize("format_", ["turtle", "xml", "nt", "json-ld", "trig"])
def test_safe_bytes_pass(format_: str) -> None:
    data = {
        "xml": RDF_XML.format('<ex:A rdf:about="https://example.org/a"/>').encode(),
        "json-ld": b'{"@context": {"ex": "https://example.org/"}, "@id": "ex:a"}',
    }.get(format_, TURTLE.encode())
    rdf_files.check_bytes("seed", data, format_, terms=TERMS)


def test_an_xml_catalog_is_recognised() -> None:
    catalog = b'<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>'
    assert rdf_files.is_xml_catalog(catalog)
    assert not rdf_files.is_xml_catalog(RDF_XML.format("").encode())
    assert not rdf_files.is_xml_catalog(TURTLE.encode())


def test_the_file_helpers_import_without_rdflib() -> None:
    code = (
        "import sys\n"
        "sys.modules['rdflib'] = None\n"
        "import unified_kg_rag.adapters.source_files.locations\n"
        "import unified_kg_rag.adapters.source_files.rdf_files\n"
        "assert 'unified_kg_rag.adapters.schema_sources' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- the parse ---------------------------------------------------------------------------


@requires_ontology
def test_the_bytes_are_parsed_into_one_graph(tmp_path: Path) -> None:
    from unified_kg_rag.adapters.source_files import rdf_parse

    statements = rdf_parse.Statements()
    seed = _write(tmp_path / "seed.ttl", TURTLE)
    for file, format_ in _locate(seed):
        rdf_parse.parse(
            statements,
            file.name,
            file.base,
            seed.read_bytes(),
            format_,
            terms=TERMS,
        )
    rdf_parse.parse(
        statements,
        "more.nq",
        "s3://seed-bucket/more.nq",
        b"<https://example.org/procure#b> <https://example.org/procure#p> "
        b'"x" <https://example.org/g> .\n',
        "nquads",
        terms=TERMS,
    )
    assert len(statements.graph) == 3
    assert statements.prefixes() == {"ex": "https://example.org/procure#"}


@requires_ontology
def test_a_file_that_does_not_parse_names_the_callers_source() -> None:
    from unified_kg_rag.adapters.source_files import rdf_parse

    with _raises(
        "^Seed graph file 'seed.ttl' does not parse as turtle: line 1"
    ) as error:
        rdf_parse.parse(
            rdf_parse.Statements(),
            "seed.ttl",
            "file:///seed.ttl",
            b"ex:a ex:b .",
            "turtle",
            terms=TERMS,
        )
    assert not isinstance(error.value, SchemaSourceError)
