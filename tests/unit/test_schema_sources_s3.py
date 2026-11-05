# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema sources in S3: kinds ``file`` and ``owl`` with an ``s3://`` path.

Against moto-mocked S3 (botocore Stubber where a response moto cannot give
is needed): one object, a prefix read as a directory (every page of the
listing, filtered as a local directory is), the size checked before any
byte is fetched, and the S3 errors as ``SchemaSourceError``s saying what to
fix. The loaded content keys the caches, the location does not.
"""

from __future__ import annotations

import io
import logging
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError
from botocore.response import StreamingBody
from botocore.stub import Stubber
from moto import mock_aws

import unified_kg_rag.adapters.schema_sources.file as file_source
from tests.fixtures.extras import requires_ontology
from tests.fixtures.ontology_models import PROCURE_GOLDEN
from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
)
from unified_kg_rag.adapters.source_files.locations import SourceReader
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.cli import preflight
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.application.config import ConfigLoader
from unified_kg_rag.domain.ingestion.cache_keys import (
    extraction_fingerprint,
    stage_input_fingerprint,
)
from unified_kg_rag.domain.models import Config, PipelineStageType, SchemaDocument
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.shared.exceptions import SchemaSourceError

pytestmark = pytest.mark.unit

REGION = "us-east-1"
BUCKET = "schema-bucket"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "ontologies"
PROCURE = FIXTURES / "procure.ttl"
PREFIXES = "@prefix ex: <https://example.org/procure#> .\n"
SCHEMA_YAML = """\
relationship_types:
  - "WORKS_FOR: Employment"
relationship_patterns:
  - "PERSON -WORKS_FOR-> ORGANIZATION"
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any socket connection: every S3 call is served by moto or a
    Stubber."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a schema source test reached for the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


@pytest.fixture
def s3() -> Iterator[Any]:
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


def _session() -> boto3.Session:
    return boto3.Session(region_name=REGION)


def _run_config(**extraction: Any) -> Config:
    return Config.model_validate(
        {
            "aws": {"region_name": REGION},
            "processing": {"graph_extraction": extraction},
        }
    )


def _context(kind: str, path: str, **options: Any) -> SchemaSourceContext:
    return SchemaSourceContext(
        SchemaSourceConfig(kind=kind, path=path, options=options),
        _run_config(),
        session_factory=_session,
    )


def _load(kind: str, path: str, **options: Any) -> SchemaDocument:
    return get_schema_source(kind)(_context(kind, path, **options))


def _put(s3: Any, key: str, body: str | bytes) -> str:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body.encode("utf-8") if isinstance(body, str) else body,
    )
    return f"s3://{BUCKET}/{key}"


def _put_split(s3: Any, prefix: str) -> None:
    """procure.ttl as two objects (classes, properties) under ``prefix``."""
    text = PROCURE.read_text(encoding="utf-8")
    marker = text.index("ex:tradesWith")
    declarations = "".join(
        line for line in text.splitlines(keepends=True) if line.startswith("@prefix")
    )
    _put(s3, f"{prefix}classes.ttl", text[:marker])
    _put(s3, f"{prefix}nested/properties.ttl", declarations + text[marker:])


def _keys(config: Config) -> dict[str, str]:
    return {
        stage.value: stage_input_fingerprint(config, stage, "corpus")
        for stage in PipelineStageType
    }


def _stubbed(context: SchemaSourceContext) -> Stubber:
    """A Stubber on the client the loader of ``context`` will use."""
    client = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # nosec B106 - placeholder
    )
    context.client = lambda service_name, **kwargs: client  # type: ignore[method-assign]
    return Stubber(client)


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(data), len(data))


# --- one object ---------------------------------------------------------------------


def test_file_reads_an_s3_object_as_a_local_file(s3: Any, tmp_path: Path) -> None:
    location = _put(s3, "schemas/schema.yaml", SCHEMA_YAML)
    local = tmp_path / "schema.yaml"
    local.write_text(SCHEMA_YAML, encoding="utf-8")
    from_s3 = _load("file", location)
    assert from_s3 == get_schema_source("file")(
        SchemaSourceContext(
            SchemaSourceConfig(kind="file", path=str(local)), _run_config()
        )
    )
    assert [str(t) for t in from_s3.relationship_types or []] == [
        "WORKS_FOR: Employment"
    ]
    json_location = _put(
        s3, "schemas/schema.json", '{"relationship_types": ["PART_OF"]}'
    )
    assert [str(t) for t in _load("file", json_location).relationship_types or []] == [
        "PART_OF"
    ]


@requires_ontology
def test_owl_reads_an_s3_object(s3: Any) -> None:
    location = _put(s3, "onto/procure.ttl", PROCURE.read_bytes())
    assert _load("owl", location) == PROCURE_GOLDEN
    rdf = _put(s3, "onto/procure.rdf", (FIXTURES / "procure.rdf").read_bytes())
    assert _load("owl", rdf) == PROCURE_GOLDEN


@pytest.mark.parametrize(
    ("key", "body", "message"),
    [
        ("schema.txt", SCHEMA_YAML, r"must end in \.yaml"),
        ("schema.yaml", b"\xff\xfe", "is not valid UTF-8"),
        ("schema.yaml", "- a list\n", "must hold a mapping"),
        ("schema.yaml", "relationship_types: [1, [2]]\n", "Invalid schema file"),
    ],
)
def test_file_refuses_a_bad_s3_object_naming_it(
    s3: Any, key: str, body: str | bytes, message: str
) -> None:
    location = _put(s3, key, body)
    with pytest.raises(SchemaSourceError, match=message) as error:
        _load("file", location)
    assert location in str(error.value)


def test_file_refuses_an_s3_prefix(s3: Any) -> None:
    _put(s3, "schemas/schema.yaml", SCHEMA_YAML)
    for location in (f"s3://{BUCKET}/schemas/", f"s3://{BUCKET}"):
        with pytest.raises(SchemaSourceError, match="names an S3 prefix"):
            _load("file", location)


# --- a prefix as a directory -----------------------------------------------------------


@requires_ontology
def test_owl_reads_a_prefix_as_a_directory(s3: Any) -> None:
    _put_split(s3, "onto/")
    # As in a local directory: other files, and names starting with ".",
    # are skipped; so are folder markers and keys beside the prefix.
    _put(s3, "onto/README.md", "# notes\n")
    _put(s3, "onto/.draft.ttl", "not turtle at all")
    _put(s3, "onto/.git/x.ttl", "not turtle either")
    _put(s3, "onto/empty/", b"")
    _put(s3, "onto-old/x.ttl", "not turtle either")
    assert _load("owl", f"s3://{BUCKET}/onto/") == PROCURE_GOLDEN


@requires_ontology
def test_every_page_of_the_listing_is_read(s3: Any) -> None:
    # Two keys a page (S3 returns up to 1,000; MaxKeys is set on each
    # request here, so no thousand objects are written): the notes fill the
    # first page and the ontology files sort after it.
    for number in range(2):
        _put(s3, f"onto/a-notes/{number:04}.txt", b"")
    _put_split(s3, "onto/z/")
    keys = s3.list_objects_v2(Bucket=BUCKET, Prefix="onto/")["KeyCount"]
    context = _context("owl", f"s3://{BUCKET}/onto/")
    client = boto3.client("s3", region_name=REGION)
    calls: list[str | None] = []

    def two_a_page(params: dict[str, Any], **kwargs: Any) -> None:
        calls.append(params.get("ContinuationToken"))
        params["MaxKeys"] = 2

    client.meta.events.register("provide-client-params.s3.ListObjectsV2", two_a_page)
    context.client = lambda service_name, **kwargs: client  # type: ignore[method-assign]
    assert get_schema_source("owl")(context) == PROCURE_GOLDEN
    assert len(calls) == -(-keys // 2) >= 2
    assert calls[0] is None and all(calls[1:])


@requires_ontology
def test_the_bucket_root_is_a_directory(s3: Any) -> None:
    _put(s3, "procure.ttl", PROCURE.read_bytes())
    assert _load("owl", f"s3://{BUCKET}/") == PROCURE_GOLDEN
    assert _load("owl", f"s3://{BUCKET}") == PROCURE_GOLDEN


def test_an_empty_prefix_holds_no_ontology_file(s3: Any) -> None:
    _put(s3, "onto/README.md", "# notes\n")
    for location in (f"s3://{BUCKET}/onto/", f"s3://{BUCKET}/absent/"):
        with pytest.raises(SchemaSourceError, match="holds no ontology file") as error:
            _load("owl", location)
        assert location in str(error.value)


def test_a_refused_format_under_a_prefix_is_refused(s3: Any) -> None:
    _put(s3, "onto/procure.ttl", PROCURE.read_bytes())
    _put(s3, "onto/procure.owx", "<Ontology/>")
    with pytest.raises(SchemaSourceError, match="is OWL/XML.*robot convert"):
        _load("owl", f"s3://{BUCKET}/onto/")


# --- the size cap ------------------------------------------------------------------


def _no_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: SourceReader, *args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("GetObject before the size check")

    monkeypatch.setattr(SourceReader, "read", refuse)


def test_file_checks_the_size_before_fetching(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _put(s3, "schema.yaml", SCHEMA_YAML)
    monkeypatch.setattr(file_source, "MAX_S3_BYTES", len(SCHEMA_YAML) - 1)
    _no_get(monkeypatch)
    with pytest.raises(
        SchemaSourceError, match=f"holds {len(SCHEMA_YAML)} bytes, more than the"
    ):
        _load("file", location)


def test_owl_checks_the_size_before_fetching(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = PROCURE.stat().st_size
    one = _put(s3, "onto/procure.ttl", PROCURE.read_bytes())
    _put(s3, "onto/copy.ttl", PROCURE.read_bytes())
    _no_get(monkeypatch)
    with pytest.raises(SchemaSourceError, match=f"holds {size} bytes, more than"):
        _load("owl", one, max_bytes=size - 1)
    with pytest.raises(SchemaSourceError, match=f"holds {2 * size} bytes"):
        _load("owl", f"s3://{BUCKET}/onto/", max_bytes=2 * size - 1)


def test_an_object_that_grew_since_it_was_sized_is_refused() -> None:
    context = _context("file", f"s3://{BUCKET}/schema.yaml")
    stubber = _stubbed(context)
    params = {"Bucket": BUCKET, "Key": "schema.yaml"}
    stubber.add_response("head_object", {"ContentLength": 10}, params)
    stubber.add_response(
        "get_object", {"Body": _body(b"x" * (file_source.MAX_S3_BYTES + 5))}, params
    )
    with stubber, pytest.raises(SchemaSourceError, match="grew since it was sized"):
        get_schema_source("file")(context)


# --- S3 errors ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["file", "owl"])
def test_a_missing_object_is_not_found(s3: Any, kind: str) -> None:
    location = f"s3://{BUCKET}/onto/absent.yaml"
    if kind == "owl":
        location = f"s3://{BUCKET}/onto"
    with pytest.raises(SchemaSourceError, match="not found .no such S3 object") as e:
        _load(kind, location)
    # Only owl reads a prefix as a directory: the hint would mislead file.
    assert ("end the path with '/'" in str(e.value)) == (kind == "owl")


def test_a_missing_bucket_is_named(s3: Any) -> None:
    with pytest.raises(SchemaSourceError, match="the S3 bucket does not exist"):
        _load("owl", "s3://absent-bucket/onto/")


@pytest.mark.parametrize(
    ("kind", "path", "operation", "code", "status"),
    [
        ("file", "schema.yaml", "head_object", "403", 403),
        ("owl", "onto.ttl", "head_object", "403", 403),
        ("owl", "onto/", "list_objects_v2", "AccessDenied", 403),
    ],
)
def test_access_denied_says_which_permission(
    kind: str, path: str, operation: str, code: str, status: int
) -> None:
    context = _context(kind, f"s3://{BUCKET}/{path}")
    stubber = _stubbed(context)
    stubber.add_client_error(
        operation,
        service_error_code=code,
        service_message="Access Denied",
        http_status_code=status,
        response_meta={"RequestId": "SYNTHETICREQUESTID"},
    )
    with stubber, pytest.raises(SchemaSourceError) as error:
        get_schema_source(kind)(context)
    message = str(error.value)
    assert f"Access to schema source 's3://{BUCKET}/{path}' denied" in message
    assert "s3:GetObject" in message and "s3:ListBucket" in message
    assert "SYNTHETICREQUESTID" not in message
    stubber.assert_no_pending_responses()


def test_get_object_denied_is_named() -> None:
    context = _context("file", f"s3://{BUCKET}/schema.yaml")
    stubber = _stubbed(context)
    stubber.add_response("head_object", {"ContentLength": 10})
    stubber.add_client_error(
        "get_object", service_error_code="AccessDenied", http_status_code=403
    )
    with stubber, pytest.raises(SchemaSourceError, match=r"denied \(GetObject\)"):
        get_schema_source("file")(context)


def test_another_s3_error_is_named_by_its_code() -> None:
    context = _context("file", f"s3://{BUCKET}/schema.yaml")
    stubber = _stubbed(context)
    stubber.add_client_error(
        "head_object", service_error_code="SlowDown", http_status_code=503
    )
    with (
        stubber,
        pytest.raises(SchemaSourceError, match=r"from S3 \(HeadObject\): SlowDown"),
    ):
        get_schema_source("file")(context)


def test_an_unreachable_s3_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context("file", f"s3://{BUCKET}/schema.yaml")
    stubber = _stubbed(context)
    client = context.client("s3")

    def unreachable(**kwargs: Any) -> Any:
        raise EndpointConnectionError(endpoint_url="https://s3.example.invalid")

    monkeypatch.setattr(client, "head_object", unreachable)
    with (
        stubber,
        pytest.raises(
            SchemaSourceError, match="from S3 .HeadObject.: EndpointConnectionError"
        ),
    ):
        get_schema_source("file")(context)


@pytest.mark.parametrize(
    "path", ["s3://Not_A_Bucket/x.yaml", "s3://user:token@bucket/x.yaml", "s3:///x"]
)
def test_an_s3_url_without_a_valid_bucket_is_refused(path: str) -> None:
    for kind in ("file", "owl"):
        with pytest.raises(SchemaSourceError, match="does not name an S3 bucket") as e:
            _load(kind, path)
        assert "token" not in str(e.value)


@pytest.mark.parametrize(
    "path", ["https://example.org/onto.ttl", "gs://bucket/onto.ttl", "s3a://b/o.ttl"]
)
def test_other_url_schemes_stay_refused(path: str) -> None:
    for kind in ("file", "owl"):
        with pytest.raises(SchemaSourceError, match="is a URL; kind"):
            _load(kind, path)


def test_a_local_source_builds_no_session(tmp_path: Path) -> None:
    local = tmp_path / "schema.yaml"
    local.write_text(SCHEMA_YAML, encoding="utf-8")

    def refuse() -> boto3.Session:
        raise AssertionError("a local schema source built an AWS session")

    get_schema_source("file")(
        SchemaSourceContext(
            SchemaSourceConfig(kind="file", path=str(local)),
            _run_config(),
            session_factory=refuse,
        )
    )


# --- start-up, preflight and the cache keys -------------------------------------------


def test_preflight_makes_no_s3_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("preflight built an AWS session")

    monkeypatch.setattr(boto3, "Session", refuse)
    file_config = _run_config(
        schema_source={"kind": "file", "path": f"s3://{BUCKET}/schema.yaml"}
    )
    assert missing_extras_error(file_config) is None
    owl_config = _run_config(schema_source={"kind": "owl", "path": f"s3://{BUCKET}/"})
    # Only the extra is checked: named when it is missing, else nothing.
    monkeypatch.setattr(preflight, "_installed", lambda module: module != "rdflib")
    assert "rdflib" in (missing_extras_error(owl_config) or "")
    monkeypatch.setattr(preflight, "_installed", lambda module: True)
    assert missing_extras_error(owl_config) is None


def test_an_s3_path_is_left_as_written(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "processing:\n  graph_extraction:\n    schema_source:\n"
        f"      kind: owl\n      path: s3://{BUCKET}/onto/\n",
        encoding="utf-8",
    )
    loaded = ConfigLoader(config_file).load_config()
    assert loaded.processing.graph_extraction.schema_source.path == (
        f"s3://{BUCKET}/onto/"
    )


def test_the_content_is_keyed_not_the_s3_location(s3: Any, tmp_path: Path) -> None:
    local = tmp_path / "schema.yaml"
    local.write_text(SCHEMA_YAML, encoding="utf-8")
    here = _put(s3, "a/schema.yaml", SCHEMA_YAML)
    there = _put(s3, "moved/b.yaml", SCHEMA_YAML)
    resolved = [
        resolve_extraction_schema(
            _run_config(schema_source={"kind": "file", "path": path}),
            boto_session=_session(),
        )
        for path in (str(local), here, there)
    ]
    assert len({extraction_fingerprint(config) for config in resolved}) == 1
    assert _keys(resolved[1]) == _keys(resolved[0]) == _keys(resolved[2])
    edited = _put(s3, "a/schema.yaml", SCHEMA_YAML.replace("Employment", "Jobs"))
    changed = resolve_extraction_schema(
        _run_config(schema_source={"kind": "file", "path": edited}),
        boto_session=_session(),
    )
    assert extraction_fingerprint(changed) != extraction_fingerprint(resolved[0])


@requires_ontology
def test_an_ontology_moved_to_s3_keys_nothing(s3: Any) -> None:
    local = resolve_extraction_schema(
        _run_config(schema_source={"kind": "owl", "path": str(PROCURE)})
    )
    location = _put(s3, "elsewhere/ontology.ttl", PROCURE.read_bytes())
    moved = resolve_extraction_schema(
        _run_config(schema_source={"kind": "owl", "path": location}),
        boto_session=_session(),
    )
    assert _keys(moved) == _keys(local)
    assert extraction_fingerprint(moved) == extraction_fingerprint(local)


@requires_ontology
def test_relative_iris_resolve_against_the_s3_url(
    s3: Any, caplog: pytest.LogCaptureFixture
) -> None:
    location = _put(
        s3,
        "onto/o.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<#Vendor> a owl:Class .\n",
    )
    with caplog.at_level(logging.WARNING):
        document = _load("owl", location)
    assert document.entity_types[0].iri == location + "#Vendor"  # type: ignore[index,union-attr]
    assert any("relative to their file" in r.getMessage() for r in caplog.records)
    # With @base the IRIs are the declared ones, wherever the object is.
    based = _put(
        s3,
        "onto/based.ttl",
        "@base <https://example.org/procure> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<#Vendor> a owl:Class .\n",
    )
    document = _load("owl", based)
    assert document.entity_types[0].iri == "https://example.org/procure#Vendor"  # type: ignore[index,union-attr]


# --- the listing and the reads in detail -------------------------------------------


@requires_ontology
def test_a_prefix_starting_with_a_dot_is_not_hidden(s3: Any) -> None:
    # Only the names below the prefix are checked for a leading ".".
    _put_split(s3, ".onto/")
    _put(s3, ".onto/nested/.cache/x.ttl", "not turtle at all")
    assert _load("owl", f"s3://{BUCKET}/.onto/") == PROCURE_GOLDEN


@requires_ontology
def test_folder_markers_are_no_files_even_with_a_format(s3: Any) -> None:
    _put(s3, "onto/procure.rdf", (FIXTURES / "procure.rdf").read_bytes())
    _put(s3, "onto/empty/", b"")
    assert _load("owl", f"s3://{BUCKET}/onto/", format="xml") == PROCURE_GOLDEN


@requires_ontology
def test_the_listing_is_read_in_key_order() -> None:
    context = _context("owl", f"s3://{BUCKET}/onto/")
    stubber = _stubbed(context)
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [
                {"Key": "onto/b.ttl", "Size": 3},
                {"Key": "onto/a.ttl", "Size": 3},
            ]
        },
        {"Bucket": BUCKET, "Prefix": "onto/"},
    )
    stubber.add_response(
        "get_object", {"Body": _body(b"a a")}, {"Bucket": BUCKET, "Key": "onto/a.ttl"}
    )
    with (
        stubber,
        pytest.raises(SchemaSourceError, match=f"'s3://{BUCKET}/onto/a.ttl' does not"),
    ):
        get_schema_source("owl")(context)


@requires_ontology
def test_one_client_serves_every_call(s3: Any) -> None:
    _put_split(s3, "onto/")
    context = _context("owl", f"s3://{BUCKET}/onto/")
    client = boto3.client("s3", region_name=REGION)
    built: list[str] = []

    def build(service_name: str, **kwargs: Any) -> Any:
        built.append(service_name)
        return client

    context.client = build  # type: ignore[method-assign]
    assert get_schema_source("owl")(context) == PROCURE_GOLDEN
    assert built == ["s3"]


class _CountingBody(StreamingBody):
    """A GetObject body that records how many bytes each read asks for."""

    def __init__(self, data: bytes) -> None:
        super().__init__(io.BytesIO(data), len(data))
        self.asked: list[int | None] = []

    def read(self, amt: int | None = None) -> bytes:
        self.asked.append(amt)
        return super().read(amt)


@pytest.mark.parametrize("kind", ["file", pytest.param("owl", marks=requires_ontology)])
def test_get_object_reads_one_byte_past_the_cap_at_most(kind: str) -> None:
    key = "schema.yaml" if kind == "file" else "onto.ttl"
    data = SCHEMA_YAML.encode() if kind == "file" else PROCURE.read_bytes()
    cap = file_source.MAX_S3_BYTES if kind == "file" else 2 * len(data)
    options = {"max_bytes": cap} if kind == "owl" else {}
    context = _context(kind, f"s3://{BUCKET}/{key}", **options)
    stubber = _stubbed(context)
    params = {"Bucket": BUCKET, "Key": key}
    stubber.add_response("head_object", {"ContentLength": len(data)}, params)
    body = _CountingBody(data)
    stubber.add_response("get_object", {"Body": body}, params)
    with stubber:
        get_schema_source(kind)(context)
    assert body.asked == [cap + 1]


@requires_ontology
def test_an_owl_object_that_grew_since_it_was_sized_is_refused() -> None:
    context = _context("owl", f"s3://{BUCKET}/onto.ttl", max_bytes=100)
    stubber = _stubbed(context)
    params = {"Bucket": BUCKET, "Key": "onto.ttl"}
    stubber.add_response("head_object", {"ContentLength": 10}, params)
    stubber.add_response("get_object", {"Body": _body(b"#" * 101)}, params)
    with stubber, pytest.raises(SchemaSourceError, match="grew since it was sized"):
        get_schema_source("owl")(context)


def test_an_object_of_exactly_the_cap_is_read(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _put(s3, "schema.yaml", SCHEMA_YAML)
    monkeypatch.setattr(file_source, "MAX_S3_BYTES", len(SCHEMA_YAML))
    assert [str(t) for t in _load("file", location).relationship_types or []] == [
        "WORKS_FOR: Employment"
    ]


@pytest.mark.parametrize("scheme", ["S3://", "s3://"])
def test_the_s3_scheme_is_read_in_any_case(s3: Any, scheme: str) -> None:
    _put(s3, "schemas/schema.yaml", SCHEMA_YAML)
    document = _load("file", f"{scheme}{BUCKET}/schemas/schema.yaml")
    assert [str(t) for t in document.relationship_types or []] == [
        "WORKS_FOR: Employment"
    ]


def test_any_403_is_access_denied() -> None:
    # Whatever the code: S3 answers a HEAD without one, and others vary.
    context = _context("file", f"s3://{BUCKET}/schema.yaml")
    stubber = _stubbed(context)
    stubber.add_client_error(
        "head_object", service_error_code="InvalidAccessKeyId", http_status_code=403
    )
    with stubber, pytest.raises(SchemaSourceError, match=r"denied \(HeadObject\)"):
        get_schema_source("file")(context)


def test_an_s3_error_never_shows_the_query(s3: Any) -> None:
    # The key is taken as written, "?" and all; the message drops the query.
    location = f"s3://{BUCKET}/onto/absent.ttl?X-Amz-Security-Token=SECRET#frag"
    with pytest.raises(SchemaSourceError, match="not found") as error:
        _load("owl", location)
    assert "SECRET" not in str(error.value)
    assert f"'s3://{BUCKET}/onto/absent.ttl' not found" in str(error.value)
