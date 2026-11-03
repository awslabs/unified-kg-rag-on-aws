# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed sources in S3: kinds ``jsonl`` and ``csv`` with an ``s3://`` path.

Against moto-mocked S3 (botocore Stubber where the calls made must be
pinned): one object, a prefix read as a directory (every page of the
listing, filtered and sorted as a local directory is), the sizes checked
before any byte is fetched, and the S3 errors as ``SeedGraphError``s naming
the seed. A seed in S3 converts as the same seed on disk.
"""

from __future__ import annotations

import io
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from moto import mock_aws

from tests.fixtures.seed_graphs import (
    EDGES,
    EDGES_CSV,
    NODES,
    NODES_CSV,
    PROCURE,
    lines,
)
from unified_kg_rag.adapters.seed_sources import (
    SeedSourceContext,
    get_seed_source,
    jsonl,
)
from unified_kg_rag.application.bootstrap import load_seed_graphs
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import SeedGraphConfig
from unified_kg_rag.domain.models.seed_graph import SeedGraph
from unified_kg_rag.shared.exceptions import SeedGraphError

pytestmark = pytest.mark.unit

REGION = "us-east-1"
BUCKET = "seed-bucket"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any socket connection: every S3 call is served by moto or a
    Stubber."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a seed source test reached for the network")

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


def _seed(path: str, kind: str = "jsonl", **settings: Any) -> SeedGraphConfig:
    return SeedGraphConfig(name="procure", kind=kind, path=path, **settings)


def _run_config(*seeds: SeedGraphConfig) -> Config:
    return Config.model_validate(
        {
            "aws": {"region_name": REGION},
            "processing": {"seed_graphs": [s.model_dump() for s in seeds]},
        }
    )


def _context(path: str, kind: str = "jsonl", **settings: Any) -> SeedSourceContext:
    return SeedSourceContext(
        _seed(path, kind, **settings), _run_config(), session_factory=_session
    )


def _load(path: str, kind: str = "jsonl", **settings: Any) -> SeedGraph:
    return get_seed_source(kind)(_context(path, kind, **settings))


def _put(s3: Any, key: str, body: str | bytes) -> str:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body.encode("utf-8") if isinstance(body, str) else body,
    )
    return f"s3://{BUCKET}/{key}"


def _shape(graph: SeedGraph) -> tuple[Any, Any]:
    return graph.nodes, graph.edges


def test_jsonl_reads_an_s3_object(s3: Any) -> None:
    location = _put(s3, "seeds/procure.jsonl", lines(*NODES, *EDGES))
    graph = _load(location)
    assert _shape(graph) == _shape(PROCURE)
    assert graph.report.files == 1


def test_jsonl_reads_an_s3_prefix_sorted_and_filtered(s3: Any) -> None:
    _put(s3, "seeds/b/edges.jsonl", lines(*EDGES))
    _put(s3, "seeds/a-nodes.jsonl", lines(NODES[1]))
    _put(s3, "seeds/a/nodes.jsonl", lines(NODES[0]))
    _put(s3, "seeds/.hidden/x.jsonl", "not json\n")
    _put(s3, "seeds/notes.txt", "not json\n")
    _put(s3, "seeds/sub/", "")
    _put(s3, "elsewhere.jsonl", "not json\n")
    graph = _load(f"s3://{BUCKET}/seeds/")
    assert [node.id for node in graph.nodes] == ["b1", "v1"]
    assert graph.edges == PROCURE.edges
    assert graph.report.files == 3


def test_jsonl_reads_every_page_of_a_listing(s3: Any, mocker: Any) -> None:
    for i in range(3):
        _put(s3, f"seeds/n{i}.jsonl", lines({"kind": "node", "id": f"n{i}"}))
    real = boto3.client

    def paged(*args: Any, **kwargs: Any) -> Any:
        client = real(*args, **kwargs)
        paginate = client.get_paginator("list_objects_v2").paginate
        client.get_paginator = lambda name: mocker.Mock(  # type: ignore[method-assign]
            paginate=lambda **kw: paginate(**kw, PaginationConfig={"PageSize": 1})
        )
        return client

    session = _session()
    mocker.patch.object(session, "client", side_effect=paged)
    context = SeedSourceContext(
        _seed(f"s3://{BUCKET}/seeds/"), _run_config(), session_factory=lambda: session
    )
    assert [n.id for n in get_seed_source("jsonl")(context).nodes] == ["n0", "n1", "n2"]


def test_csv_reads_an_s3_prefix(s3: Any) -> None:
    _put(s3, "csv/nodes.csv", NODES_CSV)
    _put(s3, "csv/edges.csv", EDGES_CSV)
    assert _shape(_load(f"s3://{BUCKET}/csv/", "csv")) == _shape(PROCURE)
    with pytest.raises(SeedGraphError, match="names an S3 object; kind 'csv' reads"):
        _load(f"s3://{BUCKET}/csv/nodes.csv", "csv")


def test_a_seed_in_s3_converts_as_the_same_seed_on_disk(
    s3: Any, tmp_path: Path
) -> None:
    text = lines(*NODES, *EDGES)
    local = tmp_path / "procure.jsonl"
    local.write_text(text, encoding="utf-8")
    remote = _put(s3, "seeds/procure.jsonl", text)
    (from_s3,) = load_seed_graphs(_run_config(_seed(remote)), boto_session=_session())
    (from_disk,) = load_seed_graphs(_run_config(_seed(str(local))))
    assert from_s3.digest == from_disk.digest


def _stubbed(context: SeedSourceContext) -> Stubber:
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


def test_an_s3_object_over_max_bytes_is_never_fetched() -> None:
    context = _context(f"s3://{BUCKET}/seeds/procure.jsonl", max_bytes=10)
    stubber = _stubbed(context)
    stubber.add_response(
        "head_object",
        {"ContentLength": 11},
        {"Bucket": BUCKET, "Key": "seeds/procure.jsonl"},
    )
    with stubber, pytest.raises(SeedGraphError, match="hold 11 bytes, more than"):
        get_seed_source("jsonl")(context)
    stubber.assert_no_pending_responses()


def test_an_s3_object_that_grew_since_it_was_sized_is_refused() -> None:
    context = _context(f"s3://{BUCKET}/seeds/procure.jsonl", max_bytes=10)
    stubber = _stubbed(context)
    stubber.add_response(
        "head_object",
        {"ContentLength": 10},
        {"Bucket": BUCKET, "Key": "seeds/procure.jsonl"},
    )
    stubber.add_response(
        "get_object",
        {"Body": _body(b"x" * 20)},
        {"Bucket": BUCKET, "Key": "seeds/procure.jsonl"},
    )
    # Read within the cap: the object is never fetched whole.
    with (
        stubber,
        pytest.raises(
            SeedGraphError, match="procure.jsonl' holds more than 10 bytes .it grew"
        ),
    ):
        get_seed_source("jsonl")(context)


def test_an_s3_prefix_reads_each_object_within_the_bytes_left(
    s3: Any, mocker: Any
) -> None:
    # Sized within the cap, the second object is read with what the first
    # left of it: one byte too few.
    first = _put(s3, "seeds/a.jsonl", lines(*NODES))
    _put(s3, "seeds/b.jsonl", lines(*EDGES))
    cap = len(lines(*NODES)) + len(lines(*EDGES)) - 1
    mocker.patch.object(jsonl, "check_total")
    with pytest.raises(
        SeedGraphError,
        match=f"b.jsonl' holds more than {len(lines(*EDGES)) - 1} bytes",
    ):
        _load(first.rsplit("/", 1)[0] + "/", max_bytes=cap)


def test_s3_prefix_files_over_max_bytes_are_never_fetched(s3: Any, mocker: Any) -> None:
    _put(s3, "csv/nodes.csv", NODES_CSV)
    _put(s3, "csv/edges.csv", EDGES_CSV)
    total = len(NODES_CSV) + len(EDGES_CSV)
    context = _context(f"s3://{BUCKET}/csv/", "csv", max_bytes=total - 1)
    client = context.client("s3")
    get = mocker.spy(client, "get_object")
    context.client = lambda service_name, **kwargs: client  # type: ignore[method-assign]
    with pytest.raises(SeedGraphError, match=f"hold {total} bytes"):
        get_seed_source("csv")(context)
    get.assert_not_called()


@pytest.mark.parametrize(
    ("path", "kind", "message"),
    [
        (f"s3://{BUCKET}/missing.jsonl", "jsonl", "not found .*end the path with '/'"),
        (f"s3://{BUCKET}/empty/", "jsonl", "holds no .jsonl object"),
        (f"s3://{BUCKET}/csv/", "csv", "csv/nodes.csv' not found"),
        ("s3://missing-bucket/seed.jsonl", "jsonl", "bucket does not exist"),
    ],
)
def test_s3_errors_name_the_seed_and_what_to_fix(
    s3: Any, path: str, kind: str, message: str
) -> None:
    _put(s3, "csv/edges.csv", EDGES_CSV)
    with pytest.raises(SeedGraphError, match=message) as error:
        _load(path, kind)
    assert str(error.value).startswith("seed graph 'procure': ")


def test_access_denied_says_which_permissions_the_role_needs() -> None:
    context = _context(f"s3://{BUCKET}/seeds/procure.jsonl")
    stubber = _stubbed(context)
    stubber.add_client_error(
        "head_object", service_error_code="403", http_status_code=403
    )
    with stubber, pytest.raises(SeedGraphError, match="s3:GetObject .* s3:ListBucket"):
        get_seed_source("jsonl")(context)
