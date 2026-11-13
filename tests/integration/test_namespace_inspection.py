# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``describe_namespace``, ``export_graph`` and ``run-describe`` on a
namespace the embedded stores hold (no container, no AWS)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import botocore.session
import networkx as nx
import pytest
from botocore.stub import Stubber

from tests.integration import test_embedded_e2e as embedded
from tests.integration.test_document_changes_api import _ok, setup  # noqa: F401
from unified_kg_rag import describe_namespace, export_graph
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.application.cli import run_describe
from unified_kg_rag.application.storage import inspection
from unified_kg_rag.shared.exceptions import ConfigurationError, ToolArgumentError

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

# The fixture that patches every AWS client to fail.
no_aws = embedded.no_aws

KOREAN = "공급자는 Vendor 이며 Buyer 에게 납품한다."


@pytest.fixture
def ingested(setup):  # noqa: F811 - the fixture
    config, providers, pipeline = setup
    _ok(
        pipeline.insert_documents(
            [
                {
                    "id": "contracts/supply.txt",
                    "text": "Vendor supplies widgets to Buyer.",
                    "filters": {"category": "contract"},
                },
                {"id": "ko/공급.txt", "text": KOREAN},
                {"id": "ticket-1", "text": "Depot stores widgets for Vendor."},
            ]
        )
    )
    return config, providers


def test_describe_names_what_the_namespace_holds(ingested) -> None:
    config, providers = ingested

    description = describe_namespace(config, providers=providers)

    assert description.exists and description.suffix == "default"
    assert description.artifacts_from == "build_record"
    assert description.methodologies == ["graphrag", "lightrag", "hipporag"]
    assert {"local", "mix", "global", "ppr"} <= set(description.strategies)
    assert "tree" not in description.strategies
    indices = {index.kind: index for index in description.indices}
    assert indices["text_units"].documents == 3
    assert indices["entities"].documents == 3
    assert indices["entities"].embedding_model_id == "synthetic-embed"
    assert indices["summary_nodes"].index is None
    assert description.build is not None
    assert description.build["documents"] == 3
    assert description.registry == {"processed": 3}


def test_describe_a_namespace_that_holds_nothing(ingested) -> None:
    config, providers = ingested
    description = describe_namespace(config, "tenant_x", providers=providers)
    assert not description.exists
    assert description.artifacts_from == "config"
    # It holds nothing: not the methodologies the config builds.
    assert (description.methodologies, description.strategies) == ([], [])
    assert (description.artifacts, description.auto_routes) == ([], [])
    with pytest.raises(ConfigurationError, match="suffix"):
        describe_namespace(config, "../x", providers=providers)


def test_jsonl_export_streams_every_kind_with_provenance(
    ingested, tmp_path: Path
) -> None:
    config, providers = ingested
    path = tmp_path / "out" / "graph.jsonl"

    exported = export_graph(config, path, providers=providers, page_size=1)

    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["kind"], []).append(record)
    assert exported.counts["text_units"] == 3 and exported.counts["entities"] == 3
    assert {r["documents"][0] for r in by_type["text"]} == {
        "contracts/supply.txt",
        "ko/공급.txt",
        "ticket-1",
    }
    # Text is left out unless asked for; no vector is written.
    assert all("text" not in r for r in by_type["text"])
    assert not [k for r in records for k in r if k.endswith("_embedding")]
    vendor = next(r for r in by_type["entity"] if r["name"] == "Vendor")
    assert vendor["type"] == "organization"
    assert set(vendor["documents"]) == {
        "contracts/supply.txt",
        "ko/공급.txt",
        "ticket-1",
    }
    assert by_type["relationship"] and all(
        r["documents"] for r in by_type["relationship"]
    )
    assert "community" in by_type
    # Korean is written as text, not escaped.
    assert "ko/공급.txt" in path.read_text("utf-8")


def test_graphml_export_reads_back_as_a_graph(ingested, tmp_path: Path) -> None:
    config, providers = ingested
    path = tmp_path / "graph.graphml"

    export_graph(config, path, format="graphml", providers=providers)

    graph = nx.read_graphml(path)
    entities = {n for n, d in graph.nodes(data=True) if d.get("kind") == "entity"}
    names = {graph.nodes[n]["name"] for n in entities}
    assert names == {"Vendor", "Buyer", "Depot"}
    assert any(d.get("kind") == "community" for _, d in graph.nodes(data=True))
    relationship_edges = [
        (u, v) for u, v, d in graph.edges(data=True) if d.get("type") != "parent_of"
    ]
    assert relationship_edges and all(
        u in entities and v in entities for u, v in relationship_edges
    )
    vendor = next(n for n in entities if graph.nodes[n]["name"] == "Vendor")
    assert "ticket-1" in json.loads(graph.nodes[vendor]["documents"])


def test_max_items_bounds_each_kind(ingested, tmp_path: Path) -> None:
    config, providers = ingested
    exported = export_graph(
        config, tmp_path / "g.jsonl", providers=providers, max_items=1, page_size=1
    )
    assert exported.counts["entities"] == 1
    assert "entities" in exported.truncated
    with pytest.raises(ConfigurationError, match="format"):
        export_graph(config, tmp_path / "g.x", providers=providers, format="csv")  # type: ignore[arg-type]


def test_run_describe_prints_one_json_document(
    ingested, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config, providers = ingested
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    monkeypatch.setattr(inspection, "build_providers", lambda _c: providers)
    out = tmp_path / "graph.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-describe", "--output-format", "json", "--output-path", str(out)],
    )

    assert run_describe.main() == 0

    document = json.loads(capsys.readouterr().out)
    assert document["methodologies"] == ["graphrag", "lightrag", "hipporag"]
    assert document["export"]["counts"]["entities"] == 3
    assert out.is_file()
    # A namespace without an index is a usage error (exit 2), of the type
    # run-rag names it by (it said UsageError).
    monkeypatch.setattr(
        sys, "argv", ["run-describe", "--suffix", "tenant_x", "--output-format", "json"]
    )
    with pytest.raises(SystemExit) as exc:
        run_describe.main()
    assert exc.value.code == 2
    status = json.loads(capsys.readouterr().out)
    assert (status["exit_code"], status["error_type"]) == (2, "IndexNotFoundError")


def test_run_describe_names_a_wrong_suffix_on_stderr(
    ingested, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Text mode exited 2 and printed nothing.
    config, providers = ingested
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    monkeypatch.setattr(inspection, "build_providers", lambda _c: providers)
    monkeypatch.setattr(sys, "argv", ["run-describe", "--suffix", "tenant_x"])
    with pytest.raises(SystemExit) as exc:
        run_describe.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    err = " ".join(captured.err.split())
    assert "run-describe: error: the index namespace" in err
    assert "holds no index: check --suffix" in err


def test_describing_where_nothing_was_ingested_creates_no_store(
    tmp_path: Path,
) -> None:
    # It left an empty .unified-kg-rag/vector.sqlite3 (and, with the
    # registry on, doc-status.sqlite3) in the working directory.
    config = embedded._config(tmp_path / "stores", tmp_path / "viz")
    assert config.providers.doc_status.enabled
    description = describe_namespace(config)
    assert not description.exists
    assert not (tmp_path / "stores" / "vector.sqlite3").exists()
    # Nor an empty doc-status.sqlite3 with the registry on.
    assert description.registry == {}
    assert not (tmp_path / "stores" / "doc-status.sqlite3").exists()


def test_run_describe_lists_the_documents_of_a_namespace_without_an_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Regression (rv25): a first ingest whose model calls all failed left
    # two pending records and no index; run-describe --documents refused
    # it as index_not_found, the one case the runbook added it for.
    from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
    from unified_kg_rag.domain.ingestion.document_identity import (
        config_registry_namespace,
    )
    from unified_kg_rag.domain.models import DocStatus, DocStatusRecord

    config = embedded._config(tmp_path / "stores", tmp_path / "viz")
    namespace = config_registry_namespace(config.indexing, None)
    (tmp_path / "stores").mkdir(parents=True, exist_ok=True)
    store = SQLiteDocStatusStore(config)
    store.put_many(
        DocStatusRecord(
            doc_id=f"doc-{name}",
            content_hash=f"hash-{name}",
            status=DocStatus.PENDING,
            suffix=namespace,
            file_path=f"{name}.txt",
        )
        for name in ("a", "b")
    )
    store.close()
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    argv = ["run-describe", "--documents", "failed,pending", "--output-format", "json"]
    monkeypatch.setattr(sys, "argv", argv)

    assert run_describe.main() == 0
    document = json.loads(capsys.readouterr().out)
    assert document["exists"] is False
    assert document["registry"] == {"pending": 2}
    assert [d["file_path"] for d in document["documents"]] == ["a.txt", "b.txt"]
    # In text, the records under a line saying no build finished.
    monkeypatch.setattr(sys, "argv", argv[:3])
    assert run_describe.main() == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "No index" in out and "a.txt: pending" in out
    # Nothing to export without an index.
    monkeypatch.setattr(
        sys, "argv", [*argv, "--output-path", str(tmp_path / "graph.jsonl")]
    )
    with pytest.raises(SystemExit) as exc:
        run_describe.main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "index_not_found"


# The registry adapter's own constructor (no_aws forbids building one).
_DYNAMODB_INIT = DynamoDBDocStatusStore.__init__


def _allow_dynamodb_store(mocker) -> None:
    """Restore the constructor through the same ``mocker`` as ``no_aws``.

    A ``monkeypatch.setattr`` here would record no_aws's forbidding mock as
    the value to restore, and monkeypatch tears down after mocker, so every
    later test in the process would find the constructor still forbidden.
    """
    mocker.patch.object(DynamoDBDocStatusStore, "__init__", _DYNAMODB_INIT)


class _DeniedSession:
    """A boto3 session whose DynamoDB client is denied its first call, as
    the deployed query task's role is (it holds no DynamoDB grant). A
    stubbed client: nothing is sent."""

    region_name = "us-west-2"

    def client(self, service_name: str, **_: Any) -> Any:
        client = botocore.session.get_session().create_client(
            service_name,
            region_name=self.region_name,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",  # nosec B106 - a stubbed client
        )
        stubber = Stubber(client)
        self.stub(stubber)
        stubber.activate()
        return client

    def stub(self, stubber: Stubber) -> None:
        stubber.add_client_error(
            "describe_table",
            service_error_code="AccessDeniedException",
            service_message="not authorized to perform: dynamodb:DescribeTable",
        )


class _EarlierTableSession(_DeniedSession):
    """The table of an earlier release (keyed by ``doc_id`` alone)."""

    def stub(self, stubber: Stubber) -> None:
        table = {
            "Table": {
                "TableName": "old-doc-status",
                "KeySchema": [{"AttributeName": "doc_id", "KeyType": "HASH"}],
            }
        }
        # Once to see that it exists (create_table_if_missing), once to
        # check its key schema.
        for _ in range(2):
            stubber.add_response("describe_table", table)


def test_describe_reports_an_unreadable_registry_and_the_rest(
    ingested, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker, capsys
) -> None:
    config, providers = ingested
    _allow_dynamodb_store(mocker)
    monkeypatch.setattr(
        providers,
        "build_doc_status",
        lambda: DynamoDBDocStatusStore(config, boto_session=_DeniedSession()),  # type: ignore[arg-type]
    )

    description = describe_namespace(config, providers=providers)

    assert description.registry is None
    assert description.registry_unavailable is not None
    assert "AccessDeniedException" in description.registry_unavailable
    # Without the hint for an indexing run (it is in the error itself).
    assert "providers.doc_status.enabled" not in description.registry_unavailable
    assert description.methodologies == ["graphrag", "lightrag", "hipporag"]
    assert {"local", "mix"} <= set(description.strategies)

    # run-describe prints the rest and says the registry is unavailable.
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    monkeypatch.setattr(inspection, "build_providers", lambda _c: providers)
    monkeypatch.setattr(sys, "argv", ["run-describe"])
    assert run_describe.main() == 0
    printed = " ".join(capsys.readouterr().out.split())
    assert "Doc-status registry: unavailable" in printed
    assert "Strategies:" in printed


def test_describe_reports_an_earlier_registry_table_without_the_run_step(
    ingested, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    # It said "re-run with indexing.reset: true", a step for an ingestion.
    config, providers = ingested
    _allow_dynamodb_store(mocker)
    monkeypatch.setattr(
        providers,
        "build_doc_status",
        lambda: DynamoDBDocStatusStore(config, boto_session=_EarlierTableSession()),  # type: ignore[arg-type]
    )

    description = describe_namespace(config, providers=providers)

    assert description.registry is None
    assert description.registry_unavailable is not None
    assert "earlier release" in description.registry_unavailable
    assert "indexing.reset" not in description.registry_unavailable
    assert {"local", "mix"} <= set(description.strategies)


def test_an_export_names_no_absolute_host_path() -> None:
    # Regression (rv8): a text unit's documents (and every item citing it)
    # were its stored file_path, the indexing host's absolute path.
    import io

    out = io.StringIO()
    writer = inspection._JsonLines(out, include_text=False)
    units: dict[str, str] = {}
    corpus_unit = {
        "id": "tu-1",
        "attributes": {
            "file_path": "/srv/host/corpus/legal/계약서.pdf",
            "file_name": "계약서.pdf",
            "relative_path": "legal/계약서.pdf",
        },
    }
    # Without a relative path (an older index): the file name.
    older_unit = {
        "id": "tu-2",
        "attributes": {"file_path": "C:\\corpus\\memo.txt", "file_name": "memo.txt"},
    }
    inspection._write_kind(
        writer, "text_units", iter([[corpus_unit, older_unit]]), units, None
    )
    inspection._write_kind(
        writer,
        "entities",
        iter([[{"id": "e1", "text_unit_ids": ["tu-1", "tu-2"]}]]),
        units,
        None,
    )
    text = out.getvalue()
    assert "/srv/host" not in text and "C:" not in text
    first, second, entity = (json.loads(line) for line in text.splitlines())
    assert first["documents"] == ["legal/계약서.pdf"]
    assert first["attributes"]["file_path"] == "legal/계약서.pdf"
    assert second["documents"] == ["memo.txt"]
    assert entity["documents"] == ["legal/계약서.pdf", "memo.txt"]


def test_an_export_replaces_an_existing_file_only_when_asked(
    ingested, tmp_path: Path
) -> None:
    config, providers = ingested
    path = tmp_path / "graph.jsonl"
    path.write_text("precious", encoding="utf-8")
    with pytest.raises(ToolArgumentError, match="exists"):
        export_graph(config, path, providers=providers)
    assert path.read_text(encoding="utf-8") == "precious"
    export_graph(config, path, providers=providers, overwrite=True)
    assert path.read_text(encoding="utf-8").startswith("{")


def test_run_describe_refuses_to_replace_a_file_without_force(
    ingested, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config, providers = ingested
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    monkeypatch.setattr(inspection, "build_providers", lambda _c: providers)
    out = tmp_path / "graph.jsonl"
    out.write_text("precious", encoding="utf-8")
    argv = ["run-describe", "--output-path", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        run_describe.main()
    assert exc.value.code == 2
    assert "exists; pass --force" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "precious"
    monkeypatch.setattr(sys, "argv", [*argv, "--force"])
    assert run_describe.main() == 0
    assert out.read_text(encoding="utf-8").startswith("{")


def test_run_describe_never_replaces_its_config_file(tmp_path: Path) -> None:
    # It replaced config.yaml with the JSON Lines export, exit 0, --force or
    # not: a process of its own, as a user runs it.
    import subprocess

    config_file = tmp_path / "config.yaml"
    config_file.write_text("backend: local\n", encoding="utf-8")
    for extra in ([], ["--force"]):
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "unified_kg_rag.application.cli.run_describe",
                "--config-path",
                str(config_file),
                "--output-path",
                str(config_file),
                "--output-format",
                "json",
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=100,
            cwd=tmp_path,
            env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true"},
        )
        assert done.returncode == 2, done.stderr
        assert "is the --config-path file" in json.loads(done.stdout)["error"]
        assert config_file.read_text(encoding="utf-8") == "backend: local\n"


def test_describe_lists_the_failed_and_quarantined_documents(
    ingested, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # The runbook said to inspect a failed document's error_info, with no
    # way to read it short of querying the registry's store.
    from unified_kg_rag.domain.models import DocStatus

    config, providers = ingested
    store = providers.build_doc_status()
    namespace = describe_namespace(config, providers=providers).namespace
    record = next(
        r for r in store.list_namespace(namespace) if r.file_path == "ticket-1"
    )
    store.put_many(
        [
            record.model_copy(
                update={
                    "status": DocStatus.FAILED,
                    "error_info": "ModelOutputError: cut off",
                    "failure_count": 2,
                }
            )
        ]
    )
    description = describe_namespace(
        config, providers=providers, documents=[DocStatus.FAILED, DocStatus.QUARANTINED]
    )
    assert description.registry == {"failed": 1, "processed": 2}
    (failed,) = description.documents or []
    assert (failed.status, failed.error_info, failed.failure_count) == (
        DocStatus.FAILED,
        "ModelOutputError: cut off",
        2,
    )
    assert describe_namespace(config, providers=providers).documents is None
    monkeypatch.setattr(run_describe, "load_cli_config", lambda _args: config)
    monkeypatch.setattr(inspection, "build_providers", lambda _c: providers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-describe",
            "--documents",
            "failed,quarantined",
            "--output-format",
            "json",
        ],
    )
    assert run_describe.main() == 0
    (listed,) = json.loads(capsys.readouterr().out)["documents"]
    assert (
        listed["status"] == "failed"
        and listed["error_info"] == "ModelOutputError: cut off"
    )
