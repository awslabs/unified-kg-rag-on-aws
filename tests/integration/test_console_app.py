# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's API end to end on the embedded stores, no AWS.

The demo workspace (``console/demo``: its synthetic corpus, the fake model
server and its config) is ingested once with ``run-ingestion``; the app
then answers questions through a real ``GraphRAGChain``, runs ingestion and
evaluation jobs as the CLIs in child processes, stops one, refuses paths
outside its directories and edits the config file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.console_demo import load, requires_console

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_console]

HEADERS = {"X-Console-Request": "1"}


@pytest.fixture(scope="module")
def models() -> Iterator[Any]:
    fake = load("fake_models")
    server, state = fake.serve(chunk_delay=0.0)
    yield server, state
    server.shutdown()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory, models: Any) -> Path:
    demo = load("demo")
    root = tmp_path_factory.mktemp("console")
    demo.write_corpus(root)
    url = f"http://127.0.0.1:{models[0].server_address[1]}"
    config = demo.demo_config("fake", model_base_url=url, region=None)
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    demo.ingest(root, root / "config.yaml", root / "corpus")
    return root


@pytest.fixture(scope="module")
def client(workspace: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from unified_kg_rag.application.cli.console.server import ConsoleOptions, create_app

    web = workspace / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>console</title>")
    (web / "assets" / "app.js").write_text("console.log(1)")
    options = ConsoleOptions(
        workspace=workspace,
        cache_directory=workspace / "cache",
        config_path=workspace / "config.yaml",
        web_directory=web,
        sample_questions=["Who carries the lidar units?"],
    )
    with TestClient(create_app(options), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _events(response: Any) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _ask(client: Any, **body: Any) -> list[dict[str, Any]]:
    response = client.post("/api/ask", json=body, headers=HEADERS)
    assert response.status_code == 200
    return _events(response)


def _wait(client: Any, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("running", "stopping"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} still running: {job}")


def test_status_names_the_config_backend_and_namespace(client: Any) -> None:
    status = client.get("/api/status").json()
    assert status["config_error"] is None
    assert status["namespace"] == "default"
    assert status["backend"]["graph"] == "sqlite"
    assert status["backend"]["llm"] == "ollama"
    assert status["prices_configured"] is True
    assert status["sample_questions"] == ["Who carries the lidar units?"]


def test_a_question_streams_its_answer_then_the_result(client: Any) -> None:
    events = _ask(client, query="Who carries the lidar units to Kestrel Depot?")
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "result"
    chunks = [e["text"] for e in events if e["kind"] == "chunk"]
    result = events[-1]
    assert len(chunks) > 1
    assert "".join(chunks) == result["answer"]
    assert "Umbra Freight" in result["answer"]
    # auto says how it chose.
    assert result["routing"]["decided_by"] == "router"
    assert result["strategy"] in result["routing"]["candidates"]
    assert result["rerank"]["status"] == "disabled"
    assert result["usage"]["llm_calls"] >= 1
    assert result["cost_usd"] is not None and result["cost_usd"] > 0
    kinds_of_sources = {s["kind"] for s in result["sources"]}
    assert {"entity", "relationship", "text"} <= kinds_of_sources
    passages = {s["rank"]: s for s in result["sources"] if s["kind"] == "text"}
    assert all(s["file_name"] for s in passages.values())
    linked = [s for s in result["sources"] if s.get("supported_by")]
    assert linked and all(r in passages for s in linked for r in s["supported_by"])


def test_questions_are_recorded_for_history_and_the_dashboard(client: Any) -> None:
    _ask(
        client,
        query="How many lidar units did Orion Labs reject?",
        search_strategy="local",
    )
    history = client.get("/api/history").json()["queries"]
    assert history[0]["question"] == "How many lidar units did Orion Labs reject?"
    assert history[0]["strategy"] == "local"
    dashboard = client.get("/api/dashboard").json()
    assert dashboard["totals"]["queries"] >= 1
    assert any(s["strategy"] == "local" for s in dashboard["strategies"])
    assert any(
        m["model"] == "demo-model" and m["cost_usd"] for m in dashboard["models"]
    )
    assert dashboard["roles"][0]["role"] == "query"


def test_a_korean_question_is_answered_from_the_korean_note(client: Any) -> None:
    result = _ask(
        client, query="케스트럴 창고까지 운송은 누가 맡나요?", search_strategy="local"
    )[-1]
    assert result["kind"] == "result"
    assert any("contract-summary.ko.txt" == s["file_name"] for s in result["sources"])


def test_a_refused_filter_is_reported_as_such(client: Any) -> None:
    events = _ask(
        client, query="Who?", search_strategy="local", filters={"no_such_key": "x"}
    )
    assert events[-1]["kind"] == "error"
    assert events[-1]["error_kind"] == "invalid_filter"
    history = client.get("/api/history").json()["queries"]
    assert history[0]["outcome"] == "failed"


def test_strategies_mark_what_the_namespace_does_not_hold(client: Any) -> None:
    found = {s["name"]: s for s in client.get("/api/strategies").json()["strategies"]}
    assert found["local"]["available"] and found["local"]["methodology"] == "graphrag"
    assert found["mix"]["methodology"] == "lightrag"
    assert not found["tree"]["available"] and found["tree"]["reason"]
    assert found["local"]["routable"] and not found["drift"]["routable"]


def test_changes_need_the_console_header_and_the_same_origin(client: Any) -> None:
    assert client.post("/api/ask", json={"query": "x"}).status_code == 403
    response = client.post(
        "/api/ask",
        json={"query": "x"},
        headers={**HEADERS, "Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert client.delete("/api/history").status_code == 403


def test_a_foreign_host_is_refused_on_loopback(client: Any) -> None:
    response = client.get("/api/status", headers={"Host": "rebound.example:8501"})
    assert response.status_code == 421


def test_responses_carry_security_headers_and_no_cors(client: Any) -> None:
    response = client.get("/api/status", headers={"Origin": "http://127.0.0.1"})
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "ref",
    [
        "workspace:../../etc/passwd",
        "workspace:/etc/passwd",
        "cache:../config.yaml",
        "elsewhere:x",
        "no-root",
        "workspace:C:/Windows",
    ],
)
def test_paths_outside_the_console_directories_are_refused(
    client: Any, ref: str
) -> None:
    for path in ("graphs/data", "datasets/preview", "evaluations/run"):
        response = client.get(f"/api/{path}", params={"ref": ref})
        assert response.status_code == 403, (path, ref, response.text)


def test_a_symlink_out_of_the_workspace_is_refused(
    client: Any, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.json"
    outside.write_text('[{"question": "x"}]')
    link = workspace / "linked.json"
    link.symlink_to(outside)
    try:
        response = client.get(
            "/api/datasets/preview", params={"ref": "workspace:linked.json"}
        )
        assert response.status_code == 403
        datasets = [d["ref"] for d in client.get("/api/datasets").json()["datasets"]]
        assert "workspace:linked.json" not in datasets
    finally:
        link.unlink()


def test_uploads_land_in_the_uploads_folder_and_unsafe_names_are_refused(
    client: Any, workspace: Path
) -> None:
    ok = client.post(
        "/api/uploads",
        data={"name": "extra"},
        files=[
            ("files", ("notes/a.txt", b"Vendor ships parts to Buyer.", "text/plain"))
        ],
        headers=HEADERS,
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["corpus"] == "uploads:extra"
    assert (
        workspace / "uploads" / "extra" / "notes" / "a.txt"
    ).read_text() == "Vendor ships parts to Buyer."
    for name, filename in (
        ("extra", "../escape.txt"),
        ("../up", "a.txt"),
        ("extra", "~/x.txt"),
        ("a/b", "a.txt"),
    ):
        response = client.post(
            "/api/uploads",
            data={"name": name},
            files=[("files", (filename, b"x", "text/plain"))],
            headers=HEADERS,
        )
        assert response.status_code == 403, (name, filename)
    assert not (workspace / "uploads" / "escape.txt").exists()
    assert not (workspace / "up").exists()
    corpora = {c["ref"]: c for c in client.get("/api/corpora").json()["corpora"]}
    assert corpora["uploads:extra"]["files"] == 1
    assert corpora["workspace:corpus"]["files"] == 7
    assert "workspace:cache" not in corpora


def test_the_graph_is_read_with_layout_communities_and_analysis(client: Any) -> None:
    graphs = client.get("/api/graphs").json()["graphs"]
    assert graphs and graphs[0]["ref"].startswith("cache:")
    data = client.get("/api/graphs/data", params={"ref": graphs[0]["ref"]}).json()
    names = {n["name"] for n in data["nodes"]}
    assert {"Acme Robotics", "Globex Sensors", "엄브라 프레이트"} <= names
    assert set(data["layout"]) == {n["id"] for n in data["nodes"]}
    assert data["communities"] and data["communities"][0]["top_entities"]
    stats = data["stats"]
    assert stats["nodes"] == len(data["nodes"])
    assert stats["evidence"]["entities"] == {"document": len(data["nodes"])}
    assert sum(stats["schema"]["entities"].values()) == len(data["nodes"])
    # The graph a source of an answer names is the same graph.
    result = _ask(
        client, query="Who works for Acme Robotics?", search_strategy="local"
    )[-1]
    node_ids = {n["id"] for n in data["nodes"]}
    entity_ids = {s["id"] for s in result["sources"] if s["kind"] == "entity"}
    assert entity_ids and entity_ids <= node_ids


def test_datasets_are_found_and_previewed(client: Any) -> None:
    datasets = client.get("/api/datasets").json()["datasets"]
    assert {
        "ref": "workspace:datasets/demo-eval.json",
        "name": "demo-eval.json",
        "items": 7,
    } in datasets
    preview = client.get(
        "/api/datasets/preview", params={"ref": "workspace:datasets/demo-eval.json"}
    ).json()
    assert preview["unanswerable"] == 1
    assert preview["feeds"]["answer_match"] == 6
    bad = client.get("/api/datasets/preview", params={"ref": "workspace:config.yaml"})
    assert bad.status_code == 422


def test_an_evaluation_job_runs_the_cli_and_its_run_is_saved(client: Any) -> None:
    job = client.post(
        "/api/jobs/evaluate",
        json={
            "dataset": "workspace:datasets/demo-eval.json",
            "strategies": ["local", "naive"],
            "repeats": 1,
        },
        headers=HEADERS,
    ).json()
    assert job["command"][0] in ("run-eval", Path(job["command"][0]).name)
    assert job["total_runs"] == 2
    done = _wait(client, job["id"])
    assert done["status"] == "succeeded", done
    assert done["progress"] == {"done": 2, "total": 2}
    assert done["result"]["gate_failures"] == []
    assert "--output-directory" in done["command"]
    runs = client.get("/api/evaluations").json()["runs"]
    mine = next(r for r in runs if r["label"] == job["id"])
    assert mine["strategies"] == ["local", "naive"]
    run = client.get("/api/evaluations/run", params={"ref": mine["ref"]}).json()
    assert run["comparison"]["baseline"] == "local"
    assert "naive" in run["comparison"]["deltas_vs_baseline"]
    assert {s["strategy"] for s in run["summaries"]} == {"local", "naive"}
    dashboard = client.get("/api/dashboard").json()
    row = next(j for j in dashboard["jobs"] if j["id"] == job["id"])
    assert row["llm_calls"] and row["cost_usd"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"strategies": ["nope"]}, "strategies"),
        ({"strategies": ["local", "local"]}, "twice"),
        ({"strategies": ["local"], "fail_if": ["x"]}, "gate"),
        (
            {"strategies": ["local"], "fail_if": ["answer_match/token_f1:ci_high<-1"]},
            "two or more",
        ),
        ({"strategies": ["local"], "pairwise_judge": True}, "two or more"),
    ],
)
def test_a_bad_evaluation_request_is_refused_before_it_starts(
    client: Any, body: dict[str, Any], message: str
) -> None:
    response = client.post(
        "/api/jobs/evaluate",
        json={"dataset": "workspace:datasets/demo-eval.json", **body},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert message in response.json()["error"]


def test_an_ingestion_job_reports_its_stages_and_refreshes_the_chain(
    client: Any,
) -> None:
    state = client.app.state.console
    _ask(client, query="Who?", search_strategy="local")
    assert state.ask._chain is not None
    job = client.post(
        "/api/jobs/ingest", json={"corpus": "uploads:extra"}, headers=HEADERS
    ).json()
    assert job["pipeline_id"] == "console-extra"
    assert "graph_extraction" in job["plan"] and "gleaning" not in job["plan"]
    done = _wait(client, job["id"])
    assert done["status"] == "succeeded", done["log"]
    stages = {s["name"]: s for s in done["progress"]["stages"]}
    assert stages["document_parsing"]["status"] == "completed"
    assert stages["indexing"]["status"] == "completed"
    assert done["result"]["status"] == "completed"
    assert state.ask._chain is None  # rebuilt at the next question


def test_ingestion_refuses_the_workspace_itself_and_a_file(client: Any) -> None:
    assert (
        client.post(
            "/api/jobs/ingest", json={"corpus": "workspace:"}, headers=HEADERS
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/jobs/ingest",
            json={"corpus": "workspace:config.yaml"},
            headers=HEADERS,
        ).status_code
        == 403
    )
    response = client.post(
        "/api/jobs/ingest",
        json={"corpus": "workspace:corpus", "methodologies": ["nope"]},
        headers=HEADERS,
    )
    assert response.status_code == 400


def test_one_job_per_namespace_and_stop_ends_it_as_the_cli_stops(
    client: Any, models: Any
) -> None:
    models[1].request_delay = 0.4
    try:
        first = client.post(
            "/api/jobs/evaluate",
            json={
                "dataset": "workspace:datasets/demo-eval.json",
                "strategies": ["local", "mix"],
                "repeats": 3,
            },
            headers=HEADERS,
        ).json()
        second = client.post(
            "/api/jobs/ingest", json={"corpus": "workspace:corpus"}, headers=HEADERS
        )
        assert second.status_code == 409
        assert second.json()["job_id"] == first["id"]
        time.sleep(3)
        stopping = client.post(f"/api/jobs/{first['id']}/stop", headers=HEADERS).json()
        assert stopping["status"] in ("stopping", "stopped")
        done = _wait(client, first["id"], timeout=60)
    finally:
        models[1].request_delay = 0.0
    assert done["status"] == "stopped"
    assert done["exit_code"] in (143, -15)
    assert client.post("/api/jobs/nope/stop", headers=HEADERS).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_settings_show_validate_diff_and_save_the_file(
    client: Any, workspace: Path
) -> None:
    view = client.get("/api/config").json()
    assert view["path"] == str(workspace / "config.yaml")
    assert view["text"] and not view["secrets_in_file"]
    assert view["effective"]["providers"]["llm"]["kind"] == "ollama"
    assert "properties" in client.get("/api/config/schema").json()
    preview = client.post(
        "/api/config/preview",
        json={"changes": [{"path": "search.top_k_default_unknown", "value": 3}]},
        headers=HEADERS,
    ).json()
    assert (
        not preview["valid"] and "search.top_k_default_unknown" in preview["errors"][0]
    )
    preview = client.post(
        "/api/config/preview",
        json={"changes": [{"path": "processing.ignore_errors", "value": True}]},
        headers=HEADERS,
    ).json()
    assert preview["valid"] and preview["changed"]
    assert "+  ignore_errors: true" in preview["diff"]
    stale = client.post(
        "/api/config/save",
        json={"text": preview["text"], "base_sha": "0" * 64},
        headers=HEADERS,
    )
    assert stale.status_code == 400 and "changed on disk" in stale.json()["error"]
    saved = client.post(
        "/api/config/save",
        json={"text": preview["text"], "base_sha": view["sha"]},
        headers=HEADERS,
    )
    assert saved.status_code == 200, saved.text
    assert (
        yaml.safe_load((workspace / "config.yaml").read_text())["processing"][
            "ignore_errors"
        ]
        is True
    )
    invalid = client.post(
        "/api/config/save",
        json={"text": "search: {top_k_nope: 1}\n", "base_sha": saved.json()["sha"]},
        headers=HEADERS,
    )
    assert invalid.status_code == 400


def test_secrets_are_never_written_or_shown(client: Any, workspace: Path) -> None:
    view = client.get("/api/config").json()
    response = client.post(
        "/api/config/preview",
        json={"changes": [{"path": "providers.llm.api_key", "value": "sk-test-value"}]},
        headers=HEADERS,
    )
    assert response.status_code == 422
    assert "UKR__PROVIDERS__LLM__API_KEY" in response.json()["error"]
    text = (
        workspace / "config.yaml"
    ).read_text() + "aws:\n  opensearch:\n    password: hunter2-value\n"
    response = client.post(
        "/api/config/save",
        json={"text": text, "base_sha": view["sha"]},
        headers=HEADERS,
    )
    assert response.status_code == 422
    assert "hunter2-value" not in (workspace / "config.yaml").read_text()
    original = (workspace / "config.yaml").read_text()
    (workspace / "config.yaml").write_text(
        original + "aws:\n  opensearch:\n    password: hunter2-value\n"
    )
    try:
        shown = client.get("/api/config")
        assert "hunter2-value" not in shown.text
        assert shown.json()["text"] is None
        assert shown.json()["secrets_in_file"] == ["aws.opensearch.password"]
        refused = client.post(
            "/api/config/preview",
            json={"changes": [{"path": "search.top_k", "value": 3}]},
            headers=HEADERS,
        )
        assert refused.status_code == 422
    finally:
        (workspace / "config.yaml").write_text(original)


def test_the_web_client_is_served_for_every_page(client: Any) -> None:
    for path in ("/", "/ask", "/graph?focus=x"):
        response = client.get(path)
        assert response.status_code == 200 and "<title>console</title>" in response.text
    asset = client.get("/assets/app.js")
    assert (
        asset.text == "console.log(1)" and "immutable" in asset.headers["cache-control"]
    )
    assert client.get("/api/nope").status_code == 404
    escaped = client.get("/%2e%2e/config.yaml")
    assert escaped.status_code == 404 and "providers" not in escaped.text


def test_without_a_build_the_page_says_how_to_build(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from unified_kg_rag.application.cli.console.server import ConsoleOptions, create_app

    options = ConsoleOptions(
        workspace=tmp_path,
        cache_directory=tmp_path / "cache",
        web_directory=tmp_path / "dist",
    )
    with TestClient(create_app(options), base_url="http://127.0.0.1") as test_client:
        response = test_client.get("/ask")
        assert response.status_code == 503 and "npm run build" in response.text
        status = test_client.get("/api/status").json()
        assert status["config_path"] is None and status["config_error"] is None
        saved = test_client.post(
            "/api/config/save",
            json={
                "text": "",
                "base_sha": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            headers=HEADERS,
        )
        assert saved.status_code == 400 and "--config-path" in saved.json()["error"]
