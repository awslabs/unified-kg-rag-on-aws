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

import asyncio
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.console_demo import load, requires_console
from tests.fixtures.graph_exports import write_export

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_console]

HEADERS = {"X-Console-Request": "1"}
TOKEN = "test-token-0123456789abcdef"


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
        token=TOKEN,
    )
    with TestClient(
        create_app(options),
        base_url="http://127.0.0.1",
        headers={"X-Console-Token": TOKEN},
    ) as test_client:
        yield test_client


def _upload(client: Any, name: str, files: dict[str, str]) -> Any:
    return client.post(
        "/api/uploads",
        data={"name": name},
        files=[
            ("files", (path, text.encode(), "text/plain"))
            for path, text in files.items()
        ],
        headers=HEADERS,
    )


@pytest.fixture
def restore_config(workspace: Path, client: Any) -> Iterator[None]:
    """Put the config file back after a test that saves it."""
    original = (workspace / "config.yaml").read_text()
    yield
    (workspace / "config.yaml").write_text(original)
    client.app.state.console.reload()


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
    # The answer cites its context; the chunks' markers are the answer's.
    cited = [c for c in result["citations"] if c["cited"]]
    assert cited and result["unknown_citation_markers"] == []
    streamed = [m for e in events if e["kind"] == "chunk" for m in e["markers"]]
    assert sorted(set(streamed)) == sorted(c["marker"] for c in cited)
    by_rank = {s["rank"]: s for s in result["sources"]}
    for citation in result["citations"]:
        if citation["source_rank"] is not None:
            assert by_rank[citation["source_rank"]]["marker"] == citation["marker"]
    assert all(c["source_rank"] for c in cited)


def test_the_tour_question_cites_one_pdf_page(client: Any) -> None:
    # Regression (rv24 ux 7, algo-index 2): under the token default the
    # demo's chunk sizes made each PDF one chunk, cited as "p. 1-3"; the
    # guide promises the page the passage is on.
    corpus = load("corpus")
    events = _ask(client, query=corpus.SAMPLE_QUESTIONS[0])
    result = events[-1]
    pdf = [
        document
        for citation in result["citations"]
        if citation["cited"]
        for document in citation.get("documents") or []
        if document["file_name"] == "supply-agreement.pdf"
    ]
    assert pdf, result["citations"]
    assert all(d["page"] is not None and d["page"] == d["page_end"] for d in pdf)
    assert {d["page"] for d in pdf} == {2}


def test_a_structured_question_shows_its_graph_query_result(client: Any) -> None:
    found = {s["name"]: s for s in client.get("/api/strategies").json()["strategies"]}
    assert found["structured"]["available"]
    assert found["structured"]["methodology"] == "graphrag"
    result = _ask(
        client,
        query="Which kinds of entities are most common?",
        search_strategy="structured",
    )[-1]
    assert result["kind"] == "result", result
    assert result["strategy"] == "structured"
    tables = [s for s in result["sources"] if s["kind"] == "query_result"]
    assert len(tables) == 1
    table = tables[0]
    assert table["operation"] and table["row_count"] and table["exact"]
    assert table["backend"] and table["excerpt"]
    # The table is a cited context section like any other.
    assert table["marker"] is not None


def test_a_follow_up_carries_the_conversation_as_history(client: Any) -> None:
    first = _ask(client, query="Who carries the lidar units?", search_strategy="local")
    answer = first[-1]["answer"]
    follow_up = _ask(
        client,
        query="Where to?",
        search_strategy="local",
        history=[
            {"role": "user", "content": "Who carries the lidar units?"},
            {"role": "assistant", "content": answer},
        ],
    )[-1]
    assert follow_up["kind"] == "result", follow_up
    # The rewrite read the history: the standalone question names the
    # carrier the first answer named.
    assert follow_up["processed_query"]["original_query"] == "Where to?"
    assert "Umbra Freight" in follow_up["processed_query"]["standalone_query"]
    response = client.post(
        "/api/ask",
        json={"query": "q", "history": [{"role": "system", "content": "x"}]},
        headers=HEADERS,
    )
    assert response.status_code == 422


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
    assert any("contract-summary.ko.md" == s["file_name"] for s in result["sources"])


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


def test_the_namespace_s_methodologies_come_from_its_build_record(
    client: Any,
) -> None:
    status = client.get("/api/status").json()
    held = client.get("/api/namespace/methodologies").json()
    assert held["namespace"] == "default" and held["recorded"] is True
    assert set(status["methodologies"]) <= set(held["held"])
    fresh = client.get(
        "/api/namespace/methodologies", params={"suffix": "never-built"}
    ).json()
    # No build record: what the config builds stands in, as for a query
    # (the strategies it serves are the ones marked available).
    assert fresh["namespace"] == "never-built" and fresh["recorded"] is False
    assert set(status["methodologies"]) <= set(fresh["held"])
    assert fresh["dropped"] == []
    assert held["dropped"] == []  # the config builds what it holds
    narrowed = client.get(
        "/api/namespace/methodologies", params={"methodologies": "lightrag"}
    ).json()
    assert narrowed["dropped"] == ["graphrag"]
    for unsafe in ("../x", "Docs", "acme*"):
        # The stores' name rule: the config and the stores refuse the others.
        bad = client.get("/api/namespace/methodologies", params={"suffix": unsafe})
        assert bad.status_code == 400 and bad.json()["code"] == "invalid"
        bad = client.get("/api/strategies", params={"suffix": unsafe})
        assert bad.status_code == 400 and bad.json()["code"] == "invalid"


def test_an_ingestion_that_narrows_the_namespace_is_refused_unconfirmed(
    client: Any,
) -> None:
    held = client.get("/api/namespace/methodologies").json()["held"]
    keep = [m for m in held if m != "graphrag"]
    assert "graphrag" in held and keep
    assert (
        _upload(client, "narrow", {"a.txt": "Vendor ships to Buyer."}).status_code
        == 200
    )
    refused = client.post(
        "/api/jobs/ingest",
        json={"corpus": "uploads:narrow", "methodologies": keep},
        headers=HEADERS,
    )
    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["code"] == "narrowing" and "graphrag" in body["methodologies"]
    assert not any(
        j["params"].get("corpus") == "uploads:narrow"
        for j in client.get("/api/jobs").json()["jobs"]
    )


def test_a_question_with_an_unsafe_suffix_is_refused_with_a_message(
    client: Any,
) -> None:
    response = client.post(
        "/api/ask", json={"query": "Who?", "suffix": "../etc"}, headers=HEADERS
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid" and "suffix" in body["error"]


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


@pytest.mark.parametrize("host", ["[::1]", "[::1]:8501", "127.0.0.2:8501", "localhost"])
def test_every_loopback_host_is_answered(client: Any, host: str) -> None:
    assert client.get("/api/status", headers={"Host": host}).status_code == 200


def test_the_loop_s_executor_is_sized_for_the_query_path(client: Any) -> None:
    async def executor() -> Any:
        return asyncio.get_running_loop()._default_executor  # type: ignore[attr-defined]

    pool = client.portal.call(executor)
    config = client.app.state.console.load_config()
    assert pool._thread_name_prefix == "graphrag-io"
    assert pool._max_workers == config.processing.io_workers


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
    for path in (
        "graphs/data",
        "datasets/preview",
        "evaluations/run",
        "evaluations/questions",
    ):
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
    assert corpora["workspace:corpus"]["files"] == len(load("corpus").documents())
    assert "workspace:cache" not in corpora


def test_an_upload_with_a_long_korean_name_lands_and_a_too_long_one_is_refused(
    client: Any, workspace: Path
) -> None:
    # 78 Hangul characters and ".txt": 238 bytes, a valid name; its
    # temporary sibling's must fit 255 bytes too. One past 255 is refused.
    fits, too_long = "가" * 78 + ".txt", "가" * 84 + ".txt"
    ok = client.post(
        "/api/uploads",
        data={"name": "long"},
        files=[("files", (fits, "공급사가 부품을 납품한다.".encode(), "text/plain"))],
        headers=HEADERS,
    )
    assert ok.status_code == 200, ok.text
    folder = workspace / "uploads" / "long"
    assert [p.name for p in folder.iterdir()] == [fits]
    refused = client.post(
        "/api/uploads",
        data={"name": "long"},
        files=[("files", (too_long, b"x", "text/plain"))],
        headers=HEADERS,
    )
    assert refused.status_code == 403, refused.text
    assert "longer than 255 bytes" in refused.text
    assert [p.name for p in folder.iterdir()] == [fits]


def test_an_upload_name_must_fit_the_folder_s_own_name_limit(
    client: Any, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # eCryptfs holds 143 bytes, not 255: a longer name is refused up front
    # (path_refused), not by a rename failing ENAMETOOLONG once written.
    from unified_kg_rag.shared.utils import files

    monkeypatch.setattr(files.os, "pathconf", lambda *_: 143)
    for name, relative in (("short", "가" * 48 + ".txt"), ("가" * 48, "a.txt")):
        refused = _upload(client, name, {relative: "Vendor ships to Buyer."})
        assert refused.status_code == 403, refused.text
        assert refused.json()["code"] == "path_refused"
        assert "longer than 143 bytes" in refused.text
    assert not (workspace / "uploads" / ("가" * 48)).exists()
    assert not (workspace / "uploads" / "short").exists()
    assert _upload(client, "short", {"가" * 40 + ".txt": "x"}).status_code == 200


def test_an_upload_over_a_file_keeps_its_mode(client: Any, workspace: Path) -> None:
    # A re-upload replaces the file through the shared atomic write: the
    # file keeps its permission bits (0600 stays 0600), and a path named
    # twice in one upload lands once, its last copy, with no stray sibling.
    folder = workspace / "uploads" / "modes"
    folder.mkdir(parents=True)
    (folder / "private.txt").write_text("old")
    (folder / "private.txt").chmod(0o600)
    upload = client.post(
        "/api/uploads",
        data={"name": "modes"},
        files=[
            ("files", ("private.txt", b"first", "text/plain")),
            ("files", ("private.txt", b"second", "text/plain")),
        ],
        headers=HEADERS,
    )
    assert upload.status_code == 200, upload.text
    assert (folder / "private.txt").read_text() == "second"
    assert (folder / "private.txt").stat().st_mode & 0o777 == 0o600
    assert [p.name for p in folder.iterdir()] == ["private.txt"]


def test_a_refused_upload_leaves_the_folder_as_it_was(
    client: Any, workspace: Path
) -> None:
    folder = workspace / "uploads" / "kept"
    folder.mkdir(parents=True)
    (folder / "original.txt").write_text("Vendor ships parts to Buyer.")
    for later in (("../bad.txt", b"x"), ("notes.exe", b"x")):
        response = client.post(
            "/api/uploads",
            data={"name": "kept"},
            files=[
                ("files", ("original.txt", b"a new version", "text/plain")),
                ("files", ("new.txt", b"another file", "text/plain")),
                ("files", (*later, "application/octet-stream")),
            ],
            headers=HEADERS,
        )
        assert response.status_code in (403, 415), response.text
        assert (folder / "original.txt").read_text() == "Vendor ships parts to Buyer."
        assert sorted(p.name for p in folder.iterdir()) == ["original.txt"]


def test_an_upload_of_an_unsupported_type_is_refused_by_name(
    client: Any, workspace: Path
) -> None:
    response = client.post(
        "/api/uploads",
        data={"name": "typed"},
        files=[
            ("files", ("a.txt", b"Vendor ships parts.", "text/plain")),
            ("files", ("tool.exe", b"MZ", "application/octet-stream")),
        ],
        headers=HEADERS,
    )
    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported" and "tool.exe" in body["error"]
    assert ".txt" in body["error"]
    assert not (workspace / "uploads" / "typed" / "a.txt").exists()


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


def test_a_large_graph_draws_only_the_best_connected(
    client: Any, workspace: Path
) -> None:
    from unified_kg_rag.application.cli.console.graph_api import MAX_DRAWN_NODES

    total = MAX_DRAWN_NODES + 1
    path = write_export(
        workspace / "cache" / "large" / "visualization" / "visualization_data.json",
        total,
    )
    # Listed last (oldest), so the demo's graph stays the newest.
    os.utime(path, (1.0, 1.0))
    data = client.get(
        "/api/graphs/data",
        params={"ref": "cache:large/visualization/visualization_data.json"},
    ).json()
    assert len(data["layout"]) == len(data["nodes"]) == MAX_DRAWN_NODES
    assert data["stats"]["nodes"] == total
    # The rest is listed by name, and read whole on demand.
    (other,) = data["others"]
    assert set(other) == {"id", "name", "degree"}
    found = client.get(
        "/api/graphs/entity",
        params={
            "ref": "cache:large/visualization/visualization_data.json",
            "id": other["id"],
        },
    )
    assert found.status_code == 200
    assert found.json()["node"]["name"] == other["name"]
    assert len(found.json()["edges"]) == other["degree"]
    missing = client.get(
        "/api/graphs/entity",
        params={
            "ref": "cache:large/visualization/visualization_data.json",
            "id": "none",
        },
    )
    assert missing.status_code == 404


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


def test_an_evaluation_job_runs_the_cli_and_its_run_is_saved(
    client: Any, workspace: Path
) -> None:
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
    progress = done["progress"]
    assert (progress["done"], progress["total"]) == (2, 2)
    # Counted from the answer batch's log lines (both runs, interleaved).
    answers = progress["answers"]
    assert (
        answers["done"]
        == answers["total"]
        == 2 * len(json.loads((workspace / "datasets" / "demo-eval.json").read_text()))
    )
    assert done["result"]["gate_failures"] == []
    assert "--output-directory" in done["command"]
    runs = client.get("/api/evaluations").json()["runs"]
    mine = next(r for r in runs if r["job_id"] == job["id"])
    assert mine["strategies"] == ["local", "naive"]
    run = client.get("/api/evaluations/run", params={"ref": mine["ref"]}).json()
    assert run["comparison"]["baseline"] == "local"
    assert "naive" in run["comparison"]["deltas_vs_baseline"]
    assert {s["strategy"] for s in run["summaries"]} == {"local", "naive"}
    # Per question: each strategy's primary-metric score and its answer.
    questions = client.get(
        "/api/evaluations/questions", params={"ref": mine["ref"]}
    ).json()
    assert questions["metric"] == run["comparison"]["multiplicity"]["primary_metric"]
    assert questions["strategies"] == ["local", "naive"]
    dataset = json.loads((workspace / "datasets" / "demo-eval.json").read_text())
    assert len(questions["questions"]) == len(dataset)
    first = questions["questions"][0]
    assert set(first["answers"]) == {"local", "naive"}
    assert all(0.0 <= v <= 1.0 for v in first["scores"].values())
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
    assert (
        _upload(client, "fresh", {"a.txt": "Vendor ships parts to Buyer."}).status_code
        == 200
    )
    _ask(client, query="Who?", search_strategy="local")
    assert state.ask._chain is not None
    job = client.post(
        "/api/jobs/ingest", json={"corpus": "uploads:fresh"}, headers=HEADERS
    ).json()
    assert job["pipeline_id"].startswith("console-fresh-")
    assert "graph_extraction" in job["plan"] and "gleaning" not in job["plan"]
    done = _wait(client, job["id"])
    assert done["status"] == "succeeded", done["log"]
    stages = {s["name"]: s for s in done["progress"]["stages"]}
    assert stages["document_parsing"]["status"] == "completed"
    assert stages["indexing"]["status"] == "completed"
    assert done["result"]["status"] == "completed"
    assert state.ask._chain is None  # rebuilt at the next question


def test_a_changed_corpus_ingests_again_once_its_question_is_confirmed(
    client: Any, workspace: Path
) -> None:
    files = {"a.txt": "Umbra Freight carries parts for Acme Robotics."}
    assert _upload(client, "changing", files).status_code == 200
    first = _wait(
        client,
        client.post(
            "/api/jobs/ingest", json={"corpus": "uploads:changing"}, headers=HEADERS
        ).json()["id"],
    )
    assert first["status"] == "succeeded", first["log"]
    (workspace / "uploads" / "changing" / "b.txt").write_text(
        "Kestrel Depot stores parts for Globex Sensors.", encoding="utf-8"
    )
    second = _wait(
        client,
        client.post(
            "/api/jobs/ingest", json={"corpus": "uploads:changing"}, headers=HEADERS
        ).json()["id"],
    )
    # The pipeline id's inputs changed: the run asks before it writes, and
    # the job names the question instead of failing for good.
    assert second["status"] == "failed"
    assert second["confirmation"] == "integrity", second["error"]
    third = _wait(
        client,
        client.post(
            "/api/jobs/ingest",
            json={"corpus": "uploads:changing", "confirm_integrity": True},
            headers=HEADERS,
        ).json()["id"],
    )
    assert "--confirm-integrity" in third["command"]
    assert third["status"] == "succeeded", third["log"]
    assert third["pipeline_id"] == first["pipeline_id"]
    result = _ask(
        client, query="What does Kestrel Depot store?", search_strategy="naive"
    )[-1]
    assert any(s["file_name"] == "b.txt" for s in result["sources"])


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
    client: Any, workspace: Path, restore_config: None
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


def test_secrets_are_never_written_or_shown(
    client: Any, workspace: Path, restore_config: None
) -> None:
    view = client.get("/api/config").json()
    response = client.post(
        "/api/config/preview",
        json={"changes": [{"path": "providers.llm.api_key", "value": "sk-test-value"}]},
        headers=HEADERS,
    )
    assert response.status_code == 422
    assert response.json()["code"] == "credential_refused"
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


def test_every_request_needs_the_access_token(client: Any) -> None:
    from fastapi.testclient import TestClient

    anonymous = TestClient(client.app, base_url="http://127.0.0.1")
    response = anonymous.get("/api/status")
    assert response.status_code == 401 and response.json()["code"] == "unauthorized"
    assert anonymous.get("/ask").status_code == 401
    assert anonymous.get("/ask?token=wrong-token-value").status_code == 401
    # A non-ASCII token is a wrong token, not a server error.
    assert anonymous.get("/ask?token=café-ключ").status_code == 401
    assert (
        anonymous.get(
            "/api/status", headers={"X-Console-Token": "caf\xe9".encode("latin-1")}
        ).status_code
        == 401
    )
    assert (
        anonymous.get("/api/status", headers={"X-Console-Token": "wrong"}).status_code
        == 401
    )
    entry = anonymous.get(f"/ask?token={TOKEN}", follow_redirects=False)
    assert entry.status_code == 303 and entry.headers["location"] == "/ask"
    cookie = entry.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and TOKEN in cookie
    assert anonymous.get("/api/status").status_code == 200  # the cookie now


def test_a_changed_provider_endpoint_is_saved_only_when_confirmed(
    client: Any, workspace: Path, restore_config: None
) -> None:
    view = client.get("/api/config").json()
    change = [
        {"path": "providers.llm.base_url", "value": "http://elsewhere.example:11434"}
    ]
    preview = client.post(
        "/api/config/preview", json={"changes": change}, headers=HEADERS
    ).json()
    assert [c["path"] for c in preview["endpoint_changes"]] == [
        "providers.llm.base_url"
    ]
    refused = client.post(
        "/api/config/save",
        json={"text": preview["text"], "base_sha": view["sha"]},
        headers=HEADERS,
    )
    assert refused.status_code == 409 and refused.json()["paths"] == [
        "providers.llm.base_url"
    ]
    assert "elsewhere.example" not in (workspace / "config.yaml").read_text()
    saved = client.post(
        "/api/config/save",
        json={
            "text": preview["text"],
            "base_sha": view["sha"],
            "confirm_endpoints": ["providers.llm.base_url"],
        },
        headers=HEADERS,
    )
    assert saved.status_code == 200
    assert not list(workspace.glob(".config.yaml.*.tmp"))


def test_missing_files_are_not_found(client: Any) -> None:
    for path in ("datasets/preview", "evaluations/run", "evaluations/questions"):
        response = client.get(f"/api/{path}", params={"ref": "workspace:no-such.json"})
        assert response.status_code == 404 and response.json()["code"] == "not_found"
    response = client.get(
        "/api/graphs/data", params={"ref": "cache:x/visualization_data.json"}
    )
    assert response.status_code == 404


def test_uploads_are_size_capped_and_take_over_a_thousand_files(
    client: Any, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = {f"f{i}.txt": "x" for i in range(1100)}
    response = _upload(client, "many", many)
    assert response.status_code == 200, response.text
    assert response.json()["files"] == 1100
    from unified_kg_rag.application.cli.console import server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 64)
    big = _upload(client, "big", {"a.txt": "x" * 500})
    assert big.status_code == 413 and big.json()["code"] == "too_large"
    assert not (workspace / "uploads" / "big").exists() or not any(
        (workspace / "uploads" / "big").iterdir()
    )


def test_saving_new_output_directories_moves_the_roots(
    client: Any, workspace: Path, restore_config: None
) -> None:
    view = client.get("/api/config").json()
    moved = workspace / "outputs" / "moved"
    preview = client.post(
        "/api/config/preview",
        json={
            "changes": [{"path": "evaluation.outputs_directory", "value": str(moved)}]
        },
        headers=HEADERS,
    ).json()
    # Moving a directory the console serves is confirmed first.
    refused = client.post(
        "/api/config/save",
        json={"text": preview["text"], "base_sha": view["sha"]},
        headers=HEADERS,
    )
    assert refused.status_code == 409 and refused.json()["code"] == "directories"
    assert refused.json()["paths"] == ["evaluation.outputs_directory"]
    assert (
        client.post(
            "/api/config/save",
            json={
                "text": preview["text"],
                "base_sha": view["sha"],
                "confirm_directories": ["evaluation.outputs_directory"],
            },
            headers=HEADERS,
        ).status_code
        == 200
    )
    status = client.get("/api/status").json()
    assert status["directories"]["outputs"] == str(moved.resolve())
    assert client.app.state.console.jobs.evaluation_outputs == moved.resolve()


def test_a_blank_question_is_an_invalid_argument(client: Any) -> None:
    # Regression (rv8): "   " passed the request's length check and failed
    # as a raw Pydantic error of kind "failure".
    events = _ask(client, query="   ", search_strategy="local")
    assert events[-1]["kind"] == "error"
    assert events[-1]["error_kind"] == "invalid_argument"
    assert events[-1]["message"] == "query: The query is empty."


def test_the_history_records_the_namespace_asked(client: Any) -> None:
    # Regression (rv8): a question without a suffix was recorded with None,
    # not the namespace it was answered from.
    _ask(client, query="Who carries the lidar units?", search_strategy="local")
    (latest,) = client.app.state.console.store.queries(limit=1)
    assert latest["suffix"] == "default"


@pytest.mark.parametrize(
    ("files", "clash"),
    [
        ({"a.txt": "x", "a.txt/e.txt": "y"}, "a.txt/e.txt"),
        ({"notes/b.txt": "x", "notes": "y"}, None),
    ],
    ids=["file-and-folder", "folder-named-like-a-file"],
)
def test_an_upload_naming_a_file_and_a_folder_is_refused(
    client: Any, workspace: Path, files: dict[str, str], clash: str | None
) -> None:
    # Regression (rv8): answered 500 and left the folder it had made.
    response = _upload(client, "clash", files)
    assert 400 <= response.status_code < 500, response.text
    if clash is not None:
        assert response.status_code == 409
        assert response.json()["code"] == "upload_conflict"
        assert clash in response.json()["error"]
    assert not (workspace / "uploads" / "clash").exists()


def test_an_upload_onto_a_file_the_folder_holds_is_refused(
    client: Any, workspace: Path
) -> None:
    folder = workspace / "uploads" / "held"
    folder.mkdir(parents=True)
    (folder / "a.txt").write_text("kept")
    (folder / "d.txt").mkdir()
    for path in ("a.txt/b.txt", "d.txt"):
        response = _upload(client, "held", {path: "x"})
        assert response.status_code == 409, response.text
    assert (folder / "a.txt").read_text() == "kept"
    assert sorted(p.name for p in folder.iterdir()) == ["a.txt", "d.txt"]
    assert not list((folder / "d.txt").iterdir())


def test_an_upload_of_a_hidden_file_is_refused_as_ingestion_skips_it(
    client: Any, workspace: Path
) -> None:
    # Regression (rv8): uploaded (200), then never ingested: the loader
    # skips hidden files and folders (is_excluded_source_path).
    for path in (".hidden.txt", ".git/notes.txt", "__pycache__/x.txt"):
        response = _upload(client, "hidden", {"a.txt": "kept", path: "x"})
        assert response.status_code == 415, (path, response.text)
        assert path in response.json()["error"]
    assert not (workspace / "uploads" / "hidden").exists()


def test_a_configuration_error_is_named_by_its_class(client: Any) -> None:
    # Regression (rv8): every ConfigurationError answered code "invalid",
    # a namespace without an index or a methodology it lacks included.
    import asyncio

    from unified_kg_rag.shared.exceptions import (
        ConfigurationError,
        MethodologyNotIndexedError,
        StoreConfigurationError,
    )

    handler = client.app.exception_handlers[ConfigurationError]
    for error, code in (
        (MethodologyNotIndexedError("holds graphrag only"), "not_indexed"),
        (StoreConfigurationError("the store refuses the mapping"), "configuration"),
        # The console refusing a request's own value.
        (ConfigurationError("choose at least one strategy"), "invalid"),
    ):
        answered = asyncio.run(handler(None, error))
        assert answered.status_code == 400
        assert json.loads(answered.body)["code"] == code
