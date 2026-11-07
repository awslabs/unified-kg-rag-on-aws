# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's parts without a server: path containment, secret
handling, the ingestion stage plan, job helpers and the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.application.cli.console import dashboard, jobs, settings_api, stages
from unified_kg_rag.application.cli.console.ask import (
    link_sources,
    mark_sources,
    source_view,
    trace_url,
)
from unified_kg_rag.application.cli.console.paths import (
    PathRefused,
    Root,
    Roots,
    contained,
    safe_name,
    safe_relative,
    walk_files,
)
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    PipelineConfig,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import ObservabilityBackend

pytestmark = pytest.mark.unit


# -- paths -------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "../x",
        "a/../../x",
        "/etc/passwd",
        "C:/x",
        "c:\\x",
        "\\\\server\\share",
        "a\x00b",
    ],
)
def test_contained_refuses_what_leaves_the_root(tmp_path: Path, relative: str) -> None:
    with pytest.raises(PathRefused):
        contained(tmp_path, relative)


def test_contained_allows_paths_inside(tmp_path: Path) -> None:
    assert contained(tmp_path, "a/b/../c") == (tmp_path / "a" / "c").resolve()
    assert contained(tmp_path, "") == tmp_path.resolve()


def test_a_symlink_out_of_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "out").symlink_to(tmp_path)
    with pytest.raises(PathRefused):
        contained(root, "out/x")


@pytest.mark.parametrize("name", ["", ".", "..", "a/../b", "~home", "a:b", "a\x07"])
def test_unsafe_upload_names_are_refused(name: str) -> None:
    with pytest.raises(PathRefused):
        safe_relative(name)


def test_upload_paths_keep_their_folders_and_compose_hangul() -> None:
    import unicodedata

    nfd = unicodedata.normalize("NFD", "계약/요약.txt")
    assert safe_relative(nfd).as_posix() == "계약/요약.txt"
    assert safe_relative("./a\\b.txt").as_posix() == "a/b.txt"
    with pytest.raises(PathRefused):
        safe_name("a/b", "name")
    with pytest.raises(PathRefused):
        safe_relative("x" * 300)


def test_roots_name_a_path_by_its_innermost_root_and_guard_writes(
    tmp_path: Path,
) -> None:
    (tmp_path / "cache").mkdir()
    roots = Roots.build(
        Root("workspace", tmp_path), Root("cache", tmp_path / "cache", writable=True)
    )
    assert roots.ref(tmp_path / "cache" / "x.json") == "cache:x.json"
    assert roots.ref(tmp_path / "a.json") == "workspace:a.json"
    assert roots.ref(tmp_path) == "workspace:"
    assert roots.ref(Path("/")) is None
    assert roots.resolve("cache:x", write=True) == (tmp_path / "cache" / "x").resolve()
    with pytest.raises(PathRefused, match="read-only"):
        roots.resolve("workspace:x", write=True)
    with pytest.raises(PathRefused, match="unknown"):
        roots.resolve("other:x")


def test_walk_files_skips_hidden_entries_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / ".hidden").write_text("h")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x").write_text("x")
    (tmp_path / "link.txt").symlink_to(tmp_path / "a.txt")
    assert [p.name for p in walk_files(tmp_path, limit=10)] == ["a.txt"]
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("f")
    assert len(walk_files(tmp_path, limit=3)) == 3


# -- settings ----------------------------------------------------------------


def test_every_secret_field_of_the_config_is_known() -> None:
    paths = {".".join(p) for p in settings_api.secret_paths()}
    assert {
        "providers.llm.api_key",
        "providers.embedding.api_key",
        "aws.opensearch.password",
    } <= paths


def test_secrets_are_found_and_masked() -> None:
    data = {
        "providers": {"llm": {"api_key": "sk-1"}},
        "aws": {"opensearch": {"password": None}},
    }
    assert settings_api.set_secrets(data) == ["providers.llm.api_key"]
    masked = settings_api.masked(data)
    assert masked["providers"]["llm"]["api_key"] == settings_api.MASK
    assert data["providers"]["llm"]["api_key"] == "sk-1"  # not modified
    assert "sk-1" not in json.dumps(masked)


def test_changes_set_and_unset_keys(tmp_path: Path) -> None:
    data = {"search": {"top_k": 3}, "a": {"b": 1}}
    changed = settings_api.apply_changes(
        data,
        [
            {"path": "search.top_k", "value": 5},
            {"path": "a.b", "unset": True},
            {"path": "x.y.z", "value": "v"},
        ],
    )
    assert changed == {"search": {"top_k": 5}, "x": {"y": {"z": "v"}}}
    with pytest.raises(settings_api.ConfigurationError):
        settings_api.apply_changes(data, [{"path": "a..b;rm", "value": 1}])


def test_a_preview_validates_like_a_run(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("# my comment\nsearch:\n  reranking:\n    enabled: true\n")
    file = settings_api.SettingsFile(path)
    preview = file.preview(
        text=None, changes=[{"path": "logging.level", "value": "DEBUG"}]
    )
    assert preview["valid"], preview
    assert preview["comments_dropped"]
    assert "+  level: DEBUG" in preview["diff"]
    with pytest.raises(settings_api.ConfigurationError, match="YAML"):
        file.preview(text="search: [1, 2\n", changes=None)
    with pytest.raises(settings_api.ConfigurationError, match="mapping"):
        file.preview(text="- 1\n", changes=None)
    unknown = file.preview(text="search:\n  nope: 1\n", changes=None)
    assert not unknown["valid"] and unknown["unknown_keys"] == ["search.nope"]


def test_moving_a_served_directory_is_saved_only_when_confirmed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: INFO\n")
    file = settings_api.SettingsFile(path)
    base = settings_api.digest(path.read_text())
    preview = file.preview(
        text=None,
        changes=[
            {"path": "evaluation.outputs_directory", "value": "/srv/elsewhere"},
            {"path": "logging.level", "value": "DEBUG"},
        ],
    )
    assert preview["directory_changes"] == [
        {"path": "evaluation.outputs_directory", "from": None, "to": "/srv/elsewhere"}
    ]
    with pytest.raises(settings_api.DirectoriesNotConfirmed) as refused:
        file.save(text=preview["text"], base_sha=base)
    assert refused.value.paths == ["evaluation.outputs_directory"]
    assert "/srv/elsewhere" not in path.read_text()
    file.save(
        text=preview["text"],
        base_sha=base,
        confirm_directories=["evaluation.outputs_directory"],
    )
    assert "/srv/elsewhere" in path.read_text()
    # Another key alone asks nothing; the visualization folder does.
    assert not file.preview(
        text=None, changes=[{"path": "logging.level", "value": "INFO"}]
    )["directory_changes"]
    viz = file.preview(
        text=None,
        changes=[
            {"path": "graph.visualization.outputs_directory", "value": "/srv/viz"}
        ],
    )
    assert [c["path"] for c in viz["directory_changes"]] == [
        "graph.visualization.outputs_directory"
    ]


def test_a_view_without_a_file_uses_the_defaults(tmp_path: Path) -> None:
    view = settings_api.SettingsFile(None).view()
    assert view["path"] is None and not view["writable"]
    assert view["effective"]["backend"] == "aws"
    assert set(view["tracing_environment"]) >= {
        "LANGFUSE_HOST",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    }
    assert view["methodologies"] == [m.value for m in Methodology]


def test_the_tracing_panel_reports_presence_not_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-secret")
    view = settings_api.SettingsFile(None).view()
    assert view["tracing_environment"]["LANGFUSE_SECRET_KEY"] is True
    assert "sk-lf-secret" not in json.dumps(view)


# -- stages ------------------------------------------------------------------


def _pipeline_stages(config: Config, tmp_path: Path) -> list[str]:
    """The stages a pipeline built for ``config`` runs (its own rule)."""
    pipeline = DataIngestionPipeline.__new__(DataIngestionPipeline)
    pipeline.config = config
    pipeline.pipeline_config = PipelineConfig(
        stages_enabled=dict.fromkeys(PipelineStageType, True), local_directory=tmp_path
    )
    return [
        s.value
        for s in DataIngestionPipeline.STAGE_CLASSES
        if pipeline._stage_enabled(s)
    ]


@pytest.mark.parametrize(
    "methodologies",
    [["graphrag", "lightrag"], ["lightrag"], ["hipporag"], ["graphrag", "raptor"]],
)
@pytest.mark.parametrize("gleaning", [True, False])
def test_the_stage_plan_is_the_pipelines(
    tmp_path: Path, methodologies: list[str], gleaning: bool
) -> None:
    config = Config()
    config.indexing.methodologies = [Methodology(m) for m in methodologies]
    config.processing.gleaning.enabled = gleaning
    assert stages.planned_stages(config) == _pipeline_stages(config, tmp_path)


def test_stage_progress_marks_running_restored_and_failed(tmp_path: Path) -> None:
    plan = ["document_parsing", "document_loading", "text_chunking", "graph_extraction"]
    metadata = {
        "stage_results": [
            {
                "stage_name": "document_parsing",
                "status": "completed",
                "start_time": "2026-01-01T00:00:00",
            },
            {
                "stage_name": "document_loading",
                "status": "completed",
                "start_time": "2026-01-02T00:00:01",
                "metrics": {
                    "failed_files": [{"file": "a.pdf", "error": "unreadable"}],
                    "llm_usage": {"llm_calls": 0},
                },
            },
            {
                "stage_name": "text_chunking",
                "status": "failed",
                "error_message": "boom",
                "start_time": "2026-01-02T00:00:02",
            },
        ]
    }
    steps = stages.stage_progress(
        plan, metadata, running=True, started_after="2026-01-02T00:00:00"
    )
    assert [s["status"] for s in steps] == [
        "restored",
        "completed",
        "failed",
        "running",
    ]
    assert steps[1]["failed_items"] == ["a.pdf: unreadable"]
    assert steps[2]["error"] == "boom"
    ended = stages.stage_progress(plan, None, running=False)
    assert [s["status"] for s in ended] == ["stopped", "not_run", "not_run", "not_run"]
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / stages.METADATA_FILENAME).write_text("{half")
    assert stages.read_metadata(tmp_path, "p") is None


# -- jobs ----------------------------------------------------------------------


def test_the_last_json_document_is_the_result() -> None:
    assert jobs._last_json('{"a": 1}') == {"a": 1}
    assert jobs._last_json('noise\n{"b": 2}') == {"b": 2}
    assert jobs._last_json("") is None
    assert jobs._last_json("[1]") is None
    assert jobs._last_json("not json") is None


def test_the_error_of_a_failed_job_is_read_from_its_log() -> None:
    assert jobs._last_error(["ok", "Error: boom", "tail"]) == "Error: boom"
    assert jobs._last_error(["only line"]) == "only line"
    assert jobs._last_error([]) is None


# -- ask -------------------------------------------------------------------------


def test_sources_keep_their_provenance_and_link_to_passages() -> None:
    entity = source_view(
        1,
        {
            "source": "e1",
            "content": "Entity: Vendor",
            "score": 0.5,
            "metadata": {
                "section_type": "entity",
                "name": "Vendor",
                "type": "organization",
                "text_unit_ids": ["t1", "t9"],
                "source_kinds": ["document", "seed_graph"],
                "iri": "https://example.org/Vendor",
                "provenance": {
                    "kind": "document+seed_graph",
                    "seed_graphs": ["g"],
                    "iri": None,
                },
            },
        },
    )
    passage = source_view(
        2,
        {
            "source": "t1",
            "content": "Vendor ships.",
            "metadata": {
                "section_type": "text",
                "attributes": {"file_name": "a.txt"},
                "chunk_id": "t1",
            },
        },
    )
    link_sources([entity, passage])
    assert entity["supported_by"] == [2] and entity["unread_passages"] == 1
    assert entity["iri"] == "https://example.org/Vendor"
    assert entity["provenance"]["seed_graphs"] == ["g"]
    assert entity["node_ids"] == ["e1"]
    assert passage["file_name"] == "a.txt" and passage["title"] == "a.txt"
    relation = source_view(
        3,
        {
            "source": "r1",
            "metadata": {
                "section_type": "relationship",
                "source_name": "A",
                "target_name": "B",
                "source_id": "a",
                "target_id": "b",
                "type": "SUPPLIES",
            },
        },
    )
    assert relation["title"] == "A → B" and relation["node_ids"] == ["a", "b"]


def test_trace_links_follow_the_template_or_langfuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config()
    assert trace_url(None, "x/{trace_id}", config) is None
    assert (
        trace_url("abc", "http://jaeger/trace/{trace_id}", config)
        == "http://jaeger/trace/abc"
    )
    assert trace_url("abc", None, config) is None
    config.observability.enabled = True
    config.observability.backend = ObservabilityBackend.LANGFUSE
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")
    assert trace_url("abc", None, config) == "http://localhost:3000/trace/abc"


# -- store and dashboard --------------------------------------------------------


def _usage(
    model: str, calls: int, tokens_in: int, tokens_out: int
) -> dict[str, object]:
    counts = {
        "llm_calls": calls,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
    }
    return {**counts, "by_role": {"query": counts}, "by_model": {model: counts}}


def test_the_dashboard_prices_usage_and_leaves_unpriced_models_out(
    tmp_path: Path,
) -> None:
    store = ConsoleStore(tmp_path / "console.sqlite3")
    store.record_query(
        {
            "id": "1",
            "created": 1000.0,
            "question": "q1",
            "requested_strategy": "auto",
            "strategy": "local",
            "outcome": "answered",
            "latency_seconds": 1.0,
            "usage": _usage("m", 2, 1000, 100),
            "cost_usd": 0.01,
        }
    )
    store.record_query(
        {
            "id": "2",
            "created": 1001.0,
            "question": "q2",
            "requested_strategy": "local",
            "strategy": "local",
            "outcome": "abstained",
            "abstained": True,
            "latency_seconds": 3.0,
            "usage": _usage("other", 1, 10, 1),
        }
    )
    store.save_job(
        {
            "id": "j",
            "kind": "evaluate",
            "namespace": "default",
            "created": 1002.0,
            "started": 1002.0,
            "ended": 1012.0,
            "status": "succeeded",
            "result": {
                "runs": [
                    {
                        "usage": {
                            "answer_generation": {
                                "llm_calls": 4,
                                "input_tokens": 40,
                                "output_tokens": 4,
                            }
                        }
                    }
                ],
                "comparison": {"cost": {"strategies": {"local": {"usd_total": 0.5}}}},
            },
        }
    )
    config = Config.model_validate(
        {
            "evaluation": {
                "prices": {
                    "m": {
                        "input_per_million_tokens": 3,
                        "output_per_million_tokens": 15,
                    }
                }
            }
        }
    )
    summary = dashboard.summarize(store.queries(), store.jobs(), config, now=2000.0)
    assert summary["totals"]["queries"] == 2 and summary["totals"]["abstained"] == 1
    assert summary["totals"]["cost_usd"] == pytest.approx(0.01)
    assert summary["unpriced_models"] == ["other"]
    models = {m["model"]: m for m in summary["models"]}
    assert models["m"]["cost_usd"] == pytest.approx((1000 * 3 + 100 * 15) / 1e6)
    assert models["other"]["cost_usd"] is None
    assert summary["strategies"][0]["latency_p95"] == 3.0
    assert (
        summary["jobs"][0]["llm_calls"] == 4 and summary["jobs"][0]["cost_usd"] == 0.5
    )
    assert summary["jobs"][0]["duration_seconds"] == 10.0
    assert store.clear_queries() == 2 and store.queries() == []
    store.close()


def test_a_saved_run_is_traced_to_the_console_job_that_wrote_it(tmp_path: Path) -> None:
    from unified_kg_rag.application.cli.console.evaluations import console_job_of

    job = tmp_path / "console" / "20261107-000000-abcdef"
    assert console_job_of(job) == job.name
    assert console_job_of(job / "run_20261107_000001_12345678") == job.name
    assert console_job_of(tmp_path / "elsewhere" / "run_1") is None


def test_citations_name_the_source_they_were_shown_as() -> None:
    sources = [
        {"rank": 1, "id": "u-2", "kind": "text"},
        {"rank": 2, "id": "e-1", "kind": "entity"},
        {"rank": 3, "id": "u-9", "kind": "text"},
    ]
    citations = [
        {"marker": 1, "source_id": "e-1"},
        {"marker": 2, "source_id": "u-2"},
        # A synthesized summary the model read is no reported source.
        {"marker": 3, "source_id": "summary"},
    ]
    mark_sources(sources, citations)
    assert [c["source_rank"] for c in citations] == [2, 1, None]
    assert [s["marker"] for s in sources] == [2, 1, None]
