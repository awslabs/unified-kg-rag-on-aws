# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's parts without a server: path containment, secret
handling, the ingestion stage plan, job helpers and the dashboard."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from unified_kg_rag.application.cli.console import dashboard, jobs, settings_api, stages
from unified_kg_rag.application.cli.console.ask import (
    link_sources,
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
from unified_kg_rag.application.presentation.answer_view import SourceView
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    PipelineConfig,
    PipelineStageType,
    Source,
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


def test_every_file_and_directory_key_of_the_config_is_confirmed() -> None:
    keys = {".".join(p) for p in settings_api.path_keys()}
    assert {
        "evaluation.outputs_directory",
        "graph.visualization.outputs_directory",
        "logging.log_file_path",
        "providers.graph.sqlite_path",
        "providers.vector.sqlite_path",
        "providers.doc_status.sqlite_path",
        "processing.seed_graphs.*.path",
        "processing.graph_extraction.schema_source.path",
        "processing.document_parsing.source_directory",
        "processing.document_parsing.target_directory",
    } <= keys
    # A switch named like one is not a path.
    assert "logging.log_to_file" not in keys


@pytest.mark.parametrize(
    "new",
    [
        {"logging": {"log_file_path": "/elsewhere/run.log"}},
        {"providers": {"graph": {"sqlite_path": "/elsewhere/graph.sqlite3"}}},
        {"processing": {"seed_graphs": [{"name": "s", "path": "/elsewhere/seed.ttl"}]}},
        {"processing": {"document_parsing": {"source_directory": "/elsewhere"}}},
    ],
)
def test_a_moved_path_is_a_directory_change(new: dict) -> None:  # type: ignore[type-arg]
    changes = settings_api.directory_changes({}, new)
    assert len(changes) == 1 and changes[0]["from"] is None
    assert changes[0]["to"].startswith("/elsewhere")
    assert settings_api.directory_changes(new, new) == []


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
    with pytest.raises(settings_api.InvalidArgumentError):
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
    with pytest.raises(settings_api.InvalidArgumentError, match="YAML"):
        file.preview(text="search: [1, 2\n", changes=None)
    with pytest.raises(settings_api.InvalidArgumentError, match="mapping"):
        file.preview(text="- 1\n", changes=None)
    unknown = file.preview(text="search:\n  nope: 1\n", changes=None)
    assert not unknown["valid"] and unknown["unknown_keys"] == ["search.nope"]


def test_a_preview_names_each_deprecated_and_removed_key_as_the_loader_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deprecation warning reached only the server log: the Settings
    page, where the config is edited, said nothing (rv33)."""
    from unified_kg_rag.application import config as loader
    from unified_kg_rag.domain.models.config import DEPRECATED_KEYS, _root

    logged: set[str] = set()

    class _Logger:
        """What the loader logs."""

        def __init__(self, logger: logging.Logger) -> None:
            self.logger = logger

        def warning(self, message: str, *args: object) -> None:
            logged.add(message % args)

        def __getattr__(self, name: str) -> object:
            return getattr(self.logger, name)

    monkeypatch.setattr(_root, "_logger", _Logger(_root._logger))
    monkeypatch.setattr(loader, "_logger", _Logger(loader._logger))
    # The loader warns of a deprecated key once a process.
    _root._warn_deprecated.cache_clear()
    text = (
        "retry:\n  max_attempts: 3\n  query_store_max_attempts: 2\n"
        "evaluation:\n  prices: {}\nfixing:\n  enabled: true\n"
    )
    loaded = settings_api.SettingsFile(tmp_path / "config.yaml").preview(
        text=text, changes=None
    )
    assert loaded["valid"], loaded
    deprecated = loaded["deprecated_keys"]
    assert [d["key"] for d in deprecated] == [
        k
        for k in DEPRECATED_KEYS
        if k
        in ("evaluation.prices", "retry.max_attempts", "retry.query_store_max_attempts")
    ]
    assert deprecated[1]["replacements"] == [
        "retry.model.max_attempts",
        "retry.ingestion_item.max_attempts",
    ]
    # The loader's own text, each; the removed key too.
    assert {d["message"] for d in deprecated} <= logged
    assert [r["key"] for r in loaded["removed_keys"]] == ["fixing"]
    assert any(
        m.startswith("Config key 'fixing' (in ")
        and m.endswith(loaded["removed_keys"][0]["reason"])
        for m in logged
    )
    # Apart from the other warnings, which the page shows as the server words them.
    assert loaded["warnings"] == []
    clean = settings_api.validate({"retry": {"model": {"max_attempts": 3}}}, None)
    assert clean["deprecated_keys"] == [] and clean["removed_keys"] == []
    # A moved key fails the load, naming its replacement.
    moved = settings_api.validate({"search": {"translation_model_id": "x"}}, None)
    assert not moved["valid"] and "models.roles.query_translation" in moved["errors"][0]


def test_the_settings_view_names_deprecated_keys_on_load(tmp_path: Path) -> None:
    """The deprecated keys a file sets were listed only by a review, which
    the form allows only with a change pending; and keys that disagree with
    their replacements were listed as none (rv42)."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "retry:\n  model_server_max_attempts: 2\nfixing:\n  enabled: true\n"
    )
    view = settings_api.SettingsFile(path).view()
    assert view["load_error"] is None
    assert view["effective"]["retry"]["model_server"]["max_attempts"] == 2
    assert view["deprecated_keys"] == [
        {
            "key": "retry.model_server_max_attempts",
            "replacements": ["retry.model_server.max_attempts"],
            "conflicts": [],
            "message": settings_api.deprecated_key_warning(
                "retry.model_server_max_attempts"
            ),
        }
    ]
    assert [r["key"] for r in view["removed_keys"]] == ["fixing"]
    # A deprecated key and its replacement set to different values fail the
    # load; the page names the pair instead of listing nothing.
    path.write_text(
        "retry:\n  model_server_max_attempts: 2\n  model_server:\n    max_attempts: 4\n"
    )
    conflict = settings_api.SettingsFile(path).view()
    assert "disagree" in conflict["load_error"]
    assert conflict["deprecated_keys"][0]["conflicts"] == [
        "retry.model_server.max_attempts"
    ]
    # The same value agrees.
    path.write_text(
        "retry:\n  model_server_max_attempts: 2\n  model_server:\n    max_attempts: 2\n"
    )
    agree = settings_api.SettingsFile(path).view()
    assert agree["load_error"] is None
    assert agree["deprecated_keys"][0]["conflicts"] == []


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
    # Regression (rv7): a run stopped in graph_extraction (saved "running"
    # as it started) blamed the next planned stage, gleaning.
    plan = ["text_chunking", "graph_extraction", "gleaning", "graph_resolution"]
    stopped = {
        "stage_results": [
            {"stage_name": "text_chunking", "status": "completed"},
            {"stage_name": "graph_extraction", "status": "running"},
        ]
    }
    assert [
        s["status"] for s in stages.stage_progress(plan, stopped, running=False)
    ] == [
        "completed",
        "stopped",
        "not_run",
        "not_run",
    ]
    assert [
        s["status"] for s in stages.stage_progress(plan, stopped, running=True)
    ] == [
        "completed",
        "running",
        "pending",
        "pending",
    ]
    # Regression (rv27): a run that stopped at its integrity question,
    # before any stage, listed the earlier run's as "from the cache", all
    # done. While it runs, they are what it restores.
    earlier = {
        "stage_results": [
            {"stage_name": name, "status": "completed", "start_time": "2026-01-01"}
            for name in plan
        ]
    }
    later = "2026-01-02T00:00:00"
    assert {
        s["status"]
        for s in stages.stage_progress(
            plan, earlier, running=False, started_after=later
        )
    } == {"not_run"}
    assert {
        s["status"]
        for s in stages.stage_progress(plan, earlier, running=True, started_after=later)
    } == {"restored"}
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


def _view(rank: int, source: dict) -> dict:  # type: ignore[type-arg]
    return source_view(
        SourceView.of_source(rank, TypeAdapter(Source).validate_python(source))
    )


def test_sources_keep_their_provenance_and_link_to_passages() -> None:
    entity = _view(
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
    passage = _view(
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
    relation = _view(
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
    assert relation["title"] == "A -> B" and relation["node_ids"] == ["a", "b"]


def test_a_claim_links_to_the_passages_that_state_it() -> None:
    # Regression: claims carry text_unit_ids, but the console only linked
    # entities, relationships and graph query results to their passages.
    claim = _view(
        1,
        {
            "source": "c1",
            "content": "Vendor must deliver.",
            "metadata": {"section_type": "claim", "text_unit_ids": ["t1"]},
        },
    )
    summary = _view(
        3,
        {
            "source": "s1",
            "content": "A summary.",
            "metadata": {"section_type": "summary", "text_unit_ids": ["t1"]},
        },
    )
    passage = _view(
        2,
        {
            "source": "t1",
            "content": "Vendor must deliver.",
            "metadata": {"section_type": "text", "chunk_id": "t1"},
        },
    )
    link_sources([claim, passage, summary])
    assert claim["stated_by_passages"] and claim["supported_by"] == [2]
    assert not summary["stated_by_passages"] and "supported_by" not in summary


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
    # Interpolated between the two latencies (1 s and 3 s), as run-eval reads.
    assert summary["strategies"][0]["latency_p95"] == pytest.approx(2.9)
    assert (
        summary["jobs"][0]["llm_calls"] == 4 and summary["jobs"][0]["cost_usd"] == 0.5
    )
    assert summary["jobs"][0]["duration_seconds"] == 10.0
    assert store.clear_queries() == 2 and store.queries() == []
    store.close()


def test_a_budget_stop_s_dashboard_row_names_its_error_class(tmp_path: Path) -> None:
    """The page marks an ingestion stopped at its budget as a warning, as
    its notice is, not a red failure: the row carries the error class."""
    store = ConsoleStore(tmp_path / "console.sqlite3")
    store.save_job(
        {
            "id": "b",
            "kind": "ingest",
            "namespace": "default",
            "created": 1.0,
            "started": 1.0,
            "ended": 2.0,
            "status": "failed",
            "result": {
                "status": "failed",
                "error_class": "limit_exceeded",
                "token_usage": {"llm_calls": 3, "input_tokens": 30, "output_tokens": 3},
            },
        }
    )
    (row,) = dashboard.summarize([], store.jobs(), Config(), now=3.0)["jobs"]
    assert row["error_class"] == "limit_exceeded"
    assert row["llm_calls"] == 3 and row["tokens"] == 33
    store.close()


def test_an_ingestion_job_s_dashboard_cost_is_priced_from_its_usage(
    tmp_path: Path,
) -> None:
    """Ingestion jobs always showed no cost, even with prices set: only an
    evaluation's comparison was read (rv42)."""
    store = ConsoleStore(tmp_path / "console.sqlite3")
    for job_id, model in (("priced", "m"), ("unpriced", "other")):
        store.save_job(
            {
                "id": job_id,
                "kind": "ingest",
                "namespace": "default",
                "created": 1.0,
                "started": 1.0,
                "ended": 2.0,
                "status": "failed",
                "result": {
                    "status": "failed",
                    "error_class": "limit_exceeded",
                    "token_usage": _usage(model, 2, 2000, 200),
                },
            }
        )
    config = Config.model_validate(
        {
            "models": {
                "prices": {
                    "m": {
                        "input_per_million_tokens": 3,
                        "output_per_million_tokens": 15,
                    }
                }
            }
        }
    )
    rows = {
        r["id"]: r
        for r in dashboard.summarize([], store.jobs(), config, now=3.0)["jobs"]
    }
    assert rows["priced"]["cost_usd"] == pytest.approx((2000 * 3 + 200 * 15) / 1e6)
    assert rows["unpriced"]["cost_usd"] is None
    store.close()


def test_questions_not_served_stay_out_of_every_latency_and_the_decline_rate() -> None:
    """A refused or cancelled question was never served: it has no latency
    to report (totals, per strategy, per day alike) and could not decline."""

    def query(outcome: str, latency: float, abstained: bool = False) -> dict:
        return {
            "created": 1000.0,
            "strategy": "local",
            "outcome": outcome,
            "abstained": abstained,
            "latency_seconds": latency,
        }

    summary = dashboard.summarize(
        [
            query("answered", 2.0),
            query("abstained", 4.0, abstained=True),
            query("refused", 0.001),
            query("cancelled", 30.0),
        ],
        [],
        Config(),
    )
    totals = summary["totals"]
    assert (totals["refused"], totals["errors"]) == (1, 0)
    assert totals["decline_rate"] == pytest.approx(0.5)
    assert totals["latency_p50"] == pytest.approx(3.0)
    (strategy,) = summary["strategies"]
    assert strategy["latency_p50"] == pytest.approx(3.0)
    assert summary["timeline"][0]["latency_p50"] == pytest.approx(3.0)
    assert (
        dashboard.summarize([query("refused", 0.1)], [], Config())["totals"][
            "decline_rate"
        ]
        is None
    )


def test_a_failed_question_could_not_decline_and_a_refused_one_ran_no_strategy() -> (
    None
):
    """rv27: a failed question counted in the decline rate's denominator (0%
    with nothing answered), and a refused one as a strategy's question with
    no tokens."""

    def query(outcome: str, abstained: bool = False, tokens: int = 0) -> dict:
        return {
            "created": 1000.0,
            "strategy": "local",
            "outcome": outcome,
            "abstained": abstained,
            "latency_seconds": 1.0,
            "usage": {"input_tokens": tokens},
        }

    summary = dashboard.summarize(
        [
            query("abstained", abstained=True, tokens=100),
            query("answered", tokens=300),
            query("failed"),
            query("error"),
            query("refused"),
        ],
        [],
        Config(),
    )
    assert summary["totals"]["decline_rate"] == pytest.approx(0.5)
    assert (summary["totals"]["queries"], summary["totals"]["refused"]) == (5, 1)
    (strategy,) = summary["strategies"]
    assert (strategy["queries"], strategy["errors"]) == (4, 2)
    assert strategy["tokens_per_query"] == pytest.approx(100.0)
    only_failed = dashboard.summarize([query("failed")], [], Config())
    assert only_failed["totals"]["decline_rate"] is None


def test_a_decline_at_a_query_limit_is_counted_apart_from_the_decline_rate(
    tmp_path: Path,
) -> None:
    """Declines at a query limit (limits.*) read as the corpus's: the rate
    counted them and no figure named them; the store kept no reason (rv42)."""
    import sqlite3

    path = tmp_path / "console.sqlite3"
    # A file an older console wrote, without the reason column.
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE queries (id TEXT PRIMARY KEY, created REAL NOT NULL, "
        "question TEXT NOT NULL, requested_strategy TEXT NOT NULL, strategy TEXT, "
        "suffix TEXT, latency_seconds REAL, outcome TEXT NOT NULL, "
        "abstained INTEGER NOT NULL DEFAULT 0, rerank_status TEXT, usage TEXT, "
        "cost_usd REAL, trace_id TEXT, error TEXT)"
    )
    old.execute(
        "INSERT INTO queries (id, created, question, requested_strategy, outcome) "
        "VALUES ('old', 999.0, 'q', 'local', 'answered')"
    )
    old.commit()
    old.close()
    store = ConsoleStore(path)
    rows = [
        ("1", "answered", False, None),
        ("2", "abstained", True, "no_relevant_context"),
        ("3", "abstained", True, "query_limit"),
        ("4", "abstained", True, "query_limit"),
    ]
    for query_id, outcome, abstained, reason in rows:
        store.record_query(
            {
                "id": query_id,
                "created": 1000.0,
                "question": "q",
                "requested_strategy": "local",
                "strategy": "local",
                "outcome": outcome,
                "abstained": abstained,
                "abstention_reason": reason,
                "latency_seconds": 1.0,
            }
        )
    queries = store.queries()
    assert {q["id"]: q["abstention_reason"] for q in queries}["3"] == "query_limit"
    totals = dashboard.summarize(queries, [], Config())["totals"]
    # Of the old answer, the answer and the corpus decline: one in three.
    assert totals["limited"] == 2 and totals["abstained"] == 3
    assert totals["decline_rate"] == pytest.approx(1 / 3)
    store.close()


def test_a_saved_run_is_traced_to_the_console_job_that_wrote_it(tmp_path: Path) -> None:
    from unified_kg_rag.application.cli.console.evaluations import console_job_of

    job = tmp_path / "console" / "20261107-000000-abcdef"
    assert console_job_of(job) == job.name
    assert console_job_of(job / "run_20261107_000001_12345678") == job.name
    assert console_job_of(tmp_path / "elsewhere" / "run_1") is None


def test_every_console_error_code_is_named_in_the_page_s_language() -> None:
    """The page names a refusal by its ``code`` (``error.<code>``, else the
    Ask page's ``ask.error.<code>``); a code without one showed the server's
    English message in every locale (rv26: 415 ``unsupported``)."""

    from unified_kg_rag.application.cli.console import server

    root = Path(__file__).parents[2]
    catalog = json.loads(
        (root / "console/web/src/i18n/locales/en.json").read_text(encoding="utf-8")
    )
    source = Path(server.__file__).read_text(encoding="utf-8")
    codes = set(re.findall(r'_error\(\s*[\w.\[\]]+,\s*"([a-z_]+)"', source))
    assert "unsupported" in codes
    unnamed = sorted(
        c
        for c in codes
        if f"error.{c}" not in catalog and f"ask.error.{c}" not in catalog
    )
    assert unnamed == []


def _reason_heads() -> set[str]:
    """The heads of the degraded reasons the package records: a reason
    given as a literal (``"query_limit"``) or a ``"<head>: ..."`` prefix,
    passed to the degradation ledger or a strategy's fallback."""
    import ast

    import unified_kg_rag

    calls = {"record_degradation": 1, "record": 1, "record_unresolved": 1}
    heads: set[str] = set()
    for path in Path(unified_kg_rag.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            index = calls.get(name or "", 0 if name == "fall_back" else None)
            if index is None or len(node.args) <= index:
                continue
            arg = node.args[index]
            if isinstance(arg, ast.JoinedStr) and arg.values:
                arg = arg.values[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if match := re.fullmatch(r"([a-z_]+)(?::.*)?", arg.value, re.DOTALL):
                    heads.add(match.group(1))
    return heads


def test_every_abstention_and_skip_reason_is_named_in_the_page_s_language() -> None:
    """The Ask page showed the raw key ``ask.abstention.query_limit`` (and
    ``no_reports``), and called a step the limits skipped a failed one
    (rv33). Every abstention reason has its message; every degraded reason
    head that is one (a step left out for that reason, not failed) has the
    partial notice's (the vitest catalog test holds every locale to the
    English keys)."""
    from unified_kg_rag.domain.retrieval import abstention

    catalog = json.loads(
        (Path(__file__).parents[2] / "console/web/src/i18n/locales/en.json").read_text(
            encoding="utf-8"
        )
    )
    reasons = {
        value
        for name, value in vars(abstention).items()
        if name.startswith("REASON_") and isinstance(value, str)
    }
    assert {"no_context", "query_limit", "no_reports"} <= reasons
    assert sorted(r for r in reasons if f"ask.abstention.{r}" not in catalog) == []
    skipped = _reason_heads() & reasons
    assert "query_limit" in skipped
    assert sorted(r for r in skipped if f"ask.partialSkipped.{r}" not in catalog) == []
    # The page tells them apart by the same list (AskPage SKIPPED_REASONS).
    page = (Path(__file__).parents[2] / "console/web/src/pages/AskPage.tsx").read_text(
        encoding="utf-8"
    )
    listed = re.search(r"SKIPPED_REASONS = \[([^\]]*)\]", page)
    assert listed and set(re.findall(r'"([a-z_]+)"', listed.group(1))) == skipped
