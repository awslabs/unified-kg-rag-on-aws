# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's parts without a server: path containment, secret
handling, the ingestion stage plan, job helpers and the dashboard."""

from __future__ import annotations

import json
import logging
import re
import time
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
from unified_kg_rag.shared.utils.deprecation import REMOVED_IN

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


def test_saving_from_fields_keeps_the_file_s_key_order(tmp_path: Path) -> None:
    """The form's changes were applied to a copy made through a sorted YAML
    dump: the saved file had every key in alphabetical order, and the diff
    rewrote every line (rv44). Comments are still lost (the review says so)."""
    path = tmp_path / "config.yaml"
    text = "search:\n  top_k: 3\n  rerank_top_k: 2\nlogging:\n  level: INFO\n"
    path.write_text(text)
    preview = settings_api.SettingsFile(path).preview(
        text=None,
        changes=[
            {"path": "logging.level", "value": "DEBUG"},
            {"path": "indexing.batch_size", "value": 8},
        ],
    )
    assert preview["text"] == (
        "search:\n  top_k: 3\n  rerank_top_k: 2\nlogging:\n  level: DEBUG\n"
        "indexing:\n  batch_size: 8\n"
    )


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
            "values": {"retry.model_server.max_attempts": 2},
            "message": settings_api.deprecated_key_warning(
                "retry.model_server_max_attempts"
            ),
            "removed_in": REMOVED_IN,
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


@pytest.mark.parametrize(
    ("text", "changes", "saved", "in_force"),
    [
        # One replacement: the edit replaces the deprecated key.
        (
            "retry:\n  model_server_max_attempts: 2\n",
            [{"path": "retry.model_server.max_attempts", "value": 4}],
            {"retry": {"model_server": {"max_attempts": 4}}},
            {"model_server": 4},
        ),
        # The same value: still only the new key.
        (
            "retry:\n  model_server_max_attempts: 2\n",
            [{"path": "retry.model_server.max_attempts", "value": 2}],
            {"retry": {"model_server": {"max_attempts": 2}}},
            {"model_server": 2},
        ),
        # Several: the others keep the deprecated key's value.
        (
            "retry:\n  max_attempts: 1\n",
            [{"path": "retry.model.max_attempts", "value": 2}],
            {
                "retry": {
                    "model": {"max_attempts": 2},
                    "ingestion_item": {"max_attempts": 1},
                }
            },
            {"model": 2, "ingestion_item": 1},
        ),
        # A replacement the file sets keeps its value.
        (
            "retry:\n  max_attempts: 1\n  ingestion_item:\n    max_attempts: 1\n",
            [{"path": "retry.model.max_attempts", "value": 5}],
            {
                "retry": {
                    "ingestion_item": {"max_attempts": 1},
                    "model": {"max_attempts": 5},
                }
            },
            {"model": 5, "ingestion_item": 1},
        ),
        # The notice's Remove: the value moves, nothing in force changes.
        (
            "retry:\n  max_attempts: 1\n",
            [{"path": "retry.max_attempts", "unset": True}],
            {
                "retry": {
                    "model": {"max_attempts": 1},
                    "ingestion_item": {"max_attempts": 1},
                }
            },
            {"model": 1, "ingestion_item": 1},
        ),
        # "Use the default" on a replacement the deprecated key also feeds:
        # the value in force stayed the deprecated key's 3 (rv48).
        (
            "retry:\n  max_attempts: 3\n  model:\n    max_attempts: 3\n",
            [{"path": "retry.model.max_attempts", "unset": True}],
            {"retry": {"ingestion_item": {"max_attempts": 3}}},
            {"model": 5, "ingestion_item": 3},
        ),
        # ... and on one only the deprecated key sets.
        (
            "retry:\n  max_attempts: 3\n",
            [{"path": "retry.model.max_attempts", "unset": True}],
            {"retry": {"ingestion_item": {"max_attempts": 3}}},
            {"model": 5, "ingestion_item": 3},
        ),
    ],
)
def test_editing_a_field_a_deprecated_key_sets_saves_only_new_keys(
    tmp_path: Path,
    text: str,
    changes: list[dict[str, object]],
    saved: dict[str, object],
    in_force: dict[str, int],
) -> None:
    """The form names the deprecated key a field is set by, but an edit of
    the field disagreed with it: the review said "would not load" and Save
    stayed off; only the YAML tab could save it (rv43)."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(text)
    file = settings_api.SettingsFile(path)
    preview = file.preview(text=None, changes=changes)
    assert preview["valid"], preview["errors"]
    assert preview["deprecated_keys"] == []
    assert yaml.safe_load(preview["text"]) == saved
    file.save(text=preview["text"], base_sha=settings_api.digest(text))
    retry = file.view()["effective"]["retry"]
    assert {name: retry[name]["max_attempts"] for name in in_force} == in_force


def test_using_the_default_on_a_mapping_only_a_deprecated_key_sets(
    tmp_path: Path,
) -> None:
    """The form offers "Use the default" on a field only a deprecated key
    feeds (rv49): the reset retires the key, so the saved file sets
    neither and the field takes its default."""
    import yaml

    price = {"input_per_million_tokens": 1.0, "output_per_million_tokens": 2.0}
    text = yaml.safe_dump({"evaluation": {"prices": {"m1": price}}})
    path = tmp_path / "config.yaml"
    path.write_text(text)
    file = settings_api.SettingsFile(path)
    assert set(file.view()["effective"]["models"]["prices"]) == {"m1"}
    preview = file.preview(
        text=None, changes=[{"path": "models.prices", "unset": True}]
    )
    assert preview["valid"], preview["errors"]
    assert preview["deprecated_keys"] == []
    assert yaml.safe_load(preview["text"]) in (None, {})
    file.save(text=preview["text"], base_sha=settings_api.digest(text))
    assert file.view()["effective"]["models"]["prices"] == {}


def test_the_settings_view_names_the_keys_the_environment_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A UKR__ variable wins over the file, but the field read "default" and
    an edit of it passed review with no effect (rv43)."""
    monkeypatch.setenv("UKR__RETRY__MODEL_SERVER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("UKR__LOGGING__LEVEL", "warning")
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: INFO\n")
    file = settings_api.SettingsFile(path)
    view = file.view()
    # The deprecated variable by the key it loads into; names, never values.
    assert view["environment_keys"] == {
        "retry.model_server.max_attempts": "UKR__RETRY__MODEL_SERVER_MAX_ATTEMPTS",
        "logging.level": "UKR__LOGGING__LEVEL",
    }
    assert view["effective"]["retry"]["model_server"]["max_attempts"] == 2
    preview = file.preview(
        text=None,
        changes=[
            {"path": "retry.model_server.max_attempts", "value": 4},
            {"path": "retry.model.max_attempts", "value": 3},
        ],
    )
    assert preview["valid"]
    assert preview["environment_overrides"] == [
        {
            "path": "retry.model_server.max_attempts",
            "variable": "UKR__RETRY__MODEL_SERVER_MAX_ATTEMPTS",
        }
    ]


def test_the_settings_view_tells_a_variable_that_only_fills_an_unset_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AWS_REGION sets aws.region_name only while the file leaves it unset,
    but the field read "wins over the file" (rv44): the view names the
    variables that only fill a key, for the form to say so."""
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: INFO\n")
    view = settings_api.SettingsFile(path).view()
    assert view["environment_keys"]["aws.region_name"] == "AWS_REGION"
    assert "AWS_REGION" in view["environment_fallbacks"]
    assert not any(v.startswith("UKR__") for v in view["environment_fallbacks"])
    path.write_text("aws:\n  region_name: eu-west-1\n")
    assert (
        "aws.region_name"
        not in settings_api.SettingsFile(path).view()["environment_keys"]
    )


def test_removing_a_key_a_fallback_variable_fills_is_not_called_no_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing aws.region_name with AWS_REGION set was listed as "no
    effect while the variable is set", but it takes effect: the region in
    force becomes the variable's (rv48)."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    path = tmp_path / "config.yaml"
    path.write_text("aws:\n  region_name: us-east-1\n")
    file = settings_api.SettingsFile(path)
    removed = file.preview(
        text=None, changes=[{"path": "aws.region_name", "value": None, "unset": True}]
    )
    assert removed["environment_overrides"] == []
    assert removed["environment_fills"] == [
        {"path": "aws.region_name", "variable": "AWS_REGION"}
    ]
    # A key the file sets wins over the fallback: nothing to say.
    edited = file.preview(
        text=None, changes=[{"path": "aws.region_name", "value": "ap-northeast-2"}]
    )
    assert edited["environment_overrides"] == edited["environment_fills"] == []


def test_editing_a_price_keeps_the_deprecated_prices() -> None:
    """A mapping: the edit joins the deprecated key's entries."""
    data = {
        "evaluation": {
            "prices": {
                "m1": {"input_per_million_tokens": 1, "output_per_million_tokens": 2}
            }
        }
    }
    result = settings_api.apply_changes(
        data,
        [
            {
                "path": "models.prices.m2",
                "value": {
                    "input_per_million_tokens": 3,
                    "output_per_million_tokens": 4,
                },
            }
        ],
    )
    assert result == {
        "models": {
            "prices": {
                "m1": {"input_per_million_tokens": 1, "output_per_million_tokens": 2},
                "m2": {"input_per_million_tokens": 3, "output_per_million_tokens": 4},
            }
        }
    }
    # An unrelated edit keeps the deprecated key.
    kept = settings_api.apply_changes(data, [{"path": "search.top_k", "value": 5}])
    assert kept["evaluation"] == data["evaluation"]
    # Removing models.prices removes the entries the deprecated key feeds
    # it too, which otherwise stayed in force (rv48).
    both = {**data, "models": {"prices": result["models"]["prices"]}}
    assert (
        settings_api.apply_changes(
            both, [{"path": "models.prices", "value": None, "unset": True}]
        )
        == {}
    )


def test_a_price_field_both_price_keys_set_starts_from_their_merged_prices(
    tmp_path: Path,
) -> None:
    """With models.prices and the deprecated evaluation.prices both set,
    the field showed the file's models.prices alone, and a save of an edit
    of it dropped the evaluation.prices entries the load had merged (rv44).
    The view gives each replacement's value with the deprecated key moved
    into it, which the form starts the field from."""
    import yaml

    price = {"input_per_million_tokens": 1, "output_per_million_tokens": 2}
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {"prices": {"extra-model": price, "m1": price}},
                "models": {"prices": {"m1": {**price, "input_per_million_tokens": 5}}},
            }
        )
    )
    file = settings_api.SettingsFile(path)
    view = file.view()
    merged = view["effective"]["models"]["prices"]
    assert set(merged) == {"extra-model", "m1"}
    (deprecated,) = view["deprecated_keys"]
    assert deprecated["values"] == {
        "models.prices": {
            "extra-model": price,
            "m1": {**price, "input_per_million_tokens": 5},
        }
    }
    # The form's edit: the merged map plus a new entry.
    edited = {**deprecated["values"]["models.prices"], "m3": price}
    preview = file.preview(
        text=None, changes=[{"path": "models.prices", "value": edited}]
    )
    assert preview["valid"], preview["errors"]
    saved = yaml.safe_load(preview["text"])
    assert "evaluation" not in saved
    assert set(saved["models"]["prices"]) == {"extra-model", "m1", "m3"}
    file.save(text=preview["text"], base_sha=view["sha"])
    assert set(file.view()["effective"]["models"]["prices"]) == {
        "extra-model",
        "m1",
        "m3",
    }


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


_OLD_QUERIES = (
    "CREATE TABLE queries (id TEXT PRIMARY KEY, created REAL NOT NULL, "
    "question TEXT NOT NULL, requested_strategy TEXT NOT NULL, strategy TEXT, "
    "suffix TEXT, latency_seconds REAL, outcome TEXT NOT NULL, "
    "abstained INTEGER NOT NULL DEFAULT 0, rerank_status TEXT, usage TEXT, "
    "cost_usd REAL, trace_id TEXT, error TEXT)"
)


def _open_store(path: str, barrier: object, results: object) -> None:
    barrier.wait()  # type: ignore[attr-defined]
    try:
        ConsoleStore(Path(path)).close()
        results.put("ok")  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 - reported to the test
        results.put(f"{type(e).__name__}: {e}")  # type: ignore[attr-defined]


@pytest.mark.timeout(60)
def test_consoles_opening_an_older_store_at_once_each_open_it(
    tmp_path: Path,
) -> None:
    """Each checked for a missing column, then added it: a second console
    opening the same older file in between failed, "duplicate column name"
    (rv43)."""
    import multiprocessing
    import sqlite3

    context = multiprocessing.get_context("spawn")
    failures = []
    for trial in range(2):
        path = tmp_path / f"console{trial}.sqlite3"
        old = sqlite3.connect(path)
        # All of the older schema, so only the added column is missing.
        old.executescript(
            _OLD_QUERIES + ";"
            "CREATE INDEX queries_created ON queries (created);"
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "namespace TEXT NOT NULL, created REAL NOT NULL, record TEXT NOT NULL);"
        )
        old.close()
        count = 6
        barrier, results = context.Barrier(count), context.Queue()
        processes = [
            context.Process(target=_open_store, args=(str(path), barrier, results))
            for _ in range(count)
        ]
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        failures += [o for o in outcomes if o != "ok"]
        columns = {
            r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(queries)")
        }
        assert "abstention_reason" in columns
    assert failures == []


@pytest.mark.timeout(60)
def test_a_console_opens_a_current_store_another_connection_is_reading(
    tmp_path: Path,
) -> None:
    """The open took the write lock to check for missing columns even when
    none was missing: while another connection held a read transaction its
    commit waited and failed after 5 s, "database is locked" (rv44)."""
    import sqlite3

    path = tmp_path / "console.sqlite3"
    ConsoleStore(path).close()
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM queries").fetchone()
    try:
        started = time.monotonic()
        ConsoleStore(path).close()
        assert time.monotonic() - started < 2
    finally:
        reader.rollback()
        reader.close()


def test_a_decline_at_a_query_limit_is_counted_apart_from_the_decline_rate(
    tmp_path: Path,
) -> None:
    """Declines at a query limit (limits.*) read as the corpus's: the rate
    counted them and no figure named them; the store kept no reason (rv42)."""
    import sqlite3

    path = tmp_path / "console.sqlite3"
    # A file an older console wrote, without the reason column.
    old = sqlite3.connect(path)
    old.execute(_OLD_QUERIES)
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
    summary = dashboard.summarize(queries, [], Config())
    totals = summary["totals"]
    # Of the old answer, the answer and the corpus decline: one in three.
    # The declines are the corpus's, in the totals as per strategy (the
    # totals counted the limit's too, 3 beside the row's 1, rv44).
    assert totals["limited"] == 2 and totals["abstained"] == 1
    assert totals["decline_rate"] == pytest.approx(1 / 3)
    # Per strategy too: its declines are the corpus's, the limit's apart
    # (the row counted 3 beside a total that left the 2 out, rv43).
    (local,) = summary["strategies"]
    assert (local["abstained"], local["limited"]) == (1, 2)
    store.close()


def test_a_job_s_cost_is_the_one_its_run_recorded() -> None:
    """A stopped run's cost was priced again at today's prices: a price
    removed since turned its recorded $0.03 into none (rv43)."""
    usage = {
        "by_model": {"m": {"llm_calls": 1, "input_tokens": 1000, "output_tokens": 0}}
    }
    job = {
        "kind": "ingest",
        "result": {"token_usage": usage, "cost_usd": 0.03},
    }
    assert dashboard.job_cost(job, Config()) == 0.03
    # Without one, priced from its usage at models.prices.
    priced = Config(
        models={
            "prices": {
                "m": {"input_per_million_tokens": 2.0, "output_per_million_tokens": 0}
            }
        }
    )
    job["result"] = {"token_usage": usage}
    assert dashboard.job_cost(job, priced) == pytest.approx(0.002)
    assert dashboard.job_cost(job, Config()) is None


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


def test_every_removed_key_reason_is_named_in_the_page_s_language() -> None:
    """The reason a removed key is gone stayed the server's English in every
    locale, where the deprecated-key messages are the page's (rv48): each
    reason has its message, the English one the loader's own words."""
    from unified_kg_rag.domain.models.config import REMOVED_KEYS

    catalog = json.loads(
        (Path(__file__).parents[2] / "console/web/src/i18n/locales/en.json").read_text(
            encoding="utf-8"
        )
    )
    worded = {
        path: catalog.get(
            f"settings.removedReason.{settings_api.removed_reason_id(path)}"
        )
        for path in REMOVED_KEYS
    }
    assert {
        path: text for path, text in worded.items() if text != REMOVED_KEYS[path]
    } == {}
    # One message per reason: the keys of a removed section share it.
    ids = {settings_api.removed_reason_id(path) for path in REMOVED_KEYS}
    assert len(ids) == len(set(REMOVED_KEYS.values()))
    assert sorted(
        k.removeprefix("settings.removedReason.")
        for k in catalog
        if k.startswith("settings.removedReason.")
    ) == sorted(ids)


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


def test_a_cancelled_question_counts_its_spend_but_not_per_question() -> None:
    """rv53: 64 disconnects beside 5 answers showed 58 tokens per question
    for about 700 per answer, and their spend in no total. A cancelled
    question's tokens and cost count in the totals and the model rows; the
    per-question figures read the questions served, and the cancelled ones
    are counted apart."""
    config = Config.model_validate(
        {
            "models": {
                "prices": {
                    "m": {"input_per_million_tokens": 1, "output_per_million_tokens": 1}
                }
            }
        }
    )

    def query(outcome: str, tokens: int, cost: float) -> dict:
        return {
            "created": 1000.0,
            "strategy": "local",
            "outcome": outcome,
            "abstained": False,
            "latency_seconds": 1.0,
            "usage": _usage("m", 1, tokens, 0),
            "cost_usd": cost,
        }

    rows = [query("answered", 700, 0.7) for _ in range(2)]
    rows += [query("cancelled", 100, 0.1) for _ in range(4)]
    summary = dashboard.summarize(rows, [], config)
    totals = summary["totals"]
    assert (totals["queries"], totals["cancelled"], totals["errors"]) == (6, 4, 0)
    assert totals["cost_usd"] == pytest.approx(1.8)
    (model,) = summary["models"]
    assert (model["llm_calls"], model["input_tokens"]) == (6, 1800)
    (strategy,) = summary["strategies"]
    assert strategy["queries"] == 6 and strategy["cancelled"] == 4
    assert strategy["tokens_per_query"] == pytest.approx(700.0)
    assert strategy["cost_per_query"] == pytest.approx(0.7)
    assert strategy["cost_usd"] == pytest.approx(1.8)
    # Only cancelled questions: nothing served to average.
    (only,) = dashboard.summarize(rows[2:], [], config)["strategies"]
    assert (only["tokens_per_query"], only["cost_per_query"]) == (None, None)


def test_the_dashboard_sums_and_prices_the_embedding_requests() -> None:
    # The embedding role's row and its model's cost read the embedding
    # counters, which the summing left out: the row showed zero calls and
    # the model's cost left its requests out.
    usage = {
        "by_role": {"embedding": {"embedding_calls": 2, "embedding_input_tokens": 300}},
        "by_model": {"emb": {"embedding_calls": 2, "embedding_input_tokens": 300}},
    }
    rows = [
        {"id": str(n), "created": 1.0, "outcome": "answered", "usage": usage}
        for n in range(2)
    ]
    priced = Config.model_validate(
        {"models": {"prices": {"emb": {"embedding_per_million_tokens": 0.1}}}}
    )
    summary = dashboard.summarize(rows, [], priced)
    (role,) = summary["roles"]
    assert role["embedding_calls"] == 4 and role["embedding_input_tokens"] == 600
    (model,) = summary["models"]
    assert model["cost_usd"] == pytest.approx(600 * 0.1 / 1e6)
    # An embedding model without a price is named, as a language model is.
    assert dashboard.summarize(rows, [], Config())["unpriced_models"] == ["emb"]
