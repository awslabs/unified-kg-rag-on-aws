# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugins are loaded from an entry point (or ``UKR_PLUGINS``) by the CLIs
and the composition root, so a kind a plugin registers can be selected
without a fork.

``register_provider`` and the other registries worked only for a library
caller that imported its plugin first: ``UKR__PROVIDERS__LLM__KIND=...``
on any CLI exited 2 with "No llm provider is registered".
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs this environment's interpreter
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from unified_kg_rag.adapters import provider_registry
from unified_kg_rag.adapters.provider_registry import ProviderRole, registered_kinds
from unified_kg_rag.application import plugins
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit

KIND = "synthetic_plugin_server"

_PLUGIN = f"""
from unified_kg_rag import ProviderRole, register_provider

CALLS = []


@register_provider(ProviderRole.LLM, "{KIND}")
def _build(providers):
    from unified_kg_rag.adapters.local.models import (
        OpenAICompatibleLanguageModelFactory,
    )

    return OpenAICompatibleLanguageModelFactory(providers.config)


@register_provider(ProviderRole.TOKEN_COUNTER, "{KIND}")
def _counter(providers):
    from unified_kg_rag.adapters.local.token_counter import LocalTokenCounter

    return lambda model_id, **_: LocalTokenCounter(model_id)


def register():
    CALLS.append(1)
"""


def _distribution(root: Path, value: str = "ukr_synthetic_plugin") -> Path:
    """A distribution installed in ``root`` (a ``sys.path`` entry) whose
    ``unified_kg_rag.plugins`` entry point is ``value``."""
    root.mkdir(parents=True)
    (root / "ukr_synthetic_plugin.py").write_text(_PLUGIN, encoding="utf-8")
    info = root / "ukr_synthetic_plugin-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ukr-synthetic-plugin\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (info / "entry_points.txt").write_text(
        f"[{plugins.ENTRY_POINT_GROUP}]\nsynthetic = {value}\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = _distribution(tmp_path / "site")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delenv(plugins.PLUGINS_VARIABLE, raising=False)
    monkeypatch.setattr(plugins, "_loaded", set())
    yield root
    sys.modules.pop("ukr_synthetic_plugin", None)
    for role in ProviderRole:
        provider_registry.unregister_provider(role, KIND)


def test_an_entry_point_plugin_is_loaded_once(installed: Path) -> None:
    assert plugins.load_plugins() == ["synthetic"]
    assert KIND in registered_kinds(ProviderRole.LLM)
    assert plugins.load_plugins() == []


def test_the_composition_root_loads_it_before_checking_the_kinds(
    installed: Path,
) -> None:
    config = Config(providers={"llm": {"kind": KIND}}, models={"default": "m"})
    providers = build_providers(config)
    assert (
        type(providers.llm_factory).__name__ == "OpenAICompatibleLanguageModelFactory"
    )


def test_ukr_plugins_names_the_plugins_instead(
    installed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plugins.PLUGINS_VARIABLE, "")
    assert plugins.load_plugins() == []
    assert KIND not in registered_kinds(ProviderRole.LLM)
    monkeypatch.setenv(plugins.PLUGINS_VARIABLE, " ukr_synthetic_plugin:register ,")
    assert plugins.load_plugins() == ["ukr_synthetic_plugin:register"]
    assert sys.modules["ukr_synthetic_plugin"].CALLS == [1]
    assert KIND in registered_kinds(ProviderRole.LLM)


def test_a_plugin_that_cannot_be_loaded_is_named(
    installed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plugins.PLUGINS_VARIABLE, "ukr_no_such_plugin")
    with pytest.raises(ConfigurationError, match="'ukr_no_such_plugin'"):
        plugins.load_plugins()
    monkeypatch.setenv(plugins.PLUGINS_VARIABLE, "ukr_synthetic_plugin:CALLS")
    with pytest.raises(ConfigurationError, match="not callable"):
        plugins.load_plugins()


def test_a_cli_selects_a_kind_its_plugin_registers(tmp_path: Path) -> None:
    # In a process of its own, as a deployed CLI runs: the provider check
    # passes, and the run gets as far as the empty namespace.
    root = _distribution(tmp_path / "site")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.yaml").write_text("backend: local\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(root), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "AWS_EC2_METADATA_DISABLED": "true",
        "UKR__PROVIDERS__LLM__KIND": KIND,
        "UKR__MODELS__DEFAULT": "synthetic-model",
    }
    env.pop(plugins.PLUGINS_VARIABLE, None)
    command = [
        sys.executable,
        "-m",
        "unified_kg_rag.application.cli.run_describe",
        "--config-path",
        "config.yaml",
        "--output-format",
        "json",
    ]
    done = subprocess.run(  # nosec B603 - this environment's interpreter
        command,
        capture_output=True,
        text=True,
        timeout=100,
        cwd=workspace,
        env=env,
        check=False,
    )
    status = json.loads(done.stdout)
    assert status["error_class"] == "index_not_found", done.stderr
    # Without the plugin the kind is unknown.
    env[plugins.PLUGINS_VARIABLE] = ""
    done = subprocess.run(  # nosec B603 - this environment's interpreter
        command,
        capture_output=True,
        text=True,
        timeout=100,
        cwd=workspace,
        env=env,
        check=False,
    )
    assert "No llm provider is registered" in json.loads(done.stdout)["error"]
