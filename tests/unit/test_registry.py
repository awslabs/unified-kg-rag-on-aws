# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one registry rule (``shared.registry``) and the registries that
follow it: providers, renderers, schema and seed sources, analyzers and
file loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.document_loaders import BaseLoader

from tests.fixtures.registries import isolate_loaders
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.adapters.local.search_engine import analyzers
from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    get_provider_builder,
    provider_requirement,
    register_provider,
    unregister_provider,
)
from unified_kg_rag.adapters.renderers import (
    BaseRenderer,
    RenderContext,
    get_renderer_class,
    register_renderer,
)
from unified_kg_rag.adapters.schema_sources.registry import _KINDS as SCHEMA_KINDS
from unified_kg_rag.adapters.schema_sources.registry import (
    get_schema_source,
    register_schema_source,
)
from unified_kg_rag.adapters.seed_sources.registry import _KINDS as SEED_KINDS
from unified_kg_rag.adapters.seed_sources.registry import (
    get_seed_source,
    register_seed_source,
)
from unified_kg_rag.shared.registry import Registry

pytestmark = pytest.mark.unit


def test_names_are_case_insensitive_and_a_later_one_replaces() -> None:
    registry: Registry[str, int] = Registry("thing")
    assert registry.register("Thing ", 1, requires=("absent_mod", "install it")) == (
        "thing"
    )
    assert registry.get("THING") == 1 and "thing" in registry
    assert registry.missing("thing") == ("absent_mod", "install it")
    registry.register("thing", 2)
    assert registry.get("thing") == 2 and registry.requirement("thing") is None
    assert registry.names() == ["thing"]


@pytest.mark.parametrize("name", ["my-kind", "", "1st", "_x", "reserved"])
def test_a_name_outside_the_pattern_or_reserved_is_refused(name: str) -> None:
    registry: Registry[str, int] = Registry("thing", reserved={"reserved"})
    with pytest.raises(ValueError, match="Cannot register a thing"):
        registry.register(name, 1)


def test_an_installed_requirement_is_not_missing() -> None:
    registry: Registry[str, int] = Registry("thing")
    registry.register("thing", 1, requires=("json", "stdlib"))
    assert registry.missing("thing") is None


class _Renderer(BaseRenderer):
    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        return []


class _Loader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Any]:
        return iter(())


def _registered(kind: str) -> dict[str, Callable[[], Any]]:
    """Register under ``kind`` in each registry; return a reader of each."""
    builder = register_provider(ProviderRole.LLM, kind)(lambda providers: None)
    register_renderer(kind)(_Renderer)
    register_schema_source(kind)(lambda context: None)
    register_seed_source(kind)(lambda context: None)
    analyzers.register_analyzer(kind)(str.split)
    lowered = kind.strip().lower()
    return {
        "provider": lambda: get_provider_builder(ProviderRole.LLM, lowered) is builder,
        "renderer": lambda: get_renderer_class(lowered) is _Renderer,
        "schema": lambda: get_schema_source(lowered) is not None,
        "seed": lambda: get_seed_source(lowered) is not None,
        "analyzer": lambda: analyzers.get_analyzer(lowered) is str.split,
    }


@pytest.mark.usefixtures("restore_renderer_registry")
def test_every_registry_follows_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry in (SCHEMA_KINDS, SEED_KINDS, analyzers._ANALYZERS):
        monkeypatch.setattr(registry, "entries", dict(registry.entries))
    try:
        readers = _registered("Rule_Probe ")
        assert all(read() for read in readers.values()), readers
        for register in (
            lambda: register_provider(ProviderRole.LLM, "rule-probe"),
            lambda: register_renderer("rule-probe"),
            lambda: register_schema_source("rule-probe"),
            lambda: register_seed_source("rule-probe"),
            lambda: analyzers.register_analyzer("rule-probe"),
        ):
            with pytest.raises(ValueError, match="Cannot register"):
                register()
        register_provider(ProviderRole.LLM, "rule_probe", requires=("m", "i"))(
            lambda providers: None
        )
        assert provider_requirement(ProviderRole.LLM, "RULE_PROBE") == ("m", "i")
    finally:
        unregister_provider(ProviderRole.LLM, "rule_probe")


def test_file_loaders_follow_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_loaders(monkeypatch)
    ParserFactory.register_loader(".RuleProbe", _Loader)
    assert ParserFactory.loader_spec(".ruleprobe")[0] is _Loader
    with pytest.raises(ValueError, match="Cannot register a file loader"):
        ParserFactory.register_loader("ruleprobe", _Loader)
