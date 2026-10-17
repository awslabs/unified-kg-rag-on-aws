# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every configuration field is classified for the stage cache keys.

``cache_keys.CONFIG_KEYING`` says, for each field, from which stage it keys
the stage caches or why it keys none. A field added without a decision
fails here, so a setting that changes a stage's output can no longer be left
out of its key by omission.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from unified_kg_rag.domain.models import Config, PipelineStageType
from unified_kg_rag.domain.models.config import (
    NeptuneIndexingConfig,
    OpenSearchIndexingConfig,
)
from unified_kg_rag.shared.utils import cache_keys
from unified_kg_rag.shared.utils.cache_keys import (
    CONFIG_KEYING,
    Keyed,
    Unkeyed,
    classify,
    extraction_fingerprint,
    stage_input_fingerprint,
)

pytestmark = pytest.mark.unit


def _sections(annotation: Any) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    for arg in get_args(annotation) or (annotation,):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            found.append(arg)
        elif get_origin(arg) is not None:
            found.extend(_sections(arg))
    return found


def _field_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Every non-section field of ``model``'s type tree, as a dotted path."""
    paths: list[str] = []
    for name, field in model.model_fields.items():
        sections = _sections(field.annotation)
        if sections and get_origin(field.annotation) not in (dict, list):
            for section in sections:
                paths.extend(_field_paths(section, f"{prefix}{name}."))
        else:
            paths.append(f"{prefix}{name}")
    return paths


_FIELDS = sorted(set(_field_paths(Config)))


def test_every_config_field_is_classified() -> None:
    unclassified = [path for path in _FIELDS if classify(path) is None]
    assert not unclassified, (
        "Classify these fields in cache_keys.CONFIG_KEYING: Keyed(stage) from "
        "the first stage whose output they shape, or Unkeyed(reason): "
        f"{unclassified}"
    )


def test_every_entry_names_a_config_field() -> None:
    stale = [
        entry
        for entry in CONFIG_KEYING
        if not any(path == entry or path.startswith(f"{entry}.") for path in _FIELDS)
    ]
    assert not stale, f"CONFIG_KEYING entries that match no field: {stale}"


def test_every_unkeyed_entry_gives_a_reason() -> None:
    assert all(
        entry.reason.strip()
        for entry in CONFIG_KEYING.values()
        if isinstance(entry, Unkeyed)
    )


@pytest.mark.parametrize(
    "section, prefix",
    [
        (OpenSearchIndexingConfig, "indexing.opensearch"),
        (NeptuneIndexingConfig, "indexing.neptune"),
    ],
)
def test_store_prefixes_key_from_parsing(section: type[BaseModel], prefix: str) -> None:
    # store_namespace_tag puts every *_prefix field into the registry
    # namespace, part of each document's identity.
    for name in section.model_fields:
        if name.endswith("_prefix"):
            assert classify(f"{prefix}.{name}") == Keyed(
                PipelineStageType.DOCUMENT_PARSING, per_document=False
            ), name


def _local(**llm: object) -> Config:
    return Config.model_validate({"backend": "local", "providers": {"llm": llm}})


@pytest.mark.parametrize(
    "field, value",
    [
        ("temperature", 0.7),
        ("max_output_tokens", 512),
        ("context_window_size", 4096),
    ],
)
def test_local_model_knobs_key_every_stage(field: str, value: object) -> None:
    # They change what every local model call returns; the allow-list the
    # keys were built from before missed them.
    baseline, changed = _local(), _local(**{field: value})
    for stage in PipelineStageType:
        assert stage_input_fingerprint(changed, stage) != stage_input_fingerprint(
            baseline, stage
        ), stage
    assert extraction_fingerprint(changed) != extraction_fingerprint(baseline)


def test_settings_of_the_inactive_model_provider_do_not_key() -> None:
    stage = PipelineStageType.COMMUNITY_DETECTION
    bedrock = Config.model_validate({"providers": {"llm": {"temperature": 0.9}}})
    assert stage_input_fingerprint(bedrock, stage) == stage_input_fingerprint(
        Config(), stage
    )
    local = Config.model_validate(
        {"backend": "local", "aws": {"bedrock": {"default_effort": "max"}}}
    )
    assert stage_input_fingerprint(local, stage) == stage_input_fingerprint(
        _local(), stage
    )


def test_an_unclassified_field_keys_every_stage(monkeypatch) -> None:
    # Forgetting to classify a field must cost a cache miss, not a stale hit.
    keying = dict(CONFIG_KEYING)
    del keying["processing.max_concurrency"]
    retuned = Config()
    retuned.processing.max_concurrency += 1
    stage = PipelineStageType.DOCUMENT_PARSING
    assert stage_input_fingerprint(retuned, stage) == stage_input_fingerprint(
        Config(), stage
    )
    monkeypatch.setattr(cache_keys, "CONFIG_KEYING", keying)
    assert stage_input_fingerprint(retuned, stage) != stage_input_fingerprint(
        Config(), stage
    )
