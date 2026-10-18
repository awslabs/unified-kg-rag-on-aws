# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""config-template.yaml must load as-is and document the real defaults.

Operators copy the template and edit a few keys, so (a) it must validate as a
``Config`` without edits, and (b) every value it shows must be the value the
code uses when the key is omitted. A template value that silently differs from
the model default reads as authoritative while the running system does
something else. Intentional differences are listed in ``_INTENTIONAL`` with the
reason; anything else is drift to fix in the template or the model.

The keys the template shows commented out (the connection keys
``backend: local`` fills in, the ``providers`` section, model ids that follow
it) are checked through the uncommented view of the template
(``tests.fixtures.config_template``): against ``Config()``, and, for the keys
``LOCAL_PRESET`` sets, by checking that ``backend: local`` alone switches the
template to the preset.

AWS-free: only YAML parsing and Pydantic validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from tests.fixtures.config_template import (
    TEMPLATE_PATH,
    commented_key_paths,
    uncommented_template,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.provider_config import LOCAL_PRESET

pytestmark = pytest.mark.unit

_TEMPLATE = TEMPLATE_PATH

# Dotted template path -> why the template value may differ from Config().
_INTENTIONAL: dict[str, str] = {}

# Commented-out keys that show an example rather than the default, and why.
_COMMENTED_EXAMPLES: dict[str, str] = {
    "graph.visualization.outputs_directory": (
        "unset by default (outputs go under the pipeline cache directory); "
        "the comment shows a path to set"
    ),
}

_MISSING = object()


def _leaves(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Yield the path of every non-mapping value (scalars and lists)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, (*path, str(key)))
    else:
        yield path


def _lookup(tree: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return _MISSING
        tree = tree[key]
    return tree


@pytest.fixture(scope="module")
def raw_template() -> dict[str, Any]:
    data = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_template_loads_as_config(raw_template: dict[str, Any]) -> None:
    Config(**raw_template)


@pytest.fixture(scope="module")
def full_template() -> dict[str, Any]:
    """The template with its commented-out keys uncommented."""
    return uncommented_template()


def test_uncommented_view_finds_the_commented_defaults() -> None:
    # Guards the parser: a template edit that breaks the commented-key layout
    # would otherwise silently shrink what the drift checks below compare.
    paths = set(commented_key_paths())
    assert {
        "aws.neptune.port",
        "aws.neptune.use_iam",
        "aws.opensearch.allow_anonymous",
        "providers.llm.kind",
        "providers.rerank.batch_size",
        "providers.doc_status.sqlite_path",
    } <= paths
    assert len(paths) >= 35
    # Examples under free-form mappings and prose stay comments.
    assert not any(
        p.startswith(("aws.bedrock.model_overrides", "graph.visualization.embeddings"))
        for p in paths
    )


def test_uncommented_template_loads_as_config(full_template: dict[str, Any]) -> None:
    Config(**full_template)


@pytest.mark.parametrize("view", ["as_shipped", "uncommented"])
def test_template_values_equal_model_defaults(
    raw_template: dict[str, Any], full_template: dict[str, Any], view: str
) -> None:
    # Compare validated values, so "0.5" vs 0.5 or an enum vs its string value
    # are not reported as drift.
    shown_template = raw_template if view == "as_shipped" else full_template
    from_template = Config(**shown_template).model_dump(mode="json")
    defaults = Config().model_dump(mode="json")

    drift = []
    for path in _leaves(shown_template):
        dotted = ".".join(path)
        if dotted in _INTENTIONAL or (
            view == "uncommented" and dotted in _COMMENTED_EXAMPLES
        ):
            continue
        shown = _lookup(from_template, path)
        default = _lookup(defaults, path)
        if shown != default:
            drift.append(
                f"{dotted}: template={shown!r} default="
                f"{'<absent>' if default is _MISSING else repr(default)}"
            )
    assert not drift, "config-template.yaml differs from Config() defaults:\n" + (
        "\n".join(drift)
    )


# Tuning knobs the template must keep documenting. The drift check above only
# compares keys the template shows, so dropping one of these would pass it.
_DOCUMENTED_KNOBS = (
    "search.token_manager.type_budgets",
    "search.token_manager.min_truncated_section_tokens",
    "search.local_search.type_quota",
    "search.lightrag_search.kg_stream_top_k",
    "search.lightrag_search.chunk_stream_top_k",
    "search.lightrag_search.related_chunk_number",
    "indexing.neptune.property_max_length",
    "evaluation.judge_effort",
    "evaluation.ragas_timeout",
    "evaluation.ragas_max_workers",
    "evaluation.ragas_max_attempts",
)


@pytest.mark.parametrize("dotted", _DOCUMENTED_KNOBS)
def test_tuning_knobs_are_documented_in_the_template(
    raw_template: dict[str, Any], dotted: str
) -> None:
    assert _lookup(raw_template, tuple(dotted.split("."))) is not _MISSING


def test_neptune_property_cap_exceeds_the_unsummarized_description_length() -> None:
    # Descriptions below the summarization threshold are stored as merged; a
    # Neptune property cap shorter than that (~4 characters per token) cut them
    # mid-sentence in the local/DRIFT context. The template value is held to
    # this default by test_template_values_equal_model_defaults.
    config = Config()
    summarization = config.processing.graph_extraction.description_summarization
    threshold_tokens = summarization.force_summary_threshold_tokens
    assert config.indexing.neptune.property_max_length >= 4 * threshold_tokens


def test_intentional_differences_are_still_present(
    raw_template: dict[str, Any],
) -> None:
    # Keep the allowlist honest: an entry whose key no longer exists (or no
    # longer differs) should be removed rather than silently exempting it.
    from_template = Config(**raw_template).model_dump(mode="json")
    defaults = Config().model_dump(mode="json")
    stale = [
        dotted
        for dotted in _INTENTIONAL
        if _lookup(raw_template, tuple(dotted.split("."))) is _MISSING
        or _lookup(from_template, tuple(dotted.split(".")))
        == _lookup(defaults, tuple(dotted.split(".")))
    ]
    assert not stale, f"stale _INTENTIONAL entries: {stale}"


def test_commented_examples_are_still_commented_examples(
    full_template: dict[str, Any],
) -> None:
    defaults = Config().model_dump(mode="json")
    shown = Config(**full_template).model_dump(mode="json")
    stale = [
        dotted
        for dotted in _COMMENTED_EXAMPLES
        if dotted not in commented_key_paths()
        or _lookup(shown, tuple(dotted.split(".")))
        == _lookup(defaults, tuple(dotted.split(".")))
    ]
    assert not stale, f"stale _COMMENTED_EXAMPLES entries: {stale}"


def test_backend_local_alone_switches_the_template_to_the_preset(
    raw_template: dict[str, Any],
) -> None:
    # The keys LOCAL_PRESET fills are commented out in the template, so a copy
    # with only `backend: local` changed gets every preset value.
    local = Config(**{**raw_template, "backend": "local"}).model_dump(mode="json")
    preset = Config(**LOCAL_PRESET).model_dump(mode="json")
    differ = [
        ".".join(path)
        for path in _leaves(LOCAL_PRESET)
        if _lookup(local, path) != _lookup(preset, path)
    ]
    assert not differ, f"template keys override the local preset: {differ}"
