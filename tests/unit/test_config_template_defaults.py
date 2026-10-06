# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""config-template.yaml must load as-is and document the real defaults.

Operators copy the template and edit a few keys, so (a) it must validate as a
``Config`` without edits, and (b) every value it shows must be the value the
code uses when the key is omitted. A template value that silently differs from
the model default reads as authoritative while the running system does
something else. Intentional differences are listed in ``_INTENTIONAL`` with the
reason; anything else is drift to fix in the template or the model.

AWS-free: only YAML parsing and Pydantic validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from unified_kg_rag.adapters.renderers.interactive import InteractiveRenderer
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_TEMPLATE = Path(__file__).resolve().parents[2] / "config-template.yaml"

# Dotted template path -> why the template value may differ from Config().
_INTENTIONAL: dict[str, str] = {
    "aws.bedrock.region_name": (
        "the template keeps Bedrock in the example deployment region: in a "
        "private-VPC deployment Bedrock is reached through the VPC endpoint, "
        "so a different Bedrock region would hang"
    ),
    "graph.visualization.interactive.max_nodes": (
        "free-form renderer dict; the default lives in "
        "InteractiveRenderer.DEFAULT_MAX_NODES (checked separately below)"
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


def test_template_values_equal_model_defaults(raw_template: dict[str, Any]) -> None:
    # Compare validated values, so "0.5" vs 0.5 or an enum vs its string value
    # are not reported as drift.
    from_template = Config(**raw_template).model_dump(mode="json")
    defaults = Config().model_dump(mode="json")

    drift = []
    for path in _leaves(raw_template):
        dotted = ".".join(path)
        if dotted in _INTENTIONAL:
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


def test_interactive_max_nodes_matches_renderer_default(
    raw_template: dict[str, Any],
) -> None:
    shown = raw_template["graph"]["visualization"]["interactive"]["max_nodes"]
    assert shown == InteractiveRenderer.DEFAULT_MAX_NODES
