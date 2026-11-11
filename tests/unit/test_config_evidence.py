# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every numeric configuration default says what it rests on.

A numeric default (an int or a float, not a bool) carries an evidence tag
(``domain.models.evidence.evidence`` in its ``json_schema_extra``):
measured by a run, the original's value, a departure from it, an operational
bound, or unmeasured. The configuration reference prints the tag and lists
the defaults no run compared (``scripts/gen_config_reference.py``).
"""

from __future__ import annotations

import importlib.util
import inspect
import types
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from tests.fixtures.docs_check import check_docs
from unified_kg_rag.domain.models.config import Config
from unified_kg_rag.domain.models.evidence import (
    EVIDENCE_KINDS,
    EVIDENCE_PAGE,
    evidence,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _models(annotation: object) -> Iterator[type[BaseModel]]:
    """The models an annotation holds: bare, optional, or as list/dict items."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for arg in typing.get_args(annotation):
        yield from _models(arg)


def _numeric(annotation: object) -> bool:
    if annotation in (int, float):
        return True
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType, typing.Annotated):
        return any(_numeric(arg) for arg in typing.get_args(annotation))
    return False


def _fields(
    cls: type[BaseModel], path: str = "", seen: frozenset[type] = frozenset()
) -> Iterator[tuple[str, FieldInfo]]:
    """Every leaf field of the configuration tree, by its dotted path."""
    for name, info in cls.model_fields.items():
        key = f"{path}.{name}" if path else name
        subs = [sub for sub in _models(info.annotation) if sub not in seen]
        if subs:
            for sub in subs:
                yield from _fields(sub, key, seen | {cls})
            continue
        yield key, info


def _numeric_defaults() -> list[tuple[str, FieldInfo]]:
    return [
        (key, info)
        for key, info in _fields(Config)
        if _numeric(info.annotation)
        and not info.is_required()
        and isinstance(info.get_default(call_default_factory=True), int | float)
        and not isinstance(info.get_default(call_default_factory=True), bool)
    ]


def _tag(info: FieldInfo) -> dict[str, object]:
    extra = info.json_schema_extra
    return extra if isinstance(extra, dict) else {}


def test_the_tree_has_numeric_defaults() -> None:
    # Guards the walk itself: a walk that found none would pass the rest.
    assert len(_numeric_defaults()) > 150


def test_every_numeric_default_is_tagged() -> None:
    untagged = [
        key
        for key, info in _numeric_defaults()
        if _tag(info).get("evidence") not in EVIDENCE_KINDS
    ]
    assert not untagged, (
        "numeric defaults without an evidence tag (json_schema_extra="
        "evidence(...), domain/models/evidence.py): " + ", ".join(untagged)
    )


def test_fragment_refs_name_a_section_of_the_evidence_page() -> None:
    sections = check_docs().fragments(REPO_ROOT / EVIDENCE_PAGE)
    broken = [
        f"{key}: {ref}"
        for key, info in _fields(Config)
        if isinstance(ref := _tag(info).get("ref"), str)
        and ref.startswith("#")
        and ref[1:] not in sections
    ]
    assert not broken, f"refs to no section of {EVIDENCE_PAGE}: {broken}"


@pytest.mark.parametrize(
    ("kind", "ref", "message"),
    [
        ("guessed", None, "unknown evidence kind"),
        ("measured", None, "needs a ref"),
        ("upstream", "", "needs a ref"),
        ("deviation", None, "needs a ref"),
        ("measured", "the gleaning run", "#fragment"),
    ],
)
def test_evidence_refuses_a_tag_it_cannot_back(
    kind: str, ref: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evidence(kind, ref)  # type: ignore[arg-type]


def test_evidence_records_the_ref_only_when_given() -> None:
    assert evidence("operational") == {"evidence": "operational"}
    assert evidence("upstream", "RAPTOR's 10") == {
        "evidence": "upstream",
        "ref": "RAPTOR's 10",
    }


def _generator() -> types.ModuleType:
    script = REPO_ROOT / "scripts" / "gen_config_reference.py"
    spec = importlib.util.spec_from_file_location("gen_config_reference", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_lists_exactly_the_defaults_no_run_compared() -> None:
    text = _generator().render()
    listed = text.split("## Defaults no run compared", 1)[1].split("\n## ", 1)[0]
    rows = {
        line.split("`")[1] for line in listed.splitlines() if line.startswith("| `")
    }
    expected = {
        key
        for key, info in _numeric_defaults()
        if _tag(info).get("evidence") in ("deviation", "unmeasured")
    }
    assert rows == expected


def test_reference_prints_each_tag() -> None:
    text = _generator().render()
    assert (
        "| `max_rounds` | integer (≥ 1) | `3` | "
        "[measured](../concepts/defaults-and-evidence.md#gleaning-rounds) |"
    ) in text
    assert "| deviation: MS GraphRAG runs 1 |" in text
