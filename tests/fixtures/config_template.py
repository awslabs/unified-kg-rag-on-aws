# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The "uncommented view" of ``config-template.yaml``.

The template shows some defaults as commented-out keys (the connection keys
``backend: local`` fills in, the model ids that follow ``providers``, the whole
``providers`` section), so a copy of it switches backends with one key. Plain
YAML parsing drops those lines, so a test comparing the template against the
defaults would never check them. :func:`uncommented_template` returns the
template with every commented key line that names a real config field
uncommented, and :func:`commented_key_paths` the dotted paths it added.

Inside a commented-out block (``# providers:``) every key-shaped line is a
key, so a misspelled one reaches the unknown-key check. Elsewhere a commented
line counts as a key only when its path resolves in the ``Config`` schema and
the template does not already set it: prose such as
``# true: a reduce LLM ...`` and examples under a free-form mapping
(``model_overrides``) stay comments.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from unified_kg_rag.application.config import _nested_model
from unified_kg_rag.domain.models import Config

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config-template.yaml"

_COMMENTED_KEY = re.compile(
    r"^(?P<indent> *)# (?P<pad> *)(?P<key>[A-Za-z_]\w*):(?: +(?P<value>\S.*))?$"
)
_REAL_KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_]\w*|\"[^\"]+\"):")


def _model_at(path: tuple[str, ...]) -> type[BaseModel] | None:
    """The config model at ``path`` (None if it is not a nested model)."""
    model: type[BaseModel] = Config
    for key in path:
        field = model.model_fields.get(key)
        if field is None:
            return None
        nested = _nested_model(field.annotation)
        if nested is None:
            return None
        model = nested
    return model


def _is_set(tree: Any, path: tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return False
        tree = tree[key]
    return True


@cache
def _view(text: str) -> tuple[str, tuple[str, ...]]:
    raw = yaml.safe_load(text) or {}
    lines = text.splitlines()
    out: list[str] = []
    added: list[str] = []
    # (indent, path, accepted, commented) of the enclosing keys.
    stack: list[tuple[int, tuple[str, ...], bool, bool]] = []
    for line in lines:
        match = _COMMENTED_KEY.match(line)
        real = None if line.lstrip().startswith("#") else _REAL_KEY.match(line)
        if real is not None:
            indent = len(real["indent"])
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else ()
            stack.append((indent, (*parent, real["key"].strip('"')), True, False))
            out.append(line)
            continue
        if match is None:
            out.append(line)
            continue
        indent = len(match["indent"]) + len(match["pad"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent_path, parent_ok, in_block = stack[-1][1:] if stack else ((), True, False)
        path = (*parent_path, match["key"])
        model = _model_at(parent_path) if parent_ok else None
        accepted = (parent_ok and in_block) or (
            model is not None
            and match["key"] in model.model_fields
            and not _is_set(raw, path)
        )
        if match["value"] is None:
            stack.append((indent, path, accepted, True))
        if not accepted:
            out.append(line)
            continue
        out.append(" " * indent + line.split("# ", 1)[1].lstrip())
        if match["value"] is not None:
            added.append(".".join(path))
    return "\n".join(out) + "\n", tuple(added)


def uncommented_template(text: str | None = None) -> dict[str, Any]:
    """The template with its commented-out config keys uncommented."""
    source = TEMPLATE_PATH.read_text(encoding="utf-8") if text is None else text
    data = yaml.safe_load(_view(source)[0])
    assert isinstance(data, dict)
    return data


def commented_key_paths(text: str | None = None) -> tuple[str, ...]:
    """Dotted paths of the leaf keys the template shows only as comments."""
    source = TEMPLATE_PATH.read_text(encoding="utf-8") if text is None else text
    return _view(source)[1]
