# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merging configuration overrides over a base tree."""

import copy
from typing import Any

from pydantic import (
    BaseModel,
)


def _plain(value: Any) -> Any:
    """``value`` as plain data: a section instance as its full dump."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return copy.deepcopy(value)


def merge_over(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every leaf of ``top`` written over it (nested dicts merge)."""
    merged = dict(base)
    for key, value in top.items():
        below = merged.get(key)
        merged[key] = (
            merge_over(below, value)
            if isinstance(value, dict) and isinstance(below, dict)
            else value
        )
    return merged


def _changes(before: BaseModel, after: BaseModel) -> dict[str, Any]:
    """The fields of ``after`` that differ from ``before``, as nested data."""
    changes: dict[str, Any] = {}
    for name in type(after).model_fields:
        old, new = getattr(before, name), getattr(after, name)
        if isinstance(old, BaseModel) and type(old) is type(new):
            if nested := _changes(old, new):
                changes[name] = nested
        elif old != new:
            changes[name] = _plain(new)
    return changes
