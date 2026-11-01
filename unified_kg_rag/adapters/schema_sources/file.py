# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``schema_source.kind: file``: a YAML or JSON file of the inline keys.

The file holds ``entity_types`` (optional: without it the config's stay),
``relationship_types`` and ``relationship_patterns``, in the forms the
config takes::

    entity_types:
      - "PERSON: People"
      - {label: ORGANIZATION, iri: "https://schema.org/Organization"}
    relationship_types:
      - "WORKS_FOR: Employment"
    relationship_patterns:
      - "PERSON -WORKS_FOR-> ORGANIZATION"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError

from .registry import register_schema_source

_SUFFIXES = (".yaml", ".yml", ".json")


@register_schema_source("file")
def load_schema_file(source: SchemaSourceConfig) -> SchemaDocument:
    """The schema in the file at ``source.path``.

    Raises:
        SchemaSourceError: No path is set, or the file is missing,
            unreadable, not a YAML/JSON mapping, or holds an unknown key or
            an invalid item.
    """
    if not source.path:
        raise SchemaSourceError(
            "processing.graph_extraction.schema_source.path is required for "
            "kind 'file'"
        )
    path = Path(source.path)
    if path.suffix.lower() not in _SUFFIXES:
        raise SchemaSourceError(
            f"Schema file '{path}' must end in {', '.join(_SUFFIXES)}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise SchemaSourceError(f"Schema file '{path}' not found") from e
    except UnicodeDecodeError as e:
        raise SchemaSourceError(
            f"Schema file '{path}' is not valid UTF-8 (byte offset {e.start})"
        ) from e
    except OSError as e:
        raise SchemaSourceError(
            f"Cannot read schema file '{path}': {e.strerror or e}"
        ) from e
    data = _parse(path, text)
    try:
        return SchemaDocument.model_validate(data)
    except ValidationError as e:
        raise SchemaSourceError(f"Invalid schema file '{path}': {_summary(e)}") from e


def _parse(path: Path, text: str) -> Any:
    try:
        data = (
            json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise SchemaSourceError(
            f"Schema file '{path}' does not parse: {' '.join(str(e).split())}"
        ) from e
    if not isinstance(data, dict):
        raise SchemaSourceError(
            f"Schema file '{path}' must hold a mapping, got {type(data).__name__}"
        )
    return data


def _summary(error: ValidationError) -> str:
    """Each invalid key and why, without the rejected values."""
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc']) or '<root>'}: "
        f"{detail['msg']}"
        for detail in error.errors(include_url=False, include_input=False)
    )
