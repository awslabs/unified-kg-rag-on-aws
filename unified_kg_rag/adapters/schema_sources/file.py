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

``path`` is a local file, or an S3 object (``s3://bucket/key.yaml``) read
with the run's session (``s3:GetObject``; at most :data:`MAX_S3_BYTES`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from unified_kg_rag.adapters.source_files.locations import (
    read_source_bytes,
    resolve_source_path,
    split_s3,
)
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError
from unified_kg_rag.shared.utils.files import read_user_text
from unified_kg_rag.shared.utils.validation import format_validation_errors

from .registry import SchemaSourceContext, register_schema_source

_SUFFIXES = (".yaml", ".yml", ".json")
# The most bytes an S3 schema file may hold (checked before it is fetched).
MAX_S3_BYTES = 16 * 1024 * 1024


@register_schema_source("file")
def load_schema_file(context: SchemaSourceContext) -> SchemaDocument:
    """The schema in the local file or S3 object at ``source.path``.

    Raises:
        SchemaSourceError: No path is set, the path is a URL other than
            ``file://`` or ``s3://`` (an HTTP location needs a source
            registered for it),
            ``options`` or ``secret_arn`` is set, or the file is missing,
            unreadable (a directory or an S3 prefix, say; access denied),
            larger than :data:`MAX_S3_BYTES` in S3, not a YAML/JSON mapping,
            or holds an unknown key or an invalid item.
    """
    source = context.source
    if not source.path:
        raise SchemaSourceError(
            "processing.graph_extraction.schema_source.path is required for "
            "kind 'file'"
        )
    resolved = resolve_source_path(
        source.path,
        refusal=(
            f"Schema file '{context.location}' is a URL; kind 'file' reads a "
            "local path (or a file:// URL) or an s3:// object only. Copy the "
            "file next to the config (or into the image, or S3), or register "
            "a schema source for the location's scheme"
        ),
    )
    if source.options or source.secret_arn:
        raise SchemaSourceError(
            "kind 'file' takes no schema_source.options or secret_arn; "
            "leave them unset"
        )
    if isinstance(resolved, str):
        return _load_s3(context, resolved)
    path = resolved
    if path.suffix.lower() not in _SUFFIXES:
        raise SchemaSourceError(
            f"Schema file '{path}' must end in {', '.join(_SUFFIXES)}"
        )
    try:
        text = read_user_text(path)
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
    return _document(str(path), path.suffix.lower(), text)


def _load_s3(context: SchemaSourceContext, location: str) -> SchemaDocument:
    name = context.location
    _, key = split_s3(location)
    suffix = Path(key.rpartition("/")[2]).suffix.lower()
    # A prefix (no key, or a trailing "/") is refused by read_source_bytes.
    if key and not key.endswith("/") and suffix not in _SUFFIXES:
        raise SchemaSourceError(
            f"Schema file '{name}' must end in {', '.join(_SUFFIXES)}"
        )
    data = read_source_bytes(context, location, max_bytes=MAX_S3_BYTES)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SchemaSourceError(
            f"Schema file '{name}' is not valid UTF-8 (byte offset {e.start})"
        ) from e
    return _document(name, suffix, text)


def _document(name: str, suffix: str, text: str) -> SchemaDocument:
    data = _parse(name, suffix, text)
    try:
        return SchemaDocument.model_validate(data)
    except ValidationError as e:
        raise SchemaSourceError(
            f"Invalid schema file '{name}': {format_validation_errors('', e)}"
        ) from e


def _parse(path: str, suffix: str, text: str) -> Any:
    try:
        data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise SchemaSourceError(
            f"Schema file '{path}' does not parse: {' '.join(str(e).split())}"
        ) from e
    if not isinstance(data, dict):
        raise SchemaSourceError(
            f"Schema file '{path}' must hold a mapping, got {type(data).__name__}"
        )
    return data
