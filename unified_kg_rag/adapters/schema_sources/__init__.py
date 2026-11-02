# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extraction-schema sources and their registry.

Importing this package registers the built-in sources (``file``).
"""

# Importing a source module registers it.
from . import file  # noqa: F401
from .registry import (
    SchemaLoader,
    SchemaSourceContext,
    get_schema_source,
    redact_location,
    register_schema_source,
    registered_schema_sources,
)

__all__ = [
    "SchemaLoader",
    "SchemaSourceContext",
    "get_schema_source",
    "redact_location",
    "register_schema_source",
    "registered_schema_sources",
]
