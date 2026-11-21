# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The namespace's one user-facing name, and the deprecated names it had.

A namespace (the name every index and graph label of a corpus ends in) was
called the *index suffix* (``--suffix``, ``RAGInput.suffix``) and the
*index value* (``processing.document_parsing.index_value``, the console's
ingest request). Every surface now says ``namespace``; each old name still
works as a deprecated alias, through the one mapping here: a flag, a model
field (``RAGInput``, the console's requests) and a config key
(``DEPRECATED_KEYS`` lists the config's) take it the same way, the old and
the new name given two different values refused.
"""

from typing import Any

from unified_kg_rag.shared.utils.deprecation import deprecation_message, warn_deprecated

# Each deprecated name of the namespace -> the name that replaces it.
DEPRECATED_NAMESPACE_NAMES: dict[str, str] = {
    "--suffix": "--namespace",
    "suffix": "namespace",
    "index_value": "namespace",
}


def deprecated_namespace_warning(old: str) -> str:
    """What a caller that names the namespace by ``old`` is told."""
    return deprecation_message(
        old, f"it names the namespace; use {DEPRECATED_NAMESPACE_NAMES[old]} instead"
    )


def pick_namespace(namespace: str | None, old: str, value: str | None) -> str | None:
    """The namespace a caller named, by its name (``namespace``) or by the
    deprecated ``old`` (``value``).

    Raises:
        ValueError: both are given, with different values.
    """
    if value is None:
        return namespace
    if namespace is not None and namespace != value:
        new = DEPRECATED_NAMESPACE_NAMES[old]
        raise ValueError(
            f"{old} '{value}' and {new} '{namespace}' name different "
            f"namespaces; give {new} alone"
        )
    return value


def namespace_argument(namespace: str | None, suffix: str | None) -> str | None:
    """A function's ``namespace`` argument, or its deprecated keyword
    ``suffix`` (a ``DeprecationWarning`` at the caller's line).

    Raises:
        ValueError: both are given, with different values.
    """
    chosen = pick_namespace(namespace, "suffix", suffix)
    if suffix is not None:
        warn_deprecated(deprecated_namespace_warning("suffix"))
    return chosen


def migrate_namespace_field(data: Any, old: str) -> Any:
    """A model's input with the deprecated field ``old`` moved into
    ``namespace`` (a ``DeprecationWarning`` at the caller's line); for a
    ``model_validator(mode="before")``.

    Raises:
        ValueError: the input sets both, to different values.
    """
    if not isinstance(data, dict) or old not in data:
        return data
    moved = {k: v for k, v in data.items() if k != old}
    moved["namespace"] = pick_namespace(data.get("namespace"), old, data[old])
    warn_deprecated(deprecated_namespace_warning(old))
    return moved
