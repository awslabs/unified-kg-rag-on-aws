# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One identity rule for source documents.

A document is identified by its path *relative to the corpus root*, and a
document *version* by that path plus a hash of its full text. The version id
does not depend on where the corpus sits (the same corpus synced to
``/tmp/graphrag-source`` on one machine and checked out under ``~/corpus`` on
another yields the same ``document_id`` and text-unit ids). The registry key
does: it also holds the corpus source scope, which defaults to the resolved
source directory, so a corpus keeps its registry records across runs only
from the same directory or with a fixed ``document_parsing.source_scope``
(the records of a corpus that moved stay under the old scope until a run
retires it, see ``indexing.retire_source_scopes``):

* ``compute_doc_id(relative_path, namespace, source_scope)`` -- the registry
  key of a document across runs: the index namespace (suffix, plus any
  additional suffix), the corpus source scope and the relative path. The
  namespace keeps two tenants' identically named files apart even when both
  corpora are staged in the same directory; the source scope keeps two
  corpora written to one namespace apart when both contain the same relative
  path. Without a source scope the key is the legacy one (namespace and path
  only) that registries written before source scopes were keyed by.
* ``compute_text_hash(text)`` -- content fingerprint (change detection).
* ``compute_document_id(source_key, content_hash)`` -- the per-version
  ``Document.document_id`` that text-unit ids derive from. Hashing the full
  text, not a prefix, keeps two files with the same name and the same opening
  lines from colliding.

Absolute paths only appear when no corpus root is known (a document parsed
outside the pipeline); the pipeline stages re-derive the id against the root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from unified_kg_rag.domain.models.config import Constants

from .common import compute_hash, generate_stable_id

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import IndexingConfig

# Document.metadata key holding the path relative to the corpus root (the
# directory loader already records it under this name).
RELATIVE_PATH_KEY = "relative_path"
# Document.metadata key holding the registry namespace the document belongs to.
REGISTRY_NAMESPACE_KEY = "registry_namespace"
# Document.metadata key holding the corpus source scope the document was read
# from (``document_parsing.source_scope``, else the resolved source directory).
REGISTRY_SOURCE_KEY = "registry_source"

# A document without an index value is registered under the default index
# suffix.
DEFAULT_NAMESPACE: str = Constants.DEFAULT_SUFFIX.value

_ID_HASH_LENGTH = 32

# ``scheme://`` at the start of a source scope (``s3://bucket/prefix/``).
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_S3_URI = re.compile(r"^s3://", re.IGNORECASE)


def normalize_source_path(file_path: str | Path) -> str:
    """Return ``file_path`` as a POSIX string without a leading ``./``."""
    return PurePosixPath(str(file_path).replace("\\", "/")).as_posix()


def relative_source_path(file_path: str | Path, source_root: str | Path) -> str | None:
    """Return ``file_path`` relative to ``source_root`` (POSIX), or ``None``
    when the file is not under the root."""
    path = Path(file_path)
    root = Path(source_root)
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return None


def compute_text_hash(text: str) -> str:
    """Content hash used for change detection and document-version ids."""
    return compute_hash(text, length=_ID_HASH_LENGTH)


def compute_document_id(source_key: str, content_hash: str) -> str:
    """Per-version document id: the document's source key plus its content."""
    return generate_stable_id(f"doc:{source_key}:{content_hash}")


def store_namespace_tag(indexing: IndexingConfig) -> str | None:
    """Tag of the store set ``indexing``'s index and label prefixes name.

    The OpenSearch ``*_index_prefix`` and Neptune ``*_label_prefix`` settings
    pick the physical stores as much as the suffixes do, so two configurations
    that differ only in them must not share a registry namespace: artifact ids
    are content-derived, and removal planning would count the other store
    set's records as owners of ids it does not hold. ``None`` while every
    prefix is at its default, so a default configuration keeps the namespace
    (and registry keys) it always had; otherwise a hash of the prefixes that
    differ from their defaults (a prefix added later, left at its default,
    does not move the tag). Neptune prefixes compare capitalized, as the
    labels are named.
    """
    changed: dict[str, str] = {}
    for section in (indexing.opensearch, indexing.neptune):
        for name, field in type(section).model_fields.items():
            if not name.endswith("_prefix"):
                continue
            value, default = getattr(section, name), field.default
            if section is indexing.neptune:
                value, default = value.capitalize(), default.capitalize()
            if value != default:
                changed[name] = value
    if not changed:
        return None
    return compute_hash(json.dumps(changed, sort_keys=True), length=12)


def registry_namespace(
    suffix: str | None,
    additional_suffix: str | None = None,
    store_tag: str | None = None,
) -> str:
    """The index namespace a document's artifacts are written to.

    Mirrors how the indexers name indices/labels: the item suffix (``index``
    attribute, ``default`` when unset) plus ``indexing.additional_suffix``,
    plus ``@<store_tag>`` (:func:`store_namespace_tag`) when the index or
    label prefixes are not the defaults. An item suffix never contains ``@``,
    so a tagged namespace cannot equal an untagged one.
    """
    namespace = suffix or DEFAULT_NAMESPACE
    if additional_suffix:
        namespace = f"{namespace}-{additional_suffix}"
    return f"{namespace}@{store_tag}" if store_tag else namespace


def config_registry_namespace(indexing: IndexingConfig, suffix: str | None) -> str:
    """:func:`registry_namespace` of item ``suffix`` under ``indexing``."""
    return registry_namespace(
        suffix, indexing.additional_suffix, store_namespace_tag(indexing)
    )


def compute_doc_id(
    relative_path: str | Path,
    namespace: str = DEFAULT_NAMESPACE,
    source_scope: str | None = None,
) -> str:
    """Registry key of a document: its index namespace, corpus source scope
    and relative path.

    ``source_scope=None`` returns the legacy key (namespace and relative path
    only), which two corpora written to one namespace share for files with
    the same relative path; the pipeline always passes the run's source scope
    and only uses the legacy key to adopt records written before it did.
    """
    path = normalize_source_path(relative_path)
    key = (
        f"{namespace}\x00{path}"
        if source_scope is None
        else f"{namespace}\x00{source_scope}\x00{path}"
    )
    return compute_hash(key, length=_ID_HASH_LENGTH)


def is_local_path_scope(source_scope: str) -> bool:
    """Whether ``source_scope`` names a local directory (an absolute path).

    A scope that defaulted to the resolved source directory is one; a URI
    (``s3://...``) or a relative name (a fixed
    ``document_parsing.source_scope``) is not. Only the string is inspected,
    never the filesystem: whether a directory exists on this host says
    nothing about the corpus another host indexes from the same path.
    """
    if _URI_SCHEME.match(source_scope):
        return False
    return source_scope.startswith("/") or PureWindowsPath(source_scope).is_absolute()


def normalize_source_scope(source_scope: str) -> str:
    """``source_scope`` in the form two spellings of one corpus compare equal.

    A local path loses trailing and repeated slashes (the pipeline stores the
    resolved directory, which has neither). An ``s3://`` URI loses its
    trailing slashes: ``aws s3 sync`` (the container entrypoint) reads
    ``s3://bucket/corpus`` and ``s3://bucket/corpus/`` as the same folder.
    Any other URI or a name is kept as is.

    Records keep the spelling their run used (it is part of their key), so
    this is for comparing scopes, not for keying them.
    """
    scope = source_scope.strip()
    if _S3_URI.match(scope):
        stripped = scope.rstrip("/")
        return stripped if len(stripped) > len("s3://") else scope
    if scope.startswith("/") and not _URI_SCHEME.match(scope):
        return PurePosixPath(scope).as_posix()
    return scope
