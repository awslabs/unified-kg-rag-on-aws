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

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from .common import compute_hash, generate_stable_id

# Document.metadata key holding the path relative to the corpus root (the
# directory loader already records it under this name).
RELATIVE_PATH_KEY = "relative_path"
# Document.metadata key holding the registry namespace the document belongs to.
REGISTRY_NAMESPACE_KEY = "registry_namespace"
# Document.metadata key holding the corpus source scope the document was read
# from (``document_parsing.source_scope``, else the resolved source directory).
REGISTRY_SOURCE_KEY = "registry_source"

DEFAULT_NAMESPACE = "default"

_ID_HASH_LENGTH = 32

# ``scheme://`` at the start of a source scope (``s3://bucket/prefix/``).
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


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
    return compute_hash(text, algorithm="sha256", length=_ID_HASH_LENGTH)


def compute_document_id(source_key: str, content_hash: str) -> str:
    """Per-version document id: the document's source key plus its content."""
    return generate_stable_id(f"doc:{source_key}:{content_hash}")


def registry_namespace(suffix: str | None, additional_suffix: str | None = None) -> str:
    """The index namespace a document's artifacts are written to.

    Mirrors how the indexers name indices/labels: the item suffix (``index``
    attribute, ``default`` when unset) plus ``indexing.additional_suffix``.
    """
    namespace = suffix or DEFAULT_NAMESPACE
    return f"{namespace}-{additional_suffix}" if additional_suffix else namespace


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
    return compute_hash(key, algorithm="sha256", length=_ID_HASH_LENGTH)


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
    """``source_scope`` as the registry stores it.

    A local path loses trailing and repeated slashes (the pipeline stores the
    resolved directory, which has neither); a URI or a name is kept as is,
    since a trailing slash can be part of an ``s3://`` prefix.
    """
    scope = source_scope.strip()
    if scope.startswith("/") and not _URI_SCHEME.match(scope):
        return PurePosixPath(scope).as_posix()
    return scope
