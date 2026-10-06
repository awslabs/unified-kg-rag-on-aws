# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One identity rule for source documents.

A document is identified by its path *relative to the corpus root* (so the same
corpus synced to ``/tmp/graphrag-source`` on one machine and checked out under
``~/corpus`` on another yields the same ids), and a document *version* by that
path plus a hash of its full text:

* ``compute_doc_id(relative_path, namespace)`` -- the registry key of a
  document across runs: the index namespace (suffix, plus any additional
  suffix) and the relative path. The namespace keeps two tenants' identically
  named files apart even when both corpora are staged in the same directory.
* ``compute_text_hash(text)`` -- content fingerprint (change detection).
* ``compute_document_id(source_key, content_hash)`` -- the per-version
  ``Document.document_id`` that text-unit ids derive from. Hashing the full
  text, not a prefix, keeps two files with the same name and the same opening
  lines from colliding.

Absolute paths only appear when no corpus root is known (a document parsed
outside the pipeline); the pipeline stages re-derive the id against the root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .common import compute_hash, generate_stable_id

# Document.metadata key holding the path relative to the corpus root (the
# directory loader already records it under this name).
RELATIVE_PATH_KEY = "relative_path"
# Document.metadata key holding the registry namespace the document belongs to.
REGISTRY_NAMESPACE_KEY = "registry_namespace"

DEFAULT_NAMESPACE = "default"

_ID_HASH_LENGTH = 32


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
    relative_path: str | Path, namespace: str = DEFAULT_NAMESPACE
) -> str:
    """Registry key of a document: its index namespace and relative path."""
    key = f"{namespace}\x00{normalize_source_path(relative_path)}"
    return compute_hash(key, algorithm="sha256", length=_ID_HASH_LENGTH)
