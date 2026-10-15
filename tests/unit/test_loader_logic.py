# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DirectoryLoader behavior against real temp files — no AWS, no network.

Exercises the pre-parsed ``Document`` JSON loader: file discovery
(recursion, extension filtering, hidden-path exclusion), JSON loading, MinHash
deduplication, stable doc ids and graceful per-file error handling.
Everything runs against real files written under tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.adapters.ingestion.loader import (
    DirectoryLoader,
    compute_minhash,
)
from unified_kg_rag.domain.models import Document

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_json_doc(path: Path, document_id: str, text: str) -> None:
    """Write a minimal Document JSON that Document.from_json_file can load."""
    path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "file_name": path.name,
                "file_path": str(path),
                "file_type": "json",
                "total_pages": 1,
                "content": {"text": text},
            }
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Directory validation
# --------------------------------------------------------------------------- #
def test_missing_directory_raises(tmp_path: Path) -> None:
    loader = DirectoryLoader(tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_path_that_is_a_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "a_file.txt"
    f.write_text("not a directory")
    loader = DirectoryLoader(f)
    with pytest.raises(ValueError, match="not a valid directory"):
        loader.load()


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def test_empty_directory_returns_empty(tmp_path: Path) -> None:
    assert DirectoryLoader(tmp_path).load() == []


def test_discover_filters_by_extension(tmp_path: Path) -> None:
    # Only pre-parsed Document JSON: a raw .txt would fail from_json_file.
    _write_json_doc(tmp_path / "keep.json", "id-keep", "kept content")
    (tmp_path / "skip.xyz").write_text("unsupported extension")
    (tmp_path / "raw.txt").write_text("raw text is the parsing stage's input")
    loader = DirectoryLoader(tmp_path)
    discovered = {p.name for p in loader.discover_files()}
    assert discovered == {"keep.json"}


def test_discover_excludes_hidden_files(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "visible.json", "id-vis", "visible content")
    _write_json_doc(tmp_path / ".hidden.json", "id-hid", "hidden content")
    loader = DirectoryLoader(tmp_path)
    discovered = {p.name for p in loader.discover_files()}
    assert discovered == {"visible.json"}


def test_discover_under_hidden_ancestor_finds_files(tmp_path: Path) -> None:
    # Regression: exclude patterns were matched against the absolute path, so
    # "**/.*" fired on a hidden ancestor and the whole corpus was skipped.
    root = tmp_path / ".hidden_parent" / "corpus"
    root.mkdir(parents=True)
    _write_json_doc(root / "doc.json", "id-doc", "corpus content")
    _write_json_doc(root / ".secret.json", "id-sec", "hidden content")
    discovered = {p.name for p in DirectoryLoader(root).discover_files()}
    assert discovered == {"doc.json"}


def test_discover_excludes_hidden_and_pycache_dirs(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "keep.json", "id-keep", "kept content")
    for sub in (".git", "__pycache__"):
        (tmp_path / sub).mkdir()
        _write_json_doc(tmp_path / sub / "x.json", f"id-{sub}", "excluded content")
    loader = DirectoryLoader(tmp_path)
    assert {p.relative_to(tmp_path).as_posix() for p in loader.discover_files()} == {
        "keep.json"
    }


def test_old_positional_arguments_fail_at_construction(tmp_path: Path) -> None:
    # DirectoryLoader(path, config) used to bind the config to a removed
    # second parameter and only failed later, inside discovery.
    with pytest.raises(TypeError):
        DirectoryLoader(tmp_path, object())  # type: ignore[call-arg]


def test_discover_recursive_includes_nested(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "top.json", "id-top", "top level content")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    _write_json_doc(nested / "deep.json", "id-deep", "deeply nested content")
    discovered = {p.name for p in DirectoryLoader(tmp_path).discover_files()}
    assert discovered == {"top.json", "deep.json"}


def test_discover_results_sorted(tmp_path: Path) -> None:
    for name in ("c.json", "a.json", "b.json"):
        _write_json_doc(tmp_path / name, f"id-{name}", "content")
    names = [p.name for p in DirectoryLoader(tmp_path).discover_files()]
    assert names == ["a.json", "b.json", "c.json"]


# --------------------------------------------------------------------------- #
# JSON loading
# --------------------------------------------------------------------------- #
def test_load_json_documents_enriches_metadata(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "doc.json", "id-doc", "hello content")
    docs = DirectoryLoader(tmp_path).load()
    assert len(docs) == 1
    doc = docs[0]
    assert doc.document_id == "id-doc"
    meta = doc.metadata
    assert meta["relative_path"] == "doc.json"
    assert meta["file_extension"] == ".json"
    assert meta["source_directory"] == str(tmp_path.resolve())
    assert meta["file_size"] > 0


def test_load_preserves_doc_id_from_json(tmp_path: Path) -> None:
    # JSON passthrough keeps the id embedded in the file (stable across runs).
    _write_json_doc(tmp_path / "stable.json", "stable-id-123", "some text")
    docs = DirectoryLoader(tmp_path).load()
    assert docs[0].document_id == "stable-id-123"


def test_relative_path_reflects_nested_structure(tmp_path: Path) -> None:
    nested = tmp_path / "x" / "y"
    nested.mkdir(parents=True)
    _write_json_doc(nested / "deep.json", "id-deep", "nested body")
    docs = DirectoryLoader(tmp_path).load()
    assert docs[0].metadata["relative_path"] == str(Path("x") / "y" / "deep.json")


def test_malformed_json_is_skipped_not_fatal(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "good.json", "id-good", "valid content")
    (tmp_path / "bad.json").write_text("{ this is not valid json ")
    loader = DirectoryLoader(tmp_path)
    docs = loader.load()
    assert {d.document_id for d in docs} == {"id-good"}
    assert loader.failed_files == [str(tmp_path / "bad.json")]


# --------------------------------------------------------------------------- #
# Deduplication (MinHash + Jaccard, real ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def test_duplicate_documents_collapse(tmp_path: Path) -> None:
    same = "the quick brown fox jumps over the lazy dog repeatedly today"
    _write_json_doc(tmp_path / "a.json", "id-a", same)
    _write_json_doc(tmp_path / "b.json", "id-b", same)
    _write_json_doc(
        tmp_path / "c.json", "id-c", "an entirely unrelated paragraph about ships"
    )
    loader = DirectoryLoader(tmp_path)
    docs = loader.deduplicate_documents(loader.load())
    ids = {d.document_id for d in docs}
    # One of the two identical docs is dropped; the distinct one stays.
    assert len(docs) == 2
    assert "id-c" in ids


def test_dedup_noop_with_single_document(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "only.json", "id-only", "solitary content here")
    loader = DirectoryLoader(tmp_path)
    docs = loader.deduplicate_documents(loader.load())
    assert len(docs) == 1


def test_dedup_keeps_all_when_distinct(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "alpha topic about astronomy stars")
    _write_json_doc(tmp_path / "b.json", "id-b", "beta topic regarding cooking recipes")
    loader = DirectoryLoader(tmp_path)
    docs = loader.deduplicate_documents(loader.load())
    assert {d.document_id for d in docs} == {"id-a", "id-b"}


# --------------------------------------------------------------------------- #
# Module-level MinHash helpers
# --------------------------------------------------------------------------- #
def test_compute_minhash_returns_none_for_short_content() -> None:
    # Content shorter than the n-gram size cannot be shingled.
    assert compute_minhash((0, "ab"), num_permutations=64, n_grams=3) is None


def test_compute_minhash_returns_none_for_empty_content() -> None:
    assert compute_minhash((0, ""), num_permutations=64, n_grams=3) is None


def test_compute_minhash_identical_content_gives_full_similarity() -> None:
    text = "deterministic shingled content for similarity comparison"
    r1 = compute_minhash((0, text), num_permutations=64, n_grams=3)
    r2 = compute_minhash((1, text), num_permutations=64, n_grams=3)
    assert r1 is not None and r2 is not None
    assert r1[1].jaccard(r2[1]) == pytest.approx(1.0)


def test_dedup_keeps_the_preferred_copy(tmp_path: Path) -> None:
    # An incremental run prefers the already-indexed document: a new copy that
    # sorts first must not replace (and so delete) the indexed original.
    same = "the quick brown fox jumps over the lazy dog repeatedly today"
    _write_json_doc(tmp_path / "a_copy.json", "id-copy", same)
    _write_json_doc(tmp_path / "b_original.json", "id-original", same)
    loader = DirectoryLoader(tmp_path)
    docs = loader.load()
    assert [d.document_id for d in docs] == ["id-copy", "id-original"]

    kept = loader.deduplicate_documents(docs)
    assert [d.document_id for d in kept] == ["id-copy"]
    kept = loader.deduplicate_documents(
        docs, keep=lambda d: d.document_id == "id-original"
    )
    assert [d.document_id for d in kept] == ["id-original"]
    # An avoided document (its indexing failed) loses to its copy.
    kept = loader.deduplicate_documents(
        docs, avoid=lambda d: d.document_id == "id-copy"
    )
    assert [d.document_id for d in kept] == ["id-original"]


# --------------------------------------------------------------------------- #
# Document JSON compatibility
# --------------------------------------------------------------------------- #
def test_document_json_from_langchain_based_versions_still_loads(
    tmp_path: Path,
) -> None:
    # Earlier versions subclassed LangChain's Document, so their JSON also
    # carries its "id" and "type" keys; the pure model must ignore them.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "id": None,
                "type": "Document",
                "page_content": "stale",
                "metadata": {"source": "legacy.txt"},
                "document_id": "doc-legacy",
                "file_name": "legacy.txt",
                "file_path": "legacy.txt",
                "file_type": "txt",
                "total_pages": 1,
                "content": {"text": "Vendor ships goods to Buyer."},
            }
        ),
        encoding="utf-8",
    )

    doc = Document.from_json_file(legacy)

    assert doc.page_content == "Vendor ships goods to Buyer."
    assert doc.metadata == {"source": "legacy.txt"}
    out = tmp_path / "out.json"
    doc.to_json_file(out)
    assert Document.from_json_file(out) == doc
