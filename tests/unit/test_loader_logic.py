# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DirectoryLoader behavior against real temp files — no AWS, no network.

Exercises the document-loading entry point end to end: file discovery
(recursion, extension filtering, exclude patterns), the two load_single paths
(JSON passthrough vs ParserFactory parsing), MinHash deduplication, stable
content-derived doc ids, directory hashing, and graceful per-file error
handling. Everything runs against real files written under tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from unified_kg_rag.adapters.ingestion.loader import (
    DirectoryLoader,
    compute_jaccard_similarity,
    compute_minhash,
)
from unified_kg_rag.domain.models import Config

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
# Construction / validation
# --------------------------------------------------------------------------- #
def test_invalid_similarity_threshold_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        DirectoryLoader(tmp_path, similarity_threshold=0.0)
    with pytest.raises(ValueError, match="similarity_threshold"):
        DirectoryLoader(tmp_path, similarity_threshold=1.5)


def test_invalid_minhash_permutations_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="minhash_permutations"):
        DirectoryLoader(tmp_path, minhash_permutations=0)


def test_invalid_n_grams_size_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_grams_size"):
        DirectoryLoader(tmp_path, n_grams_size=0)


def test_parse_files_widens_supported_extensions(tmp_path: Path) -> None:
    # With parse_files + config, the parser's formats join the supported set.
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    assert {".txt", ".csv", ".json"} <= loader.supported_extensions
    # .pdf is a ParserFactory format, not a loader default.
    assert ".pdf" in loader.supported_extensions


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
    _write_json_doc(tmp_path / "keep.json", "id-keep", "kept content")
    (tmp_path / "skip.xyz").write_text("unsupported extension")
    loader = DirectoryLoader(tmp_path)
    discovered = {p.name for p in loader.discover_files()}
    assert discovered == {"keep.json"}


def test_discover_excludes_hidden_files(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "visible.json", "id-vis", "visible content")
    _write_json_doc(tmp_path / ".hidden.json", "id-hid", "hidden content")
    loader = DirectoryLoader(tmp_path)
    discovered = {p.name for p in loader.discover_files()}
    assert discovered == {"visible.json"}


def test_discover_recursive_includes_nested(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "top.json", "id-top", "top level content")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    _write_json_doc(nested / "deep.json", "id-deep", "deeply nested content")
    discovered = {p.name for p in DirectoryLoader(tmp_path).discover_files()}
    assert discovered == {"top.json", "deep.json"}


def test_discover_non_recursive_skips_nested(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "top.json", "id-top", "top level content")
    nested = tmp_path / "sub"
    nested.mkdir()
    _write_json_doc(nested / "deep.json", "id-deep", "nested content")
    loader = DirectoryLoader(tmp_path, recursive=False)
    discovered = {p.name for p in loader.discover_files()}
    assert discovered == {"top.json"}


def test_discover_results_sorted(tmp_path: Path) -> None:
    for name in ("c.json", "a.json", "b.json"):
        _write_json_doc(tmp_path / name, f"id-{name}", "content")
    names = [p.name for p in DirectoryLoader(tmp_path).discover_files()]
    assert names == ["a.json", "b.json", "c.json"]


# --------------------------------------------------------------------------- #
# JSON passthrough loading (parse_files=False)
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
# ParserFactory loading (parse_files=True)
# --------------------------------------------------------------------------- #
def test_parse_txt_and_csv_via_factory(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Plain text body for the parser path.")
    (tmp_path / "data.csv").write_text("name,age\nAlice,30\nBob,25\n")
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    docs = loader.load()
    by_name = {d.file_name: d for d in docs}
    assert set(by_name) == {"note.txt", "data.csv"}
    assert "Plain text body" in (by_name["note.txt"].content.text or "")
    assert "Alice" in (by_name["data.csv"].content.text or "")


def test_json_uses_passthrough_even_when_parse_files(tmp_path: Path) -> None:
    # .json is explicitly routed to from_json_file, not the ParserFactory.
    _write_json_doc(tmp_path / "doc.json", "id-json", "json passthrough body")
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    docs = loader.load()
    assert docs[0].document_id == "id-json"


def test_empty_file_skipped_in_parse_mode(tmp_path: Path) -> None:
    # FileParser raises on empty parsed text; the loader skips it gracefully.
    (tmp_path / "empty.txt").write_text("")
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    docs = loader.load()
    assert docs == []
    assert loader.failed_files == [str(tmp_path / "empty.txt")]


def test_tsv_not_discovered_since_unparseable(tmp_path: Path) -> None:
    # .tsv is no longer advertised (the ParserFactory has no .tsv loader), so a
    # .tsv source is filtered out at discovery rather than discovered-then-failed.
    (tmp_path / "data.tsv").write_text("a\tb\n1\t2\n")
    (tmp_path / "ok.txt").write_text("plain text that parses fine")
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    docs = loader.load()
    assert {d.file_name for d in docs} == {"ok.txt"}
    # never discovered, so it is not counted as a parse failure either
    assert str(tmp_path / "data.tsv") not in loader.failed_files


@pytest.mark.skipif(
    importlib.util.find_spec("unstructured") is not None,
    reason="unstructured installed; .md is actually parseable",
)
def test_markdown_skipped_without_unstructured(tmp_path: Path) -> None:
    # Without the optional `unstructured` dep, the ParserFactory does not
    # advertise .md, so it is not in supported_extensions and a markdown source
    # is filtered at discovery (not a parse failure); only the .txt survives.
    (tmp_path / "notes.md").write_text("# Heading\n\nSome markdown body.\n")
    (tmp_path / "ok.txt").write_text("plain text that parses fine")
    loader = DirectoryLoader(tmp_path, config=Config(), parse_files=True)
    docs = loader.load()
    assert {d.file_name for d in docs} == {"ok.txt"}
    assert str(tmp_path / "notes.md") not in loader.failed_files


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
    loader = DirectoryLoader(tmp_path, deduplicate=True, similarity_threshold=0.9)
    docs = loader.load()
    ids = {d.document_id for d in docs}
    # One of the two identical docs is dropped; the distinct one stays.
    assert len(docs) == 2
    assert "id-c" in ids


def test_dedup_noop_with_single_document(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "only.json", "id-only", "solitary content here")
    loader = DirectoryLoader(tmp_path, deduplicate=True)
    docs = loader.load()
    assert len(docs) == 1


def test_dedup_keeps_all_when_distinct(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "alpha topic about astronomy stars")
    _write_json_doc(tmp_path / "b.json", "id-b", "beta topic regarding cooking recipes")
    loader = DirectoryLoader(tmp_path, deduplicate=True, similarity_threshold=0.9)
    docs = loader.load()
    assert {d.document_id for d in docs} == {"id-a", "id-b"}


# --------------------------------------------------------------------------- #
# Directory hash
# --------------------------------------------------------------------------- #
def test_directory_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "first body")
    loader = DirectoryLoader(tmp_path, compute_dir_hash=True)
    loader.load()
    first = loader.directory_hash
    assert isinstance(first, str) and first

    # Re-running over the same files yields the same hash.
    loader2 = DirectoryLoader(tmp_path, compute_dir_hash=True)
    loader2.load()
    assert loader2.directory_hash == first


def test_directory_hash_changes_when_file_added(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "first body")
    loader = DirectoryLoader(tmp_path, compute_dir_hash=True)
    loader.load()
    before = loader.directory_hash

    _write_json_doc(tmp_path / "b.json", "id-b", "second body")
    loader2 = DirectoryLoader(tmp_path, compute_dir_hash=True)
    loader2.load()
    assert loader2.directory_hash != before


def test_directory_hash_not_computed_unless_requested(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "body")
    loader = DirectoryLoader(tmp_path)
    loader.load()
    assert loader.directory_hash is None


def test_compute_dir_hash_empty_file_list(tmp_path: Path) -> None:
    # Direct call with no files still returns a deterministic hash string.
    loader = DirectoryLoader(tmp_path, compute_dir_hash=True)
    h = loader._compute_dir_hash([])
    assert isinstance(h, str) and h


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


def test_compute_jaccard_similarity_returns_score() -> None:
    text = "shared shingled text for jaccard estimation across two minhashes"
    m0 = compute_minhash((0, text), num_permutations=64, n_grams=3)
    m1 = compute_minhash((1, text), num_permutations=64, n_grams=3)
    assert m0 is not None and m1 is not None
    minhashes = {0: m0[1], 1: m1[1]}
    result = compute_jaccard_similarity((0, 1), minhashes)
    assert result is not None
    i, j, sim = result
    assert (i, j) == (0, 1)
    assert sim == pytest.approx(1.0)


def test_compute_jaccard_similarity_missing_minhash_returns_none() -> None:
    text = "only one side has a minhash registered for this pair"
    m0 = compute_minhash((0, text), num_permutations=64, n_grams=3)
    assert m0 is not None
    assert compute_jaccard_similarity((0, 99), {0: m0[1]}) is None
