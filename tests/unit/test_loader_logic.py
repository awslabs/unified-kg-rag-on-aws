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
from unified_kg_rag.shared.exceptions import ConfigurationError

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
                "page_content": text,
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
    # A ConfigurationError, which the CLIs report without a traceback.
    with pytest.raises(ConfigurationError, match="not a valid directory"):
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
    docs = loader.deduplicate_documents(loader.load(), threshold=0.9)
    ids = {d.document_id for d in docs}
    # One of the two identical docs is dropped; the distinct one stays.
    assert len(docs) == 2
    assert "id-c" in ids


def test_dedup_noop_with_single_document(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "only.json", "id-only", "solitary content here")
    loader = DirectoryLoader(tmp_path)
    docs = loader.deduplicate_documents(loader.load(), threshold=0.9)
    assert len(docs) == 1


def test_dedup_keeps_all_when_distinct(tmp_path: Path) -> None:
    _write_json_doc(tmp_path / "a.json", "id-a", "alpha topic about astronomy stars")
    _write_json_doc(tmp_path / "b.json", "id-b", "beta topic regarding cooking recipes")
    loader = DirectoryLoader(tmp_path)
    docs = loader.deduplicate_documents(loader.load(), threshold=0.9)
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

    kept = loader.deduplicate_documents(docs, threshold=0.9)
    assert [d.document_id for d in kept] == ["id-copy"]
    kept = loader.deduplicate_documents(
        docs, threshold=0.9, keep=lambda d: d.document_id == "id-original"
    )
    assert [d.document_id for d in kept] == ["id-original"]
    # An avoided document (its indexing failed) loses to its copy.
    kept = loader.deduplicate_documents(
        docs, threshold=0.9, avoid=lambda d: d.document_id == "id-copy"
    )
    assert [d.document_id for d in kept] == ["id-original"]


# --------------------------------------------------------------------------- #
# Near-duplicates: template documents are not copies
# --------------------------------------------------------------------------- #
def _clause(i: int) -> str:
    n, word = (i + 1) * 104729, ""
    while n:
        n, letter = divmod(n, 26)
        word += chr(97 + letter)
    return f"Clause {word}: the Vendor and the Buyer keep the {word} terms."


# A long form whose numbers are a small part of its text.
_TEMPLATE = (
    "SUPPLY AGREEMENT {number}, made on {date}. The Buyer shall pay {amount} "
    "USD within forty-five days of delivery. " + " ".join(_clause(i) for i in range(40))
)


def _template(document_id: str, **fields: str) -> Document:
    return Document(
        page_content=_TEMPLATE.format(**fields),
        document_id=document_id,
        file_name=f"{document_id}.txt",
        file_path=f"/corpus/{document_id}.txt",
    )


def test_a_template_filled_in_twice_keeps_both_documents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _template("a", number="1001", date="2020-01-15", amount="12,500")
    second = _template("b", number="1002", date="2021-03-02", amount="48,000")
    copy = _template("c", number="1001", date="2020-01-15", amount="12,500")
    loader = DirectoryLoader("/")
    loader.max_workers = 1
    with caplog.at_level("WARNING"):
        kept = loader.deduplicate_documents([first, second, copy], threshold=0.9)
    # The two contracts differ in a few numbers of a long shared text, which
    # read as copies at 0.9; the copy of the first holds its numbers.
    assert [d.document_id for d in kept] == ["a", "b"]
    assert "'c.txt' (of 'a.txt', similarity 1.00)" in caplog.text


def test_numbers_match_in_any_digit_script_in_their_order() -> None:
    from unified_kg_rag.adapters.ingestion.loader import numbers_digest

    # Full-width and Arabic-Indic digits read as the same numbers.
    assert numbers_digest("계약 금액 １２,５００원, 2020년") == numbers_digest(
        "계약 금액 12,500원, 2020년"
    )
    assert numbers_digest("المبلغ ٢٥٠٠") == numbers_digest("المبلغ 2500")
    assert numbers_digest("금액 12,500원") != numbers_digest("금액 12,600원")
    # Swapped dates or line amounts are another document.
    assert numbers_digest("from 03/04/2026") != numbers_digest("from 04/03/2026")
    # Han numerals count too.
    assert numbers_digest("合同金额三万元") != numbers_digest("合同金额四万元")


def test_the_dedup_threshold_is_the_configured_one() -> None:
    def word(i: int) -> str:
        return "".join(chr(97 + (i * k + k * k) % 26) for k in range(1, 7))

    base = " ".join(
        f"The {word(i)} crate leaves the {word(i + 40)} depot." for i in range(30)
    )
    # Two words replaced: a 3-gram similarity of 0.91.
    edited = base.replace(word(3), "zzzzzz").replace(word(7), "yyyyyy")
    documents = [
        Document(
            page_content=base, document_id="a", file_name="a.txt", file_path="a.txt"
        ),
        Document(
            page_content=edited, document_id="b", file_name="b.txt", file_path="b.txt"
        ),
    ]
    loader = DirectoryLoader("/")
    loader.max_workers = 1
    assert len(loader.deduplicate_documents(documents, threshold=0.85)) == 1
    assert len(loader.deduplicate_documents(documents, threshold=0.97)) == 2
