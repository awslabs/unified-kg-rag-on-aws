# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source digests: a file's bytes and the parser that reads them.

Equal digests must mean equal parsed text, so the digest changes with the
bytes and with anything the parse depends on (the loader, its arguments,
this package's and the parsing libraries' versions), and an unparsed
document compares by its record's content hash. Synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.registries import isolate_loaders
from unified_kg_rag.adapters.ingestion import parser as parser_module
from unified_kg_rag.adapters.ingestion.file_loaders import TextFileLoader
from unified_kg_rag.application.ingestion.source_parsing import SourceFileParser
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    document_doc_id,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    SOURCE_DIGEST_KEY,
    compute_source_digest,
    compute_text_hash,
    is_unparsed,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


def test_the_digest_is_of_the_bytes_however_they_are_read() -> None:
    raw = "가나다 Vendor supplies widgets.".encode()
    whole = compute_source_digest([raw], "parser-a")
    assert compute_source_digest([raw[:5], raw[5:9], raw[9:]], "parser-a") == whole
    assert compute_source_digest([raw + b" "], "parser-a") != whole
    assert compute_source_digest([raw], "parser-b") != whole


def test_the_parser_fingerprint_follows_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = parser_module.ParserFactory
    text = factory.parser_fingerprint(".txt")
    assert factory.parser_fingerprint(".TXT") == text
    assert factory.parser_fingerprint(".pdf") != text

    class OtherLoader(TextFileLoader):
        pass

    isolate_loaders(monkeypatch)
    factory.register_loader(".txt", OtherLoader, {}, "Text")
    assert factory.parser_fingerprint(".txt") != text
    factory.register_loader(".txt", TextFileLoader, {"encoding": "utf-8"}, "Text")
    assert factory.parser_fingerprint(".txt") != text


def test_the_parser_fingerprint_follows_the_parsing_library_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = parser_module.ParserFactory.parser_fingerprint(".txt")
    parser_module._fingerprint.cache_clear()
    monkeypatch.setattr(
        parser_module, "_PARSING_DISTRIBUTIONS", ("pypdf", "no-such-distribution")
    )
    try:
        assert parser_module.ParserFactory.parser_fingerprint(".txt") != text
    finally:
        parser_module._fingerprint.cache_clear()


def test_an_unparsed_document_has_its_records_identity_and_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus"
    (source / "sub").mkdir(parents=True)
    path = source / "sub" / "contract.txt"
    path.write_text("Vendor supplies widgets to Buyer.", encoding="utf-8")
    parser = SourceFileParser(Config(), source)
    digest = parser.source_digest(path)
    parsed, failed = parser.parse([path], {path: digest})
    assert failed == []
    document = parsed[path]
    assert document.metadata[SOURCE_DIGEST_KEY] == digest
    assert not is_unparsed(document)

    recorded = compute_text_hash(document.page_content)
    stand_in = parser.unparsed(path, digest, recorded)
    assert is_unparsed(stand_in) and stand_in.page_content == ""
    assert compute_content_hash(stand_in) == compute_content_hash(document)
    assert document_doc_id(stand_in) == document_doc_id(document)
    assert stand_in.document_id == document.document_id

    read, hashes = parser.read_unparsed([stand_in])
    assert [d.page_content for d in read] == [document.page_content]
    assert hashes == {document_doc_id(document): recorded}


def test_a_relative_path_resolves_only_when_it_must(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolving walks the filesystem (a symlinked root: twice per document);
    # a path under the root as given needs none.
    from unified_kg_rag.domain.ingestion.document_identity import (
        relative_source_path,
    )

    real = tmp_path / "corpus"
    (real / "sub").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    resolved: list[Path] = []
    resolve = Path.resolve

    def counted(self: Path, strict: bool = False) -> Path:
        resolved.append(self)
        return resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", counted)
    assert relative_source_path(linked / "sub" / "a.txt", linked) == "sub/a.txt"
    assert resolved == []
    assert relative_source_path(real / "sub" / "a.txt", linked) == "sub/a.txt"
    assert resolved
    assert relative_source_path(tmp_path / "other.txt", linked) is None
