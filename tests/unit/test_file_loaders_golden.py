# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The direct file loaders parse exactly what the langchain-community ones did.

The change hash and ``document_id`` of a source document derive from its
parsed ``page_content``, so a loader that read one byte differently (a CRLF
kept, a BOM, another decoder) would re-extract every such document of an
indexed corpus. ``_GOLDEN`` holds the text hash and corpus-relative
``document_id`` the langchain-community loaders produced for the synthetic
corpus of ``tests.fixtures.loader_corpus``; while langchain-community is
installed (ragas still pulls it in) the old loaders are also run side by side.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.loader_corpus import write_loader_corpus
from unified_kg_rag.adapters.ingestion.file_loaders import (
    CsvFileLoader,
    PdfFileLoader,
    TextFileLoader,
    UnstructuredPartitionLoader,
)
from unified_kg_rag.adapters.ingestion.parser import (
    _UNSTRUCTURED_AVAILABLE,
    FileParser,
    ParserFactory,
)
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    DocumentParsingStage,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    compute_document_id,
    compute_text_hash,
)
from unified_kg_rag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
)

pytestmark = pytest.mark.unit

_COMMUNITY = importlib.util.find_spec("langchain_community") is not None

# name -> (text hash, document_id relative to the corpus root), as the
# langchain-community TextLoader/CSVLoader/PyPDFLoader parsed the corpus.
# latin1.txt depends on what charset-normalizer detects for it (a Baltic
# code page); that is the parser's behaviour, kept as it was.
_GOLDEN: dict[str, tuple[str, str]] = {
    "bare_cr.txt": (
        "48022b44dbc640a5239e29701d5c2644",
        "159af2a5-1234-5893-b83c-d739b47b9461",
    ),
    "bom.csv": (
        "7c5d3bf901b5ccbf5132206d8975dbc9",
        "d3944451-e41f-570a-a94e-c7e34e17cf27",
    ),
    "bom.txt": (
        "24c635ca5da6dfc77acba7c4c62e3226",
        "4bc39b4f-f1ba-5546-884e-9895074b2b9f",
    ),
    "cp949.csv": (
        "2ceb7286dc259b38b67bdc5853de6856",
        "80c0d5e8-df98-5d3e-9505-95fe563743d0",
    ),
    "cp949.txt": (
        "d26861f61f8911156ad9f31d08566b08",
        "9c1aed17-5bcb-5c3f-bd00-fd67db21788d",
    ),
    "crlf.txt": (
        "81fad54cd5fc63d0bb101593de58242e",
        "2e493070-2aba-5b86-9557-a87efb702398",
    ),
    "latin1.txt": (
        "7b121ce6ebbf86797d612b54fddd4af4",
        "fb827430-b34b-5035-a094-1099d9d43e55",
    ),
    "mixed_newlines.txt": (
        "bc0775799d8b1d47111cbfee16a29589",
        "8996ffbb-80f3-5975-86e5-bf4e984a1d03",
    ),
    "multi.pdf": (
        "9e2e6692546bb2a51acc40e07e7de476",
        "c4fd3e19-1f88-567d-8a43-91bbd17f2b90",
    ),
    "quotes.csv": (
        "b9128a0a92623b3e1946e0b28d50980f",
        "f040e8f2-b345-55c1-b4e2-3809bbf4111e",
    ),
    "trailing_ws.txt": (
        "2cd9abaab9a4766e22c806f8cecb1b99",
        "a68d0838-a8c7-5c8e-bc86-29515128d5ee",
    ),
    "utf16.txt": (
        "8befaa97aeb650ddd628a13b603e705f",
        "5e939a18-df90-5e0b-a7cd-a9daaf39aae7",
    ),
}

_PDF_METADATA = {
    "producer": "fixture",
    "creator": "loader_corpus",
    "creationdate": "2024-01-02T03:04:05+09:00",
    "title": "Synthetic supply terms",
    "total_pages": 3,
    "page": 0,
    "page_label": "1",
}


def _parse(path: Path):
    return ParserFactory.create_parser(path, Config()).parse_file(path)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    write_loader_corpus(root, markup=_UNSTRUCTURED_AVAILABLE)
    return root


def test_golden_text_hashes_and_document_ids(corpus: Path) -> None:
    for name, (text_hash, document_id) in _GOLDEN.items():
        document = _parse(corpus / name)
        got = compute_text_hash(document.page_content)
        assert (got, compute_document_id(name, got)) == (
            text_hash,
            document_id,
        ), name


def test_line_endings_and_boms_read_as_before(corpus: Path) -> None:
    assert _parse(corpus / "crlf.txt").page_content == (
        "Vendor ships parts.\nBuyer pays.\n\nSecond paragraph.\n"
    )
    assert _parse(corpus / "bare_cr.txt").page_content == (
        "Old line one.\nOld line two.\n"
    )
    assert _parse(corpus / "bom.txt").page_content.startswith("Vendor with")
    # A quoted newline inside a CSV field is kept as written.
    assert (
        "Pays on\r\ndelivery\nnext line" in _parse(corpus / "quotes.csv").page_content
    )


def test_pdf_pages_and_metadata(corpus: Path) -> None:
    document = _parse(corpus / "multi.pdf")
    assert document.page_content.split("\n\n") == [
        "Vendor agrees to ship 100 units.\nDelivery within 30 days.",
        "Buyer pays on receipt.\nLate payment accrues interest.",
        "Either party may terminate.\nNotice period: 60 days.",
    ]
    metadata = {k: v for k, v in document.metadata.items() if k != "source"}
    assert metadata == _PDF_METADATA
    pages = PdfFileLoader(corpus / "multi.pdf").load()
    assert [page.metadata["page"] for page in pages] == [0, 1, 2]


def test_csv_rows_follow_dictreader_shapes(corpus: Path) -> None:
    rows = CsvFileLoader(corpus / "quotes.csv").load()
    assert [row.metadata["row"] for row in rows] == [0, 1, 2, 3, 4]
    assert rows[2].page_content == "name: Short\ndescription: row\namount: None"
    assert rows[3].page_content.endswith("\nNone: extra,fields")


def test_registered_loaders_are_the_direct_ones() -> None:
    configs = ParserFactory._loader_configs
    assert configs[".txt"][0] is TextFileLoader
    assert configs[".csv"][0] is CsvFileLoader
    assert configs[".pdf"][0] is PdfFileLoader
    if _UNSTRUCTURED_AVAILABLE:
        assert configs[".md"][:2] == (UnstructuredPartitionLoader, {"partition": "md"})
        assert configs[".htm"][:2] == (
            UnstructuredPartitionLoader,
            {"partition": "html"},
        )


def test_parser_module_does_not_import_langchain_community() -> None:
    import subprocess
    import sys

    code = (
        "import sys, unified_kg_rag.adapters.ingestion.parser; "
        "sys.exit('langchain_community' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_unknown_partition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partition"):
        UnstructuredPartitionLoader(tmp_path / "x.md", partition="docx")  # type: ignore[arg-type]


# --- side by side with the langchain-community loaders ------------------------


def _community_loaders() -> dict[str, tuple[type, dict]]:
    from langchain_community.document_loaders import (
        CSVLoader,
        PyPDFLoader,
        TextLoader,
    )

    loaders: dict[str, tuple[type, dict]] = {
        ".csv": (CSVLoader, {}),
        ".pdf": (PyPDFLoader, {}),
        ".txt": (TextLoader, {}),
    }
    if _UNSTRUCTURED_AVAILABLE:
        from langchain_community.document_loaders import (
            UnstructuredHTMLLoader,
            UnstructuredMarkdownLoader,
        )

        loaders[".md"] = (UnstructuredMarkdownLoader, {})
        loaders[".html"] = (UnstructuredHTMLLoader, {})
    return loaders


@pytest.mark.skipif(not _COMMUNITY, reason="langchain-community is not installed")
def test_same_documents_as_the_community_loaders(corpus: Path, monkeypatch) -> None:
    from langchain_community.document_loaders import CSVLoader, TextLoader

    # The old loaders took the same decode-error retry.
    monkeypatch.setattr(
        "unified_kg_rag.adapters.ingestion.parser._ENCODING_AWARE_LOADERS",
        (CsvFileLoader, TextFileLoader, CSVLoader, TextLoader),
    )
    compared = 0
    for path in sorted(corpus.iterdir()):
        loader_class, kwargs = _community_loaders()[path.suffix]
        old = FileParser(Config(), loader_class, kwargs).parse_file(path)
        new = _parse(path)
        assert new.page_content == old.page_content, path.name
        assert new.document_id == old.document_id, path.name
        assert new.metadata == old.metadata, path.name
        compared += 1
    assert compared == len(_GOLDEN) + (2 if _UNSTRUCTURED_AVAILABLE else 0)


# --- an incremental rerun after the switch -------------------------------------


def _config() -> Config:
    config = Config()
    config.providers.doc_status.enabled = True
    config.processing.deduplicate = False
    return config


def _run(root: Path, store: FakeDocStatusStore, mocker) -> PipelineContext:
    """Parse ``root``, diff it against ``store`` and record what it processed."""
    mocker.patch("boto3.Session")
    config = _config()
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    DocumentParsingStage(config, source_directory=root)._execute_core(context)
    loading = DocumentLoadingStage(config, source_directory=root, doc_status=store)
    kept, _ = loading._apply_incremental_filter(list(context.documents), context)
    lineages = build_document_lineage(documents=kept, text_units=[])
    IncrementalIndexer(
        store, mocker.MagicMock(), scope=context.incremental_scope
    )._record_processed(lineages, context.incremental_fingerprints)
    return context


@pytest.mark.parametrize("indexed_by", ["golden", "community"])
def test_rerun_after_switch_reports_no_changed_document(
    corpus: Path, mocker, monkeypatch, indexed_by: str
) -> None:
    store = FakeDocStatusStore()
    if indexed_by == "community":
        if not _COMMUNITY:
            pytest.skip("langchain-community is not installed")
        from langchain_community.document_loaders import CSVLoader, TextLoader

        with monkeypatch.context() as patch:
            patch.setattr(
                "unified_kg_rag.adapters.ingestion.parser._ENCODING_AWARE_LOADERS",
                (CSVLoader, TextLoader),
            )
            configs = dict(ParserFactory._loader_configs)
            for ext, (loader_class, kwargs) in _community_loaders().items():
                configs[ext] = (loader_class, kwargs, ext)
            patch.setattr(ParserFactory, "_loader_configs", configs)
            first = _run(corpus, store, mocker)
    else:
        # The corpus as the old loaders recorded it: their content hashes.
        first = _run(corpus, store, mocker)
        for record in store.list_all():
            name = Path(record.file_path or "").name
            if name in _GOLDEN:
                store.put(record.model_copy(update={"content_hash": _GOLDEN[name][0]}))
    assert first.incremental_delta is not None
    indexed = len(first.incremental_delta.new)
    assert indexed == len(list(corpus.iterdir()))

    rerun = _run(corpus, store, mocker).incremental_delta
    assert rerun is not None
    assert (rerun.new, rerun.changed, rerun.deleted) == ([], [], [])
    assert len(rerun.unchanged) == indexed
