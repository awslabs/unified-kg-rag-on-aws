# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document citations end to end on the embedded stores (docs/design.md §7.1).

An English, Korean or Japanese corpus with a two-page PDF is ingested with the
local preset (SQLite stores, a fake Ollama server; no AWS) and every search
strategy answers with ``[n]`` markers: each numbered section of the answer
context is a citation, the markers the answer uses are flagged, one it makes
up is never a citation, graph sections name the documents of their lineage
read from the store, and the PDF's passages name their relative path and
pages. A streamed answer carries the same markers.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Methodology,
    PipelineStageStatus,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.domain.retrieval.citations import CitationMarkerScanner

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

# (entity names, a document, the query, the answer as a model writes it).
_CORPORA = {
    LanguageCode.EN: (
        (),
        "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor.",
        "Who stores the widgets for Vendor?",
        "Depot stores the widgets for Vendor.",
    ),
    LanguageCode.KO: (
        ("공급사", "구매사", "창고"),
        "공급사는 구매사에 부품을 공급한다. 창고는 공급사의 부품을 보관한다.",
        "창고는 누구의 부품을 보관하나요?",
        "창고는 공급사의 부품을 보관합니다.",
    ),
    LanguageCode.JA: (
        ("供給者", "購入者", "倉庫"),
        "供給者は購入者に部品を供給する。倉庫は供給者の部品を保管する。",
        "倉庫は誰の部品を保管していますか？",
        "倉庫は供給者の部品を保管しています。",
    ),
}
# The PDF's pages (pypdf reads the text of a page's content stream).
_PDF_PAGES = (
    "Depot stores widgets for Vendor at the harbour.",
    "Carrier moves gadgets for Depot every week.",
)
# A number no context shows.
_MADE_UP = 99


def _pdf(pages: tuple[str, ...]) -> bytes:
    """A minimal PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    font = 3 + 2 * len(pages)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> "
                f"/Contents {4 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


class _Model:
    """The fake model: ingestion as in the stage tests; an answer that cites
    the first and the last marker of its context, and one it makes up."""

    def __init__(self, names: tuple[str, ...], answer: str) -> None:
        self.names = (*names, *e2e.NAMES)
        self.answer = answer
        self.contexts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), self.names)
        if "CITATIONS:" in system:
            with self._lock:
                self.contexts.append(human)
            markers = [int(n) for n in re.findall(r"^\[(\d+)\] ", human, re.M)]
            assert markers, "a numbered context"
            return f"{self.answer} [{markers[0]}][{markers[-1]}] [{_MADE_UP}]"
        return e2e._respond(system, human)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language", list(_CORPORA), ids=lambda c: c.value)
def test_every_strategy_cites_the_documents_its_answer_rests_on(
    tmp_path: Path, language: LanguageCode
) -> None:
    names, text, question, answer = _CORPORA[language]
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.translation.source_language = language
    config.indexing.methodologies = [*config.indexing.methodologies, Methodology.RAPTOR]
    config.search.global_search.enable_map_reduce = True
    config.search.global_search.map_reduce_min_results = 1
    model = _Model(names, answer)
    server = e2e.FakeOllama(model, answer, e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("reranker"))
    source = tmp_path / "corpus"
    (source / "docs").mkdir(parents=True)
    (source / "contract.txt").write_text(text, encoding="utf-8")
    (source / "docs" / "manual.pdf").write_bytes(_pdf(_PDF_PAGES))

    context = e2e._ingest(config, providers, source, tmp_path / "run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    # The PDF's text units carry their pages and relative path.
    units = e2e._stored(
        config, providers, config.indexing.opensearch.text_units_index_prefix
    )
    pdf_units = [
        u["_source"]
        for u in units
        if u["_source"].get("attributes", {}).get("file_name") == "manual.pdf"
    ]
    assert pdf_units
    assert {u["attributes"]["relative_path"] for u in pdf_units} == {"docs/manual.pdf"}
    assert {u["attributes"]["page_start"] for u in pdf_units} == {1, 2}
    stored_ids = {u["_id"] for u in units}

    files = {"contract.txt", "manual.pdf"}
    resolved_graph_citation = False
    chain = GraphRAGChain(config, providers=providers)
    try:
        for strategy in SearchStrategy:
            out = chain.invoke(RAGInput(query=question, search_strategy=strategy))
            assert isinstance(out, RAGOutput), strategy
            markers = [c.marker for c in out.citations]
            assert markers == list(range(1, len(markers) + 1)), strategy
            # The answer cites the context's first and last markers.
            assert [c.marker for c in out.citations if c.cited] == sorted(
                {1, len(markers)}
            ), strategy
            assert out.metadata["unknown_citation_markers"] == [_MADE_UP], strategy
            for citation in out.citations:
                named = {d.file_name for d in citation.documents}
                assert named <= files, (strategy, citation)
                if citation.section_type == "text":
                    assert citation.chunk_id in stored_ids, strategy
                    assert citation.file_name in files, strategy
                    if citation.file_name == "manual.pdf":
                        assert citation.file_path == "docs/manual.pdf"
                        assert citation.page in (1, 2)
                elif citation.documents:
                    resolved_graph_citation = True
                    assert set(citation.documents[0].text_unit_ids) <= stored_ids
            # Every citation is a section the model was shown.
            context_text = model.contexts[-1]
            for citation in out.citations:
                assert re.search(rf"^\[{citation.marker}\] ", context_text, re.M)
        assert resolved_graph_citation

        # A streamed answer carries the same markers.
        chunks = list(
            chain.stream(RAGInput(query=question, search_strategy=SearchStrategy.LOCAL))
        )
        scanner = CitationMarkerScanner()
        for chunk in chunks:
            scanner.feed(chunk)
        scanner.finish()
        assert answer in "".join(chunks)
        assert scanner.markers[-1] == _MADE_UP
        assert scanner.markers[0] == 1
    finally:
        chain.close()
