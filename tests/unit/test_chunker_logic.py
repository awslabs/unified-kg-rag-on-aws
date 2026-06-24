# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for chunker pure logic (AWS-free).

ChunkQualityValidator, ChunkProcessor.merge_small_chunks and
LineBasedBoundaryProcessor are plain classes with no AWS dependency and are
tested directly. SimpleTextChunker is exercised end-to-end on a Document with
its Bedrock token counter and boto session patched out (deterministic
word-count token estimate). The heavy LLM IntelligentTextChunker boundary path
is not exercised here (needs Bedrock); only its structured markdown/HTML
splitter helper is tested as a pure static method.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.ingestion.chunker as chunker_module
from aws_graphrag.adapters.ingestion.chunker import (
    ChunkProcessor,
    ChunkQualityValidator,
    IntelligentTextChunker,
    LineBasedBoundaryProcessor,
    SimpleTextChunker,
)
from aws_graphrag.domain.models import Config, Document, DocumentContent

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# ChunkQualityValidator
# --------------------------------------------------------------------------- #
class TestChunkQualityValidator:
    def test_no_chunks_invalid(self) -> None:
        v = ChunkQualityValidator(min_chunk_size=10, max_chunk_size=100)
        result = v.validate_chunks([])
        assert result["is_valid"] is False
        assert "No chunks generated" in result["issues"][0]

    def test_all_good_chunks_valid(self) -> None:
        v = ChunkQualityValidator(min_chunk_size=5, max_chunk_size=100)
        result = v.validate_chunks(["a" * 20, "b" * 30])
        assert result["is_valid"] is True
        assert result["issues"] == []
        assert result["metrics"]["total_chunks"] == 2

    def test_oversized_chunk_flagged(self) -> None:
        v = ChunkQualityValidator(min_chunk_size=5, max_chunk_size=10)
        result = v.validate_chunks(["a" * 50])
        assert result["is_valid"] is False
        assert result["metrics"]["oversized_chunks"] == 1
        assert any("exceed maximum" in i for i in result["issues"])

    def test_empty_chunk_flagged(self) -> None:
        v = ChunkQualityValidator(min_chunk_size=1, max_chunk_size=100)
        result = v.validate_chunks(["   ", "valid content"])
        assert result["metrics"]["empty_chunks"] == 1
        assert result["is_valid"] is False

    def test_undersized_gate_only_trips_above_half(self) -> None:
        v = ChunkQualityValidator(min_chunk_size=10, max_chunk_size=100)
        # 1 of 3 undersized (< 50%): not flagged as a too-many-undersized issue.
        ok = v.validate_chunks(["x" * 5, "y" * 20, "z" * 30])
        assert ok["metrics"]["undersized_chunks"] == 1
        assert ok["is_valid"] is True
        # 2 of 3 undersized (> 50%): flagged.
        bad = v.validate_chunks(["x" * 5, "y" * 5, "z" * 30])
        assert bad["metrics"]["undersized_chunks"] == 2
        assert any("undersized" in i for i in bad["issues"])
        assert bad["is_valid"] is False


# --------------------------------------------------------------------------- #
# ChunkProcessor.merge_small_chunks
# --------------------------------------------------------------------------- #
class TestMergeSmallChunks:
    def _proc(self, min_size=10, max_size=100) -> ChunkProcessor:
        return ChunkProcessor(
            chunk_overlap=0, min_chunk_size=min_size, max_chunk_size=max_size
        )

    def test_empty_returns_empty(self) -> None:
        assert self._proc().merge_small_chunks([]) == []

    def test_small_chunks_merged_when_within_max(self) -> None:
        proc = self._proc(min_size=10, max_size=100)
        # Two 5-char chunks (< min 10) merge into one 10-char chunk.
        out = proc.merge_small_chunks(["aaaaa", "bbbbb"])
        assert out == ["aaaaabbbbb"]

    def test_large_chunks_left_alone(self) -> None:
        proc = self._proc(min_size=10, max_size=100)
        chunks = ["a" * 20, "b" * 30]
        assert proc.merge_small_chunks(chunks) == chunks

    def test_merge_stops_at_max_size(self) -> None:
        proc = self._proc(min_size=10, max_size=12)
        # "aaaaa"(5) is small; adding "bbbbb"(5)=10 ok; adding next 5 -> 15 > 12 stop.
        out = proc.merge_small_chunks(["aaaaa", "bbbbb", "ccccc"])
        assert out == ["aaaaabbbbb", "ccccc"]

    def test_trailing_small_chunk_merged_into_previous(self) -> None:
        proc = self._proc(min_size=10, max_size=100)
        # last chunk (3 chars) is undersized -> merged into the previous.
        out = proc.merge_small_chunks(["a" * 20, "bbb"])
        assert out == ["a" * 20 + "bbb"]

    def test_trailing_small_chunk_kept_when_merge_overflows(self) -> None:
        proc = self._proc(min_size=10, max_size=22)
        out = proc.merge_small_chunks(["a" * 20, "bbb"])
        # 20 + 3 = 23 > max 22 -> last chunk stays separate.
        assert out == ["a" * 20, "bbb"]


# --------------------------------------------------------------------------- #
# LineBasedBoundaryProcessor
# --------------------------------------------------------------------------- #
class TestLineBoundaryProcessor:
    def _p(self, miss_rate=0.1) -> LineBasedBoundaryProcessor:
        return LineBasedBoundaryProcessor(max_line_miss_rate=miss_rate)

    def test_empty_boundaries(self) -> None:
        assert self._p().extract_line_numbers({}) == []

    def test_single_chunk_request_returns_empty(self) -> None:
        assert self._p().extract_line_numbers({"single_chunk": True}) == []

    def test_extract_from_chunk_boundaries_list_of_ints(self) -> None:
        nums = self._p().extract_line_numbers({"chunk_boundaries": [3, 2, 5]})
        # filtered to > 1, deduped and sorted.
        assert nums == [2, 3, 5]

    def test_line_number_one_filtered_out(self) -> None:
        assert self._p().extract_line_numbers({"chunk_boundaries": [1, 4]}) == [4]

    def test_text_key_ints_parsed(self) -> None:
        nums = self._p().extract_line_numbers(
            {"chunk_boundaries": [{"#text": "7"}, {"#text": "3"}]}
        )
        assert nums == [3, 7]

    def test_convert_line_numbers_to_indices(self) -> None:
        lines = ["abc", "de", "fghi"]  # lengths 3,2,4 (+ newline between)
        indices, missed = self._p().convert_line_numbers_to_indices(lines, [2])
        # line 2 starts at index len("abc")+1 = 4
        assert indices == [4]
        assert missed == 0

    def test_convert_counts_missed_line_numbers(self) -> None:
        lines = ["a", "b"]
        _, missed = self._p().convert_line_numbers_to_indices(lines, [2, 9])
        assert missed == 1  # line 9 > 2 total lines

    def test_validate_error_rate(self) -> None:
        p = self._p(miss_rate=0.1)
        assert p.validate_error_rate(0, 0) is True  # no lines -> trivially ok
        assert p.validate_error_rate(0, 10) is True
        assert p.validate_error_rate(1, 10) is True  # 10% == threshold
        assert p.validate_error_rate(2, 10) is False  # 20% > threshold


# --------------------------------------------------------------------------- #
# IntelligentTextChunker._get_structured_splitter (pure static helper)
# --------------------------------------------------------------------------- #
class TestStructuredSplitter:
    def test_markdown_header_split(self) -> None:
        splitter = IntelligentTextChunker._get_structured_splitter("markdown")
        docs = splitter.split_text("# Title\nbody one\n## Sub\nbody two")
        contents = [d.page_content for d in docs]
        assert len(contents) >= 2
        joined = "\n".join(contents)
        assert "body one" in joined and "body two" in joined

    # NOTE: the HTML header splitter requires the optional bs4/lxml dependency,
    # which is not installed in the AWS-free test environment, so the HTML branch
    # of _get_structured_splitter is not exercised here.


# --------------------------------------------------------------------------- #
# SimpleTextChunker (boto/token-counter patched out)
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_chunker(config: Config, mocker) -> SimpleTextChunker:
    mocker.patch.object(chunker_module, "boto3")
    mocker.patch.object(chunker_module, "get_assumed_role_boto_session")
    fake_counter = mocker.Mock()
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    mocker.patch.object(
        chunker_module, "BedrockTokenCounter", return_value=fake_counter
    )
    # Use 'text' content so we control the input regardless of default markdown.
    config.processing.chunking.content_type = "text"
    # Small sizes so a short test document actually splits.
    config.processing.chunking.min_chunk_size = 20
    config.processing.chunking.max_chunk_size = 200
    config.processing.chunking.chunk_overlap = 0
    config.processing.chunking.fallback_chunk_size = 60
    config.processing.chunking.pre_chunk_size = 200
    return SimpleTextChunker(config, show_progress=False)


def _doc(text: str) -> Document:
    return Document(
        page_content=text,
        document_id="doc1",
        file_name="f.txt",
        file_path="/tmp/f.txt",
        file_type="txt",
        total_pages=1,
        content=DocumentContent(text=text),
    )


class TestSimpleTextChunker:
    def test_chunks_a_document(self, simple_chunker) -> None:
        # ~180 chars, fallback_chunk_size 60 -> multiple chunks.
        body = "Sentence number {} here. ".format
        text = " ".join(body(i) for i in range(15))
        units = simple_chunker._chunk_single_document(_doc(text))
        assert len(units) >= 1
        joined = "".join(u.text for u in units)
        # All original content preserved across chunks.
        assert "Sentence number 0 here." in joined
        assert "Sentence number 14 here." in joined
        for u in units:
            assert u.attributes["chunking_method"] == "simple"
            assert u.document_ids == ["doc1"]

    def test_empty_content_returns_no_units(self, simple_chunker) -> None:
        assert simple_chunker._chunk_single_document(_doc("")) == []

    def test_token_count_uses_patched_counter(self, simple_chunker) -> None:
        units = simple_chunker._chunk_single_document(_doc("one two three four"))
        # word-count stub -> n_tokens equals number of words in the chunk.
        assert units[0].n_tokens == len(units[0].text.split())

    def test_chunk_documents_updates_stats(self, simple_chunker) -> None:
        units = simple_chunker.chunk_documents([_doc("alpha beta gamma delta epsilon")])
        assert simple_chunker.stats.num_total_documents == 1
        assert simple_chunker.stats.num_successful_documents == 1
        assert simple_chunker.stats.total_chunks_created == len(units)

    def test_error_document_counted_as_failed(self, simple_chunker) -> None:
        bad = _doc("text")
        bad.error_info = "parse failed"
        simple_chunker.chunk_documents([bad])
        assert simple_chunker.stats.num_failed_documents == 1
        assert simple_chunker.stats.num_successful_documents == 0
