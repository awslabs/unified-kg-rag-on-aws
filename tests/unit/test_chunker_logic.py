# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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

import unified_kg_rag.adapters.ingestion.chunker as chunker_module
from unified_kg_rag.adapters.ingestion.chunker import (
    ChunkerFactory,
    ChunkingStats,
    ChunkProcessor,
    ChunkQualityValidator,
    IntelligentTextChunker,
    LineBasedBoundaryProcessor,
    SimpleTextChunker,
)
from unified_kg_rag.domain.models import (
    ChunkingStrategy,
    Config,
    Document,
    DocumentContent,
)
from unified_kg_rag.shared import DataProcessingError

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
        return ChunkProcessor(min_chunk_size=min_size, max_chunk_size=max_size)

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


# --------------------------------------------------------------------------- #
# ChunkingStats derived properties
# --------------------------------------------------------------------------- #
class TestChunkingStats:
    def test_success_rate_zero_documents(self) -> None:
        assert ChunkingStats().success_rate == 0.0

    def test_success_rate_partial(self) -> None:
        s = ChunkingStats(num_total_documents=4, num_successful_documents=3)
        assert s.success_rate == 75.0

    def test_average_processing_time_guards_zero(self) -> None:
        # No processed documents -> denominator floored at 1 (no ZeroDivision).
        assert ChunkingStats().average_processing_time == 0.0

    def test_llm_failure_rate_no_pre_chunks(self) -> None:
        assert ChunkingStats().llm_failure_rate == 0.0

    def test_llm_failure_rate_no_failures(self) -> None:
        s = ChunkingStats(num_pre_chunks_processed=10, llm_processing_failures=0)
        assert s.llm_failure_rate == 0.0

    def test_llm_failure_rate_computed(self) -> None:
        s = ChunkingStats(num_pre_chunks_processed=10, llm_processing_failures=3)
        assert s.llm_failure_rate == 30.0

    def test_chunk_stats_empty(self) -> None:
        assert ChunkingStats().chunk_stats == {
            "total": 0,
            "min": 0,
            "avg": 0.0,
            "median": 0.0,
            "max": 0,
        }

    def test_chunk_stats_computed(self) -> None:
        s = ChunkingStats(num_chunk_chars=[10, 20, 30])
        cs = s.chunk_stats
        assert cs["total"] == 60
        assert cs["min"] == 10
        assert cs["max"] == 30
        assert cs["avg"] == 20.0
        assert cs["median"] == 20.0

    def test_add_num_chunk_chars_appends(self) -> None:
        s = ChunkingStats()
        s.add_num_chunk_chars(42)
        s.add_num_chunk_chars(7)
        assert s.num_chunk_chars == [42, 7]


# --------------------------------------------------------------------------- #
# LineBasedBoundaryProcessor — uncovered extraction branches
# --------------------------------------------------------------------------- #
class TestLineBoundaryProcessorExtra:
    def _p(self, miss_rate=0.1) -> LineBasedBoundaryProcessor:
        return LineBasedBoundaryProcessor(max_line_miss_rate=miss_rate)

    def test_top_level_line_number_field_int(self) -> None:
        assert self._p().extract_line_numbers({"line_number": 5}) == [5]

    def test_line_number_field_list_of_mixed(self) -> None:
        # int, {#text}, and string forms all coerced to int.
        nums = self._p().extract_line_numbers(
            {"line_number": [2, {"#text": "4"}, "6", {"#text": "bad"}, "nope"]}
        )
        assert nums == [2, 4, 6]

    def test_line_number_field_text_dict(self) -> None:
        assert self._p().extract_line_numbers({"line_number": {"#text": "8"}}) == [8]

    def test_line_number_field_string(self) -> None:
        assert self._p().extract_line_numbers({"line_number": "9"}) == [9]

    def test_top_level_list_of_ints_and_dicts(self) -> None:
        nums = self._p().extract_line_numbers([3, {"line_number": 5}, 1])
        assert nums == [3, 5]  # 1 filtered (must be > 1)

    def test_chunk_boundaries_dict_with_line_number(self) -> None:
        nums = self._p().extract_line_numbers(
            {"chunk_boundaries": {"line_number": [2, 4]}}
        )
        assert nums == [2, 4]

    def test_chunk_boundaries_list_nested_line_number(self) -> None:
        nums = self._p().extract_line_numbers(
            {"chunk_boundaries": [{"line_number": 3}, {"line_number": 7}]}
        )
        assert nums == [3, 7]

    def test_text_field_non_numeric_ignored(self) -> None:
        # {#text} that is not parseable as int yields nothing, not a crash.
        assert (
            self._p().extract_line_numbers({"chunk_boundaries": [{"#text": "x"}]}) == []
        )

    def test_convert_empty_line_numbers(self) -> None:
        indices, missed = self._p().convert_line_numbers_to_indices(["a", "b"], [])
        assert indices == []
        assert missed == 0


# --------------------------------------------------------------------------- #
# ChunkProcessor — fallback splitter & trailing-merge exception path
# --------------------------------------------------------------------------- #
class TestChunkProcessorExtra:
    def test_merge_swallows_exception_returns_input(self, mocker) -> None:
        proc = ChunkProcessor(min_chunk_size=10, max_chunk_size=100)
        # Force the inner loop to raise (len() called on a non-sized object).
        bad = [object()]
        out = proc.merge_small_chunks(bad)  # type: ignore[arg-type]
        assert out == bad


# --------------------------------------------------------------------------- #
# ChunkerFactory dispatch
# --------------------------------------------------------------------------- #
@pytest.fixture
def _patch_bedrock(mocker):
    """Patch every AWS/Bedrock seam so chunker __init__ never touches AWS."""
    mocker.patch.object(chunker_module, "boto3")
    mocker.patch.object(chunker_module, "get_assumed_role_boto_session")
    fake_counter = mocker.Mock()
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    mocker.patch.object(
        chunker_module, "BedrockTokenCounter", return_value=fake_counter
    )
    mocker.patch.object(chunker_module, "BedrockLanguageModelFactory")
    mocker.patch.object(chunker_module, "create_robust_xml_output_parser")
    mocker.patch.object(chunker_module, "setup_chain")


class TestChunkerFactory:
    def test_creates_simple(self, config: Config, _patch_bedrock) -> None:
        chunker = ChunkerFactory.create_chunker(
            config, ChunkingStrategy.SIMPLE, show_progress=False
        )
        assert isinstance(chunker, SimpleTextChunker)

    def test_creates_intelligent(self, config: Config, _patch_bedrock) -> None:
        chunker = ChunkerFactory.create_chunker(
            config, ChunkingStrategy.INTELLIGENT, show_progress=False
        )
        assert isinstance(chunker, IntelligentTextChunker)

    def test_unknown_type_raises(self, config: Config, _patch_bedrock) -> None:
        class _Fake:
            value = "bogus"

        with pytest.raises(DataProcessingError, match="Unknown chunker type"):
            ChunkerFactory.create_chunker(config, _Fake())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# IntelligentTextChunker — LLM boundary pipeline (chain.batch mocked)
# --------------------------------------------------------------------------- #
@pytest.fixture
def intelligent_chunker(config: Config, _patch_bedrock) -> IntelligentTextChunker:
    config.processing.chunking.content_type = "text"
    config.processing.chunking.min_chunk_size = 10
    config.processing.chunking.max_chunk_size = 400
    config.processing.chunking.chunk_overlap = 0
    config.processing.chunking.fallback_chunk_size = 60
    config.processing.chunking.pre_chunk_size = 1000
    config.processing.chunking.pre_chunk_overlap = 0
    config.processing.chunking.max_marker_miss_rate = 0.5
    config.processing.ignore_errors = True
    chunker = IntelligentTextChunker(config, show_progress=False)
    return chunker


class TestIntelligentChunkerHelpers:
    def test_create_chain_inputs_numbers_lines(self, intelligent_chunker) -> None:
        inputs = intelligent_chunker._create_chain_inputs(["line a\nline b"])
        assert inputs[0]["numbered_text"] == "1: line a\n2: line b"
        assert inputs[0]["min_chunk_size"] == 10
        assert inputs[0]["max_chunk_size"] == 400

    def test_create_fallback_chunks_splits_and_merges(
        self, intelligent_chunker
    ) -> None:
        out = intelligent_chunker._create_fallback_chunks("x" * 200)
        assert all(isinstance(c, str) for c in out)
        assert "".join(out) == "x" * 200

    def test_get_chunks_from_response_none_when_no_response(
        self, intelligent_chunker
    ) -> None:
        assert intelligent_chunker._get_chunks_from_response("text", None) is None

    def test_chunk_with_no_boundaries_returns_whole(self, intelligent_chunker) -> None:
        # No chunk_boundaries key -> single chunk (the whole text).
        assert intelligent_chunker._chunk_with_llm_boundaries("body", {}) == ["body"]

    def test_chunk_with_empty_line_numbers_returns_whole(
        self, intelligent_chunker
    ) -> None:
        out = intelligent_chunker._chunk_with_llm_boundaries(
            "body", {"chunk_boundaries": []}
        )
        assert out == ["body"]

    def test_chunk_with_boundaries_splits(self, intelligent_chunker) -> None:
        text = "first line here\nsecond chunk begins now and is fairly long indeed"
        out = intelligent_chunker._chunk_with_llm_boundaries(
            text, {"chunk_boundaries": [2]}
        )
        assert out is not None
        assert len(out) == 2
        assert out[0] == "first line here"
        assert out[1].startswith("second chunk")

    def test_high_miss_rate_returns_none(self, intelligent_chunker) -> None:
        intelligent_chunker.line_boundary_processor.max_line_miss_rate = 0.0
        # Line numbers far beyond the text -> all missed -> error rate > 0 -> None.
        out = intelligent_chunker._chunk_with_llm_boundaries(
            "a\nb", {"chunk_boundaries": [50, 60]}
        )
        assert out is None

    def test_extract_chunks_from_line_indices_empty(self, intelligent_chunker) -> None:
        assert intelligent_chunker._extract_chunks_from_line_indices("hi", []) == ["hi"]

    def test_extract_chunks_blank_text_empty_indices(self, intelligent_chunker) -> None:
        assert intelligent_chunker._extract_chunks_from_line_indices("   ", []) == []

    def test_split_large_line_chunks_splits_oversized(
        self, intelligent_chunker
    ) -> None:
        intelligent_chunker.config.processing.chunking.max_chunk_size = 30
        out = intelligent_chunker._split_large_line_chunks(["x" * 100, "small"])
        # The oversized chunk is split; "small" passes through unchanged.
        assert "small" in out
        assert len(out) > 2

    def test_create_pre_chunks_plain_text(self, intelligent_chunker) -> None:
        out = intelligent_chunker._create_pre_chunks("a b c d e f")
        assert out == ["a b c d e f"]

    def test_create_pre_chunks_markdown(self, intelligent_chunker) -> None:
        intelligent_chunker.config.processing.chunking.content_type = "markdown"
        out = intelligent_chunker._create_pre_chunks(
            "# Title\nbody text one\n## Sub\nbody text two"
        )
        joined = "\n".join(out)
        assert "body text one" in joined and "body text two" in joined

    def test_handle_llm_failure_records_stats(self, intelligent_chunker) -> None:
        final: list = []
        intelligent_chunker._handle_llm_failure("x" * 100, 0, final)
        assert intelligent_chunker.stats.llm_processing_failures == 1
        assert intelligent_chunker.stats.fallback_chunks_used is not None
        assert all(method == "fallback" for method, _, _ in final)


class TestIntelligentChunkerPipeline:
    def test_chunk_single_document_llm_path(self, intelligent_chunker, mocker) -> None:
        # The chain returns a boundary at line 2 for the single pre-chunk.
        intelligent_chunker.chunker.batch = mocker.Mock(
            return_value=[{"chunk_boundaries": [2]}]
        )
        text = "alpha beta gamma delta\nepsilon zeta eta theta iota kappa lambda"
        units = intelligent_chunker._chunk_single_document(_doc(text))
        assert len(units) >= 1
        assert all(u.attributes["chunking_method"] == "llm" for u in units)
        assert intelligent_chunker.stats.num_pre_chunks_processed == 1

    def test_chunk_single_document_fallback_on_none(
        self, intelligent_chunker, mocker
    ) -> None:
        # LLM returns None -> fallback chunks are used.
        intelligent_chunker.chunker.batch = mocker.Mock(return_value=[None])
        text = "alpha beta gamma delta epsilon zeta eta theta"
        units = intelligent_chunker._chunk_single_document(_doc(text))
        assert len(units) >= 1
        assert all(u.attributes["chunking_method"] == "fallback" for u in units)
        assert intelligent_chunker.stats.llm_processing_failures == 1

    def test_chunk_single_document_empty_content(self, intelligent_chunker) -> None:
        assert intelligent_chunker._chunk_single_document(_doc("")) == []

    def test_process_pre_chunks_exception_falls_back(
        self, intelligent_chunker, mocker
    ) -> None:
        # batch_processor raises with ignore_errors -> _get_boundary_results
        # returns [None]*n, so every pre-chunk takes the fallback branch.
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        intelligent_chunker.batch_processor = fake_bp
        out = intelligent_chunker._process_pre_chunks(["pre one", "pre two"], "f.txt")
        assert all(method == "fallback" for method, _, _ in out)

    def test_merge_structured_chunks_oversized_split(
        self, intelligent_chunker, mocker
    ) -> None:
        intelligent_chunker.config.processing.chunking.pre_chunk_size = 20
        intelligent_chunker.config.processing.chunking.min_chunk_size = 5
        Doc = mocker.Mock
        chunks = [Doc(page_content="a" * 50), Doc(page_content="b" * 3)]
        out = intelligent_chunker._merge_structured_chunks(chunks)
        assert out  # oversized first chunk is re-split via pre_splitter
