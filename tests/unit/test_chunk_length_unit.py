# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``processing.chunking.length_unit``: chunk sizes in characters or tokens.

The default (``chars``) must keep every chunk, and so every text-unit id and
content hash, byte-identical to the releases before the setting existed:
``_CHARS_GOLDEN`` was produced by the chunker before it. ``tokens`` measures
with the local estimate and must keep each chunk within its size even for
dense scripts, where 8,000 characters are ~6,000 tokens.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.fixtures.chunking_corpus import (
    english_text,
    japanese_text,
    korean_text,
    mixed_text,
)
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.ingestion import chunker as chunker_module
from unified_kg_rag.adapters.ingestion.chunker import (
    _SPLITTER_SEPARATORS,
    BaseChunker,
    ChunkProcessor,
    ChunkQualityValidator,
    IntelligentTextChunker,
    SimpleTextChunker,
)
from unified_kg_rag.adapters.ingestion.length_units import (
    GrowingText,
    TokenBudgetTextSplitter,
    _Piece,
    _Window,
    chunk_length_function,
)
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.application.ingestion.incremental import (
    run_extraction_fingerprint,
)
from unified_kg_rag.domain.ingestion.cache_keys import (
    changed_settings,
    classify,
    extraction_settings,
)
from unified_kg_rag.domain.ingestion.chunk_length import max_chars
from unified_kg_rag.domain.models import (
    ChunkLengthUnit,
    Config,
    Document,
    ExtractionSettings,
)
from unified_kg_rag.domain.prompts.base import max_chunk_chars

pytestmark = pytest.mark.unit

_TEXTS = {
    "en": lambda: english_text(30000),
    "ko": lambda: korean_text(20000),
    "ja": lambda: japanese_text(12000),
    "mixed": lambda: mixed_text(120000),
}

# Produced by the chunker of d90d1079 (before length_unit) with Config().
_CHARS_GOLDEN = {
    "simple:en": "8fefc0542e320fe1",
    "simple:ko": "3317b94e46361f4d",
    "simple:ja": "9737d4d443a0170a",
    "simple:mixed": "5fe01cb81e800178",
    "pre:en": "a7e11dcbb156f3e8",
    "fallback:en": "a9814d97927d768f",
    "needs_llm:en": "1c28f2eb0958c3d1",
    "inputs:en": "b6ba8f3940ecb0d7",
    "pre:ko": "75d932db6fac5808",
    "fallback:ko": "5aeb1bbd72780874",
    "needs_llm:ko": "1c28f2eb0958c3d1",
    "inputs:ko": "b16a7815ce98f1b3",
    "pre:ja": "f691e17bc121716c",
    "fallback:ja": "a11a226b6d4fd03b",
    "needs_llm:ja": "1c28f2eb0958c3d1",
    "inputs:ja": "dce848c5b1bdb69b",
    "pre:mixed": "c92d532818e9e51b",
    "fallback:mixed": "4d0595f6d65a8e4b",
    "needs_llm:mixed": "6fb086d658332a43",
    "inputs:mixed": "be391dd0e735b50d",
}

_TOKEN_SIZES = {
    "length_unit": "tokens",
    "min_chunk_size": 250,
    "max_chunk_size": 2000,
    "chunk_overlap": 125,
    "fallback_chunk_size": 1200,
    "pre_chunk_size": 12500,
    "pre_chunk_overlap": 125,
}


def _digest(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _config(chunking: dict | None = None) -> Config:
    return Config.model_validate({"processing": {"chunking": chunking or {}}})


def _doc(name: str, text: str) -> Document:
    return Document(
        document_id=f"doc-{name}",
        file_name=f"{name}.txt",
        file_path=f"{name}.txt",
        page_content=text,
    )


def _intelligent(config: Config) -> IntelligentTextChunker:
    with (
        mock.patch.object(chunker_module, "setup_chain"),
        mock.patch.object(chunker_module, "create_robust_xml_output_parser"),
        mock.patch("unified_kg_rag.adapters.aws.bedrock.BedrockLanguageModelFactory"),
    ):
        return IntelligentTextChunker(config, show_progress=False)


def test_default_chunks_are_byte_identical_to_before() -> None:
    simple = SimpleTextChunker(Config(), show_progress=False)
    intelligent = _intelligent(Config())
    got = {}
    for name, make in _TEXTS.items():
        text = make()
        units = simple.chunk_documents([_doc(name, text)])
        got[f"simple:{name}"] = _digest([(u.id, u.text, u.n_tokens) for u in units])
        pre = intelligent._create_pre_chunks(text)
        got[f"pre:{name}"] = _digest(pre)
        got[f"fallback:{name}"] = _digest(
            [intelligent._create_fallback_chunks(p) for p in pre]
        )
        got[f"needs_llm:{name}"] = _digest([intelligent._needs_llm(p) for p in pre])
        got[f"inputs:{name}"] = _digest(intelligent._create_chain_inputs(pre))
    assert got == _CHARS_GOLDEN


def test_default_unit_is_chars_and_uses_the_stock_splitter() -> None:
    chunker = SimpleTextChunker(Config(), show_progress=False)
    assert Config().processing.chunking.length_unit is ChunkLengthUnit.CHARS
    assert type(chunker.fallback_splitter) is not TokenBudgetTextSplitter
    assert chunker.chunk_processor.chunk_overlap == 500


def test_token_unit_wires_the_token_splitters() -> None:
    # Both splitters a chunker builds measure in the configured unit: a
    # character splitter here would cut 1,200 "tokens" as 1,200 characters.
    config = _config(_TOKEN_SIZES)
    fallback = SimpleTextChunker(config, show_progress=False).fallback_splitter
    pre = _intelligent(config).pre_splitter
    assert isinstance(fallback, TokenBudgetTextSplitter)
    assert isinstance(pre, TokenBudgetTextSplitter)
    assert (fallback._chunk_size, fallback._chunk_overlap) == (1200, 125)
    assert (pre._chunk_size, pre._chunk_overlap) == (12500, 125)


@pytest.mark.parametrize("name", ["ko", "ja", "en", "mixed"])
def test_token_chunks_stay_within_the_token_sizes(name: str) -> None:
    text = korean_text(8000) if name == "ko" else _TEXTS[name]()
    config = _config({**_TOKEN_SIZES, "chunker_type": "simple"})
    chunker = SimpleTextChunker(config, show_progress=False)

    pieces = chunker.fallback_splitter.split_text(text)
    assert max(map(estimate_token_count, pieces)) <= 1200
    units = chunker.chunk_documents([_doc(name, text)])
    assert units
    assert max(estimate_token_count(u.text) for u in units) <= 2000
    if name == "ko":
        # The same text overflows a 2,000-token budget counted in characters.
        assert estimate_token_count(text) > 4096


def test_korean_8000_chars_fit_the_reranker_limit_only_in_tokens() -> None:
    text = korean_text(8000)
    by_chars = SimpleTextChunker(
        _config({"chunker_type": "simple", "fallback_chunk_size": 8000}),
        show_progress=False,
    ).chunk_documents([_doc("ko", text)])
    by_tokens = SimpleTextChunker(
        _config({**_TOKEN_SIZES, "chunker_type": "simple"}), show_progress=False
    ).chunk_documents([_doc("ko", text)])
    assert max(estimate_token_count(u.text) for u in by_chars) > 4096
    assert max(estimate_token_count(u.text) for u in by_tokens) <= 2000


def test_the_stock_splitter_would_overshoot_a_token_budget() -> None:
    # Why TokenBudgetTextSplitter exists: summing per-piece estimates
    # undercounts the joined text (the // 4 floors per piece). " vendor" is
    # 7 characters, 1 token alone but 1.75 tokens per word when joined.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = " ".join(["vendor"] * 3000)
    kwargs = {
        "chunk_size": 600,
        "chunk_overlap": 100,
        "separators": _SPLITTER_SEPARATORS,
        "is_separator_regex": True,
    }
    stock = RecursiveCharacterTextSplitter(
        length_function=estimate_token_count, **kwargs
    )
    assert max(map(estimate_token_count, stock.split_text(text))) > 600
    budget = TokenBudgetTextSplitter(**kwargs)
    assert max(map(estimate_token_count, budget.split_text(text))) <= 600


def test_token_overlap_is_found_when_merging_small_chunks() -> None:
    # The overlap is set in tokens but located in characters: the processor
    # gets the character bound, so merged neighbours keep the overlap once.
    text = korean_text(6000, seed=11)
    config = _config(
        {**_TOKEN_SIZES, "chunker_type": "simple", "fallback_chunk_size": 300}
    )
    chunker = SimpleTextChunker(config, show_progress=False)
    assert chunker.chunk_processor.chunk_overlap == max_chars(
        125, ChunkLengthUnit.TOKENS
    )
    pieces = chunker.fallback_splitter.split_text(text)
    merged = chunker.chunk_processor.merge_small_chunks(pieces, source=text)
    for chunk in merged:
        assert chunk in text


def test_small_chunks_are_merged_by_tokens() -> None:
    # 200 characters but 50 tokens each: under a 100-token minimum, so pairs
    # merge; counted in characters none would.
    pieces = ["abcd" * 50] * 4
    by_tokens = ChunkProcessor(100, 300, length_function=estimate_token_count)
    assert len(by_tokens.merge_small_chunks(pieces)) == 2
    assert len(ChunkProcessor(100, 300).merge_small_chunks(pieces)) == 4
    chunker = SimpleTextChunker(_config(_TOKEN_SIZES), show_progress=False)
    assert chunker.chunk_processor.length_function is estimate_token_count
    assert chunker.quality_validator.length_function is estimate_token_count


@given(st.text(max_size=30), st.lists(st.text(max_size=12), max_size=6))
def test_growing_text_measures_what_it_became(text: str, tails: list[str]) -> None:
    growing = GrowingText(text, estimate_token_count)
    assert growing.size == estimate_token_count(text)
    for tail in tails:
        longer = growing.text + tail
        size = growing.size_of(longer)
        assert size == estimate_token_count(longer)
        growing.extend(longer, size)
        assert growing.size == estimate_token_count(growing.text)
    # Not an extension: measured whole.
    other = "x" + growing.text
    assert growing.size_of(other) == estimate_token_count(other)
    growing.extend(other, estimate_token_count(other))
    assert growing.size_of(other + " y") == estimate_token_count(other + " y")


@given(
    st.lists(st.text(alphabet="ab 가\n.。", max_size=40), max_size=20),
    st.booleans(),
    st.integers(min_value=1, max_value=30),
    st.integers(min_value=0, max_value=30),
    st.integers(min_value=0, max_value=8),
)
def test_merge_running_token_counts_equal_remeasuring(
    pieces: list[str], with_source: bool, minimum: int, extra: int, overlap: int
) -> None:
    # In tokens the merge measures each candidate by what it adds; any other
    # length function re-measures the joined text. Both merge alike.
    def remeasured(text: str) -> int:
        return estimate_token_count(text)

    source = "".join(pieces) if with_source else None
    running, plain = (
        ChunkProcessor(
            minimum, minimum + extra, chunk_overlap=overlap, length_function=f
        )
        for f in (estimate_token_count, remeasured)
    )
    assert running.merge_small_chunks(pieces, source=source) == (
        plain.merge_small_chunks(pieces, source=source)
    )


# The splitter never merges empty pieces (its splits drop them).
@given(
    st.lists(st.text(min_size=1, max_size=12), max_size=12),
    st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=3),
)
def test_window_tally_equals_the_estimate_of_the_joined_text(
    pieces: list[str], extra: list[str]
) -> None:
    window = _Window()
    for piece in pieces:
        window.push(_Piece.of(piece))
    assert window.tokens() == estimate_token_count("".join(pieces))
    # Appended pieces join at a seam too (a word may continue across it).
    appended = [_Piece.of(e) for e in extra]
    assert window.tokens(*appended) == estimate_token_count("".join(pieces + extra))
    if pieces:
        window.pop_front()
        assert window.tokens() == estimate_token_count("".join(pieces[1:]))
        assert window.tokens(*appended) == estimate_token_count(
            "".join(pieces[1:] + extra)
        )


def test_token_overlap_stays_within_its_budget() -> None:
    # Distinct words, so the text two neighbours share is their overlap.
    text = " ".join(f"w{i:04d}" for i in range(3000))
    splitter = TokenBudgetTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=_SPLITTER_SEPARATORS,
        is_separator_regex=True,
    )
    chunks = splitter.split_text(text)
    assert len(chunks) > 2
    for left, right in itertools.pairwise(chunks):
        seen = set(left.split())
        shared = " ".join(w for w in right.split() if w in seen)
        assert 0 < estimate_token_count(shared) <= 100


@given(
    st.text(alphabet="ab 가나\n。.", max_size=400),
    st.integers(min_value=2, max_value=60),
)
def test_token_splitter_chunks_never_exceed_the_size(text: str, size: int) -> None:
    splitter = TokenBudgetTextSplitter(
        chunk_size=size,
        chunk_overlap=size // 3,
        separators=_SPLITTER_SEPARATORS,
        is_separator_regex=True,
    )
    for chunk in splitter.split_text(text):
        assert estimate_token_count(chunk) <= size


def test_token_splitter_keeps_separators() -> None:
    # Its merge measures the splits alone: without keep_separator the
    # separators dropped between them would go uncounted.
    with pytest.raises(ValueError, match="keep_separator"):
        TokenBudgetTextSplitter(chunk_size=10, chunk_overlap=0, keep_separator=False)
    for keep in (True, "start", "end"):
        splitter = TokenBudgetTextSplitter(
            chunk_size=8, chunk_overlap=0, keep_separator=keep
        )
        for chunk in splitter.split_text("Vendor ships parts. " * 20):
            assert estimate_token_count(chunk) <= 8


def test_quality_validator_counts_in_the_configured_unit() -> None:
    chunk = "가" * 300  # 300 characters, 300 tokens
    by_tokens = ChunkQualityValidator(10, 200, length_function=estimate_token_count)
    assert by_tokens.validate_chunks([chunk])["metrics"]["oversized_chunks"] == 1
    latin = "abcd" * 100  # 400 characters, 100 tokens
    assert by_tokens.validate_chunks([latin])["metrics"]["oversized_chunks"] == 0
    assert (
        ChunkQualityValidator(10, 200).validate_chunks([latin])["metrics"][
            "oversized_chunks"
        ]
        == 1
    )


def test_intelligent_chunker_measures_in_tokens() -> None:
    chunker = _intelligent(_config(_TOKEN_SIZES))
    dense = "가" * 600  # 600 tokens: >= 2 * min (250), so worth a model call
    latin = "word " * 120  # 600 characters, 120 tokens: below 2 * min
    assert chunker._needs_llm(dense)
    assert not chunker._needs_llm(latin)
    # An LLM piece over max_chunk_size (2,000 tokens) is split by tokens.
    big = korean_text(6000, seed=5)
    assert estimate_token_count(big) > 2000
    split = chunker._split_large_line_chunks([big])
    assert max(map(estimate_token_count, split)) <= 1200
    # 5,000 characters but 1,000 tokens: within max_chunk_size, kept whole.
    english = "word " * 1000
    assert chunker._split_large_line_chunks([english]) == [english]


def test_boundary_prompt_gets_the_token_range_in_characters() -> None:
    chunker = _intelligent(_config(_TOKEN_SIZES))
    (latin,) = chunker._create_chain_inputs(["abcd" * 1000])  # 4 chars/token
    assert (latin["min_chunk_size"], latin["max_chunk_size"]) == (1000, 8000)
    (dense,) = chunker._create_chain_inputs(["가" * 1000])  # 1 char/token
    assert (dense["min_chunk_size"], dense["max_chunk_size"]) == (250, 2000)
    chars = _intelligent(Config())
    (default,) = chars._create_chain_inputs(["가" * 1000])
    assert (default["min_chunk_size"], default["max_chunk_size"]) == (1000, 8000)


def test_create_splitter_defaults_to_characters() -> None:
    splitter = BaseChunker._create_splitter(10, 0)
    assert type(splitter) is not TokenBudgetTextSplitter
    assert max(map(len, splitter.split_text("가나다 " * 20))) <= 10
    tokens = BaseChunker._create_splitter(10, 0, length_unit=ChunkLengthUnit.TOKENS)
    assert isinstance(tokens, TokenBudgetTextSplitter)


def test_the_unit_is_compared_by_value() -> None:
    # A unit set as its string (an assignment after load) is the same unit.
    tokens: Any = "tokens"
    assert chunk_length_function(tokens) is estimate_token_count
    assert max_chars(2000, tokens) == 8003
    splitter = BaseChunker._create_splitter(10, 0, length_unit=tokens)
    assert isinstance(splitter, TokenBudgetTextSplitter)
    chunker = _intelligent(_config(_TOKEN_SIZES))
    chunker.length_unit = tokens
    assert chunker._prompt_sizes("abcd" * 1000) == (1000, 8000)


def test_output_floor_char_bound_follows_the_unit() -> None:
    assert max_chunk_chars(Config()) == 8000
    assert max_chunk_chars(_config(_TOKEN_SIZES)) == 2000 * 4 + 3
    assert max_chars(2000, ChunkLengthUnit.TOKENS) == 8003
    assert max_chars(2000, ChunkLengthUnit.CHARS) == 2000


# --- cache keys and incremental re-extraction ----------------------------------


def test_length_unit_keys_the_chunking_stage() -> None:
    rule = classify("processing.chunking.length_unit")
    assert rule is not None and getattr(rule, "stage", None) is not None


def test_upgrading_keeps_the_extraction_settings_of_a_char_corpus() -> None:
    # A record written before the field existed lacks it: at its default the
    # corpus is not re-extracted; set to tokens, it is.
    new = extraction_settings(Config())
    path = "processing.chunking.length_unit"
    assert path in new.values
    old = ExtractionSettings(
        version=new.version,
        values={k: v for k, v in new.values.items() if k != path},
    )
    assert changed_settings(old, new) == []
    assert path in changed_settings(old, extraction_settings(_config(_TOKEN_SIZES)))

    store = FakeDocStatusStore()
    store.put_extraction_settings("scope", "recorded-fp", old)
    assert run_extraction_fingerprint(store, "scope", new) == "recorded-fp"
    tokens = extraction_settings(_config(_TOKEN_SIZES))
    assert run_extraction_fingerprint(store, "scope", tokens) == tokens.fingerprint
