# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The explicit decline marker and the insufficient-context policy
(``search.abstention``), AWS-free: marker detection in every answer language,
the streamed filter, and the chain's prompt, metadata and sources."""

from __future__ import annotations

import ast

import pytest
from hypothesis import given
from hypothesis import strategies as st

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    NO_CONTEXT_ANSWER,
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.ingestion.cache_keys import answer_settings
from unified_kg_rag.domain.models import (
    Config,
    LanguageCode,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts.retrieval import (
    ABSTENTION_INSTRUCTIONS,
    DECLINE_INSTRUCTIONS,
    HEDGE_INSTRUCTIONS,
)
from unified_kg_rag.domain.retrieval.abstention import (
    ABSTENTION_MARKER,
    CONTEXT_SUFFICIENT_KEY,
    MARKER_LOOKAHEAD,
    MarkerStreamFilter,
    split_marker,
)
from unified_kg_rag.domain.retrieval.no_context import NO_CONTEXT_ANSWERS
from unified_kg_rag.ports.guardrail import GuardAction, GuardSource, GuardVerdict

pytestmark = pytest.mark.unit


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "rest"),
    [
        (
            "[NO_ANSWER] The context does not name the supplier.",
            "The context does not name the supplier.",
        ),
        # Korean, Japanese and Chinese explanations after the ASCII token.
        (
            "[NO_ANSWER] 컨텍스트에 공급업체 정보가 없습니다.",
            "컨텍스트에 공급업체 정보가 없습니다.",
        ),
        (
            "[NO_ANSWER] 文脈には供給業者の情報がありません。",
            "文脈には供給業者の情報がありません。",
        ),
        ("[NO_ANSWER] 上下文中没有供应商的信息。", "上下文中没有供应商的信息。"),
        # Full-width brackets and letters (CJK output) normalize under NFKC.
        ("［NO_ANSWER］文脈には情報がありません。", "文脈には情報がありません。"),
        ("［ＮＯ＿ＡＮＳＷＥＲ］ 정보가 없습니다.", "정보가 없습니다."),
        ("【NO_ANSWER】上下文中没有。", "上下文中没有。"),
        # Markdown emphasis, a heading mark, case and separators.
        ("**[NO_ANSWER]** The context lacks it.", "The context lacks it."),
        ("## [no answer]: missing", "missing"),
        ("> [NO-ANSWER] - missing", "missing"),
        ("[NO_ANSWER]", ""),
    ],
)
def test_the_marker_is_detected_in_every_language(answer: str, rest: str) -> None:
    assert split_marker(answer) == (True, rest)


@pytest.mark.parametrize(
    "answer",
    [
        "Vendor supplies Buyer.",
        "공급업체는 Vendor입니다. [NO_ANSWER]",  # not at the start
        "The answer is [NO_ANSWER].",
        "[NOTE] Vendor supplies Buyer.",
        "No answer is given in clause 4, but Vendor supplies Buyer.",
        "答えは[NO_ANSWER]ではありません。",
        "",
    ],
)
def test_text_that_does_not_start_with_the_marker_is_an_answer(answer: str) -> None:
    assert split_marker(answer) == (False, answer)


def _streamed(text: str, size: int, *, strip: bool = True) -> tuple[bool, str]:
    marker = MarkerStreamFilter(strip=strip)
    shown = [marker.feed(text[i : i + size]) for i in range(0, len(text), size)]
    shown.append(marker.finish())
    return marker.marked, "".join(shown)


@pytest.mark.parametrize("size", [1, 2, 5, 200])
@pytest.mark.parametrize(
    ("text", "marked", "shown"),
    [
        ("[NO_ANSWER] 정보가 없습니다.", True, "정보가 없습니다."),
        ("［NO_ANSWER］文脈にはありません。", True, "文脈にはありません。"),
        ("**[NO_ANSWER]** 上下文中没有。", True, "上下文中没有。"),
        ("Vendor supplies Buyer.", False, "Vendor supplies Buyer."),
        ("[NOTE] Vendor supplies Buyer.", False, "[NOTE] Vendor supplies Buyer."),
        ("**Vendor** supplies Buyer.", False, "**Vendor** supplies Buyer."),
        ("[N", False, "[N"),
    ],
)
def test_the_stream_filter_matches_detection_for_any_chunking(
    text: str, marked: bool, shown: str, size: int
) -> None:
    assert _streamed(text, size) == (marked, shown)
    # Kept as written when not stripped, still detected.
    assert _streamed(text, size, strip=False) == (marked, text)


# Pieces a streamed answer is made of: decoration, the token (in its
# variants), separators, words in several scripts and whitespace.
_PIECES = st.sampled_from(
    [
        "[NO_ANSWER]",
        "［NO_ANSWER］",
        "【NO_ANSWER】",
        "[no answer]",
        "[NO-ANSWER]",
        "[NOTE]",
        "**",
        "## ",
        "> ",
        '"',
        ": ",
        " - ",
        " ",
        "  ",
        "\n",
        "\n\n",
        "\t",
        "The",
        " context",
        " lacks it.",
        "공급업체 ",
        "정보가 없습니다.",
        "文脈には",
        "上下文中没有。",
    ]
)


@given(
    pieces=st.lists(_PIECES, max_size=12),
    sizes=st.lists(st.integers(min_value=1, max_value=9), min_size=1, max_size=40),
)
def test_a_streamed_answer_shows_what_detection_leaves(
    pieces: list[str], sizes: list[int]
) -> None:
    # Any answer, cut at any chunk boundaries: the stream decides as the
    # whole answer is decided, and shows what split_marker leaves of it, up
    # to the whitespace that ends it (shown text is never taken back).
    answer = "".join(pieces)
    chunks, start = [], 0
    while start < len(answer):
        size = sizes[len(chunks) % len(sizes)]
        chunks.append(answer[start : start + size])
        start += size
    for strip in (True, False):
        marker = MarkerStreamFilter(strip=strip)
        shown = "".join([marker.feed(c) for c in chunks] + [marker.finish()])
        marked, rest = split_marker(answer)
        assert marker.marked is marked
        if marked and strip:
            assert shown.strip() == rest
            assert answer.endswith(shown) and not shown[:1].isspace()
        else:
            assert shown == answer


def test_a_chunk_boundary_after_the_marker_keeps_the_space_between_words() -> None:
    # "[NO_ANSWER] The " + "context ..." was shown as "Thecontext ...".
    assert _streamed("[NO_ANSWER] The context lacks it.", 16) == (
        True,
        "The context lacks it.",
    )
    marker = MarkerStreamFilter()
    assert marker.feed("[NO_ANSWER] 공급업체 ") == "공급업체 "
    assert marker.feed("정보가 없습니다.") == "정보가 없습니다."


def test_the_stream_filter_holds_back_at_most_the_lookahead() -> None:
    marker = MarkerStreamFilter()
    assert marker.feed("[") == ""
    # Past the lookahead the start is shown even if undecidable.
    text = " " * MARKER_LOOKAHEAD + "Vendor"
    assert MarkerStreamFilter().feed(text) == text
    # A non-marker start is shown at once.
    assert MarkerStreamFilter().feed("Vendor") == "Vendor"


@pytest.mark.parametrize("size", [1, 7, 200])
def test_a_marker_counts_only_within_the_lookahead(size: int) -> None:
    # The token ending on the lookahead's last character is a marker; one
    # more character of decoration before it and it is not, for the stream
    # (whatever its chunks) and the whole answer alike.
    token = "[NO_ANSWER]"
    inside = " " * (MARKER_LOOKAHEAD - len(token)) + token + " 정보가 없습니다."
    outside = " " + inside
    assert split_marker(inside) == (True, "정보가 없습니다.")
    assert split_marker(outside) == (False, outside)
    assert _streamed(inside, size) == (True, "정보가 없습니다.")
    assert _streamed(outside, size) == (False, outside)


def test_the_stream_decides_on_the_lookahead_s_last_character() -> None:
    marker = MarkerStreamFilter()
    held = [marker.feed(" ") for _ in range(MARKER_LOOKAHEAD - 1)]
    assert set(held) == {""}
    assert marker.feed(" ") == " " * MARKER_LOOKAHEAD
    assert marker.marked is False


# --- the chain -------------------------------------------------------------


class _Retriever:
    def __init__(self, tag: str, *, empty: bool = False) -> None:
        self.tag = tag
        self.empty = empty

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        if self.empty:
            return []
        return [
            RetrievalResult(
                content=f"{self.tag}: Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source=f"{self.tag}-doc-1",
                retriever_type=self.tag,
                metadata={"id": f"{self.tag}-1", "text_unit_ids": ["tu-1"]},
            )
        ]


class _Guard:
    """Records the screened answers; blocks one naming BLOCKED."""

    def __init__(self) -> None:
        self.outputs: list[str] = []

    async def acheck(self, text: str, source: GuardSource, **kwargs) -> GuardVerdict:
        if source is not GuardSource.OUTPUT:
            return GuardVerdict(action=GuardAction.NONE, text=text)
        self.outputs.append(text)
        if "BLOCKED" in text:
            return GuardVerdict(action=GuardAction.BLOCKED, text="Refused.")
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _chain(
    config: Config, respond, *, empty: bool = False, guard: _Guard | None = None
) -> tuple[GraphRAGChain, list[str]]:
    systems: list[str] = []
    config.providers.rerank.kind = "none"

    def answer(system: str, human: str) -> str:
        if system.startswith("[{"):
            # Content blocks (a prompt-cache marker): the fake joins their repr.
            system = "".join(b["text"] for b in ast.literal_eval(system))
        systems.append(system)
        return respond(system)

    chain = GraphRAGChain(
        config=config,
        model_factory=ScriptedLLMFactory(answer),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: _Retriever("document", empty=empty),
            RetrieverRole.GRAPH: lambda: _Retriever("graph", empty=empty),
        },
        guard=guard,
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain, systems


def _query(text: str = "Who supplies Buyer?") -> RAGInput:
    return RAGInput(
        query=text,
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
    )


def _declining(system: str) -> str:
    # A model that follows the instruction when it is given.
    if ABSTENTION_MARKER in system:
        return f"{ABSTENTION_MARKER} 컨텍스트에 공급업체 정보가 없습니다."
    return "Vendor supplies Buyer."


async def test_mode_off_leaves_the_prompt_and_answer_as_before(config: Config) -> None:
    chain, systems = _chain(config, _declining)
    out = await chain.ainvoke(_query())
    assert out.answer == "Vendor supplies Buyer."
    assert ABSTENTION_MARKER not in systems[0]
    assert "abstained" not in out.metadata
    assert "abstention_reason" not in out.metadata


async def test_mode_off_does_not_read_the_marker(config: Config) -> None:
    # A model that writes the token unasked is not a declined answer.
    chain, _ = _chain(config, lambda _s: f"{ABSTENTION_MARKER} Vendor.")
    out = await chain.ainvoke(_query())
    assert out.answer == f"{ABSTENTION_MARKER} Vendor."
    assert "abstained" not in out.metadata


async def test_a_marked_decline_is_abstained_stripped_and_keeps_its_sources(
    config: Config,
) -> None:
    config.search.abstention.mode = "mark"
    chain, systems = _chain(config, _declining)
    out = await chain.ainvoke(_query("누가 Buyer에게 공급합니까?"))
    assert ABSTENTION_INSTRUCTIONS in systems[0]
    assert out.answer == "컨텍스트에 공급업체 정보가 없습니다."
    assert out.metadata["abstained"] is True
    assert out.metadata["abstention_reason"] == "marker"
    # The model saw the context, so the decline still reports it.
    assert [s["source"] for s in out.sources] == ["document-doc-1"]
    assert out.sources[0]["metadata"]["text_unit_ids"] == ["tu-1"]


async def test_mark_mode_appends_to_the_comprehensive_and_custom_prompts(
    config: Config,
) -> None:
    config.search.abstention.mode = "mark"
    config.search.answer_prompt_style = {SearchStrategy.SIMPLE: "comprehensive"}
    config.custom_prompts.comprehensive_answer_system = "Answer from the context."
    chain, systems = _chain(config, _declining)
    await chain.ainvoke(_query())
    assert "Answer from the context." + ABSTENTION_INSTRUCTIONS in systems[0]
    assert "expert AI assistant" not in systems[0]


async def test_an_answer_in_mark_mode_is_not_abstained(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, _ = _chain(config, lambda _s: "Vendor supplies Buyer.")
    out = await chain.ainvoke(_query())
    assert out.answer == "Vendor supplies Buyer."
    assert "abstained" not in out.metadata


async def test_strip_marker_false_keeps_the_token(config: Config) -> None:
    config.search.abstention.mode = "mark"
    config.search.abstention.strip_marker = False
    chain, _ = _chain(config, _declining)
    out = await chain.ainvoke(_query())
    assert out.answer.startswith(ABSTENTION_MARKER)
    assert out.metadata["abstention_reason"] == "marker"


async def test_a_bare_token_becomes_the_no_context_reply(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, _ = _chain(config, lambda _s: f"  **{ABSTENTION_MARKER}**  ")
    out = await chain.ainvoke(_query())
    assert out.answer == NO_CONTEXT_ANSWER
    assert out.metadata["abstained"] is True


async def test_a_streamed_decline_is_stripped_and_saved_stripped(
    config: Config,
) -> None:
    config.search.abstention.mode = "mark"
    chain, _ = _chain(config, _declining)
    saved = []

    async def save(output, rag_input):
        saved.append(output)

    chain._save_memory = save  # type: ignore[method-assign]
    chunks = [chunk async for chunk in chain.astream(_query())]
    assert "".join(chunks) == "컨텍스트에 공급업체 정보가 없습니다."
    assert saved[0].answer == "컨텍스트에 공급업체 정보가 없습니다."
    assert saved[0].metadata["abstention_reason"] == "marker"


async def test_a_streamed_bare_token_yields_the_no_context_reply(
    config: Config,
) -> None:
    config.search.abstention.mode = "mark"
    chain, _ = _chain(config, lambda _s: ABSTENTION_MARKER)
    chunks = [chunk async for chunk in chain.astream(_query())]
    assert "".join(chunks) == NO_CONTEXT_ANSWER


def _flag_insufficient(chain: GraphRAGChain) -> None:
    """Make the strategy's results carry context_sufficient: false."""
    search = chain._search_step

    async def flagged(state, config=None):
        result = await search(state, config)
        result.metadata[CONTEXT_SUFFICIENT_KEY] = False
        result.metadata["insufficiency_reason"] = "bridge entity not found"
        return result

    chain._search_step = flagged  # type: ignore[method-assign]
    chain.chain = chain._build_chain()


async def test_an_unflagged_context_ignores_the_policy(config: Config) -> None:
    config.search.abstention.on_insufficient = "abstain"
    chain, systems = _chain(config, _declining)
    out = await chain.ainvoke(_query())
    assert out.answer == "Vendor supplies Buyer."
    assert DECLINE_INSTRUCTIONS not in systems[0]


async def test_hedge_asks_the_model_to_say_what_is_missing(config: Config) -> None:
    chain, systems = _chain(config, lambda _s: "Vendor, probably.")
    _flag_insufficient(chain)
    out = await chain.ainvoke(_query())
    assert HEDGE_INSTRUCTIONS in systems[0]
    assert out.answer == "Vendor, probably."
    assert "abstained" not in out.metadata
    assert out.metadata[CONTEXT_SUFFICIENT_KEY] is False


async def test_hedge_and_mark_combine(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, systems = _chain(config, _declining)
    _flag_insufficient(chain)
    out = await chain.ainvoke(_query())
    assert ABSTENTION_INSTRUCTIONS + HEDGE_INSTRUCTIONS in systems[0]
    assert out.metadata["abstention_reason"] == "marker"


async def test_abstain_declines_in_the_model_s_words_and_keeps_sources(
    config: Config,
) -> None:
    config.search.abstention.on_insufficient = "abstain"
    chain, systems = _chain(config, _declining)
    _flag_insufficient(chain)
    out = await chain.ainvoke(_query("誰がBuyerに供給していますか?"))
    assert DECLINE_INSTRUCTIONS in systems[0]
    assert ABSTENTION_INSTRUCTIONS not in systems[0]
    assert out.answer == "컨텍스트에 공급업체 정보가 없습니다."
    assert out.metadata["abstained"] is True
    assert out.metadata["abstention_reason"] == "insufficient_context"
    assert out.sources


async def test_abstain_counts_even_when_the_model_answers(config: Config) -> None:
    config.search.abstention.on_insufficient = "abstain"
    chain, _ = _chain(config, lambda _s: "Vendor supplies Buyer.")
    _flag_insufficient(chain)
    out = await chain.ainvoke(_query())
    assert out.metadata["abstention_reason"] == "insufficient_context"
    # The answer the model wrote instead of the decline is not returned.
    assert out.answer == NO_CONTEXT_ANSWER


async def test_a_streamed_abstain_shows_no_answer_the_model_wrote_instead(
    config: Config,
) -> None:
    config.search.abstention.on_insufficient = "abstain"
    chain, _ = _chain(config, lambda _s: "Vendor supplies Buyer.")
    _flag_insufficient(chain)
    chunks = [chunk async for chunk in chain.astream(_query())]
    assert "".join(chunks) == NO_CONTEXT_ANSWER


async def test_a_streamed_abstain_shows_the_model_s_decline(config: Config) -> None:
    config.search.abstention.on_insufficient = "abstain"
    chain, _ = _chain(config, _declining)
    _flag_insufficient(chain)
    chunks = [chunk async for chunk in chain.astream(_query())]
    assert "".join(chunks) == "컨텍스트에 공급업체 정보가 없습니다."


async def test_the_no_context_reply_names_its_reason(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, systems = _chain(config, _declining, empty=True)
    out = await chain.ainvoke(_query())
    assert out.answer == NO_CONTEXT_ANSWER
    assert out.metadata["abstention_reason"] == "no_context"
    assert systems == []


@pytest.mark.parametrize("stream", [False, True])
async def test_the_guardrail_screens_the_stripped_decline(stream: bool) -> None:
    config = Config()
    config.aws.bedrock.guardrail.identifier = "test-guardrail"
    config.search.abstention.mode = "mark"
    guard = _Guard()
    chain, _ = _chain(config, _declining, guard=guard)
    if stream:
        chunks = [chunk async for chunk in chain.astream(_query())]
        assert chunks == ["컨텍스트에 공급업체 정보가 없습니다."]
    else:
        out = await chain.ainvoke(_query())
        assert out.metadata["abstention_reason"] == "marker"
        assert out.sources
    assert guard.outputs == ["컨텍스트에 공급업체 정보가 없습니다."]


async def test_a_blocked_decline_keeps_no_sources() -> None:
    config = Config()
    config.aws.bedrock.guardrail.identifier = "test-guardrail"
    config.search.abstention.mode = "mark"
    chain, _ = _chain(
        config, lambda _s: f"{ABSTENTION_MARKER} BLOCKED text", guard=_Guard()
    )
    out = await chain.ainvoke(_query())
    assert out.answer == "Refused."
    assert out.sources == [] and out.search_results.results == []


def test_the_defaults_leave_the_answer_settings_hash_unchanged() -> None:
    # A run at the defaults records the same settings as before the
    # section existed; a changed setting is recorded.
    assert not any(k.startswith("search.abstention") for k in answer_settings(Config()))
    config = Config()
    config.search.abstention.mode = "mark"
    assert answer_settings(config)["search.abstention.mode"] == "mark"


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("누가 구매자에게 공급합니까?", LanguageCode.KO),
        ("誰がバイヤーに供給していますか?", LanguageCode.JA),
        ("谁向买方供货?", LanguageCode.ZH),
        ("Wer beliefert den Käufer?", LanguageCode.DE),
    ],
)
async def test_the_fixed_replies_are_in_the_question_s_language(
    config: Config, question: str, language: LanguageCode
) -> None:
    # The answer prompt answers in the query's language; the fixed replies
    # of a query no model answered do too, and stay flagged.
    chain, systems = _chain(config, _declining, empty=True)
    out = await chain.ainvoke(_query(question))
    assert out.answer == NO_CONTEXT_ANSWERS[language]
    assert out.metadata["abstained"] is True and systems == []
    chunks = [chunk async for chunk in chain.astream(_query(question))]
    assert "".join(chunks) == NO_CONTEXT_ANSWERS[language]
    # The insufficient-context policy declines in it as well.
    config.search.abstention.on_insufficient = "abstain"
    chain, _ = _chain(config, lambda _s: "Vendor supplies Buyer.")
    _flag_insufficient(chain)
    out = await chain.ainvoke(_query(question))
    assert out.answer == NO_CONTEXT_ANSWERS[language]
    assert out.metadata["abstention_reason"] == "insufficient_context"
