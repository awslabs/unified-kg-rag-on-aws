# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TokenManager — context-window budgeting/optimization shared by
all search strategies (AWS-free: the Bedrock token counter is patched out).

These exercise the real budget math, per-type share packing, the
quality-score blend, the budget-exceeded short circuit, and the empty-context
fallback string. The token counter is replaced with a deterministic
word-count stub so assertions are exact and AWS is never touched.
"""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import ValidationError

from unified_kg_rag.adapters.aws.bedrock_models import get_language_model_info
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.token_manager import (
    ContextSection,
    OptimizedContext,
    TokenManager,
)
from unified_kg_rag.domain.models import (
    Config,
    LanguageModelId,
    ModelRole,
    RetrievalResult,
    SectionType,
)
from unified_kg_rag.domain.models.config import ContextTypeBudgetConfig
from unified_kg_rag.ports.model_factory import effective_max_output_tokens

pytestmark = pytest.mark.unit


def _make_manager(
    mocker,
    max_context_tokens: int | None = None,
    answer_model_id: LanguageModelId | None = None,
    enable_1m_context: bool = False,
    default_max_output_tokens: int | None = 16384,
) -> TokenManager:
    """Build a TokenManager with Bedrock/boto wiring stubbed and a deterministic
    word-count token counter (1 token per whitespace-delimited word).

    ``max_context_tokens=None`` (the default) exercises the production path where
    the budget is derived from the answer model's context window.
    """
    fake_counter = mocker.Mock()
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())

    config = Config()
    config.search.token_manager.max_context_tokens = max_context_tokens
    config.aws.bedrock.enable_1m_context = enable_1m_context
    config.models.default_max_output_tokens = default_max_output_tokens
    if answer_model_id is not None:
        config.models.roles.answer_generation = answer_model_id
    providers = Providers(
        config,
        boto_session=mocker.Mock(),
        token_counter_factory=lambda *_, **__: fake_counter,
    )
    return TokenManager(config, providers=providers)


# Under the word-count stub a section costs one token more than its content
# (its "[source_id]" prefix) and every section type present costs its header
# ("##### Document Chunks", "##### Community Reports", "##### Other Context":
# three words each).
HEADER = 3


def _r(content: str, score: float, retriever_type: str, source: str) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, retriever_type=retriever_type, source=source
    )


@pytest.mark.parametrize("model_id", list(LanguageModelId))
def test_shipped_explicit_budget_is_honoured_for_every_curated_model(
    mocker, model_id: LanguageModelId
) -> None:
    # The shipped default (30000) must not be clamped for any curated answer
    # model, with or without the output cap.
    shipped = Config().search.token_manager.max_context_tokens
    assert shipped == 30000
    for cap in (16384, None):
        mgr = _make_manager(
            mocker,
            max_context_tokens=shipped,
            answer_model_id=model_id,
            default_max_output_tokens=cap,
        )
        assert mgr._max_context_tokens == shipped, (model_id, cap)


class TestContextBudgetDerivation:
    """The prompt budget must track the answer model's real context window.

    A hardcoded budget silently overflows a smaller model's window (the old
    default of 200000 left no room for a 200K model's 64K output reservation)
    and wastes a larger one.
    """

    def test_derives_from_model_window_when_unset(self, mocker) -> None:
        # Sonnet 5: 1M window - the 16K default output cap, less 10% headroom.
        mgr = _make_manager(mocker, answer_model_id=LanguageModelId.CLAUDE_V5_SONNET)
        assert mgr._max_context_tokens == int((1000000 - 16384) * 0.9)

    def test_smaller_window_gets_smaller_budget(self, mocker) -> None:
        mgr = _make_manager(mocker, answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET)
        assert mgr._max_context_tokens == int((200000 - 16384) * 0.9)

    def test_uncapped_output_reserves_the_model_maximum(self, mocker) -> None:
        mgr = _make_manager(
            mocker,
            answer_model_id=LanguageModelId.CLAUDE_V5_SONNET,
            default_max_output_tokens=None,
        )
        assert mgr._max_context_tokens == int((1000000 - 128000) * 0.9)

    def test_derived_budget_leaves_room_for_output(self, mocker) -> None:
        # The invariant the old hardcoded 200000 violated on every 200K model.
        for model_id in (
            LanguageModelId.CLAUDE_V4_5_SONNET,
            LanguageModelId.CLAUDE_V4_5_HAIKU,
            LanguageModelId.CLAUDE_V3_HAIKU,
            LanguageModelId.CLAUDE_V5_SONNET,
        ):
            mgr = _make_manager(mocker, answer_model_id=model_id)
            info = get_language_model_info(model_id)
            output = effective_max_output_tokens(
                info, Config().models.default_max_output_tokens
            )
            assert output <= info.max_output_tokens
            assert (
                mgr._max_context_tokens + output <= info.context_window_size
            ), model_id

    def test_explicit_budget_over_window_is_clamped(self, mocker) -> None:
        mgr = _make_manager(
            mocker,
            max_context_tokens=200000,
            answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
        assert mgr._max_context_tokens == int((200000 - 16384) * 0.9)

    def test_explicit_budget_within_window_is_honoured(self, mocker) -> None:
        mgr = _make_manager(
            mocker,
            max_context_tokens=50000,
            answer_model_id=LanguageModelId.CLAUDE_V5_SONNET,
        )
        assert mgr._max_context_tokens == 50000

    def test_1m_opt_in_widens_budget_for_beta_model(self, mocker) -> None:
        narrow = _make_manager(
            mocker, answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET
        )
        wide = _make_manager(
            mocker,
            answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            enable_1m_context=True,
        )
        assert wide._max_context_tokens > narrow._max_context_tokens
        assert wide._max_context_tokens == int((1000000 - 16384) * 0.9)

    def test_caller_max_tokens_cannot_exceed_model_budget(self, mocker) -> None:
        mgr = _make_manager(mocker, answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET)
        huge = [_r(" ".join(["w"] * 500), 0.9, "text", f"s{i}") for i in range(200)]
        out = mgr.optimize_context(huge, query="q", max_tokens=10_000_000)
        assert out.total_tokens <= mgr._max_context_tokens


class TestCountTokens:
    def test_empty_string_is_zero_without_calling_counter(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr.count_tokens("") == 0

    def test_delegates_to_counter(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr.count_tokens("one two three") == 3


class TestBudgeting:
    def test_budget_exceeded_returns_empty_context(self, mocker) -> None:
        # The query alone (3 tokens) fills the explicit budget -> short circuit.
        mgr = _make_manager(mocker, max_context_tokens=1024)
        results = [_r("a b c d e", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="one two three", max_tokens=3)
        assert out.sections == []
        assert out.sections_included == 0
        assert out.sections_excluded == 1

    def test_small_explicit_budget_is_not_cut_by_the_prompt_buffer(
        self, mocker
    ) -> None:
        # The 512-token prompt buffer used to come off an explicit budget too,
        # so any RAGInput.max_tokens <= 513 produced an empty context.
        mgr = _make_manager(mocker, max_context_tokens=1024)
        results = [_r("a b c d e", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="one two three", max_tokens=100)
        assert out.sections_included == 1

    def test_prompt_buffer_comes_off_the_configured_budget(self, mocker) -> None:
        mgr = _make_manager(mocker, max_context_tokens=1024)
        fits = 1024 - TokenManager.CONTEXT_BUFFER_TOKENS - 1  # minus the 1-token query
        results = [_r(" ".join(["w"] * (fits + 1)), 0.9, "text", "s1")]
        for max_tokens in (None, 5000):
            out = mgr.optimize_context(results, query="q", max_tokens=max_tokens)
            assert out.total_tokens <= fits

    def test_fits_within_budget_includes_section(self, mocker) -> None:
        mgr = _make_manager(mocker, max_context_tokens=200000)
        results = [_r("alpha beta gamma", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.total_tokens == 4  # "[s1] alpha beta gamma"

    def test_max_tokens_enforced_excludes_overflow(self, mocker) -> None:
        # Budget = max_tokens - query(1) - buffer(0); set buffer to 0 for exactness.
        mgr = _make_manager(mocker)
        # Two sections of 5 tokens each; budget allows only one.
        results = [
            _r("a b c d e", 0.9, "text", "s1"),
            _r("f g h i j", 0.8, "text", "s2"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=1 + HEADER + 6)
        # available = 10 - 1 (query) - 3 (header) = 6 tokens -> exactly one
        # 6-token line ("[s1] a b c d e") fits.
        assert out.sections_included == 1
        assert out.total_tokens == 6

    def test_zero_token_sections_skipped(self, mocker) -> None:
        mgr = _make_manager(mocker)
        results = [
            _r("", 0.9, "text", "s1"),  # 0 tokens -> skipped
            _r("real content here", 0.5, "text", "s2"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.sections[0].source_id == "s2"


class TestPriorityOrdering:
    def test_a_window_too_small_for_any_share_still_yields_one_section(
        self, mocker
    ) -> None:
        # Shares divide across present types but the truncation floor does not, so a
        # small window with several present types can drive EVERY type's share below
        # the floor. Returning nothing there would hand the generator an empty
        # context while a whole section still fit; the highest-priority section is
        # seated instead.
        mgr = _make_manager(mocker)
        results = [
            _r("g g g g g", 0.6, "general", "gen"),
            _r("t t t t t", 0.5, "text", "txt"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=1 + 2 * HEADER + 6)
        assert out.sections_included == 1
        assert out.total_tokens <= 6
        # The higher score wins whatever its type: a TEXT weight used to seat
        # the lower-scored text section here.
        assert out.sections[0].source_id == "gen"

    def test_last_resort_score_tie_goes_to_the_earlier_result(self, mocker) -> None:
        # Ranked over the input, not type by type: with the type-grouped ranking
        # the tie went to "r2", whose type happened to appear first.
        mgr = _make_manager(mocker)
        results = [
            _r("r r r r r", 0.4, "relationship", "r1"),
            _r("c c c c c", 0.5, "claim", "c1"),
            _r("s s s s s", 0.5, "relationship", "r2"),
        ]
        headers = 5 + 2  # "##### Knowledge Graph — Relationships", "##### Claims"
        out = mgr.optimize_context(results, query="q", max_tokens=1 + headers + 6)
        assert [s.source_id for s in out.sections] == ["c1"]

    def test_last_resort_section_is_truncated_when_nothing_fits_whole(
        self, mocker
    ) -> None:
        mgr = _make_manager(mocker)
        floor = mgr.config.min_truncated_section_tokens
        out = mgr.optimize_context(
            [_r(" ".join(["t"] * 500), 0.9, "text", "t0")],
            query="q",
            max_tokens=1 + HEADER + floor,
        )
        assert out.sections_included == 1
        assert out.sections[0].metadata["truncated"] is True
        assert out.total_tokens <= floor

    def test_unknown_retriever_type_falls_back_to_general(self, mocker) -> None:
        mgr = _make_manager(mocker)
        # An out-of-enum retriever_type must degrade to GENERAL, not raise
        # ValueError out of SectionType(...) and abort context-building. A future
        # retriever or a malformed result would otherwise crash the whole query.
        section = mgr._create_context_section(
            _r("hello world", 0.5, "some_future_retriever", "s1"), index=0
        )
        assert section.section_type == SectionType.GENERAL
        assert section.priority == pytest.approx(0.5)

    def test_unknown_retriever_type_does_not_crash_optimize_context(
        self, mocker
    ) -> None:
        # End-to-end guard: a result with an out-of-enum type flows through the
        # public optimize_context without raising.
        mgr = _make_manager(mocker)
        results = [_r("alpha beta gamma", 0.9, "mystery_type", "s1")]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.sections[0].section_type == SectionType.GENERAL
        assert out.sections[0].content == "alpha beta gamma"

    def test_claim_retriever_type_maps_to_claim_section(self, mocker) -> None:
        mgr = _make_manager(mocker)
        section = mgr._create_context_section(
            _r("a claim", 0.5, "claim", "c1"), index=0
        )
        assert section.section_type == SectionType.CLAIM
        assert section.priority == pytest.approx(0.5)

    def test_zero_score_stays_the_lowest_priority(self, mocker) -> None:
        mgr = _make_manager(mocker)
        result = RetrievalResult(
            content="x y", score=0.0, retriever_type="entity", source="s"
        )
        # `score or 0.5` turned a 0.0 score into a mid-ranked 0.5.
        section = mgr._create_context_section(result, index=0)
        assert section.priority == 0.0


def _tokens_by_type(out) -> dict[SectionType, int]:  # noqa: ANN001
    by_type: dict[SectionType, int] = {}
    for section in out.sections:
        by_type[section.section_type] = (
            by_type.get(section.section_type, 0) + section.token_count
        )
    return by_type


class TestPerTypeBudgetIsAHardCap:
    """A type's share CAPS it while another type still has candidates.

    An earlier implementation pooled every unused token and offered it to the
    leftovers of ANY type by raw priority, so whichever type had the
    highest-scoring leftovers absorbed the whole remainder and evicted the
    others. Budget a type leaves unused now goes only to types that still have
    candidates, by share (or to the ``spare_budget_first`` types first).
    """

    def test_one_type_cannot_exceed_its_share_when_another_type_has_candidates(
        self, mocker
    ) -> None:
        mgr = _make_manager(mocker)
        # available = 1000 - 1(query) - 0(buffer) = 1000 tokens.
        # TEXT prop 0.50 / COMMUNITY prop 0.10 -> renormalized over the two present
        # types: TEXT 5/6 (833), COMMUNITY 1/6 (166).
        # 40 text sections of 100 tokens each want 4000 tokens; they must not eat
        # the community share even though every TEXT section outscores them.
        text_results = [
            _r(" ".join(["t"] * 100), 0.9, "text", f"t{i}") for i in range(40)
        ]
        community_results = [
            _r(" ".join(["c"] * 100), 0.4, "community", f"c{i}") for i in range(5)
        ]
        out = mgr.optimize_context(
            text_results + community_results,
            query="q",
            max_tokens=1001,
        )
        by_type: dict[SectionType, int] = {}
        for section in out.sections:
            by_type[section.section_type] = (
                by_type.get(section.section_type, 0) + section.token_count
            )
        assert (
            by_type.get(SectionType.COMMUNITY, 0) > 0
        ), "a lower-priority type with candidates must keep its share"
        assert by_type[SectionType.TEXT] <= 850
        assert out.total_tokens <= 1000

    def test_absent_type_share_is_renormalized_not_wasted(self, mocker) -> None:
        # With ONLY text candidates, the props renormalize to 1.0 for TEXT, so the
        # whole window is usable — a hard cap must not mean a wasted window.
        # Renormalization over PRESENT types is static (it does not look at how many
        # candidates a type has), which is what keeps it from degenerating into the
        # leftover-pooling this fix removes.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["t"] * 100), 0.9, "text", f"t{i}") for i in range(20)]
        out = mgr.optimize_context(results, query="q", max_tokens=1 + HEADER + 1000)
        assert out.total_tokens == 1000

    def test_unused_share_of_a_sparse_type_goes_to_types_with_candidates(
        self, mocker
    ) -> None:
        # TEXT is sparse (2 candidates); COMMUNITY has plenty. The rest of
        # TEXT's share goes to COMMUNITY once TEXT has nothing left to seat:
        # a hard cap left the window mostly empty (27-34% unused in the
        # measured local/mix shapes) while community reports were being cut.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["t"] * 50), 0.9, "text", f"t{i}") for i in range(2)] + [
            _r(" ".join(["c"] * 50), 0.4, "community", f"c{i}") for i in range(40)
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=1001)
        by_type = _tokens_by_type(out)
        # Budget 1001 - 1 (query) - 2 headers = 994; TEXT seats both chunks.
        assert by_type[SectionType.TEXT] == 102
        # Renormalized COMMUNITY share alone would be 165 (three sections).
        assert by_type[SectionType.COMMUNITY] == 867  # 17 whole sections
        assert 994 - out.total_tokens < mgr.config.min_truncated_section_tokens

    def test_spare_budget_goes_by_share_and_never_by_score(self, mocker) -> None:
        # ENTITY is sparse; TEXT and COMMUNITY both still have candidates. The
        # spare is split by their shares (TEXT 0.50 : COMMUNITY 0.10), not won
        # by whichever type scores higher: COMMUNITY outscores TEXT here.
        mgr = _make_manager(mocker)
        out = mgr.optimize_context(self._three_types(), query="q", max_tokens=1001)
        by_type = _tokens_by_type(out)
        assert by_type[SectionType.ENTITY] == 50  # no "[e0] " prefix
        # First-pass caps over 991 tokens: TEXT 619, COMMUNITY 123.
        assert 619 < by_type[SectionType.TEXT]
        assert 123 < by_type[SectionType.COMMUNITY] < by_type[SectionType.TEXT] / 3

    def test_spare_budget_first_types_take_the_spare_before_the_others(
        self, mocker
    ) -> None:
        # LightRAG gives chunks what the KG sections leave unused
        # (available_chunk_tokens); the other types keep their own share.
        mgr = _make_manager(mocker)
        out = mgr.optimize_context(
            self._three_types(),
            query="q",
            max_tokens=1001,
            spare_budget_first=(SectionType.TEXT,),
        )
        by_type = _tokens_by_type(out)
        assert by_type[SectionType.ENTITY] == 50  # no "[e0] " prefix
        assert by_type[SectionType.COMMUNITY] <= 123  # its first-pass cap
        assert by_type[SectionType.TEXT] > 750
        assert 991 - out.total_tokens < mgr.config.min_truncated_section_tokens

    @staticmethod
    def _three_types() -> list[RetrievalResult]:
        return (
            [_r(" ".join(["t"] * 50), 0.5, "text", f"t{i}") for i in range(40)]
            + [_r(" ".join(["e"] * 50), 0.9, "entity", "e0")]
            + [_r(" ".join(["c"] * 50), 0.9, "community", f"c{i}") for i in range(40)]
        )

    def test_general_only_candidates_still_get_a_budget(self, mocker) -> None:
        # Global search emits ONLY GENERAL sections, so renormalizing over present
        # types must not produce a zero budget and an empty context for that whole
        # strategy.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["g"] * 50), 0.9, "general", f"g{i}") for i in range(10)]
        out = mgr.optimize_context(results, query="q", max_tokens=1001)
        assert out.sections_included == 10
        assert out.total_tokens == 510  # ten 50-token sections + prefixes

    def test_general_mixed_with_another_type_gets_a_nonzero_budget(
        self, mocker
    ) -> None:
        # Regression: GENERAL used to carry a hardcoded 0.0 share, so as soon as it
        # co-occurred with any other type its renormalized budget was 0 and the
        # section was dropped. That silently discarded global search's map-reduce
        # synthesis and drift's primer answer whenever a run also carried community
        # or text sections.
        mgr = _make_manager(mocker)
        budgets = mgr._type_budgets([SectionType.GENERAL, SectionType.COMMUNITY], 600)
        assert budgets[SectionType.GENERAL] > 0
        assert budgets[SectionType.COMMUNITY] > 0

        results = [_r(" ".join(["g"] * 20), 0.9, "general", "g0")] + [
            _r(" ".join(["c"] * 20), 0.9, "community", "c0")
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=1001)
        assert {s.section_type for s in out.sections} == {
            SectionType.GENERAL,
            SectionType.COMMUNITY,
        }

    def test_type_budgets_renormalize_over_present_types(self, mocker) -> None:
        mgr = _make_manager(mocker)
        budgets = mgr._type_budgets([SectionType.TEXT, SectionType.COMMUNITY], 600)
        # 0.50 : 0.10 -> 5/6 : 1/6 of 600.
        assert budgets[SectionType.TEXT] == 500
        assert budgets[SectionType.COMMUNITY] == 100

    def test_type_budgets_track_reconfigured_shares(self, mocker) -> None:
        # The shares are config, not constants: a deployment that rebalances them
        # must move the budgets.
        mgr = _make_manager(mocker)
        mgr.config.type_budgets.text = 0.10
        mgr.config.type_budgets.community = 0.30
        budgets = mgr._type_budgets([SectionType.TEXT, SectionType.COMMUNITY], 600)
        assert budgets[SectionType.TEXT] == 150
        assert budgets[SectionType.COMMUNITY] == 450

    def test_type_budgets_equal_split_when_present_shares_are_all_zero(
        self, mocker
    ) -> None:
        # A config may zero out the share of a type that still shows up in a
        # result set; splitting evenly beats emitting an empty context.
        mgr = _make_manager(mocker)
        mgr.config.type_budgets.general = 0.0
        mgr.config.type_budgets.claim = 0.0
        budgets = mgr._type_budgets([SectionType.GENERAL, SectionType.CLAIM], 600)
        assert budgets[SectionType.GENERAL] == 300
        assert budgets[SectionType.CLAIM] == 300

    def test_oversized_section_is_truncated_to_fit_its_share(self, mocker) -> None:
        # A single retrieved item can exceed its whole type share (a community report
        # routinely exceeds a 10% cap). Dropping the type outright cost 5 gold
        # contexts and shrank the window 29k -> 8.8k chars, so the first overflowing
        # section is truncated to fit instead.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["t"] * 5000), 0.9, "text", "t0")]
        out = mgr.optimize_context(results, query="q", max_tokens=1001)
        assert out.sections_included == 1
        section = out.sections[0]
        assert section.metadata["truncated"] is True
        assert section.content.endswith("…")
        assert section.token_count <= 1000
        assert out.total_tokens <= 1000

    def test_truncation_floor_drops_a_section_too_small_to_be_evidence(
        self, mocker
    ) -> None:
        # A share of a handful of tokens carries no usable evidence; emit nothing
        # rather than a meaningless fragment.
        mgr = _make_manager(mocker)
        section = mgr._truncate_to_tokens(
            ContextSection(
                content="a b c d e f",
                token_count=6,
                priority=1.0,
                section_type=SectionType.TEXT,
                source_id="s",
            ),
            max_tokens=mgr.config.min_truncated_section_tokens - 1,
        )
        assert section is None

    def test_truncation_preserves_a_prefix_on_a_word_boundary(self, mocker) -> None:
        original = ContextSection(
            content=" ".join(f"w{i}" for i in range(1000)),
            token_count=1000,
            priority=1.0,
            section_type=SectionType.COMMUNITY,
            source_id="c0",
        )
        mgr = _make_manager(mocker)
        section = mgr._truncate_to_tokens(original, max_tokens=100)
        assert section is not None
        assert section.content.startswith("w0 w1 w2")
        assert "  " not in section.content
        assert len(section.content) < len(original.content)
        # The original is not mutated (sections are shared across the pipeline).
        assert original.token_count == 1000
        assert not original.metadata

    @pytest.mark.parametrize(
        "content",
        [
            # One early space, then text written without spaces: backing off to
            # that space kept 2 of ~400 characters.
            "契約 " + "東京倉庫は大阪商事のために部品を保管する。" * 20,
            "供应商" + "与买方签订合同并约定付款条件" * 30,
        ],
        ids=["japanese-one-early-space", "chinese-no-space"],
    )
    def test_cjk_truncation_keeps_most_of_the_room(self, mocker, content) -> None:
        original = ContextSection(
            content=content,
            token_count=len(content),
            priority=1.0,
            section_type=SectionType.TEXT,
            source_id="s",
        )
        mgr = _make_manager(mocker)
        room = len(content) // 2
        section = mgr._truncate_to_tokens(original, max_tokens=room)
        assert section is not None
        kept = section.content.removesuffix(" …")
        assert content.startswith(kept)
        assert len(kept) >= room * 0.8
        if "。" in content:
            assert kept.endswith("。")  # a sentence mark within reach is used

    def test_a_type_whose_first_candidate_overflows_is_still_represented(
        self, mocker
    ) -> None:
        # The regression this guards: COMMUNITY's share was 10% of the window and
        # every single report was larger than that, so the whole section vanished
        # (COMM share 0.676 -> 0.0).
        mgr = _make_manager(mocker)
        results = [
            _r(" ".join(["t"] * 100), 0.9, "text", f"t{i}") for i in range(20)
        ] + [_r(" ".join(["c"] * 900), 0.4, "community", "c0")]
        out = mgr.optimize_context(results, query="q", max_tokens=1001)
        types = {s.section_type for s in out.sections}
        assert SectionType.COMMUNITY in types
        assert SectionType.TEXT in types
        assert out.total_tokens <= 1000


def _synthesized(tokens: int) -> RetrievalResult:
    return RetrievalResult(
        content=" ".join(["k"] * tokens),
        score=1.0,
        retriever_type="general",
        source="synthesized_key_points",
        metadata={"synthesized": True},
    )


def _tokens_by_source(out: OptimizedContext) -> dict[str, int]:
    return {s.source_id: s.token_count for s in out.sections}


class TestSynthesizedSectionReservation:
    """Global search's map key points are seated before the per-type split.

    They condense every selected report and are already sized by
    ``max_map_reduce_tokens``; as one GENERAL section they got the GENERAL share
    (0.10 renormalized over the present types) and were cut to a fraction of
    their size while most of the window stayed unused.
    """

    def _evidence(self) -> list[RetrievalResult]:
        return [_r(" ".join(["t"] * 100), 0.9, "text", f"t{i}") for i in range(40)] + [
            _r(" ".join(["c"] * 100), 0.4, "community", f"c{i}") for i in range(5)
        ]

    def test_key_points_are_kept_whole(self, mocker) -> None:
        mgr = _make_manager(mocker)
        out = mgr.optimize_context(
            [_synthesized(400), *self._evidence()],
            query="q",
            max_tokens=1 + 3 * HEADER + 1000,
        )
        tokens = _tokens_by_source(out)
        # Before: GENERAL 0.10 of {TEXT 0.50, COMMUNITY 0.10, GENERAL 0.10} of
        # 1000 = 142 tokens of the 400.
        assert tokens["synthesized_key_points"] == 401  # whole, with its prefix
        # The other 599 tokens keep the per-type split: TEXT 5/6, COMMUNITY 1/6.
        by_type: dict[SectionType, int] = {}
        for section in out.sections:
            by_type[section.section_type] = (
                by_type.get(section.section_type, 0) + section.token_count
            )
        assert by_type[SectionType.TEXT] == 599 * 5 // 6
        assert 0 < by_type[SectionType.COMMUNITY] <= 599 // 6
        assert out.total_tokens <= 1000

    def test_key_points_larger_than_the_window_are_truncated(self, mocker) -> None:
        mgr = _make_manager(mocker)
        out = mgr.optimize_context(
            [_synthesized(2000), *self._evidence()],
            query="q",
            max_tokens=1 + 3 * HEADER + 1000,
        )
        assert [s.source_id for s in out.sections] == ["synthesized_key_points"]
        assert out.total_tokens == 1000
        assert out.sections[0].metadata["truncated"] is True

    def test_without_synthesized_sections_the_split_is_unchanged(self, mocker) -> None:
        mgr = _make_manager(mocker)
        plain = mgr.optimize_context(self._evidence(), query="q", max_tokens=1001)
        assert (
            sum(
                s.token_count
                for s in plain.sections
                if s.section_type == SectionType.TEXT
            )
            == 808
        )  # TEXT's 828-token share seats 8 whole 101-token lines


@pytest.mark.parametrize("max_tokens", [101, 200, 600])
@pytest.mark.parametrize(
    "kinds",
    [("text",), ("text", "entity", "relationship", "community", "claim")],
    ids=["one-type", "five-types"],
)
def test_the_rendered_context_stays_within_max_tokens(
    mocker, max_tokens: int, kinds: tuple[str, ...]
) -> None:
    # Section headers and "[source_id]" prefixes were not counted: a 600-token
    # budget rendered a 690-token context.
    mgr = _make_manager(mocker)
    results = [
        _r(" ".join(["w"] * 10), 0.9 - i / 100, kind, f"{kind}-{i}")
        for i in range(60)
        for kind in kinds
    ]
    out = mgr.optimize_context(results, query="the query", max_tokens=max_tokens)
    rendered = mgr.build_context_string(out)
    assert out.sections_included > 0
    assert mgr.count_tokens(rendered) + mgr.count_tokens("the query") <= max_tokens


class TestBuildContextString:
    def test_empty_context_fallback_message(self) -> None:
        empty = OptimizedContext(
            sections=[],
            total_tokens=0,
            sections_included=0,
            sections_excluded=0,
        )
        assert (
            TokenManager.build_context_string(empty) == "No relevant information found."
        )

    def test_string_contains_headers_and_separator(self, mocker) -> None:
        mgr = _make_manager(mocker)
        results = [
            _r("first chunk text", 0.9, "text", "src-a"),
            _r("second chunk text", 0.8, "entity", "src-b"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        rendered = TokenManager.build_context_string(out)
        # The rendering groups by type: sections are now
        # GROUPED under labeled type headers (mirroring LightRAG's
        # Entities/Relationships/Chunks blocks and MS's sectioned tables) instead of
        # a flat per-item "Source Type / Source ID / Priority" dump. The meaningless
        # Priority float was dropped; ids are rendered as a citable "[id]" prefix.
        assert "##### Document Chunks" in rendered
        assert "##### Knowledge Graph — Entities" in rendered
        assert "[src-a]" in rendered
        assert "first chunk text" in rendered

    def test_only_kg_lines_go_without_their_id(self, mocker) -> None:
        # Nothing reads an entity's or relationship's id in the context (the
        # answer cites passages; sources map back by section), and a UUID
        # prefix was about a quarter of the KG sections' tokens. Passages keep
        # theirs.
        mgr = _make_manager(mocker)
        results = [
            _r("a passage", 0.9, "text", "unit-1"),
            _r("Title: Vendor", 0.9, "entity", "entity-1"),
            _r("Relationship: Vendor -> Buyer", 0.9, "relationship", "rel-1"),
            _r("Claim: Vendor -> Buyer", 0.9, "claim", "claim-1"),
            _r("A report", 0.9, "community", "report-1"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        lines = TokenManager.build_context_string(out).splitlines()
        assert "Title: Vendor" in lines
        assert "Relationship: Vendor -> Buyer" in lines
        assert "entity-1" not in "\n".join(lines)
        assert "rel-1" not in "\n".join(lines)
        assert {"[unit-1] a passage", "[claim-1] Claim: Vendor -> Buyer"} <= set(lines)
        assert "[report-1] A report" in lines
        # Counted as rendered: the entity line costs what its content does.
        entity = next(s for s in out.sections if s.source_id == "entity-1")
        assert entity.token_count == mgr.count_tokens("Title: Vendor")
        assert TokenManager.ranked_section_lines(out)[1:3] == [
            "Title: Vendor",
            "Relationship: Vendor -> Buyer",
        ]

    def test_ranked_section_lines_put_the_highest_priority_first(self) -> None:
        # A grounding check over its size limit keeps the head of this list.
        def section(source_id: str, priority: float) -> ContextSection:
            return ContextSection(
                content=f"{source_id} text",
                token_count=1,
                priority=priority,
                section_type=SectionType.TEXT,
                source_id=source_id,
            )

        context = OptimizedContext(
            sections=[section("low", 0.1), section("high", 0.9), section("mid", 0.5)],
            total_tokens=3,
            sections_included=3,
            sections_excluded=0,
        )
        assert TokenManager.ranked_section_lines(context) == [
            "[high] high text",
            "[mid] mid text",
            "[low] low text",
        ]


class TestContextTypeBudgetConfig:
    def test_all_zero_shares_are_rejected_at_config_load(self) -> None:
        # The renormalization fallback in _type_budgets only covers the types
        # PRESENT in one result set; a config with every share at zero expresses
        # "never emit a context", which is never what an operator means.
        with pytest.raises(ValidationError):
            ContextTypeBudgetConfig(
                text=0.0,
                entity=0.0,
                relationship=0.0,
                community=0.0,
                claim=0.0,
                general=0.0,
            )

    def test_negative_shares_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextTypeBudgetConfig(text=-0.1)

    def test_shares_need_not_sum_to_one(self) -> None:
        # Shares are renormalized over present types, so they are ratios, not a
        # partition — over- or under-summing is legitimate.
        budget = ContextTypeBudgetConfig(text=2.0, entity=1.0)
        assert budget.text == pytest.approx(2.0)

    def test_every_section_type_maps_to_a_budget_field(self) -> None:
        # A new SectionType with no share field would silently fall back to 0.0 and
        # be dropped whenever it co-occurred with another type — the exact defect
        # GENERAL's hardcoded 0.0 caused.
        budget = ContextTypeBudgetConfig()
        for section_type in SectionType:
            assert budget.share(section_type) == getattr(budget, section_type.value)


def test_default_config_budget_is_fixed_not_window_derived(mocker) -> None:
    # The shipped default binds (30K, upstream LightRAG parity) instead of the
    # ~785K a 1M-window derivation yields; it stays clamped to the model.
    config = Config()
    providers = Providers(
        config,
        boto_session=mocker.Mock(),
        token_counter_factory=lambda *_, **__: mocker.Mock(),
    )
    assert config.search.token_manager.max_context_tokens == 30_000
    assert TokenManager(config, providers=providers)._max_context_tokens == 30_000


class _SlowCounter:
    """Exact word-count counter that takes time per call, like CountTokens."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        with self._lock:
            self.calls.append(text)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return len(text.split())

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        return text, len(text.split())


def _manager_with_counter(
    mocker,
    counter: _SlowCounter,
    answer_model_id: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET,
) -> TokenManager:
    # Default to an answer model that accepts CountTokens, where each count is
    # a round trip worth overlapping.
    config = Config()
    config.models.roles.answer_generation = answer_model_id
    providers = Providers(
        config,
        boto_session=mocker.Mock(),
        token_counter_factory=lambda *_, **__: counter,
    )
    return TokenManager(config, providers=providers)


class TestConcurrentCounting:
    def test_optimize_context_counts_sections_concurrently_and_exactly(
        self, mocker
    ) -> None:
        counter = _SlowCounter()
        mgr = _manager_with_counter(mocker, counter)
        results = [
            _r(" ".join(["w"] * (i + 1)), 0.5, "text", f"s{i}") for i in range(12)
        ]
        # A duplicate section is counted once.
        results.append(_r("w w w", 0.5, "text", "s2"))

        out = mgr.optimize_context(results, query="the query", max_tokens=10_000)

        assert counter.max_active > 1
        # Each section is counted as rendered, "[source_id] " prefix included.
        assert sorted(counter.calls) == sorted(
            {f"[{r.source}] {r.content}" for r in results}
            | {"the query", "##### Document Chunks"}
        )
        by_source = {s.source_id: s.token_count for s in out.sections}
        assert by_source["s0"] == 2 and by_source["s11"] == 13

    def test_count_tokens_many_is_bounded(self, mocker) -> None:
        counter = _SlowCounter(delay=0.01)
        mgr = _manager_with_counter(mocker, counter)
        texts = [f"text {i}" for i in range(40)]
        assert mgr.count_tokens_many(texts) == [2] * 40
        assert 1 < counter.max_active <= TokenManager.COUNT_CONCURRENCY

    def test_count_tokens_many_counts_inline_without_count_tokens_api(
        self, mocker
    ) -> None:
        # The default answer model has no CountTokens: every count is a local
        # CPU-bound estimate, so a thread pool only adds overhead.
        counter = _SlowCounter(delay=0)
        threads: set[str] = set()
        count = counter.count_tokens

        def recording_count(text: str) -> int:
            threads.add(threading.current_thread().name)
            return count(text)

        counter.count_tokens = recording_count  # type: ignore[method-assign]
        mgr = _manager_with_counter(
            mocker,
            counter,
            answer_model_id=Config().model_id(ModelRole.ANSWER_GENERATION),
        )
        assert mgr.count_tokens_many([f"text {i}" for i in range(20)]) == [2] * 20
        assert threads == {threading.current_thread().name}

    def test_count_tokens_many_follows_the_counter_not_the_model_record(
        self, mocker
    ) -> None:
        # The answer model accepts CountTokens, but the counter says its counts
        # are local now (the Bedrock one does once the model rejected the API):
        # the counter decides, so counting stays inline.
        counter = _SlowCounter(delay=0)
        counter.counts_via_api = False  # type: ignore[attr-defined]
        threads: set[str] = set()
        count = counter.count_tokens

        def recording_count(text: str) -> int:
            threads.add(threading.current_thread().name)
            return count(text)

        counter.count_tokens = recording_count  # type: ignore[method-assign]
        mgr = _manager_with_counter(mocker, counter)
        assert mgr.count_tokens_many([f"text {i}" for i in range(20)]) == [2] * 20
        assert threads == {threading.current_thread().name}

    def test_count_tokens_many_skips_empty_text(self, mocker) -> None:
        counter = _SlowCounter(delay=0)
        mgr = _manager_with_counter(mocker, counter)
        assert mgr.count_tokens_many(["", "a b"]) == [0, 2]
        assert counter.calls == ["a b"]


def test_context_items_are_separated_by_a_blank_line(mocker) -> None:
    # Entity and relationship items span several lines and carry no "[id]"
    # prefix; a single newline between them ran one into the next.
    mgr = _make_manager(mocker)
    results = [
        _r("Entity: Vendor\nDescription: sells", 0.9, "entity", "e-1"),
        _r("Entity: Buyer\nDescription: buys", 0.8, "entity", "e-2"),
    ]
    out = mgr.optimize_context(results, query="q", max_tokens=10000)
    rendered = TokenManager.build_context_string(out)
    assert "Description: sells\n\nEntity: Buyer" in rendered


# --- an empty context the budget caused ----------------------------------------


def _mixed_results() -> list[RetrievalResult]:
    long = " ".join(["word"] * 30)
    return [
        _r("Vendor ships widgets", 1.0, "text", "tu-1"),
        _r(long, 0.5, "entity", "e-1"),
        _r(long, 0.5, "relationship", "r-1"),
        _r(long, 0.5, "community", "c-1"),
    ]


def test_headers_are_paid_for_the_types_seated_only(mocker) -> None:
    # Regression (rv23): a header was reserved for every type present, so a
    # small budget (max_tokens=40 on local) went to four headers and nothing
    # was seated, though the best passage fits beside its own header.
    manager = _make_manager(mocker, max_context_tokens=100_000)
    # 1 query token + the text header (3) + the passage with its prefix (4).
    optimized = manager.optimize_context(_mixed_results(), "q", max_tokens=9)
    assert [s.source_id for s in optimized.sections] == ["tu-1"]
    context = manager.build_context_string(optimized)
    assert len(context.split()) <= 9 - 1


@pytest.mark.parametrize("max_tokens", range(20, 1200, 7))
def test_a_type_that_seats_nothing_never_lends_its_share_twice(
    mocker, max_tokens: int
) -> None:
    # Regression (rv34): a type whose share was below the truncation floor
    # seated nothing yet kept its share when the spare was handed out, so
    # its share was counted twice: a 244-token budget rendered 262 tokens,
    # and a DRIFT action read past the input tokens planned for it.
    mgr = _make_manager(mocker)
    results = [
        _r(" ".join(["w"] * 30), 0.9, "relationship", "r-1"),
        _r(" ".join(["w"] * 120), 0.8, "entity", "e-1"),
        _r(" ".join(["w"] * 150), 0.7, "text", "t-1"),
        _r(" ".join(["w"] * 160), 0.6, "text", "t-2"),
        _r(" ".join(["w"] * 200), 0.5, "community", "c-1"),
    ]
    out = mgr.optimize_context(results, query="the query", max_tokens=max_tokens)
    rendered = mgr.build_context_string(out)
    assert mgr.count_tokens(rendered) + mgr.count_tokens("the query") <= max_tokens


@pytest.mark.parametrize("max_tokens", range(150, 3000, 37))
@pytest.mark.parametrize("words", [40, 70, 110])
def test_room_a_type_cannot_seat_goes_to_the_others(
    mocker, max_tokens: int, words: int
) -> None:
    # Regression (rv35): handing each share out once, a type that still had
    # candidates kept the room it could not seat (below the truncation
    # floor); with every type such a receiver nothing was left to hand out
    # and a 592-token budget rendered 372 tokens.
    mgr = _make_manager(mocker)
    floor = mgr.config.min_truncated_section_tokens
    kinds = ("text", "entity", "relationship", "community")
    results = [
        _r(" ".join(["w"] * (words + 9 * (i % 3))), 0.9 - i / 100, kind, f"{kind}-{i}")
        for i in range(40)
        for kind in kinds
    ]
    out = mgr.optimize_context(results, query="the query", max_tokens=max_tokens)
    rendered = mgr.count_tokens(mgr.build_context_string(out))
    assert out.sections_excluded > 0
    assert rendered + mgr.count_tokens("the query") <= max_tokens
    # What is left of the budget past the headers (each type's reserved)
    # could seat nothing.
    budget = (
        max_tokens
        - mgr.count_tokens("the query")
        - sum(
            mgr.count_tokens(TokenManager._section_header(SectionType(kind)))
            for kind in kinds
        )
    )
    assert budget - out.total_tokens < floor


def _packed_tokens(mgr: TokenManager, out: OptimizedContext) -> int:
    """What ``out``'s sections and the headers of their types count."""
    return out.total_tokens + sum(
        mgr.count_tokens(TokenManager._section_header(t))
        for t in {s.section_type for s in out.sections}
    )


class _V030Lending(TokenManager):
    """0.3.0's hand-out (each round's spare lent on top of every
    receiver's cap), the oracle the share rule keeps to where it fitted."""

    def _select_per_type(  # type: ignore[override]
        self, sections, token_budget, spare_budget_first=(), fallback=True, headers=None
    ):
        by_type: dict[SectionType, list[ContextSection]] = {}
        for s in sections:
            if s.token_count > 0:
                by_type.setdefault(s.section_type, []).append(s)
        for lst in by_type.values():
            lst.sort(key=lambda s: s.priority, reverse=True)
        shares = self._type_budget_shares(list(by_type))
        caps = self._type_budgets(list(by_type), token_budget)
        for _ in range(len(by_type) + 1):
            packed = {t: self._pack_type(lst, caps[t]) for t, lst in by_type.items()}
            spare = token_budget - sum(
                s.token_count for seated, _ in packed.values() for s in seated
            )
            receivers = [t for t, (_, done) in packed.items() if not done]
            if spare < self.config.min_truncated_section_tokens or not receivers:
                break
            total = sum(shares[t] for t in receivers)
            for t in receivers:
                caps[t] += int(
                    spare * (shares[t] / total if total > 0 else 1 / len(receivers))
                )
        return [s for seated, _ in packed.values() for s in seated]


@pytest.mark.parametrize("seed", range(40))
def test_the_share_rule_keeps_0_3_0s_contexts_where_they_fitted(
    mocker, seed: int
) -> None:
    # Regression (rv35): handing each share out once changed contexts
    # 0.3.0 built within the budget (and underfilled them). The spare is
    # lent as 0.3.0 lent it, and only a round that then seats past the
    # budget lends it once.
    import copy
    import random

    rng = random.Random(seed)
    mgr = _make_manager(mocker)
    oracle = copy.copy(mgr)
    oracle.__class__ = _V030Lending
    kept = 0
    for _ in range(25):
        results = [
            _r(" ".join(["w"] * rng.randint(5, 200)), rng.random(), kind, f"{kind}{i}")
            for kind in ("text", "entity", "relationship", "community")
            for i in range(rng.randint(1, 6))
        ]
        max_tokens = rng.randint(80, 900)
        out = mgr.optimize_context(results, "q", max_tokens=max_tokens)
        assert _packed_tokens(mgr, out) + 1 <= max_tokens
        before = oracle.optimize_context(results, "q", max_tokens=max_tokens)
        if before.sections and _packed_tokens(oracle, before) + 1 <= max_tokens:
            kept += 1
            assert [(s.source_id, s.content) for s in out.sections] == [
                (s.source_id, s.content) for s in before.sections
            ]
    assert kept


def _local_manager() -> TokenManager:
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )

    config = Config()
    return TokenManager(
        config,
        providers=Providers(
            config, token_counter_factory=local_token_counter_factory(config)
        ),
    )


@pytest.mark.parametrize(
    "content", ["Vendor {i} sells to Buyer {i}x.", "벤더 {i} 공급"]
)
@pytest.mark.parametrize("kind", ["text", "entity"])
def test_a_context_bounded_at_the_estimate_never_passes_it(
    content: str, kind: str
) -> None:
    # Regression (rv34): each section is counted apart, rounded down, and
    # the blank lines between them are not counted, so a context of many
    # short sections estimated up to 16% past its budget: past the room an
    # input-token limit left a call (the usage meter estimates the whole
    # prompt) and past the answer's headroom.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count

    mgr = _local_manager()
    results = [_r(content.format(i=i), 1 - i / 1000, kind, f"d{i}") for i in range(400)]
    overshot = []
    for budget in range(300, 6000, 211):
        out = mgr.optimize_context(
            results, "q", max_tokens=budget, estimated_within=budget - 1
        )
        built = estimate_token_count(mgr.build_context_string(out))
        assert out.sections_included > 0
        assert built <= budget - 1
        unbounded = mgr.optimize_context(results, "q", max_tokens=budget)
        overshot.append(
            estimate_token_count(mgr.build_context_string(unbounded)) > budget - 1
        )
    # The bound is what holds it: unbounded, the same contexts pass it.
    assert any(overshot)


def _ratio_manager(mocker, ratio: float) -> TokenManager:
    """A manager whose counter counts ``ratio`` times the local estimate."""
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count

    counter = mocker.Mock()
    counter.count_tokens.side_effect = lambda text: max(
        1, int(estimate_token_count(text) * ratio)
    )
    config = Config()
    return TokenManager(
        config,
        providers=Providers(
            config,
            boto_session=mocker.Mock(),
            token_counter_factory=lambda *_, **__: counter,
        ),
    )


@pytest.mark.parametrize("ratio", [0.35, 0.5, 0.55, 0.7, 1.0, 1.37, 1.8])
@pytest.mark.parametrize(
    "line",
    [
        "공급업체는 공급 계약에 따라 구매자에게 라이다 장치를 납품한다. ",
        "供給業者は供給契約に基づき購入者にライダー装置を納入する。",
    ],
    ids=["ko", "ja"],
)
def test_a_context_bounded_at_the_estimate_fills_it_at_any_counter_ratio(
    mocker, ratio: float, line: str
) -> None:
    # Regression (rv35): the re-pack took the excess, in the estimate's
    # tokens, off a budget in the counter's: with a counter counting half
    # the estimate the budget went below zero and the context came back
    # empty. Packed from the query's own budget, a counter counting more
    # than the estimate fills the bound too.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count

    mgr = _ratio_manager(mocker, ratio)
    results = [
        _r(
            line * (1 + i % 5),
            1 - i / 100,
            ("entity", "relationship", "text")[i % 3],
            f"s{i}",
        )
        for i in range(60)
    ]
    for within in (1000, 4000):
        out = mgr.optimize_context(
            results,
            "q",
            max_tokens=30_000,
            number_sections=True,
            estimated_within=within,
        )
        built = estimate_token_count(mgr.build_context_string(out))
        assert 0.85 * within <= built <= within, (within, built)


@pytest.mark.parametrize("within", [16708, 16709, 19833])
def test_a_context_cut_at_the_same_break_still_fits_the_estimate(
    mocker, within: int
) -> None:
    # Regression (rv36): a section cut to fit is counted at its room, but
    # its text ends at the last break before it, which Japanese has few
    # of: cutting the budget by the few tokens the context was over left
    # it as it was, and after the last re-pack it came back empty (the
    # answer abstained at query_limit, a DRIFT action read nothing).
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count

    mgr = _ratio_manager(mocker, 1.0)
    line = "供給業者は供給契約に基づき購入者にライダー装置を納入し、遅延時には違約金が課される。"
    kinds = ("entity", "relationship", "text", "community")
    results = [
        _r((line * 60)[: 400 + (i * 937) % 2600], 1 - i / 100, kinds[i % 4], f"s{i}")
        for i in range(30)
    ]
    out = mgr.optimize_context(results, "q", estimated_within=within)
    built = estimate_token_count(mgr.build_context_string(out))
    assert out.sections
    assert 0.9 * within <= built <= within, built


def test_a_section_cut_to_fit_is_counted_truncated(mocker) -> None:
    mgr = _make_manager(mocker)
    results = [
        _r(" ".join(["w"] * 100), 0.9, "text", "t-1"),
        _r(" ".join(["w"] * 100), 0.8, "text", "t-2"),
    ]
    whole = mgr.optimize_context(results, query="q", max_tokens=1000)
    assert (whole.sections_excluded, whole.sections_truncated) == (0, 0)
    cut = mgr.optimize_context(results, query="q", max_tokens=180)
    assert (cut.sections_included, cut.sections_excluded) == (2, 0)
    assert cut.sections_truncated == 1


@pytest.mark.parametrize("max_tokens", range(100, 400, 10))
def test_a_spare_no_part_of_which_seats_goes_to_each_type_in_turn(
    mocker, max_tokens: int
) -> None:
    # Regression (rv36): where 0.3.0 packed past the budget, a spare whose
    # parts each fell below the truncation floor went whole to the type of
    # the largest share, which could seat none of it, and the room of the
    # headers left unseated was never lent: a Japanese question's 400-token
    # budget seated 59 of its 133 tokens though a 58-token entity fitted.
    # The spare is now offered to each type in turn, largest share first.
    import copy

    mgr = _make_manager(mocker)
    oracle = copy.copy(mgr)
    oracle.__class__ = _V030Lending
    words = {
        "text": [53, 97, 90, 158],
        "community": [245],
        "summary": [150],
        "entity": [58, 101],
        "relationship": [66, 71],
    }
    results = [
        _r(" ".join(["w"] * n), 0.9 - i / 10, kind, f"{kind}{i}")
        for kind, counts in words.items()
        for i, n in enumerate(counts)
    ]
    out = mgr.optimize_context(results, "q", max_tokens=max_tokens)
    left = max_tokens - 1 - _packed_tokens(mgr, out)
    assert left >= 0
    if _packed_tokens(oracle, oracle.optimize_context(results, "q", max_tokens)) < (
        max_tokens
    ):
        return  # 0.3.0's context, kept where it fitted (the share rule)
    chosen = {s.source_id for s in out.sections}
    seated = {s.section_type for s in out.sections}
    for kind in words:
        stype = SectionType(kind)
        if any(
            s.section_type is stype and s.metadata.get("truncated")
            for s in out.sections
        ):
            continue  # its budget is spent
        rest = [
            s
            for s in mgr._create_context_sections(results)
            if s.section_type is stype and s.source_id not in chosen
        ]
        if rest:
            room = left - (
                0
                if stype in seated
                else mgr.count_tokens(TokenManager._section_header(stype))
            )
            assert (
                rest[0].token_count > room < mgr.config.min_truncated_section_tokens
            ), (kind, room)
