# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TokenManager — context-window budgeting/optimization shared by
all search strategies (AWS-free: the Bedrock token counter is patched out).

These exercise the real budget math, PRIORITY_MULTIPLIERS-driven ordering, the
quality-score blend, the budget-exceeded short circuit, and the empty-context
fallback string. The token counter is replaced with a deterministic
word-count stub so assertions are exact and AWS is never touched.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unified_kg_rag.adapters.aws.bedrock import get_language_model_info
from unified_kg_rag.adapters.retrieval import token_manager as tm_module
from unified_kg_rag.adapters.retrieval.token_manager import (
    ContextSection,
    OptimizedContext,
    SectionType,
    TokenManager,
)
from unified_kg_rag.domain.models import Config, LanguageModelId, RetrievalResult
from unified_kg_rag.domain.models.config import ContextTypeBudgetConfig

pytestmark = pytest.mark.unit


def _make_manager(
    mocker,
    max_context_tokens: int | None = None,
    answer_model_id: LanguageModelId | None = None,
    enable_1m_context: bool = False,
) -> TokenManager:
    """Build a TokenManager with Bedrock/boto wiring stubbed and a deterministic
    word-count token counter (1 token per whitespace-delimited word).

    ``max_context_tokens=None`` (the default) exercises the production path where
    the budget is derived from the answer model's context window.
    """
    mocker.patch.object(tm_module, "boto3")
    mocker.patch.object(tm_module, "get_assumed_role_boto_session")

    fake_counter = mocker.Mock()
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    mocker.patch.object(tm_module, "BedrockTokenCounter", return_value=fake_counter)

    config = Config()
    config.search.token_manager.max_context_tokens = max_context_tokens
    config.aws.bedrock.enable_1m_context = enable_1m_context
    if answer_model_id is not None:
        config.search.answer_generation_model_id = answer_model_id
    return TokenManager(config)


def _r(content: str, score: float, retriever_type: str, source: str) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, retriever_type=retriever_type, source=source
    )


class TestContextBudgetDerivation:
    """The prompt budget must track the answer model's real context window.

    A hardcoded budget silently overflows a smaller model's window (the old
    default of 200000 left no room for a 200K model's 64K output reservation)
    and wastes a larger one.
    """

    def test_derives_from_model_window_when_unset(self, mocker) -> None:
        # Sonnet 5: 1M window - 128K output, less 10% headroom.
        mgr = _make_manager(mocker, answer_model_id=LanguageModelId.CLAUDE_V5_SONNET)
        assert mgr._max_context_tokens == int((1000000 - 128000) * 0.9)

    def test_smaller_window_gets_smaller_budget(self, mocker) -> None:
        mgr = _make_manager(mocker, answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET)
        assert mgr._max_context_tokens == int((200000 - 64000) * 0.9)

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
            assert info is not None
            assert (
                mgr._max_context_tokens + info.max_output_tokens
                <= info.context_window_size
            ), model_id

    def test_explicit_budget_over_window_is_clamped(self, mocker) -> None:
        mgr = _make_manager(
            mocker,
            max_context_tokens=200000,
            answer_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
        assert mgr._max_context_tokens == int((200000 - 64000) * 0.9)

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
        assert wide._max_context_tokens == int((1000000 - 64000) * 0.9)

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
        # query alone (3 tokens) + buffer (512) exceeds a tiny target -> short circuit.
        mgr = _make_manager(mocker, max_context_tokens=1024)
        results = [_r("a b c d e", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="one two three", max_tokens=100)
        assert out.sections == []
        assert out.sections_included == 0
        assert out.sections_excluded == 1
        assert out.quality_score == 0.0

    def test_fits_within_budget_includes_section(self, mocker) -> None:
        mgr = _make_manager(mocker, max_context_tokens=200000)
        results = [_r("alpha beta gamma", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.total_tokens == 3

    def test_max_tokens_enforced_excludes_overflow(self, mocker) -> None:
        # Budget = max_tokens - query(1) - buffer(0); set buffer to 0 for exactness.
        mgr = _make_manager(mocker)
        # Two sections of 5 tokens each; budget allows only one.
        results = [
            _r("a b c d e", 0.9, "text", "s1"),
            _r("f g h i j", 0.8, "text", "s2"),
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        # available = 6 - 1 - 0 = 5 tokens -> exactly one 5-token section fits.
        assert out.sections_included == 1
        assert out.total_tokens == 5

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
    def test_priority_multiplier_orders_equal_scores(self, mocker) -> None:
        # Equal base scores; TEXT(1.3) should outrank GENERAL(0.8).
        mgr = _make_manager(mocker)
        results = [
            _r("g g g g g", 0.5, "general", "gen"),
            _r("t t t t t", 0.5, "text", "txt"),
        ]
        # Budget for only one section -> the higher-priority TEXT wins.
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        assert out.sections_included == 1
        assert out.sections[0].section_type == SectionType.TEXT

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
            _r("g g g g g", 0.5, "general", "gen"),
            _r("t t t t t", 0.5, "text", "txt"),
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        assert out.sections_included == 1
        assert out.total_tokens <= 5
        # TEXT (1.3) outranks GENERAL (0.8) at equal base score.
        assert out.sections[0].section_type == SectionType.TEXT

    def test_last_resort_section_is_truncated_when_nothing_fits_whole(
        self, mocker
    ) -> None:
        mgr = _make_manager(mocker)
        floor = mgr.config.min_truncated_section_tokens
        out = mgr.optimize_context(
            [_r(" ".join(["t"] * 500), 0.9, "text", "t0")],
            query="q",
            max_tokens=floor + 1,
            max_context_tokens_buffer=0,
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
        # GENERAL multiplier is 0.8: 0.5 * 0.8 = 0.4.
        assert section.section_type == SectionType.GENERAL
        assert section.priority == pytest.approx(0.4)

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

    def test_claim_section_type_has_weight(self, mocker) -> None:
        # Claims are evidentiary; weighted alongside relationships (1.1).
        assert SectionType.CLAIM in TokenManager.PRIORITY_MULTIPLIERS
        assert TokenManager.PRIORITY_MULTIPLIERS[SectionType.CLAIM] == pytest.approx(
            1.1
        )

    def test_claim_retriever_type_maps_to_claim_section(self, mocker) -> None:
        mgr = _make_manager(mocker)
        section = mgr._create_context_section(
            _r("a claim", 0.5, "claim", "c1"), index=0
        )
        # CLAIM multiplier is 1.1: 0.5 * 1.1 = 0.55.
        assert section.section_type == SectionType.CLAIM
        assert section.priority == pytest.approx(0.55)

    def test_missing_score_defaults_to_half(self, mocker) -> None:
        mgr = _make_manager(mocker)
        result = RetrievalResult(
            content="x y", score=0.0, retriever_type="entity", source="s"
        )
        # score 0.0 is falsy -> base_score becomes 0.5; ENTITY multiplier 1.2.
        section = mgr._create_context_section(result, index=0)
        assert section.priority == pytest.approx(0.5 * 1.2)


class TestPerTypeBudgetIsAHardCap:
    """TYPE_BUDGET_PROP must CAP each type, not floor it.

    The previous implementation packed each type to its sub-budget and then ran a
    second pass that pooled every unused token and offered it to the leftovers of
    ANY type by raw priority. That made the proportions advisory: whichever type
    had the most/highest-scoring leftovers absorbed the whole remainder (measured
    on musique50/n=50: community reports took 0.68 of the window against a 0.10
    prop; with ranked chunk scores TEXT took 0.84 and evicted every community
    report). Neither upstream pools — MS GraphRAG's mixed_context packs community
    / local / text against three independent strict budgets and LightRAG
    truncates each type separately.
    """

    def test_one_type_cannot_exceed_its_share_when_another_type_has_candidates(
        self, mocker
    ) -> None:
        mgr = _make_manager(mocker)
        # available = 1000 - 1(query) - 0(buffer) = 1000 tokens.
        # TEXT prop 0.50 / COMMUNITY prop 0.10 -> renormalized over the two present
        # types: TEXT 5/6 (833), COMMUNITY 1/6 (166).
        # 40 text sections of 100 tokens each want 4000 tokens; they must not eat
        # the community share even though TEXT has the higher priority multiplier.
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
            max_context_tokens_buffer=0,
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
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
        assert out.total_tokens == 1000

    def test_unused_share_of_a_sparse_type_is_left_unused(self, mocker) -> None:
        # TEXT is sparse (2 candidates); COMMUNITY has plenty. The rest of TEXT's
        # share must be LEFT UNUSED, not handed to community reports. Re-offering
        # the remainder to whichever type still has candidates was measured to
        # reproduce the original defect verbatim (community kept 0.66 of the window
        # because it was the only type with candidates left), which is why this is a
        # single pass.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["t"] * 50), 0.9, "text", f"t{i}") for i in range(2)] + [
            _r(" ".join(["c"] * 50), 0.4, "community", f"c{i}") for i in range(40)
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
        by_type: dict[SectionType, int] = {}
        for section in out.sections:
            by_type[section.section_type] = (
                by_type.get(section.section_type, 0) + section.token_count
            )
        # Renormalized over {TEXT 0.50, COMMUNITY 0.10} -> TEXT 833, COMMUNITY 166.
        assert by_type[SectionType.TEXT] == 100  # only 2 candidates exist
        assert by_type[SectionType.COMMUNITY] <= 200
        assert out.total_tokens < 400  # the rest of TEXT's share stays unused

    def test_general_only_candidates_still_get_a_budget(self, mocker) -> None:
        # Global search emits ONLY GENERAL sections, so renormalizing over present
        # types must not produce a zero budget and an empty context for that whole
        # strategy.
        mgr = _make_manager(mocker)
        results = [_r(" ".join(["g"] * 50), 0.9, "general", f"g{i}") for i in range(10)]
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
        assert out.sections_included == 10
        assert out.total_tokens == 500

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
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
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
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
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
        out = mgr.optimize_context(
            results, query="q", max_tokens=1001, max_context_tokens_buffer=0
        )
        types = {s.section_type for s in out.sections}
        assert SectionType.COMMUNITY in types
        assert SectionType.TEXT in types
        assert out.total_tokens <= 1000


class TestQualityScore:
    def test_empty_inputs_zero(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr._calculate_quality_score([], []) == 0.0

    def test_full_selection_scores_one(self, mocker) -> None:
        # All sections selected -> priority_coverage 1.0 and type_diversity 1.0.
        mgr = _make_manager(mocker)
        results = [
            _r("a b", 0.5, "text", "s1"),
            _r("c d", 0.5, "entity", "s2"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 2
        assert out.quality_score == pytest.approx(1.0)

    def test_partial_selection_blends_coverage_and_diversity(self, mocker) -> None:
        # Two TEXT sections, only one fits. priority_coverage = 0.5 of total
        # priority; type_diversity = 1/1 (one type) = 1.0.
        # score = 0.5*0.7 + 1.0*0.3 = 0.65.
        mgr = _make_manager(mocker)
        results = [
            _r("a b c d e", 0.5, "text", "s1"),
            _r("f g h i j", 0.5, "text", "s2"),
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        assert out.sections_included == 1
        assert out.quality_score == pytest.approx(0.65)


class TestBuildContextString:
    def test_empty_context_fallback_message(self) -> None:
        empty = OptimizedContext(
            sections=[],
            total_tokens=0,
            sections_included=0,
            sections_excluded=0,
            quality_score=0.0,
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
        for section_type in SectionType:
            assert section_type in TokenManager._BUDGET_FIELDS
            assert hasattr(
                ContextTypeBudgetConfig(), TokenManager._BUDGET_FIELDS[section_type]
            )
