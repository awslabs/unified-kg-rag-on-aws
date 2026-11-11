# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic prompt definitions (domain layer).

``BasePrompt`` holds only template strings, variable contracts, and custom-prompt
resolution — NO LangChain/backend imports (the domain dependency rule). The
adapter layer (``adapters/llm/chain_factory.py``) turns a resolved
``ResolvedPrompt`` into a LangChain ``ChatPromptTemplate``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from typing import TYPE_CHECKING, ClassVar

from unified_kg_rag.domain.ingestion.chunk_length import max_chars

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import (
        Config,
        CustomPromptConfig,
        ModelRole,
    )

# Output-floor arithmetic for the long-output ingestion prompts. Bedrock
# reserves input + max_tokens against the tokens-per-minute quota when a request
# starts, so a floor should cover the largest answer the prompt's own limits
# allow plus the model's reasoning, and no more.
#
# Worst-case output tokens per character of chunk text: dense scripts (CJK,
# kana, Hangul) run at ~1 token per character (the bound estimate_token_count
# uses), and the Claude 4.7+ tokenizer counts up to 1.35x the tokens of older
# Claude tokenizers. English runs at ~0.25-0.35, so this is 4-5x headroom there.
WORST_CASE_TOKENS_PER_CHAR = 1.35
# One extracted entity or relationship record in the XML answer: ~35 tokens of
# tags, a name or source/target pair and a type (~15), a 1-2 sentence
# description (~40), a score and a short verbatim evidence span (~20) is ~110
# tokens; x1.35 for the newer tokenizer is ~150.
TOKENS_PER_GRAPH_RECORD = 150


def max_chunk_chars(config: "Config") -> int:
    """Longest text unit the chunkers can emit, in characters."""
    chunking = config.processing.chunking
    longest = max(chunking.max_chunk_size, chunking.fallback_chunk_size)
    return max_chars(longest, chunking.length_unit)


def thinking_tokens(config: "Config", role: "ModelRole | None") -> int:
    """The reasoning budget an ingestion call on ``role``'s model is sized
    for: its tier's ingestion effort (``aws.bedrock.tier_effort``) in
    ``EFFORT_THINKING_TOKENS``, the same budget an output ceiling leaves.
    Adaptive-thinking models count their reasoning toward max_tokens, so
    every floor adds it on top of the answer itself. The default tier's
    without a role."""
    from unified_kg_rag.domain.models.config import (
        EFFORT_THINKING_TOKENS,
        ModelPurpose,
    )

    tier = config.model_tier(config.model_id(role)) if role is not None else "default"
    effort = config.aws.bedrock.tier_effort(tier, ModelPurpose.INGESTION)
    return EFFORT_THINKING_TOKENS[effort]


def chunk_output_floor(
    config: "Config", role: "ModelRole | None", copies: int = 1
) -> int:
    """Floor for an answer up to ``copies`` x a chunk's text, plus reasoning."""
    answer = copies * max_chunk_chars(config) * WORST_CASE_TOKENS_PER_CHAR
    return int(answer) + thinking_tokens(config, role)


def graph_output_floor(config: "Config", role: "ModelRole | None") -> int:
    """Floor for a per-chunk graph answer at the configured record caps."""
    extraction = config.processing.graph_extraction
    records = extraction.max_entities_per_chunk + extraction.max_relationships_per_chunk
    return records * TOKENS_PER_GRAPH_RECORD + thinking_tokens(config, role)


def fill_slots(template: str, values: Mapping[str, str]) -> str:
    """``template`` with the placeholders named in ``values`` replaced by
    their value, every other placeholder and escaped brace kept as written.

    A template that does not parse (an unmatched brace) is returned as is.
    """
    if not values:
        return template
    try:
        parts = list(Formatter().parse(template))
    except ValueError:
        return template
    out: list[str] = []
    for literal, field, spec, conversion in parts:
        out.append(literal.replace("{", "{{").replace("}", "}}"))
        if field is None:
            continue
        if field in values and not spec and not conversion:
            out.append(values[field])
            continue
        out.append(
            "{"
            + field
            + (f"!{conversion}" if conversion else "")
            + (f":{spec}" if spec else "")
            + "}"
        )
    return "".join(out)


@dataclass(frozen=True)
class ResolvedPrompt:
    """A prompt with custom overrides applied, ready for backend assembly.

    Pure data: the adapter layer consumes this to build the concrete (LangChain)
    chat-prompt messages, keeping the domain free of backend imports.
    """

    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None


class BasePrompt:
    """A prompt's templates and variable contract, as class attributes.

    Prompt classes are never instantiated: callers pass the class itself and
    :meth:`resolve` reads the attributes.
    """

    system_prompt_template: ClassVar[str]
    human_prompt_template: ClassVar[str]
    input_variables: ClassVar[list[str]]
    output_variables: ClassVar[list[str] | None] = None
    # Subclasses set this to their CustomPromptConfig field prefix (e.g.
    # "graph_extraction" -> graph_extraction_system / graph_extraction_human).
    # The base resolve() looks the overrides up by convention, so a new prompt
    # only declares its key — no per-class _get_custom_prompts boilerplate.
    prompt_key: ClassVar[str | None] = None
    # Output-token floor this prompt needs. The request's max_tokens is the
    # larger of this and the configured default cap (clamped to the model
    # maximum), so a prompt with a long output is never truncated by a cap
    # sized for short answers. Thinking tokens count toward it.
    min_output_tokens: ClassVar[int] = 0
    # The model role (``models.roles``) whose model runs this prompt where
    # the caller picks it by prompt (``GraphRAGChain._get_chain_for_prompt``).
    model_role: ClassVar["ModelRole | None"] = None
    # Prompts whose answer is parsed as XML split the built-in system prompt
    # into a domain-adaptable persona (``system_preamble``) and the rules and
    # output format the parser relies on (``output_rules``); prompt tuning
    # replaces only the preamble. ``required_output_tags`` are the tags a
    # custom override must still teach — a load-time warning names any that
    # are missing. Empty for prompts without that split.
    system_preamble: ClassVar[str] = ""
    output_rules: ClassVar[str] = ""
    required_output_tags: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def output_floor(cls, config: "Config", role: "ModelRole | None" = None) -> int:
        """Output-token floor for this prompt under ``config``, on the model
        of ``role`` (the prompt's ``model_role`` by default; the output
        fixer's when it re-emits the answer), whose reasoning it leaves room
        for (:func:`thinking_tokens`).

        Prompts whose longest answer depends on configured limits (chunk size,
        per-chunk entity caps, report length) override this to derive it; the
        rest return the static ``min_output_tokens``.
        """
        return cls.min_output_tokens

    @classmethod
    def inlined_slots(cls, config: "Config") -> dict[str, str]:
        """``{variable: value}`` of the slots that replaced fixed template text.

        The extraction-record fingerprint hashes a template with these filled
        in (:func:`fill_slots`), so turning fixed text into a variable whose
        value under ``config`` is that text re-keys no record it shaped.
        """
        return {}

    @classmethod
    def code_slots(cls) -> dict[str, str]:
        """``{variable: value}`` of the slots rendered from the configuration,
        as they render when it declares nothing: the fixed text (code) they
        hold then. A stage-cache key hashes a template with these filled in,
        so an edit of that text re-keys the stages it shapes, without
        reading a schema not loaded yet (the declared values are keyed as
        settings)."""
        return {}

    @classmethod
    def validate_templates(cls) -> None:
        """Raise ``ValueError`` when a declared input variable appears in
        neither shipped template."""
        for var in cls.input_variables:
            if not var or not isinstance(var, str):
                raise ValueError(f"Invalid input variable: {var}")

            if var == "image_data":
                continue

            if (
                f"{{{var}}}" not in cls.human_prompt_template
                and f"{{{var}}}" not in cls.system_prompt_template
            ):
                raise ValueError(
                    f"Input variable '{var}' not found in any prompt template"
                )

    @classmethod
    def resolve(
        cls,
        custom_prompts: "CustomPromptConfig | None" = None,
    ) -> ResolvedPrompt:
        """Apply any custom overrides and return a backend-agnostic prompt.

        Replaces the former ``get_prompt`` (which built a LangChain
        ``ChatPromptTemplate`` here, violating the domain dependency rule). The
        adapter now consumes the returned :class:`ResolvedPrompt`.
        """
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template

        overridden = False
        if custom_prompts:
            custom_system, custom_human = cls._get_custom_prompts(custom_prompts)
            if custom_system:
                system_template = custom_system
                overridden = True
            if custom_human:
                human_template = custom_human
                overridden = True

        # A user-supplied override owns its own variable set; the built-in
        # input_variables may not all appear in it (e.g. overriding only the
        # human template, or dropping a variable). Don't hard-fail on that —
        # the strict missing-variable check is for the SHIPPED defaults, so
        # partial / minimal overrides are allowed.
        if not overridden:
            cls.validate_templates()
        return ResolvedPrompt(
            system_prompt_template=system_template,
            human_prompt_template=human_template,
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
        )

    @classmethod
    def _get_custom_prompts(
        cls, custom_prompts: "CustomPromptConfig"
    ) -> tuple[str | None, str | None]:
        """Look up the (system, human) overrides by ``prompt_key`` convention.

        Returns ``(None, None)`` when the prompt declares no key. Subclasses set
        ``prompt_key`` instead of overriding this method.
        """
        if not cls.prompt_key:
            return None, None
        return (
            getattr(custom_prompts, f"{cls.prompt_key}_system", None),
            getattr(custom_prompts, f"{cls.prompt_key}_human", None),
        )
