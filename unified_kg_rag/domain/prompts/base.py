# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic prompt definitions (domain layer).

``BasePrompt`` holds only template strings, variable contracts, and custom-prompt
resolution — NO LangChain/backend imports (the domain dependency rule). The
adapter layer (``adapters/aws/chain_factory.py``) turns a resolved
``ResolvedPrompt`` into a LangChain ``ChatPromptTemplate``.
"""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import Config, CustomPromptConfig

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
# Adaptive-thinking models (Claude 4.7+) count their reasoning toward
# max_tokens, so every floor adds this much on top of the answer itself.
THINKING_HEADROOM_TOKENS = 8192
# One extracted entity or relationship record in the XML answer: ~35 tokens of
# tags, a name or source/target pair and a type (~15), a 1-2 sentence
# description (~40), a score and a short verbatim evidence span (~20) is ~110
# tokens; x1.35 for the newer tokenizer is ~150.
TOKENS_PER_GRAPH_RECORD = 150


def max_chunk_chars(config: "Config") -> int:
    """Longest text unit the chunkers can emit, in characters."""
    chunking = config.processing.chunking
    return max(chunking.max_chunk_size, chunking.fallback_chunk_size)


def chunk_output_floor(config: "Config", copies: int = 1) -> int:
    """Floor for an answer up to ``copies`` x a chunk's text, plus reasoning."""
    answer = copies * max_chunk_chars(config) * WORST_CASE_TOKENS_PER_CHAR
    return int(answer) + THINKING_HEADROOM_TOKENS


def graph_output_floor(config: "Config") -> int:
    """Floor for a per-chunk graph answer at the configured record caps."""
    extraction = config.processing.graph_extraction
    records = extraction.max_entities_per_chunk + extraction.max_relationships_per_chunk
    return records * TOKENS_PER_GRAPH_RECORD + THINKING_HEADROOM_TOKENS


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


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None
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
    def output_floor(cls, config: "Config") -> int:
        """Output-token floor for this prompt under ``config``.

        Prompts whose longest answer depends on configured limits (chunk size,
        per-chunk entity caps, report length) override this to derive it; the
        rest return the static ``min_output_tokens``.
        """
        return cls.min_output_tokens

    def __post_init__(self) -> None:
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        if self.input_variables is not None:
            for var in self.input_variables:
                if not var or not isinstance(var, str):
                    raise ValueError(f"Invalid input variable: {var}")

                if var == "image_data":
                    continue

                if (
                    f"{{{var}}}" not in self.human_prompt_template
                    and f"{{{var}}}" not in self.system_prompt_template
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
        # Concrete prompt subclasses define these dataclass fields as class-level
        # attributes, so class access is valid at runtime.
        system_template = cls.system_prompt_template  # type: ignore[misc]
        human_template = cls.human_prompt_template  # type: ignore[misc]

        overridden = False
        if custom_prompts:
            custom_system, custom_human = cls._get_custom_prompts(custom_prompts)
            if custom_system:
                system_template = custom_system
                overridden = True
            if custom_human:
                human_template = custom_human
                overridden = True

        if overridden:
            # A user-supplied override owns its own variable set; the built-in
            # input_variables may not all appear in it (e.g. overriding only the
            # human template, or dropping a variable). Don't hard-fail on that —
            # the strict missing-variable check is for the SHIPPED defaults
            # (enforced when the dataclass is instantiated). Skip it here so
            # partial / minimal overrides are allowed.
            return ResolvedPrompt(
                system_prompt_template=system_template,
                human_prompt_template=human_template,
                input_variables=cls.input_variables,  # type: ignore[misc]
                output_variables=cls.output_variables,
            )

        # No override: validate the shipped defaults via a throwaway instance.
        instance = cls(
            input_variables=cls.input_variables,  # type: ignore[misc]
            output_variables=cls.output_variables,
            system_prompt_template=system_template,
            human_prompt_template=human_template,
        )
        return ResolvedPrompt(
            system_prompt_template=instance.system_prompt_template,
            human_prompt_template=instance.human_prompt_template,
            input_variables=instance.input_variables,
            output_variables=instance.output_variables,
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
