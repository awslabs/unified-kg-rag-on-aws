# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Backend-agnostic prompt definitions (domain layer).

``BasePrompt`` holds only template strings, variable contracts, and custom-prompt
resolution — NO LangChain/backend imports (the domain dependency rule). The
adapter layer (``adapters/aws/chain_factory.py``) turns a resolved
``ResolvedPrompt`` into a LangChain ``ChatPromptTemplate``.
"""
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aws_graphrag.domain.models.config import CustomPromptConfig


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

        if custom_prompts:
            custom_system, custom_human = cls._get_custom_prompts(custom_prompts)
            if custom_system:
                system_template = custom_system
            if custom_human:
                human_template = custom_human

        # Run field validation via a throwaway instance.
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
        return None, None
