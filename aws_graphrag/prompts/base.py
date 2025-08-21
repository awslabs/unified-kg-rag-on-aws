from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

if TYPE_CHECKING:
    from ..models.config import CustomPromptConfig


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
    def get_prompt(
        cls,
        enable_prompt_cache: bool = False,
        custom_prompts: "CustomPromptConfig | None" = None,
    ) -> ChatPromptTemplate:
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template

        if custom_prompts:
            custom_system, custom_human = cls._get_custom_prompts(custom_prompts)
            if custom_system:
                system_template = custom_system
            if custom_human:
                human_template = custom_human

        instance = cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=system_template,
            human_prompt_template=human_template,
        )

        if enable_prompt_cache:
            messages = cls._create_cached_messages(instance)
        else:
            messages = cls._create_standard_messages(instance)

        return ChatPromptTemplate.from_messages(messages)

    @classmethod
    def _create_cached_messages(
        cls, instance: "BasePrompt"
    ) -> list[HumanMessagePromptTemplate | SystemMessagePromptTemplate]:
        return [
            SystemMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": instance.system_prompt_template},
                    {"cachePoint": {"type": "default"}},
                ],
                input_variables=instance.input_variables,
            ),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": instance.human_prompt_template},
                    {"cachePoint": {"type": "default"}},
                ],
                input_variables=instance.input_variables,
            ),
        ]

    @classmethod
    def _create_standard_messages(
        cls, instance: "BasePrompt"
    ) -> list[HumanMessagePromptTemplate | SystemMessagePromptTemplate]:
        return [
            SystemMessagePromptTemplate.from_template(
                template=instance.system_prompt_template,
                input_variables=instance.input_variables,
            ),
            HumanMessagePromptTemplate.from_template(
                template=instance.human_prompt_template,
                input_variables=instance.input_variables,
            ),
        ]

    @classmethod
    def _get_custom_prompts(
        cls, custom_prompts: "CustomPromptConfig"
    ) -> tuple[str | None, str | None]:
        return None, None
