# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"


class ConversationContext(BaseModel):
    mentioned_entities: list[str] = Field(
        default_factory=list,
        description="All entities mentioned throughout the conversation",
    )
    focused_entities: list[str] = Field(
        default_factory=list,
        description="Entities currently in focus or being actively discussed",
    )
    current_topics: list[str] = Field(
        default_factory=list, description="Current conversation topics and themes"
    )
    user_intent: str | None = Field(
        default=None, description="Detected or inferred user intent and purpose"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual metadata and tracking information",
    )
