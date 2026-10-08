# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual metadata and tracking information",
    )
