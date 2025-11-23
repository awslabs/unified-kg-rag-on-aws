# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"


class ConversationMessage(BaseModel):
    id: str = Field(description="Unique identifier for the message")
    role: MessageRole = Field(
        description="Role of the message sender (user, assistant, or system)"
    )
    content: str = Field(description="The actual content/text of the message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )
    token_count: int | None = Field(
        default=None, description="Number of tokens in the message content"
    )
    processing_time: float | None = Field(
        default=None,
        description="Time taken to process and generate the message (in seconds)",
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents and references used to generate this response",
    )
    search_strategy: str | None = Field(
        default=None,
        description="Search strategy used for retrieval-augmented generation",
    )
    confidence_score: float | None = Field(
        default=None,
        description="Confidence score of the assistant's response (0.0-1.0)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message-specific metadata and attributes",
    )


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


class ConversationMemory(BaseModel):
    conversation_id: str = Field(
        description="Unique identifier for the conversation session"
    )
    user_id: str | None = Field(
        default=None,
        description="Identifier of the user participating in the conversation",
    )
    session_id: str | None = Field(
        default=None,
        description="Identifier of the broader session containing this conversation",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Chronological history of all messages in the conversation",
    )
    context: ConversationContext = Field(
        default_factory=ConversationContext,
        description="Contextual state and tracking information for the conversation",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the conversation was initially created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the most recent conversation update",
    )
    max_messages: int = Field(
        default=20,
        description="Maximum number of messages to retain in memory (older messages are pruned)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation-level metadata and configuration",
    )

    def add_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now()

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-int(self.max_messages) :]

    def get_recent_messages(self, count: int = 5) -> list[ConversationMessage]:
        return self.messages[-count:] if self.messages else []

    def get_conversation_history(self, include_system: bool = False) -> str:
        history_parts = []
        for message in self.messages:
            if not include_system and message.role == MessageRole.SYSTEM:
                continue

            role_prefix = {
                MessageRole.USER: "User",
                MessageRole.ASSISTANT: "Assistant",
                MessageRole.SYSTEM: "System",
            }.get(message.role, "Unknown")

            history_parts.append(f"{role_prefix}: {message.content}")

        return "\n\n".join(history_parts)

    def get_context_summary(self, n_entities: int = 5, n_topics: int = 3) -> str:
        summary_parts = []

        if self.context.mentioned_entities:
            summary_parts.append(
                f"Mentioned Entities: {', '.join(self.context.mentioned_entities[:n_entities])}"
            )

        if self.context.current_topics:
            summary_parts.append(
                f"Current Topics: {', '.join(self.context.current_topics[:n_topics])}"
            )

        if self.context.user_intent:
            summary_parts.append(f"User Intent: {self.context.user_intent}")

        return " | ".join(summary_parts) if summary_parts else "New conversation"
