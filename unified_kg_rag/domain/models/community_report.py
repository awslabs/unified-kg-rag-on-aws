# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field

from .base import Named


class CommunityFinding(BaseModel):
    """One structured insight inside a community report (MS GraphRAG parity).

    MS GraphRAG community reports are not free text: each is a list of findings,
    where every finding has a one-line ``summary`` and a multi-sentence
    ``explanation``. Keeping findings structured (rather than baking them into a
    single prose blob) lets downstream search rank, truncate, and cite individual
    insights instead of an opaque paragraph.
    """

    summary: str = Field("", description="One-line headline for this insight")
    explanation: str = Field(
        "", description="Multi-sentence supporting detail for the finding"
    )


class CommunityReport(Named):
    community_id: str = Field(
        description="The ID of the community this report is associated with"
    )
    summary: str = Field("", description="Summary of the report")
    summary_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the summary"
    )
    full_content: str = Field("", description="Full content of the report")
    full_content_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the full report content"
    )
    findings: list[CommunityFinding] = Field(
        default_factory=list,
        description="Structured insights (MS GraphRAG parity). ``full_content`` "
        "is rendered deterministically from these plus the summary/rating.",
    )
    rating: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Importance/impact-severity rating 0-10 (MS GraphRAG parity)",
    )
    rating_explanation: str = Field(
        "", description="One-sentence justification for the rating"
    )
    rank: int | None = Field(
        1,
        description="Rank of the report, used for sorting. Higher means more important",
    )
    size: int | None = Field(
        None, description="The size of the report (Amount of text units)"
    )
    period: str | None = Field(None, description="The period of the report")

    def render_full_content(self) -> str:
        """Render a human/embedding-friendly report body from structured fields.

        Keeps the free-text ``full_content`` (used for embeddings, global-search
        map-reduce, and display) in sync with the structured ``findings``/``rating``
        so the new structure does not require a second LLM call or a separate
        index path.
        """
        lines: list[str] = []
        if self.name:
            lines.append(f"# {self.name}")
        if self.rating:
            rating_line = f"Importance rating: {self.rating:.1f}/10"
            if self.rating_explanation:
                rating_line += f" — {self.rating_explanation}"
            lines.append(rating_line)
        if self.summary:
            lines.append(self.summary)
        for finding in self.findings:
            if not (finding.summary or finding.explanation):
                continue
            if finding.summary and finding.explanation:
                lines.append(f"## {finding.summary}\n{finding.explanation}")
            else:
                lines.append(f"## {finding.summary or finding.explanation}")
        return "\n\n".join(lines)
