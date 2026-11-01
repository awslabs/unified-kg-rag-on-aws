# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What one text unit contributes to the graph (its extraction record)."""

from typing import Literal

from pydantic import BaseModel, Field

from .entity import Entity
from .relationship import Relationship

RejectionReason = Literal["ungrounded", "low_confidence", "off_schema"]


class TextUnitExtraction(BaseModel):
    """One text unit's own entities and relationships.

    Extraction and gleaning write here and nowhere else: every item carries
    the unit's own name, type, description, strength, confidence and
    attributes, and cites only this unit. The graph is the merge of all
    records (``domain.ingestion.contributions.merge_contributions``), so a
    gleaning correction edits what this unit says and never what another unit
    says. Endpoints the unit names only inside a relationship are not
    entities here; the merge (and the gleaning prompt) adds them.
    """

    text_unit_id: str = Field(description="The text unit the record is of")
    entities: list[Entity] = Field(
        default_factory=list, description="Entities the unit lists"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="Relationships the unit states"
    )
    rejected: dict[str, RejectionReason] = Field(
        default_factory=dict,
        description="Entity keys rejected for the unit: ungrounded (the "
        "grounding guard dropped it; lifted once the unit lists the entity "
        "with grounded evidence), low_confidence (below "
        "entity_confidence_threshold; never lifted) or off_schema (strict "
        "schema_mode dropped it for an undeclared type; lifted once the unit "
        "lists the entity with a declared type). Gleaning does not bring "
        "them back through a relationship naming them.",
    )
    gleaned: bool = Field(
        default=False, description="Whether gleaning has refined the record"
    )
    stored: bool = Field(
        default=False,
        description="Whether the record is in the extraction store (read "
        "from it, or written by this run)",
    )

    def reject(self, key: str, reason: RejectionReason) -> None:
        """Record ``key`` as rejected; a low-confidence rejection is kept."""
        if self.rejected.get(key) != "low_confidence":
            self.rejected[key] = reason
