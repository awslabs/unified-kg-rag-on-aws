# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The citations of an answer (``RAGOutput.citations``, docs/design.md §7.1)."""

from typing import Any

from pydantic import BaseModel, Field


class CitedDocument(BaseModel):
    """One document a citation rests on, through the text units it names."""

    document_id: str | None = Field(
        default=None,
        description="The document's id; None for a text unit indexed before "
        "its document was recorded with it",
    )
    file_name: str | None = Field(default=None, description="Name of the source file")
    file_path: str | None = Field(
        default=None,
        description="Source file path relative to the corpus root; None when "
        "the index holds no relative path (indexed before it was recorded)",
    )
    page: int | None = Field(
        default=None,
        description="First page (1-based) of the cited text, for a document "
        "loaded page by page (PDF)",
    )
    page_end: int | None = Field(
        default=None, description="Last page (1-based) of the cited text"
    )
    text_unit_ids: list[str] = Field(
        default_factory=list,
        description="The cited text units of this document",
    )


class Citation(BaseModel):
    """One numbered section of the answer context and what it rests on.

    ``marker`` is the number the context showed (``[marker]``); ``cited``
    whether the answer uses it. The document fields are those of
    ``documents[0]``, the document most of the section's resolved text units
    belong to.
    """

    marker: int = Field(description="The section's number in the context, [n]")
    cited: bool = Field(
        default=False, description="Whether the answer cites this marker"
    )
    section_type: str = Field(
        description="Section type: text, entity, relationship, claim, "
        "community, summary or general"
    )
    source_id: str | None = Field(
        default=None, description="The id of the retrieved item"
    )
    title: str | None = Field(
        default=None,
        description="What the section is: its file name (a passage), entity "
        "name, 'source -> target' (a relationship) or report title",
    )
    document_id: str | None = Field(
        default=None, description="The id of the main cited document"
    )
    file_name: str | None = Field(
        default=None, description="File name of the main cited document"
    )
    file_path: str | None = Field(
        default=None,
        description="Path of the main cited document relative to the corpus "
        "root (None when not recorded)",
    )
    page: int | None = Field(
        default=None, description="First cited page (1-based), when known"
    )
    page_end: int | None = Field(
        default=None, description="Last cited page (1-based), when known"
    )
    chunk_id: str | None = Field(
        default=None, description="The text unit id of a passage (None otherwise)"
    )
    text_unit_ids: list[str] = Field(
        default_factory=list,
        description="The text units the section rests on: a passage's own, "
        "an entity's, relationship's, claim's, report's or summary's lineage",
    )
    documents: list[CitedDocument] = Field(
        default_factory=list,
        description="The documents of the resolved text units, the most "
        "supported first",
    )
    provenance: dict[str, Any] | None = Field(
        default=None,
        description="Seed-graph provenance (kind, seed_graphs, iri) of an item "
        "a seed graph states; None otherwise",
    )
    iri: str | None = Field(default=None, description="The item's IRI, if any")
