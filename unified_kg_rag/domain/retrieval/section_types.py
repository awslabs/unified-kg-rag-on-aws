# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kinds of retrieved item (``RetrievalResult.retriever_type``,
``Citation.section_type``) and what each kind is.

Every reader of a kind (the evaluator's file ranking, the console's
passage links, the MCP source view, the token budget) asks a trait here
instead of listing kinds, so a new kind is one row of
:data:`SECTION_TRAITS` (a unit test checks every kind has one).

A leaf module: ``domain.models`` re-exports :class:`SectionType`.
"""

from dataclasses import dataclass
from enum import StrEnum


class SectionType(StrEnum):
    """A retrieved item's kind; its value is the stored and wire value."""

    CLAIM = "claim"
    COMMUNITY = "community"
    ENTITY = "entity"
    GENERAL = "general"
    RELATIONSHIP = "relationship"
    # A node of the summary tree (the tree strategy).
    SUMMARY = "summary"
    TEXT = "text"
    # A graph query's result table (the structured strategy).
    QUERY_RESULT = "query_result"

    @classmethod
    def of(cls, value: object) -> "SectionType":
        """The kind ``value`` names (any case), else :attr:`GENERAL`: an
        unknown kind (a plugin's, or a malformed result) degrades to
        "other context" rather than failing the query."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or cls.GENERAL.value).lower())
        except ValueError:
            return cls.GENERAL

    @property
    def traits(self) -> "SectionTraits":
        return SECTION_TRAITS[self]

    @property
    def lineage_attributed(self) -> bool:
        return self.traits.lineage_attributed

    @property
    def stated_by_passages(self) -> bool:
        return self.traits.stated_by_passages

    @property
    def named(self) -> bool:
        return self.traits.named


@dataclass(frozen=True)
class SectionTraits:
    """What a kind of retrieved item is."""

    # Attributed to the files of its ``text_unit_ids`` lineage rather than
    # to a file of its own: an evaluator ranks its files after the
    # passages'.
    lineage_attributed: bool
    # Its text units are passages that state it (an entity, a relationship,
    # a claim, a graph query's rows), so a source view links it to the
    # passages of the same answer; a report or summary only rests on them.
    stated_by_passages: bool
    # Its title names the item itself; a passage's is its file's name.
    named: bool


SECTION_TRAITS: dict[SectionType, SectionTraits] = {
    SectionType.TEXT: SectionTraits(
        lineage_attributed=False, stated_by_passages=False, named=False
    ),
    SectionType.GENERAL: SectionTraits(
        lineage_attributed=False, stated_by_passages=False, named=True
    ),
    SectionType.ENTITY: SectionTraits(
        lineage_attributed=True, stated_by_passages=True, named=True
    ),
    SectionType.RELATIONSHIP: SectionTraits(
        lineage_attributed=True, stated_by_passages=True, named=True
    ),
    SectionType.CLAIM: SectionTraits(
        lineage_attributed=True, stated_by_passages=True, named=True
    ),
    SectionType.QUERY_RESULT: SectionTraits(
        lineage_attributed=True, stated_by_passages=True, named=True
    ),
    SectionType.COMMUNITY: SectionTraits(
        lineage_attributed=True, stated_by_passages=False, named=True
    ),
    SectionType.SUMMARY: SectionTraits(
        lineage_attributed=True, stated_by_passages=False, named=True
    ),
}

# The kinds whose files are their lineage's (``lineage_attributed``).
LINEAGE_ATTRIBUTED_TYPES = frozenset(t for t in SectionType if t.lineage_attributed)
