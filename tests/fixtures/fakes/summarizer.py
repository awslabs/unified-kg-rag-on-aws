# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model-free ``DescriptionSummarizer`` whose summaries are order-independent.

A summary is the sorted set of the lines it was made from, a line that is
itself a summary contributing its own lines: summarizing a summary followed
by new lines gives what summarizing all the lines gives. So an incremental
run that extends a stored summary (``extend_summaries``) and a full build
that summarizes every line agree exactly when they summarize the same lines,
and a divergence is the indexer's, not the summarizer's.
"""

from __future__ import annotations

import re

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.domain.ingestion.cache_keys import description_summary_fingerprint
from unified_kg_rag.domain.models import Config

_SUMMARY = re.compile(r"SUM\{(.*)\}")


def atom_summary(description: str) -> str:
    """The order-independent summary of ``description``'s lines."""
    atoms: set[str] = set()
    for line in description.split("\n"):
        line = line.strip()
        if match := _SUMMARY.fullmatch(line):
            atoms |= set(match.group(1).split("|"))
        elif line:
            atoms.add(line)
    return "SUM{" + "|".join(sorted(atoms)) + "}"


class AtomSetSummarizer(DescriptionSummarizer):
    """Summarizes with :func:`atom_summary` (no model, no providers)."""

    def __init__(self, config: Config) -> None:  # noqa: D107 - no chain to build
        self.config = config
        self.summarization_config = (
            config.processing.graph_extraction.description_summarization
        )
        self.settings = description_summary_fingerprint(config)
        self.calls = 0

    def _summarize_many(
        self, names: list[str], descriptions: list[str]
    ) -> list[str | None]:
        self.calls += len(descriptions)
        return [atom_summary(d) for d in descriptions]
