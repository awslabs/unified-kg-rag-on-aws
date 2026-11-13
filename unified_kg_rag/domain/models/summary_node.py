# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The nodes of RAPTOR's summary tree and the record of a tree build
(docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in)."""

from datetime import datetime

from pydantic import BaseModel, Field

from .base import Identified

# The ``_meta`` key of the summary index that holds the build record.
SUMMARY_TREE_META = "summary_tree"


class SummaryNode(Identified):
    """One summary of a cluster of text units (level 1) or of summaries
    (higher levels). Text units are level 0 and are not copied."""

    level: int = Field(ge=1, description="1 above the text units, and so on")
    text: str = Field(description="The summary, in the index language")
    child_ids: list[str] = Field(
        description="Its direct children: text-unit ids at level 1, node ids "
        "above (a child promoted from a lower level keeps its id)"
    )
    text_unit_ids: list[str] = Field(
        description="The text units it summarizes transitively, sorted: its "
        "provenance and removal key, cut at max_lineage_text_units"
    )
    lineage_complete: bool = Field(
        default=True,
        description="False when text_unit_ids was cut: the lineage is then "
        "resolved through child_ids",
    )
    n_leaves: int = Field(ge=1, description="Text units it summarizes, uncut")
    n_tokens: int = Field(ge=0, description="Tokens of text (token counter)")
    partition: str | None = Field(
        default=None,
        description="The partition_by value tuple (JSON) or document it was "
        "built in; None for the whole namespace",
    )
    language: str = Field(description="The language the summary is written in")
    input_key: str = Field(
        description="Digest of what produced the summary (the children's "
        "texts, the language and the summary settings): a cluster with the "
        "same key reuses this summary"
    )


class SummaryTreeRecord(BaseModel):
    """What built a namespace's summary tree, kept in the summary index's
    ``_meta`` (``SUMMARY_TREE_META``): the next run compares the leaf digest
    and the fingerprint to decide whether to rebuild."""

    pipeline_id: str = Field(description="The ingestion run that built it")
    built_at: datetime = Field(description="When it was written (UTC)")
    leaf_digest: str = Field(
        description="Digest of the text units it was built over: each one's "
        "id, the text the tree read (its translation, when it holds one) and "
        "its document filters"
    )
    fingerprint: str = Field(
        description="Fingerprint of the tree settings, summary model, prompt, "
        "language and embedding model (cache_keys.summary_tree_fingerprint)"
    )
    scope: str = Field(description="corpus or document")
    algorithm: str = Field(description="kmeans or gmm")
    leaves: int = Field(ge=0, description="Text units it was built over")
    nodes: int = Field(ge=0, description="Summary nodes written")
    levels: int = Field(ge=0, description="Highest level written")
    partial: bool = Field(
        default=False,
        description="Some summaries failed (their children were promoted): "
        "the next run rebuilds, reusing what it can",
    )
    pruned: bool = Field(
        default=False,
        description="Summary nodes were deleted since it was built (a removal "
        "of the text units they cite): with refresh every_run the next run "
        "rebuilds it, even over the same text units (a reprocessed document "
        "writes its units back unchanged)",
    )
    failed_summaries: int = Field(default=0, ge=0)
    reused_summaries: int = Field(
        default=0, ge=0, description="Summaries taken from the previous tree"
    )
    package_version: str | None = Field(default=None)
