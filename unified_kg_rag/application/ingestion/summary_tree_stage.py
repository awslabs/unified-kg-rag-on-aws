# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``summary_tree`` stage: RAPTOR's summary tree over the stored text
units, for the ``raptor`` methodology (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

Runs after indexing and reads what the stores hold, not the run's context,
so the tree is a function of the stored text units: an incremental run and a
full build of the same corpus build the same tree, and a run whose delta is
empty builds a tree that is missing or stale. The tree is rebuilt whole and
swapped in; a cluster whose children are unchanged reuses its summary (the
live tree's node with the same ``input_key``) and makes no model call.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

from unified_kg_rag.adapters.ingestion.tree_summarizer import TreeSummarizer
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.pipeline_stages import _RegistryStage
from unified_kg_rag.domain.ingestion.cache_keys import (
    index_language,
    summary_tree_fingerprint,
)
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.summary_tree import (
    Cluster,
    Leaf,
    Realized,
    TreeBuild,
    TreeSettings,
    build_tree,
    leaf_entries_digest,
    leaf_entry,
    partition_key,
)
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    IndexArtifact,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SummaryNode,
    SummaryTreeRecord,
)
from unified_kg_rag.ports.indexer import BaseIndexer, VectorIndexer
from unified_kg_rag.shared import (
    NamespaceLeaseError,
    PipelineStageError,
    get_logger,
)

if TYPE_CHECKING:
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)

# Leaves read with their vectors at a time (a page of 1,024-dimension
# vectors is about 2 MB of JSON).
_VECTOR_PAGE = 250
# Text units read at a time for the leaf digest (their text, without
# vectors).
_DIGEST_PAGE = 1_000


class SummaryTreeStage(_RegistryStage):
    # The tree is built from the stored text units, so it is built after a
    # code upgrade changed the earlier stages' cache keys too.
    READS_STORES = True

    def __init__(
        self,
        config: Config,
        doc_status: DocStatusPort | None = None,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ) -> None:
        super().__init__(PipelineStageType.SUMMARY_TREE, config, providers)
        self._doc_status = doc_status
        # --force-rebuild asks the model for every summary again.
        self.reuse_stored = reuse_stored
        self.settings = TreeSettings.from_config(config.indexing.summary_tree)
        self._summarizer: TreeSummarizer | None = None

    @property
    def summarizer(self) -> TreeSummarizer:
        if self._summarizer is None:
            self._summarizer = TreeSummarizer(self.config, providers=self.providers)
        return self._summarizer

    def _allows_empty_output(self, context: PipelineContext) -> bool:
        # A corpus of at most min_nodes_to_cluster text units has no summary
        # to build, and a tree already current builds none.
        return True

    def _suffixes(self, context: PipelineContext) -> list[str]:
        """The suffixes the run writes: its text units', and the configured
        one (a run with nothing to index still checks its tree)."""
        units = context.translated_units or context.text_units or []
        configured = (
            self.config.processing.document_parsing.index_value
            or Constants.DEFAULT_SUFFIX.value
        )
        return sorted({*(BaseIndexer.get_suffix(unit) for unit in units), configured})

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        indexing = next(
            (
                r
                for r in context.stage_results
                if r.stage_name == PipelineStageType.INDEXING.value
            ),
            None,
        )
        if indexing is not None and indexing.status == PipelineStageStatus.FAILED:
            logger.warning(
                "Not building the summary tree: the indexing stage failed, so "
                "the stores may not hold the run's text units"
            )
            return 0, 0, {"tree_skipped": "indexing_failed"}
        indexer = self.providers.build_vector_indexer()
        try:
            totals: dict[str, Any] = defaultdict(int)
            notes: dict[str, dict[str, str]] = {}
            for suffix in self._suffixes(context):
                try:
                    metrics = self._build_suffix(indexer, suffix, context)
                except _IntactTreeError:
                    # Failed before writing: the stored tree stays named.
                    raise
                except Exception:
                    self._drop_artifact(indexer, suffix)
                    raise
                for key, value in metrics.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        totals[key] += value
                    else:
                        notes.setdefault(suffix, {})[key] = str(value)
            result: dict[str, Any] = dict(totals)
            if notes:
                result["tree_status"] = notes
            return int(totals["tree_leaves"]), int(totals["summary_nodes"]), result
        finally:
            indexer.close()

    # --- one suffix -------------------------------------------------------------

    def _build_suffix(
        self, indexer: VectorIndexer, suffix: str, context: PipelineContext
    ) -> dict[str, Any]:
        tree = self.config.indexing.summary_tree
        pages = indexer.scan_text_unit_pages(
            suffix,
            fields=["text", self._translated_field, "attributes"],
            page_size=_DIGEST_PAGE,
        )
        if pages is None:
            logger.warning(
                "The vector store (%s) cannot page its text units: no summary "
                "tree is built for '%s'",
                type(indexer).__name__,
                suffix,
            )
            return {"status": "unsupported"}
        # A page of text at a time: only the ids, attributes and text hashes
        # are kept (the size check runs before any text or vector is held).
        rows: list[dict[str, Any]] = []
        entries: list[tuple[str, str, str]] = []
        for page in pages:
            for row in page:
                entries.append(
                    leaf_entry(
                        str(row["id"]), self._leaf_text(row), self._leaf_filters(row)
                    )
                )
                rows.append({"id": row["id"], "attributes": row.get("attributes")})
        if not rows:
            logger.info("'%s' holds no text unit: no summary tree to build", suffix)
            return {"status": "no_text_units"}
        digest = leaf_entries_digest(entries)
        del entries
        fingerprint = summary_tree_fingerprint(self.config)
        stored = indexer.read_summary_tree_record(suffix)
        current = (
            stored is not None
            and stored.get("leaf_digest") == digest
            and stored.get("fingerprint") == fingerprint
            and not stored.get("partial")
            and self.reuse_stored
        )
        incremental = context.incremental_delta is not None and not (
            self.config.indexing.reset
        )
        if current:
            logger.info("The summary tree of '%s' is current", suffix)
            self._record_artifact(indexer, suffix)
            return {"tree_leaves": len(rows), "tree_current": 1}
        # A tree some of whose summaries failed is built again (its
        # warning says the next run does), whatever the refresh.
        if (
            tree.refresh == "on_reset"
            and incremental
            and stored is not None
            and not stored.get("partial")
        ):
            uncovered = self._uncovered(indexer, suffix, rows)
            logger.info(
                "Not rebuilding the summary tree of '%s' (refresh: on_reset): "
                "%d of %d text units are not covered by a summary until a reset",
                suffix,
                uncovered,
                len(rows),
            )
            self._record_artifact(indexer, suffix)
            return {"tree_leaves": len(rows), "tree_uncovered_leaves": uncovered}
        try:
            self._check_tree_size(suffix, rows)
        except PipelineStageError as e:
            if stored is None:
                raise
            # Nothing was written: the tree stored before is intact and
            # still serves the text units it covers.
            logger.warning(
                "The summary tree of '%s' is not rebuilt; the stored one stays "
                "in the build record and serves the text units it covers",
                suffix,
            )
            raise _IntactTreeError(str(e)) from e
        return self._rebuild(indexer, suffix, context, digest, fingerprint)

    def _check_tree_size(self, suffix: str, rows: list[dict[str, Any]]) -> None:
        """Fail before the vectors are read when a tree would hold more
        text units than ``max_leaves_in_memory``."""
        tree = self.config.indexing.summary_tree
        sizes: dict[str | None, int] = defaultdict(int)
        for row in rows:
            sizes[self._tree_of(row.get("attributes") or {})] += 1
        largest = max(sizes.values(), default=0)
        if largest > tree.max_leaves_in_memory:
            raise PipelineStageError(
                f"The summary tree of '{suffix}' would hold {largest} text units, "
                f"more than indexing.summary_tree.max_leaves_in_memory "
                f"({tree.max_leaves_in_memory}) read into memory at once: split "
                "it with indexing.summary_tree.partition_by (document filter "
                "keys), build one tree per document (scope: document), or raise "
                "the limit with the task's memory (about 50 KiB per text unit "
                "at 1,024 dimensions)."
            )

    def _tree_of(self, attributes: dict[str, Any]) -> str | None:
        """The tree a text unit of ``attributes`` belongs to: the corpus's
        (``None``), its ``partition_by`` value tuple's, or its document's
        (its stored file path)."""
        tree = self.config.indexing.summary_tree
        if tree.scope == "document":
            return str(attributes.get("file_path") or attributes.get("file_name") or "")
        if tree.partition_by:
            filters = attributes.get(Constants.FILTERS.value) or {}
            return partition_key([filters.get(key) for key in tree.partition_by])
        return None

    def _rebuild(
        self,
        indexer: VectorIndexer,
        suffix: str,
        context: PipelineContext,
        digest: str,
        fingerprint: str,
    ) -> dict[str, Any]:
        tree = self.config.indexing.summary_tree
        groups = self._leaves(indexer, suffix)
        reusable: dict[str, tuple[str, int]] = {}
        if self.reuse_stored:
            for node in (
                indexer.read_summary_nodes(
                    suffix, fields=["input_key", "text", "n_tokens"]
                )
                or []
            ):
                key, text = node.get("input_key"), node.get("text")
                if isinstance(key, str) and isinstance(text, str):
                    reusable[key] = (text, int(node.get("n_tokens") or 0))
        counts: dict[str, int] = defaultdict(int)
        nodes: list[SummaryNode] = []
        vectors: dict[str, np.ndarray] = {}
        stops: set[str] = set()
        for name, leaves in sorted(groups.items(), key=lambda item: item[0] or ""):
            if tree.scope == "document" and len(leaves) < tree.min_leaves_per_tree:
                continue
            build = self._build(indexer, leaves, suffix, name, reusable, counts)
            nodes.extend(build.nodes)
            vectors.update(build.vectors)
            stops.add(build.stop_reason)
            counts["summaries_failed"] += build.failed
        record = SummaryTreeRecord(
            pipeline_id=context.pipeline_id,
            built_at=datetime.now(UTC),
            leaf_digest=digest,
            fingerprint=fingerprint,
            scope=tree.scope,
            algorithm=tree.algorithm,
            leaves=sum(len(g) for g in groups.values()),
            nodes=len(nodes),
            levels=max((n.level for n in nodes), default=0),
            partial=counts["summaries_failed"] > 0,
            failed_summaries=counts["summaries_failed"],
            reused_summaries=counts["reused"],
            package_version=_package_version(),
        )
        stats = indexer.write_summary_tree(
            nodes, vectors, suffix, record.model_dump(mode="json")
        )
        if stats.failed_items:
            raise PipelineStageError(
                f"Writing the summary tree of '{suffix}' failed ({stats.failed_items} "
                f"of {len(nodes)} nodes; first error: "
                f"{stats.errors[0] if stats.errors else 'unknown'}): the previous "
                "tree, without the summaries of removed text, is kept."
            )
        self._record_artifact(indexer, suffix)
        if record.partial:
            logger.warning(
                "%d summaries of the tree of '%s' failed; their children moved "
                "up a level. The next run builds them again.",
                record.failed_summaries,
                suffix,
            )
        logger.info(
            "Built the summary tree of '%s': %d nodes over %d text units, %d "
            "levels (%d summaries reused, %d asked)",
            suffix,
            record.nodes,
            record.leaves,
            record.levels,
            counts["reused"],
            counts["asked"],
        )
        return {
            "tree_leaves": record.leaves,
            "summary_nodes": record.nodes,
            "tree_levels": record.levels,
            "summaries_reused": counts["reused"],
            "summaries_asked": counts["asked"],
            "summaries_failed": counts["summaries_failed"],
            "tree_rebuilt": 1,
            "stop": ",".join(sorted(stops)) or "empty",
        }

    def _build(
        self,
        indexer: VectorIndexer,
        leaves: Sequence[Leaf],
        suffix: str,
        partition: str | None,
        reusable: dict[str, tuple[str, int]],
        counts: dict[str, int],
    ) -> TreeBuild:
        def realize(clusters: list[Cluster]) -> list[Realized | None]:
            summaries = self.summarizer.summarize(clusters, reusable)
            made = [s for s in summaries if s is not None]
            counts["reused"] += sum(s.reused for s in made)
            counts["asked"] += sum(s is None or not s.reused for s in summaries)
            embedded = (
                indexer.embed_summary_texts([s.text for s in made], suffix)
                if made
                else []
            )
            vectors = iter(embedded or [None] * len(made))
            out: list[Realized | None] = []
            for summary in summaries:
                if summary is None:
                    out.append(None)
                    continue
                vector = next(vectors)
                out.append(
                    None
                    if vector is None
                    else Realized(
                        text=summary.text,
                        vector=vector,
                        n_tokens=summary.n_tokens,
                        input_key=summary.input_key,
                    )
                )
            return out

        return build_tree(
            leaves,
            settings=self.settings,
            realize=realize,
            language=self.summarizer.language,
            partition_name=partition,
            attributes={Constants.INDEX.value: suffix},
        )

    def _leaves(
        self, indexer: VectorIndexer, suffix: str
    ) -> dict[str | None, list[Leaf]]:
        """The suffix's stored text units with their vectors (float32), by
        tree (``_tree_of``)."""
        translated = self._translated_field
        rows = (
            indexer.scan_text_units(
                suffix,
                fields=["text", translated, "text_embedding", "n_tokens", "attributes"],
                page_size=_VECTOR_PAGE,
            )
            or []
        )
        groups: dict[str | None, list[Leaf]] = defaultdict(list)
        skipped = 0
        for index, row in enumerate(rows):
            # Each row is let go once it is a leaf: the JSON it was read as
            # is not kept beside the leaves.
            rows[index] = {}
            vector = row.get("text_embedding")
            if isinstance(vector, list):
                vector = np.asarray(vector, dtype=np.float32)
            text = self._leaf_text(row)
            if not isinstance(vector, np.ndarray) or not vector.size or not text:
                skipped += 1
                continue
            # The stored count is the source text's: a translation is
            # counted in its own script.
            n_tokens = 0 if row.get(translated) else int(row.get("n_tokens") or 0)
            groups[self._tree_of(row.get("attributes") or {})].append(
                Leaf(
                    id=str(row["id"]),
                    text=text,
                    vector=vector,
                    n_tokens=n_tokens or self.summarizer.count_tokens(text),
                    filters=self._leaf_filters(row),
                )
            )
        if skipped:
            logger.warning(
                "%d text units of '%s' hold no vector or text and are left out "
                "of the summary tree",
                skipped,
                suffix,
            )
        return dict(groups)

    @property
    def _translated_field(self) -> str:
        """The text-unit field holding the translation into the index
        language."""
        return f"translated_text_{index_language(self.config)}"

    def _leaf_text(self, row: dict[str, Any]) -> str:
        """The text a stored text unit is a leaf of the tree with: its
        translation, when it holds one, else its text."""
        return str(row.get(self._translated_field) or row.get("text") or "")

    @staticmethod
    def _leaf_filters(row: dict[str, Any]) -> dict[str, Any]:
        """A stored text unit's document filters."""
        filters = (row.get("attributes") or {}).get(Constants.FILTERS.value) or {}
        return filters if isinstance(filters, dict) else {}

    def _uncovered(
        self, indexer: VectorIndexer, suffix: str, rows: list[dict[str, Any]]
    ) -> int:
        """The text units no level-1 node of the live tree covers."""
        covered = {
            unit
            for node in indexer.read_summary_nodes(
                suffix, fields=["level", "text_unit_ids"]
            )
            or []
            if node.get("level") == 1
            for unit in node.get("text_unit_ids") or []
        }
        return sum(str(row["id"]) not in covered for row in rows)

    def _record_artifact(self, indexer: VectorIndexer, suffix: str) -> None:
        """Name the summary tree in the suffix's build record, which the
        indexing stage wrote before the tree existed (only for a run that
        builds it: ``raptor`` selected)."""
        if IndexArtifact.SUMMARY_TREE not in artifacts_to_build(self.config):
            return
        if indexer.add_build_artifacts([IndexArtifact.SUMMARY_TREE], suffix):
            logger.info("The build record of '%s' now names its summary tree", suffix)

    def _drop_artifact(self, indexer: VectorIndexer, suffix: str) -> None:
        """Stop naming the summary tree in the suffix's build record, which
        the indexing stage kept naming it (``IndexingStage._record_build``):
        its build failed, so the tree may not cover the stored text units
        until a run builds it. Best-effort: the stage fails either way."""
        try:
            if indexer.drop_build_artifacts([IndexArtifact.SUMMARY_TREE], suffix):
                logger.warning(
                    "The build record of '%s' no longer names its summary tree "
                    "(its build failed): raptor queries are refused until a run "
                    "builds it",
                    suffix,
                )
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - the build's error is raised
            logger.warning(
                "Could not drop the summary tree from the build record of '%s': %s",
                suffix,
                e,
            )


class _IntactTreeError(PipelineStageError):
    """A tree build that failed before writing anything (its size check),
    over a stored tree it leaves as it was: the build record keeps naming
    it."""


def _package_version() -> str | None:
    try:
        return version("unified-kg-rag-on-aws")
    except PackageNotFoundError:
        return None
