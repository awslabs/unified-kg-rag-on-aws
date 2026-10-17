# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import time
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any, NamedTuple

from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.adapters.storage.embedding_settings import (
    META_BUILD,
    META_DIMENSION,
    META_MODEL,
    resolve_embedding_dimension,
    stored_embedding_settings,
)
from unified_kg_rag.adapters.storage.search_engine import SearchEngineClient
from unified_kg_rag.domain.ingestion.community_hierarchy import max_cut_depth
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Constants,
    Entity,
    IndexBuildRecord,
    Relationship,
    TextUnit,
    model_id_value,
)
from unified_kg_rag.ports.indexer import IndexingStats, VectorIndexer
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.utils.common import compute_hash
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

if TYPE_CHECKING:
    import boto3

    from unified_kg_rag.adapters.aws.embedding_cache import S3EmbeddingCache

logger = get_logger(__name__)

# Texts per embedding request, and requests in flight at once (Bedrock
# embedding calls are IO-bound). In the real-AWS E2E 16 requests in flight
# embedded 1,600 texts in half the time 8 took, with no throttling.
_EMBED_BATCH_SIZE = 50
_EMBED_WORKERS = 16
# Items embedded and bulk-written at a time: one round of the concurrent
# embedding requests. Holding a whole item type's vectors as float lists
# before one bulk call took gigabytes on a large corpus.
_WRITE_SLICE_SIZE = _EMBED_BATCH_SIZE * _EMBED_WORKERS
# Ids per read-back search (well under index.max_result_window).
_READ_BATCH_SIZE = 1000

# Whether the doc-status registry tracks the index's documents: False when
# a run with the registry off built it (see check_index_settings).
_META_DOC_STATUS = "doc_status_registry"


class _ItemSpec(NamedTuple):
    """How one artifact type is embedded and written."""

    name: str
    alias_prefix: str
    mapping: Callable[[], dict[str, Any]]
    # One text per embedding field of the document, in prepare_doc's order.
    extractors: list[Callable[[Any], str]]
    prepare_doc: Callable[[Any, tuple[list[float], ...]], dict[str, Any]]


class OpenSearchIndexer(VectorIndexer):
    # Recorded in the _meta of the indices a full build creates (None records
    # nothing); see set_doc_status_tracking.
    _doc_status_tracked: bool | None = None

    def __init__(
        self,
        config: Config,
        boto_session: "boto3.Session | None" = None,
        *,
        embedding_factory: EmbeddingFactoryPort,
        opensearch_client: SearchEngineClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self.opensearch_config = self.config.indexing.opensearch
        # Kept here, not read back through the client: the S3 embedding cache
        # needs it whichever search engine is injected.
        self.boto_session = boto_session
        self.opensearch_client: SearchEngineClient = (
            opensearch_client
            if opensearch_client is not None
            else OpenSearchClient(config, boto_session)
        )
        self.embedding_factory = embedding_factory
        self._embedding_dimension = self._resolve_embedding_dimension()
        self.embedding_model = self.embedding_factory.get_model(
            model_id=self.opensearch_config.embedding_model_id,
            dimensions=self._embedding_dimension,
        )
        self.target_language = self.config.processing.translation.target_language.value
        self.analyzer = self._analyzer_for(self.target_language)
        # Optional S3-persisted cache so unchanged text is not re-embedded across
        # separate runs/phases (each Fargate phase is a fresh process). Loaded
        # lazily on first embed; best-effort (an S3 error leaves it in memory).
        self._s3_embedding_cache = self._build_s3_embedding_cache()

    def _analyzer_for(self, language: str) -> str:
        return self.opensearch_config.language_analyzers.get(
            language, self.opensearch_config.default_analyzer
        )

    def _build_s3_embedding_cache(self) -> "S3EmbeddingCache | None":
        """Construct the optional S3-persisted embedding cache, if enabled."""
        if not self.opensearch_config.persist_embedding_cache:
            return None
        bucket = self.config.aws.s3.bucket_name
        if not bucket:
            logger.warning(
                "persist_embedding_cache is on but aws.s3.bucket_name is unset; "
                "embedding cache will be in-process only."
            )
            return None
        from unified_kg_rag.adapters.aws.embedding_cache import S3EmbeddingCache

        return S3EmbeddingCache(
            bucket_name=bucket,
            key=self.opensearch_config.embedding_cache_s3_key,
            model_id=model_id_value(self.opensearch_config.embedding_model_id),
            dimension=self._embedding_dimension,
            boto_session=self._s3_session(),
            encryption=self.config.aws.s3.encryption,
        )

    def _s3_session(self) -> "boto3.Session":
        """The given session, else the default one ``OpenSearchClient`` makes."""
        if self.boto_session is None:
            import boto3

            self.boto_session = boto3.Session(
                profile_name=self.config.aws.profile_name,
                region_name=self.config.aws.region_name,
            )
        return self.boto_session

    def _resolve_embedding_dimension(self) -> int:
        return resolve_embedding_dimension(
            self.embedding_factory, self.opensearch_config
        )

    def close(self) -> None:
        """Release the search-engine client's connections (best-effort)."""
        self.opensearch_client.close()

    def _index_prefixes(self) -> list[str]:
        return [spec.alias_prefix for spec in self._specs()]

    def _embedding_meta(self) -> dict[str, Any]:
        """What every index records of the embeddings it holds (``_meta``)."""
        return {
            META_MODEL: model_id_value(self.opensearch_config.embedding_model_id),
            META_DIMENSION: self._embedding_dimension,
        }

    def set_doc_status_tracking(self, tracked: bool) -> None:
        """Record in the indices built from now on whether the doc-status
        registry tracks their documents (see ``check_index_settings``)."""
        self._doc_status_tracked = tracked

    def _mapping_funcs(self) -> dict[str, Callable[[], dict[str, Any]]]:
        return {spec.alias_prefix: spec.mapping for spec in self._specs()}

    @staticmethod
    def _text_field_mismatches(
        expected: dict[str, Any], live: dict[str, Any]
    ) -> list[str]:
        """The analyzed text fields of ``expected`` that ``live`` maps otherwise.

        A field the live index does not map yet would be mapped as a keyword
        by the ``strings_as_keywords`` dynamic template on the first write.
        """
        mismatches = []
        for name, field in expected.items():
            if field.get("type") != "text":
                continue
            analyzer = field.get("analyzer", "standard")
            stored = live.get(name)
            if not isinstance(stored, dict):
                mismatches.append(f"'{name}' is not mapped (needs '{analyzer}')")
            elif stored.get("type") != "text":
                mismatches.append(
                    f"'{name}' is mapped as {stored.get('type')} (needs '{analyzer}')"
                )
            elif stored.get("analyzer", "standard") != analyzer:
                mismatches.append(
                    f"'{name}' uses '{stored.get('analyzer', 'standard')}' "
                    f"(needs '{analyzer}')"
                )
        return mismatches

    def check_index_settings(self, suffix: str | None = None) -> None:
        """Raise when the live indices of ``suffix`` cannot take incremental writes.

        An incremental run writes only its new and changed documents into the
        live indices, so they must have been built for this run's settings:

        - the embedding model and dimension each index records in its
          mapping's ``_meta`` (an index built before it did is checked by the
          dimension of its ``knn_vector`` fields alone): another model's
          vectors would be mixed in, another dimension fails every write;
        - the analyzers of its text fields, which follow the translation
          languages and ``language_analyzers`` and cannot change on a live
          index (a newly translated language would be mapped as a keyword);
        - the doc-status registry: an index built by a run with the registry
          off holds documents the registry does not describe.

        The last two are checked only on an index that records its embedding
        settings. One built before it did (before #231) is given the one
        setting that can be verified, its vector dimension, when its text
        fields match this run's, so later runs check it in full; its model
        stays unrecorded, as which same-dimension model built it cannot be
        told. Such an index carries no registry marker either, so it is taken
        as registry-tracked. Giving it the dimension is best-effort: a failure
        (e.g. a role without ``indices:admin/mapping/put``) is logged and the
        run goes on. An index whose text fields do not match is left as it
        is, with a warning: an older release may have mapped them otherwise.

        Raises:
            ConfigurationError: An index of ``suffix`` does not match.
        """
        expected = self._embedding_meta()
        mapping_funcs = self._mapping_funcs()
        for prefix in self._index_prefixes():
            index_name = self.opensearch_client.get_index_name_by_alias(
                self._get_name(prefix, suffix)
            )
            if not index_name:
                continue
            mappings = self.opensearch_client.get_index_mappings(index_name)
            meta = mappings.get("_meta") or {}
            properties = mappings.get("properties") or {}
            stored_model, dimensions = stored_embedding_settings(mappings)
            if not dimensions <= {expected[META_DIMENSION]} or stored_model not in (
                None,
                expected[META_MODEL],
            ):
                stored = (
                    f"model '{stored_model}'" if stored_model else "an unrecorded model"
                )
                stored_dimensions = ", ".join(str(d) for d in sorted(dimensions))
                raise ConfigurationError(
                    f"The OpenSearch index '{index_name}' holds embeddings of "
                    f"{stored} (dimension {stored_dimensions or 'unknown'}), but "
                    f"this run embeds with '{expected[META_MODEL]}' (dimension "
                    f"{expected[META_DIMENSION]}). An incremental run embeds only "
                    "new and changed documents, so the index would mix the two. "
                    "Re-run once with indexing.reset: true to re-embed the whole "
                    "corpus, or restore the previous embedding model and dimension."
                )
            mismatches = self._text_field_mismatches(
                mapping_funcs[prefix]()["mappings"]["properties"], properties
            )
            if META_MODEL not in meta and META_DIMENSION not in meta:
                self._record_legacy_dimension(index_name, meta, dimensions, mismatches)
                continue
            if mismatches:
                raise ConfigurationError(
                    f"The OpenSearch index '{index_name}' was built for other "
                    f"language settings: {'; '.join(mismatches)}. An incremental "
                    "run writes only new and changed documents into the live "
                    "index, whose field analyzers cannot change. Re-run once with "
                    "indexing.reset: true to rebuild the indices for "
                    "processing.translation (and "
                    "indexing.opensearch.language_analyzers), or restore the "
                    "previous languages."
                )
            if meta.get(_META_DOC_STATUS) is False:
                raise ConfigurationError(
                    f"The OpenSearch index '{index_name}' was built by a run with "
                    "the doc-status registry off, so the registry does not "
                    "describe the documents it holds: an incremental run would "
                    "keep documents the registry does not list and skip ones it "
                    "lists as indexed. Re-run once with indexing.reset: true "
                    "(registry on) to rebuild the stores and the registry, or "
                    "keep the registry off for this suffix."
                )

    def _record_legacy_dimension(
        self,
        index_name: str,
        meta: dict[str, Any],
        dimensions: set[int],
        mismatches: list[str],
    ) -> None:
        """Give an index built before it recorded its settings its dimension.

        See ``check_index_settings``: only when its text fields match this
        run's, and best-effort.
        """
        if mismatches:
            logger.warning(
                "The OpenSearch index '%s' records no embedding settings and its "
                "text fields do not match this run's language settings (%s); "
                "documents written incrementally may be analyzed differently "
                "from the rest. Re-run once with indexing.reset: true to "
                "rebuild it for the current processing.translation and "
                "indexing.opensearch.language_analyzers.",
                index_name,
                "; ".join(mismatches),
            )
            return
        if len(dimensions) != 1:
            return  # no vector field to verify the dimension by
        try:
            self.opensearch_client.put_index_meta(
                index_name, {**meta, META_DIMENSION: next(iter(dimensions))}
            )
        except Exception as e:  # noqa: BLE001 - recording it is optional
            logger.warning(
                "Could not record the embedding dimension in the _meta of the "
                "OpenSearch index '%s' (%s); its dimension is still checked on "
                "every run, its languages and registry state only once recorded.",
                index_name,
                e,
            )

    def record_build(self, record: IndexBuildRecord, suffix: str | None) -> None:
        """Write ``record`` into the ``_meta`` of each live index of ``suffix``.

        Merged into the existing ``_meta`` (the embedding settings stay), so
        it needs ``indices:admin/mapping/put`` like the dimension backfill; a
        failure is logged and the run goes on.
        """
        payload = record.model_dump(mode="json")
        for prefix in self._index_prefixes():
            alias = self._get_name(prefix, suffix)
            try:
                index_name = self.opensearch_client.get_index_name_by_alias(alias)
                if not index_name:
                    continue
                meta = (
                    self.opensearch_client.get_index_mappings(index_name).get("_meta")
                    or {}
                )
                self.opensearch_client.put_index_meta(
                    index_name, {**meta, META_BUILD: payload}
                )
            except Exception as e:  # noqa: BLE001 - the record is optional
                logger.warning(
                    "Could not record the build of pipeline '%s' in the _meta of "
                    "%s (%s); run-eval manifests will not identify this index.",
                    record.pipeline_id,
                    alias,
                    e,
                )

    def clear(self, suffixes: list[str]) -> None:
        if not suffixes:
            return

        aliases_to_delete = [
            self._get_name(prefix, suffix)
            for prefix in self._index_prefixes()
            for suffix in suffixes
        ]

        # Only this namespace's own indices (``<alias>-<timestamp>``), by
        # exact name: ``<alias>-*`` also matches the indices of a sibling
        # namespace (``default-x`` under ``default``, ``acme-eu`` under
        # ``acme``).
        indices_to_delete = sorted(
            {
                name
                for alias in aliases_to_delete
                for name in self.opensearch_client.get_aliases_by_index(f"{alias}-*")
                if self._index_timestamp(alias, name) is not None
            }
        )

        if indices_to_delete:
            self.opensearch_client.delete_indices(indices_to_delete)

        # Deleting an index drops its aliases. Only an alias still on an
        # index is deleted: deleting aliases none of which exists (a
        # namespace never indexed, or a store it never wrote) is a 404.
        for alias in aliases_to_delete:
            remaining = self.opensearch_client.get_indices_by_alias(alias)
            if remaining:
                self.opensearch_client.delete_alias(
                    index_names=remaining, alias_names=[alias]
                )

        logger.info("Cleared OpenSearch indices for '%s'", aliases_to_delete)

    def _specs(self) -> tuple[_ItemSpec, ...]:
        config = self.opensearch_config
        return (
            _ItemSpec(
                "text units",
                config.text_units_index_prefix,
                self._get_text_units_mapping,
                [self._text_unit_embedding_text],
                self._prepare_text_unit_doc,
            ),
            _ItemSpec(
                "entities",
                config.entities_index_prefix,
                self._get_entities_mapping,
                [lambda e: e.name, self._entity_description_text],
                self._prepare_entity_doc,
            ),
            _ItemSpec(
                "community reports",
                config.community_reports_index_prefix,
                self._get_community_reports_mapping,
                [lambda r: r.name, lambda r: r.summary, lambda r: r.full_content],
                self._prepare_community_report_doc,
            ),
            _ItemSpec(
                "relationships",
                config.relationships_index_prefix,
                self._get_relationships_mapping,
                [self._relationship_description_text],
                self._prepare_relationship_doc,
            ),
            _ItemSpec(
                "claims",
                config.claims_index_prefix,
                self._get_claims_mapping,
                [self._claim_description_text],
                self._prepare_claim_doc,
            ),
        )

    def _spec(self, name: str) -> _ItemSpec:
        return next(spec for spec in self._specs() if spec.name == name)

    def index_text_units(
        self, text_units: list[TextUnit], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._index_item_type(text_units, self._spec("text units"), suffixes)

    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._index_item_type(entities, self._spec("entities"), suffixes)

    def _prepare_community_report_doc(
        self, report: CommunityReport, embeddings: tuple
    ) -> dict[str, Any]:
        return {
            **self._prepare_common_doc_properties(report),
            "community_id": report.community_id,
            "name": report.name or "",
            "name_embedding": embeddings[0],
            "summary": report.summary or "",
            "summary_embedding": embeddings[1],
            "full_content": report.full_content or "",
            "full_content_embedding": embeddings[2],
            "rank": report.rank if report.rank is not None else 1.0,
            "rating": report.rating,
            # Community membership provenance (candidate sources of the report's
            # community), so a retrieved report can be filtered by origin. Not
            # report-input provenance and not citation support: the union may
            # include material the prompt's entity cap left out. Reports indexed
            # before these fields existed index them as [] until regenerated.
            "text_unit_ids": list(report.text_unit_ids or []),
            "document_ids": list(report.document_ids or []),
            # The hierarchy cut global search maps over (community_hierarchy).
            "level": report.level,
            "depth": report.depth,
            "max_cut_depth": max_cut_depth(report),
            "parent_community_id": report.parent_community_id,
            "child_community_ids": list(report.child_community_ids),
        }

    def index_community_reports(
        self, reports: list[CommunityReport], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._index_item_type(reports, self._spec("community reports"), suffixes)

    def upsert_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        """Upsert community reports by id into the live index (delta semantics).

        The full-build ``index_community_reports`` does an alias-swap that
        REPLACES the whole index — fatal on an incremental run, where community
        detection only produced reports for the delta subgraph, so a swap would
        wipe every unchanged document's reports. Upsert-by-id preserves them.
        """
        return self._upsert_item_type(reports, self._spec("community reports"))

    # Fallback embedding texts. An item with an empty description would get no
    # embedding, be skipped from the index and counted as a failure (which can
    # trip ``max_failure_rate``). Every stub entity created from a relationship
    # endpoint has no description, so embed a short synthesized surrogate
    # (name/type, endpoints) instead; the stored ``description`` stays as-is.
    @staticmethod
    def _entity_description_text(entity: Entity) -> str:
        if entity.description and entity.description.strip():
            return entity.description
        name = entity.name or ""
        return f"{name} ({entity.type})" if entity.type else name

    @staticmethod
    def _relationship_description_text(rel: Relationship) -> str:
        if rel.description and rel.description.strip():
            return rel.description
        source = rel.source_name or rel.source_id
        target = rel.target_name or rel.target_id
        relation = rel.type or "related to"
        return f"{source} {relation} {target}"

    @staticmethod
    def _claim_description_text(claim: Claim) -> str:
        if claim.description and claim.description.strip():
            return claim.description
        parts = [claim.subject_name or claim.subject_id, claim.type or ""]
        if claim.object_name:
            parts.append(claim.object_name)
        return " ".join(p for p in parts if p)

    def _text_unit_embedding_text(self, unit: TextUnit) -> str:
        if hasattr(unit, "translated_texts") and unit.translated_texts:
            return unit.translated_texts.get(self.target_language) or unit.text
        return unit.text

    def _prepare_text_unit_doc(
        self, unit: TextUnit, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        doc = {
            **self._prepare_common_doc_properties(unit),
            "text": unit.text or "",
            "text_embedding": embeddings[0],
            "n_tokens": unit.n_tokens or 0,
        }
        if unit.community_ids:
            doc["community_ids"] = (
                list(unit.community_ids)
                if isinstance(unit.community_ids, (list | tuple | set))
                else unit.community_ids
            )
        if hasattr(unit, "translated_texts") and unit.translated_texts:
            for language in self.config.processing.translation.translated_languages:
                if translated := unit.translated_texts.get(language):
                    doc[f"translated_text_{language}"] = translated
        return doc

    def _prepare_entity_doc(
        self, entity: Entity, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        return {
            **self._prepare_common_doc_properties(entity),
            "name": entity.name or "",
            "name_embedding": embeddings[0],
            "description": entity.description or "",
            "description_embedding": embeddings[1],
            "type": entity.type or "",
            "rank": entity.rank if entity.rank is not None else 1.0,
            "confidence": (entity.confidence if entity.confidence is not None else 1.0),
            # Chunk lineage: lets LightRAG mix follow entity -> source chunks
            # instead of relying solely on a separate naive vector query.
            "text_unit_ids": list(entity.text_unit_ids or []),
        }

    @staticmethod
    def _prepare_relationship_doc(
        rel: Relationship, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        return {
            "id": rel.id,
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "source_name": rel.source_name or "",
            "target_name": rel.target_name or "",
            "description": rel.description or "",
            "description_embedding": embeddings[0],
            "weight": rel.weight if rel.weight is not None else 1.0,
            "rank": rel.rank if rel.rank is not None else 1.0,
            # Chunk lineage: lets LightRAG mix follow relationship -> source
            # chunks (LightRAG's _find_related_text_unit_from_relationships).
            "text_unit_ids": list(rel.text_unit_ids or []),
        }

    def index_relationships(
        self, relationships: list[Relationship], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Embed and index relationship descriptions (LightRAG global retrieval).

        Relationships are embedded as first-class vectors so high-level keywords
        can retrieve relations directly, as in LightRAG's ``relationships_vdb``.
        """
        return self._index_item_type(
            relationships, self._spec("relationships"), suffixes
        )

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Upsert relationship vectors by id into the live index (delta)."""
        return self._upsert_item_type(relationships, self._spec("relationships"))

    @staticmethod
    def _prepare_claim_doc(
        claim: Claim, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        return {
            "id": claim.id,
            "subject_id": claim.subject_id,
            "subject_name": claim.subject_name or "",
            "object_id": claim.object_id,
            "object_name": claim.object_name or "",
            "type": claim.type or "",
            "status": claim.status or "",
            "description": claim.description or "",
            "description_embedding": embeddings[0],
            "source_text": claim.source_text or "",
            # Read back by the cross-run claim merge (read_claims).
            "text_unit_ids": list(claim.text_unit_ids or []),
        }

    def index_claims(
        self, claims: list[Claim], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Embed and index claim (covariate) descriptions as searchable vectors.

        Connects claim extraction to retrieval: claims become first-class
        searchable artifacts (mirrors the relationship vector index).
        """
        return self._index_item_type(claims, self._spec("claims"), suffixes)

    def upsert_claims(self, claims: list[Claim]) -> IndexingStats:
        """Upsert claim vectors by id into the live index (delta)."""
        return self._upsert_item_type(claims, self._spec("claims"))

    def _read_sources(
        self, alias_prefix: str, ids: list[str], suffix: str | None
    ) -> list[dict[str, Any]]:
        """The stored documents (without embeddings) of ``ids`` in ``suffix``'s
        live index of ``alias_prefix``; none without a live index. Raises when
        the read fails."""
        alias_name = self._get_name(alias_prefix, suffix)
        if not ids or not self.opensearch_client.get_index_name_by_alias(alias_name):
            return []
        sources: list[dict[str, Any]] = []
        for start in range(0, len(ids), _READ_BATCH_SIZE):
            batch = ids[start : start + _READ_BATCH_SIZE]
            response = self.opensearch_client.search(
                index=alias_name,
                body={
                    "size": len(batch),
                    "query": {"ids": {"values": batch}},
                    "_source": {"excludes": ["*_embedding"]},
                },
            )
            sources.extend(
                {"id": hit["_id"], **(hit.get("_source") or {})}
                for hit in response.get("hits", {}).get("hits", [])
            )
        return sources

    def read_text_unit_ids(
        self, ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """The ids among ``ids`` in ``suffix``'s live text-unit index."""
        return [
            source["id"]
            for source in self._read_sources(
                self.opensearch_config.text_units_index_prefix, ids, suffix
            )
        ]

    def read_claims(self, ids: list[str], suffix: str | None = None) -> list[Claim]:
        """Read stored claims by id from ``suffix``'s live claims index.

        A claim indexed before claims stored their ``text_unit_ids`` reads
        back without them. The stored document does not hold the claim's
        ``attributes``, so each claim is tagged with ``suffix`` (the index
        it was read from): a write of the claim, e.g. the
        ``update_claim_lineage`` of a document deletion, goes back to that
        index rather than the default one. Raises when the read fails.
        """
        attributes = {Constants.INDEX.value: suffix} if suffix else {}
        return [
            Claim.model_validate(
                {
                    "id": source["id"],
                    "subject_id": source.get("subject_id") or "",
                    "subject_name": source.get("subject_name") or "",
                    "object_id": source.get("object_id"),
                    "object_name": source.get("object_name") or "",
                    "type": source.get("type") or "",
                    "status": source.get("status") or None,
                    "description": source.get("description") or None,
                    "source_text": source.get("source_text") or None,
                    "text_unit_ids": list(source.get("text_unit_ids") or []),
                    "attributes": dict(attributes),
                }
            )
            for source in self._read_sources(
                self.opensearch_config.claims_index_prefix, ids, suffix
            )
        ]

    def read_entity_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        """Stored entity descriptions by id (untruncated, unlike Neptune's)."""
        return self._descriptions(
            self.opensearch_config.entities_index_prefix, ids, suffix
        )

    def read_relationship_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        """Stored relationship descriptions by id (see
        :meth:`read_entity_descriptions`)."""
        return self._descriptions(
            self.opensearch_config.relationships_index_prefix, ids, suffix
        )

    def read_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Stored extraction records by key (see the port)."""
        sources = self._read_sources(
            self.opensearch_config.extraction_records_index_prefix, keys, suffix
        )
        return {
            source["id"]: source["record"]
            for source in sources
            if isinstance(source.get("record"), dict)
        }

    def write_extraction_records(
        self, records: dict[str, dict[str, Any]], suffix: str | None = None
    ) -> IndexingStats:
        """Upsert extraction records by key into ``suffix``'s store index.

        The index has no embeddings and indexes no field but the id: the
        record is kept in ``_source`` only. It is not one of the artifact
        indices (``_index_prefixes``), so a full build, ``clear`` and the
        index-settings check leave it alone.
        """
        stats = IndexingStats(total_items=len(records))
        if not records:
            return stats
        alias_name = self._get_name(
            self.opensearch_config.extraction_records_index_prefix, suffix
        )
        try:
            index_name = self.opensearch_client.get_index_name_by_alias(alias_name)
            if not index_name:
                index_name = self._new_index_name(
                    alias_name,
                    self.opensearch_config.extraction_records_index_prefix,
                    suffix or Constants.DEFAULT_SUFFIX.value,
                )
                self.opensearch_client.create_index(
                    index_name, self._get_extraction_records_mapping()
                )
                self.opensearch_client.update_alias(alias_name, index_name)
            documents = [
                {"id": key, "record": record} for key, record in records.items()
            ]
            failed: list[str] = []
            for start in range(0, len(documents), _WRITE_SLICE_SIZE):
                response = self.opensearch_client.bulk_index(
                    index_name, documents[start : start + _WRITE_SLICE_SIZE]
                )
                failed += [
                    str(next(iter(item.values())).get("_id"))
                    for item in response.get("items", [])
                ]
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Writing extraction records to '%s' failed: %s", alias_name, e)
            stats.add_error(str(e), len(records), ids=list(records))
            return stats
        if failed:
            stats.add_error("bulk write rejected", len(failed), ids=failed)
        stats.successful_items = len(records) - len(failed)
        return stats

    def _get_extraction_records_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                key: value
                for key, value in self._get_base_mapping({})["settings"].items()
                if not key.startswith("index.knn")
            },
            "mappings": {
                "dynamic": False,
                "properties": {
                    "id": {"type": "keyword"},
                    "record": {"type": "object", "enabled": False},
                },
            },
        }

    def _descriptions(
        self, alias_prefix: str, ids: list[str], suffix: str | None
    ) -> dict[str, str]:
        return {
            source["id"]: source["description"]
            for source in self._read_sources(alias_prefix, ids, suffix)
            if source.get("description")
        }

    def update_claim_lineage(self, claims: list[Claim]) -> IndexingStats:
        """Set stored claims' ``text_unit_ids`` in place (no re-embedding)."""
        return self._update_fields(
            claims,
            self.opensearch_config.claims_index_prefix,
            lambda c: {"text_unit_ids": list(c.text_unit_ids or [])},
        )

    def upsert_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        """Upsert text units by id into the live index (delta semantics)."""
        return self._upsert_item_type(text_units, self._spec("text units"))

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Upsert entities by id into the live index (delta semantics)."""
        return self._upsert_item_type(entities, self._spec("entities"))

    def update_entity_lineage(self, entities: list[Entity]) -> IndexingStats:
        """Set stored entities' ``text_unit_ids`` in place (no re-embedding)."""
        return self._update_fields(
            entities,
            self.opensearch_config.entities_index_prefix,
            lambda e: {"text_unit_ids": list(e.text_unit_ids or [])},
        )

    def update_relationship_lineage(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Set stored relationships' ``text_unit_ids`` and ``weight`` in place."""
        return self._update_fields(
            relationships,
            self.opensearch_config.relationships_index_prefix,
            lambda r: {
                "text_unit_ids": list(r.text_unit_ids or []),
                "weight": r.weight if r.weight is not None else 1.0,
            },
        )

    def _update_fields(
        self,
        items: list[Any],
        alias_prefix: str,
        fields: Callable[[Any], dict[str, Any]],
    ) -> IndexingStats:
        """Partially update ``items`` by id in their suffix's live index.

        A suffix without a live index, or an item without a document, has
        nothing to update.
        """
        stats = IndexingStats(total_items=len(items))
        for suffix, chunk in self._group_items_by_suffix(items).items():
            alias_name = self._get_name(alias_prefix, suffix)
            try:
                index_name = self.opensearch_client.get_index_name_by_alias(alias_name)
                if not index_name:
                    stats.add_success(len(chunk))
                    continue
                response = self.opensearch_client.bulk_update(
                    index_name,
                    [{"id": item.id, **fields(item)} for item in chunk],
                    refresh=self.opensearch_config.refresh_after_batch,
                )
                failed = [next(iter(r.values())) for r in response.get("items", [])]
                stats.add_success(len(chunk) - len(failed))
                if failed:
                    stats.add_error(
                        f"Bulk update errors: {len(failed)}",
                        len(failed),
                        ids=[str(f["_id"]) for f in failed if "_id" in f],
                    )
            except Exception as e:
                stats.add_error(
                    f"Update of '{alias_name}' failed: {e}",
                    len(chunk),
                    ids=[item.id for item in chunk],
                )
        return stats

    def delete_by_id(
        self, ids: list[str], alias_prefix: str, suffix: str
    ) -> IndexingStats:
        """Delete documents by id from the live aliased index (delta removals)."""
        stats = IndexingStats(total_items=len(ids))
        if not ids:
            return stats

        alias_name = self._get_name(alias_prefix, suffix)
        index_name = self.opensearch_client.get_index_name_by_alias(alias_name)
        if not index_name:
            logger.info("No live index for alias '%s'; nothing to delete", alias_name)
            return stats

        try:
            response = self.opensearch_client.bulk_delete(
                index_name, ids, refresh=self.opensearch_config.refresh_after_batch
            )
            # bulk_delete returns {"errors", "items"} where items is ONLY the
            # error results (not-found deletes are counted as success), so
            # failed + success == len(ids) == total_items exactly.
            failed = len(response.get("items", []))
            stats.add_success(len(ids) - failed)
            if failed:
                stats.add_error(f"Bulk delete errors: {failed}", failed)
        except Exception as e:
            stats.add_error(f"Delete-by-id failed: {e}", len(ids))
        return stats

    def delete_document_artifacts(
        self,
        ids: list[str],
        suffix: str,
        extra_relationship_ids: list[str] | None = None,
    ) -> dict[str, IndexingStats]:
        """Delete a document's artifacts from every OpenSearch index, by id.

        Encapsulates the per-index fan-out (text-units / entities / relationships
        / claims / community-reports) so callers stay backend-agnostic. The
        relationship index additionally receives ``extra_relationship_ids``
        (incident edges orphaned by an entity drop) so no dangling relationship
        document survives.
        """
        results: dict[str, IndexingStats] = {}
        if not ids:
            return results

        rel_prefix = self.opensearch_config.relationships_index_prefix
        orphan_ids = set(extra_relationship_ids or [])
        for prefix in (
            self.opensearch_config.text_units_index_prefix,
            self.opensearch_config.entities_index_prefix,
            rel_prefix,
            self.opensearch_config.claims_index_prefix,
            self.opensearch_config.community_reports_index_prefix,
        ):
            delete_ids = (
                sorted(set(ids) | orphan_ids)
                if prefix == rel_prefix and orphan_ids
                else ids
            )
            results[f"opensearch_delete_{prefix}_{suffix}"] = self.delete_by_id(
                delete_ids, prefix, suffix
            )
        return results

    def _upsert_item_type(self, items: list[Any], spec: _ItemSpec) -> IndexingStats:
        """Index documents by id into the live aliased index without rebuilding it.

        Unlike ``_index_item_type`` (create new index -> alias swap -> delete old),
        this writes/overwrites by document id into the index the alias already
        points at, creating a first index + alias if none exists. This makes
        incremental delta updates additive instead of a full rebuild.
        """
        if not items:
            return IndexingStats()

        logger.info("Upserting %d %s", len(items), spec.name)
        total_stats = IndexingStats()

        for suffix, chunk_items in self._group_items_by_suffix(items).items():
            alias_name = self._get_name(spec.alias_prefix, suffix)
            index_name = self.opensearch_client.get_index_name_by_alias(alias_name)

            try:
                if not index_name:
                    # First-ever write for this alias: create one index + alias.
                    index_name = self._new_index_name(
                        alias_name, spec.alias_prefix, suffix
                    )
                    mapping = spec.mapping()
                    # Only an incremental run upserts: the registry tracks it.
                    mapping["mappings"].setdefault("_meta", {})[_META_DOC_STATUS] = True
                    self.opensearch_client.create_index(index_name, mapping)
                    self.opensearch_client.update_alias(alias_name, index_name)

                total_stats.merge(self._embed_and_write(index_name, chunk_items, spec))
            except Exception as e:
                logger.error(
                    "Failed to upsert %s (suffix=%s): %s", spec.name, suffix, e
                )
                total_stats.add_error(
                    str(e), len(chunk_items), ids=[item.id for item in chunk_items]
                )

        self._flush_embedding_cache()
        return total_stats

    @staticmethod
    def _prepare_common_doc_properties(item: Any) -> dict[str, Any]:
        doc = {"id": item.id}

        if hasattr(item, "attributes") and item.attributes:
            doc["attributes"] = item.attributes
            if filters := item.attributes.get(Constants.FILTERS.value, {}):
                for key, value in filters.items():
                    if value is not None:
                        doc[f"{Constants.ATTRIBUTE_PREFIX.value}_{key}"] = value

        return doc

    def _index_item_type(
        self, items: list[Any], spec: _ItemSpec, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Build a new index per suffix and swap the alias onto it.

        A suffix in ``suffixes`` without items gets an empty index, so the
        alias stops serving the previous build's documents.
        """
        if not items and not suffixes:
            return IndexingStats()

        logger.info("Indexing %s %s", len(items), spec.name)
        total_stats = IndexingStats()

        for suffix, chunk_items in self._group_items_by_suffix(items, suffixes).items():
            alias_name = self._get_name(spec.alias_prefix, suffix)
            index_name = ""

            try:
                index_name = self._new_index_name(alias_name, spec.alias_prefix, suffix)
                self.opensearch_client.create_index(
                    index_name, spec.mapping(), exist_ok=False
                )

                written = self._embed_and_write(index_name, chunk_items, spec)
                total_stats.merge(written)

                # Swap onto the new index when it holds something, or when the
                # rebuild has nothing of this type for the suffix.
                if not chunk_items or written.successful_items > 0:
                    self.opensearch_client.update_alias(
                        alias_name, index_name, remove_pattern=f"{alias_name}-*"
                    )
                    self._reap_stale_indices(alias_name, index_name)
                else:
                    # Nothing was written, so the alias stays on the previous
                    # index; drop the unused new one instead of leaking it.
                    self._drop_unswapped_index(alias_name, index_name)

            except Exception as e:
                logger.error("Failed to index %s (suffix=%s): %s", spec.name, suffix, e)
                # A failed empty replacement still fails the run.
                total_stats.add_error(
                    str(e),
                    max(1, len(chunk_items)),
                    ids=[item.id for item in chunk_items],
                )
                if index_name:
                    self._drop_unswapped_index(alias_name, index_name)

        if total_stats.failed_items > 0:
            logger.warning(
                "Indexing %s completed: %s succeeded, %s failed",
                spec.name,
                total_stats.successful_items,
                total_stats.failed_items,
            )
        else:
            logger.info(
                "Successfully indexed %s %s",
                total_stats.successful_items,
                spec.name,
            )

        self._flush_embedding_cache()
        return total_stats

    def _embed_and_write(
        self, index_name: str, items: list[Any], spec: _ItemSpec
    ) -> IndexingStats:
        """Embed ``items`` and bulk-write them into ``index_name``, a slice at a
        time, so only one slice's vectors are held at once. The index is
        refreshed once, after the last slice."""
        stats = IndexingStats()
        for start in range(0, len(items), _WRITE_SLICE_SIZE):
            chunk = items[start : start + _WRITE_SLICE_SIZE]
            embeddings = self._generate_embeddings(chunk, spec.extractors)
            docs, failed_ids = self._prepare_documents(
                chunk, embeddings, spec.prepare_doc
            )
            if failed_ids:
                stats.add_error(
                    f"Failed to prepare {len(failed_ids)} documents",
                    len(failed_ids),
                    ids=failed_ids,
                )
            if docs:
                last = start + _WRITE_SLICE_SIZE >= len(items)
                stats.merge(self._perform_indexing(index_name, docs, refresh=last))
        return stats

    @staticmethod
    def _index_timestamp(alias_name: str, index_name: str) -> str | None:
        """Timestamp of ``<alias>-<YYYYmmddHHMMSSfff>``, or ``None`` if not that shape.

        Requiring the 17-digit timestamp (or the 14-digit one of indices built
        before millisecond names) keeps ``<alias>-*`` from matching a sibling
        alias's indices (e.g. suffix ``default-2`` under ``default``). The two
        widths compare as strings in build order: a 14-digit name sorts before
        every 17-digit name of the same second.
        """
        match = re.fullmatch(
            rf"{re.escape(alias_name)}-(\d{{17}}|\d{{14}})", index_name
        )
        return match.group(1) if match else None

    @staticmethod
    def _timestamp_ms(timestamp: str) -> int:
        return int(timestamp) * 1000 if len(timestamp) == 14 else int(timestamp)

    def _new_index_name(self, alias_name: str, alias_prefix: str, suffix: str) -> str:
        """A timestamped index name for ``alias_name`` that no index has yet.

        Two builds in the same millisecond (or under a pinned clock) would
        otherwise get the live index's name, and the replacement would be
        written into it. A taken name moves forward a millisecond at a time
        until it is free, so it still sorts after the index it collided with
        and the stale-index reaper removes that one after the swap.
        """
        name = self._get_name(alias_prefix, suffix, add_timestamp=True)
        taken = set(self.opensearch_client.get_aliases_by_index(f"{alias_name}-*"))
        timestamp = self._index_timestamp(alias_name, name)
        if timestamp is None:
            return name
        millis = self._timestamp_ms(timestamp)
        while name in taken:
            millis += 1
            name = f"{alias_name}-{millis:017d}"
        return name

    def _reap_stale_indices(self, alias_name: str, live_index: str) -> None:
        """Delete this alias's older timestamped indices after a successful swap.

        Matches indices by *name* (an alias lookup on ``<alias>-*`` matches
        nothing, which is how stale indices used to leak). Only indices of this
        alias's exact naming scheme that are older than ``live_index`` and no
        longer carry the alias are deleted, so a sibling suffix's indices and a
        newer concurrent build are left alone. A cleanup failure is logged, not
        raised: the swap already succeeded and the live index must survive.
        """
        live_ts = self._index_timestamp(alias_name, live_index)
        if live_ts is None:
            return
        try:
            candidates = self.opensearch_client.get_aliases_by_index(f"{alias_name}-*")
            stale = sorted(
                name
                for name, aliases in candidates.items()
                if name != live_index
                and alias_name not in aliases
                and (ts := self._index_timestamp(alias_name, name)) is not None
                and ts < live_ts
            )
            if stale:
                self.opensearch_client.delete_indices(stale)
                logger.info(
                    "Deleted %s stale index(es) of '%s'", len(stale), alias_name
                )
        except Exception as e:
            logger.warning(
                "Failed to delete stale indices of '%s' (live index '%s' kept): %s",
                alias_name,
                live_index,
                e,
            )

    def _drop_unswapped_index(self, alias_name: str, index_name: str) -> None:
        """Delete a new index the alias was not swapped onto.

        Skipped if the alias already points at ``index_name`` (a new name that
        turned out to be taken), so the live index is never deleted on a
        failed run. Best-effort: a failure is logged, not raised.
        """
        try:
            if index_name in self.opensearch_client.get_indices_by_alias(alias_name):
                return
            self.opensearch_client.delete_indices([index_name])
        except Exception as e:
            logger.warning("Failed to delete unused index '%s': %s", index_name, e)

    def _generate_embeddings(
        self, items: list[Any], extractors: list[Callable]
    ) -> list[tuple]:
        all_embeddings = [
            self._batch_embed([extractor(item) for item in items])
            for extractor in extractors
        ]
        return list(zip(*all_embeddings, strict=True))

    def _batch_embed(
        self, texts: list[str], batch_size: int = _EMBED_BATCH_SIZE
    ) -> list[list[float] | None]:
        result: list[list[float] | None] = [None] * len(texts)

        # Identical texts of one call are embedded once and fanned back out.
        # The optional S3 cache (loaded once, lazily) persists vectors across
        # runs and phases so unchanged text is not re-embedded. No in-process
        # tier: it held every vector of the run and texts rarely repeat
        # across calls.
        if self._s3_embedding_cache is not None:
            self._s3_embedding_cache.load()

        key_to_indices: dict[str, list[int]] = {}
        unique: list[tuple[str, str]] = []  # (content_hash, text) to embed
        seen_keys: set[str] = set()
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = compute_hash(text, length=32)
            key_to_indices.setdefault(key, []).append(i)
            cached = self._cached_embedding(key)
            if cached is not None:
                result[i] = cached
            else:
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique.append((key, text))

        def _store(key: str, emb: list[float] | None) -> None:
            if emb is not None and self._s3_embedding_cache is not None:
                self._s3_embedding_cache.put(key, emb)
            for idx in key_to_indices[key]:
                result[idx] = emb

        def _embed_batch(
            batch: list[tuple[str, str]],
        ) -> list[tuple[str, list[float] | None]]:
            keys = [k for k, _ in batch]
            batch_texts = [t for _, t in batch]
            try:
                embeddings = self.embedding_model.embed_documents(batch_texts)
                return list(zip(keys, embeddings, strict=True))
            except Exception as e:
                # The embedding wrapper already retried transient Bedrock errors
                # with backoff; re-embedding per item isolates the text(s) that
                # still fail so the rest of the batch is not lost.
                logger.warning(
                    "Batch embedding failed (%s items), retrying individually: %s",
                    len(batch),
                    e,
                )
                out: list[tuple[str, list[float] | None]] = []
                for key, text in batch:
                    try:
                        single_embs = self.embedding_model.embed_documents([text])
                        out.append((key, single_embs[0] if single_embs else None))
                    except Exception as item_error:
                        # Summarized once per call in _batch_embed (WARNING).
                        logger.debug("Failed to embed text '%s': %s", key, item_error)
                        out.append((key, None))
                return out

        # Embedding is IO-bound (Bedrock InvokeModel); fan the batches out across
        # threads instead of issuing them strictly serially. Results are stored
        # back on the calling thread to avoid races on result and the S3 cache.
        batches = [
            unique[start : start + batch_size]
            for start in range(0, len(unique), batch_size)
        ]
        if batches:
            max_workers = min(_EMBED_WORKERS, len(batches))
            failed_keys = 0
            with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
                for pairs in executor.map(_embed_batch, batches):
                    for key, emb in pairs:
                        if emb is None:
                            failed_keys += 1
                        _store(key, emb)
            if failed_keys:
                logger.warning(
                    "Embedding failed for %s of %s unique texts after retries; "
                    "the affected items will be skipped from indexing",
                    failed_keys,
                    len(unique),
                )

        # NOTE: the S3 cache is intentionally NOT flushed here. _batch_embed is
        # called once per extractor per item-type, so flushing here would
        # read-merge-overwrite the whole (growing) S3 object many times per run.
        # Newly-computed vectors are held by the S3 cache and flushed once per
        # item-type by _flush_embedding_cache() at the indexing boundary.
        return result

    def _cached_embedding(self, key: str) -> list[float] | None:
        if self._s3_embedding_cache is None:
            return None
        return self._s3_embedding_cache.get(key)

    def _flush_embedding_cache(self) -> None:
        """Persist newly-computed vectors to the S3 tier (once per item-type).

        Called at the end of an item-type's indexing rather than per embed batch,
        so the full-object read-merge-write happens a handful of times per run
        instead of dozens.
        """
        if self._s3_embedding_cache is not None:
            self._s3_embedding_cache.flush()

    @staticmethod
    def _prepare_documents(
        items: list[Any], embeddings: list[tuple], prepare_func: Callable
    ) -> tuple[list[dict], list[str]]:
        docs, failed_ids = [], []
        embedding_failed_ids: list[str] = []

        for item, embedding_tuple in zip(items, embeddings, strict=True):
            if any(emb is None for emb in embedding_tuple):
                logger.debug(
                    "Embedding generation failed for item ID: %s. Skipping.", item.id
                )
                embedding_failed_ids.append(item.id)
                failed_ids.append(item.id)
                continue

            try:
                docs.append(prepare_func(item, embedding_tuple))
            except Exception as e:
                logger.error(
                    "Failed to prepare document for item ID: %s. Error: %s", item.id, e
                )
                failed_ids.append(item.id)

        if embedding_failed_ids:
            logger.warning(
                "Skipped %s of %s items with missing embeddings (e.g. %s)",
                len(embedding_failed_ids),
                len(items),
                ", ".join(str(i) for i in embedding_failed_ids[:5]),
            )
        return docs, failed_ids

    def _perform_indexing(
        self, index_name: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> IndexingStats:
        start_time = time.time()
        stats = IndexingStats(total_items=len(documents))

        try:
            response = self.opensearch_client.bulk_index(
                index_name,
                documents,
                refresh=refresh and self.opensearch_config.refresh_after_batch,
            )

            if not response.get("errors"):
                stats.add_success(len(documents))
            else:
                failed = [
                    next(iter(item.values()))
                    for item in response.get("items", [])
                    if "error" in next(iter(item.values()))
                ]
                stats.add_error(
                    f"Bulk API errors: {len(failed)}",
                    len(failed),
                    ids=[str(item["_id"]) for item in failed if "_id" in item],
                )
                stats.add_success(len(documents) - len(failed))
        except Exception as e:
            stats.add_error(
                f"Bulk indexing failed: {e}",
                len(documents),
                ids=[str(doc["id"]) for doc in documents if "id" in doc],
            )

        stats.processing_time = time.time() - start_time
        return stats

    def _get_base_mapping(self, properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "settings": {
                "number_of_shards": self.opensearch_config.index_settings.get(
                    "number_of_shards", 1
                ),
                "number_of_replicas": self.opensearch_config.index_settings.get(
                    "number_of_replicas", 0
                ),
                # refresh_interval trades search-visibility latency for indexing
                # throughput; raising it (e.g. "30s" or "-1") speeds bulk loads.
                "refresh_interval": self.opensearch_config.index_settings.get(
                    "refresh_interval", "1s"
                ),
                "index.knn": True,
                "index.knn.algo_param.ef_search": self.opensearch_config.vector_search.get(
                    "ef_search", 100
                ),
            },
            "mappings": {
                # Read back by check_index_settings.
                "_meta": self._index_meta(),
                "dynamic_templates": [
                    {
                        "strings_as_keywords": {
                            "match_mapping_type": "string",
                            "mapping": {"type": "keyword"},
                        }
                    }
                ],
                "properties": properties,
            },
        }

    def _index_meta(self) -> dict[str, Any]:
        meta = self._embedding_meta()
        if self._doc_status_tracked is not None:
            meta[_META_DOC_STATUS] = self._doc_status_tracked
        return meta

    def _get_knn_vector_mapping(self) -> dict[str, Any]:
        vs_config = self.opensearch_config.vector_search
        return {
            "type": "knn_vector",
            "dimension": self._embedding_dimension,
            "method": {
                "name": vs_config.get("name", "hnsw"),
                "space_type": vs_config.get("space_type", "cosinesimil"),
                # lucene HNSW supports cosinesimil on all versions; faiss HNSW
                # only supports it from OpenSearch 2.19. Default engine kept in
                # sync with OpenSearchIndexingConfig.vector_search.
                "engine": vs_config.get("engine", "lucene"),
                "parameters": {
                    "ef_construction": vs_config.get("ef_construction", 128),
                    "m": vs_config.get("m", 24),
                },
            },
        }

    def _get_text_units_mapping(self) -> dict[str, Any]:
        # ``text`` holds the original chunk, so it is analyzed for the source
        # language. An untranslated corpus (source == target) is searchable
        # only through this field: Korean text under the standard analyzer
        # keeps "홍길동으로부터" as one token, which "홍길동" never matches.
        # Each translated language gets its own field and analyzer.
        translation = self.config.processing.translation
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": self._analyzer_for(translation.source_language.value),
                },
                **{
                    f"translated_text_{language}": {
                        "type": "text",
                        "analyzer": self._analyzer_for(language),
                    }
                    for language in translation.translated_languages
                },
                "text_embedding": self._get_knn_vector_mapping(),
                "community_ids": {"type": "keyword"},
                "n_tokens": {"type": "integer"},
                "attributes": {"type": "object", "dynamic": True},
            }
        )

    def _get_entities_mapping(self) -> dict[str, Any]:
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "name": {
                    "type": "text",
                    "analyzer": self.analyzer,
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "name_embedding": self._get_knn_vector_mapping(),
                "description": {"type": "text", "analyzer": self.analyzer},
                "description_embedding": self._get_knn_vector_mapping(),
                "type": {"type": "keyword"},
                "rank": {"type": "double"},
                "confidence": {"type": "double"},
                "text_unit_ids": {"type": "keyword"},
                "attributes": {"type": "object", "dynamic": True},
            }
        )

    def _get_relationships_mapping(self) -> dict[str, Any]:
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "target_id": {"type": "keyword"},
                "source_name": {"type": "text", "analyzer": self.analyzer},
                "target_name": {"type": "text", "analyzer": self.analyzer},
                "description": {"type": "text", "analyzer": self.analyzer},
                "description_embedding": self._get_knn_vector_mapping(),
                "weight": {"type": "double"},
                "rank": {"type": "double"},
                "text_unit_ids": {"type": "keyword"},
                "attributes": {"type": "object", "dynamic": True},
            }
        )

    def _get_claims_mapping(self) -> dict[str, Any]:
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "subject_id": {"type": "keyword"},
                "object_id": {"type": "keyword"},
                "subject_name": {"type": "text", "analyzer": self.analyzer},
                "object_name": {"type": "text", "analyzer": self.analyzer},
                "type": {"type": "keyword"},
                "status": {"type": "keyword"},
                "description": {"type": "text", "analyzer": self.analyzer},
                "description_embedding": self._get_knn_vector_mapping(),
                "source_text": {"type": "text", "analyzer": self.analyzer},
                "text_unit_ids": {"type": "keyword"},
                "attributes": {"type": "object", "dynamic": True},
            }
        )

    def _get_community_reports_mapping(self) -> dict[str, Any]:
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "community_id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": self.analyzer},
                "name_embedding": self._get_knn_vector_mapping(),
                "summary": {"type": "text", "analyzer": self.analyzer},
                "summary_embedding": self._get_knn_vector_mapping(),
                "full_content": {"type": "text", "analyzer": self.analyzer},
                "full_content_embedding": self._get_knn_vector_mapping(),
                "rank": {"type": "double"},
                "rating": {"type": "double"},
                # Additive: an index created before these fields existed maps
                # them as keyword via the base mapping's strings_as_keywords
                # dynamic template, so an upsert into it stays compatible.
                "text_unit_ids": {"type": "keyword"},
                "document_ids": {"type": "keyword"},
                "level": {"type": "integer"},
                "depth": {"type": "integer"},
                "max_cut_depth": {"type": "integer"},
                "parent_community_id": {"type": "keyword"},
                "child_community_ids": {"type": "keyword"},
                "attributes": {"type": "object", "dynamic": True},
            }
        )
