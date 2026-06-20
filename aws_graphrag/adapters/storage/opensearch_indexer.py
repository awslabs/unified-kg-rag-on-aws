# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from collections.abc import Callable
from typing import Any

import boto3
from opensearchpy.exceptions import NotFoundError

from aws_graphrag.adapters.aws import BedrockEmbeddingModelFactory, OpenSearchClient
from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Claim,
    CommunityReport,
    Config,
    Constants,
    Entity,
    Relationship,
    TextUnit,
)
from aws_graphrag.storage.base import IndexingStats, VectorIndexer
from aws_graphrag.utils.common import compute_hash

logger = get_logger(__name__)


class OpenSearchIndexer(VectorIndexer):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self.opensearch_config = self.config.indexing.opensearch
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.opensearch_client = OpenSearchClient(config, self.boto_session)
        self.embedding_factory = BedrockEmbeddingModelFactory(
            config,
            self.boto_session,
            self.config.aws.bedrock.region_name,
        )
        self._embedding_dimension = self._resolve_embedding_dimension()
        self.embedding_model = self.embedding_factory.get_model(
            model_id=self.opensearch_config.embedding_model_id,
            dimensions=self._embedding_dimension,
        )
        self.target_language = self.config.processing.translation.target_language.value
        self.analyzer = self.opensearch_config.language_analyzers.get(
            self.target_language, self.opensearch_config.default_analyzer
        )
        # Per-process content-hash -> embedding cache (avoids re-embedding
        # duplicate/unchanged text within and across incremental runs).
        self._embedding_cache: dict[str, list[float]] = {}

    def _resolve_embedding_dimension(self) -> int:
        model_info = self.embedding_factory.get_model_info(
            self.opensearch_config.embedding_model_id
        )
        if not model_info:
            raise ValueError(
                f"Unsupported model: '{self.opensearch_config.embedding_model_id.value}'"
            )

        if (dim := self.opensearch_config.embedding_dimension) is not None:
            supported = model_info.dimensions
            is_supported = (isinstance(supported, list) and dim in supported) or (
                isinstance(supported, int) and dim == supported
            )
            if not is_supported:
                raise ValueError(
                    f"Dimension {dim} not supported by "
                    f"'{self.opensearch_config.embedding_model_id.value}'. "
                    f"Supported: '{supported}'"
                )
            return dim

        return (
            model_info.dimensions[-1]
            if isinstance(model_info.dimensions, list)
            else model_info.dimensions or 1024
        )

    def clear(self, suffixes: list[str]) -> bool:
        if not suffixes:
            return True

        try:
            prefixes = [
                self.opensearch_config.text_units_index_prefix,
                self.opensearch_config.entities_index_prefix,
                self.opensearch_config.community_reports_index_prefix,
                self.opensearch_config.relationships_index_prefix,
                self.opensearch_config.claims_index_prefix,
            ]

            aliases_to_delete = [
                self._get_name(prefix, suffix)
                for prefix in prefixes
                for suffix in suffixes
            ]

            index_patterns_to_delete = []
            for prefix in prefixes:
                for suffix in suffixes:
                    if self.config.indexing.additional_suffix:
                        pattern = f"{prefix}-{suffix}-{self.config.indexing.additional_suffix}-*"
                    else:
                        pattern = f"{prefix}-{suffix}-*"
                    index_patterns_to_delete.append(pattern)

            if aliases_to_delete:
                self.opensearch_client.delete_alias(
                    index_names="_all", alias_names=aliases_to_delete
                )

            if index_patterns_to_delete:
                self.opensearch_client.delete_indices(index_patterns_to_delete)

            time.sleep(1)
            logger.info(f"Cleared OpenSearch indices for '{aliases_to_delete}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to clear OpenSearch indices for '{aliases_to_delete}': {e}"
            )
            return False

    def get_entity_count(self, suffixes: list[str]) -> int:
        if not suffixes:
            return 0

        try:
            alias_names = [
                self._get_name(self.opensearch_config.entities_index_prefix, suffix)
                for suffix in suffixes
            ]
            count = self.opensearch_client.count(alias_names)
            return int(count) if count is not None else 0
        except NotFoundError:
            return 0
        except Exception as e:
            logger.error(f"Failed to get entity count for '{alias_names}': {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        try:
            patterns = [
                f"{prefix}-*"
                for prefix in [
                    self.opensearch_config.text_units_index_prefix,
                    self.opensearch_config.entities_index_prefix,
                    self.opensearch_config.community_reports_index_prefix,
                    self.opensearch_config.relationships_index_prefix,
                    self.opensearch_config.claims_index_prefix,
                ]
            ]
            return {
                "last_run": self.stats.to_dict(),
                "cluster_indices": self.opensearch_client.get_index_stats(patterns),
            }
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            return {"error": str(e), "last_run": self.stats.to_dict()}

    def initialize(self) -> bool:
        try:
            pipeline_name = self.opensearch_config.hybrid_search_pipeline_name
            if not self.opensearch_client.check_search_pipeline_exists(pipeline_name):
                self._create_hybrid_search_pipeline(pipeline_name)
                logger.info(f"Created hybrid search pipeline: '{pipeline_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch indexer: {e}")
            return False

    def _create_hybrid_search_pipeline(self, pipeline_name: str) -> None:
        lex_weight, vec_weight = (
            self.config.search.hybrid.lexical_weight,
            self.config.search.hybrid.vector_weight,
        )
        body = {
            "description": "Hybrid search combining keyword and vector search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [lex_weight, vec_weight]},
                        },
                    }
                }
            ],
        }
        self.opensearch_client.create_search_pipeline(pipeline_name, body)

    def index_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        def get_embedding_text(unit: TextUnit) -> str:
            if hasattr(unit, "translated_texts") and unit.translated_texts:
                return unit.translated_texts.get(self.target_language) or unit.text
            return unit.text

        def prepare_doc(
            unit: TextUnit, embeddings: tuple[list[float], ...]
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
                if translated := unit.translated_texts.get(self.target_language):
                    doc[f"translated_text_{self.target_language}"] = translated

            return doc

        return self._index_item_type(
            items=text_units,
            item_type_name="text units",
            alias_prefix=self.opensearch_config.text_units_index_prefix,
            mapping_func=self._get_text_units_mapping,
            embedding_field_extractors=[get_embedding_text],
            prepare_doc_func=prepare_doc,
        )

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        def prepare_doc(
            entity: Entity, embeddings: tuple[list[float], ...]
        ) -> dict[str, Any]:
            return {
                **self._prepare_common_doc_properties(entity),
                "name": entity.name or "",
                "name_embedding": embeddings[0],
                "description": entity.description or "",
                "description_embedding": embeddings[1],
                "type": entity.type or "",
                "rank": entity.rank or 1.0,
                "confidence": (
                    entity.confidence if entity.confidence is not None else 1.0
                ),
            }

        return self._index_item_type(
            items=entities,
            item_type_name="entities",
            alias_prefix=self.opensearch_config.entities_index_prefix,
            mapping_func=self._get_entities_mapping,
            embedding_field_extractors=[lambda e: e.name, lambda e: e.description],
            prepare_doc_func=prepare_doc,
        )

    def index_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        def prepare_doc(report: CommunityReport, embeddings: tuple) -> dict[str, Any]:
            return {
                **self._prepare_common_doc_properties(report),
                "community_id": report.community_id,
                "name": report.name or "",
                "name_embedding": embeddings[0],
                "summary": report.summary or "",
                "summary_embedding": embeddings[1],
                "full_content": report.full_content or "",
                "full_content_embedding": embeddings[2],
                "rank": report.rank or 1.0,
            }

        return self._index_item_type(
            items=reports,
            item_type_name="community reports",
            alias_prefix=self.opensearch_config.community_reports_index_prefix,
            mapping_func=self._get_community_reports_mapping,
            embedding_field_extractors=[
                lambda r: r.name,
                lambda r: r.summary,
                lambda r: r.full_content,
            ],
            prepare_doc_func=prepare_doc,
        )

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
            "rank": rel.rank or 1.0,
        }

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Embed and index relationship descriptions (LightRAG global retrieval).

        Relationships are embedded as first-class vectors so high-level keywords
        can retrieve relations directly, as in LightRAG's ``relationships_vdb``.
        """
        return self._index_item_type(
            items=relationships,
            item_type_name="relationships",
            alias_prefix=self.opensearch_config.relationships_index_prefix,
            mapping_func=self._get_relationships_mapping,
            embedding_field_extractors=[lambda r: r.description or ""],
            prepare_doc_func=self._prepare_relationship_doc,
        )

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Upsert relationship vectors by id into the live index (delta)."""
        return self._upsert_item_type(
            items=relationships,
            item_type_name="relationships",
            alias_prefix=self.opensearch_config.relationships_index_prefix,
            mapping_func=self._get_relationships_mapping,
            embedding_field_extractors=[lambda r: r.description or ""],
            prepare_doc_func=self._prepare_relationship_doc,
        )

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
        }

    def index_claims(self, claims: list[Claim]) -> IndexingStats:
        """Embed and index claim (covariate) descriptions as searchable vectors.

        Connects claim extraction to retrieval: claims become first-class
        searchable artifacts (mirrors the relationship vector index).
        """
        return self._index_item_type(
            items=claims,
            item_type_name="claims",
            alias_prefix=self.opensearch_config.claims_index_prefix,
            mapping_func=self._get_claims_mapping,
            embedding_field_extractors=[lambda c: c.description or ""],
            prepare_doc_func=self._prepare_claim_doc,
        )

    def upsert_claims(self, claims: list[Claim]) -> IndexingStats:
        """Upsert claim vectors by id into the live index (delta)."""
        return self._upsert_item_type(
            items=claims,
            item_type_name="claims",
            alias_prefix=self.opensearch_config.claims_index_prefix,
            mapping_func=self._get_claims_mapping,
            embedding_field_extractors=[lambda c: c.description or ""],
            prepare_doc_func=self._prepare_claim_doc,
        )

    def upsert_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        """Upsert text units by id into the live index (delta semantics)."""

        def get_embedding_text(unit: TextUnit) -> str:
            if hasattr(unit, "translated_texts") and unit.translated_texts:
                return unit.translated_texts.get(self.target_language) or unit.text
            return unit.text

        def prepare_doc(
            unit: TextUnit, embeddings: tuple[list[float], ...]
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
                if translated := unit.translated_texts.get(self.target_language):
                    doc[f"translated_text_{self.target_language}"] = translated
            return doc

        return self._upsert_item_type(
            items=text_units,
            item_type_name="text units",
            alias_prefix=self.opensearch_config.text_units_index_prefix,
            mapping_func=self._get_text_units_mapping,
            embedding_field_extractors=[get_embedding_text],
            prepare_doc_func=prepare_doc,
        )

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Upsert entities by id into the live index (delta semantics)."""

        def prepare_doc(
            entity: Entity, embeddings: tuple[list[float], ...]
        ) -> dict[str, Any]:
            return {
                **self._prepare_common_doc_properties(entity),
                "name": entity.name or "",
                "name_embedding": embeddings[0],
                "description": entity.description or "",
                "description_embedding": embeddings[1],
                "type": entity.type or "",
                "rank": entity.rank or 1.0,
                "confidence": (
                    entity.confidence if entity.confidence is not None else 1.0
                ),
            }

        return self._upsert_item_type(
            items=entities,
            item_type_name="entities",
            alias_prefix=self.opensearch_config.entities_index_prefix,
            mapping_func=self._get_entities_mapping,
            embedding_field_extractors=[lambda e: e.name, lambda e: e.description],
            prepare_doc_func=prepare_doc,
        )

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
            failed = len(response.get("items", []))
            stats.add_success(len(ids) - failed)
            if failed:
                stats.add_error(f"Bulk delete errors: {failed}", failed)
        except Exception as e:
            stats.add_error(f"Delete-by-id failed: {e}", len(ids))
        return stats

    def _upsert_item_type(
        self,
        items: list[Any],
        item_type_name: str,
        alias_prefix: str,
        mapping_func: Callable[[], dict[str, Any]],
        embedding_field_extractors: list[Callable[[Any], str]],
        prepare_doc_func: Callable[[Any, tuple[list[float], ...]], dict[str, Any]],
    ) -> IndexingStats:
        """Index documents by id into the live aliased index without rebuilding it.

        Unlike ``_index_item_type`` (create new index -> alias swap -> delete old),
        this writes/overwrites by document id into the index the alias already
        points at, creating a first index + alias if none exists. This makes
        incremental delta updates additive instead of a full rebuild.
        """
        if not items:
            return IndexingStats()

        logger.info("Upserting %d %s", len(items), item_type_name)
        total_stats = IndexingStats()

        for suffix, chunk_items in self._group_items_by_suffix(items).items():
            alias_name = self._get_name(alias_prefix, suffix)
            index_name = self.opensearch_client.get_index_name_by_alias(alias_name)

            try:
                if not index_name:
                    # First-ever write for this alias: create one index + alias.
                    index_name = self._get_name(
                        alias_prefix, suffix, add_timestamp=True
                    )
                    self.opensearch_client.create_index(index_name, mapping_func())
                    self.opensearch_client.update_alias(alias_name, index_name)

                embeddings = self._generate_embeddings(
                    chunk_items, embedding_field_extractors
                )
                docs, failed_ids = self._prepare_documents(
                    chunk_items, embeddings, prepare_doc_func
                )
                if failed_ids:
                    total_stats.add_error(
                        f"Failed to prepare {len(failed_ids)} documents",
                        len(failed_ids),
                    )
                if docs:
                    total_stats.merge(self._perform_indexing(index_name, docs))
            except Exception as e:
                logger.error(
                    "Failed to upsert %s (suffix=%s): %s", item_type_name, suffix, e
                )
                total_stats.add_error(str(e), len(chunk_items))

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
        self,
        items: list[Any],
        item_type_name: str,
        alias_prefix: str,
        mapping_func: Callable[[], dict[str, Any]],
        embedding_field_extractors: list[Callable[[Any], str]],
        prepare_doc_func: Callable[[Any, tuple[list[float], ...]], dict[str, Any]],
    ) -> IndexingStats:
        if not items:
            return IndexingStats()

        logger.info(f"Indexing {len(items)} {item_type_name}")
        total_stats = IndexingStats()

        for suffix, chunk_items in self._group_items_by_suffix(items).items():
            alias_name = self._get_name(alias_prefix, suffix)
            index_name = self._get_name(alias_prefix, suffix, add_timestamp=True)

            try:
                self.opensearch_client.create_index(index_name, mapping_func())

                embeddings = self._generate_embeddings(
                    chunk_items, embedding_field_extractors
                )
                docs, failed_ids = self._prepare_documents(
                    chunk_items, embeddings, prepare_doc_func
                )

                if failed_ids:
                    total_stats.add_error(
                        f"Failed to prepare {len(failed_ids)} documents",
                        len(failed_ids),
                    )

                if docs:
                    indexing_stats = self._perform_indexing(index_name, docs)
                    total_stats.merge(indexing_stats)

                    if indexing_stats.successful_items > 0:
                        remove_pattern = (
                            f"{alias_prefix}-{suffix}-{self.config.indexing.additional_suffix}-*"
                            if self.config.indexing.additional_suffix
                            else f"{alias_prefix}-{suffix}-*"
                        )
                        self.opensearch_client.update_alias(
                            alias_name, index_name, remove_pattern=remove_pattern
                        )

                old_indices_pattern = (
                    f"{alias_prefix}-{suffix}-{self.config.indexing.additional_suffix}-*"
                    if self.config.indexing.additional_suffix
                    else f"{alias_prefix}-{suffix}-*"
                )
                all_indices_for_alias = self.opensearch_client.get_indices_by_alias(
                    old_indices_pattern
                )
                indices_to_clean = [
                    idx for idx in all_indices_for_alias if idx != index_name
                ]

                if indices_to_clean:
                    self.opensearch_client.delete_indices(indices_to_clean)

            except Exception as e:
                logger.error(f"Failed to index {item_type_name} (suffix={suffix}): {e}")
                total_stats.add_error(str(e), len(chunk_items))
                try:
                    self.opensearch_client.delete_indices([index_name])
                except Exception:
                    pass

        if total_stats.failed_items > 0:
            logger.warning(
                f"Indexing {item_type_name} completed: {total_stats.successful_items} "
                f"succeeded, {total_stats.failed_items} failed"
            )
        else:
            logger.info(
                f"Successfully indexed {total_stats.successful_items} {item_type_name}"
            )

        return total_stats

    def _generate_embeddings(
        self, items: list[Any], extractors: list[Callable]
    ) -> list[tuple]:
        all_embeddings = [
            self._batch_embed([extractor(item) for item in items])
            for extractor in extractors
        ]
        return list(zip(*all_embeddings, strict=True))

    def _batch_embed(
        self, texts: list[str], batch_size: int = 50
    ) -> list[list[float] | None]:
        result: list[list[float] | None] = [None] * len(texts)

        # Content-hash embedding cache (in-process only): identical text is
        # embedded once per process, avoiding re-embedding duplicate chunks
        # within a run and across indices that share the same text in the same
        # run. It does NOT persist across separate CLI invocations. Only
        # cache-miss, de-duplicated texts hit Bedrock; the result is fanned back
        # out to every index sharing that text.
        key_to_indices: dict[str, list[int]] = {}
        unique: list[tuple[str, str]] = []  # (content_hash, text) to embed
        seen_keys: set[str] = set()
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = compute_hash(text, length=32)
            key_to_indices.setdefault(key, []).append(i)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                result[i] = cached
            elif key not in seen_keys:
                seen_keys.add(key)
                unique.append((key, text))

        def _store(key: str, emb: list[float] | None) -> None:
            if emb is not None:
                self._embedding_cache[key] = emb
            for idx in key_to_indices[key]:
                result[idx] = emb

        for batch_start in range(0, len(unique), batch_size):
            batch = unique[batch_start : batch_start + batch_size]
            keys = [k for k, _ in batch]
            batch_texts = [t for _, t in batch]
            try:
                embeddings = self.embedding_model.embed_documents(batch_texts)
                for key, emb in zip(keys, embeddings, strict=True):
                    _store(key, emb)
            except Exception as e:
                logger.warning(
                    f"Batch embedding failed ({len(batch)} items), "
                    f"retrying individually: {e}"
                )
                for key, text in batch:
                    try:
                        single_embs = self.embedding_model.embed_documents([text])
                        _store(key, single_embs[0] if single_embs else None)
                    except Exception as item_error:
                        logger.error(f"Failed to embed text '{key}': {item_error}")
                        _store(key, None)

        return result

    @staticmethod
    def _prepare_documents(
        items: list[Any], embeddings: list[tuple], prepare_func: Callable
    ) -> tuple[list[dict], list[str]]:
        docs, failed_ids = [], []

        for item, embedding_tuple in zip(items, embeddings, strict=True):
            if any(emb is None for emb in embedding_tuple):
                logger.warning(
                    f"Embedding generation failed for item ID: {item.id}. Skipping."
                )
                failed_ids.append(item.id)
                continue

            try:
                docs.append(prepare_func(item, embedding_tuple))
            except Exception as e:
                logger.error(
                    f"Failed to prepare document for item ID: {item.id}. Error: {e}"
                )
                failed_ids.append(item.id)

        return docs, failed_ids

    def _perform_indexing(
        self, index_name: str, documents: list[dict[str, Any]]
    ) -> IndexingStats:
        start_time = time.time()
        stats = IndexingStats(total_items=len(documents))

        try:
            response = self.opensearch_client.bulk_index(
                index_name,
                documents,
                refresh=self.opensearch_config.refresh_after_batch,
            )

            if not response.get("errors"):
                stats.add_success(len(documents))
            else:
                failed_count = sum(
                    1
                    for item in response.get("items", [])
                    if "error" in next(iter(item.values()))
                )
                stats.add_error(f"Bulk API errors: {failed_count}", failed_count)
                stats.add_success(len(documents) - failed_count)
        except Exception as e:
            stats.add_error(f"Bulk indexing failed: {e}", len(documents))

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
                "index.knn": True,
                "index.knn.algo_param.ef_search": self.opensearch_config.vector_search.get(
                    "ef_search", 100
                ),
            },
            "mappings": {
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

    def _get_knn_vector_mapping(self) -> dict[str, Any]:
        vs_config = self.opensearch_config.vector_search
        return {
            "type": "knn_vector",
            "dimension": self._embedding_dimension,
            "method": {
                "name": vs_config.get("name", "hnsw"),
                "space_type": vs_config.get("space_type", "cosinesimil"),
                "engine": vs_config.get("engine", "faiss"),
                "parameters": {
                    "ef_construction": vs_config.get("ef_construction", 128),
                    "m": vs_config.get("m", 24),
                },
            },
        }

    def _get_text_units_mapping(self) -> dict[str, Any]:
        return self._get_base_mapping(
            {
                "id": {"type": "keyword"},
                "text": {"type": "text"},
                f"translated_text_{self.target_language}": {
                    "type": "text",
                    "analyzer": self.analyzer,
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
                "attributes": {"type": "object", "dynamic": True},
            }
        )
