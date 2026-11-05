# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import threading
import time
import uuid
from array import array
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Collection, Hashable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.adapters.storage.embedding_settings import (
    META_BUILD,
    META_DIMENSION,
    META_MODEL,
    resolve_embedding_dimension,
    stored_embedding_settings,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    ATTRIBUTE_KEY_PREFIX,
    IndexLimits,
)
from unified_kg_rag.adapters.storage.search_engine import SearchEngineClient
from unified_kg_rag.domain.ingestion.base_processor import SEED_PROPERTIES_ATTRIBUTE
from unified_kg_rag.domain.ingestion.claim_resolver import claim_date_value
from unified_kg_rag.domain.ingestion.community_hierarchy import max_cut_depth
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    RECORD_KINDS,
)
from unified_kg_rag.domain.ingestion.seed_graph import SEED_SOURCE_KIND, is_seed_unit
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Constants,
    Entity,
    IndexBuildRecord,
    ModelPurpose,
    Relationship,
    TextUnit,
    model_id_value,
)
from unified_kg_rag.domain.models.config import DEFAULT_INDEX_SETTINGS
from unified_kg_rag.domain.models.provider_config import BEDROCK
from unified_kg_rag.ports.indexer import (
    AnnotatedKind,
    IndexingStats,
    SchemaAnnotation,
    StoredAnnotation,
    VectorIndexer,
)
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import (
    ConfigurationError,
    DataProcessingError,
    NamespaceLeaseError,
    get_logger,
)
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError
from unified_kg_rag.shared.store_lookups import StoreLookupCache, lookup_cache
from unified_kg_rag.shared.utils.common import compute_hash
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.write_fence import check_write_fence

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)

# Texts per embedding request, and requests in flight at once (Bedrock
# embedding calls are IO-bound). In the real-AWS E2E 16 requests in flight
# embedded 1,600 texts in half the time 8 took, with no throttling.
_EMBED_BATCH_SIZE = 50
_EMBED_WORKERS = 16
# Ids per read-back search at most (fewer when the index's limits are lower:
# _limits).
_READ_BATCH_SIZE = 1000
# Items embedded and bulk-written at a time. Holding a whole item type's
# vectors as float lists before one bulk call took gigabytes on a large
# corpus. Its texts to embed take about one round of the concurrent
# embedding requests (_EMBED_BATCH_SIZE * _EMBED_WORKERS = 800), and one
# search reads the stored digests of a slice (_reusable_vectors).
_WRITE_SLICE_SIZE = _EMBED_BATCH_SIZE * _EMBED_WORKERS
# Ids per read of the stored vectors a write takes (_reusable_vectors): at
# 1,024 dimensions three vectors are about 64 KB of JSON per document, and
# reading a whole slice's at once held over 60 MB of response and its
# parsed floats.
_VECTOR_READ_BATCH_SIZE = 100
# Reads of stored vectors in flight at once: the store serves one while the
# client parses another.
_VECTOR_READ_WORKERS = 4
# Values per terms search of find_documents at most.
_FIND_BATCH_SIZE = 100
# The settings _get_base_mapping adds for the vector indices (the config
# refuses them in index_settings); the extraction-records index has no vectors.
_KNN_SETTINGS = ("knn", "knn.algo_param.ef_search")
# The index prefix (config field) and stored fields of read_vector_copies.
_VECTOR_COPY_FIELDS = {
    "entities": (
        "entities_index_prefix",
        [
            "name",
            "type",
            "confidence",
            "description",
            "in_schema",
            "type_iri",
            "iri",
            "source_kinds",
            # Its seed_properties (read_vector_copies): the embedded engine
            # filters _source by top-level field only.
            "attributes",
        ],
    ),
    "relationships": (
        "relationships_index_prefix",
        [
            "source_id",
            "target_id",
            "source_name",
            "target_name",
            "description",
            "in_schema",
            "predicate_iri",
            "source_kinds",
        ],
    ),
}

# The IRI field of each annotated kind's documents.
_IRI_FIELD: dict[AnnotatedKind, str] = {
    "entities": "type_iri",
    "relationships": "predicate_iri",
}

# Field types a dynamic mapping can give a field of another type, which then
# refuses this release's values (see _add_missing_fields).
_EXACT_TYPES = frozenset({"keyword", "boolean"})

# A seed graph's properties: kept in _source, never mapped. Mapped
# dynamically, a key's first value would type it (a later value of another
# type, a string over 32 KB, an integer over 2^63 or a key such as "" or "."
# fails the whole entity's write), and each key would count against the
# index's field limit.
_SEED_PROPERTIES_MAPPING: dict[str, Any] = {"type": "object", "enabled": False}

# Each vector field's digest of the text it embeds, under the field's name
# (_embedding_digest: the embedding settings are hashed in). A write reuses
# a stored vector whose digest matches instead of embedding its text again.
# Kept in _source only (keywords neither indexed nor in doc values: a
# dynamic mapping, before an older index is given this one, would make the
# same type).
_DIGESTS_FIELD = "embedding_digests"
_DIGEST_LENGTH = 16
# The stored vectors upserts replaced, kept (by digest, as float64 arrays) for
# a later write of their text: a removal rewrites an item the commit after it
# rewrites again. At 1,024 dimensions, 16 MB at most.
_REPLACED_VECTORS = 2048


def _present(**fields: Any) -> dict[str, Any]:
    """``fields`` but the ``None`` ones: a schema annotation (or a seed-graph
    field) is written only when set, so a document without one is what it
    was before the field."""
    return {name: value for name, value in fields.items() if value is not None}


def _missing_properties(
    expected: dict[str, Any], live: dict[str, Any]
) -> dict[str, Any]:
    """The fields of ``expected`` that ``live`` (mapping ``properties``)
    lacks, as a mapping update: a field missing whole, and an object's
    missing sub-fields under its name (the object's own parameters left out,
    as a live object's cannot change)."""
    missing: dict[str, Any] = {}
    for name, field in expected.items():
        stored = live.get(name)
        if stored is None:
            missing[name] = field
        elif isinstance(stored, dict) and isinstance(field.get("properties"), dict):
            nested = _missing_properties(
                field["properties"], stored.get("properties") or {}
            )
            if nested:
                missing[name] = {"properties": nested}
    return missing


def _mapped_disabled_objects(
    expected: dict[str, Any], live: dict[str, Any], prefix: str = ""
) -> list[str]:
    """The paths of the ``enabled: false`` objects of ``expected`` that
    ``live`` maps as enabled (a dynamic mapping made them)."""
    paths: list[str] = []
    for name, field in expected.items():
        stored = live.get(name)
        if not isinstance(stored, dict):
            continue
        if field.get("enabled") is False and stored.get("enabled") is not False:
            paths.append(f"{prefix}{name}")
        elif isinstance(field.get("properties"), dict):
            paths += _mapped_disabled_objects(
                field["properties"], stored.get("properties") or {}, f"{prefix}{name}."
            )
    return paths


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
    # The knn_vector field each extractor's text is embedded into.
    vector_fields: tuple[str, ...]


# How OpenSearch before 2.17 refuses the knn_vector storage keys.
_UNKNOWN_STORAGE_KEY = re.compile(r"unknown parameter \[(mode|compression_level)\]")
# The embedded engine was warned of ignoring them (once per process).
_storage_ignored_warned = False


def _vector_storage_of(field: dict[str, Any]) -> tuple[str, str]:
    """The workload mode and compression level a knn_vector mapping
    ``field`` stores its vectors with, OpenSearch's defaults filled in."""
    mode = field.get("mode") or "in_memory"
    level = field.get("compression_level") or ("32x" if mode == "on_disk" else "1x")
    return str(mode), str(level)


def _creation_refused(
    e: StoreRequestRejectedError, index: str | None = None
) -> StoreRequestRejectedError:
    """The error for an index creation the engine refused (``e``); ``index``
    names the index a probe stood in for."""
    if "maximum shards open" in str(e):
        hint = (
            "The cluster has no room for another index's shards: delete "
            "indices it no longer needs or raise cluster.max_shards_per_node."
        )
    elif _UNKNOWN_STORAGE_KEY.search(str(e)):
        hint = (
            "indexing.opensearch.vector_mode and compression_level need "
            "OpenSearch 2.17 or later (the CDK domain runs 2.13): upgrade the "
            "domain (docs/operations.md, Upgrading OpenSearch) or unset them."
        )
    else:
        hint = (
            "Every indexing.opensearch.index_settings key is passed to index "
            "creation, and the mapping follows vector_search, vector_mode, "
            "compression_level and the language analyzers: fix or remove the "
            "setting it names."
        )
    target = f" (a probe of '{index}')" if index else ""
    return StoreRequestRejectedError(
        f"The search engine refused to create an index{target}: {e}. {hint}",
        reason=e.reason,
    )


# The reason of a rejected bulk item and of each of its causes, each cut
# short (cut as a whole, a long reason left no room for the cause), and
# without the corpus text: a mapping error's reason quotes the rejected
# field's value ("Preview of field's value: ..."), an immense-term rejection
# the term's bytes ("The prefix of the first immense term is: ..."), and a
# cause may quote the value it could not read ('For input string: "..."'),
# none of which may reach the logs.
_BULK_REASON_LIMIT = 300
_BULK_REASON_QUOTES = re.compile(
    r"\s*(Preview of field's value|The prefix of the first immense term)\b.*",
    re.DOTALL,
)
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def _bulk_item_error(error: Any) -> str:
    """``type: reason`` of a bulk item's ``error``, then of the errors it was
    caused by (``caused_by``, which names the cause: "Vector dimension
    mismatch"), without the corpus text a reason quotes."""
    if not isinstance(error, dict):
        return str(error)[:_BULK_REASON_LIMIT]
    parts: list[str] = []
    while isinstance(error, dict) and len(parts) < 5:
        reason = _BULK_REASON_QUOTES.sub("", str(error.get("reason", "")))
        if parts:
            reason = _QUOTED.sub('"..."', reason)
        parts.append(f"{error.get('type', 'error')}: {reason}"[:_BULK_REASON_LIMIT])
        error = error.get("caused_by")
    return " (caused by ".join(parts) + ")" * (len(parts) - 1)


def _with_digests(
    prepare_doc: Callable[[Any, tuple[list[float], ...]], dict[str, Any]],
    digests: dict[int, dict[str, str]],
) -> Callable[[Any, tuple[list[float], ...]], dict[str, Any]]:
    """``prepare_doc`` adding each item's digests (by ``id(item)``)."""

    def prepare(item: Any, vectors: tuple[list[float], ...]) -> dict[str, Any]:
        return {**prepare_doc(item, vectors), _DIGESTS_FIELD: digests[id(item)]}

    return prepare


class OpenSearchIndexer(VectorIndexer):
    # Recorded in the _meta of the indices a full build creates (None records
    # nothing); see set_doc_status_tracking.
    _doc_status_tracked: bool | None = None
    # Whether writes reuse stored vectors (set_embedding_reuse).
    _reuse_embeddings: bool = True
    # Whether upserts keep the vectors they replace (set_replaced_vector_keeping).
    _keep_replaced_vectors: bool = True
    # Guards _replaced: a build writes its item types concurrently.
    _replaced_lock = threading.Lock()

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
        self.opensearch_client: SearchEngineClient = (
            opensearch_client
            if opensearch_client is not None
            else OpenSearchClient(config, boto_session)
        )
        # The limits in force on each live index read (see _limits).
        self._index_limits: dict[str, IndexLimits] = {}
        self.embedding_factory = embedding_factory
        self._embedding_dimension = self._resolve_embedding_dimension()
        self.embedding_model = self.embedding_factory.get_model(
            model_id=self.config.embedding_model_id,
            dimensions=self._embedding_dimension,
            model_purpose=ModelPurpose.INGESTION,
        )
        self.target_language = self.config.processing.translation.target_language.value
        self.analyzer = self._analyzer_for(self.target_language)

    def _resolve_vector_storage(self) -> dict[str, str]:
        """The ``mode`` / ``compression_level`` keys of the knn_vector
        mappings: none on the embedded engine, which searches exactly."""
        global _storage_ignored_warned
        storage = self.opensearch_config.vector_storage()
        if storage and self.in_process:
            if not _storage_ignored_warned:
                _storage_ignored_warned = True
                logger.warning(
                    "indexing.opensearch.vector_mode / compression_level (%s) "
                    "are ignored by the embedded search engine, which keeps "
                    "full-precision vectors and searches them exactly",
                    ", ".join(storage.values()),
                )
            return {}
        return storage

    def _analyzer_for(self, language: str) -> str:
        return self.opensearch_config.language_analyzers.get(
            language, self.opensearch_config.default_analyzer
        )

    def _resolve_embedding_dimension(self) -> int:
        return resolve_embedding_dimension(self.embedding_factory, self.config)

    def close(self) -> None:
        """Release the search-engine client's connections (best-effort)."""
        self.opensearch_client.close()

    @property
    def in_process(self) -> bool:
        """Whether its search engine is embedded in this process (SQLite)."""
        return getattr(self.opensearch_client, "in_process", False) is True

    # --- alias and mapping lookups (cached for a run: shared.store_lookups) --

    def _lookup_cache(self) -> tuple[StoreLookupCache, Hashable] | None:
        """The run's lookup cache and what names this indexer's store in it
        (``store_identity``: two clients of one store share their lookups);
        ``None`` outside a run, or for a client that names no store."""
        cache = lookup_cache()
        store: Hashable | None = getattr(self.opensearch_client, "store_identity", None)
        return None if cache is None or store is None else (cache, store)

    def _live_index(self, alias_name: str) -> str | None:
        """The index ``alias_name`` points at (``None``: none), read once per
        run."""
        if (cached := self._lookup_cache()) is None:
            return self.opensearch_client.get_index_name_by_alias(alias_name)
        cache, store = cached
        return cache.alias(
            store,
            alias_name,
            lambda: self.opensearch_client.get_index_name_by_alias(alias_name),
        )

    def _index_mappings(self, index_name: str) -> dict[str, Any]:
        """``index_name``'s mappings, read once per run."""
        if (cached := self._lookup_cache()) is None:
            return self.opensearch_client.get_index_mappings(index_name)
        cache, store = cached
        return cache.mappings(
            store,
            index_name,
            lambda: self.opensearch_client.get_index_mappings(index_name),
        )

    def _forget_aliases(self) -> None:
        if (cached := self._lookup_cache()) is not None:
            cached[0].forget_aliases(cached[1])

    def _forget_mappings(self, index_names: Collection[str]) -> None:
        if (cached := self._lookup_cache()) is not None:
            cached[0].forget_mappings(cached[1], index_names)

    def _update_alias(
        self, alias_name: str, index_name: str, remove_pattern: str | None = None
    ) -> None:
        self._forget_aliases()
        if remove_pattern is None:
            self.opensearch_client.update_alias(alias_name, index_name)
        else:
            self.opensearch_client.update_alias(
                alias_name, index_name, remove_pattern=remove_pattern
            )

    def _delete_alias(self, index_names: list[str], alias_names: list[str]) -> None:
        self._forget_aliases()
        self.opensearch_client.delete_alias(
            index_names=index_names, alias_names=alias_names
        )

    def _delete_indices(self, index_names: list[str]) -> None:
        """Delete aliased (or once aliased) indices: an index's aliases go
        with it."""
        self._forget_aliases()
        self._forget_mappings(index_names)
        self.opensearch_client.delete_indices(index_names)

    def _put_index_mapping(self, index_name: str, body: dict[str, Any]) -> None:
        self._forget_mappings([index_name])
        self.opensearch_client.put_index_mapping(index_name, body)

    def _put_index_meta(self, index_name: str, meta: dict[str, Any]) -> None:
        self._forget_mappings([index_name])
        self.opensearch_client.put_index_meta(index_name, meta)

    # --- embedding reuse ------------------------------------------------------

    def set_embedding_reuse(self, reuse: bool) -> None:
        """Whether writes reuse the stored vectors of unchanged texts (see
        ``_embed_and_write``); off, every text is embedded again."""
        self._reuse_embeddings = reuse

    def set_replaced_vector_keeping(self, keep: bool) -> None:
        """Whether upserts read and keep the stored vectors they replace (in
        ``_replaced``, see ``_reusable_vectors``); off, they read only the
        vectors they take. The vectors kept so far stay for later writes."""
        self._keep_replaced_vectors = keep

    @cached_property
    def _digest_seed(self) -> str:
        """The embedding settings a vector's digest hashes with its text: a
        vector of another provider, model, dimension or input cap is never
        taken for this run's."""
        provider = self.config.providers.embedding
        settings: list[Any] = [
            provider.kind,
            model_id_value(self.config.embedding_model_id),
            self._embedding_dimension,
        ]
        if provider.kind != BEDROCK:
            settings.append(provider.max_input_tokens)
        return "\x1f".join(str(value) for value in settings)

    @cached_property
    def _replaced(self) -> "OrderedDict[str, array[float]]":  # array[...]: 3.12+
        """The stored vectors writes replaced, by digest, oldest first (see
        ``_generate_embeddings``)."""
        return OrderedDict()

    def _embedding_digest(self, text: str) -> str:
        """The digest stored with the vector of ``text``."""
        return compute_hash(f"{self._digest_seed}\x1e{text}", length=_DIGEST_LENGTH)

    def _reusable_vectors(
        self,
        alias_prefix: str,
        suffix: str,
        items: list[Any],
        spec: _ItemSpec,
        keep_replaced: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """The stored vectors (and digests) a write of ``items`` takes from
        ``suffix``'s live index of ``alias_prefix``, by id: those whose
        digest is the digest of one of the items' texts now.

        Two steps, so a write never holds a vector it does not take: one read
        of the items' stored digests (at most ``_READ_BATCH_SIZE`` ids), then
        reads of only the matching fields' vectors,
        ``_VECTOR_READ_BATCH_SIZE`` ids at a time. With ``keep_replaced`` (an
        upsert) the vectors the write replaces are read as well, up to
        ``_REPLACED_VECTORS``, into ``_replaced``, for a later write of their
        text (a removal rewrites an item the commit after it writes again).
        None to read when reuse is off, or the live index holds no digests
        (an earlier release built it) or another model's vectors.
        Best-effort: a failed read reads none (every text is embedded)."""
        if not self._reuse_embeddings or not items:
            return {}
        try:
            index_name = self._live_index(self._get_name(alias_prefix, suffix))
            if not index_name:
                return {}
            mappings = self._index_mappings(index_name)
            model, dimensions = stored_embedding_settings(mappings)
            if (
                _DIGESTS_FIELD not in (mappings.get("properties") or {})
                or model not in (None, model_id_value(self.config.embedding_model_id))
                or not dimensions <= {self._embedding_dimension}
            ):
                return {}
            current = {
                self._embedding_digest(text)
                for item in items
                for extractor in spec.extractors
                if (text := extractor(item)) and text.strip()
            }
            # Ids by the fields whose vectors to read.
            wanted: dict[tuple[str, ...], list[str]] = defaultdict(list)
            replaced = 0
            for source in self._read_sources(
                alias_prefix,
                sorted({item.id for item in items}),
                suffix,
                fields=[_DIGESTS_FIELD],
            ):
                kept = source.get(_DIGESTS_FIELD)
                if not isinstance(kept, dict):
                    continue
                fields = []
                for field in spec.vector_fields:
                    digest = kept.get(field)
                    if not isinstance(digest, str):
                        continue
                    if digest in current:
                        fields.append(field)
                    elif keep_replaced and replaced < _REPLACED_VECTORS:
                        replaced += 1
                        fields.append(field)
                if fields:
                    wanted[tuple(fields)].append(source["id"])
            reads = [
                (
                    ids[start : start + _VECTOR_READ_BATCH_SIZE],
                    [*fields, _DIGESTS_FIELD],
                )
                for fields, ids in wanted.items()
                for start in range(0, len(ids), _VECTOR_READ_BATCH_SIZE)
            ]
            reusable: dict[str, dict[str, Any]] = {}
            if not reads:
                return reusable
            with ContextThreadPoolExecutor(
                max_workers=min(_VECTOR_READ_WORKERS, len(reads))
            ) as executor:
                for sources in executor.map(
                    lambda read: self._read_sources(
                        alias_prefix, read[0], suffix, fields=read[1]
                    ),
                    reads,
                ):
                    for source in sources:
                        reusable[source["id"]] = self._keep_replaced(source, current)
            return reusable
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - every text is embedded instead
            logger.warning(
                "Reading the stored vectors of %s failed; embedding every text: %s",
                spec.name,
                e,
            )
            return {}

    def _keep_replaced(
        self, source: dict[str, Any], current: Collection[str]
    ) -> dict[str, Any]:
        """``source`` without the vectors whose digest is none of ``current``
        (the texts written now): those are kept in ``_replaced`` instead, as
        float64 arrays."""
        kept = source.get(_DIGESTS_FIELD)
        for field, digest in (kept if isinstance(kept, dict) else {}).items():
            vector = source.get(field)
            if digest in current or not isinstance(vector, list):
                continue
            del source[field]
            if isinstance(digest, str) and len(vector) == self._embedding_dimension:
                with self._replaced_lock:
                    self._replaced[digest] = array("d", vector)
                    self._replaced.move_to_end(digest)
                    while len(self._replaced) > _REPLACED_VECTORS:
                        self._replaced.popitem(last=False)
        return source

    def _index_prefixes(self) -> list[str]:
        return [spec.alias_prefix for spec in self._specs()]

    def _embedding_meta(self) -> dict[str, Any]:
        """What every index records of the embeddings it holds (``_meta``)."""
        return {
            META_MODEL: model_id_value(self.config.embedding_model_id),
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
        by the ``strings_as_keywords`` dynamic template on the first write (or
        as a date, on an index built before date detection was turned off).
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
          mapping's ``_meta``: another model's vectors would be mixed in,
          another dimension fails every write;
        - the analyzers of its text fields, which follow the translation
          languages and ``language_analyzers`` and cannot change on a live
          index (a newly translated language would be mapped as a keyword);
        - the workload mode and compression level of its vector fields
          (``vector_mode`` / ``compression_level``), fixed when the index was
          created;
        - the doc-status registry: an index built by a run with the registry
          off holds documents the registry does not describe;
        - its keyword and boolean fields: one a dynamic mapping made a date
          (date detection, on an index built before it was turned off) or a
          keyword refuses every value of the mapped form, such as an unknown
          claim date;
        - its entities' ``attributes.seed_properties``, which this release
          leaves unmapped (``enabled: false``): an index whose earlier build
          mapped them key by key would refuse values of another type.

        An index that records no embedding settings was built by an earlier
        release, whose indices this one does not read: it is rejected too.
        An index that passes gets, in one additive mapping update, the fields
        this release maps that it lacks (the claim dates of an index built
        before #444, the schema annotations of one built before
        ``in_schema``, the seed-graph fields ``iri``, ``source_kinds`` and
        ``source_kind`` of one built before them, the unmapped
        ``attributes.seed_properties``) and date detection turned off, before
        any write.

        Raises:
            ConfigurationError: An index of ``suffix`` does not match.
        """
        expected = self._embedding_meta()
        mapping_funcs = self._mapping_funcs()
        for prefix in self._index_prefixes():
            index_name = self._live_index(self._get_name(prefix, suffix))
            if not index_name:
                continue
            mappings = self._index_mappings(index_name)
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
            expected_properties = mapping_funcs[prefix]()["mappings"]["properties"]
            self._check_vector_storage(index_name, expected_properties, properties)
            mismatches = self._text_field_mismatches(expected_properties, properties)
            if META_MODEL not in meta and META_DIMENSION not in meta:
                raise ConfigurationError(
                    f"The OpenSearch index '{index_name}' records no embedding "
                    "settings: an earlier release built it, and this release "
                    "does not read its indices. Re-run once with "
                    "indexing.reset: true to rebuild the stores."
                )
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
            self._add_missing_fields(index_name, mappings, expected_properties)

    @staticmethod
    def _check_vector_storage(
        index_name: str, expected: dict[str, Any], live: dict[str, Any]
    ) -> None:
        """Raise when a knn_vector field of ``live`` stores its vectors with
        another workload mode or compression level than ``expected``'s.

        Raises:
            ConfigurationError: A field's mode or compression level differs.
        """
        for name, field in expected.items():
            stored = live.get(name)
            if field.get("type") != "knn_vector" or not isinstance(stored, dict):
                continue
            wanted, built = _vector_storage_of(field), _vector_storage_of(stored)
            if wanted != built:
                raise ConfigurationError(
                    f"The OpenSearch index '{index_name}' stores the vectors of "
                    f"'{name}' as {built[0]} with {built[1]} compression, but "
                    "indexing.opensearch.vector_mode / compression_level ask for "
                    f"{wanted[0]} with {wanted[1]}. An incremental run writes "
                    "only new and changed documents into the live index, whose "
                    "vector mapping cannot change. Re-run once with "
                    "indexing.reset: true to rebuild the indices, or restore the "
                    "previous vector_mode and compression_level."
                )

    def _add_missing_fields(
        self,
        index_name: str,
        mappings: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        """Map on ``index_name`` the ``expected`` fields it lacks (an object's
        missing sub-fields too) and turn its date detection off, or raise
        when a keyword or boolean field is mapped otherwise or an object this
        release does not map (``enabled: false``) is mapped.

        Raises:
            ConfigurationError: A keyword or boolean field of ``expected`` is
                mapped as another type (a date, by date detection; a keyword,
                by the dynamic template), or an unmapped object was mapped
                dynamically (the seed properties of an index written before
                they were left unmapped).
        """
        live = mappings.get("properties") or {}
        mapped = _mapped_disabled_objects(expected, live)
        if mapped:
            raise ConfigurationError(
                f"The OpenSearch index '{index_name}' maps "
                f"{', '.join(repr(name) for name in mapped)} field by field: an "
                "earlier build mapped each seed property key on its first "
                "write, so a later value of another type, a string over 32 KB "
                "or a key such as '.' fails the entity's write. This release "
                "keeps seed properties unmapped, and a live index cannot "
                "unmap them. Re-run once with indexing.reset: true to rebuild "
                "the indices."
            )
        retyped = [
            f"'{name}' is mapped as {live[name].get('type')} (needs "
            f"{field['type']})"
            for name, field in expected.items()
            if field.get("type") in _EXACT_TYPES
            and isinstance(live.get(name), dict)
            and live[name].get("type") != field["type"]
        ]
        if retyped:
            raise ConfigurationError(
                f"The OpenSearch index '{index_name}' maps fields otherwise: "
                f"{'; '.join(retyped)}. A dynamic mapping typed them on a "
                "first write, and every later value of another form fails to "
                "write; a live index's field types cannot change. Re-run once "
                "with indexing.reset: true to rebuild the indices."
            )
        missing = _missing_properties(expected, live)
        if missing or mappings.get("date_detection") is not False:
            self._put_index_mapping(
                index_name, {"date_detection": False, "properties": missing}
            )

    def record_build(
        self, record: IndexBuildRecord, suffix: str | None
    ) -> IndexBuildRecord | None:
        """Write ``record`` into the ``_meta`` of each live index of ``suffix``.

        Merged into the existing ``_meta`` (the embedding settings stay), so
        it needs ``indices:admin/mapping/put``; a failure is logged and the run
        goes on. An incremental record keeps the full build the index holds
        and its community counters (``IndexBuildRecord.over``). Returns the
        first record written that carries counters (else the first written).
        """
        written: IndexBuildRecord | None = None
        for prefix in self._index_prefixes():
            alias = self._get_name(prefix, suffix)
            try:
                index_name = self._live_index(alias)
                if not index_name:
                    continue
                meta = self._index_mappings(index_name).get("_meta") or {}
                built = record.over(meta.get(META_BUILD))
                payload = built.model_dump(mode="json")
                self._put_index_meta(index_name, {**meta, META_BUILD: payload})
                if written is None or (
                    written.communities is None and built.communities is not None
                ):
                    written = built
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - the record is optional
                logger.warning(
                    "Could not record the build of pipeline '%s' in the _meta of "
                    "%s (%s); run-eval manifests will not identify this index.",
                    record.pipeline_id,
                    alias,
                    e,
                )
        return written

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
            self._delete_indices(indices_to_delete)

        # Deleting an index drops its aliases. Only an alias still on an
        # index is deleted: deleting aliases none of which exists (a
        # namespace never indexed, or a store it never wrote) is a 404.
        for alias in aliases_to_delete:
            remaining = self.opensearch_client.get_indices_by_alias(alias)
            if remaining:
                self._delete_alias(index_names=remaining, alias_names=[alias])

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
                ("text_embedding",),
            ),
            _ItemSpec(
                "entities",
                config.entities_index_prefix,
                self._get_entities_mapping,
                [lambda e: e.name, self._entity_description_text],
                self._prepare_entity_doc,
                ("name_embedding", "description_embedding"),
            ),
            _ItemSpec(
                "community reports",
                config.community_reports_index_prefix,
                self._get_community_reports_mapping,
                [lambda r: r.name, lambda r: r.summary, lambda r: r.full_content],
                self._prepare_community_report_doc,
                ("name_embedding", "summary_embedding", "full_content_embedding"),
            ),
            _ItemSpec(
                "relationships",
                config.relationships_index_prefix,
                self._get_relationships_mapping,
                [self._relationship_description_text],
                self._prepare_relationship_doc,
                ("description_embedding",),
            ),
            _ItemSpec(
                "claims",
                config.claims_index_prefix,
                self._get_claims_mapping,
                [self._claim_description_text],
                self._prepare_claim_doc,
                ("description_embedding",),
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
            # include material the prompt's entity cap left out.
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
            # Only on a seed graph's unit: a document's is stored as before.
            **_present(source_kind=SEED_SOURCE_KIND if is_seed_unit(unit.id) else None),
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
            **_present(
                in_schema=entity.in_schema,
                type_iri=entity.type_iri,
                iri=entity.iri,
                source_kinds=entity.source_kinds or None,
            ),
        }

    @classmethod
    def _prepare_relationship_doc(
        cls, rel: Relationship, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        return {
            **cls._document_filter_properties(rel),
            "id": rel.id,
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "source_name": rel.source_name or "",
            "target_name": rel.target_name or "",
            # The edge's label (neptune_codec.relationship_label): a
            # relationship without a description (every seed edge an RDF
            # seed states) reads as its endpoints and type alone.
            "type": rel.type or Constants.DEFAULT_RELATIONSHIP_TYPE.value,
            "description": rel.description or "",
            "description_embedding": embeddings[0],
            "weight": rel.weight if rel.weight is not None else 1.0,
            "rank": rel.rank if rel.rank is not None else 1.0,
            # Chunk lineage: lets LightRAG mix follow relationship -> source
            # chunks (LightRAG's _find_related_text_unit_from_relationships).
            "text_unit_ids": list(rel.text_unit_ids or []),
            **_present(
                in_schema=rel.in_schema,
                predicate_iri=rel.predicate_iri,
                source_kinds=rel.source_kinds or None,
            ),
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

    @classmethod
    def _prepare_claim_doc(
        cls, claim: Claim, embeddings: tuple[list[float], ...]
    ) -> dict[str, Any]:
        return {
            **cls._document_filter_properties(claim),
            "id": claim.id,
            "subject_id": claim.subject_id,
            "subject_name": claim.subject_name or "",
            "object_id": claim.object_id,
            "object_name": claim.object_name or "",
            # Lowercase, as entity types: a filter's term matches exactly
            # (``type:affiliation``, ``status:true``).
            "type": (claim.type or "").lower(),
            "status": (claim.status or "").lower(),
            # Stored as claim_date_value reads them, which the claim's
            # identity keys on: YYYY-MM-DD for a calendar date, so a range
            # filter orders them (a coarse end date as its period's last
            # day); another value as written, matched exactly. An unknown
            # date is left out, so it matches no filter. Read back by
            # read_claims, shown with the claim (``Claim: ... (status,
            # start–end)``).
            **{
                name: date
                for name, end in (("start_date", False), ("end_date", True))
                if (date := claim_date_value(getattr(claim, name), end=end)) is not None
            },
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

    def _limits(self, index_name: str) -> IndexLimits:
        """The limits in force on ``index_name``: the smaller of the
        configured ``index_settings`` and the settings the index was created
        with, read once per index. An index keeps its settings when the
        config changes (until a reindex), so a window configured below or
        above the live one would otherwise ask for more hits than it returns
        (a 400). Raises when the read fails."""
        limits = self._index_limits.get(index_name)
        if limits is None:
            live = self.opensearch_client.get_index_settings(index_name)
            limits = IndexLimits.of(
                self.opensearch_config.index_settings, *live.values()
            )
            self._index_limits[index_name] = limits
        return limits

    def _read_sources(
        self,
        alias_prefix: str,
        ids: list[str],
        suffix: str | None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The stored documents (without embeddings; only ``fields`` when
        given) of ``ids`` in ``suffix``'s live index of ``alias_prefix``; none
        without a live index. Raises when the read fails."""
        alias_name = self._get_name(alias_prefix, suffix)
        index_name = self._live_index(alias_name) if ids else None
        if not index_name:
            return []
        step = min(_READ_BATCH_SIZE, self._limits(index_name).id_batch_size)
        sources: list[dict[str, Any]] = []
        for start in range(0, len(ids), step):
            batch = ids[start : start + step]
            response = self.opensearch_client.search(
                index=alias_name,
                body={
                    "size": len(batch),
                    "query": {"ids": {"values": batch}},
                    # No fields: ids only (an empty ``includes`` returns all).
                    "_source": (
                        {"excludes": ["*_embedding"]}
                        if fields is None
                        else {"includes": fields} if fields else False
                    ),
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
                self.opensearch_config.text_units_index_prefix, ids, suffix, fields=[]
            )
        ]

    def read_claims(self, ids: list[str], suffix: str | None = None) -> list[Claim]:
        """Read stored claims by id from ``suffix``'s live claims index.

        A claim indexed before claims stored their ``text_unit_ids`` (or
        their dates) reads back without them. The stored document holds only the document
        filters of the claim's ``attributes`` (``_document_filter_properties``):
        they are read back, and each claim is tagged with ``suffix`` (the
        index it was read from), so a write of the claim, e.g. the
        ``update_claim_lineage`` of a document deletion, keeps them and goes
        back to that index rather than the default one. Raises when the read
        fails.
        """
        placement = {Constants.INDEX.value: suffix} if suffix else {}

        def attributes(source: dict[str, Any]) -> dict[str, Any]:
            filters = (source.get("attributes") or {}).get(Constants.FILTERS.value)
            kept = {k: v for k, v in (filters or {}).items() if v is not None}
            return {**placement, **({Constants.FILTERS.value: kept} if kept else {})}

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
                    "start_date": source.get("start_date") or None,
                    "end_date": source.get("end_date") or None,
                    "description": source.get("description") or None,
                    "source_text": source.get("source_text") or None,
                    "text_unit_ids": list(source.get("text_unit_ids") or []),
                    "attributes": attributes(source),
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

    def read_vector_copies(
        self,
        kind: Literal["entities", "relationships"],
        ids: list[str],
        suffix: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """The stored fields of ``ids`` incremental writes compare (see the
        port)."""
        prefix, fields = _VECTOR_COPY_FIELDS[kind]
        copies: dict[str, dict[str, Any]] = {}
        for source in self._read_sources(
            getattr(self.opensearch_config, prefix), ids, suffix, fields=fields
        ):
            attributes = source.pop("attributes", None) or {}
            if properties := attributes.get(SEED_PROPERTIES_ATTRIBUTE):
                source[SEED_PROPERTIES_ATTRIBUTE] = properties
            copies[source["id"]] = source
        return copies

    def read_schema_annotations(
        self, kind: AnnotatedKind, suffix: str | None = None
    ) -> dict[str, StoredAnnotation]:
        """Every stored copy's annotations by id (see the port), read in id
        order a window at a time, as :meth:`find_documents` reads."""
        prefix, _ = _VECTOR_COPY_FIELDS[kind]
        alias_name = self._get_name(getattr(self.opensearch_config, prefix), suffix)
        index_name = self._live_index(alias_name)
        if not index_name:
            return {}
        iri = _IRI_FIELD[kind]
        fields = ["id", "in_schema", iri, *(["type"] if kind == "entities" else [])]
        window = self._limits(index_name).max_result_window
        annotations: dict[str, StoredAnnotation] = {}
        after: str | None = None
        while True:
            response = self.opensearch_client.search(
                index=alias_name,
                body={
                    "size": window,
                    "query": (
                        {"match_all": {}}
                        if after is None
                        else {"bool": {"filter": [{"range": {"id": {"gt": after}}}]}}
                    ),
                    "sort": [{"id": {"order": "asc"}}],
                    "_source": {"includes": fields},
                    "track_total_hits": False,
                },
            )
            page = list(response.get("hits", {}).get("hits", []))
            for hit in page:
                source = hit.get("_source") or {}
                in_schema, iri_value = source.get("in_schema"), source.get(iri)
                annotations[str(hit["_id"])] = StoredAnnotation(
                    type=source.get("type") if kind == "entities" else None,
                    in_schema=in_schema if isinstance(in_schema, bool) else None,
                    iri=iri_value if isinstance(iri_value, str) else None,
                )
            if len(page) < window:
                return annotations
            after = str((page[-1].get("_source") or {}).get("id") or page[-1]["_id"])

    def holds_any_item(self, suffix: str | None = None) -> bool:
        """Whether the entities or relationships index holds a document
        (see the port): one size-1 search of each live index."""
        kinds: tuple[AnnotatedKind, ...] = ("entities", "relationships")
        return any(
            self._holds_a_document(_VECTOR_COPY_FIELDS[kind][0], suffix)
            for kind in kinds
        )

    def holds_any_text_unit(self, suffix: str | None = None) -> bool:
        """Whether the text-unit index holds a document (see the port): one
        size-1 search of the live index."""
        return self._holds_a_document("text_units_index_prefix", suffix)

    def _holds_a_document(self, prefix: str, suffix: str | None) -> bool:
        """Whether the live index of the ``prefix`` config field holds a
        document (``False`` without one): one size-1 search."""
        alias_name = self._get_name(getattr(self.opensearch_config, prefix), suffix)
        if not self._live_index(alias_name):
            return False
        response = self.opensearch_client.search(
            index=alias_name,
            body={
                "size": 1,
                "query": {"match_all": {}},
                "_source": False,
                "track_total_hits": False,
            },
        )
        return bool(response.get("hits", {}).get("hits"))

    def write_schema_annotations(
        self,
        kind: AnnotatedKind,
        annotations: Mapping[str, SchemaAnnotation],
        suffix: str | None = None,
    ) -> IndexingStats:
        """Set the annotations by id in place (see the port)."""
        stats = IndexingStats(total_items=len(annotations))
        if not annotations:
            return stats
        prefix, _ = _VECTOR_COPY_FIELDS[kind]
        alias_name = self._get_name(getattr(self.opensearch_config, prefix), suffix)
        iri = _IRI_FIELD[kind]
        ids = sorted(annotations)
        try:
            index_name = self._live_index(alias_name)
            if not index_name:
                stats.add_success(len(ids))
                return stats
            for start in range(0, len(ids), _READ_BATCH_SIZE):
                batch = ids[start : start + _READ_BATCH_SIZE]
                response = self.opensearch_client.bulk_update(
                    index_name,
                    [
                        {
                            "id": item_id,
                            "in_schema": annotations[item_id][0],
                            iri: annotations[item_id][1],
                        }
                        for item_id in batch
                    ],
                    refresh=self.opensearch_config.refresh_after_batch,
                )
                failed = [next(iter(r.values())) for r in response.get("items", [])]
                stats.add_success(len(batch) - len(failed))
                if failed:
                    stats.add_error(
                        f"Bulk update errors: {len(failed)}",
                        len(failed),
                        ids=[str(f["_id"]) for f in failed if "_id" in f],
                    )
        except NamespaceLeaseError:
            raise
        except Exception as e:
            stats.add_error(
                f"Update of '{alias_name}' failed: {e}",
                len(ids) - stats.successful_items - stats.failed_items,
            )
        return stats

    def prune_text_unit_communities(
        self, community_ids: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Remove deleted communities' ids from the stored text units naming
        them (see the port), in place."""
        prefix = self.opensearch_config.text_units_index_prefix
        found = self.find_documents(prefix, "community_ids", community_ids, suffix)
        if not found:
            return IndexingStats()
        gone = set(community_ids)
        units = [
            TextUnit.model_validate(
                {
                    "id": source["id"],
                    "text": "",
                    "community_ids": [
                        c for c in source.get("community_ids") or [] if c not in gone
                    ],
                    "attributes": {Constants.INDEX.value: suffix} if suffix else None,
                }
            )
            for source in self._read_sources(
                prefix, sorted(found), suffix, fields=["id", "community_ids"]
            )
        ]
        # ``community_ids`` only: these units are built without their
        # attributes, and writing their (absent) filters would null every
        # stored filter key of an unchanged document's units.
        return self._update_fields(
            units,
            prefix,
            lambda u: {"community_ids": list(u.community_ids or [])},
            with_filters=False,
        )

    def read_text_unit_attributes(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Stored text units' attributes by id (see the port)."""
        return {
            source["id"]: dict(source.get("attributes") or {})
            for source in self._read_sources(
                self.opensearch_config.text_units_index_prefix,
                ids,
                suffix,
                fields=["attributes"],
            )
        }

    def find_documents(
        self,
        alias_prefix: str,
        field: str,
        values: list[str],
        suffix: str | None = None,
    ) -> dict[str, list[str]] | None:
        """Documents whose keyword ``field`` holds any of ``values``, with
        their ``text_unit_ids`` (see the port). Read in id order, a window
        at a time, until a window comes back short: the total a search
        reports stops at 10,000 by default (``relation: gte``), so it cannot
        tell a full window from the last one. The window and the values per
        search follow the index's limits (``_limits``)."""
        alias_name = self._get_name(alias_prefix, suffix)
        index_name = self._live_index(alias_name) if values else None
        if not index_name:
            return {}
        limits = self._limits(index_name)
        window = limits.max_result_window
        step = min(_FIND_BATCH_SIZE, limits.max_terms_count)
        found: dict[str, list[str]] = {}
        for start in range(0, len(values), step):
            terms = {"terms": {field: values[start : start + step]}}
            after: str | None = None
            while True:
                filters: list[dict[str, Any]] = [terms]
                if after is not None:
                    filters.append({"range": {"id": {"gt": after}}})
                response = self.opensearch_client.search(
                    index=alias_name,
                    body={
                        "size": window,
                        "query": {"bool": {"filter": filters}},
                        "sort": [{"id": {"order": "asc"}}],
                        "_source": {"includes": ["id", "text_unit_ids"]},
                        "track_total_hits": False,
                    },
                )
                page = list(response.get("hits", {}).get("hits", []))
                for hit in page:
                    source = hit.get("_source") or {}
                    found[hit["_id"]] = list(source.get("text_unit_ids") or [])
                if len(page) < window:
                    break
                after = str(
                    (page[-1].get("_source") or {}).get("id") or page[-1]["_id"]
                )
        return found

    def delete_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Delete extraction records by key (see the port)."""
        alias_name = self._get_name(
            self.opensearch_config.extraction_records_index_prefix, suffix
        )
        if not keys or not self._live_index(alias_name):
            return IndexingStats()
        return self.delete_by_id(
            keys,
            self.opensearch_config.extraction_records_index_prefix,
            suffix or Constants.DEFAULT_SUFFIX.value,
        )

    def retain_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Keep only the records ``keys`` name (see the port).

        The kept records are copied, a batch at a time, into a new index and
        the alias is swapped onto it; the old index, with every record no key
        names, is deleted. A failure before the swap drops the new index and
        leaves the live one as it was.
        """
        prefix = self.opensearch_config.extraction_records_index_prefix
        alias_name = self._get_name(prefix, suffix)
        stats = IndexingStats()
        if not self._live_index(alias_name):
            return stats
        index_name = self._new_index_name(
            alias_name, prefix, suffix or Constants.DEFAULT_SUFFIX.value
        )
        wanted = sorted(set(keys))
        try:
            self.opensearch_client.create_index(
                index_name, self._get_extraction_records_mapping(), exist_ok=False
            )
            for start in range(0, len(wanted), _READ_BATCH_SIZE):
                documents = [
                    {"id": source["id"], "record": source["record"]}
                    for source in self._read_sources(
                        prefix, wanted[start : start + _READ_BATCH_SIZE], suffix
                    )
                    if isinstance(source.get("record"), dict)
                ]
                response = self.opensearch_client.bulk_index(index_name, documents)
                if response.get("items"):
                    raise DataProcessingError(
                        f"Copying {len(response['items'])} extraction records "
                        f"to '{index_name}' failed"
                    )
                stats.add_success(len(documents))
            check_write_fence()
            self._update_alias(alias_name, index_name, remove_pattern=f"{alias_name}-*")
        except NamespaceLeaseError:
            raise
        except Exception:
            self._drop_unswapped_index(alias_name, index_name)
            raise
        check_write_fence()
        self._reap_stale_indices(alias_name, index_name)
        stats.total_items = stats.successful_items
        return stats

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

    def stored_extraction_record_keys(
        self, keys: list[str], suffix: str | None = None
    ) -> set[str]:
        """The keys among ``keys`` stored in ``suffix``'s record index, read
        as ids only (see the port)."""
        return {
            source["id"]
            for source in self._read_sources(
                self.opensearch_config.extraction_records_index_prefix,
                keys,
                suffix,
                fields=[],
            )
        }

    def write_extraction_records(
        self, records: dict[str, dict[str, Any]], suffix: str | None = None
    ) -> IndexingStats:
        """Upsert extraction records by key into ``suffix``'s store index.

        The index has no embeddings and indexes no field but the id: the
        record is kept in ``_source`` only. It is not one of the artifact
        indices (``_index_prefixes``), so ``clear`` and the index-settings
        check leave it alone, and a full build keeps the records its text
        units name (:meth:`retain_extraction_records`).
        """
        stats = IndexingStats(total_items=len(records))
        if not records:
            return stats
        alias_name = self._get_name(
            self.opensearch_config.extraction_records_index_prefix, suffix
        )
        try:
            index_name = self._extraction_records_index(suffix)
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
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Writing extraction records to '%s' failed: %s", alias_name, e)
            stats.add_error(str(e), len(records), ids=list(records))
            return stats
        if failed:
            stats.add_error("bulk write rejected", len(failed), ids=failed)
        stats.successful_items = len(records) - len(failed)
        return stats

    def _extraction_records_index(self, suffix: str | None) -> str:
        """The live extraction-records index of ``suffix``, created (and
        aliased) when there is none."""
        prefix = self.opensearch_config.extraction_records_index_prefix
        alias_name = self._get_name(prefix, suffix)
        index_name = self._live_index(alias_name)
        if not index_name:
            index_name = self._new_index_name(
                alias_name, prefix, suffix or Constants.DEFAULT_SUFFIX.value
            )
            self.opensearch_client.create_index(
                index_name, self._get_extraction_records_mapping()
            )
            self._update_alias(alias_name, index_name)
        return index_name

    def prepare_extraction_records(self, suffix: str | None = None) -> None:
        """Create ``suffix``'s extraction-records index unless it exists."""
        try:
            self._extraction_records_index(suffix)
        except StoreRequestRejectedError as e:
            raise _creation_refused(e) from e

    def check_index_creation(
        self, suffix: str | None = None, *, rebuild: bool = True
    ) -> None:
        """Create and delete a probe index per index the run will create, with
        its settings and field definitions (see the port).

        The probes exist together, as the run's new indices do (a full build
        writes its index types concurrently), so a cluster with room for
        fewer of them refuses one. Probes a stopped run left of ``suffix``
        (it was between the two requests) are deleted first; the run holds
        the namespace's write lease. A probe that cannot be deleted is left
        to the next run.
        """
        alias_name = self._get_name(
            self.opensearch_config.extraction_records_index_prefix, suffix
        )
        # Only this run's own naming, on an unaliased index: a sibling
        # namespace's suffix may extend '<suffix>-settings-probe-'.
        probe_name = re.compile(re.escape(alias_name) + r"-settings-probe-[0-9a-f]{12}")
        try:
            if stale := sorted(
                name
                for name, aliases in self.opensearch_client.get_aliases_by_index(
                    f"{alias_name}-settings-probe-*"
                ).items()
                if probe_name.fullmatch(name) and not aliases
            ):
                self.opensearch_client.delete_indices(stale)
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - the next run deletes them
            logger.warning(
                "Deleting the probe indices of '%s' failed: %s", alias_name, e
            )
        probes: list[str] = []
        targets = self._mappings_to_create(suffix, rebuild)
        try:
            for _, mapping in targets:
                probes.append(f"{alias_name}-settings-probe-{uuid.uuid4().hex[:12]}")
                self.opensearch_client.create_index(probes[-1], mapping, exist_ok=False)
        except StoreRequestRejectedError as e:
            raise _creation_refused(e, targets[len(probes) - 1][0]) from e
        finally:
            # Also when a create failed: its response may have been lost
            # after the index was made (a missing one is no error).
            try:
                if probes:
                    self.opensearch_client.delete_indices(probes)
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - the next run deletes it
                logger.warning("Deleting the probe indices %s failed: %s", probes, e)

    def _mappings_to_create(
        self, suffix: str | None, rebuild: bool
    ) -> list[tuple[str, dict[str, Any]]]:
        """The aliases and mappings of the indices of ``suffix`` a run creates: every
        vector index it builds on a ``rebuild`` (a full build or reset, empty
        ones included), otherwise those of the types it may write that have
        no live index; and the extraction records' when it has none."""
        config = self.opensearch_config
        builds = {
            config.relationships_index_prefix: config.build_relationship_vector_index,
        }
        if not rebuild:
            # An incremental run upserts only the types it extracted.
            builds[config.claims_index_prefix] = (
                self.config.processing.claim_extraction.enabled
            )
            builds[config.community_reports_index_prefix] = (
                self.config.graph.community_detection.enabled
            )
        mappings = [
            (alias, spec.mapping())
            for spec in self._specs()
            if builds.get(spec.alias_prefix, True)
            and (alias := self._get_name(spec.alias_prefix, suffix))
            and (rebuild or not self._live_index(alias))
        ]
        records_alias = self._get_name(config.extraction_records_index_prefix, suffix)
        if not self._live_index(records_alias):
            mappings.append((records_alias, self._get_extraction_records_mapping()))
        return mappings

    def _get_extraction_records_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                key: value
                for key, value in self._get_base_mapping({})["settings"].items()
                if key not in _KNN_SETTINGS
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
            lambda c: {
                "text_unit_ids": list(c.text_unit_ids or []),
            },
            with_filters=True,
        )

    def upsert_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        """Upsert text units by id into the live index (delta semantics)."""
        return self._upsert_item_type(text_units, self._spec("text units"))

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Upsert entities by id into the live index (delta semantics)."""
        return self._upsert_item_type(entities, self._spec("entities"))

    def update_entity_lineage(self, entities: list[Entity]) -> IndexingStats:
        """Set stored entities' ``text_unit_ids`` and ``rank`` in place (no
        re-embedding)."""
        return self._update_fields(
            entities,
            self.opensearch_config.entities_index_prefix,
            lambda e: {
                "text_unit_ids": list(e.text_unit_ids or []),
                "rank": e.rank if e.rank is not None else 1.0,
            },
            with_filters=True,
        )

    def update_relationship_lineage(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Set stored relationships' ``text_unit_ids``, ``weight`` and ``rank``
        in place."""
        return self._update_fields(
            relationships,
            self.opensearch_config.relationships_index_prefix,
            lambda r: {
                "text_unit_ids": list(r.text_unit_ids or []),
                "weight": r.weight if r.weight is not None else 1.0,
                "rank": r.rank if r.rank is not None else 1.0,
            },
            with_filters=True,
        )

    def _update_fields(
        self,
        items: list[Any],
        alias_prefix: str,
        fields: Callable[[Any], dict[str, Any]],
        *,
        with_filters: bool,
    ) -> IndexingStats:
        """Partially update ``items`` by id in their suffix's live index:
        ``fields``, and with ``with_filters`` the items' document filters
        (``_filter_update``), which clears the stored keys an item lacks.

        ``with_filters`` only for items that carry their current filters (a
        lineage update's derived entities, relationships and claims); an
        item built for a field-only update has no filters to write, and
        writing them would null the stored ones. A suffix without a live
        index, or an item without a document, has nothing to update.
        """
        stats = IndexingStats(total_items=len(items))
        for suffix, chunk in self._group_items_by_suffix(items).items():
            alias_name = self._get_name(alias_prefix, suffix)
            try:
                index_name = self._live_index(alias_name)
                if not index_name:
                    stats.add_success(len(chunk))
                    continue
                stored = (
                    {
                        source["id"]: source
                        for source in self._read_sources(
                            alias_prefix,
                            [item.id for item in chunk],
                            suffix,
                            fields=[f"{ATTRIBUTE_KEY_PREFIX}*", "attributes"],
                        )
                    }
                    if with_filters
                    else {}
                )
                response = self.opensearch_client.bulk_update(
                    index_name,
                    [
                        {
                            "id": item.id,
                            **fields(item),
                            **(
                                self._filter_update(item, stored.get(item.id) or {})
                                if with_filters
                                else {}
                            ),
                        }
                        for item in chunk
                    ],
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
            except NamespaceLeaseError:
                raise
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
        index_name = self._live_index(alias_name)
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
        except NamespaceLeaseError:
            raise
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
            index_name = self._live_index(alias_name)

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
                    self._update_alias(alias_name, index_name)

                total_stats.merge(
                    self._embed_and_write(
                        index_name,
                        chunk_items,
                        spec,
                        suffix,
                        keep_replaced=self._keep_replaced_vectors,
                    )
                )
            except NamespaceLeaseError:
                raise
            except Exception as e:
                logger.error(
                    "Failed to upsert %s (suffix=%s): %s", spec.name, suffix, e
                )
                total_stats.add_error(
                    str(e), len(chunk_items), ids=[item.id for item in chunk_items]
                )

        return total_stats

    @staticmethod
    def _document_filter_properties(item: Any) -> dict[str, Any]:
        """The item's document ``filters`` as the fields an attribute filter
        matches: ``attr_<key>`` and ``attributes.filters``.

        Relationships and claims carry only these, not their whole
        ``attributes``: those hold per-text-unit maps (edge weights) a
        dynamic mapping would turn into one field per text unit. A lineage
        update writes them too, so a document's removal narrows them.
        """
        attributes = getattr(item, "attributes", None)
        filters = (
            attributes.get(Constants.FILTERS.value)
            if isinstance(attributes, dict)
            else None
        )
        if not isinstance(filters, dict) or not filters:
            return {}
        doc: dict[str, Any] = {
            f"{ATTRIBUTE_KEY_PREFIX}{key}": value
            for key, value in filters.items()
            if value is not None
        }
        doc["attributes"] = {Constants.FILTERS.value: filters}
        return doc

    @classmethod
    def _filter_update(cls, item: Any, stored: dict[str, Any]) -> dict[str, Any]:
        """The partial update that sets ``item``'s document filters on its
        ``stored`` document.

        A partial update merges objects recursively and keeps every field it
        does not name, so a filter key the item lost (its last document with
        that key was removed) would keep matching: such keys are set to
        null, which no ``term`` filter matches.
        """
        update = cls._document_filter_properties(item)
        for key in stored:
            if key.startswith(ATTRIBUTE_KEY_PREFIX) and key not in update:
                update[key] = None
        key = Constants.FILTERS.value
        old = (stored.get("attributes") or {}).get(key)
        new = (update.get("attributes") or {}).get(key) or {}
        dropped = {k: None for k in old or {} if k not in new}
        if dropped:
            update["attributes"] = {key: {**new, **dropped}}
        return update

    @classmethod
    def _prepare_common_doc_properties(cls, item: Any) -> dict[str, Any]:
        """``id``, the whole ``attributes`` and the document filter fields."""
        doc: dict[str, Any] = {"id": item.id, **cls._document_filter_properties(item)}
        if getattr(item, "attributes", None):
            doc["attributes"] = item.attributes
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

                written = self._embed_and_write(index_name, chunk_items, spec, suffix)
                total_stats.merge(written)

                # Swap onto the new index when it holds something, or when the
                # rebuild has nothing of this type for the suffix.
                if not chunk_items or written.successful_items > 0:
                    # A run that lost its lease must not swap a partial index
                    # in, nor delete the complete one. OpenSearch's requests
                    # check the fence themselves (_FencedConnection); the
                    # embedded engine's do not.
                    check_write_fence()
                    self._update_alias(
                        alias_name, index_name, remove_pattern=f"{alias_name}-*"
                    )
                    check_write_fence()
                    self._reap_stale_indices(alias_name, index_name)
                else:
                    # Nothing was written, so the alias stays on the previous
                    # index; drop the unused new one instead of leaking it.
                    self._drop_unswapped_index(alias_name, index_name)

            except NamespaceLeaseError:
                raise
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

        return total_stats

    def _embed_and_write(
        self,
        index_name: str,
        items: list[Any],
        spec: _ItemSpec,
        suffix: str,
        keep_replaced: bool = False,
    ) -> IndexingStats:
        """Embed ``items`` and bulk-write them into ``index_name``, a slice at a
        time, so only one slice's vectors are held at once. The index is
        refreshed once, after the last slice.

        A text whose digest (``_embedding_digest``) is the one stored with
        the vector of its field in ``suffix``'s live index (the index an
        upsert writes, or the one a rebuild replaces) takes that vector
        instead of being embedded again (``_reusable_vectors``; an upsert
        ``keep_replaced``)."""
        stats = IndexingStats()
        for start in range(0, len(items), _WRITE_SLICE_SIZE):
            chunk = items[start : start + _WRITE_SLICE_SIZE]
            stored = self._reusable_vectors(
                spec.alias_prefix, suffix, chunk, spec, keep_replaced
            )
            embeddings, digests = self._generate_embeddings(chunk, spec, stored)
            docs, failed_ids = self._prepare_documents(
                chunk,
                embeddings,
                _with_digests(
                    spec.prepare_doc,
                    {id(item): row for item, row in zip(chunk, digests, strict=True)},
                ),
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
        raised: the swap already succeeded and the live index must survive (a
        lost lease is raised).
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
                self._delete_indices(stale)
                logger.info(
                    "Deleted %s stale index(es) of '%s'", len(stale), alias_name
                )
        except NamespaceLeaseError:
            raise
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
        failed run. Best-effort: a failure is logged, not raised (a lost lease
        is).
        """
        try:
            if index_name in self.opensearch_client.get_indices_by_alias(alias_name):
                return
            self._delete_indices([index_name])
        except NamespaceLeaseError:
            raise
        except Exception as e:
            logger.warning("Failed to delete unused index '%s': %s", index_name, e)

    def _generate_embeddings(
        self,
        items: list[Any],
        spec: _ItemSpec,
        stored: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[list[tuple], list[dict[str, str]]]:
        """Each item's vectors (in ``spec.vector_fields`` order; ``None``
        where embedding failed or the text is empty) and digests by field.

        A text whose digest is that of a ``stored`` vector of this run's
        dimension (``_reusable_vectors``, read for these items) or of one an
        earlier upsert of this indexer replaced (``_replaced``: a run that
        rewrites an item twice, as a removal and the commit after it do)
        takes that vector; the other texts are embedded, all in one
        ``_batch_embed`` call."""
        texts = [[extractor(item) for extractor in spec.extractors] for item in items]
        digests = [
            [self._embedding_digest(t) if t and t.strip() else None for t in row]
            for row in texts
        ]
        # Digest -> stored vector, of the copies read.
        available: dict[str, list[float]] = {}
        for copy in (stored or {}).values():
            kept = copy.get(_DIGESTS_FIELD)
            for field, digest in (kept if isinstance(kept, dict) else {}).items():
                vector = copy.get(field)
                if (
                    isinstance(digest, str)
                    and isinstance(vector, list)
                    and len(vector) == self._embedding_dimension
                ):
                    available[digest] = vector
        vectors: list[list[list[float] | None]] = []
        pending: list[tuple[int, int]] = []
        with self._replaced_lock:
            for i, row in enumerate(digests):
                found: list[list[float] | None] = []
                for j, digest in enumerate(row):
                    vector = None
                    if digest is not None:
                        vector = available.get(digest)
                        if vector is None and (memo := self._replaced.get(digest)):
                            vector = memo.tolist()
                    if vector is None:
                        pending.append((i, j))
                    found.append(vector)
                vectors.append(found)
        embedded = self._batch_embed([texts[i][j] for i, j in pending])
        for (i, j), vector in zip(pending, embedded, strict=True):
            vectors[i][j] = vector
        total = len(items) * len(spec.vector_fields)
        if total > len(pending) and len(pending) < sum(
            1 for row in digests for digest in row if digest
        ):
            logger.info(
                "Reused %d of %d stored vectors of %s (text unchanged)",
                total - len(pending),
                total,
                spec.name,
            )
        return (
            [tuple(row) for row in vectors],
            [
                {f: d for f, d in zip(spec.vector_fields, row, strict=True) if d}
                for row in digests
            ],
        )

    def _batch_embed(
        self, texts: list[str], batch_size: int = _EMBED_BATCH_SIZE
    ) -> list[list[float] | None]:
        result: list[list[float] | None] = [None] * len(texts)

        # Identical texts of one call are embedded once and fanned back out.
        # No vector is kept across calls: a cache held every vector of the run
        # and texts rarely repeat across calls (an incremental run embeds only
        # what its delta changed).
        key_to_indices: dict[str, list[int]] = {}
        unique: list[tuple[str, str]] = []  # (content_hash, text) to embed
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = compute_hash(text, length=32)
            if key not in key_to_indices:
                unique.append((key, text))
            key_to_indices.setdefault(key, []).append(i)

        def _store(key: str, emb: list[float] | None) -> None:
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
        # back on the calling thread to avoid races on result.
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

        return result

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
                    f"Bulk API errors: {len(failed)} "
                    f"(first: {_bulk_item_error(failed[0]['error']) if failed else '?'})",
                    len(failed),
                    ids=[str(item["_id"]) for item in failed if "_id" in item],
                )
                stats.add_success(len(documents) - len(failed))
        except NamespaceLeaseError:
            raise
        except Exception as e:
            stats.add_error(
                f"Bulk indexing failed: {e}",
                len(documents),
                ids=[str(doc["id"]) for doc in documents if "id" in doc],
            )

        stats.processing_time = time.time() - start_time
        return stats

    def _get_base_mapping(self, properties: dict[str, Any]) -> dict[str, Any]:
        # Every index_settings key reaches the index (written without the
        # optional ``index.`` prefix, so each setting has one spelling):
        # retrieval sizes its fetches by max_result_window / max_terms_count
        # from the same map. OpenSearch rejects a setting it does not know
        # when the index is created. The knn keys follow vector_search (the
        # config refuses them in index_settings).
        settings = {
            key.removeprefix("index."): value
            for key, value in self.opensearch_config.index_settings.items()
        }
        for name, default in DEFAULT_INDEX_SETTINGS.items():
            settings.setdefault(name, default)
        vector_fields = [
            name
            for name, field in properties.items()
            if field.get("type") == "knn_vector"
        ]
        if vector_fields:
            properties = {
                **properties,
                _DIGESTS_FIELD: {
                    "type": "object",
                    "properties": {
                        name: {"type": "keyword", "index": False, "doc_values": False}
                        for name in vector_fields
                    },
                },
            }
        return {
            "settings": {
                **settings,
                "knn": True,
                "knn.algo_param.ef_search": self.opensearch_config.vector_search.get(
                    "ef_search", 100
                ),
            },
            "mappings": {
                # Read back by check_index_settings.
                "_meta": self._index_meta(),
                # A date-like string (a claim date, an attr_* value) would
                # otherwise map its field as a date on the first write, and
                # every later value of another form would fail to write.
                "date_detection": False,
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
            # Absent by default (in_memory, unquantized): OpenSearch before
            # 2.17 refuses the keys.
            **self._resolve_vector_storage(),
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
                # "seed_graph" on a seed graph's units, absent on a
                # document's: text searches leave the seed units out.
                "source_kind": {"type": "keyword"},
                "attributes": {
                    "type": "object",
                    "dynamic": True,
                    "properties": {
                        # The unit's extraction-store keys: a removal finds
                        # the records no text unit names any more by them.
                        RECORD_KEYS_ATTRIBUTE: {
                            "properties": {
                                kind: {"type": "keyword"} for kind in RECORD_KINDS
                            }
                        }
                    },
                },
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
                "in_schema": {"type": "boolean"},
                "type_iri": {"type": "keyword"},
                "iri": {"type": "keyword"},
                "source_kinds": {"type": "keyword"},
                "attributes": {
                    "type": "object",
                    "dynamic": True,
                    "properties": {SEED_PROPERTIES_ATTRIBUTE: _SEED_PROPERTIES_MAPPING},
                },
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
                "type": {"type": "keyword"},
                "description": {"type": "text", "analyzer": self.analyzer},
                "description_embedding": self._get_knn_vector_mapping(),
                "weight": {"type": "double"},
                "rank": {"type": "double"},
                "text_unit_ids": {"type": "keyword"},
                "in_schema": {"type": "boolean"},
                "predicate_iri": {"type": "keyword"},
                "source_kinds": {"type": "keyword"},
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
                "start_date": {"type": "keyword"},
                "end_date": {"type": "keyword"},
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
