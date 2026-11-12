# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``indexing`` section: Neptune and OpenSearch indexing, summary trees."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from unified_kg_rag.domain.languages import BUILT_IN_ANALYZERS, PROFILES

from ..methodology import (
    DEFAULT_METHODOLOGIES,
    Methodology,
)
from ..retrieval import (
    validate_safe_name,
)


def _check_distinct_prefixes(prefixes: dict[str, str], *, fold: bool = False) -> None:
    """Refuse two store-name prefixes that can name the same store.

    A store name is ``<prefix>-<suffix>`` (``store_names.store_name``), so
    two equal prefixes share every store, and a prefix ``a-b`` names the
    store of prefix ``a`` under the suffix ``b-<suffix>``: another
    namespace's items would be written to it and read back as the wrong
    kind. ``fold`` compares the lower-cased prefixes (the graph labels are
    capitalized, ``store_names.graph_label``).
    """
    folded = {
        field: value.lower() if fold else value for field, value in prefixes.items()
    }
    for field, value in folded.items():
        for other, other_value in folded.items():
            if field != other and (
                value == other_value or other_value.startswith(f"{value}-")
            ):
                raise ValueError(
                    f"{other} '{prefixes[other]}' and {field} "
                    f"'{prefixes[field]}' can name the same store: one prefix"
                    " must not equal another or start with another followed "
                    "by '-'"
                )


class NeptuneIndexingConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    entity_label_prefix: str = Field(
        default="Entity",
        min_length=1,
        max_length=100,
        description="Prefix for entity node labels in Neptune (letters, digits, '-', '_')",
    )
    community_label_prefix: str = Field(
        default="Community",
        min_length=1,
        max_length=100,
        description="Prefix for community node labels in Neptune (letters, digits, '-', '_')",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description=("Items per Neptune write batch."),
    )
    index_concurrency: int = Field(
        default=1,
        ge=1,
        description=(
            "Neptune write batches (and relationship edges, one per request) "
            "submitted concurrently; the Gremlin connection pool grows to at "
            "least this. Above 1, batches touching the same vertices can raise"
            " ConcurrentModificationException. On Neptune (one transaction "
            "per request) the retry recovers, but the conflicts can make a "
            "large run slower than sequential writes, so measure before "
            "raising it. A local Gremlin Server has no transactions: a "
            "retried edge rewrite can leave a duplicate edge (3,025 edges "
            "for 3,000 ids in one measurement), so keep it at 1 there."
        ),
    )
    property_max_length: int = Field(
        default=4000,
        ge=1,
        description=(
            "Character cap per Neptune property value. Keep it above the "
            "longest description that is not re-summarized (summarization "
            "triggers above force_summary_threshold_tokens, 600 tokens or "
            "~2,400 English characters). OpenSearch keeps descriptions in "
            "full. Takes effect on re-ingestion."
        ),
    )
    max_hops: int = Field(
        default=3,
        ge=1,
        description=(
            "Depth of the graph expansion around the seed entities at query " "time."
        ),
    )
    max_results_per_hop: int = Field(
        default=50,
        ge=1,
        description="Maximum number of neighbours one node contributes per "
        "traversal hop, so a hub entity cannot fan the expansion out unbounded",
    )
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance orders entities by distance from the "
            "seeds (seeds first) and, at the same distance, by this "
            "importance: 'rank' (the indexed entity rank, its degree), "
            "'degree' (the entity's edges counted at query time, as MS "
            "GraphRAG ranks entities; read as 'rank' under a document filter, "
            "whose walk runs over the search store) or 'none' (a neutral 0.5 "
            "for every entity, so expansion ranks by distance alone)."
        ),
    )
    traversal_fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description=(
            "Graph expansion (the graph walk, or under a document filter the "
            "walk over the relationships index) reaches up to top_k * "
            "retrieval_multiplier * this many entities besides the seeds, "
            "split across the seeds (evenly; for local search's in proportion "
            "to their scores: each seed reaches at most its share), ranks "
            "them by relevance, then keeps every seed plus the "
            "top top_k * retrieval_multiplier of the others. 1 = no "
            "over-fetch: every reached entity is kept, with no cut after "
            "ranking."
        ),
    )

    deterministic_order: bool = Field(
        default=True,
        description=(
            "Neptune graph expansion takes each vertex's max_results_per_hop "
            "neighbours in id order, so two builds of one graph (edges "
            "written in another order) expand to the same entities: "
            "reproducible results across rebuilds. The cost: each expanded "
            "vertex has all its neighbours read and sorted before the cut, "
            "which grows with hub degree (Gremlin Server, 2,500-neighbour "
            "hubs at the defaults: 6.8 to 23.2 ms per query; 60-neighbour "
            "hubs: 4.5 to 4.8 ms; Neptune hub latency is unmeasured). false: "
            "the first neighbours the store returns (an order Neptune does "
            "not specify), so a rebuild can change the results. Of equally "
            "short paths to an entity, the one from the better-ranked seed "
            "is kept, then the first by content, either way. The embedded "
            "SQLite graph store always takes neighbours by id and ignores it."
        ),
    )

    vertex_ids: Literal["property", "custom"] = Field(
        default="property",
        description=(
            "How Neptune vertices are identified. 'property': the vertex id "
            "Neptune assigns, and the item id in an 'id' property that every "
            "lookup filters on. 'custom': the vertex id is '<label>:<item "
            "id>' (e.g. 'Entity-default:<entity id>'), so lookups go by "
            "vertex id, which Neptune serves faster and with fewer "
            "ConcurrentModificationExceptions than a property lookup; the "
            "'id' property is still written. Edges are unchanged. A graph "
            "is built in one mode: an ingestion run in the other mode is "
            "refused, and switching needs one full build (indexing.reset: "
            "true with incremental indexing on), which rewrites the graph. "
            "A query in the other mode, or during the switch, looks vertices "
            "up by the 'id' property instead (logged once). "
            "Neptune and Gremlin Server only: the embedded SQLite graph "
            "store ignores it."
        ),
    )

    @field_validator("entity_label_prefix", "community_label_prefix")
    @classmethod
    def _check_label_prefix(cls, value: str, info: ValidationInfo) -> str:
        # Part of every graph label (capitalized, see store_names.graph_label):
        # the index-name rule, in either case.
        try:
            validate_safe_name(value.lower(), str(info.field_name))
        except ValueError:
            raise ValueError(
                f"Invalid {info.field_name} '{value}': only letters, digits, "
                "hyphens, and underscores are allowed."
            ) from None
        return value

    @model_validator(mode="after")
    def _check_label_prefixes(self) -> "NeptuneIndexingConfig":
        _check_distinct_prefixes(
            {
                "entity_label_prefix": self.entity_label_prefix,
                "community_label_prefix": self.community_label_prefix,
            },
            fold=True,
        )
        return self


# The settings an index this package creates gets when index_settings leaves
# them out (the indexer adds each one missing from a configured map).
# refresh_interval trades search-visibility latency for indexing throughput.
DEFAULT_INDEX_SETTINGS: dict[str, Any] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "1s",
}

# The HNSW parameters of vector_search unless set; a key a configured map
# leaves out reads as its value here (vector_search_value).
# lucene HNSW natively supports the cosinesimil space type on all supported
# OpenSearch versions. faiss HNSW does NOT: it accepts only l2/innerproduct
# until OpenSearch 2.19 (cosine support) / 2.18 (auto-normalization), so
# faiss + cosinesimil is rejected at index creation with a
# mapper_parsing_exception on 2.13. lucene takes up to 16,000 dimensions from
# OpenSearch 2.12 (1,024 before), the same cap as faiss and nmslib, so the
# 2.13 default needs no engine switch for wide models
# (tests/integration/test_local_stores.py creates a 3,072-dim index).
DEFAULT_VECTOR_SEARCH: dict[str, Any] = {
    "engine": "lucene",
    "space_type": "cosinesimil",
    "ef_construction": 128,
    "m": 24,
    "ef_search": 100,
}
# The method a vector_search map without a name builds.
_DEFAULT_VECTOR_METHOD = "hnsw"

# The text analyzer of each LanguageCode, from its profile
# (domain/languages/profiles), and of the other codes OpenSearch has a
# built-in language analyzer for (BUILT_IN_ANALYZERS). All but nori (the
# analysis-nori plugin) are built into OpenSearch; cjk indexes Han and Kana
# runs as overlapping character bigrams, which needs no dictionary plugin
# (kuromoji, smartcn). The embedded engine knows every name
# (adapters/local/search_engine/analyzers.py).
DEFAULT_LANGUAGE_ANALYZERS: dict[str, str] = {
    **BUILT_IN_ANALYZERS,
    **{code.value: language.analyzer for code, language in PROFILES.items()},
}


def _stripped_unique(values: list[str], field: str) -> list[str]:
    """``values`` stripped, without repeats, in order; an empty one fails."""
    stripped = [value.strip() for value in values]
    if not all(stripped):
        raise ValueError(f"{field} entries must not be empty")
    return list(dict.fromkeys(stripped))


class OpenSearchIndexingConfig(BaseModel):
    # Settings set in code (config.indexing.opensearch.index_settings = ...)
    # are checked too, as IndexingConfig's are.
    model_config = ConfigDict(validate_assignment=True)

    text_units_index_prefix: str = Field(
        default="graphrag-text-units",
        min_length=1,
        max_length=100,
        description="Index name prefix for text unit documents (lowercase letters, digits, '-' and '_', not starting with '-' or '_')",
    )
    entities_index_prefix: str = Field(
        default="graphrag-entities",
        min_length=1,
        max_length=100,
        description="Index name prefix for entity documents (lowercase letters, digits, '-' and '_', not starting with '-' or '_')",
    )
    community_reports_index_prefix: str = Field(
        default="graphrag-community-reports",
        min_length=1,
        max_length=100,
        description="Index name prefix for community report documents (lowercase letters, digits, '-' and '_', not starting with '-' or '_')",
    )
    build_relationship_vector_index: bool = Field(
        default=True,
        description=(
            "Build the relationship vector index (embeds "
            "Relationship.description). LightRAG mix/hybrid query it for "
            "high-level keywords and incident edges, GraphRAG local for its "
            "relationship section and bridge relationships, and simple and "
            "drift search it with the other indexes; only global does without "
            "it. false skips the relationship embeddings at the cost of those."
            " The Neptune relationship edges are indexed either way. Without "
            "it, graph expansion under a document filter cannot check the "
            "edges it walks: an expanded entity is kept when its entity "
            "document passes the filter, also when only excluded documents "
            "connect it."
        ),
    )
    relationships_index_prefix: str = Field(
        default="graphrag-relationships",
        min_length=1,
        max_length=100,
        description="Index name prefix for relationship documents (LightRAG global retrieval) (lowercase letters, digits, '-' and '_', not starting with '-' or '_')",
    )
    claims_index_prefix: str = Field(
        default="graphrag-claims",
        min_length=1,
        max_length=100,
        description="Index name prefix for claim (covariate) documents (lowercase letters, digits, '-' and '_', not starting with '-' or '_')",
    )
    summary_nodes_index_prefix: str = Field(
        default="graphrag-summary-nodes",
        min_length=1,
        max_length=100,
        description="Index name prefix for the summary tree's nodes "
        "(built only for the raptor methodology, indexing.summary_tree) "
        "(lowercase letters, digits, '-' and '_', not starting with '-' or "
        "'_')",
    )
    extraction_records_index_prefix: str = Field(
        default="graphrag-extraction-records",
        min_length=1,
        max_length=100,
        description="Index name prefix for the extraction store: each text "
        "unit's extracted entities, relationships and claims, keyed by its "
        "text and the extraction settings, so an unchanged chunk is not "
        "extracted again (no embeddings). A full build or reset keeps only "
        "the records its text units name. It also holds the notes a "
        "removal (pending-strip, stripped-summaries) or a commit "
        "(pending-commit) keeps until "
        "it finished, which a resumed run needs: delete the index only "
        "after a run "
        "that succeeded, or run indexing.reset next. Lowercase letters, "
        "digits, '-' and '_', not starting with '-' or '_'",
    )
    default_analyzer: str = Field(
        default="standard",
        min_length=1,
        description="OpenSearch text analyzer used when the language has no specific mapping",
    )
    language_analyzers: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_LANGUAGE_ANALYZERS),
        description=(
            "Text analyzer per language code: by default every language with "
            "a profile has one (ar: arabic, de: german, en: english, es: "
            "spanish, fr: french, he: standard, hi: hindi, it: italian, ja: "
            "cjk, ko: nori, pt: portuguese, ru: russian, th: thai, vi: "
            "standard, zh: cjk), and so has every other BCP-47 code OpenSearch "
            "has a built-in language analyzer for (nl: dutch, sv: swedish, tr: "
            "turkish, ...: see Long defaults); a language without an entry "
            "uses default_analyzer. All are built into OpenSearch "
            "but nori, which needs the analysis-nori plugin: an Amazon "
            "OpenSearch Service domain has it only once its optional package "
            "is associated (the CDK stack maps ko to cjk). An index keeps the "
            "analyzers it was created with, so a change needs indexing.reset; "
            "a field built with default_analyzer before the default map named "
            "its language keeps it while the map gives that language the "
            "default's analyzer (one WARNING). "
            "In a config file the key replaces the whole map; an environment "
            "variable sets one entry over the defaults "
            "(UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO=cjk)."
        ),
    )
    refresh_after_batch: bool = Field(
        default=True,
        description="Whether to refresh OpenSearch indices after each batch operation for immediate visibility",
    )
    max_query_size: int = Field(
        default=100,
        ge=1,
        description="Maximum number of hits returned per OpenSearch query",
    )
    index_settings: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_INDEX_SETTINGS),
        description=(
            "OpenSearch index settings of the indices this package creates "
            "(number_of_shards, number_of_replicas, refresh_interval, "
            "max_result_window, max_terms_count, ...; with or without the "
            "index. prefix or under an index: map, but not the knn keys, which "
            "follow vector_search; the two limits are whole numbers >= 1); "
            "every key is passed to index creation (OpenSearch rejects one it "
            "does not know: before its first model call, each run that "
            "extracts creates and deletes, together, a probe index per index "
            "it will create (all of them on a full build or reset, the "
            "missing ones on an incremental run) with them and that index's "
            "fields, so a refused key, vector_search method, analyzer or a "
            "shard limit fails it before any model call; the embedded engine "
            "applies only max_result_window) and "
            "applies to indices created afterwards, so "
            "reindex after changing it. Reads and searches are sized by the "
            "smaller of max_result_window / max_terms_count here and on the "
            "live index (read again when a search is refused for them): a "
            "fetch by id of more ids goes in batches that fit. Raise "
            "refresh_interval "
            '(or set "-1") for faster bulk loads, then reset it for live '
            "querying. In a config file the key replaces the whole map (a "
            "left-out shard, replica or refresh setting is 1, 0 or 1s); an "
            "environment variable sets one entry over the defaults."
        ),
    )
    vector_search: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_VECTOR_SEARCH),
        description=(
            "HNSW algorithm parameters for approximate nearest neighbor search. "
            "ef_search is the candidate list size of a query: on faiss and "
            "nmslib the index setting index.knn.algo_param.ef_search (fixed "
            "when the index is created), on lucene, which ignores that "
            "setting, the k of every kNN query (at least the hits requested), "
            "read at query time."
        ),
    )
    vector_mode: Literal["in_memory", "on_disk"] = Field(
        default="in_memory",
        description=(
            "Vector workload mode of the knn_vector fields (OpenSearch 2.17+). "
            "on_disk keeps quantized vectors in memory and the full-precision "
            "ones on disk, and rescores each kNN search's candidates with "
            "them: far less memory at slightly more latency and a little "
            "recall, which pays off only at millions of vectors. in_memory "
            "(the default) sends no mode key. A changed mode applies to "
            "indices created afterwards: an incremental run refuses a live "
            "index built otherwise, so run once with indexing.reset: true. "
            "With the lucene engine set compression_level (4x); the embedded "
            "engine of local mode ignores it (exact search)."
        ),
    )
    compression_level: Literal["1x", "2x", "4x", "8x", "16x", "32x"] | None = Field(
        default=None,
        description=(
            "Quantization of the knn_vector fields as a memory reduction "
            "factor (OpenSearch 2.17+). Unset: the mode's own (on_disk: 32x "
            "on faiss; in_memory: 1x, no key sent). faiss takes 1x, 2x, 8x, "
            "16x and 32x, lucene 1x and 4x (lucene 8x-32x need a later "
            "OpenSearch); on_disk refuses 1x. faiss with cosinesimil needs "
            "OpenSearch 2.19. Needs the l2, innerproduct or cosinesimil space "
            "type and the hnsw method. on_disk rescores with OpenSearch's "
            "default oversampling; in_memory above 1x does not rescore. "
            "Applies to indices created afterwards (reset to change it on a "
            "live namespace)."
        ),
    )

    def vector_storage(self) -> dict[str, str]:
        """The ``mode`` / ``compression_level`` keys of a knn_vector mapping:
        empty for the default (in_memory, unquantized)."""
        storage: dict[str, str] = {}
        if self.vector_mode != "in_memory":
            storage["mode"] = self.vector_mode
        if self.compression_level is not None:
            storage["compression_level"] = self.compression_level
        return storage

    def vector_search_value(self, key: str) -> Any:
        """``vector_search[key]``, or its default (``DEFAULT_VECTOR_SEARCH``;
        ``hnsw`` for the method ``name``) when the map leaves it out."""
        defaults = {**DEFAULT_VECTOR_SEARCH, "name": _DEFAULT_VECTOR_METHOD}
        return self.vector_search.get(key, defaults[key])

    @model_validator(mode="after")
    def _check_vector_storage(self) -> "OpenSearchIndexingConfig":
        # Refused by every OpenSearch version: fail at config load instead
        # of at the probe. What one version takes and another does not
        # (2.17 for the keys, lucene 8x-32x, faiss + cosinesimil) is left to
        # the creation probe, which asks the live cluster.
        if not self.vector_storage():
            return self
        engine = self.vector_search_value("engine")
        space = self.vector_search_value("space_type")
        name = self.vector_search_value("name")
        prefix = "indexing.opensearch"
        if engine not in ("lucene", "faiss"):
            raise ValueError(
                f"{prefix}.vector_mode / compression_level need the lucene or "
                f"faiss engine (vector_search.engine is '{engine}')"
            )
        if name != "hnsw":
            raise ValueError(
                f"{prefix}.vector_mode / compression_level need the hnsw method "
                f"(vector_search.name is '{name}')"
            )
        if space not in ("l2", "innerproduct", "cosinesimil"):
            raise ValueError(
                f"{prefix}.vector_mode / compression_level need the l2, "
                f"innerproduct or cosinesimil space type (vector_search."
                f"space_type is '{space}')"
            )
        level = self.compression_level
        if self.vector_mode == "on_disk" and level == "1x":
            raise ValueError(
                f"{prefix}.compression_level 1x is refused with vector_mode "
                "on_disk: on_disk quantizes (leave compression_level unset for "
                "32x, or pick 2x-32x)"
            )
        if engine == "lucene" and level == "2x":
            raise ValueError(
                f"{prefix}.compression_level 2x needs the faiss engine (lucene "
                "takes 1x and 4x)"
            )
        if engine == "faiss" and level == "4x":
            raise ValueError(
                f"{prefix}.compression_level 4x needs the lucene engine (faiss "
                "takes 1x, 2x, 8x, 16x and 32x)"
            )
        if engine == "lucene" and self.vector_mode == "on_disk" and level is None:
            raise ValueError(
                f"{prefix}.vector_mode on_disk with the lucene engine needs "
                "compression_level (4x): the on_disk default, 32x, is a faiss "
                "quantization"
            )
        return self

    @field_validator(
        "text_units_index_prefix",
        "entities_index_prefix",
        "community_reports_index_prefix",
        "relationships_index_prefix",
        "claims_index_prefix",
        "summary_nodes_index_prefix",
        "extraction_records_index_prefix",
    )
    @classmethod
    def _check_index_prefix(cls, value: str, info: ValidationInfo) -> str:
        # Starts every index name: fail at config load, not at indexing after
        # the extraction was paid for. OpenSearch also rejects a name that
        # starts with '-' or '_'.
        validate_safe_name(value, str(info.field_name))
        if value[0] in "-_":
            raise ValueError(f"{info.field_name} must not start with '-' or '_'")
        return value

    @model_validator(mode="after")
    def _check_index_prefixes(self) -> "OpenSearchIndexingConfig":
        _check_distinct_prefixes(
            {
                field: getattr(self, field)
                for field in type(self).model_fields
                if field.endswith("_index_prefix")
            }
        )
        return self

    @field_validator("index_settings")
    @classmethod
    def _check_index_settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        # A setting may be written with or without its "index." prefix, or
        # under a nested "index" mapping (as OpenSearch takes it); the
        # indexer writes it without. The knn settings follow vector_search.
        nested = value.get("index")
        if isinstance(nested, dict):
            flat = {k: v for k, v in value.items() if k != "index"}
            for key, setting in nested.items():
                if f"index.{key}" in flat:
                    raise ValueError(
                        f"index_settings sets 'index.{key}' twice (nested and flat)"
                    )
                flat[f"index.{key}"] = setting
            value = flat
        names: dict[str, str] = {}
        for key in value:
            name = key.removeprefix("index.")
            if name == "knn" or name.startswith("knn."):
                raise ValueError(
                    f"index_settings must not set '{key}': the indexer enables "
                    "knn and sets its ef_search from vector_search "
                    "(vector_search.ef_search)"
                )
            if name in names:
                raise ValueError(
                    f"index_settings sets '{name}' twice ('{names[name]}' and "
                    f"'{key}')"
                )
            names[name] = key
            if name in ("max_result_window", "max_terms_count"):
                # Read back as numbers to size reads and searches.
                setting = value[key]
                if isinstance(setting, str) and setting.strip().isdigit():
                    setting = int(setting)
                if isinstance(setting, bool) or not isinstance(setting, int):
                    setting = 0
                if setting < 1:
                    raise ValueError(
                        f"index_settings.{key} must be a whole number of at least 1"
                    )
        return value


class CommunityCompactionConfig(BaseModel):
    """When an incremental run recommends re-clustering the communities.

    Incremental runs append the communities of their delta instead of
    re-clustering the graph; a full build (indexing.reset: true) clusters it
    again. Nothing is reset automatically.
    """

    model_config = ConfigDict(validate_assignment=True)

    max_runs_since_clustering: int | None = Field(
        default=20,
        ge=1,
        description="Recommend indexing.reset: true (a WARNING after the "
        "commit) once this many incremental commits ran since the last full "
        "build clustered the graph. A suffix whose full build predates the "
        "counters starts counting at its next full build. null turns this "
        "check off.",
    )
    max_community_growth: float | None = Field(
        default=1.5,
        gt=1.0,
        description="Recommend indexing.reset: true once the suffix holds "
        "this many times the communities its last full build clustered "
        "(each commit adds the communities it appended and subtracts those "
        "its removals deleted; the community_growth_ratio metric). null "
        "turns this check off.",
    )
    on_drift_recommendation: bool = Field(
        default=False,
        description="Also give this recommendation when the run's community "
        "drift recommends a rebuild (indexing.community_drift_threshold, "
        "or a kept report citing removed text), which logs its own WARNING "
        "either way.",
    )


class SummaryTreeConfig(BaseModel):
    """RAPTOR's summary tree over the stored text units (opt-in).

    Built when ``indexing.methodologies`` selects ``raptor`` (not a
    default): after indexing, the ``summary_tree`` stage clusters the
    namespace's stored text units by their vectors, summarizes each cluster
    with ``models.roles.summary_tree`` in the index language, clusters the
    summaries again, and writes the tree into its own index
    (``indexing.opensearch.summary_nodes_index_prefix``), which the
    ``tree`` strategy searches. The tree is rebuilt whenever the stored text
    units or these settings change; a cluster whose children are unchanged
    reuses its summary and makes no model call. Adding ``raptor`` to an
    existing namespace builds only the tree, from the stored text units
    (``run-ingestion --enabled-stages summary_tree`` builds it without
    re-ingesting); a call is about one per five or six text units.
    """

    model_config = ConfigDict(validate_assignment=True)

    scope: Literal["corpus", "document"] = Field(
        default="corpus",
        description="corpus: one tree over every text unit of the index "
        "namespace (one per partition_by value tuple); document: one tree "
        "per document (RAPTOR's own setting), none for a document of fewer "
        "than min_leaves_per_tree text units.",
    )
    partition_by: list[str] = Field(
        default_factory=list,
        description="Document filter keys (a document's metadata filters) "
        "whose value tuple splits the corpus into separate trees, so every "
        "node is pure on them and survives a filter on them (a summary is "
        "kept under a filter only when all its documents pass it).",
    )
    algorithm: Literal["kmeans", "gmm"] = Field(
        default="kmeans",
        description="kmeans: hard k-means over the PCA-reduced vectors, "
        "k = ceil(n / target_children) per level. gmm: RAPTOR's Gaussian "
        "mixture with BIC-chosen k, global then local, soft membership "
        "(soft_threshold). Both are numpy-only and seeded (random_state).",
    )
    target_children: int = Field(
        default=6,
        ge=2,
        le=50,
        description="Children per summary node aimed at (kmeans' k, and the "
        "centre of gmm's bounded BIC grid). RAPTOR's trees average 6.7.",
    )
    soft_threshold: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="gmm only: a node joins every cluster whose posterior "
        "exceeds this (RAPTOR's threshold), its likeliest one when none does.",
    )
    reduction_dims: int = Field(
        default=10,
        ge=2,
        le=100,
        description="PCA dimensions the vectors are reduced to before "
        "clustering (RAPTOR's reduction_dimension; PCA replaces its UMAP).",
    )
    max_levels: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Summary levels above the text units at most (RAPTOR's "
        "num_layers).",
    )
    min_nodes_to_cluster: int = Field(
        default=11,
        ge=2,
        description="A level of at most this many nodes is not clustered "
        "again: its nodes are the roots (RAPTOR's reduction_dimension + 1).",
    )
    min_level_shrink: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Stop when the next level would hold more than "
        "(1 - this) of the current one's nodes: no structure is left to "
        "summarize. 0 never stops for it (RAPTOR's behaviour).",
    )
    max_cluster_tokens: int = Field(
        default=8000,
        ge=256,
        description="A cluster whose children hold more tokens than this "
        "(the summary prompt's input) is clustered again until it fits "
        "(RAPTOR's max_length_in_cluster, 3500 cl100k tokens).",
    )
    max_summary_tokens: int = Field(
        default=300,
        ge=32,
        le=4000,
        description="Length a summary aims at; a longer answer is cut to it.",
    )
    max_lineage_text_units: int = Field(
        default=10000,
        ge=100,
        description="Text-unit ids a node stores at most (its provenance and "
        "removal key); above it the node keeps its child ids and the "
        "lineage is resolved through them at query time.",
    )
    min_leaves_per_tree: int = Field(
        default=3,
        ge=2,
        description="scope: document only: a document with fewer text units "
        "gets no tree.",
    )
    refresh: Literal["every_run", "on_reset"] = Field(
        default="every_run",
        description="every_run: rebuild the tree on every run whose stored "
        "text units changed. on_reset: delta runs only delete the nodes of "
        "removed text (always done) and report the text units no summary "
        "covers (tree_uncovered_leaves); indexing.reset rebuilds.",
    )
    random_state: int = Field(
        default=42, description="Seed of the clustering's initializations."
    )
    max_leaves_in_memory: int = Field(
        default=100_000,
        ge=1,
        description="Text units one tree may hold, read into memory with "
        "their texts and vectors and clustered there, and the trees read "
        "at once hold (several trees are read and built a batch at a time): "
        "about 50 KiB of task memory per text unit at the peak "
        "(4,000-character units, 1,024 dimensions; read 15, build 35), so "
        "the default takes about 5 GiB. A larger tree fails the stage before "
        "its vectors are read, naming partition_by and scope: document; "
        "raise it with the task's memory.",
    )

    @field_validator("partition_by")
    @classmethod
    def _strip_partition_keys(cls, keys: list[str]) -> list[str]:
        return _stripped_unique(keys, "partition_by")


class IndexingConfig(BaseModel):
    # A suffix set in code (config.indexing.additional_suffix = ...) is
    # checked too: an index name the engine refuses read as a refused index
    # setting.
    model_config = ConfigDict(validate_assignment=True)

    methodologies: list[Methodology] = Field(
        default_factory=lambda: list(DEFAULT_METHODOLOGIES),
        min_length=1,
        description=(
            "The retrieval methodologies ingestion builds: graphrag (local, "
            "global, drift, simple; communities and community reports), "
            "lightrag (mix, hybrid, naive; the relationship vector index), "
            "hipporag (ppr; what lightrag builds), raptor (tree; the text "
            "units and a summary tree over them, indexing.summary_tree, "
            "without graph extraction). naive and simple read what every "
            "methodology builds; iterative reads what its base reads. The "
            "default [graphrag, lightrag] also holds hipporag, and so does "
            "[graphrag] while its relationship vectors are built. A run builds only what the "
            "selected ones read: without graphrag no community detection "
            "and no report calls. A query whose strategy reads a methodology "
            "the index namespace does not hold fails before any model call. "
            "Adding a graph methodology to an incremental namespace rebuilds "
            "it once from its stored extractions (only the added "
            "methodology's model calls are new; a raptor-only namespace "
            "stores none, so every text unit is extracted); removing one "
            "stops maintaining its artifacts "
            "without deleting them. Adding a graph methodology and removing "
            "any are refused while the namespace "
            "holds other source scopes' documents. "
            "graph.community_detection.enabled: "
            "false leaves out graphrag and "
            "indexing.opensearch.build_relationship_vector_index: false "
            "lightrag and hipporag. Adding raptor builds only its tree, from "
            "the stored text units."
        ),
    )
    reset: bool = Field(
        default=False,
        description=(
            "Rebuild the run's index namespace from the whole corpus instead "
            "of applying a delta, without taking it offline: the vector "
            "indices swap aliases, the graph is rewritten in place and what "
            "the rebuild did not write is dropped at the end; then the "
            "namespace's registry records are written again. Quarantined "
            "documents are skipped and keep their records. Set it for one "
            "run."
        ),
    )
    additional_suffix: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "Appended after the run's suffix in every OpenSearch index name "
            "and Neptune label (<prefix>-<suffix>-<additional_suffix>; the "
            "suffix is processing.document_parsing.index_value at ingestion "
            "and --suffix at query time, else index_value too, all 'default' "
            "when unset). Set it "
            "identically for ingestion and queries, for versioned or "
            "multi-tenant separation. Lowercase letters, digits, '-' and '_'."
        ),
    )
    community_drift_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="On incremental runs with communities on, recommend a "
        "rebuild (WARNING and the community_rebuild_recommended metric) when "
        "at least this share of the entities the run clustered also belongs "
        "to an earlier level-0 community. A delta run clusters only its own "
        "documents, so such an entity sits in two communities and the earlier "
        "community's report misses what the run added; indexing.reset: true "
        "re-clusters the whole graph.",
    )
    community_compaction: CommunityCompactionConfig = Field(
        default_factory=CommunityCompactionConfig,
        description="Incremental runs: when to recommend re-clustering the "
        "communities with indexing.reset: true (a WARNING and the "
        "community_growth_ratio and runs_since_clustering metrics).",
    )
    neptune: NeptuneIndexingConfig = Field(
        default_factory=NeptuneIndexingConfig,
        description="Configuration settings for Neptune graph database indexing",
    )
    opensearch: OpenSearchIndexingConfig = Field(
        default_factory=OpenSearchIndexingConfig,
        description="Configuration settings for OpenSearch vector and text indexing",
    )
    max_failure_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Per-index-type write failure rate above which the indexing stage "
            "fails, and share of communities without a report above which "
            "community detection fails, recording no document; the failed "
            "items are noted and written again by the next run (below it, a "
            "document whose own text units failed is recorded FAILED and "
            "retried). A fully failed index type always fails the stage. 1.0 "
            "disables the partial-failure gate."
        ),
    )
    max_document_failures: int = Field(
        default=3,
        ge=1,
        description="Incremental runs: a document recorded FAILED for an error "
        "of its own this many consecutive runs with unchanged content, under "
        "the same extraction settings, is no longer retried. It stays "
        "FAILED and is skipped as unchanged (with a WARNING) until its content "
        "changes or this value is raised. Output that fails alike on several "
        "documents (a server-side output cap) counts so too, once a run; "
        "changed extraction settings start the count over. A unit's own "
        "claim extraction error never counts: the document is indexed "
        "without those claims. A claim extraction outage fails the document "
        "uncounted, as any outage does.",
    )
    max_document_attempts: int = Field(
        default=3,
        ge=1,
        description="Incremental runs: once runs died this many consecutive "
        "times (the task was killed; a stopped or failed run does not count, "
        "nor one that lost its write lease unless another run held it when "
        "it ended) "
        "processing a document's unchanged content, the runs isolate it: a "
        "document that died processed alone "
        "is quarantined (the next run removes what the attempts wrote, "
        "records it QUARANTINED and skips it, a reset included, until its "
        "content changes or this value is raised); otherwise the "
        "next runs process half of the documents it died with and defer the "
        "other new and changed documents until the culprit is alone.",
    )
    max_deletion_share: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Incremental runs: when a run would delete more than "
        "this share of the documents the doc-status registry holds for its "
        "source scope (files gone from the corpus: a partial S3 sync, an "
        "expired prefix, a volume not mounted look the same), it stops at "
        "document_loading before any write unless the deletion is "
        "confirmed (run-ingestion asks on the terminal; --confirm-deletions, "
        "UKR_CONFIRM_DELETIONS=1 or --yes answers yes). Only scopes holding "
        "at least 10 documents are checked; retired scopes "
        "(retire_source_scopes) and seed graph shards (the configuration "
        "adds and removes them) do not count, and a reset or a run with the "
        "registry off is never checked. 1.0 or null disables the check.",
    )
    lease_seconds: float = Field(
        default=600.0,
        ge=10.0,
        description="Incremental runs: the term of an index namespace's write "
        "lease, which a run holds from its first stage to its end so no other "
        "run writes the namespace meanwhile. The run renews it every sixth of "
        "the term and stops writing once less than a third is left since its "
        "last renewal, so keep it at least three times the longest store "
        "write request (180 s; a run under 540 s logs a WARNING); a run that "
        "dies stops renewing, so its lease expires (and another run can "
        "start) at most this long after. A stopped run (SIGTERM, Ctrl-C) "
        "gives it up expired, so the next run holds its first write about "
        "185 s for the stopped run's requests in flight (with every store "
        "embedded it releases it instead).",
    )
    retire_source_scopes: list[str] = Field(
        default_factory=list,
        description="Incremental runs only: source scopes (a former "
        "document_parsing.source_scope value, or the absolute source directory "
        "a corpus was indexed from before it moved) whose registry records in "
        "this run's index namespace are treated as deleted. The run removes "
        "their exclusive artifacts (shared ones are kept) and deletes their "
        "records once the removal succeeded. Never the run's own scope. "
        "run-ingestion --retire-source-scope adds to this list.",
    )

    summary_tree: SummaryTreeConfig = Field(
        default_factory=SummaryTreeConfig,
        description="RAPTOR's summary tree over the stored text units, which "
        "the tree strategy searches (built for the raptor methodology).",
    )

    @field_validator("methodologies", mode="before")
    @classmethod
    def _check_methodologies(cls, value: Any) -> Any:
        # Named here, with the valid ones, before the enum check's generic
        # message; a duplicate is dropped.
        if not isinstance(value, list | tuple):
            return value
        known = {m.value: m for m in Methodology}
        names = ", ".join(known)
        selected: list[Methodology] = []
        for item in value:
            name = item.value if isinstance(item, Methodology) else item
            methodology = known.get(str(name).strip().lower())
            if methodology is None:
                raise ValueError(f"unknown methodology {name!r}; valid: {names}")
            if methodology not in selected:
                selected.append(methodology)
        return selected

    @field_validator("additional_suffix")
    @classmethod
    def _check_additional_suffix(cls, value: str | None) -> str | None:
        # Part of every index name: fail at config load, not at indexing
        # after the extraction was paid for.
        return (
            value if value is None else validate_safe_name(value, "additional_suffix")
        )

    @field_validator("retire_source_scopes")
    @classmethod
    def _strip_retired_scopes(cls, scopes: list[str]) -> list[str]:
        return _stripped_unique(scopes, "retire_source_scopes")
