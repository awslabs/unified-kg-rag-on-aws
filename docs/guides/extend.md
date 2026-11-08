# Extending the framework

Section 15 of the [design reference](../design.md): how to add a backend, parser, renderer, schema or seed source as a
plugin, or a search strategy or evaluator in-tree.

## 15. Extension Guide

This section is the single reference for extending the framework. Extensions come in two kinds:

- **Plugins**, which live in your own package and import only from `unified_kg_rag` (§2.5): backends through the provider registry or constructor injection, and renderers, parsers, schema sources and seed sources through their registries. None needs a change to the framework's code.
- **In-tree changes**, made in a fork or contributed upstream: a search strategy and an evaluator. `register_strategy` is keyed by the closed `SearchStrategy` enum, which `RAGInput.search_strategy`, `--search-strategy` and `search.auto_routable_strategies` validate against, so a new strategy adds an enum member; an evaluator adds an `EvaluatorType` member and a branch in `EvaluationManager._resolve_evaluator_class`. The registry still keeps the chain's dispatch code unchanged.

- **New search strategy** (in-tree): Add a `SearchStrategy` enum member (`domain/models/retrieval.py`; the CLI choices follow it) + subclass `BaseSearchStrategy` + `@register_strategy(SearchStrategy.X, required_roles=(...), query_inputs=frozenset({QueryInput.ENTITIES}))` + export from `adapters/search_strategies/__init__.py` + an entry in `STRATEGY_ROUTING_GUIDE` (`domain/prompts/retrieval.py`; a unit test requires one for every strategy but `auto`). `query_inputs` declares the query-side LLM extractions it reads (`ENTITIES` for `entity_focus`, `DUAL_KEYWORDS` for `hl_keywords`/`ll_keywords`); the chain skips the rest. Set the class attribute `methodology` (a `Methodology`: the one whose index artifacts it reads, §4.6) so a query on a namespace that does not hold it is refused before any model call and `auto` routes to it only where it is held; a new methodology adds a `Methodology` member, its `METHODOLOGY_PROFILES` entry (required and optional `IndexArtifact`s) and, for an artifact no stage builds yet, the stage and its `ARTIFACT_STAGES`/`ARTIFACT_PRODUCERS` entries (`domain/ingestion/methodologies.py`; a corpus-wide artifact also joins `CORPUS_WIDE_ARTIFACTS`). A strategy that runs another one's search (as `iterative` runs `local` or `mix`) passes `wraps=` a function of the config naming it: the chain then extracts what that one reads and packs its context as that one does (`wrapped_strategy`), the wrapper builds its base through `get_strategy_spec(...).strategy_class`, not by name, and it declares no `methodology` of its own: it reads the wrapped one's (`strategy_registry.strategy_methodology`). Fetch by id through `BaseSearchStrategy._fetch_by_ids` (batched by the live limits), and cut or batch any other list filter (`terms`) to the live `max_terms_count` from `aindex_limits` of the index it reads: a longer list is refused (`InvalidFilterError`). Override `filtered_index_prefixes` with the indexes it searches with the caller's filters, so a filter key none of them declares is refused instead of dropped (the default checks every index), and `filtered_retrievers` with the retrievers it applies them to (the default: every one it holds; `mix` and `hybrid` leave out the graph without graph expansion); build each sub-search's filters with `_scoped_filters`, which keeps the caller's and intersects a key both set (`id`/`source_id`/`target_id`: the scope's values the caller allows; the chain refuses a range on them); `reserved_filter_keys` names the keys it scopes by with values a caller value cannot be intersected with (`global`: its cut ranges or browse from the roots, and the multi-valued `community_ids`, where a text unit in a caller's community and a scope's one passes both filters but the two lists may not intersect), which the chain refuses; a strategy that may answer with another one's search in its place (`structured`'s fallback) names it in `filter_delegates`, and the chain checks the filter against that one too. No edit to `rag_chain`; add it to `search.auto_routable_strategies` if `auto` may pick it. Regenerate the references (the new member is a `--search-strategy` choice and a config value): `uv run python scripts/gen_cli_reference.py` and `uv run python scripts/gen_config_reference.py`. A minimal strategy, a document search fused and reranked like `simple` (`adapters/search_strategies/simple_search.py` is the full version, with the index gates):

  ```python
  # unified_kg_rag/adapters/search_strategies/keyword_search.py
  # (after adding SearchStrategy.KEYWORD = "keyword" to domain/models/retrieval.py)
  import time

  from langchain_core.runnables import RunnableConfig

  from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy
  from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
  from unified_kg_rag.domain.models import (
      Methodology,
      RetrieverRole,
      SearchQuery,
      SearchResult,
      SearchStrategy,
  )


  @register_strategy(SearchStrategy.KEYWORD, required_roles=(RetrieverRole.DOCUMENT,))
  class KeywordSearchStrategy(BaseSearchStrategy):
      methodology = Methodology.LIGHTRAG  # reads text units only

      async def asearch(
          self, query: SearchQuery, config: RunnableConfig | None = None
      ) -> SearchResult:
          start, rerank = time.time(), self._new_rerank_trace()
          hits = []
          if self.document_retriever:  # bound through required_roles
              hits = await self._safe_aretrieve(
                  self.document_retriever, query, "Document retrieval"
              )
          results = await self._fuse_and_rerank(
              {"documents": hits},
              rerank_trace=rerank,
              top_k=query.top_k,
              retrieval_multiplier=query.retrieval_multiplier,
              query=query.query,
          )
          return SearchResult(
              query=query,
              results=results,
              total_results=len(results),
              search_strategy="keyword_search",
              processing_time=time.time() - start,
              metadata={"rerank": rerank.summary()},
          )
  ```
- **New storage/LLM backend**: Implement the relevant port, then either register a builder for a new `providers.<role>.kind` with `@register_provider(ProviderRole.X, "kind")` (`adapters/provider_registry.py`) and select it in config, or pass the port to the constructor that uses it (see "Custom backends" below). Do not hardcode it into a manager's `__init__`.
- **New evaluator** (in-tree): Subclass `BaseGraphRAGEvaluator` (`from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator`) + add a branch in `EvaluationManager._resolve_evaluator_class` + an `EvaluatorType` enum. Regenerate the configuration reference, whose `evaluation.enabled_evaluators` lists the types: `uv run python scripts/gen_config_reference.py`.
- **New renderer**: Subclass `BaseRenderer` + `@register_renderer("name")`. Registration happens on import: `run-visualization` sees those that `adapters/renderers/__init__.py` imports. Override `check_dependencies` to fail fast when an optional package the renderer imports is missing, and set `needs_centrality = True` when `render` reads `RenderContext.centrality`: `run-visualization` computes the centrality metrics (betweenness alone is O(V·E)) only when a selected renderer sets it. Regenerate the CLI reference, where the name is a `--renderers` choice: `uv run python scripts/gen_cli_reference.py` (and `uv run python scripts/gen_config_reference.py` when it adds a `graph.visualization` settings section).
- **New parser / file format**: `ParserFactory.register_loader(".ext", MyLangChainLoader, loader_kwargs=..., file_type_name=...)` — any `langchain_core` `BaseLoader` subclass (the built-in loaders in `adapters/ingestion/file_loaders.py` are examples); no edit to the factory, and the extension is then auto-discovered + parseable. Override a built-in by registering its extension.
- **New schema source** (an ontology format or service): register a loader
  for a new `schema_source.kind` with `@register_schema_source("kind")`
  (exported from `unified_kg_rag`; the built-in `file` and `owl` loaders
  live in `adapters/schema_sources/`, imported by the package `__init__`,
  and read an `s3://` path through `adapters/source_files/locations.py`:
  HeadObject size check, paged prefix listing, S3 errors as
  `SchemaSourceError`, or the error type a `SourceTerms` names; what the
  built-in sources read: [Extraction schema](./extraction-schema.md),
  [Ontology mapping](../reference/ontology-mapping.md)).
  A loader that imports an optional package declares it:
  `@register_schema_source("kind", requires=("module", "how to install
  it"))`, and the CLIs check that the module is importable before any model
  call, exiting 2 with the install hint (`owl` declares `("rdflib", "the
  ontology extra: uv sync --extra ontology ...")`). Import the package
  inside the loader, not at module level, so registering the kind needs
  nothing, and raise `SchemaSourceError` naming the extra when the import
  fails; re-registering a kind replaces its loader and its requirement. It
  receives a `SchemaSourceContext` and returns a `SchemaDocument`: the
  `entity_types` (or `None` to keep the config's), `relationship_types` and
  `relationship_patterns` it supplies, each a string or a spec. The context
  holds:
  - `source`: the `schema_source` settings: `kind`, `path` (left as written
    when it is a URL), `options` (a free mapping for the loader's own
    settings, not keyed: the loaded content is) and `secret_arn` (a Secrets
    Manager secret to read credentials from at load time).
  - `config`: the run's config (its schema keys are not loaded yet).
  - `boto_session()`: the run's AWS session (the injected `Providers`
    one, else one from `aws.profile_name` and `aws.region_name`), built on
    the first call only, and `client(service)`, a client in the region the
    config sends that service to (for Bedrock `aws.bedrock.region_name`,
    else `aws.region_name`, read as the `config.aws.bedrock_region`
    property; `aws.region_name` for the rest).
  - `location`: `path` without the userinfo, query and fragment of a URL.
    Log and raise with it, never with `path` or the secret's value.

  The loader must return the types in the same order for the same content
  (sort what a set, a graph or a paged API yields): the order is part of
  the prompt and of the cache keys, so a reordered snapshot re-extracts the
  corpus. It translates, it does not reason.

  **Reusing the ontology mapping.** The `owl` source is split so that its
  mapping does not depend on RDF:
  - `ontology_model.py`: `OntologyModel`, a pure Pydantic model of the named
    classes and properties a reader found and what it stated about them
    (annotations as language-tagged `Text`, parents, equivalents,
    deprecation; per property its kind, domains and ranges as sets of
    unions, inverses and symmetry) plus the prefix bindings. Statements,
    not inferences, and no blank nodes.
  - `ontology_mapping.py`: `MappingOptions` (selection, labels, collisions,
    descriptions and hints, patterns, caps) and `map_ontology(model,
    options) -> OntologyMapping` (`document`, the `SchemaDocument`, and
    `report`, a `MappingReport` of what was dropped, lifted, merged,
    skipped or qualified). It owns every rule: the RDFS-style closure, the
    flat hierarchy with push-down and lift, labels, order and the caps,
    and depends only on `domain`.
  - `owl_reader.py` builds the model from parsed statements; `owl.py` is
    the loader: it locates and checks the files (formats, size, and the
    shapes the parsers take quadratic time over, refused before parsing),
    parses their bytes, reads the model and maps it with
    `OwlSourceOptions`, which is `MappingOptions` plus the reader's
    `format` and `max_bytes`.
  - The file handling is not owl's own: `adapters/source_files/` holds it
    for any source that reads RDF. `rdf_files.locate`/`locate_s3` pick the
    files and the format of each, `rdf_files.check_bytes` refuses what a
    parser must not see, and `rdf_parse.parse` (the only module there
    importing rdflib) parses the bytes; `rdf_terms` holds the RDF
    vocabulary IRIs and the pure naming helpers (`local_name`,
    `upper_snake`, `best_text`, `Text`, ...) the mapping shares with the
    `rdf` seed source. A `SourceTerms` record carries the exception type
    and how messages name the source, its options, its size cap
    (`max_bytes`, which a seed's terms name `processing.seed_graphs[].max_bytes`)
    and its kind, and any `SourceClientProvider` (`location`, `client(service)`)
    gives `locations.SourceReader` its S3 client.

  A loader for an ontology held elsewhere (an ontology service, or a
  non-RDF format such as LinkML) added to `adapters/schema_sources/` builds
  the same `OntologyModel` from its snapshot and calls `map_ontology`, so
  its users get the same labels, collisions, closure, patterns, hints,
  order and caps as `kind: owl`, and the same IRIs end up in `type_iri` and
  `predicate_iri`. Its options model subclasses `MappingOptions` with the
  reader's own settings, validated from `context.source.options` (wrap a
  `ValidationError` in `SchemaSourceError`, as `owl._options` does).
  These modules are unstable internals, not part of the public API (§2.5):
  `unified_kg_rag` does not export `map_ontology`, `MappingOptions`,
  `OntologyModel`, `OntologyClass`, `OntologyProperty` or the vocabulary
  constants, and their names, fields and mapping rules may change in any
  release without a "Breaking" entry. A loader that imports them pins the
  package version and runs its own tests on each upgrade:

  ```python
  from pydantic import Field

  from unified_kg_rag import SchemaDocument, SchemaSourceContext, register_schema_source
  from unified_kg_rag.adapters.schema_sources.ontology_mapping import (
      MappingOptions,
      map_ontology,
  )
  from unified_kg_rag.adapters.schema_sources.ontology_model import (
      RDFS_LABEL,
      OntologyClass,
      OntologyModel,
      OntologyProperty,
      Text,
  )


  class ServiceOptions(MappingOptions):
      graph: str = Field(description="The service's graph to read")


  @register_schema_source("ontology_service")
  def load_ontology_service(context: SchemaSourceContext) -> SchemaDocument:
      options = ServiceOptions.model_validate(context.source.options)
      snapshot = fetch_snapshot(context.source.path, graph=options.graph)
      model = OntologyModel(
          classes=tuple(
              OntologyClass(
                  iri=c["iri"],
                  annotations={RDFS_LABEL: (Text(value=c["name"], language="en"),)},
                  parents=frozenset(c.get("parents", ())),
              )
              for c in snapshot["classes"]
          ),
          properties=tuple(
              OntologyProperty(
                  iri=p["iri"],
                  domains=frozenset({frozenset(p["domain"])}),
                  ranges=frozenset({frozenset(p["range"])}),
              )
              for p in snapshot["properties"]
          ),
      )
      return map_ontology(model, options).document
  ```

  The order of the model's terms does not matter (`map_ontology` sorts).
  A loader that maps its snapshot itself returns a `SchemaDocument`
  directly, as a SHACL loader mapping node shapes would. A service loader
  (an ontology API, or an MCP server's schema tool) fetches its snapshot
  once:

  ```python
  import json

  from unified_kg_rag import (
      SchemaDocument,
      SchemaSourceContext,
      SchemaSourceError,
      register_schema_source,
  )

  @register_schema_source("ontology_service")
  def load_ontology_service(context: SchemaSourceContext) -> SchemaDocument:
      source = context.source
      token = None
      if source.secret_arn:
          secret = context.client("secretsmanager").get_secret_value(
              SecretId=source.secret_arn
          )
          token = json.loads(secret["SecretString"])["token"]
      try:
          # fetch_snapshot: your client of the service (HTTP, or an MCP
          # session calling the server's schema tool).
          snapshot = fetch_snapshot(
              source.path, graph=source.options.get("graph"), token=token
          )
      except OSError as e:
          raise SchemaSourceError(
              f"Cannot read the ontology service at '{context.location}': {e}"
          ) from e
      return SchemaDocument(
          entity_types=[
              {"label": c["name"], "description": c.get("doc", ""), "iri": c["iri"]}
              for c in sorted(snapshot["classes"], key=lambda c: c["iri"])
          ],
          relationship_types=[
              {"label": p["name"], "iri": p["iri"]}
              for p in sorted(snapshot["properties"], key=lambda p: p["iri"])
          ],
      )
  ```

  selected with `schema_source: {kind: ontology_service, path:
  "https://ontology.example.com/v1", options: {graph: products},
  secret_arn: ...}`. Registration happens on import, and the CLIs import
  only the built-in sources, so a run with a custom source starts from
  Python after importing its module. Nothing else changes: start-up writes
  the result into the inline keys, the config validates it (unique labels,
  patterns naming declared types), and the content keys the caches, so the
  snapshot a run used is the one it is keyed by. Raise `SchemaSourceError`
  for anything unreadable; it is a `ConfigurationError`, so the CLIs exit 2.
- **New seed source** (a graph format, or a service holding a graph):
  register a loader for a new `processing.seed_graphs[].kind` with
  `@register_seed_source("kind")` (exported from `unified_kg_rag` with
  `SeedSourceContext`, `SeedSourceOptions`, `SeedGraph`, `SeedNode`,
  `SeedEdge` and `SeedGraphError`; the built-in `jsonl`, `csv` and `rdf` loaders live in
  `adapters/seed_sources/`, imported by the package `__init__`; their
  formats: [Seed graph formats](../reference/seed-graph-formats.md)). A loader
  that imports an optional package declares it, `requires=("module", "how
  to install it")`, and the CLIs check it before any model call, as for
  schema sources (`rdf` declares rdflib). The loader receives a
  `SeedSourceContext` and returns a `SeedGraph` of `SeedNode`s and
  `SeedEdge`s:
  - `source`: the `processing.seed_graphs` item (`name`, `kind`, `path`,
    left as written when it is a URL, `options`, `secret_arn` and the
    caps). Validate `options` with `context.options(Model)`, a subclass of
    `SeedSourceOptions` (`from unified_kg_rag import SeedSourceOptions`),
    which carries `dangling` and `qualifier_property` and forbids unknown
    keys.
  - `config`: the run's config, the extraction schema already in the
    inline keys (`config.processing.graph_extraction.extraction_schema()`),
    for a loader that maps class IRIs itself.
  - `boto_session()`, `client(service)` and `location`, as on
    `SchemaSourceContext`; and `terms`, whose call builds a
    `SeedGraphError` naming the seed (`raise context.terms("...")`). Log
    and raise with `location`, never with `path` or the secret's value.

  What it returns must depend only on what it read and the options, never
  on the order (`convert_seed` sorts, applies the schema, the collisions,
  the caps and the sharding). Ids are the seed's own (edges name them);
  identity in the graph is the node's name. Nothing else changes: the
  conversion's digest keys the stages, and the shards are documents of the
  run's corpus (§4.5).

  **Reusing the readers.** `adapters/source_files/` serves seed sources as
  it serves schema sources: `locations.resolve_source_path` turns a `path`
  into a local `Path` (a plain path or a `file://` URL) or the `s3://` URL,
  refusing other schemes, as every built-in kind does;
  `locations.SourceReader(context, terms=context.terms)` heads, lists and
  reads S3 objects within a byte cap, and `rdf_files` / `rdf_parse` locate, check and parse RDF files
  with the `owl` source's safety checks (pass `context.terms`);
  `rdf_terms` names RDF terms as both built-in RDF readers do. The
  property-graph helpers of the built-ins (`property_graph.FileReader`,
  `GraphBuilder`: sized reads, line caps, refusals naming the file and
  line) are for kinds added inside `adapters/seed_sources/`. A reader of
  RDF from elsewhere (a SPARQL endpoint's `CONSTRUCT` result, say) hands
  its triples as `abox_mapping.RdfTerm` tuples to
  `abox_mapping.map_abox(triples, options, schema, terms=context.terms)`
  (pure, no rdflib) and gets the `rdf` kind's rules: instances, names and
  descriptions by language, the most specific mapped class, CURIE
  property keys, nothing inferred. Its options model subclasses
  `AboxMappingOptions`.

  A service loader fetches a snapshot once per run:

  ```python
  import json

  from pydantic import Field

  from unified_kg_rag import (
      SeedEdge,
      SeedGraph,
      SeedNode,
      SeedSourceContext,
      SeedSourceOptions,
      register_seed_source,
  )


  class ServiceOptions(SeedSourceOptions):
      graph: str = Field(description="The service's graph to read")


  @register_seed_source("ontology_service")
  def load_ontology_service(context: SeedSourceContext) -> SeedGraph:
      options = context.options(ServiceOptions)
      source = context.source
      token = None
      if source.secret_arn:
          secret = context.client("secretsmanager").get_secret_value(
              SecretId=source.secret_arn
          )
          token = json.loads(secret["SecretString"])["token"]
      try:
          # fetch_instances: your client of the service (HTTP, SPARQL, or an
          # MCP session calling the server's tool).
          snapshot = fetch_instances(source.path, graph=options.graph, token=token)
      except OSError as e:
          raise context.terms(f"cannot read '{context.location}': {e}") from e
      return SeedGraph(
          nodes=tuple(
              SeedNode(id=n["iri"], iri=n["iri"], name=n["label"], type_iri=n["class"])
              for n in snapshot["instances"]
          ),
          edges=tuple(
              SeedEdge(source=e["from"], target=e["to"], predicate_iri=e["property"])
              for e in snapshot["links"]
          ),
      )
  ```

  selected with `seed_graphs: [{name: catalog, kind: ontology_service,
  path: "https://ontology.example.com/v1", options: {graph: products},
  secret_arn: ...}]`, from Python after importing the module (the CLIs
  import only the built-in sources).

  **Copy, or federate.** Such a loader copies: the snapshot becomes part
  of the graph, embedded, clustered and keyed like the rest, and its IRIs
  line up with the service's through `Entity.iri`, `type_iri` and
  `predicate_iri`. Federation, querying the service at retrieval time
  instead, is not built. Its seam would be a retrieval-time port (look up
  the IRIs of names, a node's neighbours, a node's description) that
  expands stored entities by `iri` without copying; nodes it returns would
  be in neither Leiden nor the vectors, so global search would not see
  them, and the `source_kinds` vocabulary can name them. The registry,
  `Entity.iri` and `source_kinds` are the seams either way.
- **New config section**: Define a Pydantic `BaseModel` → attach it to its parent via `Field(default_factory=...)` → describe every field and regenerate `docs/reference/configuration.md`; add a key to `config-template.yaml` only when most deployments set it (see §12).


### Custom backends (run without AWS)

Every cross-service dependency is behind a port. Each port is a role in the
provider bundle whose adapter `providers.<role>.kind` selects through the
provider registry, and the orchestrators also accept any port by
**constructor injection**, so a custom backend needs no subclassing or
dispatch edit. The ports, their default adapters and the local ones (the
`[local]` extra):

| Port | Contract | Default adapter (local adapters) | Inject via |
|---|---|---|---|
| `LLMFactoryPort` / `EmbeddingFactoryPort` / `RerankFactoryPort` (`ports/model_factory.py`, `Protocol`) | `get_model()` / `get_model_info()` returning a LangChain-compatible model; optional LLM-factory hooks: `is_transient_error(exc)` (which errors the ingestion batches retry; default `adapters.llm.retry.looks_transient`: network errors, HTTP 408/429/5xx), `is_item_error(exc)` (which errors count against a document, `indexing.max_document_failures`; default `looks_like_item_error`: HTTP 400/413/422 other than a rejected model or setting, read from `status_code` or `response.status_code`) and `transient_retry` (a `TransientRetryConfig`: with `is_transient_error`, query chains are retried under it) | `BedrockLanguageModelFactory` / `BedrockEmbeddingModelFactory` / `BedrockRerankModelFactory` (`openai_compatible`, `ollama`; `cross_encoder`, `none`) | `providers=` (below) on `GraphRAGChain`, `DataIngestionPipeline`, `EvaluationManager`; `GraphRAGChain(model_factory=...)` is an LLM-only bundle |
| `TokenCounterPort` (`ports/model_factory.py`, `Protocol`) | `count_tokens()` / `truncate_to_token_limit()` | `BedrockTokenCounter` (`LocalTokenCounter`: estimate or tiktoken; follows the LLM kind) | `Providers(token_counter_factory=...)` |
| `VectorIndexer` / `GraphIndexer` (`ports/indexer.py`, ABC) | `index_*` / `upsert_*` / `delete_by_id`; the incremental lookups (`GraphIndexer.find_ids_citing` / `find_relationship_ids_citing` / `find_entity_ids_by_name_key`, `VectorIndexer.find_documents` / `delete_extraction_records` / `retain_extraction_records` / `prune_text_unit_communities`) default to "cannot tell" (`None`: nothing is found through them, the merge reads every name) and to deleting nothing; the extraction-record store (`VectorIndexer.read_extraction_records` / `write_extraction_records` / `stored_extraction_record_keys`, by record key) defaults to keeping none (`{}`: every text unit is extracted, nothing is stored), and `stored_extraction_record_keys` to the keys `read_extraction_records` returns; `VectorIndexer.check_index_creation(suffix, rebuild=)` (called once per run before the first model call, for the indices the run will create: all of them on a `rebuild` (full build or reset), the missing ones otherwise; raises `StoreRequestRejectedError` for settings the store refuses) defaults to accepting them; `VectorIndexer.read_vector_copies` defaults to `{}` (every item an incremental run touches is written in full); `VectorIndexer.describe_indices` / `scan_items` (`run-describe`, §5.8) default to `None`, a store that cannot be described or exported; `GraphIndexer.read_entities` / `read_relationships` default to `[]`, so without overrides the delta merge overwrites the stored descriptions, lineage (`text_unit_ids`) and weights; `GraphIndexer.begin_rebuild` / `end_rebuild` (in-place full build, §5.2) default to `False` (the full build clears each label first); `initialize()` (no-op by default) / `clear()` return `None` and raise a typed store error (`StoreConnectionError`, `StoreConfigurationError`, ...) on failure: a returned `False` is ignored; an `in_process = True` attribute (on the indexer and on the registry) says the store is embedded in this process, and only when all three say so does a stopped run release its lease at once (absent: remote) | `OpenSearchIndexer` / `NeptuneIndexer` (`gremlin`: the same adapter on Gremlin Server; `sqlite`: `OpenSearchIndexer` on `SQLiteSearchEngine` / `SQLiteGraphIndexer`) | `Providers(vector_indexer_factory=..., graph_indexer_factory=...)` (a new instance per call; its caller closes it), `DataIngestionPipeline(..., vector_indexer=..., graph_indexer=...)` (borrowed: never closed by the pipeline) or `IndexingManager(vector_indexer=..., graph_indexer=...)` (closed by `IndexingManager.close()`; §8.2) |
| retriever (role-keyed builder) | `BaseGraphRAGRetriever.aretrieve`; a document retriever must honour `SearchQuery.exclude` (leave out what it matches before the page is cut: the filtered walk's edge pages rely on it); `aindex_limits(prefix, suffix)` (the hits one search returns, the values one `terms` clause takes) defaults to the configured `index_settings`, so a backend whose live limits can be smaller overrides it; the strategies size every list they send by it (fetches by id in batches of `IndexLimits.id_batch_size`, walk frontiers and exclusions by `max_terms_count`, at most `search.max_concurrency` batches at once), so `aretrieve` may cut a search to its limits and need not batch; `filter_fields()` declares the filter keys the retriever applies (`None`, the default, accepts every key) and `filter_fields_for(prefixes)` those of the indexes a strategy filters (`BaseSearchStrategy.filtered_index_prefixes`), by default `filter_fields()`; `GraphRAGChain` rejects a caller key none of a strategy's `filtered_retrievers` declares, checking a custom retriever against the union of its keys | `OpenSearchRetriever` / `NeptuneRetriever` (`sqlite`: `OpenSearchRetriever` on `SQLiteSearchEngine` / `SQLiteGraphRetriever`) | `Providers(retriever_builders=...)` or `GraphRAGChain(retriever_builders={RetrieverRole.GRAPH: lambda: MyGraphRetriever(...)})` |
| `GraphQueryPort` (`ports/graph_query.py`, `Protocol`, on the GRAPH retriever) | `adescribe(suffix, *, sample_vertices, edges_per_vertex, max_examples)` returning a `GraphProfile` (`domain.retrieval.graph_query.build_profile` of a sample) and `aexecute(request)` answering a `GraphQueryRequest`'s operation (`count`, `group`, `tuples`, `paths`) with the §6.6 semantics: the label derived from `request.suffix` by the retriever, the walk's vertices distinct, results over distinct `id` tuples, at most `limit + 1` rows, `truncated` past it, the `hydrate` properties of every vertex named; read-only, checked before it is sent, bounded by `timeout_seconds`; a store refusal or time limit raises `GraphQueryExecutionError`. Only the `structured` strategy calls it; without it that strategy's queries fail before any model call (`ConfigurationError`). Pass `tests/integration/test_graph_query_conformance.py` | `NeptuneRetriever` (Neptune, Gremlin Server), `SQLiteGraphRetriever` | the GRAPH retriever builder |
| `ContentGuardPort` (`ports/guardrail.py`, `Protocol`) | `async acheck(text, source, *, grounding=None)` returning a `GuardVerdict` (`none` / `masked` / `blocked`, the text to continue with, and `grounding`, a `GroundingCheck`, when a `GroundingContext` was given); raises when it cannot decide. `grounding` is passed only while `aws.bedrock.guardrail.contextual_grounding.enabled` is set | `BedrockGuardrail` (built when `aws.bedrock.guardrail.identifier` is set) | `GraphRAGChain(guard=...)` |
| `DocStatusPort` (`ports/doc_status.py`, `Protocol`) | `get` / `get_many` / `put` / `put_many` / `delete` / `list_namespace` / `diff` (which must take `extraction_fingerprint` and fill `DocumentDelta.quarantined_unchanged`), `record_fits`, the write lease (`try_acquire_lease` / `lease_holder` / `release_lease`) a scope's last extraction settings (`get_extraction_settings` / `put_extraction_settings`) and the schema annotations a namespace's stores hold (`get_annotation_digest` / `put_annotation_digest`), all called directly; a record's namespace is the `<namespace>#` prefix of its `doc_id` (`doc_id_namespace`), and `diff` with a scope reads only that namespace. A subclass inherits `get_many` / `put_many` / `record_fits` defaults (one record per call, no size limit) | `DynamoDBDocStatusStore` (`SQLiteDocStatusStore`) | `Providers(doc_status=...)` or `DataIngestionPipeline(..., doc_status=...)` (either turns incremental indexing on); pass `tests/integration/test_doc_status_conformance.py` |

**Errors from a custom backend** are classified by type, never by message
(§8.1): raise a `ConnectionError` or `ConfigurationError` subclass (the typed
store errors in `shared/exceptions.py`) to fail the query; any other
exception degrades only the section that raised it.

**Protocols, not base classes.** The model-factory and doc-status ports are
`runtime_checkable` `Protocol`s, so a custom class only needs the right method
shapes:

```python
from unified_kg_rag import GraphRAGChain

# Not exported from unified_kg_rag: an internal path that may change in any
# release (§2.5); pin the package version when you import it.
from unified_kg_rag.domain.models.config import TransientRetryConfig

class MyLLMFactory:                       # structurally an LLMFactoryPort
    def get_model(self, model_id, **kwargs): ...   # returns a LangChain chat model
    def get_model_info(self, model_id): ...        # returns a LanguageModelInfo
    # Optional hooks (defaults: adapters.llm.retry.looks_transient /
    # looks_like_item_error; no query-chain retry without transient_retry):
    def is_transient_error(self, exc): ...         # retry this error?
    def is_item_error(self, exc): ...              # the document's own error?
    transient_retry = TransientRetryConfig()       # query-chain retry policy

chain = GraphRAGChain(config=cfg, model_factory=MyLLMFactory())
```

A factory must keep the `**kwargs` catch-all and ignore keys it does not
understand: callers pass framework arguments such as
`model_purpose=ModelPurpose.INGESTION` (the per-path policy: response cache,
ingestion effort, query retry), and a strict signature raises `TypeError` on
the first one.

**One bundle, every component.** `Providers` (`adapters/providers.py`) holds
the boto3 session, the model and token-counter providers and the store and
registry builders; `application/bootstrap.py` (`build_providers`) is the
composition root. Each orchestrator builds or receives one bundle and passes
it to everything it constructs (the chain's prompts, strategies, reranker,
token managers and retrievers; the pipeline's stages and indexers; the
evaluation judges, which reuse the chain's bundle by default).

```python
providers = Providers(
    cfg,
    llm_factory=MyLLMFactory(),
    embedding_factory=MyEmbeddingFactory(),
    # Reranking is on by default (providers.rerank.kind: bedrock) and would
    # otherwise call Bedrock; pass a factory or set providers.rerank.kind: none.
    rerank_factory=MyRerankFactory(),
    token_counter_factory=lambda model_id, **_: MyTokenCounter(model_id),
)
chain = GraphRAGChain(config=cfg, providers=providers)
pipeline = DataIngestionPipeline(cfg, pipeline_config, providers=providers)
```

- Orchestrator arguments (`retriever_builders=`, `doc_status=`,
  `vector_indexer=` / `graph_indexer=`, `GraphRAGChain(model_factory=...)`)
  are shorthand for a bundle built with them; passing one together with
  `providers=` raises `ValueError`.
- Anything not supplied is built lazily by the configured kind's builder (the
  model factories at most once per bundle); a component built without a bundle
  builds a default one. There is deliberately no DI container: the bundle
  covers exactly what the framework consumes.
- Conversation memory belongs to the chain and keys each conversation by index
  suffix, `user_id` and `conversation_id`; pass one `MemoryManager(cfg)` as
  `memory_manager=` to share history across chains.

**Provider registry.** `adapters/provider_registry.py` maps
`(ProviderRole, kind)` to a builder that receives the bundle; a new kind needs
no edit to the bundle, the config model or any call site:

```python
from unified_kg_rag import ProviderRole, register_provider

@register_provider(ProviderRole.LLM, "my_server")
def _build(providers):
    return MyServerFactory(providers.config)
```

`providers.llm.kind: my_server` then selects it (a non-Bedrock LLM kind also
needs a token-counter builder of the same kind). A kind that connects to a
server declares its setting (`endpoint=("<config key>", "<env var>")`), and
one that imports an optional package declares it
(`requires=("<module>", "<how to install it>")`): the CLIs refuse an unset
endpoint or a missing package at start-up (exit 2). `backend: local` is a config
preset (`domain/models/provider_config.py`), not a code path: it fills in
`providers`, `aws.neptune`, `aws.opensearch` and the context budget only
where the config leaves them unset.

**Reference implementations.** The in-memory fakes of `tests/fixtures/fakes/`
implement the indexer ports and run the whole pipeline
(`tests/integration/test_ingestion_stages.py`): the starting point for a
custom store. The embedded SQLite stores show the two ways to add a backend:
the vector store (`adapters/local/search_engine/`) implements the OpenSearch
client calls (`SearchEngineClient`), so the unchanged OpenSearch indexer and
retriever run on it; Gremlin cannot be emulated that way, so the graph store
is its own `GraphIndexer` (`SQLiteGraphIndexer`) plus a
`GraphExpansionRetriever` subclass implementing only the traversal and,
for the PageRank of §6.4, `_edges_within` (optional: without it the
strategies read the edges from the relationships index).

### Deliberate Design Boundaries

Two boundaries are intentional:

- **`SearchQuery` carries adapter vocabulary (index and label prefixes).** A
  prefix names a logical collection each store pairing maps to its own storage
  (the embedded SQLite stores reuse them). The boundaries that matter, the
  indexer and model-factory ports, are abstracted and injected; a
  backend-neutral query model would be a large, risky churn that only a
  backend unable to map named collections would justify.
- **One physical OpenSearch index per index suffix and artifact type.** Fine
  for a handful of tenants; tens of thousands of suffixes multiply the shard
  count. The scale-out fix (one index per artifact type with an enforced
  tenant field and routing) changes the index, search and delete paths and is
  deferred. Metadata filters are relevance filters, not access control:
  isolation comes from separate suffixes
  ([runbook](../operations.md#query-access-and-tenant-isolation)).

---
