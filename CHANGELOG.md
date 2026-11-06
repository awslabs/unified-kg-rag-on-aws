# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first release after 0.2.3, the initial public snapshot (commit `81ea498`,
never tagged): GraphRAG and LightRAG retrieval on one Amazon Bedrock, Neptune,
OpenSearch and DynamoDB stack, now also runnable without an AWS account. Every
entry compares with 0.2.3. A fix to a feature added in this same cycle is folded
into that feature's entry, so *Fixed* lists only bugs 0.2.3 had. Nothing 0.2.3
stored is read again: follow *Upgrading* in order.

### Highlights
- Two retrieval methodologies on one stack: GraphRAG (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG
  (`mix`/`hybrid`/`naive`).
- Global search and DRIFT follow MS GraphRAG: a cut of the community hierarchy, and DRIFT's primer, follow-up and
  reduce loop (#318, #323, #347).
- Local mode: `backend: local` runs on Ollama or an OpenAI-compatible server with embedded SQLite stores (#209,
  #271-#273).
- Incremental indexing that converges with a full build and survives interruption, with a namespace lease,
  quarantine and an online reset (#175, #186-#199, #325, #327, #331, #346).
- Re-runs pay only for unfinished work: an LLM response cache, a per-text-unit extraction store, and reused
  summaries, reports and vectors (#322, #326, #527, #537).
- One `models`, one `providers` and one `retry` section; unknown and moved keys fail the load (#334, #335, #468).
- The Bedrock guardrail screens the query and the answer with `ApplyGuardrail`, with an opt-in contextual grounding
  check (#319, #462).
- Document attribute filters apply to every index and to graph expansion (#354, #360, #370, #381).
- An optional extraction schema, inline, from a file, an OWL/RDFS ontology or S3, applied as `align` or `strict`
  (#475-#496).
- Seed graphs: an existing knowledge graph (JSONL, CSV, RDF) merged into the extracted one as virtual documents
  (#497-#515).
- A query deadline and always-on LLM usage accounting (#522, #526).
- Evaluation: LLM-free evaluators, paired strategy comparison with cost and bootstrap intervals, a pairwise judge
  and a MuSiQue preparer (#121, #324, #526).
- OpenTelemetry tracing over OTLP, Langfuse or Arize Phoenix (#207, #208).
- Claude Sonnet 5.5 and Haiku 5.5 are the default tiers (#117, #170).
- The CDK stack gains a read-only query role, alarms and a one-task Step Functions ingestion (#157, #211, #319,
  #328).
- A core install is about a third of the size: evaluation, visualization, local mode and others are extras (#467,
  #576).

### Breaking changes

#### Install and extras
- Python 3.11 or 3.12 is required (`>=3.11, <3.13`); 3.10 is dropped (#430).
- A core install (`uv sync`, `pip install -e .`) is ingestion and queries: 145 to 83 packages, 937 to 335 MB on
  Linux. Extras: `eval` (ragas, datasets, pandas, langchain, langchain-community, langchain-openai), `viz` (pyvis,
  for the `interactive` renderer), `local`, `otel`, `unstructured` (Markdown and HTML), `ontology` (rdflib). A
  feature whose extra is missing fails naming it: the CLIs exit 2 before any work, the library raises
  `ConfigurationError` (also under `processing.ignore_errors`). The LLM-free evaluators and the `static` renderer
  need no extra; the container image installs `eval` (#467, #472, #520, #576).
- No longer dependencies: `graspologic` (Leiden runs on `graspologic-native`), `bokeh` (with `contourpy`, `tornado`,
  `xyzservices`), `umap-learn`, `scikit-learn`, `nest-asyncio`; `langchain` comes only with `eval` (#225, #226,
  #465, #520, #576). Security floors follow the package that pulls the pinned one in: `pillow` in `eval`, `llvmlite`
  in `unstructured`, `langchain-openai` in `eval` and `local`.
- The `static` renderer writes its three charts (file names unchanged) as self-contained HTML with an inline SVG
  instead of Bokeh documents: no script, a tooltip on every bar, no pan or zoom, no embedded Bokeh JSON (#576).
  Ingestion renders nothing: see *Configuration*.

#### Configuration
- Unknown keys, in the file or as `UKR__` variables, fail the load; `--allow-unknown-keys` (every CLI but
  `prepare-benchmark`) or `allow_unknown_keys=True` (`get_config`, `ConfigLoader`) warns instead (#335). A moved key
  fails naming its replacement, a removed key loads with a WARNING and is ignored: the generated [Moved
  keys](docs/reference/configuration.md#moved-keys) and [Removed keys](docs/reference/configuration.md#removed-keys)
  tables list them (#334, #344, #467, #468, #470, #577). [`config-template.yaml`](config-template.yaml) is a short
  starting file; every key is in the [configuration reference](docs/reference/configuration.md) (#468).
- Models: the 0.2.3 per-stage `*_model_id` keys moved to one `models` section: the tiers `models.default` and
  `models.fast`, `models.roles.<role>` (a model id or a tier name), `models.embedding` (model id and dimension, for
  the indices, queries, RAGAS and the visualization) and `models.rerank`. `providers.<llm|embedding|rerank>.kind`
  picks each family's provider, and `--llm-model`, `--embedding-model`, `--embedding-dimension` set the new keys; a
  setting of a provider the config does not select is ignored with a WARNING; under `backend: local` another
  embedding model needs its dimension too (#334). See [models](docs/reference/models.md).
- `aws.dynamodb.enabled` moved to `providers.doc_status.enabled` (default `false`; `providers.doc_status.kind` picks
  DynamoDB or SQLite), the one switch for incremental indexing (#344). `graph.analysis.centrality` moved to
  `graph.visualization.centrality` (#335).
- Retries are one `retry` section whose knobs count total attempts: 0.2.3's `indexing.neptune.max_retries: N` is
  `retry.graph_store_max_attempts: N+1`, and `processing.max_retries` had no effect (each model call made 5
  attempts, the `retry.max_attempts` default). A Neptune retry backs off from `retry.base_delay_seconds` (2 s), as
  model calls do (#151, #216, #468, #470, #472).
- These 0.2.3 keys are gone and fail the load as unknown keys:
  - `search.context_building_model_id`, `custom_prompts.context_building_*`: no context-building call; the query
    rewrite reads the history (#212).
  - `search.drift_search.` `enable_query_refinement`, `enable_keyword_extraction`, `max_iterations`,
    `initial_top_k`, `summary_length`, `summary_char_limit`, `n_entities`, `convergence_threshold`,
    `improvement_threshold`, `enable_primer`, `primer_follow_ups`, `query_refinement_model_id`,
    `keyword_expansion_model_id`, `convergence_assessment_model_id`, and `custom_prompts.query_refinement_*` /
    `keyword_expansion_*`: DRIFT takes `k_followups` (default 10), `primer_folds` (5), `n_depth` (2),
    `local_context_tokens`, `max_answer_tokens` and `models.roles.drift_primer` / `drift_action`; a
    `custom_prompts.drift_primer_*` must emit a 0-100 integer `score` (#323, #347).
  - `search.global_search.max_communities`, `map_batch_size`, `max_text_units`: use `community_depth` (2),
    `max_reports_per_level` (100), `map_context_tokens` (12000); `text_unit_slots` is an integer, default `0`
    (#318). `graph_timeout_seconds` has no replacement (#161).
  - `indexing.cross_run_fuzzy_merge` is `processing.fuzzy_entity_resolution` (#214).
  - No replacement: `graph.community_detection.report_generation.include_statistics` / `include_key_entities`
    (#214); `processing.gleaning.quality_threshold`, `convergence_threshold`, `convergence_change_scale`,
    `min_improvement_threshold`, `quality_completeness_weight`, `initial_quality_entity_scale`,
    `initial_quality_relationship_scale` (gleaning stops per unit once it adds nothing new, #171);
    `indexing.neptune.min_entity_importance` (#155); `indexing.opensearch.hybrid_search_pipeline_name` (hybrid
    search fuses in the retriever, #306); `indexing.opensearch.terms_batch_size`, `max_total_clauses`,
    `reserved_clauses` (a list filter is one `terms` query; more values than `max_terms_count` raise
    `InvalidFilterError`, #258); `processing.chunking.content_type` (plain text, #289).
  - `processing.chunk_concurrency` and `processing.batch_size` are `processing.max_concurrency`, one pool of
    in-flight calls (#296).
- Removed settings that change behaviour where you had set them (each loads with a WARNING):
  `search.reranking.enabled: false` no longer turns reranking off (set `providers.rerank.kind: none`);
  `search.fusion.method: weighted` fuses by RRF (weighted fusion is gone); `fixing.enabled: false` no longer stops
  output fixing; `search.local_search.type_quota` is the constant `BaseSearchStrategy.TYPE_QUOTA` (the mix is
  `search.token_manager.type_budgets`); `cache.chunking` (an entry over 1,000 items or 50 MB is always chunked);
  `indexing.opensearch.persist_embedding_cache` (the S3 embedding cache is gone: an incremental run embeds only its
  delta) (#344, #468, #577).
- Visualization: ingestion only writes `visualization_data.json` (`graph.visualization.export_data`, default
  `true`); `run-visualization` renders it with a seeded spring layout, and the embedding layouts (UMAP, t-SNE, PCA)
  are gone. `graph.visualization.interactive` and `static` are typed sections: a misspelled setting is an unknown
  key (#344, #467). See [visualize](docs/guides/visualize.md).
- Environment variables: `UKR__<SECTION>__<KEY>` sets any key (double underscores between segments, any case, JSON
  for lists and mappings); the 0.2.3 names are no longer read (#335). `NEPTUNE_ENDPOINT` is
  `UKR__AWS__NEPTUNE__ENDPOINT`; `OPENSEARCH_ENDPOINT` / `_USERNAME` / `_PASSWORD` are
  `UKR__AWS__OPENSEARCH__ENDPOINT` / `__USERNAME` / `__PASSWORD`; `BEDROCK_REGION` is
  `UKR__AWS__BEDROCK__REGION_NAME`; `BEDROCK_GUARDRAIL_IDENTIFIER` is `UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER`;
  `S3_BUCKET_NAME` is `UKR__AWS__S3__BUCKET_NAME`; `LOG_LEVEL` / `LOG_FORMAT` / `LOG_TO_FILE` / `LOG_FILE_PATH` are
  `UKR__LOGGING__LEVEL` / `__LOG_FORMAT` / `__LOG_TO_FILE` / `__LOG_FILE_PATH`; the container's
  `GRAPHRAG_SOURCE_DIRECTORY` / `GRAPHRAG_PIPELINE_ID` are `UKR_SOURCE_DIRECTORY` / `UKR_PIPELINE_ID`. Names 0.2.3
  never read (only development builds after it did) are gone as well: `BEDROCK_GUARDRAIL_VERSION` is
  `UKR__AWS__BEDROCK__GUARDRAIL__VERSION`; `OPENSEARCH_NUMBER_OF_REPLICAS` is
  `UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`; `OPENSEARCH_KOREAN_ANALYZER` is
  `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`; `GRAPHRAG_DOC_STATUS_TABLE` (which the 0.2.3 CDK stack set
  but the library ignored) / `_CREATE_TABLE` are `UKR__AWS__DYNAMODB__TABLE_NAME` / `__CREATE_TABLE_IF_MISSING`;
  `GRAPHRAG_SOURCE_SCOPE` is `UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE`; `GRAPHRAG_OBSERVABILITY_ENABLED` /
  `_BACKEND` are `UKR__OBSERVABILITY__ENABLED` / `__BACKEND`; `GRAPHRAG_METRICS_ENVIRONMENT` is
  `UKR_METRICS_ENVIRONMENT`. The CDK stacks and the container entrypoint set the new names.
- `AWS_REGION` and `AWS_PROFILE` fill `aws.region_name` / `aws.profile_name` only when the file leaves them unset
  (`UKR__AWS__REGION_NAME` overrides the file); switching `--backend` or a provider flag keeps a value you set even
  when it equals the old preset's (#335).
- `indexing.opensearch.index_settings` refuses the `knn` keys and a setting spelled both with and without `index.`
  (set ef_search with `vector_search.ef_search`); the `"NONE"` alias of `aws.s3.encryption.encryption_type` is gone
  (use `BUCKET_DEFAULT`) (#216, #408).
- Fusion buckets: a simple search's is `vector_all` (an `opensearch_all` weight applies to it with a WARNING); a weight
  on `opensearch_community_reports`, or on 0.2.3's `opensearch_candidate_community_reports` and
  `opensearch_expanded_community_reports` (in its template), is dropped with a WARNING, as no strategy emits them
  (#584, #588). A `lightrag_linked_chunks` weight weighs only the chunks the matched entities cite; the
  relationships' chunks are their own `lightrag_relation_chunks` stream, 1.0 unless set (#521, #530).
- Defaults (set the 0.2.3 value to keep its behaviour; the measured ones are explained in [defaults and
  evidence](docs/concepts/defaults-and-evidence.md)):

| Setting | 0.2.3 | Now |
|---|---|---|
| `models.default` / `models.fast` | per-stage ids, Claude Sonnet 4.5 or Haiku 4.5 | Claude Sonnet 5.5 / Haiku 5.5 (#117, #170) |
| `models.roles.evaluation` (the `run-eval` judges; was `evaluation.evaluation_model_id`) | Claude Sonnet 4.5 | Claude Opus 5.5 on Bedrock, about twice Sonnet 5.5's price; `default` judges with the answer model (#533) |
| `aws.region_name` / `aws.bedrock.region_name` | `ap-northeast-2` / `us-west-2` | `us-west-2` / `null`, following `aws.region_name` (#148) |
| `aws.s3.encryption.encryption_type` | `AES256` | `BUCKET_DEFAULT` (#105) |
| `aws.dynamodb.table_name` | `unified-kg-rag-on-aws-doc-status` | `unified-kg-rag-on-aws-doc-status-v2` (#327) |
| `processing.chunking.min_chunk_size` / `max_chunk_size` / `fallback_chunk_size` | 5000 / 50000 / 50000 | 1000 / 8000 / 4800 (#125) |
| `processing.max_concurrency` | 20 | 40, 20 under `backend: local` (#310) |
| `processing.similarity_threshold` | 0.6 | 0.85; similarity matching is opt-in (`processing.fuzzy_entity_resolution`) (#214) |
| `indexing.neptune.property_max_length` | 1000 | 4000 (#132) |
| `indexing.cross_run_merge` | `false`: an incremental run overwrote the stored entities and relationships with its delta's | removed (loads with a WARNING): an incremental run always merges its delta into the stored graph (#344) |
| `indexing.max_deletion_share` | no guard | 0.5: an incremental run deleting more of its scope asks first (#518) |
| `indexing.opensearch.language_analyzers` | `en: english`, `ko: nori`, others `standard` | also `de`, `es`, `fr`, `it`, `pt`, `ru` stemmers and `ja`/`zh: cjk` bigrams (#574) |
| `graph.community_detection.auto_resolution` | `true` | `false` (#141) |
| `graph.visualization.outputs_directory` | `outputs/visualization` | `null`: `<cache directory>/<pipeline_id>/visualization` |
| `search.token_manager.max_context_tokens` | 200000 | 30000; 22000 under `backend: local` with a non-Bedrock LLM (#125, #224, #226) |
| `search.fusion.fusion_weights` | 10 entries of 1.0 | `{}`, every bucket 1.0 (#258) |
| `search.global_search.relevance_threshold` / `use_dynamic_selection` | 0.5 / `true` | 0.2 / `false` (#318) |
| `search.auto_routable_strategies` | fixed: `simple`, `local`, `global`, `drift` | `local`, `mix`, `global`, `iterative` (#347, #598) |
| `search.query_timeout_seconds` | no deadline | 180 s; `null` turns it off (#522) |
| `search.model_timeout_seconds` / `model_max_attempts` | the ingestion client's 300 s, 6 requests | 90 s / 2 for query-time calls (#522) |
| `search.auto_speculative_extraction` | extraction after the router | `true`: alongside the router call (#525) |
| `search.output_ceilings` | the model's maximum output | 4096 for routing, query extraction and relevance ratings (#525) |
| `search.answer_prompt_style` | every strategy concise | `{global: comprehensive}`, merged under your map (#533) |
| `search.global_search.evidence_tokens` | every cited report the budget holds | 4000; `null` removes the cap (#533) |
| `evaluation.enabled_evaluators` | `langchain`, `ragas` | `ragas`, `answer_match`, `retrieval`, `graph_aware`; `langchain` is opt-in (#121, #576) |
| `evaluation.max_context_tokens` | 8192 | `null`, the answer budget (#315) |
| `run-eval` warm-up | none | one unscored, untimed query per strategy and run, not configurable (#529, #577) |
| `logging.log_to_file` | `true` | `false`; the console carries every record (#520) |

#### CLI, exit codes and JSON output
See [exit codes and JSON output](docs/reference/exit-codes-and-json.md) and the
[CLI reference](docs/reference/cli.md).
- Renamed flags have no alias: `run-ingestion --target-directory` is `--output-directory`; `run-eval
  --eval-data-path` is `--data-path` and `--outputs-directory` `--output-directory`; `run-visualization
  --output-dir` is `--output-directory`; `run-prompt-tuning --output` is `--output-path` and `--source-dir` is
  `--source-directory`. An abbreviated flag is rejected (exit 2) (#333).
- `run-ingestion --repair-metadata` is removed (it repaired nothing): use `--verify-metadata` and `--force-rebuild`
  or a new `--pipeline-id` (#248).
- Exit codes: `2` for a usage or configuration error before any work (also one met while running), `1` for a failed
  run, `130` for Ctrl-C, `143` for SIGTERM (#333, #344).
- `--output-format json` prints one JSON document on stdout (a status document on failure, argparse errors included)
  and everything else on stderr; `run-ingestion` rejects it with `--metrics-sink cloudwatch` (#333, #359, #366,
  #420).
- `run-eval --search-strategy` takes a list (`["auto"]` by default); with several strategies or `--repeats` it
  overrides an item's `search_strategy`; `abstention_statistics.unanswerable` has the overall block's shape
  (`completed`, `abstained`, `answered`, `abstention_rate`) (#292, #324, #PR).
- The `indexing_results` stage metrics in `pipeline_metadata.json` (and the index types an `Indexing failed` error
  names) start `vector_` / `graph_` instead of `opensearch_` / `neptune_` (`vector_text_units`, `graph_entities`,
  ...); the EMF metrics are unchanged (#584).

#### Stored data and reindexing
- Nothing 0.2.3 stored is read: index mappings, ids, the registry format and the stage caches all changed. An index
  without recorded embedding settings (`_meta`) is refused, a 0.2.3 stage-cache entry is a miss, and a
  parsed-document JSON's `content.text` is not read (write `page_content`) (#289, #327, #331, #332, #344).
- The doc-status registry is a new table (partition key `namespace`, sort key `doc_id`); a table of the 0.2.3 key
  schema is rejected at first use (#327).
- Entity resolution merges only names equal after normalization, legal-form removal and dense-script spacing (fuzzy
  matching is opt-in). Legal forms (one table, `shared/utils/legal_forms.py`): trailing Inc, Corp, Co, Ltd, LLC,
  LLP, LP, PLC, Pty, Pte, GmbH, AG, SA, SAS, SARL, SRL, S.p.A., Ltda, BV, NV and KK in any dotting, and the Korean,
  Japanese and Chinese company forms leading or trailing ("(주)", "合同会社", "股份有限公司"), so "Acme GmbH" and "Acme" of one
  type are one entity; not after "&"/"and" nor after an article alone. Entity and relationship `rank` come from node
  degree, and a relationship's weight sums its strengths per text unit (#189, #214, #574).
- Entities have no `MemberOf` edges to communities; an entity lists its `community_ids` (#348).
- Hidden files and files under a hidden or `__pycache__` directory of the corpus are not parsed (#213).

#### Deployment (CDK)
See [`iac/README.md`](iac/README.md).
- `iac/` is a uv project (`cd iac && uv sync`; `requirements*.txt` are gone) and needs AWS CDK CLI 2.1146.1 or later
  (#251, #438).
- Ingestion is one Fargate task (state `Ingest`, result under `$.task`; was four phases and `$.last_phase`), and its
  task definition lives in `GraphRagOrchestration`: a deployed stack upgrades in two deploys (*Upgrading*) (#328,
  #558).
- Queries get their own task definition and read-only role (`QueryTaskDefinitionArn` / `QueryTaskRoleArn` of
  `GraphRagCompute`); keep `run-eval` datasets and results under `s3://<cache bucket>/eval/` (#319).
- The registry table is `graphrag-doc-status-v2` (`<env>-graphrag-doc-status-v2`); the old one is retained outside
  `dev` (#327).
- EMF metrics carry an `Environment` dimension (`UKR_METRICS_ENVIRONMENT`) that the alarms and dashboard read:
  deploy the image and the stack together (#330).
- `create_guardrail` decides whether the stack owns a guardrail (`-c create_guardrail=false` to bring your own)
  (#103).
- In `network_mode=private` (the default) the VPC has no public subnets, internet gateway or `0.0.0.0/0` routes, and
  the OpenSearch domain publishes only its application logs (both in place) (#251). The STS interface endpoint is
  added only with `-c sts_endpoint=true`, needed only when the image config sets `aws.bedrock.assumed_role_arn`
  (#566).
- On the next deploy: the Step Functions role's trust policy gains `aws:SourceAccount` / `aws:SourceArn` (#438); the
  `owner` tag is applied only when `-c owner=...` is set; `GraphRagSecurity` is synthesized only with `use_cmk`; the
  OpenSearch domain requires `Policy-Min-TLS-1-2-PFS-2023-10` (TLS 1.2+ with ECDHE) (#566); a `removal_destroy`
  environment stops writing cache-bucket access logs and deletes `CacheAccessLogs` with its logs (#583).

#### Library API
- 0.2.3 exported only `__version__`. The public API is what `unified_kg_rag` exports (`unified_kg_rag.__all__`):
  `Config` and `get_config`, `DataIngestionPipeline`, `GraphRAGChain` with `RAGInput` / `RAGOutput` /
  `SearchOutput`, `MemoryManager`, `EvaluationManager`, the errors they raise, and the extension surface (ports,
  `Providers`, the provider, strategy, parser, renderer, schema-source and seed-source registries). Import from the
  package root; every other module path is internal and many moved, so they have no entry here (#164, #210, #216,
  #312, #333, #365, #456, #464, #584). The `unified_kg_rag.visualization` package is gone (#467). See the [Python
  library guide](docs/guides/python-library.md).
- `GraphRAGChain`: retrieval-only queries are `search()` / `asearch()`, returning a `SearchOutput`; `ChainMode` and
  `mode=` are gone (`GraphRAGChain(cfg).asearch(q)`, read `out.search_results.results`) (#333). It rejects unknown
  keyword arguments, and only `Providers` / `build_providers` take `boto_session` (`providers=Providers(config,
  boto_session=...)`, for `DataIngestionPipeline` too) (#218, #577). `DataIngestionPipeline` takes every argument
  after `target_directory` by keyword, so a session passed fifth fails instead of binding to `metrics_sink` (#589).
- One injection path: `doc_status=`, `vector_indexer=`, `graph_indexer=`, `model_factory=` and `retriever_builders=`
  cannot be combined with `providers=`; a subclass that overrode `_build_graph_retriever` /
  `_build_document_retriever` injects `retriever_builders` (#212, #216, #221).
- Conversation memory is per chain, keyed by index suffix, `user_id` and `conversation_id` (`ConversationKey`): pass
  one `MemoryManager` to share history. `get_memory_manager` is gone; `load_history` / `add_turn` (were
  `get_langchain_memory` / `add_message` / `load_context`) and `get_or_create_memory` take a `ConversationKey`
  (#212, #218, #333).
- `DocStatusPort`: a custom registry implements the whole port: `get_many`, `put_many`, `record_fits`,
  `list_namespace`, `get_extraction_settings` / `put_extraction_settings`, the lease (`try_acquire_lease`,
  `lease_holder`, `release_lease`), `get_annotation_digest` / `put_annotation_digest`, the `DocStatusRecord` fields
  (`text_unit_ids`, `extraction_fingerprint`, attempt and failure counts), `DocStatus.QUARANTINED` (with `PARSING` /
  `PROCESSING` gone), and a `diff` that takes the extraction fingerprint and fills `deferred` /
  `quarantined_unchanged`; `list_all` is gone (#214, #224, #327, #331, #332, #346, #488, #584).
- `GraphIndexer` / `VectorIndexer`: a custom backend implements `GraphIndexer.find_ids_citing` and
  `VectorIndexer.find_documents` / `delete_extraction_records`, or a removal cannot find what a removed document
  stated (a WARNING says so). Optional overrides: `find_entity_ids_by_name_key`, `count_neighbors`, `begin_rebuild`
  / `end_rebuild`, `check_index_creation`, `holds_any_item`, `holds_any_text_unit`. Full-build `index_*` methods
  take a keyword-only `suffixes`; `initialize()` / `clear()` return `None` and raise the store's typed error;
  `get_stats` and `get_entity_count` are gone (#163, #169, #214, #294, #327, #331, #384, #416, #490, #514).
- `BaseSearchStrategy`: `asearch(query, config=None)`; `_fuse_and_rerank` takes a required `rerank_trace=`; the sync
  `search` and `get_metrics()` (also on retrievers) are gone (#156, #249, #253, #284). A retriever or backend whose
  error must fail the query raises a `ConnectionError` or `ConfigurationError` subclass; errors are classified by
  type only (#261).
- Model factory ports: `get_model_info` returns a `LanguageModelInfo`; `is_transient_error(exc)` and
  `transient_retry` opt into the chain-level retry; `get_model` may receive `output_ceiling` and, on an embedding
  factory, `model_purpose` (#216, #522, #525).
- `Config` reads models through `model_id(ModelRole.X)`, `tier_model_id`, `model_tier`, `embedding_model_id`,
  `embedding_dimension` and `rerank_model_id`; `ModelRole.GLOBAL_REDUCE` is gone (#334, #577).
- `PipelineContext.unit_extractions` replaces `entities`, `relationships` and `rejected_entities` (#322).

#### Query behaviour
- The guardrail no longer wraps every model call: it screens the user query and the final answer through
  `ApplyGuardrail` (`aws.bedrock.guardrail.check_input` / `check_output`, both on); ingestion and evaluation models
  are never guarded. A blocked query gets the blocked message without retrieval (`RAGOutput.metadata.blocked`), a
  masked one continues with the masked text, interventions are in `metadata.guardrail`, a guardrail error fails the
  query, and with `check_output` a stream delivers the answer as one chunk after the check. A role outside the CDK
  stack needs `bedrock:ApplyGuardrail` (#104, #319, #344).
- A query with conversation history is rewritten as a standalone question (`models.roles.query_rewrite`) that every
  step reads (#212).
- A filter key no retriever of the strategy declares raises `InvalidFilterError` (#123); a suffix nothing was
  ingested under raises `IndexNotFoundError` (#178).

### Upgrading
From 0.2.3; the sub-list after it is for a `main` checkout since. The runbook's
[Upgrading the library](docs/operations.md#upgrading-the-library) links here.
1. Install on Python 3.11 or 3.12 with the extras you use (`eval`, `viz`, `local`, `otel`, `unstructured`,
   `ontology`). A CLI whose configuration needs a missing extra exits 2 at start-up naming it (#472).
2. Load your config once and fix what it reports: unknown keys (*Configuration* lists the 0.2.3 ones), moved keys
   (the load names the replacement) and removed keys (a WARNING names it). Rename environment variables to
   `UKR__...`.
3. Review the settings whose behaviour changed: set any 0.2.3 default you rely on (*Defaults*); replace
   `search.reranking.enabled: false` with `providers.rerank.kind: none` (otherwise a Bedrock Rerank call per query),
   and expect RRF ordering where you set `search.fusion.method: weighted` (#577).
4. A custom `fusion_weights.lightrag_linked_chunks` weight: set `lightrag_relation_chunks` too (usually the same
   value) to keep weighting every LightRAG graph chunk; mix and hybrid recall are sensitive to it (#521, #530).
   Rename an `opensearch_all` weight to `vector_all` and drop `opensearch_community_reports`,
   `opensearch_candidate_community_reports` and `opensearch_expanded_community_reports` (#584, #588).
5. A role with a custom IAM policy needs `dynamodb:BatchGetItem`, `BatchWriteItem`, `Query` and `DescribeTable` on
   the registry table (`UpdateTimeToLive` where the library creates it), `bedrock:ApplyGuardrail` on the guardrail,
   and `bedrock:CountTokens` with `InvokeModel`. With `bedrock_model_arns`, list the reranking model too: the Rerank
   API checks `bedrock:InvokeModel` on it (#191, #194, #298, #319, #327, #331, #566).
6. CDK: install AWS CDK CLI 2.1146.1 or later, run `cd iac && uv sync`, and deploy the stacks with the new image
   (they share the environment names, EMF dimension, query task and registry table). Add `-c sts_endpoint=true` when
   a private-mode image config sets `aws.bedrock.assumed_role_arn` (or the role assumption hangs), and `-c
   opensearch_nori_package=true` for Korean on the CDK domain (or keep its `cjk`) (#282, #459, #566).
7. A deployed stack: run `cdk deploy GraphRagOrchestration --exclusively` (`GraphRag<Env>Orchestration` outside
   `dev`) with your usual `-c` flags and the new `image_tag`, then `cdk deploy --all`; a plain `--all` stops at
   `GraphRagCompute` ("Cannot delete export ...") and rolls back unchanged. Container Insights series restart under
   the renamed ingestion task family (#558).
8. On that deploy: pass `-c owner=aws-proserve` to keep the `owner` tag; without `use_cmk`, delete a leftover
   `GraphRagSecurity` stack (CDK metadata only) with `aws cloudformation delete-stack`; dry-run the TLS policy
   change (`aws opensearch update-domain-config ... --dry-run`) to see whether it runs blue/green; in a
   `removal_destroy` environment copy the `CacheAccessLogs` logs you need, as the bucket is deleted (#566, #583).
9. Run queries with `QueryTaskDefinitionArn` and keep evaluation data under `s3://<cache bucket>/eval/` (#319).
10. Rebuild every index: one ingestion with `indexing.reset: true` and the registry on, in the new table, with
    `--force-rebuild` or a new `--pipeline-id` (never a 0.2.3 one). It costs a first build: every text unit is
    extracted (and gleaned) again and every vector embedded (#327, #331, #332).
11. Local mode: a `backend: local` config that relied on the preset for the stores gets the embedded SQLite stores;
    add `providers.graph.kind: gremlin` and `providers.vector.kind: opensearch` (as `docker/config.local.yaml` does)
    to keep the containers (#273).
12. After the rebuild succeeds: delete the old registry table, the OpenSearch pipeline 0.2.3 wrote (`DELETE
    _search/pipeline/graphrag-hybrid-search-pipeline`) and `s3://<bucket>/embedding-cache/` if you used it.
13. Scripts: update the renamed flags, exit codes and JSON output, `.States.PrepPhase` to `.States.Ingest` and
    `$.last_phase` to `$.task`, and `indexing_results` keys to `vector_*` / `graph_*` (#328, #333, #584).
14. Evaluation: list `langchain` in `evaluation.enabled_evaluators` to keep its scores (a `--fail-if langchain/...`
    gate fails without it; leaving it out saves two judge calls per query). `graph_aware` coverage matches by entity
    key and can read lower than before on the same answers: run baselines again before comparing. Re-render static
    charts to get the SVG pages (#563, #576).

From a `main` checkout (formats and keys no release had):
- `aws.bedrock.default_max_output_tokens` is `models.default_max_output_tokens` (the load fails until renamed;
  nothing re-runs). The pre-release `search.llm_retry`, `providers.llm.max_retries`,
  `providers.embedding.max_retries` and `evaluation.ragas_max_retries` are unknown keys: set `retry` (#216, #584).
- Filters: an index built before #360 (relationship and claim filters), #370/#376 (a removal clears its document's
  values), #446 (`attr_*` keywords) or #566 (numeric and boolean values as keywords on OpenSearch; a WARNING names
  each such field, and new fields map without a reset) needs one run with `indexing.reset: true`.
- Stored items: an index built before #508 (seed properties unmapped; such an entities index is refused, exit 2),
  #562 (relationships store their type, else `Relationship: A -> B`) or #383/#410 (removals written ahead) needs the
  same reset; for a deleted document restored mid-removal, deleting its registry record is enough.
- #574 analyzers: an OpenSearch namespace translating into `de`, `es`, `fr`, `it`, `ja`, `pt`, `ru` or `zh` stops
  before any model call (exit 2, `was built for other language settings`) until one `indexing.reset: true` run, or
  set `language_analyzers: {en: english, ko: nori}`.
- #574 legal forms: entity `name_key` follows the larger table; reset with a new or no `--pipeline-id` to merge
  designator variants, or entities stored apart stay apart.
- One reset covers all of these and the items below. On an index built with #565 or later it reuses the stored
  extraction records and vectors: beyond what the items below name, the model is asked only for the reports of the
  communities that cluster differently and a summary and embedding per newly merged entity over the threshold. On
  an index built before #565 it costs a first build (the #565 prompts below).
- Stored communities keep positional names (`Level 0 Community 3`) until the next reset, which renames them and asks
  every report once more; later resets reuse the answers (#539).
- Incremental runs may have stored a summary of a summary for an entity or relationship edited documents shared
  (#564, #589), also a run resumed after dying between its removal and its commit (#592): reset once to summarize
  from the lines (one call per such item over the threshold), or leave it.
- #565 changed the translation, description summary and gleaning prompts (gleaning re-keys every gleaned record,
  claims too): a reset translates, extracts and gleans every unit again, the cost of a first build; incremental runs
  use the new prompts for new and changed documents only. A reset also types untyped merges and re-cuts one-line
  `intelligent` pre-chunks.
- Extraction records stored under non-default description summary settings are found under their earlier key only
  when the run's summary settings and `models` tiers match those they were stored under: run once with them after
  upgrading, then change them (#541, #545, #556).
- An index built before vector digests embeds its items once more on its next incremental run (#527); a build record
  written earlier has no community counters until the next full build (#540); the first incremental run of each
  namespace reads its stores once to record their schema annotations, with no model call (#488).
- `processing.chunking.length_unit: chars` (the default) keeps the chunks and ids of an index built before the
  setting; `tokens` re-chunks and re-extracts the corpus (#460).
- `run-eval` results saved before usage accounting count in `answers_without_usage` / `answers_without_latency` and
  fail a `usage/*` or `latency/*` gate; results saved before #563 carry the earlier `graph_aware` scores, which
  `--from-results` and `--baseline-results` reuse: run them again to compare (#526, #529, #563).
- #577: `search.global_search.reduce_with_llm: true` no longer makes a reduce call and `evaluation.warm_up_query:
  false` no longer skips the warm-up (both warn); a run resumed across it finds no stage-cache entry
  (`fixing.enabled` keyed every stage), but the LLM response cache serves its model calls.
- (#596) A namespace ingested with `graph.community_detection.enabled: false` holds `lightrag` only: `local` and
  `simple` are now refused there (`MethodologyNotIndexedError`, `run-rag` exits 2) where they answered without the
  report section, and `auto` routes among `mix`/`hybrid`/`naive`; ingest with `graphrag` to use them. A build
  record written earlier names no artifacts: queries read the configured `indexing.methodologies` until the next
  run that writes the namespace records them, and that run rebuilds nothing. Default configurations run the same
  stages under the same stage-cache keys.
- (#598) `auto` may now route a multi-hop question to `iterative` (`search.auto_routable_strategies` adds it): about
  10 s more at the median and +$0.002 for the queries it takes, and a LightRAG-only namespace now makes a router
  call (between `mix` and `iterative`) where it made none. To keep the earlier routing, set
  `search.auto_routable_strategies: [local, mix, global]`; nothing is re-run.
- (#599) `raptor` can be selected (`indexing.methodologies`); without it nothing changes: no stage cache, extraction
  record or evaluation manifest is keyed by the new settings at their defaults, and the text units are stored as
  before. Adding `raptor` to a namespace builds its summary tree on the next run, or with `run-ingestion
  --enabled-stages summary_tree`, from the stored text units: no re-extraction, about one summary call per five or
  six text units on the default tier (the MuSiQue benchmark store's 3,318 text units: about 580 calls, $2-8 with
  Sonnet 5.5, $0.1-0.4 with `models.roles.summary_tree: fast`). A completed pipeline id re-run resumes at
  `indexing` as before.

### Added
- RAPTOR summary tree, opt-in and unmeasured (#599): the `raptor` methodology adds a `summary_tree` ingestion stage
  after indexing that clusters the stored text units by their vectors (numpy k-means by default, RAPTOR's Gaussian
  mixture with BIC as `algorithm: gmm`; PCA in place of UMAP), summarizes each cluster bottom-up in the index
  language (`models.roles.summary_tree`, `custom_prompts.tree_summary_*`) and swaps the rebuilt tree into its own
  index (`indexing.opensearch.summary_nodes_index_prefix`); a cluster whose children are unchanged reuses its
  summary. `raptor` alone runs no graph stage. The `tree` strategy (`auto` routes to it only on a `raptor`-only
  namespace) fuses BM25 + kNN searches of the text units and of every level of summaries, plus the summaries and
  passages citing the entities a raw-query search finds where the namespace has an entities index (buckets
  `tree_text_units`, `tree_summaries`, `tree_graph_summaries`, `tree_graph_text_units`), reranks them together,
  keeps `max_nodes_per_level` summaries of a level and expands each to `expand_children` of its passages; a summary
  is reported with its lineage, keeps a document filter only when all its documents share it, and goes, with every
  summary above it, when one of its text units is removed. Settings: `indexing.summary_tree`,
  `search.tree_search`, `search.token_manager.type_budgets.summary`; the `VectorIndexer` port gains the summary-tree
  methods (each with a "cannot" default). Differential tests against the original code ship with recorded fixtures.
  See [ingest](docs/guides/ingest.md#a-raptor-summary-tree-opt-in) and
  [Design §6.5](docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).
- Methodology profiles (#596): `indexing.methodologies` (default `[graphrag, lightrag]`, today's union) selects what
  ingestion builds, so a LightRAG-only corpus (`[lightrag]`) runs no community detection and makes no report call.
  The build record names the artifacts and methodologies a namespace holds; a query whose strategy reads another
  methodology fails before any model call naming what the namespace holds and how to add it, and `auto` routes only
  among the held ones. Adding a methodology to an incremental namespace rebuilds it once from its stored
  extractions (only the added model calls are new); removing one stops maintaining its artifacts without deleting
  them. Every strategy class declares `BaseSearchStrategy.methodology` (`iterative` reads its base's); `hipporag`
  holds what `lightrag` builds and `raptor` reads a summary tree (#599). See [methodology
  profiles](docs/concepts/overview.md#methodology-profiles) and [Design §4.6](docs/design.md#46-methodology-profiles).
- Local mode: the `providers` section and provider registry (`@register_provider`), Ollama and OpenAI-compatible
  models, a sentence-transformers cross-encoder reranker, embedded SQLite graph and vector stores and registry, the
  `local` extra and an Ollama service in `docker/compose.local.yaml` (#154, #209, #271-#273). See [local
  mode](docs/guides/local-mode.md).
- Observability: the `observability` section and `otel` extra, spans for runs, stages, batches and query steps,
  GenAI model-call spans, OTLP, Langfuse and Phoenix exporters with redacted content capture; CDK
  `enable_cloudwatch_tracing` with an ADOT sidecar (#207, #208, #211, #219, #269). See
  [observability](docs/guides/observability.md).
- Incremental indexing: the extraction-settings fingerprint (#224, #332); `indexing.max_document_failures` with
  own-error accounting and `documents_retrying` (#175, #431); source scopes and `--retire-source-scope` (#134,
  #192); the namespace write lease and quarantine (`indexing.max_document_attempts`) (#331, #346). See [incremental
  indexing](docs/guides/incremental.md) and [scopes and namespaces](docs/concepts/scopes-and-namespaces.md).
- Deletion guard: an incremental run that would delete more than `indexing.max_deletion_share` of a scope of 10 or
  more documents (seed shards and retired scopes not counted) stops at `document_loading` unless confirmed
  (`--confirm-deletions`, `UKR_CONFIRM_DELETIONS=1`, `--yes`, the prompt, or
  `DataIngestionPipeline(confirm_deletions=...)`) (#518, #531).
- Community drift: drift metrics (#320, #347), and per-suffix counters in the build record (`clustered_at`,
  `communities_at_clustering`, `communities_now`, `runs_since`) with a WARNING recommending `indexing.reset: true`
  past `indexing.community_compaction.max_runs_since_clustering` (20) or `max_community_growth` (1.5), and
  `community_growth_ratio` / `runs_since_clustering` metrics; nothing resets automatically (#540, #558).
- The extraction store (per-text-unit extraction records) and the LLM response cache (`cache.llm_responses`), so a
  resume or an edit asks only the model calls it lacks. Description summary settings key graph resolution, not
  extraction: changing them re-extracts nothing (#322, #326, #541, #545, #556).
- Extraction schema under `processing.graph_extraction`: `entity_types` items may be `{label, description, iri}`;
  new `relationship_types`, `relationship_patterns` (`"PERSON -WORKS_FOR-> ORGANIZATION"`), `schema_mode`
  (`off`/`align`/`strict`, default `align`) and `schema_source`, validated on load (labels unique by case and
  separators, patterns naming declared types). Declared types replace the prompt's example relationship types (the
  `{relationship_types}` variable a custom prompt may use) and normalise spellings; `strict` drops off-schema
  entities and relationships per unit (gleaning too), `align` keeps and counts them (`extraction_stats`,
  `gleaning_stats`). Only what the model sees keys extraction: the defaults, entity types alone or an IRI edit
  change no cache key or record (#475, #476, #481, #482). See [extraction schema](docs/guides/extraction-schema.md).
- Schema annotations: indexing writes `in_schema` and `type_iri` / `predicate_iri` on entities and relationships in
  every store, filter keys there (`filters: {in_schema: false}` returns what `align` kept). A filter on an
  annotation the schema writes on no item raises `InvalidFilterError`. An annotation-only schema edit is applied by
  the next incremental run without a model call, recorded once per namespace in the registry; a rewrite that stops
  is redone by the next run (#477, #478, #481, #482, #488, #490).
- Schema sources: `schema_source.kind: file` (YAML or JSON) and `@register_schema_source` (with
  `SchemaSourceContext`, `SchemaDocument`, `SchemaSourceError`; `requires=` declares an optional dependency the CLIs
  check, exit 2). A loader gets `path`, `options` and `secret_arn` (neither keyed), the config and an AWS session
  built only when asked for; logs show a URL without userinfo or query. Sources are keyed by content, not path; a
  local path may be a `file://` URL; reading the schema of a config whose source is not loaded raises
  `ConfigurationError` (#475, #480, #492, #520).
- Ontology schemas: the `ontology` extra (rdflib 7.6) and `schema_source.kind: owl` read an OWL or RDFS ontology
  (Turtle, RDF/XML, JSON-LD, N-Triples, N-Quads, TriG; a file or directory): named classes become entity types,
  object properties relationship types with patterns from an RDFS-style domain and range closure, labels from
  `skos:prefLabel` / `rdfs:label`. Nothing is fetched (`owl:imports` not followed, no external entities or remote
  JSON-LD contexts); quadratic-parse shapes are refused before parsing; collisions and caps fail instead of
  truncating (#491, #492, #495, #496). See [ontology](docs/guides/ontology.md) and the [mapping
  reference](docs/reference/ontology-mapping.md).
- Schema and seed sources in S3: a `path` may be an `s3://bucket/key` object or, for `owl` and seeds, a prefix read
  as a directory; sizes are checked before any byte is fetched, and a missing, denied (`s3:GetObject` /
  `s3:ListBucket`), empty or oversized source fails with what to fix. Moving between local and S3 re-extracts
  nothing; `run-rag` loads the schema on the first query with a schema filter key (#493, #495, #579).
- `run-prompt-tuning --schema-out PATH` drafts a schema file: the profile's entity types, the relationship types the
  tuning examples state most (with counts), patterns seen at least twice per endpoint-type pair, and every
  configured type with its description and IRI; a draft over an `owl` source keeps that source's patterns and names
  the types it adds. It never overwrites the config, its schema file, or (without `--force`) any existing file
  (#478, #480, #483, #490, #492, #496).
- Seed graphs under `processing.seed_graphs`: a registered `kind`, a local or `s3://` `path`, `filters`,
  `on_name_collision` (`error`/`merge`/`qualify`), `description_properties`, `confidence`, `shards`, and caps that
  fail instead of truncating (100,000 nodes, 500,000 edges, 256 MiB, 32 MiB for `rdf`). `run-ingestion` loads each
  seed at start-up against the schema (exit 2 naming the file and line) and imports its shards as documents of the
  run's scope: the registry diffs them, an edit re-imports only its shards, and no extraction call reads them
  (#498-#500, #502, #503, #509, #510, #513, #514). See [seed graphs](docs/guides/seed-graphs.md).
- Seed merges are document-first: a seed fills a type, name or confidence only where no document states one, adds
  description lines after the documents', stores properties unindexed under `attributes.seed_properties`, and adds
  no degree to ranks; a community only seeds evidence gets no report (`communities_seed_only`), and an incremental
  run reports a seed's communities as a full build does (#502, #508, #515).
- Seed sources `jsonl`, `csv` and `rdf` (an A-Box, with the `ontology` extra), which refuse duplicate ids, dangling
  edges (unless `options.dangling: skip`), non-scalar properties and over-long lines naming the file and line; an
  `rdf` CURIE prefix bound to another namespace fails until `options.prefixes` settles it (#497, #500, #504, #510,
  #579). See [seed graph formats](docs/reference/seed-graph-formats.md).
- `Entity.iri` and `source_kinds` (`document`, `seed_graph`) on entities and relationships, in every store and
  filter keys. A seed's text units are left out of text search, so seed facts reach answers as entities and
  relationships with `provenance`; under a `source_kinds` filter that keeps one kind, `local`, `drift` and `simple`
  read no community report, `global` and `naive` refuse the key, and text chunks are never filtered by it (#499,
  #501, #505, #510, #534, #536, #562).
- A reset that would drop another scope's seed documents, or from seeds alone another scope's documents, asks first
  (`--confirm-reset`, `UKR_CONFIRM_RESET=1`, `--yes`, `DataIngestionPipeline(confirm=)`); a no stops with exit 1 and
  `ConfirmationDeclinedError`. With the registry off, seeds alone replace a non-empty index only when confirmed
  (#503, #509, #513, #514). The seed API is exported: `register_seed_source`, `SeedSourceContext`,
  `SeedSourceOptions`, `SeedGraph`, `SeedNode`, `SeedEdge`, `SeedGraphError` (#498, #500, #510, #513).
- Guardrail: the opt-in contextual grounding check `aws.bedrock.guardrail.contextual_grounding` (CDK `-c
  guardrail_contextual_grounding=true`; `GroundingCheck.blocked`); with `check_output`, model-written search results
  (DRIFT's intermediate answers, global key points) are screened too; interventions name word lists
  (`wordPolicy:customWords`, `wordPolicy:PROFANITY`) (#462, #471, #567, #583). See
  [guardrail](docs/guides/guardrail.md).
- Per-query rerank status (`metadata.rerank`) and opt-in `search.reranking.required` (#284).
- Query deadline: `search.query_timeout_seconds` bounds every query and stream (which must send a first chunk within
  `search.model_timeout_seconds`) and raises `QueryTimeoutError` (`run-rag` exits 1, `run-eval` counts a failure);
  query-time model calls get their own clients (`search.model_timeout_seconds`, `model_max_attempts`; `backend:
  local` presets 300 s and a 900 s deadline), and Bedrock CountTokens uses them. No retry, OpenSearch request or
  global map call starts past the deadline; global map calls are cancelled with the query (#522, #530, #567, #579).
- Query-path options: `search.answer_prompt_style` (the comprehensive answer prompt,
  `custom_prompts.comprehensive_answer_*`), `search.global_search.evidence_tokens` (caps the cited report text after
  the key points; `0` sends key points only), `search.lightrag_search.kg_chunk_pick_method: vector` (upstream's
  `VECTOR` chunk pick), and `search.local_search.report_selection: membership` (MS GraphRAG local's reports of the
  expanded entities' communities, falling back to similar reports) (#521, #525, #530, #533, #535, #541). `drift`
  stays concise: with `{drift: comprehensive}` raise `search.query_timeout_seconds` (300 s). See
  [query](docs/guides/query.md).
- `iterative` search strategy (no default selects it; `auto` may route multi-hop questions to it, see *Changed*):
  `search.iterative_search.base` (`mix` or `local`) runs first, then a fast-model controller
  (`models.roles.query_controller`) judges the packed context and, when it does not answer the question, names the
  bridge entities and follow-up queries that seed up to `max_rounds` (2) more base searches and text-unit searches
  under the same filters; the rounds are fused (buckets `iterative_round_<n>`, `iterative_round_<n>_text`) and
  reranked, and the answer is generated once. A `mix` round that names no new bridge entity runs its text searches
  alone, and a decomposition step without one keeps the question's keywords, where `mix` reran with no keyword
  (falling back to the search text, or no entity search past `raw_query_fallback_max_len`) (#598). On 200 MuSiQue
  questions over `mix`: `answer_contains` +12.5 points, token F1 +5.2, p95 11.1 s to 27.4 s
  ([evidence](docs/concepts/defaults-and-evidence.md#iterative-follow-up-rounds)).
  `decompose: true` splits the question into steps instead (KAG-style). `metadata["context_insufficient"]` reports a
  final insufficient verdict, `metadata["iterative"]` the rounds and their model usage; new
  `custom_prompts.iterative_controller_*`, `query_decomposition_*` and `decomposition_step_*` (#594). See
  [query](docs/guides/query.md#follow-up-rounds-for-multi-hop-questions).
- LLM usage accounting, tracing or not: calls, input and output tokens, prompt-cache reads and writes, resent and
  failed requests and cache hits per model role and id, in `RAGOutput` / `SearchOutput.metadata["usage"]`,
  `EvaluationResult.usage`, `run-ingestion`'s JSON `token_usage` and per-stage EMF metrics on the CDK dashboard. A
  call counts from its start, so cancelled and background calls are billed at an estimated input; a response that
  reports no tokens keeps its resent requests and an output estimated from its text (#526, #530, #532, #533, #541,
  #563, #589).
- Corpus coverage: `run-ingestion`'s JSON `units_failed` and EMF metrics `total_units_failed`, `total_files_failed`,
  `total_files_skipped`, `total_documents_excluded` / `_deleted` / `_failed` / `_retrying` and
  `total_documents_without_claims`; a run with failed units ends with a WARNING naming them per stage (#518, #532).
- Evaluation: `retrieval` (hit@k, recall@k, MRR) and `answer_match` (exact match, token F1) evaluators, grouped
  statistics, the run manifest and index build record, `s3://` dataset and output paths, guardrail block counts
  (#113, #121, #206, #315, #321, #345, #355, #579). See [evaluate](docs/guides/evaluate.md) and [evaluation
  output](docs/reference/evaluation-output.md).
- Answer scoring: `evaluation.answer_span` (`first_line` default, `full`, or `extracted` by a fast model, cached,
  falling back to the first line), `evaluation.answer_normalization` (`official_squad`, `hotpotqa`), recorded in the
  manifest's `answer_scoring` (#324, #516).
- Strategy comparison: `run-eval --search-strategy a,b --repeats N` with bootstrap intervals, `deltas_by_category`,
  a `--pairwise-judge` (LightRAG's verdicts with `evaluation.pairwise_overall`, `position_consistency`, a win-rate
  interval, `--judge-model`) and `prepare-benchmark musique` (#324, #516, #536). See
  [benchmarks](docs/guides/benchmarks.md).
- Unanswerable questions and declines (opt-in; defaults unchanged): `prepare-benchmark musique --unanswerable N` adds
  MuSiQue-Full unanswerable questions (`metadata.answerable: false`, the full dev split downloaded at a pinned
  revision, SHA-256 checked), leaving out any whose answerable twin is sampled or whose removed supporting paragraph's
  exact text is in the pooled corpus (34-50% of candidates on the dev split), and reports the counts; `--hops 3,4` or
  `--hops 2=50,3=50,4=100` filters or sets per-hop quotas. `search.abstention.mode: mark` asks the answer model to
  start a declined answer with the fixed token `[NO_ANSWER]` (detected in any answer language, stripped by default)
  and reports it in `metadata.abstained` / `abstention_reason`, with its sources kept;
  `search.abstention.on_insufficient` (`hedge`, `abstain`) applies when a strategy sets its metadata
  `context_sufficient` to false (`iterative` does). `answer_match` scores a decline on an answerable question (the
  no-context reply included) 0 without a span call, and `abstention_statistics` counts `completed` = `abstained` +
  `answered` (not declined, as in coverage) and adds decline precision, recall and F1 and per `answer_match` metric
  coverage, accuracy, selective and overall accuracy. Each saved result keeps `abstention_reason`,
  `context_sufficient`, `insufficiency_reason` and the `iterative` run record. Measured on 100 + 100 MuSiQue
  questions, `mark` stays opt-in and `hedge` the default (#593, #PR). See
  [evaluate](docs/guides/evaluate.md#unanswerable-questions-and-declines).
- Evaluation of open-ended answers, all opt-in (defaults unchanged until a measured experiment decides): the
  pairwise judge reports win rates per answer-length bucket in estimated tokens and their correlation with length;
  `evaluation.pairwise_mode: differential` (`run-eval --pairwise-mode differential`, model role
  `models.roles.pairwise_overlap`) judges only the statements unique to each answer; the `assertion` evaluator
  (`assertion_recall`, `assertion_contradiction_rate`, `negative_assertion_violation_rate`;
  `evaluation.assertion_trials`) scores a dataset's `assertions`; `prepare-benchmark assertions` generates questions
  in five classes and source-cited assertions from your corpus, after Microsoft BenchmarkQED (MIT; no BenchmarkQED
  data used); `run-eval --closed-book` adds an answer-model-only arm without retrieval (#595). See
  [evaluation](docs/guides/evaluate.md#assertions-length-control-and-a-closed-book-bound).
- Comparison cost: per strategy calls, tokens and latency per query (mean, p50, p95) with paired deltas, a USD
  estimate from an optional `evaluation.prices` table (cache read and write rates too), and per-query CSV and
  Markdown; a failed answer counts with what it spent (#526, #529, #533).
- `run-eval --from-results` rebuilds or finishes a comparison from saved runs without answer calls and
  `--baseline-results` compares with an earlier run; `--fail-if <measure>:<stat><op><value>` gates exit 1 and fails
  on fewer than 10 paired queries, on metrics an evaluator could not compute (`failed_metrics`), or on a cost
  measure when the arms failed different numbers of answers. It warns when pooled runs differ in a query flag, the
  configuration, the commit, the index or the answer model (#526, #529, #541, #563).
- The `run-eval` judge defaults to Opus 5.5 on Bedrock (a judge judging its own answers shifted directness by
  -0.27): when the answer model, or the model that wrote re-judged answers, is Opus 5.5, an unset judge falls back
  to the default tier or Sonnet 5.5; a configured judge that is the answer model is kept with a warning. The CDK
  stack's default Bedrock grant covers it (#533, #536).
- Models: Claude Sonnet/Opus 5.5, Opus 4.6-4.8, Sonnet 4.6, Haiku 5.5 and OpenAI GPT models on Bedrock, free-form
  ids with `aws.bedrock.model_overrides`, reasoning effort per tier (#113, #117, #127, #128, #162, #179);
  `aws.bedrock.query_prompt_cache_ttl: 1h` (#458). See [models](docs/reference/models.md).
- `processing.chunking.length_unit: tokens`: chunk sizes in a script-aware token estimate (#460).
- `indexing.neptune.vertex_ids: custom` (`<label>:<item id>` vertex ids); graph expansion reads either mode during a
  switch, and a running query service follows it without a restart (#466, #470, #473, #474).
- `indexing.opensearch.vector_mode` / `compression_level`: on-disk quantized vectors on OpenSearch 2.17+ (#461).
- `run-prompt-tuning` reads every format ingestion parses; each `run-*` `--help` ends with an example (#121).
- Operability: run, stage and query ids in logs, extraction-failure metrics and alarms, start-up endpoint checks,
  and runbook sections on backup, restore, replicas and store upgrades (#122, #178, #211, #457, #519).
- IaC options: `neptune_serverless_ncu`, `ingestion_ephemeral_storage_gib`, `opensearch_master_instance`,
  `flow_log_retention_days`, `sfn_timeout_hours`, `opensearch_volume_gib`, `guardrail_version` and a published
  `GuardrailVersion`, `opensearch_nori_package`, Fargate Spot and ARM64 tasks, an `EnableVersionUpgrade` policy,
  `log_retention_days`, the stores' maintenance windows, and `observability_secret_arn` / `_keys` for Langfuse or
  OTLP credentials (same-region only in private mode) (#157, #211, #319, #328, #345, #457, #459, #519, #528, #541,
  #566).
- IaC alarms: `BedrockThrottling`, `IngestionTaskMemory`, OpenSearch snapshot failures, blocked writes, thread-pool
  rejections, a yellow multi-node cluster, an unusable KMS key, ephemeral storage, `DocumentsNotIndexed`, and
  `QueryLatencyHigh` on `run-rag --metrics-sink cloudwatch`'s `query_latency_ms` (#330, #519, #528).
- The package ships a `py.typed` marker (#568). `main` reports `0.3.0.dev0`, and a `v<version>` tag runs a release
  workflow that publishes a GitHub release (nothing on PyPI yet): the tag must be on `main`, and the notes'
  repo-relative links point at the tagged tree (#572, #588).
- HippoRAG 2 retrieval, opt-in (docs/design.md §6.4): the `ppr` search strategy (`search.ppr_search`; `auto` never
  routes to it, and `search.auto_routable_strategies` refuses it until a measurement shows where it wins) seeds a
  Personalized PageRank with the question's relationship hits (kNN, `fact_search_type`) and text-unit hits (BM25 +
  vector), runs it over the graph neighbourhood of the facts' entities and the passages they cite, and fuses the
  ranked passages, the passage hits and the facts (fusion buckets `ppr_passages`, `ppr_seed_text_units`,
  `ppr_facts`) with the optional rerank; no query-time LLM call besides the answer. It reads methodology `hipporag`
  (a `[hipporag]` or `[lightrag]` namespace serves it; one without the relationship vector index refuses it).
  `search.local_search.text_unit_ranking: ppr` (default `support`) orders local's (and DRIFT's follow-ups') text
  units by the same PageRank, seeded by local's entities, the question's passage hits and the endpoints of its
  `ppr_seed_relationships` (5) relationship hits: on 200 MuSiQue questions it put 11.0 points more supporting
  passages in the context (95% CI +8.4 to +13.8) and raised recall@5 after the rerank by 4.1 (+1.0 to +7.1), for
  about 0.4 s a query; answers are not measured yet. Graph retrievers gain
  `aneighbourhood_edges` (Neptune, Gremlin Server, SQLite; the relationships index under a document filter), and
  `SearchQuery.max_hops` overrides the expansion depth (#597).

### Changed
- Global search maps every report of one cut of the community hierarchy (packed by tokens, on the fast model), steps
  down to a shallower cut over `max_reports_per_level`, credits key points only to their reports, and dynamic
  selection descends to the leaves; it makes no separate reduce call: the answer model reads the ranked key points
  (0.2.3 reduced them on `map_reduce_model_id`, Claude Haiku 4.5, into a `synthesized_summary` result) (#318, #347,
  #354).
- DRIFT is MS GraphRAG's DRIFT (HyDE primer, follow-up local searches, reduce), with one rerank per query and at
  most about 27 LLM calls at the defaults; each primer fold's reports are its answer's evidence, and its metadata
  counts follow-ups by `report_selection` (#323, #347, #541, #567).
- `auto` routes with the fast model, falls back to the first routable strategy, and starts the extractions its
  strategies read alongside the router call (one round trip fewer; the unread one still runs and is billed); query
  entities and keywords are cached (`search.query_extraction_cache_size`, 1024; 0 turns it off), so `--repeats` and
  comparison arms pay the extraction once; short-answer calls ask for `search.output_ceilings` (and
  `evaluation.answer_span_output_ceiling`) tokens, never below the thinking budget plus 1024 (#125, #132, #212,
  #224, #347, #525, #530).
- `auto` may route a question that chains facts through an entity it does not name to `iterative` (over `mix`),
  about 10 s slower at the median for those; it starts the extractions a wrapper's base reads alongside the router.
  Leave `iterative` out of `search.auto_routable_strategies` to opt out (#598).
- `mix`/`hybrid` retrievals, DRIFT steps and keyword extraction run concurrently; Neptune expansion is opt-in for
  LightRAG (#125, #132, #212, #224, #347).
- Fusion reranks before MMR and the per-type cut; hybrid search normalizes lexical and kNN hits in the retriever
  over at least 50 hits each; unused type budget goes to the others; RRF scores are plain `weight / (rrf_k + rank)`
  sums; context packing ranks by score alone (#287, #306, #317).
- Answer context: entity and relationship lines carry no `[id]` (passages, claims and reports keep theirs) and
  entity lines show the type (#521).
- Communities: Leiden leaves are bounded by `graph.community_detection.max_cluster_size` (20), which also caps a
  parent's children; components below `min_community_size` get no community (#314, #320, #347). Names derive from
  the content id (`Level 0 Community 1a2b3c4d`) instead of the rank (#539).
- Claims resolve only against the entities of their own text units, and ties of names, types and statuses break
  deterministically, so builds do not depend on document order (#398, #402, #404, #442).
- `indexing.reset` rebuilds without taking the suffix offline (new indices and an alias swap) and a failed reset
  keeps the previous graph and records (#331).
- An incremental run finds what a changed or deleted document contributed by the text units every item cites, strips
  shared items and re-derives them (summarizing from lines as a full build does, also when several edited documents
  share an item or spell it otherwise, and when the run dies between the removal and the commit and is resumed: the
  removal notes the items it stripped in the extraction store until a commit's writes succeed), moves an item to the
  id a full build gives it, and deletes extraction records nothing names (#325, #327, #564, #589, #592). With fuzzy
  resolution on, it folds a delta entity into its one best stored match and never joins two stored ones until a reset
  (#564).
- Re-runs reuse what is unchanged: vectors carry a digest of their text and model, so indexing embeds only changed
  texts; the `intelligent` chunker's boundary answers and community report answers live in the extraction store; a
  full build keeps a stored summary of the same lines and settings; a no-op incremental run writes only its lease.
  `--force-rebuild` asks and embeds again (#527, #531, #537, #541).
- Stage cache keys cover the whole configuration and the package version (#332).
- `BatchProcessor` runs one pool of `processing.max_concurrency` items, connection pools follow the concurrency, and
  OpenSearch indexing embeds and writes 800 items of a type at a time (#296, #309, #310).
- The `graph_analysis` stage computes no centrality (#248); `run-visualization` computes it only for a renderer that
  reads it (`BaseRenderer.needs_centrality`) (#563).
- The parser uses its own loaders on langchain-core (same text and ids; a 4,000-page PDF loads in about 1 s);
  `ParserFactory.register_loader` takes any `BaseLoader` (#460, #471). Markdown and HTML need the `unstructured`
  extra; an image built with `UV_EXTRAS="--extra unstructured"` installs its spaCy model, and an offline run without
  it fails at once naming the install command (#534).
- Gleaning refines each text unit's own record and a correction changes only what that unit says (#322).
- `graph_aware`'s `retrieved_entity_coverage` and `retrieved_relationship_coverage` match by entity key (case, CJK
  spacing, quotes and legal forms ignored), not as a phrase inside a name, and count same-key expectations once
  (#563).
- `run-ingestion --s3-sync` takes the bucket from `aws.s3.bucket_name` when `--s3-bucket-name` is not given (#344).
- Settings set in code on `processing.chunking`, `indexing.neptune`, `retry` and `aws.bedrock.guardrail` are
  validated on assignment; a `chunk_overlap` not below `fallback_chunk_size` fails the load; `GraphRAGChain` refuses
  contextual grounding without `check_output` (#470, #472, #474).
- CDK: the ingestion runs one task per execution on the revision deployed with it and emits metrics per stage (#328,
  #330); queries read Neptune through the reader endpoint, so with `neptune_instances` above 1 a query right after
  an ingestion can miss its last writes for the replica lag (#566); synth warns when the guardrail stack's guardrail
  is not given to compute (#566).
- CDK: the cache bucket is versioned (noncurrent versions expire after 30 days, 1 under `pipeline-runs/`); Neptune
  keeps maintenance and backups in set windows (#519); `OpenSearchJvmPressure` follows AWS's recommendation,
  `NeptuneHighCpu` needs 3 periods, `NeptuneLowFreeableMemory` is 1/32 of the class memory, `PipelineFailures`
  treats no executions as no failure, and `cdk.json` pins strong cross-stack references (#298, #566).
- Log records of resume planning, run metrics, query processing, filter checks, the indexing collaborators and the
  Leiden clustering carry their own module's logger name (`unified_kg_rag.application.ingestion.resume`,
  `...ingestion.pipeline_metrics`, `...retrieval.query_processing`, `...retrieval.filter_validation`,
  `...storage.cross_run_merge`/`removal`/`reannotation`/`stored_items`,
  `unified_kg_rag.domain.ingestion.community_clustering`), not the pipeline's, chain's or indexing manager's; filter
  on the subpackage prefix (`unified_kg_rag.application.ingestion`, ...) (#586, #587, #590).
- Dependencies: pydantic 2.14, langchain-core 1.7.3, `langchain-aws>=1.8.1`, langchain-openai 1.7, langsmith 0.14,
  datasets 5.1, tiktoken 0.14, tenacity 9.2, boto3 1.43.122; aws-cdk-lib 2.275 in the CDK lock (#241, #251, #265,
  #438, #465, #489).
- Documentation restructured around a hub ([docs/index.md](docs/index.md)) with tutorials, concepts, guides and
  references, Korean twins of the main pages and checked links; the 0.2.3 user guide is replaced by those pages
  (#337-#342, #351, #469, #542-#557, #582, #583).
- CI: development tools in the `dev` group, parallel suites on every vCPU, coverage across jobs, a layering test,
  weekly property-test exploration, license checks, and an ASH security gate (#119, #257, #297, #303, #428, #456,
  #464, #465, #470, #512, #541, #568, #578).

### Deprecated
- `run-rag` / `run-eval --enable-thinking` logs a deprecation WARNING and will be removed: the default Claude 5.x
  models always reason, so tune their depth with `aws.bedrock.default_effort` / `fast_effort`. It still turns
  thinking on for models that opt in (#577).

### Fixed
Bugs present in 0.2.3.

Queries and retrieval.
- Local search looks each query entity up on its own and seeds graph expansion with the entity of its name, else its
  top hit when that scores at least `search.local_search.seed_score_ratio` (0.6) of the best hit, read at equal
  hybrid weights; one lookup of the joined names seeded a wrong neighbour beside every right seed (#517, #530).
- Graph expansion returns every seed and its neighbourhood ranked by distance (#161, #218), fetches at most its
  per-seed budget (#310), and no longer depends on the store's edge order (`indexing.neptune.deterministic_order:
  false` keeps it) (#470, #473); an expanded entity shows its type (#567).
- A number or boolean document filter value is a keyword on OpenSearch, so a string value of the same key no longer
  fails to index (`mapper_parsing_exception`) and ranges compare as text, as documented (#566).
- Local search's frequency filter keeps the query's own entities (#218), and local search keeps its seeds when
  expansion degrades (#261); its chunk filter check and global search's step-down ask the document retriever's own
  schema (`filter_fields_for`) (#567).
- Earlier turns' entities no longer leak into a follow-up, and `memory.max_conversation_age_hours` is enforced
  (#176, #218).
- Korean BM25 recall, CJK fuzziness and NFC normalization in retrieval (#175, #177); MMR diversity compares Han,
  Hangul and Kana words by character bigrams (#567).
- A query is translated, and its entities and keywords extracted in `target_language`, only when ingestion
  translates or the caller asks (`RAGInput.target_language`) (#567, #583).
- Store failures fail the query instead of answering from no context (unreachable, denied, rejected or overloaded
  stores, a failing embedding call) (#249, #258, #261, #266); queries no longer hang on a silent store
  (`search.store_timeout_seconds`, a Neptune traversal timeout) (#258, #264); a stalled Bedrock call no longer holds
  a query for about half an hour (#522).
- Neptune reconnects after a closed websocket, a lost connection no longer leaks threads or hangs a write, and
  concurrent queries share one connection attempt (#258, #261, #263, #266, #267).
- A short OpenSearch throttling burst no longer fails a query or a bulk write (#263, #264); a long question no
  longer fails its lexical search, and long list filters no longer multiply searches (#258, #260).
- The text-units section is no longer empty for a corpus translated into five or more languages (#264).
- The answer context stays within `max_tokens`, a small budget no longer yields an empty context, and CJK sections
  truncate at a character (#247, #258); an entity hit renders as `Entity` / `Type` / `Description` lines and items
  are separated by one blank line (#567).
- A relationship without a description reached the answer prompt as `Relationship: A -> B`: the relationships index
  stores the type (`RELATED_TO` when untyped; not a filter key) and the line reads `A -[TYPE]-> B` (#562).
- With MMR on, `local` and `drift` keep every reserved text-chunk slot (#247).
- `simple` no longer ranks hits of different indexes by raw score; local and LightRAG incident-edge expansion keep
  no arbitrary hub edges; LightRAG keeps every typed edge between two entities; `hybrid` without keywords warns
  (#295, #317).
- LightRAG `mix`/`hybrid` context includes the chunks the matched relationships cite (offline hybrid one-hop
  gold-chunk recall 0.30, now 0.86, 0.96 with rerank), and incremental and full builds pick the same chunks (#521,
  #530).
- Hybrid search on OpenSearch 2.13 keeps the `size`-th lexical hit, kNN on lucene uses `vector_search.ef_search`,
  and a filter on an attribute an index never mapped no longer fails vector and hybrid searches (#295, #306).
- `auto` no longer routes to `global` or `drift` without community reports, and falls back to `local` only when it
  is routable (#224, #317).
- Global search: a map batch over the model's context is left unrated instead of failing, a map step that fails
  entirely fails the query or is recorded in `degraded_error`, and a fatal model error fails the query even under
  `processing.ignore_errors` (#264, #267, #348, #354, #360, #567). A dynamic-selection rating is its answer's first
  number clamped to 1-10 (`8/10`, `Score: 2`; none counts in `global_unrated_report_count`), not a number of a
  mention of the scale (`On a scale of 1-10, I'd rate 8` is 8) (#567, #589).
- A suffix nothing was ingested under fails with `No indices found` (#227); indices of another embedding model or
  dimension fail with a `ConfigurationError`, and queries embed with `models.embedding.dimension` (#232).
- `run-rag` and `run-eval` query `processing.document_parsing.index_value` when `--suffix` is not given (they
  queried `default`), and the suffix line names `indexing.additional_suffix` (#534, #536).
- A `GraphRAGChain` reused across event loops no longer stalls, and LangChain `ainvoke` on a package retriever runs
  on the caller's loop (#230, #232, #243, #300).
- Query translation catches a refusal with a typographic apostrophe (#292); Cohere Embed v3 no longer embeds every
  text as an empty string (#463).
- The first calls of a batch no longer fail with "dictionary changed size during iteration", and a call that failed
  once no longer fails its item for good (#369, #378).

Ingestion and incremental indexing.
- A model answer that corrects itself is parsed by its last complete answer (each XML section keeps its last closed
  occurrence; `parse_llm_json` the last object that parses, preferring the caller's keys); a single cut-off answer
  is still flagged, and nameless entities in complete sections are retried as malformed (#532, #536, #541, #560).
- Output that parsed with the wrong shape (sections saying `None`, one wrapping element, prose, no section) goes to
  the output fixer instead of failing or being read as empty; a field holding markup is read as its text
  (`coerce_llm_text`) (#565).
- Incremental runs strip shared items instead of deleting them, repair interrupted runs, keep resets and deletions
  inside their namespace, merge a claim another document stored, keep one claim across spellings of an entity, and
  carry the stored entity's name on a new edge (#163, #185, #200, #214, #252, #290, #325).
- An unreadable doc-status registry fails the loading stage (`DocStatusRegistryError`) instead of rebuilding (#186);
  incremental indexing follows the extraction settings and stops before writing after a change of the embedding
  model, languages or analyzers, or onto a suffix built with the registry off (#224, #231, #233).
- Deleting a document under a non-default suffix strips it from shared claims, a deletion-only run checks the
  configured suffix, and a run whose new documents name no entity still commits them (#248, #252, #256).
- `indexing.reset` reusing a pipeline id rebuilds from the whole corpus, fails when the registry cannot be cleared,
  never keeps the old graph beside a partly written new one, and two rebuilds within a second no longer share an
  index (#221, #233, #299, #305, #346).
- Two corpora in one namespace holding the same file no longer share text units, and store sets with other index
  prefixes no longer share a registry namespace (#283, #327).
- Neptune no longer loses edges when interrupted, a vertex write replaces its stored properties, and OpenSearch no
  longer stores a description cut at the Neptune cap (#252, #294, #377).
- A store error while preparing or clearing fails the stage, a lost create or delete response no longer fails a
  build, a failed delete batch counts every id, bulk errors name their cause, and indexing memory no longer grows
  with the corpus (#252, #265, #294, #431, #432, #440).
- Community detection survives NaN, infinite, zero and negative strengths, keeps its coarser levels, merges a small
  community into its heaviest neighbour and is the same in every process (#250, #288, #304, #308, #343); a failed
  parent report no longer marks its documents `FAILED` (#248).
- Community report input is packed edge by edge as MS GraphRAG's `sort_context`, with claims listed (241 instead of
  126 relationships in the same budget on a synthetic grid) (#517).
- The XML parser keeps `&`, `<`, repeated siblings and unclosed sections, fails truncated responses and no longer
  stores `{}` for an empty element (#165, #174, #182, #213, #407, #452).
- Graph quality: gleaning no longer revives rejected entities or overwrites descriptions; claims no longer resolve
  onto another entity; CJK names with another head character no longer merge; communities no longer duplicate (#201,
  #203, #213, #214); gleaning corrections resolve within the refined unit (#220, #291, #304, #308, #322).
- Gleaning lists each entity with its type, gives the schema's type blocks and asks for `<confidence>` and
  `<strength>` on extraction's scales (gleaned items took 1.0 for both) (#565).
- Entity resolution takes a merged entity's type from the member types documents state most when the winning name
  has none (#565); merged items carry every document's filters, edges follow merged names (#250, #304).
- `entity_confidence_threshold` applies per chunk, and `entity_grounding.action: penalize` applies to gleaning
  additions (#250, #291).
- Descriptions: an over-long one is re-summarized map-reduce (#250); a summary over the threshold is not summarized
  again (#523); the translation and summary prompts name the language ("Korean", not `LanguageCode.KO`) (#565).
- Unresolved claims are not indexed, an entity named only in a relationship gets its suffix and filters, and
  deduplication prefers an indexed copy (#220).
- Chunking and translation errors are no longer swallowed, a chunk-boundary error falls back to a size split with a
  WARNING, and the `intelligent` chunker splits a one-line pre-chunk by size without a call (#250, #289, #565).
- `DataIngestionPipeline.run(source_directory)` ingests the directory it is given (#227).
- Resume and caching: a pipeline id resumes; switching a stage on runs it; a new suffix or scope recomputes; turning
  the registry off rebuilds; `--force-rebuild` writes its outputs; `--resume-from-stage` ignores case; generated ids
  no longer collide; a cached output that no longer validates is a miss (#223, #248, #259, #564).
- `--continue-on-error` records crashed documents `FAILED`, not `PROCESSED` (#223).
- `run-ingestion --s3-sync` keeps a stopped task's finished stages, its checkpoints no longer expire after a day,
  and a SIGTERM no longer hangs it; the CDK tasks run under an init process (#316, #326, #433).
- A retried Step Functions execution resumes at the failed stage (#223).
- The summary reports cache hits and restored stages, `run-ingestion` says why it exits 1, and a failed stage's
  error is logged once (#227, #248, #439).
- A file no loader reads (a missing extra, an unimported plugin) no longer reads as deleted: an incremental run
  keeps what it indexed, and a WARNING names the extra to install (#518).
- A failed read of stored entity names under fuzzy resolution fails the run instead of storing near-duplicates
  (#564).

Evaluation, visualization and tuning.
- Judge failures are recorded per query and failed scoring exits non-zero; a failed RAGAS run marks only its own
  metrics (#158, #210, #219).
- RAGAS skips `faithfulness` without context, counts each context once per batch, judges with its own metric
  objects, and stops when cancelled (#246, #366, #386, #393, #399).
- `run-eval` reports an invalid or missing dataset with exit 2 (#227, #230).
- `graph_aware` matches a word through its possessive and finds a Latin word or number inside Chinese or Japanese
  text (`AWS` in `答案是AWS公司`) (#516, #563).
- `run-visualization` rejects unknown `--renderers` and invalid data (exit 2), lays out only the drawn nodes, keeps
  physics off when `interactive` omits it, and exits 1 naming each failed renderer (`failed_renderers`) when another
  succeeds; the centrality chart no longer draws a missing value as 0.0, and log-scale histograms include the
  largest value (#246, #255, #292, #353, #563, #576).
- `run-prompt-tuning` profiles with the `graph_extraction` model, writes types in uppercase, logs no model output at
  ERROR and rejects `--max-docs` below 1 (#227, #292, #353).

CLIs, configuration and library.
- A config with a wrong type, invalid YAML or a missing file is a usage error, an empty config loads the defaults,
  and `custom_prompts` overrides are validated at load (#158, #178, #230); `ConfigurationError` replaces bare
  `ValueError`s (#297).
- Index suffixes and prefixes are validated at load and on the command line (#367, #372, #374).
- `run-rag --filters`: repeated flags add up; a pair without `:`, a blank, duplicate, unknown or `None` value is a
  usage error (#373, #408, #411, #443). `run-rag --interactive`: a line is a command only as a whole, `set-filter`
  keeps whitespace and refuses unknown keys (#353, #413, #416, #426, #428).
- `run-rag` documents conversation memory as in-process and warns on one `--query` with `--use-memory` (#224).
- `GraphIndexer.delete_by_id` without a suffix acts on the default suffix (#344).
- Used as a library, log records reach the host's handlers without raising on `structlog` fields, a host's root
  handler gets the plain message, and importing the package configures no logging and leaves `sys.path` alone (#178,
  #245, #251, #255, #528).
- `index_settings` keys reach the indices in either spelling (#260, #404).

Deployment (CDK).
- Alarms deliver notifications (the topic policy allows CloudWatch to publish), the corpus no longer expires with
  the cache prefixes, multi-node domains get replicas, timed-out runs alarm, and `cdk destroy` leaves no fixed-name
  resources (#157, #166, #211, #534).
- The failure notification renders any error and retries its publish; the Gremlin-errors alarm needs 3 minutes
  (#219, #254); dev and prod in one account no longer trip each other's alarms, and Neptune alarms watch the writer
  (#303, #330).
- A CDK deployment indexes Korean text (#282).
- Context values are validated at synth, a misspelled key names the closest one, `max_azs=1` and an odd
  `opensearch_count` over 2 AZs fail, and a zone-aware domain spans up to 3 AZs (#224, #227, #298, #345).
- The task role gets `bedrock:CountTokens`, so deployed stacks count tokens exactly (#298).
- A deploy that replaced the ingestion task definition (a new `image_tag`, guardrail, analyzer, size, tracing or
  secret) failed on the cross-stack export, or left the state machine on the old revision (#558).
- `cdk destroy` of a `removal_destroy` environment no longer fails on a late access log in `CacheAccessLogs` (#583).

### Security
- Pinned floors for advisories: pypdf 6.21.4 (seven high-severity denial-of-service advisories, e.g.
  GHSA-5849-834g-hvw5), filelock 4.2.1 (nine advisories, GHSA-v4x8-3wr5-3wq8 high; an override lifts
  `unstructured`'s cap; locked at 4.2.5), Pillow 13.0.0 (GHSA-p4wg-p73j-c528 and others), `unstructured>=0.24.1`
  (GHSA-4mvj-m6j5-pmf7, GHSA-8mgp-746c-j5xp, the stored XSS GHSA-v5mq-3xhg-98m9), `langchain-openai>=1.1.14`
  (GHSA-r7w7-9xr2-qq2r), `langchain>=1.3.9` in `eval`, and pyarrow 26.0.1 in `eval` (CVE-2026-92287; no Parquet is
  read, so it was not reachable); the lock, not a floor, takes certifi 2026.10.23 (GHSA-6q48-6c7w-9gw2) (#307, #426,
  #430, #455, #520, #580).
- Document attribute filters no longer leak excluded documents: they apply to relationships, claims, community
  reports (which carried the first entity's attributes) and graph expansion (which walked every edge), and a removed
  document's values stop matching what outlives it. The graph retrievers check filters as OpenSearch does (an
  unknown operator, a null or empty range raises `InvalidFilterError`), and a string value is converted to the
  field's type (`true`/`false`, `1`/`0`, `yes`/`no`; `in_schema:maybe` is refused) (#354, #356, #360, #370, #376,
  #381, #479, #483, #490, #581). See [filters](docs/reference/filters.md).
- A guardrail-blocked answer returns no sources or search results, and the query span records only the screened
  query (#345).
- IaC: queries run under a read-only role that cannot write through a `_search` path, lists only `eval/` and has no
  DynamoDB access; `bedrock:ApplyGuardrail` is granted on the configured guardrail only; unused Neptune, S3 delete
  and `GetInferenceProfile` actions are dropped, and private-mode endpoint policies admit only the account (#204,
  #211, #319, #345).
- Logs carry no query text, entity names or model output at INFO, no local variables in tracebacks, no ANSI codes
  when piped, and no relationship endpoints or output previews from gleaning; a unit test fails on a new INFO+ log
  call that passes them (#156, #173, #245, #255).
- CLI output escapes the data it prints, and `run-rag` strips terminal control characters (#173, #408).
- Prompts delimit corpus and retrieved text as data that cannot close its tags, the community report's claim lines
  included (#179, #182, #565); the LLM XML parser never resolves DOCTYPE entities (#578).
- Importing the package loads no `.env` file; the CLIs read only `./.env` and never override the environment (#224).
- ragas sends no usage telemetry unless `RAGAS_DO_NOT_TRACK=false` (#361).
- `pipeline_id` must be a single path segment; metadata filters are documented as relevance filters, not access
  control (#173).
- Interactive graph tooltips are escaped (#180).
- The compose containers drop all capabilities and run with `no-new-privileges`, the data-plane image declares
  `HEALTHCHECK NONE`, and retry jitter draws from `random.SystemRandom` (#307).

[Unreleased]: https://github.com/awslabs/unified-kg-rag-on-aws/compare/81ea4987...HEAD
