# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial public release preparation: an AWS-native, open-source reference
framework that unifies the GraphRAG and LightRAG retrieval methodologies on
Amazon Bedrock, Neptune, OpenSearch, and DynamoDB, with an optional local mode
that runs without an AWS account. This section lists the changes since 0.2.3,
the first public snapshot, by user impact. Each entry names its pull requests;
their descriptions and the git history hold the details, including fixes to
code that was added and fixed again before any release.

### Highlights
- Two selectable retrieval methodologies on one ingestion, indexing, caching
  and hybrid-search stack: GraphRAG community summaries
  (`auto`/`drift`/`global`/`local`/`simple`) and LightRAG dual-level keywords
  (`mix`/`hybrid`/`naive`), with triple-hybrid (lexical + semantic + graph)
  retrieval, RRF fusion, reranking and multilingual ingestion.
- Global search and DRIFT follow MS GraphRAG: global maps every report of one
  cut of the community hierarchy (with dynamic community selection), DRIFT
  runs the primer, follow-up and reduce loop (#318, #323, #347).
- Local mode: `backend: local` (or `--backend local`) runs ingestion, queries
  and evaluation on Ollama or an OpenAI-compatible server, a
  sentence-transformers cross-encoder and embedded SQLite graph and vector
  stores and registry, so Ollama is the only service to run (#209,
  #271-#273).
- Incremental indexing that converges with a full build and survives
  interruption: a write-ahead registry record, removal by the text units
  every item cites, one writing run per namespace (a lease), a retry limit
  and quarantine for documents that keep failing, and an online
  `indexing.reset` (#175, #186-#199, #325, #327, #331, #346).
- Re-runs pay only for unfinished work: an LLM response cache and a
  per-text-unit extraction store, so a resume or an edit re-asks only the
  model calls it lacks (#322, #326).
- One `models` section (two tiers, per-role overrides, one embedding model),
  one `providers` section per backend role and one `retry` section;
  unknown and moved keys fail the load (#334, #335, #468).
- The Bedrock guardrail screens the user query and the answer with
  `ApplyGuardrail` instead of being attached to model calls, with an opt-in
  contextual grounding check (#319, #344, #462).
- Document attribute filters apply to every index and to graph expansion
  (#354, #360, #370, #381).
- Evaluation: three LLM-free evaluators (`graph_aware`, `retrieval`,
  `answer_match`), paired strategy comparison with bootstrap CIs and a
  pairwise judge, and a MuSiQue benchmark preparer (#121, #315, #324).
- Optional OpenTelemetry tracing of ingestion and queries, exported over OTLP
  or to Langfuse or Arize Phoenix (#207, #208).
- Claude Sonnet 5.5 is the default model and Claude Haiku 5.5 the fast tier
  (#117, #162, #170).
- A CDK stack (`iac/`) with Well-Architected security defaults, a read-only
  query role, alarms and a one-task Step Functions ingestion pipeline (#157,
  #211, #319, #328).
- Evaluation and visualization are optional extras: a core install is about
  a third of the size (#467).

### Breaking changes
Compared with 0.2.3. Nothing 0.2.3 stored is read again: see *Upgrading*.

Python and installation.
- Python 3.11 or 3.12 is required (`requires-python` `>=3.11, <3.13`); 3.10
  is no longer supported (#430).
- Evaluation judges and visualization are extras, so a core install
  (`uv sync`, `pip install -e .`) is ingestion and queries (145 to 83
  packages, 937 to 335 MB on Linux). Install `uv sync --extra eval --extra
  viz` (or `pip install -e '.[eval,viz]'`) where you evaluate or render
  (#467).
  - `eval` holds `ragas`, `datasets`, `pandas`, `langchain-community` and
    `langchain-openai`. The `langchain` and `ragas` evaluators stay in the
    default `evaluation.enabled_evaluators`; without the extra,
    `EvaluationManager` (and `run-eval`, exit 2, before the first query)
    raises `ConfigurationError` naming it, also under
    `processing.ignore_errors`. The LLM-free evaluators need no extra; the
    container image installs `eval`.
  - `viz` holds `bokeh` and `pyvis`. Without it `run-visualization` exits 2
    naming the extra and `render_all` raises `ConfigurationError`;
    `from unified_kg_rag import BaseRenderer, register_renderer` still works.
    `umap-learn` and `scikit-learn` are no longer dependencies.
  - The security floors follow the packages that pull the pinned package in:
    `pillow` in `eval` and `viz`, `llvmlite` in `unstructured`,
    `langchain-openai` in `eval` and `local`.
- `graspologic` is no longer a dependency: Leiden runs on
  `graspologic-native` directly (#225, #226).

Configuration loading.
- Unknown config keys, in the file or as `UKR__` environment variables, are
  an error instead of a warning. Every CLI takes `--allow-unknown-keys` to
  warn and go on; `get_config` and `ConfigLoader` take
  `allow_unknown_keys=True` (#335).
- A moved key fails the load naming the key that replaces it (the *Moved
  keys* table of `docs/configuration.md`); a removed key loads with a
  WARNING and is ignored (its *Removed keys* table) (#334, #344, #467,
  #468, #470).
- Environment overrides use one prefix: `UKR__<SECTION>__<KEY>` sets any
  config key (double underscores between path segments, any case; JSON for
  lists and mappings). The former names are no longer read (#335):
  `NEPTUNE_ENDPOINT` is `UKR__AWS__NEPTUNE__ENDPOINT`;
  `OPENSEARCH_ENDPOINT` / `_USERNAME` / `_PASSWORD` are
  `UKR__AWS__OPENSEARCH__ENDPOINT` / `__USERNAME` / `__PASSWORD`;
  `BEDROCK_REGION` is `UKR__AWS__BEDROCK__REGION_NAME`;
  `BEDROCK_GUARDRAIL_IDENTIFIER` / `_VERSION` are
  `UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER` / `__VERSION`;
  `S3_BUCKET_NAME` is `UKR__AWS__S3__BUCKET_NAME`; `LOG_LEVEL` /
  `LOG_FORMAT` / `LOG_TO_FILE` / `LOG_FILE_PATH` are `UKR__LOGGING__LEVEL` /
  `__LOG_FORMAT` / `__LOG_TO_FILE` / `__LOG_FILE_PATH`;
  `OPENSEARCH_NUMBER_OF_REPLICAS` is
  `UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`;
  `OPENSEARCH_KOREAN_ANALYZER` is
  `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`;
  `GRAPHRAG_DOC_STATUS_TABLE` / `_CREATE_TABLE` are
  `UKR__AWS__DYNAMODB__TABLE_NAME` / `__CREATE_TABLE_IF_MISSING`;
  `GRAPHRAG_SOURCE_SCOPE` is `UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE`;
  `GRAPHRAG_OBSERVABILITY_ENABLED` / `_BACKEND` are
  `UKR__OBSERVABILITY__ENABLED` / `__BACKEND`; and `GRAPHRAG_SOURCE_DIRECTORY`
  / `GRAPHRAG_PIPELINE_ID` / `GRAPHRAG_METRICS_ENVIRONMENT` are
  `UKR_SOURCE_DIRECTORY` / `UKR_PIPELINE_ID` / `UKR_METRICS_ENVIRONMENT`.
  The CDK stacks and the container entrypoint set the new names.
- `AWS_REGION` and `AWS_PROFILE` fill `aws.region_name` /
  `aws.profile_name` only when the file leaves them unset; use
  `UKR__AWS__REGION_NAME` to override the file (#335).
- Switching `--backend` (or any provider flag) rebuilds the config from what
  the file, the environment and the caller set, so a value you set that
  equals the old preset's is kept (`Config.as_input`; the removed
  `strip_backend_preset` dropped it) (#335).
- `indexing.opensearch.index_settings` refuses the `knn` keys (`knn`,
  `knn.*`, with or without `index.`) and a setting spelled both with and
  without `index.`; set ef_search through `vector_search.ef_search` (#408).

Configuration keys.
- Every model setting is in one `models` section: `models.default` and
  `models.fast` (the tiers), `models.roles.<role>` (a model id or a tier
  name), `models.embedding` (model id and dimension, used by the indices,
  queries, RAGAS and the visualization) and `models.rerank`. The provider of
  each family is `providers.<llm|embedding|rerank>.kind`. The CLI flags
  `--llm-model`, `--embedding-model` and `--embedding-dimension` set the new
  keys. A setting of a provider the config does not select is ignored with a
  WARNING. With `backend: local`, another embedding model than the preset's
  needs its dimension too. Library callers read models through
  `Config.model_id(ModelRole.X)`, `tier_model_id`, `model_tier`,
  `embedding_model_id`, `embedding_dimension` and `rerank_model_id`
  (#334). Moved keys (the load fails naming the replacement):

  | Removed key | Replacement |
  |---|---|
  | `aws.bedrock.default_model_id` | `models.default` |
  | `aws.bedrock.fast_model_id` | `models.fast` |
  | `providers.llm.default_model_id` | `models.default` |
  | `providers.llm.fast_model_id` | `models.fast` |
  | `fixing.fixing_model_id` | `models.roles.fixing` |
  | `processing.chunking.chunking_model_id` | `models.roles.chunking` |
  | `processing.translation.translation_model_id` | `models.roles.translation` |
  | `processing.graph_extraction.extraction_model_id` | `models.roles.graph_extraction` |
  | `processing.graph_extraction.description_summarization.summary_model_id` | `models.roles.description_summary` |
  | `processing.gleaning.graph_refinement_model_id` | `models.roles.gleaning` |
  | `processing.claim_extraction.extraction_model_id` | `models.roles.claim_extraction` |
  | `graph.community_detection.report_generation.report_generation_model_id` | `models.roles.community_report` |
  | `search.translation_model_id` | `models.roles.query_translation` |
  | `search.entity_extraction_model_id` | `models.roles.entity_extraction` |
  | `search.strategy_selection_model_id` | `models.roles.strategy_selection` |
  | `search.query_rewrite_model_id` | `models.roles.query_rewrite` |
  | `search.answer_generation_model_id` | `models.roles.answer_generation` |
  | `search.global_search.community_relevance_model_id` | `models.roles.community_relevance` |
  | `search.global_search.map_model_id` | `models.roles.global_map` |
  | `search.global_search.map_reduce_model_id` | `models.roles.global_reduce` |
  | `search.drift_search.primer_model_id` | `models.roles.drift_primer` |
  | `search.drift_search.action_model_id` | `models.roles.drift_action` |
  | `evaluation.evaluation_model_id` | `models.roles.evaluation` |
  | `providers.embedding.model_id` | `models.embedding.model_id` |
  | `providers.embedding.dimension` | `models.embedding.dimension` |
  | `indexing.opensearch.embedding_model_id` | `models.embedding.model_id` |
  | `indexing.opensearch.embedding_dimension` | `models.embedding.dimension` |
  | `evaluation.embedding_model_id` | `models.embedding.model_id` |
  | `providers.rerank.model_id` | `models.rerank` |
  | `search.reranking.rerank_model_id` | `models.rerank` |
- `aws.dynamodb.enabled` moved to `providers.doc_status.enabled` (default
  `false`; `providers.doc_status.kind` picks DynamoDB or SQLite), the one
  switch for incremental indexing (#344).
- `graph.analysis.centrality` moved to `graph.visualization.centrality`:
  only the visualization computes centrality (#335).
- Retries are configured in one section, `retry`, and every knob counts
  total attempts, the first included: a 0.2.3
  `indexing.neptune.max_retries: N` is `retry.graph_store_max_attempts:
  N+1`, while 0.2.3's `processing.max_retries` had no effect (each model
  call made 5 attempts, the `retry.max_attempts` default) (#151, #216,
  #468). The keys below fail the load naming the new key, so a tuned value
  is not silently lost (#470); the two 0.2.3 keys also say how to convert
  their value (#472). A
  Neptune retry backs off from
  `retry.base_delay_seconds`, as the model calls do (both default 2 s).
  `BatchProcessor.from_processing` is `BatchProcessor.from_config(config)`,
  and `TransientRetryConfig` is no longer a config section
  (`RetryConfig.transient_policy()` builds it):

  | Removed key | Replacement |
  |---|---|
  | `processing.max_attempts` | `retry.max_attempts` |
  | `aws.bedrock.transient_retry.max_attempts` | `retry.max_attempts` |
  | `aws.bedrock.transient_retry.base_delay_seconds` | `retry.base_delay_seconds` |
  | `aws.bedrock.transient_retry.max_delay_seconds` | `retry.max_delay_seconds` |
  | `aws.bedrock.transient_retry.max_total_seconds` | `retry.max_total_seconds` |
  | `providers.llm.max_attempts` | `retry.model_server_max_attempts` |
  | `providers.embedding.max_attempts` | `retry.model_server_max_attempts` |
  | `indexing.neptune.max_attempts` | `retry.graph_store_max_attempts` |
  | `indexing.neptune.retry_delay_seconds` | `retry.base_delay_seconds` |
  | `search.store_max_attempts` | `retry.query_store_max_attempts` |
  | `evaluation.ragas_max_attempts` | `retry.judge_max_attempts` |
  | `processing.max_retries` | `retry.max_attempts` |
  | `indexing.neptune.max_retries` | `retry.graph_store_max_attempts` |

  The pre-release `search.llm_retry`, `providers.llm.max_retries`,
  `providers.embedding.max_retries` and `evaluation.ragas_max_retries` are
  unknown keys now: set the `retry` section instead (#216).
- Ingestion no longer renders visualizations: it writes
  `visualization_data.json` (`graph.visualization.export_data`, default
  `true`; graph and communities, no model call) and `run-visualization`
  renders it with a seeded spring layout. The embedding-based layout (UMAP,
  t-SNE or PCA over node embeddings), the `unified_kg_rag.visualization`
  package (`GraphVisualizationManager`, `BedrockNodeEmbedder`,
  `DimensionalityReducer`, `HTMLExporter` and its
  `graph_analysis_report.html`) are gone; drive the renderers through
  `unified_kg_rag.adapters.renderers` (`render_all`, `RenderContext`). The
  keys load with a WARNING and are ignored (#467):

  | Removed key | Replacement |
  |---|---|
  | `graph.visualization.enabled` | `run-visualization --data-path <outputs_directory>/visualization_data.json` |
  | `graph.visualization.embedding_method` | none: the layout is a spring layout |
  | `graph.visualization.layout_method` | none: the layout is a spring layout |
  | `graph.visualization.layout` (`umap`, `tsne`, `pca`) | none |
  | `graph.visualization.embeddings` | none: the layout uses no embeddings |
- `graph.visualization.interactive` and `static` are typed sections: a
  misspelled renderer setting is an unknown key, values are validated, and
  setting one key keeps the others' defaults (#344).
- The persisted S3 embedding cache is gone: an incremental run embeds only
  what its delta re-indexes, and a full rebuild embeds the corpus again. The
  keys load with a WARNING and are ignored; delete
  `s3://<bucket>/embedding-cache/` once if you had turned the cache on
  (#468):

  | Removed key | Replacement |
  |---|---|
  | `indexing.opensearch.persist_embedding_cache` | incremental indexing (`providers.doc_status.enabled`) |
  | `indexing.opensearch.embedding_cache_s3_key` | none |
- `search.local_search.type_quota` (ten keys) is gone: the fusion slots per
  section type are the constant `BaseSearchStrategy.TYPE_QUOTA` with the
  former defaults; the context mix is `search.token_manager.type_budgets`
  (#468).
- `cache.chunking` (`enabled`, `chunk_size`, `max_file_size_mb`) is gone: a
  stage-cache entry of more than 1,000 items or 50 MB is always written in
  chunk files (#468).
- `aws.bedrock.guardrail.apply_to` is gone with the attached-guardrail
  path: the guardrail screens the user query and the final answer through
  `ApplyGuardrail` (`aws.bedrock.guardrail.check_input` / `check_output`,
  both on); ingestion and evaluation models are never guarded. A blocked
  query gets the blocked message without retrieval
  (`RAGOutput.metadata.blocked`), a masked one continues with the masked
  text, interventions are in `metadata.guardrail`, a guardrail error fails
  the query, and with `check_output` `stream` / `astream` deliver the answer
  as one chunk after the check. A role outside the CDK stack needs
  `bedrock:ApplyGuardrail` on the guardrail ARN (#104, #319, #344).
- `indexing.cross_run_merge` is gone: an incremental run always merges its
  delta into the stored graph (`false` overwrote entities other documents
  still cite) (#344).
- These 0.2.3 keys are removed and fail the load as unknown keys:

  | Removed key | Replacement |
  |---|---|
  | `search.context_building_model_id`, `custom_prompts.context_building_*` | none: the post-retrieval context-building call is gone; the query rewrite uses the history (#212) |
  | `search.drift_search.enable_query_refinement`, `enable_keyword_extraction`, `max_iterations`, `initial_top_k`, `summary_length`, `summary_char_limit`, `n_entities`, `convergence_threshold`, `improvement_threshold`, `enable_primer`, `primer_follow_ups`, `query_refinement_model_id`, `keyword_expansion_model_id`, `convergence_assessment_model_id`; `custom_prompts.query_refinement_*`, `keyword_expansion_*` | `k_followups` (default 10), `primer_folds` (5), `n_depth` (2), `local_context_tokens`, `max_answer_tokens`, `models.roles.drift_primer` / `drift_action`; `custom_prompts.drift_primer_*` must emit a 0-100 integer `score`; `drift_hyde_*` and `drift_action_*` are new (#323, #347) |
  | `search.global_search.max_communities`, `map_batch_size`, `max_text_units` | `community_depth` (default 2), `max_reports_per_level` (100), `map_context_tokens` (12000); `text_unit_slots` is an integer, default `0` (#318) |
  | `search.global_search.graph_timeout_seconds` | none (#161) |
  | `indexing.cross_run_fuzzy_merge` | `processing.fuzzy_entity_resolution` (#214) |
  | `graph.community_detection.report_generation.include_statistics`, `include_key_entities` | none (#214) |
  | `processing.gleaning.quality_threshold`, `convergence_threshold`, `convergence_change_scale`, `min_improvement_threshold`, `quality_completeness_weight`, `initial_quality_entity_scale`, `initial_quality_relationship_scale` | none: gleaning stops per unit once it adds nothing new (#171) |
  | `indexing.neptune.min_entity_importance` | none (#155) |
  | `indexing.opensearch.hybrid_search_pipeline_name` | none: hybrid search fuses in the retriever; delete an old pipeline with `DELETE _search/pipeline/<name>` (#306) |
  | `indexing.opensearch.terms_batch_size`, `max_total_clauses`, `reserved_clauses` | none: a list filter is one `terms` query; more values than `max_terms_count` raise `InvalidFilterError` (#258) |
  | `processing.chunking.content_type` | none: documents are chunked as plain text (#289) |
  | `processing.chunk_concurrency`, `processing.batch_size` | `processing.max_concurrency` (one pool of in-flight calls) (#296) |
- The `"NONE"` alias of `aws.s3.encryption.encryption_type` is removed; use
  `BUCKET_DEFAULT` (#216).

Defaults (set the old value to keep the 0.2.3 behaviour).

| Setting | 0.2.3 | Now |
|---|---|---|
| `models.default` / `models.fast` | Claude Sonnet 4.5 / Haiku 4.5 | Claude Sonnet 5.5 / Haiku 5.5 (#117, #170) |
| `aws.region_name` | `ap-northeast-2` | `us-west-2` (#148) |
| `aws.bedrock.region_name` | `us-west-2` | `null` (follows `aws.region_name`) (#148) |
| `aws.s3.encryption.encryption_type` | `AES256` | `BUCKET_DEFAULT` (#105) |
| `aws.dynamodb.table_name` | `unified-kg-rag-on-aws-doc-status` | `unified-kg-rag-on-aws-doc-status-v2` (#327) |
| `processing.chunking.min_chunk_size` / `max_chunk_size` / `fallback_chunk_size` | 5000 / 50000 / 50000 | 1000 / 8000 / 4800 (#125) |
| `processing.max_concurrency` | 20 | 40 (20 under `backend: local`) (#310) |
| `processing.similarity_threshold` | 0.6 | 0.85; similarity matching is opt-in (`processing.fuzzy_entity_resolution`) (#214) |
| `indexing.neptune.property_max_length` | 1000 | 4000 (#132) |
| `graph.community_detection.auto_resolution` | `true` | `false` (#141) |
| `graph.visualization.outputs_directory` | `outputs/visualization` | `null` (`<cache directory>/<pipeline_id>/visualization`) |
| `search.token_manager.max_context_tokens` | 200000 | 30000; 22000 under `backend: local` with a non-Bedrock LLM (#125, #224, #226) |
| `search.fusion.fusion_weights` | 15 entries of 1.0 | `{}` (every bucket 1.0) (#258) |
| `search.global_search.relevance_threshold` | 0.5 | 0.2 (#318) |
| `search.global_search.use_dynamic_selection` | `true` | `false` (#318) |
| `search.auto_routable_strategies` | (fixed: `simple`, `local`, `global`, `drift`) | `local`, `mix`, `global`; add `drift` to route to it (#347) |
| `evaluation.enabled_evaluators` | `langchain`, `ragas` | also `answer_match`, `retrieval`, `graph_aware` (#121) |
| `evaluation.max_context_tokens` | 8192 | `null` (the answer budget) (#315) |

Behaviour.
- Entity resolution merges only names equal after normalization, legal-form
  removal and dense-script spacing; entity and relationship `rank` come from
  node degree, and relationship weight sums the strengths per text unit
  (#189, #214).
- A query with conversation history is rewritten as a standalone question
  (`models.roles.query_rewrite`) and every step reads the rewrite (#212).
- Conversation memory is per chain, keyed by index suffix, `user_id` and
  `conversation_id` (`ConversationKey`); pass one manager to share history
  (`GraphRAGChain(memory_manager=shared)`). `get_memory_manager` is removed,
  and `MemoryManager.load_history` / `add_turn` / `get_or_create_memory` take
  a `ConversationKey` (#218, #333).
- A filter key no retriever of the strategy declares raises
  `InvalidFilterError` (#123); querying a suffix nothing was ingested under
  raises `IndexNotFoundError` (#178).
- Entities no longer have `MemberOf` edges to communities; an entity names
  its communities (`community_ids`) (#348).
- `run-eval --search-strategy` parses to a list (`["auto"]` by default);
  with several strategies or `--repeats` the listed strategy overrides an
  item's own `search_strategy` (#324). `abstention_statistics.unanswerable`
  has the overall block's shape (`abstained`, `answered`,
  `abstention_rate`) (#292).
- Hidden files and files under a hidden or `__pycache__` subdirectory of the
  corpus are not parsed (#213).

Command line. The CLIs share one convention (user guide §10, *CLI exit codes
and JSON output*); renamed flags have no alias:
- `run-ingestion --target-directory` is `--output-directory`;
  `run-eval --eval-data-path` is `--data-path` and `--outputs-directory`
  `--output-directory`; `run-visualization --output-dir` is
  `--output-directory`; `run-prompt-tuning --output` is `--output-path`, and
  its `--source-dir` alias is gone (use `--source-directory`) (#333).
- `run-ingestion --repair-metadata` is removed (it never repaired anything);
  use `--verify-metadata` and `--force-rebuild` or a new `--pipeline-id`
  (#248).
- Flags must be spelled out: an abbreviation is rejected (exit 2) (#333).
- Exit codes: `2` for every usage or configuration error before any work,
  `1` for a failed run, `130` for Ctrl-C and `143` for SIGTERM (#333, #344).
- Every CLI takes `--output-format json` (one JSON result on stdout, the
  rest on stderr); `run-ingestion` rejects it with `--metrics-sink
  cloudwatch` (#333).

Deployment (CDK stack).
- `iac/` is a uv project of its own: install it with `cd iac && uv sync`;
  `iac/requirements.txt` and `requirements-lock.txt` are gone (#251). The
  CDK lock needs AWS CDK CLI 2.1146.1 or later
  (`npm install -g aws-cdk@2.1146.1`) (#438).
- The ingestion state machine runs one Fargate task (state `Ingest`) instead
  of four phases; scripts that read `.States.PrepPhase` read `.States.Ingest`,
  and the task's result is under `$.task` (was `$.last_phase`) (#328).
- Queries get their own task definition and read-only role
  (`QueryTaskDefinitionArn` / `QueryTaskRoleArn` outputs of
  `GraphRagCompute`); run `run-rag` / `run-eval` with it and keep `run-eval`
  datasets and results under `s3://<cache bucket>/eval/` (#319).
- The doc-status table is a new one (`graphrag-doc-status-v2` /
  `<env>-graphrag-doc-status-v2`, partition key `namespace`, sort key
  `doc_id`); a table of the 0.2.3 key schema is rejected at first use. The
  old table is retained outside `dev` (#327).
- The next deploy adds `aws:SourceAccount` / `aws:SourceArn` conditions to
  the Step Functions role's trust policy (aws-cdk-lib's confused-deputy
  fix) (#438).
- The pipeline EMF metrics carry an `Environment` dimension
  (`UKR_METRICS_ENVIRONMENT`) and the alarms and dashboard read only that
  series: deploy the image and the stack together (#330).
- `create_guardrail` decides whether the stack owns a guardrail; bring your
  own with `-c create_guardrail=false` (#103).
- In `network_mode=private` (the default) the VPC has no public subnets,
  internet gateway or `0.0.0.0/0` routes, and the OpenSearch domain
  publishes only its application logs; both update in place (#251).

Library API (Python callers and custom backends).
- The public API is what `unified_kg_rag` exports (`unified_kg_rag.__all__`):
  `Config`, `get_config`, `DataIngestionPipeline`, `IndexingManager`,
  `GraphRAGChain` with `RAGInput` / `RAGOutput` / `SearchOutput`,
  `EvaluationManager`, the errors it raises and the extension surface (the
  ports, `Providers`, `register_provider`, `BaseSearchStrategy` /
  `register_strategy`, `ParserFactory`, `register_renderer`). Other module
  paths are internal and moved (#164, #210, #216, #312, #456, #464):
  import from the package root (#333).
- Retrieval-only queries are `GraphRAGChain.search()` / `asearch()`, which
  return a `SearchOutput`; `ChainMode` and `GraphRAGChain(mode=)` are
  removed. Replace `GraphRAGChain(cfg, mode=ChainMode.SEARCH).ainvoke(q)`
  with `GraphRAGChain(cfg).asearch(q)` and read attributes
  (`out.search_results.results`) (#333).
- One injection path: `doc_status=`, `vector_indexer=`, `graph_indexer=`,
  `model_factory=` and `retriever_builders=` cannot be combined with
  `providers=`. Only `Providers`, `build_providers`, `DataIngestionPipeline`
  and `GraphRAGChain` take `boto_session`; `GraphRAGChain` rejects unknown
  keyword arguments; `OpenSearchIndexer`, `OpenSearchRetriever` require
  `embedding_factory=` (#216, #218, #221).
- A custom doc-status registry implements the whole `DocStatusPort`: batch
  reads and writes (`get_many`, `put_many`, `record_fits`),
  `list_namespace`, the extraction settings (`get_extraction_settings` /
  `put_extraction_settings`), the namespace lease (`try_acquire_lease`,
  `lease_holder`, `release_lease`), the `DocStatusRecord` fields
  (`text_unit_ids` only, `extraction_fingerprint`, `attempt_count`,
  `attempted_hash`, `attempt_batch`, `crash_batch`, `failure_count`,
  `retry_count`), `DocStatus.QUARANTINED` and a `diff` that takes the
  extraction fingerprint and fills `DocumentDelta.deferred` /
  `quarantined_unchanged`. `IncrementalIndexer.plan` / `prune_changed` /
  `remove_deleted` are `detect_delta` and `remove_changed_and_deleted`
  (#214, #224, #327, #331, #332, #346).
- A custom graph backend implements `GraphIndexer.find_ids_citing` and a
  custom vector backend `VectorIndexer.find_documents` and
  `delete_extraction_records`: without them a removal cannot find what a
  removed document stated (a WARNING says so). `find_entity_ids_by_name_key`,
  `count_neighbors`, `begin_rebuild` / `end_rebuild` and
  `check_index_creation` are optional overrides. Full-build `index_*`
  methods take a keyword-only `suffixes`; `initialize()` / `clear()`
  return `None` and raise the store's typed error; `get_stats`,
  `get_entity_count` and `IndexingManager.clear_all_data` are gone (#163, #169, #214, #294, #327,
  #331, #384, #416).
- A custom retriever or backend whose error must fail the query raises a
  `ConnectionError` or `ConfigurationError` subclass (the typed store errors
  in `shared/exceptions.py`); retrieval errors are classified by type only
  (#261).
- Custom strategies: `asearch(query, config=None)` takes `config` (#156),
  `_fuse_and_rerank` takes a required `rerank_trace=` (#284), and
  `BaseSearchStrategy.search` (sync) is removed (#249). Subclasses that
  overrode `GraphRAGChain._build_graph_retriever` /
  `_build_document_retriever` inject `retriever_builders` (#212).
  `NeptuneRetriever` expands only from seed ids passed as an `id` filter
  (#249).
- Custom model factories: `get_model_info` returns a `LanguageModelInfo`;
  opt into the chain-level retry with `is_transient_error(exc)` and
  `transient_retry` (#216).
- Pipeline records: `PipelineContext.unit_extractions` replaces `entities`,
  `relationships` and `rejected_entities`; `GraphExtractor.extract_units`
  and `GraphGleaner.glean` replace `extract_from_text_units` and
  `glean_graph` (#322). `Document` keeps its text once, in `page_content`
  (`content`, `pages`, `total_pages`, `file_type`, `DocumentContent` and
  `Page` are gone) (#289). `DirectoryLoader` takes only
  `source_directory` (#299, #303).
- `BatchProcessor(max_retries=)` is `max_attempts=`; it rejects unknown
  settings (`batch_size=`, `chunk_concurrency=`), and
  `execute_with_fallback` / `aexecute_with_fallback` take no `batch_func`
  (#151, #174, #216, #301, #303).
- `EvaluationRunner(args, rag_chain, config)` takes the run's config (#292);
  a `BaseGraphRAGEvaluator` subclass implements `evaluate_single` or
  `aevaluate_single` (#305).
- Removed or moved helpers, with their replacement: incremental removal no
  longer plans from registry lineage (`DocumentLineage` keeps only
  `text_unit_ids`, `build_document_lineage(documents, text_units, suffix)`,
  `IncrementalIndexer.record(artifact_owners=)`; the lineage-overflow port
  methods, `DocumentDelta.orphan_overflow` and the legacy-key adoption are
  gone) (#327); `MemoryManager.get_langchain_memory` / `add_message` /
  `load_context` are `load_history` / `add_turn` (no `entities=`, no
  `providers=`) (#212, #218); `NeptuneClient.connection` is
  `NeptuneClient.g` (#266); `ports.CachePort` and
  `shared.get_cache_manager` are
  `application.ingestion.cache_manager.CacheManager`, and
  `S3CacheManager` requires `prefix=` (#216, #300, #345); `BaseEvaluator`
  is `BaseGraphRAGEvaluator` (#300); `LanguageModelInfo`,
  `EmbeddingModelInfo`, `RerankModelInfo` live in `ports.model_factory`,
  the evaluator bases in `adapters.evaluators.base`, the chain factory in
  `adapters.llm.chain_factory`, and the doc-status adapters in
  `adapters.doc_status` (#216, #312); `BaseIndexer.initialize` is a no-op
  by default (#309).
- Removed with no replacement: `MetricsMixin` / `get_metrics()` on
  retrievers, strategies, `HybridScorer` and `TokenManager` (#253);
  `SearchQuery.label_prefixes` (#260); the
  `SearchQuery.metadata["lightrag_mode"]` alias (#212);
  `put_search_pipeline` on the search clients (#306);
  `compute_hash(algorithm=)` (#300); `reload_config` (#344);
  `Entity.frequency` (#291); and the other unused helpers listed in #169,
  #216, #263, #266, #267, #300, #344, #414, #418 (among them
  `Document.detected_language`, `DocStatus.PARSING` / `PROCESSING`,
  `NeptuneClient.delete_vertices_in_batches`, `BaseParser.parse_files`,
  `ParsingStats`, `GraphAnalyzer.centrality_cache`,
  `registered_strategies` and `NeptuneRetriever.DEFAULT_MAX_HOPS`);
  `GraphAnalyzer.set_community_data` / `export_graph_data` / `clear_cache`
  and the `chunk_size` / `max_file_size_mb` / `enable_chunking` arguments
  of `CacheManager` (#471).

### Upgrading
From 0.2.3 (or a `main` checkout since):
1. Install on Python 3.11 or 3.12, with the extras you use (`eval`, `viz`,
   `local`, `otel`, `unstructured`). Without the `local` extra a CLI whose
   providers need it exits 2 at start-up, naming it (#472).
2. Load your config once and fix what it reports: unknown keys (the table
   of removed 0.2.3 keys above), moved keys (the load names the
   replacement), and removed keys (a WARNING names the replacement). The
   0.2.3 template's empty `processing.document_parsing.source_directory`
   loads as unset (#472). Rename
   environment variables to `UKR__...`, and set any 0.2.3 default you rely
   on (*Defaults* above). Update scripts for the renamed CLI flags and exit
   codes.
3. CDK: install AWS CDK CLI 2.1146.1 or later, run `cd iac && uv sync`, and
   deploy the stacks together with the new image (the environment names,
   the EMF dimension, the query task and the new registry table need
   both). Run queries with `QueryTaskDefinitionArn`. Korean on the CDK
   domain needs `-c opensearch_nori_package=true` (or keep the stack's
   `cjk`) (#282, #459).
4. Rebuild every index once: nothing 0.2.3 stored is read (index mappings,
   ids, registry format, stage caches). Run one ingestion with
   `indexing.reset: true` and the registry on, in the new registry table,
   and resume no 0.2.3 `--pipeline-id`: use `--force-rebuild` or a new id
   (#327, #331, #332). The compatibility code for earlier builds is gone:
   an index without recorded embedding settings (`_meta`) is refused, a
   0.2.3 stage-cache entry is a miss, and a parsed-document JSON's
   `content.text` is no longer read as its text (write `page_content`)
   (#289, #331, #344).
5. Clean up: delete the old registry table once the rebuild succeeded, an
   OpenSearch hybrid search pipeline 0.2.3 wrote
   (`DELETE _search/pipeline/graphrag-hybrid-search-pipeline`), and
   `s3://<bucket>/embedding-cache/` if you had used it.
6. A role with a custom IAM policy needs `dynamodb:BatchGetItem`,
   `BatchWriteItem`, `Query` and `DescribeTable` on the registry table,
   `dynamodb:UpdateTimeToLive` where the library creates the table,
   `bedrock:ApplyGuardrail` on the guardrail and `bedrock:CountTokens` with
   `InvokeModel` (#191, #194, #298, #319, #327, #331).
7. Library callers and custom backends: see *Library API* above.
8. Local mode: a `backend: local` config that relied on the preset for the
   stores gets the embedded SQLite stores; add
   `providers.graph.kind: gremlin` and `providers.vector.kind: opensearch`
   (as `docker/config.local.yaml` does) to keep the containers (#273).

### Added
- Local mode: the `providers` section and provider registry
  (`@register_provider`), Ollama and OpenAI-compatible models, a
  cross-encoder reranker, embedded SQLite graph and vector stores and
  registry, the `[local]` extra and an Ollama service in
  `docker/compose.local.yaml` (#154, #209, #271-#273).
- Observability: the `observability` section and `otel` extra with spans for
  runs, stages, batches and query steps, GenAI model-call spans, OTLP,
  Langfuse and Phoenix exporters and redacted content capture (#207, #208,
  #211, #219); CDK `enable_cloudwatch_tracing` with an ADOT sidecar (#269).
- Incremental indexing: the extraction-settings fingerprint (#224, #332),
  `indexing.max_document_failures` (#175) with own-error accounting and
  `documents_retrying` (#431), source scopes and
  `--retire-source-scope` (#134, #192), the namespace write lease and
  quarantine (`indexing.max_document_attempts`) (#331, #346), community
  drift metrics (#320, #347).
- The extraction store (per-text-unit extraction records) and the LLM
  response cache (`cache.llm_responses`) (#322, #326).
- Global search's community hierarchy cut and dynamic selection, and DRIFT's
  primer, follow-up and reduce loop with `search.max_concurrency` for
  query fan-out (#318, #323, #347).
- Document attribute filters on relationships, claims, reports and graph
  expansion (#354, #356, #360, #370, #381).
- The guardrail's `ApplyGuardrail` screening (#319) and opt-in contextual
  grounding check `aws.bedrock.guardrail.contextual_grounding` (CDK
  `-c guardrail_contextual_grounding=true`); `GroundingCheck.blocked` says
  whether the check blocked the answer (#462, #471).
- Per-query rerank status (`metadata.rerank`) and opt-in
  `search.reranking.required` (#284).
- Evaluation: `retrieval` and `answer_match` evaluators, grouped
  statistics, the run manifest and index build record, `s3://` dataset and
  output paths (#113, #121, #206, #315, #321); guardrail block counts
  (#345, #355); paired strategy comparison (`--search-strategy a,b
  --repeats N`, `--pairwise-judge`) and `prepare-benchmark musique` with
  `evaluation.answer_normalization: official_squad` (#324).
- Model catalog: Claude Sonnet/Opus 5.5, Opus 4.6-4.8, Sonnet 4.6, Haiku 5.5
  and OpenAI GPT models on Bedrock, free-form ids with
  `aws.bedrock.model_overrides` (#117, #127, #162); reasoning effort per
  tier, for ingestion and for judges (#113, #128, #179).
- `aws.bedrock.query_prompt_cache_ttl: 1h` for query-time prompt caching
  (#458).
- `processing.chunking.length_unit: tokens`: chunk sizes in a script-aware
  token estimate instead of characters (#460).
- `indexing.neptune.vertex_ids: custom`: Neptune vertex ids
  `<label>:<item id>`. Graph expansion reads a label of the other mode, or
  one a full build is switching, by the `id` property (one WARNING), so
  queries keep working while ingestion and the query service switch. During
  a switch it keeps each entity's nearer copy, the rewritten one of two as
  near. A running query service follows a switch without a restart: in
  `custom` mode graph expansion reads the label's switch vertex on each
  query (one indexed read) and checks the label's mode again when a
  custom-id lookup finds none of its seeds; `property` mode needs no extra
  read (#466, #470, #473, #474).
- `indexing.opensearch.vector_mode` / `compression_level`: on-disk
  quantized vectors on OpenSearch 2.17+ (#461).
- The public API exports its errors (`GraphRAGException`,
  `ConfigurationError`, `InvalidFilterError`, `IndexNotFoundError`,
  `StoreConnectionError`, `StoreRequestRejectedError`,
  `PipelineExecutionError`, `PipelineInterrupted`, `NamespaceLeaseError`),
  `ModelRole`, `ModelPurpose`, `MetricsSink` and `CloudWatchEMFSink` (#334,
  #365, #366, #416, #420).
- Generated references: `docs/configuration.md` (#337) and `docs/cli.md`
  (#351), checked by unit tests.
- `run-prompt-tuning` reads every format ingestion parses, and each `run-*`
  `--help` ends with an example and a user-guide link (#121).
- Operability: bound run, stage and query ids in logs, extraction-failure
  metrics and alarms, start-up endpoint checks, runbook sections on backup,
  restore, replicas and OpenSearch upgrades (#122, #178, #211, #457).
- IaC: `neptune_serverless_ncu` (#319), `ingestion_ephemeral_storage_gib`
  (#328), `opensearch_master_instance`, `flow_log_retention_days` (#157),
  `sfn_timeout_hours`, `opensearch_volume_gib`, `guardrail_version` (#211)
  and a published `GuardrailVersion` (#345), the `BedrockThrottling` and
  `IngestionTaskMemory` alarms and dashboard widgets (#330), Fargate Spot
  for ingestion and ARM64 tasks (#457), the Nori analyzer package
  (`opensearch_nori_package`) (#459), and an `EnableVersionUpgrade` policy
  for in-place OpenSearch upgrades (#457).
- Extraction schema settings under `processing.graph_extraction`:
  `entity_types` items may be `{label, description, iri}` mappings, and the
  new `relationship_types`, `relationship_patterns` (`"PERSON -WORKS_FOR->
  ORGANIZATION"`), `schema_mode` (`off`/`align`/`strict`, default `align`)
  and `schema_source` keys are validated on load: labels are unique by
  case and word separators, and patterns name declared types. Extraction
  does not apply them yet. At their defaults they change no cache key,
  extraction record or recorded extraction settings (#PR).
- Schema sources: `schema_source.kind: file` reads the schema keys from a
  YAML or JSON file at start-up, keyed by its content, not its path, and
  `@register_schema_source` (`adapters/schema_sources/`) adds a source for
  another format or service (#PR).

### Changed
- Global search maps every report of one cut of the community hierarchy
  (packed by tokens, on the fast model), falls back to a shallower cut over
  `max_reports_per_level`, credits key points only to their reports, and
  dynamic selection descends to the leaves, as MS GraphRAG does (#318, #347,
  #354).
- DRIFT is MS GraphRAG's DRIFT (HyDE primer, follow-up local searches,
  reduce), with one rerank per query and at most about 27 LLM calls at the
  defaults (#323, #347).
- `auto` routes with the fast model and falls back to the first routable
  strategy; `mix`/`hybrid` retrievals, DRIFT steps and keyword extraction
  run concurrently; Neptune expansion is opt-in for LightRAG (#125, #132,
  #212, #224, #347).
- Fusion reranks before MMR diversity and the per-type cut; hybrid search
  normalizes lexical and kNN hits in the retriever over at least 50 hits
  each; budget a type cannot fill goes to the others; RRF scores are plain
  `weight / (rrf_k + rank)` sums; context packing ranks by score alone
  (#287, #306, #317).
- Communities: Leiden leaves are bounded by
  `graph.community_detection.max_cluster_size` (default 20), which also caps
  a parent's children, and components smaller than `min_community_size`
  get no community (`min_community_size: 1` clusters every component)
  (#314, #320, #347).
- Claims resolve only against the entities their own text units state, and
  ties of names, types and statuses break deterministically (smallest
  value), so builds do not depend on document order (#398, #402, #404,
  #442).
- `indexing.reset` rebuilds without taking the suffix offline (new indices
  and an alias swap; the graph rewritten in place), and a failed reset
  leaves the previous graph and records (#331).
- An incremental run finds what a changed or deleted document contributed
  by the text units every item cites, strips shared items and re-derives
  them, moves an item to the id a full build gives it, and deletes
  extraction records nothing names (#325, #327).
- Stage cache keys cover the whole configuration and the package version
  (#332).
- The CDK ingestion runs as one task per execution, on the task definition
  revision deployed with it, and emits metrics as each stage finishes (#328,
  #330).
- `BatchProcessor` runs one pool of `processing.max_concurrency` items; the
  OpenSearch query client and the Bedrock clients size their connection
  pools to the concurrency; OpenSearch indexing embeds and writes 800 items
  of a type at a time with 16 embedding requests in flight (#296, #309,
  #310).
- The `graph_analysis` stage no longer computes centrality, which no later
  stage read (#248).
- The parser uses its own loaders on langchain-core (same text and
  metadata, so document ids are unchanged; a 4,000-page PDF loads in about
  1 s); `ParserFactory.register_loader` takes any `BaseLoader` (#460, #471).
- Gleaning refines each text unit's own record and a correction changes only
  what that unit says (#322).
- `run-ingestion --s3-sync` takes the bucket from `aws.s3.bucket_name` when
  `--s3-bucket-name` is not given (#344).
- `logging.library_levels` quiets `opensearch`, `httpx`, `httpcore` and
  `numba` by default (#408, #411, #413).
- `config-template.yaml` is a short starting file with the keys most
  deployments set; `docs/configuration.md` has every key (#468).
- CDK alarms: `OpenSearchJvmPressure` follows AWS's recommended alarm and
  `NeptuneHighCpu` needs 3 consecutive 5-minute periods (#298).
- Documentation: the user guide, runbook and design doc were reorganized and
  describe the current release only; only `README.ko.md` and
  `docs/user-guide.ko.md` remain in Korean, and they link the English
  design doc, runbook and model catalog (#337-#342, #351, #469).
- Dependencies: pydantic 2.14, langchain-core 1.7.0, `langchain-aws>=1.8.1`
  (context-overflow errors), langchain-openai 1.7, langsmith 0.14, datasets
  5.1, tiktoken 0.14, tenacity 9.2, boto3 1.43.118; aws-cdk-lib 2.274 in the
  CDK lock; `nest-asyncio` is no longer used (#241, #251, #265, #438, #465).
- Tooling: development tools are in the `dev` group; CI pins uv 0.12.23,
  runs the tests in parallel and on Python 3.13 (non-blocking), checks
  licenses on every supported Python, and lints async and logging rules;
  `-m "not aws"` is the pytest default; the Dockerfile takes a
  `BASE_IMAGE` argument; a layering test enforces the hexagonal import rules,
  statically and, for every module that has its backends injected, in a
  clean interpreter; the core-install job runs without the dev group (a
  `test` group holds the test runner) (#119, #257, #297, #303, #428, #456,
  #464, #465, #470).
- Settings set in code on `processing.chunking`, `indexing.neptune`,
  `retry` and `aws.bedrock.guardrail` are validated on assignment (a string
  `length_unit` becomes the enum, `retry.max_attempts: 0` is refused), and
  the load warns when `contextual_grounding.enabled` has no guardrail
  identifier or `length_unit: tokens` keeps every character-default size
  (#470).
- `GraphRAGChain` refuses to build (`ConfigurationError`) when
  `aws.bedrock.guardrail.contextual_grounding.enabled` is on and
  `check_output` off, however code set them: the nested flag is validated
  without `check_output`, and a refused assignment stays applied (#472).
- A `processing.chunking.chunk_overlap` not below `fallback_chunk_size`
  fails the load, naming both, instead of every chunker at the chunking
  stage (#474).
- The extraction prompt's relationship-type examples are the
  `{relationship_types}` variable, which a `graph_extraction` custom prompt
  may use. The default prompt renders byte for byte as before, and
  extraction records keep their keys (#PR).
- An `entity_types` list with two labels that differ only in case or word
  separators (`PERSON`, `person`) fails the load (#PR).

### Fixed
Bugs present in 0.2.3.

Queries and retrieval.
- Graph expansion returns every seed and its neighbourhood and ranks by
  distance first, so hub neighbours no longer push seeds out (#161, #218).
- Graph expansion no longer depends on the order the graph store keeps a
  vertex's edges: a hub contributes its first `max_results_per_hop`
  neighbours by id, as the embedded store does, and of two equally short
  paths to an entity the one from the better-ranked seed (the caller's seed
  order), then the first by content, is kept, so two builds of one graph
  expand alike. Ordering reads every neighbour of an expanded hub;
  `indexing.neptune.deterministic_order: false` keeps the store's order
  instead (#470, #473).
- Local search's entity frequency filter no longer drops the query's own
  matched entities (#218), and local search keeps its seeds when graph
  expansion degrades (#261).
- Earlier turns' entities no longer leak into a follow-up's entity focus
  (#218), and `memory.max_conversation_age_hours` is enforced (#176).
- Korean BM25 recall, CJK fuzziness and NFC normalization in retrieval
  (#175, #177).
- Store failures fail the query instead of answering from no context: an
  unreachable store, a denied search (HTTP 401/403), a rejected one (HTTP
  400), an overloaded cluster after its retries, and an embedding call that
  cannot succeed (#249, #258, #261, #266).
- Queries no longer hang on a store that stops answering: the query client
  waits `search.store_timeout_seconds` per attempt, and a stuck Neptune
  traversal times out (#258, #264).
- Neptune: a closed websocket (restart, failover) reconnects at the next
  query and in ingestion, a lost connection no longer leaks threads or
  hangs an in-flight write, and concurrent queries share one connection
  attempt (#258, #261, #263, #266, #267).
- A short OpenSearch throttling burst no longer fails a query or a bulk
  write (#263, #264).
- A long question no longer fails its lexical search (Lucene's clause limit,
  typo-tolerance expansion) (#258, #260).
- Long list filters no longer raise the number of hits or multiply searches
  (#258, #260).
- The text-units section is no longer empty for a corpus translated into
  five or more languages (#264).
- The answer context stays within `max_tokens` (source prefixes and headers
  counted) (#258), a small per-query budget no longer yields an empty
  context (#247), and CJK sections truncate at a character, not the first
  space (#247).
- With MMR diversity on, `local` and `drift` keep every reserved text-chunk
  slot (#247); weighted fusion no longer warns about unweighted buckets on
  every query (#247, #258).
- `simple` and DRIFT no longer rank hits of different indexes by raw score,
  and local and LightRAG incident-edge expansion no longer keeps arbitrary
  edges of a hub (#295).
- LightRAG `mix`/`hybrid` and local bridge relationships keep every typed
  edge between two entities (#317); `hybrid` without keywords logs a warning
  (#295).
- Hybrid search on OpenSearch 2.13 no longer drops the `size`-th lexical hit
  or normalizes over a lower one (#295, #306); kNN on lucene uses
  `vector_search.ef_search` (#306).
- An attribute filter no longer fails vector and hybrid searches on
  OpenSearch 2.13 when an index never mapped the attribute (#306).
- Neptune graph expansion fetches at most its per-seed budget, nearest hop
  first (#310).
- `auto` no longer routes to `global` or `drift` without community reports
  (#317), and falls back to `local` only when it is routable (#224).
- A global-search map batch over the model's context is left unrated
  instead of failing the query; a map step that fails entirely fails the
  query or is recorded in `degraded_error` (#264, #267, #348, #354, #360).
- A query against an index suffix nothing was ingested under fails with
  `No indices found` (#227), one against indices of another embedding model
  or dimension fails with a `ConfigurationError` (#232), and queries embed
  with `models.embedding.dimension` (#232).
- A `GraphRAGChain` reused across event loops closes the previous loop's
  retrievers without stalling the next query (#230, #232, #243), and a
  LangChain `ainvoke` on a package retriever runs on the caller's loop
  (#300).
- Query translation catches a refusal written with a typographic apostrophe
  (#292).
- Cohere Embed v3 no longer embeds every text as an empty string (#463).
- The first model calls of a batch no longer fail with "dictionary changed
  size during iteration" (#369), and a call that failed once no longer
  fails its item for good while other items retry (#378).

Ingestion and incremental indexing.
- Incremental runs strip shared items instead of deleting them, repair
  interrupted runs, keep resets and deletions inside their namespace, and
  no longer keep what a changed or deleted document contributed to shared
  items (#163, #185, #200, #214, #325).
- Incremental runs merge a claim another document stored instead of
  overwriting it, keep one claim across spellings of an entity, and carry
  the stored entity's name on a new edge (#252, #290).
- A doc-status registry that cannot be read fails the loading stage
  (`DocStatusRegistryError`) instead of silently rebuilding (#186).
- Incremental indexing follows the extraction settings, not only the
  content (#224), and stops before writing after a change of the embedding
  model, languages or analyzers, or onto a suffix built with the registry
  off (#231, #233).
- Deleting a document under a non-default suffix strips its text units from
  shared claims (#256); a deletion-only run checks the configured
  suffix's indices (#252).
- An incremental run whose new documents name no entity commits them and
  applies its deletions (#248); it no longer replaces
  `visualization_data.json` with the delta's graph (#224).
- `indexing.reset` reusing a pipeline id rebuilds from the whole corpus
  (#233), fails when the registry cannot be cleared (#299, #305), and a
  reset whose graph writes partly failed no longer keeps the old graph
  beside the new one (#346). Two rebuilds within one second no longer reuse
  the live index (#221).
- Two corpora in one namespace holding the same file no longer share text
  units (#327); store sets with other index prefixes no longer share a
  registry namespace (#283).
- Neptune no longer loses relationship edges when a run is interrupted
  while writing them (#252, #377); a vertex write replaces its stored
  properties (#294); a long description is no longer written to OpenSearch
  cut at the Neptune property cap (#252).
- A store error while preparing or clearing the stores fails the stage
  (#294); an index create or delete whose response was lost no longer fails
  a full build (#265); a failed delete batch counts every id in it (#252);
  OpenSearch bulk errors name their reason and cause (#431, #432, #440).
- OpenSearch indexing memory no longer grows with the corpus (#252).
- Community detection survives NaN, infinite, zero, negative and absurd
  relationship strengths (#288, #304, #308), keeps its coarser levels and
  merges a small community into its heaviest neighbour (#288), and gives
  the same communities in every process (#250, #343).
- A failed parent-level community report no longer records every document
  under it `FAILED` (#248).
- Graph extraction and gleaning: the XML parser keeps `&`, `<`, repeated
  siblings and unclosed sections and fails truncated responses (#165, #174,
  #182, #213); an element without fields or an empty element no longer
  fails the unit or stores `{}` (#407, #452).
- Graph quality: gleaning no longer revives rejected entities or overwrites
  descriptions, claims no longer resolve onto another entity, CJK names with
  another head character no longer merge, and communities no longer
  duplicate (#201, #203, #213, #214).
- Gleaning corrections resolve their target among the refined unit's items,
  change only that unit's statement, and a re-proposal merges into the
  corrected item (#220, #291, #304, #308, #322).
- `entity_confidence_threshold` applies per chunk (#291);
  `entity_grounding.action: penalize` applies to gleaning additions (#250).
- Merged items carry every document's filters, with one merge function in
  full and incremental builds (#250, #304); graph resolution renames an
  edge's endpoint names to the merged entity (#250).
- Description re-summarization splits an over-long description map-reduce
  (#250).
- Unresolved claims are not indexed (#220); an entity named only inside a
  relationship gets its suffix and filters (#220); deduplication prefers a
  successfully indexed copy (#220); a Neptune label clear is bounded (#220).
- Chunking and translation no longer hide errors in catch-alls (#250); a
  chunk-boundary error falls back to a size split with a WARNING (#289).
- `DataIngestionPipeline.run(source_directory)` ingests the directory it is
  given (#227).
- Resume and caching: rerunning a pipeline id resumes (#223), switching a
  stage on runs it (#248), changing the suffix or source scope recomputes
  (#248), turning the registry off rebuilds in full (#259),
  `--force-rebuild` writes its outputs (#223), `--resume-from-stage` is
  case-insensitive (#223), and generated pipeline ids no longer collide
  (#223).
- `--continue-on-error` records crashed documents `FAILED`, not `PROCESSED`
  (#223).
- `run-ingestion --s3-sync` keeps the stages a stopped task finished, its
  checkpoints no longer expire after a day, and a SIGTERM no longer hangs it
  or keeps retrying model calls (#316, #326); the CDK tasks run under an
  init process (#433).
- Retrying a failed Step Functions execution resumes at the failed stage
  (#223).
- The summary reports stage-cache hits and restored stages (#227, #248),
  `run-ingestion` prints why it exits 1 (#248), and a failed stage's error
  is logged once (#439).

Evaluation, visualization and tuning.
- Judge failures are recorded per query, failed scoring exits non-zero
  (#158, #210), and a failed RAGAS run marks only its own metrics (#219).
- RAGAS skips `faithfulness` for an answer without context (#366), counts
  each context once per batch (#246), judges with its own metric objects
  (#399), and stops when the evaluation is cancelled (#386, #393).
- `run-eval` reports an invalid or missing dataset with exit 2 (#227,
  #230).
- `run-visualization` rejects unknown `--renderers` and invalid data with
  exit 2 (#255, #353) and lays out only the drawn nodes (#246); physics
  stays off when `interactive` omits it (#292).
- `run-prompt-tuning` profiles with the `graph_extraction` model, writes
  types UPPERCASE in its examples, logs no model output at ERROR, and
  rejects `--max-docs` below 1 (#227, #292, #353).

CLIs, configuration and library.
- A config with a wrong type, invalid YAML or a missing file is a usage
  error instead of a traceback, an empty config loads the defaults, and
  `custom_prompts` overrides are validated at load (#158, #178, #230).
- `ConfigurationError` replaces bare `ValueError`s (#297); every CLI exits 2
  on a configuration error met while running (#344).
- Index suffixes and prefixes are validated at load and on the command line
  (#367, #372, #374).
- `--output-format json` prints a status document for errors before the
  result, argparse errors included (#359, #366, #420).
- `run-rag --filters`: repeated flags add up; a pair without `:`, a blank
  key or value, a duplicate or unknown key is a usage error (#408, #411,
  #443); a `None` value is rejected (#373).
- `run-rag --interactive`: a line is a command only as a whole, `set-filter`
  keeps the value's whitespace and refuses unknown keys, and JSON output is
  rejected (#353, #413, #416, #420, #426, #428).
- `run-rag` documents conversation memory as in-process and warns on a
  single `--query` with `--use-memory` (#224).
- `GraphIndexer.delete_by_id` without a suffix acts on the default suffix
  (#344); `GraphRAGChain(providers=..., boto_session=...)` raises (#218).
- Used as a library, log records reach the host's handlers and no longer
  raise on `structlog` context fields; importing the package configures no
  logging and does not touch `sys.path` (#178, #245, #251, #255).
- `index_settings` keys all reach the indices, in either spelling (#260,
  #404).

Deployment (CDK).
- Alarms deliver notifications, the corpus no longer expires with the
  cache prefixes, multi-node domains get replicas, timed-out runs alarm, and
  `cdk destroy` leaves no fixed-name resources (#157, #166, #211).
- The failure notification renders any error and retries its publish, and
  the Gremlin-errors alarm needs 3 consecutive minutes (#219, #254).
- Dev and prod deployments in one account no longer trip each other's
  alarms (#330); Neptune alarms watch the writer (#303).
- A CDK deployment indexes Korean text (no `nori` on the domain without its
  package) (#282).
- Context values: lists, booleans, ranges and Fargate CPU/memory pairs are
  validated at synth, and a misspelled key fails naming the closest one
  (#224, #345); `max_azs=1` and an odd `opensearch_count` over 2 AZs fail at
  synth (#298); a zone-aware domain spans up to 3 AZs (#227).
- The task role gets `bedrock:CountTokens`, so deployed stacks count tokens
  exactly (#298).

### Security
- Pinned floors for advisories: pypdf 6.21.4 (seven high-severity
  denial-of-service advisories through the PDF parser, for example
  GHSA-5849-834g-hvw5), tornado 6.6 (GHSA-pcg2-g6cj-q92c,
  GHSA-r4j5-9xpp-3985, GHSA-pw9p-v7r6-gx4p), filelock 4.2.1 (nine advisories,
  GHSA-v4x8-3wr5-3wq8 high; an override lifts `unstructured`'s cap; locked
  at 4.2.5), Pillow 13.0.0 (GHSA-p4wg-p73j-c528 and others),
  `unstructured>=0.24.1` (GHSA-4mvj-m6j5-pmf7,
  GHSA-8mgp-746c-j5xp, a stored XSS GHSA-v5mq-3xhg-98m9) and
  `langchain-openai>=1.1.14` (GHSA-r7w7-9xr2-qq2r); the lock, not a
  floor, takes certifi 2026.10.23 (GHSA-6q48-6c7w-9gw2) (#307, #426, #430,
  #455).
- Document attribute filters no longer leak excluded documents: they apply
  to relationships, claims, community reports (which carried the first
  entity's attributes) and graph expansion (which walked every edge), and a
  removed document's filter values stop matching what outlives it. Rebuild
  with `indexing.reset: true` (#354, #356, #360, #370, #376, #381).
- A guardrail-blocked answer returns no sources or search results, and the
  query span records only the screened query (#345).
- IaC: queries run under a read-only role that cannot write documents
  through a `_search` path, list only the `eval/` prefix and have no
  DynamoDB access (#319, #345); `bedrock:ApplyGuardrail` is granted on the
  configured guardrail only (#319); unused Neptune, S3 delete and
  `GetInferenceProfile` actions are dropped and private-mode endpoint
  policies admit only the account (#204, #211).
- Logs carry no query text, entity names or model output at INFO, no local
  variables in tracebacks, no ANSI codes when piped, and no relationship
  endpoint names or unparseable-output previews from gleaning; a unit test
  fails on a new INFO+ log call that passes them (#156, #173, #245, #255).
- CLI output escapes the data it prints, and `run-rag` strips terminal
  control characters (#173, #408).
- Prompts delimit corpus and retrieved text as data that cannot close its
  delimiter tags (#179, #182).
- Importing the package no longer loads a `.env` file; the CLIs read only
  `./.env` and never override the environment (#224).
- ragas sends no usage telemetry unless `RAGAS_DO_NOT_TRACK=false` (#361).
- `pipeline_id` must be a single path segment; metadata filters are
  documented as relevance filters, not access control (#173).
- Interactive graph tooltips are escaped (#180).
- The compose containers drop all capabilities and run with
  `no-new-privileges`, the data-plane image declares `HEALTHCHECK NONE`, and
  retry jitter draws from `random.SystemRandom` (#307).
