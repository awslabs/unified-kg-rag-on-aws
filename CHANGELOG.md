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
- An optional extraction schema: declared entity and relationship types,
  patterns and ontology IRIs, inline or from a schema file (a pluggable
  source), applied as `align` (one label per type, everything kept) or
  `strict` (off-schema items dropped), stored as `in_schema` / `type_iri` /
  `predicate_iri` filter keys, and drafted from a corpus by
  `run-prompt-tuning --schema-out`; the defaults change nothing (#475-#477,
  #478).
- Seed graphs: an existing knowledge graph (JSONL, CSV or an RDF A-Box)
  merged into the extracted graph as virtual documents of the corpus.
  Documents decide types, names, confidence and ranks; a seed adds
  description lines, properties, IRIs and edges, and incremental runs,
  removal and resets treat it like any file (#497-#505).
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
    `langchain-openai`. The `ragas` evaluator stays in the default
    `evaluation.enabled_evaluators` (`langchain` is opt-in, see
    *Defaults*); without the extra,
    `EvaluationManager` (and `run-eval`, exit 2, before the first query)
    raises `ConfigurationError` naming it, also under
    `processing.ignore_errors`. The LLM-free evaluators need no extra; the
    container image installs `eval`.
  - `viz` holds `pyvis`, for the `interactive` renderer. Without it a
    `run-visualization` that requests `interactive` exits 2 naming the extra
    and `resolve_renderers` raises `ConfigurationError`; the `static` renderer and
    `from unified_kg_rag import BaseRenderer, register_renderer` work on a
    core install. `umap-learn` and `scikit-learn` are no longer
    dependencies, and neither are `bokeh` and what only it pulled in
    (`contourpy`, `tornado`, `xyzservices`) (#576).
  - The security floors follow the packages that pull the pinned package in:
    `pillow` in `eval`, `llvmlite` in `unstructured`, `langchain-openai` in
    `eval` and `local`.
- `graspologic` is no longer a dependency: Leiden runs on
  `graspologic-native` directly (#225, #226).
- The `static` renderer writes its three charts (`degree_distribution.html`,
  `centrality_comparison.html`, `community_size_distribution.html`, names
  unchanged) as self-contained HTML pages with an inline SVG instead of
  Bokeh documents: no script, a tooltip on every bar, no pan or zoom. A
  tool that read the Bokeh JSON embedded in them finds none (#576).

Configuration loading.
- Unknown config keys, in the file or as `UKR__` environment variables, are
  an error instead of a warning. Every CLI takes `--allow-unknown-keys` to
  warn and go on; `get_config` and `ConfigLoader` take
  `allow_unknown_keys=True` (#335).
- A moved key fails the load naming the key that replaces it (the *Moved
  keys* table of `docs/reference/configuration.md`); a removed key loads with a
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
  `unified_kg_rag.adapters.renderers` (`resolve_renderers`,
  `render_resolved`, `RenderContext`). The
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
- Settings whose other values were duplicates, never shown, or kept only to
  reproduce 0.2.3 are gone; the behaviour is the former default. The keys
  load with a WARNING and are ignored (#577):

  | Removed key | Replacement |
  |---|---|
  | `search.global_search.reduce_with_llm`, `search.global_search.map_reduce_model_id`, `models.roles.global_reduce` | none: the ranked key points go to the answer model as one context section, with no reduce call |
  | `search.fusion.method` | none: results are fused by reciprocal rank fusion (`rrf`); weighted fusion is gone |
  | `search.reranking.enabled` | `providers.rerank.kind: none` turns reranking off; the key is ignored, so `enabled: false` alone now reranks |
  | `graph.visualization.centrality.calculate_closeness`, `calculate_eigenvector`, `eigenvector_max_iter`, `eigenvector_tol` | none: the visualization computes the degree, betweenness and PageRank centrality it displays |
  | `graph.analysis.statistics.calculate_density`, `calculate_clustering`, `calculate_components` | none: always computed (`calculate_diameter` stays) |
  | `fixing` (`fixing.enabled`) | none: output fixing is always on; `models.roles.fixing` picks its model |
  | `evaluation.warm_up_query` | none: `run-eval` always answers one untimed warm-up query per strategy and run |
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
| `indexing.max_deletion_share` | (none: no guard) | 0.5: an incremental run deleting more of its scope asks first; `null` or 1.0 turns it off (#518) |
| `graph.community_detection.auto_resolution` | `true` | `false` (#141) |
| `graph.visualization.outputs_directory` | `outputs/visualization` | `null` (`<cache directory>/<pipeline_id>/visualization`) |
| `search.token_manager.max_context_tokens` | 200000 | 30000; 22000 under `backend: local` with a non-Bedrock LLM (#125, #224, #226) |
| `search.fusion.fusion_weights` | 15 entries of 1.0 | `{}` (every bucket 1.0) (#258) |
| `search.global_search.relevance_threshold` | 0.5 | 0.2 (#318) |
| `search.global_search.use_dynamic_selection` | `true` | `false` (#318) |
| `search.auto_routable_strategies` | (fixed: `simple`, `local`, `global`, `drift`) | `local`, `mix`, `global`; add `drift` to route to it (#347) |
| `search.query_timeout_seconds` | (none: no deadline) | 180 s; `null` turns it off (#522) |
| `search.model_timeout_seconds` / `search.model_max_attempts` | (fixed: the ingestion client's 300 s read timeout, 6 requests) | 90 s / 2 for query-time model calls (#522) |
| `search.auto_speculative_extraction` | (fixed: after the router) | `true`: `auto` extracts alongside the router call (#525) |
| `search.query_extraction_cache_size` | (none: no cache) | 1024; 0 turns it off (#525) |
| `search.output_ceilings` (`strategy_selection`, `query_extraction`, `community_relevance`) | (fixed: the model's maximum output) | 4096 each; `null` asks for `aws.bedrock.default_max_output_tokens` (#525) |
| `search.answer_prompt_style` | (fixed: every strategy concise) | `{global: comprehensive}`, merged under a configured map; `{global: concise}` keeps the concise global answer (#533) |
| `search.global_search.evidence_tokens` | (none: as many cited reports as the context budget holds) | 4000; `null` turns the cap off (#533) |
| `evaluation.enabled_evaluators` | `langchain`, `ragas` | `ragas`, `answer_match`, `retrieval`, `graph_aware`: the LLM-free evaluators are added (#121) and `langchain` is opt-in, as its `correctness` and `partial_correctness` repeat `ragas`' `answer_correctness` on every query (#576) |
| `evaluation.max_context_tokens` | 8192 | `null` (the answer budget) (#315) |
| `evaluation.answer_span` | (fixed: the whole answer, `full`) | `first_line` (#516) |
| `run-eval` warm-up query | (none: no warm-up) | one extra, unscored query per strategy and run, not counted in the run's usage or USD estimate (#529); not configurable (#577) |
| `models.roles.evaluation` (the `run-eval` LLM judges) | the default tier's model (the answer model) | `anthropic.claude-opus-5-5` on Bedrock (about twice Sonnet 5.5's price per token); `default` keeps the answer model as judge (#533) |
| `logging.log_to_file` | `true` | `false` (#520) |
| `indexing.opensearch.language_analyzers` | `en: english`, `ko: nori` (every other language `standard`) | one per language: also `de: german`, `es: spanish`, `fr: french`, `it: italian`, `pt: portuguese`, `ru: russian` and `ja`/`zh: cjk` (character bigrams, no plugin), so a word inside a Japanese or Chinese run matches lexically (#574) |

Behaviour.
- Entity resolution merges only names equal after normalization, legal-form
  removal and dense-script spacing; entity and relationship `rank` come from
  node degree, and relationship weight sums the strengths per text unit
  (#189, #214). Legal-form removal and name grounding read one table
  (`shared/utils/legal_forms.py`): trailing Inc/Incorporated, Corp/
  Corporation, Co/Company, Ltd/Limited, LLC, LLP, LP, PLC, Pty, Pte, GmbH,
  AG, SA, SAS, SARL, SRL, S.p.A., Ltda, BV, NV and KK in any dotting
  ("L.L.C."), and the Korean, Japanese and Chinese company and juridical
  person forms leading or trailing ("(주)", "유한책임회사", "(사)",
  "재단법인", "合同会社", "株式會社", "股份有限公司", "有限责任公司"), so
  "Acme GmbH" and "Acme" are one entity of a type (an entity of another type
  stays apart). Not after "&"/"and" ("Smith & Co") nor after an article
  alone ("The Company") (#574).
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
  `providers=`. Only `Providers` and `build_providers` take `boto_session`
  (pass `providers=Providers(config, boto_session=...)` to
  `DataIngestionPipeline` and `GraphRAGChain`, which no
  longer take it) (#577); `GraphRAGChain` rejects unknown
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
- A custom doc-status registry also implements `get_annotation_digest` /
  `put_annotation_digest`: one record per index namespace of which schema
  annotations its stores hold (a digest, or the rewrite sentinel), kept in
  the namespace's partition and never listed as a document. The DynamoDB
  and SQLite registries implement them (#488). A custom backend that
  rewrites schema annotations may override `holds_any_item(suffix)` on
  `GraphIndexer` / `VectorIndexer` with a bounded read; the default lists
  the items through `read_schema_annotations` (#490). A custom vector
  backend may override `VectorIndexer.holds_any_text_unit(suffix)` too;
  the default (`None`, cannot tell) makes a registry-off run of seeds
  alone ask before it builds (#514).
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
  of `CacheManager` (#471); with the settings above, `MapReduceSummaryPrompt`,
  `ModelRole.GLOBAL_REDUCE`, `FusionMethod`, `FixingConfig`,
  `HybridScorer._weighted_fusion` / `_normalize_scores` and
  `CentralityMetrics.closeness` / `eigenvector` (#577).

### Upgrading
From 0.2.3 (or a `main` checkout since):
1. Install on Python 3.11 or 3.12, with the extras you use (`eval`, `viz`,
   `local`, `otel`, `unstructured`, `ontology`). Without the `local` extra a CLI whose
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
   both; a deployed stack deploys the orchestration stack first, item 14).
   Run queries with `QueryTaskDefinitionArn`. Korean on the CDK
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
9. The first incremental run of each existing namespace reads its stores
   once to record which schema annotations they hold, rewriting only those
   that differ, with no model or embedding call; later runs read nothing
   unless the schema's annotations change (#488).
10. A custom `search.fusion.fusion_weights.lightrag_linked_chunks` weight
    now weighs only the chunks the matched entities cite; the chunks the
    matched relationships cite are their own stream at
    `lightrag_relation_chunks`, 1.0 unless set. Set it as well (usually to
    the same value) to keep weighting every LightRAG graph chunk: mix and
    hybrid recall is sensitive to it (#521, #530).
11. From a `main` checkout: stored communities keep their positional names
    (`Level 0 Community 3`) until the next reset, which renames them and,
    as their report answers were stored under the old names, asks every
    report once more. No setting changes; later resets reuse the answers
    (#539).
12. From a `main` checkout: an index built before one of these fixes needs
    one run with `indexing.reset: true`:
    - Relationships and claims carry their documents' filters (#360): an
      older index returns none of them under an attribute filter.
    - A removal clears its document's filter values (#370): an index that
      saw removals before it keeps the removed documents' values on items
      not rewritten since, so its filters still match them. An index an
      incremental run wrote between #370 and #376, with a removal that
      pruned a community, lost the filters of the text units it kept, which
      then match no filter.
    - `attr_*` values are keywords (#446): an `attr_*` field an older index
      mapped as a date stays one.
    - Numeric and boolean document filter values are keywords (#566): an
      `attr_<key>` or `attributes.filters.<key>` field an older OpenSearch
      index mapped by a first number (`long`, `float`) or boolean keeps that
      type, so a later string value fails its document's write and a range
      compares numbers; a warning names each such field. The first run maps
      the fields it adds from then on as keywords without a reset; the
      reset costs what the #562 one below does (stored extraction records
      and vectors reused). The embedded SQLite store needs no reset.
    - Seed properties are stored unmapped (#508): an entities index that
      maps `attributes.seed_properties` field by field is refused (exit 2).
    - Relationships store their type (#562): a relationship an older build
      wrote reads `Relationship: A -> B` until a run rewrites it (an
      incremental run rewrites only the relationships it changes). The
      reset reuses the stored extraction records and vectors, so it asks
      the model only for the community reports it clusters differently.
    - Removals are written ahead (#383) and resumed resets converge (#410):
      in an index an older build wrote, a deleted document restored while
      its removal was unfinished, or a reset that dropped documents and was
      interrupted after it swapped an index in, makes a run fail with `The
      doc-status registry lists indexed documents of this namespace whose
      entities the graph store does not hold`. For the restored document,
      deleting that one document's registry record is enough: the next run
      indexes it again.
13. From a `main` checkout, without a rebuild: the default
    `processing.chunking.length_unit: chars` keeps the chunks, and so the
    ids, an index built before the setting holds; `tokens` re-chunks and
    re-extracts the corpus (#460). `run-eval` results saved before usage
    accounting have no usage record or latency: `--from-results` counts them
    in `answers_without_usage` / `answers_without_latency`, and a `--fail-if`
    on a `usage/*` or `latency/*` measure fails for them, so run them again
    to compare cost (#526, #529).
14. A deployed CDK stack (0.2.3 or a `main` checkout): the ingestion task
    definition moved from `GraphRagCompute` to `GraphRagOrchestration`. Run
    `cdk deploy GraphRagOrchestration --exclusively` once
    (`GraphRag<Env>Orchestration`, e.g. `GraphRagProdOrchestration`, for an
    `env_name` other than `dev`), with the deploy's usual `-c` flags (the
    new `image_tag` included), then `cdk deploy --all`. The first deploy creates the task definition in the
    orchestration stack and points the state machine at it, so the compute
    stack's old one is no longer imported and the second deploy removes it.
    A plain `cdk deploy --all` stops at `GraphRagCompute` (`Cannot delete
    export ... as it is in use by GraphRagOrchestration`) and rolls that
    stack back with nothing changed. The ingestion task's family is renamed
    (`<stack prefix>OrchestrationIngestionTask...`), so its Container
    Insights series start again under the new name; the alarms and the
    dashboard follow it (#558).
15. `run-eval` results saved before #563 scored `graph_aware`'s
    `entity_coverage` and `retrieved_*_coverage` by the earlier rule
    (phrase containment, duplicates counted twice), and `--from-results` /
    `--baseline-results` reuse saved scores. To compare those metrics with
    a new run, run the baseline again: one answer and score per query and
    repeat (#563).
15. Incremental runs from a `main` checkout may have stored a summary of a
    summary for an entity or relationship that an edited document shared
    (#564). It is still a summary of what its chunks state; to summarize
    such items from their lines, run once with `indexing.reset: true` (one
    summarization call per such item over the threshold; the other stored
    summaries are reused), or leave them until a run edits or removes one
    of their documents.
15. Ingestion fixes of #PR:
    - Translation (on when `target_language` differs from
      `source_language` or `additional_target_languages` is set): its prompt
      changed, so a run that translates again (a reset, `--force-rebuild`)
      asks the model for each unit's translation once more, and, as a new
      translation is new text, extracts that unit again. An incremental run
      translates only new and changed documents, as before. The description
      summary prompt changed too: stored summaries are kept, only new ones
      are asked with it.
    - A merged entity whose winning name had no type is typed only when
      graph resolution runs on it again: a reset (it reuses the stored
      extraction records, no extraction call) or an incremental run that
      touches it.
    - Gleaning (on by default): its prompt changed, which re-keys every
      gleaned extraction record (the claims records too while claims are
      on), as any template edit does. A reset therefore extracts and gleans
      every text unit once more (and extracts its claims), the cost of a
      first build's extraction and gleaning; later resets reuse the new
      records. An incremental run keeps the stored documents' records and
      uses the new prompt for new and changed documents only.
    - The `intelligent` chunker: a pre-chunk of one line was cut as the
      model's answer left it (whole, or by size after an answer naming
      lines that do not exist); it is kept whole up to `max_chunk_size`
      and cut by size over it now, with no call. A reset re-chunks a
      document holding one and extracts its text units again where the
      pieces differ; pre-chunks of several lines are unchanged.
15. A deployed private-mode CDK stack whose image config sets
    `aws.bedrock.assumed_role_arn`: deploy with `-c sts_endpoint=true`, or
    the next deploy removes the STS endpoint and the role assumption hangs
    (#566).
16. A deployed CDK stack, on the next `cdk deploy --all` (#566):
    - The `owner` tag is removed from every resource (an in-place tag
      update); pass `-c owner=aws-proserve` to keep it.
    - Without `use_cmk`, `GraphRagSecurity` is no longer part of the app.
      If an earlier deploy created it (it holds only CDK metadata), delete
      it with `aws cloudformation delete-stack --stack-name
      GraphRagSecurity` (`GraphRag<Env>Security` outside dev); no other
      stack refers to it.
    - The OpenSearch domain's TLS policy changes to
      `Policy-Min-TLS-1-2-PFS-2023-10`. Clients older than TLS 1.2 or
      without an ECDHE cipher can no longer connect (the package's client
      can). To see whether the change runs as a blue/green deployment,
      dry-run it first: `aws opensearch update-domain-config --domain-name
      <domain> --domain-endpoint-options
      TLSSecurityPolicy=Policy-Min-TLS-1-2-PFS-2023-10 --dry-run`.
17. From a `main` checkout, after #574 (no setting changes):
    - A namespace whose `processing.translation` languages include `de`,
      `es`, `fr`, `it`, `ja`, `pt`, `ru` or `zh`, on OpenSearch: its text
      fields keep `standard` while the default analyzer of those languages
      changed, so the next incremental run stops before any model call (exit
      2, `was built for other language settings: '<field>' uses 'standard'
      (needs 'cjk')`). Run once with `indexing.reset: true`: it rebuilds the
      indices with the stored extraction records and vectors (analyzers
      shape no record, cache key or embedding), so it asks the model only for
      the reports of communities it clusters differently. To keep `standard`
      instead, set `language_analyzers: {en: english, ko: nori}`. The
      embedded engine analyzes the new names as before and needs no reset.
    - Every graph store: the entity `name_key` (Neptune, embedded SQLite)
      follows the larger legal-form table, and an incremental run finds the
      stored entities to merge with by it, so a stored "Acme GmbH" (key
      `acme gmbh`) is not found for a new "Acme". Run once with
      `indexing.reset: true` (a new or no `--pipeline-id`, so resolution runs
      again) to merge the designator variants and rewrite the keys. No
      extraction call: the extraction records are keyed by text and
      extraction settings, not the resolver. It asks for one description
      summary per newly merged entity over the summarization threshold and
      embeds those descriptions, and for the reports of the communities it
      clusters differently. Without a reset, entities stored apart stay apart.
17. `run-eval` no longer runs the `langchain` judge unless
    `evaluation.enabled_evaluators` lists it, so a default run has no
    `langchain/correctness` or `langchain/partial_correctness` scores and a
    `--fail-if langchain/...` gate fails for want of the measure. To keep
    it, list every evaluator you want, e.g. `enabled_evaluators: [langchain,
    ragas, answer_match, retrieval, graph_aware]`; leaving it out saves its
    two judge calls per query (#576).
18. `run-visualization --renderers static` no longer needs the `viz`
    extra, which no longer installs Bokeh: re-render the static charts to
    get the SVG pages (nothing stored changes). A script that imported
    `bokeh` through the extra installs it itself (#576).
17. A config that sets one of the settings removed in #577 loads with a
    WARNING naming its replacement; nothing is re-extracted or re-embedded.
    Review it where the value was not the default: `search.reranking.enabled:
    false` reranks again (a Bedrock Rerank call per query) until you set
    `providers.rerank.kind: none`; `search.fusion.method: weighted` fuses by
    RRF, which orders results differently; `reduce_with_llm: true` saves its
    reduce call per global query; `fixing.enabled: false` asks the fixing
    model to repair malformed output instead of failing the item; and
    `evaluation.warm_up_query: false` answers one more, unscored query per
    strategy and run. A run resumed from a `main` checkout across the change
    finds no stage-cache entry (`fixing.enabled` keyed every stage), as
    after any upgrade; the LLM response cache still serves its model calls.

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
- Generated references: `docs/reference/configuration.md` (#337) and `docs/reference/cli.md`
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
- IaC: alarms on OpenSearch automated-snapshot failures, blocked index
  writes, write and search thread-pool rejections, a yellow multi-node
  cluster and (with `use_cmk`) an unusable KMS key; on the ingestion task's
  ephemeral storage; `DocumentsNotIndexed` on the run's parse failures and
  failed documents, with a dashboard widget that also shows the skipped
  files; and `QueryLatencyHigh` on the p90 latency of
  `run-rag` queries, which `run-rag --metrics-sink cloudwatch` now emits
  (`query_latency_ms`, per environment and strategy;
  `query_latency_alarm_seconds`). `neptune_maintenance_window`,
  `neptune_backup_window` and `opensearch_off_peak_start` set the stores'
  maintenance windows, and `observability_secret_arn` /
  `observability_secret_keys` inject Langfuse or OTLP credentials from
  Secrets Manager as task secrets (in `private` mode through a Secrets
  Manager interface endpoint the stack adds). The runbook has what to do
  for each pipeline-run alarm, capacity sizing and the Neptune
  major-version upgrade (#519, #528).
- Extraction schema settings under `processing.graph_extraction`:
  `entity_types` items may be `{label, description, iri}` mappings, and the
  new `relationship_types`, `relationship_patterns` (`"PERSON -WORKS_FOR->
  ORGANIZATION"`), `schema_mode` (`off`/`align`/`strict`, default `align`)
  and `schema_source` keys are validated on load: labels are unique by
  case and word separators, and patterns name declared types. At their
  defaults they change no cache key, extraction record or recorded
  extraction settings (#475). They key extraction by what the model sees:
  a type's label and description, the patterns, and `schema_mode` where it
  changes extraction (`strict`, or `off` once relationship types are
  declared). An IRI, `off` against `align` with entity types alone, or a
  string item rewritten as the same mapping re-extracts nothing (#482).
- Schema sources: `schema_source.kind: file` reads the schema keys from a
  local YAML or JSON file at start-up, keyed by its content, not its path,
  and `@register_schema_source` (exported from `unified_kg_rag` with
  `SchemaSourceContext`, `SchemaDocument` and `SchemaSourceError`) adds a
  source for another format or service. A loader gets the source settings
  (`path`, `options` for its own settings and `secret_arn`, neither keyed),
  the run's config and its AWS session, built only when asked for; logs
  and errors show a URL path without userinfo or query. Ingestion,
  evaluation and prompt tuning load the source, and reading the schema of
  a config whose source is not loaded raises `ConfigurationError` instead
  of using the default schema (#475, #480).
- The `ontology` extra (`uv sync --extra ontology`: rdflib 7.6,
  BSD-3-Clause) and the mapping the ontology schema sources share: selected
  named classes become entity types and object properties relationship
  types (with their IRIs), patterns come from a fixed RDFS-style closure of
  the domains and ranges (sub-classes and sub-properties, unions, inverses,
  symmetric properties) on a flattened hierarchy, labels from
  `skos:prefLabel`/`rdfs:label` by language preference or the local name
  (`UPPER_SNAKE`, Hangul kept), and the output depends only on the
  statements. A symmetric property with one end stated takes it for the
  other, and its sub-properties and inverses inherit both. The closure is
  one pass over the hierarchy, so a 20,000-deep `subClassOf` chain maps in
  under a second and 150 MB instead of 9 GiB, and a 2,000-property
  `subPropertyOf` chain in 0.1 s instead of about 40 s. Colliding labels
  and more types or patterns than the caps allow fail instead of being
  truncated; `on_collision: qualify` takes the suffix from
  `options.prefixes`, then the well-known prefixes, before the files' own
  bindings, and a `description_properties` entry other than the defaults
  that the ontology never uses fails the load (#491, #496).
- `schema_source.kind: owl` fills the schema from an OWL or RDFS ontology:
  a local file or a directory of Turtle, RDF/XML, JSON-LD, N-Triples,
  N-Quads or TriG files, read with the `ontology` extra and mapped by the
  options under `schema_source.options` (selection, labels, caps, and
  `format`/`max_bytes`). Nothing is fetched: `owl:imports` is not followed
  (an import no file declares is logged once), a JSON-LD context naming a
  document is refused, XML external entities are not resolved, and
  OWL/XML, functional and Manchester syntax, OBO and Notation3 are refused
  with how to convert them. Shapes the RDF/XML and N-Triples parsers take
  quadratic time over (entities expanding past the amplification limit, an
  element of more than 10,000 text pieces, more than 1,000 elements in XML
  literals, a line over 256 KiB) are refused before parsing, naming the
  limit, and so are an IRI with whitespace and a cyclic `rdf:List`; a
  parse error names the line, never the ontology's text. schema.org data
  types (`schema:Text`, `URL`, `Number`, ...) are literal ranges, not
  entity types; a class's `owl:intersectionOf` superclass or definition
  and a covering `owl:unionOf` place it in the hierarchy; sub-properties
  of an `rdf:Property` are read, and properties the reader cannot type
  are logged; a class named only in a datatype property's domain is no
  entity type. A directory skips an OASIS XML catalog (Protégé's
  `catalog-v001.xml`) and, with `options.format`, reads every file.
  Re-serialising or moving the ontology keys
  nothing when its IRIs are absolute or it declares `@base`; an IRI-only
  edit re-annotates without re-extracting.
  `register_schema_source(kind, requires=(module, install))` declares a
  source's optional dependency, which the CLIs check before any model
  call (exit 2 naming the extra), and the `run-prompt-tuning --schema-out`
  draft over such a source says which relationship types it lacks
  (#492, #495).
  draft over such a source freezes it: the types the source declares keep
  only its patterns (none under `patterns: none`), so a draft that adds
  nothing loads as the same schema, and the header names the relationship
  types it adds and counts their patterns (#492, #496).
- Schema sources in S3: `schema_source.path` of kinds `file` and `owl` may
  be an `s3://bucket/key` object, and for `owl` an `s3://bucket/prefix/`
  read as a directory (every page of the listing, filtered, sorted and
  refused as a local directory is). Objects are read with the run's AWS
  session; the size is checked with HeadObject (or the listing) before any
  byte is fetched (`file`: 16 MiB, `owl`: `options.max_bytes`), and a
  missing object, a denied access (naming `s3:GetObject`/`s3:ListBucket`),
  an empty prefix or an oversized object fails the load with what to fix.
  `run-rag` loads the schema on the first query with a schema filter key
  and, when the load fails in any way, skips that check once instead of
  reloading on every query (#495).
  The content keys the caches, so moving the schema between a local path
  and S3 re-extracts nothing; relative IRIs of an ontology in S3 resolve
  against its `s3://` URL (declare `@base`). Other URL schemes stay
  refused (#493).
- Docs: ontology schema sources in user guide §9 C and the runbook, with
  their load cost and what a re-serialisation or a new equivalent class
  can change (#494, #496).
- Extraction applies the schema. Declared relationship types and patterns
  replace the example relationship types in the extraction prompt. Once
  relationship types are declared (or under `strict`), a type naming a
  declared one is written as declared before ids are built, so its
  spellings give one edge label and id. `schema_mode: strict` drops, per
  text unit, entities of an empty or undeclared type with the unit's
  relationships naming them (rejected as `off_schema`), relationships of
  an undeclared type, and relationships a pattern does not allow between
  two endpoints of declared types (never flipped); gleaning follows the
  same rules. `align` keeps everything and counts. New counters in the
  graph extraction stage's `extraction_stats` and the gleaning stage's
  `gleaning_stats` (gleaning counts its off-schema additions under `align`
  too; an undeclared relationship type counts whatever its endpoints, so
  `relationships_off_schema` is the same in both modes); drops are not
  failures. With the defaults, or entity types alone, every cache key,
  extraction record and prompt is unchanged (#476, #481).
- Schema annotations in the stores: indexing writes `in_schema` and
  `type_iri` on entities and `in_schema` and `predicate_iri` on
  relationships (Neptune and SQLite graph properties, OpenSearch and
  embedded vector documents), from the merged type and the schema's
  `iri`s, so a full build and an incremental run agree. They are filter
  keys on the entities, relationships and graph stores (`filters:
  {in_schema: false}` returns the off-schema items `align` kept). A
  declared `iri` is written whenever the mode is not `off`, with entity
  types alone too (`type_iri` then comes without `in_schema`). Where
  `in_schema` is written, both kinds get it: every item of a kind with no
  declared type is in schema (`strict` with entity types alone no longer
  leaves relationships without it, which emptied them under `in_schema:
  true`). Under `off`, and with entity types alone and no `iri`, nothing is
  written and the stored documents and properties are unchanged; the
  entities and relationships mappings gain the fields, and an incremental
  run adds them to its live indices before its first write. A filter on an annotation the configured schema writes on
  no item raises `InvalidFilterError` before any model call, and one on an
  OpenSearch index built before the fields were mapped matches nothing
  instead of failing every vector and hybrid query (#477, #478, #481). A
  schema edit that changes only the annotations (an IRI, `off` against
  `align` with entity types alone) is applied by the next incremental run
  without a model call: it rewrites the annotations of every stored entity
  and relationship of the index suffix, every scope's, from their stored
  types and logs the counts, so the stores hold what a full build writes
  (#482). The registry records what the stores hold once per index
  namespace, so the rewrite runs once per namespace, a schema reverted on
  one scope after another scope's run is rewritten too, and a new scope
  reads nothing. A rewrite that stops (a kill, a failed write, a store
  that cannot be read, which now warns instead of failing the run) is made
  again by the next run under any schema; Neptune reads the annotations in
  bounded batches; a reset records the annotations it wrote (#488). A
  namespace whose registry holds none of its documents (a lost or replaced
  registry) records its digest without reading only when the stores hold
  no item of the suffix either, and the pipeline's own indexers no longer
  hide that a store cannot list its items (#490).
- `run-prompt-tuning --schema-out PATH` writes an extraction schema draft
  for `schema_source: {kind: file, path: PATH}`: the profile's entity types,
  the relationship types the tuning examples state most often (with their
  counts) and patterns only where every endpoint-type pair of a type is
  seen at least twice. The examples are extracted with the profile's
  entity types, or with the configured schema when the config declares
  one (a schema source it cannot load exits 2). Every type and pattern the
  config declares stays in the draft with its description and IRI, and a
  relationship type with configured patterns gets no other. It names
  `schema_mode: align` (the comment's path relative to the config file's
  directory), and a fixed model gives the same file. It never replaces a
  file silently: `--schema-out` or `--output-path` naming the
  `--config-path` file or the schema file the config loads exits 2, even
  with `--force`, and so does either naming an existing file unless
  `--force` is given (#478, #480, #483, #490).
- Seed graphs under `processing.seed_graphs`: each item names a registered
  source (`kind`), a `path` (local, or an `s3://` object or prefix), its
  `filters`, `on_name_collision` (`error`/`merge`/`qualify`),
  `description_properties`, `confidence`, `shards` and caps that fail the
  import instead of truncating (100,000 nodes, 500,000 edges, 256 MiB, or
  32 MiB for `rdf`, whose parser holds up to about 150 times the file
  size, about 5 GB, before the node cap is checked, 64 property keys; a name of at most
  1,000 characters, an IRI 4,000 and a type 256, so a store indexes each
  as a keyword; no lone surrogate). Conversion is linear in the seed (the
  100,000-node, 500,000-edge caps convert in about 40 s); a shard over
  2,000 text units fails naming its largest node, and `qualify` checks
  each homonym against the schema patterns by its own type.
  `run-ingestion` loads each seed once at start-up against
  the resolved extraction schema, before any model is built (a seed that
  cannot be imported, or whose kind's extra is missing, exits 2, naming
  the file and line; the other CLIs load no seed and need no seed kind's
  extra), checks it as a
  whole under `schema_mode`, shards it by a stable hash of the node ids
  and imports the shards as documents of the run's source scope: no
  extraction call reads them, the registry diffs them, an edit re-imports
  only its shards, a model-settings change none, and an empty corpus is a
  seed-only run for a scope the registry holds no document of (for one
  that does, under any spelling of its source, the run fails before it
  writes rather than delete them, and a scope's first run warns when
  another scope already imports the same seed); with the registry off,
  seeds alone replace an index that holds items (or only text units)
  only when the run is confirmed (`--confirm-reset`,
  `UKR_CONFIRM_RESET=1` or `--yes`), and build an empty one unasked. What a seed converts to keys every stage, where it is
  read from keys nothing, and no document re-extracts for it. Merges are
  document-first: a seed fills a type, name or confidence only where no
  document states one, its description lines follow the documents', its
  properties sit under `attributes.seed_properties`. A community only
  seeds evidence gets no report (`communities_seed_only`); on an
  incremental run its stored members count too, so a seed added beside
  unchanged documents reports their communities as a full build does,
  from the members' stored descriptions merged with the seed's as the
  run indexes them (an edited seed's earlier lines left out, a
  description over the summarization threshold summarized once for the
  report and the indexing, a seed edge's attribute-less endpoint read
  with its stored type); when those members cannot be read, the
  communities count in `reports_failed`.
  Seed descriptions reach the description summarizer, the reports of
  communities documents also state and the answer context. Loading,
  chunking and extraction report `seed_documents`, `seed_units`,
  `seed_entities` and `seed_relationships` (#498, #499, #500, #502,
  #503, #509, #510, #513, #514, #515).
- Seed sources: `jsonl` and `csv` read local files or S3, sized before
  any byte is fetched, and refuse a duplicate node id, a dangling edge
  (unless `options.dangling: skip`), a non-scalar property, an
  over-long line or one nested too deeply to decode, naming the file and
  line; the size hint names `processing.seed_graphs[].max_bytes` (#500,
  #510). `rdf` (the `ontology`
  extra) reads an A-Box with the `owl` schema source's readers and
  checks: instances become nodes typed by the most specific class the
  schema maps, object triples edges, literals the name, description and
  CURIE-keyed properties by `languages`; blank nodes, reification,
  RDF-star, `owl:sameAs` and non-instance objects are counted and
  skipped, nothing is inferred; an invalid node or edge fails naming its
  subject (#504, #510). A CURIE of the `rdf` options resolves as the
  `owl` source's do: a prefix the files bind to another namespace than
  the well-known one (`schema:` to `http://schema.org/`) fails naming the
  option until `options.prefixes` settles it, where it took the
  well-known namespace and names fell back to the local name; the
  default `name_properties` and `description_properties` are full IRIs
  (#579). Schema and seed sources share
  `adapters/source_files`, the RDF vocabulary, naming helpers, CURIE
  resolution and option types in `rdf_terms` included (#497, #510, #579).
- `Entity.iri` (a stated IRI, the smallest when several merge) and
  `source_kinds` on entities and relationships (`["document",
  "seed_graph"]` or `["seed_graph"]`, written only on items a seed
  states), stored in Neptune, OpenSearch and the embedded stores and
  added to live indices by the incremental preflight. Both are filter
  keys: `source_kinds: document` keeps what documents state, items
  without the field included, `seed_graph` what a seed states; `iri` on
  entities and entity vertices. A store without seeds holds exactly what
  it held (#499, #501). Seed properties are stored but not indexed (the
  OpenSearch entities index leaves `attributes.seed_properties` unmapped):
  no key or value of a seed fails an entity's write and none is a filter
  key, a removed seed takes its properties out of both stores, and an
  incremental run refuses an entities index an earlier build mapped them
  in, until `indexing.reset: true` (#508).
- Seed facts in retrieval: a seed's text units are stored but left out of
  every text search and chunk quota, so seed statements reach answers as
  entities and relationships only, and a reported source a seed states
  carries `provenance` (`kind`, `seed_graphs`, `iri`) with no chunk; a
  source without seed evidence is reported as before. `local`, `drift`
  and `auto` name the seed graphs of the entities the index matched, and
  `source_kinds` is a list in every store (#505, #510).
- Public API: `register_seed_source`, `SeedSourceContext`,
  `SeedSourceOptions`, `SeedGraph`, `SeedNode`, `SeedEdge`,
  `SeedGraphError` and `ConfirmationDeclinedError` (#498, #500, #510,
  #513).
- Docs: seed graphs in user guide §9 D, the runbook and design doc §4.5
  and §15 (#506).
- Evaluation: `evaluation.answer_span: extracted` scores exact match and
  token F1 on the short span a fast model (`models.roles.answer_span`,
  prompt `custom_prompts.answer_span_extraction_*`) copies out of the
  answer, cached per question and answer and falling back to the first
  line (`answer_span_statistics` counts the fallbacks);
  `evaluation.answer_normalization: hotpotqa` adds HotpotQA's yes/no rule;
  `evaluation.pairwise_overall` adds LightRAG's `overall` verdict; a
  comparison reports `deltas_by_category`, and the `run-eval` JSON runs
  carry `abstention_statistics` (#516).
- `indexing.max_deletion_share` (default `0.5`): an incremental run that
  would delete more than this share of its source scope's documents (a
  scope of 10 or more; retired scopes and seed graph shards not counted, so
  removing a seed graph never asks and its shards do not dilute the share:
  11 of 12 documents gone beside 16 shards asks) stops at
  `document_loading` before it writes unless confirmed:
  `--confirm-deletions`, `UKR_CONFIRM_DELETIONS=1`, `--yes`, the terminal
  prompt, or `DataIngestionPipeline(confirm_deletions=...)` (#518, #531).
- EMF metrics for corpus coverage: `total_files_failed`,
  `total_files_skipped`, `total_documents_excluded`,
  `total_documents_deleted`, `total_documents_failed`,
  `total_documents_retrying` and `total_documents_without_claims` (#518).
- `run-ingestion`'s JSON `units_failed` and the EMF metric
  `total_units_failed`: the text units a stage of the run failed on, which
  are indexed without that stage's output. A full build (no doc-status
  registry) records no document `FAILED`, so its `failed_documents` read 0
  with 138 of 3,318 units failed; the run now ends with a WARNING naming
  the count, each stage, how many failed for an error of their own, and
  what retries them (the next run, or a re-run, extracts them again) (#532).
- `search.lightrag_search.kg_chunk_pick_method` (`weight`, the default, or
  `vector`): `vector` picks the chunks LightRAG `mix`/`hybrid` follow from
  the matched entities and relationships by one search of each side's cited
  chunks for the query, as upstream's default `VECTOR` method does (#521).
- Query deadline: `search.query_timeout_seconds` (default 180 s, `null`
  turns it off) bounds every query (`ainvoke`/`invoke`, `asearch`/`search`,
  streams, which must also send a first chunk within
  `search.model_timeout_seconds`) and raises the new public
  `QueryTimeoutError`; `run-rag` exits `1` with its message and `run-eval`
  counts the query as failed. Query-time model calls get clients of their
  own: `search.model_timeout_seconds` (90 s) and `search.model_max_attempts`
  (2) on Bedrock, the request timeout capped at it on Ollama and
  OpenAI-compatible servers (`backend: local` presets 300 s and a 900 s
  query deadline). An embedding factory's `get_model` now receives
  `model_purpose`, as a language-model factory's does (#522).
- Query-path options: `search.answer_prompt_style` gives a strategy the
  comprehensive answer prompt (`ComprehensiveAnswerPrompt`, modelled on MS
  GraphRAG's global reduce prompt, `custom_prompts.comprehensive_answer_*`)
  instead of the concise default, which every strategy keeps;
  `search.global_search.evidence_tokens` caps the cited report text after
  the key points (null, the default, keeps the current behaviour; `0` sends
  the key points only; the `retrieval` evaluator then skips a global
  query's rank metrics, `no_sources_by_config`, instead of scoring a miss)
  (#525, #530).
- `search.local_search.report_selection: membership` gives `local` (and
  each DRIFT follow-up) MS GraphRAG local's community reports: those of
  the level-0 communities the expanded entities belong to, most matched
  entities first, then by report rank, read off the graph (graph expansion
  now projects an entity's `community_ids`). When no expanded entity names
  a community (under a document filter, or after a failed expansion) the
  similar reports are used. `similarity`, the default, keeps the current
  reports: offline the membership reports cost 5% fewer tokens, a
  difference too small to change the default without a real-model run
  (#535).
- Always-on LLM usage accounting, tracing or not: model calls, input and
  output tokens and cache hits (the LLM response cache and the query
  extraction cache) per model role and model id, in
  `RAGOutput.metadata["usage"]` / `SearchOutput.metadata["usage"]`,
  `EvaluationResult.usage` and `EvaluationSummary.usage`, `run-ingestion`'s
  JSON `token_usage`, and per-stage EMF metrics `llm_calls_<stage>`,
  `llm_input_tokens_<stage>`, `llm_output_tokens_<stage>` and
  `llm_cache_hits`, plotted on the CDK dashboard (#526). A call counts from
  its start: one still running when the usage is read (an `auto` query's
  unread speculative extraction) or cancelled in flight (a query deadline)
  was sent and is billed, so it is in `llm_calls`, and in
  `estimated_calls`, at an estimate of its input tokens; an `auto` query's
  usage now matches the model requests made (offline: 12 of 20 calls to 20
  of 20 routed to global, 8 of 16 to 16 of 16 to naive). A query extraction cache hit is
  recorded under the resolved model id (the inference profile), like the
  calls (#530).
- The `run-eval` comparison has a `cost` block: per strategy the answers'
  calls, tokens and latency per query (mean, p50, p95) and their paired
  deltas against the baseline with a bootstrap interval; an optional
  `evaluation.prices` table (no prices ship) adds a USD estimate. A failed
  or timed-out answer counts with the calls, tokens and time it spent
  before failing (`failed_answers`; each answer has its own usage scope and
  timer), so a strategy is not cheaper or faster for failing; repeat counts
  or cache hits per answer that differ from the baseline's are warned
  about. It is also written as per-query CSV and Markdown, with the
  pairwise judge's win rates (#526, #529).
- `run-eval --from-results` rebuilds a comparison from saved runs without an
  answer call (finishing an interrupted comparison by running only the
  missing strategies with the same `--repeats`), and re-judges them with
  `--pairwise-judge` and the new `--judge-model`; `--baseline-results`
  compares each strategy with the earlier run of the same strategy
  (`baseline:<s>` against `s`; the earlier run's first strategy, with a
  warning, when it has none) and is refused when the runs' `answer_scoring`
  differs; `--fail-if <measure>:<stat><op><value>` gates exit `1` on a
  regression, and a cost gate also when some answers have no value of the
  measure. Pooled runs of one strategy that differ in a query flag, the
  configuration hash, the commit, the index or the answer model, saved runs
  without a summary, extracted answer spans that fell back to the first
  line, an output directory that is the baseline's and a `--judge-model`
  that judges nothing are warned about (#526, #529).
- `evaluation.warm_up_query` (default `true`): `run-eval` answers one
  untimed, unscored synthetic query per strategy before its first timed
  run, so the first strategy's latency does not include warming the stores
  and models; it costs one extra query per strategy (#529).
- Community counters and a reset recommendation: the build record of each
  suffix keeps `communities` (`clustered_at`, `communities_at_clustering`,
  `communities_now`, `runs_since`), started by a full build and updated by
  each incremental commit (the communities it appended minus those its
  removals deleted), once per commit, so a resumed run does not count its
  commit twice. Once `indexing.community_compaction.max_runs_since_clustering`
  (default `20`) or `max_community_growth` (default `1.5`) is reached, the
  run logs a WARNING with the numbers that recommends `indexing.reset: true`
  (`on_drift_recommendation` adds the community drift recommendation); the
  `community_growth_ratio` and `runs_since_clustering` metrics let an alarm
  watch them. Nothing is reset automatically. A record written by an
  earlier version has no counters until the next full build (#540).
- The package ships a `py.typed` marker (PEP 561), so type checkers read
  its annotations instead of treating it as untyped (#568).
- Releases: pushing a `v<version>` tag runs `.github/workflows/release.yml`,
  which fails unless the tag matches `unified_kg_rag.__version__` and this
  file has a section for that version, builds the sdist and wheel, checks
  them with `twine check --strict` and for LICENSE and NOTICE, installs the
  wheel in a clean environment, and creates a GitHub release with both files
  and that section as its notes (`scripts/changelog_section.py`). A manual
  run is a dry run that publishes nothing. Nothing is uploaded to PyPI yet.
  `main` now reports version `0.3.0.dev0`; CONTRIBUTING.md describes the
  release steps (#572).

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
- `run-visualization` computes centrality only when a requested renderer
  reads it (`static`; a renderer declares it with
  `BaseRenderer.needs_centrality`): an `interactive`-only run of a
  5,000-node graph took 71 s, now 22 s (#563).
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
  deployments set; `docs/reference/configuration.md` has every key (#468).
- CDK alarms: `OpenSearchJvmPressure` follows AWS's recommended alarm and
  `NeptuneHighCpu` needs 3 consecutive 5-minute periods (#298).
- CDK: synth warns, rather than noting, when the guardrail stack owns a
  guardrail that compute is not given (`guardrail_identifier` unset): a
  later `cdk deploy --all` without the flag removed the guardrail from the
  task definitions, and queries and answers went unscreened with no sign
  (#566).
- CDK: the query task reads the graph through the Neptune reader endpoint
  (ingestion keeps the writer), so with `neptune_instances` above 1 queries
  go to the replicas instead of competing with an ingestion on the writer.
  A query right after an ingestion can miss its last writes for the
  replica lag ([Query access](docs/operations.md#query-access-and-tenant-isolation))
  (#566).
- CDK: a private-mode VPC gets the STS interface endpoint only with the new
  `sts_endpoint` context key (default `false`): only an image config with
  `aws.bedrock.assumed_role_arn` calls STS, and task role credentials come
  from the ECS credentials endpoint. One interface endpoint fewer (billed
  per AZ-hour) (#566).
- CDK: `log_retention_days` (default 30, as before) sets the retention of
  the task, pipeline, OpenSearch application and custom-resource log
  groups; the `owner` tag is applied only when `owner` is set (it was
  `aws-proserve`); the security stack is synthesized only with `use_cmk`
  (it held nothing otherwise); the `NeptuneLowFreeableMemory` threshold is
  1/32 of the instance class's memory (512 MiB on the default
  `db.r6g.large`, as before; 512 MiB for a class it does not know); the
  `PipelineFailures` alarm treats a period without executions as no
  failure, like the other alarms; the OpenSearch domain requires TLS 1.2 or
  1.3 with forward-secret ciphers (`Policy-Min-TLS-1-2-PFS-2023-10`);
  `cdk.json` pins strong cross-stack references and the custom resources'
  bundled SDK, the behaviour synth warned it defaulted to (#566).
- Docs: with `bedrock_model_arns`, list the reranking model too: the
  Rerank API also checks `bedrock:InvokeModel` on it (#566).
- CDK: the cache bucket is versioned, so an overwritten or deleted corpus
  object can be restored; noncurrent versions expire after 30 days (1 day
  under `pipeline-runs/`, which every stage overwrites). Versioning cannot
  be turned off again, only suspended. The Neptune cluster and its
  instances keep maintenance in a set weekly window (Sunday 20:00-21:00
  UTC) and backups in a daily one (18:00-19:00 UTC) instead of windows
  Neptune picked; a deployed cluster moves to them on the next deploy,
  without downtime (#519).
- Documentation: the docs describe the current release only and start at
  a hub, `docs/index.md`, with one folder per type: tutorials in
  `docs/getting-started/`, explanation in `docs/concepts/` (the overview
  with what the framework adds beyond the GraphRAG and LightRAG papers,
  retrieval, scopes and namespaces, and the measured defaults in
  `defaults-and-evidence.md`), how-to guides in `docs/guides/`, reference
  in `docs/reference/`, and error messages in `docs/troubleshooting.md`.
  `docs/operations.md` (now with *Shipping configuration to the deployed
  tasks* and *Upgrading the library*) and `docs/design.md` keep their
  paths, headings and section numbers. `docs/user-guide.md` and
  `docs/user-guide.ko.md` are now a permanent redirect: each former
  heading keeps its fragment and points to the new page, so older links
  and `--help` URLs still land. The generated references moved to
  `docs/reference/cli.md` and `docs/reference/configuration.md`, with
  `docs/reference/models.md`. The 14 Tier-1 pages have a Korean twin
  (`<page>.ko.md`, listed in `TRANSLATED_PAGES` in `scripts/check_docs.py`):
  `README.md`, `docs/index.md`, `getting-started/local.md` and `aws.md`,
  `concepts/overview.md`, `retrieval.md` and `scopes-and-namespaces.md`,
  and `guides/configure.md`, `ingest.md`, `multilingual.md`, `query.md`,
  `incremental.md`, `domain-adaptation.md` and `evaluate.md`; the other
  pages are English-only and the Korean pages link them. The docs checks,
  `scripts/check_docs.py` run by `tests/unit/test_docs_links.py` and
  `tests/unit/test_docs_translations.py` in the default suite, fail on a
  broken relative link, fragment or image, a broken doc path or URL cited
  in code, a target of either Git does not track (an untracked file, a
  tracked symlink to one, a directory holding no tracked file), an unlisted
  Korean page, or a Tier-1 pair whose structure differs without the lag note (#337-#342, #351, #469, #542-#544, #546,
  #547, #549-#554, #555, #556).
  in code, an unlisted Korean page, or a Tier-1 pair whose structure
  differs without the lag note (#337-#342, #351, #469, #542-#544, #546,
  #547, #549-#554, #555).
- Documentation corrections: the exported `visualization_data.json` path,
  the one reset that turning the doc-status registry on after a
  registry-off run needs, `aws.opensearch.use_iam` for IAM-authorized
  domains, what `--verbose` changes, the `comprehensive_answer` override,
  evaluation result names, the JSON each CLI prints on failure, and when
  `model_overrides` entries are checked. The "No indices found for suffix"
  error no longer names OpenSearch on `backend: local` (#557).
- Dependencies: pydantic 2.14, langchain-core 1.7.3, `langchain-aws>=1.8.1`
  (context-overflow errors), langchain-openai 1.7, langsmith 0.14, datasets
  5.1, tiktoken 0.14, tenacity 9.2, boto3 1.43.122; aws-cdk-lib 2.275 in the
  CDK lock; `nest-asyncio` is no longer used (#241, #251, #265, #438, #465,
  #489).
- Tooling: development tools are in the `dev` group; CI pins uv 0.12.23,
  runs the tests in parallel and on Python 3.13 (non-blocking), checks
  licenses on every supported Python, and lints async and logging rules;
  `-m "not aws"` is the pytest default; the Dockerfile takes a
  `BASE_IMAGE` argument; a layering test enforces the hexagonal import rules,
  statically and, for every module that has its backends injected, in a
  clean interpreter; the core-install job runs without the dev group (a
  `test` group holds the test runner) and runs a JSONL seed graph end to
  end too (#119, #257, #297, #303, #428, #456, #464, #465, #470, #512).
- CI runs the suites on every runner vCPU (`-n logical`; `auto` counted
  physical cores, half of them), on Python 3.11 in parallel too, and
  measures coverage with Python 3.12's `sys.monitoring` tracer; a
  `coverage-combined` job adds the local-store job's coverage of the
  Neptune and OpenSearch adapters to the summary; a weekly
  `hypothesis-explore` workflow runs the property tests from random seeds
  with a larger budget; Dependabot also moves the local store images; the
  ASH scan writes its findings table to the job summary (#568).
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
  extraction records keep their keys (#475).
- An `entity_types` list with two labels that differ only in case or word
  separators (`PERSON`, `person`) fails the load (#475).
- Entity and relationship ranks, and the order and rank of community
  report entities, count document-backed edges only: an edge every text
  unit of which is a seed graph's adds to no degree, while Leiden still
  clusters every edge. Without seed graphs every rank and report is
  unchanged (#502).
- A reset that would drop another source scope's seed graph documents,
  or, from a corpus of seeds alone, another scope's documents, asks
  before any write. `run-ingestion --confirm-reset` (or
  `UKR_CONFIRM_RESET=1`, for an ECS task overridden by environment only)
  answers it and nothing else, `--yes` answers it too; a no, or no
  terminal, stops the run with exit 1 and no error traceback.
  `DataIngestionPipeline(confirm=)` takes the callable that answers it
  from Python, and `run` raises `ConfirmationDeclinedError` (a
  `PipelineExecutionError`) on a no. A reset from seeds alone fails,
  whatever the answer, for a scope that holds documents under any
  spelling of its source (`s3://bkt/corpus` for `s3://bkt/corpus/`) and
  for a scope with no record beside other scopes' documents (a new or
  mistyped prefix); a seed-only scope's reset, which has seed records of
  its own, still asks (#503, #509, #513).
- `answer_match` exact match and token F1 read the answer's first line
  (`evaluation.answer_span: first_line`, the one-sentence answer the answer
  prompt writes) instead of the whole answer, whose supporting sentences
  diluted them: a correct `Depot stores widgets for Vendor.` plus support
  scored token F1 0.125 against `Depot`, now 1/3. `answer_span: full`
  restores the old scores; the span is recorded in the run manifest
  (`answer_scoring`) and each report, and a comparison warns when its runs
  differ. The docs no longer claim EM/F1 are "scored like the papers" (#516).
- Entity and relationship lines in the answer context no longer carry an
  `[id]` prefix (a UUID was about a quarter of their tokens; nothing reads
  those ids); passages, claims and reports keep theirs. Entity lines show
  the entity's type. The incident-edge page-cap warning is logged once a
  process, then at DEBUG (#521).
- Community report prompts show a member with no type and no description
  as `- Name (unknown): ` instead of `- Name (None): None`, on runs
  without seed graphs too: the prompt of each community with such a
  member changes once, so its report misses the LLM cache once (#515,
  #523).
- Every source kind that reads files (`schema_source` kinds `file` and
  `owl`, seed kinds `jsonl`, `csv` and `rdf`) takes a local path written
  as a `file://` URL (no host, or `localhost`), as `owl` alone did; any
  other URL fails the load with the same message for each (#520).
- `logging.log_to_file` defaults to `false`: a CLI started in a directory
  it cannot write (a read-only container root) failed creating `logs/`
  before any work. The console carries every record; set it to `true` for
  the dated `logging.log_file_path` file as before (#520).
- `langchain` is no longer a core dependency: nothing imports it, and it
  pulled in LangGraph. The `eval` extra, whose ragas needs it, declares it
  with the same `>=1.3.9` security floor, so a pip install of `[eval]`
  gets the floor too (#520).
- `auto` starts the entity and keyword extractions its routable strategies
  read, and the query translation, alongside the router call instead of
  after it (`search.auto_speculative_extraction`, on by default): `auto`
  routed to local or mix makes two model round trips before retrieval
  instead of three and takes as long as the strategy asked for directly
  (1.54 s to 1.03 s at 0.5 s a call). The router prompt and its decisions
  are unchanged; the extraction the route does not read still runs, one or
  two short fast-model calls a query (about 400-1,000 input tokens with the
  shipped prompts), in the background: the query returns without waiting
  for it and counts it from its start (cancelling it stopped no billed
  request, a Bedrock call runs on a worker thread to the end, and left it
  out of the query's usage). It still fills the extraction cache, and is
  cancelled at the query's deadline or when the chain closes (#525, #530).
- Query entities and keywords are cached per question, target language,
  model and prompt text (`search.query_extraction_cache_size`, 1024; 0
  turns it off), so `run-eval --repeats` and the arms of a strategy
  comparison search with the same ones and pay the extraction once (#525).
- The short-answer query calls ask for a smaller max_tokens
  (`search.output_ceilings`, 4096 each: the router, the query entity and
  keyword extractions, the dynamic-selection ratings; and
  `evaluation.answer_span_output_ceiling`), never less than the call's
  thinking budget plus 1024, so an `auto` query reserves 28,672 output
  tokens against the Bedrock quota instead of 49,152. A language-model
  factory's `get_model` may receive `output_ceiling` (#525).
- Indexing embeds only text it has not embedded: each vector stores a
  digest of its text with the embedding provider, model and dimension, and
  an upsert or a rebuild reads the stored digests of its items (one search
  per 800 ids), then only the vectors whose digest matches (100 ids per
  search), and embeds only the texts whose digest differs: rebuilding 2,000
  reports at 1,024 dimensions with a third of the texts changed peaks at
  165 MiB instead of 388 MiB. An index
  built before gets the digest field on its next incremental run and its
  items are embedded once more. A one-chunk edit of the cost harness's
  corpus embeds 44 texts instead of 94, a reset of the unchanged corpus 15
  (its regenerated community reports) instead of 376; `--force-rebuild`
  embeds everything (#527, #531).
- The intelligent chunker keeps each pre-chunk's boundary answer in the
  extraction store (keyed by the pre-chunk's text, the chunking settings
  and model), named by the text units cut from it: a reset of an unchanged
  corpus asks the boundary model nothing (24 calls before), and a model
  that answers otherwise each time no longer moves the chunks, and with
  them the extraction records, of unchanged text. Only an answer the
  chunker used is kept: one it fell back from (lines that do not exist) is
  asked again next run. A stored answer counts as a cache hit in the
  stage's usage. `--force-rebuild` asks again (#527, #531).
- A run reads each alias and index mapping of the vector store once
  (they change only by its own writes while it holds the lease), and a run
  that indexes nothing writes no build record and no extraction settings
  (unless they changed: a scope of seed graphs alone warns of a settings
  change once, not on every run): a no-op incremental run writes nothing
  but its lease, with 29 store calls instead of 56, and a one-chunk edit
  looks up 6 aliases instead of 61 (#527, #531).
- A full build (a reset, a first run, a run without the registry) reuses
  the corpus-wide answers whose input is unchanged: a description over the
  summarization threshold keeps the summary its stored copy made from the
  same text with the same summarization settings (recorded as
  `summarized_with` beside `summarized_from`), and a community report
  answer is kept in the extraction store by the report's packed input,
  name and settings and named by the report (`attributes.report_record`).
  The LLM response cache never served them: its prompt keys are stable,
  but it lives in one pipeline id's cache directory. On the cost
  harness, a reset right after the first build, or after another reset,
  asks 0 model calls instead of 29 and 28 (summaries and reports) and
  leaves the stores as they were; the first reset after incremental runs
  asks 8 instead of 28 (the reports of the communities the incremental
  runs clustered otherwise, and one description no run had summarized),
  and a reset with another community resolution 3 instead of 24. A reused
  answer counts as a cache hit. `--force-rebuild` asks again (#537).
  Community names derive from the content id (`Level 0 Community
  1a2b3c4d`, short id `L0_C1a2b3c4d`) instead of the rank in the sorted
  partition (`Level 0 Community 3`): the name is part of the report's
  record key and of its parent's roll-up text, so a document that added a
  community renamed every community sorted after it and the next reset
  asked their reports again. With 8 stored communities and one disjoint
  document added, the reset asks 1 report instead of 3 or 4. Names no
  longer collide between a delta run's communities and the stored ones
  either (#539).
- Defaults from a real-model evaluation (MuSiQue, 200 questions over 3,318
  documents, a 50-question sensemaking set, two repeats, three pairwise
  judges; design §9.1). `global` answers with the comprehensive prompt: it
  won comprehensiveness, diversity and empowerment at 0.95-0.99 with every
  judge for +$0.016 per query, and lost directness (0.00-0.39); pairwise
  judges favour longer answers. `drift` stays concise (comprehensive p95
  168 s, max 178 s, against the 180 s deadline): with `{drift:
  comprehensive}` raise `search.query_timeout_seconds` (300 s). A
  configured `search.answer_prompt_style` is merged over the default.
  `search.global_search.evidence_tokens` is 4000: -41% cost per global
  query ($0.1016 -> $0.0597), no consistent quality difference, sources
  kept. The `run-eval` judge (`models.roles.evaluation`) is Opus 5.5 on
  Bedrock, not the answer model: a judge judging its own answers shifted
  directness by -0.27 [-0.38, -0.16]. When the answer model is Opus 5.5
  the unset judge falls back to the default tier's model or Sonnet 5.5
  (logged at INFO); a configured judge that is the answer model is kept and
  warned about. The CDK stack's default Bedrock grant is unscoped, so it
  covers the judge. Kept, with the measurement documented: gleaning `max_rounds` 3
  (+9.8 points [+3.7, +17.1] of the questions' intermediate answers in the
  graph, +38% ingestion cost; `1` is the cost lever), `kg_chunk_pick_method:
  weight` and `diversity_lambda` 0.5 (no significant difference) (#533).
- Usage records count prompt-cache reads and writes apart
  (`cache_read_input_tokens`, `cache_write_input_tokens`; still part of
  `input_tokens`), the requests botocore resent inside a call
  (`request_retries`) and calls that failed (`failed_calls`, not in
  `llm_calls`). `evaluation.prices` takes `cache_read_per_million_tokens`
  and `cache_write_per_million_tokens`, so a cached run's USD estimate is no
  longer priced at the input rate. Bedrock's `Invocations` counted 13% more
  Sonnet calls than the meter in the evaluation above (tokens within 4.6%),
  most likely resends and failed requests (counted now) and unscoped calls
  such as the warm-up queries; the runbook's model usage metrics explain how
  to reconcile them. A per-TTL cache write
  (`ephemeral_5m/1h_input_tokens`) is no longer lost from the traces'
  `gen_ai.usage.cache_creation.input_tokens` (#533).
- An incremental commit, the run's last write, reads only the stored vectors
  it takes: the ones its upserts replace were read (up to 2,048 a write) and
  kept for a later write that never comes, 2,000 vector fields of 4,000 read
  on an upsert of 2,000 changed descriptions. The removal before it still
  keeps the vectors it replaces for the commit to take (#541).
- A `run-eval --fail-if` on a cost measure (`usage/*`, `latency/*`) fails
  when the strategy and its baseline failed different numbers of answers:
  a failed answer counts with what it spent before failing, so the delta
  compared the failures too (a warning only before) (#541).
- DRIFT search metadata counts its follow-up local searches by
  `report_selection` (#541).
- CI builds the image with the `unstructured` extra too and loads its spaCy
  model with no network (#541).
- The IaC synth refuses `network_mode=private` with an
  `observability_secret_arn` of another region: the Secrets Manager
  endpoint reaches the deploy region only, so no task could start (#541).
- An incremental run with nothing to index (an unchanged corpus, or
  deletions only) logs its empty graph at debug level instead of three
  warnings ("Cannot calculate statistics for an empty graph", "Community
  detection skipped…", "No communities provided for report generation"). A
  full build, or a delta whose documents state no entity, still warns
  (#545).
- A comparison counts the scores an evaluator could not compute (a failed
  or NaN judge call) per strategy and metric
  (`strategies.<s>.failed_metrics`), warns when the counts differ, and a
  `--fail-if` on such a metric fails: the query was only left out of the
  delta, so a judge that failed on the 4 queries a strategy lost showed
  +0.10 [+0.10, +0.10] over the other 6 and passed the gate. The Markdown
  and console tables print each delta's pairs and unpaired queries (#563).
- `graph_aware`: `retrieved_entity_coverage` and the endpoints of
  `retrieved_relationship_coverage` match a retrieved entity by entity key
  (case, CJK spacing, quotes and legal forms ignored, as entity resolution
  links names), not as a phrase inside its name: `Seoul National
  University` covered an expected `Seoul`. Expected entities with the same
  key are one expectation (any spelling in the answer meets it), so a
  duplicate no longer counts twice in either coverage. Both metrics can
  read lower than before on the same answers (*Upgrading*) (#563).
- A `run-eval --fail-if` fails on a delta over fewer than 10 paired queries
  (`gates.MIN_GATE_PAIRS`): one pair gave a zero-width interval, so
  `ci_high<-0.02` passed on one improved query. The evaluation guide says
  how the false-breach chance grows with strategies × gates (#563).
- One parser reads every S3 path (`run-eval`'s `--data-path` and
  `--output-directory`, schema and seed sources), with the scheme in any
  case: `run-eval` read `S3://bucket/key` as a local path. An OpenSearch
  query request retries through the shared transient-retry helper, so no
  retry starts past the query's deadline (#579).

### Deprecated
- `run-rag` / `run-eval --enable-thinking` logs a deprecation WARNING and
  will be removed: the default models (Claude 5.x) always reason, so it has
  no effect on them; tune their depth with `aws.bedrock.default_effort` /
  `fast_effort`. It still turns thinking on for the models that opt in
  (#577).

### Fixed
Bugs present in 0.2.3.

Queries and retrieval.
- The graph retrievers (Neptune, embedded SQLite) check the filters a
  strategy sends as the OpenSearch retriever does (`check_filter_values`):
  a range operator other than `gt`/`gte`/`lt`/`lte` (`in`, `eq`, `neq`)
  was dropped, so a custom strategy's filter matched every vertex, and a
  null or empty-range value reached the traversal. Each now raises
  `InvalidFilterError` (#PR).
- A number or boolean document filter value is a keyword on OpenSearch, as
  the filter docs said: a field's first numeric value mapped it as `long`,
  so a document with a string value for the same key failed to index
  (`mapper_parsing_exception`) and a range compared numbers where the
  embedded engine compared text. Both stores hold such a value as its JSON
  text (`2023`, `2.0`, `true`): the embedded engine no longer reads `2.0`
  as `2`, and compares a keyword's numbers as text in a range (#566).
- Local search looks each query entity up on its own, concurrently, and
  seeds graph expansion with the entity of its name, else with its top hit
  when that scores at least `search.local_search.seed_score_ratio` (new,
  0.6) of the best hit: one lookup of the joined names seeded a wrong
  neighbour beside every right seed (12 of 12 synthetic multi-hop queries;
  a generic query entity seeded unrelated ones), and all-gold context went
  from 8/12 to 12/12. The seeds share the expansion budget in proportion
  to their scores; Neptune walks each share's seeds in a request of its own
  (#517). On a hybrid search the ratio is read at equal weights and
  rescaled to `search.hybrid.vector_weight`, so a hit only the vector
  search found never seeds beside a name match at any weights (at 0.2/0.8
  every query seeded two unrelated entities, now none); a vector search
  compares raw similarities, where the floor filters less, and
  `lexical_weight` 0 logs that it filters nothing (#530).
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
- A boolean or numeric filter given as a string (`run-rag --filters
  in_schema:false rank:2`, `set-filter`, a string in a `run-eval` dataset
  or in `RAGInput.filters`) is converted to the field's type, so graph
  expansion on the SQLite graph and Gremlin, which matched nothing, agrees
  with OpenSearch and the Python API's typed values. A boolean takes
  `true`/`false`, `1`/`0` or `yes`/`no` (any case); a string the type
  cannot take (`in_schema:maybe`) raises `InvalidFilterError` (exit 2 on
  the command line). A typed value must be of the field's type too: a
  boolean field takes no number (`in_schema: 1`), a numeric one no boolean
  (`rank: true`) and an integer one no fractional number (`level: 2.5`;
  `2.0` is `2`). The errors name the value as JSON does (`true`, `null`)
  (#479, #483, #490).
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
- LightRAG `mix`/`hybrid` context includes the chunks the matched
  relationships cite. The relationship side dropped every chunk any matched
  entity cited, selected or not, and shared one chunk-width cut with the
  entity side, so its chunks almost never reached the context (offline
  hybrid one-hop gold-chunk recall 0.30, now 0.86 without rerank and 0.96
  with). Each side is now its own fusion stream (`lightrag_relation_chunks`)
  of up to `chunk_stream_top_k` chunks, dropping only the chunks the entity
  side picked, and the text quota cuts after the rerank, as upstream
  does. A lineage is read in chunk-id order, so an incremental build and a
  full build of one corpus pick the same chunks (#521). A
  `search.fusion.fusion_weights.lightrag_linked_chunks` weight now applies
  to the entity side's chunks only; the relationship side's take
  `lightrag_relation_chunks` (default 1.0) (#530).
- Bedrock CountTokens requests use the query model client's read timeout
  and attempts (`search.model_timeout_seconds`, `model_max_attempts`), not
  botocore's 60 s and 4 requests: a count sits on a query's critical path,
  and a failed count falls back to the local estimate (#530).
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
- A stalled Bedrock call no longer holds a query for about half an hour:
  queries shared the ingestion client (a 300 s read timeout, 330 s on
  `main` since #174; 6 botocore attempts; the retry budget checked only
  between attempts) and had no deadline. The query now fails at `search.query_timeout_seconds` (a
  stalled call gives up after about 3 minutes on its own), the transient
  retries start no new call past the deadline, and `run-rag` exits without
  waiting for the abandoned call (#522).
- Under a `source_kinds` filter that keeps one kind (`document`, say),
  `local` and `drift` read no community report: a report carries no source
  kind and summarizes seed-only members too, so `source_kinds: document`
  still answered with seed-only facts. DRIFT then runs one local search on
  the question; `global` refuses the key as before (#534).
- `simple` reads no community report under such a filter either: its sweep
  of every index still read the reports index, so `source_kinds: document`
  answered with reports that summarize seed-only members. Of every
  strategy, `local`, `drift` and `simple` (and `auto`, which routes to
  them) skip the reports, `global` and `naive` refuse the key, and `mix`
  and `hybrid` read no report (#536).
- The `run-rag` suffix line names `indexing.additional_suffix` also when
  `--suffix` or `processing.document_parsing.index_value` gives the index
  suffix (#536).
- `run-rag` and `run-eval` query the config's
  `processing.document_parsing.index_value` when `--suffix` is not given,
  as ingestion writes it: they queried `default` and every query of a
  corpus ingested under another value failed with "No indices found"
  (#534).

Ingestion and incremental indexing.
- A model answer that corrects itself (an answer, a line such as
  "Correction: ... the corrected, final output:", the answer again) is
  parsed by its last complete answer: each XML section written more than
  once keeps its last closed occurrence (graph extraction, gleaning,
  claims, community reports, chunk boundaries), and `parse_llm_json` reads
  the last top-level JSON object that parses when the span holds several
  (LightRAG keywords, prompt tuning, global and DRIFT search, the judges).
  Graph extraction failed such a unit as cut-off output (138 of 3,318 units
  of a Sonnet 5.5 run), and a strict keyword parse failed the whole
  `mix`/`hybrid` query with "Extra data"; the pairwise judge read the first,
  superseded verdict. A single answer cut off is still flagged (#532).
- A section that follows a closed one with only whitespace between them
  starts a section, so a correction written on one line keeps its final
  `<relationships>`: they were dropped for the superseded ones. An expected
  tag inside another element is not a section, so a community report whose
  `<finding>`s lack their `<findings>` keeps its own `<summary>`, not the
  last finding's (#536). A correction after prose on the same line
  ("... Sorry, correction: <entities>...") counts too when it writes a
  section already closed in full again: each section parsed as a list of
  the two answers, which graph extraction failed as malformed output
  ("none of the 2 entities ... has a name") and retried. Prose naming a
  section ("the <entities> above") writes no end tag and still does not
  (#560).
- Strict `parse_llm_json` (LightRAG keywords, prompt tuning, the pairwise
  judge) raises when the last object is cut off instead of returning an
  earlier one (the answer being corrected, or an example), and prefers the
  last object holding the caller's answer keys over an example object (#536).
- Graph extraction output whose entities have no name but whose sections
  are all closed is malformed, not cut off: it is retried
  (`retry.max_attempts`) and, failing every attempt, counted as the unit's
  own error as before; with a section unclosed it is still cut-off output,
  not retried. The response behind a failed graph or claim extraction check
  is logged at DEBUG (#532).
- Stored chunk-boundary answers count as cache hits under the resolved
  model id, like the calls and the query extraction cache (#532).
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
- Community report input is packed edge by edge, as MS GraphRAG's
  `sort_context`: relationships by combined endpoint degree, each with its
  two entities (and, with claim extraction on, their claims, which the
  report prompt now lists), then the entities with no relationship. Packing
  the entities first left the relationships whatever budget remained (with
  200-token descriptions, 2 of 40 edges); on a synthetic grid, 241 instead
  of 126 relationships reach the same 4000-token budget and every packed
  entity has one (51.6% before) (#517).
- A file skipped because no loader reads its type (an optional extra such
  as `unstructured` dropped from the image, a loader plugin not imported)
  no longer reads as deleted: an incremental run keeps what it indexed, as
  for a file that fails to parse, and one WARNING names the count, the
  extensions and the extra to install (#518).
- A description summary that is itself over the summarization threshold is
  not summarized again when the indexing reuses it: a first build with the
  doc-status registry on asked the model twice for such an item (graph
  resolution, then the indexing), and an incremental run summarized the
  summary it reused again, storing a description the community report did
  not read (#523).
- Without network access and without the spaCy model the `unstructured`
  extra downloads (`en_core_web_sm`), each Markdown or HTML file waited
  120 s and failed: the first failure is now kept, and every later file of
  the run fails at once with it and the command that installs the model
  (#534).
- A request OpenSearch rejects (a 400) is logged without the client's
  traceback, so the refusal of `vector_mode` on OpenSearch 2.13 shows the
  message that names the fix (#534).
- With `processing.fuzzy_entity_resolution` on, a graph store read of the
  stored entity names that failed (Neptune, the SQLite graph) returned no
  names: the incremental run merged its delta without the fuzzy cross-run
  merge and stored near-duplicates a full build merges, logging only a
  WARNING. The read raises now (`AWSServiceError` on Neptune, the store's
  error on SQLite), as every other graph read does, and the run fails before
  it writes (#564).
- A cached stage output that no longer validates as its model (a model
  field changed between releases) was returned as raw JSON and counted a
  cache hit, so the stage resumed with dicts where it expects models. It is
  a miss now, and the stage runs again, as a cached chunk that fails
  validation already was (#564).
- An incremental run summarized a description in vain or summarized a
  summary. Adding a document whose lines graph resolution summarized
  merged into a stored description that was not a summary: the indexing
  summarized all its lines again and the first summary went unused (now the
  stored description is extended with it, one call at most). Editing a
  document summarized each shared entity and relationship it still names
  twice: the removal summarized what was left, then the commit extended
  that summary, storing a summary of a summary where a full build
  summarizes the lines once. The removal now leaves the summary of an item
  the delta derives again to the commit, and the commit summarizes an item
  the removal derived again from its lines, as a full build does
  (*Upgrading*) (#564).
- The design notes said full and incremental builds group the same names
  with fuzzy matching on. Without it they do (both merge by compact name
  key); with it, an incremental run folds a delta entity into its one best
  stored match and never joins two stored entities, so a new name similar
  to two stored ones merges with one of them only until a reset. The
  design notes and the incremental guide say so now (#564).
- Entity resolution took the merged entity's type from the member whose
  name won only, so where that member had no type (an endpoint named only
  in relationships, a stub), the merge had none: "Acme Inc"
  (`organization`, one unit) and a stub "Acme" cited by two units became
  an untyped "Acme". When the winning member's type is unknown (none,
  blank, `unknown`), the merge takes the known type the most units cite
  among the members documents state (a seed's only when no document gives
  one); the incremental derivation runs the same code (#565).
- The translation prompt read "Translate the following text to
  LanguageCode.KO" (the enum formats so in a template), and the description
  summary prompt "Write the summary in ko". Both name the language now
  ("Korean"), as the query prompts do (*Upgrading*). The extraction and
  gleaning prompts keep "the language with code 'ko'", which reads right:
  rewording it would re-key, and so re-extract, every unit (#565).
- Output that parsed but had the wrong shape skipped the output fixer: the
  extraction, claims and gleaning checks ran after the chain. An extraction
  whose sections said `None`, or that one `<output>` element wrapped,
  failed its unit after every retry; a bulleted answer parsed as
  `{"items": [...]}`, which nothing reads; a claims answer without its
  section was counted as cut off (never retried); and a gleaning answer
  without its `<refinement_plan>` root was read as "nothing to add" and its
  unit marked gleaned for good. The XML parser now unwraps one element
  around the expected sections and reads a section of only "None" (or
  "N/A", "null", "-") as empty, keeping the sections beside it; the three
  parsers raise `OutputParserException` for any other wrong shape (no
  section, a section holding prose, nameless entities in a complete
  answer), so the fixer repairs it; a gleaning plan written without its
  root is read as the plan. The bullet-list fallback is gone (#565).
- A field holding markup was read three ways: entities and relationships
  stored its dict as text (`<description><p>one</p></description>` became
  `{'p': 'one'}`, `<name lang="en">Beta</name>` a name of that dict), the
  claim reader took a one-child dict's text, and community reports stored a
  repeated `<summary>` of a finding as the list's repr. One rule
  (`coerce_llm_text`) reads every field now: the element's own text, else
  its children's text, attributes and comments dropped, a repeated
  description-like field joined (#565).
- The gleaning prompt listed the unit's entities by name only and named no
  type: gleaned entities and relationships were typed without the declared
  entity types or the schema's relationship types, and every gleaned
  entity took confidence 1.0 (the
  top) and every gleaned relationship weight 1.0 (the bottom of the 1-10
  strength scale), as the prompt asked for no score. It now lists each
  entity with its type (`Acme (ORGANIZATION)`), gives the extraction
  prompt's entity and relationship type blocks, and asks for
  `<confidence>` and `<strength>` on extraction's scales, which the
  gleaner already read; a score left out takes extraction's default
  (*Upgrading*) (#565).
- The `intelligent` chunker asked the model for the boundaries of a
  pre-chunk of one line (text without line breaks, common in extracted
  PDF and CJK text), which has no line to cut at: the call answered no cut
  or lines that do not exist, and the second case counted as a failed call
  in `llm_processing_failures` (a 40,000-character line: one call, one
  "failure", ten size-split chunks). Such a pre-chunk is now split by size
  without a call, counted in `fallback_chunks_used` only, so
  `llm_failure_rate` counts failed calls (*Upgrading*) (#565).

Evaluation, visualization and tuning.
- A re-judge (`--from-results`, `--baseline-results`) with no
  `models.roles.evaluation` set also falls back from the default judge
  when it is the model that wrote the saved answers (their manifests'
  `models.answer_generation`), not only when it is this config's answer
  model (#536).
- Judge failures are recorded per query, failed scoring exits non-zero
  (#158, #210), and a failed RAGAS run marks only its own metrics (#219).
- RAGAS skips `faithfulness` for an answer without context (#366), counts
  each context once per batch (#246), judges with its own metric objects
  (#399), and stops when the evaluation is cancelled (#386, #393).
- `run-eval` reports an invalid or missing dataset with exit 2 (#227,
  #230).
- The static charts' logarithmic histograms (`static.log_scale_x`) count
  the largest values: the last bin's edge could round below them, which left
  the highest-degree node out of the degree distribution (#576).
- `run-visualization` rejects unknown `--renderers` and invalid data with
  exit 2 (#255, #353) and lays out only the drawn nodes (#246); physics
  stays off when `interactive` omits it (#292).
- `run-visualization` exits 1 when a renderer fails while another writes
  its files (it exited 0), and its JSON names each failed renderer and its
  error (`failed_renderers`). The centrality chart leaves out a metric that
  is off or failed to compute, and gives a node without a value no bar;
  both were drawn as 0.0 (#563).
- `run-prompt-tuning` profiles with the `graph_extraction` model, writes
  types UPPERCASE in its examples, logs no model output at ERROR, and
  rejects `--max-docs` below 1 (#227, #292, #353).
- A comparison scores a failed answer 0 only on the metrics its strategy
  scores somewhere, so `global` failing one query no longer shows a
  significant -1.0 `retrieved_entity_coverage` delta it never measures; a
  strategy that scored nothing is still filled on every metric (#516).
- The pairwise judge reports `position_consistency` (a judge that always
  picks the first answer scored all ties, read as equal answers, now 0.0
  and warned about) and a bootstrap interval on its win rate, gives no win
  rate when it decided no pair (it was 0.0 from failed answers alone),
  skips `answerable: false` items, and counts in the run manifest's
  `judge_is_answer_model`, with a warning when it is the answer model
  (#516).
- `answer_contains` and `graph_aware` match a word through its possessive
  (`Depot` in `Depot's`); exact match and token F1 are unchanged (#516).
- `answer_contains` and `graph_aware` find a Latin word or a number inside
  Chinese or Japanese text, which puts no space around it: `AWS` in
  `答案是AWS公司`, `Acme Corp` in `ベンダーはAcme Corpです`, `2024` in
  `2024年` all scored 0. A letter or digit next to it still ends the match
  (`AI` is not in `airport附近`, `2` not in `12年`), and Korean text is
  matched as before (#563).
- A model call whose response reports no token usage (a model or
  OpenAI-compatible server that returns none) keeps its estimated input
  tokens and counts in `estimated_calls`; it was settled at 0 tokens and
  read as a measured, free call in the usage records and the comparison's
  cost (#563).

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
  raise on `structlog` context fields, and after `setup_logging` a host's
  own root handler gets the plain message and traceback, not the event
  dict; importing the package configures no logging and does not touch
  `sys.path` (#178, #245, #251, #255, #528).
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
- CloudWatch alarms deliver to the alarm topic: its topic policy held only
  the TLS deny and replaced the default policy, so no alarm notification
  was ever published ("not authorized to perform: SNS:Publish"). It now
  allows `cloudwatch.amazonaws.com` to publish from this account (#534).
- An image built with `UV_EXTRAS="--extra unstructured"` installs the spaCy
  model the extra needs, which a task in the private VPC could not download
  (each `.md` / `.html` file waited 120 s and failed) (#534).
- The dashboard plots Bedrock invocations and throttles in two widgets: with
  both Metrics Insights queries in one, `GetMetricWidgetImage` could not
  render it (#534).
- A deploy that replaced the ingestion task definition (a new `image_tag`,
  guardrail id or version, Korean analyzer, size, tracing or secret) failed:
  the state machine's stack imported the task definition's ARN, an export
  CloudFormation cannot change while another stack imports it (`Export ...
  cannot be updated as it is in use by GraphRagOrchestration`), and a
  deploy of the compute stack alone left the state machine on the old
  revision. The ingestion task definition is now in the orchestration
  stack, next to the state machine that pins its revision, so `cdk deploy
  --all` replaces both and the state machine runs the new revision; its
  role, the cluster, the image repository and the log group stay in the
  compute stack. Upgrading a deployed stack takes one extra deploy
  (*Upgrading* above) (#558).
- The *Incremental run health* dashboard widget plots
  `community_growth_ratio` and `runs_since_clustering` as well, as the
  runbook said it did (#558).
- Local search with `report_selection: membership` read no report when a
  `community_id` or `level` filter admitted none of the level-0 communities
  of the expanded entities; it reads the similar reports under the filter
  then (#541).
- A strict JSON parse (keyword extraction, prompt tuning, the extraction
  parser) no longer refuses an answer followed by prose holding a lone `{`:
  only a `{` that starts JSON (`{"`, or one the response ends on) reads as
  a cut-off object (#541).
- A cancelled model call (its task cancelled, so LangChain reports no end)
  is let go when its task ends instead of staying in the usage handler
  until its cap; its estimate stays counted (#541).
- Simple search logged "Community reports skipped" under a `source_kinds`
  filter even when its sweep never held the reports index (#541).
- A description summary setting
  (`processing.graph_extraction.description_summarization`, its
  `custom_prompts` or `models.roles.description_summary`) re-extracted the
  whole corpus: keyed with graph extraction, it changed every unit record's
  key and every document's extraction settings, though summaries are made
  from merged descriptions only. Changing the summary threshold alone re-ran
  extraction, gleaning and reports: 822 model calls (USD 16.71) on 151
  documents. These settings now key graph resolution and are no extraction
  setting; a summary records the settings it was made with, so a stored one
  made otherwise is not reused, and a reset summarizes every description
  again. A document's recorded extraction settings stay valid whatever
  summary settings they were recorded under, so an incremental run
  re-extracts nothing. A unit record stored before keeps its key under the
  default summary settings; one stored under others is found under the key
  it was stored under and stored under the current one, so the first reset
  re-extracts nothing either (#541, #545). That earlier key is computed
  from the run's summary settings and tiers: to keep records stored under
  other summary settings, run once after upgrading with the settings (and
  `models` tiers) they were stored under, then change them.
- A unit record's key held the default summary settings resolved through
  the run's `models` tiers, so with `models.roles.description_summary`
  unset a change of the tier the summary model reads (`models.fast`) re-keyed
  every record: where nothing else a document's extraction reads goes
  through that tier (the `simple` chunker, translation off), no "settings
  changed" warning told the user and the next reset extracted every unit
  again. The key holds the default configuration's summary settings now, the
  same whatever the tiers; a record stored under the earlier forms (the
  run's summary settings, or their defaults through the run's tiers) is
  found under the key it was stored under (#556).
- A relationship's vector copy did not store its type, so a relationship
  without a description (every edge an RDF seed states, a JSONL or CSV
  seed edge without one, an extracted relationship with an empty
  description) reached the answer prompt as `Relationship: A -> B` under
  `local`, `drift`, `simple`, `mix` and `hybrid`, while the graph's edge
  holds the label: answers called the link undescribed. The relationships
  index stores the type now (a keyword, the graph's label, `RELATED_TO`
  when untyped) and the line reads `Relationship: A -[TYPE]-> B`. It is no
  filter key: a `type` filter still leaves relationships unfiltered. An
  incremental run maps the field on an existing index; a copy written
  before reads as before until it is rewritten (*Upgrading*) (#562).
- The seed-graph guide did not say that `source_kinds` filters entities
  and relationships only: text units declare no such field, so `local`,
  `mix` and `hybrid` still return the documents' text chunks under
  `seed_graph`. The CDK README said `cdk destroy --all` removes everything
  in `dev`; it lists what it leaves now (the Container Insights
  performance log group, the storage stack's two Lambda log groups and the
  `INACTIVE` task-definition revisions) and the commands that delete them
  (#562).
- An entity hit from the entities index reached the answer prompt as
  `Description: ...`, a blank line, `Title: X`, a blank line, `Type: T`,
  and the context joined items with one newline, so under `mix`, `hybrid`,
  `simple`, `local` and `drift` the next entity's description read as this
  one's title (relationships too: the description sat a blank line below
  its endpoints, right above the next relationship). An entity hit renders
  as the graph walk does (`Entity: X`, `Type: T`, `Description: ...`), a
  hit's lines are no longer split by blank lines and the context separates
  items with one (#567).
- An entity reached by graph expansion showed no type: neither the Gremlin
  projection nor the SQLite graph's read its `type`. Both project it now,
  and a one-element `text_unit_ids` or `community_ids` comes back as a list
  from either store, as from the entities index (#567).
- Local search's chunk filter check and global search's step-down to a
  shallower cut read the OpenSearch index schema whatever the document
  retriever was: a custom backend's own filter keys never narrowed the
  chunks or kept the cut, and a backend without a schema had its keys
  dropped. Both ask the document retriever (`filter_fields_for`); one
  without a schema declares every key (#567).
- Global search's map calls ran in a worker thread, so a query past
  `search.query_timeout_seconds` or a closed chain left them running and
  kept starting the queued ones, each a billed model call (all 10 of 10
  batches started after a deadline that cancelled the query at 4). They
  are awaited on the event loop now, at most `search.max_concurrency` at
  a time, as DRIFT's: a deadline cancels the calls in flight and starts no
  other. A map call is bounded by the model client's timeout
  (`search.model_timeout_seconds`) and the query deadline; the 300 s
  per-call ceiling and its "map call gave up after" error are gone (#567).
- Dynamic community selection read a rating with `float()` of the whole
  answer: `8/10`, `Score: 2` or a number followed by a reason became the
  default 5, `nan` became NaN, which no threshold drops, and `85` stood
  above every 1-10 rating. A rating is its answer's first number, clamped
  to 1-10; an answer without one keeps its report unrated (as a failed
  rating does), is recorded in `degraded_error` and counted in
  `global_unrated_report_count` (new search metadata) (#567).
- Under `processing.ignore_errors`, a fatal model error (no access to the
  model) in global search's map or reduce step degraded to an answer from
  the unrated reports; it fails the query, as in DRIFT and the dynamic
  selection. An OpenSearch search of a missing index whose ingested-suffix
  check could not reach the store skipped the index; a store that cannot
  be reached fails the query there too (#567).
- A query was translated into `processing.translation.target_language`
  when the translation stage was disabled and the languages differed, so a
  corpus indexed untranslated was searched in another language (one extra
  model call a query, and lexical recall lost). The query is translated only
  when ingestion translates, or when the caller asks for a language
  (`RAGInput.target_language`) (#567).
- DRIFT's primer answers reached the answer model, but the community
  reports they were written from did not, so an answer resting on a primer
  answer cited no source for it. Each primer fold's reports are its
  answer's evidence now, fused and reported with the actions' local search
  results (#567).
- With the output guardrail check on (`check_output`), the model-written
  search results (DRIFT's intermediate answers, global search's key points
  or summary, marked `synthesized`) were returned in `search_results` by
  `ainvoke` and `asearch` unscreened. Each is screened as the answer is (one
  ApplyGuardrail call each): masked, or dropped when refused (#567).
- MMR diversity (`search.fusion.diversity_lambda` below 1) measured
  redundancy over whitespace words, and an unspaced Japanese or Chinese
  passage is one: a near-duplicate differing by one appended word shared
  nothing with its original and was kept over a distinct passage. A word
  holding a Han, Hangul or Kana letter is compared by its character bigrams
  now; spaced text is compared as before (#567).
- Weighted fusion (`search.fusion.method: weighted`) scored a stream's only
  hit (or hits of equal score) 0.5, so the best hit of a one-hit stream
  ranked below another stream's best (a LightRAG linked chunk below the
  vector chunks' top one); it scores 1.0, as the hybrid search's own
  normalization does (#567).

### Security
- Pinned floors for advisories: pypdf 6.21.4 (seven high-severity
  denial-of-service advisories through the PDF parser, for example
  GHSA-5849-834g-hvw5), filelock 4.2.1 (nine advisories,
  GHSA-v4x8-3wr5-3wq8 high; an override lifts `unstructured`'s cap; locked
  at 4.2.5), Pillow 13.0.0 (GHSA-p4wg-p73j-c528 and others),
  `unstructured>=0.24.1` (GHSA-4mvj-m6j5-pmf7,
  GHSA-8mgp-746c-j5xp, a stored XSS GHSA-v5mq-3xhg-98m9) and
  `langchain-openai>=1.1.14` (GHSA-r7w7-9xr2-qq2r), pyarrow 26.0.1
  in the `eval` extra (CVE-2026-92287; this package reads no Parquet
  files, so it was not reachable) (#580); the lock, not a
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
  delimiter tags (#179, #182). The community report's claim block was
  open: its `<claim_data>` tags are part of the value, which the chain's
  neutralising does not cover, so a claim's text could close the block
  and write instructions after it. Each claim line is neutralised for its
  block now (#565).
- Importing the package no longer loads a `.env` file; the CLIs read only
  `./.env` and never override the environment (#224).
- ragas sends no usage telemetry unless `RAGAS_DO_NOT_TRACK=false` (#361).
- `pipeline_id` must be a single path segment; metadata filters are
  documented as relevance filters, not access control (#173).
- Interactive graph tooltips are escaped (#180).
- The compose containers drop all capabilities and run with
  `no-new-privileges`, the data-plane image declares `HEALTHCHECK NONE`, and
  retry jitter draws from `random.SystemRandom` (#307).

[Unreleased]: https://github.com/awslabs/unified-kg-rag-on-aws/commits/main
- The ASH scan (bandit, semgrep, detect-secrets, checkov, cdk-nag,
  npm-audit) gates pull requests and pushes: an unsuppressed finding at
  MEDIUM or above fails the `security` workflow. Each suppression carries
  its reason, on the line (`# nosec`, `# nosemgrep`, `# pragma: allowlist
  secret`) or path-scoped in `.ash/.ash.yaml` (tests, local-mode Gremlin
  URL prose, token-count log lines). Dependabot waits 7 days after a release
  before proposing it (security updates still open at once), and the LLM
  XML parser never resolves DOCTYPE entities, whatever lxml's default
  (#578).
