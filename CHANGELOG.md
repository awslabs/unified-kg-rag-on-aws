# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first release after 0.2.3, the initial public snapshot (commit `81ea498`,
never tagged), which ran GraphRAG and LightRAG retrieval on one Amazon Bedrock,
Neptune, OpenSearch and DynamoDB stack. This release adds HippoRAG 2 and RAPTOR
retrieval, methodology profiles that build only what the selected methodologies
read, and a local mode that runs without an AWS account. Every entry compares
with 0.2.3. A fix to a feature added in this same cycle is folded
into that feature's entry, so *Fixed* lists only bugs 0.2.3 had. Nothing 0.2.3
stored is read again: follow *Upgrading* in order.

### Highlights
- Four retrieval methodologies on one stack, selectable at index time (`indexing.methodologies`) and per query:
  GraphRAG (`drift`/`global`/`local`/`simple`), LightRAG (`mix`/`hybrid`/`naive`), HippoRAG 2's Personalized
  PageRank (`ppr`) and an opt-in RAPTOR summary tree (`tree`); a namespace builds only what its methodologies read
  and refuses a strategy it cannot serve (#596, #597, #599, #602).
- `local` orders its text units by HippoRAG 2's PageRank, and `auto` routes multi-hop questions to `iterative`
  follow-up rounds over `mix`: both defaults are measured (#594, #598, #602).
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
- Answers cite their evidence with `[n]` markers that name the documents and PDF pages behind each number, by default
  every strategy's but `global`'s and `drift`'s (#627).
- Structured graph queries: counts, rankings, lists and paths compiled from a checked JSON plan and run read-only
  by the graph store, opt-in (#630).
- A read-only MCP server for agents and a local web console, both opt-in (#616, #631).
- A query deadline and always-on LLM usage accounting (#522, #526).
- Evaluation: LLM-free evaluators, paired strategy comparison with cost and bootstrap intervals, a pairwise judge
  and a MuSiQue preparer (#121, #324, #526); declines on unanswerable questions, an assertion judge,
  length-controlled pairwise judging and a closed-book arm, all opt-in (#593, #595, #600).
- OpenTelemetry tracing over OTLP, Langfuse or Arize Phoenix (#207, #208).
- Claude Sonnet 5.5 and Haiku 5.5 are the default tiers (#117, #170).
- The CDK stack gains a read-only query role, alarms and a one-task Step Functions ingestion (#157, #211, #319,
  #328).
- A core install is about a third of the size: evaluation, visualization, local mode and others are extras (#467,
  #576).

### Breaking changes

#### Install and extras
- Python 3.11 or 3.12 is required (`>=3.11, <3.13`); 3.10 is dropped (#430).
- A core install (`uv sync`, `pip install -e .`) is ingestion and queries: 145 to 78 packages, 937 to 335 MB on Linux.
  Extras: `eval` (ragas, datasets, pandas, langchain, langchain-community, langchain-openai), `viz` (pyvis, for the
  `interactive` renderer, which a `run-visualization` without `--renderers` leaves out with a WARNING without it),
  `local`, `cross-encoder` (the local reranker, with PyTorch), `otel`, `unstructured` (the richer Markdown and HTML
  parser), `docx` (Word files), `ontology` (rdflib), `mcp` (`run-mcp-server`), `console` (`run-console`: FastAPI,
  Uvicorn, python-multipart). A feature whose extra is missing fails naming it and its install (from a checkout `uv sync
  --inexact --extra X`, as `uv sync` removes the extras it is not given; from a release wheel `pip install '<wheel
  file>[X]'`): the CLIs exit 2 before any work, the library raises `ConfigurationError` (also under
  `processing.ignore_errors`). The LLM-free evaluators and the `static` renderer need no extra; the container image
  installs none (`--build-arg UV_EXTRAS="--extra eval"` adds the ragas judge) (#467, #472, #520, #576, #636).
- No longer dependencies: `graspologic` (Leiden runs on `graspologic-native`), `bokeh` (with `contourpy`, `tornado`,
  `xyzservices`), `umap-learn`, `scikit-learn`, `nest-asyncio`; `langchain` comes only with `eval` (#225, #226,
  #465, #520, #576). Security floors follow the package that pulls the pinned one in: `pillow` in `eval`, `llvmlite`
  in `unstructured`, `langchain-openai` in `eval` and `local`.
- The `static` renderer writes its three charts (file names unchanged) as self-contained HTML with an inline SVG
  instead of Bokeh documents: no script, a tooltip on every bar, no pan or zoom, no embedded Bokeh JSON (#576).
  Ingestion renders nothing: see *Configuration*.

#### Configuration
- Unknown keys, in the file or as `UKR__` variables, fail the load, and `Config.model_validate` refuses them too;
  `--allow-unknown-keys` (every CLI) or `allow_unknown_keys=True` (`get_config`, `ConfigLoader`) warns and drops
  them instead (#335). A moved key fails naming its replacement (`processing.chunk_concurrency` and
  `processing.batch_size` are `processing.max_concurrency`, one pool of in-flight calls, #296;
  `indexing.cross_run_fuzzy_merge` is `processing.fuzzy_entity_resolution`, #214), a removed key loads with a
  WARNING and is ignored: the generated [Moved
  keys](docs/reference/configuration.md#moved-keys) and [Removed keys](docs/reference/configuration.md#removed-keys)
  tables list them (#334, #344, #467, #468, #470, #577). [`config-template.yaml`](config-template.yaml) is a short
  starting file; every key is in the [configuration reference](docs/reference/configuration.md) (#468).
- Models: the 0.2.3 per-stage `*_model_id` keys moved to one `models` section: the tiers `models.default` and
  `models.fast`, `models.roles.<role>` (a model id or a tier name), `models.embedding` (model id and dimension, for the
  indices, queries, RAGAS and the visualization) and `models.rerank`. `providers.<llm|embedding|rerank>.kind` picks each
  family's provider, and `--llm-model`, `--embedding-model`, `--embedding-dimension` set the new keys; a setting of a
  provider the config does not select (after the CLI's provider flags) is ignored with a WARNING; under `backend: local`
  another embedding model needs its dimension too (#334). See [models](docs/reference/models.md).
- Graph expansion's settings moved from `indexing.neptune` to `search.graph_expansion`, as every graph store reads them
  at query time (the embedded SQLite graph too): `max_hops`, `max_results_per_hop`, `entity_importance_source`,
  `traversal_fetch_multiplier`, `deterministic_order`.
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
  - No replacement: `graph.community_detection.report_generation.include_statistics` / `include_key_entities`
    (#214); `processing.gleaning.quality_threshold`, `convergence_threshold`, `convergence_change_scale`,
    `min_improvement_threshold`, `quality_completeness_weight`, `initial_quality_entity_scale`,
    `initial_quality_relationship_scale` (gleaning stops per unit once it adds nothing new, #171);
    `indexing.neptune.min_entity_importance` (#155); `indexing.opensearch.hybrid_search_pipeline_name` (hybrid
    search fuses in the retriever, #306); `indexing.opensearch.terms_batch_size`, `max_total_clauses`,
    `reserved_clauses` (a list filter is one `terms` query; more values than `max_terms_count` raise
    `InvalidFilterError`, #258); `processing.chunking.content_type` (plain text, #289).
- Removed settings that change behaviour where you had set them (each loads with a WARNING):
  `search.reranking.enabled: false` no longer turns reranking off (set `providers.rerank.kind: none`);
  `search.fusion.method: weighted` fuses by RRF (weighted fusion is gone); `fixing.enabled: false` no longer stops
  output fixing; `cache.chunking` (an entry over 1,000 items or 50 MB is always chunked);
  `indexing.opensearch.persist_embedding_cache` (the S3 embedding cache is gone: an incremental run embeds only its
  delta) (#344, #468, #577); `graph.community_detection.auto_resolution` (on by default in 0.2.3),
  `auto_resolution_candidates` and `auto_resolution_max_nodes` no longer sweep resolutions for the most modular one,
  which merged unrelated communities on the upper levels and lost global search accuracy in testing (set
  `graph.community_detection.resolution`); `aws.dynamodb.billing_mode` (a table the library creates is on-demand,
  the only mode that worked; provision another yourself with `aws.dynamodb.create_table_if_missing: false`).
- `aws.opensearch.sigv4_service_name` takes only `es`: `aoss` (OpenSearch Serverless) fails the load, as its vector
  collections assign their own document ids and have no refresh API, which the indexes need.
  `evaluation.ragas_metrics` takes only the five metrics ragas scores (`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`); another fails the load instead of the evaluator.
- Visualization: ingestion only writes `visualization_data.json` (`graph.visualization.export_data`, default
  `true`), at the end of every run that indexed, from the namespace's stores once the run gave up the namespace's
  lease, and prints its path; a run that wrote nothing keeps an export naming the build the stores hold (`built_at`),
  local or uploaded by the S3 cache sync from the run the build record names (`visualization_export_run`; one that
  cannot be read is written again), so a no-change run at 10k text units, a deployed task's on an empty disk too,
  no longer rewrites a 59 MB file;
  `run-visualization` renders it (without `--data-path` the newest export, of `--suffix`'s namespace when given)
  with a seeded spring layout, and the embedding layouts (UMAP, t-SNE, PCA) are gone.
  `graph.visualization.interactive` and `static` are typed sections: a misspelled setting is an unknown key (#344,
  #467). See [visualize](docs/guides/visualize.md).
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
- Fusion buckets: a simple search fuses one per index (`simple_text_units`, `simple_entities`, `simple_relationships`,
  `simple_claims`, `simple_community_reports`, `simple_summary_nodes`); a weight on its former `opensearch_all`, on
  `opensearch_community_reports`, or on 0.2.3's `opensearch_candidate_community_reports` and
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
| `processing.chunking.min_chunk_size` / `max_chunk_size` / `fallback_chunk_size` | 5000 / 50000 / 50000 characters | 250 / 2000 / 1200 tokens (`length_unit: tokens`; about 1000 / 8000 / 4800 English characters) (#125) |
| `processing.max_concurrency` | 20 | 40, 20 under `backend: local` (#310) |
| `processing.similarity_threshold` | 0.6 | 0.85; similarity matching is opt-in (`processing.fuzzy_entity_resolution`) (#214) |
| `indexing.neptune.property_max_length` | 1000 | 4000 (#132) |
| `indexing.cross_run_merge` | `false`: an incremental run overwrote the stored entities and relationships with its delta's | removed (loads with a WARNING): an incremental run always merges its delta into the stored graph (#344) |
| `indexing.max_deletion_share` | no guard | 0.5: an incremental run deleting more of its scope asks first (#518) |
| `indexing.opensearch.language_analyzers` | `en: english`, `ko: nori`, others `standard` | also `de`, `es`, `fr`, `it`, `pt`, `ru`, `ar`, `hi` stemmers, `th: thai` and `ja`/`zh: cjk` bigrams (#574, #663) |
| `graph.community_detection.auto_resolution` | `true` | removed (loads with a WARNING): `resolution` applies (#141) |
| `graph.visualization.outputs_directory` | `outputs/visualization` | `null`: `<cache directory>/<pipeline_id>/visualization` |
| `search.token_manager.max_context_tokens` | 200000 | 30000; 22000 under `backend: local` with a non-Bedrock LLM (#125, #224, #226) |
| `search.fusion.fusion_weights` | 10 entries of 1.0 | `{}`, every bucket 1.0 (#258) |
| `search.global_search.relevance_threshold` / `use_dynamic_selection` | 0.5 / `true` | 0.2 / `false` (#318) |
| `search.auto_routable_strategies` | fixed: `simple`, `local`, `global`, `drift` | `local`, `mix`, `global`, `iterative` (#347, #598) |
| `search.query_timeout_seconds` | no deadline | 180 s; `null` turns it off (#522) |
| `search.model_timeout_seconds` / `model_max_attempts` | the ingestion client's 300 s, 6 requests | 90 s / 2 for query-time calls (#522) |
| `search.auto_speculative_extraction` | extraction after the router | `true`: alongside the router call (#525) |
| `search.output_ceilings` | the model's maximum output | 1024 for routing, 4096 for query extraction and relevance ratings (#525) |
| `search.answer_prompt_style` | every strategy concise | `{global: comprehensive}`, merged under your map (#533) |
| `search.global_search.evidence_tokens` | every cited report the budget holds | 4000; `null` removes the cap (#533) |
| `search.local_search.text_unit_ranking` (local's text-unit order) | by how many expanded entities cite a unit (no setting) | `ppr`, HippoRAG 2's PageRank; `support` is the earlier order, which DRIFT's follow-ups keep (`search.drift_search.text_unit_ranking`) (#597, #602, #607) |
| `evaluation.enabled_evaluators` | `langchain`, `ragas` | `answer_match`, `retrieval`, `graph_aware`; `ragas` is opt-in, `langchain` is removed (#121, #576, #620) |
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
- `run-eval --verbose` (`-v`) is removed: use `--log-level DEBUG`, which every CLI takes (`run-rag --verbose`
  prints more of each result). Either spelling fails with exit 2 saying so.
- `run-rag --use-memory` is removed (an `--interactive` session always keeps its memory and a single `--query` has
  none), and `run-rag` / `run-eval --enable-thinking` (the default Claude 5.x models always reason: tune their depth
  with `aws.bedrock.default_effort` / `fast_effort`; from Python, `RAGInput.enable_thinking` still turns thinking on
  for a model that opts in). Either flag fails with exit 2 saying so; run-eval results no longer record
  `enable_thinking` (#628).
- Exit codes: `2` for a usage or configuration error before any work (also one met while running or while `run-rag`
  builds its query chain), `1` for a failed run, `130` for Ctrl-C, `143` for SIGTERM (#333, #344).
- `--output-format json` prints one JSON document on stdout (a status document on failure, argparse errors included)
  and everything else on stderr; `run-ingestion` rejects it with `--metrics-sink cloudwatch` (#333, #359, #366,
  #420).
- JSON failures have one schema: `status` (`failed`, or `interrupted`), `exit_code`, `error` and a new `error_type`
  (the exception's class, `UsageError`, `FailureBudgetExceeded`, `RegressionGateFailed`, `RunFailed` or
  `Interrupted`), in a status document and in a result that shows a failure: a failed `run-rag` query adds them to
  `success: false` (it had only `error`), and a failing `run-eval` or `run-visualization` result has them (#629).
- `run-rag --output-format json`: an answer's `search_results` leaves out `results`, which repeated its `sources`;
  `--verbose` keeps them (#629).
- `run-eval --search-strategy` takes a list (`["auto"]` by default); with several strategies or `--repeats` it
  overrides an item's `search_strategy`; `abstention_statistics.unanswerable` has the overall block's shape
  (`completed`, `abstained`, `answered`, `abstention_rate`) (#292, #324, #600).
- `run-eval` writes each invocation into its own `<outputs>/run_<timestamp>_<id>/` directory (0.2.3 wrote
  `evaluation_results_<timestamp>.json` and its siblings into the outputs directory itself), and the result, report
  and summary file names carry the strategy when a run used one (`evaluation_results_<strategy>_<timestamp>.json`,
  the timestamp with a random suffix); the JSON output's
  `output_directory` names the run directory, and `evaluation.run_subdirectories: false` keeps the flat layout
  (#121, #620). See [evaluation output](docs/reference/evaluation-output.md).
- `pipeline_metadata.json`: the `document_parsing` stage's `success_rate` is a fraction (0.5), as every other rate
  there, not a percentage (50).
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
  matching is opt-in). Legal forms (the language profiles, `domain/languages/legal_forms.py`): trailing Inc, Corp,
  Co, Ltd, LLC, LLP, LP, PLC, Pty, Pvt, Pte, GmbH, AG, KG, UG, SA, SAS, SARL, S.L., SRL, S.p.A., Ltda, S/A, BV,
  NV and KK in any dotting, as written or without all their accents ("Cong ty" is "Công ty"; a word is not read
  without its own: "Marcelo Sá", "Khu Phố Cổ" keep it), forms of several words (`GmbH & Co. KG`, `S.A. de C.V.`,
  `S.A.B. de C.V.`, `Private Limited`), Russian, Arabic, Hebrew, Hindi and Thai forms, those written before the name
  (`ООО`, `Công ty TNHH`, `شركة`, `บริษัท`) and the Korean, Japanese and Chinese company forms leading or trailing
  ("(주)", "合同会社", "股份有限公司"), so "Acme GmbH" and "Acme" of one type are one entity; not after "&"/"and" nor an
  article alone, and a form that is also a unit symbol not after a number ("500 kg"; "Channel 4 Ltd" is "Channel 4").
  Entity keys, which ids hash, drop the quote marks of every script («», „“, 「」, 《》), what is typed for an apostrophe
  (´, ʼ, ′; not the ʻokina) and the invisible format characters (soft hyphen, zero-width space, direction marks; not
  the joiners), read every dash as a space but a minus sign opening the name ("−40 °C" is not "40 °C"), drop a middle
  dot between Han, Hangul or Kana letters (and Korean's "ㆍ" between syllables), the Arabic and Hebrew vowel marks and
  a trailing "。" ("ジョン・スミス" is "ジョンスミス"); a key is its own key, and the fuzzy matcher reads a name's identifier
  tokens from its compact key, cut where a word passes between a Han, Hangul or Kana letter and another character
  (a dotted "한·빛" has none, as "한빛"; "제3조" and "제 3 조" both 제, 3 and 조, so spacing variants with a number are
  one entity, where they were two), and a word with a Han numeral is an identifier ("東京第一工場" never links
  "東京第二工場"). Of spellings with one key, the one its key
  folds least names the entity on a tie ("토지", not "「토지」"). Entity and relationship `rank` come from node degree,
  and a relationship's weight sums its strengths per text unit (#189, #214, #574).
- Entities have no `MemberOf` edges to communities; an entity lists its `community_ids` (#348).
- Hidden files and files under a hidden or `__pycache__` directory of the corpus are not parsed (#213).

#### Deployment (CDK)
See [`iac/README.md`](iac/README.md).
- `iac/` is a uv project (`cd iac && uv sync`; `requirements*.txt` are gone) and needs AWS CDK CLI 2.1146.1 or later
  (#251, #438).
- Ingestion is one Fargate task (state `Ingest`, result under `$.task`; was four phases and `$.last_phase`), and its
  task definition lives in `GraphRagOrchestration`: a deployed stack upgrades in three deploys (*Upgrading*) (#328,
  #558, #626).
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
- The OpenSearch domain runs 2.19 (was 2.13), and the local stack (`docker/compose.local.yaml`) 2.19.7. A 0.2.3 domain
  lacks the policy that makes a version change an in-place upgrade, so the stack deploys of the upgrade keep it on
  2.13 (adding the policy while changing the version could replace it with an empty domain) and it upgrades in place
  after them (same endpoint and indices, no reindex, a blue/green deployment of 15 minutes to several hours with
  slower queries and ingestion, and no downgrade; *Upgrading*). The application's defaults behave the same on both
  versions and it still runs on 2.13 clusters; `vector_mode` / `compression_level` and faiss with `cosinesimil` now
  work on the CDK domain, opt-in as before (#457, #622, #626).
- The app container runs on a read-only root filesystem, writing only to task volumes at `/tmp`, `/app/cache`,
  `/app/outputs` and `/app/logs` (Security Hub ECS.5), as does the trace collector sidecar
  (`enable_cloudwatch_tracing`) with a `/tmp` volume of its own, and a created VPC's default security group allows no
  traffic (EC2.2). An image of your own must declare those four paths as `VOLUME`s its user can write, as
  `docker/Dockerfile` does, and write nowhere else.

#### Library API
- 0.2.3 exported only `__version__`. The public API is what `unified_kg_rag` exports (`unified_kg_rag.__all__`):
  `Config` and `get_config`, `DataIngestionPipeline`, `GraphRAGChain` with `RAGInput` / `RAGOutput` /
  `SearchOutput`, `MemoryManager`, `EvaluationManager`, the errors they raise, and the extension surface (ports,
  `Providers`, the provider, strategy, parser, renderer, schema-source and seed-source registries; a failed model
  call is a `ModelError`: `LanguageModelError`, `EmbeddingModelError` or `RerankModelError`, #657) with every type
  their signatures name (the stored items, `SearchQuery`, `RetrievalResult`, `IndexBuildRecord`, `DocStatusRecord`,
  `RenderContext`, `TransientRetryConfig`, ...), and `load_seed_graphs` / `resolve_extraction_schema`. Import from the
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
  / `end_rebuild`, `check_index_creation`, `holds_any_item`, `holds_any_text_unit`, `update_relationship_ranks`,
  `repoint_report_findings`. Full-build `index_*` methods
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
  query, and with `check_output` a stream delivers the answer as one chunk after the check. The baseline guardrail
  `iac/` creates masks email, phone and card numbers in the query too (its PII settings acted on answers only; the
  next deploy publishes a new version). A role outside the CDK stack needs `bedrock:ApplyGuardrail` (#104, #319,
  #344).
- Answers of every strategy but `global` and `drift` carry `[n]` citation markers by default
  (`search.citations.enabled`, `search.citations.strategies`): their text and the judges' scores are not comparable
  with 0.2.3 runs; set `enabled: false` for the earlier context and prompt (#627).
- A query with conversation history is rewritten as a standalone question (`models.roles.query_rewrite`) that every
  step reads (#212).
- A filter key no retriever of the strategy declares raises `InvalidFilterError` (#123); a suffix nothing was
  ingested under raises `IndexNotFoundError` (#178).
- A `name` filter matches an entity's whole name exactly on every store (`name.keyword`; OpenSearch matched one
  analyzed token, so `name:Acme` matched nothing and a partial Korean name matched), and a filter on analyzed text
  (`description`, `text`, `summary`, the relationship and claim names, ...) or on the seed properties raises
  `InvalidFilterError` (#661).

### Upgrading
From 0.2.3; the sub-list after it is for a `main` checkout since. The runbook's
[Upgrading the library](docs/operations.md#upgrading-the-library) links here.
1. Install on Python 3.11 or 3.12 with the extras you use (`eval`, `viz`, `local`, `cross-encoder`, `otel`,
   `unstructured`, `docx`, `ontology`, `mcp`, `console`). A CLI whose configuration needs a missing extra exits 2 at
   start-up naming it (#472). Markdown and HTML files indexed with `unstructured` keep their text and ids only with
   that extra installed: without it the core readers write other text, so the next incremental run re-extracts
   those files as changed documents (#628).
2. Load your config once and fix what it reports: unknown keys (*Configuration* lists the 0.2.3 ones), moved keys
   (the load names the replacement) and removed keys (a WARNING names it). Rename environment variables to
   `UKR__...`.
3. Review the settings whose behaviour changed: set any 0.2.3 default you rely on (*Defaults*); replace
   `search.reranking.enabled: false` with `providers.rerank.kind: none` (otherwise a Bedrock Rerank call per query),
   and expect RRF ordering where you set `search.fusion.method: weighted` (#577). With
   `graph.community_detection.report_generation.enabled: false`, `global` and `drift` queries are refused (exit 2,
   naming the switch) and `auto` leaves them out; `local` answers as before (#612).
4. A custom `fusion_weights.lightrag_linked_chunks` weight: set `lightrag_relation_chunks` too (usually the same
   value) to keep weighting every LightRAG graph chunk; mix and hybrid recall are sensitive to it (#521, #530).
   Drop an `opensearch_all` weight (simple's buckets are per index: `simple_<index>`), `opensearch_community_reports`,
   `opensearch_candidate_community_reports` and `opensearch_expanded_community_reports` (#584, #588).
5. A role with a custom IAM policy needs `dynamodb:BatchGetItem`, `BatchWriteItem`, `Query` and `DescribeTable` on
   the registry table (`UpdateTimeToLive` where the library creates it), `bedrock:ApplyGuardrail` on the guardrail,
   and `bedrock:CountTokens` with `InvokeModel`. With `bedrock_model_arns`, list the reranking model too: the Rerank
   API checks `bedrock:InvokeModel` on it (#191, #194, #298, #319, #327, #331, #566).
6. CDK: install AWS CDK CLI 2.1146.1 or later, run `cd iac && uv sync`, and deploy the stacks with the new image
   (they share the environment names, EMF dimension, query task and registry table). Add `-c sts_endpoint=true` when
   a private-mode image config sets `aws.bedrock.assumed_role_arn` (or the role assumption hangs), and `-c
   opensearch_nori_package=true` for Korean on the CDK domain (or keep its `cjk`) (#282, #459, #566). Before the
   deploys of step 7, set `OPENSEARCH_VERSION = opensearch.EngineVersion.OPENSEARCH_2_13` in
   `iac/stacks/storage_stack.py`: a 0.2.3 domain has no `EnableVersionUpgrade` update policy, and a deploy that adds
   it and changes the engine version at once is not covered by it, so CloudFormation can replace the domain with a
   new, empty one (step 9 upgrades it) (#457, #622, #626).
7. A deployed stack, in three deploys with your usual `-c` flags and the new `image_tag`, and `-c
   legacy_doc_status_table=<the 0.2.3 registry table>` (`graphrag-doc-status` in `dev`, `<env>-graphrag-doc-status`
   otherwise, or the `doc_status_table` you set) on the first two: `cdk deploy GraphRagOrchestration --exclusively`
   (`GraphRag<Env>Orchestration` outside `dev`), then `cdk deploy --all`, then `cdk deploy --all` without
   `legacy_doc_status_table`. The registry table is replaced (another key schema and name), and the 0.2.3 compute
   and observability stacks import its ARN and name: without the setting the storage stack fails ("Cannot update
   export ...DocStatusTable... as it is in use") and rolls back unchanged; without the first deploy `GraphRagCompute`
   stops at "Cannot delete export ...". The old table is retained where `removal_destroy` is off (delete it in step
   12) and deleted with the stack update where it is on. A 0.2.3 `doc_status_table` of your own needs a new name
   too, as CloudFormation replaces no custom-named table under its own name. Container Insights series restart
   under the renamed ingestion task family (#558, #626).
8. On those deploys: pass `-c owner=aws-proserve` to keep the `owner` tag; without `use_cmk`, delete a leftover
   `GraphRagSecurity` stack (CDK metadata only) with `aws cloudformation delete-stack`; dry-run the TLS policy
   change (`aws opensearch update-domain-config ... --dry-run`) to see whether it runs blue/green; in a
   `removal_destroy` environment copy the `CacheAccessLogs` logs you need, as the bucket is deleted (#566, #583).
   With `opensearch_count` above 1, also pass `-c opensearch_master_instance=` the domain's `opensearch_instance`
   (`r6g.large.search` unless you set one; keep it on later deploys): 0.2.3's dedicated masters ran the data node
   type, the default is now `m6g.large.search`, and changing them is a blue/green deployment (every shard copied,
   slower for hours on a large domain) (#641).
9. OpenSearch 2.19: put back `OPENSEARCH_VERSION = opensearch.EngineVersion.OPENSEARCH_2_19` and upgrade the domain
   as the runbook's [Upgrading OpenSearch](docs/operations.md#upgrading-opensearch) says: stop ingestion, take a
   manual snapshot (the only way back to 2.13), run the eligibility check, and deploy `GraphRagStorage` once `cdk
   diff GraphRagStorage` changes only `EngineVersion` (no `UpdatePolicy` added). It takes 15 minutes to several
   hours with slower queries and ingestion, so run it before the rebuild of step 11. To stay on 2.13 for now, keep
   the pin: the application runs on both (#622).
10. Run queries with `QueryTaskDefinitionArn` and keep evaluation data under `s3://<cache bucket>/eval/` (#319).
11. Rebuild every index: one ingestion with `indexing.reset: true` and the registry on, in the new table, with
    `--force-rebuild` or a new `--pipeline-id` (never a 0.2.3 one). It costs a first build: every text unit is
    extracted (and gleaned) again and every vector embedded (#327, #331, #332).
12. After the rebuild succeeds: delete the old registry table, the OpenSearch pipeline 0.2.3 wrote (`DELETE
    _search/pipeline/graphrag-hybrid-search-pipeline`) and `s3://<bucket>/embedding-cache/` if you used it. Delete
    the three log groups 0.2.3 created that the stacks no longer manage (CloudFormation retains them, also with
    `removal_destroy`): the VPC flow-log group (logical id `VpcFlowLogsLogGroup...` in `GraphRagNetwork`, with
    `vpc_flow_logs`) and the OpenSearch slow-search and slow-index log groups (`OpenSearchSlowSearchLogs...` and
    `OpenSearchSlowIndexLogs...` in `GraphRagStorage`; `GraphRag<Env>...` outside `dev`), after copying what you
    keep: `aws logs describe-log-groups --log-group-name-prefix GraphRagNetwork-VpcFlowLogsLogGroup` (and
    `GraphRagStorage-OpenSearchSlow`) names them, `aws logs delete-log-group --log-group-name <name>` deletes each
    (#641).
13. Scripts: update the renamed flags, exit codes and JSON output (`run-eval --verbose` is `--log-level DEBUG`),
    `.States.PrepPhase` to `.States.Ingest` and `$.last_phase` to `$.task`, `indexing_results` keys to `vector_*` /
    `graph_*`, read the `document_parsing` stage's `success_rate` as a fraction (0.5, not 50), and read `run-eval`
    results from the run directory the JSON output's `output_directory` names (#328, #333, #584, #620).
14. Evaluation: list `ragas` in `evaluation.enabled_evaluators` (or `run-eval --evaluators ...,ragas`) to keep its
    scores, with the `eval` extra; a `--fail-if ragas/...` gate fails without it. The `langchain` evaluator is
    removed (a config naming it, or `evaluation.langchain_metrics`, loads with a WARNING and without it): gate on
    `ragas/answer_correctness` instead of `langchain/correctness`. `graph_aware` coverage matches by entity key and
    can read lower than before on the same answers: run baselines again before comparing. Re-render static charts
    to get the SVG pages (#563, #576, #620). The container image leaves the `eval` extra out: a query task that runs
    ragas needs an image built with `--build-arg UV_EXTRAS="--extra eval"` (about 390 MB more) (#636).
15. A Python script that runs ingestion (`DataIngestionPipeline`, `DirectoryLoader.deduplicate_documents`, the
    graph resolver) must keep its entry point behind `if __name__ == "__main__":`: the deduplication and fuzzy
    matching worker processes now start from a forkserver, which imports the main module, instead of forking a
    process that runs threads (a lock another thread held could hang the worker). The console scripts and a
    notebook need nothing; what a custom pool task gets must pickle (#628).

16. A `custom_prompts` override of `strategy_selection_*` should answer the strategy name alone, as the built-in
    prompt does, or `{"strategy": ...}` (*Changed*): another answer costs one output-fixer call per query. No stored
    data is re-keyed; the query-extraction cache keys on the prompts' text, so only its in-memory entries of changed
    prompts are re-extracted.

From a `main` checkout (formats and keys no release had):
- Local mode: a `backend: local` config that relied on the preset for the stores gets the embedded SQLite stores; add
  `providers.graph.kind: gremlin` and `providers.vector.kind: opensearch` (as `docker/config.local.yaml` does) to keep
  the containers (#273). Its reranker needs the `cross-encoder` extra: with `local` alone queries run with reranking
  off (#628). The preset turns the SQLite doc-status registry on, so runs are incremental: stores a run built with it
  off make the next run exit 2 until one run with `indexing.reset: true` rebuilds them (a first build's cost, less
  what the LLM response cache holds); `providers.doc_status.enabled: false` keeps full reloads.
- #622 OpenSearch 2.19: the next deploy of `GraphRagStorage` upgrades a 2.13 domain deployed from #457 on (it has
  the update policy; `cdk diff GraphRagStorage` must change only `EngineVersion`) in place; a domain deployed
  before #457 follows steps 6-9 above. It takes 15 minutes to several hours, queries and ingestion run slower
  meanwhile, and the only way back to 2.13 is a manual snapshot restored on a new domain: stop ingestion and take
  one first ([Upgrading OpenSearch](docs/operations.md#upgrading-opensearch)). Nothing is re-indexed or re-run. To
  postpone, set `OPENSEARCH_VERSION` back to `OPENSEARCH_2_13` before deploying. Local stores: `docker compose -f
  docker/compose.local.yaml up -d --build --wait` rebuilds the OpenSearch image and restarts it on 2.19 with its
  data volume, whose indices it reads as they are.
- `aws.bedrock.default_max_output_tokens` is `models.default_max_output_tokens` (the load fails until renamed;
  nothing re-runs). The pre-release `search.llm_retry`, `providers.llm.max_retries`,
  `providers.embedding.max_retries` and `evaluation.ragas_max_retries` are unknown keys: set `retry` (#216, #584).
- Filters: an index built before #360 (relationship and claim filters), #370/#376 (a removal clears its document's
  values), #446 (`attr_*` keywords) or #566 (numeric and boolean values as keywords on OpenSearch; a WARNING names
  each such field, and new fields map without a reset) needs one run with `indexing.reset: true`.
- Stored items: an index built before #508 (seed properties unmapped; such an entities index is refused, exit 2),
  #562 (relationships store their type, else `Relationship: A -> B`) or #383/#410 (removals written ahead) needs the
  same reset; for a deleted document restored mid-removal, deleting its registry record is enough.
- Analyzers: a namespace indexed in a language before the default `language_analyzers` named it (`de`, `es`, `fr`,
  `it`, `ja`, `pt`, `ru` and `zh` before #574; `nl`, `sv`, `tr` and the other built-in analyzers before #671) keeps
  the `default_analyzer` its fields were built with, one WARNING per run naming them, until one `indexing.reset: true`
  run gives them the language's stemming and stop words, also with `language_analyzers` set (as the CDK stack sets
  `ko`) while it gives the language the default's analyzer, decided per field by the language it holds (another
  language set to the same analyzer, `af: dutch` beside `nl`, keeps them too). A language it gives another analyzer
  than its fields hold stops the run before any model call (exit 2, `was built for other language settings`): reset,
  or set the map to what the fields hold.
- Stored edge ranks: a vector copy's rank a run killed between the stores left (before the rank was compared) is set
  by the next run deriving one of the edge's endpoints, or by a reset; a `pending-commit` note's `edge_entity_ids`
  (#694) is ignored.
- Re-keyed once before 0.3.0: entity ids and name keys fold quotes, apostrophes, invisible characters, dashes (not a
  leading minus), dense-script middle dots and Arabic and Hebrew marks and strip more legal forms, and dense-script
  spacing variants with a number are one entity (*Stored data and reindexing*); the description summary and claim
  prompts changed, claims keep a `suspected` status and are keyed by their statement (*Fixed*); chunk sizes count
  tokens (*Defaults*), which keeps English text units and re-chunks Korean, Japanese and Chinese ones; stage-cache keys
  hold the prompts in effect (a pipeline id resumed across a prompt change starts over); extraction records under
  earlier keys (other description summary settings, #541-#556) and claims under an earlier reading of their dates
  (#612, #624) are no longer looked up. Run one reset (`indexing.reset: true`) with a new or no `--pipeline-id`:
  incremental runs keep stored entities under their earlier ids beside the new ones. That one reset covers this item
  and every item below, and its cost depends on the index's age. On an index built with #565 or later it reuses the
  stored graph extraction and gleaning records and the vectors (entity and relationship ids are minted again from the
  names, as extraction mints them) and asks the model one summary, and one embedding, per entity or relationship over
  the summary threshold, every community's report (the report prompt cites sub-community reports now), with claim
  extraction on every unit's claims (one call per unit that states an entity), with the `intelligent` chunker each
  pre-chunk's boundaries once more (one fast-tier call each), and the extraction of a Korean, Japanese or Chinese
  corpus's re-chunked units. A record a build before #556 stored under other summary settings or `models` tiers is
  extracted again, and on an index built before #565 the reset costs a first build (the #565 prompts below).
- Stored communities keep positional names (`Level 0 Community 3`) until the next reset, which renames them and asks
  every report once more; later resets reuse the answers (#539).
- #640 claim statuses: claims stored before it keep the status the model wrote (`True.`, `Suspected`) until the
  reset, which asks every unit's claims again (above).
- Claims and report switches: with claim extraction on and the registry on, a namespace whose build record does not
  name claims (claims turned on after its build, or an incremental claims-off run since) is rebuilt once by its next
  run from its stored extractions, asking claims only of the units without a stored claim record (every unit after
  an incremental claims-off run of an earlier `main`, which deleted them). While a namespace holds other source
  scopes' documents, that rebuild, and a run turning report generation or claim extraction off for a namespace that
  holds reports or claims, are refused (exit 2): keep the switches the namespace was built with.
- Incremental runs may have stored a summary of a summary for an entity or relationship edited documents shared
  (#564, #589), also a run resumed after dying between its removal and its commit (#592), and a summary that kept
  the lines of a spelling a later run split off or lost those of a spelling a run folded in (#624), or the line of
  an edge a run's delta resolved under a spelling the stored graph keeps apart (#640), or the line of an edge of
  such a spelling stated in a unit beside the edge it kept (#644), or the line of a stored edge a delta's edge
  folded in: reset once to summarize from the lines (one call per such item over the threshold), or leave it.
- #565 changed the translation, description summary and gleaning prompts (gleaning re-keys every gleaned record,
  claims too): a reset translates, extracts and gleans every unit again, the cost of a first build; incremental runs
  use the new prompts for new and changed documents only. A reset also types untyped merges and re-cuts one-line
  `intelligent` pre-chunks.
- An index built before vector digests embeds its items once more on its next incremental run (#527); a build record
  written earlier has no community counters until the next full build (#540); the first incremental run of each
  namespace reads its stores once to record their schema annotations, with no model call (#488).
- Chunk sizes count tokens by default (`processing.chunking.length_unit: tokens`, sizes 250 / 2000 / 125 / 1200 /
  12500 / 125): a config that sets the sizes in characters must also set `length_unit: chars` (#460).
- `run-eval` results saved before usage accounting count in `answers_without_usage` / `answers_without_latency` and
  fail a `usage/*` or `latency/*` gate; results saved before #563 carry the earlier `graph_aware` scores, which
  `--from-results` and `--baseline-results` reuse: run them again to compare (#526, #529, #563).
- #577: `search.global_search.reduce_with_llm: true` no longer makes a reduce call and `evaluation.warm_up_query:
  false` no longer skips the warm-up (both warn); a run resumed across it finds no stage-cache entry
  (`fixing.enabled` keyed every stage), but the LLM response cache serves its model calls.
- #596 methodology profiles: a namespace ingested with `graph.community_detection.enabled: false` holds no
  `graphrag`: `local` is now refused there (`MethodologyNotIndexedError`, `run-rag` exits 2) where it answered
  without the report section, as are `global` and `drift`, while `naive` and `simple` still serve it (#607); `auto`
  routes among the strategies of `search.auto_routable_strategies` it serves (`mix` and `iterative` by default).
  Ingest with `graphrag` to use the others. A build record written earlier names no artifacts: queries take the
  namespace to hold every methodology whose artifacts a run of the configured `indexing.methodologies` builds (the
  default `[graphrag, lightrag]` holds `hipporag` too, #602) until the next run that writes the namespace records
  them, and that run rebuilds nothing. Default configurations run the same stages under the same stage-cache keys.
- #598 iterative routing: `auto` may now route a multi-hop question to `iterative`
  (`search.auto_routable_strategies` adds it): about 10 s more at the median and +$0.002 for the queries it takes,
  and a LightRAG-only namespace now makes a router call (between `mix` and `iterative`) where it made none. To keep
  the earlier routing, set `search.auto_routable_strategies: [local, mix, global]`; nothing is re-run.
- #599 RAPTOR: `raptor` can be selected (`indexing.methodologies`); without it nothing changes: no stage cache,
  extraction record or evaluation manifest is keyed by the new settings at their defaults, and the text units are
  stored as before. Adding `raptor` to a namespace builds its summary tree on the next run, or with `run-ingestion
  --enabled-stages summary_tree`, from the stored text units: no re-extraction, about one summary call per five or
  six text units on the default tier (the MuSiQue benchmark store's 3,318 text units: about 580 calls, $2-8 with
  Sonnet 5.5, $0.1-0.4 with `models.roles.summary_tree: fast`). A completed pipeline id re-run resumes at
  `indexing` as before. A tree built from a `main` checkout before #601 is summarized again by its next build (the
  summaries' length changed): as many calls as building it. `tree` retrieval scores saved before #601 rank a
  summary's lineage files above the passages; evaluate again to compare.
- #602 local ranking: `local` orders its text units by HippoRAG 2's PageRank by default
  (`search.local_search.text_unit_ranking: ppr`): answers can change. DRIFT's follow-up local searches keep the
  earlier order (`search.drift_search.text_unit_ranking: support`, #607). Nothing is re-indexed or re-run; each
  local query adds a text-unit search, a relationship kNN search, an edge read and one query embedding (about 0.1 s
  on the embedded stores), and without the relationship vector index the PageRank is seeded by the entities and
  passages alone. Set `search.local_search.text_unit_ranking: support` to keep the earlier order.
- #605 one index language: a namespace ingested with `processing.translation.enabled: false` and a `source_language`
  other than `target_language` was extracted, summarized and reported in `target_language`. Its extraction settings
  and stage caches now record the index language (#612), so its next incremental run stops before any model call where
  the analyzers differ (exit 2, `was built for other language settings`, the graph indexes' analyzers follow
  `source_language`) and otherwise extracts every document again (a WARNING names `index_language`); a re-run of a
  `--pipeline-id` does not resume from its stage caches either. Run one `indexing.reset: true`, which costs a first
  build (every unit extracted and gleaned again, every summary, report and vector made again). Any other configuration
  keeps its extraction records and indices; the community report prompt now names its language, so the next full build
  asks every report once more (one call per community), and later builds reuse them.
- `intelligent` chunking (the default) cuts a model chunk over `max_chunk_size` without pieces under
  `min_chunk_size` (#624): a reset re-cuts only the documents that had one, whose changed text units are extracted
  again (the boundary answers are kept); incremental runs cut new and changed documents so.
- With claim extraction on (`processing.claim_extraction.enabled`, off by default), the claim prompt changed (#624):
  the next run asks the claims of every text unit it extracts again (one claim call per unit that states an entity;
  the graph records are kept), a reset all of them. Stored claims keep their status until their claim is derived
  again: reset once for `disputed` statuses and the object-less claims an empty `<object>` dropped.
- #606 evaluation results: `run-eval` results saved before it record no `token_f1_tokens`: they tokenized by
  whitespace, so on a dataset with Han, Hangul or Kana text a default-normalization run is refused against them (exit
  2) until one side is scored again (an `official_squad` or `hotpotqa` run, or any run on a dataset without such text,
  still compares). Results saved from #606 until #613 record `script_aware`: their Korean token F1 dropped any one to
  three Hangul syllables after a gold word, so they are refused the same way. Their `abstention_statistics` left
  failed answers out (rate and recall over the completed answers), and pairwise length buckets are renamed (`<=0.67x`,
  `0.67-0.91x`, `0.91-1.1x`, `1.1-1.5x`, `>=1.5x`). `prepare-benchmark musique --unanswerable` without `--data-path`
  now also downloads the answerable split and draws the answerable questions from it (labelled `musique`); prepare
  again for that sample. Nothing else re-runs.
- #656 answer normalization: evaluation results of German, Spanish, French, Italian or Portuguese datasets saved
  before it scored exact match, token F1 and containment with English articles only: rescore them (`run-eval
  --from-results DIR --rescore`, no answer call; a span call per answer with `answer_span: extracted`) before
  comparing them with new runs.
- #656 A/A arms: `--arm` runs saved before it record no `arm.aa`, so a comparison rebuilt from them adjusts and
  gates an A/A arm with the others; run the arm again to mark it.
- #656 pairwise slots: differential pairwise results judged before it, in one answer order, favour the candidate:
  re-judge with `--from-results DIR --pairwise-judge` (two fast and four judge calls per pair).
- #607 rerank metering: a query's usage counts its rerank requests (`rerank_calls`, `rerank_search_units` under the
  `rerank` role and the rerank model's id), so `run-eval` with `evaluation.prices` gives no USD estimate until the
  rerank model has a price (`per_thousand_rerank_search_units`; 0 for a local cross-encoder). A price entry sets
  both token prices or the rerank price; one with neither, or one token price alone, is refused at load. Results
  saved before #613 record no `usage_schema`: a comparison against them gives no USD delta (a warning names them)
  until they are run again. Nothing is re-run.
- #608 stored translations: translations are stored from the first run after upgrading: that run still asks the model
  for every unit it translates, with the new translation prompt, so a reset of a translated namespace translates and
  re-extracts every unit once more (the cost of a first build; an incremental run only its new and changed documents),
  and later resets reuse them. Community report answers stored before are not found under the new key (the packed
  input): the full build after it pays that same report call per community that #605 names, once for both, and later
  builds reuse them. A graph with entities without relationships may lose its top community level at its next
  clustering (its reports go, no call is added, global search's `community_depth` cuts one level higher). A document
  with a paragraph longer than a chunk is cut otherwise when it is next chunked (a reset or an edit), and those chunks
  are extracted again. `graph_metrics` no longer holds `average_clustering` or `diameter`. A summary tree of more than
  100,000 text units now fails its stage (`indexing.summary_tree.max_leaves_in_memory`): raise it with
  `fargate_memory`. The tree stored before stays in the build record, so `tree` keeps serving the text units it covers
  (#612).
- #620 run directories: `run-eval` results land in `<outputs>/run_<timestamp>_<id>/`; scripts that read
  `<outputs>/evaluation_*` directly read the run subdirectory (the JSON output's `output_directory` names it), or set
  `evaluation.run_subdirectories: false`. A results directory that mixes runs of one strategy from several
  invocations (the earlier flat layout, filled by more than one `run-eval`) is refused by `--from-results` and
  `--baseline-results`: list the result files of one invocation, or add `--pool-runs`. Nothing re-runs.
- #620 run order: a `run-eval` comparison is answered interleaved, so one stopped part-way saves no run (all finish
  together); run `--run-order sequential` to complete an interrupted comparison by its missing strategies. With
  `--repeats N`, each repeat makes its own query extraction calls (fast model; one entity or keyword extraction per
  query and repeat instead of per query), and `repeat_stdev` can read higher than before on the same answers: it now
  includes the extraction's noise (`--repeat-extractions reuse` for the earlier numbers).
- #612 report generation off: a namespace built with `graph.community_detection.report_generation.enabled: false`
  holds `graphrag` again, so `local` answers on it (refused since #603); `global` and `drift`, which read the
  reports, are refused naming the switch. Nothing re-runs.
- #617 image and IaC hardening: the data-plane image has no AWS CLI and no uv, and its venv and `/app` are read-only
  to the app user, who writes only `/app/cache`, `/app/outputs`, `/app/logs`, `/tmp` and its home: a derived image
  that runs `aws` or `uv`, or a command that writes elsewhere under `/app` (`run-prompt-tuning`'s default
  `./tuned_prompts.yaml`), installs its own or writes under one of those. `aws.opensearch.use_iam: true` with a
  username or password and a negative `fusion_weights` value are refused at load (they were ignored or failed later):
  drop the setting the load names. CDK: the next deploy turns on the cache bucket's S3 Bucket Key with `use_cmk` (new
  objects only, no re-encryption; fewer KMS requests), tags the ingestion tasks and grants the state machine
  `ecs:TagResource` on them; an `env_name` too long for the bucket or guardrail name now fails the synth. Nothing
  re-runs. With `neptune_audit_log` on, the stack's audit log group is `/aws/neptune/<cluster id>/audit`, where
  Neptune writes (it was `/aws/neptune/cluster/<cluster id>/audit`): if Neptune already created that group itself,
  copy what you keep and delete it (`aws logs delete-log-group`) before the next deploy, which fails on an existing
  group otherwise.
- #618 community hierarchy and report references: stored communities and reports stay as they are until the namespace
  is clustered again. Its next full build (`indexing.reset: true`) keeps the level-0 communities and changes the
  levels above them where a community was split (it is now the parent of its pieces, so the hierarchy can be a level
  deeper and a `community_depth` cut maps other reports), and asks every report once more under the new report prompt:
  one report call per community and its embeddings, the report stage of a first build (no text unit is extracted
  again); later builds reuse them. An incremental run clusters and reports its own communities the new way (#618).
- #627 citations: answers now cite their context with `[n]` markers (`search.citations.enabled`, on), so answer texts
  and the LLM judges' scores differ from earlier runs (`answer_match`, `graph_aware`, the `ragas` and `assertion`
  judges and the pairwise judge read the words without the markers): compare
  runs of one setting, or set it `false` for the earlier context and prompt. `global` and `drift` answers no longer
  carry `[n]` markers or `RAGOutput.citations` by default, and their answer prompt drops the "cite where possible"
  instruction: set `search.citations.strategies: {global: true, drift: true}` to cite there again (with the plain
  citation instruction). A `run-eval` comparison of `global` or `drift` against the other strategies, or against a run
  saved before this change, warns that the answers differ in citing; a saved run without `citations.strategies` in its
  manifest is read and re-scored as citing in every strategy, as it did. Pages and relative paths are stored with the
  text units of documents indexed from this release on; to add them to an existing namespace, re-ingest it
  (`indexing.reset: true`): chunk boundaries and text-unit ids do not change, so the extraction store's records are
  reused and the cost is the embedding and indexing of the text units.
- Ingestion output floors add the thinking budget of the effort a call runs at, not always the `high` one (8192): a
  call on a fast-tier role (`translation` by default) or under `aws.bedrock.ingestion_effort` asks for another
  `max_tokens`, so its LLM response-cache entries miss once and a re-run of that stage pays its model calls again;
  extraction-store records are reused (they are keyed by the effort, not by `max_tokens`).
- #628 `run-visualization` no longer takes the provider flags (`--backend`, `--llm-provider`, `--llm-model`,
  `--embedding-provider`, `--embedding-model`, `--embedding-dimension`): it renders exported data and calls no model,
  so a script passing one exits 2 until it drops it.
- A `custom_prompts` override of `query_rewrite_*` (#212) should answer `{"question": ...}`, or
  `{"question": "[CANNOT_REWRITE]"}` to keep the question as asked (*Changed*): another answer costs one
  output-fixer call per follow-up.
- #616 MCP `kg_answer` returns its answer and abstention fields first, then by default only the records the answer
  cites (`records: cited`; every source when it cites none, `records_kept` says which applied): an agent that read
  every source the answer model saw passes `records: all`. Nothing re-runs.
- #661 filters: on a graph store, filter an entity by its own extracted attribute as `attributes.<key>`, not
  `attr_<key>`, which now filters by document metadata on every store (a graph matched it against the entity's own
  attribute). A `name` filter needs the whole name (`name.keyword`), and a filter on analyzed text (`description`,
  `text`, `summary`, ...) or on the seed properties raises `InvalidFilterError`: filter by an exact field instead.
  Nothing is re-indexed.
- The removed `auto_resolution` keys keyed the community detection stage and the `run-eval` manifest's
  `retrieval_config_sha256`: a resumed `--pipeline-id` runs Leiden, indexing and the summary tree again (the reports
  come from the extraction store without a model call), and runs saved before pool with new ones only under
  `--pool-runs`.
- Languages: a Hindi, Arabic or Urdu document with a paragraph longer than a chunk is cut after `।`, `؟` or `۔`
  now, so a reset re-chunks it and extracts its text units again (an incremental run only its changed documents);
  names with Devanagari or Thai vowel signs may resolve otherwise on that reset.
- Token estimate: a Russian, Greek, Arabic, Hebrew, Hindi or Thai corpus is estimated at 1.7-2.7 times the
  tokens; with `length_unit: tokens` a reset re-chunks it and extracts it again, and the next run asks again the
  description and tree summaries whose input now reads over or under their budgets (`chars` re-chunks nothing).

### Added
- A query reports every step it answered without in `metadata["degraded"]` (each `step`, `reason`, `error_type`;
  `degraded_error` stays the first error type): a store read a strategy absorbed (simple and naive reported a clean
  empty search), global's rating and map calls, DRIFT's and iterative's model calls, and the chain's
  `processing.ignore_errors` fallbacks (auto's router, the follow-up rewrite, query processing, the context budget),
  from one per-query ledger (`DegradationLedger`). A failed findings-lineage read of a strategy whose answers do not
  cite (global and DRIFT by default), or of the documents of the answer's graph citations, is listed in
  `metadata.citations_unresolved` instead, in the same shape. `run-eval` records `degraded_steps` per answer,
  `degraded_step_counts` and `degraded_queries`, which `--max-failure-rate` counts. The answer view (`AnswerView`)
  carries them with the class, type and message of an absorbed error (`degraded`, `degraded_error`,
  `citations_unresolved`, `error_class`, `error_type`, `error_message`; a search without an answer too): the
  console's Ask page shows a partial answer and unresolved citation reads as such, naming the steps in the page's
  language, MCP `kg_answer` and `kg_search` return them after the answer, and `run-rag` prints them (and its JSON
  document holds them).
- `ModelUnreachableError`: a model server that cannot be reached is named with its provider kind, `base_url` and how
  to start it (`ollama serve`, the models to pull), the one server the failed connection was made to when its
  transport says which; a bedrock one by its region's endpoint (not a `base_url` the local preset left), and a model
  that needs an inference profile in a region whose Bedrock control plane cannot be reached (one Bedrock is not
  offered in) is refused with it, naming the region, not with advice to grant ListInferenceProfiles. `run-ingestion`
  raises it at its embedding probe, before any extraction (it retried for
  minutes and ended with "produced 0 outputs"); a query raises it at once, without the transient retries' backoff
  (the error answer under `ignore_errors`, which no fallback absorbs), and `run-rag` exits 1 without a traceback.
  `ErrorClass` and `classify_error` name the class of a failure, which every surface maps: the CLIs' exit codes, the
  MCP server's caller messages and the console's error kind, also of an error answer (`metadata.error_class`).
- `run-eval` comparisons adjust for the many deltas they read: every paired delta has a sign-flip randomization
  `p_value` (exact on few pairs), and `multiplicity` Holm-adjusts them in families: a primary metric declared in advance
  (`--primary-metric`, `evaluation.primary_metric`, default `answer_match/answer_contains`) over the compared strategies
  only, every other metric, the cost measures and the pairwise judge's criteria (`pairwise`, each a sign-flip `p_value`
  of its per-query scores against a 0.5 win rate), each with `adjusted_p_value` and a simultaneous
  `adjusted_ci_low`/`adjusted_ci_high`. `--arm` runs against their own base runs form the same families apart
  (`arm_primary`, ...), and closed-book and A/A arms (`arm_roles`) are in none: they bound prior knowledge or measure
  the noise floor, and widened the other tests' adjustment. On simulated A/A comparisons ten unadjusted metrics showed a
  difference on noise in 30-40% of them, the Holm-adjusted family in about 5% or fewer. Each family reads at least 20 x
  tests / alpha resamples (`multiplicity.families.<f>.resamples`; `evaluation.resamples` sets the base, 2,000): the
  Monte Carlo p-value is never below 1 / (1 + resamples), so with a fixed 2,000 a family of 100 or more tests (15
  strategies x 7 metrics) could not reject at 0.05 however large the difference, and its simultaneous intervals read
  their tails from a fraction of one resample (#639). A family's tests of one size draw their resamples once for all of
  them (the same values; 120 tests over 300 queries adjust in 4.5 s instead of 25 s), holding at most 25 million
  resampled means at once (a group of tests at a time past it, the same values again), and the console's paired-delta
  help names the families' resample counts (#645, #646). `--fail-if` reads the four new statistics, and the tables print
  `Holm p` (#620).
- `run-eval --pairwise-repeats all` (`evaluation.pairwise_repeats`) judges every repeat's answers, repeat r against
  repeat r, at the judge's calls times `--repeats`; each query's repeats are averaged before the queries and the win
  rate's interval resamples queries. The default still judges the first repeat (#620).
- Document citations (#627): the answer context numbers its evidence sections `[1]`, `[2]`, ... (passages, entities,
  relationships, claims, reports, summaries) in every strategy, and global search's key points and DRIFT's
  intermediate answers one section each, with the text units and documents of the reports (or the local context) each
  was written from, as MS GraphRAG's reduce cites `[Data: Reports (...)]`: a report's are those of the entities and
  relationships its findings cite, read by id in the query's namespace without the caller's filters (the citation
  applies them to the units; its community's membership only for a report stored without findings, whose cited records
  are all gone, or whose read failed, reported as the search's `degraded_error`; a GraphRAG-only namespace has no
  relationships index and cites the entities), and a cited point names those documents even when the evidence budget
  left its reports out of the context; it stays model-written (screened, never a source) (#635, #643). The report
  sections of `local`, `global` and DRIFT cite the same lineage (`findings_text_unit_ids`), so a report and a point
  written from it name the same documents. The answer prompt (concise, comprehensive or a `custom_prompts` override)
  asks the model to cite them with those markers, ASCII in every answer language, in the same words for every
  strategy. Citations are selectable per strategy: `search.citations.strategies` is merged over the default `{global:
  false, drift: false}` and an entry wins over `enabled` either way; `auto` follows its route and `iterative` its base
  unless it is named itself. On 50 sensemaking questions the pairwise judge preferred the uncited `global` and `drift`
  answers on comprehensiveness (0.68 and 0.75, Holm-adjusted), and a "cite where possible" instruction for them won
  none back and cost `global` directness, so it is gone ([evidence](docs/concepts/defaults-and-evidence.md#citations));
  `{global: true, drift: true}` turns them on with the plain instruction. A strategy that does not cite gets an
  unnumbered context and no citations, and MCP `kg_answer` and the console show no markers for it.
  `RAGOutput.citations` lists one `Citation` per number: whether the answer cites it, its documents
  (`CitedDocument`: document id, file name, path relative to the corpus root, pages), text units, chunk id, title,
  seed-graph provenance and IRI; a graph section names the documents of up to `search.citations.lineage_units` (10) of
  its text units that its query's suffix and filters let through, read in batched fetches (an excluded or another
  tenant's document is never named; the next units fill its place, in at most four reads), alongside the answer call.
  A number the context did not show is reported in `metadata.unknown_citation_markers`, never as a citation (a marker
  has at most three digits: a bracketed year such as `[2020]` is text, and sections past the 999th get no number); a
  bracketed number in retrieved text (a footnote `[2]`) reads `(2)` in a numbered context, so an answer copying it
  cites no section; a blocked answer has none. Earlier answers reach the follow-up rewrite without their markers.
  `CitationMarkerScanner` reads the markers from streamed chunks; MCP `kg_answer` records carry their `citation`
  marker and `cited`, and a cited key point or intermediate answer is a record of its own (`key_point`,
  `intermediate_answer`) with its citation's documents and their text units. The document converter keeps where each
  PDF page starts and text units store `page_start` / `page_end`, `relative_path` and `document_id` in their
  attributes (ids and texts unchanged).
  `run-rag` prints the cited sources and its JSON output carries every citation. The retrieval evaluator adds
  `citation_precision` and `citation_recall` against `reference_sources`, each cited citation weighted by its files (a
  graph section of ten lineage files counts a tenth per reference file, not 1 when any matches), and
  `citation_precision_passages` over the cited passages alone, for the answers of a strategy that cites; the run
  manifest records `citations.enabled`, what it resolved to per strategy (`citations.strategies`) and the scoring, a
  comparison warns when its runs' answers differ in either, and `--rescore` reads each strategy's answers as they
  were written. The pairwise judge, `graph_aware`'s answer coverage
  and `answer_length` read the answers without their markers, as `answer_match` does ([Design
  §7.1](docs/design.md#71-document-citations), [query](docs/guides/query.md#citations)).
- `RAGInput.history` (#627): a stateless caller passes the conversation as `{role, content}` turns, which the
  follow-up rewrite reads where it reads memory otherwise, bounded to `memory.max_messages_per_conversation` turns
  and `memory.max_history_tokens` (4,000, estimated) and screened by the input guardrail like the query (a masked
  turn is used masked, a blocked one dropped); not with `use_memory`
  ([Design §7.2](docs/design.md#72-caller-supplied-conversation-history)).
- A read-only MCP (Model Context Protocol) server, opt-in through the new `mcp` extra and the `run-mcp-server` console
  script (#616): five tools serve one index namespace to agents (`kg_search`, `kg_answer`, `kg_get_entity`,
  `kg_neighbors`, `kg_describe_schema`), each annotated read-only, over stdio or stateless streamable HTTP (loopback
  by default; another address needs `--allow-remote`, and the server authenticates no caller). Results share one
  chunk-oriented record (chunk id, text, score, source document id and file name, cited entities with their IRIs,
  seed-graph provenance; a text record's document id as its text unit stores it; a structured query's
  `query_result` with its table up to 6000 characters and the entities it returned, by id and name). `kg_answer`
  returns the answer first, then by default only the records it cites (`records`: `cited`, `all`, `none`; the first
  `mcp.uncited_records`, `records_kept: top`, when it cites none; `all` lists the model-written sections too), and
  takes the caller's conversation as an optional bounded `history` for a follow-up question (`RAGInput.history`; the
  server keeps none) (#635, #643). The namespace is fixed at start (`--suffix`,
  else `index_value`) unless `mcp.allowed_namespaces` lists others; `mcp.tools`, `mcp.strategies` and the
  `mcp.max_*` bounds limit what a call may ask and costs (`mcp.strategies` binds the chain's new
  `allowed_strategies`: `auto` routes, and falls back, only within it, and is refused before any model call when the
  namespace serves none of it); every call runs under `search.query_timeout_seconds` and returns its model `usage`.
  `kg_get_entity` matches a name folded as local search's seeds are.
  The graph tools read the stores through the chain's retrievers and make no model call; they look names, IRIs and
  ids up in NFC (and a name again in NFKC), as search composes a query. Over HTTP only POST is served (405
  otherwise), and on any loopback address the `Host` and `Origin` headers must name it or a loopback name (421/403).
  `build_mcp_server` and `build_mcp_app` (package root; the latter the stateless ASGI application, its
  `Host`/`Origin` checks on with `allowed_hosts`) mount the same server in a host's application; neither changes the
  chain given. Under `processing.ignore_errors` the error answer names its error in `metadata.error_type`. Guide:
  [MCP server](docs/guides/mcp.md).
- Structured graph queries, opt-in and unmeasured (#630): the `structured` search strategy answers counts, rankings,
  lists, paths between two named entities and seed-entity properties from a query the graph store runs. One planner
  call (`models.roles.graph_query`, default tier; `custom_prompts.graph_query_*`;
  `search.output_ceilings.graph_query`) writes the question as a closed JSON plan over the graph's schema profile
  (declared types plus a sampled profile of what the suffix holds), never query text, its names and values in the
  index language whatever the question's; code validates it (one repair
  call with the refusal), links the names it mentions through the hybrid entity search (a name without a name match
  needs a lexical hit whose name holds every word of it, legal forms aside, Han, Hangul and Kana by character pairs
  and a word spelled alike past `processing.similarity_threshold`, so an absent entity that shares one word with a
  stored one links nothing and falls back; names compare folded at query time as local search's seeds do), compiles
  it to Gremlin bytecode (Neptune,
  Gremlin Server; each branch of a hop range ends in its own `simplePath()`, since Neptune ignores one after a
  `union`, and a fixed length walks one chain without a `union`, whose one-branch form Neptune rejects) or
  parameterised SQL (the embedded graph) scoped to the query's suffix label, checks the compiled
  form against a read-step allowlist (typed arguments only: no lambda) or a SQL checker backed by a SQLite authorizer
  (each alias's label restriction a top-level conjunct, no `NOT`, `IS`, `CASE` or outer join),
  and runs it read-only under a server-side time limit; every store cuts its page in the plan's order, so rankings and
  top rows are exact, and groups an integral float with the int it equals (5.0 with 5). The result table, the entities
  and paths it names (with seed provenance) and their passages are fused (buckets `structured_query_result`,
  `structured_entities`, `structured_relationships`, `structured_text_units`, `structured_text_hits`), the table
  numbered `[n]` like any evidence section and its citation naming the documents of its evidence rows (the units most of
  them cite first, as `local` orders its text units); counts are exact unless a document filter or seed-property
  condition makes code check up to `max_scan_rows` matches ("at least N" past it), and a count cites its first matches
  as entities, a minimum or a maximum the entities holding it and groups by a property their listed groups'. Filter keys
  are checked against the fallback that would run too. A count of 0 or a "no" over a string
  value no vertex holds (misspelled or translated) is no answer (`unknown_value`); a failed planner or store call and a
  plan still no JSON after the repair (`unreadable_plan`, error type `ModelOutputError`) are a degradation of the
  query (`metadata.degraded`, step `structured_query`), a designed fallback (`refused_plan`: a plan the checks still
  refuse) is not. A stored name or a value starting with `~` (a Gremlin hidden token) is refused at validation with a
  WARNING, not as a compiler bug; a type the sampled profile missed is refused saying so, and a count over a symmetric
  `both` pattern says it counts each pair twice. Without a structured
  answer the query runs `search.structured.fallback_strategy` (`local`) and records why in `metadata["structured"]`;
  `mode: augment` runs it beside the graph query. Served by every namespace holding the graph, on every graph store;
  `auto` routes to it only once it is in `search.auto_routable_strategies`. With the output guardrail on, the
  model-written plan text is screened wherever it is returned: the search and answer metadata, the result table and
  the query-result source the console, MCP and `run-rag` JSON show (#635). `GraphQueryPort` is exported for custom graph
  backends, and `search.token_manager.type_budgets.query_result` sizes the result's context share (docs/design.md
  §6.6, [guide](docs/guides/structured-queries.md)).
- `prepare-benchmark structural [--seed N] [--size small|medium|large]` (#630): a generated, never committed,
  procurement world for the `structured` strategy: a JSONL seed graph (`country`, `founded`, `employees`; `SUPPLIES`,
  `MAKES`), a document per vendor and buyer (part of those facts, and a chief executive only text states), the schema
  and seed config, and `eval_data.json` with count, ranking, two-hop, shared-buyer, yes/no, numeric and text-only
  control templates (`category`), absent-entity negatives, Korean and Japanese variants and a `large` size past the
  scan cap, each graph gold answer computed by the reference executor from its plan (`metadata.gold_plan`). `run-eval`
  keeps each answer's `structured` run record (`strategy_records`) and reports `structured_statistics` (status,
  fallback reasons, valid-plan, repair, truncation and exact rates), per arm under a comparison
  ([guide](docs/guides/benchmarks.md#structural-benchmark)). Its items say how `answer_match` reads them:
  `metadata.answer_type: number` matches when the answer's first line (the extracted span, with `extracted`)
  writes the gold number anywhere: digits, number words with their scale words, Korean native, Sino-Korean and Han
  numerals with their multipliers (十二, 一万二千, 만 이천, 一千零一; 1만 2천, 3万, 1.5万 and 3,000万 for digits), 两
  and 兩 among them, before a counter such as 곳, 군데, 가지, 개사, 대, 社, 台 or 円; list enumerators, citation
  markers and ordinals (두 번째, 三つ目, 第三) are no numbers. A gold of 0 matches a statement of none ("not ... any",
  "not one", "not a single", 한 곳도, 一社も, 没有, いません) that writes no number the question does not; which
  number is the count is left to the LLM judges. `set` needs every one of `metadata.answer_items` in any layout
  and, with `metadata.answer_candidates` (every vendor), no other candidate in the part that answers (the extracted
  span, else the first line and the list right after it), so listing them all is no match and a later "Delta is
  fourth" no pick, and `metadata.decline_is_correct` scores a decline of an absent vendor as right; the localized
  counts carry 3곳 / 세 곳 and 3社 / 三社 aliases (#639, #645). `structural-config.yaml` sets
  `evaluation.answer_span: extracted` (EM and F1 on the short span: one fast-model call per scored answer) (#645).
- A local web console, opt-in (the `console` extra, `run-console`; #631, #634): ask with a strategy picker (and how
  `auto` chose), the answer streamed as the answer model writes it, its citation markers `[n]` linked to the sources
  they cite, follow-ups sent with the page's conversation (`RAGInput.history`), and each source the answer read on a
  provenance trace back to its passage, file, seed graph and IRI and into the graph; ingest a workspace folder or an
  upload with a methodology profile (starting from what the namespace's build record holds; a run that would drop a
  held methodology for every corpus in the namespace starts only once confirmed), with live stage progress and a stop
  the pipeline checkpoints; run and compare evaluations (progress per answer; paired deltas called by their
  Holm-adjusted p-value with the family's simultaneous interval and the primary metric marked, falling back to the
  interval for older comparisons; gates, cost, declines, win rates by answer length) and open saved runs; explore the
  exported graph (search, neighbourhoods, communities, degree, schema conformance, seed versus document evidence); a
  dashboard of the questions and jobs run through it (time, tokens, cost per strategy, model and role); and edit the
  config file with validation, a diff before saving, a confirmation before a provider endpoint or a file or directory
  path changes (every config key that names one), and secrets never shown or written. Jobs run the unchanged
  `run-ingestion` and `run-eval` in child processes. It binds `127.0.0.1` (another address needs `--allow-remote`),
  requires a per-launch access token, refuses paths outside its directories, cross-site requests and uploads of a
  file ingestion does not read (another type, or hidden) or naming one path as a file and a folder, writes an
  upload all or none (a refused one leaves no folder) and atomically (a temporary file 0600 until the rename, a
  re-upload keeping the replaced file's mode; a name past its folder's limit is refused up front, `path_refused`),
  names an error by its class (`ErrorClass`), and loads
  nothing from another host. The web client
  (`console/web/`, Vite, React, TypeScript) is in English and nine more languages, in a light and a dark theme (the
  system's unless one is chosen in the top bar), set in Pretendard (Pretendard JP for Japanese, JetBrains Mono for
  figures; self-hosted, SIL OFL), with a force-directed canvas graph (the focused node and its neighbours labelled,
  the rest dimmed; the 600 best-connected entities drawn, the rest found by name or in a paged table), to the brief in `console/web/DESIGN.md`; `console/demo/demo.py` prepares a synthetic corpus of
  the formats the core install parses (multi-page PDFs, whose citations name their page, Markdown, HTML, CSV, JSON
  and text), a dataset and a config (kept on a restart, `--reset-config` to write it again) for fake, Ollama or
  Bedrock models, and stops before it writes anything without the `console` extra, the built web client, or the
  `local` extra the fake and Ollama models need, or with a `UKR_CONSOLE_TOKEN` under 16 characters; a SIGTERM to it
  stops its console ([Console](docs/guides/console.md)).
- `run-ingestion --confirm-integrity` answers only the question of a run that resumes a pipeline id whose inputs
  changed (#631).
- `GraphRAGChain.ainvoke(on_answer_chunk=...)` hands the answer's text chunks to a callback as the answer model
  streams them (the output, its citations included, is that of an unstreamed answer), and an `auto` query's output
  carries `metadata["routing"]` (how the strategy was chosen, among which candidates) (#631).
- RAPTOR summary tree, opt-in (#599): on 100 MuSiQue multi-hop questions, one repeat, `tree` answered about 8
  points of `answer_contains` above `naive` (unadjusted interval) and well below the graph strategies
  ([evidence](docs/concepts/defaults-and-evidence.md#raptor-summary-tree)). The `raptor` methodology adds a
  `summary_tree` ingestion stage after indexing that clusters the stored text units by their vectors (numpy k-means
  by default, RAPTOR's Gaussian mixture with BIC as `algorithm: gmm`; PCA in place of UMAP), summarizes each
  cluster bottom-up in the index language (`models.roles.summary_tree`, `custom_prompts.tree_summary_*`) and swaps
  the rebuilt tree into its own index (`indexing.opensearch.summary_nodes_index_prefix`); a cluster whose children
  are unchanged reuses its summary. The tree is built again when a text unit's id, text, translation or document
  filters changed (a reset after re-filtering or re-translating the same units kept the old tree, its nodes' old
  filters passing queries filtered by the old value), over the translations with their own token counts (#624).
  Each summary is asked for the words (characters for Japanese, Chinese and Korean) 70% of its token budget
  (`max_summary_tokens`) holds at the token counter's rate over the cluster's texts, as Claude Haiku writes about
  1.4 times the length asked (at 90%, 70% of the summaries were cut), and one still over the budget keeps its whole
  sentences within it (#601, #626). `run-ingestion --enabled-stages summary_tree` builds the tree under the
  pipeline id that built the namespace even after an upgrade changed the earlier stages' cache keys, running none of
  them and probing only the indices the namespace lacks (#601). `raptor` alone runs no graph stage. The `tree`
  strategy (`auto` routes to it only on a `raptor`-only namespace) fuses BM25 + kNN searches of the text units and
  of every level of summaries, plus the summaries and passages citing the entities a raw-query search finds where
  the namespace has an entities index (buckets `tree_text_units`, `tree_summaries`, `tree_graph_summaries`,
  `tree_graph_text_units`), reranks them together, keeps `max_nodes_per_level` summaries of a level and expands each
  to `expand_children` of its passages; a summary is reported with its lineage, keeps a document filter only when
  all its documents share it, and goes, with every summary above it, when one of its text units is removed (also
  one it lists as a child past its lineage cut, in a removal and a full build's sweep alike). Under a filter key
  the text units store and the summary index does not (`community_ids`, `source_kinds`; not an entity key, which
  leaves both unfiltered alike, #613) `tree` reads no summary (`metadata.summaries_skipped_for_filters`)
  where it returned them unfiltered, the graph buckets run where the queried namespace holds the entities index
  (not where the config would build one), and their text units are the best-weighted that pass the filters (#607).
  Settings: `indexing.summary_tree`, `search.tree_search`, `search.token_manager.type_budgets.summary`; the
  `VectorIndexer` port gains the summary-tree methods (each with a "cannot" default). Differential tests against the
  original code ship with recorded fixtures. The build record keeps naming a live tree while a run rebuilds it (the
  summaries citing removed text deleted first), so `tree` is served during and after a write run, and is refused
  only once a tree build fails; a full build without `raptor` deletes the tree's indices; a partial tree is built
  again under `refresh: on_reset`; the leaves' vectors are read as float32 arrays (about 15 KiB per leaf read instead
  of 40-60 KiB, about 50 KiB at the build's peak) and `max_leaves_in_memory` is checked before they are read, at a
  default of 100,000 (about 5 GiB; 200,000 did not fit the 8 GiB task) (#608). The leaf digest that tells whether a
  tree is current reads the texts a page at a time and keeps only their ids, attributes and text hashes
  (`VectorIndexer.scan_text_unit_pages`) (#640); the port's default pages a store's `scan_text_units`, so a store
  that implements only that one still builds the tree (#644). Several trees (`partition_by`, `scope: document`) are
  read and built a batch of at most `max_leaves_in_memory` leaves at a time, each batch reading its own text units by
  id (`VectorIndexer.read_text_unit_fields`), where every tree's vectors were held at once. Failed summaries and, with
  `refresh: on_reset`, the text units no summary covers are EMF metrics (`total_tree_summaries_failed`,
  `total_tree_uncovered_leaves`) on the CDK dashboard's "Summary tree (EMF)" widget (#603). See
  [ingest](docs/guides/ingest.md#a-raptor-summary-tree-opt-in) and
  [Design §6.5](docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).
- Methodology profiles (#596): `indexing.methodologies` (default `[graphrag, lightrag]`, what one ingestion built
  before) selects what ingestion builds, so a LightRAG-only corpus (`[lightrag]`) runs no community detection and
  makes no report call. On the CDK stack the `methodologies` context key sets it on both task definitions, so every
  state-machine run builds the profile (#603). The build record names the artifacts and methodologies a namespace
  holds; a query whose strategy reads what the namespace does not hold fails before any model call, naming what it
  holds and what adding the methodology costs (for `raptor`, the tree alone, with the `run-ingestion --methodologies
  ... --enabled-stages summary_tree` command that builds it) (#609, #629); an `--enabled-stages` stage no selected
  methodology needs (`summary_tree` without `raptor`) stops the run before any model call (exit 2), naming the
  methodology to add to `indexing.methodologies` (#629); `run-eval` refuses one before its first answer (exit 2, as
  `run-rag`) (#613), and `auto` routes only among the strategies it serves. A failed read of the record serves that
  query by the config and is not cached (#619). `GraphRAGChain.aread_build_record(suffix)` and `document_retriever()`
  are public (#619). Adding a methodology to an incremental namespace rebuilds it once from its stored extractions
  (only the added model calls are new); removing one stops maintaining its artifacts without deleting them, and the
  next run records them gone, one with no document change too (in every suffix its documents hold, which the
  registry's diff reads, `DocumentDelta.unchanged_suffixes`, rather than a second read of the records (#644)); while
  the namespace holds other source scopes' documents, a run (`insert_documents` and `delete_documents` too) that
  would drop a methodology it holds for them is refused before any model call (exit 2), naming the scopes and the
  ways out (the namespace's methodologies, a namespace of its own, a reset), as one that would add a methodology by
  a rebuild is, and so is one whose switches would stop building the reports or claims it holds
  (`report_generation.enabled: false`, which keeps `graphrag` for `local`, or claim extraction off), naming the
  switch (it passed, and its record dropped them for every scope, so turning the switch back on was refused and
  `global` refused for all); for `raptor` alone the refusal says adding it back needs no rebuild, its tree being
  built from every stored text unit; the build record names the source scopes it holds (`source_scopes`, absent
  from older records), so a query's refusal advises a rebuild only where one is allowed (#642); a reset to fewer
  methodologies keeps the extraction records, so adding one back extracts nothing again (#626), and so does any
  reset that builds nothing from them (`raptor` alone, a second time too) (#640); a reset prunes only the kinds of
  record it produces, so the report answers stay through any number of resets without `graphrag`, and the claim
  records through any number with claims off (the second such reset dropped them, and adding them back asked the
  model again), and so do incremental runs with claims off (they deleted the claim records of every unit they
  extracted again); claim extraction turned on for a namespace without claims rebuilds it once, as an added
  methodology does (an incremental run extracted them for its delta alone and its record never named them, so
  `local` did not read them); the `VectorIndexer` port gains `extraction_record_keys`, whose "cannot" default keeps
  every record (#644). Every strategy class declares
  `BaseSearchStrategy.methodology` (`iterative` reads its base's), and one that reads less declares its artifacts
  (`StrategySpec.required_artifacts`): `naive` and `simple` serve every namespace that holds the text units, and
  `simple` sweeps only the indexes it holds (#607). A filter key is checked, after the methodology check, against the
  indexes the strategy reads that the namespace holds and a run of the config builds, so a key only an absent index
  declares (`type` on a `raptor`-only namespace) is refused (`InvalidFilterError`) instead of being dropped; the
  default sweep and the optional indexes a strategy reads (claims, relationship vectors, reports) follow the same
  artifacts (#613). `hipporag` holds what `lightrag` builds, so `ppr` is served wherever `lightrag` is, and `raptor`
  reads a summary tree (#599). Without a build record (a namespace ingested before them), a query takes the namespace
  to hold what a run of the config builds, `hipporag` included (#602); a run writes the record before the registry
  records its documents, so a first build or reset killed between the two leaves it (the next run found nothing to
  index and wrote none, and queries took the namespace to hold what the config builds). A record any index rejected
  is written again once the documents are recorded, and a run with nothing to index writes a missing one: a full
  record naming only the artifacts the stores hold (claims with the text units and graph: a corpus stating none
  leaves them empty; relationship vectors too unless the graph holds an edge, `GraphIndexer.holds_any_relationship`,
  as a build with `build_relationship_vector_index: false` leaves it), the documents the registry records and
  community counters from
  the stored reports (a WARNING says what queries assume until one is written; the run result's `build_record` and
  `run-describe` say whether one is stored). With
  `graph.community_detection.report_generation.enabled: false` a run builds no reports, so an incremental run no
  longer takes them for missing and rebuilds the namespace, and the namespace holds `graphrag` for `local` while
  `global` and `drift`, which declare the reports among their artifacts, are refused naming the switch (a WARNING at
  load says so, #612); the rebuild that adds a methodology asks before deleting more than
  `indexing.max_deletion_share` of the scope's documents, as an incremental run does; a run without community
  detection (`[lightrag]`) exports `visualization_data.json` without communities; a run that builds no graph-store
  artifact (`[raptor]`) writes no graph store, so `run-ingestion` needs no Neptune endpoint for it (#608), and a run
  creates (and probes) no vector index of an artifact it does not build, emptying only a live one on a full build;
  that full build (a reset) also deletes the namespace's graph when a graph endpoint is configured (#612). See
  [methodology profiles](docs/concepts/overview.md#methodology-profiles) and [Design
  §4.6](docs/design.md#46-methodology-profiles).
- Local mode: the `providers` section and provider registry (`@register_provider`), Ollama and OpenAI-compatible
  models, a sentence-transformers cross-encoder reranker, embedded SQLite graph and vector stores and registry, the
  `local` extra and an Ollama service in `docker/compose.local.yaml` (#154, #209, #271-#273). The cross-encoder is
  its own `cross-encoder` extra, since it pulls in PyTorch (the CUDA build on Linux, several GB): `local` alone runs
  with reranking off, logged once; `local` declares `httpx`, which its adapters import (#628). As on OpenSearch, the
  embedded vector store matches nothing inside an `enabled: false` object or an unmapped field of a `dynamic: false`
  one, and refuses a search of an `index: false` field. A reasoning model's leading `<think>`, `<thinking>` or
  `<reasoning>` block is dropped before every parser, streamed answers included: it began every answer, was stored
  in descriptions, misled the router and hid the `[NO_ANSWER]` token. An Ollama model whose `/api/show` lists
  `thinking` is built with `reasoning=True`, and an OpenAI-compatible server's `reasoning_content` is kept beside
  the answer. The preset adds anonymous OpenSearch access only without a password (#617). See
  [local mode](docs/guides/local-mode.md).
- Filter keys for a document's metadata, `attr_<key>`, on every index and graph store, and for an entity's own
  extracted attribute, `attributes.<key>` (the entities index's key), strictly on every store: a graph store no
  longer matches `attr_<key>` against an entity's own attribute (`attr_category:finance` dropped an entity of a
  finance document whose extracted `category` was `hardware`) (#446, #661). See
  [filters](docs/reference/filters.md).
- Documents from memory: `DataIngestionPipeline.insert_documents([InputDocument(id=..., text=..., filters=...)])`
  adds or updates documents, `delete_documents(ids)` deletes them, through the incremental path of `run` (one run
  each, under the namespace lease); the source scope's other documents are kept (`DocumentDelta.kept`). Needs the
  doc-status registry; an id the registry does not hold raises the new `DocumentNotFoundError` before any write
  (#628), unless the interrupted run of the same `pipeline_id` deleted it (a retry finishes the others), and a
  methodology the namespace does not hold yet raises `ConfigurationError` before the run, naming the methodologies
  held and the directory run that adds it (#640). See [Python
  library](docs/guides/python-library.md#documents-from-memory).
- `run-describe` (and `describe_namespace`, `export_graph`): what an index namespace holds (methodologies, the
  strategies it serves, each index's item count and embedding model, the build record, the registry's records), and
  its graph streamed from the vector store as JSON Lines or GraphML with every item's text units and documents
  (a document by its path relative to the corpus root, else its file name, never the indexing host's path;
  `--output-path`, `--export-format`, `--max-items`; written atomically, so a failed export leaves the previous file
  and no partial one); `VectorIndexer.describe_indices` and `scan_items` (#628). It, `kg_describe_schema` (which
  adds `auto_routes`), the console and the chain's refusal read what a namespace holds by one rule
  (`NamespaceHoldings`): without a build record the console shows what the config builds, as queries are served, not
  nothing held. A registry the caller cannot read (the CDK query task's role has no DynamoDB access) is reported
  unavailable, with the reason in `registry_unavailable` and a warning, and the rest is still described (#636). A
  missing extra and a namespace that holds no index exit 2 with the error on stderr, and describing creates no store
  file where nothing was ingested (#641). See [inspect and export](docs/guides/inspect.md).
- Observability: the `observability` section and `otel` extra, spans for runs, stages, batches and query steps,
  GenAI model-call spans, OTLP, Langfuse and Phoenix exporters with redacted content capture (an IPv6 endpoint keeps
  its brackets in the logged URL); CDK `enable_cloudwatch_tracing` with an ADOT sidecar (#207, #208, #211, #219,
  #269, #606). See
  [observability](docs/guides/observability.md).
- Incremental indexing: the extraction-settings fingerprint (#224, #332); `indexing.max_document_failures` with
  own-error accounting and `documents_retrying` (#175, #431); source scopes and `--retire-source-scope` (#134,
  #192), a scope's first run into a namespace that holds other scopes' documents recorded as the incremental build it
  is (#640); the namespace write lease and quarantine (`indexing.max_document_attempts`) (#331, #346); the commit's
  note (`pending-commit`), which lists the old ids its derivation retires, so when a write fails, or the run dies,
  the next commit derives the items still stored under them again and deletes the old copies of those it re-keyed
  (another spelling took over their name) or dropped; the failed commit already deletes the old ids no failed write
  touches, so a hub whose write keeps failing holds back only the re-keys at it (a note keeps only the old ids still
  stored), and a suffix whose stored reports could not be pointed at the new ids keeps its old ids until a later run
  does both. The stored edges a commit or a removal's strip reads at the entities it derives are compared with their
  vector copies' rank and endpoint names too, at every edge read, and one whose rank alone differs is set in place in
  both stores: a commit writes the vector copy first, a strip the graph copy, so a run killed in between, or one whose
  graph write of an endpoint was rejected (its edges held back), left a rank the other copy did not have, kept for good
  once a later run derived the edge back to the graph copy. A removal's strip is all or retry: it writes in dependency
  order (claims under new ids, stored claims; new vertices, stored entities gaining a unit or a name key, graph edges,
  vector edges, vector entities, the kept entities' graph copies last), and the first phase with a rejected write stops
  it, so what it did not write still cites the removed text units or is in the `pending-strip` note and the retry
  derives it again. Writing on past a rejection lost a kept document's item for good (a split's origin, or the claim a
  claim was split off, overwritten while it held the rejected part's lines; an edge re-pointed at a rejected merge
  target; a stored neighbour of a split, which the strip had not read back, written with the new vertices and ranked
  with the rejected part's edge; a stored entity gaining another's name key but no unit, written with the kept ones
  after the entity that lost the key, which a later removal deleted with the key), or kept a stale vector edge after two
  failed removals in a row (also when they renamed an entity and back: its graph copy held the name derived again, so
  the edge looked current and kept the other name in its vector
  copy). The note lists the edges the strip writes under new ids too, and the stored edges that gain a statement (a
  unit, or another spelling's statement of a unit they cite: a strip stopped after the graph copy left the vector copy
  without the merged-in line and strength, and no copy cited a removed unit): a
  strip stopped (or killed) after an edge's graph copy left no vector copy, and the next one rewrote the edge from its
  graph copy, description cut at the graph's limit. While a store keeps rejecting a write, the strip writes only the
  phases before it (the claims come first), and the kept items it did not write state the removed text until the write
  succeeds. The commit writes its claims in the same order, those overwriting a stored claim only after those under new
  ids (when one of those fails, those dropping a text unit or line of their stored copy are held back and noted, the
  restated ones written): a run killed between two slices of its claim write lost a kept document's claim whose id
  another document's claim took. A failed removal stops the run before the commit, a deleted
  document's too, with an error naming the documents (a quarantined one it removes in the same pass too) and WARNING
  lines naming the rejected writes' items by id: the delta merged into the entities a failed strip left citing the
  removed text units, and dropped a new edge no later run wrote. A deleted document whose removal keeps failing so holds
  back the new documents, as a changed one does. The
  write of an edge whose rank alone changed compares the vector copies the derivation read, not read again, and
  leaves one whose vector copy cites a text unit its graph copy does not (written to one store only: a strip whose
  vector write was rejected) to the removal's retry: rewritten from those copies, it kept the removed document's
  line, which no later run found. An entity citing only removed text units with an edge citing a kept one at it is
  stripped, and retired once the edge is re-pointed, not deleted: a strip stopped before re-pointing the edge (the
  vertex it was at written first, without the edge's units) left it there, and deleting the vertex with a later
  removal deleted a kept document's edge for good.
  A write note (`pending-writes`) over 1 MiB is split into parts, each its own record,
  under a store's request limit (10 MiB on
  small OpenSearch instances). A noted edge the delta states again takes the endpoints the delta resolves, one whose
  endpoint a later removal re-keyed takes the stored spelling of its name its units state, and a noted
  claim derived again resolves its unit's statements against every stored spelling of their names (`Acme Inc` beside
  `Acme`), as the full build does. A removal resolves the claims it derives against the noted entities too (a
  rejected write of `Acme`): it pointed a claim about `Acme Corp` at an entity no store holds, and no later run
  touched it again. Not against the stored copy of an entity the run re-keyed away, though (it is still stored until its
  old id is deleted), and a claim endpoint none of the entities it derives takes follows the run's re-key: a retried
  strip derived that copy as itself and kept a claim about `Acme Corp` at its old id, beside the re-keyed one, about an
  entity the run then deleted. The stored claims about a noted entity are derived again when the commit merges it into
  another spelling (one kept a claim at its old id), and only then: one written again under its own id keeps them, where
  a hub whose write kept failing had each claim about it read and rewritten on every run. A removal counts the text
  units of an entity's graph copy beside its vector copy, which a rejected write left stale: retried after a kill,
  it deleted an entity a kept document states, with that document's edges. A removal's strip notes the edges it
  re-points (at an entity split off another, say), so a retry after a kill folds a stored duplicate there as the
  strip does; it stayed under its old id. A stored claim no statement of its units has the id of takes the one about
  its own names (by their compact keys) and of its type and dates, then the one in most of its units: it took the
  one an id tie-break gave, so a rejected claim derived again after a removal could be about another company's
  entity no store holds. See [incremental indexing](docs/guides/incremental.md) and
  [scopes and namespaces](docs/concepts/scopes-and-namespaces.md).
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
  nothing; `run-rag` loads the schema on the first query with a schema filter key; an S3 ontology without a reader
  raises `SchemaSourceError` (#493, #495, #579, #617).
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
  (DRIFT's intermediate answers, global key points) are screened too, in joined checks of at most 12,000 characters
  one after another (ApplyGuardrail's default quota is 25 text units of 1,000 characters a second), then one per
  result of a chunk whose check intervened (`search.max_concurrency` at a time; the joined intervention is recorded
  in `metadata.guardrail` with `scope: joined`); interventions name word lists
  (`wordPolicy:customWords`, `wordPolicy:PROFANITY`) (#462, #471, #567, #583, #643). See
  [guardrail](docs/guides/guardrail.md).
- Per-query rerank status (`metadata.rerank`) and opt-in `search.reranking.required` (#284). In `run-eval`, queries
  whose rerank call failed (a throttled rerank model keeps the fused order) and answers whose retrieval absorbed a step
  failure (each result's `degraded_error`, the summary's `degraded_error_counts`; a query translation or entity or
  keyword extraction that `processing.ignore_errors` skipped among them, in the chain's `metadata.degraded_error`)
  count against `--max-failure-rate`, so at the default 1.0 a run whose every rerank call failed exits 1; a
  comparison names each run's rerank failures and degraded answers with their rates and warns louder (`NOT
  COMPARABLE AS IS`) when the lowest and highest rate differ beyond chance (two-sided Fisher's exact p below 0.05),
  with a note when they differ within it (#645).
- Query deadline: `search.query_timeout_seconds` bounds every query and stream, from its admission on (the one
  build-record read of a query, where a stalled store held a new chain's first query before the deadline began); a
  stream must send a first chunk within `search.model_timeout_seconds`. Past it the query raises
  `QueryTimeoutError` (`run-rag` exits 1, `run-eval` counts a failure);
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
  alone, where `mix` reran with no keyword (falling back to the search text, or no entity search past
  `raw_query_fallback_max_len`) (#598). Over `mix` it answered multi-hop questions significantly better than `mix`, at
  more than twice the p95 latency ([evidence](docs/concepts/defaults-and-evidence.md#iterative-follow-up-rounds)); a
  KAG-style decomposition into steps measured no better than the loop and is not offered (#628).
  `metadata["context_sufficient"]: false` reports a final insufficient verdict, `metadata["iterative"]` the rounds and
  their model usage, `run-eval` reports `iterative_statistics` (rounds histogram and mean, stop reasons, the share
  stopped at `max_rounds`, degraded runs) per run and comparison arm (#639); new `custom_prompts.iterative_controller_*`
  (#594). The controller reads the context packed to the answer's budget (`controller_context_tokens: null`; a smaller
  cap left it the first passages of long chunks), a round follows at most five bridge entities, a deadline that
  cancels the round an insufficient verdict started reports it (`insufficiency_reason: deadline`), and the follow-up
  queries and bridge entities in `metadata["iterative"]` are screened by the output guardrail with `check_output`
  (#607); bridge entities are told apart by names folded as local search's seeds are, and the controller writes its
  follow-up queries in the index language (#613, #619). A strategy declares the run record it keeps and the model
  text in it (`register_strategy(run_record=RunRecord(...))`): the chain screens and clears every declared record,
  not two named ones. See
  [query](docs/guides/query.md#follow-up-rounds-for-multi-hop-questions).
- LLM usage accounting, tracing or not: calls, input and output tokens, prompt-cache reads and writes, resent,
  throttled and failed requests and cache hits per model role and id, in `RAGOutput` / `SearchOutput.metadata["usage"]`,
  `EvaluationResult.usage`, `run-ingestion`'s JSON `token_usage` and per-stage EMF metrics on the CDK dashboard. A
  call counts from its start, so cancelled and background calls are billed at an estimated input; a response that
  reports no tokens keeps its resent requests and an output estimated from its text (#526, #530, #532, #533, #541,
  #563, #589). Rerank requests count apart (`rerank_calls`, `rerank_search_units`, one per 100 documents, a document
  of more than 500 estimated tokens with the query counting as several, as Amazon Bedrock bills it: about 2x the units
  of one per 100 at 4,800-character chunks) (#626), priced by `evaluation.prices.<rerank
  model>.per_thousand_rerank_search_units` (#607); the run manifest records the usage schema (`usage_schema: 3`), and
  a comparison gives no USD delta against a run saved with an earlier one (#613, #626).
- Corpus coverage: `run-ingestion`'s JSON `units_failed` and EMF metrics `total_units_failed`, `total_files_failed`,
  `total_files_skipped`, `total_documents_excluded` / `_deleted` / `_failed` / `_retrying` and
  `total_documents_without_claims`; a run with failed units ends with a WARNING naming them per stage (#518, #532).
- Evaluation: `retrieval` (hit@k, recall@k, MRR; the passages' files ranked before the lineage files of summaries,
  reports, entities, relationships and claims, so one high-ranked summary or report cannot fill the top k) and
  `answer_match` (exact match, token F1) evaluators, grouped statistics, the run manifest and index build record,
  `s3://` dataset and output paths, guardrail block counts (#113, #121, #206, #315, #321, #345, #355, #579, #601).
  See [evaluate](docs/guides/evaluate.md) and [evaluation output](docs/reference/evaluation-output.md).
- Answer scoring: `evaluation.answer_span` (`first_line` default, `full`, or `extracted` by a fast model, cached,
  falling back to the first line), `evaluation.answer_normalization` (`official_squad`, `hotpotqa`), recorded in the
  manifest's `answer_scoring` with the token-F1 tokenization; under `default`, token F1 counts a Han, Hangul or Kana
  letter as a token and drops a run of Korean particles (their closed class) on a gold word, so Chinese, Japanese and
  Korean answers get partial credit, and a German, Spanish, French, Italian, Portuguese or Russian answer (the
  question's language, `answer_language` in each report) drops that language's articles, splits a French or Italian
  elided word from the next ("d'Avignon" is "d avignon", not "davignon") and reads its written numbers
  ("vingt-quatre", "treinta y dos", "vierundzwanzig", "три тысячи"; Russian in every case: "около двадцати пяти",
  "с пятью") and its words and phrases for none ("aucun", "kein", "ни одного", "no hay", "il n'y a pas de",
  "Поставщиков нет", as English's "not ... any"; Spanish and Portuguese "no" is no 0) in place of English's. A
  `number` item is one rule in every language: a gold other than 0 matches any number the first line writes, a gold
  of 0 a statement of none with no number the question does not write ("En el lote 7 no hay proveedores" asked
  about lot 7 is 0, "No hay más de tres" no 0); a condition or comparative is not told from a count, so a line that
  writes the gold as either matches, a gold that states no number or several ("1990-1995", "3 (as of 2024)") is
  matched as a phrase, and a Unicode minus or en-dash is a minus ("−5" is -5) but an em-dash is not ("3 vendors —2"
  states 2). Thai, Lao, Myanmar and Khmer, which write no spaces between words, are matched as substrings and
  token-scored by letter as Chinese is, and a run of Hebrew or Arabic prefixed particles before a gold word
  (`LanguageProfile.proclitics`: "בירושלים", "وبيروت") is tolerated as a Korean particle after one is, in
  `answer_contains`, token F1, `graph_aware` coverage and the extracted-span check. A yes/no gold is read by
  polarity in every profiled language but Thai (`LanguageProfile.polarity_words`: "sí", "oui"/"non", "ja"/"nein",
  "sim"/"não", "да"/"нет", "có"/"không", "לא", "لا"); a word with a second meaning (`ambiguous_polarity_words`:
  "no", "non", "não", "нет") answers only before punctuation, so "não há dúvida" is no answer. A form of one that
  is also the article reads as one alone ("Hay uno.", "Nur einer.", "Há um."), as do Korean "하나입니다" and "Not a
  single one." (0); "ni uno" and "nem um" are words for none (#324, #516, #606, #613).
- Strategy comparison: `run-eval --search-strategy a,b --repeats N` with bootstrap intervals, `deltas_by_category`,
  a `--pairwise-judge` (LightRAG's verdicts with `evaluation.pairwise_overall`, `position_consistency`, a win-rate
  interval, `--judge-model`) and `prepare-benchmark musique` (#324, #516, #536). See
  [benchmarks](docs/guides/benchmarks.md).
- Unanswerable questions and declines (opt-in; defaults unchanged): `prepare-benchmark musique --unanswerable N`
  adds MuSiQue-Full unanswerable questions (`metadata.answerable: false`, the full dev split downloaded at a pinned
  revision, SHA-256 checked), leaving out any whose answerable twin is sampled or whose removed supporting
  paragraph's exact text is in the pooled corpus (34-50% of candidates on the dev split), and reports the counts;
  `--hops 3,4` or `--hops 2=50,3=50,4=100` filters or sets per-hop quotas. `search.abstention.mode: mark` asks the
  answer model to start a declined answer with the fixed token `[NO_ANSWER]` (detected in any answer language,
  stripped by default) and reports it in `metadata.abstained` / `abstention_reason`, with its sources kept;
  `search.abstention.on_insufficient` (`hedge`, `abstain`) applies when a strategy sets its metadata
  `context_sufficient` to false (`iterative` does). `answer_match` scores a decline on an answerable question (the
  no-context reply included) 0 without a span call, and `abstention_statistics` counts `completed` = `abstained` +
  `answered` (not declined, as in coverage) and adds decline precision, recall and F1 and per `answer_match` metric
  coverage, accuracy, selective and overall accuracy; a failed answer counts as no decline (`failed`) and 0 in
  accuracy. The answerable questions stay the MuSiQue-Ans sample the seed draws without `--unanswerable` (the full
  split is downloaded beside it, or `--unanswerable-data-path`). A comparison pairs `abstention/recall` and
  `abstention/precision`, both `--fail-if` measures. Each saved result keeps `abstention_reason`,
  `context_sufficient`, `insufficiency_reason` and the strategy's run records (`strategy_records`, by the key of
  each strategy's `RunRecord`). Measured on 100 + 100 MuSiQue
  questions, `mark` stays opt-in and `hedge` the default (#593, #600, #606). Under `abstain` an answer the model
  writes instead of the decline is replaced by the no-context reply (invoke and stream), and a streamed decline
  keeps the space at a chunk boundary after the token (#607). See
  [evaluate](docs/guides/evaluate.md#unanswerable-questions-and-declines).
- Evaluation of open-ended answers, all opt-in: the pairwise judge reports win rates per answer-length bucket in
  estimated tokens (symmetric in the log ratio) and their correlation with length; `evaluation.pairwise_mode:
  differential` (`run-eval --pairwise-mode differential`, model role `models.roles.pairwise_overlap`, its output limit
  `models.default_max_output_tokens` above the judge effort's thinking budget) judges only the statements unique to each
  answer, split in both answer slots and averaged (`slots.<criterion>.slot_effect` reports the split model's slot
  preference: one slot gave an A/A pair 0.645), two fast and four judge calls per pair; the `assertion` evaluator
  (`assertion_recall`, `assertion_contradiction_rate`, `negative_assertion_violation_rate`;
  `evaluation.assertion_trials`, a majority over trials in distinct assertion orders, every judge call within
  `processing.max_concurrency`) scores a dataset's `assertions`; `prepare-benchmark assertions` generates questions in
  five classes and source-cited assertions from your corpus, after Microsoft BenchmarkQED (MIT; no BenchmarkQED data
  used), and exits 1 writing nothing when it generates no question; `run-eval --closed-book` adds an answer-model-only
  arm without retrieval, and `deltas_by_closed_book` splits the other runs' deltas into the questions it knows (a
  primary-metric mean of at least 0.5 over its repeats, a failed answer counting 0) and those it does not, exploratory
  (#595, #606). See [evaluation](docs/guides/evaluate.md#assertions-length-control-and-a-closed-book-bound).
- Comparison cost: per strategy calls, tokens and latency per query (mean, p50, p95) with paired deltas, a USD
  estimate from an optional `evaluation.prices` table (cache read and write rates too), and per-query CSV and
  Markdown; a failed answer counts with what it spent (#526, #529, #533).
- `run-eval --from-results` rebuilds or finishes a comparison from saved runs without answer calls and
  `--baseline-results` compares with an earlier run; `--fail-if <measure>:<stat><op><value>` gates exit 1 and fails on
  fewer than 10 paired queries, on metrics an evaluator could not compute (`failed_metrics`), or on a cost measure
  when the arms failed different numbers of answers. It warns when pooled runs differ in a query flag, the
  configuration, the commit, the index or the answer model (#526, #529, #541, #563).
- `run-eval --from-results ... --rescore [DATASET]` scores the saved answers again with this run's evaluators
  (`--evaluators`, the span and normalization, the judges) and compares the re-scored runs, saved in this run's
  directory; DATASET's golds and metadata replace the saved ones by query id. The run manifest records the judges'
  `judge_effort` and, with an extracted span, its model and effort in `answer_scoring` (runs without them are warned
  about, not refused) (#639).
- `run-eval --arm NAME=overlay.yaml[,suffix=SUFFIX][,aa=true]` (repeatable): a configuration arm answers the dataset
  with the run's config under a YAML overlay (validated like a config file, the relative paths it sets resolved against
  its directory and the run's own paths left as the run reads them; the run's `evaluation` section and scoring
  models are kept: an unset judge is chosen over the base and every arm's answer models and pinned first), as a chain
  and evaluation manager of its own, with every `--search-strategy`, interleaved with the base runs. Its runs
  (`<strategy>@NAME`) are compared with the same strategy's base run and Holm-adjusted in families of their own; the
  manifest records the overlay's path and sha256 (and `aa` for an A/A arm) and the comparison order, which
  `--from-results` keeps. An arm shares the base runs' query extractions only when its overlay leaves them alone (the
  extraction model, the default tier's effort, `search.output_ceilings.query_extraction`,
  `models.default_max_output_tokens`, the LLM provider with its context window and region, and a
  `search.query_extraction_cache_size` above 0). An arm whose merged config equals the run's but for cache sizes, on
  the run's own index, or one given `aa=true`, is an A/A arm: the noise floor, named in the comparison's `arm_roles`
  (with the closed-book arm), marked so in the report, the terminal table and the console (which marks none of its
  deltas better or worse) and left out of the Holm families and of `--fail-if` (an adjusted gate failed on it); it
  shares the extractions, so its floor leaves out the extraction's noise unless it sets
  `search.query_extraction_cache_size: 0` (#639, #645). See
  [evaluation](docs/guides/evaluate.md#configuration-arms).
- The `run-eval` judge defaults to Opus 5.5 on Bedrock (a judge judging its own answers shifted directness by
  -0.27): when the answer model, or the model that wrote re-judged answers, is Opus 5.5, an unset judge falls back
  to the default tier or Sonnet 5.5; a configured judge that is the answer model is kept with a warning. A comparison
  whose runs record different judges, judge efforts or ragas embedding models warns. The CDK
  stack's default Bedrock grant covers it (#533, #536, #645).
- Models: Claude Sonnet/Opus 5.5, Opus 4.6-4.8, Sonnet 4.6, Haiku 5.5 and OpenAI GPT models on Bedrock, free-form
  ids with `aws.bedrock.model_overrides`, reasoning effort per tier; a long-output ingestion prompt reserves its
  longest answer plus the thinking budget of the effort its call runs at (#113, #117, #127, #128, #162, #179);
  `aws.bedrock.query_prompt_cache_ttl: 1h` (#458). See [models](docs/reference/models.md).
- `processing.chunking.length_unit`: chunk sizes in a script-aware token estimate (`tokens`, the default) or in
  characters (`chars`) (#460).
- `indexing.neptune.vertex_ids: custom` (`<label>:<item id>` vertex ids); graph expansion reads either mode during a
  switch, and a running query service follows it without a restart (#466, #470, #473, #474).
- `indexing.opensearch.vector_mode` / `compression_level`: on-disk quantized vectors on OpenSearch 2.17+ (#461).
- `run-prompt-tuning` reads every format ingestion parses; each `run-*` `--help` ends with an example (#121).
- `run-rag` text output: `--mode search` prints each hit (type, file or name, score, the start of its text) where it
  printed a header alone, and an answer is followed by one line naming its source files and counting its graph
  elements; `--verbose` lists every source with its score and text, in `--interactive` mode too (which printed
  the details on every answer) (#629).
- A LightRAG keyword extraction whose answer is no usable JSON even after the repair is a degraded step
  (`metadata.degraded`, step `keyword_extraction`), under `processing.ignore_errors: false` too: the query searches
  the raw question as its keyword when it is shorter than `raw_query_fallback_max_len` characters, as upstream
  LightRAG does (the entry's reason says whether it did), and a WARNING names the model role and model that
  answered. It failed the whole `mix`, `hybrid` or `iterative` query with the parser's `Expecting value` (#629).
- The document loading stage samples the corpus (the first 2,000 characters of its first 20 documents) and WARNs,
  naming `processing.translation.source_language`, when the sample disagrees with the setting: a Korean, Japanese,
  Chinese, Russian, Thai, Arabic or German corpus under the default `en` was extracted with prompts asking for
  English and analyzed as English without a word. Each sampled document's language is read by the rule a query's is
  (its script, and a Latin-script one's function words), and the sample reads as the language most of them read as
  (four English documents and one Korean one read as English); a sample that reads as the setting is not reported,
  however few of its letters are in its script (a Korean document that is mostly code). `run-prompt-tuning` writes the
  `source_language` the sample reads as into its output (#629). Han, Hangul and Kana text reads by one rule
  (`domain.languages.dense_language`): Korean when a tenth or more of those letters are Hangul (and no fewer than
  Kana), so Korean written with Hanja reads as Korean, else Japanese when a tenth are Kana, else Chinese.
- `backend: local`: the refusals and retrieval logs of the embedded vector store name it ("The embedded SQLite search
  engine index ...") instead of OpenSearch, and a query where nothing was ingested reads an empty store with a
  WARNING naming `providers.vector.sqlite_path` instead of creating an empty `vector.sqlite3` (in the working
  directory, for the default path) (#629).
- `run-ingestion` failures say why: a run a stage failed names the first failed stage's error (it named the stage
  alone; the cause was only in the logs) and prints the next step, the pipeline id to resume with; and a run that
  writes vectors embeds one probe text before its first model stage and stops (exit 2) when the model returns
  another dimension than `models.embedding.dimension`, naming the setting, where every vector write failed at the
  indexing stage after the extraction was paid for (one embedding call per run) (#629).
- `run-rag --suffix` that nothing was ingested under (a typo) exits 2, not 1, and the message lists the suffixes
  ingested on the store (#629); so does a query-time `ConfigurationError` (an embedding dimension the index does not
  hold). `run-eval` stops (exit 2) when its warm-up query meets either, instead of answering the whole dataset with
  every query failing alike (#639).
- `run-eval`'s comparison and statistics tables fold a long metric name or value onto the next line instead of
  cutting it ("answer_match/answer_contai…") (#629).
- `run-rag`, `run-ingestion` and `run-prompt-tuning` load the chain, the pipeline and the tuner on first use, so
  `--help` and a refused command line import none of them: about 1 s of CPU instead of 2-4 s (#629); `run-eval` and
  `prepare-benchmark` too, whose evaluation package exports its names on first use (#639).
- Every CLI takes `--log-level` (over `logging.level`) and `--version`; with `--output-format text` the console logs
  are plain lines unless `logging.log_format` is set (they were JSON records between the CLI's own output whenever
  stdout was not a terminal; the CDK tasks set `structured` and keep them), and `prepare-benchmark assertions` logs
  to stderr with `--output-format json`. The banner and the `--help` descriptions name the product (they said "GRAPH
  RAG" and "GraphRAG", one of its methodologies) (#629).
- `run-ingestion --suffix`, `--methodologies a,b` and `--reset` set `processing.document_parsing.index_value`,
  `indexing.methodologies` and `indexing.reset` for one run, validated as the config keys are (a bad value exits
  2); `run-ingestion` and `run-rag` print the run's index namespace (suffix and `indexing.additional_suffix`) as
  one `Index Namespace` line (#629).
- Operability: run, stage and query ids in logs, extraction-failure metrics and alarms, start-up endpoint checks,
  and runbook sections on backup, restore (recovery objectives per failure, manual snapshots, a quarterly restore
  drill), replicas and store upgrades (#122, #178, #211, #457, #519).
- IaC options: `neptune_serverless_ncu`, `ingestion_ephemeral_storage_gib`, `opensearch_master_instance`,
  `flow_log_retention_days`, `sfn_timeout_hours`, `opensearch_volume_gib`, `guardrail_version` and a published
  `GuardrailVersion`, `opensearch_nori_package`, Fargate Spot and ARM64 tasks, an `EnableVersionUpgrade` policy,
  `log_retention_days`, the stores' maintenance windows, and `observability_secret_arn` / `_keys` for Langfuse or
  OTLP credentials (same-region only in private mode) (#157, #211, #319, #328, #345, #457, #459, #519, #528, #541,
  #566).
- IaC alarms: `BedrockThrottling` (the share of this deployment's model requests throttled), `IngestionTaskMemory` and
  `QueryTaskMemory`, OpenSearch snapshot failures, blocked writes, thread-pool and indexing-pressure rejections (where
  OpenSearch 2.x counts bulk 429s), the rest of AWS's recommended OpenSearch alarms (CPU, old-generation heap, nodes,
  shards per GiB of heap, 5xx share, queues, dedicated masters), a yellow multi-node cluster, an unusable KMS key,
  Neptune reader CPU and replica lag, doc-status read throttles and system errors, ephemeral storage,
  `DocumentsNotIndexed`, and `QueryLatencyHigh` on `run-rag --metrics-sink cloudwatch`'s `query_latency_ms`, and
  `QueryFailures` on the failed share of queries (`query_failed`); each also notifies when it recovers, and the
  dashboard shows the stores' metrics and every alarm's state (#330, #519, #528).
- The package ships a `py.typed` marker (#568). `main` reports `0.3.0.dev0`, and a `v<version>` tag runs a release
  workflow that publishes a GitHub release: the tag must be on `main`, the notes are the section's introduction,
  *Highlights*, *Breaking changes*, *Upgrading* and *Security* with a link to the version's heading for the rest, and
  their repo-relative links point at the tagged tree (#572, #588). Its PyPI upload (trusted publishing, `pypi`
  environment) stays off until the repository variable `PYPI_PUBLISH` is `true` (CONTRIBUTING.md, "Releasing").
- HippoRAG 2's Personalized PageRank (docs/design.md §6.4): the opt-in `ppr` strategy and `local`'s default
  text-unit order. The `ppr` search strategy (`search.ppr_search`; `auto` never routes to it, and
  `search.auto_routable_strategies` refuses it until a measurement shows where it wins) seeds a Personalized
  PageRank with the question's relationship hits (kNN, `fact_search_type`) and text-unit hits (BM25 + vector), runs
  it over the graph neighbourhood of the facts' entities and the passages they cite, and fuses the ranked passages,
  the passage hits and the facts (fusion buckets `ppr_passages`, `ppr_seed_text_units`, `ppr_facts`) with the
  optional rerank; no query-time LLM call besides the answer. It reads methodology `hipporag`, which every namespace
  with the relationship vector index holds (the default, `[lightrag]`, `[hipporag]`, `[graphrag]` with its
  relationship vectors); one without it refuses `ppr`. `search.local_search.text_unit_ranking: ppr` (the default,
  see *Defaults*) orders local's text units by the same PageRank, seeded by local's entities, the question's passage
  hits and the endpoints of its `ppr_seed_relationships` (5) relationship hits: measured better than `support` on
  multi-hop questions, for about 0.1 s more retrieval a query
  ([evidence](docs/concepts/defaults-and-evidence.md#local-ppr-ranking)) (#602); DRIFT's follow-up local searches
  keep `support` (`search.drift_search.text_unit_ranking`) (#607). Graph retrievers gain `aneighbourhood_edges`
  (Neptune, Gremlin Server, SQLite; the relationships index under a document filter and on Neptune, which reads every
  edge within the set, not one 100-edge page of them, before the per-vertex cap, paged by `search_after` in one
  request per batch of sources), and `SearchQuery.max_hops` overrides the expansion depth (#597). A walk that stops at
  `max_iterations` before it converges is logged at WARNING, naming the settings to raise.
- CDK: `-c neptune_audit_log=true` turns on Neptune audit logs, exported to CloudWatch Logs
  (`/aws/neptune/<cluster id>/audit`, created before the instances) with the CMK and `log_retention_days`
  (`neptune_parameter_group_family` names the engine's family, `neptune1.4` by default); reboot the instances after
  the deploy (#617). A weekly `pin-freshness` workflow (`scripts/check_pins.py`) opens one issue when the base image
  or uv image digest, the ADOT collector or the AWS CDK CLI is behind upstream (#617); it also checks the Trivy image
  and the Dockerfile's apt pins, and builds and scans the image with the opt-in extras, so a fix released after the
  last pull request lands in the same issue (#636, #641), and a build that broke in an issue of its own; a week that
  finds them current closes them. A pin held on a release series (`HELD`: uv on 0.12) lists the newer series as held
  and a newer patch of its own series as behind. Dependabot also moves the console's npm lock (#636). The workflows
  are linted by actionlint, a pinned pre-commit hook the quality job runs.
- Ranking settings that were constants, with the same defaults, none measured: `search.local_search.type_quota`
  (the fusion slots per section type of local, DRIFT, ppr and iterative), `entity_candidates_per_focus`,
  `search.hybrid.normalization_window` and `search.iterative_search.max_bridge_entities` (#619).
- Languages (`domain/languages/`): one `LanguageProfile` per code holds the language's name, script, OpenSearch
  analyzer, decline and the words answers in it are read by; every consumer reads the profiles. Arabic, Hebrew,
  Hindi, Thai and Vietnamese have one (`ar: arabic`, `hi: hindi`, `th: thai`, `he` and `vi`: `standard`), and
  `source_language`/`target_language` take any other BCP-47 code with one WARNING and the defaults (a region is
  dropped: `pt-BR` is `pt`); such a code's text fields get OpenSearch's built-in analyzer of the language where it
  has one (`nl: dutch`, `sv: swedish`, `tr: turkish`, ... in the default `language_analyzers`; the embedded engine
  stems those with Snowball), not `standard` (`config-template.yaml` leaves the map unset: a copy listing a few
  replaced it). The embedded engine stems German, French, Spanish, Italian,
  Portuguese, Russian, Arabic and Hindi with Snowball, normalizes Arabic as Lucene does and splits Thai runs into
  character pairs. A
  Devanagari or Thai name keeps its vowel signs in fuzzy matching (`राहुल` was `र ह ल`), Arabic short vowels and
  Hebrew points are dropped there, a Thai span or name is grounded by character pairs (a Thai sentence was one
  token, too short to judge), the chunker and sentence ends cut after `।`, `॥`, `؟`, `۔`, `။` and `។`, the query
  entity list splits at `،` and `؛`, and a Thai summary's length hint counts characters.

### Changed
- A query without a suffix (`RAGInput(suffix=None)`) reads the chain config's
  `processing.document_parsing.index_value`, as ingestion writes and the CLIs query (`resolve_namespace`, the one
  rule of every entry point); it read `default`, so the console's Ask page failed with `IndexNotFoundError` after a
  successful ingest under an `index_value`. A query checks its namespace's text-units index before its first model
  call (once per namespace), and refuses an empty or blank question (the query is stripped, invisible format
  characters such as a zero-width space too; the MCP tools refuse one as empty as well).
- `run-eval` exits 2 for a filter key no index declares, as `run-rag` does, and `run-rag` exits by the class of a
  failure `processing.ignore_errors` absorbed (2 for a `--suffix` nothing was ingested under; it exited 1).
- Global search on a namespace without community reports declines with `abstention_reason: no_reports`.
- A `name` filter matches an entity's whole name exactly on every store (`name.keyword` on the entities index):
  OpenSearch matched it against one analyzed token (`name:Acme` matched nothing, `name:가나다` matched
  `가나다 상사`), a graph the whole name. A filter on analyzed text (`description`, `text`, `summary`, the
  relationship and claim names, ...) raises `InvalidFilterError`, as does one on the seed properties
  (`attributes.seed_properties...`, kept unindexed), and community reports no longer take `name`.
- Local and LightRAG ask the relationships incident to their entities in one request naming both endpoint fields,
  the best twice `max_query_size` of them, where each end got its own page of `max_query_size`. `SearchQuery` gains
  `any_of` (filter groups one of which a hit passes; a graph retriever refuses them) and `page_through` (a text-less
  search read in pages of `max_result_window` by `search_after`, which the embedded engine supports too).
- Every JSON answer is read one way (`parse_model_object`: prose and fences around it, comments, trailing commas,
  typographic quotes, invalid escapes, a Python literal, a self-correction ending on the answer), then repaired once
  by `models.roles.fixing`, before its step degrades (a `metadata.degraded` entry with the parse error): the
  keyword extraction, DRIFT's primer and actions, iterative's controller, the structured planner, global's map,
  prompt tuning and the LLM judges. A Python dict or a trailing comma failed `mix`, `hybrid` and `iterative`, and
  left DRIFT with no intermediate answer and no degraded mark; an unreadable DRIFT step, router, translation or
  rewrite answer is now recorded in the query's degradation ledger. A model server unreachable on the repair call
  still fails the query. A global map
  output that is not map JSON goes to the fixer, not to a second map call over its batch. A response cut inside
  its last object is not repaired.
- A query is translated exactly when it is written in another language than the index language (or the requested
  `target_language`): a Korean question on an English index is now searched in English (one fast-model call), and an
  English question after a ko→en ingest no longer pays a translator call. A query whose script and words do not tell
  its language (a name or keywords: "Acme Corp", "GPT-4o") is searched as written, on any index.
- The query translation, the follow-up rewrite and the community relevance rating answer JSON: `{"translation": ...}`,
  `{"question": ...}`, and MS GraphRAG's `{"reason": ..., "rating": 0-5}` (`rate_prompt.py`; the rating is divided by
  5, so `relevance_threshold: 0.2` is still a rating of 1 or more). The router answers the strategy name, read from
  the lines that are the name alone or a short label and the name when they all name one (a reasoning block, a
  preamble line, markdown, case and a full-width colon aside: "戦略：local"; an answer listing alternatives or a
  fallback after it goes to the JSON reader, one-line and multi-line JSON alike), and reads any other answer as
  `{"strategy": ...}` (a JSON answer wrote 2.9 times the output tokens, measured; its `max_tokens` is 1,024, a local
  model that may reason keeping its cap). A model that cannot translate or rewrite answers `[CANNOT_TRANSLATE]` or
  `[CANNOT_REWRITE]`, which replace the refusal phrase lists of four languages; the router no longer routes prose to
  a strategy one of its words names ("not global ... local", "a simple lookup"), and a rating's reason in any
  language no longer reads as the rating ("Auf einer Skala von 1 bis 10: 8" was 1).
- The local token estimate (models without CountTokens, embedding truncation, `length_unit: tokens`) counts each
  script at its rate: Cyrillic 2.4 characters a token, Arabic 2, Greek, Hebrew, Devanagari and Thai 1.5 (the
  geometric mean of the cl100k and o200k counts over a sample), where it counted 4, a quarter to a half of the Hindi,
  Thai and Hebrew tokens; Latin stays at 4 and Han, Hangul and Kana at one a character, now also the rarer blocks
  (Hanja extensions, compatibility Jamo). A model whose reported input tokens are twice the estimate or more, or half
  or less, over 20 calls logs one WARNING. A budget for Claude Opus 4.7 and later (no CountTokens) also counts a text
  at least at its language's characters per Claude token, measured on Bedrock (`LanguageProfile.chars_per_token`,
  `LanguageModelInfo.tokenizer`), 10% over: the script estimate times 1.3 counted German and Vietnamese at 0.67 and
  0.57 of their tokens and overfilled the window ([evidence](docs/concepts/defaults-and-evidence.md#token-estimate)).
  The rate counts its script's letters; other letters and digits (a Korean text's Latin names and ids) count at
  English's.
- Local search's incident relationships that leave the retrieved entities are ordered as MS GraphRAG local orders
  its out-of-network relationships: by how many retrieved entities the outside endpoint links, then by rank (rank
  alone before), after the edges between two retrieved entities (#619).
- The fixed reply to a query retrieval found nothing for (and to a decline that left no text) is written in the
  question's language, one sentence per supported language told by its script and function words (English, the 0.2.3
  reply, or the index language when they do not tell), by one scoring decision (`domain/languages/detection.py`): a
  Latin-script language claims the question with its function words, one more when the question opens on a capitalised
  Latin word, and another script with its runs; the larger claim wins, a tie the Latin words. Acronyms and codes
  ("AI", "OS", "ES6", "iPhone"), parts of compounds ("to-do", "AIとは"), of URLs, emails, paths and dotted names
  ("example.com", "admin@example.es", "/usr/bin/the") and the words of a quoted name ("Bank of America和…") are no
  function words, so "AI 모델 추천", "AIとは", "OS 설치" and "example.com 无法访问" are Korean, Japanese, Korean and Chinese,
  "Who is the CEO of 한빛전자?" and "Compare 한빛전자 and 아크미" English, "with 문 사용법" Korean; a shouted text (three or more
  words, all in capitals; beside another script one longer than two letters, so "AS IS / TO BE 비교" is Korean)
  reads its words lowercased, so "QUI A SIGNÉ LE CONTRAT ?" is French; a colon joins a word only before "//", a digit
  or a colon ("Frage:Wer hat …" is German); among Latin-script languages, Spanish and Portuguese with their "how
  many" words ("como" both: the index language breaks the tie), an elision only one of them
  writes ("dell'", "qu'"; not "d'", French and
  Italian) counts with the function words, and a tie the index language is not in goes to the language whose own
  articles occur ("Qual è il prezzo?" is Italian, not English); evaluation counts a decline by `metadata.abstained`
  alone, no longer by the English text (#619).
- `run-eval` writes each invocation into a run subdirectory of its own of the outputs directory, `run_<timestamp>_<id>`
  (the manifest's `run_id`; `evaluation.run_subdirectories: false` for the flat layout), and `--from-results` /
  `--baseline-results` read a directory with its run subdirectories. They no longer pool unrelated runs silently: the
  saved runs of one strategy from several invocations, or with other settings, are refused (exit 2) unless
  `--pool-runs` is given, with a warning naming what the pooled runs mix (#620).
- `run-eval` answers the runs of a comparison interleaved: every query by every strategy and repeat before the next
  query, their order reversed on every other query (ABBA), in one batch. Answered one run after another, the later
  runs answered later, and runs of one configuration differed by 0.5 to 2.3 s of latency a query on average; the
  drift now lands on every run alike. `--run-order sequential` (`evaluation.run_order`) keeps the earlier order, and a
  comparison whose runs were not answered interleaved in one invocation (sequential, `--from-results` of several,
  `--baseline-results`) warns that its latency deltas measure time too. Each repeat extracts the query's entities and
  keywords again (`--repeat-extractions resample`, `evaluation.repeat_extractions`), so `repeat_stdev` includes the
  extraction's variance; `reuse` keeps one extraction for every repeat. The runs of one repeat share each query's
  extraction while it is in flight (the first pays, the others count a cache hit), so they retrieve on the same
  keywords; the order is also reversed on every other repeat, so an even `--repeats` balances an odd-sized dataset's
  every query. Run manifests record `run_id`, `run_order`, `repeat_extractions` and `repeat`, by which
  `--from-results` pairs the repeats (they share a start time) (#620, #639).
- `run-eval` scores with the three LLM-free evaluators by default (`answer_match`, `retrieval`, `graph_aware`): a
  core install no longer exits 2 for want of the `eval` extra, and a run calls no judge unless asked.
  `run-eval --evaluators a,b` picks the evaluators of one run; `ragas` and `assertion` are opt-in. The `langchain`
  evaluator is removed; its `correctness` repeated `ragas`' `answer_correctness`. The `eval` extra keeps
  `langchain-openai` as the security floor of the copy ragas depends on (#620).
- The document deduplication and fuzzy entity matching worker processes start from a forkserver (`spawn` where
  the platform has none) instead of forking the run, whose I/O and client-pool threads may hold a lock the forked
  worker would wait on forever; see *Upgrading* for scripts (#628). The forkserver imports both pools' modules
  whichever starts it (the second pool's workers each imported theirs again) (#640).
- Concurrent queries in one process (`run-eval`, a server) wait less on each other: the Han, Hangul and Kana test is
  one compiled pattern (MMR ran it character by character on every word of every candidate, about a third of a
  local search's CPU, under the GIL), and the embedded search engine lists neither the whole index for an id fetch
  beside `match_all` nor every other document for a kNN filter that only excludes (the seed-graph predicate on text
  units), and parses a search's `sort` once instead of at every comparison (a sorted 10k page of 50k hits: 2.1 s to
  0.8 s). `run-eval` and `run-rag` default `OPENBLAS_NUM_THREADS` to 1 (a value in the environment wins): with a BLAS
  thread per core for each kNN product, 40 concurrent local queries on the embedded stores spent about 8 times the
  CPU. Results are unchanged; on 200 MuSiQue questions on the embedded stores, 40 at once with the model calls faked,
  a local query's retrieval took about half as long (`support` 19-20 s to 9-10 s, `ppr` 23-25 s to 11 s on a loaded
  host) for a tenth of the process CPU (#602).
- Global search maps every report of one cut of the community hierarchy (packed by tokens, on the fast model), steps
  down to a shallower cut over `max_reports_per_level`, credits key points only to their reports (the numbers their
  ids hold after NFKC: "Report 2", "#2", "02", "２" and "2, 3" no longer credit the whole batch), and dynamic
  selection descends to the leaves; it makes no separate reduce call: the answer model reads the ranked key points
  (0.2.3 reduced them on `map_reduce_model_id`, Claude Haiku 4.5, into a `synthesized_summary` result) (#318, #347,
  #354).
- DRIFT is MS GraphRAG's DRIFT (HyDE primer, follow-up local searches, reduce), with one rerank per query and at
  most about 27 LLM calls at the defaults; each primer fold's reports are its answer's evidence, and its metadata
  counts follow-ups by `report_selection` (#323, #347, #541, #567).
- `auto` routes with the fast model, falls back to the first routable strategy, and starts the extractions its
  strategies read alongside the router call (one round trip fewer; the unread one still runs and is billed); query
  entities and keywords are cached (`search.query_extraction_cache_size`, 1024; 0 turns it off) per index namespace,
  so `--repeats` and comparison arms (on any namespace) pay the extraction once; short-answer calls ask for
  `search.output_ceilings` (and `evaluation.answer_span_output_ceiling`) tokens, never below the thinking budget plus
  1024 (#125, #132, #212, #224, #347, #525, #530).
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
- Communities: Leiden leaves are bounded by `graph.community_detection.max_cluster_size` (20); a community split to
  fit stays the parent of its pieces, as MS GraphRAG's hierarchical Leiden keeps it (regrouping the pieces afresh
  recovered planted and LFR benchmark partitions worse: level purity 0.55-0.92 against 0.85-1.00), and coarser
  levels above it partition the graph of communities (more parents: 3-12% more communities, so report calls, on
  those graphs); components below `min_community_size` get no community (#314, #320, #347, #618). Names derive from
  the content id (`Level 0 Community 1a2b3c4d`) instead of the rank (#539).
- Community reports: the prompt allows only what the data supports (MS GraphRAG's evidence rule, no "strategic
  value" or "actionable intelligence" asks), numbers the entities and relationships it packs and asks each finding
  for `[Data: Entities (...); Relationships (...)]` references, kept as the cited record ids on the finding
  (`CommunityFinding.entity_ids` / `relationship_ids`) and taken out of the searched text (#618). A parent's rolled-up
  sub-community reports are numbered too and cited as `Reports (n)`, which a finding reads as the records the child's
  findings cite (a parent's roll-up findings cited nothing, so `global` and `drift` attributed a parent report to the
  few members packed raw); a range `(1-4)` reads as 1 to 4 and `【Data: ...】` as `[Data: ...]`. A report is titled
  by its name alone (an English "Report for Level 0 Community ...:" was embedded with every report), and a rating
  written beside its scale reads as the rating ("10점 만점에 7점" is 7, not 10). The report stores the cited ids (an
  unindexed `findings` field): a retrieved report, and its source's metadata (each finding's ids, not its text),
  name the records each finding cites (`report_cited_ids`), pointed at the ids an incremental merge kept, and a
  stored report at the new id of a record a later run re-keys (found by its `findings_record_ids` list,
  `VectorIndexer.repoint_report_findings`); a removal leaves a removed record's id, which a query reading the records
  skips; a report stored before cites none until it is regenerated (#640, #643).
- Claims resolve only against the entities of their own text units, and ties of names, types and statuses break
  deterministically, so builds do not depend on document order (#398, #402, #404, #442).
- `indexing.reset` rebuilds without taking the suffix offline (new indices and an alias swap) and a failed reset
  keeps the previous graph and records (#331).
- An incremental run finds what a changed or deleted document contributed by the text units every item cites, strips
  shared items and re-derives them (summarizing from lines as a full build does, also when several edited documents
  share an item or spell it otherwise, when a delta folds two spellings or stored summaries into one item (one a
  removal stripped too) or splits a part off a summarized one (#624), and when the run dies between the removal and
  the commit and is resumed: the removal notes the items it stripped in the extraction store until a commit's writes
  succeed), moves an item to the id a full build gives it, and deletes extraction records nothing names (#325, #327,
  #564, #589, #592). With fuzzy resolution on, it folds a delta entity into its one best stored match (the higher
  score, then the shorter name, then the smaller one) and never joins two stored ones until a reset (#564).
- An incremental run sets the rank of a stored edge whose rank alone changed (a neighbour's degree changed) in place,
  batched, instead of dropping and adding the edge (`GraphIndexer.update_relationship_ranks`, whose default rewrites
  it): most of a delta's edge writes at a hub, 0.15 ms against 10.6 ms per edge on a Gremlin Server. A removal's
  strip does too, where it wrote each such graph edge whole (the graph write of 2,000 hub edges on a Gremlin
  Server: 0.3 to 1.0 s against 2.8 to 3.3 s, over four runs on a shared host). An id given twice is one edge with
  its last rank, on every backend.
- Re-runs reuse what is unchanged: vectors carry a digest of their text and model, so indexing embeds only changed
  texts; the `intelligent` chunker's boundary answers, the text units' translations and community report answers
  live in the extraction store; a full build keeps a stored summary of the same lines and settings; a no-op
  incremental run writes only its lease. `--force-rebuild` asks and embeds again (#527, #531, #537, #541). A reset or
  an edit translates only the text that changed (a translation asked again re-extracted the unit when the model
  answered otherwise); a report answer is keyed by its packed input, not the whole `report_generation` section, so
  a packing budget change asks only the reports whose input it changed (#608).
- The translation prompt translates a text unit as written: it no longer completes fragments, rewrites sentences or
  converts formatting to markdown, and fences the unit as data. It and the `intelligent` chunker's prompt are
  overridable (`custom_prompts.text_translation_*`, `custom_prompts.text_chunking_*`) (#608).
- The splitter cuts a paragraph longer than a chunk after a Latin or Korean sentence end (".", "!", "?" or "…"
  before a space) before it falls back to its last space (#608), and after the closing quotes or brackets of a CJK
  sentence end ("。」"), where the next chunk began with "」と言った。".
- The `graph_analysis` stage no longer computes the average clustering coefficient (about 10 s at 80,000 entities)
  or the diameter, which no stage read: `graph.analysis.statistics.calculate_clustering` and `calculate_diameter`
  load with a WARNING and are ignored (#608, #628).
- Embedding requests in flight are bounded by `processing.max_concurrency` too (at most 16 as before) (#608);
  `processing.embedding_concurrency` (16) and `processing.embedding_batch_size` (50 texts per request) set them.
- Stage cache keys cover the whole configuration and the package version (#332).
- `BatchProcessor` runs one pool of `processing.max_concurrency` items, connection pools follow the concurrency, and
  OpenSearch indexing embeds and writes 800 items of a type at a time (`embedding_batch_size` x
  `embedding_concurrency`, at most 1,000) (#296, #309, #310).
- With the doc-status registry on, a run parses only the source files whose bytes or parser changed since they were
  indexed: each record keeps its file's source digest (`DocStatusRecord.source_digest`), an unchanged file enters the
  diff with its record's content hash, and one the run processes after all (other extraction settings) is parsed
  before its attempt opens. A reset, `processing.deduplicate` or a `target_directory` export parses every file. Files
  parse in a process pool once the rest is expected to take longer than starting one. The first run after upgrading
  parses every file once and records their digests. The corpus fingerprint's read of each file computes its source
  digest too, so a run reads each file's bytes once for both (kept by inode, size, modification and change times: a
  file rewritten at its size and modification time is read again).
- Stage-cache files are compact UTF-8 JSON, 1.2 to 1.6 times fewer bytes to write and hash on English text (more on
  text the ASCII escapes multiplied); entries written indented still load.
- Faster at query time and ingestion: the query path no longer imports datasketch and SciPy (about a second per
  process; the name keys are in `domain/ingestion/names.py`), the SQLite registry reads records by key (a full table
  scan per batch before), the local model clients share one TLS context (a model built in under 1 ms, 13-28 ms
  before), the contribution merge copies only the statement each id keeps, and an XML answer is scanned for its
  sections once.
- The `graph_analysis` stage computes no centrality (#248); `run-visualization` computes it only for a renderer that
  reads it (`BaseRenderer.needs_centrality`) (#563).
- The parser uses its own loaders on langchain-core (same text and ids; a 4,000-page PDF loads in about 1 s);
  `ParserFactory.register_loader` takes any `BaseLoader` (#460, #471). Markdown and HTML are read without an extra:
  Markdown as written, HTML through the standard library's parser (one paragraph per block element, no script or
  style); the `unstructured` extra, when installed, reads both in its place, its image installs its spaCy model
  (retried and checked against the SHA-256 unstructured pins: `scripts/install_spacy_model.py`), and an offline run
  without that model fails at once naming the install command (#534, #628). Word (`.docx`) files are
  read with the new `docx` extra (python-docx); without it they are skipped with a WARNING naming the extra (#628).
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
- CDK: both task definitions drop every Linux capability (as `docker/compose.local.yaml`), and synth warns when an
  environment other than dev deploys the mutable `latest` image tag.
- CDK: a plain `cdk deploy --all` attaches the guardrail it creates when `bedrock_region` is the deploy region (every
  private deploy): the tasks read its id and version from SSM parameters the guardrail stack writes, so queries and
  answers are screened without `-c guardrail_identifier`; across regions, and for a given identifier, the context
  path is unchanged. `-c create_guardrail=false` keeps a deployment without one.
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
  #464, #465, #470, #512, #541, #568, #578). A weekly `fidelity` workflow runs the parity tests against the original
  HippoRAG 2, RAPTOR and BenchmarkQED at pinned commits and versions (their dependencies resolved as of a fixed
  date), and the heavy RAPTOR fidelity cases that pull requests now sample (`weekly` marker); the build backend
  (`hatchling`) is pinned and ASH resolves its scanners as of a fixed date (#604, #611).
- CDK: ingestion tasks carry the task definition's tags (`project`, `cost-center`, `owner`) for cost allocation, and
  the `iac/README.md` run-task examples pass `--propagate-tags TASK_DEFINITION`; with `use_cmk` the cache bucket
  uses an S3 Bucket Key (#617).
- Internal: the configuration models module `config.py` became the package `domain/models/config/`,
  one module per section. No user-visible change; imports from `unified_kg_rag.domain.models.config` are unchanged.
- Docs: the design reference is split by topic. [`docs/design.md`](docs/design.md) is an index of every section
  with a summary and a link, the sections are in `docs/design/`, and the extension guide (§15) is
  [`docs/guides/extend.md`](docs/guides/extend.md); section numbers and heading ids are unchanged.
- `RAGOutput.sources` is typed: each is a `Source` (`content`, `source`, `score`, `metadata`; package root), a
  `TypedDict`, so still the plain dict it was. The id lists of a result's metadata (`text_unit_ids`, `document_ids`,
  `entity_ids`, `community_ids`, `source_kinds`, ...) are always lists of strings: a graph store's one-element value
  or JSON list string is decoded where the result is made. The console, MCP `kg_answer` and `run-rag` read an answer
  through one view, so they name a source's file (the last path segment, its stored attributes first), title (a
  relationship's `A -> B`) and `[n]` alike; `run-rag` builds its text and JSON output from it, the model-written
  sections an answer may cite included (`written`; `--verbose` lists them), and names a passage's file once in its
  citation line. A query a surface builds that the chain refuses (a blank question) is an `invalid_argument`
  (`RAGInput.checked`, `ToolArgumentError`): `run-rag` exits 2 before any model call, the console names it.

### Fixed
Bugs present in 0.2.3.

Queries and retrieval.
- A `type` or `status` filter value matches its lowercase form too, as ingestion stores entity and claim types and
  claim statuses: `type:PERSON` matched nothing on every store.
- `@register_strategy` on a registered class with other arguments raises instead of replacing its record, which
  changed what that strategy reads for every caller in the process; the same arguments keep the record.
- A reranker that returns documents without a relevance score gives them `1 / rank`, not `1 - rank / 100`, which
  went negative past the 100th document (#619).
- Local search looks each query entity up on its own and seeds graph expansion with the entity of its name, else its
  top hit when that scores at least `search.local_search.seed_score_ratio` (0.6) of the best hit, read at equal hybrid
  weights; one lookup of the joined names seeded a wrong neighbour beside every right seed (#517, #530). A top hit
  sharing one word with the name and covering neither way (an absent "Zorblax Supplies" for "Acme Supplies") seeds
  nothing; semantic and alias hits ("NYC" for "New York City") seed as before. Local seeds compare names as entity
  keys fold them (*Stored data and reindexing*) and further at query time: a middle dot between kana, Han or Hangul
  letters ("オリオン・ラボ" for "オリオンラボ"), Arabic short vowels, tatweel, alef with or without hamza, alef maksura, Russian
  "ё" for "е" and a leading "شركة" no longer keep a name from linking, nor Latin accents a name from covering or
  seeding ("Cafe" for "Café", "Nguyen Van An" for "Nguyễn Văn An", and the letters with a stroke, a ligature or no
  dot: "Dang" for "Đặng", "Lodz" for "Łódź", "Yildiz" for "Yıldız", "Gudmundsson" for "Guðmundsson", "Izmir" for
  "İzmir"); an entity written as asked links alone ("شركة النور", not also "النور"; "Acme GmbH", not also "Acme").
  A word keeps its marks (Devanagari and Thai vowel signs: an absent "टाटा" no longer links "टीसीएस" through the
  fragment "ट") and a Thai, Lao, Myanmar, Khmer or Tai word reads as character bigrams as Han, Hangul and Kana do
  ("กรุงเทพ" links "ธนาคารกรุงเทพ"; an absent "ธนาคารกรุงเทพ" no longer links "ธนาคารกสิกรไทย"); the router reads a
  label's words the same way.
  `graph_aware`'s retrieved coverage compares names the same way, and entity grounding folds those letters as it folds
  accents.
- Local search orders its candidate text units by one shared rule (`domain/retrieval/lineage.py`, also the
  structured query result's lineage): the units most expanded entities cite, then the best-scored citing entity's,
  then first seen; an entity citing a unit twice counts once, and a lineage a graph store returns as its JSON text
  is read as its list (it was skipped).
- Graph expansion returns every seed and its neighbourhood ranked by distance (#161, #218), fetches at most its
  per-seed budget (#310), and no longer depends on the store's edge order (`search.graph_expansion.deterministic_order:
  false` keeps it) (#470, #473); an expanded entity shows its type (#567).
- A number or boolean document filter value is a keyword on OpenSearch, so a string value of the same key no longer
  fails to index (`mapper_parsing_exception`) and ranges compare as text, as documented (#566).
- Local search's frequency filter keeps the query's own entities (#218), and local search keeps its seeds when
  expansion degrades (#261); its chunk filter check and global search's step-down ask the document retriever's own
  schema (`filter_fields_for`) (#567).
- Earlier turns' entities no longer leak into a follow-up, and `memory.max_conversation_age_hours` is enforced
  (#176, #218).
- Korean BM25 recall, CJK fuzziness and NFC normalization in retrieval (#175, #177); MMR diversity compares words
  as entity linking does (`word_terms`: punctuation ignored, and Han, Hangul, Kana, Thai, Lao, Myanmar, Khmer and Tai
  runs by character bigrams), so a Thai or Khmer near-duplicate is no longer kept as distinct (#567).
- A query is translated only when ingestion translates or the caller asks (`RAGInput.target_language`), and its
  entities and keywords are extracted in the index language even when the caller asks for another: only the text
  units carry the requested language, so they matched no entity, relationship or report (#567, #583, #619).
- DRIFT (primer, actions, HyDE) and global search's map points write in the index language (`source_language` with
  translation disabled), not always `target_language` nor a requested language, whose follow-up queries found no
  entity (#613, #619); query entities a model separates with `、`, `，` or `；` (or one per line) are split into
  entities, not read as one (#605).
- `simple` leaves the claims index out of its sweep when claim extraction found no claim (and built none), as
  `local` does: every query WARNed "Index not found" for it.
- Store failures fail the query instead of answering from no context (unreachable, denied, rejected or overloaded
  stores, a failing embedding call) (#249, #258, #261, #266); queries no longer hang on a silent store
  (`search.store_timeout_seconds`, a Neptune traversal timeout) (#258, #264); a stalled Bedrock call no longer holds
  a query for about half an hour (#522).
- Neptune reconnects after a closed websocket, a lost connection no longer leaks threads or hangs a write, and
  concurrent queries share one connection attempt (#258, #261, #263, #266, #267).
- A short OpenSearch throttling burst no longer fails a query or a bulk write (#263, #264), and a throttled Neptune
  or Gremlin read no longer empties a query's graph expansion: it is retried within the query deadline, as an
  OpenSearch query request is; a long question no longer fails its lexical search, and long list filters no longer
  multiply searches (#258, #260).
- The text-units section is no longer empty for a corpus translated into five or more languages (#264).
- The answer context stays within `max_tokens`, a small budget no longer yields an empty context, and CJK sections
  truncate at a character (#247, #258); an entity hit renders as `Entity` / `Type` / `Description` lines and items
  are separated by one blank line (#567).
- A relationship without a description reached the answer prompt as `Relationship: A -> B`: the relationships index
  stores the type (`RELATED_TO` when untyped; not a filter key) and the line reads `A -[TYPE]-> B` (#562).
- With MMR on, `local` and `drift` keep every reserved text-chunk slot (#247).
- `simple` no longer ranks hits of different indexes by raw score, and each index brings its own `top_k` ×
  `retrieval_multiplier` candidates, fused and reranked as one pool before the `top_k` cut (one search kept that many
  across every index, two or three each, and lost to `naive`); it reads the summary tree where RAPTOR is built.
  Local and LightRAG incident-edge expansion keep no arbitrary hub edges; LightRAG keeps every typed edge between two
  entities; `hybrid` without keywords warns (#295, #317).
- LightRAG `mix`/`hybrid` context includes the chunks the matched relationships cite (offline hybrid one-hop
  gold-chunk recall 0.30, now 0.86, 0.96 with rerank), and incremental and full builds pick the same chunks (#521,
  #530); the pick runs on a worker thread, as PPR does: on a corpus whose hub entities cite thousands of units it held
  the event loop for up to seconds a query, stalling every concurrent one.
- An answer the output guardrail blocks writes no conversation memory, as a blocked query writes none (the refusal
  was saved and read by the next turn's follow-up rewrite).
- `structured` refused on a namespace without the graph (a `raptor`-only one) names the methodologies that build it
  (graphrag, lightrag, hipporag), not "any methodology".
- PPR passages of equal score (those no seed reaches, all 0 when the edge read degraded) rank in the support order
  (the units most entities cite first), not by id, so the text quota no longer fills in id order.
- A context budget no retrieved section fits is a `context_optimization` degradation (reason `context_budget`), not
  a silent `no_context` decline, and a section-type header is paid only for a type seated: a small budget (`local` at
  `max_tokens=40`) went to the headers of every type present and seated nothing.
- LightRAG `mix`/`hybrid` (`kg_chunk_pick_method: weight`) take each matched hit's chunks as upstream does, those
  more matched hits of the side cite first, then by chunk id: in id (hash) order a chunk three hits cite, kept on the
  first, could miss that hit's slots. No run measured the change yet; one on `mix` and `hybrid` is queued.
- Hybrid search on OpenSearch 2.13 keeps the `size`-th lexical hit, kNN on lucene uses `vector_search.ef_search`,
  and a filter on an attribute an index never mapped no longer fails vector and hybrid searches (#295, #306).
- `auto` no longer routes to `global` or `drift` without community reports, and falls back to `local` only when it
  is routable (#224, #317).
- Global search: a map batch over the model's context is left unrated instead of failing, a map step that fails
  entirely fails the query or is recorded in `degraded_error`, and a fatal model error fails the query even under
  `processing.ignore_errors` (#264, #267, #348, #354, #360, #567). A dynamic-selection rating is MS GraphRAG's JSON
  rating (0-5; an unreadable one counts in `global_unrated_report_count`), not a number of a mention of the scale
  (`On a scale of 1-10, I'd rate 8` was 1) (#567, #589). A map output `json.loads` refuses is repaired as MS GraphRAG
  repairs it (a raw line break or an invalid escape in a string, a trailing comma, the points as a bare list, a quote
  left unescaped inside a description) and otherwise goes to the output fixer (one call), its shape logged (never its
  text); an object without `points` counts as unrated, no longer as a rated batch with no points. On a real model
  about 1% of map batches came back unparseable, so one global answer in five lost a batch's reports.
- A suffix nothing was ingested under fails with `No indices found` (#227); indices of another embedding model or
  dimension fail with a `ConfigurationError`, and queries embed with `models.embedding.dimension` (#232).
- `run-rag` and `run-eval` query `processing.document_parsing.index_value` when `--suffix` is not given (they
  queried `default`), and the suffix line names `indexing.additional_suffix` (#534, #536).
- A `GraphRAGChain` reused across event loops no longer stalls, and LangChain `ainvoke` on a package retriever runs
  on the caller's loop (#230, #232, #243, #300).
- A refusal to translate is no longer searched as the query, in any language or with a typographic apostrophe: the
  model answers `[CANNOT_TRANSLATE]` (*Changed*) (#292); Cohere Embed v3 no longer embeds every text as an empty
  string (#463).
- The first calls of a batch no longer fail with "dictionary changed size during iteration", and a call that failed
  once no longer fails its item for good (#369, #378).

Ingestion and incremental indexing.
- An extraction, claims, gleaning, report-findings or chunk-boundary section that lists elements reads as empty when
  it holds a short "nothing here" marker, in any language (one line of at most three estimated tokens:
  `<relationships>없음</relationships>`, `なし`, `None.`, `Keine`): only `None`, `N/A`, `null`, `nil` and `-` did,
  so another answer failed its text unit, which was retried on every run. Longer text there may be the list written
  as prose: it gets one output-fixer call, and a repair that still holds text only is read as empty, counted
  (`sections_read_empty_from_text` in the graph and claim extraction stats) and logged with the section and its
  length.
- An incremental run, and a run into a namespace that holds other source scopes' documents, export the whole
  namespace's graph, read from its stores, not only its own delta or corpus (or none, or `Communities must be
  detected before exporting`), so the console's graph and `run-visualization` show what queries read (#640).
- With description summaries on, an incremental run no longer extends a stored summary with a delta edge's
  description when the derivation moved part of it to another edge (a delta that resolved `Vendor Inc` into `Vendor`
  where the stored graph keeps them apart): the summary held a line the edge no longer states, unlike a full
  build's (#640). Nor when the moved and the kept lines share a text unit (a delta that resolved `(주)벤더` into
  `벤더`): an edge with an endpoint the derivation split off is summarized from its lines (#644). Nor when the
  derivation folded another stored spelling into the item (a delta `Depot -> Vendor` edge that folds the stored
  `Depot -> Vendor Inc` into it): the extended summary lost the folded edge's line (#648).
- A document's doc-status registry namespace and its artifacts' index suffix follow one rule
  (`store_names.index_suffix`): an `index` value no store accepts, which a pre-parsed `Document` JSON can carry,
  fails the loading stage naming it instead of keying the document's record under a namespace no index has (#628).
- One index language (`index_language`: `target_language` when translation runs, else `source_language`) for every
  component: with translation disabled, extraction, gleaning, description summaries and community reports were
  written in `target_language` and the graph indexes analyzed for it while the text units, the summary tree and the
  queries used `source_language`, so the graph held a second language. The community report prompt names its
  output language. Claim dates written as `2024年3月15日`, `2024년 3월 15일` or `2022年第3四半期` read as calendar
  dates and placeholders such as `不明` or `미상` as unknown (#605), and so do `2024. 3. 15.` and a day followed by
  its weekday, `2024年3月15日(金)` (#612), a month alone in the dotted form (`2024. 3.`), a Japanese era's or the
  Minguo calendar's year (`令和6年3月15日`, `令和元年`, `民國113年`), the German, Spanish, French, Italian, Portuguese
  and Russian month names (`15. März 2024`, `15 de marzo de 2024`, `15 марта 2024 г.`), and placeholders however
  spaced (`해당없음`, `未詳`) (#624).
- Claims: the claim prompt names the output language of `<description>` (the index language, as extraction's
  does), states that a claim's subject is one of the unit's listed entities (the resolver drops any other) and asks
  for `NONE` when a claim has no object; a claim with an empty, `NONE`, `UNKNOWN` (or another language's "none")
  object is kept under one `NONE` literal instead of dropped or split in two; a unit whose record states no entity
  is not asked (its claims could resolve to nothing); and a claim whose statements disagree (`TRUE` and `FALSE`) is
  `disputed`, in a full build and incrementally, where a majority vote read a tie as `FALSE` and an incremental run
  kept the stored status whatever the new documents said (#624). A status is read into the prompt's vocabulary
  (`true`, `false`, `suspected`, `disputed`, `unknown`, as stored: `True.` is `true`, an unrecognised word `unknown`)
  at extraction and in every merge, so a stored off-vocabulary status no longer disputes the same one (#640). The
  words read are the prompt's and a few plain synonyms, the same in English, Korean, Japanese and Chinese (#644):
  `confirmed` and `verified` are `true`, `denied` and `refuted` `false`, and `uncertain` and `unverified` (`불확실`,
  `未確認`) `suspected`, as MS GraphRAG's SUSPECTED ("not verified") says (Korean read them as unrecognised); the
  claims whose status is none of them are counted in the `total_unrecognised_claim_statuses` metric (#644). As MS
  GraphRAG's claim prompt, built around the allegations a text reports, the prompt takes an allegation or suspicion
  the text reports as a `SUSPECTED` claim saying who alleges it, where it left out all speculation, and keeps
  `DISPUTED` for conflicting information; only the author's own opinions are left out. A `true` or `false` statement
  settles a `suspected` claim, `true` and `false` together dispute it. A claim's id holds its statement (the
  description's entity key) beside its subject, object, type and dates: claims of one type about one subject with no
  object or date were one claim, their statuses merged ("accused of bribery" read as confirmed by "fined for
  pollution"), where MS GraphRAG keeps one row per statement; only a restatement merges.
- An entity without relationships no longer adds a level to the community hierarchy: it made a single root over
  the other communities read as a partition, so global search's cut depths depended on it (#608).
- A model answer that corrects itself is parsed by its last complete answer (each XML section keeps its last closed
  occurrence; the JSON reader the last object that parses, preferring the caller's keys); a single cut-off answer
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
- An entity none of whose statements gives a confidence keeps none (stored as 1.0) in every merge: the statements'
  merge and the incremental merge read a missing one as 0.0, graph resolution as 1.0, so an incremental run could
  store 0.0 where a full build stored 1.0 (`base_processor.merged_confidence`).
- Two relationships whose endpoint names or type hold a `:` no longer share an id: `A:B -> C` and `A -> B:C` hashed
  one string and merged into one edge. Such ids are hashed from a JSON list (`shared.utils.id_content`); others keep
  theirs. An edge an extraction record already holds keeps its id until the unit is extracted again.
- A rejected entity, relationship, claim, community or report write (under `indexing.max_failure_rate`) no longer
  records the documents owning it `FAILED`: the item is noted in the extraction store (`pending-writes`) and the next
  run writes it again without extracting anything. Re-extracting the owners re-extracted every document a hub entity
  cites, on every run its write kept failing. Only a failed text-unit write fails its document. A commit over the rate
  notes its rejected items too, though its documents stay unrecorded: it also wrote what its merge derived for other
  documents, and removing an unrecorded document then deleted a kept one's edge left at the removed one's entity.
  When the note itself cannot be written, the registry documents owning the failed items' text units are recorded
  `FAILED`, over the rate too, those outside the run included (only the run's were, so that kept edge was lost).
  A noted item is cut to the units the next delta does not write again: a retried document's own units kept in it
  wrote it again as resolved when noted, beside the delta's item, at an entity a removal may have deleted since.
- Deleting a document under a non-default suffix strips it from shared claims, a deletion-only run checks the
  configured suffix, and a run whose new documents name no entity still commits them (#248, #252, #256).
- `indexing.reset` reusing a pipeline id rebuilds from the whole corpus, fails when the registry cannot be cleared,
  never keeps the old graph beside a partly written new one, and two rebuilds within a second no longer share an
  index (#221, #233, #299, #305, #346).
- Two corpora in one namespace holding the same file no longer share text units, and store sets with other index
  prefixes no longer share a registry namespace (#283, #327).
- Neptune no longer loses edges when interrupted, a vertex write replaces its stored properties, and OpenSearch no
  longer stores a description cut at the Neptune cap (#252, #294, #377); an edge write looks its endpoints up by
  vertex id (property-mode ids read in one request per batch), so its cost no longer grows with the vertex count of
  every tenant on the cluster (1.5 s an edge at 21k vertices) (#626). A vertex's multi-valued property is written
  1,000 values a request (a hub's 5,000 text units failed with `QueryTooLargeException` under load), and that error
  is no longer retried as it stands. Such a set's stored values are dropped only after every new value is added, so
  a run killed in between leaves the old values beside the new ones, never part of the set.
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
  again (#523); the translation and summary prompts name the language ("Korean", not `LanguageCode.KO`) (#565). The
  summary prompt follows MS GraphRAG's: it names the entity (both ends of a relationship) in the third person, where
  it forbade the name, and resolves contradictions, stating one the descriptions do not settle.
- Unresolved claims are not indexed, an entity named only in a relationship gets its suffix and filters, and
  deduplication prefers an indexed copy (#220).
- Chunking and translation errors are no longer swallowed, a chunk-boundary error falls back to a size split with a
  WARNING, and the `intelligent` chunker splits a one-line pre-chunk by size without a call (#250, #289, #565). A
  chunk the model's boundaries left over `max_chunk_size` is split as the size fallback splits, its pieces under
  `min_chunk_size` folded into their neighbours (a three-character tail was a text unit of its own) (#624).
- `DataIngestionPipeline.run(source_directory)` ingests the directory it is given (#227).
- Resume and caching: a pipeline id resumes; switching a stage on runs it; a new suffix or scope recomputes; turning
  the registry off rebuilds; `--force-rebuild` writes its outputs; `--resume-from-stage` ignores case; generated ids
  no longer collide; a cached output that no longer validates is a miss (#223, #248, #259, #564).
- `--continue-on-error` records crashed documents `FAILED`, not `PROCESSED` (#223).
- A `pipeline_metadata.json` holding valid JSON but no run's record (an array, say) no longer fails every save of
  the run: `--force-rebuild` replaces it (#621).
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
- `run-eval` result and comparison files (with their `.csv` and `.md`), `prepare-benchmark` datasets and corpora,
  `run-prompt-tuning` outputs, static charts and `Document.to_json_file` are written atomically: a run stopped
  mid-write left a truncated file (a JSON file a later `--from-results` refused, a tuned-prompts file over the
  previous one). A symlinked file stays a link (the file it points to is replaced), a replaced file keeps its
  permission bits, a fifo is written in place, and `/dev/stdout` (`/dev/fd/N`, `/proc/self/fd/N`, a link to one)
  is written through its descriptor, so `--output-path /dev/stdout > out.txt` (or `>>`) keeps what else the run
  prints (a file open to update ends where the write does; one open read-only is refused with its path). A file
  its directory will not let be replaced (read-only, on a read-only file system, a bind-mounted file) is staged in
  a temporary file and copied in once written, so a failed export still leaves it whole; a file that would refuse
  the copy too (read-only itself) fails before the export streams, with its own error and path. A copy that fails
  once begun (a full disk), or is refused because another writer replaced or removed the file meanwhile (checked
  just before the copy, also when the file is opened only then), keeps the staged file and names it in the error,
  and `fsync` flushes the copy too; a write that asks for `fsync` flushes the directory after the rename where it
  may. A valid name of 218-255 bytes (about
  73 or more Korean or Japanese characters) is written atomically: its temporary name is cut to fit the directory's
  name limit (255 bytes, 143 on eCryptfs, 255 where the file system says none).
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
- Index suffixes and prefixes are validated at load and on the command line (#367, #372, #374). Two index prefixes,
  or the two graph label prefixes, that can name the same store (equal, or one is the other followed by `-`) fail the
  load, and a search hit's section comes from its index's exact prefix: with nested prefixes every entity hit read as
  a text unit. `RAGInput.suffix` and every entry point's namespace (`resolve_namespace`) take the stores' rule too
  (lowercase letters, digits, `-`, `_`): a library query with the suffix `acme*` read the build record of every
  namespace the wildcard matched before the search refused it.
- `run-rag --filters`: repeated flags add up; a pair without `:`, a blank, duplicate, unknown or `None` value is a
  usage error (#373, #408, #411, #443). `run-rag --interactive`: a line is a command only as a whole, `set-filter`
  keeps whitespace and refuses unknown keys (#353, #413, #416, #426, #428).
- `run-rag` documents conversation memory as in-process (#224).
- `GraphIndexer.delete_by_id` without a suffix acts on the default suffix (#344).
- Used as a library, log records reach the host's handlers without raising on `structlog` fields, a host's root
  handler gets the plain message, and importing the package configures no logging and leaves `sys.path` alone (#178,
  #245, #251, #255, #528).
- `index_settings` keys reach the indices in either spelling (#260, #404).

Deployment (CDK).
- With a `bedrock_region` other than the deploy region the dashboard's Bedrock widgets read that region, where the
  calls publish their metrics; the synth warns that the `BedrockThrottling` alarm, which can read only the deploy
  region, cannot fire there.
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
- `bedrock_model_arns` with a `*` region, as documented, no longer fails the cdk-nag synth; an `env_name` too long
  for the cache bucket or guardrail name fails the synth (not the deploy); an odd `opensearch_count` synthesized
  without an account says why only 2 AZs are seen; the endpoint count in `iac/README.md` is six (#617).
- Configuration: `aws.opensearch.password` with `allow_anonymous` (the password was dropped) and `use_iam` with a
  username or password (ignored) are refused; a negative `search.fusion.fusion_weights` value is refused. A source
  directory that is a file raises `ConfigurationError`, and creating an OpenSearch index that exists raises
  `AWSServiceError` without being logged as an unexpected error (#617).

### Security
- An atomic write's temporary file (beside the file, or staged in the shared temp directory) is created 0600 and
  given the file's mode only at the rename, so other local users cannot read a private file's new text while an
  export streams; it was 0644 under the default umask.
- The image upgrades `zlib1g` to the Debian security release fixing CVE-2026-76844 and CVE-2026-85091 (HIGH),
  which the base image still carries, so the image scan passes (#632).
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
- A guardrail-blocked answer returns no sources, search results or model-written query texts, and the query span
  records only the screened query (#345). With `check_output`, the rewritten and translated query, its entities and
  LightRAG keywords (`processed_query`, `search_results.query`) are screened as model output, alongside the answer's
  check (with the synthesized results, not one round trip after another).
- IaC: queries run under a read-only role that cannot write through a `_search` path, lists only `eval/` and has no
  DynamoDB access; `bedrock:ApplyGuardrail` is granted on the configured guardrail only; unused Neptune, S3 delete
  and `GetInferenceProfile` actions are dropped, and private-mode endpoint policies admit only the account (#204,
  #211, #319, #345).
- Logs carry no query text, entity names or model output at INFO, no local variables in tracebacks, no ANSI codes
  when piped, and no relationship endpoints or output previews from gleaning; an evaluator or judge error is logged
  as its type and a digest, its text at DEBUG; a unit test fails on a new INFO+ log call that passes them (#156,
  #173, #245, #255, #606).
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
- The data-plane image drops the AWS CLI (the entrypoint syncs the corpus with `docker/s3_sync.py`, on boto3) and
  uv, is built on Python 3.12.16, and keeps the venv read-only to the app user: 939 MB instead of 1.98 GB, and no
  HIGH or CRITICAL finding with a fix. CI scans both images with Trivy (pinned by digest) and fails on such a
  finding (`.trivyignore.yaml` lists accepted ones with their reason and expiry) (#617). The build reinstalls the
  package from the copied source and fails unless the installed files are that source (uv's cache keys the project's
  wheel by its version, so a warm cache had put an earlier commit's code in an image); the image keeps the digest
  (`/app/.venv/unified-kg-rag-source.sha256`, `scripts/package_digest.py`), which the CI docker job checks.
- Releases attach SHA-256 checksums and a CycloneDX SBOM of the locked dependencies with each package's licence and
  the package version, with a signed build provenance attestation of the distributions (`gh attestation verify`)
  (#617).
- IaC: the ingestion role cannot change OpenSearch cluster settings or plugin configuration (writes under
  `_cluster`, `_plugins` and `_opendistro` are denied), nor write snapshot repositories, templates, ingest or search
  pipelines or stored scripts (`_snapshot`, `_index_template`, `_component_template`, `_template`, `_ingest`,
  `_search/pipeline`, `_scripts`) (#636, #641); the logger-credential semgrep finding is suppressed on each
  matched line, not the whole package (#617).
- CI audits every locked runtime dependency with pip-audit (`.github/pip-audit-accepted.txt` lists the two accepted
  advisories, diskcache PYSEC-2026-2447 and ragas PYSEC-2026-3046, in the `eval` extra on paths it never runs, each
  until a date past which it gates again) and
  checks the licenses of the npm packages the console bundles; the secret scan suppresses in tests only the three
  detectors synthetic credentials trip, so a real-format key there fails it.
- A failed query step (routing, query processing, follow-up rewrite, context optimization, the answer and its
  stream, conversation memory) or claim extraction step logs its error as its type and a digest, its text at DEBUG,
  as the evaluators do (#617).

[Unreleased]: https://github.com/awslabs/unified-kg-rag-on-aws/compare/81ea4987...HEAD
