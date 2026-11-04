# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

This is the practical, how-to-use guide for **unified-kg-rag-on-aws** — an AWS-native
knowledge-graph RAG framework that builds knowledge graphs from large,
multilingual document corpora and answers questions over them. It reimplements
two retrieval methodologies on one stack: **Microsoft GraphRAG**
(community-summary) and **LightRAG** (dual-level keyword), selectable per query.

- For the *what / why* and a one-minute quickstart, see [README.md](../README.md).
- For a map of every page, see the [documentation hub](./index.md); for the
  concepts and what the framework adds beyond the papers, the
  [Overview](./concepts/overview.md).
- For *internals / architecture* (hexagonal layers, ports & adapters, the
  dependency rule), see [docs/design.md](./design.md).

The six console entry points (defined as `pyproject` scripts) all start
through `application.cli.launch`, which then loads the module below:

| Script | Module | Purpose |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | Build / update the knowledge graph |
| `run-rag` | `application.cli.run_rag_chain` | Query the graph |
| `run-eval` | `application.cli.run_evaluation` | Evaluate retrieval + generation |
| `run-visualization` | `application.cli.run_visualization` | Render an exported graph (no ingestion) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | Generate domain-adapted prompts |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | Turn a public benchmark (MuSiQue) into a corpus + `run-eval` dataset |

Every CLI's `--help` ends with an example invocation and a link to its guide; every flag is in the [CLI Reference](./reference/cli.md).

## Contents

1. Prerequisites & Installation: moved to [Getting started on AWS](./getting-started/aws.md) (including [Required models](./getting-started/aws.md#required-models)), [Getting started locally](./getting-started/local.md) and [Local mode](./guides/local-mode.md)
2. Configuration: moved to [Configuration](./guides/configure.md) (every key: [Configuration Reference](./reference/configuration.md))
3. Ingestion: moved to [Ingestion](./guides/ingest.md) and [Multilingual corpora](./guides/multilingual.md)
4. Querying: moved to [Querying](./guides/query.md), [Retrieval](./concepts/retrieval.md) and the [Filters reference](./reference/filters.md)
5. Incremental indexing: moved to [Incremental indexing](./guides/incremental.md)
6. [Evaluation (`run-eval`)](#6-evaluation-run-eval)
7. [Visualization (`run-visualization`)](#7-visualization-run-visualization)
8. Prompt tuning: moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning)
9. Domain adaptation: moved to [Domain adaptation](./guides/domain-adaptation.md), [Extraction schema](./guides/extraction-schema.md), [Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md)
10. [Operations & troubleshooting](#10-operations--troubleshooting)
11. [Using the library from Python](#11-using-the-library-from-python)
12. [Limitations](#12-limitations)

Related pages: [CLI Reference](./reference/cli.md) · [Configuration Reference](./reference/configuration.md) · [Model Catalog](./reference/models.md) · [Operator Runbook](./operations.md) ·
[Design Doc](./design.md) · [`iac/README.md`](../iac/README.md)

---

## 6. Evaluation (`run-eval`)

`run-eval --data-path <dataset.json>` answers every question of a dataset and
scores the answers and their sources. Three evaluators need no LLM and cost
nothing; two use an LLM judge. Several strategies (or `--repeats`) turn the run
into a paired comparison.

Every flag with its default: [CLI Reference](./reference/cli.md#run-eval).

### Evaluators

`evaluation.enabled_evaluators` selects them; all five are on by default. The
deterministic ones skip a query that lacks their dataset fields; for a
judge-free run set `enabled_evaluators: [answer_match, retrieval, graph_aware]`.

| Evaluator | LLM | Needs in the dataset | Main metrics |
|---|---|---|---|
| `answer_match` | no | `answer` (+ `answer_aliases`) | `answer_contains` (headline), `exact_match`, `token_f1` |
| `retrieval` | no | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr` (k = `evaluation.retrieval_k`, default 5) |
| `graph_aware` | no | `expected_entities` / `expected_relationships` | entity and relationship coverage (recall), in the answer and in the retrieved sources |
| `langchain` | judge | `answer` | `correctness`, `partial_correctness` |
| `ragas` | judge | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |

The two judges need the `eval` extra (`uv sync --extra eval`, [Install](./getting-started/aws.md#install));
without it `run-eval` stops before the first query with exit `2`, naming the
extra, even with `processing.ignore_errors: true`. The judge-free run above
needs no extra. Any other enabled evaluator that cannot be built (no Bedrock
access for the judge, say) or rejects its configuration stops the run; with
`processing.ignore_errors: true` it is dropped and listed in
`run_manifest.dropped_evaluators`. Scoring starts after every answer is
generated, so an evaluator error never discards the run: the metric is
recorded as failed for that query (as is a judge reply without a usable
score, never 0), and `--max-failure-rate` decides the exit code.

#### `ragas` contexts

- `context_precision` makes one judge call per context, so it scores at most
  `evaluation.ragas_max_contexts` (default 20; `null` = no cap) top-ranked
  sources (`context_precision@N`); without the cap, strategies that report
  100+ sources hit `ragas_timeout`.
- `faithfulness` and `context_recall` see every source within
  `max_context_tokens`. By default (`evaluation.max_context_tokens: null`)
  that is the answer model's budget (`search.token_manager.max_context_tokens`),
  so the answer is graded against the context it was given; a smaller value
  logs a warning.
- Each report records what the judge saw (`judge_contexts`,
  `judge_context_tokens`, ...); none of this changes what the answer model
  saw.
- ragas usage telemetry is off (`RAGAS_DO_NOT_TRACK=true` by default); set
  `RAGAS_DO_NOT_TRACK=false` in the environment to opt back in.

#### `graph_aware`

The share of the expected graph artifacts that appear in the generated answer
(coverage = recall), with the phrase matcher of `answer_contains`. A
relationship given as `{"source": "A", "target": "B"}` or `"A -> B"` counts
when the answer mentions both endpoints; any other string must appear as a
phrase. Precision and F1 are not emitted: the entities of a free-text answer
cannot be enumerated reliably.

Answer-text coverage rises with answer length, so compare strategies on
`retrieved_entity_coverage` / `retrieved_relationship_coverage` (the same
expectations matched against the entities and relationship endpoints of the
sources the answer model was given) and read `answer_length` next to the
answer coverage. Retrieved coverage is skipped, not 0, when the context had
no source of that type (`global` by design), so check the metric's `pairs` /
`unpaired` in a comparison and `metric_outcomes`' skipped count in one run.

#### `retrieval`

Did the sources the answer model saw include the gold documents?

- **Attribution.** Text-unit sources name their file; entity, relationship
  and community-report sources are attributed to the files of the text units
  they cite, the file holding most of them first.
- **k counts files, not sources**: the sources are flattened into one list of
  distinct files in rank order, so a community report attributed to many
  files cannot inflate hit@k.
- **Per source type.** `by_source_type` (aggregated in
  `grouped_statistics.source_type`) has the same metrics per section type; its
  `text` row, the text units the answer model read verbatim, is the fairest
  cross-strategy comparison. `attributable_fraction` (attributable / reported
  sources) shows how much of the context the metrics cover.
- **Matching.** NFKC-normalized and case-insensitive on the file name, a
  reference's directories ignored: `docs/Terms.pdf`, `terms.pdf` and `terms`
  match the source `Terms.pdf`, `terms.docx` does not, and a version suffix
  is no extension (`notes.v1` does not match `notes.v2`). A title such as
  `AC/DC` is compared whole.
- **Skips.** Skipped when a query has no `reference_sources` or none of its
  sources is attributable; a query that retrieved no sources scores 0.

#### `answer_match`

Scores against `answer` and the optional `answer_aliases` (a string or a
list, top level or under `metadata`, which wins), taking the maximum.

- **`answer_contains`** is the headline: 1.0 when the gold answer or an alias
  appears in the generated answer as a whole-word phrase. Long RAG answers
  rarely equal a short gold span, so it tracks correctness far better than
  exact match. A possessive is read as its word (`Depot` matches `Depot's`;
  `Depots` is another word); `graph_aware` matches the same way.
- **Korean and CJK.** Korean particles are tolerated after a word (`AWS`
  matches `AWS는`); a bare number may carry a Korean unit (`2024` matches
  `2024년에`) but not a magnitude numeral (`3` does not match `3억`); spacing
  next to CJK letters is ignored (`3억 원` = `3억원`); a single-word CJK gold
  is matched as a substring, but not inside a longer number or Latin word
  (`2년` does not match `12년`).
- **Yes/no golds** (`yes`/`no`/`nope`, Korean `예`/`네`/`아니요`/`아니오`/
  `아닙니다`, Chinese `是`/`对`/`否`/`不是`/`没有`, Japanese `はい`/`いいえ`) are
  scored by the polarity of the first yes/no word that answers the question,
  as in HotpotQA (`there is no clause`, `네 가지`, `(예: …)` and `No. 5` are
  not answers, so `네, 맞습니다` answers yes but `네 맞습니다` does not).
- **`exact_match` and `token_f1`** score the part of the answer
  `evaluation.answer_span` names. SQuAD, HotpotQA and MuSiQue score a short
  predicted answer (a span such as `Depot`) against the gold answer, while
  the answer prompt writes a one-sentence answer and then its support, so
  the whole answer would score its length as much as its correctness:
  - `first_line` (default, free): the first line that is neither empty nor a
    Markdown heading, the one-sentence answer (`Depot stores widgets for
    Vendor.` scores token F1 1/3 against `Depot`; EM still under-counts).
  - `extracted`: the `models.roles.answer_span` model (fast tier; prompt
    `custom_prompts.answer_span_extraction_*`) copies the short span out of
    the answer, one call per distinct question and answer, reused across
    repeats and strategies. A failed call, an empty reply or a span that is
    not in the answer is scored on the first line instead and counted in
    `answer_span_statistics.fallbacks`. This is the setting for numbers
    close to the papers'.
  - `full`: the whole answer.

  Each report records `answer_span` and the `scored_span`. The normalization
  is SQuAD v1.1's (NFKC, lowercase, punctuation and English articles
  removed, whitespace collapsed), keeping punctuation that changes meaning
  (`-5`, `1.5`, `C++`); `evaluation.answer_normalization: official_squad` is
  the official script (MuSiQue's evaluation uses it too) and `hotpotqa` adds
  HotpotQA's rule that a `yes`/`no`/`noanswer` span or gold earns token F1
  only when the two are equal. Token F1 splits on whitespace, so for Chinese
  and Japanese it degrades to exact match, and Korean particles make both
  under-count.

### Eval data format

A JSON array of objects. Only `question` is required; `expected_entities` /
`expected_relationships` are needed only by `graph_aware`.

```json
[
  {
    "id": "q1",
    "question": "What are the main themes discussed in the documents?",
    "answer": "The main themes include AI, machine learning, and data processing.",
    "category": "general",
    "difficulty": "easy",
    "reference_sources": ["doc1.pdf", "doc2.txt"],
    "expected_entities": ["AI", "machine learning", "data processing"],
    "expected_relationships": [
      { "source": "AI", "target": "machine learning" },
      "machine learning -> data processing"
    ],
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- Per-item `metadata` (e.g. `search_strategy`) overrides the CLI defaults for
  that question. `id` may also be given as `query_id`.
- Mark a question the corpus cannot answer with
  `"metadata": {"answerable": false}`: no evaluator grades it; it counts only
  in `abstention_statistics` (below).
- The file is validated before any query runs. A file that cannot be read,
  an empty or non-array dataset, a missing `question`, a duplicate id, a field
  of the wrong type (a non-list `expected_entities`, an `answer_aliases` that
  is not a string or a list of strings) or `metadata` the RAG chain rejects
  (an unknown `search_strategy`, or `filters` it refuses: a malformed value
  such as an empty list, or a key the item's strategy refuses, checked under
  every compared strategy in a comparison) stops the run with
  `Error: invalid evaluation dataset:`, the item index and id, and exit code 2.

### Examples

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

Either path can be an `s3://` URI, so `run-eval` also runs as a one-off task on
the CDK stack ([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
Results are uploaded at the end with the `aws.s3.encryption` settings; a
failed upload exits non-zero and keeps the local files (path printed). S3 is
reached with the config's `aws.profile_name` / `aws.region_name`; the shell's
`AWS_PROFILE` / `AWS_REGION` fill them only when the config leaves them unset.
The principal needs `s3:GetObject` on the dataset and `s3:PutObject` on the
prefix.

Results are written as `evaluation_{results,reports,summary}_<timestamp>.json`
(`..._<strategy>_<timestamp>.json` when every query asked for, or used, one
strategy; an `auto` run is `..._auto_...`). `<timestamp>` is
`YYYYMMDD_HHMMSS_<8 hex characters>` (runs started in the same second do not
collide): find the latest run by sorting the names, not by building one.
The summary holds per-metric statistics (`metric_statistics`) and
scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics`: the same split by `search_strategy` (the strategy
  actually used, which varies under `auto`), `category`, `difficulty` and
  `source_type` (the retrieval metrics per source section type).
- `abstention_statistics`: how often the chain returned its fixed no-context
  reply ("I could not find relevant information…"): `abstained`, `answered`,
  `abstention_rate`, `per_strategy`, and an `unanswerable` block for
  `answerable: false` items. The chain returns that reply only when retrieval
  finds nothing, and a model declining in its own words is not detected, so
  on a populated index the `unanswerable` rate is a lower bound on correct
  declines. Each result carries `abstained`.
- `answer_span_statistics`: the `answer_span` exact match and token F1
  read, the reports `scored` and the extracted spans that fell back to the
  first line (`fallbacks`).
- `rerank_status_counts`: queries per [rerank status](./guides/query.md#rerank-status); each
  result carries `rerank_status` (`null` when answer generation failed
  first).
- `guardrail_blocked_counts`: blocked queries per `input` / `output` (a
  blocked item is scored on the blocked message; a blocked answer drops its
  sources, so its retrieval metrics and the RAGAS context metrics
  (`context_precision`, `context_recall`, `faithfulness`) are skipped as
  `guardrail_blocked`. A query blocked on input retrieved nothing, so it is a
  retrieval miss and scores 0 on context precision and recall, while
  `faithfulness` is skipped, as for any answer without context:
  `no_contexts`); each result carries `guardrail_blocked`.
- `run_manifest`: what was measured, so two runs can be compared:
  - the CLI arguments (URLs without credentials), versions, `git_sha` /
    `git_dirty`, the dataset (path, hashes, query count), a UTC timestamp and
    `dropped_evaluators`;
  - the model ids and `judge_is_answer_model` (a judge that is the answer
    model inflates judge scores; the default Bedrock judge, Opus 5.5, is not
    the default answer model, and falls back to another when the answer
    model is Opus 5.5; a configured `models.roles.evaluation` that is the
    answer model, `default` say, is kept and warned about). The `--pairwise-judge` of a
    comparison counts as a judge, and `pairwise_judge` records its settings;
  - `answer_scoring`: the `answer_span` and `answer_normalization` exact
    match and token F1 used (with `answer_match`), and `models.answer_span`
    the extracting model; runs scored differently are not comparable on
    those two metrics;
  - `retrieval_config_sha256`, a hash of every configuration setting that
    can change an answer or a score (model ids and ingestion settings
    included; regions, endpoints, credentials, paths, timeouts and other
    operational knobs left out), so the same evaluation hashes the same in
    every environment. The `run-eval` query flags (`--suffix`, `--top-k`,
    `--search-type`, `--retrieval-multiplier`, `--enable-thinking`) are not
    configuration: they are in the CLI arguments, not in the hash;
  - `index`: per queried suffix, the ingestion run that last wrote its vector
    indices (`pipeline_id`, `mode`, `built_at`, `documents`, the extraction
    settings, for an incremental run the `full_build` it was applied on,
    and the `communities` counters, see
    [Community growth](./guides/incremental.md#community-growth-across-runs)); best-effort, `null` without
    index metadata.

  `EvaluationManager.evaluate_dataset` builds it for library callers too
  (pass `dataset_path=` / `cli_args=`).

Per-query reports list each evaluator's `metrics` (no overall score: an
average of unrelated metrics means nothing) and record, per reported source in
rank order, `retrieved_source_ids` (the files it is attributed to, `[]` when
unattributable), `retrieved_source_types`, `retrieved_entities` and
`retrieved_relationships`.

### Comparing strategies

`run-eval --search-strategy local,mix,global --repeats 3` answers and scores
the same queries with each strategy, `--repeats` times each (the listed
strategy overrides a per-item `search_strategy`; list them in one flag, as a
second `--search-strategy` is a usage error), saves every run's files and
adds `evaluation_comparison_<timestamp>.json`:

- `strategies.<s>.metrics.<evaluator>/<metric>`: the mean over queries of the
  per-query mean over repeats, and `repeat_stdev` (run-to-run noise);
  `routed_strategies` shows how `auto` routed. The chain extracts each
  question's entities and keywords once (`search.query_extraction_cache_size`,
  1024 questions), so the repeats and the strategies that read them (`mix`
  and `hybrid` the keywords, `local` the entities) search with the same
  ones, and the run-to-run noise is the retrieval's and the answer's.
- `deltas_vs_baseline.<s>.<metric>`: paired against the first strategy over
  the queries both scored: `mean_delta`, a 95% paired bootstrap interval
  (`ci_low`/`ci_high`; 2,000 resamples, seed 0), `wins`/`ties`/`losses`, the
  per-query deltas and `unpaired`. A query whose answer generation failed
  scores 0 on every metric the other run scored (`answer_length` excepted)
  that its own strategy scores on some query, so failing cannot help a
  strategy, and a metric it never produces (`global` reports no entities,
  so no `retrieved_entity_coverage`) is not invented as a loss; a strategy
  that scored nothing is filled on every metric. The rule reads the data, so
  a metric that only the failed query could score (one item with
  `expected_relationships`, say) is not filled either
  (`failed_answers` / `failed_queries`;
  `guardrail_blocked_counts` counts blocks; `warnings` names strategies that
  failed, or had blocked, different numbers, runs whose exact match and
  token F1 read different `answer_span`s, and, with `answer_span:
  extracted`, each strategy whose extracted spans fell back to the first
  line (`answer_span_fallbacks`) and fallback rates that differ). A
  difference is
  supported when its interval excludes 0; a 10-point difference on binary
  metrics is detected in 15% of runs with 50 queries and 59% with 200, so
  compare on a few hundred queries.
- `deltas_by_category.<s>.<category>.<metric>`: the same deltas over each
  dataset `category` (MuSiQue's hop count, say). Each interval is per
  category and not corrected for the number of categories and metrics read,
  so treat a lone excluded 0 among many as exploratory.
- `pairwise_vs_baseline.<s>` (with `--pairwise-judge`): the GraphRAG /
  LightRAG head-to-head LLM judge on comprehensiveness, diversity,
  empowerment and directness (plus LightRAG's `overall` with
  `evaluation.pairwise_overall: true`), using `models.roles.evaluation`
  (prompt: `custom_prompts.pairwise_comparison_*`) on the first repeat's
  answers. Each pair is judged in both orders and a criterion counts as a
  win only when both agree; `position_consistency` (overall and per
  criterion) is the share of judged pairs whose two orders agreed, and a
  value under 0.5 is warned about: such a judge decides by position, and its
  ties say nothing about the answers. `candidate_win_rate` scores each
  decided pair 1, a tie 0.5 and a loss 0, with a bootstrap interval
  (`ci_low`/`ci_high`) over those per-query scores; it is `null` when the
  judge decided no pair. A pair where one answer failed is not judged: the
  other side wins (`candidate_errors`, `baseline_errors`; both failed:
  `both_errors`). `answerable: false` items are not judged
  (`unanswerable_skipped`). `failed` counts pairs the judge could not decide
  (a failed call, or a verdict other than `1`, `2` or `tie`), and
  `--max-failure-rate` applies to them. A judge that is the answer model is
  warned about.
- `cost.strategies.<s>`: the answers' model use per query, as `mean` /
  `p50` / `p95` over every answer of every repeat: `llm_calls_per_query`,
  `input_tokens_per_query`, `output_tokens_per_query`,
  `total_tokens_per_query`, `latency_seconds`, the `totals` and
  `cache_hits`, and with `evaluation.prices` set `usd_per_query` and
  `usd_total`. `cost.deltas_vs_baseline.<s>` pairs them against the
  baseline like the metrics (`usage/llm_calls`, `usage/input_tokens`,
  `usage/output_tokens`, `usage/total_tokens`, `usage/cost_usd`,
  `latency/response_time_seconds`, each query's mean over its repeats, with
  the bootstrap interval). A failed or timed-out answer counts with what it
  spent before failing: each answer runs in its own usage scope and timer,
  so its result keeps the calls and tokens of its attempts and the time from
  its first attempt's start to its last one's end (`failed_answers` counts
  them; a warning says so). A strategy is then not cheaper or faster for
  failing, though a fast failure does cost less: read the cost next to
  `failed_answers` and the failure warning. Saved answers without a usage
  record or a latency are counted in `answers_without_usage` /
  `answers_without_latency` and warned about,
  and a `--fail-if` on a `usage/*` or `latency/*` measure fails for them.
  The arms and repeats of one run share the chain's query extraction cache,
  so a later one that reuses an extraction pays no call for it (counted in
  `cache_hits`): its cost depends on what ran before it, and a strategy whose
  repeat count or cache hits per answer differ from its baseline's is warned
  about. To compare cold-cache cost, set `search.query_extraction_cache_size:
  0`; the quality deltas do not depend on it. Before each strategy's first
  run, `run-eval` answers one synthetic query with it, untimed, unscored and
  not counted (`evaluation.warm_up_query`, default `true`), so the first
  strategy's latency does not include opening the store connections and the
  models' first calls; it costs one extra query (its model calls) per
  strategy and run, which is in neither the run's usage nor its USD
  estimate (Bedrock bills it all the same), and `false` turns it off.
- `run_manifest` of the first run; with `--pairwise-judge`,
  `pairwise_judge_model` and the judge's `pairwise_usage`.

The comparison is also written as `evaluation_comparison_<timestamp>.csv`
(one row per non-baseline strategy, measure and query: `baseline`,
`baseline_value`, `value`, `delta`, `category`) and `.md` (the metric and
cost tables, the pairwise judge's win rates, the gates and the warnings, for
a pull request or a CI job summary).

No model prices ship with the package. For a USD estimate, fill in the
current [Amazon Bedrock prices](https://aws.amazon.com/bedrock/pricing/) of
your region, per million tokens, keyed by the model id the usage records
name (`usage.by_model`; an inference profile id such as `global.<model>` when
the factory resolved one). Answers that used a model without a price get no
estimate, and the comparison warns. `input_tokens` includes the prompt-cache
reads and writes; give their rates (`cache_read_per_million_tokens`,
`cache_write_per_million_tokens`) and they are priced apart, otherwise at
the input rate (an upper bound for reads, which cost a tenth of input on
Claude):

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
```

The estimate covers the calls the usage records: botocore's resent requests
(`request_retries`), failed calls (`failed_calls`) and the warm-up queries
are not priced, so CloudWatch's invocation count and the bill can run a
little higher (in the package's November 2026 benchmark runs: Sonnet 5.5
13% more invocations than recorded calls, input tokens within 4.6%, output
within 0.9%; Haiku 5.5 within 0.2%).

The console prints the same summary; the failure budget applies to every run.

### Rebuilding, re-judging and gating a comparison

`run-eval --from-results <dir|files>` rebuilds the comparison from the files
earlier runs saved (`evaluation_results_*` with the `evaluation_reports_*`
and `evaluation_summary_*` next to them; `evaluation.save_detailed_results:
true`, the default) without answering a query or opening a store. The runs
are grouped by strategy and compared in the order the oldest run listed them
(`--search-strategy` picks and orders them instead), so a comparison that was
stopped part-way is finished by running only the missing strategies into the
same directory, with the same `--repeats`, and rebuilding: its deltas are
those of an uninterrupted run. A strategy whose runs were all lost runs again
with `--repeats N`; one with some repeats saved runs only the missing ones
(`--repeats` of the difference), or delete its partial files and run all N.

```bash
# Stopped during naive (run with --repeats 3): run it alone into the same
# directory with the same --repeats, then rebuild.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# Re-judge the saved answers with another judge model (no answer call).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# CI: compare this run with an earlier one and fail on a supported regression.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- The saved runs of one strategy are pooled as its repeats. The rebuild
  warns when they differ in a query flag (`suffix`, `top_k`, `search_type`,
  `retrieval_multiplier`, `enable_thinking`), `retrieval_config_sha256`,
  `git_sha`, the `index` build or the answer model (keep runs of other
  settings in another directory, or list the result files), when the
  strategies ran different numbers of repeats, and when a run has no
  `evaluation_summary_*` file (its manifest, so none of these checks, nor
  the answer-scoring and self-preference ones, can read it).
- `--pairwise-judge` with `--from-results` judges the saved answers again
  (`--judge-model` sets `models.roles.evaluation` for the run, here and in a
  normal run; without `--pairwise-judge` and an LLM-judge evaluator it
  judges nothing, which is warned about); the self-preference warning
  compares that judge with the saved runs' answer model.
- `--baseline-results` makes an earlier run the baseline: each of its
  strategies is labelled `baseline:<strategy>`, and each strategy of this
  run, or of `--from-results`, is compared with the baseline of the same
  strategy (`deltas_vs_baseline.<s>` and `cost.deltas_vs_baseline.<s>`
  against `baseline:<s>`; `baselines` maps each strategy to its baseline).
  A strategy the earlier run did not run is compared with its first
  strategy, with a warning. Run the baseline with the same `--repeats` as
  the runs compared with it: repeats reuse the query extraction cache, so
  different counts shift the usage deltas (warned about). Save each run in
  a directory of its own: an `--output-directory` that is the
  `--baseline-results` directory is warned about, since the next run that
  reads it as its baseline pools this run in as repeats. Runs whose exact
  match and token F1 read the answer differently (`answer_scoring` in the
  run manifest: `evaluation.answer_span`, `answer_normalization`) are
  refused (exit 2), checked before any answer is generated; runs of different
  datasets are warned about (only their shared query ids pair).
- `--fail-if <measure>:<stat><op><value>` (repeatable) exits 1 when a
  compared strategy's paired delta meets the condition: `<measure>` is a key
  of `deltas_vs_baseline` (`<evaluator>/<metric>`) or of
  `cost.deltas_vs_baseline`, `<stat>` is `mean_delta`, `ci_low` or `ci_high`.
  `ci_high<-0.02` fails only when the whole interval is below -0.02, a
  regression the query sample supports. A strategy with no delta of the
  measure fails the gate too, and so does a cost measure (`usage/*`,
  `latency/*`) that some answers of the strategy or of its baseline have no
  value of, or when the strategy and its baseline failed different numbers
  of answers (each counts with what it spent before failing, so the delta
  compares the failures too). The result is in `comparison.gates` and the
  JSON `gate_failures`. It needs a comparison: two strategies, or
  `--baseline-results`.

### Benchmarks (MuSiQue)

`prepare-benchmark musique` turns
[MuSiQue](https://github.com/StonyBrookNLP/musique) (answerable, multi-hop,
CC BY 4.0; Trivedi et al., TACL 2022) into a corpus and a `run-eval` dataset.
Nothing of it is in this repository: by default it downloads the answerable
dev split (2,417 questions) from a pinned revision of a public mirror and
checks its SHA-256; `--data-path` reads a local `musique_ans_v1.0_*.jsonl`
instead. A full file (`musique_full_v1.0_*.jsonl`) is read too, but its
answerable half is not MuSiQue-Ans (other paragraphs and answer aliases):
its rows are labelled `metadata.benchmark: musique-full-answerable` instead
of `musique`, with a WARNING that the scores are not comparable to MuSiQue-Ans
results. The label is in its output (console and `--output-format json`),
and `run-eval` counts it in the summary (`benchmarks`, also printed under a
comparison of several strategies or repeats) and the run manifest
(`dataset.benchmarks`). It writes `corpus/` (one text file per distinct paragraph; each
question brings 2-4 supporting paragraphs and up to 18 distractors) and
`eval_data.json` (`answer`, `answer_aliases`, the supporting paragraphs as
`reference_sources`, the hop count as `category`). A `corpus/` that is not
empty is a usage error (exit 2), reported before anything is downloaded.

```bash
# 1. 100 questions (about 1,900 paragraphs), sampled with seed 0.
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. Ingest the paragraphs under their own suffix.
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. Paired comparison of both methodologies, EM / F1 on the short answer span.
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

with `musique.yaml` holding your usual config plus:

```yaml
processing:
  document_parsing:
    index_value: musique          # keep the benchmark out of your own index
evaluation:
  answer_span: extracted                 # EM / F1 on the short span the answer gives
  answer_normalization: official_squad   # MuSiQue's evaluation script normalization
  enabled_evaluators: [answer_match, retrieval]   # judge-free; add langchain / ragas if wanted
```

MuSiQue (like HotpotQA and SQuAD) scores the short answer a reader model
predicts against the gold answer and its aliases, with EM and token F1 under
the official normalization. A RAG answer is a sentence and its support, so
`answer_span: extracted` (one fast-model call per answer) has a model copy
the span out of it first; the numbers are then close to, but not the same
as, a paper's, whose readers were trained to output spans. The default
`first_line` costs nothing but scores the one-sentence answer, so its EM is
far below a paper's. The GraphRAG and LightRAG papers report no EM / F1;
they use the pairwise LLM judge (`--pairwise-judge`).

Read `answer_contains` / `token_f1` and `retrieval` (per `category`, and
`deltas_by_category` in a comparison).
Ingestion cost grows with the paragraph count (about 19 per question). Cite
the paper when you publish results.

---

## 7. Visualization (`run-visualization`)

A **standalone** renderer that draws from an exported visualization-data
JSON; it does **not** re-run ingestion or touch AWS. Every flag with its
default: [CLI Reference](./reference/cli.md#run-visualization).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

It needs the `viz` extra (`uv sync --extra viz`, [Install](./getting-started/aws.md#install)). The two
renderers are **`interactive`** (pyvis) and **`static`** (Bokeh), configured
under `graph.visualization` (`interactive.*`, `static.*`, `centrality.*`).
Ingestion itself renders nothing.

- **The input.** The community-detection stage of `run-ingestion` writes
  `visualization_data.json` on every run whose graph covers the whole corpus:
  a full build, a reset, or an incremental run in which every document is new
  or changed. An incremental run that skips unchanged documents writes
  nothing (it logs an INFO line) and leaves an earlier file in place; run one
  full build (`indexing.reset: true`) to visualize the whole graph.
- **Where.** `graph.visualization.outputs_directory`, by default
  `<cache directory>/<pipeline_id>/visualization/`. S3 cache sync uploads the
  `.json` only; re-render the HTML locally.
- **What.** Nodes (colored by their level-0 `community_id`), edges and the
  community hierarchy, without vectors. Writing it calls no model;
  `run-visualization` computes a seeded spring layout from the graph
  topology and the `centrality` metrics when it renders.
- **Size.** The interactive graph keeps the top `interactive.max_nodes` nodes
  by degree (default 2000) and lays out only those.
- Edge width and opacity are log-scaled to the graph's own weight range.
- Only files a renderer wrote are reported (an earlier file of the same name
  is removed first); when nothing is rendered (an empty graph) the command
  exits `1`.

---

## 10. Operations & troubleshooting

[IAM permissions](./getting-started/aws.md#iam-permissions) are in Getting started on AWS; error messages and their fix
are in [Operator Runbook: Common errors](./operations.md#common-errors).

### CLI exit codes and JSON output

Every CLI (`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`) exits with:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed: a failed stage or query (in `run-ingestion` also a `ConfigurationError` a stage meets, or an unreachable store), a breached `--max-failure-rate`, a failed `run-eval --fail-if` gate, a failed upload, nothing rendered |
| `2` | Usage or configuration error: a bad flag, a missing or invalid input (a `run-eval` dataset, a `run-visualization` data file, a `prepare-benchmark` data file), an invalid config file or override, an unset store endpoint, a missing extra the selected providers need (`local`) (all before any work), or a `ConfigurationError` found while running outside a stage, such as `run-ingestion`'s pre-run refusal before its first model call (the search engine refused to create an index the run needs, or an incremental run's live indices, a deletion-only run's included, were built for other embedding, language or registry settings) |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` uploads the finished runs' results to an `s3://` output (or keeps them locally and logs the path) |

A run stopped with `130` or `143` ends its process at once: model calls in flight are abandoned rather than waited for, the logs, tracing and stdout are flushed first, and a process pool's workers are ended with it. A SIGTERM or Ctrl-C cancels a query CLI's work at its next await, so its cleanup (closing store connections, `run-eval`'s upload) runs (at the interactive prompt a SIGTERM ends the session at once and a Ctrl-C the conversation); a query CLI stopped by a SIGTERM, or by a Ctrl-C while its event loop runs, that is still running 100 seconds later is ended then (inside the 120 s StopTimeout of the app container), with the status document if none was printed, and a second signal ends it at once. `run-ingestion` stops at a SIGTERM or Ctrl-C after the checkpoint or metadata write in progress instead, bounded by the StopTimeout. A CLI that finished ends its process with its exit code as soon as its exit handlers ran, so a signal while it exits cannot turn `0`, `1` or `2` into `-15` or `-2`.

Flags must be spelled out in full: a prefix such as `--output` is rejected.

`--output-format json` prints one JSON document to stdout and moves the
banner, progress and logs to stderr, so the output pipes into `jq` (not with
`--metrics-sink cloudwatch`, which also writes to stdout): the answer or
search results (`run-rag`), the pipeline id, status, the number of documents
the run recorded `FAILED` (`failed_documents`; a model outage that fails
every unit records none, one that fails some units records their documents,
retried by the next run without counting toward
`indexing.max_document_failures`), the number of text units a stage
failed on (`units_failed`: indexed without that stage's output and
extracted again by the next run; counted in a full build too, where no
registry records a document `FAILED` and `failed_documents` stays 0; the
run ends with a WARNING naming them by stage and cause), the number of documents failing
another consecutive run for an error not their own (`documents_retrying`:
an outage or a setting never excludes them; also warned of with each file
and its run count), the number of documents indexed without the claims of
some text units, whose claim extraction failed for an error of their own
(`documents_without_claims`: counted for none, also in the summary), the
number of documents left out because they failed
`indexing.max_document_failures` consecutive runs unchanged, at the same
extraction settings (a settings change starts the count over)
(`documents_excluded`, also in the summary), the model calls and tokens of
the stages the run executed (`token_usage`) and
per-stage results (`run-ingestion`), each run's summary, the comparison, the failure-budget
breaches and the failed gates (`gate_failures`) (`run-eval`), the written files (`run-visualization`), the output
path and profile (`run-prompt-tuning`), or the counts written
(`prepare-benchmark`). A run stopped by a
signal before it printed prints `{"status": "interrupted", "exit_code": 143}`
(`130` for Ctrl-C), and one that fails before it printed prints
`{"status": "failed", "exit_code": 1}` (or `2`) with the `error` it failed
with (an argparse error included), so stdout is always JSON. That holds
from the first moment: a SIGTERM while the CLI still loads, or after it
printed its result while the process exits, exits `143` too, and a Ctrl-C
while it loads `130` (the status document only while loading; a SIGINT the
parent ignores, as `nohup` does, stays ignored). A
`run-ingestion` result whose run a stage failed carries the same `error` and
`exit_code`; `run-ingestion` adds the `pipeline_id` to resume with (once the run resolved it) and, when a stopped
run's next run on the namespace must wait for writes possibly still in
flight (this run's, or those of the run whose expired lease it took over),
`next_run_writes_from` (UTC, ISO 8601).

---

## 11. Using the library from Python

The CLIs wrap two public classes. Import them, and everything else, from the
package root: the names in `unified_kg_rag.__all__` are the public API, other
module paths are internal and may change. `get_config` applies the same file
and environment layers as the CLIs ([Configuration](./guides/configure.md#how-configuration-is-loaded)).

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- A failed stage sets the returned `PipelineContext`'s `status` to `FAILED`
  rather than raising, so check it as above.
- The errors to tell apart are exported from the package root:
  `ConfigurationError` (a setting to fix; `StoreRequestRejectedError` among
  them), `InvalidFilterError`, `IndexNotFoundError`, `StoreConnectionError`,
  `NamespaceLeaseError`, `PipelineExecutionError` and `PipelineInterrupted`
  (a `KeyboardInterrupt`); all but the last are `GraphRAGException`s.
  `DataIngestionPipeline(...)` raises `SeedGraphError` (a
  `ConfigurationError`) for a seed graph it cannot import.
- `DataIngestionPipeline(..., confirm=callable)` answers the questions a run
  asks (`callable(question) -> bool`), today only a reset that would drop
  another source scope's seed graph documents; without it such a run fails
  ([Seed graphs](#d-seed-graphs-bring-your-own-knowledge-graph)).
  `DataIngestionPipeline.run` raises `ConfigurationError`,
  `StoreConnectionError` and `NamespaceLeaseError` as they are, and wraps
  any other crash outside a stage in `PipelineExecutionError`. A
  model client's own transport error (`httpx.ConnectError` from a local
  model server, say) is raised as is.
- `RAGInput` takes the `run-rag` options as fields (`search_strategy`,
  `search_type`, `top_k`, `suffix` ([index suffix](./concepts/scopes-and-namespaces.md#index-suffix)),
  `filters`, `conversation_id`, `use_memory`, `target_language`); a plain dict
  works too.
- `chain.search(...)` (`await chain.asearch(...)`) retrieves without
  answering and returns a `SearchOutput` (`search_results`,
  `processed_query`, `metadata`).
- In async code use `await chain.ainvoke(...)` and `await chain.aclose()`, and
  size the loop's executor once at startup:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- Always `close()` / `aclose()` what you build: both classes open connections
  on first use and keep them until garbage collection otherwise. A sync call
  still running on another thread when the chain closes raises
  `StoreConnectionError`; a later call reconnects.
- Importing the package configures no logging handler;
  `unified_kg_rag.setup_logging(config)` gives the CLI's logging. Without it,
  structured fields (including `structlog.contextvars` bindings) reach your
  handlers as record attributes, a trailing `_` added to one named like a
  `LogRecord` attribute (`name`, `msg`, ...). `setup_logging` changes
  process-wide state, so call it only where the package owns logging: it
  replaces the structlog configuration, sets the root logger's level to
  `logging.level` and the `logging.library_levels` loggers' levels, and adds
  a stdout handler (plus the dated file with `logging.log_to_file`) to the
  root logger. Handlers already on the root logger stay: they receive every
  record too, so the console shows each line twice when one of them writes
  there, and for the package's records they get the plain message and the
  exception, not the structured fields. A repeated call replaces only the
  handlers it added.

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer`. Both constructors take injected backends, either as a
`providers=Providers(config, ...)` bundle or as individual arguments
(`doc_status=`, `vector_indexer=`, `model_factory=`, ...), not both; see
[Design Doc §15](./design.md#15-extension-guide). An indexer passed as
`vector_indexer=` / `graph_indexer=` stays yours to close: the pipeline never
closes it.

---

## 12. Limitations

- **Extraction can hallucinate** entities or relationships the text does not
  support, which then reach retrieval.
  `processing.graph_extraction.entity_grounding.enabled` (off by default)
  drops or down-weights items whose quoted `source_text` is not in the chunk.
- **Runs are not reproducible**: LLM outputs (extraction, reports, the
  `auto` router, answers) vary, so compare configurations over several runs.
  Given the same model responses and the same source scope, everything else
  is deterministic in any process, whatever its `PYTHONHASHSEED`. With the
  doc-status registry on, text-unit ids derive from the source scope, which
  defaults to the absolute source directory: the same corpus ingested from
  another directory gets other ids. Set
  `processing.document_parsing.source_scope` for ids that do not depend on
  where the corpus sits.
- **Reranking is regional** ([Required models](./getting-started/aws.md#required-models)); without it
  results keep the fusion order, as `metadata.rerank` shows
  ([Rerank status](./guides/query.md#rerank-status)).
- **Shipped backends.** Models: Bedrock (default), Ollama or
  OpenAI-compatible servers; reranking: Bedrock or a sentence-transformers
  cross-encoder ([Local mode](./guides/local-mode.md)). Stores: Neptune (or a
  Gremlin Server) with OpenSearch, or the embedded SQLite stores (one host,
  about 100,000 documents per index). Others need a `register_provider`
  builder or custom adapters ([Design Doc §15](./design.md#15-extension-guide)).
- **No serving layer**: no HTTP API, end-user authentication or rate limiting.
- **Filters are not access control**
  ([Scopes and namespaces](./concepts/scopes-and-namespaces.md#not-an-access-control-boundary)):
  separate tenants by index suffix.
- **A lighter ingestion limits the strategies.** Without community detection
  (`graph.community_detection.enabled: false`) `global` and `drift` have no
  reports; without the relationship vector index
  (`indexing.opensearch.build_relationship_vector_index: false`) `mix` and
  `hybrid` lose high-level keywords, `local` its relationship section, and
  graph expansion under a document filter keeps entities on the
  entities-index check alone, not on the relationships that reach them.
- **Incremental indexing approximates**
  ([Rebuild signals](./guides/incremental.md#rebuild-signals)): delta runs append communities
  instead of re-clustering, and stored communities and reports keep an
  entity's old id after it took its current name's; a full rebuild
  (`indexing.reset: true`) refreshes both. With DynamoDB one document's chunk
  ids must fit one 400 KB item ([*Document size limit*](./guides/incremental.md#document-size-limit)).
- **Concurrent runs need the registry**, whose write lease stops a second run
  on a namespace; without it (or through the direct `IncrementalIndexer` API)
  nothing locks, and two ingestions on one suffix can delete each other's
  index ([Concurrent runs](./operations.md#concurrent-runs)).
- **Isolating a document that kills the task takes runs**
  ([*Quarantined documents*](./guides/incremental.md#quarantined-documents)): about `log2(n) + 1` for one culprit among `n` documents that
  died together, while the deferred ones wait.
- **Markdown and HTML** need the `unstructured` extra, which
  the container image leaves out by default.
- **Seed graphs link by name only** ([*Precedence*](#seed-precedence),
  [§9 D](#d-seed-graphs-bring-your-own-knowledge-graph)): a seed node and an
  unrelated document entity of the same name are one entity. IRIs are
  stored and filterable but are no identity, `owl:sameAs` is not followed,
  and nothing is inferred.
- **Changing the embedding model** or dimension needs `indexing.reset: true`;
  an incremental run refuses to write into another model's indices
  ([Workflows](./guides/incremental.md#workflows)).

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*

## Moved sections

Sections 1 to 5, 8 and 9 of this guide, and part of section 10, moved to their own pages. The headings
below keep older links, including the `--help` output of earlier
versions, working.

### 1. Prerequisites & Installation

Moved to [Getting started on AWS](./getting-started/aws.md) and
[Getting started locally](./getting-started/local.md).

### Runtime

Moved to [Getting started on AWS: Runtime](./getting-started/aws.md#runtime).

### AWS services

Moved to [Getting started on AWS: AWS services](./getting-started/aws.md#aws-services).

### Required models

Moved to [Getting started on AWS: Required models](./getting-started/aws.md#required-models).

### Install

Moved to [Getting started on AWS: Install](./getting-started/aws.md#install).

### Authentication

Moved to [Getting started on AWS: Authentication](./getting-started/aws.md#authentication).

### IAM permissions

Moved to [Getting started on AWS: IAM permissions](./getting-started/aws.md#iam-permissions).

### Local mode (no AWS)

Moved to [Getting started locally](./getting-started/local.md) (a first
run) and [Local mode](./guides/local-mode.md) (providers, differences from
AWS mode, containers).

### Differences from AWS mode

Moved to [Local mode: Differences from AWS mode](./guides/local-mode.md#differences-from-aws-mode).

### Provider flags

Moved to [Local mode: Provider flags](./guides/local-mode.md#provider-flags).

### Local stores in containers

Moved to [Local mode: Local stores in containers](./guides/local-mode.md#local-stores-in-containers).

### Local stores with Bedrock models

Moved to [Local mode: Local stores with Bedrock models](./guides/local-mode.md#local-stores-with-bedrock-models).

### 2. Configuration

Moved to [Configuration](./guides/configure.md).

### How configuration is loaded

Moved to [Configuration: How configuration is loaded](./guides/configure.md#how-configuration-is-loaded).

### Settings most runs change

Moved to [Configuration: Settings most runs change](./guides/configure.md#settings-most-runs-change).

### Model selection notes

Moved to [Model Catalog: Model selection notes](./reference/models.md#model-selection-notes).

### Guardrail

Moved to [Guardrail](./guides/guardrail.md).

### S3 cache encryption

Moved to [Configuration: S3 cache encryption](./guides/configure.md#s3-cache-encryption).

### What the logs contain

Moved to [Configuration: What the logs contain](./guides/configure.md#what-the-logs-contain).

### Custom prompts

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts).

### Environment variable overrides

Moved to [Configuration: Environment variable overrides](./guides/configure.md#environment-variable-overrides).

### Observability

Moved to [Observability](./guides/observability.md) (settings and
recipes) and the [Telemetry reference](./reference/telemetry.md) (trace
shape, usage counters).

### 3. Ingestion (`run-ingestion`)

Moved to [Ingestion](./guides/ingest.md).

### The 12 pipeline stages

Moved to [Ingestion: The 12 pipeline stages](./guides/ingest.md#the-12-pipeline-stages).

### Index suffix

Moved to [Scopes and namespaces: Index suffix](./concepts/scopes-and-namespaces.md#index-suffix).

### Multilingual ingestion

Moved to [Multilingual corpora](./guides/multilingual.md).

### 4. Querying (`run-rag`)

Moved to [Querying](./guides/query.md) and [Retrieval](./concepts/retrieval.md).

### Search strategies — when to use which

Moved to [Retrieval: Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which).

### Answer style

Moved to [Retrieval: Answer style](./concepts/retrieval.md#answer-style).

### Context budget

Moved to [Retrieval: Context budget](./concepts/retrieval.md#context-budget).

### Rerank status

Moved to [Querying: Rerank status](./guides/query.md#rerank-status).

### Query deadline

Moved to [Querying: Query deadline](./guides/query.md#query-deadline).

### Filters

Moved to [Querying: Filters](./guides/query.md#filters) (how to use them)
and the [Filters reference](./reference/filters.md) (every field, value
typing, per-strategy applicability).

### Interactive mode & conversation memory

Moved to [Querying: Interactive mode & conversation memory](./guides/query.md#interactive-mode--conversation-memory).

### 5. Incremental indexing

Moved to [Incremental indexing](./guides/incremental.md).

### Enable it

Moved to [Incremental indexing: Enable it](./guides/incremental.md#enable-it).

### Workflows

Moved to [Incremental indexing: Workflows](./guides/incremental.md#workflows)
(rebuilding: [Rebuild](./guides/incremental.md#rebuild-indexingreset) and
[Rebuild signals](./guides/incremental.md#rebuild-signals)).

### Deletion scope

Moved to [Incremental indexing: Deletion scope](./guides/incremental.md#deletion-scope).

### Failed documents

Moved to [Incremental indexing: Failed documents](./guides/incremental.md#failed-documents).

### Quarantined documents

Moved to [Incremental indexing: Quarantined documents](./guides/incremental.md#quarantined-documents).

### Concurrent runs

Moved to [Incremental indexing: Concurrent runs](./guides/incremental.md#concurrent-runs).

### Document size limit

Moved to [Incremental indexing: Document size limit](./guides/incremental.md#document-size-limit).

### Cross-run merge

Moved to [Incremental indexing: Cross-run merge](./guides/incremental.md#cross-run-merge).

### Large / multilingual / heterogeneous corpora

Moved to [Ingestion: Large and heterogeneous corpora](./guides/ingest.md#large-and-heterogeneous-corpora).

### Cost notes

Moved to [Ingestion: Cost levers](./guides/ingest.md#cost-levers).

### 8. Prompt tuning (`run-prompt-tuning`)

Moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning).

### A. `entity_types` (cheapest, highest-impact)

Moved to [Domain adaptation: Entity types](./guides/domain-adaptation.md#entity-types).

### B. `custom_prompts` overrides

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts)
(the five prompts without an override:
[Prompts without an override](./guides/domain-adaptation.md#prompts-without-an-override)).

### 9. Domain adaptation

Moved to [Domain adaptation](./guides/domain-adaptation.md) (entity types,
prompts, prompt tuning), [Extraction schema](./guides/extraction-schema.md),
[Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md).

### C. Extraction schema

Moved to [Extraction schema](./guides/extraction-schema.md).

### From an OWL/RDFS ontology

Moved to [Ontology](./guides/ontology.md) (install, configure, preview,
freeze) and the [Ontology mapping reference](./reference/ontology-mapping.md)
(options, what is read, mapping rules, worked example).

- <a id="ontology-load-cost"></a>Load cost: [Ontology: Load cost](./guides/ontology.md#load-cost).
- <a id="ontology-what-a-change-re-runs"></a>What a change re-runs: [Ontology: What a change re-runs](./guides/ontology.md#what-a-change-re-runs).
- <a id="ontology-preview"></a>Preview: [Ontology: Preview](./guides/ontology.md#preview).

### D. Seed graphs (bring your own knowledge graph)

Moved to [Seed graphs](./guides/seed-graphs.md) and the
[Seed graph formats reference](./reference/seed-graph-formats.md) (JSONL,
CSV, RDF).

- <a id="seed-precedence"></a>Precedence: [Seed graphs: Precedence](./guides/seed-graphs.md#precedence-documents-decide).
- <a id="seed-lifecycle"></a>Lifecycle: [Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle).
- <a id="seed-cost-and-limits"></a>Cost and limits: [Seed graphs: Cost and limits](./guides/seed-graphs.md#cost-and-limits).
