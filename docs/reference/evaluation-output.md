# Evaluation output

What `run-eval` writes: the result files and their names, the summary of a
run and its run manifest, the per-query reports, every field of a
comparison, the rules a rebuilt or gated comparison follows, and how each
evaluator scores. Running an evaluation, the dataset format and reading a
comparison: [Evaluation](../guides/evaluate.md). Every flag:
[CLI Reference](./cli.md#run-eval).

## Result files

Results are written as `evaluation_{results,reports,summary}_<strategy>_<timestamp>.json`
when every query asked for, or used, one strategy (an `auto` run is
`..._auto_...`), and as `evaluation_{results,reports,summary}_<timestamp>.json`
otherwise; the results and reports only with `evaluation.save_detailed_results`
(default on). `<timestamp>` is `YYYYMMDD_HHMMSS_<8 hex characters>` (runs
started in the same second do not collide). The names sort by time only
within one strategy: find the latest run by modification time (`ls -t`) or by
the timestamp part, not by building a name.
A comparison adds `evaluation_comparison_<timestamp>.json`, `.csv` and `.md`
([Comparison](#comparison)).

## Run summary

The summary holds per-metric statistics (`metric_statistics`) and
scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics`: the same split by `search_strategy` (the strategy
  actually used, which varies under `auto`), `category`, `difficulty` and
  `source_type` (the retrieval metrics per source section type).
- `abstention_statistics`: how often the chain declined (`abstained`: its
  fixed no-context reply "I could not find relevant information…", and under
  `search.abstention.mode: mark` an answer starting with `[NO_ANSWER]` or one
  declined by the insufficient-context policy): `abstained`, `answered`,
  `abstention_rate`, `per_strategy`. With `answerable: false` items:
  `unanswerable` and `answerable` blocks of the same shape, `precision`
  (declines on unanswerable items / all declines; `null` without any),
  `recall` (unanswerable items declined) and `f1`. With `answer_match`:
  `answer_match` {metric: `coverage`, `accuracy`, `selective_accuracy`,
  `overall_accuracy`} over the answerable items (a decline scores 0;
  `overall_accuracy`, present with unanswerable items, counts a decline on
  one as correct). With `mode: off` a model declining in its own words is not
  detected, so `recall` is a lower bound. Each result carries `abstained`;
  an `answer_match` report of a decline carries `abstained: true` and 0 on
  every metric
  ([Unanswerable questions and declines](../guides/evaluate.md#unanswerable-questions-and-declines)).
- `answer_span_statistics`: the `answer_span` exact match and token F1
  read, the reports `scored` and the extracted spans that fell back to the
  first line (`fallbacks`).
- `rerank_status_counts`: queries per [rerank status](../guides/query.md#rerank-status); each
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
- `benchmarks`: the questions per benchmark label of a
  [prepared benchmark](../guides/benchmarks.md) dataset (also
  `dataset.benchmarks` in the run manifest).
- `run_manifest` ([below](#run-manifest)).

The console prints the same summary.

## Run manifest

`run_manifest` records what was measured, so two runs can be compared:

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
  [Community growth](../guides/incremental.md#community-growth-across-runs)); best-effort, `null` without
  index metadata.

`EvaluationManager.evaluate_dataset` builds it for library callers too
(pass `dataset_path=` / `cli_args=`).

## Per-query reports

Per-query reports list each evaluator's `metrics` (no overall score: an
average of unrelated metrics means nothing) and record, per reported source in
rank order, `retrieved_source_ids` (the files it is attributed to, `[]` when
unattributable), `retrieved_source_types`, `retrieved_entities` and
`retrieved_relationships`.

## Comparison

`run-eval --search-strategy local,mix,global --repeats 3` (several
strategies, or `--repeats`) saves every run's files and adds
`evaluation_comparison_<timestamp>.json`
([Comparing strategies](../guides/evaluate.md#comparing-strategies)):

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
  that scored nothing is filled on every metric. A score the evaluator
  tried and could not compute (a failed or NaN judge call) is not filled:
  `strategies.<s>.failed_metrics` counts them per metric over every repeat,
  the query is `unpaired` for that metric, a warning names a metric whose
  counts differ between strategies, and the Markdown and console tables
  print each delta's pairs and unpaired queries. The rule reads the data, so
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
  not counted, so the first strategy's latency does not include opening the
  store connections and the models' first calls; it costs one extra query
  (its model calls) per strategy and run, which is in neither the run's
  usage nor its USD estimate (Bedrock bills it all the same).
- `baselines` (with `--baseline-results`): each strategy's baseline
  ([below](#rebuilding-and-gating-rules)).
- `gates` (with `--fail-if`): the `conditions` given and the `breaches`
  among them; the JSON output lists the breaches in `gate_failures`.
- `run_manifest` of the first run; with `--pairwise-judge`,
  `pairwise_judge_model` and the judge's `pairwise_usage`.

The comparison is also written as `evaluation_comparison_<timestamp>.csv`
(one row per non-baseline strategy, measure and query: `baseline`,
`baseline_value`, `value`, `delta`, `category`) and `.md` (the metric and
cost tables, the pairwise judge's win rates, the gates and the warnings, for
a pull request or a CI job summary).

## Rebuilding and gating rules

The full rules of `--from-results`, `--baseline-results` and `--fail-if`
([Rebuilding, re-judging and gating a comparison](../guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison)):

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
  reads it as its baseline pools this run in as repeats.
- Runs whose exact
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
  compares the failures too). A metric gate also fails when the metric
  could not be scored on different numbers of the two sides' answers
  (`strategies.<s>.failed_metrics`): those queries are unpaired, so a judge
  that fails exactly where a strategy would lose would hide the loss. Any
  gate fails on a delta over fewer than 10 paired queries
  (`gates.MIN_GATE_PAIRS`; one pair gives a zero-width interval, five wins
  of five an interval that excludes 0 at a sign-test p of 0.06). Each gate
  on each strategy is its own test, so k strategies and m gates multiply
  the chance of a breach on noise ([Gates](../guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison)).
  The result is in `comparison.gates` and the
  JSON `gate_failures`. It needs a comparison: two strategies, or
  `--baseline-results`.

## Evaluator details

How each evaluator scores. Which evaluators exist, what they need in the
dataset and how a failure is handled:
[Evaluation: Evaluators](../guides/evaluate.md#evaluators).

### `ragas` contexts

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

### `graph_aware`

The share of the expected graph artifacts that appear in the generated answer
(coverage = recall), with the phrase matcher of `answer_contains`. A
relationship given as `{"source": "A", "target": "B"}` or `"A -> B"` counts
when the answer mentions both endpoints; any other string must appear as a
phrase. Expected entities with the same entity key (below) are one
expectation, met when the answer mentions any of their spellings, so a
duplicate does not count twice (`expected_entity_count` counts them once).
Precision and F1 are not emitted: the entities of a free-text answer
cannot be enumerated reliably.

Answer-text coverage rises with answer length, so compare strategies on
`retrieved_entity_coverage` / `retrieved_relationship_coverage` (the same
expectations matched against the entities and relationship endpoints of the
sources the answer model was given) and read `answer_length` next to the
answer coverage. Retrieved coverage compares entity keys, not phrases: an
expected entity, or a relationship endpoint, is covered by a retrieved
entity of the same key (case, spacing next to CJK letters, quotes and a
legal form such as `Corp.` ignored, as entity resolution links them:
`ACME Corp.` covers `Acme`), not by a longer name containing it (`Seoul
National University` does not cover `Seoul`). Retrieved coverage is skipped, not 0, when the context had
no source of that type (`global` by design), so check the metric's `pairs` /
`unpaired` in a comparison and `metric_outcomes`' skipped count in one run.

### `retrieval`

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

### `answer_match`

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
  (`2년` does not match `12년`). In Chinese and Japanese, which put no space
  around a Latin word or a number, any gold is also matched as such a
  substring (`AWS` matches `AWSは`, `2024` matches `2024年`; `AI` does not
  match `airport附近` nor `2` `12年`); next to Hangul the Korean rules
  above decide.
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
