# Evaluation output

What `run-eval` writes: the result files and their names, the summary of a
run and its run manifest, the per-query reports, every field of a
comparison, the rules a rebuilt or gated comparison follows, and how each
evaluator scores. Running an evaluation, the dataset format and reading a
comparison: [Evaluation](../guides/evaluate.md). Every flag:
[CLI Reference](./cli.md#run-eval).

## Result files

Each `run-eval` invocation writes into a subdirectory of its own of the
outputs directory (`--output-directory`, else `evaluation.outputs_directory`),
named `run_<YYYYMMDD_HHMMSS>_<8 hex characters>`, the run manifest's
`run_id` (`evaluation.run_subdirectories: false` writes into the directory
itself, as before). Results are written as `evaluation_{results,reports,summary}_<strategy>_<timestamp>.json`
when every query asked for, or used, one strategy (an `auto` run is
`..._auto_...`), and as `evaluation_{results,reports,summary}_<timestamp>.json`
otherwise; the results and reports only with `evaluation.save_detailed_results`
(default on). `<timestamp>` is `YYYYMMDD_HHMMSS_<8 hex characters>` (runs
started in the same second do not collide). The names sort by time only
within one strategy: find the latest run by modification time (`ls -t`) or by
the timestamp part, not by building a name.
A comparison adds `evaluation_comparison_<timestamp>.json`, `.csv` and `.md`
([Comparison](#comparison)).

Each row of `evaluation_results_*` is one answer: the question, answer,
ground truth, retrieved contexts and source provenance, `response_time`,
`usage`, `error` / `error_message`, `rerank_status`, `degraded_error`,
`degraded_steps`, `guardrail_blocked`,
`category`, the query's `metadata`, and the chain's decline and
sufficiency fields:

- `abstained`: the chain declined (the fixed no-context reply, a
  `[NO_ANSWER]` decline or the insufficient-context policy).
- `abstention_reason`: why: `no_context`, `marker` or
  `insufficient_context`; `null` for an answer.
- `context_sufficient`: `false` when the strategy flagged its own context
  insufficient (`iterative` does), `null` when the strategy did not judge it.
- `insufficiency_reason`: the strategy's reason for that flag (for
  `iterative`, its stop reason); `null` otherwise.
- `iterative`: the `iterative` strategy's run record (`base`,
  `rounds`, `stop_reason`, `degraded`, `llm_calls`, `usage`, `searched`,
  `verdicts`, `added_results`, `round_0_rerank`;
  [Querying](../guides/query.md#follow-up-rounds-for-multi-hop-questions)); `null` for the
  other strategies.
- `structured`: the `structured` strategy's run record (`status`,
  `reason`, `fallback_strategy`, `plan`, `plan_description`, `operation`,
  `row_count`, `truncated`, `exact`, `linked`, `repairs`, `backend`,
  `elapsed_ms`, `usage`; [Structured graph queries](../guides/structured-queries.md#when-no-plan-answers));
  `null` for the other strategies.

Results saved before these fields read back with them `null`.

## Run summary

The summary holds per-metric statistics (`metric_statistics`) and
scored/failed/skipped counts (`metric_outcomes`), plus:

- `grouped_statistics`: the same split by `search_strategy` (the strategy
  actually used, which varies under `auto`), `category`, `difficulty` and
  `source_type` (the retrieval metrics per source section type).
- `abstention_statistics`: how often the chain declined (`abstained`: its
  fixed no-context reply, "I could not find relevant information…" or its
version in the question's language, read from `metadata.abstained`, never
from the text, and under
  `search.abstention.mode: mark` an answer starting with `[NO_ANSWER]` or one
  declined by the insufficient-context policy): `completed` (results whose
  answer generation did not fail, declines included) = `abstained`
  (declined) + `answered` (not declined), `failed` (answer generation
  failed: no decline), `abstention_rate`
  (`abstained / (completed + failed)`), and the same per strategy actually used
  (`per_strategy`). "Answered" means not declined everywhere, as in
  `answer_match` `coverage`. With `answerable: false` items:
  `unanswerable` and `answerable` blocks of the same shape, `precision`
  (declines on unanswerable items / all declines; `null` without any),
  `recall` (unanswerable items declined, over all of them: a failed answer
  is no decline) and `f1`. With `answer_match`:
  `answer_match` {metric: `coverage`, `accuracy`, `selective_accuracy`,
  `overall_accuracy`} over the answerable items (a decline or a failed
  answer scores 0;
  `overall_accuracy`, present with unanswerable items, counts a decline on
  one as correct). With `mode: off` a model declining in its own words is not
  detected, so `recall` is a lower bound. Each result carries `abstained`
  and `abstention_reason` ([Result files](#result-files)); an
  `answer_match` report of a decline carries `abstained: true` and 0 on
  every metric
  ([Unanswerable questions and declines](../guides/evaluate.md#unanswerable-questions-and-declines)).
- `answer_span_statistics`: the `answer_span` exact match and token F1
  read, the reports `scored` and the extracted spans that fell back to the
  first line (`fallbacks`).
- `rerank_status_counts`: queries per [rerank status](../guides/query.md#rerank-status); each
  result carries `rerank_status` (`null` when answer generation failed
  first). The `failed` ones count against `--max-failure-rate` over
  `applied` + `failed`.
- `degraded_error_counts`: answers whose retrieval absorbed a step failure
  (a global map batch, a DRIFT follow-up, an iterative round, a step
  `processing.ignore_errors` skipped: the query translation and the query
  entity and keyword extractions among them), per error type; each result carries
  `degraded_error`.
- `degraded_step_counts`: the same answers per degraded step (the chain's
  `metadata.degraded`: `retrieval`, `global_map`, `drift_action`,
  `translation`, `routing`, ...; an answer counted once per step); each
  result carries `degraded_steps`. `degraded_queries` counts the answers
  with any, an unreadable model answer (`iterative_controller`) included:
  they count against `--max-failure-rate` over every query.
- `guardrail_blocked_counts`: blocked queries per `input` / `output` (a
  blocked item is scored on the blocked message; a blocked answer drops its
  sources, so its retrieval metrics and the RAGAS context metrics
  (`context_precision`, `context_recall`, `faithfulness`) are skipped as
  `guardrail_blocked`. A query blocked on input retrieved nothing, so it is a
  retrieval miss and scores 0 on context precision and recall, while
  `faithfulness` is skipped, as for any answer without context:
  `no_contexts`); each result carries `guardrail_blocked`.
- `structured_statistics`: over the answers with a `structured` record,
  their count (`queries`), per `status` and `fallback_reasons`, and the
  `valid_plan_rate` (plans that reached the graph store), `repair_rate`,
  and of the answered ones the `truncated_rate` and `exact_rate`; `{}`
  without such answers. A strategy comparison reports it per arm.
- `iterative_statistics`: over the answers with an `iterative` record,
  their count (`queries`), the follow-up `rounds` per answer (a histogram,
  and `mean_rounds`), the `stop_reasons`, `max_rounds_rate` (the share the
  round cap ended, not the controller) and the `degraded` runs; `{}`
  without such answers. A comparison reports it per arm.
- `benchmarks`: the questions per benchmark label of a
  [prepared benchmark](../guides/benchmarks.md) dataset (also
  `dataset.benchmarks` in the run manifest).
- `run_manifest` ([below](#run-manifest)).

The console prints the same summary.

## Run manifest

`run_manifest` records what was measured, so two runs can be compared:

- the CLI arguments (URLs without credentials), versions, `git_sha` /
  `git_dirty`, the dataset (path, hashes, query count, and `dense_script`:
  whether a question, gold answer or alias holds Han, Hangul or Kana
  text), a UTC timestamp, `dropped_evaluators` and `usage_schema` (`3`:
  the usage records count the rerank requests and their search units as
  Bedrock bills them; `2` counted one unit per 100 documents, too few for
  long ones; a run saved without it counted none);
- from `run-eval`: `run_id`, the invocation that answered the run (one per
  `run-eval`, shared by its strategies and repeats), with several runs
  `run_order` (`interleaved` or `sequential`, `evaluation.run_order`),
  with `--repeats` `repeat_extractions` (`resample` or `reuse`), and
  `repeat` (1-based; the interleaved runs share a start time, so a
  rebuilt comparison pairs repeat r with repeat r by it); with `--arm`,
  the run's `arm` (`name`, `overlay` path, `overlay_sha256`, `suffix`) and
  `compared_runs`, the run labels in comparison order; after
  `--rescore`, `rescored` (when, from which file, by which code and with
  which dataset; the rest of the manifest is the answering run's, with the
  scoring entries replaced);
- the model ids and `judge_is_answer_model` (a judge that is the answer
  model inflates judge scores; the default Bedrock judge, Opus 5.5, is not
  the default answer model, and falls back to another when the answer
  model is Opus 5.5; a configured `models.roles.evaluation` that is the
  answer model, `default` say, is kept and warned about). The `--pairwise-judge` of a
  comparison counts as a judge, and `pairwise_judge` records its settings
  (`overall`, `mode`, and in the differential mode the `overlap_model`); so
  does the `assertion` evaluator, and `assertion_scoring` records its
  `trials`. `models.judge_effort` is the judges' reasoning effort
  (`evaluation.judge_effort`, else the judge model's tier effort);
- `answer_scoring`: the `answer_span` and `answer_normalization` exact
  match and token F1 used and the token-F1 tokenization they imply
  (`token_f1_tokens`: `script_aware_v2` or `whitespace`) (with
  `answer_match`), with an `extracted` span its `span_model` and
  `span_effort`, and `models.answer_span` the extracting model; runs
  scored differently are not comparable on those two metrics (a run saved
  before the span model was recorded is warned about, not refused);
- `citations`: `enabled` (`search.citations.enabled`: the answer prompt
  asked for markers) and `scoring` (`file_weighted`, how the citation
  metrics weigh a citation's files); a comparison warns when its runs
  differ in either;
- `retrieval_config_sha256`, a hash of every configuration setting that
  can change an answer or a score (model ids and ingestion settings
  included; regions, endpoints, credentials, paths, timeouts and other
  operational knobs left out), so the same evaluation hashes the same in
  every environment. The `run-eval` query flags (`--suffix`, `--top-k`,
  `--search-type`, `--retrieval-multiplier`) are not
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
`retrieved_relationships`, and per citation the answer uses
`cited_source_ids` (its files; `null` with citations off).

## Comparison

`run-eval --search-strategy local,mix,global --repeats 3` (several
strategies, or `--repeats`) saves every run's files and adds
`evaluation_comparison_<timestamp>.json`
([Comparing strategies](../guides/evaluate.md#comparing-strategies)):

- `strategies.<s>.metrics.<evaluator>/<metric>`: the mean over queries of the
  per-query mean over repeats, and `repeat_stdev` (run-to-run noise);
  `routed_strategies` shows how `auto` routed. The chain extracts each
  question's entities and keywords once per repeat
  (`search.query_extraction_cache_size`, 1024 questions, keyed by the
  repeat under `evaluation.repeat_extractions: resample`, the default), so
  the strategies that read them (`mix` and `hybrid` the keywords, `local`
  the entities) search with the same ones within a repeat, and
  `repeat_stdev` includes the extraction's run-to-run noise as well as the
  retrieval's and the answer's. With `repeat_extractions: reuse` every
  repeat reuses the first one's extractions, and `repeat_stdev` leaves the
  extraction's noise out.
- `deltas_vs_baseline.<s>.<metric>`: paired against the first strategy (an
  `--arm` run `<s>@NAME` against `<s>`; `baselines` names each one's) over
  the queries both scored: `mean_delta`, a 95% paired bootstrap interval
  (`ci_low`/`ci_high`; 2,000 resamples, seed 0), `wins`/`ties`/`losses`, the
  per-query deltas, `unpaired` and `p_value` (a two-sided sign-flip
  randomization test of the mean delta: exact over every sign pattern up to
  10 non-zero deltas, else 2,000 random patterns, seed 0; zero deltas carry
  no sign). `multiplicity` sorts the deltas into families, each
  Holm-adjusted over its tests: `primary` (`primary_metric`:
  `evaluation.primary_metric` / `--primary-metric`, default
  `answer_match/answer_contains` when scored; its tests are the compared
  strategies), `secondary` (every other metric with per-query deltas),
  `cost` (`cost.deltas_vs_baseline`) and `pairwise` (each pairwise-judge
  criterion of `pairwise_vs_baseline`, its per-query scores tested against
  a 0.5 win rate; its adjusted interval is of the win rate). The `--arm`
  runs, each against its own strategy's base run, have families of their
  own, the same four prefixed `arm_` (`arm_primary`, ...), apart from the
  strategies against the first. Each family reads at least
  20 x tests / alpha resamples (`multiplicity.families.<f>.resamples`;
  `evaluation.resamples`, 2,000, is the base): a `p_value` under alpha is
  drawn again from that many sign patterns (`p_value_resamples`), since
  the Monte Carlo p-value is never below 1 / (1 + resamples) and Holm
  multiplies the smallest by the family's size, and the simultaneous
  interval's tails hold enough resampled means; closed-book and A/A arms
  (`closed_book_arms`, `aa_arms`) are in none (`unadjusted`). Each member delta has `family`, `adjusted_p_value` and
  `adjusted_ci_low` / `adjusted_ci_high`, the bootstrap interval at
  confidence 1 - 0.05 / m for a family of m tests (simultaneous over it, so
  a little wider than Holm's decision: a delta can have `adjusted_p_value`
  under 0.05 with an adjusted interval touching 0). `abstention/precision`
  (a ratio), `deltas_by_category` and `deltas_by_closed_book` have no
  adjusted values. A query whose answer generation failed
  scores 0 on every metric the other run scored (`answer_length` and the
  lower-is-better `assertion_contradiction_rate` and
  `negative_assertion_violation_rate`, where 0 is the best score, excepted:
  there it is unpaired) that its own strategy scores on some query, so failing cannot help a
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
  `guardrail_blocked_counts` counts blocks, `rerank_status_counts` and
  `degraded_answers` (of `answers`) the degraded ones; `warnings` names
  strategies that failed, or had blocked, different numbers, each run with
  rerank failures or degraded answers and its rate (louder, `NOT
  COMPARABLE AS IS`, when the rates differ), runs whose exact match and
  token F1 read different `answer_span`s, and, with `answer_span:
  extracted`, each strategy whose extracted spans fell back to the first
  line (`answer_span_fallbacks`) and fallback rates that differ). A
  difference is
  supported when its interval excludes 0; a 10-point difference on binary
  metrics is detected in 15% of runs with 50 queries and 59% with 200, so
  compare on a few hundred queries.
- `abstention/recall` and `abstention/precision` (in `strategies.<s>.metrics`
  and `deltas_vs_baseline.<s>`, with `metadata.answerable: false` items):
  recall is per unanswerable query whether its answer declined (1) or not
  (0; a failed answer is no decline), paired like a metric, also per
  category and per query in the CSV. Precision (declines on unanswerable
  queries over all declines) is a ratio, not a per-query mean: its
  `mean_delta` is the difference of the two precisions and its interval
  recomputes both on each resample of the queries; `pairs` counts the
  queries either side declined, `queries` (in `metrics`) those the strategy
  declined, and it is absent when a side declined nothing. Both are
  `--fail-if` measures (`abstention/recall:ci_high<0`).
- `deltas_by_category.<s>.<category>.<metric>`: the same deltas over each
  dataset `category` (MuSiQue's hop count, say). Each interval is per
  category and not corrected for the number of categories and metrics read,
  so treat a lone excluded 0 among many as exploratory.
- `deltas_by_closed_book.<s>.<known|unknown>.<metric>` (with a closed-book
  arm): the same deltas over the questions the closed-book arm answered
  right in at least half of its repeats on the primary metric (`known`) and
  the rest it scored (`unknown`); the run's own `closed_book` arm, else an
  earlier run's. Exploratory and unadjusted like `deltas_by_category`.
- `aa_arms` (with an A/A `--arm`): the arm runs whose manifest records
  `arm.aa: true` (an empty overlay, or one holding only the ignored
  `evaluation` section, on the run's own index). Their deltas are the
  noise floor and stay out of every Holm family.
- `pairwise_vs_baseline.<s>` (with `--pairwise-judge`): the GraphRAG /
  LightRAG head-to-head LLM judge on comprehensiveness, diversity,
  empowerment and directness (plus LightRAG's `overall` with
  `evaluation.pairwise_overall: true`), using `models.roles.evaluation`
  (prompt: `custom_prompts.pairwise_comparison_*`) on the first repeat's
  answers, or with `evaluation.pairwise_repeats: all` (`--pairwise-repeats
  all`) on every repeat, repeat r against repeat r (`repeats` counts them;
  the judge's calls times the repeats): each query's repeats are averaged
  before the queries, the interval resamples queries (a cluster bootstrap),
  the counts are over every pair, `per_query` lists each query's outcomes
  per repeat, and `length.per_query` is keyed `<query_id>#<repeat>`. Each pair is judged in both orders and a criterion counts as a
  win only when both agree; `position_consistency` (overall and per
  criterion) is the share of judged pairs whose two orders agreed, and a
  value under 0.5 is warned about: such a judge decides by position, and its
  ties say nothing about the answers. `candidate_win_rate` scores each
  decided pair 1, a tie 0.5 and a loss 0, with a bootstrap interval
  (`ci_low`/`ci_high`) over those per-query scores (`per_query` under each
  criterion) and a sign-flip `p_value` of those scores against 0.5
  (unadjusted; `adjusted_p_value` is its Holm value in the `pairwise`
  family); all are `null` when the judge decided no pair. A pair where one answer failed is not judged: the
  other side wins (`candidate_errors`, `baseline_errors`; both failed:
  `both_errors`). `answerable: false` items are not judged
  (`unanswerable_skipped`). `failed` counts pairs the judge could not decide
  (a failed call, or a verdict other than `1`, `2` or `tie`), and
  `--max-failure-rate` applies to them. A judge that is the answer model is
  warned about. `mode` is `standard` or `differential`
  (`evaluation.pairwise_mode`, `--pairwise-mode`).
  - `length` (both modes; judged pairs only): each pair's
    `baseline_tokens` / `candidate_tokens` (the script-aware estimate, about
    one token per Hangul, kana or Han character, so not whitespace words),
    `ratio` (candidate over baseline), `relative_difference` ((candidate -
    baseline) over the longer, -1 to 1) and `bucket` under `per_query`;
    `buckets.<ratio>` (`<=0.67x`, `0.67-0.91x`, `0.91-1.1x`, `1.1-1.5x`,
    `>=1.5x`: symmetric in the log ratio, so a candidate 1.5 times as long
    and one 1/1.5 times as long sit in mirror buckets) with the pairs and,
    per criterion, the wins, ties and
    `candidate_win_rate` in that bucket; `score_correlation.<criterion>`,
    the Pearson correlation of `relative_difference` with the pair's score
    (null under 3 pairs or without variation). A win rate rising with the
    ratio, or a strong positive correlation, is the judge's length
    preference.
  - Differential mode: per pair, the `models.roles.pairwise_overlap` model
    (fast tier; prompt `custom_prompts.pairwise_overlap_*`) lists the
    statements both answers make (`shared`) and those only the `baseline`
    or the `candidate` makes, copied in the answers' language; the judge
    (prompt `custom_prompts.pairwise_differential_*`) decides each
    criterion on the unique statements, both orders, with the shared ones as
    common ground. The split runs in both slot orders, the baseline as
    Answer 1 (`baseline_first`) and the candidate as Answer 1
    (`candidate_first`), each judged in both orders: a pair's score per
    criterion is the mean of the two slots' scores, and its outcome (in
    `per_query` and the counts) the side that mean favours.
    `differential.per_query.<pair>.<slot>` holds each split and its
    `outcome`; `slots.criteria.<criterion>` the candidate's win rate per
    slot (`baseline_first_win_rate`, `candidate_first_win_rate`) and
    `slot_effect`, the first minus the second: the split model's preference
    for the second slot, near 0 for an unbiased one. `length.per_query`
    adds `shared_tokens`, `baseline_unique_tokens` and
    `candidate_unique_tokens` (means over the two splits). A split with no
    unique statement is a tie for every criterion without a judge call; a
    pair with none in either split counts in `identical_pairs` (in `judged`,
    not in `position_consistency`). A failed or malformed split (a missing
    key, a value that is not a list of strings) fails the pair.
- `closed_book_arms` (with `--closed-book`): the arms answered by the
  answer model alone, without retrieval (`closed_book`; an earlier run's
  `baseline:closed_book`). They compare like a strategy (their results carry
  `metadata.retrieval: "none"` and no source, so their retrieval and
  context metrics score an empty list; read their answer metrics), are
  judged by `--pairwise-judge`, and are skipped by `--fail-if`.
- `cost.strategies.<s>`: the answers' model use per query, as `mean` /
  `p50` / `p95` over every answer of every repeat: `llm_calls_per_query`,
  `input_tokens_per_query`, `output_tokens_per_query`,
  `total_tokens_per_query`, `latency_seconds`, the `totals` and
  `cache_hits`, and with `evaluation.prices` set `usd_per_query` and
  `usd_total`. `cost.deltas_vs_baseline.<s>` pairs them against the
  baseline like the metrics (`usage/llm_calls`, `usage/input_tokens`,
  `usage/output_tokens`, `usage/total_tokens`, `usage/cost_usd`,
  `latency/response_time_seconds`, each query's mean over its repeats, with
  the bootstrap interval). No `usage/cost_usd` delta pairs a strategy whose
  usage records count the rerank search units as billed with one saved before
  they did (a `usage_schema` below 3 in its manifest, or none; a warning
  says so): the older side's estimate prices the rerank too low or not at
  all. A failed or timed-out answer counts with what it
  spent before failing: each answer runs in its own usage scope and timer,
  so its result keeps the calls and tokens of its attempts and the time from
  its first attempt's start to its last one's end (`failed_answers` counts
  them; a warning says so). A strategy is then not cheaper or faster for
  failing, though a fast failure does cost less: read the cost next to
  `failed_answers` and the failure warning. Saved answers without a usage
  record or a latency are counted in `answers_without_usage` /
  `answers_without_latency` and warned about,
  and a `--fail-if` on a `usage/*` or `latency/*` measure fails for them.
  The arms of one repeat (and, with `repeat_extractions: reuse`, the
  repeats) share the chain's query extraction cache, so one that reuses an
  extraction pays no call for it (counted in `cache_hits`): its cost depends
  on what ran before it, and a strategy whose repeat count or cache hits per
  answer differ from its baseline's is warned about. To compare cold-cache cost, set `search.query_extraction_cache_size:
  0`; the quality deltas do not depend on it. Before each strategy's first
  run, `run-eval` answers one synthetic query with it, untimed, unscored and
  not counted, so the first strategy's latency does not include opening the
  store connections and the models' first calls; it costs one extra query
  (its model calls) per strategy and run, which is in neither the run's
  usage nor its USD estimate (Bedrock bills it all the same).
- Run order. With `evaluation.run_order: interleaved` (the default,
  `--run-order`) every query is answered by every run of the comparison
  before the next query starts, the runs' order reversed on every other
  query (ABBA) and the other way round on every other repeat (an even
  `--repeats` balances an odd-sized dataset's every query), in one batch
  under `processing.max_concurrency`; each run is then scored and saved.
  The runs of one repeat that ask a question's extraction together share
  the call in flight: the first pays for it, the others count a cache
  hit. Answered one run after another
  (`sequential`), every later run answers later, and a service whose
  latency drifts over the run, or caches warmed by the earlier runs, show
  up as latency deltas: two runs of one configuration answered so differed
  by 0.5-5.8 s per query. A comparison whose runs were not answered
  interleaved in one `run-eval` (a `sequential` run, `--from-results` of
  several invocations, any `--baseline-results`, runs saved before
  `run_order` was recorded) warns that its latency, and possibly usage,
  deltas measure time too. An interleaved comparison that is stopped
  writes no run (all of them finish together); `sequential` keeps the
  finished runs, so a long comparison you may resume by its missing
  strategies runs `--run-order sequential`.
- `baselines` (with `--baseline-results`): each strategy's baseline
  ([below](#rebuilding-and-gating-rules)).
- `gates` (with `--fail-if`): the `conditions` given and the `breaches`
  among them; the JSON output lists the breaches in `gate_failures`.
- `run_manifest` of the first run; with `--pairwise-judge`,
  `pairwise_judge_model` and the judge's `pairwise_usage`.

The comparison is also written as `evaluation_comparison_<timestamp>.csv`
(one row per non-baseline strategy, measure and query: `baseline`,
`baseline_value`, `value`, `delta`, `category`) and `.md` (the metric and
cost tables, the pairwise judge's win rates with the per-length-bucket
table, the gates and the warnings, for a pull request or a CI job summary;
a closed-book arm is marked "(no retrieval)").

## Rebuilding and gating rules

The full rules of `--from-results`, `--baseline-results` and `--fail-if`
([Rebuilding, re-judging and gating a comparison](../guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison)):

- A directory is read with its run subdirectories (one level). The saved
  runs of one strategy are pooled as its repeats only when they are one
  invocation's: runs of several `run_id`s (or one with and one without; or,
  saved before `run_id` was recorded, more runs than their
  `cli_args.repeats`), or runs that differ in a query flag (`suffix`,
  `top_k`, `search_type`, `retrieval_multiplier`),
  `retrieval_config_sha256`, `git_sha`, the `index` build or the answer
  model are refused (exit 2) by `--from-results` and `--baseline-results`
  unless `--pool-runs` is given, which pools them with a warning naming what
  differs. The rebuild also warns when the strategies ran different numbers
  of repeats, and when a run has no `evaluation_summary_*` file (its
  manifest, so none of these checks, nor the answer-scoring and
  self-preference ones, can read it).
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
  the runs compared with it: their means average that many runs, and with
  `repeat_extractions: reuse` repeats reuse the query extraction cache, so
  different counts shift the usage deltas (warned about). A baseline is a
  separate run, so the latency deltas against it are warned about (run
  order above). With `evaluation.run_subdirectories: false`, an
  `--output-directory` that is the `--baseline-results` directory is warned
  about, since the next run that reads it as its baseline would hold two
  invocations (refused without `--pool-runs`).
- Runs whose exact
  match and token F1 read the answer differently (`answer_scoring` in the
  run manifest: `evaluation.answer_span`, `answer_normalization`,
  `token_f1_tokens`; a run saved before `token_f1_tokens` was recorded
  tokenized by whitespace, and one that records `script_aware` dropped any
  one to three Hangul syllables as a particle) are
  refused (exit 2), checked once the dataset is read and before any answer
  is generated. The tokenizations differ only on Han, Hangul and Kana text,
  so `token_f1_tokens` alone is not compared when the dataset holds none
  (this run's dataset, or the manifests' `dataset.dense_script`); runs of
  different datasets are warned about (only their shared query ids pair).
- `--fail-if <measure>:<stat><op><value>` (repeatable) exits 1 when a
  compared strategy's paired delta meets the condition: `<measure>` is a key
  of `deltas_vs_baseline` (`<evaluator>/<metric>`) or of
  `cost.deltas_vs_baseline`, `<stat>` is `mean_delta`, `ci_low`, `ci_high`,
  `p_value`, `adjusted_p_value`, `adjusted_ci_low` or `adjusted_ci_high`
  (a delta without the statistic, such as `abstention/precision` for the
  adjusted ones, fails the gate).
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
  of five an interval that excludes 0 at a sign-test p of 0.06). Above the
  floor the percentile interval is still too narrow on few pairs: on
  simulated null deltas a 95% interval excluded 0 in 8% of comparisons at 10
  pairs (12% for heavy-tailed deltas) and about 5% from 20-30 pairs, so
  read a gate on under ~30 pairs as weaker than its confidence. Each gate
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

- **Attribution.** Text-unit sources name their file; entity, relationship,
  claim, community-report and summary-tree sources are attributed to the
  files of the text units they cite, the file holding most of them first.
- **k counts files, not sources**: the sources are flattened into one list of
  distinct files in rank order, so a community report attributed to many
  files cannot inflate hit@k.
- **Passages first.** The files of the text units come first, in their rank
  order, then the files of the attributed sources, in theirs. One
  high-ranked summary or report stands for every file of its lineage, and
  at its own rank it filled the top `k` with them and pushed out the
  passages ranked after it (on a MuSiQue sample, `tree` recall@5 0.59 where
  its `text` row was 0.68; with passages first both are 0.68, and `local`
  moves by -0.003 to -0.008). A strategy that reports only attributed
  sources (`global`) ranks them as before.
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
- **Citations.** `citation_precision` (over the cited citations naming a
  file, the mean share of each one's files that are references) and
  `citation_recall` (per reference, the largest `1 / n` weight of a cited
  citation of `n` files naming it, averaged over the references) score
  what the answer *cites* (`RAGOutput.citations` its `[n]` markers use; a
  graph citation stands for the files of its resolved lineage, so a hub
  entity citing ten files, one a reference, counts a tenth, not 1), kept
  per result as `cited_source_ids` and `cited_section_types`.
  `citation_precision_passages` is the share of cited passages (`text`
  sections, one file each) whose file is a reference. Skipped without `reference_sources`
  (`no_reference_sources`), with `search.citations.enabled: false`
  (`citations_disabled`) or for results saved before citations
  (`no_citation_data`); precision is also skipped for an answer that cites
  no file (`no_citations`), where recall is 0, and passage precision for
  one that cites no passage (`no_cited_passages`).

### `assertion`

Checks each answer against the item's `assertions` (`[{text, kind:
positive|negative, source?}]`, or bare strings, which are positive).

- **Calls.** Per answer, `evaluation.assertion_trials` (default 3) calls to
  `models.roles.evaluation` (prompt `custom_prompts.assertion_judgement_*`),
  each labelling every assertion `supported`, `contradicted` or `absent`
  from the answer alone. Each trial shows the assertions in a shuffled order
  seeded by the query id, one no earlier trial of the query used while any
  is left (`n` assertions have `n!` orders). A deterministic judge
  (temperature 0) labels an order the same way each time, so an item with
  fewer orders than trials (one assertion, or two with three trials)
  repeats a call; `distinct_orders` in the report counts the orders used.
  The judge calls of every query and trial share
  `processing.max_concurrency`. The labels are fixed tokens in a JSON
  reply, so the parse does not depend on the language.
- **Scores.** An assertion takes a label when more than half of the
  successful trials gave it (BenchmarkQED's pass rule). Per query:
  `assertion_recall` = positive assertions supported / positive assertions;
  `assertion_contradiction_rate` = positive ones contradicted / positive
  ones; `negative_assertion_violation_rate` = negative ones supported /
  negative ones. The summary's means are over queries. Each report records
  `assertion_labels` (per assertion, the share of trials per label),
  `trials`, `distinct_orders` and `failed_trials`.
- **Failures and skips.** A trial whose call fails, or whose reply misses an
  assertion, labels one twice or uses another label, fails; the scores read
  the others. A query with no successful trial records its metrics in
  `failed_metrics` (counted against `--max-failure-rate`), never as 0. A
  query without assertions skips all three metrics (`no_assertions`), one
  without negative or positive ones the metrics of that kind.

### Generated assertion sets

`prepare-benchmark assertions --source-directory <dir> --output-directory
<out>` writes three files (and refuses an `<out>` that holds any of them):

- `eval_data.json`: the run-eval dataset. `query_id` is
  `autoq-<class>-<n>`, `category` the class (`data_local`, `data_global`,
  `data_linked`, `activity_local`, `activity_global`). Data questions carry
  `assertions` (positive, `source`: the text-unit ids they rest on) and
  `reference_sources` (the files of those units, for `retrieval`);
  `metadata` holds `benchmark: autoq`, `question_class`, `topic`,
  `entities`, `clusters`, `source_text_units`, `assertion_importance` and,
  for activity questions, the `persona` and `task`.
- `text_units.jsonl`: one line per cited unit: `id` (`<file>#<n>`), `file`
  (relative to the corpus directory), `start` / `end` character offsets and
  `text`.
- `generation_report.json`: `seed`, `settings`, `language`, the
  `models` used, corpus counts, `questions` per class, the number of
  `assertions`, `dropped` (per reason: failed calls, candidates without
  claims or assertions, assertions that failed validation, duplicates) and
  `usage` per step (`questions`, `claims`, `assertions`, `validation`,
  `activity_context`).

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
- **Structured golds.** An item's `metadata.answer_type` changes how
  `answer_contains` reads it: `number` compares the first number the
  answer (its extracted span, with `extracted`) states, in digits, English
  words, a Korean native numeral before a counter (`세 곳`, `열두 개사`;
  `없습니다` is 0) or a Han numeral before one (`三社`), with the gold's
  number, so `Three vendors ... Two of them ...` is 3 and does not match
  the gold 2. Numbers that count nothing are skipped: list enumerators
  (`1. Acme`), citation markers (`[2]`), years (`as of 2024`, `2024년`,
  `2024年`), ordinals (`두 번째`, `三つ目`, `第三`), a negated number
  (`not one but three`, `一つではなく`, `不是一家`, `한 곳이 아니라`), the
  whole of a partitive (`Of the three vendors, only one`, `三家中只有一家`,
  `세 곳 중 한 곳`, `三社のうち一社` are 1), and
  `no` answering a yes/no question or before `single`, `more` and the like;
  `not ... any`, `not one`, `not a single`, `한 곳도`, `一社も`, `一家也`,
  `没有` and `いません`/`ありません` (not `問題ありません` or `ではありません`)
  are 0. English scale words count (`three hundred and twelve` is 312),
  `两`/`兩` is 2 and a space may stand before a Han counter (`三 社`). `set`
  scores 1.0 when every one of `metadata.answer_items` appears in the
  answer, in any order or layout (a numbered list too); with
  `metadata.answer_candidates` (the names the answer picks from,
  `prepare-benchmark structural` writes every vendor) the candidates the
  part that answers names must be exactly those items, so an answer
  listing every vendor scores 0. That part is the extracted span (with
  `extracted`, when it did not fall back), else the answer's first line
  that is no heading and the list items right after it (blank lines
  between them skipped, bullets and enumerators such as `1.`, `1、` and
  `・` dropped): a later `Delta is fourth` names no candidate.
  With `metadata.decline_is_correct` (a question about something the data
  does not hold) a decline scores 1 on every `answer_match` metric. A
  Korean counter (`곳`, `군데`, `가지`, `개사`) keeps a number's value like a
  unit (`3` matches `3곳입니다`).
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
  removed, whitespace collapsed; for an answer in German, Spanish, French,
  Italian or Portuguese that language's articles too, French and Italian
  elided ones split off first, and its number words from zero to twelve
  read as counts; the answer's language is the question's, recorded as
  `answer_language`), keeping punctuation that changes meaning
  (`-5`, `1.5`, `C++`); `evaluation.answer_normalization: official_squad` is
  the official script (MuSiQue's evaluation uses it too) and `hotpotqa` adds
  HotpotQA's rule that a `yes`/`no`/`noanswer` span or gold earns token F1
  only when the two are equal. Under `default` token F1 tokenizes by script
  (`token_f1_tokens: script_aware_v2` in each report): a Han, Hangul or Kana
  letter is a token of its own, as the CMRC 2018 and KorQuAD 1.0 scripts
  score Chinese and Korean by character (they split Latin words and
  numbers too; these stay whole), and a run of Korean particles or copula
  endings from their closed class attached to a gold word (`서울은`,
  `서울에서는`, `서울입니다` for `서울`) is dropped first, which KorQuAD
  does not do: so `東京都です` scores 0.75 against `東京都` and
  `서울대학교` 0.57 against `서울` (other Hangul after a gold word is
  another word). `answer_contains` keeps the wider rule (any one to three
  Hangul syllables after a gold word). Under `official_squad` and `hotpotqa` tokens are
  whitespace words, as in those scripts (`whitespace`), so for Chinese and
  Japanese token F1 degrades to exact match there. Exact match compares
  whole strings, so a Korean particle or a CJK copula still makes it
  under-count.
