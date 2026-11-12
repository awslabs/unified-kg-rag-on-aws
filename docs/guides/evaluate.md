# Evaluation (`run-eval`)

> 🇰🇷 한국어판: [evaluate.ko.md](./evaluate.ko.md)

`run-eval --data-path <dataset.json>` answers every question of a dataset and
scores the answers and their sources. Three evaluators need no LLM and cost
nothing (the default); two use an LLM judge. Several strategies (or `--repeats`) turn the run
into a paired comparison.

Every flag with its default: [CLI Reference](../reference/cli.md#run-eval).
What the result files hold, field by field, and how each evaluator scores:
[Evaluation output](../reference/evaluation-output.md). A public benchmark
as a corpus and a dataset: [Benchmarks](./benchmarks.md).

## Evaluators

`evaluation.enabled_evaluators` selects them (`run-eval --evaluators
a,b` for one run); the three LLM-free ones are on by default and skip a
query that lacks their dataset fields. The judges are opt-in: `ragas` makes
several judge calls per answer for its five metrics, `context_precision` one
per retrieved context (up to `evaluation.ragas_max_contexts`, 20), and needs
the `eval` extra; `assertion` needs assertions in the dataset. To add `ragas`:
`run-eval --evaluators answer_match,retrieval,graph_aware,ragas`, or list it
in `enabled_evaluators`. The `langchain` evaluator was removed (its
`correctness` repeated `ragas`' `answer_correctness`); a config that still
lists it, or sets `evaluation.langchain_metrics`, loads with a warning and
without it.

| Evaluator | LLM | Needs in the dataset | Main metrics |
|---|---|---|---|
| `answer_match` | no | `answer` (+ `answer_aliases`) | `answer_contains` (headline), `exact_match`, `token_f1` |
| `retrieval` | no | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr` (k = `evaluation.retrieval_k`, default 5) |
| `graph_aware` | no | `expected_entities` / `expected_relationships` | entity and relationship coverage (recall), in the answer and in the retrieved sources |
| `ragas` | judge | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |
| `assertion` | judge | `assertions` | `assertion_recall`, `assertion_contradiction_rate`, `negative_assertion_violation_rate` (opt-in) |

`assertion` is opt-in too (add it to `enabled_evaluators`): it checks each
answer against the item's assertions, `evaluation.assertion_trials` (default
3) judge calls per answer, and needs no extra; see
[Assertions, length control and a closed-book bound](#assertions-length-control-and-a-closed-book-bound).
The `ragas` judge needs the `eval` extra (`uv sync --extra eval`, [Install](../getting-started/aws.md#install));
without it `run-eval` stops before the first query with exit `2`, naming the
extra, even with `processing.ignore_errors: true`. The default, judge-free
run needs no extra. Any other enabled evaluator that cannot be built (no Bedrock
access for the judge, say) or rejects its configuration stops the run; with
`processing.ignore_errors: true` it is dropped and listed in
`run_manifest.dropped_evaluators`. Scoring starts after every answer is
generated, so an evaluator error never discards the run: the metric is
recorded as failed for that query (as is a judge reply without a usable
score, never 0), and `--max-failure-rate` decides the exit code.

How each one scores (the `ragas` context caps, `graph_aware` answer and
retrieved coverage, `retrieval` attribution and file matching,
`answer_match` phrase matching, the scored answer span and the
normalization): [Evaluation output: Evaluator details](../reference/evaluation-output.md#evaluator-details).

## Eval data format

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
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] },
    "assertions": [
      "The documents discuss machine learning.",
      { "text": "The documents describe quantum hardware.", "kind": "negative" }
    ]
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- `assertions` (for the `assertion` evaluator): statements a good answer
  supports (a string, or `{"text", "kind": "positive"}`) or must not state
  (`"kind": "negative"`, a known hallucination, say), each with an optional
  `source` for a reviewer. Write them in the language the answers are in.
- Per-item `metadata` (e.g. `search_strategy`) overrides the CLI defaults for
  that question. `id` may also be given as `query_id`.
- `metadata.answer_type` says how `answer_match` reads a structured gold:
  `number` matches when the answer's first line writes the gold anywhere
  (digits, number words, Korean and Han numerals before a counter: `세
  곳`, `三社`; list enumerators, citation markers and ordinals are no
  numbers), a gold of 0 when it states there is none and writes no number
  the question does not (which number is the count is the LLM judges' to
  read), `set` needs every one of
  `metadata.answer_items` (a list of strings) in the answer in any layout,
  and with `metadata.answer_candidates` (the names the answer picks from)
  no other candidate in the part that answers (the extracted span, or the
  first line and the list right after it, so a later `Delta is fourth` is
  no pick); `metadata.decline_is_correct:
  true` scores a decline as the right answer (a question about something
  the data does not hold). `prepare-benchmark structural` writes them
  ([Evaluation output: answer_match](../reference/evaluation-output.md#answer_match)).
- Mark a question the corpus cannot answer with
  `"metadata": {"answerable": false}`: no evaluator grades it; it counts only
  in `abstention_statistics`
  ([Run summary](../reference/evaluation-output.md#run-summary),
  [Unanswerable questions and declines](#unanswerable-questions-and-declines)).
- The file is validated before any query runs. A file that cannot be read,
  an empty or non-array dataset, a missing `question`, a duplicate id, a field
  of the wrong type (a non-list `expected_entities`, an `answer_aliases` that
  is not a string or a list of strings) or `metadata` the RAG chain rejects
  (an unknown `search_strategy`, or `filters` it refuses: a malformed value
  such as an empty list, or a key the item's strategy refuses, checked under
  every compared strategy in a comparison) stops the run with
  `Error: invalid evaluation dataset:`, the item index and id, and exit code 2.

## Unanswerable questions and declines

A RAG system that answers every question also answers the ones its corpus
cannot. To measure declining, mix questions marked `"answerable": false`
into the dataset (`prepare-benchmark musique --unanswerable N` builds such a
set, filtered so the corpus does not answer them:
[Benchmarks](./benchmarks.md#unanswerable-questions)) and run with
`search.abstention.mode: mark`, so the model's declines are detected
([Querying: declined answers](./query.md#declined-answers)). Then:

- A decline on an answerable question is a wrong answer: `answer_match`
  scores it 0 (no span extraction call) and records `abstained` in its report.
- An unanswerable question is graded only by whether the answer declined.
- `abstention_statistics` reports the decline `precision` (declines that were
  on unanswerable questions), `recall` (unanswerable questions declined) and
  `f1`, the `answerable` and `unanswerable` rates, and per `answer_match`
  metric `coverage` (answerable questions answered), `accuracy` (all
  answerable questions, declines as 0), `selective_accuracy` (the answered
  ones) and `overall_accuracy` (a correct answer or a correct decline, over
  both kinds). Read coverage and accuracy together: declining more raises
  selective accuracy and lowers coverage. Its counts are `completed`
  (answers that did not fail) = `abstained` + `answered` (not declined),
  and `failed`. A failed answer is no decline: it lowers recall, and on an
  answerable question it scores 0 in accuracy, so failing cannot pass for
  declining.
- A comparison (`--search-strategy a,b`, `--baseline-results`) pairs the
  declines too: `abstention/recall` (per unanswerable query, declined or
  not) and `abstention/precision` (declines on unanswerable questions over
  all declines, its interval from resampling the queries) sit with the
  metrics in `deltas_vs_baseline`, so
  `--fail-if abstention/recall:ci_high<0` gates a supported drop.
- Each saved result keeps `abstained`, `abstention_reason`,
  `context_sufficient`, `insufficiency_reason` and, for `iterative`, its run
  record (rounds, stop reason, verdicts), so a run can be analysed from its
  files ([Result files](../reference/evaluation-output.md#result-files)).
- With `mode: off` only the empty-context reply counts as a decline, so recall
  is a lower bound. `mark` changes the answer prompt: compare a `mark` run
  with an `off` run as two arms (run `off`, then `mark` with
  `--baseline-results` pointing at the `off` run's files).

## Examples

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

Either path can be an `s3://` URI, so `run-eval` also runs as a one-off task on
the CDK stack ([After deploy](../../iac/README.md#5-evaluate-from-inside-the-vpc)).
Results are uploaded at the end with the `aws.s3.encryption` settings; a
failed upload exits non-zero and keeps the local files (path printed). S3 is
reached with the config's `aws.profile_name` / `aws.region_name`; the shell's
`AWS_PROFILE` / `AWS_REGION` fill them only when the config leaves them unset.
The principal needs `s3:GetObject` on the dataset and `s3:PutObject` on the
prefix.

Each `run-eval` writes into a subdirectory of its own of the outputs
directory, `run_<timestamp>_<id>` (`evaluation.run_subdirectories`), and each
run in it `evaluation_{results,reports,summary}_<strategy>_<timestamp>.json`
(without `<strategy>` when the queries used several; find the latest run by
modification time, `ls -t`, since names of different strategies do not sort by
time). The summary holds the per-metric
statistics, scored/failed/skipped counts, abstentions, rerank and guardrail
counts and a run manifest that records what was measured, so two runs can be
compared: [Evaluation output](../reference/evaluation-output.md). The
console prints the same summary; the failure budget
(`--max-failure-rate`) applies to every run. It counts the answers that
went out degraded as well as those that failed: queries whose rerank call
failed (a throttled rerank model keeps the fused order; counted over the
queries reranking was attempted on) and answers whose retrieval absorbed a
step failure (`degraded_queries`, per step in `degraded_step_counts`; under
`processing.ignore_errors` a failed query translation or entity or keyword
extraction, auto's router and the follow-up rewrite too), so at the default 1.0 a run whose
every rerank call failed exits 1, and `--max-failure-rate 0.05` fails one
where more than 5% did. A comparison names each run's rerank failures and
degraded answers with their rates, and warns louder (`NOT COMPARABLE AS
IS`) when those rates differ between the compared runs beyond chance (the
lowest against the highest, two-sided Fisher's exact p below 0.05), since
its deltas then measure the failures too; a difference within chance (1/40
against 0/40, say) gets a note with its p-value instead. Reranking that is off or could not be built
(`disabled`, `unavailable`) is no failed call.

## Comparing strategies

`run-eval --search-strategy local,mix,global --repeats 3` answers and scores
the same queries with each strategy, `--repeats` times each (the listed
strategy overrides a per-item `search_strategy`; list them in one flag, as a
second `--search-strategy` is a usage error), saves every run's files and
adds `evaluation_comparison_<timestamp>.json`, with a `.csv` (one row per
strategy, measure and query) and a `.md` (the tables, gates and warnings, for
a pull request or a CI job summary).

How to read it:

- **Paired deltas.** `deltas_vs_baseline.<s>.<metric>` pairs each strategy
  against the first one listed, over the queries both scored: `mean_delta`,
  a 95% paired bootstrap interval (`ci_low`/`ci_high`) and
  `wins`/`ties`/`losses`, each query averaged over its repeats first. A
  failed answer scores 0 where the other side scored, so failing cannot
  help a strategy. The tables print each delta's `pairs` and `unpaired`
  queries: a judge that could not score a query leaves it unpaired, and
  different counts per strategy are warned about.
- **When a difference counts.** Declare one metric in advance, the primary
  metric (`--primary-metric`, `evaluation.primary_metric`; default
  `answer_match/answer_contains` when answers are scored): its delta is
  adjusted only over the compared strategies. Every other metric is a
  secondary one, Holm-adjusted over all of them (and the cost measures over
  theirs): `adjusted_p_value` below 0.05 is a difference that survives the
  many deltas read, and the tables print it as `Holm p` (each family reads
  enough Monte Carlo resamples for its size, at least 20 x tests / 0.05,
  so a comparison of many strategies and arms can still reject;
  `evaluation.resamples` sets the base, 2,000); read `ci_low` /
  `ci_high` (unadjusted) or `adjusted_ci_low` / `adjusted_ci_high`
  (simultaneous over the family) for its size. Ten metrics read unadjusted
  show a difference on noise in 30-40% of comparisons of one
  configuration with itself. On a paired binary metric only the queries
  where the two strategies disagree carry information, so power depends on
  how often they do: in a simulation of this interval, a 10-point difference
  is detected in 45% of runs with 50 queries and 99% with 200 when they
  disagree on 15% of the queries, and in 28% and 73% when they disagree on
  30%. Compare on a few hundred queries, and run the same configuration
  twice to see your noise floor. `deltas_by_category` splits the deltas by dataset `category`;
  its intervals are not corrected for the many categories and metrics read,
  so treat a lone excluded 0 among many as exploratory. With
  `--closed-book`, `deltas_by_closed_book` splits them the same way into the
  questions the answer model gets right without retrieval in at least half
  of the arm's repeats (`known`, on the primary metric; on a graded metric
  such as token F1, a mean of at least 0.5; a failed answer counts 0) and
  the rest (`unknown`): the unknown ones show what retrieval adds, where a
  pooled delta can hide it. Exploratory and unadjusted too.
- **Pairwise judge.** `--pairwise-judge` adds the GraphRAG / LightRAG
  head-to-head LLM judge (comprehensiveness, diversity, empowerment,
  directness), each pair judged in both orders; read `candidate_win_rate`
  next to `position_consistency` and the win rates per answer-length bucket
  (`length`): LLM judges prefer the longer answer, so a win rate that rises
  with the length ratio is length, not quality. Each criterion has a
  sign-flip `p_value` against a 0.5 win rate, and the criteria of every
  compared strategy form one Holm family (`pairwise`), so read
  `adjusted_p_value` (`Holm p` in the report) for whether a win counts.
  `--pairwise-mode differential` judges only what differs between the
  answers (below). It judges the first repeat's answers; `--pairwise-repeats
  all` judges every repeat (its calls times `--repeats`), averaging each
  query's repeats before the queries.
- **Cost and latency.** `cost.strategies.<s>` and `cost.deltas_vs_baseline`
  give the answers' calls, tokens and latency per query, paired like the
  metrics; a failed answer counts with what it spent before failing.
- **Run order.** The runs are answered interleaved by default
  (`--run-order interleaved`, `evaluation.run_order`): each query by every
  strategy and repeat before the next query, their order reversed on every
  other query, so a service that slows down over the run, or caches the
  first runs warmed, do not show up as latency deltas (and reversed again
  on every other repeat, so an even `--repeats` balances an odd-sized
  dataset's every query; runs of one configuration answered one after
  the other differed by 0.5 to 2.3 s a query on average: see
  [Defaults and evidence](../concepts/defaults-and-evidence.md#how-the-measurements-were-run)). `--run-order sequential` answers one run after another, keeps
  the finished runs when stopped part-way, and warns that its latency
  deltas measure time too; so does a comparison of runs from separate
  invocations (`--from-results`, `--baseline-results`).
- **Repeats.** Each repeat extracts the query's entities and keywords again
  (`--repeat-extractions resample`, `evaluation.repeat_extractions`), so
  `repeat_stdev` includes the extraction's noise; it costs that fast-model
  call once per query and repeat. The strategies and arms of one repeat
  share the extraction, so their deltas compare retrieval on the same
  keywords: those that ask together share the call in flight (the first
  pays, the others count a cache hit; under ABBA the payer alternates).
  `reuse` extracts once for every repeat (the earlier behaviour).

Every field, the failure and fill rules, the warnings and what the cost
counts: [Evaluation output: Comparison](../reference/evaluation-output.md#comparison).

No model prices ship with the package. For a USD estimate, fill in the
current [Amazon Bedrock prices](https://aws.amazon.com/bedrock/pricing/) of
your region, per million tokens, keyed by the model id the usage records
name (`usage.by_model`; an inference profile id such as `global.<model>` when
the factory resolved one). Answers that used a model without a price get no
estimate, and the comparison warns. `input_tokens` includes the prompt-cache
reads and writes; give their rates (`cache_read_per_million_tokens`,
`cache_write_per_million_tokens`) and they are priced apart, otherwise at
the input rate (an upper bound for reads, which cost a tenth of input on
Claude). The rerank model is priced too, per thousand search units (usage
`rerank_search_units`: one per rerank request of up to 100 documents, where
Amazon Bedrock bills a document of more than 500 tokens, the query included,
as several: a ~4,800-character chunk is about three; the tokens are the
script-aware estimate, so the count can differ from the bill by a unit; set 0
for a local cross-encoder):

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
    "<rerank-model-id>": {per_thousand_rerank_search_units: 0.0}
```

The estimate covers the calls the usage records: botocore's resent requests
(`request_retries`), failed calls (`failed_calls`) and the warm-up queries
are not priced, so CloudWatch's invocation count and the bill can run a
little higher (measured:
[Usage records against the bill](../concepts/defaults-and-evidence.md#usage-records-against-the-bill)).

## Configuration arms

`--arm NAME=overlay.yaml` compares configurations, not only strategies: the
arm answers the same dataset with the run's configuration under the
overlay, a YAML mapping merged over it, as a chain of its own, with every
`--search-strategy`, interleaved with the base runs. Repeat the flag for
more arms; `,suffix=SUFFIX` queries another index namespace with the arm
(one ingested with other settings, say).

```bash
# Rerank off against on, and an A/A copy for the noise floor.
printf 'providers:\n  rerank:\n    kind: none\n' > norerank.yaml
: > aa.yaml
run-eval --data-path eval.json --search-strategy local,mix --repeats 2 \
  --arm norerank=norerank.yaml --arm aa=aa.yaml --config-path config.yaml
```

- Its runs are labelled `<strategy>@NAME` (`local@norerank`) in the files
  and the comparison, and each is compared with the same strategy's base
  run, so its deltas measure the overlay; the base strategies compare with
  the first as usual. The arms' deltas are Holm-adjusted in families of
  their own (`arm_primary`, `arm_secondary`, `arm_cost`, `arm_pairwise`),
  apart from the strategies' against the first: the two answer different
  questions.
- The overlay is validated like a config file (an unknown key stops the
  run, exit 2, unless `--allow-unknown-keys`), and the relative paths it
  sets (SQLite stores, seed graphs, a schema file) resolve against its own
  directory; the ones it leaves are the run's, so an arm reads the run's
  stores. Its `evaluation` section and
  `models.roles.evaluation` / `answer_span` are ignored with a warning,
  since every arm is scored as the base runs are: an unset LLM judge is
  chosen over the answer models of the base runs and of every arm (one
  that wrote none of them when there is one) and pinned before the arms
  are built, and the answer-span extractor is the run's.
- A comparison whose runs were scored with different LLM judges, judge
  efforts or ragas embedding models (an arm that changes
  `models.embedding`, `--baseline-results` or `--from-results` runs of
  other invocations) warns that its judged deltas measure the judges too.
- The run manifest records each arm run's `arm` (name, overlay path and
  sha256, suffix) and `compared_runs`, so `--from-results` rebuilds the
  same pairs. A comparison whose runs differ in whether their answers cite
  (`search.citations`, resolved for each run's strategy, which the manifest
  records as `citations.strategies`) warns that the answer prompt differs
  too: `global` and `drift` do not cite by default, so they warn against
  the other strategies unless the map sets them alike.
- An arm retrieves on the base runs' query extractions (entities and
  keywords, one call per question) when its overlay leaves them alone; one
  that changes the extraction's model, the default tier's effort,
  `search.output_ceilings.query_extraction`,
  `models.default_max_output_tokens`, the LLM provider (kind, `base_url`,
  temperature, output cap, context window), the Bedrock region or sets
  `search.query_extraction_cache_size: 0` extracts on its own.
- An arm whose configuration, once merged, equals the run's (cache sizes
  aside: `search.query_extraction_cache_size` and
  `search.token_manager.token_count_cache_size`) on the run's own index is
  an A/A arm: an empty overlay (`aa.yaml` above), one holding only the
  ignored `evaluation` section, or one setting the run's own values. Its
  deltas are the noise floor of each metric for this dataset and repeat
  count. `,aa=true` (or `aa=false`) declares it either way. The manifest
  records it (`arm.aa`), the comparison lists it under `arm_roles` (`aa`)
  and the report, the terminal and the console mark it "(noise floor)"; it is left out of every Holm
  family and of the `--fail-if` gates, as the closed-book arm is, so it
  does not widen the other tests' adjustment or fail the run. It shares
  the base runs' extractions, so that floor leaves out the extraction's
  noise; an A/A overlay of `search: {query_extraction_cache_size: 0}`
  extracts on its own and includes it, at one extra extraction call per
  question and repeat.
- Each arm costs what a strategy costs (its answers and warm-up), times
  the strategies listed.

## Assertions, length control and a closed-book bound

Open-ended (global) questions have no gold answer, and a pairwise judge
prefers the longer answer. Three opt-in tools measure around both; none
changes a default until a measured experiment decides
([Design: §9.3](../design/evaluation.md#93-length-controlled-judging-assertions-and-the-closed-book-arm)).

1. Generate questions and assertions from your own corpus, in the style of
   Microsoft BenchmarkQED (its code is MIT; none of its data is used):

   ```bash
   prepare-benchmark assertions --source-directory corpus/ \
     --output-directory bench/assertions --questions-per-class 10 \
     --config-path config.yaml
   ```

   It writes `eval_data.json` with questions in five classes (`category`:
   `data_local`, `data_global`, `data_linked`, `activity_local`,
   `activity_global`), assertions for the three data classes,
   `text_units.jsonl` (the text every assertion cites, with file and
   offsets: review the assertions against it before you trust the set) and
   `generation_report.json` (seed, models, drops, usage). It calls the
   evaluation model (`models.roles.evaluation`) three to four times per data
   question plus once per cluster and per activity question, and embeds at
   most `--max-units` text units; it prints the calls and tokens it spent
   (and a USD estimate with `evaluation.prices`). The questions and
   assertions are in `--language` (default
   `processing.translation.source_language`). The same seed and the same
   model replies give the same files.
2. Score the answers on the assertions and judge them without length bias:

   ```bash
   run-eval --data-path bench/assertions/eval_data.json \
     --search-strategy global,mix --pairwise-judge --pairwise-mode differential \
     --closed-book --config-path config.yaml   # enabled_evaluators: [assertion, ...]
   ```

   - `assertion_recall` is the share of an item's positive assertions the
     answer supports (an assertion counts when more than half of its trials
     say so), `assertion_contradiction_rate` the share it contradicts and
     `negative_assertion_violation_rate` the share of negative assertions it
     states. A judge failure leaves the metric failed, never 0. Gate on it
     like any metric: `--fail-if "assertion/assertion_recall:ci_high<-0.02"`.
   - `--pairwise-mode differential` asks the fast tier
     (`models.roles.pairwise_overlap`) for the statements both answers make
     and those only one makes, then judges the unique ones only. The split
     reads the answers in a slot order and a split model favours one slot
     (an A/A pair won 0.645 with the candidate always second, 0.52
     swapped), so each pair is split both ways, each split judged in both
     orders, and the win rate averages the two; `slots` reports the win rate
     per slot and their difference (`slot_effect`, the "Split slot effect"
     table), which should sit near 0. Two fast calls and four judge calls
     per pair. That call's output limit is
     `models.default_max_output_tokens` above the thinking budget of the
     effort it runs at (`evaluation.judge_effort`, else the fast tier's), so
     a higher effort leaves the copied statements as much room.
   - `--closed-book` adds the `closed_book` arm: the answer model answers
     every question with no retrieval. A strategy that does not beat it
     measures what the model already knows. The comparison marks it "(no
     retrieval)" and gates skip it; its retrieval and context metrics score
     an empty source list.

## Rebuilding, re-judging and gating a comparison

`run-eval --from-results <dir|files>` rebuilds the comparison from the files
earlier runs saved (`evaluation_results_*` with the `evaluation_reports_*`
and `evaluation_summary_*` next to them; `evaluation.save_detailed_results:
true`, the default; a directory is read with its run subdirectories) without
answering a query or opening a store. The runs
are grouped by strategy and compared in the order the oldest run listed them
(`--search-strategy` picks and orders them instead), so a comparison that was
stopped part-way is finished by running only the missing strategies into the
same directory, with the same `--repeats`, and rebuilding: its deltas are
those of an uninterrupted run. That needs `--run-order sequential`, which
saves each run as it finishes: an interleaved comparison finishes all its
runs together, so one stopped part-way saved none. A strategy whose runs were all lost runs again
with `--repeats N`; one with some repeats saved runs only the missing ones
(`--repeats` of the difference, rebuilt with `--pool-runs`, below), or
delete its partial files and run all N.

```bash
# Stopped during naive (run with --run-order sequential --repeats 3): run it
# alone into the same directory with the same --repeats, then rebuild from
# both run subdirectories.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# Re-judge the saved answers with another judge model (no answer call).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# Re-score the saved answers with this version's evaluators (and a
# regenerated dataset's golds), saved as new runs in --output-directory.
run-eval --from-results out/ --rescore new_eval.json --output-directory rescored/

# CI: compare this run with an earlier one and fail on a supported regression.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- **Pooling.** The saved runs of one strategy are its repeats only when
  one `run-eval` answered them with the same settings. Runs of one strategy
  from several invocations (several `run_id`s; for runs saved before it was
  recorded, more runs than their `--repeats`) or with other settings (query
  flags, configuration hash, code, index or answer model) are refused
  (exit 2): point at one run subdirectory or list the result files, or give
  `--pool-runs` to pool them anyway (a warning names what they mix).
- **Re-judging.** `--pairwise-judge` with `--from-results` judges the saved
  answers again; `--judge-model` sets the judge model for the run.
- **Re-scoring.** `--rescore` scores the saved answers again with this
  run's evaluators (`--evaluators`, `evaluation.answer_span`, the citation
  metrics, the judges, which are called again) instead of comparing the
  scores they hold, against the golds saved with each answer, or with
  `--rescore DATASET` that dataset's (by query id, the same questions).
  The re-scored runs are saved in this run's directory; their manifest
  keeps what answered them and records the re-score (`rescored`).
- **A baseline run.** `--baseline-results` compares each strategy with the
  same strategy of an earlier run (`baseline:<strategy>`). Run it with the
  same `--repeats`; each run is saved in a run subdirectory of its own, and
  a baseline directory holding several invocations of one strategy is
  refused like `--from-results` (`--pool-runs`).
- **Scoring must match.** Runs whose exact match and token F1 read the
  answer differently (`evaluation.answer_span`, `answer_normalization`) are
  refused (exit 2) before any answer is generated; runs of different
  datasets are warned about and pair on their shared query ids.
- **Gates.** `--fail-if <measure>:<stat><op><value>` (repeatable) exits 1
  when a compared strategy's paired delta meets the condition;
  `ci_high<-0.02` fails only on a regression the query sample supports. A
  strategy with no delta of the measure fails too, and so does a delta over
  fewer than 10 paired queries (one pair gives a zero-width interval; under
  about 30 the bootstrap interval is too narrow, excluding 0 on noise in up
  to 8-12% of runs instead of 5%), or a
  metric the judge failed to score on different numbers of the two sides'
  answers (`strategies.<s>.failed_metrics`; those queries are unpaired). It
  needs two strategies or `--baseline-results`.
- **How many gates.** Each gate on each compared strategy is its own test:
  a `ci_high<0` gate breaches on noise alone in about 2.5% of runs, so with
  k compared strategies and m such gates a clean change fails up to
  k × m × 2.5% of the time (3 strategies × 4 gates: up to 30%). Gate a few
  measures chosen in advance, with a margin (`ci_high<-0.02`, not `<0`), or
  gate on the adjusted statistics, which account for the strategies and
  metrics compared: `--fail-if "answer_match/token_f1:adjusted_ci_high<-0.02"`
  (also `adjusted_ci_low`, `p_value`, `adjusted_p_value`).

The full rules (what each warning compares, when a cost gate fails):
[Evaluation output: Rebuilding and gating rules](../reference/evaluation-output.md#rebuilding-and-gating-rules).
