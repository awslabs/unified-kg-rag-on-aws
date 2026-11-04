# Evaluation (`run-eval`)

> 🇰🇷 한국어판: [evaluate.ko.md](./evaluate.ko.md)

`run-eval --data-path <dataset.json>` answers every question of a dataset and
scores the answers and their sources. Three evaluators need no LLM and cost
nothing; two use an LLM judge. Several strategies (or `--repeats`) turn the run
into a paired comparison.

Every flag with its default: [CLI Reference](../reference/cli.md#run-eval).
What the result files hold, field by field, and how each evaluator scores:
[Evaluation output](../reference/evaluation-output.md). A public benchmark
as a corpus and a dataset: [Benchmarks](./benchmarks.md).

## Evaluators

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

The two judges need the `eval` extra (`uv sync --extra eval`, [Install](../getting-started/aws.md#install));
without it `run-eval` stops before the first query with exit `2`, naming the
extra, even with `processing.ignore_errors: true`. The judge-free run above
needs no extra. Any other enabled evaluator that cannot be built (no Bedrock
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
  in `abstention_statistics`
  ([Run summary](../reference/evaluation-output.md#run-summary)).
- The file is validated before any query runs. A file that cannot be read,
  an empty or non-array dataset, a missing `question`, a duplicate id, a field
  of the wrong type (a non-list `expected_entities`, an `answer_aliases` that
  is not a string or a list of strings) or `metadata` the RAG chain rejects
  (an unknown `search_strategy`, or `filters` it refuses: a malformed value
  such as an empty list, or a key the item's strategy refuses, checked under
  every compared strategy in a comparison) stops the run with
  `Error: invalid evaluation dataset:`, the item index and id, and exit code 2.

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

Each run writes `evaluation_{results,reports,summary}_<timestamp>.json`
(find the latest by sorting the names). The summary holds the per-metric
statistics, scored/failed/skipped counts, abstentions, rerank and guardrail
counts and a run manifest that records what was measured, so two runs can be
compared: [Evaluation output](../reference/evaluation-output.md). The
console prints the same summary; the failure budget
(`--max-failure-rate`) applies to every run.

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
  help a strategy.
- **When a difference counts.** A difference is supported when its interval
  excludes 0; a 10-point difference on binary metrics is detected in 15% of
  runs with 50 queries and 59% with 200, so compare on a few hundred
  queries. `deltas_by_category` splits the deltas by dataset `category`;
  its intervals are not corrected for the many categories and metrics read,
  so treat a lone excluded 0 among many as exploratory.
- **Pairwise judge.** `--pairwise-judge` adds the GraphRAG / LightRAG
  head-to-head LLM judge (comprehensiveness, diversity, empowerment,
  directness), each pair judged in both orders; read `candidate_win_rate`
  next to `position_consistency`.
- **Cost and latency.** `cost.strategies.<s>` and `cost.deltas_vs_baseline`
  give the answers' calls, tokens and latency per query, paired like the
  metrics; a failed answer counts with what it spent before failing.

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
little higher (measured:
[Usage records against the bill](../concepts/defaults-and-evidence.md#usage-records-against-the-bill)).

## Rebuilding, re-judging and gating a comparison

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

- **Pooling.** The saved runs of one strategy are pooled as its repeats;
  the rebuild warns when they ran with other settings (query flags,
  configuration hash, code, index or answer model), so keep such runs in
  another directory.
- **Re-judging.** `--pairwise-judge` with `--from-results` judges the saved
  answers again; `--judge-model` sets the judge model for the run.
- **A baseline run.** `--baseline-results` compares each strategy with the
  same strategy of an earlier run (`baseline:<strategy>`). Run it with the
  same `--repeats`, and save each run in a directory of its own.
- **Scoring must match.** Runs whose exact match and token F1 read the
  answer differently (`evaluation.answer_span`, `answer_normalization`) are
  refused (exit 2) before any answer is generated; runs of different
  datasets are warned about and pair on their shared query ids.
- **Gates.** `--fail-if <measure>:<stat><op><value>` (repeatable) exits 1
  when a compared strategy's paired delta meets the condition;
  `ci_high<-0.02` fails only on a regression the query sample supports. A
  strategy with no delta of the measure fails too. It needs two strategies
  or `--baseline-results`.

The full rules (what each warning compares, when a cost gate fails):
[Evaluation output: Rebuilding and gating rules](../reference/evaluation-output.md#rebuilding-and-gating-rules).
