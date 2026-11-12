# Design: evaluation framework

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

## 9. Evaluation Framework

`EvaluationManager` (`evaluation/`) builds the evaluators through
`_resolve_evaluator_class` (one lazy-import branch per `EvaluatorType`) and
scores every result after all answers are generated. What each evaluator
measures is in [Evaluation: Evaluators](../guides/evaluate.md#evaluators) (how it
scores: [Evaluation output](../reference/evaluation-output.md#evaluator-details)); the design
choices:

- **LLM-free evaluators are first-class.** `answer_match`, `retrieval` and
  `graph_aware` are deterministic and free, so they are the default. The RAGAS
  judge is opt-in (several judge calls per answer, and the `eval` extra).
  The LangChain judge was removed: its correctness metrics repeated RAGAS'
  `answer_correctness`.
- **`graph_aware` emits recall only.** Precision and F1 would need every
  entity of a free-text answer enumerated, so they would only re-label recall.
  Because answer-text coverage grows with length, it also reports the
  coverage of the retrieved sources and the answer length.
- **`retrieval` counts files, not sources.** A community report is attributed
  to every file of its community, so counting it as one source let random
  reports score near-perfect hit@k.
- **Strategy comparison** (`evaluation/comparison.py`, `pairwise.py`). Several
  strategies or `--repeats` run the same queries; each metric is averaged per
  query over repeats and compared per query with a seeded paired bootstrap
  interval (pooled means would mix query difficulty and noise into the
  difference). The optional pairwise judge (comprehensiveness, diversity,
  empowerment, directness, as in the GraphRAG and LightRAG papers) judges both
  answer orders, keeps only order-stable verdicts and reports how often the
  orders agreed (`position_consistency`), so a position-biased judge's ties
  are not read as equal answers. A failed answer is never dropped (a strategy
  failing on the queries it would lose would look as good as one answering
  them): it scores 0 on every metric another run scored the query on and its
  own strategy scores somewhere (a metric the strategy never produces is no
  loss to invent), and a pair with one failed answer goes to the other side
  unjudged.
- **A comparison guards against time and multiplicity.** Its runs are
  answered interleaved (`EvaluationManager.answer_interleaved`): one batch,
  each query by every run before the next, the runs' order reversed on
  every other query (ABBA) and again on every other repeat, each run with its own usage meter
  (`shared.usage.metering`) and span attributes; answered one run after
  another, later runs answer later, and runs of one configuration
  showed 0.5 to 2.3 s latency "differences" a query on average
  ([Defaults and evidence](../concepts/defaults-and-evidence.md#how-the-measurements-were-run)). Each repeat extracts the query
  again in its own extraction cache scope (`extraction_cache_scope`), so
  `repeat_stdev` includes that variance while the arms of a repeat still
  share keywords: the extraction cache tracks the extractions in flight per
  scope, so arms that ask together make one call. Configuration arms
  (`run-eval --arm NAME=overlay.yaml`, `application/cli/arm_args.py`,
  `evaluation/arms.py`) are runs of their own chain and manager under the
  run's config with an overlay merged over it (its `evaluation` section and
  scoring models kept, the judge chosen over every arm's answer model),
  labelled `<strategy>@NAME` and paired with the same strategy's base run;
  an arm shares the base chain's extractions only when its extraction
  settings match (`run_evaluation.extraction_settings`). Every delta gets a sign-flip randomization p-value (exact
  on few pairs, where the percentile bootstrap is anti-conservative), and
  `adjust_for_multiplicity` Holm-adjusts them in families: the pre-declared
  primary metric (`evaluation.primary_metric`, default `answer_contains`)
  over the compared strategies only, every other metric, the cost
  measures and the pairwise judge's criteria (each against a 0.5 win rate),
  with Bonferroni-level intervals; the `--arm` runs against their own base
  runs form the same four families apart (`arm_*`), and the closed-book and
  A/A arms (on the run's own index, a merged configuration equal to the
  run's but for cache sizes, or `,aa=true`; `arm.aa` in the manifest) none, since their deltas bound prior knowledge or measure the
  noise floor. Each family reads at least
  20 x tests / alpha resamples so the Monte Carlo floor cannot hold it above
  alpha; ten unadjusted metrics would
  show a "supported" difference on noise in 30-40% of A/A runs. Gates
  can read the adjusted statistics. The pairwise judge reads the first
  repeat by default (`evaluation.pairwise_repeats: all` judges every
  repeat, clustered by query: the repeats of a query are averaged first and
  its interval resamples queries).
- **Cost is measured, not estimated.** `shared/usage.py` keeps contextvar
  scopes (one per ingestion stage, per query, per `run-eval` run and answer
  attempt) that a
  callback on every factory-built chat model fills from the tokens each
  response reports, tracing on or off (one context-variable read outside a
  scope); cache-served responses count as hits, not calls. The comparison's
  `cost` block pairs each query's calls, tokens and latency like a metric,
  and `--fail-if` gates on either. A failed answer keeps what its attempts
  spent (its own scope and timer), so failing cannot make a strategy look
  cheaper or faster by dropping out of the cost; an answer without a measure
  fails a cost gate rather than leaving the delta to the others. Prices are user-supplied
  (`evaluation.prices`): a shipped table would be stale. Input tokens
  include the prompt-cache reads and writes, also counted apart
  (`cache_read_input_tokens`, `cache_write_input_tokens`) and priced at
  their own rates when a price gives them; the reranker's requests count
  as `rerank_calls` and `rerank_search_units` (one per 100 documents) and
  are priced per thousand units (`per_thousand_rerank_search_units`). A call counts once however many
  requests botocore sent for it; those resends (`request_retries`, read from
  `ResponseMetadata.RetryAttempts`) and calls that failed (`failed_calls`)
  are counted apart, since Bedrock counts each accepted request as an
  invocation.
- **The judge is not the answer model.** `models.roles.evaluation` defaults
  to Opus 5.5 on Bedrock, a model other than the default answer model, since
  a judge favours its own answers (§9.1); when the answer model is Opus 5.5
  the unset judge falls back to the default tier's model, else Sonnet 5.5
  (`Config.judge_model_choice`, logged at INFO); a re-judge
  (`--from-results`, `--baseline-results`) also avoids the saved runs'
  answer models. `judge_is_answer_model` in
  the run manifest and a warning still flag a configured judge that is the
  answer model.
- **A comparison is a pure function of saved files.** `paired_comparison`
  reads only the runs' results and reports, so `run-eval --from-results`
  (`evaluation/saved_results.py`) rebuilds it without a model call, an
  interrupted comparison finishes by running only the missing strategies,
  and runs scored with different `answer_scoring` are refused rather than
  compared. Against `--baseline-results`, each strategy pairs with the
  earlier run of the same strategy (`baseline_comparison`: one paired
  comparison per pair, merged): pairing every strategy with one baseline
  would compare strategies, not runs, and hide a regression of all but the
  first.
- **EM/F1 read a span, not the whole answer.** The reading-comprehension
  metrics were defined over a short predicted span; over a sentence plus its
  support they reward brevity. `evaluation.answer_span` picks the scored
  text: the first answer line (default, free), a fast-model extraction
  (cached, falling back to the first line) or the whole answer. The choice is
  recorded in the manifest and each report, and a comparison warns when its
  runs differ.
- **A number item is read by one small rule.** `answer_type: number` asks
  whether the answer's first line (or extracted span) states the gold: a
  gold other than 0 when the line writes it anywhere, a gold of 0 when it
  states there is none and writes no number the question does not
  (`text_matching.states_number`). Earlier versions picked the count among
  the line's numbers (the first one, then rules for conditions,
  comparatives, clauses, years, partitives and the question's numbers);
  each rule moved other answers, and the count kept regressing across
  reviews in other languages. The rule now reads only what is
  language-independent apart from number words and words for none, and
  over-counts by design (a gold written as a condition matches); telling a
  count from a condition is the LLM judges' job in the same run. The exact
  rule and its limits are in
  [evaluation output](../reference/evaluation-output.md#answer_match).
- **Run manifest.** Ingestion records an `IndexBuildRecord` (pipeline id,
  mode, build time, documents, extraction fingerprint and settings, package
  version, community counters) in the vector index mappings' `_meta` (`VectorIndexer.record_build`),
  read back per queried suffix. The config hash covers only the settings that
  shape answers (§12), so it is equal across environments.
- **Benchmarks.** `prepare-benchmark` (`evaluation/benchmarks/musique.py`)
  builds a MuSiQue corpus and dataset at run time;
  `evaluation.answer_normalization: official_squad` with
  `evaluation.answer_span: extracted` brings EM/F1 close to published
  numbers (a RAG answer still is not a trained reader's span).

### 9.1 Measured defaults

The defaults set or kept from real-model measurements, with the method, the
numbers and dated prices, are in [Defaults and evidence](../concepts/defaults-and-evidence.md).

### 9.2 Unanswerable questions and abstention (opt-in)

Whether the chain declines a question its corpus cannot answer is measured
with a benchmark split built for it and an explicit decline marker. Both are
opt-in: with no `--unanswerable` and `search.abstention.mode: off` (the
defaults) the corpus and the dataset hold no unanswerable question and the
answer prompts ask for no decline marker.

**Benchmark** (`evaluation/benchmarks/musique.py`, driven by
`prepare-benchmark musique --unanswerable N [--hops ...]`). MuSiQue-Full
pairs every answerable question with an unanswerable twin of the same id:
the same question with one or more supporting paragraphs replaced by
distractors. Two leaks make a naive pool answerable:

1. *The twin.* An unanswerable item pooled with its own answerable twin
   gets the removed paragraph back from the twin's paragraphs. Any
   unanswerable item whose twin is in the answerable sample is excluded.
2. *Other questions.* MuSiQue reuses Wikipedia paragraphs across questions:
   in a sample of 200 unanswerable plus 200 non-twin answerable items,
   28-42% of the unanswerable ones still find every removed paragraph in
   the pool (measured offline on the dev split). Selection is greedy over a
   seeded shuffle of the candidates: an item is kept only when no removed
   paragraph's exact text (whitespace-normalized) is in the pool so far, and
   none of its own paragraphs is a removed paragraph of an item already kept;
   a kept item's paragraphs join the pool. Either rule excludes it as
   `contaminated`. The pool starts from the answerable sample.

The answerable sample is drawn from the answerable file (MuSiQue-Ans: the
default download, or `--data-path`) and the unanswerable items from the full
file (downloaded beside it, or `--unanswerable-data-path`), whose answerable
half supplies each item's twin (which paragraphs were removed). So the
answerable set is the one the same seed draws without `--unanswerable`. The
two files are not interchangeable: every item's paragraph set differs
between them, and a seed draws a different sample from the full file's
answerable half (17 of 200 shared on the dev split at seed 0), so drawing
both from the full file, as a full file given alone as `--data-path` still
does (labelled `musique-full-answerable`), changes the answerable scores.

Unanswerable rows carry `metadata.answerable: false`, no answer or
reference sources, `metadata.benchmark: musique-full-unanswerable` and the
twin's answer (`metadata.twin_answer`, for auditing an answer that names it).
The counts (`unanswerable_filter`: candidates, kept, excluded twins,
contaminated, no twin) are printed and in the JSON output. `--hops` filters
both pools by hop count (`3,4`) or draws an exact answerable quota per hop
(`2=50,3=50,4=100`); sampling stays deterministic per seed. The exact-text
filter does not catch near-duplicate paragraphs or facts a model knows
without the corpus; `twin_answer` exposes the latter. Cost is ingestion's:
about 19 paragraphs a question.

**Decline marker** (`search.abstention`, applied by `GraphRAGChain`). The
chain already answered an empty context with a fixed reply
(`metadata.abstained`; `domain/retrieval/no_context.py`, one sentence per
`LanguageCode`, picked by the question's script and, among the Latin-script
languages, by their function words (one two of them share ties them, and
the index language among them decides); the index language, else
English, when they do not tell), but a model that declines in its own words was
invisible to evaluation. With `mode: mark` the answer step appends
`ABSTENTION_INSTRUCTIONS` (`domain/prompts/retrieval.py`) to the system
prompt of whichever answer prompt runs (concise, comprehensive or a
`custom_prompts` override): when the context does not support the answer,
start with the fixed token `[NO_ANSWER]`, then one sentence in the query's
language naming what is missing. Detection
(`domain/retrieval/abstention.py`) is a pure function over the first
characters after NFKC normalization, so full-width brackets and leading
Markdown emphasis are accepted and the answer's language does not matter
(the token stays ASCII). It sets `metadata.abstained` and
`metadata.abstention_reason` (`no_context`, `marker`, `insufficient_context`)
and strips the token unless `strip_marker: false`; a streamed answer holds
back at most the first 32 characters until the token is ruled in or out.
An abstained answer keeps its sources: the model saw them, and evaluators
audit them (only a guardrail block or an empty context leaves none). The
marker reaches the output guardrail stripped, like any answer.

**Insufficiency hook** (`search.abstention.on_insufficient`). A strategy
that judges its own context (an iterative or sufficiency-checking strategy)
sets `SearchResult.metadata["context_sufficient"] = False` (constant
`CONTEXT_SUFFICIENT_KEY`) and optionally `insufficiency_reason`. Only then
does the answer step apply the policy: `hedge` (default) appends
`HEDGE_INSTRUCTIONS` (answer what the context supports and say what it
does not), `abstain` appends `DECLINE_INSTRUCTIONS` (do not answer; give the
token and the missing part in the query's language) and marks the output
abstained with reason `insufficient_context` whatever the model writes.
Both are LLM-written so the wording follows the query language; neither
costs an extra call. Strategies that never set the flag are unaffected;
`iterative` (§6.3) is the shipped strategy that sets it.

**Scoring** (`EvaluationManager._calculate_abstention_statistics`). An
abstention on an answerable item is a wrong answer: `answer_match` scores it
0 on every metric (no span extraction call) and records `abstained` in the
report. Unanswerable items are scored only by abstention:
`abstention_statistics` adds `answerable` (same shape as `unanswerable`),
`precision` (abstentions that were on unanswerable items), `recall`
(unanswerable items abstained on) and `f1`, and per `answer_match` metric
`coverage` (answerable items answered), `accuracy` (over all answerable
items, abstentions as 0), `selective_accuracy` (over the answered ones) and,
with unanswerable items, `overall_accuracy` (a correct answer or a correct
decline, over both). Coverage and accuracy are reported together so a
strategy cannot look precise by declining more. A failed answer is no
decline: it counts against recall and scores 0 in accuracy (with a
reference answer), so a strategy cannot look better by failing either.

A paired comparison carries the declines as measures: `abstention/recall`
is a per-query 0/1 (declined) on the unanswerable queries, averaged over
repeats and paired like any metric (bootstrap interval, per category, per
query); `abstention/precision` is a ratio over the declines, so its delta is
the difference of the two precisions and its interval comes from
recomputing both on each resample of the queries (`pairs` = queries either
side declined; absent when one side declined none). `--fail-if` reads both.

Failure modes: a model that ignores the marker instruction looks like
`mode: off` (recall falls, nothing breaks); one that writes the token
before a real answer is scored as a decline (precision on answerable items
shows it). The marker instruction changes the answer prompt, so `mark`
runs are a separate arm, not comparable EM-for-EM with `off` runs; the
manifest's config hash differs. A measured run (100 unanswerable + 100
answerable MuSiQue questions, `local` and `mix`, `off` vs `mark`, and
`iterative` `hedge` vs `abstain`) kept `mark` opt-in and `hedge` the
default ([Defaults and evidence](../concepts/defaults-and-evidence.md#declines-and-abstention)).

### 9.3 Length-controlled judging, assertions and the closed-book arm

The pairwise judge's evidence for the global strategies was weak in two ways:
an LLM judge prefers the longer answer (Zeng et al., arXiv 2506.06331,
measured LightRAG's win over naive RAG falling from 66.7% to 39.1% once the
length effect is removed), and the global question set came from our own
community reports, so it favours the method that wrote them. Four opt-in
additions, all in the `evaluation` layer and its prompts, measure around
both. None is on by default: a run without their flags and evaluator
judges and scores with the defaults, and each stays opt-in until a
measurement shows it should be a default.

**Length buckets (always on with `--pairwise-judge`).** `PairwiseJudge`
records each judged pair's answer lengths in estimated tokens
(`adapters.llm.token_estimate.estimate_token_count`: about one token per
Hangul, kana or Han character, about four characters per token in Latin
text, each other script at its own rate, so a Korean or Japanese answer is
not measured in whitespace words) and
reports, under `length`, the candidate's win rate per bucket of the length
ratio (candidate over baseline: at most 0.67, 0.67-0.91, 0.91-1.1, 1.1-1.5,
at least 1.5; symmetric in the log ratio, so 1.5 and 1/1.5 sit in mirror
buckets) and, per criterion, the correlation between the relative length
difference and the pair's score. A win rate that rises with the ratio, with
few pairs in the 0.91-1.1 bucket, is a length effect, not a quality one. Free:
no model call.

**Differential mode (`evaluation.pairwise_mode: differential`, or
`run-eval --pairwise-mode differential`).** Per judged pair:

1. Two `pairwise_overlap` calls (fast tier, `PairwiseOverlapPrompt`), one
   per slot order (the baseline as Answer 1, then the candidate), split the
   two answers into the statements both make and the statements only one
   makes, copied from the answers in their own language (never translated);
   a shared statement is kept in the shorter of its two wordings.
2. For each split, the judge (`models.roles.evaluation`,
   `PairwiseDifferentialPrompt`) sees the shared statements as common
   ground that earns neither answer credit, and decides each criterion on
   the unique statements only: both orders, ties, the same criteria (with
   `overall` under `evaluation.pairwise_overall`) and the same
   order-agreement rule and `position_consistency` as the standard mode.
3. A split in which neither answer has a unique statement is a tie for
   every criterion without a judge call (`identical_pairs` counts the
   pairs with none in either split).
4. The pair's score per criterion is the mean of its two slots' scores;
   `slots` reports the candidate's win rate per slot and their difference
   (`slot_effect`).

Why both slots: one split per pair, the baseline always Answer 1, favoured
the candidate. On an A/A pair (x5, 2026-11-08: `simple` against an empty
overlay of itself, sensemaking-50, two repeats) the candidate won
comprehensiveness 0.645 (p 0.005) in that slot, 0.52 swapped, and 0.562
[0.49, 0.635] averaged over both (a slot effect of +0.165); the slot moved
the measured citations on/off effect by 0.125-0.215. The judge's own two
orders do not remove it: the bias is in what the split lists as unique, and
the prompt asked for the shared statements "worded as in Answer 1".
`tests/unit/test_eval_judging.py` pins it with an A/A pair and a split
model biased to one slot.

A failed or malformed overlap call fails the pair (`failed`, held to
`--max-failure-rate`), never a tie. Cost: two fast calls plus four judge
calls per pair, on shorter inputs than the standard mode's. The per-pair
shared and unique token counts go to `length.per_query`, so a reader sees
how much of each answer was common ground. The overlap model can miss a
paraphrase; the shared list is in the saved comparison for review.

**`assertion` evaluator.** A dataset item may carry
`assertions: [{text, kind: positive|negative, source?}]`: statements a good
answer supports (positive) or must not state (negative, for example a known
hallucination). Per answer and trial, one `assertion_judgement` call (the
evaluation role) labels every assertion of the item `supported`,
`contradicted` or `absent`, in a shuffled order seeded by the query id and
trial, so the assertion order is no position bias (the trials of a query
take distinct orders while any is left). An assertion takes the label more
than half of the query's successful trials gave it (a majority vote,
BenchmarkQED's pass rule). Per query: `assertion_recall` (positive
assertions labelled supported), `assertion_contradiction_rate` (positive
ones contradicted) and `negative_assertion_violation_rate` (negative ones
supported). A deterministic judge (temperature 0) labels one order the same
way every time, so an item with fewer orders than trials (one assertion, or
two with three trials) repeats a call that adds nothing; the report records
`distinct_orders`. The judge calls of all queries and trials share
`processing.max_concurrency`. A trial whose call fails, or whose reply misses an assertion or
uses another label, is a failed trial; a query with no successful trial
records its metrics under `failed_metrics`, never as 0. A query without
assertions, or without one of the two kinds, skips the metrics it cannot
score. The scores are ordinary report metrics, so the paired comparison,
`deltas_by_category` and `--fail-if assertion/assertion_recall:...` work
unchanged; the two rates are lower-is-better, so a failed answer is left
unpaired on them rather than filled with 0 (`comparison._UNFILLED`), which
would count a crash as a perfect score. Cost: `evaluation.assertion_trials`
(default 3) judge calls per answer. The labels are fixed English tokens in a
JSON reply, so parsing does not depend on the answer's language.

**Question and assertion generation (`prepare-benchmark assertions`).** A
native reimplementation of the approach of Microsoft's BenchmarkQED
(AutoQ question classes, AutoE-style assertions; code MIT, Copyright (c)
Microsoft Corporation; https://github.com/microsoft/benchmark-qed). Its
prompts are rewritten here and none of its datasets is used or shipped (they
are under the Microsoft Research License). `evaluation/benchmarks/autoq.py`:

1. Reads a corpus directory (`.txt`, `.md`), splits each file into text
   units of about `--unit-tokens` estimated tokens (paragraphs packed; an
   overlong paragraph is cut by the script-aware estimate, so CJK text
   without spaces splits too) with ids `<file>#<n>` and character offsets,
   samples at most `--max-units` of them with the seed and embeds them with
   the configured embedding model.
2. Clusters the embeddings with seeded k-means (k-means++ start, numpy) into
   `--clusters` groups. Each cluster's sample, largest cluster first, is
   its member nearest the centroid and that unit's `--units-per-cluster - 1`
   nearest untaken units by cosine distance (BenchmarkQED's centroid
   representative sampling).
3. Generates questions in five classes, each with the evaluation role:
   - `data_local`: from a sample of one cluster's units.
   - `data_global`: from local questions of several clusters that share a
     topic label, asking what the corpus as a whole says.
   - `data_linked`: from two local questions of different clusters that
     name a common entity, needing both.
   - `activity_local` / `activity_global`: from personas and tasks generated
     from a cross-cluster sample (one call), about one cluster's entities or
     about the whole corpus.
4. For every data question, assertions by map-reduce over claims: the map
   step extracts claims from its source units, each citing unit ids; the
   reduce step merges them into at most `--max-assertions` assertions citing
   the units of their claims; a validation pass scores each assertion's
   grounding, relevance and verifiability (1-5) against the cited unit texts
   and drops those below 3, and an assertion citing no known unit is dropped
   before validation. Activity questions get no assertions (BenchmarkQED's
   assertion scoring does not cover them); they are for the pairwise judge.
5. Writes `eval_data.json` (the `run-eval` format: `category` is the class,
   `reference_sources` the cited files, `assertions` with
   `source: [unit ids]`), `text_units.jsonl` (each unit's id, file, offsets
   and text, so a reviewer can check every assertion against its sources)
   and `generation_report.json` (seed, models, counts per class, dropped
   items and why, usage per step).

Every prompt names the output language (`--language`, default
`processing.translation.source_language`). Sampling, clustering, the order of
the calls and the selection among the generated candidates depend only on
the seed and the corpus, so a run with the same models' replies writes the
same files (tested with scripted models); the model replies themselves are
not deterministic. A failed or malformed call drops the item it was for and
is counted in the report. Cost is printed from the usage records (with a USD
estimate under `evaluation.prices`); a typical run (20 clusters, 10 questions
per class, 3 to 4 calls per data question) is a few hundred thousand input
tokens.

**Parity with BenchmarkQED, and the deliberate deviations.**
`tests/unit/test_benchmarkqed_parity.py` (skipped unless `benchmark_qed` is
importable; it is not a dependency, install it in a scratch environment; the
weekly `fidelity` workflow runs it against `benchmark-qed` 0.4.0)
feeds the same synthetic sample and scripted replies to both
implementations. Identical decisions: claim parsing (non-empty statement,
integer score above 0, sources read as strings and kept only when given),
assertion parsing (score 1-10, 5 when missing; `claim_<n>` ids mapped to the
claims' units; an assertion citing no known claim dropped), the assertion
ranking (weighted reciprocal rank fusion, importance 0.3 and source count
0.7, k 60, competition ranks), the validation rule (grounding, relevance and
verifiability clamped to 1-5, each at least 3), the per-question pass rate
over trials (an assertion passes above a trial mean of 0.5) and the
clustered sampling on fixed clusters; on well-separated groups our k-means
and scikit-learn's find the same partition. Deviations, each on purpose:

- One judge call labels all of an answer's assertions per trial (theirs:
  one call per assertion and trial), with three labels (`supported`,
  `contradicted`, `absent`; theirs: a 0/1 score) and a shuffled order per
  trial: an answer of 10 assertions over 3 trials costs 3 calls, not 30.
  A negative assertion is written as the false claim itself and is violated
  when supported (theirs: "The answer does not mention X", passed when
  absent).
- Validation scores all of a question's assertions in one call (theirs: one
  per assertion), and an assertion the reply does not score, or scores with
  a value that is not a number, fails; BenchmarkQED defaults such a score to
  3, a pass.
- A claim citing only ids the prompt never gave is dropped; BenchmarkQED's
  extractor keeps it with its raw sources (its source mapping is skipped
  and the later filter only checks that the list is non-empty).
- Booleans are no integer score, and blank statements are dropped after
  stripping (Python's `True` is an `int`; BenchmarkQED accepts it).
- Data-local questions come from one call per cluster (theirs: a generate
  then expand pair), the kept questions are taken round-robin over the
  clusters (theirs: a k-means over question embeddings ranked by similarity
  ratio, coverage and assertion count), and the claims of a global or linked
  question are its source questions' claims (theirs: claims re-extracted for
  the global question). Fewer calls; the selection is less tuned.
- Text units are packed paragraphs measured by the script-aware estimate
  (theirs: 600-token tiktoken windows with 100-token overlap), so Korean,
  Japanese and Chinese corpora split without a BPE tokenizer.
- Prompts: written here, each naming `{language}`; ours ask for JSON with
  fixed keys and cite short numeric ids that the code maps back to unit ids.
  BenchmarkQED's claim prompt scores importance 0-100, ours 1-10 (only the
  sign matters to the parse).

An end-to-end comparison with the same model on the same small corpus
(BenchmarkQED's AutoQ and AutoE against these) needs a model budget and is
listed as a measured experiment, not run here.

**Closed-book arm (`run-eval --closed-book`).** One more arm, labelled
`closed_book`, answers every question with the answer model alone
(`ClosedBookAnswerPrompt`: no context, no retrieval, no store call) through
an `EvaluationManager` whose chain is `evaluation/closed_book.py`'s runnable.
It runs after the strategies with the same repeats and evaluators and is
compared with the first strategy like any other arm, so its scores bound what
the answer model knows without the corpus: a strategy that does not beat it
on a question set measures prior knowledge, not retrieval. Its results carry
`metadata.retrieval: none` and no sources; the comparison lists it under
`arm_roles` (`closed_book`) and every surface marks it "(no retrieval)";
retrieval and context metrics of it score an empty source list (read only
the answer metrics); `--fail-if` gates skip it. Cost: one answer call per
question and repeat. Its answers also split the questions into those the
model knows (a primary-metric mean of at least 0.5 over the arm's repeats, a
failed answer counting 0: right in at least half of them on a 0/1 metric)
and the rest, and `deltas_by_closed_book` repeats every other delta over
each (exploratory, unadjusted; the arm's own is left out, since it would
select on its scores): on MuSiQue the unknown subset carried
differences the pooled deltas hid (iterative - mix +18.9 there; mix's
lead over ppr only on known questions).

**Layer placement.** Everything above lives in `evaluation/` (judges,
evaluator, generator, closed-book runnable), `domain/prompts/evaluation.py`
and `domain/prompts/benchmark.py` (prompts, overridable through
`custom_prompts`), the domain models (`EvaluationAssertion`, the evaluator
and metric enum members, the `pairwise_overlap` model role) and the two CLIs'
flags. The `assertion` evaluator is one branch in
`EvaluationManager._resolve_evaluator_class`. No store, retriever or
ingestion code changes.

### 9.4 Structural benchmark (the `structured` strategy)

`prepare-benchmark structural` (`evaluation/benchmarks/structural.py`)
generates a neutral procurement world from a seed, nothing committed: a
JSONL seed graph with seed properties (`country`, `founded`, `employees`)
and every `SUPPLIES`/`MAKES` edge, a document per vendor and buyer stating
part of those facts plus a fact only text states (each vendor's chief
executive), the extraction schema and seed entry, and a `run-eval` dataset.
Templates: count by relationship, count with a seed-property filter (of
vendors and of buyers), the top supplier and the top three (left out when
the cut ties), the vendor between a named buyer and a named product (a
two-hop chain, kept when one vendor matches), the buyer two named vendors
share (a two-hop path, kept when unique), yes/no supply (balanced), the
largest and smallest employee count, questions about vendors no data
names (`absent_entity`: no, or 0), Korean and Japanese forms of two
templates, and text-only controls; `--size large` makes the seed-filtered
counts scan past `max_scan_rows`. The gold
answer of every graph template is computed by the reference executor
(§6.6) over the generated graph from the template's plan
(`metadata.gold_plan`), so the generator and the strategy share one
semantics; numbers carry digit and word aliases.

The measurement plan (needs the maintainer's budget): `run-eval
--search-strategy structured,local,mix --repeats 3` (paired), and `auto`
with and without `structured` in `search.auto_routable_strategies`;
`answer_match` (containment the headline, EM and F1 with
`answer_span: extracted`) per template (`category`); from
`structured_statistics` the valid-plan, repair, fallback-by-reason and
truncation rates; calls, tokens and latency from the usage. Expected: large
gains on the count, ranking and path templates, where `local` counts from a
top-k context, and parity or small losses on the controls under
`fallback`. MuSiQue non-regression: `auto` with the default routable set
against the set with `structured`, no more than 1 point absolute loss on
containment, and the share routed to `structured` and its fallback rate;
`structured` joins the default routable set only after both.

---
