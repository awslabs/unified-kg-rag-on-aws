# Defaults and evidence

Several defaults were set, or kept, from real-model measurements rather than
by analogy with the papers. This page is the one place that holds those
numbers: how they were measured, what each one showed, what it cost, and how
to run the same comparison on your own corpus. The pages that explain a
setting (the [guides](../index.md#page-map), [Operator Runbook](../operations.md),
[Design Doc](../design.md)) link here for the evidence.

**Intervals.** Every interval below is a paired bootstrap interval over
questions, of one of two kinds:

- **95%**: the interval of that one difference, not adjusted for the other
  differences the run read. Every interval of the runs before 2026-11-07
  is of this kind, and so is an unlabelled one.
- **adjusted**: the simultaneous interval over its family of tests, as
  `run-eval` reports it since 2026-11-07 (confidence 1 - 0.05 / m for a
  family of m tests, so wider;
  [evaluation output](../reference/evaluation-output.md)). The sections on
  the runs of 2026-11-07 and later label each interval.

**Significance.** A difference is significant when its Holm-adjusted p
value (*Holm p*) is below 0.05, over the family of tests the section
names. A run before 2026-11-07 that names no Holm correction predates the
adjustment: there a difference is significant when its 95% interval
excludes zero, which is weaker evidence
([Limits of the evidence](#limits-of-the-evidence)). "n.s."
means not significant. A difference that is not significant kept the
existing default.

## How the measurements were run

One real-model evaluation in November 2026 (package at commit `f04ee50d`)
measured most of the defaults below:

- **Models and stores.** Amazon Bedrock in us-west-2: Claude Sonnet 5.5 for
  the default tier (extraction, reports, answers), Claude Haiku 5.5 for the
  fast tier, Titan Text Embeddings V2 and Cohere Rerank 3.5. The graph and
  vector stores were the embedded SQLite stores
  ([local stores with Bedrock models](../guides/local-mode.md#local-stores-with-bedrock-models)),
  so latencies include no network hop to Neptune or
  OpenSearch.
- **Index.** `prepare-benchmark musique --limit 200 --seed 0`: 200 MuSiQue
  questions over 3,318 paragraphs (about 280,000 words), ingested once with
  the defaults and shared by every arm. Building it took 16,069 model calls
  (51.4M input and 26.9M output tokens), $371.64 at the list prices below,
  in about 2.5 hours.
- **Question sets.** The 200 MuSiQue questions for `local`, `mix` and
  `hybrid` (EM, token F1, `answer_contains`, retrieval MRR); a 50-question
  sensemaking set (5 personas x 5 tasks x 2, written by Opus 5.5 from the
  level-0 community report summaries) for `global` and `drift`; and a
  109-document slice with 30 questions (their supporting paragraphs plus one
  distractor each).
- **Repeats and judges.** Every arm answered twice and each query was
  averaged over the repeats before pairing. Pairwise verdicts
  (comprehensiveness, diversity, empowerment, directness) count only when
  both answer orders agree, judged by Opus 5.5, Sonnet 5.5 and GPT-5.5;
  the pairwise judge read each arm's first repeat only. The repeats and the
  arms of a comparison reused one query extraction per question
  (`search.query_extraction_cache_size`), so they differ in the answer call
  and what follows the extraction, and the spread between repeats
  understates the run-to-run noise.
- **Answer scoring.** EM and token F1 scored a short answer span the fast
  model extracted from each answer (`evaluation.answer_span: extracted`)
  with SQuAD's official normalization
  (`evaluation.answer_normalization: official_squad`), not the package
  defaults (`first_line`, `default`), so the numbers sit closer to the
  papers'. `answer_contains` always reads the whole answer.
- **Latency.** The arms of a comparison ran one after the other, all the
  repeats of one arm before the next, so a latency difference between arms
  also measures what changed on the host and the endpoints in between.
  Three runs of one configuration on one index differed by 0.5 to 2.3 s a
  query on average, every difference "significant" by the paired interval.
  So a latency difference between arms below about 2.5 s is not evidence,
  and a larger one is an estimate; only the PPR ranking's retrieval time
  ([Local PPR ranking](#local-ppr-ranking)) was paired one query at a time.
  The runs of 2026-11-07 answered every comparison interleaved in one
  `run-eval`; there an empty arm moved the mean latency by -0.10 and
  +0.14 s, but its 95% intervals reached +0.67 s, and separate invocations
  differed by about 1 s
  ([Noise floor and baselines](#noise-floor-and-baselines)).
- **Cost.** Each run's library-metered tokens times the list prices. The
  meter of that version counted prompt-cache reads and writes as input
  tokens, so every cost below is an upper bound when the cache hit.

## List prices used

On-demand list prices on 2026-11-04 (AWS Price List API, us-west-2, global
cross-Region inference profile, standard tier), per million tokens unless
noted. They were the prices at the time of measurement; check current
pricing before you reuse a dollar figure.

| Model | Input | Output | Cache write | Cache read |
|---|---|---|---|---|
| Claude Sonnet 5.5 | $2.00 | $10.00 | $2.50 | $0.10 |
| Claude Haiku 5.5 | $0.10 | $0.50 | $0.125 | $0.01 |
| Claude Opus 5.5 | $4.00 | $20.00 | $5.00 | $0.20 |
| GPT-5.5 (judge only; in-Region model card price, used as an upper bound) | $5.50 | $33.00 | — | — |
| Titan Text Embeddings V2 | $0.02 | — | — | — |
| Cohere Rerank 3.5 | $2.00 per 1,000 search units | — | — | — |

An in-Region inference profile lists Sonnet 5.5 about 10% higher. To price
your own runs, give `run-eval` your prices (`evaluation.prices`,
[Evaluation: Comparing strategies](../guides/evaluate.md#comparing-strategies)): the package
ships no price table, because one would go stale.

## Measured defaults at a glance

| Setting | Default | Measured |
|---|---|---|
| `search.answer_prompt_style` | `{global: comprehensive}` | Comprehensive global answers beat concise ones on comprehensiveness, diversity and empowerment at 0.95-0.99 with all three judges, +$0.016 per query, and at 0.89-0.95 with a judge that compares only what each answer adds; they lose directness (0.29 [0.19, 0.39] with that judge). DRIFT stays concise for its latency ([Answer style](#answer-style)) |
| `search.global_search.evidence_tokens` | 4000 | -41% cost per global query against no cap, no consistent quality difference, sources kept ([Global evidence cap](#global-evidence-cap)) |
| `models.roles.evaluation` | Opus 5.5 on Bedrock | Against a GPT-5.5 judge, a Sonnet 5.5 judge shifted directness by -0.27 [-0.38, -0.16] when it judged its own answers ([The evaluation judge](#the-evaluation-judge)) |
| `processing.gleaning.max_rounds` | 3 | Within one ingestion, round 1 adds +6.0 points of bridge relationships and rounds 2-3 +2.0; decomposition answers found as graph entities +1.3 points [+0.4, +2.3] over extraction alone, +0.4 [0.0, +1.0] from rounds 2-3. Gleaning is 47% of the ingestion cost. Answers with round 1 alone against 3 rounds were mixed: `local` `answer_contains` -3.0 [-6.5, 0.0], `mix` EM +5.0 [+1.0, +9.5] ([Gleaning rounds](#gleaning-rounds)) |
| `search.lightrag_search.kg_chunk_pick_method` | `weight` | `vector` n.s. on `mix` and `hybrid`; its +1.5 s latency is within the run-to-run variation ([LightRAG chunk picking and MMR diversity](#lightrag-chunk-picking-and-mmr-diversity)) |
| `search.fusion.diversity_lambda` | 0.5 | 0.7 and 1.0 n.s. on `mix` ([LightRAG chunk picking and MMR diversity](#lightrag-chunk-picking-and-mmr-diversity)) |
| `search.abstention.mode` | `off` | `mark` against `off` on 100 answerable and 100 unanswerable multi-hop questions: EM -12 points [-20, -5] on `local`, -10 [-18, -3] on `mix`, for decline recall +8 [+3, +13] over the declines the model already writes in prose. Machine-readable declines, so opt-in ([Declines and abstention](#declines-and-abstention)) |
| `search.abstention.on_insufficient` | `hedge` | `abstain` with `iterative`: decline precision -9.0 points, token F1 -13.8 [-21.7, -6.0]; the controller flagged 45% of the answerable questions insufficient ([Declines and abstention](#declines-and-abstention)) |
| `search.local_search.report_selection` | `similarity` | `membership` (MS GraphRAG local's membership, over leaf communities rather than upstream's `community_level` cut) scored EM -4.0 and token F1 -4.1 points, both significant ([Local report selection](#local-report-selection)) |
| `search.local_search.text_unit_ranking` | `ppr` | Against `support` on 200 MuSiQue questions, two repeats: recall@5 +6.1 [+2.9, +9.4] and supporting paragraphs in the context +12 to +13 points; answers no worse (`answer_contains` +3.5 [+0.5, +6.75], borderline after a multiplicity correction; EM and token F1 n.s.); +0.10 s retrieval a query ([Local PPR ranking](#local-ppr-ranking)) |
| `search.local_search.entity_frequency_threshold` | 20 | Without the cut, recall@5 -1.75 points [-3.12, -0.50] and supporting paragraphs in the context -2.42 [-3.96, -1.04], both significant; +3.08 s per answer, run separately (see *Latency* in [How the measurements were run](#how-the-measurements-were-run)) ([Retrieval switches](#retrieval-switches)) |
| `search.auto_routable_strategies` | `local`, `mix`, `global`, `iterative` | `iterative` over `mix` against `mix` on multi-hop questions: `answer_contains` +12.5 points [+7.5, +18.0], token F1 +5.2 [+0.2, +10.1], last hop in the context +21.5; p95 11.1 s to 27.4 s. It cost +$0.0099 a query over `mix` (+18%) in the baselines of 2026-11-07, not the +$0.002 of that first run. `decompose` was no better than the loop, so it stays off ([Iterative follow-up rounds](#iterative-follow-up-rounds)) |
| `indexing.methodologies` | `[graphrag, lightrag]` (no `raptor`); `search.auto_routable_strategies` refuses `tree` | `tree` against `naive` on 100 MuSiQue questions, one repeat, on a namespace that also held the graph (so `tree` fused its graph signal too): `answer_contains` +8.0 points [+3.0, +14.0], EM +9.0 [+2.0, +16.0], token F1 +7.5 [+1.1, +14.1], recall@5 -0.6 n.s., +1.2 s and +$0.0060 a query; `mix` and `local` scored +19 and +18 points of `answer_contains` over `naive` on the same questions. A `[raptor]`-only namespace was not measured. One repeat on one corpus, so opt-in. On sensemaking questions, with the comprehensive answer, `mix`, which needs no community reports (`[lightrag]`; the reports were 26% of that index's model cost), answered at least as well as `global` ([RAPTOR summary tree](#raptor-summary-tree), [Sensemaking strategies](#sensemaking-strategies)) |
| `aws.bedrock.ingestion_effort` | `null` (`default_effort`, `high`) | `low` on a 109-document slice: ingestion cost -19%, answers n.s. on 30 questions ([Ingestion effort](#ingestion-effort)) |
| `search.model_timeout_seconds`, `search.query_timeout_seconds` | 90 s, 180 s | From the query times of the package's real-AWS evaluation runs ([Query deadlines and model timeouts](#query-deadlines-and-model-timeouts)) |
| `search.citations.enabled`, `search.citations.strategies` | `true`; `{global: false, drift: false}` | Against off on 200 MuSiQue questions, two repeats, interleaved: `local` and `mix` answers n.s. (`answer_contains` -0.75 [-2.5, +0.75] adjusted, and 0.0), citation precision 0.75 and 0.82, recall 0.88 and 0.85. On 50 sensemaking questions the answers without citations won comprehensiveness 0.617 [0.540, 0.695] 95% (`global`, Holm p 0.037) and 0.642 [0.573, 0.708] 95% (`drift`, Holm p 0.002) with the differential judge, averaged over both answer orders; a "cite where possible" instruction for them won no comprehensiveness back and cost `global` directness ([Citations](#citations)) |
| `search.iterative_search.max_rounds` | 2 | 3 against 2: `answer_contains` +1.5 [-0.5, +3.75] adjusted, n.s., +2.7 s and +$0.0024 a query; 44% of questions still stop at the cap ([Iterative settings](#iterative-settings)) |
| `search.token_manager.max_context_tokens` | 30000 | 12000 against 30000: `mix` -40% cost a query, `answer_contains` -2.25 [-5.75, +0.75] adjusted, n.s., citation precision -3.4 [-6.0, -1.1] adjusted; `local` -13% cost, n.s. ([Iterative settings](#iterative-settings)) |
| `aws.bedrock.default_effort` | `high` | `medium` against `high`: about 2 s faster and -4% cost a query, `answer_contains` n.s., token F1 -3.6 and -4.4 (their 95% intervals exclude zero; Holm p 0.50 and 0.41, n.s.) ([Iterative settings](#iterative-settings)) |
| `providers.rerank.kind` | `bedrock` | Off against on: recall@5 -7.9 (`local`), -25.4 (`mix`), -9.7 (`ppr`), all significant; answers n.s. (`mix` `answer_contains` -2.75 [-6.5, +0.5] adjusted); saves about $0.002 and 0.5 s a query ([Reranking](#reranking)) |
| `evaluation.judge_effort` | `low` | `high` against `low` on the same answers: spans identical in 84-86% (93-95% between two `low` runs), EM -1.0 to -1.75 n.s.; pairwise position consistency 0.82 to 0.86, but 10 of 100 overlap splits failed at their output limit ([Judge effort](#judge-effort)) |

## Answer style

On the 50 sensemaking questions, comprehensive `global` answers against
concise ones:

- **Win rates** on comprehensiveness, diversity and empowerment: 0.98 (Opus
  5.5 judge), 0.99 (Sonnet 5.5) and 0.95-0.99 (GPT-5.5). Directness: 0.02,
  0.00 and 0.39. Position consistency (both orders agreed): 0.97, 0.98 and
  0.84.
- **Cost and length.** +$0.0164 per query; answers grew from about 3.2k to
  7.6k characters.

`drift` with the comprehensive answer won the same way (comprehensiveness
1.00, diversity 0.88-0.95, empowerment 0.98, directness 0.03-0.04; position
consistency 0.92-0.96) for +$0.0169 per query. Its latency is why it stays
concise: see [Query deadlines and model timeouts](#query-deadlines-and-model-timeouts).

Pairwise LLM judges favour the longer answer, so a run on 2026-11-06
re-judged the same saved answers (no answer call) with a judge built to
discount length, an experiment that is not in the package: Haiku 5.5
split each pair into the points both answers make and the points unique to
each, and Opus 5.5 judged the unique points alone, in both orders.

| Strategy | Judged | Orders agreed | Comprehensiveness | Diversity | Empowerment | Directness |
|---|---|---|---|---|---|---|
| `global` | 49 of 50 (one split failed) | 81% | 0.95 [0.89, 0.99] | 0.91 [0.85, 0.97] | 0.89 [0.83, 0.94] | 0.29 [0.19, 0.39] |
| `drift` | 50 | 79% | 0.94 [0.89, 0.98] | 0.87 [0.81, 0.93] | 0.80 [0.70, 0.89] | 0.41 [0.30, 0.52], n.s. |

The comprehensive answers still won the three coverage criteria, so their
lead is not length alone. Their directness cost is smaller than the
standard Opus 5.5 judge's 0.02 (`global`) and 0.04 (`drift`) suggest: 0.29
for `global`, and not significant for `drift`. This is one judge model on
one question set.
`search.answer_prompt_style: {global: concise}` keeps the short answers
([Answer style](./retrieval.md#answer-style)).

## Global evidence cap

`search.global_search.evidence_tokens` caps the report text a global answer
reads beside its ranked key points. On the 50 sensemaking questions:

| Cap | Cost per global query | Quality against no cap | Sources |
|---|---|---|---|
| `null` (no cap) | $0.1016 | — | kept |
| `4000` (default) | $0.0597 (-41%) | No criterion consistently different. Position consistency 0.43-0.57: the judges could not tell the answers apart. Only Opus 5.5's empowerment, 0.41 [0.33, 0.48], left 0.5, and Sonnet 5.5 ([0.34, 0.50]) and GPT-5.5 ([0.37, 0.55]) did not replicate it | kept |
| `0` (key points only) | $0.0514 (-49%) | All criteria n.s. (position consistency 0.44-0.64) | none |

No quality difference was detected, but the run could only see a large
one: on 50 questions a win-rate interval spans about 0.08 either side of
its estimate, so a smaller effect would go unseen, and the questions were written from the level-0 report
summaries, the text the cap truncates. 4000 keeps the sources the
`retrieval` evaluator and the user read.

## The evaluation judge

Opus 5.5 and Sonnet 5.5 answered the same 50 global questions from the same
retrieval (concise answers), and Opus 5.5, Sonnet 5.5 and GPT-5.5 judged
the pairs:

- **Opus-answer win rates** (comprehensiveness / diversity / empowerment /
  directness): GPT-5.5 judge 0.74 / 0.64 / 0.67 / 0.76, Opus 5.5 judge
  0.73 / 0.55 / 0.68 / 0.65, Sonnet 5.5 judge 0.71 / 0.59 / 0.62 / 0.49.
- **Self-preference.** The Sonnet 5.5 judge's directness win rate for the
  Opus answers was 0.27 [0.16, 0.38] lower than the GPT-5.5 judge's, and
  0.16 [0.06, 0.26] lower than the Opus 5.5 judge's; the other criteria
  were n.s. The Opus 5.5 judge also differed from GPT-5.5 on diversity
  (-0.09 [-0.16, -0.03]) and directness (-0.11 [-0.21, -0.02]).
- **Agreement.** On the same pairs the judges agreed on 0.68-0.98 of the
  verdicts, and opposite decided verdicts were rare (0-10 of 74-194).

So the unset judge on Bedrock is Opus 5.5, a model other than the default
answer model. At the list prices above it costs twice Sonnet 5.5 per token,
and only `run-eval`'s LLM judges call it
([Model selection notes](../reference/models.md#model-selection-notes)).

**Answer slots.** The differential judge (`evaluation.pairwise_mode:
differential`) splits a pair into the points both answers make and the
points unique to each. Until 2026-11-08 it split each pair once, with the
baseline as Answer 1, and the split favoured Answer 2. A run on 2026-11-08
(package at commit `d6fc92ce`) judged one configuration against itself on
the 50 sensemaking questions, two repeats: the candidate won
comprehensiveness 0.645 as Answer 2 and 0.48 as Answer 1 (the baseline
0.52); averaged over both slots 0.562 [0.490, 0.635] 95%, n.s. That slot effect (as
Answer 2 minus as Answer 1) was +0.165 on comprehensiveness, and +0.125 to
+0.215 in the citation comparisons below, larger than most differences
between strategies there (-0.12 to +0.07). So the comparisons from
2026-11-08 on average both slot orders, as `run-eval` does
([Evaluation](../guides/evaluate.md); results in
[Sensemaking strategies](#sensemaking-strategies) and [Citations](#citations)).
They used the earlier overlap prompt, which kept a shared statement in
Answer 1's wording; `run-eval`'s split keeps the shorter wording, which
these runs did not measure.

## Gleaning rounds

Each item stores the time it was written, so one ingestion of the shared
index (3 rounds) separates what extraction and each gleaning round added:

- **Graph recall.** The 522 decomposition answers of the 200 questions
  (bridge and final answers) found as graph entities, by exact entity name:
  83.5% after extraction, 84.5%, 84.7% and 84.9% after rounds 1-3. Gleaning
  adds +1.3 points [+0.4, +2.3], rounds 2-3 +0.4 [0.0, +1.0] (these
  intervals resample the 522 answers as if independent; they come from 200
  questions, so the intervals are somewhat too narrow); every gain is
  a last-hop answer (68.5% to 72.0%), none a bridge answer. A looser name
  match raises every level by about 4 points and leaves the deltas unchanged.
- **Relationships.** On the supporting paragraphs, gleaning adds 3,761
  relationships to extraction's 4,922 (+76%) and corrects 19,936. A bridge
  (the entity linking one hop's paragraph to the next) has an edge in the
  next paragraph for 183 of 298 bridges after extraction, 201 after round 1
  (+6.0 points) and 207 after round 3 (+2.0 from rounds 2-3). Like the
  graph recall above, these counts come from the first index, before its
  138 units without an extraction were repaired
  ([Limits of the evidence](#limits-of-the-evidence)).
- **Ingestion cost.** Gleaning is $173.6 of the $371.64 build (47%);
  rounds 2-3 are about $111 of it, by call count.

The 109-document slice ingested once with `max_rounds: 3` and once with `1`
cannot measure rounds: its two extractions differ before any gleaning (65
against 59 of 82 answers found), because the runs lost different units to
failed extractions and named some entities differently ("Rio Linda,
Sacramento, CA" as one entity). Of the 7 answers only the 3-round run found,
at most one came from gleaning.

**Answers.** A run on 2026-11-06 measured the answers on the repaired
index ([Limits of the evidence](#limits-of-the-evidence)) against a copy
with round 1 alone: everything rounds 2-3 wrote removed by its write time
(52 entities and 1,291 relationships that rounds 2-3 corrected kept their
corrected form, as no earlier version was stored), the communities and
reports unchanged. On the 200 MuSiQue questions, one repeat, round 1 alone
against 3 rounds:

| Strategy | Answers | Query cost |
|---|---|---|
| `local` | `answer_contains` -3.0 points [-6.5, 0.0] (2 wins, 8 losses), borderline; EM -2.0 and token F1 -2.2, n.s. | -$0.0006 a query |
| `mix` | EM +5.0 [+1.0, +9.5] (14 wins, 4 losses); `answer_contains` 0.0 and token F1 +2.1, n.s. | -$0.0032 a query (-6%) |
| `hybrid` | n.s. on every metric | -$0.0032 a query |

So the answers do not favour either setting: `local` lost on one metric at
the edge of significance and `mix` gained on another. 3 rounds stays the
default; a second repeat of `local` would settle its difference. MuSiQue
paragraphs are about 110 tokens; the GraphRAG paper motivates gleaning
with 600-2,400-token chunks, which this index does not test.
`max_rounds: 1` keeps most of the relationship gain at about a third of
the gleaning cost ([Cost levers](../guides/ingest.md#cost-levers)).

## Ingestion effort

`aws.bedrock.ingestion_effort` sets the reasoning effort of the default
tier's ingestion calls without changing the queries'. A run on 2026-11-06
ingested the 109-document slice twice with that day's code, `low` against
`high` (the `default_effort` that `null` inherits), and answered its 30
questions from each, one repeat:

- **Ingestion.** $15.11 to $12.31 at the list prices (-19%), output tokens
  -24%, 419 s to 384 s. The graph had 11% fewer entities and 20% fewer
  relationships, but more of the questions' 82 decomposition answers were
  graph entities by exact name with `low` (64 against 61; 79 each by a
  looser match). No text unit was left without an extraction in either
  run.
- **Answers.** n.s. on `local` and `mix` (30 questions: `answer_contains`
  +3.3 [-6.7, +13.3] and 0.0 [-10.0, +10.0], `low` against `high`).
- **Queries.** Over the smaller graph: -$0.003 and -$0.004 a query, both
  significant; -2.7 s and -1.6 s (`local`, `mix`), from arms run one after
  the other (see *Latency* in [How the measurements were run](#how-the-measurements-were-run)).

So `low` is a cost lever of about a fifth of the ingestion. Thirty
questions cannot show a small quality loss, so the default is unchanged
([Cost levers](../guides/ingest.md#cost-levers)).

## LightRAG chunk picking and MMR diversity

On the 200 MuSiQue questions:

| Change | `mix` | `hybrid` | Latency and cost |
|---|---|---|---|
| `kg_chunk_pick_method: vector` (upstream LightRAG's default) instead of `weight` | `answer_contains` -1.5 points [-4.5, +1.25], EM and F1 n.s. | `answer_contains` -1.25 [-4.5, +2.0], MRR +1.5 [-0.01, +3.2], n.s. | +1.5 s per query, within the run-to-run variation (see *Latency* in [How the measurements were run](#how-the-measurements-were-run)); cost unchanged |
| `diversity_lambda: 0.7` instead of 0.5 | n.s. on every metric | — | +0.57 s |
| `diversity_lambda: 1.0` instead of 0.5 | `answer_contains` -2.25 [-5.25, +0.5] n.s.; MRR +0.9 [+0.05, +2.15], significant but small | — | -$0.0009 per query |

No change was significantly better on the answers, so both defaults stayed
([Search strategies](./retrieval.md#search-strategies--when-to-use-which)).

## Retrieval switches

A later run (2026-11-05) on the same index, models and 200 MuSiQue
questions, two repeats each, compared five retrieval switches against the
defaults. "Supporting paragraphs" is the share of each question's MuSiQue
supporting paragraphs among the text chunks of the answer context, and
latencies are the answer call's alone. Each arm ran on its own, so a
latency below about 2.5 s is within the run-to-run variation (see
*Latency* in [How the measurements were run](#how-the-measurements-were-run)).

| Change | Answers | Retrieval | Latency and cost |
|---|---|---|---|
| `local`: no `entity_frequency_threshold` cut instead of 20 | `answer_contains` -1.5 points [-3.0, 0.0], borderline; EM and F1 n.s. | recall@5 -1.75 [-3.12, -0.50] and supporting paragraphs -2.42 [-3.96, -1.04], both significant | +3.08 s (significant), +370 input tokens |
| LightRAG: no per-side cut of the knowledge-graph chunks before the rerank | n.s. on `mix` and `hybrid` | `hybrid` supporting paragraphs +1.87 [+0.12, +3.71], significant; `mix` n.s. | `hybrid` +0.82 s (significant), +195 input tokens |
| LightRAG: chunks picked from upstream's merged, token-truncated entity and relationship lists | n.s. on `mix` and `hybrid` | n.s. | +0.48 s `hybrid`, +0.32 s `mix` (both significant) |
| Relationships embedded as upstream LightRAG does (source, target and description) instead of the description (`hybrid`, one repeat) | n.s. | n.s. | +0.24 s; a re-embedding of every relationship |
| `search.token_manager.type_budgets.relationship` 0.27 instead of 0.13 | Not measurable: the shares never bound on MuSiQue, so both arms read the same context | — | — |

None was better on the answers, so every default stayed: the frequency cut
for its retrieval, the others because they gained nothing on the answers
(and the re-embedding would cost a pass over every relationship). The context shares need a corpus whose contexts fill their budget to
be measured.

## Local report selection

`search.local_search.report_selection: membership` reads the reports of
the leaf (level-0) communities the traversed entities belong to;
`similarity` (the default) reads the reports most similar to the query
entities. MS GraphRAG local search reads each entity's community at its
`community_level` cut (default 2) instead, so the measurement covers the
leaf variant, not upstream's selection. A later run on the same 200 questions, two repeats
each, measured membership against similarity: exact match -4.0 points
[-7.75, -0.25], token F1 -4.1 [-7.7, -0.6], both significant, for about 0.5%
fewer tokens. Similarity stays the default
([Design §6.1](../design/retrieval-graphrag-lightrag.md#61-graphrag-methodology-adapterssearch_strategies)).

## Local PPR ranking

`search.local_search.text_unit_ranking` orders the text units of `local`'s
expanded entities before the text quota cuts them (DRIFT's follow-ups keep
`support`, `search.drift_search.text_unit_ranking`: these runs measured
`local` alone). `ppr`, the default, ranks them by HippoRAG 2's Personalized PageRank
over the expanded entities, their relationships and the passages they cite,
seeded by local's entities, the question's passage hits and the endpoints of
its relationship hits ([Design §6.4](../design/retrieval-hipporag-raptor.md#64-hipporag-2-personalized-pagerank-ppr-local-ranking));
`support`, the earlier default, by how many expanded entities cite a unit.
A retrieval-only screen first chose the seeding: on the same 200 MuSiQue
questions, one repeat, real models, the ranking as configured by default
against `support` put more supporting passages in the context (+11.0
points [+8.4, +13.8]) and raised recall@5 after the rerank by 4.1 [+1.0,
+7.1]. Fusing the same passage hits into `support`'s order without the
PageRank gained +6.4, so the PageRank adds 4.7 [+2.5, +6.8] beyond the
passage hits; seeding from the entities alone gained only +3.9. The
screen's retrieval took about 0.4 s more a query on the embedded stores
(the PageRank itself about 50 ms); the answer runs below measured the
shipped ranking at +0.10 s.

Two runs on 2026-11-06 (package at commits `e4cd96e5` and `85d7e6ec`)
measured `ppr` against `support` on the same index, models and 200 MuSiQue
questions, one repeat each, the rerank on, both arms reading the same query
extractions:

| Measure | `ppr` against `support` |
|---|---|
| Answers (both runs, averaged per question) | `answer_contains` +3.5 points [+0.5, +6.75] (13 wins, 7 losses); EM +2.25 and token F1 +2.5, both n.s. |
| Retrieval | recall@5 +6.1 [+2.9, +9.4]; supporting paragraphs in the context +13.1 and +12.3 [+9.4, +15.5], last hop in the context +16.5 and +15.5 [+10.0, +21.0] (first and second run) |
| Where it gains | The 95 questions whose `support` context lacked a supporting paragraph: token F1 +12.7 and `answer_contains` +8.4 points, both significant. The 105 whose context held them all: token F1 -4.0 [-8.5, +0.6], n.s. (first run) |
| Latency | Retrieval +0.10 s [+0.05, +0.18] a query, paired one query at a time on the embedded stores (rerank off); the answer's latency n.s. |
| Cost | No model call more, one query embedding more. The runs' -$0.0023 a query is not the ranking's: the `ppr` arm read the query extractions the `support` arm had saved, one model call a query fewer |

`ppr` became the default for its retrieval gain (recall@5 +6.1, supporting
paragraphs in the context +12 to +13 points in each run), with answers no worse.
The answer gain is weak evidence on its own: the `answer_contains` interval
over the two repeats averaged excludes zero (permutation p = 0.048), but
neither repeat alone does (p = 0.12 each), the sign test over its 13 wins
and 7 losses gives p = 0.26, and a Holm correction over the three answer
metrics leaves it at 0.15. Three runs of the `support` configuration
differed from each other by up to 2.5 points of `answer_contains`.
`support` keeps the earlier order:

```yaml
search:
  local_search:
    text_unit_ranking: support
```

Limits. The gain is small and comes from the questions whose `support`
context missed a paragraph; MuSiQue is English multi-hop text, and DRIFT's
follow-ups, which use the same ranking, were not measured. Under 40
concurrent queries in one `run-eval` process the retrieval step took 1.1 to
1.7 s more instead of 0.10 s: the extra searches' Python work waited behind
the other queries' for the interpreter lock, and numpy started a BLAS
thread per core for every kNN product. This release cut that shared work
(*Changed* in the CHANGELOG): on the same index with the model calls faked,
40 concurrent queries on a host loaded by other work, the mean retrieval
time of `support` fell from 19-20 s to 9-10 s and of `ppr` from 23-25 s to
11 s, but `ppr` still adds 1.6 to 2.2 s there, the Python work of its two
extra searches queued behind the other queries'. A deployment serving many
concurrent queries from one process should measure its own latency. Without the
relationship vector index (`indexing.opensearch.build_relationship_vector_index:
false`) the PageRank is seeded by the entities and passages alone, which was
not measured.

## The ppr strategy

The `ppr` strategy answers from HippoRAG 2's PageRank alone: no query
extraction, one model call (the answer). The first run above measured it
on the same 200 questions, one repeat, against `local` and against a `mix`
run on the same index:

| Against | Answers | Retrieval | Latency and cost |
|---|---|---|---|
| `local` | `answer_contains` -2.0 [-7.5, +3.5], token F1 -2.1 [-7.0, +3.4], EM -3.5: all n.s. | supporting paragraphs in the context +10.3 [+6.6, +14.1], recall@5 +4.4 [+0.7, +8.2] | p50 10.8 s to 5.1 s; $0.0391 to $0.0164 a query (2.4x cheaper) |
| `mix` | `answer_contains` -2.0 [-7.5, +3.0], token F1 -2.9 [-7.1, +1.4]: n.s.; 4-hop questions token F1 -15.7 [-28.5, -5.2] (a subgroup, exploratory: not corrected for the subgroups read) | supporting paragraphs +5.9 [+3.2, +8.6] | p50 6.8 s to 5.1 s; $0.0553 to $0.0164 (3.4x cheaper) |

So `ppr` is the cheap, fast option: answers as good within the noise for
a third to half of the cost and a lower median latency, weaker on the
longest chains. `auto` does not route to it (`search.auto_routable_strategies`
refuses it): select it explicitly. A variant closer to HippoRAG 2's
release (rerank off, passage-seed weight 0) scored EM +5.5 and token F1
+5.9 against the default `ppr` with `answer_contains` and the supporting
paragraphs unchanged and MRR -9.2, which reads as answer phrasing rather
than retrieval. It changed two settings at once, so it cannot say which one
moved the scores; it needs a repeat, one setting at a time, before any
default changes.

## Iterative follow-up rounds

`iterative` (§6.3 of the [Design Doc](../design/retrieval-graphrag-lightrag.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in))
reruns its base strategy for the bridge entities a fast-model controller
says the answer still depends on. A run on 2026-11-06 (package at commit
`d0615b8f`) measured it on the same 200 MuSiQue questions, embedded stores
and default models (the controller on the fast tier), one repeat, every arm
paired with its base on the same questions. "Last hop in the context" is the share of questions whose
last-hop supporting paragraph reached the answer context.

| Arm against its base | Answers | Retrieval | Latency and cost |
|---|---|---|---|
| `iterative` over `mix` (the default base, controller loop) against `mix` | `answer_contains` +12.5 points [+7.5, +18.0] (27 wins, 2 losses), token F1 +5.2 [+0.2, +10.1], EM +4.0 [-0.5, +8.5] n.s. | recall@5 +13.9 points, supporting paragraphs in the context +13.3, last hop in the context +21.5 (43 wins, 0 losses) | p50 6.8 s to 16.1 s, p95 11.1 s to 27.4 s; $0.0553 to $0.0571 a query (+$0.002) |
| `iterative` over `local` against `local` | `answer_contains` +10.0 [+5.5, +15.0], token F1 +5.7 [+0.4, +10.7], EM +7.5 [+2.0, +13.5] | recall@5 +14.0, last hop in the context +23.0 | p95 13.5 s to 32.1 s; $0.0392 to $0.0400 a query |
| A KAG-style decomposition (removed) over `mix` against the controller loop | token F1 -0.6 n.s., `answer_contains` -2.5 n.s. | recall@5 +2.7 n.s. | +$0.0009 a query, latency the same |

The gain comes from the questions whose base context lacked a supporting
paragraph (89 of 200: token F1 +13.6, `answer_contains` +29.2); where the
base context already held every one, the follow-up did not hurt (token F1
-1.5, n.s.). The controller stopped after round 0 on 57 questions, after
one follow-up round on 48 and after two on 95, at 2.19 controller calls a
question; the answers were no longer.

That run's +$0.002 a query is not today's cost: in the
[baselines of 2026-11-07](#noise-floor-and-baselines) `iterative` over
`mix` cost $0.0653 a query against `mix`'s $0.0554, +$0.0099 (+18%), and
read 82.7k input tokens against 25.2k. Plan with the later figure.

So `auto` may route to `iterative` (`search.auto_routable_strategies`),
whose router entry keeps it to questions that chain facts through an
entity they do not name, but no strategy defaults to it: it adds about 10 s to
the median query. The decomposition arm cost more for no gain over the
loop, so it was removed.

This run did not measure whether the router leaves single-hop and
sensemaking questions to the other strategies (the second run below did).
In 62 of its 238 controller rounds over `mix` the verdict named no new
bridge entity and mix reran without keywords; such a round now runs only
its text searches (Design §6.3), which this run predates.

A second run on 2026-11-06 (package at commit `85d7e6ec`, with that fix)
repeated the comparison and measured the routing:

- **`iterative` over `mix` against `mix`**, one repeat: `answer_contains`
  +13.0 points [+8.0, +18.0] (28 wins, 2 losses), token F1 +3.7 [-1.5, +9.1]
  and EM +2.5, both n.s.; recall@5 +14.4, last hop in the context +21.5 (43
  wins, 0 losses). p50 8.0 s to 15.3 s, p95 12.6 s to 26.3 s; +$0.0016 a
  query. No round reran `mix` without keywords; the controller stopped after round
  0 on 61 questions, after one follow-up round on 44 and after two on 95.
- **`auto` on the 200 MuSiQue questions** (two routing passes): 191 and 193
  routed to `iterative`, the rest to `local`. Each question answered by the
  strategy it was routed to, against `mix`: `answer_contains` +12.0 [+7.0,
  +17.0]. The router added
  1.35 s at the median and $0.0005 a query; the entity extraction it starts
  alongside is not read on a route to `iterative` over `mix`, about $0.003
  a query.
- **Misroutes.** Of 150 routing decisions on the 50 sensemaking questions
  (three passes) none went to `iterative` (45 to 48 of 50 to `global`, the
  rest to `mix`). On 100 single-hop questions (the first sub-question of
  100 of the MuSiQue questions, one supporting paragraph each) 2 and 4 of
  100 went to `iterative`, all answered correctly, and `auto` matched `mix`:
  `answer_contains` 0.99 against 0.99.

So `iterative` stays routable and the router's guidance stands; route a
sample of your own questions with `run-eval --search-strategy auto` before
relying on it. Both runs predate the router's JSON answer
(`{"strategy": ...}`, the same candidates and guide entries); its agreement
with these routes has not been measured yet. To keep `auto` off it, list the strategies without
`iterative`:

```yaml
search:
  auto_routable_strategies: [local, mix, global]
```

## RAPTOR summary tree

`tree` searches the text units and every level of a summary tree (§6.5 of
the [Design Doc](../design/retrieval-hipporag-raptor.md#65-raptor-summary-tree-raptor-tree-opt-in)).
A run on 2026-11-07 (package at commit `1c80d7da`) built the tree on a copy
of the repaired index with `run-ingestion --enabled-stages summary_tree`
and `models.roles.summary_tree: fast`: 603 summaries in 4 levels over the
3,318 text units, 603 Haiku 5.5 calls (1.08M input and 0.23M output
tokens), $0.22 at the list prices, in 116 s. Every summary ends a
sentence; 421 of them (70%) were longer than `max_summary_tokens` and cut
at a sentence end, which is why later packages ask for less (§6.5).

The copy held the default `[graphrag, lightrag]` index beside the tree, so
`tree` ran with its graph signal (`search.tree_search.graph_signal: auto`
fuses the summaries and passages citing the entities the raw query finds
where the namespace holds the entities index). The numbers below measure
that configuration, not a `[raptor]`-only namespace, whose `tree` has no
graph signal. On 100 of the 200 MuSiQue questions, one repeat, paired on
the same questions:

| Arm against `naive` | Answers | Retrieval | Latency and cost |
|---|---|---|---|
| `tree` | `answer_contains` +8.0 points [+3.0, +14.0] (0.62 to 0.70), EM +9.0 [+2.0, +16.0], token F1 +7.5 [+1.1, +14.1]: each significant on its own (95% intervals, not adjusted for the several metrics read) | recall@5 -0.6 [-3.2, +2.0], n.s.; MRR -1.9 [-4.0, -0.3]: the passages come first, and the 5.2 summaries a context gains follow them | +1.2 s [+0.9, +1.4] (4.1 s to 5.3 s); +$0.0060 a query ($0.0182 to $0.0243), the summaries' 2,922 input tokens |
| `mix` | `answer_contains` +19.0 [+11.0, +27.0], EM +15.0 [+8.0, +23.0] | recall@5 +2.5, n.s. | +4.3 s; +$0.0407 a query |
| `local` | `answer_contains` +18.0 [+7.0, +29.0], EM +10.0 [+1.0, +19.0] | recall@5 +2.0, n.s. | +5.0 s; +$0.0266 a query |

The `mix` and `local` rows are from the run of 2026-11-06 (package at
`1d11766b`) on the same questions and index. That run measured an earlier
tree (summaries cut mid-sentence, summary nodes ranked above the passages):
`tree` against `naive` was `answer_contains` +4.0 [-1.0, +10.0], n.s.,
with recall@5 -8.7 [-13.6, -3.8] and +5.5 s. The RAPTOR changes of
2026-11-06 (summaries kept to whole sentences, the passages ranked before the
summaries' lineage) came between the two runs; with them the gain is
significant and the retrieval loss is gone.

So on multi-hop questions over short paragraphs the tree is cheap to build
and adds about 8 points of `answer_contains` over `naive` for $0.006 and
1.2 s a query, still well below the graph strategies. It stays opt-in and
`auto` does not route to `tree`: this is one repeat on one corpus, and the
graph strategies win there by more than twice as much. Not measured: a
`[raptor]`-only namespace (no graph signal), `graph_signal: off`, long
documents, `scope: document` and `algorithm: gmm`; sensemaking questions
are in [Sensemaking strategies](#sensemaking-strategies).

## Sensemaking strategies

A run on 2026-11-08 (package at commit `d6fc92ce`) answered the 50
sensemaking questions with every strategy against `global`, two repeats,
on the index of the [Citations](#citations) run with a summary tree built
beside it ($0.20). Every strategy used the comprehensive answer
(`search.answer_prompt_style`), citations kept their defaults (off for
`global` and `drift`, on for the others), and the differential pairwise
judge (`judge_effort: low`) averaged both answer slots
([The evaluation judge](#the-evaluation-judge)). Win rates against
`global`, with the Holm p over the 28 tests (seven comparisons, four
criteria) where it is below 0.05; each estimate's 95% interval spans about
0.1 either side:

| Strategy | Comprehensiveness | Diversity | Empowerment | Directness | Cost a query | p50 / p95 |
|---|---|---|---|---|---|---|
| `drift` | 0.655, n.s. (Holm p 0.078) | 0.472 | 0.746 (Holm p 0.001) | 0.829 (0.001) | $0.132 | 53 / 63 s |
| `tree` | 0.627 | 0.569 | 0.706 (0.002) | 0.746 (0.001) | $0.049 | 20 / 27 s |
| `mix` | 0.576 | 0.569 | 0.674 (0.029) | 0.757 (0.001) | $0.095 | 27 / 35 s |
| `local` | 0.542 | 0.568 | 0.675 (0.029) | 0.710 (0.002) | $0.066 | 25 / 33 s |
| `naive` | 0.445 | 0.473 | 0.625 | 0.652 (0.0495) | $0.038 | 18 / 25 s |
| `global` (the reference) | — | — | — | — | $0.071 | 29 / 40 s |

`drift` against `local` won comprehensiveness 0.792 (Holm p 0.001),
empowerment 0.657 (0.030) and directness 0.636 (0.030), with diversity
0.482, n.s.

- **No strategy beat `global` on comprehensiveness or diversity**, and
  `drift`, `tree`, `mix` and `local` won empowerment and directness.
- **`drift` answered best**, ahead of `local` on three criteria, at about
  twice `global`'s cost and median latency.
- **`mix` matched `global` without community reports.** A `[lightrag]`
  namespace builds no communities or reports
  ([methodology profiles](./overview.md#methodology-profiles)); on this
  index the reports were $95.6 of the $371.64 build (26%).
- **`simple`** is not in the table: its retrieval is under revision and is
  measured again after the change.

No default changed. Limits: one corpus of short Wikipedia paragraphs, one
judge (Opus 5.5), and questions written from the level-0 report summaries
that `global` reads. The challengers cited and `global` and `drift` did
not, which costs comprehensiveness ([Citations](#citations)); and the
defaults answer concisely with every strategy but `global`.

## Query deadlines and model timeouts

**Why 90 s and 180 s.** In the package's real-AWS evaluation runs (5,846
queries) the p99 query took 47 s and the slowest 61 s, in all. A
non-streamed Converse answer at `default_effort: high` can spend most of
that in one read, so `search.model_timeout_seconds` is 90 s: about 1.5x the
slowest whole query and 1.9x the p99, for one request. The deadline,
`search.query_timeout_seconds` 180 s, is `model_max_attempts x
model_timeout_seconds`, so a call that stalls once can still be answered by
its second attempt; it is about 3x the slowest query and 3.8x the p99. Two
attempts, not more: a third would start only after the deadline.

**DRIFT near the deadline.** Those runs did not include DRIFT. On the
3,318-document MuSiQue index (50 sensemaking questions, two repeats, one
`run-eval` at a time), DRIFT with the concise answer took 147 s at p95;
with the comprehensive answer 168 s at p95 and 178 s at most, 2 s inside
the 180 s deadline. A second evaluation running at the same time pushed 20
of 50 comprehensive DRIFT answers past it. So DRIFT keeps the concise
answer by default, and the runbook's advice for routing DRIFT, serving it
under load or giving it the comprehensive answer is a 300 s deadline
([Query deadlines and model timeouts](../operations.md#query-deadlines-and-model-timeouts)).

## Declines and abstention

`search.abstention.mode: mark` asks the answer model to start a declined
answer with `[NO_ANSWER]`, so a decline is machine-readable;
`on_insufficient: abstain` declines whenever the strategy flagged its own
context insufficient, which `iterative` does
([Querying: declined answers](../guides/query.md#declined-answers)). A run
on 2026-11-06 (package at commit `d0615b8f`, embedded stores, default
models, one repeat) measured both on a dedicated index:

- **Benchmark.** `prepare-benchmark musique --unanswerable 100 --hops
  2=20,3=40,4=40 --seed 0`: 100 answerable questions (20 two-hop, 40
  three-hop, 40 four-hop) and 100 MuSiQue-Full unanswerable ones over 3,312
  paragraphs. Of 188 unanswerable candidates examined, 82 (44%) were left
  out because the text of the removed supporting paragraph was still in the
  pooled corpus, and 6 because their answerable twin was sampled. The
  filter left the unanswerable half 85 two-hop, 13 three-hop and 2
  four-hop, so the two halves differ in hop count.
- **Index.** One ingestion with the defaults, no failed text unit: $361.53
  at billed prompt-cache prices ($427.76 counting cache reads and writes as
  input). The whole experiment cost about $427 billed ($496 upper bound).
- **Declines in the model's own words.** With `mode: off` only the
  empty-context reply is detected, so `off`'s detected decline recall is 0.
  To compare like with like, an English pattern over each answer's first
  line ("the context does not...", "I cannot...", in the analysis script,
  not the package) estimated the declines the model writes in prose.
  "Coverage" is the share of answerable questions not declined.

| Arm against its base | Answers on the answerable half | Declines | Cost and latency |
|---|---|---|---|
| `mark` against `off`, `local` | EM -12 points [-20, -5], token F1 -7.7 [-14.8, -1.1], coverage -34 [-43, -25]; token F1 over the answered ones +18.0 [+10.3, +26.4] | recall 0.90, precision 0.73; recall +8 [+3, +13] over `off`'s own-word estimate (0.83) | -$0.0011 a query, -0.6 s |
| `mark` against `off`, `mix` | EM -10 [-18, -3], token F1 -11.3 [-18.9, -4.2], coverage -33 [-42, -24]; token F1 over the answered ones +10.2 [+1.2, +19.3] | recall 0.90, precision 0.73; +3 [-2, +8] n.s. over `off`'s own-word estimate (0.87) | -$0.0014 a query |
| `iterative` over `mix` (`mark`), `on_insufficient: abstain` against `hedge` | EM -12 [-20, -4], token F1 -13.8 [-21.7, -6.0], coverage -18 [-26, -10] | recall +6 [+1, +12], precision -9.0 [-14.0, -4.3] | -$0.0007 a query, -0.9 s |

Why `mark` costs answers: Sonnet 5.5 already declines in prose with `mode:
off` (an estimated 83-87% of the unanswerable questions), so the marker
adds little recall. What it changes is the hedged answer that gives the
right entity and says what the context does not cover: under `mark` the
model starts such an answer with the token, and it counts as a decline. On
`local`, `mark` declined 34 answerable questions; 9 of them had been exact
matches under `off`. Token F1 over
the answered questions rises by up to 18 points because the declined ones
are the hard ones, so it is a selection effect, not a better answer. So
`mark` stays opt-in: turn it on when a caller needs to tell a decline from
an answer without reading it, and accept the lower coverage.

Why `hedge` stays the default: the `iterative` controller still found the
context insufficient when it stopped on 92-93 of the 100 unanswerable
questions, but also on 45-46 of the 100 answerable ones, so `abstain`
declined many questions the fused context did answer (coverage 0.55).
`hedge` answers them with a caveat instead.

The same run confirmed `iterative` on a second index: `iterative` over
`mix` (`mark`, `hedge`) against `mix` (`mark`) scored EM +11 [+5, +18] and
token F1 +9.2 [+2.5, +16.6], for +$0.0047 a query and +8.4 s
([Iterative follow-up rounds](#iterative-follow-up-rounds)).

Not settled by this run: one repeat; English questions only (the own-word
pattern is English, the marker is not); and noisy unanswerable labels: in
about 19 of the 100 unanswerable questions the answer named the removed
twin's answer, inferred from other paragraphs, so part of those
"unanswerable" questions are answerable from the corpus and decline
precision and recall carry that noise. The different hop mix of the two
halves also lets a model separate them by question shape.

## Reset cost after incremental runs

An incremental run clusters only its delta and appends its communities, so
a [rebuild signal](../guides/incremental.md#rebuild-signals) eventually
recommends `indexing.reset: true`. A reset re-clusters the whole graph and
reuses the stored description summaries and community reports whose input
is unchanged, so it asks the model mostly for the communities it clusters
differently. An offline harness, which runs ingestion on the embedded local
backend with a scripted model and counts the model calls, measured it:
after 23 incremental runs, a reset made 17-20 model calls, against 197 for
the first build. These are call counts, not tokens or dollars: they show
how much a reset reuses, not what it costs with your models and corpus.

## Ingestion memory and run time

The runbook's [capacity sizing](../operations.md#capacity-sizing) scales
these figures. A full local build of 5,000 to 60,000 short text units
(about 200 characters, two or three entities each) with fake models and
the embedded stores measured:

- **Memory.** About 42 KiB of resident memory per text unit on top of about
  0.2 GiB, growing linearly with the units of the run.
- **Stage cache.** About 11 KiB per text unit.
- **Time outside the model calls.** About 8 ms per text unit for parsing,
  resolution, community detection and indexing.

Real units are longer (`fallback_chunk_size` 4,800 characters) and name more
entities and relationships, so the runbook plans for 2-3 times the memory
per unit. That factor is an estimate, not a measurement. The time excludes
model latency and network writes to Neptune and OpenSearch.

A `raptor`-only run on AWS (2026-11-07, package at `1c80d7da`, 2 vCPU and
8 GiB) over 50,000 MuSiQue paragraphs (about 500 characters each; 33,165
distinct) took 1,474 s: indexing 562 s (Titan embeddings), the summary tree
811 s (9,294 Haiku 5.5 summary calls, 16.0M input and 3.27M output tokens,
about $3.2). Its task memory peaked at 1,361 MiB (Container Insights,
1-minute samples), about 21 KiB a unit end to end.

## On-disk vector recall

[On-disk vector storage](../operations.md#on-disk-vector-storage) trades
recall for data-node memory. On a synthetic 50,000-vector set (256
dimensions, OpenSearch 2.19.3,
`tests/integration/test_opensearch_vector_storage.py`):

| Engine and level | recall@10 | faiss native memory |
|---|---|---|
| Unquantized (faiss) | 1.0 | 61 MB |
| faiss 8x | 0.95 | 17 MB |
| lucene 4x | 0.93 | — |
| faiss 32x (`on_disk`'s default) | 0.78 | 12 MB |

A synthetic set is not a corpus's embeddings, so measure
recall on your own corpus before choosing a level.

## Ontology and RDF parse cost

An `owl` [schema source](../guides/ontology.md#load-cost) and an `rdf`
[seed graph](../guides/seed-graphs.md#configure) parse their files into
memory with the `ontology` extra's parser (rdflib) before anything is
mapped, so parsing sets the time and memory of a load:

- **Ontology.** A 14 MB Turtle file of 110,000 classes and 500 properties
  took about 45 s and 1.2 GB peak memory (parse 30 s, reading the classes
  and properties 10 s, mapping 5 s), that is about 3 s and 85 MB per MB of
  Turtle; a 20,000-deep `subClassOf` chain maps in under a second. At the
  default `options.max_bytes` (64 MiB) that would be about 5 GB.
- **RDF seed.** The parsed graph takes from about 25 (N-Triples) up to
  about 150 (dense Turtle) times the file size in memory, before
  `max_nodes` is checked: up to about 5 GB at the `rdf` default
  `max_bytes` of 32 MiB.

These are per-process load figures, not a benchmark: they depend on the
machine, the rdflib version and how densely the file is written. Measure
your own file with the
[ontology preview](../guides/ontology.md#preview) or the
[seed preview](../guides/seed-graphs.md#preview) before you size a task.

## Usage records against the bill

A `run-eval` USD estimate prices the calls the usage records count
([Evaluation: Comparing strategies](../guides/evaluate.md#comparing-strategies)):
botocore's resent requests, failed calls and the warm-up queries are not in
it. In the package's November 2026 benchmark runs Bedrock counted 13% more
Sonnet 5.5 invocations than recorded calls while the tokens agreed within
4.6% (input, cache included) and 0.9% (output), which points to resent and
failed requests that spent few tokens; Haiku 5.5 agreed within 0.2%. Those
runs predate the `request_retries` and `failed_calls` counters, so the split
between the two was not measured. Reconciling the counters with CloudWatch:
[Operator Runbook: Alarms](../operations.md#alarms) (model usage metrics).

## Noise floor and baselines

A run on 2026-11-07 (package at commit `b0ba0231`) measured the noise floor
and the baselines on the repaired index, the 200 MuSiQue questions and the
default models, every comparison interleaved in one `run-eval`
(`--run-order interleaved`), two repeats each, Holm-adjusted as `run-eval`
reports. EM and token F1 scored the extracted span with SQuAD's official
normalization, as above.

**Noise floor.** An empty `--arm` against the base, which reads the base's
query extractions: `local` `answer_contains` -1.25 points [-3.0, 0.0] 95%,
EM +0.5 and token F1 +0.9; `mix` -0.25, -0.25 and +0.5, all n.s.; latency
+0.14 s (`local`) and -0.10 s (`mix`), n.s., with 95% intervals up to
+0.67 s. Two separate `run-eval` invocations of one configuration,
each extracting its own query entities and keywords, differed by up to 1.5
points of `answer_contains`, 3.25 of EM and 2.4 of token F1 (n.s.), and by
0.6 to 1.0 s a query (significant: separate invocations measure what
changed between them). So a difference below about 1.5 points of
`answer_contains` or 3 points of EM or token F1 is within the noise on this
index.

**Baselines.** On the same questions:

| Strategy | `answer_contains` | EM | Token F1 | recall@5 | Cost a query | p50 / p95 |
|---|---|---|---|---|---|---|
| Closed book (answer model alone) | 0.443 | 0.235 | 0.333 | — | $0.0068 | 4.9 / 10.4 s |
| `naive` | 0.598 | 0.338 | 0.402 | 0.681 | $0.0143 | 3.9 / 6.6 s |
| `ppr` | 0.745 | 0.445 | 0.531 | 0.722 | $0.0158 | 4.0 / 7.2 s |
| `mix` | 0.760 | 0.460 | 0.551 | 0.700 | $0.0554 | 5.8 / 10.1 s |
| `local` | 0.808 | 0.495 | 0.576 | 0.741 | $0.0370 | 6.1 / 10.3 s |
| `iterative` | 0.878 | 0.518 | 0.600 | 0.847 | $0.0653 | 14.4 / 24.0 s |

Against `naive`, `answer_contains` rose +21.0 points [+13.6, +29.1]
adjusted (`local`), +16.25 [+9.4, +23.75] adjusted (`mix`), +14.75 [+8.5,
+21.5] adjusted (`ppr`) and +28.0 [+20.1, +36.25] adjusted (`iterative`),
each significant (Holm p 0.002), and so did EM and token F1. `iterative`
read 82.7k input tokens a query against `mix`'s 25.2k, +$0.0099 a query
(+18%). The closed-book answers already contain the gold answer for 44% of
the questions by `answer_contains`, which credits an answer that names the
gold one anywhere, so 44% is an upper bound on what the model answers from
memory: it knows much of this Wikipedia text, so the absolute scores
overstate what retrieval adds, and the paired differences between
strategies are the evidence.

## Citations

`search.citations` numbers the answer context and asks for `[n]` markers
([Citations](../guides/query.md#citations)). Its defaults are on for every
strategy but `global` and `drift`, from two runs.

**`local` and `mix`.** The run of 2026-11-07 above answered with citations
off as an `--arm`:

| Strategy | Answers, off against on | Cost and latency |
|---|---|---|
| `local` (200 MuSiQue questions) | `answer_contains` -0.75 [-2.5, +0.75] adjusted, EM +2.0, token F1 +2.2: n.s. | +39 output tokens, +$0.0015 a query; latency n.s. |
| `mix` | 0.0, 0.0 and -0.1: n.s. | +24 output tokens, -$0.0003 a query |

With citations on, citation precision (file-weighted) was 0.75 (`local`)
and 0.82 (`mix`), passage precision 0.83 and 0.85, citation recall 0.88 and
0.85. So citations stay on there: they cost nothing measurable and make
answers attributable.

**`global` and `drift`.** In that run their key points and intermediate
answers reached the context without a number, and the answers without
citations won comprehensiveness (0.75 and 0.79). A run on 2026-11-08
(package at commit `876c938b`) measured them again after key points and
intermediate answers became citable, each naming the documents of the
reports it was written from (its findings' records). It regenerated the
3,877 community reports of the index with their findings, on the same
communities (about $96), and answered the 50 sensemaking questions twice
with citations on and off, judged by the differential pairwise judge
(`evaluation.pairwise_mode: differential`, `judge_effort: low`, both orders,
every repeat). A rerun of that judge on 2026-11-08 averaged both answer
slots ([The evaluation judge](#the-evaluation-judge)). Win rates of the
answers without citations, with 95% intervals and Holm p over the 8 tests
(four criteria, two strategies):

| Strategy | Comprehensiveness | Empowerment | Diversity, directness | Orders agreed (each slot order) | Unique statements (estimated tokens), on / off |
|---|---|---|---|---|---|
| `global` | 0.617 [0.540, 0.695] 95%, Holm p 0.037 | 0.560, n.s. | 0.547 and 0.520, n.s. | 85% and 81% | 1,588 / 1,704 |
| `drift` | 0.642 [0.573, 0.708] 95%, Holm p 0.002 | 0.598 [0.531, 0.665] 95%, Holm p 0.058, n.s. | 0.490 and 0.578, n.s. | 78% and 76% | 796 / 819 |

Judged with the baseline in one slot only, the same answers had given
0.68 (`global`) and 0.75 (`drift`) on comprehensiveness and 0.705 on
`drift` empowerment, the latter significant. The cited answers were less
comprehensive in both readings, although their key points could now be
cited, so the default stands.

On 100 MuSiQue questions, two repeats, where `global` declines most
questions, its citations attributed little: it declined 113 of 199 answers
and cited a file in 86 (citation precision 0.087, recall 0.023), and a
cited key point named about 9.07 files (`lineage_units` caps it at 10): a
point credits every report it came from, often a whole map batch. That
lineage width also caps the file-weighted precision (a point of 9 files
scores one ninth per reference file it names), so the figure
says as much about the width as about the attribution; `global`'s
citations on sensemaking questions are not measured yet. `drift` scored citation
precision 0.571 and recall 0.903. Naming the reports' findings rather than
their communities' members changed neither significantly (`drift`
precision 0.571 against 0.576, `global` 0.087 against 0.071).

**The "cite where possible" instruction.** Those cited `global` and `drift`
answers carried an extra instruction: cite where possible and keep a
relevant point that has no marker. It departs from MS GraphRAG's global
reduce prompt ("Do not include information where the supporting evidence
for it is not provided"). Against the same configuration without it (four
runs, ABBA order, pooled), the answers without it won comprehensiveness
0.48 [0.375, 0.58] 95% (`global`) and 0.49 [0.395, 0.585] 95% (`drift`),
n.s., and `global` directness 0.665 [0.585, 0.74] 95%, Holm p 0.003; `global` citation
precision on MuSiQue was 0.1225 against 0.1222. It bought no
comprehensiveness, so it was removed.

So `global` and `drift` answer without citations by default
(`search.citations.strategies: {global: false, drift: false}`), with the
reduce prompts they had before citations; set
`strategies: {global: true, drift: true}` where attribution matters more
than coverage, which gives them the plain citation instruction of the other
strategies. `auto` follows the strategy it routes to and `iterative` its
base. One judge on one question set, and about one `global` answer in five
of those runs lost a map batch to a reply the parser could not read (12 to
21 of 100). A JSON repair and one retry fixed that since: in a run on
2026-11-08, none of 500 `global` answers (300 sensemaking, 200 MuSiQue)
lost a batch, and the repair read 17 of 1,286 map outputs in one
instrumented pass.

## Iterative settings

The run of 2026-11-07, `iterative` over `mix` and the answer settings, each
against the default:

| Change | Answers | Cost and latency |
|---|---|---|
| `iterative_search.max_rounds: 3` instead of 2 (`iterative`) | `answer_contains` +1.5 [-0.5, +3.75] adjusted, EM +3.0, token F1 +2.7: n.s. | +$0.0024 a query (+4%), +2.7 s [+1.9, +3.5] adjusted; p95 33.3 to 38.1 s |
| `token_manager.max_context_tokens: 12000` instead of 30000, `local` | `answer_contains` -0.25, EM -0.75, token F1 -1.4: n.s. | -$0.0046 a query (-13%); input 16.2k to 13.8k tokens |
| the same, `mix` | `answer_contains` -2.25 [-5.75, +0.75] adjusted, EM -0.75, token F1 -1.55: n.s.; citation precision -3.4 [-6.0, -1.1] adjusted, Holm p 0.007 | -$0.0222 a query (-40%); input 25.3k to 14.1k tokens |
| `aws.bedrock.default_effort: medium` instead of `high`, `local` | `answer_contains` -0.5 [-3.75, +2.75] adjusted, n.s.; token F1 -3.6 [-6.9, -0.25] 95%, Holm p 0.50, n.s. | -151 output tokens, -$0.0015 a query, -2.2 s |
| the same, `mix` | `answer_contains` -1.0 n.s.; token F1 -4.4 [-8.2, -0.6] 95%, Holm p 0.41, n.s. | -155 output tokens, -$0.0017 a query, -1.8 s |

At that commit an arm read the base's query extractions whatever its
effort, so the effort rows ran as separate invocations (base, medium,
medium, base; one repeat each, pooled), each extracting its own; their
latency carries the 0.6-1.0 s between invocations noted above. (An arm now
shares the base's extractions only when its overlay leaves the extraction
settings alone, the default tier's effort included, so an effort arm
extracts its own.) In the `max_rounds` comparison, with 2 rounds 207 of
the 400 answers (52%) stopped at `max_rounds` and 170 as `sufficient`;
with 3, 177 (44%) and 197. So more rounds
buy little because the controller rarely judges the context sufficient,
not because the rounds run out. Every default stays. 12000 is a cost lever
for `mix` with a possible small loss, and `medium` a latency lever with a
possible small token-F1 cost.

## Reranking

The run of 2026-11-07, the rerank off (`providers.rerank.kind: none`)
against on:

| Strategy | Answers | Retrieval | Cost and latency |
|---|---|---|---|
| `local` | `answer_contains` -0.75 [-3.25, +1.5] adjusted, EM -1.5, token F1 -0.8: n.s. | recall@5 -7.9 [-14.1, -1.8] adjusted, MRR -10.8 | -$0.0020 a query, -0.53 s |
| `mix` | `answer_contains` -2.75 [-6.5, +0.5] adjusted, n.s.; EM and token F1 n.s. | recall@5 -25.4 [-32.2, -18.8] adjusted, MRR -8.3 | -$0.0028 a query, -0.58 s |
| `ppr` | `answer_contains` -1.5, token F1 -4.2, both n.s. after a Holm correction | recall@5 -9.7, MRR -5.6 | -$0.0023 a query (-15%), -0.28 s |

Every retrieval loss is significant after a Holm correction and every
answer estimate is negative, for about $0.002 and half a second a query.
The rerank stays on.

## Judge effort

`evaluation.judge_effort` sets the effort of the span extraction and the
pairwise judge. The run of 2026-11-07 re-scored the same saved answers
(`run-eval --from-results ... --rescore`, no answer call):

- **Span extraction.** `high` against `low` on 1,600 answers: the spans
  were identical in 84-86% (two `low` passes: 93-95%), and EM fell by 1.0
  to 1.75 points in each of four runs (n.s. each), token F1 by 1.1 to 1.8.
  It cost $0.18 against $0.10.
- **Pairwise judge** (differential, that run's `global` and `drift`
  citation pairs): position consistency rose from 0.82 to 0.86 (`global`)
  and 0.78 to 0.85 (`drift`), and the verdicts grew stronger in the same
  direction (comprehensiveness 0.82 for both). But 10 of the 100 `global`
  pairs failed, against 2 at `low`: the fast model's overlap split ran out
  of output tokens while it reasoned. About 1.7x the judge cost.

So `low` stays the default: a higher effort does not extract better spans,
and the pairwise judge at `high` needs a larger output limit for the
overlap split.

## Reproducing a measurement

The same tooling runs any of these comparisons on your corpus or on
MuSiQue ([Benchmarks](../guides/benchmarks.md)):

```bash
# The benchmark corpus and dataset (downloaded at run time, never committed).
prepare-benchmark musique --output-directory bench/musique --limit 200 --seed 0
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# Two settings on the same index: run each config, then compare the saved runs.
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy mix --repeats 2 --output-directory runs/base --config-path musique.yaml
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy mix --repeats 2 --output-directory runs/vector --config-path vector.yaml \
  --baseline-results runs/base

# Pairwise LLM judge across strategies, on your own question set.
run-eval --data-path my_questions.json --suffix musique --search-strategy global,drift \
  --repeats 2 --pairwise-judge --config-path musique.yaml
```

`--baseline-results` pairs each strategy with the earlier run of the same
strategy, `--from-results` rebuilds or re-judges a comparison from saved
answers without a model call, and `--fail-if` turns a delta into a CI gate
([Rebuilding, re-judging and gating a comparison](../guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison)).
Run at least two repeats, and read a difference only when its interval
excludes zero.

## Limits of the evidence

- **One corpus, one model family.** Every answer measurement ran on a
  single corpus, MuSiQue (English, multi-hop Wikipedia text, or a slice
  of it); the answers came from Claude models on Bedrock. A
  default that was n.s. here may matter on your corpus: measure it there.
- **LLM judges.** Pairwise judges favour longer answers and, as measured
  above, their own model's answers and, judged in one slot order, the
  second answer. Read win rates with the position
  consistency, and prefer a judge that is not the answer model.
- **Embedded stores.** Latencies were measured on the SQLite stores, not on
  Neptune and OpenSearch, except the 90 s and 180 s timeouts, which come
  from real-AWS runs.
- **A partial graph in the first runs.** In the index the first runs
  measured (package at `f04ee50d`: answer style, the global evidence cap,
  the judge, gleaning's graph recall, LightRAG chunk picking and MMR, the
  retrieval switches and local report selection), 138 of 3,318 text units
  (4.2%) had no extraction (a model output the parser could not read).
  Every arm read the same graph, so those comparisons are paired, but
  their absolute scores are lower than a complete graph would give. A
  later run extracted those 138 units into a copy (0 units left without
  one, $13.93), and the gleaning answers, the PPR ranking, the `ppr`
  strategy, `iterative` and the summary tree were measured on that
  repaired copy.
- **Costs are upper bounds** when the prompt cache hit, and they use the
  dated list prices above.
- **Earlier answer prompts and run order.** The runs before 2026-11-07
  predate three changes to what a run with today's defaults does. Answers
  of every strategy but `global` and `drift` now cite their context with
  `[n]` markers (`search.citations`, since 2026-11-07): the measurements before
  2026-11-07 were written without them, from a context without numbered
  sections. The comparison in [Citations](#citations) found no difference
  on `local` and `mix` answers; `global` and `drift` answer without
  citations by default, as those measurements did. Community reports now follow an evidence-only prompt with data
  references, and split communities stay parents of their pieces (both since
  2026-11-07):
  the `global` and `drift` measurements read reports of the earlier prompt
  and hierarchy. And `run-eval` now answers a comparison interleaved
  (since 2026-11-07): these arms ran one after another (see *Latency* above), so a
  latency delta here carries the drift between them, which an interleaved
  run spreads over every arm.
- **Unadjusted intervals.** The intervals of the runs before 2026-11-07
  are per metric, 95% each, with no correction for the several metrics and
  arms one run reads (the Holm adjustment `run-eval` reports came on
  2026-11-07): a single
  "significant" metric among many is weaker evidence than its interval
  alone suggests. Where a section says a Holm correction, it applied one
  by hand.
- **Settings no run compared.** `search.local_search.type_quota` (only its
  entity and relationship floors follow MS GraphRAG),
  `search.local_search.entity_candidates_per_focus`,
  `search.hybrid.normalization_window` and
  `search.iterative_search.max_bridge_entities` keep the values they were
  written with. The type quota decides what a short-chunk corpus's context
  holds (its type budgets never bound on MuSiQue), so it is the first to
  measure there. These depart from upstream with no run behind the
  departure:
  - `graph.community_detection.max_cluster_size` 20 and
    `max_report_context_tokens` 4000, against MS GraphRAG's 10 and 8000
    (kept for the report cost:
    [Design §4.3](../design/ingestion.md#43-communities-stages-10-11));
  - `search.drift_search.k_followups` 10 and `n_depth` 2, against MS
    GraphRAG's 20 and 3 (at most 27 model calls a query, against upstream's
    67);
  - `search.hybrid.lexical_weight` and `vector_weight`, 0.5 each, an equal
    split no run compared with another.
