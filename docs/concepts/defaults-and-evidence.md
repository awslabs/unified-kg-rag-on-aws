# Defaults and evidence

Several defaults were set, or kept, from real-model measurements rather than
by analogy with the papers. This page is the one place that holds those
numbers: how they were measured, what each one showed, what it cost, and how
to run the same comparison on your own corpus. The pages that explain a
setting (the [guides](../index.md#page-map), [Operator Runbook](../operations.md),
[Design Doc](../design.md)) link here for the evidence.

Every interval below is a 95% paired bootstrap interval over questions, and
"n.s." means the interval includes zero. A difference that is not
significant kept the existing default.

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
  both answer orders agree, judged by Opus 5.5, Sonnet 5.5 and GPT-5.5.
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
| `search.answer_prompt_style` | `{global: comprehensive}` | Comprehensive global answers beat concise ones on comprehensiveness, diversity and empowerment at 0.95-0.99 with all three judges, +$0.016 per query; they lose directness. DRIFT stays concise for its latency ([Answer style](#answer-style)) |
| `search.global_search.evidence_tokens` | 4000 | -41% cost per global query against no cap, no consistent quality difference, sources kept ([Global evidence cap](#global-evidence-cap)) |
| `models.roles.evaluation` | Opus 5.5 on Bedrock | Against a GPT-5.5 judge, a Sonnet 5.5 judge shifted directness by -0.27 [-0.38, -0.16] when it judged its own answers ([The evaluation judge](#the-evaluation-judge)) |
| `processing.gleaning.max_rounds` | 3 | Within one ingestion, round 1 adds +6.0 points of bridge relationships and rounds 2-3 +2.0; decomposition answers found as graph entities +1.3 points [+0.4, +2.3] over extraction alone, +0.4 [0.0, +1.0] from rounds 2-3. Gleaning is 47% of the ingestion cost ([Gleaning rounds](#gleaning-rounds)) |
| `search.lightrag_search.kg_chunk_pick_method` | `weight` | `vector` n.s. on `mix` and `hybrid`, +1.5 s latency ([LightRAG chunk picking and MMR diversity](#lightrag-chunk-picking-and-mmr-diversity)) |
| `search.fusion.diversity_lambda` | 0.5 | 0.7 and 1.0 n.s. on `mix` ([LightRAG chunk picking and MMR diversity](#lightrag-chunk-picking-and-mmr-diversity)) |
| `search.local_search.report_selection` | `similarity` | `membership` (MS GraphRAG local's membership, over leaf communities rather than upstream's `community_level` cut) scored EM -4.0 and token F1 -4.1 points, both significant ([Local report selection](#local-report-selection)) |
| `search.local_search.entity_frequency_threshold` | 20 | Without the cut, recall@5 -1.75 points [-3.12, -0.50] and supporting paragraphs in the context -2.42 [-3.96, -1.04], both significant, +3.08 s per answer ([Retrieval switches](#retrieval-switches)) |
| `search.auto_routable_strategies` | `local`, `mix`, `global`, `iterative` | `iterative` over `mix` against `mix` on multi-hop questions: `answer_contains` +12.5 points [+7.5, +18.0], token F1 +5.2 [+0.2, +10.1], last hop in the context +21.5; p95 11.1 s to 27.4 s, +$0.002 a query. `decompose` was no better than the loop, so it stays off ([Iterative follow-up rounds](#iterative-follow-up-rounds)) |
| `search.model_timeout_seconds`, `search.query_timeout_seconds` | 90 s, 180 s | From the query times of the package's real-AWS evaluation runs ([Query deadlines and model timeouts](#query-deadlines-and-model-timeouts)) |

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

Pairwise LLM judges favour the longer answer, so these win rates measure
coverage more than quality for every reader. The directness loss is the
cost, and `search.answer_prompt_style: {global: concise}` keeps the short
answers ([Answer style](./retrieval.md#answer-style)).

## Global evidence cap

`search.global_search.evidence_tokens` caps the report text a global answer
reads beside its ranked key points. On the 50 sensemaking questions:

| Cap | Cost per global query | Quality against no cap | Sources |
|---|---|---|---|
| `null` (no cap) | $0.1016 | — | kept |
| `4000` (default) | $0.0597 (-41%) | No criterion consistently different. Position consistency 0.43-0.57: the judges could not tell the answers apart. Only Opus 5.5's empowerment, 0.41 [0.33, 0.48], left 0.5, and Sonnet 5.5 ([0.34, 0.50]) and GPT-5.5 ([0.37, 0.55]) did not replicate it | kept |
| `0` (key points only) | $0.0514 (-49%) | All criteria n.s. (position consistency 0.44-0.64) | none |

The cap is a cost change, not a quality change: 4000 keeps the sources the
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

## Gleaning rounds

Each item stores the time it was written, so one ingestion of the shared
index (3 rounds) separates what extraction and each gleaning round added:

- **Graph recall.** The 522 decomposition answers of the 200 questions
  (bridge and final answers) found as graph entities, by exact entity name:
  83.5% after extraction, 84.5%, 84.7% and 84.9% after rounds 1-3. Gleaning
  adds +1.3 points [+0.4, +2.3], rounds 2-3 +0.4 [0.0, +1.0]; every gain is
  a last-hop answer (68.5% to 72.0%), none a bridge answer. A looser name
  match raises every level by about 4 points and leaves the deltas unchanged.
- **Relationships.** On the supporting paragraphs, gleaning adds 3,761
  relationships to extraction's 4,922 (+76%) and corrects 19,936. A bridge
  (the entity linking one hop's paragraph to the next) has an edge in the
  next paragraph for 183 of 298 bridges after extraction, 201 after round 1
  (+6.0 points) and 207 after round 3 (+2.0 from rounds 2-3).
- **Ingestion cost.** Gleaning is $173.6 of the $371.64 build (47%);
  rounds 2-3 are about $111 of it, by call count.

The 109-document slice ingested once with `max_rounds: 3` and once with `1`
cannot measure rounds: its two extractions differ before any gleaning (65
against 59 of 82 answers found), because the runs lost different units to
failed extractions and named some entities differently ("Rio Linda,
Sacramento, CA" as one entity). Of the 7 answers only the 3-round run found,
at most one came from gleaning.

MuSiQue paragraphs are about 110 tokens; the GraphRAG paper motivates
gleaning with 600-2,400-token chunks, which this index does not test. 3
rounds stays the default until a measurement of answers decides it;
`max_rounds: 1` keeps most of the relationship gain at about a third of
the gleaning cost ([Cost levers](../guides/ingest.md#cost-levers)).

## LightRAG chunk picking and MMR diversity

On the 200 MuSiQue questions:

| Change | `mix` | `hybrid` | Latency and cost |
|---|---|---|---|
| `kg_chunk_pick_method: vector` (upstream LightRAG's default) instead of `weight` | `answer_contains` -1.5 points [-4.5, +1.25], EM and F1 n.s. | `answer_contains` -1.25 [-4.5, +2.0], MRR +1.5 [-0.01, +3.2], n.s. | +1.5 s per query (significant), cost unchanged |
| `diversity_lambda: 0.7` instead of 0.5 | n.s. on every metric | — | +0.57 s |
| `diversity_lambda: 1.0` instead of 0.5 | `answer_contains` -2.25 [-5.25, +0.5] n.s.; MRR +0.9 [+0.05, +2.15], significant but small | — | -$0.0009 per query |

No change was significantly better on the answers, so both defaults stayed
([Search strategies](./retrieval.md#search-strategies--when-to-use-which)).

## Retrieval switches

A later run (2026-11-05) on the same index, models and 200 MuSiQue
questions, two repeats each, compared five retrieval switches against the
defaults. "Supporting paragraphs" is the share of each question's MuSiQue
supporting paragraphs among the text chunks of the answer context, and
latencies are the answer call's alone.

| Change | Answers | Retrieval | Latency and cost |
|---|---|---|---|
| `local`: no `entity_frequency_threshold` cut instead of 20 | `answer_contains` -1.5 points [-3.0, 0.0], borderline; EM and F1 n.s. | recall@5 -1.75 [-3.12, -0.50] and supporting paragraphs -2.42 [-3.96, -1.04], both significant | +3.08 s (significant), +370 input tokens |
| LightRAG: no per-side cut of the knowledge-graph chunks before the rerank | n.s. on `mix` and `hybrid` | `hybrid` supporting paragraphs +1.87 [+0.12, +3.71], significant; `mix` n.s. | `hybrid` +0.82 s (significant), +195 input tokens |
| LightRAG: chunks picked from upstream's merged, token-truncated entity and relationship lists | n.s. on `mix` and `hybrid` | n.s. | +0.48 s `hybrid`, +0.32 s `mix` (both significant) |
| Relationships embedded as upstream LightRAG does (source, target and description) instead of the description (`hybrid`, one repeat) | n.s. | n.s. | +0.24 s; a re-embedding of every relationship |
| `search.token_manager.type_budgets.relationship` 0.27 instead of 0.13 | Not measurable: the shares never bound on MuSiQue, so both arms read the same context | — | — |

None was better on the answers, so every default stayed: the frequency cut
for its retrieval and latency, the others for their latency or re-embedding
cost. The context shares need a corpus whose contexts fill their budget to
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
([Design §6.1](../design.md#61-graphrag-methodology-adapterssearch_strategies)).

## Iterative follow-up rounds

`iterative` (§6.3 of the [Design Doc](../design.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in))
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
| `decompose: true` over `mix` against the controller loop | token F1 -0.6 n.s., `answer_contains` -2.5 n.s. | recall@5 +2.7 n.s. | +$0.0009 a query, latency the same |

The gain comes from the questions whose base context lacked a supporting
paragraph (89 of 200: token F1 +13.6, `answer_contains` +29.2); where the
base context already held every one, the follow-up did not hurt (token F1
-1.5, n.s.). The controller stopped after round 0 on 57 questions, after
one follow-up round on 48 and after two on 95, at 2.19 controller calls a
question; the answers were no longer.

So `auto` may route to `iterative` (`search.auto_routable_strategies`),
whose router entry keeps it to questions that chain facts through an
entity they do not name, but no default selects it: it adds about 10 s to
the median query. `decompose` stays off, costing more for no gain over
the loop.

Not measured: MuSiQue is the only question set, all multi-hop, so whether
the router leaves single-hop and sensemaking questions to the other
strategies, and what a misroute costs there, is untested; route a sample
of your own questions with `run-eval --search-strategy auto` before
relying on it. In 62 of the 238 controller rounds over `mix` the verdict
named no new bridge entity and mix reran without keywords; such a round
now runs only its text searches (Design §6.3), which this run predates.
To keep `auto` off it, list the strategies without `iterative`:

```yaml
search:
  auto_routable_strategies: [local, mix, global]
```

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

- **One corpus, one model family.** MuSiQue is English, multi-hop
  Wikipedia text; the answers came from Claude models on Bedrock. A
  default that was n.s. here may matter on your corpus: measure it there.
- **LLM judges.** Pairwise judges favour longer answers and, as measured
  above, their own model's answers. Read win rates with the position
  consistency, and prefer a judge that is not the answer model.
- **Embedded stores.** Latencies were measured on the SQLite stores, not on
  Neptune and OpenSearch, except the 90 s and 180 s timeouts, which come
  from real-AWS runs.
- **A partial graph.** In the measured index, 138 of 3,318 text units (4.2%)
  had no extraction (a model output the parser could not read); every arm
  read the same graph, so the comparisons are paired, but absolute scores
  are lower than a complete graph would give.
- **Costs are upper bounds** when the prompt cache hit, and they use the
  dated list prices above.
