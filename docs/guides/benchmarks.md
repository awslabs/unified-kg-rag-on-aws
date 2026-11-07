# Benchmarks (MuSiQue)

`prepare-benchmark musique` turns
[MuSiQue](https://github.com/StonyBrookNLP/musique) (multi-hop, CC BY 4.0;
Trivedi et al., TACL 2022) into a corpus and a `run-eval` dataset, optionally
with [unanswerable questions](#unanswerable-questions).
Every flag with its default: [CLI Reference](../reference/cli.md#prepare-benchmark).
Running and reading the evaluation itself: [Evaluation](./evaluate.md).
Questions and source-cited assertions generated from your own corpus
(`prepare-benchmark assertions`): [Evaluation: Assertions, length control and a
closed-book bound](./evaluate.md#assertions-length-control-and-a-closed-book-bound).
The package's own measured runs on MuSiQue, and the commands that reproduce
them: [Defaults and evidence](../concepts/defaults-and-evidence.md#how-the-measurements-were-run).
A generated benchmark of graph-structure questions for the `structured`
strategy (`prepare-benchmark structural`): [below](#structural-benchmark).

## Prepare the corpus and the dataset

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

## Run it

```bash
# 1. 100 questions (about 1,900 paragraphs), sampled with seed 0.
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. Ingest the paragraphs under their own suffix.
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. Paired comparison of GraphRAG and LightRAG strategies, EM / F1 on the short answer span.
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
  enabled_evaluators: [answer_match, retrieval]   # judge-free; add ragas if wanted
```

## Unanswerable questions

`--unanswerable N` adds N questions the corpus cannot answer, to measure
whether the chain declines them
([Evaluation: unanswerable questions](./evaluate.md#unanswerable-questions-and-declines)).
They come from MuSiQue-Full, which pairs every answerable question with an
unanswerable twin: the same question with one or more supporting paragraphs
replaced by distractors. The full dev split is downloaded the same way
(pinned mirror revision, SHA-256 checked), next to the answerable one;
`--unanswerable-data-path` takes a local `musique_full_v1.0_*.jsonl` instead
(an answerable-only file there is a usage error).

Pooled naively, many of them are answerable after all, so each is checked
against the corpus being written:

- **Its twin.** An unanswerable question whose answerable twin is in the
  sample is left out: the twin's paragraphs give the removed one back.
- **Other questions.** MuSiQue reuses paragraphs across questions. A question
  is left out when the exact text (whitespace-normalized) of a removed
  paragraph is anywhere in the pooled corpus, or when one of its own
  paragraphs is a removed paragraph of a question already kept. On the dev
  split this drops 34-50% of the candidates for 100-200 answerable plus
  100-200 unanswerable questions.

The output reports the counts (`unanswerable_filter`: `requested`, `kept`,
`examined`, `excluded_twin`, `excluded_contaminated`, `excluded_no_twin`),
and fails with exit 2 when fewer than N pass (ask for fewer, or fewer
answerable questions). Unanswerable rows have the query id
`<id>__unanswerable`, `metadata.answerable: false`, no `answer` or
`reference_sources`, `metadata.benchmark: musique-full-unanswerable` and the
twin's answer (`metadata.twin_answer`, `twin_answer_aliases`): an answer that
names it came from a near-duplicate paragraph the exact-text filter cannot
see, or from the model's own knowledge. The answerable questions still come
from the answerable file (the default download or `--data-path`), labelled
`musique`: the same sample the seed draws without `--unanswerable`, so the
answerable scores compare with a run without it. A full file given as
`--data-path` without `--unanswerable-data-path` supplies both kinds; its
answerable half is labelled `musique-full-answerable` (its paragraphs and
answer aliases differ from MuSiQue-Ans) and is another sample.

`--hops` picks the hop counts: `--hops 3,4` samples both kinds from 3- and
4-hop questions only; `--hops 2=50,3=50,4=100` draws exactly that many
answerable questions per hop count (instead of `--limit`), to oversample the
longer chains. Sampling is deterministic per `--seed`.

```bash
# 150 answerable (oversampling 3-4 hops) + 150 unanswerable questions.
prepare-benchmark musique --output-directory bench/unans \
  --hops 2=50,3=50,4=50 --unanswerable 150

run-ingestion --source-directory bench/unans/corpus --config-path musique.yaml
run-eval --data-path bench/unans/eval_data.json --suffix musique \
  --search-strategy local,mix --config-path musique-mark.yaml
```

with `musique-mark.yaml` = `musique.yaml` plus `search.abstention.mode: mark`.
Ingestion cost is about 19 paragraphs a question, as for answerable ones.

## Structural benchmark

`prepare-benchmark structural` generates, from `--seed`, a made-up
procurement world and the questions the
[`structured` strategy](./structured-queries.md) is for: counts by
relationship and by a seed property, the top supplier and the top three, the
vendor between a buyer and a product, the buyer two vendors share, yes/no
supply, the largest and smallest number of employees, and text-only
controls (each vendor's chief executive, stated only in its document) that
measure what misrouting costs. Nothing is downloaded or committed.

```bash
prepare-benchmark structural --output-directory bench/structural --size medium --seed 0
run-ingestion --config-path bench/structural/structural-config.yaml \
  --source-directory bench/structural/corpus --suffix structural
run-eval --config-path bench/structural/structural-config.yaml \
  --data-path bench/structural/eval_data.json --suffix structural \
  --search-strategy structured,local,mix --repeats 3
```

It writes `seed/structural.jsonl` (a [JSONL seed graph](../reference/seed-graph-formats.md#jsonl)
of every node with its `country`, `founded` and `employees` properties and
every `SUPPLIES` and `MAKES` edge), `corpus/` (one document per vendor and
buyer, stating part of those edges, the products and the chief executive),
`structural-config.yaml` (the extraction schema and the seed graph entry)
and `eval_data.json`. `--size small` writes 8 vendors and up to 4 questions
a template, `medium` 40 vendors and up to 25 (about 200 questions; templates
whose answer would tie or be ambiguous in a world are left out for it), and
`large` 400 vendors and 300 buyers, whose seed-filtered counts scan more
matches than `search.structured.max_scan_rows` (the truncated "at least N"
case; a few minutes to generate). Every size also asks about vendors no
data names (`absent_entity`: the answer is no, or 0) and asks two templates
in Korean and Japanese (`multilingual_ko`, `multilingual_ja`, with `예` /
`아니요` and `はい` / `いいえ` as yes/no; `metadata.language`).
Each graph question's gold answer is computed by the reference executor over
the generated graph from the template's plan (`metadata.gold_plan`); a
number has its digit and word forms as `answer_aliases` (the Korean and
Japanese counts `3곳` / `세 곳` and `3社` / `三社` too), and the top three
every order of the three names. Each item also says how `answer_match`
reads it: a count or maximum (`metadata.answer_type: number`) by the first
number the answer states, so "Three vendors supply it. Two of them are
local." is 3, not a match of the gold 2; the top three
(`answer_type: set`, `metadata.answer_items`) by all three names in any
order or layout, a numbered list too; and an absent-vendor question
(`metadata.decline_is_correct`) counts a declined answer as right
([Evaluation output: answer_match](../reference/evaluation-output.md#answer_match)).
`category` is the template, so the
per-category scores break the comparison down by question shape; the
control questions carry their document as `reference_sources`.

Score with `answer_match` (containment is the headline; EM and F1 with
`evaluation.answer_span: extracted`). The run summary's
`structured_statistics` reports the structured arm's plan validity, repair
and fallback rates by reason, and truncation
([Evaluation output](../reference/evaluation-output.md#run-summary)); the
usage gives its calls and tokens per query. Gold answers are what the world
states: an extraction that merges or misses an entity, or a seed edge that
documents do not state, shows up as a lower score for every strategy.

## Reading the scores

MuSiQue (like HotpotQA and SQuAD) scores the short answer a reader model
predicts against the gold answer and its aliases, with EM and token F1 under
the official normalization. A RAG answer is a sentence and its support, so
`answer_span: extracted` (one fast-model call per answer) has a model copy
the span out of it first; the numbers are then close to, but not the same
as, a paper's, whose readers were trained to output spans. The default
`first_line` costs nothing but scores the one-sentence answer, so its EM is
far below a paper's. The GraphRAG and LightRAG papers report no EM / F1;
they use the pairwise LLM judge (`--pairwise-judge`). How the span and the
normalization are scored:
[Evaluation output: `answer_match`](../reference/evaluation-output.md#answer_match).

Read `answer_contains` / `token_f1` and `retrieval` (per `category`, and
`deltas_by_category` in a comparison).
Ingestion cost grows with the paragraph count (about 19 per question); the
measured cost of a 200-question index is in
[Defaults and evidence](../concepts/defaults-and-evidence.md#how-the-measurements-were-run). Cite
the paper when you publish results.
