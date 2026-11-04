# Benchmarks (MuSiQue)

`prepare-benchmark musique` turns
[MuSiQue](https://github.com/StonyBrookNLP/musique) (answerable, multi-hop,
CC BY 4.0; Trivedi et al., TACL 2022) into a corpus and a `run-eval` dataset.
Every flag with its default: [CLI Reference](../reference/cli.md#prepare-benchmark).
Running and reading the evaluation itself: [Evaluation](./evaluate.md).
The package's own measured runs on MuSiQue, and the commands that reproduce
them: [Defaults and evidence](../concepts/defaults-and-evidence.md#how-the-measurements-were-run).

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
