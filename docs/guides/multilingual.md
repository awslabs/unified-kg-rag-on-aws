# Multilingual corpora

> 🇰🇷 한국어판: [multilingual.ko.md](./multilingual.ko.md)

A corpus in Korean or another language, or in several, needs its language
set for translation, for the search analyzers and for chunk sizes. How the
pieces work inside: [Design §8.3](../design.md#83-multilingual-processing).

## Translation

Set the corpus's predominant language and the target you want to index in:

```yaml
processing:
  translation:
    enabled: true
    source_language: "ko"
    target_language: "en"
    additional_target_languages: ["ja"]   # index additional languages too
```

With `source_language == target_language` and no additional targets, the
translation stage is skipped at no cost, even with `enabled: true`. With
`enabled: false` the corpus is indexed as written, and queries are processed
in `source_language` (no query translation) unless a query asks for a
language with `target_language`.

**One index language.** The corpus is indexed in `target_language` when
translation runs and in `source_language` otherwise. Every component follows
it: extraction, gleaning, description summaries, community reports and the
summary tree are written in it, the text units are embedded in it, the
entity, relationship, claim and report indexes are analyzed for it, and
DRIFT and global search write their follow-up queries and map points in it
(or in a query's requested language). With `enabled: false`,
`target_language` therefore has no effect but the translated-text fields'
mapping.

**Failures.** A text unit whose translation fails is extracted in its
source language; a failed call for a whole target language fails the stage
unless `processing.ignore_errors: true`. An incremental run records the
documents of the `failed_units` as failed, so the next run retries them.

## Analyzers

- **Analyzers** come from `indexing.opensearch.language_analyzers`. Every
  supported language has one by default: `de: german`, `en: english`,
  `es: spanish`, `fr: french`, `it: italian`, `pt: portuguese` and
  `ru: russian` (OpenSearch's language analyzers: stop words and stemming),
  `ja: cjk` and `zh: cjk` (overlapping character pairs, so "首都" matches
  inside "日本の首都は"; no plugin), and `ko: nori`. A language without an
  entry uses `default_analyzer`. Chunk `text` is analyzed for
  `source_language`, and the graph indexes for the index language, so set it
  to the corpus language even without translation: with `source_language:
  en` a Korean corpus keeps words with their particles ("홍길동으로부터") as
  single tokens. Changing it changes the mapping: an incremental run stops
  (`was built for other language settings`) until one `indexing.reset: true`
  run.
- **Unicode.** Text and queries are NFC-normalized, so decomposed Hangul (NFD)
  matches its composed form.
- **The analyzer must exist on the cluster**, or the indexing stage fails
  with `the cluster has no analyzer '<name>'`. `nori` comes from the
  `analysis-nori` plugin: the
  [local OpenSearch container](./local-mode.md#local-stores-in-containers) has it, an Amazon
  OpenSearch Service domain only once its Nori package is associated (a
  blue/green deployment). The
  [CDK deployment](../../iac/README.md) therefore maps `ko` to the built-in `cjk`
  analyzer (character bigrams); `-c opensearch_nori_package=true`
  associates the package and maps `ko` to `nori`
  ([Korean with Nori](../operations.md#korean-with-nori)).

## Chunk sizes

The chunk sizes count characters by default, and a Hangul, Han or Kana
character is about one token, so an 8,000-character Korean chunk is ~6,000
tokens: over the 4,096-token document limit of Cohere Rerank 3.5, which then
never sees its tail. Halve the sizes, or count them in tokens with
`processing.chunking.length_unit: tokens` (a local estimate, no model call)
and give every size in tokens:

```yaml
processing:
  chunking:
    length_unit: tokens
    min_chunk_size: 250
    max_chunk_size: 2000
    chunk_overlap: 125
    fallback_chunk_size: 1200
    pre_chunk_size: 12500
    pre_chunk_overlap: 125
```

Switching the unit re-chunks the corpus and re-extracts its documents (an
incremental run re-extracts every document of the scope).

## Evaluating Korean and CJK answers

The `answer_match` evaluator tolerates Korean particles and units, ignores
spacing next to CJK letters and reads Korean, Chinese and Japanese yes/no
answers; under the default normalization token F1 counts a Han, Hangul or
Kana letter as a token (character F1), while exact match still under-counts
an answer with a particle. The rules:
[`answer_match`](../reference/evaluation-output.md#answer_match).
