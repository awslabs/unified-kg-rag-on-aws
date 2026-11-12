# Multilingual corpora

> 🇰🇷 한국어판: [multilingual.ko.md](./multilingual.ko.md)

A corpus in Korean or another language, or in several, needs its language
set for translation, for the search analyzers and for chunk sizes. How the
pieces work inside: [Design §8.3](../design/aws-integration.md#83-multilingual-processing).

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
`enabled: false` the corpus is indexed as written.

**Query translation.** A query is translated into the index language (below),
or into the language it requests with `target_language`, exactly when it is
written in another one: a Korean question on an English index is searched in
English, and an English question after a ko→en ingest costs no translator
call. The query's language is read from its script and, among Latin-script
languages, its function words; a query they do not tell apart counts as
written in the index language and is not translated.

**One index language.** The corpus is indexed in `target_language` when
translation runs and in `source_language` otherwise. Every component follows
it: extraction, gleaning, description summaries, community reports and the
summary tree are written in it, the text units are embedded in it, the
entity, relationship, claim and report indexes are analyzed for it, and a
query's entities and keywords, DRIFT's and iterative's follow-up queries and
global search's map points are written in it. A language a query requests
(`RAGInput.target_language`) is the query translation's only: the text units
carry a `translated_text_<language>` field per additional target, which its
lexical search reads, while the graph indexes hold the index language alone.
The answer is written in the language of the query. With `enabled: false`,
`target_language` therefore has no effect but the translated-text fields'
mapping.

**Failures.** A text unit whose translation fails is extracted in its
source language; a failed call for a whole target language fails the stage
unless `processing.ignore_errors: true`. An incremental run records the
documents of the `failed_units` as failed, so the next run retries them.

**Stored translations.** A text unit's translations are stored by its text
under the translation settings, every language in one record, so a reset or
an edited document translates only new text. Changing
`additional_target_languages` (or the source language, the translation
model or its prompt) keys every record anew: the next build translates every
unit into every listed language again, and a unit whose target-language
translation then reads otherwise is extracted again. The records of text no
unit holds any more are deleted by the next full build
(`indexing.reset: true`); incremental runs leave them.

## Analyzers

- **Analyzers** come from `indexing.opensearch.language_analyzers`. Every
  language with a profile has one by default: `ar: arabic`, `de: german`,
  `en: english`, `es: spanish`, `fr: french`, `hi: hindi`, `it: italian`,
  `pt: portuguese`, `ru: russian` and `th: thai` (OpenSearch's language
  analyzers: stop words and stemming, dictionary word segmentation for
  Thai), `ja: cjk` and `zh: cjk` (overlapping character pairs, so "首都"
  matches inside "日本の首都は"; no plugin), `ko: nori`, and `he: standard`
  and `vi: standard` (OpenSearch has no built-in Hebrew or Vietnamese
  analyzer; a plugin's can be named instead). A language without an entry
  uses `default_analyzer`. The embedded engine of `backend: local` stems the
  stemming languages with Snowball (from the `[local]` extra) and splits a
  Thai run into character pairs; it keeps no stop words. Chunk `text` is analyzed for
  `source_language`, and the graph indexes for the index language, so set it
  to the corpus language even without translation: with `source_language:
  en` a Korean corpus keeps words with their particles ("홍길동으로부터") as
  single tokens. Changing it changes the mapping: an incremental run stops
  (`was built for other language settings`) until one `indexing.reset: true`
  run. The loading stage samples the corpus (the first 2,000 characters of
  its first 20 documents) and logs a WARNING naming
  `processing.translation.source_language` when the sample's script
  disagrees with it: mostly Hangul, Kana or Han letters under a language of
  another script (it suggests `ko`, `ja` or `zh`), or few of them under
  `ko`, `ja` or `zh` (not when the sample still reads as that language: a
  Korean document that is mostly code). It cannot tell two languages of one script apart
  (English from German), and the run goes on.
- **Unicode.** Text and queries are NFC-normalized, so decomposed Hangul (NFD)
  matches its composed form.
- **Other languages.** `source_language` and `target_language` take the
  codes with a language profile (`ar`, `de`, `en`, `es`, `fr`, `he`, `hi`,
  `it`, `ja`, `ko`, `pt`, `ru`, `th`, `vi`, `zh`) or any other BCP-47
  language code (`sw`, `fa`; a region is dropped, `pt-BR` is `pt`). A code
  without a profile logs one WARNING and gets the defaults: OpenSearch's
  built-in analyzer of the language where it has one (`nl`: `dutch`, `sv`:
  `swedish`, `tr`: `turkish`, ...; the default `language_analyzers` lists
  them), else the `default_analyzer`, English declines, and prompts that
  name the language by its code. `nn` (Nynorsk) gets `norwegian`, whose
  stemmer and stop words are Bokmål's. An index built before the default
  named its language keeps the `default_analyzer` its fields were built
  with (one WARNING per run) until an `indexing.reset: true` run, also
  with a `language_analyzers` you set (the CDK stack sets `ko`) that gives
  the language the default's analyzer (each field by the language it
  holds); a language you give another
  analyzer is followed (a field built otherwise stops the run). A built-in stemmer can miss a root: on an
  Amazon OpenSearch Service domain (OpenSearch 2.19) `indonesian` reduced
  "pemasok" to "asok" and "mengirimkan" to "irim" while the roots "pasok"
  and "kirim" stayed, so a query of the root found no document (Lucene's
  Indonesian stemmer restores no nasal prefix); the vector and graph
  searches still match. A profile
  (`unified_kg_rag/domain/languages/profiles/`) is one module: its name,
  script, analyzer, decline and the words answers in it are read by.
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
spacing next to CJK letters and reads the yes/no answers of every profiled
language but Thai ("Não, ele não fez" answers no, "não há dúvida" does not); under the default normalization token F1 counts a Han, Hangul or
Kana letter as a token (character F1), while exact match still under-counts
an answer with a particle. Thai, Lao, Myanmar and Khmer, which write no
spaces between words either, are matched and token-scored the same way, and
a Hebrew or Arabic prefixed particle before a gold word ("בירושלים",
"وبيروت") is tolerated as a Korean particle after one is. An answer in German, Spanish, French, Italian or
Portuguese (the question's language) also drops that language's articles
and reads its number words and its words and phrases for none ("aucun",
"no hay", "il n'y a pas de") in place of the English ones (Spanish "no" is
no zero). A `number` item asks one thing in every language: whether the
answer's first line states the gold. A gold other than 0 matches when the
line writes it anywhere (digits, number words, Korean and Han numerals); a
gold of 0 matches a statement of none ("No hay proveedores", "仕入先はあり
ません", "공급업체는 없습니다") with no number the question does not write
("En el lote 7 no hay proveedores" asked about lot 7 is 0). Which number
is the count is left to the LLM judges. The rule and its limits:
[`answer_match`](../reference/evaluation-output.md#answer_match).
