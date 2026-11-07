# Console (`run-console`)

> 🇰🇷 한국어판: [console.ko.md](./console.ko.md)

A local web app over the same entry points as the CLIs: ask questions and
follow each source of an answer back to its passage and its place in the
graph, ingest a folder of documents, run and compare evaluations, explore
the exported graph, see what queries and jobs cost, and edit the
configuration. It is a prototype for one person on one host: it has no
login, and it binds `127.0.0.1` unless told otherwise. How it is built:
[Design Doc §10.1](../design.md#101-the-console-run-console-opt-in). Every
flag: [CLI Reference](../reference/cli.md#run-console).

## Start the console

The console is the `console` extra; its web client is built once with
Node.js 20.19 or later (the build output is not in the repository):

```bash
uv sync --extra console
(cd console/web && npm ci && npm run build)
uv run run-console --config-path config.yaml --source-directory ./workspace
```

It prints `Console on http://127.0.0.1:8501/`. Open that address in a
browser on the same host.

- `--source-directory` is the **workspace**: the folders of documents
  offered as corpora, the `run-eval` datasets (JSON files holding a list of
  questions) and the `uploads/` folder uploads go to. Default: the working
  directory.
- `--cache-directory` (default `cache`) holds the ingestion jobs' pipeline
  caches and the console's own records in `console/`: the questions asked
  (`console.sqlite3`), the jobs and their logs.
- `--config-path` is the file the Settings page edits and every job runs
  with. Without one the console runs on the defaults and saves nothing.
  A file that does not load still starts the console, with its errors on
  every page and the Settings page to fix them.
- `--trace-url http://localhost:16686/trace/{trace_id}` links each answer to
  its trace (Jaeger here). With `observability.backend: langfuse` and
  `LANGFUSE_HOST` set, answers link to Langfuse without it.
- A `sample-questions.json` (a JSON list of strings) in the workspace
  becomes the Ask page's suggested questions.

Without a build, the page at the address says how to build the client;
the API still answers at `/api/status`.

## Try the demo

`console/demo/demo.py` prepares a workspace with a small synthetic corpus
(a robot maker buying lidar units, its carrier, depot and inspection lab,
with a Korean and a Japanese note on the same deal; every name is made up),
a matching seven-question `run-eval` dataset with one unanswerable
question, a config with the embedded SQLite stores and incremental
indexing on, ingests the corpus with `run-ingestion` and starts the
console on `127.0.0.1:8501`:

```bash
uv sync --extra console
(cd console/web && npm ci && npm run build)
uv run python console/demo/demo.py --models fake
```

| `--models` | What answers | Needs |
|---|---|---|
| `fake` | A deterministic model server on a loopback port: the extraction finds the corpus' names, answers quote the context sentences closest to the question. For screenshots, recordings and tests; illustrative prices show the cost columns | Nothing |
| `ollama` | A running Ollama with the local preset's models (`--model-base-url` for another server) | `ollama pull qwen2.5:7b-instruct` and `bge-m3` |
| `bedrock` | Amazon Bedrock in `--region` (default `AWS_REGION`, else `us-west-2`) with your credentials (an SSO session, an instance role); the stores stay local | Model access in the region; every call is billed |

The workspace is `.console-demo/` (`--workspace` to change it). Running
the script again rewrites the corpus and the config, and the ingestion
finds nothing new. `--prepare-only` stops before starting the console;
`--no-ingest` skips the ingestion.

## Reach it on a remote host

On a remote host (a cloud desktop, an EC2 instance) keep the console on
`127.0.0.1` and forward its port:

- **SSH:** `ssh -L 8501:127.0.0.1:8501 <host>`, then open
  `http://127.0.0.1:8501/` on your machine.
- **VS Code Remote:** the Ports view forwards 8501 when the console starts
  (or add it there); open the forwarded address.

Never bind `0.0.0.0` for the demo. Another address needs `--allow-remote`,
and then an authenticating reverse proxy in front: anyone who reaches the
port can query the index, start jobs and edit the configuration.

## Pages

**Ask.** A question box with the strategy picker: `auto` and the strategies
grouped by methodology, those the index namespace does not hold disabled
with the reason. The answer streams in as the answer model writes it; the
final answer then replaces it (it is the one a decline or the output
guardrail decided). Under it: for `auto`, the strategy it chose and how
(the router named it, it was the only candidate under the filters, or the
router failed and it fell back), and a line with the strategy, the time,
the tokens in and out, the model calls, the cost (when `evaluation.prices`
prices the models), the rerank status and the trace link. A declined
answer says why (no context, the model's `[NO_ANSWER]`, an insufficient
context under the decline policy); a blocked one says the guardrail
blocked it. Follow-ups keep the conversation unless unticked. Search
options set the results per search, the search type and `key: value`
filters.

The right column is the **provenance trace**, the console's signature: a
rail with every source the answer model read, in rank order, filterable by
kind (passage, entity, relationship, community report). Opening a source
draws its trail: an entity or relationship lists the passages of the same
answer that state it (arcs to their stations on the rail) and how many of
its passages the answer did not read; a passage names its file and chunk
and the graph sources that cite it; a seed-graph statement shows its seed
graph and IRI. **Show in the graph** opens the Graph page on that entity
(or edge, or community), with the trail continued above the graph back to
the question.

**Ingest.** The workspace's folders of documents with their file counts,
sizes and types, plus an upload (a folder or files) into `uploads/`. Pick
what to build (the methodologies: build only what you will query; adding
one later builds only its artifacts) and, under more options, another
index namespace, a forced recompute and the answers to the two questions a
run can ask (a reset that drops another source's content, a mass
deletion). Start runs `run-ingestion` with a pipeline id per corpus, so
running the same corpus again resumes or runs incrementally. The run's
stages appear as a column of stations as they finish (from the pipeline's
own metadata), with their time, outputs and model calls, each failed
stage's error and failed files, and the log's tail. Stop sends the CLI
the stop it is built for: the finished stages are kept and the next run of
the corpus resumes.

**Evaluate.** Pick a dataset (its questions, the unanswerable ones and
which evaluators it feeds are shown), the strategies (the first is the
baseline), repeats, a closed-book arm, the pairwise judge, regression gates
(`MEASURE:STAT<OP>VALUE`, as `run-eval --fail-if`) and a failure budget.
Start runs `run-eval --output-format json` into its own directory under
`evaluation.outputs_directory`; its runs count up as they finish. Any saved
run (the console's or another `run-eval`'s) opens as a results table (each
metric's mean per strategy), the paired comparison against the baseline (a
row per metric: the mean difference with its interval drawn on the row's
own scale, marked better or worse only when the interval excludes zero, and
the wins, ties and losses), the comparison's warnings, the cost per
question (USD, tokens, median and 95th percentile time), the declines
(rate, and precision and recall over the unanswerable questions) and, with
the pairwise judge, the win rate per answer-length bucket. A job's gates
show as passed or failed.

**Graph.** A `visualization_data.json` the ingestion exported (the newest
by default), drawn with a seeded layout per connected component: pan with a
drag or the arrow keys, zoom with the wheel or `+`/`-`. Colour by community
or by type (the eight largest groups, the rest grey, named in the legend).
Find an entity by name; selecting one shows its type, description,
connections, passages, community, IRI and schema status, and lists its
relationships to walk to. The Communities tab lists the hierarchy and
highlights a community's members; the Analysis tab counts the entities,
relationships, connected parts and density, draws the degree histogram,
and counts the entity and relationship types, schema conformance (in the
schema, outside it, not checked) and the evidence (documents, a seed graph,
both). A table of every entity is under the graph.

**Dashboard.** Over the last day, week, month or year: the questions asked
through the console, their median and 95th percentile time, the decline
and error counts and the cost; questions and median time per day; per
strategy, model and role the calls, tokens and cost; and the jobs with
their duration, calls, tokens and cost. Costs need `evaluation.prices`
([Cost and prices](./evaluate.md#comparing-strategies)); a model without a
price is listed and left out instead of counted as free. Clear question
history deletes the recorded questions.

**Settings.** The config file, as fields or as YAML. Fields come from the
configuration's schema with each key's description as help, its default
and whether the file sets it; find one by key or description. Changes
collect on the right; Review checks them as a run would load the file
(unknown keys and the `UKR__` environment included) and shows the diff of
the file, its errors and warnings; Save writes it, refused if the file
changed on disk meanwhile. Saving from fields rewrites the YAML without its
comments (the review says so); the YAML tab keeps them. The Methodologies
tab edits `indexing.methodologies`; the Tracing tab edits `observability`
and shows which `LANGFUSE_*`, `PHOENIX_*` and `OTEL_*` variables are set,
never their values. The Effective tab shows the configuration a run gets.
The next question and job use the saved file.

## What it reads and writes

- **Reads:** the workspace, the cache directory, the evaluation outputs and
  `graph.visualization.outputs_directory`. Every path the page names is
  resolved (symlinks too) and must stay inside one of them.
- **Writes:** `uploads/` in the workspace, the cache directory (pipeline
  caches, `console/`), the evaluation outputs and the config file. The
  stores are written by the `run-ingestion` jobs, as the config says.
- **Secrets:** never sent to the page and never written. A file that holds
  one (`providers.llm.api_key`, `aws.opensearch.password`) can be viewed
  as fields with the secret masked, but not saved from the console: move it
  to its environment variable (`UKR__PROVIDERS__LLM__API_KEY`) first.
- **Questions:** the dashboard's store keeps each question's text, strategy,
  timing, usage and cost, never the answer or its sources.

## Security

- Loopback by default; `--allow-remote` for any other address, behind an
  authenticating proxy. The console has no login of its own.
- Requests that change anything need a header a cross-site form cannot
  send, and one from another origin is refused; on loopback a request for
  another host name is refused (DNS rebinding). No response carries a CORS
  header.
- The page loads nothing from another host: its fonts (Bricolage Grotesque,
  Atkinson Hyperlegible Next and Mono, SIL Open Font License) are bundled by
  the build, so it works offline and in a private VPC.
- One ingestion or evaluation job per index namespace at a time; the
  pipeline's namespace lease still guards against a `run-ingestion` started
  elsewhere.

## Languages

The page is in English, Korean, Japanese, Simplified Chinese, German,
Spanish, French, Italian, Portuguese and Russian (the languages the
library supports), chosen in the side bar or on the Settings page,
defaulting to the browser's language and kept in the browser. Documents,
questions and answers can be in any language; numbers, dates, durations
and currency follow the chosen language.

To add a language: copy `console/web/src/i18n/locales/en.json` to
`<code>.json` and translate the values (keep every key and every
`{{placeholder}}`; plural keys take the suffixes `Intl.PluralRules` gives
that language: `_one`, `_few`, `_many`, `_other`...), add a glossary of
its terms in `src/i18n/glossary/<code>.json`, and register it in `LOCALES`
in `src/i18n/i18n.tsx`. `npm test` fails when a locale lacks a key, has an
extra one, changes a placeholder or has the wrong plural forms.

## Develop the web client

```bash
cd console/web
npm ci
npm run dev        # http://127.0.0.1:5173, /api forwarded to run-console on 8501
npm run typecheck && npm run lint && npm test
npm run build
```

- `npm run test:smoke` runs the Playwright smoke test against a running
  console (`CONSOLE_URL`, default `http://127.0.0.1:8501`); it skips when
  no browser is installed (`npx playwright install chromium`).
- `node scripts/screenshots.mjs --base-url <url> --out <dir>` screenshots
  every page (`--locales en,ko --widths 1440,390`).
- `node scripts/record-tour.mjs --base-url <url> --out <dir>` records a
  scripted 25-second tour (a question, its sources, the trail into the
  graph, an evaluation comparison, the dashboard) as a video, and with
  `ffmpeg` on the path converts it to a GIF and an MP4 of the same framing.

## Limits

- One user on one host: no accounts, no access control between users, no
  history shared between browsers.
- The Graph page draws the 600 best-connected entities of a large graph;
  the others are found by name or in the table.
- A job's progress is per stage (ingestion) or per run (evaluation), not
  per document or question.
- Running the console against the AWS stack's Neptune and OpenSearch works
  as the CLIs do, through the config, but has not been measured there yet.
