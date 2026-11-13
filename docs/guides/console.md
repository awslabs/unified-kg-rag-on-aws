# Console (`run-console`)

> 🇰🇷 한국어판: [console.ko.md](./console.ko.md)

A local web app over the same entry points as the CLIs: ask questions and
follow each source of an answer back to its passage and its place in the
graph, ingest a folder of documents, run and compare evaluations, explore
the exported graph, see what queries and jobs cost, and edit the
configuration. It is a prototype for one person on one host: it has no
login, and it binds `127.0.0.1` unless told otherwise. How it is built:
[Design Doc §10.1](../design/surfaces.md#101-the-console-run-console-opt-in). Every
flag: [CLI Reference](../reference/cli.md#run-console).

## Start the console

The console is the `console` extra; its web client is built once with
Node.js 20.19 or later (the build output is not in the repository):

```bash
uv sync --inexact --extra console
(cd console/web && npm ci && npm run build)
uv run run-console --config-path config.yaml --source-directory ./workspace
```

It prints `Console on http://127.0.0.1:8501/?token=…`. Open that address in
a browser on the same host: the token is new at every start (or
`UKR_CONSOLE_TOKEN`, 16 characters or more, when a launcher sets it), the page
keeps it in an HttpOnly cookie, and every request without it is refused.

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
- `--metrics-sink cloudwatch` publishes each question's query metrics
  (`query_latency_ms`, `query_failed`, ...) as CloudWatch EMF on stdout
  (stderr with `--output-format json`), for the CDK stack's query alarms
  ([runbook](../operations.md#alarms)).
- A `sample-questions.json` (a JSON list of strings) in the workspace
  becomes the Ask page's suggested questions.

Without a build, the page at the address says how to build the client;
the API still answers at `/api/status`. The client is found in
`console/web/dist` of the source checkout the package runs from, or where
`UKR_CONSOLE_WEB_DIRECTORY` points: an installed wheel carries no build, so
build it in a checkout and point the variable at its `dist`.

## Try the demo

`console/demo/demo.py` prepares a workspace with a small synthetic corpus
(a robot maker buying lidar units, its carrier, depot and inspection lab,
with a Korean and a Japanese note on the same deal; every name is made up),
a matching seven-question `run-eval` dataset with one unanswerable
question, a config with the embedded SQLite stores and incremental
indexing on, ingests the corpus with `run-ingestion` and starts the
console on `127.0.0.1:8501`.

The corpus mixes the formats the core install parses without an extra,
so the citations show each kind of source:

| File | Format |
|---|---|
| `supply-agreement.pdf` | PDF, 3 pages: parties, delivery and penalties with a table of figures, warranty and signatures |
| `quality-report.pdf` | PDF, 2 pages: an inspection table and the corrective action |
| `logistics-plan.md` | Markdown with headings and a list |
| `warranty-terms.html` | HTML with a table |
| `shipments.csv` | CSV, a row per shipment |
| `steering-minutes.json` | JSON meeting minutes |
| `contract-summary.ko.md` | Korean Markdown note |
| `quality-memo.ja.txt` | Japanese text note |

The demo writes the PDFs when it starts (`console/demo/pdf_writer.py`, a
small standard-library writer: the same bytes on every run) and chunks the
corpus about a page at a time, so a passage cited from a PDF names its
page (with the fake models, *Who carries the lidar units to Kestrel
Depot, and how fast?* cites `supply-agreement.pdf` page 2).

```bash
(cd console/web && npm ci && npm run build)
uv run --no-dev --extra console --extra local python console/demo/demo.py --models fake
```

The fake and Ollama models run in local mode and need the `local` extra
as well (`bedrock` does not); the demo checks both extras and the built web
client before it writes or ingests anything, and a SIGTERM to it stops the
console it started.

| `--models` | What answers | Needs |
|---|---|---|
| `fake` | A deterministic model server on a loopback port: the extraction finds the corpus' names, answers quote the context sentences closest to the question. For screenshots, recordings and tests; illustrative prices show the cost columns | Nothing |
| `ollama` | A running Ollama with the local preset's models (`--model-base-url` for another server) | `ollama pull qwen2.5:7b-instruct` and `bge-m3` |
| `bedrock` | Amazon Bedrock in `--region` (default `AWS_REGION`, else `us-west-2`) with your credentials (an SSO session, an instance role); the stores stay local | Model access in the region; every call is billed |

The script prints the console's address with its access token. The
workspace is `.console-demo/` (`--workspace` to change it). Running
the script again rewrites the corpus and keeps the config it wrote the first
time, with whatever Settings saved into it (prices, models, directories),
after checking it loads; with the fake models it serves them on the port
the kept config names. `--reset-config` writes the demo's config again
(also the way to switch `--models`). The ingestion finds nothing new. `--prepare-only` stops before starting the console;
`--no-ingest` skips the ingestion.

## Reach it on a remote host

On a remote host (a cloud desktop, an EC2 instance) keep the console on
`127.0.0.1` and forward its port:

- **SSH:** `ssh -L 8501:127.0.0.1:8501 <host>`, then open the printed
  address (`http://127.0.0.1:8501/?token=…`) on your machine.
- **VS Code Remote:** the Ports view forwards 8501 when the console starts
  (or add it there); open the forwarded address.

Never bind `0.0.0.0` for the demo. Another address needs `--allow-remote`,
and then an authenticating reverse proxy in front: anyone who reaches the
port can query the index, start jobs and edit the configuration.

## Pages

**Ask.** A question box with the strategy picker: `auto` and the strategies
grouped by methodology, those the index namespace does not hold disabled
with the reason. A namespace nothing was ingested into yet says so
before the first question, with a way to Ingest, and a
question to it names the same. The answer streams in as the answer model writes it; the
final answer then replaces it (it is the one a decline or the output
guardrail decided). Under it: for `auto`, the strategy it chose and how
(the router named it, it was the only candidate under the filters, or the
router failed and it fell back), and a line with the strategy, the time,
the tokens in and out, the model calls, the cost (when `evaluation.prices`
prices the models), the rerank status and the trace link. A declined
answer says why (no context, the model's `[NO_ANSWER]`, an insufficient
context under the decline policy); a blocked one says the guardrail
blocked it. The answer's citation markers (`[1]`, `[2, 3]`) open the
source they cite in the provenance trace, and the sources the answer cites
are listed under it with their file and pages; a number the context did
not show is struck through. An answer of a strategy that does not cite
(`global` and `drift` by default, [Citations](./query.md#citations)) is
shown as written, without markers. Follow-ups continue the conversation unless
unticked: the page sends the earlier questions and answers with the
question (`RAGInput.history`), so the conversation survives a settings
save or an ingestion. Search options set the results per search, the
search type and `key: value` filters.

The right column is the **provenance trace**, the console's signature: a
rail with every source the answer model read, in rank order, filterable by
kind (passage, entity, relationship, community report). Opening a source
draws its trail: an entity or relationship lists the passages of the same
answer that state it (arcs to their stations on the rail) and how many of
its passages the answer did not read; a passage names its file and chunk
and the graph sources that cite it; a seed-graph statement shows its seed
graph and IRI. A source carries the context marker it was shown under,
highlighted when the answer cites it. The `structured` strategy's graph query result
is a source of its own: its table, the query it ran, the number of rows
and whether the table is whole, the graph store, the passages its entities
cite and **Show in the graph** on the first entity it returned. **Show in the graph** opens the Graph page on that entity
(or edge, or community), with the trail continued above the graph back to
the question.

**Ingest.** The workspace's folders of documents with their file counts,
sizes and types, plus an upload (a folder or files) into `uploads/`. Pick
what to build (the methodologies: build only what you will query; adding
one later builds only its artifacts). The picker starts from what the
target namespace's build record says it holds, so another corpus ingested
into it keeps every methodology its strategies serve; a pick that builds
less (a held methodology whose artifacts the run would not build) is
listed, and the run starts only once you confirm removing them: the
namespace then stops serving their strategies for every corpus in it, not
only this one (the API refuses it as `narrowing` without
`confirm_narrowing`). Under more options: another
index namespace, a rebuild of the namespace from the whole corpus for this
run only (`--reset`, so no setting has to be switched back; what the corpus
does not hold is dropped from the namespace), a forced recompute and the answers to the two questions a
run can ask (a reset that drops another source's content, a mass
deletion). Start runs `run-ingestion` with a pipeline id per corpus folder
(its name and a hash of its path and namespace), so running the same corpus
again resumes or runs incrementally. Adding, editing or removing files
and starting again is one run: with the doc-status registry on, the delta
says what is processed, and no question is asked. A run that stops at one of
its questions (a setting changed since this corpus's last run, an incomplete
stage cache, a reset, a mass deletion) says which, and offers to start again answering
that question alone (`--confirm-integrity`, `--confirm-reset`,
`--confirm-deletions`); the same answers can be ticked under more options
beforehand. The run's
stages appear as a column of stations as they finish (from the pipeline's
own metadata), with their time, outputs and model calls, each failed
stage's error and failed files, and the log's tail. A finished run counts
the WARNINGs it logged and lists the last 20 (one that advises a rebuild,
community drift or re-clustering due, offers to start the run again with
the rebuild), and flags a run that
processed documents but extracted no entity from them. Stop sends the CLI
the stop it is built for: the finished stages are kept and the next run of
the corpus resumes. A job a console left running when it stopped is shown
as lost when the console starts again (its namespace is free; the
pipeline's namespace lease still guards the stores). Uploads go in batches
of up to 500 files or 100 MiB; the server takes up to 10,000 files and
500 MiB a request. The page leaves out the files ingestion does not read
(of another type, or hidden: a file or folder whose name starts with `.`,
or `__pycache__`), and the server refuses a request that holds one, or that
names one path as a file and as a folder (`upload_conflict`); a request is
written all or none, so a refused one leaves the folder as it was, without
a folder it created (only a folder changed by something else while the
files move in, a path made a folder, can stop the moves partway; those
moved before stay).

**Evaluate.** Pick a dataset (its questions, the unanswerable ones and
which evaluators it feeds are shown), the strategies (the first is the
baseline), repeats, a closed-book arm, the pairwise judge, regression gates
(`MEASURE:STAT<OP>VALUE`, as `run-eval --fail-if`) and a failure budget.
Start runs `run-eval --output-format json` into its own directory under
`evaluation.outputs_directory`; its progress counts the answers (from the
answer batch's progress lines in its log, every ~10% or minute) and the
runs saved. Interleaved runs (`evaluation.run_order`, the default with two
or more) are answered in one batch and scored and saved only once every
answer is in, so stopping one keeps no run; one at a time, Stop keeps the
runs that finished. Any saved
run (the console's or another `run-eval`'s) opens as a results table (each
metric's mean per strategy), the paired comparison against the baseline (a
row per metric: the mean difference with its interval drawn on the row's
own scale, and the wins, ties and losses); hovering a metric's name gives
its description and which way is better. A comparison adjusted for
multiplicity ([Comparing strategies](./evaluate.md#comparing-strategies))
marks a difference better or worse only when its Holm-adjusted p-value is
below the comparison's alpha, shows that p-value, draws the family's
simultaneous interval (with its confidence) and marks the primary metric;
an older comparison, and a delta without per-query pairs or a closed-book
arm, falls back to the interval excluding zero. Then the comparison's warnings, the cost per
question (USD, tokens, median and 95th percentile time), the declines
(rate, and precision and recall over the unanswerable questions) and, with
the pairwise judge, the win rate per answer-length bucket. **Per
question**, read once opened, lists each question with every strategy's
score on the primary metric (the mean over its repeats) and its outcome
(win, tie or loss) for the shown candidate against its baseline, filterable
by outcome; a question opens the reference answer and every strategy's
answer side by side. A question marked unanswerable shows how many of each
strategy's repeats declined, the right answer to it, and its outcome compares
those shares. A job's gates show as passed or failed.

**Graph.** A `visualization_data.json` of the namespace (the newest by
default): a full build writes the namespace's whole graph, and no console
job exports one, so when the page opens and no export shows what the stores
hold (after an incremental job) the console builds it from them, as
`run-visualization` does. The picker names each export's build time,
namespace and run, and marks one a later build of its namespace replaced as
outdated. Without a graph the page says why: nothing ingested yet, a
namespace indexed before build records (rebuild it once), or
`graph.visualization.export_data` off (Settings); when building it failed,
it says so over the newest export. It is drawn with a seeded layout per connected component: pan with a
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
rate, the errors and the refused questions (refused as asked, on an empty
namespace or with a filter to change: not errors, like a cancelled
one left out of every time and of the decline rate, and not counted as a
strategy's) and the cost (a failed question is left out of the decline rate
too: it could not decline);
questions and median time per day; per
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
comments (the review says so); the YAML tab keeps them. A change to where
a provider sends its requests (`providers.*.base_url`,
`aws.neptune.endpoint`, `aws.opensearch.endpoint`, their ports) is listed in
the review and saved only once confirmed there: the requests carry the
credentials the console's environment holds for that provider (an API key,
a store password, signed AWS requests). A change to a file or directory
path (every key named `path`, `directory` or ending in `_path` or
`_directory`: the directories the console serves,
`evaluation.outputs_directory` and `graph.visualization.outputs_directory`,
and what its jobs read and write, such as the SQLite store files,
`logging.log_file_path`, schema and seed graph files and the parsing
directories) is confirmed the same way: it opens the new path to the page. The file is written whole (a
temporary file beside it, renamed over it). The Methodologies
tab edits `indexing.methodologies`; the Tracing tab edits `observability`
and shows which `LANGFUSE_*`, `PHOENIX_*` and `OTEL_*` variables are set,
never their values. The Effective tab shows the configuration a run gets.
The next question and job use the saved file.

## What it reads and writes

- **Reads:** the workspace, the cache directory, the evaluation outputs and
  `graph.visualization.outputs_directory`. Every path the page names is
  resolved (symlinks too) and must stay inside one of them. Folder listings
  skip symlinks, and an ingest job, like `run-ingestion`, skips a corpus
  file that resolves outside its folder (a WARNING names it).
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
  authenticating proxy. The console has no accounts: a per-launch access
  token, in the printed address and then an HttpOnly, SameSite=Strict
  cookie, is required on every request.
- Requests that change anything need a header a cross-site form cannot
  send, and one from another origin is refused; on loopback a request for
  another host name is refused (DNS rebinding). No response carries a CORS
  header.
- The page loads nothing from another host: its fonts (Pretendard,
  Pretendard JP and JetBrains Mono, SIL Open Font License; the texts are in
  `console/web/public/licenses/`) are part of the build, so it works
  offline and in a private VPC.
- One ingestion or evaluation job per index namespace at a time, also for
  requests made at the same moment; the pipeline's namespace lease still
  guards against a `run-ingestion` started elsewhere. Stopping the console
  stops its jobs and waits up to the CLIs' stop grace (110 s) for them.

## Light and dark

The page has a light and a dark theme. It follows the operating system's
setting by default; **Theme** in the top bar (under **Menu** on a
narrow screen) switches it to *System*, *Light* or *Dark*, and the browser keeps
the choice. The graph, the charts and the evidence highlights follow the
theme.

## Languages

The page is in English, Korean, Japanese, Simplified Chinese, German,
Spanish, French, Italian, Portuguese and Russian (the languages the
library supports), chosen in the top bar,
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

- The design brief (palette, type scale, references, what to avoid) is
  [`console/web/DESIGN.md`](../../console/web/DESIGN.md); change it first
  when the direction changes. `npm run fonts` (run by `dev`, `build` and
  `typecheck`) writes `src/styles/fonts.css` from the `@fontsource`
  packages: WOFF2 only, the weights and subsets the styles use.
- `npm run test:smoke` runs the Playwright smoke test against a running
  console (`CONSOLE_URL`, the printed address with its token); it skips when
  no browser is installed (`npx playwright install chromium`).
- `node scripts/screenshots.mjs --base-url <url> --out <dir>` screenshots
  every page (`--locales en,ko --widths 1440,390`).
- `node scripts/record-tour.mjs --base-url <printed address> --out <dir>`
  records a scripted tour of at most 25 seconds (a question, its cited file and page, its sources, the trail into the
  graph, an evaluation comparison, the dashboard): each step waits for what
  it shows and fails with a screenshot rather than record a broken state;
  waits on models are cut. With `ffmpeg` on the path it writes the edit as
  an MP4 and a GIF of the same framing.

## Limits

- One user on one host: no accounts, no access control between users, no
  history shared between browsers.
- The Graph page draws the 600 best-connected entities of a large graph;
  the others are found by name or in the table.
- A job's progress is per stage (ingestion), or per answer and saved run
  (evaluation), not per document; the answers count only at a log level
  that shows INFO lines (`logging.level`).
- Running the console against the AWS stack's Neptune and OpenSearch works
  as the CLIs do, through the config, but has not been measured there yet.
