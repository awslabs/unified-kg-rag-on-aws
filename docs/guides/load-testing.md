# Load, soak and chaos testing (local)

`scripts/loadtest/` drives queries and ingestion runs on one host, with the
console demo's deterministic fake models served on a loopback port, and checks
what a production deployment cares about: latency under concurrency, resources
over a long run, and whether a run hit by model faults, a store outage or a
kill converges with a full build. It is a regression harness for the code
around the models (retrieval, fusion, stores, the event loop, recovery). The
fake models answer in milliseconds, so its numbers are not product
performance: with real models the model calls dominate latency, cost and
throttling.

Nothing leaves the host. Every script unsets every `AWS_*` variable, points
the shared AWS config and credentials files at `/dev/null` (so no profile or
SSO cache in `~/.aws` signs a request) and turns the instance metadata lookup
off. The chaos model server binds 127.0.0.1 only, and `load.py` refuses a
config that is not `backend: local`, whose models or stores are not of a
local kind (Ollama or OpenAI-compatible models, a cross-encoder or no
reranker, the SQLite stores and registry, a Gremlin Server or OpenSearch) on
a loopback address, or that screens with the Bedrock guardrail.

| Script | What it does |
|---|---|
| `scripts/loadtest/workspace.py` | Writes a workspace: the demo corpus (or `--docs N` synthetic documents in English, Korean or both), the demo's eval dataset and a `backend: local` config on the embedded SQLite stores (`--stores containers`: the stores of `docker/compose.local.yaml`), with every methodology so every strategy runs |
| `scripts/loadtest/chaos_models.py` | The fake model server (`console/demo/fake_models.py`) with fault injection: delay and jitter, hangs, dropped connections and error answers (429, 500, 503), changed at run time with `POST /chaos`; `GET /stats` counts requests, faults and requests in flight |
| `scripts/loadtest/load.py` | `sweep`, `soak` and `http` load modes, and `gate`, which checks a report against the nightly gates |
| `scripts/loadtest/compare_exports.py` | The convergence check: two `run-describe --include-text` exports hold the same records with the same content |
| `scripts/loadtest/nightly.py` | The nightly run below, end to end: a sweep, a soak and three chaos scenarios, then the gates |
| `scripts/loadtest/harness.py` | The statistics and the gates, which `tests/unit/test_loadtest_harness.py` checks in every pull request |

## Set up

```bash
uv sync --extra local                      # the Ollama client the fake models are reached with
uv run python scripts/loadtest/chaos_models.py --port 11613 &
uv run python scripts/loadtest/workspace.py ws --model-port 11613
cd ws && uv run run-ingestion --config-path config.yaml --source-directory corpus --yes && cd ..
```

`--lightrag-graphrag-only` indexes the default methodologies only (no `ppr`
or `tree`). The config keeps retries on (`retry.max_attempts: 3`) with a
short `retry.base_delay_seconds` (0.2 s) for the model clients' own retries.
Ingestion items back off on their own schedule (up to 120 s between
attempts), so an ingestion run under injected faults takes minutes.

## Scenarios

### Concurrency sweep

```bash
uv run python scripts/loadtest/load.py sweep --config ws/config.yaml \
  --levels 1,4,16,64 --queries 140 --output sweep.json
```

After one warm-up pass of every query, each level runs `--queries` queries
drawn from a fixed mix: one or more per strategy, in English, Korean and
Japanese, each answerable from the demo corpus (`--strategies mix,local`
narrows it). Each level reports p50, p95, p99 and max latency, throughput,
errors by `ErrorClass` (`errors_by_class`, and `errors_by_strategy` as
`strategy:class`: a raised error, or the error answer of
`processing.ignore_errors`) with when the failed queries ended
(`error_p99`, `error_max`), the flagged answers (`flags`, as
`flag:strategy`), and the event-loop lag. An answer is flagged when it has
no evidence (`no_sources`, `abstained`, `empty_answer`), was `degraded` or
has `citations_unresolved`: a retrieval regression that empties the
context answers "could not find" without an error, and shows here. The lag is how
late a 10 ms ticker wakes up: the time the loop spent in code that does not
yield, which delays every concurrent query. A rising p99 lag with
concurrency points at CPU work on the loop that belongs in a thread.

### Soak

```bash
uv run python scripts/loadtest/load.py soak --config ws/config.yaml \
  --concurrency 8 --duration 2400 --sample 30 --memory-share 0.3 --csv soak.csv
```

Runs the mix for `--duration` seconds (`--memory-share`: the share of
queries that start a conversation with memory on). Every `--sample` seconds
it records RSS, file descriptors (SQLite files and sockets counted apart),
threads, gc objects, asyncio tasks, the conversations held, and the window's
p50 and p95 (`--csv`). The report adds the per-hour slope of each over the
second half of the run (`trends_per_hour`; the first half is warm-up while
caches and pools fill) and how much each rose over that half
(`second_half_rise`, the fitted slope times the half's span). A slope that
stays positive over a longer run is a leak. The harness keeps its own memory
flat (latencies in a compact array, the loop lag in a fixed histogram), so
the slope is the chain's. The slope of a short run overstates growth that
levels off later (about 1 MB of allocator noise over a 5-minute half reads as
12 MB/h): judge a slow leak on a run of 30 minutes or more. The RSS gate
therefore fails only when RSS grows both faster than its rate and by more
than its floor over the second half, so a short soak catches a gross leak
only and a long one is judged by the rate.

### Console over HTTP

```bash
UKR_CONSOLE_TOKEN=... uv run run-console --config-path ws/config.yaml &   # needs the console extra
UKR_CONSOLE_TOKEN=... uv run python scripts/loadtest/load.py http \
  --console http://127.0.0.1:8501 --levels 1,8,32 --queries 80
```

The sweep through the console's `/api/ask` stream, with the time to the
first event (`first_event_p50`, `first_event_p95`). An error event counts
under its `error_kind`, and a result is flagged as in the sweep. The loop lag
it reports is this client's (`client_loop_lag_p99_ms`), not the console's.

### Bulk ingestion throughput

```bash
uv run python scripts/loadtest/workspace.py bulk --model-port 11613 --docs 1500 --chars 4000 --lang mixed
cd bulk && /usr/bin/time -v uv run run-ingestion --config-path config.yaml \
  --source-directory corpus --output-format json --yes > ingest.json
```

Wall time and peak RSS (`Maximum resident set size`) per document count show
how ingestion scales with fake models: the pipeline's own cost (parsing,
chunking, merging, resolution, clustering, embedding writes). With real
models the model calls dominate, and their concurrency and throttling are
set by `processing` (see [Ingestion: Cost levers](./ingest.md#cost-levers)).

### Chaos

Each scenario ends with a convergence check: export the namespace and
compare it with a full build of the same corpus in the same workspace
directory (record ids name the source path, so build the reference in the
same place, wiping `stores/` and `cache/` between builds). The export holds
every kind of item the vector store keeps (text units, entities,
relationships, community reports, claims, RAPTOR summary nodes), so every
methodology's index is compared; the graph store's vertices and edges are
not exported.

```bash
cd ws && uv run run-describe --config-path config.yaml --output-path ../reference.jsonl --include-text && cd ..
# ... the scenario, then the same export to ../after.jsonl
uv run python scripts/loadtest/compare_exports.py reference.jsonl after.jsonl [--incremental]
```

| Scenario | How | What must hold |
|---|---|---|
| Model throttling and errors | `curl -X POST 127.0.0.1:11613/chaos -d '{"error_rate": 0.3, "error_status": 429}'` (or 500, 503) during a run, `{"error_rate": 0}` and run again | The faulted run records the documents it could not finish `FAILED` (or fails a stage over its failure budget, exit 1); the clean run after it exits 0 with none failed and converges with `--incremental` |
| Dropped connections | `{"drop_rate": 0.05}` | As above |
| Hung model | `{"hang_rate": 0.15, "hang_seconds": 60}` during a sweep with `search.query_timeout_seconds` set | Every query ends within the query deadline, as an answer or as a `timeout` error: `max` and `error_max` stay under it |
| Store outage | With `--stores containers`: `docker compose -f docker/compose.local.yaml pause opensearch` (or `gremlin-server`) during a run or a sweep, then `unpause` | Queries fail as `store_unreachable` or `timeout` (a strategy that does not read the paused store still answers); the next ingestion run converges |
| Killed run | `kill -9` the `run-ingestion` process mid-run, then at once run again with the same `--pipeline-id` | The resume takes the killed run's lease over (the SQLite registry hands on the lease of a holder of this host whose process has exited: no `NamespaceLeaseError`), reuses the finished stages and converges exactly |
| Concurrent runs | Two `run-ingestion` runs on one namespace | The second stops with `NamespaceLeaseError` (the console: `409 job_conflict`) |

Without `--incremental` the comparison is exact: same records, same content,
lists in the same order, with only build times and run ids left out. A run
resumed after a kill must pass it. A clean run after a partly failed one
re-indexes the failed documents as a delta, so it matches a full build only
up to what [incremental indexing documents](./incremental.md#rebuild-signals):
it clusters only the delta and appends its communities, a merged description
keeps the stored summary and appends the delta's lines, and lists of ids may
come in another order. `--incremental` compares within that (no communities
or community memberships, no descriptions with their embedding and summary
record, lists as multisets): the same documents, text units, entities and
relationships, cited by the same text units, with the same embeddings of
everything else.

## Nightly workflow

[`load-test`](../../.github/workflows/load-test.yml) runs
`scripts/loadtest/nightly.py` every night on the embedded stores (and on
demand): ingest the demo corpus, a sweep at
concurrency 1, 4 and 16, a 10-minute soak at 8 with 30% memory turns, and three
chaos scenarios on a 60-document synthetic corpus (30% 429s, then 15% 503s
with 5% dropped connections, each followed by a clean run; a SIGKILL halfway
through a run, then its resume at once). It uploads `report.md`, `report.json`, the
soak samples, the exports and the logs as the `load-test-report` artifact and
writes `report.md` to the job summary. It fails when a gate fails:

| Gate | Threshold |
|---|---|
| Errors on the fixed corpus (every sweep level and the soak) | 0 |
| Flagged answers (every sweep level and the soak): no sources, abstained, empty, degraded, unresolved citations | 0 (every query of the mix is answerable from the demo corpus, and the fake models are deterministic) |
| Event-loop lag p99 (every sweep level and the soak) | under 100 ms |
| RSS over the soak's second half | fails only when it grows both at 10 MB/h or more and by 8 MB or more |
| Chaos scenarios | each converges (the kill exactly, the others `--incremental`); the kill's resume, started at once, finds no lease held |

The RSS floor (8 MB) sits well above the noise of a clean 10-minute soak
(8 to 13.5 MB/h measured, from about 1 MB over its second half), so the
nightly soak catches gross leaks only (a leak of about 1.2 KB a query passes
the floor over its second half); a slow leak is judged on a soak of
30 minutes or more, where the rate decides. The floor is not yet calibrated
on the runner: repeated clean soaks there would set it so that false
failures stay rare.

It only reports: no branch protection requires it, and pull requests do not
run it. Run it locally with
`uv run --extra local python scripts/loadtest/nightly.py out/` (about 20
minutes; `--soak-seconds` and `--chaos-docs` scale it), and check another
report with `load.py gate report.json` (`--max-errors`, `--max-flags`,
`--max-loop-lag-ms`, `--max-rss-mb-per-hour`, `--rss-rise-floor-mb`).

## Local baseline (fake models)

Measured once on a development host (Linux, 16 vCPUs, shared with other
work) at the 2026-11-12 review, with the demo corpus (8 documents) on the
embedded stores and the fake models. **These are local numbers with fake
models, not product performance**: they show the overhead of the code around
the models and are the reference a regression is measured against on the same
host.

| Run | Requests | Throughput | Latency | Errors |
|---|---|---|---|---|
| In process, concurrency 1 | 140 | 8.3 rps | p50 0.11 s, p99 0.29 s | 0 |
| In process, concurrency 16 | 140 | 18 rps (saturated) | p50 0.86 s, p99 1.44 s | 0 |
| In process, concurrency 64 | 140 | 17 rps | p50 2.99 s, p99 6.6 s | 0 |
| Console over HTTP, concurrency 1 | 80 | 2.9 rps | first event 0.05 s | 0 |
| Console over HTTP, concurrency 32 | 80 | 9.1 rps | p99 6.2 s | 0 |
| Soak, 40 min, concurrency 8, 30% memory turns | 31,210 | 12.9 rps | p50 0.55 s, p99 1.72 s | 0 |

- Event-loop lag p99 stayed at 9 ms from concurrency 1 to 16.
- Soak resources: RSS went from 163 to 168 MB (+4.5 MB/h over the second
  half). File descriptors, SQLite files, threads and gc objects stayed flat,
  and the conversations held stayed at the 100 cap: no leak.
- Bulk ingestion with the fake models: 1,500 documents (3,000 text units) in
  100 s at 617 MB peak RSS; one 15 MB file (6,710 units) in 202 s at 966 MB;
  8,000 documents (15.5k units) in 362 s at 1.77 GB, about 101 KiB per unit.
- Chaos: the runs after 30% 429s, after a SIGKILL and after an OpenSearch
  pause converged (the pause up to the community drift incremental indexing
  documents). A kill mid-run resumed with the stages it had finished reused
  (141 model calls instead of 551). With 15% hung, 5% dropped and 5% 503
  model calls and an 8 s query deadline, 14% of queries failed, all within
  7.2 s. 968 queries during a 300-document incremental run
  had no errors, at twice the p50.

The demo corpus has no hub entities, so a cost that grows with how many text
units an entity cites does not show in the nightly run: measure that on a
`--docs` corpus, whose 13 entities each appear in most documents.
