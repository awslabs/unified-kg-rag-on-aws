# Security Policy

## Reporting a Vulnerability

If you discover a potential security issue in this project, we ask that you notify
AWS/Amazon Security via our
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or directly via email to aws-security@amazon.com.

**Please do _not_ create a public GitHub issue** for security vulnerabilities.

## Logging policy

Log records at `INFO` and above carry only lengths, counts, ids and a short
stable hash for user queries, corpus text (including entity names) and model
output; the text itself is logged only at `DEBUG`. An error whose text may
come from a model provider or a library is logged at `INFO` and above by
what it is, never by what it says: its type, a service error code and HTTP
status, and for an error whose text holds no data the OS error number and
path or a decode position; a digest of its message; and where it happened
(the type chain and `file:line` frames). Its text and traceback are logged
only at `DEBUG`. An expected outcome with its own message (an unreachable
store or model, the deadline, a refusal) and a message the package composes
itself (a stage failure named by its cause) are logged with their text. A
library's own records at `INFO` and above (on the CLIs' handlers) keep their
logger, level and message template; their arguments and traceback are
replaced by digests, and a library's Python warning is logged by its
category and code location with its text digested. `DEBUG` logs therefore
contain user and corpus data, and every error's full message, and must be
handled with the same controls as that data.

The rule is applied where an error's text enters: an error kept to be
logged or shown later (a store write's recorded errors, a worker's outcome,
a saved evaluation result) is recorded by its name and digest, a store
error the package wraps shows the package's own context (the operation, the
URI) and its cause's name, and a static check over the package fails the
build on an error's text passed to an `INFO`+ record or kept for one. Some
text remains outside it, by design or out of the package's reach:

- File paths (corpus files, exports, caches, the config) appear at `INFO` and
  above and in error messages. Name corpus files accordingly.
- A third-party library that writes to stderr itself, formats text into its
  message template, or logs through a handler of its own is not filtered.
  Set its level in `logging.library_levels`.
- Result output holds corpus and model text (below).

Result output is not a log record and holds corpus and model text by design:
`run-ingestion` prints entity names, community titles and summaries and
claim descriptions, `run-rag` the answer and its sources, and the saved
evaluation results hold the questions and answers. An error's text in the
output, in the JSON status' `error`, in a stage's recorded `error_message`
(the run's `pipeline_metadata.json`, also its S3 checkpoint) and in a failed
answer's result follows the rule above: an unexpected failure's text is
replaced by its type and a generic message, an outage's (naming its
endpoint) is kept.
On AWS the CDK tasks send stdout and stderr to the same CloudWatch log group
as the structured records, so that log group holds corpus and model text:
control access to it as to the corpus. See the logging section of the
[user guide](docs/guides/configure.md#what-the-logs-contain).

## Dependency advisories

Use the committed lockfile for reproducible installations. The optional
`unstructured` extra requires a patched parser (`>=0.24.1`). Do not install an
older parser.

The following upstream advisories currently have no patched release identified
in the GitHub Advisory Database and remain open for tracking:

- [Ragas multimodal SSRF (GHSA-95ww-475f-pr4f)](https://github.com/advisories/GHSA-95ww-475f-pr4f):
  the built-in evaluator uses text metrics and does not invoke the affected
  `metrics.collections.multi_modal_faithfulness` helpers. Custom evaluators
  must not pass untrusted URLs or local paths to those helpers.
- [DiskCache pickle deserialization (GHSA-w8v5-vhqr-4h9v)](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v):
  Ragas brings DiskCache in transitively. The built-in evaluator does not
  configure Ragas's disk-cache backend. If custom code enables it, keep its
  directory writable only by the trusted application identity and never load
  cache files from untrusted sources.

The lockfile overrides one upstream version cap to reach a patched release:

- filelock is forced to `>=4.2.1` (`[tool.uv] override-dependencies` in
  `pyproject.toml`), which fixes the nine filelock advisories present in 3.29.4,
  including [GHSA-v4x8-3wr5-3wq8](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8)
  (`ReadWriteLock`) and [GHSA-69hj-f68j-2879](https://github.com/advisories/GHSA-69hj-f68j-2879)
  (`UnixFileLock`, which `datasets` and `huggingface-hub` use). The optional
  `unstructured` extra caps filelock below 4 in every release up to 0.27.x,
  but uses only a blocking `FileLock(path, timeout=-1)` around its spaCy model
  install; filelock 4.0's breaking change is limited to `SoftReadWriteLock`.
  The override applies to installs from this lockfile (`uv sync`); a plain
  `pip install` with the `unstructured` extra still resolves filelock 3.x.
  Remove the override once `unstructured` lifts the cap.

These usage restrictions do not patch the upstream packages. Reassess them when
changing evaluation or caching behavior and update when fixes become available.
