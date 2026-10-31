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
output; the text itself is logged only at `DEBUG`, and exception messages do
not include model output. `DEBUG` logs therefore contain user and corpus data
and must be handled with the same controls as that data. See the logging
section of the [user guide](docs/user-guide.md#what-the-logs-contain).

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
