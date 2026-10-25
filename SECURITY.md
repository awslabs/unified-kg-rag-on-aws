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
`unstructured` extra requires Python 3.11+ and a patched parser (`>=0.24.1`).
Python 3.10 supports the core PDF/TXT/CSV/JSON formats; upgrade Python to use
the Markdown/HTML extra. Do not install an older parser to bypass this limit.

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

The following advisories have a patched release that this project cannot use
yet:

- Pillow 12.3.0 advisories (for example
  [GHSA-p4wg-p73j-c528](https://github.com/advisories/GHSA-p4wg-p73j-c528)):
  the fix, Pillow 13.0.0, requires Python 3.11+, so Python 3.11 and 3.12
  installs (and the container image) get 13.0.0 and Python 3.10 installs keep
  12.3.0. The package does not import Pillow itself (Bokeh and Ragas do);
  on Python 3.10, do not pass untrusted images to those libraries, or
  upgrade Python.
- [filelock ReadWriteLock (GHSA-v4x8-3wr5-3wq8)](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8)
  and two Windows-only filelock advisories (GHSA-9vhv-jxww-xfm5,
  GHSA-567q-5qvw-pf92): the fix, filelock 4.2.1, requires Python 3.11+, and
  the optional `unstructured` extra (up to its latest release, 0.27.34) caps
  filelock below 4, so the lock holds every install, including the container
  image, at 3.29.4. The core path gets filelock through `datasets` and
  `huggingface-hub`, which use `FileLock`/`SoftFileLock`, not the affected
  SQLite-backed `ReadWriteLock`. The soft-lock advisory applies on Windows
  only.

These usage restrictions do not patch the upstream packages. Reassess them when
changing evaluation or caching behavior and update when fixes become available.
