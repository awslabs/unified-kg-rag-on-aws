# Security Policy

## Reporting a Vulnerability

If you discover a potential security issue in this project, we ask that you notify
AWS/Amazon Security via our
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or directly via email to aws-security@amazon.com.

**Please do _not_ create a public GitHub issue** for security vulnerabilities.

## Dependency advisories

Use the committed lockfile for reproducible installations. The optional
`unstructured` extra requires Python 3.11+ and a patched parser (`>=0.24.0`).
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

These usage restrictions do not patch the upstream packages. Reassess them when
changing evaluation or caching behavior and update when fixes become available.
