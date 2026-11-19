# 0008. npm audit gates with an expiring list of accepted advisories

Status: accepted (recorded 2026-11-19).

**Context.** The `console-web` job runs `npm audit` over the console's lock
and fails on an advisory at moderate or above. Some advisories have no
fixed release and do not reach the built client (a build tool's).

**Decision.** `.github/npm-audit-accepted.txt` lists the accepted
advisories, each with an expiry date, the package and version, and why it
does not apply; `.github/scripts/check_npm_audit.py` fails on any other, and
on a listed one past its date. Only an advisory with no fixed release that
does not reach the built client is accepted, and its line goes when a fixed
release is locked.

**Rejected.** A higher audit level or `continue-on-error` (hides real
advisories); no exceptions (every pull request blocked on an unfixable
build-tool advisory).

**Evidence.** [CONTRIBUTING.md](../../CONTRIBUTING.md#checks-to-run-before-pushing) and the
list's own header.
