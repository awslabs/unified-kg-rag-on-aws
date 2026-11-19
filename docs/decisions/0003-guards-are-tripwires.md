# 0003. The single-owner guards are tripwires, not coverage

Status: accepted (recorded 2026-11-19).

**Context.** `tests/unit/test_single_owner_guards.py` fails when code outside
a concept's owner matches a finder for that concept (a second JSON reader of
model output, a second script table, ...). A held-out review set of planted
copies written outside the finders' vocabulary mostly passed them, and
widening the finders idiom by idiom added false positives on ordinary
integers.

**Decision.** A finder matches the forms copies took and the forms its pins
and planted files spell; review owns the rest. A finder is widened only
where one rule covers a whole family (the decoding modules of G2 and G10),
never one idiom at a time for a form review found: the copy is fixed
instead. The planted files are tuning data, not a held-out set.

**Rejected.** Treating the guards as coverage and widening them after each
missed form.

**Evidence.** [CONTRIBUTING.md, the guards](../../CONTRIBUTING.md#single-owner-guards)
("The guards are tripwires for the forms they know, not coverage").
