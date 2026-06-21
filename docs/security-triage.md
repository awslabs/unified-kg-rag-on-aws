# Security Scan Triage

Record of how GitLab Dependency-Scanning (SBoM) and SAST findings were assessed.
Update this when a new finding appears or a pinned version changes.

## SAST

| Finding | Location | Disposition |
|---|---|---|
| CWE-338 weak PRNG | `adapters/storage/neptune_indexer.py` (`random.uniform`) | **False positive.** The only `random` use is retry-backoff *jitter* (desynchronizes concurrent retries against a throttled endpoint). No tokens/secrets/nonces/keys derive from it, so `random` is intentional. Rule disabled in `.gitlab/sast-ruleset.toml`. |

## Dependency Scanning (transitive, in `uv.lock`)

None of the flagged vulnerable code paths are imported by this package — verified
by grepping the source. They are pulled transitively and are **not reachable**:

| CVE / advisory | Package | Why not reachable |
|---|---|---|
| CVE-2026-34070 (path traversal) | langchain-core | legacy `load_prompt` — we build prompts in code (`domain/prompts/*`), never load from disk/URL |
| CVE-2026-41481 (SSRF) | langchain-text-splitters | `HTMLHeaderTextSplitter.split_text_from_url` — unused; we parse local files via `ParserFactory` |
| CWE-1035 (path traversal / sandbox) | langchain | file-search middleware / loaders — unused |
| CVE-2025-69872 (pickle) | diskcache | transitive via `ragas`; we never construct a diskcache Cache |
| CVE-2025-50817 (code exec) | future | **withdrawn advisory**; transitive via `autograd`→`graspologic`; not a direct dep and not imported |
| CVE-2026-6587 (SSRF) | ragas | multi-modal faithfulness module — we use text-similarity evaluators only |
| CVE-2026-41488 / CVE-2026-26013 (SSRF) | langchain-openai / langchain | `image_url` token counting in `ChatOpenAI` — we use Bedrock, never ChatOpenAI |

### Posture

- Patched versions for the langchain-stack CVEs land in the **1.x** line; this
  project deliberately pins `langchain* < 1.0.0` (major-version stability). They
  are tracked for the eventual 1.x migration.
- `langsmith` and `pydantic-settings` were already bumped to their patched
  releases (the only fixes available within our constraints).
- `future` was removed as a *direct* dependency (never imported); it remains in
  the lock only transitively via `autograd`.
- Run `uv run pytest -m "not aws"` after any dependency bump.
- **`ragas` is the root of 4 of these** (it pulls `langchain-openai`, `openai`,
  `diskcache`). Removing the `ragas` evaluator would clear CVE-2026-6587,
  -41488, -26013, and -2025-69872 and drop 5 packages — kept for now as a
  first-class evaluator; revisit if these block.

## How to dismiss in GitLab

These are accepted (not fixed) — all are transitive, marked *Reachable: Not
available*, none are KEV, and the only available fixes are in the langchain 1.x
line we deliberately pin away from. In **Secure → Vulnerability report**, select
each finding above and set status **"Dismissed → Not applicable"** (or
"Acknowledged"), with a comment linking this file
(`docs/security-triage.md`). Re-evaluate when the project migrates to langchain
1.x or drops `ragas`.
