# Guardrail

Setting `aws.bedrock.guardrail.identifier` screens two texts of every query
with the `ApplyGuardrail` API; the guardrail must exist in
`aws.bedrock.region_name`. The principal needs `bedrock:ApplyGuardrail` on
the guardrail's ARN ([IAM permissions](../getting-started/aws.md#iam-permissions)).
To create the guardrail with the CDK stack, see
[iac/README.md: Guardrail](../../iac/README.md#guardrail).

## Query and answer screening

- **What is screened.** The raw user query, once, before memory, rewriting,
  routing and retrieval (`check_input`), and the final answer (`check_output`;
  a streamed answer then arrives as one chunk). With `check_output`, the
  model-written search results (`metadata.synthesized`: DRIFT's
  intermediate answers, global search's key points or summary) are screened
  too, one check each, a refused one dropped from `search_results`. Only
  these are billed.
- **What happens.**
  - A blocked query gets the guardrail's message, retrieves nothing, writes
    no memory and sets `RAGOutput.metadata.blocked`. A masked query continues
    with the masked text only.
  - A blocked or masked answer is replaced by the guardrail's text (blocked:
    no `sources` either). Masking covers the answer, not its sources (corpus
    text as retrieved): keep text a caller must not see out of the corpus.
  - Interventions are in `RAGOutput.metadata.guardrail`, the WARNING log and
    the trace. A guardrail API error fails the query, or with
    `processing.ignore_errors` gives the degraded answer with
    `metadata.error`: never unscreened.
- **Not on model calls.** Model calls inside a query run unguarded: attached
  to a call, a guardrail evaluates the whole prompt, so a `PROMPT_ATTACK`
  filter judges retrieved text as user input (false blocks, cost growing
  with context)
  ([prompt attacks](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html)).
  The contextual grounding check is opt-in and runs on the answer's check
  only (below).
- **Ingestion and the evaluation judges run unguarded** (`run-eval`'s answers
  come from the screened query path). A PII guardrail that anonymizes
  `NAME` would rewrite entity names to `{NAME}` (merging distinct people), and
  a `PROMPT_ATTACK` filter can block instruction-like corpus text. On the
  query path, `NAME` anonymization also strips the name a question retrieves
  by, so the baseline guardrail `iac/` creates anonymizes email, phone and
  card numbers but not `NAME` ([iac/README.md](../../iac/README.md#guardrail)).

## Contextual grounding check

The check is opt-in and scores English, French and Spanish only.
`aws.bedrock.guardrail.contextual_grounding.enabled: true` sends the context
the answer model was given (`grounding_source`) and the question it answered
(`query`) with the answer's check, so the guardrail's
[contextual grounding policy](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)
scores whether the answer is grounded in that context and relevant to the
question, and blocks it below its threshold (`iac/`:
`-c guardrail_contextual_grounding=true`, `guardrail_grounding_threshold`,
default 0.7). Needs `check_output`.

- **Language.** The Classic tier (the default, and the one `iac/` creates)
  scores English, French and Spanish only: answers in other languages,
  Korean included, score low and are blocked. The Standard tier covers
  more languages but needs a cross-Region guardrail profile and is offered
  in fewer regions. Measure the threshold on your own questions first.
- A blocked answer takes the refusal path above, with
  `reason: contextual_grounding` in its `metadata.guardrail` record. Each
  checked answer carries `metadata.grounding` (`checked`, `scores`,
  `failed`, `blocked`) and the `kg_rag.guardrail` span its scores. An answer that
  fails without being blocked (a detect-only policy), and a guardrail that
  returns no scores (no grounding policy), are logged at WARNING.
- **Limits.** Bedrock takes 100,000 characters of reference, 1,000 of
  question and 5,000 of answer. A longer context keeps its highest-ranked
  sections and a longer question its head. A longer answer is screened by
  the other policies only: `metadata.grounding.skipped_reason` is
  `answer_too_long`, with a WARNING.
- **Cost.** Billed per text unit (1,000 characters) of reference, question
  and answer, on top of the other policies: a full context is about 100
  text units per answer
  ([pricing](https://aws.amazon.com/bedrock/pricing/)). `ApplyGuardrail`'s
  per-second text-unit quota (Service Quotas, guardrail region) throttles
  concurrent queries with long contexts first.

## Custom code

Code calling `setup_chain` or `get_model` for non-query work
passes `model_purpose=ModelPurpose.INGESTION` (or `EVALUATION`;
`from unified_kg_rag import ModelPurpose`). Unmarked
calls default to `QUERY`: the query-time transient-error retry, and neither
the response cache nor the ingestion effort.
