# Model Catalog

Which Bedrock models the framework knows, how it shapes requests for each
provider, and how to describe a model it does not know. For which models to
enable before a first run, see
[Required models](../getting-started/aws.md#required-models); for the two
model tiers and per-role overrides, see
[Model selection notes](#model-selection-notes) below.

## Model ids and capability records

Model-id keys accept any Bedrock model id, or an inference-profile id such as
`us.anthropic.claude-sonnet-5-5` (used as-is). The models below have a curated
capability record (as do the older Claude 3.x/4.x ids). Any other id still
works: `anthropic.claude-*` ids get the request shape of their generation,
`openai.gpt-*` ids the GPT shape, and other providers a conservative Converse
request (no reasoning or sampling parameters, 32K window, 4K output), each with
one WARNING. Describe or correct a model with `aws.bedrock.model_overrides`
(keys are capability-record fields such as `context_window_size` and
`max_output_tokens`). The config load does not check the entries: an unknown
key or a wrong value fails with `LanguageModelError` when the model's record
is first read, as its model is built or a budget is sized, and an entry for a
model no role uses is never read:

```yaml
aws:
  bedrock:
    model_overrides:
      "amazon.nova-pro-v1:0":
        context_window_size: 300000
        max_output_tokens: 10000
```

Embedding and rerank model ids work differently: with the Bedrock providers
(the default `providers.embedding.kind` / `providers.rerank.kind`) they are a
closed list, and `model_overrides` does not apply to them. With a local
provider any model name is accepted (see
[Local mode](../guides/local-mode.md)). `models.embedding.model_id` accepts
`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3` or
`cohere.embed-multilingual-v3`; `models.rerank` accepts
`cohere.rerank-v3-5:0` or `amazon.rerank-v1:0`. Any other id fails config
validation. The embedding dimension is written into the OpenSearch vector
mappings and checked against the model's record, so an unknown model has no
safe default; `models.embedding.dimension` picks among the
dimensions a listed model supports (Titan Embed V2: 256, 512 or 1024; unset
uses the largest). Adding a model is a code change: a member in
`EmbeddingModelId`/`RerankModelId` (`domain/models/config.py`) and a record in
`adapters/aws/bedrock_models.py`.

An embedding model is invoked on demand by the id in
`models.embedding.model_id` wherever it has on-demand throughput. Only in a
region without it does the factory resolve a cross-region inference profile,
the way language models do: the `global.` profile (with
`enable_global_profile`) or the geography profile. This applies to
`cohere.embed-v4:0`, which has on-demand throughput only in `us-east-1`,
`eu-west-1` and `ap-northeast-1`: there it keeps the bare id, in `us-west-2`
it runs through `us.`/`global.`, and in `ap-northeast-2` only through
`global.`. When no profile resolves where one is needed (global profile off,
or no `bedrock:ListInferenceProfiles` grant) the factory fails at model
creation with the remedy instead of on the first embed call. The index
metadata keeps the bare id, so switching between profile and on-demand
invocation needs no reindex.

## Curated language models

| Model id | Provider | Context / max output | Reasoning control |
| --- | --- | --- | --- |
| `anthropic.claude-sonnet-5-5` (default) | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-opus-5-5` | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-haiku-5-5` | Anthropic | 1M / 128K | adaptive, on by default; `effort` low–max |
| `anthropic.claude-sonnet-5`, `anthropic.claude-opus-5` | Anthropic | 1M / 128K | adaptive, always on; `effort` |
| `anthropic.claude-opus-4-8`, `anthropic.claude-opus-4-7` | Anthropic | 1M / 128K | adaptive, always on; `effort` low–max |
| `anthropic.claude-opus-4-6-v1` | Anthropic | 1M / 128K | opt-in (`--enable-thinking`), adaptive; `effort` low/medium/high/max |
| `anthropic.claude-sonnet-4-6` | Anthropic | 1M / 64K | opt-in, adaptive; `effort` low/medium/high/max |
| `openai.gpt-6.1-sol` | OpenAI | 1M / 131K | `reasoning.effort` low–max, always on |
| `openai.gpt-6-astra`, `openai.gpt-6-sol`, `openai.gpt-6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |
| `openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, `openai.gpt-5.6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |
| `openai.gpt-5.5`, `openai.gpt-5.4` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, always on |

All of these are inference-profile-only. Only OpenAI's proprietary GPT models
are offered; the open-weight `gpt-oss` models are not.

Four things differ for Claude 4.7-and-later models:

- **Inference profiles are mandatory.** They ship without `ON_DEMAND`
  throughput, so the bare model id is not invocable — a cross-region profile
  must resolve. Keep `enable_global_profile: true` and grant
  `bedrock:ListInferenceProfiles`; the adapter fails fast with the remedy if no
  profile resolves. Note that in `ap-northeast-2` only `global.` profiles exist
  for Claude 5 (no `apac.`), so disabling the global profile leaves no path.
- **`effort` replaces the thinking token budget.** `thinking_budget_tokens` is
  ignored for these models (the old `budget_tokens` request shape is rejected
  with a 400); set `bedrock.default_effort` / `bedrock.fast_effort` instead.
  A call uses `fast_effort` when its model is the fast tier's (`models.fast`,
  when that differs from `models.default`), otherwise `default_effort`; the shipped fast model,
  Claude Haiku 5.5, thinks adaptively, so `fast_effort` (default `low`) sets
  how much it reasons on fast-tier calls. A default-tier ingestion call uses
  `ingestion_effort` instead when it is set. Claude Sonnet 5.5 always thinks, so `--enable-thinking` is a
  no-op for it — depth is `effort` only. A level the model does not accept
  (e.g. `xhigh` on Opus 4.6 or Sonnet 4.6) fails fast.
- **Sampling parameters are dropped.** `temperature`/`top_k` are not accepted
  and are omitted from requests automatically; steer behaviour by prompting.
- **Token estimates are scaled.** Their tokenizer counts roughly 1x-1.35x the
  tokens of older Claude models for the same text and CountTokens rejects
  them, so the local estimate (~4 characters per token for Latin text) is
  multiplied by the record's `token_estimate_multiplier` (1.3) when sizing
  context budgets. Override it per model with
  `aws.bedrock.model_overrides`.

OpenAI GPT models differ from Claude in these ways:

- They always go through the Converse API on a `us.`/`global.` inference
  profile (no `apac.`/`eu.` geo profiles; keep `enable_global_profile: true`
  outside the US). The tier's effort (`bedrock.default_effort` /
  `fast_effort`) is sent as
  `reasoning: {effort: ...}` (the flat `reasoning_effort` field is rejected).
  GPT-5.6 and GPT-6.x answered a trivial prompt in roughly 10-25 s even at
  `effort: low`, so size timeouts and concurrency accordingly.
- No Anthropic-only fields are sent (`thinking`, `output_config`,
  `anthropic_beta`), and sampling parameters are omitted.
- Explicit prompt-cache markers are not sent: Converse supports only implicit
  caching for these models. Bedrock CountTokens does not support them, so the
  retrieval context budget uses the local token estimate.

Claude Fable 5 / 5.1 are not offered: they need a non-default account
data-retention mode (Data Retention API only), and accounts on the default mode
get `data retention mode 'default' is not available for this model` on every
call.

## Prompt caching

On Claude models that support explicit prompt caching, the
end of each system prompt is marked as a cache checkpoint: a `cachePoint` block
on the Converse API (every inference profile) and `cache_control` on
InvokeModel. A system prompt shorter than the model's minimum checkpoint size
(512 tokens on Claude Sonnet/Opus/Haiku 5.5 and Opus 5, 1024 on most others,
2048 on Claude 3.5 Haiku, 4096 on Claude Haiku 4.5 and Opus 4.5-4.7) gets no marker, because Bedrock would accept
it but cache nothing. Cache reads show up as `cache_read` in the response's
`usage_metadata.input_token_details`, and cached input tokens do not count
against the tokens-per-minute quota.

`aws.bedrock.query_prompt_cache_ttl: 1h` (default `5m`) keeps the checkpoint
of query-time chains for an hour. Only a chain whose system prompt reaches the
minimum size above has a checkpoint: with the shipped prompts, answer
generation, query entity extraction and the global-search reduce step on the
models with a 512-token minimum; on a 1,024- or 4,096-token model no shipped
query prompt is long enough. Answers that `run-eval` generates are query
chains too. Ingestion chains and evaluation judges reuse each prompt within
seconds and always use 5 minutes. A 1-hour cache write is billed at 2x the
input-token price (5 minutes: 1.25x) and a read costs the same, so it pays off
only at low query rates, where the gap between queries often exceeds 5
minutes.
The 1-hour TTL is sent only on Converse requests to Claude 4.5 and later
(`supports_1h_cache_ttl` in the capability record); an InvokeModel request
keeps the 5-minute checkpoint, because langchain-aws rewrites its
`cache_control` to `{"type": "ephemeral"}` and drops the TTL.

## Model selection notes

The `models` section names the model of each role; ids are checked against
the family's provider (`providers.llm.kind`, `providers.embedding.kind`,
`providers.rerank.kind`). Every LLM role runs on one of two tiers, so one line
switches the pipeline; a `models.roles` entry (a model id, or `default` /
`fast` for the other tier) wins over its tier:

| Tier | Roles |
| --- | --- |
| `default` | `fixing`, `graph_extraction`, `gleaning`, `claim_extraction`, `community_report`, `entity_extraction`, `answer_generation`, `evaluation` (with a provider other than Bedrock; see below) |
| `fast` | `chunking`, `translation`, `description_summary`, `query_translation`, `strategy_selection`, `query_rewrite`, `community_relevance`, `global_map`, `global_reduce`, `drift_primer`, `drift_action`, `answer_span` |

```yaml
models:
  default: "openai.gpt-6-sol"          # every default-tier role
  roles:
    answer_generation: "anthropic.claude-opus-5-5"   # one role pinned
    query_rewrite: default                          # moved to the other tier
```

- **The judge is its own model.** On Bedrock an unset
  `models.roles.evaluation` is `anthropic.claude-opus-5-5`, not the
  default tier: an LLM judge favours its own answers
  ([measurement and dated list prices](../concepts/defaults-and-evidence.md#the-evaluation-judge)).
  Opus 5.5 costs more per token than Sonnet 5.5, and only `run-eval`'s LLM
  judges call it. When the
  answer model is Opus 5.5 itself, the unset judge falls back to the default
  tier's model, or to Sonnet 5.5 if that is the answer model too; `run-eval`
  logs the judge it chose, and why, at INFO. With `--from-results` or
  `--baseline-results` the saved runs' answer models (their manifests'
  `models.answer_generation`) count as answer models too. Set
  `models.roles.evaluation: default` for the earlier judge (the answer
  model; `run-eval` then warns that it judges its own answers, as it does
  for any configured judge that is the answer model), or another model when
  Opus 5.5 is not enabled in your account or region. With another provider
  the judge stays on the `default` tier (and is warned about when it judges
  its own answers). The CDK stack grants every Bedrock model by default; a
  scoped `bedrock_model_arns` must list the judge model too.
- The embedding model (`models.embedding.model_id`) serves the indices, the
  queries and RAGAS; changing it or its
  `dimension` needs a re-index. On Bedrock, embedding and rerank ids are a
  closed list ([above](#model-ids-and-capability-records));
  other providers take any model name.
- Settings of an unselected provider (`providers.llm.temperature` with
  Bedrock models) are ignored with a WARNING.
- **Output cap.** `aws.bedrock.default_max_output_tokens` is 16384 because
  Bedrock reserves input + `max_tokens` against the tokens-per-minute quota
  when a request starts; asking for the model maximum throttles concurrent
  ingestion. Long-output prompts derive a higher floor that wins (by default
  23192 for graph extraction and gleaning, 18992 for document translation,
  29792 for claims), rising with their per-chunk limits. The short-answer
  query calls ask for less (`search.output_ceilings`, 4096 each: the
  router, the query entity and keyword extractions, the dynamic-selection
  report ratings; `evaluation.answer_span_output_ceiling` for the answer-span
  extractor), never less than the call's thinking budget plus 1024 (an
  effort of low budgets 2048, high 8192), so an `auto` query reserves about
  29k output tokens instead of 49k.
- Curated models, request shapes, prompt caching and
  `aws.bedrock.model_overrides` are described above.
