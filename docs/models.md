# Model Catalog

> 🇰🇷 한국어판: [docs/models.ko.md](./models.ko.md)

Which Bedrock models the framework knows, how it shapes requests for each
provider, and how to describe a model it does not know. For which models to
enable before a first run, see the
[User Guide §1 Required models](./user-guide.md#required-models); for the two
model tiers and per-role overrides, see
[§2.1 Model selection notes](./user-guide.md#model-selection-notes).

## Model ids and capability records

Model-id keys accept any Bedrock model id, or an inference-profile id such as
`us.anthropic.claude-sonnet-5-5` (used as-is). The models below have a curated
capability record (as do the older Claude 3.x/4.x ids). Any other id still
works: `anthropic.claude-*` ids get the request shape of their generation,
`openai.gpt-*` ids the GPT shape, and other providers a conservative Converse
request (no reasoning or sampling parameters, 32K window, 4K output), each with
one WARNING. Describe or correct a model with `aws.bedrock.model_overrides`
(keys are capability-record fields such as `context_window_size` and
`max_output_tokens`; an unknown key fails fast):

```yaml
aws:
  bedrock:
    model_overrides:
      "amazon.nova-pro-v1:0":
        context_window_size: 300000
        max_output_tokens: 10000
```

Embedding and rerank model ids work differently: they are a closed list, and
`model_overrides` does not apply to them. `embedding_model_id` accepts
`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3` or
`cohere.embed-multilingual-v3`; `rerank_model_id` accepts
`cohere.rerank-v3-5:0` or `amazon.rerank-v1:0`. Any other id fails config
validation. The embedding dimension is written into the OpenSearch vector
mappings and checked against the model's record, so an unknown model has no
safe default; `indexing.opensearch.embedding_dimension` picks among the
dimensions a listed model supports (Titan Embed V2: 256, 512 or 1024; unset
uses the largest). Adding a model is a code change: a member in
`EmbeddingModelId`/`RerankModelId` (`domain/models/config.py`) and a record in
`adapters/aws/bedrock_models.py`.

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
  A call uses `fast_effort` when its model is `fast_model_id` (and that differs
  from `default_model_id`), otherwise `default_effort`; the shipped fast model,
  Claude Haiku 5.5, thinks adaptively, so `fast_effort` (default `low`) sets
  how much it reasons on fast-tier calls. A default-tier ingestion call uses
  `ingestion_effort` instead when it is set. Claude Sonnet 5.5 always thinks, so `--enable-thinking` is a
  no-op for it — depth is `effort` only. A level the model does not accept
  (e.g. `xhigh` on Opus or Sonnet 4.6) fails fast.
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
(512 tokens on Claude Sonnet/Opus 5.5 and Opus 5, 1024 on most others, 4096 on
Claude Haiku 4.5 and Opus 4.5-4.7) gets no marker, because Bedrock would accept
it but cache nothing. Cache reads show up as `cache_read` in the response's
`usage_metadata.input_token_details`, and cached input tokens do not count
against the tokens-per-minute quota.
