# 모델 카탈로그

> 🇬🇧 English version: [docs/models.md](./models.md)

이 문서는 프레임워크가 아는 Bedrock 모델, 공급자별 요청 형식, 프레임워크가 모르는
모델을 기술하는 방법을 설명합니다. 처음 실행하기 전에 활성화할 모델은
[사용자 가이드 §1 필수 모델](./user-guide.ko.md#필수-모델)을, 두 모델 계층과 역할별
재정의는 [§2.1 모델 선택 참고 사항](./user-guide.ko.md#모델-선택-참고-사항)을
참고하세요.

## 모델 ID와 기능 정보 레코드

모델 ID 키에는 어떤 Bedrock 모델 ID든 지정할 수 있고, `us.anthropic.claude-sonnet-5-5`
같은 추론 프로파일 ID도 지정할 수 있습니다(그대로 사용). 아래 모델에는 선별된 기능
정보 레코드가 있습니다(Claude 3.x/4.x 계열 ID도 마찬가지). 그 밖의 ID도 동작합니다.
`anthropic.claude-*` ID는 해당 세대의 요청 형식을, `openai.gpt-*` ID는 GPT 형식을,
다른 공급자는 보수적인 Converse 요청(추론·샘플링 파라미터 없음, 32K 컨텍스트 창, 4K
출력)을 쓰며, 각각 WARNING을 한 번 남깁니다. 모델을 기술하거나 정보를 고치려면
`aws.bedrock.model_overrides`를 쓰세요. 키는 `context_window_size`,
`max_output_tokens` 같은 기능 정보 레코드 필드이며, 알 수 없는 키는 즉시 오류가
납니다.

```yaml
aws:
  bedrock:
    model_overrides:
      "amazon.nova-pro-v1:0":
        context_window_size: 300000
        max_output_tokens: 10000
```

임베딩 모델 ID와 재순위화 모델 ID는 다르게 동작합니다. 정해진 목록의 값만 받으며
`model_overrides`도 적용되지 않습니다. `embedding_model_id`는
`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3`,
`cohere.embed-multilingual-v3` 중 하나를, `rerank_model_id`는
`cohere.rerank-v3-5:0` 또는 `amazon.rerank-v1:0`을 받습니다. 그 밖의 ID는 설정
검증에서 실패합니다. 임베딩 차원은 OpenSearch 벡터 매핑에 기록되고 모델 레코드와
대조하므로, 모르는 모델에는 안전한 기본값이 없습니다.
`indexing.opensearch.embedding_dimension`은 목록에 있는 모델이 지원하는 차원 중
하나를 고릅니다(Titan Embed V2: 256, 512, 1024. 지정하지 않으면 가장 큰 값). 모델을
추가하려면 코드를 바꿔야 합니다. `EmbeddingModelId`/`RerankModelId`
(`domain/models/config.py`)에 멤버를, `adapters/aws/bedrock_models.py`에 레코드를
추가합니다.

## 선별된 언어 모델

| 모델 ID | 공급자 | 컨텍스트 / 최대 출력 | 추론 제어 |
| --- | --- | --- | --- |
| `anthropic.claude-sonnet-5-5`(기본값) | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-opus-5-5` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-haiku-5-5` | Anthropic | 1M / 128K | adaptive, 기본으로 켜짐. `effort` low–max |
| `anthropic.claude-sonnet-5`, `anthropic.claude-opus-5` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` |
| `anthropic.claude-opus-4-8`, `anthropic.claude-opus-4-7` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-opus-4-6-v1` | Anthropic | 1M / 128K | 선택(`--enable-thinking`), adaptive. `effort` low/medium/high/max |
| `anthropic.claude-sonnet-4-6` | Anthropic | 1M / 64K | 선택, adaptive. `effort` low/medium/high/max |
| `openai.gpt-6.1-sol` | OpenAI | 1M / 131K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-6-astra`, `openai.gpt-6-sol`, `openai.gpt-6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, `openai.gpt-5.6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-5.5`, `openai.gpt-5.4` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |

모두 추론 프로파일로만 호출할 수 있습니다. OpenAI는 독점 GPT 모델만 지원하며, 오픈
웨이트 `gpt-oss` 모델은 지원하지 않습니다.

Claude 4.7 이후 모델은 네 가지가 다릅니다.

- **추론 프로파일이 필수입니다.** 이 모델들은 `ON_DEMAND` 처리량 없이 제공되므로 모델
  ID만으로는 호출할 수 없고, 교차 리전 프로파일을 찾을 수 있어야 합니다.
  `enable_global_profile: true`를 유지하고 `bedrock:ListInferenceProfiles` 권한을
  부여하세요. 프로파일을 찾지 못하면 어댑터가 해결 방법을 알리며 즉시 실패합니다.
  `ap-northeast-2`에는 Claude 5용 프로파일이 `global.`뿐이고 `apac.`은 없으므로,
  글로벌 프로파일을 끄면 호출할 경로가 없습니다.
- **`effort`가 thinking 토큰 예산을 대신합니다.** 이 모델들에서는
  `thinking_budget_tokens`를 무시합니다(구형 `budget_tokens` 요청 형식은 400으로
  거부됨). 대신 `bedrock.default_effort` / `bedrock.fast_effort`를 지정하세요. 호출
  모델이 `fast_model_id`이고 그 값이 `default_model_id`와 다르면 `fast_effort`를, 그
  밖에는 `default_effort`를 씁니다. 기본 제공 빠른 모델인 Claude Haiku 5.5는 adaptive
  thinking을 하므로, `fast_effort`(기본 `low`)가 빠른 계층 호출의 추론 강도를
  정합니다. 기본 계층 수집 호출은 `ingestion_effort`가 지정되어 있으면 그 값을 씁니다.
  Claude Sonnet 5.5는 항상 추론하므로 `--enable-thinking`은 효과가 없으며, 추론 강도는
  `effort`로만 조절합니다. 모델이 받지 않는 수준(예: Opus 4.6이나 Sonnet 4.6의
  `xhigh`)은 즉시 실패합니다.
- **샘플링 파라미터를 뺍니다.** `temperature`/`top_k`를 받지 않으므로 요청에서 자동으로
  뺍니다. 동작은 프롬프트로 조절하세요.
- **토큰 추정치를 보정합니다.** 이 모델들의 토크나이저는 같은 텍스트를 구세대 Claude
  모델보다 약 1~1.35배 많은 토큰으로 세며 CountTokens도 이 모델들을 거부합니다. 그래서
  컨텍스트 예산을 계산할 때 로컬 추정치(라틴 문자 텍스트 약 4자당 1토큰)에 레코드의
  `token_estimate_multiplier`(1.3)를 곱합니다. 모델별 값은
  `aws.bedrock.model_overrides`로 바꿀 수 있습니다.

OpenAI GPT 모델은 Claude와 다음이 다릅니다.

- 항상 `us.`/`global.` 추론 프로파일에서 Converse API로 호출합니다(`apac.`/`eu.`
  지역 프로파일이 없으므로 미국 밖에서는 `enable_global_profile: true`를 유지하세요).
  계층의 추론 강도(`bedrock.default_effort` / `fast_effort`)는
  `reasoning: {effort: ...}`로 보냅니다(평면 필드 `reasoning_effort`는 거부됨).
  GPT-5.6과 GPT-6.x는 `effort: low`에서도 아주 짧은 프롬프트에 답하는 데 약 10-25초가
  걸렸으므로, 타임아웃과 동시성을 이에 맞춰 정하세요.
- Anthropic 전용 필드(`thinking`, `output_config`, `anthropic_beta`)는 보내지 않으며,
  샘플링 파라미터도 뺍니다.
- 명시적 프롬프트 캐시 마커를 보내지 않습니다. 이 모델들은 Converse에서 암묵적
  캐싱만 지원합니다. Bedrock CountTokens도 이 모델들을 지원하지 않으므로 검색 컨텍스트
  예산은 로컬 토큰 추정치를 씁니다.

Claude Fable 5 / 5.1은 지원하지 않습니다. 기본값이 아닌 계정 데이터 보존 모드(Data
Retention API로만 설정 가능)가 필요하며, 기본 모드 계정에서는 모든 호출이
`data retention mode 'default' is not available for this model`로 실패합니다.

## 프롬프트 캐싱

명시적 프롬프트 캐싱을 지원하는 Claude 모델에서는 각 시스템 프롬프트의 끝을 캐시
체크포인트로 표시합니다. Converse API(모든 추론 프로파일)에서는 `cachePoint` 블록을,
InvokeModel에서는 `cache_control`을 씁니다. 시스템 프롬프트가 모델의 최소 체크포인트
크기(Claude Sonnet/Opus 5.5와 Opus 5는 512토큰, 대부분은 1024토큰, Claude Haiku 4.5와
Opus 4.5-4.7은 4096토큰)보다 짧으면 표시하지 않습니다. Bedrock이 요청은 받지만
아무것도 캐시하지 않기 때문입니다. 캐시 읽기는 응답의
`usage_metadata.input_token_details`에 `cache_read`로 나타나며, 캐시된 입력 토큰은
분당 토큰 할당량에 포함되지 않습니다.
