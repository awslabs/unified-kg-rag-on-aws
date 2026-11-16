# 질의

> 🇬🇧 English version: [query.md](./query.md)

`run-rag --query "..."`는 질문 하나에 답하고, `--interactive`는 대화를 이어 갑니다.
전략은 `--search-strategy`로 고르며(기본값 `auto`, LLM 라우터), 모든 답변은 답변 모델이
실제로 본 출처를 함께 돌려줍니다. 어떤 전략을 고를지와 답변 형식은
[검색](../concepts/retrieval.ko.md)에, 질의가 읽는 인덱스는
[인덱스 접미사](../concepts/scopes-and-namespaces.ko.md#인덱스-접미사)에 있습니다.

모든 플래그와 기본값은 [CLI 레퍼런스](../reference/cli.md#run-rag)(영문)에 있습니다.

## 예시

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy and search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# Retrieval only, JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# 개수, 순위, 경로를 위한 그래프 질의(선택 사항)
run-rag --query "Norland의 구매처에 공급하는 벤더는 몇 곳입니까?" \
  --search-strategy structured --config-path config.yaml

# HippoRAG 2 PageRank (opt-in; auto never routes to it)
run-rag --query "Where was the founder of the company that acquired X born?" \
  --search-strategy ppr --config-path config.yaml

# Verbose (entities, every source with its score and text, metrics) with
# attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

텍스트 출력에는 다음이 나옵니다.

- **답변** 뒤에는 출처를 한 줄로 보여 줍니다. 텍스트 유닛이 나온 파일과 모델이 읽은 그래프
  요소의 유형별 개수입니다(`Sources (13): files: acme.txt, globex.txt; 7 relationships, 4
  entities, 1 community report`). `--verbose`를 주면 대신 출처마다 유형, 파일이나 이름,
  점수, 텍스트 앞부분을 나열하고 질의 처리 정보와 지표를 이어서 보여 줍니다. `--interactive`
  모드에서도 같습니다.
- **`--mode search`**는 검색 결과를 출력합니다. 유형, 파일(텍스트 유닛이며 파일에 청크가
  여럿이면 청크 번호도) 또는 이름(엔티티, 커뮤니티 리포트, 관계는 양 끝), 점수, 텍스트
  앞부분이며 `--verbose`가 없으면 처음 20개만 보여 줍니다.
- **부분 답변**(global 검색의 맵 호출이 스로틀링된 경우처럼 실패한 단계 없이 만든 답변)은
  답변 뒤 한 줄에 그렇다고 알립니다(`Partial answer: given without steps that failed:
  global_map (1/3 reports unrated)`). 질의 한도 때문에 건너뛴 단계는 따로 한 줄로
  알리고(`Skipped to stay within the query's limits: ...`), 한도 때문에 컨텍스트가 없어
  거절한 답변은 `Declined: the query's limits ...`로 알리며 올릴 설정(`search.query_max_model_calls`,
  `search.query_max_input_tokens` 또는 그 `UKR__` 변수)을 함께 알려 줍니다. 인용 계보(lineage) 읽기가
  실패하면 `Some citations name no document`가 붙습니다. `--verbose`는 답변이 인용할 수 있는 모델 작성 섹션(global 검색의
  핵심 요점, DRIFT의 중간 답변)도 나열합니다.
- **`--output-format json`**은 결과를 문서 하나로 출력합니다. 답변의 `search_results`에서는
  `sources`와 겹치는 `results`를 빼며, `--verbose`를 주면 남깁니다(컨텍스트 예산에서 잘린
  결과도 들어 있습니다). 콘솔과 MCP 서버가 보여 주는 `degraded`, `degraded_error`,
  `citations_unresolved`, `error_class`와 답변이면 `written`(모델 작성 섹션)도 담습니다.

## 필터

`--filters key:value ...`(또는 `RAGInput.filters`)는 검색이 순위를 매길 대상을
좁힙니다. 키·`:`·값 중 하나가 없는 쌍, 두 번 준 키, 전략이 읽는 어떤 인덱스도 선언하지
않은 키는 사용법 오류(종료 코드 2)입니다. 목록(값 중 하나와 일치)이나 범위는
`--filters-json`에 `RAGInput.filters`와 같은 JSON 객체 하나로 줍니다
(`'{"start_date": {"gte": "2022-07-01"}, "category": ["a", "b"]}'`). 검사 방식은 같고,
`--filters`와 함께 주면 키마다 둘 중 한쪽에만 둡니다.

- 필터는 해당 필드를 선언한 저장소에만 적용됩니다(`type:person`은 텍스트 단위를
  거르지 않음). 선택한 전략이 읽는 저장소 어디에도 선언되지 않은 키(예: `category`,
  `naive`나 `global`의 `type`)는 필터 가능한 키 목록과 함께 `InvalidFilterError`를
  일으킵니다. `auto`는 라우팅될 수 있는 전략 중 키를 모두 적용하는 전략 사이에서만
  라우팅합니다.
- `attr_<key>`는 문서 `filters` 메타데이터의 `<key>` 항목과 비교합니다. OpenSearch는 이를
  엄격하게(fail-closed) 적용하므로 속성이 없는 항목은 제외됩니다. 여러 문서에서 나온
  엔티티, 관계, 주장, 커뮤니티 리포트는 그 문서 중 하나가 일치하면 일치합니다.
- 엔티티와 주장의 `type`, 주장의 `status`는 소문자로 저장되고 필터 값은 소문자 형태로도
  비교하므로 `type:PERSON`도 `person`과 일치합니다. 불리언·숫자 필드는 그 유형의 값을 받습니다(`in_schema:false`, `rank:2`).

저장소별로 필터할 수 있는 모든 필드, 값의 유형, 날짜, 전략별로 적용되는 키, 그래프
확장의 필터 처리는 [필터 레퍼런스](../reference/filters.md)(영문)에 있습니다. 필터는 접근
제어 수단이 아닙니다([접근 제어 경계가 아님](../concepts/scopes-and-namespaces.ko.md#접근-제어-경계가-아님)).

## 개인화 PageRank(`ppr`)

`--search-strategy ppr`은 HippoRAG 2의 검색으로 답합니다. 질문의 관계·문단 검색
결과를 시작점으로 그 주변 그래프에 PageRank를 실행해 문단 순위를 정합니다(질의 엔티티
추출 없이 LLM 호출 1회). 관계 벡터 인덱스와 그래프가 필요하고 커뮤니티 보고서는 필요하지
않습니다. 설정은 `search.ppr_search`이고, 융합 버킷(`ppr_passages`,
`ppr_seed_text_units`, `ppr_facts`)에는 `search.fusion.fusion_weights`를 쓸 수 있습니다.
비용과 지연 시간이 적은 선택지입니다. 다중 홉 질문에서 답변은 `local`, `mix`와 오차 범위
안에서 같았고 비용은 그 일부였습니다([근거](../concepts/defaults-and-evidence.md#the-ppr-strategy), 영문).
어디서 나은지 측정으로 보이기 전까지 `auto`는 이 전략을 고르지 않으므로 직접 지정하세요. `local`은
기본값으로 같은 방식으로 텍스트 단위 순서를 정합니다(`search.local_search.text_unit_ranking: ppr`.
`support`는 각 단위를 인용하는 확장 엔티티 수로 정함). DRIFT의 후속 local 검색은 `support`를 유지합니다
(`search.drift_search.text_unit_ranking`. `ppr`로는 측정하지 않음). [검색](../concepts/retrieval.ko.md)과
[설계 §6.4](../design/retrieval-hipporag-raptor.md#64-hipporag-2-personalized-pagerank-ppr-local-ranking)(영문)을
참고하세요.

## 재순위화 상태

모든 질의는 재순위화 결과를 `SearchResult`와 RAG 출력의 `metadata.rerank`(따라서
`--output-format json`에도)에 남깁니다. `status`, `reason`, `stages`로 구성되며,
`stages`에는 재순위화 단계마다 `status`, `scope`(`all`, 또는 재순위화한 결과 유형. 예:
`local`, `drift`, `mix`, `hybrid`의 `text`), `reason`이 있습니다.

| 상태 | 의미 |
|---|---|
| `applied` | 재순위화 모델이 후보 순서를 바꿨습니다. |
| `disabled` | `providers.rerank.kind: none`. |
| `unavailable` | 재순위화 모델을 만들 수 없습니다(초기화 실패는 ERROR로 기록, 또는 크로스 인코더의 `[cross-encoder]` extra 미설치). |
| `skipped` | 재순위화할 대상이 없습니다: 후보 없음, 재순위화 대상 유형의 결과 없음, 질의 텍스트 없음. |
| `failed` | 호출이 예외를 냈거나(`bedrock:Rerank` 거부, 모델이 없는 리전, 연결할 수 없는 엔드포인트) 쓸 수 있는 결과를 돌려주지 않아 융합 순서를 유지했습니다. |

- `failed`와 `unavailable`의 `reason`은 메시지 없이 AWS 오류 코드나 예외 유형만 밝힙니다.
- 기본 제공 전략은 질의마다 한 번 재순위화합니다. 같은 기록에 여러 번 융합하는 호출자는
  융합마다 단계가 생기고, 최상위 `status`는 `failed` > `unavailable` > `applied` >
  `skipped` > `disabled` 순으로 가장 나쁜 단계의 값입니다. 모델이 순서를 바꾼 결과에는
  `metadata.reranked`가 표시됩니다.
- Bedrock 권한이나 리전 문제는 `unavailable`이 아니라 첫 질의의 `failed`로 나타납니다.
  `search.reranking.required: true`이면 이런 질의를 융합 순서로 답하지 않고
  `RerankModelError`로 실패시킵니다.

## 질의 제한 시간

질의 하나(`run-rag`, `invoke`/`ainvoke`, `search`/`asearch`, 스트리밍)는 전체
`search.query_timeout_seconds`(기본 180초) 안에 끝나야 합니다. 이 시간을 넘기면
`QueryTimeoutError`(`unified_kg_rag`에서 내보냄, 내장 `TimeoutError`이기도 함)로 실패하고,
진행 중인 모델 호출과 저장소 요청은 취소됩니다.

- `run-rag`는 메시지를 출력하고 `1`로 종료합니다. `run-eval`은 그 질의를 실패로 세고
  (`error_message` 기록, 채점 제외) 나머지 질의를 계속 처리합니다.
- `processing.ignore_errors`를 켜면 오류 답변을 대신 돌려주고, 메시지는
  `search_results.metadata.error`에 남습니다.
- 스트리밍 답변은 모델 호출 후 `search.model_timeout_seconds` 안에 첫 청크를 보내야 합니다.
  청크가 오기 시작하면 전체 제한 시간이 적용됩니다. 어떤 청크든 첫 청크로 칩니다. 추론 모델이
  스트리밍하는 추론(답변에서는 빠짐)도 이 대기를 끝냅니다. 답변 텍스트가 오기 전에 일시적
  오류(Bedrock 스트림의 스로틀링 이벤트나 끊긴 연결 포함)가 나면 어느 제공자든 스트리밍하지
  않는 질의처럼 재시도합니다.
- `null`로 두면 두 제한을 모두 끕니다.

질의의 모델 호출(답변 생성과 그 밖의 질의 시점 LLM 호출, 질의 임베딩, Bedrock 재순위화)은
전용 클라이언트를 씁니다. 요청마다 응답을 `search.model_timeout_seconds`(기본 90초)까지
기다리고 botocore 시도는 `search.model_max_attempts`(기본 2회)라서, 멈춘 Bedrock 호출은
약 3분 뒤 포기합니다. 수집과 평가 판정 호출은 읽기 제한 시간 330초와 6회 시도를
유지합니다. OpenAI 호환 서버나 Ollama 서버에서는 질의 호출의 요청 제한 시간이
`providers.llm.timeout_seconds` / `providers.embedding.timeout_seconds`이되
`search.model_timeout_seconds`를 넘지 않습니다. 로컬 서버는 요청을 대기열에 쌓으므로
`backend: local`은 모델 제한 300초, 질의 제한 900초를 미리 채웁니다(여기서 `null`은
미리 채운 값으로 바뀌므로 큰 숫자를 지정).

스트리밍하지 않는 답변은 한 번에 도착하므로 `search.model_timeout_seconds`가 모델이 답변을
생성하는 시간의 상한이기도 합니다. 긴 답변이나 높은 추론 강도(effort)에는 이 값과 질의
제한 시간을 함께 올리세요. 기본값은 실제 AWS에서 측정한 질의 시간에서
정했습니다([근거](../concepts/defaults-and-evidence.md#query-deadlines-and-model-timeouts), 영문).
조정 방법은 [운영 런북](../operations.md#query-deadlines-and-model-timeouts)(영문)에 있습니다.

### 모델 호출 수와 입력 토큰 한도

제한 시간은 질의 한도(`QueryLimits`) 가운데 하나입니다. 기본으로 꺼져 있는 한도가 둘 더
있습니다. `search.query_max_model_calls`는 질의의 모든 모델 호출(라우터, 질의 처리, 전략
자체의 호출, 답변) 수를, `search.query_max_input_tokens`는 그 호출들의 입력 토큰을
제한합니다. 입력 토큰은 호출이 시작될 때 프롬프트의 로컬 추정치로 셉니다. 한도를 넘는
호출은 요청을 보내기 전에 거부됩니다.

- `global`의 평가·맵 호출, `drift`의 확장·초안·후속 호출, `iterative`의 판정 호출은 답변
  호출을 남길 수 있는 만큼만 실행합니다. 들어가지 못한 호출은 빼고
  `metadata.degraded`(사유 `query_limit`)에 기록합니다. global은 관련성이 높은 보고서 묶음부터
  맵 호출을 하고(`use_dynamic_selection`에서는 평가하지 못한 보고서를 평가에 실패한 보고서처럼
  남기고 그 깊이에서 하향 탐색을 멈춥니다),
  DRIFT는 점수가 높은 후속 질의부터 실행하고(근거 구획이 하나도 몫에 들어가지 않는 후속 호출도
  뺍니다), iterative는 라운드를 멈춥니다
  (`stop_reason: query_limit`). 계획한 호출은 계측기가 세는 방식 그대로, 즉 프롬프트와 호출이
  담는 질문, 읽을 내용을 더해 셉니다.
- `auto`는 라우터 호출 뒤에 남은 호출 수 안에 최소 호출 수(`fewest_model_calls`, 그 전략이 읽는 질의
  추출, 질문에 필요한 번역, 답변)가
  들어가는 전략으로만 분기하고, 제외한 전략은 `metadata.routing.over_limits`에 남습니다.
  그런 전략이 없으면 번역과 추출을 빼고도 들어가는 전략으로 분기하고, 그 번역과 추출은 선택적
  호출로 건너뜁니다.
  두 한도 가운데 하나라도 있으면 미리 시작하는 추출은 하지 않습니다.
- 답변 컨텍스트는 남은 입력 토큰에서 답변 프롬프트 자체(템플릿과 지시문)와 질문을 빼고, 다시
  `search.token_manager.context_window_headroom_ratio`만큼 뺀 크기로 줄입니다. 실제로 만든
  컨텍스트(구획 머리글과 구획 사이 빈 줄 포함)는 모델의 토큰 계수기가 추정치보다 많이 세든 적게
  세든 계측기 추정치로 이 크기에 맞춰 채우고, 넘지 않습니다. 들어가는 구획이 하나라도 있으면
  비우지 않습니다. 호출을 펼치는 전략도 답변 프롬프트
  몫의 토큰을 답변용으로 남깁니다. 이렇게 줄인 컨텍스트(빠진 구획과 잘린 구획)는 `metadata.degraded`(단계 `context_optimization`, 사유
  `query_limit: N of M sections cut`)에 남습니다.
- 호출 하나는 답변용으로 남기고, 입력 토큰 한도가 있으면 답변의 입력(프롬프트, 질문, 최소
  컨텍스트)도 남깁니다. 둘 중 하나라도 쓰게 될 선택적 호출은 보내지 않고, 그 단계는 빼고 답하며
  `degraded`(사유 `query_limit`)에 남깁니다. 선택적 호출은 질의 번역과 추출, 후속 질문
  재작성(이때 후속 질문을 쓰인 그대로 검색), structured 전략의 계획 호출(대체 전략이
  답함), 출력 보정(`models.roles.fixing`. 해석하지 못한 응답은 보정하지 못한 응답으로 처리)입니다.
  `drift`를 지정한 질의는 후속 호출 하나의 몫도 남기고(그 호출이 없으면 DRIFT는 아무것도 찾지
  못합니다), 입력 토큰 한도가 있으면 그 호출의 최소 입력도 남깁니다. 검색(`asearch`, `kg_search`, `run-rag --mode search`)은 답변을 만들지 않으므로
  답변 몫을 남기지 않으며, 이때도 한도를 넘는 선택적 호출은 같은 방식으로 건너뜁니다.
- 입력 토큰 한도에서는 호출을 펼치는 전략이 설정 상한이 아니라 답변이 실제로 읽을 만큼(답변 프롬프트,
  DRIFT가 지금까지 받은 답변, 최소 컨텍스트)만 답변용으로 남기고, DRIFT의 후속 호출은 남은 토큰을
  나눠 씁니다. 후속 호출마다 만든 컨텍스트는 제 몫을 넘지 않으며, 그래도 계측기가 거부한 후속
  호출은 들어가지 못한 호출로 빼고 그 후속 질의로 찾은 근거는 남깁니다. `auto`는 한도가 감당하지 못하는 라우터 호출은 하지 않고, 호출 수 한도에 들어가는 전략 가운데 대체 전략으로 보냅니다.
- 전략을 지정한 질의에서 답변 호출이 들어가지 못하면(한도가 남긴 크기에 컨텍스트 구획이 하나도
  들어가지 않으면) 답변 호출 없이 정해진 거절 답변을 돌려주고 `abstention_reason: query_limit`을
  남깁니다. 미리 실패하는 것은 `auto`뿐입니다. 분기할 수 있는 전략 가운데 호출 수 한도에 들어가는
  전략이 없으면 호출을 하나도 하기 전에 `UsageLimitExceededError`(`unified_kg_rag`에서 내보냄,
  오류 분류 `limit_exceeded`, 종료 코드 `1`. 호출하는 쪽의 오류로 보고 `query_failed` 지표에는
  세지 않음)로 실패하고, `processing.ignore_errors`를 켜면 오류 답변을 돌려줍니다.
- 질의의 `effort`(아래)는 그 수준의 추론 몫만큼 답변의 `max_tokens`를 늘립니다. 이때 답변
  컨텍스트는 모델의 컨텍스트 창에서 그 출력 몫을 뺀 만큼만 씁니다.

호출하는 쪽은 질의 하나에 대해 세 한도를 낮출 수만 있고 올릴 수는 없습니다.

```python
from unified_kg_rag import QueryLimits, RAGInput

chain.invoke(RAGInput(query="...", limits=QueryLimits(max_model_calls=6, deadline_seconds=60)))
```

MCP 도구 `kg_search`와 `kg_answer`도 같은 `limits` 인수를 받습니다([MCP 서버](./mcp.ko.md)).

### 추론 강도

`RAGInput.effort`(`low`, `medium`, `high`, `xhigh`, `max`)는 질의 하나의 답변 모델이
얼마나 깊이 추론할지를 정하며, 해당 계층에 설정한 `aws.bedrock.default_effort` /
`fast_effort` 대신 쓰입니다([모델](../reference/models.md)(영문)). 추론을 선택 사항으로
두는 모델이면 추론을 켜고, 모든 모델 계열에서 그 강도의 추론 분량(`EFFORT_THINKING_TOKENS`,
`low` 2,048토큰부터 `max` 32,768토큰까지)만큼 여유를 둡니다. 필요하면 답변의 max_tokens를
`models.default_max_output_tokens`나 출력 상한(`search.output_ceilings`)보다 높이되, 모델의
최댓값은 넘지 않습니다. 답변 호출만 이 값을 따르고, 라우터, 질의 처리, 전략 자체의 호출은
계층 설정을 유지합니다. 강도 대신 추론 토큰 예산을 받는 이전 Claude 모델(4.5 이하)은 그
강도의 예산(`low` 2,048토큰부터 `max` 32,768토큰까지)으로 추론합니다. 답변 모델이 적용할 수
없는 강도는 모델을 호출하기 전에 거부합니다(`InvalidArgumentError`, 오류 클래스
`invalid_argument`). 모델이 받지 않는 강도이거나, 추론하지 않는 모델(로컬 서버의 모델,
추론 기능이 없는 Bedrock 모델)이면 그렇습니다. 그대로 두면 강도가 조용히 무시되기
때문입니다. 지정하지 않으면 바뀌는 것이 없습니다. `run-rag --effort`와 `run-eval --effort`는
실행의 모든 질의에 이 값을 지정합니다(`run-eval`에서는 데이터셋 항목의 `metadata.effort`가
우선합니다. 항목의 값을 포함해 실행들이 요청하는 모든 추론 강도를 어느 `--run-order`에서든
어떤 실행도 답하기 전에 실행마다(`--arm`과 closed-book 비교군 포함) 그 답변 모델로 확인하므로,
답변 모델이 거부하는 값이 있으면 답변하기 전에 `2`로 종료합니다. closed-book 비교군도 전략처럼
항목의 추론 강도로 답합니다). 질의 스팬(`kg_rag.effort`)과 질의 지표 기록(`effort`)에도 남습니다. MCP 도구와
콘솔의 Ask 페이지는 이 값을 받지 않습니다. 한도와 달리 추론 강도는 질의의 비용과 지연을 늘릴 수
있어, MCP 호출자에게 열려면 먼저 서버 쪽 상한이 필요합니다. `RAGInput.enable_thinking`은
더 이상 권장하지 않으며(`DeprecationWarning`), 계층의 강도로 추론을 켭니다.

```python
chain.invoke(RAGInput(query="...", effort="low"))
```

## 스트리밍

`GraphRAGChain.aquery_events`(동기: `query_events`)는 `ainvoke`와 같은 질의(제한 시간, 한도, 가드레일, 메모리
포함)를 실행하고, `kind`로 구분하는 타입이 있는 이벤트를 차례로 내보냅니다.

| 이벤트 | `kind` | 시점 |
|---|---|---|
| `QueryStart` | `start` | 질의가 받아들여짐: `query_id`, 대화, 요청한 전략, 네임스페이스, 추적 ID |
| `QueryRouted` | `routed` | 전략이 정해짐. `routing`은 `auto`가 고른 방법 |
| `QuerySources` | `sources` | 답변 모델에 주는 출처. 첫 청크보다 먼저 |
| `AnswerChunk` | `chunk` | 모델이 쓰는 대로 나오는 답변 텍스트 |
| `QueryResult` | `result` | `ainvoke`가 돌려주는 `RAGOutput` |

```python
from unified_kg_rag import AnswerChunk, QueryResult, RAGInput

async for event in chain.aquery_events(RAGInput(query="...")):
    if isinstance(event, AnswerChunk):
        print(event.text, end="", flush=True)
    elif isinstance(event, QueryResult):
        output = event.output  # 출처, 인용, 사용량, 메타데이터
```

- 청크는 미리 보기이고 기준은 `QueryResult`입니다. 출력 가드레일을 켜도 답변 모델은
  스트리밍으로 호출하지만(첫 청크 제한 시간 적용), 가드레일이 답변 전체를 검사할 때까지 아무것도
  보여 주지 않습니다. 검사가 끝나면 출처와 검사한 답변을 청크 하나로 보내고, 그다음에 대화를
  메모리에 씁니다. 답변 거절이나 빈 컨텍스트의 고정 문구도 마찬가지입니다(가드레일이 없으면
  고정 문구는 결과에만 들어 있습니다).
  `[NO_ANSWER]` 표식(`search.abstention`)은 판단이 설 때까지 보류하고 지웁니다.
- 실패하면 그 전에 보낸 이벤트 뒤에 예외가 납니다. `processing.ignore_errors`를 켜면
  결과에 오류 답변이 담깁니다.
- 질의는 읽는 쪽이 이벤트를 하나씩 가져갈 때까지 기다립니다. 읽기를 멈추고 반복자를
  닫으면(`aclose()`, 또는 `async with aclosing(...)` 안에서 `break`) 질의는 그 자리에서
  끝나고 메모리에는 아무것도 저장하지 않습니다. 질의 지표는 이를 실패로 세지 않습니다.
- 질의 지표, 로그와 오류의 `query_id`·네임스페이스, 추적은 `ainvoke`와 같습니다.
- `astream` / `stream`은 답변 텍스트만 내보냅니다. 각 청크를, 보여 준 청크가 없으면 최종
  답변을 한 번 내보냅니다. `ainvoke(on_answer_chunk=...)`는 더 이상 권장하지 않으며
  (`DeprecationWarning`), `AnswerChunk` 이벤트를 읽으면 됩니다. LangChain 자체의
  `astream_events`(콜백 이벤트)는 LangChain이 정의한 대로 둡니다.

## 인용

답변은 근거가 된 문서를 인용합니다(`search.citations`). 기본값으로는 `global`과 `drift`를 뺀
모든 전략의 답변이 인용합니다. 답변 컨텍스트는 근거 항목(본문 조각, 엔티티, 관계, 클레임, 커뮤니티
보고서, 요약)에 `[1]`, `[2]`, ... 번호를 붙이고, 답변 프롬프트는 모델에게 이 표식으로 인용하라고
지시합니다. 표식은 답변 언어와 관계없이 ASCII로 씁니다. `global`과 `drift`는 켜지 않으면 표식 없이
답합니다. 개괄형(sensemaking) 질문에서 인용하지 않은 답변이 더 포괄적이었기 때문입니다
([근거](../concepts/defaults-and-evidence.md#citations), 영문). 켜면 전역 검색의 핵심 요점과 DRIFT의
중간 답변에도 하나씩 번호가 붙습니다. 각 항목은 MS GraphRAG의 `[Data: Reports (...)]`처럼 그 글의 바탕이 된 보고서(또는
로컬 컨텍스트)의 문서를 밝히며, 근거 예산 때문에 보고서가 컨텍스트에서 빠져도 마찬가지입니다.
보고서를 읽는 모든 전략에서 보고서는 커뮤니티 전체가 아니라 발견 사항(finding)이 인용한 엔터티와
관계의 문서를 밝힙니다(발견 사항을 저장하기 전에 색인한 보고서는 커뮤니티의 문서를 쓰고, 관계
색인이 없는 GraphRAG 전용 네임스페이스에서는 엔터티의 문서를 밝힙니다). 모델이 쓴 글이므로 출처로는
나열하지 않습니다. 표식의 숫자는 세 자리까지이므로 `[2020]` 같은
괄호 속 연도는 본문으로 남습니다.

```text
창고는 공급사의 부품을 보관합니다 [1][3].
```

`RAGOutput.citations`는 번호마다 기록 하나를 돌려줍니다. `marker`, `cited`(답변이 인용했는지),
`section_type`, `title`, 대표 문서의 `document_id` / `file_name` / `file_path`(코퍼스
루트 기준 상대 경로) / `page`와 `page_end`, `chunk_id`(본문 조각), `text_unit_ids`, 모든
문서를 담은 `documents`, 시드 그래프 `provenance`와 `iri`가 들어 있습니다. 그래프 항목은
텍스트 단위 중 질의의 필터와 접미사를 통과한 최대 `search.citations.lineage_units`개의 문서를
밝힙니다. 이 정보는 묶어서 최대 네 번까지 읽으며, 필터가 제외한 문서는 드러나지 않고 그 자리는
다음 텍스트 단위로 채웁니다. 컨텍스트에 없던 번호를 답변이 쓰면 인용이 아니라
`metadata.unknown_citation_markers`에 남습니다. 저장소 읽기는 답변 호출과 동시에 진행합니다.
후속 질문을 다시 쓸 때는 이전 답변의 표식을 빼고 읽습니다.

- `run-rag`는 답변 아래에 인용한 출처를 출력하고, `--output-format json`은 모든 인용을
  담습니다.
- PDF 본문 조각에는 페이지가 붙습니다. 이번 릴리스 전에 수집한 텍스트에는 문서를 다시
  수집하기 전까지 페이지와 상대 경로가 없습니다.
- 스트리밍 답변에도 같은 표식이 들어 있습니다. `CitationMarkerScanner`로 청크에서 읽으면
  청크 경계에서 잘린 표식은 닫힐 때까지 보류합니다. `astream`은 텍스트만 돌려주고,
  `aquery_events`의 `QueryResult`와 `ainvoke`는 인용까지 돌려줍니다.
- 가드레일이 차단한 답변에는 인용이 없고, 마스킹한 답변은 마스킹된 텍스트 기준으로 인용
  여부를 표시합니다. 거절한 답변은 출처와 마찬가지로 인용을 유지합니다.
- 전략별 설정: `search.citations.strategies`는 기본값 `{global: false, drift: false}` 위에
  병합되며, 항목이 있으면 켜든 끄든 `enabled`보다 우선합니다. `auto`는 라우팅된 전략을 따르고,
  `iterative`는 자신이 따로 지정되지 않으면 기반 전략(`search.iterative_search.base`)을 따릅니다.
  인용하지 않는 전략은 번호 없는 컨텍스트를 받고, 인용 지시와 `RAGOutput.citations`가 없으며,
  `run-rag`, `kg_answer`, 콘솔도 표식을 보여 주지 않습니다.

  ```yaml
  search:
    citations:
      strategies: {global: true, drift: true}  # global과 DRIFT도 인용
  ```

- `search.citations.enabled: false`이면 맵에 지정되지 않은 모든 전략이 이전 컨텍스트와 프롬프트로
  돌아갑니다. 인용하라고 지시받은 답변의 표식은 텍스트에 남고, `run-eval`의 `answer_match`는
  그런 답변은 표식을 뺀 단어로, 인용 지시를 받지 않은 답변은 쓰인 그대로 채점합니다.

## 답변 거절

검색 결과가 없으면 체인은 모델을 호출하지 않고 질문의 언어로 정해진 문장("사용 가능한 데이터에서
이 질문에 답할 관련 정보를 찾지 못했습니다." 등 지원 언어마다 하나)으로 답하며
`metadata.abstained`와 `metadata.abstention_reason: no_context`를 남깁니다(커뮤니티 리포트가 없는
네임스페이스에 대한 `global`은 `no_reports`. 커뮤니티가 생기기에는 너무 작은 코퍼스이거나 리포트 없이
수집한 경우. 질의 한도 때문에 컨텍스트를 찾을 단계를 모두 빠뜨렸으면 `query_limit`). 언어는 질문의 문자 체계와
기능어로 정하고, 이것으로 알 수 없으면 색인 언어(문자 체계가 맞지 않으면 영어)를 씁니다. 거절인지는
문장이 아니라 `metadata.abstained`로 판단합니다. 그 밖에 모델이 자기 말로 답변을 거절하면 일반 답변과 구별되지 않습니다.
`search.abstention`(선택 기능, 기본값은 꺼짐)을 켜면 거절을 명시적으로 표시합니다.

```yaml
search:
  abstention:
    mode: mark            # off (default) | mark
    strip_marker: true    # remove the token from the returned answer
    on_insufficient: hedge   # hedge (default) | abstain
```

- **`mode: mark`**는 답변 프롬프트(간결형, 상세형, `custom_prompts` 재정의 모두)에 지시를
  덧붙입니다. 컨텍스트가 답을 뒷받침하지 않으면 고정 토큰 `[NO_ANSWER]`로 시작하고, 무엇이
  빠졌는지 질의 언어로 말하라는 지시입니다. 이 토큰으로 시작하는 답변은 답변 언어와 관계없이
  `metadata.abstained: true`, `abstention_reason: marker`가 됩니다(전각 `［NO_ANSWER］`와
  앞쪽 Markdown 강조도 인식). `strip_marker: false`가 아니면 토큰을 지우고, 토큰만 있는
  답변은 고정 무응답 문구로 바꿉니다. `stream`은 판단이 설 때까지 처음 최대 32자를
  보류합니다.
- 거절한 답변도 `sources`를 유지합니다. 모델이 본 컨텍스트이고, 평가기가 확인할 수 있어야
  하기 때문입니다. 가드레일이 차단했거나 컨텍스트가 비었을 때만 출처가 없습니다. 출력
  가드레일은 토큰을 뺀 답변을 검사합니다.
- **`on_insufficient`**는 전략이 자기 컨텍스트를 불충분하다고 표시했을 때만
  적용됩니다(`search_results.metadata.context_sufficient: false`, 선택적으로
  `insufficiency_reason`). 이 표시를 남기는 기본 제공 전략은 `iterative`입니다. `hedge`는
  컨텍스트가 뒷받침하는 부분만 답하고 다루지 못한 부분을 밝히게 하고, `abstain`은 답하지
  말고 질의 언어로 거절 문장을 쓰게 한 뒤 출력을 거절로 표시합니다
  (`abstention_reason: insufficient_context`). 어느 쪽도 모델 호출을 늘리지 않습니다.
  `iterative`로 측정했을 때 `abstain`은 답할 수 있는 질문을 많이 거절했고 token F1도
  떨어졌으므로 기본값은 `hedge`로 유지합니다
  ([근거](../concepts/defaults-and-evidence.md#declines-and-abstention), 영문).
- `mark`는 답변 프롬프트를 바꾸므로 답변과 EM / F1이 달라집니다. 다중 홉 질문에서 측정한
  결과, 단서를 달아 맞힌 답변까지 거절로 바뀌어 EM이 떨어졌습니다. 기계가 읽을 수
  있는 거절 표시가 필요할 때만 쓰는 선택 기능으로 유지합니다
  ([근거](../concepts/defaults-and-evidence.md#declines-and-abstention), 영문). `mark` 실행은
  별도 비교군으로 다루세요. 거절을 채점하는 방법:
  [평가: 답할 수 없는 질문과 거절](./evaluate.ko.md#답할-수-없는-질문과-거절).

## 다중 홉 질문의 후속 라운드

`--search-strategy iterative`는 `search.iterative_search.base`(기본 `mix`, 또는 `local`)를
실행한 뒤 답변 전에 fast 모델(`models.roles.query_controller`)이 컨텍스트를 판단하게 합니다.
컨텍스트로 질문에 답할 수 없으면 컨트롤러가 지목한 연결(bridge) 엔티티(빠진 홉이 다루는
엔티티. 예를 들어 "공급사가 맡긴 운송사의 창고는 어디에 있나?"의 창고)에서 기본 검색을 다시
시작하고, 후속 질의로 원문 구절을 검색하는 라운드를 최대 `max_rounds`(기본 2)번 반복합니다.
답변 모델은 융합한 후보로 한 번만 답합니다. 기본값으로 선택되지는 않지만, 질문에 이름이 나오지
않는 엔티티를 거쳐 사실을 이어야 하는 질문은 `auto`가 이 전략으로 보낼 수 있습니다
(`search.auto_routable_strategies`). `mix` 기반으로 실행하면 다중 홉 질문에서 `mix`만 쓸 때보다
유의하게 잘 답했고, 질의 시간 중앙값은 눈에 띄게 늘었습니다
([근거](../concepts/defaults-and-evidence.md#iterative-follow-up-rounds), 영문). `auto`가 이
전략을 고르지 않게 하려면 이 전략을 뺀 목록을 지정합니다:
`search.auto_routable_strategies: [local, mix, global]`.

```yaml
search:
  iterative_search:
    base: local          # 또는 mix(기본값)
    max_rounds: 2        # 첫 검색 뒤의 후속 라운드 수
    max_next_queries: 2  # 라운드마다 후속 질의 수
    round_top_k: 5       # 후속 질의마다 가져오는 구절 수
models:
  roles:
    query_controller: fast   # 컨트롤러 모델(기본값: fast 등급)
```

- **비용.** fast 등급 컨트롤러 호출이 최대 `max_rounds + 1`번이고, 라운드마다 기본 검색과 재순위화가 한 번씩 더 듭니다. 답변은 한 번만
  생성합니다. `metadata["iterative"]`에 라운드 수, 멈춘 이유(`stop_reason`: `sufficient`,
  `max_rounds`, `no_follow_up`, `deadline` 또는 대체 동작으로 넘어간(degraded) 단계), 판정마다의 결과, 후속 질의와
  라운드의 모델 사용량이 기록됩니다.
- **제한 시간.** `search.query_timeout_seconds`가 답변에 쓸 `search.model_timeout_seconds`를
  남길 때만 라운드를 시작합니다. 그 시점을 넘는 라운드는 취소되고, 답변은 앞선 라운드가 찾은
  후보로 생성합니다.
- **언어.** 후속 질의와 연결 엔티티 이름은 질의 언어로 씁니다(번역을 끄면 색인 언어).
- **필터**는 모든 라운드에 적용됩니다. **출처**는 답변 모델이 읽은 구절과 그래프 항목이며
  출처 정보(provenance)는 기본 전략과 같습니다.
- **부족한 컨텍스트.** 마지막 라운드 뒤에도 컨트롤러가 컨텍스트가 부족하다고 판단하면
  `metadata["context_sufficient"]`가 `false`가 되고(멈춘 이유는 `insufficiency_reason`),
  `search.abstention.on_insufficient`에 따라 단서를 달아 답하거나 답변을 거절합니다
  ([답변 거절](#답변-거절)).

동작 방식과 IRCoT·KAG와의 비교는
[설계 문서](../design/retrieval-graphrag-lightrag.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in)(영문)에 있습니다.

## 요약 트리 검색(선택 사항)

`--search-strategy tree`는 `raptor` 방법론으로 만든 네임스페이스를
([수집](./ingest.ko.md#raptor-요약-트리선택-사항)) 검색합니다. 구절과 요약 트리의 모든 레벨을
한 번에, 각각 BM25 + kNN으로 검색하고 융합해 함께 재순위화하며, 남은 요약마다 가장 잘 맞는
구절을 함께 가져옵니다. 멀티홉 질문에서 `naive`보다 나았지만 그래프 전략보다 크게 낮았고 한
코퍼스에서 1회 반복한 결과이므로 선택 사항입니다([근거](../concepts/defaults-and-evidence.md#raptor-summary-tree), 영문). `auto`는 `raptor`만 가진
네임스페이스에서만 이 전략을 고르고, 트리가 없는 네임스페이스는 모델을 호출하기 전에 이
전략을 거절합니다.

```yaml
search:
  tree_search:
    summary_top_k: 10        # 검색해 남길 요약 수
    text_top_k: 20           # 검색해 남길 구절 수
    max_nodes_per_level: 3   # 레벨마다 남길 요약 수의 상한
    expand_children: 2       # 요약마다 가져올 구절 수(0: 가져오지 않음)
    graph_signal: auto       # 또는 off: 엔티티로 찾는 버킷
  token_manager:
    type_budgets:
      summary: 0.5           # text의 0.5와 함께 쓰는 요약의 몫
```

- **출처.** 요약은 `level`, `n_leaves`, `child_ids`, `text_unit_ids`(요약한 구절)와 함께
  보고되고, 함께 가져온 구절은 `expanded_from`(요약의 id)을 가집니다. `run-eval`의 검색
  평가기는 요약을 그 구절로 파일에 귀속합니다(`by_source_type`의 `summary` 행).
- **그래프 신호.** 네임스페이스에 엔티티 인덱스도 있으면(`graphrag`나 `lightrag`를 고른
  경우) 원 질의로 찾은 엔티티를 인용하는 요약과 구절을 더합니다(모델 호출 없음,
  `metadata["graph_signal"]`이 실행 여부를 알려 줌).
- **필터.** 요약은 모든 문서가 같은 값을 가진 문서 필터 키만 가지므로, 필터를 걸면 모든
  문서가 통과하는 요약만 남고, 함께 가져오는 구절도 같은 필터로 찾습니다.
- **비용.** `naive` 질의에 검색 몇 번과 더 넓은 재순위화가 더해지며, 답변 전에 모델을
  호출하지 않습니다.

동작 방식과 RAPTOR 원 코드와의 차이는 [설계](../design/retrieval-hipporag-raptor.md#65-raptor-summary-tree-raptor-tree-opt-in)(영문)에
있습니다.

## 구조화 그래프 질의(선택 사항)

`--search-strategy structured`는 그래프 저장소가 실행한 질의로 개수, 순위, 목록,
이름이 주어진 두 엔티티 사이의 경로, 시드 엔티티의 속성에 답합니다. 모델이 질문을
JSON 계획으로 쓰고(질의 원문은 쓰지 않음), 코드가 계획을 검사해 질의의 접미사
안에서 읽기 전용으로 실행하며, 답변은 결과 표와 결과가 가리키는 엔티티, 그 구절을
읽습니다. 구조화된 답을 얻지 못하면 `search.structured.fallback_strategy`
(`local`)를 실행하고, 무슨 일이 있었는지는 `metadata["structured"]`에 남습니다.
선택 사항이며 측정 전이므로 `search.auto_routable_strategies`에 넣어야 `auto`가
분기합니다. 가이드: [구조화 그래프 질의](./structured-queries.ko.md).

```bash
run-rag --query "구매처에 가장 많이 공급하는 벤더는?" \
  --search-strategy structured --config-path config.yaml
```

## 대화형 모드와 대화 메모리

```bash
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

대화형 모드에서는 메모리가 켜집니다. 명령: `help`, `new`(새 대화 ID),
`set-filter key:value`, `clear-filters`, `show-config`, `quit` / `exit`. 명령은 줄 전체여야
하므로(인자를 받는 명령은 `set-filter`뿐) `help me find ...`는 질문으로 처리합니다.
`set-filter`는 `--filters`처럼 검색 전략의 어느 인덱스에도 없는 키를 거부합니다(`--interactive`와
함께 준 `--filters`는 첫 질문 전에 검사하며, 거부하면 종료 코드 2). `auto`에서는 이미 설정한
필터와 함께, 라우팅될 수 있는 어떤 전략도 모두 적용하지 못할 때만 키를 거부합니다.

대화 메모리는 프로세스 안에만 있습니다(`MemoryManager`, 한도는 `memory` 섹션).

- 세션 하나의 턴 사이에서, 또는 `use_memory=True`와 같은 `suffix`, `user_id`로
  `conversation_id`를 다시 쓰는 `GraphRAGChain` 하나의 호출 사이에서 이어집니다(다른
  테넌트나 사용자의 턴은 섞이지 않음). 체인끼리는 같은 `MemoryManager(config)`를
  `memory_manager=`로 넘겨야만 공유합니다.
- `run-rag --query`를 따로 실행하면 매번 이력 없이 시작하므로 `--conversation-id`가
  효과가 없습니다(`run-rag`가 WARNING을 남김). CLI에서 여러 턴을 이어 가려면 세션의
  메모리를 늘 유지하는 `--interactive`를 쓰세요.
- 검색 전에 후속 질문을 대화 이력으로 독립된 질문으로 다시 씁니다(이력이 있는 턴마다
  `models.roles.query_rewrite` 호출 한 번, 프롬프트 키 `query_rewrite`). 그래서 "그 회사의
  공급업체는?"도 앞선 턴의 엔티티 이름이 들어간 질문이 됩니다. 재작성은 최근
  `memory.max_messages_per_conversation`개 메시지를 읽고 앞선 답변은 각각 1,000자까지만
  쓰며, `enable_query_processing`을 꺼도 실행됩니다(이 플래그는 번역과 엔티티·키워드
  추출을 끔). 메모리는 이 외에 LLM을 추가로 호출하지 않습니다.

### 호출자가 넘기는 대화 이력

대화를 직접 관리하는 서버(턴마다 요청 하나)는 메모리 대신 `RAGInput.history`로 대화를
넘깁니다.

```python
chain.invoke(RAGInput(
    query="그럼 가격은요?",
    history=[
        {"role": "user", "content": "공급사는 어떤 부품을 공급하나요?"},
        {"role": "assistant", "content": "공급사는 부품 A를 공급합니다 [1]."},
    ],
))
```

후속 질문 재작성은 메모리와 같은 방식으로 이력을 읽습니다. 최근
`memory.max_messages_per_conversation`개 턴만, 답변은 각각 1,000자까지만 쓰고,
`memory.max_history_tokens`(기본값 4,000, 추정치)에 들어가는 최근 턴만 남깁니다. 입력
가드레일이 켜져 있으면 질의처럼 검사합니다. 모든 턴을 한 번에 검사하고, 개입했을 때만 턴마다
다시 검사합니다. 마스킹된 턴은 마스킹된 채로 쓰고 차단된 턴은 빼며(둘 다
`metadata.guardrail`에 기록), 질의는 계속 진행합니다. `history`와 `use_memory=True`를 함께
주면 거부하며, `history`를 준 질의는 메모리에 아무것도 쓰지 않습니다.
