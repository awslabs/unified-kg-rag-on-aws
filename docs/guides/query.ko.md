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

# Verbose (entities, top sources, metrics) with attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

## 필터

`--filters key:value ...`(또는 `RAGInput.filters`)는 검색이 순위를 매길 대상을
좁힙니다. 키·`:`·값 중 하나가 없는 쌍, 두 번 준 키, 전략이 읽는 어떤 인덱스도 선언하지
않은 키는 사용법 오류(종료 코드 2)입니다.

- 필터는 해당 필드를 선언한 저장소에만 적용됩니다(`type:person`은 텍스트 단위를
  거르지 않음). 선택한 전략이 읽는 저장소 어디에도 선언되지 않은 키(예: `category`,
  `naive`나 `global`의 `type`)는 필터 가능한 키 목록과 함께 `InvalidFilterError`를
  일으킵니다. `auto`는 라우팅될 수 있는 전략 중 키를 모두 적용하는 전략 사이에서만
  라우팅합니다.
- `attr_<key>`는 문서 `filters` 메타데이터의 `<key>` 항목과 비교합니다. OpenSearch는 이를
  엄격하게(fail-closed) 적용하므로 속성이 없는 항목은 제외됩니다. 여러 문서에서 나온
  엔티티, 관계, 주장, 커뮤니티 리포트는 그 문서 중 하나가 일치하면 일치합니다.
- 엔티티와 주장의 `type`은 소문자로 저장되므로 `type:PERSON`이 아니라 `type:person`으로
  필터링하세요. 불리언·숫자 필드는 그 유형의 값을 받습니다(`in_schema:false`, `rank:2`).

저장소별로 필터할 수 있는 모든 필드, 값의 유형, 날짜, 전략별로 적용되는 키, 그래프
확장의 필터 처리는 [필터 레퍼런스](../reference/filters.md)(영문)에 있습니다. 필터는 접근
제어 수단이 아닙니다([접근 제어 경계가 아님](../concepts/scopes-and-namespaces.ko.md#접근-제어-경계가-아님)).

## 재순위화 상태

모든 질의는 재순위화 결과를 `SearchResult`와 RAG 출력의 `metadata.rerank`(따라서
`--output-format json`에도)에 남깁니다. `status`, `reason`, `stages`로 구성되며,
`stages`에는 재순위화 단계마다 `status`, `scope`(`all`, 또는 재순위화한 결과 유형. 예:
`local`, `drift`, `mix`, `hybrid`의 `text`), `reason`이 있습니다.

| 상태 | 의미 |
|---|---|
| `applied` | 재순위화 모델이 후보 순서를 바꿨습니다. |
| `disabled` | `providers.rerank.kind: none`. |
| `unavailable` | 재순위화 모델을 만들 수 없습니다(초기화 실패는 ERROR로 기록, 또는 크로스 인코더의 `[local]` extra 미설치). |
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
  청크가 오기 시작하면 전체 제한 시간이 적용됩니다.
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

## 답변 거절

검색 결과가 없으면 체인은 모델을 호출하지 않고 "I could not find relevant
information..."으로 답하며 `metadata.abstained`와 `metadata.abstention_reason: no_context`를
남깁니다. 그 밖에 모델이 자기 말로 답변을 거절하면 일반 답변과 구별되지 않습니다.
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
- `mark`는 답변 프롬프트를 바꾸므로 답변과 EM / F1이 달라질 수 있습니다. 측정으로 결정할
  때까지 선택 기능이며, `mark` 실행은 별도 비교군으로 다루세요. 거절을 채점하는 방법:
  [평가: 답할 수 없는 질문과 거절](./evaluate.ko.md#답할-수-없는-질문과-거절).

## 다중 홉 질문의 후속 라운드(선택 사항)

`--search-strategy iterative`는 `search.iterative_search.base`(기본 `mix`, 또는 `local`)를
실행한 뒤 답변 전에 fast 모델(`models.roles.query_controller`)이 컨텍스트를 판단하게 합니다.
컨텍스트로 질문에 답할 수 없으면 컨트롤러가 지목한 연결(bridge) 엔티티(빠진 홉이 다루는
엔티티. 예를 들어 "공급사가 맡긴 운송사의 창고는 어디에 있나?"의 창고)에서 기본 검색을 다시
시작하고, 후속 질의로 원문 구절을 검색하는 라운드를 최대 `max_rounds`(기본 2)번 반복합니다.
답변 모델은 융합한 후보로 한 번만 답합니다. 선택 사항이며 아직 측정하지 않았습니다. `auto`는
이 전략으로 분기하지 않고, 고르지 않으면 아무것도 바뀌지 않습니다.

```yaml
search:
  iterative_search:
    base: local          # 또는 mix(기본값)
    max_rounds: 2        # 첫 검색 뒤의 후속 라운드 수
    max_next_queries: 2  # 라운드마다 후속 질의 수
    round_top_k: 5       # 후속 질의마다 가져오는 구절 수
    decompose: false     # true면 대신 질문을 단계로 나눔
models:
  roles:
    query_controller: fast   # 컨트롤러 모델(기본값: fast 등급)
```

- **비용.** fast 등급 컨트롤러 호출이 최대 `max_rounds + 1`번(`decompose`이면
  `max_rounds + 2`번)이고, 라운드마다 기본 검색과 재순위화가 한 번씩 더 듭니다. 답변은 한 번만
  생성합니다. `metadata["iterative"]`에 라운드 수, 멈춘 이유(`stop_reason`: `sufficient`,
  `max_rounds`, `no_follow_up`, `deadline` 또는 실패한 단계), 판정마다의 결과, 후속 질의와
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
[설계 문서](../design.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in)(영문)에 있습니다.

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
- `run-rag --query`를 따로 실행하면 매번 이력 없이 시작하므로 `--use-memory`와
  `--conversation-id`가 효과가 없습니다(`run-rag`가 WARNING을 남김). CLI에서 여러 턴을
  이어 가려면 `--interactive`를 쓰세요.
- 검색 전에 후속 질문을 대화 이력으로 독립된 질문으로 다시 씁니다(이력이 있는 턴마다
  `models.roles.query_rewrite` 호출 한 번, 프롬프트 키 `query_rewrite`). 그래서 "그 회사의
  공급업체는?"도 앞선 턴의 엔티티 이름이 들어간 질문이 됩니다. 재작성은 최근
  `memory.max_messages_per_conversation`개 메시지를 읽고 앞선 답변은 각각 1,000자까지만
  쓰며, `enable_query_processing`을 꺼도 실행됩니다(이 플래그는 번역과 엔티티·키워드
  추출을 끔). 메모리는 이 외에 LLM을 추가로 호출하지 않습니다.
