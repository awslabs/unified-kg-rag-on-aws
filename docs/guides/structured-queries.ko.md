# 구조화 그래프 질의

> 🇬🇧 English version: [structured-queries.md](./structured-queries.md)

`--search-strategy structured`는 구절 몇 개가 아니라 그래프의 구조가 필요한
질문에 답합니다. 어느 나라의 구매처에 공급하는 벤더가 몇 곳인지, 구매처에 가장
많이 공급하는 벤더가 어디인지, 이름이 주어진 구매처의 공급사가 어떤 제품을
만드는지, 이름이 주어진 두 엔티티가 어떻게 연결되는지, 한쪽이 다른 쪽에
공급하는지, 큐레이션된(시드) 엔티티의 속성이 무엇인지 같은 질문입니다. 모델은
질문을 그래프의 타입, 관계, 속성에 대한 작은 JSON *계획*(plan)으로 쓰고, 코드가
그 계획을 검사해 질의의 인덱스 접미사(suffix)로 범위를 한정한 그래프 질의로
컴파일한 뒤 그래프 저장소에서 읽기 전용으로 실행합니다. 답변 모델은 결과 표,
결과가 가리키는 엔티티와 경로, 그것들이 인용하는 구절을 읽습니다.

선택 사항이며 아직 측정 전입니다. 기본값으로 고르는 곳이 없고,
`search.auto_routable_strategies`에 넣기 전까지 `auto`도 이 전략으로 분기하지
않습니다. 내부 동작:
[Design §6.6](../design/structured-queries.md#66-structured-graph-queries-structured-opt-in)(영문).

## 실행

```bash
run-rag --query "Norland의 구매처에 공급하는 벤더는 몇 곳입니까?" \
  --search-strategy structured --config-path config.yaml
```

그래프를 가진 모든 네임스페이스(`graphrag`, `lightrag`, `hipporag`로 만든
네임스페이스)가 이 전략을 제공하고, `raptor`만 가진 네임스페이스는 모델을
호출하기 전에 거부합니다. Neptune, Gremlin Server(`providers.graph.kind:
gremlin`), 내장 SQLite 그래프(`backend: local`) 세 그래프 저장소 모두에서
동작합니다.

## 계획으로 물을 수 있는 것

| 질문 형태 | 계획 |
|---|---|
| Norland의 구매처에 공급하는 벤더는 몇 곳인가? | `(v: VENDOR) -[SUPPLIES]-> (b: BUYER, seed.country = "Norland")`, 서로 다른 `v`의 수 |
| 구매처에 가장 많이 공급하는 벤더는? | 같은 패턴, `v`마다 서로 다른 `b`의 수, 많은 순, 상위 1개 |
| Acme Retail의 공급사는 어떤 제품을 만드는가? | `(x named "Acme Retail") <-[SUPPLIES]- (s: VENDOR) -[MAKES]-> (p: PRODUCT)`, 서로 다른 `p` |
| Acme Retail은 Globex Parts와 어떻게 연결되는가? | 두 이름 사이의 1~3홉 경로, 짧은 순 |
| Globex Parts는 Acme Retail에 공급하는가? | 패턴이 일치하는지 여부 |
| 가장 흔한 엔티티 종류는? | `type`별 엔티티 수 |
| 벤더의 직원 수 최댓값은? | 벤더들의 `seed.employees`의 `max` |

계획은 노드 최대 네 개로 이루어진 한 줄의 사슬입니다. 노드에는 엔티티 타입,
질문이 주는 이름, 속성 조건을 적습니다. 속성은 핵심 속성(`name`, `type`, `rank`, `confidence`,
`in_schema`, `iri`, `type_iri`, `source_kinds`, `description`), 추출된 속성
`attr.<key>`, 시드 속성 `seed.<key>`입니다. 엣지에는 관계 타입, 방향, 1에서
`max_hops`까지의 홉 수를 적습니다. 계획 작성기는 선언된 추출 스키마와 그래프가
가진 것의 프로필(개수가 붙은 타입과 관계, 관찰된 패턴, 값 종류가 적은 문자열
속성이면 최대 `max_example_values`개의 표본 값이 붙은 속성)을 보며, 엔티티 이름은
보지 않습니다. 스키마에 없는 타입, 관계, 속성을 쓴 계획은 거부되고, 거부 사유를
보여 주며 계획 작성기를 한 번 더 호출합니다. 값 종류가 여러 가지인 속성에 대한
부분 문자열 비교나 대소 비교도 거부합니다. 필터가 있으면 스키마에 표본 값을 보여
주지 않습니다.

계획의 이름은 하이브리드 엔티티 검색 결과 중 그 이름을 가진 엔티티(띄어쓰기와 법인
형태 표기는 따지지 않으므로 `가나다상사`가 `가나다 상사`를 찾습니다)에 연결하고,
그런 엔티티가 없으면 어휘(BM25) 검색의 상위 결과에 연결합니다. 저장된 어떤
엔티티와도 단어가 겹치지 않는 이름은 아무것에도 연결되지 않아 대체 전략으로
넘어갑니다(`unlinked_mention`). "Zorblax Industries는 Acme에 공급합니까?"를 비슷한
엔티티로 답하지 않습니다. 컨텍스트에는 이름마다 연결된 엔티티가 나옵니다.

## 계획으로 답하지 못할 때

```yaml
search:
  structured:
    mode: fallback            # 또는 augment
    fallback_strategy: local  # null: 컨텍스트 없이 답변
```

`fallback`(기본값)에서는 계획으로 답하지 못한 질문에 `fallback_strategy`를
실행합니다. 계획 작성 호출이 실패했거나 재작성 뒤에도 계획이 유효하지 않은 경우
(`planner_error`, `invalid_plan`), 질문을 `unanswerable`로 판단한 경우, 이름이
아무것에도 연결되지 않은 경우(`unlinked_mention`), 결과가 빈 경우
(`empty_result`, 개수 0과 "아니오"는 답입니다), 저장소가 질의를 거부하거나 시간
제한에 걸린 경우(`execution_error`, `timeout`)입니다. 대체 전략은 계획 작성기가
적은 엔티티 이름과 주제를 읽으므로 추출 호출이 늘지 않습니다. 네임스페이스가
제공하지 않는 대체 전략은 `local`, 그다음 `mix`로 바뀝니다. `augment`에서는 모든
질의에서 대체 전략을 그래프 질의 옆에서 함께 실행하고 두 컨텍스트를 융합하되 질의
결과를 앞에 둡니다. 개수와 텍스트가 섞인 질문의 재현율이 올라가는 대신 모든
질의가 대체 전략의 비용을 냅니다. 필터는 두 경로 모두 적용할 수 있어야 하므로
실행될 대체 전략에 대해서도 키를 확인하며, `naive`나 `tree`가 버릴 키는 모델
호출 전에 대체 전략을 밝히며 거부합니다.

`RAGOutput.metadata["structured"]`에 무슨 일이 있었는지 남습니다. `status`
(`answered`, `fallback`, `failed`), `reason`, `fallback_strategy`, 검증된
`plan`, `plan_description`, `operation`, `row_count`, `truncated`, `exact`,
`linked`와 `linked_names`(이름마다 연결된 엔티티), `repairs`, `backend`, 단계별
`elapsed_ms`, `usage`(계획 작성 호출)입니다.

## 정확한 답, 필터, 시드 속성

문서 필터나 시드 속성 조건이 없으면 그래프 저장소가 모든 일치 항목을 대상으로
개수와 그룹을 계산하므로 정확하고, 모든 저장소가 계획의 순서대로 페이지를 자르므로
순위와 목록도 정확한 상위 항목을 유지합니다. 개수, 집계, "예" 답도 순위가 높은
일치 항목을 엔티티와 구절로 인용합니다. 결과 표도 다른 근거 섹션처럼 `[n]`
번호를 받고, 그 인용은 해당 엔티티가 나온 문서를 가리킵니다. 그래프 안에서 평가할 수 없는 조건이 두 가지
있습니다. 문서 필터(`attr_<key>`, `attributes.filters.<key>`, 그래프 정점에는
없습니다)와 시드 속성(정점마다 JSON 문자열 하나)입니다. 둘 중 하나가 있으면
저장소는 일치 항목을 최대 `max_scan_rows`(5,000)개까지 돌려주고, 코드가 시드
조건을 적용하고 바인딩된 모든 엔티티를 문서 필터 아래 엔티티 인덱스로(돌려준
경로의 엣지는 관계 인덱스로) 확인한 뒤 집계합니다. 상한을 넘으면 답에 잘림
표시가 붙어, 개수는 컨텍스트에 "at least N"으로 적히고 답변도 그렇게 말합니다.
호출자의 `type`, `source_kinds`, `attr_*` 등 엔티티 키는 바인딩된 모든 엔티티에,
`source_kinds`는 경로의 모든 엣지에도 적용됩니다
([필터 레퍼런스](../reference/filters.md#structured-queries), 영문).

개수는 추출과 시드 그래프가 저장한 그대로를 셉니다. 엔티티 해소(resolution)가
합치지 못한 한 엔티티의 두 표기는 두 번 셉니다.

## 설정

```yaml
models:
  roles:
    graph_query: default       # 계획 작성기(질의마다 호출 1회, fast로 바꾸기도 쉽습니다)
search:
  structured:
    max_hops: 3                # 계획의 최장 경로(1-4)
    rows_in_context: 50        # 컨텍스트에 넣는 행, 그룹, 경로 수
    max_scan_rows: 5000        # 저장소가 코드로 돌려주는 최대 일치 항목 수
    store_timeout_seconds: 10  # 그래프 질의 하나의 서버 측 제한
    repair_attempts: 1         # 거부된 계획 뒤의 재호출 수
    schema_source: both        # declared, observed, both
    max_schema_tokens: 3000    # 스키마 텍스트 예산
    profile_sample_vertices: 2000
    profile_ttl_seconds: 600
    max_example_values: 10     # 값 종류가 적은 속성의 표본 값 수
    link_candidates: 3         # 정확한 이름이 없을 때 이름 하나가 연결되는 엔티티 수
    supporting_chunks: 5       # 결과 엔티티에 대해 가져오는 구절 수
  token_manager:
    type_budgets:
      query_result: 0.25       # 결과 표의 컨텍스트 몫
```

모든 키: [`search.structured`](../reference/configuration.md#searchstructured)(영문).
계획 작성기 프롬프트는 `custom_prompts.graph_query_system` / `_human`으로 바꿀
수 있으며 `{query}`와 `{schema}`는 남겨야 합니다.

## 비용과 한계

계획 작성 호출 1회(스키마와 예시로 입력 약 3천~4천 토큰, 출력 수백 토큰), 최대
`repair_attempts`회의 추가 호출, 접미사와 `profile_ttl_seconds`마다 프로필 읽기
1회, 이름마다 엔티티 검색 1회, 그래프 질의 1회, 인덱스 조회 한두 번, 구절 검색
1회, 그리고 답변입니다. `fallback`에서는 실패한 계획에 대체 전략의 비용이
더해집니다.

계획은 한 줄의 사슬입니다. 선택적 부분, 합집합, 하위 질의는 없고, 경로는 두 노드
사이에서만 구합니다. 밀도가 높은 그래프에서는 3홉 패턴이 시간 제한에 걸릴 수
있으며, 그러면 대체 전략으로 넘어갑니다. 모델이 쓴 원문 질의(openCypher,
Gremlin)는 지원하지 않습니다. 계약은 계획입니다.

## 안전성

모델은 계획만 씁니다. 계획에는 정점 레이블(label), 접미사, 인덱스, id를 적을
필드가 없습니다. 레이블은 컴파일러가 질의의 접미사로 붙이며, 컴파일된 Gremlin의
모든 정점 단계 뒤에 그 레이블이 오고 컴파일된 SQL의 모든 테이블이 그 레이블에
묶입니다. 검사기가 보내기 전에 컴파일 결과를 읽어 그 레이블의 읽기 단계가 아닌
것은 모두 거부합니다. 값은 바이트코드 인자나 바인딩된 매개변수로 전달됩니다. 내장
그래프는 읽기 전용 연결에서, Gremlin은 서버 측 시간 제한을 걸고 실행하며, 배포된
질의 역할에는 Neptune 쓰기 권한이 없습니다
([운영](../operations.md#query-access-and-tenant-isolation), 영문). 필터는 접근
제어가 아니므로 접미사는 여전히 인증된 주체에서 정해야 합니다
([범위](../concepts/scopes-and-namespaces.ko.md)). 출력 가드레일을 켜면 모델이 쓴 계획
텍스트를 돌려주는 모든 곳에서 답변처럼 검사합니다. 검색과 답변의 메타데이터, 결과 표, 보고하는
질의 결과 출처(콘솔, MCP `kg_answer`, `run-rag --output-format json`이 보여 줌)가 여기에 해당합니다.
