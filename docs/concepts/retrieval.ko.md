# 검색

> 🇬🇧 English version: [retrieval.md](./retrieval.md)

질의는 두 방법론, GraphRAG(커뮤니티 요약)와 LightRAG(이중 레벨 키워드) 중 한쪽의 전략을
고르거나, `auto`(기본값)가 전략을 고르게 합니다. 모든 전략은 같은 인덱스 위에서 하나의
체인으로 실행됩니다. 질의 처리, 검색, 결과 융합(fusion)과 재순위화(rerank)를 거쳐
컨텍스트 예산 안에서 답변을 만들며, 모든 답변은 답변 모델이 실제로 본 출처를 함께
돌려줍니다. 이 문서는 전략과 답변을 좌우하는 선택을 설명합니다. 질의 실행 방법은
[질의](../guides/query.ko.md)에 있고, 전략별 검색 방식은
[설계 §6](../design.md#6-retrieval-two-methodologies)(영문)의 그림에 있습니다.

## 검색 전략 선택

GraphRAG 전략은 코퍼스 전체를 *요약하고 주제별로 종합*하는 데 강하고, LightRAG 전략은
더 빠르며 *이중 레벨 키워드* 검색을 중심으로 동작하므로 키워드 중심 조회와 저비용
기준선에 적합합니다. 두 방법론 모두 같은 하이브리드 스코어러(BM25, 벡터, 그래프, RRF,
재순위화)를 씁니다.

각 전략은 자기 방법론의 인덱스 산출물을 읽습니다. 그 방법론이 없는 인덱스 네임스페이스에서는
([방법론 프로필](./overview.ko.md#방법론-프로필)) 모델을 호출하기 전에 거절되고, `auto`는
네임스페이스가 가진 방법론의 전략 중에서만 고릅니다.

**GraphRAG(커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회 | 그래프 순회 없이 OpenSearch 벡터·키워드로 검색하며, 주장이 켜져 있으면 주장 인덱스도 포함합니다. 가장 빠릅니다. |
| `local` | 특정 엔티티에 관한 질문 | 질의 엔티티 → 그래프 순회로 이웃과 관계를 찾아 벡터·키워드 결과와 결합합니다. 켜져 있으면 주장도 넣습니다. 질의 엔티티는 이름으로 순회를 시작하고, 이름이 같은 엔티티가 없으면 최상위 결과가 가장 좋은 결과 점수의 `search.local_search.seed_score_ratio`배 이상일 때 시작점이 됩니다([레퍼런스](../reference/configuration.md#searchlocal_search), 영문). 커뮤니티 보고서는 기본값으로 질의 엔티티와 가장 비슷한 것을 넣습니다([membership을 기본값으로 쓰지 않는 이유](./defaults-and-evidence.md#local-report-selection), 영문). |
| `global` | "주요 주제는 무엇인가" 같은 넓은 주제 질문 | 계층 한 단면(`search.global_search.community_depth`, 기본값 2. 리포트가 `max_reports_per_level`보다 많은 단면은 그보다 얕은 단면 중 한도 안에 드는 가장 깊은 단면으로 대신하고, 리포트 키 필터가 있으면 가장 비슷한 리포트를 고름)의 커뮤니티 리포트를 fast 모델로 점수 매긴 핵심 내용으로 map하고(리포트 `map_context_tokens`마다 호출 1회), 순위를 매긴 핵심 내용으로 답합니다. 동적 선택은 대신 루트부터 내려갑니다. |
| `drift` | 큰 그림과 구체적 사실이 함께 필요한 여러 부분으로 된 질문 | 가상 답변(HyDE)과 비슷한 리포트로 중간 답변과 후속 질의를 만들고, 후속 질의마다 local search를 실행해 새 후속 질의를 얻는 과정을 `n_depth` 라운드 반복한 뒤 답변 모델이 종합합니다. LLM 호출이 가장 많습니다(약 `primer_folds + k_followups * n_depth`회). |
| `auto` | 일반 용도(기본값) | LLM 라우터(`models.roles.strategy_selection`)가 `search.auto_routable_strategies`(기본 local, mix, global. 질의마다 LLM 호출이 최대 27번인 drift는 추가했을 때만) 중에서 고릅니다. 후보 전략이 읽는 추출을 라우터와 동시에 실행하므로 `auto`는 고른 전략을 직접 실행할 때와 걸리는 시간이 같고, 분기한 전략이 읽지 않는 추출도 과금됩니다(질의마다 fast 모델 호출 약 1회, [`search.auto_speculative_extraction`](../reference/configuration.md#search), 영문). |

**LightRAG(이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반적인 LightRAG 사용 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉은 `search.lightrag_search.enable_graph_expansion`으로 선택)에 **더해** naive 청크 검색 결과를 섞습니다. |
| `hybrid` | 키워드 중심의 그래프 질문 | naive 청크 결과를 섞지 않는 `mix`입니다. |
| `naive` | 빠른 기준선, 비교 평가 | 청크 검색만 합니다(`search.lightrag_search.chunk_stream_top_k`, 기본 20개). 기본값 `--search-type hybrid`는 BM25를 더하며, `--search-type vector`는 upstream LightRAG의 순수 벡터 기준선과 MS GraphRAG의 `basic` 검색에 가깝습니다. |

**HippoRAG 2(개인화 PageRank, 선택 사항):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `ppr` | 사실이 몇 개의 엔티티를 거쳐 이어지는 다중 홉 질문, 커뮤니티 보고서 없이 `hipporag`나 `lightrag`만으로 만든 네임스페이스(방법론 `hipporag`) | 질문의 관계 검색 결과(사실, kNN)와 텍스트 단위 검색 결과(BM25 + 벡터)를 시작점으로, 사실의 엔티티 주변 그래프와 그 엔티티가 인용하는 문단에 개인화 PageRank(Personalized PageRank)를 실행합니다(HippoRAG 2, 감쇠 계수 0.5). 순위를 매긴 문단, 문단 검색 결과, 사실을 다른 전략처럼 융합하고 재순위화합니다. 질의마다 LLM 호출은 답변 1회입니다. 답변 품질을 아직 측정하지 않았으므로 `auto`는 이 전략으로 보내지 않으며, 직접 지정해야 합니다([`search.ppr_search`](../reference/configuration.md#searchppr_search), [설계](../design.md#64-hipporag-2-personalized-pagerank-opt-in-ppr-local-ranking), 영문). `search.local_search.text_unit_ranking: ppr`는 같은 PageRank로 `local`의 텍스트 단위 순서를 정합니다(MuSiQue 200문항에서 컨텍스트에 든 근거 문단이 11포인트 늘었고, 답변은 아직 측정하지 않았습니다). |

`mix`와 `hybrid`는 수집이 만드는 관계 인덱스를 읽습니다. 키워드가 나오지 않는 짧은
질의는 원래 질의를 저수준 키워드로 씁니다(`search.lightrag_search.raw_query_fallback_max_len`
이하).

두 전략은 찾은 엔티티가 인용하는 원문 청크와, 별도 스트림으로 찾은 관계가 인용하는
청크를 더합니다(각각 최대 `search.lightrag_search.chunk_stream_top_k`개. 결과 융합(fusion)이
두 스트림을 번갈아 섞고, 합친 청크는 재순위화(rerank) 뒤에 이 개수로 자릅니다).
`search.lightrag_search.kg_chunk_pick_method`는 항목마다 어떤 청크를 고를지 정합니다.
`weight`(기본값)는 찾은 항목마다 감소하는 비율로 `related_chunk_number`개씩 배정하고,
`vector`는 upstream LightRAG의 기본값처럼 인용된 청크 중 질의와 가장 잘 맞는 청크를
엔티티 쪽과 관계 쪽에서 한 번씩 검색해 고릅니다. 기본값은 `weight` 그대로이고 MMR의
`search.fusion.diversity_lambda`도 0.5를 유지합니다. 다른 값은 측정에서 유의한 차이가
없었습니다([근거](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity), 영문).

**후속 라운드(선택 사항, 측정 전):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `iterative` | 마지막 홉이 질문에 이름이 나오지 않는 엔티티에 관한 다중 홉 질문 | `search.iterative_search.base`(기본 `mix`, 또는 `local`)를 실행한 뒤 fast 모델 컨트롤러(`models.roles.query_controller`)가 묶은 컨텍스트를 읽습니다. 컨텍스트로 답할 수 없으면 컨트롤러가 지목한 연결(bridge) 엔티티로 기본 검색을 다시 시작하고 후속 질의로 원문 구절을 검색하는 라운드를 최대 `max_rounds`(2)번 반복합니다. 라운드마다 찾은 후보를 융합·재순위화하고 답변 모델은 한 번만 답합니다. `decompose: true`이면 대신 질문을 단계로 나눕니다. 질의마다 기본 전략보다 fast 모델 호출이 1-3번 많습니다. 명시적으로 고를 때만 실행되며, 측정하기 전까지 `auto`는 이 전략으로 분기하지 않습니다([설계](../design.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in), 영문). |

## 하이브리드 점수와 재순위화

모든 전략은 출처별 결과(BM25, 벡터, 그래프)를 RRF(reciprocal rank fusion)나 가중치로
융합한 뒤 재순위화하고, MMR 다양성을 적용하고, 유형별 할당량으로 자릅니다. 이 순서이므로
융합 점수로 자르다가 재순위화 모델이 맨 위에 놓을 결과를 잃는 일이 없습니다
([설계 §7](../design.md#7-hybrid-scoring-and-token-management), 영문). 재순위화 결과는
질의마다 보고합니다([재순위화 상태](../guides/query.ko.md#재순위화-상태)).

## 답변 형식

`global`은 기본으로 포괄적(comprehensive) 답변 프롬프트를, 나머지 전략은 간결한 프롬프트를
씁니다. 간결한 프롬프트는 한 문장으로 바로 답한 뒤 근거를 짧게 붙입니다
(`custom_prompts.answer_generation_*`). 포괄적 프롬프트는 MS GraphRAG의 global reduce
프롬프트를 본뜬 것으로, 핵심 내용을 모두 설명하는 여러 문단의 마크다운 답변을
씁니다(`custom_prompts.comprehensive_answer_*`). `search.answer_prompt_style`은 이 기본값
위에 합쳐지므로 한 전략만 지정해도 나머지는 그대로입니다.

```yaml
search:
  answer_prompt_style: {drift: comprehensive}   # global stays comprehensive
  query_timeout_seconds: 300                    # see below
# answer_prompt_style: {global: concise}        # concise global answers
```

포괄적 global 답변은 간결한 답변보다 더 많은 내용을 다루지만 덜 직접적이고, 맞비교
LLM 판정은 긴 답변을 선호합니다. 짧은 답을 원하면 `{global: concise}`를 유지하세요. `drift`의
기본값이 간결한 형식인 이유는 포괄적 답변을 쓰면 지연 시간이 기본
`search.query_timeout_seconds` 180초에 가까워지기 때문입니다. `{drift: comprehensive}`를
쓰면 이 마감 시간을 늘리세요(300초면 여유가 있습니다.
[운영 런북](../operations.md#query-deadlines-and-model-timeouts), 영문). 측정 결과는
[기본값과 근거](./defaults-and-evidence.md#answer-style)(영문)에 있습니다.

`auto`는 분기한 전략의 형식을 따릅니다. `answer_match` 점수는 제목이 아닌 첫 줄을 읽으므로(`evaluation.answer_span`)
포괄적 답변은 품질과 관계없이 정확 일치(exact match) 점수가 낮게 나옵니다.

global search는 순위를 매긴 핵심 내용 뒤에 그 내용이 인용한 리포트를 붙여 답변 모델에
넘깁니다. `search.global_search.evidence_tokens`는 이 리포트 본문의 토큰 수를 제한합니다
(가장 점수가 높은 핵심 내용의 리포트부터 리포트 단위로). 기본값은 4000이며, 제한이 없을
때보다 global 질의 비용이 줄었고 측정된 품질 차이는 없었으며 답변은 출처를
유지합니다([근거](./defaults-and-evidence.md#global-evidence-cap), 영문).
`null`(제한 없음)은 컨텍스트 예산이 허용하는 만큼 넣으며, 단면이 크면 global 질의 입력
토큰의 대부분을 차지합니다. `0`(비용이 더 적음)은 MS GraphRAG의 reduce
단계처럼 핵심 내용만 보내며, 이때 답변은 출처를 보고하지 않습니다(핵심
내용은 모델 출력이기 때문입니다). 그래서 `retrieval` 평가기는 global 질의의 순위 지표를 0점
처리하지 않고 건너뜁니다(`no_sources_by_config`).

## 컨텍스트 예산

`search.token_manager.max_context_tokens`(`30000`, upstream LightRAG의 예산)는 답변
프롬프트가 받는 검색 컨텍스트입니다. `models.roles.answer_generation`가 출력 예약분과
함께 받을 수 있는 크기로 줄이며 줄일 때 경고를 남기고, `null`이면 그 모델의 컨텍스트
창에서 도출합니다. 섹션 유형마다 그 일부를 상한으로 받아 점수 순서로 채우고, 한 유형이
쓰지 않은 몫은 후보가 남은 유형에 넘어갑니다([설계 §7](../design.md#7-hybrid-scoring-and-token-management), 영문).
유형별 비율과 `local` 슬롯 할당은
[레퍼런스](../reference/configuration.md#searchtoken_manager)(영문)에 있습니다.
