# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

한국어 문서는 README, 문서 안내, 개요, 검색, 범위와 네임스페이스, 시작하기 두 문서, 설정·수집·다국어 코퍼스·질의·증분 인덱싱·도메인 적응 가이드와 이 사용자 가이드입니다. 설계 문서, 운영 런북, 모델 카탈로그, CLI·설정 레퍼런스와 그 밖의 가이드는 영문으로만 제공합니다.

이 문서는 **unified-kg-rag-on-aws**를 실제로 사용하는 방법을 설명하는 가이드입니다.
unified-kg-rag-on-aws는 대규모 다국어 문서 코퍼스로 지식 그래프를 만들고 그 그래프를
근거로 질문에 답하는 AWS 네이티브 지식 그래프 RAG 프레임워크입니다. 두 가지 검색
방법론인 **Microsoft GraphRAG**(커뮤니티 요약)와 **LightRAG**(이중 레벨 키워드)를
하나의 스택 위에 재구현했으며, 질의마다 방법론을 고를 수 있습니다.

- 프로젝트 소개와 1분 빠른 시작은 [README.ko.md](../README.ko.md)를 참고하세요.
- 모든 문서의 지도는 [문서 안내](./index.ko.md), 개념과 원 논문 대비 확장한 부분은
  [개요](./concepts/overview.ko.md)를 참고하세요.
- 내부 구조와 아키텍처(헥사고날 레이어, 포트와 어댑터, 의존성 규칙)는
  [설계 문서](./design.md)(영문)를 참고하세요.

콘솔 진입점 여섯 개(`pyproject` 스크립트로 정의)는 모두 `application.cli.launch`로
시작해 아래 모듈을 불러옵니다.

| 스크립트 | 모듈 | 용도 |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | 지식 그래프 구축과 갱신 |
| `run-rag` | `application.cli.run_rag_chain` | 그래프 질의 |
| `run-eval` | `application.cli.run_evaluation` | 검색과 생성 평가 |
| `run-visualization` | `application.cli.run_visualization` | 내보낸 그래프 렌더링(수집 없음) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | 도메인에 맞춘 프롬프트 생성 |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | 공개 벤치마크(MuSiQue)를 코퍼스와 `run-eval` 데이터셋으로 변환 |

각 CLI의 `--help` 끝에는 실행 예시와 해당 가이드 링크가 있으며, 모든 플래그는 [CLI 레퍼런스](./reference/cli.md)(영문)에 있습니다.

## 목차

1. 사전 요구사항과 설치: [AWS에서 시작하기](./getting-started/aws.ko.md)([필수 모델](./getting-started/aws.ko.md#필수-모델) 포함), [로컬에서 시작하기](./getting-started/local.ko.md), [Local mode](./guides/local-mode.md)(영문)로 옮김
2. 설정: [설정](./guides/configure.ko.md)으로 옮김(모든 키: [설정 레퍼런스](./reference/configuration.md))
3. 수집: [수집](./guides/ingest.ko.md)과 [다국어 코퍼스](./guides/multilingual.ko.md)로 옮김
4. 질의: [질의](./guides/query.ko.md), [검색](./concepts/retrieval.ko.md), [필터 레퍼런스](./reference/filters.md)(영문)로 옮김
5. 증분 인덱싱: [증분 인덱싱](./guides/incremental.ko.md)으로 옮김
6. [평가 (`run-eval`)](#6-평가-run-eval)
7. [시각화 (`run-visualization`)](#7-시각화-run-visualization)
8. 프롬프트 튜닝: [도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)으로 옮김
9. 도메인 적응: [도메인 적응](./guides/domain-adaptation.ko.md), [Extraction schema](./guides/extraction-schema.md)(영문), [Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)로 옮김
10. [운영과 문제 해결](#10-운영과-문제-해결)
11. [Python 라이브러리로 사용하기](#11-python-라이브러리로-사용하기)
12. [제한 사항](#12-제한-사항)

관련 문서: [CLI 레퍼런스](./reference/cli.md)(영문) · [설정 레퍼런스](./reference/configuration.md)(영문) · [모델 카탈로그](./reference/models.md)(영문) · [운영 런북](./operations.md)(영문) ·
[설계 문서](./design.md)(영문) · [`iac/README.md`](../iac/README.md)

---

## 6. 평가 (`run-eval`)

`run-eval --data-path <dataset.json>`은 데이터셋의 모든 질문에 답한 뒤 답변과 출처를
채점합니다. 평가기 셋은 LLM 없이 비용 없이 동작하고, 둘은 LLM 심사 모델을 씁니다.
전략을 여럿 주거나 `--repeats`를 주면 짝지은 비교가 됩니다.

모든 플래그와 기본값은 [CLI 레퍼런스](./reference/cli.md#run-eval)(영문)에 있습니다.

### 평가기

`evaluation.enabled_evaluators`로 고르며, 다섯 개 모두 기본으로 켜져 있습니다. 결정적
평가기는 데이터셋 필드가 없는 질의를 건너뜁니다. 심사 모델 없이 실행하려면
`enabled_evaluators: [answer_match, retrieval, graph_aware]`로 설정하세요.

| 평가기 | LLM | 데이터셋에 필요한 필드 | 주요 지표 |
|---|---|---|---|
| `answer_match` | 없음 | `answer`(+ `answer_aliases`) | `answer_contains`(대표 지표), `exact_match`, `token_f1` |
| `retrieval` | 없음 | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr`(k = `evaluation.retrieval_k`, 기본값 5) |
| `graph_aware` | 없음 | `expected_entities` / `expected_relationships` | 답변과 검색된 출처에서의 엔티티·관계 커버리지(재현율) |
| `langchain` | 심사 | `answer` | `correctness`, `partial_correctness` |
| `ragas` | 심사 | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |

심사 평가기 둘은 `eval` extra가 필요합니다(`uv sync --extra eval`, [설치](./getting-started/aws.ko.md#설치)). extra가
없으면 `run-eval`은 첫 질의 전에 extra 이름을 알려 주고 상태 코드 `2`로 종료하며,
`processing.ignore_errors: true`여도 같습니다. 위의 심사 모델 없는 실행에는 extra가
필요 없습니다. 그 밖에 켜 둔 평가기를 만들 수 없거나(예: 심사 모델의 Bedrock 접근 권한
없음) 설정을 거부하면 실행이 멈춥니다. `processing.ignore_errors: true`이면 그 평가기를 빼고
`run_manifest.dropped_evaluators`에 기록합니다. 채점은 모든 답변을 생성한 뒤에
시작하므로 평가기 오류가 나도 실행 결과는 버리지 않습니다. 해당 질의의 지표를 실패로
기록하고(쓸 만한 점수가 없는 심사 응답도 0이 아니라 실패), 종료 코드는
`--max-failure-rate`가 정합니다.

#### `ragas` 컨텍스트

- `context_precision`은 컨텍스트마다 심사 호출을 한 번 하므로 순위 상위 출처를 최대
  `evaluation.ragas_max_contexts`개(기본값 20, `null`이면 제한 없음)까지만
  채점합니다(`context_precision@N`). 이 제한이 없으면 출처를 100개 넘게 보고하는 전략은
  `ragas_timeout`에 걸립니다.
- `faithfulness`와 `context_recall`은 `max_context_tokens` 안의 모든 출처를 봅니다.
  기본값(`evaluation.max_context_tokens: null`)에서는 답변 모델의
  예산(`search.token_manager.max_context_tokens`)이므로 답변을 그 답변이 받은
  컨텍스트로 채점합니다. 더 작은 값을 지정하면 경고를 남깁니다.
- 보고서마다 심사 모델이 본 내용을 기록하며(`judge_contexts`, `judge_context_tokens`
  등), 이 설정들은 답변 모델이 본 내용을 바꾸지 않습니다.
- ragas 사용 통계 전송은 꺼져 있습니다(기본값 `RAGAS_DO_NOT_TRACK=true`). 다시 켜려면
  환경 변수 `RAGAS_DO_NOT_TRACK=false`를 지정합니다.

#### `graph_aware`

기대한 그래프 산출물 가운데 생성된 답변에 나타난 비율(커버리지 = 재현율)을
`answer_contains`의 구문 매처로 계산합니다. `{"source": "A", "target": "B"}`나
`"A -> B"`로 준 관계는 답변이 두 끝점을 모두 언급하면 맞은 것으로 보고, 그 밖의 문자열은
구문으로 나타나야 합니다. 자유 형식 답변의 엔티티를 빠짐없이 나열하기 어려우므로 정밀도와
F1은 내지 않습니다.

답변 텍스트의 커버리지는 답변이 길수록 올라가므로, 전략은
`retrieved_entity_coverage` / `retrieved_relationship_coverage`(같은 기대값을 답변
모델이 받은 출처의 엔티티와 관계 끝점에 맞춰 본 값)로 비교하고, 답변 커버리지는
`answer_length`와 함께 읽으세요. 검색 커버리지는 컨텍스트에 그 유형의 출처가 없으면
0이 아니라 건너뛰므로(`global`은 설계상 그러함), 비교에서는 그 지표의 `pairs` /
`unpaired`를, 단일 실행에서는 `metric_outcomes`의 건너뛴 수를 확인하세요.

#### `retrieval`

답변 모델이 본 출처에 정답 문서가 들어 있었는지 봅니다.

- **귀속.** 텍스트 단위 출처는 파일 이름을 직접 가집니다. 엔티티, 관계, 커뮤니티 리포트
  출처는 인용하는 텍스트 단위의 파일에 귀속하며, 그 텍스트 단위를 가장 많이 가진 파일이
  먼저입니다.
- **k는 출처가 아니라 파일 수입니다.** 출처를 순위대로 서로 다른 파일 목록 하나로
  펼치므로, 파일 여러 개에 귀속되는 커뮤니티 리포트가 hit@k를 부풀리지 못합니다.
- **출처 유형별.** `by_source_type`(`grouped_statistics.source_type`에 집계)은 같은
  지표를 섹션 유형별로 계산합니다. 답변 모델이 그대로 읽은 텍스트 단위만 채점하는 `text`
  행이 전략 간 비교에 가장 공정합니다. `attributable_fraction`(귀속할 수 있는 출처 /
  보고된 출처)은 지표가 컨텍스트의 어느 정도를 다루는지 보여 줍니다.
- **매칭.** 파일 이름에 NFKC 정규화와 대소문자 무시를 적용하고 참조의 디렉터리는
  무시합니다. `docs/Terms.pdf`, `terms.pdf`, `terms`는 모두 출처 `Terms.pdf`와 맞고
  `terms.docx`는 맞지 않으며, 버전 접미사는 확장자가 아닙니다(`notes.v1`은
  `notes.v2`와 맞지 않음). `AC/DC` 같은 제목은 통째로 비교합니다.
- **건너뛰기.** 질의에 `reference_sources`가 없거나 귀속할 수 있는 출처가 없으면
  건너뜁니다. 출처를 하나도 가져오지 못한 질의는 0점입니다.

#### `answer_match`

`answer`와 선택 항목 `answer_aliases`(문자열이나 목록, 최상위나 `metadata` 아래에 두며
`metadata`가 우선)에 맞춰 채점하고 그중 최댓값을 씁니다.

- **`answer_contains`**가 대표 지표입니다. 정답이나 별칭이 생성된 답변에 단어 단위
  구문으로 나타나면 1.0입니다. 긴 RAG 답변이 짧은 정답 구간과 똑같은 경우는 드물어서
  exact match보다 정답 여부를 훨씬 잘 따라갑니다. 소유격은 그 단어로 읽습니다(`Depot`은
  `Depot's`와 맞고, `Depots`는 다른 단어). `graph_aware`도 같은 방식으로 맞춥니다.
- **한국어와 CJK.** 단어 뒤의 한국어 조사는 허용합니다(`AWS`는 `AWS는`과 맞음). 숫자만
  있는 정답에는 한국어 단위가 붙어도 되지만(`2024`는 `2024년에`와 맞음) 수의 크기 단위는
  안 됩니다(`3`은 `3억`과 맞지 않음). CJK 문자 옆의 띄어쓰기는 무시하고(`3억 원` =
  `3억원`), 한 단어로 된 CJK 정답은 부분 문자열로 맞추되 더 긴 숫자나 라틴 단어 안에서는
  맞추지 않습니다(`2년`은 `12년`과 맞지 않음).
- **예/아니요 정답**(`yes`/`no`/`nope`, 한국어 `예`/`네`/`아니요`/`아니오`/`아닙니다`,
  중국어 `是`/`对`/`否`/`不是`/`没有`, 일본어 `はい`/`いいえ`)은 HotpotQA처럼 질문에
  답하는 첫 예/아니요 단어의 극성으로 채점합니다(`there is no clause`, `네 가지`,
  `(예: …)`, `No. 5`는 답이 아니므로 `네, 맞습니다`는 예로 답한 것이지만
  `네 맞습니다`는 아님).
- **`exact_match`와 `token_f1`**은 답변 가운데 `evaluation.answer_span`이 가리키는 부분을
  채점합니다. SQuAD, HotpotQA, MuSiQue는 모델이 예측한 짧은 답(`Depot` 같은 구간)을 정답과
  비교하는데, 답변 프롬프트는 한 문장 답에 근거를 덧붙여 쓰므로 답변 전체를 채점하면 정답
  여부만큼 답변 길이를 재게 됩니다.
  - `first_line`(기본값, 비용 없음): 비어 있지도 않고 Markdown 제목도 아닌 첫 줄, 즉 한
    문장 답입니다(`Depot stores widgets for Vendor.`는 `Depot`에 대해 token F1 1/3. EM은
    여전히 낮게 나옴).
  - `extracted`: `models.roles.answer_span` 모델(fast 계층. 프롬프트
    `custom_prompts.answer_span_extraction_*`)이 답변에서 짧은 구간을 그대로 옮겨 옵니다.
    서로 다른 질문·답변마다 한 번 호출하고 반복과 전략 사이에서 다시 씁니다. 호출이
    실패하거나, 빈 응답이거나, 답변에 없는 구간이면 첫 줄로 대신 채점하고
    `answer_span_statistics.fallbacks`에 셉니다. 논문 수치에 가깝게 보려면 이 설정을
    쓰세요.
  - `full`: 답변 전체.

  리포트마다 `answer_span`과 채점한 `scored_span`이 기록됩니다. 정규화는 SQuAD v1.1
  방식(NFKC, 소문자화, 문장 부호와 영어 관사 제거, 공백 압축)이되 의미를 바꾸는 문장
  부호(`-5`, `1.5`, `C++`)는 남깁니다. `evaluation.answer_normalization: official_squad`는
  공식 스크립트(MuSiQue 평가도 같음)이고, `hotpotqa`는 여기에 `yes`/`no`/`noanswer`인
  구간이나 정답은 둘이 같을 때만 token F1을 받는 HotpotQA 규칙을 더합니다. token F1은
  공백으로 나누므로 중국어와 일본어에서는 exact match와 같아지고, 한국어 조사가 붙으면 두
  지표 모두 낮게 나옵니다.

### 평가 데이터 형식

객체의 JSON 배열입니다. `question`만 필수이며, `expected_entities` /
`expected_relationships`는 `graph_aware`에만 필요합니다.

```json
[
  {
    "id": "q1",
    "question": "What are the main themes discussed in the documents?",
    "answer": "The main themes include AI, machine learning, and data processing.",
    "category": "general",
    "difficulty": "easy",
    "reference_sources": ["doc1.pdf", "doc2.txt"],
    "expected_entities": ["AI", "machine learning", "data processing"],
    "expected_relationships": [
      { "source": "AI", "target": "machine learning" },
      "machine learning -> data processing"
    ],
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- 항목별 `metadata`(예: `search_strategy`)는 그 질문에 한해 CLI 기본값보다 우선합니다.
  `id`는 `query_id`로 써도 됩니다.
- 코퍼스로 답할 수 없는 질문에는 `"metadata": {"answerable": false}`를 지정하세요. 어떤
  평가기로도 채점하지 않고 `abstention_statistics`(아래)에만 셉니다.
- 파일은 질의를 실행하기 전에 검증합니다. 읽을 수 없는 파일, 비어 있거나 배열이 아닌
  데이터셋, `question` 누락, 중복 ID, 타입이 틀린 필드(목록이 아닌 `expected_entities`,
  문자열이나 문자열 목록이 아닌 `answer_aliases`), RAG 체인이 거부하는 `metadata`(알 수
  없는 `search_strategy`, 또는 체인이 거부하는 `filters`: 빈 목록 같은 잘못된 값이나
  항목의 전략이 거부하는 키. 비교 실행이면 비교하는 모든 전략으로 확인)가 있으면
  `Error: invalid evaluation dataset:`과 항목 번호, ID를 출력하고 종료 코드 2로 멈춥니다.

### 예시

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

두 경로 모두 `s3://` URI를 받으므로 `run-eval`도 CDK 스택에서 일회성 태스크로
실행할 수 있습니다([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
결과는 실행이 끝날 때 `aws.s3.encryption` 설정으로 올리며, 업로드가 실패하면 0이 아닌
코드로 끝나고 로컬 파일을 남깁니다(경로를 출력). S3에는 설정의 `aws.profile_name` /
`aws.region_name`으로 접속하며, 셸의 `AWS_PROFILE` / `AWS_REGION`은 설정이 비워 둔 경우에만
그 값을 채웁니다. 실행 주체에는 데이터셋에 대한
`s3:GetObject`와 접두사에 대한 `s3:PutObject` 권한이 필요합니다.

결과는 `evaluation_{results,reports,summary}_<timestamp>.json`으로 씁니다(모든 질의가
한 전략을 요청했거나 썼다면 `..._<strategy>_<timestamp>.json`, `auto` 실행은
`..._auto_...`). `<timestamp>`는 `YYYYMMDD_HHMMSS_<16진수 8자리>`이므로(같은 초에 시작한
실행끼리 겹치지 않음) 최신 실행은 이름을 만들어 내지 말고 이름을 정렬해 찾으세요. 요약에는
지표별 통계(`metric_statistics`)와 채점/실패/건너뜀 개수(`metric_outcomes`)가 있고, 다음
항목도 들어 있습니다.

- `grouped_statistics`: 같은 통계를 `search_strategy`(실제로 쓴 전략. `auto`에서는
  질의마다 다름), `category`, `difficulty`, `source_type`(출처 섹션 유형별 검색 지표)으로
  나눈 값입니다.
- `abstention_statistics`: 체인이 답변 대신 고정된 컨텍스트 없음 응답("I could not find
  relevant information…")을 반환한 빈도입니다. `abstained`, `answered`,
  `abstention_rate`, `per_strategy`가 있고 `answerable: false` 항목에는 `unanswerable`
  블록이 있습니다. 체인은 검색 결과가 없을 때만 이 응답을 반환하고 모델이 자기 말로 거절한
  것은 감지하지 않으므로, 데이터가 색인된 상태에서 `unanswerable` 비율은 올바른 거절의
  하한입니다. 각 결과에는 `abstained`가 기록됩니다.
- `answer_span_statistics`: exact match와 token F1이 읽은 `answer_span`, 채점한 리포트
  수(`scored`), 추출에 실패해 첫 줄로 채점한 수(`fallbacks`)입니다.
- `rerank_status_counts`: [재순위화 상태](./guides/query.ko.md#재순위화-상태)별 질의 수입니다. 각 결과에는
  `rerank_status`가 기록됩니다(답변 생성이 먼저 실패했으면 `null`).
- `guardrail_blocked_counts`: `input` / `output`별 차단된 질의 수입니다(차단된 항목은
  차단 메시지로 채점. 차단된 답변은 출처를 버리므로 검색 지표와 RAGAS 컨텍스트 지표
  (`context_precision`, `context_recall`, `faithfulness`)는 `guardrail_blocked`로 건너뜀.
  입력에서 차단된 질의는 아무것도 검색하지 않았으므로 검색 실패로 보아 컨텍스트 정밀도·재현율이
  0이고, `faithfulness`는 컨텍스트 없는 다른 답변처럼 건너뜀: `no_contexts`). 각 결과에는
  `guardrail_blocked`가 기록됩니다.
- `run_manifest`: 무엇을 측정했는지 기록하므로 두 실행을 비교할 수 있습니다.
  - CLI 인자(URL은 자격 정보를 뺌), 버전, `git_sha` / `git_dirty`, 데이터셋(경로, 해시,
    질의 수), UTC 타임스탬프, `dropped_evaluators`
  - 모델 ID와 `judge_is_answer_model`(심사 모델이 답변 모델이면 심사 점수가 부풀려집니다.
    Bedrock의 기본 심사 모델인 Opus 5.5는 기본 답변 모델과 다르고, 답변 모델이 Opus 5.5이면
    다른 모델로 바뀝니다. 답변 모델과 같은 `models.roles.evaluation`을 직접 지정하면(`default`
    등) 그대로 쓰고 경고합니다).
    비교의 `--pairwise-judge`도 심사 모델로 세며, `pairwise_judge`에 그 설정이 기록됩니다.
  - `answer_scoring`: exact match와 token F1에 쓴 `answer_span`과
    `answer_normalization`(`answer_match` 사용 시), `models.answer_span`은 구간을 추출한
    모델입니다. 채점 방식이 다른 실행은 이 두 지표로 비교할 수 없습니다.
  - `retrieval_config_sha256`: 답변이나 점수를 바꿀 수 있는 모든 구성 설정의 해시(모델 ID와
    수집 설정 포함, 리전·엔드포인트·자격 증명·경로·시간 제한 같은 운영 설정은 제외)이므로
    같은 평가는 어느 환경에서든 같은 해시가 됩니다. `run-eval`의 질의 플래그(`--suffix`,
    `--top-k`, `--search-type`, `--retrieval-multiplier`, `--enable-thinking`)는 구성이
    아니므로 해시에 들어가지 않고 CLI 인수에 기록됩니다.
  - `index`: 질의한 접미사마다 벡터 인덱스를 마지막으로 쓴 수집 실행(`pipeline_id`,
    `mode`, `built_at`, `documents`, 추출 설정, 증분 실행이면 그 실행이 올라선
    `full_build`, `communities` 카운터.
    [커뮤니티 증가](./guides/incremental.ko.md#커뮤니티-증가-실행-누적) 참고)입니다. 최선 노력 방식이며 인덱스 메타데이터가 없으면 `null`입니다.

  `EvaluationManager.evaluate_dataset`이 만들므로 라이브러리 호출자도 받습니다
  (`dataset_path=` / `cli_args=`를 넘김).

질의별 리포트에는 평가기마다 `metrics`가 있고(서로 무관한 지표의 평균은 의미가 없으므로
종합 점수는 없음), 보고된 출처마다 순위 순으로 `retrieved_source_ids`(귀속된 파일 이름,
귀속할 수 없으면 `[]`), `retrieved_source_types`, `retrieved_entities`,
`retrieved_relationships`가 기록됩니다.

### 전략 비교

`run-eval --search-strategy local,mix,global --repeats 3`은 같은 질의를 전략마다
`--repeats`번 답하고 채점하며(지정한 전략이 항목별 `search_strategy`보다 우선. 전략은 플래그 하나에
나열하며, `--search-strategy`를 두 번 주면 사용법 오류), 실행마다
결과 파일을 저장하고 `evaluation_comparison_<timestamp>.json`을 추가로 씁니다.

- `strategies.<s>.metrics.<evaluator>/<metric>`: 질의별 반복 평균을 질의 전체로 평균한
  값과 `repeat_stdev`(실행 간 잡음)입니다. `routed_strategies`는 `auto`가 어떻게
  분기했는지 보여 줍니다. 체인은 질문마다 엔티티와 키워드를 한 번만 추출하므로
  (`search.query_extraction_cache_size`, 질문 1024개), 반복 실행과 이를 읽는 전략(`mix`와
  `hybrid`는 키워드, `local`은 엔티티)은 같은 엔티티·키워드로 검색하고, 실행 간 잡음은
  검색과 답변에서만 생깁니다.
- `deltas_vs_baseline.<s>.<metric>`: 두 전략이 모두 채점한 질의에서 첫 전략과 짝지어
  비교한 값입니다. `mean_delta`, 95% 짝지은 부트스트랩 구간(`ci_low`/`ci_high`. 2,000번
  재표집, 시드 0), `wins`/`ties`/`losses`, 질의별 차이, `unpaired`가 있습니다. 답변
  생성에 실패한 질의는 다른 실행이 채점한 지표 가운데 자기 전략이 어느 질의에서든 채점하는
  지표에서 0점이므로(`answer_length` 제외) 실패로 이득을 볼 수 없고, 그 전략이 만들지 않는
  지표(`global`은 엔터티를 보고하지 않으므로 `retrieved_entity_coverage`가 없음)는 손실로
  지어내지 않습니다. 아무것도 채점하지 못한 전략은 모든 지표에서 0점입니다. 이 규칙은
  데이터로 판단하므로 실패한 질의만 채점할 수 있던 지표(`expected_relationships`가 있는
  항목이 하나뿐인 경우 등)도 0점으로 채우지 않습니다(`failed_answers` / `failed_queries`.
  차단 수는 `guardrail_blocked_counts`. 전략마다 실패하거나 차단된 수가 다르거나, 실행마다
  exact match와 token F1이 읽은 `answer_span`이 다르거나, `answer_span: extracted`에서 추출한
  구간이 첫 줄로 대체된 전략이 있거나(`answer_span_fallbacks`) 그 대체 비율이 전략마다 다르면
  `warnings`에 나옴). 구간이 0을 포함하지 않으면 그 차이가 뒷받침됩니다.
  이진 지표의 10%p 차이는 질의 50개에서 15%, 200개에서 59%의 실행에서만 찾아지므로 수백
  개 질의로 비교하세요.
- `deltas_by_category.<s>.<category>.<metric>`: 같은 차이를 데이터셋 `category`(MuSiQue의
  홉 수 등)별로 나눈 값입니다. 구간은 범주마다 따로 구하고 범주와 지표 수에 따른 다중 비교
  보정을 하지 않으므로, 여러 개 가운데 하나만 0을 벗어난 결과는 탐색적으로 읽으세요.
- `pairwise_vs_baseline.<s>`(`--pairwise-judge` 사용 시): GraphRAG / LightRAG 논문의
  맞비교 LLM 판정으로, `models.roles.evaluation`이 첫 반복의 답변을 포괄성
  (comprehensiveness), 다양성(diversity), 이해 도움(empowerment), 직접성(directness)으로
  판정합니다(`evaluation.pairwise_overall: true`이면 LightRAG의 `overall`도 추가. 프롬프트:
  `custom_prompts.pairwise_comparison_*`). 쌍마다 순서를 바꿔 두 번 판정하고 두 순서가
  일치한 기준만 승리로 셉니다. `position_consistency`(전체와 기준별)는 판정한 쌍 가운데 두
  순서가 일치한 비율이며, 0.5 미만이면 경고합니다. 그런 판정 모델은 위치로 결정하므로
  무승부가 답변에 대해 말해 주는 것이 없습니다. `candidate_win_rate`는 결정된 쌍을 승리 1,
  무승부 0.5, 패배 0으로 매긴 평균이고, 이 질의별 점수에 대한 부트스트랩 구간
  (`ci_low`/`ci_high`)이 함께 나옵니다. 판정 모델이 결정한 쌍이 없으면 `null`입니다. 한쪽
  답변이 실패한 쌍은 판정하지 않고 다른 쪽이 이깁니다(`candidate_errors`,
  `baseline_errors`. 둘 다 실패: `both_errors`). `answerable: false` 항목은 판정하지
  않습니다(`unanswerable_skipped`). `failed`는 판정 모델이 결정하지 못한 쌍(호출 실패, 또는
  `1`, `2`, `tie`가 아닌 판정)의 수이며 `--max-failure-rate`가 적용됩니다. 판정 모델이 답변
  모델과 같으면 경고합니다.
- `cost.strategies.<s>`: 답변의 질의당 모델 사용량을 모든 반복의 모든 답변에 대한
  `mean` / `p50` / `p95`로 보여 줍니다. `llm_calls_per_query`, `input_tokens_per_query`,
  `output_tokens_per_query`, `total_tokens_per_query`, `latency_seconds`, `totals`,
  `cache_hits`가 있고, `evaluation.prices`를 설정하면 `usd_per_query`와 `usd_total`도
  있습니다. `cost.deltas_vs_baseline.<s>`는 지표처럼 기준과 짝지어 비교합니다
  (`usage/llm_calls`, `usage/input_tokens`, `usage/output_tokens`, `usage/total_tokens`,
  `usage/cost_usd`, `latency/response_time_seconds`. 질의별 반복 평균, 부트스트랩 구간
  포함). 실패하거나 시간 초과된 답변은 실패하기 전까지 쓴 만큼 셉니다. 답변마다 사용량 범위와
  타이머를 따로 두므로 그 결과에는 시도들의 호출·토큰과, 첫 시도 시작부터 마지막 시도 끝까지의
  시간이 남습니다(`failed_answers`가 그 수이며 경고로도 알림). 그래서 실패했다고 더 싸거나
  빨라 보이지 않지만, 빨리 실패한 답변은 실제로 비용이 적으므로 비용은 `failed_answers`와 실패
  경고와 함께 읽으세요. 저장된 답변 가운데 사용량 기록이나 지연 시간이 없는 답변은
  `answers_without_usage` / `answers_without_latency`로 세고 경고하며, `usage/*`나
  `latency/*` 측정값의 `--fail-if`는 실패합니다. 한 실행의 전략과 반복은 체인의 질의 추출
  캐시를 공유하므로, 앞선 실행의 추출을 재사용한 뒤쪽 실행은 그 호출 비용을 내지 않으며
  (`cache_hits`에 셈) 비용이 앞서 실행된 것에 따라 달라집니다. 반복 횟수나 답변당 캐시 적중
  수가 기준과 다른 전략은 경고합니다. 캐시 없는 비용을 비교하려면
  `search.query_extraction_cache_size: 0`으로 설정하세요. 품질 차이는 이 설정과 무관합니다.
  `run-eval`은 전략마다 첫 실행 전에 그 전략으로 합성 질의 하나에 답하며, 이 질의는 시간을
  재지 않고 채점하지 않으며 사용량에도 넣지 않습니다(`evaluation.warm_up_query`, 기본값
  `true`). 그래서 첫 전략의 지연 시간에 저장소 연결을 열고 모델을 처음 호출하는 시간이 들어가지
  않습니다. 실행마다 전략당 질의 하나(그 모델 호출)만큼 비용이 더 들며, 이 비용은 실행의
사용량에도 USD 추정치에도 들어가지 않습니다(Bedrock 청구에는 들어감). `false`로 끌 수 있습니다.
- 첫 실행의 `run_manifest`. `--pairwise-judge`를 쓰면 `pairwise_judge_model`과 판정 모델의
  `pairwise_usage`도 있습니다.

비교는 `evaluation_comparison_<timestamp>.csv`(기준이 아닌 전략, 측정값, 질의마다 한 행.
`baseline`, `baseline_value`, `value`, `delta`, `category`)와 `.md`(지표 표와 비용 표, 맞비교
판정의 승률, 게이트, 경고. 풀 리퀘스트나 CI 작업 요약용)로도 저장합니다.

패키지에는 모델 가격이 들어 있지 않습니다. USD 추정치가 필요하면 사용하는 리전의 현재
[Amazon Bedrock 가격](https://aws.amazon.com/bedrock/pricing/)을 100만 토큰 단위로,
사용량 기록이 쓰는 모델 ID(`usage.by_model`. 팩터리가 추론 프로필을 확정했으면
`global.<model>` 같은 프로필 ID)를 키로 채우세요. 가격이 없는 모델을 쓴 답변은 추정치가
없고 비교가 경고합니다. `input_tokens`에는 프롬프트 캐시 읽기·쓰기 토큰이 들어 있습니다.
그 단가(`cache_read_per_million_tokens`, `cache_write_per_million_tokens`)를 주면 따로
계산하고, 없으면 입력 단가로 계산합니다(Claude에서 캐시 읽기는 입력의 10분의 1이므로 상한값).

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
```

추정치는 사용량에 기록된 호출만 다룹니다. botocore가 다시 보낸 요청(`request_retries`), 실패한
호출(`failed_calls`), 예열 질의는 가격에 들어가지 않으므로 CloudWatch 호출 수와 청구액이 조금
더 많을 수 있습니다(패키지의 2026년 11월 벤치마크 실행에서 Sonnet 5.5는 기록된 호출보다
호출 수가 13% 많았고 입력 토큰은 4.6%, 출력 토큰은 0.9% 이내였으며, Haiku 5.5는 0.2% 이내).

콘솔에도 같은 요약이 출력되며, 실패 허용 기준은 모든 실행에 적용됩니다.

### 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트

`run-eval --from-results <dir|files>`는 이전 실행이 저장한 파일(`evaluation_results_*`와 그 옆의
`evaluation_reports_*`, `evaluation_summary_*`. 기본값인 `evaluation.save_detailed_results:
true`일 때 저장됨)로 질의에 답하거나 저장소를 열지 않고 비교를 다시 만듭니다. 실행을 전략별로
묶어 가장 오래된 실행이 나열한 순서로 비교하므로(`--search-strategy`를 주면 그 전략을 그 순서로
비교), 중간에 멈춘 비교는 빠진 전략만 같은 디렉터리에 같은 `--repeats`로 실행한 뒤 다시 만들면
끝납니다. 이렇게 만든 차이는 중단 없이 실행한 비교와 같습니다. 실행이 모두 사라진 전략은
`--repeats N`으로 다시 실행하고, 반복 일부가 저장된 전략은 빠진 반복만 실행하거나(남은 횟수를
`--repeats`로) 그 전략의 일부 파일을 지운 뒤 N번 모두 실행하세요.

```bash
# naive 실행 중에 멈춤(--repeats 3으로 실행): naive만 같은 디렉터리에 같은 --repeats로
# 실행하고 다시 만들기.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# 저장한 답변을 다른 판정 모델로 다시 판정(답변 호출 없음).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# CI: 이전 실행과 비교해 뒷받침되는 성능 저하가 있으면 실패.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- 한 전략의 저장된 실행들은 그 전략의 반복으로 합칩니다. 이때 질의 플래그(`suffix`, `top_k`,
  `search_type`, `retrieval_multiplier`, `enable_thinking`), `retrieval_config_sha256`,
  `git_sha`, `index` 빌드나 답변 모델이 서로 다르면(설정이 다른 실행은 다른 디렉터리에 두거나
  결과 파일을 직접 나열), 전략마다 반복 횟수가 다르면, `evaluation_summary_*` 파일이 없는
  실행이 있으면(매니페스트가 없어 이 확인과 답변 채점·자기 선호 확인을 할 수 없음) 경고합니다.
- `--from-results`와 `--pairwise-judge`를 함께 쓰면 저장한 답변을 다시 판정합니다
  (`--judge-model`은 여기서나 일반 실행에서나 그 실행의 `models.roles.evaluation`을 정함.
  `--pairwise-judge`도 LLM 판정 평가기도 없으면 아무것도 판정하지 않으므로 경고함).
  자기 선호 경고는 그 판정 모델을 저장된 실행의 답변 모델과 비교합니다.
- `--baseline-results`는 이전 실행을 기준으로 삼습니다. 그 실행의 전략마다
  `baseline:<strategy>`라는 이름을 붙이고, 이번 실행이나 `--from-results`의 각 전략을 같은
  전략의 기준과 비교합니다(`deltas_vs_baseline.<s>`와 `cost.deltas_vs_baseline.<s>`는
  `baseline:<s>`와의 차이이며, `baselines`가 전략마다 기준을 알려 줌). 이전 실행에 없던 전략은
  그 실행의 첫 전략과 비교하고 경고합니다. 기준 실행은 비교할 실행과 같은 `--repeats`로
  만드세요. 반복 실행은 질의 추출 캐시를 재사용하므로 횟수가 다르면 사용량 차이가 달라집니다
  (경고로 알림). 실행마다 디렉터리를 따로 두세요. `--output-directory`가 `--baseline-results`
  디렉터리이면, 다음에 그 디렉터리를 기준으로 읽는 실행이 이번 결과를 반복으로 합치므로
  경고합니다. exact match와
  token F1이 답변을 다르게 읽은 실행(실행 매니페스트의 `answer_scoring`:
  `evaluation.answer_span`, `answer_normalization`)은 거부하며(종료 코드 2), 답변을 생성하기
  전에 확인합니다. 데이터셋이 다른 실행은 경고합니다(공통 질의 ID만 짝지음).
- `--fail-if <measure>:<stat><op><value>`(여러 번 지정 가능)는 비교한 전략의 짝지은 차이가
  조건을 만족하면 종료 코드 1로 끝냅니다. `<measure>`는 `deltas_vs_baseline`의 키
  (`<evaluator>/<metric>`)나 `cost.deltas_vs_baseline`의 키이고, `<stat>`은 `mean_delta`,
  `ci_low`, `ci_high` 가운데 하나입니다. `ci_high<-0.02`는 구간 전체가 -0.02 아래일 때, 즉
  질의 표본이 뒷받침하는 성능 저하일 때만 실패합니다. 그 측정값의 차이가 없는 전략도 게이트에서
  실패하며, 전략이나 그 기준의 일부 답변에 값이 없는 비용 측정값(`usage/*`, `latency/*`)도
  실패합니다. 전략과 그 기준의 실패한 답변 수가 다를 때도 비용 측정값 게이트는 실패합니다.
  실패한 답변도 실패 전까지 쓴 비용으로 세므로 그 차이에는 실패의 차이도 섞이기 때문입니다. 결과는 `comparison.gates`와 JSON의 `gate_failures`에 있습니다. 비교가 있어야
  하므로 전략 두 개 이상이나 `--baseline-results`가 필요합니다.

### 벤치마크 (MuSiQue)

`prepare-benchmark musique`는 [MuSiQue](https://github.com/StonyBrookNLP/musique)(답변
가능, 다중 홉, CC BY 4.0. Trivedi 외, TACL 2022)를 코퍼스와 `run-eval` 데이터셋으로
바꿉니다. 데이터는 이 저장소에 포함하지 않습니다. 기본적으로 공개 미러의 고정 리비전에서
답변 가능 dev 분할(질문 2,417개)을 내려받아 SHA-256을 확인하며, `--data-path`로 로컬
`musique_ans_v1.0_*.jsonl`을 대신 읽을 수 있습니다. 전체 파일(`musique_full_v1.0_*.jsonl`)도
읽지만 그 답변 가능한 절반은 MuSiQue-Ans가 아니므로(단락과 답변 별칭이 다름) 행에
`musique` 대신 `metadata.benchmark: musique-full-answerable`을 붙이고, 점수를 MuSiQue-Ans
결과와 비교할 수 없다는 WARNING을 남깁니다. 이 레이블은 출력(콘솔과
`--output-format json`)에 나오며, `run-eval`은 요약(`benchmarks`, 여러 전략이나 반복의 비교 표
아래에도 출력)과 실행 매니페스트(`dataset.benchmarks`)에 레이블별 질문 수를 남깁니다. `corpus/`(서로 다른 단락마다 텍스트
파일 하나. 질문마다 근거 단락 2-4개와 방해 단락 최대 18개)와 `eval_data.json`(`answer`,
`answer_aliases`, 근거 단락인 `reference_sources`, 홉 수인 `category`)을 씁니다.
`corpus/`가 비어 있지 않으면 아무것도 내려받기 전에 사용 오류(종료 코드 2)로 끝납니다.

```bash
# 1. 시드 0으로 질문 100개(단락 약 1,900개) 표집
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. 단락을 별도 접미사로 수집
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. 두 방법론을 짝지어 비교하고 EM / F1은 짧은 답 구간으로 채점
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

`musique.yaml`에는 평소 설정에 다음을 더합니다.

```yaml
processing:
  document_parsing:
    index_value: musique          # 벤치마크를 기존 인덱스와 분리
evaluation:
  answer_span: extracted                 # 답변이 제시한 짧은 구간으로 EM / F1 채점
  answer_normalization: official_squad   # MuSiQue 평가 스크립트의 정규화
  enabled_evaluators: [answer_match, retrieval]   # 판정 모델 없이. 필요하면 langchain / ragas 추가
```

MuSiQue는 HotpotQA, SQuAD처럼 독해(reader) 모델이 예측한 짧은 답을 정답과 별칭에 대해 공식
정규화의 EM과 token F1로 채점합니다. RAG 답변은 한 문장 답과 근거이므로
`answer_span: extracted`(답변마다 fast 모델 호출 한 번)가 먼저 모델로 구간을 옮겨 옵니다.
이렇게 하면 수치가 논문에 가까워지지만, 논문의 독해 모델은 구간을 출력하도록 학습되었으므로
같지는 않습니다. 기본값 `first_line`은 비용이 없지만 한 문장 답을 채점하므로 EM이 논문보다
훨씬 낮습니다. GraphRAG와 LightRAG 논문은 EM / F1을 보고하지 않고 맞비교 LLM 판정
(`--pairwise-judge`)을 씁니다.

`answer_contains` / `token_f1`과 `retrieval`(`category`별, 비교에서는
`deltas_by_category`)을 읽으세요. 수집 비용은 단락
수(질문당 약 19개)에 비례합니다. 결과를 발표할 때는 논문을 인용하세요.

---

## 7. 시각화 (`run-visualization`)

이미 내보낸 시각화 데이터 JSON으로 그림을 그리는 **독립** 렌더러이며, 수집을 다시
실행하거나 AWS에 접근하지 **않습니다**. 모든 플래그와 기본값은
[CLI 레퍼런스](./reference/cli.md#run-visualization)(영문)에 있습니다.

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

`viz` extra가 필요합니다(`uv sync --extra viz`, [설치](./getting-started/aws.ko.md#설치)). 렌더러는
**`interactive`**(pyvis)와 **`static`**(Bokeh) 두 개이며 `graph.visualization`
아래(`interactive.*`, `static.*`, `centrality.*`)에서 설정합니다. 수집 단계는 직접
렌더링하지 않습니다.

- **입력.** `run-ingestion`의 커뮤니티 탐지 단계는 그래프가 코퍼스 전체를 담는
  실행(전체 빌드, 초기화, 모든 문서가 새 문서이거나 변경된 증분 실행)마다
  `visualization_data.json`을 씁니다. 변경되지 않은 문서를 건너뛴 증분 실행은 아무것도
  쓰지 않고(INFO 로그만 남김) 이전 파일을 그대로 둡니다. 전체 그래프를 보려면 전체
  빌드(`indexing.reset: true`)를 한 번 실행하세요.
- **위치.** `graph.visualization.outputs_directory`이며 기본값은
  `<캐시 디렉터리>/<pipeline_id>/visualization/`입니다. S3 캐시 동기화는 `.json`만
  올리므로 HTML은 로컬에서 다시 렌더링하세요.
- **내용.** 노드(레벨 0 `community_id`로 색을 정함), 간선, 커뮤니티 계층이며 벡터는
  뺍니다. 파일을 쓸 때 모델을 호출하지 않으며, `run-visualization`이 렌더링할 때 그래프
  구조로 시드를 고정한 spring 레이아웃과 `centrality` 지표를 계산합니다.
- **크기.** 대화형 그래프는 차수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만
  남기고 그 노드만 배치합니다.
- 간선 두께와 불투명도는 그래프 자체의 가중치 범위에 맞춰 로그 스케일로 조정합니다.
- 렌더러가 실제로 쓴 파일만 보고하며(같은 이름의 기존 파일은 먼저 지움), 아무것도
  렌더링하지 못하면(예: 빈 그래프) 상태 코드 `1`로 종료합니다.

---

## 10. 운영과 문제 해결

[IAM 권한](./getting-started/aws.ko.md#iam-권한)은 AWS에서 시작하기에, 오류 메시지와 해결 방법은
[운영 런북: 자주 보는 오류](./operations.md#common-errors)에 있습니다.

### CLI 종료 코드와 JSON 출력

모든 CLI(`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`)는 다음 코드로 종료합니다.

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 실행 실패: 단계나 질의 실패(`run-ingestion`은 단계에서 발생한 `ConfigurationError`와 연결할 수 없는 저장소도 포함), `--max-failure-rate` 초과, `run-eval --fail-if` 게이트 실패, 업로드 실패, 렌더링한 파일 없음 |
| `2` | 사용법·설정 오류: 잘못된 플래그, 없거나 잘못된 입력(`run-eval` 데이터셋, `run-visualization` 데이터 파일, `prepare-benchmark` 데이터 파일), 잘못된 설정 파일이나 덮어쓰기, 설정하지 않은 저장소 엔드포인트, 선택한 제공자에 필요한데 설치하지 않은 extra(`local`)(모두 작업 시작 전), 또는 실행 중 단계 밖에서 발견한 `ConfigurationError`. 예: `run-ingestion`이 첫 모델 호출 전에 실행을 거부한 경우(실행에 필요한 인덱스를 검색 엔진이 만들지 않거나, 삭제만 있는 실행을 포함해 증분 실행의 현재 인덱스가 다른 임베딩·언어·레지스트리 설정으로 만들어짐) |
| `130` | Ctrl-C(SIGINT)로 중단 |
| `143` | SIGTERM으로 중지(ECS `StopTask`, Step Functions 시간 초과). `run-ingestion`은 끝난 단계를 먼저 체크포인트로 저장하고, `run-eval`은 끝난 실행의 결과를 `s3://` 출력에 업로드함(실패하면 로컬에 남기고 경로를 로그로 남김) |

`130`이나 `143`으로 중지된 실행은 프로세스를 바로 끝냅니다. 진행 중인 모델 호출은 기다리지 않고 버리며, 그 전에 로그, 추적 데이터, 표준 출력을 비우고 프로세스 풀의 작업자도 함께 끝냅니다. SIGTERM과 Ctrl-C는 질의 CLI의 작업을 다음 await 지점에서 취소하므로 정리 작업(저장소 연결 닫기, `run-eval`의 업로드)은 실행됩니다(대화형 프롬프트에서 기다리는 중이면 SIGTERM은 세션을, Ctrl-C는 대화를 바로 끝냄). SIGTERM이나 이벤트 루프 실행 중의 Ctrl-C로 중지된 질의 CLI가 100초 뒤에도 남아 있으면 그때 끝내며(앱 컨테이너의 StopTimeout 120초 안), 출력한 문서가 없으면 상태 문서를 출력합니다. 두 번째 신호는 바로 끝냅니다. `run-ingestion`은 대신 SIGTERM이나 Ctrl-C를 받으면 진행 중인 체크포인트나 메타데이터 쓰기를 마친 뒤 멈추며, StopTimeout이 그 상한입니다. 끝난 CLI는 종료 처리기를 실행한 즉시 자기 종료 코드로 프로세스를 끝내므로, 종료 중에 받은 신호가 `0`, `1`, `2`를 `-15`나 `-2`로 바꾸지 못합니다.

플래그는 전체 이름으로 써야 하며, `--output`처럼 앞부분만 쓰면 거부합니다.

`--output-format json`은 JSON 문서 하나를 표준 출력에 쓰고 배너·진행 상황·로그는 표준
오류로 보내므로 출력을 `jq`로 바로 넘길 수 있습니다(표준 출력에 함께 쓰는
`--metrics-sink cloudwatch`와는 같이 쓸 수 없음). 담기는 내용은 답변이나 검색
결과(`run-rag`), 파이프라인 ID와 상태, 실행이 `FAILED`로 기록한 문서 수(`failed_documents`. 모든
단위를 실패시킨 모델 장애는 아무것도 기록하지 않고, 일부 단위만 실패시킨 장애는 그 문서들을 기록하며
다음 실행이 `indexing.max_document_failures`에 세지 않고 다시 처리함), 단계가 실패한 텍스트 단위 수(`units_failed`. 그 단계의 출력 없이 인덱싱되고 다음 실행이 다시 추출함. 레지스트리가 문서를 `FAILED`로 기록하지 않아 `failed_documents`가 0인 전체 빌드에서도 셈. 실행 끝에 단계와 원인별 개수를 WARNING으로 알림), 문서 자체의 것이 아닌 오류로 연속 실행에서
또 실패한 문서 수(`documents_retrying`. 장애나 설정 때문이면 제외되지 않음. 파일과 실행 횟수를 경고로도 알림), 일부 텍스트 단위의 주장 추출이 단위 자체의 오류로 실패해 그 주장 없이 인덱싱된 문서 수(`documents_without_claims`. 실패로 세지 않음. 요약에도 표시), 내용과 추출 설정이 그대로인 채
`indexing.max_document_failures`번 연속 실패해 이번 실행에서 빠진 문서 수(`documents_excluded`. 설정을 바꾸면 다시 셈. 요약에도 표시),
이번 실행이 수행한 단계의 모델 호출과 토큰 수(`token_usage`),
단계별 결과(`run-ingestion`), 실행별 요약과 비교,
실패 예산 초과 내역과 실패한 게이트(`gate_failures`)(`run-eval`), 쓴 파일(`run-visualization`), 출력 경로와 프로필
(`run-prompt-tuning`), 쓴 항목 수(`prepare-benchmark`)입니다. 문서를 출력하기 전에 신호로 중지되면
대신 `{"status": "interrupted", "exit_code": 143}`(Ctrl-C는 `130`)을, 출력하기 전에
실패하면 `{"status": "failed", "exit_code": 1}`(또는 `2`)에 실패 원인 `error`(argparse 오류 포함)를
붙여 출력하므로 표준 출력은 항상 JSON입니다. 시작 직후에도 같습니다. CLI가 아직 로드 중일 때나 결과를
출력한 뒤 프로세스가 끝나는 중에 받은 SIGTERM도 `143`으로, 로드 중의 Ctrl-C는 `130`으로 끝납니다(상태
문서는 로드 중에만 출력. `nohup`처럼 부모가 무시하게 한 SIGINT는 계속 무시함). 단계가 실패한 `run-ingestion` 실행의 결과에도
같은 `error`와 `exit_code`가 담깁니다. `run-ingestion`은 재개할 때 쓸 `pipeline_id`를
(실행이 정한 뒤라면) 덧붙이고, 중지된 실행 다음에 같은 네임스페이스의 다음 실행이 아직 진행
중일 수 있는 쓰기(이 실행의 쓰기, 또는 이 실행이 만료된 임대를 넘겨받은 실행의 쓰기)를
기다려야 하면 `next_run_writes_from`(UTC, ISO 8601)도 덧붙입니다.

---

## 11. Python 라이브러리로 사용하기

CLI는 공개 클래스 두 개를 감싼 래퍼입니다. 이 클래스와 그 밖에 쓰는 이름은 모두 패키지
최상위에서 import하세요. `unified_kg_rag.__all__`의 이름이 공개 API이고, 그 밖의 모듈
경로는 내부 구현이라 바뀔 수 있습니다. `get_config`는 CLI와 같은 설정 파일·환경 변수
단계를 적용합니다([설정](./guides/configure.ko.md#설정을-읽는-순서)).

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- 단계가 실패하면 예외를 던지지 않고 반환한 `PipelineContext`의 `status`를 `FAILED`로
  설정하므로 위 예시처럼 확인하세요.
- 구분해 처리할 오류는 패키지 최상위에서 내보냅니다. `ConfigurationError`(고칠 설정.
  `StoreRequestRejectedError` 포함), `InvalidFilterError`, `IndexNotFoundError`,
  `StoreConnectionError`, `NamespaceLeaseError`, `PipelineExecutionError`,
  `PipelineInterrupted`(`KeyboardInterrupt`)이며, 마지막을 빼면 모두
  `GraphRAGException`입니다. `DataIngestionPipeline.run`은 `ConfigurationError`,
  `StoreConnectionError`, `NamespaceLeaseError`는 그대로 올리고, 단계 밖의 다른
  오류는 `PipelineExecutionError`로 감쌉니다. 모델 클라이언트 자체의 전송 오류(로컬 모델 서버의
  `httpx.ConnectError` 등)는 그대로 전달됩니다. `DataIngestionPipeline(...)`은 가져올 수
  없는 시드 그래프에 `SeedGraphError`(`ConfigurationError`의 하위 클래스)를 던집니다.
- `DataIngestionPipeline(..., confirm=callable)`은 실행이 묻는 질문에 답합니다
  (`callable(question) -> bool`). 지금은 다른 소스 범위의 시드 그래프 문서를 지우게 되는
  재설정만 묻습니다. 넘기지 않으면 그런 실행은 실패합니다
  ([시드 그래프](#d-시드-그래프가지고-있는-지식-그래프-가져오기)).
- `RAGInput`은 `run-rag` 옵션을 필드로 받습니다(`search_strategy`, `search_type`,
  `top_k`, `suffix`([인덱스 접미사](./concepts/scopes-and-namespaces.ko.md#인덱스-접미사)), `filters`, `conversation_id`,
  `use_memory`, `target_language`). 일반 딕셔너리도 됩니다.
- `chain.search(...)`(`await chain.asearch(...)`)는 답변 없이 검색만 해서
  `SearchOutput`(`search_results`, `processed_query`, `metadata`)을 반환합니다.
- 비동기 코드에서는 `await chain.ainvoke(...)`와 `await chain.aclose()`를 쓰고, 시작할
  때 이벤트 루프의 실행기 크기를 한 번 지정하세요:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- 만든 객체는 항상 `close()` / `aclose()`로 닫으세요. 두 클래스 모두 처음 쓸 때 연결을
  열고, 닫지 않으면 가비지 컬렉션 때까지 유지합니다. 체인을 닫을 때 다른 스레드에서
  실행 중인 동기 호출은 `StoreConnectionError`를 일으키고, 그 뒤의 호출은 다시 연결합니다.
- 패키지를 import해도 로깅 핸들러는 설정하지 않습니다. CLI와 같은 로깅은
  `unified_kg_rag.setup_logging(config)`로 켭니다. 호출하지 않으면 구조화 필드
  (`structlog.contextvars`로 바인딩한 값 포함)가 레코드 속성으로 호스트 핸들러에
  전달되며, `LogRecord` 속성과 이름이 같은 필드(`name`, `msg` 등)에는 끝에 `_`가
  붙습니다. `setup_logging`은 프로세스 전체 상태를 바꾸므로 패키지가 로깅을 맡는
  곳에서만 호출하세요. structlog 설정을 교체하고, 루트 로거의 레벨을
  `logging.level`로, `logging.library_levels`에 있는 로거의 레벨을 각 값으로 정하고,
  루트 로거에 stdout 핸들러(`logging.log_to_file`이면 날짜가 붙은 파일 핸들러도)를
  추가합니다. 루트 로거에 이미 있던 핸들러는 그대로 남아 모든 레코드를 함께 받습니다.
  그중 하나가 콘솔에 쓰면 같은 줄이 두 번 나오며, 패키지 레코드에 대해서는 구조화 필드
  없이 메시지와 예외만 받습니다. 다시 호출하면 이전 호출이 추가한 핸들러만 교체합니다.

확장 코드도 패키지 최상위에서 import합니다. 포트(`LLMFactoryPort`, `GraphIndexer`,
`DocStatusPort` 등), `Providers`, `register_provider`, `BaseSearchStrategy`와
`register_strategy`, `ParserFactory`, `register_renderer`가 여기에 있습니다. 두 생성자
모두 백엔드를 `providers=Providers(config, ...)` 번들이나 개별 인자(`doc_status=`,
`vector_indexer=`, `model_factory=` 등) 중 하나로 주입받으며, 둘을 함께 쓸 수는
없습니다. [설계 문서 §15](./design.md#15-extension-guide)를 참고하세요.
`vector_indexer=` / `graph_indexer=`로 넘긴 인덱서는 호출한 쪽이 닫습니다. 파이프라인은
닫지 않습니다.

---

## 12. 제한 사항

- **추출 단계에서 환각이 생길 수 있습니다.** 원문이 뒷받침하지 않는 엔티티나 관계가
  만들어져 검색까지 이어질 수 있습니다.
  `processing.graph_extraction.entity_grounding.enabled`(기본은 꺼짐)는 인용한
  `source_text`가 청크에 없는 항목을 버리거나 가중치를 낮춥니다.
- **실행 결과를 재현할 수 없습니다.** LLM 출력(추출, 리포트, `auto` 라우터, 답변)은
  달라지므로 설정은 여러 번 실행해 비교하세요. 모델 응답과 소스 범위가 같으면 나머지
  처리는 어느 프로세스에서든(`PYTHONHASHSEED`와 관계없이) 결정적입니다. 문서 상태
  레지스트리를 켜면 텍스트 단위 ID가 소스 범위에서 나오며, 소스 범위의 기본값은 소스
  디렉터리의 절대 경로이므로 같은 코퍼스도 다른 디렉터리에서 수집하면 ID가 달라집니다.
  코퍼스 위치와 관계없는 ID가 필요하면 `processing.document_parsing.source_scope`를
  설정하세요.
- **재순위화는 리전에 따라 다릅니다**([필수 모델](./getting-started/aws.ko.md#필수-모델)). 없으면 융합 순서를 쓰며,
  `metadata.rerank`에 나타납니다([재순위화 상태](./guides/query.ko.md#재순위화-상태)).
- **기본 제공 백엔드.** 모델: Bedrock(기본값), Ollama, OpenAI 호환 서버. 재순위화:
  Bedrock이나 sentence-transformers 크로스 인코더([로컬 모드](./guides/local-mode.md), 영문).
  저장소: Neptune(또는 Gremlin Server)과 OpenSearch, 또는 내장 SQLite 저장소(호스트 한
  대, 인덱스당 문서 약 100,000개). 그 밖의 것은 `register_provider` 빌더나 사용자 정의
  어댑터가 필요합니다([설계 문서 §15](./design.md#15-extension-guide)).
- **서빙 계층이 없습니다.** HTTP API, 최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**
  ([범위와 네임스페이스](./concepts/scopes-and-namespaces.ko.md#접근-제어-경계가-아님)).
  테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.** 커뮤니티 탐지를 끄면
  (`graph.community_detection.enabled: false`) `global`과 `drift`가 쓸 리포트가 없고,
  관계 벡터 인덱스를 끄면(`indexing.opensearch.build_relationship_vector_index: false`)
  `mix`와 `hybrid`가 고수준 키워드를, `local`이 관계 섹션을 쓰지 못하고, 문서 필터가 걸린 그래프
  확장은 엔티티에 닿는 관계가 아니라 엔티티 인덱스 확인만으로 엔티티를 남깁니다.
- **증분 인덱싱은 근사치입니다**([재구축 신호](./guides/incremental.ko.md#재구축-신호)). 증분 실행은 다시 군집화하지 않고 커뮤니티를
  덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와 리포트에는 이전
  id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)으로 갱신됩니다. DynamoDB에서는
  문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다([*문서 크기 한도*](./guides/incremental.ko.md#문서-크기-한도)).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리의 쓰기 임대가 한
  네임스페이스의 두 번째 실행을 멈춥니다. 레지스트리가 없거나 `IncrementalIndexer`
  API를 직접 쓰면 잠그는 장치가 없어, 같은 접미사의 수집 두 개가 서로의 인덱스를 지울
  수 있습니다([동시 실행](./operations.md#concurrent-runs)).
- **태스크를 죽이는 문서를 가려내려면 여러 번 실행해야 합니다**([*격리된 문서*](./guides/incremental.ko.md#격리된-문서)). 함께
  죽은 문서 `n`개 가운데 원인 문서 하나에 대략 `log2(n) + 1`번이 걸리고, 그동안 미룬
  문서는 기다립니다.
- **Markdown과 HTML**에는 `unstructured` extra가 필요하며, 컨테이너
  이미지에는 기본으로 빠져 있습니다.
- **시드 그래프는 이름으로만 이어집니다**([*우선순위*](#seed-precedence),
  [§9 D](#d-시드-그래프가지고-있는-지식-그래프-가져오기)). 시드 노드와 이름이 같은 관계없는
  문서 엔티티는 한 엔티티가 됩니다. IRI는 저장하고 필터로 쓸 수 있지만 정체성이 아니며,
  `owl:sameAs`는 따라가지 않고 아무것도 추론하지 않습니다.
- **임베딩 모델(또는 차원)을 바꾸면** `indexing.reset: true`가 필요합니다. 증분 실행은
  다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](./guides/incremental.ko.md#작업-흐름)).

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.md)를 참고하세요.*

## 옮긴 절

이 가이드의 1절부터 5절까지와 8절, 9절, 10절 일부는 별도 문서로 옮겼습니다. 아래 제목은 이전 버전의 `--help`
출력을 비롯한 기존 링크가 계속 동작하도록 남겨 둔 것입니다.

### 1. 사전 요구사항과 설치

[AWS에서 시작하기](./getting-started/aws.ko.md)와
[로컬에서 시작하기](./getting-started/local.ko.md)로 옮겼습니다.

### 런타임

[AWS에서 시작하기: 런타임](./getting-started/aws.ko.md#런타임)으로 옮겼습니다.

### AWS 서비스

[AWS에서 시작하기: AWS 서비스](./getting-started/aws.ko.md#aws-서비스)로 옮겼습니다.

### 필수 모델

[AWS에서 시작하기: 필수 모델](./getting-started/aws.ko.md#필수-모델)로 옮겼습니다.

### 설치

[AWS에서 시작하기: 설치](./getting-started/aws.ko.md#설치)로 옮겼습니다.

### 인증

[AWS에서 시작하기: 인증](./getting-started/aws.ko.md#인증)으로 옮겼습니다.

### IAM 권한

[AWS에서 시작하기: IAM 권한](./getting-started/aws.ko.md#iam-권한)으로 옮겼습니다.

### 로컬 모드 (AWS 없이)

[로컬에서 시작하기](./getting-started/local.ko.md)(첫 실행)와
[Local mode](./guides/local-mode.md)(영문. 공급자, AWS 모드와 다른 점, 컨테이너)로 옮겼습니다.

### AWS 모드와 다른 점

[Local mode: Differences from AWS mode](./guides/local-mode.md#differences-from-aws-mode)(영문)로 옮겼습니다.

### 공급자 플래그

[Local mode: Provider flags](./guides/local-mode.md#provider-flags)(영문)로 옮겼습니다.

### 컨테이너로 띄운 로컬 저장소

[Local mode: Local stores in containers](./guides/local-mode.md#local-stores-in-containers)(영문)로 옮겼습니다.

### 로컬 저장소와 Bedrock 모델

[Local mode: Local stores with Bedrock models](./guides/local-mode.md#local-stores-with-bedrock-models)(영문)로 옮겼습니다.

### 2. 설정

[설정](./guides/configure.ko.md)으로 옮겼습니다.

### 설정을 읽는 순서

[설정: 설정을 읽는 순서](./guides/configure.ko.md#설정을-읽는-순서)로 옮겼습니다.

### 대부분의 실행에서 바꾸는 설정

[설정: 대부분의 실행에서 바꾸는 설정](./guides/configure.ko.md#대부분의-실행에서-바꾸는-설정)으로 옮겼습니다.

### 모델 선택 참고 사항

[Model Catalog: Model selection notes](./reference/models.md#model-selection-notes)(영문)로 옮겼습니다.

### 가드레일

[Guardrail](./guides/guardrail.md)(영문)로 옮겼습니다.

### S3 캐시 암호화

[설정: S3 캐시 암호화](./guides/configure.ko.md#s3-캐시-암호화)로 옮겼습니다.

### 로그에 남는 내용

[설정: 로그에 남는 내용](./guides/configure.ko.md#로그에-남는-내용)으로 옮겼습니다.

### 사용자 정의 프롬프트

[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로 옮겼습니다.

### 환경 변수로 덮어쓰기

[설정: 환경 변수로 덮어쓰기](./guides/configure.ko.md#환경-변수로-덮어쓰기)로 옮겼습니다.

### 관측성

[Observability](./guides/observability.md)(영문. 설정과 절차)와
[Telemetry reference](./reference/telemetry.md)(영문. 트레이스 구조, 사용량 집계)로 옮겼습니다.

### 3. 수집 (`run-ingestion`)

[수집](./guides/ingest.ko.md)으로 옮겼습니다.

### 파이프라인 단계 12개

[수집: 파이프라인 단계 12개](./guides/ingest.ko.md#파이프라인-단계-12개)로 옮겼습니다.

### 인덱스 접미사

[범위와 네임스페이스: 인덱스 접미사](./concepts/scopes-and-namespaces.ko.md#인덱스-접미사)로 옮겼습니다.

### 다국어 수집

[다국어 코퍼스](./guides/multilingual.ko.md)로 옮겼습니다.

### 4. 질의 (`run-rag`)

[질의](./guides/query.ko.md)와 [검색](./concepts/retrieval.ko.md)으로 옮겼습니다.

### 검색 전략 선택

[검색: 검색 전략 선택](./concepts/retrieval.ko.md#검색-전략-선택)으로 옮겼습니다.

### 답변 형식

[검색: 답변 형식](./concepts/retrieval.ko.md#답변-형식)으로 옮겼습니다.

### 컨텍스트 예산

[검색: 컨텍스트 예산](./concepts/retrieval.ko.md#컨텍스트-예산)으로 옮겼습니다.

### 재순위화 상태

[질의: 재순위화 상태](./guides/query.ko.md#재순위화-상태)로 옮겼습니다.

### 질의 제한 시간

[질의: 질의 제한 시간](./guides/query.ko.md#질의-제한-시간)으로 옮겼습니다.

### 필터

[질의: 필터](./guides/query.ko.md#필터)(사용 방법)와
[필터 레퍼런스](./reference/filters.md)(영문. 모든 필드, 값의 유형, 전략별 적용)로 옮겼습니다.

### 대화형 모드와 대화 메모리

[질의: 대화형 모드와 대화 메모리](./guides/query.ko.md#대화형-모드와-대화-메모리)로 옮겼습니다.

### 5. 증분 인덱싱

[증분 인덱싱](./guides/incremental.ko.md)으로 옮겼습니다.

### 켜기

[증분 인덱싱: 켜기](./guides/incremental.ko.md#켜기)로 옮겼습니다.

### 작업 흐름

[증분 인덱싱: 작업 흐름](./guides/incremental.ko.md#작업-흐름)으로 옮겼습니다(재구축은
[재구축](./guides/incremental.ko.md#재구축-indexingreset)과
[재구축 신호](./guides/incremental.ko.md#재구축-신호) 참고).

### 삭제 범위

[증분 인덱싱: 삭제 범위](./guides/incremental.ko.md#삭제-범위)로 옮겼습니다.

### 실패한 문서

[증분 인덱싱: 실패한 문서](./guides/incremental.ko.md#실패한-문서)로 옮겼습니다.

### 격리된 문서

[증분 인덱싱: 격리된 문서](./guides/incremental.ko.md#격리된-문서)로 옮겼습니다.

### 동시 실행

[증분 인덱싱: 동시 실행](./guides/incremental.ko.md#동시-실행)으로 옮겼습니다.

### 문서 크기 한도

[증분 인덱싱: 문서 크기 한도](./guides/incremental.ko.md#문서-크기-한도)로 옮겼습니다.

### 실행 간 병합

[증분 인덱싱: 실행 간 병합](./guides/incremental.ko.md#실행-간-병합)으로 옮겼습니다.

### 대규모, 다국어, 이질적인 코퍼스

[수집: 대규모, 이질적인 코퍼스](./guides/ingest.ko.md#대규모-이질적인-코퍼스)로 옮겼습니다.

### 비용 참고

[수집: 비용 조절 수단](./guides/ingest.ko.md#비용-조절-수단)으로 옮겼습니다.

### 8. 프롬프트 튜닝 (`run-prompt-tuning`)

[도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)으로 옮겼습니다.

### A. `entity_types`(비용이 가장 적고 효과가 가장 큼)

[도메인 적응: 엔티티 유형](./guides/domain-adaptation.ko.md#엔티티-유형)으로 옮겼습니다.

### B. `custom_prompts` 재정의

[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로
옮겼습니다(재정의 키가 없는 프롬프트 다섯 개:
[재정의할 수 없는 프롬프트](./guides/domain-adaptation.ko.md#재정의할-수-없는-프롬프트)).

### 9. 도메인 적응

[도메인 적응](./guides/domain-adaptation.ko.md)(엔티티 유형, 프롬프트, 프롬프트 튜닝),
[Extraction schema](./guides/extraction-schema.md)(영문),
[Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)로
옮겼습니다.

### C. 추출 스키마

[Extraction schema](./guides/extraction-schema.md)(영문)로 옮겼습니다.

### OWL/RDFS 온톨로지에서 읽기

[Ontology](./guides/ontology.md)(영문: 설치, 설정, 미리 보기, 매핑 고정)와
[Ontology mapping 레퍼런스](./reference/ontology-mapping.md)(영문: 옵션, 읽는 것, 매핑 규칙,
예제)로 옮겼습니다.

- <a id="ontology-load-cost"></a>읽기 비용: [Ontology: Load cost](./guides/ontology.md#load-cost)(영문).
- <a id="ontology-what-a-change-re-runs"></a>바꾸면 다시 실행되는 것: [Ontology: What a change re-runs](./guides/ontology.md#what-a-change-re-runs)(영문).
- <a id="ontology-preview"></a>미리 보기: [Ontology: Preview](./guides/ontology.md#preview)(영문).

### D. 시드 그래프(가지고 있는 지식 그래프 가져오기)

[Seed graphs](./guides/seed-graphs.md)(영문)와
[Seed graph formats 레퍼런스](./reference/seed-graph-formats.md)(영문: JSONL, CSV, RDF)로
옮겼습니다.

- <a id="seed-precedence"></a>우선순위: [Seed graphs: Precedence](./guides/seed-graphs.md#precedence-documents-decide)(영문).
- <a id="seed-lifecycle"></a>수명 주기: [Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle)(영문).
- <a id="seed-cost-and-limits"></a>비용과 한계: [Seed graphs: Cost and limits](./guides/seed-graphs.md#cost-and-limits)(영문).
