# 평가 (`run-eval`)

> 🇬🇧 English version: [evaluate.md](./evaluate.md)

`run-eval --data-path <dataset.json>`은 데이터셋의 모든 질문에 답한 뒤 답변과 출처를
채점합니다. 평가기 셋은 LLM 없이 비용 없이 동작하고, 둘은 LLM 심사 모델을 씁니다.
전략을 여럿 주거나 `--repeats`를 주면 짝지은 비교가 됩니다.

모든 플래그와 기본값은 [CLI 레퍼런스](../reference/cli.md#run-eval)(영문)에 있습니다.
결과 파일의 필드별 내용과 평가기마다의 채점 방식은
[Evaluation output](../reference/evaluation-output.md)(영문), 공개 벤치마크를 코퍼스와
데이터셋으로 바꾸는 방법은 [Benchmarks](./benchmarks.md)(영문)에 있습니다.

## 평가기

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

심사 평가기 둘은 `eval` extra가 필요합니다(`uv sync --extra eval`, [설치](../getting-started/aws.ko.md#설치)). extra가
없으면 `run-eval`은 첫 질의 전에 extra 이름을 알려 주고 상태 코드 `2`로 종료하며,
`processing.ignore_errors: true`여도 같습니다. 위의 심사 모델 없는 실행에는 extra가
필요 없습니다. 그 밖에 켜 둔 평가기를 만들 수 없거나(예: 심사 모델의 Bedrock 접근 권한
없음) 설정을 거부하면 실행이 멈춥니다. `processing.ignore_errors: true`이면 그 평가기를 빼고
`run_manifest.dropped_evaluators`에 기록합니다. 채점은 모든 답변을 생성한 뒤에
시작하므로 평가기 오류가 나도 실행 결과는 버리지 않습니다. 해당 질의의 지표를 실패로
기록하고(쓸 만한 점수가 없는 심사 응답도 0이 아니라 실패), 종료 코드는
`--max-failure-rate`가 정합니다.

평가기마다의 채점 방식(`ragas` 컨텍스트 상한, `graph_aware`의 답변·검색 커버리지,
`retrieval`의 귀속과 파일 이름 매칭, `answer_match`의 구문 매칭, 채점하는 답 구간과
정규화)은 [Evaluation output: Evaluator details](../reference/evaluation-output.md#evaluator-details)(영문)에
있습니다.

## 평가 데이터 형식

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
  평가기로도 채점하지 않고 `abstention_statistics`
  ([Run summary](../reference/evaluation-output.md#run-summary), 영문)에만 셉니다.
- 파일은 질의를 실행하기 전에 검증합니다. 읽을 수 없는 파일, 비어 있거나 배열이 아닌
  데이터셋, `question` 누락, 중복 ID, 타입이 틀린 필드(목록이 아닌 `expected_entities`,
  문자열이나 문자열 목록이 아닌 `answer_aliases`), RAG 체인이 거부하는 `metadata`(알 수
  없는 `search_strategy`, 또는 체인이 거부하는 `filters`: 빈 목록 같은 잘못된 값이나
  항목의 전략이 거부하는 키. 비교 실행이면 비교하는 모든 전략으로 확인)가 있으면
  `Error: invalid evaluation dataset:`과 항목 번호, ID를 출력하고 종료 코드 2로 멈춥니다.

## 예시

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

두 경로 모두 `s3://` URI를 받으므로 `run-eval`도 CDK 스택에서 일회성 태스크로
실행할 수 있습니다([After deploy](../../iac/README.md#5-evaluate-from-inside-the-vpc)).
결과는 실행이 끝날 때 `aws.s3.encryption` 설정으로 올리며, 업로드가 실패하면 0이 아닌
코드로 끝나고 로컬 파일을 남깁니다(경로를 출력). S3에는 설정의 `aws.profile_name` /
`aws.region_name`으로 접속하며, 셸의 `AWS_PROFILE` / `AWS_REGION`은 설정이 비워 둔 경우에만
그 값을 채웁니다. 실행 주체에는 데이터셋에 대한
`s3:GetObject`와 접두사에 대한 `s3:PutObject` 권한이 필요합니다.

실행마다 `evaluation_{results,reports,summary}_<strategy>_<timestamp>.json`을
씁니다(질의가 여러 전략을 썼으면 `<strategy>`가 빠짐). 전략이 다른 파일은 이름순이
시간순과 다르므로 최신 실행은 수정 시각(`ls -t`)으로 찾으세요. 요약에는 지표별 통계,
채점/실패/건너뜀 개수, 답변 거절, 재순위화와 가드레일 집계, 그리고 무엇을 측정했는지 기록해 두 실행을 비교할 수 있게 하는 실행
매니페스트가 있습니다([Evaluation output](../reference/evaluation-output.md), 영문).
콘솔에도 같은 요약이 출력되며, 실패 허용 기준(`--max-failure-rate`)은 모든 실행에
적용됩니다.

## 전략 비교

`run-eval --search-strategy local,mix,global --repeats 3`은 같은 질의를 전략마다
`--repeats`번 답하고 채점하며(지정한 전략이 항목별 `search_strategy`보다 우선. 전략은 플래그 하나에
나열하며, `--search-strategy`를 두 번 주면 사용법 오류), 실행마다
결과 파일을 저장하고 `evaluation_comparison_<timestamp>.json`과 함께 `.csv`(전략, 측정값,
질의마다 한 행)와 `.md`(표, 게이트, 경고. 풀 리퀘스트나 CI 작업 요약용)를 추가로 씁니다.

읽는 방법은 다음과 같습니다.

- **짝지은 차이.** `deltas_vs_baseline.<s>.<metric>`은 각 전략을 처음 나열한 전략과 두
  전략이 모두 채점한 질의에서 짝지어 비교합니다. `mean_delta`, 95% 짝지은 부트스트랩
  구간(`ci_low`/`ci_high`), `wins`/`ties`/`losses`가 있으며, 질의마다 반복 평균을 먼저
  구합니다. 실패한 답변은 상대가 채점한 지표에서 0점이므로 실패로 이득을 볼 수 없습니다.
- **차이를 믿을 수 있는 경우.** 구간이 0을 포함하지 않으면 그 차이가 뒷받침됩니다. 이진
  지표의 10%p 차이는 질의 50개에서 15%, 200개에서 59%의 실행에서만 찾아지므로 수백 개
  질의로 비교하세요. `deltas_by_category`는 차이를 데이터셋 `category`별로 나누며, 범주와
  지표 수에 따른 다중 비교 보정을 하지 않으므로 여러 개 가운데 하나만 0을 벗어난 결과는
  탐색적으로 읽으세요.
- **맞비교 판정.** `--pairwise-judge`는 GraphRAG / LightRAG 논문의 맞비교 LLM 판정(포괄성,
  다양성, 이해 도움, 직접성)을 더하며, 쌍마다 순서를 바꿔 두 번 판정합니다.
  `candidate_win_rate`는 `position_consistency`와 함께 읽으세요.
- **비용과 지연 시간.** `cost.strategies.<s>`와 `cost.deltas_vs_baseline`은 답변의 질의당
  호출, 토큰, 지연 시간을 지표처럼 짝지어 보여 줍니다. 실패한 답변은 실패하기 전까지 쓴
  만큼 셉니다.

모든 필드, 실패와 0점 채우기 규칙, 경고, 비용에 세는 항목은
[Evaluation output: Comparison](../reference/evaluation-output.md#comparison)(영문)에 있습니다.

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
더 많을 수 있습니다(측정 결과:
[Usage records against the bill](../concepts/defaults-and-evidence.md#usage-records-against-the-bill), 영문).

## 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트

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

- **합치기.** 한 전략의 저장된 실행들은 그 전략의 반복으로 합칩니다. 설정(질의 플래그,
  구성 해시, 코드, 인덱스, 답변 모델)이 다른 실행이 섞이면 경고하므로, 그런 실행은 다른
  디렉터리에 두세요.
- **다시 판정하기.** `--from-results`와 `--pairwise-judge`를 함께 쓰면 저장한 답변을 다시
  판정합니다. `--judge-model`은 그 실행의 판정 모델을 정합니다.
- **기준 실행.** `--baseline-results`는 각 전략을 이전 실행의 같은 전략(`baseline:<strategy>`)과
  비교합니다. 기준 실행은 같은 `--repeats`로 만들고, 실행마다 디렉터리를 따로 두세요.
- **채점 방식이 같아야 함.** exact match와 token F1이 답변을 다르게 읽은 실행
  (`evaluation.answer_span`, `answer_normalization`)은 답변을 생성하기 전에 거부합니다(종료
  코드 2). 데이터셋이 다른 실행은 경고하고 공통 질의 ID만 짝짓습니다.
- **게이트.** `--fail-if <measure>:<stat><op><value>`(여러 번 지정 가능)는 비교한 전략의
  짝지은 차이가 조건을 만족하면 종료 코드 1로 끝냅니다. `ci_high<-0.02`는 질의 표본이
  뒷받침하는 성능 저하일 때만 실패합니다. 그 측정값의 차이가 없는 전략도 실패합니다. 전략 두
  개 이상이나 `--baseline-results`가 필요합니다.

전체 규칙(경고마다 비교하는 항목, 비용 게이트가 실패하는 경우)은
[Evaluation output: Rebuilding and gating rules](../reference/evaluation-output.md#rebuilding-and-gating-rules)(영문)에
있습니다.
