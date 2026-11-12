# 평가 (`run-eval`)

> 🇬🇧 English version: [evaluate.md](./evaluate.md)

`run-eval --data-path <dataset.json>`은 데이터셋의 모든 질문에 답한 뒤 답변과 출처를
채점합니다. 평가기 셋은 LLM 없이 비용 없이 동작하고(기본값), 둘은 LLM 심사 모델을 씁니다.
전략을 여럿 주거나 `--repeats`를 주면 짝지은 비교가 됩니다.

모든 플래그와 기본값은 [CLI 레퍼런스](../reference/cli.md#run-eval)(영문)에 있습니다.
결과 파일의 필드별 내용과 평가기마다의 채점 방식은
[Evaluation output](../reference/evaluation-output.md)(영문), 공개 벤치마크를 코퍼스와
데이터셋으로 바꾸는 방법은 [Benchmarks](./benchmarks.md)(영문)에 있습니다.

## 평가기

`evaluation.enabled_evaluators`로 고르며(한 번의 실행에서는 `run-eval --evaluators
a,b`), LLM이 필요 없는 세 개가 기본으로 켜져 있고 데이터셋 필드가 없는 질의는 건너뜁니다.
심사 평가기는 선택 사항입니다. `ragas`는 지표 다섯 개를 위해 답변마다 심사 호출을 여러 번 하고
(`context_precision`은 검색된 컨텍스트마다 한 번, 최대 `evaluation.ragas_max_contexts`인 20번)
`eval` extra가 필요하며, `assertion`은 데이터셋에 단언문이 있어야 합니다. `ragas`를 더하려면
`run-eval --evaluators answer_match,retrieval,graph_aware,ragas`로 실행하거나
`enabled_evaluators`에 넣으세요. `langchain` 평가기는 삭제했습니다(`correctness`가
`ragas`의 `answer_correctness`와 같은 것을 쟀습니다). 아직 이를 나열하거나
`evaluation.langchain_metrics`를 설정한 설정 파일은 경고와 함께 그 항목 없이 로드됩니다.

| 평가기 | LLM | 데이터셋에 필요한 필드 | 주요 지표 |
|---|---|---|---|
| `answer_match` | 없음 | `answer`(+ `answer_aliases`) | `answer_contains`(대표 지표), `exact_match`, `token_f1` |
| `retrieval` | 없음 | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr`(k = `evaluation.retrieval_k`, 기본값 5) |
| `graph_aware` | 없음 | `expected_entities` / `expected_relationships` | 답변과 검색된 출처에서의 엔티티·관계 커버리지(재현율) |
| `ragas` | 심사 | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |
| `assertion` | 심사 | `assertions` | `assertion_recall`, `assertion_contradiction_rate`, `negative_assertion_violation_rate`(선택 사항) |

`assertion`도 선택 사항입니다(`enabled_evaluators`에 추가). 답변마다 항목의 단언문(assertion)을
확인하며, 답변당 심사 호출을 `evaluation.assertion_trials`번(기본값 3) 하고 extra는 필요 없습니다.
자세한 내용은 [단언문, 길이 통제, 클로즈드북 기준선](#단언문-길이-통제-클로즈드북-기준선)에 있습니다.
`ragas` 심사 평가기는 `eval` extra가 필요합니다(`uv sync --extra eval`, [설치](../getting-started/aws.ko.md#설치)). extra가
없으면 `run-eval`은 첫 질의 전에 extra 이름을 알려 주고 종료 코드 `2`로 종료하며,
`processing.ignore_errors: true`여도 같습니다. 심사 모델 없는 기본 실행에는 extra가
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
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] },
    "assertions": [
      "The documents discuss machine learning.",
      { "text": "The documents describe quantum hardware.", "kind": "negative" }
    ]
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- `assertions`(`assertion` 평가기용): 좋은 답변이 뒷받침해야 하는 문장(문자열 또는
  `{"text", "kind": "positive"}`)이나 말하면 안 되는 문장(`"kind": "negative"`, 예: 이미 알려진
  환각)이며, 검토자를 위한 `source`를 붙일 수 있습니다. 답변과 같은 언어로 쓰세요.
- 항목별 `metadata`(예: `search_strategy`)는 그 질문에 한해 CLI 기본값보다 우선합니다.
  `id`는 `query_id`로 써도 됩니다.
- `metadata.answer_type`은 `answer_match`가 구조가 있는 정답을 읽는 방식을 정합니다.
  `number`는 답변 첫 줄 어디에든 정답 수(숫자, 수 단어, 단위 앞의 고유어·한자 수사: `세 곳`,
  `三社`. 목록 번호, 인용 표시, 서수는 수가 아님)가 있으면 맞고, 정답이 0이면 '없음'을 말하면서
  질문에 없는 수를 쓰지 않아야 맞습니다(어느 수가 개수인지는 LLM 심사가 판단). `set`은 `metadata.answer_items`(문자열 목록)의
  항목이 순서나 형식과 관계없이 모두 답변에 있어야 맞습니다.
  `metadata.answer_candidates`(답변이 고를 수 있는 이름)가 있으면 답하는 부분(추출한 스팬,
  없으면 첫 줄과 바로 뒤의 목록)에서 그 밖의 후보를 말하지 않아야 맞습니다. 뒤에 나오는
  `Delta is fourth` 같은 문장은 고른 후보가 아닙니다. `metadata.decline_is_correct: true`는
  데이터에 없는 대상을 묻는 질문에서 답변 거절을 정답으로 채점합니다. `prepare-benchmark
  structural`이 이 필드를 씁니다([Evaluation output: answer_match](../reference/evaluation-output.md#answer_match), 영문).
- 코퍼스로 답할 수 없는 질문에는 `"metadata": {"answerable": false}`를 지정하세요. 어떤
  평가기로도 채점하지 않고 `abstention_statistics`
  ([Run summary](../reference/evaluation-output.md#run-summary), 영문,
  [답할 수 없는 질문과 거절](#답할-수-없는-질문과-거절))에만 셉니다.
- 파일은 질의를 실행하기 전에 검증합니다. 읽을 수 없는 파일, 비어 있거나 배열이 아닌
  데이터셋, `question` 누락, 중복 ID, 타입이 틀린 필드(목록이 아닌 `expected_entities`,
  문자열이나 문자열 목록이 아닌 `answer_aliases`), RAG 체인이 거부하는 `metadata`(알 수
  없는 `search_strategy`, 또는 체인이 거부하는 `filters`: 빈 목록 같은 잘못된 값이나
  항목의 전략이 거부하는 키. 비교 실행이면 비교하는 모든 전략으로 확인)가 있으면
  `Error: invalid evaluation dataset:`과 항목 번호, ID를 출력하고 종료 코드 2로 멈춥니다.

## 답할 수 없는 질문과 거절

모든 질문에 답하는 RAG 시스템은 코퍼스로 답할 수 없는 질문에도 답합니다. 거절하는 능력을
측정하려면 `"answerable": false`로 표시한 질문을 데이터셋에 섞고(`prepare-benchmark
musique --unanswerable N`이 코퍼스로 답할 수 없도록 걸러 낸 질문 집합을 만듭니다:
[Benchmarks](./benchmarks.md#unanswerable-questions), 영문) `search.abstention.mode: mark`로
실행해 모델의 거절을 인식하게 하세요([질의: 답변 거절](./query.ko.md#답변-거절)). 그러면
다음과 같이 채점합니다.

- 답할 수 있는 질문에서 거절하면 오답입니다. `answer_match`는 0점을 주고(답변 구간 추출
  호출 없음) 보고서에 `abstained`를 남깁니다.
- 답할 수 없는 질문은 거절했는지로만 채점합니다.
- `abstention_statistics`에 거절 `precision`(거절 중 답할 수 없는 질문의 비율), `recall`(답할
  수 없는 질문 중 거절한 비율), `f1`, `answerable`·`unanswerable` 비율, 그리고 `answer_match`
  지표마다 `coverage`(답한 비율), `accuracy`(답할 수 있는 질문 전체, 거절은 0),
  `selective_accuracy`(답한 질문만), `overall_accuracy`(맞힌 답이나 맞는 거절, 두 종류
  전체)를 보고합니다. coverage와 accuracy를 함께 읽으세요. 거절을 늘리면 selective
  accuracy는 오르고 coverage는 떨어집니다. 개수는 `completed`(답변 생성이 실패하지 않은
  결과) = `abstained`(거절) + `answered`(거절하지 않음)와 `failed`입니다. 답변 생성이 실패한
  결과는 거절이 아니므로 recall을 낮추고, 답할 수 있는 질문이면 accuracy에서 0점입니다.
  실패가 거절로 보이지 않습니다.
- 비교(`--search-strategy a,b`, `--baseline-results`)는 거절도 짝지어 비교합니다.
  `abstention/recall`(답할 수 없는 질의마다 거절 여부)과 `abstention/precision`(거절 중 답할
  수 없는 질문의 비율. 구간은 질의를 재표본추출해 구합니다)이 `deltas_vs_baseline`에 지표와
  함께 들어가므로 `--fail-if abstention/recall:ci_high<0`으로 뒷받침되는 하락을 게이트로
  걸 수 있습니다.
- 저장된 결과마다 `abstained`, `abstention_reason`, `context_sufficient`,
  `insufficiency_reason`과 `iterative`의 실행 기록(라운드 수, 멈춘 이유, 판정)이 남으므로 결과
  파일만으로 실행을 분석할 수 있습니다
  ([Evaluation output: Result files](../reference/evaluation-output.md#result-files)(영문)).
- `mode: off`이면 빈 컨텍스트에 대한 고정 문구만 거절로 세므로 recall은 하한값입니다.
  `mark`는 답변 프롬프트를 바꾸므로 `mark` 실행과 `off` 실행은 두 비교군으로
  비교하세요(`off`로 실행한 뒤, 그 결과 파일을 `--baseline-results`로 지정해 `mark`로
  실행).

## 예시

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

두 경로 모두 `s3://` URI를 받으므로 `run-eval`도 CDK 스택에서 일회성 태스크로
실행할 수 있습니다([After deploy](../../iac/README.md#5-evaluate-from-inside-the-vpc), 영문).
결과는 실행이 끝날 때 `aws.s3.encryption` 설정으로 올리며, 업로드가 실패하면 0이 아닌
코드로 끝나고 로컬 파일을 남깁니다(경로를 출력). S3에는 설정의 `aws.profile_name` /
`aws.region_name`으로 접속하며, 셸의 `AWS_PROFILE` / `AWS_REGION`은 설정이 비워 둔 경우에만
그 값을 채웁니다. 실행 주체에는 데이터셋에 대한
`s3:GetObject`와 접두사에 대한 `s3:PutObject` 권한이 필요합니다.

`run-eval`은 출력 디렉터리 아래 자기만의 하위 디렉터리 `run_<timestamp>_<id>`에 쓰며
(`evaluation.run_subdirectories`), 그 안에 실행마다
`evaluation_{results,reports,summary}_<strategy>_<timestamp>.json`을 씁니다(질의가 여러 전략을
썼으면 `<strategy>`가 빠짐). 전략이 다른 파일은 이름순이
시간순과 다르므로 최신 실행은 수정 시각(`ls -t`)으로 찾으세요. 요약에는 지표별 통계,
채점/실패/건너뜀 개수, 답변 거절, 재순위화와 가드레일 집계, 그리고 무엇을 측정했는지 기록해 두 실행을 비교할 수 있게 하는 실행
매니페스트가 있습니다([Evaluation output](../reference/evaluation-output.md), 영문).
콘솔에도 같은 요약이 출력되며, 실패 허용 기준(`--max-failure-rate`)은 모든 실행에
적용됩니다. 이 기준은 실패한 답변뿐 아니라 품질이 떨어진 채 나간 답변도 셉니다. 재순위화
호출이 실패한 질의(재순위화 모델이 스로틀링되면 융합 순서를 그대로 씀, 재순위화를 시도한 질의
기준)와 검색 단계의 실패를 전략이 흡수한 답변(`degraded_queries`, 단계별로는 `degraded_step_counts`.
`processing.ignore_errors`에서 실패한 질의 번역이나 엔터티·키워드 추출, auto 라우터와 후속 질문 재작성도
포함)입니다. 그래서 기본값
1.0에서는 모든 재순위화 호출이 실패한 실행이 종료 코드 1로 끝나고, `--max-failure-rate 0.05`는
5%를 넘게 실패한 실행을 실패로 처리합니다. 비교 결과는 실행마다 재순위화 실패와 품질이 떨어진
답변의 개수와 비율을 알리고, 비교한 실행 사이에 그 비율이 우연 이상으로 다르면(가장 낮은 비율과
가장 높은 비율의 양측 Fisher 정확 검정(Fisher's exact test) p가 0.05 미만) 차이에 실패의 영향도
섞이므로 더 강하게(`NOT COMPARABLE AS IS`) 경고합니다. 우연 범위 안의 차이(예: 1/40과 0/40)에는
p값과 함께 안내만 남깁니다. 재순위화가 꺼져 있거나 만들 수 없는 경우
(`disabled`, `unavailable`)는 실패한 호출이 아닙니다.

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
  표에는 차이마다 `pairs`와 `unpaired` 질의 수가 나옵니다. 판정 모델이 채점하지 못한 질의는
  짝을 이루지 못하며, 전략마다 그 수가 다르면 경고합니다.
- **차이를 믿을 수 있는 경우.** 지표 하나를 미리 주 지표(primary metric)로 정하세요
  (`--primary-metric`, `evaluation.primary_metric`. 답변을 채점하면 기본값은
  `answer_match/answer_contains`). 주 지표의 차이는 비교한 전략 수만큼만 보정합니다. 나머지
  지표는 보조 지표로 모두 묶어 Holm 방법으로 보정하고(비용 측정값은 비용끼리), `adjusted_p_value`가
  0.05보다 작으면 여러 차이를 함께 읽어도 남는 차이입니다. 표에는 `Holm p`로 나옵니다(묶음마다
  크기에 맞춰 몬테카를로 재표본을 최소 20 x 검정 수 / 0.05개 뽑으므로 전략과 비교군이 많아도
  기각할 수 있습니다. 기본 재표본 수는 `evaluation.resamples`, 2,000). 차이의
  크기는 `ci_low`/`ci_high`(보정 전)나 `adjusted_ci_low`/`adjusted_ci_high`(묶음 전체에 대한
  동시 구간)로 읽으세요. 지표 열 개를 보정 없이 읽으면 같은 설정끼리 비교해도 30-40%의 비교에서
  우연히 차이가 나타납니다. 이진 지표를 짝지어 비교하면 두 전략의 결과가 갈린 질의만 정보가
  되므로, 검정력은 결과가 얼마나 자주 갈리는지에 달려 있습니다. 이 구간으로 시뮬레이션하면
  10%p 차이는 질의의 15%에서 결과가 갈릴 때 질의 50개에서 45%, 200개에서 99%의 실행에서
  찾아지고, 30%에서 갈릴 때는 28%와 73%에서 찾아집니다. 수백 개 질의로 비교하고, 같은 설정을
  두 번 실행해 잡음 수준을 확인하세요. `deltas_by_category`는 차이를 데이터셋 `category`별로 나누며, 범주와
  지표 수에 따른 다중 비교 보정을 하지 않으므로 여러 개 가운데 하나만 0을 벗어난 결과는
  탐색적으로 읽으세요. `--closed-book`을 쓰면 `deltas_by_closed_book`이 같은 방식으로 차이를
  답변 모델이 검색 없이 비교군 반복의 절반 이상에서 맞힌 질문(`known`, 주 지표 기준. token F1
  같은 연속 지표는 평균 0.5 이상, 실패한 답변은 0으로 계산)과 나머지 질문(`unknown`)으로
  나눕니다. `unknown` 쪽이 검색이 더하는 부분을 보여 주며, 전체 차이에서는
  이 부분이 가려질 수 있습니다. 이것도 보정하지 않는 탐색적 결과입니다.
- **맞비교 판정.** `--pairwise-judge`는 GraphRAG / LightRAG 논문의 맞비교 LLM 판정(포괄성,
  다양성, 이해 도움, 직접성)을 더하며, 쌍마다 순서를 바꿔 두 번 판정합니다.
  `candidate_win_rate`는 `position_consistency`, 그리고 답변 길이 구간별 승률(`length`)과 함께
  읽으세요. LLM 판정 모델은 긴 답변을 선호하므로, 길이 비율이 커질수록 승률이 오르면 품질이
  아니라 길이의 효과입니다. 기준마다 승률 0.5에 대한 부호 뒤집기 `p_value`가 있고, 비교한
  모든 전략의 기준이 Holm 묶음 하나(`pairwise`)를 이루므로 승리가 유효한지는
  `adjusted_p_value`(보고서의 `Holm p`)로 판단하세요.
  `--pairwise-mode differential`은 두 답변이 다른 부분만 판정합니다(아래).
  첫 번째 반복의 답변을 판정하며, `--pairwise-repeats all`은 모든 반복을 판정합니다(호출 수는
  `--repeats`배). 이때 질의마다 반복의 점수를 먼저 평균한 뒤 질의끼리 평균합니다.
- **비용과 지연 시간.** `cost.strategies.<s>`와 `cost.deltas_vs_baseline`은 답변의 질의당
  호출, 토큰, 지연 시간을 지표처럼 짝지어 보여 줍니다. 실패한 답변은 실패하기 전까지 쓴
  만큼 셉니다.
- **실행 순서.** 기본값은 번갈아 답하기입니다(`--run-order interleaved`,
  `evaluation.run_order`). 질의 하나를 모든 전략과 반복이 답한 뒤 다음 질의로 넘어가며,
  질의마다 실행 순서를 뒤집고, 반복마다 한 번 더 뒤집습니다(`--repeats`가 짝수이면 질의
  수가 홀수여도 모든 질의의 순서가 균형을 이룹니다). 그래서 실행 도중 느려지는 서비스나 앞선 실행이 데운
  캐시가 지연 시간 차이로 나타나지 않습니다(같은 설정의 실행을 차례로 답하면 질의당 평균
  0.5-2.3초 차이가 났습니다. [기본값과 근거](../concepts/defaults-and-evidence.md#how-the-measurements-were-run)(영문)). `--run-order sequential`은 실행을 하나씩 차례로 답하고,
  도중에 멈추면 끝난 실행을 남기며, 지연 시간 차이가 시간의 영향도 잰다고 경고합니다.
  따로 실행한 결과를 비교할 때(`--from-results`, `--baseline-results`)도 같은 경고가 나옵니다.
- **반복.** 반복마다 질의의 엔티티와 키워드를 다시 추출하므로
  (`--repeat-extractions resample`, `evaluation.repeat_extractions`) `repeat_stdev`에 추출의
  변동도 들어갑니다. 질의와 반복마다 빠른 모델 추출 호출이 한 번씩 듭니다. 한 반복 안의
  전략과 비교군은 추출을 공유하므로 같은 키워드로 검색을 비교합니다. 같은 질문을 동시에
  추출하면 진행 중인 호출을 함께 씁니다(먼저 시작한 쪽이 호출 비용을 내고 나머지는 캐시
  적중으로 셉니다. ABBA 순서라 비용을 내는 쪽이 번갈아 바뀝니다). `reuse`는 모든 반복에서
  한 번만 추출합니다(이전 동작).

모든 필드, 실패와 0점 채우기 규칙, 경고, 비용에 세는 항목은
[Evaluation output: Comparison](../reference/evaluation-output.md#comparison)(영문)에 있습니다.

패키지에는 모델 가격이 들어 있지 않습니다. USD 추정치가 필요하면 사용하는 리전의 현재
[Amazon Bedrock 가격](https://aws.amazon.com/bedrock/pricing/)을 100만 토큰 단위로,
사용량 기록이 쓰는 모델 ID(`usage.by_model`. 팩터리가 추론 프로필을 확정했으면
`global.<model>` 같은 프로필 ID)를 키로 채우세요. 가격이 없는 모델을 쓴 답변은 추정치가
없고 비교가 경고합니다. `input_tokens`에는 프롬프트 캐시 읽기·쓰기 토큰이 들어 있습니다.
그 단가(`cache_read_per_million_tokens`, `cache_write_per_million_tokens`)를 주면 따로
계산하고, 없으면 입력 단가로 계산합니다(Claude에서 캐시 읽기는 입력의 10분의 1이므로 상한값).
재순위화(rerank) 모델도 검색 단위(search unit) 1,000개당 가격으로 넣습니다(사용량
`rerank_search_units`: 문서 100개까지의 재순위화 요청 하나에 1단위. Amazon Bedrock은 질의를
포함해 500토큰이 넘는 문서를 여러 개로 계산하므로 약 4,800자 청크는 3개쯤입니다. 토큰은
문자 체계를 고려한 추정치라 청구서와 1단위쯤 다를 수 있습니다. 로컬 크로스 인코더는 0).

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
    "<rerank-model-id>": {per_thousand_rerank_search_units: 0.0}
```

추정치는 사용량에 기록된 호출만 다룹니다. botocore가 다시 보낸 요청(`request_retries`), 실패한
호출(`failed_calls`), 예열 질의는 가격에 들어가지 않으므로 CloudWatch 호출 수와 청구액이 조금
더 많을 수 있습니다(측정 결과:
[Usage records against the bill](../concepts/defaults-and-evidence.md#usage-records-against-the-bill), 영문).

## 구성 비교군

`--arm NAME=overlay.yaml`은 전략뿐 아니라 구성을 비교합니다. 비교군(arm)은 실행 구성에
오버레이(구성 위에 병합하는 YAML 매핑)를 덮은 구성으로 같은 데이터셋에 답하며, 자체 체인으로
모든 `--search-strategy`를 실행하고 기본 실행과 번갈아 답합니다. 비교군을 늘리려면 플래그를
여러 번 지정하세요. `,suffix=SUFFIX`를 붙이면 그 비교군은 다른 인덱스 네임스페이스(예: 다른
설정으로 수집한 것)를 질의합니다.

```bash
# 재순위화 끄기와 켜기, 그리고 잡음 수준을 재는 A/A 복사본.
printf 'providers:\n  rerank:\n    kind: none\n' > norerank.yaml
: > aa.yaml
run-eval --data-path eval.json --search-strategy local,mix --repeats 2 \
  --arm norerank=norerank.yaml --arm aa=aa.yaml --config-path config.yaml
```

- 비교군의 실행은 파일과 비교 결과에서 `<strategy>@NAME`(`local@norerank`)으로 표시되며, 같은
  전략의 기본 실행과 짝지어 비교하므로 그 차이는 오버레이의 효과입니다. 기본 전략들은 평소처럼
  첫 번째 전략과 비교합니다. 비교군의 차이는 첫 번째 전략과 비교한 전략의 차이와 묻는 질문이
  다르므로 따로 Holm 묶음(`arm_primary`, `arm_secondary`, `arm_cost`, `arm_pairwise`)으로
  보정합니다.
- 오버레이는 구성 파일처럼 검증합니다(`--allow-unknown-keys`가 없으면 알 수 없는 키에서
  종료 코드 2로 멈춤). 오버레이가 지정한 상대 경로(SQLite 저장소, 시드 그래프, 스키마 파일)만
  오버레이 파일의 디렉터리를 기준으로 풀고, 지정하지 않은 경로는 실행 구성의 것을 그대로 쓰므로
  비교군은 실행과 같은 저장소를 읽습니다. 모든 비교군은 기본 실행과 같은 방식으로 채점하므로 오버레이의
  `evaluation` 섹션과 `models.roles.evaluation` / `answer_span`은 경고와 함께 무시합니다.
  LLM 판정 모델을 지정하지 않았으면 기본 실행과 모든 비교군의 답변 모델을 함께 보고(가능하면
  그중 어느 것도 아닌 모델로) 고른 뒤 비교군을 만들기 전에 고정하며, 답변 구간 추출 모델은
  실행의 것을 씁니다.
- 비교한 실행들의 LLM 판정 모델, 판정 추론 강도, ragas 임베딩 모델이 다르면(`models.embedding`을
  바꾼 비교군, 다른 호출의 `--baseline-results`나 `--from-results` 실행) 판정 지표의 차이에
  판정 모델의 차이도 섞인다고 경고합니다.
- 실행 매니페스트는 비교군 실행마다 `arm`(이름, 오버레이 경로와 sha256, suffix)과
  `compared_runs`를 기록하므로 `--from-results`가 같은 짝을 다시 만듭니다. 실행마다 그
  전략의 답변이 인용하는지(`search.citations`를 전략별로 판정한 값, 매니페스트의
  `citations.strategies`)가 다르면 답변 프롬프트도 다르다고 경고합니다. `global`과 `drift`는
  기본값으로 인용하지 않으므로, 맵에서 같게 맞추지 않으면 다른 전략과 비교할 때 경고합니다.
- 오버레이가 질의 추출(질문마다 한 번 호출하는 엔터티와 키워드)을 바꾸지 않으면 비교군은 기본
  실행의 추출 결과로 검색합니다. 추출 모델, 기본 계층의 추론 강도(effort),
  `search.output_ceilings.query_extraction`, `models.default_max_output_tokens`, LLM
  제공자(kind, `base_url`, temperature, 출력 상한, 컨텍스트 창), Bedrock 리전을 바꾸거나 `search.query_extraction_cache_size: 0`을 지정한 비교군은 직접
  추출합니다.
- 병합한 구성이 실행의 구성과 같고(캐시 크기 `search.query_extraction_cache_size`와
  `search.token_manager.token_count_cache_size`는 제외) 실행 자체의 인덱스를 쓰는 비교군은
  A/A 비교군입니다. 빈 오버레이(위의 `aa.yaml`), 무시되는 `evaluation` 섹션만 있는 오버레이,
  실행과 같은 값만 지정한 오버레이가 여기에 해당합니다. 그 차이가 이 데이터셋과 반복 횟수에서
  지표마다의 잡음 수준입니다. `,aa=true`(또는 `aa=false`)로 직접 지정할 수도 있습니다.
  매니페스트에 `arm.aa`로 기록되고 비교 결과의 `arm_roles`(`aa`)에 나열되며, 보고서와 터미널,
  콘솔은 이를 "(noise floor)"로 표시합니다. 폐쇄형 비교군처럼 모든 Holm 묶음과 `--fail-if` 게이트에서
  빠지므로 다른 검정의 보정을 넓히거나 실행을 실패시키지 않습니다. 기본 실행과 추출 결과를 공유하므로 이 잡음 수준에는 추출의 잡음이 빠집니다.
  `search: {query_extraction_cache_size: 0}` 오버레이로 만든 A/A 비교군은 직접 추출해 그 잡음을
  포함하며, 질문과 반복마다 추출 호출이 한 번 더 듭니다.
- 비교군 하나는 전략 하나만큼(답변과 워밍업) 들며, 나열한 전략 수만큼 곱해집니다.

## 단언문, 길이 통제, 클로즈드북 기준선

열린(전역) 질문에는 정답이 없고, 맞비교 판정 모델은 긴 답변을 선호합니다. 이를 보완하는
선택 기능 세 가지가 있으며, 측정 실험으로 결정하기 전까지 어느 것도 기본값을 바꾸지
않습니다([Design: §9.3](../design/evaluation.md#93-length-controlled-judging-assertions-and-the-closed-book-arm), 영문).

1. Microsoft BenchmarkQED 방식으로 사용자의 코퍼스에서 질문과 단언문을 만듭니다(코드는 MIT
   라이선스이며 BenchmarkQED의 데이터는 쓰지 않음).

   ```bash
   prepare-benchmark assertions --source-directory corpus/ \
     --output-directory bench/assertions --questions-per-class 10 \
     --config-path config.yaml
   ```

   다섯 유형(`category`: `data_local`, `data_global`, `data_linked`, `activity_local`,
   `activity_global`)의 질문과 데이터 유형 셋의 단언문을 담은 `eval_data.json`,
   `text_units.jsonl`(단언문마다 인용한 텍스트와 파일, 오프셋. 세트를 믿기 전에 이것으로
   단언문을 검토하세요), `generation_report.json`(시드, 모델, 버린 항목, 사용량)을 씁니다.
   평가 모델(`models.roles.evaluation`)을 데이터 질문마다 3~4번, 그리고 클러스터와 활동
   질문마다 한 번 부르고, 텍스트 단위를 최대 `--max-units`개 임베딩하며, 쓴 호출과 토큰을
   출력합니다(`evaluation.prices`가 있으면 USD 추정치도). 질문과 단언문은 `--language`
   (기본값 `processing.translation.source_language`)로 씁니다. 같은 시드와 같은 모델 응답이면
   같은 파일이 나옵니다.
2. 답변을 단언문으로 채점하고 길이 편향 없이 맞비교합니다.

   ```bash
   run-eval --data-path bench/assertions/eval_data.json \
     --search-strategy global,mix --pairwise-judge --pairwise-mode differential \
     --closed-book --config-path config.yaml   # enabled_evaluators: [assertion, ...]
   ```

   - `assertion_recall`은 항목의 긍정 단언문 가운데 답변이 뒷받침하는 비율(시도의 절반 넘게
     그렇다고 판정한 단언문만 셈), `assertion_contradiction_rate`는 답변이 반박하는 비율,
     `negative_assertion_violation_rate`는 부정 단언문을 답변이 말한 비율입니다. 판정 모델이
     실패하면 지표는 0이 아니라 실패로 남습니다. 다른 지표처럼 게이트를 걸 수 있습니다:
     `--fail-if "assertion/assertion_recall:ci_high<-0.02"`.
   - `--pairwise-mode differential`은 빠른 계층 모델(`models.roles.pairwise_overlap`)에게 두
     답변이 함께 말하는 문장과 한쪽만 말하는 문장을 나누게 한 뒤, 한쪽만 말하는 문장만
     판정합니다. 이 분할은 답변을 정해진 자리 순서로 읽고, 분할 모델은 한쪽 자리를 선호합니다
     (A/A 쌍에서 후보가 늘 두 번째일 때 0.645, 바꾸면 0.52). 그래서 쌍마다 두 자리 순서로 모두
     나누고, 각 분할을 두 순서로 판정해 두 자리의 승률을 평균합니다. `slots`는 자리별 승률과
     그 차이(`slot_effect`, 보고서의 "Split slot effect" 표)를 보여 주며, 이 값은 0에 가까워야
     합니다. 쌍마다 빠른 모델 호출 두 번과 판정 호출 네 번이 듭니다. 이 호출의 출력 상한은 실행하는
     추론 강도(`evaluation.judge_effort`, 없으면 빠른 계층의 강도)의 사고 예산에
     `models.default_max_output_tokens`를 더한 값이므로, 추론 강도를 높여도 옮겨 적는 문장의
     여유는 줄지 않습니다.
   - `--closed-book`은 `closed_book` 비교군을 더합니다. 답변 모델이 검색 없이 모든 질문에
     답하므로, 이를 이기지 못하는 전략은 모델이 이미 아는 것을 재고 있는 셈입니다. 비교
     결과에는 "(no retrieval)"로 표시하고 게이트에서는 제외하며, 검색·컨텍스트 지표는 빈
     출처 목록으로 채점됩니다.

## 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트

`run-eval --from-results <dir|files>`는 이전 실행이 저장한 파일(`evaluation_results_*`와 그 옆의
`evaluation_reports_*`, `evaluation_summary_*`. 기본값인 `evaluation.save_detailed_results:
true`일 때 저장됨. 디렉터리는 그 실행 하위 디렉터리까지 읽음)로 질의에 답하거나 저장소를 열지 않고 비교를 다시 만듭니다. 실행을 전략별로
묶어 가장 오래된 실행이 나열한 순서로 비교하므로(`--search-strategy`를 주면 그 전략을 그 순서로
비교), 중간에 멈춘 비교는 빠진 전략만 같은 디렉터리에 같은 `--repeats`로 실행한 뒤 다시 만들면
끝납니다. 이렇게 만든 차이는 중단 없이 실행한 비교와 같습니다. 단, 이렇게 하려면 실행이
끝날 때마다 저장하는 `--run-order sequential`로 실행해야 합니다. 번갈아 답하는 비교는 모든
실행이 함께 끝나므로 도중에 멈추면 저장된 실행이 없습니다. 실행이 모두 사라진 전략은
`--repeats N`으로 다시 실행하고, 반복 일부가 저장된 전략은 빠진 반복만 실행하거나(남은 횟수를
`--repeats`로 실행하고 `--pool-runs`로 다시 만들기. 아래 참고) 그 전략의 일부 파일을 지운 뒤
N번 모두 실행하세요.

```bash
# naive 실행 중에 멈춤(--run-order sequential --repeats 3으로 실행): naive만 같은
# 디렉터리에 같은 --repeats로 실행하고 두 실행 하위 디렉터리로 다시 만들기.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# 저장한 답변을 다른 판정 모델로 다시 판정(답변 호출 없음).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# 저장한 답변을 이 버전의 평가기(와 새로 만든 데이터셋의 정답)로 다시 채점해
# --output-directory에 새 실행으로 저장.
run-eval --from-results out/ --rescore new_eval.json --output-directory rescored/

# CI: 이전 실행과 비교해 뒷받침되는 성능 저하가 있으면 실패.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- **합치기.** 한 전략의 저장된 실행들은 `run-eval` 한 번이 같은 설정으로 답했을 때만 그
  전략의 반복입니다. 여러 번 실행한 결과(`run_id`가 여러 개. `run_id`를 기록하기 전에 저장한
  실행은 `--repeats`보다 실행이 많을 때)나 설정(질의 플래그, 구성 해시, 코드, 인덱스, 답변
  모델)이 다른 실행이 섞이면 거부합니다(종료 코드 2). 실행 하위 디렉터리 하나를 가리키거나
  결과 파일을 나열하고, 그래도 합치려면 `--pool-runs`를 주세요(무엇이 섞였는지 경고합니다).
- **다시 판정하기.** `--from-results`와 `--pairwise-judge`를 함께 쓰면 저장한 답변을 다시
  판정합니다. `--judge-model`은 그 실행의 판정 모델을 정합니다.
- **다시 채점하기.** `--rescore`는 저장된 점수로 비교하는 대신 저장한 답변을 이 실행의
  평가기(`--evaluators`, `evaluation.answer_span`, 인용 지표, 다시 호출하는 판정 모델)로 다시
  채점합니다. 정답은 답변과 함께 저장된 것을 쓰고, `--rescore DATASET`이면 그 데이터셋의
  정답을 질의 ID로 짝지어 씁니다(질문이 같아야 함). 다시 채점한 실행은 이 실행의 디렉터리에
  저장하며, 매니페스트는 답변한 실행의 내용을 유지하고 다시 채점한 기록(`rescored`)을 더합니다.
- **기준 실행.** `--baseline-results`는 각 전략을 이전 실행의 같은 전략(`baseline:<strategy>`)과
  비교합니다. 기준 실행은 같은 `--repeats`로 만드세요. 실행마다 하위 디렉터리에 따로
  저장되며, 한 전략을 여러 번 실행한 결과가 든 기준 디렉터리는 `--from-results`처럼
  거부합니다(`--pool-runs`).
- **채점 방식이 같아야 함.** exact match와 token F1이 답변을 다르게 읽은 실행
  (`evaluation.answer_span`, `answer_normalization`)은 답변을 생성하기 전에 거부합니다(종료
  코드 2). 데이터셋이 다른 실행은 경고하고 공통 질의 ID만 짝짓습니다.
- **게이트.** `--fail-if <measure>:<stat><op><value>`(여러 번 지정 가능)는 비교한 전략의
  짝지은 차이가 조건을 만족하면 종료 코드 1로 끝냅니다. `ci_high<-0.02`는 질의 표본이
  뒷받침하는 성능 저하일 때만 실패합니다. 그 측정값의 차이가 없는 전략도 실패하고, 짝지은
  질의가 10개 미만인 차이(짝 하나면 구간의 폭이 0입니다. 30개 안팎보다 적으면 부트스트랩 구간이
  너무 좁아 잡음만으로 0을 벗어나는 비율이 5%가 아니라 최대 8-12%입니다)와 판정 모델이 두 쪽 답변에서 채점하지
  못한 횟수가 다른 지표(`strategies.<s>.failed_metrics`. 그 질의는 짝을 이루지 못합니다)도
  실패합니다. 전략 두 개 이상이나 `--baseline-results`가 필요합니다.
- **게이트 개수.** 비교한 전략마다 게이트 하나하나가 별개의 검정입니다. `ci_high<0` 게이트는
  잡음만으로도 실행의 약 2.5%에서 실패하므로, 비교 전략 k개와 그런 게이트 m개면 문제없는
  변경도 최대 k × m × 2.5%의 확률로 실패합니다(전략 3개 × 게이트 4개면 최대 30%). 미리 정한
  측정값 몇 개만, 여유를 두고(`<0`이 아니라 `ci_high<-0.02`) 게이트로 걸거나, 비교한 전략과
  지표 수를 반영한 보정 통계량으로 거세요:
  `--fail-if "answer_match/token_f1:adjusted_ci_high<-0.02"`(`adjusted_ci_low`, `p_value`,
  `adjusted_p_value`도 가능).

전체 규칙(경고마다 비교하는 항목, 비용 게이트가 실패하는 경우)은
[Evaluation output: Rebuilding and gating rules](../reference/evaluation-output.md#rebuilding-and-gating-rules)(영문)에
있습니다.
