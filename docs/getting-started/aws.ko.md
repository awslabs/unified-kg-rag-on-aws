# AWS에서 시작하기

> 🇬🇧 English version: [aws.md](./aws.md)

이 문서는 저장소를 내려받은 상태에서 이미 있는 Amazon Neptune 클러스터와 Amazon
OpenSearch 도메인을 대상으로 첫 수집과 질의를 실행하기까지의 과정을 설명합니다. 이
프레임워크는 **이미 있는** 서비스에 연결할 뿐 서비스를 만들지 않습니다. 스택 전체를
만들려면 번들 CDK 앱 [`iac/`](../../iac/README.md)를 쓰세요
([Quick start](../../iac/README.md#quick-start), 영문). CDK 스택의 저장소는 VPC 안에서만
접근할 수 있으므로 CLI는 엔드포인트를 환경 변수로 받는 태스크로 VPC 안에서
실행합니다([After deploy](../../iac/README.md#after-deploy)). AWS 계정 없이 써 보려면
[로컬에서 시작하기](./local.ko.md)를 참고하세요.

## 사전 요구사항

Python, uv, Bedrock 액세스 권한이 있는 AWS 자격 증명, 그리고 아래 서비스가 필요합니다.

### 런타임

- **Python 3.11 – 3.12**
- **[uv](https://docs.astral.sh/uv/)**(권장 패키지 관리자. `pip`도 사용 가능)

### AWS 서비스

[로컬 모드](../guides/local-mode.md)(영문)에서는 각 서비스를 로컬 구성 요소가 대신합니다.

| 서비스 | 필수 여부 | 용도 |
|---|---|---|
| **Amazon Bedrock** | 필수 | 모든 LLM 호출, 임베딩, 재순위화(설정한 모델 ID의 모델 액세스를 활성화해야 함) |
| **Amazon Neptune** | 필수 | 지식 그래프와 질의 시점의 다중 홉 순회 |
| **Amazon OpenSearch** | 필수 | 벡터 인덱스와 BM25 어휘 인덱스(텍스트 단위, 엔티티, 커뮤니티 리포트, 관계, 주장) |
| **Amazon S3** | `--s3-sync`나 `s3://` 경로를 쓸 때만 | 단계 캐시 동기화, `s3://` 데이터셋, CDK 코퍼스 |
| **Amazon DynamoDB** | 증분 인덱싱을 쓸 때만 | 콘텐츠 해시로 코퍼스의 변경분을 계산하는 문서 상태 레지스트리 |

### 네트워크와 버전

- **CLI 호스트에서 두 저장소에 접근할 수 있어야 합니다.** Neptune 클러스터는 VPC 안에서만
  연결을 받으므로 CLI는 그 VPC 안의 호스트(또는 VPN, 피어링, 터널로 연결된 호스트)에서
  실행하고, 보안 그룹이 Neptune 포트와 OpenSearch 도메인으로 가는 연결을 허용해야 합니다.
  VPC 밖의 호스트는 연결할 수 없습니다.
- **OpenSearch.** CDK 도메인이 쓰는 OpenSearch 2.19와 2.13에서 테스트했습니다. 선택 기능인
  온디스크 벡터 키는 2.17 이상, 기본 `cosinesimil` 공간을 쓰는 faiss는 2.19가 필요합니다
  ([운영 런북: 온디스크 벡터 저장](../operations.md#on-disk-vector-storage), 영문).
- **Neptune.** CDK 앱은 엔진 버전을 고정하지 않습니다(새 클러스터는 Neptune 기본 버전으로
  시작). 클라이언트가 3.8 미만의 `gremlinpython`이므로 엔진도 TinkerPop 3.8 미만이어야
  합니다([`iac/README.md`](../../iac/README.md#configuration-cdkjson-context---c-keyvalue), 영문).

## 필수 모델

처음 실행하기 전에 다음 Bedrock 모델의 액세스를 활성화하세요(코드 기본값이며, 다른
모델을 설정하면 그 행의 모델이 바뀝니다).

| 역할 | 기본 모델 ID | 설정 키 |
|---|---|---|
| 기본 계층(추출, 글리닝, 커뮤니티 리포트, 답변) | `anthropic.claude-sonnet-5-5` | `models.default` |
| `run-eval` LLM 판정 모델(LLM 판정을 실행할 때만) | `anthropic.claude-opus-5-5` | `models.roles.evaluation` |
| 빠른 계층(청킹, 번역, 요약, 라우팅, map 단계) | `anthropic.claude-haiku-5-5` | `models.fast` |
| 임베딩(인덱싱, 질의, 평가) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| 재순위화 | `cohere.rerank-v3-5:0` | `models.rerank` |

- **리전.** 모든 Bedrock 호출은 `aws.bedrock.region_name`(기본값은
  `aws.region_name`인 `us-west-2`이며 다섯 모델이 모두 있음)으로 갑니다. 재순위화 모델이
  없는 리전(예: `ap-northeast-2`)에서는 이 값을 모델이 있는 리전으로 지정하거나
  `providers.rerank.kind: none`으로 설정하세요. Bedrock으로 가는 경로가 VPC
  엔드포인트뿐인 프라이빗 VPC는 자기 리전을 유지해야 합니다.
- **추론 프로파일.** Claude Sonnet 5.5와 Haiku 5.5는 교차 리전 추론 프로파일로만
  호출할 수 있습니다. `aws.bedrock.enable_global_profile: true`를 유지하고
  `bedrock:ListInferenceProfiles` 권한을 부여하세요(프로파일을 찾지 못하면 모델 생성이
  실패하며 오류 메시지에 이 해결 방법이 나옴).
- **권한.** [IAM 권한](#iam-권한)을 참고하세요. 재순위화에는 별도 정책 문이 필요합니다.

[모델 카탈로그](../reference/models.md)(영문)에 선별된 모델 전체와 각 모델의 한도가 있고,
[모델 선택 참고 사항](../reference/models.md#model-selection-notes)(영문)에 두 모델 계층과
역할별 모델을 바꾸는 방법이 있습니다.

## 설치

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync --extra eval --extra viz  # or: pip install -e '.[eval,viz]'
```

`uv sync`만 실행하면 수집과 질의만 설치됩니다. 선택 기능은 extra로 추가하며(이 페이지는
배포에 쓰는 extra를 다룹니다), 필요한 extra가 없으면 해당 명령이 extra 이름을 알려 주고 종료
코드 `2`로 종료합니다. 위 명령의 두 extra는 다음과 같습니다.

- **`eval`**: `run-eval`의 선택 사항인 RAGAS 심사 평가기(ragas, datasets, pandas.
  [평가기](../guides/evaluate.ko.md#평가기)). 기본으로 켜진, LLM을 쓰지 않는 평가기에는
  extra가 필요 없습니다.
- **`viz`**: `run-visualization`의 대화형 그래프용 pyvis([Visualization](../guides/visualize.md), 영문). 정적 차트는 추가 패키지 없이 동작합니다.

**Markdown과 HTML**은 추가 패키지 없이 파싱합니다. Markdown은 쓰인 그대로, HTML은 표준
라이브러리 파서로 읽습니다. `unstructured` extra는 둘을 더 정교하게 파싱하며, 설치되어 있으면 기본
리더 대신 사용됩니다(`uv sync --extra unstructured`. URL 분할의 SSRF와 마크업 미이스케이프 문제를
고친 `unstructured>=0.24.1`을 요구함). **Word**(`.docx`) 파일에는 `docx` extra가 필요하며
(`uv sync --extra docx`), 없으면 수집은 이 파일을 건너뛰고 extra 이름을 담은 경고를 한 번 남깁니다.
[원본 형식](../troubleshooting.md#which-source-formats-are-read)(영문)을 참고하세요.

`unstructured` extra는 spaCy 모델(`en_core_web_sm`)로 토큰을 나누며, 첫 Markdown·HTML 파일을 읽을 때
GitHub에서 이 모델을 내려받습니다. 네트워크에 접근할 수 없는 환경이라면 네트워크가 되는 곳에서
한 번 미리 설치하세요:
`python -c "from unstructured.nlp.tokenize import sent_tokenize; sent_tokenize('Ready.')"`.
설치하지 않으면 첫 파일은 최대 120초를 기다린 뒤 실패하고, 같은 실행의 나머지 파일은 같은
오류로 바로 실패합니다.

배포용 이미지(`docker/Dockerfile`)에는 두 extra 모두 빠져 있습니다(`unstructured`는 약 200MB 증가).
Word 파일을 읽으려면 `--build-arg UV_EXTRAS="--extra docx"`로, `unstructured`를 쓰려면
`"--extra unstructured"`로 빌드하세요. 후자는 이미지에 spaCy 모델도 설치합니다(프라이빗 VPC의 태스크는 모델을 내려받을 수 없음).
`eval`도 빠져 있으므로 질의 태스크에서 ragas 심사 평가기를 쓰려면 `"--extra eval"`을 추가하세요.
이미지의 `docker/config.yaml`에는 엔드포인트가 없으며 CDK 태스크 정의(task definition)가 주입합니다.

**OWL/RDFS 온톨로지**를 추출 스키마로 쓰려면 `ontology` extra가 필요합니다
(`uv sync --extra ontology`, [Ontology](../guides/ontology.md#install), 영문).

## 인증

1. **AWS 자격 증명**은 표준 자격 증명 체인에서 가져옵니다. `aws.profile_name`은
   프로파일 이름이며, `null`이면 기본 체인(환경 변수, 인스턴스 역할 등)을 씁니다.
   `aws.neptune.use_iam: true`이면 Neptune 요청에 SigV4로 서명합니다.
2. **OpenSearch**는 IAM(`aws.opensearch.use_iam: true`)을 쓰거나,
   `use_iam: false`(기본값)이면 `.env` 파일(`.env-template` 복사)의 사용자 이름과 비밀번호를 씁니다. `run-*` CLI가
   이 파일을 읽습니다.

   ```bash
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

   인증 방식은 하나만 씁니다. `use_iam: true`와 사용자 이름이나 비밀번호를 함께 설정하면 서명한 요청이 이를 무시하므로
   설정 검증에서 실패합니다.

## IAM 권한

CLI를 실행하는 주체에는 Bedrock(설정한 모델 ID와 임베딩에 대한 `InvokeModel`,
`InvokeModelWithResponseStream`, `CountTokens`. `CountTokens`가 없으면 토큰 수를
추정함), Neptune(`use_iam: true`일 때 connect / SigV4), OpenSearch(설정한 인덱스
읽기·쓰기), S3(설정한 버킷), DynamoDB(증분 인덱싱을 켠 경우) 권한이 필요합니다. Bedrock
작업 세 가지는 별도 정책 문이 필요합니다.

- **`Resource: "*"`에 `bedrock:Rerank`.** Rerank API는 `InvokeModel`과 다른 리소스
  형태로 인가하므로 모델 ARN이나 추론 프로파일 ARN으로 범위를 좁힌 정책 문으로는
  거부됩니다(CDK 태스크 역할은 별도로 허용함). 이 권한이 없으면 질의마다
  `Reranking failed: ...` ERROR가 남고 재순위화하지 않은 융합 결과를
  반환합니다(`metadata.rerank.status: failed`). 권한을 줄 수 없다면
  `providers.rerank.kind: none`으로 설정하세요.
- **`Resource: "*"`에 `bedrock:ListInferenceProfiles`.** 모델 해석기가 호출하는 계정
  수준 읽기 권한입니다.
- **`bedrock:ApplyGuardrail`.** 가드레일을 쓰면 해당 가드레일
  ARN(`arn:aws:bedrock:<Bedrock 리전>:<계정>:guardrail/<id>`)에 필요합니다
  ([가드레일](../guides/guardrail.md), 영문).

## 첫 실행

템플릿을 복사한 뒤 `config.yaml`에 저장소 엔드포인트 두 개(`aws.neptune.endpoint`,
`aws.opensearch.endpoint`)를 적고, 저장소가 `us-west-2`에 있지 않으면
`aws.region_name`도 지정하세요. IAM(SigV4) 서명 요청을 허용하는 도메인이라면
`aws.opensearch.use_iam: true`도 설정하세요. 기본값 `false`는 `.env`의 사용자 이름과
비밀번호를 씁니다([인증](#인증)).

```bash
cp config-template.yaml config.yaml
```

`uv run`은 프로젝트 환경에서 CLI를 실행합니다. `pip`로 설치했다면 가상 환경을 활성화하고
`uv run` 없이 실행하세요. 코퍼스를 인덱싱한 뒤 GraphRAG나 LightRAG 전략으로 질의합니다.

```bash
uv run run-ingestion --source-directory ./source --config-path config.yaml
uv run run-rag --query "What are the main themes?" --search-strategy global --config-path config.yaml
uv run run-rag --query "How are Alice and Acme related?" --search-strategy mix --config-path config.yaml
```

## 다음 단계

- [설정](../guides/configure.ko.md): 설정을 읽는 순서, 대부분의 실행에서 바꾸는 설정, 환경
  변수, 로그에 남는 내용.
- [수집](../guides/ingest.ko.md)과 [질의](../guides/query.ko.md): 단계, 재개와 재구축,
  필터. 어떤 전략을 쓸지는 [검색](../concepts/retrieval.ko.md)에 있습니다.
- [증분 인덱싱](../guides/incremental.ko.md): 새 문서와 바뀐 문서만 다시 인덱싱.
- [평가](../guides/evaluate.ko.md)(`run-eval`), [Visualization](../guides/visualize.md)(`run-visualization`, 영문),
  [도메인 적응](../guides/domain-adaptation.ko.md)(`run-prompt-tuning`): 전략 채점과 비교,
  내보낸 그래프 렌더링, 코퍼스에 맞춘 프롬프트 튜닝.
- [`iac/README.md`](../../iac/README.md#quick-start)(영문): AWS CDK로 스택 전체를 배포.
