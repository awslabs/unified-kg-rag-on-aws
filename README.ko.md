# Unified Knowledge Graph RAG on AWS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)

🇬🇧 **[English README](./README.md)** · 🤝 **[기여 가이드](./CONTRIBUTING.md)**

<p align="center">
  <img src="./assets/profile.png" alt="Unified Knowledge Graph RAG on AWS" width="320">
</p>

대규모 다국어 문서 코퍼스를 Amazon Neptune과 Amazon OpenSearch Service 위의 지식 그래프로 만들고, Amazon Bedrock으로 그 그래프에 대한 질문에 답하는 AWS 네이티브 지식 그래프 RAG(검색 증강 생성) 프레임워크입니다. 멀티홉 그래프 순회로 여러 문서에 걸친 추론을 수행합니다. Microsoft GraphRAG와 LightRAG를 하나의 공통 스택 위에 다시 구현했기 때문에, 검색 방법론을 배포 단위가 아니라 질의 단위로 고를 수 있습니다. 하나의 스택에서 두 방법론 운영, 3중 하이브리드 검색, 증분 인덱싱, 다국어 처리는 원 논문을 의도적으로 확장한 부분입니다.

전체 아키텍처, 두 검색 방법론과 벤치마크 결과는 [AWS Open Source Blog 소개 글](https://aws.amazon.com/ko/blogs/opensource/unified-knowledge-graph-rag-on-aws-graphrag-and-lightrag-on-one-stack/)에서 볼 수 있습니다.

## 이 프레임워크를 쓰는 이유

- **하나의 AWS 스택에서 두 방법론.** GraphRAG(커뮤니티 요약)와 LightRAG(이중 레벨 키워드)가 수집, 인덱싱, 캐싱, 검색을 함께 쓰고 검색 알고리즘만 다르며, 방법론은 질의마다 고릅니다. 모든 답변은 실제로 모델 컨텍스트에 들어간 출처를 함께 돌려줍니다.
- **3중 하이브리드 검색.** BM25, 벡터 검색, Neptune 그래프 순회를 RRF(Reciprocal Rank Fusion)로 합치고 Bedrock rerank 모델로 재순위화합니다.
- **증분 인덱싱.** 문서 상태 레지스트리(`providers.doc_status.enabled`)를 켜면 새 문서와 바뀐 문서만 다시 인덱싱해 운영 중인 그래프에 병합하고, 문서를 지우면 다른 문서가 기술하지 않는 내용만 제거합니다.
- **다국어.** 인덱싱과 질의 시점의 선택적 번역, 언어별 분석기(예: 한국어 `nori`), 다국어 키워드 추출을 지원합니다.
- **프롬프트 튜닝, 평가, 시각화.** `run-prompt-tuning`은 코퍼스 표본에서 도메인에 맞춘 프롬프트를 쓰고, `run-eval`은 LangChain·RAGAS 심사에 LLM 없는 커버리지·검색·답변 일치 지표를 더하며, `run-visualization`은 내보낸 그래프를 렌더링합니다.
- **교체 가능한 구조.** 저장소와 모델 백엔드는 포트 뒤에 있고, 전략·렌더러·파서·공급자는 레지스트리로 등록합니다([확장 가이드](./docs/design.ko.md#15-확장-가이드)).

## 아키텍처

![아키텍처 한눈에 보기](./assets/architecture-overview.drawio.png)

수집은 중단 후 재개할 수 있는 12단계 파이프라인(파싱, 청킹, 엔티티·관계 추출, 해석, LLM 리포트가 붙는 커뮤니티 탐지, 인덱싱)으로 그래프·검색 저장소에 씁니다. AWS에서는 Neptune과 OpenSearch, 로컬 모드에서는 내장 SQLite나 컨테이너입니다.

모든 전략은 같은 하이브리드 스코어러와 토큰 예산을 거칩니다. CLI에서는 `--search-strategy`, Python에서는 `RAGInput.search_strategy`로 전략을 고릅니다.

| 전략 | 방법론 | 적합한 경우 |
|---|---|---|
| `auto`(기본값) | GraphRAG | 어떤 전략이 맞을지 모를 때. LLM 라우터가 질의마다 `search.auto_routable_strategies`(기본값 `local`, `mix`, `global`. `drift`도 고르게 하려면 추가) 중 하나를 고릅니다 |
| `simple` | GraphRAG | 빠른 사실 조회. 그래프 순회 없이 벡터와 키워드로 검색합니다 |
| `local` | GraphRAG | 특정 엔티티와 그 관계에 대한 질문 |
| `global` | GraphRAG | 넓은 범위나 주제 중심 질문. 커뮤니티 리포트에 map-reduce를 적용해 답합니다 |
| `drift` | GraphRAG | 반복 탐색이 필요한 복잡한 질문 |
| `mix` | LightRAG | 일반적인 LightRAG 사용. 엔티티·관계 키워드 검색에 청크 검색을 더합니다 |
| `hybrid` | LightRAG | 키워드 중심의 그래프 질문. 청크 검색은 더하지 않습니다 |
| `naive` | LightRAG | 그래프 없이 청크만 검색하는(BM25 + 벡터) 빠른 기준선과 비교 실험 |

전략별 동작은 [사용자 가이드 §4](./docs/user-guide.ko.md#4-질의-run-rag)를, 레이어 구조·알고리즘·데이터 모델과 [수집](./docs/design.ko.md#4-수집-파이프라인)·[증분 인덱싱](./docs/design.ko.md#5-증분-인덱싱)·[검색](./docs/design.ko.md#6-검색-두-방법론) 상세 다이어그램은 [설계 문서](./docs/design.ko.md)를 참고하세요.

## 빠른 시작

### 실행 방법 선택

| 방식 | 그래프·벡터 저장소 | 모델 | 시작 방법 |
|---|---|---|---|
| **로컬 모드**(개발용, AWS 불필요) | 서버 없는 내장 SQLite 그래프·벡터 저장소와 레지스트리, 또는 컨테이너로 띄운 Gremlin Server와 OpenSearch([`docker/compose.local.yaml`](./docker/compose.local.yaml)) | Ollama(또는 OpenAI 호환 서버)나 Amazon Bedrock | [로컬 모드](./docs/user-guide.ko.md#로컬-모드-aws-없이)를 따라 Ollama 모델 두 개를 내려받은 뒤 `--backend local`을 넘깁니다 |
| **기존 AWS 리소스** | 사용 중인 Neptune 클러스터와 OpenSearch 도메인(증분 인덱싱을 쓰면 DynamoDB, 단계 캐시 동기화나 임베딩 캐시 저장을 쓸 때만 S3 버킷) | Amazon Bedrock | `config-template.yaml`을 `config.yaml`로 복사하고 엔드포인트를 입력합니다 |
| **번들 CDK 스택** | [`iac/`](./iac/README.md)가 Fargate 데이터 플레인, Step Functions 수집 파이프라인과 함께 생성 | Amazon Bedrock | [AWS에 배포](#aws에-배포선택)를 진행한 뒤 [`iac/README.md`의 After deploy](./iac/README.md#after-deploy)를 따릅니다 |

AWS 방식은 Amazon Bedrock을 호출하므로 먼저 [필요한 모델](./docs/user-guide.ko.md#필수-모델)의
액세스를 활성화하세요. 로컬 모드는 개발과 평가 전용입니다. 내장 저장소는 호스트 한 대와
인덱스당 문서 약 100,000개를 기준으로 하고, 컨테이너에는 인증과 TLS가 없으며, 작은
로컬 모델은 그래프를 더 약하게 추출합니다.

### 사전 요구사항

Python 3.11–3.12와 [uv](https://docs.astral.sh/uv/)(`pip`도 가능)가 필요합니다. 로컬 모드에는 [Ollama](https://ollama.com)(컨테이너 저장소를 쓸 때만 Compose 2.23.1 이상의 Docker)가, AWS 방식에는 Bedrock 접근 권한이 있는 자격 증명과, 기존 리소스를 쓸 때 Neptune 클러스터와 OpenSearch 도메인이 필요합니다(프레임워크는 서비스에 연결만 하고 만들지 않음). CDK 스택에 필요한 것은 [`iac/README.md`](./iac/README.md#quick-start)에 있습니다.

### 설치와 설정

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync --extra eval --extra viz        # 또는: pip install -e '.[eval,viz]'
cp config-template.yaml config.yaml     # Bedrock 리전과 서비스 엔드포인트 입력
```

OpenSearch가 IAM 대신 사용자 이름/비밀번호 인증을 쓴다면(`aws.opensearch.use_iam: false`) `.env-template`을 CLI를 실행할 디렉터리에 `.env`로 복사하고 자격 증명을 입력하세요. `uv sync`만으로는 수집과 질의만 설치되며, `run-eval`의 LangChain·RAGAS 심사 평가기에는 `eval` extra, `run-visualization`에는 `viz` extra가 필요합니다. Markdown과 HTML을 파싱하려면 선택 extra인 `unstructured`가 필요합니다(`uv sync --extra unstructured`). PDF, TXT, CSV, JSON은 별도 설치 없이 처리합니다.

### 수집, 질의, 평가

`uv run`은 프로젝트 환경에서 CLI를 실행합니다. `pip`으로 설치했다면 가상 환경을 활성화하고 `uv run` 없이 실행하세요.

```bash
# 코퍼스 수집과 인덱싱 (문서 상태 레지스트리를 켜면 증분 인덱싱)
uv run run-ingestion --source-directory ./source --config-path config.yaml

# 두 방법론 중 하나로 질의하거나 대화 메모리를 켜고 대화형으로 실행
uv run run-rag --query "문서의 주요 주제는?" --search-strategy global --config-path config.yaml
uv run run-rag --query "Alice와 Acme는 어떤 관계인가?" --search-strategy mix --config-path config.yaml
uv run run-rag --interactive --use-memory --conversation-id my-session --config-path config.yaml

# 평가 (LangChain + RAGAS + 그래프 인식 커버리지, 검색·답변 일치 지표)
uv run run-eval --data-path eval_data.json --config-path config.yaml

# 선택: 내보낸 그래프 시각화, 도메인 맞춤 프롬프트 튜닝
uv run run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml
uv run run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

모든 CLI, 평가 데이터 형식과 평가기는 [사용자 가이드](./docs/user-guide.ko.md)에, 모든 플래그는 [CLI 레퍼런스](./docs/cli.md)(영문)에, 모든 설정 키는 [설정 레퍼런스](./docs/configuration.md)(영문)에 있습니다.

## AWS에 배포(선택)

[`iac/`](./iac/README.md)의 AWS CDK 앱이 스택 전체를 만듭니다. 엔드포인트를 갖춘 VPC, Neptune, OpenSearch, DynamoDB 문서 상태 테이블, S3 캐시 버킷, ECS Fargate 데이터 플레인, Step Functions 수집 파이프라인, CloudWatch 대시보드와 경보, 선택적 Bedrock Guardrail이 포함됩니다.

`cdk synth`, `cdk bootstrap`, `cdk deploy --all` 명령과 `-c key=value` 옵션 전체는 [`iac/README.md`](./iac/README.md#quick-start)에 있습니다. 저장소는 VPC 안에서만 접근할 수 있으므로 배포된 Fargate 태스크가 CLI를 실행하며, 이미지를 올리고 수집을 시작하고 질의나 평가 태스크를 실행하는 명령은 [After deploy](./iac/README.md#after-deploy)에 있습니다.

> **비용과 정리.** Neptune과 OpenSearch는 시간 단위로 과금되며, `public` 네트워크 모드에서는 NAT 게이트웨이 비용이 추가됩니다. 기본 `dev` 환경은 `cdk destroy --all`로 정리되지만, dev가 아닌 환경은 기본적으로 상태 저장소를 보존하고 삭제 방지를 켭니다. 프로덕션에 쓰기 전에 [`iac/README.md`](./iac/README.md)의 강화 옵션(`use_cmk`, `deletion_protection`, 다중 AZ 구성, `vpc_flow_logs`)을 검토하세요.

## 문서 안내

| 문서 | 내용 |
|---|---|
| [사용자 가이드](./docs/user-guide.ko.md) ([English](./docs/user-guide.md)) | 설정, CLI 플래그, 증분 인덱싱, 평가, 라이브러리 사용법, [제한 사항](./docs/user-guide.ko.md#12-제한-사항) |
| [CLI 레퍼런스](./docs/cli.md) (영문) | 모든 CLI 플래그의 기본값과 설명(파서에서 생성) |
| [설정 레퍼런스](./docs/configuration.md) (영문) | 모든 설정 키의 타입, 기본값, 설명(코드에서 생성) |
| [모델 카탈로그](./docs/models.ko.md) ([English](./docs/models.md)) | 지원하는 Bedrock 모델과 공급자별 요청 차이 |
| [운영 런북](./docs/operations.ko.md) ([English](./docs/operations.md)) | 재수집, 증분 실행, 실패한 문서, 캐시, 자주 보는 오류 |
| [설계 문서](./docs/design.ko.md) ([English](./docs/design.md)) | 헥사고날 아키텍처, 알고리즘, 데이터 모델, 확장 가이드, 참고 자료 |
| [용어집](./docs/glossary.ko.md) | 한국어 문서와 영문 문서·코드의 용어 대응 |
| [`iac/README.md`](./iac/README.md) | CDK 스택, 배포 토폴로지, 비용 요인, `-c key=value` 옵션, Guardrail 배포, 배포 후 명령 |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 개발 환경, 테스트, 품질 게이트, 브랜치와 커밋 규칙 |
| [CHANGELOG.md](./CHANGELOG.md) | 릴리스 노트 |
| [SECURITY.md](./SECURITY.md) | 보안 이슈 신고 |

## 보안 및 면책

이 프로젝트는 **교육과 예시 목적의 참조 프레임워크**입니다. 어떠한 보증도 없이 "있는 그대로(AS IS)" 제공됩니다([LICENSE](./LICENSE) 참고). **별도의 보안 테스트, 위협 모델링, 보안 강화 없이 프로덕션 환경에 배포해서는 안 됩니다.**

- 본인의 AWS 계정에서 본인의 리소스를 대상으로 실행하세요. 배포 환경의 IAM 정책, 네트워크 구성, 데이터 분류, 최종 사용자 인증은 사용자 책임입니다.
- 선택 사항인 CDK 스택(`iac/`)은 안전한 기본값(프라이빗 VPC 격리, AWS 관리형 키 또는 `use_cmk`로 켜는 고객 관리형 KMS 키를 이용한 저장 데이터 암호화, TLS 강제, 최소 권한 IAM, PII·프롬프트 공격 필터링용 선택적 Bedrock Guardrail)을 제공하지만, 배포 책임은 사용자에게 있으므로 환경에 맞게 검토해야 합니다.
- 프로덕션에 쓰기 전에 Bedrock Guardrail(`aws.bedrock.guardrail`)을 켜고, 용도에 맞는 호출 제한과 모니터링을 적용하세요.
- 보안 이슈는 공개 이슈로 올리지 말고 [SECURITY.md](./SECURITY.md)의 절차를 따라 신고하세요.

## 기여

기여를 환영합니다. 개발 환경, 테스트, PR 절차는 [CONTRIBUTING.md](./CONTRIBUTING.md)를, 확장 방법은 [설계 문서 §15](./docs/design.ko.md#15-확장-가이드)를 참고하세요. 대부분의 확장은 레지스트리 등록만으로 끝나며 디스패치 코드를 고칠 필요가 없습니다.

## 라이선스

이 프로젝트는 Apache-2.0 라이선스로 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 참고하세요. AWS가 awslabs 조직에서 유지관리합니다.

## 감사의 말

라이브러리 기능 개발과 검증에 크게 기여해 주신 **강지현**님, 그리고 출시 전 꼼꼼한 리뷰와 실제 코퍼스 테스트를 진행해 주신 **Yusuke Tanimiya**님께 감사드립니다. 두 분 덕분에 프레임워크가 한층 견고해졌습니다.

## 참고문헌

- Microsoft GraphRAG: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) · [라이브러리](https://github.com/microsoft/graphrag)
- LightRAG: [Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) · [라이브러리](https://github.com/HKUDS/LightRAG)

DRIFT 검색, 동적 커뮤니티 선택, 자동 튜닝, LazyGraphRAG 등 GraphRAG 관련 추가 자료는 [설계 문서](./docs/design.ko.md#16-참고-자료)에 정리되어 있습니다.
