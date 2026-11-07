# MCP 서버

> 🇬🇧 English version: [mcp.md](./mcp.md)

`run-mcp-server`는 수집한 인덱스 하나를 읽기 전용
[Model Context Protocol](https://modelcontextprotocol.io)(MCP) 도구로 에이전트에
제공합니다. MCP를 쓰는 에이전트(IDE 어시스턴트, Strands·LangGraph 같은 프레임워크로
만든 에이전트, Amazon Bedrock AgentCore에서 실행하는 에이전트)는 이 패키지의 Python
API 없이도 인덱스를 검색하고, 질문하고, 그래프를 따라갈 수 있습니다. 서버는 아무것도
쓰지 않습니다. 수집·삭제·메모리 도구가 없습니다. 계층 배치, 보안 판단 근거, 실패 유형은
[설계 문서](../design.md#84-agent-access-the-read-only-mcp-server)(영문)에, 모든
플래그는 [CLI 레퍼런스](../reference/cli.md#run-mcp-server)(영문)에 있습니다.

## 설치와 시작

서버에는 `mcp` extra(공식 MCP Python SDK)가 필요합니다.

```bash
uv sync --extra mcp                      # from a clone
pip install "unified-kg-rag-on-aws[mcp]" # from a package index
```

extra가 없으면 `run-mcp-server`는 extra 이름을 알려 주고 종료 코드 2로 끝납니다.
서버는 `run-rag`와 같은 설정을 읽습니다. 저장소, 모델, 인덱스 네임스페이스
(`--suffix`, 없으면 `processing.document_parsing.index_value`. [질의](./query.ko.md)와
같음)입니다. 에이전트의 MCP 설정에서 서버를 자식 프로세스로 띄우고 표준 입출력으로
통신합니다(기본값인 `stdio` 전송). 대부분의 MCP 클라이언트는 다음과 같은 설정을 받습니다.

```json
{
  "mcpServers": {
    "unified-kg-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/unified-kg-rag-on-aws",
               "run-mcp-server", "--config-path", "/path/to/config.yaml"]
    }
  }
}
```

로그는 stderr로 나가고 stdout에는 프로토콜 메시지만 나갑니다. 클라이언트가 연결을
닫으면 0, SIGTERM이면 143, Ctrl-C면 130으로 끝납니다.

## 도구

| 도구 | 하는 일 | 모델 호출 |
|---|---|---|
| `kg_search` | 검색만 합니다. 질의(어느 언어든)에 대한 근거 콘텐츠를 순위대로 돌려주며, `strategy`, `search_type`, `top_k`, `filters`는 `run-rag --mode search`와 같습니다 | 전략의 질의 처리, 질의 임베딩, 재순위화 |
| `kg_answer` | RAG 답변 전체, 답변 모델이 본 출처, `usage`, 답변 거절 필드(`abstained`, `abstention_reason`) | 위의 호출과 답변 생성 |
| `kg_get_entity` | `id`, `iri`, `name`으로 엔터티를 찾습니다(이름은 먼저 정확히 일치하는 것을, 없으면 가장 비슷한 이름을 찾고 `match: search`로 표시) | 없음 |
| `kg_neighbors` | 한 엔터티에서 `max_hops` 안에 있는 엔터티와 그 사이의 관계 | 없음 |
| `kg_describe_schema` | 네임스페이스가 가진 방법론과 전략, 마지막 빌드, 추출 스키마의 타입과 패턴, 필터 키, 서버 한도 | 없음 |

모든 도구에는 읽기 전용, 멱등(idempotent) 표시가 붙어 있습니다. 먼저
`kg_describe_schema`를 호출하세요. 네임스페이스가 실행할 수 있는 전략(가지지 않은
전략은 `run-rag`처럼 거부됨)과 검색에 쓸 수 있는 필터 키를 알려 줍니다.

## 근거 콘텐츠

`kg_search`와 `kg_answer`는 `records`를 돌려주고, 그래프 도구도 엔터티와 관계를 같은
형태로 돌려줍니다. 그래서 매핑 하나로 어느 결과든 에이전트·온톨로지 도구가 주고받는
청크 중심의 "근거 콘텐츠(supporting content)" 레코드로 바꿀 수 있습니다.

| 필드 | 내용 |
|---|---|
| `rank`, `kind`, `score` | 순위(1이 가장 높음), 항목 종류(`text`, `entity`, `relationship`, `community`, `claim`, `summary`), 관련도 점수 |
| `chunk_id`, `text`, `text_truncated` | 텍스트 레코드의 청크(텍스트 유닛) id, `mcp.max_text_chars`로 자른 텍스트 |
| `source_id`, `name`, `type` | 저장된 항목의 id, 엔터티나 리포트의 이름(관계는 `A -> B`), 엔터티나 관계의 타입 |
| `document_id`, `document_name`, `documents` | 원본 문서. 저장된 경우 id, 파일 이름(경로는 절대 아님). 그래프 항목은 추출된 청크들의 파일 이름 |
| `text_unit_ids` | 그래프 항목이 추출된 청크(처음 20개) |
| `entities` | 참조(`id`, `name`, `type`, `iri`). 엔터티 자신, 관계나 클레임의 양 끝, 또는 같은 응답에서 이 청크를 인용하는 엔터티 |
| `provenance`, `synthesized` | 시드 그래프 출처(`kind`, `seed_graphs`, `iri`). 글로벌 검색의 핵심 요점처럼 모델이 쓴 텍스트면 `true` |
| `attributes` | 종류별 스칼라 값: `rank`, `weight`, `in_schema`, `type_iri`, `predicate_iri`, `hops` |

응답이 `mcp.max_response_chars`를 넘으면 순위가 낮은 레코드부터 빼고 그 사실을
알립니다(`truncated`, `omitted`).

## 범위와 한도

네임스페이스는 서버가 시작할 때 서버 쪽에서 정해집니다.
[테넌트 접미사가 그래야 하는 것](../concepts/scopes-and-namespaces.ko.md#접근-제어-경계가-아님)과
같습니다. 도구 호출은 `mcp.allowed_namespaces`에 있는 네임스페이스만 추가로 읽을 수
있으므로(이때 도구에 `namespace` 인자가 생기고, 목록이 없으면 인자도 없음) 모든 호출자가
읽어도 되는 네임스페이스만 넣으세요. 필터는 검색을 좁힐 뿐 격리 경계가 아닙니다.
엔터티 설명에는 그 엔터티를 언급한 모든 문서의 내용이 합쳐집니다. 다른 사람이 읽으면
안 되는 테넌트는 자체 서버와 네임스페이스로 제공하고
[운영 런북](../operations.md#query-access-and-tenant-isolation)(영문)의 저장소 역할을
적용하세요.

| 키 | 기본값 | 제한 대상 |
|---|---|---|
| `mcp.tools` | 다섯 개 모두 | 제공할 도구. `kg_answer`를 빼면 답변 모델을 호출하지 않음 |
| `mcp.allowed_namespaces` | `[]` | 호출이 추가로 지정할 수 있는 네임스페이스 |
| `mcp.strategies` | 모든 전략 | `kg_search`와 `kg_answer`가 실행할 수 있는 전략. 목록에 `auto`가 있으면 `auto`도 목록 안에서만 라우팅하고, 네임스페이스가 목록의 전략을 하나도 제공하지 않으면 모델 호출 전에 거부 |
| `mcp.max_top_k` | `20` | 검색의 `top_k`, `kg_get_entity`의 `limit` |
| `mcp.max_hops`, `mcp.max_neighbors` | `2`, `50` | `kg_neighbors` 탐색 |
| `mcp.max_query_chars` | `2000` | 질의나 엔터티 이름 |
| `mcp.max_text_chars`, `mcp.max_response_chars` | `2000`, `60000` | 레코드 하나의 텍스트, 응답 하나의 레코드 |
| `mcp.max_concurrent_calls` | `4` | 동시에 실행하는 호출 수. 나머지는 제한 시간 안에서 기다림 |

모든 호출은 `search.query_timeout_seconds`(기본 180초) 안에서 실행되고, 넘으면 시간
초과 메시지로 실패합니다. 호출 슬롯을 기다린 시간도 여기에 포함됩니다. 응답의 레코드는
`mcp.max_response_chars`로 제한되지만 MCP 메시지는 결과를 두 번(구조화된 콘텐츠와 그 JSON
텍스트) 담으므로 실제 메시지는 그 2~4배까지 될 수 있습니다. 도구는 대화 메모리를 유지하지 않습니다. 모든 키는
[설정 레퍼런스](../reference/configuration.md#mcp)(영문)에 있습니다.

## HTTP와 원격 배포

`--transport streamable-http`는 `http://127.0.0.1:8000/mcp`에서 상태 없는(stateless)
엔드포인트를 제공합니다(POST만 받으며 GET과 DELETE에는 405). 어떤 루프백 주소든 그 주소나
루프백 이름을 가리키는 `Host`와 `Origin`만 받으므로 웹 페이지가 DNS 리바인딩으로 접근할 수
없습니다(다른 `Host`는 421, 다른 `Origin`은 403). **서버는 호출자를 인증하지 않습니다.**
다른 주소에 바인딩하려면 `--allow-remote`가 필요하며, 이때 헤더 검사는 앞단의 프록시에
맡깁니다.

```bash
run-mcp-server --config-path config.yaml --transport streamable-http \
  --host 0.0.0.0 --port 8000 --allow-remote
```

인증된 요청만 서버에 닿도록 실행하세요. 인바운드 인증을 켠 Amazon Bedrock AgentCore
Gateway나 AgentCore Runtime 뒤, 모든 요청을 인증하는 리버스 프록시 뒤, 또는 모든
클라이언트가 그 네임스페이스를 읽어도 되는 사설 네트워크 안이어야 합니다. 필요한 IAM
정책은 질의 태스크의 역할(저장소 읽기 권한과 전략이 호출하는 모델)과 같습니다.
AgentCore나 CDK 스택의 질의 태스크에서 실행하는 구성은 아직 AWS에서 검증하지 않았습니다.

호스트 애플리케이션이 서버를 직접 마운트할 수도 있습니다. `build_mcp_app`은 상태 없는
ASGI 애플리케이션을 돌려줍니다. `allowed_hosts`(프록시가 전달하는 `Host` 값)를 주면
`Host`/`Origin` 검사가 켜지고, 없으면 꺼집니다. `mcp.strategies`를 설정했다면 체인을
`allowed_strategies`로 만드세요(더 많은 전략을 허용하는 체인은 도구가 거부합니다).

```python
from unified_kg_rag import GraphRAGChain, build_mcp_app, get_config

config = get_config("config.yaml")
chain = GraphRAGChain(config, allowed_strategies=config.mcp.strategies)
app = build_mcp_app(chain, namespace="default", allowed_hosts=["mcp.example.com"])
# Serve `app` with any ASGI server behind authentication; close the chain
# (await chain.aclose()) when the application stops.
```

## 비용, 텔레메트리, 신뢰할 수 없는 텍스트

- **비용.** `kg_search`는 `run-rag --mode search`, `kg_answer`는 `run-rag`와 비용이
  같고, 그래프 도구는 모델을 호출하지 않습니다. 에이전트는 작업 하나에서 도구를 여러 번
  호출할 수 있으니 `mcp.tools`와 `mcp.strategies`로 범위를 제한하세요. `kg_search`와
  `kg_answer`는 `usage`(역할·모델별 모델 호출 수와 토큰)를 돌려줍니다.
- **텔레메트리.** [관측성](./observability.md)(영문)을 켜면 호출마다
  `kg_rag.mcp.tool` 스팬이 생깁니다. 인자는 `observability.capture_content`를 켰을 때만
  `redact_patterns`를 적용한 뒤 기록합니다. 로그에는 질의 텍스트가 아니라 요약값(digest)만
  남습니다. 저장소나 모델 오류는 일반 메시지로 호출자에게 전달되고, 자세한 내용은 서버
  로그에 있습니다.
- **신뢰할 수 없는 텍스트.** 청크 텍스트, 설명, 답변은 코퍼스나 모델에서 나옵니다. 도구
  설명은 에이전트에게 이를 지시가 아닌 데이터로 다루라고 알리고, 도구는 이를 구조화된
  필드로 돌려줍니다. 서버에는 쓰기 경로가 없으므로 심어 둔 텍스트로 서버가 무언가를 바꾸게
  할 수는 없습니다. 다만 다른 도구를 가진 에이전트를 움직일 수는 있으며, 이는 에이전트
  자체의 보호 장치가 막아야 합니다. `kg_search`와 `kg_answer`는
  설정된 [가드레일](./guardrail.md)(영문)을 거치고, 그래프 도구는 모델을 호출하지 않으며
  저장된 텍스트를 그대로 돌려줍니다.

## 문제 해결

| 메시지 | 해결 |
|---|---|
| `The MCP server needs the mcp extra` | `uv sync --extra mcp` |
| `Missing endpoint configuration for the MCP server` | `run-rag`처럼 저장소 엔드포인트를 설정([설정](./configure.ko.md)) |
| `is not a loopback address ... needs --allow-remote` | `127.0.0.1`에 바인딩하거나, 인증 프록시 뒤에서 `--allow-remote` 추가 |
| `Strategy '...' is not offered` / `does not hold` | `kg_describe_schema`를 호출해 목록에 있는 전략 선택 |
| `holds no knowledge graph` | 네임스페이스를 `raptor`만으로 빌드함. 그래프 도구에는 graphrag, lightrag, hipporag 중 하나가 필요 |
| `The call failed inside the server` | 서버의 stderr 로그 확인 |
