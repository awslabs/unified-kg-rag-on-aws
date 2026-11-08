# 범위와 네임스페이스

> 🇬🇧 English version: [scopes-and-namespaces.md](./scopes-and-namespaces.md)

실행이 어디에 쓰고 무엇을 삭제할 수 있는지, 질의가 무엇을 읽는지는 세 가지 이름으로
정해집니다. **인덱스 접미사(index suffix)**는 코퍼스의 인덱스 이름을, **소스 범위(source
scope)**는 실행이 비교하는 코퍼스 소스를 정하고, **네임스페이스(namespace)**는 쓰기
실행 하나가 한 번에 맡는 인덱스 묶음입니다. 셋 다 접근 제어 수단은 아닙니다.

## 인덱스 접미사

코퍼스의 인덱스와 그래프 레이블 이름은 모두 `<prefix>-<suffix>`이며
`indexing.additional_suffix`를 지정하면 `-<additional_suffix>`가 붙습니다. 이
`<suffix>`가 **인덱스 접미사**이고, CLI는 그 결과를 실행의 **Index Namespace**로
출력합니다. 수집은 `run-ingestion --suffix`로 다른 값을 지정하지 않으면
`processing.document_parsing.index_value`에서 가져오고, `run-rag`와
`run-eval`은 `--suffix`로 다른 값을 지정하지 않으면 설정 파일의 같은 `index_value`를
질의합니다. Python에서는 `RAGInput.suffix`로 고릅니다(`None`이면 체인 설정의
`index_value`). 모두 기본값은 `default`입니다.
테넌트나 코퍼스 버전마다 인덱스 접미사를 하나씩 두고(수집과 질의에 같은 `--suffix`나 같은
설정 파일),
`indexing.additional_suffix`는 양쪽에서 같게 하세요. 증분 인덱싱과 삭제는 인덱스 접미사
단위입니다([증분 인덱싱](../guides/incremental.ko.md)).

## 소스 범위

실행은 자기 **범위** 안에서 비교합니다. 범위는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 **소스 범위**, 즉 코퍼스 소스(`document_parsing.source_scope`. 기본값은
해석된 소스 디렉터리이며, 컨테이너 엔트리포인트는 동기화 원본인 S3 URI로 설정. 끝의
슬래시가 있든 없든 같은 소스 범위이며, 쓰인 표기는 WARNING에 남음)입니다.
문서의 레지스트리 키는 이 범위와 소스 디렉터리 기준 상대 경로이며, 실행은 자기 범위의
문서만 비교하고 삭제 대상으로 봅니다. 따라서 다른 테넌트(다른 `index_value`)의 실행은 이
테넌트의 문서를 삭제하지 않고, 하위 폴더를 소스 디렉터리로 따로 실행해도 코퍼스의
나머지는 삭제되지 않으며(그 파일들은 별도 문서로 등록되므로 같은 파일을 두 루트에서 한
접미사로 인덱싱하지 마세요), 소스가 다른 두 코퍼스는 한 접미사에서 상대 경로가 같아도
서로 다른 문서입니다. 문서와 텍스트 단위 ID도 범위에서 만들어지므로, 경로와 머신이 달라도
같은 ID가 나와야 하면 `source_scope`를 지정하세요(지정하지 않으면 같은 코퍼스라도 다른
디렉터리나 호스트에서 인덱싱하면 ID가 달라집니다).

코퍼스를 한 소스에 두는 방법, 코퍼스 옮기기, 대량 삭제 확인은
[증분 인덱싱: 삭제 범위](../guides/incremental.ko.md#삭제-범위)에 있습니다.

## 네임스페이스

**네임스페이스**는 인덱스 접미사(`index_value`, 지정하지 않으면 `default`)에
`indexing.additional_suffix`와, 기본값이 아닌 `indexing.opensearch.*_index_prefix` /
`indexing.neptune.*_label_prefix` 값의 태그를 더한 것입니다. 로그와 오류 메시지에서는
*index namespace*라고 부릅니다(`Index namespace '<namespace>' is being written by ...`).
문서 상태 레지스트리를 쓰면
네임스페이스 하나에는 한 번에 실행 하나만 씁니다(그 실행이 네임스페이스의 쓰기 임대(lease)를
가짐). 네임스페이스가 다르면 동시에 실행할 수 있습니다. 임대, 비정상 종료 뒤의 인계,
레지스트리 없이 실행할 때의 동작은 [운영 런북: 동시 실행](../operations.md#concurrent-runs)(영문)에
있습니다.

## 접근 제어 경계가 아님

접미사는 이름을 나눌 뿐 접근을 막지 않습니다. 질의할 수 있는 호출자는 어떤 접미사든
지정할 수 있으므로, 멀티 테넌트 서비스는 인증된 호출자를 기준으로 서버에서 접미사를
정해야 합니다.

**필터도 접근 제어 경계가 아닙니다.** 키를 선언하지 않은 저장소는 내용을 거르지 않고
돌려주며, 여러 문서에서 나온 엔티티, 관계, 커뮤니티 리포트는 그중 하나만 일치해도
통과하므로 제외된 문서의 내용을 담을 수 있습니다. 여러 문서에서 나온 항목은 여러 키로 된
필터의 키마다 따로 일치하기도 합니다. 시드, 관계, 주장, 커뮤니티 리포트는 키마다 모든 문서의
값을 가지므로, 문서가 `{category: x, grp: a}`와 `{category: y, grp: b}`인 항목은 두 조건을
모두 갖춘 문서가 없어도 `attr_category:x attr_grp:b`에 일치합니다.
테넌트나 사용자는 서버에서 정한 [인덱스 접미사](#인덱스-접미사)로 나누세요.

인프라에서 테넌트를 격리하려면 질의에 읽기 전용 역할을 따로 주고, 테넌트별 OpenSearch
역할을 그 테넌트의 인덱스로 한정하고, 다른 테넌트가 그래프를 읽으면 안 되는 테넌트에는
Neptune 클러스터를 따로 두세요.
[운영 런북: 질의 권한과 테넌트 격리](../operations.md#query-access-and-tenant-isolation)(영문)를
참고하세요.
