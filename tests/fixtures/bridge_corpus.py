# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multilingual bridge corpus: a two-hop question per language whose
answer only a passage citing both of its facts' people states.

``tests/integration/test_ppr_embedded.py`` pins HippoRAG 2's PageRank on it
and the golden-metrics gate (``tests/fixtures/golden_metrics.py``) scores
every strategy on it. Synthetic names only.
"""

from __future__ import annotations

from typing import Any

from unified_kg_rag.domain.models.config import LanguageCode

# Per language: the people (A founded C; B is the mayor of D), the files and
# the question and its answer. accord.txt, the bridge, is the only passage citing A and B;
# each decoy repeats the question's words and names C alone (no fact), so a
# seed cites it. A passage is one chunk (under 60 characters).
CORPORA: dict[LanguageCode, dict[str, Any]] = {
    LanguageCode.EN: {
        "names": ("Alvar Pell", "Corin Works", "Bex Lund", "Dorrit"),
        "files": {
            "founder.txt": "Alvar Pell founded Corin Works.",
            "mayor.txt": "Bex Lund is the mayor of Dorrit.",
            "accord.txt": "Alvar Pell and Bex Lund signed the Tarn Accord.",
            **{
                f"decoy{i}.txt": f"Corin Works founder and mayor: what was signed? {i}"
                for i in range(4)
            },
        },
        "question": "What did the founder of Corin Works and the mayor of Dorrit sign?",
        "answer": "Tarn Accord",
    },
    LanguageCode.KO: {
        "names": ("홍길동", "가나다상사", "김철수", "도리시"),
        "files": {
            "founder.txt": "홍길동은 가나다상사를 설립했다.",
            "mayor.txt": "김철수는 도리시의 시장이다.",
            "accord.txt": "홍길동과 김철수는 타른협정에 서명했다.",
            **{
                f"decoy{i}.txt": f"가나다상사 설립자와 시장이 서명한 것은 무엇인가 {i}"
                for i in range(4)
            },
        },
        "question": "가나다상사 설립자와 도리시 시장이 서명한 것은 무엇인가?",
        "answer": "타른협정",
    },
    LanguageCode.JA: {
        "names": ("山田太郎", "青葉商事", "佐藤花子", "緑市"),
        "files": {
            "founder.txt": "山田太郎は青葉商事を設立した。",
            "mayor.txt": "佐藤花子は緑市の市長である。",
            "accord.txt": "山田太郎と佐藤花子は北浜協定に署名した。",
            **{
                f"decoy{i}.txt": f"青葉商事の設立者と市長が署名したものは何か{i}"
                for i in range(4)
            },
        },
        "question": "青葉商事の設立者と緑市の市長が署名したものは何か？",
        "answer": "北浜協定",
    },
    LanguageCode.ZH: {
        "names": ("王小明", "青松公司", "李华", "绿城"),
        "files": {
            "founder.txt": "王小明创立了青松公司。",
            "mayor.txt": "李华是绿城的市长。",
            "accord.txt": "王小明和李华签署了北海协定。",
            **{
                f"decoy{i}.txt": f"青松公司的创立者和市长签署了什么{i}"
                for i in range(4)
            },
        },
        "question": "青松公司的创立者和绿城的市长签署了什么？",
        "answer": "北海协定",
    },
}

# The passages the answer needs, in every language: both facts and the bridge.
SUPPORT = frozenset({"founder.txt", "mayor.txt", "accord.txt"})
