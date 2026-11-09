# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Korean."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.KO,
    name="Korean",
    script="hangul",
    # The analysis-nori plugin's analyzer.
    analyzer="nori",
    decline="사용 가능한 데이터에서 이 질문에 답할 관련 정보를 찾지 못했습니다.",
    claim_statuses={
        "true": ("참", "진실", "사실", "확인됨", "검증됨"),
        "false": ("거짓", "허위", "부인됨", "반박됨"),
        "disputed": ("논란", "분쟁", "의심", "불확실", "미확인"),
        "unknown": ("알수없음", "불명", "없음"),
    },
    no_value_words=("미상", "불명", "알수없음", "없음", "해당없음", "모름"),
    open_date_words=("미정", "현재", "진행중"),
    # Stock, limited, limited liability, limited partnership and general
    # partnership companies; incorporated associations, foundations.
    dense_legal_forms=(
        "(주)",
        "주식회사",
        "(유)",
        "유한회사",
        "유한책임회사",
        "합자회사",
        "합명회사",
        "(사)",
        "사단법인",
        "(재)",
        "재단법인",
    ),
    # Native numerals, in the form they take before a counter ("세 곳",
    # "두 군데", "스무 개사") or alone ("셋"), ones and tens apart: 열세 is
    # 13. A native numeral counts only before a counter, since "네" alone is
    # also "yes" and "세" a word of its own.
    counter_numerals={
        "한": 1,
        "하나": 1,
        "두": 2,
        "둘": 2,
        "세": 3,
        "셋": 3,
        "석": 3,
        "네": 4,
        "넷": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
    },
    number_tens={"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40},
    # Sino-Korean numerals ("십이 대", "만 이천 명").
    multiplier_numerals=dict(zip("일이삼사오육칠팔구", range(1, 10), strict=True)),
    numeral_multipliers={"십": 10, "백": 100, "천": 1000, "만": 10**4, "억": 10**8},
    year_suffixes=("년", "이후", "이전", "이래"),
    range_particles=("부터", "까지"),
    # The comparatives written after a number and its counter ("100명
    # 이상", "100명을 넘는"): a bound, or a condition of what is absent.
    postposed_comparatives=("이상", "이하", "초과", "미만", "넘는"),
    counters=("개사", "군데", "가지", "곳", "개", "명", "건", "회", "번", "대"),
    polarity_words={
        "예": "yes",
        "네": "yes",
        "아니요": "no",
        "아니오": "no",
        "아닙니다": "no",
    },
    chars_per_token={CLAUDE_TOKENIZER: 1.0},
)
