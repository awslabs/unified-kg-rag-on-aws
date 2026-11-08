# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One synthetic sample per language of the matrix the language tests run
over: Arabic, Hebrew, Thai, Hindi and Vietnamese (scripts and spacing the
CJK and Latin fixtures do not exercise), German and Russian, and Korean,
Japanese and Chinese.

Each sample is two sentences of one made-up delivery (a vendor, a
warehouse, an invoice), a question about it, a name that occurs in the
text as written, and the tokens two BPE vocabularies count in the text
(tiktoken's ``cl100k_base`` and ``o200k_base``; ``test_token_counter``
recounts them when tiktoken has both cached).
"""

from __future__ import annotations

from dataclasses import dataclass

from unified_kg_rag.domain.languages import LanguageCode


@dataclass(frozen=True)
class LanguageSample:
    code: LanguageCode
    text: str
    question: str
    name: str
    # The text's tokens under cl100k_base and o200k_base.
    bpe_tokens: tuple[int, int]


MATRIX: tuple[LanguageSample, ...] = (
    LanguageSample(
        LanguageCode.AR,
        "سلّم المورد وحدات الليدار إلى المستودع في الثاني من يونيو. "
        "وافق المشتري على الفاتورة بعد الفحص.",
        "من هو المورد الذي سلّم الوحدات؟",
        "المستودع",
        (57, 29),
    ),
    LanguageSample(
        LanguageCode.HE,
        "הספק מסר את יחידות הלידאר למחסן בשני ביוני. "
        "הקונה אישר את החשבונית לאחר הבדיקה.",
        "מי הספק שמסר את היחידות?",
        "החשבונית",
        (74, 31),
    ),
    LanguageSample(
        LanguageCode.TH,
        "ผู้จัดหาส่งมอบหน่วยลิดาร์ไปยังคลังสินค้าในวันที่สองมิถุนายน "
        "ผู้ซื้ออนุมัติใบแจ้งหนี้หลังการตรวจสอบ",
        "ใครเป็นผู้จัดหาที่ส่งมอบหน่วยลิดาร์",
        "คลังสินค้า",
        (92, 37),
    ),
    LanguageSample(
        LanguageCode.HI,
        "आपूर्तिकर्ता ने दो जून को गोदाम में लिडार इकाइयाँ पहुँचाईं। "
        "निरीक्षण के बाद खरीदार ने चालान को मंज़ूरी दी।",
        "इकाइयाँ किस आपूर्तिकर्ता ने पहुँचाईं?",
        "गोदाम",
        (114, 38),
    ),
    LanguageSample(
        LanguageCode.VI,
        "Nhà cung cấp đã giao các thiết bị lidar đến kho vào ngày hai tháng sáu. "
        "Người mua đã phê duyệt hóa đơn sau khi kiểm tra.",
        "Nhà cung cấp nào đã giao các thiết bị này?",
        "hóa đơn",
        (52, 33),
    ),
    LanguageSample(
        LanguageCode.DE,
        "Der Lieferant hat die Lidar-Einheiten am zweiten Juni an das Lager "
        "geliefert. Der Käufer genehmigte die Rechnung nach der Prüfung.",
        "Welcher Lieferant hat die Einheiten geliefert?",
        "Lager",
        (40, 29),
    ),
    LanguageSample(
        LanguageCode.RU,
        "Поставщик доставил лидарные блоки на склад второго июня. "
        "Покупатель утвердил счёт после проверки.",
        "Какой поставщик доставил блоки?",
        "склад",
        (43, 26),
    ),
    LanguageSample(
        LanguageCode.KO,
        "공급업체는 6월 2일 창고에 라이다 장치를 납품했다. "
        "구매자는 검사 후 송장을 승인했다.",
        "어느 공급업체가 장치를 납품했습니까?",
        "창고",
        (54, 32),
    ),
    LanguageSample(
        LanguageCode.JA,
        "供給業者は6月2日に倉庫へライダー装置を納品した。"
        "買い手は検査後に請求書を承認した。",
        "どの供給業者が装置を納品しましたか？",
        "倉庫",
        (55, 38),
    ),
    LanguageSample(
        LanguageCode.ZH,
        "供应商于6月2日将激光雷达设备交付到仓库。买方在检查后批准了发票。",
        "哪个供应商交付了设备？",
        "仓库",
        (40, 29),
    ),
)

MATRIX_IDS = [sample.code.value for sample in MATRIX]
