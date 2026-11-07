# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's synthetic corpus and evaluation dataset.

Every company, person and figure here is made up: a small procurement story
(a robot maker buys lidar units, a carrier ships them to a depot, a lab
inspects them) in English, with a Korean and a Japanese note on the same
deal, so the graph, the citations and the multilingual paths have something
to show. The dataset's answers are stated in the corpus.
"""

from __future__ import annotations

DOCUMENTS: dict[str, str] = {
    "supply-agreement.txt": (
        "Acme Robotics signed a supply agreement with Globex Sensors on "
        "3 March 2026.\n"
        "Under the supply agreement, Globex Sensors supplies every lidar unit "
        "for the Acme Robotics warehouse robots.\n"
        "The supply agreement sets a late-delivery penalty of $1,000 per day, "
        "capped at $50,000.\n"
        "Alice Chen, the chief engineer at Acme Robotics, approved the lidar "
        "unit specification.\n"
    ),
    "logistics-plan.txt": (
        "Umbra Freight carries each lidar unit from the Globex Sensors plant "
        "to Kestrel Depot.\n"
        "Kestrel Depot stores the units until Acme Robotics calls them off.\n"
        "Umbra Freight must deliver to Kestrel Depot within 10 business days "
        "of a call-off.\n"
        "Ravi Patel, the logistics lead at Globex Sensors, books every "
        "shipment with Umbra Freight.\n"
    ),
    "quality-report.txt": (
        "Orion Labs inspected 400 lidar units from Globex Sensors in May 2026.\n"
        "Orion Labs rejected 12 units for calibration drift.\n"
        "Alice Chen asked Globex Sensors for a corrective action plan.\n"
        "Globex Sensors replaced the 12 rejected units within two weeks.\n"
    ),
    "warranty-terms.txt": (
        "Globex Sensors warrants each lidar unit for 24 months from delivery "
        "to Kestrel Depot.\n"
        "A lidar unit that fails inside the warranty is replaced at no cost "
        "to Acme Robotics.\n"
        "The warranty does not cover damage in transit, which Umbra Freight "
        "insures.\n"
    ),
    "steering-minutes.txt": (
        "The steering group of Acme Robotics met on 2 June 2026.\n"
        "Alice Chen reported that Orion Labs found the replacement units "
        "within tolerance.\n"
        "Acme Robotics decided to extend the supply agreement with Globex "
        "Sensors to a second warehouse.\n"
        "Kestrel Depot will add a second storage bay for Acme Robotics.\n"
    ),
    "contract-summary.ko.txt": (
        "애크미 로보틱스는 2026년 3월 글로벡스 센서스와 라이다 공급 계약을 "
        "맺었다.\n"
        "지연 납품 위약금은 하루 1,000달러이며 상한은 50,000달러다.\n"
        "엄브라 프레이트는 케스트럴 창고까지 운송을 맡는다.\n"
    ),
    "quality-memo.ja.txt": (
        "オリオン・ラボはグローベックス・センサーズのライダー400台を検査した。\n"
        "そのうち12台を較正ずれで不合格とした。\n"
    ),
}

# A run-eval dataset over the corpus (docs/guides/evaluate.md#eval-data-format):
# answers for answer_match, reference documents for retrieval, entities for
# graph_aware, and one question the corpus cannot answer.
DATASET: list[dict[str, object]] = [
    {
        "id": "penalty",
        "question": "What is the late-delivery penalty in the supply agreement?",
        "answer": "$1,000 per day, capped at $50,000",
        "reference_sources": ["supply-agreement.txt"],
        "expected_entities": ["supply agreement", "Globex Sensors"],
        "metadata": {"answer_aliases": ["$1,000 per day"]},
    },
    {
        "id": "carrier",
        "question": "Who carries the lidar units to Kestrel Depot?",
        "answer": "Umbra Freight",
        "reference_sources": ["logistics-plan.txt"],
        "expected_entities": ["Umbra Freight", "Kestrel Depot"],
    },
    {
        "id": "rejected",
        "question": "How many lidar units did Orion Labs reject?",
        "answer": "12",
        "reference_sources": ["quality-report.txt"],
        "expected_entities": ["Orion Labs", "Globex Sensors"],
        "metadata": {"answer_aliases": ["12 units"]},
    },
    {
        "id": "warranty",
        "question": "How long does Globex Sensors warrant each lidar unit?",
        "answer": "24 months",
        "reference_sources": ["warranty-terms.txt"],
        "expected_entities": ["Globex Sensors", "lidar unit"],
    },
    {
        "id": "engineer",
        "question": "Who approved the lidar unit specification at Acme Robotics?",
        "answer": "Alice Chen",
        "reference_sources": ["supply-agreement.txt"],
        "expected_entities": ["Alice Chen", "Acme Robotics"],
    },
    {
        "id": "ko-carrier",
        "question": "케스트럴 창고까지 운송은 누가 맡나요?",
        "answer": "엄브라 프레이트",
        "reference_sources": ["contract-summary.ko.txt"],
        "expected_entities": ["엄브라 프레이트", "케스트럴 창고"],
    },
    {
        "id": "unanswerable",
        "question": "What did Acme Robotics pay for the second warehouse?",
        "metadata": {"answerable": False},
    },
]

# Questions the Ask page suggests for this corpus.
SAMPLE_QUESTIONS: list[str] = [
    "Who carries the lidar units to Kestrel Depot, and how fast?",
    "What happened to the units Orion Labs rejected?",
    "How are Acme Robotics and Globex Sensors related?",
    "지연 납품 위약금은 얼마인가요?",
]
