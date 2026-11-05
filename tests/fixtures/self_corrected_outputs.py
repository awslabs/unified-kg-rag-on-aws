# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model answers that correct themselves, as a real model run returned them.

The model writes a whole answer, then a line such as "Correction: ... Here is
the corrected, final output:", then the whole answer again. The structure
(two complete answers, the correction line between them, a malformed tag in
the first) is the one a graph extraction and a keyword extraction returned;
the content is synthetic.
"""

from __future__ import annotations

import json


def _entity(name: str, kind: str, description: str, source: str) -> str:
    return (
        "<entity>\n"
        f"<name>{name}</name>\n"
        f"<type>{kind}</type>\n"
        f"<description>{description}</description>\n"
        "<confidence>9</confidence>\n"
        f"<source_text>{source}</source_text>\n"
        "</entity>\n"
    )


def _relationship(
    source: str, target: str, kind: str, evidence: str, source_close: str = "source"
) -> str:
    return (
        "<relationship>\n"
        f"<source>{source}</{source_close}>\n"
        f"<target>{target}</target>\n"
        f"<type>{kind}</type>\n"
        f"<description>{source} {kind.lower().replace('_', ' ')} {target}.</description>\n"
        "<strength>8</strength>\n"
        f"<source_text>{evidence}</source_text>\n"
        "</relationship>\n"
    )


_ENTITIES = (
    "<entities>\n"
    + _entity("Vendor Corp", "ORGANIZATION", "The supplier.", "Vendor Corp")
    + _entity("Harbor City", "LOCATION", "Where Vendor Corp is based.", "Harbor City")
    + _entity("Buyer Ltd", "ORGANIZATION", "The buyer.", "Buyer Ltd")
    + _entity("Depot Inc", "ORGANIZATION", "Stores goods for Buyer Ltd.", "Depot Inc")
    + "</entities>\n"
)

_FIRST_RELATIONSHIPS = (
    "<relationships>\n"
    + _relationship("Vendor Corp", "Buyer Ltd", "SUPPLIES", "signed a supply agreement")
    # The malformed tag the model then corrects: <source> closed as </source_text>.
    + _relationship(
        "Vendor Corp", "Harbor City", "BASED_IN", "based in Harbor City", "source_text"
    )
    + "</relationships>\n"
)

_FINAL_RELATIONSHIPS = (
    "<relationships>\n"
    + _relationship("Vendor Corp", "Buyer Ltd", "SUPPLIES", "signed a supply agreement")
    + _relationship("Vendor Corp", "Harbor City", "BASED_IN", "based in Harbor City")
    + _relationship("Depot Inc", "Buyer Ltd", "STORES_FOR", "stores the goods")
    + "</relationships>\n"
)

CORRECTION_LINE = (
    "Correction: the relationship above contains a malformed tag. Here is the "
    "corrected, final output:"
)

# Two complete answers; the second has one more relationship than the first.
EXTRACTION_SELF_CORRECTED = (
    f"{_ENTITIES}\n{_FIRST_RELATIONSHIPS}\n{CORRECTION_LINE}\n\n"
    f"{_ENTITIES}\n{_FINAL_RELATIONSHIPS}"
)
FINAL_RELATIONSHIP_TYPES = ["SUPPLIES", "BASED_IN", "STORES_FOR"]

# The correction cut off in its relationships (no stop reason reported).
EXTRACTION_CORRECTION_CUT = (
    f"{_ENTITIES}\n{_FINAL_RELATIONSHIPS}\n{CORRECTION_LINE}\n\n"
    f"{_ENTITIES}\n<relationships>\n<relationship>\n<source>Vendor Corp</source>\n"
    "<target>Buy"
)

# A complete answer (every section closed) whose entities name no entity:
# malformed output, not output cut off.
EXTRACTION_UNNAMED = (
    "<entities>\n<entity>\n<entity_name>Vendor Corp</entity_name>\n"
    "<type>ORGANIZATION</type>\n</entity>\n</entities>\n"
    "<relationships>\n</relationships>\n"
)

# One answer cut off before its first entity's name arrived: cut-off output.
EXTRACTION_TRUNCATED = "<entities>\n<entity>\n<type>ORGANIZATION</type>\n<na"


def _claims(*claims: tuple[str, str, str]) -> str:
    return (
        "<claims>\n"
        + "".join(
            "<claim>\n"
            f"<subject>{subject}</subject>\n"
            f"<object>{obj}</object>\n"
            f"<claim_type>{kind}</claim_type>\n"
            "<status>TRUE</status>\n"
            f"<description>{subject} {kind.lower()} {obj}.</description>\n"
            "<source_text>signed a supply agreement</source_text>\n"
            "</claim>\n"
            for subject, obj, kind in claims
        )
        + "</claims>\n"
    )


CLAIMS_SELF_CORRECTED = (
    _claims(("Vendor Corp", "Buyer Ltd", "OBLIGATION"))
    + f"\n{CORRECTION_LINE}\n\n"
    + _claims(
        ("Vendor Corp", "Buyer Ltd", "OBLIGATION"),
        ("Buyer Ltd", "Vendor Corp", "PAYMENT"),
    )
)


def _refinement(*names: str) -> str:
    issues = "".join(
        "<issue>\n<issue_type>MISSING_ENTITY</issue_type>\n"
        f"<details>\n<name>{name}</name>\n<type>ORGANIZATION</type>\n"
        f"<description>{name} in the agreement.</description>\n</details>\n"
        f"<text_evidence>{name}</text_evidence>\n</issue>\n"
        for name in names
    )
    return (
        "<refinement_plan>\n<quality_scores>\n"
        "<completeness_score>0.8</completeness_score>\n"
        "<accuracy_score>0.9</accuracy_score>\n</quality_scores>\n"
        f"<identified_issues>\n{issues}</identified_issues>\n</refinement_plan>\n"
    )


GLEANING_SELF_CORRECTED = (
    _refinement("Depot Inc")
    + f"\n{CORRECTION_LINE}\n\n"
    + _refinement("Depot Inc", "Harbor City")
)


def _report(name: str) -> str:
    return (
        f"<community_name>{name}</community_name>\n"
        "<summary>Vendor Corp supplies Buyer Ltd.</summary>\n"
        "<rating>5.0</rating>\n"
        "<rating_explanation>A small trading group.</rating_explanation>\n"
        "<findings>\n<finding>\n<summary>Vendor Corp is central</summary>\n"
        "<explanation>It supplies Buyer Ltd.</explanation>\n</finding>\n"
        "</findings>\n"
    )


REPORT = _report("Supply partners")
REPORT_SELF_CORRECTED = (
    _report("Supply partner") + f"\n{CORRECTION_LINE}\n\n" + _report("Supply partners")
)

_FIRST_KEYWORDS = {
    "high_level_keywords": ["supply agreement"],
    "low_level_keywords": ["Vendor Corp"],
}
FINAL_KEYWORDS = {
    "high_level_keywords": ["supply agreement", "storage"],
    "low_level_keywords": ["Vendor Corp", "Depot Inc"],
}
# One object per line, a blank line and the correction line between them, as
# the keyword extraction answered ("Extra data: line 3 column 1").
KEYWORDS_SELF_CORRECTED = (
    f"{json.dumps(_FIRST_KEYWORDS)}\n\n"
    "Correction: the low-level keywords above miss an entity. Here is the "
    f"corrected, final output:\n\n{json.dumps(FINAL_KEYWORDS)}"
)
