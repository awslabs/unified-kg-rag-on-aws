# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The data references of a community report's findings (pure).

The report prompt numbers the entities and relationships it packs and asks
for MS GraphRAG's grounding syntax at the end of each finding:
``[Data: Entities (1, 3); Relationships (2, +more)]``. The numbers mean
something only beside the report's own input, so they are taken out of the
prose (which is embedded and searched) and kept, resolved to record ids, on
the finding. The syntax is ASCII in every report language; a model writing
CJK text may still use full-width brackets, parentheses, colons or commas,
which are read the same.
"""

import re

__all__ = ["DATASET_ENTITIES", "DATASET_RELATIONSHIPS", "split_data_references"]

DATASET_ENTITIES = "entities"
DATASET_RELATIONSHIPS = "relationships"

_REFERENCE = re.compile(
    r"\s*[\[［]\s*Data\s*[:：](?P<body>[^\]］]*)[\]］]", re.IGNORECASE
)
_DATASET = re.compile(r"(?P<name>[A-Za-z]+)\s*[(（](?P<ids>[^)）]*)[)）]")
_NUMBER = re.compile(r"\d+")


def _dataset(name: str) -> str | None:
    name = name.lower()
    if name.startswith("entit"):
        return DATASET_ENTITIES
    if name.startswith("relation"):
        return DATASET_RELATIONSHIPS
    return None


def split_data_references(text: str) -> tuple[str, dict[str, list[int]]]:
    """``text`` without its ``[Data: ...]`` references, and the record
    numbers they cite by dataset (:data:`DATASET_ENTITIES`,
    :data:`DATASET_RELATIONSHIPS`; others, such as ``Claims``, are
    dropped), each in first-cited order without repeats. ``+more`` is no
    number."""
    cited: dict[str, list[int]] = {}

    def take(match: re.Match[str]) -> str:
        for dataset in _DATASET.finditer(match.group("body")):
            name = _dataset(dataset.group("name"))
            if name is None:
                continue
            numbers = cited.setdefault(name, [])
            for number in _NUMBER.findall(dataset.group("ids")):
                if int(number) not in numbers:
                    numbers.append(int(number))
        return ""

    return _REFERENCE.sub(take, text).strip(), cited
