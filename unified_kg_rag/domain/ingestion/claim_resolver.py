# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from unified_kg_rag.domain.ingestion.base_resolver import BaseResolver, FuzzyMatcher
from unified_kg_rag.domain.models import Claim, Entity
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import entity_key, generate_stable_id

# See graph_resolver: log progress every N items instead of pulling tqdm
# (a terminal-UI dependency) into the technology-agnostic domain layer.
_RESOLVE_PROGRESS_EVERY = 2000

logger = get_logger(__name__)

# Placeholders the claim prompt (or the model) uses for an absent date.
_UNKNOWN_DATE_VALUES = frozenset({"", "unknown", "none", "n/a", "na", "null"})


def claim_date_key(value: str | None) -> str:
    """Normalize a claim date for identity ("" when absent or a placeholder)."""
    key = " ".join((value or "").split()).casefold()
    return "" if key in _UNKNOWN_DATE_VALUES else key


def claim_identity_suffix(start_date: str | None, end_date: str | None) -> str:
    """Date component of a claim's identity ("" when neither date is known).

    Two claims with the same subject, object and type but different dates
    (e.g. two delivery obligations due on different days) are distinct facts;
    keying on the dates keeps them apart instead of collapsing them and keeping
    only one date. Date-less claims keep a date-free identity, so restatements
    of the same undated claim still merge.
    """
    start, end = claim_date_key(start_date), claim_date_key(end_date)
    return f":{start}:{end}" if start or end else ""


def resolved_claim_id(claim: Claim) -> str:
    """Stable id of a claim whose subject (and object) point at entities.

    Keyed on what makes two claims the same fact: the subject entity id, the
    object entity id (or, for a literal object such as an amount or a date,
    its ``entity_key``), the type and the known dates. The extractor's id
    hashes the names as written, so "Acme Inc" and "Acme" gave one claim two
    ids; keyed on the entities, the claim gets the same id whichever spelling
    or document came first, in a full build and across incremental runs.
    """
    object_key = claim.object_id or f"literal:{entity_key(claim.object_name)}"
    return generate_stable_id(
        f"claim:{claim.subject_id}:{object_key}:{(claim.type or '').casefold()}"
        f"{claim_identity_suffix(claim.start_date, claim.end_date)}"
    )


def resolve_entity_reference(
    entity_reference: str,
    names_by_key: dict[str, str],
    fuzzy_matcher: FuzzyMatcher,
) -> str | None:
    """Return the name of the entity ``entity_reference`` denotes, else None.

    Tried in order: the same ``entity_key``, then the guarded fuzzy match of
    :meth:`FuzzyMatcher.find_all_matches` (spelling and legal-form variants
    share a ``compact_name_key`` and score 1.0; a candidate whose identifier
    tokens or head character differ is never returned), the best score
    winning. The guards matter as much here as in entity grouping: "PO 1003"
    must not become "PO 1002", and a literal value must stay a literal value.
    """
    key = entity_key(entity_reference)
    if not key:
        return None
    if key in names_by_key:
        return names_by_key[key]
    matches = fuzzy_matcher.find_all_matches(entity_reference)
    if not matches:
        return None
    # Equal scores break on the name, so the choice does not depend on the
    # order the LSH returns candidates in.
    return min(matches, key=lambda match: (-match[1], match[0]))[0]


def resolve_single_claim(
    claim: Claim,
    names_by_key: dict[str, str],
    ids_by_key: dict[str, str],
    fuzzy_matcher: FuzzyMatcher,
) -> Claim | None:
    """Point a claim's subject (required) and object (optional) at entities.

    ``names_by_key`` / ``ids_by_key`` map an ``entity_key`` to the entity's
    name and id; ``fuzzy_matcher`` holds the entity names as candidates.
    """
    resolved_subject_name = resolve_entity_reference(
        claim.subject_name,
        names_by_key,
        fuzzy_matcher,
    )

    # A claim is anchored to its SUBJECT entity. The object may legitimately be a
    # literal value (a date, amount, status, ...) rather than an extracted
    # entity — claim types like PERFORMANCE/ATTRIBUTE/FINANCIAL/TEMPORAL routinely
    # have non-entity objects. Dropping such claims discards most of the
    # extraction. So we only require the subject to resolve; if the object does
    # not resolve to an entity we keep the claim with object_id=None and the
    # original object text preserved as a literal value.
    if resolved_subject_name is None:
        logger.debug("Dropping claim: subject not an entity: '%s'", claim.subject_name)
        return None

    object_name = resolve_entity_reference(
        claim.object_name, names_by_key, fuzzy_matcher
    )
    return claim.model_copy(
        update={
            "subject_id": ids_by_key[entity_key(resolved_subject_name)],
            "subject_name": resolved_subject_name,
            "object_id": ids_by_key[entity_key(object_name)] if object_name else None,
            "object_name": object_name or claim.object_name,
        }
    )


class ClaimResolutionStats(BaseModel):
    original_claims: int = 0
    resolved_claims: int = 0
    unresolved_claims: int = 0
    claim_groups_created: int = 0
    processing_time: float = 0.0

    @property
    def reduction_rate(self) -> float:
        if self.original_claims == 0:
            return 0.0
        return (
            (self.original_claims - self.resolved_claims) / self.original_claims
        ) * 100


class ClaimResolver(BaseResolver):
    def resolve(
        self, claims: list[Claim], entities: list[Entity], *args: Any, **kwargs: Any
    ) -> tuple[list[Claim], ClaimResolutionStats]:
        logger.info(
            "Starting claim resolution for %s claims against %s entities",
            len(claims),
            len(entities),
        )
        return self._resolve_claims(claims, entities)

    def _resolve_claims(
        self,
        claims: list[Claim],
        entities: list[Entity],
    ) -> tuple[list[Claim], ClaimResolutionStats]:
        start_time = time.time()
        stats = ClaimResolutionStats(original_claims=len(claims))

        method = self.config.processing.resolution_method.value
        logger.info("Using resolution method: '%s'", method)

        names_by_key: dict[str, str] = {}
        ids_by_key: dict[str, str] = {}
        for entity in entities:
            key = entity_key(entity.name)
            if key:
                names_by_key[key] = entity.name
                ids_by_key[key] = entity.id

        logger.info("Created normalized entity map with %s entries.", len(ids_by_key))

        # The guarded similarity step stays on whatever
        # processing.fuzzy_entity_resolution says: that flag decides which
        # entities merge, while a claim only points at an existing entity
        # (bounded by processing.similarity_threshold).
        fuzzy_matcher = self._create_fuzzy_matcher(
            candidate_texts=list(names_by_key.values())
        )
        resolved_claims = self._resolve_all_claims(
            claims, names_by_key, ids_by_key, fuzzy_matcher
        )

        claim_groups = self._group_similar_claims(resolved_claims)
        stats.claim_groups_created = len(claim_groups)
        logger.info(
            "Grouped %s claims into %s groups", len(resolved_claims), len(claim_groups)
        )

        merged_claims = []
        for group in claim_groups:
            if group:
                merged_claim = self._merge_claims(group)
                claim_id = resolved_claim_id(merged_claim)
                merged_claims.append(
                    merged_claim.model_copy(
                        update={"id": claim_id, "short_id": claim_id[:8]}
                    )
                )

        stats.resolved_claims = len(merged_claims)
        stats.unresolved_claims = stats.original_claims - len(resolved_claims)
        stats.processing_time = time.time() - start_time

        self._log_completion_summary(stats)
        return merged_claims, stats

    def _resolve_all_claims(
        self,
        claims: list[Claim],
        names_by_key: dict[str, str],
        ids_by_key: dict[str, str],
        fuzzy_matcher: FuzzyMatcher,
    ) -> list[Claim]:
        """Resolve every claim in input order, in this process.

        A lookup is a dict hit or one guarded fuzzy query, so a worker pool
        costs more than it saves (each task would ship the matcher and the
        entity maps). Input order also fixes which claim of a group is the
        merge's primary.
        """
        resolved_claims = []
        total = len(claims)
        for done, claim in enumerate(claims, start=1):
            if self.show_progress and done % _RESOLVE_PROGRESS_EVERY == 0:
                logger.info("  ...resolved %s/%s claims", done, total)
            resolved_claim = resolve_single_claim(
                claim, names_by_key, ids_by_key, fuzzy_matcher
            )
            if resolved_claim:
                resolved_claims.append(resolved_claim)

        unresolved_count = total - len(resolved_claims)
        if unresolved_count > 0:
            logger.warning(
                "Failed to resolve %s out of %s claims", unresolved_count, total
            )
        return resolved_claims

    @staticmethod
    def _group_similar_claims(claims: list[Claim]) -> list[list[Claim]]:
        if not claims:
            return []

        # Grouped by the resolved id: a literal object (object_id None: an
        # amount, date, status) is keyed by its value, so "$100" and "$200"
        # claims stay distinct; dates are part of the identity for the same
        # reason.
        groups_dict: defaultdict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            groups_dict[resolved_claim_id(claim)].append(claim)

        return list(groups_dict.values())

    def _merge_claims(self, claims: list[Claim]) -> Claim:
        if len(claims) == 1:
            return claims[0]

        primary_claim = claims[0]
        return Claim(
            id=primary_claim.id,
            short_id=primary_claim.short_id,
            subject_id=primary_claim.subject_id,
            subject_name=primary_claim.subject_name,
            object_id=primary_claim.object_id,
            object_name=primary_claim.object_name,
            type=primary_claim.type,
            status=self._get_most_common_value([c.status for c in claims if c.status]),
            start_date=primary_claim.start_date,
            end_date=primary_claim.end_date,
            description=self._merge_descriptions(
                [c.description for c in claims if c.description]
            ),
            description_embedding=primary_claim.description_embedding,
            text_unit_ids=self._merge_lists(
                [c.text_unit_ids for c in claims if c.text_unit_ids]
            ),
            source_text=self._merge_descriptions(
                [c.source_text for c in claims if c.source_text]
            ),
            attributes=self._merge_attributes(
                [c.attributes for c in claims if c.attributes]
            ),
            # Leave created_at None when unknown (see graph_resolver) so the
            # merge stays a pure function of its inputs.
            created_at=min(
                (c.created_at for c in claims if c.created_at),
                default=None,
            ),
            updated_at=datetime.now(),
        )

    @staticmethod
    def _log_completion_summary(stats: ClaimResolutionStats) -> None:
        logger.info(
            "Claim resolution completed - Processing time: %.2fs",
            stats.processing_time,
        )
        logger.info(
            "Results: %s -> %s claims (%.2f%% reduction, %s groups)",
            stats.original_claims,
            stats.resolved_claims,
            stats.reduction_rate,
            stats.claim_groups_created,
        )
