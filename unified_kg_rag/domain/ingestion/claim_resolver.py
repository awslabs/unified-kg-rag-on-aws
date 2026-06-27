# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from unified_kg_rag.domain.ingestion.base_resolver import BaseResolver, FuzzyMatcher
from unified_kg_rag.domain.models import Claim, Config, Entity
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import normalize_name

logger = get_logger(__name__)


def resolve_entity_reference_task(
    entity_reference: str,
    normalized_name_to_id: dict[str, str],
    normalized_name_to_original_name: dict[str, str],
    fuzzy_matcher: FuzzyMatcher,
) -> str | None:
    if not entity_reference or not entity_reference.strip():
        return None

    normalized_reference = normalize_name(entity_reference)
    if normalized_reference in normalized_name_to_id:
        return normalized_name_to_original_name[normalized_reference]

    match_result = fuzzy_matcher.find_best_match(entity_reference)
    if match_result:
        matched_name, _ = match_result
        if matched_name in fuzzy_matcher.candidates:
            return matched_name

    return None


def resolve_single_claim_task(
    claim: Claim,
    normalized_name_to_id: dict[str, str],
    normalized_name_to_original_name: dict[str, str],
    fuzzy_matcher: FuzzyMatcher,
) -> Claim | None:
    resolved_subject_name = resolve_entity_reference_task(
        claim.subject_name,
        normalized_name_to_id,
        normalized_name_to_original_name,
        fuzzy_matcher,
    )
    resolved_object_name = resolve_entity_reference_task(
        claim.object_name,
        normalized_name_to_id,
        normalized_name_to_original_name,
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

    subject_id = normalized_name_to_id.get(normalize_name(resolved_subject_name))
    if subject_id is None:
        return None

    if resolved_object_name is not None:
        object_id = normalized_name_to_id.get(normalize_name(resolved_object_name))
        object_name = resolved_object_name
    else:
        # Object is a literal value: preserve the original text, no entity id.
        object_id = None
        object_name = claim.object_name

    return claim.model_copy(
        update={
            "subject_id": subject_id,
            "subject_name": resolved_subject_name,
            "object_id": object_id,
            "object_name": object_name,
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
    def __init__(
        self,
        config: Config,
        max_workers: int | None = None,
        use_process_pool: bool = True,
    ) -> None:
        super().__init__(
            config, max_workers=max_workers, use_process_pool=use_process_pool
        )

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

        normalized_name_to_id: dict[str, str] = {}
        normalized_name_to_original_name: dict[str, str] = {}

        for entity in entities:
            normalized_name = normalize_name(entity.name)
            if normalized_name:
                normalized_name_to_id[normalized_name] = entity.id
                normalized_name_to_original_name[normalized_name] = entity.name

        logger.info(
            "Created normalized entity map with %s entries.", len(normalized_name_to_id)
        )

        entity_names = list(normalized_name_to_original_name.values())
        fuzzy_matcher = self._create_fuzzy_matcher(candidate_texts=entity_names)

        resolved_claims = self._resolve_all_claims(
            claims,
            normalized_name_to_id,
            normalized_name_to_original_name,
            fuzzy_matcher,
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
                merged_claims.append(merged_claim)

        stats.resolved_claims = len(merged_claims)
        stats.unresolved_claims = stats.original_claims - len(resolved_claims)
        stats.processing_time = time.time() - start_time

        self._log_completion_summary(stats)
        return merged_claims, stats

    def _resolve_all_claims(
        self,
        claims: list[Claim],
        normalized_name_to_id: dict[str, str],
        normalized_name_to_original_name: dict[str, str],
        fuzzy_matcher: FuzzyMatcher,
    ) -> list[Claim]:
        if not claims:
            return []

        resolved_claims = []
        executor_class = (
            ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            future_to_claim = {
                executor.submit(
                    resolve_single_claim_task,
                    claim,
                    normalized_name_to_id,
                    normalized_name_to_original_name,
                    fuzzy_matcher,
                ): claim
                for claim in claims
            }

            unresolved_count = 0
            for future in tqdm(
                as_completed(future_to_claim),
                total=len(claims),
                desc="Resolving Claims",
                disable=not self.show_progress,
            ):
                original_claim = future_to_claim[future]
                try:
                    resolved_claim = future.result()
                    if resolved_claim:
                        resolved_claims.append(resolved_claim)
                    else:
                        unresolved_count += 1
                except Exception as e:
                    logger.error("Error resolving claim '%s': %s", original_claim.id, e)
                    unresolved_count += 1

            if unresolved_count > 0:
                logger.warning(
                    "Failed to resolve %s out of %s claims",
                    unresolved_count,
                    len(claims),
                )

        return resolved_claims

    @staticmethod
    def _group_similar_claims(claims: list[Claim]) -> list[list[Claim]]:
        if not claims:
            return []

        groups_dict = defaultdict(list)
        for claim in claims:
            key = (claim.subject_id, claim.object_id, claim.type)
            groups_dict[key].append(claim)

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
