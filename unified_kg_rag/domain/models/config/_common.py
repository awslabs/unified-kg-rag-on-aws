# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages, constants and shared enums of the configuration tree."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class _AssignmentChecked(BaseModel):
    """A model whose assignment is validated against its field and the
    model's own checks, the old value kept when they refuse the new one:
    the base of ``Config`` and of every section (:class:`ConfigSection`)."""

    model_config = ConfigDict(validate_assignment=True)

    def __setattr__(self, name: str, value: Any) -> None:
        before = self.__dict__.get(name)
        try:
            super().__setattr__(name, value)
        except ValidationError:
            # A model check (a model validator) that refuses the new value
            # runs after pydantic has stored it: put the old one back.
            if name in type(self).model_fields:
                self.__dict__[name] = before
            raise


class ConfigSection(_AssignmentChecked):
    """The base of every configuration section: an assignment is validated
    against its field and the section's own checks
    (``config.search.max_concurrency = 0`` raises as loading it would, and leaves the
    old value), and an unknown key is refused. The checks across sections
    (``Config``'s own validators) run on a whole config: change one with
    :meth:`Config.updated`."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class PipelineStageType(Enum):
    """Ingestion stages, declared in pipeline order.

    The declaration order is the canonical stage order: the pipeline runs the
    stages, the cache keys fold in upstream inputs, and the run metadata
    sorts stage results in this order.
    """

    DOCUMENT_PARSING = "document_parsing"
    DOCUMENT_LOADING = "document_loading"
    TEXT_CHUNKING = "text_chunking"
    TRANSLATION = "translation"
    GRAPH_EXTRACTION = "graph_extraction"
    GLEANING = "gleaning"
    GRAPH_RESOLUTION = "graph_resolution"
    CLAIM_EXTRACTION = "claim_extraction"
    CLAIM_RESOLUTION = "claim_resolution"
    GRAPH_ANALYSIS = "graph_analysis"
    COMMUNITY_DETECTION = "community_detection"
    INDEXING = "indexing"
    # RAPTOR's summary tree over the stored text units (raptor methodology).
    SUMMARY_TREE = "summary_tree"


class ChunkingStrategy(str, Enum):
    SIMPLE = "simple"
    INTELLIGENT = "intelligent"


class ChunkLengthUnit(str, Enum):
    """What the chunk size settings count."""

    CHARS = "chars"
    # The local script-aware estimate (adapters.llm.token_estimate), never a
    # model's tokenizer: chunk boundaries stay deterministic and offline.
    TOKENS = "tokens"


class Constants(str, Enum):
    ATTRIBUTE_PREFIX = "attr"
    # Type of a relationship extracted without one (its Neptune edge label).
    DEFAULT_RELATIONSHIP_TYPE = "RELATED_TO"
    DEFAULT_SUFFIX = "default"
    FILTERS = "filters"
    INDEX = "index"


class ResolutionMethod(str, Enum):
    MINHASH = "minhash"
    SEQUENCE_MATCHER = "sequence_matcher"


class S3EncryptionType(str, Enum):
    """Server-side encryption header sent on cache uploads.

    ``BUCKET_DEFAULT`` sends no SSE header, so S3 applies the bucket's default
    encryption (SSE-S3 at minimum, or the bucket's SSE-KMS CMK). ``AES256`` /
    ``aws:kms`` force a per-object header that OVERRIDES the bucket default.
    """

    BUCKET_DEFAULT = "BUCKET_DEFAULT"
    AES256 = "AES256"
    KMS = "aws:kms"
