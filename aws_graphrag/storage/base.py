from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    Relationship,
    TextUnit,
)

logger = get_logger(__name__)


class IndexingStats(BaseModel):
    total_items: int = Field(default=0)
    successful_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

    @property
    def success_rate(self) -> float:
        return self.successful_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.failed_items / self.total_items if self.total_items > 0 else 0.0

    def add_success(self, count: int = 1) -> None:
        self.successful_items += count

    def add_error(self, error_message: str, count: int = 1) -> None:
        self.failed_items += count
        if error_message not in self.errors:
            self.errors.append(error_message)

    def merge(self, other: "IndexingStats") -> None:
        self.total_items += other.total_items
        self.successful_items += other.successful_items
        self.failed_items += other.failed_items
        self.processing_time += other.processing_time

        unique_errors = set(self.errors)
        unique_errors.update(other.errors)
        self.errors = list(unique_errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "processing_time": self.processing_time,
            "error_count": len(self.errors),
            "sample_errors": self.errors[:5],
        }


class BaseIndexer(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stats = IndexingStats()

    @abstractmethod
    def clear(self, suffixes: list[str]) -> bool:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @classmethod
    def get_suffix(cls, item: Any) -> str:
        if not hasattr(item, "attributes") or not isinstance(item.attributes, dict):
            return Constants.DEFAULT_SUFFIX.value

        index_value = item.attributes.get(Constants.INDEX.value)
        if isinstance(index_value, str) and index_value:
            return index_value
        if isinstance(index_value, (list | tuple)) and index_value:
            return str(index_value[0])

        return Constants.DEFAULT_SUFFIX.value

    def _get_name(
        self, base: str, suffix: str | None, add_timestamp: bool = False
    ) -> str:
        final_suffix = suffix or Constants.DEFAULT_SUFFIX.value

        if self.config.indexing.additional_suffix:
            final_suffix = f"{final_suffix}-{self.config.indexing.additional_suffix}"

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"{base}-{final_suffix}-{timestamp}"

        return f"{base}-{final_suffix}"

    def _group_items_by_suffix(self, items: list[Any]) -> dict[str, list[Any]]:
        grouped = defaultdict(list)
        for item in items:
            grouped[self.get_suffix(item)].append(item)

        if len(grouped) > 1:
            logger.info(
                f"Grouped {len(items)} items into {len(grouped)} different suffixes"
            )

        return dict(grouped)


class GraphIndexer(BaseIndexer):
    @abstractmethod
    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        pass

    @abstractmethod
    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        pass

    @abstractmethod
    def index_communities(self, communities: list[Community]) -> IndexingStats:
        pass

    @abstractmethod
    def get_entity_count(self, suffixes: list[str]) -> int:
        pass


class VectorIndexer(BaseIndexer):
    @abstractmethod
    def index_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        pass

    @abstractmethod
    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        pass

    @abstractmethod
    def index_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        pass

    @abstractmethod
    def get_entity_count(self, suffixes: list[str]) -> int:
        pass
