# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import json
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.documents import Document as BaseDocument
from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T", bound=BaseModel)


class ElementContent(BaseModel):
    text: str | None = Field(
        default=None, description="Plain text content of the element"
    )
    html: str | None = Field(
        default=None, description="HTML formatted content of the element"
    )
    markdown: str | None = Field(
        default=None, description="Markdown formatted content of the element"
    )


class ElementType(str, Enum):
    CAPTION = "caption"
    CHART = "chart"
    EQUATION = "equation"
    FIGURE = "figure"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    HEADER = "header"
    HEADING1 = "heading1"
    INDEX = "index"
    LIST = "list"
    PARAGRAPH = "paragraph"
    TABLE = "table"


class DocumentElement(BaseModel):
    id: int = Field(description="Unique identifier for the document element")
    page: int = Field(
        ge=1, description="Page number where the element is located (1-based)"
    )
    category: ElementType = Field(description="Type/category of the document element")
    content: ElementContent | None = Field(
        default=None, description="Content of the element in various formats"
    )
    coordinates: list[float] | None = Field(
        default=None,
        min_length=4,
        description="Bounding box coordinates of the element [x1, y1, x2, y2, ...]",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for element detection (0.0 to 1.0)",
    )
    base64_encoding: str | None = Field(
        default=None,
        description="Base64 encoded representation of the element (e.g., for images)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the element",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> ElementContent | None:
        if v is None or isinstance(v, ElementContent):
            return v
        if isinstance(v, dict):
            return ElementContent(**v)
        raise ValueError("Content must be an ElementContent instance, a dict, or None")

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v: list[Any] | None) -> list[float] | None:
        if not v:
            return None

        if not isinstance(v, list):
            raise ValueError("Coordinates must be a list")

        first_item = v[0]
        if isinstance(first_item, dict):
            normalized_coords = cls._normalize_from_dicts(v)
        elif isinstance(first_item, (int | float)):
            normalized_coords = cls._normalize_from_numbers(v)
        else:
            raise ValueError(
                "Coordinates must be a list of dicts with 'x'/'y' keys or a list of numbers"
            )

        if len(normalized_coords) % 2 != 0:
            raise ValueError("Coordinates must contain an even number of values")

        return normalized_coords

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v: Any) -> dict[str, Any]:
        return v or {}

    @staticmethod
    def _normalize_from_dicts(coords: list[Any]) -> list[float]:
        normalized: list[float] = []
        try:
            for coord in coords:
                if not isinstance(coord, dict) or "x" not in coord or "y" not in coord:
                    raise ValueError("Coordinate dict must have 'x' and 'y' keys")
                normalized.extend([float(coord["x"]), float(coord["y"])])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid coordinate dictionary values: {e}") from e
        return normalized

    @staticmethod
    def _normalize_from_numbers(coords: list[Any]) -> list[float]:
        try:
            return [float(c) for c in coords]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric coordinate value: {e}") from e

    def __str__(self) -> str:
        return f"Element({self.category.value}, id={self.id}, page={self.page})"


class DocumentContent(BaseModel):
    text: str | None = Field(
        default=None, description="Plain text content of the entire document"
    )
    html: str | None = Field(
        default=None, description="HTML formatted content of the entire document"
    )
    markdown: str | None = Field(
        default=None, description="Markdown formatted content of the entire document"
    )


class Page(BaseModel):
    page_number: int = Field(ge=1, description="Page number (1-based)")
    height: int | None = Field(
        default=None, gt=0, description="Height of the page in pixels"
    )
    width: int | None = Field(
        default=None, gt=0, description="Width of the page in pixels"
    )
    text_content: str | None = Field(
        default=None, description="Extracted text content from the page"
    )
    elements: list[DocumentElement] = Field(
        default_factory=list, description="List of elements found on this page"
    )
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Raw data from the document processing service"
    )

    def __str__(self) -> str:
        return f"Page(page_number={self.page_number}, elements={len(self.elements)})"


class DocStatus(str, Enum):
    """Processing lifecycle of a document in the incremental-indexing registry.

    Adapted from LightRAG's DocStatus state machine. A run advances a document
    PENDING -> PARSING -> PROCESSING -> PROCESSED, or to FAILED on error.
    """

    PENDING = "pending"
    PARSING = "parsing"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocStatusRecord(BaseModel):
    """Per-document state persisted across indexing runs (DocStatusPort).

    Stores the content hash for change detection and the ids of every graph
    artifact the document produced, so a delta run can merge or remove exactly
    the affected entities/relationships/text-units/communities.
    """

    doc_id: str = Field(description="Stable document identifier (path-normalized)")
    content_hash: str = Field(description="Content hash used for change detection")
    status: DocStatus = Field(
        default=DocStatus.PENDING, description="Current processing status"
    )
    suffix: str = Field(
        default="default",
        description="Index/label suffix the document's artifacts were written under",
    )
    file_path: str | None = Field(default=None, description="Source file path")
    content_summary: str | None = Field(
        default=None, description="Short summary/preview of the document content"
    )
    content_length: int | None = Field(
        default=None, ge=0, description="Length of the document content in characters"
    )
    entity_ids: list[str] = Field(
        default_factory=list, description="Entity ids produced by this document"
    )
    relationship_ids: list[str] = Field(
        default_factory=list, description="Relationship ids produced by this document"
    )
    text_unit_ids: list[str] = Field(
        default_factory=list, description="Text-unit ids produced by this document"
    )
    community_ids: list[str] = Field(
        default_factory=list, description="Community ids this document contributed to"
    )
    claim_ids: list[str] = Field(
        default_factory=list, description="Claim ids produced by this document"
    )
    community_report_ids: list[str] = Field(
        default_factory=list,
        description="Community-report ids this document contributed to",
    )
    error_info: str | None = Field(
        default=None, description="Error detail if status is FAILED"
    )
    created_at: str | None = Field(
        default=None, description="ISO timestamp when first registered"
    )
    updated_at: str | None = Field(
        default=None, description="ISO timestamp of the last status change"
    )


class DocumentLineage(BaseModel):
    """Per-document artifact attribution for an incremental commit.

    Lets the orchestrator record exactly which entities/relationships/text-units/
    communities a single document produced (and under which index suffix), so a
    later change/delete can remove that document's *exclusive* artifacts without
    touching ones shared with surviving documents.
    """

    doc_id: str = Field(description="Stable document id these artifacts belong to")
    suffix: str = Field(
        default="default",
        description="Index/label suffix the artifacts were written under",
    )
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    text_unit_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    community_report_ids: list[str] = Field(default_factory=list)


class DocumentDelta(BaseModel):
    """Partition of an incoming corpus relative to the persisted registry.

    Produced by ``DocStatusPort.diff``: ``changed`` documents exist with a
    different content hash, ``deleted`` were present before but are absent now.
    ``is_empty`` lets an update run short-circuit when nothing changed.
    """

    new: list[str] = Field(
        default_factory=list, description="Doc ids not previously seen"
    )
    changed: list[str] = Field(
        default_factory=list, description="Doc ids whose content hash changed"
    )
    unchanged: list[str] = Field(
        default_factory=list, description="Doc ids with an identical content hash"
    )
    deleted: list[str] = Field(
        default_factory=list, description="Doc ids removed from the corpus"
    )

    @property
    def is_empty(self) -> bool:
        """True when no documents were added, changed, or removed."""
        return not (self.new or self.changed or self.deleted)

    @property
    def to_process(self) -> list[str]:
        """Doc ids requiring (re)indexing this run: new + changed."""
        return self.new + self.changed


class Document(BaseDocument):
    document_id: str = Field(description="Unique identifier for the document")
    file_name: str = Field(description="Name of the source file")
    file_path: str = Field(description="Path to the source file")
    file_type: str = Field(
        description="Type/format of the source file (e.g., pdf, docx)"
    )
    detected_language: str = Field(
        default="Unknown", description="Detected language of the document content"
    )
    total_pages: int = Field(ge=0, description="Total number of pages in the document")
    pages: list[Page] = Field(
        default_factory=list, description="List of pages in the document"
    )
    elements: list[DocumentElement] = Field(
        default_factory=list, description="List of all elements across all pages"
    )
    content: DocumentContent | None = Field(
        default=None, description="Aggregated content of the entire document"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the document"
    )
    error_info: str | None = Field(
        default=None, description="Error information if document processing failed"
    )

    @field_validator("document_id", "file_name", mode="before")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v.strip()

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, v: Any) -> str:
        return str(v)

    @field_validator("pages", mode="before")
    @classmethod
    def validate_pages(cls, v: list[Any] | None) -> list[Page]:
        return cls._validate_model_list(v, Page)

    @classmethod
    def _validate_model_list(cls, v: list[Any] | None, model_type: type[T]) -> list[T]:
        if v is None:
            return []

        validated_list = []
        for item in v:
            if isinstance(item, model_type):
                validated_list.append(item)
            elif isinstance(item, dict):
                validated_list.append(model_type(**item))
            else:
                raise TypeError(
                    f"Items must be of type '{model_type.__name__}' or 'dict'"
                )
        return validated_list

    @field_validator("elements", mode="before")
    @classmethod
    def validate_elements(cls, v: list[Any] | None) -> list[DocumentElement]:
        return cls._validate_model_list(v, DocumentElement)

    @model_validator(mode="after")
    def set_content_if_none(self) -> "Document":
        if self.content is None:
            self.content = self._generate_aggregated_content()

        return self

    def _generate_aggregated_content(self) -> DocumentContent:
        text_parts, html_parts, markdown_parts = self._extract_content_from_elements(
            self.elements
        )

        if not any((text_parts, html_parts, markdown_parts)):
            page_elements = [el for page in self.pages for el in page.elements]
            text_from_els, html_parts, markdown_parts = (
                self._extract_content_from_elements(page_elements)
            )

            text_from_pages = [p.text_content for p in self.pages if p.text_content]
            text_parts = text_from_pages + text_from_els

        return DocumentContent(
            text="\n\n".join(text_parts) if text_parts else None,
            html="\n".join(html_parts) if html_parts else None,
            markdown="\n\n".join(markdown_parts) if markdown_parts else None,
        )

    @staticmethod
    def _extract_content_from_elements(
        elements: list[DocumentElement],
    ) -> tuple[list[str], list[str], list[str]]:
        text_parts, html_parts, markdown_parts = [], [], []
        for element in elements:
            if element.content:
                if element.content.text:
                    text_parts.append(element.content.text)
                if element.content.html:
                    html_parts.append(element.content.html)
                if element.content.markdown:
                    markdown_parts.append(element.content.markdown)
        return text_parts, html_parts, markdown_parts

    @classmethod
    def from_json_file(cls, input_path: str | Path) -> "Document":
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found at '{path}'")

        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_content"] = data.get("content", {}).get("text", "")
        return cls.model_validate(data)

    def to_json_file(self, output_path: str | Path | None = None, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        json_data = self.model_dump_json(**kwargs)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_data, encoding="utf-8")

        return json_data

    @property
    def is_error(self) -> bool:
        return self.error_info is not None

    def __str__(self) -> str:
        status = "ERROR" if self.is_error else "OK"
        return f"Document(id={self.document_id}, file='{self.file_name}', pages={self.total_pages}, status={status})"
