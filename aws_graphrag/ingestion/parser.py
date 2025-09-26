from abc import ABC, abstractmethod
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from pydantic import BaseModel, Field

from aws_graphrag.core import DataProcessingError, get_logger
from aws_graphrag.models import Config, Document
from aws_graphrag.utils import convert_langchain_to_document

logger = get_logger(__name__)


class ParsingStats(BaseModel):
    num_total_files: int = Field(default=0)
    num_successful_files: int = Field(default=0)
    num_failed_files: int = Field(default=0)
    total_processing_time: float = Field(default=0.0)

    @property
    def success_rate(self) -> float:
        if self.num_total_files == 0:
            return 0.0
        return (self.num_successful_files / self.num_total_files) * 100


class BaseParser(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stats = ParsingStats()

    @abstractmethod
    def parse_file(
        self, file_path: str | Path, index_value: str | None = None
    ) -> Document:
        pass

    def parse_files(
        self, file_paths: list[str | Path], index_value: str | None = None
    ) -> list[Document]:
        documents = []
        self.stats = ParsingStats(num_total_files=len(file_paths))

        for file_path in file_paths:
            try:
                doc = self.parse_file(file_path, index_value)
                documents.append(doc)
                self.stats.num_successful_files += 1
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                self.stats.num_failed_files += 1

        logger.info(f"Parsing completed - Success rate: {self.stats.success_rate:.2f}%")
        return documents


class FileParser(BaseParser):
    def __init__(
        self,
        config: Config,
        loader_class: type[BaseLoader],
        loader_kwargs: dict | None = None,
        file_type_name: str = "file",
    ):
        super().__init__(config)
        self.loader_class = loader_class
        self.loader_kwargs = loader_kwargs or {}
        self.file_type_name = file_type_name

    def parse_file(
        self, file_path: str | Path, index_value: str | None = None
    ) -> Document:
        try:
            loader_kwargs = {**self.loader_kwargs, "file_path": str(file_path)}
            loader = self.loader_class(**loader_kwargs)
            langchain_docs = loader.load()
            document = convert_langchain_to_document(
                langchain_docs, file_path, index_value=index_value
            )

            if (
                document.content
                and document.content.text
                and document.content.text.strip() == ""
            ):
                raise DataProcessingError(
                    f"Parsed text is empty for '{self.file_type_name}' '{file_path}'"
                )

            return document
        except Exception as e:
            raise DataProcessingError(
                f"Failed to parse '{self.file_type_name}' '{file_path}': {e}"
            ) from e


class ParserFactory:
    _loader_configs = {
        ".csv": (CSVLoader, {}, "CSV"),
        ".htm": (UnstructuredHTMLLoader, {}, "HTML"),
        ".html": (UnstructuredHTMLLoader, {}, "HTML"),
        ".json": (JSONLoader, {"jq_schema": "."}, "JSON"),
        ".markdown": (UnstructuredMarkdownLoader, {}, "Markdown"),
        ".md": (UnstructuredMarkdownLoader, {}, "Markdown"),
        ".pdf": (PyPDFLoader, {}, "PDF"),
        ".txt": (TextLoader, {}, "Text"),
    }

    @classmethod
    def create_parser(cls, file_path: str | Path, config: Config) -> BaseParser:
        extension = Path(file_path).suffix.lower()

        if extension not in cls._loader_configs:
            raise DataProcessingError(f"Unsupported file type: '{extension}'")

        loader_class, loader_kwargs, file_type_name = cls._loader_configs[extension]
        return FileParser(config, loader_class, loader_kwargs, file_type_name)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        return list(cls._loader_configs.keys())
