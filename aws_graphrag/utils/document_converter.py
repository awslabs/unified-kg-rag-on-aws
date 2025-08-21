from pathlib import Path

from langchain_core.documents import Document as LangChainDocument

from aws_graphrag.models import Document, DocumentContent, Page
from aws_graphrag.utils import generate_stable_id


def convert_langchain_to_document(
    langchain_docs: list[LangChainDocument], file_path: str | Path, n_chars: int = 100
) -> Document:
    path = Path(file_path)

    combined_text = "\n\n".join(
        doc.page_content for doc in langchain_docs if doc.page_content
    )

    metadata = langchain_docs[0].metadata if langchain_docs else {}
    document_id = generate_stable_id(f"doc:{path.name}:{combined_text[:n_chars]}")

    pages = [
        Page(
            page_number=i + 1,
            text_content=doc.page_content,
            elements=[],
            raw_data=doc.metadata,
        )
        for i, doc in enumerate(langchain_docs)
    ]

    content = DocumentContent(text=combined_text, html=None, markdown=None)

    file_type = path.suffix.lower().lstrip(".")

    return Document(
        document_id=document_id,
        file_name=path.name,
        file_path=str(path),
        file_type=file_type,
        detected_language="Unknown",
        total_pages=len(langchain_docs),
        pages=pages,
        elements=[],
        content=content,
        metadata=metadata,
        page_content=combined_text,
    )
