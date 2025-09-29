"""Module for utility to load a document from URL given a type"""

from typing import List
from langfuse import observe
from langchain_core.documents import Document
from app.models.schemas import DocumentType

@observe(name="load_document_from_content")
def load_document_from_content(content: str, document_type: DocumentType) -> List[Document]:
    """Load document from direct content"""
    metadata = {"source": "direct_input", "document_type": document_type.value}
    docs = [Document(page_content=content, metadata=metadata)]
    return docs
