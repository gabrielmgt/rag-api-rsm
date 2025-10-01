"""Module for utility to load a document from URL given a type"""

import hashlib
from typing import List
from langfuse import observe
from langchain_core.documents import Document
from app.models.schemas import DocumentType

@observe(name="load_document_from_content")
def load_document_from_content(content: str, document_type: DocumentType) -> List[Document]:
    """Load document from direct content"""

    content_checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()

    metadata = {
        "identifier": content_checksum,
        "source_url": None,
        "source_type": "content", 
        "document_type": document_type.value}
    docs = [Document(page_content=content, metadata=metadata)]
    return docs
