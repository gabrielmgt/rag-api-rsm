"""Module for utility to load a document from URL given a type"""

import os
import tempfile
from typing import List
import requests
from langfuse import observe
from pydantic import HttpUrl
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from app.models.schemas import DocumentType
from app.core.logging import logger

@observe(name="load_document_from_url")
async def load_document_from_url(url: HttpUrl, document_type: DocumentType) -> List[Document]:
    """Load document from URL based on document type"""
    try:
        url_str = str(url)
        docs: List[Document] = []

        if document_type == DocumentType.HTML:
            loader = WebBaseLoader([url_str])
            docs = loader.load()

        elif document_type == DocumentType.PDF:
            response = requests.get(url_str, timeout=10)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            finally:
                os.unlink(tmp_file_path)

        elif document_type in [DocumentType.TEXT, DocumentType.MARKDOWN]:
            response = requests.get(str(url), timeout=10)
            response.raise_for_status()
            return [Document(page_content=response.text)]

        for doc in docs:
                    doc.metadata = {
                        "identifier": url_str,  
                        "source_type": "url", 
                        "document_type": document_type.value,
                        "source_url": url_str,
                    }
        return docs

    except Exception as e:
        logger.error("document_loading_from_url_failed", url=str(url), error=str(e))
        raise ValueError(f"Failed to load document from URL: {e}") from e
