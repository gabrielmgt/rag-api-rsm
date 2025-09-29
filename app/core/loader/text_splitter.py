"""Module to configure the LangChain text splitter"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langfuse import observe
from app.core.logging import logger

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

@observe(name="document_splitting")
def split_documents_with_tracing(docs: List[Document]):
    """Do a tracing span for the method split_documents which has no callback for langfuse. 
    There is no async version of split_documents"""
    logger.debug("splitting_documents", document_count=len(docs))
    chunks = text_splitter.split_documents(docs)
    logger.info("documents_split", chunks_created=len(chunks))
    return chunks
