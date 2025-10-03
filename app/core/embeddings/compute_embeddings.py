"""Module to oversee embedding calculation. 
This is currently achieved within LangChain's VectorStore implementation"""

from typing import List
from langfuse import observe
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore
from app.core.logging import logger
from app.core.observability.langfuse import langfuse_callback_handler

@observe(name="embedding_computation")
async def compute_embeddings_and_add_to_store(
        chunks: List[Document],
        vector_store_instance: VectorStore):
    """The method add_documents already computes embeddings internally which we 
    trace for using the observe() decorator because it doesn't seem to accept 
    the langfuse callback handler even if we add a callback argument"""
    logger.debug("computing_embeddings", chunks_count=len(chunks))
    await vector_store_instance.aadd_documents(chunks, callbacks=[langfuse_callback_handler])
    logger.info("embeddings_computed_and_stored", chunks_processed=len(chunks))
