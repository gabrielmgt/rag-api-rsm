"""Module to oversee embedding calculation. 
This is currently achieved within LangChain's VectorStore implementation"""

from typing import List
from langfuse import observe
from langchain_core.documents import Document
from app.core.logging import logger
from app.services.vectorstore import vector_store
from app.core.observability.langfuse import langfuse_callback_handler

@observe(name="embedding_computation")
async def compute_embeddings_and_add_to_store(chunks: List[Document]):
    """The method add_documents already computes embeddings internally which we 
    trace for using the observe() decorator because it doesn't accept the langfuse callback handler"""
    logger.debug("computing_embeddings", chunks_count=len(chunks))
    await vector_store.aadd_documents(chunks, callbacks=[langfuse_callback_handler])
    logger.info("embeddings_computed_and_stored", chunks_processed=len(chunks))
