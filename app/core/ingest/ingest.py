"""Module for document ingestion"""

from langfuse import observe
from app.core.exceptions.exceptions import DuplicateDocumentException
from app.core.logging import logger
from app.models.schemas import IngestRequest
from app.core.loader.url_loader import load_document_from_url
from app.core.loader.content_loader import load_document_from_content
from app.core.loader.text_splitter import split_documents_with_tracing
from app.core.embeddings.compute_embeddings import compute_embeddings_and_add_to_store
from app.services.vectorstore import is_already_ingested

@observe(name="document_ingestion")    
async def document_from_content_or_url_and_trace(request: IngestRequest):
    """
    Load document from URL if url is detected 
    Nest both chunk documents and embedding computations
    return: List[Document] chunks
    """

    is_duplicate, source_check_value = is_already_ingested(request)

    if is_duplicate:
        logger.warning("duplicate_document_rejected", source_check=source_check_value)
        raise DuplicateDocumentException(f"document already exists in vector store: {source_check_value}")
    
    if request.url:
        logger.info("loading_document_from_url", 
                    url=str(request.url), 
                    document_type=request.document_type)
        docs = await load_document_from_url(request.url, 
                                            request.document_type)
        #for doc in docs:
        #    doc.metadata["source_url"] = str(request.url)
        #    doc.metadata["source"] = str(request.url)
        logger.info("url_documents_loaded",
                    documents_count=len(docs))
    elif request.content:
        logger.debug("loading_document_from_content", 
                      document_type=request.document_type, 
                      content_length=len(request.content))
        docs = load_document_from_content(request.content, 
                                          request.document_type)
        logger.info("document_loaded_from_content")

    chunks = split_documents_with_tracing(docs)

    await compute_embeddings_and_add_to_store(chunks)
    return chunks
    