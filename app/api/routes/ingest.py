"""Ingest endpoint"""

from fastapi import APIRouter, status
from langfuse import observe
from app.core.logging import logger
from app.models.schemas import IngestRequest, IngestResponse
from app.core.loader.url_loader import load_document_from_url
from app.core.loader.content_loader import load_document_from_content
from app.core.loader.text_splitter import split_documents_with_tracing
from app.core.embeddings.compute_embeddings import compute_embeddings_and_add_to_store
from app.core.exceptions import IngestionException

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_document(request: IngestRequest):
    """
    Document Ingestion endpoint
    """
    logger.info("ingest_started", document_type=request.document_type)
    try:
        chunks = await document_from_content_or_url_and_trace(request)
        logger.info("ingest_completed", chunks_created=len(chunks))

        return IngestResponse(
            status="success",
            message="Successfully ingested document.",
            chunks_created=len(chunks)
        )
    except IngestionException as e:
        logger.error("ingest_failed", error=str(e))
        return IngestResponse(
            status="error",
            message=f"An error occurred: {e}",
            chunks_created=0
        )

@observe(name="document_ingestion")    
async def document_from_content_or_url_and_trace(request: IngestRequest):
    """
    Load document from URL if url is detected 
    Nest both chunk documents and embedding computations
    """

    try:
        if request.url:
            logger.info("loading_document_from_url", 
                        url=str(request.url), 
                        document_type=request.document_type)
            docs = await load_document_from_url(request.url, 
                                                request.document_type)
            for doc in docs:
                doc.metadata["source_url"] = str(request.url)
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
    except Exception as e:
        raise IngestionException(status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                 f"Document ingestion failed: {e}") from e
