"""Module for features to load in during FastAPI lifespan"""

from app.models.schemas import IngestRequest
from app.core.logging import logger
from app.core.ingest.ingest import document_from_content_or_url_and_trace
from app.core.exceptions.exceptions import DuplicateDocumentException

async def auto_ingest_base_documents():
    """
    Auto ingest base documents
    """
    doc_1 = IngestRequest(content = None,
                          url = "https://allendowney.github.io/ThinkPython/index.html",
                          document_type = "html")

    doc_2 = IngestRequest(content = None,
                          url = "https://peps.python.org/pep-0008/",
                          document_type = "html")

    docs = [doc_1, doc_2]

    logger.info("auto_ingest_started")
    for i, doc in enumerate(docs):
        try:
            chunks = await document_from_content_or_url_and_trace(doc)
            logger.info("auto_ingest_completed", document_url = doc.url, chunks_created=(len(chunks)))
        except DuplicateDocumentException:
            logger.warning("auto_ingest_found_duplicate", auto_ingest_doc_id = i, document_url = doc.url)
