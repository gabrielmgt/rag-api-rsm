"""Module to configure and initialize vectorstore"""

import hashlib
from langchain_chroma import Chroma
from app.core.embeddings.embeddings_model import embeddings
from app.core.logging import logger
from app.config.pydantic_settings import settings
from app.models.schemas import IngestRequest

def initialize_vectorstore():
    """
    Setup Chroma vector store here 
    We consider an in-memory Chroma and Chroma running on a 
    separate container depending on the running mode defined 
    by environment variable ENV
    """
    logger.debug("initializing_vectorstore",
                 env=settings.ENV,
                 host=settings.chroma_host,
                 port=settings.chroma_port)
    chroma_instance = None
    if settings.ENV == "prod":
        chroma_instance = Chroma(
            embedding_function=embeddings,
            host=settings.chroma_host,
            port=settings.chroma_port,
            )
    else:
        chroma_instance = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db",
            #host="localhost",
            #ssl=,
            #port=
            )

    logger.info("vectorstore_initialized", env=settings.ENV)
    return chroma_instance

vector_store = initialize_vectorstore()

def is_already_ingested(request: IngestRequest, vector_store_instance: Chroma):
    """
    Check if document is already ingested in vector store
    return: bool (true if exists)
    """
    try:
        source_check = ""
        if request.url:
            source_check = str(request.url)
        elif request.content:
            source_check = hashlib.sha256(request.content.encode('utf-8')).hexdigest()
        else:
            raise ValueError("request format not supported")

        existing_docs = vector_store_instance.get(
            where={"identifier": source_check},
            limit=1
        )

        return len(existing_docs['ids']) > 0, source_check

    except Exception as e:
        logger.error("duplicate_check_failed", error=str(e))
        raise RuntimeError(f"failed to check if document is already ingested {e}") from e
