"""Module to configure and initialize vectorstore"""

from langchain_chroma import Chroma
from app.core.embeddings.embeddings_model import embeddings
from app.core.logging import logger
from app.config.pydantic_settings import settings

def initialize_vectorstore():
    """
    Setup Chroma vector store here 
    We consider an in-memory Chroma and Chroma running on a 
    separate container depending on the running mode defined 
    by environment variable ENV
    """
    logger.debug("initializing_vectorstore", env=settings.ENV, host=settings.chroma_host, port=settings.chroma_port)
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
