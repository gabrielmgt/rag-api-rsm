"""Module to setup embeddings model"""

from langchain_huggingface import HuggingFaceEmbeddings
from app.core.logging import logger

def initialize_embeddings_model():
    """
    Setup embeddings models here, add more models, etc
    For our use case we don't really need more than HuggingFaceEmbeddings
    """
    logger.debug("initializing_embeddings_model", 
                 model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings_model_instance = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("embeddings_model_initialized")
    return embeddings_model_instance

embeddings = initialize_embeddings_model()
