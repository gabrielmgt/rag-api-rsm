"""Module to setup embeddings model"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.logging import logger
from app.config.pydantic_settings import settings

def initialize_embeddings_model():
    """
    Setup embeddings models here, add more models, etc
    For our use case we don't really need more than HuggingFaceEmbeddings
    """
    logger.debug("initializing_embeddings_model",
                 model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings_model_instance = GoogleGenerativeAIEmbeddings(
                                    model="models/gemini-embedding-001",
                                    google_api_key=settings.Google_API_Key # type: ignore
                                    )
    logger.info("embeddings_model_initialized")
    return embeddings_model_instance

embeddings = initialize_embeddings_model()
