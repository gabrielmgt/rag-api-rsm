"""Module to setup LangFuse Observability"""

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from app.core.logging import logger
from app.config.pydantic_settings import settings

def initialize_langfuse():
    """
    Setup Langfuse instance here
    """
    logger.debug("initializing_langfuse", host=settings.langfuse_host)
    langfuse_instance = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    logger.info("langfuse_initialized")
    return langfuse_instance


langfuse = initialize_langfuse()
langfuse_callback_handler = CallbackHandler()