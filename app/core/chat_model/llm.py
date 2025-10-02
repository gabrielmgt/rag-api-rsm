"""Module to setup LLM chat model"""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.logging import logger
from app.config.pydantic_settings import settings

def initialize_chat_model():
    """
    Setup chat model here
    Google's model works best for our use case as it can be tried without 
    setting up billing information
    """
    logger.debug("initializing_chat_model", model=settings.LLM_model)
    model = ChatGoogleGenerativeAI(
        model=settings.LLM_model,  
        google_api_key=settings.Google_API_Key,
        temperature=0.1
        )
    logger.info("chat_model_initialized")
    return model


llm = initialize_chat_model()
