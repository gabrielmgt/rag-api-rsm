from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal, Optional
import logging
import os 

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    ENV: Literal['dev', 'prod'] = Field(default='dev', description="Runtime mode: dev, prod. " \
    "dev mode is for in-memory ChromaDB and external LangFuse with API keys, " \
    "prod mode is for docker-compose.yml use with local Chroma and local LangFuse containers.")
    
    langfuse_host: str = Field(..., env="LANGFUSE_HOST")
    langfuse_public_key: str = Field(..., env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(..., env="LANGFUSE_SECRET_KEY")

    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default="8001", env="CHROMA_PORT")
    
    model_config = SettingsConfigDict(
        env_file = (".env.prod" if os.getenv("ENV") == "prod" else ".env.dev"),
        env_file_encoding = "utf-8"
    )

    @field_validator('LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY')
    @classmethod
    def validate_langfuse_keys(cls, v, info):
        """Validate that Langfuse keys are provided in local environment"""
        # info.data contains the current values of other fields
        if info.data.get('ENV') == 'dev' and v is None:
            field_name = info.field_name
            raise ValueError(f"{field_name} is required in production environment")
        return v
    
    LLM_provider: str = Field(..., env="LLM_PROVIDER")
    LLM_model: str = Field(..., env="LLM_MODEL")

settings = Settings()