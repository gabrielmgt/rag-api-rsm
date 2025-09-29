"""Module for environment configuration using pydantic-settings"""

from typing import Literal
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    ENV: Literal['dev', 'prod'] = Field(
        default='dev',
        description="Runtime mode: dev, prod. " \
    "dev mode is for in-memory ChromaDB" \
    "prod mode is for docker-compose.yml use with a local ChromaDB container")

    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str

    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)

    model_config = SettingsConfigDict(
        env_file = (".env.prod" if os.getenv("ENV") == "prod" else ".env.dev"),
        env_file_encoding = "utf-8"
    )

    LLM_provider: str
    LLM_model: str
    Google_API_Key: str
    Embeddings_model: str = Field(default="huggingface")

settings = Settings() # type: ignore
