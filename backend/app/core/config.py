"""
Application settings — loaded from .env via pydantic-settings.

All values have sensible defaults so the app starts without a .env file,
but OPENAI_API_KEY must be set for the /query endpoint to function.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root regardless of where uvicorn is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    llm_base_url: str = ""          # leave blank for OpenAI; set for Groq/Mistral/etc.
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.1

    # HuggingFace
    hf_token: str = ""

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Retrieval
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

    # Chunking
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # Storage
    vector_store_dir: str = "data/vector_store"
    raw_data_dir: str = "data/raw"

    # Qdrant
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "arabic_rag"

    # Upload limits
    max_upload_mb: int = 50

    @property
    def vector_store_path(self) -> Path:
        p = Path(self.vector_store_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def raw_data_path(self) -> Path:
        p = Path(self.raw_data_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p


settings = Settings()
