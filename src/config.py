"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from environment variables."""

    groq_api_key: str
    groq_model: str
    embedding_model: str
    doc_directory: Path
    chroma_persist_directory: Path
    chroma_collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    checkpoint_db_path: Path

    @classmethod
    def load(cls, require_groq_key: bool = True) -> "Settings":
        """Build settings from environment variables, applying sensible defaults.

        Args:
            require_groq_key: If True, raise ConfigError when GROQ_API_KEY is
                missing. Set to False for flows (e.g. ingestion) that do not
                need the LLM.
        """
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if require_groq_key and not groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Add it to your .env file or export it "
                "as an environment variable. See .env.example for reference."
            )

        return cls(
            groq_api_key=groq_api_key,
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            doc_directory=Path(os.getenv("DOC_DIRECTORY", str(PROJECT_ROOT / "doc"))),
            chroma_persist_directory=Path(
                os.getenv(
                    "CHROMA_PERSIST_DIRECTORY", str(PROJECT_ROOT / "chroma_db")
                )
            ),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME", "documents"
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            checkpoint_db_path=Path(
                os.getenv("CHECKPOINT_DB_PATH", str(PROJECT_ROOT / "checkpoints.sqlite"))
            ),
        )
