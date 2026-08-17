"""Retrieval module: loads the persisted ChromaDB store and runs similarity search.

Ingestion (src/ingestion/ingest.py) must have been run at least once before
this module has anything to retrieve.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when the vector store cannot be loaded or queried."""


def load_vector_store(settings: Settings) -> Chroma:
    """Load the existing, persisted ChromaDB collection.

    Uses the same embedding model that was used during ingestion, which is
    required for query vectors to be comparable to the stored vectors.
    """
    if not settings.chroma_persist_directory.exists():
        raise RetrieverError(
            f"ChromaDB directory not found at {settings.chroma_persist_directory}. "
            "Run 'python -m src.ingestion.ingest' first to build the vector store."
        )

    try:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_directory),
        )
    except Exception as exc:
        raise RetrieverError(f"Failed to load ChromaDB collection: {exc}") from exc

    if vector_store._collection.count() == 0:
        raise RetrieverError(
            f"ChromaDB collection '{settings.chroma_collection_name}' is empty. "
            "Run 'python -m src.ingestion.ingest' first to populate it."
        )

    logger.info(
        "Loaded ChromaDB collection '%s' with %d chunk(s)",
        settings.chroma_collection_name,
        vector_store._collection.count(),
    )
    return vector_store


def build_retriever(settings: Settings) -> VectorStoreRetriever:
    """Build a retriever performing top-k semantic similarity search."""
    vector_store = load_vector_store(settings)
    return vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})


def search_documents(
    retriever: VectorStoreRetriever, query: str
) -> list[Document]:
    """Run a semantic similarity search and return the most relevant chunks."""
    try:
        results = retriever.invoke(query)
    except Exception as exc:
        raise RetrieverError(f"Document retrieval failed: {exc}") from exc

    logger.info("Retrieved %d chunk(s) for query", len(results))
    return results


def format_sources(documents: list[Document]) -> str:
    """Render retrieved documents into a citation-friendly text block."""
    if not documents:
        return "No relevant documents found."

    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header = f"Source: {source}" + (f", page {page + 1}" if page is not None else "")
        blocks.append(f"[{header}]\n{doc.page_content}")

    return "\n\n".join(blocks)
