"""Document ingestion pipeline: doc/ -> load -> split -> embed -> ChromaDB.

Run as a standalone, one-time (or on-demand) step whenever the contents of
the document directory change:

    python -m src.ingestion.ingest
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ConfigError, Settings

logger = logging.getLogger(__name__)

# Maps lowercase file extensions to the loader class used to read them.
_LOADERS_BY_EXTENSION = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


class IngestionError(Exception):
    """Raised when the ingestion pipeline cannot complete successfully."""


def discover_documents(doc_directory: Path) -> list[Path]:
    """Return supported document files found under ``doc_directory``."""
    if not doc_directory.exists():
        raise IngestionError(
            f"Document directory not found: {doc_directory}. Create it and add "
            "documents before running ingestion."
        )
    if not doc_directory.is_dir():
        raise IngestionError(f"Expected a directory but found a file: {doc_directory}")

    files = [
        path
        for path in sorted(doc_directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in _LOADERS_BY_EXTENSION
    ]

    if not files:
        raise IngestionError(
            f"No supported documents found in {doc_directory}. Supported types: "
            f"{', '.join(sorted(_LOADERS_BY_EXTENSION))}"
        )

    unsupported = [
        path
        for path in doc_directory.rglob("*")
        if path.is_file() and path.suffix.lower() not in _LOADERS_BY_EXTENSION
    ]
    for path in unsupported:
        logger.warning("Skipping unsupported file type: %s", path.name)

    return files


def load_documents(file_paths: list[Path]) -> list[Document]:
    """Load raw documents from disk, attaching source/file-type metadata."""
    documents: list[Document] = []
    for file_path in file_paths:
        loader_cls = _LOADERS_BY_EXTENSION[file_path.suffix.lower()]
        try:
            loaded = loader_cls(str(file_path)).load()
        except Exception as exc:  # malformed documents must not crash ingestion
            logger.warning("Failed to load %s: %s", file_path.name, exc)
            continue

        for doc in loaded:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = file_path.suffix.lower().lstrip(".")
        documents.extend(loaded)

    if not documents:
        raise IngestionError("No documents could be successfully loaded.")

    logger.info("Loaded %d document(s) from %d file(s)", len(documents), len(file_paths))
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_index = per_source_counter.get(source, 0)
        per_source_counter[source] = chunk_index + 1
        chunk.metadata["chunk_id"] = chunk_index
        chunk.metadata["chunk_uid"] = _stable_chunk_id(source, chunk)

    logger.info("Created %d chunk(s)", len(chunks))
    return chunks


def _stable_chunk_id(source: str, chunk: Document) -> str:
    """Deterministic id for a chunk based on its source, page, and position.

    Re-running ingestion on unchanged documents reproduces the same ids, so
    ChromaDB upserts existing chunks instead of duplicating them.
    """
    page = chunk.metadata.get("page", "na")
    start_index = chunk.metadata.get("start_index", "na")
    raw = f"{source}::page-{page}::start-{start_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_vector_store(
    chunks: list[Document], settings: Settings
) -> Chroma:
    """Embed chunks and persist them into the ChromaDB collection."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_directory),
    )

    ids = [chunk.metadata["chunk_uid"] for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)

    logger.info(
        "Stored %d chunk embedding(s) in ChromaDB collection '%s' at %s",
        len(chunks),
        settings.chroma_collection_name,
        settings.chroma_persist_directory,
    )
    return vector_store


def run_ingestion(settings: Settings | None = None) -> Chroma:
    """Execute the full ingestion pipeline and return the populated store."""
    settings = settings or Settings.load(require_groq_key=False)

    logger.info("Discovering documents in %s", settings.doc_directory)
    file_paths = discover_documents(settings.doc_directory)

    logger.info("Loading documents")
    documents = load_documents(file_paths)

    logger.info("Splitting documents into chunks")
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    logger.info("Generating embeddings and storing in ChromaDB")
    return build_vector_store(chunks, settings)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    try:
        run_ingestion()
    except (ConfigError, IngestionError) as exc:
        logger.error("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
