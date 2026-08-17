from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.ingest import (
    IngestionError,
    discover_documents,
    load_documents,
    split_documents,
)


def test_discover_documents_finds_supported_files(sample_doc_dir: Path) -> None:
    files = discover_documents(sample_doc_dir)
    names = {f.name for f in files}
    assert names == {"handbook.txt", "onboarding.md"}


def test_discover_documents_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        discover_documents(tmp_path / "does_not_exist")


def test_discover_documents_raises_on_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_doc"
    empty_dir.mkdir()
    with pytest.raises(IngestionError):
        discover_documents(empty_dir)


def test_load_documents_attaches_source_metadata(sample_doc_dir: Path) -> None:
    files = discover_documents(sample_doc_dir)
    documents = load_documents(files)
    assert len(documents) == 2
    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {"handbook.txt", "onboarding.md"}
    for doc in documents:
        assert doc.metadata["file_type"] in {"txt", "md"}


def test_split_documents_creates_chunks_with_stable_ids(sample_doc_dir: Path) -> None:
    files = discover_documents(sample_doc_dir)
    documents = load_documents(files)
    chunks = split_documents(documents, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > len(documents)
    for chunk in chunks:
        assert "chunk_uid" in chunk.metadata
        assert "chunk_id" in chunk.metadata

    # Re-splitting the same documents must produce identical chunk ids
    # (idempotency for re-ingestion).
    chunks_again = split_documents(documents, chunk_size=80, chunk_overlap=10)
    assert [c.metadata["chunk_uid"] for c in chunks] == [
        c.metadata["chunk_uid"] for c in chunks_again
    ]
