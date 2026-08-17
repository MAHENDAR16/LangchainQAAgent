"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


@pytest.fixture
def sample_doc_dir(tmp_path: Path) -> Path:
    """A temporary doc/ directory with a couple of small text files."""
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    (doc_dir / "handbook.txt").write_text(
        "The company offers 20 days of paid leave per year. "
        "Employees must request leave at least two weeks in advance. "
        "Unused leave does not carry over to the next year.",
        encoding="utf-8",
    )
    (doc_dir / "onboarding.md").write_text(
        "# Onboarding\n\nNew employees receive a laptop on their first day. "
        "IT setup is completed within the first week.",
        encoding="utf-8",
    )
    return doc_dir


@pytest.fixture
def test_settings(tmp_path: Path, sample_doc_dir: Path) -> Settings:
    return Settings(
        groq_api_key="test-key",
        groq_model="openai/gpt-oss-120b",
        embedding_model="fake-embedding-model",
        doc_directory=sample_doc_dir,
        chroma_persist_directory=tmp_path / "chroma_db",
        chroma_collection_name="test_documents",
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=2,
        checkpoint_db_path=tmp_path / "checkpoints.sqlite",
        context_edit_trigger_tokens=4000,
        context_edit_keep_tool_outputs=3,
        summarization_trigger_tokens=6000,
        summarization_keep_messages=12,
    )
