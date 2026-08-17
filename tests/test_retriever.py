from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import Settings
from src.ingestion import ingest as ingest_module
from src.retrieval import retriever as retriever_module
from src.retrieval.retriever import RetrieverError, build_retriever, search_documents


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading a real embedding model in tests."""

    def _fake_factory(model_name: str) -> DeterministicFakeEmbedding:
        return DeterministicFakeEmbedding(size=32)

    monkeypatch.setattr(ingest_module, "HuggingFaceEmbeddings", _fake_factory)
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", _fake_factory)


def _ingest(test_settings: Settings) -> None:
    ingest_module.run_ingestion(test_settings)


def test_build_retriever_raises_when_store_missing(test_settings: Settings) -> None:
    with pytest.raises(RetrieverError):
        build_retriever(test_settings)


def test_retriever_returns_documents_after_ingestion(test_settings: Settings) -> None:
    _ingest(test_settings)
    retriever = build_retriever(test_settings)

    results = search_documents(retriever, "How many days of leave do employees get?")

    assert len(results) > 0
    assert all("source" in doc.metadata for doc in results)


def test_retriever_respects_configured_k(test_settings: Settings) -> None:
    _ingest(test_settings)
    retriever = build_retriever(test_settings)

    results = search_documents(retriever, "laptop onboarding")

    assert len(results) <= test_settings.retrieval_k
