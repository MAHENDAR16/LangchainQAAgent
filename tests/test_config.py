from __future__ import annotations

import pytest

from src.config import ConfigError, Settings


def test_load_raises_without_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        Settings.load()


def test_load_succeeds_without_key_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings.load(require_groq_key=False)
    assert settings.groq_api_key == ""


def test_load_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "abc123")
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    settings = Settings.load()
    assert settings.groq_api_key == "abc123"
    assert settings.chunk_size == 1000
    assert settings.retrieval_k == 4


def test_load_respects_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "abc123")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("RETRIEVAL_K", "8")
    settings = Settings.load()
    assert settings.chunk_size == 500
    assert settings.retrieval_k == 8
