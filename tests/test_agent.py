from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agent.agent import build_agent
from src.config import Settings
from src.ingestion import ingest as ingest_module
from src.retrieval import retriever as retriever_module


class _FakeToolCallingChatModel(BaseChatModel):
    """Minimal fake chat model that supports bind_tools, for agent tests."""

    response_text: str = "This is a test answer."

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_FakeToolCallingChatModel":
        return self

    def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> ChatResult:
        message = AIMessage(content=self.response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat-model"


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_factory(model_name: str) -> DeterministicFakeEmbedding:
        return DeterministicFakeEmbedding(size=32)

    monkeypatch.setattr(ingest_module, "HuggingFaceEmbeddings", _fake_factory)
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", _fake_factory)


def test_build_agent_initializes_with_mocked_llm(test_settings: Settings) -> None:
    ingest_module.run_ingestion(test_settings)

    fake_llm = _FakeToolCallingChatModel()
    agent = build_agent(test_settings, llm=fake_llm)

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    assert result["messages"][-1].content == "This is a test answer."
