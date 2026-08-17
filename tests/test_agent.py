from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agent.agent import build_agent
from src.agent.memory import thread_config
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
    config = thread_config("test-user")

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

    assert result["messages"][-1].content == "This is a test answer."


def test_conversation_history_persists_within_a_thread(test_settings: Settings) -> None:
    ingest_module.run_ingestion(test_settings)
    agent = build_agent(test_settings, llm=_FakeToolCallingChatModel())
    config = thread_config("alice")

    agent.invoke({"messages": [HumanMessage(content="First question")]}, config=config)
    agent.invoke({"messages": [HumanMessage(content="Second question")]}, config=config)

    state = agent.get_state(config)
    human_messages = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert human_messages == ["First question", "Second question"]


def test_conversation_history_survives_agent_restart(test_settings: Settings) -> None:
    """Simulates an app restart: a fresh build_agent() call, same checkpoint db."""
    ingest_module.run_ingestion(test_settings)
    config = thread_config("bob")

    first_agent = build_agent(test_settings, llm=_FakeToolCallingChatModel())
    first_agent.invoke({"messages": [HumanMessage(content="Remember this")]}, config=config)

    second_agent = build_agent(test_settings, llm=_FakeToolCallingChatModel())
    state = second_agent.get_state(config)
    human_messages = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]

    assert human_messages == ["Remember this"]


def test_different_users_have_isolated_histories(test_settings: Settings) -> None:
    ingest_module.run_ingestion(test_settings)
    agent = build_agent(test_settings, llm=_FakeToolCallingChatModel())

    agent.invoke(
        {"messages": [HumanMessage(content="Alice's question")]},
        config=thread_config("alice"),
    )
    agent.invoke(
        {"messages": [HumanMessage(content="Carol's question")]},
        config=thread_config("carol"),
    )

    alice_state = agent.get_state(thread_config("alice"))
    carol_state = agent.get_state(thread_config("carol"))

    alice_questions = [
        m.content for m in alice_state.values["messages"] if isinstance(m, HumanMessage)
    ]
    carol_questions = [
        m.content for m in carol_state.values["messages"] if isinstance(m, HumanMessage)
    ]

    assert alice_questions == ["Alice's question"]
    assert carol_questions == ["Carol's question"]
