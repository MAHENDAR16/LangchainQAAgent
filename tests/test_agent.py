from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agent import agent as agent_module
from src.agent.agent import _rewrite_query, build_agent
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


class _FakeRewriteLLM(BaseChatModel):
    """Fake chat model that records calls and returns a fixed rewrite."""

    rewritten_query: str = "rewritten standalone query"
    should_raise: bool = False
    call_count: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_FakeRewriteLLM":
        return self

    def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> ChatResult:
        self.call_count += 1
        if self.should_raise:
            raise RuntimeError("Groq API error")
        message = AIMessage(content=self.rewritten_query)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-rewrite-chat-model"


class _FakeAgentLLM(BaseChatModel):
    """Fake tool-calling model driving a two-turn conversation.

    Turn 1 ("Tell me about the CAP theorem") answers directly, with no tool
    call, to seed conversation history. Turn 2 ("What about page 2?") emits a
    tool call to search_documents with the bare follow-up text as the query
    — mimicking how a real model would pass along a follow-up without fully
    resolving its reference. Any prompt containing "Standalone query:" is
    treated as the internal query-rewrite call and answered with a fixed
    standalone query, so the test can assert the retriever received the
    rewritten form rather than the raw follow-up.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_FakeAgentLLM":
        return self

    def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> ChatResult:
        last = messages[-1]
        content = getattr(last, "content", "")
        if "Standalone query:" in content:
            message = AIMessage(content="rewritten standalone query")
        elif isinstance(last, HumanMessage) and last.content == "What about page 2?":
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_documents", "args": {"query": "page 2"}, "id": "call-1"}
                ],
            )
        else:
            message = AIMessage(content="Final answer")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-agent-chat-model"


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


def test_rewrite_query_returns_original_when_no_history() -> None:
    llm = _FakeRewriteLLM()

    result = _rewrite_query("what about it?", [], llm)

    assert result == "what about it?"
    assert llm.call_count == 0


def test_rewrite_query_returns_original_with_only_current_turn() -> None:
    """A single human message (the current question) isn't prior conversation."""
    llm = _FakeRewriteLLM()

    result = _rewrite_query("what about it?", [HumanMessage(content="what about it?")], llm)

    assert result == "what about it?"
    assert llm.call_count == 0


def test_rewrite_query_uses_llm_when_history_present() -> None:
    llm = _FakeRewriteLLM(rewritten_query="what does page 2 of the handbook say?")
    history = [
        HumanMessage(content="Tell me about the leave policy"),
        AIMessage(content="Employees get 20 days of paid leave."),
        HumanMessage(content="what about page 2?"),
    ]

    result = _rewrite_query("what about page 2?", history, llm)

    assert result == "what does page 2 of the handbook say?"
    assert llm.call_count == 1


def test_rewrite_query_falls_back_to_original_on_llm_error() -> None:
    llm = _FakeRewriteLLM(should_raise=True)
    history = [
        HumanMessage(content="Tell me about the leave policy"),
        AIMessage(content="Employees get 20 days of paid leave."),
    ]

    result = _rewrite_query("what about page 2?", history, llm)

    assert result == "what about page 2?"


def test_search_tool_rewrites_follow_up_query_using_history(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the search_documents tool should receive the rewritten
    query (not the raw follow-up text) once there's prior conversation."""
    ingest_module.run_ingestion(test_settings)

    captured_queries: list[str] = []
    original_search_documents = agent_module.search_documents

    def _spy_search_documents(retriever: Any, query: str) -> Any:
        captured_queries.append(query)
        return original_search_documents(retriever, query)

    monkeypatch.setattr(agent_module, "search_documents", _spy_search_documents)

    agent = build_agent(test_settings, llm=_FakeAgentLLM())
    config = thread_config("erin")

    agent.invoke(
        {"messages": [HumanMessage(content="Tell me about the CAP theorem")]}, config=config
    )
    agent.invoke({"messages": [HumanMessage(content="What about page 2?")]}, config=config)

    assert captured_queries == ["rewritten standalone query"]


def test_thread_config_tags_run_with_thread_id() -> None:
    config = thread_config("erin")

    assert config["configurable"]["thread_id"] == "erin"
    assert "thread:erin" in config["tags"]
    assert config["metadata"]["thread_id"] == "erin"
