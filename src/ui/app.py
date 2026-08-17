"""Streamlit chat UI for the Document Q&A Agent.

Run with:

    streamlit run src/ui/app.py

Like the CLI (src/main.py), this only *queries* the existing ChromaDB
collection. Run `python -m src.ingestion.ingest` first to populate it.

Users log in as one of a small set of sample accounts (see SAMPLE_USERS).
Conversation history is persisted per user via the agent's LangGraph SQLite
checkpointer (src/agent/memory.py), so logging out and back in as the same
user restores that user's chat history — even across app restarts.

Note: this is a demo login (no passwords), meant to demonstrate per-user
persisted memory, not real authentication.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `from src...` imports regardless of Streamlit's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.agent import build_agent
from src.agent.memory import thread_config
from src.config import ConfigError, Settings
from src.retrieval.retriever import RetrieverError

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
for _noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

st.set_page_config(page_title="Document Q&A Agent", page_icon="📄")

# Sample accounts to demonstrate per-user persisted conversation history.
# The dict key is used as the LangGraph thread_id; the value is the display name.
SAMPLE_USERS = {
    "alice": "Alice Johnson",
    "bob": "Bob Smith",
    "carol": "Carol Martinez",
    "dave": "Dave Chen",
}


@st.cache_resource(show_spinner="Loading agent and vector store...")
def get_agent_and_settings():
    """Build the agent once and reuse it across reruns/sessions."""
    settings = Settings.load()
    return build_agent(settings), settings


def _reconstruct_display_history(messages: list) -> list[dict]:
    """Turn a raw LangGraph message list (as stored by the checkpointer)
    into chat turns for rendering, pairing each answer with the document
    excerpts that were retrieved for it.
    """
    history: list[dict] = []
    pending_sources: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
            pending_sources = []
        elif isinstance(msg, ToolMessage):
            if msg.name == "search_documents":
                pending_sources.append(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            history.append(
                {"role": "assistant", "content": msg.content, "sources": pending_sources}
            )
            pending_sources = []

    return history


def _render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(turn["sources"], start=1):
                    st.markdown(f"**Search {i}:**\n\n{chunk}")


def render_login() -> None:
    """Login screen: pick one of the sample users. No password (demo only)."""
    st.title("📄 Document Q&A Agent")
    st.subheader("Login")
    st.caption(
        "Select a sample user to start or resume that user's conversation. "
        "This is a demo login, not real authentication."
    )

    username = st.selectbox(
        "User",
        options=list(SAMPLE_USERS),
        format_func=lambda u: SAMPLE_USERS[u],
    )
    if st.button("Login", type="primary"):
        st.session_state.username = username
        st.rerun()

    st.stop()


def main() -> None:
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.username is None:
        render_login()

    username = st.session_state.username
    display_name = SAMPLE_USERS.get(username, username)

    try:
        agent, settings = get_agent_and_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except RetrieverError as exc:
        st.error(f"Vector store error: {exc}")
        st.info("Run `python -m src.ingestion.ingest` to build the vector store.")
        st.stop()

    st.title("📄 Document Q&A Agent")
    st.caption(f"Logged in as **{display_name}**. Ask questions grounded in the documents under `doc/`.")

    with st.sidebar:
        st.subheader(f"👤 {display_name}")
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()

        st.subheader("Settings")
        st.write(f"**Model:** `{settings.groq_model}`")
        st.write(f"**Embedding model:** `{settings.embedding_model}`")
        st.write(f"**Retrieval k:** {settings.retrieval_k}")

        st.subheader("Conversation")
        if st.button("Clear this conversation"):
            agent.checkpointer.delete_thread(username)
            st.rerun()

    config = thread_config(username)

    state = agent.get_state(config)
    saved_messages = state.values.get("messages", []) if state.values else []
    display_history = _reconstruct_display_history(saved_messages)

    for turn in display_history:
        _render_turn(turn)

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    _render_turn({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                prior_len = len(saved_messages)
                result = agent.invoke(
                    {"messages": [HumanMessage(content=question)]}, config=config
                )
                new_messages = result["messages"]
                answer = new_messages[-1].content
                sources = [
                    msg.content
                    for msg in new_messages[prior_len:]
                    if isinstance(msg, ToolMessage) and msg.name == "search_documents"
                ]
            except Exception as exc:  # Groq API errors, network errors, etc.
                logging.getLogger(__name__).error("Agent invocation failed: %s", exc)
                answer = (
                    "Sorry, something went wrong while generating an answer. "
                    "Please try again."
                )
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(f"**Search {i}:**\n\n{chunk}")


main()
