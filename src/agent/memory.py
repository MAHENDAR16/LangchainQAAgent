"""Persistent conversation memory for the agent, backed by LangGraph's SQLite checkpointer.

Each conversation is identified by a ``thread_id`` (typically a per-user id).
The checkpointer persists the full message history for a thread to disk, so
conversations survive application restarts and stay isolated per user/thread
— callers only ever need to send the newest message, not the full history.
"""

from __future__ import annotations

import logging
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import Settings

logger = logging.getLogger(__name__)


def build_checkpointer(settings: Settings) -> SqliteSaver:
    """Open (creating if needed) the persistent SQLite checkpoint store."""
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: the connection is reused across requests in
    # both the CLI's single-threaded loop and Streamlit's per-session reruns.
    conn = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    logger.info("Loaded conversation checkpoint store at %s", settings.checkpoint_db_path)
    return checkpointer


def thread_config(thread_id: str) -> dict:
    """Build the LangGraph run config that scopes a call to one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}
