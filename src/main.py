"""CLI entry point for the Document Q&A Agent.

Usage:
    python -m src.main
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import HumanMessage

from src.agent.agent import build_agent
from src.agent.memory import thread_config
from src.config import ConfigError, Settings
from src.retrieval.retriever import RetrieverError

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    try:
        settings = Settings.load()
        agent = build_agent(settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc
    except RetrieverError as exc:
        print(f"Vector store error: {exc}")
        raise SystemExit(1) from exc

    print("Document Q&A Agent")
    print("Type 'exit' to quit.\n")

    user_id = input(
        "User ID (reuse an ID to resume your saved conversation, "
        "leave blank for a new anonymous session): "
    ).strip()
    thread_id = user_id or f"anonymous-{uuid.uuid4()}"
    config = thread_config(thread_id)

    prior_state = agent.get_state(config)
    prior_messages = prior_state.values.get("messages", []) if prior_state.values else []
    if prior_messages:
        print(f"\nResuming saved conversation for '{thread_id}' "
              f"({len(prior_messages)} prior message(s)).\n")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]}, config=config
            )
            answer = result["messages"][-1].content
        except Exception as exc:  # Groq API errors, network errors, etc.
            logger.error("Agent invocation failed: %s", exc)
            answer = (
                "Sorry, something went wrong while generating an answer. "
                "Please try again."
            )

        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    run_cli()
