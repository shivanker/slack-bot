"""ADK Runner utilities for executing agents."""

from typing import AsyncGenerator, Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

from adk_agents.utils.session_manager import ADKSessionManager


async def run_agent_async(
    agent: Agent,
    user_id: str,
    session_id: str,
    query: str,
    app_name: str = "slack_bot",
) -> AsyncGenerator[Any, None]:
    """Run an ADK agent asynchronously and yield events.

    Args:
        agent: The ADK Agent to run.
        user_id: The user identifier.
        session_id: The session identifier.
        query: The user's query text.
        app_name: The application name for session management.

    Yields:
        Events from the agent execution.
    """
    session_manager = ADKSessionManager.get_instance(app_name)
    await session_manager.get_or_create_session(user_id, session_id)

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_manager.session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        yield event
