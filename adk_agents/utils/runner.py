"""ADK Runner utilities for executing agents."""

from typing import AsyncGenerator, Any

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.genai import types

from adk_agents.utils.session_manager import ADKSessionManager
from messages import ChatMessage, ChatRole


async def run_agent_async(
    agent: Agent,
    user_id: str,
    session_id: str,
    history: list[ChatMessage],
    app_name: str = "slack_bot",
) -> AsyncGenerator[Any, None]:
    """Run an ADK agent asynchronously and yield events.

    Args:
        agent: The ADK Agent to run.
        user_id: The user identifier.
        session_id: The session identifier.
        history: The conversation history, including the new message at the end.
        app_name: The application name for session management.

    Yields:
        Events from the agent execution.
    """
    session_manager = ADKSessionManager.get_instance(app_name)
    session = await session_manager.get_or_create_session(user_id, session_id)

    # Ingest history events (all except last)
    # The last message is considered the new user input
    for msg in history[:-1]:
        # Map ChatRole to ADK roles (user or model)
        role = "model" if msg.role == ChatRole.ASSISTANT else "user"
        content = types.Content(role=role, parts=[types.Part(text=msg.content)])
        event = Event(author=role, content=content, actions=EventActions())
        await session_manager.session_service.append_event(session, event)

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_manager.session_service,
    )

    new_message_text = history[-1].content
    content = types.Content(
        role="model" if history[-1].role == ChatRole.ASSISTANT else "user",
        parts=[types.Part(text=new_message_text)],
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        yield event
