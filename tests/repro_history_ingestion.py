import sys
import os

print("Starting script...", flush=True)

from unittest.mock import MagicMock, patch
from google.adk.events import Event
from adk_agents.utils.runner import run_agent_async
from adk_agents.utils.session_manager import ADKSessionManager
from messages import ChatMessage, ChatRole
import asyncio


async def test_history_ingestion():
    user_id = "test_user_1"
    session_id = "test_session_1"

    # Create history
    history = [
        ChatMessage.from_user("Hello"),
        ChatMessage.from_assistant("Hi there"),
        ChatMessage.from_user("How are you?"),
    ]

    print("Running verification logic...", flush=True)

    # Mock Runner to avoid LLM calls
    with patch("adk_agents.utils.runner.Runner") as MockRunnerClass:
        mock_runner_instance = MagicMock()
        MockRunnerClass.return_value = mock_runner_instance

        # Mock run_async to yield nothing
        async def mock_run_async(*args, **kwargs):
            print("Mock run_async called", flush=True)
            new_message = kwargs.get("new_message")
            if new_message:
                print(f"Passed new_message: {new_message.parts[0].text}", flush=True)
            if False:
                yield
            print("Mock run_async returning", flush=True)
            return

        mock_runner_instance.run_async.side_effect = mock_run_async

        print("Calling run_agent_async...", flush=True)
        # Run
        gen = run_agent_async(
            agent=MagicMock(), user_id=user_id, session_id=session_id, history=history
        )

        print("Iterating generator...", flush=True)
        async for event in gen:
            print("Received event", flush=True)
            pass
        print("Generator finished.", flush=True)

        print("Verifying session...", flush=True)
        session_manager = ADKSessionManager.get_instance("slack_bot")
        session = await session_manager.get_or_create_session(user_id, session_id)

        print(f"Session events count: {len(session.events)}", flush=True)

        for i, event in enumerate(session.events):
            text = "None"
            if event.content and event.content.parts:
                text = event.content.parts[0].text
            print(f"Event {i}: Author={event.author}, Content={text}", flush=True)

        assert len(session.events) == 2, f"Expected 2 events, got {len(session.events)}"
        assert session.events[0].author == "user"
        assert session.events[0].content.parts[0].text == "Hello"
        assert session.events[1].author == "model"
        assert session.events[1].content.parts[0].text == "Hi there"

        call_kwargs = mock_runner_instance.run_async.call_args.kwargs
        new_message = call_kwargs.get("new_message")
        print(
            f"New message passed to runner verification: {new_message.parts[0].text}",
            flush=True,
        )
        assert new_message.parts[0].text == "How are you?"

        print("Verification Successful!", flush=True)


if __name__ == "__main__":
    asyncio.run(test_history_ingestion())
