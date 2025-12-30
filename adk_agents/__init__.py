"""ADK Agents package.

Provides a registry of available ADK agents and utilities for running them.
"""

from typing import Optional

from google.adk.agents import Agent

from adk_agents.search_agent import search_agent


# Agent registry mapping agent names to Agent instances
AGENTS: dict[str, Agent] = {
    "search": search_agent,
}


def get_agent(name: str) -> Optional[Agent]:
    """Get an agent by name.

    Args:
        name: The name of the agent to retrieve.

    Returns:
        The Agent instance if found, None otherwise.
    """
    return AGENTS.get(name)


def list_agents() -> list[str]:
    """Get a list of available agent names.

    Returns:
        List of registered agent names.
    """
    return list(AGENTS.keys())
