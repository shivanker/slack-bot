"""Google Search Agent using ADK.

This agent uses the google_search tool to provide grounded answers.
"""

from google.adk.agents import Agent
from google.adk.tools import google_search


search_agent = Agent(
    name="search_agent",
    model="gemini-3-pro-preview",
    description="Agent that answers questions using Google Search for grounded, factual responses.",
    instruction=(
        "You are an expert researcher. You always stick to the facts and provide "
        "well-sourced answers. Use Google Search to find accurate, up-to-date information. "
        "When answering, synthesize the search results into a clear, helpful response."
    ),
    tools=[google_search],
)
