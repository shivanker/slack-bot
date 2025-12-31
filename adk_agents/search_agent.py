"""Google Search Agent using ADK.

This agent uses the google_search tool to provide grounded answers.
"""

from google.adk.agents import Agent
from google.adk.tools import google_search

# This import relies on the script running from the repo root
from system_instructions import *


search_agent = Agent(
    name="search_agent",
    model="gemini-3-pro-preview",
    description="Agent that answers questions using Google Search for grounded, factual responses.",
    instruction=(
        SLACKBOT_SYSTEM_INSTRUCTION
        + "\n\nYou are an expert researcher. You always stick to the facts and provide "
        "well-sourced answers. Use Google Search to find accurate, up-to-date information. "
        "When answering, synthesize the search results into a clear, helpful response.\n\n"
        + SYSTEM_INSTRUCTION_EPILOGUE
    ),
    tools=[google_search],
)
