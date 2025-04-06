import uuid

from agno.agent.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.models.anthropic.claude import Claude
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.youtube import YouTubeTools
from agno.tools.calculator import CalculatorTools
from agno.tools.thinking import ThinkingTools
from agno.tools.crawl4ai import Crawl4aiTools
from agno.storage.dynamodb import DynamoDbStorage

storage = DynamoDbStorage(
    table_name="agent_slackbot_agno_sessions",
    region_name="us-east-1",
)

agent = Agent(
    model=Claude(id="claude-3-7-sonnet-20250219"),
    user_id="shivanker",
    session_id=str(uuid.uuid4()),
    description="You are a helpful agent.",
    tools=[
        DuckDuckGoTools(),
        YFinanceTools(enable_all=True),
        YouTubeTools(),
        CalculatorTools(enable_all=True),
        ThinkingTools(),
        Crawl4aiTools(max_length=None),
    ],
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    tool_call_limit=25,
    storage=storage,
    add_history_to_messages=True,
    # reasoning=True,
    telemetry=False,
)
while True:
    user_input = input("You: ")
    for response in agent.run(user_input, stream=True):
        print(f"{response=}")
