import asyncio

import agents
from openai import AsyncOpenAI
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    WebSearchTool,
    ModelSettings,
)
from lite_llms import TextModel
from litellm import acompletion  # type: ignore
from agents.mcp import MCPServer, MCPServerStdio
from typing import Any
from httpx import URL
from openai import NOT_GIVEN
from openai.types.chat.chat_completion import ChatCompletion
from litellm import CustomStreamWrapper
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.completion_usage import CompletionUsage
from typing import List

agents.set_tracing_disabled(True)
agents.set_default_openai_api("chat_completions")


class LiteLLMClient(AsyncOpenAI):
    def __init__(
        self, custom_litellm_completion_args: dict[str, Any] = {}, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.base_url = URL("http://shivanker.litellm.ai")
        self.custom_litellm_completion_args = custom_litellm_completion_args
        self.chat.completions.create = self.acompletion  # type: ignore

    async def acompletion(self, **kwargs):
        print("yooo!")
        args_copy = {}
        for k, v in kwargs.items():
            if v != NOT_GIVEN and v != None:
                args_copy[k] = v
        for k, v in self.custom_litellm_completion_args.items():
            args_copy[k] = v  # First override with custom args
            if v == None:
                del args_copy[k]  # Then remove if None
        response = await acompletion(**args_copy)
        if isinstance(response, CustomStreamWrapper):
            response = response.model_response_creator()
        print(f"{response.choices[0].model_dump()=}")
        completion = ChatCompletion(
            id=response.id,
            choices=[Choice(**({"delta": {}} | x.model_dump())) for x in response.choices],  # type: ignore
            created=response.created,
            model=response.model or "",
            object="chat.completion",
            system_fingerprint=response.system_fingerprint,
            usage=CompletionUsage(**response.usage.model_dump()),  # type: ignore
        )
        return completion


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        extra_completion_params: dict[str, Any] = {
            "max_tokens": 128000,
        }
        if model_name and model_name.startswith("o"):
            extra_completion_params["reasoning_effort"] = "high"
        elif model_name == TextModel.CLAUDE_37_SONNET.value:
            extra_completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000,
            }
            extra_completion_params["max_completion_tokens"] = 64000
            extra_completion_params["store"] = None
        return OpenAIChatCompletionsModel(
            model=model_name or "claude-3-7-sonnet-20250219",
            openai_client=LiteLLMClient(custom_litellm_completion_args=extra_completion_params),  # type: ignore
        )


async def main():
    async with MCPServerStdio(
        name="Filesystem Server, via npx",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "~/tmp"],
        },
    ) as server:
        agent = Agent(
            name="Math Tutor",
            instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
            model="claude-3-7-sonnet-20250219",
            model_settings=ModelSettings(max_tokens=8000),
            mcp_servers=[server],
            # tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})],
        )

        result = await Runner.run(
            agent,
            "What is the latest capital of France in 2025? I think it was recently moved from Paris. Check the news.",
            run_config=RunConfig(
                model_provider=CustomModelProvider(),
            ),
        )

        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
