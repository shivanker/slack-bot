from enum import Enum

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.google_ai import (
    GoogleAIGeminiChatGenerator,
)

GPT_35 = OpenAIChatGenerator(model="gpt-3.5-turbo").run
GPT_4O = OpenAIChatGenerator(model="gpt-4o").run
GPT_4_TURBO = OpenAIChatGenerator(model="gpt-4-turbo").run
CLAUDE_3_OPUS = AnthropicChatGenerator(model="claude-3-opus-20240229").run
CLAUDE_3_SONNET = AnthropicChatGenerator(model="claude-3-sonnet-20240229").run
CLAUDE_3_HAIKU = AnthropicChatGenerator(model="claude-3-haiku-20240307").run
GEMINI_15_PRO = GoogleAIGeminiChatGenerator(model="gemini-1.5-pro-latest").run
GROQ_LLAMA3_70B = OpenAIChatGenerator(
    model="llama3-70b-8192",
    api_base_url="https://api.groq.com/openai/v1",
    api_key=Secret.from_env_var("GROQ_API_KEY"),
).run
GROQ_LLAMA3_8B = OpenAIChatGenerator(
    model="llama3-8b-8192",
    api_base_url="https://api.groq.com/openai/v1",
    api_key=Secret.from_env_var("GROQ_API_KEY"),
).run
LLAMA3_70B = OpenAIChatGenerator(
    model="accounts/fireworks/models/llama-v3-70b-instruct",
    api_base_url="https://api.fireworks.ai/inference/v1",
    api_key=Secret.from_env_var("FIREWORKS_API_KEY"),
).run
LLAMA3_8B = OpenAIChatGenerator(
    model="accounts/fireworks/models/llama-v3-8b-instruct",
    api_base_url="https://api.fireworks.ai/inference/v1",
    api_key=Secret.from_env_var("FIREWORKS_API_KEY"),
).run
