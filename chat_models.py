from enum import Enum

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.generators.anthropic import \
    AnthropicChatGenerator
from haystack_integrations.components.generators.google_ai import \
    GoogleAIGeminiChatGenerator

GPT_35 = OpenAIChatGenerator(model="gpt-3.5-turbo")
GPT_4_TURBO = OpenAIChatGenerator(model="gpt-4-turbo")
CLAUDE_3_OPUS = AnthropicChatGenerator(model="claude-3-opus-20240229")
CLAUDE_3_SONNET = AnthropicChatGenerator(model="claude-3-sonnet-20240229")
CLAUDE_3_HAIKU = AnthropicChatGenerator(model="claude-3-haiku-20240307")
GEMINI_15_PRO = GoogleAIGeminiChatGenerator(model="gemini-1.5-pro-latest")
LLAMA3_70B = OpenAIChatGenerator(
    model="llama3-70b-8192",
    api_base_url="https://api.groq.com/openai/v1",
    api_key=Secret.from_env_var("GROQ_API_KEY"),
)
LLAMA3_8B = OpenAIChatGenerator(
    model="llama3-8b-8192",
    api_base_url="https://api.groq.com/openai/v1",
    api_key=Secret.from_env_var("GROQ_API_KEY"),
)
