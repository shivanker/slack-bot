from enum import Enum


class TextModel(Enum):
    GPT_35 = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    GEMINI_15_PRO = "gemini/gemini-1.5-pro-latest"
    GROQ_LLAMA3_70B = "groq/llama3-70b-8192"
    GROQ_LLAMA3_8B = "groq/llama3-8b-8192"
    LLAMA3_70B = "fireworks_ai/llama-v3-70b-instruct"
    LLAMA3_8B = "fireworks_ai/llama-v3-8b-instruct"
