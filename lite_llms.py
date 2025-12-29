from enum import Enum


class TextModel(Enum):
    GPT_52 = "gpt-5.2"
    CLAUDE_45_SONNET = "claude-sonnet-4-5"
    CLAUDE_45_HAIKU = "claude-haiku-4-5"
    CLAUDE_45_OPUS = "claude-opus-4-5"
    GEMINI_3_FLASH = "gemini/gemini-3-flash-preview"
    GEMINI_3_PRO = "gemini/gemini-3-pro-preview"
    # GROQ_LLAMA3_70B = "groq/llama3-70b-8192"
    # GROQ_LLAMA3_8B = "groq/llama3-8b-8192"
    # LLAMA3_70B = "fireworks_ai/llama-v3-70b-instruct"
    # LLAMA31_405B = "fireworks_ai/llama-v3p1-405b-instruct"
    DEEPSEEK_R1 = "fireworks_ai/deepseek-r1"
