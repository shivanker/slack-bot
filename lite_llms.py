from enum import Enum


class TextModel(Enum):
    GPT_35 = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    GEMINI_15_PRO = "gemini/gemini-1.5-pro-latest"
    GEMINI_15_FLASH = "gemini/gemini-1.5-flash-latest"
    # GROQ_LLAMA3_70B = "groq/llama3-70b-8192"
    # GROQ_LLAMA3_8B = "groq/llama3-8b-8192"
    LLAMA3_70B = "fireworks_ai/llama-v3-70b-instruct"
    LLAMA31_405B = "fireworks_ai/llama-v3p1-405b-instruct"
    LLAMA31_8B = "fireworks_ai/llama-v3p1-8b-instruct"
