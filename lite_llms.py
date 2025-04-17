from enum import Enum


class TextModel(Enum):
    GPT_41 = "gpt-4.1"
    GPT_4_TURBO = "gpt-4-turbo"
    O3 = "o3"
    O4_MINI = "o4-mini"
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"
    GEMINI_2_FLASH = "gemini/gemini-2.0-flash"
    GEMINI_25 = "gemini/gemini-2.5-pro-exp-03-25"
    # GROQ_LLAMA3_70B = "groq/llama3-70b-8192"
    # GROQ_LLAMA3_8B = "groq/llama3-8b-8192"
    LLAMA3_70B = "fireworks_ai/llama-v3-70b-instruct"
    LLAMA31_405B = "fireworks_ai/llama-v3p1-405b-instruct"
    DEEPSEEK_R1 = "fireworks_ai/deepseek-r1"
