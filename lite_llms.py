from enum import Enum


class TextModel(Enum):
    GPT_35 = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"
    GEMINI_2_PRO = "gemini/gemini-exp-1206"
    GEMINI_2_FLASH = "gemini/gemini-2.0-flash-exp"
    GEMINI_FLASH_THINKING = "gemini/gemini-2.0-flash-thinking-exp-01-21"
    # GROQ_LLAMA3_70B = "groq/llama3-70b-8192"
    # GROQ_LLAMA3_8B = "groq/llama3-8b-8192"
    LLAMA3_70B = "fireworks_ai/llama-v3-70b-instruct"
    LLAMA31_405B = "fireworks_ai/llama-v3p1-405b-instruct"
    LLAMA31_8B = "fireworks_ai/llama-v3p1-8b-instruct"
    DEEPSEEK_R1 = "fireworks_ai/deepseek-r1"
