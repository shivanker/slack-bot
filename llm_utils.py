from typing import Any

import litellm  # type: ignore

from aws_lambda_powertools import Logger
from lite_llms import TextModel
from litellm import completion  # type: ignore
from messages import ChatMessage

logger = Logger()

# Allow litellm to insert empty user msg in claude requests for instance
litellm.modify_params = True


def generate_title(messages: list[ChatMessage]) -> str:
    """Generate a title for a chat thread.

    Args:
        messages: A list of messages in openai format.

    Returns:
        A title for the chat thread.
    """
    try:
        messages = [
            ChatMessage.from_user(
                "I will give you a chat thread below,\n"
                "and your job is to generate a title for it in less than 7 words.\n"
                "Only respond with the title of the chat thread, nothing else.\n"
                "Here goes the chat thread below.\n\n"
            )
        ] + messages
        response = completion(
            model=TextModel.GEMINI_3_FLASH.value,
            messages=messages,
        )
        return response.choices[0].message.content.strip()  # type: ignore
    except Exception as e:
        logger.exception(f"Failed to generate title for chat thread: {e}")
        return ""
