from typing import Any

import litellm  # type: ignore

from aws_lambda_powertools import Logger
from lite_llms import TextModel
from litellm import completion  # type: ignore
from messages import ChatMessage

logger = Logger()

# Allow litellm to insert empty user msg in claude requests for instance
litellm.modify_params = True


def generate_title(messages: list[dict[str, Any]]) -> str:
    """Generate a title for a chat thread.

    Args:
        messages: A list of messages in openai format.

    Returns:
        A title for the chat thread.
    """
    try:
        messages += [
            ChatMessage.from_user(
                "Generate a title for the above chat thread in less than 7 words."
            ).to_openai_format()
        ]
        response = completion(
            model=TextModel.GEMINI_3_FLASH.value,
            messages=messages,
        )
        return response.choices[0].message.content.strip()  # type: ignore
    except Exception as e:
        logger.exception(f"Failed to generate title for chat thread: {e}")
        return ""
