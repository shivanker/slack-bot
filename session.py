import logging
import os
import requests  # type: ignore
from typing import Any

from haystack.dataclasses import ChatMessage, ChatRole  # type: ignore
from slack_sdk import WebClient

from chat_models import *
from pdf_utils import extract_text_from_pdf

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
ERROR_HEADER = "Something went wrong.\nHere's the traceback for the brave of heart:\n"
logger = logging.getLogger(__name__)

download_cache: dict[str, Any] = {}


def download_file(file_url: str):
    if file_url in download_cache:
        return download_cache[file_url]
    response = requests.get(file_url, headers={"Authorization": f"Bearer {BOT_TOKEN}"})
    if response.status_code != 200:
        raise Exception(f"Error downloading file: {response.status_code}")
    if len(download_cache) > 20:
        oldest_url = next(iter(download_cache))
        del download_cache[oldest_url]
    return response.content


class ChatSession:
    def __init__(self, user_id: str, channel_id: str, client: WebClient):
        self.user_id = user_id
        self.channel_id = channel_id
        self.client = client
        # Retrieve the sender's information using the Slack API
        sender_info = client.users_info(user=user_id)
        self.user_name = sender_info["user"]["real_name"]
        self.model = CLAUDE_3_SONNET
        self.system_instr = (
            "You are a helpful assistant called SushiBot running as a Slack App. Keep the "
            "conversation natural and flowing, don't respond with robotic or closing statements like "
            "'Is there anything else?'. You are a friend, not a bot. "
            "Whatever you say will be sent back as a text to the user. Feel free to use rich text "
            "formatting appropriate for the Slack API. "
            # If you don't know something, look it up on the \
            # internet. If Search results are not useful, try to navigate to known expert \
            # websites to fetch real, up-to-date data, and then root your answers to those facts."
            "Here goes the chat history so far and the latest activity..."
        )
        self.say = None

    def fetch_conversation_history(self):
        try:
            conversation_history = self.client.conversations_history(
                channel=self.channel_id, limit=50
            )
            messages = conversation_history["messages"]

            history = []
            for message in messages:
                text = message.get("text")
                sent_by_user = message.get("user") == self.user_id
                if text:
                    if text.startswith("\\") and " " not in text:
                        # This message was a command, exclude it (and the response) from chat history
                        history.pop()
                        if text == "\\reset":
                            break
                        continue
                    elif text.startswith(ERROR_HEADER):
                        history.append(ChatMessage.from_assistant("<Unknown Error />"))
                        continue
                    else:
                        history.append(
                            ChatMessage.from_user(text)
                            if sent_by_user
                            else ChatMessage.from_assistant(text)
                        )
                files = message.get("files", [])
                for file in files:
                    logger.debug(f"Files:\n{file}")
                    msg = None
                    if file.get("mimetype", "").startswith("image/"):
                        msg = f"<Image name:{file['name']}/>"
                        file_url = file["url_private"]
                        logger.error("Found image attachment.")
                    elif file.get("mimetype", "") == "text/plain":
                        file_url = file["url_private"]
                        content = download_file(file_url)
                        msg = f"<File mimetype={file['mimetype']}>\n{content}\n</File>"
                    elif file.get("mimetype", "") == "application/pdf":
                        file_url = file["url_private"]
                        msg = f"<File mimetype={file['mimetype']}>\n{extract_text_from_pdf(file_url)}\n</File>"
                    else:
                        msg = f"<File name={file['name']}/>"
                    if msg:
                        history.append(
                            ChatMessage.from_user(msg)
                            if sent_by_user
                            else ChatMessage.from_assistant(msg)
                        )

            history = list(reversed(history))
            # Ensure first message is from user
            if history and not history[0].is_from(ChatRole.USER):
                history = [ChatMessage.from_user("...")] + history

            # Merge consecutive user messages into one
            merged_messages = []
            prev_role = None
            for msg in history:
                if msg.is_from(prev_role):  # type: ignore
                    merged_messages[-1].content += "\n" + msg.content
                else:
                    merged_messages.append(msg)
                    prev_role = msg.role
            logger.debug(f"<history>\n{merged_messages}</history>")
            return merged_messages

        except Exception as e:
            print(f"Error fetching conversation history: {str(e)}")
            return []

    def process_direct_message(self, text, say, logger):
        self.say = say
        if text == "\\reset":
            say(text="Session has been reset.")
        elif text == "\\gpt4":
            self.model = GPT_4_TURBO
            say(text="Model set to GPT-4.")
        elif text == "\\llama70b":
            self.model = LLAMA3_70B
            say(text="Model set to LLaMA 3 70B.")
        elif text == "\\llama8b":
            self.model = LLAMA3_8B
            say(text="Model set to LLaMA 3 8B.")
        elif text == "\\opus":
            self.model = CLAUDE_3_OPUS
            say(text="Model set to Claude 3 Opus.")
        elif text == "\\sonnet":
            self.model = CLAUDE_3_SONNET
            say(text="Model set to Claude 3 Sonnet.")
        elif text == "\\haiku":
            self.model = CLAUDE_3_HAIKU
            say(text="Model set to Claude 3 Haiku.")
        elif text == "\\gemini":
            self.model = GEMINI_15_PRO
            say(text="Model set to Gemini 1.5 Pro.")
        else:
            messages = self.fetch_conversation_history()
            messages = [ChatMessage.from_system(self.system_instr)] + messages
            logger.debug(messages)
            # Process the user's message using the selected model and conversation history
            # Implement your message processing logic here
            response = self.model.run(messages)
            for reply in response["replies"]:
                say(text=reply.content)
