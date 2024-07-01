import os
import time
from typing import Any

import litellm  # type: ignore
import requests  # type: ignore

from aws_lambda_powertools import Logger
from lite_llms import TextModel
from litellm import completion  # type: ignore
from messages import ChatMessage, ChatRole
from slack_sdk import WebClient

from pdf_utils import extract_text_from_pdf
from web_reader import scrape_text
from ytsubs import is_youtube_video, yt_transcript

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
ERROR_HEADER = "Something went wrong.\nHere's the traceback for the brave of heart:\n"
HELP_PREAMBLE = "Welcome to SushiBot."
logger = Logger()

################
## Global config
################

# Allow litellm to insert empty user msg in claude requests for instance
litellm.modify_params = True


download_cache: dict[str, bytes] = {}


def download_file(file_url: str):
    if file_url in download_cache:
        return download_cache[file_url]
    response = requests.get(file_url, headers={"Authorization": f"Bearer {BOT_TOKEN}"})
    if response.status_code != 200:
        raise Exception(f"Error downloading file: {response.status_code}")
    download_cache[file_url] = response.content
    if len(download_cache) > 20:
        oldest_url = next(iter(download_cache))
        del download_cache[oldest_url]
    return download_cache[file_url]


def check_mimetype(url) -> str:
    try:
        response = requests.head(url)
        return response.headers.get("Content-Type", "unknown")
    except requests.exceptions.RequestException:
        return "unknown"


class ChatSession:
    def __init__(self, user_id: str, channel_id: str, client: WebClient):
        self.user_id = user_id
        self.channel_id = channel_id
        self.client = client
        self.streaming_mode = True
        # Retrieve the sender's information using the Slack API
        sender_info = client.users_info(user=user_id)
        self.user_name = sender_info["user"]["real_name"]
        self.model = TextModel.CLAUDE_35_SONNET
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

    def fetch_conversation_history(self) -> tuple[list[ChatMessage], list[str]]:
        try:
            conversation_history = self.client.conversations_history(
                channel=self.channel_id, limit=50
            )
        except Exception as e:
            logger.error(f"Error fetching conversation history: {str(e)}")
            raise e
        try:
            messages = conversation_history["messages"]

            history: list[ChatMessage] = []
            commands: list[str] = []
            for message in messages:
                text = message.get("text")
                sent_by_user = message.get("user") == self.user_id
                if text:
                    if self.is_command(text):
                        # Exclude command's response from chat history
                        if history:
                            history.pop()
                        commands.append(text)
                        if text == "\\reset":
                            break
                        continue
                    elif text.startswith(ERROR_HEADER):
                        history.append(ChatMessage.from_assistant("<Unknown Error />"))
                        continue
                    elif text.startswith(HELP_PREAMBLE):
                        continue
                    else:
                        history.append(
                            ChatMessage.from_user(text)
                            if sent_by_user
                            else ChatMessage.from_assistant(text)
                        )
                    # Append the content of URLs to this text
                    if sent_by_user:
                        # ["blocks"][0]["elements"][0]["elements"][1]["url"]
                        blocks = message.get("blocks")
                        for block in blocks:
                            elements = block.get("elements")
                            for element in elements:
                                inner_elements = element.get("elements")
                                for unit in inner_elements:
                                    if unit.get("type") == "link":
                                        url = unit.get("url")
                                        if not url:
                                            continue

                                        mimetype = check_mimetype(url)
                                        logger.info(
                                            f"Found link [{url}] of type [{mimetype}]."
                                        )
                                        if mimetype.startswith(
                                            "image/"
                                        ) or mimetype in [
                                            "text/plain",
                                            "application/pdf",
                                        ]:
                                            message.setdefault("files", []).append(
                                                {
                                                    "name": url,
                                                    "url_private": url,
                                                    "mimetype": mimetype,
                                                }
                                            )
                                            continue

                                        if is_youtube_video(url):
                                            logger.debug(
                                                f"Fetching youtube transcript for [{url}]."
                                            )
                                            content = yt_transcript(url)
                                            tag = "YoutubeTranscript"
                                        else:
                                            logger.debug(f"Reading text from [{url}].")
                                            content = scrape_text(url)
                                            tag = "ScrapedTextFromURL"
                                        if content:
                                            history.append(
                                                ChatMessage.from_user(
                                                    f"<{tag} url={url}>\n{content}\n</{tag}>"
                                                )
                                            )

                files = message.get("files", [])
                for file in files:
                    logger.debug(f"Files:\n{file}")
                    msg = None
                    mimetype = file.get("mimetype", "")
                    logger.info(f"Found file [{file['name']}] of type [{mimetype}].")
                    if mimetype.startswith("image/"):
                        msg = f"<Image name:{file['name']}/>"
                        file_url = file["url_private"]
                        logger.error("Found image attachment.")
                    elif mimetype == "text/plain":
                        file_url = file["url_private"]
                        content = download_file(file_url)
                        msg = f"<File mimetype={file['mimetype']}>\n{content}\n</File>"
                    elif mimetype == "application/pdf":
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
            merged_messages: list[ChatMessage] = []
            prev_role = None
            for chatmsg in history:
                if chatmsg.is_from(prev_role):  # type: ignore
                    merged_messages[-1].content += "\n" + chatmsg.content
                else:
                    merged_messages.append(chatmsg)
                    prev_role = chatmsg.role
            logger.debug(f"<history>\n{merged_messages}</history>")
            return (merged_messages, list(reversed(commands)))

        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            raise e

    def is_command(self, text):
        if not isinstance(text, str):
            return False
        cmd = text.strip()
        return cmd.startswith("\\")

    def process_command(self, text, say=lambda text: None):
        cmd = text.strip()
        if cmd == "\\reset":
            say(text="Session has been reset.")
        elif cmd in ("\\who?", "\\who", "\\llm", "\\model"):
            say(text=f"You are currently chatting with {self.model.value}.")
        elif cmd in ["\\gpt4o", "\\gpt"]:
            self.model = TextModel.GPT_4O
            say(text="Model set to GPT-4o (Omni).")
        elif cmd == "\\gpt4":
            self.model = TextModel.GPT_4_TURBO
            say(text="Model set to GPT-4.")
        elif cmd in ["\\llama70b", "\\llama70", "\\llama"]:
            self.model = TextModel.LLAMA3_70B
            say(text="Model set to LLaMA 3 70B.")
        elif cmd in ["\\llama8b", "\\llama8"]:
            self.model = TextModel.LLAMA3_8B
            say(text="Model set to LLaMA 3 8B.")
        elif cmd in ["\\groq", "\\groq70", "\\groq70b"]:
            self.model = TextModel.GROQ_LLAMA3_70B
            say(text="Model set to LLaMA 3 70B (Groq).")
        elif cmd == "\\opus":
            self.model = TextModel.CLAUDE_3_OPUS
            say(text="Model set to Claude 3 Opus.")
        elif cmd == "\\sonnet":
            self.model = TextModel.CLAUDE_35_SONNET
            say(text="Model set to Claude 3.5 Sonnet.")
        elif cmd == "\\haiku":
            self.model = TextModel.CLAUDE_3_HAIKU
            say(text="Model set to Claude 3 Haiku.")
        elif cmd == "\\gemini":
            self.model = TextModel.GEMINI_15_PRO
            say(text="Model set to Gemini 1.5 Pro.")
        elif cmd == "\\flash":
            self.model = TextModel.GEMINI_15_FLASH
            say(text="Model set to Gemini 1.5 Flash.")
        elif cmd == "\\stream":
            self.streaming_mode ^= True
            say(
                text=f'Streaming mode {"enabled" if self.streaming_mode else "disabled"}.'
            )
        elif cmd == "\\nostream":
            self.streaming_mode = False
            say(text="Streaming mode disabled.")
        elif cmd == "\\help":
            say(
                f"""
{HELP_PREAMBLE} I am a basic chatbot to quickly use GPT4, Claude, LLaMA & Gemini in one place. The chat is organized in sessions. Once you reset a session, all the previous conversation is lost. I am incapable of analyzing images or writing code right now, but feel free to upload PDFs, text files, or link to any websites, and I'll try to scrape whatever text I can. Here's the full list of available commands you can use:\n
- \\reset: Reset the chat session. Preserves the previous LLM you were chatting with.\n
- \\who: Returns the name of the chat model you are chatting with.\n
- \\gpt4: Use GPT-4o (Omni) for future messages. Preserves the session so far.\n
- \\gpt4: Use GPT-4 Turbo for future messages. Preserves the session so far.\n
- \\llama70: Use LLaMA-3-70B for future messages. Preserves the session so far.\n
- \\groq70: Use LLaMA-3-70B (served by Groq - faster but lower token limit) for future messages. Preserves the session so far.\n
- \\opus: Use Claude 3 Opus for future messages. Preserves the session so far.\n
- \\sonnet: Use Claude 3.5 Sonnet for future messages. Preserves the session so far.\n
- \\haiku: Use Claude 3 Haiku for future messages. Preserves the session so far.\n
- \\gemini: Use Gemini 1.5 Pro for future messages. Preserves the session so far.\n
- \\stream: Toggle streaming mode. In streaming mode, the bot will send you a message every time it generates a new token.\n
                """
            )
        else:
            # say(f"Unknown command: [{cmd}]")
            return False
        return True

    def process_direct_message(self, text, say, logger):
        self.say = say

        messages, commands = self.fetch_conversation_history()
        if len(messages) < 2 and len(commands) == 0 and not self.is_command(text):
            say(
                HELP_PREAMBLE
                + ' At any time, enter "\\help" for a list of commands. Response to your first message will follow now.'
            )

        # Re-run previous commands in session
        for cmd in commands[:-1]:
            self.process_command(cmd)

        # Run the latest command, responding if it's the current message
        if self.is_command(text):
            if self.process_command(text, say):
                return  # Don't return if command processing failed. Let's process it like a text
        elif commands:
            self.process_command(commands[-1])

        messages = [ChatMessage.from_system(self.system_instr)] + messages
        messages = [msg.to_openai_format() for msg in messages]
        logger.debug(messages)

        # Process the user's message using the selected model and conversation history
        if not self.streaming_mode:
            response = completion(model=self.model.value, messages=messages)
            say(text=response.choices[0].message.content)  # type: ignore
            return

        response = completion(model=self.model.value, messages=messages, stream=True)
        initial_message = self.client.chat_postMessage(
            channel=self.channel_id, text=f"[[ {self.model.value} ]] Thinking ..."
        )["ts"]
        last_update_time = time.time()
        update_interval = 1  # Start with 1 second interval
        start_time = time.time()
        full_response = ""
        current_message = ""
        message_ts = initial_message

        for part in response:
            last_chunk = part.choices[0].delta.content or ""  # type: ignore
            full_response += last_chunk
            current_message += last_chunk
            current_time = time.time()

            # Check if it's time to send an update or start a new message
            if (
                current_time - last_update_time >= update_interval
                or len((current_message).split()) > 320
                or len((current_message)) > 2400
            ):
                # Update existing message
                self.client.chat_update(
                    channel=self.channel_id,
                    ts=message_ts,
                    text=f"{current_message} ... [[ {self.model.value} thinking ]] ...",
                )
                if len(current_message.split()) > 320 or len(current_message) > 2400:
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=f"{current_message} ... [[ continued below ]] ...",
                    )
                    # Start a new message with just the new content
                    message_ts = self.client.chat_postMessage(
                        channel=self.channel_id,
                        text=f"... [[ {self.model.value} thinking ]] ...",
                    )["ts"]
                    current_message = ""
                last_update_time = current_time

            # Adjust the update interval if the process takes more than 30 seconds
            if current_time - start_time > 30:
                update_interval = 2
            if current_time - start_time > 90:
                update_interval = 2.5

        # Final update to remove the suffix
        self.client.chat_update(
            channel=self.channel_id, ts=message_ts, text=current_message
        )
