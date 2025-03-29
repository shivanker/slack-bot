import re
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
from llm_utils import generate_title

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


def extract(text):
    url = text.strip()
    match = re.search(r"(https?:[^|>\]]+)", url)
    if match:
        url = match.group(1)
    if is_youtube_video(url):
        logger.warning(f"Fetching youtube transcript for {url}. Original text {text}")
        return yt_transcript(url) or f"Failed to extract transcript for {url}."
    logger.info(f"Reading text from [{url}]. Original text {text}")
    return scrape_text(url) or f"Failed to scrape text from {url}."


class ChatSession:
    def __init__(
        self, user_id: str, channel_id: str, thread_ts: str, client: WebClient
    ):
        self.user_id = user_id
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.client = client
        self.streaming_mode = True
        self.show_thoughts = False
        # Retrieve the sender's information using the Slack API
        sender_info = client.users_info(user=user_id)
        self.user_name = sender_info["user"]["real_name"]
        self.model = TextModel.GEMINI_25
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
        self.say = lambda text: self.client.chat_postMessage(
            channel=self.channel_id, thread_ts=self.thread_ts, text=text
        )

    def fetch_conversation_history(self) -> tuple[list[ChatMessage], list[str]]:
        try:
            conversation_history = self.client.conversations_replies(
                channel=self.channel_id, ts=self.thread_ts, limit=100, inclusive=True
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
            return (merged_messages, commands)

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
        elif cmd == "\\o1":
            self.model = TextModel.O1
            say(text="Model set to O1.")
        elif cmd in ["\\o3-mini", "\\o3mini", "\\mini"]:
            self.model = TextModel.O3_MINI
            say(text="Model set to O3 Mini.")
        elif cmd in ["\\gpt4o", "\\gpt"]:
            self.model = TextModel.GPT_4O
            say(text="Model set to GPT-4o.")
        elif cmd == "\\gpt4":
            self.model = TextModel.GPT_4_TURBO
            say(text="Model set to GPT-4.")
        elif cmd in ["\\llama", "\\llama31", "\\llama405", "\\llama405b"]:
            self.model = TextModel.LLAMA31_405B
            say(text="Model set to LLaMA-3.1 405B.")
        elif cmd in ["\\llama70b", "\\llama70"]:
            self.model = TextModel.LLAMA3_70B
            say(text="Model set to LLaMA-3 70B.")
        # elif cmd in ["\\groq", "\\groq70", "\\groq70b"]:
        #     self.model = TextModel.GROQ_LLAMA3_70B
        #     say(text="Model set to LLaMA 3 70B (Groq).")
        elif cmd in ["\\sonnet", "\\claude"]:
            self.model = TextModel.CLAUDE_37_SONNET
            say(text="Model set to Claude 3.7 Sonnet.")
        elif cmd == "\\haiku":
            self.model = TextModel.CLAUDE_35_HAIKU
            say(text="Model set to Claude 3.5 Haiku.")
        elif cmd == "\\gemini":
            self.model = TextModel.GEMINI_25
            say(text="Model set to Gemini 2.5 Pro.")
        elif cmd == "\\deepseek":
            self.model = TextModel.DEEPSEEK_R1
            say(text="Model set to Deepseek R1.")
        elif cmd == "\\stream":
            self.streaming_mode ^= True
            say(
                text=f'Streaming mode {"enabled" if self.streaming_mode else "disabled"}.'
            )
        elif cmd == "\\nostream":
            self.streaming_mode = False
            say(text="Streaming mode disabled.")
        elif cmd == "\\thoughts":
            self.show_thoughts ^= True
            say(
                text=f'Displaying thoughts {"enabled" if self.show_thoughts else "disabled"}.'
            )
        elif cmd == "\\nothoughts":
            self.show_thoughts = False
            say(text="Displaying thoughts disabled.")
        elif cmd.startswith("\\extract "):
            if say:
                say(text=(extract(cmd[8:]) or "None"))
        elif cmd == "\\help":
            say(
                f"""
{HELP_PREAMBLE} I am a basic chatbot to quickly use GPT4, Claude, LLaMA & Gemini in one place. The chat is organized in sessions. Once you reset a session, all the previous conversation is lost. I am incapable of analyzing images or writing code right now, but feel free to upload PDFs, text files, or link to any websites, and I'll try to scrape whatever text I can. Note that model changes preserve the session so far. Here's the full list of available commands you can use:\n
- \\reset: Reset the chat session. Preserves the previous LLM you were chatting with.\n
- \\who: Returns the name of the chat model you are chatting with.\n
- \\o1: Use O1 for future messages.\n
- \\o3mini: Use O3 Mini for future messages.\n
- \\gpt4o: Use GPT-4o for future messages.\n
- \\sonnet: Use Claude 3.7 Sonnet for future messages.\n
- \\llama: Use LLaMA-3.1 405B for future messages.\n
- \\gemini: Use Gemini 2.5 Pro for future messages.\n
- \\deepseek: Use Deepseek R1 for future messages.\n
- \\stream: Toggle streaming mode. In streaming mode, the bot will send you a message every time it generates a new token.\n
- \\extract: [debug] Extract text from a URL or a YT video.\n
- \\thoughts: Toggle thoughts display. When enabled, thoughts will be shared.\n
                """
            )
        else:
            # say(f"Unknown command: [{cmd}]")
            return False
        return True

    def break_message(self, text: str, max_size: int = 2400) -> list[str]:
        """Split text into chunks of approximately max_size characters, preserving whitespace.
        Attempts to break at newlines first, then spaces if necessary."""
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i : i + max_size]

            # If this isn't the last chunk, try to break at a natural boundary
            if i + max_size < len(text):
                last_newline = chunk.rfind("\n")
                last_space = chunk.rfind(" ")
                # Prefer breaking at newlines, fall back to spaces
                break_at = last_newline if last_newline != -1 else last_space
                if break_at != -1:
                    chunk = chunk[:break_at]
                    i = i + break_at  # Adjust the next starting point
                else:
                    i = i + max_size
            else:
                i = i + max_size

            if chunk.strip():  # Only include non-empty chunks
                chunks.append(chunk)

        return chunks

    def process_direct_message(self, text: str, logger: Any) -> None:
        messages, commands = self.fetch_conversation_history()

        # Re-run previous commands in session
        for cmd in commands[:-1]:
            self.process_command(cmd)

        # Run the latest command, responding if it's the current message
        if self.is_command(text):
            if self.process_command(text, self.say):
                return  # Don't return if command processing failed. Let's process it like a text
        elif commands:
            self.process_command(commands[-1])

        messages_with_instr = [
            msg.to_openai_format()
            for msg in ([ChatMessage.from_user(self.system_instr)] + messages)
        ]
        logger.debug(messages_with_instr)
        extra_completion_params: dict[str, Any] = {
            "max_tokens": 128000,
        }
        if self.model.value.startswith("o"):
            extra_completion_params["reasoning_effort"] = "high"
        elif self.model == TextModel.CLAUDE_37_SONNET:
            extra_completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000,
            }
            extra_completion_params["max_completion_tokens"] = 64000

        self.client.assistant_threads_setStatus(
            channel_id=self.channel_id,
            thread_ts=self.thread_ts,
            status=f"{self.model.value} is generating ...",
        )
        # Process the user's message using the selected model and conversation history
        if not self.streaming_mode:
            response = completion(
                model=self.model.value,
                messages=messages_with_instr,
                **extra_completion_params,
            )
            reasoning_content = response.choices[0].get("reasoning_content", "")  # type: ignore
            full_text: str = response.choices[0].message.content  # type: ignore

            if not self.show_thoughts:
                reasoning_content = ""

            if reasoning_content:
                reasoning_content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n"
                for chunk in self.break_message(reasoning_content):
                    self.say(text=chunk)
            # Send response in chunks
            for chunk in self.break_message(full_text):
                self.say(text=chunk)
            return

        response = completion(
            model=self.model.value,
            messages=messages_with_instr,
            stream=True,
            **extra_completion_params,
        )
        initial_message = self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text=f"[[ {self.model.value} ]] Thinking ...",
        )["ts"]
        last_update_time = time.time()
        update_interval = 2.0  # Start with 2 seconds interval
        start_time = time.time()
        current_message = ""
        currently_thinking = False
        message_ts = initial_message

        for chunk in response:
            last_reasoning_chunk: str = chunk.choices[0].delta.get("reasoning_content", "")  # type: ignore
            last_chunk: str = chunk.choices[0].delta.content or ""  # type: ignore
            if len(last_reasoning_chunk) > 0:
                self.client.assistant_threads_setStatus(
                    channel_id=self.channel_id,
                    thread_ts=self.thread_ts,
                    status=f"{self.model.value} is thinking...",
                )
            else:
                self.client.assistant_threads_setStatus(
                    channel_id=self.channel_id,
                    thread_ts=self.thread_ts,
                    status=f"{self.model.value} is generating...",
                )
            if not self.show_thoughts:
                last_reasoning_chunk = ""
            if not currently_thinking and len(last_reasoning_chunk) > 0:
                currently_thinking = True
                last_reasoning_chunk = f"<thinking>\n{last_reasoning_chunk}"
            if currently_thinking and len(last_reasoning_chunk) == 0:
                currently_thinking = False
                self.client.chat_update(
                    channel=self.channel_id,
                    ts=message_ts,
                    text=f"{current_message}\n</thinking>\n\n",
                )
                # Start a new message for post-thinking response
                message_ts = self.client.chat_postMessage(
                    channel=self.channel_id,
                    thread_ts=self.thread_ts,
                    text=f"... [[ {self.model.value} generating response ]] ...",
                )["ts"]
                current_message = ""

            current_message += last_reasoning_chunk + last_chunk
            current_time = time.time()

            # Check if it's time to send an update or start a new message
            if (
                current_time - last_update_time >= update_interval
                or len(current_message) > 2400
            ):
                # TODO: Not sure if we can have a single big chunk and need to use break_message here
                last_update_time = current_time
                if len(current_message) > 2400:
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=current_message,
                    )
                    # Start a new message with just the new content
                    message_ts = self.client.chat_postMessage(
                        channel=self.channel_id,
                        thread_ts=self.thread_ts,
                        text=f"... [[ {self.model.value} {"thinking" if currently_thinking else "generating"} ]] ...",
                    )["ts"]
                    current_message = ""
                else:
                    # Update existing message
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=f"{current_message} ... [[ {self.model.value} {"thinking" if currently_thinking else "generating"} ]] ...",
                    )

            # Adjust the update interval if the process takes more than 30 seconds
            if current_time - start_time > 60:
                update_interval = 3.0

        # Final update to remove the suffix
        self.client.chat_update(
            channel=self.channel_id, ts=message_ts, text=current_message
        )
        title = generate_title(messages_with_instr)
        self.client.assistant_threads_setTitle(
            channel_id=self.channel_id,
            thread_ts=self.thread_ts,
            title=title,
        )
