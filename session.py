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

# Agno imports
from agno.agent.agent import Agent
from agno.models.base import Model
from agno.models.anthropic.claude import Claude
from agno.models.openai.chat import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.youtube import YouTubeTools
from agno.tools.calculator import CalculatorTools
from agno.tools.thinking import ThinkingTools
from agno.storage.dynamodb import DynamoDbStorage

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
        self.debug_mode = False
        self.agent: str = ""
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
        # Initialize Agno storage
        self.agno_storage = DynamoDbStorage(
            table_name="agent_slackbot_agno_sessions",
            region_name="us-east-1",
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
                            elements = block.get("elements", [])
                            for element in elements:
                                inner_elements = element.get("elements", [])
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

                                        content = None
                                        tag = None
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
                    msg_content = None
                    mimetype = file.get("mimetype", "")
                    file_url = file.get("url_private")
                    file_name = file.get("name", "Unknown File")
                    logger.info(f"Found file [{file_name}] of type [{mimetype}].")
                    if mimetype.startswith("image/"):
                        msg_content = f"<Image name:{file_name}/>"
                        # TODO: Images are not supported yet.
                        logger.error("Found image attachment.")
                    elif mimetype == "text/plain":
                        content = download_file(file_url).decode(
                            "utf-8", errors="replace"
                        )
                        msg_content = f"<File name='{file_name}' mimetype='{file['mimetype']}'>\n{content}\n</File>"
                    elif mimetype == "application/pdf":
                        msg_content = f"<File name='{file_name}' mimetype='{mimetype}'>\n{extract_text_from_pdf(file_url)}\n</File>"
                    else:
                        msg_content = f"<File name={file_name}/>"
                    if msg_content:
                        history.append(
                            ChatMessage.from_user(msg_content)
                            if sent_by_user
                            else ChatMessage.from_assistant(msg_content)
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

    def process_command(self, text: str, say=lambda text: None) -> bool:
        """Processes a command string.
        Args:
            text: The command string (e.g., "\\reset").
            say: A function to send a response back to the user (optional).
        Returns:
            True if the text was a known command and processed, False otherwise.
        """
        cmd = text.strip()
        if cmd == "\\reset":
            say(text="Session has been reset.")
        elif cmd in ("\\who?", "\\who", "\\llm", "\\model"):
            say(text=f"You are currently chatting with {self.model.value}.")
        elif cmd == "\\o3":
            self.model = TextModel.O3
            say(text="Model set to O3.")
        elif cmd in ["\\o4-mini", "\\o4mini", "\\mini"]:
            self.model = TextModel.O4_MINI
            say(text="Model set to O4 Mini.")
        elif cmd in ["\\gpt41", "\\gpt"]:
            self.model = TextModel.GPT_41
            say(text="Model set to GPT-4.1.")
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
                url_to_extract = cmd[len("\\extract ") :].strip()
                extracted_content = (
                    extract(url_to_extract) or "Failed to extract content."
                )
                say(text=extracted_content)
        elif cmd == "\\debug":
            self.debug_mode ^= True
            say(text=f'Debug mode {"enabled" if self.debug_mode else "disabled"}.')
        elif cmd == "\\agno-sonnet":
            self.model = TextModel.CLAUDE_37_SONNET
            self.agent = "agno"
            say(text="Model set to Agno with Claude 3.7 Sonnet.")
        elif cmd == "\\help":
            say(
                f"""
{HELP_PREAMBLE} I am a basic chatbot to quickly use GPT4, Claude, LLaMA & Gemini in one place. The chat is organized in sessions. Once you reset a session, all the previous conversation is lost. I am incapable of analyzing images or writing code right now, but feel free to upload PDFs, text files, or link to any websites, and I'll try to scrape whatever text I can. Note that model changes preserve the session so far. Here's the full list of available commands you can use:\n
- \\reset: Reset the chat session. Preserves the previous LLM you were chatting with.\n
- \\who: Returns the name of the chat model you are chatting with.\n
- \\o3: Use O3 for future messages.\n
- \\o4mini: Use O4 Mini for future messages.\n
- \\gpt41: Use GPT-4.1 for future messages.\n
- \\sonnet: Use Claude 3.7 Sonnet for future messages.\n
- \\agno-sonnet: Use Agno agent with Claude 3.7 Sonnet for future messages.\n
- \\llama: Use LLaMA-3.1 405B for future messages.\n
- \\gemini: Use Gemini 2.5 Pro for future messages.\n
- \\deepseek: Use Deepseek R1 for future messages.\n
- \\stream: Toggle streaming mode. In streaming mode, the bot will send you a message every time it generates a new token.\n
- \\extract: [debug] Extract text from a URL or a YT video.\n
- \\thoughts: Toggle thoughts display. When enabled, thoughts will be shared.\n
- \\debug: Toggle debug mode. When enabled, raw agent responses will be shown.\n
                """
            )
        else:
            # say(f"Unknown command: [{cmd}]")
            return False
        return True

    def break_message(self, text: str, max_size: int = 2400) -> list[str]:
        """Split text into chunks of approximately max_size characters, preserving whitespace.
        Attempts to break at newlines first, then spaces, to maintain readability.
        Avoids breaking mid-word if possible. Preserves whitespace.
        Args:
            text: The text to split.
            max_size: The approximate maximum size for each chunk.
        Returns:
            A list of text chunks.
        """
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

    def _get_completion_params(self) -> dict[str, Any]:
        """Determines extra parameters for the litellm.completion call based on the model."""
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
        return extra_completion_params

    def _handle_non_streaming_response(
        self, messages_with_instr: list[dict], extra_completion_params: dict
    ) -> None:
        """Handles the response from the LLM when streaming is disabled."""
        response = completion(
            model=self.model.value,
            messages=messages_with_instr,
            **extra_completion_params,
        )
        reasoning_content = response.choices[0].get("reasoning_content", "") if self.show_thoughts else ""  # type: ignore
        full_text: str = response.choices[0].message.content or ""  # type: ignore

        if reasoning_content:
            formatted_reasoning = f"<thinking>\n{reasoning_content}\n</thinking>\n\n"
            for chunk in self.break_message(formatted_reasoning):
                self.say(text=chunk)

        # Send the main response content in chunks
        for chunk in self.break_message(full_text):
            self.say(text=chunk)

    def _handle_streaming_response(
        self, messages_with_instr: list[dict], extra_completion_params: dict
    ) -> None:
        """Handles the response from the LLM when streaming is enabled."""
        response_stream = completion(
            model=self.model.value,
            messages=messages_with_instr,
            stream=True,
            **extra_completion_params,
        )

        # Post initial message and track updates
        message_ts = self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text=f"[[ {self.model.value} ]] Thinking ...",
        )["ts"]

        last_update_time = time.time()
        update_interval = 2.0  # Start with 2 seconds interval, adjust later
        start_time = time.time()
        current_message = ""
        currently_thinking = False  # Track if the current chunk is part of 'thinking'

        for chunk in response_stream:
            delta = chunk.choices[0].delta  # type: ignore
            last_reasoning_chunk: str = delta.get("reasoning_content", "")  # type: ignore
            last_chunk: str = delta.content or ""  # type: ignore

            # Update Slack thread status based on whether reasoning or text is received
            if last_reasoning_chunk:
                # Regardless of show_thoughts, we want to show the thinking status.
                self._set_chat_status(f"{self.model.value} is thinking...")
            else:
                self._set_chat_status(f"{self.model.value} is generating...")
            if not self.show_thoughts:
                last_reasoning_chunk = ""

            # Handle transitions in & out of thinking
            if not currently_thinking and last_reasoning_chunk:
                # Started thinking
                currently_thinking = True
                last_reasoning_chunk = f"<thinking>\n{last_reasoning_chunk}"
            elif currently_thinking and not last_reasoning_chunk:
                currently_thinking = False
                # Close the thinking tag
                self.client.chat_update(
                    channel=self.channel_id,
                    ts=message_ts,
                    text=f"{current_message}\n</thinking>\n\n",
                )
                # Start a *new* message for the main response content
                message_ts = self.client.chat_postMessage(
                    channel=self.channel_id,
                    thread_ts=self.thread_ts,
                    text=f"... [[ {self.model.value} generating response ]] ...",
                )["ts"]
                current_message = ""  # Reset content for the new message

            # Append the current chunk (reasoning or text)
            current_message += last_reasoning_chunk + last_chunk
            current_time = time.time()

            # Throttle Slack updates: Update message if interval passed or message is long
            if (
                current_time - last_update_time >= update_interval
                or len(current_message) > 2400
            ):
                # TODO: Not sure if we can have a single big chunk and need to use break_message here
                last_update_time = current_time
                # If message exceeds limit, finalize current message and start a new one
                if len(current_message) > 2400:
                    # TODO: Use break_message logic here? For now, just post and start new.
                    # This might slightly exceed the limit if the last chunk pushes it over.
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=current_message,  # Post the full content before starting new
                    )
                    # Start a new message
                    status_indicator = (
                        "thinking" if currently_thinking else "generating"
                    )
                    message_ts = self.client.chat_postMessage(
                        channel=self.channel_id,
                        thread_ts=self.thread_ts,
                        text=f"... [[ {self.model.value} {status_indicator} ]] ...",
                    )["ts"]
                    current_message = ""  # Reset content for the new message
                else:
                    # Update the existing message with a progress indicator
                    status_indicator = (
                        "thinking" if currently_thinking else "generating"
                    )
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=f"{current_message} ... [[ {self.model.value} {status_indicator} ]] ...",
                    )

            # Adjust update interval if generation is taking a long time
            if current_time - start_time > 60:
                update_interval = 3.0

        # Final update to remove the suffix
        if currently_thinking:
            current_message += "\n</thinking>"
        self.client.chat_update(
            channel=self.channel_id, ts=message_ts, text=current_message
        )

    def _init_agno_agent(self) -> Agent:
        """Initialize the Agno agent with the appropriate configuration."""
        # Get a unique user id for the agent
        user_identity = self.client.users_identity()
        unique_user_id: str = user_identity.get("user", {}).get("id", self.user_id)  # type: ignore

        model: Model = Claude(id=TextModel.CLAUDE_37_SONNET.value)
        if self.model.value.startswith("claude"):
            model = Claude(id=self.model.value)
        elif self.model.value.startswith("gemini"):
            model = Gemini(id=self.model.value[len("gemini/") :], vertexai=True)
        elif self.model.value.startswith("gpt"):
            model = OpenAIChat(id=self.model.value)
        elif self.model.value.startswith("o"):
            model = OpenAIChat(id=self.model.value, reasoning_effort="high")
        else:
            self.say(text=f"Unknown Agno model: {self.model.value}, using Sonnet 3.7.")
        # Initialize the agent with Claude Sonnet model
        agent = Agent(
            model=model,
            user_id=unique_user_id,
            session_id=self.thread_ts,  # Use thread_ts as session_id
            description="You are a helpful agent running in a Slack bot.",
            tools=[
                DuckDuckGoTools(),
                YFinanceTools(enable_all=True),
                YouTubeTools(),
                CalculatorTools(enable_all=True),
                ThinkingTools(),
                # Crawl4aiTools(max_length=None),
            ],
            add_datetime_to_instructions=True,
            show_tool_calls=True,
            tool_call_limit=25,
            storage=self.agno_storage,
            add_history_to_messages=True,
            # reasoning=True,
            telemetry=False,
        )
        return agent

    def _handle_agno_streaming_response(self, agent: Agent, query: str) -> None:
        """Handles streaming responses from the Agno agent."""
        # Post initial message and track updates
        message_ts = self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text="[[ Agno Claude 3.7 Sonnet ]] Processing ...",
        )["ts"]

        last_update_time = time.time()
        update_interval = 2.0  # Start with 2 seconds interval
        start_time = time.time()
        current_message = ""

        for response in agent.run(query, stream=True):
            # If in debug mode, show the raw response object
            if self.debug_mode:
                chunk = str(response) + "\n"
            else:
                # Otherwise just show the content
                chunk = response.content or ""

            # Update Slack thread status
            self._set_chat_status("Agno agent is processing...")

            # Append the current chunk
            current_message += chunk
            current_time = time.time()

            # Throttle Slack updates: Update message if interval passed or message is long
            if (
                current_time - last_update_time >= update_interval
                or len(current_message) > 2400
            ):
                last_update_time = current_time
                # If message exceeds limit, finalize current message and start a new one
                if len(current_message) > 2400:
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=current_message,
                    )
                    # Start a new message
                    message_ts = self.client.chat_postMessage(
                        channel=self.channel_id,
                        thread_ts=self.thread_ts,
                        text="... [[ Agno agent continuing ]] ...",
                    )["ts"]
                    current_message = ""  # Reset content for the new message
                else:
                    # Update the existing message with a progress indicator
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=message_ts,
                        text=f"{current_message} ... [[ Agno agent processing ]] ...",
                    )

            # Adjust update interval if processing is taking a long time
            if current_time - start_time > 60:
                update_interval = 3.0

        # Final update to remove the suffix
        self.client.chat_update(
            channel=self.channel_id, ts=message_ts, text=current_message
        )

    def _handle_agno_non_streaming_response(self, agent: Agent, query: str) -> None:
        """Handles non-streaming responses from the Agno agent."""
        response = agent.run(query, stream=False)

        # If in debug mode, show the raw response object
        if self.debug_mode:
            full_text = str(response)
        else:
            # Otherwise just show the content
            full_text = response.content or ""

        # Send the main response content in chunks
        for chunk in self.break_message(full_text):
            self.say(text=chunk)

    def _generate_from_agno(self, text: str, logger: Any) -> None:
        """
        Generates a response from the configured LLM using the provided messages.

        Handles both streaming and non-streaming modes, updates Slack status,
        and sets the thread title.

        Args:
            messages_with_instr: The list of messages formatted for the LLM API.
            logger: The logger instance.
        """
        logger.debug(f"Generating response using model: {self.model.value}")

        # Initialize Agno agent
        agent = self._init_agno_agent()

        # Generate response based on streaming mode
        if self.streaming_mode:
            self._handle_agno_streaming_response(agent, text)
        else:
            self._handle_agno_non_streaming_response(agent, text)

    def _generate_from_model(
        self, messages_with_instr: list[dict], logger: Any
    ) -> None:
        """
        Generates a response from the configured LLM using the provided messages.

        Handles both streaming and non-streaming modes, updates Slack status,
        and sets the thread title.

        Args:
            messages_with_instr: The list of messages formatted for the LLM API.
            logger: The logger instance.
        """
        logger.debug(f"Generating response using model: {self.model.value}")
        extra_completion_params = self._get_completion_params()

        # Generate response based on streaming mode
        if self.streaming_mode:
            self._handle_streaming_response(
                messages_with_instr, extra_completion_params
            )
        else:
            self._handle_non_streaming_response(
                messages_with_instr, extra_completion_params
            )

    def _set_chat_status(self, status: str) -> None:
        """Sets the chat status in Slack."""
        try:
            self.client.assistant_threads_setStatus(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                status=status,
            )
        except Exception as e:
            logger.warning(f"Could not set thread status: {e}")  # Non-fatal

    def _set_thread_title(self, messages_with_instr: list[dict]) -> None:
        """Sets the thread title based on the messages."""
        try:
            title = generate_title(messages_with_instr)
            self.client.assistant_threads_setTitle(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                title=title,
            )
            logger.info(f"Set thread title to: {title}")
        except Exception as e:
            logger.error(f"Failed to generate or set thread title: {e}")

    def process_direct_message(self, text: str, logger: Any) -> None:
        """Processes an incoming direct message or mention in a thread.

        Fetches history, handles commands, generates a response using the LLM,
        and sends the response back to Slack.

        Args:
            text: The text content of the incoming Slack message.
            logger: The logger instance.
        """
        # 1. Fetch conversation history and past commands
        messages, commands = self.fetch_conversation_history()

        # 2. Re-apply state changes from previous commands in the session
        # (e.g., model changes, streaming mode)
        # We skip the last command if the current `text` is that command.
        commands_to_replay = commands
        if self.is_command(text) and commands and commands[-1] == text:
            commands_to_replay = commands[:-1]

        for cmd in commands_to_replay:
            self.process_command(cmd)

        # 3. Process the current message if it's a command
        if self.is_command(text):
            # Process the command and send a response back.
            if self.process_command(text, self.say):
                return
            # If process_command returned False, it's an unknown command.
            # We'll treat it as regular text input for the LLM below.
            logger.info(f"Unknown command '{text}', treating as text input.")

        # Set initial status in Slack thread
        self._set_chat_status(
            f"{self.agent}[{self.model.value}] is generating ..."
            if self.agent == "agno"
            else f"{self.model.value} is generating ..."
        )

        # 4. Prepare messages for the LLM API
        # Combine system instructions with the fetched/merged message history
        messages_with_instr = [
            msg.to_openai_format()
            for msg in ([ChatMessage.from_user(self.system_instr)] + messages)
        ]
        logger.debug(f"Messages being sent to LLM:\n{messages_with_instr}")

        # 5. Generate and send the response using the LLM
        if self.agent == "agno":
            self._generate_from_agno(text, logger)
        else:
            self._generate_from_model(messages_with_instr, logger)

        # 6. Generate and set the thread title after the response is complete
        self._set_thread_title(messages_with_instr)
