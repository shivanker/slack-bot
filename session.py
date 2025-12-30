import asyncio
import re
import os
import time
from typing import Any

import litellm  # type: ignore
import requests  # type: ignore
import boto3  # type: ignore

from aws_lambda_powertools import Logger
from lite_llms import TextModel
from litellm import completion  # type: ignore
from messages import ChatMessage, ChatRole
from slack_sdk import WebClient

from pdf_utils import extract_text_from_pdf
from web_reader import scrape_text
from ytsubs import is_youtube_video, yt_transcript
from llm_utils import generate_title
from adk_agents import get_agent, list_agents
from adk_agents.utils import run_agent_async

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
settings_table = boto3.resource("dynamodb").Table("slackbot_user_settings")  # type: ignore


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
        self, user_id: str, channel_id: str, thread_ts: str, client: WebClient, logger: Any
    ):
        self.user_id = user_id
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.client = client
        self.logger = logger
        self.streaming_mode = True
        self.show_thoughts = False
        self.debug_mode = False
        self.agent: str = ""
        # Retrieve the sender's information using the Slack API
        sender_info = client.users_info(user=user_id)
        self.user_name = sender_info["user"]["real_name"]
        self.model = TextModel.GEMINI_3_PRO
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
            self.logger.error(f"Error fetching conversation history: {str(e)}")
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
                                        self.logger.info(
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
                                            self.logger.debug(
                                                f"Fetching youtube transcript for [{url}]."
                                            )
                                            content = yt_transcript(url)
                                            tag = "YoutubeTranscript"
                                        else:
                                            self.logger.debug(f"Reading text from [{url}].")
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
                    self.logger.debug(f"Files:\n{file}")
                    msg_content = None
                    mimetype = file.get("mimetype", "")
                    file_url = file.get("url_private")
                    file_name = file.get("name", "Unknown File")
                    self.logger.info(f"Found file [{file_name}] of type [{mimetype}].")
                    if mimetype.startswith("image/"):
                        msg_content = f"<Image name:{file_name}/>"
                        # TODO: Images are not supported yet.
                        self.logger.error("Found image attachment.")
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
            self.logger.debug(f"<history>\n{merged_messages}</history>")
            return (merged_messages, commands)

        except Exception as e:
            self.logger.error(f"Error processing conversation: {str(e)}")
            raise e

    def is_command(self, text):
        if not isinstance(text, str):
            return False
        cmd = text.strip()
        return cmd.startswith("\\")

    def _save_settings(self, session_id: str = "default"):
        item = {
            "user_id": self.user_id,
            "session_id": session_id,
            "agent": self.agent,
            "model": self.model.value,
            "streaming_mode": self.streaming_mode,
            "show_thoughts": self.show_thoughts,
            "debug_mode": self.debug_mode,
        }
        try:
            settings_table.put_item(Item=item)
            self.logger.info(f"Saved settings for user {self.user_id}.")
        except Exception as e:
            self.logger.error(f"Failed to save settings for user {self.user_id}: {e}")

    def _load_settings(self, session_id: str = "default"):
        item = settings_table.get_item(
            Key={"user_id": self.user_id, "session_id": session_id}
        )
        if "Item" in item:
            settings = item["Item"]
            self.model = TextModel(settings["model"])
            self.streaming_mode = settings["streaming_mode"]
            self.show_thoughts = settings["show_thoughts"]
            self.debug_mode = settings["debug_mode"]
            self.agent = settings["agent"]
        else:
            self.logger.error(f"No settings found for user {self.user_id}.")

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
            say(
                text=f"You are currently chatting with {self.model.value} (agent: [{self.agent}])."
            )
        elif cmd in ["\\gpt5", "\\gpt"]:
            self.model = TextModel.GPT_52
            say(text="Model set to GPT-5.2.")
        # elif cmd in ["\\llama", "\\llama31", "\\llama405", "\\llama405b"]:
        #     self.model = TextModel.LLAMA31_405B
        #     say(text="Model set to LLaMA-3.1 405B.")
        # elif cmd in ["\\llama70b", "\\llama70"]:
        #     self.model = TextModel.LLAMA3_70B
        #     say(text="Model set to LLaMA-3 70B.")
        # # elif cmd in ["\\groq", "\\groq70", "\\groq70b"]:
        #     self.model = TextModel.GROQ_LLAMA3_70B
        #     say(text="Model set to LLaMA 3 70B (Groq).")
        elif cmd in ["\\sonnet", "\\claude"]:
            self.model = TextModel.CLAUDE_45_SONNET
            say(text="Model set to Claude 4.5 Sonnet.")
        elif cmd == "\\haiku":
            self.model = TextModel.CLAUDE_45_HAIKU
            say(text="Model set to Claude 4.5 Haiku.")
        elif cmd == "\\opus":
            self.model = TextModel.CLAUDE_45_OPUS
            say(text="Model set to Claude 4.5 Opus.")
        elif cmd == "\\gemini":
            self.model = TextModel.GEMINI_3_PRO
            say(text="Model set to Gemini 3 Pro.")
        elif cmd == "\\deepseek":
            self.model = TextModel.DEEPSEEK_V32
            say(text="Model set to Deepseek v3.2.")
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
        elif cmd == "\\agent" or cmd == "\\agents":
            # Show available agents
            available = ", ".join(list_agents()) or "none"
            current = self.agent or "none"
            say(text=f"Available agents: [{available}]. Current: [{current}]. Use \\agent <name> to set.")
        elif cmd.startswith("\\agent "):
            agent_name = cmd[len("\\agent ") :].strip().lower()
            if agent_name in ("none", "off", "clear", ""):
                self.agent = ""
                say(text="Agent cleared. Using default LLM mode.")
            elif get_agent(agent_name):
                self.agent = agent_name
                say(text=f"Agent set to [{agent_name}].")
            else:
                available = ", ".join(list_agents())
                say(text=f"Unknown agent [{agent_name}]. Available: [{available}].")
        elif cmd == "\\help":
            say(
                f"""
{HELP_PREAMBLE} I am a basic chatbot to quickly use GPT4, Claude, LLaMA & Gemini in one place. The chat is organized in sessions. Once you reset a session, all the previous conversation is lost. I am incapable of analyzing images or writing code right now, but feel free to upload PDFs, text files, or link to any websites, and I'll try to scrape whatever text I can. Note that model changes preserve the session so far. Here's the full list of available commands you can use:\n
- \\reset: Reset the chat session. Preserves the previous LLM you were chatting with.\n
- \\who: Returns the name of the chat model you are chatting with.\n
- \\gpt5: Use GPT-5.2 for future messages.\n
- \\sonnet: Use Claude 4.5 Sonnet for future messages.\n
- \\opus: Use Claude 4.5 Opus for future messages.\n
- \\gemini: Use Gemini 3 Pro for future messages.\n
- \\deepseek: Use Deepseek v3.2 for future messages.\n
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
        if self.model.value.startswith("gpt") or self.model.value.startswith("gemini"):
            extra_completion_params["reasoning_effort"] = "high"
        elif self.model.value.startswith("claude"):
            extra_completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 16384,
            }
            extra_completion_params["max_completion_tokens"] = 65536
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

    def _generate_from_adk_agent(
        self, messages: list[ChatMessage]
    ) -> None:
        """
        Generates a response from an ADK agent using the conversation history.

        Args:
            messages: The list of ChatMessage objects from conversation history.
        """
        agent = get_agent(self.agent)
        if not agent:
            self.logger.error(f"Agent '{self.agent}' not found, falling back to LLM.")
            self.agent = ""
            return

        self.logger.debug(f"Generating response using ADK agent: {self.agent}")
        self._set_chat_status(f"ADK agent [{self.agent}] is generating...")

        # Get the latest user message as the query
        query = messages[-1].content if messages else ""

        # Post initial message
        message_ts = self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text=f"[[ ADK: {self.agent} ]] Processing ...",
        )["ts"]

        async def run_agent():
            final_response = "Agent did not produce a response."
            async for event in run_agent_async(
                agent=agent,
                user_id=self.user_id,
                session_id=self.thread_ts,
                query=query,
            ):
                if self.debug_mode and event.content:
                    self.logger.debug(f"ADK Event: {event}")
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
            return final_response

        # Run the async agent
        try:
            response_text = asyncio.run(run_agent())
        except Exception as e:
            self.logger.error(f"Error running ADK agent: {e}")
            response_text = f"Error running agent: {e}"

        # Update the message with the response
        for chunk in self.break_message(response_text):
            self.client.chat_update(
                channel=self.channel_id,
                ts=message_ts,
                text=chunk,
            )
            # If there are more chunks, post new messages
            if chunk != self.break_message(response_text)[-1]:
                message_ts = self.client.chat_postMessage(
                    channel=self.channel_id,
                    thread_ts=self.thread_ts,
                    text="...",
                )["ts"]

    def _generate_from_model(
        self, messages_with_instr: list[dict]
    ) -> None:
        """
        Generates a response from the configured LLM using the provided messages.

        Handles both streaming and non-streaming modes, updates Slack status,
        and sets the thread title.

        Args:
            messages_with_instr: The list of messages formatted for the LLM API.
        """
        self.logger.debug(f"Generating response using model: {self.model.value}")
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
            self.logger.warning(f"Could not set thread status: {e}")  # Non-fatal

    def _set_thread_title(self, messages_with_instr: list[dict]) -> None:
        """Sets the thread title based on the messages."""
        try:
            title = generate_title(messages_with_instr)
            self.client.assistant_threads_setTitle(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                title=title,
            )
            self.logger.info(f"Set thread title to: {title}")
        except Exception as e:
            self.logger.error(f"Failed to generate or set thread title: {e}")

    def process_direct_message(self, text: str) -> None:
        """Processes an incoming direct message or mention in a thread.

        Fetches history, handles commands, generates a response using the LLM,
        and sends the response back to Slack.

        Args:
            text: The text content of the incoming Slack message.
        """
        # 1. Load settings for the user
        self._load_settings()

        # 2. Fetch conversation history and past commands
        messages, commands = self.fetch_conversation_history()

        # 3. Process commands
        if self.is_command(text):
            # Process the command and send a response back.
            if self.process_command(text, self.say):
                self._save_settings()
                return
            # If process_command returned False, it's an unknown command.
            # We'll treat it as regular text input for the LLM below.
            self.logger.error(f"Unknown command '{text}', treating as text input.")

        # Set initial status in Slack thread
        self._set_chat_status(f"{self.model.value} is generating ...")

        # 4. Prepare messages for the LLM API
        # Combine system instructions with the fetched/merged message history
        messages_with_instr = [
            msg.to_openai_format()
            for msg in ([ChatMessage.from_user(self.system_instr)] + messages)
        ]
        self.logger.debug(f"Messages being sent to LLM:\n{messages_with_instr}")

        # 5. Generate and send the response
        # Route to ADK agent if one is configured
        if self.agent and get_agent(self.agent):
            # TODO: Figure out a good way to pass the system instruction to the agent
            self._generate_from_adk_agent(messages_with_instr)
        else:
            # Use default LLM flow
            self._generate_from_model(messages_with_instr)

        # 6. Generate and set the thread title after the response is complete
        self._set_thread_title(messages_with_instr)
