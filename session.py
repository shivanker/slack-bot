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
            # Extract messages from the Slack conversation history
            messages = conversation_history["messages"]

            history: list[ChatMessage] = [] # Stores the chat messages for the LLM
            commands: list[str] = [] # Stores any commands executed in the session
            for message in messages:
                text = message.get("text")
                sent_by_user = message.get("user") == self.user_id

                # Process text content
                if text:
                    if self.is_command(text):
                        # Handle commands: remove the bot's response to the command from history
                        # and store the command itself. Stop processing if it's a reset.
                        if history:
                            history.pop() # Remove the bot's ack/response to the command
                        commands.append(text)
                        if text == "\\reset":
                            # History processing stops after a reset command
                            break
                        continue
                    elif text.startswith(ERROR_HEADER):
                        # Represent errors generically in history
                        history.append(ChatMessage.from_assistant("<Unknown Error />"))
                        continue
                    elif text.startswith(HELP_PREAMBLE):
                        # Ignore help messages in history
                        continue
                    else:
                        # Add regular messages to history
                        history.append(
                            ChatMessage.from_user(text)
                            if sent_by_user
                            else ChatMessage.from_assistant(text)
                        )

                    # Process links within user messages
                    if sent_by_user:
                        # Attempt to find links within Slack's block structure
                        blocks = message.get("blocks")
                        if not blocks: continue # Skip if no blocks found
                        for block in blocks:
                            elements = block.get("elements", [])
                            for element in elements:
                                inner_elements = element.get("elements", [])
                                for unit in inner_elements:
                                    if unit.get("type") == "link":
                                        url = unit.get("url")
                                        if not url: continue

                                        mimetype = check_mimetype(url)
                                        logger.info(f"Found link [{url}] of type [{mimetype}].")

                                        # Treat certain link types as file attachments for simplicity
                                        if mimetype.startswith("image/") or mimetype in ["text/plain", "application/pdf"]:
                                            # Add to the message's file list for later processing
                                            message.setdefault("files", []).append({
                                                "name": url,
                                                "url_private": url,
                                                "mimetype": mimetype,
                                            })
                                            continue

                                        # Extract content from YouTube or web pages
                                        content = None
                                        tag = None
                                        if is_youtube_video(url):
                                            logger.debug(f"Fetching youtube transcript for [{url}].")
                                            content = yt_transcript(url)
                                            tag = "YoutubeTranscript"
                                        else:
                                            logger.debug(f"Reading text from [{url}].")
                                            content = scrape_text(url)
                                            tag = "ScrapedTextFromURL"

                                        # Add extracted content as a separate user message in history
                                        if content and tag:
                                            history.append(
                                                ChatMessage.from_user(
                                                    f"<{tag} url={url}>\n{content}\n</{tag}>"
                                                )
                                            )

                # Process file attachments
                files = message.get("files", [])
                for file in files:
                    logger.debug(f"Processing file: {file.get('name', 'N/A')}")
                    msg_content = None
                    mimetype = file.get("mimetype", "")
                    file_url = file.get("url_private")
                    file_name = file.get("name", "Unknown File")
                    logger.info(f"Found file [{file_name}] of type [{mimetype}].")

                    if not file_url:
                        logger.warning(f"File '{file_name}' has no URL, skipping.")
                        continue

                    # Handle different file types
                    if mimetype.startswith("image/"):
                        # Represent image files generically for now
                        msg_content = f"<Image name:{file_name}/>"
                        # Note: Image content is not downloaded or processed further here.
                        logger.info(f"Found image attachment: {file_name}")
                    elif mimetype == "text/plain":
                        content = download_file(file_url).decode('utf-8', errors='replace') # Decode bytes to string
                        msg_content = f"<File name='{file_name}' mimetype='{mimetype}'>\n{content}\n</File>"
                    elif mimetype == "application/pdf":
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(file_url)
                        msg_content = f"<File name='{file_name}' mimetype='{mimetype}'>\n{pdf_text}\n</File>"
                    else:
                        # Represent other file types generically
                        msg_content = f"<File name={file_name}/>"

                    # Add file content/representation to history
                    if msg_content:
                        history.append(
                            ChatMessage.from_user(msg_content)
                            if sent_by_user
                            else ChatMessage.from_assistant(msg_content)
                        )

            # Ensure the conversation history starts with a user message for the LLM
            if history and not history[0].is_from(ChatRole.USER):
                history = [ChatMessage.from_user("...")] + history # Prepend a placeholder if needed

            # Merge consecutive messages from the same role to reduce token count
            merged_messages: list[ChatMessage] = []
            if history: # Check if history is not empty before merging
                prev_role = None
                for chatmsg in history:
                    if chatmsg.is_from(prev_role): # type: ignore
                        # Append content to the last message if the role is the same
                        merged_messages[-1].content += "\n" + chatmsg.content
                    else:
                        # Start a new message if the role changes
                        merged_messages.append(chatmsg)
                        prev_role = chatmsg.role

            logger.debug(f"<history>\n{merged_messages}</history>")
            return (merged_messages, commands)

        except Exception as e:
            logger.error(f"Error processing conversation history: {str(e)}")
            raise e

    def is_command(self, text):
        if not isinstance(text, str):
            return False
        cmd = text.strip()
        return cmd.startswith("\\")

    def process_command(self, text: str, say=lambda text: None) -> bool:
        """
        Processes a command string.

        Args:
            text: The command string (e.g., "\\reset").
            say: A function to send a response back to the user (optional).

        Returns:
            True if the text was a known command and processed, False otherwise.
        """
        cmd = text.strip()

        # Model selection commands
        if cmd == "\\o1":
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

        # Session control commands
        elif cmd == "\\reset":
            say(text="Session has been reset.") # Note: History processing stops earlier
        elif cmd in ("\\who?", "\\who", "\\llm", "\\model"):
            say(text=f"You are currently chatting with {self.model.value}.")

        # Feature toggles
        elif cmd == "\\stream":
            self.streaming_mode ^= True
            say(text=f'Streaming mode {"enabled" if self.streaming_mode else "disabled"}.')
        elif cmd == "\\nostream":
            self.streaming_mode = False
            say(text="Streaming mode disabled.")
        elif cmd == "\\thoughts":
            self.show_thoughts ^= True
            say(text=f'Displaying thoughts {"enabled" if self.show_thoughts else "disabled"}.')
        elif cmd == "\\nothoughts":
            self.show_thoughts = False
            say(text="Displaying thoughts disabled.")

        # Utility commands
        elif cmd.startswith("\\extract "):
            # Extract text from a URL (primarily for debugging)
            if say:
                url_to_extract = cmd[len("\\extract "):].strip()
                extracted_content = extract(url_to_extract) or "Failed to extract content."
                say(text=extracted_content)
        elif cmd == "\\help":
            # Display help message
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
            # Command not recognized
            # say(f"Unknown command: [{cmd}]") # Optionally inform user
            return False # Indicate command was not processed

        return True # Indicate command was processed successfully

    def break_message(self, text: str, max_size: int = 2400) -> list[str]:
        """
        Split text into chunks suitable for Slack messages (approx. max_size chars).

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
            "max_tokens": 128000, # Default max tokens
        }
        # Model-specific parameter adjustments
        if self.model.value.startswith("o"):
            extra_completion_params["reasoning_effort"] = "high"
        elif self.model == TextModel.CLAUDE_37_SONNET:
            extra_completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000,
            }
            extra_completion_params["max_completion_tokens"] = 64000
        # Add other model-specific params here if needed
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
        # Extract reasoning (if available and enabled) and the main message content
        reasoning_content = response.choices[0].get("reasoning_content", "") if self.show_thoughts else "" # type: ignore
        full_text: str = response.choices[0].message.content or "" # type: ignore

        # Send reasoning content first if present
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
        initial_status = f"[[ {self.model.value} ]] Thinking ..."
        message_ts = self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text=initial_status,
        )["ts"]

        last_update_time = time.time()
        update_interval = 2.0  # Start with 2 seconds interval, adjust later
        start_time = time.time()
        current_message_content = ""
        currently_thinking = False # Track if the current chunk is part of 'thinking'
        active_message_ts = message_ts # Timestamp of the message being updated

        for chunk in response_stream:
            delta = chunk.choices[0].delta
            reasoning_chunk: str = delta.get("reasoning_content", "") if self.show_thoughts else "" # type: ignore
            text_chunk: str = delta.content or "" # type: ignore

            # Update Slack thread status based on whether reasoning or text is received
            if reasoning_chunk:
                self.client.assistant_threads_setStatus(
                    channel_id=self.channel_id, thread_ts=self.thread_ts,
                    status=f"{self.model.value} is thinking..."
                )
            elif text_chunk: # Only update if there's actual text content being generated
                 self.client.assistant_threads_setStatus(
                    channel_id=self.channel_id, thread_ts=self.thread_ts,
                    status=f"{self.model.value} is generating..."
                )

            # Handle transitions between thinking and generating phases
            if not currently_thinking and reasoning_chunk:
                # Started thinking
                currently_thinking = True
                reasoning_chunk = f"<thinking>\n{reasoning_chunk}" # Add opening tag
            elif currently_thinking and not reasoning_chunk and text_chunk:
                 # Finished thinking, starting to generate main response
                currently_thinking = False
                # Update the previous message to close the thinking tag
                self.client.chat_update(
                    channel=self.channel_id, ts=active_message_ts,
                    text=f"{current_message_content}\n</thinking>\n\n"
                )
                # Start a *new* message for the main response content
                active_message_ts = self.client.chat_postMessage(
                    channel=self.channel_id, thread_ts=self.thread_ts,
                    text=f"... [[ {self.model.value} generating response ]] ..."
                )["ts"]
                current_message_content = "" # Reset content for the new message

            # Append the current chunk (reasoning or text)
            current_message_content += reasoning_chunk + text_chunk
            current_time = time.time()

            # Throttle Slack updates: Update message if interval passed or message is long
            if (current_time - last_update_time >= update_interval or
                len(current_message_content) > 2400): # Threshold close to Slack limit

                # If message exceeds limit, finalize current message and start a new one
                if len(current_message_content) > 2400:
                    # TODO: Use break_message logic here? For now, just post and start new.
                    # This might slightly exceed the limit if the last chunk pushes it over.
                    self.client.chat_update(
                        channel=self.channel_id, ts=active_message_ts,
                        text=current_message_content # Post the full content before starting new
                    )
                    # Start a new message
                    status_indicator = "thinking" if currently_thinking else "generating"
                    active_message_ts = self.client.chat_postMessage(
                        channel=self.channel_id, thread_ts=self.thread_ts,
                        text=f"... [[ {self.model.value} {status_indicator} ]] ..."
                    )["ts"]
                    current_message_content = "" # Reset content for the new message
                else:
                    # Update the existing message with a progress indicator
                    status_indicator = "thinking" if currently_thinking else "generating"
                    self.client.chat_update(
                        channel=self.channel_id, ts=active_message_ts,
                        text=f"{current_message_content} ... [[ {self.model.value} {status_indicator} ]] ..."
                    )
                last_update_time = current_time

            # Adjust update interval if generation is taking a long time
            if current_time - start_time > 60:
                update_interval = 3.0 # Slow down updates slightly

        # Final update after stream ends to remove the progress indicator
        # Close thinking tag if the stream ended during thinking
        if currently_thinking:
            current_message_content += "\n</thinking>"
        self.client.chat_update(
            channel=self.channel_id, ts=active_message_ts, text=current_message_content
        )


    def _generate_from_model(self, messages_with_instr: list[dict], logger: Any) -> None:
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

        # Set initial status in Slack thread
        initial_status = f"{self.model.value} is thinking..." if self.streaming_mode else f"{self.model.value} is generating..."
        try:
            self.client.assistant_threads_setStatus(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                status=initial_status,
            )
        except Exception as e:
            logger.warning(f"Could not set thread status: {e}") # Non-fatal

        # Generate response based on streaming mode
        if self.streaming_mode:
            self._handle_streaming_response(messages_with_instr, extra_completion_params)
        else:
            self._handle_non_streaming_response(messages_with_instr, extra_completion_params)

        # Generate and set the thread title after the response is complete
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
        """
        Processes an incoming direct message or mention in a thread.

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
        elif not self.is_command(text) and commands:
             # If current text is not a command, but there were previous commands, replay all
             pass # commands_to_replay is already set correctly

        for cmd in commands_to_replay:
            self.process_command(cmd) # Re-run command logic without sending ack message

        # 3. Process the current message if it's a command
        if self.is_command(text):
            # Process the command and send an acknowledgement message back.
            # If it's a known command, stop processing here.
            if self.process_command(text, self.say):
                return
            # If process_command returned False, it's an unknown command.
            # We'll treat it as regular text input for the LLM below.
            logger.info(f"Unknown command '{text}', treating as text input.")


        # 4. Prepare messages for the LLM API
        # Combine system instructions with the fetched/merged message history
        messages_with_instr = [
            msg.to_openai_format()
            for msg in ([ChatMessage.from_system(self.system_instr)] + messages) # Use system role
        ]
        logger.debug(f"Messages being sent to LLM:\n{messages_with_instr}")

        # 5. Generate and send the response using the LLM
        self._generate_from_model(messages_with_instr, logger)
