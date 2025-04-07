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

    def is_command(self, text: str) -> bool:
        """Check if the given text is a command (starts with '\\')."""
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

    def _update_thread_status(self, status: str) -> None:
        """Update the status displayed in the Slack thread."""
        try:
            self.client.assistant_threads_setStatus(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                status=status,
            )
        except Exception as e:
            logger.warning(f"Failed to update thread status: {e}")

    def _update_thread_title(self, messages_for_title: list[dict[str, Any]]) -> None:
        """Generate and update the title of the Slack thread based on the conversation."""
        try:
            title = generate_title(messages_for_title)
            self.client.assistant_threads_setTitle(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                title=title,
            )
        except Exception as e:
            logger.warning(f"Failed to update thread title: {e}")

    # --- LLM Interaction Methods ---

    def _get_completion_params(self) -> dict[str, Any]:
        """Prepare extra parameters for the litellm.completion call based on the selected model."""
        extra_completion_params: dict[str, Any] = {
            "max_tokens": 128000,  # Set a high default max_tokens
        }
        if self.model.value.startswith("o"):
            # Specific parameter for 'o' models
            extra_completion_params["reasoning_effort"] = "high"
        elif self.model == TextModel.CLAUDE_37_SONNET:
            # Specific parameters for Claude 3.7 Sonnet
            extra_completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000,  # Allow tokens for thinking process
            }
            extra_completion_params["max_completion_tokens"] = 64000 # Max output tokens
        # Add other model-specific parameters here if needed
        return extra_completion_params

    def _generate_response(self, messages_with_instr: list[dict[str, Any]]) -> Any:
        """
        Calls the LLM to generate a response, either streaming or non-streaming.

        Args:
            messages_with_instr: The list of messages formatted for the LLM API,
                                 including system instructions.

        Returns:
            The response object from litellm.completion (either a full response or a stream iterator).
        """
        extra_params = self._get_completion_params()
        logger.info(f"Generating response with model {self.model.value} (streaming={self.streaming_mode})")
        logger.debug(f"LLM call parameters: {extra_params}")
        return completion(
            model=self.model.value,
            messages=messages_with_instr,
            stream=self.streaming_mode,
            **extra_params,
        )

    def _handle_non_streaming_response(self, response: Any) -> None:
        """Processes and sends a non-streaming response to Slack."""
        try:
            # Extract reasoning and main content from the response
            # Note: .get("reasoning_content", "") might need adjustment based on actual response structure
            reasoning_content: str = response.choices[0].get("reasoning_content", "")
            full_text: str = response.choices[0].message.content

            # Display reasoning/thoughts if enabled
            if self.show_thoughts and reasoning_content:
                thinking_block = f"<thinking>\n{reasoning_content}\n</thinking>\n\n"
                for chunk in self.break_message(thinking_block):
                    self.say(text=chunk)

            # Send the main response content in chunks if necessary
            for chunk in self.break_message(full_text):
                self.say(text=chunk)
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(f"Error processing non-streaming response: {e}\nResponse: {response}")
            self.say(f"{ERROR_HEADER}Failed to parse LLM response.")
        except Exception as e:
            logger.error(f"Unexpected error in _handle_non_streaming_response: {e}")
            self.say(f"{ERROR_HEADER}An unexpected error occurred while processing the response.")


    def _handle_streaming_response(self, response_stream: Any) -> None:
        """Processes and sends a streaming response to Slack, updating a message."""
        initial_message_ts = None
        try:
            # Post an initial message to update
            initial_message = self.client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=self.thread_ts,
                text=f"[[ {self.model.value} ]] Thinking ...",
            )
            initial_message_ts = initial_message["ts"]
            message_ts = initial_message_ts # The timestamp of the message being updated

            last_update_time = time.time()
            update_interval = 2.0  # Start with 2 seconds interval
            start_time = time.time()
            current_message_content = ""
            currently_thinking = False # Flag to track if we are currently processing thought block
            accumulated_reasoning = "" # Accumulate reasoning content separately if needed

            for chunk in response_stream:
                # Extract content and reasoning from the current chunk
                # Note: .get("reasoning_content", "") might need adjustment
                delta = chunk.choices[0].delta
                last_reasoning_chunk: str = delta.get("reasoning_content", "")
                last_content_chunk: str = delta.content or ""

                # Update thread status based on whether reasoning or content is received
                if last_reasoning_chunk:
                    self._update_thread_status(f"{self.model.value} is thinking...")
                elif last_content_chunk:
                     self._update_thread_status(f"{self.model.value} is generating...")

                # --- Handle Thinking Blocks ---
                if self.show_thoughts:
                    if not currently_thinking and last_reasoning_chunk:
                        # Start of a thinking block
                        currently_thinking = True
                        accumulated_reasoning += f"<thinking>\n{last_reasoning_chunk}"
                    elif currently_thinking and last_reasoning_chunk:
                        # Continuation of thinking block
                         accumulated_reasoning += last_reasoning_chunk
                    elif currently_thinking and not last_reasoning_chunk:
                         # End of thinking block (content chunk received after thinking)
                        currently_thinking = False
                        accumulated_reasoning += "\n</thinking>\n\n"
                        # Post the completed thinking block immediately
                        # We break it just in case it's huge
                        for thinking_chunk in self.break_message(accumulated_reasoning):
                             self.client.chat_postMessage(
                                channel=self.channel_id,
                                thread_ts=self.thread_ts,
                                text=thinking_chunk,
                            )
                        accumulated_reasoning = "" # Reset accumulated reasoning

                        # Start a new message for the actual response content
                        new_msg = self.client.chat_postMessage(
                            channel=self.channel_id,
                            thread_ts=self.thread_ts,
                            text=f"... [[ {self.model.value} generating response ]] ...",
                        )
                        message_ts = new_msg["ts"]
                        current_message_content = "" # Reset content for the new message

                # --- Accumulate Content ---
                current_message_content += last_content_chunk

                # --- Update Slack Message Periodically or if Too Long ---
                current_time = time.time()
                if (
                    current_time - last_update_time >= update_interval
                    or len(current_message_content) > 2400 # Force update if message gets long
                ):
                    last_update_time = current_time
                    update_text = f"{current_message_content} ... [[ {self.model.value} generating ]] ..."

                    if len(current_message_content) > 2400:
                        # Message is too long, finalize this one and start a new one
                        final_chunk = self.break_message(current_message_content, 2400)[0]
                        self.client.chat_update(
                            channel=self.channel_id,
                            ts=message_ts,
                            text=final_chunk, # Send the first part
                        )
                        # Start a new message for the rest
                        remaining_content = current_message_content[len(final_chunk):]
                        new_msg = self.client.chat_postMessage(
                            channel=self.channel_id,
                            thread_ts=self.thread_ts,
                            text=f"{remaining_content}... [[ {self.model.value} generating ]] ...",
                        )
                        message_ts = new_msg["ts"]
                        current_message_content = remaining_content # Continue with the remainder
                    else:
                        # Update the existing message
                        self.client.chat_update(
                            channel=self.channel_id,
                            ts=message_ts,
                            text=update_text,
                        )

                # Adjust update interval for long-running generations
                if current_time - start_time > 60:
                    update_interval = 3.0 # Slow down updates slightly

            # --- Final Update ---
            # Ensure any remaining accumulated reasoning (if thoughts enabled) is posted
            if self.show_thoughts and accumulated_reasoning:
                 if currently_thinking: # Add closing tag if stream ended mid-thought
                     accumulated_reasoning += "\n</thinking>"
                 for thinking_chunk in self.break_message(accumulated_reasoning):
                     self.client.chat_postMessage(
                        channel=self.channel_id,
                        thread_ts=self.thread_ts,
                        text=thinking_chunk,
                    )

            # Final update for the main content message to remove the "generating" suffix
            self.client.chat_update(
                channel=self.channel_id, ts=message_ts, text=current_message_content
            )

        except Exception as e:
            logger.error(f"Error during streaming response: {e}", exc_info=True)
            # Attempt to update the initial message with an error, if it exists
            if initial_message_ts:
                try:
                    self.client.chat_update(
                        channel=self.channel_id,
                        ts=initial_message_ts,
                        text=f"{ERROR_HEADER}An error occurred during streaming.",
                    )
                except Exception as update_err:
                     logger.error(f"Failed to update message with streaming error: {update_err}")
            else:
                 # If initial message failed, post a new error message
                 self.say(f"{ERROR_HEADER}An error occurred during streaming.")


    # --- Text Processing Utilities ---

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
                # No natural break found, force break at max_size
                i += max_size

            # Remove leading/trailing whitespace and add if not empty
            trimmed_chunk = chunk.strip()
            if trimmed_chunk:
                chunks.append(trimmed_chunk)

        return chunks

    # --- Main Processing Logic ---

    def process_direct_message(self, text: str, logger: Any) -> None:
        """
        Handles an incoming direct message or mention in a thread.
        Fetches history, processes commands, generates a response from the LLM,
        and sends it back to the Slack thread.

        Args:
            text: The text content of the incoming Slack message.
            logger: The logger instance.
        """
        try:
            messages, commands = self.fetch_conversation_history()

            # Apply commands from history (excluding the last one if it's the current text)
            # This ensures the correct model and settings are applied based on history.
            processed_commands = commands[:-1] if commands and self.is_command(text) else commands
            for cmd in processed_commands:
                self.process_command(cmd) # Apply historical commands silently

            # Process the current text if it's a command
            if self.is_command(text):
                if self.process_command(text, self.say): # Process and respond if it's a known command
                    return # Command handled, no LLM call needed
                else:
                    logger.warning(f"Unknown command received: {text}. Treating as text.")
                    # If it's an unknown command, fall through to treat it as regular text.

            # --- Prepare for LLM call ---
            messages_with_instr = [
                msg.to_openai_format()
                for msg in ([ChatMessage.from_system(self.system_instr)] + messages) # Prepend system instruction
            ]
            logger.debug(f"Messages sent to LLM:\n{messages_with_instr}")

            # --- Generate and Handle Response ---
            self._update_thread_status(f"{self.model.value} is thinking...") # Initial status
            response = self._generate_response(messages_with_instr)

            if self.streaming_mode:
                self._handle_streaming_response(response)
            else:
                self._handle_non_streaming_response(response)

            # --- Finalize ---
            # Update thread title after response generation
            self._update_thread_title(messages_with_instr)
            # Reset status to idle or model name after completion
            self._update_thread_status(f"Ready ({self.model.value})")

        except Exception as e:
            logger.error(f"Error processing direct message: {e}", exc_info=True)
            try:
                # Attempt to send an error message back to the user
                self.say(f"{ERROR_HEADER}{e}")
                self._update_thread_status("Error occurred")
            except Exception as send_error:
                 logger.error(f"Failed to send error message to Slack: {send_error}")


# --- Old code kept for reference during refactoring ---
# Original process_direct_message structure:
# def process_direct_message(self, text: str, logger: Any) -> None:
#     messages, commands = self.fetch_conversation_history()
#
#     # Re-run previous commands in session
#     for cmd in commands[:-1]:
#         self.process_command(cmd)
#
#     # Run the latest command, responding if it's the current message
#     if self.is_command(text):
#         if self.process_command(text, self.say):
#             return  # Don't return if command processing failed. Let's process it like a text
#     elif commands:
#         self.process_command(commands[-1])
#
#     messages_with_instr = [
#         msg.to_openai_format()
#         for msg in ([ChatMessage.from_user(self.system_instr)] + messages)
#     ]
#     logger.debug(messages_with_instr)
#     extra_completion_params: dict[str, Any] = {
#         "max_tokens": 128000,
#     }
#     if self.model.value.startswith("o"):
#         extra_completion_params["reasoning_effort"] = "high"
#     elif self.model == TextModel.CLAUDE_37_SONNET:
#         extra_completion_params["thinking"] = {
#             "type": "enabled",
#             "budget_tokens": 32000,
#         }
#         extra_completion_params["max_completion_tokens"] = 64000
#
#     self.client.assistant_threads_setStatus(
#         channel_id=self.channel_id,
#         thread_ts=self.thread_ts,
#         status=f"{self.model.value} is generating ...",
#     )
#     # Process the user's message using the selected model and conversation history
#     if not self.streaming_mode:
#         response = completion(
#             model=self.model.value,
#             messages=messages_with_instr,
#             **extra_completion_params,
#         )
#         reasoning_content = response.choices[0].get("reasoning_content", "")  # type: ignore
#         full_text: str = response.choices[0].message.content  # type: ignore
#
#         if not self.show_thoughts:
#             reasoning_content = ""
#
#         if reasoning_content:
#             reasoning_content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n"
#             for chunk in self.break_message(reasoning_content):
#                 self.say(text=chunk)
#         # Send response in chunks
#         for chunk in self.break_message(full_text):
#             self.say(text=chunk)
#         return
#
#     response = completion(
#         model=self.model.value,
#         messages=messages_with_instr,
#         stream=True,
#         **extra_completion_params,
#     )
#     initial_message = self.client.chat_postMessage(
#         channel=self.channel_id,
#         thread_ts=self.thread_ts,
#         text=f"[[ {self.model.value} ]] Thinking ...",
#     )["ts"]
#     last_update_time = time.time()
#     update_interval = 2.0  # Start with 2 seconds interval
#     start_time = time.time()
#     current_message = ""
#     currently_thinking = False
#     message_ts = initial_message
#
#     for chunk in response:
#         last_reasoning_chunk: str = chunk.choices[0].delta.get("reasoning_content", "")  # type: ignore
#         last_chunk: str = chunk.choices[0].delta.content or ""  # type: ignore
#         if len(last_reasoning_chunk) > 0:
#             self.client.assistant_threads_setStatus(
#                 channel_id=self.channel_id,
#                 thread_ts=self.thread_ts,
#                 status=f"{self.model.value} is thinking...",
#             )
#         else:
#             self.client.assistant_threads_setStatus(
#                 channel_id=self.channel_id,
#                 thread_ts=self.thread_ts,
#                 status=f"{self.model.value} is generating...",
#             )
#         if not self.show_thoughts:
#             last_reasoning_chunk = ""
#         if not currently_thinking and len(last_reasoning_chunk) > 0:
#             currently_thinking = True
#             last_reasoning_chunk = f"<thinking>\n{last_reasoning_chunk}"
#         if currently_thinking and len(last_reasoning_chunk) == 0:
#             currently_thinking = False
#             self.client.chat_update(
#                 channel=self.channel_id,
#                 ts=message_ts,
#                 text=f"{current_message}\n</thinking>\n\n",
#             )
#             # Start a new message for post-thinking response
#             message_ts = self.client.chat_postMessage(
#                 channel=self.channel_id,
#                 thread_ts=self.thread_ts,
#                 text=f"... [[ {self.model.value} generating response ]] ...",
#             )["ts"]
#             current_message = ""
#
#         current_message += last_reasoning_chunk + last_chunk
#         current_time = time.time()
#
#         # Check if it's time to send an update or start a new message
#         if (
#             current_time - last_update_time >= update_interval
#             or len(current_message) > 2400
#         ):
#             # TODO: Not sure if we can have a single big chunk and need to use break_message here
#             last_update_time = current_time
#             if len(current_message) > 2400:
#                 self.client.chat_update(
#                     channel=self.channel_id,
#                     ts=message_ts,
#                     text=current_message,
#                 )
#                 # Start a new message with just the new content
#                 message_ts = self.client.chat_postMessage(
#                     channel=self.channel_id,
#                     thread_ts=self.thread_ts,
#                     text=f"... [[ {self.model.value} {"thinking" if currently_thinking else "generating"} ]] ...",
#                 )["ts"]
#                 current_message = ""
#             else:
#                 # Update existing message
#                 self.client.chat_update(
#                     channel=self.channel_id,
#                     ts=message_ts,
#                     text=f"{current_message} ... [[ {self.model.value} {"thinking" if currently_thinking else "generating"} ]] ...",
#                 )
#
#         # Adjust the update interval if the process takes more than 30 seconds
#         if current_time - start_time > 60:
#             update_interval = 3.0
#
#     # Final update to remove the suffix
#     self.client.chat_update(
#         channel=self.channel_id, ts=message_ts, text=current_message
#     )
#     title = generate_title(messages_with_instr)
#     self.client.assistant_threads_setTitle(
#         channel_id=self.channel_id,
#         thread_ts=self.thread_ts,
#         title=title,
#     )

