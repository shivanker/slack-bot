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
    """
    Manages a chat conversation within a Slack thread.

    Handles fetching history, processing user commands, interacting with LLMs,
    and sending responses back to Slack.
    """
    def __init__(
        self, user_id: str, channel_id: str, thread_ts: str, client: WebClient
    ):
        """
        Initializes a new chat session.

        Args:
            user_id: The Slack ID of the user initiating the session.
            channel_id: The Slack channel ID where the session takes place.
            thread_ts: The timestamp of the parent message initiating the thread.
            client: An initialized Slack WebClient.
        """
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
        """
        Fetches the message history from the Slack thread associated with this session.

        Parses messages, handles commands, extracts content from links and files,
        and formats the history for the LLM.

        Returns:
            A tuple containing:
                - A list of ChatMessage objects representing the conversation history.
                - A list of command strings encountered in the history.
        Raises:
            Exception: If fetching or processing the conversation history fails.
        """
        try:
            # Retrieve all replies in the thread
            conversation_history = self.client.conversations_replies(
                channel=self.channel_id, ts=self.thread_ts, limit=100, inclusive=True
            )
        except Exception as e:
            logger.error(f"Error fetching conversation history: {str(e)}")
            raise e
        try:
            messages = conversation_history.get("messages", [])

            history: list[ChatMessage] = []
            commands: list[str] = []

            for message in messages:
                text = message.get("text")
                user = message.get("user")
                sent_by_user = user == self.user_id
                is_bot_reply = user == self.client.auth_test()["user_id"] # Check if message is from our bot

                # --- Process Text Content ---
                if text:
                    if self.is_command(text):
                        # If a command is found, record it.
                        # If the previous message in history was the bot's response to this command, remove it.
                        if history and is_bot_reply and history[-1].role == ChatRole.ASSISTANT:
                             # This assumes the bot's reply immediately follows the user's command.
                             # Might need adjustment if there can be delays or other messages in between.
                             # We pop the bot's ack message for the command.
                             history.pop()
                        commands.append(text)
                        # Stop processing history if reset command is found
                        if text == "\\reset":
                            history = [] # Clear history up to the reset command
                            break # Stop processing older messages
                        continue # Skip adding command text to history
                    elif text.startswith(ERROR_HEADER):
                        # Represent errors generically in history
                        history.append(ChatMessage.from_assistant("<Unknown Error />"))
                        continue
                    elif text.startswith(HELP_PREAMBLE):
                        # Ignore help messages in history
                        continue
                    else:
                        # Add regular text messages to history
                        history.append(
                            ChatMessage.from_user(text)
                            if sent_by_user
                            else ChatMessage.from_assistant(text)
                        )

                    # --- Process Links in User Messages ---
                    # If the message is from the user, check for links in rich text blocks
                    if sent_by_user:
                        blocks = message.get("blocks", [])
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
                                        logger.info(f"Found link [{url}] of type [{mimetype}].")

                                        # If it's an image, PDF, or plain text link treat it like a file upload
                                        if mimetype.startswith("image/") or mimetype in ["text/plain", "application/pdf"]:
                                            # Add to files list to be processed below
                                            message.setdefault("files", []).append(
                                                {
                                                    "name": url.split('/')[-1] or "linked_file", # Basic name extraction
                                                    "url_private": url, # Assuming public links for now
                                                    "mimetype": mimetype,
                                                }
                                            )
                                            continue # Skip further processing for this link

                                        # Handle YouTube links
                                        if is_youtube_video(url):
                                            logger.debug(f"Fetching youtube transcript for [{url}].")
                                            content = yt_transcript(url)
                                            tag = "YoutubeTranscript"
                                        # Handle other web links
                                        else:
                                            logger.debug(f"Reading text from [{url}].")
                                            content = scrape_text(url)
                                            tag = "ScrapedTextFromURL"

                                        # Add extracted content as a separate user message
                                        if content:
                                            history.append(
                                                ChatMessage.from_user(
                                                    f"<{tag} url={url}>\n{content}\n</{tag}>"
                                                )
                                            )

                # --- Process File Attachments ---
                files = message.get("files", [])
                for file in files:
                    logger.debug(f"Processing file: {file.get('name', 'N/A')}")
                    msg_content = None
                    mimetype = file.get("mimetype", "")
                    file_url = file.get("url_private") # Use private URL which requires auth
                    file_name = file.get("name", "UnknownFile")

                    if not file_url:
                        logger.warning(f"File '{file_name}' has no URL, skipping.")
                        continue

                    logger.info(f"Found file [{file_name}] of type [{mimetype}].")

                    try:
                        if mimetype.startswith("image/"):
                            # Represent image file in history (content not downloaded here)
                            msg_content = f"<Image name='{file_name}'/>"
                            # Note: Actual image processing might need separate handling/model capabilities
                        elif mimetype == "text/plain":
                            # Download and include text content
                            content = download_file(file_url).decode('utf-8', errors='ignore') # Decode bytes to string
                            msg_content = f"<File name='{file_name}' mimetype='{mimetype}'>\n{content}\n</File>"
                        elif mimetype == "application/pdf":
                            # Extract text from PDF
                            # Note: extract_text_from_pdf needs the URL, not downloaded content
                            pdf_text = extract_text_from_pdf(file_url)
                            msg_content = f"<File name='{file_name}' mimetype='{mimetype}'>\n{pdf_text}\n</File>"
                        else:
                            # Represent other file types by name
                            msg_content = f"<File name='{file_name}' mimetype='{mimetype}'/>"

                        if msg_content:
                            # Add file representation/content as a message from the uploader
                            history.append(
                                ChatMessage.from_user(msg_content)
                                if sent_by_user
                                else ChatMessage.from_assistant(msg_content) # Or handle bot uploads differently if needed
                            )
                    except Exception as file_e:
                        logger.error(f"Error processing file {file_name} ({file_url}): {str(file_e)}")
                        # Add an error message to history for the failed file
                        error_msg = f"<FileProcessingError name='{file_name}' error='{str(file_e)}'/>"
                        history.append(
                            ChatMessage.from_user(error_msg) if sent_by_user else ChatMessage.from_assistant(error_msg)
                        )


            # --- Final History Adjustments ---

            # Ensure the conversation starts with a user message if history is not empty
            if history and not history[0].is_from(ChatRole.USER):
                history.insert(0, ChatMessage.from_user("...")) # Add a placeholder user message

            # Merge consecutive messages from the same role into single messages
            merged_messages: list[ChatMessage] = []
            if history: # Check if history is not empty before merging
                current_merged_message = history[0]
                for i in range(1, len(history)):
                    chatmsg = history[i]
                    # Merge if the current message role matches the last merged message role
                    if chatmsg.role == current_merged_message.role:
                        current_merged_message.content += "\n" + chatmsg.content
                    else:
                        # If roles differ, add the completed merged message and start a new one
                        merged_messages.append(current_merged_message)
                        current_merged_message = chatmsg
                # Add the last merged message
                merged_messages.append(current_merged_message)

            logger.debug(f"<history>\n{merged_messages}</history>")
            return (merged_messages, commands)

        except Exception as e:
            logger.error(f"Error fetching/processing conversation history: {str(e)}")
            raise e

    def is_command(self, text):
        if not isinstance(text, str):
            return False
        cmd = text.strip()
        return isinstance(text, str) and text.strip().startswith("\\")

    def process_command(self, text: str, say=lambda text: None) -> bool:
        """
        Processes a command string, updates session state, and optionally sends a confirmation message.

        Args:
            text: The command string (e.g., "\\reset").
            say: A function to send a message back to the user (optional).

        Returns:
            True if the text was a recognized and processed command, False otherwise.
        """
        cmd = text.strip()

        # --- Session Control ---
        if cmd == "\\reset":
            # Note: History clearing happens in fetch_conversation_history
            if say: say(text="Session has been reset.")
        # --- Model Information ---
        elif cmd in ("\\who?", "\\who", "\\llm", "\\model"):
            if say: say(text=f"You are currently chatting with {self.model.value}.")
        # --- Model Selection ---
        elif cmd == "\\o1":
            self.model = TextModel.O1
            if say: say(text="Model set to O1.")
        elif cmd in ["\\o3-mini", "\\o3mini", "\\mini"]:
            self.model = TextModel.O3_MINI
            if say: say(text="Model set to O3 Mini.")
        elif cmd in ["\\gpt4o", "\\gpt"]:
            self.model = TextModel.GPT_4O
            if say: say(text="Model set to GPT-4o.")
        elif cmd == "\\gpt4":
            self.model = TextModel.GPT_4_TURBO
            if say: say(text="Model set to GPT-4.")
        elif cmd in ["\\llama", "\\llama31", "\\llama405", "\\llama405b"]:
            self.model = TextModel.LLAMA31_405B
            if say: say(text="Model set to LLaMA-3.1 405B.")
        elif cmd in ["\\llama70b", "\\llama70"]:
            self.model = TextModel.LLAMA3_70B
            if say: say(text="Model set to LLaMA-3 70B.")
        # elif cmd in ["\\groq", "\\groq70", "\\groq70b"]:
        #     self.model = TextModel.GROQ_LLAMA3_70B
        #     if say: say(text="Model set to LLaMA 3 70B (Groq).")
        elif cmd in ["\\sonnet", "\\claude"]:
            self.model = TextModel.CLAUDE_37_SONNET
            if say: say(text="Model set to Claude 3.7 Sonnet.")
        elif cmd == "\\haiku":
            self.model = TextModel.CLAUDE_35_HAIKU
            if say: say(text="Model set to Claude 3.5 Haiku.")
        elif cmd == "\\gemini":
            self.model = TextModel.GEMINI_25
            if say: say(text="Model set to Gemini 2.5 Pro.")
        elif cmd == "\\deepseek":
            self.model = TextModel.DEEPSEEK_R1
            if say: say(text="Model set to Deepseek R1.")
        # --- Feature Toggles ---
        elif cmd == "\\stream":
            self.streaming_mode ^= True
            if say: say(text=f'Streaming mode {"enabled" if self.streaming_mode else "disabled"}.')
        elif cmd == "\\nostream":
            self.streaming_mode = False
            if say: say(text="Streaming mode disabled.")
        elif cmd == "\\thoughts":
            self.show_thoughts ^= True
            if say: say(text=f'Displaying thoughts {"enabled" if self.show_thoughts else "disabled"}.')
        elif cmd == "\\nothoughts":
            self.show_thoughts = False
            if say: say(text="Displaying thoughts disabled.")
        # --- Debug/Utility Commands ---
        elif cmd.startswith("\\extract "):
            # Extract text from URL (for debugging)
            if say:
                extracted_text = extract(cmd[len("\\extract "):]) or "Failed to extract text."
                say(text=extracted_text)
        elif cmd == "\\help":
            # Display help message
            if say:
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
            # if say: say(f"Unknown command: [{cmd}]") # Optionally notify user of unknown command
            return False # Indicate command was not processed

        return True # Indicate command was processed

    def break_message(self, text: str, max_size: int = 2400) -> list[str]:
        """
        Splits a long text message into smaller chunks suitable for Slack messages.

        Attempts to break at newlines first, then spaces, to maintain readability.
        Avoids breaking mid-word if possible within the max_size limit.

        Args:
            text: The text content to split.
            max_size: The approximate maximum size for each chunk.

        Returns:
            A list of text chunks.
        """
        chunks = []
        start_index = 0
        while start_index < len(text):
            # Determine the end index for the current chunk
            end_index = start_index + max_size

            # If the chunk extends beyond the text length, take the rest
            if end_index >= len(text):
                chunk = text[start_index:]
                start_index = len(text) # Move index to the end
            else:
                # Find the best place to break within the potential chunk
                sub_chunk = text[start_index:end_index]
                last_newline = sub_chunk.rfind("\n")
                last_space = sub_chunk.rfind(" ")

                # Prefer breaking at the last newline, then last space
                break_at = -1
                if last_newline != -1:
                    break_at = last_newline + 1 # Include the newline in the break
                elif last_space != -1:
                    break_at = last_space + 1 # Include the space in the break

                # If a natural break point is found, use it
                if break_at > 0: # Use > 0 because rfind returns -1 if not found
                    chunk = text[start_index : start_index + break_at]
                    start_index += break_at # Move index past the break point
                else:
                    # No natural break found, force break at max_size
                    chunk = text[start_index:end_index]
                    start_index = end_index # Move index to the end of this chunk

            # Add the chunk if it's not just whitespace
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _prepare_llm_messages(self, history: list[ChatMessage]) -> list[dict[str, Any]]:
        """Adds system instructions and formats messages for the LLM API."""
        # Prepend system instruction as a user message (required by some models like Claude)
        # Note: Ideally, system instructions should use the 'system' role, but compatibility needs checking.
        messages_with_instr = [ChatMessage.from_system(self.system_instr)] + history
        # Convert to the format expected by the litellm.completion function (e.g., OpenAI format)
        return [msg.to_openai_format() for msg in messages_with_instr]

    def _get_llm_parameters(self) -> dict[str, Any]:
        """Returns model-specific parameters for the litellm completion call."""
        params: dict[str, Any] = {
            "max_tokens": 128000, # Default max tokens
        }
        # Add model-specific reasoning/thinking parameters if applicable
        if self.model.value.startswith("o"): # Example for 'o' models
            params["reasoning_effort"] = "high"
        elif self.model == TextModel.CLAUDE_37_SONNET:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000, # Example budget
            }
            params["max_completion_tokens"] = 64000 # Example completion token limit

        return params

    def _update_thread_status(self, status: str) -> None:
        """Updates the status message shown in the Slack thread header."""
        try:
            self.client.assistant_threads_setStatus(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                status=status,
            )
        except Exception as e:
            logger.error(f"Failed to update thread status to '{status}': {e}")

    def _update_thread_title(self, messages_for_llm: list[dict[str, Any]]) -> None:
        """Generates a title based on the conversation and updates the Slack thread title."""
        try:
            title = generate_title(messages_for_llm)
            self.client.assistant_threads_setTitle(
                channel_id=self.channel_id,
                thread_ts=self.thread_ts,
                title=title,
            )
        except Exception as e:
            logger.error(f"Failed to update thread title: {e}")


    def _generate_non_streaming_response(self, messages_for_llm: list[dict[str, Any]], llm_params: dict[str, Any]) -> None:
        """Generates a response from the LLM without streaming and sends it."""
        self._update_thread_status(f"{self.model.value} is generating...")
        try:
            response = completion(
                model=self.model.value,
                messages=messages_for_llm,
                **llm_params,
            )
            # Extract reasoning and message content
            # Note: .get("reasoning_content", "") might be specific to certain models/litellm setup
            reasoning_content = response.choices[0].get("reasoning_content", "") if self.show_thoughts else ""
            full_text: str = response.choices[0].message.content or "" # Ensure content is string

            # Send reasoning content if enabled and present
            if reasoning_content:
                formatted_reasoning = f"<thinking>\n{reasoning_content}\n</thinking>\n\n"
                for chunk in self.break_message(formatted_reasoning):
                    self.say(text=chunk)

            # Send the main response content in chunks
            for chunk in self.break_message(full_text):
                self.say(text=chunk)

            # Update title after successful generation
            self._update_thread_title(messages_for_llm)

        except Exception as e:
            logger.error(f"Error during non-streaming generation: {e}")
            self.say(text=f"{ERROR_HEADER}{e}") # Send error to user
        finally:
            # Clear status after completion or error
             self._update_thread_status("")


    def _generate_streaming_response(self, messages_for_llm: list[dict[str, Any]], llm_params: dict[str, Any]) -> None:
        """Generates a response from the LLM with streaming and updates the message."""
        initial_status_text = f"[[ {self.model.value} ]] Thinking ..."
        self._update_thread_status(initial_status_text.strip("[] ")) # Set initial status

        # Post an initial message to be updated
        try:
            initial_post = self.client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=self.thread_ts,
                text=initial_status_text,
            )
            message_ts = initial_post["ts"]
        except Exception as e:
            logger.error(f"Failed to post initial streaming message: {e}")
            self.say(text=f"{ERROR_HEADER}Failed to start streaming response.")
            self._update_thread_status("") # Clear status
            return

        last_update_time = time.time()
        update_interval = 2.0  # Start with 2 seconds interval
        start_time = time.time()
        current_message_content = ""
        currently_thinking = False # Track if the current chunk is reasoning
        stream_error = None

        try:
            # Start the streaming completion call
            response_stream = completion(
                model=self.model.value,
                messages=messages_for_llm,
                stream=True,
                **llm_params,
            )

            for chunk in response_stream:
                # Extract reasoning and content delta from the current chunk
                # Note: Accessing delta might differ slightly based on LLM provider via litellm
                delta = chunk.choices[0].delta
                last_reasoning_chunk: str = delta.get("reasoning_content", "") or ""
                last_content_chunk: str = delta.content or ""

                # Update thread status based on whether reasoning or content is received
                if self.show_thoughts and len(last_reasoning_chunk) > 0:
                    self._update_thread_status(f"{self.model.value} is thinking...")
                elif len(last_content_chunk) > 0:
                     self._update_thread_status(f"{self.model.value} is generating...")

                # Skip reasoning chunks if thoughts are disabled
                if not self.show_thoughts:
                    last_reasoning_chunk = ""

                # Handle transitions between thinking and generating states
                if not currently_thinking and len(last_reasoning_chunk) > 0:
                    # Started thinking
                    currently_thinking = True
                    last_reasoning_chunk = f"<thinking>\n{last_reasoning_chunk}" # Add opening tag
                elif currently_thinking and len(last_reasoning_chunk) == 0 and len(last_content_chunk) > 0:
                    # Finished thinking, starting to generate content
                    currently_thinking = False
                    # Update the existing message with the closing tag for thoughts
                    try:
                        self.client.chat_update(
                            channel=self.channel_id,
                            ts=message_ts,
                            text=f"{current_message_content}\n</thinking>\n\n", # Append closing tag
                        )
                    except Exception as e:
                         logger.warning(f"Minor error updating message at thought end: {e}")

                    # Start a *new* message for the actual response content
                    try:
                        new_post = self.client.chat_postMessage(
                            channel=self.channel_id,
                            thread_ts=self.thread_ts,
                            text=f"... [[ {self.model.value} generating response ]] ...",
                        )
                        message_ts = new_post["ts"] # Update message_ts to the new message
                        current_message_content = "" # Reset content for the new message
                    except Exception as e:
                        logger.error(f"Failed to post new message after thinking: {e}")
                        # Attempt to continue updating the previous message as fallback
                        current_message_content += "\n</thinking>\n\n" # Add closing tag anyway


                # Append the latest chunks to the current message content
                current_message_content += last_reasoning_chunk + last_content_chunk
                current_time = time.time()

                # --- Update Slack Message Periodically or if Too Long ---
                # Check if it's time to update the Slack message or if it exceeds size limit
                if (current_time - last_update_time >= update_interval) or len(current_message_content) > 2400:
                    last_update_time = current_time
                    update_text = f"{current_message_content} ... [[ {self.model.value} {'thinking' if currently_thinking else 'generating'} ]] ..."

                    # If message is too long, finalize the current one and start a new one
                    if len(current_message_content) > 2400:
                         # TODO: This logic might split mid-thought block if a thought is very long.
                         # Consider using break_message here if that's an issue.
                        try:
                            self.client.chat_update(
                                channel=self.channel_id,
                                ts=message_ts,
                                text=current_message_content, # Update with final content for this part
                            )
                            # Start a new message for continuation
                            new_post = self.client.chat_postMessage(
                                channel=self.channel_id,
                                thread_ts=self.thread_ts,
                                text=f"... [[ {self.model.value} {'thinking' if currently_thinking else 'generating'} ]] ...",
                            )
                            message_ts = new_post["ts"]
                            current_message_content = "" # Reset for the new message
                        except Exception as e:
                            logger.error(f"Failed to split long streaming message: {e}")
                            # Continue updating the existing message as fallback
                            try:
                                self.client.chat_update(channel=self.channel_id, ts=message_ts, text=update_text)
                            except Exception as update_e:
                                logger.error(f"Fallback update failed: {update_e}")
                    else:
                        # Just update the existing message
                        try:
                            self.client.chat_update(channel=self.channel_id, ts=message_ts, text=update_text)
                        except Exception as e:
                            logger.warning(f"Minor error updating streaming message: {e}")


                # Adjust update interval for very long generations
                if current_time - start_time > 60:
                    update_interval = 3.0 # Increase interval slightly

        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            stream_error = e # Store error to report later

        # --- Finalize Streaming ---
        try:
            # Final update to remove the "generating/thinking" suffix
            final_text = current_message_content
            # Add closing tag if stream ended mid-thought
            if currently_thinking and self.show_thoughts:
                final_text += "\n</thinking>"

            self.client.chat_update(
                channel=self.channel_id, ts=message_ts, text=final_text
            )

            # Report any error that occurred during the stream
            if stream_error:
                 self.say(text=f"{ERROR_HEADER}{stream_error}")

            # Update title only if streaming was successful (or partially successful)
            if not stream_error:
                 self._update_thread_title(messages_for_llm)

        except Exception as e:
            logger.error(f"Error finalizing streaming message: {e}")
            # Attempt to send final content as new message if update fails
            if not stream_error: # Avoid double-reporting errors
                 self.say(text=f"{current_message_content}\n\n{ERROR_HEADER}Failed to finalize stream.")

        finally:
            # Clear status after completion or error
            self._update_thread_status("")


    def process_direct_message(self, text: str, logger: Any) -> None:
        """
        Processes an incoming direct message or mention in the thread.

        Fetches history, handles commands, calls the appropriate LLM generation method
        (streaming or non-streaming), and updates the Slack thread.

        Args:
            text: The text content of the incoming Slack message.
            logger: The logger instance.
        """
        try:
            # 1. Fetch history and identify past commands
            messages, commands = self.fetch_conversation_history()

            # 2. Apply state changes from previous commands in the thread (e.g., model changes)
            # We don't resend the confirmation messages ('say' is None)
            for cmd in commands:
                 self.process_command(cmd, say=None) # Apply state changes silently

            # 3. Process the *current* message if it's a command
            if self.is_command(text):
                # Process the command and send confirmation back to the user ('say' is self.say)
                if self.process_command(text, say=self.say):
                    return # If it was a valid command, stop processing further

            # 4. Prepare messages for the LLM
            messages_for_llm = self._prepare_llm_messages(messages)
            logger.debug(f"Messages prepared for LLM: {messages_for_llm}")

            # 5. Get LLM parameters
            llm_params = self._get_llm_parameters()
            logger.debug(f"LLM parameters: {llm_params}")

            # 6. Generate response using the appropriate mode
            if self.streaming_mode:
                self._generate_streaming_response(messages_for_llm, llm_params)
            else:
                self._generate_non_streaming_response(messages_for_llm, llm_params)

        except Exception as e:
            logger.exception("An unexpected error occurred in process_direct_message")
            try:
                # Attempt to notify the user in Slack about the failure
                self.say(text=f"{ERROR_HEADER}An unexpected error occurred:\n```\n{e}\n```")
            except Exception as notify_e:
                logger.error(f"Failed to notify user about the error: {notify_e}")
