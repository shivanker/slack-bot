import logging
import os
import traceback

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from session import ERROR_HEADER, ChatSession

# Configure logger
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m%d %H:%M:%S",
)

# Initializes your app with your bot token and socket mode handler
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)

# Also initialize the WebClient with bot token
client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))


@slack_app.event("app_mention")
def handle_mention(body, say, logger):
    logger.debug(body)
    text = body["event"]["text"]
    app_user_id = body["authorizations"][0]["user_id"]
    sender_id = body["event"]["user"]
    channel_id = body["event"].get("channel") or None
    thread_ts = body["event"].get("thread_ts") or None
    say(
        "Hi! Currently I only support direct messages. Please ping Shiv with your use-case for adding support for mentions."
    )


@slack_app.event("message")
def handle_message(body, say, logger):
    logger.debug(body)
    event = body["event"]
    channel_type = event.get("channel_type")
    if channel_type != "im":
        return
    channel_id = event.get("channel") or ""
    user_id = event["user"]
    text = event.get("text") or ""

    try:
        user_session = ChatSession(user_id, channel_id, client)
        user_session.process_direct_message(text, say, logger)
    except Exception as e:
        say(ERROR_HEADER + "\n```\n" + str(e) + "\n```\n")
        traceback.print_exc()


slack_handler = SlackRequestHandler(app=slack_app)


def lambda_handler(event, context):
    return slack_handler.handle(event, context)


# Start your app
if __name__ == "__main__":
    # SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    slack_app.start(port=int(os.environ.get("PORT", 80)))
