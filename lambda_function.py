import logging
import os
import traceback

from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_sdk import WebClient

from session import ERROR_HEADER, ChatSession

# Configure logger
app = APIGatewayRestResolver()
logger = Logger()
metrics = Metrics(namespace="Powertools")

# Initializes your app with your bot token and socket mode handler
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    process_before_response=True,
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

    # Check if the message was sent by the user and not this app
    bot_user_id = slack_app.client.auth_test()["user_id"]
    if user_id == bot_user_id:
        logger.info("This message was sent by the bot itself. Ignoring.")
        return

    try:
        user_session = ChatSession(user_id, channel_id, client)
        user_session.process_direct_message(text, say, logger)
    except Exception as e:
        say(ERROR_HEADER + "\n```\n" + str(e) + "\n```\n")
        traceback.print_exc()


SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
)


@logger.inject_lambda_context(
    log_event=True, correlation_id_path=correlation_paths.API_GATEWAY_REST
)
def lambda_handler(event, context):
    slack_handler = SlackRequestHandler(app=slack_app)
    return slack_handler.handle(event, context)  # type:ignore


# Start your app
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m%d %H:%M:%S",
    )
    slack_app.start(port=int(os.environ.get("PORT", 80)))
