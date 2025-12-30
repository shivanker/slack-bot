"""ADK Session Manager for managing InMemorySessionService instances."""

from google.adk.sessions import InMemorySessionService


class ADKSessionManager:
    """Manages ADK sessions using InMemorySessionService.

    Provides a singleton-like pattern for session service management,
    keyed by app_name for reuse across agent calls.
    """

    _instances: dict[str, "ADKSessionManager"] = {}

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.session_service = InMemorySessionService()

    @classmethod
    def get_instance(cls, app_name: str = "slack_bot") -> "ADKSessionManager":
        """Get or create an ADKSessionManager instance for the given app_name."""
        if app_name not in cls._instances:
            cls._instances[app_name] = cls(app_name)
        return cls._instances[app_name]

    async def get_or_create_session(self, user_id: str, session_id: str):
        """Get an existing session or create a new one.

        Args:
            user_id: The user identifier.
            session_id: The session identifier (e.g., Slack thread_ts).

        Returns:
            The ADK session object.
        """
        # Try to get existing session first
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        if session:
            return session

        # Create new session if it doesn't exist
        return await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
