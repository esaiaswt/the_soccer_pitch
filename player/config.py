"""Configuration constants and Pydantic models for the Agent Control Panel."""

from pydantic import BaseModel, Field


# Server connection
DEFAULT_SERVER_IP = "localhost"
SERVER_PORT = 8000
REQUEST_TIMEOUT = 5  # seconds
LLM_TIMEOUT = 10  # seconds
LOOP_DELAY = 1.5  # seconds

# Input length limits
MAX_AGENT_NAME_LENGTH = 50
MAX_SYSTEM_PROMPT_LENGTH = 2000
MAX_BEHAVIOR_OVERRIDE_LENGTH = 500

# Default system prompt for the LLM
DEFAULT_SYSTEM_PROMPT = (
    "You are a soccer player agent on a 1200x800 pitch. "
    "You receive pre-computed spatial analysis with your position, ball direction, "
    "and distances. "
    "\n\nMOVEMENT: Use the 'ball_direction' dx/dy values to move toward the ball. "
    "\n\nSHOOTING STRATEGY: The kick pushes the ball AWAY from you. To score, you must "
    "position yourself BEHIND the ball (between the ball and your own goal) before kicking. "
    "Check 'Behind ball' — if YES and 'In kick range' is YES, set kick=True. "
    "If you are in kick range but NOT behind the ball, do NOT kick. Instead move behind "
    "the ball first, then kick on the next turn. "
    "\n\nAlways respond with dx (float -1 to 1), dy (float -1 to 1), and kick (boolean)."
)

# Team and position options
TEAMS = ["Red", "Blue"]
POSITIONS = ["Striker", "Goalkeeper", "Midfielder", "Defender"]


class ActionModel(BaseModel):
    """Structured output schema for LLM decisions."""

    dx: float = Field(ge=-1.0, le=1.0, description="Horizontal movement direction")
    dy: float = Field(ge=-1.0, le=1.0, description="Vertical movement direction")
    kick: bool = Field(description="Whether to kick the ball")


# Default safe action used when LLM fails or errors occur
BRAKE_ACTION = ActionModel(dx=0.0, dy=0.0, kick=False)


def validate_api_key(api_key: str | None) -> bool:
    """Validate that an API key is not empty, None, or whitespace-only.

    Args:
        api_key: The API key string to validate.

    Returns:
        True if the API key is valid, False otherwise.
    """
    if api_key is None:
        return False
    if not isinstance(api_key, str):
        return False
    if api_key.strip() == "":
        return False
    return True


def build_url(server_ip: str, endpoint: str) -> str:
    """Construct a full API URL from server IP and endpoint path.

    Args:
        server_ip: The server hostname or IP address.
        endpoint: The API endpoint path (e.g., "state" or "action").

    Returns:
        Full URL in the format http://{server_ip}:8000/api/{endpoint}
    """
    return f"http://{server_ip}:{SERVER_PORT}/api/{endpoint}"
