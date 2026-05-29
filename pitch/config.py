"""Configuration module for The Pitch.

Loads environment variables from .env and provides typed access
to all application constants via the Config dataclass.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file from the pitch directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class Config:
    """Application configuration with all game constants."""

    # Network settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Pitch dimensions
    PITCH_WIDTH: int = 1200
    PITCH_HEIGHT: int = 800

    # Player movement
    MAX_SPEED: int = 20
    POSSESSION_RANGE: int = 30

    # Match settings
    MATCH_DURATION: float = 90.0

    # Physics settings
    PHYSICS_TICK_RATE: int = 60
    RENDER_FPS: int = 30
    FRICTION: float = 0.97
    MAX_BALL_SPEED: float = 40.0
    KICK_IMPULSE: float = 20.0

    # Goal settings
    GOAL_PAUSE: float = 2.0

    # Lock settings
    LOCK_TIMEOUT: float = 5.0
