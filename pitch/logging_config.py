"""Logging configuration for The Pitch.

Provides a centralized logging setup with file handler writing to pitch.log
in append mode. All log entries use ISO 8601 timestamps.

Format: {ISO8601_TIMESTAMP} {LEVEL} {message}
Example: 2024-01-15T10:30:00.123456 INFO Server starting on 0.0.0.0:8000
"""

import logging
import os
import traceback

# Log file path relative to the pitch directory
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "pitch.log")

# ISO 8601 format with microseconds
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# Module-level logger
logger = logging.getLogger("pitch")


class ISO8601Formatter(logging.Formatter):
    """Custom formatter that outputs ISO 8601 timestamps with microseconds."""

    def formatTime(self, record, datefmt=None):
        """Format time as ISO 8601 with microseconds."""
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def setup_logging() -> logging.Logger:
    """Configure logging with file handler writing to pitch/pitch.log.

    Sets up:
    - File handler in append mode
    - ISO 8601 timestamp format
    - INFO level logging

    Should be called once at application startup from main.py.

    Returns:
        The configured 'pitch' logger instance.
    """
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler - append mode
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Apply ISO 8601 formatter
    formatter = ISO8601Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def log_startup(local_ip: str, host: str, port: int) -> None:
    """Log server startup information.

    Args:
        local_ip: Detected local IP address.
        host: Bound host address.
        port: Bound port number.
    """
    logger.info(f"Server starting - Local_IP={local_ip}, host={host}, port={port}")


def log_player_spawn(player_name: str, team: str, x: float, y: float) -> None:
    """Log a player spawn event.

    Args:
        player_name: The player's unique name (e.g., Red_Striker).
        team: The player's team (Red or Blue).
        x: Starting x coordinate.
        y: Starting y coordinate.
    """
    logger.info(f"Player spawned: {player_name} team={team} at ({x}, {y})")


def log_goal(scoring_team: str, score: dict) -> None:
    """Log a goal scoring event.

    Args:
        scoring_team: The team that scored (Red or Blue).
        score: Current score dict, e.g. {"Red": 1, "Blue": 0}.
    """
    logger.info(
        f"Goal scored by {scoring_team} - Score: Red={score.get('Red', 0)}, Blue={score.get('Blue', 0)}"
    )


def log_state_transition(old_state: str, new_state: str, trigger: str) -> None:
    """Log a match state transition.

    Args:
        old_state: Previous match state (e.g., "Waiting").
        new_state: New match state (e.g., "Playing").
        trigger: What caused the transition (e.g., "spacebar", "timer_expiry").
    """
    logger.info(
        f"State transition: {old_state} -> {new_state} (trigger={trigger})"
    )


def log_api_request(method: str, path: str, status_code: int) -> None:
    """Log an API request completion.

    Args:
        method: HTTP method (GET or POST).
        path: Endpoint path (e.g., /api/state).
        status_code: HTTP response status code.
    """
    logger.info(f"API request: {method} {path} -> {status_code}")


def log_error(exception: BaseException) -> None:
    """Log an error with full traceback.

    Args:
        exception: The exception that occurred.
    """
    tb = traceback.format_exception(type(exception), exception, exception.__traceback__)
    logger.error(f"{type(exception).__name__}: {exception}\n{''.join(tb)}")
