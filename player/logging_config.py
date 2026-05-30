"""Logging configuration for the Agent Control Panel.

Provides file-based logging setup with a file handler writing to agent.log
in append mode. All log entries use ISO 8601 timestamps.

Format: {ISO_8601_TIMESTAMP} | {LEVEL} | {message}
Example: 2024-01-15T10:30:00 | INFO | Agent loop started
"""

import logging
import os
from datetime import datetime, timezone

# Log file path relative to the player directory
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "agent.log")

# Log format with pipe separators
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

# ISO 8601 date format (YYYY-MM-DDTHH:MM:SS)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Module-level logger
logger = logging.getLogger("agent")


class ISO8601Formatter(logging.Formatter):
    """Custom formatter that outputs ISO 8601 timestamps (YYYY-MM-DDTHH:MM:SS)."""

    def formatTime(self, record, datefmt=None):
        """Format time as ISO 8601 without microseconds."""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        return dt.strftime(DATE_FORMAT)


def setup_logging() -> logging.Logger:
    """Configure file-based logging to agent.log with ISO 8601 timestamps.

    Sets up:
    - File handler in append mode writing to player/agent.log
    - ISO 8601 timestamp format (YYYY-MM-DDTHH:MM:SS)
    - Log format: {ISO_8601_TIMESTAMP} | {LEVEL} | {message}
    - DEBUG level logging to capture all events

    Should be called once at application startup.

    Returns:
        The configured 'agent' logger instance.
    """
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler - append mode
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Apply ISO 8601 formatter with pipe-separated format
    formatter = ISO8601Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
