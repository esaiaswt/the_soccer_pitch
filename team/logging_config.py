"""Structured, thread-safe logging configuration for the multi-agent team.

Provides team-specific log files (e.g., team_red.log, team_blue.log) with
ISO 8601 timestamps at microsecond precision. All log entries follow the format:

    {timestamp} | {level} | {agent_identity} | {message} | {structured_context}

Python's logging module is inherently thread-safe for individual log calls,
so concurrent writes from State Poller, Coach Agent, and Player Agent threads
will not interleave or corrupt log entries.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.5
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone


# Module-level logger — configured per team instance via setup_logging()
_team_logger: logging.Logger | None = None


class TeamLogFormatter(logging.Formatter):
    """Custom formatter producing ISO 8601 timestamps with microsecond precision.

    Output format:
        {ISO8601_timestamp} | {level} | {agent_identity} | {message} | {context}

    The agent_identity and structured_context are passed via the LogRecord's
    extra dict.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format time as ISO 8601 with microsecond precision."""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with pipe-delimited structured fields."""
        timestamp = self.formatTime(record)
        level = record.levelname
        agent_identity = getattr(record, "agent_identity", "SYSTEM")
        structured_context = getattr(record, "structured_context", "")
        message = record.getMessage()

        return f"{timestamp} | {level} | {agent_identity} | {message} | {structured_context}"


def setup_logging(team_color: str) -> logging.Logger:
    """Configure structured logging for a team instance.

    Creates a file handler writing to a team-specific log file in append mode
    (e.g., ``team_red.log``, ``team_blue.log``) located in the ``team/`` directory.

    Parameters
    ----------
    team_color : str
        The team color (e.g., "Red" or "Blue"). Used to name the log file.

    Returns
    -------
    logging.Logger
        The configured team logger instance.
    """
    global _team_logger

    logger_name = f"team_{team_color.lower()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        _team_logger = logger
        return logger

    # Log file in the team/ directory
    log_filename = f"team_{team_color.lower()}.log"
    log_path = os.path.join(os.path.dirname(__file__), log_filename)

    # File handler — append mode, UTF-8 encoding
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Apply structured formatter
    formatter = TeamLogFormatter()
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    _team_logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the configured team logger.

    Returns
    -------
    logging.Logger
        The team logger. Falls back to a basic logger named 'team' if
        setup_logging() has not been called yet.
    """
    if _team_logger is None:
        return logging.getLogger("team")
    return _team_logger


def log_coach_instruction(target_player: str, content: str) -> None:
    """Log a Coach instruction sent to a Player Agent.

    Parameters
    ----------
    target_player : str
        The target player identity (e.g., "Goalkeeper", "Striker").
    content : str
        The natural-language instruction content.

    Validates: Requirement 7.1
    """
    logger = get_logger()
    logger.info(
        "Coach instruction issued",
        extra={
            "agent_identity": "Coach",
            "structured_context": f"target={target_player} | content={content}",
        },
    )


def log_token_usage(
    agent_identity: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Log token usage metadata from an LLM invocation.

    Parameters
    ----------
    agent_identity : str
        The agent that made the invocation (e.g., "Coach", "Player_Striker").
    prompt_tokens : int
        Number of prompt tokens used.
    completion_tokens : int
        Number of completion tokens used.
    total_tokens : int
        Total tokens used.

    Validates: Requirement 7.2
    """
    logger = get_logger()
    logger.info(
        "Token usage",
        extra={
            "agent_identity": agent_identity,
            "structured_context": (
                f"prompt_tokens={prompt_tokens} | "
                f"completion_tokens={completion_tokens} | "
                f"total_tokens={total_tokens}"
            ),
        },
    )


def log_token_usage_unavailable(agent_identity: str) -> None:
    """Log a warning when token usage metadata is unavailable.

    Parameters
    ----------
    agent_identity : str
        The agent that made the invocation.

    Validates: Requirement 7.3
    """
    logger = get_logger()
    logger.warning(
        "Token usage metadata unavailable",
        extra={
            "agent_identity": agent_identity,
            "structured_context": "token_data=unavailable",
        },
    )


def log_decision_latency(agent_identity: str, latency_ms: float) -> None:
    """Log decision latency for an LLM invocation.

    Parameters
    ----------
    agent_identity : str
        The agent that made the invocation.
    latency_ms : float
        Time in milliseconds from invocation start to response received.

    Validates: Requirement 7.4
    """
    logger = get_logger()
    logger.info(
        "Decision latency",
        extra={
            "agent_identity": agent_identity,
            "structured_context": f"latency_ms={latency_ms:.2f}",
        },
    )


def log_agent_error(
    agent_identity: str,
    error_type: str,
    match_state: str,
    attempted_action: str,
    error_details: str,
) -> None:
    """Log an agent error with structured context.

    Parameters
    ----------
    agent_identity : str
        The agent that encountered the error.
    error_type : str
        The type/class of the error (e.g., "TimeoutError", "APIError").
    match_state : str
        The current match state when the error occurred (e.g., "Playing").
    attempted_action : str
        The action the agent was attempting (e.g., "generate_instruction").
    error_details : str
        Additional error details or message.

    Validates: Requirement 7.5
    """
    logger = get_logger()
    logger.error(
        f"Agent error: {error_details}",
        extra={
            "agent_identity": agent_identity,
            "structured_context": (
                f"error_type={error_type} | "
                f"match_state={match_state} | "
                f"attempted_action={attempted_action}"
            ),
        },
    )
