"""Configuration module for the multi-agent team application.

Loads and validates all settings from team/.env. Exposes a frozen TeamConfig
dataclass with validated fields.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class TeamConfig:
    """Immutable configuration for a team instance."""

    pitch_host: str
    pitch_port: int
    nvidia_api_key: str
    coach_model: str
    player_model: str
    coaching_frequency: float
    poll_interval: float
    streamlit_port: int | None
    team_color: str
    coach_memory_size: int
    agent_name: str


def load_config(env_path: str | Path | None = None) -> TeamConfig:
    """Load configuration from team/.env and return a validated TeamConfig.

    Parameters
    ----------
    env_path : str | Path | None
        Optional explicit path to the .env file. If None, defaults to
        ``team/.env`` relative to the project root (parent of team/).

    Returns
    -------
    TeamConfig
        A frozen dataclass with all validated configuration values.

    Raises
    ------
    SystemExit
        If the NVIDIA API key is missing/empty or any parameter is out of range.
    """
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    else:
        env_path = Path(env_path)

    load_dotenv(env_path)

    # --- NVIDIA API Key (required) ---
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not nvidia_api_key:
        sys.exit("Error: NVIDIA_API_KEY is missing or empty. Please set it in team/.env")

    # --- Pitch Server ---
    pitch_host = os.getenv("PITCH_HOST", "localhost").strip()
    try:
        pitch_port = int(os.getenv("PITCH_PORT", "8000"))
    except ValueError:
        sys.exit("Error: PITCH_PORT must be an integer")

    # --- LLM Models ---
    coach_model = os.getenv("COACH_MODEL", "meta/llama-3.3-70b-instruct").strip()
    player_model = os.getenv("PLAYER_MODEL", "meta/llama-3.1-8b-instruct").strip()

    # --- Coaching Frequency ---
    coaching_frequency_str = os.getenv("COACHING_FREQUENCY", "7")
    try:
        coaching_frequency = float(coaching_frequency_str)
    except ValueError:
        sys.exit("Error: COACHING_FREQUENCY must be a number")

    if coaching_frequency < 2 or coaching_frequency > 30:
        sys.exit(
            f"Error: COACHING_FREQUENCY={coaching_frequency} is out of range. "
            f"Valid range: 2 to 30 seconds."
        )

    # --- Poll Interval ---
    poll_interval_str = os.getenv("POLL_INTERVAL", "1")
    try:
        poll_interval = float(poll_interval_str)
    except ValueError:
        sys.exit("Error: POLL_INTERVAL must be a number")

    if poll_interval < 0.1 or poll_interval > 10:
        sys.exit(
            f"Error: POLL_INTERVAL={poll_interval} is out of range. "
            f"Valid range: 0.1 to 10 seconds."
        )

    # --- Streamlit Port ---
    streamlit_port_str = os.getenv("STREAMLIT_PORT", "").strip()
    if streamlit_port_str:
        try:
            streamlit_port = int(streamlit_port_str)
        except ValueError:
            sys.exit("Error: STREAMLIT_PORT must be an integer")

        if streamlit_port < 1024 or streamlit_port > 65535:
            sys.exit(
                f"Error: STREAMLIT_PORT={streamlit_port} is out of range. "
                f"Valid range: 1024 to 65535."
            )
    else:
        streamlit_port = None

    # --- Team Color ---
    team_color = os.getenv("TEAM_COLOR", "Red").strip()

    # --- Coach Memory Size ---
    coach_memory_size_str = os.getenv("COACH_MEMORY_SIZE", "50")
    try:
        coach_memory_size = int(coach_memory_size_str)
    except ValueError:
        sys.exit("Error: COACH_MEMORY_SIZE must be an integer")

    # --- Agent Name ---
    agent_name = os.getenv("AGENT_NAME", "TeamBot").strip()

    return TeamConfig(
        pitch_host=pitch_host,
        pitch_port=pitch_port,
        nvidia_api_key=nvidia_api_key,
        coach_model=coach_model,
        player_model=player_model,
        coaching_frequency=coaching_frequency,
        poll_interval=poll_interval,
        streamlit_port=streamlit_port,
        team_color=team_color,
        coach_memory_size=coach_memory_size,
        agent_name=agent_name,
    )
