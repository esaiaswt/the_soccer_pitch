"""Unit tests for team/config.py."""

import os
import tempfile
from pathlib import Path

import pytest

from team.config import TeamConfig, load_config


@pytest.fixture
def env_file(tmp_path):
    """Create a temporary .env file with valid defaults."""
    env_content = """
NVIDIA_API_KEY=nvapi-test-key-12345
PITCH_HOST=localhost
PITCH_PORT=8000
COACH_MODEL=meta/llama-3.3-70b-instruct
PLAYER_MODEL=meta/llama-3.1-8b-instruct
COACHING_FREQUENCY=7
POLL_INTERVAL=1
STREAMLIT_PORT=
TEAM_COLOR=Red
COACH_MEMORY_SIZE=50
"""
    env_path = tmp_path / ".env"
    env_path.write_text(env_content)
    return env_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove config-related env vars before each test to avoid leakage."""
    for key in [
        "NVIDIA_API_KEY", "PITCH_HOST", "PITCH_PORT", "COACH_MODEL",
        "PLAYER_MODEL", "COACHING_FREQUENCY", "POLL_INTERVAL",
        "STREAMLIT_PORT", "TEAM_COLOR", "COACH_MEMORY_SIZE",
    ]:
        monkeypatch.delenv(key, raising=False)


class TestLoadConfigDefaults:
    """Test that defaults are applied correctly."""

    def test_defaults_applied(self, tmp_path, monkeypatch):
        """Minimal .env with only required key should use all defaults."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=nvapi-test-key\n")
        config = load_config(env_path)

        assert config.pitch_host == "localhost"
        assert config.pitch_port == 8000
        assert config.coaching_frequency == 7.0
        assert config.poll_interval == 1.0
        assert config.streamlit_port is None
        assert config.coach_memory_size == 50

    def test_all_fields_loaded(self, env_file):
        """All fields from a complete .env should be loaded."""
        config = load_config(env_file)

        assert config.nvidia_api_key == "nvapi-test-key-12345"
        assert config.pitch_host == "localhost"
        assert config.pitch_port == 8000
        assert config.coach_model == "meta/llama-3.3-70b-instruct"
        assert config.player_model == "meta/llama-3.1-8b-instruct"
        assert config.coaching_frequency == 7.0
        assert config.poll_interval == 1.0
        assert config.streamlit_port is None
        assert config.team_color == "Red"
        assert config.coach_memory_size == 50


class TestLoadConfigValidation:
    """Test validation and error exits."""

    def test_missing_nvidia_key_exits(self, tmp_path):
        """Missing NVIDIA_API_KEY should cause sys.exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("PITCH_HOST=localhost\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "NVIDIA_API_KEY" in str(exc_info.value)

    def test_empty_nvidia_key_exits(self, tmp_path):
        """Empty NVIDIA_API_KEY should cause sys.exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "NVIDIA_API_KEY" in str(exc_info.value)

    def test_coaching_frequency_too_low_exits(self, tmp_path):
        """coaching_frequency below 2 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nCOACHING_FREQUENCY=1.5\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "COACHING_FREQUENCY" in str(exc_info.value)

    def test_coaching_frequency_too_high_exits(self, tmp_path):
        """coaching_frequency above 30 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nCOACHING_FREQUENCY=31\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "COACHING_FREQUENCY" in str(exc_info.value)

    def test_poll_interval_too_low_exits(self, tmp_path):
        """poll_interval below 0.1 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nPOLL_INTERVAL=0.05\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "POLL_INTERVAL" in str(exc_info.value)

    def test_poll_interval_too_high_exits(self, tmp_path):
        """poll_interval above 10 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nPOLL_INTERVAL=11\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "POLL_INTERVAL" in str(exc_info.value)

    def test_streamlit_port_too_low_exits(self, tmp_path):
        """streamlit_port below 1024 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nSTREAMLIT_PORT=80\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "STREAMLIT_PORT" in str(exc_info.value)

    def test_streamlit_port_too_high_exits(self, tmp_path):
        """streamlit_port above 65535 should exit."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nSTREAMLIT_PORT=70000\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(env_path)
        assert "STREAMLIT_PORT" in str(exc_info.value)

    def test_valid_streamlit_port(self, tmp_path):
        """A valid streamlit_port should be accepted."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nSTREAMLIT_PORT=8501\n")
        config = load_config(env_path)
        assert config.streamlit_port == 8501

    def test_boundary_coaching_frequency_min(self, tmp_path):
        """coaching_frequency at exactly 2 should be accepted."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nCOACHING_FREQUENCY=2\n")
        config = load_config(env_path)
        assert config.coaching_frequency == 2.0

    def test_boundary_coaching_frequency_max(self, tmp_path):
        """coaching_frequency at exactly 30 should be accepted."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nCOACHING_FREQUENCY=30\n")
        config = load_config(env_path)
        assert config.coaching_frequency == 30.0

    def test_boundary_poll_interval_min(self, tmp_path):
        """poll_interval at exactly 0.1 should be accepted."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nPOLL_INTERVAL=0.1\n")
        config = load_config(env_path)
        assert config.poll_interval == 0.1

    def test_boundary_poll_interval_max(self, tmp_path):
        """poll_interval at exactly 10 should be accepted."""
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=key\nPOLL_INTERVAL=10\n")
        config = load_config(env_path)
        assert config.poll_interval == 10.0


class TestTeamConfigFrozen:
    """Test that TeamConfig is immutable."""

    def test_cannot_modify_fields(self, env_file):
        """Frozen dataclass should reject attribute assignment."""
        config = load_config(env_file)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.pitch_host = "other"
