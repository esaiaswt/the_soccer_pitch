"""Unit tests for team/main.py entry point.

Tests cover:
- Config loading and validation (exits on error)
- Port finding (success and PortUnavailableError)
- Streamlit subprocess launch with correct arguments
- KeyboardInterrupt handling for clean shutdown
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from team.config import TeamConfig
from team.main import main


def _make_config(**overrides) -> TeamConfig:
    """Create a TeamConfig with sensible defaults for testing."""
    defaults = {
        "pitch_host": "localhost",
        "pitch_port": 8000,
        "nvidia_api_key": "test-key",
        "coach_model": "meta/llama-3.3-70b-instruct",
        "player_model": "meta/llama-3.1-8b-instruct",
        "coaching_frequency": 7.0,
        "poll_interval": 1.0,
        "streamlit_port": 8501,
        "team_color": "Red",
        "coach_memory_size": 50, "agent_name": "TeamBot",
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


@patch("team.main.subprocess.run")
@patch("team.main.find_available_port")
@patch("team.main.setup_logging")
@patch("team.main.load_config")
def test_main_launches_streamlit_with_correct_port(
    mock_load_config, mock_setup_logging, mock_find_port, mock_subprocess_run
):
    """main() should launch streamlit with the port from find_available_port."""
    config = _make_config(streamlit_port=8505, team_color="Blue")
    mock_load_config.return_value = config
    mock_find_port.return_value = 8505
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    main()

    mock_load_config.assert_called_once()
    mock_setup_logging.assert_called_once_with("Blue")
    mock_find_port.assert_called_once_with(8505)

    # Verify subprocess.run was called with correct streamlit args
    call_args = mock_subprocess_run.call_args
    cmd = call_args[0][0]
    assert "streamlit" in cmd[2]  # -m streamlit
    assert "run" in cmd[3]
    assert "--server.port=8505" in cmd
    assert "--server.headless=true" in cmd
    assert call_args[1]["check"] is False


@patch("team.main.subprocess.run")
@patch("team.main.find_available_port")
@patch("team.main.setup_logging")
@patch("team.main.load_config")
def test_main_uses_none_streamlit_port(
    mock_load_config, mock_setup_logging, mock_find_port, mock_subprocess_run
):
    """main() should pass None to find_available_port when streamlit_port is None."""
    config = _make_config(streamlit_port=None)
    mock_load_config.return_value = config
    mock_find_port.return_value = 8501
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    main()

    mock_find_port.assert_called_once_with(None)


@patch("team.main.find_available_port")
@patch("team.main.setup_logging")
@patch("team.main.load_config")
def test_main_exits_on_port_unavailable(
    mock_load_config, mock_setup_logging, mock_find_port
):
    """main() should exit with code 1 when no port is available."""
    from team.port_scanner import PortUnavailableError

    config = _make_config()
    mock_load_config.return_value = config
    mock_find_port.side_effect = PortUnavailableError("All ports occupied")

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


@patch("team.main.subprocess.run")
@patch("team.main.find_available_port")
@patch("team.main.setup_logging")
@patch("team.main.load_config")
def test_main_handles_keyboard_interrupt(
    mock_load_config, mock_setup_logging, mock_find_port, mock_subprocess_run
):
    """main() should exit cleanly on KeyboardInterrupt."""
    config = _make_config()
    mock_load_config.return_value = config
    mock_find_port.return_value = 8501
    mock_subprocess_run.side_effect = KeyboardInterrupt()

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


@patch("team.main.subprocess.run")
@patch("team.main.find_available_port")
@patch("team.main.setup_logging")
@patch("team.main.load_config")
def test_main_prints_url(
    mock_load_config, mock_setup_logging, mock_find_port, mock_subprocess_run, capsys
):
    """main() should print the dashboard URL before launching."""
    config = _make_config()
    mock_load_config.return_value = config
    mock_find_port.return_value = 8503
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    main()

    captured = capsys.readouterr()
    assert "http://localhost:8503" in captured.out
