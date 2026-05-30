"""Unit tests for the State Poller.

Tests cover:
- Successful polling updates SharedState
- HTTP errors are handled without crashing, preserving last good snapshot
- Connection timeouts are handled gracefully
- stop_event causes clean shutdown within one polling interval
- URL is constructed correctly from config
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

from team.config import TeamConfig
from team.shared_state import SharedState
from team.state_poller import StatePoller


def _make_config(**overrides) -> TeamConfig:
    """Create a TeamConfig with sensible defaults for testing."""
    defaults = {
        "pitch_host": "localhost",
        "pitch_port": 8000,
        "nvidia_api_key": "test-key",
        "coach_model": "test-model",
        "player_model": "test-model",
        "coaching_frequency": 7.0,
        "poll_interval": 0.1,  # Fast polling for tests
        "streamlit_port": None,
        "team_color": "Red",
        "coach_memory_size": 50, "agent_name": "TeamBot",
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


class TestStatePollerInit:
    """Tests for StatePoller initialization."""

    def test_url_construction(self):
        config = _make_config(pitch_host="192.168.1.10", pitch_port=9000)
        shared_state = SharedState()
        stop_event = threading.Event()

        poller = StatePoller(config, shared_state, stop_event)

        assert poller._url == "http://192.168.1.10:9000/api/state"

    def test_url_default_localhost(self):
        config = _make_config()
        shared_state = SharedState()
        stop_event = threading.Event()

        poller = StatePoller(config, shared_state, stop_event)

        assert poller._url == "http://localhost:8000/api/state"


class TestStatePollerRun:
    """Tests for the StatePoller.run() thread target."""

    @patch("team.state_poller.requests.get")
    def test_successful_poll_updates_shared_state(self, mock_get):
        """A successful GET response should update SharedState with parsed JSON."""
        game_state = {"match_state": "Playing", "ball": {"x": 100, "y": 200}}
        mock_response = MagicMock()
        mock_response.json.return_value = game_state
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        # Run poller in a thread, let it poll once, then stop
        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.15)  # Allow at least one poll cycle
        stop_event.set()
        thread.join(timeout=2.0)

        assert shared_state.get_snapshot() == game_state
        assert shared_state.get_last_update_time() is not None

    @patch("team.state_poller.requests.get")
    def test_http_error_preserves_last_good_snapshot(self, mock_get):
        """On HTTP error, SharedState should retain the previous good snapshot."""
        good_state = {"match_state": "Playing", "ball": {"x": 50, "y": 50}}

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        shared_state.set_snapshot(good_state)
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        # Simulate HTTP 500 error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

        # Last good snapshot should be preserved
        assert shared_state.get_snapshot() == good_state

    @patch("team.state_poller.requests.get")
    def test_timeout_preserves_last_good_snapshot(self, mock_get):
        """On connection timeout, SharedState should retain the previous good snapshot."""
        good_state = {"match_state": "Waiting", "time_left": 90}

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        shared_state.set_snapshot(good_state)
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

        assert shared_state.get_snapshot() == good_state

    @patch("team.state_poller.requests.get")
    def test_connection_error_preserves_last_good_snapshot(self, mock_get):
        """On connection error, SharedState should retain the previous good snapshot."""
        good_state = {"match_state": "Playing", "score": {"Red": 1, "Blue": 0}}

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        shared_state.set_snapshot(good_state)
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

        assert shared_state.get_snapshot() == good_state

    @patch("team.state_poller.requests.get")
    def test_stop_event_causes_clean_shutdown(self, mock_get):
        """Setting stop_event should cause the poller to exit within one poll interval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"match_state": "Playing"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        config = _make_config(poll_interval=0.5)  # Longer interval to test responsiveness
        shared_state = SharedState()
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.1)  # Let it start

        start = time.time()
        stop_event.set()
        thread.join(timeout=2.0)
        elapsed = time.time() - start

        # Should stop well within one poll interval (0.5s)
        assert elapsed < 0.5
        assert not thread.is_alive()

    @patch("team.state_poller.requests.get")
    def test_polls_at_configured_interval(self, mock_get):
        """The poller should call GET at approximately the configured interval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"match_state": "Playing"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.25)  # Allow ~5 poll cycles at 0.05s interval
        stop_event.set()
        thread.join(timeout=2.0)

        # Should have polled multiple times
        assert mock_get.call_count >= 3

    @patch("team.state_poller.requests.get")
    def test_request_uses_5_second_timeout(self, mock_get):
        """The GET request should use a 5-second timeout."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"match_state": "Playing"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=2.0)

        # Verify timeout=5 was passed to requests.get
        mock_get.assert_called_with("http://localhost:8000/api/state", timeout=5)

    @patch("team.state_poller.requests.get")
    def test_no_update_on_json_decode_error(self, mock_get):
        """On malformed JSON, SharedState should not be updated."""
        config = _make_config(poll_interval=0.05)
        shared_state = SharedState()
        stop_event = threading.Event()
        poller = StatePoller(config, shared_state, stop_event)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "", 0
        )
        mock_get.return_value = mock_response

        thread = threading.Thread(target=poller.run)
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

        # No snapshot should have been set
        assert shared_state.get_snapshot() is None
