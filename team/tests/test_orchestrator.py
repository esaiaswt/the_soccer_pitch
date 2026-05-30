"""Unit tests for the Team Orchestrator.

Tests cover:
- start() creates and launches all expected threads (1 poller + 1 coach + 4 players)
- stop() signals all threads and joins them within timeout
- is_running() reflects thread health
- Multiple instances do not share mutable state (Req 8.1)
- stop() before start() does not crash
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from team.config import TeamConfig
from team.orchestrator import TeamOrchestrator, PLAYER_POSITIONS


def _make_config(**overrides) -> TeamConfig:
    """Create a TeamConfig with sensible defaults for testing."""
    defaults = {
        "pitch_host": "localhost",
        "pitch_port": 8000,
        "nvidia_api_key": "test-key",
        "coach_model": "test-model",
        "player_model": "test-model",
        "coaching_frequency": 7.0,
        "poll_interval": 0.1,
        "streamlit_port": None,
        "team_color": "Red",
        "coach_memory_size": 50, "agent_name": "TeamBot",
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


class TestTeamOrchestratorInit:
    """Tests for TeamOrchestrator initialization."""

    def test_initial_state(self):
        config = _make_config()
        orchestrator = TeamOrchestrator(config)

        assert orchestrator.debug_store is None
        assert orchestrator.instruction_store is None
        assert orchestrator.shared_state is None
        assert orchestrator.is_running() is False


class TestTeamOrchestratorStart:
    """Tests for TeamOrchestrator.start()."""

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_start_creates_six_threads(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """start() should create 6 threads: 1 poller + 1 coach + 4 players."""
        # Make run() methods do nothing (just return immediately)
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()

        # Should have 6 threads total
        assert len(orchestrator._threads) == 6

        # Clean up
        orchestrator.stop(timeout=2.0)

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_start_creates_fresh_state_containers(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """start() should create fresh SharedState, InstructionStore, and DebugStore."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()

        assert orchestrator.shared_state is not None
        assert orchestrator.instruction_store is not None
        assert orchestrator.debug_store is not None

        orchestrator.stop(timeout=2.0)

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_start_launches_player_for_each_position(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """start() should create a PlayerAgent for each of the 4 positions."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()

        # PlayerAgent should be instantiated 4 times, once per position
        assert mock_player_cls.call_count == 4
        positions_created = [
            call.kwargs["position"] for call in mock_player_cls.call_args_list
        ]
        assert set(positions_created) == set(PLAYER_POSITIONS)

        orchestrator.stop(timeout=2.0)

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_threads_are_daemon(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """All threads should be daemon threads."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()

        for thread in orchestrator._threads:
            assert thread.daemon is True

        orchestrator.stop(timeout=2.0)


class TestTeamOrchestratorStop:
    """Tests for TeamOrchestrator.stop()."""

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_stop_sets_stop_event(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """stop() should set the stop_event to signal all threads."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()
        orchestrator.stop(timeout=2.0)

        assert orchestrator._stop_event.is_set()

    def test_stop_before_start_does_not_crash(self):
        """Calling stop() before start() should not raise an exception."""
        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        # Should not raise
        orchestrator.stop(timeout=1.0)


class TestTeamOrchestratorIsRunning:
    """Tests for TeamOrchestrator.is_running()."""

    def test_is_running_false_before_start(self):
        """is_running() should return False before start() is called."""
        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        assert orchestrator.is_running() is False

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_is_running_false_after_stop(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """is_running() should return False after all threads have stopped."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config = _make_config()
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()

        # Wait for threads to finish (they return immediately due to mocked run())
        time.sleep(0.1)
        orchestrator.stop(timeout=2.0)

        assert orchestrator.is_running() is False


class TestMultiInstanceIsolation:
    """Tests for multi-instance support (Req 8.1)."""

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_two_instances_have_separate_state(self, mock_player_cls, mock_coach_cls, mock_poller_cls):
        """Two orchestrator instances should not share mutable state."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config_red = _make_config(team_color="Red")
        config_blue = _make_config(team_color="Blue")

        orch_red = TeamOrchestrator(config_red)
        orch_blue = TeamOrchestrator(config_blue)

        orch_red.start()
        orch_blue.start()

        # Verify separate state containers
        assert orch_red.shared_state is not orch_blue.shared_state
        assert orch_red.instruction_store is not orch_blue.instruction_store
        assert orch_red.debug_store is not orch_blue.debug_store
        assert orch_red._stop_event is not orch_blue._stop_event

        # Clean up
        orch_red.stop(timeout=2.0)
        orch_blue.stop(timeout=2.0)
