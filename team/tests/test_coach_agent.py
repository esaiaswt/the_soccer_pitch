"""Unit tests for the CoachAgent class.

Tests cover:
- Initialization with correct config
- Coaching cycle with successful LLM invocation
- Graceful handling of LLM failures
- Instruction parsing from LLM response
- Token usage logging (available and unavailable)
- Decision latency logging
- DebugStore updates
- Responsive shutdown via stop_event
"""

from __future__ import annotations

import time
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from team.coach_agent import CoachAgent, CoachMemory, PLAYER_POSITIONS
from team.config import TeamConfig
from team.debug_store import DebugStore
from team.instruction_store import InstructionStore
from team.shared_state import SharedState


@pytest.fixture
def config():
    """Create a minimal TeamConfig for testing."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test-api-key",
        coach_model="meta/llama-3.3-70b-instruct",
        player_model="meta/llama-3.1-8b-instruct",
        coaching_frequency=2.0,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


@pytest.fixture
def shared_state():
    return SharedState()


@pytest.fixture
def instruction_store():
    return InstructionStore()


@pytest.fixture
def stop_event():
    return Event()


@pytest.fixture
def debug_store():
    return DebugStore()


@pytest.fixture
def sample_snapshot():
    """A valid game state snapshot."""
    return {
        "match_state": "Playing",
        "time_left": 60.0,
        "score": {"Red": 1, "Blue": 0},
        "ball": {"x": 500.0, "y": 300.0},
        "players": {
            "Red_Goalkeeper": {"x": 50.0, "y": 425.0},
            "Red_Defender": {"x": 200.0, "y": 350.0},
            "Red_Midfielder": {"x": 400.0, "y": 400.0},
            "Red_Striker": {"x": 600.0, "y": 300.0},
            "Blue_Goalkeeper": {"x": 1150.0, "y": 425.0},
        },
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response with proper structure."""
    response = MagicMock()
    response.content = (
        "Goalkeeper: Stay near the goal line and watch for counter-attacks\n"
        "Defender: Push up to midfield to support the attack\n"
        "Midfielder: Control the center and distribute to the wings\n"
        "Striker: Make runs behind the defense to exploit space"
    )
    response.usage_metadata = {
        "input_tokens": 150,
        "output_tokens": 60,
        "total_tokens": 210,
    }
    return response


class TestCoachAgentInit:
    """Tests for CoachAgent initialization."""

    @patch("team.coach_agent.ChatNVIDIA")
    def test_initializes_with_config(self, mock_chat, config, shared_state, instruction_store, stop_event):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        mock_chat.assert_called_once_with(
            model=config.coach_model,
            api_key=config.nvidia_api_key,
        )

    @patch("team.coach_agent.ChatNVIDIA")
    def test_initializes_memory_with_config_size(self, mock_chat, config, shared_state, instruction_store, stop_event):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        assert agent._memory._max_size == config.coach_memory_size


class TestCoachAgentCoachingCycle:
    """Tests for the coaching cycle logic."""

    @patch("team.coach_agent.ChatNVIDIA")
    def test_skips_cycle_when_no_snapshot(self, mock_chat, config, shared_state, instruction_store, stop_event):
        """When SharedState has no snapshot, the cycle is skipped."""
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        agent._coaching_cycle()
        # No instructions should be stored
        assert instruction_store.get_all_instructions() == {}

    @patch("team.coach_agent.ChatNVIDIA")
    def test_stores_instructions_on_success(
        self, mock_chat, config, shared_state, instruction_store, stop_event, sample_snapshot, mock_llm_response
    ):
        """Successful LLM invocation stores instructions for all positions."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance

        shared_state.set_snapshot(sample_snapshot)
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        agent._coaching_cycle()

        # All 4 positions should have instructions
        all_instructions = instruction_store.get_all_instructions()
        assert len(all_instructions) == 4
        for position in PLAYER_POSITIONS:
            assert position in all_instructions
            assert all_instructions[position].content != ""
            assert all_instructions[position].target_position == position

    @patch("team.coach_agent.ChatNVIDIA")
    def test_handles_llm_failure_gracefully(
        self, mock_chat, config, shared_state, instruction_store, stop_event, sample_snapshot
    ):
        """LLM failure does not crash the agent and no instructions are stored."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.side_effect = RuntimeError("API timeout")
        mock_chat.return_value = mock_chat_instance

        shared_state.set_snapshot(sample_snapshot)
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        # Should not raise
        agent._coaching_cycle()
        # No instructions stored on failure
        assert instruction_store.get_all_instructions() == {}

    @patch("team.coach_agent.ChatNVIDIA")
    def test_adds_snapshot_to_memory(
        self, mock_chat, config, shared_state, instruction_store, stop_event, sample_snapshot, mock_llm_response
    ):
        """Each coaching cycle adds the current snapshot to CoachMemory."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance

        shared_state.set_snapshot(sample_snapshot)
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        agent._coaching_cycle()

        history = agent._memory.get_history()
        assert len(history) == 1
        assert history[0]["ball"] == sample_snapshot["ball"]

    @patch("team.coach_agent.ChatNVIDIA")
    def test_updates_debug_store(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store, sample_snapshot, mock_llm_response
    ):
        """Coaching cycle updates the DebugStore with observations and instructions."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance

        shared_state.set_snapshot(sample_snapshot)
        agent = CoachAgent(config, shared_state, instruction_store, stop_event, debug_store=debug_store)
        agent._coaching_cycle()

        coach_debug = debug_store.get_coach()
        assert "observations" in coach_debug
        assert "instructions" in coach_debug
        assert len(coach_debug["instructions"]) == 4


class TestCoachAgentParseInstructions:
    """Tests for instruction parsing from LLM response."""

    @patch("team.coach_agent.ChatNVIDIA")
    def test_parses_well_formatted_response(self, mock_chat, config, shared_state, instruction_store, stop_event):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        response_text = (
            "Goalkeeper: Guard the near post\n"
            "Defender: Mark the striker tightly\n"
            "Midfielder: Press high up the pitch\n"
            "Striker: Look for through balls"
        )
        result = agent._parse_instructions(response_text)
        assert result["Goalkeeper"] == "Guard the near post"
        assert result["Defender"] == "Mark the striker tightly"
        assert result["Midfielder"] == "Press high up the pitch"
        assert result["Striker"] == "Look for through balls"

    @patch("team.coach_agent.ChatNVIDIA")
    def test_fallback_on_unparseable_response(self, mock_chat, config, shared_state, instruction_store, stop_event):
        """If response can't be parsed, all positions get the full response."""
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        response_text = "Just play well everyone!"
        result = agent._parse_instructions(response_text)
        for position in PLAYER_POSITIONS:
            assert result[position] == response_text

    @patch("team.coach_agent.ChatNVIDIA")
    def test_handles_long_instructions_without_truncation(self, mock_chat, config, shared_state, instruction_store, stop_event):
        """Instructions of any length are preserved without truncation (Req 3.4)."""
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        long_instruction = "A" * 1000
        response_text = f"Goalkeeper: {long_instruction}\nDefender: Short\nMidfielder: Short\nStriker: Short"
        result = agent._parse_instructions(response_text)
        assert result["Goalkeeper"] == long_instruction
        assert len(result["Goalkeeper"]) == 1000


class TestCoachAgentTokenLogging:
    """Tests for token usage logging."""

    @patch("team.coach_agent.log_token_usage")
    @patch("team.coach_agent.ChatNVIDIA")
    def test_logs_token_usage_from_usage_metadata(
        self, mock_chat, mock_log_tokens, config, shared_state, instruction_store, stop_event
    ):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        response = MagicMock()
        response.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        response.response_metadata = None

        agent._log_token_usage(response)
        mock_log_tokens.assert_called_once_with("Coach", 100, 50, 150)

    @patch("team.coach_agent.log_token_usage_unavailable")
    @patch("team.coach_agent.ChatNVIDIA")
    def test_logs_warning_when_no_token_metadata(
        self, mock_chat, mock_log_unavailable, config, shared_state, instruction_store, stop_event
    ):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = None

        agent._log_token_usage(response)
        mock_log_unavailable.assert_called_once_with("Coach")

    @patch("team.coach_agent.log_token_usage")
    @patch("team.coach_agent.ChatNVIDIA")
    def test_logs_token_usage_from_response_metadata(
        self, mock_chat, mock_log_tokens, config, shared_state, instruction_store, stop_event
    ):
        agent = CoachAgent(config, shared_state, instruction_store, stop_event)
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {
            "token_usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        }

        agent._log_token_usage(response)
        mock_log_tokens.assert_called_once_with("Coach", 80, 40, 120)


class TestCoachAgentRunLoop:
    """Tests for the run() loop behavior."""

    @patch("team.coach_agent.ChatNVIDIA")
    def test_run_stops_on_stop_event(self, mock_chat, config, shared_state, instruction_store, stop_event):
        """The run loop exits when stop_event is set."""
        import threading

        agent = CoachAgent(config, shared_state, instruction_store, stop_event)

        # Set stop event immediately so the loop exits after first check
        stop_event.set()

        # Run in a thread with a timeout to prevent hanging
        thread = threading.Thread(target=agent.run)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "Coach agent thread did not stop"
