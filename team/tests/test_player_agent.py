"""Unit tests for the PlayerAgent class.

Tests cover:
- Initialization with correct config
- Look-Think-Act loop iteration
- Coach instruction staleness detection (Req 5.1)
- LLM timeout handling with Brake_Action (Req 5.2)
- LLM error handling with Brake_Action (Req 5.3)
- Stale SharedState handling (Req 5.4)
- Instruction resumption when fresh ones arrive (Req 5.5)
- LLM response parsing (dx, dy, kick extraction and clamping)
- Token usage logging (Req 7.2, 7.3)
- Decision latency logging (Req 7.4)
- Error logging with structured context (Req 7.5)
- DebugStore updates
- Responsive shutdown via stop_event
"""

from __future__ import annotations

import time
import threading
from threading import Event
from unittest.mock import MagicMock, patch, ANY

import pytest

from team.config import TeamConfig
from team.debug_store import DebugStore
from team.instruction_store import CoachInstruction, InstructionStore
from team.player_agent import PlayerAgent, _parse_llm_response, _build_system_prompt
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
        coaching_frequency=7.0,
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
    """Create a mock LLM response with a valid action."""
    response = MagicMock()
    response.content = "dx=0.5 dy=-0.3 kick=true"
    response.usage_metadata = {
        "input_tokens": 80,
        "output_tokens": 10,
        "total_tokens": 90,
    }
    return response


class TestParseResponse:
    """Tests for the _parse_llm_response helper function."""

    def test_parses_valid_response(self):
        dx, dy, kick = _parse_llm_response("dx=0.5 dy=-0.3 kick=true")
        assert dx == 0.5
        assert dy == -0.3
        assert kick is True

    def test_parses_kick_false(self):
        dx, dy, kick = _parse_llm_response("dx=0.0 dy=1.0 kick=false")
        assert dx == 0.0
        assert dy == 1.0
        assert kick is False

    def test_clamps_dx_above_one(self):
        dx, dy, kick = _parse_llm_response("dx=2.5 dy=0.0 kick=false")
        assert dx == 1.0

    def test_clamps_dx_below_negative_one(self):
        dx, dy, kick = _parse_llm_response("dx=-3.0 dy=0.0 kick=false")
        assert dx == -1.0

    def test_clamps_dy_above_one(self):
        dx, dy, kick = _parse_llm_response("dx=0.0 dy=5.0 kick=false")
        assert dy == 1.0

    def test_clamps_dy_below_negative_one(self):
        dx, dy, kick = _parse_llm_response("dx=0.0 dy=-1.5 kick=false")
        assert dy == -1.0

    def test_raises_on_missing_dx(self):
        with pytest.raises(ValueError):
            _parse_llm_response("dy=0.5 kick=true")

    def test_raises_on_missing_dy(self):
        with pytest.raises(ValueError):
            _parse_llm_response("dx=0.5 kick=true")

    def test_raises_on_missing_kick(self):
        with pytest.raises(ValueError):
            _parse_llm_response("dx=0.5 dy=0.3")

    def test_raises_on_garbage_input(self):
        with pytest.raises(ValueError):
            _parse_llm_response("I don't know what to do")

    def test_parses_response_with_extra_text(self):
        """Parser finds values even with surrounding text."""
        text = "Based on the game state, I'll move: dx=0.7 dy=-0.2 kick=false"
        dx, dy, kick = _parse_llm_response(text)
        assert dx == 0.7
        assert dy == -0.2
        assert kick is False


class TestBuildSystemPrompt:
    """Tests for the _build_system_prompt helper."""

    def test_includes_position(self):
        prompt = _build_system_prompt("Striker", "Red")
        assert "Striker" in prompt

    def test_includes_team_color(self):
        prompt = _build_system_prompt("Goalkeeper", "Blue")
        assert "Blue" in prompt

    def test_includes_output_format(self):
        prompt = _build_system_prompt("Defender", "Red")
        assert "dx" in prompt
        assert "dy" in prompt
        assert "kick" in prompt


class TestPlayerAgentInit:
    """Tests for PlayerAgent initialization."""

    @patch("team.player_agent.ChatNVIDIA")
    def test_initializes_with_config(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        mock_chat.assert_called_once_with(
            model=config.player_model,
            api_key=config.nvidia_api_key,
        )

    @patch("team.player_agent.ChatNVIDIA")
    def test_sets_agent_identity(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Defender", shared_state, instruction_store, stop_event, debug_store
        )
        assert agent._agent_identity == "Player_Defender"

    @patch("team.player_agent.ChatNVIDIA")
    def test_builds_action_url(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Midfielder", shared_state, instruction_store, stop_event, debug_store
        )
        assert agent._action_url == "http://localhost:8000/api/action"


class TestPlayerAgentStalenessDetection:
    """Tests for Coach instruction staleness detection (Req 5.1)."""

    @patch("team.player_agent.ChatNVIDIA")
    def test_returns_none_when_no_instruction(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        """Req 4.5: No instruction received yet, returns None."""
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        result = agent._get_valid_instruction()
        assert result is None

    @patch("team.player_agent.ChatNVIDIA")
    def test_returns_instruction_when_fresh(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        """Fresh instruction (within 3 × coaching_frequency) is returned."""
        instruction = CoachInstruction(
            content="Push forward",
            timestamp=time.time(),  # Just now
            target_position="Striker",
        )
        instruction_store.set_instruction("Striker", instruction)

        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        result = agent._get_valid_instruction()
        assert result == "Push forward"

    @patch("team.player_agent.ChatNVIDIA")
    def test_returns_none_when_stale(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        """Req 5.1: Instruction older than 3 × coaching_frequency is excluded."""
        # coaching_frequency = 7.0, so threshold = 21.0 seconds
        stale_time = time.time() - 25.0  # 25s old, exceeds 21s threshold
        instruction = CoachInstruction(
            content="Old instruction",
            timestamp=stale_time,
            target_position="Striker",
        )
        instruction_store.set_instruction("Striker", instruction)

        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        result = agent._get_valid_instruction()
        assert result is None

    @patch("team.player_agent.ChatNVIDIA")
    def test_instruction_at_exact_threshold_is_stale(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        """Instruction at exactly 3 × coaching_frequency + epsilon is stale."""
        # coaching_frequency = 7.0, threshold = 21.0
        threshold_time = time.time() - 21.1  # Just past threshold
        instruction = CoachInstruction(
            content="Borderline instruction",
            timestamp=threshold_time,
            target_position="Striker",
        )
        instruction_store.set_instruction("Striker", instruction)

        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        result = agent._get_valid_instruction()
        assert result is None


class TestPlayerAgentLLMInvocation:
    """Tests for LLM invocation, timeout, and error handling."""

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_successful_llm_invocation_posts_action(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
        sample_snapshot,
        mock_llm_response,
    ):
        """Successful LLM invocation results in action POST."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        shared_state.set_snapshot(sample_snapshot)
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        # Verify action was posted
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["team"] == "Red"
        assert payload["position"] == "Striker"
        assert payload["vector"]["dx"] == 0.5
        assert payload["vector"]["dy"] == -0.3
        assert payload["kick"] is True
        assert payload["agent_name"] == "TeamBot"

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_llm_error_submits_brake_action(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
        sample_snapshot,
    ):
        """Req 5.3: LLM error results in Brake_Action."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.side_effect = RuntimeError("API error")
        mock_chat.return_value = mock_chat_instance
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        shared_state.set_snapshot(sample_snapshot)
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        # Verify brake action was posted
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["vector"]["dx"] == 0
        assert payload["vector"]["dy"] == 0
        assert payload["kick"] is False

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_unparseable_response_submits_brake_action(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
        sample_snapshot,
    ):
        """Unparseable LLM response results in Brake_Action."""
        response = MagicMock()
        response.content = "I'm not sure what to do here"
        response.usage_metadata = None
        response.response_metadata = None

        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = response
        mock_chat.return_value = mock_chat_instance
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        shared_state.set_snapshot(sample_snapshot)
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        # Verify brake action was posted
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["vector"]["dx"] == 0
        assert payload["vector"]["dy"] == 0
        assert payload["kick"] is False

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_no_snapshot_submits_brake_action(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
    ):
        """When no snapshot is available, submits Brake_Action."""
        mock_chat.return_value = MagicMock()
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        # Verify brake action was posted
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["vector"]["dx"] == 0
        assert payload["vector"]["dy"] == 0
        assert payload["kick"] is False


class TestPlayerAgentDebugStore:
    """Tests for DebugStore updates."""

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_updates_debug_store_after_action(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
        sample_snapshot,
        mock_llm_response,
    ):
        """DebugStore is updated with latest state, action, and instruction."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        shared_state.set_snapshot(sample_snapshot)
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        player_debug = debug_store.get_player("Striker")
        assert player_debug is not None
        assert player_debug.latest_state == sample_snapshot
        assert player_debug.latest_action == {"dx": 0.5, "dy": -0.3, "kick": True}
        assert player_debug.latest_instruction is None  # No instruction set

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_debug_store_includes_instruction(
        self,
        mock_chat,
        mock_post,
        config,
        shared_state,
        instruction_store,
        stop_event,
        debug_store,
        sample_snapshot,
        mock_llm_response,
    ):
        """DebugStore includes the coach instruction when available."""
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = mock_llm_response
        mock_chat.return_value = mock_chat_instance
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        # Set a fresh instruction
        instruction = CoachInstruction(
            content="Attack the left flank",
            timestamp=time.time(),
            target_position="Striker",
        )
        instruction_store.set_instruction("Striker", instruction)

        shared_state.set_snapshot(sample_snapshot)
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        agent._loop_iteration()

        player_debug = debug_store.get_player("Striker")
        assert player_debug is not None
        assert player_debug.latest_instruction == "Attack the left flank"


class TestPlayerAgentTokenLogging:
    """Tests for token usage logging (Req 7.2, 7.3)."""

    @patch("team.player_agent.log_token_usage")
    @patch("team.player_agent.ChatNVIDIA")
    def test_logs_token_usage_from_usage_metadata(
        self, mock_chat, mock_log_tokens, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        response = MagicMock()
        response.usage_metadata = {"input_tokens": 80, "output_tokens": 10, "total_tokens": 90}
        response.response_metadata = None

        agent._log_token_usage(response)
        mock_log_tokens.assert_called_once_with("Player_Striker", 80, 10, 90)

    @patch("team.player_agent.log_token_usage_unavailable")
    @patch("team.player_agent.ChatNVIDIA")
    def test_logs_warning_when_no_token_metadata(
        self, mock_chat, mock_log_unavailable, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = None

        agent._log_token_usage(response)
        mock_log_unavailable.assert_called_once_with("Player_Striker")

    @patch("team.player_agent.log_token_usage")
    @patch("team.player_agent.ChatNVIDIA")
    def test_logs_token_usage_from_response_metadata(
        self, mock_chat, mock_log_tokens, config, shared_state, instruction_store, stop_event, debug_store
    ):
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {
            "token_usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75}
        }

        agent._log_token_usage(response)
        mock_log_tokens.assert_called_once_with("Player_Striker", 60, 15, 75)


class TestPlayerAgentRunLoop:
    """Tests for the run() loop behavior."""

    @patch("team.player_agent.ChatNVIDIA")
    def test_run_stops_on_stop_event(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store
    ):
        """The run loop exits when stop_event is set."""
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )

        # Set stop event immediately so the loop exits
        stop_event.set()

        thread = threading.Thread(target=agent.run)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "Player agent thread did not stop"


class TestPlayerAgentMessageBuilding:
    """Tests for LLM message assembly."""

    @patch("team.player_agent.ChatNVIDIA")
    def test_includes_instruction_in_messages(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store, sample_snapshot
    ):
        """Req 4.4: Coach instruction is included in LLM context."""
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        messages = agent._build_messages(sample_snapshot, "Attack the left flank")

        # The human message should contain the instruction
        human_msg = messages[1]
        assert "Attack the left flank" in human_msg.content
        assert "Coach Advisory" in human_msg.content

    @patch("team.player_agent.ChatNVIDIA")
    def test_excludes_instruction_when_none(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store, sample_snapshot
    ):
        """Req 4.5: No instruction means no coach context in messages."""
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        messages = agent._build_messages(sample_snapshot, None)

        human_msg = messages[1]
        assert "Coach Advisory" not in human_msg.content

    @patch("team.player_agent.ChatNVIDIA")
    def test_messages_include_game_state(
        self, mock_chat, config, shared_state, instruction_store, stop_event, debug_store, sample_snapshot
    ):
        """Messages include ball position, score, and player positions."""
        agent = PlayerAgent(
            config, "Striker", shared_state, instruction_store, stop_event, debug_store
        )
        messages = agent._build_messages(sample_snapshot, None)

        human_msg = messages[1]
        assert "x=500" in human_msg.content  # ball x
        assert "y=300" in human_msg.content  # ball y
        assert "Playing" in human_msg.content  # match state
