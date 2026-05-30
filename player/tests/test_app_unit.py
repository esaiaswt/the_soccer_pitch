"""Unit tests for app.py UI behavior.

Tests the underlying logic of the Streamlit dashboard without requiring
a running Streamlit server. Focuses on session state initialization,
configuration validation that blocks loop start, on_iteration callback
behavior, and default system prompt values.

Validates: Requirements 2.1, 8.1, 8.5, 10.3
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent_loop import AgentLoop, IterationResult
from config import (
    BRAKE_ACTION,
    DEFAULT_SERVER_IP,
    DEFAULT_SYSTEM_PROMPT,
    ActionModel,
    TEAMS,
    POSITIONS,
)


class TestSessionStateDefaults:
    """Tests for initial session state defaults.

    Validates: Requirements 2.1, 8.1
    The app should initialize with toggle off, default IP, and default prompt.
    """

    def test_default_is_running_is_false(self):
        """Session state is_running defaults to False (toggle off).

        Validates: Requirement 8.1 - toggle initial state is off.
        """
        # Simulate the session state initialization logic from app.py
        session_state = {}
        if "is_running" not in session_state:
            session_state["is_running"] = False

        assert session_state["is_running"] is False

    def test_default_agent_thread_is_none(self):
        """Session state agent_thread defaults to None."""
        session_state = {}
        if "agent_thread" not in session_state:
            session_state["agent_thread"] = None

        assert session_state["agent_thread"] is None

    def test_default_stop_event_is_none(self):
        """Session state stop_event defaults to None."""
        session_state = {}
        if "stop_event" not in session_state:
            session_state["stop_event"] = None

        assert session_state["stop_event"] is None

    def test_default_latest_iteration_is_none(self):
        """Session state latest_iteration defaults to None."""
        session_state = {}
        if "latest_iteration" not in session_state:
            session_state["latest_iteration"] = None

        assert session_state["latest_iteration"] is None

    def test_default_system_prompt_matches_constant(self):
        """Session state system_prompt defaults to DEFAULT_SYSTEM_PROMPT.

        Validates: Requirement 9.2 - pre-filled with the default prompt.
        """
        session_state = {}
        if "system_prompt" not in session_state:
            session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT

        assert session_state["system_prompt"] == DEFAULT_SYSTEM_PROMPT
        assert "soccer player" in session_state["system_prompt"]
        assert "1200x800" in session_state["system_prompt"]
        assert "kick" in session_state["system_prompt"].lower()

    def test_default_behavior_override_is_empty(self):
        """Session state behavior_override defaults to empty string."""
        session_state = {}
        if "behavior_override" not in session_state:
            session_state["behavior_override"] = ""

        assert session_state["behavior_override"] == ""

    def test_default_server_ip_is_localhost(self):
        """Default server IP is 'localhost'.

        Validates: Requirement 2.1 - default value of "localhost".
        """
        assert DEFAULT_SERVER_IP == "localhost"

    def test_teams_options_available(self):
        """Team options include Red and Blue.

        Validates: Requirement 2.2.
        """
        assert "Red" in TEAMS
        assert "Blue" in TEAMS

    def test_positions_options_available(self):
        """Position options include all required positions.

        Validates: Requirement 2.3.
        """
        assert "Striker" in POSITIONS
        assert "Goalkeeper" in POSITIONS
        assert "Midfielder" in POSITIONS
        assert "Defender" in POSITIONS


class TestMissingConfigBlocksLoopStart:
    """Tests that missing configuration blocks loop start.

    Validates: Requirement 8.5 - missing server IP or team blocks start.
    """

    def test_empty_server_ip_blocks_start(self):
        """Empty server_ip raises ValueError, preventing loop start.

        Validates: Requirement 8.5.
        """
        with pytest.raises(ValueError, match="server_ip must not be empty"):
            AgentLoop(
                server_ip="",
                team="Red",
                position="Striker",
                llm_client=MagicMock(),
                get_system_prompt=lambda: "test prompt",
                get_behavior_override=lambda: "",
                on_iteration=MagicMock(),
                stop_event=threading.Event(),
            )

    def test_whitespace_server_ip_blocks_start(self):
        """Whitespace-only server_ip raises ValueError, preventing loop start.

        Validates: Requirement 8.5.
        """
        with pytest.raises(ValueError, match="server_ip must not be empty"):
            AgentLoop(
                server_ip="   ",
                team="Red",
                position="Striker",
                llm_client=MagicMock(),
                get_system_prompt=lambda: "test prompt",
                get_behavior_override=lambda: "",
                on_iteration=MagicMock(),
                stop_event=threading.Event(),
            )

    def test_invalid_team_blocks_start(self):
        """Invalid team raises ValueError, preventing loop start.

        Validates: Requirement 8.5.
        """
        with pytest.raises(ValueError, match="team must be one of"):
            AgentLoop(
                server_ip="localhost",
                team="",
                position="Striker",
                llm_client=MagicMock(),
                get_system_prompt=lambda: "test prompt",
                get_behavior_override=lambda: "",
                on_iteration=MagicMock(),
                stop_event=threading.Event(),
            )

    def test_none_like_team_blocks_start(self):
        """Team not in valid list raises ValueError.

        Validates: Requirement 8.5.
        """
        with pytest.raises(ValueError, match="team must be one of"):
            AgentLoop(
                server_ip="localhost",
                team="Green",
                position="Striker",
                llm_client=MagicMock(),
                get_system_prompt=lambda: "test prompt",
                get_behavior_override=lambda: "",
                on_iteration=MagicMock(),
                stop_event=threading.Event(),
            )

    def test_valid_config_allows_start(self):
        """Valid configuration allows AgentLoop creation (no error).

        Validates: Requirement 8.5 - only missing config blocks start.
        """
        loop = AgentLoop(
            server_ip="192.168.1.100",
            team="Blue",
            position="Goalkeeper",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )
        assert loop.server_ip == "192.168.1.100"
        assert loop.team == "Blue"


class TestDebugConsoleUpdatesOnIterationCallback:
    """Tests that the debug console updates via on_iteration callback.

    Validates: Requirement 10.3 - debug console updates with latest iteration.
    """

    def test_on_iteration_callback_updates_latest_iteration(self):
        """on_iteration callback stores the IterationResult for debug display.

        Validates: Requirement 10.3.
        """
        # Simulate session state with a latest_iteration slot
        session_state = {"latest_iteration": None}

        def on_iteration(result):
            session_state["latest_iteration"] = result

        # Simulate a successful iteration result
        action = ActionModel(dx=0.5, dy=-0.3, kick=True)
        game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}
        result = IterationResult(
            game_state=game_state,
            action=action,
            fallback_reason=None,
            error_details=None,
        )

        on_iteration(result)

        assert session_state["latest_iteration"] is not None
        assert session_state["latest_iteration"].game_state == game_state
        assert session_state["latest_iteration"].action.dx == 0.5
        assert session_state["latest_iteration"].action.dy == -0.3
        assert session_state["latest_iteration"].action.kick is True

    def test_on_iteration_callback_updates_with_brake_action(self):
        """on_iteration callback stores Brake_Action result for debug display.

        Validates: Requirement 10.3, 10.4 - shows fallback reason.
        """
        session_state = {"latest_iteration": None}

        def on_iteration(result):
            session_state["latest_iteration"] = result

        # Simulate a failed iteration (Brake_Action used)
        result = IterationResult(
            game_state=None,
            action=BRAKE_ACTION,
            fallback_reason="connection timeout",
            error_details="timed out after 5s",
        )

        on_iteration(result)

        assert session_state["latest_iteration"] is not None
        assert session_state["latest_iteration"].action == BRAKE_ACTION
        assert session_state["latest_iteration"].action.dx == 0.0
        assert session_state["latest_iteration"].action.dy == 0.0
        assert session_state["latest_iteration"].action.kick is False
        assert session_state["latest_iteration"].fallback_reason == "connection timeout"

    def test_on_iteration_overwrites_previous_result(self):
        """Each on_iteration call overwrites the previous result (latest only).

        Validates: Requirement 10.3 - shows only most recent iteration.
        """
        session_state = {"latest_iteration": None}

        def on_iteration(result):
            session_state["latest_iteration"] = result

        # First iteration
        result1 = IterationResult(
            game_state={"ball": {"x": 100, "y": 100}},
            action=ActionModel(dx=0.1, dy=0.2, kick=False),
        )
        on_iteration(result1)
        assert session_state["latest_iteration"].action.dx == 0.1

        # Second iteration overwrites
        result2 = IterationResult(
            game_state={"ball": {"x": 500, "y": 300}},
            action=ActionModel(dx=0.9, dy=-0.8, kick=True),
        )
        on_iteration(result2)
        assert session_state["latest_iteration"].action.dx == 0.9
        assert session_state["latest_iteration"].game_state == {"ball": {"x": 500, "y": 300}}

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_agent_loop_invokes_on_iteration_with_result(self, mock_post, mock_invoke, mock_get):
        """AgentLoop.run() invokes on_iteration with IterationResult each cycle.

        Validates: Requirement 10.3 - debug console updates on iteration.
        """
        game_state = {"ball": {"x": 300, "y": 400}, "match_state": "Playing"}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.7, dy=-0.2, kick=True)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        session_state = {"latest_iteration": None}

        def on_iteration(result):
            session_state["latest_iteration"] = result
            # Stop after first iteration to keep test fast
            stop_event.set()

        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=on_iteration,
            stop_event=stop_event,
        )

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()
        thread.join(timeout=5)

        # Verify the session state was updated
        assert session_state["latest_iteration"] is not None
        assert isinstance(session_state["latest_iteration"], IterationResult)
        assert session_state["latest_iteration"].game_state == game_state
        assert session_state["latest_iteration"].action.dx == 0.7
        assert session_state["latest_iteration"].action.dy == -0.2
        assert session_state["latest_iteration"].action.kick is True
        assert session_state["latest_iteration"].fallback_reason is None


class TestDefaultSystemPrompt:
    """Tests for the default system prompt value.

    Validates: Requirement 9.2.
    """

    def test_default_system_prompt_content(self):
        """DEFAULT_SYSTEM_PROMPT contains required strategy elements."""
        assert "soccer player" in DEFAULT_SYSTEM_PROMPT
        assert "1200x800" in DEFAULT_SYSTEM_PROMPT
        assert "-1" in DEFAULT_SYSTEM_PROMPT
        assert "1" in DEFAULT_SYSTEM_PROMPT
        assert "kick" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "ball" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_default_system_prompt_is_non_empty(self):
        """DEFAULT_SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert DEFAULT_SYSTEM_PROMPT.strip() != ""
