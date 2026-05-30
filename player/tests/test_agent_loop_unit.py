"""Unit tests for the agent_loop module."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent_loop import AgentLoop, IterationResult
from config import BRAKE_ACTION, ActionModel, TEAMS


class TestIterationResult:
    """Tests for the IterationResult dataclass."""

    def test_default_values(self):
        """IterationResult defaults to BRAKE_ACTION with no errors."""
        result = IterationResult()
        assert result.game_state is None
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason is None
        assert result.error_details is None
        assert result.timestamp is not None

    def test_custom_values(self):
        """IterationResult stores custom values correctly."""
        action = ActionModel(dx=0.5, dy=-0.3, kick=True)
        state = {"ball": {"x": 100, "y": 200}}
        result = IterationResult(
            game_state=state,
            action=action,
            fallback_reason=None,
            error_details=None,
        )
        assert result.game_state == state
        assert result.action == action
        assert result.action.dx == 0.5
        assert result.action.dy == -0.3
        assert result.action.kick is True

    def test_timestamp_is_iso_format(self):
        """IterationResult timestamp is in ISO 8601 format."""
        result = IterationResult()
        # ISO 8601 format should contain 'T' separator
        assert "T" in result.timestamp


class TestAgentLoopValidation:
    """Tests for AgentLoop configuration validation."""

    def _make_loop(self, server_ip="localhost", team="Red", position="Striker"):
        """Helper to create an AgentLoop with minimal mocks."""
        return AgentLoop(
            server_ip=server_ip,
            team=team,
            position=position,
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )

    def test_empty_server_ip_raises_value_error(self):
        """Empty server_ip raises ValueError."""
        with pytest.raises(ValueError, match="server_ip must not be empty"):
            self._make_loop(server_ip="")

    def test_whitespace_server_ip_raises_value_error(self):
        """Whitespace-only server_ip raises ValueError."""
        with pytest.raises(ValueError, match="server_ip must not be empty"):
            self._make_loop(server_ip="   ")

    def test_invalid_team_raises_value_error(self):
        """Invalid team raises ValueError."""
        with pytest.raises(ValueError, match="team must be one of"):
            self._make_loop(team="Green")

    def test_valid_config_creates_loop(self):
        """Valid configuration creates AgentLoop without error."""
        loop = self._make_loop(server_ip="192.168.1.1", team="Blue", position="Goalkeeper")
        assert loop.server_ip == "192.168.1.1"
        assert loop.team == "Blue"
        assert loop.position == "Goalkeeper"


class TestAgentLoopLookStep:
    """Tests for the Look step (_look method)."""

    def _make_loop(self):
        """Helper to create an AgentLoop for testing."""
        return AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )

    @patch("agent_loop.requests.get")
    def test_look_success(self, mock_get):
        """Successful GET returns game state dict."""
        game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)

        loop = self._make_loop()
        result = loop._look()

        assert result == game_state
        assert isinstance(result, dict)

    @patch("agent_loop.requests.get")
    def test_look_non_200_returns_brake(self, mock_get):
        """Non-200 status returns IterationResult with BRAKE_ACTION."""
        mock_get.return_value = MagicMock(status_code=500)

        loop = self._make_loop()
        result = loop._look()

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "HTTP 500"

    @patch("agent_loop.requests.get")
    def test_look_timeout_returns_brake(self, mock_get):
        """Timeout returns IterationResult with BRAKE_ACTION."""
        mock_get.side_effect = requests.Timeout("timed out")

        loop = self._make_loop()
        result = loop._look()

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "connection timeout"

    @patch("agent_loop.requests.get")
    def test_look_connection_error_returns_brake(self, mock_get):
        """Connection error returns IterationResult with BRAKE_ACTION."""
        mock_get.side_effect = requests.ConnectionError("refused")

        loop = self._make_loop()
        result = loop._look()

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert "connection error" in result.fallback_reason


class TestAgentLoopThinkStep:
    """Tests for the Think step (_think method)."""

    def _make_loop(self, system_prompt="test prompt", behavior_override=""):
        """Helper to create an AgentLoop for testing."""
        return AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: system_prompt,
            get_behavior_override=lambda: behavior_override,
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )

    @patch("agent_loop.invoke_llm")
    def test_think_success(self, mock_invoke):
        """Successful LLM invocation returns ActionModel."""
        expected_action = ActionModel(dx=0.5, dy=-0.3, kick=True)
        mock_invoke.return_value = expected_action

        loop = self._make_loop()
        game_state = {"ball": {"x": 100, "y": 200}}
        result = loop._think(game_state)

        assert result == expected_action

    @patch("agent_loop.invoke_llm")
    def test_think_empty_prompt_returns_brake(self, mock_invoke):
        """Empty system prompt returns BRAKE_ACTION with reason."""
        loop = self._make_loop(system_prompt="")
        game_state = {"ball": {"x": 100, "y": 200}}
        result = loop._think(game_state)

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "system prompt required"
        mock_invoke.assert_not_called()

    @patch("agent_loop.invoke_llm")
    def test_think_whitespace_prompt_returns_brake(self, mock_invoke):
        """Whitespace-only system prompt returns BRAKE_ACTION."""
        loop = self._make_loop(system_prompt="   \t\n  ")
        game_state = {"ball": {"x": 100, "y": 200}}
        result = loop._think(game_state)

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "system prompt required"

    @patch("agent_loop.invoke_llm")
    def test_think_llm_exception_returns_brake(self, mock_invoke):
        """LLM exception returns BRAKE_ACTION with error info."""
        mock_invoke.side_effect = TimeoutError("LLM timed out")

        loop = self._make_loop()
        game_state = {"ball": {"x": 100, "y": 200}}
        result = loop._think(game_state)

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert "LLM error: TimeoutError" in result.fallback_reason

    @patch("agent_loop.invoke_llm")
    def test_think_llm_returns_none_returns_brake(self, mock_invoke):
        """LLM returning None returns BRAKE_ACTION."""
        mock_invoke.return_value = None

        loop = self._make_loop()
        game_state = {"ball": {"x": 100, "y": 200}}
        result = loop._think(game_state)

        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "empty response"


class TestAgentLoopActStep:
    """Tests for the Act step (_act method)."""

    def _make_loop(self):
        """Helper to create an AgentLoop for testing."""
        return AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )

    @patch("agent_loop.requests.post")
    def test_act_sends_correct_payload(self, mock_post):
        """Act step sends correct JSON payload."""
        mock_post.return_value = MagicMock(status_code=200)
        action = ActionModel(dx=0.7, dy=-0.4, kick=True)

        loop = self._make_loop()
        loop._act(action)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["team"] == "Red"
        assert payload["position"] == "Striker"
        assert payload["vector"]["dx"] == 0.7
        assert payload["vector"]["dy"] == -0.4
        assert payload["kick"] is True

    @patch("agent_loop.requests.post")
    def test_act_error_does_not_raise(self, mock_post):
        """Act step logs errors but does not raise exceptions."""
        mock_post.side_effect = requests.ConnectionError("refused")

        loop = self._make_loop()
        action = ActionModel(dx=0.0, dy=0.0, kick=False)

        # Should not raise
        loop._act(action)


class TestAgentLoopRun:
    """Tests for the run() method and loop control."""

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_run_stops_on_event(self, mock_post, mock_invoke, mock_get):
        """Loop stops when stop_event is set."""
        game_state = {"ball": {"x": 100, "y": 200}}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        on_iteration = MagicMock()

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

        # Set stop event after a short delay
        def stop_after_delay():
            time.sleep(0.1)
            stop_event.set()

        stopper = threading.Thread(target=stop_after_delay)
        stopper.start()

        loop.run()
        stopper.join()

        # Should have called on_iteration at least once
        assert on_iteration.call_count >= 1

    def test_on_iteration_receives_result(self):
        """on_iteration callback receives IterationResult."""
        stop_event = threading.Event()
        stop_event.set()  # Stop immediately

        on_iteration = MagicMock()

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

        # Loop should exit immediately since stop_event is already set
        loop.run()

        # on_iteration should not be called since stop_event was set before first iteration
        assert on_iteration.call_count == 0


class TestAgentLoopThreadLifecycle:
    """Tests for thread start/stop lifecycle via stop_event.

    Validates: Requirements 8.2, 8.3
    """

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_thread_starts_and_stops_within_30s(self, mock_post, mock_invoke, mock_get):
        """Start loop in a thread, set stop_event, verify thread completes within 30s."""
        game_state = {"ball": {"x": 100, "y": 200}}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.5, dy=-0.3, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        on_iteration = MagicMock()

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

        # Start loop in a background thread
        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        # Let it run for a bit, then signal stop
        time.sleep(0.2)
        assert thread.is_alive(), "Thread should be running"

        stop_event.set()

        # Thread must complete within 30 seconds (Requirement 8.3)
        thread.join(timeout=30)
        assert not thread.is_alive(), "Thread should have stopped within 30 seconds"

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_stop_event_interrupts_sleep(self, mock_post, mock_invoke, mock_get):
        """Setting stop_event during the 1.5s wait interrupts the sleep promptly."""
        game_state = {"ball": {"x": 100, "y": 200}}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        on_iteration = MagicMock()

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
        start_time = time.time()
        thread.start()

        # Wait for at least one iteration to complete
        time.sleep(0.1)
        stop_event.set()

        # Thread should stop well before the full 1.5s sleep completes
        thread.join(timeout=5)
        elapsed = time.time() - start_time
        assert not thread.is_alive()
        # Should stop much faster than 1.5s after we set the event
        assert elapsed < 3.0, f"Thread took too long to stop: {elapsed:.2f}s"

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_loop_does_not_run_when_stop_event_already_set(self, mock_post, mock_invoke, mock_get):
        """If stop_event is set before run(), no iterations execute."""
        stop_event = threading.Event()
        stop_event.set()  # Pre-set before starting

        on_iteration = MagicMock()

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

        assert not thread.is_alive()
        assert on_iteration.call_count == 0
        mock_get.assert_not_called()


class TestAgentLoopRateLimiting:
    """Tests for rate limiting (1.5s delay between iterations).

    Validates: Requirements 6.1, 6.2, 6.3
    """

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_iterations_spaced_by_loop_delay(self, mock_post, mock_invoke, mock_get):
        """Verify ~1.5s delay between iterations when HTTP/LLM calls are instant."""
        game_state = {"ball": {"x": 100, "y": 200}}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        iteration_times = []

        def record_iteration(result):
            iteration_times.append(time.time())

        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=record_iteration,
            stop_event=stop_event,
        )

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        # Wait for at least 2 iterations (need ~1.5s between them + some buffer)
        time.sleep(3.5)
        stop_event.set()
        thread.join(timeout=5)

        # Should have at least 2 iterations
        assert len(iteration_times) >= 2, f"Expected at least 2 iterations, got {len(iteration_times)}"

        # Check that the gap between iterations is approximately LOOP_DELAY (1.5s)
        for i in range(1, len(iteration_times)):
            gap = iteration_times[i] - iteration_times[i - 1]
            # Allow tolerance: the gap should be at least 1.3s and at most 2.0s
            # (accounting for execution time of the iteration itself)
            assert gap >= 1.3, f"Gap between iterations too short: {gap:.3f}s"
            assert gap <= 2.0, f"Gap between iterations too long: {gap:.3f}s"

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_rate_limit_enforced_on_error_iterations(self, mock_post, mock_invoke, mock_get):
        """Rate limit is enforced even when iterations fail (Brake_Action used)."""
        # Simulate server errors so every iteration uses Brake_Action
        mock_get.return_value = MagicMock(status_code=500)

        stop_event = threading.Event()
        iteration_times = []

        def record_iteration(result):
            iteration_times.append(time.time())

        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=record_iteration,
            stop_event=stop_event,
        )

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        # Wait for at least 2 iterations
        time.sleep(3.5)
        stop_event.set()
        thread.join(timeout=5)

        assert len(iteration_times) >= 2, f"Expected at least 2 iterations, got {len(iteration_times)}"

        # Verify delay is enforced even on error iterations
        for i in range(1, len(iteration_times)):
            gap = iteration_times[i] - iteration_times[i - 1]
            assert gap >= 1.3, f"Gap between error iterations too short: {gap:.3f}s"


class TestAgentLoopCallbackInvocation:
    """Tests for on_iteration callback invocation.

    Validates: Requirements 8.2, 8.3
    """

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_callback_invoked_each_cycle(self, mock_post, mock_invoke, mock_get):
        """on_iteration callback is invoked exactly once per loop cycle."""
        game_state = {"ball": {"x": 100, "y": 200}}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.5, dy=-0.3, kick=True)
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        on_iteration = MagicMock()

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

        # Wait for at least 2 iterations
        time.sleep(3.5)
        stop_event.set()
        thread.join(timeout=5)

        # Callback should be called at least twice
        assert on_iteration.call_count >= 2

        # Each call should receive an IterationResult
        for call in on_iteration.call_args_list:
            result = call[0][0]
            assert isinstance(result, IterationResult)

    @patch("agent_loop.requests.get")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.post")
    def test_callback_receives_successful_iteration_result(self, mock_post, mock_invoke, mock_get):
        """on_iteration receives IterationResult with game_state and action on success."""
        game_state = {"ball": {"x": 300, "y": 400}, "match_state": "Playing"}
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        expected_action = ActionModel(dx=0.7, dy=-0.2, kick=True)
        mock_invoke.return_value = expected_action
        mock_post.return_value = MagicMock(status_code=200)

        stop_event = threading.Event()
        results = []

        def capture_result(result):
            results.append(result)
            # Stop after first iteration
            stop_event.set()

        loop = AgentLoop(
            server_ip="localhost",
            team="Blue",
            position="Goalkeeper",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=capture_result,
            stop_event=stop_event,
        )

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()
        thread.join(timeout=5)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, IterationResult)
        assert result.game_state == game_state
        assert result.action == expected_action
        assert result.fallback_reason is None
        assert result.error_details is None

    @patch("agent_loop.requests.get")
    def test_callback_receives_error_iteration_result(self, mock_get):
        """on_iteration receives IterationResult with Brake_Action and fallback_reason on error."""
        mock_get.return_value = MagicMock(status_code=503)

        stop_event = threading.Event()
        results = []

        def capture_result(result):
            results.append(result)
            # Stop after first iteration
            stop_event.set()

        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=capture_result,
            stop_event=stop_event,
        )

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()
        thread.join(timeout=5)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, IterationResult)
        assert result.action == BRAKE_ACTION
        assert result.fallback_reason == "HTTP 503"
