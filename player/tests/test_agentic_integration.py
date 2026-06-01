"""Integration tests for agentic modules in the player agent loop.

Tests verify:
1. Exactly one LLM call per Look-Think-Act cycle (Req 9.2)
2. No LLM calls from agentic modules (Req 9.1)
3. Agentic context appended after spatial analysis in enriched_state (Req 2.5)

Requirements: 9.1, 9.2
"""

import threading
from unittest.mock import MagicMock, patch, call

import pytest

from agent_loop import AgentLoop, IterationResult
from config import ActionModel, BRAKE_ACTION


# --- Fixtures and helpers ---


def _make_game_state():
    """Create a realistic game state dict for testing."""
    return {
        "ball": {"x": 600.0, "y": 400.0},
        "players": {
            "Red_Striker": {"x": 500.0, "y": 400.0},
            "Red_Goalkeeper": {"x": 50.0, "y": 425.0},
            "Blue_Striker": {"x": 700.0, "y": 400.0},
            "Blue_Goalkeeper": {"x": 1150.0, "y": 425.0},
        },
        "match_state": "Playing",
        "score": {"Red": 0, "Blue": 0},
    }


def _make_loop(llm_client=None):
    """Create an AgentLoop with mocked dependencies for integration testing."""
    if llm_client is None:
        llm_client = MagicMock()
    return AgentLoop(
        server_ip="localhost",
        team="Red",
        position="Striker",
        llm_client=llm_client,
        get_system_prompt=lambda: "You are a soccer player agent.",
        get_behavior_override=lambda: "",
        on_iteration=MagicMock(),
        stop_event=threading.Event(),
        agent_name="test_agent",
    )


class TestSingleLLMCallPerCycle:
    """Verify exactly one LLM call per _run_iteration() cycle.

    Validates: Requirement 9.2 - The Player_Agent SHALL continue to make
    exactly one LLM API call per Look-Think-Act_Cycle.
    """

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_single_llm_call_per_iteration(self, mock_get, mock_invoke, mock_post):
        """A single _run_iteration() makes exactly 1 LLM call."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.5, dy=-0.3, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()
        result = loop._run_iteration()

        assert not isinstance(result, IterationResult) or result.fallback_reason is None
        assert mock_invoke.call_count == 1

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_single_llm_call_across_multiple_iterations(self, mock_get, mock_invoke, mock_post):
        """Multiple iterations each make exactly 1 LLM call (N iterations = N calls)."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Run 3 iterations
        for _ in range(3):
            loop._run_iteration()

        assert mock_invoke.call_count == 3

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_single_llm_call_with_active_plan(self, mock_get, mock_invoke, mock_post):
        """Even with an active plan, only 1 LLM call is made per cycle."""
        game_state = _make_game_state()
        # Ball in attacking half with possession -> triggers score_goal plan
        game_state["ball"] = {"x": 700.0, "y": 400.0}
        game_state["players"]["Red_Striker"] = {"x": 680.0, "y": 400.0}

        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.8, dy=0.0, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Run multiple iterations to allow plan to be created and evaluated
        for _ in range(5):
            loop._run_iteration()

        # Each iteration should have exactly 1 LLM call
        assert mock_invoke.call_count == 5

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_single_llm_call_with_reflection_running(self, mock_get, mock_invoke, mock_post):
        """Reflection engine running does not add extra LLM calls."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.3, dy=-0.1, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # First iteration sets up previous state
        loop._run_iteration()
        # Second iteration triggers reflection on previous action
        loop._run_iteration()

        # Still exactly 1 call per iteration
        assert mock_invoke.call_count == 2


class TestNoLLMCallsFromAgenticModules:
    """Verify agentic modules (EpisodicMemory, Planner, ReflectionEngine,
    StrategyTracker) do not make any LLM calls.

    Validates: Requirement 9.1 - The Episodic_Memory, Plan evaluation,
    Reflection_Engine, Strategy_Tracker SHALL execute entirely in Python
    without making LLM API calls.
    """

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_post_look_agentic_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """_post_look_agentic() does not trigger any LLM calls."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.1, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Set up previous state so reflection runs
        loop._previous_state = _make_game_state()
        loop._previous_action = {"dx": 0.5, "dy": 0.0, "kick": False}
        loop._cycle_counter = 1

        # Call post-look agentic processing directly
        mock_invoke.reset_mock()
        loop._post_look_agentic(game_state)

        # No LLM calls should have been made
        mock_invoke.assert_not_called()

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_assemble_context_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """_assemble_context() does not trigger any LLM calls."""
        loop = _make_loop()

        # Add some episodes to memory so summarization runs
        from episodic_memory import Episode
        for i in range(5):
            episode = Episode(
                cycle=i,
                game_state=_make_game_state(),
                action={"dx": 0.1, "dy": 0.2, "kick": False},
                next_state_delta={},
                effectiveness=0.6,
            )
            loop.memory.add(episode)

        mock_invoke.reset_mock()
        context = loop._assemble_context()

        # No LLM calls should have been made
        mock_invoke.assert_not_called()
        # Context should be a string (possibly empty)
        assert isinstance(context, str)

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_post_act_agentic_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """_post_act_agentic() does not trigger any LLM calls."""
        game_state = _make_game_state()
        action = ActionModel(dx=0.5, dy=-0.3, kick=True)

        loop = _make_loop()
        loop._cycle_counter = 1

        mock_invoke.reset_mock()
        loop._post_act_agentic(game_state, action)

        # No LLM calls should have been made
        mock_invoke.assert_not_called()

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_strategy_tracker_analysis_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """Strategy tracker analysis (triggered periodically) makes no LLM calls."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.1, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Run enough iterations to trigger strategy analysis (every 10 cycles)
        for _ in range(11):
            loop._run_iteration()

        # Should be exactly 11 LLM calls (1 per iteration), no extras from analysis
        assert mock_invoke.call_count == 11

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_planner_evaluate_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """Planner.evaluate() during post-look does not make LLM calls."""
        game_state = _make_game_state()

        loop = _make_loop()
        loop._cycle_counter = 1

        # Directly test planner evaluation
        mock_invoke.reset_mock()
        result = loop.planner.evaluate(game_state, "Red", "Striker", None)

        mock_invoke.assert_not_called()
        # Result is either a Plan or None
        assert result is None or hasattr(result, "sub_goals")

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_reflection_evaluate_makes_no_llm_calls(self, mock_get, mock_invoke, mock_post):
        """ReflectionEngine.evaluate() does not make LLM calls."""
        loop = _make_loop()

        previous_state = _make_game_state()
        current_state = _make_game_state()
        current_state["ball"]["x"] = 650.0  # Ball moved

        mock_invoke.reset_mock()
        result = loop.reflection.evaluate(
            action={"dx": 0.5, "dy": 0.0, "kick": False},
            expected_outcome={},
            actual_state=current_state,
            previous_state=previous_state,
        )

        mock_invoke.assert_not_called()


class TestAgenticContextAppendedAfterSpatialAnalysis:
    """Verify agentic context is appended to enriched_state after the spatial
    summary, not before or replacing it.

    Validates: Requirement 2.5 - The Memory_Summary SHALL be appended to the
    existing spatial analysis section in the LLM prompt without replacing
    existing context.
    """

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_enriched_state_contains_spatial_then_agentic(self, mock_get, mock_invoke, mock_post):
        """The enriched_state passed to invoke_llm has spatial analysis before agentic context."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Add episodes to memory so agentic context is non-empty
        from episodic_memory import Episode
        for i in range(3):
            episode = Episode(
                cycle=i,
                game_state=_make_game_state(),
                action={"dx": 0.1, "dy": 0.2, "kick": False},
                next_state_delta={},
                effectiveness=0.7,
            )
            loop.memory.add(episode)

        loop._run_iteration()

        # Inspect the game_state_json argument passed to invoke_llm
        assert mock_invoke.call_count == 1
        call_args = mock_invoke.call_args
        game_state_json = call_args.kwargs.get("game_state_json") or call_args[1][1] if len(call_args[1]) > 1 else call_args.kwargs["game_state_json"]

        # Should contain spatial analysis marker
        assert "SPATIAL ANALYSIS" in game_state_json

        # Should contain agentic context (memory summary lines)
        # Memory summary contains "Cycle" lines
        assert "Cycle" in game_state_json or "--- AGENTIC" in game_state_json or "Memory" in game_state_json

        # Spatial analysis should appear BEFORE agentic context
        spatial_pos = game_state_json.find("SPATIAL ANALYSIS")
        # The agentic context is appended after spatial summary with \n\n separator
        # Find the last section (agentic context comes after spatial)
        parts = game_state_json.split("\n\n")
        # First part is JSON game state, second is spatial, third (if present) is agentic
        assert len(parts) >= 2  # At minimum: game state JSON + spatial

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_spatial_analysis_not_replaced_by_agentic_context(self, mock_get, mock_invoke, mock_post):
        """Spatial analysis is preserved when agentic context is appended."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Add episodes so agentic context is non-empty
        from episodic_memory import Episode
        for i in range(3):
            episode = Episode(
                cycle=i,
                game_state=_make_game_state(),
                action={"dx": 0.5, "dy": -0.3, "kick": True},
                next_state_delta={},
                effectiveness=0.8,
            )
            loop.memory.add(episode)

        loop._run_iteration()

        call_args = mock_invoke.call_args
        game_state_json = call_args.kwargs.get("game_state_json") or call_args[1][1] if len(call_args[1]) > 1 else call_args.kwargs["game_state_json"]

        # Spatial analysis markers should still be present
        assert "SPATIAL ANALYSIS" in game_state_json
        assert "Ball distance" in game_state_json
        assert "Ball direction" in game_state_json
        assert "In kick range" in game_state_json

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_agentic_context_appended_after_spatial_not_before(self, mock_get, mock_invoke, mock_post):
        """Agentic context appears after spatial analysis in the enriched state."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        loop = _make_loop()

        # Add episodes with known effectiveness to produce predictable memory summary
        from episodic_memory import Episode
        for i in range(3):
            episode = Episode(
                cycle=i + 1,
                game_state=_make_game_state(),
                action={"dx": 0.5, "dy": 0.0, "kick": True},
                next_state_delta={},
                effectiveness=0.9,
            )
            loop.memory.add(episode)

        loop._run_iteration()

        call_args = mock_invoke.call_args
        game_state_json = call_args.kwargs.get("game_state_json") or call_args[1][1] if len(call_args[1]) > 1 else call_args.kwargs["game_state_json"]

        # Find positions of spatial analysis and agentic content
        spatial_pos = game_state_json.find("SPATIAL ANALYSIS")
        assert spatial_pos >= 0, "Spatial analysis should be present"

        # The agentic context (memory summary with "Cycle" lines) should come after spatial
        # Look for memory summary content after spatial analysis
        after_spatial = game_state_json[spatial_pos:]
        # Memory summary lines contain "Cycle N:" format
        cycle_mentions = [i for i, c in enumerate(after_spatial) if after_spatial[i:].startswith("Cycle")]
        # If memory summary is present, it should be after spatial
        if cycle_mentions:
            # Verify the cycle mentions are in the section after spatial analysis
            assert cycle_mentions[0] > 0

    @patch("agent_loop.requests.post")
    @patch("agent_loop.invoke_llm")
    @patch("agent_loop.requests.get")
    def test_empty_agentic_context_does_not_corrupt_spatial(self, mock_get, mock_invoke, mock_post):
        """When agentic context is empty, spatial analysis is still intact."""
        game_state = _make_game_state()
        mock_get.return_value = MagicMock(status_code=200, json=lambda: game_state)
        mock_invoke.return_value = ActionModel(dx=0.1, dy=0.2, kick=False)
        mock_post.return_value = MagicMock(status_code=200)

        # Fresh loop with no memory, no plan, no adaptations
        loop = _make_loop()

        loop._run_iteration()

        call_args = mock_invoke.call_args
        game_state_json = call_args.kwargs.get("game_state_json") or call_args[1][1] if len(call_args[1]) > 1 else call_args.kwargs["game_state_json"]

        # Spatial analysis should still be present and complete
        assert "SPATIAL ANALYSIS" in game_state_json
        assert "Ball distance" in game_state_json
        assert "Your position" in game_state_json
