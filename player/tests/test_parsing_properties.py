# Feature: agent-control-panel, Property 4: Game state parsing extracts all required fields
"""Property-based tests for game state parsing.

Property 4: For any valid game state JSON object containing match_state,
time_left, score, ball (with x, y), and players (with x, y per player),
the parsing function SHALL extract all fields without data loss.

Validates: Requirements 3.2
"""

import json
from unittest.mock import MagicMock, patch
import threading

from hypothesis import given, settings, strategies as st

from agent_loop import AgentLoop, IterationResult
from config import ActionModel


# Strategy for generating valid game state dictionaries
def game_state_strategy():
    """Generate valid game state JSON objects with all required fields."""
    return st.fixed_dictionaries({
        "match_state": st.sampled_from(["Playing", "Waiting"]),
        "time_left": st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False),
        "score": st.fixed_dictionaries({
            "Red": st.integers(min_value=0, max_value=99),
            "Blue": st.integers(min_value=0, max_value=99),
        }),
        "ball": st.fixed_dictionaries({
            "x": st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
            "y": st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
        }),
        "players": st.dictionaries(
            keys=st.from_regex(r"(Red|Blue)_(Striker|Goalkeeper|Midfielder|Defender)", fullmatch=True),
            values=st.fixed_dictionaries({
                "x": st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
                "y": st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
            }),
            min_size=1,
            max_size=8,
        ),
    })


@settings(max_examples=100)
@given(game_state=game_state_strategy())
def test_game_state_parsing_extracts_all_fields(game_state):
    """Property 4: Game state parsing extracts all required fields.

    **Validates: Requirements 3.2**

    For any valid game state JSON, when the Look step successfully retrieves
    the state, all fields are preserved without data loss in the returned dict.
    """
    # Simulate a successful HTTP response returning the game state
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = game_state

    with patch("agent_loop.requests.get", return_value=mock_response):
        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )
        result = loop._look()

    # The result should be the game state dict (not an IterationResult)
    assert isinstance(result, dict)

    # Verify all required fields are present and preserved
    assert result["match_state"] == game_state["match_state"]
    assert result["time_left"] == game_state["time_left"]
    assert result["score"] == game_state["score"]
    assert result["ball"]["x"] == game_state["ball"]["x"]
    assert result["ball"]["y"] == game_state["ball"]["y"]

    # Verify all player positions are preserved
    assert result["players"] == game_state["players"]
    for player_key, player_pos in game_state["players"].items():
        assert player_key in result["players"]
        assert result["players"][player_key]["x"] == player_pos["x"]
        assert result["players"][player_key]["y"] == player_pos["y"]
