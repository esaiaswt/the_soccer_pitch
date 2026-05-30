# Feature: agent-control-panel, Property 8: Action payload construction preserves ActionModel values
"""Property-based tests for action payload construction.

Property 8: For any valid ActionModel instance and any valid team/position
combination, the constructed POST payload SHALL contain the correct values.

Validates: Requirements 5.1, 5.2
"""

import threading
from unittest.mock import MagicMock, patch, call

from hypothesis import given, settings, strategies as st

from agent_loop import AgentLoop
from config import ActionModel, TEAMS, POSITIONS


# Strategy for valid ActionModel instances
action_model_strategy = st.builds(
    ActionModel,
    dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    kick=st.booleans(),
)

# Strategy for valid team/position combinations
team_strategy = st.sampled_from(TEAMS)
position_strategy = st.sampled_from(POSITIONS)


@settings(max_examples=100)
@given(
    action=action_model_strategy,
    team=team_strategy,
    position=position_strategy,
)
def test_action_payload_preserves_action_model_values(action, team, position):
    """Property 8: Action payload construction preserves ActionModel values.

    **Validates: Requirements 5.1, 5.2**

    For any valid ActionModel instance and any valid team/position combination,
    the constructed POST payload SHALL contain the team as a string, position
    as a string, a vector object with dx and dy matching the ActionModel's
    values exactly, and kick matching the ActionModel's kick value.
    """
    mock_post = MagicMock()
    mock_post.return_value = MagicMock(status_code=200)

    with patch("agent_loop.requests.post", mock_post):
        loop = AgentLoop(
            server_ip="localhost",
            team=team,
            position=position,
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )
        loop._act(action)

    # Verify the POST was called
    mock_post.assert_called_once()

    # Extract the payload from the call
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # Verify payload structure and values
    assert payload["team"] == team
    assert payload["position"] == position
    assert payload["vector"]["dx"] == action.dx
    assert payload["vector"]["dy"] == action.dy
    assert payload["kick"] == action.kick
