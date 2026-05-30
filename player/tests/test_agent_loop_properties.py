# Feature: agent-control-panel, Property 5: Look step errors always produce Brake_Action
# Feature: agent-control-panel, Property 7: LLM invocation failure always produces Brake_Action
# Feature: agent-control-panel, Property 9: Missing configuration prevents loop start
# Feature: agent-control-panel, Property 10: Empty system prompt produces Brake_Action
"""Property-based tests for the agent loop module.

Property 5: For any HTTP error condition, the Look step SHALL return the
Brake_Action and a non-None fallback reason string.
Validates: Requirements 3.4

Property 7: For any exception type raised during LLM invocation, the Think
step SHALL return the Brake_Action and SHALL NOT propagate the exception.
Validates: Requirements 4.5, 7.1, 7.2, 7.3, 7.5

Property 9: For any configuration state where server IP is empty or team is
not selected, attempting to start the AgentLoop SHALL fail with an error.
Validates: Requirements 8.5

Property 10: For any system prompt that is empty or composed entirely of
whitespace, the Think step SHALL return the Brake_Action.
Validates: Requirements 9.5
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
import requests
from hypothesis import given, settings, strategies as st, assume

from agent_loop import AgentLoop, IterationResult
from config import BRAKE_ACTION, ActionModel, TEAMS


# --- Strategies ---

# Strategy for HTTP error status codes (non-200)
http_error_codes = st.integers(min_value=100, max_value=599).filter(lambda x: x != 200)

# Strategy for whitespace-only strings (for empty system prompt)
whitespace_strings = st.from_regex(r"[\s]*", fullmatch=True).filter(lambda s: s.strip() == "")

# Strategy for empty-like server IPs
empty_server_ips = st.one_of(
    st.just(""),
    st.from_regex(r"[\s]+", fullmatch=True),
)

# Strategy for invalid teams (not in TEAMS list)
invalid_teams = st.text(min_size=1, max_size=20).filter(lambda t: t not in TEAMS)

# Strategy for various exception types that could occur during LLM invocation
exception_messages = st.text(min_size=1, max_size=100)


def make_exception_strategy():
    """Generate various exception types that could occur during LLM invocation."""
    return st.sampled_from([
        TimeoutError,
        ValueError,
        RuntimeError,
        ConnectionError,
        OSError,
        TypeError,
        KeyError,
        AttributeError,
    ])


# --- Property 5: Look step errors always produce Brake_Action ---


@settings(max_examples=100)
@given(status_code=http_error_codes)
def test_look_non_200_status_produces_brake_action(status_code):
    """Property 5: Non-200 HTTP status codes produce Brake_Action.

    **Validates: Requirements 3.4**

    For any non-200 HTTP status code, the Look step SHALL return the
    Brake_Action and a non-None fallback reason string.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code

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

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None
    assert len(result.fallback_reason) > 0


@settings(max_examples=100)
@given(exc_type=make_exception_strategy(), msg=exception_messages)
def test_look_connection_errors_produce_brake_action(exc_type, msg):
    """Property 5: Connection errors produce Brake_Action.

    **Validates: Requirements 3.4**

    For any connection error (timeout, refused, DNS failure), the Look step
    SHALL return the Brake_Action and a non-None fallback reason string.
    """
    # Map exception types to requests exceptions for realistic simulation
    if exc_type == TimeoutError:
        side_effect = requests.Timeout(msg)
    else:
        side_effect = requests.RequestException(msg)

    with patch("agent_loop.requests.get", side_effect=side_effect):
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

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None
    assert len(result.fallback_reason) > 0


# --- Property 7: LLM invocation failure always produces Brake_Action ---


@settings(max_examples=100)
@given(exc_type=make_exception_strategy(), msg=exception_messages)
def test_llm_exception_produces_brake_action(exc_type, msg):
    """Property 7: LLM invocation failure always produces Brake_Action.

    **Validates: Requirements 4.5, 7.1, 7.2, 7.3, 7.5**

    For any exception type raised during LLM invocation, the Think step
    SHALL return the Brake_Action and SHALL NOT propagate the exception.
    """
    game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}

    with patch("agent_loop.invoke_llm", side_effect=exc_type(msg)):
        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "valid system prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )
        # This should NOT raise - exception must be caught
        result = loop._think(game_state)

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None
    assert len(result.fallback_reason) > 0


@settings(max_examples=100)
@given(data=st.data())
def test_llm_none_response_produces_brake_action(data):
    """Property 7: LLM returning None produces Brake_Action.

    **Validates: Requirements 4.5, 7.5**

    When the LLM returns None or empty response, the Think step SHALL
    return the Brake_Action.
    """
    game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}

    with patch("agent_loop.invoke_llm", return_value=None):
        loop = AgentLoop(
            server_ip="localhost",
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "valid system prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )
        result = loop._think(game_state)

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None


# --- Property 9: Missing configuration prevents loop start ---


@settings(max_examples=100)
@given(server_ip=empty_server_ips)
def test_empty_server_ip_prevents_loop_start(server_ip):
    """Property 9: Empty server IP prevents loop start.

    **Validates: Requirements 8.5**

    For any configuration state where server IP is empty, attempting to
    start the AgentLoop SHALL fail with an error.
    """
    with pytest.raises(ValueError):
        AgentLoop(
            server_ip=server_ip,
            team="Red",
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )


@settings(max_examples=100)
@given(team=invalid_teams)
def test_invalid_team_prevents_loop_start(team):
    """Property 9: Invalid team selection prevents loop start.

    **Validates: Requirements 8.5**

    For any configuration state where team is not selected from valid options,
    attempting to start the AgentLoop SHALL fail with an error.
    """
    with pytest.raises(ValueError):
        AgentLoop(
            server_ip="localhost",
            team=team,
            position="Striker",
            llm_client=MagicMock(),
            get_system_prompt=lambda: "test prompt",
            get_behavior_override=lambda: "",
            on_iteration=MagicMock(),
            stop_event=threading.Event(),
        )


# --- Property 10: Empty system prompt produces Brake_Action ---


@settings(max_examples=100)
@given(prompt=whitespace_strings)
def test_empty_system_prompt_produces_brake_action(prompt):
    """Property 10: Empty system prompt produces Brake_Action.

    **Validates: Requirements 9.5**

    For any system prompt that is empty or composed entirely of whitespace,
    the Think step SHALL return the Brake_Action and a fallback reason
    indicating that a system prompt is required.
    """
    game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}

    loop = AgentLoop(
        server_ip="localhost",
        team="Red",
        position="Striker",
        llm_client=MagicMock(),
        get_system_prompt=lambda: prompt,
        get_behavior_override=lambda: "",
        on_iteration=MagicMock(),
        stop_event=threading.Event(),
    )
    result = loop._think(game_state)

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None
    assert "system prompt required" in result.fallback_reason


@settings(max_examples=100)
@given(data=st.data())
def test_empty_string_prompt_produces_brake_action(data):
    """Property 10: Empty string system prompt produces Brake_Action.

    **Validates: Requirements 9.5**

    The empty string case specifically must produce Brake_Action.
    """
    game_state = {"ball": {"x": 100, "y": 200}, "match_state": "Playing"}

    loop = AgentLoop(
        server_ip="localhost",
        team="Red",
        position="Striker",
        llm_client=MagicMock(),
        get_system_prompt=lambda: "",
        get_behavior_override=lambda: "",
        on_iteration=MagicMock(),
        stop_event=threading.Event(),
    )
    result = loop._think(game_state)

    assert isinstance(result, IterationResult)
    assert result.action == BRAKE_ACTION
    assert result.fallback_reason is not None
    assert "system prompt required" in result.fallback_reason
