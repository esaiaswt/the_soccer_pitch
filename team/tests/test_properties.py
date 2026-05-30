"""Property-based tests for the multi-agent team application.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from team.config import load_config


# Feature: multi-agent-team, Property 10: Configuration parameter validation
# **Validates: Requirements 9.2, 3.5**


# --- Strategies for valid parameter ranges ---

valid_coaching_frequency = st.floats(min_value=2.0, max_value=30.0, allow_nan=False, allow_infinity=False)
valid_poll_interval = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
valid_streamlit_port = st.integers(min_value=1024, max_value=65535)

# --- Strategies for invalid parameter ranges ---

invalid_coaching_frequency_low = st.floats(
    min_value=-1e6, max_value=1.999999, allow_nan=False, allow_infinity=False
)
invalid_coaching_frequency_high = st.floats(
    min_value=30.000001, max_value=1e6, allow_nan=False, allow_infinity=False
)
invalid_coaching_frequency = st.one_of(invalid_coaching_frequency_low, invalid_coaching_frequency_high)

invalid_poll_interval_low = st.floats(
    min_value=-1e6, max_value=0.0999999, allow_nan=False, allow_infinity=False
)
invalid_poll_interval_high = st.floats(
    min_value=10.000001, max_value=1e6, allow_nan=False, allow_infinity=False
)
invalid_poll_interval = st.one_of(invalid_poll_interval_low, invalid_poll_interval_high)

invalid_streamlit_port_low = st.integers(min_value=1, max_value=1023)
invalid_streamlit_port_high = st.integers(min_value=65536, max_value=100000)
invalid_streamlit_port = st.one_of(invalid_streamlit_port_low, invalid_streamlit_port_high)


# Keys that load_config reads from the environment
_CONFIG_KEYS = [
    "NVIDIA_API_KEY", "PITCH_HOST", "PITCH_PORT", "COACH_MODEL",
    "PLAYER_MODEL", "COACHING_FREQUENCY", "POLL_INTERVAL",
    "STREAMLIT_PORT", "TEAM_COLOR", "COACH_MEMORY_SIZE",
]


def _clear_config_env():
    """Remove all config-related env vars to prevent leakage between examples."""
    for key in _CONFIG_KEYS:
        os.environ.pop(key, None)


class TestConfigParameterValidation:
    """Property 10: Configuration parameter validation.

    For any set of configuration parameter values, the TeamConfig loader SHALL
    accept values within valid ranges (coaching_frequency: 2-30s, poll_interval:
    0.1-10s, streamlit_port: 1024-65535) and SHALL reject values outside those
    ranges with an error message identifying the invalid parameter.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        coaching_freq=valid_coaching_frequency,
        poll_int=valid_poll_interval,
        port=valid_streamlit_port,
    )
    def test_valid_parameters_accepted(self, tmp_path, coaching_freq, poll_int, port):
        """Valid parameter values within ranges SHALL be accepted without SystemExit."""
        _clear_config_env()

        env_content = (
            f"NVIDIA_API_KEY=nvapi-test-key\n"
            f"COACHING_FREQUENCY={coaching_freq}\n"
            f"POLL_INTERVAL={poll_int}\n"
            f"STREAMLIT_PORT={port}\n"
        )
        env_path = tmp_path / ".env"
        env_path.write_text(env_content)

        try:
            config = load_config(env_path)

            assert config.coaching_frequency == coaching_freq
            assert config.poll_interval == poll_int
            assert config.streamlit_port == port
        finally:
            _clear_config_env()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(coaching_freq=invalid_coaching_frequency)
    def test_invalid_coaching_frequency_rejected(self, tmp_path, coaching_freq):
        """Coaching frequency outside 2-30s SHALL be rejected with error identifying the parameter."""
        _clear_config_env()

        env_content = (
            f"NVIDIA_API_KEY=nvapi-test-key\n"
            f"COACHING_FREQUENCY={coaching_freq}\n"
        )
        env_path = tmp_path / ".env"
        env_path.write_text(env_content)

        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(env_path)

            # Error message SHALL identify the invalid parameter
            assert "COACHING_FREQUENCY" in str(exc_info.value)
        finally:
            _clear_config_env()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(poll_int=invalid_poll_interval)
    def test_invalid_poll_interval_rejected(self, tmp_path, poll_int):
        """Poll interval outside 0.1-10s SHALL be rejected with error identifying the parameter."""
        _clear_config_env()

        env_content = (
            f"NVIDIA_API_KEY=nvapi-test-key\n"
            f"POLL_INTERVAL={poll_int}\n"
        )
        env_path = tmp_path / ".env"
        env_path.write_text(env_content)

        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(env_path)

            # Error message SHALL identify the invalid parameter
            assert "POLL_INTERVAL" in str(exc_info.value)
        finally:
            _clear_config_env()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(port=invalid_streamlit_port)
    def test_invalid_streamlit_port_rejected(self, tmp_path, port):
        """Streamlit port outside 1024-65535 SHALL be rejected with error identifying the parameter."""
        _clear_config_env()

        env_content = (
            f"NVIDIA_API_KEY=nvapi-test-key\n"
            f"STREAMLIT_PORT={port}\n"
        )
        env_path = tmp_path / ".env"
        env_path.write_text(env_content)

        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(env_path)

            # Error message SHALL identify the invalid parameter
            assert "STREAMLIT_PORT" in str(exc_info.value)
        finally:
            _clear_config_env()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        coaching_freq=valid_coaching_frequency,
        poll_int=valid_poll_interval,
    )
    def test_valid_params_without_streamlit_port_accepted(self, tmp_path, coaching_freq, poll_int):
        """Valid coaching_frequency and poll_interval without streamlit_port SHALL be accepted."""
        _clear_config_env()

        env_content = (
            f"NVIDIA_API_KEY=nvapi-test-key\n"
            f"COACHING_FREQUENCY={coaching_freq}\n"
            f"POLL_INTERVAL={poll_int}\n"
        )
        env_path = tmp_path / ".env"
        env_path.write_text(env_content)

        try:
            config = load_config(env_path)

            assert config.coaching_frequency == coaching_freq
            assert config.poll_interval == poll_int
            assert config.streamlit_port is None
        finally:
            _clear_config_env()


# Feature: multi-agent-team, Property 1: State snapshot propagation
# **Validates: Requirements 1.2**

from team.shared_state import SharedState


# --- Strategies for generating arbitrary game state snapshots ---

# Strategy for JSON-compatible primitive values
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=50),
)

# Recursive strategy for arbitrary JSON-like dict snapshots
json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=20,
)

# Strategy for game state snapshots (arbitrary dicts with string keys)
game_state_snapshots = st.dictionaries(
    st.text(min_size=1, max_size=30),
    json_values,
    min_size=1,
    max_size=10,
)


class TestStateSnapshotPropagation:
    """Property 1: State snapshot propagation.

    For any valid game state snapshot returned by the Pitch server, after the
    State Poller processes it, the SharedState container SHALL return that exact
    snapshot (unchanged) to any agent that reads it.
    """

    @settings(max_examples=100)
    @given(snapshot=game_state_snapshots)
    def test_snapshot_stored_and_retrieved_unchanged(self, snapshot):
        """After set_snapshot(s), get_snapshot() SHALL return exactly s."""
        shared_state = SharedState()
        shared_state.set_snapshot(snapshot)
        retrieved = shared_state.get_snapshot()
        assert retrieved == snapshot

    @settings(max_examples=100)
    @given(snapshot=game_state_snapshots)
    def test_snapshot_not_corrupted_during_storage(self, snapshot):
        """The snapshot SHALL not be modified or corrupted during storage and retrieval."""
        import copy

        original = copy.deepcopy(snapshot)
        shared_state = SharedState()
        shared_state.set_snapshot(snapshot)
        retrieved = shared_state.get_snapshot()

        # Verify the retrieved snapshot matches the original (deep equality)
        assert retrieved == original

    @settings(max_examples=100)
    @given(snapshots=st.lists(game_state_snapshots, min_size=2, max_size=5))
    def test_latest_snapshot_always_returned(self, snapshots):
        """After multiple set_snapshot calls, get_snapshot() SHALL return the last one set."""
        shared_state = SharedState()
        for s in snapshots:
            shared_state.set_snapshot(s)
        retrieved = shared_state.get_snapshot()
        assert retrieved == snapshots[-1]

    @settings(max_examples=100)
    @given(snapshot=game_state_snapshots)
    def test_multiple_readers_get_same_snapshot(self, snapshot):
        """Any agent that reads the SharedState SHALL get the same exact snapshot."""
        shared_state = SharedState()
        shared_state.set_snapshot(snapshot)

        # Simulate multiple agents reading the same snapshot
        reader_results = [shared_state.get_snapshot() for _ in range(5)]
        for result in reader_results:
            assert result == snapshot


# Feature: multi-agent-team, Property 5: Instruction delivery integrity
# **Validates: Requirements 3.4**

import time

from team.instruction_store import CoachInstruction, InstructionStore


# --- Strategies for instruction delivery integrity ---

# All four valid player positions
player_positions = st.sampled_from(["Goalkeeper", "Defender", "Midfielder", "Striker"])

# Arbitrary strings including very long ones (>500 chars) to verify no truncation
arbitrary_instruction_content = st.text(
    min_size=0,
    max_size=5000,
    alphabet=st.characters(codec="utf-8", categories=("L", "M", "N", "P", "S", "Z")),
)

# Strategy that biases toward long strings exceeding the 500-char target
long_instruction_content = st.text(
    min_size=501,
    max_size=5000,
    alphabet=st.characters(codec="utf-8", categories=("L", "M", "N", "P", "S", "Z")),
)


class TestInstructionDeliveryIntegrity:
    """Property 5: Instruction delivery integrity.

    For any Coach Instruction string of arbitrary length (including strings
    exceeding 500 characters), the InstructionStore SHALL store and return
    the instruction content without truncation or modification.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        position=player_positions,
        content=arbitrary_instruction_content,
    )
    def test_instruction_stored_and_retrieved_unchanged(self, position, content):
        """Any instruction content SHALL be stored and retrieved without modification."""
        store = InstructionStore()
        timestamp = time.time()

        instruction = CoachInstruction(
            content=content,
            timestamp=timestamp,
            target_position=position,
        )
        store.set_instruction(position, instruction)

        retrieved = store.get_instruction(position)

        assert retrieved is not None
        assert retrieved.content == content
        assert retrieved.timestamp == timestamp
        assert retrieved.target_position == position

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        position=player_positions,
        content=long_instruction_content,
    )
    def test_long_instruction_not_truncated(self, position, content):
        """Instructions exceeding 500 characters SHALL NOT be truncated."""
        store = InstructionStore()
        timestamp = time.time()

        instruction = CoachInstruction(
            content=content,
            timestamp=timestamp,
            target_position=position,
        )
        store.set_instruction(position, instruction)

        retrieved = store.get_instruction(position)

        assert retrieved is not None
        assert len(retrieved.content) == len(content)
        assert retrieved.content == content

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        content=arbitrary_instruction_content,
    )
    def test_instruction_delivered_to_all_positions(self, content):
        """Instructions SHALL be stored and retrieved correctly for all 4 positions."""
        store = InstructionStore()
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
        timestamp = time.time()

        for pos in positions:
            instruction = CoachInstruction(
                content=content,
                timestamp=timestamp,
                target_position=pos,
            )
            store.set_instruction(pos, instruction)

        for pos in positions:
            retrieved = store.get_instruction(pos)
            assert retrieved is not None
            assert retrieved.content == content
            assert retrieved.target_position == pos

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        position=player_positions,
        content1=arbitrary_instruction_content,
        content2=arbitrary_instruction_content,
    )
    def test_instruction_overwrite_preserves_latest(self, position, content1, content2):
        """When a new instruction overwrites an old one, the latest content SHALL be returned unchanged."""
        store = InstructionStore()
        ts1 = time.time()
        ts2 = ts1 + 1.0

        instruction1 = CoachInstruction(content=content1, timestamp=ts1, target_position=position)
        instruction2 = CoachInstruction(content=content2, timestamp=ts2, target_position=position)

        store.set_instruction(position, instruction1)
        store.set_instruction(position, instruction2)

        retrieved = store.get_instruction(position)

        assert retrieved is not None
        assert retrieved.content == content2
        assert retrieved.timestamp == ts2


# Feature: multi-agent-team, Property 12: Structured log completeness
# **Validates: Requirements 7.1, 7.2, 7.5**

import logging
import re
from unittest.mock import patch

from team.logging_config import (
    log_coach_instruction,
    log_token_usage,
    log_agent_error,
    TeamLogFormatter,
)

# ISO 8601 pattern with microsecond precision: YYYY-MM-DDTHH:MM:SS.ffffff
_ISO8601_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}")

# Strategies for log test inputs
_player_positions = st.sampled_from(["Goalkeeper", "Defender", "Midfielder", "Striker"])
_agent_identities = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=("L", "N", "P"),
    whitelist_characters="_- ",
))
_instruction_content = st.text(min_size=1, max_size=500, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z"),
    whitelist_characters=" .,!?;:-",
))
_error_types = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters="_",
))
_match_states = st.sampled_from(["Playing", "Waiting", "Finished", "Paused"])
_attempted_actions = st.text(min_size=1, max_size=100, alphabet=st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters="_- ",
))
_token_counts = st.integers(min_value=0, max_value=100000)


def _create_test_logger() -> tuple[logging.Logger, logging.handlers.MemoryHandler | list]:
    """Create a logger with a list handler to capture log output as formatted strings."""
    logger = logging.getLogger(f"test_log_completeness_{id(object())}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Use a custom handler that stores formatted records
    captured: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(TeamLogFormatter())
    logger.addHandler(handler)

    return logger, captured


class TestStructuredLogCompleteness:
    """Property 12: Structured log completeness.

    For any coach instruction event, the log entry SHALL contain an ISO 8601
    timestamp, target player identity, and instruction content.

    For any LLM invocation with token usage metadata, the log entry SHALL
    contain prompt tokens, completion tokens, and total tokens with agent identity.

    For any agent error event, the log entry SHALL contain agent identity,
    error type, match state, and attempted action.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        target_player=_player_positions,
        content=_instruction_content,
    )
    def test_coach_instruction_log_completeness(self, target_player, content):
        """Coach instruction log entries SHALL contain ISO 8601 timestamp,
        target player identity, and instruction content.

        **Validates: Requirements 7.1**
        """
        logger, captured = _create_test_logger()

        with patch("team.logging_config.get_logger", return_value=logger):
            log_coach_instruction(target_player=target_player, content=content)

        assert len(captured) == 1, "Expected exactly one log entry"
        log_entry = captured[0]

        # SHALL contain ISO 8601 timestamp
        assert _ISO8601_PATTERN.search(log_entry), (
            f"Log entry missing ISO 8601 timestamp: {log_entry!r}"
        )

        # SHALL contain target player identity
        assert target_player in log_entry, (
            f"Log entry missing target player '{target_player}': {log_entry!r}"
        )

        # SHALL contain instruction content
        assert content in log_entry, (
            f"Log entry missing instruction content: {log_entry!r}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        agent_identity=_agent_identities,
        prompt_tokens=_token_counts,
        completion_tokens=_token_counts,
        total_tokens=_token_counts,
    )
    def test_token_usage_log_completeness(
        self, agent_identity, prompt_tokens, completion_tokens, total_tokens
    ):
        """Token usage log entries SHALL contain prompt tokens, completion tokens,
        total tokens, and agent identity.

        **Validates: Requirements 7.2**
        """
        logger, captured = _create_test_logger()

        with patch("team.logging_config.get_logger", return_value=logger):
            log_token_usage(
                agent_identity=agent_identity,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        assert len(captured) == 1, "Expected exactly one log entry"
        log_entry = captured[0]

        # SHALL contain ISO 8601 timestamp
        assert _ISO8601_PATTERN.search(log_entry), (
            f"Log entry missing ISO 8601 timestamp: {log_entry!r}"
        )

        # SHALL contain agent identity
        assert agent_identity in log_entry, (
            f"Log entry missing agent identity '{agent_identity}': {log_entry!r}"
        )

        # SHALL contain prompt tokens
        assert str(prompt_tokens) in log_entry, (
            f"Log entry missing prompt_tokens={prompt_tokens}: {log_entry!r}"
        )

        # SHALL contain completion tokens
        assert str(completion_tokens) in log_entry, (
            f"Log entry missing completion_tokens={completion_tokens}: {log_entry!r}"
        )

        # SHALL contain total tokens
        assert str(total_tokens) in log_entry, (
            f"Log entry missing total_tokens={total_tokens}: {log_entry!r}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        agent_identity=_agent_identities,
        error_type=_error_types,
        match_state=_match_states,
        attempted_action=_attempted_actions,
    )
    def test_agent_error_log_completeness(
        self, agent_identity, error_type, match_state, attempted_action
    ):
        """Agent error log entries SHALL contain agent identity, error type,
        match state, and attempted action.

        **Validates: Requirements 7.5**
        """
        logger, captured = _create_test_logger()

        with patch("team.logging_config.get_logger", return_value=logger):
            log_agent_error(
                agent_identity=agent_identity,
                error_type=error_type,
                match_state=match_state,
                attempted_action=attempted_action,
                error_details="test error details",
            )

        assert len(captured) == 1, "Expected exactly one log entry"
        log_entry = captured[0]

        # SHALL contain ISO 8601 timestamp
        assert _ISO8601_PATTERN.search(log_entry), (
            f"Log entry missing ISO 8601 timestamp: {log_entry!r}"
        )

        # SHALL contain agent identity
        assert agent_identity in log_entry, (
            f"Log entry missing agent identity '{agent_identity}': {log_entry!r}"
        )

        # SHALL contain error type
        assert error_type in log_entry, (
            f"Log entry missing error_type '{error_type}': {log_entry!r}"
        )

        # SHALL contain match state
        assert match_state in log_entry, (
            f"Log entry missing match_state '{match_state}': {log_entry!r}"
        )

        # SHALL contain attempted action
        assert attempted_action in log_entry, (
            f"Log entry missing attempted_action '{attempted_action}': {log_entry!r}"
        )


# Feature: multi-agent-team, Property 2: State Poller error resilience with snapshot preservation
# **Validates: Requirements 1.3**

import threading
from json import JSONDecodeError
from unittest.mock import patch, MagicMock

import requests as _requests_module

from team.config import TeamConfig
from team.state_poller import StatePoller


# --- Strategies for generating poll attempt sequences ---

# A successful poll returns a game state snapshot dict
successful_poll = st.fixed_dictionaries({
    "type": st.just("success"),
    "snapshot": game_state_snapshots,
})

# Error types that the State Poller must handle without crashing
error_types = st.sampled_from(["HTTPError", "Timeout", "ConnectionError", "JSONDecodeError"])

failed_poll = st.fixed_dictionaries({
    "type": st.just("error"),
    "error_type": error_types,
})

# A sequence of poll attempts with at least one success and at least one failure
poll_attempt = st.one_of(successful_poll, failed_poll)

# Sequences that contain at least one success and one failure
poll_sequences_with_mixed = st.lists(
    poll_attempt, min_size=2, max_size=15
).filter(
    lambda seq: any(a["type"] == "success" for a in seq) and any(a["type"] == "error" for a in seq)
)

# Sequences that are all errors (no successes) - snapshot should remain None or last set value
poll_sequences_all_errors = st.lists(failed_poll, min_size=1, max_size=10)


def _make_mock_response(attempt: dict):
    """Create a mock response or raise an exception based on the attempt type."""
    if attempt["type"] == "success":
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = attempt["snapshot"]
        return mock_resp
    else:
        error_type = attempt["error_type"]
        if error_type == "HTTPError":
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            exc = _requests_module.exceptions.HTTPError(response=mock_resp)
            raise exc
        elif error_type == "Timeout":
            raise _requests_module.exceptions.Timeout("Connection timed out")
        elif error_type == "ConnectionError":
            raise _requests_module.exceptions.ConnectionError("Connection refused")
        elif error_type == "JSONDecodeError":
            raise _requests_module.exceptions.JSONDecodeError("Expecting value", "", 0)
        else:
            raise RuntimeError(f"Unknown error type: {error_type}")


def _create_test_config() -> TeamConfig:
    """Create a minimal TeamConfig for testing the State Poller."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test",
        coach_model="m",
        player_model="m",
        coaching_frequency=7.0,
        poll_interval=0.01,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


def _run_poller_with_sequence(poll_sequence, shared_state=None):
    """Helper to run the StatePoller with a mocked sequence of poll attempts.

    Sets stop_event after the sequence is exhausted and patches stop_event.wait
    to return immediately (no real sleeping) for fast test execution.
    """
    config = _create_test_config()
    if shared_state is None:
        shared_state = SharedState()
    stop_event = threading.Event()

    call_index = [0]

    def mock_get(*args, **kwargs):
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(poll_sequence):
            attempt = poll_sequence[idx]
            return _make_mock_response(attempt)
        else:
            # Signal stop after all attempts are exhausted
            stop_event.set()
            raise _requests_module.exceptions.Timeout("done")

    poller = StatePoller(config, shared_state, stop_event)

    # Patch both requests.get and the stop_event.wait to avoid real sleeps
    original_wait = stop_event.wait

    def fast_wait(timeout=None):
        """Return immediately but still check if stop is set."""
        return stop_event.is_set()

    with patch("team.state_poller.requests.get", side_effect=mock_get):
        with patch.object(stop_event, "wait", side_effect=fast_wait):
            poller.run()

    return shared_state


class TestStatePollerErrorResilience:
    """Property 2: State Poller error resilience with snapshot preservation.

    For any sequence of poll attempts where some succeed and some fail (HTTP
    errors or timeouts), the SharedState SHALL always contain the most recent
    successfully received snapshot, and the State Poller SHALL never crash
    regardless of error type.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(poll_sequence=poll_sequences_with_mixed)
    def test_shared_state_contains_last_successful_snapshot(self, poll_sequence):
        """SharedState SHALL always contain the most recent successfully received snapshot
        after processing a mixed sequence of successes and failures.

        **Validates: Requirements 1.3**
        """
        shared_state = _run_poller_with_sequence(poll_sequence)

        # Determine the expected last successful snapshot
        last_success_snapshot = None
        for attempt in poll_sequence:
            if attempt["type"] == "success":
                last_success_snapshot = attempt["snapshot"]

        # SharedState SHALL contain the most recent successful snapshot
        assert shared_state.get_snapshot() == last_success_snapshot

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(poll_sequence=poll_sequences_with_mixed)
    def test_state_poller_never_crashes(self, poll_sequence):
        """State Poller SHALL never crash regardless of error type in the sequence.

        **Validates: Requirements 1.3**
        """
        config = _create_test_config()
        shared_state = SharedState()
        stop_event = threading.Event()

        call_index = [0]

        def mock_get(*args, **kwargs):
            idx = call_index[0]
            call_index[0] += 1
            if idx < len(poll_sequence):
                attempt = poll_sequence[idx]
                return _make_mock_response(attempt)
            else:
                stop_event.set()
                raise _requests_module.exceptions.Timeout("done")

        poller = StatePoller(config, shared_state, stop_event)

        # Run the poller in a thread to detect crashes
        thread_exception = [None]

        def fast_wait(timeout=None):
            return stop_event.is_set()

        def run_poller():
            try:
                with patch("team.state_poller.requests.get", side_effect=mock_get):
                    with patch.object(stop_event, "wait", side_effect=fast_wait):
                        poller.run()
            except Exception as exc:
                thread_exception[0] = exc

        thread = threading.Thread(target=run_poller)
        thread.start()
        thread.join(timeout=10.0)

        # The thread SHALL complete without crashing
        assert not thread.is_alive(), "State Poller thread did not terminate"
        assert thread_exception[0] is None, (
            f"State Poller crashed with: {thread_exception[0]}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(poll_sequence=poll_sequences_all_errors)
    def test_all_errors_preserves_none_snapshot(self, poll_sequence):
        """When all polls fail, SharedState SHALL preserve its initial state (None)
        and the State Poller SHALL not crash.

        **Validates: Requirements 1.3**
        """
        shared_state = _run_poller_with_sequence(poll_sequence)

        # No successful snapshot was received, so SharedState should remain None
        assert shared_state.get_snapshot() is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        initial_snapshot=game_state_snapshots,
        poll_sequence=poll_sequences_all_errors,
    )
    def test_errors_after_success_preserve_last_good_snapshot(self, initial_snapshot, poll_sequence):
        """When errors occur after a successful snapshot, SharedState SHALL preserve
        the most recent successful snapshot.

        **Validates: Requirements 1.3**
        """
        shared_state = SharedState()
        # Pre-set a successful snapshot
        shared_state.set_snapshot(initial_snapshot)

        _run_poller_with_sequence(poll_sequence, shared_state=shared_state)

        # The initial snapshot SHALL be preserved since all polls failed
        assert shared_state.get_snapshot() == initial_snapshot


# Feature: multi-agent-team, Property 3: Coach Memory buffer invariants
# **Validates: Requirements 2.2, 2.3**

from team.coach_agent import CoachMemory, REQUIRED_SNAPSHOT_FIELDS


# --- Strategies for generating valid game state snapshots for CoachMemory ---

# Strategy for a valid snapshot containing all required fields
_ball_position = st.fixed_dictionaries({
    "x": st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
    "y": st.floats(min_value=0.0, max_value=850.0, allow_nan=False, allow_infinity=False),
})

_player_positions = st.dictionaries(
    st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    st.fixed_dictionaries({
        "x": st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        "y": st.floats(min_value=0.0, max_value=850.0, allow_nan=False, allow_infinity=False),
    }),
    min_size=1,
    max_size=8,
)

_score = st.fixed_dictionaries({
    "Red": st.integers(min_value=0, max_value=99),
    "Blue": st.integers(min_value=0, max_value=99),
})

_time_left = st.floats(min_value=0.0, max_value=90.0, allow_nan=False, allow_infinity=False)

_match_state = st.sampled_from(["Playing", "Waiting", "Finished", "Paused"])

valid_snapshot = st.fixed_dictionaries({
    "ball": _ball_position,
    "players": _player_positions,
    "score": _score,
    "time_left": _time_left,
    "match_state": _match_state,
})

# Strategy for buffer max sizes (small to keep tests fast but meaningful)
buffer_max_sizes = st.integers(min_value=1, max_value=100)


class TestCoachMemoryBufferInvariants:
    """Property 3: Coach Memory buffer invariants.

    For any sequence of N snapshots added to a CoachMemory with max size M,
    the buffer SHALL maintain chronological insertion order, never contain more
    than M entries, and always contain the min(N, M) most recently added snapshots.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        max_size=buffer_max_sizes,
        snapshots=st.lists(valid_snapshot, min_size=1, max_size=200),
    )
    def test_buffer_size_never_exceeds_max(self, max_size, snapshots):
        """Buffer SHALL never contain more than M entries.

        **Validates: Requirements 2.2, 2.3**
        """
        memory = CoachMemory(max_size=max_size)

        for snapshot in snapshots:
            memory.add_snapshot(snapshot)

        history = memory.get_history()
        assert len(history) <= max_size

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        max_size=buffer_max_sizes,
        snapshots=st.lists(valid_snapshot, min_size=1, max_size=200),
    )
    def test_buffer_contains_min_n_m_entries(self, max_size, snapshots):
        """Buffer SHALL always contain min(N, M) entries after adding N snapshots.

        **Validates: Requirements 2.2, 2.3**
        """
        memory = CoachMemory(max_size=max_size)

        for snapshot in snapshots:
            memory.add_snapshot(snapshot)

        history = memory.get_history()
        expected_size = min(len(snapshots), max_size)
        assert len(history) == expected_size

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        max_size=buffer_max_sizes,
        snapshots=st.lists(valid_snapshot, min_size=1, max_size=200),
    )
    def test_buffer_maintains_chronological_order(self, max_size, snapshots):
        """Buffer SHALL maintain chronological insertion order.

        **Validates: Requirements 2.2**
        """
        memory = CoachMemory(max_size=max_size)

        for snapshot in snapshots:
            memory.add_snapshot(snapshot)

        history = memory.get_history()

        # Verify chronological order via received_at timestamps
        for i in range(len(history) - 1):
            assert history[i]["received_at"] <= history[i + 1]["received_at"], (
                f"Entry {i} (received_at={history[i]['received_at']}) is not <= "
                f"entry {i+1} (received_at={history[i+1]['received_at']})"
            )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        max_size=buffer_max_sizes,
        snapshots=st.lists(valid_snapshot, min_size=1, max_size=200),
    )
    def test_buffer_contains_most_recent_snapshots(self, max_size, snapshots):
        """Buffer SHALL always contain the min(N, M) most recently added snapshots.

        **Validates: Requirements 2.2, 2.3**
        """
        memory = CoachMemory(max_size=max_size)

        for snapshot in snapshots:
            memory.add_snapshot(snapshot)

        history = memory.get_history()
        expected_count = min(len(snapshots), max_size)

        # The buffer should contain the last `expected_count` snapshots added
        # Each stored entry has the original fields plus "received_at"
        expected_snapshots = snapshots[-expected_count:]

        assert len(history) == expected_count
        for stored, original in zip(history, expected_snapshots):
            # Verify all original fields are preserved in stored entry
            for field in REQUIRED_SNAPSHOT_FIELDS:
                assert stored[field] == original[field], (
                    f"Field '{field}' mismatch: stored={stored[field]}, original={original[field]}"
                )


# Feature: multi-agent-team, Property 4: Invalid snapshot rejection
# **Validates: Requirements 2.5**


# Strategy for generating snapshots with one or more required fields removed
@st.composite
def invalid_snapshot_missing_fields(draw):
    """Generate a snapshot with one or more required fields removed."""
    # Start with a valid snapshot
    base = draw(valid_snapshot)

    # Choose at least one field to remove
    fields_to_remove = draw(
        st.lists(
            st.sampled_from(list(REQUIRED_SNAPSHOT_FIELDS)),
            min_size=1,
            max_size=len(REQUIRED_SNAPSHOT_FIELDS),
            unique=True,
        )
    )

    # Remove the selected fields
    for field in fields_to_remove:
        del base[field]

    return base


class TestInvalidSnapshotRejection:
    """Property 4: Invalid snapshot rejection.

    For any game state snapshot that is missing one or more required fields
    (ball position, player positions, score, time remaining, or match state),
    the CoachMemory SHALL reject it and its size SHALL remain unchanged after
    the attempted addition.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        invalid_snap=invalid_snapshot_missing_fields(),
    )
    def test_invalid_snapshot_rejected_empty_buffer(self, invalid_snap):
        """CoachMemory SHALL reject snapshots missing required fields (empty buffer case).

        **Validates: Requirements 2.5**
        """
        memory = CoachMemory(max_size=50)

        memory.add_snapshot(invalid_snap)

        history = memory.get_history()
        assert len(history) == 0, (
            f"Invalid snapshot was accepted. Missing fields detected. "
            f"Buffer size should be 0 but is {len(history)}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        valid_snaps=st.lists(valid_snapshot, min_size=1, max_size=10),
        invalid_snap=invalid_snapshot_missing_fields(),
    )
    def test_invalid_snapshot_does_not_change_buffer_size(self, valid_snaps, invalid_snap):
        """Buffer size SHALL remain unchanged after attempting to add an invalid snapshot.

        **Validates: Requirements 2.5**
        """
        memory = CoachMemory(max_size=50)

        # Add valid snapshots first
        for snap in valid_snaps:
            memory.add_snapshot(snap)

        size_before = len(memory.get_history())

        # Attempt to add invalid snapshot
        memory.add_snapshot(invalid_snap)

        size_after = len(memory.get_history())
        assert size_after == size_before, (
            f"Buffer size changed from {size_before} to {size_after} "
            f"after adding invalid snapshot"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        valid_snaps=st.lists(valid_snapshot, min_size=1, max_size=10),
        invalid_snap=invalid_snapshot_missing_fields(),
    )
    def test_invalid_snapshot_does_not_corrupt_existing_entries(self, valid_snaps, invalid_snap):
        """Existing buffer entries SHALL not be corrupted by a rejected invalid snapshot.

        **Validates: Requirements 2.5**
        """
        memory = CoachMemory(max_size=50)

        # Add valid snapshots first
        for snap in valid_snaps:
            memory.add_snapshot(snap)

        history_before = memory.get_history()

        # Attempt to add invalid snapshot
        memory.add_snapshot(invalid_snap)

        history_after = memory.get_history()

        # Existing entries should be unchanged
        assert len(history_after) == len(history_before)
        for before, after in zip(history_before, history_after):
            for field in REQUIRED_SNAPSHOT_FIELDS:
                assert before[field] == after[field]


# Feature: multi-agent-team, Property 6: Coach error resilience
# **Validates: Requirements 3.6**

from team.coach_agent import CoachAgent


# --- Strategies for generating random exception types ---

# Exception classes that can occur during LLM invocation
_exception_classes = st.sampled_from([
    RuntimeError,
    TimeoutError,
    ValueError,
    ConnectionError,
    OSError,
    IOError,
    TypeError,
    AttributeError,
    KeyError,
    Exception,
])

# Strategy for exception messages
_exception_messages = st.text(
    min_size=0,
    max_size=200,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)


def _create_coach_config() -> TeamConfig:
    """Create a minimal TeamConfig for testing the Coach Agent."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test",
        coach_model="m",
        player_model="m",
        coaching_frequency=7.0,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


def _create_valid_game_snapshot() -> dict:
    """Create a valid game state snapshot with all required fields."""
    return {
        "ball": {"x": 600.0, "y": 425.0},
        "players": {
            "Red_Goalkeeper": {"x": 100.0, "y": 425.0},
            "Red_Defender": {"x": 250.0, "y": 325.0},
            "Red_Midfielder": {"x": 500.0, "y": 425.0},
            "Red_Striker": {"x": 800.0, "y": 425.0},
        },
        "score": {"Red": 0, "Blue": 0},
        "time_left": 60.0,
        "match_state": "Playing",
    }


class TestCoachErrorResilience:
    """Property 6: Coach error resilience.

    For any LLM invocation failure (timeout, API error, malformed response, or
    exception), the Coach Agent SHALL not crash, SHALL not corrupt its memory or
    instruction store, and SHALL proceed to the next coaching cycle.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        exception_class=_exception_classes,
        exception_message=_exception_messages,
    )
    def test_coaching_cycle_does_not_raise_on_llm_failure(
        self, exception_class, exception_message
    ):
        """CoachAgent._coaching_cycle() SHALL NOT raise when the LLM invocation fails.

        **Validates: Requirements 3.6**
        """
        config = _create_coach_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()

        # Set up a valid snapshot so the coach proceeds to LLM invocation
        shared_state.set_snapshot(_create_valid_game_snapshot())

        # Patch ChatNVIDIA at the module level before constructing the CoachAgent
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = exception_class(exception_message)

        with patch("team.coach_agent.ChatNVIDIA", return_value=mock_llm):
            coach = CoachAgent(config, shared_state, instruction_store, stop_event)
            # _coaching_cycle() SHALL NOT raise
            coach._coaching_cycle()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        exception_class=_exception_classes,
        exception_message=_exception_messages,
    )
    def test_instruction_store_not_corrupted_on_llm_failure(
        self, exception_class, exception_message
    ):
        """InstructionStore SHALL not be corrupted (no partial writes) after LLM failure.

        **Validates: Requirements 3.6**
        """
        config = _create_coach_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()

        # Pre-populate instruction store with known instructions
        pre_existing_instruction = CoachInstruction(
            content="Hold position near the goal",
            timestamp=time.time(),
            target_position="Goalkeeper",
        )
        instruction_store.set_instruction("Goalkeeper", pre_existing_instruction)

        shared_state.set_snapshot(_create_valid_game_snapshot())

        # Capture instruction store state before the failed cycle
        instructions_before = instruction_store.get_all_instructions()

        # Patch ChatNVIDIA at the module level before constructing the CoachAgent
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = exception_class(exception_message)

        with patch("team.coach_agent.ChatNVIDIA", return_value=mock_llm):
            coach = CoachAgent(config, shared_state, instruction_store, stop_event)
            coach._coaching_cycle()

        # Instruction store SHALL not be corrupted - pre-existing instructions preserved
        instructions_after = instruction_store.get_all_instructions()

        # The pre-existing instruction should still be intact
        assert "Goalkeeper" in instructions_after
        assert instructions_after["Goalkeeper"].content == pre_existing_instruction.content
        assert instructions_after["Goalkeeper"].timestamp == pre_existing_instruction.timestamp

        # No new partial writes should have occurred (same keys as before)
        assert set(instructions_after.keys()) == set(instructions_before.keys())

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        exception_class=_exception_classes,
        exception_message=_exception_messages,
    )
    def test_coach_memory_preserves_snapshot_on_llm_failure(
        self, exception_class, exception_message
    ):
        """CoachMemory SHALL still have the snapshot added before the LLM call after failure.

        **Validates: Requirements 3.6**
        """
        config = _create_coach_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()

        snapshot = _create_valid_game_snapshot()
        shared_state.set_snapshot(snapshot)

        # Patch ChatNVIDIA at the module level before constructing the CoachAgent
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = exception_class(exception_message)

        with patch("team.coach_agent.ChatNVIDIA", return_value=mock_llm):
            coach = CoachAgent(config, shared_state, instruction_store, stop_event)
            coach._coaching_cycle()

        # Memory SHALL still contain the snapshot that was added before the LLM call
        history = coach._memory.get_history()
        assert len(history) == 1, (
            f"Expected 1 snapshot in memory after failed cycle, got {len(history)}"
        )

        # Verify the stored snapshot matches the original (all required fields preserved)
        stored = history[0]
        for field in REQUIRED_SNAPSHOT_FIELDS:
            assert stored[field] == snapshot[field], (
                f"Memory corrupted: field '{field}' mismatch after LLM failure"
            )


# Feature: multi-agent-team, Property 8: Coach instruction staleness detection
# **Validates: Requirements 5.1**

from team.player_agent import PlayerAgent
from team.debug_store import DebugStore


# --- Strategies for staleness detection ---

# Coaching frequency values in valid range (2-30 seconds)
_coaching_frequencies = st.floats(min_value=2.0, max_value=30.0, allow_nan=False, allow_infinity=False)

# Instruction ages that are NOT stale (age < 3 * F)
# We generate a fraction strictly less than 1.0 to stay below the threshold
# (boundary age == 3*F is excluded to avoid floating-point precision issues)
_fresh_age_fraction = st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False)

# Instruction ages that ARE stale (age > 3 * F)
# We generate an amount above the threshold
_stale_age_excess = st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False)


def _create_player_agent_config(coaching_frequency: float) -> TeamConfig:
    """Create a TeamConfig with the given coaching frequency for staleness tests."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test",
        coach_model="m",
        player_model="m",
        coaching_frequency=coaching_frequency,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


class TestCoachInstructionStalenessDetection:
    """Property 8: Coach instruction staleness detection.

    For any coaching frequency F and instruction timestamp T, a Player Agent
    SHALL classify the instruction as stale if and only if the current time
    minus T exceeds 3 × F, and SHALL exclude stale instructions from its LLM
    context.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        coaching_freq=_coaching_frequencies,
        age_fraction=_fresh_age_fraction,
    )
    def test_fresh_instruction_is_returned(self, coaching_freq, age_fraction):
        """When instruction age <= 3 * F, the instruction SHALL be returned (not stale).

        **Validates: Requirements 5.1**
        """
        config = _create_player_agent_config(coaching_freq)
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        # Use a fixed "now" to avoid timing drift between timestamp creation and check
        fixed_now = 1000000.0
        threshold = 3 * coaching_freq
        age = age_fraction * threshold
        instruction_timestamp = fixed_now - age

        instruction = CoachInstruction(
            content="Press high and close down passing lanes",
            timestamp=instruction_timestamp,
            target_position="Striker",
        )
        instruction_store.set_instruction("Striker", instruction)

        with patch("team.player_agent.ChatNVIDIA"):
            player = PlayerAgent(
                config=config,
                position="Striker",
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )

        # Patch time.time() in the player_agent module to return our fixed_now
        with patch("team.player_agent.time.time", return_value=fixed_now):
            result = player._get_valid_instruction()

        # Instruction age <= 3*F, so it SHALL be returned (not stale)
        assert result is not None
        assert result == instruction.content

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        coaching_freq=_coaching_frequencies,
        stale_excess=_stale_age_excess,
    )
    def test_stale_instruction_is_excluded(self, coaching_freq, stale_excess):
        """When instruction age > 3 * F, the instruction SHALL be excluded (stale).

        **Validates: Requirements 5.1**
        """
        config = _create_player_agent_config(coaching_freq)
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        # Use a fixed "now" to avoid timing drift
        fixed_now = 1000000.0
        threshold = 3 * coaching_freq
        age = threshold + stale_excess
        instruction_timestamp = fixed_now - age

        instruction = CoachInstruction(
            content="Drop back and defend the lead",
            timestamp=instruction_timestamp,
            target_position="Defender",
        )
        instruction_store.set_instruction("Defender", instruction)

        with patch("team.player_agent.ChatNVIDIA"):
            player = PlayerAgent(
                config=config,
                position="Defender",
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )

        # Patch time.time() in the player_agent module to return our fixed_now
        with patch("team.player_agent.time.time", return_value=fixed_now):
            result = player._get_valid_instruction()

        # Instruction age > 3*F, so it SHALL be excluded (None returned)
        assert result is None


# Feature: multi-agent-team, Property 7: Coach instruction inclusion in player context
# **Validates: Requirements 4.4**

from langchain_core.messages import HumanMessage, SystemMessage
from team.player_agent import PlayerAgent
from team.debug_store import DebugStore


# --- Strategies for generating arbitrary coach instruction text ---

_coach_instruction_text = st.text(
    min_size=1,
    max_size=2000,
    alphabet=st.characters(codec="utf-8", categories=("L", "M", "N", "P", "S", "Z")),
)

_player_position_for_context = st.sampled_from(["Goalkeeper", "Defender", "Midfielder", "Striker"])


def _create_player_config() -> TeamConfig:
    """Create a minimal TeamConfig for testing the Player Agent."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test",
        coach_model="m",
        player_model="m",
        coaching_frequency=7.0,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


def _create_valid_game_state_snapshot() -> dict:
    """Create a valid game state snapshot for player agent tests."""
    return {
        "ball": {"x": 600.0, "y": 425.0},
        "players": {
            "Red_Goalkeeper": {"x": 100.0, "y": 425.0},
            "Red_Defender": {"x": 250.0, "y": 325.0},
            "Red_Midfielder": {"x": 500.0, "y": 425.0},
            "Red_Striker": {"x": 800.0, "y": 425.0},
        },
        "score": {"Red": 0, "Blue": 0},
        "time_left": 60.0,
        "match_state": "Playing",
    }


class TestCoachInstructionInclusionInPlayerContext:
    """Property 7: Coach instruction inclusion in player context.

    For any non-stale Coach Instruction present in the InstructionStore for a
    given player position, the Player Agent's LLM message assembly SHALL include
    that instruction's content in the messages sent to the LLM.

    **Validates: Requirements 4.4**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        position=_player_position_for_context,
        instruction_text=_coach_instruction_text,
    )
    def test_instruction_text_included_in_human_message(self, position, instruction_text):
        """When instruction_text is provided, the HumanMessage content SHALL contain
        the instruction text verbatim.

        **Validates: Requirements 4.4**
        """
        config = _create_player_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        snapshot = _create_valid_game_state_snapshot()

        with patch("team.player_agent.ChatNVIDIA") as mock_chat:
            player = PlayerAgent(
                config=config,
                position=position,
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )

            messages = player._build_messages(snapshot, instruction_text)

        # Messages should be [SystemMessage, HumanMessage]
        assert len(messages) == 2
        human_msg = messages[1]
        assert isinstance(human_msg, HumanMessage)

        # The instruction text SHALL appear in the HumanMessage content
        assert instruction_text in human_msg.content, (
            f"Instruction text not found in HumanMessage content.\n"
            f"Instruction: {instruction_text!r}\n"
            f"HumanMessage content: {human_msg.content!r}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        position=_player_position_for_context,
    )
    def test_no_coach_advisory_when_instruction_is_none(self, position):
        """When instruction_text is None, 'Coach Advisory' SHALL NOT appear
        in the HumanMessage content.

        **Validates: Requirements 4.4**
        """
        config = _create_player_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        snapshot = _create_valid_game_state_snapshot()

        with patch("team.player_agent.ChatNVIDIA") as mock_chat:
            player = PlayerAgent(
                config=config,
                position=position,
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )

            messages = player._build_messages(snapshot, None)

        # Messages should be [SystemMessage, HumanMessage]
        assert len(messages) == 2
        human_msg = messages[1]
        assert isinstance(human_msg, HumanMessage)

        # "Coach Advisory" SHALL NOT appear when instruction_text is None
        assert "Coach Advisory" not in human_msg.content, (
            f"'Coach Advisory' found in HumanMessage when instruction_text is None.\n"
            f"HumanMessage content: {human_msg.content!r}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        position=_player_position_for_context,
        instruction_text=_coach_instruction_text,
    )
    def test_coach_advisory_label_present_with_instruction(self, position, instruction_text):
        """When instruction_text is provided, the HumanMessage SHALL contain
        the 'Coach Advisory' label indicating advisory context.

        **Validates: Requirements 4.4**
        """
        config = _create_player_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        snapshot = _create_valid_game_state_snapshot()

        with patch("team.player_agent.ChatNVIDIA") as mock_chat:
            player = PlayerAgent(
                config=config,
                position=position,
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )

            messages = player._build_messages(snapshot, instruction_text)

        human_msg = messages[1]
        assert isinstance(human_msg, HumanMessage)

        # "Coach Advisory" label SHALL be present when instruction is included
        assert "Coach Advisory" in human_msg.content, (
            f"'Coach Advisory' label not found in HumanMessage when instruction is provided.\n"
            f"HumanMessage content: {human_msg.content!r}"
        )


# Feature: multi-agent-team, Property 9: Player Brake Action on LLM failure
# **Validates: Requirements 5.2, 5.3**

from team.player_agent import PlayerAgent
from team.debug_store import DebugStore


# --- Strategies for generating random exception types for Player LLM failures ---

_player_exception_classes = st.sampled_from([
    RuntimeError,
    TimeoutError,
    ValueError,
    ConnectionError,
    OSError,
    Exception,
])

_player_exception_messages = st.text(
    min_size=0,
    max_size=200,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)

# Player positions for the test
_player_test_positions = st.sampled_from(["Goalkeeper", "Defender", "Midfielder", "Striker"])


def _create_player_config() -> TeamConfig:
    """Create a minimal TeamConfig for testing the Player Agent."""
    return TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test",
        coach_model="m",
        player_model="m",
        coaching_frequency=7.0,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TeamBot",
    )


def _create_player_game_snapshot() -> dict:
    """Create a valid game state snapshot for Player Agent testing."""
    return {
        "ball": {"x": 600.0, "y": 425.0},
        "players": {
            "Red_Goalkeeper": {"x": 100.0, "y": 425.0},
            "Red_Defender": {"x": 250.0, "y": 325.0},
            "Red_Midfielder": {"x": 500.0, "y": 425.0},
            "Red_Striker": {"x": 800.0, "y": 425.0},
        },
        "score": {"Red": 0, "Blue": 0},
        "time_left": 60.0,
        "match_state": "Playing",
    }


class TestPlayerBrakeActionOnLLMFailure:
    """Property 9: Player Brake Action on LLM failure.

    For any LLM invocation that raises an exception or times out, the Player
    Agent SHALL submit exactly a Brake Action (dx=0, dy=0, kick=false) and
    SHALL continue to the next loop iteration without crashing.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        exception_class=_player_exception_classes,
        exception_message=_player_exception_messages,
        position=_player_test_positions,
    )
    def test_loop_iteration_does_not_raise_on_llm_failure(
        self, exception_class, exception_message, position
    ):
        """_loop_iteration() SHALL NOT raise when the LLM invocation fails.

        **Validates: Requirements 5.2, 5.3**
        """
        config = _create_player_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        # Set up a valid snapshot so the player proceeds to LLM invocation
        shared_state.set_snapshot(_create_player_game_snapshot())

        # Mock ChatNVIDIA to raise the exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = exception_class(exception_message)

        # Mock requests.post to capture the action posted
        mock_post = MagicMock()
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        with patch("team.player_agent.ChatNVIDIA", return_value=mock_llm):
            player = PlayerAgent(
                config, position, shared_state, instruction_store, stop_event, debug_store
            )
            with patch("team.player_agent.requests.post", mock_post):
                # _loop_iteration() SHALL NOT raise
                player._loop_iteration()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    @given(
        exception_class=_player_exception_classes,
        exception_message=_player_exception_messages,
        position=_player_test_positions,
    )
    def test_brake_action_posted_on_llm_failure(
        self, exception_class, exception_message, position
    ):
        """The action POSTed to the Pitch server SHALL be a Brake_Action
        (dx=0, dy=0, kick=false) when the LLM invocation fails.

        **Validates: Requirements 5.2, 5.3**
        """
        config = _create_player_config()
        shared_state = SharedState()
        instruction_store = InstructionStore()
        stop_event = threading.Event()
        debug_store = DebugStore()

        # Set up a valid snapshot so the player proceeds to LLM invocation
        shared_state.set_snapshot(_create_player_game_snapshot())

        # Mock ChatNVIDIA to raise the exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = exception_class(exception_message)

        # Mock requests.post to capture the action posted
        mock_post = MagicMock()
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        with patch("team.player_agent.ChatNVIDIA", return_value=mock_llm):
            player = PlayerAgent(
                config, position, shared_state, instruction_store, stop_event, debug_store
            )
            with patch("team.player_agent.requests.post", mock_post):
                player._loop_iteration()

        # Verify that requests.post was called with a Brake_Action payload
        assert mock_post.called, "requests.post was not called after LLM failure"

        # Extract the payload from the call
        call_kwargs = mock_post.call_args
        # requests.post is called as: requests.post(url, json=payload, timeout=5)
        posted_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        # Verify Brake_Action: vector.dx=0, vector.dy=0, kick=False
        assert posted_payload["vector"]["dx"] == 0, (
            f"Expected dx=0 in Brake_Action, got dx={posted_payload['vector']['dx']}"
        )
        assert posted_payload["vector"]["dy"] == 0, (
            f"Expected dy=0 in Brake_Action, got dy={posted_payload['vector']['dy']}"
        )
        assert posted_payload["kick"] is False, (
            f"Expected kick=False in Brake_Action, got kick={posted_payload['kick']}"
        )

# Feature: multi-agent-team, Property 11: Port auto-assignment
# **Validates: Requirements 8.2**

from team.port_scanner import (
    find_available_port,
    PORT_RANGE_START,
    PORT_RANGE_END,
    PortUnavailableError,
)

# --- Strategies for generating occupied port subsets ---

# Generate arbitrary subsets of ports in the range 8501-8510
_occupied_port_subsets = st.sets(
    st.integers(min_value=PORT_RANGE_START, max_value=PORT_RANGE_END)
)

# Generate subsets that leave at least one port free
_occupied_port_subsets_partial = _occupied_port_subsets.filter(
    lambda s: len(s) < (PORT_RANGE_END - PORT_RANGE_START + 1)
)

# Generate the full set (all ports occupied)
_all_ports_occupied = st.just(
    set(range(PORT_RANGE_START, PORT_RANGE_END + 1))
)


class TestPortAutoAssignment:
    """Property 11: Port auto-assignment.

    For any subset of ports in the range 8501-8510 that are already occupied,
    the port scanner SHALL select the lowest-numbered port in that range that
    is not occupied, or report failure if all ports are occupied.

    **Validates: Requirements 8.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(occupied_ports=_occupied_port_subsets_partial)
    def test_returns_lowest_free_port(self, occupied_ports):
        """When at least one port is free, find_available_port(None) SHALL return
        the lowest-numbered free port in the range.

        **Validates: Requirements 8.2**
        """
        def mock_is_port_available(port: int) -> bool:
            return port not in occupied_ports

        with patch("team.port_scanner.is_port_available", side_effect=mock_is_port_available):
            result = find_available_port(None)

        # Determine the expected lowest free port
        all_ports = set(range(PORT_RANGE_START, PORT_RANGE_END + 1))
        free_ports = sorted(all_ports - occupied_ports)
        expected = free_ports[0]

        assert result == expected, (
            f"Expected lowest free port {expected}, got {result}. "
            f"Occupied: {sorted(occupied_ports)}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(occupied_ports=_all_ports_occupied)
    def test_raises_when_all_ports_occupied(self, occupied_ports):
        """When all ports in 8501-8510 are occupied, find_available_port(None)
        SHALL raise PortUnavailableError.

        **Validates: Requirements 8.2**
        """
        def mock_is_port_available(port: int) -> bool:
            return port not in occupied_ports

        with patch("team.port_scanner.is_port_available", side_effect=mock_is_port_available):
            with pytest.raises(PortUnavailableError):
                find_available_port(None)
