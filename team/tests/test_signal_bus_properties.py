"""Property-based tests for the team/ SignalBus module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

import time

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.signal_bus import Signal, SignalBus


# --- Strategies for generating valid inputs ---

position_strategy = st.sampled_from([
    "Goalkeeper", "Defender_L", "Defender_R", "Midfielder_L",
    "Midfielder_R", "Striker", "Center_Back", "Winger_L",
    "Winger_R", "Attacking_Mid", "Defensive_Mid",
])

signal_type_strategy = st.sampled_from([
    "requesting_pass", "making_run", "covering_zone", "ready_to_pass", "supporting",
])

valid_payload_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=50,
)

invalid_payload_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=51,
    max_size=100,
)

timestamp_strategy = st.floats(
    min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False
)

valid_signal_strategy = st.builds(
    Signal,
    sender_position=position_strategy,
    signal_type=signal_type_strategy,
    payload=valid_payload_strategy,
    timestamp=timestamp_strategy,
)

invalid_signal_strategy = st.builds(
    Signal,
    sender_position=position_strategy,
    signal_type=signal_type_strategy,
    payload=invalid_payload_strategy,
    timestamp=timestamp_strategy,
)


# Feature: full-agentic-upgrade, Property 19: Signal Bus publish/read/replace invariant
# **Validates: Requirements 7.1, 7.2, 7.3, 8.5**


class TestSignalBusPublishReadReplaceInvariant:
    """Property 19: Signal Bus publish/read/replace invariant.

    For any sequence of signals published to the Signal_Bus, reading all signals
    SHALL return only the most recent signal from each sender position, and each
    signal SHALL be visible to all positions other than the sender (when using
    exclude_position). Signals with payloads exceeding 50 characters SHALL be rejected.
    """

    @settings(max_examples=100)
    @given(signals=st.lists(valid_signal_strategy, min_size=1, max_size=20))
    def test_read_returns_most_recent_signal_per_sender(self, signals):
        """Reading all signals SHALL return only the most recent signal from each
        sender position.

        **Validates: Requirements 7.3**
        """
        bus = SignalBus()

        # Track the most recent signal per sender position
        expected_latest: dict[str, Signal] = {}
        for signal in signals:
            bus.publish(signal)
            expected_latest[signal.sender_position] = signal

        # Read all signals (no exclusion)
        result = bus.read_all()

        # Should have exactly one signal per unique sender position
        assert len(result) == len(expected_latest), (
            f"Expected {len(expected_latest)} signals (one per sender), got {len(result)}"
        )

        # Each result should match the most recent signal for that sender
        result_by_sender = {s.sender_position: s for s in result}
        for sender, expected_signal in expected_latest.items():
            assert sender in result_by_sender, (
                f"Signal from {sender} not found in read_all() result"
            )
            actual = result_by_sender[sender]
            assert actual.signal_type == expected_signal.signal_type
            assert actual.payload == expected_signal.payload
            assert actual.timestamp == expected_signal.timestamp

    @settings(max_examples=100)
    @given(
        signals=st.lists(valid_signal_strategy, min_size=1, max_size=20),
        reader_position=position_strategy,
    )
    def test_exclude_position_hides_sender_signals(self, signals, reader_position):
        """Each signal SHALL be visible to all positions other than the sender
        (when using exclude_position).

        **Validates: Requirements 7.2**
        """
        bus = SignalBus()

        for signal in signals:
            bus.publish(signal)

        # Read with exclusion
        result = bus.read_all(exclude_position=reader_position)

        # No signal from the excluded position should be present
        for signal in result:
            assert signal.sender_position != reader_position, (
                f"Signal from excluded position {reader_position} found in result"
            )

        # All signals from other positions should be present
        all_signals = bus.read_all()
        expected_visible = [s for s in all_signals if s.sender_position != reader_position]
        assert len(result) == len(expected_visible), (
            f"Expected {len(expected_visible)} visible signals for {reader_position}, "
            f"got {len(result)}"
        )

    @settings(max_examples=100)
    @given(signal=invalid_signal_strategy)
    def test_payload_exceeding_50_chars_rejected(self, signal):
        """Signals with payloads exceeding 50 characters SHALL be rejected.

        **Validates: Requirements 7.1**
        """
        bus = SignalBus()
        bus.publish(signal)

        # Signal should not be stored
        result = bus.read_all()
        assert len(result) == 0, (
            f"Expected 0 signals after publishing invalid payload "
            f"({len(signal.payload)} chars), got {len(result)}"
        )

    @settings(max_examples=100)
    @given(signal=valid_signal_strategy)
    def test_valid_payload_accepted(self, signal):
        """Signals with payloads of at most 50 characters SHALL be accepted.

        **Validates: Requirements 7.1**
        """
        bus = SignalBus()
        bus.publish(signal)

        result = bus.read_all()
        assert len(result) == 1, (
            f"Expected 1 signal after publishing valid payload "
            f"({len(signal.payload)} chars), got {len(result)}"
        )
        assert result[0].sender_position == signal.sender_position
        assert result[0].signal_type == signal.signal_type
        assert result[0].payload == signal.payload

    @settings(max_examples=100)
    @given(
        signals=st.lists(valid_signal_strategy, min_size=2, max_size=15),
        data=st.data(),
    )
    def test_replace_overwrites_previous_signal_from_same_sender(self, signals, data):
        """Publishing a new signal from the same sender SHALL replace the previous one.

        **Validates: Requirements 7.3**
        """
        bus = SignalBus()

        # Publish all signals
        for signal in signals:
            bus.publish(signal)

        # Pick a sender that has published at least one signal
        senders_used = list({s.sender_position for s in signals})
        chosen_sender = data.draw(st.sampled_from(senders_used))

        # Publish a new signal from the chosen sender
        new_signal = Signal(
            sender_position=chosen_sender,
            signal_type="making_run",
            payload="replacement",
            timestamp=time.time(),
        )
        bus.publish(new_signal)

        # Read and verify the replacement
        result = bus.read_all()
        sender_signals = [s for s in result if s.sender_position == chosen_sender]
        assert len(sender_signals) == 1, (
            f"Expected exactly 1 signal from {chosen_sender}, got {len(sender_signals)}"
        )
        assert sender_signals[0].payload == "replacement"
        assert sender_signals[0].signal_type == "making_run"


# Feature: full-agentic-upgrade, Property 20: Dead ball clears all signals
# **Validates: Requirements 7.1, 7.2, 7.3, 8.5**


class TestDeadBallClearsAllSignals:
    """Property 20: Dead ball clears all signals.

    For any Signal_Bus containing any number of active signals, when a dead ball
    state is detected and clear() is invoked, read_all() SHALL return an empty list.
    """

    @settings(max_examples=100)
    @given(signals=st.lists(valid_signal_strategy, min_size=0, max_size=20))
    def test_clear_empties_all_signals(self, signals):
        """When clear() is invoked, read_all() SHALL return an empty list.

        **Validates: Requirements 8.5**
        """
        bus = SignalBus()

        # Publish signals
        for signal in signals:
            bus.publish(signal)

        # Simulate dead ball detection: clear the bus
        bus.clear()

        # read_all() should return empty list
        result = bus.read_all()
        assert result == [], (
            f"Expected empty list after clear(), got {len(result)} signals"
        )

    @settings(max_examples=100)
    @given(signals=st.lists(valid_signal_strategy, min_size=1, max_size=20))
    def test_clear_then_read_with_exclusion_also_empty(self, signals):
        """After clear(), read_all(exclude_position) SHALL also return an empty list.

        **Validates: Requirements 8.5**
        """
        bus = SignalBus()

        for signal in signals:
            bus.publish(signal)

        bus.clear()

        # Even with exclude_position, should be empty
        for signal in signals:
            result = bus.read_all(exclude_position=signal.sender_position)
            assert result == [], (
                f"Expected empty list after clear() with exclude_position="
                f"{signal.sender_position}, got {len(result)} signals"
            )

    @settings(max_examples=100)
    @given(
        signals_before=st.lists(valid_signal_strategy, min_size=1, max_size=10),
        signals_after=st.lists(valid_signal_strategy, min_size=1, max_size=10),
    )
    def test_publish_after_clear_works_normally(self, signals_before, signals_after):
        """After clear(), new signals can be published and read normally.

        **Validates: Requirements 8.5**
        """
        bus = SignalBus()

        # Publish initial signals
        for signal in signals_before:
            bus.publish(signal)

        # Clear (dead ball)
        bus.clear()
        assert bus.read_all() == []

        # Publish new signals after clear
        expected_latest: dict[str, Signal] = {}
        for signal in signals_after:
            bus.publish(signal)
            expected_latest[signal.sender_position] = signal

        # Should only contain the new signals
        result = bus.read_all()
        assert len(result) == len(expected_latest), (
            f"Expected {len(expected_latest)} signals after re-publishing, got {len(result)}"
        )

        # Verify none of the pre-clear signals leaked through
        result_by_sender = {s.sender_position: s for s in result}
        for sender, expected_signal in expected_latest.items():
            assert sender in result_by_sender
            actual = result_by_sender[sender]
            assert actual.signal_type == expected_signal.signal_type
            assert actual.payload == expected_signal.payload
            assert actual.timestamp == expected_signal.timestamp
