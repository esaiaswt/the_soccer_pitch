"""Property-based tests for the team/ StrategyTracker module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.strategy_tracker import AdaptationRecord, PatternEntry, StrategyTracker


# --- Strategies for generating valid inputs ---

position_strategy = st.fixed_dictionaries({
    "x": st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    "y": st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
})

opponent_positions_strategy = st.lists(
    position_strategy,
    min_size=1,
    max_size=11,
)

ball_position_strategy = position_strategy

effectiveness_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

pattern_entry_strategy = st.builds(
    PatternEntry,
    opponent_positions=opponent_positions_strategy,
    ball_position=ball_position_strategy,
    effectiveness=effectiveness_strategy,
)


def biased_direction_entry_strategy(direction: str):
    """Generate a PatternEntry where opponents are biased in a specific direction relative to ball.

    Directions:
    - left: avg_dx < 0 and abs_dx >= abs_dy and not near center (abs_dx >= 5 or abs_dy >= 5)
    - right: avg_dx > 0 and abs_dx >= abs_dy and not near center
    - center: abs_dx < 5 and abs_dy < 5
    - forward: avg_dy > 0 and abs_dy > abs_dx and not near center
    - back: avg_dy < 0 and abs_dy > abs_dx and not near center
    """
    # Use ball at origin for simplicity, then place opponents in the desired direction
    ball = {"x": 0.0, "y": 0.0}

    if direction == "left":
        # Opponents to the left: negative x, abs_dx >= abs_dy, not near center
        opp_x = st.floats(min_value=-80.0, max_value=-10.0, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    elif direction == "right":
        # Opponents to the right: positive x, abs_dx >= abs_dy, not near center
        opp_x = st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    elif direction == "center":
        # Opponents near center: abs_dx < 5 and abs_dy < 5
        opp_x = st.floats(min_value=-4.9, max_value=4.9, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=-4.9, max_value=4.9, allow_nan=False, allow_infinity=False)
    elif direction == "forward":
        # Opponents forward: positive y, abs_dy > abs_dx, not near center
        opp_x = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False)
    elif direction == "back":
        # Opponents back: negative y, abs_dy > abs_dx, not near center
        opp_x = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=-80.0, max_value=-10.0, allow_nan=False, allow_infinity=False)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    opponent = st.fixed_dictionaries({
        "x": opp_x,
        "y": opp_y,
    })

    return st.builds(
        PatternEntry,
        opponent_positions=st.lists(opponent, min_size=1, max_size=5),
        ball_position=st.just(ball),
        effectiveness=effectiveness_strategy,
    )


# Feature: full-agentic-upgrade, Property 15: Strategy tracker pattern recording
# **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**


class TestStrategyTrackerPatternRecording:
    """Property 15: Strategy tracker pattern recording.

    For any valid action context (opponent positions, ball position) and effectiveness
    score, recording it in the Strategy_Tracker SHALL increase the pattern entry count
    by 1, and the recorded entry SHALL be retrievable with its original values.
    """

    @settings(max_examples=100)
    @given(entry=pattern_entry_strategy)
    def test_recording_increases_entry_count_by_one(self, entry):
        """Recording a pattern entry SHALL increase the pattern entry count by 1.

        **Validates: Requirements 6.1**
        """
        tracker = StrategyTracker()
        count_before = len(tracker._entries)
        tracker.record(entry)
        count_after = len(tracker._entries)

        assert count_after == count_before + 1, (
            f"Expected entry count to increase by 1, got {count_before} -> {count_after}"
        )

    @settings(max_examples=100)
    @given(entry=pattern_entry_strategy)
    def test_recorded_entry_retrievable_with_original_values(self, entry):
        """The recorded entry SHALL be retrievable with its original values.

        **Validates: Requirements 6.1**
        """
        tracker = StrategyTracker()
        tracker.record(entry)

        stored = tracker._entries[-1]
        assert stored.opponent_positions == entry.opponent_positions
        assert stored.ball_position == entry.ball_position
        assert stored.effectiveness == entry.effectiveness

    @settings(max_examples=100)
    @given(entries=st.lists(pattern_entry_strategy, min_size=1, max_size=20))
    def test_multiple_recordings_accumulate(self, entries):
        """Recording multiple entries SHALL accumulate them all.

        **Validates: Requirements 6.1**
        """
        tracker = StrategyTracker()

        for i, entry in enumerate(entries):
            tracker.record(entry)
            assert len(tracker._entries) == i + 1

        # Verify all entries are stored with original values
        for original, stored in zip(entries, tracker._entries):
            assert stored.opponent_positions == original.opponent_positions
            assert stored.ball_position == original.ball_position
            assert stored.effectiveness == original.effectiveness


# Feature: full-agentic-upgrade, Property 16: Directional analysis produces adaptation above confidence threshold
# **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**


class TestDirectionalAnalysisProducesAdaptation:
    """Property 16: Directional analysis produces adaptation above confidence threshold.

    For any Strategy_Tracker with at least 10 pattern entries where opponent movements
    show a directional bias exceeding 70% in one direction, the analysis SHALL produce
    an AdaptationRecord with confidence > 0.7 describing that tendency.
    """

    @settings(max_examples=100)
    @given(
        direction=st.sampled_from(["left", "right", "center", "forward", "back"]),
        num_biased=st.integers(min_value=8, max_value=15),
        data=st.data(),
    )
    def test_directional_bias_produces_adaptation(self, direction, num_biased, data):
        """When 70%+ entries show directional bias, analysis SHALL produce an AdaptationRecord
        with confidence > 0.7.

        **Validates: Requirements 6.2, 6.3**
        """
        # Ensure we have at least 10 entries total with >70% in one direction
        total_entries = num_biased + 2  # 2 non-biased entries to keep it realistic
        assume(num_biased / total_entries > 0.7)
        assume(total_entries >= 10)

        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add biased entries
        for _ in range(num_biased):
            entry = data.draw(biased_direction_entry_strategy(direction))
            tracker.record(entry)

        # Add a couple of non-biased (center) entries if direction is not center
        # Otherwise add entries in a different direction
        other_direction = "left" if direction != "left" else "right"
        for _ in range(2):
            if direction == "center":
                entry = data.draw(biased_direction_entry_strategy(other_direction))
            else:
                entry = data.draw(biased_direction_entry_strategy("center"))
            tracker.record(entry)

        # Run analysis
        new_adaptations = tracker.analyze()

        # Should produce at least one adaptation
        assert len(new_adaptations) >= 1, (
            f"Expected at least 1 adaptation for direction '{direction}' "
            f"with {num_biased}/{total_entries} biased entries, got {len(new_adaptations)}"
        )

        # The adaptation should have confidence > 0.7
        for adaptation in new_adaptations:
            assert adaptation.confidence > 0.7, (
                f"Expected confidence > 0.7, got {adaptation.confidence}"
            )
            assert isinstance(adaptation.observed_pattern, str)
            assert isinstance(adaptation.counter_strategy, str)
            assert len(adaptation.observed_pattern) > 0
            assert len(adaptation.counter_strategy) > 0

    @settings(max_examples=100)
    @given(
        entries=st.lists(pattern_entry_strategy, min_size=1, max_size=9),
    )
    def test_insufficient_entries_produces_no_adaptation(self, entries):
        """With fewer than 10 entries, analysis SHALL produce no adaptations.

        **Validates: Requirements 6.2**
        """
        tracker = StrategyTracker(min_entries_for_analysis=10)

        for entry in entries:
            tracker.record(entry)

        result = tracker.analyze()
        assert result == [], (
            f"Expected empty list with {len(entries)} entries, got {len(result)} adaptations"
        )


# Feature: full-agentic-upgrade, Property 17: Active adaptations count limit
# **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**


class TestActiveAdaptationsCountLimit:
    """Property 17: Active adaptations count limit.

    For any Strategy_Tracker regardless of how many AdaptationRecords have been
    generated, get_active_adaptations() SHALL return at most 2 records.
    """

    @settings(max_examples=100)
    @given(
        num_adaptations=st.integers(min_value=0, max_value=10),
        data=st.data(),
    )
    def test_active_adaptations_at_most_two(self, num_adaptations, data):
        """get_active_adaptations() SHALL return at most 2 records.

        **Validates: Requirements 6.4**
        """
        tracker = StrategyTracker()

        # Directly inject adaptation records to test the limit
        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            record = AdaptationRecord(
                observed_pattern=f"pattern_{i}",
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            )
            tracker._adaptations.append(record)

        active = tracker.get_active_adaptations()
        assert len(active) <= 2, (
            f"Expected at most 2 active adaptations, got {len(active)}"
        )

    @settings(max_examples=100)
    @given(
        num_adaptations=st.integers(min_value=3, max_value=10),
        data=st.data(),
    )
    def test_active_adaptations_returns_highest_confidence(self, num_adaptations, data):
        """get_active_adaptations() SHALL return the most confident records.

        **Validates: Requirements 6.4**
        """
        tracker = StrategyTracker()

        confidences = []
        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            confidences.append(confidence)
            record = AdaptationRecord(
                observed_pattern=f"pattern_{i}",
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            )
            tracker._adaptations.append(record)

        active = tracker.get_active_adaptations()

        # Should be sorted by confidence descending
        assert len(active) == 2
        assert active[0].confidence >= active[1].confidence

        # The top 2 should be the highest confidence values
        sorted_confidences = sorted(confidences, reverse=True)
        assert active[0].confidence == sorted_confidences[0]
        assert active[1].confidence == sorted_confidences[1]


# Feature: full-agentic-upgrade, Property 18: Match reset preserves adaptations but clears raw entries
# **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**


class TestMatchResetPreservesAdaptations:
    """Property 18: Match reset preserves adaptations but clears raw entries.

    For any Strategy_Tracker that has accumulated AdaptationRecords and raw pattern
    entries, calling reset_for_new_match() SHALL retain all existing AdaptationRecords
    while clearing all raw pattern entries to zero.
    """

    @settings(max_examples=100)
    @given(
        entries=st.lists(pattern_entry_strategy, min_size=1, max_size=20),
        num_adaptations=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    def test_reset_clears_raw_entries(self, entries, num_adaptations, data):
        """reset_for_new_match() SHALL clear all raw pattern entries to zero.

        **Validates: Requirements 6.6**
        """
        tracker = StrategyTracker()

        # Add pattern entries
        for entry in entries:
            tracker.record(entry)

        # Add adaptation records
        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            tracker._adaptations.append(AdaptationRecord(
                observed_pattern=f"pattern_{i}",
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            ))

        # Verify entries exist before reset
        assert len(tracker._entries) == len(entries)

        # Reset
        tracker.reset_for_new_match()

        # Raw entries should be cleared
        assert len(tracker._entries) == 0, (
            f"Expected 0 raw entries after reset, got {len(tracker._entries)}"
        )

    @settings(max_examples=100)
    @given(
        entries=st.lists(pattern_entry_strategy, min_size=1, max_size=20),
        num_adaptations=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    def test_reset_retains_adaptations(self, entries, num_adaptations, data):
        """reset_for_new_match() SHALL retain all existing AdaptationRecords.

        **Validates: Requirements 6.6**
        """
        tracker = StrategyTracker()

        # Add pattern entries
        for entry in entries:
            tracker.record(entry)

        # Add adaptation records
        adaptations_before = []
        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            record = AdaptationRecord(
                observed_pattern=f"pattern_{i}",
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            )
            tracker._adaptations.append(record)
            adaptations_before.append(record)

        # Reset
        tracker.reset_for_new_match()

        # Adaptations should be retained
        assert len(tracker._adaptations) == num_adaptations, (
            f"Expected {num_adaptations} adaptations after reset, "
            f"got {len(tracker._adaptations)}"
        )

        # Verify each adaptation is preserved with original values
        for original, stored in zip(adaptations_before, tracker._adaptations):
            assert stored.observed_pattern == original.observed_pattern
            assert stored.counter_strategy == original.counter_strategy
            assert stored.confidence == original.confidence

    @settings(max_examples=100)
    @given(
        entries=st.lists(pattern_entry_strategy, min_size=1, max_size=20),
        num_adaptations=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    def test_reset_allows_new_recordings_after_clear(self, entries, num_adaptations, data):
        """After reset_for_new_match(), new entries can be recorded starting from zero.

        **Validates: Requirements 6.6**
        """
        tracker = StrategyTracker()

        # Add initial entries and adaptations
        for entry in entries:
            tracker.record(entry)

        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            tracker._adaptations.append(AdaptationRecord(
                observed_pattern=f"pattern_{i}",
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            ))

        # Reset
        tracker.reset_for_new_match()
        assert len(tracker._entries) == 0

        # Record new entries after reset
        new_entry = data.draw(pattern_entry_strategy)
        tracker.record(new_entry)

        assert len(tracker._entries) == 1
        assert tracker._entries[0] == new_entry
