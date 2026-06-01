"""Property-based tests for the StrategyTracker module (Properties 15-18).

# Feature: full-agentic-upgrade, Property 15: Strategy tracker pattern recording
# Feature: full-agentic-upgrade, Property 16: Directional analysis produces adaptation above confidence threshold
# Feature: full-agentic-upgrade, Property 17: Active adaptations count limit
# Feature: full-agentic-upgrade, Property 18: Match reset preserves adaptations but clears raw entries

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from strategy_tracker import PatternEntry, AdaptationRecord, StrategyTracker


# --- Strategies ---

position_strategy = st.fixed_dictionaries(
    {
        "x": st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        "y": st.floats(min_value=-350, max_value=350, allow_nan=False, allow_infinity=False),
    }
)

opponent_positions_strategy = st.lists(
    position_strategy, min_size=1, max_size=11
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


def directional_bias_entry_strategy(direction: str, ball_pos: dict):
    """Generate a PatternEntry where opponents are biased in a specific direction relative to ball.

    Directions use the classification logic:
    - left: avg opponent x - ball x < -10 (and abs(dx) > abs(dy))
    - right: avg opponent x - ball x > 10 (and abs(dx) > abs(dy))
    - center: abs(dx) <= 10 and abs(dy) <= 10
    - forward: avg opponent y - ball y > 10 (and abs(dy) >= abs(dx))
    - back: avg opponent y - ball y < -10 (and abs(dy) >= abs(dx))
    """
    bx = ball_pos["x"]
    by = ball_pos["y"]

    if direction == "left":
        # opponent avg x must be < ball_x - 10, and abs(dx) > abs(dy)
        opp_x = st.floats(min_value=bx - 100, max_value=bx - 15, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=by - 5, max_value=by + 5, allow_nan=False, allow_infinity=False)
    elif direction == "right":
        # opponent avg x must be > ball_x + 10, and abs(dx) > abs(dy)
        opp_x = st.floats(min_value=bx + 15, max_value=bx + 100, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=by - 5, max_value=by + 5, allow_nan=False, allow_infinity=False)
    elif direction == "forward":
        # opponent avg y must be > ball_y + 10, and abs(dy) >= abs(dx)
        opp_x = st.floats(min_value=bx - 5, max_value=bx + 5, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=by + 15, max_value=by + 100, allow_nan=False, allow_infinity=False)
    elif direction == "back":
        # opponent avg y must be < ball_y - 10, and abs(dy) >= abs(dx)
        opp_x = st.floats(min_value=bx - 5, max_value=bx + 5, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=by - 100, max_value=by - 15, allow_nan=False, allow_infinity=False)
    else:  # center
        opp_x = st.floats(min_value=bx - 5, max_value=bx + 5, allow_nan=False, allow_infinity=False)
        opp_y = st.floats(min_value=by - 5, max_value=by + 5, allow_nan=False, allow_infinity=False)

    # Use a single opponent so the average IS the opponent position
    opp_pos = st.builds(lambda x, y: [{"x": x, "y": y}], opp_x, opp_y)

    return st.builds(
        PatternEntry,
        opponent_positions=opp_pos,
        ball_position=st.just(ball_pos),
        effectiveness=effectiveness_strategy,
    )


# Feature: full-agentic-upgrade, Property 15: Strategy tracker pattern recording
class TestStrategyTrackerPatternRecording:
    """Property 15: Strategy tracker pattern recording.

    For any valid action context (opponent positions, ball position) and
    effectiveness score, recording it in the Strategy_Tracker SHALL increase
    the pattern entry count by 1, and the recorded entry SHALL be retrievable
    with its original values.

    **Validates: Requirements 6.1**
    """

    @given(entry=pattern_entry_strategy)
    @settings(max_examples=100)
    def test_recording_increases_entry_count_by_one(self, entry):
        """Recording a pattern entry increases the internal count by exactly 1."""
        tracker = StrategyTracker()
        count_before = len(tracker._entries)
        tracker.record(entry)
        count_after = len(tracker._entries)
        assert count_after == count_before + 1

    @given(entries=st.lists(pattern_entry_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_recorded_entries_retrievable_with_original_values(self, entries):
        """All recorded entries are retrievable with their original values."""
        tracker = StrategyTracker()

        for entry in entries:
            tracker.record(entry)

        assert len(tracker._entries) == len(entries)

        for i, entry in enumerate(entries):
            stored = tracker._entries[i]
            assert stored.opponent_positions == entry.opponent_positions
            assert stored.ball_position == entry.ball_position
            assert stored.effectiveness == entry.effectiveness

    @given(
        existing_entries=st.lists(pattern_entry_strategy, min_size=0, max_size=10),
        new_entry=pattern_entry_strategy,
    )
    @settings(max_examples=100)
    def test_recording_appends_to_existing_entries(self, existing_entries, new_entry):
        """Recording appends to existing entries without modifying them."""
        tracker = StrategyTracker()

        for entry in existing_entries:
            tracker.record(entry)

        tracker.record(new_entry)

        # The last entry should be the new one
        assert tracker._entries[-1].opponent_positions == new_entry.opponent_positions
        assert tracker._entries[-1].ball_position == new_entry.ball_position
        assert tracker._entries[-1].effectiveness == new_entry.effectiveness


# Feature: full-agentic-upgrade, Property 16: Directional analysis produces adaptation above confidence threshold
class TestDirectionalAnalysisProducesAdaptation:
    """Property 16: Directional analysis produces adaptation above confidence threshold.

    For any Strategy_Tracker with at least 10 pattern entries where opponent
    movements show a directional bias exceeding 70% in one direction, the
    analysis SHALL produce an AdaptationRecord with confidence > 0.7
    describing that tendency.

    **Validates: Requirements 6.2, 6.3**
    """

    @given(
        direction=st.sampled_from(["left", "right", "center", "forward", "back"]),
        num_biased=st.integers(min_value=8, max_value=15),
        num_total=st.integers(min_value=10, max_value=15),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_directional_bias_produces_adaptation(self, direction, num_biased, num_total, data):
        """When >70% of entries show directional bias, an AdaptationRecord is produced."""
        # Ensure biased entries exceed 70% of total
        assume(num_biased <= num_total)
        assume(num_biased / num_total > 0.7)

        ball_pos = {"x": 0.0, "y": 0.0}
        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add biased entries
        for _ in range(num_biased):
            entry = data.draw(directional_bias_entry_strategy(direction, ball_pos))
            tracker.record(entry)

        # Add remaining entries with center bias (neutral)
        num_other = num_total - num_biased
        for _ in range(num_other):
            entry = data.draw(directional_bias_entry_strategy("center", ball_pos))
            tracker.record(entry)

        adaptations = tracker.analyze()

        # Should produce at least one adaptation
        assert len(adaptations) >= 1, (
            f"Expected adaptation for direction '{direction}' with "
            f"{num_biased}/{num_total} entries but got none"
        )

        # At least one adaptation should have confidence > 0.7
        high_confidence = [a for a in adaptations if a.confidence > 0.7]
        assert len(high_confidence) >= 1, (
            f"Expected adaptation with confidence > 0.7 but got: "
            f"{[(a.observed_pattern, a.confidence) for a in adaptations]}"
        )

        # The direction map should match
        direction_patterns = {
            "left": "opponent_favors_left_flank",
            "right": "opponent_favors_right_flank",
            "center": "opponent_favors_center",
            "forward": "opponent_pushes_forward",
            "back": "opponent_sits_back",
        }
        expected_pattern = direction_patterns[direction]
        patterns_found = [a.observed_pattern for a in high_confidence]
        assert expected_pattern in patterns_found, (
            f"Expected pattern '{expected_pattern}' but found: {patterns_found}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_insufficient_entries_produces_no_adaptation(self, data):
        """With fewer than 10 entries, no adaptation is produced."""
        num_entries = data.draw(st.integers(min_value=0, max_value=9))
        tracker = StrategyTracker(min_entries_for_analysis=10)

        for _ in range(num_entries):
            entry = data.draw(pattern_entry_strategy)
            tracker.record(entry)

        adaptations = tracker.analyze()
        assert adaptations == []


# Feature: full-agentic-upgrade, Property 17: Active adaptations count limit
class TestActiveAdaptationsCountLimit:
    """Property 17: Active adaptations count limit.

    For any Strategy_Tracker regardless of how many AdaptationRecords have
    been generated, get_active_adaptations() SHALL return at most 2 records.

    **Validates: Requirements 6.4**
    """

    @given(
        num_adaptations=st.integers(min_value=0, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_active_adaptations_at_most_two(self, num_adaptations, data):
        """get_active_adaptations() returns at most 2 records."""
        tracker = StrategyTracker()

        # Directly inject adaptations to test the limit
        for i in range(num_adaptations):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            tracker._adaptations.append(
                AdaptationRecord(
                    observed_pattern=f"pattern_{i}",
                    counter_strategy=f"counter_{i}",
                    confidence=confidence,
                )
            )

        active = tracker.get_active_adaptations()
        assert len(active) <= 2

    @given(
        direction=st.sampled_from(["left", "right", "forward", "back"]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_active_adaptations_limit_after_analysis(self, direction, data):
        """After analysis produces adaptations, get_active_adaptations() still returns at most 2."""
        ball_pos = {"x": 0.0, "y": 0.0}
        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add enough biased entries to trigger adaptation
        for _ in range(12):
            entry = data.draw(directional_bias_entry_strategy(direction, ball_pos))
            tracker.record(entry)

        tracker.analyze()
        active = tracker.get_active_adaptations()
        assert len(active) <= 2


# Feature: full-agentic-upgrade, Property 18: Match reset preserves adaptations but clears raw entries
class TestMatchResetPreservesAdaptationsClearsEntries:
    """Property 18: Match reset preserves adaptations but clears raw entries.

    For any Strategy_Tracker that has accumulated AdaptationRecords and raw
    pattern entries, calling reset_for_new_match() SHALL retain all existing
    AdaptationRecords while clearing all raw pattern entries to zero.

    **Validates: Requirements 6.6**
    """

    @given(
        direction=st.sampled_from(["left", "right", "forward", "back"]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_reset_clears_raw_entries(self, direction, data):
        """reset_for_new_match() clears all raw pattern entries."""
        ball_pos = {"x": 0.0, "y": 0.0}
        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add entries and trigger analysis
        for _ in range(12):
            entry = data.draw(directional_bias_entry_strategy(direction, ball_pos))
            tracker.record(entry)

        tracker.analyze()
        assert len(tracker._entries) > 0

        tracker.reset_for_new_match()
        assert len(tracker._entries) == 0

    @given(
        direction=st.sampled_from(["left", "right", "forward", "back"]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_reset_preserves_adaptation_records(self, direction, data):
        """reset_for_new_match() retains all existing AdaptationRecords."""
        ball_pos = {"x": 0.0, "y": 0.0}
        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add entries and trigger analysis
        for _ in range(12):
            entry = data.draw(directional_bias_entry_strategy(direction, ball_pos))
            tracker.record(entry)

        adaptations_before = tracker.analyze()
        assert len(adaptations_before) > 0

        # Capture adaptations before reset
        adaptations_snapshot = list(tracker._adaptations)

        tracker.reset_for_new_match()

        # Adaptations should be preserved
        assert len(tracker._adaptations) == len(adaptations_snapshot)
        for i, adaptation in enumerate(tracker._adaptations):
            assert adaptation.observed_pattern == adaptations_snapshot[i].observed_pattern
            assert adaptation.counter_strategy == adaptations_snapshot[i].counter_strategy
            assert adaptation.confidence == adaptations_snapshot[i].confidence

    @given(
        direction=st.sampled_from(["left", "right", "forward", "back"]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_reset_adaptations_still_retrievable(self, direction, data):
        """After reset, adaptations are still retrievable via get_active_adaptations()."""
        ball_pos = {"x": 0.0, "y": 0.0}
        tracker = StrategyTracker(min_entries_for_analysis=10)

        # Add entries and trigger analysis
        for _ in range(12):
            entry = data.draw(directional_bias_entry_strategy(direction, ball_pos))
            tracker.record(entry)

        tracker.analyze()
        active_before = tracker.get_active_adaptations()
        assert len(active_before) > 0

        tracker.reset_for_new_match()

        active_after = tracker.get_active_adaptations()
        assert len(active_after) == len(active_before)
        assert active_after[0].observed_pattern == active_before[0].observed_pattern
