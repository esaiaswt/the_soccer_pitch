"""Property-based tests for the team/ ReflectionEngine module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.reflection import ReflectionEngine, ReflectionResult
from team.episodic_memory import Episode, EpisodicMemory


# --- Strategies for generating valid game states ---

game_state_strategy = st.fixed_dictionaries({
    "ball_distance": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    "goal_distance": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    "has_possession": st.booleans(),
})

action_strategy = st.fixed_dictionaries({
    "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "kick": st.booleans(),
})

expected_outcome_strategy = st.fixed_dictionaries({
    "ball_distance": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    "goal_distance": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    "has_possession": st.booleans(),
})


# Feature: full-agentic-upgrade, Property 12: Effectiveness score range invariant
# **Validates: Requirements 5.3, 5.4**


class TestEffectivenessScoreRangeInvariant:
    """Property 12: Effectiveness score range invariant.

    For any pair of consecutive game states and any action taken, the
    Effectiveness_Score computed by the Reflection_Engine SHALL be a value
    in the closed interval [0.0, 1.0].
    """

    @settings(max_examples=100)
    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    def test_effectiveness_score_within_zero_one(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """The Effectiveness_Score SHALL be in the closed interval [0.0, 1.0].

        **Validates: Requirements 5.3, 5.4**
        """
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None, "Result should not be None when previous_state is provided"
        assert 0.0 <= result.effectiveness_score <= 1.0, (
            f"Effectiveness score {result.effectiveness_score} is outside [0.0, 1.0]"
        )

    @settings(max_examples=100)
    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
    )
    def test_returns_none_when_no_previous_state(
        self, action, expected_outcome, actual_state
    ):
        """The Reflection_Engine SHALL return None when previous_state is None (first cycle).

        **Validates: Requirements 5.3, 5.4**
        """
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, None)

        assert result is None, "Result should be None when previous_state is None"

    @settings(max_examples=100)
    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    def test_score_clamped_with_extreme_values(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """Even with extreme state values, the score SHALL remain in [0.0, 1.0].

        **Validates: Requirements 5.3, 5.4**
        """
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None
        assert isinstance(result.effectiveness_score, float)
        assert 0.0 <= result.effectiveness_score <= 1.0


# Feature: full-agentic-upgrade, Property 13: Effectiveness score stored in episode
# **Validates: Requirements 5.5**


class TestEffectivenessScoreStoredInEpisode:
    """Property 13: Effectiveness score stored in episode.

    For any completed reflection evaluation, the computed Effectiveness_Score
    SHALL be stored in the corresponding Episode's effectiveness field, and
    retrieving that episode SHALL return the same score value.
    """

    @settings(max_examples=100)
    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    def test_score_stored_in_episode_effectiveness_field(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """The computed Effectiveness_Score SHALL be stored in the Episode's effectiveness field.

        **Validates: Requirements 5.5**
        """
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None

        # Create an episode and store the effectiveness score
        episode = Episode(
            cycle=1,
            game_state=previous_state,
            action=action,
            next_state_delta={},
            effectiveness=result.effectiveness_score,
        )

        assert episode.effectiveness == result.effectiveness_score, (
            f"Episode effectiveness {episode.effectiveness} does not match "
            f"computed score {result.effectiveness_score}"
        )

    @settings(max_examples=100)
    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    def test_score_retrievable_from_episodic_memory(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """Retrieving the episode from EpisodicMemory SHALL return the same score value.

        **Validates: Requirements 5.5**
        """
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None

        # Store episode in memory with the effectiveness score
        memory = EpisodicMemory(max_capacity=100)
        episode = Episode(
            cycle=1,
            game_state=previous_state,
            action=action,
            next_state_delta={},
            effectiveness=result.effectiveness_score,
        )
        memory.add(episode)

        # Retrieve and verify the score is preserved
        retrieved = memory.get_recent(1)
        assert len(retrieved) == 1
        assert retrieved[0].effectiveness == result.effectiveness_score, (
            f"Retrieved effectiveness {retrieved[0].effectiveness} does not match "
            f"stored score {result.effectiveness_score}"
        )


# Feature: full-agentic-upgrade, Property 14: Abandonment signal on consecutive low scores
# **Validates: Requirements 5.7**


class TestAbandonmentSignalOnConsecutiveLowScores:
    """Property 14: Abandonment signal on consecutive low scores.

    For any sequence of Effectiveness_Scores where the last 5 consecutive scores
    are all below 0.3, the Reflection_Engine SHALL signal plan abandonment. For
    any sequence where at least one of the last 5 scores is >= 0.3, abandonment
    SHALL NOT be signaled.
    """

    @settings(max_examples=100)
    @given(
        low_scores=st.lists(
            st.floats(min_value=0.0, max_value=0.2999, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=5,
        ),
    )
    def test_abandonment_signaled_when_5_consecutive_low_scores(self, low_scores):
        """When the last 5 consecutive scores are all below 0.3, abandonment SHALL be signaled.

        **Validates: Requirements 5.7**
        """
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        # Directly inject scores into the engine's recent scores
        for score in low_scores:
            engine._recent_scores.append(score)

        # Verify abandonment is signaled
        assert engine._check_abandonment() is True, (
            f"Abandonment should be signaled for scores {low_scores} "
            f"(all below 0.3)"
        )

    @settings(max_examples=100)
    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=5,
        ),
    )
    def test_no_abandonment_when_at_least_one_score_above_threshold(self, scores):
        """When at least one of the last 5 scores is >= 0.3, abandonment SHALL NOT be signaled.

        **Validates: Requirements 5.7**
        """
        # Ensure at least one score is >= 0.3
        assume(any(s >= 0.3 for s in scores))

        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        for score in scores:
            engine._recent_scores.append(score)

        assert engine._check_abandonment() is False, (
            f"Abandonment should NOT be signaled for scores {scores} "
            f"(at least one is >= 0.3)"
        )

    @settings(max_examples=100)
    @given(
        num_scores=st.integers(min_value=0, max_value=4),
        scores=st.lists(
            st.floats(min_value=0.0, max_value=0.2999, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=4,
        ),
    )
    def test_no_abandonment_when_fewer_than_window_scores(self, num_scores, scores):
        """When fewer than 5 scores have been recorded, abandonment SHALL NOT be signaled.

        **Validates: Requirements 5.7**
        """
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        # Only add up to 4 scores (fewer than the window)
        for score in scores[:num_scores]:
            engine._recent_scores.append(score)

        assert engine._check_abandonment() is False, (
            f"Abandonment should NOT be signaled with only {len(engine._recent_scores)} scores "
            f"(need at least 5)"
        )

    @settings(max_examples=100)
    @given(
        prefix_scores=st.lists(
            st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=10,
        ),
        low_scores=st.lists(
            st.floats(min_value=0.0, max_value=0.2999, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=5,
        ),
    )
    def test_abandonment_after_prefix_of_good_scores(self, prefix_scores, low_scores):
        """After a prefix of good scores, 5 consecutive low scores SHALL still trigger abandonment.

        **Validates: Requirements 5.7**
        """
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        # Add prefix of good scores
        for score in prefix_scores:
            engine._recent_scores.append(score)

        # Add 5 consecutive low scores
        for score in low_scores:
            engine._recent_scores.append(score)

        assert engine._check_abandonment() is True, (
            f"Abandonment should be signaled after 5 consecutive low scores "
            f"regardless of prefix. Recent scores: {list(engine._recent_scores)}"
        )
