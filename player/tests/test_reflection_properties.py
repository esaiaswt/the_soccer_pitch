"""Property-based tests for the ReflectionEngine module (Properties 12, 13, 14).

# Feature: full-agentic-upgrade, Property 12: Effectiveness score range invariant
# Feature: full-agentic-upgrade, Property 13: Effectiveness score stored in episode
# Feature: full-agentic-upgrade, Property 14: Abandonment signal on consecutive low scores
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from reflection import ReflectionEngine, ReflectionResult
from episodic_memory import Episode, EpisodicMemory


# --- Strategies ---

game_state_strategy = st.fixed_dictionaries(
    {
        "ball_distance": st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        "goal_distance": st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        "has_possession": st.booleans(),
    }
)

action_strategy = st.fixed_dictionaries(
    {
        "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "kick": st.booleans(),
    }
)

expected_outcome_strategy = st.fixed_dictionaries(
    {
        "ball_distance": st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        "goal_distance": st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
    }
)


# Feature: full-agentic-upgrade, Property 12: Effectiveness score range invariant
class TestEffectivenessScoreRangeInvariant:
    """Property 12: Effectiveness score range invariant.

    For any pair of consecutive game states and any action taken, the
    Effectiveness_Score computed by the Reflection_Engine SHALL be a value
    in the closed interval [0.0, 1.0].

    **Validates: Requirements 5.3, 5.4**
    """

    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    @settings(max_examples=100)
    def test_effectiveness_score_always_in_unit_interval(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """Effectiveness score is always between 0.0 and 1.0 inclusive."""
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None
        assert 0.0 <= result.effectiveness_score <= 1.0, (
            f"Score {result.effectiveness_score} is outside [0.0, 1.0]. "
            f"Previous state: {previous_state}, Actual state: {actual_state}"
        )

    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
    )
    @settings(max_examples=100)
    def test_score_clamped_with_extreme_state_changes(
        self, action, expected_outcome, actual_state, previous_state
    ):
        """Score remains in [0.0, 1.0] even with extreme state transitions."""
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None
        assert isinstance(result.effectiveness_score, float)
        assert result.effectiveness_score >= 0.0
        assert result.effectiveness_score <= 1.0

    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
    )
    @settings(max_examples=100)
    def test_returns_none_when_previous_state_is_none(
        self, action, expected_outcome, actual_state
    ):
        """Returns None when previous_state is None (first cycle)."""
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, None)

        assert result is None


# Feature: full-agentic-upgrade, Property 13: Effectiveness score stored in episode
class TestEffectivenessScoreStoredInEpisode:
    """Property 13: Effectiveness score stored in episode.

    For any completed reflection evaluation, the computed Effectiveness_Score
    SHALL be stored in the corresponding Episode's effectiveness field, and
    retrieving that episode SHALL return the same score value.

    **Validates: Requirements 5.5**
    """

    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        actual_state=game_state_strategy,
        previous_state=game_state_strategy,
        cycle=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_effectiveness_score_stored_and_retrievable(
        self, action, expected_outcome, actual_state, previous_state, cycle
    ):
        """Computed effectiveness score can be stored in an episode and retrieved."""
        engine = ReflectionEngine()
        result = engine.evaluate(action, expected_outcome, actual_state, previous_state)

        assert result is not None

        # Store the score in an episode
        episode = Episode(
            cycle=cycle,
            game_state=previous_state,
            action=action,
            next_state_delta={
                "ball_dx": actual_state["ball_distance"] - previous_state["ball_distance"],
                "ball_dy": 0.0,
                "possession_changed": actual_state["has_possession"] != previous_state["has_possession"],
            },
            effectiveness=result.effectiveness_score,
        )

        # Store in memory and retrieve
        memory = EpisodicMemory(max_capacity=100)
        memory.add(episode)

        retrieved = memory.get_all()
        assert len(retrieved) == 1
        assert retrieved[0].effectiveness == result.effectiveness_score, (
            f"Stored effectiveness {retrieved[0].effectiveness} does not match "
            f"computed score {result.effectiveness_score}"
        )

    @given(
        num_evaluations=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_multiple_scores_stored_in_corresponding_episodes(
        self, num_evaluations, data
    ):
        """Multiple effectiveness scores are stored in their corresponding episodes."""
        engine = ReflectionEngine()
        memory = EpisodicMemory(max_capacity=100)
        scores = []

        for i in range(num_evaluations):
            action = data.draw(action_strategy)
            expected_outcome = data.draw(expected_outcome_strategy)
            actual_state = data.draw(game_state_strategy)
            previous_state = data.draw(game_state_strategy)

            result = engine.evaluate(action, expected_outcome, actual_state, previous_state)
            assert result is not None

            scores.append(result.effectiveness_score)

            episode = Episode(
                cycle=i,
                game_state=previous_state,
                action=action,
                next_state_delta={},
                effectiveness=result.effectiveness_score,
            )
            memory.add(episode)

        # Verify all stored scores match
        retrieved = memory.get_all()
        assert len(retrieved) == num_evaluations
        for i, ep in enumerate(retrieved):
            assert ep.effectiveness == scores[i], (
                f"Episode {i}: stored {ep.effectiveness} != computed {scores[i]}"
            )


# Feature: full-agentic-upgrade, Property 14: Abandonment signal on consecutive low scores
class TestAbandonmentSignalOnConsecutiveLowScores:
    """Property 14: Abandonment signal on consecutive low scores.

    For any sequence of Effectiveness_Scores where the last 5 consecutive
    scores are all below 0.3, the Reflection_Engine SHALL signal plan
    abandonment. For any sequence where at least one of the last 5 scores
    is >= 0.3, abandonment SHALL NOT be signaled.

    **Validates: Requirements 5.7**
    """

    @given(
        prefix_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=10,
        ),
        low_scores=st.lists(
            st.floats(min_value=0.0, max_value=0.2999, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_abandonment_signaled_after_five_consecutive_low_scores(
        self, prefix_scores, low_scores
    ):
        """Abandonment is signaled when last 5 consecutive scores are all below 0.3."""
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        all_scores = prefix_scores + low_scores

        # Feed all scores through the engine by creating appropriate game states
        for score in all_scores:
            # Construct states that produce the desired score
            # We directly manipulate the internal state for precise control
            engine._recent_scores.append(score)

        # Check abandonment based on the last 5 scores
        should_abandon = engine._check_abandonment()
        assert should_abandon is True, (
            f"Expected abandonment with last 5 scores: "
            f"{list(engine._recent_scores)[-5:]}"
        )

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=15,
        ),
    )
    @settings(max_examples=100)
    def test_no_abandonment_when_any_recent_score_above_threshold(self, scores):
        """Abandonment is NOT signaled when at least one of last 5 scores >= 0.3."""
        # Ensure at least one of the last 5 scores is >= 0.3
        assume(any(s >= 0.3 for s in scores[-5:]))

        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        for score in scores:
            engine._recent_scores.append(score)

        should_abandon = engine._check_abandonment()
        assert should_abandon is False, (
            f"Should NOT abandon when last 5 scores include one >= 0.3: "
            f"{list(engine._recent_scores)[-5:]}"
        )

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100)
    def test_no_abandonment_with_fewer_than_window_scores(self, scores):
        """Abandonment is NOT signaled when fewer than 5 scores have been recorded."""
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        for score in scores:
            engine._recent_scores.append(score)

        should_abandon = engine._check_abandonment()
        assert should_abandon is False, (
            f"Should NOT abandon with only {len(scores)} scores "
            f"(need {engine._abandonment_window})"
        )

    @given(
        action=action_strategy,
        expected_outcome=expected_outcome_strategy,
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_abandonment_via_evaluate_with_low_scoring_states(
        self, action, expected_outcome, data
    ):
        """Abandonment signal is correctly set via evaluate() after 5 low-scoring evaluations."""
        engine = ReflectionEngine(abandonment_window=5, abandonment_threshold=0.3)

        # Generate states that produce low scores:
        # ball_distance increases (getting further from ball) and
        # goal_distance increases (ball getting further from goal) and
        # losing possession
        for i in range(5):
            previous_state = {
                "ball_distance": data.draw(
                    st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False)
                ),
                "goal_distance": data.draw(
                    st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False)
                ),
                "has_possession": True,
            }
            # Make actual state worse: further from ball, further from goal, lost possession
            actual_state = {
                "ball_distance": previous_state["ball_distance"] * 3.0,
                "goal_distance": previous_state["goal_distance"] * 3.0,
                "has_possession": False,
            }
            result = engine.evaluate(action, expected_outcome, actual_state, previous_state)
            assert result is not None
            # Verify the score is indeed low (below 0.3)
            assert result.effectiveness_score < 0.3, (
                f"Expected low score but got {result.effectiveness_score}"
            )

        # After 5 consecutive low scores, the last result should signal abandonment
        assert result.should_abandon_plan is True, (
            f"Expected abandonment after 5 low scores. "
            f"Recent scores: {engine.get_recent_scores()}"
        )
