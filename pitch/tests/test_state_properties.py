"""Property-based tests for the state module."""

from hypothesis import given, settings
import hypothesis.strategies as st

from pitch.state import StateManager, MatchState, Player, Ball, _get_default_position


# Feature: the-pitch, Property 5: Movement vector calculation
class TestMovementVectorCalculation:
    """Property 5: Movement vector calculation.

    For any input vector (dx, dy) with arbitrary float values, the applied
    movement shall equal (clamp(dx, -1, 1) * 20, clamp(dy, -1, 1) * 20).

    **Validates: Requirements 8.2, 8.3**
    """

    @given(
        dx=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_movement_equals_clamped_vector_times_max_speed(self, dx: float, dy: float):
        """Applied movement equals (clamp(dx,-1,1)*20, clamp(dy,-1,1)*20)."""
        sm = StateManager()
        acquired = sm.acquire()
        assert acquired, "Failed to acquire state lock"
        try:
            # Spawn a player at center of pitch so boundaries don't interfere
            player_name = "Red_TestPlayer"
            start_x = 600.0
            start_y = 400.0
            sm.state.players[player_name] = Player(
                name=player_name,
                team="Red",
                x=start_x,
                y=start_y,
            )

            # Apply action with the generated dx/dy
            sm.apply_action("Red", "TestPlayer", {"dx": dx, "dy": dy}, False)

            player = sm.state.players[player_name]

            # Calculate expected movement
            clamped_dx = max(-1.0, min(1.0, dx))
            clamped_dy = max(-1.0, min(1.0, dy))
            expected_x = start_x + clamped_dx * 20
            expected_y = start_y + clamped_dy * 20

            assert player.x == expected_x, (
                f"Expected x={expected_x}, got x={player.x} for dx={dx}"
            )
            assert player.y == expected_y, (
                f"Expected y={expected_y}, got y={player.y} for dy={dy}"
            )
        finally:
            sm.release()


# --- Strategies for Property 11 ---

_teams = st.sampled_from(["Red", "Blue"])
_positions = st.sampled_from(
    ["Goalkeeper", "Defender", "Midfielder", "Striker", "Winger", "Sweeper"]
)

# Generate a player entry as (team, position, full_name)
_player_entries = st.tuples(_teams, _positions).map(
    lambda tp: (tp[0], tp[1], f"{tp[0]}_{tp[1]}")
)

# Generate a list of 1-6 unique player entries
_player_lists = st.lists(
    _player_entries,
    min_size=1,
    max_size=6,
    unique_by=lambda entry: entry[2],
)

_ball_x = st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False)
_ball_y = st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False)
_ball_vx = st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False)
_ball_vy = st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False)
_player_x = st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False)
_player_y = st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False)


# Feature: the-pitch, Property 11: Post-goal reset invariant
class TestPostGoalResetInvariant:
    """Property 11: Post-goal reset invariant.

    For any game state immediately after a goal is scored, the ball position
    shall be (600, 400) with velocity (0, 0), and all players shall be at
    their team's default starting coordinates.

    **Validates: Requirements 9.5, 9.6**
    """

    @given(
        players=_player_lists,
        bx=_ball_x,
        by=_ball_y,
        bvx=_ball_vx,
        bvy=_ball_vy,
        px_values=st.lists(_player_x, min_size=6, max_size=6),
        py_values=st.lists(_player_y, min_size=6, max_size=6),
    )
    @settings(max_examples=100)
    def test_post_goal_reset_invariant(self, players, bx, by, bvx, bvy, px_values, py_values):
        """After reset_after_goal(), ball resets to (600, 400) with zero velocity
        and all players return to their team's default positions."""
        manager = StateManager()
        state = manager._state

        # Set ball to arbitrary position/velocity
        state.ball.x = bx
        state.ball.y = by
        state.ball.vx = bvx
        state.ball.vy = bvy

        # Add players at arbitrary positions
        for i, (team, position, full_name) in enumerate(players):
            state.players[full_name] = Player(
                name=full_name,
                team=team,
                x=px_values[i % len(px_values)],
                y=py_values[i % len(py_values)],
            )

        # Call reset_after_goal
        manager.reset_after_goal()

        # Assert ball resets to center with zero velocity
        assert state.ball.x == 600.0
        assert state.ball.y == 425.0
        assert state.ball.vx == 0.0
        assert state.ball.vy == 0.0

        # Assert all players are at their team's default starting coordinates
        for name, player in state.players.items():
            position_part = name.split("_", 1)[1]
            expected = _get_default_position(player.team, position_part)
            assert player.x == expected["x"], (
                f"Player {name} x={player.x} != expected {expected['x']}"
            )
            assert player.y == expected["y"], (
                f"Player {name} y={player.y} != expected {expected['y']}"
            )


# Feature: the-pitch, Property 9: Score preservation across match reset
class TestScoreResetAcrossMatchReset:
    """Property 9: Score reset across match reset.

    For any score state (Red: n, Blue: m) when the timer expires and match
    transitions from Playing to Waiting, the score values shall reset to 0
    for the new match. The previous match score is saved in previous_match.

    **Validates: Requirements 5.6**
    """

    @given(
        red_score=st.integers(min_value=0, max_value=100),
        blue_score=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_scores_reset_after_reset_match(self, red_score: int, blue_score: int):
        """Scores reset to 0 after Playing -> Waiting transition via reset_match()."""
        # Set up a StateManager with match_state=PLAYING and the generated scores
        sm = StateManager()
        sm.acquire()
        sm.state.match_state = MatchState.PLAYING
        sm.state.score = {"Red": red_score, "Blue": blue_score}

        # Call reset_match (simulates timer expiry triggering Playing -> Waiting)
        sm.reset_match()

        # Assert that the score resets to 0 for new match
        assert sm.state.score["Red"] == 0
        assert sm.state.score["Blue"] == 0
        sm.release()
