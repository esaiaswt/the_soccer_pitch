"""Property-based tests for physics engine and kick mechanics.

Uses Hypothesis to verify correctness properties of the physics simulation.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from pitch.config import Config
from pitch.physics import PhysicsEngine, FRICTION
from pitch.state import Ball, MatchState, Player, StateManager

_config = Config()


# Feature: the-pitch, Property 4: Kick distance threshold
# **Validates: Requirements 4.5, 4.6**
@settings(max_examples=100)
@given(
    player_x=st.floats(min_value=100.0, max_value=1100.0),
    player_y=st.floats(min_value=100.0, max_value=700.0),
    angle=st.floats(min_value=0.0, max_value=2 * math.pi),
    distance_factor=st.floats(min_value=0.1, max_value=60.0),
)
def test_kick_distance_threshold(
    player_x: float,
    player_y: float,
    angle: float,
    distance_factor: float,
) -> None:
    """For any player and ball positions, kick applies impulse only when distance < 30.

    If distance(player, ball) < 30: ball velocity changes (impulse applied with
    magnitude = KICK_IMPULSE = 20).
    If distance(player, ball) >= 30: ball velocity unchanged (kick ignored).
    """
    # Compute ball position at a given distance and angle from the player
    ball_x = player_x + distance_factor * math.cos(angle)
    ball_y = player_y + distance_factor * math.sin(angle)

    # Set up state manager
    sm = StateManager()
    assert sm.acquire()
    try:
        state = sm.state
        # Set match state to Playing so actions are processed
        from pitch.state import MatchState

        state.match_state = MatchState.PLAYING

        # Place ball at computed position with zero velocity
        state.ball = Ball(x=ball_x, y=ball_y, vx=0.0, vy=0.0)

        # Place player at generated position
        player_name = "Red_Striker"
        state.players[player_name] = Player(
            name=player_name, team="Red", x=player_x, y=player_y
        )

        # Calculate expected distance
        dist = math.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)

        # Apply action with kick=True
        sm.apply_action(
            team="Red",
            position="Striker",
            vector={"dx": 0.0, "dy": 0.0},
            kick=True,
        )

        ball = state.ball

        if dist < _config.POSSESSION_RANGE:
            # Ball velocity should have changed - impulse applied
            speed = math.sqrt(ball.vx**2 + ball.vy**2)
            assert speed > 0, (
                f"Expected kick impulse when distance={dist:.2f} < {_config.POSSESSION_RANGE}, "
                f"but ball velocity is zero"
            )
            # Verify impulse magnitude equals KICK_IMPULSE (20)
            assert math.isclose(speed, _config.KICK_IMPULSE, rel_tol=1e-6), (
                f"Expected impulse magnitude {_config.KICK_IMPULSE}, got {speed:.4f}"
            )
        else:
            # Ball velocity should remain unchanged (zero)
            assert ball.vx == 0.0 and ball.vy == 0.0, (
                f"Expected no kick when distance={dist:.2f} >= {_config.POSSESSION_RANGE}, "
                f"but ball velocity is ({ball.vx}, {ball.vy})"
            )
    finally:
        sm.release()


from pitch.state import GameState, MatchState


# Feature: the-pitch, Property 10: Goal detection and scoring
class TestGoalDetectionAndScoring:
    """Property 10: Goal detection and scoring.

    For any ball position that enters a goal zone (left: x 0-30, y 300-500 or
    right: x 1170-1200, y 300-500) for the first time (ball was previously outside),
    the opposing team's score shall increment by exactly 1. Subsequent ticks with
    the ball still inside the same zone shall not increment the score again.

    **Validates: Requirements 9.2, 9.3**
    """

    @given(
        ball_x=st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_goal_detection_scores_correctly(self, ball_x: float, ball_y: float):
        """Ball in goal zone increments correct team score exactly once."""
        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)

        # Set up state with ball at generated position, goal_scored_flag=False
        state = state_manager.state
        state.ball.x = ball_x
        state.ball.y = ball_y
        state.goal_scored_flag = False
        state.score = {"Red": 0, "Blue": 0}

        # Call check_goal
        engine.check_goal(state)

        in_left_goal = ball_x <= 30.0 and 300.0 <= ball_y <= 500.0
        in_right_goal = ball_x >= 1170.0 and 300.0 <= ball_y <= 500.0

        if in_left_goal:
            # Blue scores when ball enters left goal
            assert state.score["Blue"] == 1, (
                f"Blue should have scored 1 but got {state.score['Blue']} "
                f"for ball at ({ball_x}, {ball_y})"
            )
            assert state.score["Red"] == 0
            assert state.goal_scored_flag is True
        elif in_right_goal:
            # Red scores when ball enters right goal
            assert state.score["Red"] == 1, (
                f"Red should have scored 1 but got {state.score['Red']} "
                f"for ball at ({ball_x}, {ball_y})"
            )
            assert state.score["Blue"] == 0
            assert state.goal_scored_flag is True
        else:
            # No goal: scores unchanged, flag stays False
            assert state.score["Red"] == 0
            assert state.score["Blue"] == 0
            assert state.goal_scored_flag is False

    @given(
        ball_x=st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_no_double_scoring_when_flag_set(self, ball_x: float, ball_y: float):
        """Calling check_goal with goal_scored_flag=True does NOT increment score."""
        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)

        # Set up state with goal_scored_flag=True (simulating already scored)
        state = state_manager.state
        state.ball.x = ball_x
        state.ball.y = ball_y
        state.goal_scored_flag = True
        state.score = {"Red": 5, "Blue": 3}

        # Call check_goal - should NOT increment any score
        engine.check_goal(state)

        # Scores must remain unchanged regardless of ball position
        assert state.score["Red"] == 5, (
            f"Red score changed from 5 to {state.score['Red']} "
            f"despite goal_scored_flag=True at ball ({ball_x}, {ball_y})"
        )
        assert state.score["Blue"] == 3, (
            f"Blue score changed from 3 to {state.score['Blue']} "
            f"despite goal_scored_flag=True at ball ({ball_x}, {ball_y})"
        )


# Feature: the-pitch, Property 3: Ball velocity cap
# **Validates: Requirements 4.7**
class TestBallVelocityCap:
    """Property 3: Ball velocity cap.

    For any ball velocity after any operation, the velocity magnitude
    shall never exceed 40 pixels per tick.

    **Validates: Requirements 4.7**
    """

    @settings(max_examples=100)
    @given(
        vx=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        vy=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    def test_velocity_magnitude_never_exceeds_max(self, vx: float, vy: float) -> None:
        """After cap_velocity, the ball speed magnitude is at most 40.0."""
        from pitch.physics import PhysicsEngine

        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)

        ball = Ball(vx=vx, vy=vy)
        engine.cap_velocity(ball)

        speed = math.sqrt(ball.vx**2 + ball.vy**2)
        assert speed <= 40.0 + 1e-9, (
            f"Ball speed {speed} exceeds max 40.0 after cap_velocity "
            f"(input vx={vx}, vy={vy})"
        )

    @settings(max_examples=100)
    @given(
        vx=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        vy=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    def test_direction_preserved_after_cap(self, vx: float, vy: float) -> None:
        """After cap_velocity, the direction (vx/vy ratio) is preserved when both are non-zero."""
        from pitch.physics import PhysicsEngine

        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)

        ball = Ball(vx=vx, vy=vy)
        engine.cap_velocity(ball)

        # Only check direction preservation when both components are non-zero
        if abs(vx) > 1e-12 and abs(vy) > 1e-12:
            original_ratio = vx / vy
            capped_ratio = ball.vx / ball.vy
            assert math.isclose(original_ratio, capped_ratio, rel_tol=1e-7), (
                f"Direction not preserved: original ratio {original_ratio} != "
                f"capped ratio {capped_ratio} (input vx={vx}, vy={vy})"
            )



# Feature: the-pitch, Property 1: Friction convergence
class TestFrictionConvergence:
    """Property 1: Friction convergence.

    For any initial ball velocity within the valid range (magnitude <= 40 px/tick),
    repeatedly applying the friction factor (multiplying velocity components by the
    friction coefficient each tick) shall reduce the ball's speed to below 0.1 px/tick
    within a bounded number of ticks.

    **Validates: Requirements 4.2**
    """

    # Calculate the maximum ticks needed for worst case:
    # max initial speed = sqrt(40^2 + 40^2) = 40*sqrt(2) ≈ 56.57
    # 0.97^n * 56.57 < 0.1 → n > log(0.1/56.57) / log(0.97) ≈ 209
    MAX_TICKS = 210

    @given(
        vx=st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False),
        vy=st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_friction_reduces_speed_below_threshold(self, vx: float, vy: float):
        """Applying friction repeatedly reduces speed below 0.1 within bounded ticks."""
        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)

        ball = Ball(x=600.0, y=400.0, vx=vx, vy=vy)

        for _ in range(self.MAX_TICKS):
            engine.apply_friction(ball)

        speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
        assert speed < 0.1, (
            f"After {self.MAX_TICKS} ticks of friction (factor={FRICTION}), "
            f"speed={speed:.6f} is not below 0.1. "
            f"Initial velocity: vx={vx}, vy={vy}"
        )


# Feature: the-pitch, Property 2: Ball boundary invariant
class TestBallBoundaryInvariant:
    """Property 2: Ball boundary invariant.

    For any ball position and velocity, after a physics tick (position update +
    boundary handling), the ball's position shall always remain within the bounds
    [0, 1200] x [0, 800], and if the ball was at a boundary, the velocity
    component perpendicular to that boundary shall be negated.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(
        x=st.floats(min_value=-50.0, max_value=1250.0, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-50.0, max_value=850.0, allow_nan=False, allow_infinity=False),
        vx=st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False),
        vy=st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_ball_position_stays_within_bounds_after_boundary_handling(
        self, x: float, y: float, vx: float, vy: float
    ):
        """Ball position is always within [0, 1200] x [0, 800] after
        handle_boundary_collision(), and velocity reflects at boundaries."""
        sm = StateManager()
        engine = PhysicsEngine(sm)

        ball = Ball(x=x, y=y, vx=vx, vy=vy)

        # Record original velocity before boundary handling
        original_vx = ball.vx
        original_vy = ball.vy

        # Call handle_boundary_collision
        engine.handle_boundary_collision(ball)

        # Assert position is within bounds
        assert 0.0 <= ball.x <= 1200.0, (
            f"Ball x={ball.x} out of bounds [0, 1200] "
            f"(original x={x}, vx={vx})"
        )
        assert 0.0 <= ball.y <= 800.0, (
            f"Ball y={ball.y} out of bounds [0, 800] "
            f"(original y={y}, vy={vy})"
        )

        # If original position was outside left or right bounds,
        # verify vx was negated
        if x <= 0.0 or x >= 1200.0:
            assert ball.vx == -original_vx, (
                f"Expected vx to be negated: got {ball.vx}, "
                f"expected {-original_vx} (original x={x})"
            )

        # If original position was outside top or bottom bounds,
        # verify vy was negated
        if y <= 0.0 or y >= 800.0:
            assert ball.vy == -original_vy, (
                f"Expected vy to be negated: got {ball.vy}, "
                f"expected {-original_vy} (original y={y})"
            )


# Feature: the-pitch, Property 8: Timer decrement
class TestTimerDecrement:
    """Property 8: Timer decrement.

    For any game state where match_state is Playing and time_left > 0,
    a physics tick of duration dt shall reduce time_left by exactly dt,
    and the resulting time_left shall never be negative (clamped to 0.0).

    **Validates: Requirements 5.4**
    """

    @settings(max_examples=100)
    @given(
        time_left=st.floats(min_value=0.1, max_value=90.0),
        dt=st.floats(min_value=0.001, max_value=1.0),
    )
    def test_timer_decrements_by_dt_and_never_negative(
        self, time_left: float, dt: float
    ) -> None:
        """Timer decreases by exactly dt when time_left > dt, and never goes negative.

        When time_left <= dt, the timer reaches 0 and reset_match() is called,
        which sets time_left back to 90.0. The key invariant is that time_left
        is NEVER negative.
        """
        # Arrange
        state_manager = StateManager()
        engine = PhysicsEngine(state_manager)
        state = state_manager.state
        state.match_state = MatchState.PLAYING
        state.time_left = time_left

        # Act
        engine.decrement_timer(state, dt)

        # Assert: time_left is NEVER negative
        assert state.time_left >= 0.0, (
            f"time_left went negative: {state.time_left} "
            f"(was {time_left}, dt={dt})"
        )

        # Assert: correct decrement behavior
        if time_left > dt:
            # Timer should decrease by exactly dt
            assert math.isclose(state.time_left, time_left - dt, rel_tol=1e-9), (
                f"Expected time_left={time_left - dt}, got {state.time_left} "
                f"(was {time_left}, dt={dt})"
            )
        else:
            # Timer reached 0, reset_match() was called, time_left is now 90.0
            assert state.time_left == 90.0, (
                f"Expected time_left=90.0 after reset, got {state.time_left} "
                f"(was {time_left}, dt={dt})"
            )
