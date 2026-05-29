"""Physics engine module for The Pitch.

Runs a deterministic physics simulation at 60 ticks/second in a
dedicated thread. Handles ball friction, boundary collisions,
velocity capping, goal detection, timer management, and goal pauses.
"""

import math
import time
from typing import Callable, Optional

from pitch.config import Config
from pitch.state import Ball, GameState, MatchState, StateManager

_config = Config()

# Physics constants from config
TICK_RATE: int = _config.PHYSICS_TICK_RATE
FRICTION: float = _config.FRICTION
MAX_BALL_SPEED: float = _config.MAX_BALL_SPEED
GOAL_PAUSE_DURATION: float = _config.GOAL_PAUSE


class PhysicsEngine:
    """Deterministic physics engine running at a fixed 60Hz tick rate.

    Manages ball movement, friction, boundary collisions, velocity
    capping, goal detection, timer countdown, and goal pause logic.
    """

    def __init__(
        self,
        state_manager: StateManager,
        on_goal: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the physics engine.

        Args:
            state_manager: Thread-safe state manager for game state access.
            on_goal: Optional callback invoked when a goal is scored.
        """
        self._state_manager = state_manager
        self._on_goal = on_goal
        self._running = True
        self._goal_pause_remaining: float = 0.0

    def run(self) -> None:
        """Main physics loop at 60Hz using fixed-timestep sleep.

        Only processes ticks when match_state is PLAYING.
        Uses time.sleep to maintain a consistent tick rate.
        """
        dt = 1.0 / TICK_RATE

        while self._running:
            start_time = time.perf_counter()

            # Only process physics when match is playing
            if self._state_manager.state.match_state == MatchState.PLAYING:
                self.tick(dt)

            # Sleep to maintain fixed tick rate
            elapsed = time.perf_counter() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def tick(self, dt: float) -> None:
        """Execute a single physics step.

        Acquires the state lock, then applies friction, updates position,
        handles boundaries, caps velocity, checks goals, and decrements
        the timer.

        Args:
            dt: Time delta for this tick (typically 1/60 seconds).
        """
        if not self._state_manager.acquire():
            return  # Skip tick if lock unavailable

        try:
            state = self._state_manager.state
            ball = state.ball

            # Handle goal pause: freeze physics during pause
            if self._goal_pause_remaining > 0:
                self._goal_pause_remaining -= dt
                if self._goal_pause_remaining <= 0:
                    self._goal_pause_remaining = 0.0
                    state.goal_scored_flag = False
                    self._state_manager.reset_after_goal()
                return

            # Guard against NaN/Inf ball positions
            if not self._is_valid_position(ball):
                self._reset_ball_to_center(ball)
                return

            # Apply physics steps
            self.apply_friction(ball)
            self.update_ball_position(ball)
            self.handle_boundary_collision(ball)
            self.cap_velocity(ball)

            # Check for NaN/Inf after position update
            if not self._is_valid_position(ball):
                self._reset_ball_to_center(ball)
                return

            # Check for goals
            self.check_goal(state)

            # Decrement timer
            self.decrement_timer(state, dt)

        finally:
            self._state_manager.release()

    def apply_friction(self, ball: Ball) -> None:
        """Apply friction to ball velocity.

        Multiplies both velocity components by the FRICTION coefficient
        (0.97) each tick, causing the ball to decelerate over time.

        Args:
            ball: The ball entity to apply friction to.
        """
        ball.vx *= FRICTION
        ball.vy *= FRICTION

    def update_ball_position(self, ball: Ball) -> None:
        """Update ball position based on current velocity.

        Adds velocity components to position components.

        Args:
            ball: The ball entity to update.
        """
        ball.x += ball.vx
        ball.y += ball.vy

    def handle_boundary_collision(self, ball: Ball) -> None:
        """Handle ball collisions with pitch boundaries.

        Clamps ball position to [0, 1200] x [0, 800] and negates
        the velocity component perpendicular to any boundary hit.

        Args:
            ball: The ball entity to check and correct.
        """
        pitch_width = float(_config.PITCH_WIDTH)
        pitch_height = float(_config.PITCH_HEIGHT)

        # Left boundary
        if ball.x <= 0.0:
            ball.x = 0.0
            ball.vx = -ball.vx

        # Right boundary
        elif ball.x >= pitch_width:
            ball.x = pitch_width
            ball.vx = -ball.vx

        # Top boundary
        if ball.y <= 0.0:
            ball.y = 0.0
            ball.vy = -ball.vy

        # Bottom boundary
        elif ball.y >= pitch_height:
            ball.y = pitch_height
            ball.vy = -ball.vy

    def cap_velocity(self, ball: Ball) -> None:
        """Cap ball velocity to MAX_BALL_SPEED (40 px/tick).

        If the velocity magnitude exceeds the maximum, scale both
        components down proportionally to maintain direction.

        Args:
            ball: The ball entity to cap velocity for.
        """
        speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
        if speed > MAX_BALL_SPEED:
            scale = MAX_BALL_SPEED / speed
            ball.vx *= scale
            ball.vy *= scale

    def check_goal(self, state: GameState) -> None:
        """Detect if the ball has entered a goal zone and handle scoring.

        Goal zones:
        - Left goal: x in [0, 30], y in [300, 500] → Blue team scores
        - Right goal: x in [1170, 1200], y in [300, 500] → Red team scores

        Uses goal_scored_flag to prevent double-scoring on the same
        goal zone entry.

        Args:
            state: The current game state.
        """
        if state.goal_scored_flag:
            return

        ball = state.ball

        # Left goal zone: Blue scores
        if ball.x <= 30.0 and 300.0 <= ball.y <= 500.0:
            state.score["Blue"] += 1
            state.goal_scored_flag = True
            self._goal_pause_remaining = GOAL_PAUSE_DURATION
            if self._on_goal:
                self._on_goal()

        # Right goal zone: Red scores
        elif ball.x >= 1170.0 and 300.0 <= ball.y <= 500.0:
            state.score["Red"] += 1
            state.goal_scored_flag = True
            self._goal_pause_remaining = GOAL_PAUSE_DURATION
            if self._on_goal:
                self._on_goal()

    def decrement_timer(self, state: GameState, dt: float) -> None:
        """Decrement the match timer by dt, clamped to 0.0.

        When the timer reaches zero, triggers a match reset via
        the state manager.

        Args:
            state: The current game state.
            dt: Time delta to subtract from the timer.
        """
        state.time_left -= dt
        if state.time_left <= 0.0:
            state.time_left = 0.0
            self._state_manager.reset_match()

    def stop(self) -> None:
        """Signal the physics loop to stop."""
        self._running = False

    def _is_valid_position(self, ball: Ball) -> bool:
        """Check if ball position and velocity are valid (not NaN/Inf).

        Args:
            ball: The ball entity to validate.

        Returns:
            True if all values are finite, False otherwise.
        """
        return (
            math.isfinite(ball.x)
            and math.isfinite(ball.y)
            and math.isfinite(ball.vx)
            and math.isfinite(ball.vy)
        )

    def _reset_ball_to_center(self, ball: Ball) -> None:
        """Reset ball to center position with zero velocity.

        Used when NaN/Inf values are detected.

        Args:
            ball: The ball entity to reset.
        """
        ball.x = 600.0
        ball.y = 400.0
        ball.vx = 0.0
        ball.vy = 0.0
