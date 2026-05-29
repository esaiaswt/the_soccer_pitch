"""Game state module for The Pitch.

Defines data models (MatchState, Ball, Player, GameState) and the
thread-safe StateManager that wraps all state access with a lock.
"""

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pitch.config import Config

_config = Config()


class MatchState(Enum):
    """Match lifecycle states."""

    WAITING = "Waiting"
    PLAYING = "Playing"


@dataclass
class Ball:
    """Ball entity with position and velocity."""

    x: float = 600.0
    y: float = 400.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Player:
    """Player entity with name, team, and position."""

    name: str
    team: str  # "Red" or "Blue"
    x: float
    y: float


@dataclass
class GameState:
    """Complete game state for a match."""

    match_state: MatchState = MatchState.WAITING
    time_left: float = 90.0
    score: dict = field(default_factory=lambda: {"Red": 0, "Blue": 0})
    ball: Ball = field(default_factory=Ball)
    players: dict = field(default_factory=dict)  # name -> Player
    goal_scored_flag: bool = False


# Default starting positions for each team.
# Red team occupies x=100-550 (left half), Blue team occupies x=650-1100 (right half).
# Positions are distributed vertically across the pitch height (800px).
DEFAULT_POSITIONS = {
    "Red": [
        {"x": 100.0, "y": 400.0},   # Goalkeeper
        {"x": 250.0, "y": 200.0},   # Defender left
        {"x": 250.0, "y": 600.0},   # Defender right
        {"x": 400.0, "y": 300.0},   # Midfielder left
        {"x": 400.0, "y": 500.0},   # Midfielder right
        {"x": 550.0, "y": 400.0},   # Striker
    ],
    "Blue": [
        {"x": 1100.0, "y": 400.0},  # Goalkeeper
        {"x": 950.0, "y": 200.0},   # Defender left
        {"x": 950.0, "y": 600.0},   # Defender right
        {"x": 800.0, "y": 300.0},   # Midfielder left
        {"x": 800.0, "y": 500.0},   # Midfielder right
        {"x": 650.0, "y": 400.0},   # Striker
    ],
}


def _get_default_position(team: str, position: str) -> dict:
    """Get a default starting position for a player.

    Uses a hash of the position name to pick from the team's available
    default positions, ensuring consistent placement for the same name.
    """
    positions = DEFAULT_POSITIONS.get(team, DEFAULT_POSITIONS["Red"])
    index = hash(position) % len(positions)
    return positions[index]


class StateManager:
    """Thread-safe wrapper around GameState with a threading.Lock.

    All state access should go through this manager to ensure
    consistency across the API, physics, and renderer threads.
    """

    def __init__(self) -> None:
        self._state: GameState = GameState()
        self._lock: threading.Lock = threading.Lock()

    @property
    def state(self) -> GameState:
        """Direct access to state (caller must hold lock)."""
        return self._state

    def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire the state lock with a timeout.

        Args:
            timeout: Maximum seconds to wait for the lock.

        Returns:
            True if the lock was acquired, False on timeout.
        """
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        """Release the state lock."""
        self._lock.release()

    def read_snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the current game state.

        Acquires the lock internally. Returns a dict suitable for
        the GET /api/state response.
        """
        if not self.acquire():
            raise TimeoutError("Failed to acquire state lock")
        try:
            state = self._state
            players_snapshot = {}
            for name, player in state.players.items():
                players_snapshot[name] = {"x": player.x, "y": player.y}

            return {
                "match_state": state.match_state.value,
                "time_left": state.time_left,
                "score": dict(state.score),
                "ball": {"x": state.ball.x, "y": state.ball.y},
                "players": players_snapshot,
            }
        finally:
            self.release()

    def apply_action(
        self,
        team: str,
        position: str,
        vector: dict,
        kick: bool,
    ) -> dict:
        """Apply a player action to the game state.

        Clamps dx/dy to [-1, 1], multiplies by MAX_SPEED, moves the player,
        and optionally applies a kick if within possession range.

        Args:
            team: "Red" or "Blue"
            position: Player position name (e.g., "Striker")
            vector: Dict with "dx" and "dy" float values
            kick: Whether the player is attempting to kick

        Returns:
            Dict with result status.
        """
        player_name = f"{team}_{position}"
        dx = max(-1.0, min(1.0, float(vector.get("dx", 0.0))))
        dy = max(-1.0, min(1.0, float(vector.get("dy", 0.0))))

        move_x = dx * _config.MAX_SPEED
        move_y = dy * _config.MAX_SPEED

        state = self._state

        # Spawn player if not exists
        if player_name not in state.players:
            default_pos = _get_default_position(team, position)
            state.players[player_name] = Player(
                name=player_name,
                team=team,
                x=default_pos["x"],
                y=default_pos["y"],
            )

        player = state.players[player_name]

        # Apply movement
        player.x += move_x
        player.y += move_y

        # Clamp player position to pitch bounds
        player.x = max(0.0, min(float(_config.PITCH_WIDTH), player.x))
        player.y = max(0.0, min(float(_config.PITCH_HEIGHT), player.y))

        # Handle kick
        if kick:
            ball = state.ball
            dist = math.sqrt((player.x - ball.x) ** 2 + (player.y - ball.y) ** 2)
            if dist < _config.POSSESSION_RANGE:
                # Apply kick impulse in direction from player to ball
                if dist > 0:
                    direction_x = (ball.x - player.x) / dist
                    direction_y = (ball.y - player.y) / dist
                else:
                    # Player is exactly on ball, kick in positive x direction
                    direction_x = 1.0
                    direction_y = 0.0

                ball.vx += direction_x * _config.KICK_IMPULSE
                ball.vy += direction_y * _config.KICK_IMPULSE

        return {"status": "ok", "player": player_name}

    def reset_after_goal(self) -> None:
        """Reset positions after a goal is scored.

        Ball returns to center with zero velocity.
        All players return to their team's default positions.
        """
        state = self._state

        # Reset ball to center
        state.ball.x = 600.0
        state.ball.y = 400.0
        state.ball.vx = 0.0
        state.ball.vy = 0.0

        # Reset all players to default positions
        for name, player in state.players.items():
            default_pos = _get_default_position(player.team, name.split("_", 1)[1])
            player.x = default_pos["x"]
            player.y = default_pos["y"]

    def reset_match(self) -> None:
        """Reset match state for a new round.

        Transitions to Waiting, preserves score, resets positions and ball.
        """
        state = self._state

        # Transition to Waiting
        state.match_state = MatchState.WAITING

        # Reset timer
        state.time_left = 90.0

        # Reset ball
        state.ball.x = 600.0
        state.ball.y = 400.0
        state.ball.vx = 0.0
        state.ball.vy = 0.0

        # Reset goal flag
        state.goal_scored_flag = False

        # Reset all players to default positions
        for name, player in state.players.items():
            default_pos = _get_default_position(player.team, name.split("_", 1)[1])
            player.x = default_pos["x"]
            player.y = default_pos["y"]
