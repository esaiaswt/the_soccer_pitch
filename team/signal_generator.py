"""Signal Generator module for the team/ application.

Generates signals automatically from the player's current plan and game state.
All logic executes in Python without LLM calls.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import time

from team.planner import Plan
from team.signal_bus import Signal


def _is_dead_ball(game_state: dict) -> bool:
    """Check if the game is in a dead ball state (kickoff, goal scored, etc.)."""
    return game_state.get("dead_ball", False) or game_state.get("is_dead_ball", False)


def _get_ball_position(game_state: dict) -> tuple[float, float]:
    """Extract ball position from game state."""
    ball = game_state.get("ball", {})
    return ball.get("x", 600.0), ball.get("y", 400.0)


def _get_player_position_coords(game_state: dict, team: str, position: str) -> tuple[float, float]:
    """Extract a specific player's coordinates from game state."""
    players = game_state.get("players", {})
    key = f"{team}_{position}"
    player = players.get(key, {})
    return player.get("x", 600.0), player.get("y", 400.0)


def _compute_ball_distance(game_state: dict, team: str, position: str) -> float:
    """Compute the distance between the player and the ball."""
    ball_x, ball_y = _get_ball_position(game_state)
    player_x, player_y = _get_player_position_coords(game_state, team, position)
    return ((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2) ** 0.5


def _classify_zone(game_state: dict, team: str, position: str) -> str:
    """Classify the player's current zone on the pitch.

    Divides the pitch into zones based on player position:
    - X-axis: defense (own third), midfield (middle third), attack (opponent third)
    - Y-axis: left, center, right

    Returns a string like "left_attack", "center_midfield", "right_defense".
    """
    player_x, player_y = _get_player_position_coords(game_state, team, position)

    # Pitch dimensions: 1200 x 800
    # Determine lateral zone (y-axis): left (0-267), center (267-533), right (533-800)
    if player_y < 267:
        lateral = "left"
    elif player_y < 533:
        lateral = "center"
    else:
        lateral = "right"

    # Determine longitudinal zone (x-axis) relative to team direction
    # Red attacks right (x > 800 = attack, 400-800 = midfield, < 400 = defense)
    # Blue attacks left (x < 400 = attack, 400-800 = midfield, > 800 = defense)
    if team == "Red":
        if player_x > 800:
            longitudinal = "attack"
        elif player_x > 400:
            longitudinal = "midfield"
        else:
            longitudinal = "defense"
    else:
        if player_x < 400:
            longitudinal = "attack"
        elif player_x < 800:
            longitudinal = "midfield"
        else:
            longitudinal = "defense"

    return f"{lateral}_{longitudinal}"


def _has_teammate_making_run(game_state: dict, team: str, position: str) -> bool:
    """Check if a teammate is making a run (has a 'making_run' signal or is moving forward).

    Checks the game state for teammate signals indicating a run, or
    checks if any teammate is in an advanced position relative to the ball.
    """
    # Check for teammate signals in game state
    signals = game_state.get("signals", [])
    for signal in signals:
        if isinstance(signal, dict):
            if (
                signal.get("signal_type") == "making_run"
                and signal.get("sender_position") != position
            ):
                return True

    # Fallback: check if any teammate is making a forward run
    # (positioned ahead of the ball in the attacking direction)
    ball_x, ball_y = _get_ball_position(game_state)
    players = game_state.get("players", {})
    my_key = f"{team}_{position}"

    for player_name, pos in players.items():
        if player_name.startswith(f"{team}_") and player_name != my_key:
            px = pos.get("x", 0.0)
            # Check if teammate is ahead of ball in attacking direction
            if team == "Red" and px > ball_x + 50:
                return True
            elif team == "Blue" and px < ball_x - 50:
                return True

    return False


def _is_nearest_to_ball_carrier(game_state: dict, team: str, position: str) -> bool:
    """Check if this player is the nearest teammate to the ball carrier.

    The ball carrier is the player closest to the ball on the same team.
    This player must be the second-closest (nearest support player).
    """
    ball_x, ball_y = _get_ball_position(game_state)
    players = game_state.get("players", {})
    my_key = f"{team}_{position}"

    # Find all teammates and their distances to ball
    team_distances: list[tuple[str, float]] = []
    for player_name, pos in players.items():
        if player_name.startswith(f"{team}_"):
            px = pos.get("x", 0.0)
            py = pos.get("y", 0.0)
            dist = ((px - ball_x) ** 2 + (py - ball_y) ** 2) ** 0.5
            team_distances.append((player_name, dist))

    if len(team_distances) < 2:
        return False

    # Sort by distance to ball
    team_distances.sort(key=lambda x: x[1])

    # The ball carrier is the closest player
    # This player should be the second closest (nearest to ball carrier / supporting)
    if len(team_distances) >= 2 and team_distances[1][0] == my_key:
        # Only consider as supporting if the ball carrier is actually close to ball
        # (within kick range, meaning they actually have the ball)
        if team_distances[0][1] <= 30:
            return True

    return False


def _sub_goal_benefits_from_awareness(plan: Plan) -> bool:
    """Check if the current sub-goal benefits from teammate awareness.

    Returns True if the current sub-goal description contains keywords
    indicating it would benefit from teammates knowing about it
    (e.g., "receive_pass", "receive pass").
    """
    if plan.completed or plan.current_index >= len(plan.sub_goals):
        return False

    description = plan.sub_goals[plan.current_index].description.lower()
    awareness_keywords = ["receive_pass", "receive pass", "pass", "distribute", "teammate"]
    return any(keyword in description for keyword in awareness_keywords)


class SignalGenerator:
    """Generates signals from the player's current plan and game state.

    All logic executes in Python without LLM calls. Signals are generated
    based on spatial analysis and plan state.
    """

    def generate(
        self,
        plan: Plan | None,
        game_state: dict,
        team: str,
        position: str,
    ) -> Signal | None:
        """Generate a signal based on the current plan and game state.

        Rules (evaluated in priority order):
        1. Dead ball state → return None
        2. Plan has sub-goal benefiting from teammate awareness → "requesting_pass"
        3. In kick range and teammate making a run → "ready_to_pass"
        4. Nearest to ball carrier → "supporting" with current zone

        Parameters
        ----------
        plan : Plan | None
            The player's current active plan, or None.
        game_state : dict
            The current game state snapshot.
        team : str
            The team color ("Red" or "Blue").
        position : str
            The player's position (e.g., "Striker", "Midfielder").

        Returns
        -------
        Signal | None
            The generated signal, or None if no signal should be published.
        """
        # Rule: Dead ball state → return None
        if _is_dead_ball(game_state):
            return None

        # Rule: Plan has sub-goal benefiting from teammate awareness → "requesting_pass"
        if plan is not None and _sub_goal_benefits_from_awareness(plan):
            return Signal(
                sender_position=position,
                signal_type="requesting_pass",
                payload=plan.sub_goals[plan.current_index].description[:50],
                timestamp=time.time(),
            )

        # Rule: In kick range and teammate making a run → "ready_to_pass"
        ball_distance = _compute_ball_distance(game_state, team, position)
        if ball_distance <= 30 and _has_teammate_making_run(game_state, team, position):
            return Signal(
                sender_position=position,
                signal_type="ready_to_pass",
                payload="teammate_in_space",
                timestamp=time.time(),
            )

        # Rule: Nearest to ball carrier → "supporting" with current zone
        if _is_nearest_to_ball_carrier(game_state, team, position):
            zone = _classify_zone(game_state, team, position)
            return Signal(
                sender_position=position,
                signal_type="supporting",
                payload=zone,
                timestamp=time.time(),
            )

        return None
