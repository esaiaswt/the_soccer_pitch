"""Spatial utility module for pre-computing game state analysis.

Computes distances, direction vectors, kick range checks, and relative
positions so the LLM can focus on tactical decisions rather than math.
"""

import math
from typing import Optional


# Pitch constants (matching the server)
PITCH_WIDTH = 1200
PITCH_HEIGHT = 800
POSSESSION_RANGE = 30

# Goal positions (center of each goal zone in the play area)
GOAL_RED = {"x": 0.0, "y": 425.0}      # Red defends left
GOAL_BLUE = {"x": 1200.0, "y": 425.0}  # Blue defends right


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_direction(x1: float, y1: float, x2: float, y2: float) -> dict:
    """Normalized direction vector from (x1,y1) toward (x2,y2).

    Returns dx, dy in [-1.0, 1.0]. Returns (0, 0) if points are identical.
    """
    dist = compute_distance(x1, y1, x2, y2)
    if dist == 0:
        return {"dx": 0.0, "dy": 0.0}
    return {
        "dx": round((x2 - x1) / dist, 4),
        "dy": round((y2 - y1) / dist, 4),
    }


def analyze_game_state(game_state: dict, team: str, position: str) -> dict:
    """Compute spatial analysis from the game state for a specific player.

    Args:
        game_state: The raw game state dict from the server.
        team: The agent's team ("Red" or "Blue").
        position: The agent's position (e.g., "Striker").

    Returns:
        A dict with pre-computed spatial data for the LLM:
        - ball_distance: distance to ball in pixels
        - ball_direction: normalized (dx, dy) vector toward ball
        - in_kick_range: whether within 30px of ball
        - goal_to_attack: position of the opponent's goal
        - goal_distance: distance to the opponent's goal
        - goal_direction: normalized vector toward opponent's goal
        - own_goal_distance: distance to own goal (for defensive awareness)
        - nearest_opponent: distance to closest opponent player
        - nearest_teammate: distance to closest teammate
        - player_position: current (x, y) of this agent
    """
    player_key = f"{team}_{position}"
    players = game_state.get("players", {})
    ball = game_state.get("ball", {"x": 600.0, "y": 400.0})

    # Find our player position
    player_pos = players.get(player_key)
    if player_pos is None:
        # Player not yet spawned, use center as fallback
        player_pos = {"x": 600.0, "y": 400.0}

    px, py = player_pos["x"], player_pos["y"]
    bx, by = ball["x"], ball["y"]

    # Ball analysis
    ball_distance = compute_distance(px, py, bx, by)
    ball_direction = compute_direction(px, py, bx, by)
    in_kick_range = ball_distance <= POSSESSION_RANGE

    # Goal analysis - Red attacks Blue's goal (right), Blue attacks Red's goal (left)
    if team == "Red":
        attack_goal = GOAL_BLUE
        defend_goal = GOAL_RED
    else:
        attack_goal = GOAL_RED
        defend_goal = GOAL_BLUE

    goal_distance = compute_distance(px, py, attack_goal["x"], attack_goal["y"])
    goal_direction = compute_direction(px, py, attack_goal["x"], attack_goal["y"])
    own_goal_distance = compute_distance(px, py, defend_goal["x"], defend_goal["y"])

    # Nearest opponent and teammate
    nearest_opponent = _find_nearest(px, py, players, team, is_teammate=False)
    nearest_teammate = _find_nearest(px, py, players, team, is_teammate=True)

    # Shoot direction: from ball toward the opponent goal (for aiming shots)
    shoot_direction = compute_direction(bx, by, attack_goal["x"], attack_goal["y"])

    # Alignment check: is the player behind the ball relative to the goal?
    # "Behind" means the player is on the opposite side of the ball from the goal
    if team == "Red":
        # Red attacks right (x=1200), so player should be to the LEFT of the ball
        is_aligned = px < bx
    else:
        # Blue attacks left (x=0), so player should be to the RIGHT of the ball
        is_aligned = px > bx

    return {
        "player_position": {"x": round(px, 1), "y": round(py, 1)},
        "ball_distance": round(ball_distance, 1),
        "ball_direction": ball_direction,
        "in_kick_range": in_kick_range,
        "goal_to_attack": attack_goal,
        "goal_distance": round(goal_distance, 1),
        "goal_direction": goal_direction,
        "shoot_direction": shoot_direction,
        "is_behind_ball": is_aligned,
        "own_goal_distance": round(own_goal_distance, 1),
        "nearest_opponent_distance": nearest_opponent,
        "nearest_teammate_distance": nearest_teammate,
    }


def _find_nearest(
    px: float, py: float, players: dict, team: str, is_teammate: bool
) -> Optional[float]:
    """Find distance to nearest teammate or opponent.

    Args:
        px, py: Current player position.
        players: Dict of player_key -> {x, y}.
        team: The agent's team.
        is_teammate: If True, find nearest same-team player; else opponent.

    Returns:
        Distance to nearest matching player, or None if no match found.
    """
    min_dist = None
    player_key = None

    for key, pos in players.items():
        # Skip self
        key_team = key.split("_")[0]
        same_team = key_team == team

        if is_teammate and not same_team:
            continue
        if not is_teammate and same_team:
            continue

        # Skip self (same position check)
        dist = compute_distance(px, py, pos["x"], pos["y"])
        if dist == 0:
            continue  # This is us

        if min_dist is None or dist < min_dist:
            min_dist = dist

    return round(min_dist, 1) if min_dist is not None else None


def format_spatial_summary(analysis: dict) -> str:
    """Format the spatial analysis as a concise text block for the LLM.

    Returns a human-readable summary that gets appended to the game state
    in the user message sent to the LLM.
    """
    lines = [
        "--- SPATIAL ANALYSIS (pre-computed) ---",
        f"Your position: ({analysis['player_position']['x']}, {analysis['player_position']['y']})",
        f"Ball distance: {analysis['ball_distance']}px",
        f"Ball direction (normalized): dx={analysis['ball_direction']['dx']}, dy={analysis['ball_direction']['dy']}",
        f"In kick range (<30px): {'YES' if analysis['in_kick_range'] else 'NO'}",
        f"Behind ball (good shooting position): {'YES' if analysis['is_behind_ball'] else 'NO'}",
        f"Opponent goal distance: {analysis['goal_distance']}px",
        f"Opponent goal direction: dx={analysis['goal_direction']['dx']}, dy={analysis['goal_direction']['dy']}",
        f"Shoot direction (ball→goal): dx={analysis['shoot_direction']['dx']}, dy={analysis['shoot_direction']['dy']}",
        f"Own goal distance: {analysis['own_goal_distance']}px",
    ]

    if analysis["nearest_opponent_distance"] is not None:
        lines.append(f"Nearest opponent: {analysis['nearest_opponent_distance']}px away")
    if analysis["nearest_teammate_distance"] is not None:
        lines.append(f"Nearest teammate: {analysis['nearest_teammate_distance']}px away")

    if analysis["in_kick_range"] and analysis["is_behind_ball"]:
        lines.append("RECOMMENDATION: Perfect position! Kick now — you are behind the ball facing the goal.")
    elif analysis["in_kick_range"] and not analysis["is_behind_ball"]:
        lines.append("RECOMMENDATION: In kick range but NOT behind the ball. Move behind the ball first (get between your own goal and the ball), then kick.")
    elif not analysis["in_kick_range"]:
        lines.append("RECOMMENDATION: Move toward the ball. Get behind it (between ball and your own goal) for a clean shot.")

    return "\n".join(lines)
