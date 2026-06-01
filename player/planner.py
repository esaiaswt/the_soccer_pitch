"""Multi-step planning module for decomposing high-level goals into sub-goal sequences.

Provides template-based plan generation and execution without LLM calls.
Plans are selected from a predefined template library based on game state
conditions (ball possession, field position, player proximity) and executed
across multiple Look-Think-Act cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# Pitch constants (matching spatial.py and the server)
PITCH_WIDTH = 1200
PITCH_HEIGHT = 800
POSSESSION_RANGE = 30
HALF_PITCH_X = PITCH_WIDTH / 2


@dataclass
class SubGoal:
    """A single sub-goal within a plan.

    Attributes:
        description: Human-readable description for LLM context.
        target_condition: Callable that evaluates whether this sub-goal
            is satisfied given (game_state, team, position) -> bool.
    """

    description: str
    target_condition: Callable[[dict, str, str], bool]


@dataclass
class Plan:
    """An active multi-step plan being executed by a player agent.

    Attributes:
        name: Template name this plan was instantiated from.
        sub_goals: Ordered sequence of sub-goals (max 5).
        current_index: Index of the currently active sub-goal.
        completed: True when all sub-goals have been satisfied.
    """

    name: str
    sub_goals: list[SubGoal] = field(default_factory=list)
    current_index: int = 0
    completed: bool = False


@dataclass
class PlanTemplate:
    """A predefined plan template that can be instantiated when conditions match.

    Attributes:
        name: Unique template identifier.
        trigger_condition: Callable that evaluates whether this template
            should activate given (game_state, team, position) -> bool.
        priority: Position-based priority mapping. Higher values are preferred
            when multiple templates match.
        sub_goals: The sub-goals to instantiate when this template is selected.
    """

    name: str
    trigger_condition: Callable[[dict, str, str], bool]
    priority: dict[str, int]
    sub_goals: list[SubGoal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions for game state queries
# ---------------------------------------------------------------------------


def _get_player_pos(game_state: dict, team: str, position: str) -> dict:
    """Get the player's position from game state, with center fallback."""
    players = game_state.get("players", {})
    key = f"{team}_{position}"
    return players.get(key, {"x": PITCH_WIDTH / 2, "y": PITCH_HEIGHT / 2})


def _get_ball_pos(game_state: dict) -> dict:
    """Get the ball position from game state."""
    return game_state.get("ball", {"x": PITCH_WIDTH / 2, "y": PITCH_HEIGHT / 2})


def _distance(p1: dict, p2: dict) -> float:
    """Euclidean distance between two position dicts with x, y keys."""
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return (dx * dx + dy * dy) ** 0.5


def _team_has_possession(game_state: dict, team: str) -> bool:
    """Check if any player on the given team is within possession range of the ball."""
    ball = _get_ball_pos(game_state)
    players = game_state.get("players", {})
    for key, pos in players.items():
        if key.startswith(f"{team}_"):
            if _distance(pos, ball) <= POSSESSION_RANGE:
                return True
    return False


def _opponent_team(team: str) -> str:
    """Return the opposing team name."""
    return "Blue" if team == "Red" else "Red"


def _ball_in_attacking_half(game_state: dict, team: str) -> bool:
    """Check if the ball is in the attacking half for the given team."""
    ball = _get_ball_pos(game_state)
    if team == "Red":
        # Red attacks right (x=1200), attacking half is x > 600
        return ball["x"] > HALF_PITCH_X
    else:
        # Blue attacks left (x=0), attacking half is x < 600
        return ball["x"] < HALF_PITCH_X


def _ball_in_own_half(game_state: dict, team: str) -> bool:
    """Check if the ball is in the team's own (defensive) half."""
    ball = _get_ball_pos(game_state)
    if team == "Red":
        # Red defends left (x=0), own half is x < 600
        return ball["x"] < HALF_PITCH_X
    else:
        # Blue defends right (x=1200), own half is x > 600
        return ball["x"] > HALF_PITCH_X


def _ball_is_loose(game_state: dict) -> bool:
    """Check if no player from either team has possession of the ball."""
    ball = _get_ball_pos(game_state)
    players = game_state.get("players", {})
    for _key, pos in players.items():
        if _distance(pos, ball) <= POSSESSION_RANGE:
            return False
    return True


def _player_is_nearest_to_ball(game_state: dict, team: str, position: str) -> bool:
    """Check if this player is the nearest player on their team to the ball."""
    ball = _get_ball_pos(game_state)
    players = game_state.get("players", {})
    player_key = f"{team}_{position}"
    player_pos = players.get(player_key)
    if player_pos is None:
        return False

    player_dist = _distance(player_pos, ball)

    for key, pos in players.items():
        if key == player_key:
            continue
        if key.startswith(f"{team}_"):
            if _distance(pos, ball) < player_dist:
                return False
    return True


def _is_goalkeeper(position: str) -> bool:
    """Check if the position is a goalkeeper."""
    return position.lower() == "goalkeeper"


def _player_in_kick_range(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is within kick range of the ball."""
    player_pos = _get_player_pos(game_state, team, position)
    ball = _get_ball_pos(game_state)
    return _distance(player_pos, ball) <= POSSESSION_RANGE


def _player_is_behind_ball(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is behind the ball relative to the attacking goal."""
    player_pos = _get_player_pos(game_state, team, position)
    ball = _get_ball_pos(game_state)
    if team == "Red":
        # Red attacks right, so behind means player.x < ball.x
        return player_pos["x"] < ball["x"]
    else:
        # Blue attacks left, so behind means player.x > ball.x
        return player_pos["x"] > ball["x"]


def _player_near_own_goal(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is within 200px of their own goal."""
    player_pos = _get_player_pos(game_state, team, position)
    if team == "Red":
        own_goal = {"x": 0.0, "y": 425.0}
    else:
        own_goal = {"x": 1200.0, "y": 425.0}
    return _distance(player_pos, own_goal) <= 200.0


# ---------------------------------------------------------------------------
# Template library: predefined plan templates
# ---------------------------------------------------------------------------


def _build_score_goal_template() -> PlanTemplate:
    """Build the score_goal plan template.

    Trigger: team has possession AND ball is in attacking half.
    Sub-goals: position behind ball → approach ball → kick toward goal.
    """

    def trigger(game_state: dict, team: str, position: str) -> bool:
        return (
            _team_has_possession(game_state, team)
            and _ball_in_attacking_half(game_state, team)
        )

    sub_goals = [
        SubGoal(
            description="Position behind the ball relative to opponent goal",
            target_condition=lambda gs, t, p: _player_is_behind_ball(gs, t, p),
        ),
        SubGoal(
            description="Approach the ball to get within kick range",
            target_condition=lambda gs, t, p: _player_in_kick_range(gs, t, p),
        ),
        SubGoal(
            description="Kick the ball toward the opponent goal",
            target_condition=lambda gs, t, p: (
                _player_in_kick_range(gs, t, p)
                and _player_is_behind_ball(gs, t, p)
            ),
        ),
    ]

    return PlanTemplate(
        name="score_goal",
        trigger_condition=trigger,
        priority={"Striker": 10, "Midfielder": 7, "Defender": 3, "Goalkeeper": 1},
        sub_goals=sub_goals,
    )


def _build_defend_goal_template() -> PlanTemplate:
    """Build the defend_goal plan template.

    Trigger: opponent has possession AND ball is in own half.
    Sub-goals: move toward own goal → intercept ball path.
    """

    def trigger(game_state: dict, team: str, position: str) -> bool:
        opponent = _opponent_team(team)
        return (
            _team_has_possession(game_state, opponent)
            and _ball_in_own_half(game_state, team)
        )

    sub_goals = [
        SubGoal(
            description="Move toward own goal to establish defensive position",
            target_condition=lambda gs, t, p: _player_near_own_goal(gs, t, p),
        ),
        SubGoal(
            description="Intercept the ball path to regain possession",
            target_condition=lambda gs, t, p: _player_in_kick_range(gs, t, p),
        ),
    ]

    return PlanTemplate(
        name="defend_goal",
        trigger_condition=trigger,
        priority={"Goalkeeper": 10, "Defender": 9, "Midfielder": 5, "Striker": 2},
        sub_goals=sub_goals,
    )


def _build_intercept_ball_template() -> PlanTemplate:
    """Build the intercept_ball plan template.

    Trigger: ball is loose AND player is nearest on their team.
    Sub-goals: move to ball predicted position → gain possession.
    """

    def trigger(game_state: dict, team: str, position: str) -> bool:
        return (
            _ball_is_loose(game_state)
            and _player_is_nearest_to_ball(game_state, team, position)
        )

    sub_goals = [
        SubGoal(
            description="Move to the ball's predicted position",
            target_condition=lambda gs, t, p: _distance(
                _get_player_pos(gs, t, p), _get_ball_pos(gs)
            )
            <= POSSESSION_RANGE * 2,
        ),
        SubGoal(
            description="Gain possession of the ball",
            target_condition=lambda gs, t, p: _player_in_kick_range(gs, t, p),
        ),
    ]

    return PlanTemplate(
        name="intercept_ball",
        trigger_condition=trigger,
        priority={"Midfielder": 8, "Striker": 7, "Defender": 6, "Goalkeeper": 3},
        sub_goals=sub_goals,
    )


def _build_distribute_ball_template() -> PlanTemplate:
    """Build the distribute_ball plan template.

    Trigger: goalkeeper has possession.
    Sub-goals: identify nearest teammate → kick toward teammate.
    """

    def trigger(game_state: dict, team: str, position: str) -> bool:
        return (
            _is_goalkeeper(position)
            and _player_in_kick_range(game_state, team, position)
        )

    sub_goals = [
        SubGoal(
            description="Identify nearest teammate for distribution",
            target_condition=lambda gs, t, p: (
                _is_goalkeeper(p) and _player_in_kick_range(gs, t, p)
            ),
        ),
        SubGoal(
            description="Kick the ball toward the nearest teammate",
            target_condition=lambda gs, t, p: not _player_in_kick_range(gs, t, p),
        ),
    ]

    return PlanTemplate(
        name="distribute_ball",
        trigger_condition=trigger,
        priority={"Goalkeeper": 10, "Defender": 1, "Midfielder": 1, "Striker": 1},
        sub_goals=sub_goals,
    )


def get_default_templates() -> list[PlanTemplate]:
    """Return the default template library with all predefined plan templates."""
    return [
        _build_score_goal_template(),
        _build_defend_goal_template(),
        _build_intercept_ball_template(),
        _build_distribute_ball_template(),
    ]


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------


class Planner:
    """Evaluates game state against plan templates and manages plan lifecycle.

    The Planner selects plans from a template library based on current game
    conditions, advances active plans as sub-goals are satisfied, and signals
    when plans should be abandoned. All logic executes in Python without LLM calls.
    """

    def __init__(self, templates: list[PlanTemplate] | None = None) -> None:
        """Initialize the Planner with a template library.

        Args:
            templates: List of PlanTemplates to evaluate. If None, uses the
                default template library.
        """
        self._templates = templates if templates is not None else get_default_templates()

    def evaluate(
        self,
        game_state: dict,
        team: str,
        position: str,
        active_plan: Plan | None,
    ) -> Plan | None:
        """Evaluate game state and return the appropriate plan.

        1. If no active plan, find matching templates and select highest priority.
        2. If active plan exists, check if a higher-priority template now matches
           and replace if so.
        3. Return the plan (new, replaced, or existing).

        Args:
            game_state: Current game state dict from the server.
            team: The agent's team ("Red" or "Blue").
            position: The agent's position (e.g., "Striker").
            active_plan: The currently active plan, or None.

        Returns:
            A Plan instance (new or existing), or None if no template matches.
        """
        best_template = self._find_best_template(game_state, team, position)

        if active_plan is None:
            # No active plan — instantiate from best matching template
            if best_template is None:
                return None
            return self._instantiate_plan(best_template)

        # Active plan exists — check if a higher-priority template should replace it
        if best_template is not None and best_template.name != active_plan.name:
            active_priority = self._get_template_priority(
                active_plan.name, position
            )
            new_priority = best_template.priority.get(position, 0)
            if new_priority > active_priority:
                return self._instantiate_plan(best_template)

        return active_plan

    def advance(
        self, plan: Plan, game_state: dict, team: str, position: str
    ) -> Plan:
        """Advance the plan if the current sub-goal's target condition is satisfied.

        1. Check if current sub-goal's target_condition is satisfied.
        2. If satisfied and not final: advance current_index by 1.
        3. If satisfied and final: mark plan as completed.

        Args:
            plan: The active plan to advance.
            game_state: Current game state dict.
            team: The agent's team.
            position: The agent's position.

        Returns:
            The updated plan (same instance, mutated in place).
        """
        if plan.completed:
            return plan

        if plan.current_index >= len(plan.sub_goals):
            plan.completed = True
            return plan

        current_sub_goal = plan.sub_goals[plan.current_index]

        try:
            condition_met = current_sub_goal.target_condition(
                game_state, team, position
            )
        except Exception:
            # If condition evaluation raises, treat as unsatisfied
            condition_met = False

        if condition_met:
            if plan.current_index >= len(plan.sub_goals) - 1:
                # Final sub-goal satisfied — plan is complete
                plan.completed = True
            else:
                # Advance to next sub-goal
                plan.current_index += 1

        return plan

    def should_abandon(
        self, plan: Plan, game_state: dict, team: str, position: str
    ) -> bool:
        """Check if the plan's high-level goal is no longer achievable.

        Abandonment conditions by plan type:
        - score_goal: opponent gains possession or ball leaves attacking half
        - defend_goal: team regains possession
        - intercept_ball: another player gains possession or player is no longer nearest
        - distribute_ball: goalkeeper no longer has the ball

        Args:
            plan: The active plan to check.
            game_state: Current game state dict.
            team: The agent's team.
            position: The agent's position.

        Returns:
            True if the plan should be abandoned, False otherwise.
        """
        if plan.completed:
            return False

        if plan.name == "score_goal":
            # Abandon if team lost possession or ball left attacking half
            opponent = _opponent_team(team)
            if _team_has_possession(game_state, opponent):
                return True
            if not _ball_in_attacking_half(game_state, team):
                return True

        elif plan.name == "defend_goal":
            # Abandon if team regains possession
            if _team_has_possession(game_state, team):
                return True

        elif plan.name == "intercept_ball":
            # Abandon if ball is no longer loose or player is no longer nearest
            if not _ball_is_loose(game_state):
                return True
            if not _player_is_nearest_to_ball(game_state, team, position):
                return True

        elif plan.name == "distribute_ball":
            # Abandon if goalkeeper no longer has the ball
            if not _player_in_kick_range(game_state, team, position):
                return True

        return False

    def _find_best_template(
        self, game_state: dict, team: str, position: str
    ) -> PlanTemplate | None:
        """Find the highest-priority matching template for the given position."""
        best: PlanTemplate | None = None
        best_priority = -1

        for template in self._templates:
            try:
                if template.trigger_condition(game_state, team, position):
                    priority = template.priority.get(position, 0)
                    if priority > best_priority:
                        best = template
                        best_priority = priority
            except Exception:
                # If trigger evaluation raises, skip this template
                continue

        return best

    def _instantiate_plan(self, template: PlanTemplate) -> Plan:
        """Create a new Plan instance from a template."""
        # Deep copy sub-goals to avoid shared state between plan instances
        sub_goals = [
            SubGoal(
                description=sg.description,
                target_condition=sg.target_condition,
            )
            for sg in template.sub_goals
        ]
        return Plan(
            name=template.name,
            sub_goals=sub_goals,
            current_index=0,
            completed=False,
        )

    def _get_template_priority(self, template_name: str, position: str) -> int:
        """Get the priority of a template by name for the given position."""
        for template in self._templates:
            if template.name == template_name:
                return template.priority.get(position, 0)
        return 0
