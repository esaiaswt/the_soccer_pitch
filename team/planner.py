"""Planner module for the multi-agent soccer team.

Provides multi-step planning via predefined tactical templates evaluated
entirely in Python. No LLM calls are made during plan evaluation, advancement,
or abandonment checks.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 10.2, 10.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SubGoal:
    """A single sub-goal within a plan.

    Attributes
    ----------
    description : str
        Human-readable description for LLM context.
    target_condition : Callable[[dict, str, str], bool]
        Function (game_state, team, position) -> bool that returns True
        when this sub-goal is satisfied.
    """

    description: str
    target_condition: Callable[[dict, str, str], bool]


@dataclass
class Plan:
    """A multi-step plan consisting of ordered sub-goals.

    Attributes
    ----------
    name : str
        Template name (e.g., "score_goal").
    sub_goals : list[SubGoal]
        Ordered sequence of sub-goals, max 5.
    current_index : int
        Index of the currently active sub-goal.
    completed : bool
        True when all sub-goals have been satisfied.
    """

    name: str
    sub_goals: list[SubGoal] = field(default_factory=list)
    current_index: int = 0
    completed: bool = False


@dataclass
class PlanTemplate:
    """A template for generating plans based on game state conditions.

    Attributes
    ----------
    name : str
        Template identifier (e.g., "score_goal").
    trigger_condition : Callable[[dict, str, str], bool]
        Function (game_state, team, position) -> bool that returns True
        when this template should be activated.
    priority : dict[str, int]
        Position -> priority mapping. Higher values are preferred.
    sub_goals : list[SubGoal]
        The sub-goals to instantiate when this template is selected.
    """

    name: str
    trigger_condition: Callable[[dict, str, str], bool]
    priority: dict[str, int] = field(default_factory=dict)
    sub_goals: list[SubGoal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions for game state queries
# ---------------------------------------------------------------------------


def _get_ball_position(game_state: dict) -> tuple[float, float]:
    """Extract ball position from game state."""
    ball = game_state.get("ball", {})
    return ball.get("x", 600.0), ball.get("y", 400.0)


def _get_player_position(game_state: dict, team: str, position: str) -> tuple[float, float]:
    """Extract a specific player's position from game state."""
    players = game_state.get("players", {})
    key = f"{team}_{position}"
    player = players.get(key, {})
    return player.get("x", 600.0), player.get("y", 400.0)


def _team_has_possession(game_state: dict, team: str) -> bool:
    """Check if the specified team has ball possession.

    Possession is determined by checking if any player on the team
    is within kick range (30px) of the ball.
    """
    ball_x, ball_y = _get_ball_position(game_state)
    players = game_state.get("players", {})

    for player_name, pos in players.items():
        if player_name.startswith(f"{team}_"):
            px = pos.get("x", 0.0)
            py = pos.get("y", 0.0)
            dist = ((px - ball_x) ** 2 + (py - ball_y) ** 2) ** 0.5
            if dist <= 30:
                return True
    return False


def _opponent_has_possession(game_state: dict, team: str) -> bool:
    """Check if the opponent team has ball possession."""
    opponent = "Blue" if team == "Red" else "Red"
    return _team_has_possession(game_state, opponent)


def _ball_in_attacking_half(game_state: dict, team: str) -> bool:
    """Check if the ball is in the team's attacking half.

    Red attacks right (x > 600), Blue attacks left (x < 600).
    """
    ball_x, _ = _get_ball_position(game_state)
    if team == "Red":
        return ball_x > 600
    else:
        return ball_x < 600


def _ball_in_own_half(game_state: dict, team: str) -> bool:
    """Check if the ball is in the team's own (defensive) half.

    Red's own half is x < 600, Blue's own half is x > 600.
    """
    ball_x, _ = _get_ball_position(game_state)
    if team == "Red":
        return ball_x < 600
    else:
        return ball_x > 600


def _ball_is_loose(game_state: dict) -> bool:
    """Check if the ball is not possessed by either team."""
    return not _team_has_possession(game_state, "Red") and not _team_has_possession(
        game_state, "Blue"
    )


def _player_is_nearest_to_ball(game_state: dict, team: str, position: str) -> bool:
    """Check if the specified player is the nearest on their team to the ball."""
    ball_x, ball_y = _get_ball_position(game_state)
    players = game_state.get("players", {})
    my_key = f"{team}_{position}"
    my_pos = players.get(my_key, {})
    my_x = my_pos.get("x", 600.0)
    my_y = my_pos.get("y", 400.0)
    my_dist = ((my_x - ball_x) ** 2 + (my_y - ball_y) ** 2) ** 0.5

    for player_name, pos in players.items():
        if player_name.startswith(f"{team}_") and player_name != my_key:
            px = pos.get("x", 0.0)
            py = pos.get("y", 0.0)
            dist = ((px - ball_x) ** 2 + (py - ball_y) ** 2) ** 0.5
            if dist < my_dist:
                return False
    return True


def _is_goalkeeper(position: str) -> bool:
    """Check if the position is Goalkeeper."""
    return position == "Goalkeeper"


def _goalkeeper_has_possession(game_state: dict, team: str) -> bool:
    """Check if the team's goalkeeper has ball possession."""
    ball_x, ball_y = _get_ball_position(game_state)
    players = game_state.get("players", {})
    gk_key = f"{team}_Goalkeeper"
    gk_pos = players.get(gk_key, {})
    gk_x = gk_pos.get("x", 600.0)
    gk_y = gk_pos.get("y", 400.0)
    dist = ((gk_x - ball_x) ** 2 + (gk_y - ball_y) ** 2) ** 0.5
    return dist <= 30


# ---------------------------------------------------------------------------
# Sub-goal target conditions
# ---------------------------------------------------------------------------


def _is_behind_ball(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is positioned behind the ball (between ball and own goal)."""
    ball_x, _ = _get_ball_position(game_state)
    player_x, _ = _get_player_position(game_state, team, position)
    if team == "Red":
        return player_x < ball_x
    else:
        return player_x > ball_x


def _is_near_ball(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is within approach distance of the ball (< 50px)."""
    ball_x, ball_y = _get_ball_position(game_state)
    player_x, player_y = _get_player_position(game_state, team, position)
    dist = ((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2) ** 0.5
    return dist < 50


def _is_in_kick_range(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is within kick range of the ball (<= 30px)."""
    ball_x, ball_y = _get_ball_position(game_state)
    player_x, player_y = _get_player_position(game_state, team, position)
    dist = ((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2) ** 0.5
    return dist <= 30


def _is_near_own_goal(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is near their own goal (within 200px)."""
    player_x, player_y = _get_player_position(game_state, team, position)
    if team == "Red":
        goal_x, goal_y = 0.0, 400.0
    else:
        goal_x, goal_y = 1200.0, 400.0
    dist = ((player_x - goal_x) ** 2 + (player_y - goal_y) ** 2) ** 0.5
    return dist < 200


def _is_on_intercept_path(game_state: dict, team: str, position: str) -> bool:
    """Check if the player is between the ball and their own goal."""
    ball_x, ball_y = _get_ball_position(game_state)
    player_x, player_y = _get_player_position(game_state, team, position)
    if team == "Red":
        # Player should be between ball and left goal (x=0)
        return player_x < ball_x
    else:
        # Player should be between ball and right goal (x=1200)
        return player_x > ball_x


def _has_possession(game_state: dict, team: str, position: str) -> bool:
    """Check if this specific player has possession (within kick range)."""
    return _is_in_kick_range(game_state, team, position)


def _nearest_teammate_identified(game_state: dict, team: str, position: str) -> bool:
    """Check if there's a teammate visible to pass to (always true if teammates exist)."""
    players = game_state.get("players", {})
    my_key = f"{team}_{position}"
    for player_name in players:
        if player_name.startswith(f"{team}_") and player_name != my_key:
            return True
    return False


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------


def _build_score_goal_template() -> PlanTemplate:
    """Build the score_goal plan template.

    Trigger: team has possession + ball in attacking half.
    Sub-goals: position behind ball → approach ball → kick toward goal.
    """
    return PlanTemplate(
        name="score_goal",
        trigger_condition=lambda gs, team, pos: (
            _team_has_possession(gs, team) and _ball_in_attacking_half(gs, team)
        ),
        priority={
            "Striker": 10,
            "Midfielder": 7,
            "Defender": 3,
            "Goalkeeper": 1,
        },
        sub_goals=[
            SubGoal(
                description="Position behind the ball for a shot",
                target_condition=_is_behind_ball,
            ),
            SubGoal(
                description="Approach the ball to get within kick range",
                target_condition=_is_near_ball,
            ),
            SubGoal(
                description="Kick the ball toward the opponent's goal",
                target_condition=_is_in_kick_range,
            ),
        ],
    )


def _build_defend_goal_template() -> PlanTemplate:
    """Build the defend_goal plan template.

    Trigger: opponent has possession + ball in own half.
    Sub-goals: move toward own goal → intercept ball path.
    """
    return PlanTemplate(
        name="defend_goal",
        trigger_condition=lambda gs, team, pos: (
            _opponent_has_possession(gs, team) and _ball_in_own_half(gs, team)
        ),
        priority={
            "Goalkeeper": 10,
            "Defender": 9,
            "Midfielder": 5,
            "Striker": 2,
        },
        sub_goals=[
            SubGoal(
                description="Move toward own goal to defend",
                target_condition=_is_near_own_goal,
            ),
            SubGoal(
                description="Intercept the ball path",
                target_condition=_is_on_intercept_path,
            ),
        ],
    )


def _build_intercept_ball_template() -> PlanTemplate:
    """Build the intercept_ball plan template.

    Trigger: ball is loose + player is nearest on team.
    Sub-goals: move to ball predicted position → gain possession.
    """
    return PlanTemplate(
        name="intercept_ball",
        trigger_condition=lambda gs, team, pos: (
            _ball_is_loose(gs) and _player_is_nearest_to_ball(gs, team, pos)
        ),
        priority={
            "Midfielder": 8,
            "Striker": 7,
            "Defender": 6,
            "Goalkeeper": 4,
        },
        sub_goals=[
            SubGoal(
                description="Move to the ball's predicted position",
                target_condition=_is_near_ball,
            ),
            SubGoal(
                description="Gain possession of the ball",
                target_condition=_has_possession,
            ),
        ],
    )


def _build_distribute_ball_template() -> PlanTemplate:
    """Build the distribute_ball plan template.

    Trigger: goalkeeper has possession.
    Sub-goals: identify nearest teammate → kick toward teammate.
    """
    return PlanTemplate(
        name="distribute_ball",
        trigger_condition=lambda gs, team, pos: (
            _is_goalkeeper(pos) and _goalkeeper_has_possession(gs, team)
        ),
        priority={
            "Goalkeeper": 10,
            "Defender": 0,
            "Midfielder": 0,
            "Striker": 0,
        },
        sub_goals=[
            SubGoal(
                description="Identify the nearest teammate for distribution",
                target_condition=_nearest_teammate_identified,
            ),
            SubGoal(
                description="Kick the ball toward the nearest teammate",
                target_condition=_is_in_kick_range,
            ),
        ],
    )


def build_default_templates() -> list[PlanTemplate]:
    """Build the default template library with all required plan templates.

    Returns
    -------
    list[PlanTemplate]
        The list of default plan templates: score_goal, defend_goal,
        intercept_ball, distribute_ball.
    """
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
    """Multi-step planner that selects and manages plans from templates.

    All logic executes in Python without LLM calls.

    Parameters
    ----------
    templates : list[PlanTemplate]
        The plan template library to select from.
    """

    def __init__(self, templates: list[PlanTemplate]) -> None:
        self._templates = templates

    def evaluate(
        self,
        game_state: dict,
        team: str,
        position: str,
        active_plan: Plan | None,
    ) -> Plan | None:
        """Evaluate the current game state and return the appropriate plan.

        1. If no active plan, find matching templates and select highest priority
           for position.
        2. If active plan exists, check if a higher-priority template now matches
           and replace if so.
        3. Return the plan (new, replaced, or existing).

        Parameters
        ----------
        game_state : dict
            The current game state snapshot.
        team : str
            The team color ("Red" or "Blue").
        position : str
            The player's position.
        active_plan : Plan | None
            The currently active plan, or None.

        Returns
        -------
        Plan | None
            The selected plan, or None if no template matches.
        """
        # Find all matching templates
        matching = self._find_matching_templates(game_state, team, position)

        if not matching:
            # No templates match — keep active plan if it exists
            return active_plan

        # Select the highest priority template for this position
        best_template = self._select_highest_priority(matching, position)

        if active_plan is None:
            # No active plan — instantiate from best matching template
            return self._instantiate_plan(best_template)

        # Active plan exists — check if a higher-priority template should replace it
        active_template_priority = self._get_template_priority(
            active_plan.name, position
        )
        best_priority = best_template.priority.get(position, 0)

        if best_priority > active_template_priority and best_template.name != active_plan.name:
            # Higher priority template matches — replace
            return self._instantiate_plan(best_template)

        # Keep existing plan
        return active_plan

    def advance(
        self, plan: Plan, game_state: dict, team: str, position: str
    ) -> Plan:
        """Advance the plan if the current sub-goal's target condition is satisfied.

        1. Check if current sub-goal's target_condition is satisfied.
        2. If satisfied and not final: advance current_index by 1.
        3. If satisfied and final: mark plan as completed.

        Parameters
        ----------
        plan : Plan
            The active plan to advance.
        game_state : dict
            The current game state snapshot.
        team : str
            The team color.
        position : str
            The player's position.

        Returns
        -------
        Plan
            The (possibly advanced or completed) plan.
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
            # If condition evaluation fails, treat as not satisfied
            condition_met = False

        if condition_met:
            if plan.current_index >= len(plan.sub_goals) - 1:
                # Final sub-goal satisfied — mark completed
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
        - defend_goal: team gains possession and ball is no longer in own half
        - intercept_ball: ball is no longer loose (someone gained possession)
        - distribute_ball: goalkeeper no longer has possession

        Parameters
        ----------
        plan : Plan
            The active plan to check.
        game_state : dict
            The current game state snapshot.
        team : str
            The team color.
        position : str
            The player's position.

        Returns
        -------
        bool
            True if the plan should be abandoned.
        """
        if plan.completed:
            return False

        if plan.name == "score_goal":
            # Abandon if opponent gains possession
            return _opponent_has_possession(game_state, team)

        elif plan.name == "defend_goal":
            # Abandon if team gains possession and ball is out of own half
            return _team_has_possession(game_state, team) and not _ball_in_own_half(
                game_state, team
            )

        elif plan.name == "intercept_ball":
            # Abandon if ball is no longer loose
            return not _ball_is_loose(game_state)

        elif plan.name == "distribute_ball":
            # Abandon if goalkeeper no longer has possession
            return not _goalkeeper_has_possession(game_state, team)

        return False

    def _find_matching_templates(
        self, game_state: dict, team: str, position: str
    ) -> list[PlanTemplate]:
        """Find all templates whose trigger condition is satisfied."""
        matching = []
        for template in self._templates:
            try:
                if template.trigger_condition(game_state, team, position):
                    matching.append(template)
            except Exception:
                # If trigger evaluation fails, skip this template
                continue
        return matching

    def _select_highest_priority(
        self, templates: list[PlanTemplate], position: str
    ) -> PlanTemplate:
        """Select the template with the highest priority for the given position."""
        return max(templates, key=lambda t: t.priority.get(position, 0))

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
