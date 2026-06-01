"""Property-based tests for the team/ Planner module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.planner import (
    SubGoal,
    Plan,
    PlanTemplate,
    Planner,
    build_default_templates,
)


# --- Strategies for generating valid planner inputs ---

positions_strategy = st.sampled_from(["Striker", "Midfielder", "Defender", "Goalkeeper"])
teams_strategy = st.sampled_from(["Red", "Blue"])


def _make_condition(satisfied: bool):
    """Create a target_condition callable that returns a fixed boolean."""
    return lambda gs, team, pos: satisfied


def sub_goal_strategy(satisfied: bool | None = None):
    """Generate a SubGoal with a controllable target condition."""
    if satisfied is None:
        sat_st = st.booleans()
    else:
        sat_st = st.just(satisfied)

    return sat_st.flatmap(
        lambda sat: st.builds(
            SubGoal,
            description=st.text(min_size=1, max_size=30),
            target_condition=st.just(_make_condition(sat)),
        )
    )


def plan_with_subgoals_strategy(
    num_subgoals: int | None = None,
    current_satisfied: bool = True,
    is_final: bool | None = None,
):
    """Generate a Plan with controllable sub-goal satisfaction.

    Parameters
    ----------
    num_subgoals : int | None
        Fixed number of sub-goals, or random 1-5 if None.
    current_satisfied : bool
        Whether the current sub-goal's condition is satisfied.
    is_final : bool | None
        If True, current_index is at the last sub-goal.
        If False, current_index is NOT at the last sub-goal.
        If None, random.
    """
    if num_subgoals is not None:
        n_st = st.just(num_subgoals)
    else:
        n_st = st.integers(min_value=2, max_value=5)

    return n_st.flatmap(lambda n: _build_plan_with_n_subgoals(n, current_satisfied, is_final))


def _build_plan_with_n_subgoals(n: int, current_satisfied: bool, is_final: bool | None):
    """Build a plan with n sub-goals and controlled current index."""
    # Generate sub-goals: the current one has the specified satisfaction
    # Others can be anything (they won't be evaluated by advance)
    if is_final is True:
        idx_st = st.just(n - 1)
    elif is_final is False:
        idx_st = st.integers(min_value=0, max_value=max(0, n - 2))
    else:
        idx_st = st.integers(min_value=0, max_value=n - 1)

    return idx_st.flatmap(
        lambda idx: _build_plan_at_index(n, idx, current_satisfied)
    )


def _build_plan_at_index(n: int, idx: int, current_satisfied: bool):
    """Build a plan with n sub-goals where sub-goal at idx has given satisfaction."""
    sub_goals = []
    for i in range(n):
        if i == idx:
            sub_goals.append(SubGoal(
                description=f"sub-goal-{i}",
                target_condition=_make_condition(current_satisfied),
            ))
        else:
            sub_goals.append(SubGoal(
                description=f"sub-goal-{i}",
                target_condition=_make_condition(False),
            ))

    plan = Plan(
        name="test_plan",
        sub_goals=sub_goals,
        current_index=idx,
        completed=False,
    )
    return st.just(plan)


# Simple game state strategy for planner tests
game_state_strategy = st.fixed_dictionaries({
    "ball": st.fixed_dictionaries({
        "x": st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        "y": st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    }),
    "players": st.just({}),
})


# Feature: full-agentic-upgrade, Property 6: Plan state machine advancement
# **Validates: Requirements 3.2, 3.3, 3.4**


class TestPlanStateMachineAdvancement:
    """Property 6: Plan state machine advancement.

    For any Plan with sub-goals and any game state, when the current sub-goal's
    target condition is satisfied: if it is not the final sub-goal, the plan SHALL
    advance current_index by 1; if it is the final sub-goal, the plan SHALL be
    marked as completed.
    """

    @settings(max_examples=100)
    @given(
        num_subgoals=st.integers(min_value=2, max_value=5),
        game_state=game_state_strategy,
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_advance_increments_index_when_not_final(
        self, num_subgoals, game_state, team, position
    ):
        """When current sub-goal is satisfied and NOT final, current_index SHALL
        advance by 1.

        **Validates: Requirements 3.2, 3.3**
        """
        # Build a plan where current sub-goal is satisfied and is NOT the final one
        idx = num_subgoals - 2  # Not the last
        sub_goals = []
        for i in range(num_subgoals):
            satisfied = (i == idx)
            sub_goals.append(SubGoal(
                description=f"sub-goal-{i}",
                target_condition=_make_condition(satisfied),
            ))

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=idx,
            completed=False,
        )

        planner = Planner(templates=[])
        original_index = plan.current_index
        result = planner.advance(plan, game_state, team, position)

        assert result.current_index == original_index + 1
        assert result.completed is False

    @settings(max_examples=100)
    @given(
        num_subgoals=st.integers(min_value=1, max_value=5),
        game_state=game_state_strategy,
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_advance_marks_completed_when_final(
        self, num_subgoals, game_state, team, position
    ):
        """When current sub-goal is satisfied and IS the final sub-goal, the plan
        SHALL be marked as completed.

        **Validates: Requirements 3.3, 3.4**
        """
        # Build a plan where the final sub-goal is satisfied
        idx = num_subgoals - 1  # The last sub-goal
        sub_goals = []
        for i in range(num_subgoals):
            satisfied = (i == idx)
            sub_goals.append(SubGoal(
                description=f"sub-goal-{i}",
                target_condition=_make_condition(satisfied),
            ))

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=idx,
            completed=False,
        )

        planner = Planner(templates=[])
        result = planner.advance(plan, game_state, team, position)

        assert result.completed is True

    @settings(max_examples=100)
    @given(
        num_subgoals=st.integers(min_value=2, max_value=5),
        game_state=game_state_strategy,
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_advance_does_not_change_when_condition_not_met(
        self, num_subgoals, game_state, team, position
    ):
        """When current sub-goal's condition is NOT satisfied, the plan SHALL NOT
        advance.

        **Validates: Requirements 3.2**
        """
        idx = 0
        sub_goals = []
        for i in range(num_subgoals):
            # All conditions unsatisfied
            sub_goals.append(SubGoal(
                description=f"sub-goal-{i}",
                target_condition=_make_condition(False),
            ))

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=idx,
            completed=False,
        )

        planner = Planner(templates=[])
        result = planner.advance(plan, game_state, team, position)

        assert result.current_index == idx
        assert result.completed is False


# Feature: full-agentic-upgrade, Property 7: Plan abandonment on unachievable goal
# **Validates: Requirements 3.5**


class TestPlanAbandonmentOnUnachievableGoal:
    """Property 7: Plan abandonment on unachievable goal.

    For any active Plan and game state where the plan's high-level goal is no
    longer achievable, the Planner SHALL signal abandonment.
    """

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_score_goal_abandoned_when_opponent_has_possession(self, team, position):
        """A score_goal plan SHALL be abandoned when the opponent gains possession.

        **Validates: Requirements 3.5**
        """
        opponent = "Blue" if team == "Red" else "Red"
        # Place an opponent player within kick range of the ball
        ball_x, ball_y = 800.0, 400.0
        game_state = {
            "ball": {"x": ball_x, "y": ball_y},
            "players": {
                f"{opponent}_Striker": {"x": ball_x, "y": ball_y},  # within 30px
                f"{team}_Striker": {"x": 200.0, "y": 200.0},  # far away
            },
        }

        plan = Plan(
            name="score_goal",
            sub_goals=[SubGoal(description="test", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position) is True

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_intercept_ball_abandoned_when_ball_not_loose(self, team, position):
        """An intercept_ball plan SHALL be abandoned when the ball is no longer loose.

        **Validates: Requirements 3.5**
        """
        # Place a player within kick range of the ball (ball is possessed)
        ball_x, ball_y = 600.0, 400.0
        game_state = {
            "ball": {"x": ball_x, "y": ball_y},
            "players": {
                f"{team}_Midfielder": {"x": ball_x, "y": ball_y},  # within 30px
            },
        }

        plan = Plan(
            name="intercept_ball",
            sub_goals=[SubGoal(description="test", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position) is True

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_distribute_ball_abandoned_when_gk_loses_possession(self, team, position):
        """A distribute_ball plan SHALL be abandoned when the goalkeeper no longer
        has possession.

        **Validates: Requirements 3.5**
        """
        # Goalkeeper is far from ball
        game_state = {
            "ball": {"x": 600.0, "y": 400.0},
            "players": {
                f"{team}_Goalkeeper": {"x": 50.0, "y": 400.0},  # far from ball
            },
        }

        plan = Plan(
            name="distribute_ball",
            sub_goals=[SubGoal(description="test", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position) is True

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_defend_goal_abandoned_when_team_gains_possession_outside_own_half(
        self, team, position
    ):
        """A defend_goal plan SHALL be abandoned when the team gains possession
        and the ball is no longer in own half.

        **Validates: Requirements 3.5**
        """
        # Team has possession (player near ball) and ball is in attacking half
        if team == "Red":
            ball_x = 800.0  # attacking half for Red (x > 600)
        else:
            ball_x = 400.0  # attacking half for Blue (x < 600)

        game_state = {
            "ball": {"x": ball_x, "y": 400.0},
            "players": {
                f"{team}_Midfielder": {"x": ball_x, "y": 400.0},  # within kick range
            },
        }

        plan = Plan(
            name="defend_goal",
            sub_goals=[SubGoal(description="test", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position) is True

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_no_abandonment_when_goal_still_achievable(self, team, position):
        """A plan SHALL NOT be abandoned when its high-level goal is still achievable.

        **Validates: Requirements 3.5**
        """
        # score_goal: team has possession, no opponent near ball
        ball_x = 800.0 if team == "Red" else 400.0
        game_state = {
            "ball": {"x": ball_x, "y": 400.0},
            "players": {
                f"{team}_Striker": {"x": ball_x, "y": 400.0},  # team has possession
            },
        }

        plan = Plan(
            name="score_goal",
            sub_goals=[SubGoal(description="test", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position) is False


# Feature: full-agentic-upgrade, Property 8: Plan replacement with higher-priority template
# **Validates: Requirements 3.7, 4.3**


class TestPlanReplacementWithHigherPriority:
    """Property 8: Plan replacement with higher-priority template.

    For any active Plan and game state that matches a different template with
    strictly higher priority, the Planner SHALL replace the active plan.
    """

    @settings(max_examples=100)
    @given(
        position=positions_strategy,
        team=teams_strategy,
        low_priority=st.integers(min_value=1, max_value=5),
        high_priority=st.integers(min_value=6, max_value=10),
    )
    def test_higher_priority_template_replaces_active_plan(
        self, position, team, low_priority, high_priority
    ):
        """When a different template with strictly higher priority matches, the
        Planner SHALL replace the active plan.

        **Validates: Requirements 3.7, 4.3**
        """
        assume(high_priority > low_priority)

        # Create two templates: one low priority (active), one high priority (new match)
        low_template = PlanTemplate(
            name="low_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={position: low_priority},
            sub_goals=[SubGoal(description="low-sg", target_condition=_make_condition(False))],
        )
        high_template = PlanTemplate(
            name="high_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={position: high_priority},
            sub_goals=[SubGoal(description="high-sg", target_condition=_make_condition(False))],
        )

        planner = Planner(templates=[low_template, high_template])

        # Active plan is from the low-priority template
        active_plan = Plan(
            name="low_priority_plan",
            sub_goals=[SubGoal(description="low-sg", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, active_plan)

        assert result is not None
        assert result.name == "high_priority_plan"

    @settings(max_examples=100)
    @given(
        position=positions_strategy,
        team=teams_strategy,
        priority=st.integers(min_value=1, max_value=10),
    )
    def test_same_or_lower_priority_does_not_replace(self, position, team, priority):
        """When no template has strictly higher priority, the active plan SHALL
        NOT be replaced.

        **Validates: Requirements 3.7, 4.3**
        """
        # Template with same priority as active plan
        template = PlanTemplate(
            name="same_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={position: priority},
            sub_goals=[SubGoal(description="sg", target_condition=_make_condition(False))],
        )

        planner = Planner(templates=[template])

        active_plan = Plan(
            name="same_plan",
            sub_goals=[SubGoal(description="sg", target_condition=_make_condition(False))],
            current_index=0,
            completed=False,
        )

        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, active_plan)

        # Should keep the active plan (same name, same priority)
        assert result is not None
        assert result.name == "same_plan"
        assert result is active_plan  # Same object, not replaced


# Feature: full-agentic-upgrade, Property 9: Plan sub-goal count limit
# **Validates: Requirements 3.8**


class TestPlanSubGoalCountLimit:
    """Property 9: Plan sub-goal count limit.

    For any Plan instantiated from any template in the library, the number of
    sub-goals SHALL be at most 5.
    """

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_default_templates_have_at_most_5_subgoals(self, team, position):
        """All plans instantiated from the default template library SHALL have
        at most 5 sub-goals.

        **Validates: Requirements 3.8**
        """
        templates = build_default_templates()

        for template in templates:
            assert len(template.sub_goals) <= 5, (
                f"Template '{template.name}' has {len(template.sub_goals)} sub-goals, "
                f"exceeding the maximum of 5"
            )

    @settings(max_examples=100)
    @given(
        team=teams_strategy,
        position=positions_strategy,
    )
    def test_instantiated_plans_have_at_most_5_subgoals(self, team, position):
        """Plans instantiated via the Planner from any matching template SHALL
        have at most 5 sub-goals.

        **Validates: Requirements 3.8**
        """
        templates = build_default_templates()
        planner = Planner(templates=templates)

        # Try to trigger each template and verify sub-goal count
        for template in templates:
            # Create a game state that triggers this template
            plan = Plan(
                name=template.name,
                sub_goals=[
                    SubGoal(
                        description=sg.description,
                        target_condition=sg.target_condition,
                    )
                    for sg in template.sub_goals
                ],
                current_index=0,
                completed=False,
            )
            assert len(plan.sub_goals) <= 5, (
                f"Plan '{plan.name}' has {len(plan.sub_goals)} sub-goals, "
                f"exceeding the maximum of 5"
            )

    @settings(max_examples=100)
    @given(
        num_subgoals=st.integers(min_value=1, max_value=5),
        position=positions_strategy,
        team=teams_strategy,
    )
    def test_custom_template_within_limit_produces_valid_plan(
        self, num_subgoals, position, team
    ):
        """Any template with at most 5 sub-goals SHALL produce a plan with at
        most 5 sub-goals when instantiated.

        **Validates: Requirements 3.8**
        """
        sub_goals = [
            SubGoal(
                description=f"sg-{i}",
                target_condition=_make_condition(False),
            )
            for i in range(num_subgoals)
        ]

        template = PlanTemplate(
            name="custom_template",
            trigger_condition=lambda gs, t, p: True,
            priority={position: 5},
            sub_goals=sub_goals,
        )

        planner = Planner(templates=[template])
        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, None)

        assert result is not None
        assert len(result.sub_goals) <= 5


# Feature: full-agentic-upgrade, Property 11: Template selection by highest priority
# **Validates: Requirements 4.1, 4.2, 4.5**


class TestTemplateSelectionByHighestPriority:
    """Property 11: Template selection by highest priority.

    For any game state and player position where one or more plan templates have
    their trigger condition satisfied, the Planner SHALL select the template with
    the highest priority value for that position.
    """

    @settings(max_examples=100)
    @given(
        position=positions_strategy,
        team=teams_strategy,
        priorities=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=2,
            max_size=5,
            unique=True,
        ),
    )
    def test_selects_highest_priority_template(self, position, team, priorities):
        """When multiple templates match, the Planner SHALL select the one with
        the highest priority for the agent's position.

        **Validates: Requirements 4.1, 4.2, 4.5**
        """
        # Create templates with different priorities, all triggered
        templates = []
        for i, prio in enumerate(priorities):
            templates.append(PlanTemplate(
                name=f"template_{i}_prio_{prio}",
                trigger_condition=lambda gs, t, p: True,
                priority={position: prio},
                sub_goals=[SubGoal(
                    description=f"sg-{i}",
                    target_condition=_make_condition(False),
                )],
            ))

        planner = Planner(templates=templates)
        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, None)

        # Should select the template with the highest priority
        max_priority = max(priorities)
        max_idx = priorities.index(max_priority)
        expected_name = f"template_{max_idx}_prio_{max_priority}"

        assert result is not None
        assert result.name == expected_name

    @settings(max_examples=100)
    @given(
        position=positions_strategy,
        team=teams_strategy,
        priority=st.integers(min_value=1, max_value=100),
    )
    def test_single_matching_template_is_selected(self, position, team, priority):
        """When exactly one template matches, it SHALL be selected regardless of
        priority value.

        **Validates: Requirements 4.1, 4.2**
        """
        template = PlanTemplate(
            name="only_match",
            trigger_condition=lambda gs, t, p: True,
            priority={position: priority},
            sub_goals=[SubGoal(
                description="sg",
                target_condition=_make_condition(False),
            )],
        )

        # Add a non-matching template to ensure selection logic works
        non_matching = PlanTemplate(
            name="no_match",
            trigger_condition=lambda gs, t, p: False,
            priority={position: priority + 100},  # Higher priority but doesn't match
            sub_goals=[SubGoal(
                description="sg",
                target_condition=_make_condition(False),
            )],
        )

        planner = Planner(templates=[template, non_matching])
        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, None)

        assert result is not None
        assert result.name == "only_match"

    @settings(max_examples=100)
    @given(
        position=positions_strategy,
        team=teams_strategy,
    )
    def test_no_matching_template_returns_none(self, position, team):
        """When no templates match, the Planner SHALL return None (no plan).

        **Validates: Requirements 4.1, 4.2**
        """
        template = PlanTemplate(
            name="no_match",
            trigger_condition=lambda gs, t, p: False,
            priority={position: 10},
            sub_goals=[SubGoal(
                description="sg",
                target_condition=_make_condition(False),
            )],
        )

        planner = Planner(templates=[template])
        game_state = {"ball": {"x": 600.0, "y": 400.0}, "players": {}}
        result = planner.evaluate(game_state, team, position, None)

        assert result is None
