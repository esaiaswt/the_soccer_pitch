"""Property-based tests for the Planner module (Properties 6, 7, 8, 9, and 11).

# Feature: full-agentic-upgrade, Property 6: Plan state machine advancement
# Feature: full-agentic-upgrade, Property 7: Plan abandonment on unachievable goal
# Feature: full-agentic-upgrade, Property 8: Plan replacement with higher-priority template
# Feature: full-agentic-upgrade, Property 9: Plan sub-goal count limit
# Feature: full-agentic-upgrade, Property 11: Template selection by highest priority
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from planner import (
    Plan,
    PlanTemplate,
    Planner,
    SubGoal,
    get_default_templates,
    PITCH_WIDTH,
    PITCH_HEIGHT,
    POSSESSION_RANGE,
)


# --- Strategies ---

POSITIONS = ["Striker", "Midfielder", "Defender", "Goalkeeper"]
TEAMS = ["Red", "Blue"]


def position_strategy():
    """Generate a valid player position."""
    return st.sampled_from(POSITIONS)


def team_strategy():
    """Generate a valid team name."""
    return st.sampled_from(TEAMS)


def coordinate_strategy():
    """Generate a valid coordinate within the pitch."""
    return st.fixed_dictionaries(
        {
            "x": st.floats(min_value=0.0, max_value=PITCH_WIDTH, allow_nan=False, allow_infinity=False),
            "y": st.floats(min_value=0.0, max_value=PITCH_HEIGHT, allow_nan=False, allow_infinity=False),
        }
    )


def game_state_strategy():
    """Generate a valid game state with ball and players."""
    return st.fixed_dictionaries(
        {
            "ball": coordinate_strategy(),
            "players": st.fixed_dictionaries(
                {
                    "Red_Striker": coordinate_strategy(),
                    "Red_Midfielder": coordinate_strategy(),
                    "Red_Defender": coordinate_strategy(),
                    "Red_Goalkeeper": coordinate_strategy(),
                    "Blue_Striker": coordinate_strategy(),
                    "Blue_Midfielder": coordinate_strategy(),
                    "Blue_Defender": coordinate_strategy(),
                    "Blue_Goalkeeper": coordinate_strategy(),
                }
            ),
        }
    )


def sub_goal_always_true():
    """Create a sub-goal whose target condition is always satisfied."""
    return SubGoal(
        description="Always satisfied sub-goal",
        target_condition=lambda gs, t, p: True,
    )


def sub_goal_always_false():
    """Create a sub-goal whose target condition is never satisfied."""
    return SubGoal(
        description="Never satisfied sub-goal",
        target_condition=lambda gs, t, p: False,
    )


def plan_with_n_subgoals_strategy(min_goals=1, max_goals=5):
    """Generate a Plan with a configurable number of sub-goals."""
    return st.integers(min_value=min_goals, max_value=max_goals).map(
        lambda n: Plan(
            name="test_plan",
            sub_goals=[
                SubGoal(
                    description=f"Sub-goal {i}",
                    target_condition=lambda gs, t, p: False,
                )
                for i in range(n)
            ],
            current_index=0,
            completed=False,
        )
    )


# Feature: full-agentic-upgrade, Property 6: Plan state machine advancement
class TestPlanStateMachineAdvancement:
    """Property 6: Plan state machine advancement.

    For any Plan with sub-goals and any game state, when the current sub-goal's
    target condition is satisfied: if it is not the final sub-goal, the plan SHALL
    advance current_index by 1; if it is the final sub-goal, the plan SHALL be
    marked as completed.

    **Validates: Requirements 3.2, 3.3, 3.4**
    """

    @given(
        num_sub_goals=st.integers(min_value=2, max_value=5),
        current_index=st.data(),
        game_state=game_state_strategy(),
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_advance_non_final_subgoal_increments_index(
        self, num_sub_goals, current_index, game_state, team, position
    ):
        """When a non-final sub-goal is satisfied, current_index advances by 1."""
        # current_index must be a non-final position
        idx = current_index.draw(st.integers(min_value=0, max_value=num_sub_goals - 2))

        # Build plan with the current sub-goal always satisfied
        sub_goals = []
        for i in range(num_sub_goals):
            if i == idx:
                sub_goals.append(sub_goal_always_true())
            else:
                sub_goals.append(sub_goal_always_false())

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=idx,
            completed=False,
        )

        planner = Planner(templates=[])
        result = planner.advance(plan, game_state, team, position)

        assert result.current_index == idx + 1, (
            f"Expected current_index to advance from {idx} to {idx + 1}, "
            f"got {result.current_index}"
        )
        assert not result.completed, (
            "Plan should not be marked completed when advancing a non-final sub-goal."
        )

    @given(
        num_sub_goals=st.integers(min_value=1, max_value=5),
        game_state=game_state_strategy(),
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_advance_final_subgoal_marks_completed(
        self, num_sub_goals, game_state, team, position
    ):
        """When the final sub-goal is satisfied, the plan is marked as completed."""
        final_index = num_sub_goals - 1

        # Build plan with the final sub-goal always satisfied
        sub_goals = []
        for i in range(num_sub_goals):
            if i == final_index:
                sub_goals.append(sub_goal_always_true())
            else:
                sub_goals.append(sub_goal_always_false())

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=final_index,
            completed=False,
        )

        planner = Planner(templates=[])
        result = planner.advance(plan, game_state, team, position)

        assert result.completed, (
            f"Plan should be marked completed when final sub-goal (index {final_index}) "
            f"is satisfied, but completed={result.completed}"
        )

    @given(
        num_sub_goals=st.integers(min_value=1, max_value=5),
        current_index=st.data(),
        game_state=game_state_strategy(),
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_no_advance_when_condition_not_satisfied(
        self, num_sub_goals, current_index, game_state, team, position
    ):
        """When the current sub-goal's condition is NOT satisfied, plan does not advance."""
        idx = current_index.draw(st.integers(min_value=0, max_value=num_sub_goals - 1))

        # All sub-goals have unsatisfied conditions
        sub_goals = [sub_goal_always_false() for _ in range(num_sub_goals)]

        plan = Plan(
            name="test_plan",
            sub_goals=sub_goals,
            current_index=idx,
            completed=False,
        )

        planner = Planner(templates=[])
        result = planner.advance(plan, game_state, team, position)

        assert result.current_index == idx, (
            f"Expected current_index to remain at {idx}, got {result.current_index}"
        )
        assert not result.completed, (
            "Plan should not be marked completed when condition is not satisfied."
        )


# Feature: full-agentic-upgrade, Property 7: Plan abandonment on unachievable goal
class TestPlanAbandonmentOnUnachievableGoal:
    """Property 7: Plan abandonment on unachievable goal.

    For any active Plan and game state where the plan's high-level goal is no
    longer achievable, the Planner SHALL signal abandonment.

    **Validates: Requirements 3.5**
    """

    @given(
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_score_goal_abandoned_when_opponent_has_possession(self, team, position):
        """score_goal plan is abandoned when opponent gains possession."""
        opponent = "Blue" if team == "Red" else "Red"

        # Place ball near an opponent player (within POSSESSION_RANGE)
        ball_pos = {"x": 800.0, "y": 400.0}
        opponent_player_pos = {"x": ball_pos["x"] + 5.0, "y": ball_pos["y"]}

        # Build game state where opponent has possession
        game_state = {
            "ball": ball_pos,
            "players": {
                f"{team}_Striker": {"x": 200.0, "y": 200.0},
                f"{team}_Midfielder": {"x": 300.0, "y": 300.0},
                f"{team}_Defender": {"x": 100.0, "y": 400.0},
                f"{team}_Goalkeeper": {"x": 50.0, "y": 400.0},
                f"{opponent}_Striker": opponent_player_pos,
                f"{opponent}_Midfielder": {"x": 600.0, "y": 300.0},
                f"{opponent}_Defender": {"x": 900.0, "y": 400.0},
                f"{opponent}_Goalkeeper": {"x": 1150.0, "y": 400.0},
            },
        }

        plan = Plan(
            name="score_goal",
            sub_goals=[sub_goal_always_false(), sub_goal_always_false()],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position), (
            "score_goal plan should be abandoned when opponent has possession."
        )

    @given(
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_score_goal_abandoned_when_ball_leaves_attacking_half(self, team, position):
        """score_goal plan is abandoned when ball is no longer in attacking half."""
        # Place ball in own half (not attacking half)
        if team == "Red":
            ball_pos = {"x": 200.0, "y": 400.0}  # Red's own half (x < 600)
        else:
            ball_pos = {"x": 1000.0, "y": 400.0}  # Blue's own half (x > 600)

        # No one has possession (ball is far from all players)
        game_state = {
            "ball": ball_pos,
            "players": {
                "Red_Striker": {"x": 900.0, "y": 200.0},
                "Red_Midfielder": {"x": 700.0, "y": 400.0},
                "Red_Defender": {"x": 300.0, "y": 400.0},
                "Red_Goalkeeper": {"x": 50.0, "y": 400.0},
                "Blue_Striker": {"x": 300.0, "y": 200.0},
                "Blue_Midfielder": {"x": 500.0, "y": 400.0},
                "Blue_Defender": {"x": 900.0, "y": 400.0},
                "Blue_Goalkeeper": {"x": 1150.0, "y": 400.0},
            },
        }

        plan = Plan(
            name="score_goal",
            sub_goals=[sub_goal_always_false(), sub_goal_always_false()],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position), (
            "score_goal plan should be abandoned when ball leaves attacking half."
        )

    @given(
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_defend_goal_abandoned_when_team_regains_possession(self, team, position):
        """defend_goal plan is abandoned when team regains possession."""
        # Place ball near a team player (within POSSESSION_RANGE)
        ball_pos = {"x": 400.0, "y": 400.0}
        team_player_pos = {"x": ball_pos["x"] + 5.0, "y": ball_pos["y"]}

        opponent = "Blue" if team == "Red" else "Red"

        game_state = {
            "ball": ball_pos,
            "players": {
                f"{team}_Striker": team_player_pos,
                f"{team}_Midfielder": {"x": 300.0, "y": 300.0},
                f"{team}_Defender": {"x": 100.0, "y": 400.0},
                f"{team}_Goalkeeper": {"x": 50.0, "y": 400.0},
                f"{opponent}_Striker": {"x": 800.0, "y": 200.0},
                f"{opponent}_Midfielder": {"x": 600.0, "y": 300.0},
                f"{opponent}_Defender": {"x": 900.0, "y": 400.0},
                f"{opponent}_Goalkeeper": {"x": 1150.0, "y": 400.0},
            },
        }

        plan = Plan(
            name="defend_goal",
            sub_goals=[sub_goal_always_false(), sub_goal_always_false()],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position), (
            "defend_goal plan should be abandoned when team regains possession."
        )

    @given(
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_intercept_ball_abandoned_when_ball_no_longer_loose(self, team, position):
        """intercept_ball plan is abandoned when ball is no longer loose."""
        # Place ball near a player (someone has possession)
        ball_pos = {"x": 600.0, "y": 400.0}
        possessor_pos = {"x": ball_pos["x"] + 5.0, "y": ball_pos["y"]}

        opponent = "Blue" if team == "Red" else "Red"

        game_state = {
            "ball": ball_pos,
            "players": {
                f"{team}_Striker": {"x": 200.0, "y": 200.0},
                f"{team}_Midfielder": {"x": 300.0, "y": 300.0},
                f"{team}_Defender": {"x": 100.0, "y": 400.0},
                f"{team}_Goalkeeper": {"x": 50.0, "y": 400.0},
                f"{opponent}_Striker": possessor_pos,
                f"{opponent}_Midfielder": {"x": 800.0, "y": 300.0},
                f"{opponent}_Defender": {"x": 900.0, "y": 400.0},
                f"{opponent}_Goalkeeper": {"x": 1150.0, "y": 400.0},
            },
        }

        plan = Plan(
            name="intercept_ball",
            sub_goals=[sub_goal_always_false(), sub_goal_always_false()],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, position), (
            "intercept_ball plan should be abandoned when ball is no longer loose."
        )

    @given(
        team=team_strategy(),
    )
    @settings(max_examples=100)
    def test_distribute_ball_abandoned_when_goalkeeper_loses_ball(self, team):
        """distribute_ball plan is abandoned when goalkeeper no longer has the ball."""
        # Ball is far from goalkeeper
        ball_pos = {"x": 600.0, "y": 400.0}

        opponent = "Blue" if team == "Red" else "Red"

        game_state = {
            "ball": ball_pos,
            "players": {
                f"{team}_Striker": {"x": 800.0, "y": 200.0},
                f"{team}_Midfielder": {"x": 500.0, "y": 300.0},
                f"{team}_Defender": {"x": 200.0, "y": 400.0},
                f"{team}_Goalkeeper": {"x": 50.0, "y": 400.0},
                f"{opponent}_Striker": {"x": 900.0, "y": 200.0},
                f"{opponent}_Midfielder": {"x": 700.0, "y": 300.0},
                f"{opponent}_Defender": {"x": 1000.0, "y": 400.0},
                f"{opponent}_Goalkeeper": {"x": 1150.0, "y": 400.0},
            },
        }

        plan = Plan(
            name="distribute_ball",
            sub_goals=[sub_goal_always_false(), sub_goal_always_false()],
            current_index=0,
            completed=False,
        )

        planner = Planner(templates=[])
        assert planner.should_abandon(plan, game_state, team, "Goalkeeper"), (
            "distribute_ball plan should be abandoned when goalkeeper loses the ball."
        )


# Feature: full-agentic-upgrade, Property 8: Plan replacement with higher-priority template
class TestPlanReplacementWithHigherPriority:
    """Property 8: Plan replacement with higher-priority template.

    For any active Plan and game state that matches a different template with
    strictly higher priority, the Planner SHALL replace the active plan.

    **Validates: Requirements 3.7, 4.3**
    """

    @given(
        position=position_strategy(),
        team=team_strategy(),
        game_state=game_state_strategy(),
    )
    @settings(max_examples=100)
    def test_higher_priority_template_replaces_active_plan(
        self, position, team, game_state
    ):
        """A matching template with higher priority replaces the active plan."""
        low_priority = 3
        high_priority = 10

        # Template A: low priority, always triggers
        template_a = PlanTemplate(
            name="low_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={pos: low_priority for pos in POSITIONS},
            sub_goals=[
                SubGoal(description="Low priority goal", target_condition=lambda gs, t, p: False)
            ],
        )

        # Template B: high priority, always triggers
        template_b = PlanTemplate(
            name="high_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={pos: high_priority for pos in POSITIONS},
            sub_goals=[
                SubGoal(description="High priority goal", target_condition=lambda gs, t, p: False)
            ],
        )

        planner = Planner(templates=[template_a, template_b])

        # Active plan is from template_a (low priority)
        active_plan = Plan(
            name="low_priority_plan",
            sub_goals=[SubGoal(description="Low priority goal", target_condition=lambda gs, t, p: False)],
            current_index=0,
            completed=False,
        )

        result = planner.evaluate(game_state, team, position, active_plan)

        assert result is not None, "Planner should return a plan."
        assert result.name == "high_priority_plan", (
            f"Expected plan to be replaced with 'high_priority_plan', got '{result.name}'"
        )

    @given(
        position=position_strategy(),
        team=team_strategy(),
        game_state=game_state_strategy(),
    )
    @settings(max_examples=100)
    def test_same_or_lower_priority_does_not_replace(
        self, position, team, game_state
    ):
        """A matching template with same or lower priority does NOT replace the active plan."""
        high_priority = 10
        low_priority = 3

        # Template A: high priority, always triggers (this is the active plan)
        template_a = PlanTemplate(
            name="high_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={pos: high_priority for pos in POSITIONS},
            sub_goals=[
                SubGoal(description="High priority goal", target_condition=lambda gs, t, p: False)
            ],
        )

        # Template B: low priority, always triggers
        template_b = PlanTemplate(
            name="low_priority_plan",
            trigger_condition=lambda gs, t, p: True,
            priority={pos: low_priority for pos in POSITIONS},
            sub_goals=[
                SubGoal(description="Low priority goal", target_condition=lambda gs, t, p: False)
            ],
        )

        planner = Planner(templates=[template_a, template_b])

        # Active plan is from template_a (high priority)
        active_plan = Plan(
            name="high_priority_plan",
            sub_goals=[SubGoal(description="High priority goal", target_condition=lambda gs, t, p: False)],
            current_index=0,
            completed=False,
        )

        result = planner.evaluate(game_state, team, position, active_plan)

        assert result is not None, "Planner should return a plan."
        assert result.name == "high_priority_plan", (
            f"Expected active plan to remain 'high_priority_plan', got '{result.name}'"
        )
        assert result is active_plan, (
            "Expected the same plan instance to be returned (no replacement)."
        )


# Feature: full-agentic-upgrade, Property 9: Plan sub-goal count limit
class TestPlanSubGoalCountLimit:
    """Property 9: Plan sub-goal count limit.

    For any Plan instantiated from any template in the library, the number of
    sub-goals SHALL be at most 5.

    **Validates: Requirements 3.8**
    """

    @given(
        game_state=game_state_strategy(),
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_all_default_templates_have_at_most_5_subgoals(
        self, game_state, team, position
    ):
        """Every template in the default library produces plans with at most 5 sub-goals."""
        templates = get_default_templates()

        for template in templates:
            assert len(template.sub_goals) <= 5, (
                f"Template '{template.name}' has {len(template.sub_goals)} sub-goals, "
                f"exceeding the maximum of 5."
            )

    @given(
        game_state=game_state_strategy(),
        team=team_strategy(),
        position=position_strategy(),
    )
    @settings(max_examples=100)
    def test_instantiated_plans_have_at_most_5_subgoals(
        self, game_state, team, position
    ):
        """Plans instantiated by the Planner from any matching template have at most 5 sub-goals."""
        planner = Planner()

        result = planner.evaluate(game_state, team, position, None)

        if result is not None:
            assert len(result.sub_goals) <= 5, (
                f"Instantiated plan '{result.name}' has {len(result.sub_goals)} sub-goals, "
                f"exceeding the maximum of 5."
            )


# Feature: full-agentic-upgrade, Property 11: Template selection by highest priority
class TestTemplateSelectionByHighestPriority:
    """Property 11: Template selection by highest priority.

    For any game state and player position where one or more plan templates have
    their trigger condition satisfied, the Planner SHALL select the template with
    the highest priority value for that position.

    **Validates: Requirements 4.1, 4.2, 4.5**
    """

    @given(
        position=position_strategy(),
        team=team_strategy(),
        game_state=game_state_strategy(),
        priorities=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_highest_priority_template_selected(
        self, position, team, game_state, priorities
    ):
        """When multiple templates match, the one with highest priority is selected."""
        # Ensure priorities are unique to have a clear winner
        assume(len(set(priorities)) == len(priorities))

        templates = []
        for i, priority in enumerate(priorities):
            templates.append(
                PlanTemplate(
                    name=f"template_{i}",
                    trigger_condition=lambda gs, t, p: True,  # All trigger
                    priority={pos: priority for pos in POSITIONS},
                    sub_goals=[
                        SubGoal(
                            description=f"Goal for template {i}",
                            target_condition=lambda gs, t, p: False,
                        )
                    ],
                )
            )

        planner = Planner(templates=templates)
        result = planner.evaluate(game_state, team, position, None)

        assert result is not None, "Planner should select a plan when templates match."

        # Find the expected winner (highest priority)
        max_priority = max(priorities)
        expected_index = priorities.index(max_priority)
        expected_name = f"template_{expected_index}"

        assert result.name == expected_name, (
            f"Expected template '{expected_name}' (priority {max_priority}) to be selected, "
            f"but got '{result.name}'. Priorities: {priorities}"
        )

    @given(
        position=position_strategy(),
        team=team_strategy(),
        game_state=game_state_strategy(),
    )
    @settings(max_examples=100)
    def test_no_template_selected_when_none_match(
        self, position, team, game_state
    ):
        """When no templates match, None is returned."""
        templates = [
            PlanTemplate(
                name="never_triggers",
                trigger_condition=lambda gs, t, p: False,
                priority={pos: 10 for pos in POSITIONS},
                sub_goals=[
                    SubGoal(description="Goal", target_condition=lambda gs, t, p: False)
                ],
            )
        ]

        planner = Planner(templates=templates)
        result = planner.evaluate(game_state, team, position, None)

        assert result is None, (
            f"Expected None when no templates match, got plan '{result.name}'"
        )

    @given(
        team=team_strategy(),
        game_state=game_state_strategy(),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_position_specific_priority_determines_selection(
        self, team, game_state, data
    ):
        """Template selection respects position-specific priority values."""
        position = data.draw(position_strategy())

        # Template A has high priority for the drawn position
        # Template B has low priority for the drawn position
        priority_a = {pos: 1 for pos in POSITIONS}
        priority_a[position] = 10

        priority_b = {pos: 20 for pos in POSITIONS}
        priority_b[position] = 5

        template_a = PlanTemplate(
            name="template_a",
            trigger_condition=lambda gs, t, p: True,
            priority=priority_a,
            sub_goals=[
                SubGoal(description="Goal A", target_condition=lambda gs, t, p: False)
            ],
        )

        template_b = PlanTemplate(
            name="template_b",
            trigger_condition=lambda gs, t, p: True,
            priority=priority_b,
            sub_goals=[
                SubGoal(description="Goal B", target_condition=lambda gs, t, p: False)
            ],
        )

        planner = Planner(templates=[template_a, template_b])
        result = planner.evaluate(game_state, team, position, None)

        assert result is not None, "Planner should select a plan."
        assert result.name == "template_a", (
            f"Expected 'template_a' (priority {priority_a[position]} for {position}) "
            f"to be selected over 'template_b' (priority {priority_b[position]}), "
            f"but got '{result.name}'"
        )
