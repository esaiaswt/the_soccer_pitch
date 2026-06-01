"""Property-based tests for the team/ SignalGenerator module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

import time

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.signal_generator import SignalGenerator
from team.planner import Plan, SubGoal
from team.signal_bus import Signal


# --- Strategies for generating valid inputs ---

team_strategy = st.sampled_from(["Red", "Blue"])

position_strategy = st.sampled_from([
    "Goalkeeper", "Defender_L", "Defender_R", "Midfielder_L",
    "Midfielder_R", "Striker",
])

# Descriptions that benefit from teammate awareness
awareness_description_strategy = st.sampled_from([
    "receive_pass from midfielder",
    "receive pass in the box",
    "pass the ball forward",
    "distribute to teammate",
    "find teammate in space",
    "receive_pass",
    "distribute ball wide",
])

# Descriptions that do NOT benefit from teammate awareness
non_awareness_description_strategy = st.sampled_from([
    "move toward own goal",
    "intercept the ball",
    "shoot at goal",
    "position behind ball",
    "gain possession",
    "defend the flank",
])

# Valid pitch coordinates
x_coord_strategy = st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False)
y_coord_strategy = st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False)


def _make_plan_with_awareness_subgoal(description: str) -> Plan:
    """Create a plan with a sub-goal that benefits from teammate awareness."""
    return Plan(
        name="test_plan",
        sub_goals=[
            SubGoal(
                description=description,
                target_condition=lambda gs, t, p: False,
            ),
        ],
        current_index=0,
        completed=False,
    )


def _make_plan_with_non_awareness_subgoal(description: str) -> Plan:
    """Create a plan with a sub-goal that does NOT benefit from teammate awareness."""
    return Plan(
        name="test_plan",
        sub_goals=[
            SubGoal(
                description=description,
                target_condition=lambda gs, t, p: False,
            ),
        ],
        current_index=0,
        completed=False,
    )


def _make_game_state_not_dead_ball(
    team: str,
    position: str,
    player_x: float,
    player_y: float,
    ball_x: float,
    ball_y: float,
    extra_players: dict | None = None,
    signals: list | None = None,
) -> dict:
    """Create a game state that is NOT a dead ball situation."""
    players = {f"{team}_{position}": {"x": player_x, "y": player_y}}
    if extra_players:
        players.update(extra_players)
    gs = {
        "dead_ball": False,
        "is_dead_ball": False,
        "ball": {"x": ball_x, "y": ball_y},
        "players": players,
    }
    if signals is not None:
        gs["signals"] = signals
    return gs


def _make_dead_ball_game_state(
    team: str,
    position: str,
    player_x: float = 600.0,
    player_y: float = 400.0,
    use_dead_ball_key: bool = True,
) -> dict:
    """Create a game state that IS a dead ball situation."""
    gs = {
        "ball": {"x": 600.0, "y": 400.0},
        "players": {f"{team}_{position}": {"x": player_x, "y": player_y}},
    }
    if use_dead_ball_key:
        gs["dead_ball"] = True
    else:
        gs["is_dead_ball"] = True
    return gs


# Feature: full-agentic-upgrade, Property 21: Signal generation from awareness-benefiting sub-goals
# **Validates: Requirements 8.1, 8.2, 8.3**


class TestSignalGenerationFromAwarenessSubGoals:
    """Property 21: Signal generation from awareness-benefiting sub-goals.

    For any active Plan with a sub-goal that benefits from teammate awareness
    (e.g., "receive_pass") and a game state that is not a dead ball situation,
    the Signal_Generator SHALL produce a non-None signal. For dead ball situations,
    it SHALL produce None.
    """

    @settings(max_examples=100)
    @given(
        description=awareness_description_strategy,
        team=team_strategy,
        position=position_strategy,
        player_x=x_coord_strategy,
        player_y=y_coord_strategy,
        ball_x=x_coord_strategy,
        ball_y=y_coord_strategy,
    )
    def test_awareness_subgoal_not_dead_ball_produces_signal(
        self, description, team, position, player_x, player_y, ball_x, ball_y
    ):
        """For any active Plan with an awareness-benefiting sub-goal and a non-dead-ball
        game state, the Signal_Generator SHALL produce a non-None signal.

        **Validates: Requirements 8.1**
        """
        generator = SignalGenerator()
        plan = _make_plan_with_awareness_subgoal(description)
        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y
        )

        signal = generator.generate(plan, game_state, team, position)

        assert signal is not None, (
            f"Expected non-None signal for awareness sub-goal '{description}' "
            f"in non-dead-ball state, but got None"
        )

    @settings(max_examples=100)
    @given(
        description=awareness_description_strategy,
        team=team_strategy,
        position=position_strategy,
        use_dead_ball_key=st.booleans(),
    )
    def test_awareness_subgoal_dead_ball_produces_none(
        self, description, team, position, use_dead_ball_key
    ):
        """For dead ball situations, the Signal_Generator SHALL produce None.

        **Validates: Requirements 8.1**
        """
        generator = SignalGenerator()
        plan = _make_plan_with_awareness_subgoal(description)
        game_state = _make_dead_ball_game_state(
            team, position, use_dead_ball_key=use_dead_ball_key
        )

        signal = generator.generate(plan, game_state, team, position)

        assert signal is None, (
            f"Expected None signal for dead ball state with awareness sub-goal "
            f"'{description}', but got signal: {signal}"
        )

    @settings(max_examples=100)
    @given(
        description=awareness_description_strategy,
        team=team_strategy,
        position=position_strategy,
        player_x=x_coord_strategy,
        player_y=y_coord_strategy,
        ball_x=x_coord_strategy,
        ball_y=y_coord_strategy,
    )
    def test_awareness_subgoal_signal_type_is_requesting_pass(
        self, description, team, position, player_x, player_y, ball_x, ball_y
    ):
        """When an awareness-benefiting sub-goal generates a signal, it SHALL have
        signal_type "requesting_pass".

        **Validates: Requirements 8.1**
        """
        generator = SignalGenerator()
        plan = _make_plan_with_awareness_subgoal(description)
        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y
        )

        signal = generator.generate(plan, game_state, team, position)

        assert signal is not None
        assert signal.signal_type == "requesting_pass", (
            f"Expected signal_type 'requesting_pass', got '{signal.signal_type}'"
        )


# Feature: full-agentic-upgrade, Property 22: Ready-to-pass signal generation
# **Validates: Requirements 8.1, 8.2, 8.3**


class TestReadyToPassSignalGeneration:
    """Property 22: Ready-to-pass signal generation.

    For any game state where the player is within kick range (ball_distance <= 30)
    and a teammate has published a "making_run" signal, the Signal_Generator SHALL
    produce a signal with signal_type "ready_to_pass".
    """

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        ball_x=st.floats(min_value=100.0, max_value=1100.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=100.0, max_value=700.0, allow_nan=False, allow_infinity=False),
        offset_x=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        offset_y=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    def test_kick_range_with_making_run_signal_produces_ready_to_pass(
        self, team, position, ball_x, ball_y, offset_x, offset_y
    ):
        """When player is within kick range and a teammate has a "making_run" signal,
        the Signal_Generator SHALL produce a signal with signal_type "ready_to_pass".

        **Validates: Requirements 8.2**
        """
        # Position player within kick range (distance <= 30)
        player_x = ball_x + offset_x
        player_y = ball_y + offset_y
        dist = ((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2) ** 0.5
        assume(dist <= 30)

        # Pick a different position for the teammate making a run
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        teammate_position = other_positions[0]

        # Create signals with a teammate making a run
        signals = [{"signal_type": "making_run", "sender_position": teammate_position}]

        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y,
            signals=signals,
        )

        generator = SignalGenerator()
        # No plan or a plan without awareness sub-goal so we don't trigger rule 2 first
        signal = generator.generate(None, game_state, team, position)

        assert signal is not None, (
            f"Expected non-None signal when in kick range (dist={dist:.1f}) "
            f"and teammate making a run"
        )
        assert signal.signal_type == "ready_to_pass", (
            f"Expected signal_type 'ready_to_pass', got '{signal.signal_type}'"
        )

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        ball_x=st.floats(min_value=100.0, max_value=1100.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=100.0, max_value=700.0, allow_nan=False, allow_infinity=False),
    )
    def test_kick_range_with_teammate_ahead_produces_ready_to_pass(
        self, team, position, ball_x, ball_y
    ):
        """When player is within kick range and a teammate is positioned ahead of ball
        by 50+ pixels, the Signal_Generator SHALL produce "ready_to_pass".

        **Validates: Requirements 8.2**
        """
        # Position player exactly at ball (distance = 0, within kick range)
        player_x = ball_x
        player_y = ball_y

        # Pick a different position for the teammate
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        teammate_position = other_positions[0]

        # Position teammate ahead of ball by 50+ pixels in attacking direction
        if team == "Red":
            teammate_x = ball_x + 60  # ahead for Red (attacking right)
        else:
            teammate_x = ball_x - 60  # ahead for Blue (attacking left)

        extra_players = {
            f"{team}_{teammate_position}": {"x": teammate_x, "y": ball_y}
        }

        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y,
            extra_players=extra_players,
        )

        generator = SignalGenerator()
        signal = generator.generate(None, game_state, team, position)

        assert signal is not None, (
            "Expected non-None signal when in kick range and teammate ahead of ball"
        )
        assert signal.signal_type == "ready_to_pass", (
            f"Expected signal_type 'ready_to_pass', got '{signal.signal_type}'"
        )

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        ball_x=st.floats(min_value=100.0, max_value=1100.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=100.0, max_value=700.0, allow_nan=False, allow_infinity=False),
        player_offset=st.floats(min_value=31.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    def test_outside_kick_range_does_not_produce_ready_to_pass(
        self, team, position, ball_x, ball_y, player_offset
    ):
        """When player is NOT within kick range (ball_distance > 30), the Signal_Generator
        SHALL NOT produce a "ready_to_pass" signal even if a teammate is making a run.

        **Validates: Requirements 8.2**
        """
        # Position player outside kick range
        player_x = ball_x + player_offset
        player_y = ball_y

        # Teammate making a run signal
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        teammate_position = other_positions[0]
        signals = [{"signal_type": "making_run", "sender_position": teammate_position}]

        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y,
            signals=signals,
        )

        generator = SignalGenerator()
        signal = generator.generate(None, game_state, team, position)

        # Should not produce "ready_to_pass" since outside kick range
        if signal is not None:
            assert signal.signal_type != "ready_to_pass", (
                f"Expected no 'ready_to_pass' signal when outside kick range "
                f"(offset={player_offset:.1f}), but got one"
            )


# Feature: full-agentic-upgrade, Property 23: Supporting signal generation
# **Validates: Requirements 8.1, 8.2, 8.3**


class TestSupportingSignalGeneration:
    """Property 23: Supporting signal generation.

    For any game state where the player is the nearest teammate to the ball carrier,
    the Signal_Generator SHALL produce a signal with signal_type "supporting" and a
    payload containing the player's current zone.
    """

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        ball_x=st.floats(min_value=100.0, max_value=1100.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=100.0, max_value=700.0, allow_nan=False, allow_infinity=False),
        support_offset=st.floats(min_value=35.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    )
    def test_nearest_to_ball_carrier_produces_supporting_signal(
        self, team, position, ball_x, ball_y, support_offset
    ):
        """When the player is the nearest teammate to the ball carrier, the
        Signal_Generator SHALL produce a signal with signal_type "supporting".

        **Validates: Requirements 8.3**
        """
        # Pick a different position for the ball carrier (closest to ball)
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        carrier_position = other_positions[0]
        third_position = other_positions[1]

        # Ball carrier is within kick range (distance <= 30) - at ball position
        carrier_x = ball_x
        carrier_y = ball_y

        # Our player is the second closest (nearest to ball carrier)
        player_x = ball_x + support_offset
        player_y = ball_y

        # Third player is further away
        third_x = ball_x + support_offset + 100
        third_y = ball_y

        extra_players = {
            f"{team}_{carrier_position}": {"x": carrier_x, "y": carrier_y},
            f"{team}_{third_position}": {"x": third_x, "y": third_y},
        }

        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y,
            extra_players=extra_players,
        )

        generator = SignalGenerator()
        signal = generator.generate(None, game_state, team, position)

        assert signal is not None, (
            f"Expected non-None signal when nearest to ball carrier "
            f"(support_offset={support_offset:.1f})"
        )
        assert signal.signal_type == "supporting", (
            f"Expected signal_type 'supporting', got '{signal.signal_type}'"
        )

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        ball_x=st.floats(min_value=100.0, max_value=1100.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=100.0, max_value=700.0, allow_nan=False, allow_infinity=False),
        support_offset=st.floats(min_value=35.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    )
    def test_supporting_signal_payload_contains_zone(
        self, team, position, ball_x, ball_y, support_offset
    ):
        """The supporting signal SHALL have a payload containing the player's current zone.

        **Validates: Requirements 8.3**
        """
        # Set up the same scenario: player is nearest to ball carrier
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        carrier_position = other_positions[0]
        third_position = other_positions[1]

        carrier_x = ball_x
        carrier_y = ball_y

        player_x = ball_x + support_offset
        player_y = ball_y

        third_x = ball_x + support_offset + 100
        third_y = ball_y

        extra_players = {
            f"{team}_{carrier_position}": {"x": carrier_x, "y": carrier_y},
            f"{team}_{third_position}": {"x": third_x, "y": third_y},
        }

        game_state = _make_game_state_not_dead_ball(
            team, position, player_x, player_y, ball_x, ball_y,
            extra_players=extra_players,
        )

        generator = SignalGenerator()
        signal = generator.generate(None, game_state, team, position)

        assert signal is not None, "Expected non-None supporting signal"
        assert signal.signal_type == "supporting"

        # Payload should be a valid zone string
        valid_zones = [
            "left_defense", "left_midfield", "left_attack",
            "center_defense", "center_midfield", "center_attack",
            "right_defense", "right_midfield", "right_attack",
        ]
        assert signal.payload in valid_zones, (
            f"Expected payload to be a valid zone, got '{signal.payload}'. "
            f"Valid zones: {valid_zones}"
        )

    @settings(max_examples=100)
    @given(
        team=team_strategy,
        position=position_strategy,
        player_x=x_coord_strategy,
        player_y=y_coord_strategy,
        ball_x=x_coord_strategy,
        ball_y=y_coord_strategy,
    )
    def test_not_nearest_to_carrier_does_not_produce_supporting(
        self, team, position, player_x, player_y, ball_x, ball_y
    ):
        """When the player is NOT the nearest teammate to the ball carrier,
        the Signal_Generator SHALL NOT produce a "supporting" signal.

        **Validates: Requirements 8.3**
        """
        # Set up scenario where another teammate is closer to ball carrier than us
        other_positions = [p for p in ["Goalkeeper", "Defender_L", "Defender_R",
                                        "Midfielder_L", "Midfielder_R", "Striker"]
                          if p != position]
        carrier_position = other_positions[0]
        closer_teammate_position = other_positions[1]

        # Ball carrier at ball position (within kick range)
        carrier_x = ball_x
        carrier_y = ball_y

        # Another teammate is closer to ball than us
        # Place closer teammate very near ball
        closer_x = ball_x + 5.0
        closer_y = ball_y + 5.0

        # Our player is far from ball
        far_player_x = ball_x + 200.0
        far_player_y = ball_y + 200.0

        extra_players = {
            f"{team}_{carrier_position}": {"x": carrier_x, "y": carrier_y},
            f"{team}_{closer_teammate_position}": {"x": closer_x, "y": closer_y},
        }

        game_state = _make_game_state_not_dead_ball(
            team, position, far_player_x, far_player_y, ball_x, ball_y,
            extra_players=extra_players,
        )

        generator = SignalGenerator()
        signal = generator.generate(None, game_state, team, position)

        # Should not produce "supporting" since we're not the nearest to carrier
        if signal is not None:
            assert signal.signal_type != "supporting", (
                f"Expected no 'supporting' signal when not nearest to ball carrier, "
                f"but got one"
            )
