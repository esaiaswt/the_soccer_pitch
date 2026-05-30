"""Unit tests for the spatial utility module."""

import math

from spatial import (
    analyze_game_state,
    compute_direction,
    compute_distance,
    format_spatial_summary,
    POSSESSION_RANGE,
)


class TestComputeDistance:
    def test_same_point(self):
        assert compute_distance(100, 200, 100, 200) == 0.0

    def test_horizontal(self):
        assert compute_distance(0, 0, 100, 0) == 100.0

    def test_vertical(self):
        assert compute_distance(0, 0, 0, 50) == 50.0

    def test_diagonal(self):
        dist = compute_distance(0, 0, 3, 4)
        assert abs(dist - 5.0) < 0.001


class TestComputeDirection:
    def test_same_point_returns_zero(self):
        d = compute_direction(100, 100, 100, 100)
        assert d["dx"] == 0.0
        assert d["dy"] == 0.0

    def test_right(self):
        d = compute_direction(0, 0, 100, 0)
        assert d["dx"] == 1.0
        assert d["dy"] == 0.0

    def test_left(self):
        d = compute_direction(100, 0, 0, 0)
        assert d["dx"] == -1.0
        assert d["dy"] == 0.0

    def test_normalized(self):
        d = compute_direction(0, 0, 3, 4)
        magnitude = math.sqrt(d["dx"] ** 2 + d["dy"] ** 2)
        assert abs(magnitude - 1.0) < 0.01


class TestAnalyzeGameState:
    def _make_state(self, player_x=100, player_y=200, ball_x=300, ball_y=200):
        return {
            "match_state": "Playing",
            "time_left": 60.0,
            "score": {"Red": 0, "Blue": 0},
            "ball": {"x": ball_x, "y": ball_y},
            "players": {
                "Red_Striker": {"x": player_x, "y": player_y},
                "Blue_Goalkeeper": {"x": 1100, "y": 400},
            },
        }

    def test_ball_distance(self):
        state = self._make_state(player_x=100, player_y=200, ball_x=100, ball_y=230)
        analysis = analyze_game_state(state, "Red", "Striker")
        assert analysis["ball_distance"] == 30.0

    def test_in_kick_range_true(self):
        state = self._make_state(player_x=100, player_y=200, ball_x=100, ball_y=220)
        analysis = analyze_game_state(state, "Red", "Striker")
        assert analysis["in_kick_range"] is True

    def test_in_kick_range_false(self):
        state = self._make_state(player_x=100, player_y=200, ball_x=300, ball_y=200)
        analysis = analyze_game_state(state, "Red", "Striker")
        assert analysis["in_kick_range"] is False

    def test_ball_direction_points_toward_ball(self):
        state = self._make_state(player_x=100, player_y=200, ball_x=200, ball_y=200)
        analysis = analyze_game_state(state, "Red", "Striker")
        assert analysis["ball_direction"]["dx"] == 1.0
        assert analysis["ball_direction"]["dy"] == 0.0

    def test_red_attacks_right_goal(self):
        state = self._make_state()
        analysis = analyze_game_state(state, "Red", "Striker")
        assert analysis["goal_to_attack"] == {"x": 1200.0, "y": 425.0}

    def test_blue_attacks_left_goal(self):
        state = self._make_state()
        analysis = analyze_game_state(state, "Blue", "Goalkeeper")
        assert analysis["goal_to_attack"] == {"x": 0.0, "y": 425.0}

    def test_nearest_opponent(self):
        state = self._make_state(player_x=100, player_y=200)
        analysis = analyze_game_state(state, "Red", "Striker")
        # Blue_Goalkeeper is at (1100, 400), distance from (100, 200)
        expected = compute_distance(100, 200, 1100, 400)
        assert analysis["nearest_opponent_distance"] == round(expected, 1)

    def test_no_teammate_when_alone(self):
        state = self._make_state()
        analysis = analyze_game_state(state, "Red", "Striker")
        # Only one Red player, so no teammate
        assert analysis["nearest_teammate_distance"] is None


class TestFormatSpatialSummary:
    def test_contains_key_info(self):
        analysis = {
            "player_position": {"x": 100.0, "y": 200.0},
            "ball_distance": 150.0,
            "ball_direction": {"dx": 0.8, "dy": 0.6},
            "in_kick_range": False,
            "is_behind_ball": True,
            "goal_to_attack": {"x": 1200.0, "y": 425.0},
            "goal_distance": 1100.0,
            "goal_direction": {"dx": 1.0, "dy": 0.0},
            "shoot_direction": {"dx": 1.0, "dy": 0.0},
            "own_goal_distance": 200.0,
            "nearest_opponent_distance": 500.0,
            "nearest_teammate_distance": 300.0,
        }
        text = format_spatial_summary(analysis)
        assert "SPATIAL ANALYSIS" in text
        assert "150.0px" in text
        assert "dx=0.8" in text
        assert "NO" in text  # not in kick range

    def test_kick_range_recommendation(self):
        analysis = {
            "player_position": {"x": 100.0, "y": 200.0},
            "ball_distance": 20.0,
            "ball_direction": {"dx": 0.0, "dy": 1.0},
            "in_kick_range": True,
            "is_behind_ball": True,
            "goal_to_attack": {"x": 1200.0, "y": 425.0},
            "goal_distance": 1100.0,
            "goal_direction": {"dx": 1.0, "dy": 0.0},
            "shoot_direction": {"dx": 1.0, "dy": 0.0},
            "own_goal_distance": 200.0,
            "nearest_opponent_distance": 500.0,
            "nearest_teammate_distance": None,
        }
        text = format_spatial_summary(analysis)
        assert "YES" in text
        assert "RECOMMENDATION" in text
        assert "Kick now" in text
