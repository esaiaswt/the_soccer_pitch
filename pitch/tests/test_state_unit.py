"""Unit tests for the state module."""

import math

from pitch.state import (
    Ball,
    GameState,
    MatchState,
    Player,
    StateManager,
    DEFAULT_POSITIONS,
    _get_default_position,
)


class TestMatchState:
    """Tests for MatchState enum."""

    def test_waiting_value(self):
        assert MatchState.WAITING.value == "Waiting"

    def test_playing_value(self):
        assert MatchState.PLAYING.value == "Playing"


class TestBall:
    """Tests for Ball dataclass."""

    def test_default_values(self):
        ball = Ball()
        assert ball.x == 600.0
        assert ball.y == 425.0
        assert ball.vx == 0.0
        assert ball.vy == 0.0

    def test_custom_values(self):
        ball = Ball(x=100.0, y=200.0, vx=5.0, vy=-3.0)
        assert ball.x == 100.0
        assert ball.y == 200.0
        assert ball.vx == 5.0
        assert ball.vy == -3.0


class TestPlayer:
    """Tests for Player dataclass."""

    def test_creation(self):
        player = Player(name="Red_Striker", team="Red", x=300.0, y=400.0)
        assert player.name == "Red_Striker"
        assert player.team == "Red"
        assert player.x == 300.0
        assert player.y == 400.0


class TestGameState:
    """Tests for GameState dataclass."""

    def test_defaults(self):
        state = GameState()
        assert state.match_state == MatchState.WAITING
        assert state.time_left == 90.0
        assert state.score == {"Red": 0, "Blue": 0}
        assert state.ball.x == 600.0
        assert state.ball.y == 425.0
        assert state.players == {}
        assert state.goal_scored_flag is False


class TestDefaultPositions:
    """Tests for default starting positions."""

    def test_red_team_x_range(self):
        for pos in DEFAULT_POSITIONS["Red"].values():
            assert 100.0 <= pos["x"] <= 550.0

    def test_blue_team_x_range(self):
        for pos in DEFAULT_POSITIONS["Blue"].values():
            assert 650.0 <= pos["x"] <= 1100.0

    def test_positions_within_pitch(self):
        for team in DEFAULT_POSITIONS.values():
            for pos in team.values():
                assert 0.0 <= pos["x"] <= 1200.0
                assert 0.0 <= pos["y"] <= 800.0


class TestStateManager:
    """Tests for StateManager."""

    def test_initial_snapshot(self):
        sm = StateManager()
        snap = sm.read_snapshot()
        assert snap["match_state"] == "Waiting"
        assert snap["time_left"] == 90.0
        assert snap["score"] == {"Red": 0, "Blue": 0}
        assert snap["ball"] == {"x": 600.0, "y": 425.0}
        assert snap["players"] == {}

    def test_acquire_release(self):
        sm = StateManager()
        assert sm.acquire(timeout=1.0) is True
        sm.release()

    def test_acquire_timeout(self):
        sm = StateManager()
        sm.acquire()
        # Second acquire should timeout
        assert sm.acquire(timeout=0.1) is False
        sm.release()

    def test_apply_action_spawns_player(self):
        sm = StateManager()
        sm.acquire()
        result = sm.apply_action("Red", "Striker", {"dx": 0.0, "dy": 0.0}, False)
        assert result["status"] == "ok"
        assert result["player"] == "Red_Striker"
        assert "Red_Striker" in sm.state.players
        sm.release()

    def test_apply_action_clamps_vector(self):
        sm = StateManager()
        sm.acquire()
        sm.apply_action("Red", "Striker", {"dx": 5.0, "dy": -5.0}, False)
        player = sm.state.players["Red_Striker"]
        # Should have moved by MAX_SPEED (20) in each direction from default
        default_pos = _get_default_position("Red", "Striker")
        assert player.x == default_pos["x"] + 20.0
        assert player.y == default_pos["y"] - 20.0
        sm.release()

    def test_apply_action_movement(self):
        sm = StateManager()
        sm.acquire()
        sm.apply_action("Blue", "Midfielder", {"dx": 0.5, "dy": -0.3}, False)
        player = sm.state.players["Blue_Midfielder"]
        default_pos = _get_default_position("Blue", "Midfielder")
        assert player.x == default_pos["x"] + 0.5 * 20
        assert player.y == default_pos["y"] + (-0.3) * 20
        sm.release()

    def test_apply_action_kick_within_range(self):
        sm = StateManager()
        sm.acquire()
        # Place player near ball
        sm.apply_action("Red", "Striker", {"dx": 0.0, "dy": 0.0}, False)
        player = sm.state.players["Red_Striker"]
        # Move ball close to player
        sm.state.ball.x = player.x + 10.0
        sm.state.ball.y = player.y
        sm.apply_action("Red", "Striker", {"dx": 0.0, "dy": 0.0}, True)
        # Ball should have received kick impulse
        assert sm.state.ball.vx == 20.0  # KICK_IMPULSE in x direction
        assert sm.state.ball.vy == 0.0
        sm.release()

    def test_apply_action_kick_out_of_range(self):
        sm = StateManager()
        sm.acquire()
        sm.apply_action("Red", "Striker", {"dx": 0.0, "dy": 0.0}, False)
        player = sm.state.players["Red_Striker"]
        # Place ball far from player (> 30px)
        sm.state.ball.x = player.x + 100.0
        sm.state.ball.y = player.y
        sm.state.ball.vx = 0.0
        sm.state.ball.vy = 0.0
        sm.apply_action("Red", "Striker", {"dx": 0.0, "dy": 0.0}, True)
        # Ball velocity should be unchanged
        assert sm.state.ball.vx == 0.0
        assert sm.state.ball.vy == 0.0
        sm.release()

    def test_reset_after_goal(self):
        sm = StateManager()
        sm.acquire()
        # Set up some state
        sm.apply_action("Red", "Striker", {"dx": 1.0, "dy": 1.0}, False)
        sm.state.ball.x = 100.0
        sm.state.ball.y = 200.0
        sm.state.ball.vx = 15.0
        sm.state.ball.vy = -10.0
        sm.reset_after_goal()
        # Ball should be at center with zero velocity
        assert sm.state.ball.x == 600.0
        assert sm.state.ball.y == 425.0
        assert sm.state.ball.vx == 0.0
        assert sm.state.ball.vy == 0.0
        # Player should be at default position
        player = sm.state.players["Red_Striker"]
        default_pos = _get_default_position("Red", "Striker")
        assert player.x == default_pos["x"]
        assert player.y == default_pos["y"]
        sm.release()

    def test_reset_match_resets_score(self):
        sm = StateManager()
        sm.acquire()
        sm.state.match_state = MatchState.PLAYING
        sm.state.score = {"Red": 3, "Blue": 2}
        sm.state.time_left = 10.0
        sm.reset_match()
        assert sm.state.match_state == MatchState.WAITING
        assert sm.state.score == {"Red": 0, "Blue": 0}
        assert sm.state.time_left == 90.0
        assert sm.state.ball.x == 600.0
        assert sm.state.ball.y == 425.0
        assert sm.state.ball.vx == 0.0
        assert sm.state.ball.vy == 0.0
        assert sm.state.goal_scored_flag is False
        sm.release()

    def test_player_name_convention(self):
        sm = StateManager()
        sm.acquire()
        sm.apply_action("Red", "Goalkeeper", {"dx": 0.0, "dy": 0.0}, False)
        sm.apply_action("Blue", "Striker", {"dx": 0.0, "dy": 0.0}, False)
        assert "Red_Goalkeeper" in sm.state.players
        assert "Blue_Striker" in sm.state.players
        assert sm.state.players["Red_Goalkeeper"].team == "Red"
        assert sm.state.players["Blue_Striker"].team == "Blue"
        sm.release()
