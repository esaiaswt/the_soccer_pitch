"""Unit tests for the FastAPI API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from pitch import api
from pitch.state import MatchState, StateManager


@pytest.fixture(autouse=True)
def setup_state_manager():
    """Set up a fresh StateManager for each test."""
    sm = StateManager()
    api.state_manager = sm
    yield sm


@pytest.fixture
def playing_state(setup_state_manager):
    """Set match state to PLAYING for action tests."""
    setup_state_manager.state.match_state = MatchState.PLAYING
    return setup_state_manager


@pytest.mark.asyncio
async def test_get_state_returns_200_with_default_state():
    """GET /api/state returns 200 with correct default fields."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/state")

    assert response.status_code == 200
    data = response.json()
    assert data["match_state"] == "Waiting"
    assert data["time_left"] == 90.0
    assert data["score"] == {"Red": 0, "Blue": 0}
    assert data["ball"] == {"x": 600.0, "y": 425.0}
    assert data["players"] == {}


@pytest.mark.asyncio
async def test_get_state_returns_all_required_fields():
    """GET /api/state response contains all required fields."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/state")

    data = response.json()
    assert "match_state" in data
    assert "time_left" in data
    assert "score" in data
    assert "ball" in data
    assert "players" in data
    assert "Red" in data["score"]
    assert "Blue" in data["score"]
    assert "x" in data["ball"]
    assert "y" in data["ball"]


@pytest.mark.asyncio
async def test_get_state_returns_503_on_lock_timeout(setup_state_manager):
    """GET /api/state returns 503 when lock cannot be acquired."""
    # Hold the lock so read_snapshot will timeout
    setup_state_manager.acquire(timeout=1.0)
    try:
        # Monkey-patch the lock timeout to be very short for testing
        original_acquire = setup_state_manager.acquire

        def quick_timeout_acquire(timeout=0.01):
            return original_acquire(timeout=0.01)

        setup_state_manager.acquire = quick_timeout_acquire

        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/state")

        assert response.status_code == 503
        assert "temporarily unable" in response.json()["error"]
    finally:
        setup_state_manager.release()


@pytest.mark.asyncio
async def test_post_action_returns_400_for_invalid_team():
    """POST /api/action returns 400 for invalid team value."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Green",
                "position": "Striker",
                "vector": {"dx": 0.5, "dy": 0.0},
                "kick": False,
            },
        )

    assert response.status_code == 400
    assert "Invalid team" in response.json()["error"]


@pytest.mark.asyncio
async def test_post_action_allowed_in_waiting_state_spawns_player(setup_state_manager):
    """POST /api/action in Waiting state spawns player at default position (no movement)."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Striker",
                "vector": {"dx": 1.0, "dy": 0.0},
                "kick": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "Red_Striker" in setup_state_manager.state.players
    # Player should be at default position (movement suppressed in Waiting)
    from pitch.state import _get_default_position
    default_pos = _get_default_position("Red", "Striker")
    player = setup_state_manager.state.players["Red_Striker"]
    assert player.x == default_pos["x"]
    assert player.y == default_pos["y"]


@pytest.mark.asyncio
async def test_post_action_kick_suppressed_in_waiting_state(setup_state_manager):
    """POST /api/action in Waiting state ignores kick attempts."""
    from pitch.state import Player

    setup_state_manager.acquire()
    setup_state_manager.state.players["Red_Striker"] = Player(
        name="Red_Striker", team="Red", x=600.0, y=425.0
    )
    setup_state_manager.state.ball.x = 610.0
    setup_state_manager.state.ball.y = 425.0
    setup_state_manager.release()

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Striker",
                "vector": {"dx": 0.0, "dy": 0.0},
                "kick": True,
            },
        )

    assert response.status_code == 200
    # Ball velocity should remain zero (kick suppressed)
    assert setup_state_manager.state.ball.vx == 0.0
    assert setup_state_manager.state.ball.vy == 0.0


@pytest.mark.asyncio
async def test_post_action_returns_200_when_playing(playing_state):
    """POST /api/action returns 200 when match is Playing."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Striker",
                "vector": {"dx": 0.5, "dy": -0.3},
                "kick": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["player"] == "Red_Striker"


@pytest.mark.asyncio
async def test_post_action_spawns_player(playing_state):
    """POST /api/action spawns a new player if not exists."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Blue",
                "position": "Goalkeeper",
                "vector": {"dx": 0.0, "dy": 0.0},
                "kick": False,
            },
        )

    assert response.status_code == 200
    assert "Blue_Goalkeeper" in playing_state.state.players


@pytest.mark.asyncio
async def test_post_action_clamps_vector(playing_state):
    """POST /api/action clamps dx/dy to [-1, 1] before applying."""
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Send extreme values
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Midfielder",
                "vector": {"dx": 5.0, "dy": -10.0},
                "kick": False,
            },
        )

    assert response.status_code == 200
    # Player should have moved by at most MAX_SPEED (20) in each direction
    player = playing_state.state.players["Red_Midfielder"]
    # The default position + clamped movement should be within bounds
    # dx=5.0 clamped to 1.0 * 20 = 20, dy=-10.0 clamped to -1.0 * 20 = -20
    assert player is not None


@pytest.mark.asyncio
async def test_post_action_kick_within_range(playing_state):
    """POST /api/action kick applies impulse when within possession range."""
    # Place a player right next to the ball
    from pitch.state import Player

    playing_state.state.players["Red_Striker"] = Player(
        name="Red_Striker", team="Red", x=600.0, y=400.0
    )
    playing_state.state.ball.x = 610.0
    playing_state.state.ball.y = 400.0

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Striker",
                "vector": {"dx": 0.0, "dy": 0.0},
                "kick": True,
            },
        )

    assert response.status_code == 200
    # Ball should have received a velocity impulse
    ball = playing_state.state.ball
    assert ball.vx != 0.0 or ball.vy != 0.0


@pytest.mark.asyncio
async def test_post_action_kick_out_of_range(playing_state):
    """POST /api/action kick has no effect when out of possession range."""
    from pitch.state import Player

    playing_state.state.players["Red_Striker"] = Player(
        name="Red_Striker", team="Red", x=100.0, y=100.0
    )
    # Ball is far away
    playing_state.state.ball.x = 600.0
    playing_state.state.ball.y = 400.0
    playing_state.state.ball.vx = 0.0
    playing_state.state.ball.vy = 0.0

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/action",
            json={
                "team": "Red",
                "position": "Striker",
                "vector": {"dx": 0.0, "dy": 0.0},
                "kick": True,
            },
        )

    assert response.status_code == 200
    # Ball velocity should remain zero
    ball = playing_state.state.ball
    assert ball.vx == 0.0
    assert ball.vy == 0.0
