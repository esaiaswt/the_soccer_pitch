"""Integration tests for the full API lifecycle.

Tests the complete request/response cycle including:
- GET /api/state schema validation in Waiting state
- POST /api/action rejection in Waiting state
- Full match lifecycle (start → action → goal → reset → timer expiry)
- Concurrent action submissions (no lost writes)
- Lock timeout behavior (503 response)

Requirements: 6.2, 7.1, 7.2, 7.3, 7.4, 8.5, 8.7
"""

import asyncio
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from pitch import api
from pitch.state import MatchState, Player, StateManager


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


# ---------------------------------------------------------------------------
# Test 1: GET /api/state returns correct JSON schema in Waiting state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_state_waiting_returns_correct_schema(setup_state_manager):
    """GET /api/state in Waiting state returns all fields with correct defaults.

    Validates: Requirements 7.1, 7.2, 7.3
    """
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/state")

    assert response.status_code == 200
    data = response.json()

    # Verify all required top-level fields are present
    assert "match_state" in data
    assert "time_left" in data
    assert "score" in data
    assert "ball" in data
    assert "players" in data

    # Verify correct types and default values in Waiting state
    assert data["match_state"] == "Waiting"
    assert isinstance(data["match_state"], str)

    assert data["time_left"] == 90.0
    assert isinstance(data["time_left"], float)

    assert data["score"] == {"Red": 0, "Blue": 0}
    assert isinstance(data["score"], dict)
    assert isinstance(data["score"]["Red"], int)
    assert isinstance(data["score"]["Blue"], int)

    assert data["ball"] == {"x": 600.0, "y": 400.0}
    assert isinstance(data["ball"], dict)
    assert isinstance(data["ball"]["x"], float)
    assert isinstance(data["ball"]["y"], float)

    assert data["players"] == {}
    assert isinstance(data["players"], dict)


# ---------------------------------------------------------------------------
# Test 2: POST /api/action returns 403 in Waiting state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_action_rejected_in_waiting_state(setup_state_manager):
    """POST /api/action returns 403 when match_state is Waiting.

    Validates: Requirements 8.5
    """
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

    assert response.status_code == 403
    data = response.json()
    assert "error" in data
    assert "Match has not started" in data["error"]

    # Verify no players were spawned (state unchanged)
    assert len(setup_state_manager.state.players) == 0


# ---------------------------------------------------------------------------
# Test 3: Full match lifecycle (start → action → goal → reset → timer expiry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_match_lifecycle(setup_state_manager):
    """Test complete match lifecycle: start → action → goal → reset → timer expiry.

    Validates: Requirements 6.2, 7.1, 7.2, 7.4, 8.7
    """
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: Verify initial Waiting state
        response = await client.get("/api/state")
        assert response.status_code == 200
        assert response.json()["match_state"] == "Waiting"

        # Step 2: Transition to Playing
        setup_state_manager.state.match_state = MatchState.PLAYING

        # Step 3: Verify state is now Playing
        response = await client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["match_state"] == "Playing"

        # Step 4: Submit action to move a player
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
        assert response.json()["player"] == "Red_Striker"

        # Step 5: Verify player appears in state
        response = await client.get("/api/state")
        data = response.json()
        assert "Red_Striker" in data["players"]

        # Step 6: Simulate a goal by moving ball into left goal zone
        # (Blue scores when ball enters left goal zone: x 0-30, y 300-500)
        setup_state_manager.state.ball.x = 10.0
        setup_state_manager.state.ball.y = 400.0

        # Use physics check_goal to detect and score
        from pitch.physics import PhysicsEngine

        engine = PhysicsEngine(setup_state_manager)
        # Acquire lock manually for the goal check
        setup_state_manager.acquire()
        try:
            engine.check_goal(setup_state_manager.state)
        finally:
            setup_state_manager.release()

        # Step 7: Verify score incremented
        response = await client.get("/api/state")
        data = response.json()
        assert data["score"]["Blue"] == 1
        assert data["score"]["Red"] == 0

        # Step 8: Reset after goal
        setup_state_manager.acquire()
        try:
            setup_state_manager.reset_after_goal()
        finally:
            setup_state_manager.release()

        # Step 9: Verify ball reset to center
        response = await client.get("/api/state")
        data = response.json()
        assert data["ball"]["x"] == 600.0
        assert data["ball"]["y"] == 400.0

        # Step 10: Simulate timer expiry
        setup_state_manager.acquire()
        try:
            setup_state_manager.state.time_left = 0.0
            setup_state_manager.reset_match()
        finally:
            setup_state_manager.release()

        # Step 11: Verify match transitioned back to Waiting with score preserved
        response = await client.get("/api/state")
        data = response.json()
        assert data["match_state"] == "Waiting"
        assert data["time_left"] == 90.0
        # Score is preserved across match reset
        assert data["score"]["Blue"] == 1


# ---------------------------------------------------------------------------
# Test 4: Concurrent action submissions (no lost writes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_actions_no_lost_writes(playing_state):
    """Concurrent action submissions all get applied with no lost writes.

    Validates: Requirements 6.2
    """
    transport = ASGITransport(app=api.app)

    num_players = 10

    async def submit_action(client: AsyncClient, index: int):
        """Submit a unique player action."""
        team = "Red" if index % 2 == 0 else "Blue"
        position = f"Player{index}"
        response = await client.post(
            "/api/action",
            json={
                "team": team,
                "position": position,
                "vector": {"dx": 0.5, "dy": 0.5},
                "kick": False,
            },
        )
        return response

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Submit multiple actions concurrently
        tasks = [submit_action(client, i) for i in range(num_players)]
        responses = await asyncio.gather(*tasks)

    # Verify all responses were successful
    for resp in responses:
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    # Verify all players were created (no lost writes)
    assert len(playing_state.state.players) == num_players

    # Verify each player exists with the expected name
    for i in range(num_players):
        team = "Red" if i % 2 == 0 else "Blue"
        player_name = f"{team}_Player{i}"
        assert player_name in playing_state.state.players


# ---------------------------------------------------------------------------
# Test 5: Lock timeout behavior (503 response)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_timeout_returns_503(setup_state_manager):
    """API returns 503 when the state lock cannot be acquired within timeout.

    Validates: Requirements 7.4
    """
    # Hold the lock externally using a background thread
    lock_held = threading.Event()
    release_signal = threading.Event()

    def hold_lock():
        """Hold the state lock until signaled to release."""
        setup_state_manager.acquire(timeout=5.0)
        lock_held.set()
        release_signal.wait(timeout=10.0)
        setup_state_manager.release()

    lock_thread = threading.Thread(target=hold_lock, daemon=True)
    lock_thread.start()

    # Wait for the lock to be held
    lock_held.wait(timeout=5.0)

    # Monkey-patch acquire to use a very short timeout for testing
    original_acquire = setup_state_manager.acquire

    def quick_timeout_acquire(timeout=0.01):
        return original_acquire(timeout=0.01)

    setup_state_manager.acquire = quick_timeout_acquire

    try:
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # GET /api/state should return 503
            response = await client.get("/api/state")
            assert response.status_code == 503
            assert "temporarily unable" in response.json()["error"]
    finally:
        # Release the lock and clean up
        setup_state_manager.acquire = original_acquire
        release_signal.set()
        lock_thread.join(timeout=5.0)


@pytest.mark.asyncio
async def test_lock_timeout_post_action_returns_503(setup_state_manager):
    """POST /api/action returns 503 when lock cannot be acquired.

    Validates: Requirements 7.4
    """
    # Set state to Playing so we get past the Waiting check
    setup_state_manager.state.match_state = MatchState.PLAYING

    # Hold the lock externally
    lock_held = threading.Event()
    release_signal = threading.Event()

    def hold_lock():
        """Hold the state lock until signaled to release."""
        setup_state_manager.acquire(timeout=5.0)
        lock_held.set()
        release_signal.wait(timeout=10.0)
        setup_state_manager.release()

    lock_thread = threading.Thread(target=hold_lock, daemon=True)
    lock_thread.start()

    # Wait for the lock to be held
    lock_held.wait(timeout=5.0)

    # Monkey-patch acquire to use a very short timeout
    original_acquire = setup_state_manager.acquire

    def quick_timeout_acquire(timeout=0.01):
        return original_acquire(timeout=0.01)

    setup_state_manager.acquire = quick_timeout_acquire

    try:
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
            assert response.status_code == 503
            assert "temporarily unable" in response.json()["error"]
    finally:
        setup_state_manager.acquire = original_acquire
        release_signal.set()
        lock_thread.join(timeout=5.0)
