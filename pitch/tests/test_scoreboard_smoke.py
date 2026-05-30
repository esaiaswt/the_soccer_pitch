"""Smoke test for scoreboard endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from pitch import api, scoreboard
from pitch.state import MatchState, StateManager, GoalEvent


@pytest.fixture(autouse=True)
def setup():
    sm = StateManager()
    api.state_manager = sm
    scoreboard.state_manager = sm
    yield sm


@pytest.mark.asyncio
async def test_scoreboard_html_returns_200():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/scoreboard")
    assert response.status_code == 200
    assert "Match Scoreboard" in response.text


@pytest.mark.asyncio
async def test_scoreboard_json_empty():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/scoreboard")
    assert response.status_code == 200
    data = response.json()
    assert data["red_goals"] == []
    assert data["blue_goals"] == []
    assert data["red_top"] == []
    assert data["blue_top"] == []


@pytest.mark.asyncio
async def test_scoreboard_json_with_goals(setup):
    setup.acquire()
    setup.state.goal_log.append(GoalEvent(time=12.5, team="Red", scorer="MyBot (Striker)"))
    setup.state.goal_log.append(GoalEvent(time=45.0, team="Blue", scorer="BlueBot (Midfielder)"))
    setup.state.goal_log.append(GoalEvent(time=60.0, team="Red", scorer="MyBot (Striker)"))
    setup.state.score = {"Red": 2, "Blue": 1}
    setup.release()

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/scoreboard")
    data = response.json()
    assert len(data["red_goals"]) == 2
    assert len(data["blue_goals"]) == 1
    assert data["red_top"][0]["name"] == "MyBot (Striker)"
    assert data["red_top"][0]["goals"] == 2


@pytest.mark.asyncio
async def test_scoreboard_download_markdown(setup):
    setup.acquire()
    setup.state.goal_log.append(GoalEvent(time=10.0, team="Red", scorer="Agent1 (Striker)"))
    setup.state.score = {"Red": 1, "Blue": 0}
    setup.release()

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/scoreboard/download")
    assert response.status_code == 200
    assert "scoreboard.md" in response.headers.get("content-disposition", "")
    assert "# Match Scoreboard" in response.text
    assert "Agent1 (Striker)" in response.text
