"""Property-based tests for the API module."""

import asyncio

from hypothesis import given, settings
import hypothesis.strategies as st
from httpx import AsyncClient, ASGITransport

from pitch.api import app
from pitch.state import StateManager, MatchState, Ball, Player, GameState
import pitch.api as api


# Feature: the-pitch, Property 12: API response schema completeness
class TestAPIResponseSchemaCompleteness:
    """Property 12: API response schema completeness.

    For any valid GameState (any combination of match_state, time_left, score,
    ball position, and player positions), serializing it to the GET /api/state
    response shall produce a JSON object containing all required fields:
    `match_state` (string), `time_left` (float), `score` (object with Red and
    Blue integers), `ball` (object with x and y floats), and `players` (object
    mapping names to position objects).

    **Validates: Requirements 7.2**
    """

    @given(
        match_state=st.sampled_from(list(MatchState)),
        time_left=st.floats(min_value=0.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        red_score=st.integers(min_value=0, max_value=100),
        blue_score=st.integers(min_value=0, max_value=100),
        ball_x=st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        ball_y=st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
        players=st.lists(
            st.tuples(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=10,
                ),
                st.sampled_from(["Red", "Blue"]),
                st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_api_response_contains_all_required_fields(
        self,
        match_state: MatchState,
        time_left: float,
        red_score: int,
        blue_score: int,
        ball_x: float,
        ball_y: float,
        players: list,
    ):
        """GET /api/state response contains all required fields with correct types."""

        async def _run():
            # Set up a StateManager with the generated state
            sm = StateManager()
            sm._state.match_state = match_state
            sm._state.time_left = time_left
            sm._state.score = {"Red": red_score, "Blue": blue_score}
            sm._state.ball = Ball(x=ball_x, y=ball_y, vx=0.0, vy=0.0)

            # Add generated players
            for name, team, px, py in players:
                player_name = f"{team}_{name}"
                sm._state.players[player_name] = Player(
                    name=player_name, team=team, x=px, y=py
                )

            # Set the state_manager on the api module
            api.state_manager = sm

            # Call GET /api/state
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/state")

            assert response.status_code == 200
            data = response.json()

            # Assert all required top-level fields are present
            assert "match_state" in data
            assert "time_left" in data
            assert "score" in data
            assert "ball" in data
            assert "players" in data

            # Assert match_state is a string
            assert isinstance(data["match_state"], str)
            assert data["match_state"] in ("Waiting", "Playing")

            # Assert time_left is a float (or int, which is valid JSON number)
            assert isinstance(data["time_left"], (int, float))

            # Assert score is an object with Red and Blue integers
            assert isinstance(data["score"], dict)
            assert "Red" in data["score"]
            assert "Blue" in data["score"]
            assert isinstance(data["score"]["Red"], int)
            assert isinstance(data["score"]["Blue"], int)

            # Assert ball is an object with x and y floats
            assert isinstance(data["ball"], dict)
            assert "x" in data["ball"]
            assert "y" in data["ball"]
            assert isinstance(data["ball"]["x"], (int, float))
            assert isinstance(data["ball"]["y"], (int, float))

            # Assert players is an object mapping names to position objects
            assert isinstance(data["players"], dict)
            for pname, position in data["players"].items():
                assert isinstance(pname, str)
                assert isinstance(position, dict)
                assert "x" in position
                assert "y" in position
                assert isinstance(position["x"], (int, float))
                assert isinstance(position["y"], (int, float))

        asyncio.run(_run())


# Feature: the-pitch, Property 7: Invalid team rejection
class TestInvalidTeamRejection:
    """Property 7: Invalid team rejection.

    For any string value that is not exactly "Red" or "Blue" used as the
    team field in a POST /api/action request, the server shall return
    HTTP 400 regardless of other field values.

    **Validates: Requirements 8.6**
    """

    @given(team=st.text().filter(lambda t: t not in ("Red", "Blue")))
    @settings(max_examples=100)
    def test_invalid_team_returns_400(self, team: str):
        """Any team value that is not 'Red' or 'Blue' returns HTTP 400."""

        async def _run():
            # Set up StateManager in PLAYING state so we don't get 403 first
            sm = StateManager()
            sm.state.match_state = MatchState.PLAYING
            api.state_manager = sm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/action",
                    json={
                        "team": team,
                        "position": "Striker",
                        "vector": {"dx": 0.0, "dy": 0.0},
                        "kick": False,
                    },
                )

            assert response.status_code == 400
            assert "Invalid team" in response.json()["error"]

        asyncio.run(_run())
