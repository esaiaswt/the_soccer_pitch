"""FastAPI application module for The Pitch.

Provides REST endpoints for AI agents to query game state and submit
player actions. All state access is thread-safe via the StateManager.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pitch.state import MatchState, StateManager

# Module-level state manager reference, set by main.py before starting uvicorn
state_manager: StateManager = None  # type: ignore


app = FastAPI(title="The Pitch", description="Agentic Football Game Server")


class ActionRequest(BaseModel):
    """Request model for POST /api/action."""

    team: str
    position: str
    vector: dict  # {"dx": float, "dy": float}
    kick: bool


@app.get("/api/state")
async def get_state() -> JSONResponse:
    """Return a JSON snapshot of the current game state.

    Acquires the state lock via read_snapshot(). Returns 503 on lock
    timeout and 500 on unhandled exceptions.
    """
    try:
        snapshot = state_manager.read_snapshot()
        return JSONResponse(status_code=200, content=snapshot)
    except TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"error": "Server temporarily unable to process request"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {e}"},
        )


@app.post("/api/action")
async def post_action(action: ActionRequest) -> JSONResponse:
    """Process a player action (movement and/or kick).

    Validates team, checks match state, acquires lock, applies action
    via StateManager, and returns appropriate HTTP responses.
    """
    # Validate team
    if action.team not in ("Red", "Blue"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid team: must be 'Red' or 'Blue'"},
        )

    # Check match state (quick read without lock for rejection)
    if state_manager.state.match_state == MatchState.WAITING:
        return JSONResponse(
            status_code=403,
            content={"error": "Match has not started"},
        )

    # Acquire lock and apply action
    try:
        if not state_manager.acquire():
            return JSONResponse(
                status_code=503,
                content={"error": "Server temporarily unable to process request"},
            )
        try:
            # Re-check match state under lock to avoid race condition
            if state_manager.state.match_state == MatchState.WAITING:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Match has not started"},
                )

            result = state_manager.apply_action(
                team=action.team,
                position=action.position,
                vector=action.vector,
                kick=action.kick,
            )
            return JSONResponse(status_code=200, content=result)
        finally:
            state_manager.release()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {e}"},
        )
