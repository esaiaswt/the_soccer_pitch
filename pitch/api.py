"""FastAPI application module for The Pitch.

Provides REST endpoints for AI agents to query game state and submit
player actions. All state access is thread-safe via the StateManager.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from pitch.state import MatchState, StateManager
from pitch import scoreboard

# Module-level state manager reference, set by main.py before starting uvicorn
state_manager: StateManager = None  # type: ignore


app = FastAPI(title="The Pitch", description="Agentic Football Game Server")

# Include scoreboard routes
app.include_router(scoreboard.router)


class ActionRequest(BaseModel):
    """Request model for POST /api/action."""

    team: str
    position: str
    vector: dict  # {"dx": float, "dy": float}
    kick: bool
    agent_name: Optional[str] = ""


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

    Validates team, acquires lock, applies action via StateManager.
    In Waiting state, players can spawn and move but kicks are ignored.
    """
    # Validate team
    if action.team not in ("Red", "Blue"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid team: must be 'Red' or 'Blue'"},
        )

    # Acquire lock and apply action
    try:
        if not state_manager.acquire():
            return JSONResponse(
                status_code=503,
                content={"error": "Server temporarily unable to process request"},
            )
        try:
            # In Waiting state, allow spawn but suppress movement and kicks
            is_waiting = state_manager.state.match_state == MatchState.WAITING
            if is_waiting:
                effective_kick = False
                effective_vector = {"dx": 0.0, "dy": 0.0}
            else:
                effective_kick = action.kick
                effective_vector = action.vector

            result = state_manager.apply_action(
                team=action.team,
                position=action.position,
                vector=effective_vector,
                kick=effective_kick,
                agent_name=action.agent_name or "",
            )
            return JSONResponse(status_code=200, content=result)
        finally:
            state_manager.release()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {e}"},
        )
